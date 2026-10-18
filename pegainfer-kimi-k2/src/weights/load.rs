use bytesize::ByteSize;

use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum KimiTensorLoadSlice {
    Full,
    RowRange { start: usize, end: usize },
    ColRange { start: usize, end: usize },
}

impl KimiTensorLoadSlice {
    fn local_shape(&self, full_shape: &[usize]) -> Result<Vec<usize>> {
        match *self {
            Self::Full => Ok(full_shape.to_vec()),
            Self::RowRange { start, end } => {
                ensure!(
                    full_shape.len() == 2 && start <= end && end <= full_shape[0],
                    "Kimi row slice [{start}..{end}) is invalid for shape {:?}",
                    full_shape
                );
                Ok(vec![end - start, full_shape[1]])
            }
            Self::ColRange { start, end } => {
                ensure!(
                    full_shape.len() == 2 && start <= end && end <= full_shape[1],
                    "Kimi col slice [{start}..{end}) is invalid for shape {:?}",
                    full_shape
                );
                Ok(vec![full_shape[0], end - start])
            }
        }
    }

    fn local_bytes(&self, full_shape: &[usize], dtype: Dtype) -> Result<usize> {
        Ok(self.local_shape(full_shape)?.iter().product::<usize>() * dtype_element_bytes(dtype)?)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiTensorLoadSpec {
    pub name: String,
    pub shard: String,
    pub slice: KimiTensorLoadSlice,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiShardTensorLoadPlan {
    pub shard: String,
    pub tensors: Vec<KimiTensorLoadSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiRankSlicedLoadPlan {
    pub rank: usize,
    pub shards: Vec<KimiShardTensorLoadPlan>,
    pub tensor_count: usize,
}

pub(crate) struct KimiRankSlicedLoadOutput {
    pub weights: KimiRankGpuWeights,
    pub expert_kernel_weights: KimiRankExpertMarlinWeights,
    pub loaded_tensor_count: usize,
    pub loaded_total_bytes: usize,
}

pub(crate) fn ensure_text_only_model_index(model_path: &Path) -> Result<KimiK2WeightManifest> {
    let manifest = KimiK2WeightManifest::from_model_dir(model_path)?;
    if manifest.text_tensor_count == 0 {
        bail!("Kimi safetensors index contains no language_model tensors");
    }
    Ok(manifest)
}

pub(crate) fn load_rank_sliced_weights_to_gpu(
    ctx: &KimiRankGpuContext,
    model_path: &Path,
    load_plan: &KimiRankSlicedLoadPlan,
    names: &KimiRankWeightNames,
) -> Result<KimiRankSlicedLoadOutput> {
    ctx.set_current()?;
    ensure!(
        load_plan.rank == names.rank,
        "Kimi rank sliced load plan {} does not match typed names rank {}",
        load_plan.rank,
        names.rank
    );
    let mut weights = KimiRankGpuWeights {
        rank: load_plan.rank,
        tensors: BTreeMap::new(),
        total_bytes: 0,
    };
    let mut loaded_tensor_count = 0usize;
    let mut loaded_total_bytes = 0usize;
    let mut packed_moe_layers = BTreeSet::new();
    let mut expert_layers = Vec::with_capacity(KIMI_K2_MOE_LAYERS);
    let load_started = Instant::now();
    debug!(
        "rank {} start weight load: tensors={}, shards={}",
        load_plan.rank,
        load_plan.tensor_count,
        load_plan.shards.len()
    );
    let mut slowest_shard: Option<(String, f64)> = None;
    for shard in &load_plan.shards {
        let path = model_path.join(&shard.shard);
        let shard_started = Instant::now();
        let mmap = mmap_file(&path)?;
        let safetensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("failed to deserialize {}", path.display()))?;
        for spec in &shard.tensors {
            let view = safetensors
                .tensor(&spec.name)
                .with_context(|| format!("missing tensor {} in {}", spec.name, path.display()))?;
            let shape = spec.slice.local_shape(view.shape())?;
            let bytes = spec.slice.local_bytes(view.shape(), view.dtype())?;
            let data = if spec.slice == KimiTensorLoadSlice::Full {
                ctx.stream()
                    .clone_htod(view.data())
                    .with_context(|| format!("failed to copy Kimi tensor {} to GPU", spec.name))?
            } else {
                let sliced =
                    sliced_tensor_bytes(view.data(), view.shape(), view.dtype(), &spec.slice)
                        .with_context(|| format!("failed to slice Kimi tensor {}", spec.name))?;
                ctx.stream()
                    .clone_htod(sliced.as_slice())
                    .with_context(|| format!("failed to copy Kimi tensor {} to GPU", spec.name))?
            };
            let tensor = KimiGpuRawTensor {
                name: spec.name.clone(),
                dtype: view.dtype(),
                shape,
                bytes,
                data,
            };
            weights.total_bytes += tensor.bytes;
            loaded_total_bytes += tensor.bytes;
            loaded_tensor_count += 1;
            ensure!(
                weights.tensors.insert(spec.name.clone(), tensor).is_none(),
                "duplicate Kimi tensor {} in rank {} sliced load plan",
                spec.name,
                load_plan.rank
            );
        }
        weights.pack_loaded_expert_marlin_layers(
            ctx,
            names,
            &mut packed_moe_layers,
            &mut expert_layers,
        )?;
        let shard_secs = shard_started.elapsed().as_secs_f64();
        match &slowest_shard {
            Some((_, slowest_secs)) if *slowest_secs >= shard_secs => {}
            _ => slowest_shard = Some((shard.shard.clone(), shard_secs)),
        }
    }
    ensure!(
        loaded_tensor_count == load_plan.tensor_count,
        "Kimi rank {} sliced GPU tensor count {} does not match load plan {}",
        load_plan.rank,
        loaded_tensor_count,
        load_plan.tensor_count
    );
    ensure!(
        expert_layers.len() == KIMI_K2_MOE_LAYERS,
        "Kimi rank {} expected {KIMI_K2_MOE_LAYERS} streamed MoE Marlin weight layers, got {}",
        load_plan.rank,
        expert_layers.len()
    );
    weights.validate_non_expert_typed_view(names)?;
    let expert_kernel_total_bytes = expert_layers.iter().map(|layer| layer.total_bytes).sum();
    let expert_kernel_weights = KimiRankExpertMarlinWeights {
        rank: load_plan.rank,
        local_expert_range: names.plan.local_expert_range.clone(),
        layers: expert_layers,
        total_bytes: expert_kernel_total_bytes,
    };
    debug!("rank {} start weight copy sync", load_plan.rank);
    let sync_started = Instant::now();
    ctx.sync().with_context(|| {
        format!(
            "failed to finish Kimi rank {} sliced GPU tensor copies",
            load_plan.rank
        )
    })?;
    debug!(
        "rank {} weight copy sync cost {:.2}s",
        load_plan.rank,
        sync_started.elapsed().as_secs_f64()
    );
    let (slowest_shard, slowest_secs) = slowest_shard.unwrap_or_else(|| ("none".to_owned(), 0.0));
    debug!(
        "rank {} weight load cost {:.2}s: loaded_tensors={}, loaded_bytes={}, resident_raw_bytes={}, expert_package_bytes={}, packed_moe_layers={}, slowest_shard={} {:.2}s",
        load_plan.rank,
        load_started.elapsed().as_secs_f64(),
        loaded_tensor_count,
        ByteSize(loaded_total_bytes as u64),
        ByteSize(weights.total_bytes as u64),
        ByteSize(expert_kernel_weights.total_bytes as u64),
        packed_moe_layers.len(),
        slowest_shard,
        slowest_secs
    );
    Ok(KimiRankSlicedLoadOutput {
        weights,
        expert_kernel_weights,
        loaded_tensor_count,
        loaded_total_bytes,
    })
}

pub(super) fn sliced_tensor_bytes(
    data: &[u8],
    shape: &[usize],
    dtype: Dtype,
    slice: &KimiTensorLoadSlice,
) -> Result<Vec<u8>> {
    let element_bytes = dtype_element_bytes(dtype)?;
    match *slice {
        KimiTensorLoadSlice::Full => Ok(data.to_vec()),
        KimiTensorLoadSlice::RowRange { start, end } => {
            ensure!(
                shape.len() == 2 && start <= end && end <= shape[0],
                "invalid row slice [{start}..{end}) for shape {:?}",
                shape
            );
            let row_bytes = shape[1] * element_bytes;
            let start_byte = start * row_bytes;
            let end_byte = end * row_bytes;
            ensure!(
                end_byte <= data.len(),
                "row slice byte range [{start_byte}..{end_byte}) exceeds tensor bytes {}",
                data.len()
            );
            Ok(data[start_byte..end_byte].to_vec())
        }
        KimiTensorLoadSlice::ColRange { start, end } => {
            ensure!(
                shape.len() == 2 && start <= end && end <= shape[1],
                "invalid col slice [{start}..{end}) for shape {:?}",
                shape
            );
            let rows = shape[0];
            let cols = shape[1];
            let row_bytes = cols * element_bytes;
            let local_cols = end - start;
            let local_row_bytes = local_cols * element_bytes;
            let mut out = vec![0u8; rows * local_row_bytes];
            for row in 0..rows {
                let src = row * row_bytes + start * element_bytes;
                let dst = row * local_row_bytes;
                out[dst..dst + local_row_bytes].copy_from_slice(&data[src..src + local_row_bytes]);
            }
            Ok(out)
        }
    }
}

fn mmap_file(path: &Path) -> Result<Mmap> {
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    // SAFETY: checkpoint shards are opened read-only and the mapping is only
    // consumed while reading safetensors metadata or copying tensor bytes.
    unsafe { Mmap::map(&file) }.with_context(|| format!("failed to mmap {}", path.display()))
}

pub(super) fn dtype_element_bytes(dtype: Dtype) -> Result<usize> {
    match dtype {
        Dtype::BF16 => Ok(2),
        Dtype::F32 | Dtype::I32 => Ok(4),
        Dtype::U8 => Ok(1),
        other => bail!("Kimi loader does not support dtype {:?}", other),
    }
}
