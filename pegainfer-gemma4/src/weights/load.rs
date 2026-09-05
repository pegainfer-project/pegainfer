//! Checkpoint loading: classify the whole manifest, then upload.

use std::time::Instant;

use anyhow::Result;
use log::info;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::weight_loader::ByteWeightStager;
use pegainfer_core::weight_loader::SlotId;
use pegainfer_core::weight_loader::StagedWeightLoader;
use pegainfer_core::weight_loader::VecSlotId;
use pegainfer_core::weight_loader::WeightPrefetch;
use pegainfer_core::weight_loader::deserialize_shards;
use pegainfer_core::weight_loader::load_shard_info;
use pegainfer_core::weight_loader::mmap_shards;
use safetensors::Dtype;
use safetensors::SafeTensors;

use super::Gemma4Attention;
use super::Gemma4Layer;
use super::Gemma4Mlp;
use super::Gemma4Moe;
use super::Gemma4Weights;
use super::StackedProjection;
use crate::config::Gemma4Config;
use crate::manifest::schema::ExpertTensors;
use crate::manifest::schema::Manifest;
use crate::manifest::schema::Matrix2d;
use crate::manifest::schema::QuantMatrix;
use crate::manifest::schema::Vector1d;
use crate::manifest::validate::ObservedTensor;
use crate::nvfp4::QuantSource;

/// The wall figures are probes, not a partition: context creation, slot
/// redemption, prefetch join and unmap fall between them, and the allocations
/// submitted under `record_api_wall_ms` execute under
/// `execute_and_drain_wall_ms`. `elapsed_ms` is the submission total: it
/// samples when the call returns, after the A4B expert kernels have drained
/// on the loader stream; the unmap's host cost overlaps that drain.
struct LoadStats {
    /// Every required tensor at its dtype.
    manifest_bytes: usize,
    /// Free-before minus free-after. Signed: this measures the device, not the
    /// process, so anything else running on it moves the number too.
    device_bytes: i64,
    device_free_bytes: usize,
    /// Manifest, shard index and header classification (the config arrives
    /// parsed from the caller). No device. The advisory prefetch workers
    /// start inside this window and keep running past it.
    validate_wall_ms: f64,
    /// Submitting one allocation and one staging plan per BF16-staged tensor
    /// (the shared loader's plan). Wider than the shared loader's
    /// `alloc_api_wall`, which times the allocs alone; the A4B expert byte
    /// staging sits outside this window.
    record_api_wall_ms: f64,
    /// The BF16 staged tensors are consumed here, source read and transfer.
    /// The A4B expert upload, Marlin repack and scale preparation are
    /// enqueued after both windows and are not waited on by any figure here.
    execute_and_drain_wall_ms: f64,
    /// The whole submission call, unmap included; see the struct note.
    elapsed_ms: f64,
    skipped_modality_tensors: usize,
}

struct LayerSlots {
    input_layernorm: VecSlotId,
    post_attention_layernorm: VecSlotId,
    pre_feedforward_layernorm: VecSlotId,
    post_feedforward_layernorm: VecSlotId,
    layer_scalar: f32,
    q_proj: SlotId,
    k_proj: SlotId,
    v_proj: Option<SlotId>,
    o_proj: SlotId,
    q_norm: VecSlotId,
    k_norm: VecSlotId,
    gate: SlotId,
    up: SlotId,
    down: SlotId,
    /// The bf16 half of a routed layer.
    moe: Option<MoeSlots>,
}

struct MoeSlots {
    pre_feedforward_layernorm_2: VecSlotId,
    post_feedforward_layernorm_1: VecSlotId,
    post_feedforward_layernorm_2: VecSlotId,
    router_proj: SlotId,
    router_scale: VecSlotId,
    router_per_expert_scale: VecSlotId,
}

struct RecordedPlan {
    embed_tokens: SlotId,
    norm: VecSlotId,
    layers: Vec<LayerSlots>,
}

fn record_matrix(loader: &mut StagedWeightLoader, tensor: &Matrix2d) -> Result<SlotId> {
    loader.matrix(&tensor.name, tensor.rows, tensor.cols)
}

fn record_vector(loader: &mut StagedWeightLoader, tensor: &Vector1d) -> Result<VecSlotId> {
    loader.vector(&tensor.name, tensor.len)
}

fn read_headers(shards: &[SafeTensors]) -> Result<Vec<(String, Dtype, Vec<usize>)>> {
    let mut headers = Vec::new();
    for shard in shards {
        for name in shard.names() {
            let view = shard
                .tensor(name)
                .map_err(|e| anyhow::anyhow!("Gemma 4: cannot read header of '{name}': {e}"))?;
            headers.push((name.to_string(), view.dtype(), view.shape().to_vec()));
        }
    }
    Ok(headers)
}

fn classify_checkpoint(manifest: &Manifest, shards: &[SafeTensors]) -> Result<usize> {
    let headers = read_headers(shards)?;
    let observed: Vec<ObservedTensor> = headers
        .iter()
        .map(|(name, dtype, shape)| ObservedTensor {
            name,
            dtype: *dtype,
            shape,
        })
        .collect();
    let report = manifest.classify(&observed);
    report.check()?;
    let skipped = report.skipped_modality.len();
    info!(
        "Gemma 4 manifest: {} text tensors, {skipped} modality tensors skipped",
        observed.len() - skipped
    );
    Ok(skipped)
}

fn read_scalar_bf16(shards: &[SafeTensors], name: &str) -> Result<f32> {
    for shard in shards {
        if let Ok(view) = shard.tensor(name) {
            anyhow::ensure!(
                view.dtype() == Dtype::BF16 && view.data().len() == 2,
                "Gemma 4: '{name}' must be a single bf16 scalar"
            );
            let bits = u16::from_le_bytes([view.data()[0], view.data()[1]]);
            let value = half::bf16::from_bits(bits).to_f32();
            anyhow::ensure!(
                value.is_finite(),
                "Gemma 4: '{name}' = {value} is not finite"
            );
            return Ok(value);
        }
    }
    anyhow::bail!("Gemma 4: tensor '{name}' missing from every shard")
}

/// One layer's experts, already stacked and resident.
struct StackedExperts {
    gate: StackedProjection,
    up: StackedProjection,
    down: StackedProjection,
}

/// Upload every routed layer's experts. Returns one entry per layer, empty on
/// the sizes that do not route.
fn upload_experts(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    manifest: &Manifest,
) -> Result<Vec<Option<StackedExperts>>> {
    // A dense checkpoint (12B, 31B) has no routed layer; skip the stager's
    // pinned buffers and thread pool instead of allocating them for nothing.
    if manifest.layers.iter().all(|layer| layer.moe.is_none()) {
        return Ok(manifest.layers.iter().map(|_| None).collect());
    }
    let mut stager = ByteWeightStager::new(ctx)?;
    manifest
        .layers
        .iter()
        .map(|layer| {
            layer
                .moe
                .as_ref()
                .map(|moe| {
                    Ok(StackedExperts {
                        gate: upload_stacked(ctx, &mut stager, shards, &moe.experts, |e| &e.gate)?,
                        up: upload_stacked(ctx, &mut stager, shards, &moe.experts, |e| &e.up)?,
                        down: upload_stacked(ctx, &mut stager, shards, &moe.experts, |e| &e.down)?,
                    })
                })
                .transpose()
        })
        .collect()
}

/// Stack one projection of every expert into a pair of device buffers and
/// upload it as the checkpoint stores it.
///
/// Each expert lands in its own row range, so the buffer is already the shape
/// a batched call wants.
fn upload_stacked(
    ctx: &DeviceContext,
    stager: &mut ByteWeightStager,
    shards: &[SafeTensors],
    experts: &[ExpertTensors],
    pick: fn(&ExpertTensors) -> &QuantMatrix,
) -> Result<StackedProjection> {
    let first = pick(experts.first().ok_or_else(|| {
        anyhow::anyhow!("Gemma 4: a routed layer declares no experts, so nothing can be stacked")
    })?);
    let (rows, values) = first.geometry()?;
    let packed_per_expert = rows * values / crate::nvfp4::PER_BYTE;
    let scales_per_expert = rows * values / crate::nvfp4::GROUP;

    let mut packed = ctx
        .stream
        .alloc_zeros::<u8>(packed_per_expert * experts.len())
        .map_err(|e| anyhow::anyhow!("Gemma 4: cannot hold the stacked experts: {e}"))?;
    let mut scales = ctx
        .stream
        .alloc_zeros::<u8>(scales_per_expert * experts.len())
        .map_err(|e| anyhow::anyhow!("Gemma 4: cannot hold the stacked block scales: {e}"))?;
    let mut sources = Vec::with_capacity(experts.len());
    let mut tensor_scales = Vec::with_capacity(experts.len());
    let mut scale_peak = 0.0f32;

    for (index, expert) in experts.iter().enumerate() {
        let plan = pick(expert);
        // Every expert of one projection is the same shape; a checkpoint that
        // disagrees would otherwise write past its row range.
        let geometry = plan.geometry()?;
        anyhow::ensure!(
            geometry == (rows, values),
            "Gemma 4: expert {index} is {geometry:?}, but expert 0 is {:?}",
            (rows, values)
        );
        let source = QuantSource::read(shards, plan)?;
        anyhow::ensure!(
            source.packed().len() == packed_per_expert
                && source.scales().len() == scales_per_expert,
            "Gemma 4: expert {index} carries {} packed bytes and {} scales, not {packed_per_expert} and {scales_per_expert}",
            source.packed().len(),
            source.scales().len()
        );
        scale_peak = source.scales().iter().fold(scale_peak, |peak, byte| {
            peak.max(crate::nvfp4::decode_e4m3(*byte) * 128.0)
        });
        tensor_scales.push(source.tensor_scale());
        sources.push(source);
    }

    let packed_sources: Vec<&[u8]> = sources.iter().map(QuantSource::packed).collect();
    stager
        .upload(&packed_sources, &mut packed)
        .map_err(|e| anyhow::anyhow!("Gemma 4: expert weights did not upload: {e}"))?;
    let scale_sources: Vec<&[u8]> = sources.iter().map(QuantSource::scales).collect();
    stager
        .upload(&scale_sources, &mut scales)
        .map_err(|e| anyhow::anyhow!("Gemma 4: expert scales did not upload: {e}"))?;

    // Marlin reads the block scale as S0E5M3, so every scale is normalized by
    // one shared power of two and the per-tensor scale takes it back. The
    // factor has to be the same across a projection's experts, which is why it
    // is found here rather than per expert.
    let rescale = marlin_rescale(scale_peak);
    let mut qweight = ctx
        .stream
        .alloc_zeros::<u8>(packed_per_expert * experts.len())
        .map_err(|e| anyhow::anyhow!("Gemma 4: cannot hold the reordered experts: {e}"))?;
    pegainfer_kernels::ops::marlin_repack_4bit(
        ctx,
        &packed,
        &mut qweight,
        experts.len(),
        values,
        rows,
    )?;
    let mut prepared = ctx
        .stream
        .alloc_zeros::<u8>(scales_per_expert * experts.len())
        .map_err(|e| anyhow::anyhow!("Gemma 4: cannot hold the reordered scales: {e}"))?;
    pegainfer_kernels::ops::gemma4_marlin_nvfp4_prepare_scales(
        ctx,
        &scales,
        &mut prepared,
        experts.len(),
        values,
        rows,
        rescale,
    )?;
    // The encoding reads the byte one bit higher than e4m3 does and drops the
    // 2^7 the normalization applied, which is what this bias pays back.
    let bias = 2f32.powi(119);
    let global: Vec<f32> = tensor_scales
        .iter()
        .map(|scale| scale * bias / rescale)
        .collect();
    let global_scales = ctx
        .stream
        .clone_htod(&global)
        .map_err(|e| anyhow::anyhow!("Gemma 4: cannot hold the per-tensor scales: {e}"))?;

    Ok(StackedProjection {
        qweight,
        scales: prepared,
        global_scales,
        rows,
        values,
    })
}

/// The shared power of two that lifts every block scale so its leading bit
/// survives the S0E5M3 re-encoding. Mirrors vLLM's
/// `_nvfp4_compute_scale_factor`, whose bound is the e4m3 maximum.
fn marlin_rescale(peak: f32) -> f32 {
    const CEILING: f32 = 448.0 * 128.0;
    if peak <= 0.0 || peak >= CEILING {
        return 1.0;
    }
    (CEILING / peak).log2().floor().exp2()
}

fn record_plan(
    loader: &mut StagedWeightLoader,
    shards: &[SafeTensors],
    manifest: &Manifest,
) -> Result<RecordedPlan> {
    let embed_tokens = record_matrix(loader, &manifest.embed_tokens)?;
    let norm = record_vector(loader, &manifest.norm)?;
    let mut layers = Vec::with_capacity(manifest.layers.len());
    for layer in &manifest.layers {
        let attention = &layer.attention;
        layers.push(LayerSlots {
            input_layernorm: record_vector(loader, &layer.input_layernorm)?,
            post_attention_layernorm: record_vector(loader, &layer.post_attention_layernorm)?,
            pre_feedforward_layernorm: record_vector(loader, &layer.pre_feedforward_layernorm)?,
            post_feedforward_layernorm: record_vector(loader, &layer.post_feedforward_layernorm)?,
            layer_scalar: read_scalar_bf16(shards, &layer.layer_scalar.name)?,
            q_proj: record_matrix(loader, &attention.q_proj)?,
            k_proj: record_matrix(loader, &attention.k_proj)?,
            v_proj: attention
                .v_proj
                .as_ref()
                .map(|v_proj| record_matrix(loader, v_proj))
                .transpose()?,
            o_proj: record_matrix(loader, &attention.o_proj)?,
            q_norm: record_vector(loader, &attention.q_norm)?,
            k_norm: record_vector(loader, &attention.k_norm)?,
            gate: record_matrix(loader, &layer.mlp.gate)?,
            up: record_matrix(loader, &layer.mlp.up)?,
            down: record_matrix(loader, &layer.mlp.down)?,
            moe: layer
                .moe
                .as_ref()
                .map(|moe| {
                    Ok::<_, anyhow::Error>(MoeSlots {
                        pre_feedforward_layernorm_2: record_vector(
                            loader,
                            &moe.pre_feedforward_layernorm_2,
                        )?,
                        post_feedforward_layernorm_1: record_vector(
                            loader,
                            &moe.post_feedforward_layernorm_1,
                        )?,
                        post_feedforward_layernorm_2: record_vector(
                            loader,
                            &moe.post_feedforward_layernorm_2,
                        )?,
                        router_proj: record_matrix(loader, &moe.router.proj)?,
                        router_scale: record_vector(loader, &moe.router.scale)?,
                        router_per_expert_scale: record_vector(
                            loader,
                            &moe.router.per_expert_scale,
                        )?,
                    })
                })
                .transpose()?,
        });
    }
    Ok(RecordedPlan {
        embed_tokens,
        norm,
        layers,
    })
}

/// Only valid after a successful `finish`.
fn materialize(
    loader: &mut StagedWeightLoader,
    plan: RecordedPlan,
    config: Gemma4Config,
    experts: Vec<Option<StackedExperts>>,
) -> Result<Gemma4Weights> {
    anyhow::ensure!(
        experts.len() == plan.layers.len(),
        "Gemma 4: {} expert sets for {} layers",
        experts.len(),
        plan.layers.len()
    );
    Ok(Gemma4Weights {
        embed_tokens: loader.take(plan.embed_tokens),
        norm: loader.take_vec(plan.norm),
        layers: plan
            .layers
            .into_iter()
            .zip(experts)
            .map(|(slots, experts)| -> Result<Gemma4Layer> {
                Ok(Gemma4Layer {
                    input_layernorm: loader.take_vec(slots.input_layernorm),
                    post_attention_layernorm: loader.take_vec(slots.post_attention_layernorm),
                    pre_feedforward_layernorm: loader.take_vec(slots.pre_feedforward_layernorm),
                    post_feedforward_layernorm: loader.take_vec(slots.post_feedforward_layernorm),
                    layer_scalar: slots.layer_scalar,
                    attention: Gemma4Attention {
                        q_proj: loader.take(slots.q_proj),
                        k_proj: loader.take(slots.k_proj),
                        v_proj: slots.v_proj.map(|slot| loader.take(slot)),
                        o_proj: loader.take(slots.o_proj),
                        q_norm: loader.take_vec(slots.q_norm),
                        k_norm: loader.take_vec(slots.k_norm),
                    },
                    mlp: Gemma4Mlp {
                        gate: loader.take(slots.gate),
                        up: loader.take(slots.up),
                        down: loader.take(slots.down),
                    },
                    moe: match (slots.moe, experts) {
                        (Some(slots), Some(experts)) => Some(Gemma4Moe {
                            pre_feedforward_layernorm_2: loader
                                .take_vec(slots.pre_feedforward_layernorm_2),
                            post_feedforward_layernorm_1: loader
                                .take_vec(slots.post_feedforward_layernorm_1),
                            post_feedforward_layernorm_2: loader
                                .take_vec(slots.post_feedforward_layernorm_2),
                            router_proj: loader.take(slots.router_proj),
                            router_scale: loader.take_vec(slots.router_scale),
                            router_per_expert_scale: loader.take_vec(slots.router_per_expert_scale),
                            gate: experts.gate,
                            up: experts.up,
                            down: experts.down,
                        }),
                        (None, None) => None,
                        // The manifest builds both halves from the same config,
                        // so one without the other is a loader bug rather than
                        // a checkpoint fault.
                        (slots, experts) => anyhow::bail!(
                            "Gemma 4: a layer has {} routed slots and {} expert sets",
                            if slots.is_some() { "some" } else { "no" },
                            if experts.is_some() { "some" } else { "no" }
                        ),
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?,
        config,
    })
}

fn free_device_bytes() -> Result<usize> {
    let (free, _total) = cudarc::driver::result::mem_get_info()
        .map_err(|e| anyhow::anyhow!("Gemma 4: cuMemGetInfo failed: {e}"))?;
    Ok(free)
}

fn elapsed_ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

fn gib(bytes: i64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

impl Gemma4Weights {
    /// Loads the text tower onto one device. Config, manifest and headers are
    /// all checked before a device context exists, so a checkpoint that does
    /// not match its config costs no GPU.
    /// The caller parses and validates the config before the multi-GiB
    /// load; taking it here keeps that validated copy the one the weights
    /// are built from.
    pub(crate) fn from_safetensors(
        model_path: &str,
        device_ordinal: usize,
        config: Gemma4Config,
    ) -> Result<Self> {
        let started = Instant::now();
        let manifest = Manifest::from_config(&config)?;
        let manifest_bytes = manifest.weight_bytes()?;

        let (shard_paths, weight_map) = load_shard_info(model_path)?;
        let prefetch = WeightPrefetch::spawn(&shard_paths);
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = deserialize_shards(&mmaps)?;
        let skipped_modality_tensors = classify_checkpoint(&manifest, &shards)?;
        let validate_wall_ms = elapsed_ms(started);

        let ctx = DeviceContext::new_with_device(device_ordinal)?;
        let free_before = free_device_bytes()?;
        let mut loader = StagedWeightLoader::new(&ctx, &shards, &weight_map)?;

        let recording = Instant::now();
        let plan = record_plan(&mut loader, &shards, &manifest)?;
        let record_api_wall_ms = elapsed_ms(recording);

        let uploading = Instant::now();
        loader.finish()?;
        let execute_and_drain_wall_ms = elapsed_ms(uploading);

        let experts = upload_experts(&ctx, &shards, &manifest)?;
        let weights = materialize(&mut loader, plan, config, experts)?;
        drop(loader);
        drop(prefetch);
        let device_free_bytes = free_device_bytes()?;
        drop(shards);
        // A few hundred ms at this size. Qwen3 backgrounds it to protect its
        // ready time; kept synchronous here so the unmap's host cost lands
        // inside the reported submission total. Lift that spawn into core
        // once an executor wants it too.
        drop(mmaps);

        // The expert kernels ran on this stream while the unmap paid its host cost.
        // Every later weight consumer uses another stream, so this drain is the handoff.
        ctx.sync()
            .map_err(|e| anyhow::anyhow!("Gemma 4: cannot drain the expert kernels: {e}"))?;

        let stats = LoadStats {
            manifest_bytes,
            device_bytes: free_before as i64 - device_free_bytes as i64,
            device_free_bytes,
            validate_wall_ms,
            record_api_wall_ms,
            execute_and_drain_wall_ms,
            elapsed_ms: elapsed_ms(started),
            skipped_modality_tensors,
        };
        info!(
            "Gemma 4 weights resident: {:.2} GiB manifest, {:.2} GiB device, {:.2} GiB free, \
             {} modality tensors skipped; \
             {:.0} ms submission total, of which {:.0} validate, {:.0} record-api, \
             {:.0} execute-and-drain (expert kernels drained)",
            gib(stats.manifest_bytes as i64),
            gib(stats.device_bytes),
            gib(stats.device_free_bytes as i64),
            stats.skipped_modality_tensors,
            stats.elapsed_ms,
            stats.validate_wall_ms,
            stats.record_api_wall_ms,
            stats.execute_and_drain_wall_ms
        );
        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    /// Out of range on any host, so opening a device becomes a visible failure.
    /// Not `usize::MAX`, whose cast to the driver's `int` is -1 by accident.
    const UNOPENABLE_DEVICE: usize = 1024;

    fn model_path() -> Result<String> {
        std::env::var("PEGAINFER_TEST_MODEL_PATH")
            .context("PEGAINFER_TEST_MODEL_PATH must point to a Gemma 4 checkpoint directory")
    }

    /// Every faulty tensor named, before a device exists — hence the unopenable
    /// ordinal: a load that created its context first would fail with a driver
    /// error instead of the manifest's.
    #[test]
    #[ignore = "requires the pinned 12B checkpoint"]
    fn a_disagreeing_config_names_every_faulty_tensor() -> Result<()> {
        let path = model_path()?;
        let staged = tempfile::tempdir()?;
        for entry in std::fs::read_dir(&path)? {
            let entry = entry?;
            // Original name, resolved path: a Hugging Face snapshot points its
            // entries at `blobs/<hash>`, whose name is not the tensor file's.
            let target = entry.path().canonicalize()?;
            std::os::unix::fs::symlink(&target, staged.path().join(entry.file_name()))?;
        }

        let config_path = staged.path().join("config.json");
        let text = std::fs::read_to_string(format!("{path}/config.json"))?;
        let mut config: serde_json::Value = serde_json::from_str(&text)?;
        let width = config["text_config"]["intermediate_size"]
            .as_u64()
            .context("config.json has no text_config.intermediate_size")?;
        let hidden = config["text_config"]["hidden_size"]
            .as_u64()
            .context("config.json has no text_config.hidden_size")?;
        config["text_config"]["intermediate_size"] = serde_json::json!(width - 1);
        // Drop the symlink first: writing through it would edit the checkpoint.
        std::fs::remove_file(&config_path)?;
        std::fs::write(&config_path, serde_json::to_string(&config)?)?;

        let staged_path = staged.path().to_str().context("temp path is not UTF-8")?;
        let parsed = Gemma4Config::from_file(staged_path).expect("config");
        let err = Gemma4Weights::from_safetensors(staged_path, UNOPENABLE_DEVICE, parsed)
            .err()
            .context("a config that disagrees with the checkpoint was accepted")?
            .to_string();

        let layers = config["text_config"]["layer_types"]
            .as_array()
            .context("config.json has no text_config.layer_types")?
            .len();
        assert!(
            err.contains(&format!("{} fault(s)", layers * 3)),
            "expected one fault per MLP tensor per layer: {err}"
        );
        let named = format!(
            "model.language_model.layers.0.mlp.down_proj.weight: checkpoint has [{hidden}, \
             {width}], config implies [{hidden}, {}]",
            width - 1
        );
        assert!(err.contains(&named), "expected `{named}` in:\n{err}");
        Ok(())
    }
}
