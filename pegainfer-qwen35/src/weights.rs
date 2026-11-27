use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::nccl::safe::Comm;
use cudarc::nccl::safe::ReduceOp;
use log::debug;
use log::info;
use pegainfer_core::kv_pool::KvLayout as DecodeKvLayout;
use pegainfer_core::ops::gemm_rows_into_checked;
use pegainfer_core::ops::suppress_logits_bf16_in_place;
use pegainfer_core::rope::RopeTableSpec;
use pegainfer_core::rope::precompute_rope;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_core::weight_loader::WeightPrefetch;
use pegainfer_core::weight_loader::deserialize_shards;
use pegainfer_core::weight_loader::load_shard_info_fixed;
use pegainfer_core::weight_loader::mmap_shards;
use pegainfer_kv_cache::KvBuffer;
use safetensors::SafeTensors;

use super::capacity::ServingCapacity;
use super::config::Config35;
use super::config::LayerType;
use super::config::LocalGeometry;
use super::config::TensorParallelConfig;
use super::recurrent_state::RecurrentState;
pub(crate) mod layers;
pub(crate) use layers::FullAttentionLayer;
pub(crate) use layers::LayerKind;
pub(crate) use layers::LinearAttentionLayer;
pub(crate) use layers::TransformerBlock35;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelRuntimeConfig {
    pub(crate) enable_cuda_graph: bool,
    pub(crate) tensor_parallel: Option<TensorParallelConfig>,
    pub(crate) device_ordinal: usize,
    pub(crate) gdn_backend: crate::Qwen35GdnBackend,
    /// Per-rank GPU budget reserved for complete recurrent/conv snapshots.
    pub(crate) prefix_snapshot_bytes: usize,
}

impl Default for ModelRuntimeConfig {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
            tensor_parallel: None,
            device_ordinal: 0,
            gdn_backend: crate::Qwen35GdnBackend::Triton,
            prefix_snapshot_bytes: 0,
        }
    }
}

/// Qwen3.5 model (text-only).
pub struct Qwen35Model {
    pub(super) ctx: DeviceContext,
    /// The selected, already-loaded candidate. `None` means Triton was requested.
    pub(super) flashinfer_gdn: Option<pegainfer_kernels::ops::Qwen35GdnAot>,
    pub(super) candidate_decode: Vec<CandidateDecodeGemm>,
    pub(super) config: Config35,
    pub(super) geometry: LocalGeometry,
    pub(super) embed_tokens: DeviceMatrix,
    lm_head: Option<DeviceMatrix>,
    pub(super) layers: Vec<TransformerBlock35>,
    pub(super) norm: DeviceVec,
    // Partial RoPE cache: [max_seq_len * rotary_dim]
    pub(super) cos_cache: DeviceVec,
    pub(super) sin_cache: DeviceVec,
    /// Kernel-facing view of the immutable KV geometry.
    pub(super) decode_kv_layout: DecodeKvLayout,
    /// Rank-local physical full-attention KV storage.
    kv_buffer: KvBuffer,
    /// Capacity ledger planned and frozen at startup (decode_width / max_resident / cohorts / KV / scratch).
    capacity: ServingCapacity,
    tp_comm: Option<Comm>,
    /// -inf suppression for the tile-alignment pad rows, applied by
    /// [`Qwen35Model::output_logits_into`] (absent when the selection width
    /// needs no padding).
    pad_logit_suppress: Option<crate::ops::SuppressIds>,
}

// SAFETY: A Qwen3.5 model instance is bound to one CUDA device and driven from
// one owning scheduler/worker thread at a time. TP constructs one independent
// rank-local model per worker; the model is moved between threads only during
// startup, never shared for concurrent mutation.
unsafe impl Send for Qwen35Model {}
unsafe impl Sync for Qwen35Model {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DecodeProjection {
    Down,
    V,
}

pub(super) struct CandidateDecodeGemm {
    pub(super) layer: usize,
    pub(super) projection: DecodeProjection,
    pub(super) batch: usize,
    recipe: OnceLock<pegainfer_kernels::ops::Qwen35DecodeGemm>,
}

impl CandidateDecodeGemm {
    pub(super) fn weights<'a>(&self, model: &'a Qwen35Model) -> Result<&'a DeviceMatrix> {
        let layer = model
            .layers
            .get(self.layer)
            .context("decode target layer is out of range")?;
        match self.projection {
            DecodeProjection::Down => Ok(&layer.mlp.down_proj),
            DecodeProjection::V => Ok(&layer
                .attn
                .full_attention()
                .context("V target requires a full-attention layer")?
                .v_proj),
        }
    }

    fn prepare(&self, model: &Qwen35Model) -> Result<()> {
        anyhow::ensure!(
            model.flashinfer_gdn.is_some(),
            "decode adjustment requires a candidate"
        );
        let weights = self.weights(model)?;
        anyhow::ensure!(
            super::batch_decode_graph::BATCH_BUCKETS.contains(&self.batch)
                && self.batch <= crate::ops::GEMM_LT_MAX_N,
            "decode target must use a cuBLASLt decode bucket"
        );
        anyhow::ensure!(
            self.batch <= model.capacity.max_graph_bucket(),
            "decode target exceeds allocated decode capacity"
        );
        if self.recipe.get().is_none() {
            let recipe =
                pegainfer_kernels::ops::Qwen35DecodeGemm::prepare(&model.ctx, weights, self.batch)
                    .with_context(|| {
                        format!(
                            "prepare {:?} layer={} batch={}",
                            self.projection, self.layer, self.batch
                        )
                    })?;
            anyhow::ensure!(
                self.recipe.set(recipe).is_ok(),
                "decode recipe initialized concurrently"
            );
        }
        Ok(())
    }
}

fn candidate_decode_targets(
    config: &Config35,
    geometry: LocalGeometry,
    capacity: usize,
    candidate: bool,
) -> Vec<CandidateDecodeGemm> {
    // Keep the 4B projection policy model-owned, outside the shared shape cache.
    if candidate
        && config.hidden_size == 2560
        && geometry.local_intermediate_size() == 9216
        && geometry.local_full_attn_kv_dim() == 1024
    {
        [8, 4, 2]
            .into_iter()
            .filter(|&batch| batch <= capacity)
            .flat_map(|batch| {
                [(0, DecodeProjection::Down), (3, DecodeProjection::V)]
                    .into_iter()
                    .map(move |(layer, projection)| CandidateDecodeGemm {
                        layer,
                        projection,
                        batch,
                        recipe: OnceLock::new(),
                    })
            })
            .collect()
    } else {
        Vec::new()
    }
}

impl Qwen35Model {
    /// Loads a single-GPU model with capacity planned for `(decode_width, max_resident)` (used by tests/executors).
    pub(crate) fn from_safetensors(
        model_path: &str,
        device_ordinal: usize,
        decode_width: usize,
        prefix_snapshot_bytes: usize,
    ) -> Result<Self> {
        Self::from_safetensors_with_runtime_and_capacity(
            model_path,
            ModelRuntimeConfig {
                device_ordinal,
                prefix_snapshot_bytes,
                ..Default::default()
            },
            decode_width,
            decode_width,
        )
    }

    pub(crate) fn from_safetensors_with_runtime_and_capacity(
        model_path: &str,
        runtime: ModelRuntimeConfig,
        decode_width: usize,
        max_resident: usize,
    ) -> Result<Self> {
        // Qwen3.5's logits gates ran on the batch-1 matvec; every other line
        // keeps the cublas plan it was validated on.
        pegainfer_kernels::ops::set_n1_matvec(true);
        let tensor_parallel = runtime.tensor_parallel.unwrap_or_default();
        ServingCapacity::validate_request(
            decode_width,
            max_resident,
            tensor_parallel.world_size(),
        )?;
        info!("Loading Qwen3.5 model from: {}", model_path);
        debug!("Initializing GPU device {}", runtime.device_ordinal);
        let ctx = DeviceContext::new_with_device(runtime.device_ordinal)?;

        let mut config = Config35::from_file(model_path)?;
        let geometry =
            LocalGeometry::try_new(&config, tensor_parallel).map_err(anyhow::Error::from)?;
        // Resolve and preload exactly once, before measuring free memory or
        // allocating KV. A linked artifact alone never changes the default.
        let flashinfer_gdn = match runtime.gdn_backend {
            crate::Qwen35GdnBackend::Triton => None,
            crate::Qwen35GdnBackend::FlashInferCandidate => {
                anyhow::ensure!(
                    geometry.world_size() == 1,
                    "Qwen3.5 flashinfer-candidate requires TP world_size=1"
                );
                anyhow::ensure!(
                    config.linear_value_head_dim == 128,
                    "Qwen3.5 flashinfer-candidate requires value head dimension 128"
                );
                Some(pegainfer_kernels::ops::Qwen35GdnAot::load_for_production(
                    &ctx,
                    super::flashinfer_gdn::model_geometry(&config),
                )?)
            }
        };
        if let Some(backend) = &flashinfer_gdn {
            info!(
                "Qwen3.5 GDN: requested={} resolved=flashinfer-candidate object_sha256={}",
                runtime.gdn_backend,
                backend.artifact_sha256()
            );
        } else {
            info!(
                "Qwen3.5 GDN: requested={} resolved=triton",
                runtime.gdn_backend
            );
        }
        debug!(
            "Config: hidden_size={}, num_layers={}, full_attn={}, linear_attn={}, max_position_embeddings={}, tp_rank={}, tp_world_size={}",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_full_attention_layers(),
            config.num_hidden_layers - config.num_full_attention_layers(),
            config.max_position_embeddings,
            geometry.rank(),
            geometry.world_size(),
        );
        let effective_vocab =
            pegainfer_frontend::decodable_vocab::decodable_vocab_width(model_path)?;
        config
            .bound_selection_vocab(effective_vocab)
            .map_err(anyhow::Error::from)?;
        if config.selection_vocab != effective_vocab {
            info!(
                "output projection: selection width {} = decodable vocab {} + tile-alignment pad (checkpoint has {})",
                config.selection_vocab, effective_vocab, config.vocab_size
            );
        }

        let (shard_paths, weight_map) = load_shard_info_fixed(model_path)?;
        debug!("Loading {} safetensor shard(s)", shard_paths.len());
        let prefetch = (geometry.world_size() == 1).then(|| WeightPrefetch::spawn(&shard_paths));
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = deserialize_shards(&mmaps)?;

        let t_gpu = Instant::now();
        // Weight prefix for Qwen3.5 text model
        let wp = "model.language_model";
        let src = layers::WeightSource::new(&ctx, &shards, &weight_map, &config, geometry);

        debug!("Loading embeddings to GPU");
        let embed_tokens = src.tensor_2d(
            &format!("{}.embed_tokens.weight", wp),
            config.vocab_size,
            config.hidden_size,
        )?;
        debug!(
            "embed_tokens: [{}, {}]",
            embed_tokens.rows, embed_tokens.cols
        );

        let lm_head = if config.tie_word_embeddings {
            info!("output projection: tied embed_tokens");
            None
        } else {
            let m = src.tensor_2d("lm_head.weight", config.vocab_size, config.hidden_size)?;
            info!("output projection: untied lm_head [{}, {}]", m.rows, m.cols);
            Some(m)
        };

        debug!(
            "Loading layers to GPU: num_layers={}",
            config.num_hidden_layers
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("{}.layers.{}", wp, i);
            let layer_type = config.layer_types[i];
            layers.push(TransformerBlock35::load(&src, &prefix, layer_type)?);
            debug!(
                "Loaded layer {}/{}: {:?}",
                i + 1,
                config.num_hidden_layers,
                layer_type
            );
        }

        let norm = src.tensor_1d(&format!("{}.norm.weight", wp))?;

        debug!(
            "Precomputing partial RoPE cache (rotary_dim={}, max_position_embeddings={})",
            config.rotary_dim, config.max_position_embeddings
        );
        let (cos_cache, sin_cache) = precompute_rope(
            &ctx,
            &RopeTableSpec {
                rotary_dim: config.rotary_dim,
                frequency_dim: config.rotary_dim,
                max_seq_len: config.max_position_embeddings,
                theta: config.rope_theta,
            },
        )?;

        ctx.sync()?;
        drop(prefetch);
        info!(
            "GPU model loaded in {:.0}ms",
            t_gpu.elapsed().as_secs_f64() * 1e3
        );
        if runtime.enable_cuda_graph {
            debug!("Decode path CUDA Graph is enabled");
        } else {
            debug!("Decode path CUDA Graph is disabled");
        }
        // Paged KV pool for full-attention layers, sized by ServingCapacity.
        let page_size = super::capacity::KV_PAGE_SIZE;
        let num_full_layers = config.num_full_attention_layers();
        let decode_kv_layout = DecodeKvLayout::new(
            num_full_layers,
            geometry.local_num_key_value_heads(),
            config.head_dim,
            page_size,
        )?;
        let (free_bytes, _total_bytes) = cudarc::driver::result::mem_get_info()
            .map_err(|e| anyhow::anyhow!("cuMemGetInfo failed: {e}"))?;
        let (cc_major, cc_minor) = ctx.ctx.compute_capability()?;
        let split_decode_max_batch =
            super::decode_buffers::split_decode_max_batch(cc_major * 10 + cc_minor);
        let max_prefill_len = super::prefill::SCRATCH_ESTIMATE_SEQ;
        let scratch_reserve = match &flashinfer_gdn {
            Some(backend) => super::flashinfer_gdn::FlashInferGdnChunkResources::estimate_bytes(
                &config,
                geometry,
                backend,
                max_prefill_len,
                page_size,
            ),
            None => {
                super::prefill_buffers::prefill_scratch_bytes(&config, geometry, max_prefill_len)
            }
        };
        let capacity = ServingCapacity::plan(
            &config,
            geometry,
            decode_width,
            max_resident,
            runtime.enable_cuda_graph,
            split_decode_max_batch,
            scratch_reserve,
            runtime.prefix_snapshot_bytes,
            free_bytes,
        )?;
        capacity.log_ledger(geometry);
        let kv_buffer = KvBuffer::new(
            &ctx.stream,
            num_full_layers,
            geometry.local_num_key_value_heads(),
            config.head_dim,
            capacity.kv_page_size,
            capacity.kv_pages,
        )?;
        // The alignment pad rows are real checkpoint embeddings but not
        // decodable tokens, so they must never win selection.
        let pad_logit_suppress = if config.selection_vocab > config.decodable_vocab {
            let ids: Vec<u32> = (config.decodable_vocab..config.selection_vocab)
                .map(|id| id as u32)
                .collect();
            Some(crate::ops::SuppressIds::upload(
                &ctx,
                &ids,
                config.selection_vocab,
            )?)
        } else {
            None
        };

        let candidate_decode = candidate_decode_targets(
            &config,
            geometry,
            capacity.max_graph_bucket(),
            flashinfer_gdn.is_some(),
        );
        Ok(Self {
            ctx,
            flashinfer_gdn,
            candidate_decode,
            config,
            geometry,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            kv_buffer,
            decode_kv_layout,
            capacity,
            tp_comm: None,
            pad_logit_suppress,
        })
    }

    pub(crate) fn config(&self) -> &Config35 {
        &self.config
    }

    pub(crate) fn capacity(&self) -> &ServingCapacity {
        &self.capacity
    }

    /// Only the GEMM tuning helper samples this directly; logits go through
    /// [`Qwen35Model::output_logits_into`] so the pad-row mask cannot be skipped.
    fn output_projection(&self) -> &DeviceMatrix {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
    }

    /// Write selectable logits for `normed` rows: the output-projection GEMM
    /// followed by the tile-alignment pad-row mask.
    ///
    /// The two belong together — a site that ran the GEMM alone would leave the
    /// pad rows selectable, which on the wire is an undecodable id fed back into
    /// later decode steps rather than a failure — so this is the only way the
    /// model produces logits.
    pub(crate) fn output_logits_into(
        &self,
        normed: &HiddenStates,
        logits: &mut HiddenStates,
    ) -> Result<()> {
        let vocab = self.config.selection_vocab;
        gemm_rows_into_checked(
            &self.ctx,
            self.output_projection(),
            0,
            vocab,
            normed,
            logits,
        )?;
        if let Some(suppress) = &self.pad_logit_suppress {
            suppress_logits_bf16_in_place(&self.ctx, logits, suppress)?;
        }
        Ok(())
    }

    pub(crate) fn ensure_rope_cache_covers(&self, positions: usize) -> Result<()> {
        let cache_positions = self.cos_cache.len / self.config.rotary_dim;
        anyhow::ensure!(
            positions <= cache_positions,
            "Qwen3.5 RoPE cache covers {cache_positions} positions, requested {positions}; max_position_embeddings={}",
            self.config.max_position_embeddings
        );
        Ok(())
    }

    pub(crate) fn device_ctx(&self) -> &DeviceContext {
        &self.ctx
    }

    pub(crate) fn kv_buffer(&self) -> &KvBuffer {
        &self.kv_buffer
    }

    pub(crate) fn prefix_snapshot_slots(&self) -> usize {
        self.capacity.prefix_snapshot_slots
    }

    pub(crate) fn attach_tp_comm(&mut self, comm: Comm) {
        self.tp_comm = Some(comm);
    }

    /// Force NCCL connect before any CUDA Graph capture records a collective
    /// (lazy connect inside `cuStreamBeginCapture` wedges the capture). NCCL
    /// 2.22+ connects per size-selected algorithm, so warm one all-reduce at
    /// every decode bucket's message size. No-op without a TP communicator.
    pub(crate) fn warmup_tp_collective(&self) -> Result<()> {
        if let Some(comm) = &self.tp_comm {
            let buckets = super::batch_decode_graph::BATCH_BUCKETS;
            let max_elems = buckets.last().unwrap() * self.config.hidden_size;
            let mut scratch = self
                .ctx
                .stream
                .alloc_zeros::<half::bf16>(max_elems)
                .map_err(|e| anyhow::anyhow!("alloc NCCL warm-up scratch: {e}"))?;
            for &bucket in buckets {
                let mut view = scratch.slice_mut(0..bucket * self.config.hidden_size);
                comm.all_reduce_in_place(&mut view, &ReduceOp::Sum)
                    .map_err(|e| {
                        anyhow::anyhow!("Qwen3.5 NCCL warm-up all-reduce failed: {e:?}")
                    })?;
            }
            self.ctx
                .stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!("Qwen3.5 NCCL warm-up sync failed: {e}"))?;
        }
        Ok(())
    }

    pub(crate) fn all_reduce_hidden(&self, hidden: &mut HiddenStates) -> Result<()> {
        self.all_reduce_hidden_untraced(hidden)
    }

    fn all_reduce_hidden_untraced(&self, hidden: &mut HiddenStates) -> Result<()> {
        if let Some(comm) = &self.tp_comm {
            comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum)
                .map_err(|e| anyhow::anyhow!("Qwen3.5 NCCL all-reduce failed: {e:?}"))?;
        }
        Ok(())
    }

    /// Tune small-batch decode GEMM algorithms on the thread that will capture
    /// or replay the CUDA Graph. cuBLASLt plans are thread-local, so scheduler
    /// workers and model-local executors must call this after binding CUDA.
    /// Repeated calls on the same thread return from the existing plan cache;
    /// calls on different worker threads populate separate thread-local plans.
    pub(crate) fn tune_decode_gemm_algos(&self) -> Result<()> {
        let ctx = &self.ctx;
        let hidden = self.config.hidden_size;
        let vocab = self.config.selection_vocab;
        let geom = self.geometry;
        let full_q = geom.local_full_attn_gated_q_dim();
        let full_kv = geom.local_full_attn_kv_dim();
        let linear_qkv = geom.local_linear_qkv_dim();
        let linear_z = geom.local_linear_z_dim();
        let linear_ba = geom.local_linear_num_value_heads();
        let intermediate = geom.local_intermediate_size();

        let full_attn = || {
            self.layers
                .iter()
                .filter_map(|layer| layer.attn.full_attention())
        };
        let linear_attn = || {
            self.layers
                .iter()
                .filter_map(|layer| layer.attn.linear_attention())
        };
        let full_q_samples = sample_mats(full_attn().map(|attn| &attn.q_proj));
        let full_kv_samples =
            sample_mats(full_attn().flat_map(|attn| [&attn.k_proj, &attn.v_proj]));
        let full_o_samples = sample_mats(full_attn().map(|attn| &attn.o_proj));
        let linear_qkvz_samples = sample_mats(linear_attn().map(|attn| &attn.in_proj_qkvz));
        let linear_ba_samples = sample_mats(linear_attn().map(|attn| &attn.in_proj_ba));
        let linear_out_samples = sample_mats(linear_attn().map(|attn| &attn.out_proj));
        let gate_up_samples = sample_mats(self.layers.iter().map(|layer| &layer.mlp.gate_up_proj));
        let down_samples = sample_mats(self.layers.iter().map(|layer| &layer.mlp.down_proj));
        let lm_head_samples = sample_mats([self.output_projection()]);

        let max_tuned_bucket = self.capacity.max_graph_bucket();
        for &n in super::batch_decode_graph::BATCH_BUCKETS
            .iter()
            .filter(|&&bucket| {
                // Keep in sync with MAX_SHARED_SM_DECODE_BATCH: buckets above
                // GEMM_LT_MAX_N do not have an independent tuned decode GEMM
                // route today, so Shared-SM overlap rejects them at startup.
                bucket <= crate::ops::GEMM_LT_MAX_N && bucket <= max_tuned_bucket
            })
        {
            tune_if_nonempty(ctx, &full_q_samples, full_q, n)?;
            tune_if_nonempty(ctx, &full_kv_samples, full_kv, n)?;
            tune_if_nonempty(ctx, &full_o_samples, hidden, n)?;
            tune_if_nonempty(ctx, &linear_qkvz_samples, linear_qkv + linear_z, n)?;
            tune_if_nonempty(ctx, &linear_ba_samples, 2 * linear_ba, n)?;
            tune_if_nonempty(ctx, &linear_out_samples, hidden, n)?;
            crate::ops::gemm_lt_tune(ctx, &gate_up_samples, 2 * intermediate, n)?;
            crate::ops::gemm_lt_tune(ctx, &down_samples, hidden, n)?;
            crate::ops::gemm_lt_tune(ctx, &lm_head_samples, vocab, n)?;
        }
        for target in &self.candidate_decode {
            target.prepare(self)?;
        }
        Ok(())
    }

    pub(super) fn candidate_decode_recipe(
        &self,
        weights: &DeviceMatrix,
        batch: usize,
    ) -> Result<Option<&pegainfer_kernels::ops::Qwen35DecodeGemm>> {
        // K/V share a tuned shape, and full-attention uses a separate KV layer
        // index. Match the model-owned weight object, not either of those keys.
        for target in &self.candidate_decode {
            if target.batch == batch && std::ptr::eq(target.weights(self)?, weights) {
                return target
                    .recipe
                    .get()
                    .map(Some)
                    .context("candidate decode recipe must be prepared before decode");
            }
        }
        Ok(None)
    }

    /// Creates the shared CUDA Graph padding recurrent state used across all single-GPU cohorts.
    pub(crate) fn create_padding_recurrent_state(&self) -> Result<Arc<RecurrentState>> {
        Ok(Arc::new(RecurrentState::new(
            &self.ctx,
            &self.config,
            self.geometry,
        )?))
    }

    /// Creates CUDA Graph batch decode state for one single-GPU cohort (slot pool + shared padding state).
    pub(crate) fn create_batch_decode_graph_state_for_cohort(
        &self,
        slot_capacity: usize,
        padding_page_id: i32,
        padding_state: Arc<RecurrentState>,
    ) -> anyhow::Result<super::batch_decode_graph::BatchDecodeGraphState> {
        anyhow::ensure!(
            slot_capacity <= self.capacity.decode_width,
            "requested cohort capacity {slot_capacity} exceeds loaded decode_width {}",
            self.capacity.decode_width
        );
        super::batch_decode_graph::BatchDecodeGraphState::with_capacity_and_padding(
            &self.ctx,
            &self.config,
            self.geometry,
            self.kv_buffer.layout().page_size,
            padding_page_id,
            slot_capacity,
            padding_state,
        )
    }

    /// Creates a single-cohort CUDA Graph batch decode state (used by tests and logits executor).
    pub(crate) fn create_batch_decode_graph_state(
        &self,
        padding_page_id: i32,
    ) -> anyhow::Result<super::batch_decode_graph::BatchDecodeGraphState> {
        let padding_state = self.create_padding_recurrent_state()?;
        self.create_batch_decode_graph_state_for_cohort(
            self.capacity.cohort_widths[0],
            padding_page_id,
            padding_state,
        )
    }

    /// Creates the fixed-layout single-cohort CUDA Graph batch decode state for TP mode.
    pub(crate) fn create_batch_decode_graph_state_with_capacity(
        &self,
        max_batch: usize,
        padding_page_id: i32,
    ) -> anyhow::Result<super::batch_decode_graph::BatchDecodeGraphState> {
        anyhow::ensure!(
            max_batch <= self.capacity.max_graph_bucket(),
            "requested graph capacity {max_batch} exceeds loaded capacity {}",
            self.capacity.max_graph_bucket()
        );
        super::batch_decode_graph::BatchDecodeGraphState::with_capacity(
            &self.ctx,
            &self.config,
            self.geometry,
            self.kv_buffer.layout().page_size,
            padding_page_id,
            max_batch,
        )
    }

    pub(crate) fn create_batch_decode_buffers_with_capacity(
        &self,
        max_batch: usize,
        padding_page_id: i32,
    ) -> anyhow::Result<super::decode_buffers::BatchDecodeBuffers35> {
        super::decode_buffers::BatchDecodeBuffers35::new(
            &self.ctx,
            &self.config,
            self.geometry,
            max_batch,
            self.kv_buffer.layout().page_size,
            padding_page_id,
        )
    }

    pub(crate) fn is_stop_token(&self, token_id: u32) -> bool {
        self.config.eos_token_ids.contains(&token_id)
    }
}

/// Wrap the matrices one tuning pass samples into cuBLASLt tune candidates.
fn sample_mats<'a>(
    mats: impl IntoIterator<Item = &'a DeviceMatrix>,
) -> Vec<(&'a DeviceMatrix, usize)> {
    mats.into_iter().map(|m| (m, 0)).collect()
}

fn tune_if_nonempty(
    ctx: &DeviceContext,
    samples: &[(&DeviceMatrix, usize)],
    rows: usize,
    n: usize,
) -> Result<()> {
    if samples.is_empty() {
        return Ok(());
    }
    crate::ops::gemm_lt_tune(ctx, samples, rows, n)
}
