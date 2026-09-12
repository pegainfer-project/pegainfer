use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::nccl::safe::Comm;
use cudarc::nccl::safe::ReduceOp;
use log::debug;
use log::info;
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

use super::config::Config35;
use super::config::LayerType;
use super::config::LocalGeometry;
use super::config::TensorParallelConfig;
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
    /// Per-rank GPU budget reserved for complete recurrent/conv snapshots.
    pub(crate) prefix_snapshot_bytes: usize,
}

impl Default for ModelRuntimeConfig {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
            tensor_parallel: None,
            device_ordinal: 0,
            prefix_snapshot_bytes: 0,
        }
    }
}

/// Qwen3.5 model (text-only).
pub struct Qwen35Model {
    pub(super) ctx: DeviceContext,
    pub(super) config: Config35,
    pub(super) geometry: LocalGeometry,
    pub(super) embed_tokens: DeviceMatrix,
    lm_head: Option<DeviceMatrix>,
    pub(super) layers: Vec<TransformerBlock35>,
    pub(super) norm: DeviceVec,
    // Partial RoPE cache: [max_seq_len * rotary_dim]
    pub(super) cos_cache: DeviceVec,
    pub(super) sin_cache: DeviceVec,
    /// Rank-local physical full-attention KV storage.
    kv_buffer: KvBuffer,
    /// Complete recurrent snapshot slots reserved by the load-time budget.
    prefix_snapshot_slots: usize,
    /// Decode-slot count the recurrent-state reserve was sized for.
    /// Physical decode capacity actually allocated (recurrent-state slots,
    /// decode buffers, CUDA-graph slots). Always a `BATCH_BUCKETS` value.
    reserved_decode_slots: usize,
    /// Scheduler concurrent-request cap requested at load (`--max-batch`). May
    /// sit below `reserved_decode_slots` when the request is not a bucket
    /// (e.g. `--max-batch 5` allocates bucket 8 but admits at most 5). See #470.
    pub(super) decode_admission_batch: usize,
    tp_comm: Option<Comm>,
}

// SAFETY: A Qwen3.5 model instance is bound to one CUDA device and driven from
// one owning scheduler/worker thread at a time. TP constructs one independent
// rank-local model per worker; the model is moved between threads only during
// startup, never shared for concurrent mutation.
unsafe impl Send for Qwen35Model {}
unsafe impl Sync for Qwen35Model {}

/// Graph slot state + one in-flight prefill transient per decode slot.
const STATES_PER_DECODE_SLOT: usize = 2;
/// KV-pool floor, also the low-memory fail-fast threshold.
const MIN_KV_PAGES: usize = 64;

impl Qwen35Model {
    pub fn from_safetensors_with_options(
        model_path: &str,
        enable_cuda_graph: bool,
    ) -> Result<Self> {
        Self::from_safetensors_with_runtime(
            model_path,
            ModelRuntimeConfig {
                enable_cuda_graph,
                ..Default::default()
            },
        )
    }
}

impl Qwen35Model {
    /// `max_batch` is the requested concurrent-request cap in `1..=MAX_BATCH`.
    /// It need not be a decode bucket: the physical decode capacity is rounded
    /// up to the next `BATCH_BUCKETS` value while the scheduler still admits at
    /// most `max_batch` (see #470 and `decode_admission_batch`).
    pub(crate) fn from_safetensors(
        model_path: &str,
        device_ordinal: usize,
        max_batch: usize,
        prefix_snapshot_bytes: usize,
    ) -> Result<Self> {
        Self::from_safetensors_with_runtime_and_capacity(
            model_path,
            ModelRuntimeConfig {
                device_ordinal,
                prefix_snapshot_bytes,
                ..Default::default()
            },
            max_batch,
        )
    }

    pub(crate) fn from_safetensors_with_runtime(
        model_path: &str,
        runtime: ModelRuntimeConfig,
    ) -> Result<Self> {
        Self::from_safetensors_with_runtime_and_capacity(
            model_path,
            runtime,
            super::batch_decode_graph::MAX_BATCH,
        )
    }

    fn from_safetensors_with_runtime_and_capacity(
        model_path: &str,
        runtime: ModelRuntimeConfig,
        max_batch: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            (1..=super::batch_decode_graph::MAX_BATCH).contains(&max_batch),
            "decode batch capacity must be in 1..={}, got {max_batch}",
            super::batch_decode_graph::MAX_BATCH,
        );
        // Requested scheduler admission cap; physical decode capacity is the
        // next CUDA-graph bucket >= this (e.g. `--max-batch 5` allocates bucket
        // 8 but admits at most 5, see #470). Everything below sizes to the
        // physical bucket; only `decode_admission_batch` keeps the request.
        let decode_admission_batch = max_batch;
        let max_batch = super::batch_decode_graph::bucket_for(max_batch);
        info!("Loading Qwen3.5 model from: {}", model_path);
        debug!("Initializing GPU device {}", runtime.device_ordinal);
        let ctx = DeviceContext::new_with_device(runtime.device_ordinal)?;

        let mut config = Config35::from_file(model_path)?;
        let tensor_parallel = runtime.tensor_parallel.unwrap_or_default();
        let geometry =
            LocalGeometry::try_new(&config, tensor_parallel).map_err(anyhow::Error::from)?;
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
        let effective_vocab = super::config::tokenizer_effective_vocab(model_path)?;
        config
            .bound_selection_vocab(effective_vocab)
            .map_err(anyhow::Error::from)?;
        if config.selection_vocab < config.vocab_size {
            info!(
                "output projection: selection bounded to decodable vocab {} (checkpoint pads to {})",
                config.selection_vocab, config.vocab_size
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
        let embed_tokens = src.tensor_2d(&format!("{}.embed_tokens.weight", wp))?;
        debug!(
            "embed_tokens: [{}, {}]",
            embed_tokens.rows, embed_tokens.cols
        );

        let lm_head = if config.tie_word_embeddings {
            info!("output projection: tied embed_tokens");
            None
        } else {
            let m = src.tensor_2d("lm_head.weight")?;
            anyhow::ensure!(
                m.rows == config.vocab_size && m.cols == config.hidden_size,
                "lm_head.weight is [{}, {}], expected [vocab {}, hidden {}]",
                m.rows,
                m.cols,
                config.vocab_size,
                config.hidden_size,
            );
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
        // Paged KV pool for the 8 full-attention layers.
        let page_size = 16usize;
        let num_full_layers = config.num_full_attention_layers();
        let layout = pegainfer_kv_cache::KvLayout::new(
            num_full_layers,
            geometry.local_num_key_value_heads(),
            config.head_dim,
            page_size,
        );
        let bytes_per_page = layout.page_stride * std::mem::size_of::<half::bf16>();
        let (free_bytes, _total_bytes) = cudarc::driver::result::mem_get_info()
            .map_err(|e| anyhow::anyhow!("cuMemGetInfo failed: {e}"))?;
        // Reserve space for prefill scratch (GDR chunkwise + per-layer transients)
        // before allocating KV pool, so prefill doesn't OOM.
        let max_prefill_len = super::prefill::SCRATCH_ESTIMATE_SEQ;
        let scratch_reserve = super::prefill_buffers::GdrChunkwiseScratch35::estimate_bytes(
            &config,
            geometry,
            max_prefill_len,
        );
        let recurrent_reserve = STATES_PER_DECODE_SLOT
            * max_batch
            * super::recurrent_state::bytes_per_request(&config, geometry);
        let prefix_snapshot_bytes = runtime.prefix_snapshot_bytes;
        let snapshot_bytes_per_slot = super::recurrent_state::bytes_per_request(&config, geometry);
        let snapshot_slots = prefix_snapshot_bytes / snapshot_bytes_per_slot;
        anyhow::ensure!(
            prefix_snapshot_bytes == 0 || snapshot_slots > 0,
            "Qwen3.5 prefix-cache budget is {} MiB, but one recurrent/conv snapshot requires {:.3} MiB",
            prefix_snapshot_bytes / (1024 * 1024),
            snapshot_bytes_per_slot as f64 / 1024.0 / 1024.0,
        );
        let snapshot_reserve = snapshot_slots * snapshot_bytes_per_slot;
        let min_kv_bytes = MIN_KV_PAGES * bytes_per_page;
        anyhow::ensure!(
            free_bytes >= scratch_reserve + recurrent_reserve + snapshot_reserve + min_kv_bytes,
            "insufficient device memory for Qwen3.5: {} MB free, but prefill scratch needs {} MB, \
             recurrent state needs {} MB ({STATES_PER_DECODE_SLOT} x {max_batch} decode slots), \
             prefix snapshots need {} MB ({} slots), and the minimal KV pool needs {} MB; \
             lower the decode batch capacity (--max-batch) or the prefix-cache budget \
             or use a smaller model",
            free_bytes / (1024 * 1024),
            scratch_reserve / (1024 * 1024),
            recurrent_reserve / (1024 * 1024),
            snapshot_reserve / (1024 * 1024),
            snapshot_slots,
            min_kv_bytes / (1024 * 1024),
        );
        let available = free_bytes - scratch_reserve - recurrent_reserve - snapshot_reserve;
        let kv_budget = (available as f64 * 0.85) as usize;
        let num_pages = (kv_budget / bytes_per_page).max(MIN_KV_PAGES);
        let kv_mb = num_pages * bytes_per_page / (1024 * 1024);
        let scratch_mb = scratch_reserve / (1024 * 1024);
        let recurrent_mb = recurrent_reserve / (1024 * 1024);
        let snapshot_mb = snapshot_reserve / (1024 * 1024);
        info!(
            "Qwen3.5 KV cache: {num_pages} pages ({kv_mb} MB), prefill scratch reserve: {scratch_mb} MB, recurrent-state reserve: {recurrent_mb} MB ({STATES_PER_DECODE_SLOT} x {max_batch} slots), prefix snapshots: {snapshot_slots} slots ({snapshot_mb} MB), {:.0}% of {:.0} MB free",
            kv_budget as f64 / free_bytes as f64 * 100.0,
            free_bytes as f64 / 1024.0 / 1024.0
        );
        let kv_buffer = KvBuffer::new(
            &ctx.stream,
            num_full_layers,
            geometry.local_num_key_value_heads(),
            config.head_dim,
            page_size,
            num_pages,
        )?;

        Ok(Self {
            ctx,
            config,
            geometry,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            kv_buffer,
            prefix_snapshot_slots: snapshot_slots,
            reserved_decode_slots: max_batch,
            decode_admission_batch,
            tp_comm: None,
        })
    }

    pub(crate) fn config(&self) -> &Config35 {
        &self.config
    }

    pub(super) fn output_projection(&self) -> &DeviceMatrix {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
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
        self.prefix_snapshot_slots
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
        let linear_qkv_samples = sample_mats(linear_attn().map(|attn| &attn.in_proj_qkv));
        let linear_z_samples = sample_mats(linear_attn().map(|attn| &attn.in_proj_z));
        let linear_ba_samples =
            sample_mats(linear_attn().flat_map(|attn| [&attn.in_proj_b, &attn.in_proj_a]));
        let linear_out_samples = sample_mats(linear_attn().map(|attn| &attn.out_proj));
        let gate_up_samples = sample_mats(self.layers.iter().map(|layer| &layer.mlp.gate_up_proj));
        let down_samples = sample_mats(self.layers.iter().map(|layer| &layer.mlp.down_proj));
        let lm_head_samples = sample_mats([self.output_projection()]);

        for &n in super::batch_decode_graph::BATCH_BUCKETS
            .iter()
            .filter(|&&bucket| {
                // Keep in sync with MAX_SHARED_SM_DECODE_BATCH: buckets above
                // GEMM_LT_MAX_N do not have an independent tuned decode GEMM
                // route today, so Shared-SM overlap rejects them at startup.
                bucket <= crate::ops::GEMM_LT_MAX_N && bucket <= self.reserved_decode_slots
            })
        {
            tune_if_nonempty(ctx, &full_q_samples, full_q, n)?;
            tune_if_nonempty(ctx, &full_kv_samples, full_kv, n)?;
            tune_if_nonempty(ctx, &full_o_samples, hidden, n)?;
            tune_if_nonempty(ctx, &linear_qkv_samples, linear_qkv, n)?;
            tune_if_nonempty(ctx, &linear_z_samples, linear_z, n)?;
            tune_if_nonempty(ctx, &linear_ba_samples, linear_ba, n)?;
            tune_if_nonempty(ctx, &linear_out_samples, hidden, n)?;
            crate::ops::gemm_lt_tune(ctx, &gate_up_samples, 2 * intermediate, n)?;
            crate::ops::gemm_lt_tune(ctx, &down_samples, hidden, n)?;
            crate::ops::gemm_lt_tune(ctx, &lm_head_samples, vocab, n)?;
        }
        Ok(())
    }

    /// Create the CUDA Graph batch decode state at the loaded capacity.
    pub(crate) fn create_batch_decode_graph_state(
        &self,
        padding_page_id: i32,
    ) -> anyhow::Result<super::batch_decode_graph::BatchDecodeGraphState> {
        self.create_batch_decode_graph_state_with_capacity(
            self.reserved_decode_slots,
            padding_page_id,
        )
    }

    pub(crate) fn create_batch_decode_graph_state_with_capacity(
        &self,
        max_batch: usize,
        padding_page_id: i32,
    ) -> anyhow::Result<super::batch_decode_graph::BatchDecodeGraphState> {
        anyhow::ensure!(
            max_batch <= self.reserved_decode_slots,
            "requested graph capacity {max_batch} exceeds loaded capacity {}",
            self.reserved_decode_slots
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
        token_id == self.config.eos_token_id
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
