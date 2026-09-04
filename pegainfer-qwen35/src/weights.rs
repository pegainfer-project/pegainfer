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
use pegainfer_core::weight_loader::load_tensor_1d;
use pegainfer_core::weight_loader::load_tensor_1d_f32;
use pegainfer_core::weight_loader::load_tensor_1d_f32_shard;
use pegainfer_core::weight_loader::load_tensor_1d_shard;
use pegainfer_core::weight_loader::load_tensor_1d_stitch;
use pegainfer_core::weight_loader::load_tensor_2d;
use pegainfer_core::weight_loader::load_tensor_2d_col_shard;
use pegainfer_core::weight_loader::load_tensor_2d_row_shard;
use pegainfer_core::weight_loader::load_tensor_2d_row_stitch;
use pegainfer_core::weight_loader::mmap_shards;
use safetensors::SafeTensors;

use super::config::Config35;
use super::config::LayerType;
use super::config::LocalGeometry;
use super::config::TensorParallelConfig;

/// Full attention layer weights (8 layers in Qwen3.5-4B).
pub(super) struct FullAttentionLayer {
    /// Q projection including gate: [num_heads * head_dim * 2, hidden_size]
    pub(super) q_proj: DeviceMatrix,
    /// K projection: [num_kv_heads * head_dim, hidden_size]
    pub(super) k_proj: DeviceMatrix,
    /// V projection: [num_kv_heads * head_dim, hidden_size]
    pub(super) v_proj: DeviceMatrix,
    /// Output projection: [hidden_size, num_heads * head_dim]
    pub(super) o_proj: DeviceMatrix,
    /// QK norm weights: [head_dim] (broadcast to all heads)
    pub(super) q_norm: DeviceVec,
    pub(super) k_norm: DeviceVec,
}

/// Linear attention layer weights (24 layers in Qwen3.5-4B).
pub(super) struct LinearAttentionLayer {
    /// Fused QKV projection: [local_linear_qkv_dim, hidden_size] — rows keep
    /// the global [q | k | v] segment layout, with each segment restricted to
    /// this rank's head-local slice (see `linear_qkv_shard_segments`).
    pub(super) in_proj_qkv: DeviceMatrix,
    /// Z projection (for output gating): [local_linear_z_dim, hidden_size]
    pub(super) in_proj_z: DeviceMatrix,
    /// Beta projection: [local_linear_num_value_heads, hidden_size]
    pub(super) in_proj_b: DeviceMatrix,
    /// Alpha projection: [local_linear_num_value_heads, hidden_size]
    pub(super) in_proj_a: DeviceMatrix,
    /// Depthwise conv1d weight: [local_linear_qkv_dim * conv_kernel_dim]
    /// (flattened from [qkv_dim, 1, 4]); channel layout mirrors in_proj_qkv.
    pub(super) conv1d_weight: DeviceVec,
    /// dt_bias: [local_linear_num_value_heads] bf16
    pub(super) dt_bias: DeviceVec,
    /// A_log: [local_linear_num_value_heads] f32
    pub(super) a_log: CudaSlice<f32>,
    /// RMSNorm weight for output normalization: [value_head_dim] f32 —
    /// head-shared, so replicated on every rank.
    pub(super) norm_weight: CudaSlice<f32>,
    /// Output projection: [hidden_size, local_linear_z_dim] (row-parallel;
    /// the layer all-reduces the partial hidden sum under TP).
    pub(super) out_proj: DeviceMatrix,
}

/// Attention layer — either full or linear.
pub(super) enum LayerKind {
    FullAttention(FullAttentionLayer),
    LinearAttention(LinearAttentionLayer),
}

/// MLP layer weights (shared between both layer types).
#[allow(clippy::struct_field_names)]
pub(super) struct MLP35 {
    pub(super) gate_up_proj: DeviceMatrix,
    pub(super) down_proj: DeviceMatrix,
}

/// Transformer block for Qwen3.5.
pub(super) struct TransformerBlock35 {
    pub(super) input_layernorm: DeviceVec,
    pub(super) attn: LayerKind,
    pub(super) post_attention_layernorm: DeviceVec,
    pub(super) mlp: MLP35,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelRuntimeConfig {
    pub(crate) enable_cuda_graph: bool,
    pub(crate) tensor_parallel: Option<TensorParallelConfig>,
    pub(crate) device_ordinal: usize,
}

impl Default for ModelRuntimeConfig {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
            tensor_parallel: None,
            device_ordinal: 0,
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
    /// Shared paged KV pool for full-attention layers.
    kv_pool: pegainfer_core::kv_pool::KvPool,
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
    ) -> Result<Self> {
        Self::from_safetensors_with_runtime_and_capacity(
            model_path,
            ModelRuntimeConfig {
                device_ordinal,
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
        let geometry = LocalGeometry::try_new(&config, tensor_parallel, runtime.enable_cuda_graph)
            .map_err(anyhow::Error::from)?;
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

        debug!("Loading embeddings to GPU");
        let embed_tokens = load_tensor_2d(
            &ctx,
            &shards,
            &weight_map,
            &format!("{}.embed_tokens.weight", wp),
        )?;
        debug!(
            "embed_tokens: [{}, {}]",
            embed_tokens.rows, embed_tokens.cols
        );

        let lm_head = if config.tie_word_embeddings {
            info!("output projection: tied embed_tokens");
            None
        } else {
            let m = load_tensor_2d(&ctx, &shards, &weight_map, "lm_head.weight")?;
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
        let (_, q_rows) = geometry.shard_range(config.full_attn_q_dim());
        let (kv_row_offset, kv_rows) = geometry.shard_range(config.full_attn_kv_dim());
        let (inter_row_offset, inter_rows) = geometry.shard_range(config.intermediate_size);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("{}.layers.{}", wp, i);
            let layer_type = config.layer_types[i];

            let attn = match layer_type {
                LayerType::FullAttention => {
                    let attn_prefix = format!("{}.self_attn", prefix);
                    LayerKind::FullAttention(FullAttentionLayer {
                        q_proj: load_full_attention_gated_q_proj(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.q_proj.weight", attn_prefix),
                            &config,
                            geometry,
                        )?,
                        k_proj: load_tensor_2d_row_shard_if_needed(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.k_proj.weight", attn_prefix),
                            geometry,
                            kv_row_offset,
                            kv_rows,
                        )?,
                        v_proj: load_tensor_2d_row_shard_if_needed(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.v_proj.weight", attn_prefix),
                            geometry,
                            kv_row_offset,
                            kv_rows,
                        )?,
                        o_proj: load_tensor_2d_col_shard_if_needed(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.o_proj.weight", attn_prefix),
                            geometry,
                            geometry.shard_range(config.full_attn_q_dim()).0,
                            q_rows,
                        )?,
                        q_norm: load_tensor_1d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.q_norm.weight", attn_prefix),
                        )?,
                        k_norm: load_tensor_1d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.k_norm.weight", attn_prefix),
                        )?,
                    })
                }
                LayerType::LinearAttention => {
                    let attn_prefix = format!("{}.linear_attn", prefix);
                    // Phase 2b: shard linear attention over TP ranks. Value-head
                    // unit drives z/b/a/dt_bias/a_log rows; the fused qkv weight
                    // and conv need per-segment head-local stitching.
                    let (vh_offset, vh_rows) =
                        tensor_parallel.shard_range(config.linear_num_value_heads);
                    let (z_row_offset, z_rows) =
                        tensor_parallel.shard_range(config.linear_attn_z_dim());
                    let (z_col_offset, z_cols) = (z_row_offset, z_rows);
                    LayerKind::LinearAttention(LinearAttentionLayer {
                        in_proj_qkv: load_linear_in_proj_qkv_shard(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_qkv.weight", attn_prefix),
                            &config,
                            tensor_parallel,
                        )?,
                        in_proj_z: load_tensor_2d_row_shard_if_needed(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_z.weight", attn_prefix),
                            geometry,
                            z_row_offset,
                            z_rows,
                        )?,
                        in_proj_b: load_tensor_2d_row_shard_if_needed(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_b.weight", attn_prefix),
                            geometry,
                            vh_offset,
                            vh_rows,
                        )?,
                        in_proj_a: load_tensor_2d_row_shard_if_needed(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_a.weight", attn_prefix),
                            geometry,
                            vh_offset,
                            vh_rows,
                        )?,
                        conv1d_weight: load_linear_conv1d_shard(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.conv1d.weight", attn_prefix),
                            &config,
                            tensor_parallel,
                        )?,
                        dt_bias: if tensor_parallel.is_sharded() {
                            load_tensor_1d_shard(
                                &ctx,
                                &shards,
                                &weight_map,
                                &format!("{}.dt_bias", attn_prefix),
                                vh_offset,
                                vh_rows,
                            )?
                        } else {
                            load_tensor_1d(
                                &ctx,
                                &shards,
                                &weight_map,
                                &format!("{}.dt_bias", attn_prefix),
                            )?
                        },
                        a_log: if tensor_parallel.is_sharded() {
                            load_tensor_1d_f32_shard(
                                &ctx,
                                &shards,
                                &weight_map,
                                &format!("{}.A_log", attn_prefix),
                                vh_offset,
                                vh_rows,
                            )?
                        } else {
                            load_tensor_1d_f32(
                                &ctx,
                                &shards,
                                &weight_map,
                                &format!("{}.A_log", attn_prefix),
                            )?
                        },
                        // Gated RMSNorm weight is per value-head dim (128) and
                        // shared by every head: replicated, never sharded.
                        norm_weight: load_tensor_1d_f32(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.norm.weight", attn_prefix),
                        )?,
                        // Row-parallel out_proj: shard input columns to the
                        // local z dim; the layer all-reduces the partial sum.
                        out_proj: load_tensor_2d_col_shard_if_needed(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.out_proj.weight", attn_prefix),
                            geometry,
                            z_col_offset,
                            z_cols,
                        )?,
                    })
                }
            };

            let gate_proj = load_tensor_2d_row_shard_if_needed(
                &ctx,
                &shards,
                &weight_map,
                &format!("{}.mlp.gate_proj.weight", prefix),
                geometry,
                inter_row_offset,
                inter_rows,
            )?;
            let up_proj = load_tensor_2d_row_shard_if_needed(
                &ctx,
                &shards,
                &weight_map,
                &format!("{}.mlp.up_proj.weight", prefix),
                geometry,
                inter_row_offset,
                inter_rows,
            )?;
            let gate_up_proj = DeviceMatrix::vstack(&ctx, &[&gate_proj, &up_proj])?;
            drop(gate_proj);
            drop(up_proj);

            let block = TransformerBlock35 {
                input_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.input_layernorm.weight", prefix),
                )?,
                attn,
                post_attention_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.post_attention_layernorm.weight", prefix),
                )?,
                mlp: MLP35 {
                    gate_up_proj,
                    down_proj: load_tensor_2d_col_shard_if_needed(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.mlp.down_proj.weight", prefix),
                        geometry,
                        inter_row_offset,
                        inter_rows,
                    )?,
                },
            };

            debug!(
                "Loaded layer {}/{}: {:?}",
                i + 1,
                config.num_hidden_layers,
                layer_type
            );
            layers.push(block);
        }

        let norm = load_tensor_1d(&ctx, &shards, &weight_map, &format!("{}.norm.weight", wp))?;

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
        let layout = pegainfer_core::kv_pool::KvLayout::new(
            num_full_layers,
            geometry.local_num_key_value_heads(),
            config.head_dim,
            page_size,
        )
        .expect("kv layout geometry");
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
        let min_kv_bytes = MIN_KV_PAGES * bytes_per_page;
        anyhow::ensure!(
            free_bytes >= scratch_reserve + recurrent_reserve + min_kv_bytes,
            "insufficient device memory for Qwen3.5: {} MB free, but prefill scratch needs {} MB, \
             recurrent state needs {} MB ({STATES_PER_DECODE_SLOT} x {max_batch} decode slots), \
             and the minimal KV pool needs {} MB; lower the decode batch capacity (--max-batch) \
             or use a smaller model",
            free_bytes / (1024 * 1024),
            scratch_reserve / (1024 * 1024),
            recurrent_reserve / (1024 * 1024),
            min_kv_bytes / (1024 * 1024),
        );
        let available = free_bytes - scratch_reserve - recurrent_reserve;
        let kv_budget = (available as f64 * 0.85) as usize;
        let num_pages = (kv_budget / bytes_per_page).max(MIN_KV_PAGES);
        let kv_mb = num_pages * bytes_per_page / (1024 * 1024);
        let scratch_mb = scratch_reserve / (1024 * 1024);
        let recurrent_mb = recurrent_reserve / (1024 * 1024);
        info!(
            "Qwen3.5 KV cache: {num_pages} pages ({kv_mb} MB), prefill scratch reserve: {scratch_mb} MB, recurrent-state reserve: {recurrent_mb} MB ({STATES_PER_DECODE_SLOT} x {max_batch} slots), {:.0}% of {:.0} MB free",
            kv_budget as f64 / free_bytes as f64 * 100.0,
            free_bytes as f64 / 1024.0 / 1024.0
        );
        let kv_pool = pegainfer_core::kv_pool::KvPool::new(
            &ctx,
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
            kv_pool,
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

    pub(crate) fn alloc_kv(&self) -> pegainfer_core::kv_pool::KvState {
        self.kv_pool.alloc()
    }

    pub(crate) fn kv_pool(&self) -> &pegainfer_core::kv_pool::KvPool {
        &self.kv_pool
    }

    pub(crate) fn attach_tp_comm(&mut self, comm: Comm) {
        self.tp_comm = Some(comm);
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

        let full_q_samples: Vec<_> = self
            .layers
            .iter()
            .filter_map(|layer| match &layer.attn {
                LayerKind::FullAttention(attn) => Some((&attn.q_proj, 0)),
                LayerKind::LinearAttention(_) => None,
            })
            .collect();
        let full_kv_samples: Vec<_> = self
            .layers
            .iter()
            .filter_map(|layer| match &layer.attn {
                LayerKind::FullAttention(attn) => Some([(&attn.k_proj, 0), (&attn.v_proj, 0)]),
                LayerKind::LinearAttention(_) => None,
            })
            .flatten()
            .collect();
        let full_o_samples: Vec<_> = self
            .layers
            .iter()
            .filter_map(|layer| match &layer.attn {
                LayerKind::FullAttention(attn) => Some((&attn.o_proj, 0)),
                LayerKind::LinearAttention(_) => None,
            })
            .collect();
        let linear_qkv_samples: Vec<_> = self
            .layers
            .iter()
            .filter_map(|layer| match &layer.attn {
                LayerKind::LinearAttention(attn) => Some((&attn.in_proj_qkv, 0)),
                LayerKind::FullAttention(_) => None,
            })
            .collect();
        let linear_z_samples: Vec<_> = self
            .layers
            .iter()
            .filter_map(|layer| match &layer.attn {
                LayerKind::LinearAttention(attn) => Some((&attn.in_proj_z, 0)),
                LayerKind::FullAttention(_) => None,
            })
            .collect();
        let linear_ba_samples: Vec<_> = self
            .layers
            .iter()
            .filter_map(|layer| match &layer.attn {
                LayerKind::LinearAttention(attn) => {
                    Some([(&attn.in_proj_b, 0), (&attn.in_proj_a, 0)])
                }
                LayerKind::FullAttention(_) => None,
            })
            .flatten()
            .collect();
        let linear_out_samples: Vec<_> = self
            .layers
            .iter()
            .filter_map(|layer| match &layer.attn {
                LayerKind::LinearAttention(attn) => Some((&attn.out_proj, 0)),
                LayerKind::FullAttention(_) => None,
            })
            .collect();
        let gate_up_samples: Vec<_> = self
            .layers
            .iter()
            .map(|layer| (&layer.mlp.gate_up_proj, 0))
            .collect();
        let down_samples: Vec<_> = self
            .layers
            .iter()
            .map(|layer| (&layer.mlp.down_proj, 0))
            .collect();
        let lm_head_samples = [(self.output_projection(), 0)];

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
    ) -> anyhow::Result<super::batch_decode_graph::BatchDecodeGraphState> {
        self.create_batch_decode_graph_state_with_capacity(self.reserved_decode_slots)
    }

    pub(crate) fn create_batch_decode_graph_state_with_capacity(
        &self,
        max_batch: usize,
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
            &self.kv_pool,
            max_batch,
        )
    }

    pub(crate) fn create_batch_decode_buffers_with_capacity(
        &self,
        max_batch: usize,
    ) -> anyhow::Result<super::decode_buffers::BatchDecodeBuffers35> {
        super::decode_buffers::BatchDecodeBuffers35::new(
            &self.ctx,
            &self.config,
            self.geometry,
            max_batch,
            self.kv_pool.capacity_pages(),
            self.kv_pool.padding_page_id(),
        )
    }

    pub(crate) fn is_stop_token(&self, token_id: u32) -> bool {
        token_id == self.config.eos_token_id
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GatedQShardRange {
    row_offset: usize,
    rows: usize,
}

fn full_attention_gated_q_shard_range(
    config: &Config35,
    geometry: LocalGeometry,
) -> GatedQShardRange {
    // HF/PegaInfer kernels interpret q_proj rows as per-head [q, gate] chunks.
    // Keep each local head's q rows adjacent to its gate rows.
    let local_heads = geometry.local_num_attention_heads();
    let head_start = geometry.rank() * local_heads;
    GatedQShardRange {
        row_offset: head_start * config.head_dim * 2,
        rows: local_heads * config.head_dim * 2,
    }
}

fn load_full_attention_gated_q_proj(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    config: &Config35,
    geometry: LocalGeometry,
) -> Result<DeviceMatrix> {
    if !geometry.is_sharded() {
        return load_tensor_2d(ctx, shards, weight_map, name);
    }

    let range = full_attention_gated_q_shard_range(config, geometry);
    load_tensor_2d_row_shard(ctx, shards, weight_map, name, range.row_offset, range.rows)
}

/// Row ranges this rank owns inside the fused global linear-attention qkv
/// projection. The checkpoint stores [all q rows | all k rows | all v rows];
/// each segment contributes its head-local slice so the rank's stitched rows
/// stay [q_local | k_local | v_local]. Never reblock across segments — q rows
/// key on key heads, v rows on value heads (the gated-q lesson).
fn linear_qkv_shard_segments(
    config: &Config35,
    tensor_parallel: TensorParallelConfig,
) -> [(usize, usize); 3] {
    let global_q = config.linear_num_key_heads * config.linear_key_head_dim;
    let global_k = global_q;
    let global_v = config.linear_attn_z_dim();
    let (q_rel, q_rows) = tensor_parallel.shard_range(global_q);
    let (k_rel, k_rows) = tensor_parallel.shard_range(global_k);
    let (v_rel, v_rows) = tensor_parallel.shard_range(global_v);
    [
        (q_rel, q_rows),
        (global_q + k_rel, k_rows),
        (global_q + global_k + v_rel, v_rows),
    ]
}

fn load_linear_in_proj_qkv_shard(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    config: &Config35,
    tensor_parallel: TensorParallelConfig,
) -> Result<DeviceMatrix> {
    if !tensor_parallel.is_sharded() {
        return load_tensor_2d(ctx, shards, weight_map, name);
    }
    let segments = linear_qkv_shard_segments(config, tensor_parallel);
    load_tensor_2d_row_stitch(ctx, shards, weight_map, name, &segments)
}

/// The flattened conv1d weight keeps each channel's kernel taps contiguous
/// ([channel, 1, kernel_dim]); its channel layout mirrors the fused qkv rows,
/// so shard it with the same per-segment ranges scaled by the kernel dim.
fn linear_conv1d_shard_segments(
    config: &Config35,
    tensor_parallel: TensorParallelConfig,
) -> [(usize, usize); 3] {
    let kernel_dim = config.linear_conv_kernel_dim;
    linear_qkv_shard_segments(config, tensor_parallel)
        .map(|(offset, len)| (offset * kernel_dim, len * kernel_dim))
}

fn load_linear_conv1d_shard(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    config: &Config35,
    tensor_parallel: TensorParallelConfig,
) -> Result<DeviceVec> {
    if !tensor_parallel.is_sharded() {
        return load_tensor_1d(ctx, shards, weight_map, name);
    }
    let segments = linear_conv1d_shard_segments(config, tensor_parallel);
    load_tensor_1d_stitch(ctx, shards, weight_map, name, &segments)
}

fn load_tensor_2d_row_shard_if_needed(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    geometry: LocalGeometry,
    row_offset: usize,
    rows: usize,
) -> Result<DeviceMatrix> {
    if geometry.is_sharded() {
        load_tensor_2d_row_shard(ctx, shards, weight_map, name, row_offset, rows)
    } else {
        load_tensor_2d(ctx, shards, weight_map, name)
    }
}

fn load_tensor_2d_col_shard_if_needed(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    geometry: LocalGeometry,
    col_offset: usize,
    cols: usize,
) -> Result<DeviceMatrix> {
    if geometry.is_sharded() {
        load_tensor_2d_col_shard(ctx, shards, weight_map, name, col_offset, cols)
    } else {
        load_tensor_2d(ctx, shards, weight_map, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config35 {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
  "max_position_embeddings": 262144,
  "tie_word_embeddings": true,
  "text_config": {
    "hidden_size": 2560,
    "intermediate_size": 9216,
    "num_hidden_layers": 1,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "vocab_size": 248320,
    "rms_norm_eps": 1e-6,
    "layer_types": ["linear_attention"],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "rope_parameters": { "rope_theta": 10000.0, "partial_rotary_factor": 0.25 },
    "eos_token_id": 151645
  }
}"#,
        )
        .unwrap();
        Config35::from_file(dir.path().to_str().unwrap()).expect("fixture validates")
    }

    fn test_geometry(rank: usize, world_size: usize) -> LocalGeometry {
        let config = test_config();
        let tp = TensorParallelConfig::try_from((rank, world_size)).unwrap();
        LocalGeometry::try_new(&config, tp, false).unwrap()
    }

    #[test]
    fn gated_q_shard_range_keeps_matching_q_and_gate_rows() {
        let config = test_config();

        let rank0 = full_attention_gated_q_shard_range(&config, test_geometry(0, 2));
        assert_eq!(
            rank0,
            GatedQShardRange {
                row_offset: 0,
                rows: 4096,
            }
        );

        let rank1 = full_attention_gated_q_shard_range(&config, test_geometry(1, 2));
        assert_eq!(
            rank1,
            GatedQShardRange {
                row_offset: 4096,
                rows: 4096,
            }
        );
    }

    #[test]
    fn mlp_tp2_uses_matching_gate_up_rows_and_down_cols() {
        let config = test_config();
        let geom = test_geometry(1, 2);

        let (inter_offset, inter_rows) = geom.shard_range(config.intermediate_size);
        assert_eq!((inter_offset, inter_rows), (4608, 4608));
        assert_eq!(geom.local_intermediate_size(), inter_rows);

        let local_gate_up_rows = 2 * inter_rows;
        let local_down_cols = inter_rows;
        assert_eq!(local_gate_up_rows, 9216);
        assert_eq!(local_down_cols, 4608);
    }

    #[test]
    fn linear_qkv_shard_segments_stitch_head_local_slices() {
        // test_config: k heads 16, v heads 32, head dim 128 → q=k=2048, v=4096.
        let config = test_config();
        let rank0 =
            linear_qkv_shard_segments(&config, TensorParallelConfig::try_from((0, 2)).unwrap());
        assert_eq!(rank0, [(0, 1024), (2048, 1024), (4096, 2048)]);

        let rank1 =
            linear_qkv_shard_segments(&config, TensorParallelConfig::try_from((1, 2)).unwrap());
        assert_eq!(rank1, [(1024, 1024), (3072, 1024), (6144, 2048)]);

        // Every rank's stitched rows tile [0, qkv) with no overlap: each
        // segment's local slices across ranks are contiguous and complete.
        for (r0, r1) in rank0.iter().zip(rank1.iter()) {
            assert_eq!(r0.1, r1.1);
            assert_eq!(r1.0, r0.0 + r0.1);
        }
    }

    #[test]
    fn linear_conv1d_shard_segments_scale_by_kernel_dim() {
        let config = test_config();
        let rank1 =
            linear_conv1d_shard_segments(&config, TensorParallelConfig::try_from((1, 2)).unwrap());
        // conv1d.weight is [qkv * 4]: same ranges as qkv, scaled by 4.
        assert_eq!(rank1, [(4096, 4096), (12288, 4096), (24576, 8192)]);
    }

    #[test]
    fn tp1_linear_qkv_shard_segments_cover_full_segments() {
        // TP1 identity: the segments used by the sharded loader would reproduce
        // the full tensor (TP1 itself never stitches — it uses load_tensor_2d).
        let config = test_config();
        let segments = linear_qkv_shard_segments(&config, TensorParallelConfig::default());
        assert_eq!(segments, [(0, 2048), (2048, 2048), (4096, 4096)]);
    }

    /// In-memory safetensors fixture: one F32 tensor whose element `i`
    /// carries value `i` (exact in f32), so any slice maps back to its source
    /// offset. The row/col range math is dtype-agnostic (element units), so
    /// an f32 blob exercises the same layout contract as the bf16 loaders.
    fn safetensors_fixture_f32(name: &str, shape: &[usize]) -> Vec<u8> {
        let len: usize = shape.iter().product();
        let data: Vec<u8> = (0..len).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.to_vec(), &data)
                .unwrap();
        safetensors::serialize([(name.to_string(), view)], None).unwrap()
    }

    #[test]
    fn linear_attention_tp2_slices_match_synthetic_checkpoint() {
        // CPU-only layout contract test: parse a synthetic safetensors blob and
        // verify each rank's stitched slices land on the expected source
        // offsets, including per-segment head-local contiguity.
        let config = test_config();
        let qkv_rows = test_geometry(0, 1).local_linear_qkv_dim(); // 8192
        let kernel_dim = config.linear_conv_kernel_dim; // 4
        // Column count is irrelevant to the row range math; keep it tiny.
        let cols = 8;

        let qkv_blob = safetensors_fixture_f32("w", &[qkv_rows, cols]);
        let qkv = safetensors::SafeTensors::deserialize(&qkv_blob).unwrap();
        let qkv_view = qkv.tensor("w").unwrap();
        assert_eq!(qkv_view.shape(), [qkv_rows, cols]);
        let qkv_elems: Vec<f32> = qkv_view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();

        // conv1d.weight fixture: [qkv * kernel_dim] flattened channels.
        let conv_blob = safetensors_fixture_f32("c", &[qkv_rows * kernel_dim]);
        let conv = safetensors::SafeTensors::deserialize(&conv_blob).unwrap();
        let conv_view = conv.tensor("c").unwrap();
        let conv_elems: Vec<f32> = conv_view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();

        let global_q = config.linear_num_key_heads * config.linear_key_head_dim;
        for rank in 0..2usize {
            let tp = TensorParallelConfig::try_from((rank, 2)).unwrap();
            let geom = LocalGeometry::try_new(&config, tp, false).unwrap();
            let segments = linear_qkv_shard_segments(&config, tp);

            // Stitched matrix = per-segment head-local row slices, in storage
            // order; assert full row content of every stitched row.
            let mut stitched = Vec::new();
            for &(offset, rows) in &segments {
                stitched.extend_from_slice(&qkv_elems[offset * cols..(offset + rows) * cols]);
            }
            assert_eq!(stitched.len(), geom.local_linear_qkv_dim() * cols);
            let mut expected = Vec::new();
            for row in (0..segments[0].1) // q: rank-local key-head slice
                .map(|r| segments[0].0 + r)
                .chain((0..segments[1].1).map(|r| segments[1].0 + r))
                .chain((0..segments[2].1).map(|r| segments[2].0 + r))
            {
                expected.extend_from_slice(&qkv_elems[row * cols..(row + 1) * cols]);
            }
            assert_eq!(stitched, expected, "rank {rank} fused qkv layout");

            // Head-locality: q segment stays inside the rank's key-head range,
            // v segment starts after all global q+k rows plus the rank offset.
            // Local segment dims derive from the fixture config (TP2).
            let lq = config.linear_num_key_heads / 2 * config.linear_key_head_dim;
            let lk = lq;
            let lv = config.linear_num_value_heads / 2 * config.linear_value_head_dim;
            assert_eq!(segments[0].0, rank * lq);
            assert_eq!(segments[2].0, 2 * global_q + rank * lv);

            // conv1d channels mirror qkv rows, each scaled by kernel_dim.
            // The expectation is built from first principles (rank-local
            // head-dim channel windows), not from the segment tuples.
            let conv_segments = linear_conv1d_shard_segments(&config, tp);
            let mut conv_stitched = Vec::new();
            for &(conv_off, conv_len) in &conv_segments {
                conv_stitched.extend_from_slice(&conv_elems[conv_off..conv_off + conv_len]);
            }
            let channels = (rank * lq..(rank + 1) * lq)
                .chain(global_q + rank * lk..global_q + (rank + 1) * lk)
                .chain(2 * global_q + rank * lv..2 * global_q + (rank + 1) * lv);
            let mut conv_expected = Vec::new();
            for c in channels {
                conv_expected.extend_from_slice(&conv_elems[c * kernel_dim..(c + 1) * kernel_dim]);
            }
            assert_eq!(
                conv_stitched.len(),
                geom.local_linear_qkv_dim() * kernel_dim
            );
            assert_eq!(conv_stitched, conv_expected, "rank {rank} conv1d layout");

            // Value-head unit drives in_proj_z rows, in_proj_b/a rows, dt_bias
            // and a_log; out_proj takes the same range as columns.
            let (vh_offset, vh_rows) = tp.shard_range(config.linear_num_value_heads);
            assert_eq!((vh_offset, vh_rows), (rank * 16, 16));
            let (z_offset, z_rows) = tp.shard_range(config.linear_attn_z_dim());
            assert_eq!((z_offset, z_rows), (rank * 2048, 2048));
            assert_eq!(z_rows, geom.local_linear_z_dim());
        }
    }
}
