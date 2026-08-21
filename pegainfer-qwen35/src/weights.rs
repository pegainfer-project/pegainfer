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
use pegainfer_core::weight_loader::load_tensor_2d;
use pegainfer_core::weight_loader::load_tensor_2d_col_shard;
use pegainfer_core::weight_loader::load_tensor_2d_row_shard;
use pegainfer_core::weight_loader::mmap_shards;
use safetensors::SafeTensors;

use super::config::Config35;
use super::config::LayerType;
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
    /// Fused QKV projection: [q_dim + k_dim + v_dim, hidden_size]
    pub(super) in_proj_qkv: DeviceMatrix,
    /// Z projection (for output gating): [z_dim, hidden_size]
    pub(super) in_proj_z: DeviceMatrix,
    /// Beta projection: [num_value_heads, hidden_size]
    pub(super) in_proj_b: DeviceMatrix,
    /// Alpha projection: [num_value_heads, hidden_size]
    pub(super) in_proj_a: DeviceMatrix,
    /// Depthwise conv1d weight: [qkv_dim * conv_kernel_dim] (flattened from [qkv_dim, 1, 4])
    pub(super) conv1d_weight: DeviceVec,
    /// dt_bias: [num_value_heads] bf16
    pub(super) dt_bias: DeviceVec,
    /// A_log: [num_value_heads] f32
    pub(super) a_log: CudaSlice<f32>,
    /// RMSNorm weight for output normalization: [value_head_dim] f32
    pub(super) norm_weight: CudaSlice<f32>,
    /// Output projection: [hidden_size, z_dim]
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
    /// Opaque kernels-owned AOT operation. `None` is an explicit capability
    /// fallback (non-SM120 or non-Hv32), never a corrupt-artifact fallback.
    pub(super) flashinfer_gdn: Option<pegainfer_kernels::ops::Qwen35GdnAot>,
    #[cfg(feature = "gdn-validation")]
    pub(super) gdn_validation_evidence: super::gdn_validation::GdnValidationEvidenceHandle,
    pub(super) config: Config35,
    pub(super) tensor_parallel: TensorParallelConfig,
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
        tensor_parallel.validate_for(&config, runtime.enable_cuda_graph)?;
        debug!(
            "Config: hidden_size={}, num_layers={}, full_attn={}, linear_attn={}, max_position_embeddings={}, tp_rank={}, tp_world_size={}",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_full_attention_layers(),
            config.num_hidden_layers - config.num_full_attention_layers(),
            config.max_position_embeddings,
            tensor_parallel.rank,
            tensor_parallel.world_size,
        );
        let effective_vocab = super::config::tokenizer_effective_vocab(model_path)?;
        anyhow::ensure!(
            effective_vocab <= config.vocab_size,
            "tokenizer defines ids up to {} but checkpoint vocab_size is {}",
            effective_vocab - 1,
            config.vocab_size,
        );
        if effective_vocab < config.vocab_size {
            config.selection_vocab = effective_vocab;
            info!(
                "output projection: selection bounded to decodable vocab {} (checkpoint pads to {})",
                effective_vocab, config.vocab_size
            );
        }

        let (shard_paths, weight_map) = load_shard_info_fixed(model_path)?;
        debug!("Loading {} safetensor shard(s)", shard_paths.len());
        let prefetch =
            (tensor_parallel.world_size == 1).then(|| WeightPrefetch::spawn(&shard_paths));
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
        let (_, q_rows) = tensor_parallel.shard_range(config.full_attn_q_dim());
        let (kv_row_offset, kv_rows) = tensor_parallel.shard_range(config.full_attn_kv_dim());
        let (inter_row_offset, inter_rows) = tensor_parallel.shard_range(config.intermediate_size);
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
                            tensor_parallel,
                        )?,
                        k_proj: load_tensor_2d_row_shard_if_needed(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.k_proj.weight", attn_prefix),
                            tensor_parallel,
                            kv_row_offset,
                            kv_rows,
                        )?,
                        v_proj: load_tensor_2d_row_shard_if_needed(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.v_proj.weight", attn_prefix),
                            tensor_parallel,
                            kv_row_offset,
                            kv_rows,
                        )?,
                        o_proj: load_tensor_2d_col_shard_if_needed(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.o_proj.weight", attn_prefix),
                            tensor_parallel,
                            tensor_parallel.shard_range(config.full_attn_q_dim()).0,
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
                    LayerKind::LinearAttention(LinearAttentionLayer {
                        in_proj_qkv: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_qkv.weight", attn_prefix),
                        )?,
                        in_proj_z: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_z.weight", attn_prefix),
                        )?,
                        in_proj_b: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_b.weight", attn_prefix),
                        )?,
                        in_proj_a: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_a.weight", attn_prefix),
                        )?,
                        conv1d_weight: load_tensor_1d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.conv1d.weight", attn_prefix),
                        )?,
                        dt_bias: load_tensor_1d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.dt_bias", attn_prefix),
                        )?,
                        a_log: load_tensor_1d_f32(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.A_log", attn_prefix),
                        )?,
                        norm_weight: load_tensor_1d_f32(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.norm.weight", attn_prefix),
                        )?,
                        out_proj: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.out_proj.weight", attn_prefix),
                        )?,
                    })
                }
            };

            let gate_proj = load_tensor_2d_row_shard_if_needed(
                &ctx,
                &shards,
                &weight_map,
                &format!("{}.mlp.gate_proj.weight", prefix),
                tensor_parallel,
                inter_row_offset,
                inter_rows,
            )?;
            let up_proj = load_tensor_2d_row_shard_if_needed(
                &ctx,
                &shards,
                &weight_map,
                &format!("{}.mlp.up_proj.weight", prefix),
                tensor_parallel,
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
                        tensor_parallel,
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
            config.local_num_key_value_heads(tensor_parallel),
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
        let scratch_reserve =
            super::prefill_buffers::GdrChunkwiseScratch35::estimate_bytes(&config, max_prefill_len);
        let recurrent_reserve =
            STATES_PER_DECODE_SLOT * max_batch * super::recurrent_state::bytes_per_request(&config);
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
            config.local_num_key_value_heads(tensor_parallel),
            config.head_dim,
            page_size,
            num_pages,
        )?;

        // The first production specialization is deliberately single-GPU.
        // TP remains an explicit capability fallback to the existing Triton path.
        let flashinfer_gdn = if tensor_parallel.world_size == 1 {
            pegainfer_kernels::ops::Qwen35GdnAot::load_for_production(
                &ctx,
                super::flashinfer_gdn::model_geometry(&config),
            )?
        } else {
            None
        };
        if let Some(backend) = &flashinfer_gdn {
            info!(
                "Qwen3.5 GDN production backend: FlashInfer AOT object {} ({} bytes)",
                backend.artifact_sha256(),
                backend.artifact_size_bytes()
            );
        } else if tensor_parallel.world_size > 1 {
            info!(
                "Qwen3.5 GDN production backend: Triton (explicit capability fallback: TP world_size={})",
                tensor_parallel.world_size
            );
        } else {
            let (major, minor) = ctx.ctx.compute_capability()?;
            info!(
                "Qwen3.5 GDN production backend: Triton (explicit capability fallback: sm_{}{}, geometry={:?})",
                major,
                minor,
                super::flashinfer_gdn::model_geometry(&config)
            );
        }
        Ok(Self {
            ctx,
            flashinfer_gdn,
            #[cfg(feature = "gdn-validation")]
            gdn_validation_evidence: Default::default(),
            config,
            tensor_parallel,
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
        let tp = self.tensor_parallel;
        let full_q = self.config.local_full_attn_gated_q_dim(tp);
        let full_kv = self.config.local_full_attn_kv_dim(tp);
        let linear_qkv = self.config.linear_attn_qkv_dim();
        let linear_z = self.config.linear_attn_z_dim();
        let linear_ba = self.config.linear_num_value_heads;
        let intermediate = self.config.local_intermediate_size(tp);

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
        let graph = super::batch_decode_graph::BatchDecodeGraphState::with_capacity(
            &self.ctx,
            &self.config,
            self.tensor_parallel,
            &self.kv_pool,
            max_batch,
        )?;
        #[cfg(feature = "gdn-validation")]
        let graph = graph.with_validation_evidence(self.gdn_validation_evidence.clone());
        Ok(graph)
    }

    pub(crate) fn create_batch_decode_buffers_with_capacity(
        &self,
        max_batch: usize,
    ) -> anyhow::Result<super::decode_buffers::BatchDecodeBuffers35> {
        super::decode_buffers::BatchDecodeBuffers35::new(
            &self.ctx,
            &self.config,
            self.tensor_parallel,
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
    tensor_parallel: TensorParallelConfig,
) -> GatedQShardRange {
    // HF/PegaInfer kernels interpret q_proj rows as per-head [q, gate] chunks.
    // Keep each local head's q rows adjacent to its gate rows.
    let local_heads = config.local_num_attention_heads(tensor_parallel);
    let head_start = tensor_parallel.rank * local_heads;
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
    tensor_parallel: TensorParallelConfig,
) -> Result<DeviceMatrix> {
    if !tensor_parallel.is_sharded() {
        return load_tensor_2d(ctx, shards, weight_map, name);
    }

    let range = full_attention_gated_q_shard_range(config, tensor_parallel);
    load_tensor_2d_row_shard(ctx, shards, weight_map, name, range.row_offset, range.rows)
}

fn load_tensor_2d_row_shard_if_needed(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    tensor_parallel: TensorParallelConfig,
    row_offset: usize,
    rows: usize,
) -> Result<DeviceMatrix> {
    if tensor_parallel.is_sharded() {
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
    tensor_parallel: TensorParallelConfig,
    col_offset: usize,
    cols: usize,
) -> Result<DeviceMatrix> {
    if tensor_parallel.is_sharded() {
        load_tensor_2d_col_shard(ctx, shards, weight_map, name, col_offset, cols)
    } else {
        load_tensor_2d(ctx, shards, weight_map, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config35 {
        Config35 {
            hidden_size: 2560,
            intermediate_size: 9216,
            num_hidden_layers: 32,
            vocab_size: 248_320,
            selection_vocab: 248_320,
            rms_norm_eps: 1e-6,
            eos_token_id: 151_645,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_num_value_heads: 32,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            rope_theta: 10_000.0,
            rotary_dim: 64,
            max_position_embeddings: 262_144,
            tie_word_embeddings: true,
            layer_types: vec![LayerType::LinearAttention; 32],
        }
    }

    #[test]
    fn gated_q_shard_range_keeps_matching_q_and_gate_rows() {
        let config = test_config();

        let rank0 = full_attention_gated_q_shard_range(
            &config,
            TensorParallelConfig {
                rank: 0,
                world_size: 2,
            },
        );
        assert_eq!(
            rank0,
            GatedQShardRange {
                row_offset: 0,
                rows: 4096,
            }
        );

        let rank1 = full_attention_gated_q_shard_range(
            &config,
            TensorParallelConfig {
                rank: 1,
                world_size: 2,
            },
        );
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
        let tp = TensorParallelConfig {
            rank: 1,
            world_size: 2,
        };

        let (inter_offset, inter_rows) = tp.shard_range(config.intermediate_size);
        assert_eq!((inter_offset, inter_rows), (4608, 4608));
        assert_eq!(config.local_intermediate_size(tp), inter_rows);

        let local_gate_up_rows = 2 * inter_rows;
        let local_down_cols = inter_rows;
        assert_eq!(local_gate_up_rows, 9216);
        assert_eq!(local_down_cols, 4608);
    }
}
