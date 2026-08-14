//! Checkpoint loading for [`Qwen3Model`]: record the whole plan, then upload.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use log::debug;
use log::info;
use pegainfer_core::rope::RopeTableSpec;
use pegainfer_core::rope::precompute_rope;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::weight_loader::FusedPart;
use pegainfer_core::weight_loader::SlotId;
use pegainfer_core::weight_loader::StagedWeightLoader;
use pegainfer_core::weight_loader::VecSlotId;
use pegainfer_core::weight_loader::WeightPrefetch;
use pegainfer_core::weight_loader::deserialize_shards;
use pegainfer_core::weight_loader::load_shard_info;
use pegainfer_core::weight_loader::mmap_shards;

use super::Attention;
use super::MLP;
use super::PackedLoraRegistry;
use super::Qwen3Model;
use super::TransformerBlock;
use crate::config::Config;
use crate::config::TensorParallelConfig;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelRuntimeConfig {
    pub(crate) enable_cuda_graph: bool,
    pub(crate) tensor_parallel: Option<TensorParallelConfig>,
    pub(crate) device_ordinal: usize,
    pub(crate) max_loras: usize,
    pub(crate) max_lora_rank: usize,
}

impl Default for ModelRuntimeConfig {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
            tensor_parallel: None,
            device_ordinal: 0,
            max_loras: crate::Qwen3LoraOptions::DEFAULT_MAX_LORAS,
            max_lora_rank: crate::Qwen3LoraOptions::DEFAULT_MAX_LORA_RANK,
        }
    }
}

struct LayerSlots {
    input_layernorm: VecSlotId,
    qkv: SlotId,
    o: SlotId,
    q_norm: VecSlotId,
    k_norm: VecSlotId,
    post_attention_layernorm: VecSlotId,
    gate_up: SlotId,
    down: SlotId,
}

impl Qwen3Model {
    pub(crate) fn from_safetensors_with_runtime(
        model_path: &str,
        runtime: ModelRuntimeConfig,
    ) -> Result<Self> {
        info!("Loading model from: {}", model_path);
        debug!("Initializing GPU device {}", runtime.device_ordinal);
        let ctx = DeviceContext::new_with_device(runtime.device_ordinal)?;

        let config = Config::from_file(model_path)?;
        let tensor_parallel = runtime.tensor_parallel.unwrap_or_default();
        tensor_parallel.validate_for(&config)?;

        let (shard_paths, weight_map) = load_shard_info(model_path)?;
        debug!("Loading {} safetensor shard(s)", shard_paths.len());
        let prefetch =
            (tensor_parallel.world_size == 1).then(|| WeightPrefetch::spawn(&shard_paths));
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = deserialize_shards(&mmaps)?;

        let t_gpu = Instant::now();
        let mut loader = StagedWeightLoader::new(&ctx, &shards, &weight_map)?;
        let hidden = config.hidden_size;
        debug!("Loading embeddings to GPU");
        let embed_slot = loader.matrix("model.embed_tokens.weight", config.vocab_size, hidden)?;
        let lm_head_slot = if config.tie_word_embeddings {
            debug!("Using tied input/output embeddings");
            None
        } else {
            debug!("Loading untied LM head to GPU");
            Some(loader.matrix(config.lm_head_tensor_name(), config.vocab_size, hidden)?)
        };

        debug!(
            "Loading layers to GPU: num_layers={}, tp_rank={}, tp_world_size={}",
            config.num_hidden_layers, tensor_parallel.rank, tensor_parallel.world_size,
        );
        let mut layer_slots = Vec::with_capacity(config.num_hidden_layers);
        let q_total = config.num_attention_heads * config.head_dim;
        let kv_total = config.num_key_value_heads * config.head_dim;
        let inter_total = config.intermediate_size;
        let (q_row_offset, q_rows) = tensor_parallel.shard_range(q_total);
        let (kv_row_offset, kv_rows) = tensor_parallel.shard_range(kv_total);
        let (inter_row_offset, inter_rows) = tensor_parallel.shard_range(inter_total);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);

            // shard_range covers world_size == 1 too (offset 0, full rows), so
            // one fused load serves both the sharded and unsharded paths.
            let q_name = format!("{prefix}.self_attn.q_proj.weight");
            let k_name = format!("{prefix}.self_attn.k_proj.weight");
            let v_name = format!("{prefix}.self_attn.v_proj.weight");
            let qkv_proj = loader.fused_rows(
                hidden,
                &[
                    FusedPart {
                        name: &q_name,
                        src_rows: q_total,
                        row_offset: q_row_offset,
                        rows: q_rows,
                    },
                    FusedPart {
                        name: &k_name,
                        src_rows: kv_total,
                        row_offset: kv_row_offset,
                        rows: kv_rows,
                    },
                    FusedPart {
                        name: &v_name,
                        src_rows: kv_total,
                        row_offset: kv_row_offset,
                        rows: kv_rows,
                    },
                ],
            )?;

            let gate_name = format!("{prefix}.mlp.gate_proj.weight");
            let up_name = format!("{prefix}.mlp.up_proj.weight");
            let gate_up_proj = loader.fused_rows(
                hidden,
                &[
                    FusedPart {
                        name: &gate_name,
                        src_rows: inter_total,
                        row_offset: inter_row_offset,
                        rows: inter_rows,
                    },
                    FusedPart {
                        name: &up_name,
                        src_rows: inter_total,
                        row_offset: inter_row_offset,
                        rows: inter_rows,
                    },
                ],
            )?;

            layer_slots.push(LayerSlots {
                input_layernorm: loader
                    .vector(&format!("{}.input_layernorm.weight", prefix), hidden)?,
                qkv: qkv_proj,
                o: if tensor_parallel.is_sharded() {
                    loader.col_shard(
                        &format!("{}.self_attn.o_proj.weight", prefix),
                        hidden,
                        q_total,
                        q_row_offset,
                        q_rows,
                    )?
                } else {
                    loader.matrix(
                        &format!("{}.self_attn.o_proj.weight", prefix),
                        hidden,
                        q_total,
                    )?
                },
                q_norm: loader.vector(
                    &format!("{}.self_attn.q_norm.weight", prefix),
                    config.head_dim,
                )?,
                k_norm: loader.vector(
                    &format!("{}.self_attn.k_norm.weight", prefix),
                    config.head_dim,
                )?,
                post_attention_layernorm: loader.vector(
                    &format!("{}.post_attention_layernorm.weight", prefix),
                    hidden,
                )?,
                gate_up: gate_up_proj,
                down: if tensor_parallel.is_sharded() {
                    loader.col_shard(
                        &format!("{}.mlp.down_proj.weight", prefix),
                        hidden,
                        inter_total,
                        inter_row_offset,
                        inter_rows,
                    )?
                } else {
                    loader.matrix(
                        &format!("{}.mlp.down_proj.weight", prefix),
                        hidden,
                        inter_total,
                    )?
                },
            });
        }

        let norm_slot = loader.vector("model.norm.weight", hidden)?;

        debug!("Precomputing RoPE cache on GPU");
        let (cos_cache, sin_cache) = precompute_rope(
            &ctx,
            &RopeTableSpec {
                rotary_dim: config.head_dim,
                frequency_dim: config.head_dim,
                max_seq_len: config.max_position_embeddings,
                theta: config.rope_theta,
            },
        )?;

        loader.finish()?;
        let embed_tokens = loader.take(embed_slot);
        let lm_head = lm_head_slot.map(|slot| loader.take(slot));
        let norm = loader.take_vec(norm_slot);
        let layers: Vec<TransformerBlock> = layer_slots
            .into_iter()
            .map(|slots| TransformerBlock {
                input_layernorm: loader.take_vec(slots.input_layernorm),
                attention: Attention {
                    qkv_proj: loader.take(slots.qkv),
                    o_proj: loader.take(slots.o),
                    q_norm: loader.take_vec(slots.q_norm),
                    k_norm: loader.take_vec(slots.k_norm),
                    q_dim: q_rows,
                    kv_dim: kv_rows,
                },
                post_attention_layernorm: loader.take_vec(slots.post_attention_layernorm),
                mlp: MLP {
                    gate_up_proj: loader.take(slots.gate_up),
                    down_proj: loader.take(slots.down),
                },
            })
            .collect();
        drop(loader);
        drop(prefetch);
        info!(
            "GPU model loaded in {:.0}ms",
            t_gpu.elapsed().as_secs_f64() * 1e3
        );
        drop(shards);
        // Page-table teardown of the multi-GB mapping is not worth blocking
        // startup on; nothing references the mmaps once every weight is uploaded.
        std::thread::Builder::new()
            .name("qwen3-weights-unmap".into())
            .spawn(move || drop(mmaps))
            .map_err(|e| anyhow::anyhow!("weight unmap thread spawn failed: {e}"))?;

        let num_hidden_layers = config.num_hidden_layers;
        let model = Self {
            ctx,
            config,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            enable_cuda_graph: runtime.enable_cuda_graph,
            tensor_parallel,
            tp_comm: None,
            lora_adapters: HashMap::new(),
            packed_lora: PackedLoraRegistry::empty(runtime.max_loras, num_hidden_layers),
            max_loras: runtime.max_loras,
            max_lora_rank: runtime.max_lora_rank,
        };

        if model.enable_cuda_graph {
            debug!(
                "Decode path CUDA Graph is enabled (single GPU captures on first decode step; TP pre-captures every bucket at startup)"
            );
        } else {
            debug!("Decode path CUDA Graph is disabled");
        }

        Ok(model)
    }
}
