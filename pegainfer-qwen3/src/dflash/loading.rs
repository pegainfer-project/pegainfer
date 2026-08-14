use anyhow::Context;
use anyhow::Result;
use log::debug;
use pegainfer_core::rope::RopeTableSpec;
use pegainfer_core::rope::precompute_rope;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::weight_loader::deserialize_shards;
use pegainfer_core::weight_loader::load_shard_info;
use pegainfer_core::weight_loader::load_tensor_1d;
use pegainfer_core::weight_loader::load_tensor_2d;
use pegainfer_core::weight_loader::mmap_shards;

use super::DFlashDraftModel;
use crate::config::DFlashConfig;
use crate::dspark::MARKOV_W1_TENSOR;
use crate::dspark::MARKOV_W2_TENSOR;
use crate::dspark::MarkovHead;
use crate::weights::Attention;
use crate::weights::MLP;
use crate::weights::Qwen3Model;
use crate::weights::TransformerBlock;

impl DFlashDraftModel {
    pub(crate) fn from_safetensors_for_target(
        ctx: &DeviceContext,
        model_path: &str,
        target: &Qwen3Model,
    ) -> Result<Self> {
        let config = DFlashConfig::from_file(model_path)
            .with_context(|| format!("load DFlash config from {model_path}"))?;
        config.validate_for_target(target.config())?;

        let (shard_paths, weight_map) = load_shard_info(model_path)?;
        debug!(
            "Loading DFlash drafter from {model_path}: {} shard(s)",
            shard_paths.len()
        );
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = deserialize_shards(&mmaps)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("layers.{layer_idx}");

            let q_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{prefix}.self_attn.q_proj.weight"),
            )?;
            let k_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{prefix}.self_attn.k_proj.weight"),
            )?;
            let v_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{prefix}.self_attn.v_proj.weight"),
            )?;
            let q_dim = q_proj.rows;
            let kv_dim = k_proj.rows;
            let qkv_proj = DeviceMatrix::vstack(ctx, &[&q_proj, &k_proj, &v_proj])?;
            drop(q_proj);
            drop(k_proj);
            drop(v_proj);

            let gate_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{prefix}.mlp.gate_proj.weight"),
            )?;
            let up_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{prefix}.mlp.up_proj.weight"),
            )?;
            let gate_up_proj = DeviceMatrix::vstack(ctx, &[&gate_proj, &up_proj])?;
            drop(gate_proj);
            drop(up_proj);

            layers.push(TransformerBlock {
                input_layernorm: load_tensor_1d(
                    ctx,
                    &shards,
                    &weight_map,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                attention: Attention {
                    qkv_proj,
                    o_proj: load_tensor_2d(
                        ctx,
                        &shards,
                        &weight_map,
                        &format!("{prefix}.self_attn.o_proj.weight"),
                    )?,
                    q_norm: load_tensor_1d(
                        ctx,
                        &shards,
                        &weight_map,
                        &format!("{prefix}.self_attn.q_norm.weight"),
                    )?,
                    k_norm: load_tensor_1d(
                        ctx,
                        &shards,
                        &weight_map,
                        &format!("{prefix}.self_attn.k_norm.weight"),
                    )?,
                    q_dim,
                    kv_dim,
                },
                post_attention_layernorm: load_tensor_1d(
                    ctx,
                    &shards,
                    &weight_map,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                mlp: MLP {
                    gate_up_proj,
                    down_proj: load_tensor_2d(
                        ctx,
                        &shards,
                        &weight_map,
                        &format!("{prefix}.mlp.down_proj.weight"),
                    )?,
                },
            });
        }

        let norm = load_tensor_1d(ctx, &shards, &weight_map, "norm.weight")?;
        let hidden_norm = load_tensor_1d(ctx, &shards, &weight_map, "hidden_norm.weight")?;
        let fc = load_tensor_2d(ctx, &shards, &weight_map, "fc.weight")?;

        // DSpark Markov head (Phase 1). The confidence head and the tied
        // embed_tokens/lm_head are intentionally skipped: the head is byte-identical
        // to the target's, which we reuse for the verify-equivalent logits.
        let markov = if config.uses_markov_head() {
            let w1 = load_tensor_2d(ctx, &shards, &weight_map, MARKOV_W1_TENSOR)?;
            let w2 = load_tensor_2d(ctx, &shards, &weight_map, MARKOV_W2_TENSOR)?;
            if config.enable_confidence_head {
                log::info!(
                    "DSpark confidence head present in {model_path} but unused in Phase 1 \
                     (full-block verify, no confidence-scheduled truncation)"
                );
            }
            Some(MarkovHead::new(config.markov_rank, w1, w2)?)
        } else {
            None
        };

        let (cos_cache, sin_cache) = precompute_rope(
            ctx,
            &RopeTableSpec {
                rotary_dim: config.head_dim,
                frequency_dim: config.head_dim,
                max_seq_len: config.max_position_embeddings,
                theta: config.rope_theta,
            },
        )?;
        ctx.sync()?;

        Ok(Self {
            config,
            layers,
            norm,
            hidden_norm,
            fc,
            cos_cache,
            sin_cache,
            markov,
        })
    }
}
