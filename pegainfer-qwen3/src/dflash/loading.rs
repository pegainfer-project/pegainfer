use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use log::debug;
use pegainfer_core::rope::RopeTableSpec;
use pegainfer_core::rope::precompute_rope;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::weight_loader::StagedWeightLoader;
use pegainfer_core::weight_loader::deserialize_shards;
use pegainfer_core::weight_loader::load_shard_info;
use pegainfer_core::weight_loader::load_tensor_1d;
use pegainfer_core::weight_loader::load_tensor_1d_stitch;
use pegainfer_core::weight_loader::load_tensor_2d;
use pegainfer_core::weight_loader::mmap_shards;
use safetensors::Dtype;
use safetensors::tensor::TensorView;

use super::DFlashDraftModel;
use super::conv::GroupedConv;
use super::conv::LayerConvs;
use super::selector::SelectorHead;
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
        let mut config = DFlashConfig::from_file(model_path)
            .with_context(|| format!("load DFlash config from {model_path}"))?;
        config.validate_for_target(target.config())?;
        config.logits_rows = target.config().logits_rows;

        let (shard_paths, weight_map) = load_shard_info(model_path)?;
        debug!(
            "Loading DFlash drafter from {model_path}: {} shard(s)",
            shard_paths.len()
        );
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = deserialize_shards(&mmaps)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let hidden = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let inter = config.intermediate_size;
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("layers.{layer_idx}");

            let q_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{prefix}.self_attn.q_proj.weight"),
                q_dim,
                hidden,
            )?;
            let k_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{prefix}.self_attn.k_proj.weight"),
                kv_dim,
                hidden,
            )?;
            let v_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{prefix}.self_attn.v_proj.weight"),
                kv_dim,
                hidden,
            )?;
            let qkv_proj = DeviceMatrix::vstack(ctx, &[&q_proj, &k_proj, &v_proj])?;
            drop(q_proj);
            drop(k_proj);
            drop(v_proj);

            let gate_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{prefix}.mlp.gate_proj.weight"),
                inter,
                hidden,
            )?;
            let up_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{prefix}.mlp.up_proj.weight"),
                inter,
                hidden,
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
                        hidden,
                        q_dim,
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
                        hidden,
                        inter,
                    )?,
                },
            });
        }

        let convs = if let Some(conv) = &config.conv {
            let mut loader = StagedWeightLoader::new(ctx, &shards, &weight_map)?;
            let width = 2 * conv.taps * (config.hidden_size / conv.group_size);
            let mut kernels = Vec::with_capacity(config.num_hidden_layers);
            for layer in 0..config.num_hidden_layers {
                let mut load = |name: &str| -> Result<_> {
                    let prefix = format!("layers.{layer}.{name}");
                    let base_name = format!("{prefix}.base_kernel");
                    let base = match weight_map.get(&base_name) {
                        Some(&index) => shards[index].tensor(&base_name)?,
                        None => shards
                            .iter()
                            .find_map(|shard| shard.tensor(&base_name).ok())
                            .with_context(|| {
                                format!("Tensor '{base_name}' not found in any shard")
                            })?,
                    };
                    validate_conv_base(&base, &base_name, conv.taps, config.hidden_size)?;
                    Ok((
                        loader.matrix(
                            &format!("{prefix}.kernel_projection.weight"),
                            width,
                            config.hidden_size,
                        )?,
                        load_tensor_1d_stitch(
                            ctx,
                            &shards,
                            &weight_map,
                            &base_name,
                            &[(0, base.data().len() / std::mem::size_of::<half::bf16>())],
                        )?,
                    ))
                };
                kernels.push((load("attention_conv")?, load("mlp_conv")?));
            }

            loader.finish()?;

            let mut take = |(projection, base)| GroupedConv {
                projection: loader.take(projection),
                base,
                block_size: config.block_size,
                group_size: conv.group_size,
            };
            let layers = kernels
                .into_iter()
                .map(|(attention, mlp)| LayerConvs {
                    attention: take(attention),
                    mlp: take(mlp),
                })
                .collect();
            Some(layers)
        } else {
            None
        };

        // Embeddings and the LM head are shared with the target, including the
        // byte-identical copies in the supported native Qwen3-4B checkpoint.
        let norm = load_tensor_1d(ctx, &shards, &weight_map, "norm.weight")?;
        let hidden_norm = load_tensor_1d(ctx, &shards, &weight_map, "hidden_norm.weight")?;
        let fc = load_tensor_2d(
            ctx,
            &shards,
            &weight_map,
            "fc.weight",
            hidden,
            hidden * config.target_layer_ids.len(),
        )?;

        // DSpark Markov head (Phase 1). The confidence head is intentionally skipped.
        let markov = if config.uses_markov_head() {
            let w1 = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                MARKOV_W1_TENSOR,
                config.vocab_size,
                config.markov_rank,
            )?;
            let w2 = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                MARKOV_W2_TENSOR,
                config.vocab_size,
                config.markov_rank,
            )?;
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

        let selector = if config.selector_rank > 0 {
            let rank = config.selector_rank;
            let mut loader = StagedWeightLoader::new(ctx, &shards, &weight_map)?;
            let w = loader.matrix(
                "candidate_selector.hidden_projection.weight",
                rank,
                config.hidden_size,
            )?;
            let a = loader.matrix(
                "candidate_selector.predecessor_codebook",
                config.vocab_size,
                rank,
            )?;
            let b = loader.matrix(
                "candidate_selector.successor_codebook",
                config.vocab_size,
                rank,
            )?;
            loader.finish()?;
            log::info!("selector: rank={rank}, top_k=16");
            Some(SelectorHead {
                projection: loader.take(w),
                predecessor: loader.take(a),
                successor: loader.take(b),
            })
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
            convs,
            norm,
            hidden_norm,
            fc,
            cos_cache,
            sin_cache,
            markov,
            selector,
        })
    }
}

fn validate_conv_base(
    tensor: &TensorView<'_>,
    name: &str,
    taps: usize,
    hidden: usize,
) -> Result<()> {
    ensure!(
        tensor.dtype() == Dtype::BF16,
        "Tensor '{name}': expected dtype BF16, got {:?}",
        tensor.dtype()
    );
    ensure!(
        tensor.shape() == [2, taps, hidden],
        "Tensor '{name}' has shape {:?}, expected [2, {taps}, {hidden}]",
        tensor.shape()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_base_requires_bf16_and_exact_native_shape() -> Result<()> {
        let name = "layers.0.attention_conv.base_kernel";
        for (dtype, shape, expected_error) in [
            (Dtype::BF16, vec![2, 2, 4], None),
            (Dtype::F16, vec![2, 2, 4], Some("expected dtype BF16")),
            (Dtype::BF16, vec![2, 4, 2], Some("expected [2, 2, 4]")),
            (Dtype::BF16, vec![2, 2, 3], Some("expected [2, 2, 4]")),
        ] {
            let bytes = vec![0; shape.iter().product::<usize>() * 2];
            let tensor = TensorView::new(dtype, shape, &bytes)?;
            let result = validate_conv_base(&tensor, name, 2, 4);
            if let Some(expected) = expected_error {
                let error = result
                    .expect_err("invalid convolution base accepted")
                    .to_string();
                assert!(error.contains(name) && error.contains(expected), "{error}");
            } else {
                result?;
            }
        }
        Ok(())
    }
}
