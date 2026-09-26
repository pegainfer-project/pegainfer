use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use half::bf16;
use log::info;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_core::tensor::HiddenStatesRef;
use pegainfer_core::weight_loader::deserialize_shards;
use pegainfer_core::weight_loader::load_shard_info;
use pegainfer_core::weight_loader::load_tensor_1d;
use pegainfer_core::weight_loader::load_tensor_2d;
use pegainfer_core::weight_loader::mmap_shards;
use safetensors::Dtype;

use crate::Config;
use crate::device::activate;
use crate::ep::ExpertParallelLayout;

pub(crate) struct DriverRankModel {
    pub(crate) ctx: DeviceContext,
    pub(crate) layout: ExpertParallelLayout,
    pub(crate) embed_tokens: DeviceMatrix,
    pub(crate) lm_head: DeviceMatrix,
    pub(crate) norm_host: Vec<f32>,
    pub(crate) norm_device: pegainfer_core::tensor::DeviceVec,
    pub(crate) layers: Vec<LayerWeights>,
}

pub(crate) struct ExpertRankModel {
    pub(crate) ctx: DeviceContext,
    pub(crate) layout: ExpertParallelLayout,
    gate_devices: Vec<Option<DeviceMatrix>>,
    layers: Vec<Option<Vec<ExpertMlp>>>,
}

pub(crate) struct LayerWeights {
    pub(crate) input_layernorm_host: Vec<f32>,
    pub(crate) input_layernorm_device: pegainfer_core::tensor::DeviceVec,
    pub(crate) post_attention_layernorm_host: Vec<f32>,
    pub(crate) post_attention_layernorm_device: pegainfer_core::tensor::DeviceVec,
    pub(crate) attention: AttentionWeights,
    pub(crate) mlp: MlpWeights,
}

pub(crate) struct AttentionWeights {
    pub(crate) q_proj: DeviceMatrix,
    pub(crate) kv_a_proj: DeviceMatrix,
    pub(crate) kv_a_norm_host: Vec<f32>,
    pub(crate) kv_a_norm_device: pegainfer_core::tensor::DeviceVec,
    pub(crate) kv_b_proj: DeviceMatrix,
    pub(crate) o_proj: DeviceMatrix,
}

pub(crate) enum MlpWeights {
    Dense(DenseMlp),
    Moe(MoeMlp),
}

pub(crate) struct DenseMlp {
    gate_up_proj: DeviceMatrix,
    down_proj: DeviceMatrix,
}

pub(crate) struct DenseMlpForwardScratch {
    gate_up: HiddenStates,
    act: HiddenStates,
    pub(crate) out: HiddenStates,
}

impl DenseMlpForwardScratch {
    pub(crate) fn new(ctx: &DeviceContext, mlp: &DenseMlp, seq_len: usize) -> Result<Self> {
        Ok(Self {
            gate_up: HiddenStates::zeros(ctx, mlp.gate_up_proj.rows, seq_len)?,
            act: HiddenStates::zeros(ctx, mlp.gate_up_proj.rows / 2, seq_len)?,
            out: HiddenStates::zeros(ctx, mlp.down_proj.rows, seq_len)?,
        })
    }
}

pub(crate) struct MoeMlp {
    pub(crate) gate_host: Vec<f32>,
    // Probe-only fixed-topology MoE uses this for device routing. The eager
    // oracle path keeps using `gate_host` for host-side routing.
    pub(crate) gate_device: DeviceMatrix,
    pub(crate) shared: DenseMlp,
    experts: Vec<ExpertMlp>,
}

pub(crate) struct ExpertMlp {
    global_expert: usize,
    pub(crate) dense: DenseMlp,
}

impl DriverRankModel {
    pub(crate) fn load(
        model_path: &Path,
        config: &Config,
        layout: ExpertParallelLayout,
        device_ordinal: usize,
    ) -> Result<Self> {
        let ctx = DeviceContext::new_with_device(device_ordinal)?;
        activate(&ctx)?;

        with_weight_shards(model_path, layout.rank(), "driver", |shards, weight_map| {
            let gpu_started = Instant::now();
            let hidden = config.hidden_size;
            let embed_tokens = load_tensor_2d(
                &ctx,
                shards,
                weight_map,
                "model.embed_tokens.weight",
                config.vocab_size,
                hidden,
            )?;
            ensure!(
                !config.tie_word_embeddings,
                "DeepSeek-V2-Lite first gate expects untied lm_head"
            );
            let lm_head = load_tensor_2d(
                &ctx,
                shards,
                weight_map,
                "lm_head.weight",
                config.vocab_size,
                hidden,
            )?;
            let norm_host = load_tensor_1d_host(shards, weight_map, "model.norm.weight")?;
            let norm_device = load_tensor_1d(&ctx, shards, weight_map, "model.norm.weight")?;

            let mut layers = Vec::with_capacity(config.num_hidden_layers);
            for layer_idx in 0..config.num_hidden_layers {
                let prefix = format!("model.layers.{layer_idx}");
                let input_layernorm_name = format!("{prefix}.input_layernorm.weight");
                let input_layernorm_host =
                    load_tensor_1d_host(shards, weight_map, &input_layernorm_name)?;
                let input_layernorm_device =
                    load_tensor_1d(&ctx, shards, weight_map, &input_layernorm_name)?;
                let post_attention_layernorm_name =
                    format!("{prefix}.post_attention_layernorm.weight");
                let post_attention_layernorm_host =
                    load_tensor_1d_host(shards, weight_map, &post_attention_layernorm_name)?;
                let post_attention_layernorm_device =
                    load_tensor_1d(&ctx, shards, weight_map, &post_attention_layernorm_name)?;
                let attn = format!("{prefix}.self_attn");
                let kv_a_norm_name = format!("{attn}.kv_a_layernorm.weight");
                let kv_a_norm_host = load_tensor_1d_host(shards, weight_map, &kv_a_norm_name)?;
                let kv_a_norm_device = load_tensor_1d(&ctx, shards, weight_map, &kv_a_norm_name)?;
                let attention = AttentionWeights {
                    q_proj: load_tensor_2d(
                        &ctx,
                        shards,
                        weight_map,
                        &format!("{attn}.q_proj.weight"),
                        config.q_proj_rows(),
                        hidden,
                    )?,
                    kv_a_proj: load_tensor_2d(
                        &ctx,
                        shards,
                        weight_map,
                        &format!("{attn}.kv_a_proj_with_mqa.weight"),
                        config.kv_a_proj_rows(),
                        hidden,
                    )?,
                    kv_a_norm_host,
                    kv_a_norm_device,
                    kv_b_proj: load_tensor_2d(
                        &ctx,
                        shards,
                        weight_map,
                        &format!("{attn}.kv_b_proj.weight"),
                        config.kv_b_proj_rows(),
                        config.kv_lora_rank,
                    )?,
                    o_proj: load_tensor_2d(
                        &ctx,
                        shards,
                        weight_map,
                        &format!("{attn}.o_proj.weight"),
                        hidden,
                        config.o_proj_cols(),
                    )?,
                };
                let mlp_prefix = format!("{prefix}.mlp");
                let mlp = if config.is_moe_layer(layer_idx) {
                    MlpWeights::Moe(load_moe_mlp(
                        &ctx,
                        shards,
                        weight_map,
                        config,
                        &layout,
                        &mlp_prefix,
                    )?)
                } else {
                    MlpWeights::Dense(load_dense_mlp(
                        &ctx,
                        shards,
                        weight_map,
                        &mlp_prefix,
                        hidden,
                        config.intermediate_size,
                    )?)
                };
                layers.push(LayerWeights {
                    input_layernorm_host,
                    input_layernorm_device,
                    post_attention_layernorm_host,
                    post_attention_layernorm_device,
                    attention,
                    mlp,
                });
            }

            ctx.sync()?;
            info!(
                "DeepSeek-V2-Lite EP rank {} driver GPU model loaded in {:.0}ms",
                layout.rank(),
                duration_ms(gpu_started.elapsed())
            );

            Ok(Self {
                ctx,
                layout,
                embed_tokens,
                lm_head,
                norm_host,
                norm_device,
                layers,
            })
        })
    }

    pub(crate) fn routed_expert(
        &self,
        layer_idx: usize,
        global_expert: usize,
    ) -> Result<&ExpertMlp> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} out of range"))?;
        let MlpWeights::Moe(moe) = &layer.mlp else {
            bail!("layer {layer_idx} is not a MoE layer");
        };
        routed_expert_from_slice(&self.layout, &moe.experts, global_expert)
    }
}

impl ExpertRankModel {
    pub(crate) fn load(
        model_path: &Path,
        config: &Config,
        layout: ExpertParallelLayout,
        device_ordinal: usize,
    ) -> Result<Self> {
        let ctx = DeviceContext::new_with_device(device_ordinal)?;
        activate(&ctx)?;

        with_weight_shards(model_path, layout.rank(), "expert", |shards, weight_map| {
            let gpu_started = Instant::now();
            let mut layers = Vec::with_capacity(config.num_hidden_layers);
            let mut gate_devices = Vec::with_capacity(config.num_hidden_layers);
            for layer_idx in 0..config.num_hidden_layers {
                if config.is_moe_layer(layer_idx) {
                    let prefix = format!("model.layers.{layer_idx}.mlp");
                    gate_devices.push(Some(load_tensor_2d(
                        &ctx,
                        shards,
                        weight_map,
                        &format!("{prefix}.gate.weight"),
                        config.n_routed_experts,
                        config.hidden_size,
                    )?));
                    layers.push(Some(load_owned_experts(
                        &ctx, shards, weight_map, config, &layout, &prefix,
                    )?));
                } else {
                    gate_devices.push(None);
                    layers.push(None);
                }
            }

            ctx.sync()?;
            info!(
                "DeepSeek-V2-Lite EP rank {} expert GPU model loaded in {:.0}ms",
                layout.rank(),
                duration_ms(gpu_started.elapsed())
            );

            Ok(Self {
                ctx,
                layout,
                gate_devices,
                layers,
            })
        })
    }

    pub(crate) fn gate_device(&self, layer_idx: usize) -> Result<&DeviceMatrix> {
        self.gate_devices
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} out of range"))?
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} is not a MoE layer"))
    }

    pub(crate) fn routed_expert(
        &self,
        layer_idx: usize,
        global_expert: usize,
    ) -> Result<&ExpertMlp> {
        let experts = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} out of range"))?
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} is not a MoE layer"))?;
        routed_expert_from_slice(&self.layout, experts, global_expert)
    }
}

fn with_weight_shards<T>(
    model_path: &Path,
    rank: usize,
    role: &str,
    load: impl FnOnce(&[safetensors::SafeTensors<'_>], &HashMap<String, usize>) -> Result<T>,
) -> Result<T> {
    let model_path_str = model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path must be valid UTF-8"))?;
    let (shard_paths, weight_map) = load_shard_info(model_path_str)?;
    info!(
        "DeepSeek-V2-Lite EP rank {rank} {role}: loading {} safetensor shard(s)",
        shard_paths.len()
    );
    let host_started = Instant::now();
    let mmaps = mmap_shards(&shard_paths)?;
    let shards = deserialize_shards(&mmaps)?;
    info!(
        "DeepSeek-V2-Lite EP rank {rank} {role}: mmap+deserialize {} safetensor shard(s) in {:.0}ms",
        shard_paths.len(),
        duration_ms(host_started.elapsed())
    );
    load(&shards, &weight_map)
}

fn routed_expert_from_slice<'a>(
    layout: &ExpertParallelLayout,
    experts: &'a [ExpertMlp],
    global_expert: usize,
) -> Result<&'a ExpertMlp> {
    let local_expert = layout.local_expert(global_expert)?;
    let expert = experts.get(local_expert).ok_or_else(|| {
        anyhow::anyhow!(
            "rank {} local expert {} missing for global expert {}",
            layout.rank(),
            local_expert,
            global_expert
        )
    })?;
    ensure!(
        expert.global_expert == global_expert,
        "rank {} local expert {} expected global {}, got {}",
        layout.rank(),
        local_expert,
        global_expert,
        expert.global_expert
    );
    Ok(expert)
}

fn load_dense_mlp(
    ctx: &DeviceContext,
    shards: &[safetensors::SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
) -> Result<DenseMlp> {
    let gate_proj = load_tensor_2d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.gate_proj.weight"),
        intermediate,
        hidden,
    )?;
    let up_proj = load_tensor_2d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.up_proj.weight"),
        intermediate,
        hidden,
    )?;
    let gate_up_proj = DeviceMatrix::vstack(ctx, &[&gate_proj, &up_proj])?;
    let down_proj = load_tensor_2d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.down_proj.weight"),
        hidden,
        intermediate,
    )?;
    Ok(DenseMlp {
        gate_up_proj,
        down_proj,
    })
}

fn load_moe_mlp(
    ctx: &DeviceContext,
    shards: &[safetensors::SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    config: &Config,
    layout: &ExpertParallelLayout,
    prefix: &str,
) -> Result<MoeMlp> {
    let gate_name = format!("{prefix}.gate.weight");
    let gate_host = load_tensor_2d_host(shards, weight_map, &gate_name)?;
    let gate_device = load_tensor_2d(
        ctx,
        shards,
        weight_map,
        &gate_name,
        config.n_routed_experts,
        config.hidden_size,
    )?;
    let shared = load_dense_mlp(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.shared_experts"),
        config.hidden_size,
        config.shared_moe_intermediate(),
    )?;
    let experts = load_owned_experts(ctx, shards, weight_map, config, layout, prefix)?;
    Ok(MoeMlp {
        gate_host,
        gate_device,
        shared,
        experts,
    })
}

fn load_tensor_1d_host(
    shards: &[safetensors::SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<Vec<f32>> {
    load_bf16_tensor_host(shards, weight_map, name, 1)
}

fn load_tensor_2d_host(
    shards: &[safetensors::SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<Vec<f32>> {
    load_bf16_tensor_host(shards, weight_map, name, 2)
}

fn load_bf16_tensor_host(
    shards: &[safetensors::SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    name: &str,
    expected_rank: usize,
) -> Result<Vec<f32>> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    ensure!(
        tensor.dtype() == Dtype::BF16,
        "tensor {name} expected BF16, got {:?}",
        tensor.dtype()
    );
    ensure!(
        shape.len() == expected_rank,
        "tensor {name} expected {expected_rank}D, got {shape:?}"
    );
    let elem_count = shape.iter().product::<usize>();
    let expected_bytes = elem_count * std::mem::size_of::<bf16>();
    ensure!(
        tensor.data().len() == expected_bytes,
        "tensor {name} expected {expected_bytes} bf16 bytes, got {}",
        tensor.data().len()
    );
    Ok(tensor
        .data()
        .as_chunks::<{ std::mem::size_of::<bf16>() }>()
        .0
        .iter()
        .map(|bytes| bf16::from_bits(u16::from_le_bytes(*bytes)).to_f32())
        .collect())
}

fn find_tensor<'a>(
    shards: &'a [safetensors::SafeTensors<'a>],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<safetensors::tensor::TensorView<'a>> {
    if let Some(&idx) = weight_map.get(name) {
        return shards[idx]
            .tensor(name)
            .map_err(|err| anyhow::anyhow!("failed to load tensor {name}: {err}"));
    }
    for shard in shards {
        if let Ok(tensor) = shard.tensor(name) {
            return Ok(tensor);
        }
    }
    bail!("tensor {name} not found in any shard")
}

fn load_owned_experts(
    ctx: &DeviceContext,
    shards: &[safetensors::SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    config: &Config,
    layout: &ExpertParallelLayout,
    prefix: &str,
) -> Result<Vec<ExpertMlp>> {
    let mut experts = Vec::with_capacity(layout.experts_per_rank());
    for global_expert in layout.owned_experts() {
        let dense = load_dense_mlp(
            ctx,
            shards,
            weight_map,
            &format!("{prefix}.experts.{global_expert}"),
            config.hidden_size,
            config.moe_intermediate_size,
        )?;
        experts.push(ExpertMlp {
            global_expert,
            dense,
        });
    }
    ensure!(
        experts.len() == config.n_routed_experts / layout.ep_size(),
        "rank {} loaded {} routed experts, expected {}",
        layout.rank(),
        experts.len(),
        config.n_routed_experts / layout.ep_size()
    );
    Ok(experts)
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

pub(crate) fn dense_mlp_forward(
    ctx: &DeviceContext,
    mlp: &DenseMlp,
    input: &HiddenStates,
) -> Result<HiddenStates> {
    activate(ctx)?;
    let gate_up = ops::gemm(ctx, &mlp.gate_up_proj, input)?;
    let mut act = HiddenStates::zeros(ctx, gate_up.hidden_dim / 2, input.seq_len)?;
    ops::silu_mul_fused_batch_into(ctx, &gate_up, &mut act)?;
    ops::gemm(ctx, &mlp.down_proj, &act)
}

pub(crate) fn dense_mlp_forward_per_token(
    ctx: &DeviceContext,
    mlp: &DenseMlp,
    input: &HiddenStates,
) -> Result<HiddenStates> {
    activate(ctx)?;
    let gate_up = ops::gemm_per_token(ctx, &mlp.gate_up_proj, input)?;
    let mut act = HiddenStates::zeros(ctx, gate_up.hidden_dim / 2, input.seq_len)?;
    ops::silu_mul_fused_batch_into(ctx, &gate_up, &mut act)?;
    ops::gemm_per_token(ctx, &mlp.down_proj, &act)
}

pub(crate) fn dense_mlp_forward_preallocated_into(
    ctx: &DeviceContext,
    mlp: &DenseMlp,
    input: &HiddenStates,
    scratch: &mut DenseMlpForwardScratch,
) -> Result<()> {
    dense_mlp_forward_preallocated_ref_into(ctx, mlp, input.as_ref(), scratch)
}

pub(crate) fn dense_mlp_forward_preallocated_ref_into(
    ctx: &DeviceContext,
    mlp: &DenseMlp,
    input: HiddenStatesRef<'_>,
    scratch: &mut DenseMlpForwardScratch,
) -> Result<()> {
    activate(ctx)?;
    ensure!(
        scratch.gate_up.hidden_dim == mlp.gate_up_proj.rows
            && scratch.gate_up.seq_len == input.seq_len,
        "DeepSeek-V2-Lite MLP preallocated scratch gate_up shape mismatch"
    );
    ensure!(
        scratch.act.hidden_dim == mlp.gate_up_proj.rows / 2 && scratch.act.seq_len == input.seq_len,
        "DeepSeek-V2-Lite MLP preallocated scratch act shape mismatch"
    );
    ensure!(
        scratch.out.hidden_dim == mlp.down_proj.rows && scratch.out.seq_len == input.seq_len,
        "DeepSeek-V2-Lite MLP preallocated scratch out shape mismatch"
    );
    ops::gemm_graphsafe_ref_into_checked(ctx, &mlp.gate_up_proj, input, &mut scratch.gate_up)?;
    ops::silu_mul_fused_batch_into(ctx, &scratch.gate_up, &mut scratch.act)?;
    ops::gemm_graphsafe_into_checked(ctx, &mlp.down_proj, &scratch.act, &mut scratch.out)
}

#[cfg(test)]
mod tests;
