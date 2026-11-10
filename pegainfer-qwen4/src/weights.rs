//! One decoder layer's weights on the device, read by name from the checkpoint.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use pegainfer_core::weight_loader::SlotId;
use pegainfer_core::weight_loader::StagedWeightLoader;
use pegainfer_core::weight_loader::VecSlotId;
use pegainfer_core::weight_loader::deserialize_shards;
use pegainfer_core::weight_loader::find_tensor;
use pegainfer_core::weight_loader::load_shard_info;
use pegainfer_core::weight_loader::load_tensor_1d;
use pegainfer_core::weight_loader::load_tensor_1d_f32;
use pegainfer_core::weight_loader::mmap_shards;
use pegainfer_kernels::ops::Qwen4MoeGeometry;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;
use safetensors::SafeTensors;

use crate::config::Config;
use crate::config::LayerKind;
use crate::manifest::schema::Manifest;

const TEXT_PREFIX: &str = "model.language_model.";

/// The widths one layer's forward needs, taken from the validated config.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub hidden: usize,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
    pub experts: usize,
    pub top_k: usize,
    pub moe_intermediate: usize,
    pub shared_intermediate: usize,
    pub rms_eps: f32,
    pub norm_topk_prob: bool,
}

impl Geometry {
    pub(crate) fn from_config(config: &Config) -> Self {
        let text = &config.text;
        Self {
            hidden: text.hidden_size,
            hc_count: text.hc_count,
            hc_lowrank: text.hc_lowrank,
            key_heads: text.linear_num_key_heads,
            value_heads: text.linear_num_value_heads,
            key_dim: text.linear_key_head_dim,
            value_dim: text.linear_value_head_dim,
            conv_kernel: text.linear_conv_kernel_dim,
            experts: text.num_experts,
            top_k: text.num_experts_per_tok,
            moe_intermediate: text.moe_intermediate_size,
            shared_intermediate: text.shared_expert_intermediate_size,
            rms_eps: text.rms_norm_eps,
            norm_topk_prob: text.norm_topk_prob,
        }
    }

    /// Width of the hyper-connection residual stream.
    pub fn stream_width(&self) -> usize {
        self.hc_count * self.hidden
    }

    /// Width of the GDN `in_proj_qkv` output: `[q | k | v]`.
    pub fn qkv_width(&self) -> usize {
        2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim
    }

    pub fn value_width(&self) -> usize {
        self.value_heads * self.value_dim
    }

    /// The MoE block as the routing kernels see it.
    pub fn moe(&self) -> Qwen4MoeGeometry {
        Qwen4MoeGeometry {
            hidden: self.hidden,
            experts: self.experts,
            top_k: self.top_k,
            shared_intermediate: self.shared_intermediate,
        }
    }
}

pub struct HyperConnection {
    pub norm: DeviceVec,
    pub down: DeviceMatrix,
    pub up: DeviceMatrix,
    /// Present on the two per-layer modules, absent on the top-level mixer.
    pub inject: Option<DeviceMatrix>,
}

pub struct GatedDeltaNet {
    pub in_proj_qkv: DeviceMatrix,
    pub in_proj_z: DeviceMatrix,
    pub in_proj_b: DeviceMatrix,
    pub in_proj_a: DeviceMatrix,
    /// `[qkv_width, 1, conv_kernel]` flattened.
    pub conv_weight: DeviceVec,
    pub dt_bias: DeviceVec,
    pub a_log: CudaSlice<f32>,
    pub norm_weight: CudaSlice<f32>,
    pub out_proj: DeviceMatrix,
}

pub struct MoeBlock {
    pub router: DeviceMatrix,
    pub shared_gate_proj: DeviceMatrix,
    pub shared_up_proj: DeviceMatrix,
    pub shared_down_proj: DeviceMatrix,
    pub shared_gate: DeviceMatrix,
    /// `[experts * 2 * moe_intermediate, hidden]`, expert `e` at rows `e * 2 * moe_intermediate..`.
    pub gate_up: DeviceMatrix,
    /// `[experts * hidden, moe_intermediate]`, expert `e` at rows `e * hidden..`.
    pub down: DeviceMatrix,
}

pub struct LinearLayer {
    pub attn_hc: HyperConnection,
    pub gdn: GatedDeltaNet,
    pub mlp_hc: HyperConnection,
    pub moe: MoeBlock,
}

#[cfg_attr(
    not(test),
    allow(dead_code, reason = "the model line is the next caller")
)]
pub(crate) fn load_config(model_dir: &Path) -> Result<Config> {
    let path = model_dir.join("config.json");
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    Config::from_json(&serde_json::from_slice(&bytes)?)
}

/// The shards this layer lives in, and the name map into that subset. A single
/// file checkpoint has no index: its one shard is mapped and looked up by name.
fn layer_shards(model_dir: &Path, prefix: &str) -> Result<(Vec<String>, HashMap<String, usize>)> {
    let model_dir = model_dir.to_str().context("model path is not UTF-8")?;
    let (paths, full_map) = load_shard_info(model_dir)?;
    if full_map.is_empty() {
        return Ok((paths, full_map));
    }
    let mut used: Vec<usize> = full_map
        .iter()
        .filter(|(name, _)| name.starts_with(prefix))
        .map(|(_, &shard)| shard)
        .collect();
    used.sort_unstable();
    used.dedup();
    ensure!(
        !used.is_empty(),
        "no tensor under {prefix} in the checkpoint index"
    );
    let subset = full_map
        .iter()
        .filter(|(name, _)| name.starts_with(prefix))
        .map(|(name, shard)| {
            (
                name.clone(),
                used.binary_search(shard).expect("shard listed"),
            )
        })
        .collect();
    Ok((
        used.into_iter().map(|shard| paths[shard].clone()).collect(),
        subset,
    ))
}

fn check_layer_contract(
    config: &Config,
    prefix: &str,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
) -> Result<()> {
    let manifest = Manifest::from_config(config);
    let expected = manifest
        .expected_tensors()
        .range(prefix.to_string()..)
        .take_while(|(name, _)| name.starts_with(prefix));
    for (name, (shape, dtype)) in expected {
        let view = find_tensor(shards, weight_map, name)?;
        ensure!(
            view.dtype() == *dtype,
            "{name}: dtype {:?}, the contract expects {dtype:?}",
            view.dtype()
        );
        ensure!(
            view.shape() == shape.dims().as_slice(),
            "{name}: shape {:?}, the contract expects {shape}",
            view.shape()
        );
    }
    Ok(())
}

struct HcSlots {
    norm: VecSlotId,
    down: SlotId,
    up: SlotId,
    inject: Option<SlotId>,
}

fn record_hc(
    loader: &mut StagedWeightLoader,
    prefix: &str,
    g: &Geometry,
    inject: bool,
) -> Result<HcSlots> {
    let stream = g.stream_width();
    Ok(HcSlots {
        norm: loader.vector(&format!("{prefix}.hc_norm.weight"), stream)?,
        down: loader.matrix(
            &format!("{prefix}.input_mix_weight_down.weight"),
            g.hc_lowrank,
            stream,
        )?,
        up: loader.matrix(
            &format!("{prefix}.input_mix_weight_up.weight"),
            stream,
            g.hc_lowrank,
        )?,
        inject: inject
            .then(|| {
                loader.matrix(
                    &format!("{prefix}.block_inject_weight.weight"),
                    g.hc_count,
                    stream,
                )
            })
            .transpose()?,
    })
}

fn take_hc(loader: &mut StagedWeightLoader, slots: &HcSlots) -> HyperConnection {
    HyperConnection {
        norm: loader.take_vec(slots.norm),
        down: loader.take(slots.down),
        up: loader.take(slots.up),
        inject: slots.inject.map(|slot| loader.take(slot)),
    }
}

/// Load one `linear_attention` decoder layer. Only the shards holding that layer
/// are mapped, so a checkpoint cut down to those shards loads too.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "the model line is the next caller")
)]
pub(crate) fn load_linear_layer(
    ctx: &DeviceContext,
    model_dir: &Path,
    config: &Config,
    layer: usize,
) -> Result<LinearLayer> {
    ensure!(
        config.layer_kinds.get(layer) == Some(&LayerKind::LinearAttention),
        "layer {layer} is not a linear attention layer"
    );
    let g = Geometry::from_config(config);
    // The routing kernels are compiled for one expert count and top-k; refuse
    // another before anything is mapped.
    g.moe().validate()?;
    let prefix = format!("{TEXT_PREFIX}layers.{layer}.");
    let (shard_paths, weight_map) = layer_shards(model_dir, &prefix)?;
    let mmaps = mmap_shards(&shard_paths)?;
    let shards = deserialize_shards(&mmaps)?;
    check_layer_contract(config, &prefix, &shards, &weight_map)?;
    let name = |suffix: &str| format!("{prefix}{suffix}");

    let mut loader = StagedWeightLoader::new(ctx, &shards, &weight_map)?;
    let attn_hc = record_hc(&mut loader, &name("attn_hyper_connection"), &g, true)?;
    let mlp_hc = record_hc(&mut loader, &name("mlp_hyper_connection"), &g, true)?;
    let in_proj_qkv = loader.matrix(
        &name("linear_attn.in_proj_qkv.weight"),
        g.qkv_width(),
        g.hidden,
    )?;
    let in_proj_z = loader.matrix(
        &name("linear_attn.in_proj_z.weight"),
        g.value_width(),
        g.hidden,
    )?;
    let in_proj_b = loader.matrix(
        &name("linear_attn.in_proj_b.weight"),
        g.value_heads,
        g.hidden,
    )?;
    let in_proj_a = loader.matrix(
        &name("linear_attn.in_proj_a.weight"),
        g.value_heads,
        g.hidden,
    )?;
    let dt_bias = loader.vector(&name("linear_attn.dt_bias"), g.value_heads)?;
    let out_proj = loader.matrix(
        &name("linear_attn.out_proj.weight"),
        g.hidden,
        g.value_width(),
    )?;
    let router = loader.matrix(&name("mlp.gate.weight"), g.experts, g.hidden)?;
    let shared_gate_proj = loader.matrix(
        &name("mlp.shared_expert.gate_proj.weight"),
        g.shared_intermediate,
        g.hidden,
    )?;
    let shared_up_proj = loader.matrix(
        &name("mlp.shared_expert.up_proj.weight"),
        g.shared_intermediate,
        g.hidden,
    )?;
    let shared_down_proj = loader.matrix(
        &name("mlp.shared_expert.down_proj.weight"),
        g.hidden,
        g.shared_intermediate,
    )?;
    let shared_gate = loader.matrix(&name("mlp.shared_expert_gate.weight"), 1, g.hidden)?;
    let gate_up = loader.expert_stack(
        &name("mlp.experts.gate_up_proj"),
        g.experts,
        2 * g.moe_intermediate,
        g.hidden,
    )?;
    let down = loader.expert_stack(
        &name("mlp.experts.down_proj"),
        g.experts,
        g.hidden,
        g.moe_intermediate,
    )?;
    loader.finish()?;

    let attn_hc = take_hc(&mut loader, &attn_hc);
    let mlp_hc = take_hc(&mut loader, &mlp_hc);
    let gdn = GatedDeltaNet {
        in_proj_qkv: loader.take(in_proj_qkv),
        in_proj_z: loader.take(in_proj_z),
        in_proj_b: loader.take(in_proj_b),
        in_proj_a: loader.take(in_proj_a),
        conv_weight: load_tensor_1d(
            ctx,
            &shards,
            &weight_map,
            &name("linear_attn.conv1d.weight"),
        )?,
        dt_bias: loader.take_vec(dt_bias),
        a_log: load_tensor_1d_f32(ctx, &shards, &weight_map, &name("linear_attn.A_log"))?,
        norm_weight: load_tensor_1d_f32(
            ctx,
            &shards,
            &weight_map,
            &name("linear_attn.norm.weight"),
        )?,
        out_proj: loader.take(out_proj),
    };
    let moe = MoeBlock {
        router: loader.take(router),
        shared_gate_proj: loader.take(shared_gate_proj),
        shared_up_proj: loader.take(shared_up_proj),
        shared_down_proj: loader.take(shared_down_proj),
        shared_gate: loader.take(shared_gate),
        gate_up: loader.take(gate_up),
        down: loader.take(down),
    };
    ctx.sync()?;
    Ok(LinearLayer {
        attn_hc,
        gdn,
        mlp_hc,
        moe,
    })
}
