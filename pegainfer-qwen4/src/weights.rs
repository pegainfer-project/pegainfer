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
    pub vocab: usize,
    pub hidden: usize,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
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
            vocab: text.vocab_size,
            hidden: text.hidden_size,
            hc_count: text.hc_count,
            hc_lowrank: text.hc_lowrank,
            heads: text.num_attention_heads,
            kv_heads: text.num_key_value_heads,
            head_dim: text.head_dim,
            rotary_dim: config.rotary_dim,
            rope_theta: text.rope_parameters.rope_theta as f32,
            indexer_budget: text.indexer_budget,
            indexer_compress_ratio: text.indexer_compress_ratio,
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

    /// Width of the attention output, `o_proj`'s input.
    pub fn attn_width(&self) -> usize {
        self.heads * self.head_dim
    }

    /// Width of the `q_proj` output: `[q | gate]` per head.
    pub fn q_width(&self) -> usize {
        2 * self.attn_width()
    }

    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// The longest prompt whose QSA keeps every causal key: the indexer selects
    /// at most `budget / ratio` complete blocks of `ratio` keys and always keeps
    /// the incomplete tail, so up to this many rows dense causal attention is
    /// the reference and past it the selection starts dropping blocks.
    pub fn dense_attention_rows(&self) -> usize {
        self.indexer_budget + self.indexer_compress_ratio - 1
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

pub struct FullAttention {
    /// `[q | gate]` per head, so twice the attention width tall.
    pub q_proj: DeviceMatrix,
    pub k_proj: DeviceMatrix,
    pub v_proj: DeviceMatrix,
    pub o_proj: DeviceMatrix,
    /// The checkpoint's `w`; the kernels add the 1.
    pub q_norm: DeviceVec,
    pub k_norm: DeviceVec,
}

pub enum Mixer {
    Linear(GatedDeltaNet),
    Full(FullAttention),
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

pub struct DecoderLayer {
    pub attn_hc: HyperConnection,
    pub mixer: Mixer,
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

/// The shards the tensors `keep` selects live in, and the name map into that
/// subset. A single file checkpoint has no index: its one shard is mapped and
/// looked up by name.
fn shards_for(
    model_dir: &Path,
    keep: impl Fn(&str) -> bool,
) -> Result<(Vec<String>, HashMap<String, usize>)> {
    let model_dir = model_dir.to_str().context("model path is not UTF-8")?;
    let (paths, full_map) = load_shard_info(model_dir)?;
    if full_map.is_empty() {
        return Ok((paths, full_map));
    }
    let mut used: Vec<usize> = full_map
        .iter()
        .filter(|(name, _)| keep(name))
        .map(|(_, &shard)| shard)
        .collect();
    used.sort_unstable();
    used.dedup();
    ensure!(
        !used.is_empty(),
        "none of the tensors asked for is in the checkpoint index"
    );
    let subset = full_map
        .iter()
        .filter(|(name, _)| keep(name))
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

struct GdnSlots {
    in_proj_qkv: SlotId,
    in_proj_z: SlotId,
    in_proj_b: SlotId,
    in_proj_a: SlotId,
    dt_bias: VecSlotId,
    out_proj: SlotId,
}

fn record_gdn(loader: &mut StagedWeightLoader, prefix: &str, g: &Geometry) -> Result<GdnSlots> {
    Ok(GdnSlots {
        in_proj_qkv: loader.matrix(
            &format!("{prefix}.in_proj_qkv.weight"),
            g.qkv_width(),
            g.hidden,
        )?,
        in_proj_z: loader.matrix(
            &format!("{prefix}.in_proj_z.weight"),
            g.value_width(),
            g.hidden,
        )?,
        in_proj_b: loader.matrix(
            &format!("{prefix}.in_proj_b.weight"),
            g.value_heads,
            g.hidden,
        )?,
        in_proj_a: loader.matrix(
            &format!("{prefix}.in_proj_a.weight"),
            g.value_heads,
            g.hidden,
        )?,
        dt_bias: loader.vector(&format!("{prefix}.dt_bias"), g.value_heads)?,
        out_proj: loader.matrix(
            &format!("{prefix}.out_proj.weight"),
            g.hidden,
            g.value_width(),
        )?,
    })
}

/// The staged slots plus the three tensors the kernels read outside the bf16
/// matrix path (the conv taps as one vector, `A_log` and the norm weight in f32).
fn take_gdn(
    ctx: &DeviceContext,
    loader: &mut StagedWeightLoader,
    slots: &GdnSlots,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    prefix: &str,
) -> Result<GatedDeltaNet> {
    Ok(GatedDeltaNet {
        in_proj_qkv: loader.take(slots.in_proj_qkv),
        in_proj_z: loader.take(slots.in_proj_z),
        in_proj_b: loader.take(slots.in_proj_b),
        in_proj_a: loader.take(slots.in_proj_a),
        conv_weight: load_tensor_1d(ctx, shards, weight_map, &format!("{prefix}.conv1d.weight"))?,
        dt_bias: loader.take_vec(slots.dt_bias),
        a_log: load_tensor_1d_f32(ctx, shards, weight_map, &format!("{prefix}.A_log"))?,
        norm_weight: load_tensor_1d_f32(ctx, shards, weight_map, &format!("{prefix}.norm.weight"))?,
        out_proj: loader.take(slots.out_proj),
    })
}

struct AttentionSlots {
    q_proj: SlotId,
    k_proj: SlotId,
    v_proj: SlotId,
    o_proj: SlotId,
    q_norm: VecSlotId,
    k_norm: VecSlotId,
}

fn record_attention(
    loader: &mut StagedWeightLoader,
    prefix: &str,
    g: &Geometry,
) -> Result<AttentionSlots> {
    Ok(AttentionSlots {
        q_proj: loader.matrix(&format!("{prefix}.q_proj.weight"), g.q_width(), g.hidden)?,
        k_proj: loader.matrix(&format!("{prefix}.k_proj.weight"), g.kv_width(), g.hidden)?,
        v_proj: loader.matrix(&format!("{prefix}.v_proj.weight"), g.kv_width(), g.hidden)?,
        o_proj: loader.matrix(&format!("{prefix}.o_proj.weight"), g.hidden, g.attn_width())?,
        q_norm: loader.vector(&format!("{prefix}.q_norm.weight"), g.head_dim)?,
        k_norm: loader.vector(&format!("{prefix}.k_norm.weight"), g.head_dim)?,
    })
}

fn take_attention(loader: &mut StagedWeightLoader, slots: &AttentionSlots) -> FullAttention {
    FullAttention {
        q_proj: loader.take(slots.q_proj),
        k_proj: loader.take(slots.k_proj),
        v_proj: loader.take(slots.v_proj),
        o_proj: loader.take(slots.o_proj),
        q_norm: loader.take_vec(slots.q_norm),
        k_norm: loader.take_vec(slots.k_norm),
    }
}

enum MixerSlots {
    Linear(GdnSlots),
    Full(AttentionSlots),
}

struct MoeSlots {
    router: SlotId,
    shared_gate_proj: SlotId,
    shared_up_proj: SlotId,
    shared_down_proj: SlotId,
    shared_gate: SlotId,
    gate_up: SlotId,
    down: SlotId,
}

fn record_moe(loader: &mut StagedWeightLoader, prefix: &str, g: &Geometry) -> Result<MoeSlots> {
    Ok(MoeSlots {
        router: loader.matrix(&format!("{prefix}.gate.weight"), g.experts, g.hidden)?,
        shared_gate_proj: loader.matrix(
            &format!("{prefix}.shared_expert.gate_proj.weight"),
            g.shared_intermediate,
            g.hidden,
        )?,
        shared_up_proj: loader.matrix(
            &format!("{prefix}.shared_expert.up_proj.weight"),
            g.shared_intermediate,
            g.hidden,
        )?,
        shared_down_proj: loader.matrix(
            &format!("{prefix}.shared_expert.down_proj.weight"),
            g.hidden,
            g.shared_intermediate,
        )?,
        shared_gate: loader.matrix(&format!("{prefix}.shared_expert_gate.weight"), 1, g.hidden)?,
        gate_up: loader.expert_stack(
            &format!("{prefix}.experts.gate_up_proj"),
            g.experts,
            2 * g.moe_intermediate,
            g.hidden,
        )?,
        down: loader.expert_stack(
            &format!("{prefix}.experts.down_proj"),
            g.experts,
            g.hidden,
            g.moe_intermediate,
        )?,
    })
}

fn take_moe(loader: &mut StagedWeightLoader, slots: &MoeSlots) -> MoeBlock {
    MoeBlock {
        router: loader.take(slots.router),
        shared_gate_proj: loader.take(slots.shared_gate_proj),
        shared_up_proj: loader.take(slots.shared_up_proj),
        shared_down_proj: loader.take(slots.shared_down_proj),
        shared_gate: loader.take(slots.shared_gate),
        gate_up: loader.take(slots.gate_up),
        down: loader.take(slots.down),
    }
}

/// Load one decoder layer. Only the shards holding that layer are mapped, so a
/// checkpoint cut down to those shards loads too.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "the model line is the next caller")
)]
pub(crate) fn load_layer(
    ctx: &DeviceContext,
    model_dir: &Path,
    config: &Config,
    layer: usize,
) -> Result<DecoderLayer> {
    let kind = *config.layer_kinds.get(layer).with_context(|| {
        format!(
            "layer {layer} is past the model's {} layers",
            config.layer_kinds.len()
        )
    })?;
    ensure!(
        layer != config.ple_layer,
        "layer {layer} carries the PLE block, which nothing loads yet"
    );
    let g = Geometry::from_config(config);
    // The routing kernels are compiled for one expert count and top-k, and the
    // hyper-connection mix folds `1 / hc_count` into a multiply that is exact
    // only for a power of two; refuse either before anything is mapped.
    g.moe().validate()?;
    ensure!(
        g.hc_count.is_power_of_two(),
        "hc_count {} is not a power of two, which the hyper-connection mix kernels need",
        g.hc_count
    );
    let prefix = format!("{TEXT_PREFIX}layers.{layer}.");
    let (shard_paths, weight_map) = shards_for(model_dir, |name| name.starts_with(&prefix))?;
    let mmaps = mmap_shards(&shard_paths)?;
    let shards = deserialize_shards(&mmaps)?;
    check_layer_contract(config, &prefix, &shards, &weight_map)?;

    let mut loader = StagedWeightLoader::new(ctx, &shards, &weight_map)?;
    let attn_hc = record_hc(
        &mut loader,
        &format!("{prefix}attn_hyper_connection"),
        &g,
        true,
    )?;
    let mlp_hc = record_hc(
        &mut loader,
        &format!("{prefix}mlp_hyper_connection"),
        &g,
        true,
    )?;
    let mixer = match kind {
        LayerKind::LinearAttention => MixerSlots::Linear(record_gdn(
            &mut loader,
            &format!("{prefix}linear_attn"),
            &g,
        )?),
        LayerKind::FullAttention => MixerSlots::Full(record_attention(
            &mut loader,
            &format!("{prefix}self_attn"),
            &g,
        )?),
    };
    let moe = record_moe(&mut loader, &format!("{prefix}mlp"), &g)?;
    loader.finish()?;

    let mixer = match mixer {
        MixerSlots::Linear(slots) => Mixer::Linear(take_gdn(
            ctx,
            &mut loader,
            &slots,
            &shards,
            &weight_map,
            &format!("{prefix}linear_attn"),
        )?),
        MixerSlots::Full(slots) => Mixer::Full(take_attention(&mut loader, &slots)),
    };
    let layer = DecoderLayer {
        attn_hc: take_hc(&mut loader, &attn_hc),
        mixer,
        mlp_hc: take_hc(&mut loader, &mlp_hc),
        moe: take_moe(&mut loader, &moe),
    };
    ctx.sync()?;
    Ok(layer)
}

/// What surrounds the layers: the token embedding, the top-level mixer that
/// reads the stream out (there is no final norm) and the untied head.
pub struct Bookends {
    pub embed: DeviceMatrix,
    pub mixer: HyperConnection,
    pub lm_head: DeviceMatrix,
}

const MIXER_PREFIX: &str = "model.language_model.hyper_connection_mixer";
const EMBED_NAME: &str = "model.language_model.embed_tokens.weight";
const HEAD_NAME: &str = "lm_head.weight";

#[cfg_attr(
    not(test),
    allow(dead_code, reason = "the model line is the next caller")
)]
pub(crate) fn load_bookends(
    ctx: &DeviceContext,
    model_dir: &Path,
    config: &Config,
) -> Result<Bookends> {
    let g = Geometry::from_config(config);
    let (shard_paths, weight_map) = shards_for(model_dir, |name| {
        name == EMBED_NAME || name == HEAD_NAME || name.starts_with(MIXER_PREFIX)
    })?;
    let mmaps = mmap_shards(&shard_paths)?;
    let shards = deserialize_shards(&mmaps)?;
    for prefix in [EMBED_NAME, HEAD_NAME, MIXER_PREFIX] {
        check_layer_contract(config, prefix, &shards, &weight_map)?;
    }
    let mut loader = StagedWeightLoader::new(ctx, &shards, &weight_map)?;
    let embed = loader.matrix(EMBED_NAME, g.vocab, g.hidden)?;
    let mixer = record_hc(&mut loader, MIXER_PREFIX, &g, false)?;
    let lm_head = loader.matrix(HEAD_NAME, g.vocab, g.hidden)?;
    loader.finish()?;
    let bookends = Bookends {
        embed: loader.take(embed),
        mixer: take_hc(&mut loader, &mixer),
        lm_head: loader.take(lm_head),
    };
    ctx.sync()?;
    Ok(bookends)
}
