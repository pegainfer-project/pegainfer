//! One decoder layer's weights on the device, read by name from the checkpoint.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use memmap2::Mmap;
use pegainfer_core::weight_loader::SlotId;
use pegainfer_core::weight_loader::StagedWeightLoader;
use pegainfer_core::weight_loader::VecSlotId;
use pegainfer_core::weight_loader::deserialize_shards;
use pegainfer_core::weight_loader::find_tensor;
use pegainfer_core::weight_loader::load_shard_info;
use pegainfer_core::weight_loader::load_tensor_1d;
use pegainfer_core::weight_loader::load_tensor_1d_f32;
use pegainfer_core::weight_loader::mmap_shards;
use pegainfer_kernels::ops::PleBlockGeometry;
use pegainfer_kernels::ops::Qwen4MoeGeometry;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;
use safetensors::Dtype;
use safetensors::SafeTensors;

use crate::config::Config;
use crate::config::LayerKind;
use crate::manifest::schema::Manifest;
use crate::ngram::NgramHasher;

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
    pub indexer_heads: usize,
    pub indexer_head_dim: usize,
    pub ple_embed: usize,
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
            indexer_heads: text.indexer_n_heads,
            indexer_head_dim: text.indexer_head_dim,
            ple_embed: text.ple_embed_dim,
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

    /// Width of the indexer's fused `index_qk_proj` output: its query heads,
    /// then its one key head.
    pub fn indexer_width(&self) -> usize {
        (self.indexer_heads + 1) * self.indexer_head_dim
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
    pub indexer: Indexer,
}

/// QSA's token indexer: one fused projection to its query heads and its key
/// head, and their `(1 + w)` norms.
pub struct Indexer {
    pub qk_proj: DeviceMatrix,
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
    /// The n-gram block, on the one layer `Config::ple_layer` names.
    pub ple: Option<PleBlock>,
    pub attn_hc: HyperConnection,
    pub mixer: Mixer,
    pub mlp_hc: HyperConnection,
    pub moe: MoeBlock,
}

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
    indexer_qk_proj: SlotId,
    indexer_q_norm: VecSlotId,
    indexer_k_norm: VecSlotId,
}

fn record_attention(
    loader: &mut StagedWeightLoader,
    prefix: &str,
    g: &Geometry,
) -> Result<AttentionSlots> {
    // The config pins structure, not widths; the QSA kernels compile these.
    ensure!(
        (g.heads, g.kv_heads, g.head_dim) == (24, 2, 256)
            && (
                g.indexer_heads,
                g.indexer_head_dim,
                g.indexer_compress_ratio
            ) == (4, 128, 4)
            && g.indexer_budget <= 2048,
        "the QSA kernels compile 24 query and 2 K/V heads at 256 and an indexer of 4 heads at \
         128 over blocks of 4 keeping at most 2048 tokens; this checkpoint has {} and {} heads \
         at {}, an indexer of {} heads at {} over blocks of {} and a budget of {}",
        g.heads,
        g.kv_heads,
        g.head_dim,
        g.indexer_heads,
        g.indexer_head_dim,
        g.indexer_compress_ratio,
        g.indexer_budget
    );
    let indexer = format!("{prefix}.indexer");
    Ok(AttentionSlots {
        indexer_qk_proj: loader.matrix(
            &format!("{indexer}.index_qk_proj.weight"),
            g.indexer_width(),
            g.hidden,
        )?,
        indexer_q_norm: loader
            .vector(&format!("{indexer}.q_layernorm.weight"), g.indexer_head_dim)?,
        indexer_k_norm: loader
            .vector(&format!("{indexer}.k_layernorm.weight"), g.indexer_head_dim)?,
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
        indexer: Indexer {
            qk_proj: loader.take(slots.indexer_qk_proj),
            q_norm: loader.take_vec(slots.indexer_q_norm),
            k_norm: loader.take_vec(slots.indexer_k_norm),
        },
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
    let ple = (layer == config.ple_layer)
        .then(|| record_ple(&mut loader, &format!("{prefix}ple"), &g))
        .transpose()?;
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
    let ple = ple
        .map(|slots| {
            take_ple(
                ctx,
                &mut loader,
                &slots,
                &shards,
                &weight_map,
                model_dir,
                config,
                &format!("{prefix}ple"),
            )
        })
        .transpose()?;
    let layer = DecoderLayer {
        ple,
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

/// The n-gram block: the hash that turns tokens into table rows, the table
/// those rows are read from on the host, and the device half's weights.
pub struct PleBlock {
    pub hasher: NgramHasher,
    pub rows: NgramRows,
    pub geometry: PleBlockGeometry,
    /// `[stream, ple_embed]`.
    pub key_proj: DeviceMatrix,
    /// `[hidden, ple_embed]`.
    pub value_proj: DeviceMatrix,
    pub norm_key: DeviceVec,
    pub norm_query: DeviceVec,
    pub norm_conv: DeviceVec,
    /// `[stream, 1, conv_kernel]` flattened.
    pub conv_weight: DeviceVec,
}

/// The n-gram table as the checkpoint ships it: `shards` equal named pieces of
/// `rows_per_shard` rows, read in place from the mapped files one row at a
/// time. The table is far too large for a device (95 GiB here), and a prompt
/// touches `heads` rows per token, so the gather stays on the host.
pub struct NgramRows {
    maps: Vec<Mmap>,
    /// Per shard: the map it lives in and the byte offset of its first row.
    shards: Vec<(usize, usize)>,
    rows_per_shard: usize,
    head_dim: usize,
}

impl NgramRows {
    fn open(model_dir: &Path, prefix: &str, config: &Config) -> Result<Self> {
        let table = &config.ngram;
        let shard_prefix = format!("{prefix}.ple_embedding.ngram_embedding.shard_");
        let (paths, weight_map) = shards_for(model_dir, |name| name.starts_with(&shard_prefix))?;
        let maps = mmap_shards(&paths)?;
        let views = deserialize_shards(&maps)?;
        let mut shards = Vec::with_capacity(table.shards);
        for shard in 0..table.shards {
            let name = format!("{shard_prefix}{shard}.weight");
            let view = find_tensor(&views, &weight_map, &name)?;
            ensure!(
                view.dtype() == Dtype::BF16
                    && view.shape() == [table.rows_per_shard, table.head_dim],
                "{name} is {:?} {:?}, the table needs BF16 [{}, {}]",
                view.dtype(),
                view.shape(),
                table.rows_per_shard,
                table.head_dim
            );
            let start = view.data().as_ptr();
            let map = maps
                .iter()
                .position(|mapped| mapped.as_ptr_range().contains(&start))
                .expect("a tensor view lies in the file it was read from");
            shards.push((map, start as usize - maps[map].as_ptr() as usize));
        }
        Ok(Self {
            maps,
            shards,
            rows_per_shard: table.rows_per_shard,
            head_dim: table.head_dim,
        })
    }

    /// The rows the ids name, each `head_dim` wide, in order.
    pub fn gather(&self, ids: &[i64]) -> Result<Vec<bf16>> {
        let row_bytes = self.head_dim * 2;
        let mut out = Vec::with_capacity(ids.len() * self.head_dim);
        for &id in ids {
            let id = usize::try_from(id).with_context(|| format!("n-gram row {id} is negative"))?;
            let (map, offset) = *self
                .shards
                .get(id / self.rows_per_shard)
                .with_context(|| format!("n-gram row {id} is past the table"))?;
            let start = offset + (id % self.rows_per_shard) * row_bytes;
            out.extend(
                self.maps[map][start..start + row_bytes]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| bf16::from_bits(u16::from_le_bytes(*pair))),
            );
        }
        Ok(out)
    }
}

struct PleSlots {
    key_proj: SlotId,
    value_proj: SlotId,
    norm_key: VecSlotId,
    norm_query: VecSlotId,
    norm_conv: VecSlotId,
}

fn record_ple(loader: &mut StagedWeightLoader, prefix: &str, g: &Geometry) -> Result<PleSlots> {
    let stream = g.stream_width();
    Ok(PleSlots {
        key_proj: loader.matrix(&format!("{prefix}.key_proj.weight"), stream, g.ple_embed)?,
        value_proj: loader.matrix(
            &format!("{prefix}.value_proj.weight"),
            g.hidden,
            g.ple_embed,
        )?,
        norm_key: loader.vector(&format!("{prefix}.norm_key.weight"), stream)?,
        norm_query: loader.vector(&format!("{prefix}.norm_query.weight"), stream)?,
        norm_conv: loader.vector(&format!("{prefix}.norm_conv.weight"), stream)?,
    })
}

/// A small I64 vector read straight from its shard.
fn vector_i64(
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<Vec<i64>> {
    let view = find_tensor(shards, weight_map, name)?;
    ensure!(
        view.dtype() == Dtype::I64,
        "{name} is {:?}, expected I64",
        view.dtype()
    );
    Ok(view
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|bytes| i64::from_le_bytes(*bytes))
        .collect())
}

/// The staged slots plus what the hash needs: the checkpoint's multipliers
/// (the seed that made them is not stored), its per head primes and offsets
/// checked against the ones the config derives, and the mapped table.
#[allow(clippy::too_many_arguments)]
fn take_ple(
    ctx: &DeviceContext,
    loader: &mut StagedWeightLoader,
    slots: &PleSlots,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    model_dir: &Path,
    config: &Config,
    prefix: &str,
) -> Result<PleBlock> {
    let table = &config.ngram;
    let embedding = format!("{prefix}.ple_embedding");
    let multipliers = vector_i64(
        shards,
        weight_map,
        &format!("{embedding}.layer_multipliers"),
    )?;
    let sizes = vector_i64(
        shards,
        weight_map,
        &format!("{embedding}.ngram_heads_vocab_sizes"),
    )?;
    let offsets = vector_i64(
        shards,
        weight_map,
        &format!("{embedding}.ngram_heads_offsets"),
    )?;
    let derived = |values: &[usize]| values.iter().map(|&v| v as i64).collect::<Vec<_>>();
    ensure!(
        sizes == derived(&table.head_vocab_sizes) && offsets == derived(&table.head_offsets),
        "{embedding}: the stored per head primes and offsets differ from the ones the config derives"
    );
    let hasher = NgramHasher::new(
        config.text.eos_token_id,
        multipliers,
        sizes,
        offsets,
        config.text.heads_per_ngram,
    )?;
    let geometry = PleBlockGeometry {
        hidden: config.text.hidden_size,
        hc_count: config.text.hc_count,
        conv_kernel: config.text.ple_conv_kernel_size,
        dilation: config.text.ngram_size,
        rms_eps: config.text.rms_norm_eps,
    };
    geometry.validate()?;
    Ok(PleBlock {
        hasher,
        rows: NgramRows::open(model_dir, prefix, config)?,
        geometry,
        key_proj: loader.take(slots.key_proj),
        value_proj: loader.take(slots.value_proj),
        norm_key: loader.take_vec(slots.norm_key),
        norm_query: loader.take_vec(slots.norm_query),
        norm_conv: loader.take_vec(slots.norm_conv),
        conv_weight: load_tensor_1d(ctx, shards, weight_map, &format!("{prefix}.conv1d.weight"))?,
    })
}
