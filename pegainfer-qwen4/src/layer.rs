//! One decoder layer, stage by stage: the n-gram block on the one layer that
//! carries it, the attention hyper-connection mix, the token mixer (Gated
//! DeltaNet on the qwen35 kernels, or gated full attention on the qwen35 hd256
//! prefill), the scatter back into the stream, then the same around the MoE
//! block. Prefill walks a whole prompt from a fresh state; decode advances the
//! state by one token.

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtrMut;
use half::bf16;
use pegainfer_core::rope::RopeTableSpec;
use pegainfer_core::rope::precompute_rope;
use pegainfer_kernels::ops::GatedNormActivation;
use pegainfer_kernels::ops::GdrChunkwiseScratch35;
use pegainfer_kernels::ops::HcMixScratch;
use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::ops::add_batch_into;
use pegainfer_kernels::ops::attention_prefill_hd256_into;
use pegainfer_kernels::ops::conv1d_decode_batch_into;
use pegainfer_kernels::ops::conv1d_prefill_batch_into;
use pegainfer_kernels::ops::copy_hidden_token_range_into;
use pegainfer_kernels::ops::gated_delta_rule_decode_batch_into;
use pegainfer_kernels::ops::gated_delta_rule_prefill_chunkwise_into;
use pegainfer_kernels::ops::gemm_into_checked;
use pegainfer_kernels::ops::hc_mix_into;
use pegainfer_kernels::ops::hc_scatter_add_into;
use pegainfer_kernels::ops::ple_block_into;
use pegainfer_kernels::ops::rms_norm_gated_batch_into;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

use crate::moe::MoeTrace;
use crate::moe::moe_prefill;
use crate::weights::DecoderLayer;
use crate::weights::FullAttention;
use crate::weights::GatedDeltaNet;
use crate::weights::Geometry;
use crate::weights::HyperConnection;
use crate::weights::Mixer;
use crate::weights::PleBlock;

/// Tokens per page of the attention layers' K/V, the size every FlashInfer
/// caller in the tree uses.
const KV_PAGE: usize = 16;

/// The attention layers' cos/sin tables, `[positions * rotary_dim]` half split,
/// frequencies spaced over the rotated dims as the reference spaces them.
pub struct Rope {
    pub cos: DeviceVec,
    pub sin: DeviceVec,
}

impl Rope {
    pub fn new(ctx: &DeviceContext, g: &Geometry, positions: usize) -> Result<Self> {
        let (cos, sin) = precompute_rope(
            ctx,
            &RopeTableSpec {
                rotary_dim: g.rotary_dim,
                frequency_dim: g.rotary_dim,
                max_seq_len: positions,
                theta: g.rope_theta,
            },
        )?;
        Ok(Self { cos, sin })
    }
}

/// What a Gated DeltaNet layer carries from one token to the next.
pub struct GdnState {
    /// The previous `conv_kernel - 1` inputs per channel, oldest first.
    pub conv: DeviceVec,
    /// `[value_heads, key_dim, value_dim]` f32.
    pub recurrent: CudaSlice<f32>,
}

/// One sequence's K/V in the one layer paged format the hd256 prefill reads.
pub struct KvScratch {
    pub pool: CudaSlice<bf16>,
    pub layout: PagedKvLayout,
    pub capacity: usize,
    /// Tokens written so far; the next token lands at this position.
    pub len: usize,
}

/// The token mixer's state, zeroed, sized for one sequence.
pub enum MixerState {
    Linear(GdnState),
    Full(KvScratch),
}

impl MixerState {
    pub fn new(ctx: &DeviceContext, g: &Geometry, mixer: &Mixer, capacity: usize) -> Result<Self> {
        Ok(match mixer {
            Mixer::Linear(_) => Self::Linear(GdnState {
                conv: DeviceVec::zeros(ctx, g.qkv_width() * (g.conv_kernel - 1))?,
                recurrent: ctx
                    .stream
                    .alloc_zeros(g.value_heads * g.key_dim * g.value_dim)?,
            }),
            Mixer::Full(_) => {
                let layout = PagedKvLayout::new(1, g.kv_heads, g.head_dim, KV_PAGE);
                let pages = capacity.div_ceil(KV_PAGE).max(1);
                Self::Full(KvScratch {
                    pool: ctx.stream.alloc_zeros(pages * layout.page_stride)?,
                    layout,
                    capacity,
                    len: 0,
                })
            }
        })
    }
}

/// What the n-gram block carries from one token to the next: the tokens its
/// hash still looks back at, and the block inputs of the rows its dilated conv
/// still reads. The conv kernel takes a whole sequence with a zero state in
/// front of it, so a decode step recomputes those rows from their inputs and
/// keeps the last output row; the block is a per row map before the conv, so
/// the recomputed rows are the rows the conv saw.
pub struct PleState {
    /// The last `context` tokens, oldest first.
    pub history: Vec<i64>,
    /// The last `conv_state` rows' stream, key and value inputs, oldest first;
    /// `rows` of them are live.
    pub stream: HiddenStates,
    pub key: HiddenStates,
    pub value: HiddenStates,
    pub rows: usize,
}

/// One layer's state for one sequence.
pub struct LayerState {
    pub mixer: MixerState,
    pub ple: Option<PleState>,
}

impl LayerState {
    pub fn new(
        ctx: &DeviceContext,
        g: &Geometry,
        layer: &DecoderLayer,
        capacity: usize,
    ) -> Result<Self> {
        let ple = layer
            .ple
            .as_ref()
            .map(|ple| -> Result<PleState> {
                let rows = ple.geometry.conv_state();
                Ok(PleState {
                    history: Vec::new(),
                    stream: HiddenStates::zeros(ctx, g.stream_width(), rows)?,
                    key: HiddenStates::zeros(ctx, g.stream_width(), rows)?,
                    value: HiddenStates::zeros(ctx, g.hidden, rows)?,
                    rows: 0,
                })
            })
            .transpose()?;
        Ok(Self {
            mixer: MixerState::new(ctx, g, &layer.mixer, capacity)?,
            ple,
        })
    }
}

pub struct GdnTrace {
    pub qkv: HiddenStates,
    pub z: HiddenStates,
    pub b: HiddenStates,
    pub a: HiddenStates,
    pub core: HiddenStates,
    pub normed: HiddenStates,
}

pub struct AttentionTrace {
    pub q_proj: HiddenStates,
    pub k_proj: HiddenStates,
    pub v_proj: HiddenStates,
    /// Q after the `(1 + w)` norm and the partial RoPE, `[rows, heads * head_dim]`.
    pub q_prepped: HiddenStates,
    /// The attention output times the sigmoid gate, `o_proj`'s input.
    pub gated: HiddenStates,
}

pub enum MixerTrace {
    Linear(GdnTrace),
    Full(AttentionTrace),
}

pub struct PleTrace {
    /// The gathered n-gram rows, `[rows, ple_embed]`.
    pub embeds: HiddenStates,
    pub key: HiddenStates,
    pub value: HiddenStates,
    /// The block's output, added onto the stream.
    pub out: HiddenStates,
    /// The stream after the addition, what the attention mix reads.
    pub stream: HiddenStates,
}

/// Every stage of one layer forward, kept for the reference gate.
pub struct LayerTrace {
    pub ple: Option<PleTrace>,
    pub attn_mixed: HiddenStates,
    pub attn_inject: HiddenStates,
    pub mixer: MixerTrace,
    pub attn_out: HiddenStates,
    pub stream_after_attn: HiddenStates,
    pub mlp_mixed: HiddenStates,
    pub mlp_inject: HiddenStates,
    pub moe: MoeTrace,
    pub stream_out: HiddenStates,
}

fn mix(
    ctx: &DeviceContext,
    g: &Geometry,
    hc: &HyperConnection,
    stream: &HiddenStates,
) -> Result<(HiddenStates, HiddenStates)> {
    let rows = stream.seq_len;
    let mut scratch = HcMixScratch::new(ctx, rows, g.hidden, g.hc_count, g.hc_lowrank, true)?;
    let mut mixed = HiddenStates::zeros(ctx, g.hidden, rows)?;
    let mut inject = HiddenStates::zeros(ctx, g.hc_count, rows)?;
    hc_mix_into(
        ctx,
        &hc.norm,
        &hc.down,
        &hc.up,
        hc.inject.as_ref(),
        stream,
        g.hc_count,
        g.rms_eps,
        &mut scratch,
        &mut mixed,
        Some(&mut inject),
    )?;
    Ok((mixed, inject))
}

/// The n-gram rows of `ids` gathered on the host and their two projections.
fn ple_project(
    ctx: &DeviceContext,
    g: &Geometry,
    ple: &PleBlock,
    row_ids: &[i64],
    rows: usize,
) -> Result<(HiddenStates, HiddenStates, HiddenStates)> {
    let gathered = ple.rows.gather(row_ids)?;
    let embeds = HiddenStates::from_host(ctx, &gathered, g.ple_embed, rows)?;
    let mut key = HiddenStates::zeros(ctx, g.stream_width(), rows)?;
    gemm_into_checked(ctx, &ple.key_proj, &embeds, &mut key)?;
    let mut value = HiddenStates::zeros(ctx, g.hidden, rows)?;
    gemm_into_checked(ctx, &ple.value_proj, &embeds, &mut value)?;
    Ok((embeds, key, value))
}

/// The block over `rows` rows whose inputs are given, with the zero conv state
/// in front; returns the block output.
fn ple_block(
    ctx: &DeviceContext,
    g: &Geometry,
    ple: &PleBlock,
    stream: &HiddenStates,
    key: &HiddenStates,
    value: &HiddenStates,
    rows: usize,
) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, g.stream_width(), rows)?;
    let mut workspace: CudaSlice<bf16> =
        ctx.stream.alloc_zeros(ple.geometry.workspace_len(rows))?;
    ple_block_into(
        ctx,
        &ple.geometry,
        &stream.data,
        &key.data,
        &value.data,
        &ple.norm_key.data,
        &ple.norm_query.data,
        &ple.norm_conv.data,
        &ple.conv_weight.data,
        rows,
        &mut out.data,
        &mut workspace,
    )?;
    Ok(out)
}

/// Keep the last `conv_state` rows of the block inputs and the last `context`
/// tokens for the next step.
fn ple_remember(
    ctx: &DeviceContext,
    ple: &PleBlock,
    state: &mut PleState,
    ids: &[i64],
    stream: &HiddenStates,
    key: &HiddenStates,
    value: &HiddenStates,
) -> Result<()> {
    let rows = stream.seq_len;
    let keep = rows.min(ple.geometry.conv_state());
    copy_hidden_token_range_into(ctx, stream, rows - keep, &mut state.stream, 0, keep)?;
    copy_hidden_token_range_into(ctx, key, rows - keep, &mut state.key, 0, keep)?;
    copy_hidden_token_range_into(ctx, value, rows - keep, &mut state.value, 0, keep)?;
    state.rows = keep;
    let context = ple.hasher.context();
    state.history = ids[ids.len().saturating_sub(context)..].to_vec();
    Ok(())
}

/// The n-gram block over a prompt from a fresh state; returns its stages and
/// the stream with the block added.
fn ple_prefill(
    ctx: &DeviceContext,
    g: &Geometry,
    ple: &PleBlock,
    state: &mut PleState,
    ids: &[i64],
    stream_in: &HiddenStates,
) -> Result<PleTrace> {
    let rows = stream_in.seq_len;
    let (embeds, key, value) = ple_project(ctx, g, ple, &ple.hasher.ids(ids), rows)?;
    let out = ple_block(ctx, g, ple, stream_in, &key, &value, rows)?;
    let mut stream = HiddenStates::zeros(ctx, g.stream_width(), rows)?;
    add_batch_into(ctx, stream_in, &out, &mut stream)?;
    ple_remember(ctx, ple, state, ids, stream_in, &key, &value)?;
    Ok(PleTrace {
        embeds,
        key,
        value,
        out,
        stream,
    })
}

/// The n-gram block for one token on the carried state: the hash sees the
/// remembered tokens, the block runs over the remembered rows plus this one
/// and keeps the last output row.
fn ple_decode(
    ctx: &DeviceContext,
    g: &Geometry,
    ple: &PleBlock,
    state: &mut PleState,
    id: i64,
    stream_in: &HiddenStates,
) -> Result<PleTrace> {
    let mut ids = state.history.clone();
    ids.push(id);
    let heads = ple.hasher.heads();
    let row_ids = ple.hasher.ids(&ids);
    let (embeds, key, value) = ple_project(ctx, g, ple, &row_ids[row_ids.len() - heads..], 1)?;
    let rows = state.rows + 1;
    let mut all_stream = HiddenStates::zeros(ctx, g.stream_width(), rows)?;
    let mut all_key = HiddenStates::zeros(ctx, g.stream_width(), rows)?;
    let mut all_value = HiddenStates::zeros(ctx, g.hidden, rows)?;
    copy_hidden_token_range_into(ctx, &state.stream, 0, &mut all_stream, 0, state.rows)?;
    copy_hidden_token_range_into(ctx, &state.key, 0, &mut all_key, 0, state.rows)?;
    copy_hidden_token_range_into(ctx, &state.value, 0, &mut all_value, 0, state.rows)?;
    copy_hidden_token_range_into(ctx, stream_in, 0, &mut all_stream, state.rows, 1)?;
    copy_hidden_token_range_into(ctx, &key, 0, &mut all_key, state.rows, 1)?;
    copy_hidden_token_range_into(ctx, &value, 0, &mut all_value, state.rows, 1)?;
    let all_out = ple_block(ctx, g, ple, &all_stream, &all_key, &all_value, rows)?;
    let mut out = HiddenStates::zeros(ctx, g.stream_width(), 1)?;
    copy_hidden_token_range_into(ctx, &all_out, rows - 1, &mut out, 0, 1)?;
    let mut stream = HiddenStates::zeros(ctx, g.stream_width(), 1)?;
    add_batch_into(ctx, stream_in, &out, &mut stream)?;
    ple_remember(ctx, ple, state, &ids, &all_stream, &all_key, &all_value)?;
    Ok(PleTrace {
        embeds,
        key,
        value,
        out,
        stream,
    })
}

/// The four GDN projections of one block input.
fn gdn_project(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &GatedDeltaNet,
    mixed: &HiddenStates,
) -> Result<(HiddenStates, HiddenStates, HiddenStates, HiddenStates)> {
    let rows = mixed.seq_len;
    let mut qkv = HiddenStates::zeros(ctx, g.qkv_width(), rows)?;
    gemm_into_checked(ctx, &w.in_proj_qkv, mixed, &mut qkv)?;
    let mut z_proj = HiddenStates::zeros(ctx, g.value_width(), rows)?;
    gemm_into_checked(ctx, &w.in_proj_z, mixed, &mut z_proj)?;
    let mut b_proj = HiddenStates::zeros(ctx, g.value_heads, rows)?;
    gemm_into_checked(ctx, &w.in_proj_b, mixed, &mut b_proj)?;
    let mut a_proj = HiddenStates::zeros(ctx, g.value_heads, rows)?;
    gemm_into_checked(ctx, &w.in_proj_a, mixed, &mut a_proj)?;
    Ok((qkv, z_proj, b_proj, a_proj))
}

/// The gated norm and `out_proj` that close the GDN block.
fn gdn_finish(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &GatedDeltaNet,
    core: &HiddenStates,
    z_proj: &HiddenStates,
) -> Result<(HiddenStates, HiddenStates)> {
    let rows = core.seq_len;
    let mut normed = HiddenStates::zeros(ctx, g.value_width(), rows)?;
    rms_norm_gated_batch_into(
        ctx,
        core,
        &w.norm_weight,
        z_proj,
        &mut normed,
        g.value_heads,
        g.value_dim,
        g.rms_eps,
        GatedNormActivation::Sigmoid,
    );
    let mut out = HiddenStates::zeros(ctx, g.hidden, rows)?;
    gemm_into_checked(ctx, &w.out_proj, &normed, &mut out)?;
    Ok((normed, out))
}

/// The Gated DeltaNet block over a prompt, reading and leaving the state;
/// returns its stages and the `out_proj` output.
fn gdn_prefill(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &GatedDeltaNet,
    state: &mut GdnState,
    mixed: &HiddenStates,
) -> Result<(GdnTrace, HiddenStates)> {
    let rows = mixed.seq_len;
    let (qkv, z_proj, b_proj, a_proj) = gdn_project(ctx, g, w, mixed)?;
    let mut qkv_conv = HiddenStates::zeros(ctx, g.qkv_width(), rows)?;
    conv1d_prefill_batch_into(
        ctx,
        &qkv,
        &w.conv_weight,
        &mut state.conv,
        &mut qkv_conv,
        g.conv_kernel,
    );
    let mut scratch =
        GdrChunkwiseScratch35::from_dims(ctx, g.value_heads, g.key_dim, g.value_dim, rows)?;
    let mut core = HiddenStates::zeros(ctx, g.value_width(), rows)?;
    gated_delta_rule_prefill_chunkwise_into(
        ctx,
        &qkv_conv,
        &b_proj,
        &a_proj,
        &w.dt_bias,
        &w.a_log,
        &mut state.recurrent,
        &mut scratch,
        &mut core,
        g.key_heads,
        g.value_heads,
        g.key_dim,
        g.value_dim,
    )?;
    let (normed, out) = gdn_finish(ctx, g, w, &core, &z_proj)?;
    Ok((
        GdnTrace {
            qkv,
            z: z_proj,
            b: b_proj,
            a: a_proj,
            core,
            normed,
        },
        out,
    ))
}

/// The Gated DeltaNet block for one token on the carried state. The decode
/// kernels take one request per row through a pointer per request; this is
/// the one request form the step will batch.
fn gdn_decode(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &GatedDeltaNet,
    state: &mut GdnState,
    mixed: &HiddenStates,
) -> Result<(GdnTrace, HiddenStates)> {
    let (qkv, z_proj, b_proj, a_proj) = gdn_project(ctx, g, w, mixed)?;
    let conv_ptrs: CudaSlice<u64> = {
        let (ptr, _guard) = state.conv.data.device_ptr_mut(&ctx.stream);
        ctx.stream.clone_htod(&[ptr])?
    };
    let recurrent_ptrs: CudaSlice<u64> = {
        let (ptr, _guard) = state.recurrent.device_ptr_mut(&ctx.stream);
        ctx.stream.clone_htod(&[ptr])?
    };
    let mut qkv_conv = HiddenStates::zeros(ctx, g.qkv_width(), 1)?;
    conv1d_decode_batch_into(
        ctx,
        &qkv,
        &w.conv_weight,
        &conv_ptrs,
        &mut qkv_conv,
        g.conv_kernel,
    );
    let mut core = HiddenStates::zeros(ctx, g.value_width(), 1)?;
    gated_delta_rule_decode_batch_into(
        ctx,
        &qkv_conv,
        &b_proj,
        &a_proj,
        &w.dt_bias,
        &w.a_log,
        &recurrent_ptrs,
        &mut core,
        1,
        g.key_heads,
        g.value_heads,
        g.key_dim,
        g.value_dim,
    );
    let (normed, out) = gdn_finish(ctx, g, w, &core, &z_proj)?;
    Ok((
        GdnTrace {
            qkv,
            z: z_proj,
            b: b_proj,
            a: a_proj,
            core,
            normed,
        },
        out,
    ))
}

/// Gated full attention for `rows` new tokens appended at `kv.len`, dense and
/// causal over everything in the scratch: the reference is QSA, whose
/// selection keeps every causal key while the visible prefix fits its budget,
/// so this equals it up to that length and no further. A whole prompt and a
/// single decode token take the same path: the hd256 prefill entry over a plan
/// that starts at `kv.len`, the route Qwen3.5 takes for a GQA group the batch
/// decode kernels do not compile. Returns the stages and the `o_proj` output.
fn attention_rows(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &FullAttention,
    rope: &Rope,
    kv: &mut KvScratch,
    mixed: &HiddenStates,
) -> Result<(AttentionTrace, HiddenStates)> {
    let rows = mixed.seq_len;
    let start = kv.len;
    let kv_len = start + rows;
    ensure!(
        kv_len <= g.dense_attention_rows(),
        "the sequence reaches {kv_len} tokens; past {} the reference's QSA drops keys and dense attention no longer matches it",
        g.dense_attention_rows()
    );
    ensure!(
        kv_len <= kv.capacity,
        "the K/V scratch holds {} tokens, the sequence reaches {kv_len}",
        kv.capacity
    );
    let covered = rope.cos.len / g.rotary_dim;
    ensure!(
        kv_len <= covered,
        "the RoPE table covers {covered} positions, the sequence reaches {kv_len}"
    );
    let mut q_proj = HiddenStates::zeros(ctx, g.q_width(), rows)?;
    gemm_into_checked(ctx, &w.q_proj, mixed, &mut q_proj)?;
    let mut k_proj = HiddenStates::zeros(ctx, g.kv_width(), rows)?;
    gemm_into_checked(ctx, &w.k_proj, mixed, &mut k_proj)?;
    let mut v_proj = HiddenStates::zeros(ctx, g.kv_width(), rows)?;
    gemm_into_checked(ctx, &w.v_proj, mixed, &mut v_proj)?;

    let pages = kv_len.div_ceil(KV_PAGE);
    let page_ids: Vec<i32> = (0..pages).map(i32::try_from).collect::<Result<_, _>>()?;
    let plan = PrefillPagedPlan::new_with_cta_tile_q(
        ctx,
        &page_ids,
        kv_len - KV_PAGE * (pages - 1),
        start,
        rows,
        g.heads,
        g.kv_heads,
        g.head_dim,
        0,
    )?;
    let mut q_prepped = HiddenStates::zeros(ctx, g.attn_width(), rows)?;
    let mut gated = HiddenStates::zeros(ctx, g.attn_width(), rows)?;
    attention_prefill_hd256_into(
        ctx,
        &q_proj,
        &k_proj,
        &v_proj,
        &w.q_norm,
        &w.k_norm,
        &rope.cos,
        &rope.sin,
        &kv.pool,
        &kv.layout,
        0,
        &plan,
        start,
        g.rotary_dim,
        g.rms_eps,
        &mut q_prepped,
        &mut gated,
    )?;
    kv.len = kv_len;
    let mut out = HiddenStates::zeros(ctx, g.hidden, rows)?;
    gemm_into_checked(ctx, &w.o_proj, &gated, &mut out)?;
    Ok((
        AttentionTrace {
            q_proj,
            k_proj,
            v_proj,
            q_prepped,
            gated,
        },
        out,
    ))
}

/// The scatter after the mixer, the second mix, the MoE block and the scatter out.
#[allow(clippy::too_many_arguments)]
fn finish_layer(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &DecoderLayer,
    stream_in: &HiddenStates,
    ple: Option<PleTrace>,
    attn_mixed: HiddenStates,
    attn_inject: HiddenStates,
    mixer: MixerTrace,
    attn_out: HiddenStates,
) -> Result<LayerTrace> {
    let rows = stream_in.seq_len;
    // The scatter adds onto the stream the mix read: with the n-gram block,
    // the stream with its output added.
    let base = ple.as_ref().map_or(stream_in, |p| &p.stream);
    let mut stream_after_attn = HiddenStates::zeros(ctx, g.stream_width(), rows)?;
    hc_scatter_add_into(ctx, &attn_out, &attn_inject, base, &mut stream_after_attn)?;
    let (mlp_mixed, mlp_inject) = mix(ctx, g, &w.mlp_hc, &stream_after_attn)?;
    let moe = moe_prefill(ctx, g, &w.moe, &mlp_mixed)?;
    let mut stream_out = HiddenStates::zeros(ctx, g.stream_width(), rows)?;
    hc_scatter_add_into(
        ctx,
        &moe.out,
        &mlp_inject,
        &stream_after_attn,
        &mut stream_out,
    )?;
    Ok(LayerTrace {
        ple,
        attn_mixed,
        attn_inject,
        mixer,
        attn_out,
        stream_after_attn,
        mlp_mixed,
        mlp_inject,
        moe,
        stream_out,
    })
}

fn check_stream(g: &Geometry, ids: &[u32], stream_in: &HiddenStates) -> Result<()> {
    ensure!(stream_in.seq_len > 0, "the prompt is empty");
    ensure!(
        stream_in.hidden_dim == g.stream_width(),
        "stream is {} wide, the geometry has {}",
        stream_in.hidden_dim,
        g.stream_width()
    );
    ensure!(
        ids.len() == stream_in.seq_len,
        "{} token ids for {} stream rows",
        ids.len(),
        stream_in.seq_len
    );
    Ok(())
}

/// One whole-sequence pass from the state as given (zeroed for a fresh
/// prompt). `ids` are the prompt's tokens, which only the n-gram block reads.
pub fn layer_prefill(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &DecoderLayer,
    rope: &Rope,
    state: &mut LayerState,
    ids: &[u32],
    stream_in: &HiddenStates,
) -> Result<LayerTrace> {
    check_stream(g, ids, stream_in)?;
    let ple = match (&w.ple, state.ple.as_mut()) {
        (Some(ple), Some(st)) => {
            let ids: Vec<i64> = ids.iter().map(|&id| i64::from(id)).collect();
            Some(ple_prefill(ctx, g, ple, st, &ids, stream_in)?)
        }
        (None, None) => None,
        _ => bail!("the layer state was built for the other PLE shape"),
    };
    let (attn_mixed, attn_inject) = mix(
        ctx,
        g,
        &w.attn_hc,
        ple.as_ref().map_or(stream_in, |p| &p.stream),
    )?;
    let (mixer, attn_out) = match (&w.mixer, &mut state.mixer) {
        (Mixer::Linear(gdn), MixerState::Linear(state)) => {
            let (trace, out) = gdn_prefill(ctx, g, gdn, state, &attn_mixed)?;
            (MixerTrace::Linear(trace), out)
        }
        (Mixer::Full(attn), MixerState::Full(kv)) => {
            ensure!(
                kv.len == 0,
                "the K/V scratch already holds {} tokens",
                kv.len
            );
            let (trace, out) = attention_rows(ctx, g, attn, rope, kv, &attn_mixed)?;
            (MixerTrace::Full(trace), out)
        }
        _ => bail!("the mixer state was built for the other layer kind"),
    };
    finish_layer(
        ctx,
        g,
        w,
        stream_in,
        ple,
        attn_mixed,
        attn_inject,
        mixer,
        attn_out,
    )
}

/// One token through the layer on the carried state.
pub fn layer_decode(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &DecoderLayer,
    rope: &Rope,
    state: &mut LayerState,
    id: u32,
    stream_in: &HiddenStates,
) -> Result<LayerTrace> {
    check_stream(g, &[id], stream_in)?;
    let ple = match (&w.ple, state.ple.as_mut()) {
        (Some(ple), Some(st)) => Some(ple_decode(ctx, g, ple, st, i64::from(id), stream_in)?),
        (None, None) => None,
        _ => bail!("the layer state was built for the other PLE shape"),
    };
    let (attn_mixed, attn_inject) = mix(
        ctx,
        g,
        &w.attn_hc,
        ple.as_ref().map_or(stream_in, |p| &p.stream),
    )?;
    let (mixer, attn_out) = match (&w.mixer, &mut state.mixer) {
        (Mixer::Linear(gdn), MixerState::Linear(state)) => {
            let (trace, out) = gdn_decode(ctx, g, gdn, state, &attn_mixed)?;
            (MixerTrace::Linear(trace), out)
        }
        (Mixer::Full(attn), MixerState::Full(kv)) => {
            let (trace, out) = attention_rows(ctx, g, attn, rope, kv, &attn_mixed)?;
            (MixerTrace::Full(trace), out)
        }
        _ => bail!("the mixer state was built for the other layer kind"),
    };
    finish_layer(
        ctx,
        g,
        w,
        stream_in,
        ple,
        attn_mixed,
        attn_inject,
        mixer,
        attn_out,
    )
}
