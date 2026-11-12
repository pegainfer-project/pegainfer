//! One decoder layer, stage by stage: the n-gram block on the one layer that
//! carries it, the attention hyper-connection mix, the token mixer (Gated
//! DeltaNet on the qwen35 kernels, or gated full attention on the qwen35 hd256
//! prefill), the scatter back into the stream, then the same around the MoE
//! block. A layer's state holds several sequences, one per slot: prefill walks
//! one sequence's prompt in its slot, and decode advances a batch of slots by
//! one token each.

use std::collections::BTreeSet;

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
use pegainfer_kernels::ops::attention_decode_via_prefill_hd256_into;
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

/// The K/V of every slot in one paged pool, in the one layer format the hd256
/// kernels read, so one plan covers a batch: a slot owns `capacity` tokens'
/// worth of pages, starting at [`Self::first_page`].
pub struct KvPool {
    pub pool: CudaSlice<bf16>,
    pub layout: PagedKvLayout,
    /// Tokens one slot holds.
    pub capacity: usize,
    /// Tokens written so far per slot; a slot's next token lands at its length.
    pub lens: Vec<usize>,
}

impl KvPool {
    pub fn first_page(&self, slot: usize) -> usize {
        slot * self.capacity.div_ceil(KV_PAGE).max(1)
    }

    /// The pages holding the first `kv_len` tokens of `slot`.
    fn pages(&self, slot: usize, kv_len: usize) -> Result<Vec<i32>> {
        let first = self.first_page(slot);
        Ok((first..first + kv_len.div_ceil(KV_PAGE))
            .map(i32::try_from)
            .collect::<Result<_, _>>()?)
    }
}

/// The token mixer's state for every slot, zeroed.
pub enum MixerState {
    Linear(Vec<GdnState>),
    Full(KvPool),
}

impl MixerState {
    pub fn new(
        ctx: &DeviceContext,
        g: &Geometry,
        mixer: &Mixer,
        slots: usize,
        capacity: usize,
    ) -> Result<Self> {
        Ok(match mixer {
            Mixer::Linear(_) => Self::Linear(
                (0..slots)
                    .map(|_| -> Result<GdnState> {
                        Ok(GdnState {
                            conv: DeviceVec::zeros(ctx, g.qkv_width() * (g.conv_kernel - 1))?,
                            recurrent: ctx
                                .stream
                                .alloc_zeros(g.value_heads * g.key_dim * g.value_dim)?,
                        })
                    })
                    .collect::<Result<_>>()?,
            ),
            Mixer::Full(_) => {
                let layout = PagedKvLayout::new(1, g.kv_heads, g.head_dim, KV_PAGE);
                let pages = slots * capacity.div_ceil(KV_PAGE).max(1);
                Self::Full(KvPool {
                    pool: ctx.stream.alloc_zeros(pages * layout.page_stride)?,
                    layout,
                    capacity,
                    lens: vec![0; slots],
                })
            }
        })
    }

    fn slots(&self) -> usize {
        match self {
            Self::Linear(states) => states.len(),
            Self::Full(kv) => kv.lens.len(),
        }
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

impl PleState {
    fn new(ctx: &DeviceContext, g: &Geometry, ple: &PleBlock) -> Result<Self> {
        let rows = ple.geometry.conv_state();
        Ok(Self {
            history: Vec::new(),
            stream: HiddenStates::zeros(ctx, g.stream_width(), rows)?,
            key: HiddenStates::zeros(ctx, g.stream_width(), rows)?,
            value: HiddenStates::zeros(ctx, g.hidden, rows)?,
            rows: 0,
        })
    }
}

/// One layer's state for `slots` sequences, each `capacity` tokens at most.
pub struct LayerState {
    pub mixer: MixerState,
    /// Per slot, on the layer that carries the n-gram block.
    pub ple: Option<Vec<PleState>>,
}

impl LayerState {
    pub fn new(
        ctx: &DeviceContext,
        g: &Geometry,
        layer: &DecoderLayer,
        slots: usize,
        capacity: usize,
    ) -> Result<Self> {
        let ple = layer
            .ple
            .as_ref()
            .map(|ple| {
                (0..slots)
                    .map(|_| PleState::new(ctx, g, ple))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        Ok(Self {
            mixer: MixerState::new(ctx, g, &layer.mixer, slots, capacity)?,
            ple,
        })
    }

    /// `slots` must be distinct slots of this state.
    fn check_slots(&self, slots: &[usize]) -> Result<()> {
        let held = self.mixer.slots();
        ensure!(
            slots.iter().all(|&slot| slot < held),
            "slots {slots:?} reach past the {held} the state holds"
        );
        ensure!(
            slots.iter().collect::<BTreeSet<_>>().len() == slots.len(),
            "slots {slots:?} repeat"
        );
        Ok(())
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

/// The n-gram block for one token in each of `slots` on its carried state:
/// the hash sees the slot's remembered tokens, the rows of the whole batch are
/// gathered and projected together, and the block runs per slot over the
/// remembered rows plus the new one, keeping the last output row.
fn ple_decode(
    ctx: &DeviceContext,
    g: &Geometry,
    ple: &PleBlock,
    states: &mut [PleState],
    slots: &[usize],
    ids: &[u32],
    stream_in: &HiddenStates,
) -> Result<PleTrace> {
    let batch = slots.len();
    let heads = ple.hasher.heads();
    let mut histories = Vec::with_capacity(batch);
    let mut row_ids = Vec::with_capacity(batch * heads);
    for (&slot, &id) in slots.iter().zip(ids) {
        let mut history = states[slot].history.clone();
        history.push(i64::from(id));
        let all = ple.hasher.ids(&history);
        row_ids.extend_from_slice(&all[all.len() - heads..]);
        histories.push(history);
    }
    let (embeds, key, value) = ple_project(ctx, g, ple, &row_ids, batch)?;
    let mut out = HiddenStates::zeros(ctx, g.stream_width(), batch)?;
    for (row, (&slot, history)) in slots.iter().zip(&histories).enumerate() {
        let state = &mut states[slot];
        let rows = state.rows + 1;
        let mut all_stream = HiddenStates::zeros(ctx, g.stream_width(), rows)?;
        let mut all_key = HiddenStates::zeros(ctx, g.stream_width(), rows)?;
        let mut all_value = HiddenStates::zeros(ctx, g.hidden, rows)?;
        copy_hidden_token_range_into(ctx, &state.stream, 0, &mut all_stream, 0, state.rows)?;
        copy_hidden_token_range_into(ctx, &state.key, 0, &mut all_key, 0, state.rows)?;
        copy_hidden_token_range_into(ctx, &state.value, 0, &mut all_value, 0, state.rows)?;
        copy_hidden_token_range_into(ctx, stream_in, row, &mut all_stream, state.rows, 1)?;
        copy_hidden_token_range_into(ctx, &key, row, &mut all_key, state.rows, 1)?;
        copy_hidden_token_range_into(ctx, &value, row, &mut all_value, state.rows, 1)?;
        let all_out = ple_block(ctx, g, ple, &all_stream, &all_key, &all_value, rows)?;
        copy_hidden_token_range_into(ctx, &all_out, rows - 1, &mut out, row, 1)?;
        ple_remember(ctx, ple, state, history, &all_stream, &all_key, &all_value)?;
    }
    let mut stream = HiddenStates::zeros(ctx, g.stream_width(), batch)?;
    add_batch_into(ctx, stream_in, &out, &mut stream)?;
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

/// The Gated DeltaNet block for one token in each of `slots` on its carried
/// state: the decode kernels take one row per request and reach its state
/// through a pointer per request.
fn gdn_decode(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &GatedDeltaNet,
    states: &mut [GdnState],
    slots: &[usize],
    mixed: &HiddenStates,
) -> Result<(GdnTrace, HiddenStates)> {
    let batch = slots.len();
    let (qkv, z_proj, b_proj, a_proj) = gdn_project(ctx, g, w, mixed)?;
    let mut conv = Vec::with_capacity(batch);
    let mut recurrent = Vec::with_capacity(batch);
    for &slot in slots {
        let state = &mut states[slot];
        conv.push(state.conv.data.device_ptr_mut(&ctx.stream).0);
        recurrent.push(state.recurrent.device_ptr_mut(&ctx.stream).0);
    }
    let conv_ptrs: CudaSlice<u64> = ctx.stream.clone_htod(&conv)?;
    let recurrent_ptrs: CudaSlice<u64> = ctx.stream.clone_htod(&recurrent)?;
    let mut qkv_conv = HiddenStates::zeros(ctx, g.qkv_width(), batch)?;
    conv1d_decode_batch_into(
        ctx,
        &qkv,
        &w.conv_weight,
        &conv_ptrs,
        &mut qkv_conv,
        g.conv_kernel,
    );
    let mut core = HiddenStates::zeros(ctx, g.value_width(), batch)?;
    gated_delta_rule_decode_batch_into(
        ctx,
        &qkv_conv,
        &b_proj,
        &a_proj,
        &w.dt_bias,
        &w.a_log,
        &recurrent_ptrs,
        &mut core,
        batch,
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

/// Dense causal attention stands in for the reference's QSA, whose selection
/// keeps every causal key while the visible prefix fits its budget, so it
/// equals it up to that length and no further.
fn check_reach(g: &Geometry, kv: &KvPool, rope: &Rope, kv_len: usize) -> Result<()> {
    ensure!(
        kv_len <= g.dense_attention_rows(),
        "the sequence reaches {kv_len} tokens; past {} the reference's QSA drops keys and dense attention no longer matches it",
        g.dense_attention_rows()
    );
    ensure!(
        kv_len <= kv.capacity,
        "a K/V slot holds {} tokens, the sequence reaches {kv_len}",
        kv.capacity
    );
    let covered = rope.cos.len / g.rotary_dim;
    ensure!(
        kv_len <= covered,
        "the RoPE table covers {covered} positions, the sequence reaches {kv_len}"
    );
    Ok(())
}

fn attention_project(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &FullAttention,
    mixed: &HiddenStates,
) -> Result<(HiddenStates, HiddenStates, HiddenStates)> {
    let rows = mixed.seq_len;
    let mut q_proj = HiddenStates::zeros(ctx, g.q_width(), rows)?;
    gemm_into_checked(ctx, &w.q_proj, mixed, &mut q_proj)?;
    let mut k_proj = HiddenStates::zeros(ctx, g.kv_width(), rows)?;
    gemm_into_checked(ctx, &w.k_proj, mixed, &mut k_proj)?;
    let mut v_proj = HiddenStates::zeros(ctx, g.kv_width(), rows)?;
    gemm_into_checked(ctx, &w.v_proj, mixed, &mut v_proj)?;
    Ok((q_proj, k_proj, v_proj))
}

/// Gated full attention over one prompt appended in `slot` at its length,
/// through the hd256 prefill entry. Returns the stages and the `o_proj`
/// output.
fn attention_prefill(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &FullAttention,
    rope: &Rope,
    kv: &mut KvPool,
    slot: usize,
    mixed: &HiddenStates,
) -> Result<(AttentionTrace, HiddenStates)> {
    let rows = mixed.seq_len;
    let start = kv.lens[slot];
    let kv_len = start + rows;
    check_reach(g, kv, rope, kv_len)?;
    let (q_proj, k_proj, v_proj) = attention_project(ctx, g, w, mixed)?;
    let pages = kv.pages(slot, kv_len)?;
    let plan = PrefillPagedPlan::new_with_cta_tile_q(
        ctx,
        &pages,
        kv_len - KV_PAGE * (pages.len() - 1),
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
    kv.lens[slot] = kv_len;
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

/// Gated full attention for one token in each of `slots`, appended at the
/// slot's length: the batched hd256 decode entry, which runs FlashInfer's
/// paged prefill with one query row per slot, the route Qwen3.5 takes for a GQA
/// group the batch decode kernels do not compile. Returns the stages and the
/// `o_proj` output.
fn attention_decode(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &FullAttention,
    rope: &Rope,
    kv: &mut KvPool,
    slots: &[usize],
    mixed: &HiddenStates,
) -> Result<(AttentionTrace, HiddenStates)> {
    let batch = slots.len();
    let mut pages = Vec::with_capacity(batch);
    let mut last_page_lens = Vec::with_capacity(batch);
    let mut starts = Vec::with_capacity(batch);
    for &slot in slots {
        let start = kv.lens[slot];
        let kv_len = start + 1;
        check_reach(g, kv, rope, kv_len)?;
        let slot_pages = kv.pages(slot, kv_len)?;
        last_page_lens.push(kv_len - KV_PAGE * (slot_pages.len() - 1));
        pages.push(slot_pages);
        starts.push(start);
    }
    let (q_proj, k_proj, v_proj) = attention_project(ctx, g, w, mixed)?;
    let plan = PrefillPagedPlan::new_batch_with_cta_tile_q(
        ctx,
        &pages,
        &last_page_lens,
        &starts,
        &vec![1; batch],
        g.heads,
        g.kv_heads,
        g.head_dim,
        0,
    )?;
    // The entry norms and rotates K in place; the trace keeps the projection.
    let mut k = HiddenStates::zeros(ctx, g.kv_width(), batch)?;
    copy_hidden_token_range_into(ctx, &k_proj, 0, &mut k, 0, batch)?;
    let mut q_prepped = HiddenStates::zeros(ctx, g.attn_width(), batch)?;
    let mut gated = HiddenStates::zeros(ctx, g.attn_width(), batch)?;
    attention_decode_via_prefill_hd256_into(
        ctx,
        &q_proj,
        &mut k,
        &v_proj,
        &w.q_norm,
        &w.k_norm,
        &rope.cos,
        &rope.sin,
        &kv.pool,
        &kv.layout,
        0,
        &plan,
        g.rotary_dim,
        g.rms_eps,
        &mut q_prepped,
        &mut gated,
    )?;
    for &slot in slots {
        kv.lens[slot] += 1;
    }
    let mut out = HiddenStates::zeros(ctx, g.hidden, batch)?;
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

/// One sequence's whole prompt in `slot`, from the slot's state as given
/// (zeroed for a fresh prompt). `ids` are the prompt's tokens, which only the
/// n-gram block reads.
#[allow(clippy::too_many_arguments)]
pub fn layer_prefill(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &DecoderLayer,
    rope: &Rope,
    state: &mut LayerState,
    slot: usize,
    ids: &[u32],
    stream_in: &HiddenStates,
) -> Result<LayerTrace> {
    check_stream(g, ids, stream_in)?;
    state.check_slots(&[slot])?;
    let ple = match (&w.ple, state.ple.as_mut()) {
        (Some(ple), Some(states)) => {
            let ids: Vec<i64> = ids.iter().map(|&id| i64::from(id)).collect();
            Some(ple_prefill(
                ctx,
                g,
                ple,
                &mut states[slot],
                &ids,
                stream_in,
            )?)
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
        (Mixer::Linear(gdn), MixerState::Linear(states)) => {
            let (trace, out) = gdn_prefill(ctx, g, gdn, &mut states[slot], &attn_mixed)?;
            (MixerTrace::Linear(trace), out)
        }
        (Mixer::Full(attn), MixerState::Full(kv)) => {
            ensure!(
                kv.lens[slot] == 0,
                "K/V slot {slot} already holds {} tokens",
                kv.lens[slot]
            );
            let (trace, out) = attention_prefill(ctx, g, attn, rope, kv, slot, &attn_mixed)?;
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

/// One token in each of `slots` through the layer on the carried state, as one
/// batch: row `i` of `stream_in` and `ids[i]` belong to `slots[i]`.
#[allow(clippy::too_many_arguments)]
pub fn layer_decode(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &DecoderLayer,
    rope: &Rope,
    state: &mut LayerState,
    slots: &[usize],
    ids: &[u32],
    stream_in: &HiddenStates,
) -> Result<LayerTrace> {
    check_stream(g, ids, stream_in)?;
    ensure!(
        slots.len() == ids.len(),
        "{} slots for {} tokens",
        slots.len(),
        ids.len()
    );
    state.check_slots(slots)?;
    let ple = match (&w.ple, state.ple.as_mut()) {
        (Some(ple), Some(states)) => Some(ple_decode(ctx, g, ple, states, slots, ids, stream_in)?),
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
        (Mixer::Linear(gdn), MixerState::Linear(states)) => {
            let (trace, out) = gdn_decode(ctx, g, gdn, states, slots, &attn_mixed)?;
            (MixerTrace::Linear(trace), out)
        }
        (Mixer::Full(attn), MixerState::Full(kv)) => {
            let (trace, out) = attention_decode(ctx, g, attn, rope, kv, slots, &attn_mixed)?;
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
