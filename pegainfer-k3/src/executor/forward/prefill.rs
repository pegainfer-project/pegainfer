//! The prefill arm of the forward pass: the chunk entry point, the chunkwise
//! KDA and dense-FMHA MLA leaves, and the boundary sample that replaces the
//! epilogue the chunk steps skip.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::K3_CONV_WIDTH;
use pegainfer_kernels::ops::K3_MLA_HEADS;
use pegainfer_kernels::ops::argmax_bf16_split_into;
use pegainfer_kernels::ops::gemm_rows_into_checked;
use pegainfer_kernels::ops::k3_conv_silu_batched_launch;
use pegainfer_kernels::ops::k3_flash_kda_fwd_launch;
use pegainfer_kernels::ops::k3_flash_mla_prefill_fwd_launch;
use pegainfer_kernels::ops::k3_land_batched_launch;
use pegainfer_kernels::ops::k3_mla_prefill_expand_k_launch;
use pegainfer_kernels::ops::k3_mla_prefill_gather_launch;
use pegainfer_kernels::ops::k3_o_norm_gate_batched_launch;
use pegainfer_kernels::ops::k3_rms_norm_rbs_batched_launch;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;

use super::super::buffers::K3_CONV_STATE;
use super::super::buffers::K3_KDA_FUSED;
use super::super::buffers::K3_KDA_STATE;
use super::super::buffers::K3_KDA_WSM_PADDED;
use super::super::buffers::K3KdaState;
use super::super::buffers::K3Scratch;
use super::super::buffers::K3StatePool;
use super::super::buffers::copy_rows;
use super::super::buffers::copy_rows_2d;
use super::super::buffers::parity_pair;
use super::super::cp::K3CpScratch;
use super::super::cp::K3CpWindowKind;
use super::super::cp::k3_cp_copy_in;
use super::super::paged_kv::K3_KV_PAGE_TOKENS;
use super::super::paged_kv::K3_MLA_LATENT_ROW;
use super::super::paged_kv::K3PagedKv;
use super::gemm::K3PartialSpan;
use super::gemm::k3_gemm_full;
use super::gemm::k3_gemm_partial;
use super::step::K3AuxSink;
use super::step::K3KdaGroup;
use super::step::K3StepMode;
use super::step::K3StepShape;
use super::step::attn_res;
use super::step::k3_step;
use crate::config::K3_ATTN_INNER;
use crate::config::K3_HEAD_DIM;
use crate::config::K3_HEADS;
use crate::config::K3_HIDDEN;
use crate::config::K3_KV_B_OUT;
use crate::config::K3_KV_LORA_RANK;
use crate::config::K3_VOCAB;
use crate::model::K3KdaWeights;
use crate::model::K3LayerWeights;
use crate::model::K3MlaWeights;
use crate::model::K3RankModel;

/// One prefill chunk: the same batched step, with the bucket's rows carrying
/// consecutive tokens of ONE sequence instead of independent sequences.
///
/// `shape.live_rows` is the chunk's token count and `shape.parity` the KDA
/// state slab the chunk reads (it lands in the other — parity double-buffers
/// per chunk here, not per token). The caller stages `token_ids`, ascending
/// per-token `context_len` and per-token `kv_row` exactly as for decode, and
/// mirrors the sequence's block-table row across the bucket — causality in the
/// MLA layers comes from the per-row context length, so the batched attention
/// kernel serves the chunk unchanged. Only the KDA layers walk the chunk's
/// tokens through the recurrence sequentially ([`kda_attention_chunk`]);
/// every other stage runs batched over the rows.
pub(crate) fn k3_prefill_chunk_step(
    ctx: &DeviceContext,
    model: &K3RankModel,
    shape: K3StepShape,
    state: &mut K3StatePool,
    scratch: &mut K3Scratch,
    aux: Option<K3AuxSink<'_>>,
    cp: Option<&mut K3CpScratch>,
) -> Result<()> {
    ensure!(
        (1..=shape.bucket).contains(&shape.live_rows),
        "K3 prefill chunk of {} tokens does not fit its {} bucket",
        shape.live_rows,
        shape.bucket
    );
    k3_step(
        ctx,
        model,
        shape,
        K3StepMode::PrefillChunk,
        state,
        scratch,
        aux,
        cp,
    )
}

/// One speculative verify step: the batched step over packed per-slot row
/// groups (deferred-commit replay + anchor + drafts per slot).
///
/// The caller stages `token_ids`, per-row `context_len` and `kv_row` (replay
/// rows re-run positions whose latents are already cached, so their `kv_row`
/// is `-1`), and stages the packed verify block table. The argmax epilogue
/// runs over the whole bucket; the caller reads the span rows' argmaxes back
/// and decides acceptance.
pub(crate) fn k3_verify_step(
    ctx: &DeviceContext,
    model: &K3RankModel,
    shape: K3StepShape,
    groups: &[K3KdaGroup],
    state: &mut K3StatePool,
    scratch: &mut K3Scratch,
    aux: Option<K3AuxSink<'_>>,
) -> Result<()> {
    for group in groups {
        ensure!(
            group.row + group.commit_rows + group.spec_rows <= shape.bucket
                && group.spec_rows > 0
                && group.state_row < state.rows,
            "K3 verify group at row {} ({}+{} rows, state row {}) does not fit the step",
            group.row,
            group.commit_rows,
            group.spec_rows,
            group.state_row,
        );
    }
    k3_step(
        ctx,
        model,
        shape,
        K3StepMode::Verify(groups),
        state,
        scratch,
        aux,
        None,
    )
}

/// Sample the prefill boundary token: the epilogue the chunk steps skipped,
/// at `b = 1` over the final chunk's last live row.
///
/// The caller must have collapsed the last live row's attention-residual
/// snapshots to row 0 first (the same collapse the decode handover needs) —
/// this reads row 0 of `snapshots`. The boundary hidden state is copied from
/// `hidden[last_row]` into row 0 of the `prefix` scratch, so the sample lands
/// in `argmax_indices[0]`.
pub(crate) fn k3_prefill_boundary_sample(
    ctx: &DeviceContext,
    model: &K3RankModel,
    last_row: usize,
    snapshots: &CudaSlice<bf16>,
    scratch: &mut K3Scratch,
) -> Result<()> {
    copy_rows_2d(
        ctx,
        &scratch.hidden,
        last_row * K3_HIDDEN,
        K3_HIDDEN,
        &mut scratch.prefix,
        0,
        K3_HIDDEN,
        1,
        K3_HIDDEN,
    )?;
    attn_res(
        ctx,
        1,
        model.blocks,
        &scratch.prefix,
        snapshots,
        &model.sw_out,
        &mut scratch.scores,
        &mut scratch.mixed,
    )?;
    k3_rms_norm_rbs_batched_launch(
        ctx,
        1,
        K3_HIDDEN,
        &scratch.mixed,
        &model.gamma_final.data,
        &mut scratch.normed,
    )?;
    k3_gemm_full(
        ctx,
        &model.w_lm,
        &scratch.normed,
        1,
        &mut scratch.logit_partial,
    )?;
    k3_land_batched_launch(
        ctx,
        1,
        K3_VOCAB,
        K3_VOCAB,
        0,
        1,
        &scratch.logit_partial,
        &mut scratch.logits,
    )?;
    argmax_bf16_split_into(
        ctx,
        &scratch.logits,
        1,
        K3_VOCAB,
        &mut scratch.argmax_partial_values,
        &mut scratch.argmax_partial_indices,
        &mut scratch.argmax_values,
        &mut scratch.argmax_indices,
    )
}

/// The KDA layer of a prefill chunk or verify step: batched projections, a
/// batched convolution over prebuilt windows, and the delta rule as chunkwise
/// FlashKDA forwards (third_party/flash-kda, MIT, MoonshotAI) — the per-token
/// walk collapses into a handful of launches per layer.
///
/// `groups` carves the bucket into per-sequence segments (see
/// [`K3KdaGroup`]): a prefill chunk is the one-group all-commit case, a
/// verify step packs one group per slot with a commit prefix (the deferred
/// replay) and a speculative tail whose successor state is discarded.
///
/// Numerics: the convolution windows are the landed bf16 inputs themselves
/// (`window[t][j] = x[t - 3 + j]`, carried window for the first rows), so the
/// conv is bitwise against sequential stepping. FlashKDA computes the same
/// delta rule the fused TileLang core spells — same gate formula from the
/// same pre-activation landing, beta sigmoid in-kernel — but with an f32 q/k
/// l2norm chain (the TileLang core mirrors the reference's bf16 chain) and
/// chunkwise accumulation order, and the projections run at the step's
/// bucket. Chunked prefill and verify are therefore held to the fixture's
/// noise floor, like any cross-bucket comparison; decode keeps the
/// bit-matched fused core.
pub(super) fn kda_attention_chunk(
    ctx: &DeviceContext,
    shape: K3StepShape,
    layer: &K3LayerWeights,
    w: &K3KdaWeights,
    kda_state: &mut K3KdaState,
    groups: &[K3KdaGroup],
    s: &mut K3Scratch,
    mut cp: Option<&mut K3CpScratch>,
) -> Result<()> {
    let b = shape.bucket;
    // Batched projections — the decode arm's launches, at the chunk's bucket.
    k3_rms_norm_rbs_batched_launch(
        ctx,
        b,
        K3_HIDDEN,
        &s.mixed,
        &layer.gamma_in.data,
        &mut s.normed,
    )?;
    k3_gemm_partial(
        ctx,
        &w.wbig,
        3 * K3_ATTN_INNER,
        K3_ATTN_INNER,
        &s.normed,
        b,
        &mut s.kda_gate_partial,
        K3PartialSpan {
            offset: 3 * K3_ATTN_INNER,
            stride: K3_KDA_FUSED,
        },
    )?;
    k3_gemm_full(ctx, &w.wsm, &s.normed, b, &mut s.kda_wsm_partial)?;
    k3_land_batched_launch(
        ctx,
        b,
        K3_KDA_WSM_PADDED,
        K3_HEADS,
        0,
        1,
        &s.kda_wsm_partial,
        &mut s.beta,
    )?;
    k3_land_batched_launch(
        ctx,
        b,
        K3_KDA_WSM_PADDED,
        K3_HEAD_DIM,
        K3_HEADS,
        1,
        &s.kda_wsm_partial,
        &mut s.forget_low,
    )?;
    k3_land_batched_launch(
        ctx,
        b,
        K3_KDA_FUSED,
        K3_ATTN_INNER,
        3 * K3_ATTN_INNER,
        1,
        &s.kda_gate_partial,
        &mut s.out_gate,
    )?;
    k3_gemm_full(ctx, &w.w_f_b, &s.forget_low, b, &mut s.kda_forget_partial)?;

    // CP conv halo: this segment's last normed rows go upstream-to-downstream
    // so the next rank's first conv windows see them; the receiver projects
    // them through the very bands above and lands them as its carried window.
    if let Some(cp) = cp.as_deref_mut() {
        ensure!(
            groups.len() == 1
                && groups[0].spec_rows == 0
                && groups[0].commit_rows == shape.live_rows
                && groups[0].state_row == 0,
            "K3 CP prefill runs the one-group all-commit chunk"
        );
        if cp.cp_rank + 1 < cp.cp_size {
            copy_rows_2d(
                ctx,
                &s.normed,
                (shape.live_rows - K3_CONV_STATE) * K3_HIDDEN,
                K3_HIDDEN,
                &mut cp.normed_tail,
                0,
                K3_HIDDEN,
                K3_CONV_STATE,
                K3_HIDDEN,
            )?;
        }
        let group = cp.group.clone();
        let sync = cp.window_sync(K3CpWindowKind::Halo);
        group.exchange(ctx, &sync, || {
            if cp.cp_rank > 0 {
                let source = cp.peers[cp.cp_rank - 1].normed_tail;
                k3_cp_copy_in(
                    ctx,
                    source,
                    0,
                    &mut cp.halo_normed,
                    0,
                    K3_CONV_STATE * K3_HIDDEN,
                )?;
            }
            Ok(())
        })?;
        if cp.cp_rank > 0 {
            // Bucket 4 is the narrowest compiled batch covering the
            // K3_CONV_STATE halo rows; row 3 is zero padding, never copied out.
            let halo_bucket = K3_CONV_STATE + 1;
            for (stream_index, band) in [0usize, K3_ATTN_INNER, 2 * K3_ATTN_INNER]
                .into_iter()
                .enumerate()
            {
                k3_gemm_partial(
                    ctx,
                    &w.wbig,
                    band,
                    K3_ATTN_INNER,
                    &cp.halo_normed,
                    halo_bucket,
                    &mut cp.halo_partial,
                    K3PartialSpan::whole(K3_ATTN_INNER),
                )?;
                k3_land_batched_launch(
                    ctx,
                    halo_bucket,
                    K3_ATTN_INNER,
                    K3_ATTN_INNER,
                    0,
                    1,
                    &cp.halo_partial,
                    &mut cp.halo_xs,
                )?;
                copy_rows_2d(
                    ctx,
                    &cp.halo_xs,
                    0,
                    K3_ATTN_INNER,
                    &mut kda_state.conv[shape.parity][stream_index],
                    0,
                    K3_ATTN_INNER,
                    K3_CONV_STATE,
                    K3_ATTN_INNER,
                )?;
            }
        }
    }

    kda_conv_stream_chunk(
        ctx,
        b,
        groups,
        &w.wbig,
        0,
        &w.cw_q,
        &mut kda_state.conv,
        0,
        &s.normed,
        &mut s.kda_conv_partial,
        &mut s.conv_x,
        &mut s.conv_window,
        &mut s.conv_window_next,
        &mut s.conv_q,
    )?;
    kda_conv_stream_chunk(
        ctx,
        b,
        groups,
        &w.wbig,
        K3_ATTN_INNER,
        &w.cw_k,
        &mut kda_state.conv,
        1,
        &s.normed,
        &mut s.kda_conv_partial,
        &mut s.conv_x,
        &mut s.conv_window,
        &mut s.conv_window_next,
        &mut s.conv_k,
    )?;
    kda_conv_stream_chunk(
        ctx,
        b,
        groups,
        &w.wbig,
        2 * K3_ATTN_INNER,
        &w.cw_v,
        &mut kda_state.conv,
        2,
        &s.normed,
        &mut s.kda_conv_partial,
        &mut s.conv_x,
        &mut s.conv_window,
        &mut s.conv_window_next,
        &mut s.conv_v,
    )?;

    // The chunkwise delta rule: each group's segment through the vendored
    // FlashKDA forward (third_party/flash-kda) instead of a per-token walk.
    // The pre-activation gate lands bf16 first, the fused core's own
    // `bf16(Σ gp)` landing; FlashKDA adds dt_bias and applies the activation
    // in-kernel, from the same formula. The recurrent state reads the group's
    // parity slab and the commit rows land in the other one: under the
    // chunkwise kernel, parity is a per-segment double buffer, not a
    // per-token flip. A group's speculative tail continues from the state
    // its commit rows just landed and writes its successor back into the
    // now-dead read slab — junk the next round's commit overwrites.
    k3_land_batched_launch(
        ctx,
        b,
        K3_ATTN_INNER,
        K3_ATTN_INNER,
        0,
        1,
        &s.kda_forget_partial,
        &mut s.kda_g,
    )?;
    match cp {
        None => {
            for group in groups {
                let segments = [
                    (group.row, group.commit_rows, group.parity),
                    (
                        group.row + group.commit_rows,
                        group.spec_rows,
                        group.parity ^ usize::from(group.commit_rows > 0),
                    ),
                ];
                // A pure prefill chunk has no speculative tail; a first verify
                // round after prefill has no commit prefix.
                for (row, rows, read_parity) in segments.into_iter().filter(|segment| segment.1 > 0)
                {
                    let (recurrent_read, recurrent_write) =
                        parity_pair(&mut kda_state.recurrent, read_parity);
                    k3_flash_kda_fwd_launch(
                        ctx,
                        rows,
                        K3_HEADS,
                        pegainfer_kernels::ops::K3FlashKdaSpan {
                            row,
                            state_in_row: group.state_row,
                            state_out_row: group.state_row,
                        },
                        &s.conv_q,
                        &s.conv_k,
                        &s.conv_v,
                        &s.kda_g,
                        &s.beta,
                        &mut s.kda_beta_t,
                        &w.a_log,
                        &w.dt_bias,
                        recurrent_read,
                        recurrent_write,
                        &mut s.kda_attn,
                        &mut s.flash_kda_ws,
                        (K3_HEAD_DIM as f32).powf(-0.5),
                        crate::config::K3_KDA_GATE_LOWER_BOUND as f32,
                    )?;
                }
            }
        }
        Some(cp) => {
            // KCP: the recurrence is affine in the state, so the segment is
            // `S_out = M·S_in + D`. Export `(M, D)` by running the same
            // forward with doctored operands (`v = 0` from identity gives
            // `M`; real `v` from zero gives `D`), exchange, fold the
            // upstream packages, and run the real forward from the true
            // input state. Rank 0's real forward doubles as its package.
            let group = groups[0];
            let rows = shape.live_rows;
            let scale = (K3_HEAD_DIM as f32).powf(-0.5);
            let lower_bound = crate::config::K3_KDA_GATE_LOWER_BOUND as f32;
            let K3Scratch {
                conv_q,
                conv_k,
                conv_v,
                kda_g,
                beta,
                kda_beta_t,
                kda_attn,
                flash_kda_ws,
                ..
            } = s;
            let mut forward = |v: &CudaSlice<bf16>,
                               state_in: &CudaSlice<f32>,
                               state_out: &mut CudaSlice<f32>|
             -> Result<()> {
                k3_flash_kda_fwd_launch(
                    ctx,
                    rows,
                    K3_HEADS,
                    pegainfer_kernels::ops::K3FlashKdaSpan::default(),
                    conv_q,
                    conv_k,
                    v,
                    kda_g,
                    beta,
                    kda_beta_t,
                    &w.a_log,
                    &w.dt_bias,
                    state_in,
                    state_out,
                    kda_attn,
                    flash_kda_ws,
                    scale,
                    lower_bound,
                )
            };
            if cp.cp_rank == 0 {
                // The true initial state is the zeroed slab: one real forward
                // is both the answer and the package (its final state IS
                // `D_0`, and downstream never needs `M_0`).
                let (recurrent_read, recurrent_write) =
                    parity_pair(&mut kda_state.recurrent, group.parity);
                forward(&*conv_v, recurrent_read, recurrent_write)?;
                copy_rows(
                    ctx,
                    &kda_state.recurrent[group.parity ^ 1],
                    0,
                    &mut cp.kda_d,
                    0,
                    1,
                    K3_KDA_STATE,
                )?;
            } else if cp.cp_rank + 1 < cp.cp_size {
                forward(&cp.zero_v, &cp.identity, &mut cp.kda_m)?;
                forward(&*conv_v, &cp.zero_state, &mut cp.kda_d)?;
            }
            let cp_group = cp.group.clone();
            let cp_sync = cp.window_sync(K3CpWindowKind::Upstream);
            cp_group.exchange(ctx, &cp_sync, || {
                for j in 0..cp.cp_rank {
                    k3_cp_copy_in(
                        ctx,
                        cp.peers[j].kda_d,
                        0,
                        &mut cp.recv_d,
                        j * K3_KDA_STATE,
                        K3_KDA_STATE,
                    )?;
                    if j > 0 {
                        k3_cp_copy_in(
                            ctx,
                            cp.peers[j].kda_m,
                            0,
                            &mut cp.recv_m,
                            j * K3_KDA_STATE,
                            K3_KDA_STATE,
                        )?;
                    }
                }
                Ok(())
            })?;
            if cp.cp_rank > 0 {
                let merged = cp.merge_upstream(ctx)?;
                copy_rows(
                    ctx,
                    merged,
                    0,
                    &mut kda_state.recurrent[group.parity],
                    0,
                    1,
                    K3_KDA_STATE,
                )?;
                let (recurrent_read, recurrent_write) =
                    parity_pair(&mut kda_state.recurrent, group.parity);
                forward(&*conv_v, recurrent_read, recurrent_write)?;
            }
        }
    }
    // o_norm × output gate — the fused core's tail as its own batched launch;
    // rows past the chunk carry stale data the step discards with the rest of
    // the padding.
    k3_o_norm_gate_batched_launch(
        ctx,
        b,
        K3_HEADS,
        K3_HEAD_DIM,
        &s.kda_attn,
        &s.out_gate,
        &w.gamma_o,
        &mut s.gated,
    )?;

    // Batched output projection; rows past the chunk carry stale data the
    // step discards with the rest of the padding.
    k3_gemm_full(ctx, &w.w_o, &s.gated, b, &mut s.hidden_partial)?;
    k3_land_batched_launch(
        ctx,
        b,
        K3_HIDDEN,
        K3_HIDDEN,
        0,
        1,
        &s.hidden_partial,
        &mut s.attn_out,
    )
}

/// One q/k/v stream of a prefill chunk or verify step: the batched band
/// projection, the per-group window builds, one batched convolution, and each
/// group's carry into its next segment.
#[allow(clippy::too_many_arguments)]
fn kda_conv_stream_chunk(
    ctx: &DeviceContext,
    b: usize,
    groups: &[K3KdaGroup],
    fused: &DeviceMatrix,
    band: usize,
    taps: &CudaSlice<f32>,
    conv_state: &mut [[CudaSlice<bf16>; 3]; 2],
    stream_index: usize,
    normed: &CudaSlice<bf16>,
    partial: &mut CudaSlice<f32>,
    xs: &mut CudaSlice<bf16>,
    window: &mut CudaSlice<bf16>,
    window_next: &mut CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    let inner = K3_ATTN_INNER;
    let window_row = K3_CONV_STATE * inner;
    k3_gemm_partial(
        ctx,
        fused,
        band,
        inner,
        normed,
        b,
        partial,
        K3PartialSpan::whole(inner),
    )?;
    // Land the step's inputs once: the window entries ARE these bf16 rows,
    // the same cast the conv kernel itself applies.
    k3_land_batched_launch(ctx, b, inner, inner, 0, 1, partial, xs)?;
    // Row t of a group's window slot j holds the group's input `t -
    // K3_CONV_STATE + j`: from the segment itself once that token exists,
    // from the slot's carried window before it. Rows outside every group
    // keep stale windows; their conv output is padding and is discarded.
    for group in groups {
        let tokens = group.commit_rows + group.spec_rows;
        let carry = &conv_state[group.parity][stream_index];
        for j in 0..K3_CONV_STATE {
            let lead = K3_CONV_STATE - j;
            if tokens > lead {
                copy_rows_2d(
                    ctx,
                    xs,
                    group.row * inner,
                    inner,
                    window,
                    ((group.row + lead) * K3_CONV_STATE + j) * inner,
                    window_row,
                    tokens - lead,
                    inner,
                )?;
            }
            for t in 0..lead.min(tokens) {
                copy_rows_2d(
                    ctx,
                    carry,
                    (group.state_row * K3_CONV_STATE + t + j) * inner,
                    inner,
                    window,
                    ((group.row + t) * K3_CONV_STATE + j) * inner,
                    inner,
                    1,
                    inner,
                )?;
            }
        }
    }
    k3_conv_silu_batched_launch(
        ctx,
        b,
        inner,
        K3_CONV_WIDTH,
        1,
        partial,
        taps,
        window,
        xs,
        out,
        window_next,
    )?;
    // A group's carry into its next segment is the successor window of its
    // last COMMIT row — for a segment shorter than the window it already
    // folds the slots carried in above. It lands in the other parity slab,
    // agreeing with the recurrent state's per-segment double buffering. The
    // speculative tail's successors are never carried: its tokens replay as
    // the next round's commit rows.
    for group in groups.iter().filter(|group| group.commit_rows > 0) {
        copy_rows_2d(
            ctx,
            window_next,
            (group.row + group.commit_rows - 1) * window_row,
            window_row,
            &mut conv_state[group.parity ^ 1][stream_index],
            group.state_row * window_row,
            window_row,
            1,
            window_row,
        )?;
    }
    Ok(())
}

/// One prefill chunk's MLA attention over FlashMLA's dense FMHA.
///
/// The workspace covers the whole context (`max_ctx` rows), so the vLLM
/// context loop degenerates to a single call: `t_q = live_rows` queries from
/// `s.query` attend `t_kv = chunk_start + live_rows` keys expanded from the
/// gathered latent, and `CausalMask<false>`'s Q-at-the-end alignment gives
/// chunk token `i` exactly `chunk_start + i + 1` visible keys. Prefill drives
/// slot 0, whose block-table row leads `table_dev`.
pub(super) fn mla_attention_chunk_fmha(
    ctx: &DeviceContext,
    shape: K3StepShape,
    w: &K3MlaWeights,
    kv: &K3PagedKv,
    mla_index: usize,
    s: &mut K3Scratch,
) -> Result<()> {
    let t_q = shape.live_rows;
    let t_kv = shape.chunk_start + t_q;
    ensure!(
        t_kv * K3_KV_LORA_RANK <= s.mla_ctx_latent.data.len(),
        "K3 MLA prefill workspace of {} tokens cannot span the {t_kv}-token context",
        s.mla_ctx_latent.data.len() / K3_KV_LORA_RANK
    );
    k3_mla_prefill_gather_launch(
        ctx,
        t_kv,
        &kv.slab,
        &kv.table_dev,
        kv.page_stride(),
        mla_index * K3_KV_PAGE_TOKENS * K3_MLA_LATENT_ROW,
        &mut s.mla_ctx_latent.data,
        &mut s.mla_ctx_rope,
    )?;
    mla_chunk_attend(ctx, t_q, t_kv, w, s)
}

/// The CP variant of one chunk's MLA attention: the paged gather is replaced
/// by assembly from the gang's published post-norm latents. Each rank
/// publishes its own segment's `kv_norm`/`rope` rows, copies its upstream
/// peers' rows into the context scratch at their global offsets, lands its
/// own rows from local scratch, and runs the same bottom-right-aligned FMHA
/// at `t_kv = seg_start + seg_len` — causality falls out of the alignment
/// exactly as for a local chunk.
pub(super) fn mla_attention_chunk_cp(
    ctx: &DeviceContext,
    shape: K3StepShape,
    w: &K3MlaWeights,
    kv: &mut K3PagedKv,
    mla_index: usize,
    s: &mut K3Scratch,
    cp: &mut K3CpScratch,
) -> Result<()> {
    let t_q = shape.live_rows;
    let t_kv = shape.chunk_start + t_q;
    let (seg_start, seg_len) = cp.segments[cp.cp_rank];
    ensure!(
        seg_len == t_q && seg_start == shape.chunk_start,
        "K3 CP MLA step shape ({}+{t_q}) disagrees with the rank's segment ({seg_start}+{seg_len})",
        shape.chunk_start
    );
    copy_rows(
        ctx,
        &s.kv_norm,
        0,
        &mut cp.mla_latent_pub,
        0,
        t_q,
        K3_KV_LORA_RANK,
    )?;
    copy_rows(
        ctx,
        &s.rope,
        0,
        &mut cp.mla_rope_pub,
        0,
        t_q,
        crate::config::K3_QK_ROPE_HEAD_DIM,
    )?;
    let group = cp.group.clone();
    let sync = cp.window_sync(K3CpWindowKind::Upstream);
    group.exchange(ctx, &sync, || {
        for j in 0..cp.cp_rank {
            let (peer_start, peer_len) = cp.segments[j];
            k3_cp_copy_in(
                ctx,
                cp.peers[j].mla_latent,
                0,
                &mut s.mla_ctx_latent.data,
                peer_start * K3_KV_LORA_RANK,
                peer_len * K3_KV_LORA_RANK,
            )?;
            k3_cp_copy_in(
                ctx,
                cp.peers[j].mla_rope,
                0,
                &mut s.mla_ctx_rope,
                peer_start * crate::config::K3_QK_ROPE_HEAD_DIM,
                peer_len * crate::config::K3_QK_ROPE_HEAD_DIM,
            )?;
        }
        Ok(())
    })?;
    copy_rows(
        ctx,
        &s.kv_norm,
        0,
        &mut s.mla_ctx_latent.data,
        seg_start,
        t_q,
        K3_KV_LORA_RANK,
    )?;
    copy_rows(
        ctx,
        &s.rope,
        0,
        &mut s.mla_ctx_rope,
        seg_start,
        t_q,
        crate::config::K3_QK_ROPE_HEAD_DIM,
    )?;
    // The owner rank persists the upstream context it just assembled into its
    // paged pool so decode after the CP handoff attends fully locally. The
    // upstream rows sit at 0..upstream_len of the assembled context, exactly
    // the layout `append_latent` reads.
    if cp.upstream_len > 0 {
        kv.append_latent(
            ctx,
            mla_index,
            cp.upstream_len,
            &cp.upstream_kv_rows,
            &s.mla_ctx_latent.data,
            &s.mla_ctx_rope,
        )?;
    }
    mla_chunk_attend(ctx, t_q, t_kv, w, s)
}

/// The shared tail of a chunk's MLA attention: kv_b expansion of the
/// assembled `[t_kv]` context latents and one causal Q-at-the-end FMHA.
fn mla_chunk_attend(
    ctx: &DeviceContext,
    t_q: usize,
    t_kv: usize,
    w: &K3MlaWeights,
    s: &mut K3Scratch,
) -> Result<()> {
    ensure!(
        t_kv * K3_KV_LORA_RANK <= s.mla_ctx_latent.data.len(),
        "K3 MLA prefill workspace of {} tokens cannot span the {t_kv}-token context",
        s.mla_ctx_latent.data.len() / K3_KV_LORA_RANK
    );
    s.mla_ctx_latent.seq_len = t_kv;
    s.mla_ctx_nope_v.seq_len = t_kv;
    gemm_rows_into_checked(
        ctx,
        &w.w_kv_b,
        0,
        K3_KV_B_OUT,
        &s.mla_ctx_latent,
        &mut s.mla_ctx_nope_v,
    )?;
    k3_mla_prefill_expand_k_launch(
        ctx,
        t_kv,
        K3_MLA_HEADS,
        &s.mla_ctx_nope_v.data,
        &s.mla_ctx_rope,
        &mut s.mla_ctx_k,
    )?;
    // The decode kernel reads the softmax scale as a bf16 device scalar; feed
    // the FMHA the same rounded constant so the two paths agree on it.
    let scale = bf16::from_f64(crate::model::k3_mla_scale()).to_f32();
    k3_flash_mla_prefill_fwd_launch(
        ctx,
        t_q,
        t_kv,
        K3_MLA_HEADS,
        &s.query,
        &s.mla_ctx_k,
        &s.mla_ctx_nope_v.data,
        &mut s.attn,
        scale,
    )
}
