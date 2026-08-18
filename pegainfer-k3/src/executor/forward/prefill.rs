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
use super::super::buffers::K3_KDA_WSM_PADDED;
use super::super::buffers::K3KdaState;
use super::super::buffers::K3Scratch;
use super::super::buffers::K3StatePool;
use super::super::buffers::copy_rows_2d;
use super::super::buffers::parity_pair;
use super::super::paged_kv::K3_KV_PAGE_TOKENS;
use super::super::paged_kv::K3_MLA_LATENT_ROW;
use super::super::paged_kv::K3PagedKv;
use super::gemm::K3PartialSpan;
use super::gemm::k3_gemm_full;
use super::gemm::k3_gemm_partial;
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
) -> Result<()> {
    ensure!(
        (1..=shape.bucket).contains(&shape.live_rows),
        "K3 prefill chunk of {} tokens does not fit its {} bucket",
        shape.live_rows,
        shape.bucket
    );
    k3_step(ctx, model, shape, true, state, scratch)
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

/// The KDA layer of a prefill chunk: batched projections, a batched
/// convolution over prebuilt windows, and the delta rule as one chunkwise
/// FlashKDA forward (third_party/flash-kda, MIT, MoonshotAI) — the per-token
/// walk collapses into two launches per layer.
///
/// Numerics: the convolution windows are the landed bf16 inputs themselves
/// (`window[t][j] = x[t - 3 + j]`, carried window for the first rows), so the
/// conv is bitwise against sequential stepping. FlashKDA computes the same
/// delta rule the fused TileLang core spells — same gate formula from the
/// same pre-activation landing, beta sigmoid in-kernel — but with an f32 q/k
/// l2norm chain (the TileLang core mirrors the reference's bf16 chain) and
/// chunkwise accumulation order, and the projections run at the chunk's
/// bucket. Chunked prefill is therefore held to the fixture's noise floor,
/// like any cross-bucket comparison; decode keeps the bit-matched fused core.
pub(super) fn kda_attention_chunk(
    ctx: &DeviceContext,
    shape: K3StepShape,
    layer: &K3LayerWeights,
    w: &K3KdaWeights,
    kda_state: &mut K3KdaState,
    s: &mut K3Scratch,
) -> Result<()> {
    let b = shape.bucket;
    let tokens = shape.live_rows;
    let start_parity = shape.parity;
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

    kda_conv_stream_chunk(
        ctx,
        b,
        tokens,
        start_parity,
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
        tokens,
        start_parity,
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
        tokens,
        start_parity,
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

    // The chunkwise delta rule: the whole chunk through the vendored FlashKDA
    // forward (third_party/flash-kda) — two launches instead of a per-token
    // walk. The pre-activation gate lands bf16 first, the fused core's own
    // `bf16(Σ gp)` landing; FlashKDA adds dt_bias and applies the activation
    // in-kernel, from the same formula. The recurrent state reads the
    // start-parity slab and lands in the other one: under the chunkwise
    // kernel, parity is a per-chunk double buffer, not a per-token flip.
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
    {
        let (recurrent_read, recurrent_write) = parity_pair(&mut kda_state.recurrent, start_parity);
        k3_flash_kda_fwd_launch(
            ctx,
            tokens,
            K3_HEADS,
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

/// One q/k/v stream of a prefill chunk: the batched band projection, the
/// window build, one batched convolution, and the carry into the next chunk.
#[allow(clippy::too_many_arguments)]
fn kda_conv_stream_chunk(
    ctx: &DeviceContext,
    b: usize,
    tokens: usize,
    start_parity: usize,
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
    // Land the chunk's inputs once: the window entries ARE these bf16 rows,
    // the same cast the conv kernel itself applies.
    k3_land_batched_launch(ctx, b, inner, inner, 0, 1, partial, xs)?;
    // Row t's window slot j holds input `t - K3_CONV_STATE + j`: from the
    // chunk itself once that token exists, from the carried window before it.
    {
        let carry = &conv_state[start_parity][stream_index];
        for j in 0..K3_CONV_STATE {
            let lead = K3_CONV_STATE - j;
            if tokens > lead {
                copy_rows_2d(
                    ctx,
                    xs,
                    0,
                    inner,
                    window,
                    (lead * K3_CONV_STATE + j) * inner,
                    K3_CONV_STATE * inner,
                    tokens - lead,
                    inner,
                )?;
            }
            for t in 0..lead.min(tokens) {
                copy_rows_2d(
                    ctx,
                    carry,
                    (t + j) * inner,
                    inner,
                    window,
                    (t * K3_CONV_STATE + j) * inner,
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
    // The carry into the next chunk is the successor window of the chunk's
    // last token — for a chunk shorter than the window it already folds the
    // slots carried in above. It lands in the other parity slab, agreeing
    // with the recurrent state's per-chunk double buffering.
    let end_parity = start_parity ^ 1;
    copy_rows_2d(
        ctx,
        window_next,
        (tokens - 1) * K3_CONV_STATE * inner,
        K3_CONV_STATE * inner,
        &mut conv_state[end_parity][stream_index],
        0,
        K3_CONV_STATE * inner,
        1,
        K3_CONV_STATE * inner,
    )
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
