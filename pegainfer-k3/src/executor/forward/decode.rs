//! The decode arm of the forward pass: the per-token entry point and the
//! sequential KDA walk the certified engine spells.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::K3_CONV_WIDTH;
use pegainfer_kernels::ops::k3_conv_silu_batched_launch;
use pegainfer_kernels::ops::k3_kda_core_batched_launch;
use pegainfer_kernels::ops::k3_land_batched_launch;
use pegainfer_kernels::ops::k3_rms_norm_rbs_batched_launch;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;

use super::super::buffers::K3_KDA_FUSED;
use super::super::buffers::K3_KDA_WSM_PADDED;
use super::super::buffers::K3Scratch;
use super::super::buffers::K3StatePool;
use super::gemm::K3PartialSpan;
use super::gemm::k3_gemm_full;
use super::gemm::k3_gemm_partial;
use super::step::K3StepMode;
use super::step::K3StepShape;
use super::step::k3_step;
use crate::config::K3_ATTN_INNER;
use crate::config::K3_HEAD_DIM;
use crate::config::K3_HEADS;
use crate::config::K3_HIDDEN;
use crate::model::K3KdaWeights;
use crate::model::K3LayerWeights;
use crate::model::K3RankModel;

/// Advance every row of the bucket by one token and leave the sampled ids in
/// `scratch.argmax_indices`.
///
/// Reads `scratch.token_ids`, `scratch.context_len` and `scratch.kv_row`;
/// the caller fills those before the step (or before the graph replay).
///
/// Every MoE layer issues the same launches in the same order on every rank —
/// including a step whose batch is empty — which is what lets an
/// expert-parallel group run without a coordinator.
pub(crate) fn k3_decode_step(
    ctx: &DeviceContext,
    model: &K3RankModel,
    shape: K3StepShape,
    state: &mut K3StatePool,
    scratch: &mut K3Scratch,
) -> Result<()> {
    k3_step(
        ctx,
        model,
        shape,
        K3StepMode::Decode,
        state,
        scratch,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn kda_attention(
    ctx: &DeviceContext,
    b: usize,
    layer: &K3LayerWeights,
    w: &K3KdaWeights,
    recurrent_read: &CudaSlice<f32>,
    recurrent_write: &mut CudaSlice<f32>,
    conv_read: &[CudaSlice<bf16>; 3],
    conv_write: &mut [CudaSlice<bf16>; 3],
    s: &mut K3Scratch,
) -> Result<()> {
    k3_rms_norm_rbs_batched_launch(
        ctx,
        b,
        K3_HIDDEN,
        &s.mixed,
        &layer.gamma_in.data,
        &mut s.normed,
    )?;

    // The output-gate quarter of the fused q|k|v|gate projection, placed where
    // the certified `(4 * inner, inner, 3 * inner)` landing span expects it.
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

    kda_conv_stream(
        ctx,
        b,
        &w.wbig,
        0,
        &w.cw_q,
        &conv_read[0],
        &mut conv_write[0],
        &s.normed,
        &mut s.kda_conv_partial,
        &mut s.conv_x,
        &mut s.conv_q,
    )?;
    kda_conv_stream(
        ctx,
        b,
        &w.wbig,
        K3_ATTN_INNER,
        &w.cw_k,
        &conv_read[1],
        &mut conv_write[1],
        &s.normed,
        &mut s.kda_conv_partial,
        &mut s.conv_x,
        &mut s.conv_k,
    )?;
    kda_conv_stream(
        ctx,
        b,
        &w.wbig,
        2 * K3_ATTN_INNER,
        &w.cw_v,
        &conv_read[2],
        &mut conv_write[2],
        &s.normed,
        &mut s.kda_conv_partial,
        &mut s.conv_x,
        &mut s.conv_v,
    )?;

    k3_kda_core_batched_launch(
        ctx,
        b,
        K3_HEADS,
        K3_HEAD_DIM,
        1,
        &s.conv_q,
        &s.conv_k,
        &s.conv_v,
        &s.kda_forget_partial,
        &w.dt_bias,
        &w.a_log,
        &s.beta,
        &s.out_gate,
        &w.gamma_o,
        recurrent_read,
        recurrent_write,
        &mut s.gated,
    )?;
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

/// One q/k/v stream: its band of the fused projection, then the window.
///
/// The band goes to a partial of its own rather than into the fused one: the
/// convolution reads a dense `[rows, inner]` partial, and slicing a column
/// block out of the fused product would cost a strided f32 copy per stream.
#[allow(clippy::too_many_arguments)]
fn kda_conv_stream(
    ctx: &DeviceContext,
    b: usize,
    fused: &DeviceMatrix,
    band: usize,
    taps: &CudaSlice<f32>,
    window_read: &CudaSlice<bf16>,
    window_write: &mut CudaSlice<bf16>,
    normed: &CudaSlice<bf16>,
    partial: &mut CudaSlice<f32>,
    landed: &mut CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    k3_gemm_partial(
        ctx,
        fused,
        band,
        K3_ATTN_INNER,
        normed,
        b,
        partial,
        K3PartialSpan::whole(K3_ATTN_INNER),
    )?;
    k3_conv_silu_batched_launch(
        ctx,
        b,
        K3_ATTN_INNER,
        K3_CONV_WIDTH,
        1,
        partial,
        taps,
        window_read,
        landed,
        out,
        window_write,
    )
}
