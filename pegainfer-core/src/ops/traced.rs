use anyhow::Result;
use cudarc::driver::CudaSlice;
use pegainfer_kernels::tensor::Hidden;
use pegainfer_kernels::tensor::InDim;
use pegainfer_kernels::tensor::OutDim;

use crate::ops::call_spec::embedding_batch_call;
use crate::ops::call_spec::fused_add_rms_norm_batch_call;
use crate::ops::call_spec::gemm_call;
use crate::ops::call_spec::gemm_rows_call;
use crate::ops::call_spec::qk_norm_rope_batch_decode_call;
use crate::ops::call_spec::rms_norm_batch_call;
use crate::ops::call_spec::silu_mul_fused_batch_call;
use crate::ops::call_trace;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceMatrix;
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;

pub fn embedding_batch(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_ids_gpu: &CudaSlice<u32>,
    out: &mut HiddenStates,
) -> Result<()> {
    if call_trace::is_enabled() {
        let label = call_trace::current_label("embedding_batch");
        call_trace::record_call(embedding_batch_call(
            label,
            embed.rows,
            embed.cols,
            out.seq_len,
        ));
    }
    pegainfer_kernels::ops::embedding_batch(ctx, embed, token_ids_gpu, out)
}

pub fn rms_norm_batch_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) {
    if call_trace::is_enabled() {
        let label = call_trace::current_label("rms_norm_batch");
        call_trace::record_call(rms_norm_batch_call::<Hidden>(
            label,
            x.hidden_dim,
            x.seq_len,
            eps,
        ));
    }
    pegainfer_kernels::ops::rms_norm_batch_into(ctx, x, weight, eps, out);
}

pub fn gemm_rows_into(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    row_offset: usize,
    num_rows: usize,
    x: &HiddenStates,
    out: &mut HiddenStates,
) {
    gemm_rows_into_checked(ctx, weight, row_offset, num_rows, x, out)
        .expect("GEMM row-range launch failed");
}

pub fn gemm_rows_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    row_offset: usize,
    num_rows: usize,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    if call_trace::is_enabled() {
        let label = call_trace::current_label("gemm_rows");
        call_trace::record_call(gemm_rows_call::<OutDim>(
            label,
            weight.rows,
            weight.cols,
            num_rows,
            row_offset,
            x.seq_len,
        ));
    }
    pegainfer_kernels::ops::gemm_rows_into_checked(ctx, weight, row_offset, num_rows, x, out)
}

pub fn gemm_into(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) {
    gemm_into_checked(ctx, weight, x, out).expect("GEMM launch failed");
}

pub fn gemm_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    if call_trace::is_enabled() {
        let label = call_trace::current_label("gemm");
        call_trace::record_call(gemm_call::<OutDim, InDim>(
            label,
            weight.rows,
            weight.cols,
            x.seq_len,
        ));
    }
    pegainfer_kernels::ops::gemm_into_checked(ctx, weight, x, out)
}

pub fn gemm_token_range_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    token_offset: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    if call_trace::is_enabled() {
        let label = call_trace::current_label("gemm_token_range");
        call_trace::record_call(gemm_call::<OutDim, InDim>(
            label,
            weight.rows,
            weight.cols,
            out.seq_len,
        ));
    }
    pegainfer_kernels::ops::gemm_token_range_into_checked(ctx, weight, x, token_offset, out)
}

pub fn qk_norm_rope_batch_decode_into(
    ctx: &DeviceContext,
    q: &mut HiddenStates,
    k: &mut HiddenStates,
    row_offset: usize,
    num_rows: usize,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    positions_d: &CudaSlice<i32>,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
) -> Result<()> {
    if call_trace::is_enabled() {
        let label = call_trace::current_label("qk_norm_rope_batch_decode");
        let rope_seq = cos_cache.len / head_dim;
        call_trace::record_call(qk_norm_rope_batch_decode_call(
            label,
            q.hidden_dim,
            k.hidden_dim,
            num_rows,
            rope_seq,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rms_eps,
        ));
    }
    pegainfer_kernels::ops::qk_norm_rope_batch_decode_into(
        ctx,
        q,
        k,
        row_offset,
        num_rows,
        q_norm_weight,
        k_norm_weight,
        cos_cache,
        sin_cache,
        positions_d,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rms_eps,
    )
}

pub fn fused_add_rms_norm_batch_into(
    ctx: &DeviceContext,
    hidden: &mut HiddenStates,
    residual: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) {
    if call_trace::is_enabled() {
        let label = call_trace::current_label("fused_add_rms_norm_batch");
        call_trace::record_call(fused_add_rms_norm_batch_call::<Hidden>(
            label,
            hidden.hidden_dim,
            hidden.seq_len,
            eps,
        ));
    }
    pegainfer_kernels::ops::fused_add_rms_norm_batch_into(ctx, hidden, residual, weight, eps, out);
}

pub fn silu_mul_fused_batch_into(
    ctx: &DeviceContext,
    gate_up: &HiddenStates,
    out: &mut HiddenStates,
) -> anyhow::Result<()> {
    if call_trace::is_enabled() {
        let label = call_trace::current_label("silu_mul_fused_batch");
        call_trace::record_call(silu_mul_fused_batch_call(
            label,
            out.hidden_dim,
            gate_up.seq_len,
        ));
    }
    pegainfer_kernels::ops::silu_mul_fused_batch_into(ctx, gate_up, out)
}
