use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;

use crate::config::GDN_AOT_KEY_HEAD_DIM;
use crate::config::GDN_AOT_VALUE_HEAD_DIM;
use crate::config::LINEAR_CONV_MAX_KERNEL_DIM;
use crate::ffi;
use crate::prefill_buffers::GdnPrepareScratch35;
use crate::prefill_buffers::GdrChunkwiseScratch35;

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_rule_decode_vec_into(
    ctx: &DeviceContext,
    qkv: &DeviceVec,
    b_proj: &DeviceVec,
    a_proj: &DeviceVec,
    dt_bias: &DeviceVec,
    a_log: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    output: &mut DeviceVec,
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    val_dim: usize,
) {
    let (qkv_ptr, _gq) = qkv.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b_proj.data.device_ptr(&ctx.stream);
    let (a_ptr, _ga) = a_proj.data.device_ptr(&ctx.stream);
    let (dt_ptr, _gdt) = dt_bias.data.device_ptr(&ctx.stream);
    let (alog_ptr, _gal) = a_log.device_ptr(&ctx.stream);
    let (s_ptr, _gs) = state.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::gated_delta_rule_decode_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_ptr as *const ffi::Half,
            alog_ptr as *const f32,
            s_ptr as *mut f32,
            o_ptr as *mut ffi::Half,
            num_key_heads as i32,
            num_value_heads as i32,
            key_dim as i32,
            val_dim as i32,
            ctx.stream.cu_stream(),
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_rule_decode_batch_into(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    b_proj: &HiddenStates,
    a_proj: &HiddenStates,
    dt_bias: &DeviceVec,
    a_log: &CudaSlice<f32>,
    state_ptrs: &CudaSlice<u64>,
    output: &mut HiddenStates,
    batch_size: usize,
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    val_dim: usize,
) {
    assert_eq!(qkv.seq_len, batch_size);
    assert_eq!(b_proj.seq_len, batch_size);
    assert_eq!(a_proj.seq_len, batch_size);
    assert_eq!(output.seq_len, batch_size);
    assert_eq!(b_proj.hidden_dim, num_value_heads);
    assert_eq!(a_proj.hidden_dim, num_value_heads);
    assert_eq!(output.hidden_dim, num_value_heads * val_dim);
    assert_eq!(key_dim, GDN_AOT_KEY_HEAD_DIM);
    assert_eq!(val_dim, GDN_AOT_VALUE_HEAD_DIM);
    assert!(state_ptrs.len() >= batch_size);

    let (qkv_ptr, _gq) = qkv.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b_proj.data.device_ptr(&ctx.stream);
    let (a_ptr, _ga) = a_proj.data.device_ptr(&ctx.stream);
    let (dt_ptr, _gdt) = dt_bias.data.device_ptr(&ctx.stream);
    let (alog_ptr, _gal) = a_log.device_ptr(&ctx.stream);
    let (state_ptrs, _gsp) = state_ptrs.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::gated_delta_rule_decode_batch_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_ptr as *const ffi::Half,
            alog_ptr as *const f32,
            state_ptrs as *const u64,
            o_ptr as *mut ffi::Half,
            batch_size as i32,
            num_key_heads as i32,
            num_value_heads as i32,
            key_dim as i32,
            val_dim as i32,
            ctx.stream.cu_stream(),
        );
    }
}

pub(crate) fn conv1d_decode_batch_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    conv_weight: &DeviceVec,
    conv_state_ptrs: &CudaSlice<u64>,
    out: &mut HiddenStates,
    kernel_size: usize,
) {
    let batch_size = x.seq_len;
    let num_channels = x.hidden_dim;
    assert_eq!(out.hidden_dim, num_channels);
    assert_eq!(out.seq_len, batch_size);
    assert_eq!(conv_weight.len, num_channels * kernel_size);
    assert!(kernel_size <= LINEAR_CONV_MAX_KERNEL_DIM);
    assert!(conv_state_ptrs.len() >= batch_size);

    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = conv_weight.data.device_ptr(&ctx.stream);
    let (s_ptrs, _gs) = conv_state_ptrs.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::conv1d_decode_batch_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            s_ptrs as *const u64,
            o_ptr as *mut ffi::Half,
            num_channels as i32,
            batch_size as i32,
            kernel_size as i32,
            ctx.stream.cu_stream(),
        );
    }
}

/// Causal depthwise conv1d prefill over a HiddenStates batch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_prefill_batch_into(
    ctx: &DeviceContext,
    x_seq: &HiddenStates,
    conv_weight: &DeviceVec,
    conv_state: &mut DeviceVec,
    out_seq: &mut HiddenStates,
    kernel_size: usize,
) {
    let num_channels = x_seq.hidden_dim;
    assert_eq!(out_seq.hidden_dim, num_channels);
    assert_eq!(out_seq.seq_len, x_seq.seq_len);
    assert_eq!(conv_weight.len, num_channels * kernel_size);
    assert_eq!(conv_state.len, num_channels * (kernel_size - 1));

    let (x_ptr, _gx) = x_seq.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = conv_weight.data.device_ptr(&ctx.stream);
    let (s_ptr, _gs) = conv_state.data.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = out_seq.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::conv1d_prefill_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            s_ptr as *mut ffi::Half,
            o_ptr as *mut ffi::Half,
            num_channels as i32,
            x_seq.seq_len as i32,
            kernel_size as i32,
            ctx.stream.cu_stream(),
        );
    }
}

/// Prepare native Q/K/V plus per-token alpha/beta for the FlashInfer GDN
/// candidate.
///
/// The preparation kernel reports non-finite inputs through a sticky device
/// status word owned by the chunk. The caller validates it once after the
/// layer loop, avoiding one D2H synchronization per layer while still refusing
/// to return a candidate result containing invalid inputs.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_rule_prefill_native_prepare_into(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    b_proj: &HiddenStates,
    a_proj: &HiddenStates,
    dt_bias: &DeviceVec,
    a_log: &CudaSlice<f32>,
    scratch: &mut GdnPrepareScratch35,
    h_q: usize,
    h_k: usize,
    h_v: usize,
    head_dim: usize,
) -> Result<()> {
    anyhow::ensure!(
        matches!((h_q, h_k, h_v, head_dim), (16, 16, 32 | 48, 128)),
        "native GDN prepare supports Hq/Hk/Hv/D=16/16/{{32,48}}/128, got {h_q}/{h_k}/{h_v}/{head_dim}"
    );
    anyhow::ensure!(qkv.seq_len > 0, "native GDN prepare requires T>=1");
    let expected_qkv = (h_q + h_k + h_v) * head_dim;
    anyhow::ensure!(
        qkv.hidden_dim == expected_qkv,
        "native GDN qkv hidden dim mismatch: expected {expected_qkv}, got {}",
        qkv.hidden_dim
    );
    anyhow::ensure!(
        b_proj.hidden_dim == h_v && b_proj.seq_len == qkv.seq_len,
        "native GDN b projection must be [T,Hv]=[{},{}]",
        qkv.seq_len,
        h_v
    );
    anyhow::ensure!(
        a_proj.hidden_dim == h_v && a_proj.seq_len == qkv.seq_len,
        "native GDN a projection must be [T,Hv]=[{},{}]",
        qkv.seq_len,
        h_v
    );
    anyhow::ensure!(
        dt_bias.len == h_v,
        "native GDN dt_bias length must be {h_v}, got {}",
        dt_bias.len
    );
    anyhow::ensure!(
        a_log.len() == h_v,
        "native GDN A_log length must be {h_v}, got {}",
        a_log.len()
    );
    anyhow::ensure!(
        scratch.q.hidden_dim == h_q * head_dim && scratch.q.seq_len == qkv.seq_len,
        "native GDN Q output shape mismatch"
    );
    anyhow::ensure!(
        scratch.k.hidden_dim == h_k * head_dim && scratch.k.seq_len == qkv.seq_len,
        "native GDN K output shape mismatch"
    );
    anyhow::ensure!(
        scratch.v.hidden_dim == h_v * head_dim && scratch.v.seq_len == qkv.seq_len,
        "native GDN V output shape mismatch"
    );
    anyhow::ensure!(
        scratch.alpha.len() == qkv.seq_len * h_v,
        "native GDN alpha output length mismatch"
    );
    anyhow::ensure!(
        scratch.beta.len() == qkv.seq_len * h_v,
        "native GDN beta output length mismatch"
    );
    anyhow::ensure!(
        scratch.non_finite_status.len() == 1,
        "native GDN status output length mismatch"
    );

    {
        let (qkv_ptr, _gqkv) = qkv.data.device_ptr(&ctx.stream);
        let (b_ptr, _gb) = b_proj.data.device_ptr(&ctx.stream);
        let (a_ptr, _ga) = a_proj.data.device_ptr(&ctx.stream);
        let (dt_ptr, _gdt) = dt_bias.data.device_ptr(&ctx.stream);
        let (alog_ptr, _gal) = a_log.device_ptr(&ctx.stream);
        let (q_out, _gqo) = scratch.q.data.device_ptr_mut(&ctx.stream);
        let (k_out, _gko) = scratch.k.data.device_ptr_mut(&ctx.stream);
        let (v_out, _gvo) = scratch.v.data.device_ptr_mut(&ctx.stream);
        let (alpha_out, _gaout) = scratch.alpha.device_ptr_mut(&ctx.stream);
        let (beta_out, _gbout) = scratch.beta.device_ptr_mut(&ctx.stream);
        let (status_out, _gsout) = scratch.non_finite_status.device_ptr_mut(&ctx.stream);

        let result = unsafe {
            ffi::gated_delta_rule_prefill_native_prepare_cuda(
                qkv_ptr as *const ffi::Half,
                b_ptr as *const ffi::Half,
                a_ptr as *const ffi::Half,
                dt_ptr as *const ffi::Half,
                alog_ptr as *const f32,
                q_out as *mut ffi::Half,
                k_out as *mut ffi::Half,
                v_out as *mut ffi::Half,
                alpha_out as *mut f32,
                beta_out as *mut f32,
                status_out as *mut u32,
                h_q as i32,
                h_k as i32,
                h_v as i32,
                head_dim as i32,
                qkv.hidden_dim as i32,
                qkv.seq_len as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_prefill_chunk_prepare_into(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    b_proj: &HiddenStates,
    a_proj: &HiddenStates,
    dt_bias: &DeviceVec,
    a_log: &CudaSlice<f32>,
    q_out: &mut HiddenStates,
    k_out: &mut HiddenStates,
    v_out: &mut HiddenStates,
    g_out: &mut CudaSlice<f32>,
    beta_out: &mut CudaSlice<f32>,
    num_key_heads: usize,
    num_value_heads: usize,
) -> Result<()> {
    let (qkv_ptr, _gqkv) = qkv.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b_proj.data.device_ptr(&ctx.stream);
    let (a_ptr, _ga) = a_proj.data.device_ptr(&ctx.stream);
    let (dt_ptr, _gdt) = dt_bias.data.device_ptr(&ctx.stream);
    let (alog_ptr, _gal) = a_log.device_ptr(&ctx.stream);
    let (q_out_ptr, _gqo) = q_out.data.device_ptr_mut(&ctx.stream);
    let (k_out_ptr, _gko) = k_out.data.device_ptr_mut(&ctx.stream);
    let (v_out_ptr, _gvo) = v_out.data.device_ptr_mut(&ctx.stream);
    let (g_out_ptr, _ggo) = g_out.device_ptr_mut(&ctx.stream);
    let (beta_out_ptr, _gbetao) = beta_out.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_prepare_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_ptr as *const ffi::Half,
            alog_ptr as *const f32,
            q_out_ptr as *mut ffi::Half,
            k_out_ptr as *mut ffi::Half,
            v_out_ptr as *mut ffi::Half,
            g_out_ptr as *mut f32,
            beta_out_ptr as *mut f32,
            num_key_heads as i32,
            num_value_heads as i32,
            qkv.hidden_dim as i32,
            qkv.seq_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

fn gated_delta_rule_prefill_chunk_cumsum_inplace(
    ctx: &DeviceContext,
    g_cumsum: &mut CudaSlice<f32>,
    seq_len: usize,
    num_value_heads: usize,
) -> Result<()> {
    let (g_ptr, _gg) = g_cumsum.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_cumsum_cuda(
            g_ptr as *const f32,
            g_ptr as *mut f32,
            seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

fn gated_delta_rule_prefill_chunk_a_into(
    ctx: &DeviceContext,
    k: &HiddenStates,
    g_cumsum: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    a_tril: &mut CudaSlice<f32>,
    num_value_heads: usize,
) -> Result<()> {
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);
    let (beta_ptr, _gb) = beta.device_ptr(&ctx.stream);
    let (a_ptr, _ga) = a_tril.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_a_cuda(
            k_ptr as *const ffi::Half,
            g_ptr as *const f32,
            beta_ptr as *const f32,
            a_ptr as *mut f32,
            k.seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

fn gated_delta_rule_prefill_chunk_solve_into(
    ctx: &DeviceContext,
    a_tril: &CudaSlice<f32>,
    a_inv: &mut CudaSlice<half::bf16>,
    seq_len: usize,
    num_value_heads: usize,
) -> Result<()> {
    let (a_ptr, _ga) = a_tril.device_ptr(&ctx.stream);
    let (ai_ptr, _gai) = a_inv.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_solve_cuda(
            a_ptr as *const f32,
            ai_ptr as *mut ffi::Half,
            seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_prefill_chunk_recompute_into(
    ctx: &DeviceContext,
    k: &HiddenStates,
    v: &HiddenStates,
    beta: &CudaSlice<f32>,
    w: &mut HiddenStates,
    u: &mut HiddenStates,
    a_inv: &CudaSlice<half::bf16>,
    g_cumsum: &CudaSlice<f32>,
    num_value_heads: usize,
) -> Result<()> {
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v.data.device_ptr(&ctx.stream);
    let (beta_ptr, _gb) = beta.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = w.data.device_ptr_mut(&ctx.stream);
    let (u_ptr, _gu) = u.data.device_ptr_mut(&ctx.stream);
    let (ai_ptr, _gai) = a_inv.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_recompute_cuda(
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            beta_ptr as *const f32,
            w_ptr as *mut ffi::Half,
            u_ptr as *mut ffi::Half,
            ai_ptr as *const ffi::Half,
            g_ptr as *const f32,
            k.seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_prefill_chunk_state_stage_into(
    ctx: &DeviceContext,
    k: &HiddenStates,
    w: &HiddenStates,
    u: &HiddenStates,
    g_cumsum: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    chunk_state: &mut CudaSlice<f32>,
    v_new: &mut HiddenStates,
    num_value_heads: usize,
) -> Result<()> {
    assert_eq!(k.hidden_dim, w.hidden_dim);
    assert_eq!(u.hidden_dim, v_new.hidden_dim);
    assert_eq!(k.seq_len, w.seq_len);
    assert_eq!(k.seq_len, u.seq_len);
    assert_eq!(k.seq_len, v_new.seq_len);

    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = w.data.device_ptr(&ctx.stream);
    let (u_ptr, _gu) = u.data.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);
    let (s_ptr, _gs) = state.device_ptr_mut(&ctx.stream);
    let (cs_ptr, _gcs) = chunk_state.device_ptr_mut(&ctx.stream);
    let (vn_ptr, _gvn) = v_new.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_state_cuda(
            k_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            u_ptr as *const ffi::Half,
            g_ptr as *const f32,
            s_ptr as *const f32,
            cs_ptr as *mut f32,
            vn_ptr as *mut ffi::Half,
            s_ptr as *mut f32,
            k.seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_prefill_chunk_o_stage_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v_new: &HiddenStates,
    chunk_state: &CudaSlice<f32>,
    g_cumsum: &CudaSlice<f32>,
    output: &mut HiddenStates,
    num_value_heads: usize,
    scale: f32,
) -> Result<()> {
    assert_eq!(q.hidden_dim, k.hidden_dim);
    assert_eq!(v_new.hidden_dim, output.hidden_dim);
    assert_eq!(q.seq_len, k.seq_len);
    assert_eq!(q.seq_len, v_new.seq_len);
    assert_eq!(q.seq_len, output.seq_len);

    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (vn_ptr, _gvn) = v_new.data.device_ptr(&ctx.stream);
    let (cs_ptr, _gcs) = chunk_state.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_o_cuda(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            vn_ptr as *const ffi::Half,
            cs_ptr as *const f32,
            g_ptr as *const f32,
            o_ptr as *mut ffi::Half,
            q.seq_len as i32,
            num_value_heads as i32,
            scale,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Chunk-wise GDR prefill operator contract for Qwen3.5.
///
/// The chunk-wise path is an explicit multi-stage operator with pre-allocated
/// scratch instead of one opaque kernel launch.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_rule_prefill_chunkwise_into(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    b_proj: &HiddenStates,
    a_proj: &HiddenStates,
    dt_bias: &DeviceVec,
    a_log: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    scratch: &mut GdrChunkwiseScratch35,
    output: &mut HiddenStates,
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    val_dim: usize,
) -> Result<()> {
    assert_eq!(scratch.q_expanded.seq_len, qkv.seq_len);
    assert_eq!(scratch.k_expanded.seq_len, qkv.seq_len);
    assert_eq!(scratch.v_raw.seq_len, qkv.seq_len);
    assert_eq!(scratch.w.seq_len, qkv.seq_len);
    assert_eq!(scratch.u.seq_len, qkv.seq_len);
    assert_eq!(scratch.v_new.seq_len, qkv.seq_len);
    assert_eq!(scratch.q_expanded.hidden_dim, num_value_heads * key_dim);
    assert_eq!(scratch.k_expanded.hidden_dim, num_value_heads * key_dim);
    assert_eq!(scratch.v_raw.hidden_dim, num_value_heads * val_dim);
    assert_eq!(scratch.w.hidden_dim, num_value_heads * key_dim);
    assert_eq!(scratch.u.hidden_dim, num_value_heads * val_dim);
    assert_eq!(scratch.v_new.hidden_dim, num_value_heads * val_dim);

    let expected_gate_len = qkv.seq_len * num_value_heads;
    let expected_chunk_a_len = qkv.seq_len * num_value_heads * GdrChunkwiseScratch35::CHUNK_SIZE;
    let expected_chunk_ai_len = expected_chunk_a_len;
    let expected_chunk_state_len =
        GdrChunkwiseScratch35::num_chunks(qkv.seq_len) * num_value_heads * val_dim * key_dim;
    assert_eq!(scratch.g_cumsum.len(), expected_gate_len);
    assert_eq!(scratch.beta.len(), expected_gate_len);
    assert_eq!(scratch.a_tril.len(), expected_chunk_a_len);
    assert_eq!(scratch.a_inv.len(), expected_chunk_ai_len);
    assert_eq!(scratch.chunk_state.len(), expected_chunk_state_len);

    gated_delta_rule_prefill_chunk_prepare_into(
        ctx,
        qkv,
        b_proj,
        a_proj,
        dt_bias,
        a_log,
        &mut scratch.q_expanded,
        &mut scratch.k_expanded,
        &mut scratch.v_raw,
        &mut scratch.g_cumsum,
        &mut scratch.beta,
        num_key_heads,
        num_value_heads,
    )
    .map_err(|e| anyhow::anyhow!("GDR prefill prepare failed: {e}"))?;
    gated_delta_rule_prefill_chunk_cumsum_inplace(
        ctx,
        &mut scratch.g_cumsum,
        qkv.seq_len,
        num_value_heads,
    )
    .map_err(|e| anyhow::anyhow!("GDR prefill cumsum failed: {e}"))?;
    gated_delta_rule_prefill_chunk_a_into(
        ctx,
        &scratch.k_expanded,
        &scratch.g_cumsum,
        &scratch.beta,
        &mut scratch.a_tril,
        num_value_heads,
    )
    .map_err(|e| anyhow::anyhow!("GDR prefill A stage failed: {e}"))?;
    gated_delta_rule_prefill_chunk_solve_into(
        ctx,
        &scratch.a_tril,
        &mut scratch.a_inv,
        qkv.seq_len,
        num_value_heads,
    )
    .map_err(|e| anyhow::anyhow!("GDR prefill solve failed: {e}"))?;
    gated_delta_rule_prefill_chunk_recompute_into(
        ctx,
        &scratch.k_expanded,
        &scratch.v_raw,
        &scratch.beta,
        &mut scratch.w,
        &mut scratch.u,
        &scratch.a_inv,
        &scratch.g_cumsum,
        num_value_heads,
    )
    .map_err(|e| anyhow::anyhow!("GDR prefill recompute failed: {e}"))?;
    gated_delta_rule_prefill_chunk_state_stage_into(
        ctx,
        &scratch.k_expanded,
        &scratch.w,
        &scratch.u,
        &scratch.g_cumsum,
        state,
        &mut scratch.chunk_state,
        &mut scratch.v_new,
        num_value_heads,
    )
    .map_err(|e| anyhow::anyhow!("GDR prefill state stage failed: {e}"))?;
    gated_delta_rule_prefill_chunk_o_stage_into(
        ctx,
        &scratch.q_expanded,
        &scratch.k_expanded,
        &scratch.v_new,
        &scratch.chunk_state,
        &scratch.g_cumsum,
        output,
        num_value_heads,
        1.0 / (key_dim as f32).sqrt(),
    )
    .map_err(|e| anyhow::anyhow!("GDR prefill output stage failed: {e}"))
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cudarc::driver::DevicePtrMut;
    use half::bf16;
    use pegainfer_core::tensor::DeviceContext;
    use pegainfer_core::tensor::DeviceVec;
    use pegainfer_core::tensor::HiddenStates;

    use super::conv1d_prefill_batch_into;
    use super::gated_delta_rule_decode_batch_into;
    use super::gated_delta_rule_decode_vec_into;
    use super::gated_delta_rule_prefill_chunkwise_into;
    use super::gated_delta_rule_prefill_native_prepare_into;
    use crate::prefill_buffers::GdnPrepareScratch35;
    use crate::prefill_buffers::GdrChunkwiseScratch35;

    fn bf16_vec(data: &[f32]) -> Vec<bf16> {
        data.iter().map(|&x| bf16::from_f32(x)).collect()
    }

    fn assert_f32_close_with_stats(
        label: &str,
        expected: &[f32],
        actual: &[f32],
        atol: f32,
        rtol: f32,
    ) {
        assert_eq!(expected.len(), actual.len(), "{label} length mismatch");
        let mut deltas = Vec::with_capacity(expected.len());
        let mut max_relative = 0.0_f32;
        let mut violation_count = 0usize;
        let mut first_violation = None;
        for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
            let delta = (expected - actual).abs();
            let relative = delta / expected.abs().max(actual.abs()).max(1.0e-12);
            deltas.push(delta);
            max_relative = max_relative.max(relative);
            let violation = !expected.is_finite()
                || !actual.is_finite()
                || delta > atol + rtol * expected.abs().max(actual.abs());
            if violation {
                violation_count += 1;
                if first_violation.is_none() {
                    first_violation = Some((index, expected, actual, delta));
                }
            }
        }
        deltas.sort_by(f32::total_cmp);
        let max = deltas.last().copied().unwrap_or(0.0);
        let mean = if deltas.is_empty() {
            0.0
        } else {
            deltas.iter().sum::<f32>() / deltas.len() as f32
        };
        let p99 = deltas
            .get(deltas.len().saturating_sub(1) * 99 / 100)
            .copied()
            .unwrap_or(0.0);
        eprintln!(
            "{label}: elements={} violations={violation_count} max_abs={max:.8} mean_abs={mean:.8} p99_abs={p99:.8} max_rel={max_relative:.8} atol={atol} rtol={rtol}",
            deltas.len()
        );
        assert!(
            first_violation.is_none(),
            "{label} first violation {:?}; violations={violation_count}/{} max_abs={max} mean_abs={mean} p99_abs={p99} max_rel={max_relative}",
            first_violation,
            expected.len(),
        );
    }

    fn assert_bf16_bits_equal(label: &str, expected: &[bf16], actual: &[bf16]) {
        assert_eq!(expected.len(), actual.len(), "{label} length mismatch");
        let first_mismatch = expected
            .iter()
            .zip(actual)
            .position(|(expected, actual)| expected.to_bits() != actual.to_bits());
        assert!(
            first_mismatch.is_none(),
            "{label} first bitwise mismatch at {:?}: expected={:?} actual={:?}",
            first_mismatch,
            first_mismatch.map(|index| expected[index].to_f32()),
            first_mismatch.map(|index| actual[index].to_f32()),
        );
        eprintln!("{label}: elements={} bitwise_mismatches=0", expected.len());
    }

    fn softplus(value: f32) -> f32 {
        if value > 20.0 {
            value
        } else if value < -20.0 {
            value.exp()
        } else {
            value.exp().ln_1p()
        }
    }

    fn sigmoid(value: f32) -> f32 {
        let exp = if value < 0.0 {
            value.exp()
        } else {
            (-value).exp()
        };
        if value >= 0.0 {
            1.0 / (1.0 + exp)
        } else {
            exp / (1.0 + exp)
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn native_prepare_hv32_dynamic_t_and_non_finite_inputs() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let h_q = 16usize;
        let h_k = 16usize;
        let h_v = 32usize;
        let d = 128usize;
        let qkv_dim = (h_q + h_k + h_v) * d;
        let dt_host = bf16_vec(
            &(0..h_v)
                .map(|head| (head as f32 - h_v as f32 / 2.0) / 64.0)
                .collect::<Vec<_>>(),
        );
        let a_log_host = (0..h_v)
            .map(|head| -2.5 + head as f32 / h_v as f32)
            .collect::<Vec<_>>();
        let dt_bias = DeviceVec::from_host(&ctx, &dt_host)?;
        let a_log = ctx.stream.clone_htod(&a_log_host)?;

        for tokens in [1usize, 63, 64, 65, 128, 2048] {
            let qkv_host = bf16_vec(
                &(0..tokens * qkv_dim)
                    .map(|index| {
                        let signed = ((index * 37 + 11) % 251) as i32 - 125;
                        signed as f32 / 31.0
                    })
                    .collect::<Vec<_>>(),
            );
            let b_host = bf16_vec(
                &(0..tokens * h_v)
                    .map(|index| ((index * 13 % 41) as f32 - 20.0) / 7.0)
                    .collect::<Vec<_>>(),
            );
            let a_host = bf16_vec(
                &(0..tokens * h_v)
                    .map(|index| ((index * 17 % 47) as f32 - 23.0) / 9.0)
                    .collect::<Vec<_>>(),
            );
            let qkv = HiddenStates {
                data: ctx.stream.clone_htod(&qkv_host)?,
                hidden_dim: qkv_dim,
                seq_len: tokens,
            };
            let b = HiddenStates {
                data: ctx.stream.clone_htod(&b_host)?,
                hidden_dim: h_v,
                seq_len: tokens,
            };
            let a = HiddenStates {
                data: ctx.stream.clone_htod(&a_host)?,
                hidden_dim: h_v,
                seq_len: tokens,
            };
            let mut prepared = GdnPrepareScratch35::from_dims(&ctx, h_q, h_k, h_v, d, tokens)?;
            gated_delta_rule_prefill_native_prepare_into(
                &ctx,
                &qkv,
                &b,
                &a,
                &dt_bias,
                &a_log,
                &mut prepared,
                h_q,
                h_k,
                h_v,
                d,
            )?;

            let status = ctx.stream.clone_dtoh(&prepared.non_finite_status)?;
            let q_actual = ctx.stream.clone_dtoh(&prepared.q.data)?;
            let k_actual = ctx.stream.clone_dtoh(&prepared.k.data)?;
            let v_actual = ctx.stream.clone_dtoh(&prepared.v.data)?;
            let alpha_actual = ctx.stream.clone_dtoh(&prepared.alpha)?;
            let beta_actual = ctx.stream.clone_dtoh(&prepared.beta)?;
            ctx.sync()?;
            assert_eq!(status, [0], "finite Hv32 T={tokens} fixture was rejected");

            let mut q_expected = Vec::with_capacity(tokens * h_q * d);
            let mut k_expected = Vec::with_capacity(tokens * h_k * d);
            let mut v_expected = Vec::with_capacity(tokens * h_v * d);
            for token in 0..tokens {
                let token_qkv = token * qkv_dim;
                for head in 0..h_q {
                    let input = token_qkv + head * d;
                    let output = (token * h_q + head) * d;
                    let sum_sq = qkv_host[input..input + d]
                        .iter()
                        .map(|value| value.to_f32().powi(2))
                        .sum::<f32>();
                    let inv_norm = (sum_sq + 1.0e-12).sqrt().recip();
                    for lane in 0..d {
                        q_expected.push(qkv_host[input + lane].to_f32() * inv_norm);
                    }
                    debug_assert_eq!(q_expected.len(), output + d);
                }
                for head in 0..h_k {
                    let input = token_qkv + h_q * d + head * d;
                    let output = (token * h_k + head) * d;
                    let sum_sq = qkv_host[input..input + d]
                        .iter()
                        .map(|value| value.to_f32().powi(2))
                        .sum::<f32>();
                    let inv_norm = (sum_sq + 1.0e-12).sqrt().recip();
                    for lane in 0..d {
                        k_expected.push(qkv_host[input + lane].to_f32() * inv_norm);
                    }
                    debug_assert_eq!(k_expected.len(), output + d);
                }
                let v_input = token_qkv + (h_q + h_k) * d;
                v_expected.extend_from_slice(&qkv_host[v_input..v_input + h_v * d]);
            }
            let q_actual_f32 = q_actual
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();
            let k_actual_f32 = k_actual
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();
            assert_f32_close_with_stats(
                &format!("native prepare Q [T={tokens},H={h_q},D={d},bf16]"),
                &q_expected,
                &q_actual_f32,
                1.0 / 256.0,
                0.0,
            );
            assert_f32_close_with_stats(
                &format!("native prepare K [T={tokens},H={h_k},D={d},bf16]"),
                &k_expected,
                &k_actual_f32,
                1.0 / 256.0,
                0.0,
            );
            assert_bf16_bits_equal(
                &format!("native prepare V [T={tokens},H={h_v},D={d},bf16]"),
                &v_expected,
                &v_actual,
            );
            let mut alpha_expected = Vec::with_capacity(tokens * h_v);
            let mut beta_expected = Vec::with_capacity(tokens * h_v);
            for index in 0..tokens * h_v {
                let head = index % h_v;
                let a_value = a_host[index].to_f32();
                let b_value = b_host[index].to_f32();
                let expected_alpha =
                    (-a_log_host[head].exp() * softplus(a_value + dt_host[head].to_f32())).exp();
                let expected_beta = sigmoid(b_value);
                alpha_expected.push(expected_alpha);
                beta_expected.push(expected_beta);
            }
            assert_f32_close_with_stats(
                &format!("native prepare alpha [T={tokens},H={h_v},f32]"),
                &alpha_expected,
                &alpha_actual,
                2.0e-6,
                2.0e-6,
            );
            assert_f32_close_with_stats(
                &format!("native prepare beta [T={tokens},H={h_v},f32]"),
                &beta_expected,
                &beta_actual,
                2.0e-6,
                2.0e-6,
            );
        }

        for non_finite_source in ["q", "v", "gate"] {
            let mut qkv_host = vec![bf16::from_f32(0.25); qkv_dim];
            let b_host = vec![bf16::from_f32(-0.5); h_v];
            let mut a_host = vec![bf16::from_f32(0.5); h_v];
            match non_finite_source {
                "q" => qkv_host[0] = bf16::from_bits(0x7fc0),
                "v" => qkv_host[(h_q + h_k) * d + 7] = bf16::from_bits(0x7fc0),
                "gate" => a_host[0] = bf16::from_bits(0x7fc0),
                _ => unreachable!(),
            }
            let qkv = HiddenStates {
                data: ctx.stream.clone_htod(&qkv_host)?,
                hidden_dim: qkv_dim,
                seq_len: 1,
            };
            let b = HiddenStates {
                data: ctx.stream.clone_htod(&b_host)?,
                hidden_dim: h_v,
                seq_len: 1,
            };
            let a = HiddenStates {
                data: ctx.stream.clone_htod(&a_host)?,
                hidden_dim: h_v,
                seq_len: 1,
            };
            let mut prepared = GdnPrepareScratch35::from_dims(&ctx, h_q, h_k, h_v, d, 1)?;
            gated_delta_rule_prefill_native_prepare_into(
                &ctx,
                &qkv,
                &b,
                &a,
                &dt_bias,
                &a_log,
                &mut prepared,
                h_q,
                h_k,
                h_v,
                d,
            )?;
            let status = ctx.stream.clone_dtoh(&prepared.non_finite_status)?;
            ctx.sync()?;
            assert_eq!(
                status,
                [1],
                "non-finite {non_finite_source} input was not reported"
            );
        }

        let finite_qkv_host = vec![bf16::from_f32(0.25); qkv_dim];
        let mut non_finite_qkv_host = finite_qkv_host.clone();
        non_finite_qkv_host[0] = bf16::from_bits(0x7fc0);
        let gate_b_host = vec![bf16::from_f32(-0.5); h_v];
        let gate_a_host = vec![bf16::from_f32(0.5); h_v];
        let make_hidden = |values: &[bf16], hidden_dim: usize| -> Result<HiddenStates> {
            Ok(HiddenStates {
                data: ctx.stream.clone_htod(values)?,
                hidden_dim,
                seq_len: 1,
            })
        };
        let non_finite_qkv = make_hidden(&non_finite_qkv_host, qkv_dim)?;
        let finite_qkv = make_hidden(&finite_qkv_host, qkv_dim)?;
        let gate_b = make_hidden(&gate_b_host, h_v)?;
        let gate_a = make_hidden(&gate_a_host, h_v)?;
        let mut sticky = GdnPrepareScratch35::from_dims(&ctx, h_q, h_k, h_v, d, 1)?;
        gated_delta_rule_prefill_native_prepare_into(
            &ctx,
            &non_finite_qkv,
            &gate_b,
            &gate_a,
            &dt_bias,
            &a_log,
            &mut sticky,
            h_q,
            h_k,
            h_v,
            d,
        )?;
        gated_delta_rule_prefill_native_prepare_into(
            &ctx,
            &finite_qkv,
            &gate_b,
            &gate_a,
            &dt_bias,
            &a_log,
            &mut sticky,
            h_q,
            h_k,
            h_v,
            d,
        )?;
        let sticky_status = ctx.stream.clone_dtoh(&sticky.non_finite_status)?;
        ctx.sync()?;
        assert_eq!(
            sticky_status,
            [1],
            "a later finite layer cleared the chunk-owned non-finite status"
        );

        let mut fresh_chunk = GdnPrepareScratch35::from_dims(&ctx, h_q, h_k, h_v, d, 1)?;
        gated_delta_rule_prefill_native_prepare_into(
            &ctx,
            &finite_qkv,
            &gate_b,
            &gate_a,
            &dt_bias,
            &a_log,
            &mut fresh_chunk,
            h_q,
            h_k,
            h_v,
            d,
        )?;
        let fresh_status = ctx.stream.clone_dtoh(&fresh_chunk.non_finite_status)?;
        ctx.sync()?;
        assert_eq!(
            fresh_status,
            [0],
            "a new chunk did not start with a clear non-finite status"
        );
        Ok(())
    }

    #[test]
    fn conv1d_prefill_handoff_matches_single_prefill() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let num_channels = 1024usize;
        let kernel_size = 4usize;
        let total_seq = 18usize;
        let prefix_seq = 5usize;

        let x_host = bf16_vec(
            &(0..num_channels * total_seq)
                .map(|i| ((i % 71) as f32 - 35.0) * 0.03125)
                .collect::<Vec<_>>(),
        );
        let w_host = bf16_vec(
            &(0..num_channels * kernel_size)
                .map(|i| ((i % 19) as f32 - 9.0) * 0.0625)
                .collect::<Vec<_>>(),
        );

        let x_all = HiddenStates {
            data: ctx.stream.clone_htod(&x_host)?,
            hidden_dim: num_channels,
            seq_len: total_seq,
        };
        let conv_weight = DeviceVec::from_host(&ctx, &w_host)?;
        let state_len = num_channels * (kernel_size - 1);
        let zero_state = vec![bf16::ZERO; state_len];

        let mut state_all = DeviceVec::from_host(&ctx, &zero_state)?;
        let mut out_all = HiddenStates::zeros(&ctx, num_channels, total_seq)?;
        conv1d_prefill_batch_into(
            &ctx,
            &x_all,
            &conv_weight,
            &mut state_all,
            &mut out_all,
            kernel_size,
        );

        let x_prefix = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&x_host[..num_channels * prefix_seq])?,
            hidden_dim: num_channels,
            seq_len: prefix_seq,
        };
        let mut state_split = DeviceVec::from_host(&ctx, &zero_state)?;
        let mut out_prefix = HiddenStates::zeros(&ctx, num_channels, prefix_seq)?;
        conv1d_prefill_batch_into(
            &ctx,
            &x_prefix,
            &conv_weight,
            &mut state_split,
            &mut out_prefix,
            kernel_size,
        );

        for step in prefix_seq..total_seq {
            let x_step = HiddenStates {
                data: ctx
                    .stream
                    .clone_htod(&x_host[num_channels * step..num_channels * (step + 1)])?,
                hidden_dim: num_channels,
                seq_len: 1,
            };
            let mut out_step = HiddenStates::zeros(&ctx, num_channels, 1)?;
            conv1d_prefill_batch_into(
                &ctx,
                &x_step,
                &conv_weight,
                &mut state_split,
                &mut out_step,
                kernel_size,
            );
        }

        let out_all_host = ctx.stream.clone_dtoh(&out_all.data)?;
        let state_all_host = state_all.to_host(&ctx)?;
        let state_split_host = state_split.to_host(&ctx)?;
        ctx.sync()?;

        let out_all_host: Vec<f32> = out_all_host.iter().map(|x| x.to_f32()).collect();
        let expected_last = &out_all_host[num_channels * (total_seq - 1)..num_channels * total_seq];

        let x_last = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&x_host[num_channels * (total_seq - 1)..num_channels * total_seq])?,
            hidden_dim: num_channels,
            seq_len: 1,
        };
        let mut state_last = DeviceVec::from_host(&ctx, &zero_state)?;
        let x_before_last = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&x_host[..num_channels * (total_seq - 1)])?,
            hidden_dim: num_channels,
            seq_len: total_seq - 1,
        };
        let mut scratch_before_last = HiddenStates::zeros(&ctx, num_channels, total_seq - 1)?;
        conv1d_prefill_batch_into(
            &ctx,
            &x_before_last,
            &conv_weight,
            &mut state_last,
            &mut scratch_before_last,
            kernel_size,
        );
        let mut out_last = HiddenStates::zeros(&ctx, num_channels, 1)?;
        conv1d_prefill_batch_into(
            &ctx,
            &x_last,
            &conv_weight,
            &mut state_last,
            &mut out_last,
            kernel_size,
        );
        let out_last_host = ctx.stream.clone_dtoh(&out_last.data)?;
        ctx.sync()?;
        let out_last_host: Vec<f32> = out_last_host.iter().map(|x| x.to_f32()).collect();

        let max_out_diff = expected_last
            .iter()
            .zip(out_last_host.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let max_state_diff = state_all_host
            .iter()
            .zip(state_split_host.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        assert!(max_out_diff < 0.02, "output diff {max_out_diff}");
        assert!(max_state_diff < 0.02, "state diff {max_state_diff}");
        Ok(())
    }

    #[test]
    fn gdr_decode_batch_matches_single_slot_reference() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let batch_size = 3usize;
        let num_key_heads = 16usize;
        let num_value_heads = 48usize;
        let key_dim = 128usize;
        let val_dim = 128usize;

        let qkv_dim = 2 * num_key_heads * key_dim + num_value_heads * val_dim;
        let out_dim = num_value_heads * val_dim;
        let state_len = num_value_heads * key_dim * val_dim;

        let qkv_host = bf16_vec(
            &(0..batch_size * qkv_dim)
                .map(|i| ((i % 89) as f32 - 44.0) * 0.007_812_5)
                .collect::<Vec<_>>(),
        );
        let b_host = bf16_vec(
            &(0..batch_size * num_value_heads)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.03125)
                .collect::<Vec<_>>(),
        );
        let a_host = bf16_vec(
            &(0..batch_size * num_value_heads)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.03125)
                .collect::<Vec<_>>(),
        );
        let dt_host = bf16_vec(
            &(0..num_value_heads)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.0625)
                .collect::<Vec<_>>(),
        );
        let alog_host: Vec<f32> = (0..num_value_heads)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.125)
            .collect();

        let qkv_batch = HiddenStates {
            data: ctx.stream.clone_htod(&qkv_host)?,
            hidden_dim: qkv_dim,
            seq_len: batch_size,
        };
        let b_batch = HiddenStates {
            data: ctx.stream.clone_htod(&b_host)?,
            hidden_dim: num_value_heads,
            seq_len: batch_size,
        };
        let a_batch = HiddenStates {
            data: ctx.stream.clone_htod(&a_host)?,
            hidden_dim: num_value_heads,
            seq_len: batch_size,
        };
        let dt_bias = DeviceVec::from_host(&ctx, &dt_host)?;
        let a_log = ctx.stream.clone_htod(&alog_host)?;

        let mut batch_states: Vec<cudarc::driver::CudaSlice<f32>> = (0..batch_size)
            .map(|_| ctx.stream.alloc_zeros(state_len))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut state_ptrs = Vec::with_capacity(batch_size);
        for state in &mut batch_states {
            let (ptr, _guard) = state.device_ptr_mut(&ctx.stream);
            state_ptrs.push(ptr);
        }
        let state_ptrs_d = ctx.stream.clone_htod(&state_ptrs)?;

        let mut out_batch = HiddenStates::zeros(&ctx, out_dim, batch_size)?;
        gated_delta_rule_decode_batch_into(
            &ctx,
            &qkv_batch,
            &b_batch,
            &a_batch,
            &dt_bias,
            &a_log,
            &state_ptrs_d,
            &mut out_batch,
            batch_size,
            num_key_heads,
            num_value_heads,
            key_dim,
            val_dim,
        );

        let mut out_ref_rows: Vec<f32> = Vec::with_capacity(batch_size * out_dim);
        let mut ref_states = Vec::with_capacity(batch_size);
        for row in 0..batch_size {
            let qkv_row =
                DeviceVec::from_host(&ctx, &qkv_host[row * qkv_dim..(row + 1) * qkv_dim])?;
            let b_row = DeviceVec::from_host(
                &ctx,
                &b_host[row * num_value_heads..(row + 1) * num_value_heads],
            )?;
            let a_row = DeviceVec::from_host(
                &ctx,
                &a_host[row * num_value_heads..(row + 1) * num_value_heads],
            )?;
            let mut state_ref: cudarc::driver::CudaSlice<f32> =
                ctx.stream.alloc_zeros(state_len)?;
            let mut out_row = DeviceVec::zeros(&ctx, out_dim)?;
            gated_delta_rule_decode_vec_into(
                &ctx,
                &qkv_row,
                &b_row,
                &a_row,
                &dt_bias,
                &a_log,
                &mut state_ref,
                &mut out_row,
                num_key_heads,
                num_value_heads,
                key_dim,
                val_dim,
            );
            out_ref_rows.extend_from_slice(&out_row.to_host(&ctx)?);
            ref_states.push(state_ref);
        }

        let out_batch_host = ctx.stream.clone_dtoh(&out_batch.data)?;
        ctx.sync()?;
        let out_batch_host: Vec<f32> = out_batch_host.iter().map(|x| x.to_f32()).collect();
        let max_out_diff = out_batch_host
            .iter()
            .zip(out_ref_rows.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        let mut max_state_diff = 0.0_f32;
        for (batch_state, ref_state) in batch_states.iter().zip(ref_states.iter()) {
            let batch_state_host = ctx.stream.clone_dtoh(batch_state)?;
            let ref_state_host = ctx.stream.clone_dtoh(ref_state)?;
            ctx.sync()?;
            max_state_diff = max_state_diff.max(
                batch_state_host
                    .iter()
                    .zip(ref_state_host.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max),
            );
        }

        assert!(max_out_diff < 0.05, "output diff {max_out_diff}");
        assert!(max_state_diff < 0.05, "state diff {max_state_diff}");
        Ok(())
    }

    #[test]
    fn gdn_chunkwise_prefill_matches_stepwise_decode_at_48_value_heads() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let num_key_heads = 16usize;
        let num_value_heads = 48usize;
        let key_dim = 128usize;
        let val_dim = 128usize;
        let seq_len = 96usize;

        let qkv_dim = 2 * num_key_heads * key_dim + num_value_heads * val_dim;
        let out_dim = num_value_heads * val_dim;
        let state_len = num_value_heads * key_dim * val_dim;

        let qkv_host = bf16_vec(
            &(0..seq_len * qkv_dim)
                .map(|i| ((i % 73) as f32 - 36.0) * 0.01)
                .collect::<Vec<_>>(),
        );
        let b_host = bf16_vec(
            &(0..seq_len * num_value_heads)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
                .collect::<Vec<_>>(),
        );
        let a_host = bf16_vec(
            &(0..seq_len * num_value_heads)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
                .collect::<Vec<_>>(),
        );
        let dt_host = bf16_vec(
            &(0..num_value_heads)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
                .collect::<Vec<_>>(),
        );
        let alog_host: Vec<f32> = (0..num_value_heads)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.2)
            .collect();

        let dt_bias = DeviceVec::from_host(&ctx, &dt_host)?;
        let a_log = ctx.stream.clone_htod(&alog_host)?;

        let qkv_all = HiddenStates {
            data: ctx.stream.clone_htod(&qkv_host)?,
            hidden_dim: qkv_dim,
            seq_len,
        };
        let b_all = HiddenStates {
            data: ctx.stream.clone_htod(&b_host)?,
            hidden_dim: num_value_heads,
            seq_len,
        };
        let a_all = HiddenStates {
            data: ctx.stream.clone_htod(&a_host)?,
            hidden_dim: num_value_heads,
            seq_len,
        };
        let mut state_chunk: cudarc::driver::CudaSlice<f32> = ctx.stream.alloc_zeros(state_len)?;
        let mut scratch =
            GdrChunkwiseScratch35::from_dims(&ctx, num_value_heads, key_dim, val_dim, seq_len)?;
        let mut out_chunk = HiddenStates::zeros(&ctx, out_dim, seq_len)?;
        gated_delta_rule_prefill_chunkwise_into(
            &ctx,
            &qkv_all,
            &b_all,
            &a_all,
            &dt_bias,
            &a_log,
            &mut state_chunk,
            &mut scratch,
            &mut out_chunk,
            num_key_heads,
            num_value_heads,
            key_dim,
            val_dim,
        )?;

        let mut state_step: cudarc::driver::CudaSlice<f32> = ctx.stream.alloc_zeros(state_len)?;
        let mut out_step_rows: Vec<f32> = Vec::with_capacity(seq_len * out_dim);
        for t in 0..seq_len {
            let qkv_t = DeviceVec::from_host(&ctx, &qkv_host[t * qkv_dim..(t + 1) * qkv_dim])?;
            let b_t = DeviceVec::from_host(
                &ctx,
                &b_host[t * num_value_heads..(t + 1) * num_value_heads],
            )?;
            let a_t = DeviceVec::from_host(
                &ctx,
                &a_host[t * num_value_heads..(t + 1) * num_value_heads],
            )?;
            let mut out_t = DeviceVec::from_host(&ctx, &vec![bf16::ZERO; out_dim])?;
            gated_delta_rule_decode_vec_into(
                &ctx,
                &qkv_t,
                &b_t,
                &a_t,
                &dt_bias,
                &a_log,
                &mut state_step,
                &mut out_t,
                num_key_heads,
                num_value_heads,
                key_dim,
                val_dim,
            );
            let row = out_t.to_host(&ctx)?;
            out_step_rows.extend_from_slice(&row);
        }

        let out_chunk_host = ctx.stream.clone_dtoh(&out_chunk.data)?;
        let state_chunk_host = ctx.stream.clone_dtoh(&state_chunk)?;
        let state_step_host = ctx.stream.clone_dtoh(&state_step)?;
        ctx.sync()?;
        let out_chunk_host: Vec<f32> = out_chunk_host.iter().map(|x| x.to_f32()).collect();

        let max_out_diff = out_chunk_host
            .iter()
            .zip(out_step_rows.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let max_state_diff = state_chunk_host
            .iter()
            .zip(state_step_host.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            out_chunk_host.iter().all(|x| x.is_finite())
                && state_chunk_host.iter().all(|x| x.is_finite()),
            "chunkwise outputs must be finite"
        );
        assert!(max_out_diff < 0.05, "output diff {max_out_diff}");
        assert!(max_state_diff < 0.05, "state diff {max_state_diff}");
        Ok(())
    }
}
