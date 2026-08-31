use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;

/// RMSNorm into pre-allocated output buffer
pub fn rms_norm_into(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(x.len, out.len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.len as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
    Ok(())
}

/// Slice-level batched RMSNorm over `rows` rows of `dim`: same
/// `flashinfer::norm::RMSNorm` template as [`rms_norm_into`] with
/// batch_size=rows (one CTA per row), so each row is bit-identical to the
/// single-row launch. For callers whose buffers live in a persistent decode
/// arena rather than owned `HiddenStates`.
pub fn rms_norm_rows_into(
    ctx: &DeviceContext,
    x: &CudaSlice<bf16>,
    weight: &DeviceVec,
    eps: f32,
    dim: usize,
    rows: usize,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(rows > 0 && dim > 0, "rms_norm_rows needs positive rows/dim");
    ensure!(
        x.len() >= rows * dim && out.len() >= rows * dim,
        "rms_norm_rows buffers too small for {rows}x{dim}: x {}, out {}",
        x.len(),
        out.len()
    );
    ensure!(
        weight.len == dim,
        "rms_norm_rows weight len {} != dim {dim}",
        weight.len
    );
    let (x_ptr, _gx) = x.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_batched_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            dim as i32,
            rows as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
    Ok(())
}

/// RMSNorm (allocating)
pub fn rms_norm(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
) -> Result<DeviceVec> {
    let mut out = DeviceVec::zeros(ctx, x.len)?;
    rms_norm_into(ctx, x, weight, eps, &mut out)?;
    Ok(out)
}

/// LayerNorm (with bias) of a single bf16 vector `[dim]` — GLM5.2 DSA indexer
/// k_norm (eps=1e-6, has bias). Wraps `flashinfer::norm::LayerNorm` (same
/// vendored template that `rms_norm_cuda` wraps). Unlike RMSNorm, LayerNorm
/// subtracts the mean and applies a per-element bias. gamma/beta are f32
/// (FlashInfer's LayerNorm template requires f32 weight types).
pub fn layer_norm_into(
    ctx: &DeviceContext,
    x: &CudaSlice<bf16>,
    gamma: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    eps: f32,
    dim: usize,
    rows: usize,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(rows > 0 && dim > 0, "layer_norm needs positive rows/dim");
    ensure!(
        x.len() >= rows * dim && out.len() >= rows * dim,
        "layer_norm x/out too small for {rows}x{dim}: x {}, out {}",
        x.len(),
        out.len()
    );
    ensure!(gamma.len() >= dim, "layer_norm gamma too small");
    ensure!(beta.len() >= dim, "layer_norm beta too small");
    let (x_ptr, _gx) = x.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = gamma.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = beta.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::layer_norm_cuda(
            x_ptr as *const ffi::Half,
            g_ptr as *const f32,
            b_ptr as *const f32,
            o_ptr as *mut ffi::Half,
            dim as i32,
            rows as i32,
            eps,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow::anyhow!("GLM5.2 LayerNorm launch failed: {err}"))
}

/// Fused add + RMSNorm: hidden += residual; out = rms_norm(hidden, weight)
/// Saves one global read of hidden compared to separate add + rms_norm.
pub fn fused_add_rms_norm_into(
    ctx: &DeviceContext,
    hidden: &mut DeviceVec,
    residual: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(hidden.len, residual.len);
    assert_eq!(hidden.len, out.len);
    let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = residual.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::fused_add_rms_norm_cuda(
            h_ptr as *mut ffi::Half,
            r_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden.len as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
    Ok(())
}

/// Two RMSNorms of the same input in one launch: `out_a = rms(x) * weight_a *
/// scale_a` and `out_b = rms(x) * weight_b`, each bitwise what its standalone
/// operations produce. The norms share one reduction, and the input is read
/// twice instead of four times.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_batch_dual_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight_a: &DeviceVec,
    weight_b: &DeviceVec,
    eps: f32,
    scale_a: f32,
    out_a: &mut HiddenStates,
    out_b: &mut HiddenStates,
) -> Result<()> {
    const OP: &str = "rms_norm_batch_dual_into";
    ensure!(
        x.hidden_dim > 0 && x.seq_len > 0,
        "{OP} dimensions must be non-zero"
    );
    ensure!(weight_a.len == x.hidden_dim, "{OP} weight_a len mismatch");
    ensure!(weight_b.len == x.hidden_dim, "{OP} weight_b len mismatch");
    ensure!(
        weight_a.data.len() >= x.hidden_dim,
        "{OP} weight_a backing too small"
    );
    ensure!(
        weight_b.data.len() >= x.hidden_dim,
        "{OP} weight_b backing too small"
    );
    ensure!(
        out_a.hidden_dim == x.hidden_dim && out_a.seq_len == x.seq_len,
        "{OP} out_a shape mismatch"
    );
    ensure!(
        out_b.hidden_dim == x.hidden_dim && out_b.seq_len == x.seq_len,
        "{OP} out_b shape mismatch"
    );
    ensure!(scale_a.is_finite(), "{OP} scale_a must be finite");
    x.checked_extent("rms_norm_batch_dual_into x")?;
    out_a.checked_extent("rms_norm_batch_dual_into out_a")?;
    out_b.checked_extent("rms_norm_batch_dual_into out_b")?;
    let hidden_dim = super::checked_i32(x.hidden_dim, "rms_norm_batch_dual_into hidden_dim")?;
    let seq_len = super::checked_i32(x.seq_len, "rms_norm_batch_dual_into seq_len")?;
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (wa_ptr, _ga) = weight_a.data.device_ptr(&ctx.stream);
    let (wb_ptr, _gb) = weight_b.data.device_ptr(&ctx.stream);
    let (oa_ptr, _goa) = out_a.data.device_ptr_mut(&ctx.stream);
    let (ob_ptr, _gob) = out_b.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::rms_norm_batched_dual_cuda(
            x_ptr as *const ffi::Half,
            wa_ptr as *const ffi::Half,
            wb_ptr as *const ffi::Half,
            oa_ptr as *mut ffi::Half,
            ob_ptr as *mut ffi::Half,
            hidden_dim,
            seq_len,
            eps,
            scale_a,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// The MoE combine tail in one launch: `out = rms_norm(a, weight_a) +
/// rms_norm(b, weight_b)`, bitwise what the two separate norms and the add
/// it replaces produce.
pub fn dual_rms_norm_add_batch_into(
    ctx: &DeviceContext,
    a: &HiddenStates,
    weight_a: &DeviceVec,
    b: &HiddenStates,
    weight_b: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    const OP: &str = "dual_rms_norm_add_batch_into";
    ensure!(
        a.hidden_dim > 0 && a.seq_len > 0,
        "{OP} dimensions must be non-zero"
    );
    ensure!(weight_a.len == a.hidden_dim, "{OP} weight_a len mismatch");
    ensure!(weight_b.len == a.hidden_dim, "{OP} weight_b len mismatch");
    ensure!(
        weight_a.data.len() >= a.hidden_dim,
        "{OP} weight_a backing too small"
    );
    ensure!(
        weight_b.data.len() >= a.hidden_dim,
        "{OP} weight_b backing too small"
    );
    ensure!(
        b.hidden_dim == a.hidden_dim && b.seq_len == a.seq_len,
        "{OP} b shape mismatch"
    );
    ensure!(
        out.hidden_dim == a.hidden_dim && out.seq_len == a.seq_len,
        "{OP} out shape mismatch"
    );
    a.checked_extent("dual_rms_norm_add_batch_into a")?;
    b.checked_extent("dual_rms_norm_add_batch_into b")?;
    out.checked_extent("dual_rms_norm_add_batch_into out")?;
    let hidden_dim = super::checked_i32(a.hidden_dim, "dual_rms_norm_add_batch_into hidden_dim")?;
    let seq_len = super::checked_i32(a.seq_len, "dual_rms_norm_add_batch_into seq_len")?;
    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    let (wa_ptr, _gwa) = weight_a.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.data.device_ptr(&ctx.stream);
    let (wb_ptr, _gwb) = weight_b.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::dual_rms_norm_add_batched_cuda(
            a_ptr as *const ffi::Half,
            wa_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            wb_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden_dim,
            seq_len,
            eps,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// The attention epilogue's norm pair in one launch:
/// `residual_out = bf16(rms_norm(x, weight_post) + res_in)` then
/// `out = rms_norm(residual_out, weight_pre)`, bitwise what the separate
/// norm and [`fused_add_rms_norm_round_batch_into`] produce.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_add_rms_norm_round_batch_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight_post: &DeviceVec,
    res_in: &HiddenStates,
    weight_pre: &DeviceVec,
    eps: f32,
    residual_out: &mut HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    const OP: &str = "rms_norm_add_rms_norm_round_batch_into";
    ensure!(
        x.hidden_dim > 0 && x.seq_len > 0,
        "{OP} dimensions must be non-zero"
    );
    ensure!(
        weight_post.len == x.hidden_dim,
        "{OP} weight_post len mismatch"
    );
    ensure!(
        weight_pre.len == x.hidden_dim,
        "{OP} weight_pre len mismatch"
    );
    ensure!(
        weight_post.data.len() >= x.hidden_dim,
        "{OP} weight_post backing too small"
    );
    ensure!(
        weight_pre.data.len() >= x.hidden_dim,
        "{OP} weight_pre backing too small"
    );
    ensure!(
        res_in.hidden_dim == x.hidden_dim && res_in.seq_len == x.seq_len,
        "{OP} res_in shape mismatch"
    );
    ensure!(
        residual_out.hidden_dim == x.hidden_dim && residual_out.seq_len == x.seq_len,
        "{OP} residual_out shape mismatch"
    );
    ensure!(
        out.hidden_dim == x.hidden_dim && out.seq_len == x.seq_len,
        "{OP} out shape mismatch"
    );
    x.checked_extent("rms_norm_add_rms_norm_round_batch_into x")?;
    res_in.checked_extent("rms_norm_add_rms_norm_round_batch_into res_in")?;
    residual_out.checked_extent("rms_norm_add_rms_norm_round_batch_into residual_out")?;
    out.checked_extent("rms_norm_add_rms_norm_round_batch_into out")?;
    let hidden_dim = super::checked_i32(
        x.hidden_dim,
        "rms_norm_add_rms_norm_round_batch_into hidden_dim",
    )?;
    let seq_len = super::checked_i32(x.seq_len, "rms_norm_add_rms_norm_round_batch_into seq_len")?;
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (wp_ptr, _gwp) = weight_post.data.device_ptr(&ctx.stream);
    let (r_ptr, _gr) = res_in.data.device_ptr(&ctx.stream);
    let (wq_ptr, _gwq) = weight_pre.data.device_ptr(&ctx.stream);
    let (ro_ptr, _gro) = residual_out.data.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::rms_norm_add_rms_norm_round_batched_cuda(
            x_ptr as *const ffi::Half,
            wp_ptr as *const ffi::Half,
            r_ptr as *const ffi::Half,
            wq_ptr as *const ffi::Half,
            ro_ptr as *mut ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden_dim,
            seq_len,
            eps,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// The layer tail in one launch: `out = (residual + rms_norm(x, weight)) *
/// scale`, bit-identical to the separate norm, add and scale calls it
/// replaces.
pub fn rms_norm_add_scale_batch_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    residual: &HiddenStates,
    scale: f32,
    eps: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    const OP: &str = "rms_norm_add_scale_batch_into";
    ensure!(
        x.hidden_dim > 0 && x.seq_len > 0,
        "{OP} dimensions must be non-zero"
    );
    ensure!(weight.len == x.hidden_dim, "{OP} weight len mismatch");
    ensure!(
        weight.data.len() >= x.hidden_dim,
        "{OP} weight backing too small"
    );
    ensure!(
        residual.hidden_dim == x.hidden_dim && residual.seq_len == x.seq_len,
        "{OP} residual shape mismatch"
    );
    ensure!(
        out.hidden_dim == x.hidden_dim && out.seq_len == x.seq_len,
        "{OP} out shape mismatch"
    );
    ensure!(scale.is_finite(), "{OP} scale must be finite");
    x.checked_extent("rms_norm_add_scale_batch_into x")?;
    residual.checked_extent("rms_norm_add_scale_batch_into residual")?;
    out.checked_extent("rms_norm_add_scale_batch_into out")?;
    let hidden_dim = super::checked_i32(x.hidden_dim, "rms_norm_add_scale_batch_into hidden_dim")?;
    let seq_len = super::checked_i32(x.seq_len, "rms_norm_add_scale_batch_into seq_len")?;
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (r_ptr, _gr) = residual.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::rms_norm_add_scale_batched_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            r_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden_dim,
            seq_len,
            eps,
            scale,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Batched fused add + RMSNorm for HiddenStates.
/// hidden[i] += residual[i]; out[i] = rms_norm(hidden[i], weight) for each batch element.
pub fn fused_add_rms_norm_batch_into(
    ctx: &DeviceContext,
    hidden: &mut HiddenStates,
    residual: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) {
    assert_eq!(hidden.hidden_dim, residual.hidden_dim);
    assert_eq!(hidden.hidden_dim, out.hidden_dim);
    assert_eq!(hidden.seq_len, residual.seq_len);
    assert_eq!(hidden.seq_len, out.seq_len);
    assert_eq!(weight.len, hidden.hidden_dim);
    let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = residual.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::fused_add_rms_norm_batched_cuda(
            h_ptr as *mut ffi::Half,
            r_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden.hidden_dim as i32,
            hidden.seq_len as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
}

/// Batched exact-preserving fused add + RMSNorm.
/// hidden[i] = bf16(hidden[i] + residual[i]); out[i] = rms_norm(hidden[i], weight).
pub fn fused_add_rms_norm_round_batch_into(
    ctx: &DeviceContext,
    hidden: &mut HiddenStates,
    residual: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(hidden.hidden_dim, residual.hidden_dim);
    assert_eq!(hidden.hidden_dim, out.hidden_dim);
    assert_eq!(hidden.seq_len, residual.seq_len);
    assert_eq!(hidden.seq_len, out.seq_len);
    assert_eq!(weight.len, hidden.hidden_dim);
    let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = residual.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::fused_add_rms_norm_round_batched_cuda(
            h_ptr as *mut ffi::Half,
            r_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden.hidden_dim as i32,
            hidden.seq_len as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Slice-level twin of [`fused_add_rms_norm_round_batch_into`] — same kernel
/// — for callers whose buffers live in a persistent decode arena rather than
/// owned `HiddenStates`: `hidden += residual` (sum rounded to bf16), then
/// `out = rms_norm(hidden, weight)` over `seq_len` rows of `hidden_dim`.
pub fn fused_add_rms_norm_round_into(
    ctx: &DeviceContext,
    hidden: &mut CudaSlice<bf16>,
    residual: &CudaSlice<bf16>,
    weight: &DeviceVec,
    eps: f32,
    hidden_dim: usize,
    seq_len: usize,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    let n = hidden_dim * seq_len;
    ensure!(
        hidden.len() >= n && residual.len() >= n && out.len() >= n,
        "fused_add_rms_norm_round_into buffers too small for {seq_len}x{hidden_dim}: hidden {}, residual {}, out {}",
        hidden.len(),
        residual.len(),
        out.len()
    );
    ensure!(
        weight.len == hidden_dim,
        "fused_add_rms_norm_round_into weight len {} != hidden_dim {hidden_dim}",
        weight.len
    );
    let (h_ptr, _gh) = hidden.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = residual.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::fused_add_rms_norm_round_batched_cuda(
            h_ptr as *mut ffi::Half,
            r_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden_dim as i32,
            seq_len as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Batched RMSNorm into pre-allocated output buffer (zero allocation).
pub fn rms_norm_batch_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) {
    assert_eq!(weight.len, x.hidden_dim);
    assert_eq!(out.hidden_dim, x.hidden_dim);
    assert_eq!(out.seq_len, x.seq_len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_batched_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.hidden_dim as i32,
            x.seq_len as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
}

/// Batched (1+weight) RMSNorm over HiddenStates — one kernel launch for all tokens.
pub fn rms_norm_batch_offset_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(weight.len, x.hidden_dim);
    assert_eq!(out.hidden_dim, x.hidden_dim);
    assert_eq!(out.seq_len, x.seq_len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_batched_offset_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.hidden_dim as i32,
            x.seq_len as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
    Ok(())
}

/// (1+weight) RMSNorm into pre-allocated output buffer (Gemma/Qwen3.5 style)
pub fn rms_norm_offset_into(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(x.len, out.len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_offset_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.len as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
    Ok(())
}

/// Batched per-head RMSNorm with F32 weight + SiLU gate multiplication.
/// HiddenStates are flattened as (seq_len * num_heads) contiguous head slices.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_gated_batch_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &CudaSlice<f32>,
    gate: &HiddenStates,
    out: &mut HiddenStates,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    let total_heads = x.seq_len * num_heads;
    assert_eq!(x.hidden_dim, num_heads * head_dim);
    assert_eq!(gate.hidden_dim, x.hidden_dim);
    assert_eq!(gate.seq_len, x.seq_len);
    assert_eq!(out.hidden_dim, x.hidden_dim);
    assert_eq!(out.seq_len, x.seq_len);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = gate.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_gated_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const f32,
            g_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            total_heads as i32,
            head_dim as i32,
            eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
}

#[cfg(test)]
mod parity {
    use super::*;

    fn assert_bitwise(ctx: &DeviceContext, a: &HiddenStates, b: &HiddenStates, what: &str) {
        let a = ctx.stream.clone_dtoh(&a.data).expect("a D2H");
        let b = ctx.stream.clone_dtoh(&b.data).expect("b D2H");
        assert_eq!(a.len(), b.len(), "{what} length mismatch");
        let diffs = a
            .iter()
            .zip(&b)
            .enumerate()
            .filter(|(_, (u, v))| u.to_bits() != v.to_bits())
            .take(4)
            .map(|(i, (u, v))| format!("[{i}] {u:?} vs {v:?}"))
            .collect::<Vec<_>>();
        assert!(diffs.is_empty(), "{what} diverges: {diffs:?}");
    }

    fn seeded(ctx: &DeviceContext, d: usize, rows: usize, seed: usize) -> HiddenStates {
        let host: Vec<bf16> = (0..d * rows)
            .map(|i| bf16::from_f32((((i * 131 + seed * 17) % 4093) as f32 - 2046.0) / 512.0))
            .collect();
        let mut x = HiddenStates::zeros(ctx, d, rows).expect("buf");
        ctx.stream.memcpy_htod(&host, &mut x.data).expect("up");
        x
    }

    fn seeded_weight(ctx: &DeviceContext, d: usize, seed: usize) -> DeviceVec {
        let mut host: Vec<bf16> = (0..d)
            .map(|i| bf16::from_f32((((i * 37 + seed * 13) % 511) as f32 - 255.0) / 256.0))
            .collect();
        host[0] = bf16::from_f32(-0.0);
        DeviceVec {
            data: ctx.stream.clone_htod(&host).expect("w up"),
            len: d,
        }
    }

    #[test]
    #[ignore = "requires a GPU"]
    fn the_dual_norm_matches_two_standalone_norms() {
        let ctx = DeviceContext::new_with_device(0).expect("device");
        let (d, rows, eps) = (2816usize, 16usize, 1e-6f32);
        let x = seeded(&ctx, d, rows, 4);
        let wa = seeded_weight(&ctx, d, 4);
        let wb = seeded_weight(&ctx, d, 5);
        let scale = (d as f32).powf(-0.5);
        let mut split_a = HiddenStates::zeros(&ctx, d, rows).expect("sa");
        let mut split_b = HiddenStates::zeros(&ctx, d, rows).expect("sb");
        rms_norm_batch_into(&ctx, &x, &wa, eps, &mut split_a);
        rms_norm_batch_into(&ctx, &x, &wb, eps, &mut split_b);
        crate::ops::scale_bf16_in_place(&ctx, &mut split_a, scale).expect("scale");

        let mut fused_a = HiddenStates::zeros(&ctx, d, rows).expect("fa");
        let mut fused_b = HiddenStates::zeros(&ctx, d, rows).expect("fb");
        rms_norm_batch_dual_into(&ctx, &x, &wa, &wb, eps, scale, &mut fused_a, &mut fused_b)
            .expect("fused dual norm");
        assert_bitwise(&ctx, &split_a, &fused_a, "dual norm router branch");
        assert_bitwise(&ctx, &split_b, &fused_b, "dual norm expert branch");
    }

    #[test]
    #[ignore = "requires a GPU"]
    fn the_layer_tail_matches_its_parts() {
        let ctx = DeviceContext::new_with_device(0).expect("device");
        let (d, rows, eps) = (2816usize, 16usize, 1e-6f32);
        let x = seeded(&ctx, d, rows, 0);
        let residual = seeded(&ctx, d, rows, 1);
        let weight = seeded_weight(&ctx, d, 0);
        let scale = 0.7383f32;

        let mut normed = HiddenStates::zeros(&ctx, d, rows).expect("normed");
        rms_norm_batch_into(&ctx, &x, &weight, eps, &mut normed);
        let mut split = HiddenStates::zeros(&ctx, d, rows).expect("split");
        crate::ops::add_batch_into(&ctx, &residual, &normed, &mut split).expect("add");
        crate::ops::scale_bf16_in_place(&ctx, &mut split, scale).expect("scale");

        let mut fused = HiddenStates::zeros(&ctx, d, rows).expect("fused");
        rms_norm_add_scale_batch_into(&ctx, &x, &weight, &residual, scale, eps, &mut fused)
            .expect("fused tail");
        assert_bitwise(&ctx, &split, &fused, "fused layer tail");
    }

    #[test]
    #[ignore = "requires a GPU"]
    fn the_epilogue_norm_pair_matches_its_parts() {
        let ctx = DeviceContext::new_with_device(0).expect("device");
        let (d, rows, eps) = (2816usize, 16usize, 1e-6f32);
        let attn = seeded(&ctx, d, rows, 2);
        let x = seeded(&ctx, d, rows, 3);
        let w_post = seeded_weight(&ctx, d, 2);
        let w_pre = seeded_weight(&ctx, d, 3);
        // The standalone chain the epilogue ran: norm, a separate bf16 add,
        // then the second norm reading the rounded sum.
        let mut normed = HiddenStates::zeros(&ctx, d, rows).expect("normed");
        rms_norm_batch_into(&ctx, &attn, &w_post, eps, &mut normed);
        let mut residual = HiddenStates::zeros(&ctx, d, rows).expect("residual");
        crate::ops::add_batch_into(&ctx, &x, &normed, &mut residual).expect("add");
        let mut mlp_in = HiddenStates::zeros(&ctx, d, rows).expect("mlp_in");
        rms_norm_batch_into(&ctx, &residual, &w_pre, eps, &mut mlp_in);

        let mut fused_res = HiddenStates::zeros(&ctx, d, rows).expect("fr");
        let mut fused_mlp = HiddenStates::zeros(&ctx, d, rows).expect("fm");
        rms_norm_add_rms_norm_round_batch_into(
            &ctx,
            &attn,
            &w_post,
            &x,
            &w_pre,
            eps,
            &mut fused_res,
            &mut fused_mlp,
        )
        .expect("fused pair");
        assert_bitwise(&ctx, &residual, &fused_res, "fused epilogue residual");
        assert_bitwise(&ctx, &mlp_in, &fused_mlp, "fused epilogue mlp input");
    }

    #[test]
    #[ignore = "requires a GPU"]
    fn the_moe_combine_tail_matches_its_parts() {
        let ctx = DeviceContext::new_with_device(0).expect("device");
        let (d, rows, eps) = (2816usize, 16usize, 1e-6f32);
        let a = seeded(&ctx, d, rows, 0);
        let b = seeded(&ctx, d, rows, 1);
        let wa = seeded_weight(&ctx, d, 0);
        let wb = seeded_weight(&ctx, d, 1);

        let mut na = HiddenStates::zeros(&ctx, d, rows).expect("na");
        let mut nb = HiddenStates::zeros(&ctx, d, rows).expect("nb");
        rms_norm_batch_into(&ctx, &a, &wa, eps, &mut na);
        rms_norm_batch_into(&ctx, &b, &wb, eps, &mut nb);
        let mut split = HiddenStates::zeros(&ctx, d, rows).expect("split");
        crate::ops::add_batch_into(&ctx, &na, &nb, &mut split).expect("add");

        let mut fused = HiddenStates::zeros(&ctx, d, rows).expect("fused");
        dual_rms_norm_add_batch_into(&ctx, &a, &wa, &b, &wb, eps, &mut fused)
            .expect("fused combine");
        assert_bitwise(&ctx, &split, &fused, "fused MoE combine");
    }
}
