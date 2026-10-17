//! Qwen3.8-Flash-Next hyper-connection gated residual: the elementwise tail.
//!
//! `Qwen4ExpTextGatedResidual` composes a grouped `(1 + w)` RMSNorm, three
//! cuBLAS projections and four elementwise steps. The norm and the projections
//! already have shared entry points ([`crate::ops::rms_norm_batch_offset_into`]
//! and [`crate::ops::gemm_into_checked`]). A standalone sigmoid and the branch
//! combine had no entry point at all, so those two are here.
//!
//! The scaled SiLU is the exception: an equivalent already exists on default
//! features as two launches — [`crate::ops::scale_bf16_in_place`] followed by
//! [`crate::ops::silu_mul_batch_into`] against a buffer of ones.
//! [`hc_silu_scaled_into`] keeps the single launch that fuses scale and SiLU,
//! and the two agree bitwise, because the reference's scale is a power of two
//! and the intermediate bf16 rounding the two-launch form adds is then exact.
//! It is not a second implementation of SiLU to keep in sync by hand: the
//! `x / (1 + expf(-x))` body is the tree's, shared with the five other silus
//! under `csrc/`.
//!
//! They are not fused into one kernel on purpose: the reference materialises a
//! bf16 tensor per step, so keeping the steps separate is what lets the caller
//! put a projection between them and still land every rounding where the
//! reference lands it. The contract each one honours is documented on
//! `csrc/qwen38fn/qwen38fn_hc_block.cu`.
//!
//! "hc" below is the reference's `hc_count` hyper-connection stream: `B`
//! branches of `hidden` features, one row-major `[rows, B * hidden]` tensor.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys::CUresult;

use crate::ffi;
use crate::ops::checked_i32;
use crate::tensor::DeviceContext;
use crate::tensor::HiddenStates;
use crate::tensor::active_cu_stream;

fn ensure_finite(value: f32, what: &str) -> Result<()> {
    ensure!(
        value.is_finite(),
        "hyper-connection {what} {value} is not finite"
    );
    Ok(())
}

fn launch(result: CUresult, what: &str) -> Result<()> {
    ensure!(
        result == CUresult::CUDA_SUCCESS,
        "hyper-connection {what} launch failed: {result:?}"
    );
    Ok(())
}

/// `out = bf16(silu(f32(x) * scale))`, elementwise.
///
/// The mix path's `silu(input_mix_weight_down(n) / hc_count)`: `hc_count` is a
/// power of two, so dividing the bf16 projection output by it is exact and the
/// caller folds it into `scale`, leaving the SiLU's own cast as the only
/// rounding — which is where the reference's lands.
pub fn hc_silu_scaled_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    scale: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure_finite(scale, "silu scale")?;
    ensure!(
        out.hidden_dim == x.hidden_dim && out.seq_len == x.seq_len,
        "hyper-connection silu output [{}, {}] does not match input [{}, {}]",
        out.hidden_dim,
        out.seq_len,
        x.hidden_dim,
        x.seq_len
    );
    let n = x.checked_extent("hyper-connection silu input")?;
    out.checked_extent("hyper-connection silu output")?;
    let n = checked_i32(n, "hyper-connection silu element count")?;
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::qwen38fn_hc_silu_scaled_cuda(
            x_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n,
            scale,
            active_cu_stream(ctx),
        )
    };
    launch(result, "silu_scaled")
}

/// `out = bf16(gain * f32(bf16(sigmoid(f32(x) * scale))))`, elementwise.
///
/// Two reference steps share it: `scale = 1, gain = 1` is the mix path's
/// `sigmoid(input_mix_weight_up(g))`, and `scale = 1/hc_count, gain = 2` is the
/// injection weights `2 * sigmoid(block_inject_weight(n) / hc_count)`. The
/// inner cast is the reference's own — `torch.sigmoid` on a bf16 tensor
/// materialises bf16 before anything reads it.
pub fn hc_sigmoid_affine_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    scale: f32,
    gain: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure_finite(scale, "sigmoid scale")?;
    ensure_finite(gain, "sigmoid gain")?;
    ensure!(
        out.hidden_dim == x.hidden_dim && out.seq_len == x.seq_len,
        "hyper-connection sigmoid output [{}, {}] does not match input [{}, {}]",
        out.hidden_dim,
        out.seq_len,
        x.hidden_dim,
        x.seq_len
    );
    let n = x.checked_extent("hyper-connection sigmoid input")?;
    out.checked_extent("hyper-connection sigmoid output")?;
    let n = checked_i32(n, "hyper-connection sigmoid element count")?;
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::qwen38fn_hc_sigmoid_affine_cuda(
            x_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n,
            scale,
            gain,
            active_cu_stream(ctx),
        )
    };
    launch(result, "sigmoid_affine")
}

/// Combine `branches` gated branches of a hyper-connection stream:
///
/// ```text
/// out[r, c] = bf16(combine_scale * sum_j f32(bf16(a[r, j*hidden+c] * b[r, j*hidden+c])))
/// ```
///
/// `a` and `b` are row-major `[rows, branches * hidden]`, `out` is
/// `[rows, hidden]`. The inner cast is the reference's bf16 `gate * normed`
/// product; the f32 accumulate and the single cast out are torch's
/// `.mean(dim=-2)`, so `combine_scale = 1.0 / branches as f32` reproduces it.
/// A different `combine_scale` is a caller asking for a different reduce — the
/// parity gate's negative control passes `1.0` for the sum.
pub fn hc_mul_branch_combine_into(
    ctx: &DeviceContext,
    a: &HiddenStates,
    b: &HiddenStates,
    branches: usize,
    combine_scale: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure_finite(combine_scale, "combine scale")?;
    ensure!(
        branches > 0,
        "hyper-connection branch count must be positive"
    );
    ensure!(
        a.hidden_dim == b.hidden_dim,
        "hyper-connection combine reads mismatched streams: {} against {}",
        a.hidden_dim,
        b.hidden_dim
    );
    ensure!(
        a.hidden_dim == branches * out.hidden_dim,
        "hyper-connection combine: {branches} branches of {} do not span a {}-wide stream",
        out.hidden_dim,
        a.hidden_dim
    );
    ensure!(
        a.seq_len == b.seq_len && a.seq_len == out.seq_len,
        "hyper-connection combine row counts disagree: {} / {} / {}",
        a.seq_len,
        b.seq_len,
        out.seq_len
    );
    // The shape checks above make `a`'s logical extent `rows * branches *
    // hidden` and `out`'s `rows * hidden` — exactly the two element counts the
    // launcher bounds, so this is what keeps the kernel's source index inside
    // the backing allocations.
    a.checked_extent("hyper-connection combine gate stream")?;
    b.checked_extent("hyper-connection combine normed stream")?;
    out.checked_extent("hyper-connection combine output")?;
    let branches_i32 = checked_i32(branches, "hyper-connection branch count")?;
    let rows = checked_i32(a.seq_len, "hyper-connection combine row count")?;
    let hidden = checked_i32(out.hidden_dim, "hyper-connection combine branch width")?;
    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::qwen38fn_hc_mul_branch_combine_cuda(
            a_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            rows,
            hidden,
            branches_i32,
            combine_scale,
            active_cu_stream(ctx),
        )
    };
    launch(result, "mul_branch_combine")
}
