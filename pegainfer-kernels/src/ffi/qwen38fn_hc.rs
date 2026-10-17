// Qwen3.8-Flash-Next (Qwen4 line) hyper-connection gated residual: the
// elementwise tail the shared surface has no default-feature entry point for.
// The rounding contract these three honour is documented on the bodies in
// `csrc/qwen38fn/qwen38fn_hc_block.cu`.

use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use super::Half;

unsafe extern "C" {
    /// `out = bf16(silu(f32(x) * scale))` over `n` elements — the mix path's
    /// `silu(down / hc_count)`, with the exact power-of-two division folded
    /// into `scale`.
    pub fn qwen38fn_hc_silu_scaled_cuda(
        x: *const Half,
        out: *mut Half,
        n: i32,
        scale: f32,
        stream: CUstream,
    ) -> CUresult;

    /// `out = bf16(gain * f32(bf16(sigmoid(f32(x) * scale))))` over `n`
    /// elements. `scale = 1, gain = 1` is the mix path's `sigmoid(up)`;
    /// `scale = 1/hc_count, gain = 2` is the reference's injection weights.
    pub fn qwen38fn_hc_sigmoid_affine_cuda(
        x: *const Half,
        out: *mut Half,
        n: i32,
        scale: f32,
        gain: f32,
        stream: CUstream,
    ) -> CUresult;

    /// `out[r, c] = bf16(combine_scale * sum_j f32(bf16(a[r, j*hidden+c] *
    /// b[r, j*hidden+c])))`, with `a`/`b` row-major `[rows, branches*hidden]`
    /// and `out` `[rows, hidden]` — the branch combine. `combine_scale =
    /// 1/branches` is the reference's `.mean(-2)`; anything else is a caller
    /// asking for a different reduce.
    pub fn qwen38fn_hc_mul_branch_combine_cuda(
        a: *const Half,
        b: *const Half,
        out: *mut Half,
        rows: i32,
        hidden: i32,
        branches: i32,
        combine_scale: f32,
        stream: CUstream,
    ) -> CUresult;
}
