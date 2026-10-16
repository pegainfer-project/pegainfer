// Qwen3.8-Flash-Next (Qwen4 line) PLE block kernels. Geometry is frozen in
// `csrc/qwen38fn/qwen38fn_ple_block.cu`, and every entry point refuses a call
// whose stream width, group size, branch count or conv kernel disagrees with
// what it was compiled for.
//
// The block's three grouped `(1 + w)` RMSNorms are not here: they are the
// existing batched entry point `rms_norm_batched_offset_cuda`, called once per
// branch from `ops::qwen38fn_ple`.

use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use super::Half;

unsafe extern "C" {
    /// Row-major `[seq_len, width]` into branch-major
    /// `[hc_count][seq_len][hidden]`, so the grouped norms run on contiguous
    /// groups. `width` is the caller's row stride and is refused unless it is
    /// the compiled one.
    pub fn qwen38fn_ple_split_branches_cuda(
        row_major: *const Half,
        branch_major: *mut Half,
        seq_len: i32,
        width: i32,
        hidden: i32,
        hc_count: i32,
        stream: CUstream,
    ) -> CUresult;

    /// One gate scalar per (row, branch) — the per-branch dot, its double
    /// rounding around the `1 / sqrt_hidden` divide, the signed sqrt and the
    /// sigmoid — times `value`, broadcast over the branches that share it.
    ///
    /// `key_normed`, `query_normed` and `gated` are branch-major; `value` is
    /// `[seq_len, hidden]`. `gate_span` is `hidden` for the reference's
    /// per-branch dot and `width` for the full-stream negative control;
    /// `signed_sqrt` is 1 for the reference and 0 for the unsigned control.
    pub fn qwen38fn_ple_gate_cuda(
        key_normed: *const Half,
        query_normed: *const Half,
        value: *const Half,
        gated: *mut Half,
        seq_len: i32,
        hidden: i32,
        hc_count: i32,
        gate_span: i32,
        sqrt_hidden: f32,
        signed_sqrt: i32,
        stream: CUstream,
    ) -> CUresult;

    /// `out = gated + silu(bf16(dilated causal depthwise conv(conv_in)))`.
    /// `conv_in` and `gated` are branch-major, `out` is row-major
    /// `[seq_len, width]`, `conv_weight` is `[width, 1, conv_kernel]` flattened.
    /// The zero initial state of `(conv_kernel - 1) * dilation` rows is a bound
    /// inside the kernel, not a buffer.
    pub fn qwen38fn_ple_conv_residual_cuda(
        conv_in: *const Half,
        gated: *const Half,
        conv_weight: *const Half,
        out: *mut Half,
        seq_len: i32,
        hidden: i32,
        hc_count: i32,
        conv_kernel: i32,
        dilation: i32,
        stream: CUstream,
    ) -> CUresult;
}
