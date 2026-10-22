// Qwen3.8-Flash-Next (Qwen4 line) kernels. Geometry is compiled into the
// matching `csrc/qwen4/qwen4_*.cu`; the entry points take only the per-call
// scalars noted on each.

use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use super::Half;

unsafe extern "C" {
    /// Pool the complete micro-blocks of `fused_qk` and score them for every
    /// query row. `pooled_keys` is the pool pass's workspace.
    ///
    /// The geometry is compiled in: the fused row is `640` wide and the cos/sin
    /// tables stride `64`. A caller whose buffers are laid out differently is
    /// reading the wrong row, which the buffer-size checks in the wrapper cannot
    /// see; the checkpoint's config is checked where the model crate reads it.
    pub fn qwen4_qsa_indexer_block_scores_cuda(
        fused_qk: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_table: *const Half,
        sin_table: *const Half,
        rows: *const i32,
        visible_len: *const i32,
        scores: *mut f32,
        pooled_keys: *mut Half,
        query_rows: i32,
        tokens: i32,
        score_cols: i32,
        rms_eps: f32,
        stream: CUstream,
    ) -> CUresult;
}

unsafe extern "C" {
    /// `out = bf16(silu(f32(x) * scale))` over `n` elements — the mix path's
    /// `silu(down / hc_count)`, with the exact power-of-two division folded
    /// into `scale`.
    pub fn qwen4_hc_silu_scaled_cuda(
        x: *const Half,
        out: *mut Half,
        n: i32,
        scale: f32,
        stream: CUstream,
    ) -> CUresult;

    /// `out = bf16(gain * f32(bf16(sigmoid(f32(x) * scale))))` over `n`
    /// elements. `scale = 1, gain = 1` is the mix path's `sigmoid(up)`;
    /// `scale = 1/hc_count, gain = 2` is the reference's injection weights.
    pub fn qwen4_hc_sigmoid_affine_cuda(
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
    pub fn qwen4_hc_mul_branch_combine_cuda(
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
