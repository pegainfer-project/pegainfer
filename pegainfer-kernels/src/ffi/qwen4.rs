// Qwen3.8-Flash-Next (Qwen4 line) QSA token-indexer kernels. Geometry is compiled
// into `csrc/qwen4/qwen4_qsa_indexer_score.cu`; the entry point takes none.

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
    /// Row-major `[seq_len, width]` into branch-major
    /// `[hc_count][seq_len][hidden]`, so the grouped norms run on contiguous
    /// groups. `width` is the caller's row stride and is refused unless it is
    /// the compiled one.
    pub fn qwen4_ple_split_branches_cuda(
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
    pub fn qwen4_ple_gate_cuda(
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
    pub fn qwen4_ple_conv_residual_cuda(
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
