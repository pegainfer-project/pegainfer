// Qwen3.8-Flash-Next (Qwen4 line) QSA token-indexer kernels. Geometry is frozen
// in `csrc/qwen38fn/qwen38fn_qsa_indexer_score.cu`, which refuses a call whose
// rotary geometry disagrees with what it was compiled for.

use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use super::Half;

unsafe extern "C" {
    /// Pool the complete micro-blocks of `fused_qk` and score them for every
    /// query row. `pooled_keys` is the pool pass's workspace.
    ///
    /// `fused_width` is the caller's row stride for `fused_qk`; the body reads
    /// with the width it was compiled for, so a mismatch is refused rather than
    /// read out of bounds.
    pub fn qwen38fn_qsa_indexer_block_scores_cuda(
        fused_qk: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        rows: *const i32,
        visible_len: *const i32,
        scores: *mut f32,
        pooled_keys: *mut Half,
        query_rows: i32,
        tokens: i32,
        score_cols: i32,
        fused_width: i32,
        rotary_dim: i32,
        rope_theta: f32,
        rms_eps: f32,
        stream: CUstream,
    ) -> CUresult;
}
