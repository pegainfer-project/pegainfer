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

    /// Batched grouped `(1+weight)` RMSNorm; see
    /// `ops::rms_norm_grouped_offset_batch_into` for the contract.
    pub fn rms_norm_grouped_offset_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        group_size: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;
}
