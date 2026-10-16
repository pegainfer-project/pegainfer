// Qwen3.8-Flash-Next (Qwen4 line) QSA kernels. Geometry is compiled into the
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
    /// The QSA sparse-attention forward over one layer's captured activations:
    /// the key pass, then one CTA per (query row, head) for the scores, the
    /// weights and the gated context. `normed_keys` is the key pass's workspace.
    pub fn qwen4_qsa_attn_forward_cuda(
        queries: *const Half,
        keys: *const Half,
        values: *const Half,
        mask: *const u8,
        rows: *const i32,
        positions: *const i32,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        attn_weights: *mut Half,
        attn_out: *mut Half,
        normed_keys: *mut Half,
        query_rows: i32,
        tokens: i32,
        rotary_dim: i32,
        rope_theta: f32,
        rms_eps: f32,
        stream: CUstream,
    ) -> CUresult;
}
