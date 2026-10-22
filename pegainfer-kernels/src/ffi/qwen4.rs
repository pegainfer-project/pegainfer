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
    /// fp32 softmax over the router logits of one token row, top-10 in the
    /// canonical order (descending probability, ascending index inside a tie),
    /// optional `norm_topk_prob` renormalization, bf16 weights and i32 expert
    /// ids. `logits` is `[rows, experts]` bf16; `weights` / `ids` are
    /// `[rows, top_k]`. `renorm` and `probs_bf16` are 0/1 flags; `probs_bf16`
    /// is the parity gate's negative control and must be 0 in production.
    pub fn qwen4_moe_route_cuda(
        logits: *const Half,
        weights: *mut Half,
        ids: *mut i32,
        rows: i32,
        experts: i32,
        top_k: i32,
        renorm: i32,
        probs_bf16: i32,
        stream: CUstream,
    ) -> CUresult;

    /// `out = bf16(expert_out + bf16(bf16(sigmoid(gate_logit)) * shared_out))`
    /// per token — the block sum with the shared expert's learned scalar
    /// sigmoid gate. `gate_logit` is `[tokens]` bf16; the three matrices are
    /// `[tokens, hidden]` bf16.
    pub fn qwen4_moe_shared_combine_cuda(
        gate_logit: *const Half,
        expert_out: *const Half,
        shared_out: *const Half,
        out: *mut Half,
        tokens: i32,
        hidden: i32,
        stream: CUstream,
    ) -> CUresult;
}
