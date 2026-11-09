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
        score_splits: i32,
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

    /// Select the kept micro-blocks of every query row plus its incomplete
    /// tail, and write the row's token visibility mask (one byte per cell,
    /// `0` or `1`).
    ///
    /// `rows[j]` is the row's position, `scores` its `score_cols`-wide f32
    /// block-score row and `mask` its `tokens`-wide byte row. `budget` is the
    /// token budget. The compiled geometry is four tokens per block and one
    /// byte per mask cell; a caller whose buffers are laid out differently
    /// cannot be detected here.
    pub fn qwen4_qsa_indexer_select_cuda(
        scores: *const f32,
        rows: *const i32,
        mask: *mut u8,
        query_rows: i32,
        tokens: i32,
        score_cols: i32,
        budget: i32,
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

    /// Scatter a sublayer output back into the `branches * hidden` stream:
    /// `out[s, b*hidden + h] = bf16(hyper[s, b*hidden + h] +
    /// bf16(f32(x[s, h]) * f32(inject[s, b])))`. `out` may alias `hyper`.
    pub fn qwen4_hc_scatter_cuda(
        x: *const Half,
        inject: *const Half,
        hyper: *const Half,
        out: *mut Half,
        rows: i32,
        hidden: i32,
        branches: i32,
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
