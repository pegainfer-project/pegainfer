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

    /// Gated attention for `rows` consecutive query rows of one sequence over
    /// the tokens `mask` keeps, reading the sequence's K/V pages from
    /// `first_page` on. See `ops::qsa_sparse_attention_into` for the contract.
    ///
    /// The geometry is compiled in: 24 query and 2 K/V heads at head dim 256,
    /// pages of 16 tokens, and at most 2051 kept tokens a row (the device traps
    /// past that).
    pub fn qwen4_qsa_sparse_attention_cuda(
        q_full: *const Half,
        q_prepped: *const Half,
        pool: *const Half,
        mask: *const u8,
        out: *mut Half,
        rows: i32,
        first_position: i32,
        first_page: i32,
        page_stride: i64,
        kv_block_len: i64,
        mask_stride: i32,
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

    /// `out[r, c] = bf16(sum_j f32(bf16(a[r, j*hidden+c] * b[r, j*hidden+c]))
    /// / branches)`, with `a`/`b` row-major `[rows, branches*hidden]` and `out`
    /// `[rows, hidden]` — the branch combine, the reference's `.mean(-2)`.
    pub fn qwen4_hc_mul_branch_combine_cuda(
        a: *const Half,
        b: *const Half,
        out: *mut Half,
        rows: i32,
        hidden: i32,
        branches: i32,
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
    /// `[rows, top_k]`. `renorm` is a 0/1 flag.
    pub fn qwen4_moe_route_cuda(
        logits: *const Half,
        weights: *mut Half,
        ids: *mut i32,
        rows: i32,
        experts: i32,
        top_k: i32,
        renorm: i32,
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

unsafe extern "C" {
    /// One gate scalar per (row, branch) — the per-branch dot, its double
    /// rounding around the `1 / sqrt_hidden` divide, the signed sqrt and the
    /// sigmoid — times `value`, broadcast over the branches that share it.
    ///
    /// `key_normed`, `query_normed` and `gated` are the row-major
    /// `[seq_len, hc_count * hidden]` stream; `value` is `[seq_len, hidden]`.
    pub fn qwen4_ple_gate_cuda(
        key_normed: *const Half,
        query_normed: *const Half,
        value: *const Half,
        gated: *mut Half,
        seq_len: i32,
        hidden: i32,
        hc_count: i32,
        sqrt_hidden: f32,
        stream: CUstream,
    ) -> CUresult;

    /// `out = gated + silu(bf16(dilated causal depthwise conv(conv_in)))`.
    /// `conv_in`, `gated` and `out` are the row-major
    /// `[seq_len, hc_count * hidden]` stream, `conv_weight` is
    /// `[width, 1, conv_kernel]` flattened. The zero initial state of
    /// `(conv_kernel - 1) * dilation` rows is a bound inside the kernel, not a
    /// buffer.
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

unsafe extern "C" {
    /// Orders the `[rows, top_k]` expert ids by expert: `indptr` (`experts + 1`)
    /// holds each expert's first slot, `gather_rows` the token of each slot and
    /// `route_slot` the slot of each (token, pick) pair.
    pub fn qwen4_moe_route_meta_cuda(
        ids: *const i32,
        indptr: *mut i32,
        gather_rows: *mut i32,
        route_slot: *mut i32,
        rows: i32,
        top_k: i32,
        experts: i32,
        stream: CUstream,
    ) -> CUresult;

    /// `out[t] = sum_j w[t, j] * down_out[route_slot[t, j]]`, f32 accumulation,
    /// one bf16 rounding. `down_out` is `[rows * top_k, hidden]`, `weights`
    /// `[rows, top_k]`, `out` `[rows, hidden]`.
    pub fn qwen4_moe_weighted_combine_cuda(
        down_out: *const Half,
        weights: *const Half,
        route_slot: *const i32,
        out: *mut Half,
        tokens: i32,
        top_k: i32,
        hidden: i32,
        stream: CUstream,
    ) -> CUresult;

    /// The same ordering as [`qwen4_moe_route_meta_cuda`] as a histogram, an
    /// exclusive scan and a deterministic placement, with work proportional to
    /// the pair count instead of `experts * pairs`. `chunk_hist` is
    /// `ceil(pairs / 256) * experts` i32 of scratch (the scan leaves the
    /// per-chunk prefix there). Bit-identical output.
    pub fn qwen4_moe_route_meta_tiled_cuda(
        ids: *const i32,
        indptr: *mut i32,
        gather_rows: *mut i32,
        route_slot: *mut i32,
        chunk_hist: *mut i32,
        rows: i32,
        top_k: i32,
        experts: i32,
        chunk_hist_entries: i64,
        stream: CUstream,
    ) -> CUresult;

    /// Bytes the grouped GEMM's per-group argument scratch needs for `experts`
    /// groups.
    pub fn qwen4_moe_grouped_args_bytes(experts: i32) -> usize;

    /// The CTA count the grouped kernel schedules with on this device.
    pub fn qwen4_moe_grouped_threadblocks() -> i32;

    /// One grouped GEMM over the experts: group `e` multiplies the activation
    /// rows `indptr[e]..indptr[e + 1]` of `x` against weight slab `e` and
    /// writes the same row range of `out`. `args` is
    /// `qwen4_moe_grouped_args_bytes(experts)` bytes of scratch.
    pub fn qwen4_moe_grouped_gemm_cuda(
        x: *const Half,
        w: *const Half,
        out: *mut Half,
        indptr: *const i32,
        args: *mut core::ffi::c_void,
        args_bytes: usize,
        experts: i32,
        n: i32,
        k: i32,
        threadblocks: i32,
        stream: CUstream,
    ) -> CUresult;
}
