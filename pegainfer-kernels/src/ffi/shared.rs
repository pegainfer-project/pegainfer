use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use super::Half;

// Shared kernels used across all models (CUDA / cuBLAS / FlashInfer).
unsafe extern "C" {
    pub fn rms_norm_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    );

    pub fn rms_norm_batched_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    );

    pub fn rms_norm_batched_dual_cuda(
        x: *const Half,
        weight_a: *const Half,
        weight_b: *const Half,
        out_a: *mut Half,
        out_b: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        scale_a: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dual_rms_norm_add_batched_cuda(
        a: *const Half,
        weight_a: *const Half,
        b: *const Half,
        weight_b: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn rms_norm_add_rms_norm_round_batched_cuda(
        x: *const Half,
        weight_post: *const Half,
        res_in: *const Half,
        weight_pre: *const Half,
        residual_out: *mut Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn rms_norm_add_scale_batched_cuda(
        x: *const Half,
        weight: *const Half,
        residual: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn add_cuda(
        a: *const Half,
        b: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn advance_decode_metadata_cuda(
        positions: *mut i32,
        local_last: *mut i32,
        pseudo_last: *mut i32,
        kv_chunk: *mut i32,
        rows: i32,
        factor: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn add_scaled_bf16_cuda(
        routed: *const Half,
        scale: f32,
        shared: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn extract_hidden_rows_cuda(
        src: *const Half,
        dst: *mut Half,
        src_hidden_dim: i32,
        dst_hidden_dim: i32,
        col_offset: i32,
        rows: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn copy_hidden_rows_cuda(
        src: *const Half,
        dst: *mut Half,
        src_hidden_dim: i32,
        dst_hidden_dim: i32,
        row_offset: i32,
        rows: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn mask_position_zero_rows_cuda(
        src: *const Half,
        positions: *const u32,
        dst: *mut Half,
        hidden_dim: i32,
        rows: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn copy_hidden_token_range_cuda(
        src: *const Half,
        dst: *mut Half,
        hidden_dim: i32,
        src_token_offset: i32,
        dst_token_offset: i32,
        token_count: i32,
        src_seq_len: i32,
        dst_seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn fused_add_rms_norm_cuda(
        hidden: *mut Half,
        residual: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    );

    pub fn layer_norm_cuda(
        x: *const Half,
        gamma: *const f32,
        beta: *const f32,
        out: *mut Half,
        n: i32,
        rows: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn fused_add_rms_norm_batched_cuda(
        hidden: *mut Half,
        residual: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        batch_size: i32,
        eps: f32,
        stream: CUstream,
    );

    pub fn silu_mul_triton_aot_cuda(
        gate: *const Half,
        up: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gelu_tanh_mul_cuda(
        gate: *const Half,
        up: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn scale_bf16_in_place_cuda(
        buf: *mut Half,
        scale: f32,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn softcap_bf16_in_place_cuda(
        buf: *mut Half,
        cap: f32,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn suppress_logits_bf16_in_place_cuda(
        logits: *mut Half,
        ids: *const u32,
        vocab: i32,
        rows: i32,
        id_count: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn embedding_batched_cuda(
        embed: *const Half,
        token_ids: *const u32,
        out: *mut Half,
        hidden_size: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn embedding_batched_vocab_shard_cuda(
        embed: *const Half,
        token_ids: *const u32,
        out: *mut Half,
        hidden_size: i32,
        seq_len: i32,
        vocab_start: u32,
        part_vocab_size: u32,
        stream: CUstream,
    ) -> CUresult;

    pub fn argmax_cuda(x: *const Half, out: *mut i32, n: i32, stream: CUstream);

    pub fn logprob_topk_batch_bf16_cuda(
        x: *const Half,
        row_indices: *const i32,
        picked: *const i32,
        top_k: *const i32,
        out_picked_lp: *mut f32,
        out_topk_vals: *mut f32,
        out_topk_ids: *mut i32,
        rows: i32,
        n: i32,
        k_max: i32,
        stream: CUstream,
    );

    pub fn flashinfer_top1_cuda(
        logits: *const Half,
        top1_value_scratch: *mut Half,
        row_states_scratch: *mut u8,
        output: *mut i32,
        vocab_size: i32,
        stream: CUstream,
    );

    pub fn flashinfer_top1_row_states_bytes_cuda() -> usize;

    pub fn gpu_sample_topk_renorm_row_states_bytes_cuda() -> usize;

    pub fn gpu_sample_batch_flashinfer_cuda(
        logits: *const Half,
        row_indices: *const i32,
        probs_scratch: *mut f32,
        temperature_arr: *const f32,
        top_k_arr: *const i32,
        top_p_arr: *const f32,
        min_p_arr: *const f32,
        topk_row_states_scratch: *mut u8,
        valid_scratch: *mut u8,
        output: *mut i32,
        softmax_workspace: *mut u8,
        softmax_workspace_bytes: usize,
        n_rows: i32,
        vocab_size: i32,
        has_top_k_filter: i32,
        has_top_p_filter: i32,
        seed: u64,
        offset: u64,
        stream: CUstream,
    ) -> i32;

    pub fn gemm_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    ) -> i32;

    pub fn gemm_graphsafe_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    ) -> i32;

    pub fn gemm_lt_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    ) -> i32;

    /// cuBLAS `cublasGemmStridedBatchedEx` (bf16, workspace-free, graph-safe).
    pub fn gemm_strided_batched_bf16_cuda(
        op_a: i32,
        op_b: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const Half,
        lda: i32,
        stride_a: i64,
        b: *const Half,
        ldb: i32,
        stride_b: i64,
        c: *mut Half,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
        stream: CUstream,
    ) -> i32;

    /// cuBLAS `cublasGemmStridedBatchedEx` (f32 in/out, f32 compute). `beta`
    /// selects overwrite (0.0) vs accumulate-into-C (1.0).
    pub fn gemm_strided_batched_f32_cuda(
        op_a: i32,
        op_b: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const f32,
        lda: i32,
        stride_a: i64,
        b: *const f32,
        ldb: i32,
        stride_b: i64,
        beta: f32,
        c: *mut f32,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
        stream: CUstream,
    ) -> i32;

    pub fn gemm_bf16_f32_cuda(
        op_a: i32,
        op_b: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const Half,
        lda: i32,
        b: *const Half,
        ldb: i32,
        c: *mut f32,
        ldc: i32,
        stream: CUstream,
    ) -> i32;

    pub fn gemm_lt_tune_cuda(
        Ws: *const *const Half,
        num_ws: i32,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    ) -> i32;

    // Batch-invariant pinned-algo path (csrc/shared/linear.cu).
    pub fn gemm_lt_pin_tune_cuda(
        M: i32,
        rep_n: i32,
        K: i32,
        out_splitk: *mut i32,
        out_reduction_scheme: *mut i32,
    ) -> i32;
    pub fn gemm_lt_pin_check_cuda(M: i32, N: i32, K: i32) -> i32;

    pub fn gemm_lt_pin_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    ) -> i32;

    // Embedding lookup reading token_id from decode_meta[0] (CUDA Graph safe)
    pub fn embedding_decode_cuda(
        embed: *const Half,
        token_id: *const u32,
        out: *mut Half,
        hidden_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn silu_mul_fused_cuda(
        gate_up: *const Half,
        out: *mut Half,
        intermediate_size: i32,
        bs: i32,
        stream: CUstream,
    ) -> i32;
    pub fn split_qkv_cuda(
        qkv: *const Half,
        q: *mut Half,
        k: *mut Half,
        v: *mut Half,
        q_dim: i32,
        kv_dim: i32,
        tokens: i32,
        stream: CUstream,
    ) -> i32;

    pub fn cublas_init();
    pub fn cublas_activate_device_handles() -> i32;
    pub fn cublas_destroy();
    pub fn cuda_set_device(device_ordinal: i32) -> i32;

    // ========================================================================
    // RMSNorm variants (offset / gated)
    // ========================================================================

    // Batched (1+weight) RMSNorm — one block per token
    pub fn rms_norm_batched_offset_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    );

    // (1+weight) RMSNorm — Qwen3.5 / Gemma style
    pub fn rms_norm_offset_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    );

    // Per-head RMSNorm with F32 weight + SiLU gate
    pub fn rms_norm_gated_cuda(
        x: *const Half,
        weight: *const f32,
        gate: *const Half,
        out: *mut Half,
        num_heads: i32,
        head_dim: i32,
        eps: f32,
        stream: CUstream,
    );

    // ========================================================================
    // Paged attention (FlashInfer)
    // ========================================================================

    // Batched QK RMSNorm + RoPE for decode with per-request positions.
    pub fn qk_norm_rope_batched_decode_cuda(
        q: *mut Half,
        k: *mut Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        batch_size: i32,
        rms_eps: f32,
        cos_max_pos: i32,
        stream: CUstream,
    );

    pub fn dflash_qk_norm_rope_cuda(
        q: *mut Half,
        k: *mut Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        q_len: i32,
        k_len: i32,
        q_start_pos: i32,
        k_start_pos: i32,
        rms_eps: f32,
        cos_max_pos: i32,
        stream: CUstream,
    ) -> i32;

    // Plain RoPE (no QK-norm) for EAGLE-3 — no norm-weight / eps params.
    pub fn eagle3_rope_cuda(
        q: *mut Half,
        k: *mut Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        q_len: i32,
        k_len: i32,
        q_start_pos: i32,
        k_start_pos: i32,
        cos_max_pos: i32,
        stream: CUstream,
    ) -> i32;

    // Scatter contiguous KV → paged layout (one layer, FlashInfer prefill append).
    pub fn paged_kv_scatter_cuda(
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        src_k: *const Half,
        src_v: *const Half,
        batch_indices: *const i32,
        positions: *const i32,
        nnz: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        stride_page: i64,
        src_stride_n: i64,
        src_stride_h: i64,
        stream: CUstream,
    ) -> i32;

    // Return the number of Q tiles for batch prefill (needed to size plan arrays).
    pub fn batch_prefill_paged_num_tiles(
        seq_len: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;

    pub fn batch_prefill_paged_num_tiles_with_cta_tile_q(
        seq_len: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        cta_tile_q_override: i32,
    ) -> i32;

    // Return the CTA tile size for batch prefill planning.
    pub fn batch_prefill_cta_tile_q(
        total_seq_len: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;

    pub fn batch_prefill_cta_tile_q_with_override(
        total_seq_len: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        cta_tile_q_override: i32,
    ) -> i32;

    // Batch prefill with paged KV cache (FlashInfer BatchPrefill, causal, kNone).
    pub fn batch_prefill_paged_cuda(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        q_indptr: *const i32,
        request_indices: *const i32,
        qo_tile_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        total_num_rows: *const u32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        seq_len: i32,
        batch_size: i32,
        padded_batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn batch_prefill_paged_cuda_with_cta_tile_q(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        q_indptr: *const i32,
        request_indices: *const i32,
        qo_tile_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        total_num_rows: *const u32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        seq_len: i32,
        batch_size: i32,
        padded_batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        cta_tile_q_override: i32,
        stream: CUstream,
    ) -> i32;

    // Single-request prefill with contiguous HND KV cache (FlashInfer SinglePrefill, causal).
    pub fn single_prefill_cuda(
        q: *const Half,
        output: *mut Half,
        k_cache: *const Half,
        v_cache: *const Half,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        kv_len: i32,
        max_seq_len: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn single_prefill_nhd_noncausal_cuda(
        q: *const Half,
        output: *mut Half,
        k_cache: *const Half,
        v_cache: *const Half,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        kv_len: i32,
        max_seq_len: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn single_prefill_nhd_noncausal_cuda_hd64(
        q: *const Half,
        output: *mut Half,
        k_cache: *const Half,
        v_cache: *const Half,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        kv_len: i32,
        max_seq_len: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    // Causal NHD single-sequence prefill (same layout, causal mask).
    pub fn single_prefill_nhd_causal_cuda(
        q: *const Half,
        output: *mut Half,
        k_cache: *const Half,
        v_cache: *const Half,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        kv_len: i32,
        max_seq_len: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    // Single-query NHD decode over a contiguous KV cache (FlashInfer SingleDecode,
    // no partition-KV). Structurally one query, so there is no `seq_len` parameter.
    pub fn single_decode_nhd_cuda(
        q: *const Half,
        output: *mut Half,
        k_cache: *const Half,
        v_cache: *const Half,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_len: i32,
        max_seq_len: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    // Paged attention decode (FlashInfer BatchDecode, no partition-KV).
    pub fn paged_attention_decode_cuda(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        request_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    // Paged attention decode (FlashInfer BatchDecode, partition-KV / split-K).
    pub fn paged_attention_decode_split_kv_cuda(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        request_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        o_indptr: *const i32,
        block_valid_mask: *const u8,
        tmp_v: *mut Half,
        tmp_s: *mut f32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        batch_size: i32,
        padded_batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;
}

// HEAD_DIM=256 paged attention. Qwen3.5-4B calls the full-attention pair; the
// windowed pair carries the sliding-window mask for Gemma 4's local layers.
// `window_left` is an inclusive distance: an N-token window passes N - 1, and
// -1 degrades to full attention.
unsafe extern "C" {
    pub fn paged_attention_decode_cuda_hd256(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        request_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn paged_attention_decode_window_cuda_hd256(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        request_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        window_left: i32,
        stream: CUstream,
    ) -> i32;

    pub fn paged_attention_decode_split_kv_cuda_hd512(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        request_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        o_indptr: *const i32,
        block_valid_mask: *const u8,
        tmp_v: *mut Half,
        tmp_s: *mut f32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        batch_size: i32,
        padded_batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn batch_prefill_paged_cuda_hd256(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        q_indptr: *const i32,
        request_indices: *const i32,
        qo_tile_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        total_num_rows: *const u32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        seq_len: i32,
        batch_size: i32,
        padded_batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn batch_prefill_paged_window_cuda_hd256(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        q_indptr: *const i32,
        request_indices: *const i32,
        qo_tile_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        total_num_rows: *const u32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        seq_len: i32,
        batch_size: i32,
        padded_batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        window_left: i32,
        stream: CUstream,
    ) -> i32;
}

// Added during rebase onto main: generic dtype/scale helpers, batched argmax/top1,
// rms-norm-round variant, gemm-per-token.
unsafe extern "C" {
    pub fn argmax_batch_bf16_cuda(
        x: *const Half,
        values: *mut Half,
        indices: *mut i32,
        rows: i32,
        n: i32,
        stream: CUstream,
    );

    pub fn argmax_batch_bf16_split_indexed_cuda(
        x: *const Half,
        row_indices: *const i32,
        values: *mut Half,
        indices: *mut i32,
        partial_values: *mut f32,
        partial_indices: *mut i32,
        rows: i32,
        n: i32,
        stream: CUstream,
    );

    pub fn markov_step_argmax_cuda(
        base: *const Half,
        bias: *const Half,
        block_size: i32,
        step: i32,
        rows: i32,
        n: i32,
        partial_values: *mut f32,
        partial_indices: *mut i32,
        out_tokens: *mut u32,
        sampled_tokens: *mut u32,
        stream: CUstream,
    );

    pub fn bf16_to_f32_cuda(
        input: *const Half,
        output: *mut f32,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn f32_to_bf16_cuda(
        input: *const f32,
        output: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn flashinfer_top1_batch_cuda(
        logits: *const Half,
        top1_values: *mut Half,
        row_states_scratch: *mut u8,
        output: *mut i32,
        num_rows: i32,
        vocab_size: i32,
        stream: CUstream,
    );

    pub fn fused_add_rms_norm_round_batched_cuda(
        hidden: *mut Half,
        residual: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        batch_size: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gemm_per_token_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        batch: i32,
        K: i32,
        stream: CUstream,
    ) -> i32;

    pub fn repeat_f32_for_reduce_scatter_cuda(
        local: *const f32,
        repeated: *mut f32,
        local_elems: i32,
        world_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn scaled_add_rows_cuda(
        delta: *const Half,
        scale: f32,
        out: *mut Half,
        out_hidden_dim: i32,
        row_offset: i32,
        rows: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gather_hidden_tokens_cuda(
        input: *const Half,
        token_indices: *const i32,
        out: *mut Half,
        hidden_dim: i32,
        token_count: i32,
        input_seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn scaled_add_rows_indexed_cuda(
        delta: *const Half,
        scale: f32,
        token_indices: *const i32,
        out: *mut Half,
        out_hidden_dim: i32,
        row_offset: i32,
        rows: i32,
        token_count: i32,
        out_seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn store_rows_indexed_cuda(
        src: *const Half,
        token_indices: *const i32,
        out: *mut Half,
        out_hidden_dim: i32,
        row_offset: i32,
        rows: i32,
        token_count: i32,
        out_seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn scale_f32_cuda(values: *mut f32, scale: f32, n: i32, stream: CUstream) -> CUresult;

    pub fn accumulate_bf16_token_scaled_to_f32_cuda(
        token: *const Half,
        scale: f32,
        out: *mut f32,
        hidden_dim: i32,
        token_idx: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

}

// Added during rebase: split argmax variant.
unsafe extern "C" {
    pub fn argmax_batch_bf16_split_cuda(
        x: *const Half,
        values: *mut Half,
        indices: *mut i32,
        partial_values: *mut f32,
        partial_indices: *mut i32,
        rows: i32,
        n: i32,
        stream: CUstream,
    );

}

// hd256 single prefill (qwen35 full-attention layers, Gemma 4 local layers):
// csrc/shared/paged_attention.cu. K/V are a contiguous HND cache —
// k[head, pos, dim] with max_seq_len rows per head — not token-major NHD.
unsafe extern "C" {
    pub fn single_prefill_cuda_hd256(
        q: *const Half,
        output: *mut Half,
        k_cache: *const Half,
        v_cache: *const Half,
        num_qo_heads: i32,
        num_kv_heads: i32,
        seq_len: i32,
        kv_len: i32,
        max_seq_len: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;
}

// hd512 (Gemma 4 global layers): csrc/shared/paged_attention_hd512.cu
unsafe extern "C" {
    pub fn single_prefill_cuda_hd512(
        q: *const Half,
        output: *mut Half,
        k_cache: *const Half,
        v_cache: *const Half,
        num_qo_heads: i32,
        num_kv_heads: i32,
        seq_len: i32,
        kv_len: i32,
        max_seq_len: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn batch_prefill_paged_cuda_hd512(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        q_indptr: *const i32,
        request_indices: *const i32,
        qo_tile_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        total_num_rows: *const u32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        seq_len: i32,
        batch_size: i32,
        padded_batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;
}

// hd256 plain-w QK-norm + RoPE prep (Gemma 4 local layers):
// csrc/shared/prefill_attention_hd256_plain.cu. Contract and validation live
// on the Rust wrapper in ops::attention; the entry returns 0 on success, -1
// with a diagnostic on failure.
unsafe extern "C" {
    // Oracle form: Q and K both land in contiguous buffers shaped like
    // their inputs; the paged serving form is the qkv_ entry below.
    pub fn qk_norm_rope_prefill_hd256_plain_cuda(
        q_batch: *const Half,
        k_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        q_batch_out: *mut Half,
        k_batch_out: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        seq_len: i32,
        start_pos: i32,
        cos_max_pos: i32,
        rotary_dim: i32,
        rms_eps: f32,
        stream: CUstream,
    ) -> i32;

    // Paged serving form: Q → contiguous q_batch_out; K (normed + rotated)
    // and V (weightless-normed, never rotated) → straight into the paged KV
    // pool at k_offset_elems / v_offset_elems.
    pub fn qkv_norm_rope_paged_prefill_hd256_plain_cuda(
        q_batch: *const Half,
        k_batch: *const Half,
        v_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        q_batch_out: *mut Half,
        kv_data: *mut Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indices_len: i32,
        page_origin: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        seq_len: i32,
        start_pos: i32,
        cos_max_pos: i32,
        rotary_dim: i32,
        rms_eps: f32,
        page_size: i32,
        num_pages: i32,
        stride_page: i64,
        stream: CUstream,
    ) -> i32;

    pub fn qkv_norm_rope_paged_decode_hd256_plain_cuda(
        q_batch: *const Half,
        k_batch: *const Half,
        v_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        q_batch_out: *mut Half,
        kv_data: *mut Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indices_len: i32,
        page_indptr: *const i32,
        page_origins: *const i32,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        batch: i32,
        cos_max_pos: i32,
        rotary_dim: i32,
        rms_eps: f32,
        page_size: i32,
        num_pages: i32,
        stride_page: i64,
        stream: CUstream,
    ) -> i32;
}

// hd512 QK-norm + partial RoPE prep (Gemma 4 global layers):
// csrc/shared/prefill_attention_hd512.cu. Contract and validation live on
// the Rust wrappers in ops::attention; both entries return 0 on success,
// -1 with a diagnostic on failure.
unsafe extern "C" {
    // Prefill: Q → contiguous q_batch_out; K → straight into the paged KV
    // pool at k_offset_elems (feeds batch_prefill_paged, not single_prefill).
    // V is the K=V fork — the weightless norm of the same raw K, sharing
    // its denominator — written to v_offset_elems in the same pass.
    pub fn qk_norm_partial_rope_paged_prefill_hd512_cuda(
        q_batch: *const Half,
        k_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        q_batch_out: *mut Half,
        kv_data: *mut Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indices_len: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        seq_len: i32,
        start_pos: i32,
        cos_max_pos: i32,
        rotary_dim: i32,
        rms_eps: f32,
        page_size: i32,
        num_pages: i32,
        stride_page: i64,
        stream: CUstream,
    ) -> i32;

    // Batched decode straight into the pool: per-token position and page
    // window; V is the K=V fork written alongside K.
    pub fn qk_norm_partial_rope_paged_decode_hd512_cuda(
        q_batch: *const Half,
        k_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        q_batch_out: *mut Half,
        kv_data: *mut Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indices_len: i32,
        page_indptr: *const i32,
        page_origins: *const i32,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        batch: i32,
        cos_max_pos: i32,
        rotary_dim: i32,
        rms_eps: f32,
        page_size: i32,
        num_pages: i32,
        stride_page: i64,
        stream: CUstream,
    ) -> i32;

    // Batched decode: Q → contiguous q_batch_out; K normalised + partially
    // rotated in place (caller scatters it into the paged pool afterwards).
    pub fn qk_norm_partial_rope_batched_decode_hd512_cuda(
        q_batch: *const Half,
        k_batch: *mut Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        positions: *const i32,
        cos_max_pos: i32,
        q_batch_out: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        batch_size: i32,
        rotary_dim: i32,
        rms_eps: f32,
        stream: CUstream,
    ) -> i32;
}

unsafe extern "C" {
    pub fn pegainfer_kernels_last_error() -> *const std::os::raw::c_char;
}
