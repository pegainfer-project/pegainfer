use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use super::Half;

unsafe extern "C" {
    /// Softmax over every expert, then the top `top_k` renormalized among
    /// themselves and scaled per expert. `logits` is `[rows, experts]`;
    /// `index_out` and `weight_out` are `[rows, top_k]`.
    pub fn gemma4_moe_router_topk_cuda(
        logits: *const Half,
        per_expert_scale: *const Half,
        rows: i32,
        experts: i32,
        top_k: i32,
        index_out: *mut i32,
        weight_out: *mut f32,
        stream: CUstream,
    ) -> CUresult;

    /// `out[token] = sum over picks of routed[token * top_k + pick]`.
    pub fn gemma4_moe_sum_topk_cuda(
        routed: *const Half,
        rows: i32,
        top_k: i32,
        hidden: i32,
        out: *mut Half,
        stream: CUstream,
    ) -> CUresult;

    /// One expert-blocked NVFP4 GEMM. `b_scales` and `global_scale` are what
    /// the preparation below produced, not what the checkpoint holds.
    pub fn gemma4_marlin_nvfp4_moe_cuda(
        input: *const Half,
        output: *mut Half,
        c_tmp: *mut f32,
        b_qweight: *const u8,
        b_scales: *const u8,
        global_scale: *const f32,
        workspace: *mut i32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        num_tokens_post_padded: *const i32,
        topk_weights: *const f32,
        workspace_len: i32,
        sorted_token_ids_len: i32,
        moe_block_size: i32,
        top_k: i32,
        mul_topk_weights: bool,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        sm_count: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Rewrites the checkpoint's e4m3 block scales into Marlin's order and
    /// S0E5M3 encoding. `rescale` is the shared power of two the caller
    /// divides back out of the per-tensor scale.
    pub fn gemma4_marlin_nvfp4_prepare_scales_cuda(
        checkpoint: *const u8,
        prepared: *mut u8,
        experts: i32,
        in_dim: i32,
        out_dim: i32,
        rescale: f32,
        stream: CUstream,
    ) -> CUresult;

    /// Marlin's B layout for any four-bit weight, over `[experts, out_dim,
    /// in_dim / 2]` bytes.
    pub fn marlin_repack_4bit_cuda(
        src: *const u8,
        dst: *mut u8,
        experts: i32,
        in_dim: i32,
        out_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Build the expert-blocked dispatch on the device. `expert_offsets` holds
    /// `experts + 1` scratch counters; the cursor slot after it is an ignored
    /// compatibility parameter and may be null.
    pub fn marlin_moe_align_block_size_cuda(
        topk_idx: *const i32,
        sorted_token_ids: *mut i32,
        expert_ids: *mut i32,
        num_tokens_post_padded: *mut i32,
        expert_offsets: *mut u32,
        unused_expert_cursor: *mut u32,
        active_tokens: i32,
        topk: i32,
        global_start: i32,
        local_experts: i32,
        block_size: i32,
        max_padded_tokens: i32,
        max_m_blocks: i32,
        stream: CUstream,
    ) -> CUresult;

    /// fp8-KV twin of `batch_prefill_paged_window_cuda_hd256`: same launch
    /// body over an e4m3 pool (scale 1.0), Q and output bf16.
    pub fn gemma4_batch_prefill_paged_window_hd256_fp8kv_cuda(
        q: *const Half,
        output: *mut Half,
        kv_data: *const core::ffi::c_void,
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
        window_left: i32,
        stream: CUstream,
    ) -> i32;

    /// A compressed-tensors W4A16 linear (`packed` [n, k / 8] int32 words,
    /// `scales` [n, k / 32] bf16) rewritten into the TileLang GEMMs' fragment
    /// layout: `wq` [n / 16, k / 64, 32, 4] and `sq` [n / 16, k / 32, 8].
    /// `split` > 0 interleaves two stacked halves of that many rows 32 at a
    /// time (gate|up); zero keeps stored order.
    pub fn gemma4_w4a16_pack_cuda(
        packed: *const u32,
        scales: *const u16,
        wq: *mut u32,
        sq: *mut u32,
        n: i32,
        k: i32,
        split: i32,
        stream: CUstream,
    ) -> CUresult;

    /// The fragment layout back to a bf16 `[n, k]` matrix in stored row order.
    pub fn gemma4_w4a16_dequant_cuda(
        wq: *const u32,
        sq: *const u32,
        out: *mut Half,
        n: i32,
        k: i32,
        split: i32,
        stream: CUstream,
    ) -> CUresult;
}

// hd512 attention (global layers): csrc/gemma4/paged_attention_hd512.cu
unsafe extern "C" {
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

// Windowed hd256 prefill (local layers): csrc/gemma4/paged_attention_window_hd256.cu.
// `window_left` is an inclusive distance: an N-token window passes N - 1, and
// -1 degrades to full attention.
unsafe extern "C" {
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

// hd256 plain-w QK-norm + RoPE prep (local layers):
// csrc/gemma4/prefill_attention_hd256_plain.cu. Contract and validation live
// on the Rust wrappers in ops::gemma4_attention; each entry returns 0 on
// success, -1 with a diagnostic on failure.
unsafe extern "C" {
    // Q → contiguous q_batch_out; K (normed + rotated) and V (weightless-normed,
    // never rotated) → straight into the paged KV pool at k_offset_elems /
    // v_offset_elems.
    pub fn qkv_norm_rope_paged_prefill_hd256_plain_cuda(
        q_batch: *const Half,
        k_batch: *const Half,
        v_batch: *const Half,
        q_stride: i32,
        k_stride: i32,
        v_stride: i32,
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
        q_stride: i32,
        k_stride: i32,
        v_stride: i32,
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

    /// E4m3 KV twin.
    pub fn qkv_norm_rope_paged_prefill_hd256_plain_fp8kv_cuda(
        q_batch: *const Half,
        k_batch: *const Half,
        v_batch: *const Half,
        q_stride: i32,
        k_stride: i32,
        v_stride: i32,
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

    /// E4m3 KV twin.
    pub fn qkv_norm_rope_paged_decode_hd256_plain_fp8kv_cuda(
        q_batch: *const Half,
        k_batch: *const Half,
        v_batch: *const Half,
        q_stride: i32,
        k_stride: i32,
        v_stride: i32,
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

// hd512 QK-norm + partial RoPE prep (global layers):
// csrc/gemma4/prefill_attention_hd512.cu. Contract and validation live on
// the Rust wrappers in ops::gemma4_attention; both entries return 0 on
// success, -1 with a diagnostic on failure.
unsafe extern "C" {
    // Prefill: Q → contiguous q_batch_out; K → straight into the paged KV
    // pool at k_offset_elems (feeds batch_prefill_paged).
    // V is the K=V fork — the weightless norm of the same raw K, sharing
    // its denominator — written to v_offset_elems in the same pass. The
    // pool row is described as bands: `row_width` columns per (token, kv
    // head) and `fold_rotary`, zero for the split K|V format and the
    // rotated columns K keeps in the folded one, where both offsets name
    // the layer's single block.
    pub fn qk_norm_partial_rope_paged_prefill_hd512_cuda(
        q_batch: *const Half,
        k_batch: *const Half,
        q_stride: i32,
        k_stride: i32,
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
        row_width: i32,
        fold_rotary: i32,
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
        q_stride: i32,
        k_stride: i32,
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
        row_width: i32,
        fold_rotary: i32,
        rms_eps: f32,
        page_size: i32,
        num_pages: i32,
        stride_page: i64,
        stream: CUstream,
    ) -> i32;
}
