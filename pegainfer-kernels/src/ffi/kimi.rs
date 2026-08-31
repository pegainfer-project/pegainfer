use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use super::Half;

// Kimi K2 private kernels (feature `kimi-k2`).
// Sources: csrc/kimi_k2/*.cu (+ vendored vllm-marlin headers).
unsafe extern "C" {
    pub fn kimi_add_f32_bf16_to_bf16_cuda(
        a: *const f32,
        b: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_flashinfer_batch_decode_mla_cuda(
        q_nope: *const Half,
        q_pe: *const Half,
        output: *mut Half,
        ckv_cache: *const Half,
        kpe_cache: *const Half,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len: *const i32,
        request_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        num_qo_heads: i32,
        ckv_stride_page: i64,
        ckv_stride_n: i64,
        kpe_stride_page: i64,
        kpe_stride_n: i64,
        page_size: i32,
        batch_size: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn kimi_flashinfer_single_prefill_mla_cuda(
        q: *const Half,
        output: *mut Half,
        k_cache: *const Half,
        v_cache: *const Half,
        local_heads: i32,
        seq_len: i32,
        kv_len: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn kimi_k2_router_noaux_tc_cuda(
        hidden: *const Half,
        gate_weight: *const Half,
        e_score_correction_bias: *const f32,
        logits: *mut f32,
        topk_weight: *mut f32,
        topk_idx: *mut i32,
        active_tokens: i32,
        padded_tokens: i32,
        hidden_dim: i32,
        n_experts: i32,
        topk: i32,
        route_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_marlin_int4_fuse_w13_cuda(
        gate_weight_packed_marlin: *const u8,
        up_weight_packed_marlin: *const u8,
        w13_weight_packed_marlin: *mut u8,
        gate_scale_marlin: *const Half,
        up_scale_marlin: *const Half,
        w13_scale_marlin: *mut Half,
        in_dim: i32,
        intermediate_dim: i32,
        local_experts: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_marlin_int4_reorder_scale_cuda(
        weight_scale_checkpoint: *const Half,
        weight_scale_marlin: *mut Half,
        in_dim: i32,
        out_dim: i32,
        local_experts: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_marlin_int4_reorder_weight_cuda(
        weight_packed_checkpoint_offset_binary: *const u8,
        weight_packed_marlin: *mut u8,
        in_dim: i32,
        out_dim: i32,
        local_experts: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_marlin_sum_topk_rows_f32_cuda(
        route_output: *const Half,
        out: *mut f32,
        active_tokens: i32,
        topk: i32,
        hidden_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_marlin_w13_swiglu_cuda(
        w13: *const Half,
        out: *mut Half,
        rows: i32,
        intermediate_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_marlin_w13_swiglu_expanded_cuda(
        w13: *const Half,
        out: *mut Half,
        num_tokens_post_padded: *const i32,
        max_rows: i32,
        intermediate_dim: i32,
        sm_count: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_marlin_wna16_gemm_cuda(
        input: *const Half,
        output: *mut Half,
        c_tmp: *mut f32,
        b_qweight: *const u8,
        b_scales: *const Half,
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
        local_experts: i32,
        group_size: i32,
        sm_count: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_mla_absorb_q_nope_cuda(
        kv_b_proj: *const Half,
        q_nope: *const Half,
        q_abs_nope: *mut Half,
        batch_size: i32,
        local_heads: i32,
        stream: CUstream,
    ) -> i32;

    pub fn kimi_mla_paged_kv_append_cuda(
        ckv_cache: *mut Half,
        kpe_cache: *mut Half,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len: *const i32,
        append_ckv: *const Half,
        append_kpe: *const Half,
        batch_indices: *const i32,
        positions: *const i32,
        nnz: i32,
        ckv_stride_page: i64,
        ckv_stride_n: i64,
        kpe_stride_page: i64,
        kpe_stride_n: i64,
        page_size: i32,
        batch_size: i32,
        stream: CUstream,
    ) -> i32;

    pub fn kimi_mla_rope_apply_kpe_cuda(
        k_rope: *const Half,
        cos: *const Half,
        sin: *const Half,
        positions: *const i32,
        append_kpe: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_mla_rope_assemble_prefill_cuda(
        q_proj: *const Half,
        k_rope: *const Half,
        kv_b: *const Half,
        cos: *const Half,
        sin: *const Half,
        q_attn: *mut Half,
        k_cache: *mut Half,
        v_cache: *mut Half,
        seq_len: i32,
        start_pos: i32,
        kv_len: i32,
        local_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_mla_gather_cached_ckv_cuda(
        ckv_cache: *const Half,
        page_indices: *const i32,
        ckv_out: *mut Half,
        cached_len: i32,
        page_size: i32,
        ckv_stride_page: i64,
        ckv_stride_n: i64,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_mla_assemble_cached_kv_cuda(
        kv_b: *const Half,
        kpe_cache: *const Half,
        page_indices: *const i32,
        k_cache: *mut Half,
        v_cache: *mut Half,
        cached_len: i32,
        kv_len: i32,
        local_heads: i32,
        page_size: i32,
        kpe_stride_page: i64,
        kpe_stride_n: i64,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_mla_rope_split_decode_cuda(
        q_proj: *const Half,
        k_rope: *const Half,
        cos: *const Half,
        sin: *const Half,
        positions: *const i32,
        q_nope: *mut Half,
        q_pe: *mut Half,
        append_kpe: *mut Half,
        batch_size: i32,
        local_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_mla_split_qkv_a_cuda(
        qkv_a: *const Half,
        q_a: *mut Half,
        compressed: *mut Half,
        k_rope: *mut Half,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_mla_split_qkv_a_norm_cuda(
        qkv_a: *const Half,
        q_a_weight: *const Half,
        ckv_weight: *const Half,
        q_a_normed: *mut Half,
        ckv_normed: *mut Half,
        k_rope: *mut Half,
        eps: f32,
        batch_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_mla_v_up_cuda(
        kv_b_proj: *const Half,
        latent: *const Half,
        output: *mut Half,
        batch_size: i32,
        local_heads: i32,
        stream: CUstream,
    ) -> i32;

    pub fn kimi_moe_marlin_align_block_size_cuda(
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

    pub fn kimi_deepep_build_marlin_routing_on_stream(
        psum_expert: *const i32,
        sorted_token_ids: *mut i32,
        expert_ids: *mut i32,
        num_tokens_post_padded: *mut i32,
        num_local_experts: i32,
        alignment: i32,
        block_size: i32,
        max_padded_tokens: i32,
        max_m_blocks: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_residual_add_scaled_f32_cuda(
        hidden: *const Half,
        projected: *const Half,
        routed_f32: *const f32,
        scale: f32,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn kimi_residual_add_scaled_bf16_cuda(
        hidden: *const Half,
        projected: *const Half,
        routed: *const Half,
        scale: f32,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

}

// Added during rebase: cuBLASLt-backed MLA / o_proj / shared MLP entry points.
unsafe extern "C" {
    pub fn kimi_mla_cublaslt_init_cuda() -> i32;

    pub fn kimi_mla_cublaslt_destroy_cuda();

    pub fn kimi_o_proj_cublaslt_init_cuda() -> i32;

    pub fn kimi_o_proj_cublaslt_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    ) -> i32;

    pub fn kimi_o_proj_cublaslt_destroy_cuda();

    pub fn kimi_shared_gate_up_cublaslt_init_cuda() -> i32;

    pub fn kimi_shared_gate_up_cublaslt_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    ) -> i32;

    pub fn kimi_shared_gate_up_cublaslt_destroy_cuda();

}
