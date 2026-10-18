use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use super::Half;

mod flashmla_sparse;
pub use flashmla_sparse::*;
mod flashinfer_sparse;
pub use flashinfer_sparse::*;
mod indexer;
pub use indexer::*;
mod indexer_rope;
pub use indexer_rope::*;
mod topk;
pub use topk::*;
mod deepgemm_mqa;
pub use deepgemm_mqa::*;

unsafe extern "C" {
    pub fn glm52_router_select_cuda(
        logits: *const f32,
        e_score_correction_bias: *const f32,
        topk_weight: *mut f32,
        topk_idx: *mut i32,
        active_tokens: i32,
        padded_tokens: i32,
        n_experts: i32,
        topk: i32,
        route_scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_fp8_groupwise_gemm_sm100_cuda(
        activation: *const u8,
        activation_scale: *const f32,
        weight: *const u8,
        weight_scale: *const f32,
        output: *mut Half,
        workspace: *mut u8,
        workspace_bytes: usize,
        m: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    /// CuTe DSL tcgen05 AOT twin of the groupwise GEMM (glm52_fp8_dsl_gemm.c).
    /// Table-dispatched on exact (m, n, k); load once before any CUDA-graph
    /// capture. Returns cudaError_t values (0 = success).
    pub fn glm52_fp8_dsl_gemm_load_cuda() -> i32;

    pub fn glm52_fp8_dsl_gemm_supported_cuda(m: i32, n: i32, k: i32) -> i32;

    pub fn glm52_fp8_dsl_gemm_cuda(
        activation: *const u8,
        activation_scale: *const f32,
        weight: *const u8,
        weight_scale: *const f32,
        output: *mut Half,
        workspace: *mut u8,
        workspace_bytes: usize,
        m: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> i32;

    pub fn glm52_fp8_grouped_gemm_sm100_cuda(
        activation: *const u8,
        activation_scale: *const f32,
        weight: *const u8,
        weight_scale: *const f32,
        output: *mut Half,
        m_indptr: *const i32,
        max_m: i32,
        n: i32,
        k: i32,
        num_groups: i32,
        workspace: *mut u8,
        workspace_bytes: usize,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_prefill_moe_route_cuda(
        topk_idx: *const i32,
        rows: i32,
        topk: i32,
        num_experts: i32,
        expert_counts: *mut i32,
        m_indptr: *mut i32,
        gather_rows: *mut i32,
        route_slot: *mut i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_prefill_moe_gather_rows_cuda(
        input: *const Half,
        gather_rows: *const i32,
        output: *mut Half,
        total: i32,
        hidden: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_prefill_moe_gather_fp8_cuda(
        input: *const u8,
        input_scale: *const f32,
        gather_rows: *const i32,
        output: *mut u8,
        output_scale: *mut f32,
        total: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_prefill_moe_combine_cuda(
        w2_out: *const Half,
        route_slot: *const i32,
        topk_weight: *const f32,
        shared_out: *const Half,
        output: *mut Half,
        rows: i32,
        topk: i32,
        hidden: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_vocab_parallel_pack_cuda(
        local_values: *const Half,
        local_indices: *const i32,
        partial: *mut Half,
        rows: i32,
        rank: i32,
        vocab_start: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_vocab_parallel_unpack_cuda(
        gathered: *const Half,
        values: *mut Half,
        indices: *mut i32,
        rows: i32,
        ranks: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_decode_feed_launch_cuda(
        argmax_indices: *const i32,
        token_ids: *mut u32,
        positions: *mut u32,
        slot_mapping: *mut i64,
        seq_lens: *mut i32,
        rows: i32,
        stream: CUstream,
    ) -> CUresult;

    // --- SM100 (Blackwell) masked grouped fp8, packed-UE8M0 scales ----------
    // Per-expert cap is a runtime multiple of 128; groups is a dispatch over
    // the EP widths {64,32,16,8,4,2}, and both scale operands are the
    // Blackwell packed-UE8M0 i32 layout ([groups, ceil(k/512), mn] MN-major,
    // 4 exponent bytes per i32) instead of f32.

    pub fn glm52_deepgemm_sm100_grouped_fp8_metadata_cuda(
        psum_expert: *const i32,
        expert_offsets: *mut i64,
        masked_m: *mut i32,
        row_map: *mut i32,
        groups: i32,
        m_capacity: i32,
        expert_alignment: i32,
        masked_cap: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_deepgemm_sm100_masked_out_to_aligned_cuda(
        masked_out: *const Half,
        masked_m: *const i32,
        expert_offsets: *const i64,
        row_weights: *const f32,
        aligned_out: *mut Half,
        groups: i32,
        masked_cap: i32,
        aligned_rows: i32,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_deepgemm_sm100_masked_grouped_fp8_launch_cuda(
        operand_kind: i32,
        a: *const u8,
        a_scale: *const i32,
        b: *const u8,
        b_scale: *const i32,
        masked_m: *const i32,
        out: *mut Half,
        groups: i32,
        n: i32,
        k: i32,
        masked_cap: i32,
        num_sms: i32,
        stream: CUstream,
    ) -> CUresult;

    // --- MLA decode assembly (projections -> FlashMLA glue) -------------------
    pub fn glm52_mla_query_assemble_cuda(
        ql_nope: *const Half,
        q_pe_base: *const Half,
        q_pe_offset: i32,
        q_pe_head_stride: i32,
        num_q_heads: i32,
        cos: *const Half,
        sin: *const Half,
        query: *mut Half,
        tokens: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_mla_cache_pack_cuda(
        ckv_fp8: *const u8,
        ckv_scales: *const f32,
        k_pe: *const Half,
        cos: *const Half,
        sin: *const Half,
        cache: *mut u8,
        slot_mapping: *const i64,
        max_slots: i64,
        block_stride_bytes: i64,
        tokens: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_mla_front_pack_fp8_cuda(
        ql_nope: *const Half,
        q_pe_base: *const Half,
        q_pe_offset: i32,
        q_pe_head_stride: i32,
        num_q_heads: i32,
        ckv_raw: *const Half,
        norm_weight: *const Half,
        eps: f32,
        cos: *const Half,
        sin: *const Half,
        query: *mut u8,
        cache: *mut u8,
        slot_mapping: *const i64,
        max_slots: i64,
        tokens: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_mla_ckv_split_cuda(
        ckv: *const Half,
        kv_c: *mut Half,
        k_pe: *mut Half,
        tokens: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_indexer_weights_fold_cuda(
        weights: *const Half,
        q_scale: *const f32,
        softmax_scale: f32,
        n_heads_scale: f32,
        out: *mut f32,
        heads: i32,
        stream: CUstream,
    ) -> CUresult;

    // --- FP8 per-token-group quant (shared by MLA cache, MoE, dense) ----------
    pub fn glm52_fp8_per_token_group_quant_bf16_cuda(
        input: *const Half,
        output: *mut u8,
        scales: *mut f32,
        rows: i32,
        hidden_dim: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_fp8_per_token_group_quant_bf16_ue8m0_cuda(
        input: *const Half,
        output: *mut u8,
        scales: *mut f32,
        rows: i32,
        hidden_dim: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_fp8_per_token_group_quant_bf16_masked_cuda(
        input: *const Half,
        output: *mut u8,
        scales: *mut f32,
        rows: i32,
        hidden_dim: i32,
        group_size: i32,
        row_bound: *const i64,
        row_map: *const i32,
        masked_cap: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_silu_and_mul_per_token_group_quant_bf16_masked_cuda(
        input: *const Half,
        output: *mut u8,
        scales: *mut f32,
        rows: i32,
        hidden_size: i32,
        group_size: i32,
        row_bound: *const i64,
        row_map: *const i32,
        masked_cap: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_fp8_scale_pack_ue8m0_cuda(
        scales: *const f32,
        packed: *mut i32,
        groups: i32,
        scale_cols: i32,
        cap: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_fp8_weight_ue8m0_requant_cuda(
        weight: *mut u8,
        scales: *const f32,
        groups: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_fp8_weight_scale_pack_ue8m0_cuda(
        scales: *const f32,
        packed: *mut i32,
        groups: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    // --- MoE router (csrc/glm52/glm52_router.cu) ------------------------------
    pub fn glm52_router_noaux_tc_cuda(
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

    pub fn glm52_fp8_weight_only_gemv_pair_cuda(
        activation: *const Half,
        weight_a: *const u8,
        weight_scale_a: *const f32,
        out_a: *mut Half,
        n_a: i32,
        weight_b: *const u8,
        weight_scale_b: *const f32,
        out_b: *mut Half,
        n_b: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_fp8_weight_only_gemv_batched_cuda(
        activation: *const Half,
        weight: *const u8,
        weight_scale: *const f32,
        out: *mut Half,
        scratch: *mut f32,
        scratch_floats: usize,
        batch: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_silu_and_mul_bf16_cuda(
        input: *const Half,
        output: *mut Half,
        rows: i32,
        inter: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_fp8_weight_only_gemv_partials_cuda(
        activation: *const Half,
        weight: *const u8,
        weight_scale: *const f32,
        out: *mut Half,
        scratch: *mut f32,
        scratch_floats: usize,
        batch: i32,
        n: i32,
        k: i32,
        stream: CUstream,
        ksplit_out: *mut i32,
    ) -> CUresult;

    pub fn glm52_prefill_unpack_pages_cuda(
        packed: *const u8,
        block_ids: *const i32,
        blocks: i32,
        packed_bytes: i32,
        packed_block_stride: i64,
        max_slots: i64,
        unpacked: *mut Half,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_gemv_reduce_silu_mul_cuda(
        partial: *const f32,
        output: *mut Half,
        rows: i32,
        inter: i32,
        ksplit: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_gemv_split_reduce_cuda(
        partial: *const f32,
        out_a: *mut Half,
        out_b: *mut Half,
        rows: i32,
        n_a: i32,
        n_b: i32,
        ksplit: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_gemv_mma_ksplit_cuda(batch: i32, n: i32, k: i32, ksplit_out: *mut i32)
    -> CUresult;
}
