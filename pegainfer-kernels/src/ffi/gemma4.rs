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
}
