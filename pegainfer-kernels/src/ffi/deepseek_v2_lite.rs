use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use super::Half;

// DeepSeek-V2-Lite private kernels (feature `deepseek-v2-lite`).
// Sources: csrc/deepseek_v2_lite/*.cu.
unsafe extern "C" {
    pub fn dsv2_lite_route_logits_cuda(
        logits: *const f32,
        weights: *mut f32,
        ids: *mut i32,
        errors: *mut i32,
        summary: *mut u64,
        batch: i32,
        layer_idx: i32,
        stream: CUstream,
    ) -> CUresult;
    pub fn dsv2_lite_route_pointers_cuda(
        ids: *const i32,
        hidden: *const Half,
        zero: *const Half,
        w13: *const u64,
        w2: *const u64,
        gate: *mut Half,
        act: *mut Half,
        rows: *mut Half,
        a13: *mut u64,
        x13: *mut u64,
        y13: *mut u64,
        a2: *mut u64,
        x2: *mut u64,
        y2: *mut u64,
        summary: *mut u64,
        batch: i32,
        first_expert: i32,
        hidden_dim: i32,
        intermediate: i32,
        stream: CUstream,
    ) -> CUresult;
    pub fn dsv2_lite_route_reduce_cuda(
        rows: *const Half,
        ids: *const i32,
        weights: *const f32,
        errors: *const i32,
        out: *mut f32,
        batch: i32,
        first_expert: i32,
        hidden_dim: i32,
        stream: CUstream,
    ) -> CUresult;
    pub fn dsv2_lite_pointer_gemm_cuda(
        weights: *const u64,
        inputs: *const u64,
        outputs: *const u64,
        m: i32,
        k: i32,
        routes: i32,
        stream: CUstream,
    ) -> i32;

    pub fn dsv2_lite_router_logits_cuda(
        hidden: *const Half,
        gate_weight: *const Half,
        logits: *mut f32,
        seq_len: i32,
        hidden_dim: i32,
        n_experts: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv2_lite_router_softmax_topk_cuda(
        hidden: *const Half,
        gate_weight: *const Half,
        topk_weight: *mut f32,
        topk_idx: *mut i32,
        seq_len: i32,
        hidden_dim: i32,
        n_experts: i32,
        topk: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv2_lite_accumulate_fixed_expert_cuda(
        expert_output: *const Half,
        topk_weight: *const f32,
        topk_idx: *const i32,
        accum: *mut f32,
        global_expert: i32,
        seq_len: i32,
        hidden_dim: i32,
        topk: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv2_lite_kv_norm_cuda(
        kv_a: *const Half,
        norm_weight: *const Half,
        compressed: *mut Half,
        kv_lora_rank: i32,
        kv_a_rows: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv2_lite_decode_attention_cuda(
        q: *const Half,
        kv_a: *const Half,
        kv_b: *const Half,
        key_cache: *mut f32,
        value_cache: *mut f32,
        out: *mut Half,
        position: i32,
        num_heads: i32,
        qk_nope_head_dim: i32,
        qk_rope_head_dim: i32,
        v_head_dim: i32,
        kv_lora_rank: i32,
        kv_a_rows: i32,
        kv_b_rows: i32,
        max_seq_len: i32,
        rope_theta: f32,
        rope_factor: f32,
        rope_mscale: f32,
        rope_mscale_all_dim: f32,
        rope_beta_fast: f32,
        rope_beta_slow: f32,
        rope_original_max_position_embeddings: i32,
        has_rope_scaling: i32,
        stream: CUstream,
    ) -> CUresult;
}
