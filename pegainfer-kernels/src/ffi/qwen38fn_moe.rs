// Qwen3.8-Flash-Next (Qwen4 line) MoE block kernels. Geometry is frozen in
// `csrc/qwen38fn/qwen38fn_moe_route.cu`: the route entry point refuses any
// expert count or top-k it was not compiled for.

use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use super::Half;

unsafe extern "C" {
    /// fp32 softmax over the router logits of one token row, top-10 in the
    /// canonical order (descending probability, ascending index inside a tie),
    /// optional `norm_topk_prob` renormalization, bf16 weights and i32 expert
    /// ids. `logits` is `[rows, experts]` bf16; `weights` / `ids` are
    /// `[rows, top_k]`. `renorm` and `probs_bf16` are 0/1 flags; `probs_bf16`
    /// is the parity gate's negative control and must be 0 in production.
    pub fn qwen38fn_moe_route_cuda(
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
    pub fn qwen38fn_moe_shared_combine_cuda(
        gate_logit: *const Half,
        expert_out: *const Half,
        shared_out: *const Half,
        out: *mut Half,
        tokens: i32,
        hidden: i32,
        stream: CUstream,
    ) -> CUresult;
}
