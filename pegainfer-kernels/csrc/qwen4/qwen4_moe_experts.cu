// Qwen3.8-Flash-Next (Qwen4 line) MoE block: the routed experts.
//
// Frozen semantics: transformers 5.16.1 `qwen4_exp`, `Qwen4ExpTextExperts`
// (`modeling_qwen4_exp.py:859-895`) at the frozen checkpoint's geometry — the
// two fused expert stacks `mlp.experts.gate_up_proj` `[512, 2*640, 2560]` and
// `mlp.experts.down_proj` `[512, 2560, 640]`, expert outermost, no `.weight`
// suffix. The reference's forward for one token, per expert hit and in
// *ascending expert index*:
//
//   gate, up = F.linear(x, gate_up_proj[e]).chunk(2)      bf16 out
//   h        = act(gate) * up                             act = silu
//   y        = F.linear(h, down_proj[e])                  bf16 out
//   y       *= routing_weight[token, slot]                bf16 round
//   final   .index_add_(0, token, y)                      bf16 round per add
//
// Three properties of that sequence are contract, not implementation detail:
//
//  * The per-token accumulation order is **ascending expert index**, because
//    the reference's loop over `range(num_experts)` is outer and each
//    `index_add_` is one call. `torch.where` inside an expert visits
//    `(slot, token)` row-major, but a token appears at most once per expert
//    (top-k picks distinct experts), so that order reaches no token's sum.
//  * `final_hidden_states` is a bf16 zeros tensor and every add rounds to bf16,
//    so the per-token fold is a chain of bf16-rounded additions, not one f32
//    sum rounded at the end.
//  * Both projections round to bf16 where they store, and the routing scale and
//    the activation's bf16 cast each round again.
//
// Backend (single rank, correctness first): per-route cuBLAS. One `(token,
// slot)` route is one N=1 GEMM against its own expert's rows, driven through
// device pointer arrays exactly as `csrc/deepseek_v2_lite/dsv2_lite_moe.cu`
// does, so no grouping pass, no host-visible expert counts and no fixed
// per-expert capacity exist to be wrong: an empty expert issues no call and a
// skewed one issues one call per row. What it costs is weight traffic — the
// same expert's rows are read once per route, so a 512-token prefill reads
// ~54 GB where a grouped GEMM would read ~5.4 GB — which is recorded in
// `docs/models/qwen4/bring-up.md` rather than optimized here: 1107a is the
// correctness baseline that a grouped backend is measured against.
//
// Target: the sm_89-common portable subset — no TMA, no cluster, no wgmma. The
// two reductions here (the fold's insertion order and the scan's) are plain
// SIMT with a fixed instruction sequence, so every output is deterministic run
// to run.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

// The graph-safe thread-local cuBLAS handle every in-tree GEMM uses; this file
// takes it for the same reason `dsv2_lite_moe.cu` does — a batched pointer
// GEMM has no shared entry point.
extern thread_local cublasHandle_t g_cublas_handle;

namespace {

// The routing kernel's compiled top-k, which is also this file's fold bound:
// the per-token order is an insertion sort over at most this many slots.
constexpr int kTopK = 10;
constexpr int kFoldThreads = 256;
constexpr int kPointerThreads = 128;
// A batch count cuBLAS accepts and an i32 cannot overflow; a caller above it is
// refused rather than silently truncated. 512 tokens x top-10 is 5120 routes,
// so this is four times the largest single-step prefill the line serves.
constexpr int kMaxRoutes = 16384;

// One thread per route: resolve the route's expert and token into the three
// pointers each batched GEMM needs. Ids are checked here rather than in the
// GEMM, because a pointer array is formed before any kernel could see them: an
// out-of-range id falls back to expert 0 so the read stays inside the stack,
// and the fold writes NaN for that token, so the substituted expert's numbers
// never reach the output.
__global__ void qwen4_moe_route_pointers_kernel(
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ gate_up_weights,
    const __nv_bfloat16* __restrict__ down_weights,
    __nv_bfloat16* __restrict__ gate_up, __nv_bfloat16* __restrict__ activation,
    __nv_bfloat16* __restrict__ rows, const __nv_bfloat16** __restrict__ a_gate_up,
    const __nv_bfloat16** __restrict__ x_gate_up,
    __nv_bfloat16** __restrict__ y_gate_up, const __nv_bfloat16** __restrict__ a_down,
    const __nv_bfloat16** __restrict__ x_down, __nv_bfloat16** __restrict__ y_down,
    int routes, int top_k, int experts, int hidden_dim, int intermediate) {
  const int route = blockIdx.x * blockDim.x + threadIdx.x;
  if (route >= routes) return;
  const int id = ids[route];
  const int expert = (id >= 0 && id < experts) ? id : 0;
  const int token = route / top_k;
  const long long gate_up_span =
      static_cast<long long>(2) * intermediate * hidden_dim;
  const long long down_span = static_cast<long long>(hidden_dim) * intermediate;

  a_gate_up[route] = gate_up_weights + static_cast<long long>(expert) * gate_up_span;
  x_gate_up[route] = hidden + static_cast<long long>(token) * hidden_dim;
  y_gate_up[route] = gate_up + static_cast<long long>(route) * 2 * intermediate;

  a_down[route] = down_weights + static_cast<long long>(expert) * down_span;
  x_down[route] = activation + static_cast<long long>(route) * intermediate;
  y_down[route] = rows + static_cast<long long>(route) * hidden_dim;
}

// The reference's per-token fold: scale each route's down-projection row by its
// bf16 routing weight, then add the token's contributions in ascending expert
// index, rounding to bf16 after every add. Both roundings are the reference's
// own — `y *= routing_weight` rounds its bf16 store, and `index_add_` rounds the
// bf16 accumulator once per call.
//
// Every thread of the block walks the row's `top_k` ids and derives the same
// ascending order, so no shared-memory hand-off is needed and the order is the
// same in every thread by construction. The sort is a stable insertion sort on
// the expert id, so equal ids (which the router cannot produce) would keep slot
// order instead of becoming a nondeterministic choice.
__global__ void qwen4_moe_route_scale_fold_kernel(
    const __nv_bfloat16* __restrict__ rows, const __nv_bfloat16* __restrict__ weights,
    const int* __restrict__ ids, __nv_bfloat16* __restrict__ out, int top_k,
    int experts, int hidden) {
  const int token = blockIdx.x;
  int order[kTopK];
  bool poisoned = false;
  for (int slot = 0; slot < top_k; ++slot) {
    const int id = ids[static_cast<size_t>(token) * top_k + slot];
    poisoned |= (id < 0 || id >= experts);
    int at = slot;
    while (at > 0 &&
           ids[static_cast<size_t>(token) * top_k + order[at - 1]] > id) {
      order[at] = order[at - 1];
      --at;
    }
    order[at] = slot;
  }
  if (poisoned) {
    // Fail the token closed: a NaN row is unusable by inspection and cannot be
    // mistaken for a plausible expert sum. Same policy as the router's own
    // fail-closed path, which hands out NaN weights and slot-index ids.
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
      out[static_cast<size_t>(token) * hidden + col] = __float2bfloat16(NAN);
    }
    return;
  }
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    float acc = 0.0f;
    for (int pick = 0; pick < top_k; ++pick) {
      const size_t route = static_cast<size_t>(token) * top_k + order[pick];
      const float value =
          __bfloat162float(rows[route * hidden + col]);
      const float weight = __bfloat162float(weights[route]);
      // `y *= routing_weight`: one bf16 store of the scaled row.
      const float scaled = __bfloat162float(__float2bfloat16(value * weight));
      // `index_add_`: the bf16 accumulator rounds once per contribution.
      acc = __bfloat162float(__float2bfloat16(acc + scaled));
    }
    out[static_cast<size_t>(token) * hidden + col] = __float2bfloat16(acc);
  }
}

}  // namespace

extern "C" {

// Resolve one batch of `(token, slot)` routes into the pointer arrays the two
// batched GEMMs consume. `ids` is the routing kernel's `[routes]` i32 output,
// `hidden` is `[tokens, hidden_dim]`, the two stacks are the checkpoint's fused
// `[experts, ...]` tensors, and `gate_up` / `activation` / `rows` are this
// route batch's `[routes, 2*intermediate]` / `[routes, intermediate]` /
// `[routes, hidden_dim]` scratch.
//
// Every id is range-checked here: an out-of-range one is replaced by expert 0
// (so the formed pointer stays inside the stack) and the fold writes NaN for
// that token. Nothing is read from or written to the device by the host.
CUresult qwen4_moe_route_pointers_cuda(
    const int* ids, const __nv_bfloat16* hidden, const __nv_bfloat16* gate_up_weights,
    const __nv_bfloat16* down_weights, __nv_bfloat16* gate_up,
    __nv_bfloat16* activation, __nv_bfloat16* rows,
    const __nv_bfloat16** a_gate_up, const __nv_bfloat16** x_gate_up,
    __nv_bfloat16** y_gate_up, const __nv_bfloat16** a_down,
    const __nv_bfloat16** x_down, __nv_bfloat16** y_down, int routes, int top_k,
    int experts, int hidden_dim, int intermediate, cudaStream_t stream) {
  if (ids == nullptr || hidden == nullptr || gate_up_weights == nullptr ||
      down_weights == nullptr || gate_up == nullptr || activation == nullptr ||
      rows == nullptr || a_gate_up == nullptr || x_gate_up == nullptr ||
      y_gate_up == nullptr || a_down == nullptr || x_down == nullptr ||
      y_down == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (routes < 0 || routes > kMaxRoutes || top_k <= 0 || top_k > kTopK ||
      experts <= 0 || hidden_dim <= 0 || intermediate <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (routes == 0) return CUDA_SUCCESS;
  const int grid = (routes + kPointerThreads - 1) / kPointerThreads;
  qwen4_moe_route_pointers_kernel<<<grid, kPointerThreads, 0, stream>>>(
      ids, hidden, gate_up_weights, down_weights, gate_up, activation, rows,
      a_gate_up, x_gate_up, y_gate_up, a_down, x_down, y_down, routes, top_k,
      experts, hidden_dim, intermediate);
  return map_cuda_error(cudaGetLastError());
}

// One batched bf16 GEMM over device-resident pointers: route `r` computes
// `y[r] = w[r] @ x[r]` with `w` row-major `[m, k]` (consumed as OP_T, so cuBLAS
// sees `k x m`), `x` `[k]` and `y` `[m]`. This is DeepSeek-V2-Lite's
// `dsv2_lite_pointer_gemm_cuda` shape — each route is an independent N=1
// problem, not one multi-row expert GEMM — and it is why the folded list of
// experiments that produced this backend could skip a grouping pass entirely.
//
// f32 accumulate into a bf16 store, exactly as `gemm_cuda` does, so the
// projection rounds where the reference's `F.linear` rounds.
int qwen4_moe_pointer_gemm_cuda(const void* const* weights, const void* const* inputs,
                                void* const* outputs, int m, int k, int routes,
                                cudaStream_t stream) {
  if (g_cublas_handle == nullptr) {
    return static_cast<int>(cudaErrorInvalidResourceHandle);
  }
  if (weights == nullptr || inputs == nullptr || outputs == nullptr || m <= 0 ||
      k <= 0 || routes <= 0 || routes > kMaxRoutes) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cublasStatus_t status = cublasSetStream(g_cublas_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return PEGAINFER_CUBLAS_STATUS_BASE + static_cast<int>(status);
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  status = cublasGemmBatchedEx(
      g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, 1, k, &alpha, weights,
      CUDA_R_16BF, k, inputs, CUDA_R_16BF, k, &beta, outputs, CUDA_R_16BF, m,
      routes, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return PEGAINFER_CUBLAS_STATUS_BASE + static_cast<int>(status);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

// Fold one route batch's scaled expert rows into per-token outputs.
//
// `rows` is `[tokens * top_k, hidden]` bf16 indexed by route (`token * top_k +
// slot`), `weights` the routing kernel's bf16 `[tokens * top_k]`, `ids` its i32
// expert ids, and `out` the `[tokens, hidden]` routed output. The order is the
// reference's: ascending expert index, bf16 rounding after every add. A token
// whose routing ids are out of range gets a NaN row.
CUresult qwen4_moe_route_scale_fold_cuda(const __nv_bfloat16* rows,
                                       const __nv_bfloat16* weights, const int* ids,
                                       __nv_bfloat16* out, int tokens, int top_k,
                                       int experts, int hidden,
                                       cudaStream_t stream) {
  if (rows == nullptr || weights == nullptr || ids == nullptr || out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (tokens < 0 || top_k <= 0 || top_k > kTopK || experts <= 0 || hidden <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (tokens == 0) return CUDA_SUCCESS;
  if (static_cast<long long>(tokens) * top_k > kMaxRoutes) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  qwen4_moe_route_scale_fold_kernel<<<tokens, kFoldThreads, 0, stream>>>(
      rows, weights, ids, out, top_k, experts, hidden);
  return map_cuda_error(cudaGetLastError());
}

}  // extern "C"
