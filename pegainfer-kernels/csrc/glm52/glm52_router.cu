#include "../common.cuh"
#include "../shared/ffi_guard.cuh"
#include "glm52_min_gemv.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

constexpr int kGlm52Experts = 256;
// One thread per expert; threads past n_experts only idled (512 was inherited).
constexpr int kRouterSelectThreads = kGlm52Experts;
constexpr int kGlm52Topk = 8;

__device__ __forceinline__ bool better_router_choice(float value, int expert,
                                                     float best_value,
                                                     int best_expert) {
  return value > best_value || (value == best_value && expert < best_expert);
}

__global__ void router_scores_topk_normalize_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ e_score_correction_bias,
    float* __restrict__ topk_weight, int* __restrict__ topk_idx,
    int active_tokens, int padded_tokens, int n_experts, int topk,
    float route_scale) {
  int token = blockIdx.x;
  int tid = threadIdx.x;
  if (token >= padded_tokens) return;

  extern __shared__ char shared[];
  float* scores = reinterpret_cast<float*>(shared);
  float* choice_scores = scores + blockDim.x;
  float* selected_scores = choice_scores + blockDim.x;

  const int expert = tid;
  if (expert < n_experts) {
    float score = 1.0f / (1.0f + expf(-logits[token * n_experts + expert]));
    scores[tid] = score;
    choice_scores[tid] = score + e_score_correction_bias[expert];
  } else {
    scores[tid] = 0.0f;
    choice_scores[tid] = -CUDART_INF_F;
  }
  if (token >= active_tokens) return;
  __syncthreads();

  // Single-pass rank-count selection. `better_router_choice` is a strict total order over
  // (value desc, index asc) — `choice_scores` is always finite here (sigmoid in (0,1) plus a
  // finite bias), so no NaN can break the ordering — and expert indices are unique, so each
  // expert's rank — the count of strictly-better experts — is a bijection with the
  // sequential-argmax pick order. That makes
  // the selected idx/weight bit-identical to the old 8 masked full-block reductions (the
  // descending route order, hence the f32 `selected_sum` order, is preserved) while collapsing
  // ~8*log(blockDim) __syncthreads to two. For bs=1 this is one block over 256 experts — an
  // irreducibly tiny grid (cf. SGLang's warp-per-token moe_fused_gate, the bs>1 form); the only
  // lever at bs=1 is removing the serial structure, which this does (ncu 12.2 -> 8.2 us/call).
  if (expert < n_experts) {
    const float cv = choice_scores[tid];
    int rank = 0;
    for (int j = 0; j < n_experts; ++j) {
      if (better_router_choice(choice_scores[j], j, cv, expert)) ++rank;
    }
    if (rank < topk) {
      topk_idx[token * topk + rank] = expert;
      selected_scores[rank] = scores[expert];
    }
  }
  __syncthreads();

  if (tid == 0) {
    float selected_sum = 0.0f;
    for (int route = 0; route < topk; ++route) selected_sum += selected_scores[route];
    const float scale = selected_sum > 0.0f ? route_scale / selected_sum : 0.0f;
    for (int route = 0; route < topk; ++route) {
      topk_weight[token * topk + route] = selected_scores[route] * scale;
    }
  }
}

CUresult consume_last_cuda_error() {
  cudaError_t err = cudaGetLastError();
  return map_cuda_error(err);
}

CUresult glm52_router_logits_gemm(const __nv_bfloat16* hidden,
                                  const __nv_bfloat16* gate_weight,
                                  float* logits, int padded_tokens,
                                  int hidden_dim, int n_experts,
                                  cudaStream_t stream) {
  if (hidden_dim != glm52_min_gemv::kHidden || n_experts != kGlm52Experts) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return map_cuda_error(glm52_min_gemv::launch_tokens<kGlm52Experts, float>(
      logits, hidden, gate_weight, padded_tokens, stream));
}

}  // namespace

extern "C" {

CUresult glm52_router_noaux_tc_cuda(
    const __nv_bfloat16* hidden, const __nv_bfloat16* gate_weight,
    const float* e_score_correction_bias, float* logits, float* topk_weight,
    int* topk_idx, int active_tokens, int padded_tokens, int hidden_dim,
    int n_experts, int topk, float route_scale, cudaStream_t stream) {
  if (hidden == nullptr || gate_weight == nullptr ||
      e_score_correction_bias == nullptr || logits == nullptr ||
      topk_weight == nullptr || topk_idx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (active_tokens <= 0 || padded_tokens <= 0 ||
      active_tokens > padded_tokens || hidden_dim <= 0 || n_experts <= 0 ||
      topk <= 0 || topk > n_experts || n_experts != kGlm52Experts ||
      topk != kGlm52Topk || !(route_scale > 0.0f)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUresult result = glm52_router_logits_gemm(
      hidden, gate_weight, logits, padded_tokens, hidden_dim, n_experts, stream);
  if (result != CUDA_SUCCESS) return result;

  size_t select_smem =
      static_cast<size_t>(kRouterSelectThreads) * (2 * sizeof(float)) +
      static_cast<size_t>(topk) * sizeof(float);
  router_scores_topk_normalize_kernel<<<padded_tokens, kRouterSelectThreads,
                                        select_smem, stream>>>(
      logits, e_score_correction_bias, topk_weight, topk_idx, active_tokens,
      padded_tokens, n_experts, topk, route_scale);
  result = consume_last_cuda_error();
  if (result != CUDA_SUCCESS) return result;

  return CUDA_SUCCESS;
}

CUresult glm52_router_select_cuda(
    const float* logits, const float* e_score_correction_bias,
    float* topk_weight, int* topk_idx, int active_tokens, int padded_tokens,
    int n_experts, int topk, float route_scale, cudaStream_t stream) {
  if (!logits || !e_score_correction_bias || !topk_weight || !topk_idx ||
      active_tokens <= 0 || active_tokens > padded_tokens ||
      n_experts != kGlm52Experts || topk != kGlm52Topk ||
      !(route_scale > 0.0f)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const size_t smem =
      static_cast<size_t>(kRouterSelectThreads) * (2 * sizeof(float)) +
      static_cast<size_t>(topk) * sizeof(float);
  router_scores_topk_normalize_kernel<<<padded_tokens, kRouterSelectThreads,
                                        smem, stream>>>(
      logits, e_score_correction_bias, topk_weight, topk_idx, active_tokens,
      padded_tokens, n_experts, topk, route_scale);
  return consume_last_cuda_error();
}

}  // extern "C"
