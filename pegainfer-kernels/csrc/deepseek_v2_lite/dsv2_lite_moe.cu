#include "../common.cuh"

#include <cuda.h>
#include <cublas_v2.h>
#include <math_constants.h>

extern thread_local cublasHandle_t g_cublas_handle;

namespace {

constexpr int kMaxExperts = 64;
constexpr int kMaxTopk = 8;
constexpr int kRoutesPerToken = 6;
constexpr int kLocalExperts = kMaxExperts / 2;
constexpr int kMaxRows = 8;
constexpr int kRouterThreads = 128;
constexpr int kAccumThreads = 256;
constexpr int kCublasErrorOffset = 100000;

__device__ __forceinline__ bool better_prob_choice(float value, int expert, float best_value,
                                                   int best_expert) {
  return value > best_value || (value == best_value && expert < best_expert);
}

__device__ __forceinline__ void select_and_sort_routes(
    const float *probs, int n_experts, int topk, int *ids, float *weights) {
  bool selected[kMaxExperts] = {};
  for (int route = 0; route < topk; ++route) {
    int best_expert = n_experts;
    float best_value = -CUDART_INF_F;
    for (int expert = 0; expert < n_experts; ++expert) {
      if (!selected[expert] &&
          better_prob_choice(probs[expert], expert, best_value, best_expert)) {
        best_value = probs[expert];
        best_expert = expert;
      }
    }
    selected[best_expert] = true;
    ids[route] = best_expert;
    weights[route] = best_expert < n_experts ? probs[best_expert] : 0.0f;
  }
  for (int i = 1; i < topk; ++i) {
    int id = ids[i];
    float weight = weights[i];
    int j = i - 1;
    while (j >= 0 && ids[j] > id) {
      ids[j + 1] = ids[j];
      weights[j + 1] = weights[j];
      --j;
    }
    ids[j + 1] = id;
    weights[j + 1] = weight;
  }
}

// One thread per row deliberately retains the host softmax summation order.
__global__ void route_from_logits_kernel(const float *logits, float *weights,
                                        int *ids, int *errors,
                                        unsigned long long *summary,
                                        int layer_idx) {
  const int token = blockIdx.x;
  float probs[kMaxExperts];
  float maximum = -CUDART_INF_F;
  bool valid = true;
  for (int e = 0; e < kMaxExperts; ++e) {
    float x = logits[token * kMaxExperts + e];
    valid = valid && isfinite(x);
    maximum = fmaxf(maximum, x);
  }
  errors[token] = valid ? 0 : 1;
  if (!valid) {
    atomicAdd(summary + 2, 1ULL);
    for (int r = 0; r < kRoutesPerToken; ++r) {
      ids[token * kRoutesPerToken + r] = -1;
      weights[token * kRoutesPerToken + r] = 0.0f;
    }
    return;
  }
  float sum = 0.0f;
  for (int e = 0; e < kMaxExperts; ++e) {
    probs[e] = expf(logits[token * kMaxExperts + e] - maximum);
    sum = __fadd_rn(sum, probs[e]);
  }
  for (int e = 0; e < kMaxExperts; ++e) probs[e] = __fdiv_rn(probs[e], sum);
  int chosen[kMaxTopk];
  float chosen_weights[kMaxTopk];
  select_and_sort_routes(probs, kMaxExperts, kRoutesPerToken, chosen, chosen_weights);
  for (int r = 0; r < kRoutesPerToken; ++r) {
    ids[token * kRoutesPerToken + r] = chosen[r];
    weights[token * kRoutesPerToken + r] = chosen_weights[r];
  }
  unsigned long long route_hash = 1469598103934665603ULL;
  for (int r = 0; r < kRoutesPerToken; ++r) {
    route_hash ^= static_cast<unsigned long long>(chosen[r] + 1 + 67 * r + 4099 * token);
    route_hash *= 1099511628211ULL;
  }
  atomicAdd(summary + 3 + layer_idx, route_hash);
}

__global__ void route_pointers_kernel(
    const int *ids, const __nv_bfloat16 *hidden, const __nv_bfloat16 *zero,
    const __nv_bfloat16 *const *w13, const __nv_bfloat16 *const *w2,
    __nv_bfloat16 *gate, __nv_bfloat16 *act, __nv_bfloat16 *rows,
    const __nv_bfloat16 **a13, const __nv_bfloat16 **x13, __nv_bfloat16 **y13,
    const __nv_bfloat16 **a2, const __nv_bfloat16 **x2, __nv_bfloat16 **y2,
    unsigned long long *summary, int routes, int first_expert, int hidden_dim,
    int intermediate) {
  int r = threadIdx.x;
  if (r >= routes) return;
  int global_expert = ids[r];
  bool local = global_expert >= first_expert && global_expert < first_expert + kLocalExperts;
  int expert = local ? global_expert - first_expert : 0;
  atomicAdd(summary + 1, 1ULL);
  if (local) atomicAdd(summary, 1ULL);
  // Dummy reads use a valid weight and zero input; every output has its own slot.
  a13[r] = w13[local ? expert : 0];
  x13[r] = local ? hidden + (r / kRoutesPerToken) * hidden_dim : zero;
  y13[r] = gate + r * 2 * intermediate;
  a2[r] = w2[local ? expert : 0];
  x2[r] = act + r * intermediate;
  y2[r] = rows + r * hidden_dim;
}

__global__ void route_reduce_kernel(const __nv_bfloat16 *rows, const int *ids,
                                     const float *weights, const int *errors,
                                     float *out, int first_expert, int hidden_dim) {
  int token = blockIdx.x;
  for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int r = 0; r < kRoutesPerToken; ++r) {
      int slot = token * kRoutesPerToken + r;
      int e = ids[slot] - first_expert;
      if (e >= 0 && e < kLocalExperts)
        acc += __bfloat162float(rows[slot * hidden_dim + d]) * weights[slot];
    }
    out[token * hidden_dim + d] = errors[token] ? CUDART_NAN_F : acc;
  }
}

__global__ void router_logits_kernel(
    const __nv_bfloat16 *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ gate_weight,
    float *__restrict__ logits,
    int seq_len,
    int hidden_dim,
    int n_experts) {
  int token = blockIdx.x;
  int expert = threadIdx.x;
  if (token >= seq_len || expert >= n_experts) return;

  float acc = 0.0f;
  const int hidden_base = token * hidden_dim;
  const int weight_base = expert * hidden_dim;
  for (int dim = 0; dim < hidden_dim; ++dim) {
    float product = __fmul_rn(
        __bfloat162float(hidden[hidden_base + dim]),
        __bfloat162float(gate_weight[weight_base + dim]));
    acc = __fadd_rn(acc, product);
  }
  logits[token * n_experts + expert] = acc;
}

__global__ void router_softmax_topk_kernel(
    const __nv_bfloat16 *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ gate_weight,
    float *__restrict__ topk_weight,
    int *__restrict__ topk_idx,
    int seq_len,
    int hidden_dim,
    int n_experts,
    int topk) {
  int token = blockIdx.x;
  int tid = threadIdx.x;
  if (token >= seq_len) return;

  __shared__ float logits[kMaxExperts];
  __shared__ float probs[kMaxExperts];
  __shared__ int selected_idx[kMaxTopk];
  __shared__ float selected_weight[kMaxTopk];

  if (tid < n_experts) {
    float acc = 0.0f;
    const int hidden_base = token * hidden_dim;
    const int weight_base = tid * hidden_dim;
    for (int dim = 0; dim < hidden_dim; ++dim) {
      acc = fmaf(
          __bfloat162float(hidden[hidden_base + dim]),
          __bfloat162float(gate_weight[weight_base + dim]),
          acc);
    }
    logits[tid] = acc;
  }
  __syncthreads();

  if (tid == 0) {
    // Probe-only fixed-topology router: keep the first version simple and
    // deterministic. Performance is not claimed for this diagnostic path.
    float max_score = -CUDART_INF_F;
    for (int expert = 0; expert < n_experts; ++expert) {
      max_score = fmaxf(max_score, logits[expert]);
    }

    float denom = 0.0f;
    for (int expert = 0; expert < n_experts; ++expert) {
      float value = expf(logits[expert] - max_score);
      probs[expert] = value;
      denom += value;
    }
    float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
    for (int expert = 0; expert < n_experts; ++expert) {
      probs[expert] *= inv_denom;
    }

    select_and_sort_routes(probs, n_experts, topk, selected_idx, selected_weight);

    int out_base = token * topk;
    for (int route = 0; route < topk; ++route) {
      topk_idx[out_base + route] = selected_idx[route];
      topk_weight[out_base + route] = selected_weight[route];
    }
  }
}

__global__ void accumulate_fixed_expert_kernel(
    const __nv_bfloat16 *__restrict__ expert_output,
    const float *__restrict__ topk_weight,
    const int *__restrict__ topk_idx,
    float *__restrict__ accum,
    int global_expert,
    int seq_len,
    int hidden_dim,
    int topk) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * hidden_dim;
  if (idx >= total) return;

  int token = idx / hidden_dim;
  int route_base = token * topk;
  float weight = 0.0f;
  for (int route = 0; route < topk; ++route) {
    if (topk_idx[route_base + route] == global_expert) {
      weight += topk_weight[route_base + route];
    }
  }
  if (weight != 0.0f) {
    accum[idx] = fmaf(__bfloat162float(expert_output[idx]), weight, accum[idx]);
  }
}

CUresult map_cuda_error(cudaError_t err) {
  switch (err) {
    case cudaSuccess:
      return CUDA_SUCCESS;
    case cudaErrorInvalidValue:
    case cudaErrorInvalidDevicePointer:
      return CUDA_ERROR_INVALID_VALUE;
    case cudaErrorInvalidDevice:
      return CUDA_ERROR_INVALID_DEVICE;
    case cudaErrorInvalidResourceHandle:
      return CUDA_ERROR_INVALID_HANDLE;
    case cudaErrorMemoryAllocation:
      return CUDA_ERROR_OUT_OF_MEMORY;
    case cudaErrorNotSupported:
      return CUDA_ERROR_NOT_SUPPORTED;
    case cudaErrorIllegalAddress:
      return CUDA_ERROR_ILLEGAL_ADDRESS;
    case cudaErrorLaunchOutOfResources:
      return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    case cudaErrorLaunchTimeout:
      return CUDA_ERROR_LAUNCH_TIMEOUT;
    case cudaErrorLaunchFailure:
      return CUDA_ERROR_LAUNCH_FAILED;
    case cudaErrorAssert:
      return CUDA_ERROR_ASSERT;
    case cudaErrorIllegalInstruction:
      return CUDA_ERROR_ILLEGAL_INSTRUCTION;
    case cudaErrorMisalignedAddress:
      return CUDA_ERROR_MISALIGNED_ADDRESS;
    case cudaErrorInvalidAddressSpace:
      return CUDA_ERROR_INVALID_ADDRESS_SPACE;
    case cudaErrorInvalidPc:
      return CUDA_ERROR_INVALID_PC;
    default:
      return CUDA_ERROR_UNKNOWN;
  }
}

CUresult consume_last_cuda_error() {
  cudaError_t err = cudaGetLastError();
  return map_cuda_error(err);
}

}  // namespace

extern "C" {

// DSV2-Lite route consumer. Pointer arrays reside on device; each entry is an
// independent N=1 problem, not a multi-row expert GEMM.
int dsv2_lite_pointer_gemm_cuda(const void *const *weights,
                               const void *const *inputs, void *const *outputs,
                               int m, int k, int routes, cudaStream_t stream) {
  if (!g_cublas_handle) return static_cast<int>(cudaErrorInvalidResourceHandle);
  if (!weights || !inputs || !outputs || m <= 0 || k <= 0 || routes <= 0 || routes > kMaxRows * kRoutesPerToken)
    return static_cast<int>(cudaErrorInvalidValue);
  cublasStatus_t status = cublasSetStream(g_cublas_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return kCublasErrorOffset + static_cast<int>(status);
  const float alpha = 1.0f, beta = 0.0f;
  status = cublasGemmBatchedEx(
      g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, 1, k,
      &alpha, weights, CUDA_R_16BF, k, inputs, CUDA_R_16BF, k,
      &beta, outputs, CUDA_R_16BF, m, routes,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS ? static_cast<int>(cudaPeekAtLastError())
                                       : kCublasErrorOffset + static_cast<int>(status);
}

CUresult dsv2_lite_route_logits_cuda(const float *logits, float *weights,
                                    int *ids, int *errors,
                                    unsigned long long *summary, int batch,
                                    int layer_idx, cudaStream_t stream) {
  if (!logits || !weights || !ids || !errors || !summary || batch < 1 || batch > kMaxRows ||
      layer_idx < 0)
    return CUDA_ERROR_INVALID_VALUE;
  route_from_logits_kernel<<<batch, 1, 0, stream>>>(
      logits, weights, ids, errors, summary, layer_idx);
  return map_cuda_error(cudaPeekAtLastError());
}

CUresult dsv2_lite_route_pointers_cuda(
    const int *ids, const __nv_bfloat16 *hidden, const __nv_bfloat16 *zero,
    const __nv_bfloat16 *const *w13, const __nv_bfloat16 *const *w2,
    __nv_bfloat16 *gate, __nv_bfloat16 *act, __nv_bfloat16 *rows,
    const __nv_bfloat16 **a13, const __nv_bfloat16 **x13, __nv_bfloat16 **y13,
    const __nv_bfloat16 **a2, const __nv_bfloat16 **x2, __nv_bfloat16 **y2,
    unsigned long long *summary, int batch, int first_expert, int hidden_dim,
    int intermediate, cudaStream_t stream) {
  if (!ids || !hidden || !zero || !w13 || !w2 || !gate || !act || !rows ||
      !a13 || !x13 || !y13 || !a2 || !x2 || !y2 || !summary ||
      batch < 1 || batch > kMaxRows ||
      (first_expert != 0 && first_expert != kLocalExperts) || hidden_dim <= 0 || intermediate <= 0)
    return CUDA_ERROR_INVALID_VALUE;
  route_pointers_kernel<<<1, kMaxExperts, 0, stream>>>(
      ids, hidden, zero, w13, w2, gate, act, rows, a13, x13, y13, a2, x2, y2,
      summary, batch * kRoutesPerToken, first_expert, hidden_dim, intermediate);
  return map_cuda_error(cudaPeekAtLastError());
}

CUresult dsv2_lite_route_reduce_cuda(const __nv_bfloat16 *rows, const int *ids,
                                    const float *weights, const int *errors,
                                    float *out, int batch, int first_expert,
                                    int hidden_dim, cudaStream_t stream) {
  if (!rows || !ids || !weights || !errors || !out || batch < 1 || batch > kMaxRows ||
      (first_expert != 0 && first_expert != kLocalExperts) || hidden_dim <= 0)
    return CUDA_ERROR_INVALID_VALUE;
  route_reduce_kernel<<<batch, kAccumThreads, 0, stream>>>(
      rows, ids, weights, errors, out, first_expert, hidden_dim);
  return map_cuda_error(cudaPeekAtLastError());
}

CUresult dsv2_lite_router_logits_cuda(
    const __nv_bfloat16 *hidden,
    const __nv_bfloat16 *gate_weight,
    float *logits,
    int seq_len,
    int hidden_dim,
    int n_experts,
    cudaStream_t stream) {
  if (hidden == nullptr || gate_weight == nullptr || logits == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (seq_len <= 0 || hidden_dim <= 0 || n_experts <= 0 || n_experts > kMaxExperts) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cudaGetLastError();
  router_logits_kernel<<<seq_len, kRouterThreads, 0, stream>>>(
      hidden, gate_weight, logits, seq_len, hidden_dim, n_experts);
  return consume_last_cuda_error();
}

CUresult dsv2_lite_router_softmax_topk_cuda(
    const __nv_bfloat16 *hidden,
    const __nv_bfloat16 *gate_weight,
    float *topk_weight,
    int *topk_idx,
    int seq_len,
    int hidden_dim,
    int n_experts,
    int topk,
    cudaStream_t stream) {
  if (hidden == nullptr || gate_weight == nullptr || topk_weight == nullptr ||
      topk_idx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (seq_len <= 0 || hidden_dim <= 0 || n_experts <= 0 || n_experts > kMaxExperts ||
      topk <= 0 || topk > kMaxTopk || topk > n_experts) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cudaGetLastError();
  router_softmax_topk_kernel<<<seq_len, kRouterThreads, 0, stream>>>(
      hidden, gate_weight, topk_weight, topk_idx, seq_len, hidden_dim, n_experts, topk);
  return consume_last_cuda_error();
}

CUresult dsv2_lite_accumulate_fixed_expert_cuda(
    const __nv_bfloat16 *expert_output,
    const float *topk_weight,
    const int *topk_idx,
    float *accum,
    int global_expert,
    int seq_len,
    int hidden_dim,
    int topk,
    cudaStream_t stream) {
  if (expert_output == nullptr || topk_weight == nullptr || topk_idx == nullptr ||
      accum == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (global_expert < 0 || seq_len <= 0 || hidden_dim <= 0 || topk <= 0 ||
      topk > kMaxTopk) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = seq_len * hidden_dim;
  int blocks = (total + kAccumThreads - 1) / kAccumThreads;
  cudaGetLastError();
  accumulate_fixed_expert_kernel<<<blocks, kAccumThreads, 0, stream>>>(
      expert_output, topk_weight, topk_idx, accum, global_expert, seq_len, hidden_dim, topk);
  return consume_last_cuda_error();
}

}  // extern "C"
