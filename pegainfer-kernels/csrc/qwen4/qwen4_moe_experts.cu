// Qwen3.8-Flash-Next routed experts: the route metadata that orders the
// (token, pick) pairs by expert, and the weighted combine that sums each
// token's expert outputs back. The expert GEMMs between the two are cuBLAS,
// one per expert over its slot range.
//
// Route metadata. `ids` is `[rows, top_k]` i32. `indptr[e]` is the first slot
// of expert `e` and `indptr[experts]` the pair count; `gather_rows[slot]` is the
// token a slot reads; `route_slot[t * top_k + j]` is the slot of that pair, or
// -1 for an id outside `[0, experts)`. Slots of one expert keep pair order, so
// the layout is the same on every run.
//
// Weighted combine. `out[t] = sum_j f32(w[t, j]) * f32(down_out[route_slot[t, j]])`,
// accumulated in f32 and rounded once. The reference adds expert by expert in
// bf16; the difference sits below the gate's floor.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <climits>

namespace {

constexpr int kMetaThreads = 512;
constexpr int kCombineThreads = 256;

// One block. Thread e owns expert e: a count pass, a serial exclusive scan over
// the counts in shared memory, then a placement pass in pair order.
__global__ void qwen4_moe_route_meta_kernel(const int* __restrict__ ids,
                                            int* __restrict__ indptr,
                                            int* __restrict__ gather_rows,
                                            int* __restrict__ route_slot, int pairs,
                                            int top_k, int experts) {
  extern __shared__ int counts[];
  for (int e = threadIdx.x; e <= experts; e += blockDim.x) counts[e] = 0;
  for (int p = threadIdx.x; p < pairs; p += blockDim.x) route_slot[p] = -1;
  __syncthreads();
  for (int e = threadIdx.x; e < experts; e += blockDim.x) {
    int n = 0;
    for (int p = 0; p < pairs; ++p) n += (ids[p] == e);
    counts[e + 1] = n;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int e = 0; e < experts; ++e) counts[e + 1] += counts[e];
  }
  __syncthreads();
  for (int e = threadIdx.x; e <= experts; e += blockDim.x) indptr[e] = counts[e];
  for (int e = threadIdx.x; e < experts; e += blockDim.x) {
    int slot = counts[e];
    for (int p = 0; p < pairs; ++p) {
      if (ids[p] == e) {
        gather_rows[slot] = p / top_k;
        route_slot[p] = slot;
        ++slot;
      }
    }
  }
}

__global__ void qwen4_moe_weighted_combine_kernel(
    const __nv_bfloat16* __restrict__ down_out, const __nv_bfloat16* __restrict__ weights,
    const int* __restrict__ route_slot, __nv_bfloat16* __restrict__ out, long long total,
    int hidden, int top_k) {
  for (long long at = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
       at < total; at += gridDim.x * static_cast<long long>(blockDim.x)) {
    const long long token = at / hidden;
    const int h = static_cast<int>(at - token * hidden);
    float acc = 0.0f;
    for (int j = 0; j < top_k; ++j) {
      const long long pair = token * top_k + j;
      const int slot = route_slot[pair];
      if (slot < 0) continue;
      acc += __bfloat162float(weights[pair]) *
             __bfloat162float(down_out[static_cast<long long>(slot) * hidden + h]);
    }
    out[at] = __float2bfloat16(acc);
  }
}

}  // namespace

extern "C" {

CUresult qwen4_moe_route_meta_cuda(const int* ids, int* indptr, int* gather_rows,
                                   int* route_slot, int rows, int top_k, int experts,
                                   cudaStream_t stream) {
  if (ids == nullptr || indptr == nullptr || gather_rows == nullptr || route_slot == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rows < 0 || top_k <= 0 || experts <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const long long pairs = static_cast<long long>(rows) * top_k;
  const size_t smem = static_cast<size_t>(experts + 1) * sizeof(int);
  if (pairs > INT_MAX || smem > 48 * 1024) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  qwen4_moe_route_meta_kernel<<<1, kMetaThreads, smem, stream>>>(
      ids, indptr, gather_rows, route_slot, static_cast<int>(pairs), top_k, experts);
  return consume_last_cuda_error();
}

CUresult qwen4_moe_weighted_combine_cuda(const __nv_bfloat16* down_out,
                                         const __nv_bfloat16* weights, const int* route_slot,
                                         __nv_bfloat16* out, int tokens, int top_k, int hidden,
                                         cudaStream_t stream) {
  if (down_out == nullptr || weights == nullptr || route_slot == nullptr || out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (tokens < 0 || top_k <= 0 || hidden <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (tokens == 0) return CUDA_SUCCESS;
  const long long total = static_cast<long long>(tokens) * hidden;
  const long long grid = (total + kCombineThreads - 1) / kCombineThreads;
  if (grid > INT_MAX) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  qwen4_moe_weighted_combine_kernel<<<static_cast<int>(grid), kCombineThreads, 0, stream>>>(
      down_out, weights, route_slot, out, total, hidden, top_k);
  return consume_last_cuda_error();
}

}  // extern "C"
