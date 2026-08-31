#include "../common.cuh"
#include "../marlin/ffi.cuh"

#include <cuda.h>
#include <stdint.h>

extern "C" {

namespace {

constexpr int kKimiLocalExperts = 48;
__global__ void kimi_add_f32_bf16_to_bf16_kernel(
    const float* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = __float2bfloat16(a[idx] + __bfloat162float(b[idx]));
}

__global__ void kimi_residual_add_scaled_f32_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ projected,
    const float* __restrict__ routed_f32,
    float scale,
    __nv_bfloat16* __restrict__ out,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  __nv_bfloat16 rounded = __float2bfloat16(
      __bfloat162float(hidden[idx]) + __bfloat162float(projected[idx]));
  float scaled = __fmul_rn(routed_f32[idx], scale);
  float sum = __fadd_rn(scaled, __bfloat162float(rounded));
  out[idx] = __float2bfloat16(sum);
}

}  // namespace

CUresult kimi_moe_marlin_align_block_size_cuda(
    const int* topk_idx,
    int* sorted_token_ids,
    int* expert_ids,
    int* num_tokens_post_padded,
    uint32_t* expert_offsets,
    uint32_t* unused_expert_cursor,
    int active_tokens,
    int topk,
    int global_start,
    int local_experts,
    int block_size,
    int max_padded_tokens,
    int max_m_blocks,
    cudaStream_t stream) {
  if (topk != 8 || local_experts != kKimiLocalExperts) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return marlin_moe_align_block_size_cuda(
      topk_idx, sorted_token_ids, expert_ids, num_tokens_post_padded,
      expert_offsets, unused_expert_cursor, active_tokens, topk, global_start, local_experts,
      block_size, max_padded_tokens, max_m_blocks, stream);
}

CUresult kimi_add_f32_bf16_to_bf16_cuda(
    const float* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    int n,
    cudaStream_t stream) {
  if (a == nullptr || b == nullptr || out == nullptr || n < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n == 0) return CUDA_SUCCESS;
  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  kimi_add_f32_bf16_to_bf16_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

CUresult kimi_residual_add_scaled_f32_cuda(
    const __nv_bfloat16* hidden,
    const __nv_bfloat16* projected,
    const float* routed_f32,
    float scale,
    __nv_bfloat16* out,
    int n,
    cudaStream_t stream) {
  if (hidden == nullptr || projected == nullptr || routed_f32 == nullptr ||
      out == nullptr || n < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n == 0) return CUDA_SUCCESS;
  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  kimi_residual_add_scaled_f32_kernel<<<blocks, threads, 0, stream>>>(
      hidden, projected, routed_f32, scale, out, n);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

__global__ void kimi_residual_add_scaled_bf16_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ projected,
    const __nv_bfloat16* __restrict__ routed,
    float scale,
    __nv_bfloat16* __restrict__ out,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  __nv_bfloat16 rounded = __float2bfloat16(
      __bfloat162float(hidden[idx]) + __bfloat162float(projected[idx]));
  float scaled = __fmul_rn(__bfloat162float(routed[idx]), scale);
  float sum = __fadd_rn(scaled, __bfloat162float(rounded));
  out[idx] = __float2bfloat16(sum);
}

// BF16-routed sibling of kimi_residual_add_scaled_f32: the DeepEP combine
// reduces expert outputs to bf16, so the routed contribution arrives one
// bf16 rounding earlier than the old f32 PPLX combine did.
CUresult kimi_residual_add_scaled_bf16_cuda(
    const __nv_bfloat16* hidden,
    const __nv_bfloat16* projected,
    const __nv_bfloat16* routed,
    float scale,
    __nv_bfloat16* out,
    int n,
    cudaStream_t stream) {
  if (hidden == nullptr || projected == nullptr || routed == nullptr ||
      out == nullptr || n < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n == 0) return CUDA_SUCCESS;
  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  kimi_residual_add_scaled_bf16_kernel<<<blocks, threads, 0, stream>>>(
      hidden, projected, routed, scale, out, n);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

// Marlin routing metadata from a DeepEP post-epilogue expert prefix sum.
//
// DeepEP's dispatch writes psum_expert as the exclusive prefix sum of the
// per-local-expert ALIGNED counts (num_local_experts + 1 entries), and the
// copy epilogue's per-slot atomicAdd advances entry i from aligned_start_i
// to aligned_start_i + count_i. Entry [num_local_experts] is untouched and
// holds the total aligned expanded row count. Recover, for expert i:
//   start_i = round_up(psum[i-1], alignment)   (psum[-1] == 0)
//   count_i = psum[i] - start_i
// The expanded recv buffer is already expert-major and aligned, so
// sorted_token_ids is identity over real rows and sentinel over pad rows.
//
// All capacity-sized writes are spread across the whole block: every slot
// binary-searches the shared segment table instead of one thread owning one
// expert's segment (skewed routing) or the tail (serial writes dominated the
// decode profile at 8528-slot capacity).
//
// The expert_ids lookup maps m-block -> owning expert via the slot at the
// block's start, which is only well-defined when alignment == block_size
// (segment boundaries land on block boundaries); the Rust wrapper enforces
// that equality.
constexpr int kRoutingBuilderThreads = 1024;
constexpr int kRoutingBuilderMaxExperts = 64;

__global__ void kimi_deepep_build_marlin_routing_kernel(
    const int32_t* __restrict__ psum_expert,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ num_tokens_post_padded,
    int num_local_experts,
    int alignment,
    int block_size,
    int max_padded_tokens,
    int max_m_blocks) {
  __shared__ int s_start[kRoutingBuilderMaxExperts];
  __shared__ int s_count[kRoutingBuilderMaxExperts];
  __shared__ int s_total;

  int tid = threadIdx.x;
  if (tid < num_local_experts) {
    int start = 0;
    if (tid > 0) {
      int prev = psum_expert[tid - 1];
      start = ((prev + alignment - 1) / alignment) * alignment;
    }
    int count = psum_expert[tid] - start;
    s_start[tid] = start;
    s_count[tid] = count < 0 ? 0 : count;
  }
  if (tid == 0) s_total = psum_expert[num_local_experts];
  __syncthreads();

  int total = s_total;
  int sentinel = max_padded_tokens;

  // Segment lookup: largest expert e with s_start[e] <= idx. Aligned starts
  // are non-decreasing, so a binary search over the shared table suffices.
  for (int idx = tid; idx < max_padded_tokens; idx += blockDim.x) {
    int value = sentinel;
    if (idx < total) {
      int lo = 0, hi = num_local_experts - 1;
      while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (s_start[mid] <= idx) lo = mid; else hi = mid - 1;
      }
      if (idx - s_start[lo] < s_count[lo]) value = idx;
    }
    sorted_token_ids[idx] = value;
  }

  for (int b = tid; b < max_m_blocks; b += blockDim.x) {
    int slot = b * block_size;
    int expert = 0;
    if (slot < total) {
      int lo = 0, hi = num_local_experts - 1;
      while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (s_start[mid] <= slot) lo = mid; else hi = mid - 1;
      }
      expert = lo;
    }
    expert_ids[b] = expert;
  }

  if (tid == 0) num_tokens_post_padded[0] = total;
}

CUresult kimi_deepep_build_marlin_routing_on_stream(
    const int32_t* psum_expert,
    int32_t* sorted_token_ids,
    int32_t* expert_ids,
    int32_t* num_tokens_post_padded,
    int num_local_experts,
    int alignment,
    int block_size,
    int max_padded_tokens,
    int max_m_blocks,
    cudaStream_t stream) {
  if (num_local_experts > kRoutingBuilderMaxExperts) return CUDA_ERROR_INVALID_VALUE;
  kimi_deepep_build_marlin_routing_kernel<<<1, kRoutingBuilderThreads, 0, stream>>>(
      psum_expert,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_padded,
      num_local_experts,
      alignment,
      block_size,
      max_padded_tokens,
      max_m_blocks);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

}  // extern "C"
