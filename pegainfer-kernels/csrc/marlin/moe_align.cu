// The expert-blocked dispatch the Marlin MoE kernel reads.
//
// Every expert's routes become one run padded up to a whole number of blocks,
// `expert_ids` names the owner of each block, and the unfilled entries point
// past the routes so the kernel skips them. All of it on the device: a host
// round trip here costs a stream synchronize per layer and rules out graph
// capture.
//
// The contract is vLLM's `moe_align_block_size`.

#include "ffi.cuh"

namespace {

__device__ __forceinline__ int round_up_to_block(int value, int block_size) {
  return ((value + block_size - 1) / block_size) * block_size;
}

__global__ void marlin_moe_align_small_kernel(
    const int* __restrict__ topk_idx,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ num_tokens_post_padded,
    uint32_t* __restrict__ expert_offsets,
    int route_elems,
    int global_start,
    int local_experts,
    int block_size) {
  extern __shared__ uint32_t metadata[];
  uint32_t* counts = metadata;
  uint32_t* starts = counts + local_experts;
  int tid = static_cast<int>(threadIdx.x);
  for (int expert = tid; expert < local_experts; expert += blockDim.x) {
    counts[expert] = 0;
  }
  __syncthreads();

  for (int route_offset = tid; route_offset < route_elems; route_offset += blockDim.x) {
    int expert = topk_idx[route_offset];
    if (expert >= global_start && expert < global_start + local_experts) {
      atomicAdd(&counts[expert - global_start], 1u);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int total = 0;
    for (int expert = 0; expert < local_experts; ++expert) {
      starts[expert] = static_cast<uint32_t>(total);
      total += round_up_to_block(static_cast<int>(counts[expert]), block_size);
    }
    starts[local_experts] = static_cast<uint32_t>(total);
    expert_offsets[local_experts] = static_cast<uint32_t>(total);
    num_tokens_post_padded[0] = total;
  }
  __syncthreads();

  for (int local_expert = tid; local_expert < local_experts;
       local_expert += blockDim.x) {
    int start = static_cast<int>(starts[local_expert]);
    int count = static_cast<int>(counts[local_expert]);
    int padded = round_up_to_block(count, block_size);
    for (int pos = start; pos < start + padded; pos += block_size) {
      expert_ids[pos / block_size] = local_expert;
    }
    int rank = 0;
    int global_expert = global_start + local_expert;
    for (int route_offset = 0; route_offset < route_elems; ++route_offset) {
      if (topk_idx[route_offset] == global_expert) {
        sorted_token_ids[start + rank] = route_offset;
        ++rank;
      }
    }
    for (int pos = start + count; pos < start + padded; ++pos) {
      sorted_token_ids[pos] = route_elems;
    }
    expert_offsets[local_expert] = starts[local_expert];
  }
}

__global__ void marlin_moe_align_clear_kernel(
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ num_tokens_post_padded,
    uint32_t* __restrict__ expert_offsets,
    int route_elems,
    int local_experts,
    int max_padded_tokens,
    int max_m_blocks) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int pos = idx; pos < max_padded_tokens; pos += stride) {
    sorted_token_ids[pos] = route_elems;
  }
  for (int block = idx; block < max_m_blocks; block += stride) {
    expert_ids[block] = -1;
  }
  for (int expert = idx; expert <= local_experts; expert += stride) {
    expert_offsets[expert] = 0;
  }
  if (idx == 0) {
    num_tokens_post_padded[0] = 0;
  }
}

__global__ void marlin_moe_align_count_kernel(
    const int* __restrict__ topk_idx,
    uint32_t* __restrict__ expert_offsets,
    int route_elems,
    int global_start,
    int local_experts) {
  int route_offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (route_offset >= route_elems) return;
  int expert = topk_idx[route_offset];
  if (expert >= global_start && expert < global_start + local_experts) {
    atomicAdd(&expert_offsets[expert - global_start + 1], 1u);
  }
}

__global__ void marlin_moe_align_prefix_kernel(
    int* __restrict__ expert_ids,
    int* __restrict__ num_tokens_post_padded,
    uint32_t* __restrict__ expert_offsets,
    int local_experts,
    int block_size) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int total = 0;
  for (int expert = 0; expert < local_experts; ++expert) {
    int count = static_cast<int>(expert_offsets[expert + 1]);
    int padded = round_up_to_block(count, block_size);
    expert_offsets[expert] = static_cast<uint32_t>(total);
    for (int pos = total; pos < total + padded; pos += block_size) {
      expert_ids[pos / block_size] = expert;
    }
    total += padded;
  }
  expert_offsets[local_experts] = static_cast<uint32_t>(total);
  num_tokens_post_padded[0] = total;
}

__global__ void marlin_moe_align_stable_fill_kernel(
    const int* __restrict__ topk_idx,
    int* __restrict__ sorted_token_ids,
    const uint32_t* __restrict__ expert_offsets,
    int route_elems,
    int global_start,
    int local_experts) {
  int local_expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_expert >= local_experts) return;
  int expert = global_start + local_expert;
  uint32_t rank = 0;
  uint32_t start = expert_offsets[local_expert];
  // M-tile row placement is numerically observable, so each expert follows input order.
  for (int route_offset = 0; route_offset < route_elems; ++route_offset) {
    if (topk_idx[route_offset] == expert) {
      sorted_token_ids[start + rank] = route_offset;
      ++rank;
    }
  }
}


}  // namespace

extern "C" {

CUresult marlin_moe_align_block_size_cuda(
    const int* topk_idx,
    int* sorted_token_ids,
    int* expert_ids,
    int* num_tokens_post_padded,
    uint32_t* expert_offsets,
    // Kept so the symbol's C signature does not change; never read.
    uint32_t* unused_expert_cursor,
    int active_tokens,
    int topk,
    int global_start,
    int local_experts,
    int block_size,
    int max_padded_tokens,
    int max_m_blocks,
    cudaStream_t stream) {
  (void)unused_expert_cursor;
  if (topk_idx == nullptr || sorted_token_ids == nullptr || expert_ids == nullptr ||
      num_tokens_post_padded == nullptr || expert_offsets == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (active_tokens <= 0 || topk <= 0 || global_start < 0 || local_experts <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!(block_size == 8 || (block_size >= 16 && block_size <= 64 && block_size % 16 == 0))) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int route_elems = active_tokens * topk;
  int required_padded = route_elems + local_experts * (block_size - 1);
  int required_blocks = (required_padded + block_size - 1) / block_size;
  if (max_padded_tokens < required_padded || max_m_blocks < required_blocks) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  constexpr int threads = 256;
  if (route_elems < 1024) {
    size_t shared = (2ull * static_cast<size_t>(local_experts) + 1) * sizeof(uint32_t);
    marlin_moe_align_small_kernel<<<1, threads, shared, stream>>>(
        topk_idx, sorted_token_ids, expert_ids, num_tokens_post_padded, expert_offsets,
        route_elems, global_start, local_experts, block_size);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
  }

  int clear_elems = max_padded_tokens;
  if (max_m_blocks > clear_elems) clear_elems = max_m_blocks;
  if (local_experts + 1 > clear_elems) clear_elems = local_experts + 1;
  int clear_blocks = (clear_elems + threads - 1) / threads;
  marlin_moe_align_clear_kernel<<<clear_blocks, threads, 0, stream>>>(
      sorted_token_ids, expert_ids, num_tokens_post_padded, expert_offsets, route_elems,
      local_experts, max_padded_tokens, max_m_blocks);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return CUDA_ERROR_LAUNCH_FAILED;

  int route_blocks = (route_elems + threads - 1) / threads;
  marlin_moe_align_count_kernel<<<route_blocks, threads, 0, stream>>>(
      topk_idx, expert_offsets, route_elems, global_start, local_experts);
  err = cudaGetLastError();
  if (err != cudaSuccess) return CUDA_ERROR_LAUNCH_FAILED;

  marlin_moe_align_prefix_kernel<<<1, 1, 0, stream>>>(
      expert_ids, num_tokens_post_padded, expert_offsets, local_experts, block_size);
  err = cudaGetLastError();
  if (err != cudaSuccess) return CUDA_ERROR_LAUNCH_FAILED;

  int expert_blocks = (local_experts + threads - 1) / threads;
  marlin_moe_align_stable_fill_kernel<<<expert_blocks, threads, 0, stream>>>(
      topk_idx, sorted_token_ids, expert_offsets, route_elems, global_start, local_experts);
  err = cudaGetLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

}  // extern "C"
