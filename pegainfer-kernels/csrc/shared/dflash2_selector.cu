#include "common.cuh"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// DFlash2 keeps a small candidate set per draft position. The selector is
// deliberately split into two kernels: top-k is embarrassingly parallel over
// logits, while the path walk has a request-local dependency on the previously
// selected token. Keeping that dependency out of the top-k kernel makes both
// launches easy to reason about and keeps the temporary layout graph-safe.
namespace {

constexpr int SELECTOR_TOP_K = 16;
constexpr int SELECTOR_TOPK_THREADS = 256;
constexpr int SELECTOR_WALK_THREADS = 512;

__device__ __forceinline__ bool selector_better(float lhs_value, int lhs_id,
                                                float rhs_value, int rhs_id) {
  return lhs_value > rhs_value ||
         (lhs_value == rhs_value && lhs_id < rhs_id);
}

__device__ __forceinline__ void selector_insert(float value, int id, float* values, int* ids) {
  if (!selector_better(value, id, values[SELECTOR_TOP_K - 1],
                       ids[SELECTOR_TOP_K - 1])) {
    return;
  }
  int slot = SELECTOR_TOP_K - 1;
  while (slot > 0 &&
         selector_better(value, id, values[slot - 1], ids[slot - 1])) {
    values[slot] = values[slot - 1];
    ids[slot] = ids[slot - 1];
    --slot;
  }
  values[slot] = value;
  ids[slot] = id;
}

__global__ void dflash2_selector_topk_kernel(
    const __nv_bfloat16* __restrict__ logits, uint32_t* __restrict__ ids,
    float* __restrict__ scores, int rows, int input_block_size,
    int position_offset, int positions_per_request, int vocab) {
  // Output rows are compact, while the source logits retain the anchor row.
  // Translate each compact row back to its request-major input row.
  const int row = blockIdx.x;
  if (row >= rows || positions_per_request <= 0) {
    return;
  }
  const int request = row / positions_per_request;
  const int position = row % positions_per_request;
  const size_t source_row = static_cast<size_t>(request) * input_block_size +
                            position_offset + position;

  // Each thread keeps a private top-16 list. The lists occupy 32 KiB of
  // shared memory and are merged by thread zero in canonical score/id order.
  __shared__ float local_values[SELECTOR_TOPK_THREADS][SELECTOR_TOP_K];
  __shared__ int local_ids[SELECTOR_TOPK_THREADS][SELECTOR_TOP_K];
  float* my_values = local_values[threadIdx.x];
  int* my_ids = local_ids[threadIdx.x];
  for (int j = 0; j < SELECTOR_TOP_K; ++j) {
    my_values[j] = -INFINITY;
    my_ids[j] = INT_MAX;
  }

  const __nv_bfloat16* row_logits = logits + source_row * vocab;
  for (int token = threadIdx.x; token < vocab;
       token += SELECTOR_TOPK_THREADS) {
    selector_insert(__bfloat162float(row_logits[token]), token, my_values,
                    my_ids);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float best_values[SELECTOR_TOP_K];
    int best_ids[SELECTOR_TOP_K];
    for (int j = 0; j < SELECTOR_TOP_K; ++j) {
      best_values[j] = -INFINITY;
      best_ids[j] = INT_MAX;
    }
    for (int thread = 0; thread < SELECTOR_TOPK_THREADS; ++thread) {
      for (int j = 0; j < SELECTOR_TOP_K; ++j) {
        selector_insert(local_values[thread][j], local_ids[thread][j],
                        best_values, best_ids);
      }
    }
    for (int j = 0; j < SELECTOR_TOP_K; ++j) {
      ids[static_cast<size_t>(row) * SELECTOR_TOP_K + j] =
          static_cast<uint32_t>(best_ids[j]);
      scores[static_cast<size_t>(row) * SELECTOR_TOP_K + j] = best_values[j];
    }
  }
}

__global__ void dflash2_selector_walk_kernel(
    const __nv_bfloat16* __restrict__ projected_hidden,
    const __nv_bfloat16* __restrict__ predecessor,
    const __nv_bfloat16* __restrict__ successor,
    const uint32_t* __restrict__ anchor_tokens,
    const uint32_t* __restrict__ candidate_ids,
    const float* __restrict__ candidate_unary, uint32_t* __restrict__ output,
    int requests, int input_block_size, int position_offset,
    int positions_per_request, int vocab, int rank) {
  const int request = blockIdx.x;
  if (request >= requests) {
    return;
  }

  __shared__ float edge_scores[SELECTOR_TOP_K];
  __shared__ uint32_t edge_ids[SELECTOR_TOP_K];
  __shared__ uint32_t previous;
  if (threadIdx.x == 0) {
    previous = anchor_tokens[request];
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int candidate = threadIdx.x >> 5;
  for (int position = 0; position < positions_per_request; ++position) {
    // Candidate/output rows are compact; hidden rows retain the anchor slot.
    const int row = request * positions_per_request + position;
    const size_t source_row = static_cast<size_t>(request) * input_block_size +
                              position_offset + position;
    if (candidate < SELECTOR_TOP_K) {
      const uint32_t candidate_id =
          candidate_ids[static_cast<size_t>(row) * SELECTOR_TOP_K + candidate];
      float dot = 0.0f;
      if (candidate_id < static_cast<uint32_t>(vocab) &&
          previous < static_cast<uint32_t>(vocab)) {
        const __nv_bfloat16* hidden_row =
            projected_hidden + source_row * rank;
        const __nv_bfloat16* predecessor_row =
            predecessor + static_cast<size_t>(previous) * rank;
        const __nv_bfloat16* successor_row =
            successor + static_cast<size_t>(candidate_id) * rank;
        for (int component = lane; component < rank; component += 32) {
          dot += __bfloat162float(predecessor_row[component]) *
                 __bfloat162float(hidden_row[component]) *
                 __bfloat162float(successor_row[component]);
        }
      }
      dot = warp_reduce_sum(dot);
      if (lane == 0) {
        edge_ids[candidate] = candidate_id;
        edge_scores[candidate] =
            candidate_unary[static_cast<size_t>(row) * SELECTOR_TOP_K +
                           candidate] +
            dot;
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      uint32_t best_id = edge_ids[0];
      float best_score = edge_scores[0];
      for (int j = 1; j < SELECTOR_TOP_K; ++j) {
        if (selector_better(edge_scores[j], static_cast<int>(edge_ids[j]),
                            best_score, static_cast<int>(best_id))) {
          best_score = edge_scores[j];
          best_id = edge_ids[j];
        }
      }
      output[row] = best_id;
      previous = best_id;
    }
    __syncthreads();
  }
}

}  // namespace

extern "C" int dflash2_selector_topk_cuda(
    const __nv_bfloat16* logits, uint32_t* candidate_ids,
    float* candidate_scores, int rows, int input_block_size,
    int position_offset, int positions_per_request, int vocab,
    cudaStream_t stream) {
  if (logits == nullptr || candidate_ids == nullptr || candidate_scores == nullptr ||
      rows <= 0 || input_block_size <= 0 || position_offset < 0 ||
      positions_per_request <= 0 ||
      position_offset > input_block_size - positions_per_request ||
      vocab < SELECTOR_TOP_K) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  dflash2_selector_topk_kernel<<<rows, SELECTOR_TOPK_THREADS, 0, stream>>>(
      logits, candidate_ids, candidate_scores, rows, input_block_size,
      position_offset, positions_per_request, vocab);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int dflash2_selector_walk_cuda(
    const __nv_bfloat16* projected_hidden,
    const __nv_bfloat16* predecessor,
    const __nv_bfloat16* successor,
    const uint32_t* anchor_tokens,
    const uint32_t* candidate_ids,
    const float* candidate_unary,
    uint32_t* output,
    int requests,
    int input_block_size,
    int position_offset,
    int positions_per_request,
    int vocab,
    int rank,
    cudaStream_t stream) {
  if (projected_hidden == nullptr || predecessor == nullptr ||
      successor == nullptr || anchor_tokens == nullptr || candidate_ids == nullptr ||
      candidate_unary == nullptr || output == nullptr || requests <= 0 ||
      input_block_size <= 0 || position_offset < 0 ||
      positions_per_request <= 0 ||
      position_offset > input_block_size - positions_per_request ||
      vocab < SELECTOR_TOP_K || rank <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  dflash2_selector_walk_kernel<<<requests, SELECTOR_WALK_THREADS, 0, stream>>>(
      projected_hidden, predecessor, successor, anchor_tokens,
      candidate_ids, candidate_unary, output, requests, input_block_size,
      position_offset, positions_per_request, vocab, rank);
  return static_cast<int>(cudaGetLastError());
}
