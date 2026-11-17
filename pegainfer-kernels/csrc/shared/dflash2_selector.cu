#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>
#include <math_constants.h>
#include <flashinfer/sampling.cuh>
#include <flashinfer/topk.cuh>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "ffi_guard.cuh"

namespace {

constexpr int kCandidates = 16;
constexpr unsigned kInvalidLogit = 1;
constexpr unsigned kInsufficientCandidates = 2;
constexpr unsigned kInvalidToken = 4;
constexpr unsigned kNonfiniteScore = 8;

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

int blocks(int64_t elements) {
  return static_cast<int>(std::min<int64_t>((elements + 255) / 256, 65535));
}

// TopK keeps the input row layout. Drop anchor rows only when unpacking the
// small candidate set; walk resolves score ties without sorting candidates.
__global__ void unpack_candidates(const __nv_bfloat16* values, float* unary,
                                 int block_size, unsigned* error) {
  const int64_t row = blockIdx.x;
  const int64_t source = (row / (block_size - 1)) * block_size + 1 + row % (block_size - 1);

  if (threadIdx.x < kCandidates) {
    const float score = __bfloat162float(values[source * kCandidates + threadIdx.x]);
    unary[row * kCandidates + threadIdx.x] = score;
    if (!isfinite(score)) {
      atomicOr(error, score == -CUDART_INF_F ? kInsufficientCandidates : kInvalidLogit);
    }
  }
}

// These are model-specific gathers, fused with BF16->FP32 conversion and the
// hidden gate. Matrix multiplication remains the shared cuBLAS operation.
__global__ void gather_gated_codebooks(
    const __nv_bfloat16* hidden, const __nv_bfloat16* predecessor,
    const __nv_bfloat16* successor, const unsigned* anchors, const unsigned* ids,
    float* gated, float* successors, int rows, int block_size, int vocab,
    int rank, unsigned* error) {
  const int64_t count = static_cast<int64_t>(rows) * kCandidates * rank;

  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int component = index % rank;
    const int candidate = (index / rank) % kCandidates;
    const int row = index / (static_cast<int64_t>(rank) * kCandidates);
    const int request = row / (block_size - 1);
    const int position = row % (block_size - 1);
    const int64_t source = static_cast<int64_t>(request) * block_size + 1 + position;
    const unsigned prev =
        position == 0
            ? anchors[request]
            : ids[(source - 1) * kCandidates + candidate];
    const unsigned next = ids[source * kCandidates + candidate];

    if (prev >= static_cast<unsigned>(vocab) || next >= static_cast<unsigned>(vocab)) {
      atomicOr(error, kInvalidToken);
      gated[index] = 0.0f;
      successors[index] = 0.0f;
      continue;
    }

    const float h = __bfloat162float(hidden[source * rank + component]);
    const float a = __bfloat162float(predecessor[static_cast<int64_t>(prev) * rank + component]);
    const float b = __bfloat162float(successor[static_cast<int64_t>(next) * rank + component]);
    const float product = a * h;
    if (!isfinite(h) || !isfinite(a) || !isfinite(b) || !isfinite(product)) {
      atomicOr(error, kNonfiniteScore);
    }
    gated[index] = product;
    successors[index] = b;
  }
}

__global__ void add_unary(float* edges, const float* unary, int rows, unsigned* error) {
  const int64_t count = static_cast<int64_t>(rows) * kCandidates * kCandidates;

  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = index / (kCandidates * kCandidates);
    const float score = edges[index] + unary[row * kCandidates + index % kCandidates];
    edges[index] = score;
    if (!isfinite(score)) {
      atomicOr(error, kNonfiniteScore);
    }
  }
}

// One logical 16-lane warp handles a request. Positions remain sequential;
// candidates within a position share reductions and the CDF scan.
__global__ void walk(const float* edges, const unsigned* ids,
                    const float* temperatures, const float* uniforms,
                    float* proposal_probs, unsigned* selected, int batch,
                    int length, const unsigned* error) {
  constexpr int kWalkThreads = 256;
  constexpr int kRequestsPerBlock = kWalkThreads / kCandidates;
  using Sum = cub::WarpReduce<float, kCandidates>;
  using Scan = cub::WarpScan<float, kCandidates>;
  __shared__ typename Sum::TempStorage sums[kRequestsPerBlock];
  __shared__ typename Scan::TempStorage scans[kRequestsPerBlock];

  const int group = threadIdx.x / kCandidates;
  const int lane = threadIdx.x % kCandidates;
  const int warp_base = threadIdx.x & kCandidates;
  const unsigned mask = 0xffffu << warp_base;
  for (int request = blockIdx.x * kRequestsPerBlock + group; request < batch;
       request += gridDim.x * kRequestsPerBlock) {
    int predecessor = 0;
    for (int position = 0; position < length; ++position) {
      const int64_t row = static_cast<int64_t>(request) * length + position;
      if (*error != 0) {
        if (lane == 0) selected[row] = 0xffffffffu;
        continue;
      }

      const float* scores = edges + (row * kCandidates + predecessor) * kCandidates;
      const int64_t source = static_cast<int64_t>(request) * (length + 1) + 1 + position;
      const unsigned candidate = ids[source * kCandidates + lane];
      const float score = scores[lane];
      float best_score = score;
      unsigned best_id = candidate;
      for (int distance = kCandidates / 2; distance > 0; distance /= 2) {
        const float other_score = __shfl_xor_sync(mask, best_score, distance, kCandidates);
        const unsigned other_id = __shfl_xor_sync(mask, best_id, distance, kCandidates);
        if (other_score > best_score || (other_score == best_score && other_id < best_id)) {
          best_score = other_score;
          best_id = other_id;
        }
      }
      const unsigned best_mask = (__ballot_sync(mask, candidate == best_id) & mask) >> warp_base;
      int picked = __ffs(best_mask) - 1;
      const float temperature = temperatures[request];
      float probability;
      if (temperature > 0.0f) {
        probability = expf((score - best_score) / temperature);
        float mass = Sum(sums[group]).Sum(probability);
        mass = __shfl_sync(mask, mass, 0, kCandidates);
        probability /= mass;
        float cumulative;
        Scan(scans[group]).InclusiveSum(probability, cumulative);
        const unsigned matches = (__ballot_sync(mask, uniforms[row] < cumulative) & mask) >> warp_base;
        if (matches != 0) {
          picked = __ffs(matches) - 1;
        } else {
          // As in the scalar walk, an FP32 CDF below one uses its last positive entry.
          const unsigned positive = (__ballot_sync(mask, probability > 0.0f) & mask) >> warp_base;
          picked = 31 - __clz(positive);
        }
      } else {
        probability = lane == picked ? 1.0f : 0.0f;
      }
      proposal_probs[row * kCandidates + lane] = probability;
      const unsigned token = __shfl_sync(mask, candidate, picked, kCandidates);
      if (lane == 0) selected[row] = token;
      predecessor = picked;
      // CUB storage is reused by this logical warp at the next position.
      __syncwarp(mask);
    }
  }
}

}  // namespace

// Dimensions and buffer extents are checked by dflash2_select_into.
extern "C" int dflash2_prepare_cuda(
    const __nv_bfloat16* logits, const __nv_bfloat16* hidden,
    const __nv_bfloat16* predecessor, const __nv_bfloat16* successor,
    const unsigned* anchors, unsigned* ids, float* unary, float* gated,
    float* successors, unsigned* error, __nv_bfloat16* topk_values,
    uint8_t* row_states, int batch, int block_size, int vocab, int rank,
    cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  check_cuda(cudaMemsetAsync(error, 0, sizeof(unsigned), stream));
  const int rows = batch * (block_size - 1);

  check_cuda(flashinfer::sampling::RadixTopKMultiCTA<__nv_bfloat16, int32_t>(
      const_cast<__nv_bfloat16*>(logits), reinterpret_cast<int32_t*>(ids), topk_values, nullptr,
      batch * block_size, kCandidates, vocab,
      reinterpret_cast<flashinfer::sampling::RadixRowState*>(row_states),
      /*deterministic=*/true, stream));

  unpack_candidates<<<rows, 32, 0, stream>>>(topk_values, unary, block_size, error);
  check_cuda(cudaPeekAtLastError());

  gather_gated_codebooks<<<blocks(static_cast<int64_t>(rows) * kCandidates * rank), 256, 0, stream>>>(
      hidden, predecessor, successor, anchors, ids, gated, successors,
      rows, block_size, vocab, rank, error);
  check_cuda(cudaPeekAtLastError());

  return 0;
  PEGAINFER_FFI_GUARD_END(-1)
}

extern "C" int dflash2_finish_cuda(float* edges, const float* unary,
                                  const unsigned* ids, const float* temperatures,
                                  const float* uniforms, float* proposal_probs,
                                  unsigned* selected,
                                  unsigned* error, int batch, int length,
                                  cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  add_unary<<<blocks(static_cast<int64_t>(batch) * length * kCandidates * kCandidates), 256, 0, stream>>>(
      edges, unary, batch * length, error);
  check_cuda(cudaPeekAtLastError());

  walk<<<blocks(static_cast<int64_t>(batch) * kCandidates), 256, 0, stream>>>(
      edges, ids, temperatures, uniforms, proposal_probs, selected, batch, length, error);
  check_cuda(cudaPeekAtLastError());

  return 0;
  PEGAINFER_FFI_GUARD_END(-1)
}
