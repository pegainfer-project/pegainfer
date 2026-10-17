#include <cuda_bf16.h>
#include <cuda_runtime.h>
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

// One bounded walk per request; no host-visible predecessor dependency. This
// is greedy along the selected edges, not independent argmax or Viterbi.
__global__ void walk(const float* edges, const unsigned* ids, unsigned* selected,
                    int batch, int length, const unsigned* error) {
  for (int request = blockIdx.x * blockDim.x + threadIdx.x; request < batch;
       request += blockDim.x * gridDim.x) {
    int predecessor = 0;
    for (int position = 0; position < length; ++position) {
      const int64_t row = static_cast<int64_t>(request) * length + position;
      if (*error != 0) {
        selected[row] = 0xffffffffu;
        continue;
      }

      const float* scores = edges + (row * kCandidates + predecessor) * kCandidates;
      const int64_t source = static_cast<int64_t>(request) * (length + 1) + 1 + position;
      const unsigned* candidates = ids + source * kCandidates;
      int best = 0;
      for (int candidate = 1; candidate < kCandidates; ++candidate) {
        if (scores[candidate] > scores[best] ||
            (scores[candidate] == scores[best] && candidates[candidate] < candidates[best])) {
          best = candidate;
        }
      }

      selected[row] = candidates[best];
      predecessor = best;
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
                                  const unsigned* ids, unsigned* selected,
                                  unsigned* error, int batch, int length,
                                  cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  add_unary<<<blocks(static_cast<int64_t>(batch) * length * kCandidates * kCandidates), 256, 0, stream>>>(
      edges, unary, batch * length, error);
  check_cuda(cudaPeekAtLastError());

  walk<<<blocks(batch), 256, 0, stream>>>(edges, ids, selected, batch, length, error);
  check_cuda(cudaPeekAtLastError());

  return 0;
  PEGAINFER_FFI_GUARD_END(-1)
}
