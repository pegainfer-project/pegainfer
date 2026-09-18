#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_topk.cuh>
#include <cuda/stream>

#include <algorithm>
#include <climits>
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

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

int blocks(int64_t elements) {
  return static_cast<int>(std::min<int64_t>((elements + 255) / 256, 65535));
}

// A unique key orders logits descending, then token IDs ascending, including
// ties at the K/K+1 boundary. Canonicalize signed zero without perturbing logits.
__device__ uint64_t score_key(float score, unsigned token) {
  if (score == 0.0f) {
    score = 0.0f;
  }

  const unsigned bits = __float_as_uint(score);
  const unsigned ordered = bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
  return (static_cast<uint64_t>(ordered) << 32) | (0xffffffffu - token);
}

__device__ float key_score(uint64_t key) {
  const unsigned ordered = static_cast<unsigned>(key >> 32);
  return __uint_as_float(ordered ^ ((ordered >> 31) ? 0x80000000u : 0xffffffffu));
}

__global__ void pack_keys(const __nv_bfloat16* logits, uint64_t* keys,
                         int vocab, int block_size, int chunk_rows,
                         int row_start, int rows, unsigned* error) {
  const int64_t count = static_cast<int64_t>(chunk_rows) * vocab;

  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = static_cast<int64_t>(row_start) + index / vocab;
    const unsigned token = index % vocab;
    float score = -CUDART_INF_F;

    if (row < rows) {
      const int request = row / (block_size - 1);
      const int position = row % (block_size - 1);
      const int64_t source = static_cast<int64_t>(request) * block_size + 1 + position;
      score = __bfloat162float(logits[source * vocab + token]);
      if (isnan(score) || score == CUDART_INF_F) {
        atomicOr(error, kInvalidLogit);
        score = -CUDART_INF_F;
      }
    }

    keys[index] = score_key(score, token);
  }
}

// DeviceTopK returns an unordered set. Sort just those 16 unique keys, not the
// whole vocabulary. Zero padding sorts below every encoded key, including -Inf.
__global__ void unpack_candidates(const uint64_t* keys, unsigned* ids, float* unary,
                                 int row_start, int rows, unsigned* error) {
  using Sort = cub::BlockRadixSort<uint64_t, 32, 1>;
  __shared__ typename Sort::TempStorage storage;

  const int64_t row = static_cast<int64_t>(row_start) + blockIdx.x;
  uint64_t key[1] = {
      threadIdx.x < kCandidates
          ? keys[static_cast<int64_t>(blockIdx.x) * kCandidates + threadIdx.x]
          : 0};
  Sort(storage).SortDescending(key);

  if (row < rows && threadIdx.x < kCandidates) {
    const int64_t output = row * kCandidates + threadIdx.x;
    ids[output] = 0xffffffffu - static_cast<unsigned>(key[0]);
    unary[output] = key_score(key[0]);
    if (!isfinite(unary[output])) {
      atomicOr(error, kInsufficientCandidates);
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
    const unsigned prev =
        position == 0
            ? anchors[request]
            : ids[static_cast<int64_t>(row - 1) * kCandidates + candidate];
    const unsigned next = ids[static_cast<int64_t>(row) * kCandidates + candidate];

    if (prev >= static_cast<unsigned>(vocab) || next >= static_cast<unsigned>(vocab)) {
      atomicOr(error, kInvalidToken);
      gated[index] = 0.0f;
      successors[index] = 0.0f;
      continue;
    }

    const int64_t source = static_cast<int64_t>(request) * block_size + 1 + position;
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
      const unsigned* candidates = ids + row * kCandidates;
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

auto topk_environment(cudaStream_t stream) {
  // CUB permits non-deterministic membership only among equal keys. Each key
  // includes its token ID, so the top-16 set is unique even when logits tie.
  auto requirements = cuda::execution::require(
      cuda::execution::determinism::not_guaranteed,
      cuda::execution::output_ordering::unsorted);
  return cuda::std::execution::env{cuda::stream_ref{stream}, requirements};
}

}  // namespace

extern "C" int dflash2_topk_workspace_bytes_cuda(int vocab, size_t* bytes,
                                               cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  require(vocab >= kCandidates, "DFlash2 vocab must contain at least 16 candidates");
  require(bytes != nullptr, "DFlash2 null workspace size output");

  check_cuda(cub::DeviceTopK::MaxKeys(
      nullptr, *bytes, static_cast<const uint64_t*>(nullptr),
      static_cast<uint64_t*>(nullptr), vocab, kCandidates, topk_environment(stream)));

  return 0;
  PEGAINFER_FFI_GUARD_END(-1)
}

extern "C" int dflash2_prepare_cuda(
    const __nv_bfloat16* logits, const __nv_bfloat16* hidden,
    const __nv_bfloat16* predecessor, const __nv_bfloat16* successor,
    const unsigned* anchors, unsigned* ids, float* unary, float* gated,
    float* successors, unsigned* error, uint64_t* keys_in, uint64_t* keys_out,
    void* workspace, size_t workspace_bytes, int batch, int block_size,
    int vocab, int rank, int chunk_rows, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  require(batch > 0 && block_size >= 2 && rank > 0 && chunk_rows > 0 &&
              vocab >= kCandidates &&
              static_cast<int64_t>(batch) * block_size <= INT_MAX,
          "DFlash2 invalid prepare dimensions");
  require(logits && hidden && predecessor && successor && anchors && ids && unary &&
              gated && successors && error && keys_in && keys_out && workspace,
          "DFlash2 null prepare pointer");

  check_cuda(cudaMemsetAsync(error, 0, sizeof(unsigned), stream));
  const int rows = batch * (block_size - 1);
  const auto env = topk_environment(stream);

  for (int64_t row_start = 0; row_start < rows; row_start += chunk_rows) {
    pack_keys<<<blocks(static_cast<int64_t>(chunk_rows) * vocab), 256, 0, stream>>>(
        logits, keys_in, vocab, block_size, chunk_rows, row_start, rows, error);
    check_cuda(cudaPeekAtLastError());

    // Each row reuses the same library workspace on the owning stream.
    for (int row = 0; row < chunk_rows; ++row) {
      check_cuda(cub::DeviceTopK::MaxKeys(
          workspace, workspace_bytes,
          static_cast<const uint64_t*>(keys_in) + static_cast<int64_t>(row) * vocab,
          keys_out + static_cast<int64_t>(row) * kCandidates,
          vocab, kCandidates, env));
    }

    unpack_candidates<<<chunk_rows, 32, 0, stream>>>(
        keys_out, ids, unary, row_start, rows, error);
    check_cuda(cudaPeekAtLastError());
  }

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
  require(batch > 0 && length > 0 && static_cast<int64_t>(batch) * length <= INT_MAX,
          "DFlash2 invalid finish dimensions");
  require(edges && unary && ids && selected && error, "DFlash2 null finish pointer");

  add_unary<<<blocks(static_cast<int64_t>(batch) * length * kCandidates * kCandidates), 256, 0, stream>>>(
      edges, unary, batch * length, error);
  check_cuda(cudaPeekAtLastError());

  walk<<<blocks(batch), 256, 0, stream>>>(edges, ids, selected, batch, length, error);
  check_cuda(cudaPeekAtLastError());

  return 0;
  PEGAINFER_FFI_GUARD_END(-1)
}
