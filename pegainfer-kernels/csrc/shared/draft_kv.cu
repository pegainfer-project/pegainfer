#include "common.cuh"
#include "ffi_guard.cuh"

#include <algorithm>
#include <cstdint>

namespace {

// Scatter/gather whole bf16 rows between a dense scratch and the paged draft
// KV pool. Row destinations come from the target KV page table, so they are
// arbitrary: one launch copies every row instead of one launch per token.
__global__ void draft_kv_scatter_kernel(
    const __nv_bfloat16* src, __nv_bfloat16* dst, const int* locs,
    int rows, int hidden) {
  const int64_t total = static_cast<int64_t>(rows) * hidden;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int row = static_cast<int>(index / hidden);
    const int col = static_cast<int>(index % hidden);
    dst[static_cast<int64_t>(locs[row]) * hidden + col] =
        src[static_cast<int64_t>(row) * hidden + col];
  }
}

__global__ void draft_kv_gather_kernel(
    const __nv_bfloat16* src, __nv_bfloat16* dst, const int* locs,
    int rows, int hidden) {
  const int64_t total = static_cast<int64_t>(rows) * hidden;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int row = static_cast<int>(index / hidden);
    const int col = static_cast<int>(index % hidden);
    dst[static_cast<int64_t>(row) * hidden + col] =
        src[static_cast<int64_t>(locs[row]) * hidden + col];
  }
}

inline int grid_for(int64_t count) {
  return static_cast<int>(std::min<int64_t>((count + 255) / 256, 65535));
}

}  // namespace

extern "C" CUresult draft_kv_scatter_cuda(
    const __nv_bfloat16* src, __nv_bfloat16* dst, const int* locs,
    int rows, int hidden, cudaStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return CUDA_SUCCESS;
  }
  draft_kv_scatter_kernel<<<grid_for(static_cast<int64_t>(rows) * hidden), 256, 0, stream>>>(
      src, dst, locs, rows, hidden);
  return map_cuda_error(cudaGetLastError());
}

extern "C" CUresult draft_kv_gather_cuda(
    const __nv_bfloat16* src, __nv_bfloat16* dst, const int* locs,
    int rows, int hidden, cudaStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return CUDA_SUCCESS;
  }
  draft_kv_gather_kernel<<<grid_for(static_cast<int64_t>(rows) * hidden), 256, 0, stream>>>(
      src, dst, locs, rows, hidden);
  return map_cuda_error(cudaGetLastError());
}
