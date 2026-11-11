#include "common.cuh"
#include "ffi_guard.cuh"

#include <cstdint>

namespace {

// The drafter's dense K/V mirrors and the paged pool are `2 * layers` tensors
// with the same row count, so one launch copies every buffer a call names. The
// restore gathers all of them at once; the materialize folds a layer's K and V
// into one launch (its per-layer scratch is reused, so the layers themselves
// cannot share a launch). The pair table travels in the launch parameters, so
// no device-side staging is needed.
constexpr int kDraftKvMaxPairs = 32;

struct DraftKvPair {
  const __nv_bfloat16* src;
  __nv_bfloat16* dst;
};

struct DraftKvPairTable {
  DraftKvPair pairs[kDraftKvMaxPairs];
};

// One block per row per buffer, 16 bytes per thread: a warp moves 512
// contiguous bytes, four 128-byte sectors. The page table makes the
// destination row arbitrary, but each row is contiguous, so a block never
// needs the 64-bit div/mod a flat element index paid to recover (row, col).
//
// `VEC == 8` is the aligned path: a row stride that is a multiple of 8 bf16 is
// 16-byte aligned because the tensor allocations are. `VEC == 1` copies
// elements when the stride is not.
template <bool Gather, int VEC>
__global__ void draft_kv_rows_kernel(
    const DraftKvPairTable table, const int* __restrict__ locs,
    int pair_count, int hidden) {
  const int pair_index = static_cast<int>(blockIdx.y);
  if (pair_index >= pair_count) {
    return;
  }
  const int row = static_cast<int>(blockIdx.x);
  const int64_t row_off = static_cast<int64_t>(row) * hidden;
  const int64_t slot_off = static_cast<int64_t>(locs[row]) * hidden;
  const DraftKvPair pair = table.pairs[pair_index];
  const __nv_bfloat16* src_row = Gather ? pair.src + slot_off : pair.src + row_off;
  __nv_bfloat16* dst_row = Gather ? pair.dst + row_off : pair.dst + slot_off;

  const int chunks = hidden / VEC;
  if constexpr (VEC == 8) {
    const uint4* src_vec = reinterpret_cast<const uint4*>(src_row);
    uint4* dst_vec = reinterpret_cast<uint4*>(dst_row);
    for (int i = static_cast<int>(threadIdx.x); i < chunks;
         i += static_cast<int>(blockDim.x)) {
      dst_vec[i] = src_vec[i];
    }
  } else {
    for (int i = static_cast<int>(threadIdx.x); i < chunks;
         i += static_cast<int>(blockDim.x)) {
      dst_row[i] = src_row[i];
    }
  }
}

template <bool Gather>
CUresult launch_rows(const __nv_bfloat16* const* srcs,
                     __nv_bfloat16* const* dsts, int pair_count,
                     const int* locs, int rows, int hidden,
                     cudaStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return CUDA_SUCCESS;
  }
  if (srcs == nullptr || dsts == nullptr || locs == nullptr ||
      pair_count <= 0 || pair_count > kDraftKvMaxPairs) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  DraftKvPairTable table{};
  for (int i = 0; i < pair_count; ++i) {
    if (srcs[i] == nullptr || dsts[i] == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    table.pairs[i].src = srcs[i];
    table.pairs[i].dst = dsts[i];
  }
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(pair_count));
  if (hidden % 8 == 0) {
    draft_kv_rows_kernel<Gather, 8>
        <<<grid, 128, 0, stream>>>(table, locs, pair_count, hidden);
  } else {
    draft_kv_rows_kernel<Gather, 1>
        <<<grid, 256, 0, stream>>>(table, locs, pair_count, hidden);
  }
  return map_cuda_error(cudaGetLastError());
}

}  // namespace

extern "C" CUresult draft_kv_scatter_cuda(
    const __nv_bfloat16* const* srcs, __nv_bfloat16* const* dsts,
    int pair_count, const int* locs, int rows, int hidden,
    cudaStream_t stream) {
  return launch_rows<false>(srcs, dsts, pair_count, locs, rows, hidden, stream);
}

extern "C" CUresult draft_kv_gather_cuda(
    const __nv_bfloat16* const* srcs, __nv_bfloat16* const* dsts,
    int pair_count, const int* locs, int rows, int hidden,
    cudaStream_t stream) {
  return launch_rows<true>(srcs, dsts, pair_count, locs, rows, hidden, stream);
}
