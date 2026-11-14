// Qwen3.8-Flash-Next routed experts: the route metadata that orders the
// (token, pick) pairs by expert, and the weighted combine that sums each
// token's expert outputs back. The expert GEMMs between the two are a grouped
// CUTLASS GEMM (`qwen4_moe_grouped.cu`) for prefill and cuBLAS, one call per
// expert over its slot range, for small token counts.
//
// Route metadata. `ids` is `[rows, top_k]` i32. `indptr[e]` is the first slot
// of expert `e` and `indptr[experts]` the pair count; `gather_rows[slot]` is the
// token a slot reads; `route_slot[t * top_k + j]` is the slot of that pair, or
// -1 for an id outside `[0, experts)`. Slots of one expert keep pair order, so
// the layout is the same on every run.
//
// Two route-metadata kernels produce exactly that layout. The sequential one
// (one CTA, thread `e` owns expert `e`) is the small-token path and the tiled
// one's reference; the tiled one (histogram, scan, place) is the prefill path,
// where the sequential kernel's `2 * experts * pairs` comparisons on one SM
// dominate the layer.
//
// Weighted combine. `out[t] = sum_j f32(w[t, j]) * f32(down_out[route_slot[t, j]])`,
// accumulated in f32 in ascending `j` and rounded once. The reference adds
// expert by expert in bf16; the difference sits below the gate's floor.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>

namespace {

constexpr int kMetaThreads = 512;
constexpr int kCombineThreads = 256;
/// Pairs one warp owns in the tiled route-metadata kernels.
constexpr int kRouteChunkPairs = 256;
/// Warps (chunks) per CTA in the tiled route-metadata kernels.
constexpr int kRouteWarps = 4;

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

// Pass 1: one warp per chunk of `kRouteChunkPairs` pairs counts that chunk's
// experts into its own shared histogram and stores the row. Counts only, so
// nothing here depends on the order the lanes retire in.
__global__ void qwen4_moe_route_meta_hist_kernel(const int* __restrict__ ids,
                                                 int* __restrict__ chunk_hist, int pairs,
                                                 int experts, int chunks) {
  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int chunk = blockIdx.x * kRouteWarps + warp;
  if (chunk >= chunks) return;
  extern __shared__ int hist[];
  int* mine = hist + warp * experts;
  for (int e = lane; e < experts; e += WARP_SIZE) mine[e] = 0;
  __syncwarp();
  const int first = chunk * kRouteChunkPairs;
  const int last = min(first + kRouteChunkPairs, pairs);
  // One step of lookahead: the next 32 ids are in flight while this step's
  // ballots run, which is what the pass is short of (one 128 B L2 read per step
  // and nothing else to hide it with).
  int e = first + lane < last ? ids[first + lane] : -1;
  for (int base = first; base < last; base += WARP_SIZE) {
    const int next = base + WARP_SIZE + lane;
    const int next_e = next < last ? __ldg(ids + next) : -1;
    const unsigned mask = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && e < experts && lane == __ffs(mask) - 1) mine[e] += __popc(mask);
    // This step's leader is the lowest lane holding the expert, which differs
    // step to step, so the next step's leader must see this one's shared update.
    __syncwarp();
    e = next_e;
  }
  __syncwarp();
  for (int e = lane; e < experts; e += WARP_SIZE) chunk_hist[chunk * experts + e] = mine[e];
}

// Pass 2: one block, one thread per expert. Thread e turns the chunk histogram
// into the exclusive chunk prefix in place (so the place pass needs one add,
// not a walk), then the per-expert totals take a block-wide exclusive scan into
// `indptr`. `indptr[experts]` is the valid pair count.
__global__ void qwen4_moe_route_meta_scan_kernel(int* __restrict__ chunk_hist,
                                                 int* __restrict__ indptr, int experts,
                                                 int chunks) {
  extern __shared__ int totals[];
  const int e = threadIdx.x;
  int run = 0;
  for (int b = 0; b < chunks; ++b) {
    const int count = chunk_hist[b * experts + e];
    chunk_hist[b * experts + e] = run;
    run += count;
  }
  totals[e] = run;
  __syncthreads();
  for (int off = 1; off < experts; off <<= 1) {
    const int add = e >= off ? totals[e - off] : 0;
    __syncthreads();
    totals[e] += add;
    __syncthreads();
  }
  if (e == 0) indptr[0] = 0;
  indptr[e + 1] = totals[e];
}

// Pass 3: the placement. Chunks ascend, warp steps ascend inside a chunk and
// lanes ascend inside a step, so `rank` is the count of same-expert pairs
// before this one — the sequential kernel's slot order, pair for pair.
__global__ void qwen4_moe_route_meta_place_kernel(const int* __restrict__ ids,
                                                  const int* __restrict__ chunk_hist,
                                                  const int* __restrict__ indptr,
                                                  int* __restrict__ gather_rows,
                                                  int* __restrict__ route_slot, int pairs,
                                                  int top_k, int experts, int chunks) {
  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int chunk = blockIdx.x * kRouteWarps + warp;
  if (chunk >= chunks) return;
  extern __shared__ int shared[];
  int* base_of = shared + warp * 2 * experts;
  int* mine = base_of + experts;
  const int* chunk_prefix = chunk_hist + chunk * experts;
  // This warp's slot bases, one scattered global pair of loads per expert
  // instead of two per pair.
  for (int e = lane; e < experts; e += WARP_SIZE) {
    base_of[e] = indptr[e] + chunk_prefix[e];
    mine[e] = 0;
  }
  __syncwarp();
  const int first = chunk * kRouteChunkPairs;
  const int last = min(first + kRouteChunkPairs, pairs);
  const unsigned lane_below = (1u << lane) - 1u;
  int e = first + lane < last ? ids[first + lane] : -1;
  for (int base = first; base < last; base += WARP_SIZE) {
    const int next = base + WARP_SIZE + lane;
    const int next_e = next < last ? __ldg(ids + next) : -1;
    const unsigned mask = __match_any_sync(0xffffffffu, e);
    const bool valid = e >= 0 && e < experts;
    int slot = -1;
    if (valid) {
      const int leader = __ffs(mask) - 1;
      int step_base = 0;
      if (lane == leader) {
        step_base = mine[e];
        mine[e] = step_base + __popc(mask);
      }
      step_base = __shfl_sync(mask, step_base, leader);
      slot = base_of[e] + step_base + __popc(mask & lane_below);
    }
    const int p = base + lane;
    if (p < last) {
      route_slot[p] = slot;
      if (valid) gather_rows[slot] = p / top_k;
    }
    // The leader per expert changes with the ballot, so order the shared
    // counter updates between steps (racecheck flags this without it).
    __syncwarp();
    e = next_e;
  }
  __syncwarp();
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

// The tiled route-metadata path: three launches whose work grows with `pairs`
// instead of with `experts * pairs`. `chunk_hist` is `chunks * experts` i32 of
// caller-owned scratch, `chunks = ceil(pairs / kRouteChunkPairs)`; the scan
// leaves it holding the exclusive per-chunk prefix, so the buffer is scratch,
// not output. The output is bit-identical to `qwen4_moe_route_meta_cuda`.
CUresult qwen4_moe_route_meta_tiled_cuda(const int* ids, int* indptr, int* gather_rows,
                                         int* route_slot, int* chunk_hist, int rows, int top_k,
                                         int experts, long long chunk_hist_entries,
                                         cudaStream_t stream) {
  if (ids == nullptr || indptr == nullptr || gather_rows == nullptr || route_slot == nullptr ||
      chunk_hist == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // The scan kernel holds one thread per expert; the tiled kernels hold one
  // shared row of one int per expert per warp.
  if (rows <= 0 || top_k <= 0 || experts <= 0 || experts > 1024) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const long long pairs = static_cast<long long>(rows) * top_k;
  const long long chunks = (pairs + kRouteChunkPairs - 1) / kRouteChunkPairs;
  if (pairs > INT_MAX || chunks > INT_MAX) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (chunk_hist_entries < chunks * experts) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int blocks = static_cast<int>((chunks + kRouteWarps - 1) / kRouteWarps);
  const size_t row_smem = static_cast<size_t>(kRouteWarps) * experts * sizeof(int);
  if (row_smem > 48 * 1024) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int chunk_count = static_cast<int>(chunks);
  qwen4_moe_route_meta_hist_kernel<<<blocks, kRouteWarps * WARP_SIZE, row_smem, stream>>>(
      ids, chunk_hist, static_cast<int>(pairs), experts, chunk_count);
  {
    CUresult status = consume_last_cuda_error();
    if (status != CUDA_SUCCESS) return status;
  }
  qwen4_moe_route_meta_scan_kernel<<<1, experts, static_cast<size_t>(experts) * sizeof(int),
                                     stream>>>(chunk_hist, indptr, experts, chunk_count);
  {
    CUresult status = consume_last_cuda_error();
    if (status != CUDA_SUCCESS) return status;
  }
  qwen4_moe_route_meta_place_kernel<<<blocks, kRouteWarps * WARP_SIZE, 2 * row_smem, stream>>>(
      ids, chunk_hist, indptr, gather_rows, route_slot, static_cast<int>(pairs), top_k, experts,
      chunk_count);
  return consume_last_cuda_error();
}

}  // extern "C"
