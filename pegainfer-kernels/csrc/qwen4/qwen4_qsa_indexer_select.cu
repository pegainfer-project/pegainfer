// Qwen3.8-Flash-Next (Qwen4 line) QSA token-indexer block selection.
//
// Turns the per-(row, block) scores of `qwen4_qsa_indexer_score.cu` into the
// token-level visibility mask the reference builds, frozen semantics from
// transformers 5.16.1 `qwen4_exp` (`Qwen4ExpTextQSAIndexer.forward`). For a
// query row at position `p` (0-based, tokens 0..p visible):
//
//   blocks = (p + 1) / ratio          complete micro-blocks, each 4 tokens
//   tail   = (p + 1) - 4 * blocks     tokens 4 * blocks ..= p
//   keep   = min(budget / ratio, blocks)
//   visible = the tokens of the `keep` highest block scores, ties broken by
//             the LOWER block index, plus the tail
//
// The selection order is descending score, ascending block index. `torch.topk`
// leaves the pick at the cut open, so a block tied with the last kept one is an
// equally valid pick; this kernel's canonical tie rule is the one the parity
// gate holds the fixture to.
//
// GRID CONTRACT, inherited from the score pass: the visible list starts at
// token 0, so a row at `p` sees exactly tokens [0, p] and block b covers
// [4b, 4b + 4). A caller whose visible list starts elsewhere (an evicted
// prefix) has to hand both passes a shifted key array; nothing here can detect
// one.
//
// A non-finite score inside `[0, blocks)` or an out-of-range `rows` entry is an
// upstream contract violation and traps. The score pass leaves the columns past
// `blocks` untouched, and this kernel never reads them.
//
// One CTA per query row. The target is the portable subset (no TMA, no cluster,
// no wgmma). A row under the budget keeps every complete block and writes the
// visible prefix in one pass. Otherwise an exact 8-bit radix select over the
// row's sortable f32 keys (four passes) finds the `keep`-th score, and the
// blocks equal to that threshold are marked in ascending block-index order to
// preserve the tie rule, one bit per block in dynamic shared memory sized from
// the widest row this launch can see (`tokens / ratio` blocks). The host
// refuses a launch whose bitmap plus the block's static shared state exceeds
// the default per-block ceiling.
//

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kRatio = 4;  // indexer_compress_ratio
// The budget's block ceiling: 2048 tokens / 4. `tokens` itself is not bounded
// by this (a row can see far more blocks than the budget keeps).
constexpr int kMaxKeptBlocks = 512;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / WARP_SIZE;
constexpr int kRadixBins = 256;
// Per-warp radix histograms, the threshold state and the equality prefix,
// allocated statically beside the dynamic bitmap and therefore counted against
// the same per-block ceiling.
constexpr int kStaticSharedBytes =
    kWarps * kRadixBins * static_cast<int>(sizeof(unsigned int)) +
    2 * static_cast<int>(sizeof(unsigned int)) +
    (kThreads + 1) * static_cast<int>(sizeof(int));
// The default per-block shared-memory ceiling on every supported arch; the
// bitmap plus the block's static state must fit under it without an opt-in.
constexpr int kMaxSharedBytes = 48 * 1024;

// f32 -> monotonic u32 for the radix select. -0.0 and +0.0 compare equal in the
// selection order, so fold both zeros to +0.0 before taking the bits.
__device__ __forceinline__ unsigned int sortable_key(float score) {
  const unsigned int bits = __float_as_uint(score == 0.0f ? 0.0f : score);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// One bit per block, set once a threshold-equal block has been taken; every
// block above the threshold is kept without a mark.
__device__ __forceinline__ bool block_taken(const unsigned int* taken, int block) {
  return ((taken[block >> 5] >> (block & 31)) & 1u) != 0u;
}

__device__ __forceinline__ void mark_taken(unsigned int* taken, int block) {
  // The equality pass marks from several threads, and equal keys share a word.
  atomicOr(&taken[block >> 5], 1u << (block & 31));
}

// Per query row: validate the row, check its scores, select the kept blocks and
// write the row's visibility mask. One CTA per row, the warps split the blocks.
__global__ void qwen4_qsa_select_kernel(
    const float* __restrict__ scores,   // [query_rows, score_cols]
    const int* __restrict__ rows,       // [query_rows] positions
    unsigned char* __restrict__ mask,   // [query_rows, tokens]
    int tokens,
    int score_cols,
    int budget) {
  extern __shared__ unsigned int taken[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  const int pos = rows[row];
  if (pos < 0 || pos >= tokens) {
    __trap();
  }
  const int visible = pos + 1;
  const int blocks = visible / kRatio;
  const int tail = visible - blocks * kRatio;

  const float* row_scores = scores + static_cast<int64_t>(row) * score_cols;
  for (int block = tid; block < blocks; block += blockDim.x) {
    if (!isfinite(row_scores[block])) {
      __trap();
    }
  }

  const int keep_cap = budget / kRatio;
  const int keep = keep_cap < blocks ? keep_cap : blocks;
  unsigned char* row_mask = mask + static_cast<int64_t>(row) * tokens;

  // Under the budget every complete block is kept: one pass writes the whole
  // visible prefix and the search is skipped.
  if (keep == blocks) {
    for (int token = tid; token < tokens; token += blockDim.x) {
      row_mask[token] = token < visible ? 1 : 0;
    }
    return;
  }

  // Exact radix select of the keep-th key: four 8-bit passes narrow the
  // threshold one key byte at a time.
  __shared__ unsigned int hist[kWarps * kRadixBins];
  __shared__ unsigned int radix_prefix;
  __shared__ int radix_remaining;
  const int warp = tid / WARP_SIZE;
  if (tid == 0) {
    radix_prefix = 0u;
    radix_remaining = keep;
  }
  __syncthreads();
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = tid; i < kWarps * kRadixBins; i += blockDim.x) {
      hist[i] = 0u;
    }
    __syncthreads();
    const unsigned int prefix = radix_prefix;
    const unsigned int high_mask =
        shift == 24 ? 0u : (0xFFFFFFFFu << (shift + 8));
    for (int block = tid; block < blocks; block += blockDim.x) {
      const unsigned int key = sortable_key(row_scores[block]);
      if ((key & high_mask) != (prefix & high_mask)) {
        continue;
      }
      atomicAdd(&hist[warp * kRadixBins + ((key >> shift) & 0xFFu)], 1u);
    }
    __syncthreads();
    if (tid == 0) {
      int remaining = radix_remaining;
      int chosen = -1;
      for (int bin = kRadixBins - 1; bin >= 0; --bin) {
        int count = 0;
        for (int w = 0; w < kWarps; ++w) {
          count += static_cast<int>(hist[w * kRadixBins + bin]);
        }
        if (count >= remaining) {
          chosen = bin;
          break;
        }
        remaining -= count;
      }
      if (chosen < 0) {
        __trap();  // `keep` exceeds the row's block count
      }
      radix_prefix = prefix | (static_cast<unsigned int>(chosen) << shift);
      radix_remaining = remaining;
    }
    __syncthreads();
  }
  const unsigned int threshold = radix_prefix;
  const int need = radix_remaining;

  // Zero the mask row and the bitmap that marks the threshold-equal blocks.
  const int max_blocks = tokens / kRatio;
  const int words = (max_blocks + 31) >> 5;
  for (int token = tid; token < tokens; token += blockDim.x) {
    row_mask[token] = 0;
  }
  for (int word = tid; word < words; word += blockDim.x) {
    taken[word] = 0u;
  }
  __syncthreads();

  // Equal keys are ordered by block index, matching the canonical tie rule.
  // Contiguous ranges make a per-thread prefix the global order.
  __shared__ int equal_offsets[kThreads + 1];
  const int start = static_cast<int>(static_cast<int64_t>(tid) * blocks / blockDim.x);
  const int end =
      static_cast<int>(static_cast<int64_t>(tid + 1) * blocks / blockDim.x);
  int equal_count = 0;
  for (int block = start; block < end; ++block) {
    equal_count += sortable_key(row_scores[block]) == threshold;
  }
  equal_offsets[tid] = equal_count;
  __syncthreads();
  if (tid == 0) {
    int running = 0;
    for (int i = 0; i < kThreads; ++i) {
      const int count = equal_offsets[i];
      equal_offsets[i] = running;
      running += count;
    }
    equal_offsets[kThreads] = running;
    if (running < need) {
      __trap();  // the radix threshold must cover the kept rank
    }
  }
  __syncthreads();
  int take = need - equal_offsets[tid];
  for (int block = start; block < end && take > 0; ++block) {
    if (sortable_key(row_scores[block]) == threshold) {
      mark_taken(taken, block);
      --take;
    }
  }
  __syncthreads();

  // Every block above the threshold is kept, the chosen equals fill the rest,
  // and the tail is unconditional.
  for (int block = tid; block < blocks; block += blockDim.x) {
    const unsigned int key = sortable_key(row_scores[block]);
    const bool kept =
        key > threshold || (key == threshold && block_taken(taken, block));
    if (!kept) {
      continue;
    }
    const int base = block * kRatio;
#pragma unroll
    for (int i = 0; i < kRatio; ++i) {
      row_mask[base + i] = 1;
    }
  }
  for (int i = tid; i < tail; i += blockDim.x) {
    row_mask[blocks * kRatio + i] = 1;
  }
}

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

}  // namespace

extern "C" {

// Select the kept blocks of every query row and write its token visibility
// mask. Row `j` covers tokens `[0, rows[j]]`; `mask` is one byte per (row,
// token), 0 or 1, with row stride `tokens`. `scores` is one f32 row of
// `score_cols` per query row, of which `(rows[j] + 1) / kRatio` are read.
//
// `rows` carries the query rows' positions in the token order (also the row
// order of `scores` and `mask`); it is not assumed to be every position.
// `budget` is the token budget, a positive multiple of kRatio whose block count
// is at most kMaxKeptBlocks.
//
// The compiled geometry is kRatio = 4 and one byte per mask cell; a caller
// whose buffers are laid out differently cannot be detected here. An empty
// launch is refused; `qsa_indexer_select_into` treats one as a no-op before it
// reaches this entry.
CUresult qwen4_qsa_indexer_select_cuda(
    const float* scores,
    const int* rows,
    unsigned char* mask,
    int query_rows,
    int tokens,
    int score_cols,
    int budget,
    cudaStream_t stream) {
  if (scores == nullptr || rows == nullptr || mask == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (query_rows <= 0 || tokens <= 0 || score_cols < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (budget <= 0 || budget % kRatio != 0 || budget / kRatio > kMaxKeptBlocks) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // Every row reads at most `tokens / kRatio` blocks from its score row.
  if (tokens / kRatio > score_cols) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int max_blocks = tokens / kRatio;
  const int words = (max_blocks + 31) >> 5;
  const int64_t shared_bytes =
      static_cast<int64_t>(words) * static_cast<int64_t>(sizeof(unsigned int));
  if (shared_bytes + kStaticSharedBytes > kMaxSharedBytes) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  qwen4_qsa_select_kernel<<<query_rows, kThreads,
                            static_cast<size_t>(shared_bytes), stream>>>(
      scores, rows, mask, tokens, score_cols, budget);
  return consume_last_cuda_error();
}

}  // extern "C"
