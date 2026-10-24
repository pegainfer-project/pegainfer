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
// The tail is visible unconditionally: it is the incomplete block the top-k
// grid never scores, and the reference keeps it outside the block grid. Since
// blocks only ever cover tokens below `p`, the mask is causal by construction;
// no token past the row's own position can be set.
//
// GRID CONTRACT, inherited from the score pass: the visible list starts at
// token 0, so a row at `p` sees exactly tokens [0, p] and block b covers
// [4b, 4b + 4). This kernel takes no `visible_len`: the mask's row order and
// its causal ceiling both come from `rows`, which is what the reference's
// causal path serves. A caller whose visible list starts elsewhere (an evicted
// prefix) has to hand both passes a shifted key array; nothing here can detect
// one, and a score left over past a row's real prefix would be selected as if
// it were valid.
//
// The selection order is a total order over `(score, block index)`: descending
// score, ascending index. `torch.topk` leaves the pick at the cut open, so a
// block tied with the last kept one is an equally valid pick; this kernel's
// tie rule is the canonical one the gate holds the fixture to.
//
// A non-finite score inside `[0, blocks)` is an upstream contract violation and
// traps. Zeroing it (or skipping it, which is what a bare comparison does)
// would be worse than a visible failure: the reference's score is a relu sum,
// so a poison reaching a block means the indexer input is already broken, and
// silently dropping that block hands the sparse attention a mask whose kept
// count is short of the budget. The mask's complement drives the attention
// softmax, and an all-masked row is undefined there. The kernel never reads the
// NaN padding past `blocks`: the score pass leaves those columns untouched and
// they are not part of any row's contract.
//
// One CTA per query row. Portability is the point (no TMA, no cluster, no
// wgmma): the selection is a comparison-only SIMT reduction. The search is
// `keep` masked argmax passes over the row's blocks — O(keep * blocks) — with an
// exclusion bitmap in dynamic shared memory, one bit per block. The bitmap is
// sized from the widest row this launch can see (`tokens / ratio` blocks); the
// host refuses a launch whose bitmap plus the block's static shared state
// passes the 48 KiB default per-block ceiling (392960 blocks, 1.57M tokens),
// which is far past the checkpoint's 262144 positions, so no caller this line
// serves is refused by it.
//
// VARIANT: the last argument selects gate-only mutations of the same launch
// (production always passes 0). The kernel refuses an unknown value. These
// exist so the parity gate can prove its assertions bite on the real kernel
// path without weakening the production branch:
//   0 production                 as described above
//   1 drop the tail              the tail bits are left 0
//   2 keep fewer                 `keep - 1` blocks instead of `keep`
//   3 keep more                  `keep + 1` blocks, capped at `blocks`
//   4 ignore causality           every column past the row's own token is set
//   5 invert the order           the LOWEST scores are kept, ties still to the
//                                lower index

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
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / WARP_SIZE;
// One score and one index per warp, allocated statically beside the dynamic
// bitmap and therefore counted against the same per-block ceiling.
constexpr int kStaticSharedBytes =
    kWarps * (static_cast<int>(sizeof(float)) + static_cast<int>(sizeof(int)));
// The default per-block shared-memory ceiling on every supported arch; the
// exclusion bitmap plus the block's static state must fit under it without an
// opt-in.
constexpr int kMaxSharedBytes = 48 * 1024;

enum SelectVariant : int {
  kProduction = 0,
  kDropTail = 1,
  kKeepFewer = 2,
  kKeepMore = 3,
  kIgnoreCausality = 4,
  kInvertOrder = 5,
};

// The canonical order: descending score, ascending block index; `invert` flips
// only the score direction (the gate's negative control), never the tie rule,
// so a tied pair still resolves to the lower index either way. An absent
// candidate (`index < 0`) never outranks a present one, which is what lets the
// pass start from "nothing found yet".
__device__ __forceinline__ bool outranks(float score, int index, float best_score,
                                         int best_index, bool invert) {
  if (index < 0) {
    return false;
  }
  if (best_index < 0) {
    return true;
  }
  if (score != best_score) {
    return invert ? (score < best_score) : (score > best_score);
  }
  return index < best_index;
}

// One bit per block, set once a block has been picked. The bitmap is the whole
// exclusion state; no per-pick list is kept, so memory stays one bit per block.
__device__ __forceinline__ bool block_picked(const unsigned int* picked, int block) {
  return ((picked[block >> 5] >> (block & 31)) & 1u) != 0u;
}

__device__ __forceinline__ void mark_picked(unsigned int* picked, int block) {
  picked[block >> 5] |= 1u << (block & 31);
}

// Per query row: validate the row, check its scores, select the kept blocks and
// write the row's visibility mask. One CTA per row, the warps split the blocks.
__global__ void qwen4_qsa_select_kernel(
    const float* __restrict__ scores,   // [query_rows, score_cols]
    const int* __restrict__ rows,       // [query_rows] positions
    unsigned char* __restrict__ mask,   // [query_rows, tokens]
    int tokens,
    int score_cols,
    int budget,
    int variant) {
  extern __shared__ unsigned int picked[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  // `rows[row]` is the row's own position, and `[0, pos]` the tokens it sees.
  // Both a negative entry and one past the key array trap: clamping either
  // would hand the caller a mask for blocks the row never asked about. The
  // budget is checked here too, although the host entry refuses a bad one, so a
  // caller that reaches the kernel through another path still fails visibly.
  const int pos = rows[row];
  if (pos < 0 || pos >= tokens) {
    __trap();
  }
  const int visible = pos + 1;
  const int blocks = visible / kRatio;
  const int tail = visible - blocks * kRatio;
  if (blocks > score_cols) {
    __trap();
  }
  if (budget <= 0 || budget % kRatio != 0 || budget / kRatio > kMaxKeptBlocks) {
    __trap();
  }
  if (variant < kProduction || variant > kInvertOrder) {
    __trap();
  }

  const float* row_scores = scores + static_cast<int64_t>(row) * score_cols;
  // Fail closed on a poisoned score before any comparison can skip it. NaN
  // compares false against everything, so without this check a NaN block would
  // silently lose every pick; +inf would always win and hide the same fault.
  for (int block = tid; block < blocks; block += blockDim.x) {
    if (!isfinite(row_scores[block])) {
      __trap();
    }
  }

  const int keep_cap = budget / kRatio;
  int keep = keep_cap < blocks ? keep_cap : blocks;
  if (variant == kKeepFewer && keep > 0) {
    --keep;
  } else if (variant == kKeepMore && keep < blocks) {
    ++keep;
  }
  const bool invert = variant == kInvertOrder;

  // The bitmap covers every block any row of this launch can see.
  const int max_blocks = tokens / kRatio;
  const int words = (max_blocks + 31) >> 5;
  for (int word = tid; word < words; word += blockDim.x) {
    picked[word] = 0u;
  }
  __syncthreads();

  __shared__ float warp_score[kWarps];
  __shared__ int warp_index[kWarps];
  for (int pick = 0; pick < keep; ++pick) {
    // Each thread's best among the candidates it owns; the scan is in ascending
    // block order, and the explicit tie rule keeps the result independent of
    // which thread found a tied score first.
    float best_score = 0.0f;
    int best_index = -1;
    for (int block = tid; block < blocks; block += blockDim.x) {
      if (block_picked(picked, block)) {
        continue;
      }
      const float score = row_scores[block];
      if (outranks(score, block, best_score, best_index, invert)) {
        best_score = score;
        best_index = block;
      }
    }
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      const float other_score = __shfl_down_sync(0xffffffffu, best_score, offset);
      const int other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
      if (outranks(other_score, other_index, best_score, best_index, invert)) {
        best_score = other_score;
        best_index = other_index;
      }
    }
    const int lane = tid & (WARP_SIZE - 1);
    if (lane == 0) {
      warp_score[tid / WARP_SIZE] = best_score;
      warp_index[tid / WARP_SIZE] = best_index;
    }
    __syncthreads();
    if (tid == 0) {
      float pick_score = warp_score[0];
      int pick_index = warp_index[0];
#pragma unroll
      for (int warp = 1; warp < kWarps; ++warp) {
        if (outranks(warp_score[warp], warp_index[warp], pick_score, pick_index,
                     invert)) {
          pick_score = warp_score[warp];
          pick_index = warp_index[warp];
        }
      }
      // `keep <= blocks` and every pass marks one block, so at least one
      // unpicked candidate remains; an empty pick means the launch is broken.
      if (pick_index < 0) {
        __trap();
      }
      mark_picked(picked, pick_index);
    }
    __syncthreads();
  }

  unsigned char* row_mask = mask + static_cast<int64_t>(row) * tokens;
  for (int token = tid; token < tokens; token += blockDim.x) {
    row_mask[token] = 0;
  }
  __syncthreads();
  for (int block = tid; block < blocks; block += blockDim.x) {
    if (!block_picked(picked, block)) {
      continue;
    }
    const int base = block * kRatio;
#pragma unroll
    for (int i = 0; i < kRatio; ++i) {
      row_mask[base + i] = 1;
    }
  }
  if (variant != kDropTail) {
    for (int i = tid; i < tail; i += blockDim.x) {
      row_mask[blocks * kRatio + i] = 1;
    }
  }
  if (variant == kIgnoreCausality) {
    // Gate-only: pretend the row's own token is not the end of the visible set.
    for (int token = visible + tid; token < tokens; token += blockDim.x) {
      row_mask[token] = 1;
    }
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
// is at most kMaxKeptBlocks. `variant` is 0 in production; the other values are
// the gate's controls (see above).
//
// The compiled geometry is kRatio = 4 and one byte per mask cell; a caller
// whose buffers are laid out differently cannot be detected here. An empty
// launch (no rows, or no tokens) is refused rather than treated as a no-op: a
// caller asking for nothing has a scheduling bug, and returning success would
// hide it behind an all-zero mask.
CUresult qwen4_qsa_indexer_select_cuda(
    const float* scores,
    const int* rows,
    unsigned char* mask,
    int query_rows,
    int tokens,
    int score_cols,
    int budget,
    int variant,
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
  if (variant < kProduction || variant > kInvertOrder) {
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
      scores, rows, mask, tokens, score_cols, budget, variant);
  return consume_last_cuda_error();
}

}  // extern "C"
