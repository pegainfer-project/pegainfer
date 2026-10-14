// Qwen3.8-Flash-Next QSA token-indexer block scoring (Stage 1, plan P2).
//
// Frozen semantics: transformers 5.16.1 `qwen4_exp`,
// `Qwen4ExpTextQSAIndexer.forward`. For a query row whose position is `p`,
// over a visible set that starts at token 0:
//
//   qk           = index_qk_proj(hidden)   // one fused Linear(2560 -> 640)
//   q_h          = (1 + q_layernorm_w) RMSNorm128(qk_q_h), partial RoPE at p
//   pooled_key_b = mean(keys[4b .. 4b+4)) in float32, cast back to bf16,
//                  then (1 + k_layernorm_w) RMSNorm128, RoPE at 4b
//   score_b      = sum_{h < 4} relu(q_h . pooled_key_b) / sqrt(128)
//
// The four query heads share one key head (indexer_kv_heads = 1), and the relu
// is applied per head before the sum over heads.
//
// Two kernels own one half each. `pool_keys` builds every complete micro-block's
// pooled key once: the reference rebuilds them per query row, from the same raw
// keys through the same bf16 rounding points, so one pass is the same numbers —
// a block's contents do not depend on the row that scores it. `block_scores`
// does each row's query prep and the score reduction, over the blocks that row
// can see. The incomplete tail (< compress_ratio tokens) is never scored: the
// reference keeps it unconditionally, outside the block grid, and selection is
// a separate stage.
//
// `compress_ratio = 4` divides the 16-token KV page, so a micro-block never
// crosses a page. This kernel does not need that (it reads the raw key array);
// it is what lets the selection stage map blocks onto pages with no mapping
// layer.
//
// GRID CONTRACT: the reference groups the visible list from its FIRST visible
// token, not from absolute position 0. This kernel requires that first visible
// token to be token 0 of the key array (visible = [0, visible_len[row])), which
// is the causal-text case with no prefix eviction — so block b always covers
// tokens [4b, 4b+4). Nothing here can detect a caller whose visible list is
// anchored elsewhere: it would pass positions and counts that look in range and
// get a shifted grid, and such a caller must hand this kernel a shifted key
// array instead.
//
// Target: the sm_89-first portable subset (decision D3-B) — no TMA, no cluster,
// no wgmma. These kernels are SIMT shuffle reductions rather than `mma.sync`
// bodies: one block's score is a 128-wide dot over 4 query heads, a shape
// m16n8k16 cannot tile without padding 4 rows to 16.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHeads = 4;      // indexer_n_heads
constexpr int kKvHeads = 1;    // indexer_kv_heads
constexpr int kHeadDim = 128;  // indexer_head_dim
constexpr int kRatio = 4;      // indexer_compress_ratio
constexpr float kRopeTheta = 1.0e7f;
// head_dim(256) * partial_rotary_factor(0.25): the rope covers the first half
// of a query/key head and leaves the rest alone.
constexpr int kRotaryDim = 64;
constexpr int kQWidth = kHeads * kHeadDim;
constexpr int kFusedWidth = (kHeads + kKvHeads) * kHeadDim;
// One lane owns dims {lane, lane+32, lane+64, lane+96}, so the rope pair
// (d, d + rotary_dim/2) is lane-local for d < 32.
constexpr int kDimPerLane = kHeadDim / WARP_SIZE;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / WARP_SIZE;

static_assert(kRotaryDim / 2 == WARP_SIZE, "a rope pair must be one lane's dims 0 and 1");
static_assert(kDimPerLane == kHeadDim / WARP_SIZE);

// Sum across all 32 lanes, leaving it in every lane (common.cuh's
// `warp_reduce_sum` leaves it in lane 0, which the norm needs broadcast).
__device__ __forceinline__ float warp_all_sum(float value) {
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, mask);
  }
  return value;
}

// One `(1 + w)` RMSNorm element, rounded to bf16 exactly once: the reference
// normalizes in float32, scales by `1 + w` in float32, and then casts to the
// activation dtype.
__device__ __forceinline__ float indexer_norm_elem(float x, float inv_rms,
                                                  float weight) {
  return __bfloat162float(__float2bfloat16(x * inv_rms * (1.0f + weight)));
}

// `1 / (theta ** (arange(0, rotary_dim, 2) / rotary_dim))` in float32, for the
// pair whose low dim is `pair`.
__device__ __forceinline__ float indexer_inv_freq(int pair) {
  return 1.0f / powf(kRopeTheta, static_cast<float>(2 * pair) /
                                     static_cast<float>(kRotaryDim));
}

// Partial RoPE at `pos` over the pair (d, d + rotary_dim/2), for d < 32. The
// reference casts cos/sin to the activation dtype and rotates in bf16, so each
// product and their sum round as well.
__device__ __forceinline__ void indexer_rope_pair(float& lo, float& hi, int pos,
                                                  float inv_freq) {
  const float angle = static_cast<float>(pos) * inv_freq;
  const float cos_v = __bfloat162float(__float2bfloat16(cosf(angle)));
  const float sin_v = __bfloat162float(__float2bfloat16(sinf(angle)));
  const float lo_rot = __bfloat162float(__float2bfloat16(lo * cos_v)) +
                       __bfloat162float(__float2bfloat16(-hi * sin_v));
  const float hi_rot = __bfloat162float(__float2bfloat16(hi * cos_v)) +
                       __bfloat162float(__float2bfloat16(lo * sin_v));
  lo = __bfloat162float(__float2bfloat16(lo_rot));
  hi = __bfloat162float(__float2bfloat16(hi_rot));
}

// One pooled key per complete micro-block: the float32 mean of the block's raw
// keys, cast back to bf16 the way the reference casts it, then `(1 + w)` RMSNorm
// and RoPE at the block's first token. One warp per block, four blocks per CTA.
__global__ void qwen38fn_qsa_pool_keys_kernel(
    const __nv_bfloat16* __restrict__ fused_qk,         // [tokens, kFusedWidth]
    const __nv_bfloat16* __restrict__ k_norm_weight,    // [kHeadDim]
    __nv_bfloat16* __restrict__ pooled_keys,            // [blocks, kHeadDim]
    int blocks,
    float rms_eps) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  const int block = blockIdx.x * kWarps + warp;
  if (block >= blocks) {
    return;
  }
  const int64_t first = static_cast<int64_t>(block) * kRatio;

  float mean[kDimPerLane];
  float squares = 0.0f;
#pragma unroll
  for (int i = 0; i < kDimPerLane; ++i) {
    const int d = lane + i * WARP_SIZE;
    float sum = 0.0f;
#pragma unroll
    for (int t = 0; t < kRatio; ++t) {
      sum += __bfloat162float(
          fused_qk[(first + t) * kFusedWidth + kQWidth + d]);
    }
    // The reference's float32 mean is cast back to the activation dtype before
    // the norm runs, so the norm squares the rounded value.
    mean[i] = __bfloat162float(__float2bfloat16(sum / static_cast<float>(kRatio)));
    squares += mean[i] * mean[i];
  }
  const float inv_rms =
      rsqrtf(warp_all_sum(squares) / static_cast<float>(kHeadDim) + rms_eps);

  float pooled[kDimPerLane];
#pragma unroll
  for (int i = 0; i < kDimPerLane; ++i) {
    const int d = lane + i * WARP_SIZE;
    pooled[i] =
        indexer_norm_elem(mean[i], inv_rms, __bfloat162float(k_norm_weight[d]));
  }
  if (lane < kRotaryDim / 2) {
    indexer_rope_pair(pooled[0], pooled[1], static_cast<int>(first),
                      indexer_inv_freq(lane));
  }
#pragma unroll
  for (int i = 0; i < kDimPerLane; ++i) {
    pooled_keys[static_cast<int64_t>(block) * kHeadDim + lane + i * WARP_SIZE] =
        __float2bfloat16(pooled[i]);
  }
}

// Per query row: prepare the row's four query heads, then score every block the
// row can see. One CTA per row, the four warps split the row's blocks.
__global__ void qwen38fn_qsa_block_scores_kernel(
    const __nv_bfloat16* __restrict__ fused_qk,         // [tokens, kFusedWidth]
    const __nv_bfloat16* __restrict__ q_norm_weight,    // [kHeadDim]
    const __nv_bfloat16* __restrict__ pooled_keys,      // [blocks, kHeadDim]
    const int* __restrict__ rows,                       // [query_rows]
    const int* __restrict__ visible_len,                // [query_rows]
    float* __restrict__ scores,                         // [query_rows, score_cols]
    int tokens,
    int score_cols,
    int blocks,
    float rms_eps) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;

  // `rows[row]` is the row's own position, and the row of `fused_qk` its query
  // heads are read from. Contract: 0 <= rows[row] < tokens — clamped, like
  // `visible_len` below, so a broken caller gets a wrong position rather than an
  // out-of-bounds read.
  const int raw_pos = rows[row];
  const int pos = raw_pos < 0 ? 0 : (raw_pos >= tokens ? tokens - 1 : raw_pos);
  // A visible list longer than the key array, or longer than the block grid, is
  // clamped rather than read past: the adapter's contract is that
  // visible_len[row] <= tokens, and the clamp is what keeps a broken caller from
  // touching memory past the pool.
  int visible = visible_len[row];
  if (visible < 0) {
    visible = 0;
  } else if (visible > tokens) {
    visible = tokens;
  }
  int row_blocks = visible / kRatio;
  if (row_blocks > blocks) {
    row_blocks = blocks;
  }
  if (row_blocks <= 0) {
    return;
  }

  const int64_t q_base = static_cast<int64_t>(pos) * kFusedWidth;
  float q[kHeads][kDimPerLane];
  float squares[kHeads];
#pragma unroll
  for (int h = 0; h < kHeads; ++h) {
    squares[h] = 0.0f;
#pragma unroll
    for (int i = 0; i < kDimPerLane; ++i) {
      const float value =
          __bfloat162float(fused_qk[q_base + h * kHeadDim + lane + i * WARP_SIZE]);
      q[h][i] = value;
      squares[h] += value * value;
    }
  }
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
      squares[h] += __shfl_xor_sync(0xffffffffu, squares[h], mask);
    }
  }
#pragma unroll
  for (int h = 0; h < kHeads; ++h) {
    const float inv_rms =
        rsqrtf(squares[h] / static_cast<float>(kHeadDim) + rms_eps);
#pragma unroll
    for (int i = 0; i < kDimPerLane; ++i) {
      const int d = lane + i * WARP_SIZE;
      q[h][i] =
          indexer_norm_elem(q[h][i], inv_rms, __bfloat162float(q_norm_weight[d]));
    }
  }
  if (lane < kRotaryDim / 2) {
    const float inv_freq = indexer_inv_freq(lane);
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
      indexer_rope_pair(q[h][0], q[h][1], pos, inv_freq);
    }
  }

  for (int block = warp; block < row_blocks; block += kWarps) {
    float acc[kHeads];
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
      acc[h] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < kDimPerLane; ++i) {
      const float key_value = __bfloat162float(
          pooled_keys[static_cast<int64_t>(block) * kHeadDim + lane + i * WARP_SIZE]);
#pragma unroll
      for (int h = 0; h < kHeads; ++h) {
        acc[h] += q[h][i] * key_value;
      }
    }
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
#pragma unroll
      for (int h = 0; h < kHeads; ++h) {
        acc[h] += __shfl_xor_sync(0xffffffffu, acc[h], mask);
      }
    }
    if (lane == 0) {
      // relu per head, then the sum over heads, then the 1/sqrt(head_dim)
      // scale: all three are the reference's own order.
      const float relu_sum = fmaxf(acc[0], 0.0f) + fmaxf(acc[1], 0.0f) +
                             fmaxf(acc[2], 0.0f) + fmaxf(acc[3], 0.0f);
      scores[static_cast<int64_t>(row) * score_cols + block] =
          relu_sum / sqrtf(static_cast<float>(kHeadDim));
    }
  }
}

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

}  // namespace

extern "C" {

// Pooled indexer keys for the first `tokens / kRatio` complete micro-blocks of
// a key array that starts at token 0, then the per-row block scores. The
// incomplete tail is not pooled and not scored.
//
// Per row, writes `visible_len[row] / kRatio` entries of `scores` (row stride
// `score_cols`) and leaves the rest untouched; the caller's block grid, budget
// and selection are the next stage's business.
//
// `pooled_keys` is the pool pass's workspace — one head-dim row per complete
// block, `tokens / kRatio` rows. It is one entry point because the score pass
// reads all of it and every row shares it: pooling it once per call is the
// whole reason it is not inside the per-row kernel.
//
// `fused_width` is the caller's own row stride for `fused_qk`, checked here:
// this body reads with the width it was compiled for, so a caller whose rows
// are strided differently would be read out of bounds rather than computed
// wrongly, which is the one mismatch the pointer arithmetic cannot survive.
CUresult qwen38fn_qsa_indexer_block_scores_cuda(
    const __nv_bfloat16* fused_qk,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const int* rows,
    const int* visible_len,
    float* scores,
    __nv_bfloat16* pooled_keys,
    int query_rows,
    int tokens,
    int score_cols,
    int fused_width,
    int rotary_dim,
    float rope_theta,
    float rms_eps,
    cudaStream_t stream) {
  if (fused_qk == nullptr || q_norm_weight == nullptr ||
      k_norm_weight == nullptr || rows == nullptr || visible_len == nullptr ||
      scores == nullptr || pooled_keys == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (query_rows < 0 || tokens < 0 || score_cols < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (fused_width != kFusedWidth || rotary_dim != kRotaryDim ||
      rope_theta != kRopeTheta) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!(rms_eps >= 0.0f)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (query_rows == 0 || tokens == 0) {
    return CUDA_SUCCESS;
  }
  const int blocks = tokens / kRatio;
  // The score pass writes one entry per visible block into a `score_cols`-wide
  // row, so a stride under the block grid is an out-of-bounds write.
  if (blocks > score_cols) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (blocks > 0) {
    const int grid = (blocks + kWarps - 1) / kWarps;
    qwen38fn_qsa_pool_keys_kernel<<<grid, kThreads, 0, stream>>>(
        fused_qk, k_norm_weight, pooled_keys, blocks, rms_eps);
    const CUresult pooled_status = consume_last_cuda_error();
    if (pooled_status != CUDA_SUCCESS) {
      return pooled_status;
    }
  }
  qwen38fn_qsa_block_scores_kernel<<<query_rows, kThreads, 0, stream>>>(
      fused_qk, q_norm_weight, pooled_keys, rows, visible_len, scores, tokens,
      score_cols, blocks, rms_eps);
  return consume_last_cuda_error();
}

}  // extern "C"
