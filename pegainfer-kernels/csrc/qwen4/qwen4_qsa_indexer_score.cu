// Qwen3.8-Flash-Next (Qwen4 line) QSA token-indexer block scoring.
//
// Frozen semantics: transformers 5.16.1 `qwen4_exp`,
// `Qwen4ExpTextQSAIndexer.forward`. For a query row at position `p`, over a
// visible set that starts at token 0:
//
//   qk           = index_qk_proj(hidden)   // one fused Linear(2560 -> 640)
//   q_h          = (1 + q_layernorm_w) RMSNorm128(qk_q_h), partial RoPE at p
//   pooled_key_b = mean(keys[4b .. 4b+4)) in float32, cast back to bf16,
//                  then (1 + k_layernorm_w) RMSNorm128, RoPE at 4b
//   score_b      = sum_{h < 4} relu(q_h . pooled_key_b) / sqrt(128)
//
// The partial RoPE reads the attention line's cos/sin table (the indexer shares
// the position embeddings), so this file computes no trig: the table row is
// already bf16-rounded like the reference's cast. The kernel does not own the
// table; the caller builds it once and passes it with every call.
//
// Two kernels own one half each: `pool_keys` builds every complete
// micro-block's pooled key once — a block's contents do not depend on the row
// that scores it — and `block_scores` does each row's query prep and score
// reduction over the blocks that row can see. The incomplete tail (< ratio
// tokens) is never scored: the reference keeps it unconditionally, outside the
// block grid.
//
// GRID CONTRACT: the reference groups the visible list from its FIRST visible
// token. This kernel requires that token to be token 0 of the key array
// (causal text over an un-evicted cache), so block b always covers tokens
// [4b, 4b+4). Nothing here can detect a caller whose visible list starts
// elsewhere; such a caller must hand this kernel a shifted key array.
//
// Portable SIMT — no TMA, no cluster, no wgmma; shuffle reductions rather than
// `mma.sync`, because one block's score is a 128-wide dot over 4 query heads,
// a shape m16n8k16 cannot tile without padding 4 rows to 16.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHeads = 4;      // indexer_n_heads
constexpr int kKvHeads = 1;    // indexer_kv_heads
constexpr int kHeadDim = 128;  // indexer_head_dim
constexpr int kRatio = 4;      // indexer_compress_ratio
// head_dim(256) * partial_rotary_factor(0.25): the rope covers the first half
// of a query/key head and leaves the rest alone. It is also the stride of the
// shared attention cos/sin table this kernel reads.
constexpr int kRotaryDim = 64;
constexpr int kQWidth = kHeads * kHeadDim;
constexpr int kFusedWidth = (kHeads + kKvHeads) * kHeadDim;
// One lane owns dims {lane, lane+32, lane+64, lane+96}, so the rope pair
// (d, d + rotary_dim/2) is lane-local for d < 32.
constexpr int kDimPerLane = kHeadDim / WARP_SIZE;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / WARP_SIZE;

static_assert(kRotaryDim / 2 == WARP_SIZE, "a rope pair must be one lane's dims 0 and 1");
static_assert(kHeadDim % WARP_SIZE == 0, "a head must split into whole lanes");

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

// Partial RoPE over the pair (d, d + rotary_dim/2), for d < 32, at the shared
// attention table's row for the position. The table is bf16 (built by
// `pegainfer_core::rope`, already rounded like the reference), and the
// reference rotates in bf16, so each product and their sum round as well.
__device__ __forceinline__ void indexer_rope_pair(float& lo, float& hi,
                                                  float cos_v, float sin_v) {
  const float lo_rot = __bfloat162float(__float2bfloat16(lo * cos_v)) +
                       __bfloat162float(__float2bfloat16(-hi * sin_v));
  const float hi_rot = __bfloat162float(__float2bfloat16(hi * cos_v)) +
                       __bfloat162float(__float2bfloat16(lo * sin_v));
  lo = __bfloat162float(__float2bfloat16(lo_rot));
  hi = __bfloat162float(__float2bfloat16(hi_rot));
}

// The table pair for dimension `dim` at `pos`; the table stores
// `[max_pos, kRotaryDim]` with the half split duplicated, so dim and
// dim + rotary_dim/2 hold the same cos/sin.
__device__ __forceinline__ float indexer_table_at(
    const __nv_bfloat16* __restrict__ table, int64_t pos, int dim) {
  return __bfloat162float(table[pos * kRotaryDim + dim]);
}

// One pooled key per complete micro-block: the float32 mean of the block's raw
// keys, cast back to bf16 the way the reference casts it, then `(1 + w)` RMSNorm
// and RoPE at the block's first token. One warp per block, four blocks per CTA.
__global__ void qwen4_qsa_pool_keys_kernel(
    const __nv_bfloat16* __restrict__ fused_qk,         // [tokens, kFusedWidth]
    const __nv_bfloat16* __restrict__ k_norm_weight,    // [kHeadDim]
    const __nv_bfloat16* __restrict__ cos_table,        // [max_pos, kRotaryDim]
    const __nv_bfloat16* __restrict__ sin_table,        // [max_pos, kRotaryDim]
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
    indexer_rope_pair(pooled[0], pooled[1],
                      indexer_table_at(cos_table, first, lane),
                      indexer_table_at(sin_table, first, lane));
  }
#pragma unroll
  for (int i = 0; i < kDimPerLane; ++i) {
    pooled_keys[static_cast<int64_t>(block) * kHeadDim + lane + i * WARP_SIZE] =
        __float2bfloat16(pooled[i]);
  }
}

// Per query row: prepare the row's four query heads, then score every block the
// row can see. One CTA per row, the four warps split the row's blocks.
__global__ void qwen4_qsa_block_scores_kernel(
    const __nv_bfloat16* __restrict__ fused_qk,         // [tokens, kFusedWidth]
    const __nv_bfloat16* __restrict__ q_norm_weight,    // [kHeadDim]
    const __nv_bfloat16* __restrict__ pooled_keys,      // [blocks, kHeadDim]
    const __nv_bfloat16* __restrict__ cos_table,        // [max_pos, kRotaryDim]
    const __nv_bfloat16* __restrict__ sin_table,        // [max_pos, kRotaryDim]
    const int* __restrict__ rows,                       // [query_rows]
    const int* __restrict__ visible_len,                // [query_rows]
    float* __restrict__ scores,                         // [query_rows, score_cols]
    int tokens,
    int score_cols,
    float rms_eps) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;

  // `rows[row]` is the row's own position and the fused_qk row its query
  // heads are read from; `visible_len[row]` is how many tokens it sees, which
  // causal text caps at the row's own token. Both are caller-provided device
  // values and are checked here: an out-of-range entry traps, because clamping
  // it, or scoring blocks past the row, would hand the caller a plausible score
  // it never asked for. With `pos < tokens`, `pos + 1` also bounds the array.
  const int pos = rows[row];
  if (pos < 0 || pos >= tokens) {
    __trap();
  }
  const int visible = visible_len[row];
  if (visible < 0 || visible > pos + 1) {
    __trap();
  }
  const int row_blocks = visible / kRatio;
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
    const float cos_v = indexer_table_at(cos_table, pos, lane);
    const float sin_v = indexer_table_at(sin_table, pos, lane);
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
      indexer_rope_pair(q[h][0], q[h][1], cos_v, sin_v);
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
      // scale: all three are the reference's own order. Written as a
      // comparison rather than `fmaxf`, which returns the other operand when
      // one is NaN — a poisoned score must stay NaN like `torch.relu`, not
      // turn into a valid-looking zero.
      const float relu_sum = (acc[0] < 0.0f ? 0.0f : acc[0]) +
                             (acc[1] < 0.0f ? 0.0f : acc[1]) +
                             (acc[2] < 0.0f ? 0.0f : acc[2]) +
                             (acc[3] < 0.0f ? 0.0f : acc[3]);
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
// incomplete tail is not pooled and not scored. Per row, writes
// `visible_len[row] / kRatio` entries of `scores` (row stride `score_cols`)
// and leaves the rest untouched.
//
// `pooled_keys` is the pool pass's workspace — one head-dim row per complete
// block — shared by every row, which is why pooling is its own pass.
//
// The row strides are compiled in: `kFusedWidth` for `fused_qk` and
// `kRotaryDim` for the shared attention cos/sin tables
// (`pegainfer_core::rope`), so this kernel does no trig of its own. A caller
// whose buffers are laid out differently cannot be detected here; the model
// crate checks its config where it reads it.
CUresult qwen4_qsa_indexer_block_scores_cuda(
    const __nv_bfloat16* fused_qk,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_table,
    const __nv_bfloat16* sin_table,
    const int* rows,
    const int* visible_len,
    float* scores,
    __nv_bfloat16* pooled_keys,
    int query_rows,
    int tokens,
    int score_cols,
    float rms_eps,
    cudaStream_t stream) {
  if (fused_qk == nullptr || q_norm_weight == nullptr ||
      k_norm_weight == nullptr || cos_table == nullptr || sin_table == nullptr ||
      rows == nullptr || visible_len == nullptr || scores == nullptr ||
      pooled_keys == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (query_rows < 0 || tokens < 0 || score_cols < 0) {
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
    qwen4_qsa_pool_keys_kernel<<<grid, kThreads, 0, stream>>>(
        fused_qk, k_norm_weight, cos_table, sin_table, pooled_keys, blocks,
        rms_eps);
    const CUresult pooled_status = consume_last_cuda_error();
    if (pooled_status != CUDA_SUCCESS) {
      return pooled_status;
    }
  }
  qwen4_qsa_block_scores_kernel<<<query_rows, kThreads, 0, stream>>>(
      fused_qk, q_norm_weight, pooled_keys, cos_table, sin_table, rows,
      visible_len, scores, tokens, score_cols, rms_eps);
  return consume_last_cuda_error();
}

}  // extern "C"
