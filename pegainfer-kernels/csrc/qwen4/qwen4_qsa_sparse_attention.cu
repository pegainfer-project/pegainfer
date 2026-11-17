// Qwen3.8-Flash-Next (Qwen4 line) QSA sparse attention over the paged K/V.
//
// Frozen semantics: transformers 5.16.1 `qwen4_exp`, `Qwen4ExpTextAttention`
// once `Qwen4ExpTextQSAIndexer` has chosen a row's tokens. For a query row at
// position `p`, over the tokens its mask keeps (`mask[row][t] = 1`, `t <= p`):
//
//   out_h = sum_t softmax_t(q_h . k_t / sqrt(256)) v_t
//
// for every query head `h`, its K/V head being `h / 12`, then the output gate
// `out_h * sigmoid(gate_h)`. `q` is the prepped query (`(1 + w)` q norm and
// partial RoPE applied, the rows the dense entry writes), the gate is the second
// half of each head in the `q_proj` output, and K/V are read from the paged pool
// after that entry has written the row's own token. Scores, the running max and
// the sums stay f32 and the attention is rounded to bf16 once, as the dense
// FlashInfer path does; the reference rounds its scores and its weights to bf16
// on the way, which the accuracy gate's floor absorbs on the dense rows
// already. The gate repeats the dense path's gate kernel operation for
// operation (`attention_gate_batch_hd256_kernel`), so a row gets the same gate
// whichever path computed its attention.
//
// A row keeps at most the budget plus its incomplete tail, kMaxKept = 2048 + 3
// tokens, which is what lets a CTA hold the row's kept positions in shared
// memory; a row that keeps more is an upstream contract violation and traps.
//
// POOL CONTRACT: one layer's split K/V pool in NHD pages of kPageSize tokens
// (the K block of a page, then its V block), kKvHeads heads at kHeadDim, and a
// sequence's pages contiguous from `first_page`, which is how the model line
// lays a slot out.
//
// One CTA per (row, K/V head), one warp per query head of the group. The CTA
// first gathers the row's kept positions in ascending order, then walks them in
// tiles of kTile tokens: every K and V row of a tile goes to shared memory once
// and all twelve warps read it there. Portable SIMT: no TMA, no cluster, no
// wgmma.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"
#include "qwen4_qsa_norm_rope.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kHeads = 24;     // num_attention_heads
constexpr int kKvHeads = 2;    // num_key_value_heads
constexpr int kGroup = kHeads / kKvHeads;
constexpr int kHeadDim = 256;  // head_dim
constexpr int kPageSize = 16;  // the line's K/V page
constexpr int kMaxKept = 2048 + 4 - 1;  // indexer budget plus the incomplete tail
constexpr int kTile = 32;
constexpr int kThreads = kGroup * WARP_SIZE;
constexpr int kWarps = kThreads / WARP_SIZE;
// One lane owns eight consecutive dims, one 16-byte vector.
constexpr int kDimPerLane = kHeadDim / WARP_SIZE;
constexpr int kVectorsPerRow = kHeadDim / kDimPerLane;

static_assert(kDimPerLane == 8, "a lane's dims must be one 16-byte vector");
static_assert(kVectorsPerRow == WARP_SIZE, "a head row must be one vector per lane");

__device__ __forceinline__ void load_row(const __nv_bfloat16* source, float (&dest)[kDimPerLane]) {
  const uint4 packed = *reinterpret_cast<const uint4*>(source);
  const __nv_bfloat16* values = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
  for (int i = 0; i < kDimPerLane; ++i) {
    dest[i] = __bfloat162float(values[i]);
  }
}

__global__ void __launch_bounds__(kThreads) qwen4_qsa_sparse_attention_kernel(
    const __nv_bfloat16* __restrict__ q_full,     // [rows, kHeads * 2 * kHeadDim]
    const __nv_bfloat16* __restrict__ q_prepped,  // [rows, kHeads * kHeadDim]
    const __nv_bfloat16* __restrict__ pool,       // the layer's paged K/V
    const unsigned char* __restrict__ mask,       // [rows, mask_stride]
    __nv_bfloat16* __restrict__ out,              // [rows, kHeads * kHeadDim]
    int first_position,
    int first_page,
    int64_t page_stride,
    int64_t kv_block_len,
    int mask_stride) {
  __shared__ int s_kept_at[kMaxKept];
  __shared__ int s_warp_count[kWarps];
  __shared__ int s_warp_offset[kWarps];
  __shared__ int s_kept;
  __shared__ __align__(16) __nv_bfloat16 s_k[kTile][kHeadDim];
  __shared__ __align__(16) __nv_bfloat16 s_v[kTile][kHeadDim];

  const int row = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid % WARP_SIZE;
  const int warp = tid / WARP_SIZE;
  const int position = first_position + row;
  const unsigned char* mask_row = mask + static_cast<int64_t>(row) * mask_stride;

  // The row's kept positions, ascending: each pass covers kThreads positions,
  // a ballot orders a warp's picks and one thread orders the warps.
  if (tid == 0) {
    s_kept = 0;
  }
  __syncthreads();
  for (int base = 0; base <= position; base += kThreads) {
    const int t = base + tid;
    const bool keep = t <= position && mask_row[t] != 0;
    const unsigned ballot = __ballot_sync(0xffffffffu, keep);
    if (lane == 0) {
      s_warp_count[warp] = __popc(ballot);
    }
    __syncthreads();
    if (tid == 0) {
      int running = s_kept;
      for (int w = 0; w < kWarps; ++w) {
        s_warp_offset[w] = running;
        running += s_warp_count[w];
      }
      if (running > kMaxKept) {
        __trap();
      }
      s_kept = running;
    }
    __syncthreads();
    if (keep) {
      s_kept_at[s_warp_offset[warp] + __popc(ballot & ((1u << lane) - 1u))] = t;
    }
  }
  __syncthreads();
  const int kept = s_kept;

  const int head = kv_head * kGroup + warp;
  const int64_t row_offset = static_cast<int64_t>(row) * kHeads * kHeadDim +
                             static_cast<int64_t>(head) * kHeadDim + lane * kDimPerLane;
  float q[kDimPerLane];
  load_row(q_prepped + row_offset, q);
  const float scale = rsqrtf(static_cast<float>(kHeadDim));

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc[kDimPerLane];
#pragma unroll
  for (int i = 0; i < kDimPerLane; ++i) {
    acc[i] = 0.0f;
  }

  for (int start = 0; start < kept; start += kTile) {
    const int count = min(kTile, kept - start);
    for (int e = tid; e < count * kVectorsPerRow; e += kThreads) {
      const int i = e / kVectorsPerRow;
      const int vector = e % kVectorsPerRow;
      const int t = s_kept_at[start + i];
      const int64_t page = static_cast<int64_t>(first_page) + t / kPageSize;
      const int64_t k_at = page * page_stride +
                           static_cast<int64_t>(t % kPageSize) * kKvHeads * kHeadDim +
                           static_cast<int64_t>(kv_head) * kHeadDim + vector * kDimPerLane;
      *reinterpret_cast<uint4*>(&s_k[i][vector * kDimPerLane]) =
          *reinterpret_cast<const uint4*>(pool + k_at);
      *reinterpret_cast<uint4*>(&s_v[i][vector * kDimPerLane]) =
          *reinterpret_cast<const uint4*>(pool + k_at + kv_block_len);
    }
    __syncthreads();
    for (int i = 0; i < count; ++i) {
      float k[kDimPerLane];
      load_row(&s_k[i][lane * kDimPerLane], k);
      float dot = 0.0f;
#pragma unroll
      for (int d = 0; d < kDimPerLane; ++d) {
        dot += q[d] * k[d];
      }
      const float score = qsa_warp_all_sum(dot) * scale;
      const float next_max = fmaxf(running_max, score);
      const float rescale = expf(running_max - next_max);
      const float weight = expf(score - next_max);
      running_sum = running_sum * rescale + weight;
      float v[kDimPerLane];
      load_row(&s_v[i][lane * kDimPerLane], v);
#pragma unroll
      for (int d = 0; d < kDimPerLane; ++d) {
        acc[d] = acc[d] * rescale + weight * v[d];
      }
      running_max = next_max;
    }
    __syncthreads();
  }

  float gate[kDimPerLane];
  load_row(q_full + static_cast<int64_t>(row) * kHeads * 2 * kHeadDim +
               static_cast<int64_t>(head) * 2 * kHeadDim + kHeadDim + lane * kDimPerLane,
           gate);
  uint4 packed;
  __nv_bfloat16* values = reinterpret_cast<__nv_bfloat16*>(&packed);
  const float inv_sum = 1.0f / running_sum;
#pragma unroll
  for (int d = 0; d < kDimPerLane; ++d) {
    const float attention = __bfloat162float(__float2bfloat16(acc[d] * inv_sum));
    const float sig_gate = 1.0f / (1.0f + expf(-gate[d]));
    values[d] = __float2bfloat16(attention * sig_gate);
  }
  *reinterpret_cast<uint4*>(out + row_offset) = packed;
}

}  // namespace

extern "C" {

// Gated QSA attention for `rows` consecutive query rows of one sequence, the
// first at `first_position`: `q_full` is `[rows, 24 * 2 * 256]` (each head's
// query, then its gate), `q_prepped` and `out` are `[rows, 24 * 256]`, all from
// their first row, `mask` is `[rows, mask_stride]` bytes, and the sequence's pages
// start at `first_page` of a pool whose pages are `page_stride` elements apart
// with the V block `kv_block_len` elements after the K block. Every row's
// positions must lie inside `mask_stride`. The compiled geometry is the
// checkpoint's: 24 query and 2 K/V heads at head dim 256, pages of 16 tokens;
// a caller laid out differently cannot be detected here.
CUresult qwen4_qsa_sparse_attention_cuda(
    const __nv_bfloat16* q_full,
    const __nv_bfloat16* q_prepped,
    const __nv_bfloat16* pool,
    const unsigned char* mask,
    __nv_bfloat16* out,
    int rows,
    int first_position,
    int first_page,
    int64_t page_stride,
    int64_t kv_block_len,
    int mask_stride,
    cudaStream_t stream) {
  if (q_full == nullptr || q_prepped == nullptr || pool == nullptr || mask == nullptr ||
      out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rows <= 0 || first_position < 0 || first_page < 0 || page_stride <= 0 ||
      kv_block_len <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // The last row reads mask cells up to its own position.
  if (static_cast<int64_t>(first_position) + rows > mask_stride) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const dim3 grid(rows, kKvHeads);
  qwen4_qsa_sparse_attention_kernel<<<grid, kThreads, 0, stream>>>(
      q_full, q_prepped, pool, mask, out, first_position, first_page, page_stride,
      kv_block_len, mask_stride);
  return consume_last_cuda_error();
}

}  // extern "C"
