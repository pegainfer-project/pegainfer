// Kimi-K3 absorbed-MLA decode over the paged latent KV cache.
//
// One kernel replaces the expanded-cache chain (kv_b expansion + slot-indexed
// attention): per (row, head) block it absorbs the query into latent space,
// walks the row's block table page by page, and expands the attended latent
// back to a value head — so the cache holds 576 bf16 per token per layer (the
// post-norm kv latent | the shared rope half) instead of the 96-head expanded
// K/V. NoPE: nothing here is position-dependent, which is what makes the
// latent row cacheable at all.
//
// ---------------------------------------------------------------------------
// Math (standard MLA absorption)
// ---------------------------------------------------------------------------
// The expanded path scores head h at position t as
//     score = q_nope[h] . (W_UK[h] c_t)  +  q_rope[h] . rope_t
// which regroups to
//     score = (W_UK[h]^T q_nope[h]) . c_t  +  q_rope[h] . rope_t
// so the absorbed query q_abs[h] = [W_UK[h]^T q_nope[h] | q_rope[h]] is one
// 576-wide row dotted against the cached row — MQA over a shared cache. The
// output regroups the same way:
//     o[h] = sum_t p_t (W_UV[h] c_t) = W_UV[h] (sum_t p_t c_t)
// with the probs applied in 512-wide latent space and one W_UV expansion per
// head at the end. W_UK/W_UV are read straight out of the checkpoint's
// `w_kv_b` ([96 heads x (128 nope | 128 value)] x 512): rows [h*256, h*256+128)
// are W_UK[h], rows [h*256+128, h*256+256) are W_UV[h].
//
// ---------------------------------------------------------------------------
// Rounding chain (each landing deliberate; 1-2 mirror every projection's
// f32-matmul-then-one-bf16-landing, 3-6 are the certified slot-indexed
// kernel's spelling)
// ---------------------------------------------------------------------------
//   1. q_abs[0..512)  = bf16(f32 sum_d q_nope[d] * W_UK[d, j]), d ascending;
//      q_abs[512..576) = q_rope, copied bf16.
//   2. dot(t) = f32 sum_d q_abs[d] * c_t[d] over 576, d ascending, one thread.
//   3. scl(t) = f32( bf16(dot) * scale ), the product taken in bf16.
//   4. m = max_t scl;  tot = sum_t exp(scl - m) in f32 (per-thread strided
//      partials in ascending t, then a fixed-order tree reduction).
//   5. p_t = bf16( exp(scl - m) / tot ).
//   6. o_lat[j] = bf16( f32 sum_t p_t * c_t[j] ), t ascending (chunk-major).
//   7. o[dv] = bf16( f32 sum_j W_UV[dv, j] * o_lat[j] ), j ascending.
//
// The context walk is three sweeps over the pages (max, sum, probs+attend):
// scores are recomputed rather than stored, so nothing is sized by the context
// length and there is no compile-time cap. A recomputed score is the same
// expression over the same operands in the same order, hence bit-identical.
// The walk is by *logical* position — the block table only selects which
// physical page backs a 64-token window — so any permutation of physical
// pages produces bit-identical output. A page id below zero (padding rows)
// reads as a zero latent row, which is what the retired slot-indexed kernel's
// zeroed cache produced.
//
// CUDA-graph safety: no allocation, no host readback, no device-side launch;
// launch geometry is (b, heads) with everything else read from device tensors.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>

namespace {

constexpr int kNope = 128;    // qk_nope_head_dim
constexpr int kRope = 64;     // qk_rope_head_dim
constexpr int kLatent = 512;  // kv_lora_rank
constexpr int kRow = kLatent + kRope;  // cached latent row width
constexpr int kVd = 128;      // v_head_dim
constexpr int kPageTokens = 64;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / WARP_SIZE;
constexpr int kDimsPerThread = kLatent / kThreads;
constexpr float kNeg = -1.0e30f;

// Fixed-order block reductions: warp shuffle trees, then thread 0 folds the
// warp partials in ascending warp order. The result is broadcast via stage[0].
__device__ __forceinline__ float block_max(float value, float* stage) {
  value = warp_reduce_max(value);
  if ((threadIdx.x & (WARP_SIZE - 1)) == 0) stage[threadIdx.x / WARP_SIZE] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    float folded = stage[0];
    for (int w = 1; w < kWarps; ++w) folded = fmaxf(folded, stage[w]);
    stage[0] = folded;
  }
  __syncthreads();
  float out = stage[0];
  __syncthreads();
  return out;
}

__device__ __forceinline__ float block_sum(float value, float* stage) {
  value = warp_reduce_sum(value);
  if ((threadIdx.x & (WARP_SIZE - 1)) == 0) stage[threadIdx.x / WARP_SIZE] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    float folded = stage[0];
    for (int w = 1; w < kWarps; ++w) folded += stage[w];
    stage[0] = folded;
  }
  __syncthreads();
  float out = stage[0];
  __syncthreads();
  return out;
}

// Landed score of logical position `s`: chain steps 2-3. Deterministic in the
// operands alone, so the three sweeps recompute it bit-identically.
__device__ __forceinline__ float position_score(
    const __nv_bfloat16* __restrict__ q_abs,
    const __nv_bfloat16* __restrict__ cache, const int* __restrict__ bt,
    long long page_stride, int s, __nv_bfloat16 sc) {
  const int page = bt[s / kPageTokens];
  float acc = 0.0f;
  if (page >= 0) {
    const __nv_bfloat16* c = cache + (long long)page * page_stride +
                             (long long)(s % kPageTokens) * kRow;
    for (int d = 0; d < kRow; ++d) {
      acc += __bfloat162float(q_abs[d]) * __bfloat162float(c[d]);
    }
  }
  return __bfloat162float(__hmul(__float2bfloat16_rn(acc), sc));
}

__global__ void mla_paged_absorbed_attn_kernel(
    const __nv_bfloat16* __restrict__ q,        // [b, heads * 192]
    const __nv_bfloat16* __restrict__ w_kv_b,   // [heads * 256, 512]
    const __nv_bfloat16* __restrict__ cache,    // layer-shifted slab base
    const int* __restrict__ table,              // [b, max_pages]
    int max_pages,
    long long page_stride,                      // elements from page to page
    const int* __restrict__ n,                  // [b] context lengths
    const __nv_bfloat16* __restrict__ scale,    // [1] softmax scale
    __nv_bfloat16* __restrict__ o) {            // [b, heads * 128]
  const int bb = blockIdx.x;
  const int bh = blockIdx.y;
  const int heads = gridDim.y;
  const int tid = threadIdx.x;
  const int ctx = n[bb];
  const __nv_bfloat16* qh =
      q + ((size_t)bb * heads + bh) * (size_t)(kNope + kRope);
  const int* bt = table + (size_t)bb * max_pages;
  const __nv_bfloat16 sc = scale[0];

  __shared__ __nv_bfloat16 q_abs[kRow];
  __shared__ __nv_bfloat16 probs[kPageTokens];
  __shared__ __nv_bfloat16 o_lat[kLatent];
  __shared__ float stage[kWarps];

  // Chain step 1: the absorbed query.
  const __nv_bfloat16* w_uk = w_kv_b + (size_t)bh * 2 * kVd * kLatent;
  for (int j = tid; j < kLatent; j += kThreads) {
    float acc = 0.0f;
    for (int d = 0; d < kNope; ++d) {
      acc += __bfloat162float(qh[d]) * __bfloat162float(w_uk[(size_t)d * kLatent + j]);
    }
    q_abs[j] = __float2bfloat16_rn(acc);
  }
  for (int j = tid; j < kRope; j += kThreads) {
    q_abs[kLatent + j] = qh[kNope + j];
  }
  __syncthreads();

  // Sweep 1 (chain step 4a): the score maximum.
  float local = kNeg;
  for (int s = tid; s < ctx; s += kThreads) {
    local = fmaxf(local, position_score(q_abs, cache, bt, page_stride, s, sc));
  }
  const float mx = block_max(local, stage);

  // Sweep 2 (chain step 4b): the softmax denominator.
  local = 0.0f;
  for (int s = tid; s < ctx; s += kThreads) {
    local += expf(position_score(q_abs, cache, bt, page_stride, s, sc) - mx);
  }
  const float tot = block_sum(local, stage);

  // Sweep 3 (chain steps 5-6): bf16 probs per page chunk, then the latent
  // accumulation, each thread owning kDimsPerThread strided latent dims.
  float oacc[kDimsPerThread];
  for (int i = 0; i < kDimsPerThread; ++i) oacc[i] = 0.0f;
  const int chunks = (ctx + kPageTokens - 1) / kPageTokens;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int base = chunk * kPageTokens;
    const int len = min(kPageTokens, ctx - base);
    const int page = bt[chunk];
    __syncthreads();  // probs from the previous chunk are consumed
    if (tid < len) {
      const float scl =
          position_score(q_abs, cache, bt, page_stride, base + tid, sc);
      probs[tid] = __float2bfloat16_rn(expf(scl - mx) / tot);
    }
    __syncthreads();
    if (page >= 0) {
      const __nv_bfloat16* cpage = cache + (long long)page * page_stride;
      for (int i = 0; i < kDimsPerThread; ++i) {
        const int j = i * kThreads + tid;
        float acc = oacc[i];
        for (int t = 0; t < len; ++t) {
          acc += __bfloat162float(probs[t]) *
                 __bfloat162float(cpage[(size_t)t * kRow + j]);
        }
        oacc[i] = acc;
      }
    }
  }
  for (int i = 0; i < kDimsPerThread; ++i) {
    o_lat[i * kThreads + tid] = __float2bfloat16_rn(oacc[i]);
  }
  __syncthreads();

  // Chain step 7: the W_UV expansion, one value dim per thread.
  const __nv_bfloat16* w_uv = w_kv_b + ((size_t)bh * 2 * kVd + kNope) * kLatent;
  const int dv = tid;  // kThreads == kVd
  float acc = 0.0f;
  for (int j = 0; j < kLatent; ++j) {
    acc += __bfloat162float(w_uv[(size_t)dv * kLatent + j]) *
           __bfloat162float(o_lat[j]);
  }
  o[((size_t)bb * heads + bh) * (size_t)kVd + dv] = __float2bfloat16_rn(acc);
}

CUresult map_cuda_error(cudaError_t err) {
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  if (err == cudaErrorNotSupported) return CUDA_ERROR_NOT_SUPPORTED;
  return CUDA_ERROR_LAUNCH_FAILED;
}

}  // namespace

extern "C" {

// Absorbed-MLA decode over the paged latent cache. `cache` is the pool slab;
// `layer_offset` (elements) shifts it to this layer's slice inside every page
// and `page_stride` (elements) is the page-to-page distance, so the kernel is
// layout-agnostic beyond "64-token slices of 576-wide rows". `table` is the
// device block table (`-1` = unmapped, read as zero latent), `n` the per-row
// device context lengths — no host sync anywhere. Geometry is pinned: qk_dim
// 192 (128 nope + 64 rope), v_dim 128, 96-head `w_kv_b` rows per head
// `[128 nope | 128 value] x 512`.
CUresult k3_mla_paged_attn_cuda(const __nv_bfloat16* q,
                                const __nv_bfloat16* w_kv_b,
                                const __nv_bfloat16* cache,
                                long long layer_offset, long long page_stride,
                                const int* table, int max_pages, const int* n,
                                const __nv_bfloat16* scale, __nv_bfloat16* o,
                                int b, int num_heads, int qk_dim, int v_dim,
                                cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (q == nullptr || w_kv_b == nullptr || cache == nullptr ||
      table == nullptr || n == nullptr || scale == nullptr || o == nullptr ||
      b <= 0 || num_heads <= 0 || qk_dim != kNope + kRope || v_dim != kVd ||
      max_pages <= 0 || layer_offset < 0 || page_stride < kPageTokens * kRow) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dim3 grid(b, num_heads);
  mla_paged_absorbed_attn_kernel<<<grid, kThreads, 0, stream>>>(
      q, w_kv_b, cache + layer_offset, table, max_pages, page_stride, n, scale,
      o);
  return map_cuda_error(cudaGetLastError());
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

}  // extern "C"
