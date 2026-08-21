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
// Schedule: two passes, warp-cooperative scores
// ---------------------------------------------------------------------------
// The context is walked twice, page by page. Pass one computes each page's
// landed scores (a warp per token, lanes striding the 576 dims with paired
// bf16 loads, then the retired kernel's `bf16(dot) * scale` landing — which
// also absorbs the shuffle tree's f32 summation-order noise) and folds them
// into a running max and an online softmax denominator. Pass two recomputes
// the landed scores (bit-identical by construction), quantizes the
// normalized probabilities to bf16 — the retired kernel's spelling, kept so
// the whole probability chain stays inside the fixtures' calibrated noise
// floor — and accumulates the attended latent in f32, tokens ascending.
// Versus the retired kernel this trades its three serial-per-token score
// sweeps for two warp-cooperative ones and drops nothing else.
//
// Determinism (the property the verify gates require — NOT bit-compat with
// the retired three-sweep spelling): every reduction has a fixed order (lane
// shuffle trees, warp partials folded in ascending warp order, pages walked
// ascending, tokens ascending within a page), so a launch at the same
// geometry over the same operands is bit-identical run to run. The walk is by
// *logical* position — the block table only selects which physical page backs
// a 64-token window — so any permutation of physical pages produces
// bit-identical output. A page id below zero (padding rows) reads as a zero
// latent row: its score is exactly 0.0f (participating in the softmax like
// the zeroed-cache rows of the retired slot-indexed kernel) and it
// contributes nothing to the attended latent.
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
  const int warp = tid / WARP_SIZE;
  const int lane = tid & (WARP_SIZE - 1);
  const int ctx = n[bb];
  const __nv_bfloat16* qh =
      q + ((size_t)bb * heads + bh) * (size_t)(kNope + kRope);
  const int* bt = table + (size_t)bb * max_pages;
  const __nv_bfloat16 sc = scale[0];

  __shared__ alignas(8) __nv_bfloat16 q_abs[kRow];
  __shared__ float scores[kPageTokens];
  __shared__ float stage[kWarps];

  // The absorbed query: q_abs = [W_UK[h]^T q_nope | q_rope], f32 accumulate,
  // one bf16 landing (mirroring every projection's landing discipline).
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

  const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(q_abs);
  const int chunks = (ctx + kPageTokens - 1) / kPageTokens;

  // A page of landed scores: a warp per token (warps stride the page), lanes
  // stride the 576 dims in bf16 pairs, one fixed-order shuffle tree per token,
  // then the retired kernel's landing — bf16(dot) scaled in bf16. The landing
  // also absorbs the shuffle tree's f32 summation-order noise, and it makes
  // the recompute in the attend pass bit-identical to the stats pass.
  auto page_scores = [&](const __nv_bfloat16* cpage, int len) {
    for (int t = warp; t < len; t += kWarps) {
      float acc = 0.0f;
      if (cpage != nullptr) {
        const __nv_bfloat162* c2 = reinterpret_cast<const __nv_bfloat162*>(
            cpage + (size_t)t * kRow);
        for (int d = lane; d < kRow / 2; d += WARP_SIZE) {
          const float2 cf = __bfloat1622float2(c2[d]);
          const float2 qf = __bfloat1622float2(q2[d]);
          acc += qf.x * cf.x + qf.y * cf.y;
        }
        acc = warp_reduce_sum(acc);
      }
      if (lane == 0) {
        scores[t] = __bfloat162float(__hmul(__float2bfloat16_rn(acc), sc));
      }
    }
  };

  // Stats pass: the running score maximum and, online against it, the
  // softmax denominator. The max over landed scores is order-free, so it is
  // exactly the retired kernel's; the denominator differs from a flat
  // ascending-t sum only in f32 summation order — noise the bf16 prob
  // landing below absorbs.
  float m_run = kNeg;
  float l_run = 0.0f;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int base = chunk * kPageTokens;
    const int len = min(kPageTokens, ctx - base);
    const int page = bt[chunk];
    const __nv_bfloat16* cpage =
        page >= 0 ? cache + (long long)page * page_stride : nullptr;
    __syncthreads();  // scores consumed by the previous chunk's reductions
    page_scores(cpage, len);
    __syncthreads();
    float local = kNeg;
    for (int t = tid; t < len; t += kThreads) local = fmaxf(local, scores[t]);
    const float page_max = block_max(local, stage);
    const float m_new = fmaxf(m_run, page_max);
    local = 0.0f;
    for (int t = tid; t < len; t += kThreads) {
      local += expf(scores[t] - m_new);
    }
    const float page_sum = block_sum(local, stage);
    l_run = l_run * expf(m_run - m_new) + page_sum;  // exp(-1e30)==0 on entry
    m_run = m_new;
  }

  // Attend pass: recompute each page's landed scores (bit-identical by
  // construction), take the retired kernel's bf16 probabilities against the
  // final max/denominator, and accumulate the latent row — threads own
  // kDimsPerThread strided latent dims, tokens ascending.
  float oacc[kDimsPerThread];
  for (int i = 0; i < kDimsPerThread; ++i) oacc[i] = 0.0f;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int base = chunk * kPageTokens;
    const int len = min(kPageTokens, ctx - base);
    const int page = bt[chunk];
    const __nv_bfloat16* cpage =
        page >= 0 ? cache + (long long)page * page_stride : nullptr;
    __syncthreads();  // scores consumed by the previous chunk's attend
    page_scores(cpage, len);
    __syncthreads();
    for (int t = tid; t < len; t += kThreads) {
      scores[t] = __bfloat162float(
          __float2bfloat16_rn(expf(scores[t] - m_run) / l_run));
    }
    __syncthreads();
    if (cpage != nullptr) {
      for (int i = 0; i < kDimsPerThread; ++i) {
        const int j = i * kThreads + tid;
        float acc = oacc[i];
        for (int t = 0; t < len; ++t) {
          acc += scores[t] * __bfloat162float(cpage[(size_t)t * kRow + j]);
        }
        oacc[i] = acc;
      }
    }
  }

  // Land the attended latent row in bf16.
  __shared__ __nv_bfloat16 o_lat[kLatent];
  for (int i = 0; i < kDimsPerThread; ++i) {
    o_lat[i * kThreads + tid] = __float2bfloat16_rn(oacc[i]);
  }
  __syncthreads();

  // The W_UV expansion, one value dim per thread.
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
// `[128 nope | 128 value] x 512`. `layer_offset`/`page_stride` must be even
// (the score loop reads the cached rows as bf16 pairs).
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
      max_pages <= 0 || layer_offset < 0 || page_stride < kPageTokens * kRow ||
      (layer_offset & 1) != 0 || (page_stride & 1) != 0) {
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
