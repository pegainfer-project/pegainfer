// Split-KV paged attention decode at HEAD_DIM=256 (Qwen3.5 full-attention
// layers), the path small decode batches take instead of FlashInfer's.

#include "../shared/ffi_guard.cuh"

#include <cfloat>
#include <cmath>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" {

// ---------------------------------------------------------------------------
// Split-KV decode at head_dim 256, written for sm_80.
//
// The FlashInfer path (paged_attention_hd256.cu) launches nblks(padded_batch_size, num_kv_heads) and
// never grows the grid with KV length, so a single-request decode runs four CTAs
// and a 16-request decode runs 64, each with 128 threads walking a four-position
// inner tile. On A100 that leaves the device idle: measured per layer-step, the
// FlashInfer kernel costs about 154 us at c16 and about 156 us at bs1, where
// vLLM's flash_fwd_splitkv costs 72 us and 26 us.
//
// This kernel splits the KV range across CTAs instead. One CTA per (split, kv
// head, request); each warp owns one query head of the GQA group and streams its
// share of the range one position at a time straight into registers, with no
// shared memory and no barrier. Partials are merged by a second kernel, one CTA
// per (request, query head).
//
// Measured standalone against the same paged layout: 33.6 us at bs1 / ctx 1024
// and 124.3 us at c16 / ctx 1024, against a 54 us memory floor at c16 (a
// load-only control reaches 1235 GB/s). Serving measurements at every bucket
// this kernel is admitted for beat the FlashInfer path — see the threshold in
// pegainfer-qwen35/src/decode_buffers.rs — and c16 gains less than bs1 does
// because at that size the kernel is closer to its own arithmetic limit. The
// remaining c16 headroom needs tensor cores.
// ---------------------------------------------------------------------------

constexpr int kSplitDecodeHeadDim = 256;
constexpr int kSplitDecodeVec = 8;   // bf16 per lane per head-dim slice (32 * 8 = 256)
constexpr int kSplitDecodeGqa = 4;   // Qwen3.5-4B: 16 query heads over 4 kv heads
constexpr int kSplitDecodeThreads = 32 * kSplitDecodeGqa;

// KV positions staged per buffer. One cooperative copy serves all four warps
// instead of each warp reading the same K and V through L1, and `cp.async`
// copies global to shared without holding a register or stalling the warp, so
// the next tile's copy overlaps the whole of the current tile's compute. At 8
// the staging costs 16 KB of shared memory and leaves enough CTAs resident to
// keep the compute side fed; 4 and 16 both measure worse, and 32 collapses
// occupancy to two CTAs per SM and nearly doubles the c16 time.
constexpr int kSplitDecodeTile = 8;

__device__ __forceinline__ void split_decode_cp_async16(void* smem_dst, const void* gmem_src) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(gmem_src));
}

__device__ __forceinline__ void split_decode_cp_commit() {
  asm volatile("cp.async.commit_group;\n");
}

// `cp.async.wait_group` takes an immediate, and this file's entry points sit in
// an `extern "C"` block where a template may not be declared, so the two depths
// this kernel needs are two functions rather than one template.
__device__ __forceinline__ void split_decode_cp_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}

__device__ __forceinline__ void split_decode_cp_wait_one() {
  asm volatile("cp.async.wait_group 1;\n");
}

__global__ void __launch_bounds__(kSplitDecodeThreads)
split_decode_partial_hd256_kernel(
    const __nv_bfloat16* __restrict__ q,        // [batch, num_qo_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_pool,   // paged NHD, [page][page_size][kv_heads][head_dim]
    const __nv_bfloat16* __restrict__ v_pool,
    int64_t stride_page,
    int page_size,
    const int32_t* __restrict__ page_indices,
    const int32_t* __restrict__ page_indptr,
    const int32_t* __restrict__ kv_lens,
    float* __restrict__ partial_o,              // [splits, batch, num_qo_heads, head_dim]
    float* __restrict__ partial_m,              // [splits, batch, num_qo_heads]
    float* __restrict__ partial_l,              // [splits, batch, num_qo_heads]
    int num_qo_heads, int num_kv_heads, int num_splits, float sm_scale) {
  extern __shared__ __nv_bfloat16 split_decode_smem[];
  __nv_bfloat16* k_bufs = split_decode_smem;                              // [2][tile][head_dim]
  __nv_bfloat16* v_bufs = k_bufs + 2 * kSplitDecodeTile * kSplitDecodeHeadDim;

  const int split = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int b = blockIdx.z;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int q_head = kv_head * kSplitDecodeGqa + warp;
  const int head_base = lane * kSplitDecodeVec;
  const int batch_stride = gridDim.z;

  float* o_out =
      partial_o + (((int64_t)split * batch_stride + b) * num_qo_heads + q_head) * kSplitDecodeHeadDim;
  float* m_out = partial_m + ((int64_t)split * batch_stride + b) * num_qo_heads + q_head;
  float* l_out = partial_l + ((int64_t)split * batch_stride + b) * num_qo_heads + q_head;

  // A split that holds no positions falls out of the loop below with the
  // neutral state already in its registers: o = 0, l = 0, m = -inf. Each warp
  // writes its own query head's partial, so the merge kernel sees a zero weight
  // (exp(-inf - m) == 0) for that split instead of a stale row. Writing that
  // state from one warp, or once per CTA, would silently leave the other heads
  // holding whatever the previous layer left in the buffer.
  const int kv_len = kv_lens[b];
  const int chunk = (kv_len + num_splits - 1) / num_splits;
  const int kv_start = split * chunk;
  const int kv_end = min(kv_start + chunk, kv_len);

  float qf[kSplitDecodeVec];
  {
    const __nv_bfloat16* qp =
        q + ((int64_t)b * num_qo_heads + q_head) * kSplitDecodeHeadDim + head_base;
    const uint4 qv = *reinterpret_cast<const uint4*>(qp);
    const __nv_bfloat16* qh = reinterpret_cast<const __nv_bfloat16*>(&qv);
#pragma unroll
    for (int i = 0; i < kSplitDecodeVec; ++i) {
      qf[i] = __bfloat162float(qh[i]);
    }
  }

  float m_i = -INFINITY;
  float l_i = 0.0f;
  float o_i[kSplitDecodeVec];
#pragma unroll
  for (int i = 0; i < kSplitDecodeVec; ++i) {
    o_i[i] = 0.0f;
  }

  const int page_base = page_indptr[b];
  const int tile_count =
      (kv_end > kv_start) ? ((kv_end - kv_start + kSplitDecodeTile - 1) / kSplitDecodeTile) : 0;

  // One copy per tile, into the buffer whose parity is the tile index. The
  // gather is per position because a tile can straddle pages, and the guard
  // covers the short final tile.
  auto split_decode_issue = [&](int tile) {
    const int tile_start = kv_start + tile * kSplitDecodeTile;
    const int tn = min(kSplitDecodeTile, kv_end - tile_start);
    const int buf = tile & 1;
    for (int c = threadIdx.x; c < tn * 32; c += kSplitDecodeThreads) {
      const int row = c >> 5;
      const int slice = c & 31;
      const int token = tile_start + row;
      const int page = token / page_size;
      const int slot = token - page * page_size;
      const int64_t off = (int64_t)page_indices[page_base + page] * stride_page +
                          ((int64_t)slot * num_kv_heads + kv_head) * kSplitDecodeHeadDim +
                          slice * kSplitDecodeVec;
      const int at = (buf * kSplitDecodeTile + row) * kSplitDecodeHeadDim + slice * kSplitDecodeVec;
      split_decode_cp_async16(&k_bufs[at], &k_pool[off]);
      split_decode_cp_async16(&v_bufs[at], &v_pool[off]);
    }
    split_decode_cp_commit();
  };

  if (tile_count > 0) {
    split_decode_issue(0);
  }
  for (int tile = 0; tile < tile_count; ++tile) {
    const int tile_start = kv_start + tile * kSplitDecodeTile;
    const int tn = min(kSplitDecodeTile, kv_end - tile_start);
    const int buf = tile & 1;

    // Issue the next tile before waiting on this one, so the copy overlaps the
    // whole of the compute below rather than preceding it.
    if (tile + 1 < tile_count) {
      split_decode_issue(tile + 1);
      split_decode_cp_wait_one();
    } else {
      split_decode_cp_wait_all();
    }
    __syncthreads();

    const __nv_bfloat16* sk = &k_bufs[buf * kSplitDecodeTile * kSplitDecodeHeadDim];
    const __nv_bfloat16* sv = &v_bufs[buf * kSplitDecodeTile * kSplitDecodeHeadDim];
    // Two positions at a time. With the operands already in shared memory the
    // per-position cost is an eight-deep FFMA chain behind five dependent
    // shuffles, which is latency rather than throughput, and a second
    // independent position gives the scheduler something to issue while the
    // first chain waits.
    for (int j = 0; j < tn; j += 2) {
      const int last = min(j + 1, tn - 1);  // keeps the second read in bounds
      const bool has_second = (j + 1) < tn;

      float s0 = 0.0f;
      float s1 = 0.0f;
#pragma unroll
      for (int i = 0; i < kSplitDecodeVec; ++i) {
        s0 += qf[i] * __bfloat162float(sk[j * kSplitDecodeHeadDim + head_base + i]);
        s1 += qf[i] * __bfloat162float(sk[last * kSplitDecodeHeadDim + head_base + i]);
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, o);
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
      }
      s0 *= sm_scale;
      s1 = has_second ? s1 * sm_scale : -INFINITY;

      const float pair_max = fmaxf(s0, s1);
      if (pair_max > m_i) {
        const float alpha = __expf(m_i - pair_max);
#pragma unroll
        for (int i = 0; i < kSplitDecodeVec; ++i) {
          o_i[i] *= alpha;
        }
        l_i *= alpha;
        m_i = pair_max;
      }
      const float p0 = __expf(s0 - m_i);
      const float p1 = has_second ? __expf(s1 - m_i) : 0.0f;
      l_i += p0 + p1;
#pragma unroll
      for (int i = 0; i < kSplitDecodeVec; ++i) {
        o_i[i] += p0 * __bfloat162float(sv[j * kSplitDecodeHeadDim + head_base + i]);
      }
      if (has_second) {
#pragma unroll
        for (int i = 0; i < kSplitDecodeVec; ++i) {
          o_i[i] += p1 * __bfloat162float(sv[last * kSplitDecodeHeadDim + head_base + i]);
        }
      }
    }
    // Every warp is done with this buffer before tile + 2 copies into it.
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kSplitDecodeVec; ++i) {
    o_out[head_base + i] = o_i[i];
  }
  if (lane == 0) {
    *m_out = m_i;
    *l_out = l_i;
  }
}

// One CTA per (request, query head), one thread per head dimension.
__global__ void __launch_bounds__(kSplitDecodeHeadDim)
split_decode_combine_hd256_kernel(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __nv_bfloat16* __restrict__ output,
    int num_qo_heads, int num_splits, int batch_size) {
  const int b = blockIdx.x;
  const int q_head = blockIdx.y;
  const int d = threadIdx.x;

  float m = -FLT_MAX;
  for (int s = 0; s < num_splits; ++s) {
    m = fmaxf(m, partial_m[((int64_t)s * batch_size + b) * num_qo_heads + q_head]);
  }
  float acc = 0.0f;
  float l = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const int64_t row = ((int64_t)s * batch_size + b) * num_qo_heads + q_head;
    const float scale = __expf(partial_m[row] - m);
    acc += partial_o[row * kSplitDecodeHeadDim + d] * scale;
    l += partial_l[row] * scale;
  }
  output[((int64_t)b * num_qo_heads + q_head) * kSplitDecodeHeadDim + d] =
      __float2bfloat16(acc / l);
}

int paged_attention_decode_split_hd256_cuda(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr,
    int32_t* kv_chunk_size_ptr,
    void* partial_o, void* partial_m, void* partial_l,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t batch_size, int32_t num_splits,
    int64_t stride_page, float sm_scale, void* stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    if (num_splits < 1) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    // The kernel is shaped for Qwen3.5-4B's GQA group: one warp per query head.
    if (head_dim != kSplitDecodeHeadDim ||
        num_qo_heads != num_kv_heads * kSplitDecodeGqa) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    const __nv_bfloat16* k_base =
        reinterpret_cast<const __nv_bfloat16*>(kv_data) + k_offset_elems;
    const __nv_bfloat16* v_base =
        reinterpret_cast<const __nv_bfloat16*>(kv_data) + v_offset_elems;

    // Two buffers each of K and V, and the opt-in shared-memory size they need.
    // Set once per process; a second call with the same value is harmless.
    const int stage_smem = 4 * kSplitDecodeTile * kSplitDecodeHeadDim *
                           static_cast<int>(sizeof(__nv_bfloat16));
    cudaError_t smem_status = cudaFuncSetAttribute(
        split_decode_partial_hd256_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, stage_smem);
    if (smem_status != cudaSuccess) {
      return static_cast<int>(smem_status);
    }

    dim3 grid(num_splits, num_kv_heads, batch_size);
    split_decode_partial_hd256_kernel<<<grid, kSplitDecodeThreads, stage_smem, cu_stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), k_base, v_base, stride_page, page_size,
        page_indices, page_indptr, kv_chunk_size_ptr,
        reinterpret_cast<float*>(partial_o), reinterpret_cast<float*>(partial_m),
        reinterpret_cast<float*>(partial_l), num_qo_heads, num_kv_heads, num_splits, sm_scale);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
      return static_cast<int>(status);
    }

    dim3 combine_grid(batch_size, num_qo_heads);
    split_decode_combine_hd256_kernel<<<combine_grid, kSplitDecodeHeadDim, 0, cu_stream>>>(
        reinterpret_cast<const float*>(partial_o), reinterpret_cast<const float*>(partial_m),
        reinterpret_cast<const float*>(partial_l), reinterpret_cast<__nv_bfloat16*>(output),
        num_qo_heads, num_splits, batch_size);
    return static_cast<int>(cudaGetLastError());
  PEGAINFER_FFI_GUARD_END(-1)
}

} // extern "C"
