#include "../shared/ffi_guard.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdint>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>

extern thread_local cublasHandle_t g_cublas_handle;

extern "C" int kimi_mla_absorb_q_nope_cublaslt_cuda(const __nv_bfloat16* kv_b_proj,
                                                    const __nv_bfloat16* q_nope,
                                                    __nv_bfloat16* q_abs_nope,
                                                    int batch_size,
                                                    int local_heads,
                                                    cudaStream_t stream);
extern "C" int kimi_mla_v_up_cublaslt_cuda(const __nv_bfloat16* kv_b_proj,
                                           const __nv_bfloat16* latent,
                                           __nv_bfloat16* output,
                                           int batch_size,
                                           int local_heads,
                                           cudaStream_t stream);

namespace {

using DType = __nv_bfloat16;
using IdType = int32_t;
using PrefillParamsT = flashinfer::SinglePrefillParams<DType, DType, DType>;
using MlaDecodeParamsT = flashinfer::BatchDecodeParamsMLA<DType, DType, DType, IdType>;
using Variant = flashinfer::DefaultAttention</*custom_mask=*/false,
                                           /*sliding_window=*/false,
                                           /*logits_soft_cap=*/false,
                                           /*alibi=*/false>;

constexpr int kKvLoraRank = 512;
constexpr int kRopeDim = 64;
constexpr int kNopeDim = 128;
constexpr int kQHeadDim = 192;
constexpr int kVHeadDim = 128;
constexpr int kKvBHeadDim = kNopeDim + kVHeadDim;
constexpr int kLocalHeads = 8;
constexpr int kQLoraRank = 1536;
constexpr int kQkvAOut = kQLoraRank + kKvLoraRank + kRopeDim;
constexpr int kMlaLtLocalHeads = 64;
constexpr int kMlaLtMaxBatch = 8;

flashinfer::paged_kv_mla_t<DType, IdType> make_paged_kv_mla(void* ckv_cache,
                                                             void* kpe_cache,
                                                             IdType* page_indices,
                                                             IdType* page_indptr,
                                                             IdType* last_page_len,
                                                             int64_t ckv_stride_page,
                                                             int64_t ckv_stride_n,
                                                             int64_t kpe_stride_page,
                                                             int64_t kpe_stride_n,
                                                             int page_size,
                                                             int batch_size) {
  int64_t ckv_strides[2] = {ckv_stride_page, ckv_stride_n};
  int64_t kpe_strides[2] = {kpe_stride_page, kpe_stride_n};
  return flashinfer::paged_kv_mla_t<DType, IdType>(
      static_cast<uint32_t>(page_size),
      static_cast<uint32_t>(kKvLoraRank),
      static_cast<uint32_t>(kRopeDim),
      static_cast<uint32_t>(batch_size),
      reinterpret_cast<DType*>(ckv_cache),
      ckv_strides,
      reinterpret_cast<DType*>(kpe_cache),
      kpe_strides,
      page_indices,
      page_indptr,
      last_page_len,
      /*rope_pos_offset=*/nullptr);
}

__global__ void split_qkv_a_kernel(const DType* __restrict__ qkv_a,
                                   DType* __restrict__ q_a,
                                   DType* __restrict__ compressed,
                                   DType* __restrict__ k_rope,
                                   int seq_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * kQkvAOut;
  if (idx >= total) {
    return;
  }
  int token = idx / kQkvAOut;
  int dim = idx - token * kQkvAOut;
  if (dim < kQLoraRank) {
    q_a[token * kQLoraRank + dim] = qkv_a[idx];
  } else if (dim < kQLoraRank + kKvLoraRank) {
    compressed[token * kKvLoraRank + (dim - kQLoraRank)] = qkv_a[idx];
  } else {
    k_rope[token * kRopeDim + (dim - kQLoraRank - kKvLoraRank)] = qkv_a[idx];
  }
}

__device__ __forceinline__ float fi_rsqrt_norm(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float fi_shfl_xor(float x, int lane_mask) {
  return __shfl_xor_sync(0xffffffff, x, lane_mask);
}

__global__ void split_qkv_a_norm_kernel(
    const DType* __restrict__ qkv_a,
    const DType* __restrict__ q_a_weight,
    const DType* __restrict__ ckv_weight,
    DType* __restrict__ q_a_normed,
    DType* __restrict__ ckv_normed,
    DType* __restrict__ k_rope_out,
    float eps,
    int batch_size) {

  const int token = blockIdx.x;
  if (token >= batch_size) return;

  const uint32_t tx = threadIdx.x;
  const uint32_t ty = threadIdx.y;
  constexpr uint32_t kWarpSize = 32;
  const uint32_t thread_id = tx + ty * kWarpSize;

  constexpr int VEC = 8;
  constexpr int CKV_THREADS = kKvLoraRank / VEC;
  constexpr int ROPE_THREADS = kRopeDim / VEC;
  constexpr int Q_WARPS = 6;
  constexpr int CKV_WARPS = 2;

  extern __shared__ float smem[];
  float* smem_q = smem;
  float* smem_ckv = smem + Q_WARPS;

  const DType* token_qkv = qkv_a + token * kQkvAOut;

  float q_vals[VEC];
  float sum_sq_q = 0.f;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  {
    const DType* src = token_qkv + thread_id * VEC;
#pragma unroll
    for (int j = 0; j < VEC; j++) {
      q_vals[j] = __bfloat162float(src[j]);
      sum_sq_q += q_vals[j] * q_vals[j];
    }
  }

  float ckv_vals[VEC];
  float sum_sq_ckv = 0.f;
  if (thread_id < CKV_THREADS) {
    const DType* src = token_qkv + kQLoraRank + thread_id * VEC;
#pragma unroll
    for (int j = 0; j < VEC; j++) {
      ckv_vals[j] = __bfloat162float(src[j]);
      sum_sq_ckv += ckv_vals[j] * ckv_vals[j];
    }
  }

#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
    sum_sq_q += fi_shfl_xor(sum_sq_q, offset);
  smem_q[ty] = sum_sq_q;

#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
    sum_sq_ckv += fi_shfl_xor(sum_sq_ckv, offset);
  smem_ckv[ty] = sum_sq_ckv;

  __syncthreads();

  if (ty == 0) {
    float total_q = (tx < Q_WARPS) ? smem_q[tx] : 0.f;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
      total_q += fi_shfl_xor(total_q, offset);
    smem_q[0] = total_q;

    float total_ckv = (tx < CKV_WARPS) ? smem_ckv[tx] : 0.f;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
      total_ckv += fi_shfl_xor(total_ckv, offset);
    smem_ckv[0] = total_ckv;
  }
  __syncthreads();

  float rms_rcp_q = fi_rsqrt_norm(smem_q[0] / float(kQLoraRank) + eps);
  float rms_rcp_ckv = fi_rsqrt_norm(smem_ckv[0] / float(kKvLoraRank) + eps);

  {
    DType* dst = q_a_normed + token * kQLoraRank + thread_id * VEC;
    const DType* w = q_a_weight + thread_id * VEC;
#pragma unroll
    for (int j = 0; j < VEC; j++)
      dst[j] = __float2bfloat16(q_vals[j] * rms_rcp_q * __bfloat162float(w[j]));
  }

  if (thread_id < CKV_THREADS) {
    DType* dst = ckv_normed + token * kKvLoraRank + thread_id * VEC;
    const DType* w = ckv_weight + thread_id * VEC;
#pragma unroll
    for (int j = 0; j < VEC; j++)
      dst[j] = __float2bfloat16(ckv_vals[j] * rms_rcp_ckv * __bfloat162float(w[j]));
  }

  if (thread_id < ROPE_THREADS) {
    const DType* src = token_qkv + kQLoraRank + kKvLoraRank + thread_id * VEC;
    DType* dst = k_rope_out + token * kRopeDim + thread_id * VEC;
#pragma unroll
    for (int j = 0; j < VEC; j++)
      dst[j] = src[j];
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

__device__ __forceinline__ DType rope_out(DType even, DType odd, DType cos_v,
                                          DType sin_v, bool upper) {
  float x_even = __bfloat162float(even);
  float x_odd = __bfloat162float(odd);
  float c = __bfloat162float(cos_v);
  float s = __bfloat162float(sin_v);
  float value = upper ? (x_odd * c + x_even * s) : (x_even * c - x_odd * s);
  return __float2bfloat16(value);
}

// Builds suffix rows of the prefill attention inputs. q rows cover the
// seq_len suffix tokens only; k/v rows land at start_pos..start_pos+seq_len
// inside caches whose token dimension is kv_len (prefix-cache hit: rows
// 0..start_pos were assembled from pooled latent KV by
// assemble_cached_kv_kernel). RoPE positions are absolute: start_pos + token.
// Cold prefill is the start_pos = 0, kv_len = seq_len special case.
__global__ void rope_assemble_prefill_kernel(const DType* __restrict__ q_proj,
                                             const DType* __restrict__ k_rope,
                                             const DType* __restrict__ kv_b,
                                             const DType* __restrict__ cos,
                                             const DType* __restrict__ sin,
                                             DType* __restrict__ q_attn,
                                             DType* __restrict__ k_cache,
                                             DType* __restrict__ v_cache,
                                             int seq_len,
                                             int start_pos,
                                             int kv_len,
                                             int local_heads) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * local_heads * kQHeadDim;
  if (idx >= total) {
    return;
  }

  int dim = idx % kQHeadDim;
  int head_token = idx / kQHeadDim;
  int head = head_token % local_heads;
  int token = head_token / local_heads;
  int q_base = token * local_heads * kQHeadDim + head * kQHeadDim;

  if (dim < kNopeDim) {
    q_attn[idx] = q_proj[idx];
    int k_dst = head * kv_len * kQHeadDim + (start_pos + token) * kQHeadDim + dim;
    int kv_src = token * local_heads * kKvBHeadDim + head * kKvBHeadDim + dim;
    k_cache[k_dst] = kv_b[kv_src];
    return;
  }

  int rope_dim = dim - kNopeDim;
  int pair = rope_dim % (kRopeDim / 2);
  bool upper = rope_dim >= (kRopeDim / 2);
  int q_even_idx = q_base + kNopeDim + pair * 2;
  int q_odd_idx = q_even_idx + 1;
  int k_even_idx = token * kRopeDim + pair * 2;
  int k_odd_idx = k_even_idx + 1;
  // Kimi's HF code first converts adjacent RoPE pairs from
  // [x0, x1, x2, x3, ...] to split-half [x0, x2, ..., x1, x3, ...]
  // with view(..., d/2, 2).transpose(...).  The rotated Q/K tail is
  // intentionally written in that split-half layout.
  int rope_cache_idx = (start_pos + token) * kRopeDim + rope_dim;
  DType cos_v = cos[rope_cache_idx];
  DType sin_v = sin[rope_cache_idx];

  q_attn[idx] = rope_out(q_proj[q_even_idx], q_proj[q_odd_idx], cos_v, sin_v, upper);
  int k_dst = head * kv_len * kQHeadDim + (start_pos + token) * kQHeadDim + dim;
  k_cache[k_dst] = rope_out(k_rope[k_even_idx], k_rope[k_odd_idx], cos_v, sin_v, upper);
}

__global__ void assemble_v_cache_kernel(const DType* __restrict__ kv_b,
                                        DType* __restrict__ v_cache,
                                        int seq_len,
                                        int start_pos,
                                        int kv_len,
                                        int local_heads) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * local_heads * kVHeadDim;
  if (idx >= total) {
    return;
  }
  int dim = idx % kVHeadDim;
  int head_token = idx / kVHeadDim;
  int head = head_token % local_heads;
  int token = head_token / local_heads;
  int src = token * local_heads * kKvBHeadDim + head * kKvBHeadDim + kNopeDim + dim;
  int dst = head * kv_len * kVHeadDim + (start_pos + token) * kVHeadDim + dim;
  v_cache[dst] = kv_b[src];
}

// Prefix-cache hit support: scatter pooled latent KV pages into a contiguous
// token-major buffer so the kv_b decompression GEMM can run over the cached
// prefix. ckv_out is [cached_len, 512] row-major, matching the layout the
// cold path feeds to the same GEMM.
__global__ void gather_cached_ckv_kernel(const DType* __restrict__ ckv_cache,
                                         const IdType* __restrict__ page_indices,
                                         DType* __restrict__ ckv_out,
                                         int cached_len,
                                         int page_size,
                                         int64_t ckv_stride_page,
                                         int64_t ckv_stride_n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = cached_len * kKvLoraRank;
  if (idx >= total) {
    return;
  }
  int dim = idx % kKvLoraRank;
  int token = idx / kKvLoraRank;
  IdType page = page_indices[token / page_size];
  int entry = token % page_size;
  ckv_out[idx] = ckv_cache[page * ckv_stride_page + entry * ckv_stride_n + dim];
}

// Prefix-cache hit support: build k/v cache rows 0..cached_len from the
// decompressed latent (kv_b GEMM output over the gathered prefix) plus the
// pooled kpe pages. Pool kpe is stored post-RoPE (both append paths rotate
// before writing), so it is broadcast per head verbatim — re-rotating here
// would corrupt the prefix.
__global__ void assemble_cached_kv_kernel(const DType* __restrict__ kv_b,
                                          const DType* __restrict__ kpe_cache,
                                          const IdType* __restrict__ page_indices,
                                          DType* __restrict__ k_cache,
                                          DType* __restrict__ v_cache,
                                          int cached_len,
                                          int kv_len,
                                          int local_heads,
                                          int page_size,
                                          int64_t kpe_stride_page,
                                          int64_t kpe_stride_n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = cached_len * local_heads * kQHeadDim;
  if (idx >= total) {
    return;
  }
  int dim = idx % kQHeadDim;
  int head_token = idx / kQHeadDim;
  int head = head_token % local_heads;
  int token = head_token / local_heads;
  int k_dst = head * kv_len * kQHeadDim + token * kQHeadDim + dim;
  if (dim < kNopeDim) {
    int kv_src = token * local_heads * kKvBHeadDim + head * kKvBHeadDim + dim;
    k_cache[k_dst] = kv_b[kv_src];
    int v_dst = head * kv_len * kVHeadDim + token * kVHeadDim + dim;
    int v_src = token * local_heads * kKvBHeadDim + head * kKvBHeadDim + kNopeDim + dim;
    v_cache[v_dst] = kv_b[v_src];
    return;
  }
  int rope_dim = dim - kNopeDim;
  IdType page = page_indices[token / page_size];
  int entry = token % page_size;
  k_cache[k_dst] = kpe_cache[page * kpe_stride_page + entry * kpe_stride_n + rope_dim];
}

__global__ void rope_split_decode_kernel(const DType* __restrict__ q_proj,
                                         const DType* __restrict__ k_rope,
                                         const DType* __restrict__ cos,
                                         const DType* __restrict__ sin,
                                         const IdType* __restrict__ positions,
                                         DType* __restrict__ q_nope,
                                         DType* __restrict__ q_pe,
                                         DType* __restrict__ append_kpe,
                                         int batch_size,
                                         int local_heads) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * local_heads * kQHeadDim;
  if (idx >= total) {
    return;
  }

  int dim = idx % kQHeadDim;
  int head_token = idx / kQHeadDim;
  int head = head_token % local_heads;
  int token = head_token / local_heads;
  int q_base = token * local_heads * kQHeadDim + head * kQHeadDim;

  if (dim < kNopeDim) {
    int dst = token * local_heads * kNopeDim + head * kNopeDim + dim;
    q_nope[dst] = q_proj[idx];
    return;
  }

  int rope_dim = dim - kNopeDim;
  int pair = rope_dim % (kRopeDim / 2);
  bool upper = rope_dim >= (kRopeDim / 2);
  int position = positions[token];
  int rope_cache_idx = position * kRopeDim + rope_dim;
  DType cos_v = cos[rope_cache_idx];
  DType sin_v = sin[rope_cache_idx];

  int q_even_idx = q_base + kNopeDim + pair * 2;
  int q_odd_idx = q_even_idx + 1;
  int q_dst = token * local_heads * kRopeDim + head * kRopeDim + rope_dim;
  q_pe[q_dst] = rope_out(q_proj[q_even_idx], q_proj[q_odd_idx], cos_v, sin_v, upper);

  if (head == 0) {
    int k_even_idx = token * kRopeDim + pair * 2;
    int k_odd_idx = k_even_idx + 1;
    append_kpe[token * kRopeDim + rope_dim] =
        rope_out(k_rope[k_even_idx], k_rope[k_odd_idx], cos_v, sin_v, upper);
  }
}

__global__ void rope_apply_kpe_kernel(const DType* __restrict__ k_rope,
                                      const DType* __restrict__ cos,
                                      const DType* __restrict__ sin,
                                      const IdType* __restrict__ positions,
                                      DType* __restrict__ append_kpe,
                                      int seq_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * kRopeDim;
  if (idx >= total) {
    return;
  }

  int rope_dim = idx % kRopeDim;
  int token = idx / kRopeDim;
  int pair = rope_dim % (kRopeDim / 2);
  bool upper = rope_dim >= (kRopeDim / 2);
  int position = positions[token];
  int rope_cache_idx = position * kRopeDim + rope_dim;
  DType cos_v = cos[rope_cache_idx];
  DType sin_v = sin[rope_cache_idx];
  int k_even_idx = token * kRopeDim + pair * 2;
  int k_odd_idx = k_even_idx + 1;
  append_kpe[idx] = rope_out(k_rope[k_even_idx], k_rope[k_odd_idx], cos_v, sin_v, upper);
}

}  // namespace

extern "C" {

cudaError_t kimi_mla_split_qkv_a_cuda(const DType* qkv_a,
                                      DType* q_a,
                                      DType* compressed,
                                      DType* k_rope,
                                      int seq_len,
                                      cudaStream_t stream) {
  if (seq_len <= 0) {
    return cudaErrorInvalidValue;
  }
  int total = seq_len * kQkvAOut;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  split_qkv_a_kernel<<<blocks, threads, 0, stream>>>(qkv_a, q_a, compressed, k_rope, seq_len);
  return cudaGetLastError();
}

cudaError_t kimi_mla_split_qkv_a_norm_cuda(const DType* qkv_a,
                                            const DType* q_a_weight,
                                            const DType* ckv_weight,
                                            DType* q_a_normed,
                                            DType* ckv_normed,
                                            DType* k_rope,
                                            float eps,
                                            int batch_size,
                                            cudaStream_t stream) {
  if (batch_size <= 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int Q_WARPS = 6;
  int smem_bytes = (Q_WARPS + Q_WARPS) * static_cast<int>(sizeof(float));
  dim3 grid(batch_size);
  dim3 block(32, Q_WARPS);
  split_qkv_a_norm_kernel<<<grid, block, smem_bytes, stream>>>(
      qkv_a, q_a_weight, ckv_weight,
      q_a_normed, ckv_normed, k_rope, eps, batch_size);
  return cudaGetLastError();
}

cudaError_t kimi_mla_rope_assemble_prefill_cuda(const DType* q_proj,
                                                const DType* k_rope,
                                                const DType* kv_b,
                                                const DType* cos,
                                                const DType* sin,
                                                DType* q_attn,
                                                DType* k_cache,
                                                DType* v_cache,
                                                int seq_len,
                                                int start_pos,
                                                int kv_len,
                                                int local_heads,
                                                cudaStream_t stream) {
  if (seq_len <= 0 || local_heads <= 0 || start_pos < 0 || kv_len != start_pos + seq_len) {
    return cudaErrorInvalidValue;
  }
  int threads = 256;
  int qk_total = seq_len * local_heads * kQHeadDim;
  int qk_blocks = (qk_total + threads - 1) / threads;
  rope_assemble_prefill_kernel<<<qk_blocks, threads, 0, stream>>>(
      q_proj, k_rope, kv_b, cos, sin, q_attn, k_cache, v_cache, seq_len, start_pos, kv_len,
      local_heads);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }
  int v_total = seq_len * local_heads * kVHeadDim;
  int v_blocks = (v_total + threads - 1) / threads;
  assemble_v_cache_kernel<<<v_blocks, threads, 0, stream>>>(kv_b, v_cache, seq_len, start_pos,
                                                            kv_len, local_heads);
  return cudaGetLastError();
}

cudaError_t kimi_mla_gather_cached_ckv_cuda(const DType* ckv_cache,
                                            const IdType* page_indices,
                                            DType* ckv_out,
                                            int cached_len,
                                            int page_size,
                                            int64_t ckv_stride_page,
                                            int64_t ckv_stride_n,
                                            cudaStream_t stream) {
  if (cached_len <= 0 || page_size <= 0) {
    return cudaErrorInvalidValue;
  }
  int threads = 256;
  int total = cached_len * kKvLoraRank;
  int blocks = (total + threads - 1) / threads;
  gather_cached_ckv_kernel<<<blocks, threads, 0, stream>>>(
      ckv_cache, page_indices, ckv_out, cached_len, page_size, ckv_stride_page, ckv_stride_n);
  return cudaGetLastError();
}

cudaError_t kimi_mla_assemble_cached_kv_cuda(const DType* kv_b,
                                             const DType* kpe_cache,
                                             const IdType* page_indices,
                                             DType* k_cache,
                                             DType* v_cache,
                                             int cached_len,
                                             int kv_len,
                                             int local_heads,
                                             int page_size,
                                             int64_t kpe_stride_page,
                                             int64_t kpe_stride_n,
                                             cudaStream_t stream) {
  if (cached_len <= 0 || kv_len < cached_len || local_heads <= 0 || page_size <= 0) {
    return cudaErrorInvalidValue;
  }
  int threads = 256;
  int total = cached_len * local_heads * kQHeadDim;
  int blocks = (total + threads - 1) / threads;
  assemble_cached_kv_kernel<<<blocks, threads, 0, stream>>>(
      kv_b, kpe_cache, page_indices, k_cache, v_cache, cached_len, kv_len, local_heads, page_size,
      kpe_stride_page, kpe_stride_n);
  return cudaGetLastError();
}

cudaError_t kimi_mla_rope_split_decode_cuda(const DType* q_proj,
                                            const DType* k_rope,
                                            const DType* cos,
                                            const DType* sin,
                                            const IdType* positions,
                                            DType* q_nope,
                                            DType* q_pe,
                                            DType* append_kpe,
                                            int batch_size,
                                            int local_heads,
                                            cudaStream_t stream) {
  if (batch_size <= 0 || local_heads <= 0) {
    return cudaErrorInvalidValue;
  }
  int total = batch_size * local_heads * kQHeadDim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  rope_split_decode_kernel<<<blocks, threads, 0, stream>>>(
      q_proj, k_rope, cos, sin, positions, q_nope, q_pe, append_kpe, batch_size, local_heads);
  return cudaGetLastError();
}

cudaError_t kimi_mla_rope_apply_kpe_cuda(const DType* k_rope,
                                         const DType* cos,
                                         const DType* sin,
                                         const IdType* positions,
                                         DType* append_kpe,
                                         int seq_len,
                                         cudaStream_t stream) {
  if (seq_len <= 0) {
    return cudaErrorInvalidValue;
  }
  int total = seq_len * kRopeDim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  rope_apply_kpe_kernel<<<blocks, threads, 0, stream>>>(
      k_rope, cos, sin, positions, append_kpe, seq_len);
  return cudaGetLastError();
}

int kimi_flashinfer_single_prefill_mla_cuda(void* q,
                                            void* output,
                                            void* k_cache,
                                            void* v_cache,
                                            int local_heads,
                                            int seq_len,
                                            int kv_len,
                                            float sm_scale,
                                            cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  // kv_len > seq_len is the prefix-cache suffix prefill: q covers the suffix
  // only, k/v cover cached prefix + suffix. MaskMode::kCausal aligns
  // bottom-right, so query i attends to kv positions <= kv_len - seq_len + i —
  // exactly the absolute-position causal mask for a suffix starting at
  // kv_len - seq_len.
  if (local_heads <= 0 || seq_len <= 0 || kv_len < seq_len) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  PrefillParamsT params;
  params.q = reinterpret_cast<DType*>(q);
  params.k = reinterpret_cast<DType*>(k_cache);
  params.v = reinterpret_cast<DType*>(v_cache);
  params.maybe_custom_mask = nullptr;
  params.o = reinterpret_cast<DType*>(output);
  params.lse = nullptr;
  params.maybe_alibi_slopes = nullptr;
  params.group_size = flashinfer::uint_fastdiv(1);
  params.qo_len = static_cast<uint32_t>(seq_len);
  params.kv_len = static_cast<uint32_t>(kv_len);
  params.num_qo_heads = static_cast<uint32_t>(local_heads);
  params.num_kv_heads = static_cast<uint32_t>(local_heads);
  params.q_stride_n = static_cast<uint32_t>(local_heads * kQHeadDim);
  params.q_stride_h = static_cast<uint32_t>(kQHeadDim);
  params.k_stride_n = static_cast<uint32_t>(kQHeadDim);
  params.k_stride_h = static_cast<uint32_t>(kv_len * kQHeadDim);
  params.v_stride_n = static_cast<uint32_t>(kVHeadDim);
  params.v_stride_h = static_cast<uint32_t>(kv_len * kVHeadDim);
  params.head_dim = static_cast<uint32_t>(kQHeadDim);
  params.window_left = -1;
  params.logits_soft_cap = 0.0f;
  params.sm_scale = sm_scale;
  params.rope_rcp_scale = 1.0f;
  params.rope_rcp_theta = 1.0e-6f;
  params.partition_kv = false;

  return static_cast<int>(
      flashinfer::SinglePrefillWithKVCacheDispatched<
          /*HEAD_DIM_QK=*/kQHeadDim,
          /*HEAD_DIM_VO=*/kVHeadDim,
          flashinfer::PosEncodingMode::kNone,
          /*USE_FP16_QK_REDUCTION=*/false,
          flashinfer::MaskMode::kCausal,
          Variant,
          PrefillParamsT>(params, /*tmp=*/nullptr, stream));
  PEGAINFER_FFI_GUARD_END(-1)
}

int kimi_mla_absorb_q_nope_cuda(const DType* kv_b_proj,
                                const DType* q_nope,
                                DType* q_abs_nope,
                                int batch_size,
                                int local_heads,
                                cudaStream_t stream) {
  if (batch_size <= 0 || local_heads <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (g_cublas_handle == nullptr) {
    return static_cast<int>(cudaErrorInitializationError);
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = cublasSetStream(g_cublas_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return static_cast<int>(cudaErrorInvalidResourceHandle);
  }

  if (local_heads == kMlaLtLocalHeads && batch_size <= kMlaLtMaxBatch) {
    // cuBLASLt owns this shape; a failure propagates as an error instead of
    // silently falling back to the reference GEMM below.
    return kimi_mla_absorb_q_nope_cublaslt_cuda(kv_b_proj, q_nope, q_abs_nope,
                                                batch_size, local_heads, stream);
  }

  // kv_b_proj is row-major [local_heads, k_nope + v, kv_lora_rank].
  // Row-major [k_nope, kv_lora_rank] is the same memory as column-major
  // [kv_lora_rank, k_nope], which is exactly W_UK_T for q absorption.
  status = cublasGemmStridedBatchedEx(
      g_cublas_handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      /*m=*/kKvLoraRank,
      /*n=*/batch_size,
      /*k=*/kNopeDim,
      &alpha,
      kv_b_proj,
      CUDA_R_16BF,
      /*lda=*/kKvLoraRank,
      /*strideA=*/static_cast<long long>(kKvBHeadDim) * kKvLoraRank,
      q_nope,
      CUDA_R_16BF,
      /*ldb=*/local_heads * kNopeDim,
      /*strideB=*/kNopeDim,
      &beta,
      q_abs_nope,
      CUDA_R_16BF,
      /*ldc=*/local_heads * kKvLoraRank,
      /*strideC=*/kKvLoraRank,
      /*batchCount=*/local_heads,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS ? 0 : static_cast<int>(cudaErrorUnknown);
}

int kimi_mla_v_up_cuda(const DType* kv_b_proj,
                       const DType* latent,
                       DType* output,
                       int batch_size,
                       int local_heads,
                       cudaStream_t stream) {
  if (batch_size <= 0 || local_heads <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (g_cublas_handle == nullptr) {
    return static_cast<int>(cudaErrorInitializationError);
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = cublasSetStream(g_cublas_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return static_cast<int>(cudaErrorInvalidResourceHandle);
  }

  const DType* w_uv = kv_b_proj + static_cast<int64_t>(kNopeDim) * kKvLoraRank;
  if (local_heads == kMlaLtLocalHeads && batch_size <= kMlaLtMaxBatch) {
    // cuBLASLt owns this shape; a failure propagates as an error instead of
    // silently falling back to the reference GEMM below.
    return kimi_mla_v_up_cublaslt_cuda(kv_b_proj, latent, output, batch_size,
                                       local_heads, stream);
  }

  status = cublasGemmStridedBatchedEx(
      g_cublas_handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      /*m=*/kVHeadDim,
      /*n=*/batch_size,
      /*k=*/kKvLoraRank,
      &alpha,
      w_uv,
      CUDA_R_16BF,
      /*lda=*/kKvLoraRank,
      /*strideA=*/static_cast<long long>(kKvBHeadDim) * kKvLoraRank,
      latent,
      CUDA_R_16BF,
      /*ldb=*/local_heads * kKvLoraRank,
      /*strideB=*/kKvLoraRank,
      &beta,
      output,
      CUDA_R_16BF,
      /*ldc=*/local_heads * kVHeadDim,
      /*strideC=*/kVHeadDim,
      /*batchCount=*/local_heads,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS ? 0 : static_cast<int>(cudaErrorUnknown);
}

int kimi_mla_paged_kv_append_cuda(void* ckv_cache,
                                  void* kpe_cache,
                                  IdType* page_indices,
                                  IdType* page_indptr,
                                  IdType* last_page_len,
                                  void* append_ckv,
                                  void* append_kpe,
                                  IdType* batch_indices,
                                  IdType* positions,
                                  int nnz,
                                  int64_t ckv_stride_page,
                                  int64_t ckv_stride_n,
                                  int64_t kpe_stride_page,
                                  int64_t kpe_stride_n,
                                  int page_size,
                                  int batch_size,
                                  cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (nnz <= 0 || page_size <= 0 || batch_size <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  auto paged_kv =
      make_paged_kv_mla(ckv_cache, kpe_cache, page_indices, page_indptr, last_page_len,
                        ckv_stride_page, ckv_stride_n, kpe_stride_page, kpe_stride_n, page_size,
                        batch_size);
  return static_cast<int>(
      flashinfer::AppendPagedKVMlaCache<DType, IdType>(
          paged_kv,
          reinterpret_cast<DType*>(append_ckv),
          reinterpret_cast<DType*>(append_kpe),
          batch_indices,
          positions,
          static_cast<uint32_t>(nnz),
          /*append_ckv_stride_n=*/kKvLoraRank,
          /*append_kpe_stride_n=*/kRopeDim,
          stream));
  PEGAINFER_FFI_GUARD_END(-1)
}

int kimi_flashinfer_batch_decode_mla_cuda(void* q_nope,
                                          void* q_pe,
                                          void* output,
                                          void* ckv_cache,
                                          void* kpe_cache,
                                          IdType* page_indices,
                                          IdType* page_indptr,
                                          IdType* last_page_len,
                                          IdType* request_indices,
                                          IdType* kv_tile_indices,
                                          IdType* kv_chunk_size_ptr,
                                          int num_qo_heads,
                                          int64_t ckv_stride_page,
                                          int64_t ckv_stride_n,
                                          int64_t kpe_stride_page,
                                          int64_t kpe_stride_n,
                                          int page_size,
                                          int batch_size,
                                          float sm_scale,
                                          cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (num_qo_heads <= 0 || page_size <= 0 || batch_size <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  auto paged_kv =
      make_paged_kv_mla(ckv_cache, kpe_cache, page_indices, page_indptr, last_page_len,
                        ckv_stride_page, ckv_stride_n, kpe_stride_page, kpe_stride_n, page_size,
                        batch_size);
  MlaDecodeParamsT params(
      reinterpret_cast<DType*>(q_nope),
      reinterpret_cast<DType*>(q_pe),
      /*q_rope_offset=*/nullptr,
      paged_kv,
      reinterpret_cast<DType*>(output),
      /*lse=*/nullptr,
      static_cast<uint32_t>(num_qo_heads),
      /*window_left=*/-1,
      /*logits_soft_cap=*/0.0f,
      sm_scale,
      /*rope_scale=*/1.0f,
      /*rope_theta=*/1.0e6f);

  params.padded_batch_size = static_cast<uint32_t>(batch_size);
  params.request_indices = request_indices;
  params.kv_tile_indices = kv_tile_indices;
  params.o_indptr = nullptr;
  params.kv_chunk_size_ptr = kv_chunk_size_ptr;
  params.block_valid_mask = nullptr;
  params.partition_kv = false;

  return static_cast<int>(
      flashinfer::BatchDecodeWithPagedKVCacheDispatchedMLA<
          /*HEAD_DIM_CKV=*/kKvLoraRank,
          /*HEAD_DIM_KPE=*/kRopeDim,
          Variant,
          MlaDecodeParamsT>(
          params,
          /*tmp_v=*/nullptr,
          /*tmp_s=*/nullptr,
          /*enable_pdl=*/false,
          stream));
  PEGAINFER_FFI_GUARD_END(-1)
}

}  // extern "C"
