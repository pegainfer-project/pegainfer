// GLM5.2 MLA decode "assembly" kernels: the glue between the projections and the
// FlashMLA sparse decode. Two pieces, both bs=1 decode (one new token):
//
//   1. query assemble — query[H,576] = [ ql_nope(512) | rope(q_pe)(64) ] per head.
//      The nope part is the absorbed q (copied through); the pe part is the
//      interleave RoPE applied to q_pe.
//
//   2. cache pack — one fp8_ds_mla 656-byte cache token =
//      [ 512 e4m3 ckv | 4 f32 group scales | 64 bf16 rope(k_pe) ]. The ckv fp8 +
//      scales come pre-computed from glm52_fp8_per_token_group_quant (amax/448,
//      the same convention the cache wants); this kernel lays them out and applies
//      interleave RoPE to the shared k_pe.
//
// RoPE is interleave-in / block-out (rope_interleave=True): the input pair lives at
// even/odd indices, the output at [i, i+rope/2]. cos/sin are the first half
// (rope_dim/2 = 32) of the position's rotary table. Mirrors the kimi_k2 rope_out
// device function; validated bit-for-bit against the HF oracle (q_rot/k_rot).

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

constexpr int kHeads = 64;
constexpr int kQkNope = 512;      // absorbed ql_nope width
constexpr int kRopeDim = 64;      // q_pe / k_pe width
constexpr int kRopeHalf = 32;     // rope_dim / 2 = cos/sin length used
constexpr int kQueryDim = kQkNope + kRopeDim;  // 576
constexpr int kKvLora = 512;      // ckv width
constexpr int kCacheBytes = 656;  // 512 fp8 + 16 scale + 128 bf16 kpe
constexpr int kPageTokens = 64;   // FlashMLA sparse page: tokens per pool block
constexpr int kFlashInferCacheBytes = 576;  // 512 fp8 ckv + 64 fp8 rope(k_pe)
constexpr int kScaleOffset = 512;
constexpr int kKpeOffset = 528;

// Interleave RoPE of one rope_dim vector: out index r (0..63) reads the pair
// (even=2*pair, odd=2*pair+1), pair = r % 32; the lower half (r<32) is the real
// part, the upper half the imaginary part. cos/sin indexed by pair.
__device__ __forceinline__ __nv_bfloat16 rope_block(const __nv_bfloat16* x, int r,
                                                    const __nv_bfloat16* cos,
                                                    const __nv_bfloat16* sin) {
  const int pair = r % kRopeHalf;
  const bool upper = r >= kRopeHalf;
  const float c = __bfloat162float(cos[pair]);
  const float s = __bfloat162float(sin[pair]);
  const float even = __bfloat162float(x[2 * pair]);
  const float odd = __bfloat162float(x[2 * pair + 1]);
  const float v = upper ? (odd * c + even * s) : (even * c - odd * s);
  return __float2bfloat16(v);
}

__device__ __forceinline__ unsigned char quantize_static_e4m3(float value) {
  value = fminf(fmaxf(value, -448.0f), 448.0f);
  return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

constexpr int kScaleGroups = 4;  // KvLora / 128

// q_pe lives at `q_pe_base[q_pe_offset + h*q_pe_head_stride + ...]`: contiguous
// [T,H,64] (offset 0, stride 64) when split out, or embedded in the q_b output
// [T,H,256] (offset 192, stride 256) in the fused forward. The per-token q_pe
// stride is num_q_heads * q_pe_head_stride in both layouts. cos/sin carry one
// [32] row PER TOKEN (each token sits at its own position).
//
// num_q_heads may be a head-parallel shard (attention-TP: 8 of 64). The
// ql_nope / q_pe inputs are COMPACT [T, num_q_heads, .]; the query output
// stays FULL-WIDTH [T, kHeads, 576] — the FlashMLA sparse kernel only runs
// its h_q % 64 == 0 shape, so shard heads occupy slots 0..num_q_heads and the
// pad slots keep their zero fill (zero query -> uniform softmax -> finite
// latent, discarded by the shard-sized W_UV batch downstream).
__global__ void glm52_mla_query_assemble_kernel(
    const __nv_bfloat16* __restrict__ ql_nope,    // [T, num_q_heads, 512]
    const __nv_bfloat16* __restrict__ q_pe_base,  // q_pe at offset/stride
    int q_pe_offset, int q_pe_head_stride, int num_q_heads,
    const __nv_bfloat16* __restrict__ cos,        // [T, 32]
    const __nv_bfloat16* __restrict__ sin,        // [T, 32]
    __nv_bfloat16* __restrict__ query) {          // [T, kHeads, 576]
  const int h = blockIdx.x;
  const int t = blockIdx.y;
  if (h >= num_q_heads) return;
  const __nv_bfloat16* q_pe = q_pe_base + q_pe_offset +
                              (size_t)t * num_q_heads * q_pe_head_stride +
                              h * q_pe_head_stride;
  const __nv_bfloat16* cos_t = cos + (size_t)t * kRopeHalf;
  const __nv_bfloat16* sin_t = sin + (size_t)t * kRopeHalf;
  const __nv_bfloat16* ql_t = ql_nope + (size_t)t * num_q_heads * kQkNope;
  __nv_bfloat16* query_t = query + (size_t)t * kHeads * kQueryDim;
  for (int i = threadIdx.x; i < kQueryDim; i += blockDim.x) {
    if (i < kQkNope) {
      query_t[h * kQueryDim + i] = ql_t[h * kQkNope + i];
    } else {
      const int r = i - kQkNope;  // 0..63
      query_t[h * kQueryDim + i] = rope_block(q_pe, r, cos_t, sin_t);
    }
  }
}

__global__ void glm52_mla_cache_pack_kernel(
    const unsigned char* __restrict__ ckv_fp8,   // [T, 512]
    const float* __restrict__ ckv_scales,        // [T, 4]
    const __nv_bfloat16* __restrict__ k_pe,      // [T, 64] pre-rope
    const __nv_bfloat16* __restrict__ cos,       // [T, 32]
    const __nv_bfloat16* __restrict__ sin,       // [T, 32]
    unsigned char* __restrict__ cache,           // [max_slots/64 pages, stride]
    const long long* __restrict__ slot_mapping,  // [T] write slots
    long long max_slots, long long block_stride_bytes) {
  // The write slot comes from device memory so the launch is CUDA-graph
  // replayable (a host scalar would bake the capture step's position into
  // the graph). Out-of-window slots trap: a silent modulo/clamp would stomp
  // a live cache line.
  const int t = blockIdx.x;
  const long long slot = slot_mapping[t];
  if (slot < 0 || slot >= max_slots) {
    __trap();
  }
  // Page-first slab: consecutive 64-token pages sit block_stride_bytes apart
  // (tight layout passes 64 * kCacheBytes and reduces to slot * kCacheBytes).
  unsigned char* __restrict__ cache_token =
      cache + (slot / kPageTokens) * block_stride_bytes +
      (slot % kPageTokens) * static_cast<long long>(kCacheBytes);
  const unsigned char* ckv_t = ckv_fp8 + (size_t)t * kKvLora;
  const float* scales_t = ckv_scales + (size_t)t * kScaleGroups;
  const __nv_bfloat16* k_pe_t = k_pe + (size_t)t * kRopeDim;
  const __nv_bfloat16* cos_t = cos + (size_t)t * kRopeHalf;
  const __nv_bfloat16* sin_t = sin + (size_t)t * kRopeHalf;
  const int tid = threadIdx.x;
  // 512 e4m3 ckv
  for (int i = tid; i < kKvLora; i += blockDim.x) {
    cache_token[i] = ckv_t[i];
  }
  // 4 f32 group scales at byte 512 (slot base is 4-aligned: 656 % 4 == 0)
  if (tid < kScaleGroups) {
    reinterpret_cast<float*>(cache_token + kScaleOffset)[tid] = scales_t[tid];
  }
  // 64 bf16 rope(k_pe) at byte 528
  __nv_bfloat16* kpe_out = reinterpret_cast<__nv_bfloat16*>(cache_token + kKpeOffset);
  for (int r = tid; r < kRopeDim; r += blockDim.x) {
    kpe_out[r] = rope_block(k_pe_t, r, cos_t, sin_t);
  }
}

// Fused FlashInfer front pack: one launch replaces the per-layer
// ckv_split -> kv_a RMSNorm -> query assemble -> cache pack chain
// (4 graph nodes -> 1). Grid (num_q_heads + 1, T), block (32, 2):
//   blockIdx.x <  num_q_heads : query assemble + static-e4m3 quantize
//                               (elementwise).
//   blockIdx.x == num_q_heads : kv_a RMSNorm over ckv_raw[t, 0..512) with the
//                               EXACT FlashInfer RMSNormKernel d=512 reduction
//                               (vec8 x 64 threads, shfl_xor 16..1, cross-warp
//                               smem, rsqrtf(sum/512+eps)), then the 576-byte
//                               cache pack. The normed value is rounded to
//                               bf16 BEFORE e4m3 quantization, preserving the
//                               unfused chain's double rounding bit-for-bit
//                               (validated byte-identical over randomized
//                               trials against the production chain with the
//                               vendored flashinfer::norm::RMSNorm as
//                               reference).
__global__ void glm52_mla_front_pack_fp8_kernel(
    const __nv_bfloat16* __restrict__ ql_nope,
    const __nv_bfloat16* __restrict__ q_pe_base, int q_pe_offset,
    int q_pe_head_stride, int num_q_heads,
    const __nv_bfloat16* __restrict__ ckv_raw,      // [T, 576] kv_a output
    const __nv_bfloat16* __restrict__ norm_weight,  // [512] kv_a_ln
    float eps,
    const __nv_bfloat16* __restrict__ cos, const __nv_bfloat16* __restrict__ sin,
    unsigned char* __restrict__ query,              // [T, num_q_heads, 576]
    unsigned char* __restrict__ cache,              // [max_slots, 576]
    const long long* __restrict__ slot_mapping, long long max_slots) {
  const int t = blockIdx.y;
  const __nv_bfloat16* cos_t = cos + (size_t)t * kRopeHalf;
  const __nv_bfloat16* sin_t = sin + (size_t)t * kRopeHalf;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int thread_id = tx + ty * 32;

  if ((int)blockIdx.x < num_q_heads) {
    const int h = blockIdx.x;
    const __nv_bfloat16* q_pe = q_pe_base + q_pe_offset +
                                (size_t)t * num_q_heads * q_pe_head_stride +
                                h * q_pe_head_stride;
    const __nv_bfloat16* ql_t = ql_nope + ((size_t)t * num_q_heads + h) * kQkNope;
    unsigned char* query_t = query + ((size_t)t * num_q_heads + h) * kQueryDim;
    for (int i = thread_id; i < kQueryDim; i += 64) {
      float value = i < kQkNope
                        ? __bfloat162float(ql_t[i])
                        : __bfloat162float(rope_block(q_pe, i - kQkNope, cos_t, sin_t));
      query_t[i] = quantize_static_e4m3(value);
    }
    return;
  }

  // ---- kv role ----
  const long long slot = slot_mapping[t];
  if (slot < 0 || slot >= max_slots) __trap();
  const __nv_bfloat16* row = ckv_raw + (size_t)t * (kKvLora + kRopeDim);
  __shared__ float smem[2];

  float sum_sq = 0.f;
  {
    const __nv_bfloat16* v = row + thread_id * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float value = __bfloat162float(v[j]);
      sum_sq += value * value;
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
  }
  if (tx == 0) smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < 2) ? smem[tx] : 0.f;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();
  const float rms_rcp = rsqrtf(smem[0] / float(kKvLora) + eps);

  unsigned char* cache_token =
      cache + slot * static_cast<long long>(kFlashInferCacheBytes);
  for (int i = thread_id; i < kFlashInferCacheBytes; i += 64) {
    float value;
    if (i < kKvLora) {
      float normed = __bfloat162float(row[i]) * rms_rcp *
                     (0.f + __bfloat162float(norm_weight[i]));
      value = __bfloat162float(__float2bfloat16(normed));
    } else {
      value = __bfloat162float(rope_block(row + kKvLora, i - kKvLora, cos_t, sin_t));
    }
    cache_token[i] = quantize_static_e4m3(value);
  }
}

// Split the kv_a projection output [T, 576] into the contiguous kv_c [T, 512]
// (pre-norm compressed kv) and k_pe [T, 64] (pre-rope shared key) the decode
// chain consumes — replaces the per-token dtod slice copies that don't batch.
__global__ void glm52_mla_ckv_split_kernel(
    const __nv_bfloat16* __restrict__ ckv,  // [T, 576]
    __nv_bfloat16* __restrict__ kv_c,       // [T, 512]
    __nv_bfloat16* __restrict__ k_pe) {     // [T, 64]
  const int t = blockIdx.x;
  const __nv_bfloat16* row = ckv + (size_t)t * (kKvLora + kRopeDim);
  for (int i = threadIdx.x; i < kKvLora + kRopeDim; i += blockDim.x) {
    if (i < kKvLora) {
      kv_c[(size_t)t * kKvLora + i] = row[i];
    } else {
      k_pe[(size_t)t * kRopeDim + (i - kKvLora)] = row[i];
    }
  }
}

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

}  // namespace

extern "C" {

CUresult glm52_mla_query_assemble_cuda(const __nv_bfloat16* ql_nope,
                                       const __nv_bfloat16* q_pe_base,
                                       int q_pe_offset, int q_pe_head_stride,
                                       int num_q_heads,
                                       const __nv_bfloat16* cos,
                                       const __nv_bfloat16* sin,
                                       __nv_bfloat16* query, int tokens,
                                       cudaStream_t stream) {
  if (ql_nope == nullptr || q_pe_base == nullptr || cos == nullptr ||
      sin == nullptr || query == nullptr || tokens <= 0 || num_q_heads <= 0 ||
      num_q_heads > kHeads) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const dim3 grid(num_q_heads, tokens, 1);
  glm52_mla_query_assemble_kernel<<<grid, 192, 0, stream>>>(
      ql_nope, q_pe_base, q_pe_offset, q_pe_head_stride, num_q_heads, cos, sin,
      query);
  return consume_last_cuda_error();
}

CUresult glm52_mla_cache_pack_cuda(const unsigned char* ckv_fp8,
                                   const float* ckv_scales,
                                   const __nv_bfloat16* k_pe,
                                   const __nv_bfloat16* cos,
                                   const __nv_bfloat16* sin,
                                   unsigned char* cache,
                                   const long long* slot_mapping,
                                   long long max_slots,
                                   long long block_stride_bytes, int tokens,
                                   cudaStream_t stream) {
  if (ckv_fp8 == nullptr || ckv_scales == nullptr || k_pe == nullptr ||
      cos == nullptr || sin == nullptr || cache == nullptr ||
      slot_mapping == nullptr || max_slots <= 0 || tokens <= 0 ||
      block_stride_bytes < (long long)kPageTokens * kCacheBytes) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  glm52_mla_cache_pack_kernel<<<tokens, 128, 0, stream>>>(
      ckv_fp8, ckv_scales, k_pe, cos, sin, cache, slot_mapping, max_slots,
      block_stride_bytes);
  return consume_last_cuda_error();
}

CUresult glm52_mla_front_pack_fp8_cuda(
    const __nv_bfloat16* ql_nope, const __nv_bfloat16* q_pe_base,
    int q_pe_offset, int q_pe_head_stride, int num_q_heads,
    const __nv_bfloat16* ckv_raw, const __nv_bfloat16* norm_weight, float eps,
    const __nv_bfloat16* cos, const __nv_bfloat16* sin, unsigned char* query,
    unsigned char* cache, const long long* slot_mapping, long long max_slots,
    int tokens, cudaStream_t stream) {
  if (ql_nope == nullptr || q_pe_base == nullptr || ckv_raw == nullptr ||
      norm_weight == nullptr || cos == nullptr || sin == nullptr ||
      query == nullptr || cache == nullptr || slot_mapping == nullptr ||
      max_slots <= 0 || tokens <= 0 || num_q_heads <= 0 ||
      num_q_heads > kHeads) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const dim3 grid(num_q_heads + 1, tokens, 1);
  const dim3 block(32, 2, 1);
  glm52_mla_front_pack_fp8_kernel<<<grid, block, 0, stream>>>(
      ql_nope, q_pe_base, q_pe_offset, q_pe_head_stride, num_q_heads, ckv_raw,
      norm_weight, eps, cos, sin, query, cache, slot_mapping, max_slots);
  return consume_last_cuda_error();
}

CUresult glm52_mla_ckv_split_cuda(const __nv_bfloat16* ckv, __nv_bfloat16* kv_c,
                                  __nv_bfloat16* k_pe, int tokens,
                                  cudaStream_t stream) {
  if (ckv == nullptr || kv_c == nullptr || k_pe == nullptr || tokens <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  glm52_mla_ckv_split_kernel<<<tokens, 192, 0, stream>>>(ckv, kv_c, k_pe);
  return consume_last_cuda_error();
}

}  // extern "C"
