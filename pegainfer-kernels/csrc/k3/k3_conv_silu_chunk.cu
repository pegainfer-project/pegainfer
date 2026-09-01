// Kimi-K3 chunked-prefill causal conv + silu over one q/k/v stream, walking
// the segment's rows in place.
//
// The chunk path used to land the projection to bf16, build every row's
// 3-slot window with strided device copies (row t's slot j <- landed row
// t-3+j, or the carried window for the first rows), run the batched decode
// conv over the materialized windows, and copy the last commit row's successor
// window out as the carry. The CP8 128k anatomy charged that to 182 ms of
// conv plus 235 ms of cuMemcpy2DAsync per deep rank. This kernel reads the
// f32 partial rows t-3..t directly (the neighbours hit L2), takes the first
// rows' missing inputs from the carried window, and writes the carry itself.
//
// The arithmetic is the batched conv kernel's spelling, term for term:
//
//   xb      = bf16(0 + P[r])                          (the landing)
//   ca      = 0
//   ca      = ca + (f32(w_j) * Cw[j])   for j = 0, 1, 2 (oldest first)
//   ca      = ca + (f32(xb_t) * Cw[3])
//   sb      = f32(bf16(ca))
//   Y[t]    = bf16(sb * (1 / (1 + expf(0 - sb))))
//   next[j] = w at position commit_rows - 3 + j       (successor window)
//
// where w at position r is xb_r for r >= 0 and carry[r + 3] before the
// segment. Written as the same expressions so nvcc contracts the same
// products into the same FMAs; compiled with the same -O3 and no fast-math.
// Eight columns per thread with 16-byte loads and stores, a run of rows per
// block with the taps in registers and the window sliding through them; the
// launcher guarantees inner % 8 == 0 and 16-byte-aligned rows. Deterministic
// and CUDA-graph safe.

#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;
constexpr int kVec = 8;
constexpr int kState = 3;  // K3_CONV_WIDTH - 1
constexpr int kRowsPerBlock = 16;

struct Vec8 {
  float v[kVec];
};

__device__ __forceinline__ Vec8 land8(const float* __restrict__ p) {
  const float4 lo = *reinterpret_cast<const float4*>(p);
  const float4 hi = *reinterpret_cast<const float4*>(p + 4);
  const float raw[kVec] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
  Vec8 out;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    float xa = 0.0f;
    xa = xa + raw[i];
    out.v[i] = __bfloat162float(__float2bfloat16_rn(xa));
  }
  return out;
}

__device__ __forceinline__ Vec8 load8(const __nv_bfloat16* __restrict__ p) {
  const uint4 packed = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
  Vec8 out;
#pragma unroll
  for (int i = 0; i < kVec / 2; ++i) {
    const float2 f = __bfloat1622float2(pairs[i]);
    out.v[2 * i] = f.x;
    out.v[2 * i + 1] = f.y;
  }
  return out;
}

__device__ __forceinline__ void store8(__nv_bfloat16* __restrict__ p,
                                       const Vec8& x) {
  __nv_bfloat162 packed[kVec / 2];
#pragma unroll
  for (int i = 0; i < kVec / 2; ++i) {
    packed[i] = __floats2bfloat162_rn(x.v[2 * i], x.v[2 * i + 1]);
  }
  *reinterpret_cast<uint4*>(p) = *reinterpret_cast<const uint4*>(packed);
}

__device__ __forceinline__ Vec8 taps8(const float* __restrict__ p) {
  const float4 lo = *reinterpret_cast<const float4*>(p);
  const float4 hi = *reinterpret_cast<const float4*>(p + 4);
  return Vec8{{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w}};
}

// Each block owns one 8-column-per-thread column slice and a run of
// kRowsPerBlock consecutive rows: the taps live in registers and the window
// slides through them, so every partial element is read once and every
// output written once (the f32 read and the bf16 write are the whole HBM
// story; the taps and the carry come from L2 once per block).
__global__ void __launch_bounds__(kThreads)
    conv_silu_chunk_kernel(const float* __restrict__ P,
                           const float* __restrict__ Cw,
                           const __nv_bfloat16* __restrict__ carry,
                           __nv_bfloat16* __restrict__ Y,
                           __nv_bfloat16* __restrict__ next, int tokens,
                           int commit_rows, int inner) {
  const int c = (blockIdx.x * kThreads + threadIdx.x) * kVec;
  if (c >= inner) return;
  const int t0 = blockIdx.y * kRowsPerBlock;
  const int t1 = min(t0 + kRowsPerBlock, tokens);

  Vec8 cw[kState + 1];
#pragma unroll
  for (int j = 0; j <= kState; ++j) {
    cw[j] = taps8(Cw + (size_t)j * inner + c);
  }

  // The window entering row t0: positions t0-3 .. t0-1, the segment's own
  // rows once they exist, the carried window before it.
  Vec8 w[kState + 1];
#pragma unroll
  for (int j = 0; j < kState; ++j) {
    const int r = t0 - kState + j;
    w[j] = r >= 0 ? land8(P + (size_t)r * inner + c)
                  : load8(carry + (size_t)(r + kState) * inner + c);
  }

  for (int t = t0; t < t1; ++t) {
    w[kState] = land8(P + (size_t)t * inner + c);
    Vec8 y;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      float ca = 0.0f;
#pragma unroll
      for (int j = 0; j < kState; ++j) {
        ca = ca + (w[j].v[i] * cw[j].v[i]);
      }
      ca = ca + (w[kState].v[i] * cw[kState].v[i]);
      const float sb = __bfloat162float(__float2bfloat16_rn(ca));
      y.v[i] = sb * (1.0f / (1.0f + expf(0.0f - sb)));
    }
    store8(Y + (size_t)t * inner + c, y);
    // The carry into the next segment: the successor window of the last
    // commit row, i.e. the inputs at positions commit_rows-3 .. commit_rows-1.
    if (next != nullptr && t == commit_rows - 1) {
#pragma unroll
      for (int j = 0; j < kState; ++j) {
        store8(next + (size_t)j * inner + c, w[j + 1]);
      }
    }
#pragma unroll
    for (int j = 0; j < kState; ++j) {
      w[j] = w[j + 1];
    }
  }
}

CUresult map_cuda_error(cudaError_t err) {
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue) return CUDA_ERROR_INVALID_VALUE;
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  if (err == cudaErrorNotSupported) return CUDA_ERROR_NOT_SUPPORTED;
  return CUDA_ERROR_LAUNCH_FAILED;
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

}  // namespace

extern "C" {

// One q/k/v stream of one segment of `tokens` consecutive rows: `p [tokens,
// inner]` f32 partial (the segment's rows), `cw [4, inner]` f32 taps, `carry
// [3, inner]` bf16 window preceding the segment, `y [tokens, inner]` bf16
// conv+silu output. When `next` is non-null (commit_rows >= 1) it receives
// the `[3, inner]` window carrying into the segment after row commit_rows-1.
// Shapes are runtime values.
CUresult k3_conv_silu_chunk_cuda(const float* p, const float* cw,
                                 const void* carry, void* y, void* next,
                                 int tokens, int commit_rows, int inner,
                                 cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (p == nullptr || cw == nullptr || carry == nullptr || y == nullptr ||
      tokens <= 0 || commit_rows < 0 || commit_rows > tokens || inner <= 0 ||
      inner % kVec != 0 || !aligned16(p) || !aligned16(cw) ||
      !aligned16(carry) || !aligned16(y) || !aligned16(next) ||
      (next == nullptr) != (commit_rows == 0)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const dim3 grid((inner / kVec + kThreads - 1) / kThreads,
                  (tokens + kRowsPerBlock - 1) / kRowsPerBlock);
  conv_silu_chunk_kernel<<<grid, kThreads, 0, stream>>>(
      p, cw, static_cast<const __nv_bfloat16*>(carry),
      static_cast<__nv_bfloat16*>(y), static_cast<__nv_bfloat16*>(next),
      tokens, commit_rows, inner);
  return map_cuda_error(cudaGetLastError());
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

}  // extern "C"
