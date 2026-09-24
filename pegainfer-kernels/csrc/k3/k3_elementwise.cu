// Kimi-K3 bf16 elementwise family, eight columns per thread.
//
// Hand-written replacements for the retired TileLang `add2_batched`,
// `mul_sigmoid_batched`, `situ_batched` and `o_norm_gate_batched` kernels.
// Those walked one element per thread with 2-byte loads and stores and ran at
// 1–2 TB/s on the chunked-prefill rows (16896 x 7168..33792); every launch
// here moves 16 bytes per thread per operand.
//
// Each kernel keeps the retired kernel's arithmetic term for term. TileLang's
// `bfloat16_t` is cutlass's: its `+` is `__hadd`, its `*` is `__hmul`, and
// the bf16 cast is `cvt.rn.bf16.f32` — so the bf16x2 forms below (`add.rn`,
// `mul.rn`, `cvt.rn` per lane) round identically. The f32 chains use the
// same `expf` / `tanhf` / `rsqrtf` and IEEE division, and both compile under
// the same `-O3` without fast-math, so every landing is bit-identical.
//
// `o_norm_gate` reduces 128 squares per (row, head). The retired kernel used
// TileLang's xor butterfly (offsets 64 and 32 through shared memory, then
// 16..1 by shuffle), whose value is the same in every lane; sixteen lanes
// holding eight columns each reproduce that tree exactly — lane xor 8/4/2/1
// pairs columns 64/32/16/8 apart, then slots j^4/j^2/j^1 pair the rest —
// so the norm scale matches bit for bit.

#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kVec = 8;
constexpr int kThreads = 256;
constexpr int kHeadDim = 128;
constexpr int kLanes = kHeadDim / kVec;   // lanes per (row, head)
constexpr int kRowsPerBlock = kThreads / kLanes;

__device__ __forceinline__ void unpack8(uint4 v, float f[kVec]) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float2 t = __bfloat1622float2(p[j]);
    f[2 * j] = t.x;
    f[2 * j + 1] = t.y;
  }
}

__device__ __forceinline__ uint4 pack8(const float f[kVec]) {
  uint4 v;
  __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    p[j] = __floats2bfloat162_rn(f[2 * j], f[2 * j + 1]);
  }
  return v;
}

__device__ __forceinline__ float sigmoid_f32(float x) {
  return 1.0f / (1.0f + expf(0.0f - x));
}

// O = A + Bt in bf16 addition.
__global__ void add2_kernel(const uint4* __restrict__ a, const uint4* __restrict__ bt,
                            uint4* __restrict__ o, long long total_vec) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_vec) {
    return;
  }
  uint4 va = a[idx];
  uint4 vb = bt[idx];
  uint4 vo;
  const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&va);
  const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&vb);
  __nv_bfloat162* po = reinterpret_cast<__nv_bfloat162*>(&vo);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    po[j] = __hadd2(pa[j], pb[j]);
  }
  o[idx] = vo;
}

// O = A * bf16(sigmoid(Bt)): the sigmoid in f32, landed bf16, then the bf16
// product.
__global__ void mul_sigmoid_kernel(const uint4* __restrict__ a, const uint4* __restrict__ bt,
                                   uint4* __restrict__ o, long long total_vec) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_vec) {
    return;
  }
  uint4 va = a[idx];
  float b[kVec];
  unpack8(bt[idx], b);
  float s[kVec];
#pragma unroll
  for (int j = 0; j < kVec; ++j) {
    s[j] = sigmoid_f32(b[j]);
  }
  uint4 vs = pack8(s);
  uint4 vo;
  const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&va);
  const __nv_bfloat162* ps = reinterpret_cast<const __nv_bfloat162*>(&vs);
  __nv_bfloat162* po = reinterpret_cast<__nv_bfloat162*>(&vo);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    po[j] = __hmul2(pa[j], ps[j]);
  }
  o[idx] = vo;
}

// O = bf16(4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25)), the f32 chain landed once.
__global__ void situ_kernel(const uint4* __restrict__ g, const uint4* __restrict__ u,
                            uint4* __restrict__ o, long long total_vec) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_vec) {
    return;
  }
  float gf[kVec];
  float uf[kVec];
  unpack8(g[idx], gf);
  unpack8(u[idx], uf);
  float r[kVec];
#pragma unroll
  for (int j = 0; j < kVec; ++j) {
    r[j] = ((4.0f * tanhf(gf[j] / 4.0f)) * sigmoid_f32(gf[j])) * (25.0f * tanhf(uf[j] / 25.0f));
  }
  o[idx] = pack8(r);
}

// Out = bf16(x * rsqrt(mean(x^2) + eps) * go) * bf16(sigmoid(g2)) per
// (row, head) of 128 columns. `rows` counts (row, head) pairs; a lane past
// the last pair clamps its loads and skips its store so the shuffles stay
// converged.
__global__ void o_norm_gate_kernel(const uint4* __restrict__ x, const uint4* __restrict__ g2,
                                   const float4* __restrict__ go, uint4* __restrict__ out,
                                   long long rows, float eps) {
  long long row = (long long)blockIdx.x * kRowsPerBlock + threadIdx.x / kLanes;
  int lane = threadIdx.x % kLanes;
  bool live = row < rows;
  long long vec = (live ? row : rows - 1) * kLanes + lane;
  float xf[kVec];
  unpack8(x[vec], xf);
  float s[kVec];
#pragma unroll
  for (int j = 0; j < kVec; ++j) {
    float sq = xf[j] * xf[j];
    s[j] = 0.0f + sq;
  }
#pragma unroll
  for (int off = kLanes / 2; off >= 1; off >>= 1) {
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      s[j] = s[j] + __shfl_xor_sync(0xffffffffu, s[j], off);
    }
  }
  float t[kVec];
#pragma unroll
  for (int j = 0; j < kVec; ++j) {
    t[j] = s[j] + s[j ^ 4];
  }
#pragma unroll
  for (int j = 0; j < kVec; ++j) {
    s[j] = t[j] + t[j ^ 2];
  }
#pragma unroll
  for (int j = 0; j < kVec; ++j) {
    t[j] = s[j] + s[j ^ 1];
  }
  float atot = t[0];
  float scale = rsqrtf((atot / 128.0f) + eps);
  float4 go0 = go[lane * 2];
  float4 go1 = go[lane * 2 + 1];
  float gof[kVec] = {go0.x, go0.y, go0.z, go0.w, go1.x, go1.y, go1.z, go1.w};
  float g2f[kVec];
  unpack8(g2[vec], g2f);
  float n[kVec];
  float sg[kVec];
#pragma unroll
  for (int j = 0; j < kVec; ++j) {
    n[j] = (xf[j] * scale) * gof[j];
    sg[j] = sigmoid_f32(g2f[j]);
  }
  uint4 vn = pack8(n);
  uint4 vg = pack8(sg);
  uint4 vo;
  const __nv_bfloat162* pn = reinterpret_cast<const __nv_bfloat162*>(&vn);
  const __nv_bfloat162* pg = reinterpret_cast<const __nv_bfloat162*>(&vg);
  __nv_bfloat162* po = reinterpret_cast<__nv_bfloat162*>(&vo);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    po[j] = __hmul2(pn[j], pg[j]);
  }
  if (live) {
    out[vec] = vo;
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

template <typename Kernel>
CUresult launch_binary(Kernel kernel, const void* a, const void* bt, void* o, int b, int n,
                       cudaStream_t stream) {
  if (a == nullptr || bt == nullptr || o == nullptr || b <= 0 || n <= 0 || n % kVec != 0 ||
      !aligned16(a) || !aligned16(bt) || !aligned16(o)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  long long total_vec = (long long)b * n / kVec;
  long long blocks = (total_vec + kThreads - 1) / kThreads;
  kernel<<<(unsigned)blocks, kThreads, 0, stream>>>(
      static_cast<const uint4*>(a), static_cast<const uint4*>(bt), static_cast<uint4*>(o),
      total_vec);
  return map_cuda_error(cudaGetLastError());
}

}  // namespace

extern "C" {

// O = A + Bt in bf16 addition, all [b, n]. n must be a multiple of 8.
CUresult k3_add2_cuda(const void* a, const void* bt, void* o, int b, int n,
                      cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  return launch_binary(add2_kernel, a, bt, o, b, n, stream);
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

// O = A * bf16(sigmoid(Bt)), all [b, n]. n must be a multiple of 8.
CUresult k3_mul_sigmoid_cuda(const void* a, const void* bt, void* o, int b, int n,
                             cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  return launch_binary(mul_sigmoid_kernel, a, bt, o, b, n, stream);
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

// O = situ(G, U), all [b, n]. n must be a multiple of 8.
CUresult k3_situ_cuda(const void* g, const void* u, void* o, int b, int n,
                      cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  return launch_binary(situ_kernel, g, u, o, b, n, stream);
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

// Out[b, heads * 128] = bf16(rms_norm(X) * Go) * bf16(sigmoid(G2)) per
// (row, head); Go is [128] f32. head_dim must be 128 (the reduction tree is
// the 128-wide one).
CUresult k3_o_norm_gate_cuda(const void* x, const void* g2, const float* go, void* out, int b,
                             int heads, int head_dim, float eps, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (x == nullptr || g2 == nullptr || go == nullptr || out == nullptr || b <= 0 ||
      heads <= 0 || head_dim != kHeadDim || !aligned16(x) || !aligned16(g2) ||
      !aligned16(go) || !aligned16(out)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  long long rows = (long long)b * heads;
  long long blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
  o_norm_gate_kernel<<<(unsigned)blocks, kThreads, 0, stream>>>(
      static_cast<const uint4*>(x), static_cast<const uint4*>(g2),
      reinterpret_cast<const float4*>(go), static_cast<uint4*>(out), rows, eps);
  return map_cuda_error(cudaGetLastError());
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

}  // extern "C"
