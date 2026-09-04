// Kimi-K3 matmul landing: merge the column span [off, off+n) of each row's
// [split_k, nt] f32 partial and land bf16 once.
//
// Replaces the retired TileLang `land_batched` kernel, which landed one
// element per thread with 2-byte stores and reached 2.3 TB/s on the 12288-wide
// chunked-prefill landings (nsys, CP8 128k anatomy: 385 ms of a 6.1 s deep
// rank). Each thread here lands 8 consecutive columns — two float4 loads per
// segment, one 16-byte store — so the pass streams at HBM rate.
//
// The arithmetic is the retired kernel's spelling: the segments are summed in
// f32 in ascending s order onto a zero accumulator (so a -0 partial lands as
// +0, as it did), then cast once, round-to-nearest-even. At split_k = 1 — the only
// launch site — that is the bare cast, so the landing is bit-identical to the
// retired kernel and the certified single-row spelling.
//
// The vector path needs every row 16-byte aligned in both tensors: nt and off
// multiples of 4, n a multiple of 8. Every K3 span satisfies this; anything
// else takes the scalar path. Batch is a runtime value — no per-bucket
// instantiation. Deterministic and CUDA-graph safe.

#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;
constexpr int kVec = 8;
constexpr long long kMaxBlocks = 1 << 20;

__device__ __forceinline__ void add4(float4& a, const float4& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

__global__ void land_vec8_kernel(const float* __restrict__ P,
                                 __nv_bfloat16* __restrict__ O, int nt, int n,
                                 int off, int split_k, long long total_vec) {
  const int vec_per_row = n / kVec;
  for (long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       v < total_vec; v += (long long)gridDim.x * blockDim.x) {
    const long long row = v / vec_per_row;
    const int col = (int)(v - row * vec_per_row) * kVec;
    const float* src = P + row * (long long)split_k * nt + off + col;
    float4 lo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 hi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    add4(lo, *reinterpret_cast<const float4*>(src));
    add4(hi, *reinterpret_cast<const float4*>(src + 4));
    for (int s = 1; s < split_k; ++s) {
      src += nt;
      add4(lo, *reinterpret_cast<const float4*>(src));
      add4(hi, *reinterpret_cast<const float4*>(src + 4));
    }
    __nv_bfloat162 packed[4];
    packed[0] = __floats2bfloat162_rn(lo.x, lo.y);
    packed[1] = __floats2bfloat162_rn(lo.z, lo.w);
    packed[2] = __floats2bfloat162_rn(hi.x, hi.y);
    packed[3] = __floats2bfloat162_rn(hi.z, hi.w);
    *reinterpret_cast<uint4*>(O + row * (long long)n + col) =
        *reinterpret_cast<const uint4*>(packed);
  }
}

__global__ void land_scalar_kernel(const float* __restrict__ P,
                                   __nv_bfloat16* __restrict__ O, int nt, int n,
                                   int off, int split_k, long long total) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       i < total; i += (long long)gridDim.x * blockDim.x) {
    const long long row = i / n;
    const int col = (int)(i - row * n);
    const float* src = P + row * (long long)split_k * nt + off + col;
    float acc = 0.0f;
    acc = acc + src[0];
    for (int s = 1; s < split_k; ++s) {
      acc += src[(long long)s * nt];
    }
    O[i] = __float2bfloat16_rn(acc);
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

// Merge the column span [off, off+n) of each row's [split_k, nt] f32 partial
// `p [b, split_k, nt]` and land `o [b, n]` bf16 once. Shapes are runtime
// values; split_k = 1 is the single partial a framework GEMM produces, where
// the merge degenerates to the slice and the cast.
CUresult k3_land_cuda(const float* p, void* o, int b, int nt, int n, int off,
                      int split_k, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (p == nullptr || o == nullptr || b <= 0 || nt <= 0 || n <= 0 || off < 0 ||
      split_k <= 0 || off + n > nt) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  __nv_bfloat16* out = static_cast<__nv_bfloat16*>(o);
  const bool vec = aligned16(p) && aligned16(o) && nt % 4 == 0 &&
                   off % 4 == 0 && n % kVec == 0;
  if (vec) {
    const long long total = (long long)b * (n / kVec);
    const long long blocks = (total + kThreads - 1) / kThreads;
    land_vec8_kernel<<<(unsigned)(blocks < kMaxBlocks ? blocks : kMaxBlocks),
                       kThreads, 0, stream>>>(p, out, nt, n, off, split_k,
                                              total);
  } else {
    const long long total = (long long)b * n;
    const long long blocks = (total + kThreads - 1) / kThreads;
    land_scalar_kernel<<<(unsigned)(blocks < kMaxBlocks ? blocks : kMaxBlocks),
                         kThreads, 0, stream>>>(p, out, nt, n, off, split_k,
                                                total);
  }
  return map_cuda_error(cudaGetLastError());
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

}  // extern "C"
