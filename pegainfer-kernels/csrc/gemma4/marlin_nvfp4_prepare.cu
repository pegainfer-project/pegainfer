// NVFP4 block scales, in the form Marlin reads them.
//
// The checkpoint stores one e4m3 scale per sixteen values, row-major over the
// projection's output. Marlin wants them transposed, permuted into its 64-wide
// tile order, and re-encoded: it reads the byte as S0E5M3 rather than e4m3, so
// a scale is multiplied by 2^7 and shifted up one bit, which buys an exponent
// bias near zero at dequantization time. The caller folds the same 2^7 and the
// rescaling factor back into the per-tensor scale.
//
// The reference this must agree with is vLLM's `nvfp4_marlin_process_scales`.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kGroup = 16;
constexpr int kPerm = 64;

/// Marlin's scale order within a 64-wide run: eight interleaved lanes of
/// eight, which is the transpose of an 8x8 block.
__device__ __forceinline__ int scale_perm_64(int at) {
  return (at % 8) * 8 + at / 8;
}

/// The four-element swizzle the fp8 dequantization expects on top of the
/// permutation.
__device__ __forceinline__ int fp8_pair_swizzle(int at) {
  constexpr int order[4] = {0, 2, 1, 3};
  return (at & ~3) + order[at & 3];
}

__global__ void nvfp4_prepare_scales_kernel(
    const unsigned char *__restrict__ checkpoint,
    unsigned char *__restrict__ prepared, int out_dim, int scale_k,
    float rescale, long long total) {
  long long at = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (at >= total) {
    return;
  }
  const long long per_expert = (long long)out_dim * scale_k;
  const long long expert_base = at - at % per_expert;
  const long long within = at % per_expert;

  // Undo the two reorderings to find which checkpoint scale lands here.
  const long long run = within / kPerm;
  const int slot = fp8_pair_swizzle((int)(within % kPerm));
  const long long source = run * kPerm + scale_perm_64(slot);
  const int group = (int)(source / out_dim);
  const int row = (int)(source - (long long)group * out_dim);

  const unsigned char raw = checkpoint[expert_base + (long long)row * scale_k + group];
  float value = __half2float(__half(__nv_cvt_fp8_to_halfraw(raw, __NV_E4M3))) * rescale;

  // S0E5M3: the top bit of the e4m3 byte is the sign, which these scales do
  // not use, so scaling by 2^7 and shifting up one bit reads back as a wider
  // exponent. A value too small to keep its leading bit encodes as zero.
  __half scaled = __float2half(value * 128.0f);
  unsigned short bits = __half_as_ushort(scaled);
  if (__half2float(scaled) < 2.0f) {
    bits = 0;
  }
  prepared[at] = (unsigned char)((bits << 1) >> 8);
}

} // namespace

extern "C" {

// `checkpoint` is `[experts, out_dim, in_dim / 16]` e4m3 bytes and `prepared`
// receives the same count. `rescale` is the shared power of two the caller
// divides out of the per-tensor scale.
CUresult gemma4_marlin_nvfp4_prepare_scales_cuda(const unsigned char *checkpoint,
                                                 unsigned char *prepared,
                                                 int experts, int in_dim,
                                                 int out_dim, float rescale,
                                                 cudaStream_t stream) {
  if (checkpoint == nullptr || prepared == nullptr || experts <= 0 ||
      in_dim <= 0 || out_dim <= 0 || in_dim % kGroup != 0 ||
      !isfinite(rescale) || rescale <= 0.0f) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int scale_k = in_dim / kGroup;
  if (((long long)out_dim * scale_k) % kPerm != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const long long total = (long long)experts * out_dim * scale_k;
  const int block = 256;
  const long long grid = (total + block - 1) / block;
  if (grid > 2147483647LL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  nvfp4_prepare_scales_kernel<<<(int)grid, block, 0, stream>>>(
      checkpoint, prepared, out_dim, scale_k, rescale, total);
  return (CUresult)cudaGetLastError();
}

} // extern "C"
