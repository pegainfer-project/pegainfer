#include "common.cuh"

#include <algorithm>
#include <cstdint>
#include <cuda.h>

namespace {

// Dynamic coefficients belong to the destination row. Causal taps never cross
// a request's draft-block boundary, and no convolution state persists.
__global__ void grouped_conv_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* dynamic,
    const __nv_bfloat16* base, __nv_bfloat16* output, int rows, int hidden,
    int block, int group_size, int taps, int side) {
  const int groups = hidden / group_size;
  const int64_t count = static_cast<int64_t>(rows) * hidden;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int row = index / hidden;
    const int channel = index % hidden;
    const int position = row % block;
    const int group = channel / group_size;
    __nv_bfloat16 sum = __float2bfloat16(0.0f);

    for (int tap = 0; tap < taps && tap <= position; ++tap) {
      const int64_t coefficient_index =
          (static_cast<int64_t>(row) * 2 * taps + side * taps + tap) * groups + group;
      const int64_t base_index = static_cast<int64_t>(side * taps + tap) * hidden + channel;
      // Match the checkpoint's BF16 coefficient, product and per-tap sum.
      // Combining these into one FP32 FMA changes the trained forward path.
      const __nv_bfloat16 coefficient = __float2bfloat16(
          __bfloat162float(base[base_index]) + __bfloat162float(dynamic[coefficient_index]));
      const __nv_bfloat16 product = __float2bfloat16(
          __bfloat162float(input[index - static_cast<int64_t>(tap) * hidden]) *
          __bfloat162float(coefficient));
      sum = __float2bfloat16(__bfloat162float(sum) + __bfloat162float(product));
    }
    output[index] = sum;
  }
}

}  // namespace

extern "C" CUresult dflash2_grouped_conv_cuda(
    const __nv_bfloat16* input, const __nv_bfloat16* dynamic,
    const __nv_bfloat16* base, __nv_bfloat16* output, int rows, int hidden,
    int block, int group_size, int taps, int side, cudaStream_t stream) {
  if (!input || !dynamic || !base || !output || rows <= 0 || hidden <= 0 ||
      block <= 0 || group_size <= 0 || taps <= 0 || side < 0 || side > 1 ||
      rows % block != 0 || hidden % group_size != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int64_t count = static_cast<int64_t>(rows) * hidden;
  const int grid = static_cast<int>(std::min<int64_t>((count + 255) / 256, 65535));
  grouped_conv_kernel<<<grid, 256, 0, stream>>>(
      input, dynamic, base, output, rows, hidden, block, group_size, taps, side);
  return static_cast<CUresult>(cudaGetLastError());
}
