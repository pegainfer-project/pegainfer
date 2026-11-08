#include "common.cuh"

// A batch-1 projection reads every weight once, so the tiled GEMM it replaces
// runs 40..192 CTAs on a 108-SM part at 63..73% of read bandwidth. One warp per
// row keeps every CTA busy on the only work there is.
namespace {

constexpr int WARPS_PER_BLOCK = 8;

// One row's dot product: every lane strides the row by 16-byte loads, the warp
// folds the lane sums at the end. `x` is read without the streaming hint because
// every row of the block reads it.
__device__ __forceinline__ float row_dot(const __nv_bfloat16* row,
                                         const __nv_bfloat16* x, int cols) {
  const int lane = static_cast<int>(threadIdx.x) & (WARP_SIZE - 1);
  float acc = 0.0f;
  for (int col = lane * 8; col < cols; col += WARP_SIZE * 8) {
    const uint4 wv = __ldcs(reinterpret_cast<const uint4*>(row + col));
    const uint4 xv = *reinterpret_cast<const uint4*>(x + col);
    const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
    const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 wf = __bfloat1622float2(wp[i]);
      const float2 xf = __bfloat1622float2(xp[i]);
      acc = fmaf(wf.x, xf.x, acc);
      acc = fmaf(wf.y, xf.y, acc);
    }
  }
  return warp_reduce_sum(acc);
}

__global__ void gemv_bf16_kernel(const __nv_bfloat16* __restrict__ w,
                                 const __nv_bfloat16* __restrict__ x,
                                 __nv_bfloat16* __restrict__ y, int rows,
                                 int cols) {
  const int warp = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int row = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (row >= rows) {
    return;
  }
  const float acc = row_dot(w + static_cast<long long>(row) * cols, x, cols);
  if ((static_cast<int>(threadIdx.x) & (WARP_SIZE - 1)) == 0) {
    y[row] = __float2bfloat16(acc);
  }
}

}  // namespace

// y[rows] = W[rows, cols] * x[cols], bf16 in and out with fp32 accumulation.
// The caller owns the shape checks: cols must be a multiple of 8 so every 16-byte
// load stays aligned, and both pointers 16-byte aligned.
extern "C" {

int gemv_bf16_cuda(const __nv_bfloat16* w, const __nv_bfloat16* x,
                   __nv_bfloat16* y, int rows, int cols,
                   cudaStream_t stream) {
  if (w == nullptr || x == nullptr || y == nullptr || rows <= 0 || cols <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int grid = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  gemv_bf16_kernel<<<grid, WARPS_PER_BLOCK * WARP_SIZE, 0, stream>>>(w, x, y,
                                                                     rows, cols);
  return static_cast<int>(cudaGetLastError());
}

}  // extern "C"
