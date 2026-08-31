// FlashInfer-backed norm kernels.
//
// Provides the same extern "C" API surface as our hand-written norm.cu,
// but delegates to FlashInfer's header-only RMSNorm / FusedAddRMSNorm /
// GemmaRMSNorm / GemmaFusedAddRMSNorm templates.
//
// Semantic adapter for FusedAddRMSNorm:
//   Our API:       hidden += residual; out = norm(hidden, weight)
//   FlashInfer:    residual_arg += input_arg; input_arg = norm(residual_arg, weight)
//
//   To bridge the gap we memcpy residual → out, then call FlashInfer with
//   (input=out, residual=hidden). After the call:
//     hidden = hidden + out(=residual)   ← what we want
//     out    = norm(hidden)              ← what we want
//   The memcpy is ≤14 KB per row (hidden_size=3584 × 2 bytes) and negligible.

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <numeric>

#include <flashinfer/norm.cuh>

using DType = __nv_bfloat16;

namespace pegainfer {
namespace norm {

// Exact-preserving variant for the decode pattern:
//   hidden = bf16(hidden + residual)
//   out = RMSNorm(hidden, weight)
//
// FlashInfer's FusedAddRMSNorm keeps the pre-BF16-round add value in shared
// memory for the RMS reduction. Kimi token correctness currently depends on
// the separate add kernel's BF16 rounding boundary, so this kernel mirrors the
// FlashInfer reduction/order but feeds it the rounded BF16 sum.
template <uint32_t VEC_SIZE, typename T>
__global__ void FusedAddRMSNormRoundKernel(T* __restrict__ hidden,
                                           const T* __restrict__ residual,
                                           T* __restrict__ weight,
                                           T* __restrict__ out,
                                           const uint32_t d,
                                           const uint32_t stride_hidden,
                                           const uint32_t stride_residual,
                                           const uint32_t stride_out,
                                           float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = flashinfer::ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + flashinfer::ceil_div(num_warps, 4) * 4;

  float sum_sq = 0.f;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (uint32_t i = 0; i < rounds; i++) {
    flashinfer::vec_t<T, VEC_SIZE> hidden_vec;
    flashinfer::vec_t<T, VEC_SIZE> residual_vec;
    flashinfer::vec_t<float, VEC_SIZE> rounded_vec;
    hidden_vec.fill(0.f);
    residual_vec.fill(0.f);
    rounded_vec.fill(0.f);
    const uint32_t elem = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
    if (elem < d) {
      hidden_vec.load(hidden + bx * stride_hidden + elem);
      residual_vec.load(residual + bx * stride_residual + elem);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      T rounded = static_cast<T>(float(hidden_vec[j]) + float(residual_vec[j]));
      float x = float(rounded);
      hidden_vec[j] = rounded;
      rounded_vec[j] = x;
      sum_sq += x * x;
    }
    if (elem < d) {
      hidden_vec.store(hidden + bx * stride_hidden + elem);
      rounded_vec.store(smem_x + elem);
    }
  }

#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += flashinfer::math::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += flashinfer::math::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = flashinfer::math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    flashinfer::vec_t<T, VEC_SIZE> weight_vec;
    flashinfer::vec_t<float, VEC_SIZE> rounded_vec;
    flashinfer::vec_t<T, VEC_SIZE> out_vec;
    weight_vec.fill(0.f);
    rounded_vec.fill(0.f);
    out_vec.fill(0.f);
    const uint32_t elem = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
    if (elem < d) {
      weight_vec.load(weight + elem);
      rounded_vec.load(smem_x + elem);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      out_vec[j] = rounded_vec[j] * rms_rcp * float(weight_vec[j]);
    }
    if (elem < d) {
      out_vec.store(out + bx * stride_out + elem);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T>
cudaError_t FusedAddRMSNormRound(T* hidden, const T* residual, T* weight, T* out,
                                 uint32_t batch_size, uint32_t d,
                                 uint32_t stride_hidden, uint32_t stride_residual,
                                 uint32_t stride_out, float eps, cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);
  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = flashinfer::ceil_div(block_size, 32);
  dim3 nblks(batch_size);
  dim3 nthrs(32, num_warps);
  const uint32_t smem_size = (flashinfer::ceil_div(num_warps, 4) * 4 + d) * sizeof(float);

  cudaLaunchConfig_t config;
  config.gridDim = nblks;
  config.blockDim = nthrs;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = false;
  config.numAttrs = 1;
  config.attrs = attrs;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = FusedAddRMSNormRoundKernel<VEC_SIZE, T>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, hidden, residual, weight, out, d,
                                            stride_hidden, stride_residual, stride_out, eps));
  });
  return cudaSuccess;
}

template <uint32_t VEC_SIZE, typename T>
__global__ void DualRMSNormKernel(const T* __restrict__ input,
                                  const T* __restrict__ weight_a,
                                  const T* __restrict__ weight_b, T* __restrict__ out_a,
                                  T* __restrict__ out_b, const uint32_t d, float eps,
                                  float scale_a) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = flashinfer::ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq = 0.f;
  for (uint32_t i = 0; i < rounds; i++) {
    flashinfer::vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * d + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      sum_sq += float(input_vec[j]) * float(input_vec[j]);
    }
  }
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += flashinfer::math::shfl_xor_sync(sum_sq, offset);
  }
  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += flashinfer::math::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = flashinfer::math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    flashinfer::vec_t<T, VEC_SIZE> input_vec;
    flashinfer::vec_t<T, VEC_SIZE> weight_a_vec;
    flashinfer::vec_t<T, VEC_SIZE> weight_b_vec;
    flashinfer::vec_t<T, VEC_SIZE> out_a_vec;
    flashinfer::vec_t<T, VEC_SIZE> out_b_vec;
    input_vec.fill(0.f);
    weight_a_vec.fill(0.f);
    weight_b_vec.fill(0.f);
    const uint32_t elem = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
    if (elem < d) {
      input_vec.load(input + bx * d + elem);
      weight_a_vec.load(weight_a + elem);
      weight_b_vec.load(weight_b + elem);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      // Preserve the standalone bf16 norm rounding before the scalar multiply.
      T na = float(input_vec[j]) * rms_rcp * (0.f + float(weight_a_vec[j]));
      out_a_vec[j] = float(na) * scale_a;
      out_b_vec[j] = float(input_vec[j]) * rms_rcp * (0.f + float(weight_b_vec[j]));
    }
    if (elem < d) {
      out_a_vec.store(out_a + bx * d + elem);
      out_b_vec.store(out_b + bx * d + elem);
    }
  }
}

template <uint32_t VEC_SIZE, typename T>
__global__ void RMSNormAddScaleKernel(const T* __restrict__ input,
                                      const T* __restrict__ weight,
                                      const T* __restrict__ residual, T* __restrict__ out,
                                      const uint32_t d, float eps, float scale) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = flashinfer::ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq = 0.f;
  for (uint32_t i = 0; i < rounds; i++) {
    flashinfer::vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * d + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      sum_sq += float(input_vec[j]) * float(input_vec[j]);
    }
  }
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += flashinfer::math::shfl_xor_sync(sum_sq, offset);
  }
  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += flashinfer::math::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = flashinfer::math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    flashinfer::vec_t<T, VEC_SIZE> input_vec;
    flashinfer::vec_t<T, VEC_SIZE> weight_vec;
    flashinfer::vec_t<T, VEC_SIZE> residual_vec;
    flashinfer::vec_t<T, VEC_SIZE> out_vec;
    input_vec.fill(0.f);
    weight_vec.fill(0.f);
    residual_vec.fill(0.f);
    const uint32_t elem = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
    if (elem < d) {
      input_vec.load(input + bx * d + elem);
      weight_vec.load(weight + elem);
      residual_vec.load(residual + bx * d + elem);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      // Preserve the standalone bf16 norm and residual-add rounding boundaries.
      T normed = float(input_vec[j]) * rms_rcp * (0.f + float(weight_vec[j]));
      T summed = float(normed) + float(residual_vec[j]);
      out_vec[j] = float(summed) * scale;
    }
    if (elem < d) {
      out_vec.store(out + bx * d + elem);
    }
  }
}

template <uint32_t VEC_SIZE, typename T>
__global__ void RMSNormAddRMSNormRoundKernel(const T* __restrict__ x,
                                             const T* __restrict__ weight_post,
                                             const T* __restrict__ res_in,
                                             const T* __restrict__ weight_pre,
                                             T* __restrict__ residual_out, T* __restrict__ out,
                                             const uint32_t d, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = flashinfer::ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + flashinfer::ceil_div(num_warps, 4) * 4;

  float sum_sq1 = 0.f;
  for (uint32_t i = 0; i < rounds; i++) {
    flashinfer::vec_t<T, VEC_SIZE> x_vec;
    flashinfer::vec_t<float, VEC_SIZE> stash_vec;
    x_vec.fill(0.f);
    stash_vec.fill(0.f);
    const uint32_t elem = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
    if (elem < d) {
      x_vec.load(x + bx * d + elem);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float v = float(x_vec[j]);
      stash_vec[j] = v;
      sum_sq1 += v * v;
    }
    if (elem < d) {
      stash_vec.store(smem_x + elem);
    }
  }
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq1 += flashinfer::math::shfl_xor_sync(sum_sq1, offset);
  }
  smem[ty] = sum_sq1;
  __syncthreads();
  if (ty == 0) {
    sum_sq1 = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq1 += flashinfer::math::shfl_xor_sync(sum_sq1, offset);
    }
    smem[0] = sum_sq1;
  }
  __syncthreads();
  float rcp1 = flashinfer::math::rsqrt(smem[0] / float(d) + eps);
  // The warp-sum slots are reused for the second reduction; every thread has
  // read `smem[0]` into `rcp1` above.
  __syncthreads();

  float sum_sq2 = 0.f;
  for (uint32_t i = 0; i < rounds; i++) {
    flashinfer::vec_t<T, VEC_SIZE> weight_vec;
    flashinfer::vec_t<T, VEC_SIZE> res_vec;
    flashinfer::vec_t<T, VEC_SIZE> summed_vec;
    flashinfer::vec_t<float, VEC_SIZE> stash_vec;
    weight_vec.fill(0.f);
    res_vec.fill(0.f);
    stash_vec.fill(0.f);
    const uint32_t elem = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
    if (elem < d) {
      weight_vec.load(weight_post + elem);
      res_vec.load(res_in + bx * d + elem);
      stash_vec.load(smem_x + elem);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      // Round the first norm and residual sum before the second reduction.
      T normed = stash_vec[j] * rcp1 * (0.f + float(weight_vec[j]));
      T summed = float(normed) + float(res_vec[j]);
      float v = float(summed);
      summed_vec[j] = summed;
      stash_vec[j] = v;
      sum_sq2 += v * v;
    }
    if (elem < d) {
      summed_vec.store(residual_out + bx * d + elem);
      stash_vec.store(smem_x + elem);
    }
  }
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq2 += flashinfer::math::shfl_xor_sync(sum_sq2, offset);
  }
  smem[ty] = sum_sq2;
  __syncthreads();
  if (ty == 0) {
    sum_sq2 = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq2 += flashinfer::math::shfl_xor_sync(sum_sq2, offset);
    }
    smem[0] = sum_sq2;
  }
  __syncthreads();
  float rcp2 = flashinfer::math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    flashinfer::vec_t<T, VEC_SIZE> weight_vec;
    flashinfer::vec_t<float, VEC_SIZE> stash_vec;
    flashinfer::vec_t<T, VEC_SIZE> out_vec;
    weight_vec.fill(0.f);
    stash_vec.fill(0.f);
    out_vec.fill(0.f);
    const uint32_t elem = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
    if (elem < d) {
      weight_vec.load(weight_pre + elem);
      stash_vec.load(smem_x + elem);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      // Preserve standalone RMSNorm's signed-zero addition before the multiply.
      out_vec[j] = stash_vec[j] * rcp2 * (0.f + float(weight_vec[j]));
    }
    if (elem < d) {
      out_vec.store(out + bx * d + elem);
    }
  }
}

template <uint32_t VEC_SIZE, typename T>
__global__ void DualRMSNormAddKernel(const T* __restrict__ a,
                                     const T* __restrict__ weight_a,
                                     const T* __restrict__ b,
                                     const T* __restrict__ weight_b,
                                     T* __restrict__ out, const uint32_t d, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = flashinfer::ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq_a = 0.f, sum_sq_b = 0.f;
  for (uint32_t i = 0; i < rounds; i++) {
    flashinfer::vec_t<T, VEC_SIZE> a_vec;
    flashinfer::vec_t<T, VEC_SIZE> b_vec;
    a_vec.fill(0.f);
    b_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      a_vec.load(a + bx * d + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      b_vec.load(b + bx * d + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      sum_sq_a += float(a_vec[j]) * float(a_vec[j]);
      sum_sq_b += float(b_vec[j]) * float(b_vec[j]);
    }
  }
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq_a += flashinfer::math::shfl_xor_sync(sum_sq_a, offset);
    sum_sq_b += flashinfer::math::shfl_xor_sync(sum_sq_b, offset);
  }
  smem[ty] = sum_sq_a;
  smem[num_warps + ty] = sum_sq_b;
  __syncthreads();
  if (ty == 0) {
    sum_sq_a = (tx < num_warps) ? smem[tx] : 0.f;
    sum_sq_b = (tx < num_warps) ? smem[num_warps + tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq_a += flashinfer::math::shfl_xor_sync(sum_sq_a, offset);
      sum_sq_b += flashinfer::math::shfl_xor_sync(sum_sq_b, offset);
    }
    smem[0] = sum_sq_a;
    smem[1] = sum_sq_b;
  }
  __syncthreads();

  float rcp_a = flashinfer::math::rsqrt(smem[0] / float(d) + eps);
  float rcp_b = flashinfer::math::rsqrt(smem[1] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    flashinfer::vec_t<T, VEC_SIZE> a_vec;
    flashinfer::vec_t<T, VEC_SIZE> b_vec;
    flashinfer::vec_t<T, VEC_SIZE> weight_a_vec;
    flashinfer::vec_t<T, VEC_SIZE> weight_b_vec;
    flashinfer::vec_t<T, VEC_SIZE> out_vec;
    a_vec.fill(0.f);
    b_vec.fill(0.f);
    weight_a_vec.fill(0.f);
    weight_b_vec.fill(0.f);
    const uint32_t elem = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
    if (elem < d) {
      a_vec.load(a + bx * d + elem);
      b_vec.load(b + bx * d + elem);
      weight_a_vec.load(weight_a + elem);
      weight_b_vec.load(weight_b + elem);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      // Round both standalone norms before their bf16 sum.
      T na = float(a_vec[j]) * rcp_a * (0.f + float(weight_a_vec[j]));
      T nb = float(b_vec[j]) * rcp_b * (0.f + float(weight_b_vec[j]));
      out_vec[j] = float(na) + float(nb);
    }
    if (elem < d) {
      out_vec.store(out + bx * d + elem);
    }
  }
}

}  // namespace norm
}  // namespace pegainfer

__global__ void rms_norm_batched_serial_kernel(const DType *x, const DType *weight, DType *out,
                                               int hidden_dim, int seq_len, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hidden_dim * seq_len;
    if (idx >= total) return;

    int dim = idx % hidden_dim;
    int row = idx / hidden_dim;
    const DType *row_x = x + row * hidden_dim;
    float sum_sq = 0.0f;
    for (int k = 0; k < hidden_dim; ++k) {
        float value = __bfloat162float(row_x[k]);
        sum_sq += value * value;
    }
    float inv_rms = rsqrtf(sum_sq / hidden_dim + eps);
    float value = __bfloat162float(row_x[dim]) * inv_rms * __bfloat162float(weight[dim]);
    out[row * hidden_dim + dim] = __float2bfloat16(value);
}

extern "C" {

// ============================================================================
// RMSNorm (single vector, decode path)
// ============================================================================
void rms_norm_cuda(const DType *x, const DType *weight, DType *out,
                   int n, float eps, cudaStream_t stream) {
    flashinfer::norm::RMSNorm<DType>(
        const_cast<DType*>(x), const_cast<DType*>(weight), out,
        1, n, n, n, eps, false, stream);
}

// ============================================================================
// RMSNorm batched (prefill path, one block per token)
// ============================================================================
void rms_norm_batched_cuda(const DType *x, const DType *weight, DType *out,
                           int hidden_dim, int seq_len,
                           float eps, cudaStream_t stream) {
    flashinfer::norm::RMSNorm<DType>(
        const_cast<DType*>(x), const_cast<DType*>(weight), out,
        seq_len, hidden_dim, hidden_dim, hidden_dim, eps, false, stream);
}

CUresult rms_norm_batched_dual_cuda(const DType *x, const DType *weight_a, const DType *weight_b,
                                    DType *out_a, DType *out_b, int hidden_dim, int seq_len,
                                    float eps, float scale_a, cudaStream_t stream) {
    const uint32_t d = static_cast<uint32_t>(hidden_dim);
    const uint32_t vec_size = std::gcd<uint32_t>(16 / sizeof(DType), d);
    const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
    const uint32_t num_warps = flashinfer::ceil_div(block_size, 32);
    dim3 nblks(static_cast<uint32_t>(seq_len));
    dim3 nthrs(32, num_warps);
    const uint32_t smem_size = num_warps * sizeof(float);
    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
        pegainfer::norm::DualRMSNormKernel<VEC_SIZE, DType><<<nblks, nthrs, smem_size, stream>>>(
            x, weight_a, weight_b, out_a, out_b, d, eps, scale_a);
    });
    return static_cast<CUresult>(cudaGetLastError());
}

CUresult dual_rms_norm_add_batched_cuda(const DType *a, const DType *weight_a, const DType *b,
                                        const DType *weight_b, DType *out, int hidden_dim,
                                        int seq_len, float eps, cudaStream_t stream) {
    const uint32_t d = static_cast<uint32_t>(hidden_dim);
    const uint32_t vec_size = std::gcd<uint32_t>(16 / sizeof(DType), d);
    const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
    const uint32_t num_warps = flashinfer::ceil_div(block_size, 32);
    dim3 nblks(static_cast<uint32_t>(seq_len));
    dim3 nthrs(32, num_warps);
    const uint32_t smem_size = 2 * num_warps * sizeof(float);
    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
        pegainfer::norm::DualRMSNormAddKernel<VEC_SIZE, DType><<<nblks, nthrs, smem_size, stream>>>(
            a, weight_a, b, weight_b, out, d, eps);
    });
    return static_cast<CUresult>(cudaGetLastError());
}

CUresult rms_norm_add_rms_norm_round_batched_cuda(const DType *x, const DType *weight_post,
                                                  const DType *res_in, const DType *weight_pre,
                                                  DType *residual_out, DType *out, int hidden_dim,
                                                  int seq_len, float eps, cudaStream_t stream) {
    const uint32_t d = static_cast<uint32_t>(hidden_dim);
    const uint32_t vec_size = std::gcd<uint32_t>(16 / sizeof(DType), d);
    const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
    const uint32_t num_warps = flashinfer::ceil_div(block_size, 32);
    dim3 nblks(static_cast<uint32_t>(seq_len));
    dim3 nthrs(32, num_warps);
    const uint32_t smem_size = (flashinfer::ceil_div(num_warps, 4) * 4 + d) * sizeof(float);
    cudaError_t err = cudaSuccess;
    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
        auto kernel = pegainfer::norm::RMSNormAddRMSNormRoundKernel<VEC_SIZE, DType>;
        constexpr uint32_t default_smem_size = 48 * 1024;
        if (smem_size > default_smem_size) {
            err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       smem_size);
        }
        if (err == cudaSuccess) {
            kernel<<<nblks, nthrs, smem_size, stream>>>(
                x, weight_post, res_in, weight_pre, residual_out, out, d, eps);
            err = cudaGetLastError();
        }
    });
    return static_cast<CUresult>(err);
}

CUresult rms_norm_add_scale_batched_cuda(const DType *x, const DType *weight,
                                         const DType *residual, DType *out, int hidden_dim,
                                         int seq_len, float eps, float scale,
                                         cudaStream_t stream) {
    const uint32_t d = static_cast<uint32_t>(hidden_dim);
    const uint32_t vec_size = std::gcd<uint32_t>(16 / sizeof(DType), d);
    const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
    const uint32_t num_warps = flashinfer::ceil_div(block_size, 32);
    dim3 nblks(static_cast<uint32_t>(seq_len));
    dim3 nthrs(32, num_warps);
    const uint32_t smem_size = num_warps * sizeof(float);
    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
        pegainfer::norm::RMSNormAddScaleKernel<VEC_SIZE, DType><<<nblks, nthrs, smem_size, stream>>>(
            x, weight, residual, out, d, eps, scale);
    });
    return static_cast<CUresult>(cudaGetLastError());
}

// ============================================================================
// Fused Add + RMSNorm (single vector, decode path)
//   hidden += residual; out = norm(hidden, weight)
// ============================================================================
void fused_add_rms_norm_cuda(DType *hidden, const DType *residual,
                             const DType *weight, DType *out,
                             int n, float eps, cudaStream_t stream) {
    // Copy residual → out so FlashInfer can read it as the "input" addend.
    cudaMemcpyAsync(out, residual, static_cast<size_t>(n) * sizeof(DType),
                    cudaMemcpyDeviceToDevice, stream);

    // FlashInfer: hidden(=residual_arg) += out(=input_arg); out = norm(hidden)
    flashinfer::norm::FusedAddRMSNorm<DType>(
        /*input=*/out, /*residual=*/hidden, const_cast<DType*>(weight),
        /*batch_size=*/1, /*d=*/static_cast<uint32_t>(n),
        /*stride_input=*/static_cast<uint32_t>(n),
        /*stride_residual=*/static_cast<uint32_t>(n),
        eps, /*enable_pdl=*/false, stream);
}

// ============================================================================
// Fused Add + RMSNorm batched (prefill path)
// ============================================================================
void fused_add_rms_norm_batched_cuda(DType *hidden, const DType *residual,
                                     const DType *weight, DType *out,
                                     int hidden_dim, int batch_size,
                                     float eps, cudaStream_t stream) {
    size_t total_bytes = static_cast<size_t>(hidden_dim) * batch_size * sizeof(DType);
    cudaMemcpyAsync(out, residual, total_bytes,
                    cudaMemcpyDeviceToDevice, stream);

    flashinfer::norm::FusedAddRMSNorm<DType>(
        /*input=*/out, /*residual=*/hidden, const_cast<DType*>(weight),
        /*batch_size=*/static_cast<uint32_t>(batch_size),
        /*d=*/static_cast<uint32_t>(hidden_dim),
        /*stride_input=*/static_cast<uint32_t>(hidden_dim),
        /*stride_residual=*/static_cast<uint32_t>(hidden_dim),
        eps, /*enable_pdl=*/false, stream);
}

CUresult fused_add_rms_norm_round_batched_cuda(DType *hidden, const DType *residual,
                                               const DType *weight, DType *out,
                                               int hidden_dim, int batch_size,
                                               float eps, cudaStream_t stream) {
    cudaError_t err = pegainfer::norm::FusedAddRMSNormRound<DType>(
        hidden, residual, const_cast<DType*>(weight), out,
        /*batch_size=*/static_cast<uint32_t>(batch_size),
        /*d=*/static_cast<uint32_t>(hidden_dim),
        /*stride_hidden=*/static_cast<uint32_t>(hidden_dim),
        /*stride_residual=*/static_cast<uint32_t>(hidden_dim),
        /*stride_out=*/static_cast<uint32_t>(hidden_dim),
        eps, stream);
    return static_cast<CUresult>(err);
}

// ============================================================================
// (1+weight) RMSNorm — Qwen3.5 / Gemma style
// ============================================================================
void rms_norm_offset_cuda(const DType *x, const DType *weight, DType *out,
                          int n, float eps, cudaStream_t stream) {
    flashinfer::norm::GemmaRMSNorm<DType>(
        const_cast<DType*>(x), const_cast<DType*>(weight), out,
        /*batch_size=*/1, /*d=*/static_cast<uint32_t>(n),
        /*stride_input=*/static_cast<uint32_t>(n),
        /*stride_output=*/static_cast<uint32_t>(n),
        eps, /*enable_pdl=*/false, stream);
}

// ============================================================================
// Batched (1+weight) RMSNorm
// ============================================================================
void rms_norm_batched_offset_cuda(const DType *x, const DType *weight, DType *out,
                                  int hidden_dim, int seq_len,
                                  float eps, cudaStream_t stream) {
    flashinfer::norm::GemmaRMSNorm<DType>(
        const_cast<DType*>(x), const_cast<DType*>(weight), out,
        /*batch_size=*/static_cast<uint32_t>(seq_len),
        /*d=*/static_cast<uint32_t>(hidden_dim),
        /*stride_input=*/static_cast<uint32_t>(hidden_dim),
        /*stride_output=*/static_cast<uint32_t>(hidden_dim),
        eps, /*enable_pdl=*/false, stream);
}

// ============================================================================
// LayerNorm (with bias) — GLM5.2 DSA indexer k_norm.
// HAND-WRITTEN: FlashInfer's generalLayerNorm template depends on
// tensorrt_llm::common::packed_as / num_elems traits that are not available in
// this build's include path. This kernel is a simple single-token LayerNorm
// (mean + variance + affine with bias), memory-bound elementwise — same
// pattern as the hand-written rms_norm_batched_serial_kernel above.
// eps=1e-6, with bias (unlike RMSNorm which has no bias).
// Aligned to vllm DeepseekV32Indexer: nn.LayerNorm(head_dim, eps=1e-6).
// ============================================================================
__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;
}

__global__ void layer_norm_kernel(const DType *x, const float *gamma, const float *beta,
                                   DType *out, int n, float eps) {
    int tid = threadIdx.x;
    // One block per row; each row's reduction is self-contained, so the
    // batched launch is bit-identical per row to the rows=1 launch.
    x += (size_t)blockIdx.x * n;
    out += (size_t)blockIdx.x * n;
    extern __shared__ float smem[];  // [n] for val, reused for partial sums

    // Phase 1: load + mean (warp shuffle reduction).
    float val = 0.0f;
    if (tid < n) {
        val = __bfloat162float(x[tid]);
    }
    float sum = warp_reduce_sum(val);

    // Cross-warp reduction via shared memory (only lane 0 of each warp writes).
    int lane = tid % 32;
    int warp = tid / 32;
    int num_warps = blockDim.x / 32;
    if (lane == 0) {
        smem[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        sum = (lane < num_warps) ? smem[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            smem[0] = sum;
        }
    }
    __syncthreads();
    float mean = smem[0] / n;

    // Phase 2: variance (same reduction pattern).
    float diff_sum = 0.0f;
    if (tid < n) {
        float diff = val - mean;
        diff_sum = diff * diff;
    }
    float var_sum = warp_reduce_sum(diff_sum);
    if (lane == 0) {
        smem[warp] = var_sum;
    }
    __syncthreads();
    if (warp == 0) {
        var_sum = (lane < num_warps) ? smem[lane] : 0.0f;
        var_sum = warp_reduce_sum(var_sum);
        if (lane == 0) {
            smem[0] = var_sum;
        }
    }
    __syncthreads();
    float rstd = rsqrtf(smem[0] / n + eps);

    // Phase 3: output = (x - mean) * rstd * gamma + beta.
    if (tid < n) {
        float normalized = (val - mean) * rstd;
        out[tid] = __float2bfloat16(normalized * gamma[tid] + beta[tid]);
    }
}

CUresult layer_norm_cuda(const DType *x, const float *gamma, const float *beta,
                         DType *out, int n, int rows, float eps,
                         cudaStream_t stream) {
    if (x == nullptr || gamma == nullptr || beta == nullptr || out == nullptr ||
        rows <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    int block_size = std::min(n, 1024);
    block_size = 32 * ((block_size + 31) / 32);
    size_t shmem_size = std::min(block_size / 32, (n + 31) / 32) * sizeof(float);
    layer_norm_kernel<<<rows, block_size, shmem_size, stream>>>(x, gamma, beta, out, n, eps);
    cudaError_t err = cudaGetLastError();
    return static_cast<CUresult>(err);
}

// ============================================================================
// Fused Add + (1+weight) RMSNorm — Qwen3.5 / Gemma style
// ============================================================================
void fused_add_rms_norm_offset_cuda(DType *hidden, const DType *residual,
                                    const DType *weight, DType *out,
                                    int n, float eps, cudaStream_t stream) {
    cudaMemcpyAsync(out, residual, static_cast<size_t>(n) * sizeof(DType),
                    cudaMemcpyDeviceToDevice, stream);

    flashinfer::norm::GemmaFusedAddRMSNorm<DType>(
        /*input=*/out, /*residual=*/hidden, const_cast<DType*>(weight),
        /*batch_size=*/1, /*d=*/static_cast<uint32_t>(n),
        /*stride_input=*/static_cast<uint32_t>(n),
        /*stride_residual=*/static_cast<uint32_t>(n),
        eps, /*enable_pdl=*/false, stream);
}

} // extern "C"
