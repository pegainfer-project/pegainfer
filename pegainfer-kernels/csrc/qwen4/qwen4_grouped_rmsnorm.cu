// Qwen3.8-Flash-Next grouped (1+weight) RMSNorm.
//
// The reference (`Qwen4ExpTextRMSNorm(hidden, eps, group_size)`) reduces each
// `group_size`-wide group on its own and then scales by `(1 + w)` with `w`
// indexed by the row's global column. FlashInfer's `RMSNormKernel` loads
// `weight + elem` with no row term, so the "view the stream as more rows"
// reshape serves every group group 0's slice; this kernel advances the weight
// base by the row's group index and is otherwise the vendor kernel.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>

#include <flashinfer/norm.cuh>

using DType = __nv_bfloat16;

namespace {

template <uint32_t VEC_SIZE, typename T>
__global__ void GroupedRMSNormKernel(const T* __restrict__ input,
                                     const T* __restrict__ weight,
                                     T* __restrict__ out,
                                     const uint32_t group_size,
                                     const uint32_t groups,
                                     float eps) {
    // Flat row index over the `seq_len * groups` contiguous `group_size`-wide
    // rows: the group runs fastest, matching `unflatten(-1, (groups, g))`.
    const uint32_t bx = blockIdx.x;
    const uint32_t tx = threadIdx.x, ty = threadIdx.y;
    constexpr uint32_t warp_size = 32;
    const uint32_t num_warps = blockDim.y;
    const uint32_t thread_id = tx + ty * warp_size;
    const uint32_t num_threads = num_warps * warp_size;
    const uint32_t rounds = flashinfer::ceil_div(group_size, VEC_SIZE * num_threads);
    extern __shared__ float group_smem[];

    const T* row_in = input + static_cast<size_t>(bx) * group_size;
    T* row_out = out + static_cast<size_t>(bx) * group_size;
    // The one difference from `RMSNormKernel`: the weight slice follows the row.
    const T* row_weight = weight + static_cast<size_t>(bx % groups) * group_size;

    float sum_sq = 0.f;

    for (uint32_t i = 0; i < rounds; i++) {
        flashinfer::vec_t<T, VEC_SIZE> input_vec;
        input_vec.fill(0.f);
        if ((i * num_threads + thread_id) * VEC_SIZE < group_size) {
            input_vec.load(row_in + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
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

    group_smem[ty] = sum_sq;
    __syncthreads();
    if (ty == 0) {
        sum_sq = (tx < num_warps) ? group_smem[tx] : 0.f;
#pragma unroll
        for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
            sum_sq += flashinfer::math::shfl_xor_sync(sum_sq, offset);
        }
        group_smem[0] = sum_sq;
    }
    __syncthreads();

    // One eps for every group: the reference holds a single `eps` on the module
    // and the group loop reshapes the tensor, not the norm.
    const float rms_rcp = flashinfer::math::rsqrt(group_smem[0] / float(group_size) + eps);

    for (uint32_t i = 0; i < rounds; i++) {
        flashinfer::vec_t<T, VEC_SIZE> input_vec;
        flashinfer::vec_t<T, VEC_SIZE> weight_vec;
        flashinfer::vec_t<T, VEC_SIZE> out_vec;
        input_vec.fill(0.f);
        weight_vec.fill(0.f);
        const uint32_t elem = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
        if (elem < group_size) {
            input_vec.load(row_in + elem);
            weight_vec.load(row_weight + elem);
        }
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
            // The `(1 + w)` convention, and the only rounding: like GemmaRMSNorm,
            // the f32 product is rounded once on the store to bf16.
            out_vec[j] = float(input_vec[j]) * rms_rcp * (1.f + float(weight_vec[j]));
        }
        if (elem < group_size) {
            out_vec.store(row_out + elem);
        }
    }
}

}  // namespace

extern "C" {

// `x`/`out` are [seq_len, hidden_dim] row-major with `hidden_dim` split into
// `hidden_dim / group_size` equal groups; flattened, row `r` of the
// `seq_len * groups` g-wide rows belongs to group `r % groups` and reads that
// group's slice of `weight`.
CUresult rms_norm_grouped_offset_cuda(const DType *x, const DType *weight, DType *out,
                                      int hidden_dim, int group_size, int seq_len,
                                      float eps, cudaStream_t stream) {
    if (hidden_dim <= 0 || group_size <= 0 || seq_len <= 0 || hidden_dim % group_size != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const uint32_t d = static_cast<uint32_t>(group_size);
    const uint32_t groups = static_cast<uint32_t>(hidden_dim / group_size);
    // `dim3.x` is capped at 2^31-1. Compute the row count in 64 bits so a large
    // call is rejected instead of wrapping into a partial launch.
    const uint64_t rows = static_cast<uint64_t>(seq_len) * groups;
    if (rows > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const uint32_t vec_size = std::gcd<uint32_t>(16 / sizeof(DType), d);
    const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
    const uint32_t num_warps = flashinfer::ceil_div(block_size, 32);
    // One block per (token, group), the grid the flattened view would get.
    dim3 nblks(static_cast<uint32_t>(rows));
    dim3 nthrs(32, num_warps);
    const uint32_t smem_size = num_warps * sizeof(float);
    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
        GroupedRMSNormKernel<VEC_SIZE, DType>
            <<<nblks, nthrs, smem_size, stream>>>(x, weight, out, d, groups, eps);
    });
    return map_cuda_error(cudaGetLastError());
}

}  // extern "C"
