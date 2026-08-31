#include "../marlin/ffi.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace pegainfer_kimi_marlin_int4 {

constexpr int kKimiLocalExperts = 48;
constexpr int kKimiInt4GroupSize = 32;
constexpr int kPackFactorInt4 = 8;
constexpr int kMarlinThreads = 256;
constexpr int kMarlinTileK = 16;
constexpr int kMarlinTileN = 64;

bool kimi_marlin_common_shape_ok(int in_dim, int out_dim, int local_experts, int group_size) {
  return in_dim > 0 && out_dim > 0 && local_experts == kKimiLocalExperts &&
         group_size == kKimiInt4GroupSize && (in_dim % kMarlinTileK) == 0 &&
         (out_dim % kMarlinTileN) == 0 && (in_dim % group_size) == 0;
}

__global__ void kimi_marlin_fuse_w13_weight_kernel(
    const uint32_t* __restrict__ gate_weight,
    const uint32_t* __restrict__ up_weight,
    uint32_t* __restrict__ w13_weight,
    int size_k,
    int intermediate_size) {
  int k_tiles = size_k / kMarlinTileK;
  int src_cols = intermediate_size * 2;
  int dst_cols = src_cols * 2;
  size_t total = static_cast<size_t>(kKimiLocalExperts) * k_tiles * src_cols;
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (; idx < total; idx += stride) {
    int col = static_cast<int>(idx % src_cols);
    size_t row = idx / src_cols;
    size_t dst_base = row * static_cast<size_t>(dst_cols);
    w13_weight[dst_base + col] = gate_weight[idx];
    w13_weight[dst_base + static_cast<size_t>(src_cols) + col] = up_weight[idx];
  }
}

__device__ __forceinline__ int kimi_marlin_scale_perm_64(int offset) {
  return (offset / 8) + 8 * (offset % 8);
}

__global__ void kimi_marlin_reorder_scale_kernel(
    const __nv_bfloat16* scale_checkpoint,
    __nv_bfloat16* scale_marlin,
    int out_dim,
    int scale_k,
    size_t total_elements) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  size_t elements_per_expert = static_cast<size_t>(out_dim) * scale_k;
  for (; idx < total_elements; idx += stride) {
    size_t in_expert = idx % elements_per_expert;
    size_t expert_base = idx - in_expert;
    size_t block = in_expert / 64;
    int offset = static_cast<int>(in_expert % 64);
    size_t transposed = block * 64 + static_cast<size_t>(kimi_marlin_scale_perm_64(offset));
    int group = static_cast<int>(transposed / out_dim);
    int row = static_cast<int>(transposed - static_cast<size_t>(group) * out_dim);
    scale_marlin[idx] = scale_checkpoint[expert_base + static_cast<size_t>(row) * scale_k + group];
  }
}

__global__ void kimi_marlin_fuse_w13_scale_kernel(
    const __nv_bfloat16* __restrict__ gate_scale,
    const __nv_bfloat16* __restrict__ up_scale,
    __nv_bfloat16* __restrict__ w13_scale,
    int size_k,
    int intermediate_size) {
  int scale_groups = size_k / kKimiInt4GroupSize;
  int src_cols = intermediate_size;
  int dst_cols = intermediate_size * 2;
  size_t total = static_cast<size_t>(kKimiLocalExperts) * scale_groups * src_cols;
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (; idx < total; idx += stride) {
    int col = static_cast<int>(idx % src_cols);
    size_t row = idx / src_cols;
    size_t dst_base = row * static_cast<size_t>(dst_cols);
    w13_scale[dst_base + col] = gate_scale[idx];
    w13_scale[dst_base + static_cast<size_t>(src_cols) + col] = up_scale[idx];
  }
}

}  // namespace pegainfer_kimi_marlin_int4

using namespace pegainfer_kimi_marlin_int4;

extern "C" {

CUresult kimi_marlin_int4_reorder_weight_cuda(
    const uint8_t* weight_packed_checkpoint_offset_binary,
    uint8_t* weight_packed_marlin,
    int in_dim,
    int out_dim,
    int local_experts,
    int group_size,
    cudaStream_t stream) {
  if (weight_packed_checkpoint_offset_binary == nullptr || weight_packed_marlin == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!kimi_marlin_common_shape_ok(in_dim, out_dim, local_experts, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return marlin_repack_4bit_cuda(weight_packed_checkpoint_offset_binary,
                                 weight_packed_marlin, local_experts, in_dim,
                                 out_dim, stream);
}

CUresult kimi_marlin_int4_reorder_scale_cuda(
    const __nv_bfloat16* weight_scale_checkpoint,
    __nv_bfloat16* weight_scale_marlin,
    int in_dim,
    int out_dim,
    int local_experts,
    int group_size,
    cudaStream_t stream) {
  if (weight_scale_checkpoint == nullptr || weight_scale_marlin == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (in_dim <= 0 || out_dim <= 0 || local_experts != kKimiLocalExperts ||
      group_size != kKimiInt4GroupSize || (in_dim % group_size) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  int scale_k = in_dim / group_size;
  size_t elements_per_expert = static_cast<size_t>(out_dim) * scale_k;
  if ((elements_per_expert % 64) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t total_elements = static_cast<size_t>(local_experts) * elements_per_expert;
  dim3 block(256);
  dim3 grid(static_cast<unsigned>((total_elements + block.x - 1) / block.x));
  kimi_marlin_reorder_scale_kernel<<<grid, block, 0, stream>>>(
      weight_scale_checkpoint, weight_scale_marlin, out_dim, scale_k, total_elements);
  cudaError_t err = cudaPeekAtLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult kimi_marlin_int4_fuse_w13_cuda(
    const uint8_t* gate_weight_packed_marlin,
    const uint8_t* up_weight_packed_marlin,
    uint8_t* w13_weight_packed_marlin,
    const __nv_bfloat16* gate_scale_marlin,
    const __nv_bfloat16* up_scale_marlin,
    __nv_bfloat16* w13_scale_marlin,
    int in_dim,
    int intermediate_dim,
    int local_experts,
    int group_size,
    cudaStream_t stream) {
  if (gate_weight_packed_marlin == nullptr || up_weight_packed_marlin == nullptr ||
      w13_weight_packed_marlin == nullptr || gate_scale_marlin == nullptr ||
      up_scale_marlin == nullptr || w13_scale_marlin == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!kimi_marlin_common_shape_ok(in_dim, intermediate_dim, local_experts, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  int weight_cols = intermediate_dim * 2;
  size_t weight_u32 =
      static_cast<size_t>(local_experts) * (in_dim / kMarlinTileK) * weight_cols;
  dim3 block(256);
  dim3 weight_grid(static_cast<unsigned>((weight_u32 + block.x - 1) / block.x));
  kimi_marlin_fuse_w13_weight_kernel<<<weight_grid, block, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(gate_weight_packed_marlin),
      reinterpret_cast<const uint32_t*>(up_weight_packed_marlin),
      reinterpret_cast<uint32_t*>(w13_weight_packed_marlin), in_dim, intermediate_dim);
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) return CUDA_ERROR_INVALID_VALUE;

  size_t scale_elements =
      static_cast<size_t>(local_experts) * (in_dim / group_size) * intermediate_dim;
  dim3 scale_grid(static_cast<unsigned>((scale_elements + block.x - 1) / block.x));
  kimi_marlin_fuse_w13_scale_kernel<<<scale_grid, block, 0, stream>>>(
      gate_scale_marlin, up_scale_marlin, w13_scale_marlin, in_dim, intermediate_dim);
  err = cudaPeekAtLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

}  // extern "C"
