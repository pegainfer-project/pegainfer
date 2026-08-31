#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define MARLIN_NAMESPACE_NAME pegainfer_kimi_marlin_moe_wna16
#include "vllm_marlin/moe/marlin_moe_wna16/kernel.h"
#include "vllm_marlin/moe/marlin_moe_wna16/marlin_template.h"
#include "moe_wna16_launch.cuh"

namespace pegainfer_kimi_marlin_moe_wna16 {

const ThreadConfig kSmallBatch[] = {
    {128, 128, 256},
    {64, 128, 128},
};

const ThreadConfig kLargeBatch[] = {
    {64, 256, 256},
    {64, 128, 128},
};

const ThreadConfigs kTables{kSmallBatch, 2, kLargeBatch, 2};

constexpr int kKimiLocalExperts = 48;
constexpr int kKimiGroupSize = 32;
#define KIMI_MARLIN_GET_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, \
                           M_BLOCK_SIZE_8, GROUP_BLOCKS, NUM_THREADS)         \
  else if (thread_m_blocks == THREAD_M_BLOCKS &&                               \
           thread_n_blocks == THREAD_N_BLOCKS &&                               \
           thread_k_blocks == THREAD_K_BLOCKS &&                               \
           m_block_size_8 == M_BLOCK_SIZE_8 && group_blocks == GROUP_BLOCKS && \
           num_threads == NUM_THREADS) {                                       \
    kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(),                    \
                    vllm::kBFloat16.id(), vllm::kBFloat16.id(), NUM_THREADS,   \
                    THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,         \
                    M_BLOCK_SIZE_8, pipe_stages, GROUP_BLOCKS, false, false>;  \
  }

#define KIMI_MARLIN_COMMON_GET_IF_M1(N_BLOCKS, K_BLOCKS, NUM_THREADS)    \
  KIMI_MARLIN_GET_IF(1, N_BLOCKS, K_BLOCKS, true, 2, NUM_THREADS)         \
  KIMI_MARLIN_GET_IF(1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS)

#define KIMI_MARLIN_COMMON_GET_IF_M234(N_BLOCKS, K_BLOCKS, NUM_THREADS) \
  KIMI_MARLIN_GET_IF(2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS)       \
  KIMI_MARLIN_GET_IF(3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS)       \
  KIMI_MARLIN_GET_IF(4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS)

MarlinFuncPtr get_marlin_kernel(
    int thread_m_blocks,
    int thread_n_blocks,
    int thread_k_blocks,
    bool m_block_size_8,
    int group_blocks,
    int num_threads) {
  MarlinFuncPtr kernel = MarlinDefault;
  if (false) {
  }
  KIMI_MARLIN_COMMON_GET_IF_M1(8, 8, 256)
  KIMI_MARLIN_COMMON_GET_IF_M1(8, 4, 128)
  KIMI_MARLIN_COMMON_GET_IF_M234(16, 4, 256)
  KIMI_MARLIN_COMMON_GET_IF_M234(8, 4, 128)
  return kernel;
}

CUresult launch_marlin_gemm(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    float* c_tmp,
    const uint8_t* b_qweight,
    const __nv_bfloat16* b_scales,
    int* workspace,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded,
    const float* topk_weights,
    int workspace_len,
    int sorted_token_ids_len,
    int moe_block_size,
    int top_k,
    bool mul_topk_weights,
    int size_m,
    int size_n,
    int size_k,
    int local_experts,
    int group_size,
    int sm_count,
    cudaStream_t stream) {
  if (local_experts != kKimiLocalExperts || group_size != kKimiGroupSize) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return launch_marlin_moe_gemm(
      input, output, c_tmp, b_qweight, b_scales, nullptr, workspace,
      sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights,
      workspace_len, sorted_token_ids_len, moe_block_size, top_k,
      mul_topk_weights, size_m, size_n, size_k, group_size, sm_count, kTables,
      get_marlin_kernel, stream);
}

__global__ void swiglu_w13_kernel(
    const __nv_bfloat16* __restrict__ w13,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int intermediate_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * intermediate_dim;
  if (idx >= total) return;
  int row = idx / intermediate_dim;
  int col = idx - row * intermediate_dim;
  const __nv_bfloat16* row_ptr = w13 + row * (2 * intermediate_dim);
  float gate = __bfloat162float(row_ptr[col]);
  float up = __bfloat162float(row_ptr[intermediate_dim + col]);
  float silu = gate / (1.0f + expf(-gate));
  float silu_bf16 = __bfloat162float(__float2bfloat16(silu));
  out[idx] = __float2bfloat16(silu_bf16 * up);
}

// Grid-stride over the device-resident row count: the launch grid is fixed
// (occupancy-sized), so cost tracks the actual expanded rows instead of the
// worst-case capacity. A capacity-sized grid spent more time draining empty
// blocks than computing at decode shapes (68k blocks for ~512 live rows).
__global__ void swiglu_w13_expanded_kernel(
    const __nv_bfloat16* __restrict__ w13,
    __nv_bfloat16* __restrict__ out,
    const int32_t* __restrict__ num_tokens_post_padded,
    int max_rows,
    int intermediate_dim) {
  int actual_rows = num_tokens_post_padded[0];
  if (actual_rows <= 0) return;
  // The routing builder guarantees actual_rows <= capacity == max_rows; the
  // clamp keeps a broken device-side count from writing out of bounds.
  if (actual_rows > max_rows) actual_rows = max_rows;
  int total = actual_rows * intermediate_dim;
  int stride = gridDim.x * blockDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    int row = idx / intermediate_dim;
    int col = idx - row * intermediate_dim;
    const __nv_bfloat16* row_ptr = w13 + row * (2 * intermediate_dim);
    float gate = __bfloat162float(row_ptr[col]);
    float up = __bfloat162float(row_ptr[intermediate_dim + col]);
    float silu = gate / (1.0f + expf(-gate));
    float silu_bf16 = __bfloat162float(__float2bfloat16(silu));
    out[idx] = __float2bfloat16(silu_bf16 * up);
  }
}

__global__ void sum_topk_rows_kernel(
    const __nv_bfloat16* __restrict__ route_output,
    float* __restrict__ out,
    int active_tokens,
    int topk,
    int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = active_tokens * hidden_dim;
  if (idx >= total) return;
  int token = idx / hidden_dim;
  int dim = idx - token * hidden_dim;
  float acc = 0.0f;
  for (int k = 0; k < topk; ++k) {
    acc += __bfloat162float(route_output[(token * topk + k) * hidden_dim + dim]);
  }
  out[idx] = acc;
}

}  // namespace pegainfer_kimi_marlin_moe_wna16

extern "C" {

CUresult kimi_marlin_wna16_gemm_cuda(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    float* c_tmp,
    const uint8_t* b_qweight,
    const __nv_bfloat16* b_scales,
    int* workspace,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded,
    const float* topk_weights,
    int workspace_len,
    int sorted_token_ids_len,
    int moe_block_size,
    int top_k,
    bool mul_topk_weights,
    int size_m,
    int size_n,
    int size_k,
    int local_experts,
    int group_size,
    int sm_count,
    cudaStream_t stream) {
  return pegainfer_kimi_marlin_moe_wna16::launch_marlin_gemm(
      input, output, c_tmp, b_qweight, b_scales, workspace, sorted_token_ids,
      expert_ids, num_tokens_post_padded, topk_weights, workspace_len,
      sorted_token_ids_len, moe_block_size, top_k, mul_topk_weights, size_m,
      size_n, size_k, local_experts, group_size, sm_count, stream);
}

CUresult kimi_marlin_w13_swiglu_cuda(
    const __nv_bfloat16* w13,
    __nv_bfloat16* out,
    int rows,
    int intermediate_dim,
    cudaStream_t stream) {
  if (w13 == nullptr || out == nullptr || rows < 0 || intermediate_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rows == 0) return CUDA_SUCCESS;
  constexpr int threads = 256;
  int total = rows * intermediate_dim;
  int blocks = (total + threads - 1) / threads;
  pegainfer_kimi_marlin_moe_wna16::swiglu_w13_kernel<<<blocks, threads, 0, stream>>>(
      w13, out, rows, intermediate_dim);
  return pegainfer_kimi_marlin_moe_wna16::last_error_to_cu(cudaPeekAtLastError());
}

CUresult kimi_marlin_w13_swiglu_expanded_cuda(
    const __nv_bfloat16* w13,
    __nv_bfloat16* out,
    const int32_t* num_tokens_post_padded,
    int max_rows,
    int intermediate_dim,
    int sm_count,
    cudaStream_t stream) {
  if (w13 == nullptr || out == nullptr || num_tokens_post_padded == nullptr ||
      max_rows < 0 || intermediate_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (max_rows == 0) return CUDA_SUCCESS;
  // Same convention as the Marlin GEMM launcher: sm_count <= 0 resolves the
  // current device's multiprocessor count.
  if (sm_count <= 0) {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return CUDA_ERROR_INVALID_VALUE;
    err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess || sm_count <= 0) return CUDA_ERROR_INVALID_VALUE;
  }
  constexpr int threads = 256;
  // Fixed occupancy-sized grid; the kernel grid-strides up to the actual
  // device-side row count (<= max_rows by the recv buffer contract).
  int max_blocks = (max_rows * intermediate_dim + threads - 1) / threads;
  int blocks = sm_count * 8;
  if (blocks > max_blocks) blocks = max_blocks;
  pegainfer_kimi_marlin_moe_wna16::swiglu_w13_expanded_kernel<<<blocks, threads, 0, stream>>>(
      w13, out, num_tokens_post_padded, max_rows, intermediate_dim);
  return pegainfer_kimi_marlin_moe_wna16::last_error_to_cu(cudaPeekAtLastError());
}

CUresult kimi_marlin_sum_topk_rows_f32_cuda(
    const __nv_bfloat16* route_output,
    float* out,
    int active_tokens,
    int topk,
    int hidden_dim,
    cudaStream_t stream) {
  if (route_output == nullptr || out == nullptr || active_tokens < 0 || topk <= 0 ||
      hidden_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (active_tokens == 0) return CUDA_SUCCESS;
  constexpr int threads = 256;
  int total = active_tokens * hidden_dim;
  int blocks = (total + threads - 1) / threads;
  pegainfer_kimi_marlin_moe_wna16::sum_topk_rows_kernel<<<blocks, threads, 0, stream>>>(
      route_output, out, active_tokens, topk, hidden_dim);
  return pegainfer_kimi_marlin_moe_wna16::last_error_to_cu(cudaPeekAtLastError());
}

}  // extern "C"
