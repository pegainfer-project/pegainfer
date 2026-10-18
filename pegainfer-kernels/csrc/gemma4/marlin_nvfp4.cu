// NVFP4 expert GEMM for Gemma 4, over the vendored Marlin MoE kernel.
//
// The template already carries this weight type — e2m1 values with one e4m3
// scale per sixteen, which is `group_blocks == 1` — so all this file adds is
// the kernel table for it and an entry that passes the per-tensor scale the
// INT4 path has no use for.
//
// Marlin reads its own weight and scale layouts, which the checkpoint's are
// not: `marlin_nvfp4_prepare.cu` produces them once at load.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define MARLIN_NAMESPACE_NAME pegainfer_gemma4_marlin_nvfp4
#include "vllm_marlin/moe/marlin_moe_wna16/kernel.h"
#include "vllm_marlin/moe/marlin_moe_wna16/marlin_template.h"
#include "moe_wna16_launch.cuh"

namespace pegainfer_gemma4_marlin_nvfp4 {

// One e4m3 scale per sixteen values is one 16x16 block, so `group_blocks` is
// fixed at 1 and the whole table exists at that one value.
#define GEMMA4_MARLIN_GET_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, \
                             M_BLOCK_SIZE_8, NUM_THREADS)                       \
  else if (thread_m_blocks == THREAD_M_BLOCKS &&                                \
           thread_n_blocks == THREAD_N_BLOCKS &&                                \
           thread_k_blocks == THREAD_K_BLOCKS &&                                \
           m_block_size_8 == M_BLOCK_SIZE_8 && group_blocks == 1 &&             \
           num_threads == NUM_THREADS) {                                        \
    kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),                   \
                    vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), NUM_THREADS,     \
                    THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,          \
                    M_BLOCK_SIZE_8, pipe_stages, 1, false, true>;               \
  }

#define GEMMA4_MARLIN_GET_IF_M1(N_BLOCKS, K_BLOCKS, NUM_THREADS) \
  GEMMA4_MARLIN_GET_IF(1, N_BLOCKS, K_BLOCKS, true, NUM_THREADS) \
  GEMMA4_MARLIN_GET_IF(1, N_BLOCKS, K_BLOCKS, false, NUM_THREADS)

// The expert projections are 704 and 2816 wide over 2816 and 704 deep. Only a
// 64-wide output tile divides 704, and only a 64-deep one divides it, so the
// table below is narrower than a model whose widths are multiples of 128 needs.
const ThreadConfig kSmallBatch[] = {
    {128, 64, 128},
    {64, 128, 128},
};

// Prefill-sized dispatches share decode's tiles: a 64-wide tile has no
// correct 256-thread specialization, and a 256-thread, 256-wide tile for the
// down projection is byte-identical and no faster.
const ThreadConfigs kTables{kSmallBatch, 2, kSmallBatch, 2};

MarlinFuncPtr get_nvfp4_kernel(
    int thread_m_blocks,
    int thread_n_blocks,
    int thread_k_blocks,
    bool m_block_size_8,
    int group_blocks,
    int num_threads) {
  MarlinFuncPtr kernel = MarlinDefault;
  if (false) {
  }
  GEMMA4_MARLIN_GET_IF_M1(4, 8, 128)
  GEMMA4_MARLIN_GET_IF_M1(8, 4, 128)
  GEMMA4_MARLIN_GET_IF(4, 4, 8, false, 128)
  GEMMA4_MARLIN_GET_IF(4, 8, 4, false, 128)
  return kernel;
}

}  // namespace pegainfer_gemma4_marlin_nvfp4

extern "C" {

// `b_scales` holds the prepared e4m3 group scales and `global_scale` the
// per-tensor factor, both as `marlin_nvfp4_prepare` left them. `size_n` is
// the projection's output width and `size_k` its input width.
CUresult gemma4_marlin_nvfp4_moe_cuda(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    float* c_tmp,
    const uint8_t* b_qweight,
    const uint8_t* b_scales,
    const float* global_scale,
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
    int sm_count,
    cudaStream_t stream) {
  if (global_scale == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return pegainfer_gemma4_marlin_nvfp4::launch_marlin_moe_gemm(
      input, output, c_tmp, b_qweight, b_scales, global_scale, workspace,
      sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights,
      workspace_len, sorted_token_ids_len, moe_block_size, top_k,
      mul_topk_weights, size_m, size_n, size_k, 16, sm_count,
      pegainfer_gemma4_marlin_nvfp4::kTables,
      pegainfer_gemma4_marlin_nvfp4::get_nvfp4_kernel, stream);
}

}  // extern "C"
