#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

CUresult marlin_moe_align_block_size_cuda(
    const int* topk_idx,
    int* sorted_token_ids,
    int* expert_ids,
    int* num_tokens_post_padded,
    uint32_t* expert_offsets,
    uint32_t* unused_expert_cursor,
    int active_tokens,
    int topk,
    int global_start,
    int local_experts,
    int block_size,
    int max_padded_tokens,
    int max_m_blocks,
    cudaStream_t stream);

CUresult marlin_repack_4bit_cuda(
    const uint8_t* src,
    uint8_t* dst,
    int experts,
    int in_dim,
    int out_dim,
    cudaStream_t stream);

}  // extern "C"
