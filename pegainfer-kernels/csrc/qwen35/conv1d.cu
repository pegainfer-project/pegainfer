#include "common.cuh"
#include <stdint.h>

// ============================================================================
// Causal Depthwise Conv1d for Gated Delta Net linear attention
//
// Prefill: Parallel causal conv1d over the entire sequence.
// ============================================================================

#define CONV1D_BLOCK 256

// ============================================================================
// Prefill kernel: parallel causal conv1d over sequence
// x_seq: [num_channels, seq_len] bf16 (column-major: token i at offset i * num_channels)
// Actually stored as [seq_len * num_channels] row-major (token i starts at i * num_channels)
// out_seq: [seq_len * num_channels] bf16
// conv_state: [num_channels, K-1] bf16 (updated with last K-1 values)
// ============================================================================
__global__ void conv1d_prefill_kernel(
    const __nv_bfloat16* __restrict__ x_seq,
    const __nv_bfloat16* __restrict__ conv_weight,
    __nv_bfloat16* __restrict__ conv_state,
    __nv_bfloat16* __restrict__ out_seq,
    int num_channels,
    int seq_len,
    int kernel_size
) {
    // Each thread handles one (channel, position) pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_channels * seq_len;
    if (idx >= total) return;

    int c = idx % num_channels;
    int t = idx / num_channels;

    int state_width = kernel_size - 1;

    // Compute causal conv1d at position t for channel c
    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        int src_t = t - (kernel_size - 1) + k;  // position in sequence
        float val;
        if (src_t < 0) {
            // Read from conv_state (pre-existing state)
            int state_idx = state_width + src_t;  // maps [-state_width, -1] → [0, state_width-1]
            if (state_idx >= 0) {
                val = __bfloat162float(conv_state[c * state_width + state_idx]);
            } else {
                val = 0.0f;  // beyond state buffer
            }
        } else {
            val = __bfloat162float(x_seq[src_t * num_channels + c]);
        }
        sum += val * __bfloat162float(conv_weight[c * kernel_size + k]);
    }

    // Match HF/PyTorch: conv1d writes bf16, then SiLU consumes bf16 input.
    float sum_bf16 = __bfloat162float(__float2bfloat16(sum));
    float silu_out = sum_bf16 / (1.0f + expf(-sum_bf16));
    out_seq[t * num_channels + c] = __float2bfloat16(silu_out);

    // Last (state_width) tokens update conv_state
    // Only the last thread for each channel updates state
    if (t == seq_len - 1) {
        float old_state[4];
        for (int i = 0; i < state_width; i++) {
            old_state[i] = __bfloat162float(conv_state[c * state_width + i]);
        }
        for (int i = 0; i < state_width; i++) {
            int src_t = seq_len - state_width + i;
            if (src_t >= 0) {
                conv_state[c * state_width + i] = x_seq[src_t * num_channels + c];
            } else {
                int state_idx = state_width + src_t;  // maps [-state_width, -1] → [0, state_width-1]
                conv_state[c * state_width + i] =
                    state_idx >= 0 ? __float2bfloat16(old_state[state_idx]) : __float2bfloat16(0.0f);
            }
        }
    }
}

__global__ void conv1d_decode_batch_kernel(
    const __nv_bfloat16* __restrict__ x_batch,
    const __nv_bfloat16* __restrict__ conv_weight,
    const uint64_t* __restrict__ conv_state_ptrs,
    __nv_bfloat16* __restrict__ out_batch,
    int num_channels,
    int batch_size,
    int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_channels * batch_size;
    if (idx >= total) return;

    int c = idx % num_channels;
    int slot = idx / num_channels;
    int state_width = kernel_size - 1;
    const __nv_bfloat16* x = x_batch + (size_t)slot * num_channels;
    __nv_bfloat16* conv_state =
        reinterpret_cast<__nv_bfloat16*>(static_cast<uintptr_t>(conv_state_ptrs[slot]));

    float old_state[4];
    #pragma unroll
    for (int i = 0; i < state_width; i++) {
        old_state[i] = __bfloat162float(conv_state[c * state_width + i]);
    }

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        if (k >= kernel_size) break;
        float val = (k == kernel_size - 1)
            ? __bfloat162float(x[c])
            : old_state[k];
        sum += val * __bfloat162float(conv_weight[c * kernel_size + k]);
    }

    float sum_bf16 = __bfloat162float(__float2bfloat16(sum));
    float silu_out = sum_bf16 / (1.0f + expf(-sum_bf16));
    out_batch[(size_t)slot * num_channels + c] = __float2bfloat16(silu_out);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (i >= state_width) break;
        conv_state[c * state_width + i] =
            (i + 1 < state_width) ? __float2bfloat16(old_state[i + 1]) : x[c];
    }
}

extern "C" {

void conv1d_prefill_cuda(
    const __nv_bfloat16* x_seq,
    const __nv_bfloat16* conv_weight,
    __nv_bfloat16* conv_state,
    __nv_bfloat16* out_seq,
    int num_channels,
    int seq_len,
    int kernel_size,
    cudaStream_t stream
) {
    int total = num_channels * seq_len;
    int blocks = (total + CONV1D_BLOCK - 1) / CONV1D_BLOCK;
    conv1d_prefill_kernel<<<blocks, CONV1D_BLOCK, 0, stream>>>(
        x_seq, conv_weight, conv_state, out_seq, num_channels, seq_len, kernel_size
    );
}

void conv1d_decode_batch_cuda(
    const __nv_bfloat16* x_batch,
    const __nv_bfloat16* conv_weight,
    const uint64_t* conv_state_ptrs,
    __nv_bfloat16* out_batch,
    int num_channels,
    int batch_size,
    int kernel_size,
    cudaStream_t stream
) {
    int total = num_channels * batch_size;
    int blocks = (total + CONV1D_BLOCK - 1) / CONV1D_BLOCK;
    conv1d_decode_batch_kernel<<<blocks, CONV1D_BLOCK, 0, stream>>>(
        x_batch, conv_weight, conv_state_ptrs, out_batch,
        num_channels, batch_size, kernel_size
    );
}

} // extern "C"
