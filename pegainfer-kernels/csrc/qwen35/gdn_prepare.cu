#include "common.cuh"

#include <cuda.h>

namespace {

constexpr int kHeadDim = 128;
constexpr int kThreads = 128;

__device__ __forceinline__ float block_sum_128(float value) {
    __shared__ float warp_sums[4];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    return warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
}

// Production Hv32 specialization. One block owns one native Q or K head and
// the corresponding V head:
//
//   item [0,16)  -> Q[item] + V[item]
//   item [16,32) -> K[item-16] + V[item]
//
// Q and K retain independent reductions and output layouts. Pairing each with
// one V head removes the separate 32 V CTAs without expanding Q/K.
__global__ void gdn_prefill_native_prepare_hv32_kernel(
    const __nv_bfloat16* __restrict__ qkv,       // [T, 64*D]
    const __nv_bfloat16* __restrict__ b_proj,    // [T, 32]
    const __nv_bfloat16* __restrict__ a_proj,    // [T, 32]
    const __nv_bfloat16* __restrict__ dt_bias,   // [32]
    const float* __restrict__ a_log,             // [32]
    __nv_bfloat16* __restrict__ q_out,           // [T, 16, D]
    __nv_bfloat16* __restrict__ k_out,           // [T, 16, D]
    __nv_bfloat16* __restrict__ v_out,           // [T, 32, D]
    float* __restrict__ alpha_out,               // [T, 32]
    float* __restrict__ beta_out,                // [T, 32]
    int qkv_dim,
    int tokens) {
    const int token = blockIdx.x;
    const int item = blockIdx.y;
    const int d = threadIdx.x;
    if (token >= tokens) {
        return;
    }

    constexpr int kHq = 16;
    constexpr int kHk = 16;
    constexpr int kHv = 32;
    const bool is_q = item < kHq;
    const int qk_head = is_q ? item : item - kHq;
    const int v_head = item;
    const size_t token_base = static_cast<size_t>(token) * qkv_dim;
    const size_t qk_base = is_q ? 0 : static_cast<size_t>(kHq) * kHeadDim;
    const size_t v_base = static_cast<size_t>(kHq + kHk) * kHeadDim;

    const float qk_value =
        __bfloat162float(qkv[token_base + qk_base + qk_head * kHeadDim + d]);
    const __nv_bfloat16 v = qkv[token_base + v_base + v_head * kHeadDim + d];

    const float inv_norm = rsqrtf(block_sum_128(qk_value * qk_value) + 1.0e-12f);
    const __nv_bfloat16 normalized = __float2bfloat16(qk_value * inv_norm);
    if (is_q) {
        q_out[(static_cast<size_t>(token) * kHq + qk_head) * kHeadDim + d] =
            normalized;
    } else {
        k_out[(static_cast<size_t>(token) * kHk + qk_head) * kHeadDim + d] =
            normalized;
    }
    v_out[(static_cast<size_t>(token) * kHv + v_head) * kHeadDim + d] = v;

    if (d == 0) {
        const size_t gate_offset = static_cast<size_t>(token) * kHv + v_head;
        const float a = __bfloat162float(a_proj[gate_offset]);
        const float b = __bfloat162float(b_proj[gate_offset]);
        const float bias = __bfloat162float(dt_bias[v_head]);
        const float log_a = a_log[v_head];

        const float x = a + bias;
        const float softplus =
            x > 20.0f ? x : (x < -20.0f ? expf(x) : log1pf(expf(x)));
        const float log_alpha = -expf(log_a) * softplus;
        alpha_out[gate_offset] = expf(log_alpha);
        const float exp_b = expf(b < 0.0f ? b : -b);
        beta_out[gate_offset] =
            b >= 0.0f ? 1.0f / (1.0f + exp_b) : exp_b / (1.0f + exp_b);
    }

}

CUresult map_cuda_error(cudaError_t error) {
    if (error == cudaSuccess) {
        return CUDA_SUCCESS;
    }
    if (error == cudaErrorInvalidValue || error == cudaErrorInvalidDevicePointer) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_ERROR_UNKNOWN;
}

}  // namespace

extern "C" CUresult gated_delta_rule_prefill_native_prepare_cuda(
    const __nv_bfloat16* qkv,
    const __nv_bfloat16* b_proj,
    const __nv_bfloat16* a_proj,
    const __nv_bfloat16* dt_bias,
    const float* a_log,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    __nv_bfloat16* v_out,
    float* alpha_out,
    float* beta_out,
    int tokens,
    cudaStream_t stream) {
    constexpr int kHq = 16;
    constexpr int kHk = 16;
    constexpr int kHv = 32;
    constexpr int kQkvDim = (kHq + kHk + kHv) * kHeadDim;
    if (qkv == nullptr || b_proj == nullptr || a_proj == nullptr || dt_bias == nullptr ||
        a_log == nullptr || q_out == nullptr || k_out == nullptr || v_out == nullptr ||
        alpha_out == nullptr || beta_out == nullptr ||
        tokens <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const dim3 grid(tokens, kHv);
    gdn_prefill_native_prepare_hv32_kernel<<<grid, kThreads, 0, stream>>>(
        qkv, b_proj, a_proj, dt_bias, a_log, q_out, k_out, v_out, alpha_out,
        beta_out, kQkvDim, tokens);
    return map_cuda_error(cudaGetLastError());
}
