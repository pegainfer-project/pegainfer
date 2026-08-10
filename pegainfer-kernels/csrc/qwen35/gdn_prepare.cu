#include "common.cuh"

#include <cuda.h>
#include <stdint.h>

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

__device__ __forceinline__ void record_non_finite(float value, uint32_t* status) {
    if (!isfinite(value)) {
        atomicExch(status, 1u);
    }
}

// One block owns one (token, native head). The y-grid is Q heads followed by
// K heads followed by V heads. Q/K are never expanded to the V-head count.
__global__ void gdn_prefill_native_prepare_kernel(
    const __nv_bfloat16* __restrict__ qkv,       // [T, Hq*D + Hk*D + Hv*D]
    const __nv_bfloat16* __restrict__ b_proj,    // [T, Hv]
    const __nv_bfloat16* __restrict__ a_proj,    // [T, Hv]
    const __nv_bfloat16* __restrict__ dt_bias,   // [Hv]
    const float* __restrict__ a_log,             // [Hv]
    __nv_bfloat16* __restrict__ q_out,           // [T, Hq, D]
    __nv_bfloat16* __restrict__ k_out,           // [T, Hk, D]
    __nv_bfloat16* __restrict__ v_out,           // [T, Hv, D]
    float* __restrict__ alpha_out,               // [T, Hv], per-token decay
    float* __restrict__ beta_out,                // [T, Hv]
    uint32_t* __restrict__ non_finite_status,
    int h_q,
    int h_k,
    int h_v,
    int head_dim,
    int qkv_dim,
    int tokens) {
    const int token = blockIdx.x;
    const int item = blockIdx.y;
    const int d = threadIdx.x;
    if (token >= tokens) {
        return;
    }

    const __nv_bfloat16* token_qkv = qkv + static_cast<size_t>(token) * qkv_dim;
    if (item < h_q) {
        const int head = item;
        const float value = __bfloat162float(token_qkv[head * head_dim + d]);
        record_non_finite(value, non_finite_status);
        const float inv_norm = rsqrtf(block_sum_128(value * value) + 1.0e-12f);
        q_out[(static_cast<size_t>(token) * h_q + head) * head_dim + d] =
            __float2bfloat16(value * inv_norm);
        return;
    }

    if (item < h_q + h_k) {
        const int head = item - h_q;
        const size_t k_base = static_cast<size_t>(h_q) * head_dim;
        const float value = __bfloat162float(token_qkv[k_base + head * head_dim + d]);
        record_non_finite(value, non_finite_status);
        const float inv_norm = rsqrtf(block_sum_128(value * value) + 1.0e-12f);
        k_out[(static_cast<size_t>(token) * h_k + head) * head_dim + d] =
            __float2bfloat16(value * inv_norm);
        return;
    }

    const int head = item - h_q - h_k;
    const size_t v_base = static_cast<size_t>(h_q + h_k) * head_dim;
    const __nv_bfloat16 v = token_qkv[v_base + head * head_dim + d];
    const float v_f32 = __bfloat162float(v);
    record_non_finite(v_f32, non_finite_status);
    v_out[(static_cast<size_t>(token) * h_v + head) * head_dim + d] = v;

    if (d == 0) {
        const size_t gate_offset = static_cast<size_t>(token) * h_v + head;
        const float a = __bfloat162float(a_proj[gate_offset]);
        const float b = __bfloat162float(b_proj[gate_offset]);
        const float bias = __bfloat162float(dt_bias[head]);
        const float log_a = a_log[head];
        record_non_finite(a, non_finite_status);
        record_non_finite(b, non_finite_status);
        record_non_finite(bias, non_finite_status);
        record_non_finite(log_a, non_finite_status);

        const float x = a + bias;
        const float softplus =
            x > 20.0f ? x : (x < -20.0f ? expf(x) : log1pf(expf(x)));
        const float log_alpha = -expf(log_a) * softplus;
        alpha_out[gate_offset] = expf(log_alpha);
        const float exp_b = expf(b < 0.0f ? b : -b);
        beta_out[gate_offset] = b >= 0.0f ? 1.0f / (1.0f + exp_b)
                                                  : exp_b / (1.0f + exp_b);
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
    uint32_t* non_finite_status,
    int h_q,
    int h_k,
    int h_v,
    int head_dim,
    int qkv_dim,
    int tokens,
    cudaStream_t stream) {
    if (qkv == nullptr || b_proj == nullptr || a_proj == nullptr || dt_bias == nullptr ||
        a_log == nullptr || q_out == nullptr || k_out == nullptr || v_out == nullptr ||
        alpha_out == nullptr || beta_out == nullptr || non_finite_status == nullptr ||
        h_q != 16 || h_k != 16 || (h_v != 32 && h_v != 48) || head_dim != kHeadDim ||
        qkv_dim != (h_q + h_k + h_v) * head_dim || tokens <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // The chunk owner allocates this status word zeroed once. Every layer ORs
    // into the same sticky status so the host can validate once at the chunk
    // boundary instead of introducing one D2H synchronization per layer.
    const dim3 grid(tokens, h_q + h_k + h_v);
    gdn_prefill_native_prepare_kernel<<<grid, kThreads, 0, stream>>>(
        qkv, b_proj, a_proj, dt_bias, a_log, q_out, k_out, v_out, alpha_out, beta_out,
        non_finite_status, h_q, h_k, h_v, head_dim, qkv_dim, tokens);
    return map_cuda_error(cudaGetLastError());
}
