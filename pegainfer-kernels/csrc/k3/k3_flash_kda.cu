// Kimi-K3 chunked-prefill KDA core: C ABI over the vendored FlashKDA kernel.
//
// FlashKDA (third_party/flash-kda, MIT, MoonshotAI — see PROVENANCE.md) is the
// upstream chunkwise Kimi-Delta-Attention forward: chunk-16 intra-tile
// preprocessing (kernel 1) and the inter-chunk state recurrence + output
// (kernel 2), CUTLASS/CuTe on SM90 TMA. This TU includes the vendored launch
// layer directly — one translation unit, implicit instantiation — and exposes
// the single configuration pegainfer-k3 launches: D = 128, fp32 carried state
// in and out, non-varlen (one sequence per call, `B = 1`).
//
// Contract notes (mirroring the upstream torch binding, which is not vendored):
//   - q/k/v/g/out are [T, H, 128] bf16, contiguous — exactly the engine's
//     [T, 12288] rows. g is the pre-activation gate projection; the kernel
//     applies dt_bias, exp(A_log), sigmoid and the lower-bound scale itself.
//   - beta is [H, T] bf16 (transposed; upstream binding does `.t()`), hence
//     the transpose helper below.
//   - state is [H, 128, 128] f32, [head, v, k] — the engine's recurrent slab
//     layout as-is.
//   - workspace: the upstream `get_workspace_size` arithmetic, reproduced
//     verbatim (the +N tile is a varlen upper bound; harmless here).
//   - gate_scale = lower_bound * log2(e): the kernel works in exp2 space.
//
// Compiled for the accelerated SM targets (sm_90a..sm_121a family) when
// available; otherwise every entry point is a NOT_SUPPORTED stub so the build
// stays green off-Blackwell.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#ifdef K3_FLASH_KDA_SM90A
#include "fwd_launch.cu"
#include "k3_flash_kda_md.cuh"
#endif

// Not in an anonymous namespace: nvcc's device-stub generation trips over an
// anonymous-namespace __global__ when an included header (cute) opens its own.
static CUresult k3_flash_kda_map_cuda_error(cudaError_t err) {
    if (err == cudaSuccess) return CUDA_SUCCESS;
    if (err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
    if (err == cudaErrorNotSupported) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_ERROR_LAUNCH_FAILED;
}

static __global__ void k3_flash_kda_beta_transpose_kernel(
    const __nv_bfloat16* __restrict__ beta_th,
    __nv_bfloat16* __restrict__ beta_ht,
    int t,
    int h
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t * h) {
        return;
    }
    int row = idx / h;
    int col = idx - row * h;
    beta_ht[col * t + row] = beta_th[idx];
}

extern "C" {

// Upstream get_workspace_size, N = 1.
long long k3_flash_kda_workspace_bytes(int t_total, int h) {
    const long long chunk = 16;
    const long long d = 128;
    const long long n = 1;
    long long total_tiles = (t_total + chunk - 1) / chunk + n;
    long long per_tile_bytes = 3 * chunk * d * 2 + d * 4 + 2 * chunk * chunk * 2;
    long long tile_prefix_bytes = ((n + 1) * 4 + 127) / 128 * 128;
    return h * total_tiles * per_tile_bytes + tile_prefix_bytes;
}

// beta [T, H] bf16 -> beta [H, T] bf16 (the layout kernel 1's 1D TMA loads).
CUresult k3_flash_kda_beta_transpose(
    const void* beta_th,
    void* beta_ht,
    int t_total,
    int h,
    cudaStream_t stream
) {
    int total = t_total * h;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k3_flash_kda_beta_transpose_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(beta_th),
        static_cast<__nv_bfloat16*>(beta_ht),
        t_total,
        h
    );
    return k3_flash_kda_map_cuda_error(cudaGetLastError());
}

// One sequence of `t_total` tokens through the chunkwise KDA forward,
// f32 state carried in and out.
CUresult k3_flash_kda_fwd(
    const void* q,          // [T, H, 128] bf16
    const void* k,          // [T, H, 128] bf16
    const void* v,          // [T, H, 128] bf16
    const void* g,          // [T, H, 128] bf16, pre-activation gate
    const void* beta_ht,    // [H, T] bf16, beta logits
    const float* a_log,     // [H]
    const float* dt_bias,   // [H, 128]
    const void* state_in,   // [H, 128, 128] f32
    void* state_out,        // [H, 128, 128] f32
    void* out,              // [T, H, 128] bf16
    void* workspace,        // k3_flash_kda_workspace_bytes(t_total, h)
    int t_total,
    int h,
    float scale,
    float lower_bound,
    cudaStream_t stream
) {
#ifndef K3_FLASH_KDA_SM90A
    (void)q; (void)k; (void)v; (void)g; (void)beta_ht; (void)a_log;
    (void)dt_bias; (void)state_in; (void)state_out; (void)out;
    (void)workspace; (void)t_total; (void)h; (void)scale; (void)lower_bound;
    (void)stream;
    return CUDA_ERROR_NOT_SUPPORTED;
#else
    constexpr int kChunk = 16;
    // Non-varlen N=1: tiles are exact (the workspace kept the varlen slack).
    int total_tiles = (t_total + kChunk - 1) / kChunk;
    float gate_scale = lower_bound * 1.4426950408889634f;
    launch_fwd<128, true, true, true, false>(
        static_cast<cutlass::bfloat16_t const*>(q),
        static_cast<cutlass::bfloat16_t const*>(k),
        static_cast<cutlass::bfloat16_t const*>(v),
        static_cast<cutlass::bfloat16_t const*>(g),
        static_cast<cutlass::bfloat16_t const*>(beta_ht),
        state_in,
        scale,
        state_out,
        static_cast<cutlass::bfloat16_t*>(out),
        workspace,
        total_tiles,
        t_total,
        h,
        /*N=*/1,
        /*cu_seqlens=*/nullptr,
        a_log,
        dt_bias,
        gate_scale,
        stream
    );
    return k3_flash_kda_map_cuda_error(cudaGetLastError());
#endif
}

// Fused KCP package forward: one kernel-1 pass plus the dual-state kernel 2
// (k3_flash_kda_md.cuh), producing the segment's affine package in one sweep:
// `state_out_d` = D (real v from zero state) and `state_out_m` = M (v = 0
// from identity state). No token output — a CP middle rank discards it.
// Same operand contract and workspace as k3_flash_kda_fwd.
CUresult k3_flash_kda_fwd_md(
    const void* q,          // [T, H, 128] bf16
    const void* k,          // [T, H, 128] bf16
    const void* v,          // [T, H, 128] bf16
    const void* g,          // [T, H, 128] bf16, pre-activation gate
    const void* beta_ht,    // [H, T] bf16, beta logits
    const float* a_log,     // [H]
    const float* dt_bias,   // [H, 128]
    void* state_out_d,      // [H, 128, 128] f32
    void* state_out_m,      // [H, 128, 128] f32
    void* workspace,        // k3_flash_kda_workspace_bytes(t_total, h)
    int t_total,
    int h,
    float scale,
    float lower_bound,
    cudaStream_t stream
) {
#ifndef K3_FLASH_KDA_SM90A
    (void)q; (void)k; (void)v; (void)g; (void)beta_ht; (void)a_log;
    (void)dt_bias; (void)state_out_d; (void)state_out_m; (void)workspace;
    (void)t_total; (void)h; (void)scale; (void)lower_bound; (void)stream;
    return CUDA_ERROR_NOT_SUPPORTED;
#else
    constexpr int kChunk = 16;
    int total_tiles = (t_total + kChunk - 1) / kChunk;
    float gate_scale = lower_bound * 1.4426950408889634f;
    launch_fwd_md<128>(
        static_cast<cutlass::bfloat16_t const*>(q),
        static_cast<cutlass::bfloat16_t const*>(k),
        static_cast<cutlass::bfloat16_t const*>(v),
        static_cast<cutlass::bfloat16_t const*>(g),
        static_cast<cutlass::bfloat16_t const*>(beta_ht),
        scale,
        static_cast<float*>(state_out_d),
        static_cast<float*>(state_out_m),
        workspace,
        total_tiles,
        t_total,
        h,
        a_log,
        dt_bias,
        gate_scale,
        stream
    );
    return k3_flash_kda_map_cuda_error(cudaGetLastError());
#endif
}

}  // extern "C"
