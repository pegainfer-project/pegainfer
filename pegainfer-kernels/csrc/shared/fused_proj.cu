#include "common.cuh"

// ============================================================================
// Split contiguous QKV projection output into compact Q/K/V buffers.
//
// All tensors use the HiddenStates column-major layout: token `col` is a
// contiguous column. This kernel is a BF16 bitwise copy only; it deliberately
// performs no normalization, head reordering, or floating-point conversion.
// ============================================================================

__global__ void split_qkv_kernel(
    const __nv_bfloat16 *__restrict__ qkv, // [Q+2*KV, tokens]
    __nv_bfloat16 *__restrict__ q,          // [Q, tokens]
    __nv_bfloat16 *__restrict__ k,          // [KV, tokens]
    __nv_bfloat16 *__restrict__ v,          // [KV, tokens]
    int q_dim, int kv_dim, int qkv_dim, int tokens) {

  int total = qkv_dim * tokens;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    int col = idx / qkv_dim;
    int row = idx % qkv_dim;
    __nv_bfloat16 value = qkv[idx];
    if (row < q_dim) {
      q[col * q_dim + row] = value;
    } else if (row < q_dim + kv_dim) {
      k[col * kv_dim + row - q_dim] = value;
    } else {
      v[col * kv_dim + row - q_dim - kv_dim] = value;
    }
  }
}

// ============================================================================
// Fused SiLU-mul from combined [2*I, bs] gate+up buffer.
// Column-major: token j at offset j * 2*I.
//   gate = combined[j * 2*I + i]     for i in [0, I)
//   up   = combined[j * 2*I + I + i] for i in [0, I)
//   out[j * I + i] = bf16(silu(gate)) * up, rounded to bf16
// ============================================================================

__global__ void silu_mul_fused_kernel(
    const __nv_bfloat16 *__restrict__ gate_up, // [2*I, bs] col-major
    __nv_bfloat16 *__restrict__ out,            // [I, bs] col-major
    int intermediate_size, int bs) {

  int total = intermediate_size * bs;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    int col = idx / intermediate_size;
    int row = idx % intermediate_size;

    int src_offset = col * 2 * intermediate_size;
    float g = __bfloat162float(gate_up[src_offset + row]);
    float u = __bfloat162float(gate_up[src_offset + intermediate_size + row]);

    float silu_g = g / (1.0f + expf(-g));
    float silu_bf16 = __bfloat162float(__float2bfloat16(silu_g));
    out[idx] = __float2bfloat16(silu_bf16 * u);
  }
}

extern "C" {

int split_qkv_cuda(
    const __nv_bfloat16 *qkv, __nv_bfloat16 *q,
    __nv_bfloat16 *k, __nv_bfloat16 *v,
    int q_dim, int kv_dim, int tokens, cudaStream_t stream) {
  int qkv_dim = q_dim + 2 * kv_dim;
  int total = qkv_dim * tokens;
  int block = 256;
  int grid = (total + block - 1) / block;
  split_qkv_kernel<<<grid, block, 0, stream>>>(
      qkv, q, k, v, q_dim, kv_dim, qkv_dim, tokens);
  return static_cast<int>(cudaGetLastError());
}

int silu_mul_fused_cuda(
    const __nv_bfloat16 *gate_up, __nv_bfloat16 *out,
    int intermediate_size, int bs, cudaStream_t stream) {
  int total = intermediate_size * bs;
  int block = 256;
  int grid = (total + block - 1) / block;
  silu_mul_fused_kernel<<<grid, block, 0, stream>>>(
      gate_up, out, intermediate_size, bs);
  return static_cast<int>(cudaGetLastError());
}

} // extern "C"
