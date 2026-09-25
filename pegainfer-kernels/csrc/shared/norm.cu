// Norm kernels that have no FlashInfer equivalent.
//
// The standard RMSNorm / FusedAddRMSNorm / Gemma variants have been migrated
// to flashinfer_norm.cu which delegates to FlashInfer's header-only templates.
//
// This file retains only:
//   rms_norm_gated_cuda — per-head RMSNorm with SiLU gate (Qwen3.5 linear attention output).

#include "common.cuh"

// ============================================================================
// Gated RMSNorm for linear attention output:
//   out = rms_norm(x, f32_weight) * silu(gate)
// Per-head normalization: x is [num_heads * head_dim], weight is [head_dim] (broadcast).
// Grid: num_heads blocks, head_dim threads.
//
// Blocks are flattened over (slot, head within the slot), so a head knows its
// slot as `head / heads_per_slot`. `x` and `gate` may each be a band of a fused
// projection and take that tensor's slot stride; `out` is this operator's own
// buffer and keeps the tile's own width as its stride.
// ============================================================================
__global__ void rms_norm_gated_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ gate,
    __nv_bfloat16 *__restrict__ out,
    int heads_per_slot,
    int head_dim,
    int x_stride,
    int gate_stride,
    float eps
) {
  int head = blockIdx.x;
  int tid = threadIdx.x;
  if (tid >= head_dim) return;

  int slot = head / heads_per_slot;
  int offset = head * head_dim + tid;
  int x_offset = slot * x_stride + (head - slot * heads_per_slot) * head_dim + tid;
  int gate_offset = slot * gate_stride + (head - slot * heads_per_slot) * head_dim + tid;

  float x_val = __bfloat162float(x[x_offset]);
  float sq = x_val * x_val;

  #pragma unroll
  for (int off = WARP_SIZE / 2; off > 0; off /= 2) {
    sq += __shfl_down_sync(0xffffffff, sq, off);
  }

  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int num_warps = (head_dim + WARP_SIZE - 1) / WARP_SIZE;

  __shared__ float warp_sums[8];
  if (lane_id == 0) warp_sums[warp_id] = sq;
  __syncthreads();

  __shared__ float s_inv_rms;
  if (tid == 0) {
    float total = 0.0f;
    for (int i = 0; i < num_warps; i++) total += warp_sums[i];
    s_inv_rms = rsqrtf(total / head_dim + eps);
  }
  __syncthreads();

  float normed = x_val * s_inv_rms * weight[tid];

  float g = __bfloat162float(gate[gate_offset]);
  float silu_g = g / (1.0f + expf(-g));

  out[offset] = __float2bfloat16(normed * silu_g);
}

extern "C" {

void rms_norm_gated_cuda(const __nv_bfloat16 *x, const float *weight,
                          const __nv_bfloat16 *gate, __nv_bfloat16 *out,
                          int num_heads, int heads_per_slot, int head_dim,
                          int x_stride, int gate_stride, float eps,
                          cudaStream_t stream) {
  rms_norm_gated_kernel<<<num_heads, head_dim, 0, stream>>>(
      x, weight, gate, out, heads_per_slot, head_dim, x_stride, gate_stride, eps);
}

} // extern "C"
