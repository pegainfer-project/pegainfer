// Norm kernels that have no FlashInfer equivalent.
//
// The standard RMSNorm / FusedAddRMSNorm / Gemma variants have been migrated
// to flashinfer_norm.cu which delegates to FlashInfer's header-only templates.
//
// This file retains only the per-head gated RMSNorm used on a linear-attention
// output, in two activations:
//   rms_norm_gated_cuda         — SiLU gate    (Qwen3.5 Gated DeltaNet)
//   rms_norm_gated_sigmoid_cuda — sigmoid gate (Qwen3.8-Flash-Next, whose
//                                 `output_gate_type` selects this)
// Both are the same kernel; the activation is a template parameter so each
// instantiation compiles to exactly the arithmetic it had before.

#include "common.cuh"

// ============================================================================
// Gated RMSNorm for linear attention output:
//   out = rms_norm(x, f32_weight) * activation(gate)
// Per-head normalization: x is [num_heads * head_dim], weight is [head_dim] (broadcast).
// Grid: num_heads blocks, head_dim threads.
// ============================================================================
enum class GatedNormActivation { kSilu, kSigmoid };

template <GatedNormActivation kActivation>
__global__ void rms_norm_gated_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ gate,
    __nv_bfloat16 *__restrict__ out,
    int head_dim,
    float eps
) {
  int head = blockIdx.x;
  int tid = threadIdx.x;
  if (tid >= head_dim) return;

  int offset = head * head_dim + tid;

  float x_val = __bfloat162float(x[offset]);
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

  float g = __bfloat162float(gate[offset]);
  // The SiLU branch keeps its original `g / (1 + exp(-g))` form rather than
  // `g * sigmoid(g)`: the two are mathematically equal but not bit-identical in
  // fp32, and Qwen3.5's golden gates are bit-exact against this expression.
  float activated = (kActivation == GatedNormActivation::kSilu)
                        ? g / (1.0f + expf(-g))
                        : 1.0f / (1.0f + expf(-g));

  out[offset] = __float2bfloat16(normed * activated);
}

extern "C" {

void rms_norm_gated_cuda(const __nv_bfloat16 *x, const float *weight,
                          const __nv_bfloat16 *gate, __nv_bfloat16 *out,
                          int num_heads, int head_dim, float eps, cudaStream_t stream) {
  rms_norm_gated_kernel<GatedNormActivation::kSilu><<<num_heads, head_dim, 0, stream>>>(x, weight, gate, out, head_dim, eps);
}

void rms_norm_gated_sigmoid_cuda(const __nv_bfloat16 *x, const float *weight,
                                  const __nv_bfloat16 *gate, __nv_bfloat16 *out,
                                  int num_heads, int head_dim, float eps, cudaStream_t stream) {
  rms_norm_gated_kernel<GatedNormActivation::kSigmoid><<<num_heads, head_dim, 0, stream>>>(x, weight, gate, out, head_dim, eps);
}

} // extern "C"
