// Qwen3.8-Flash-Next hyper-connection gated residual: the elementwise tail.
//
// Frozen semantics: transformers 5.16.1 `qwen4_exp`,
// `Qwen4ExpTextGatedResidual.forward`. Reading `n` for `hc_norm`'s grouped
// `(1 + w)` RMSNorm output, `H` for `hidden_size` and `B` for `hc_count`:
//
//   g    = silu(input_mix_weight_down(n) / B)          // [S, lowrank]
//   gate = sigmoid(input_mix_weight_up(g))             // [S, B*H]
//   mix  = (gate * n unflattened to [S, B, H]).mean(-2)  // [S, H]
//   inj  = 2 * sigmoid(block_inject_weight(n) / B)     // [S, B]
//
// The three GEMMs and the grouped norm are not here: they are the shared
// cuBLAS wrapper and `rms_norm_batched_offset_cuda`, and the parity gate
// composes them with these three kernels. A standalone sigmoid and the branch
// combine have no default-feature entry point at all. The scaled SiLU does
// have an equivalent on default features — `scale_bf16_in_place` followed by
// `silu_mul_kernel` against a buffer of ones, two launches; this file's kernel
// folds the pair into one launch and is bitwise identical to it, because the
// reference's scale is a power of two, which makes the intermediate bf16
// rounding the two-launch form adds exact. Each kernel is written to land its
// bf16 roundings where the reference lands them.
//
// ROUNDING CONTRACT. Every module in the reference materialises a bf16 tensor,
// so each of its steps rounds once and the next step reads that rounded value:
//
//   * `down / B` is a bf16 division. `B` is a power of two, so it is exact and
//     equals `f32(down) * (1/B)` — which is what `scale` does here, one rounding
//     for the SiLU or sigmoid that follows.
//   * `silu` / `sigmoid` run in f32 over the bf16 input and cast once.
//   * `gate * n` is a bf16-by-bf16 product: f32 multiply, cast to bf16.
//   * `.mean(-2)` accumulates those *rounded* products in f32, divides by `B`
//     and casts once. `combine_scale` is that division, kept a parameter so a
//     caller can ask for the sum instead — which is what the parity gate's
//     negative control does.
//   * `2 * sigmoid(...)` is exact for the same power-of-two reason, so the gain
//     is applied to the already-rounded sigmoid and rounds again to no effect.
//
// Target: the sm_89-first portable subset — plain SIMT elementwise bodies, no
// TMA, no cluster, no wgmma.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

__device__ __forceinline__ float sigmoid_f32(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// `x / (1 + expf(-x))`, not `x * sigmoid_f32(x)`: torch's SiLU, and the form
// every other silu in this tree uses (`shared/elementwise.cu`,
// `shared/norm.cu`, `shared/fused_proj.cu`, `qwen35/conv1d.cu`,
// `kimi_k2/kimi_marlin_wna16.cu`). The two differ in the last f32 ulp on some
// inputs, which is enough to flip a bf16 cast at a midpoint — so matching the
// tree is what keeps this kernel and the default-feature two-launch equivalent
// bitwise identical.
__device__ __forceinline__ float silu_f32(float x) { return x / (1.0f + expf(-x)); }

// out = bf16(silu(f32(x) * scale)). The reference divides by `hc_count` in bf16
// first; for the power-of-two counts that is exact, so folding it into `scale`
// leaves one rounding — the SiLU's — in the same place.
__global__ void qwen38fn_hc_silu_scaled_kernel(const __nv_bfloat16 *__restrict__ x,
                                               __nv_bfloat16 *__restrict__ out, int n,
                                               float scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = __float2bfloat16(silu_f32(__bfloat162float(x[idx]) * scale));
}

// out = bf16(gain * f32(bf16(sigmoid(f32(x) * scale)))). The inner cast is the
// reference's own: `torch.sigmoid` on a bf16 tensor materialises bf16 before
// the `2 *` reads it.
__global__ void qwen38fn_hc_sigmoid_affine_kernel(const __nv_bfloat16 *__restrict__ x,
                                                  __nv_bfloat16 *__restrict__ out, int n,
                                                  float scale, float gain) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const float gated = __bfloat162float(
      __float2bfloat16(sigmoid_f32(__bfloat162float(x[idx]) * scale)));
  out[idx] = __float2bfloat16(gated * gain);
}

// out[r, c] = bf16(combine_scale * sum_j f32(bf16(a[r, j*H+c] * b[r, j*H+c]))).
// `a` and `b` are row-major [rows, branches*H]; `out` is [rows, H]. The inner
// cast is the reference's bf16 product; the f32 accumulate over branches and
// the single cast out are torch's `mean(dim=-2)`.
__global__ void qwen38fn_hc_mul_branch_combine_kernel(const __nv_bfloat16 *__restrict__ a,
                                                      const __nv_bfloat16 *__restrict__ b,
                                                      __nv_bfloat16 *__restrict__ out, int n_out,
                                                      int hidden, int branches,
                                                      float combine_scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_out) return;
  const int row = idx / hidden;
  const int base = row * branches * hidden + (idx - row * hidden);
  float acc = 0.0f;
  for (int j = 0; j < branches; ++j) {
    const int src = base + j * hidden;
    const float product = __bfloat162float(a[src]) * __bfloat162float(b[src]);
    acc += __bfloat162float(__float2bfloat16(product));
  }
  out[idx] = __float2bfloat16(acc * combine_scale);
}

// Grid for `n` elements. The count is widened to 64 bits because the rounding
// term can push an `n` that already sits at the i32 ceiling over it: all three
// launchers bound their counts with `dims_fit_i32`, whose top end is
// 0x7ffffffe, and 0x7ffffffe + 255 does not fit an int. Every launcher goes
// through here, so the guard is the same for the elementwise pair and for the
// combine.
unsigned grid_for(long long n) {
  return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

// Bounds a *product* of dimensions, not each factor: no single dimension is
// separately checked, and none needs to be. What has to survive is the count a
// launch passes to `grid_for` and the i32 source index the combine kernel
// forms, `row * branches * hidden + j * hidden + c`, and both are bounded by
// the product `rows * branches * hidden` itself.
bool dims_fit_i32(long long product) { return product > 0 && product < 0x7fffffffLL; }

}  // namespace

extern "C" {

CUresult qwen38fn_hc_silu_scaled_cuda(const __nv_bfloat16 *x, __nv_bfloat16 *out, int n,
                                      float scale, cudaStream_t stream) {
  if (x == nullptr || out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n == 0) {
    return CUDA_SUCCESS;
  }
  // The same ceiling the combine launcher puts on its products.
  if (!dims_fit_i32(n)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  qwen38fn_hc_silu_scaled_kernel<<<grid_for(n), kThreads, 0, stream>>>(x, out, n, scale);
  return consume_last_cuda_error();
}

CUresult qwen38fn_hc_sigmoid_affine_cuda(const __nv_bfloat16 *x, __nv_bfloat16 *out, int n,
                                         float scale, float gain, cudaStream_t stream) {
  if (x == nullptr || out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n == 0) {
    return CUDA_SUCCESS;
  }
  // The same ceiling the combine launcher puts on its products.
  if (!dims_fit_i32(n)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  qwen38fn_hc_sigmoid_affine_kernel<<<grid_for(n), kThreads, 0, stream>>>(x, out, n, scale, gain);
  return consume_last_cuda_error();
}

CUresult qwen38fn_hc_mul_branch_combine_cuda(const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                                             __nv_bfloat16 *out, int rows, int hidden,
                                             int branches, float combine_scale,
                                             cudaStream_t stream) {
  if (a == nullptr || b == nullptr || out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rows < 0 || hidden <= 0 || branches <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rows == 0) {
    return CUDA_SUCCESS;
  }
  const long long stream_width = static_cast<long long>(branches) * hidden;
  const long long source_elems = static_cast<long long>(rows) * stream_width;
  const long long out_elems = static_cast<long long>(rows) * hidden;
  if (!dims_fit_i32(source_elems) || !dims_fit_i32(out_elems)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  qwen38fn_hc_mul_branch_combine_kernel<<<grid_for(out_elems), kThreads, 0, stream>>>(
      a, b, out, static_cast<int>(out_elems), hidden, branches, combine_scale);
  return consume_last_cuda_error();
}

}  // extern "C"
