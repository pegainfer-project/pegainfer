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
// Two bodies per op. The scalar one indexes a flat element count and serves any
// width; the `vec4` one moves eight bf16 (16 bytes) per thread and serves this
// line's geometry, where the HC stream is 10240 wide in 2560-wide branches and
// every buffer starts on a 256-byte allocation. `vec4_ok` picks between them,
// the same rule `shared/elementwise.cu` uses. Both land identical bits — the
// arithmetic is per element and the branch order is unchanged — and the vec4
// bodies also hoist the row/branch decomposition out of the per-element loop,
// which was the whole cost of the first scalar scatter (a 64-bit divide per
// element).
//
// Target: the sm_89-first portable subset — plain SIMT elementwise bodies, no
// TMA, no cluster, no wgmma.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

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

__device__ __forceinline__ float silu_scaled_f32(float x, float scale) {
  return silu_f32(__bfloat162float(x) * scale);
}

__device__ __forceinline__ float sigmoid_affine_f32(float x, float scale, float gain) {
  const float gated =
      __bfloat162float(__float2bfloat16(sigmoid_f32(__bfloat162float(x) * scale)));
  return gated * gain;
}

// out = bf16(silu(f32(x) * scale)). The reference divides by `hc_count` in bf16
// first; for the power-of-two counts that is exact, so folding it into `scale`
// leaves one rounding — the SiLU's — in the same place.
__global__ void qwen4_hc_silu_scaled_kernel(const __nv_bfloat16 *__restrict__ x,
                                            __nv_bfloat16 *__restrict__ out, int n,
                                            float scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = __float2bfloat16(silu_scaled_f32(x[idx], scale));
}

// out = bf16(gain * f32(bf16(sigmoid(f32(x) * scale)))). The inner cast is the
// reference's own: `torch.sigmoid` on a bf16 tensor materialises bf16 before
// the `2 *` reads it.
__global__ void qwen4_hc_sigmoid_affine_kernel(const __nv_bfloat16 *__restrict__ x,
                                               __nv_bfloat16 *__restrict__ out, int n,
                                               float scale, float gain) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = __float2bfloat16(sigmoid_affine_f32(x[idx], scale, gain));
}

// Eight bf16 per thread; the launcher only picks these when `n % 8 == 0`, so
// there is no tail.
__global__ void qwen4_hc_silu_scaled_vec4_kernel(const __nv_bfloat16 *__restrict__ x,
                                                 __nv_bfloat16 *__restrict__ out, int n,
                                                 float scale) {
  const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (base >= n) return;
  const uint4 v = *reinterpret_cast<const uint4 *>(x + base);
  const unsigned short *s = reinterpret_cast<const unsigned short *>(&v);
  uint4 w;
  unsigned short *d = reinterpret_cast<unsigned short *>(&w);
  for (int k = 0; k < 8; ++k) {
    d[k] = __bfloat16_as_ushort(
        __float2bfloat16(silu_scaled_f32(__ushort_as_bfloat16(s[k]), scale)));
  }
  *reinterpret_cast<uint4 *>(out + base) = w;
}

__global__ void qwen4_hc_sigmoid_affine_vec4_kernel(const __nv_bfloat16 *__restrict__ x,
                                                    __nv_bfloat16 *__restrict__ out, int n,
                                                    float scale, float gain) {
  const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (base >= n) return;
  const uint4 v = *reinterpret_cast<const uint4 *>(x + base);
  const unsigned short *s = reinterpret_cast<const unsigned short *>(&v);
  uint4 w;
  unsigned short *d = reinterpret_cast<unsigned short *>(&w);
  for (int k = 0; k < 8; ++k) {
    d[k] = __bfloat16_as_ushort(
        __float2bfloat16(sigmoid_affine_f32(__ushort_as_bfloat16(s[k]), scale, gain)));
  }
  *reinterpret_cast<uint4 *>(out + base) = w;
}

// out[r, c] = bf16(combine_scale * sum_j f32(bf16(a[r, j*H+c] * b[r, j*H+c]))).
// `a` and `b` are row-major [rows, branches*H]; `out` is [rows, H]. The inner
// cast is the reference's bf16 product; the f32 accumulate over branches and
// the single cast out are torch's `mean(dim=-2)`.
__global__ void qwen4_hc_mul_branch_combine_kernel(const __nv_bfloat16 *__restrict__ a,
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

// Scatter a sublayer output back into the four-branch stream:
//
//   out[s, b*H + h] = bf16(hyper[s, b*H + h] + bf16(f32(x[s, h]) * f32(inj[s, b])))
//
// which is the reference decoder layer's `hyper_input + (out * inj).flatten()`:
// one bf16 multiply, then one bf16 add, each rounding where torch rounds. `out`
// may alias `hyper`: every element reads its own `hyper` value before writing.
__global__ void qwen4_hc_scatter_kernel(const __nv_bfloat16 *__restrict__ x,
                                        const __nv_bfloat16 *__restrict__ inject,
                                        const __nv_bfloat16 *__restrict__ hyper,
                                        __nv_bfloat16 *__restrict__ out, int rows,
                                        int hidden, int branches) {
  const long long n = static_cast<long long>(rows) * branches * hidden;
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const int h = static_cast<int>(idx % hidden);
  const long long row_branch = idx / hidden;
  const int b = static_cast<int>(row_branch % branches);
  const long long row = row_branch / branches;
  const float prod = __bfloat162float(x[row * hidden + h]) *
                     __bfloat162float(inject[row * branches + b]);
  const float base = __bfloat162float(hyper[idx]);
  const float rounded = __bfloat162float(__float2bfloat16_rn(prod));
  out[idx] = __float2bfloat16_rn(base + rounded);
}

// The same scatter, eight elements per thread. Blocks are laid out as
// `rows * branches * per_row_blocks`: every block owns one 8*kThreads-wide
// slice of one row-branch, so the row and branch decomposition is a division
// on `blockIdx.x` (uniform across the block, once) instead of two 64-bit
// divides per element — which ncu put at 72% SM throughput and 29% memory on
// the flat form.
__global__ void qwen4_hc_scatter_vec4_kernel(const __nv_bfloat16 *__restrict__ x,
                                             const __nv_bfloat16 *__restrict__ inject,
                                             const __nv_bfloat16 *__restrict__ hyper,
                                             __nv_bfloat16 *__restrict__ out, int hidden,
                                             int branches, int per_row_blocks) {
  const int row_branch = blockIdx.x / per_row_blocks;
  const int chunk = blockIdx.x - row_branch * per_row_blocks;
  const int b = row_branch % branches;
  const int row = row_branch / branches;
  const int h0 = chunk * blockDim.x * 8 + threadIdx.x * 8;
  if (h0 >= hidden) {
    return;
  }
  const float inject_b = __bfloat162float(inject[row * branches + b]);
  const int idx = row_branch * hidden + h0;
  const int x_at = row * hidden + h0;
  const uint4 xv = *reinterpret_cast<const uint4 *>(x + x_at);
  const uint4 hv = *reinterpret_cast<const uint4 *>(hyper + idx);
  const unsigned short *xs = reinterpret_cast<const unsigned short *>(&xv);
  const unsigned short *hs = reinterpret_cast<const unsigned short *>(&hv);
  uint4 w;
  unsigned short *d = reinterpret_cast<unsigned short *>(&w);
  for (int k = 0; k < 8; ++k) {
    const float prod = __bfloat162float(__ushort_as_bfloat16(xs[k])) * inject_b;
    const float rounded = __bfloat162float(__float2bfloat16_rn(prod));
    const float base = __bfloat162float(__ushort_as_bfloat16(hs[k]));
    d[k] = __bfloat16_as_ushort(__float2bfloat16_rn(base + rounded));
  }
  *reinterpret_cast<uint4 *>(out + idx) = w;
}

// The branch combine with the same block layout (one row, one 8*kThreads-wide
// slice) and eight outputs per thread: the four branch reads become 16-byte
// loads instead of 2-byte ones, and the branch sum keeps its order.
__global__ void qwen4_hc_mul_branch_combine_vec4_kernel(const __nv_bfloat16 *__restrict__ a,
                                                        const __nv_bfloat16 *__restrict__ b,
                                                        __nv_bfloat16 *__restrict__ out,
                                                        int hidden, int branches,
                                                        int per_row_blocks,
                                                        float combine_scale) {
  const int row = blockIdx.x / per_row_blocks;
  const int chunk = blockIdx.x - row * per_row_blocks;
  const int c0 = chunk * blockDim.x * 8 + threadIdx.x * 8;
  if (c0 >= hidden) {
    return;
  }
  const int src = row * branches * hidden + c0;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int j = 0; j < branches; ++j) {
    const uint4 av = *reinterpret_cast<const uint4 *>(a + src + j * hidden);
    const uint4 bv = *reinterpret_cast<const uint4 *>(b + src + j * hidden);
    const unsigned short *as = reinterpret_cast<const unsigned short *>(&av);
    const unsigned short *bs = reinterpret_cast<const unsigned short *>(&bv);
    for (int k = 0; k < 8; ++k) {
      const float product = __bfloat162float(__ushort_as_bfloat16(as[k])) *
                            __bfloat162float(__ushort_as_bfloat16(bs[k]));
      acc[k] += __bfloat162float(__float2bfloat16(product));
    }
  }
  uint4 w;
  unsigned short *d = reinterpret_cast<unsigned short *>(&w);
  for (int k = 0; k < 8; ++k) {
    d[k] = __bfloat16_as_ushort(__float2bfloat16(acc[k] * combine_scale));
  }
  *reinterpret_cast<uint4 *>(out + row * hidden + c0) = w;
}

// Grid for `n` scalar elements. The count is widened to 64 bits because the
// rounding term can push an `n` that already sits at the i32 ceiling over it:
// the launchers bound their counts with `dims_fit_i32`, whose top end is
// 0x7ffffffe, and 0x7ffffffe + 255 does not fit an int.
unsigned grid_for(long long n) {
  return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

// Grid for `n` elements at eight per thread.
unsigned vec4_grid_for(long long n) { return grid_for((n + 7) / 8); }

// Blocks that cover `width` elements at eight per thread, one row at a time.
int per_row_blocks(int width) { return (width + kThreads * 8 - 1) / (kThreads * 8); }

// Whether the vec4 bodies can serve this call: `width` a whole number of
// 8-wide vectors, so no row starts mid-vector, and every base pointer 16-byte
// aligned, which every kernel allocation is. Anything else takes the scalar
// body.
bool vec4_ok(int width, const void *p0, const void *p1, const void *p2 = nullptr,
             const void *p3 = nullptr) {
  if (width % 8 != 0) return false;
  uintptr_t bits = reinterpret_cast<uintptr_t>(p0) | reinterpret_cast<uintptr_t>(p1);
  if (p2 != nullptr) bits |= reinterpret_cast<uintptr_t>(p2);
  if (p3 != nullptr) bits |= reinterpret_cast<uintptr_t>(p3);
  return (bits & 15) == 0;
}

// Bounds a *product* of dimensions, not each factor: no single dimension is
// separately checked, and none needs to be. What has to survive is the count a
// launch passes to `grid_for` and the i32 source index the combine kernel
// forms, `row * branches * hidden + j * hidden + c`, and both are bounded by
// the product `rows * branches * hidden` itself.
bool dims_fit_i32(long long product) { return product > 0 && product < 0x7fffffffLL; }

}  // namespace

extern "C" {

CUresult qwen4_hc_silu_scaled_cuda(const __nv_bfloat16 *x, __nv_bfloat16 *out, int n,
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
  if (vec4_ok(n, x, out)) {
    qwen4_hc_silu_scaled_vec4_kernel<<<vec4_grid_for(n), kThreads, 0, stream>>>(x, out, n,
                                                                                scale);
  } else {
    qwen4_hc_silu_scaled_kernel<<<grid_for(n), kThreads, 0, stream>>>(x, out, n, scale);
  }
  return consume_last_cuda_error();
}

CUresult qwen4_hc_sigmoid_affine_cuda(const __nv_bfloat16 *x, __nv_bfloat16 *out, int n,
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
  if (vec4_ok(n, x, out)) {
    qwen4_hc_sigmoid_affine_vec4_kernel<<<vec4_grid_for(n), kThreads, 0, stream>>>(
        x, out, n, scale, gain);
  } else {
    qwen4_hc_sigmoid_affine_kernel<<<grid_for(n), kThreads, 0, stream>>>(x, out, n, scale, gain);
  }
  return consume_last_cuda_error();
}

CUresult qwen4_hc_mul_branch_combine_cuda(const __nv_bfloat16 *a, const __nv_bfloat16 *b,
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
  if (vec4_ok(hidden, a, b, out)) {
    qwen4_hc_mul_branch_combine_vec4_kernel<<<rows * per_row_blocks(hidden), kThreads, 0,
                                              stream>>>(a, b, out, hidden, branches,
                                                        per_row_blocks(hidden),
                                                        combine_scale);
  } else {
    qwen4_hc_mul_branch_combine_kernel<<<grid_for(out_elems), kThreads, 0, stream>>>(
        a, b, out, static_cast<int>(out_elems), hidden, branches, combine_scale);
  }
  return consume_last_cuda_error();
}

CUresult qwen4_hc_scatter_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *inject,
                               const __nv_bfloat16 *hyper, __nv_bfloat16 *out, int rows,
                               int hidden, int branches, cudaStream_t stream) {
  if (x == nullptr || inject == nullptr || hyper == nullptr || out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rows < 0 || hidden <= 0 || branches <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rows == 0) {
    return CUDA_SUCCESS;
  }
  const long long out_elems = static_cast<long long>(rows) * branches * hidden;
  if (!dims_fit_i32(out_elems)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (vec4_ok(hidden, x, hyper, out)) {
    qwen4_hc_scatter_vec4_kernel<<<static_cast<long long>(rows) * branches * per_row_blocks(hidden),
                                   kThreads, 0, stream>>>(x, inject, hyper, out, hidden, branches,
                                                          per_row_blocks(hidden));
  } else {
    qwen4_hc_scatter_kernel<<<grid_for(out_elems), kThreads, 0, stream>>>(
        x, inject, hyper, out, rows, hidden, branches);
  }
  return consume_last_cuda_error();
}

}  // extern "C"
