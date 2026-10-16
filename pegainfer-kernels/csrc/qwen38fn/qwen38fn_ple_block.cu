// Qwen3.8-Flash-Next PLE block (per-layer embedding), the device half.
//
// Frozen semantics: transformers 5.16.1 `qwen4_exp`,
// `Qwen4ExpTextPLELayer.forward`. From the two recorded projections and the
// residual stream, over `hc_count` hyper-connection branches of `hidden_size`
// channels each (`width = hc_count * hidden_size`):
//
//   k = (1 + w_key)   RMSNorm_2560(key)      // grouped: one norm per branch
//   q = (1 + w_query) RMSNorm_2560(hidden)   // grouped, same split
//   dot_rg   = sum_j bf16(k[r,g,j] * q[r,g,j])          // bf16 per product
//   pre_rg   = bf16(bf16(dot_rg) / sqrt(hidden_size))   // TWO roundings
//   gate_rg  = bf16(bf16(max(|pre_rg|, 1e-6)).sqrt() * sign(pre_rg))
//   gated    = bf16(sigmoid(gate_rg) * value[r])        // value broadcast over g
//   c        = (1 + w_conv) RMSNorm_2560(gated)
//   out[r,ch]= bf16(gated[r,ch] + silu(bf16(dilated_causal_conv1d(c)[r,ch])))
//
// The conv is depthwise over `width` channels, kernel 4, dilation `ngram_size`
// (3), so its receptive field is `(kernel - 1) * dilation + 1 = 10` rows and its
// zero initial state is the 9 rows before the sequence.
//
// Three things here are easy to get wrong and are the reason the rounding points
// are spelled out above:
//
//   * The gate rounds TWICE around the `1 / sqrt(hidden_size)` divide: the group
//     sum to bf16 first, the quotient to bf16 second. Collapsing them is worth
//     ~1 ulp on every gate. It is a divide by `sqrt_hidden`, not a multiply by a
//     reciprocal — the two round differently, which is why `sqrt_hidden` is an
//     argument rather than a compiled constant: the caller computes it the way
//     the reference does.
//   * The sqrt is SIGNED. `abs().clamp_min(1e-6).sqrt()` loses the sign, and the
//     reference restores it with `torch.sign` — which is 0 at 0, not 1, so a zero
//     gate sigmoid's to exactly 0.5 rather than to `sigmoid(1e-3)`.
//   * The dot is per BRANCH (2560), not per stream (10240). One gate per
//     branch, and all four branches then scale the SAME `value` row.
//
// Both grouped norms are the existing batched `(1 + w)` RMSNorm entry point,
// `rms_norm_batched_offset_cuda`, called once per branch from the Rust ops layer:
// it normalizes `seq_len` contiguous rows of `hidden_dim`, and this stream's
// branches are interleaved row-major, so the ops layer splits into branch-major
// scratch first (kernel 1) and the norms then run on contiguous rows. That split
// is what the first kernel below is for.
//
// Target: the sm_89-first portable subset — no TMA, no cluster, no wgmma. All
// three kernels are SIMT: two are element-wise copies/maps and the third is a
// 2560-wide dot with a shuffle-and-shared-memory block reduction, a shape an
// `m16n8k16` body cannot tile without padding one row to sixteen.
//
// Negative-control surface: `gate_span` and `signed_sqrt` are arguments rather
// than compiled constants so the parity gate can turn exactly one knob on the
// device path — a full-stream dot, or an unsigned sqrt — and assert the output
// leaves its bound. `dilation` is genuine geometry (`ngram_size`), and the gate
// drives it to 1 for the same reason. Production callers pass the reference
// values; `ops::ple_block_into` is the only caller and hard-codes them.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <climits>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHidden = 2560;  // hidden_size == ple_embed_dim
constexpr int kHc = 4;         // hc_count
constexpr int kWidth = kHidden * kHc;
constexpr int kConvKernel = 4;  // ple_conv_kernel_size
// The reference's `clamp_min(1e-6)` on the gate's magnitude.
constexpr float kGateClampMin = 1.0e-6f;
// One CTA per (row, branch) for the gate; 256 threads over 2560 channels is 10
// elements per thread, and 8 warps keeps the cross-warp reduction one round.
constexpr int kGateThreads = 256;
constexpr int kGateWarps = kGateThreads / WARP_SIZE;
// Element-wise passes: 256 lanes over a width that is a multiple of 32, so a
// warp always covers consecutive channels.
constexpr int kMapThreads = 256;

static_assert(kWidth % WARP_SIZE == 0, "a warp must cover whole channels");
static_assert(kHidden % kGateThreads == 0, "the gate CTA must tile the group");

// Round to bf16 and back, which is what the reference does every time a
// submodule boundary is crossed. Written out rather than left to the compiler
// because the rounding IS the semantics.
__device__ __forceinline__ float bf16_round(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

// `torch.sign`: 0 at zero (and at -0.0), unlike a `copysign`-style restore.
__device__ __forceinline__ float torch_sign(float value) {
  return static_cast<float>(value > 0.0f) - static_cast<float>(value < 0.0f);
}

// Sum over the whole CTA, left in every thread: the gate scalar is uniform
// across the CTA and each thread then writes its own channels, so broadcasting
// it through shared memory costs one barrier and saves a second reduce.
__device__ __forceinline__ float block_all_sum(float value, float* shared) {
  value = warp_reduce_sum(value);
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < kGateWarps; ++w) {
    total += shared[w];
  }
  return total;
}

// Row-major `[seq_len, kWidth]` -> branch-major `[kHc][seq_len][kHidden]`.
// The grouped norms run on contiguous `kHidden` rows, and the block's branches
// are interleaved in the stream, so this is a transpose of the two middle axes.
__global__ void qwen38fn_ple_split_branches_kernel(
    const __nv_bfloat16* __restrict__ row_major,
    __nv_bfloat16* __restrict__ branch_major, int seq_len) {
  const int64_t total = static_cast<int64_t>(seq_len) * kWidth;
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * kMapThreads + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int64_t row = index / kWidth;
  const int64_t channel = index - row * kWidth;
  const int64_t branch = channel / kHidden;
  const int64_t within = channel - branch * kHidden;
  branch_major[(branch * seq_len + row) * kHidden + within] = row_major[index];
}

// One gate scalar per (row, branch), then `sigmoid(gate) * value` over that
// branch's channels. Inputs are branch-major `[kHc][seq_len][kHidden]`; `value`
// is `[seq_len, kHidden]` and is broadcast over the branches that share it.
//
// `gate_span` is `kHidden` for the reference's per-branch dot and `kWidth` for
// the full-stream control, which folds all four branches into one dot — every
// branch then gets the same gate, which is the fault the control stands for.
__global__ void qwen38fn_ple_gate_kernel(
    const __nv_bfloat16* __restrict__ key_normed,
    const __nv_bfloat16* __restrict__ query_normed,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ gated, int seq_len, float sqrt_hidden,
    int gate_span, int signed_sqrt) {
  __shared__ float shared[kGateWarps];

  const int block = blockIdx.x;
  const int branch = block % kHc;
  const int row = block / kHc;
  const int branches = (gate_span == kWidth) ? kHc : 1;
  const int first = (gate_span == kWidth) ? 0 : branch;

  // Each product rounds to bf16 before it is summed, as the reference's does:
  // the multiply happens inside the submodule that produces it.
  float partial = 0.0f;
  for (int j = threadIdx.x; j < kHidden; j += kGateThreads) {
    for (int b = 0; b < branches; ++b) {
      const int64_t at =
          (static_cast<int64_t>(first + b) * seq_len + row) * kHidden + j;
      const float k = __bfloat162float(key_normed[at]);
      const float q = __bfloat162float(query_normed[at]);
      partial += bf16_round(k * q);
    }
  }
  const float dot = block_all_sum(partial, shared);

  // Two bf16 roundings around the divide, then the signed sqrt, then the
  // sigmoid — every one of them a rounding point of the reference.
  const float pre_gate = bf16_round(bf16_round(dot) / sqrt_hidden);
  const float magnitude = bf16_round(fmaxf(fabsf(pre_gate), kGateClampMin));
  const float root = bf16_round(sqrtf(magnitude));
  const float restore = signed_sqrt != 0 ? torch_sign(pre_gate) : 1.0f;
  const float gate = bf16_round(root * restore);
  const float weight = bf16_round(1.0f / (1.0f + expf(-gate)));

  const __nv_bfloat16* value_row =
      value + static_cast<int64_t>(row) * kHidden;
  __nv_bfloat16* gated_row =
      gated + (static_cast<int64_t>(branch) * seq_len + row) * kHidden;
  for (int j = threadIdx.x; j < kHidden; j += kGateThreads) {
    gated_row[j] = __float2bfloat16(weight * __bfloat162float(value_row[j]));
  }
}

// `out = gated + silu(bf16(dilated causal depthwise conv(conv_in)))`.
//
// `conv_in` and `gated` are branch-major `[kHc][seq_len][kHidden]`; `out` is
// row-major `[seq_len, kWidth]`, which is the stream the block injects into.
// `conv_weight` is `[kWidth, 1, kConvKernel]` flattened. The zero initial state
// of `(kConvKernel - 1) * dilation` rows is the `source >= 0` bound, not a
// buffer: a tap that reaches before the sequence contributes nothing.
//
// The four taps accumulate in tap order in f32, which is the reference's own
// order and short enough that a tree would only change which way the last bit
// falls.
__global__ void qwen38fn_ple_conv_residual_kernel(
    const __nv_bfloat16* __restrict__ conv_in,
    const __nv_bfloat16* __restrict__ gated,
    const __nv_bfloat16* __restrict__ conv_weight,
    __nv_bfloat16* __restrict__ out, int seq_len, int dilation) {
  const int64_t total = static_cast<int64_t>(seq_len) * kWidth;
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * kMapThreads + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int64_t row = index / kWidth;
  const int64_t channel = index - row * kWidth;
  const int64_t branch = channel / kHidden;
  const int64_t within = channel - branch * kHidden;

  float taps[kConvKernel];
#pragma unroll
  for (int t = 0; t < kConvKernel; ++t) {
    taps[t] = __bfloat162float(conv_weight[channel * kConvKernel + t]);
  }

  // Branch-major base for this channel's column: `(branch * seq_len) * kHidden
  // + within`, then walked by `source * kHidden`. Branch 0 is the only one where
  // dropping the group factor would still land on the right element, which is
  // exactly why the parity gate compares every branch rather than the first.
  const int64_t column = branch * seq_len * kHidden + within;
  const __nv_bfloat16* conv_column = conv_in + column;
  const __nv_bfloat16* gated_column = gated + column;

  float acc = 0.0f;
#pragma unroll
  for (int t = 0; t < kConvKernel; ++t) {
    const int64_t source =
        row - static_cast<int64_t>(kConvKernel - 1 - t) * dilation;
    if (source >= 0) {
      acc += __bfloat162float(conv_column[source * kHidden]) * taps[t];
    }
  }
  const float convolved = bf16_round(acc);
  const float activated =
      bf16_round(convolved / (1.0f + expf(-convolved)));
  out[index] = __float2bfloat16(
      __bfloat162float(gated_column[row * kHidden]) + activated);
}

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

// Grid for the element-wise passes, one thread per stream element, or -1 if it
// does not fit an `int`. Formed in 64 bits: `seq_len * kWidth` overflows an
// `int` at ~209k rows, long before the grid dimension's own 2^31 - 1 limit, and
// a wrapped grid would launch nothing at all rather than too little.
int map_grid(int seq_len) {
  const int64_t total = static_cast<int64_t>(seq_len) * kWidth;
  const int64_t grid = (total + kMapThreads - 1) / kMapThreads;
  return grid > static_cast<int64_t>(INT_MAX) ? -1 : static_cast<int>(grid);
}

}  // namespace

extern "C" {

// Row-major `[seq_len, width]` -> branch-major `[hc_count][seq_len][hidden]`,
// for each of the two streams the grouped norms read.
//
// `width` is the caller's row stride, checked here: the body splits with the
// width and the group size it was compiled for, so a caller whose rows are
// strided differently would be read out of bounds rather than computed wrongly.
CUresult qwen38fn_ple_split_branches_cuda(const __nv_bfloat16* row_major,
                                          __nv_bfloat16* branch_major,
                                          int seq_len, int width,
                                          int hidden, int hc_count,
                                          cudaStream_t stream) {
  if (row_major == nullptr || branch_major == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (seq_len < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (width != kWidth || hidden != kHidden || hc_count != kHc) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (seq_len == 0) {
    return CUDA_SUCCESS;
  }
  const int grid = map_grid(seq_len);
  if (grid < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  qwen38fn_ple_split_branches_kernel<<<grid, kMapThreads, 0, stream>>>(
      row_major, branch_major, seq_len);
  return consume_last_cuda_error();
}

// One gate scalar per (row, branch) and `sigmoid(gate) * value` over that
// branch's channels. All four buffers are branch-major `[hc_count][seq_len]
// [hidden]`, except `value`, which is `[seq_len, hidden]` and shared by the
// branches.
//
// `sqrt_hidden` is `sqrt(hidden_size)` in the f32 the reference divides by; it
// is an argument because a divide and a reciprocal multiply round differently.
// `gate_span` is `hidden` for the reference and `width` for the full-stream
// control; `signed_sqrt` is 1 for the reference and 0 for the unsigned control.
CUresult qwen38fn_ple_gate_cuda(const __nv_bfloat16* key_normed,
                                const __nv_bfloat16* query_normed,
                                const __nv_bfloat16* value,
                                __nv_bfloat16* gated, int seq_len, int hidden,
                                int hc_count, int gate_span, float sqrt_hidden,
                                int signed_sqrt, cudaStream_t stream) {
  if (key_normed == nullptr || query_normed == nullptr || value == nullptr ||
      gated == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (seq_len < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (hidden != kHidden || hc_count != kHc) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (gate_span != kHidden && gate_span != kWidth) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (signed_sqrt != 0 && signed_sqrt != 1) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // A non-finite or non-positive divisor would put NaN in every gate; refuse it
  // rather than poison the stream silently.
  if (!(sqrt_hidden > 0.0f)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (seq_len == 0) {
    return CUDA_SUCCESS;
  }
  // One CTA per (row, branch): the grid is `seq_len * kHc`, which has to be
  // formed in 64 bits before it is narrowed, or a long enough sequence would
  // wrap into a negative grid and launch nothing at all.
  const int64_t gate_grid = static_cast<int64_t>(seq_len) * kHc;
  if (gate_grid > static_cast<int64_t>(INT_MAX)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int grid = static_cast<int>(gate_grid);
  qwen38fn_ple_gate_kernel<<<grid, kGateThreads, 0, stream>>>(
      key_normed, query_normed, value, gated, seq_len, sqrt_hidden, gate_span,
      signed_sqrt);
  return consume_last_cuda_error();
}

// The dilated causal depthwise conv, its silu, and the residual add back onto
// the gated value. `conv_in` and `gated` are branch-major, `out` is row-major
// `[seq_len, hc_count * hidden]`.
//
// `dilation` is `ngram_size`; the state it implies is `(conv_kernel - 1) *
// dilation` rows and is zero by the `source >= 0` bound rather than by a
// buffer, so any positive dilation is served without resizing anything.
CUresult qwen38fn_ple_conv_residual_cuda(const __nv_bfloat16* conv_in,
                                         const __nv_bfloat16* gated,
                                         const __nv_bfloat16* conv_weight,
                                         __nv_bfloat16* out, int seq_len,
                                         int hidden, int hc_count,
                                         int conv_kernel, int dilation,
                                         cudaStream_t stream) {
  if (conv_in == nullptr || gated == nullptr || conv_weight == nullptr ||
      out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (seq_len < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (hidden != kHidden || hc_count != kHc || conv_kernel != kConvKernel) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (dilation < 1) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (seq_len == 0) {
    return CUDA_SUCCESS;
  }
  const int grid = map_grid(seq_len);
  if (grid < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  qwen38fn_ple_conv_residual_kernel<<<grid, kMapThreads, 0, stream>>>(
      conv_in, gated, conv_weight, out, seq_len, dilation);
  return consume_last_cuda_error();
}

}  // extern "C"
