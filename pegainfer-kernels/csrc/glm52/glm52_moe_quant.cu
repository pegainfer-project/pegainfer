#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_fp8.h>
#include <math_constants.h>

namespace {

constexpr int kGroupSize = 128;
constexpr float kFp8Min = -448.0f;
constexpr float kFp8Max = 448.0f;
constexpr float kPerTokenGroupQuantEps = 1.0e-10f;

__device__ __forceinline__ unsigned char quantize_e4m3(float value,
                                                       float scale) {
  float q = fminf(fmaxf(value / scale, kFp8Min), kFp8Max);
  return __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
}

// Grid-strided over rows: the row grid is capped (kMaxRowBlocks) and each
// block loops rows with stride gridDim.x up to the effective end. At the MoE
// recv capacity (2080 rows x 48 groups) a one-block-per-(row,group) grid is
// ~100k tiny blocks whose SCHEDULING alone costs ~60 us — far more than the
// actual quant work — and a device-bound early-return does not help because
// retired blocks are still scheduled. The device-side `row_bound` (the
// grouped-GEMM aligned segment end) instead bounds the loop, so the work AND
// the scheduling scale with the real row count while the launch shape stays
// fixed (CUDA-graph stable). Per-row math is unchanged (bit-identical).
//
// kMasked redirects the writes for the DeepGEMM masked grouped layout: the
// loop space stays the aligned recv rows, row_map[row] gives the masked slot
// (g*masked_cap + r_local; -1 on alignment-gap rows, which are skipped), the
// value goes to the fixed-stride masked row and the scale to the mn-major
// TMA layout [g, scale_cols, masked_cap] the GEMM's SFA descriptor reads —
// no separate scale-relayout kernel.
//
// kUe8m0 rounds the group scale UP to the next power of two (exact bit
// manipulation, no log2f rounding hazard). This is the FlashMLA V3.2 fp8
// sparse KV-cache contract: the sm100 decode kernel converts the stored f32
// scales to e8m0 with round-toward-zero for the tcgen05 block-scaled MMA
// (upstream tests/quant.py `_cast_scale_inv_to_ue8m0`), so a non-power-of-two
// scale is silently truncated — up to 2x too small — on Blackwell, while
// sm90 reads the f32 scale exactly. Power-of-two scales make both archs read
// the identical value.
template <bool kMasked, bool kUe8m0 = false>
__global__ void fp8_per_token_group_quant_bf16_k128_kernel(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ output, float* __restrict__ scales, int rows,
    int hidden_dim, const long long* __restrict__ row_bound,
    const int* __restrict__ row_map, int masked_cap) {
  const int group = blockIdx.y;
  const int tid = threadIdx.x;
  const int group_start = group * kGroupSize;
  const int col = group_start + tid;
  const int scale_cols = hidden_dim / kGroupSize;
  int end = rows;
  if (row_bound != nullptr) {
    const long long b = *row_bound;
    if (b < end) end = static_cast<int>(b < 0 ? 0 : b);
  }

  __shared__ float shared[kGroupSize];
  for (int row = blockIdx.x; row < end; row += gridDim.x) {
    int out_row = row;
    if constexpr (kMasked) {
      out_row = row_map[row];
      if (out_row < 0) continue;
    }
    float value = 0.0f;
    if (col < hidden_dim) {
      value = __bfloat162float(input[(size_t)row * hidden_dim + col]);
    }
    shared[tid] = fabsf(value);
    __syncthreads();

#pragma unroll
    for (int stride = kGroupSize / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      float s = fmaxf(shared[0], kPerTokenGroupQuantEps) / kFp8Max;
      if constexpr (kUe8m0) {
        // Next power of two >= s: bump the mantissa into the exponent field.
        // s is always positive, normal, and far from f32 max here.
        s = __uint_as_float((__float_as_uint(s) + 0x007FFFFFu) & 0x7F800000u);
      }
      shared[0] = s;
      if constexpr (kMasked) {
        const int g = out_row / masked_cap;
        const int r_local = out_row % masked_cap;
        scales[((size_t)g * scale_cols + group) * masked_cap + r_local] =
            shared[0];
      } else {
        scales[(size_t)row * scale_cols + group] = shared[0];
      }
    }
    __syncthreads();

    if (col < hidden_dim) {
      output[(size_t)out_row * hidden_dim + col] = quantize_e4m3(value, shared[0]);
    }
    __syncthreads();
  }
}

// Grid-strided over aligned receive rows. The gate|up input rows are already
// in the masked layout written by W13. Router weights are deliberately not
// applied here: vLLM quantizes the unweighted SwiGLU activation and applies
// routing weights after W2.
// kUe8m0 rounds the group scale UP to the next power of two (same bit
// manipulation as the plain quant kernel) — the Blackwell packed-SF contract.
template <bool kUe8m0 = false>
__global__ void silu_and_mul_per_token_group_quant_bf16_k128_masked_kernel(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ output, float* __restrict__ scales, int rows,
    int hidden_size,
    const long long* __restrict__ row_bound,
    const int* __restrict__ row_map, int masked_cap) {
  const int group = blockIdx.y;
  const int tid = threadIdx.x;
  const int group_start = group * kGroupSize;
  const int col = group_start + tid;
  const int input_stride = hidden_size * 2;
  const int scale_cols = hidden_size / kGroupSize;
  int end = rows;
  if (row_bound != nullptr) {
    const long long b = *row_bound;
    if (b < end) end = static_cast<int>(b < 0 ? 0 : b);
  }

  __shared__ float shared[kGroupSize];
  for (int row = blockIdx.x; row < end; row += gridDim.x) {
    const int data_row = row_map[row];
    if (data_row < 0) continue;
    float activated = 0.0f;
    if (col < hidden_size) {
      const __nv_bfloat16* token_gate =
          input + (size_t)data_row * input_stride + group_start;
      const __nv_bfloat16* token_up = token_gate + hidden_size;
      float gate = __bfloat162float(token_gate[tid]);
      float up = __bfloat162float(token_up[tid]);
      float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
      // Match vLLM's fused Triton kernel: SiLU narrows to the input dtype
      // before the BF16 multiply, then the product is quantized from F32.
      __nv_bfloat16 glu = __float2bfloat16_rn(gate * sigmoid_gate);
      activated = __bfloat162float(glu) * up;
    }
    shared[tid] = fabsf(activated);
    __syncthreads();

#pragma unroll
    for (int stride = kGroupSize / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      shared[0] = fmaxf(shared[0], kPerTokenGroupQuantEps) / kFp8Max;
      if constexpr (kUe8m0) {
        shared[0] = __uint_as_float(
            (__float_as_uint(shared[0]) + 0x007FFFFFu) & 0x7F800000u);
      }
      const int g = data_row / masked_cap;
      const int r_local = data_row % masked_cap;
      scales[((size_t)g * scale_cols + group) * masked_cap + r_local] = shared[0];
    }
    __syncthreads();

    if (col < hidden_size) {
      output[(size_t)data_row * hidden_size + col] =
          quantize_e4m3(activated, shared[0]);
    }
    __syncthreads();
  }
}

// f32 mn-major scales [groups, scale_cols, cap] → packed UE8M0 i32
// [groups, scale_cols/4, cap], LSB-first exponent bytes
// (u32 = b0 | b1<<8 | b2<<16 | b3<<24 — smxx_layout.cuh's
// (f0>>23)|(f1>>15)|(f2>>7)|(f3<<1)). Dense full-cover pass: every packed
// word is rewritten each step, so stale bytes never reach the GEMM's UTCCP.
// Inputs must already be power-of-two scales (emit them with the kUe8m0
// quant variants); the SM100 kernel device-asserts exponent-only values.
__global__ void fp8_scale_pack_ue8m0_kernel(const float* __restrict__ scales,
                                            int* __restrict__ packed,
                                            int groups, int scale_cols,
                                            int cap) {
  const int packed_cols = scale_cols / 4;
  const size_t total = (size_t)groups * packed_cols * cap;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
       idx += (size_t)gridDim.x * blockDim.x) {
    const int m = idx % cap;
    const size_t gi = idx / cap;
    const float* base = scales + (gi * 4) * (size_t)cap + m;
    unsigned int word = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned int bits = __float_as_uint(base[(size_t)j * cap]);
      word |= ((bits >> 23) & 0xFFu) << (8 * j);
    }
    packed[idx] = static_cast<int>(word);
  }
}

// Loader-time weight UE8M0 requant: per 128x128 block,
// s' = 2^ceil(log2 s) and q' = round_e4m3(q * s / s'), in place on the fp8
// bank. s/s' ∈ (0.5, 1] so no overflow is possible. Grid-strided elementwise
// over the bank bytes (gridDim.y == groups).
__global__ void fp8_weight_ue8m0_requant_kernel(unsigned char* __restrict__ weight,
                                                const float* __restrict__ scales,
                                                int n, int k) {
  const int n_blocks = n / 128;
  const int k_blocks = k / 128;
  const size_t per_group = (size_t)n * k;
  const size_t group_base = (size_t)blockIdx.y * per_group;
  const float* group_scales =
      scales + (size_t)blockIdx.y * n_blocks * k_blocks;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
       idx < per_group; idx += (size_t)gridDim.x * blockDim.x) {
    const int row = idx / k;
    const int col = idx % k;
    const float s = group_scales[(row >> 7) * k_blocks + (col >> 7)];
    // s' = 2^ceil(log2 s): same bit bump as the quant kernels.
    const float sp =
        __uint_as_float((__float_as_uint(s) + 0x007FFFFFu) & 0x7F800000u);
    const float value =
        __half2float(__nv_cvt_fp8_to_halfraw(weight[group_base + idx], __NV_E4M3));
    weight[group_base + idx] = quantize_e4m3(value * (s / sp), 1.0f);
  }
}

// Loader-time weight-scale pack: block scales [groups, n/128, k/128] f32 →
// packed UE8M0 i32 [groups, k/512, n], each block's exponent replicated
// across its 128 rows (the SFB contract stores per-row exponents). The scale
// tensor itself stays unmodified (requant above only touches the fp8 bank),
// so this kernel applies the same 2^ceil(log2 s) bump before extracting the
// exponent byte — identical to the s' the requant used (gridDim.y == groups).
__global__ void fp8_weight_scale_pack_ue8m0_kernel(
    const float* __restrict__ scales, int* __restrict__ packed, int n, int k) {
  const int n_blocks = n / 128;
  const int k_blocks = k / 128;
  const int packed_cols = k_blocks / 4;  // == k / 512
  const float* group_scales =
      scales + (size_t)blockIdx.y * n_blocks * k_blocks;
  int* group_packed = packed + (size_t)blockIdx.y * packed_cols * n;
  const size_t per_group = (size_t)packed_cols * n;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
       idx < per_group; idx += (size_t)gridDim.x * blockDim.x) {
    const int row = idx % n;
    const int i = idx / n;
    const float* base = group_scales + (size_t)(row >> 7) * k_blocks + i * 4;
    unsigned int word = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned int sp = (__float_as_uint(base[j]) + 0x007FFFFFu) & 0x7F800000u;
      word |= ((sp >> 23) & 0xFFu) << (8 * j);
    }
    group_packed[(size_t)i * n + row] = static_cast<int>(word);
  }
}

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

// Row-grid cap for the grid-strided quant kernels: enough blocks to fill the
// SMs at 128 threads/block, small enough that a capacity-sized (2080-row)
// launch does not pay ~100k block-schedules for ~400 real rows.
constexpr int kMaxRowBlocks = 256;
int row_grid(int rows) { return rows < kMaxRowBlocks ? rows : kMaxRowBlocks; }

bool valid_quant_shape(int rows, int width, int group_size) {
  return rows > 0 && width > 0 && group_size == kGroupSize &&
         width % kGroupSize == 0;
}

}  // namespace

extern "C" {

CUresult glm52_fp8_per_token_group_quant_bf16_cuda(
    const __nv_bfloat16* input, unsigned char* output, float* scales, int rows,
    int hidden_dim, int group_size, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!valid_quant_shape(rows, hidden_dim, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  dim3 grid(row_grid(rows), hidden_dim / kGroupSize, 1);
  fp8_per_token_group_quant_bf16_k128_kernel<false>
      <<<grid, kGroupSize, 0, stream>>>(input, output, scales, rows,
                                        hidden_dim, nullptr, nullptr, 0);
  return consume_last_cuda_error();
}

// UE8M0-scale variant for the FlashMLA fp8 sparse KV cache (see the kernel
// comment: sm100 truncates stored scales to powers of two).
CUresult glm52_fp8_per_token_group_quant_bf16_ue8m0_cuda(
    const __nv_bfloat16* input, unsigned char* output, float* scales, int rows,
    int hidden_dim, int group_size, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!valid_quant_shape(rows, hidden_dim, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  dim3 grid(row_grid(rows), hidden_dim / kGroupSize, 1);
  fp8_per_token_group_quant_bf16_k128_kernel<false, true>
      <<<grid, kGroupSize, 0, stream>>>(input, output, scales, rows,
                                        hidden_dim, nullptr, nullptr, 0);
  return consume_last_cuda_error();
}

CUresult glm52_fp8_per_token_group_quant_bf16_masked_cuda(
    const __nv_bfloat16* input, unsigned char* output, float* scales, int rows,
    int hidden_dim, int group_size, const long long* row_bound,
    const int* row_map, int masked_cap, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      row_bound == nullptr || row_map == nullptr || masked_cap <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!valid_quant_shape(rows, hidden_dim, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dim3 grid(row_grid(rows), hidden_dim / kGroupSize, 1);
  fp8_per_token_group_quant_bf16_k128_kernel<true, true>
      <<<grid, kGroupSize, 0, stream>>>(input, output, scales, rows,
                                        hidden_dim, row_bound, row_map,
                                        masked_cap);
  return consume_last_cuda_error();
}

CUresult glm52_silu_and_mul_per_token_group_quant_bf16_masked_cuda(
    const __nv_bfloat16* input, unsigned char* output, float* scales, int rows,
    int hidden_size, int group_size, const long long* row_bound,
    const int* row_map, int masked_cap, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      row_bound == nullptr || row_map == nullptr || masked_cap <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!valid_quant_shape(rows, hidden_size, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dim3 grid(row_grid(rows), hidden_size / kGroupSize, 1);
  silu_and_mul_per_token_group_quant_bf16_k128_masked_kernel<true>
      <<<grid, kGroupSize, 0, stream>>>(input, output, scales, rows,
                                        hidden_size, row_bound, row_map,
                                        masked_cap);
  return consume_last_cuda_error();
}

CUresult glm52_fp8_scale_pack_ue8m0_cuda(
    const float* scales, int* packed, int groups, int scale_cols, int cap,
    cudaStream_t stream) {
  if (scales == nullptr || packed == nullptr || groups <= 0 || cap <= 0 ||
      scale_cols <= 0 || scale_cols % 4 != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const size_t total = (size_t)groups * (scale_cols / 4) * cap;
  const int threads = 256;
  const size_t needed = (total + threads - 1) / threads;
  const int blocks = static_cast<int>(needed < 256 ? needed : 256);
  fp8_scale_pack_ue8m0_kernel<<<blocks, threads, 0, stream>>>(
      scales, packed, groups, scale_cols, cap);
  return consume_last_cuda_error();
}

CUresult glm52_fp8_weight_ue8m0_requant_cuda(
    unsigned char* weight, const float* scales, int groups, int n, int k,
    cudaStream_t stream) {
  if (weight == nullptr || scales == nullptr || groups <= 0 || n <= 0 ||
      k <= 0 || n % kGroupSize != 0 || k % kGroupSize != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const size_t elems = (size_t)n * k;
  const int threads = 256;
  const size_t needed = (elems + threads - 1) / threads;
  const int blocks = static_cast<int>(needed < 256 ? needed : 256);
  fp8_weight_ue8m0_requant_kernel<<<dim3(blocks, groups), threads, 0, stream>>>(
      weight, scales, n, k);
  return consume_last_cuda_error();
}

CUresult glm52_fp8_weight_scale_pack_ue8m0_cuda(
    const float* scales, int* packed, int groups, int n, int k,
    cudaStream_t stream) {
  if (scales == nullptr || packed == nullptr || groups <= 0 || n <= 0 ||
      k <= 0 || n % kGroupSize != 0 || k % (4 * kGroupSize) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const size_t elems = (size_t)(k / (4 * kGroupSize)) * n;
  const int threads = 256;
  const size_t needed = (elems + threads - 1) / threads;
  const int blocks = static_cast<int>(needed < 256 ? needed : 256);
  fp8_weight_scale_pack_ue8m0_kernel
      <<<dim3(blocks, groups), threads, 0, stream>>>(scales, packed, n, k);
  return consume_last_cuda_error();
}

}  // extern "C"
