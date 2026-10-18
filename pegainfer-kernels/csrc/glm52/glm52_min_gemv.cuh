// Min-latency bf16 GEMV for skinny decode-side projections, adapted from
// TRT-LLM dsv3MinLatencyKernels dsv3RouterGemm.cu (Apache-2.0) via
// SGLang/vLLM. One block per output row, fixed f32 reduction order
// (deterministic); replaces cublas splitK plans that cost 4 whole-step-graph
// nodes per call (GEMM + splitKreduce + workspace alloc/free).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// GLM5.2 is Blackwell-only; refuse Hopper and older targets.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
#error "GLM5.2 kernels require sm_100+ (check PEGAINFER_CUDA_SM / detected GPU targets)"
#endif

namespace glm52_min_gemv {

constexpr int kHidden = 6144;
constexpr int kMaxTokens = 96;  // = GLM52_MAX_STEP_ROWS (#812/#817);
                                // dispatch covers 1..=8 plus the bucket
                                // sizes. 64/96 decompose into two sub-launches
                                // (token-major out lets the tile split on the
                                // token axis) — a 96-wide accumulator tile
                                // would spill far past 255 regs/thread.

// out[token * kNumRows + row] = dot(hidden[token], weight[row]); weight is
// row-major [kNumRows, kHidden].
template <int kNumTokens, int kNumRows, typename OutT>
__global__ __launch_bounds__(128, 1) void kernel(
    OutT* __restrict__ out, const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight) {
  constexpr int kBlockSize = 128;
  constexpr int kVpt = 8;  // bf16 per uint4 vector load
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  constexpr int kElemsPerIter = kVpt * kBlockSize;
  constexpr int kIters = kHidden / kElemsPerIter;
  static_assert(kHidden % kElemsPerIter == 0);

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const __nv_bfloat16* w_row = weight + static_cast<size_t>(row) * kHidden;

  float acc[kNumTokens] = {};
  __shared__ float sm_reduction[kNumTokens][kNumWarps];

  cudaGridDependencySynchronize();

  for (int ki = 0; ki < kIters; ++ki) {
    const int k_base = ki * kElemsPerIter + tid * kVpt;
    const uint4 w_vec = *reinterpret_cast<const uint4*>(w_row + k_base);
    const __nv_bfloat16* w_bf16 = reinterpret_cast<const __nv_bfloat16*>(&w_vec);
#pragma unroll
    for (int m = 0; m < kNumTokens; ++m) {
      const uint4 h_vec = *reinterpret_cast<const uint4*>(
          hidden + static_cast<size_t>(m) * kHidden + k_base);
      const __nv_bfloat16* h_bf16 =
          reinterpret_cast<const __nv_bfloat16*>(&h_vec);
#pragma unroll
      for (int k = 0; k < kVpt; ++k) {
        acc[m] += __bfloat162float(h_bf16[k]) * __bfloat162float(w_bf16[k]);
      }
    }
  }

  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
#pragma unroll
  for (int m = 0; m < kNumTokens; ++m) {
    float sum = acc[m];
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    if (lane_id == 0) sm_reduction[m][warp_id] = sum;
  }
  __syncthreads();

  if (tid == 0) {
#pragma unroll
    for (int m = 0; m < kNumTokens; ++m) {
      float final_sum = 0.0f;
#pragma unroll
      for (int w = 0; w < kNumWarps; ++w) final_sum += sm_reduction[m][w];
      out[m * kNumRows + row] = static_cast<OutT>(final_sum);
    }
  }
  // Fence-free like upstream TRT-LLM: dependents can only consume after
  // their cudaGridDependencySynchronize, which orders on our full completion.
  cudaTriggerProgrammaticLaunchCompletion();
}

template <int kNumTokens, int kNumRows, typename OutT>
cudaError_t launch(OutT* out, const __nv_bfloat16* hidden,
                   const __nv_bfloat16* weight, cudaStream_t stream) {
  // PSS needs cc >= 9.0 — a build invariant here (see the #error above).
  cudaLaunchConfig_t config = {};
  config.gridDim = kNumRows;
  config.blockDim = 128;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.numAttrs = 1;
  config.attrs = attrs;
  return cudaLaunchKernelEx(&config, kernel<kNumTokens, kNumRows, OutT>, out,
                            hidden, weight);
}

// Runtime dispatch over 1..=kMaxTokens.
template <int kNumRows, typename OutT>
cudaError_t launch_tokens(OutT* out, const __nv_bfloat16* hidden,
                          const __nv_bfloat16* weight, int tokens,
                          cudaStream_t stream) {
  switch (tokens) {
    case 1: return launch<1, kNumRows>(out, hidden, weight, stream);
    case 2: return launch<2, kNumRows>(out, hidden, weight, stream);
    case 3: return launch<3, kNumRows>(out, hidden, weight, stream);
    case 4: return launch<4, kNumRows>(out, hidden, weight, stream);
    case 5: return launch<5, kNumRows>(out, hidden, weight, stream);
    case 6: return launch<6, kNumRows>(out, hidden, weight, stream);
    case 7: return launch<7, kNumRows>(out, hidden, weight, stream);
    case 8: return launch<8, kNumRows>(out, hidden, weight, stream);
    // Verify-span buckets (#812): token counts land on the decode bucket
    // sizes, so only the bucket members above 8 are instantiated.
    case 16: return launch<16, kNumRows>(out, hidden, weight, stream);
    case 32: return launch<32, kNumRows>(out, hidden, weight, stream);
    case 48: return launch<48, kNumRows>(out, hidden, weight, stream);
    case 64: {
      const cudaError_t first =
          launch<32, kNumRows>(out, hidden, weight, stream);
      if (first != cudaSuccess) return first;
      return launch<32, kNumRows>(out + 32 * static_cast<size_t>(kNumRows),
                                  hidden + 32 * static_cast<size_t>(kHidden),
                                  weight, stream);
    }
    case 96: {
      const cudaError_t first =
          launch<48, kNumRows>(out, hidden, weight, stream);
      if (first != cudaSuccess) return first;
      return launch<48, kNumRows>(out + 48 * static_cast<size_t>(kNumRows),
                                  hidden + 48 * static_cast<size_t>(kHidden),
                                  weight, stream);
    }
    default: return cudaErrorInvalidValue;
  }
}

}  // namespace glm52_min_gemv
