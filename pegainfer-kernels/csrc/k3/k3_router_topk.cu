// Kimi-K3 MoE router: sigmoid scores plus biased top-k selection.
//
// Replaces the retired TileLang `router_topk_batched` kernel, which ran the
// whole top-k as a serial scan on thread 0 (TOPK x E iterations per row —
// 65.7us per launch at E=896, ~6ms of every EP16 verify round). The selection
// here is a block-parallel argmax per round, but every arithmetic step is the
// retired kernel's spelling so the outputs are bit-identical:
//
//   scores[e] = 1 / (1 + expf(0 - s[e]))          (plain expf, f32 division)
//   biased[e] = scores[e] + bias[e]               (bias read in f32)
//   TOPK rounds of argmax over `biased` with strict `<` comparison — the
//     first index attaining the maximum wins, i.e. ties break to the lowest
//     expert index; the winner's biased score is set to -1e30 and its
//     *un-biased* score joins the weight row and the denominator, in
//     selection order;
//   wts[t] = (wts[t] / (den + 1e-20)) * (float)rs  (division, then scale)
//
// The parallel argmax preserves the serial tie-break exactly: each thread
// scans its stride-256 subsequence ascending with strict `<` (keeping its
// local first maximum), and the shuffle/shared reduction prefers the
// strictly-greater value, breaking equal values to the lower index — the
// result is the minimum index among the global maxima, which is precisely
// the index the serial first-match scan selects. All comparisons are on
// identical f32 values, so no summation-order freedom exists anywhere.
//
// Deterministic and CUDA-graph safe: fixed reduction order, no allocation,
// no host readback; grid is (b) with everything else read from device
// tensors.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / WARP_SIZE;
constexpr float kNeg = -1.0e30f;

__global__ void router_topk_kernel(const float* __restrict__ s,
                                   const float* __restrict__ bias,
                                   const __nv_bfloat16* __restrict__ rs,
                                   int* __restrict__ idx,
                                   float* __restrict__ wts, int num_experts,
                                   int topk) {
  extern __shared__ float smem[];
  float* scores = smem;                // [num_experts]
  float* biased = smem + num_experts;  // [num_experts]
  __shared__ float red_v[kWarps];
  __shared__ int red_i[kWarps];

  const int bb = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (WARP_SIZE - 1);
  const float* srow = s + (size_t)bb * num_experts;

  for (int e = tid; e < num_experts; e += kThreads) {
    const float sig = 1.0f / (1.0f + expf(0.0f - srow[e]));
    scores[e] = sig;
    biased[e] = sig + bias[e];
  }
  __syncthreads();

  float den = 0.0f;  // only thread 0's copy accumulates
  for (int t = 0; t < topk; ++t) {
    float best = kNeg;
    int bi = 0;
    for (int e = tid; e < num_experts; e += kThreads) {
      if (best < biased[e]) {
        best = biased[e];
        bi = e;
      }
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, best, off);
      const int oi = __shfl_down_sync(0xffffffffu, bi, off);
      if (ov > best || (ov == best && oi < bi)) {
        best = ov;
        bi = oi;
      }
    }
    if (lane == 0) {
      red_v[tid / WARP_SIZE] = best;
      red_i[tid / WARP_SIZE] = bi;
    }
    __syncthreads();
    if (tid == 0) {
      best = red_v[0];
      bi = red_i[0];
      for (int w = 1; w < kWarps; ++w) {
        if (red_v[w] > best || (red_v[w] == best && red_i[w] < bi)) {
          best = red_v[w];
          bi = red_i[w];
        }
      }
      idx[(size_t)bb * topk + t] = bi;
      wts[(size_t)bb * topk + t] = scores[bi];
      biased[bi] = kNeg;
      den += scores[bi];
    }
    // The winner's knock-out (and red_* reuse) must land before the next scan.
    __syncthreads();
  }

  if (tid == 0) {
    const float rsf = __bfloat162float(rs[0]);
    for (int t = 0; t < topk; ++t) {
      wts[(size_t)bb * topk + t] =
          (wts[(size_t)bb * topk + t] / (den + 1e-20f)) * rsf;
    }
  }
}

CUresult map_cuda_error(cudaError_t err) {
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  if (err == cudaErrorNotSupported) return CUDA_ERROR_NOT_SUPPORTED;
  return CUDA_ERROR_LAUNCH_FAILED;
}

}  // namespace

extern "C" {

// Sigmoid router plus biased top-k over merged f32 score rows `s [b, E]`,
// with `bias [E]` f32 and the bf16 routed scale `rs [1]`. Writes
// `idx [b, topk]` i32 and `wts [b, topk]` f32. Shapes are runtime values —
// no per-bucket instantiation; shared memory holds 2*E f32.
CUresult k3_router_topk_cuda(const float* s, const float* bias, const void* rs,
                             int* idx, float* wts, int b, int num_experts,
                             int topk, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (s == nullptr || bias == nullptr || rs == nullptr || idx == nullptr ||
      wts == nullptr || b <= 0 || num_experts <= 0 || topk <= 0 ||
      topk > num_experts) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const size_t shmem = 2 * (size_t)num_experts * sizeof(float);
  router_topk_kernel<<<b, kThreads, shmem, stream>>>(
      s, bias, static_cast<const __nv_bfloat16*>(rs), idx, wts, num_experts,
      topk);
  return map_cuda_error(cudaGetLastError());
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

}  // extern "C"
