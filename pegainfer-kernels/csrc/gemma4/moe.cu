// Gemma 4 routed-expert dispatch.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace {

// The strided lane scan and shuffle tree preserve lower-expert tie breaking.
constexpr int kRouterBlock = 32;
constexpr int kRouterExperts = 128;
constexpr int kRouterLaneExperts = kRouterExperts / kRouterBlock;

__global__ void gemma4_moe_router_topk_kernel(
    const __nv_bfloat16 *__restrict__ logits,
    const __nv_bfloat16 *__restrict__ per_expert_scale, int experts, int top_k,
    int *__restrict__ index_out, float *__restrict__ weight_out) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  const __nv_bfloat16 *row_logits = logits + (long long)row * experts;

  float held[kRouterLaneExperts];
  bool poisoned = false;
  for (int slot = 0; slot < kRouterLaneExperts; ++slot) {
    int e = lane + slot * kRouterBlock;
    held[slot] = e < experts ? __bfloat162float(row_logits[e]) : -FLT_MAX;
    poisoned |= e < experts && !isfinite(held[slot]);
  }
  // One non-finite logit fails the whole row closed, -inf included: the
  // softmax below would otherwise mask it and hand out ordinary weights.
  if (__any_sync(0xffffffff, poisoned)) {
    for (int slot = 0; slot < kRouterLaneExperts; ++slot) {
      held[slot] = NAN;
    }
  }
  float mine = -FLT_MAX;
  for (int slot = 0; slot < kRouterLaneExperts; ++slot) {
    mine = fmaxf(mine, held[slot]);
  }
  for (int width = kRouterBlock / 2; width > 0; width >>= 1) {
    mine = fmaxf(mine, __shfl_down_sync(0xffffffff, mine, width));
  }
  const float peak = __shfl_sync(0xffffffff, mine, 0);

  float total = 0.0f;
  for (int slot = 0; slot < kRouterLaneExperts; ++slot) {
    int e = lane + slot * kRouterBlock;
    if (e < experts) {
      held[slot] = __expf(held[slot] - peak);
      total += held[slot];
    }
  }
  for (int width = kRouterBlock / 2; width > 0; width >>= 1) {
    total += __shfl_down_sync(0xffffffff, total, width);
  }
  const float norm = __shfl_sync(0xffffffff, total, 0);
  for (int slot = 0; slot < kRouterLaneExperts; ++slot) {
    held[slot] /= norm;
  }

  float selected = 0.0f;
  float my_win = 0.0f;
  int my_taken = 0;
  for (int k = 0; k < top_k; ++k) {
    float best = -FLT_MAX;
    int best_at = experts;
    for (int slot = 0; slot < kRouterLaneExperts; ++slot) {
      int e = lane + slot * kRouterBlock;
      if (e < experts && (held[slot] > best || (held[slot] == best && e < best_at))) {
        best = held[slot];
        best_at = e;
      }
    }
    for (int width = kRouterBlock / 2; width > 0; width >>= 1) {
      float other = __shfl_down_sync(0xffffffff, best, width);
      int other_at = __shfl_down_sync(0xffffffff, best_at, width);
      if (other > best || (other == best && other_at < best_at)) {
        best = other;
        best_at = other_at;
      }
    }
    float win = __shfl_sync(0xffffffff, best, 0);
    int taken = __shfl_sync(0xffffffff, best_at, 0);
    if (taken >= experts) {
      // No candidate compared true: a non-finite logit poisoned the whole
      // row through the softmax. The pick stays valid and row-unique and
      // its weight goes out NaN, so nothing indexes past the arrays.
      taken = k;
      win = NAN;
    } else if (taken % kRouterBlock == lane) {
      // A taken slot must lose every later pick, including the lower-expert
      // tie against `best`'s floor that a -FLT_MAX clear would win once the
      // rest of the row is NaN; NaN compares false everywhere.
      held[taken / kRouterBlock] = NAN;
    }
    if (lane == k) {
      my_win = win;
      my_taken = taken;
    }
    selected += win;
  }

  if (lane < top_k) {
    const long long at = (long long)row * top_k + lane;
    index_out[at] = my_taken;
    weight_out[at] =
        my_win / selected * __bfloat162float(per_expert_scale[my_taken]);
  }
}

__global__ void gemma4_moe_sum_topk_kernel(
    const __nv_bfloat16 *__restrict__ routed, int top_k, int hidden,
    long long total, __nv_bfloat16 *__restrict__ out) {
  long long at = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (at >= total) {
    return;
  }
  const long long token = at / hidden;
  const int column = (int)(at % hidden);
  float sum = 0.0f;
  for (int pick = 0; pick < top_k; ++pick) {
    sum += __bfloat162float(routed[(token * top_k + pick) * hidden + column]);
  }
  out[at] = __float2bfloat16(sum);
}

} // namespace

extern "C" {

// `logits` is `[rows, experts]` bf16 as the router projection leaves it.
// `index_out` and `weight_out` are `[rows, top_k]`.
CUresult gemma4_moe_router_topk_cuda(const __nv_bfloat16 *logits,
                                     const __nv_bfloat16 *per_expert_scale,
                                     int rows, int experts, int top_k,
                                     int *index_out, float *weight_out,
                                     cudaStream_t stream) {
  if (logits == nullptr || per_expert_scale == nullptr ||
      index_out == nullptr || weight_out == nullptr || rows <= 0 ||
      experts != kRouterExperts || top_k <= 0 || top_k > experts ||
      top_k > kRouterBlock) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  gemma4_moe_router_topk_kernel<<<rows, kRouterBlock, 0, stream>>>(
      logits, per_expert_scale, experts, top_k, index_out, weight_out);
  return (CUresult)cudaGetLastError();
}

// The routed GEMM leaves one row per (token, pick); this folds the picks back
// onto their token. The per-pick weights are already in the rows.
CUresult gemma4_moe_sum_topk_cuda(const __nv_bfloat16 *routed, int rows,
                                  int top_k, int hidden, __nv_bfloat16 *out,
                                  cudaStream_t stream) {
  if (routed == nullptr || out == nullptr || rows <= 0 || top_k <= 0 ||
      hidden <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const long long total = (long long)rows * hidden;
  const int block = 256;
  const long long grid = (total + block - 1) / block;
  if (grid > 2147483647LL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  gemma4_moe_sum_topk_kernel<<<(int)grid, block, 0, stream>>>(
      routed, top_k, hidden, total, out);
  return (CUresult)cudaGetLastError();
}

} // extern "C"
