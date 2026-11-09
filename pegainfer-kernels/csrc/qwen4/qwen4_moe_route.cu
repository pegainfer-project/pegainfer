// Qwen3.8-Flash-Next (Qwen4 line) MoE block: router routing and the
// shared-expert combine.
//
// Frozen semantics: transformers 5.16.1 `qwen4_exp`,
// `Qwen4ExpTextSparseMoeBlock` at the frozen checkpoint's geometry — hidden
// 2560, 512 experts, top-10 routed plus 1 shared. This file holds the two
// pieces with no shared entry point; the router GEMM, the shared branch's
// projections and the expert GEMMs go through the ops every line uses.
//
// Routing (one block per token):
//
//   p     = softmax(logits) in fp32 over all 512 experts — the reference
//           upcasts (`softmax(router_logits, dtype=torch.float)`); the logits
//           themselves stay bf16, as the router GEMM stores them.
//   top-k = 10 picks in the canonical order: descending probability, ascending
//           expert index inside a tie (`torch.topk` defines no order; this
//           kernel is deterministic).
//   w     = p / sum(selected p) when `renorm` — the checkpoint's
//           `norm_topk_prob`, absent from `config.json` and defaulting to True
//           upstream — else p. Rounded to bf16 once, at the store.
//
// A non-finite logit fails the whole row closed — NaN weights and placeholder
// ids (the slot index) — the policy gemma4's router established, by the
// opposite mechanism (the kernel's vote carries the comparison). The check is a
// block-uniform vote *before* the softmax, so no NaN ever reaches the
// arithmetic: max-subtraction would otherwise mask a -inf into an
// ordinary-looking weight, and a NaN compares false against everything, so an
// argmax tree meeting one returns the reduction's own padding index as the
// "winner".
//
// Shared combine (grid-stride elementwise):
//
//   g   = bf16(sigmoid(bf16 gate_logit))    — the learned per-token scalar
//         gate, `shared_expert_gate`, a [1, 2560] linear whose output the
//         reference sigmoid-materializes in bf16 before the multiply.
//   out = bf16(expert_out + bf16(g * shared_out))
//
// every operator bf16-rounded exactly where the reference rounds. The gate is
// fused here rather than left to the caller.
//
// Target: the sm_89-common portable subset — no TMA, no cluster, no wgmma;
// plain SIMT shuffle/shared-memory block reductions with a fixed tree, so the
// outputs are deterministic run to run.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>

namespace {

constexpr int kExperts = 512;  // num_experts
constexpr int kTopK = 10;      // num_experts_per_tok
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / WARP_SIZE;
constexpr int kPerThread = kExperts / kThreads;

static_assert(kExperts % kThreads == 0,
              "one thread owns a whole number of experts");
static_assert(kTopK <= kThreads,
              "the fail-closed path writes one slot per thread");

// The canonical tie rule: descending probability, ascending expert index.
__device__ __forceinline__ bool route_better(float value, int expert,
                                             float best_value, int best_expert) {
  return value > best_value || (value == best_value && expert < best_expert);
}

// Block-wide best (value, index) under `route_better`, broadcast to every
// thread. Fixed shuffle tree plus a fixed second level over the warps, so the
// result — including which index wins an exact tie — is deterministic.
__device__ __forceinline__ void block_reduce_best(float value, int index,
                                                  float (&s_v)[kWarps],
                                                  int (&s_i)[kWarps],
                                                  float& b_v, int& b_i,
                                                  float& out_v, int& out_i) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;
#pragma unroll
  for (int width = WARP_SIZE / 2; width > 0; width >>= 1) {
    const float other_v = __shfl_down_sync(0xffffffffu, value, width);
    const int other_i = __shfl_down_sync(0xffffffffu, index, width);
    if (route_better(other_v, other_i, value, index)) {
      value = other_v;
      index = other_i;
    }
  }
  if (lane == 0) {
    s_v[warp] = value;
    s_i[warp] = index;
  }
  __syncthreads();
  if (warp == 0) {
    value = lane < kWarps ? s_v[lane] : -FLT_MAX;
    index = lane < kWarps ? s_i[lane] : kExperts;
#pragma unroll
    for (int width = WARP_SIZE / 2; width > 0; width >>= 1) {
      const float other_v = __shfl_down_sync(0xffffffffu, value, width);
      const int other_i = __shfl_down_sync(0xffffffffu, index, width);
      if (route_better(other_v, other_i, value, index)) {
        value = other_v;
        index = other_i;
      }
    }
    if (lane == 0) {
      b_v = value;
      b_i = index;
    }
  }
  __syncthreads();
  out_v = b_v;
  out_i = b_i;
}

// Block-wide sum, broadcast to every thread. Same fixed tree as the max, so
// the fp32 softmax denominator is run-to-run deterministic (its *order* still
// differs from torch's; the bf16 store is where the reference rounds).
__device__ __forceinline__ float block_reduce_sum(float value,
                                                  float (&s)[kWarps],
                                                  float& b) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;
  value = warp_reduce_sum(value);
  if (lane == 0) s[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < kWarps ? s[lane] : 0.0f;
    value = warp_reduce_sum(value);
    if (lane == 0) b = value;
  }
  __syncthreads();
  return b;
}

// Fail one token's routing closed: NaN weight and the slot index as a
// placeholder id for all `kTopK` slots, so the row is unusable by inspection
// and no gather ever indexes it by a winner value. One thread per slot; needs
// `kTopK <= kThreads`, and no barrier, because every caller invokes it
// block-uniformly.
__device__ __forceinline__ void fail_row_closed(__nv_bfloat16* weights,
                                                int* ids, int row, int tid) {
  if (tid < kTopK) {
    const size_t at = static_cast<size_t>(row) * kTopK + tid;
    weights[at] = __float2bfloat16(NAN);
    ids[at] = tid;
  }
}

__global__ void qwen4_moe_route_kernel(const __nv_bfloat16* __restrict__ logits,
                                          __nv_bfloat16* __restrict__ weights,
                                          int* __restrict__ ids, int rows,
                                          int renorm) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;

  __shared__ float s_v[kWarps];
  __shared__ int s_i[kWarps];
  __shared__ float b_v;
  __shared__ int b_i;
  __shared__ float s_sum[kWarps];
  __shared__ float b_sum;
  __shared__ float s_probs[kExperts];

  const __nv_bfloat16* row_logits = logits + static_cast<size_t>(row) * kExperts;
  float held[kPerThread];
  bool poisoned = false;
#pragma unroll
  for (int slot = 0; slot < kPerThread; ++slot) {
    const int expert = tid + slot * kThreads;
    const float logit = __bfloat162float(row_logits[expert]);
    poisoned |= !isfinite(logit);
    held[slot] = logit;
  }
  // One non-finite logit fails the whole row closed. `__syncthreads_or` makes
  // the verdict block-uniform, so the early return is uniform too — and it
  // keeps the rest of the kernel free of NaN, whose comparisons are all false
  // and would otherwise slip through the argmax tree as a "winner" that is
  // really the thread's own index. The softmax itself would mask a -inf and
  // hand out ordinary-looking weights; that is the trap this closes.
  //
  // Same *policy* as gemma4's router, opposite *mechanism* — and this crate now
  // carries both, so the difference is worth stating. `csrc/gemma4/moe.cu`
  // votes with `__any_sync` and then floods `held[]` with NaN, letting it ride
  // through the softmax and the argmax tree and catching it at the far end in
  // the `taken >= experts` sentinel branch, which rewrites the pick to the slot
  // index and the weight to NaN. Here the vote returns instead, so NaN never
  // enters the arithmetic and no later branch has to recognize one. Both land
  // on the same observable bytes — NaN weights, slot-index placeholder ids —
  // but only the flood depends on a sentinel surviving comparisons against
  // NaN. A new router should copy this early return, not the flood.
  if (__syncthreads_or(poisoned)) {
    fail_row_closed(weights, ids, row, tid);
    return;
  }

  // fp32 softmax over all experts: max-subtracted, so the largest term is
  // exactly 1 and the sum is >= 1 — no zero-sum guard needed.
  float peak_value = -FLT_MAX;
  int peak_expert = kExperts;
#pragma unroll
  for (int slot = 0; slot < kPerThread; ++slot) {
    const int expert = tid + slot * kThreads;
    if (route_better(held[slot], expert, peak_value, peak_expert)) {
      peak_value = held[slot];
      peak_expert = expert;
    }
  }
  float peak;
  int peak_at;
  block_reduce_best(peak_value, peak_expert, s_v, s_i, b_v, b_i, peak, peak_at);

  float partial = 0.0f;
#pragma unroll
  for (int slot = 0; slot < kPerThread; ++slot) {
    held[slot] = expf(held[slot] - peak);
    partial += held[slot];
  }
  const float total = block_reduce_sum(partial, s_sum, b_sum);

#pragma unroll
  for (int slot = 0; slot < kPerThread; ++slot) {
    held[slot] /= total;
  }

  // Top-k by K masked argmax passes, all of them inside one warp: the row's
  // 512 probabilities cross to shared memory once, lane l takes the contiguous
  // slice [16l, 16l+16) into registers, and each pass is a 16-deep lane scan
  // plus a shuffle tree. Pass k takes the k-th canonical pick, so the picks
  // come out in the canonical order and the running selected sum accumulates in
  // that same descending order.
  //
  // The picks and the weights are bit-identical to a block-wide reduction per
  // pass: (value desc, expert asc) is a total order, the passes take the same
  // sequence of winners, and `selected_sum` accumulates in that same order.
  // What changes is the price of the serial structure — a block-wide reduction
  // costs two `__syncthreads` per pass where this costs none, because the
  // winner is masked in the owning lane's registers (the pass broadcasts
  // `win_at`, so the sweep is over compile-time indices and no lane needs a
  // shared-memory write to hide a pick). Measured on an RTX 5060 at 1.2 GHz,
  // one token: 12.9 us per launch against 16.4 us for the block-wide form, and
  // 36.5 vs 52.4 us at 512 tokens, with a 3.2-4.2 us empty-kernel floor.
#pragma unroll
  for (int slot = 0; slot < kPerThread; ++slot) {
    s_probs[tid + slot * kThreads] = held[slot];
  }
  __syncthreads();

  if (tid >= WARP_SIZE) return;
  const int lane = tid;
  constexpr int kLane = kExperts / WARP_SIZE;
  float vals[kLane];
#pragma unroll
  for (int i = 0; i < kLane; ++i) {
    vals[i] = s_probs[lane * kLane + i];
  }
  int win_expert[kTopK];
  float win_prob[kTopK];
  float selected_sum = 0.0f;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float value = vals[0];
    int expert = lane * kLane;
#pragma unroll
    for (int i = 1; i < kLane; ++i) {
      const int e = lane * kLane + i;
      if (route_better(vals[i], e, value, expert)) {
        value = vals[i];
        expert = e;
      }
    }
#pragma unroll
    for (int width = WARP_SIZE / 2; width > 0; width >>= 1) {
      const float other_v = __shfl_down_sync(0xffffffffu, value, width);
      const int other_i = __shfl_down_sync(0xffffffffu, expert, width);
      if (route_better(other_v, other_i, value, expert)) {
        value = other_v;
        expert = other_i;
      }
    }
    // kExperts == WARP_SIZE * kLane, so every lane scans a full slice and the
    // tree's winner is a real expert: the block-wide path's out-of-range
    // sentinel (a padding lane winning the row) has no counterpart here. Every
    // probability is finite — a non-finite row returned above.
    const float win_value = __shfl_sync(0xffffffffu, value, 0);
    const int win_at = __shfl_sync(0xffffffffu, expert, 0);
    selected_sum += win_value;
    if (lane == 0) {
      win_prob[k] = win_value;
      win_expert[k] = win_at;
    }
    // Mask the winner so no later pass can take it again. The owner is known to
    // every lane (`win_at`), and the sweep is over compile-time indices, so the
    // mask stays in registers and the next pass needs no barrier.
#pragma unroll
    for (int i = 0; i < kLane; ++i) {
      if (lane * kLane + i == win_at) vals[i] = -INFINITY;
    }
  }
  if (lane != 0) return;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float value = win_prob[k];
    if (renorm != 0) value = value / selected_sum;
    const size_t at = static_cast<size_t>(row) * kTopK + k;
    weights[at] = __float2bfloat16(value);
    ids[at] = win_expert[k];
  }
}

__global__ void qwen4_moe_shared_combine_kernel(
    const __nv_bfloat16* __restrict__ gate_logit,
    const __nv_bfloat16* __restrict__ expert_out,
    const __nv_bfloat16* __restrict__ shared_out, __nv_bfloat16* __restrict__ out,
    long long total, int hidden) {
  for (long long at = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
       at < total;
       at += gridDim.x * static_cast<long long>(blockDim.x)) {
    const long long token = at / hidden;
    // The reference materializes the sigmoid in bf16 before the multiply, and
    // rounds the product before the sum — both roundings reproduced.
    const float logit = __bfloat162float(gate_logit[token]);
    const float gate =
        __bfloat162float(__float2bfloat16(1.0f / (1.0f + expf(-logit))));
    const float shared = __bfloat162float(shared_out[at]);
    const float expert = __bfloat162float(expert_out[at]);
    const float gated = __bfloat162float(__float2bfloat16(gate * shared));
    out[at] = __float2bfloat16(expert + gated);
  }
}

}  // namespace

extern "C" {

// `logits` is `[rows, 512]` bf16 as the router GEMM stores it; `weights` and
// `ids` are `[rows, 10]`. `renorm` is the checkpoint's `norm_topk_prob`.
CUresult qwen4_moe_route_cuda(const __nv_bfloat16* logits,
                                 __nv_bfloat16* weights, int* ids, int rows,
                                 int experts, int top_k, int renorm,
                                 cudaStream_t stream) {
  if (logits == nullptr || weights == nullptr || ids == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rows < 0 || experts != kExperts || top_k != kTopK) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (renorm != 0 && renorm != 1) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rows == 0) return CUDA_SUCCESS;
  qwen4_moe_route_kernel<<<rows, kThreads, 0, stream>>>(
      logits, weights, ids, rows, renorm);
  return map_cuda_error(cudaGetLastError());
}


// `gate_logit` is `[tokens]` bf16 (the `[1, 2560]` gate linear's output);
// `expert_out`, `shared_out` and `out` are `[tokens, hidden]` bf16.
CUresult qwen4_moe_shared_combine_cuda(const __nv_bfloat16* gate_logit,
                                          const __nv_bfloat16* expert_out,
                                          const __nv_bfloat16* shared_out,
                                          __nv_bfloat16* out, int tokens,
                                          int hidden, cudaStream_t stream) {
  if (gate_logit == nullptr || expert_out == nullptr || shared_out == nullptr ||
      out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (tokens < 0 || hidden <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (tokens == 0) return CUDA_SUCCESS;
  const long long total = static_cast<long long>(tokens) * hidden;
  const int block = 256;
  const long long grid = (total + block - 1) / block;
  if (grid > 2147483647LL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  qwen4_moe_shared_combine_kernel<<<static_cast<int>(grid), block, 0,
                                       stream>>>(gate_logit, expert_out,
                                                 shared_out, out, total, hidden);
  return map_cuda_error(cudaGetLastError());
}

}  // extern "C"
