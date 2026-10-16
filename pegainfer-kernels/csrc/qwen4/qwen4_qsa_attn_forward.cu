// Qwen3.8-Flash-Next QSA sparse full-attention forward (Stage 2, plan P2).
//
// Frozen semantics: transformers 5.16.1 `qwen4_exp`,
// `Qwen4ExpTextAttention.forward` with the eager attention interface. Over one
// layer's captured activations, for the query rows a caller evaluates:
//
//   q_h, gate_h = chunk(q_proj(hidden).view(S, heads, head_dim * 2), 2, -1)
//   q_h         = (1 + q_norm_w) RMSNorm256(q_h), partial RoPE at the row's pos
//   k_kv        = (1 + k_norm_w) RMSNorm256(k_proj(hidden)_kv), RoPE at its pos
//   v_kv        = v_proj(hidden)_kv                       // neither normed nor
//                                                         // rotated
//   scores      = q_h . k_{h / group} / sqrt(head_dim)    // bf16 GEMM, then the
//                                                         // exact 1/16 scale
//   weights     = softmax(scores + additive_mask, fp32) cast back to bf16
//   out         = (weights . v) reshaped head-major * sigmoid(gate)
//
// `out` is the input to `o_proj`, not its output: the golden's expected tensor is
// captured with an `o_proj` forward-pre-hook, and `o_proj` is a plain GEMM this
// fixture does not carry.
//
// The mask arrives as the caller's own bool array — one row per evaluated query
// row, one byte per token, non-zero where the token is visible. This kernel never
// derives it: QSA's mask is the indexer's block selection AND causality, which is
// the indexer's and the selection stage's business, and a serving caller hands the
// result in. A masked score is set to `torch.finfo(bfloat16).min` rather than
// -inf, which is what the reference's additive float mask carries at a position
// one of its two masks hid (where both hid it the two mins sum to -inf, which
// softmax treats identically as long as one key in the row is visible) and what
// makes a fully hidden row come out uniform instead of NaN, exactly as the
// reference's does.
//
// Rounding is reproduced at every point the reference rounds, because the whole
// chain is bf16 and the compared tensors are bf16: the norms normalize in float32
// and cast once, cos/sin are cast to bf16 and then rotate in bf16 (each product
// and their sum rounding), the score is the bf16 GEMM result rounded before the
// scale, the softmax runs in float32 and its weights are rounded to bf16 *before*
// the context GEMM reads them, and the gate's sigmoid rounds before the final
// product. Float32 reduction *order* is this kernel's own, which is where the
// parity bound's residual comes from.
//
// Two kernels own one half each. `norm_rope_keys` prepares every token's two key
// heads once, because all `group` query heads of a KV head read the same prepared
// key. `attn` is one CTA per (query row, head): it prepares that head's query in
// shared memory, scores the whole key row with one warp per key stride, softmaxes
// it, and reduces the context over the row. Holding the row's scores in shared
// memory is what bounds `tokens` at `kMaxTokens`; a longer context needs the
// streaming (online-softmax) form, which is a later stage's business.
//
// Target: the sm_89-first portable subset (decision D3-B) — no TMA, no cluster,
// no wgmma. Both kernels are SIMT shuffle reductions: one CTA's score is a
// 256-wide dot and its context a 256-wide column reduction, neither of which
// wants an mma tile at these extents.
//
// The `(1 + w)` RMSNorm and partial-RoPE helpers here are deliberately the same
// code as the QSA token-indexer kernel's in `qwen4_qsa_indexer_score.cu`,
// which this directory gets from its own change: both reproduce the same
// reference rounding points, so a shared header would move the duplication
// rather than remove it, and lifting them belongs to whichever of the two lands
// second.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHeads = 24;      // num_attention_heads
constexpr int kKvHeads = 2;     // num_key_value_heads
constexpr int kGroup = kHeads / kKvHeads;
constexpr int kHeadDim = 256;   // head_dim
constexpr float kRopeTheta = 1.0e7f;
// head_dim(256) * partial_rotary_factor(0.25): the rope covers the first quarter
// of an attention head and leaves the rest alone.
constexpr int kRotaryDim = 64;
// The fused `q_proj` row: every head's query followed by that same head's output
// gate, which is what `torch.chunk(view(S, heads, head_dim * 2), 2, -1)` splits.
constexpr int kQueryWidth = kHeads * kHeadDim * 2;
constexpr int kKvWidth = kKvHeads * kHeadDim;
constexpr int kOutWidth = kHeads * kHeadDim;
// head_dim ** -0.5 is 1/16, a power of two, so scaling a rounded score is an
// exponent shift and cannot round a second time.
constexpr float kScale = 1.0f / 16.0f;
// `torch.finfo(bfloat16).min`: the reference's additive mask at a hidden
// position.
constexpr float kHiddenScore = -3.3895313892515355e38f;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / WARP_SIZE;
// One lane owns dims {lane * 8 .. lane * 8 + 8} in the score pass, so a key row
// is eight consecutive bf16 per lane and a warp reads a whole 512-byte row.
constexpr int kDimsPerLane = kHeadDim / WARP_SIZE;
// One row of float32 scores lives in dynamic shared memory, inside the 48 KiB a
// launch may ask for without opting in.
constexpr int kMaxTokens = 8192;

static_assert(kHeadDim == kThreads, "one thread owns one head dim");
static_assert(kGroup * kKvHeads == kHeads, "the GQA group must divide the heads");

// Sum across all 32 lanes, leaving it in every lane (common.cuh's
// `warp_reduce_sum` leaves it in lane 0, which the norms need broadcast).
__device__ __forceinline__ float warp_all_sum(float value) {
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, mask);
  }
  return value;
}

// Sum across the whole CTA, leaving it in every thread: each warp's partial goes
// to shared and every thread folds the `kWarps` partials in the same order, so
// one value is what all of them see. The leading barrier is what lets a caller
// reuse `scratch` for a second reduction.
__device__ __forceinline__ float block_all_sum(float value, float* scratch) {
  value = warp_all_sum(value);
  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    scratch[threadIdx.x / WARP_SIZE] = value;
  }
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int warp = 0; warp < kWarps; ++warp) {
    total += scratch[warp];
  }
  return total;
}

__device__ __forceinline__ float load_bf16(const __nv_bfloat16* value) {
  return __bfloat162float(*value);
}

__device__ __forceinline__ float round_bf16(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

// One `(1 + w)` RMSNorm element, rounded to bf16 exactly once: the reference
// normalizes in float32, scales by `1 + w` in float32, and then casts to the
// activation dtype.
__device__ __forceinline__ float norm_elem(float x, float inv_rms, float weight) {
  return round_bf16(x * inv_rms * (1.0f + weight));
}

// `1 / (theta ** (arange(0, rotary_dim, 2) / rotary_dim))`, rounded to float32
// once. The reference builds this table in float32 on the host, where the
// power is correctly rounded; CUDA's `powf` is specified to up to 8 ulp, which
// at the 2054-token case's positions is a 1e-3 radian angle error — an eighth of
// a bf16 ulp in every cos and sin it feeds. The double power, rounded once, is
// the same table the reference has.
__device__ __forceinline__ float inv_freq(int pair) {
  const double exponent =
      static_cast<double>(2 * pair) / static_cast<double>(kRotaryDim);
  return static_cast<float>(
      1.0 / pow(static_cast<double>(kRopeTheta), exponent));
}

// Partial RoPE at `pos` over the pair (d, d + rotary_dim/2), for d < 32. The
// reference casts cos/sin to the activation dtype and rotates in bf16, so each
// product and their sum round as well.
__device__ __forceinline__ void rope_pair(float& lo, float& hi, int pos,
                                          float freq) {
  const float angle = static_cast<float>(pos) * freq;
  const float cos_v = round_bf16(cosf(angle));
  const float sin_v = round_bf16(sinf(angle));
  const float lo_rot = round_bf16(lo * cos_v) + round_bf16(-hi * sin_v);
  const float hi_rot = round_bf16(hi * cos_v) + round_bf16(lo * sin_v);
  lo = round_bf16(lo_rot);
  hi = round_bf16(hi_rot);
}

// Every token's key heads, `(1 + w)` RMSNormed and rotated at the token's own
// position. One CTA per (token, KV head) row, one thread per dim.
__global__ void qwen4_qsa_norm_rope_keys_kernel(
    const __nv_bfloat16* __restrict__ keys,           // [tokens, kKvWidth]
    const __nv_bfloat16* __restrict__ k_norm_weight,  // [kHeadDim]
    const int* __restrict__ positions,                // [tokens]
    __nv_bfloat16* __restrict__ normed_keys,          // [tokens, kKvWidth]
    float rms_eps) {
  const int row = blockIdx.x;
  const int token = row / kKvHeads;
  const int dim = threadIdx.x;
  const int64_t base = static_cast<int64_t>(row) * kHeadDim;
  __shared__ float scratch[kWarps];
  __shared__ float head[kHeadDim];

  const float x = load_bf16(&keys[base + dim]);
  const float mean = block_all_sum(x * x, scratch) / static_cast<float>(kHeadDim);
  // The reference's `torch.rsqrt` on the host is `1 / sqrt` to the last bit, so
  // the correctly rounded pair is what reproduces it.
  const float inv_rms = 1.0f / sqrtf(mean + rms_eps);
  head[dim] = norm_elem(x, inv_rms, load_bf16(&k_norm_weight[dim]));
  __syncthreads();
  if (dim < kRotaryDim / 2) {
    rope_pair(head[dim], head[dim + kRotaryDim / 2], positions[token],
              inv_freq(dim));
  }
  __syncthreads();
  normed_keys[base + dim] = __float2bfloat16(head[dim]);
}

// One CTA per (query row, head): prepare the head's query, score the whole key
// row, softmax it, write the weights and reduce the gated context.
__global__ void qwen4_qsa_attn_kernel(
    const __nv_bfloat16* __restrict__ queries,       // [query_rows, kQueryWidth]
    const __nv_bfloat16* __restrict__ normed_keys,   // [tokens, kKvWidth]
    const __nv_bfloat16* __restrict__ values,        // [tokens, kKvWidth]
    const unsigned char* __restrict__ mask,          // [query_rows, tokens]
    const int* __restrict__ rows,                    // [query_rows]
    const int* __restrict__ positions,               // [tokens]
    const __nv_bfloat16* __restrict__ q_norm_weight, // [kHeadDim]
    __nv_bfloat16* __restrict__ attn_weights,        // [kHeads, rows, tokens]
    __nv_bfloat16* __restrict__ attn_out,            // [query_rows, kOutWidth]
    int query_rows,
    int tokens,
    float rms_eps) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid % WARP_SIZE;
  const int warp = tid / WARP_SIZE;
  __shared__ float scratch[kWarps];
  __shared__ float query[kHeadDim];
  // One row of scores, then of exps, then of rounded weights: each pass reads
  // what the previous one left and a barrier separates them.
  extern __shared__ float row_scores[];

  // `rows[row]` is the token this query row belongs to, and so the position its
  // query is rotated at. Contract: 0 <= rows[row] < tokens — clamped, so a
  // broken caller reads a wrong row rather than memory it does not own.
  const int raw_token = rows[row];
  const int token =
      raw_token < 0 ? 0 : (raw_token >= tokens ? tokens - 1 : raw_token);
  const int pos = positions[token];
  const int64_t query_base =
      static_cast<int64_t>(row) * kQueryWidth + head * (2 * kHeadDim);

  const float x = load_bf16(&queries[query_base + tid]);
  const float mean = block_all_sum(x * x, scratch) / static_cast<float>(kHeadDim);
  const float inv_rms = 1.0f / sqrtf(mean + rms_eps);
  query[tid] = norm_elem(x, inv_rms, load_bf16(&q_norm_weight[tid]));
  __syncthreads();
  if (tid < kRotaryDim / 2) {
    rope_pair(query[tid], query[tid + kRotaryDim / 2], pos, inv_freq(tid));
  }
  __syncthreads();

  // GQA: every head reads the KV head its group maps to, `repeat_kv`'s own
  // order, so head h takes KV head h / group.
  const int kv = head / kGroup;
  const __nv_bfloat16* key_base = normed_keys + kv * kHeadDim;
  const unsigned char* mask_row = mask + static_cast<int64_t>(row) * tokens;

  float qreg[kDimsPerLane];
#pragma unroll
  for (int i = 0; i < kDimsPerLane; ++i) {
    qreg[i] = query[lane * kDimsPerLane + i];
  }

  // Scores: warp `w` owns keys w, w + kWarps, ..., and one lane owns eight dims
  // of each, so a warp reads a key row as eight consecutive bf16 per lane.
  float warp_max = kHiddenScore;
  for (int s = warp; s < tokens; s += kWarps) {
    const __nv_bfloat16* key =
        key_base + static_cast<int64_t>(s) * kKvWidth + lane * kDimsPerLane;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < kDimsPerLane; ++i) {
      acc += qreg[i] * load_bf16(&key[i]);
    }
    acc = warp_all_sum(acc);
    if (lane == 0) {
      // The reference's bf16 GEMM rounds, then scales: the rounding comes first
      // and the scale is exact.
      const float scaled = round_bf16(acc) * kScale;
      const float score = mask_row[s] != 0 ? scaled : kHiddenScore;
      row_scores[s] = score;
      warp_max = fmaxf(warp_max, score);
    }
  }
  __syncthreads();
  if (lane == 0) {
    scratch[warp] = warp_max;
  }
  __syncthreads();
  float max_score = kHiddenScore;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    max_score = fmaxf(max_score, scratch[w]);
  }

  // Softmax in float32 over the rounded scores, as the reference upcasts them:
  // subtract the row max, exponentiate, divide by the sum — a division, not a
  // reciprocal multiply, because that is what its epilogue does — then round the
  // weight to bf16 once, which is the value the context GEMM reads.
  float local_sum = 0.0f;
  for (int s = tid; s < tokens; s += kThreads) {
    const float exponent = expf(row_scores[s] - max_score);
    row_scores[s] = exponent;
    local_sum += exponent;
  }
  const float sum = block_all_sum(local_sum, scratch);
  const int64_t weight_base =
      (static_cast<int64_t>(head) * query_rows + row) * tokens;
  for (int s = tid; s < tokens; s += kThreads) {
    const float weight = round_bf16(row_scores[s] / sum);
    row_scores[s] = weight;
    attn_weights[weight_base + s] = __float2bfloat16(weight);
  }
  __syncthreads();

  // Context: one thread owns one dim and folds the row's rounded weights over
  // that dim of every value row, in key order.
  const __nv_bfloat16* value_col = values + kv * kHeadDim + tid;
  float context = 0.0f;
  for (int s = 0; s < tokens; ++s) {
    context += row_scores[s] *
               load_bf16(&value_col[static_cast<int64_t>(s) * kKvWidth]);
  }

  // The output gate is the second half of this head's slice of the fused
  // projection row, and it lands after the context reshape and before `o_proj`.
  // Both factors are bf16, so their product is exact in float32 and the single
  // cast below is the reference's own.
  const float gate = load_bf16(&queries[query_base + kHeadDim + tid]);
  const float sigmoid = round_bf16(1.0f / (1.0f + expf(-gate)));
  attn_out[static_cast<int64_t>(row) * kOutWidth + head * kHeadDim + tid] =
      __float2bfloat16(round_bf16(context) * sigmoid);
}

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

}  // namespace

extern "C" {

// The QSA sparse-attention forward over one layer's captured activations: the
// key pass, then one CTA per (query row, head) for the scores, the weights and
// the gated context.
//
// `queries` rows are the fused `q_proj` output, `kQueryWidth` wide, each head's
// 256 query dims followed by that head's 256 gate dims. `rows` gives the token
// each evaluated query row belongs to and `positions` the position of every
// token, so the query rows may be any subset of the key array in any order.
// `mask` is the caller's visibility, one byte per (query row, token).
//
// Writes every element of `attn_weights` ([kHeads, query_rows, tokens], the
// reference's head-major eager layout) and of `attn_out` ([query_rows,
// kOutWidth], the input to `o_proj`). `normed_keys` is the key pass's workspace,
// `tokens * kKvWidth` bf16; it is one entry point because every query head of a
// KV head reads all of it and no row depends on the query side.
//
// Refuses a `tokens` whose score row would not fit the shared-memory budget —
// that shape needs the streaming softmax, not a longer row.
CUresult qwen4_qsa_attn_forward_cuda(
    const __nv_bfloat16* queries,
    const __nv_bfloat16* keys,
    const __nv_bfloat16* values,
    const unsigned char* mask,
    const int* rows,
    const int* positions,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    __nv_bfloat16* attn_weights,
    __nv_bfloat16* attn_out,
    __nv_bfloat16* normed_keys,
    int query_rows,
    int tokens,
    int rotary_dim,
    float rope_theta,
    float rms_eps,
    cudaStream_t stream) {
  if (queries == nullptr || keys == nullptr || values == nullptr ||
      mask == nullptr || rows == nullptr || positions == nullptr ||
      q_norm_weight == nullptr || k_norm_weight == nullptr ||
      attn_weights == nullptr || attn_out == nullptr || normed_keys == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (query_rows < 0 || tokens < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rotary_dim != kRotaryDim || rope_theta != kRopeTheta) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // `!(rms_eps >= 0)` refuses NaN as well, which would otherwise poison every
  // norm silently rather than fail loudly.
  if (!(rms_eps >= 0.0f)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (query_rows == 0 || tokens == 0) {
    return CUDA_SUCCESS;
  }
  if (tokens > kMaxTokens) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int64_t key_rows = static_cast<int64_t>(tokens) * kKvHeads;
  qwen4_qsa_norm_rope_keys_kernel<<<static_cast<unsigned>(key_rows),
                                       kThreads, 0, stream>>>(
      keys, k_norm_weight, positions, normed_keys, rms_eps);
  const CUresult key_status = consume_last_cuda_error();
  if (key_status != CUDA_SUCCESS) {
    return key_status;
  }
  const dim3 grid(static_cast<unsigned>(query_rows), kHeads);
  const size_t shared = static_cast<size_t>(tokens) * sizeof(float);
  qwen4_qsa_attn_kernel<<<grid, kThreads, shared, stream>>>(
      queries, normed_keys, values, mask, rows, positions, q_norm_weight,
      attn_weights, attn_out, query_rows, tokens, rms_eps);
  return consume_last_cuda_error();
}

}  // extern "C"
