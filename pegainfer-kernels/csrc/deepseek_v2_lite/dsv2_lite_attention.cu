#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cfloat>
#include <math_constants.h>

namespace {

constexpr int kNormThreads = 256;
// Squares live in dynamic shared memory; stay under the 48 KiB default.
constexpr int kMaxNormDim = 8192;
constexpr int kAttentionThreads = 256;
constexpr int kMaxRopeDim = 128;
constexpr int kMaxDecodeSeqLen = 4096;
constexpr int kMaxQueryHeadDim = 256;

__device__ __forceinline__ float bf16_round(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

__device__ __forceinline__ float yarn_get_mscale(float scale, float mscale) {
  if (scale <= 1.0f) return 1.0f;
  return 0.1f * mscale * logf(scale) + 1.0f;
}

__device__ __forceinline__ float yarn_find_correction_dim(
    float num_rotations,
    int dim,
    float base,
    float factor,
    int original_max_position_embeddings) {
  return static_cast<float>(dim) *
         logf(static_cast<float>(original_max_position_embeddings) /
              (num_rotations * 2.0f * CUDART_PI_F)) /
         (2.0f * logf(base));
}

__device__ __forceinline__ float rope_inv_freq(
    int pair,
    int rope_dim,
    float base,
    int has_rope_scaling,
    float factor,
    float beta_fast,
    float beta_slow,
    int original_max_position_embeddings) {
  float freq_extra = 1.0f / powf(base, static_cast<float>(2 * pair) / rope_dim);
  if (!has_rope_scaling) return freq_extra;

  float freq_inter = freq_extra / factor;
  float low =
      floorf(fmaxf(0.0f, yarn_find_correction_dim(beta_fast, rope_dim, base, factor,
                                                  original_max_position_embeddings)));
  float high = ceilf(fminf(static_cast<float>(rope_dim - 1),
                           yarn_find_correction_dim(beta_slow, rope_dim, base, factor,
                                                    original_max_position_embeddings)));
  float ramp;
  if (fabsf(high - low) < FLT_EPSILON) {
    ramp = static_cast<float>(pair) <= low ? 0.0f : 1.0f;
  } else {
    ramp = fminf(1.0f, fmaxf(0.0f, (static_cast<float>(pair) - low) / (high - low)));
  }
  float inv_freq_mask = 1.0f - ramp;
  return freq_inter * (1.0f - inv_freq_mask) + freq_extra * inv_freq_mask;
}

__device__ __forceinline__ float rope_mscale(
    int has_rope_scaling,
    float factor,
    float mscale,
    float mscale_all_dim) {
  if (!has_rope_scaling) return 1.0f;
  return yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim);
}

__device__ __forceinline__ void apply_rope_pair(
    float x0,
    float x1,
    int pair,
    int pos,
    int rope_dim,
    float rope_theta,
    int has_rope_scaling,
    float rope_factor,
    float rope_mscale_value,
    float rope_beta_fast,
    float rope_beta_slow,
    int rope_original_max_position_embeddings,
    float *out0,
    float *out1) {
  float angle = static_cast<float>(pos) *
                rope_inv_freq(pair, rope_dim, rope_theta, has_rope_scaling, rope_factor,
                              rope_beta_fast, rope_beta_slow,
                              rope_original_max_position_embeddings);
  float cos_value = bf16_round(cosf(angle) * rope_mscale_value);
  float sin_value = bf16_round(sinf(angle) * rope_mscale_value);
  float x0_cos = bf16_round(x0 * cos_value);
  float neg_x1_sin = bf16_round(-x1 * sin_value);
  float x1_cos = bf16_round(x1 * cos_value);
  float x0_sin = bf16_round(x0 * sin_value);
  *out0 = bf16_round(x0_cos + neg_x1_sin);
  *out1 = bf16_round(x1_cos + x0_sin);
}

__global__ void kv_norm_kernel(
    const __nv_bfloat16 *__restrict__ kv_a,
    const __nv_bfloat16 *__restrict__ norm_weight,
    __nv_bfloat16 *__restrict__ compressed,
    int kv_lora_rank,
    int kv_a_rows,
    int seq_len,
    float eps) {
  extern __shared__ float squares[];
  int token = blockIdx.x;
  int tid = threadIdx.x;
  if (token >= seq_len) return;

  const __nv_bfloat16 *row = kv_a + static_cast<size_t>(token) * kv_a_rows;
  for (int dim = tid; dim < kv_lora_rank; dim += blockDim.x) {
    float value = __bfloat162float(row[dim]);
    squares[dim] = __fmul_rn(value, value);
  }
  __syncthreads();

  __shared__ float inv_rms;
  if (tid == 0) {
    float total = 0.0f;
    for (int dim = 0; dim < kv_lora_rank; ++dim) total = __fadd_rn(total, squares[dim]);
    float mean = __fdiv_rn(total, static_cast<float>(kv_lora_rank));
    inv_rms = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
  }
  __syncthreads();

  for (int dim = tid; dim < kv_lora_rank; dim += blockDim.x) {
    float value = __bfloat162float(row[dim]);
    float normalized = bf16_round(__fmul_rn(value, inv_rms));
    float scaled = __fmul_rn(normalized, __bfloat162float(norm_weight[dim]));
    compressed[static_cast<size_t>(token) * kv_lora_rank + dim] = __float2bfloat16(scaled);
  }
}

__global__ void decode_attention_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ kv_a,
    const __nv_bfloat16 *__restrict__ kv_b,
    __nv_bfloat16 *__restrict__ key_cache,
    __nv_bfloat16 *__restrict__ value_cache,
    const int *__restrict__ slot_ids,
    const int *__restrict__ positions,
    __nv_bfloat16 *__restrict__ out,
    int num_heads,
    int qk_nope_head_dim,
    int qk_rope_head_dim,
    int v_head_dim,
    int kv_lora_rank,
    int kv_a_rows,
    int kv_b_rows,
    int num_slots,
    int max_seq_len,
    float rope_theta,
    float rope_factor,
    float rope_mscale_arg,
    float rope_mscale_all_dim,
    float rope_beta_fast,
    float rope_beta_slow,
    int rope_original_max_position_embeddings,
    int has_rope_scaling) {
  int head = blockIdx.x;
  int row = blockIdx.y;
  int tid = threadIdx.x;
  if (head >= num_heads) return;

  int q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
  int q_rows = num_heads * q_head_dim;
  int kv_b_stride = qk_nope_head_dim + v_head_dim;
  int out_rows = num_heads * v_head_dim;
  int slot = slot_ids[row];
  int position = positions[row];
  if (slot < 0 || slot >= num_slots || position < 0 || position >= max_seq_len ||
      position >= kMaxDecodeSeqLen) {
    // Last line of defense; the host rejects bad rows before launch.
    for (int dim = tid; dim < v_head_dim; dim += blockDim.x) {
      out[static_cast<size_t>(row) * out_rows + head * v_head_dim + dim] =
          __float2bfloat16(CUDART_NAN_F);
    }
    return;
  }
  int kv_len = position + 1;
  float rope_scale = rope_mscale(has_rope_scaling, rope_factor, rope_mscale_arg,
                                 rope_mscale_all_dim);

  const __nv_bfloat16 *row_q = q + static_cast<size_t>(row) * q_rows;
  const __nv_bfloat16 *row_kv_a = kv_a + static_cast<size_t>(row) * kv_a_rows;
  const __nv_bfloat16 *row_kv_b = kv_b + static_cast<size_t>(row) * kv_b_rows;
  size_t key_token_stride = static_cast<size_t>(num_heads) * q_head_dim;
  size_t value_token_stride = static_cast<size_t>(num_heads) * v_head_dim;
  __nv_bfloat16 *slot_keys =
      key_cache + static_cast<size_t>(slot) * max_seq_len * key_token_stride;
  __nv_bfloat16 *slot_values =
      value_cache + static_cast<size_t>(slot) * max_seq_len * value_token_stride;

  __nv_bfloat16 *new_key = slot_keys + position * key_token_stride + head * q_head_dim;
  __nv_bfloat16 *new_value =
      slot_values + position * value_token_stride + head * v_head_dim;
  int kv_b_base = head * kv_b_stride;
  for (int dim = tid; dim < qk_nope_head_dim; dim += blockDim.x) {
    new_key[dim] = row_kv_b[kv_b_base + dim];
  }
  for (int dim = tid; dim < v_head_dim; dim += blockDim.x) {
    new_value[dim] = row_kv_b[kv_b_base + qk_nope_head_dim + dim];
  }
  for (int pair = tid; pair < qk_rope_head_dim / 2; pair += blockDim.x) {
    float k0 = __bfloat162float(row_kv_a[kv_lora_rank + 2 * pair]);
    float k1 = __bfloat162float(row_kv_a[kv_lora_rank + 2 * pair + 1]);
    float k_out0;
    float k_out1;
    apply_rope_pair(k0, k1, pair, position, qk_rope_head_dim, rope_theta, has_rope_scaling,
                    rope_factor, rope_scale, rope_beta_fast, rope_beta_slow,
                    rope_original_max_position_embeddings, &k_out0, &k_out1);
    // apply_rope_pair rounds to bf16, so these stores are exact.
    new_key[qk_nope_head_dim + pair] = __float2bfloat16(k_out0);
    new_key[qk_nope_head_dim + pair + qk_rope_head_dim / 2] = __float2bfloat16(k_out1);
  }

  __shared__ float scores[kMaxDecodeSeqLen];
  __shared__ float denom;
  __shared__ float max_score;
  __shared__ float query[kMaxQueryHeadDim];

  for (int dim = tid; dim < qk_nope_head_dim; dim += blockDim.x) {
    query[dim] = __bfloat162float(row_q[head * q_head_dim + dim]);
  }
  for (int pair = tid; pair < qk_rope_head_dim / 2; pair += blockDim.x) {
    int q_base = head * q_head_dim + qk_nope_head_dim;
    float q0 = __bfloat162float(row_q[q_base + 2 * pair]);
    float q1 = __bfloat162float(row_q[q_base + 2 * pair + 1]);
    float q_out0;
    float q_out1;
    apply_rope_pair(q0, q1, pair, position, qk_rope_head_dim, rope_theta, has_rope_scaling,
                    rope_factor, rope_scale, rope_beta_fast, rope_beta_slow,
                    rope_original_max_position_embeddings, &q_out0, &q_out1);
    query[qk_nope_head_dim + pair] = q_out0;
    query[qk_nope_head_dim + pair + qk_rope_head_dim / 2] = q_out1;
  }
  // Orders this block's K/V append before the score and value loops read it.
  __syncthreads();

  float local_max = -CUDART_INF_F;
  float scale = rsqrtf(static_cast<float>(q_head_dim));
  if (has_rope_scaling && rope_mscale_all_dim > 0.0f) {
    float mscale = yarn_get_mscale(rope_factor, rope_mscale_all_dim);
    scale *= mscale * mscale;
  }

  for (int pos = tid; pos < kv_len; pos += blockDim.x) {
    const __nv_bfloat16 *key = slot_keys + pos * key_token_stride + head * q_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < q_head_dim; ++dim) {
      dot = __fadd_rn(dot, __fmul_rn(query[dim], __bfloat162float(key[dim])));
    }
    float score = bf16_round(bf16_round(dot) * scale);
    scores[pos] = score;
    local_max = fmaxf(local_max, score);
  }
  local_max = warp_reduce_max(local_max);

  __shared__ float warp_max[8];
  int lane = tid % WARP_SIZE;
  int warp = tid / WARP_SIZE;
  if (lane == 0) warp_max[warp] = local_max;
  __syncthreads();
  if (tid == 0) {
    float value = -CUDART_INF_F;
    int warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    for (int i = 0; i < warps; ++i) value = fmaxf(value, warp_max[i]);
    max_score = value;
  }
  __syncthreads();

  for (int pos = tid; pos < kv_len; pos += blockDim.x) {
    scores[pos] = expf(scores[pos] - max_score);
  }
  __syncthreads();
  if (tid == 0) {
    // Sequential sum in position order, as the host softmax does.
    float value = 0.0f;
    for (int pos = 0; pos < kv_len; ++pos) value = __fadd_rn(value, scores[pos]);
    denom = value;
  }
  __syncthreads();

  for (int dim = tid; dim < v_head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int pos = 0; pos < kv_len; ++pos) {
      float prob = denom > 0.0f ? scores[pos] / denom : 0.0f;
      prob = bf16_round(prob);
      float value = __bfloat162float(slot_values[pos * value_token_stride +
                                                 head * v_head_dim + dim]);
      acc = __fadd_rn(acc, __fmul_rn(prob, value));
    }
    out[static_cast<size_t>(row) * out_rows + head * v_head_dim + dim] = __float2bfloat16(acc);
  }
}

CUresult consume_last_cuda_error() {
  cudaError_t err = cudaGetLastError();
  return map_cuda_error(err);
}

}  // namespace

extern "C" {

CUresult dsv2_lite_kv_norm_cuda(
    const __nv_bfloat16 *kv_a,
    const __nv_bfloat16 *norm_weight,
    __nv_bfloat16 *compressed,
    int kv_lora_rank,
    int kv_a_rows,
    int seq_len,
    float eps,
    cudaStream_t stream) {
  if (kv_a == nullptr || norm_weight == nullptr || compressed == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (kv_lora_rank <= 0 || kv_lora_rank > kMaxNormDim || kv_a_rows < kv_lora_rank ||
      seq_len <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  cudaGetLastError();
  size_t shared_bytes = static_cast<size_t>(kv_lora_rank) * sizeof(float);
  kv_norm_kernel<<<seq_len, kNormThreads, shared_bytes, stream>>>(
      kv_a, norm_weight, compressed, kv_lora_rank, kv_a_rows, seq_len, eps);
  return consume_last_cuda_error();
}

CUresult dsv2_lite_decode_attention_cuda(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *kv_a,
    const __nv_bfloat16 *kv_b,
    __nv_bfloat16 *key_cache,
    __nv_bfloat16 *value_cache,
    const int *slot_ids,
    const int *positions,
    __nv_bfloat16 *out,
    int batch,
    int num_heads,
    int qk_nope_head_dim,
    int qk_rope_head_dim,
    int v_head_dim,
    int kv_lora_rank,
    int kv_a_rows,
    int kv_b_rows,
    int num_slots,
    int max_seq_len,
    float rope_theta,
    float rope_factor,
    float rope_mscale,
    float rope_mscale_all_dim,
    float rope_beta_fast,
    float rope_beta_slow,
    int rope_original_max_position_embeddings,
    int has_rope_scaling,
    cudaStream_t stream) {
  if (q == nullptr || kv_a == nullptr || kv_b == nullptr || key_cache == nullptr ||
      value_cache == nullptr || slot_ids == nullptr || positions == nullptr || out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (batch <= 0 || num_heads <= 0 || qk_nope_head_dim <= 0 || qk_rope_head_dim <= 0 ||
      (qk_rope_head_dim % 2) != 0 || qk_rope_head_dim > kMaxRopeDim ||
      qk_nope_head_dim + qk_rope_head_dim > kMaxQueryHeadDim || v_head_dim <= 0 ||
      kv_lora_rank <= 0 || num_slots <= 0 || max_seq_len <= 0 ||
      max_seq_len > kMaxDecodeSeqLen || kv_a_rows < kv_lora_rank + qk_rope_head_dim ||
      kv_b_rows < num_heads * (qk_nope_head_dim + v_head_dim)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  cudaGetLastError();
  dim3 grid(num_heads, batch);
  decode_attention_kernel<<<grid, kAttentionThreads, 0, stream>>>(
      q, kv_a, kv_b, key_cache, value_cache, slot_ids, positions, out, num_heads,
      qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank, kv_a_rows, kv_b_rows,
      num_slots, max_seq_len, rope_theta, rope_factor, rope_mscale, rope_mscale_all_dim,
      rope_beta_fast, rope_beta_slow, rope_original_max_position_embeddings,
      has_rope_scaling);
  return consume_last_cuda_error();
}

}  // extern "C"
