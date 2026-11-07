#pragma once

#include "../common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

// The `(1 + w)` RMSNorm and partial-RoPE rounding points shared by the QSA
// kernels. The indexer and the sparse-attention forward reproduce the same
// reference rounding at the same points, so they use one copy.
__device__ __forceinline__ float qsa_warp_all_sum(float value) {
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, mask);
  }
  return value;
}

__device__ __forceinline__ float qsa_round_bf16(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

// One `(1 + w)` RMSNorm element, rounded to bf16 exactly once: the reference
// normalizes in float32, scales by `1 + w` in float32, and then casts to the
// activation dtype.
__device__ __forceinline__ float qsa_norm_elem(float x, float inv_rms,
                                               float weight) {
  return qsa_round_bf16(x * inv_rms * (1.0f + weight));
}

// The shared attention table stores `[max_pos, rotary_dim]`, position-major,
// with the half split duplicated, so dim and dim + rotary_dim/2 hold the same
// cos/sin.
__device__ __forceinline__ float qsa_table_at(const __nv_bfloat16* table,
                                              int64_t pos, int dim,
                                              int rotary_dim) {
  return __bfloat162float(table[pos * rotary_dim + dim]);
}

// Partial RoPE over the pair (d, d + rotary_dim/2), for d < rotary_dim/2. The
// table values are bf16, and the reference rotates in bf16, so each product and
// their sum round as well.
__device__ __forceinline__ void qsa_rope_pair(float& lo, float& hi, float cos_v,
                                              float sin_v) {
  const float lo_rot = qsa_round_bf16(lo * cos_v) + qsa_round_bf16(-hi * sin_v);
  const float hi_rot = qsa_round_bf16(hi * cos_v) + qsa_round_bf16(lo * sin_v);
  lo = qsa_round_bf16(lo_rot);
  hi = qsa_round_bf16(hi_rot);
}
