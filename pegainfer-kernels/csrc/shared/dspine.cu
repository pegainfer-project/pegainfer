#include "common.cuh"
#include "ffi_guard.cuh"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace {

constexpr int kThreads = 256;
constexpr int kTopkChunk = 64;

__device__ __forceinline__ float bf2f(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

// Warp-wide all-reduce: every lane receives the sum.
__device__ __forceinline__ float warp_sum_all(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ float block_sum(float value, float* red) {
  const int tid = threadIdx.x;
  red[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red[tid] += red[tid + stride];
    }
    __syncthreads();
  }
  return red[0];
}

// raw -> normalized = raw / (rms + eps), rms = sqrt(max(mean(raw^2), FLT_MIN)).
__global__ void prepare_read_kernel(const __nv_bfloat16* __restrict__ raw,
                                    __nv_bfloat16* __restrict__ normalized,
                                    float* __restrict__ rms, int hidden,
                                    float eps) {
  const int row = blockIdx.x;
  const int64_t base = static_cast<int64_t>(row) * hidden;
  __shared__ float red[kThreads];
  const bool vector = (hidden & 7) == 0;
  float partial = 0.0f;
  if (vector) {
    const int vectors = hidden / 8;
    for (int v = threadIdx.x; v < vectors; v += blockDim.x) {
      const uint4 r = *reinterpret_cast<const uint4*>(raw + base + v * 8);
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const float lo = bf2f(__low2bfloat16(p[t]));
        const float hi = bf2f(__high2bfloat16(p[t]));
        partial += lo * lo + hi * hi;
      }
    }
  } else {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      const float x = bf2f(raw[base + i]);
      partial += x * x;
    }
  }
  const float mean = block_sum(partial, red) / static_cast<float>(hidden);
  __shared__ float row_rms;
  if (threadIdx.x == 0) {
    row_rms = sqrtf(fmaxf(mean, 1.1754943508222875e-38f));
    rms[row] = row_rms;
  }
  __syncthreads();
  const float denom = row_rms + eps;
  if (vector) {
    const int vectors = hidden / 8;
    for (int v = threadIdx.x; v < vectors; v += blockDim.x) {
      const uint4 r = *reinterpret_cast<const uint4*>(raw + base + v * 8);
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&r);
      uint4 out;
      __nv_bfloat162* q = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        q[t] = __floats2bfloat162_rn(bf2f(__low2bfloat16(p[t])) / denom,
                                     bf2f(__high2bfloat16(p[t])) / denom);
      }
      *reinterpret_cast<uint4*>(normalized + base + v * 8) = out;
    }
  } else {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      normalized[base + i] = __float2bfloat16(bf2f(raw[base + i]) / denom);
    }
  }
}

// codes = sqrt(C) * proj / (||proj|| + eps); block row 0 is the anchor codebook row.
__global__ void normalize_codes_kernel(const __nv_bfloat16* __restrict__ projected,
                                       const __nv_bfloat16* __restrict__ codebook,
                                       const unsigned int* __restrict__ input_ids,
                                       __nv_bfloat16* __restrict__ codes,
                                       int code_size, int block, float eps) {
  const int row = blockIdx.x;
  const int64_t base = static_cast<int64_t>(row) * code_size;
  const float scale = sqrtf(static_cast<float>(code_size));
  if (row % block == 0) {
    const int64_t anchor = static_cast<int64_t>(input_ids[row]) * code_size;
    for (int i = threadIdx.x; i < code_size; i += blockDim.x) {
      codes[base + i] = __float2bfloat16(scale * bf2f(codebook[anchor + i]));
    }
    return;
  }
  __shared__ float red[kThreads];
  float partial = 0.0f;
  for (int i = threadIdx.x; i < code_size; i += blockDim.x) {
    const float x = bf2f(projected[base + i]);
    partial += x * x;
  }
  const float norm = sqrtf(block_sum(partial, red));
  const float factor = scale / (norm + eps);
  for (int i = threadIdx.x; i < code_size; i += blockDim.x) {
    codes[base + i] =
        __float2bfloat16(bf2f(projected[base + i]) * factor);
  }
}

// message[r] = 0 on anchors, else sigmoid(Q[r] + S[r-1] + bias) * V[r-1].
__global__ void feature_message_kernel(const __nv_bfloat16* __restrict__ packed,
                                       const __nv_bfloat16* __restrict__ gate_bias,
                                       __nv_bfloat16* __restrict__ message,
                                       int rank, int block) {
  const int row = blockIdx.x;
  const bool anchor = (row % block) == 0;
  const int predecessor = anchor ? row : row - 1;
  const int64_t q_base = static_cast<int64_t>(row) * 3 * rank;
  const int64_t p_base = static_cast<int64_t>(predecessor) * 3 * rank;
  const int64_t out_base = static_cast<int64_t>(row) * rank;
  for (int i = threadIdx.x; i < rank; i += blockDim.x) {
    if (anchor) {
      message[out_base + i] = __float2bfloat16(0.0f);
      continue;
    }
    const float q = bf2f(packed[q_base + i]);
    const float s = bf2f(packed[p_base + rank + i]);
    const float v = bf2f(packed[p_base + 2 * rank + i]);
    const float gate = 1.0f / (1.0f + expf(-(q + s + bf2f(gate_bias[i]))));
    message[out_base + i] = __float2bfloat16(gate * v);
  }
}

// out = raw on anchors, else raw + rms * delta. One 16 B vector per iteration
// when the row width allows it; the scalar tail keeps odd widths exact.
__device__ __forceinline__ __nv_bfloat162 finish_pair(
    const __nv_bfloat162 r, const __nv_bfloat162 d, float scale,
    bool add_delta) {
  const float add = add_delta ? scale : 0.0f;
  return __floats2bfloat162_rn(
      bf2f(__low2bfloat16(r)) + add * bf2f(__low2bfloat16(d)),
      bf2f(__high2bfloat16(r)) + add * bf2f(__high2bfloat16(d)));
}

__device__ __forceinline__ uint4 finish_vec8(const uint4& r, const uint4& d,
                                             float scale, bool add_delta,
                                             float* sum8) {
  const __nv_bfloat162* rp = reinterpret_cast<const __nv_bfloat162*>(&r);
  const __nv_bfloat162* dp = reinterpret_cast<const __nv_bfloat162*>(&d);
  __nv_bfloat162 o[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    o[t] = finish_pair(rp[t], dp[t], scale, add_delta);
    const float lo = bf2f(__low2bfloat16(o[t]));
    const float hi = bf2f(__high2bfloat16(o[t]));
    *sum8 += lo * lo + hi * hi;
  }
  uint4 packed;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    op[t] = o[t];
  }
  return packed;
}

__global__ void feature_finish_kernel(const __nv_bfloat16* __restrict__ raw,
                                      const __nv_bfloat16* __restrict__ delta,
                                      const float* __restrict__ rms,
                                      __nv_bfloat16* __restrict__ out,
                                      int64_t total, int hidden, int block) {
  const bool vector = (hidden & 7) == 0;
  if (vector) {
    const int64_t vectors = total / 8;
    const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         v < vectors; v += step) {
      const int64_t index = v * 8;
      const int row = static_cast<int>(index / hidden);
      const float scale = rms[row];
      const bool add_delta = (row % block) != 0;
      const uint4 r = *reinterpret_cast<const uint4*>(raw + index);
      const uint4 d = *reinterpret_cast<const uint4*>(delta + index);
      float unused = 0.0f;
      *reinterpret_cast<uint4*>(out + index) =
          finish_vec8(r, d, scale, add_delta, &unused);
    }
    return;
  }
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += step) {
    const int row = static_cast<int>(index / hidden);
    float value = bf2f(raw[index]);
    if (row % block != 0) {
      value += rms[row] * bf2f(delta[index]);
    }
    out[index] = __float2bfloat16(value);
  }
}

// Fusion of feature_finish and the next layer's input RMSNorm. Both outputs are
// produced from the same rounded bf16 row, matching the two-kernel path.
__global__ void feature_finish_norm_kernel(
    const __nv_bfloat16* __restrict__ raw,
    const __nv_bfloat16* __restrict__ delta, const float* __restrict__ rms,
    __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ normed,
    const __nv_bfloat16* __restrict__ weight, int hidden, int block, float eps) {
  const int row = blockIdx.x;
  extern __shared__ unsigned char norm_smem[];
  __nv_bfloat16* rowbuf = reinterpret_cast<__nv_bfloat16*>(norm_smem);
  const int nwarps = blockDim.x / 32;
  float* warp_sums = reinterpret_cast<float*>(rowbuf + hidden);
  const int64_t base = static_cast<int64_t>(row) * hidden;
  const float scale = rms[row];
  const bool add_delta = (row % block) != 0;
  float sum = 0.0f;
  const bool vector = (hidden & 7) == 0;
  if (vector) {
    const int vectors = hidden / 8;
    for (int v = threadIdx.x; v < vectors; v += blockDim.x) {
      const int i = v * 8;
      const uint4 r = *reinterpret_cast<const uint4*>(raw + base + i);
      const uint4 d = *reinterpret_cast<const uint4*>(delta + base + i);
      float squares = 0.0f;
      const uint4 o = finish_vec8(r, d, scale, add_delta, &squares);
      sum += squares;
      *reinterpret_cast<uint4*>(rowbuf + i) = o;
      *reinterpret_cast<uint4*>(out + base + i) = o;
    }
  } else {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      float value = bf2f(raw[base + i]);
      if (add_delta) {
        value += scale * bf2f(delta[base + i]);
      }
      const __nv_bfloat16 rounded = __float2bfloat16(value);
      rowbuf[i] = rounded;
      out[base + i] = rounded;
      const float as_float = bf2f(rounded);
      sum += as_float * as_float;
    }
  }
  sum = warp_sum_all(sum);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int w = 0; w < nwarps; ++w) {
      total += warp_sums[w];
    }
    warp_sums[0] = rsqrtf(total / static_cast<float>(hidden) + eps);
  }
  __syncthreads();
  const float rcp = warp_sums[0];
  if (vector) {
    const int vectors = hidden / 8;
    for (int v = threadIdx.x; v < vectors; v += blockDim.x) {
      const int i = v * 8;
      const uint4 r = *reinterpret_cast<const uint4*>(rowbuf + i);
      const uint4 w = *reinterpret_cast<const uint4*>(weight + i);
      const __nv_bfloat162* rp = reinterpret_cast<const __nv_bfloat162*>(&r);
      const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&w);
      __nv_bfloat162 o[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        o[t] = __floats2bfloat162_rn(
            bf2f(__low2bfloat16(rp[t])) * rcp * bf2f(__low2bfloat16(wp[t])),
            bf2f(__high2bfloat16(rp[t])) * rcp * bf2f(__high2bfloat16(wp[t])));
      }
      uint4 packed;
      __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        op[t] = o[t];
      }
      *reinterpret_cast<uint4*>(normed + base + i) =
          *reinterpret_cast<const uint4*>(&packed);
    }
  } else {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      normed[base + i] = __float2bfloat16(bf2f(rowbuf[i]) * rcp *
                                          bf2f(weight[i]));
    }
  }
}

// Walk the lane table from the anchor, emitting [anchor, 15 drafts] per request.
__global__ void serial_chain_kernel(const int* __restrict__ idx,
                                    const int* __restrict__ candidates,
                                    const unsigned int* __restrict__ input_ids,
                                    unsigned int* __restrict__ out, int block,
                                    int k) {
  const int b = blockIdx.x;
  const int n0 = b * (block - 1);
  int lane = idx[static_cast<int64_t>(n0) * k];
  out[static_cast<int64_t>(b) * block] = input_ids[static_cast<int64_t>(b) * block];
  for (int i = 0; i < block - 1; ++i) {
    const int64_t row = n0 + i;
    out[static_cast<int64_t>(b) * block + 1 + i] =
        static_cast<unsigned int>(candidates[row * k + lane]);
    if (i + 1 < block - 1) {
      lane = idx[(row + 1) * k + lane];
    }
  }
}

// Warp-wide (max, lowest-id) argmax; lane 0 holds the result.
__device__ __forceinline__ void warp_argmax(float& value, int& index) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_value = __shfl_down_sync(0xffffffffu, value, offset);
    const int other_index = __shfl_down_sync(0xffffffffu, index, offset);
    if (other_value > value ||
        (other_value == value && other_index < index)) {
      value = other_value;
      index = other_index;
    }
  }
}

// Coalesced 16 B chunk scan with warp-level chunk and value selection.
__global__ void topk_kernel(const __nv_bfloat16* __restrict__ logits,
                            int* __restrict__ out, int block, int vocab,
                            int k) {
  const int n = blockIdx.x;
  const int b = n / (block - 1);
  const int i = n % (block - 1);
  const int64_t row = static_cast<int64_t>(b) * block + 1 + i;
  const __nv_bfloat16* src = logits + row * vocab;
  const int nch = (vocab + kTopkChunk - 1) / kTopkChunk;
  const int tid = threadIdx.x;
  const int nwarps = blockDim.x / 32;
  const int warp = tid / 32;
  const int lane = tid % 32;

  extern __shared__ unsigned char smem[];
  float* chunk_max = reinterpret_cast<float*>(smem);
  unsigned char* chunk_used = reinterpret_cast<unsigned char*>(chunk_max + nch);
  unsigned char* cursor = chunk_used + ((nch + 3) & ~3);
  float* lane_value = reinterpret_cast<float*>(cursor);
  cursor += static_cast<size_t>(k) * kTopkChunk * sizeof(float);
  int* lane_id = reinterpret_cast<int*>(cursor);
  cursor += static_cast<size_t>(k) * kTopkChunk * sizeof(int);
  int* selected = reinterpret_cast<int*>(cursor);
  cursor += static_cast<size_t>(k) * sizeof(int);
  float* red = reinterpret_cast<float*>(cursor);
  cursor += static_cast<size_t>(blockDim.x) * sizeof(float);
  int* red_id = reinterpret_cast<int*>(cursor);
  cursor += static_cast<size_t>(blockDim.x) * sizeof(int);
  int* winner = reinterpret_cast<int*>(cursor);

  // Chunks are scanned by 8-lane groups: each lane loads one aligned 16 B
  // vector so a warp reads four full 128 B runs per instruction, and each
  // group carries eight chunks in flight before reducing.
  constexpr int kGroupLanes = kTopkChunk / 8;
  constexpr int kChunksPerGroup = 8;
  constexpr int kGroupsPerWarp = 32 / kGroupLanes;
  const int group = lane / kGroupLanes;
  const int glane = lane % kGroupLanes;
  const int full_chunks = vocab / kTopkChunk;
  const bool aligned = (reinterpret_cast<uintptr_t>(src) & 15u) == 0;
  if (aligned) {
    const int per_warp = kGroupsPerWarp * kChunksPerGroup;
    for (int base = warp * per_warp; base < full_chunks;
         base += nwarps * per_warp) {
      uint4 v[kChunksPerGroup];
      int c[kChunksPerGroup];
#pragma unroll
      for (int q = 0; q < kChunksPerGroup; ++q) {
        c[q] = base + group * kChunksPerGroup + q;
        if (c[q] < full_chunks) {
          v[q] = *reinterpret_cast<const uint4*>(
              src + c[q] * kTopkChunk + glane * 8);
        }
      }
#pragma unroll
      for (int q = 0; q < kChunksPerGroup; ++q) {
        float best = -CUDART_INF_F;
        if (c[q] < full_chunks) {
          const __nv_bfloat162* h =
              reinterpret_cast<const __nv_bfloat162*>(&v[q]);
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            best = fmaxf(best, fmaxf(bf2f(__low2bfloat16(h[t])),
                                     bf2f(__high2bfloat16(h[t]))));
          }
        }
#pragma unroll
        for (int offset = kGroupLanes / 2; offset > 0; offset >>= 1) {
          best = fmaxf(
              best, __shfl_down_sync(0xffffffffu, best, offset, kGroupLanes));
        }
        if (glane == 0 && c[q] < full_chunks) {
          chunk_max[c[q]] = best;
        }
      }
    }
  } else {
    for (int c = tid; c < full_chunks; c += blockDim.x) {
      const int begin = c * kTopkChunk;
      float best = -CUDART_INF_F;
      for (int j = begin; j < begin + kTopkChunk; ++j) {
        best = fmaxf(best, bf2f(src[j]));
      }
      chunk_max[c] = best;
    }
  }
  for (int c = full_chunks + tid; c < nch; c += blockDim.x) {
    float best = -CUDART_INF_F;
    for (int j = c * kTopkChunk; j < vocab; ++j) {
      best = fmaxf(best, bf2f(src[j]));
    }
    chunk_max[c] = best;
  }
  for (int c = tid; c < nch; c += blockDim.x) {
    chunk_used[c] = 0;
  }
  __syncthreads();

  for (int slot = 0; slot < k; ++slot) {
    float best = -CUDART_INF_F;
    int best_index = INT_MAX;
    for (int c = tid; c < nch; c += blockDim.x) {
      if (chunk_used[c] != 0) {
        continue;
      }
      const float value = chunk_max[c];
      if (value > best || (value == best && c < best_index)) {
        best = value;
        best_index = c;
      }
    }
    warp_argmax(best, best_index);
    if (lane == 0) {
      red[warp] = best;
      red_id[warp] = best_index;
    }
    __syncthreads();
    if (tid == 0) {
      int chosen = INT_MAX;
      float chosen_value = -CUDART_INF_F;
      for (int w = 0; w < nwarps; ++w) {
        if (red[w] > chosen_value ||
            (red[w] == chosen_value && red_id[w] < chosen)) {
          chosen_value = red[w];
          chosen = red_id[w];
        }
      }
      selected[slot] = chosen;
      chunk_used[chosen] = 1;
    }
    __syncthreads();
  }

  for (int t = tid; t < k * kTopkChunk; t += blockDim.x) {
    const int chunk = selected[t / kTopkChunk];
    const int index = chunk * kTopkChunk + (t % kTopkChunk);
    if (index < vocab) {
      lane_value[t] = bf2f(src[index]);
      lane_id[t] = index;
    } else {
      lane_value[t] = -CUDART_INF_F;
      lane_id[t] = INT_MAX;
    }
  }
  __syncthreads();

  for (int slot = 0; slot < k; ++slot) {
    float best = -CUDART_INF_F;
    int best_index = INT_MAX;
    for (int t = tid; t < k * kTopkChunk; t += blockDim.x) {
      const float value = lane_value[t];
      const int id = lane_id[t];
      if (value > best || (value == best && id < best_index)) {
        best = value;
        best_index = id;
      }
    }
    warp_argmax(best, best_index);
    if (lane == 0) {
      red[warp] = best;
      red_id[warp] = best_index;
    }
    __syncthreads();
    if (tid == 0) {
      int chosen = INT_MAX;
      float chosen_value = -CUDART_INF_F;
      for (int w = 0; w < nwarps; ++w) {
        if (red[w] > chosen_value ||
            (red[w] == chosen_value && red_id[w] < chosen)) {
          chosen_value = red[w];
          chosen = red_id[w];
        }
      }
      *winner = chosen;
      selected[slot] = chosen;
    }
    __syncthreads();
    const int chosen = *winner;
    for (int t = tid; t < k * kTopkChunk; t += blockDim.x) {
      if (lane_id[t] == chosen) {
        lane_value[t] = -CUDART_INF_F;
        lane_id[t] = INT_MAX;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int a = 1; a < k; ++a) {
      const int value = selected[a];
      int j = a;
      while (j > 0 && selected[j - 1] > value) {
        selected[j] = selected[j - 1];
        --j;
      }
      selected[j] = value;
    }
    for (int a = 0; a < k; ++a) {
      out[static_cast<int64_t>(n) * k + a] = selected[a];
    }
  }
}

// Warp-level reductions with the candidate and tg rows staged in shared
// memory; the per-lane loop needs no block barrier.
__global__ void serial_table_kernel(
    const __nv_bfloat16* __restrict__ packed, const float* __restrict__ rms,
    const __nv_bfloat16* __restrict__ raw,
    const __nv_bfloat16* __restrict__ lm_head,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ tg, const __nv_bfloat16* __restrict__ tsv,
    const __nv_bfloat16* __restrict__ gate_bias,
    const int* __restrict__ candidates, const unsigned int* __restrict__ input_ids,
    int* __restrict__ idx, int block, int hidden, int rank, int k,
    int cache_tg) {
  const int n = blockIdx.x;
  const int b = n / (block - 1);
  const int i = n % (block - 1);
  const int row = b * block + 1 + i;
  const int tid = threadIdx.x;
  const int nwarps = blockDim.x / 32;
  const int warp = tid / 32;
  const int lane = tid % 32;

  extern __shared__ unsigned char table_smem[];
  float* base = reinterpret_cast<float*>(table_smem);
  int* cand = reinterpret_cast<int*>(base + k);
  float* msg = reinterpret_cast<float*>(cand + k);
  __nv_bfloat16* tg_s = reinterpret_cast<__nv_bfloat16*>(
      msg + static_cast<size_t>(nwarps) * rank);

  if (tid < k) {
    cand[tid] = candidates[static_cast<int64_t>(n) * k + tid];
  }
  __syncthreads();
  if (cache_tg != 0) {
    for (int t = tid; t < k * rank; t += blockDim.x) {
      const int j = t / rank;
      const int d = t - j * rank;
      tg_s[t] = tg[static_cast<int64_t>(cand[j]) * rank + d];
    }
  }
  __syncthreads();

  const __nv_bfloat16* raw_row = raw + static_cast<int64_t>(row) * hidden;
  for (int j = warp; j < k; j += nwarps) {
    const __nv_bfloat16* head_row =
        lm_head + static_cast<int64_t>(cand[j]) * hidden;
    float partial = 0.0f;
    for (int h = lane; h < hidden; h += 32) {
      partial = fmaf(bf2f(raw_row[h]) * bf2f(norm_weight[h]),
                     bf2f(head_row[h]), partial);
    }
    partial = warp_sum_all(partial);
    if (lane == 0) {
      base[j] = partial;
    }
  }
  __syncthreads();

  const int64_t packed_base = static_cast<int64_t>(row) * 3 * rank;
  const float row_rms = rms[row];
  const int anchor = static_cast<int>(input_ids[b * block]);
  for (int receive = warp; receive < k; receive += nwarps) {
    const int sender =
        (i == 0) ? anchor
                 : candidates[static_cast<int64_t>(n - 1) * k + receive];
    const __nv_bfloat16* tsv_row =
        tsv + static_cast<int64_t>(sender) * 2 * rank;
    float* message = msg + static_cast<size_t>(warp) * rank;
    for (int d = lane; d < rank; d += 32) {
      const float q = bf2f(packed[packed_base + d]);
      const float s = bf2f(tsv_row[d]);
      const float v = bf2f(tsv_row[rank + d]);
      const float gate =
          1.0f / (1.0f + expf(-(q + s + bf2f(gate_bias[d]))));
      message[d] = gate * v;
    }
    float best = -CUDART_INF_F;
    int best_lane = 0;
    for (int j = 0; j < k; ++j) {
      const __nv_bfloat16* tg_row =
          cache_tg != 0
              ? tg_s + static_cast<size_t>(j) * rank
              : tg + static_cast<int64_t>(cand[j]) * rank;
      float dot = 0.0f;
      for (int d = lane; d < rank; d += 32) {
        dot = fmaf(message[d], bf2f(tg_row[d]), dot);
      }
      dot = warp_sum_all(dot);
      const float score = base[j] + row_rms * dot;
      if (score > best || (score == best && j < best_lane)) {
        best = score;
        best_lane = j;
      }
    }
    if (lane == 0) {
      idx[static_cast<int64_t>(n) * k + receive] = best_lane;
    }
  }
}

__global__ void scale_cols_kernel(const __nv_bfloat16* __restrict__ input,
                                  const __nv_bfloat16* __restrict__ weight,
                                  __nv_bfloat16* __restrict__ out,
                                  int64_t count, int cols) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += step) {
    out[index] = __float2bfloat16(bf2f(input[index]) *
                                  bf2f(weight[index % cols]));
  }
}

__global__ void scale_const_kernel(const __nv_bfloat16* __restrict__ input,
                                   __nv_bfloat16* __restrict__ out,
                                   int64_t count, float factor) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += step) {
    out[index] = __float2bfloat16(bf2f(input[index]) * factor);
  }
}

int grid_for(int64_t count) {
  return static_cast<int>(std::min<int64_t>((count + kThreads - 1) / kThreads,
                                            65535));
}

}  // namespace

extern "C" CUresult dspine_prepare_read_cuda(const __nv_bfloat16* raw,
                                             __nv_bfloat16* normalized,
                                             float* rms, int rows, int hidden,
                                             float eps, cudaStream_t stream) {
  if (rows <= 0 || hidden <= 0 || hidden > 65535) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  prepare_read_kernel<<<rows, kThreads, 0, stream>>>(raw, normalized, rms,
                                                     hidden, eps);
  return map_cuda_error(cudaGetLastError());
}

extern "C" CUresult dspine_normalize_codes_cuda(
    const __nv_bfloat16* projected, const __nv_bfloat16* codebook,
    const unsigned int* input_ids, __nv_bfloat16* codes, int rows,
    int code_size, int block, float eps, cudaStream_t stream) {
  if (rows <= 0 || code_size <= 0 || code_size > 65535 || block <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  normalize_codes_kernel<<<rows, kThreads, 0, stream>>>(
      projected, codebook, input_ids, codes, code_size, block, eps);
  return map_cuda_error(cudaGetLastError());
}

extern "C" CUresult dspine_feature_message_cuda(
    const __nv_bfloat16* packed, const __nv_bfloat16* gate_bias,
    __nv_bfloat16* message, int rows, int rank, int block,
    cudaStream_t stream) {
  if (rows <= 0 || rank <= 0 || rank > 8192 || block <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  feature_message_kernel<<<rows, kThreads, 0, stream>>>(
      packed, gate_bias, message, rank, block);
  return map_cuda_error(cudaGetLastError());
}

extern "C" CUresult dspine_feature_finish_cuda(
    const __nv_bfloat16* raw, const __nv_bfloat16* delta, const float* rms,
    __nv_bfloat16* out, int rows, int hidden, int block, cudaStream_t stream) {
  if (rows <= 0 || hidden <= 0 || block <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int64_t count = static_cast<int64_t>(rows) * hidden;
  feature_finish_kernel<<<grid_for(count), kThreads, 0, stream>>>(
      raw, delta, rms, out, count, hidden, block);
  return map_cuda_error(cudaGetLastError());
}

extern "C" CUresult dspine_feature_finish_norm_cuda(
    const __nv_bfloat16* raw, const __nv_bfloat16* delta, const float* rms,
    __nv_bfloat16* out, __nv_bfloat16* normed, const __nv_bfloat16* weight,
    int rows, int hidden, int block, float eps, cudaStream_t stream) {
  if (rows <= 0 || hidden <= 0 || hidden > 24576 || block <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const size_t smem = static_cast<size_t>(hidden) * sizeof(__nv_bfloat16) +
                      static_cast<size_t>(kThreads / 32) * sizeof(float);
  if (smem > 48 * 1024) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  feature_finish_norm_kernel<<<rows, kThreads, smem, stream>>>(
      raw, delta, rms, out, normed, weight, hidden, block, eps);
  return map_cuda_error(cudaGetLastError());
}

extern "C" CUresult dspine_topk_cuda(const __nv_bfloat16* logits,
                                        int* candidate_ids, int batch, int block,
                                        int vocab, int k,
                                        cudaStream_t stream) {
  if (batch <= 0 || block < 2 || vocab <= 0 || k < 1 || k > 64 ||
      (k & (k - 1)) != 0 || vocab < k * kTopkChunk) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int rows = batch * (block - 1);
  const int nch = (vocab + kTopkChunk - 1) / kTopkChunk;
  const size_t smem = static_cast<size_t>(nch) * sizeof(float) +
                      ((static_cast<size_t>(nch) + 3) & ~size_t(3)) +
                      static_cast<size_t>(k) * kTopkChunk * sizeof(float) +
                      static_cast<size_t>(k) * kTopkChunk * sizeof(int) +
                      static_cast<size_t>(k) * sizeof(int) +
                      static_cast<size_t>(kThreads) * sizeof(float) +
                      static_cast<size_t>(kThreads) * sizeof(int) +
                      sizeof(int);
  if (smem > 48 * 1024) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  topk_kernel<<<rows, kThreads, smem, stream>>>(logits, candidate_ids, block,
                                                   vocab, k);
  return map_cuda_error(cudaGetLastError());
}

extern "C" CUresult dspine_serial_table_cuda(
    const __nv_bfloat16* packed, const float* rms, const __nv_bfloat16* raw,
    const __nv_bfloat16* lm_head, const __nv_bfloat16* norm_weight,
    const __nv_bfloat16* tg, const __nv_bfloat16* tsv,
    const __nv_bfloat16* gate_bias, const int* candidates,
    const unsigned int* input_ids, int* idx, int batch, int block, int hidden,
    int rank, int k, cudaStream_t stream) {
  if (batch <= 0 || block < 2 || hidden <= 0 || rank <= 0 || rank > 2048 ||
      k < 1 || k > 64) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int rows = batch * (block - 1);
  const int nwarps = kThreads / 32;
  const size_t fixed = static_cast<size_t>(k) * sizeof(float) +
                       static_cast<size_t>(k) * sizeof(int);
  const size_t msg = static_cast<size_t>(nwarps) * rank * sizeof(float);
  const size_t tg_bytes = static_cast<size_t>(k) * rank * sizeof(__nv_bfloat16);
  const bool cache_tg = fixed + msg + tg_bytes <= 48 * 1024;
  const size_t smem = fixed + msg + (cache_tg ? tg_bytes : 0);
  serial_table_kernel<<<rows, kThreads, smem, stream>>>(
      packed, rms, raw, lm_head, norm_weight, tg, tsv, gate_bias, candidates,
      input_ids, idx, block, hidden, rank, k, cache_tg ? 1 : 0);
  return map_cuda_error(cudaGetLastError());
}

extern "C" CUresult dspine_serial_chain_cuda(
    const int* idx, const int* candidates, const unsigned int* input_ids,
    unsigned int* out, int batch, int block, int k, cudaStream_t stream) {
  if (batch <= 0 || block < 2 || k < 1 || k > 64) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  serial_chain_kernel<<<batch, 1, 0, stream>>>(idx, candidates, input_ids, out,
                                               block, k);
  return map_cuda_error(cudaGetLastError());
}

extern "C" CUresult dspine_scale_cols_cuda(const __nv_bfloat16* input,
                                           const __nv_bfloat16* weight,
                                           __nv_bfloat16* out, int64_t rows,
                                           int cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int64_t count = rows * cols;
  scale_cols_kernel<<<grid_for(count), kThreads, 0, stream>>>(input, weight,
                                                              out, count, cols);
  return map_cuda_error(cudaGetLastError());
}

extern "C" CUresult dspine_scale_const_cuda(const __nv_bfloat16* input,
                                            __nv_bfloat16* out, int64_t count,
                                            float factor,
                                            cudaStream_t stream) {
  if (count <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  scale_const_kernel<<<grid_for(count), kThreads, 0, stream>>>(input, out,
                                                               count, factor);
  return map_cuda_error(cudaGetLastError());
}
