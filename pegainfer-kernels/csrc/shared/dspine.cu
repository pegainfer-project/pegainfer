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

// Block-wide max with lowest-index tie-breaking; blockDim must be a power of two.
__device__ void block_argmax(float& value, int& index, float* red,
                             int* red_index) {
  const int tid = threadIdx.x;
  red[tid] = value;
  red_index[tid] = index;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other = red[tid + stride];
      const int other_index = red_index[tid + stride];
      if (other > red[tid] ||
          (other == red[tid] && other_index < red_index[tid])) {
        red[tid] = other;
        red_index[tid] = other_index;
      }
    }
    __syncthreads();
  }
  value = red[0];
  index = red_index[0];
  __syncthreads();
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
  float partial = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float x = bf2f(raw[base + i]);
    partial += x * x;
  }
  const float mean = block_sum(partial, red) / static_cast<float>(hidden);
  __shared__ float row_rms;
  if (threadIdx.x == 0) {
    row_rms = sqrtf(fmaxf(mean, 1.1754943508222875e-38f));
    rms[row] = row_rms;
  }
  __syncthreads();
  const float denom = row_rms + eps;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    normalized[base + i] =
        __float2bfloat16(bf2f(raw[base + i]) / denom);
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

// out = raw on anchors, else raw + rms * delta.
__global__ void feature_finish_kernel(const __nv_bfloat16* __restrict__ raw,
                                      const __nv_bfloat16* __restrict__ delta,
                                      const float* __restrict__ rms,
                                      __nv_bfloat16* __restrict__ out,
                                      int64_t total, int hidden, int block) {
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

// Per-row top-K (K <= 64) over the proposal rows of a [batch*block, vocab] logits
// buffer, skipping each block's anchor row. Ids are emitted ascending.
__global__ void topk_kernel(const __nv_bfloat16* __restrict__ logits,
                            int* __restrict__ out, int block, int vocab, int k) {
  const int n = blockIdx.x;
  const int b = n / (block - 1);
  const int i = n % (block - 1);
  const int64_t row = static_cast<int64_t>(b) * block + 1 + i;
  const __nv_bfloat16* src = logits + row * vocab;
  const int nch = (vocab + kTopkChunk - 1) / kTopkChunk;

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

  const int tid = threadIdx.x;
  for (int c = tid; c < nch; c += blockDim.x) {
    const int begin = c * kTopkChunk;
    const int end = min(begin + kTopkChunk, vocab);
    float best = -CUDART_INF_F;
    for (int j = begin; j < end; ++j) {
      best = fmaxf(best, bf2f(src[j]));
    }
    chunk_max[c] = best;
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
    block_argmax(best, best_index, red, red_id);
    if (tid == 0) {
      selected[slot] = best_index;
      chunk_used[best_index] = 1;
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
    block_argmax(best, best_index, red, red_id);
    if (tid == 0) {
      for (int t = 0; t < k * kTopkChunk; ++t) {
        if (lane_id[t] == best_index) {
          lane_value[t] = -CUDART_INF_F;
          lane_id[t] = INT_MAX;
          break;
        }
      }
      selected[slot] = best_index;
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

// idx[n, k] = best receiver lane for proposal row n when the previous row chose
// lane k. One block per proposal row; lanes loop inside the block.
__global__ void serial_table_kernel(
    const __nv_bfloat16* __restrict__ packed, const float* __restrict__ rms,
    const __nv_bfloat16* __restrict__ raw,
    const __nv_bfloat16* __restrict__ lm_head,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ tg, const __nv_bfloat16* __restrict__ tsv,
    const __nv_bfloat16* __restrict__ gate_bias,
    const int* __restrict__ candidates, const unsigned int* __restrict__ input_ids,
    int* __restrict__ idx, int block, int hidden, int rank, int k) {
  const int n = blockIdx.x;
  const int b = n / (block - 1);
  const int i = n % (block - 1);
  const int row = b * block + 1 + i;
  const int tid = threadIdx.x;

  extern __shared__ unsigned char table_smem[];
  float* base = reinterpret_cast<float*>(table_smem);
  float* message = base + k;
  float* red = message + rank;

  const __nv_bfloat16* raw_row = raw + static_cast<int64_t>(row) * hidden;
  for (int lane = 0; lane < k; ++lane) {
    const int token = candidates[static_cast<int64_t>(n) * k + lane];
    const __nv_bfloat16* head_row =
        lm_head + static_cast<int64_t>(token) * hidden;
    float partial = 0.0f;
    for (int h = tid; h < hidden; h += blockDim.x) {
      partial += bf2f(raw_row[h]) * bf2f(norm_weight[h]) * bf2f(head_row[h]);
    }
    const float sum = block_sum(partial, red);
    if (tid == 0) {
      base[lane] = sum;
    }
    __syncthreads();
  }

  const int64_t packed_base = static_cast<int64_t>(row) * 3 * rank;
  const float row_rms = rms[row];
  const int anchor = static_cast<int>(input_ids[b * block]);
  for (int lane = 0; lane < k; ++lane) {
    const int sender =
        (i == 0) ? anchor : candidates[static_cast<int64_t>(n - 1) * k + lane];
    const int64_t tsv_base = static_cast<int64_t>(sender) * 2 * rank;
    for (int d = tid; d < rank; d += blockDim.x) {
      const float q = bf2f(packed[packed_base + d]);
      const float s = bf2f(tsv[tsv_base + d]);
      const float v = bf2f(tsv[tsv_base + rank + d]);
      const float gate = 1.0f / (1.0f + expf(-(q + s + bf2f(gate_bias[d]))));
      message[d] = gate * v;
    }
    __syncthreads();

    float best = -CUDART_INF_F;
    int best_lane = 0;
    for (int j = 0; j < k; ++j) {
      const int token = candidates[static_cast<int64_t>(n) * k + j];
      const __nv_bfloat16* tg_row = tg + static_cast<int64_t>(token) * rank;
      float dot = 0.0f;
      for (int d = tid; d < rank; d += blockDim.x) {
        dot += message[d] * bf2f(tg_row[d]);
      }
      const float score = base[j] + row_rms * block_sum(dot, red);
      if (score > best || (score == best && j < best_lane)) {
        best = score;
        best_lane = j;
      }
      __syncthreads();
    }
    if (tid == 0) {
      idx[static_cast<int64_t>(n) * k + lane] = best_lane;
    }
    __syncthreads();
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

extern "C" CUresult dspine_topk_cuda(const __nv_bfloat16* logits,
                                     int* candidate_ids, int batch, int block,
                                     int vocab, int k, cudaStream_t stream) {
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
                      static_cast<size_t>(kThreads) * sizeof(int);
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
  const size_t smem = static_cast<size_t>(k) * sizeof(float) +
                      static_cast<size_t>(rank) * sizeof(float) +
                      static_cast<size_t>(kThreads) * sizeof(float);
  serial_table_kernel<<<rows, kThreads, smem, stream>>>(
      packed, rms, raw, lm_head, norm_weight, tg, tsv, gate_bias, candidates,
      input_ids, idx, block, hidden, rank, k);
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
