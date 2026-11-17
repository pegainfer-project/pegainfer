#include "common.cuh"

#define LOGPROB_BLOCK 256

// Fixed block size and reduction order keep outputs bitwise reproducible and
// independent of batch composition.

__device__ __forceinline__ bool rank_better(float lhs_val, int lhs_idx,
                                            float rhs_val, int rhs_idx) {
  return lhs_val > rhs_val || (lhs_val == rhs_val && lhs_idx < rhs_idx);
}

__global__ void logprob_topk_batch_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ row_indices,
    const int* __restrict__ picked,
    const int* __restrict__ top_k,
    float* __restrict__ out_picked_lp,
    int* __restrict__ out_picked_rank,
    float* __restrict__ out_topk_vals,
    int* __restrict__ out_topk_ids,
    int rows,
    int n,
    int k_max) {
  extern __shared__ char shared_mem[];
  float* shared_vals = reinterpret_cast<float*>(shared_mem);
  int* shared_idxs =
      reinterpret_cast<int*>(shared_mem + blockDim.x * sizeof(float));
  // Aliases the arrays above: sizeof(double) == sizeof(float) + sizeof(int).
  double* shared_sums = reinterpret_cast<double*>(shared_mem);

  int slot = blockIdx.x;
  if (slot >= rows) return;
  const __nv_bfloat16* row_x = x + static_cast<size_t>(row_indices[slot]) * n;
  int tid = threadIdx.x;

  float picked_val = __bfloat162float(row_x[picked[slot]]);
  float local_max = -INFINITY;
  int local_rank = 0;
  for (int i = tid; i < n; i += blockDim.x) {
    float value = __bfloat162float(row_x[i]);
    local_max = fmaxf(local_max, value);
    local_rank += value >= picked_val;
  }
  shared_vals[tid] = local_max;
  shared_idxs[tid] = local_rank;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_vals[tid] = fmaxf(shared_vals[tid], shared_vals[tid + s]);
      shared_idxs[tid] += shared_idxs[tid + s];
    }
    __syncthreads();
  }
  float row_max = shared_vals[0];
  if (tid == 0) out_picked_rank[slot] = shared_idxs[0];
  __syncthreads();

  double local_sum = 0.0;
  for (int i = tid; i < n; i += blockDim.x) {
    local_sum +=
        static_cast<double>(expf(__bfloat162float(row_x[i]) - row_max));
  }
  shared_sums[tid] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_sums[tid] += shared_sums[tid + s];
    }
    __syncthreads();
  }
  float lse = row_max + static_cast<float>(log(shared_sums[0]));
  __syncthreads();

  if (tid == 0) {
    out_picked_lp[slot] = picked_val - lse;
  }

  // Selection runs in strictly decreasing (value, -id) lexicographic order,
  // so "after the previous winner" replaces a selected-set membership test.
  int k = top_k[slot];
  float prev_val = INFINITY;
  int prev_id = -1;
  for (int j = 0; j < k; ++j) {
    float local_best = -INFINITY;
    int local_id = n;
    for (int i = tid; i < n; i += blockDim.x) {
      float v = __bfloat162float(row_x[i]);
      bool eligible = v < prev_val || (v == prev_val && i > prev_id);
      if (eligible && rank_better(v, i, local_best, local_id)) {
        local_best = v;
        local_id = i;
      }
    }
    shared_vals[tid] = local_best;
    shared_idxs[tid] = local_id;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        float rhs_val = shared_vals[tid + s];
        int rhs_idx = shared_idxs[tid + s];
        if (rank_better(rhs_val, rhs_idx, shared_vals[tid], shared_idxs[tid])) {
          shared_vals[tid] = rhs_val;
          shared_idxs[tid] = rhs_idx;
        }
      }
      __syncthreads();
    }
    prev_val = shared_vals[0];
    prev_id = shared_idxs[0];
    if (tid == 0) {
      out_topk_vals[static_cast<size_t>(slot) * k_max + j] = prev_val - lse;
      out_topk_ids[static_cast<size_t>(slot) * k_max + j] = prev_id;
    }
    __syncthreads();
  }
}

extern "C" {
void logprob_topk_batch_bf16_cuda(const __nv_bfloat16* x,
                                  const int* row_indices, const int* picked,
                                  const int* top_k, float* out_picked_lp,
                                  int* out_picked_rank,
                                  float* out_topk_vals, int* out_topk_ids,
                                  int rows, int n, int k_max,
                                  cudaStream_t stream) {
  size_t smem = LOGPROB_BLOCK * (sizeof(float) + sizeof(int));
  logprob_topk_batch_kernel<<<rows, LOGPROB_BLOCK, smem, stream>>>(
      x, row_indices, picked, top_k, out_picked_lp, out_picked_rank,
      out_topk_vals, out_topk_ids, rows, n, k_max);
}
}
