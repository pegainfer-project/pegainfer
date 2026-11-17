#include "common.cuh"

#include <atomic>

#define SAMPLE_BLOCK 256
#define ARGMAX_BATCH_TILE_ELEMS 4096
// The Markov-step argmax reads only ~0.6 MB (one logits row + one bias row) —
// at 4096-elem tiles its ~38 blocks leave the kernel latency-bound (~10 us for
// a ~0.2 us read on H100). 1024-elem tiles give ~152 blocks and put it back on
// the bandwidth curve (measured 9.7 -> 4.5 us). Markov-only: the batched
// argmax reads a full row per block and keeps the wider tile.
#define MARKOV_STEP_TILE_ELEMS 1024

__device__ __forceinline__ bool argmax_better(float lhs_val, int lhs_idx,
                                              float rhs_val, int rhs_idx) {
  return lhs_val > rhs_val || (lhs_val == rhs_val && lhs_idx < rhs_idx);
}

__device__ __forceinline__ void markov_trigger_dependent_launch() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __threadfence();
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

__device__ __forceinline__ void markov_wait_on_dependent_launch() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
}

inline bool markov_pdl_supported_on_current_device() {
  static std::atomic<int> cached{-1};
  int value = cached.load(std::memory_order_acquire);
  if (value >= 0) return value != 0;

  int device = 0;
  int major = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    cached.store(0, std::memory_order_release);
    return false;
  }
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) !=
      cudaSuccess) {
    cached.store(0, std::memory_order_release);
    return false;
  }
  value = major >= 9 ? 1 : 0;
  cached.store(value, std::memory_order_release);
  return value != 0;
}

__global__ void argmax_kernel(const __nv_bfloat16* __restrict__ x,
                              int* __restrict__ out, int n) {
  extern __shared__ char shared_mem[];
  float* shared_vals = (float*)shared_mem;
  int* shared_idxs = (int*)(shared_mem + blockDim.x * sizeof(float));

  int tid = threadIdx.x;
  int stride = blockDim.x;

  float local_max = -INFINITY;
  int local_idx = 0;
  for (int i = tid; i < n; i += stride) {
    float val = __bfloat162float(x[i]);
    if (argmax_better(val, i, local_max, local_idx)) {
      local_max = val;
      local_idx = i;
    }
  }
  shared_vals[tid] = local_max;
  shared_idxs[tid] = local_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (argmax_better(shared_vals[tid + s], shared_idxs[tid + s],
                        shared_vals[tid], shared_idxs[tid])) {
        shared_vals[tid] = shared_vals[tid + s];
        shared_idxs[tid] = shared_idxs[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[0] = shared_idxs[0];
  }
}

__global__ void argmax_batch_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ values,
    int* __restrict__ indices,
    int rows,
    int n) {
  extern __shared__ char shared_mem[];
  float* shared_vals = reinterpret_cast<float*>(shared_mem);
  int* shared_idxs =
      reinterpret_cast<int*>(shared_mem + blockDim.x * sizeof(float));

  int row = blockIdx.x;
  if (row >= rows) return;
  const __nv_bfloat16* row_x = x + static_cast<size_t>(row) * n;
  int tid = threadIdx.x;

  float local_max = -INFINITY;
  int local_idx = 0;
  for (int i = tid; i < n; i += blockDim.x) {
    float val = __bfloat162float(row_x[i]);
    if (argmax_better(val, i, local_max, local_idx)) {
      local_max = val;
      local_idx = i;
    }
  }
  shared_vals[tid] = local_max;
  shared_idxs[tid] = local_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float rhs_val = shared_vals[tid + s];
      int rhs_idx = shared_idxs[tid + s];
      if (argmax_better(rhs_val, rhs_idx, shared_vals[tid], shared_idxs[tid])) {
        shared_vals[tid] = rhs_val;
        shared_idxs[tid] = rhs_idx;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    indices[row] = shared_idxs[0];
    values[row] = __float2bfloat16(shared_vals[0]);
  }
}

__global__ void argmax_batch_bf16_partial_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ row_indices,  // nullptr → row maps to itself
    float* __restrict__ partial_values,
    int* __restrict__ partial_indices,
    int rows,
    int n,
    int tiles_per_row) {
  extern __shared__ char shared_mem[];
  float* shared_vals = reinterpret_cast<float*>(shared_mem);
  int* shared_idxs =
      reinterpret_cast<int*>(shared_mem + blockDim.x * sizeof(float));

  int tile = blockIdx.x;
  int row = blockIdx.y;
  if (row >= rows || tile >= tiles_per_row) return;

  int start = tile * ARGMAX_BATCH_TILE_ELEMS;
  int end = start + ARGMAX_BATCH_TILE_ELEMS;
  if (end > n) end = n;
  int source_row = row_indices ? row_indices[row] : row;
  const __nv_bfloat16* row_x = x + static_cast<size_t>(source_row) * n;
  int tid = threadIdx.x;

  float local_max = -INFINITY;
  int local_idx = 0;
  for (int i = start + tid; i < end; i += blockDim.x) {
    float val = __bfloat162float(row_x[i]);
    if (argmax_better(val, i, local_max, local_idx)) {
      local_max = val;
      local_idx = i;
    }
  }
  shared_vals[tid] = local_max;
  shared_idxs[tid] = local_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float rhs_val = shared_vals[tid + s];
      int rhs_idx = shared_idxs[tid + s];
      if (argmax_better(rhs_val, rhs_idx, shared_vals[tid], shared_idxs[tid])) {
        shared_vals[tid] = rhs_val;
        shared_idxs[tid] = rhs_idx;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    int out = row * tiles_per_row + tile;
    partial_values[out] = shared_vals[0];
    partial_indices[out] = shared_idxs[0];
  }
}

__global__ void argmax_batch_bf16_finalize_kernel(
    const float* __restrict__ partial_values,
    const int* __restrict__ partial_indices,
    __nv_bfloat16* __restrict__ values,
    int* __restrict__ indices,
    int rows,
    int tiles_per_row) {
  extern __shared__ char shared_mem[];
  float* shared_vals = reinterpret_cast<float*>(shared_mem);
  int* shared_idxs =
      reinterpret_cast<int*>(shared_mem + blockDim.x * sizeof(float));

  int row = blockIdx.x;
  int tid = threadIdx.x;
  int base = row * tiles_per_row;
  float local_max = -INFINITY;
  int local_idx = 0;
  for (int tile = tid; tile < tiles_per_row; tile += blockDim.x) {
    float val = partial_values[base + tile];
    int idx = partial_indices[base + tile];
    if (argmax_better(val, idx, local_max, local_idx)) {
      local_max = val;
      local_idx = idx;
    }
  }
  shared_vals[tid] = local_max;
  shared_idxs[tid] = local_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float rhs_val = shared_vals[tid + s];
      int rhs_idx = shared_idxs[tid + s];
      if (argmax_better(rhs_val, rhs_idx, shared_vals[tid], shared_idxs[tid])) {
        shared_vals[tid] = rhs_val;
        shared_idxs[tid] = rhs_idx;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    indices[row] = shared_idxs[0];
    values[row] = __float2bfloat16(shared_vals[0]);
  }
}

// DSpark Markov-head step. For request `row` at draft position `step`, scan
// `base[(row*block_size + step)*n + v] + bias[row*n + v]` over the vocab and
// argmax. `base` is the request-major block logits [rows*block_size, n]; `bias`
// is the per-request Markov logit bias [rows, n] for this step. Two-stage like
// the indexed argmax above; the finalize writes the chosen token id as u32 so it
// feeds straight back as the next step's previous-token embedding lookup.
__global__ void markov_step_partial_kernel(
    const __nv_bfloat16* __restrict__ base,
    const __nv_bfloat16* __restrict__ bias,
    int block_size,
    int step,
    int chains,
    const int* __restrict__ req_map,
    float* __restrict__ partial_values,
    int* __restrict__ partial_indices,
    int rows,
    int n,
    int tiles_per_row) {
  extern __shared__ char shared_mem[];
  float* shared_vals = reinterpret_cast<float*>(shared_mem);
  int* shared_idxs =
      reinterpret_cast<int*>(shared_mem + blockDim.x * sizeof(float));

  int tile = blockIdx.x;
  int row = blockIdx.y;
  if (row >= rows || tile >= tiles_per_row) return;

  int start = tile * MARKOV_STEP_TILE_ELEMS;
  int end = start + MARKOV_STEP_TILE_ELEMS;
  if (end > n) end = n;
  const __nv_bfloat16* base_row =
      base +
      static_cast<size_t>(
          (req_map ? req_map[row / chains] : (row / chains)) * block_size +
          step) *
          n;
  const __nv_bfloat16* bias_row = bias + static_cast<size_t>(row) * n;
  int tid = threadIdx.x;

  float local_max = -INFINITY;
  int local_idx = 0;
  for (int i = start + tid; i < end; i += blockDim.x) {
    float val = __bfloat162float(base_row[i]) + __bfloat162float(bias_row[i]);
    if (argmax_better(val, i, local_max, local_idx)) {
      local_max = val;
      local_idx = i;
    }
  }
  shared_vals[tid] = local_max;
  shared_idxs[tid] = local_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float rhs_val = shared_vals[tid + s];
      int rhs_idx = shared_idxs[tid + s];
      if (argmax_better(rhs_val, rhs_idx, shared_vals[tid], shared_idxs[tid])) {
        shared_vals[tid] = rhs_val;
        shared_idxs[tid] = rhs_idx;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    int out = row * tiles_per_row + tile;
    partial_values[out] = shared_vals[0];
    partial_indices[out] = shared_idxs[0];
  }
  markov_trigger_dependent_launch();
}

__global__ void markov_step_finalize_kernel(
    const float* __restrict__ partial_values,
    const int* __restrict__ partial_indices,
    unsigned int* __restrict__ out_tokens,
    unsigned int* __restrict__ sampled_tokens,
    int block_size,
    int step,
    int rows,
    int tiles_per_row) {
  markov_wait_on_dependent_launch();

  extern __shared__ char shared_mem[];
  float* shared_vals = reinterpret_cast<float*>(shared_mem);
  int* shared_idxs =
      reinterpret_cast<int*>(shared_mem + blockDim.x * sizeof(float));

  int row = blockIdx.x;
  int tid = threadIdx.x;
  int base = row * tiles_per_row;
  float local_max = -INFINITY;
  int local_idx = 0;
  for (int tile = tid; tile < tiles_per_row; tile += blockDim.x) {
    float val = partial_values[base + tile];
    int idx = partial_indices[base + tile];
    if (argmax_better(val, idx, local_max, local_idx)) {
      local_max = val;
      local_idx = idx;
    }
  }
  shared_vals[tid] = local_max;
  shared_idxs[tid] = local_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float rhs_val = shared_vals[tid + s];
      int rhs_idx = shared_idxs[tid + s];
      if (argmax_better(rhs_val, rhs_idx, shared_vals[tid], shared_idxs[tid])) {
        shared_vals[tid] = rhs_val;
        shared_idxs[tid] = rhs_idx;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    unsigned int token = static_cast<unsigned int>(shared_idxs[0]);
    out_tokens[row] = token;
    sampled_tokens[row * block_size + step] = token;
  }
}

// Top-2 over `base[(row*block_size + step)*n + v] + bias[row*n + v]` — exact
// when a row has two or more finite candidates; a fully masked row emits
// index 0 for both slots (callers deduplicate).
// The single-best partial kernel above cannot back a top-2: when the winner
// and runner-up land in the same tile, the tile's top-1 partial has already
// discarded the runner-up. So the partial stage tracks a (top1, top2) pair per
// tile and the finalize merges pairs. Used by the speculative hedge to open a
// second draft chain at the runner-up token.
__device__ __forceinline__ void top2_consider(float val, int idx, float& v1,
                                              int& i1, float& v2, int& i2) {
  if (argmax_better(val, idx, v1, i1)) {
    v2 = v1;
    i2 = i1;
    v1 = val;
    i1 = idx;
  } else if (argmax_better(val, idx, v2, i2)) {
    v2 = val;
    i2 = idx;
  }
}

__global__ void markov_step_top2_partial_kernel(
    const __nv_bfloat16* __restrict__ base,
    const __nv_bfloat16* __restrict__ bias,
    int block_size,
    int step,
    int chains,
    float* __restrict__ partial_v1,
    int* __restrict__ partial_i1,
    float* __restrict__ partial_v2,
    int* __restrict__ partial_i2,
    int rows,
    int n,
    int tiles_per_row) {
  extern __shared__ char shared_mem[];
  float* sv1 = reinterpret_cast<float*>(shared_mem);
  float* sv2 = sv1 + blockDim.x;
  int* si1 = reinterpret_cast<int*>(sv2 + blockDim.x);
  int* si2 = si1 + blockDim.x;

  int tile = blockIdx.x;
  int row = blockIdx.y;
  if (row >= rows || tile >= tiles_per_row) return;

  int start = tile * MARKOV_STEP_TILE_ELEMS;
  int end = start + MARKOV_STEP_TILE_ELEMS;
  if (end > n) end = n;
  const __nv_bfloat16* base_row =
      base + static_cast<size_t>((row / chains) * block_size + step) * n;
  const __nv_bfloat16* bias_row = bias + static_cast<size_t>(row) * n;
  int tid = threadIdx.x;

  float v1 = -INFINITY, v2 = -INFINITY;
  int i1 = 0, i2 = 0;
  for (int i = start + tid; i < end; i += blockDim.x) {
    float val = __bfloat162float(base_row[i]) + __bfloat162float(bias_row[i]);
    top2_consider(val, i, v1, i1, v2, i2);
  }
  sv1[tid] = v1;
  si1[tid] = i1;
  sv2[tid] = v2;
  si2[tid] = i2;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      top2_consider(sv1[tid + s], si1[tid + s], sv1[tid], si1[tid], sv2[tid],
                    si2[tid]);
      top2_consider(sv2[tid + s], si2[tid + s], sv1[tid], si1[tid], sv2[tid],
                    si2[tid]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    int out = row * tiles_per_row + tile;
    partial_v1[out] = sv1[0];
    partial_i1[out] = si1[0];
    partial_v2[out] = sv2[0];
    partial_i2[out] = si2[0];
  }
}

__global__ void markov_step_top2_finalize_kernel(
    const float* __restrict__ partial_v1,
    const int* __restrict__ partial_i1,
    const float* __restrict__ partial_v2,
    const int* __restrict__ partial_i2,
    unsigned int* __restrict__ out_tokens,
    unsigned int* __restrict__ sampled_tokens,
    unsigned int* __restrict__ out_top2,
    int block_size,
    int step,
    int rows,
    int tiles_per_row) {
  extern __shared__ char shared_mem[];
  float* sv1 = reinterpret_cast<float*>(shared_mem);
  float* sv2 = sv1 + blockDim.x;
  int* si1 = reinterpret_cast<int*>(sv2 + blockDim.x);
  int* si2 = si1 + blockDim.x;

  int row = blockIdx.x;
  int tid = threadIdx.x;
  int base = row * tiles_per_row;
  float v1 = -INFINITY, v2 = -INFINITY;
  int i1 = 0, i2 = 0;
  for (int tile = tid; tile < tiles_per_row; tile += blockDim.x) {
    top2_consider(partial_v1[base + tile], partial_i1[base + tile], v1, i1, v2,
                  i2);
    top2_consider(partial_v2[base + tile], partial_i2[base + tile], v1, i1, v2,
                  i2);
  }
  sv1[tid] = v1;
  si1[tid] = i1;
  sv2[tid] = v2;
  si2[tid] = i2;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      top2_consider(sv1[tid + s], si1[tid + s], sv1[tid], si1[tid], sv2[tid],
                    si2[tid]);
      top2_consider(sv2[tid + s], si2[tid + s], sv1[tid], si1[tid], sv2[tid],
                    si2[tid]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    unsigned int token = static_cast<unsigned int>(si1[0]);
    out_tokens[row] = token;
    sampled_tokens[row * block_size + step] = token;
    out_top2[row] = static_cast<unsigned int>(si2[0]);
  }
}


// Hedge-ladder branch forcing: chain-row (i*c + j) takes its device-resident
// runner-up token at `step` — written into both the ping-pong prev buffer
// (so later steps condition on it) and the sampled block (so the single
// final D2H carries it). Replaces per-branch-step D2H syncs.
__global__ void hedge_ladder_force_kernel(unsigned int* __restrict__ prev,
                                          unsigned int* __restrict__ sampled,
                                          const unsigned int* __restrict__ runners,
                                          const int* __restrict__ req_map,
                                          int n,
                                          int c,
                                          int j,
                                          int runner_stride,
                                          int block_size,
                                          int step) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int req = req_map[i];
  unsigned int tok = runners[j * runner_stride + req];
  int row = i * c + j;
  prev[row] = tok;
  sampled[row * block_size + step] = tok;
}

extern "C" {
void argmax_cuda(const __nv_bfloat16* x, int* out, int n, cudaStream_t stream) {
  argmax_kernel<<<1, SAMPLE_BLOCK,
                  SAMPLE_BLOCK * (sizeof(float) + sizeof(int)), stream>>>(x, out, n);
}

void argmax_batch_bf16_cuda(const __nv_bfloat16* x, __nv_bfloat16* values,
                            int* indices, int rows, int n,
                            cudaStream_t stream) {
  argmax_batch_bf16_kernel<<<rows, SAMPLE_BLOCK,
                             SAMPLE_BLOCK * (sizeof(float) + sizeof(int)),
                             stream>>>(x, values, indices, rows, n);
}

void argmax_batch_bf16_split_cuda(const __nv_bfloat16* x, __nv_bfloat16* values,
                                  int* indices, float* partial_values,
                                  int* partial_indices, int rows, int n,
                                  cudaStream_t stream) {
  int tiles_per_row = (n + ARGMAX_BATCH_TILE_ELEMS - 1) / ARGMAX_BATCH_TILE_ELEMS;
  size_t smem = SAMPLE_BLOCK * (sizeof(float) + sizeof(int));
  argmax_batch_bf16_partial_kernel<<<dim3(tiles_per_row, rows), SAMPLE_BLOCK, smem, stream>>>(
      x, nullptr, partial_values, partial_indices, rows, n, tiles_per_row);
  argmax_batch_bf16_finalize_kernel<<<rows, SAMPLE_BLOCK, smem, stream>>>(
      partial_values, partial_indices, values, indices, rows, tiles_per_row);
}

void argmax_batch_bf16_split_indexed_cuda(const __nv_bfloat16* x,
                                          const int* row_indices,
                                          __nv_bfloat16* values, int* indices,
                                          float* partial_values,
                                          int* partial_indices, int rows, int n,
                                          cudaStream_t stream) {
  int tiles_per_row = (n + ARGMAX_BATCH_TILE_ELEMS - 1) / ARGMAX_BATCH_TILE_ELEMS;
  size_t smem = SAMPLE_BLOCK * (sizeof(float) + sizeof(int));
  argmax_batch_bf16_partial_kernel<<<dim3(tiles_per_row, rows), SAMPLE_BLOCK, smem, stream>>>(
      x, row_indices, partial_values, partial_indices, rows, n, tiles_per_row);
  argmax_batch_bf16_finalize_kernel<<<rows, SAMPLE_BLOCK, smem, stream>>>(
      partial_values, partial_indices, values, indices, rows, tiles_per_row);
}

// The three markov/hedge launchers below return `cudaGetLastError()` so the
// safe wrappers fail closed on launch rejection (bad grid, shared-memory
// exhaustion) instead of surfacing it at the next unrelated CUDA call.
// Legal during stream capture: the error state is thread-local and querying
// it does not touch the stream.
cudaError_t markov_step_argmax_cuda(const __nv_bfloat16* base,
                                    const __nv_bfloat16* bias, int block_size, int step,
                                    int chains, const int* req_map, int rows, int n,
                                    float* partial_values,
                                    int* partial_indices, unsigned int* out_tokens,
                                    unsigned int* sampled_tokens,
                                    cudaStream_t stream) {
  int tiles_per_row = (n + MARKOV_STEP_TILE_ELEMS - 1) / MARKOV_STEP_TILE_ELEMS;
  size_t smem = SAMPLE_BLOCK * (sizeof(float) + sizeof(int));
  markov_step_partial_kernel<<<dim3(tiles_per_row, rows), SAMPLE_BLOCK, smem, stream>>>(
      base, bias, block_size, step, chains, req_map, partial_values,
      partial_indices, rows, n, tiles_per_row);
  if (!markov_pdl_supported_on_current_device()) {
    markov_step_finalize_kernel<<<rows, SAMPLE_BLOCK, smem, stream>>>(
        partial_values, partial_indices, out_tokens, sampled_tokens, block_size,
        step, rows, tiles_per_row);
    return cudaGetLastError();
  }

  cudaLaunchAttribute attr[1] = {};
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(rows);
  config.blockDim = dim3(SAMPLE_BLOCK);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  config.attrs = attr;
  config.numAttrs = 1;
  cudaLaunchKernelEx(&config, markov_step_finalize_kernel, partial_values,
                     partial_indices, out_tokens, sampled_tokens, block_size,
                     step, rows, tiles_per_row);
  return cudaGetLastError();
}

cudaError_t markov_step_top2_cuda(const __nv_bfloat16* base, const __nv_bfloat16* bias,
                                  int block_size, int step, int chains, int rows, int n,
                                  float* partial_v1, int* partial_i1, float* partial_v2,
                                  int* partial_i2, unsigned int* out_tokens,
                                  unsigned int* sampled_tokens, unsigned int* out_top2,
                                  cudaStream_t stream) {
  int tiles_per_row = (n + MARKOV_STEP_TILE_ELEMS - 1) / MARKOV_STEP_TILE_ELEMS;
  size_t smem = SAMPLE_BLOCK * 2 * (sizeof(float) + sizeof(int));
  markov_step_top2_partial_kernel<<<dim3(tiles_per_row, rows), SAMPLE_BLOCK, smem,
                                    stream>>>(base, bias, block_size, step, chains,
                                              partial_v1, partial_i1, partial_v2,
                                              partial_i2, rows, n, tiles_per_row);
  markov_step_top2_finalize_kernel<<<rows, SAMPLE_BLOCK, smem, stream>>>(
      partial_v1, partial_i1, partial_v2, partial_i2, out_tokens, sampled_tokens,
      out_top2, block_size, step, rows, tiles_per_row);
  return cudaGetLastError();
}

cudaError_t hedge_ladder_force_cuda(unsigned int* prev, unsigned int* sampled,
                                    const unsigned int* runners, const int* req_map,
                                    int n, int c, int j, int runner_stride,
                                    int block_size, int step, cudaStream_t stream) {
  int threads = 128;
  int blocks = (n + threads - 1) / threads;
  hedge_ladder_force_kernel<<<blocks, threads, 0, stream>>>(
      prev, sampled, runners, req_map, n, c, j, runner_stride, block_size, step);
  return cudaGetLastError();
}
}
