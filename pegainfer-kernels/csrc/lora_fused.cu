#include "common.cuh"
#include "shared/ffi_guard.cuh"
#include <cuda.h>
#include <cstdint>

__device__ __forceinline__ float block_reduce_sum(float value,
                                                  float *scratch) {
  int lane = threadIdx.x & (warpSize - 1);
  int warp = threadIdx.x / warpSize;
  int num_warps = (blockDim.x + warpSize - 1) / warpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < num_warps ? scratch[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  if (threadIdx.x == 0) {
    scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

static size_t lora_rank_smem_bytes(int rank) {
  int floats = rank < 32 ? 32 : rank;
  return static_cast<size_t>(floats) * sizeof(float);
}

template <int RANK>
__device__ __forceinline__ void compute_lora_rank_dot(
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ x,
    int rank,
    int in_dim,
    float *__restrict__ rank_buf) {
  if (RANK == 1) {
    float rank_val = 0.0f;
    for (int col = threadIdx.x; col < in_dim; col += blockDim.x) {
      rank_val += __bfloat162float(a[col]) * __bfloat162float(x[col]);
    }
    rank_val = block_reduce_sum(rank_val, rank_buf);
    if (threadIdx.x == 0) {
      rank_buf[0] = rank_val;
    }
    return;
  }

  int lane = threadIdx.x & (warpSize - 1);
  int warp = threadIdx.x / warpSize;
  int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  for (int r = warp; r < rank; r += num_warps) {
    float rank_val = 0.0f;
    const __nv_bfloat16 *a_row = a + static_cast<int64_t>(r) * in_dim;
    for (int col = lane; col < in_dim; col += warpSize) {
      rank_val += __bfloat162float(a_row[col]) * __bfloat162float(x[col]);
    }
    rank_val = warp_reduce_sum(rank_val);
    if (lane == 0) {
      rank_buf[r] = rank_val;
    }
  }
}

__device__ __forceinline__ void compute_lora_rank_dot_runtime(
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ x,
    int rank,
    int in_dim,
    float *__restrict__ rank_buf) {
  if (rank == 1) {
    float rank_val = 0.0f;
    for (int col = threadIdx.x; col < in_dim; col += blockDim.x) {
      rank_val += __bfloat162float(a[col]) * __bfloat162float(x[col]);
    }
    rank_val = block_reduce_sum(rank_val, rank_buf);
    if (threadIdx.x == 0) {
      rank_buf[0] = rank_val;
    }
    return;
  }

  int lane = threadIdx.x & (warpSize - 1);
  int warp = threadIdx.x / warpSize;
  int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  for (int r = warp; r < rank; r += num_warps) {
    float rank_val = 0.0f;
    const __nv_bfloat16 *a_row = a + static_cast<int64_t>(r) * in_dim;
    for (int col = lane; col < in_dim; col += warpSize) {
      rank_val += __bfloat162float(a_row[col]) * __bfloat162float(x[col]);
    }
    rank_val = warp_reduce_sum(rank_val);
    if (lane == 0) {
      rank_buf[r] = rank_val;
    }
  }
}

__global__ void lora_pack_b_rows_kernel(
    const __nv_bfloat16 *__restrict__ src,
    __nv_bfloat16 *__restrict__ dst,
    int rank,
    int max_rank,
    int out_dim) {
  int total = out_dim * rank;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int row = idx / rank;
    int col = idx - row * rank;
    dst[static_cast<int64_t>(row) * max_rank + col] = src[idx];
  }
}

__global__ void lora_decode_fused_delta_kernel(
    const __nv_bfloat16 *__restrict__ a_packed,
    const __nv_bfloat16 *__restrict__ b_packed,
    const float *__restrict__ scales,
    const int *__restrict__ token_slots,
    const __nv_bfloat16 *__restrict__ input,
    __nv_bfloat16 *__restrict__ out,
    int batch,
    int max_loras,
    int max_rank,
    int rank,
    int in_dim,
    int out_dim,
    int out_hidden_dim,
    int row_offset) {
  extern __shared__ float rank_buf[];

  int token = blockIdx.x;
  if (token >= batch) {
    return;
  }

  int slot = token_slots[token];
  if (slot < 0 || slot >= max_loras) {
    return;
  }

  float scale = scales[slot];
  if (scale == 0.0f) {
    return;
  }

  const __nv_bfloat16 *a =
      a_packed + (static_cast<int64_t>(slot) * max_rank * in_dim);
  const __nv_bfloat16 *b =
      b_packed + (static_cast<int64_t>(slot) * out_dim * max_rank);
  const __nv_bfloat16 *x =
      input + (static_cast<int64_t>(token) * in_dim);

  compute_lora_rank_dot_runtime(a, x, rank, in_dim, rank_buf);
  __syncthreads();

  for (int row = threadIdx.x; row < out_dim; row += blockDim.x) {
    float delta = 0.0f;
    const __nv_bfloat16 *b_row = b + static_cast<int64_t>(row) * max_rank;
    for (int r = 0; r < rank; ++r) {
      delta += __bfloat162float(b_row[r]) * rank_buf[r];
    }

    int out_idx = token * out_hidden_dim + row_offset + row;
    float base = __bfloat162float(out[out_idx]);
    out[out_idx] = __float2bfloat16(base + scale * delta);
  }
}

template <int RANK>
__global__ void lora_decode_fused_delta_rank_kernel(
    const __nv_bfloat16 *__restrict__ a_packed,
    const __nv_bfloat16 *__restrict__ b_packed,
    const float *__restrict__ scales,
    const int *__restrict__ token_slots,
    const __nv_bfloat16 *__restrict__ input,
    __nv_bfloat16 *__restrict__ out,
    int batch,
    int max_loras,
    int max_rank,
    int in_dim,
    int out_dim,
    int out_hidden_dim,
    int row_offset) {
  extern __shared__ float rank_buf[];

  int token = blockIdx.x;
  if (token >= batch) {
    return;
  }

  int slot = token_slots[token];
  if (slot < 0 || slot >= max_loras) {
    return;
  }

  float scale = scales[slot];
  if (scale == 0.0f) {
    return;
  }

  const __nv_bfloat16 *a =
      a_packed + (static_cast<int64_t>(slot) * max_rank * in_dim);
  const __nv_bfloat16 *b =
      b_packed + (static_cast<int64_t>(slot) * out_dim * max_rank);
  const __nv_bfloat16 *x =
      input + (static_cast<int64_t>(token) * in_dim);

  compute_lora_rank_dot<RANK>(a, x, RANK, in_dim, rank_buf);
  __syncthreads();

  for (int row = threadIdx.x; row < out_dim; row += blockDim.x) {
    float delta = 0.0f;
    const __nv_bfloat16 *b_row = b + static_cast<int64_t>(row) * max_rank;
    for (int r = 0; r < RANK; ++r) {
      delta += __bfloat162float(b_row[r]) * rank_buf[r];
    }

    int out_idx = token * out_hidden_dim + row_offset + row;
    float base = __bfloat162float(out[out_idx]);
    out[out_idx] = __float2bfloat16(base + scale * delta);
  }
}

__device__ void apply_lora_decode_projection(
    const __nv_bfloat16 *__restrict__ a_packed,
    const __nv_bfloat16 *__restrict__ b_packed,
    const float *__restrict__ scales,
    const int slot,
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ out,
    int token,
    int max_rank,
    int rank,
    int in_dim,
    int out_dim,
    int out_hidden_dim,
    float *__restrict__ rank_buf) {
  if (a_packed == nullptr || b_packed == nullptr || scales == nullptr ||
      out == nullptr || rank <= 0 || out_dim <= 0) {
    return;
  }

  float scale = scales[slot];
  if (scale == 0.0f) {
    return;
  }

  const __nv_bfloat16 *a =
      a_packed + (static_cast<int64_t>(slot) * max_rank * in_dim);
  const __nv_bfloat16 *b =
      b_packed + (static_cast<int64_t>(slot) * out_dim * max_rank);

  compute_lora_rank_dot_runtime(a, x, rank, in_dim, rank_buf);
  __syncthreads();

  for (int row = threadIdx.x; row < out_dim; row += blockDim.x) {
    float delta = 0.0f;
    const __nv_bfloat16 *b_row = b + static_cast<int64_t>(row) * max_rank;
    for (int r = 0; r < rank; ++r) {
      delta += __bfloat162float(b_row[r]) * rank_buf[r];
    }

    int out_idx = token * out_hidden_dim + row;
    float base = __bfloat162float(out[out_idx]);
    out[out_idx] = __float2bfloat16(base + scale * delta);
  }
  __syncthreads();
}

template <int RANK>
__device__ void apply_lora_decode_projection_rank(
    const __nv_bfloat16 *__restrict__ a_packed,
    const __nv_bfloat16 *__restrict__ b_packed,
    const float *__restrict__ scales,
    const int slot,
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ out,
    int token,
    int max_rank,
    int rank,
    int in_dim,
    int out_dim,
    int out_hidden_dim,
    float *__restrict__ rank_buf) {
  if (a_packed == nullptr || b_packed == nullptr || scales == nullptr ||
      out == nullptr || rank <= 0 || out_dim <= 0) {
    return;
  }

  float scale = scales[slot];
  if (scale == 0.0f) {
    return;
  }

  const __nv_bfloat16 *a =
      a_packed + (static_cast<int64_t>(slot) * max_rank * in_dim);
  const __nv_bfloat16 *b =
      b_packed + (static_cast<int64_t>(slot) * out_dim * max_rank);

  compute_lora_rank_dot<RANK>(a, x, RANK, in_dim, rank_buf);
  __syncthreads();

  for (int row = threadIdx.x; row < out_dim; row += blockDim.x) {
    float delta = 0.0f;
    const __nv_bfloat16 *b_row = b + static_cast<int64_t>(row) * max_rank;
    for (int r = 0; r < RANK; ++r) {
      delta += __bfloat162float(b_row[r]) * rank_buf[r];
    }

    int out_idx = token * out_hidden_dim + row;
    float base = __bfloat162float(out[out_idx]);
    out[out_idx] = __float2bfloat16(base + scale * delta);
  }
  __syncthreads();
}

__global__ void lora_decode_fused_delta_group3_kernel(
    const __nv_bfloat16 *__restrict__ a0,
    const __nv_bfloat16 *__restrict__ b0,
    const float *__restrict__ scales0,
    __nv_bfloat16 *__restrict__ out0,
    int rank0,
    int out_dim0,
    int out_hidden_dim0,
    const __nv_bfloat16 *__restrict__ a1,
    const __nv_bfloat16 *__restrict__ b1,
    const float *__restrict__ scales1,
    __nv_bfloat16 *__restrict__ out1,
    int rank1,
    int out_dim1,
    int out_hidden_dim1,
    const __nv_bfloat16 *__restrict__ a2,
    const __nv_bfloat16 *__restrict__ b2,
    const float *__restrict__ scales2,
    __nv_bfloat16 *__restrict__ out2,
    int rank2,
    int out_dim2,
    int out_hidden_dim2,
    const int *__restrict__ token_slots,
    const __nv_bfloat16 *__restrict__ input,
    int batch,
    int max_loras,
    int max_rank,
    int in_dim) {
  extern __shared__ float rank_buf[];

  int token = blockIdx.x;
  if (token >= batch) {
    return;
  }

  int slot = token_slots[token];
  if (slot < 0 || slot >= max_loras) {
    return;
  }

  const __nv_bfloat16 *x =
      input + (static_cast<int64_t>(token) * in_dim);

  apply_lora_decode_projection(a0, b0, scales0, slot, x, out0, token,
                               max_rank, rank0, in_dim, out_dim0,
                               out_hidden_dim0, rank_buf);
  apply_lora_decode_projection(a1, b1, scales1, slot, x, out1, token,
                               max_rank, rank1, in_dim, out_dim1,
                               out_hidden_dim1, rank_buf);
  apply_lora_decode_projection(a2, b2, scales2, slot, x, out2, token,
                               max_rank, rank2, in_dim, out_dim2,
                               out_hidden_dim2, rank_buf);
}

template <int RANK>
__global__ void lora_decode_fused_delta_group3_rank_kernel(
    const __nv_bfloat16 *__restrict__ a0,
    const __nv_bfloat16 *__restrict__ b0,
    const float *__restrict__ scales0,
    __nv_bfloat16 *__restrict__ out0,
    int rank0,
    int out_dim0,
    int out_hidden_dim0,
    const __nv_bfloat16 *__restrict__ a1,
    const __nv_bfloat16 *__restrict__ b1,
    const float *__restrict__ scales1,
    __nv_bfloat16 *__restrict__ out1,
    int rank1,
    int out_dim1,
    int out_hidden_dim1,
    const __nv_bfloat16 *__restrict__ a2,
    const __nv_bfloat16 *__restrict__ b2,
    const float *__restrict__ scales2,
    __nv_bfloat16 *__restrict__ out2,
    int rank2,
    int out_dim2,
    int out_hidden_dim2,
    const int *__restrict__ token_slots,
    const __nv_bfloat16 *__restrict__ input,
    int batch,
    int max_loras,
    int max_rank,
    int in_dim) {
  extern __shared__ float rank_buf[];

  int token = blockIdx.x;
  if (token >= batch) {
    return;
  }

  int slot = token_slots[token];
  if (slot < 0 || slot >= max_loras) {
    return;
  }

  const __nv_bfloat16 *x =
      input + (static_cast<int64_t>(token) * in_dim);

  apply_lora_decode_projection_rank<RANK>(
      a0, b0, scales0, slot, x, out0, token, max_rank, rank0, in_dim,
      out_dim0, out_hidden_dim0, rank_buf);
  apply_lora_decode_projection_rank<RANK>(
      a1, b1, scales1, slot, x, out1, token, max_rank, rank1, in_dim,
      out_dim1, out_hidden_dim1, rank_buf);
  apply_lora_decode_projection_rank<RANK>(
      a2, b2, scales2, slot, x, out2, token, max_rank, rank2, in_dim,
      out_dim2, out_hidden_dim2, rank_buf);
}

static bool valid_lora_group_projection(bool has,
                                        const __nv_bfloat16 *a,
                                        const __nv_bfloat16 *b,
                                        const float *scales,
                                        const __nv_bfloat16 *out,
                                        int rank,
                                        int out_dim,
                                        int out_hidden_dim,
                                        int max_rank) {
  if (!has) {
    return true;
  }
  return a != nullptr && b != nullptr && scales != nullptr && out != nullptr &&
         rank > 0 && rank <= max_rank && out_dim > 0 &&
         out_hidden_dim >= out_dim;
}

template <int RANK>
static CUresult launch_lora_decode_fused_delta_rank(
    const __nv_bfloat16 *a_packed,
    const __nv_bfloat16 *b_packed,
    const float *scales,
    const int *token_slots,
    const __nv_bfloat16 *input,
    __nv_bfloat16 *out,
    int batch,
    int max_loras,
    int max_rank,
    int in_dim,
    int out_dim,
    int out_hidden_dim,
    int row_offset,
    cudaStream_t stream) {
  dim3 block(256);
  dim3 grid(batch);
  size_t smem_bytes = lora_rank_smem_bytes(RANK);
  lora_decode_fused_delta_rank_kernel<RANK><<<grid, block, smem_bytes, stream>>>(
      a_packed, b_packed, scales, token_slots, input, out, batch, max_loras,
      max_rank, in_dim, out_dim, out_hidden_dim, row_offset);
  return map_cuda_error(cudaGetLastError());
}

static CUresult launch_lora_decode_fused_delta(
    const __nv_bfloat16 *a_packed,
    const __nv_bfloat16 *b_packed,
    const float *scales,
    const int *token_slots,
    const __nv_bfloat16 *input,
    __nv_bfloat16 *out,
    int batch,
    int max_loras,
    int max_rank,
    int rank,
    int in_dim,
    int out_dim,
    int out_hidden_dim,
    int row_offset,
    cudaStream_t stream) {
  switch (rank) {
  case 1:
    return launch_lora_decode_fused_delta_rank<1>(
        a_packed, b_packed, scales, token_slots, input, out, batch, max_loras,
        max_rank, in_dim, out_dim, out_hidden_dim, row_offset, stream);
  case 8:
    return launch_lora_decode_fused_delta_rank<8>(
        a_packed, b_packed, scales, token_slots, input, out, batch, max_loras,
        max_rank, in_dim, out_dim, out_hidden_dim, row_offset, stream);
  case 16:
    return launch_lora_decode_fused_delta_rank<16>(
        a_packed, b_packed, scales, token_slots, input, out, batch, max_loras,
        max_rank, in_dim, out_dim, out_hidden_dim, row_offset, stream);
  case 32:
    return launch_lora_decode_fused_delta_rank<32>(
        a_packed, b_packed, scales, token_slots, input, out, batch, max_loras,
        max_rank, in_dim, out_dim, out_hidden_dim, row_offset, stream);
  case 64:
    return launch_lora_decode_fused_delta_rank<64>(
        a_packed, b_packed, scales, token_slots, input, out, batch, max_loras,
        max_rank, in_dim, out_dim, out_hidden_dim, row_offset, stream);
  case 128:
    return launch_lora_decode_fused_delta_rank<128>(
        a_packed, b_packed, scales, token_slots, input, out, batch, max_loras,
        max_rank, in_dim, out_dim, out_hidden_dim, row_offset, stream);
  case 256:
    return launch_lora_decode_fused_delta_rank<256>(
        a_packed, b_packed, scales, token_slots, input, out, batch, max_loras,
        max_rank, in_dim, out_dim, out_hidden_dim, row_offset, stream);
  case 320:
    return launch_lora_decode_fused_delta_rank<320>(
        a_packed, b_packed, scales, token_slots, input, out, batch, max_loras,
        max_rank, in_dim, out_dim, out_hidden_dim, row_offset, stream);
  case 512:
    return launch_lora_decode_fused_delta_rank<512>(
        a_packed, b_packed, scales, token_slots, input, out, batch, max_loras,
        max_rank, in_dim, out_dim, out_hidden_dim, row_offset, stream);
  default:
    dim3 block(256);
    dim3 grid(batch);
    size_t smem_bytes = lora_rank_smem_bytes(rank);
    lora_decode_fused_delta_kernel<<<grid, block, smem_bytes, stream>>>(
        a_packed, b_packed, scales, token_slots, input, out, batch, max_loras,
        max_rank, rank, in_dim, out_dim, out_hidden_dim, row_offset);
    return map_cuda_error(cudaGetLastError());
  }
}

template <int RANK>
static CUresult launch_lora_decode_fused_delta_group3_rank(
    const __nv_bfloat16 *a0,
    const __nv_bfloat16 *b0,
    const float *scales0,
    __nv_bfloat16 *out0,
    int rank0,
    int out_dim0,
    int out_hidden_dim0,
    const __nv_bfloat16 *a1,
    const __nv_bfloat16 *b1,
    const float *scales1,
    __nv_bfloat16 *out1,
    int rank1,
    int out_dim1,
    int out_hidden_dim1,
    const __nv_bfloat16 *a2,
    const __nv_bfloat16 *b2,
    const float *scales2,
    __nv_bfloat16 *out2,
    int rank2,
    int out_dim2,
    int out_hidden_dim2,
    const int *token_slots,
    const __nv_bfloat16 *input,
    int batch,
    int max_loras,
    int max_rank,
    int in_dim,
    cudaStream_t stream) {
  dim3 block(256);
  dim3 grid(batch);
  size_t smem_bytes = lora_rank_smem_bytes(RANK);
  lora_decode_fused_delta_group3_rank_kernel<RANK>
      <<<grid, block, smem_bytes, stream>>>(
          a0, b0, scales0, out0, rank0, out_dim0, out_hidden_dim0, a1, b1,
          scales1, out1, rank1, out_dim1, out_hidden_dim1, a2, b2, scales2,
          out2, rank2, out_dim2, out_hidden_dim2, token_slots, input, batch,
          max_loras, max_rank, in_dim);
  return map_cuda_error(cudaGetLastError());
}

static CUresult launch_lora_decode_fused_delta_group3(
    const __nv_bfloat16 *a0,
    const __nv_bfloat16 *b0,
    const float *scales0,
    __nv_bfloat16 *out0,
    int rank0,
    int out_dim0,
    int out_hidden_dim0,
    const __nv_bfloat16 *a1,
    const __nv_bfloat16 *b1,
    const float *scales1,
    __nv_bfloat16 *out1,
    int rank1,
    int out_dim1,
    int out_hidden_dim1,
    const __nv_bfloat16 *a2,
    const __nv_bfloat16 *b2,
    const float *scales2,
    __nv_bfloat16 *out2,
    int rank2,
    int out_dim2,
    int out_hidden_dim2,
    const int *token_slots,
    const __nv_bfloat16 *input,
    int batch,
    int max_loras,
    int max_rank,
    int in_dim,
    int shared_rank,
    cudaStream_t stream) {
  switch (shared_rank) {
  case 1:
    return launch_lora_decode_fused_delta_group3_rank<1>(
        a0, b0, scales0, out0, rank0, out_dim0, out_hidden_dim0, a1, b1,
        scales1, out1, rank1, out_dim1, out_hidden_dim1, a2, b2, scales2,
        out2, rank2, out_dim2, out_hidden_dim2, token_slots, input, batch,
        max_loras, max_rank, in_dim, stream);
  case 8:
    return launch_lora_decode_fused_delta_group3_rank<8>(
        a0, b0, scales0, out0, rank0, out_dim0, out_hidden_dim0, a1, b1,
        scales1, out1, rank1, out_dim1, out_hidden_dim1, a2, b2, scales2,
        out2, rank2, out_dim2, out_hidden_dim2, token_slots, input, batch,
        max_loras, max_rank, in_dim, stream);
  case 16:
    return launch_lora_decode_fused_delta_group3_rank<16>(
        a0, b0, scales0, out0, rank0, out_dim0, out_hidden_dim0, a1, b1,
        scales1, out1, rank1, out_dim1, out_hidden_dim1, a2, b2, scales2,
        out2, rank2, out_dim2, out_hidden_dim2, token_slots, input, batch,
        max_loras, max_rank, in_dim, stream);
  case 32:
    return launch_lora_decode_fused_delta_group3_rank<32>(
        a0, b0, scales0, out0, rank0, out_dim0, out_hidden_dim0, a1, b1,
        scales1, out1, rank1, out_dim1, out_hidden_dim1, a2, b2, scales2,
        out2, rank2, out_dim2, out_hidden_dim2, token_slots, input, batch,
        max_loras, max_rank, in_dim, stream);
  case 64:
    return launch_lora_decode_fused_delta_group3_rank<64>(
        a0, b0, scales0, out0, rank0, out_dim0, out_hidden_dim0, a1, b1,
        scales1, out1, rank1, out_dim1, out_hidden_dim1, a2, b2, scales2,
        out2, rank2, out_dim2, out_hidden_dim2, token_slots, input, batch,
        max_loras, max_rank, in_dim, stream);
  case 128:
    return launch_lora_decode_fused_delta_group3_rank<128>(
        a0, b0, scales0, out0, rank0, out_dim0, out_hidden_dim0, a1, b1,
        scales1, out1, rank1, out_dim1, out_hidden_dim1, a2, b2, scales2,
        out2, rank2, out_dim2, out_hidden_dim2, token_slots, input, batch,
        max_loras, max_rank, in_dim, stream);
  case 256:
    return launch_lora_decode_fused_delta_group3_rank<256>(
        a0, b0, scales0, out0, rank0, out_dim0, out_hidden_dim0, a1, b1,
        scales1, out1, rank1, out_dim1, out_hidden_dim1, a2, b2, scales2,
        out2, rank2, out_dim2, out_hidden_dim2, token_slots, input, batch,
        max_loras, max_rank, in_dim, stream);
  case 320:
    return launch_lora_decode_fused_delta_group3_rank<320>(
        a0, b0, scales0, out0, rank0, out_dim0, out_hidden_dim0, a1, b1,
        scales1, out1, rank1, out_dim1, out_hidden_dim1, a2, b2, scales2,
        out2, rank2, out_dim2, out_hidden_dim2, token_slots, input, batch,
        max_loras, max_rank, in_dim, stream);
  case 512:
    return launch_lora_decode_fused_delta_group3_rank<512>(
        a0, b0, scales0, out0, rank0, out_dim0, out_hidden_dim0, a1, b1,
        scales1, out1, rank1, out_dim1, out_hidden_dim1, a2, b2, scales2,
        out2, rank2, out_dim2, out_hidden_dim2, token_slots, input, batch,
        max_loras, max_rank, in_dim, stream);
  default:
    dim3 block(256);
    dim3 grid(batch);
    size_t smem_bytes = lora_rank_smem_bytes(shared_rank);
    lora_decode_fused_delta_group3_kernel<<<grid, block, smem_bytes, stream>>>(
        a0, b0, scales0, out0, rank0, out_dim0, out_hidden_dim0, a1, b1,
        scales1, out1, rank1, out_dim1, out_hidden_dim1, a2, b2, scales2,
        out2, rank2, out_dim2, out_hidden_dim2, token_slots, input, batch,
        max_loras, max_rank, in_dim);
    return map_cuda_error(cudaGetLastError());
  }
}

extern "C" {

CUresult lora_pack_b_rows_cuda(
    const __nv_bfloat16 *src,
    __nv_bfloat16 *dst,
    int rank,
    int max_rank,
    int out_dim,
    cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || rank <= 0 || max_rank <= 0 ||
      rank > max_rank || out_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  int total = out_dim * rank;
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);
  lora_pack_b_rows_kernel<<<grid, block, 0, stream>>>(
      src, dst, rank, max_rank, out_dim);
  return map_cuda_error(cudaGetLastError());
}

CUresult lora_decode_fused_delta_cuda(
    const __nv_bfloat16 *a_packed,
    const __nv_bfloat16 *b_packed,
    const float *scales,
    const int *token_slots,
    const __nv_bfloat16 *input,
    __nv_bfloat16 *out,
    int batch,
    int max_loras,
    int max_rank,
    int rank,
    int in_dim,
    int out_dim,
    int out_hidden_dim,
    int row_offset,
    cudaStream_t stream) {
  if (a_packed == nullptr || b_packed == nullptr || scales == nullptr ||
      token_slots == nullptr || input == nullptr || out == nullptr ||
      batch <= 0 || max_loras <= 0 || max_rank <= 0 || rank <= 0 ||
      rank > max_rank || in_dim <= 0 || out_dim <= 0 || out_hidden_dim <= 0 ||
      row_offset < 0 || row_offset + out_dim > out_hidden_dim) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  return launch_lora_decode_fused_delta(
      a_packed, b_packed, scales, token_slots, input, out, batch, max_loras,
      max_rank, rank, in_dim, out_dim, out_hidden_dim, row_offset, stream);
}

CUresult lora_decode_fused_delta_group3_cuda(
    const __nv_bfloat16 *a0,
    const __nv_bfloat16 *b0,
    const float *scales0,
    __nv_bfloat16 *out0,
    int rank0,
    int out_dim0,
    int out_hidden_dim0,
    const __nv_bfloat16 *a1,
    const __nv_bfloat16 *b1,
    const float *scales1,
    __nv_bfloat16 *out1,
    int rank1,
    int out_dim1,
    int out_hidden_dim1,
    const __nv_bfloat16 *a2,
    const __nv_bfloat16 *b2,
    const float *scales2,
    __nv_bfloat16 *out2,
    int rank2,
    int out_dim2,
    int out_hidden_dim2,
    const int *token_slots,
    const __nv_bfloat16 *input,
    int batch,
    int max_loras,
    int max_rank,
    int in_dim,
    cudaStream_t stream) {
  bool has0 = a0 != nullptr || b0 != nullptr || scales0 != nullptr ||
              out0 != nullptr || rank0 != 0 || out_dim0 != 0 ||
              out_hidden_dim0 != 0;
  bool has1 = a1 != nullptr || b1 != nullptr || scales1 != nullptr ||
              out1 != nullptr || rank1 != 0 || out_dim1 != 0 ||
              out_hidden_dim1 != 0;
  bool has2 = a2 != nullptr || b2 != nullptr || scales2 != nullptr ||
              out2 != nullptr || rank2 != 0 || out_dim2 != 0 ||
              out_hidden_dim2 != 0;
  if (token_slots == nullptr || input == nullptr || batch <= 0 ||
      max_loras <= 0 || max_rank <= 0 || in_dim <= 0 ||
      (!has0 && !has1 && !has2) ||
      !valid_lora_group_projection(has0, a0, b0, scales0, out0, rank0,
                                   out_dim0, out_hidden_dim0, max_rank) ||
      !valid_lora_group_projection(has1, a1, b1, scales1, out1, rank1,
                                   out_dim1, out_hidden_dim1, max_rank) ||
      !valid_lora_group_projection(has2, a2, b2, scales2, out2, rank2,
                                   out_dim2, out_hidden_dim2, max_rank)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  int shared_rank = rank0;
  if (rank1 > shared_rank) {
    shared_rank = rank1;
  }
  if (rank2 > shared_rank) {
    shared_rank = rank2;
  }
  return launch_lora_decode_fused_delta_group3(
      a0, b0, scales0, out0, rank0, out_dim0, out_hidden_dim0, a1, b1,
      scales1, out1, rank1, out_dim1, out_hidden_dim1, a2, b2, scales2,
      out2, rank2, out_dim2, out_hidden_dim2, token_slots, input, batch,
      max_loras, max_rank, in_dim, shared_rank, stream);
}

} // extern "C"
