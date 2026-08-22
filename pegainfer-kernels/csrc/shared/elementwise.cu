#include "common.cuh"
#include <climits>
#include <cstdint>
#include <cuda.h>
#include <math_constants.h>

// ============================================================================
// Element-wise add: out = a + b (bf16, computed in f32)
// ============================================================================

__global__ void add_kernel(
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    __nv_bfloat16 *__restrict__ out,
    int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += gridDim.x * blockDim.x) {
    float va = __bfloat162float(a[idx]);
    float vb = __bfloat162float(b[idx]);
    out[idx] = __float2bfloat16(va + vb);
  }
}

__global__ void add_scaled_bf16_kernel(
    const __nv_bfloat16 *__restrict__ routed,
    float scale,
    const __nv_bfloat16 *__restrict__ shared,
    __nv_bfloat16 *__restrict__ out,
    int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += gridDim.x * blockDim.x) {
    // Match vLLM's two BF16 operations: scale the routed output in place,
    // then add the shared expert.
    __nv_bfloat16 scaled = __float2bfloat16(
        __bfloat162float(routed[idx]) * scale);
    out[idx] = __float2bfloat16(
        __bfloat162float(shared[idx]) + __bfloat162float(scaled));
  }
}

__global__ void scaled_add_rows_kernel(
    const __nv_bfloat16 *__restrict__ delta,
    float scale,
    __nv_bfloat16 *__restrict__ out,
    int out_hidden_dim,
    int row_offset,
    int rows,
    int seq_len) {
  for (int token = blockIdx.y * blockDim.y + threadIdx.y;
       token < seq_len;
       token += gridDim.y * blockDim.y) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += gridDim.x * blockDim.x) {
      int delta_idx = token * rows + row;
      int out_idx = token * out_hidden_dim + row_offset + row;
      float base = __bfloat162float(out[out_idx]);
      float add = __bfloat162float(delta[delta_idx]) * scale;
      out[out_idx] = __float2bfloat16(base + add);
    }
  }
}

__global__ void gather_hidden_tokens_kernel(
    const __nv_bfloat16 *__restrict__ input,
    const int *__restrict__ token_indices,
    __nv_bfloat16 *__restrict__ out,
    int hidden_dim,
    int token_count,
    int input_seq_len) {
  int total = hidden_dim * token_count;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    int token = idx / hidden_dim;
    int row = idx % hidden_dim;
    int src_token = token_indices[token];
    if (src_token < 0 || src_token >= input_seq_len) {
      continue;
    }
    out[idx] = input[(size_t)src_token * hidden_dim + row];
  }
}

__global__ void copy_hidden_rows_kernel(
    const __nv_bfloat16 *__restrict__ src,
    __nv_bfloat16 *__restrict__ dst,
    int src_hidden_dim,
    int dst_hidden_dim,
    int row_offset,
    int rows,
    int seq_len) {
  int total = rows * seq_len;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    int token = idx / rows;
    int row = idx % rows;
    dst[(size_t)token * dst_hidden_dim + row_offset + row] =
        src[(size_t)token * src_hidden_dim + row];
  }
}

// Inverse of copy_hidden_rows_kernel: extract a column window
// [col_offset, col_offset + rows) of each token's wide src row into a
// compact dst row (the packed-projection split, #812).
__global__ void extract_hidden_rows_kernel(
    const __nv_bfloat16 *__restrict__ src,
    __nv_bfloat16 *__restrict__ dst,
    int src_hidden_dim,
    int dst_hidden_dim,
    int col_offset,
    int rows,
    int seq_len) {
  int total = rows * seq_len;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    int token = idx / rows;
    int row = idx % rows;
    dst[(size_t)token * dst_hidden_dim + row] =
        src[(size_t)token * src_hidden_dim + col_offset + row];
  }
}

__global__ void mask_position_zero_rows_kernel(
    const __nv_bfloat16 *__restrict__ src,
    const uint32_t *__restrict__ positions,
    __nv_bfloat16 *__restrict__ dst,
    int hidden_dim,
    int rows) {
  int total = hidden_dim * rows;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    int row = idx / hidden_dim;
    dst[idx] = positions[row] == 0 ? __float2bfloat16(0.0f) : src[idx];
  }
}

__global__ void copy_hidden_token_range_kernel(
    const __nv_bfloat16 *__restrict__ src,
    __nv_bfloat16 *__restrict__ dst,
    int hidden_dim,
    int src_token_offset,
    int dst_token_offset,
    int token_count) {
  int total = hidden_dim * token_count;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    int token = idx / hidden_dim;
    int row = idx % hidden_dim;
    dst[(size_t)(dst_token_offset + token) * hidden_dim + row] =
        src[(size_t)(src_token_offset + token) * hidden_dim + row];
  }
}

__global__ void scaled_add_rows_indexed_kernel(
    const __nv_bfloat16 *__restrict__ delta,
    float scale,
    const int *__restrict__ token_indices,
    __nv_bfloat16 *__restrict__ out,
    int out_hidden_dim,
    int row_offset,
    int rows,
    int token_count,
    int out_seq_len) {
  for (int token = blockIdx.y * blockDim.y + threadIdx.y;
       token < token_count;
       token += gridDim.y * blockDim.y) {
    int out_token = token_indices[token];
    if (out_token < 0 || out_token >= out_seq_len) {
      continue;
    }
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += gridDim.x * blockDim.x) {
      int delta_idx = token * rows + row;
      // out_token * out_hidden_dim overflows i32 for large row pools (the K3
      // paged-KV slab exceeds 2^31 elements), so the element index is size_t.
      size_t out_idx =
          (size_t)out_token * out_hidden_dim + row_offset + row;
      float base = __bfloat162float(out[out_idx]);
      float add = __bfloat162float(delta[delta_idx]) * scale;
      out[out_idx] = __float2bfloat16(base + add);
    }
  }
}

__global__ void store_rows_indexed_kernel(
    const __nv_bfloat16 *__restrict__ src,
    const int *__restrict__ token_indices,
    __nv_bfloat16 *__restrict__ out,
    int out_hidden_dim,
    int row_offset,
    int rows,
    int token_count,
    int out_seq_len) {
  for (int token = blockIdx.y * blockDim.y + threadIdx.y;
       token < token_count;
       token += gridDim.y * blockDim.y) {
    int out_token = token_indices[token];
    if (out_token < 0 || out_token >= out_seq_len) {
      continue;
    }
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += gridDim.x * blockDim.x) {
      int src_idx = token * rows + row;
      // out_token * out_hidden_dim overflows i32 for large row pools (the K3
      // paged-KV slab exceeds 2^31 elements), so the element index is size_t.
      size_t out_idx =
          (size_t)out_token * out_hidden_dim + row_offset + row;
      out[out_idx] = src[src_idx];
    }
  }
}

// ============================================================================
// Type conversion helpers for deterministic decode collectives.
// ============================================================================

__global__ void bf16_to_f32_kernel(
    const __nv_bfloat16 *__restrict__ input,
    float *__restrict__ output,
    int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += gridDim.x * blockDim.x) {
    output[idx] = __bfloat162float(input[idx]);
  }
}

__global__ void f32_to_bf16_kernel(
    const float *__restrict__ input,
    __nv_bfloat16 *__restrict__ output,
    int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += gridDim.x * blockDim.x) {
    output[idx] = __float2bfloat16(input[idx]);
  }
}

__global__ void scale_f32_kernel(float *__restrict__ values, float scale, int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += gridDim.x * blockDim.x) {
    values[idx] *= scale;
  }
}

__global__ void accumulate_bf16_token_scaled_to_f32_kernel(
    const __nv_bfloat16 *__restrict__ token,
    float scale,
    float *__restrict__ out,
    int hidden_dim,
    int token_idx,
    int seq_len) {
  if (token_idx < 0 || token_idx >= seq_len) {
    return;
  }
  int base = token_idx * hidden_dim;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < hidden_dim;
       idx += gridDim.x * blockDim.x) {
    out[base + idx] += __bfloat162float(token[idx]) * scale;
  }
}

__global__ void repeat_f32_rows_for_reduce_scatter_kernel(
    const float *__restrict__ local,
    float *__restrict__ repeated,
    int local_elems,
    int world_size) {
  int total = local_elems * world_size;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    repeated[idx] = local[idx % local_elems];
  }
}

// ============================================================================
// SiLU-mul from separate gate/up buffers: out = silu(gate) * up
// Matches Triton silu_mul_kernel rounding: silu computed in f32,
// cast to bf16, then multiplied with up in bf16→f32.
// ============================================================================

__global__ void silu_mul_kernel(
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up,
    __nv_bfloat16 *__restrict__ out,
    int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += gridDim.x * blockDim.x) {
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float silu_g = g / (1.0f + expf(-g));
    // Match Triton rounding: silu result cast to bf16 before multiply
    out[idx] = __float2bfloat16(__bfloat162float(__float2bfloat16(silu_g)) * u);
  }
}

// ============================================================================
// GELU-tanh-mul from separate gate/up buffers: out = gelu_tanh(gate) * up
// (Gemma 4 MLP, hidden_activation "gelu_pytorch_tanh" — the tanh
// approximation, not erf). Matches the HF op sequence rounding: act_fn(gate)
// materializes a bf16 tensor before the elementwise multiply, so the
// activation is computed in f32, cast to bf16, then multiplied in f32 and
// rounded once more.
// ============================================================================

__global__ void gelu_tanh_mul_kernel(
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up,
    __nv_bfloat16 *__restrict__ out,
    int n) {
  // sqrt(2/pi), the gelu_pytorch_tanh constant.
  const float kSqrt2OverPi = 0.7978845608028654f;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += gridDim.x * blockDim.x) {
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float inner = kSqrt2OverPi * (g + 0.044715f * g * g * g);
    float gelu_g = 0.5f * g * (1.0f + tanhf(inner));
    out[idx] = __float2bfloat16(__bfloat162float(__float2bfloat16(gelu_g)) * u);
  }
}

// ============================================================================
// In-place final-logit softcap: buf[i] = bf16(cap * tanh(f32(buf[i]) / cap)).
// Gemma 4 declares final_logit_softcapping (30.0 at every published size) and
// applies it to the LM head output in the compute dtype; f32 internal with a
// single rounding matches the reference's bf16 elementwise chain.
// ============================================================================

__global__ void softcap_bf16_kernel(
    __nv_bfloat16 *__restrict__ buf,
    float cap,
    int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += gridDim.x * blockDim.x) {
    float x = __bfloat162float(buf[idx]);
    buf[idx] = __float2bfloat16(cap * tanhf(x / cap));
  }
}

// ============================================================================
// In-place logit suppression. Ids reach here only through the Rust
// `SuppressIds` type, which refuses any id outside the head at upload.
// ============================================================================

__global__ void suppress_logits_bf16_kernel(
    __nv_bfloat16 *__restrict__ logits,
    const uint32_t *__restrict__ ids,
    int vocab,
    int id_count,
    int total) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    int row = idx / id_count;
    int id_slot = idx % id_count;
    logits[(size_t)row * vocab + ids[id_slot]] =
        __float2bfloat16(-CUDART_INF_F);
  }
}

// ============================================================================
// In-place multiply by a host scalar: buf[i] = bf16(f32(buf[i]) * scale).
// Serves Gemma 4's per-layer layer_scalar, a [1] weight the model reads to
// the host at load; f32 compute with a single rounding matches HF's bf16
// elementwise multiply.
// ============================================================================

__global__ void scale_bf16_kernel(
    __nv_bfloat16 *__restrict__ buf,
    float scale,
    int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += gridDim.x * blockDim.x) {
    buf[idx] = __float2bfloat16(__bfloat162float(buf[idx]) * scale);
  }
}

// ============================================================================
// Embedding lookup: out = embed[token_id, :]
// Reads token_id from token_id[0] (CUDA Graph safe).
// ============================================================================

__global__ void embedding_decode_kernel(
    const __nv_bfloat16 *__restrict__ embed,
    const uint32_t *__restrict__ token_id,
    __nv_bfloat16 *__restrict__ out,
    int hidden_size) {
  uint32_t token_idx = __ldg(&token_id[0]);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < hidden_size;
       idx += gridDim.x * blockDim.x) {
    out[idx] = embed[(size_t)token_idx * hidden_size + idx];
  }
}

// ============================================================================
// Batched embedding lookup: out[:, i] = embed[token_ids[i], :]
// Column-major output: [hidden_size, seq_len].
// ============================================================================

__global__ void embedding_batched_kernel(
    const __nv_bfloat16 *__restrict__ embed,
    const uint32_t *__restrict__ token_ids,
    __nv_bfloat16 *__restrict__ out,
    int hidden_size, int seq_len) {
  int total = hidden_size * seq_len;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    int token_offset = idx / hidden_size;
    int dim_offset = idx % hidden_size;
    uint32_t token_id = token_ids[token_offset];
    out[idx] = embed[(size_t)token_id * hidden_size + dim_offset];
  }
}

// Vectorized batched embedding lookup: grid.x indexes the token, grid.y
// splits the row into block-wide segments of 16-byte vectors, token id
// loaded once per block. The scalar kernel above moves 2 bytes per thread
// per access with a div+mod and a token_ids reload per element, which caps
// it near a quarter of DRAM bandwidth; the vectorized row copy streams at
// full width, and the 2D grid keeps enough blocks in flight at small token
// counts (decode) that the launch floor does not regress. Requires
// hidden_size % 8 == 0 and 16-byte-aligned base pointers — the launcher
// falls back to the scalar kernel otherwise.
__global__ void embedding_batched_vec4_kernel(
    const uint4 *__restrict__ embed,
    const uint32_t *__restrict__ token_ids,
    uint4 *__restrict__ out,
    int hidden_vec, int seq_len) {
  const int token = blockIdx.x;
  if (token >= seq_len) {
    return;
  }
  const uint32_t token_id = __ldg(&token_ids[token]);
  const uint4 *__restrict__ src = embed + (size_t)token_id * hidden_vec;
  uint4 *__restrict__ dst = out + (size_t)token * hidden_vec;
  for (int v = blockIdx.y * blockDim.x + threadIdx.x; v < hidden_vec;
       v += gridDim.y * blockDim.x) {
    dst[v] = src[v];
  }
}

// ============================================================================
// Tensor-parallel vocab-sharded embedding lookup.
//
// Each rank owns [vocab_start, vocab_start + part_vocab_size). Tokens outside
// the local shard write zeros. An all-reduce over ranks recovers the full
// embedding result, matching the official ParallelEmbedding implementation.
// Output layout remains [seq_len, hidden_size].
// ============================================================================

__global__ void embedding_batched_vocab_shard_kernel(
    const __nv_bfloat16 *__restrict__ embed,
    const uint32_t *__restrict__ token_ids,
    __nv_bfloat16 *__restrict__ out,
    int hidden_size, int seq_len, uint32_t vocab_start,
    uint32_t part_vocab_size) {
  int total = hidden_size * seq_len;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    int token_offset = idx / hidden_size;
    int dim_offset = idx % hidden_size;
    uint32_t token_id = token_ids[token_offset];
    if (token_id >= vocab_start && token_id < vocab_start + part_vocab_size) {
      uint32_t local_token_id = token_id - vocab_start;
      out[idx] = embed[(size_t)local_token_id * hidden_size + dim_offset];
    } else {
      out[idx] = __float2bfloat16(0.0f);
    }
  }
}

// Vectorized variant of the vocab-sharded lookup; same layout contract as
// embedding_batched_vec4_kernel, with non-local tokens writing zero vectors.
__global__ void embedding_batched_vocab_shard_vec4_kernel(
    const uint4 *__restrict__ embed,
    const uint32_t *__restrict__ token_ids,
    uint4 *__restrict__ out,
    int hidden_vec, int seq_len, uint32_t vocab_start,
    uint32_t part_vocab_size) {
  const int token = blockIdx.x;
  if (token >= seq_len) {
    return;
  }
  const uint32_t token_id = __ldg(&token_ids[token]);
  uint4 *__restrict__ dst = out + (size_t)token * hidden_vec;
  if (token_id >= vocab_start && token_id < vocab_start + part_vocab_size) {
    const uint4 *__restrict__ src =
        embed + (size_t)(token_id - vocab_start) * hidden_vec;
    for (int v = blockIdx.y * blockDim.x + threadIdx.x; v < hidden_vec;
         v += gridDim.y * blockDim.x) {
      dst[v] = src[v];
    }
  } else {
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    for (int v = blockIdx.y * blockDim.x + threadIdx.x; v < hidden_vec;
         v += gridDim.y * blockDim.x) {
      dst[v] = zero;
    }
  }
}

extern "C" {

CUresult add_cuda(
    const __nv_bfloat16 *a, const __nv_bfloat16 *b,
    __nv_bfloat16 *out, int n, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  add_kernel<<<grid, block, 0, stream>>>(a, b, out, n);
  return (CUresult)cudaGetLastError();
}

CUresult add_scaled_bf16_cuda(
    const __nv_bfloat16 *routed, float scale,
    const __nv_bfloat16 *shared, __nv_bfloat16 *out,
    int n, cudaStream_t stream) {
  if (routed == nullptr || shared == nullptr || out == nullptr ||
      !isfinite(scale) || n <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int block = 256;
  int grid = (n + block - 1) / block;
  add_scaled_bf16_kernel<<<grid, block, 0, stream>>>(
      routed, scale, shared, out, n);
  return (CUresult)cudaGetLastError();
}

CUresult scaled_add_rows_cuda(
    const __nv_bfloat16 *delta,
    float scale,
    __nv_bfloat16 *out,
    int out_hidden_dim,
    int row_offset,
    int rows,
    int seq_len,
    cudaStream_t stream) {
  if (delta == nullptr || out == nullptr || out_hidden_dim <= 0 ||
      row_offset < 0 || rows <= 0 || seq_len <= 0 ||
      row_offset + rows > out_hidden_dim) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dim3 block(32, 8);
  int grid_x = (rows + block.x - 1) / block.x;
  int grid_y = (seq_len + block.y - 1) / block.y;
  grid_x = grid_x > 65535 ? 65535 : grid_x;
  grid_y = grid_y > 65535 ? 65535 : grid_y;
  dim3 grid(grid_x, grid_y);
  scaled_add_rows_kernel<<<grid, block, 0, stream>>>(
      delta, scale, out, out_hidden_dim, row_offset, rows, seq_len);
  return (CUresult)cudaGetLastError();
}

CUresult gather_hidden_tokens_cuda(
    const __nv_bfloat16 *input,
    const int *token_indices,
    __nv_bfloat16 *out,
    int hidden_dim,
    int token_count,
    int input_seq_len,
    cudaStream_t stream) {
  if (input == nullptr || token_indices == nullptr || out == nullptr ||
      hidden_dim <= 0 || token_count <= 0 || input_seq_len <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = hidden_dim * token_count;
  int block = 256;
  int grid = (total + block - 1) / block;
  gather_hidden_tokens_kernel<<<grid, block, 0, stream>>>(
      input, token_indices, out, hidden_dim, token_count, input_seq_len);
  return (CUresult)cudaGetLastError();
}

CUresult extract_hidden_rows_cuda(
    const __nv_bfloat16 *src,
    __nv_bfloat16 *dst,
    int src_hidden_dim,
    int dst_hidden_dim,
    int col_offset,
    int rows,
    int seq_len,
    cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || src_hidden_dim <= 0 ||
      dst_hidden_dim <= 0 || col_offset < 0 || rows <= 0 || seq_len <= 0 ||
      rows > dst_hidden_dim || col_offset + rows > src_hidden_dim) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = rows * seq_len;
  int block = 256;
  int grid = (total + block - 1) / block;
  extract_hidden_rows_kernel<<<grid, block, 0, stream>>>(
      src, dst, src_hidden_dim, dst_hidden_dim, col_offset, rows, seq_len);
  return (CUresult)cudaGetLastError();
}

CUresult copy_hidden_rows_cuda(
    const __nv_bfloat16 *src,
    __nv_bfloat16 *dst,
    int src_hidden_dim,
    int dst_hidden_dim,
    int row_offset,
    int rows,
    int seq_len,
    cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || src_hidden_dim <= 0 ||
      dst_hidden_dim <= 0 || row_offset < 0 || rows <= 0 || seq_len <= 0 ||
      rows > src_hidden_dim || row_offset + rows > dst_hidden_dim) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = rows * seq_len;
  int block = 256;
  int grid = (total + block - 1) / block;
  copy_hidden_rows_kernel<<<grid, block, 0, stream>>>(
      src, dst, src_hidden_dim, dst_hidden_dim, row_offset, rows, seq_len);
  return (CUresult)cudaGetLastError();
}

CUresult mask_position_zero_rows_cuda(
    const __nv_bfloat16 *src,
    const uint32_t *positions,
    __nv_bfloat16 *dst,
    int hidden_dim,
    int rows,
    cudaStream_t stream) {
  if (src == nullptr || positions == nullptr || dst == nullptr ||
      hidden_dim <= 0 || rows <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = hidden_dim * rows;
  int block = 256;
  int grid = (total + block - 1) / block;
  mask_position_zero_rows_kernel<<<grid, block, 0, stream>>>(
      src, positions, dst, hidden_dim, rows);
  return (CUresult)cudaGetLastError();
}

CUresult copy_hidden_token_range_cuda(
    const __nv_bfloat16 *src,
    __nv_bfloat16 *dst,
    int hidden_dim,
    int src_token_offset,
    int dst_token_offset,
    int token_count,
    int src_seq_len,
    int dst_seq_len,
    cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || hidden_dim <= 0 ||
      src_token_offset < 0 || dst_token_offset < 0 || token_count <= 0 ||
      src_seq_len <= 0 || dst_seq_len <= 0 ||
      src_token_offset + token_count > src_seq_len ||
      dst_token_offset + token_count > dst_seq_len) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = hidden_dim * token_count;
  int block = 256;
  int grid = (total + block - 1) / block;
  copy_hidden_token_range_kernel<<<grid, block, 0, stream>>>(
      src, dst, hidden_dim, src_token_offset, dst_token_offset, token_count);
  return (CUresult)cudaGetLastError();
}

CUresult scaled_add_rows_indexed_cuda(
    const __nv_bfloat16 *delta,
    float scale,
    const int *token_indices,
    __nv_bfloat16 *out,
    int out_hidden_dim,
    int row_offset,
    int rows,
    int token_count,
    int out_seq_len,
    cudaStream_t stream) {
  if (delta == nullptr || token_indices == nullptr || out == nullptr ||
      out_hidden_dim <= 0 || row_offset < 0 || rows <= 0 ||
      token_count <= 0 || out_seq_len <= 0 ||
      row_offset + rows > out_hidden_dim) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dim3 block(32, 8);
  int grid_x = (rows + block.x - 1) / block.x;
  int grid_y = (token_count + block.y - 1) / block.y;
  grid_x = grid_x > 65535 ? 65535 : grid_x;
  grid_y = grid_y > 65535 ? 65535 : grid_y;
  dim3 grid(grid_x, grid_y);
  scaled_add_rows_indexed_kernel<<<grid, block, 0, stream>>>(
      delta, scale, token_indices, out, out_hidden_dim, row_offset, rows,
      token_count, out_seq_len);
  return (CUresult)cudaGetLastError();
}

CUresult store_rows_indexed_cuda(
    const __nv_bfloat16 *src,
    const int *token_indices,
    __nv_bfloat16 *out,
    int out_hidden_dim,
    int row_offset,
    int rows,
    int token_count,
    int out_seq_len,
    cudaStream_t stream) {
  if (src == nullptr || token_indices == nullptr || out == nullptr ||
      out_hidden_dim <= 0 || row_offset < 0 || rows <= 0 ||
      token_count <= 0 || out_seq_len <= 0 ||
      row_offset + rows > out_hidden_dim) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dim3 block(32, 8);
  int grid_x = (rows + block.x - 1) / block.x;
  int grid_y = (token_count + block.y - 1) / block.y;
  grid_x = grid_x > 65535 ? 65535 : grid_x;
  grid_y = grid_y > 65535 ? 65535 : grid_y;
  dim3 grid(grid_x, grid_y);
  store_rows_indexed_kernel<<<grid, block, 0, stream>>>(
      src, token_indices, out, out_hidden_dim, row_offset, rows,
      token_count, out_seq_len);
  return (CUresult)cudaGetLastError();
}

CUresult bf16_to_f32_cuda(
    const __nv_bfloat16 *input, float *output, int n, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  bf16_to_f32_kernel<<<grid, block, 0, stream>>>(input, output, n);
  return (CUresult)cudaGetLastError();
}

CUresult f32_to_bf16_cuda(
    const float *input, __nv_bfloat16 *output, int n, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  f32_to_bf16_kernel<<<grid, block, 0, stream>>>(input, output, n);
  return (CUresult)cudaGetLastError();
}

CUresult scale_f32_cuda(float *values, float scale, int n, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  scale_f32_kernel<<<grid, block, 0, stream>>>(values, scale, n);
  return (CUresult)cudaGetLastError();
}

CUresult accumulate_bf16_token_scaled_to_f32_cuda(
    const __nv_bfloat16 *token,
    float scale,
    float *out,
    int hidden_dim,
    int token_idx,
    int seq_len,
    cudaStream_t stream) {
  if (token == nullptr || out == nullptr || hidden_dim <= 0 || seq_len <= 0 ||
      token_idx < 0 || token_idx >= seq_len) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int block = 256;
  int grid = (hidden_dim + block - 1) / block;
  accumulate_bf16_token_scaled_to_f32_kernel<<<grid, block, 0, stream>>>(
      token, scale, out, hidden_dim, token_idx, seq_len);
  return (CUresult)cudaGetLastError();
}

CUresult repeat_f32_for_reduce_scatter_cuda(
    const float *local, float *repeated, int local_elems, int world_size,
    cudaStream_t stream) {
  int total = local_elems * world_size;
  int block = 256;
  int grid = (total + block - 1) / block;
  repeat_f32_rows_for_reduce_scatter_kernel<<<grid, block, 0, stream>>>(
      local, repeated, local_elems, world_size);
  return (CUresult)cudaGetLastError();
}

CUresult silu_mul_triton_aot_cuda(
    const __nv_bfloat16 *gate, const __nv_bfloat16 *up,
    __nv_bfloat16 *out, int n, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  silu_mul_kernel<<<grid, block, 0, stream>>>(gate, up, out, n);
  return (CUresult)cudaGetLastError();
}

CUresult gelu_tanh_mul_cuda(
    const __nv_bfloat16 *gate, const __nv_bfloat16 *up,
    __nv_bfloat16 *out, int n, cudaStream_t stream) {
  int block = 256;
  int grid = n / block + (n % block != 0);
  gelu_tanh_mul_kernel<<<grid, block, 0, stream>>>(gate, up, out, n);
  return (CUresult)cudaGetLastError();
}

CUresult scale_bf16_in_place_cuda(
    __nv_bfloat16 *buf, float scale, int n, cudaStream_t stream) {
  int block = 256;
  int grid = n / block + (n % block != 0);
  scale_bf16_kernel<<<grid, block, 0, stream>>>(buf, scale, n);
  return (CUresult)cudaGetLastError();
}

CUresult softcap_bf16_in_place_cuda(
    __nv_bfloat16 *buf, float cap, int n, cudaStream_t stream) {
  int block = 256;
  int grid = n / block + (n % block != 0);
  softcap_bf16_kernel<<<grid, block, 0, stream>>>(buf, cap, n);
  return (CUresult)cudaGetLastError();
}

CUresult suppress_logits_bf16_in_place_cuda(
    __nv_bfloat16 *logits, const uint32_t *ids,
    int vocab, int rows, int id_count, cudaStream_t stream) {
  if (logits == nullptr || ids == nullptr || vocab <= 0 || rows <= 0 ||
      id_count <= 0 || id_count > INT_MAX / rows) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = rows * id_count;
  int block = 256;
  int grid = total / block + (total % block != 0);
  suppress_logits_bf16_kernel<<<grid, block, 0, stream>>>(
      logits, ids, vocab, id_count, total);
  return (CUresult)cudaGetLastError();
}

CUresult embedding_decode_cuda(
    const __nv_bfloat16 *embed, const uint32_t *token_id,
    __nv_bfloat16 *out, int hidden_size, cudaStream_t stream) {
  int block = 256;
  int grid = (hidden_size + block - 1) / block;
  embedding_decode_kernel<<<grid, block, 0, stream>>>(embed, token_id, out, hidden_size);
  return (CUresult)cudaGetLastError();
}

// Whether the vectorized row-copy embedding kernels can serve this call:
// whole 16-byte vectors per row and 16-byte-aligned base pointers. Row
// offsets stay aligned because every row is a whole number of vectors.
static bool embedding_vec4_ok(
    const void *embed, const void *out, int hidden_size) {
  return hidden_size % 8 == 0 &&
         ((reinterpret_cast<uintptr_t>(embed) |
           reinterpret_cast<uintptr_t>(out)) & 15) == 0;
}

// Measured on H100 with the 2D grid (seq 128/2048/10000, bs 1/32): 128
// threads edges out 256 by ~2% at 10k tokens and is level elsewhere — more
// row segments per token hide the gather latency slightly better.
static const int EMBEDDING_VEC4_BLOCK = 128;

CUresult embedding_batched_cuda(
    const __nv_bfloat16 *embed, const uint32_t *token_ids,
    __nv_bfloat16 *out, int hidden_size, int seq_len, cudaStream_t stream) {
  if (embedding_vec4_ok(embed, out, hidden_size)) {
    int hidden_vec = hidden_size / 8;
    int row_segs =
        (hidden_vec + EMBEDDING_VEC4_BLOCK - 1) / EMBEDDING_VEC4_BLOCK;
    dim3 grid(seq_len, row_segs);
    embedding_batched_vec4_kernel<<<grid, EMBEDDING_VEC4_BLOCK, 0, stream>>>(
        reinterpret_cast<const uint4 *>(embed), token_ids,
        reinterpret_cast<uint4 *>(out), hidden_vec, seq_len);
    return (CUresult)cudaGetLastError();
  }
  int block = 256;
  int total = hidden_size * seq_len;
  int grid = (total + block - 1) / block;
  embedding_batched_kernel<<<grid, block, 0, stream>>>(embed, token_ids, out, hidden_size, seq_len);
  return (CUresult)cudaGetLastError();
}

CUresult embedding_batched_vocab_shard_cuda(
    const __nv_bfloat16 *embed, const uint32_t *token_ids,
    __nv_bfloat16 *out, int hidden_size, int seq_len,
    uint32_t vocab_start, uint32_t part_vocab_size, cudaStream_t stream) {
  if (embedding_vec4_ok(embed, out, hidden_size)) {
    int hidden_vec = hidden_size / 8;
    int row_segs =
        (hidden_vec + EMBEDDING_VEC4_BLOCK - 1) / EMBEDDING_VEC4_BLOCK;
    dim3 grid(seq_len, row_segs);
    embedding_batched_vocab_shard_vec4_kernel<<<
        grid, EMBEDDING_VEC4_BLOCK, 0, stream>>>(
        reinterpret_cast<const uint4 *>(embed), token_ids,
        reinterpret_cast<uint4 *>(out), hidden_vec, seq_len,
        vocab_start, part_vocab_size);
    return (CUresult)cudaGetLastError();
  }
  int block = 256;
  int total = hidden_size * seq_len;
  int grid = (total + block - 1) / block;
  embedding_batched_vocab_shard_kernel<<<grid, block, 0, stream>>>(
      embed, token_ids, out, hidden_size, seq_len, vocab_start, part_vocab_size);
  return (CUresult)cudaGetLastError();
}

} // extern "C"
