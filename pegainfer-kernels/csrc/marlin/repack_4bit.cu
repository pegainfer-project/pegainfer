// Marlin's B layout for any four-bit weight.
//
// The permutation carries nibbles without reading them, so INT4 and NVFP4
// share it: what the code point means is the instantiated `b_type`'s business,
// not this kernel's. The source is one expert-major `[out_dim, in_dim / 2]`
// plane per expert, which is how both checkpoints store it.

#include "ffi.cuh"

namespace {

constexpr int kPackFactor = 8;
constexpr int kTileK = 16;
constexpr int kTileN = 64;
constexpr int kThreads = 256;

__global__ void marlin_repack_4bit_kernel(
    const uint32_t* __restrict__ checkpoint_weight,
    uint32_t* __restrict__ marlin_weight,
    int size_k,
    int size_n) {
  constexpr int tile_ints = kTileK / kPackFactor;
  constexpr int stage_n_threads = kTileN / 4;
  constexpr int stage_elements = tile_ints * kTileN;
  __shared__ uint32_t sh_stage[stage_elements];

  int expert = blockIdx.y;
  int k_tile = blockIdx.x;
  int k_packed_cols = size_k / kPackFactor;
  int n_tiles = size_n / kTileN;
  const uint32_t* expert_checkpoint =
      checkpoint_weight + static_cast<size_t>(expert) * size_n * k_packed_cols;
  uint32_t* expert_marlin =
      marlin_weight + static_cast<size_t>(expert) * (size_k / kTileK) *
                          (size_n * kTileK / kPackFactor);

  int first_k_packed = k_tile * tile_ints;
  for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
    if (threadIdx.x < tile_ints * stage_n_threads) {
      int k_id = threadIdx.x / stage_n_threads;
      int n4 = threadIdx.x % stage_n_threads;
      int n_base = n_tile * kTileN + n4 * 4;
      uint32_t* dst = sh_stage + k_id * kTileN + n4 * 4;
      int src_k = first_k_packed + k_id;
      dst[0] = expert_checkpoint[(n_base + 0) * k_packed_cols + src_k];
      dst[1] = expert_checkpoint[(n_base + 1) * k_packed_cols + src_k];
      dst[2] = expert_checkpoint[(n_base + 2) * k_packed_cols + src_k];
      dst[3] = expert_checkpoint[(n_base + 3) * k_packed_cols + src_k];
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int th_id = threadIdx.x % 32;
    if (warp_id < 4) {
      int tc_col = th_id / 4;
      int tc_row = (th_id % 4) * 2;
      constexpr int tc_offsets[4] = {0, 1, 8, 9};
      int cur_n = warp_id * 16 + tc_col;
      constexpr uint32_t mask = 0x0f;

      uint32_t b1_vals[tile_ints];
      uint32_t b2_vals[tile_ints];
#pragma unroll
      for (int i = 0; i < tile_ints; ++i) {
        b1_vals[i] = sh_stage[cur_n + kTileN * i];
        b2_vals[i] = sh_stage[cur_n + 8 + kTileN * i];
      }

      uint32_t vals[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int cur_elem = tc_row + tc_offsets[i];
        int cur_int = cur_elem / kPackFactor;
        int cur_pos = cur_elem % kPackFactor;
        vals[i] = (b1_vals[cur_int] >> (cur_pos * 4)) & mask;
        vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * 4)) & mask;
      }

      constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        res |= vals[pack_idx[i]] << (i * 4);
      }

      constexpr int tile_size = kTileK * kTileN / kPackFactor;
      int out_offset = (k_tile * n_tiles + n_tile) * tile_size;
      expert_marlin[out_offset + th_id * 4 + warp_id] = res;
    }
    __syncthreads();
  }
}

}  // namespace

extern "C" {

/// `src` is `[experts, out_dim, in_dim / 2]` bytes and `dst` receives the same
/// count in Marlin's order. `in_dim` must be a multiple of 16 and `out_dim` of
/// 64, which is the tile the layout is built from.
CUresult marlin_repack_4bit_cuda(
    const uint8_t* src,
    uint8_t* dst,
    int experts,
    int in_dim,
    int out_dim,
    cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || experts <= 0 || in_dim <= 0 ||
      out_dim <= 0 || (in_dim % kTileK) != 0 || (out_dim % kTileN) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dim3 grid(in_dim / kTileK, experts);
  marlin_repack_4bit_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(src),
      reinterpret_cast<uint32_t*>(dst), in_dim, out_dim);
  cudaError_t err = cudaPeekAtLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

}  // extern "C"
