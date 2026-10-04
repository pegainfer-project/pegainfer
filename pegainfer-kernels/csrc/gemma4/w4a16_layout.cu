// The compressed-tensors W4A16 linears in the layout the TileLang decode GEMMs
// read (pegainfer-gemma4/kernels/w4a16_defs.py), and that layout back to a
// bf16 matrix for the steps too wide for them.
//
// Checkpoint: `packed` [n, k / 8] int32, value k at bits 4 * (k % 8) of word
// k / 8, stored with an offset of 8; `scales` [n, k / 32] bf16.
// Fragment layout: `wq` [n / 16, k / 64, 32, 4] int32. Word (tile, kq, lane, v)
// covers the 16 x 16 tile (rows 16 * tile, columns 16 * (4 * kq + v)); lane
// (g, t) = (lane / 4, lane % 4) holds its m16n8k16 A-fragment locals
// l = 0..7 -- row g + 8 * ((l % 4) / 2), column 2t + l % 2 + 8 * (l / 4) --
// with nibble slot j holding local kSlotLocal[j]. `sq` [n / 16, k / 32, 8]
// int32 holds rows g and g + 8 of a group's scales as one bf16x2 word.
//
// With `split` > 0 the checkpoint's rows are two stacked halves of `split`
// rows (gate, then up), and the layout interleaves them 32 at a time: each 64
// row tile holds 32 gate rows and the same 32 up rows, so the GEMM that owns
// the tile can write gelu(gate) * up. Dequantization restores stored order.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace pegainfer_gemma4_w4a16 {

constexpr int kGroup = 32;
__constant__ int kSlotLocal[8] = {0, 2, 4, 6, 1, 3, 5, 7};
// The inverse: the nibble slot holding local l.
__constant__ int kLocalSlot[8] = {0, 4, 1, 5, 2, 6, 3, 7};

// The stored row that layout row `row` holds.
__device__ __forceinline__ size_t stored_row(size_t row, int split) {
  if (split == 0) return row;
  const size_t half = row % 64 / 32;
  return half * split + row / 64 * 32 + row % 32;
}

__global__ void pack_kernel(const uint32_t* __restrict__ packed, const uint16_t* __restrict__ scales,
                            uint32_t* __restrict__ wq, uint32_t* __restrict__ sq, int n, int k,
                            int split) {
  const size_t kq_count = k / 64;
  const size_t words = static_cast<size_t>(n / 16) * kq_count * 128;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < words; idx += stride) {
    const int v = idx % 4;
    const int lane = (idx / 4) % 32;
    const size_t kq = (idx / 128) % kq_count;
    const size_t tile = idx / (128 * kq_count);
    const int g = lane / 4, t = lane % 4;
    uint32_t word = 0;
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
      const int l = kSlotLocal[slot];
      const size_t row = stored_row(tile * 16 + g + 8 * ((l % 4) / 2), split);
      const size_t col = (kq * 4 + v) * 16 + 2 * t + l % 2 + 8 * (l / 4);
      const uint32_t q = (packed[row * (k / 8) + col / 8] >> (4 * (col % 8))) & 0xFu;
      word |= q << (4 * slot);
    }
    wq[idx] = word;
  }
  const size_t groups = k / kGroup;
  const size_t scale_words = static_cast<size_t>(n / 16) * groups * 8;
  for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < scale_words; idx += stride) {
    const int g = idx % 8;
    const size_t group = (idx / 8) % groups;
    const size_t tile = idx / (8 * groups);
    const uint32_t lo = scales[stored_row(tile * 16 + g, split) * groups + group];
    const uint32_t hi = scales[stored_row(tile * 16 + g + 8, split) * groups + group];
    sq[idx] = lo | (hi << 16);
  }
}

// One thread per eight consecutive columns of one output row, so the store is
// one 16-byte write. Column kc of a 16-wide tile is lane (g, t) =
// (row % 8, (kc % 8) / 2), local (kc % 2) + 2 * (row % 16 / 8) + 4 * (kc / 8).
__global__ void dequant_kernel(const uint32_t* __restrict__ wq, const uint32_t* __restrict__ sq,
                               __nv_bfloat16* __restrict__ out, int n, int k, int split) {
  const size_t kq_count = k / 64;
  const size_t groups = k / kGroup;
  const size_t chunks = static_cast<size_t>(n) * (k / 8);
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < chunks; idx += stride) {
    const size_t row = idx / (k / 8);
    const size_t col0 = (idx % (k / 8)) * 8;
    const size_t tile = row / 16;
    const int g = row % 8;
    const int high = (row % 16) / 8;
    const size_t kb16 = col0 / 16;
    const int half = (col0 % 16) / 8;
    const size_t word_base = ((tile * kq_count + kb16 / 4) * 32) * 4 + kb16 % 4;
    const uint32_t pair = sq[(tile * groups + col0 / kGroup) * 8 + g];
    const uint16_t bits = high ? static_cast<uint16_t>(pair >> 16) : static_cast<uint16_t>(pair & 0xFFFFu);
    const float scale = __bfloat162float(__ushort_as_bfloat16(bits));
    alignas(16) __nv_bfloat16 vals[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int t = j / 2;
      const uint32_t word = wq[word_base + static_cast<size_t>(g * 4 + t) * 4];
      const int l = (j % 2) + 2 * high + 4 * half;
      const int slot = kLocalSlot[l];
      const float q = static_cast<float>(static_cast<int>((word >> (4 * slot)) & 0xFu) - 8);
      vals[j] = __float2bfloat16_rn(q * scale);
    }
    *reinterpret_cast<uint4*>(out + stored_row(row, split) * k + col0) =
        *reinterpret_cast<const uint4*>(vals);
  }
}

inline bool split_ok(int n, int split) { return split == 0 || (split % 32 == 0 && n == 2 * split); }

}  // namespace pegainfer_gemma4_w4a16

using namespace pegainfer_gemma4_w4a16;

extern "C" {

CUresult gemma4_w4a16_pack_cuda(const uint32_t* packed, const uint16_t* scales, uint32_t* wq, uint32_t* sq,
                                int n, int k, int split, cudaStream_t stream) {
  if (!packed || !scales || !wq || !sq || n <= 0 || n % 16 != 0 || k <= 0 || k % 64 != 0 ||
      !split_ok(n, split)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  pack_kernel<<<1024, 256, 0, stream>>>(packed, scales, wq, sq, n, k, split);
  return cudaGetLastError() == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

CUresult gemma4_w4a16_dequant_cuda(const uint32_t* wq, const uint32_t* sq, __nv_bfloat16* out, int n, int k,
                                   int split, cudaStream_t stream) {
  if (!wq || !sq || !out || n <= 0 || n % 16 != 0 || k <= 0 || k % 64 != 0 || !split_ok(n, split)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dequant_kernel<<<1024, 256, 0, stream>>>(wq, sq, out, n, k, split);
  return cudaGetLastError() == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

}  // extern "C"
