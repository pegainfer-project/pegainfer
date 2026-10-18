// GLM5.2 DSA indexer prefill: paged-cache gather + top-k slot conversion.
//
// Hand-written, new for the TP4 prefill unpaged-MQA path.
//
// k_gather: gathers the paged fp8 indexer K cache into the compact unpaged
// layout the DeepGEMM unpaged MQA logits kernel
// (glm52_deepgemm_mqa_logits_unpaged_cuda) consumes, and emits the
// row -> global-KV-slot LUT used to convert top-k picks back to slots.
// Paged layout per cache block (matches glm52_indexer.cu::
// indexer_k_quant_and_cache_kernel):
//   [block_size * 128 bytes fp8 keys][block_size * 4 bytes f32 scales]
// with blocks strided by `block_stride_bytes`.
//
// topk_to_slots_lut: unpaged twin of glm52_indexer_local_topk_to_slots_cuda —
// maps top-k offsets through the gather LUT instead of walking the block
// table, mirroring the -1-padding and topk_lens semantics of the paged twin.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>

namespace {

constexpr int kHeadDim = 128;             // fp8 bytes per token row
constexpr int kChunkBytes = 16;           // one int4 per thread
constexpr int kChunksPerToken = kHeadDim / kChunkBytes;  // 8

// One thread moves one 16-byte chunk of a token row; the chunk-0 thread also
// writes the token's f32 scale and global slot id. blockIdx.y = request.
__global__ void indexer_k_gather_kernel(
    const unsigned char* __restrict__ paged_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const int* __restrict__ out_offsets,
    unsigned char* __restrict__ k_out,
    float* __restrict__ scale_out,
    int* __restrict__ slot_out,
    int table_stride,
    int block_size,
    long long block_stride_bytes) {
  const int req = blockIdx.y;
  const int len = seq_lens[req];
  const long long out_base = out_offsets[req];
  const long long total_chunks =
      static_cast<long long>(len) * kChunksPerToken;

  for (long long idx =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total_chunks;
       idx += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int t = static_cast<int>(idx / kChunksPerToken);
    const int chunk = static_cast<int>(idx % kChunksPerToken);
    const int page = block_table[req * table_stride + t / block_size];
    const int in_page = t % block_size;

    const unsigned char* src = paged_cache +
                               static_cast<long long>(page) * block_stride_bytes +
                               static_cast<long long>(in_page) * kHeadDim +
                               chunk * kChunkBytes;
    unsigned char* dst = k_out + (out_base + t) * kHeadDim + chunk * kChunkBytes;
    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);

    if (chunk == 0) {
      // Scales live after the fp8 region of each block: bs*128 + in_page*4.
      const float* scale = reinterpret_cast<const float*>(
          paged_cache + static_cast<long long>(page) * block_stride_bytes +
          static_cast<long long>(block_size) * kHeadDim +
          static_cast<long long>(in_page) * 4);
      scale_out[out_base + t] = *scale;
      // Global KV slot id, same convention as
      // glm52_indexer.cu::local_topk_to_global_slots_kernel and the
      // FlashMLA sparse attention consumer: page * block_size + in-page.
      slot_out[out_base + t] = page * block_size + in_page;
    }
  }
}

// Unpaged twin of local_topk_to_global_slots_kernel: one block per query
// row; offsets are RELATIVE to the row's kv range start (cu_seqlen_ks), the
// same origin as the compacted candidate range [0, context_len). Valid picks
// map through the gather LUT; invalid ones stay -1. topk_lens[m] is the
// block-reduced count of valid picks (== min(context_len, topk) when the
// producer -1-pads), mirroring the paged twin bit-for-bit.
__global__ void topk_to_slots_lut_kernel(
    const int* __restrict__ topk_offsets,
    const int* __restrict__ context_lens,
    const int* __restrict__ cu_seqlen_ks,
    const int* __restrict__ slot_lut,
    int* __restrict__ global_slots,
    int* __restrict__ topk_lens,
    int topk) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  const int context_len = context_lens[row];
  const int ks = cu_seqlen_ks[row];

  int count = 0;
  for (int start = 0; start < topk; start += blockDim.x) {
    const int k = start + tid;
    if (k < topk) {
      const int off = topk_offsets[row * topk + k];
      const bool valid = (off >= 0) && (off < context_len);
      global_slots[row * topk + k] = valid ? slot_lut[ks + off] : -1;
      if (valid) {
        count += 1;
      }
    }
  }

  // Warp-reduce count, then block-reduce via shared memory (same shape as
  // the paged twin).
  for (int mask = 16; mask > 0; mask /= 2) {
    count += __shfl_xor_sync(0xffffffff, count, mask);
  }

  __shared__ int warp_counts[32];
  const int warp = tid / 32;
  const int lane = tid % 32;
  if (lane == 0) {
    warp_counts[warp] = count;
  }
  __syncthreads();

  const int num_warps = blockDim.x / 32;
  int total = 0;
  for (int w = 0; w < num_warps; ++w) {
    total += warp_counts[w];
  }

  if (tid == 0) {
    topk_lens[row] = total;
  }
}

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

}  // namespace

extern "C" {

// Gather the paged fp8 indexer K cache into the compact unpaged layout the
// DeepGEMM unpaged MQA kernel consumes, plus the row -> global-slot LUT.
// Token t of request r reads page block_table[r*table_stride + t/block_size]
// at in-page index t%block_size and writes row out_offsets[r] + t of
// k_out/scale_out/slot_out. Row ranges of distinct requests must not overlap.
CUresult glm52_indexer_k_gather_cuda(
    const unsigned char* paged_cache,  // indexer K cache base
    const int* block_table,            // [num_requests, table_stride] page ids
    const int* seq_lens,               // [num_requests] tokens to gather
    const int* out_offsets,            // [num_requests] destination row offset
    unsigned char* k_out,              // [total_kv, 128] fp8
    float* scale_out,                  // [total_kv] f32
    int* slot_out,                     // [total_kv] i32 global kv slot ids
    int num_requests,
    int table_stride,
    int block_size,                    // 64
    long long block_stride_bytes,      // kv_cache_stride_bytes
    CUstream stream) {
  if (paged_cache == nullptr || block_table == nullptr || seq_lens == nullptr ||
      out_offsets == nullptr || k_out == nullptr || scale_out == nullptr ||
      slot_out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_requests <= 0 || table_stride <= 0 || block_size <= 0 ||
      block_stride_bytes <
          static_cast<long long>(block_size) * (kHeadDim + 4)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  constexpr int kThreads = 256;
  constexpr int kBlocksX = 132;  // grid-stride; sized for one full GPU
  dim3 grid(kBlocksX, num_requests);
  dim3 block(kThreads);
  indexer_k_gather_kernel<<<grid, block, 0, stream>>>(
      paged_cache, block_table, seq_lens, out_offsets, k_out, scale_out,
      slot_out, table_stride, block_size, block_stride_bytes);
  return consume_last_cuda_error();
}

// Convert per-query top-k offsets over the unpaged logits into global KV
// slots via the gather LUT. Offsets are relative to the query's kv range
// start: valid iff 0 <= off < context_lens[m] (context_len = ke - ks), and
// map to slot_lut[cu_seqlen_ks[m] + off]. With ks == 0 (the GLM5.2 DSA
// prefill case) relative and absolute offsets coincide. Invalid or -1-padded
// offsets produce -1, and topk_lens[m] counts the valid picks, exactly like
// glm52_indexer_local_topk_to_slots_cuda.
CUresult glm52_indexer_topk_to_slots_lut_cuda(
    const int* topk_offsets,   // [rows, topk]
    const int* context_lens,   // [rows] per-query kv length (ke - ks)
    const int* cu_seqlen_ks,   // [rows]
    const int* slot_lut,       // [total_kv] from k_gather's slot_out
    int* global_slots,         // [rows, topk] out, -1 padded
    int* topk_lens,            // [rows] out
    int rows,
    int topk,
    CUstream stream) {
  if (topk_offsets == nullptr || context_lens == nullptr ||
      cu_seqlen_ks == nullptr || slot_lut == nullptr ||
      global_slots == nullptr || topk_lens == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rows <= 0 || topk <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  constexpr int kBlockThreads = 256;
  dim3 grid(rows);
  dim3 block(kBlockThreads);
  topk_to_slots_lut_kernel<<<grid, block, 0, stream>>>(
      topk_offsets, context_lens, cu_seqlen_ks, slot_lut, global_slots,
      topk_lens, topk);
  return consume_last_cuda_error();
}

}  // extern "C"
