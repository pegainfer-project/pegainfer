#include "ffi_guard.cuh"

#include "common.cuh"
#include "flashinfer_radix_scratch.cuh"

#include <cuda_bf16.h>
#include <stdint.h>

#include <flashinfer/sampling.cuh>
#include <flashinfer/topk.cuh>

namespace {

constexpr int GATHER_CAST_BLOCK = 256;
constexpr int GATHER_CAST_ELEMS_PER_THREAD = 8;

// Gather rows of a bf16 logits arena into a compact f32 buffer.
// grid = (ceil(vocab / (BLOCK * ELEMS)), n_rows); row_indices == nullptr means identity.
__global__ void gather_cast_logits_f32_kernel(const __nv_bfloat16* __restrict__ logits,
                                              const int* __restrict__ row_indices,
                                              float* __restrict__ out, int vocab_size) {
  int compact_row = blockIdx.y;
  int src_row = row_indices == nullptr ? compact_row : row_indices[compact_row];
  const __nv_bfloat16* src = logits + static_cast<size_t>(src_row) * vocab_size;
  float* dst = out + static_cast<size_t>(compact_row) * vocab_size;

  int base = (blockIdx.x * GATHER_CAST_BLOCK + threadIdx.x) * GATHER_CAST_ELEMS_PER_THREAD;
#pragma unroll
  for (int j = 0; j < GATHER_CAST_ELEMS_PER_THREAD; ++j) {
    int i = base + j;
    if (i < vocab_size) {
      dst[i] = __bfloat162float(src[i]);
    }
  }
}

}  // namespace

// Batched temperature/top-k/top-p sampling over a bf16 logits arena.
//
// Three launches for the whole batch: gather+cast (bf16 -> f32), FlashInfer
// OnlineSoftmax (per-row temperature, vocab-splitting strategy for the
// small-batch x large-vocab decode regime), then one FlashInfer sampling kernel
// (Sampling/TopP/TopKTopP depending on the row params). One philox seed per
// call; rows decorrelate through the philox subsequence (= row index), so the
// caller must supply a fresh seed per decode step.
//
// top_k_arr entries must be pre-clamped to [1, vocab_size] when top-k is used;
// temperature_arr entries must be > 0 — greedy rows belong on the argmax path,
// not here.
// Workspace for the radix top-k renorm used on the min_p pipeline; same
// layout contract as flashinfer_top1_row_states_bytes_cuda.
extern "C" size_t gpu_sample_topk_renorm_row_states_bytes_cuda() {
  return flashinfer_radix_row_states_bytes();
}

// min_p_arr enables the min_p pipeline: (optional) top-k renorm, (optional)
// top-p renorm, then FlashInfer's MinPSamplingFromProb with the per-row
// thresholds. min_p_arr == nullptr keeps the original fused single-kernel
// paths bit-for-bit (the fast path).
//
// Per-request seeds deliberately do NOT go through FlashInfer's `seed_arr`:
// these kernels read `seed_arr[0]` (one seed for the whole batch) and fold
// `blockIdx.x` into the philox subsequence, so a request's stream would
// change with its position in the batch. Seeded rows are instead sampled as
// their own n_rows=1 calls by the Rust layer, with the request seed and step
// mixed into `seed` — blockIdx is then always 0 and replay is independent of
// batch composition.
extern "C" int gpu_sample_batch_flashinfer_cuda(
    const __nv_bfloat16* logits, const int* row_indices, float* probs_scratch,
    const float* temperature_arr, const int* top_k_arr, const float* top_p_arr,
    const float* min_p_arr, uint8_t* topk_row_states_scratch,
    uint8_t* valid_scratch, int* output, void* softmax_workspace,
    size_t softmax_workspace_bytes, int n_rows, int vocab_size, int has_top_k_filter,
    int has_top_p_filter, uint64_t seed, uint64_t offset, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  dim3 gather_grid(
      (vocab_size + GATHER_CAST_BLOCK * GATHER_CAST_ELEMS_PER_THREAD - 1) /
          (GATHER_CAST_BLOCK * GATHER_CAST_ELEMS_PER_THREAD),
      n_rows);
  gather_cast_logits_f32_kernel<<<gather_grid, GATHER_CAST_BLOCK, 0, stream>>>(
      logits, row_indices, probs_scratch, vocab_size);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }

  // In-place: phase 2 of both OnlineSoftmax strategies is an elementwise
  // read-then-write of the same index.
  err = flashinfer::sampling::OnlineSoftmax<float>(
      probs_scratch, probs_scratch, n_rows, vocab_size, const_cast<float*>(temperature_arr),
      /*temperature_val=*/1.0f, softmax_workspace, softmax_workspace_bytes,
      /*enable_pdl=*/false, stream);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }

  bool* valid = reinterpret_cast<bool*>(valid_scratch);
  if (min_p_arr != nullptr) {
    if (has_top_k_filter) {
      auto* row_states =
          reinterpret_cast<flashinfer::sampling::RadixRowState*>(topk_row_states_scratch);
      if (row_states == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
      }
      // In-place: the renorm kernels reduce a threshold first, then rewrite
      // each element from its own index.
      err = flashinfer::sampling::RadixTopKRenormProbMultiCTA<float, int>(
          probs_scratch, probs_scratch, const_cast<int*>(top_k_arr), n_rows,
          /*top_k_val=*/0, vocab_size, row_states, stream);
      if (err != cudaSuccess) {
        return static_cast<int>(err);
      }
    }
    if (has_top_p_filter) {
      err = flashinfer::sampling::TopPRenormProb<float>(
          probs_scratch, probs_scratch, const_cast<float*>(top_p_arr), n_rows,
          /*top_p_val=*/0.0f, vocab_size, stream);
      if (err != cudaSuccess) {
        return static_cast<int>(err);
      }
    }
    err = flashinfer::sampling::MinPSamplingFromProb<float, int>(
        probs_scratch, const_cast<float*>(min_p_arr), output, valid,
        /*indices=*/nullptr, n_rows, /*min_p_val=*/0.0f, vocab_size,
        /*deterministic=*/true, /*seed_arr=*/nullptr, seed, /*offset_arr=*/nullptr, offset,
        stream);
    return static_cast<int>(err);
  }
  if (has_top_k_filter) {
    err = flashinfer::sampling::TopKTopPSamplingFromProb<float, int>(
        probs_scratch, const_cast<int*>(top_k_arr), const_cast<float*>(top_p_arr), output, valid,
        /*indices=*/nullptr, n_rows, /*top_k_val=*/0, /*top_p_val=*/0.0f, vocab_size,
        /*deterministic=*/true, /*seed_arr=*/nullptr, seed, /*offset_arr=*/nullptr, offset,
        stream);
  } else if (has_top_p_filter) {
    err = flashinfer::sampling::TopPSamplingFromProb<float, int>(
        probs_scratch, output, valid, /*indices=*/nullptr, const_cast<float*>(top_p_arr), n_rows,
        /*top_p_val=*/1.0f, vocab_size, /*deterministic=*/true, /*seed_arr=*/nullptr, seed,
        /*offset_arr=*/nullptr, offset, stream);
  } else {
    err = flashinfer::sampling::SamplingFromProb<float, int>(
        probs_scratch, output, valid, /*indices=*/nullptr, n_rows, vocab_size,
        /*deterministic=*/true, /*seed_arr=*/nullptr, seed, /*offset_arr=*/nullptr, offset,
        stream);
  }
  return static_cast<int>(err);
  PEGAINFER_FFI_GUARD_END(-1)
}
