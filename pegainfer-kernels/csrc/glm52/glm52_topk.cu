// GLM5.2 FlashInfer deterministic top-k K=2048 wrapper.
//
// Calls vendored FlashInfer `FilteredTopK` + `LaunchFinalizeTopKIndices` directly
// (bypassing `TopKDispatch`, which hardcodes lengths=nullptr) so per-row
// `lengths` filter padded/stale logits before slot conversion. Matches
// TokenSpeed's `deterministic_decode_topk` contract: deterministic=true,
// TopKTieBreak::Small, dsa_graph_safe=true, K=2048. Mirrors vLLM
// `sparse_attn_indexer` which always passes `seq_lens` as lengths.

#include "../shared/ffi_guard.cuh"

#include <cstddef>
#include <cstdio>

#include <flashinfer/sampling.cuh>
#include <flashinfer/topk.cuh>

extern "C" int glm52_flashinfer_topk_2048_cuda(
    const float* logits, int* output_indices, float* output_values,
    const int* lengths, int num_rows, int top_k, int max_len,
    cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (logits == nullptr || output_indices == nullptr || lengths == nullptr) {
    return static_cast<int>(CUDA_ERROR_INVALID_VALUE);
  }
  if (num_rows <= 0 || top_k <= 0 || max_len <= 0) {
    return static_cast<int>(CUDA_ERROR_INVALID_VALUE);
  }

  // FilteredTopK requires ~128KB dynamic smem (Hopper+). Reject early on
  // GPUs that can't fit it so callers can fall back instead of crashing
  // inside cudaFuncSetAttribute. dsa_graph_safe=true forces this path, so
  // there is no radix fallback here (radix doesn't accept per-row lengths).
  if (!flashinfer::sampling::CanImplementFilteredTopK()) {
    return static_cast<int>(CUDA_ERROR_NOT_SUPPORTED);
  }

  auto* input = const_cast<float*>(logits);

  cudaError_t err = flashinfer::sampling::FilteredTopK<float, int>(
      input, output_indices, output_values, lengths,
      static_cast<uint32_t>(num_rows), static_cast<uint32_t>(top_k),
      static_cast<uint32_t>(max_len),
      /*deterministic=*/true, flashinfer::sampling::TopKTieBreak::Small, stream,
      /*dsa_graph_safe=*/true);
  if (err != cudaSuccess) {
    fprintf(stderr, "glm52_flashinfer_topk_2048_cuda: FilteredTopK failed: %s\n",
            cudaGetErrorString(err));
    return static_cast<int>(CUDA_ERROR_LAUNCH_FAILED);
  }
  err = flashinfer::sampling::LaunchFinalizeTopKIndices<
      /*SORT_LOCAL_INDICES=*/true, flashinfer::sampling::FilteredTopKMode::Plain, float, int>(
      output_indices, output_values, nullptr, 0, nullptr, nullptr,
      static_cast<uint32_t>(num_rows), static_cast<uint32_t>(top_k),
      static_cast<uint32_t>(max_len), stream);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "glm52_flashinfer_topk_2048_cuda: LaunchFinalizeTopKIndices failed: %s\n",
            cudaGetErrorString(err));
    return static_cast<int>(CUDA_ERROR_LAUNCH_FAILED);
  }
  return static_cast<int>(CUDA_SUCCESS);
  PEGAINFER_FFI_GUARD_END(-1)
}
