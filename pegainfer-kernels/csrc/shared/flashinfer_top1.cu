#include "common.cuh"
#include "flashinfer_radix_scratch.cuh"

#include <cstdio>
#include <cstdlib>

#include <flashinfer/sampling.cuh>
#include <flashinfer/topk.cuh>

extern "C" size_t flashinfer_top1_row_states_bytes_cuda() {
  return flashinfer_radix_row_states_bytes();
}

extern "C" void flashinfer_top1_cuda(const __nv_bfloat16* logits,
                                     __nv_bfloat16* top1_value_scratch,
                                     uint8_t* row_states_scratch, int* output,
                                     int vocab_size, cudaStream_t stream) {
  auto* row_states =
      reinterpret_cast<flashinfer::sampling::RadixRowState*>(row_states_scratch);
  auto* input = const_cast<__nv_bfloat16*>(logits);
  // deterministic=true: bf16 logits tie at the max in practice (8 mantissa
  // bits), and the non-deterministic collect picks an arbitrary winner per
  // run. Deterministic collect resolves ties to the smallest index (matching
  // torch.argmax) for ~2us extra latency.
  cudaError_t err = flashinfer::sampling::TopKDispatch<__nv_bfloat16, int>(
      input, output, top1_value_scratch, 1, 1, vocab_size, row_states,
      /*sorted_output=*/false, /*deterministic=*/true,
      flashinfer::sampling::TopKTieBreak::None, stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "flashinfer_top1_cuda: TopKDispatch failed: %s\n",
            cudaGetErrorString(err));
    abort();
  }
}

extern "C" void flashinfer_top1_batch_cuda(const __nv_bfloat16* logits,
                                           __nv_bfloat16* top1_values,
                                           uint8_t* row_states_scratch,
                                           int* output, int num_rows,
                                           int vocab_size,
                                           cudaStream_t stream) {
  auto* row_states =
      reinterpret_cast<flashinfer::sampling::RadixRowState*>(row_states_scratch);
  for (int row = 0; row < num_rows; ++row) {
    auto* input = const_cast<__nv_bfloat16*>(logits + row * vocab_size);
    cudaError_t err = flashinfer::sampling::TopKDispatch<__nv_bfloat16, int>(
        input, output + row, top1_values + row, 1, 1, vocab_size, row_states,
        /*sorted_output=*/false, /*deterministic=*/true,
        flashinfer::sampling::TopKTieBreak::None, stream);
    if (err != cudaSuccess) {
      fprintf(stderr, "flashinfer_top1_batch_cuda: TopKDispatch failed: %s\n",
              cudaGetErrorString(err));
      abort();
    }
  }
}
