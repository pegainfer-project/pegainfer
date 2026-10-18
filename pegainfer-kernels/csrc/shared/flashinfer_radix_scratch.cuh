#pragma once

#include <algorithm>
#include <cstddef>

#include <cuda_runtime.h>

#include <flashinfer/sampling.cuh>

// Scratch sizing for FlashInfer's deterministic multi-CTA radix passes
// (top-1 argmax and the top-k renorm on the min_p pipeline): one
// RadixRowState + collect-scratch slot per SM, with a 1MiB floor kept from
// the original top1 wrapper. Single source for both `extern "C"` entry
// points so the layout contract can't drift.
inline size_t flashinfer_radix_row_states_bytes() {
  int device = 0;
  int sm_count = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) {
    err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  }
  size_t groups = err == cudaSuccess && sm_count > 0
                      ? static_cast<size_t>(sm_count)
                      : static_cast<size_t>(
                            flashinfer::sampling::RADIX_TOPK_MAX_DETERMINISTIC_CTAS_PER_GROUP);
  size_t radix_bytes =
      groups * (sizeof(flashinfer::sampling::RadixRowState) +
                sizeof(flashinfer::sampling::RadixDeterministicCollectScratch));
  return std::max<size_t>(1024 * 1024, radix_bytes);
}
