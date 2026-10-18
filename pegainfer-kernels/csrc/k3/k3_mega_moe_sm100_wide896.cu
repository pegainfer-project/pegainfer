// K3 MegaMoE serving worlds for the full 896-expert checkpoint: ranks 8, 16,
// 32 and 64, situ activation only.
//
// The full model fits no single-tray shape (its MXFP4 routed experts alone
// are ~1.4 TB), so these widths are all cross-machine: EP8 = 112 experts/rank
// over 2 GB300 trays, EP16 = 56 over 4, EP32 = 28 over 8, EP64 = 14 over 16.
// See `k3_mega_moe_sm100.cu` for the contract; the shared host machinery
// lives in `k3_mega_moe_sm100_common.cuh`. This is its own TU so nvcc
// compiles the worlds in parallel.

#include <cuda.h>

#ifdef K3_MEGA_MOE_SM100F
#include "k3_mega_moe_sm100_common.cuh"

namespace {
constexpr int kExperts = 896;
}
#endif  // K3_MEGA_MOE_SM100F

extern "C" CUresult k3_mega_moe_launch_wide896(
    int activation, int num_ranks, unsigned short* y, const unsigned char* l1_weights,
    const int* l1_weights_sf, const unsigned char* l2_weights, const int* l2_weights_sf,
    const void* buffers, const long long* symm_ptrs, int rank_idx, int num_tokens,
    int* cumulative_stats, cudaStream_t stream) {
#ifdef K3_MEGA_MOE_SM100F
  const auto& mega_buffers = *static_cast<const k3_mega::MegaBuffers*>(buffers);
  switch (num_ranks) {
    case 8:
      return k3_mega::launch_mega_pinned<kExperts, 8, /*kSituOnly=*/true>(
          activation, y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf, mega_buffers,
          symm_ptrs, rank_idx, num_tokens, cumulative_stats, stream);
    case 16:
      return k3_mega::launch_mega_pinned<kExperts, 16, /*kSituOnly=*/true>(
          activation, y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf, mega_buffers,
          symm_ptrs, rank_idx, num_tokens, cumulative_stats, stream);
    case 32:
      return k3_mega::launch_mega_pinned<kExperts, 32, /*kSituOnly=*/true>(
          activation, y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf, mega_buffers,
          symm_ptrs, rank_idx, num_tokens, cumulative_stats, stream);
    case 64:
      return k3_mega::launch_mega_pinned<kExperts, 64, /*kSituOnly=*/true>(
          activation, y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf, mega_buffers,
          symm_ptrs, rank_idx, num_tokens, cumulative_stats, stream);
    default:
      return CUDA_ERROR_NOT_SUPPORTED;
  }
#else
  (void)activation;
  (void)num_ranks;
  (void)y;
  (void)l1_weights;
  (void)l1_weights_sf;
  (void)l2_weights;
  (void)l2_weights_sf;
  (void)buffers;
  (void)symm_ptrs;
  (void)rank_idx;
  (void)num_tokens;
  (void)cumulative_stats;
  (void)stream;
  return CUDA_ERROR_NOT_SUPPORTED;
#endif  // K3_MEGA_MOE_SM100F
}
