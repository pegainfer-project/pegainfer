// K3 MegaMoE cross-machine worlds for the 224-expert (pruned) checkpoint:
// ranks 8 and 16, situ activation only.
//
// The pruned checkpoint's EP16 shard (14 experts/rank) and especially its
// EP4-vs-EP16 comparison are the cross-machine transport gate: the same
// tokens greedily decoded on one tray at EP4 and on four trays at EP16 must
// agree, and the pruned weights make that a 4-minute load instead of a
// 1.5 TB one. See `k3_mega_moe_sm100.cu` for the contract; the shared host
// machinery lives in `k3_mega_moe_sm100_common.cuh`. This is its own TU so
// nvcc compiles the worlds in parallel.

#include <cuda.h>

#ifdef K3_MEGA_MOE_SM100F
#include "k3_mega_moe_sm100_common.cuh"

namespace {
constexpr int kExperts = 224;
}
#endif  // K3_MEGA_MOE_SM100F

extern "C" CUresult k3_mega_moe_launch_wide224(
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
