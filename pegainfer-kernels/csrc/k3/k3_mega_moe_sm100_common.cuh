// Shared host machinery for the K3 MegaMoE AOT instantiations.
//
// The instantiation matrix is split across translation units so nvcc can
// compile the worlds in parallel: `k3_mega_moe_sm100.cu` owns the entry
// points and the bring-up worlds (224 experts at 1 and 4 ranks), and the
// `k3_mega_moe_sm100_wide*.cu` TUs own the cross-machine widths. Everything
// they must agree on — the constexpr replicas of DeepGEMM's heuristics, the
// ring capacities, the TMA descriptor construction and the launch body —
// lives here, templated on the GLOBAL expert count and the rank count.
//
// Include only under `K3_MEGA_MOE_SM100F`; the layout/instantiation contract
// is documented at the top of `k3_mega_moe_sm100.cu`.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

// `layout::Data`'s constructor uses DG_UNIFIED_ASSERT, which expands to a raw
// `asm("trap;")`. That is device-only asm, but `MegaMoEBuffer` is also built on
// the host here (and inside `CUTLASS_HOST_DEVICE` constexpr code), so give the
// macro a definition that is valid in both passes before the headers see it.
#if defined(__CUDA_ARCH__)
#define K3_MEGA_TRAP() __trap()
#else
#define K3_MEGA_TRAP() __builtin_trap()
#endif
#ifndef DG_UNIFIED_ASSERT
#define DG_UNIFIED_ASSERT(cond)    \
  do {                             \
    if (not(cond)) K3_MEGA_TRAP(); \
  } while (0)
#endif

#include <jit_kernels/impls/runtime_utils.hpp>

#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

// nvcc emits float non-type template arguments verbatim into the generated
// device stub, where +infinity comes out as the bare token `inf`. Give the host
// compiler something to bind it to. (`kActivationClamp == infinity` is the
// "no clamp" spelling; the kernel `if constexpr`s the clamp away for it.)
#ifndef K3_MEGA_INF_DEFINED
#define K3_MEGA_INF_DEFINED
constexpr float inf = __builtin_huge_valf();
#endif

namespace k3_mega {

// Per-expert K3 latent-MoE shapes. L1 is the fused gate|up projection. The
// GLOBAL routed-expert count is a template parameter — the two published K3
// text towers differ in that number alone (224 pruned, 896 full).
constexpr int kHidden = 3584;
constexpr int kIntermediate = 3072;
constexpr int kNumTopk = 16;

// Token capacity one rank's slab and kernel instantiation carry
// (`num_max_tokens_per_rank`): the chunked-prefill ceiling, 11x the upstream
// token alignment (`layout::kLCMCandidateBlockM`, 384). This is the ONLY value
// any launch accepts: the ring capacities derived from it are kernel template
// parameters, so a slab allocated for any other value addresses the rings
// wrong. Exported to the Rust side through `k3_mega_max_tokens_per_rank`.
constexpr int kProtocolMaxTokensPerRank = 4224;

// MXFP4 / activation scale-factor group size along K, and how many such groups
// pack into one i32 word.
constexpr int kSfGroupK = 32;
constexpr int kSfPerWord = 4;
constexpr int kSfWordK = kSfGroupK * kSfPerWord;  // 128

constexpr int kActSwiglu = 0;
constexpr int kActSitu = 1;

using namespace deep_gemm;

constexpr int kSmemCapacity = 232448;  // SM100ArchSpec::smem_capacity
constexpr int kMegaBlockN = 128;
constexpr int kNumDispatchThreads = 128;
constexpr int kNumNonEpilogueThreads = 128;

constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }
constexpr int alignup(int a, int b) { return cdiv(a, b) * b; }

inline CUresult map_cuda_error(cudaError_t err) {
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  if (err == cudaErrorNotSupported) return CUDA_ERROR_NOT_SUPPORTED;
  return CUDA_ERROR_LAUNCH_FAILED;
}

// `get_symm_buffer_size_for_mega_moe`'s ring-capacity loop.
constexpr int mega_ring_tokens(int num_ranks, int num_experts, int num_max_tokens_per_rank,
                               int num_topk, int hidden, int intermediate, int num_sms) {
  const int per_rank = num_experts / num_ranks;
  const int active_topk = num_topk < per_rank ? num_topk : per_rank;
  const int routed = num_max_tokens_per_rank * num_ranks * active_topk;
  int best = 0;
  for (int i = 0; i < layout::kNumCandidateBlockMs; ++i) {
    const int bm = layout::kCandidateBlockM[i];
    const int pool = cdiv(routed, bm) + per_rank;
    const int live = sched::get_num_max_live_pool_blocks(pool, num_sms, hidden, intermediate);
    const int tokens = live * bm;
    if (tokens > best) best = tokens;
  }
  return alignup(best, layout::kLCMCandidateBlockM);
}

constexpr int mega_sf_ring_tokens(int num_ring_tokens) {
  int best = 0;
  for (int i = 0; i < layout::kNumCandidateBlockMs; ++i) {
    const int t = layout::get_num_sf_ring_tokens(num_ring_tokens, layout::kCandidateBlockM[i]);
    if (t > best) best = t;
  }
  return best;
}

// `num_bytes_per_pull`: halve the per-token byte count until it fits 4096.
constexpr int mega_bytes_per_pull(int hidden) {
  int b = hidden;  // 1 byte per MMA element for fp8xfp4
  while (b > 4096) b /= 2;
  return b;
}

// One entry of `get_block_config_for_mega_moe`'s ladder. `bucket` is the
// inclusive upper bound on expected tokens per expert, scaled by 2 so it stays
// integral (8.5 -> 17).
struct BlockConfig {
  int bucket_x2;
  int block_m, store_block_m, block_k, num_epilogue_warpgroups;
};

constexpr BlockConfig kBlockLadder[] = {
    {17, 16, 8, 256, 2},    // <= 8.5 expected tokens/expert
    {33, 32, 16, 128, 2},   // <= 16.5
    {65, 64, 32, 128, 1},   // <= 32.5
    {129, 96, 16, 128, 2},  // <= 64.5
    {193, 128, 32, 128, 2}, // <= 96.5
    {0, 192, 32, 128, 2},   // otherwise
};
constexpr int kNumBlockConfigs = (int)(sizeof(kBlockLadder) / sizeof(kBlockLadder[0]));

// `num_expected_tokens_per_expert = num_tokens * num_ranks * num_topk / num_experts`
// compared against the ladder's thresholds. Done in integers: the comparison
// `expected <= bucket` becomes `2 * num_tokens * ranks * topk <= bucket_x2 * experts`.
// Above one rank `num_topk` saturates at the experts-per-rank shard size, the
// most a token can route to one rank.
constexpr int mega_block_config_index(int num_tokens, int num_ranks, int num_experts,
                                      int num_topk) {
  const long long lhs = 2LL * num_tokens * num_ranks * num_topk;
  for (int i = 0; i < kNumBlockConfigs - 1; ++i) {
    if (lhs <= (long long)kBlockLadder[i].bucket_x2 * num_experts) return i;
  }
  return kNumBlockConfigs - 1;
}

// `get_pipeline_config_for_mega_moe`, fp8xfp4 branch.
struct PipelineConfig {
  int num_stages;
  int smem_size;
};

constexpr PipelineConfig mega_pipeline(int num_experts, int block_m, int block_n, int block_k,
                                       int store_block_m, int sf_block_m, int sf_block_n,
                                       int num_dispatch_warps, int num_epilogue_warps,
                                       int num_bytes_per_pull) {
  constexpr int kSmemAlignment = 1024;
  const int smem_dispatch = alignup(num_experts * (int)sizeof(uint32_t), kSmemAlignment) +
                            alignup(num_bytes_per_pull * num_dispatch_warps, kSmemAlignment);
  const int wg = num_epilogue_warps / 4;
  const int smem_cd_l1 = wg * store_block_m * (block_n / 2) * 2;
  const int smem_cd_l2 = wg * store_block_m * block_n * (int)sizeof(nv_bfloat16);
  const int smem_cd = alignup(smem_cd_l1 > smem_cd_l2 ? smem_cd_l1 : smem_cd_l2, kSmemAlignment);
  const int smem_task_info = 2 * (int)sizeof(sched::TaskInfo<true>);
  const int smem_barriers = (num_dispatch_warps + 2 * 2 + num_epilogue_warps * 2 + 2 * 2) * 8;
  const int smem_amax = store_block_m * num_epilogue_warps * (int)sizeof(float);
  const int smem_fixed =
      smem_dispatch + smem_cd + smem_amax + smem_barriers + smem_task_info + 4;
  const int per_stage = (block_m / 2) * block_k + block_n * block_k +
                        sf_block_m * (block_k / kSfGroupK) + sf_block_n * (block_k / kSfGroupK) +
                        2 * 8;
  const int stages = (kSmemCapacity - smem_fixed) / per_stage;
  return {stages, smem_fixed + stages * per_stage};
}

constexpr int kMaxTokensPerRank = kProtocolMaxTokensPerRank;
static_assert(kMaxTokensPerRank % layout::kLCMCandidateBlockM == 0,
              "the protocol maximum must satisfy the upstream token alignment");
constexpr int kGb300Sms = 152;
constexpr int kBytesPerPull = mega_bytes_per_pull(kHidden);

// The ring capacities are template parameters, and they depend on the world
// (through the experts-per-rank and the worst-case routed-token count), so
// every supported world is its own set of constants.
template <int kExperts, int kRanks>
struct MegaRing {
  static constexpr int kTokens = mega_ring_tokens(kRanks, kExperts, kMaxTokensPerRank, kNumTopk,
                                                  kHidden, kIntermediate, kGb300Sms);
  static constexpr int kSfTokens = mega_sf_ring_tokens(kTokens);
};

// One AOT kernel: a world (GLOBAL expert count x rank count) crossed with a
// ladder entry and an activation.
template <int kExperts, int kRanks, int kCfgIdx, bool kSitu>
struct MegaKernel {
  static constexpr BlockConfig kCfg = kBlockLadder[kCfgIdx];
  static constexpr int kBlockM = kCfg.block_m;
  static constexpr int kBlockK = kCfg.block_k;
  static constexpr int kStoreBlockM = kCfg.store_block_m;
  static constexpr int kEpilogueThreads = kCfg.num_epilogue_warpgroups * 128;
  // SM100ArchSpec::get_sf_uttcp_aligned_block_sizes for MXFP8FP4.
  static constexpr int kSfBlockM = alignup(kBlockM, 128);
  static constexpr int kSfBlockN = alignup(kMegaBlockN, 128);
  // The dispatch smem holds one counter per GLOBAL expert, not per local one.
  static constexpr PipelineConfig kPipe =
      mega_pipeline(kExperts, kBlockM, kMegaBlockN, kBlockK, kStoreBlockM, kSfBlockM,
                    kSfBlockN, kNumDispatchThreads / 32, kEpilogueThreads / 32, kBytesPerPull);
  static constexpr int kSmemSize = kPipe.smem_size;
  static constexpr int kNumThreads =
      kNumDispatchThreads + kNumNonEpilogueThreads + kEpilogueThreads;

  static_assert(kPipe.num_stages >= 2, "MegaMoE pipeline needs at least 2 stages");
  static_assert(kSmemSize <= kSmemCapacity, "MegaMoE smem budget overflow");

  static constexpr auto kFn = &sm100_fp8_fp4_mega_moe_impl<
      kMaxTokensPerRank, kHidden, kIntermediate, kExperts, /*shared=*/0, kNumTopk, kBlockM,
      kMegaBlockN, kBlockK, kStoreBlockM, kSfBlockM, kSfBlockN,
      MegaRing<kExperts, kRanks>::kTokens, MegaRing<kExperts, kRanks>::kSfTokens,
      kPipe.num_stages, kBytesPerPull, kNumDispatchThreads, kNumNonEpilogueThreads,
      kEpilogueThreads, kGb300Sms, kRanks, /*clamp=*/inf, kSitu,
      /*fast_math=*/false>;
};

// Every multi-rank world pins ONE config for every launch, chosen at the
// protocol maximum (`num_tokens == num_max_tokens_per_rank`) rather than from
// the live token count.
//
// Two reasons. First, a per-step config would let peers disagree about BLOCK_M
// within one collective launch; nothing in the kernel forces the world to
// agree, and cross-rank behaviour under heterogeneous tiling is unverified
// territory. (The addressing itself is safe — see the layout note in
// `k3_mega_moe_sm100.cu` — but "safe to address" is not "known to be
// correct".) Second, a fixed config makes a row's tile shape independent of
// how much traffic its peers happen to be sending, which is what makes traffic
// invariance testable as bitwise equality rather than as a tolerance.
//
// The cost is small-batch efficiency: a 16-token step still runs BLOCK_M 192
// tiles. That is accepted — the fused path is still ahead of the masked chain,
// which additionally pays four NCCL collectives per MoE layer at EP4.
//
// Every instantiated world must land on the same pinned entry: BLOCK_K decides
// a row's MMA K-accumulation order, so worlds that shared it stay comparable
// (the EP-vs-EP1 oracle leans on this).
template <int kExperts, int kRanks>
constexpr int pinned_config() {
  constexpr int cfg = mega_block_config_index(kMaxTokensPerRank, kRanks, kExperts, kNumTopk);
  static_assert(kBlockLadder[cfg].block_m == 192,
                "the multi-rank protocol-max config is expected to be the BLOCK_M 192 entry");
  static_assert(kBlockLadder[cfg].block_k == kBlockLadder[2].block_k,
                "the pinned config and the widest single-rank config must share BLOCK_K, or a "
                "row's MMA K-accumulation order differs between world sizes");
  return cfg;
}

// K-major packed-FP4 TMA descriptor. `make_tma_2d_desc_raw` only knows the
// `DgDtype` enum, which has no sub-byte member; this reproduces what the
// torch-side path emits for a `kPackedFP4` tensor of shape [rows, k / 2].
inline CUresult make_fp4_k_major_tma_desc(CUtensorMap* out, const void* ptr, int shape_k,
                                          long long rows, int smem_outer_dim) {
  if (shape_k % 128 != 0) return CUDA_ERROR_INVALID_VALUE;
  const cuuint64_t gmem_dims[2] = {(cuuint64_t)shape_k, (cuuint64_t)rows};
  const cuuint64_t gmem_strides[1] = {(cuuint64_t)(shape_k / 2)};
  const cuuint32_t smem_dims[2] = {128u, (cuuint32_t)smem_outer_dim};
  const cuuint32_t elem_strides[2] = {1, 1};
  return deep_gemm::lazy_cuTensorMapEncodeTiled(
      out, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 2, const_cast<void*>(ptr), gmem_dims,
      gmem_strides, smem_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// `make_tma_sf_desc` with the torch tensor replaced by a raw i32 pointer: the
// MN extent is TMA-aligned to 4 elements and doubles as the outer stride, the
// outer extent is `ceil_div(shape_k, gran_k * 4) * num_groups`, unswizzled.
inline CUtensorMap make_sf_desc(void* ptr, int shape_mn, int shape_k, int block_mn, int num_groups,
                                int smem_outer_dim) {
  const int aligned_mn = alignup(shape_mn, 4);
  return deep_gemm::make_tma_2d_desc_raw(ptr, 4, deep_gemm::DgDtype::Int, aligned_mn,
                                         cdiv(shape_k, kSfWordK) * num_groups, block_mn,
                                         smem_outer_dim, aligned_mn, 0);
}

struct MegaBuffers {
  void* base;
  void* l1_acts;
  void* l1_acts_sf;
  void* l2_acts;
  void* l2_acts_sf;
};

template <int kExperts, int kRanks, typename Kernel>
CUresult launch_mega(unsigned short* y, const unsigned char* l1_weights, const int* l1_weights_sf,
                     const unsigned char* l2_weights, const int* l2_weights_sf,
                     const MegaBuffers& buffers, const long long* symm_ptrs, int rank_idx,
                     int num_tokens, int* cumulative_stats, cudaStream_t stream) {
  constexpr int kRingTokens = MegaRing<kExperts, kRanks>::kTokens;
  constexpr int kSfRingTokens = MegaRing<kExperts, kRanks>::kSfTokens;
  // Weight tensors are sharded: a rank holds only its own experts.
  constexpr int kExpertsPerRank = kExperts / kRanks;
  const auto func = reinterpret_cast<const void*>(Kernel::kFn);
  const cudaError_t attr_err = cudaFuncSetAttribute(
      func, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::kSmemSize);
  if (attr_err != cudaSuccess) return map_cuda_error(attr_err);

  constexpr int kLoadBlockM = Kernel::kBlockM / 2;
  constexpr int kLoadBlockN = kMegaBlockN;
  // `sf_smem_outer_dim = block_k / (gran_k * 4)`
  constexpr int kSfSmemOuter = Kernel::kBlockK / kSfWordK;

  CUtensorMap tm_l1_acts = deep_gemm::make_tma_2d_desc_raw(
      buffers.l1_acts, 1, deep_gemm::DgDtype::Float8_e4m3, kHidden, kRingTokens, Kernel::kBlockK,
      kLoadBlockM, kHidden, 128);
  CUtensorMap tm_l1_acts_sf =
      make_sf_desc(buffers.l1_acts_sf, kSfRingTokens, kHidden, Kernel::kSfBlockM, 1, kSfSmemOuter);
  CUtensorMap tm_l1_weights;
  {
    const CUresult err = make_fp4_k_major_tma_desc(
        &tm_l1_weights, l1_weights, kHidden,
        (long long)kExpertsPerRank * (long long)(kIntermediate * 2), kLoadBlockN);
    if (err != CUDA_SUCCESS) return err;
  }
  CUtensorMap tm_l1_weights_sf = make_sf_desc(const_cast<int*>(l1_weights_sf), kIntermediate * 2,
                                              kHidden, kMegaBlockN, kExpertsPerRank, kSfSmemOuter);
  // L1 output and L2 activations are the same tensor; the post-activation
  // output is BLOCK_N / 2 wide, so its swizzle halves too.
  CUtensorMap tm_l1_output = deep_gemm::make_tma_2d_desc_raw(
      buffers.l2_acts, 1, deep_gemm::DgDtype::Float8_e4m3, kIntermediate, kRingTokens,
      kMegaBlockN / 2, Kernel::kStoreBlockM, kIntermediate, 64);
  CUtensorMap tm_l2_acts = deep_gemm::make_tma_2d_desc_raw(
      buffers.l2_acts, 1, deep_gemm::DgDtype::Float8_e4m3, kIntermediate, kRingTokens,
      Kernel::kBlockK, kLoadBlockM, kIntermediate, 128);
  CUtensorMap tm_l2_acts_sf = make_sf_desc(buffers.l2_acts_sf, kSfRingTokens, kIntermediate,
                                           Kernel::kSfBlockM, 1, kSfSmemOuter);
  CUtensorMap tm_l2_weights;
  {
    const CUresult err = make_fp4_k_major_tma_desc(&tm_l2_weights, l2_weights, kIntermediate,
                                                   (long long)kExpertsPerRank * (long long)kHidden,
                                                   kLoadBlockN);
    if (err != CUDA_SUCCESS) return err;
  }
  CUtensorMap tm_l2_weights_sf = make_sf_desc(const_cast<int*>(l2_weights_sf), kHidden,
                                              kIntermediate, kMegaBlockN, kExpertsPerRank,
                                              kSfSmemOuter);

  // No shared experts: upstream passes the routed descriptors through.
  CUtensorMap tm_sl1_acts = tm_l1_acts;
  CUtensorMap tm_sl1_acts_sf = tm_l1_acts_sf;
  CUtensorMap tm_sl1_weights = tm_l1_weights;
  CUtensorMap tm_sl1_weights_sf = tm_l1_weights_sf;
  CUtensorMap tm_sl1_output = tm_l1_output;
  CUtensorMap tm_sl2_acts = tm_l2_acts;
  CUtensorMap tm_sl2_acts_sf = tm_l2_acts_sf;
  CUtensorMap tm_sl2_weights = tm_l2_weights;
  CUtensorMap tm_sl2_weights_sf = tm_l2_weights_sf;

  // The kernel maps a local pointer onto rank r by adding `offsets[r]`, so the
  // table is peer-base minus own-base. At `kRanks == 1` `map` is the identity
  // and the offsets are never read.
  deep_gemm::layout::SymBuffer<kRanks> sym_buffer;
  sym_buffer.rank_idx = (uint32_t)rank_idx;
  sym_buffer.base = symm_ptrs[rank_idx];
  for (uint32_t i = 0; i < deep_gemm::layout::kNumMaxRanks; ++ i) {
    sym_buffer.offsets[i] = i < (uint32_t)kRanks ? symm_ptrs[i] - sym_buffer.base : 0;
  }

  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {2, 1, 1};
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(kGb300Sms, 1, 1);
  config.blockDim = dim3(Kernel::kNumThreads, 1, 1);
  config.dynamicSmemBytes = (size_t)Kernel::kSmemSize;
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = 2;

  void* y_ptr = y;
  uint32_t num_tokens_u32 = (uint32_t)num_tokens;
  void* args[] = {&y_ptr,
                  &cumulative_stats,
                  &num_tokens_u32,
                  &sym_buffer,
                  &tm_l1_acts,
                  &tm_l1_acts_sf,
                  &tm_l1_weights,
                  &tm_l1_weights_sf,
                  &tm_l1_output,
                  &tm_l2_acts,
                  &tm_l2_acts_sf,
                  &tm_l2_weights,
                  &tm_l2_weights_sf,
                  &tm_sl1_acts,
                  &tm_sl1_acts_sf,
                  &tm_sl1_weights,
                  &tm_sl1_weights_sf,
                  &tm_sl1_output,
                  &tm_sl2_acts,
                  &tm_sl2_acts_sf,
                  &tm_sl2_weights,
                  &tm_sl2_weights_sf};
  return map_cuda_error(cudaLaunchKernelExC(&config, func, args));
}

// One multi-rank world's launch body: the pinned config crossed with the two
// activations. Wide worlds instantiate situ only — swiglu exists as a
// single-node regression handle against upstream's Python path, and carrying
// it across every cross-machine width would double an already heavy TU for a
// kernel nothing launches.
template <int kExperts, int kRanks, bool kSituOnly>
CUresult launch_mega_pinned(int activation, unsigned short* y, const unsigned char* l1_weights,
                            const int* l1_weights_sf, const unsigned char* l2_weights,
                            const int* l2_weights_sf, const MegaBuffers& buffers,
                            const long long* symm_ptrs, int rank_idx, int num_tokens,
                            int* cumulative_stats, cudaStream_t stream) {
  constexpr int kCfg = pinned_config<kExperts, kRanks>();
  if (activation == kActSitu) {
    return launch_mega<kExperts, kRanks, MegaKernel<kExperts, kRanks, kCfg, true>>(
        y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf, buffers, symm_ptrs, rank_idx,
        num_tokens, cumulative_stats, stream);
  }
  if (activation == kActSwiglu) {
    if constexpr (kSituOnly) {
      return CUDA_ERROR_NOT_SUPPORTED;
    } else {
      return launch_mega<kExperts, kRanks, MegaKernel<kExperts, kRanks, kCfg, false>>(
          y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf, buffers, symm_ptrs, rank_idx,
          num_tokens, cumulative_stats, stream);
    }
  }
  return CUDA_ERROR_INVALID_VALUE;
}

}  // namespace k3_mega
