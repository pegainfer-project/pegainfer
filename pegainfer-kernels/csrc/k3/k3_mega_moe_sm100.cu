// Kimi-K3 routed-expert MoE: DeepGEMM SM100 fused "MegaMoE" FP8 x FP4 kernel,
// AOT-instantiated from the vendored device headers (no JIT, no torch).
//
// The mega kernel fuses dispatch + W13 GEMM + activation + mid-quantization +
// W2 GEMM + combine into a single persistent grid. Everything it needs beyond
// the weights lives in one flat "symmetric" byte slab whose sub-buffer offsets
// are pure host arithmetic over the shapes; at ep_size == 1 the slab is a plain
// device allocation and the kernel's cross-rank barriers degrade to grid-local
// synchronisation (`layout::SymBuffer<1>::map` is the identity).
//
// ---------------------------------------------------------------------------
// What this TU replicates from the upstream host side
// ---------------------------------------------------------------------------
// * `deep_gemm::mega::get_symm_buffer_size_for_mega_moe` (csrc/apis/mega.hpp)
//   -> `k3_mega_symm_buffer_layout_cuda`. The upstream version is header
//   arithmetic over `layout::MegaMoEBuffer` plus a torch tensor-slicing lambda;
//   stripping the lambda leaves the total byte count and the 12 sub-buffer
//   offsets, which is exactly what a torch-free caller needs.
// * `get_block_config_for_mega_moe` / `get_pipeline_config_for_mega_moe`
//   (csrc/jit_kernels/heuristics/mega_moe.hpp) -> the `constexpr` replicas
//   below. They are `constexpr` so the AOT instantiations and the launch-time
//   selection provably agree: `MegaCfg` derives BLOCK_M/N/K, stages and the
//   smem budget from the same expressions the JIT would have used.
// * `sm100_fp8_fp4_mega_moe` (csrc/jit_kernels/impls/...) -> `launch_mega`,
//   including all 9 distinct TMA descriptors. The shared-expert descriptors
//   alias the routed ones, mirroring upstream's `num_shared_experts == 0` path.
//
// ---------------------------------------------------------------------------
// Weight / SF layout contract (mirrors `transform_weights_for_mega_moe`)
// ---------------------------------------------------------------------------
// L1 (fused gate|up, "W13") arrives from the checkpoint in split-half row order
// [gate(0..I-1); up(0..I-1)] and must be re-ordered into granularity-8
// interleaved blocks [gate 0..7, up 0..7, gate 8..15, up 8..15, ...]. Both the
// packed FP4 bytes and the packed i32 scale factors get that permutation; the
// L1 SF then gets the UTCCP transpose. L2 weights are untouched and only its SF
// gets the UTCCP transpose.
//
// The scale-factor pipeline from checkpoint bytes to mega-ready SF is a pure
// byte permutation plus a 4:1 pack:
//   checkpoint  [E, n, k/32]     u8, one UE8M0 exponent per (row, K-group)
//   packed      [E, k/128, n]    i32, MN-major, 4 exponents per word LSB-first
//                                (== `transform_sf_into_required_layout` with
//                                 recipe (1, 32); see the note in
//                                 k3_deepgemm_fp8_fp4_grouped_sm100.cu)
//   + row permutation applied in the `n` axis (interleave for L1, then UTCCP)
//
// UTCCP transpose (`_transpose_sf_for_utccp`): view (E, mn/128, 4, 32, pk),
// transpose the two middle axes, flatten back. Row r of the output therefore
// reads row `(r / 128) * 128 + (r % 4) * 32 + (r % 128) / 4` of the input.
//
// ---------------------------------------------------------------------------
// Instantiation matrix
// ---------------------------------------------------------------------------
// hidden 3584 (K3 latent), intermediate 3072, 224 experts, topk 16, ranks 1,
// num_max_tokens_per_rank 384 (`kLCMCandidateBlockM`, the token alignment the
// upstream API enforces), 152 SMs (GB300). Three block configs are reachable
// for num_tokens <= 384 under `get_block_config_for_mega_moe`, times the two
// activations (situ for K3, swiglu as a regression handle) = 6 kernels.
//
// build.rs compiles this TU for sm_100f ONLY when a sm_100-family target
// exists; otherwise every entry point compiles as a NOT_SUPPORTED stub.

#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cstdint>

#ifdef K3_MEGA_MOE_SM100F

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// `layout::Data`'s constructor uses DG_UNIFIED_ASSERT, which expands to a raw
// `asm("trap;")`. That is device-only asm, but `MegaMoEBuffer` is also built on
// the host here (and inside `CUTLASS_HOST_DEVICE` constexpr code), so give the
// macro a definition that is valid in both passes before the headers see it.
#if defined(__CUDA_ARCH__)
#define K3_MEGA_TRAP() __trap()
#else
#define K3_MEGA_TRAP() __builtin_trap()
#endif
#define DG_UNIFIED_ASSERT(cond)   \
  do {                            \
    if (not(cond)) K3_MEGA_TRAP(); \
  } while (0)

#include <jit_kernels/impls/runtime_utils.hpp>

#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

// nvcc emits float non-type template arguments verbatim into the generated
// device stub, where +infinity comes out as the bare token `inf`. Give the host
// compiler something to bind it to. (`kActivationClamp == infinity` is the
// "no clamp" spelling; the kernel `if constexpr`s the clamp away for it.)
constexpr float inf = __builtin_huge_valf();

#endif  // K3_MEGA_MOE_SM100F

namespace {

// Per-expert K3 latent-MoE shapes. L1 is the fused gate|up projection.
constexpr int kHidden = 3584;
constexpr int kIntermediate = 3072;
constexpr int kNumExperts = 224;
constexpr int kNumTopk = 16;

// MXFP4 / activation scale-factor group size along K, and how many such groups
// pack into one i32 word.
constexpr int kSfGroupK = 32;
constexpr int kSfPerWord = 4;
constexpr int kSfWordK = kSfGroupK * kSfPerWord;  // 128

// Weight-row interleave granularity for the fused gate|up projection.
constexpr int kInterleaveGran = 8;

constexpr int kActSwiglu = 0;
constexpr int kActSitu = 1;

CUresult map_cuda_error(cudaError_t err) {
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  if (err == cudaErrorNotSupported) return CUDA_ERROR_NOT_SUPPORTED;
  return CUDA_ERROR_LAUNCH_FAILED;
}

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

// ---------------------------------------------------------------------------
// Row permutations (host/device agreeing pure index math)
// ---------------------------------------------------------------------------

// `_interleave_weights(t, gran=8)` on the row axis: output row `r` reads input
// row `(r / gran) % 2 ? half + ... : ...`, i.e. alternating gran-sized runs of
// the gate half and the up half.
__host__ __device__ __forceinline__ int interleave_src_row(int r, int half) {
  const int pair = r / (kInterleaveGran * 2);
  const int is_up = (r / kInterleaveGran) & 1;
  const int lane = r % kInterleaveGran;
  return (is_up ? half : 0) + pair * kInterleaveGran + lane;
}

// `_transpose_sf_for_utccp`: output row `r` reads input row
// `(r / 128) * 128 + (r % 4) * 32 + (r % 128) / 4`.
__host__ __device__ __forceinline__ int utccp_src_row(int r) {
  return (r / 128) * 128 + (r % 4) * 32 + (r % 128) / 4;
}

// ---------------------------------------------------------------------------
// Transform kernels
// ---------------------------------------------------------------------------

// L1 packed-FP4 weight bytes: [groups, n, k/2] int8, rows permuted by the
// gate/up interleave. Grid-strided over output bytes; `gridDim.y` walks groups.
__global__ void mega_interleave_l1_weights_kernel(const unsigned char* __restrict__ src,
                                                  unsigned char* __restrict__ dst, int n,
                                                  int row_bytes) {
  const size_t group_off = (size_t)blockIdx.y * (size_t)n * (size_t)row_bytes;
  const int half = n / 2;
  const size_t total = (size_t)n * (size_t)row_bytes;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
       idx += (size_t)gridDim.x * blockDim.x) {
    const int row = (int)(idx / (size_t)row_bytes);
    const int col = (int)(idx % (size_t)row_bytes);
    const int src_row = interleave_src_row(row, half);
    dst[group_off + idx] = src[group_off + (size_t)src_row * row_bytes + col];
  }
}

// Checkpoint UE8M0 scale factors -> mega-ready packed SF.
//
// in : [groups, n, k / 32]  u8, K-major (one exponent per row and K-group)
// out: [groups, k / 128, n] i32, MN-major, 4 exponents per word LSB-first
//
// `kInterleave` additionally applies the gate/up interleave before the UTCCP
// transpose, which is what L1 needs; L2 only gets the UTCCP transpose.
template <bool kInterleave>
__global__ void mega_prepare_sf_kernel(const unsigned char* __restrict__ sf, int* __restrict__ out,
                                       int n, int k) {
  const int sf_cols = k / kSfGroupK;
  const int packed_cols = k / kSfWordK;
  const int half = n / 2;
  const unsigned char* group_sf = sf + (size_t)blockIdx.y * (size_t)n * (size_t)sf_cols;
  int* group_out = out + (size_t)blockIdx.y * (size_t)packed_cols * (size_t)n;
  const size_t total = (size_t)packed_cols * (size_t)n;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
       idx += (size_t)gridDim.x * blockDim.x) {
    const int row = (int)(idx % (size_t)n);
    const int word = (int)(idx / (size_t)n);
    int src_row = utccp_src_row(row);
    if constexpr (kInterleave) src_row = interleave_src_row(src_row, half);
    const unsigned char* base =
        group_sf + (size_t)src_row * sf_cols + (size_t)word * kSfPerWord;
    unsigned int packed = 0;
#pragma unroll
    for (int j = 0; j < kSfPerWord; ++j) {
      packed |= (unsigned int)base[j] << (8 * j);
    }
    group_out[(size_t)word * n + row] = (int)packed;
  }
}

#ifdef K3_MEGA_MOE_SM100F

// bf16 activations -> e4m3 + packed UE8M0 SF, written straight into the
// symmetric buffer's `x` / `x_sf` regions.
//
// Bit-for-bit `per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32,
// use_packed_ue8m0=True)`: per 32-element group, amax over |x| in f32 clamped
// to >= 1e-4, sf = ceil_to_ue8m0(amax / 448), quantized value = f32(x) / sf
// rounded to e4m3 (round-nearest-even, and never saturating because
// amax / sf <= 448 by construction).
//
// One warp per (token, 4-group) tuple: 4 consecutive groups = 128 elements =
// one packed SF word, so the whole word is produced by a single warp with no
// cross-warp communication.
__global__ void mega_quant_x_kernel(const __nv_bfloat16* __restrict__ x,
                                    unsigned char* __restrict__ x_fp8, int* __restrict__ x_sf,
                                    int num_tokens, int hidden, int x_stride, int x_sf_stride) {
  const int words_per_token = hidden / kSfWordK;
  const int warps_per_block = blockDim.x / 32;
  const int warp_id = (blockIdx.x * warps_per_block) + (threadIdx.x / 32);
  const int lane = threadIdx.x % 32;
  const long long total_warps = (long long)num_tokens * words_per_token;
  if (warp_id >= total_warps) return;

  const int token = warp_id / words_per_token;
  const int word = warp_id % words_per_token;

  const __nv_bfloat16* row = x + (size_t)token * x_stride + (size_t)word * kSfWordK;
  unsigned char* out_row = x_fp8 + (size_t)token * hidden + (size_t)word * kSfWordK;

  // Lane `l` owns group `l / 8` and element `(l % 8) * 4 .. + 3` of it: 4
  // consecutive elements each, 32 lanes covering 128 elements.
  const int group = lane / 8;
  const int base = (lane % 8) * 4;

  float v[4];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[i] = __bfloat162float(row[group * kSfGroupK + base + i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  // Reduce across the 8 lanes of this group (lanes group*8 .. group*8+7).
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
  }
  amax = fmaxf(amax, 1e-4f);

  // ceil_to_ue8m0(amax / 448)
  const float raw = amax / 448.0f;
  unsigned int bits = __float_as_uint(raw) & 0x7fffffffu;
  int exp = (int)((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u ? 1 : 0);
  exp = exp < 1 ? 1 : (exp > 254 ? 254 : exp);
  const float sf = __uint_as_float((unsigned int)exp << 23);
  const float inv_sf = 1.0f / sf;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_fp8_storage_t q =
        __nv_cvt_float_to_fp8(v[i] * inv_sf, __NV_SATFINITE, __NV_E4M3);
    out_row[group * kSfGroupK + base + i] = (unsigned char)q;
  }

  // Lane `group * 8` of each group holds the authoritative exponent; lane 0
  // gathers the four bytes into the packed word.
  const unsigned int my_exp = (unsigned int)exp;
  const unsigned int e0 = __shfl_sync(0xffffffffu, my_exp, 0);
  const unsigned int e1 = __shfl_sync(0xffffffffu, my_exp, 8);
  const unsigned int e2 = __shfl_sync(0xffffffffu, my_exp, 16);
  const unsigned int e3 = __shfl_sync(0xffffffffu, my_exp, 24);
  if (lane == 0) {
    x_sf[(size_t)token * x_sf_stride + word] =
        (int)(e0 | (e1 << 8) | (e2 << 16) | (e3 << 24));
  }
}

// Routing arrays into the symmetric buffer. K3 carries topk ids as i32 while
// the mega kernel reads i64, so the widening happens here rather than forcing
// the router to a wider type.
__global__ void mega_write_routing_kernel(const int* __restrict__ topk_idx,
                                          const float* __restrict__ topk_weight,
                                          long long* __restrict__ dst_idx,
                                          float* __restrict__ dst_weight, int entries) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < entries;
       idx += gridDim.x * blockDim.x) {
    dst_idx[idx] = (long long)topk_idx[idx];
    dst_weight[idx] = topk_weight[idx];
  }
}

// ---------------------------------------------------------------------------
// Host-side heuristics (constexpr replicas of heuristics/mega_moe.hpp)
// ---------------------------------------------------------------------------

using namespace deep_gemm;

constexpr int kSmemCapacity = 232448;  // SM100ArchSpec::smem_capacity
constexpr int kMegaBlockN = 128;
constexpr int kNumDispatchThreads = 128;
constexpr int kNumNonEpilogueThreads = 128;

constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }
constexpr int alignup(int a, int b) { return cdiv(a, b) * b; }

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
inline int mega_block_config_index(int num_tokens, int num_ranks, int num_experts,
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

// ---------------------------------------------------------------------------
// AOT instantiations
// ---------------------------------------------------------------------------

constexpr int kNumRanks = 1;
constexpr int kMaxTokensPerRank = 384;  // layout::kLCMCandidateBlockM
constexpr int kGb300Sms = 152;

constexpr int kRingTokens = mega_ring_tokens(kNumRanks, kNumExperts, kMaxTokensPerRank, kNumTopk,
                                             kHidden, kIntermediate, kGb300Sms);
constexpr int kSfRingTokens = mega_sf_ring_tokens(kRingTokens);
constexpr int kBytesPerPull = mega_bytes_per_pull(kHidden);

// One AOT kernel: a ladder entry crossed with an activation.
template <int kCfgIdx, bool kSitu>
struct MegaKernel {
  static constexpr BlockConfig kCfg = kBlockLadder[kCfgIdx];
  static constexpr int kBlockM = kCfg.block_m;
  static constexpr int kBlockK = kCfg.block_k;
  static constexpr int kStoreBlockM = kCfg.store_block_m;
  static constexpr int kEpilogueThreads = kCfg.num_epilogue_warpgroups * 128;
  // SM100ArchSpec::get_sf_uttcp_aligned_block_sizes for MXFP8FP4.
  static constexpr int kSfBlockM = alignup(kBlockM, 128);
  static constexpr int kSfBlockN = alignup(kMegaBlockN, 128);
  static constexpr PipelineConfig kPipe =
      mega_pipeline(kNumExperts, kBlockM, kMegaBlockN, kBlockK, kStoreBlockM, kSfBlockM,
                    kSfBlockN, kNumDispatchThreads / 32, kEpilogueThreads / 32, kBytesPerPull);
  static constexpr int kSmemSize = kPipe.smem_size;
  static constexpr int kNumThreads =
      kNumDispatchThreads + kNumNonEpilogueThreads + kEpilogueThreads;

  static_assert(kPipe.num_stages >= 2, "MegaMoE pipeline needs at least 2 stages");
  static_assert(kSmemSize <= kSmemCapacity, "MegaMoE smem budget overflow");

  static constexpr auto kFn = &sm100_fp8_fp4_mega_moe_impl<
      kMaxTokensPerRank, kHidden, kIntermediate, kNumExperts, /*shared=*/0, kNumTopk, kBlockM,
      kMegaBlockN, kBlockK, kStoreBlockM, kSfBlockM, kSfBlockN, kRingTokens, kSfRingTokens,
      kPipe.num_stages, kBytesPerPull, kNumDispatchThreads, kNumNonEpilogueThreads,
      kEpilogueThreads, kGb300Sms, kNumRanks, /*clamp=*/inf, kSitu, /*fast_math=*/false>;
};

// Only the first three ladder entries are reachable at
// num_max_tokens_per_rank == 384 with 224 experts and topk 16 (the 96/128/192
// entries need > 455 tokens), so those are the only ones instantiated.
constexpr int kNumReachableConfigs = 3;

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------

// K-major packed-FP4 TMA descriptor. `make_tma_2d_desc_raw` only knows the
// `DgDtype` enum, which has no sub-byte member; this reproduces what the
// torch-side path emits for a `kPackedFP4` tensor of shape [rows, k / 2].
CUresult make_fp4_k_major_tma_desc(CUtensorMap* out, const void* ptr, int shape_k, long long rows,
                                   int smem_outer_dim) {
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
CUtensorMap make_sf_desc(void* ptr, int shape_mn, int shape_k, int block_mn, int num_groups,
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

template <typename Kernel>
CUresult launch_mega(unsigned short* y, const unsigned char* l1_weights, const int* l1_weights_sf,
                     const unsigned char* l2_weights, const int* l2_weights_sf,
                     const MegaBuffers& buffers, const long long* symm_ptrs, int num_ranks,
                     int rank_idx, int num_tokens, int* cumulative_stats, cudaStream_t stream) {
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
        (long long)kNumExperts * (long long)(kIntermediate * 2), kLoadBlockN);
    if (err != CUDA_SUCCESS) return err;
  }
  CUtensorMap tm_l1_weights_sf = make_sf_desc(const_cast<int*>(l1_weights_sf), kIntermediate * 2,
                                              kHidden, kMegaBlockN, kNumExperts, kSfSmemOuter);
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
                                                   (long long)kNumExperts * (long long)kHidden,
                                                   kLoadBlockN);
    if (err != CUDA_SUCCESS) return err;
  }
  CUtensorMap tm_l2_weights_sf = make_sf_desc(const_cast<int*>(l2_weights_sf), kHidden,
                                              kIntermediate, kMegaBlockN, kNumExperts, kSfSmemOuter);

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

  deep_gemm::layout::SymBuffer<kNumRanks> sym_buffer;
  sym_buffer.rank_idx = (uint32_t)rank_idx;
  sym_buffer.base = symm_ptrs[rank_idx];
  for (uint32_t i = 0; i < deep_gemm::layout::kNumMaxRanks; ++i) {
    sym_buffer.offsets[i] = i < (uint32_t)num_ranks ? symm_ptrs[i] - sym_buffer.base : 0;
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

template <bool kSitu>
CUresult launch_mega_by_config(int cfg_idx, unsigned short* y, const unsigned char* l1_weights,
                               const int* l1_weights_sf, const unsigned char* l2_weights,
                               const int* l2_weights_sf, const MegaBuffers& buffers,
                               const long long* symm_ptrs, int num_ranks, int rank_idx,
                               int num_tokens, int* cumulative_stats, cudaStream_t stream) {
  switch (cfg_idx) {
    case 0:
      return launch_mega<MegaKernel<0, kSitu>>(y, l1_weights, l1_weights_sf, l2_weights,
                                               l2_weights_sf, buffers, symm_ptrs, num_ranks,
                                               rank_idx, num_tokens, cumulative_stats, stream);
    case 1:
      return launch_mega<MegaKernel<1, kSitu>>(y, l1_weights, l1_weights_sf, l2_weights,
                                               l2_weights_sf, buffers, symm_ptrs, num_ranks,
                                               rank_idx, num_tokens, cumulative_stats, stream);
    case 2:
      return launch_mega<MegaKernel<2, kSitu>>(y, l1_weights, l1_weights_sf, l2_weights,
                                               l2_weights_sf, buffers, symm_ptrs, num_ranks,
                                               rank_idx, num_tokens, cumulative_stats, stream);
    default:
      return CUDA_ERROR_NOT_SUPPORTED;
  }
}

#endif  // K3_MEGA_MOE_SM100F

}  // namespace

extern "C" {

// Token-count alignment the mega API enforces (`layout::kLCMCandidateBlockM`).
int k3_mega_token_alignment(void) { return 384; }

#ifdef K3_MEGA_MOE_SM100F

// Symmetric-buffer sizing: total bytes plus the 12 sub-buffer byte offsets, in
// the same order the Python wrapper slices them:
//   x, x_sf, topk_idx, topk_weights,
//   shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
//   l1_acts, l1_acts_sf, l2_acts, l2_acts_sf
CUresult k3_mega_symm_buffer_layout_cuda(int num_ranks, int num_experts,
                                         int num_max_tokens_per_rank, int num_topk, int hidden,
                                         int intermediate_hidden, int num_sms,
                                         unsigned long long* out_num_bytes,
                                         unsigned long long* out_offsets, int* out_ring_tokens,
                                         int* out_sf_ring_tokens) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (out_num_bytes == nullptr || out_offsets == nullptr || out_ring_tokens == nullptr ||
      out_sf_ring_tokens == nullptr || num_ranks <= 0 || num_experts <= 0 || num_topk <= 0 ||
      num_max_tokens_per_rank <= 0 || hidden <= 0 || intermediate_hidden <= 0 || num_sms <= 0 ||
      num_experts % num_ranks != 0 || num_max_tokens_per_rank % 384 != 0 || hidden % 128 != 0 ||
      intermediate_hidden % 128 != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int ring = mega_ring_tokens(num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
                                    hidden, intermediate_hidden, num_sms);
  const int sf_ring = mega_sf_ring_tokens(ring);
  if (sf_ring % 4 != 0) return CUDA_ERROR_INVALID_VALUE;

  const auto buffer = deep_gemm::layout::MegaMoEBuffer(
      nullptr, (uint32_t)hidden, (uint32_t)intermediate_hidden, (uint32_t)num_ranks,
      (uint32_t)num_experts, (uint32_t)num_max_tokens_per_rank, (uint32_t)num_topk, (uint32_t)ring,
      (uint32_t)sf_ring, /*with_sf=*/true, /*num_shared_experts=*/0);

  *out_num_bytes = (unsigned long long)buffer.get_num_bytes();
  *out_ring_tokens = ring;
  *out_sf_ring_tokens = sf_ring;

  auto off = [](const void* p) { return (unsigned long long)reinterpret_cast<uintptr_t>(p); };
  out_offsets[0] = off(buffer.input_token_buffer.base);
  out_offsets[1] = off(buffer.input_sf_buffer.base);
  out_offsets[2] = off(buffer.input_topk_idx_buffer.base);
  out_offsets[3] = off(buffer.input_topk_weights_buffer.base);
  out_offsets[4] = off(buffer.shared_l1_token_buffer.base);
  out_offsets[5] = off(buffer.shared_l1_sf_buffer.base);
  out_offsets[6] = off(buffer.shared_l2_token_buffer.base);
  out_offsets[7] = off(buffer.shared_l2_sf_buffer.base);
  out_offsets[8] = off(buffer.l1_token_buffer.base);
  out_offsets[9] = off(buffer.l1_sf_buffer.base);
  out_offsets[10] = off(buffer.l2_token_buffer.base);
  out_offsets[11] = off(buffer.l2_sf_buffer.base);
  return CUDA_SUCCESS;
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

CUresult k3_mega_prepare_l1_weights_cuda(const unsigned char* src, unsigned char* dst, int groups,
                                         int n, int k, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (src == nullptr || dst == nullptr || groups <= 0 || n <= 0 || k <= 0 || k % 2 != 0 ||
      n % (2 * kInterleaveGran) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int row_bytes = k / 2;
  const size_t elems = (size_t)n * (size_t)row_bytes;
  const int threads = 256;
  const size_t needed = (elems + threads - 1) / threads;
  const int blocks = (int)(needed < 1024 ? needed : 1024);
  mega_interleave_l1_weights_kernel<<<dim3(blocks, groups), threads, 0, stream>>>(src, dst, n,
                                                                                 row_bytes);
  return consume_last_cuda_error();
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

// `interleave != 0` selects the L1 pipeline (gate/up interleave then UTCCP
// transpose); otherwise only the UTCCP transpose is applied (L2).
CUresult k3_mega_prepare_sf_cuda(const unsigned char* sf, int* out, int groups, int n, int k,
                                 int interleave, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (sf == nullptr || out == nullptr || groups <= 0 || n <= 0 || k <= 0 || n % 128 != 0 ||
      k % kSfWordK != 0 || (interleave != 0 && n % (2 * kInterleaveGran) != 0)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const size_t elems = (size_t)(k / kSfWordK) * (size_t)n;
  const int threads = 256;
  const size_t needed = (elems + threads - 1) / threads;
  const int blocks = (int)(needed < 1024 ? needed : 1024);
  if (interleave != 0) {
    mega_prepare_sf_kernel<true><<<dim3(blocks, groups), threads, 0, stream>>>(sf, out, n, k);
  } else {
    mega_prepare_sf_kernel<false><<<dim3(blocks, groups), threads, 0, stream>>>(sf, out, n, k);
  }
  return consume_last_cuda_error();
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

CUresult k3_mega_quant_x_cuda(const void* x, unsigned char* x_fp8, int* x_sf, int num_tokens,
                              int hidden, int x_stride, int x_sf_stride, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (x == nullptr || x_fp8 == nullptr || x_sf == nullptr || num_tokens < 0 || hidden <= 0 ||
      hidden % kSfWordK != 0 || x_stride < hidden || x_sf_stride < hidden / kSfWordK) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  const long long warps = (long long)num_tokens * (hidden / kSfWordK);
  const int threads = 256;
  const int warps_per_block = threads / 32;
  const long long blocks = (warps + warps_per_block - 1) / warps_per_block;
  mega_quant_x_kernel<<<(int)blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), x_fp8, x_sf, num_tokens, hidden, x_stride,
      x_sf_stride);
  return consume_last_cuda_error();
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

CUresult k3_mega_write_routing_cuda(const int* topk_idx, const float* topk_weight,
                                    long long* dst_idx, float* dst_weight, int num_tokens,
                                    int num_topk, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (topk_idx == nullptr || topk_weight == nullptr || dst_idx == nullptr ||
      dst_weight == nullptr || num_tokens < 0 || num_topk <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  const int entries = num_tokens * num_topk;
  const int threads = 256;
  const int blocks = (entries + threads - 1) / threads;
  mega_write_routing_kernel<<<blocks, threads, 0, stream>>>(topk_idx, topk_weight, dst_idx,
                                                            dst_weight, entries);
  return consume_last_cuda_error();
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

// `symm_ptrs` is the per-rank base pointer table (`buffer_ptrs` upstream); at
// ep_size == 1 it is a single entry holding this rank's own slab.
CUresult k3_mega_moe_launch_cuda(unsigned short* y, const unsigned char* l1_weights,
                                 const int* l1_weights_sf, const unsigned char* l2_weights,
                                 const int* l2_weights_sf, const long long* symm_ptrs,
                                 const unsigned long long* symm_offsets, int num_ranks,
                                 int rank_idx, int num_max_tokens_per_rank, int num_tokens,
                                 int num_experts, int num_topk, int hidden,
                                 int intermediate_hidden, int num_sms, int activation,
                                 int* cumulative_stats, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (y == nullptr || l1_weights == nullptr || l1_weights_sf == nullptr ||
      l2_weights == nullptr || l2_weights_sf == nullptr || symm_ptrs == nullptr ||
      symm_offsets == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_ranks != kNumRanks || rank_idx != 0 || num_experts != kNumExperts ||
      num_topk != kNumTopk || hidden != kHidden || intermediate_hidden != kIntermediate ||
      num_max_tokens_per_rank != kMaxTokensPerRank || num_sms != kGb300Sms) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  if (num_tokens < 0 || num_tokens > num_max_tokens_per_rank) return CUDA_ERROR_INVALID_VALUE;

  auto* base = reinterpret_cast<unsigned char*>((uintptr_t)symm_ptrs[rank_idx]);
  MegaBuffers buffers{};
  buffers.base = base;
  buffers.l1_acts = base + symm_offsets[8];
  buffers.l1_acts_sf = base + symm_offsets[9];
  buffers.l2_acts = base + symm_offsets[10];
  buffers.l2_acts_sf = base + symm_offsets[11];

  const int cfg_idx = mega_block_config_index(num_tokens, num_ranks, num_experts, num_topk);
  if (cfg_idx >= kNumReachableConfigs) return CUDA_ERROR_NOT_SUPPORTED;

  if (activation == kActSitu) {
    return launch_mega_by_config<true>(cfg_idx, y, l1_weights, l1_weights_sf, l2_weights,
                                       l2_weights_sf, buffers, symm_ptrs, num_ranks, rank_idx,
                                       num_tokens, cumulative_stats, stream);
  }
  if (activation == kActSwiglu) {
    return launch_mega_by_config<false>(cfg_idx, y, l1_weights, l1_weights_sf, l2_weights,
                                        l2_weights_sf, buffers, symm_ptrs, num_ranks, rank_idx,
                                        num_tokens, cumulative_stats, stream);
  }
  return CUDA_ERROR_INVALID_VALUE;
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

#else  // !K3_MEGA_MOE_SM100F

CUresult k3_mega_symm_buffer_layout_cuda(int /*num_ranks*/, int /*num_experts*/,
                                         int /*num_max_tokens_per_rank*/, int /*num_topk*/,
                                         int /*hidden*/, int /*intermediate_hidden*/,
                                         int /*num_sms*/, unsigned long long* /*out_num_bytes*/,
                                         unsigned long long* /*out_offsets*/,
                                         int* /*out_ring_tokens*/, int* /*out_sf_ring_tokens*/) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult k3_mega_prepare_l1_weights_cuda(const unsigned char* /*src*/, unsigned char* /*dst*/,
                                         int /*groups*/, int /*n*/, int /*k*/,
                                         cudaStream_t /*stream*/) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult k3_mega_prepare_sf_cuda(const unsigned char* /*sf*/, int* /*out*/, int /*groups*/,
                                 int /*n*/, int /*k*/, int /*interleave*/,
                                 cudaStream_t /*stream*/) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult k3_mega_quant_x_cuda(const void* /*x*/, unsigned char* /*x_fp8*/, int* /*x_sf*/,
                              int /*num_tokens*/, int /*hidden*/, int /*x_stride*/,
                              int /*x_sf_stride*/, cudaStream_t /*stream*/) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult k3_mega_write_routing_cuda(const int* /*topk_idx*/, const float* /*topk_weight*/,
                                    long long* /*dst_idx*/, float* /*dst_weight*/,
                                    int /*num_tokens*/, int /*num_topk*/,
                                    cudaStream_t /*stream*/) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult k3_mega_moe_launch_cuda(unsigned short* /*y*/, const unsigned char* /*l1_weights*/,
                                 const int* /*l1_weights_sf*/, const unsigned char* /*l2_weights*/,
                                 const int* /*l2_weights_sf*/, const long long* /*symm_ptrs*/,
                                 const unsigned long long* /*symm_offsets*/, int /*num_ranks*/,
                                 int /*rank_idx*/, int /*num_max_tokens_per_rank*/,
                                 int /*num_tokens*/, int /*num_experts*/, int /*num_topk*/,
                                 int /*hidden*/, int /*intermediate_hidden*/, int /*num_sms*/,
                                 int /*activation*/, int* /*cumulative_stats*/,
                                 cudaStream_t /*stream*/) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

#endif  // K3_MEGA_MOE_SM100F

}  // extern "C"
