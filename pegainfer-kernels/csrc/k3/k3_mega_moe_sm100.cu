// Kimi-K3 routed-expert MoE: DeepGEMM SM100 fused "MegaMoE" FP8 x FP4 kernel,
// AOT-instantiated from the vendored device headers (no JIT, no torch).
//
// The mega kernel fuses dispatch + W13 GEMM + activation + mid-quantization +
// W2 GEMM + combine into a single persistent grid. Everything it needs beyond
// the weights lives in one flat "symmetric" byte slab whose sub-buffer offsets
// are pure host arithmetic over the shapes; at ep_size == 1 the slab is a plain
// device allocation and the kernel's cross-rank barriers degrade to grid-local
// synchronisation (`layout::SymBuffer<1>::map` is the identity). Above one rank
// each rank owns one such slab on its own device and the launch is handed the
// whole base-pointer table; the kernel's NVLink barriers pair the ranks, so the
// host issues no collective at all. The table's pointers only have to be
// dereferenceable from this rank's context — plain peer access for an
// in-process group, imported `CU_MEM_HANDLE_TYPE_FABRIC` mappings for a
// cross-machine one (`k3_mega_fabric.cu`); the kernel cannot tell the
// difference.
//
// Cross-rank addressing is layout-only: every access a rank makes into a peer's
// slab (`sym_buffer.map(...)`) targets a region whose offset and stride come
// from `MegaMoEBuffer(hidden, intermediate, ranks, experts, max_tokens, topk,
// ring_tokens, sf_ring_tokens)`. None of those terms involve BLOCK_M, BLOCK_N,
// the stage count or the epilogue width. The block-config-dependent quantities
// (pool block indices, the L1/L2 ring counters, SF paging) are only ever used
// against the rank's OWN workspace and rings. A sender therefore never needs to
// know the receiver's block config.
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
//   in `k3_mega_moe_sm100_common.cuh`. They are `constexpr` so the AOT
//   instantiations and the launch-time selection provably agree.
// * `sm100_fp8_fp4_mega_moe` (csrc/jit_kernels/impls/...) ->
//   `k3_mega::launch_mega`, including all 9 distinct TMA descriptors. The
//   shared-expert descriptors alias the routed ones, mirroring upstream's
//   `num_shared_experts == 0` path.
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
// hidden 3584 (K3 latent), intermediate 3072, topk 16, num_max_tokens_per_rank
// 16896 (the chunked-prefill ceiling; 44x `kLCMCandidateBlockM`, the token
// alignment the upstream API enforces), 152 SMs (GB300). The GLOBAL expert
// count and the rank count together name a world, and the worlds are split
// across translation units so nvcc compiles them in parallel:
//
//   this TU                       224 experts x ranks {1, 4} — the bring-up
//                                 single-tray worlds, situ + swiglu
//   k3_mega_moe_sm100_wide224.cu  224 experts x ranks {8, 16} — the pruned
//                                 checkpoint stretched over 2 / 4 machines
//                                 (its EP16 shard is shape-identical to the
//                                 full model's, so it is the cross-machine
//                                 transport gate), situ only
//   k3_mega_moe_sm100_wide896.cu  896 experts x ranks {8, 16, 32, 64} — the
//                                 full checkpoint's serving worlds, situ only
//
// At ranks 1 the launch picks among the block configs by live token count
// under `get_block_config_for_mega_moe`, so a single-rank launch stays
// bit-identical to what upstream's Python wrapper would run; at the protocol
// maximum the whole six-entry ladder is reachable and instantiated.
// Every multi-rank world has exactly ONE config, taken at the protocol maximum
// rather than from the live token count — see `k3_mega::pinned_config`.
//
// build.rs compiles these TUs for sm_100f ONLY when a sm_100-family target
// exists; otherwise every entry point compiles as a NOT_SUPPORTED stub.

#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cstdint>

#ifdef K3_MEGA_MOE_SM100F
#include "k3_mega_moe_sm100_common.cuh"
#endif

namespace {

// Duplicated from the common header so the transform entry points (which
// compile on every architecture) do not depend on the SM100F-only include.
constexpr int kTransformSfGroupK = 32;
constexpr int kTransformSfPerWord = 4;
constexpr int kTransformSfWordK = kTransformSfGroupK * kTransformSfPerWord;  // 128

// Weight-row interleave granularity for the fused gate|up projection.
constexpr int kInterleaveGran = 8;

CUresult transform_map_cuda_error(cudaError_t err) {
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  if (err == cudaErrorNotSupported) return CUDA_ERROR_NOT_SUPPORTED;
  return CUDA_ERROR_LAUNCH_FAILED;
}

CUresult consume_last_cuda_error() {
  return transform_map_cuda_error(cudaGetLastError());
}

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
  const int sf_cols = k / kTransformSfGroupK;
  const int packed_cols = k / kTransformSfWordK;
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
        group_sf + (size_t)src_row * sf_cols + (size_t)word * kTransformSfPerWord;
    unsigned int packed = 0;
#pragma unroll
    for (int j = 0; j < kTransformSfPerWord; ++j) {
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
  const int words_per_token = hidden / k3_mega::kSfWordK;
  const int warps_per_block = blockDim.x / 32;
  const int warp_id = (blockIdx.x * warps_per_block) + (threadIdx.x / 32);
  const int lane = threadIdx.x % 32;
  const long long total_warps = (long long)num_tokens * words_per_token;
  if (warp_id >= total_warps) return;

  const int token = warp_id / words_per_token;
  const int word = warp_id % words_per_token;

  const __nv_bfloat16* row = x + (size_t)token * x_stride + (size_t)word * k3_mega::kSfWordK;
  unsigned char* out_row = x_fp8 + (size_t)token * hidden + (size_t)word * k3_mega::kSfWordK;

  // Lane `l` owns group `l / 8` and element `(l % 8) * 4 .. + 3` of it: 4
  // consecutive elements each, 32 lanes covering 128 elements.
  const int group = lane / 8;
  const int base = (lane % 8) * 4;

  float v[4];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[i] = __bfloat162float(row[group * k3_mega::kSfGroupK + base + i]);
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
    out_row[group * k3_mega::kSfGroupK + base + i] = (unsigned char)q;
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
// This TU's worlds: 224 experts at ranks 1 and 4
// ---------------------------------------------------------------------------

constexpr int kPrunedExperts = 224;

// Single rank: with 224 experts and topk 16 every ladder entry is reachable
// below the protocol maximum (the thresholds land at 119/231/455/903/1351
// tokens), so the whole ladder is instantiated, and the launch picks among the
// entries by the upstream heuristic so a single-rank launch stays bit-identical
// to what DeepGEMM's own Python wrapper would have run.
static_assert(k3_mega::mega_block_config_index(k3_mega::kMaxTokensPerRank, 1, kPrunedExperts,
                                               k3_mega::kNumTopk) < k3_mega::kNumBlockConfigs,
              "the single-rank protocol max must land inside the instantiated ladder");

template <bool kSitu>
CUresult launch_mega_ranks1(int cfg_idx, unsigned short* y, const unsigned char* l1_weights,
                            const int* l1_weights_sf, const unsigned char* l2_weights,
                            const int* l2_weights_sf, const k3_mega::MegaBuffers& buffers,
                            const long long* symm_ptrs, int rank_idx, int num_tokens,
                            int* cumulative_stats, cudaStream_t stream) {
  using k3_mega::MegaKernel;
  using k3_mega::launch_mega;
  switch (cfg_idx) {
    case 0:
      return launch_mega<kPrunedExperts, 1, MegaKernel<kPrunedExperts, 1, 0, kSitu>>(
          y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf, buffers, symm_ptrs, rank_idx,
          num_tokens, cumulative_stats, stream);
    case 1:
      return launch_mega<kPrunedExperts, 1, MegaKernel<kPrunedExperts, 1, 1, kSitu>>(
          y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf, buffers, symm_ptrs, rank_idx,
          num_tokens, cumulative_stats, stream);
    case 2:
      return launch_mega<kPrunedExperts, 1, MegaKernel<kPrunedExperts, 1, 2, kSitu>>(
          y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf, buffers, symm_ptrs, rank_idx,
          num_tokens, cumulative_stats, stream);
    case 3:
      return launch_mega<kPrunedExperts, 1, MegaKernel<kPrunedExperts, 1, 3, kSitu>>(
          y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf, buffers, symm_ptrs, rank_idx,
          num_tokens, cumulative_stats, stream);
    case 4:
      return launch_mega<kPrunedExperts, 1, MegaKernel<kPrunedExperts, 1, 4, kSitu>>(
          y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf, buffers, symm_ptrs, rank_idx,
          num_tokens, cumulative_stats, stream);
    case 5:
      return launch_mega<kPrunedExperts, 1, MegaKernel<kPrunedExperts, 1, 5, kSitu>>(
          y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf, buffers, symm_ptrs, rank_idx,
          num_tokens, cumulative_stats, stream);
    default:
      return CUDA_ERROR_NOT_SUPPORTED;
  }
}

#endif  // K3_MEGA_MOE_SM100F

}  // namespace

#ifdef K3_MEGA_MOE_SM100F

// The wide worlds, each owned by its own TU (NOT_SUPPORTED stubs when that TU
// is built without sm_100f).
extern "C" CUresult k3_mega_moe_launch_wide224(
    int activation, int num_ranks, unsigned short* y, const unsigned char* l1_weights,
    const int* l1_weights_sf, const unsigned char* l2_weights, const int* l2_weights_sf,
    const void* buffers, const long long* symm_ptrs, int rank_idx, int num_tokens,
    int* cumulative_stats, cudaStream_t stream);
extern "C" CUresult k3_mega_moe_launch_wide896(
    int activation, int num_ranks, unsigned short* y, const unsigned char* l1_weights,
    const int* l1_weights_sf, const unsigned char* l2_weights, const int* l2_weights_sf,
    const void* buffers, const long long* symm_ptrs, int rank_idx, int num_tokens,
    int* cumulative_stats, cudaStream_t stream);

#endif  // K3_MEGA_MOE_SM100F

extern "C" {

// Token-count alignment the mega API enforces (`layout::kLCMCandidateBlockM`).
int k3_mega_token_alignment(void) { return 384; }

// Token capacity one rank's slab and the AOT kernels are built for
// (`num_max_tokens_per_rank`). The launch accepts exactly this value — the
// ring capacities derived from it are kernel template parameters — so slabs
// are allocated at exactly this size, whatever the executor's live batch is.
int k3_mega_max_tokens_per_rank(void) { return 16896; }

// Whether the AOT matrix carries a kernel for this world (GLOBAL expert count
// x rank count, situ activation). One definition of the supported set — the
// Rust side asks instead of mirroring the list.
int k3_mega_world_supported(int num_experts, int num_ranks) {
  if (num_experts == 224) {
    return num_ranks == 1 || num_ranks == 4 || num_ranks == 8 || num_ranks == 16;
  }
  if (num_experts == 896) {
    return num_ranks == 8 || num_ranks == 16 || num_ranks == 32 || num_ranks == 64;
  }
  return 0;
}

// Let this rank's device read and write `peer_ordinal`'s memory, and let
// `peer_ordinal` read and write ours.
//
// Two separate things have to be opened, and getting only the first is the
// classic way to earn an illegal address here:
//
//  * `cudaDeviceEnablePeerAccess` makes the CALLING device's context able to
//    address `peer_ordinal`'s allocations. That is what the kernel's
//    `SymBuffer::map` arithmetic relies on.
//  * `cudaMemPoolSetAccess` makes THIS device's stream-ordered pool allocations
//    visible from `peer_ordinal`. Peer-access enablement explicitly does not
//    cover memory-pool allocations, and every buffer here comes from the
//    stream-ordered allocator, so without this the peers hold a pointer they
//    may not dereference.
//
// The pool grant only reliably covers allocations made AFTER it, so the caller
// has to open every device it will ever expose memory to before it allocates
// the slab. Idempotent, and a no-op for the self pair.
//
// In-process groups only: a cross-machine group's slabs are fabric mappings
// (`k3_mega_fabric.cu`), whose access grants travel with the mapping.
CUresult k3_mega_open_peer_access(int self_ordinal, int peer_ordinal) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (self_ordinal < 0 || peer_ordinal < 0) return CUDA_ERROR_INVALID_VALUE;
  if (self_ordinal == peer_ordinal) return CUDA_SUCCESS;
  int reachable = 0;
  const cudaError_t query = cudaDeviceCanAccessPeer(&reachable, self_ordinal, peer_ordinal);
  if (query != cudaSuccess) return transform_map_cuda_error(query);
  if (reachable == 0) return CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;

  const cudaError_t bind = cudaSetDevice(self_ordinal);
  if (bind != cudaSuccess) return transform_map_cuda_error(bind);
  const cudaError_t enable = cudaDeviceEnablePeerAccess(peer_ordinal, 0);
  if (enable == cudaErrorPeerAccessAlreadyEnabled) {
    (void)cudaGetLastError();
  } else if (enable != cudaSuccess) {
    return transform_map_cuda_error(enable);
  }

  cudaMemPool_t pool = nullptr;
  const cudaError_t pool_err = cudaDeviceGetDefaultMemPool(&pool, self_ordinal);
  if (pool_err != cudaSuccess) return transform_map_cuda_error(pool_err);
  cudaMemAccessDesc desc{};
  desc.location.type = cudaMemLocationTypeDevice;
  desc.location.id = peer_ordinal;
  desc.flags = cudaMemAccessFlagsProtReadWrite;
  return transform_map_cuda_error(cudaMemPoolSetAccess(pool, &desc, 1));
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

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
  const int ring =
      k3_mega::mega_ring_tokens(num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
                                hidden, intermediate_hidden, num_sms);
  const int sf_ring = k3_mega::mega_sf_ring_tokens(ring);
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
      k % kTransformSfWordK != 0 || (interleave != 0 && n % (2 * kInterleaveGran) != 0)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const size_t elems = (size_t)(k / kTransformSfWordK) * (size_t)n;
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
      hidden % k3_mega::kSfWordK != 0 || x_stride < hidden ||
      x_sf_stride < hidden / k3_mega::kSfWordK) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  const long long warps = (long long)num_tokens * (hidden / k3_mega::kSfWordK);
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

// `symm_ptrs` is the per-rank base pointer table (`buffer_ptrs` upstream): entry
// `i` is rank `i`'s slab base, mapped into this context (peer access or an
// imported fabric mapping). At ep_size == 1 it is a single entry holding this
// rank's own slab.
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
  if (k3_mega_world_supported(num_experts, num_ranks) == 0 ||
      num_topk != k3_mega::kNumTopk || hidden != k3_mega::kHidden ||
      intermediate_hidden != k3_mega::kIntermediate ||
      num_max_tokens_per_rank != k3_mega::kMaxTokensPerRank || num_sms != k3_mega::kGb300Sms) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  if (rank_idx < 0 || rank_idx >= num_ranks || num_experts % num_ranks != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens < 0 || num_tokens > num_max_tokens_per_rank) return CUDA_ERROR_INVALID_VALUE;

  auto* base = reinterpret_cast<unsigned char*>((uintptr_t)symm_ptrs[rank_idx]);
  k3_mega::MegaBuffers buffers{};
  buffers.base = base;
  buffers.l1_acts = base + symm_offsets[8];
  buffers.l1_acts_sf = base + symm_offsets[9];
  buffers.l2_acts = base + symm_offsets[10];
  buffers.l2_acts_sf = base + symm_offsets[11];

  // Single rank follows the upstream heuristic (live token count), which is what
  // the Python parity fixture exercises. Multi-rank pins one block config for
  // every rank and every step: cross-rank writes must not depend on the
  // receiver's tiling, and a traffic-independent tile shape is what makes a
  // rank's own rows bitwise-stable as its peers' batches move.
  if (num_experts == kPrunedExperts && num_ranks == 1) {
    const int cfg_idx =
        k3_mega::mega_block_config_index(num_tokens, num_ranks, num_experts, num_topk);
    if (activation == k3_mega::kActSitu) {
      return launch_mega_ranks1<true>(cfg_idx, y, l1_weights, l1_weights_sf, l2_weights,
                                      l2_weights_sf, buffers, symm_ptrs, rank_idx, num_tokens,
                                      cumulative_stats, stream);
    }
    if (activation == k3_mega::kActSwiglu) {
      return launch_mega_ranks1<false>(cfg_idx, y, l1_weights, l1_weights_sf, l2_weights,
                                       l2_weights_sf, buffers, symm_ptrs, rank_idx, num_tokens,
                                       cumulative_stats, stream);
    }
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_experts == kPrunedExperts && num_ranks == 4) {
    return k3_mega::launch_mega_pinned<kPrunedExperts, 4, /*kSituOnly=*/false>(
        activation, y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf, buffers, symm_ptrs,
        rank_idx, num_tokens, cumulative_stats, stream);
  }
  if (num_experts == kPrunedExperts) {
    return k3_mega_moe_launch_wide224(activation, num_ranks, y, l1_weights, l1_weights_sf,
                                      l2_weights, l2_weights_sf, &buffers, symm_ptrs, rank_idx,
                                      num_tokens, cumulative_stats, stream);
  }
  return k3_mega_moe_launch_wide896(activation, num_ranks, y, l1_weights, l1_weights_sf,
                                    l2_weights, l2_weights_sf, &buffers, symm_ptrs, rank_idx,
                                    num_tokens, cumulative_stats, stream);
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
