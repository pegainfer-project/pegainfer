// Kimi-K3 routed-expert GEMM: DeepGEMM SM100 MGroupedMasked FP8 x FP4
// (tcgen05), AOT-instantiated from the vendored device headers (no JIT, no
// torch). The A side is the same FP8 e4m3 / per-1x128 UE8M0 activation recipe
// GLM5.2 uses; the B side is MXFP4 weights (e2m1, K-major, 2 values per byte)
// with group-32 UE8M0 scale factors.
//
// ---------------------------------------------------------------------------
// Masked layout contract
// ---------------------------------------------------------------------------
// Mirrors DeepGEMM's own host wrapper `sm100_m_grouped_fp8_fp4_gemm_masked_1d1d`
// (csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:244-286). `groups` is
// always concatenated onto the OUTER dimension of every operand, and the
// masked scheduler derives the per-group offsets from `masked_m`.
//
//   activation  [groups, masked_cap, k]              fp8 e4m3, K-major
//   act scales  [groups, ceil(k/512), masked_cap]    i32, MN-major; each i32
//                 packs 4 consecutive 1x128 K-blocks' UE8M0 exponent bytes,
//                 LSB first (u32 = b0 | b1<<8 | b2<<16 | b3<<24)
//   weight      [groups, n, k]                       fp4 e2m1, K-major,
//                 2 values per byte (so k/2 bytes per row)
//   wt scales   [groups, ceil(k/128), n]             i32, MN-major; each i32
//                 packs 4 consecutive group-32 K-blocks' UE8M0 exponent bytes,
//                 same LSB-first order (see `k3_fp4_sf_prepare_cuda`)
//   masked_m    i32[groups]                          real rows per expert
//   out         [groups, masked_cap, n]              bf16
//
// ---------------------------------------------------------------------------
// B-side scale-factor layout (the part that differs from the FP8xFP8 path)
// ---------------------------------------------------------------------------
// DeepGEMM never exposes a "granularity" flag for FP4: the SF layout is fully
// determined by the `kGranKB` template argument and the SF TMA descriptor.
// `sm100_fp8_fp4_gemm_1d1d.cuh:66` sets `kNumSFBStagesPerLoad = kGranKB == 32
// ? 1 : 4`, and `:127` computes `shape_sfb_k = ceil_div(shape_k, kGranKB * 4)`.
// With `kGranKB = 32` that means one i32 word covers 128 K elements (4 x 32),
// and a fresh SFB TMA is issued for every 128-wide K block.
//
// The host-side layout transform is `make_tma_sf_desc` (runtime_utils.hpp:286)
// plus the shape/stride contract asserted by `check_sf_layout`
// (csrc/utils/layout.hpp:102-116):
//   * i32 element type (torch::kInt), so the k extent is
//     `ceil_div(k, gran_k * 4)` = `k / 128` for gran 32;
//   * the MN axis is CONTIGUOUS (`stride(-2) == 1`) and the K axis strides by
//     `get_tma_aligned_size(mn, 4)` = `align(n, 4)` — i.e. the tensor is
//     physically `[groups][k / 128][align(n, 4)]`, "MN-major";
//   * the SF TMA descriptor is unswizzled with an smem box of
//     `(BLOCK_N, 1)`, so the kernel reads BLOCK_N contiguous i32 words per
//     K block.
// For K3 both N values are multiples of 4, so `align(n, 4) == n` and no row
// padding is needed. `k3_fp4_sf_prepare_cuda` below performs the checkpoint ->
// runtime repack (K-major exponent bytes -> MN-major packed i32).
//
// ---------------------------------------------------------------------------
// Template parameterization
// ---------------------------------------------------------------------------
// Upstream's FP8xFP4 kernel has no `kIsFP4` boolean: FP4-ness is carried by the
// operand TYPE plus its K granularity. The JIT wrapper stamps out
// `to_string(b_dtype)` (runtime_utils.hpp:71), which for a packed-FP4 tensor is
// `cutlass::detail::float_e2m1_unpacksmem_t` — a 4-bit-wide CUTLASS type whose
// C++ `sizeof` is 1, selecting the unpacked-smem TMA/MMA flavour. Everything
// else matches the GLM5.2 FP8xFP8 masked instantiation, including the derived
// shared-memory budget: `SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K *
// sizeof(b_dtype_t)` is 128*128*1 either way, so the 8-stage / 213804 B
// pipeline config carries over unchanged.
//
// Instantiation config mirrors SM100ArchSpec's single masked-layout candidate
// (heuristics/sm100.hpp:31-42): swap_ab=true, BLOCK_M/N/K=128/128/128, cluster
// (1,2) -> multicast 2 on A, LOAD_BLOCK_M=64 / LOAD_BLOCK_N=128,
// STORE_BLOCK_M=16 / STORE_BLOCK_N=128, 128B swizzles, 128+128 threads. The
// persistent scheduler bakes the SM count into its template, so B200 (148) and
// GB300 (152) get separate AOT instantiations selected explicitly at launch.
//
// `groups` is a runtime dispatch over the supported local-expert counts:
// 56 (EP4 dev / EP16 full), 112 (EP8 full), 224 (single-GPU bring-up).
//
// build.rs compiles the GEMM section for sm_100f ONLY when a sm_100-family
// target exists (tcgen05 needs the family arch; runs on sm_103). Otherwise the
// GEMM entry compiles as a NOT_SUPPORTED stub. The scale-prepare kernel is
// plain CUDA and compiles for every target.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int kKindW13 = 1;
constexpr int kKindW2 = 2;

// Per-expert K3 MoE shapes. W13 is the fused gate|up projection.
constexpr int kW13N = 6144;
constexpr int kW13K = 3584;
constexpr int kW2N = 3584;
constexpr int kW2K = 3072;

// MXFP4 scale-factor group size along K, and the number of such groups packed
// into one i32 SF word.
constexpr int kFp4SfGroupK = 32;
constexpr int kFp4SfPerWord = 4;
constexpr int kFp4SfWordK = kFp4SfGroupK * kFp4SfPerWord;  // 128

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

// Checkpoint MXFP4 weight scales -> the runtime SFB tensor.
//
// in : [groups, n, k / 32]  u8, one UE8M0 exponent byte per (row, K-group),
//                           K-major (matches the K-major packed FP4 bank)
// out: [groups, k / 128, n] i32, MN-major, 4 exponent bytes per word LSB-first
//
// This is a transpose plus a 4:1 pack, so it is a loader-time helper rather
// than a step-time kernel. Grid-strided over the output words; `gridDim.y`
// walks the expert groups.
__global__ void fp4_sf_prepare_kernel(const unsigned char* __restrict__ sf,
                                      int* __restrict__ packed, int n, int k) {
  const int sf_cols = k / kFp4SfGroupK;      // exponent bytes per row
  const int packed_cols = k / kFp4SfWordK;   // i32 words per row
  const unsigned char* group_sf =
      sf + (size_t)blockIdx.y * (size_t)n * sf_cols;
  int* group_packed = packed + (size_t)blockIdx.y * (size_t)packed_cols * n;
  const size_t per_group = (size_t)packed_cols * n;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
       idx < per_group; idx += (size_t)gridDim.x * blockDim.x) {
    const int row = idx % n;
    const int i = idx / n;
    const unsigned char* base =
        group_sf + (size_t)row * sf_cols + (size_t)i * kFp4SfPerWord;
    unsigned int word = 0;
#pragma unroll
    for (int j = 0; j < kFp4SfPerWord; ++j) {
      word |= static_cast<unsigned int>(base[j]) << (8 * j);
    }
    group_packed[(size_t)i * n + row] = static_cast<int>(word);
  }
}

}  // namespace

extern "C" {

CUresult k3_fp4_sf_prepare_cuda(const unsigned char* sf, int* packed,
                                int groups, int n, int k,
                                cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (sf == nullptr || packed == nullptr || groups <= 0 || n <= 0 || k <= 0 ||
      n % 4 != 0 || k % kFp4SfWordK != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const size_t elems = (size_t)(k / kFp4SfWordK) * n;
  const int threads = 256;
  const size_t needed = (elems + threads - 1) / threads;
  const int blocks = static_cast<int>(needed < 256 ? needed : 256);
  fp4_sf_prepare_kernel<<<dim3(blocks, groups), threads, 0, stream>>>(
      sf, packed, n, k);
  return consume_last_cuda_error();
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

}  // extern "C"

#ifdef K3_DEEPGEMM_FP8_FP4_SM100F

#include <jit_kernels/impls/runtime_utils.hpp>

#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>

namespace {

constexpr int kSmemCapacity = 232448;
constexpr int kB200Sms = 148;
constexpr int kGb300Sms = 152;

// SM100 masked FP8xFP4 GEMM, one (n, k, groups, num_sms) instantiation.
// SHAPE_M stays runtime (compiled_dims='nk'); m arrives as the masked-cap
// launch argument.
template <uint32_t N, uint32_t K, uint32_t GROUPS, uint32_t NUM_SMS>
struct MaskedFp8Fp4GemmSm100 {
  static constexpr uint32_t kShapeN = N;
  static constexpr uint32_t kShapeK = K;
  static constexpr uint32_t kGroups = GROUPS;
  static constexpr uint32_t kNumSms = NUM_SMS;
  static constexpr auto kKernel = &deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      /*gran_k A=*/128, /*gran_k B=*/kFp4SfGroupK, /*k_alignment=*/128,
      /*SHAPE_M=*/0, N, K,
      /*BLOCK_M=*/128, /*BLOCK_N=*/128, /*BLOCK_K=*/128, GROUPS,
      /*swizzle A/B/CD=*/128, 128, 128,
      /*stages=*/8,
      /*non-epilogue threads=*/128, /*epilogue threads=*/128,
      /*multicast=*/2, /*multicast on A=*/true, NUM_SMS,
      /*swap_ab=*/true, /*ensure_zero_padding=*/false,
      deep_gemm::GemmType::MGroupedMasked, /*with_accumulation=*/false,
      cutlass::float_e4m3_t, cutlass::detail::float_e2m1_unpacksmem_t,
      cutlass::bfloat16_t, deep_gemm::epilogue::transform::EpilogueIdentity>;

  // Mirrors SM100ArchSpec::get_pipeline_config for this config. Both operands
  // are 1 byte per element in smem (FP4 uses the unpacked-smem type), so the
  // budget matches the FP8xFP8 masked instantiation exactly:
  //   smem_cd       = 16*128*2B*2 stages        = 8192
  //   smem_barriers = 32*8*3 + 2*8*2 + 8        =  808
  //   smem_tmem_ptr                              =    4
  //   per stage: A 64*128 + B 128*128 + SFA 512 + SFB 512 = 25600
  static constexpr int smem_size() {
    const int smem_extra = 8192 + 808 + 4;
    const int per_stage = 64 * 128 + 128 * 128 + 128 * 4 + 128 * 4;
    return smem_extra + 8 * per_stage;  // 213804
  }
};

static_assert(
    MaskedFp8Fp4GemmSm100<kW13N, kW13K, 224, kGb300Sms>::smem_size() <=
    kSmemCapacity);

// K-major packed-FP4 TMA descriptor.
//
// DeepGEMM's torch-free `make_tma_2d_desc_raw` only knows the `DgDtype`
// enum (Int / Float / BFloat16 / Float8_e4m3), which has no sub-byte member,
// so the FP4 B descriptor is built here. The geometry reproduces what the
// torch-side `make_tma_b_desc` -> `make_tma_2d_desc` pair emits for a
// `kPackedFP4` tensor of shape `[rows, k / 2]` (runtime_utils.hpp:155-190,
// 249-267):
//   * `kPackedFP4` aliases `torch::kInt8` (csrc/utils/math.hpp:14), so
//     `elem_size == 1`; the gmem INNER extent is passed in FP4 ELEMENTS (k)
//     while the outer stride is in BYTES (k / 2);
//   * `smem_inner_dim = swizzle_mode / elem_size = 128` FP4 elements;
//   * `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B` is the "unpacked smem" flavour
//     matching `float_e2m1_unpacksmem_t`, and requires the gmem inner extent
//     to be a multiple of 128 (asserted by the torch path at :164).
CUresult make_fp4_k_major_tma_desc(CUtensorMap* out, const void* ptr,
                                   int shape_k, int rows, int smem_outer_dim) {
  if (shape_k % 128 != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(shape_k),
                                   static_cast<cuuint64_t>(rows)};
  // Row stride in bytes: two FP4 values per byte.
  const cuuint64_t gmem_strides[1] = {static_cast<cuuint64_t>(shape_k / 2)};
  const cuuint32_t smem_dims[2] = {128u,
                                   static_cast<cuuint32_t>(smem_outer_dim)};
  const cuuint32_t elem_strides[2] = {1, 1};
  return deep_gemm::lazy_cuTensorMapEncodeTiled(
      out, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 2, const_cast<void*>(ptr),
      gmem_dims, gmem_strides, smem_dims, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template <typename Gemm>
CUresult launch_masked_fp8_fp4_sm100(const unsigned char* a, const int* a_scale,
                                     const unsigned char* b,
                                     const int* b_scale, const int* masked_m,
                                     unsigned short* out, int masked_cap,
                                     cudaStream_t stream) {
  const auto func = reinterpret_cast<const void*>(Gemm::kKernel);
  const int smem_size = Gemm::smem_size();
  const cudaError_t attr_err = cudaFuncSetAttribute(
      func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  if (attr_err != cudaSuccess) {
    return map_cuda_error(attr_err);
  }

  const uint32_t n = Gemm::kShapeN, k = Gemm::kShapeK;
  const int groups = Gemm::kGroups;

  // TMA descriptors mirror sm100_m_grouped_fp8_fp4_gemm_masked_1d1d's host
  // wrapper. Built per launch on the host — a whole-step graph capture bakes
  // them into the recorded node params; pointers are the persistent per-rank
  // state buffers, so replay stays valid.
  //
  // A: K-major [masked_cap*groups, k] fp8, smem box [block_k, load_block_m].
  const auto tma_a = deep_gemm::make_tma_2d_desc_raw(
      const_cast<unsigned char*>(a), 1, deep_gemm::DgDtype::Float8_e4m3, k,
      masked_cap * groups, 128, 64, k, 128);
  // B: K-major [n*groups, k] fp4, smem box [block_k, load_block_n].
  CUtensorMap tma_b;
  const CUresult b_desc_err =
      make_fp4_k_major_tma_desc(&tma_b, b, static_cast<int>(k), n * groups,
                                /*smem_outer_dim=*/128);
  if (b_desc_err != CUDA_SUCCESS) {
    return b_desc_err;
  }
  // C/D: [masked_cap*groups, n], store box [store_block_n, store_block_m]; the
  // raw builder replaces smem_inner with swizzle/elem_size (=64 elems bf16).
  const auto tma_cd = deep_gemm::make_tma_2d_desc_raw(
      out, 2, deep_gemm::DgDtype::BFloat16, n, masked_cap * groups, 128, 16, n,
      128);
  // SFA: MN-major packed UE8M0 i32, gran-K 128 -> ceil(k/512) words per row.
  // Inner dim = mn (contiguous, stride 1), outer = packed columns * groups,
  // outer stride = mn elements, smem box = (BLOCK_M, 1), unswizzled.
  const auto tma_sfa = deep_gemm::make_tma_2d_desc_raw(
      const_cast<int*>(a_scale), 4, deep_gemm::DgDtype::Int, masked_cap,
      (k / 512) * groups, 128, 1, masked_cap, 0);
  // SFB: same MN-major packed layout at gran-K 32 -> ceil(k/128) words per row.
  const auto tma_sfb = deep_gemm::make_tma_2d_desc_raw(
      const_cast<int*>(b_scale), 4, deep_gemm::DgDtype::Int, n,
      (k / kFp4SfWordK) * groups, 128, 1, n, 0);

  // Cluster (2,1,1) (the A-side TMA multicast pair) + PDL, per DeepGEMM's own
  // launch config. The attrs array is per-call stack storage.
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {2, 1, 1};
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(Gemm::kNumSms, 1, 1);
  config.blockDim = dim3(128 + 128, 1, 1);
  config.dynamicSmemBytes = static_cast<size_t>(smem_size);
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = 2;

  uint32_t shape_m = masked_cap, shape_n = n, shape_k = k;
  int* grouped_layout = const_cast<int*>(masked_m);
  void* args[] = {
      &grouped_layout, &shape_m, &shape_n, &shape_k,
      const_cast<CUtensorMap*>(&tma_a),   &tma_b,
      const_cast<CUtensorMap*>(&tma_sfa), const_cast<CUtensorMap*>(&tma_sfb),
      const_cast<CUtensorMap*>(&tma_cd),
  };
  return map_cuda_error(cudaLaunchKernelExC(&config, func, args));
}

template <uint32_t GROUPS, uint32_t NUM_SMS>
CUresult launch_masked_fp8_fp4_sm100_groups(
    int operand_kind, const unsigned char* a, const int* a_scale,
    const unsigned char* b, const int* b_scale, const int* masked_m,
    unsigned short* out, int n, int k, int masked_cap, cudaStream_t stream) {
  if (operand_kind == kKindW13 && n == kW13N && k == kW13K) {
    return launch_masked_fp8_fp4_sm100<
        MaskedFp8Fp4GemmSm100<kW13N, kW13K, GROUPS, NUM_SMS>>(
        a, a_scale, b, b_scale, masked_m, out, masked_cap, stream);
  }
  if (operand_kind == kKindW2 && n == kW2N && k == kW2K) {
    return launch_masked_fp8_fp4_sm100<
        MaskedFp8Fp4GemmSm100<kW2N, kW2K, GROUPS, NUM_SMS>>(
        a, a_scale, b, b_scale, masked_m, out, masked_cap, stream);
  }
  return CUDA_ERROR_INVALID_VALUE;
}

template <uint32_t NUM_SMS>
CUresult launch_masked_fp8_fp4_sm100_dispatch(
    int operand_kind, const unsigned char* a, const int* a_scale,
    const unsigned char* b, const int* b_scale, const int* masked_m,
    unsigned short* out, int groups, int n, int k, int masked_cap,
    cudaStream_t stream) {
  switch (groups) {
    case 224:
      return launch_masked_fp8_fp4_sm100_groups<224, NUM_SMS>(
          operand_kind, a, a_scale, b, b_scale, masked_m, out, n, k, masked_cap,
          stream);
    case 112:
      return launch_masked_fp8_fp4_sm100_groups<112, NUM_SMS>(
          operand_kind, a, a_scale, b, b_scale, masked_m, out, n, k, masked_cap,
          stream);
    case 56:
      return launch_masked_fp8_fp4_sm100_groups<56, NUM_SMS>(
          operand_kind, a, a_scale, b, b_scale, masked_m, out, n, k, masked_cap,
          stream);
    default:
      return CUDA_ERROR_INVALID_VALUE;
  }
}

}  // namespace

extern "C" {

CUresult k3_deepgemm_sm100_masked_grouped_fp8_fp4_launch_cuda(
    int operand_kind, const unsigned char* a, const int* a_scale,
    const unsigned char* b, const int* b_scale, const int* masked_m,
    unsigned short* out, int groups, int n, int k, int masked_cap, int num_sms,
    cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (a == nullptr || a_scale == nullptr || b == nullptr ||
      b_scale == nullptr || masked_m == nullptr || out == nullptr ||
      masked_cap <= 0 || masked_cap % 128 != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  switch (num_sms) {
    case kB200Sms:
      return launch_masked_fp8_fp4_sm100_dispatch<kB200Sms>(
          operand_kind, a, a_scale, b, b_scale, masked_m, out, groups, n, k,
          masked_cap, stream);
    case kGb300Sms:
      return launch_masked_fp8_fp4_sm100_dispatch<kGb300Sms>(
          operand_kind, a, a_scale, b, b_scale, masked_m, out, groups, n, k,
          masked_cap, stream);
    default:
      return CUDA_ERROR_NOT_SUPPORTED;
  }
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

}  // extern "C"

#else  // !K3_DEEPGEMM_FP8_FP4_SM100F

extern "C" {

CUresult k3_deepgemm_sm100_masked_grouped_fp8_fp4_launch_cuda(
    int /*operand_kind*/, const unsigned char* /*a*/, const int* /*a_scale*/,
    const unsigned char* /*b*/, const int* /*b_scale*/,
    const int* /*masked_m*/, unsigned short* /*out*/, int /*groups*/,
    int /*n*/, int /*k*/, int /*masked_cap*/, int /*num_sms*/,
    cudaStream_t /*stream*/) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

}  // extern "C"

#endif  // K3_DEEPGEMM_FP8_FP4_SM100F
