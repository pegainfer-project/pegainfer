// GLM5.2 EP4/Blackwell routed-expert GEMM: DeepGEMM SM100 MGroupedMasked fp8
// blockscale (tcgen05), AOT-instantiated from the vendored device headers (no
// JIT, no torch). This replaces the in-house tile/mma chain with an AOT
// instantiation of the vendored DeepGEMM implementation.
//
// Masked layout contract (mirrors DeepGEMM's sm100 masked host wrapper
// sm100_m_grouped_fp8_fp4_gemm_masked_1d1d; SF is the Blackwell packed-UE8M0
// contract — arbitrary f32 scales are REJECTED device-side):
//   activation  [groups, masked_cap, k]                fp8 e4m3
//   act scales  [groups, ceil(k/512), masked_cap]      i32, MN-major; each
//                 i32 packs 4 K-blocks' UE8M0 exponents:
//                 u32 = (f0>>23) | (f1>>15) | (f2>>7) | (f3<<1)
//   weight      [groups, n, k]                         fp8 (bank as-is)
//   wt scales   [groups, ceil(k/512), n]               i32, same packing
//                 (loader-time UE8M0 requant+pack)
//   masked_m    i32[groups]                            real rows per expert
//   out         [groups, masked_cap, n]                bf16
//
// Instantiation config mirrors SM100ArchSpec's single masked-layout candidate
// (heuristics/sm100.hpp:27-43): swap_ab=true, BLOCK_M/N/K=128/128/128,
// cluster (1,2) → multicast 2 on A, LOAD_BLOCK_M=64 / LOAD_BLOCK_N=128,
// STORE_BLOCK_M=16 / STORE_BLOCK_N=128, 128B swizzles, num_stages=8 from the
// 232448B smem budget (mirror get_pipeline_config below), 128+128 threads.
// The persistent scheduler bakes the SM count into its template, so B200
// (148) and GB300 (152) get separate AOT instantiations selected explicitly
// at launch.
//
// build.rs compiles the GEMM section for sm_100f ONLY when a sm_100-family
// target exists (tcgen05 needs the family arch; runs on sm_103). Otherwise
// the GEMM entry compiles as a NOT_SUPPORTED stub. The metadata/remap
// kernels are plain CUDA and compile for every target.
//
// `groups` is a runtime dispatch over the 6 supported local-expert counts
// serves (EP4/8/16/32/64 → 64/32/16/8/4/2 local experts); num_groups is
// baked into the kernel template, so each width gets its own instantiation.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int kExpertAlignment = 64;
constexpr int kMetadataThreads = 32;

constexpr int kKindW13 = 1;
constexpr int kKindW2 = 2;
constexpr int kW13N = 4096;
constexpr int kW13K = 6144;
constexpr int kW2N = 6144;
constexpr int kW2K = 2048;

__device__ __forceinline__ int align_up_int(int value, int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

__device__ __forceinline__ int clamp_nonnegative(int value) {
  return value < 0 ? 0 : value;
}

// psum → aligned segment starts / per-expert real rows / aligned-row →
// masked-slot map. Same contract as the SM90 TU's metadata kernel, with
// groups/cap as runtime params.
__global__ void deepgemm_sm100_grouped_fp8_metadata_kernel(
    const int* __restrict__ psum_expert, int64_t* __restrict__ expert_offsets,
    int* __restrict__ masked_m, int* __restrict__ row_map, int groups,
    int m_capacity, int expert_alignment, int masked_cap) {
  int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= groups) {
    return;
  }

  int previous_end =
      expert == 0 ? 0 : clamp_nonnegative(psum_expert[expert - 1]);
  int end = clamp_nonnegative(psum_expert[expert]);
  int start = expert == 0 ? 0 : align_up_int(previous_end, expert_alignment);
  int count = end - start;

  // A segment past m_capacity means the ranks disagreed about the token
  // count; a segment longer than the masked capacity would alias the next
  // expert's rows. Crash instead of silently corrupting.
  if (start > m_capacity || align_up_int(end, expert_alignment) > m_capacity ||
      count < 0 || count > masked_cap) {
    __trap();
  }

  expert_offsets[expert] = static_cast<int64_t>(start);
  if (expert == groups - 1) {
    expert_offsets[groups] =
        static_cast<int64_t>(align_up_int(end, expert_alignment));
  }
  masked_m[expert] = count;

  for (int r = 0; r < count; ++r) {
    row_map[start + r] = expert * masked_cap + r;
  }
  int gap_end = align_up_int(end, expert_alignment);
  for (int r = end; r < gap_end; ++r) {
    row_map[r] = -1;
  }
}

// Masked GEMM output → the DeepEP aligned-segment slots decode_combine
// addresses, applying the per-row router weight. Capacity-shaped grid
// (graph-stable); blocks past a segment's real row count retire immediately.
__global__ void deepgemm_sm100_masked_out_to_aligned_kernel(
    const __nv_bfloat16* __restrict__ masked_out,
    const int* __restrict__ masked_m, const int64_t* __restrict__ offsets,
    const float* __restrict__ row_weights,
    __nv_bfloat16* __restrict__ aligned_out, int masked_cap, int aligned_rows,
    int n) {
  const int g = blockIdx.x;
  const int r = blockIdx.y;
  if (r >= masked_m[g]) {
    return;
  }
  const __nv_bfloat16* src = masked_out + ((size_t)g * masked_cap + r) * n;
  const int64_t aligned_row = offsets[g] + r;
  if (aligned_row < 0 || aligned_row >= aligned_rows) {
    __trap();
  }
  __nv_bfloat16* dst = aligned_out + (size_t)aligned_row * n;
  const float weight = __ldg(row_weights + aligned_row);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    dst[i] = __float2bfloat16_rn(__bfloat162float(src[i]) * weight);
  }
}

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

}  // namespace

extern "C" {

CUresult glm52_deepgemm_sm100_grouped_fp8_metadata_cuda(
    const int* psum_expert, int64_t* expert_offsets, int* masked_m,
    int* row_map, int groups, int m_capacity, int expert_alignment,
    int masked_cap, cudaStream_t stream) {
  if (psum_expert == nullptr || expert_offsets == nullptr ||
      masked_m == nullptr || row_map == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (groups <= 0 || m_capacity <= 0 || expert_alignment != kExpertAlignment ||
      masked_cap <= 0 || masked_cap % 128 != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  int blocks = (groups + kMetadataThreads - 1) / kMetadataThreads;
  deepgemm_sm100_grouped_fp8_metadata_kernel<<<blocks, kMetadataThreads, 0,
                                               stream>>>(
      psum_expert, expert_offsets, masked_m, row_map, groups, m_capacity,
      expert_alignment, masked_cap);
  return consume_last_cuda_error();
}

CUresult glm52_deepgemm_sm100_masked_out_to_aligned_cuda(
    const __nv_bfloat16* masked_out, const int* masked_m,
    const int64_t* expert_offsets, const float* row_weights,
    __nv_bfloat16* aligned_out, int groups, int masked_cap, int aligned_rows,
    int n, cudaStream_t stream) {
  if (masked_out == nullptr || masked_m == nullptr ||
      expert_offsets == nullptr || row_weights == nullptr ||
      aligned_out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (groups <= 0 || masked_cap <= 0 || masked_cap % 128 != 0 ||
      aligned_rows <= 0 || n <= 0 || n % 4 != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  deepgemm_sm100_masked_out_to_aligned_kernel<<<dim3(groups, masked_cap), 256,
                                                0, stream>>>(
      masked_out, masked_m, expert_offsets, row_weights, aligned_out,
      masked_cap, aligned_rows, n);
  return consume_last_cuda_error();
}

}  // extern "C"

#ifdef GLM52_DEEPGEMM_GROUPED_SM100F

#include <jit_kernels/impls/runtime_utils.hpp>

#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>

namespace {

constexpr int kSmemCapacity = 232448;
constexpr int kB200Sms = 148;
constexpr int kGb300Sms = 152;

// One (n, k, groups) instantiation of the masked GEMM. Config mirrors
// SM100ArchSpec's masked candidate (see header comment): swap_ab, cluster
// (1,2) → multicast 2 on A, stages 8, 128+128 threads. SHAPE_M stays runtime
// (compiled_dims='nk'); m arrives as the runtime masked-cap launch arg.
template <uint32_t N, uint32_t K, uint32_t GROUPS, uint32_t NUM_SMS>
struct MaskedGemmSm100 {
  static constexpr uint32_t kShapeN = N;
  static constexpr uint32_t kShapeK = K;
  static constexpr uint32_t kGroups = GROUPS;
  static constexpr uint32_t kNumSms = NUM_SMS;
  static constexpr auto kKernel = &deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      /*gran_k A/B=*/128, 128, /*k_alignment=*/128,
      /*SHAPE_M=*/0, N, K,
      /*BLOCK_M=*/128, /*BLOCK_N=*/128, /*BLOCK_K=*/128, GROUPS,
      /*swizzle A/B/CD=*/128, 128, 128,
      /*stages=*/8,
      /*non-epilogue threads=*/128, /*epilogue threads=*/128,
      /*multicast=*/2, /*multicast on A=*/true, NUM_SMS,
      /*swap_ab=*/true, /*ensure_zero_padding=*/false,
      deep_gemm::GemmType::MGroupedMasked, /*with_accumulation=*/false,
      cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,
      deep_gemm::epilogue::transform::EpilogueIdentity>;

  // Mirrors SM100ArchSpec::get_pipeline_config for this config:
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
    MaskedGemmSm100<kW13N, kW13K, 64, kGb300Sms>::smem_size() <=
    kSmemCapacity);

template <typename Gemm>
CUresult launch_masked_sm100(const unsigned char* a, const int* a_scale,
                             const unsigned char* b, const int* b_scale,
                             const int* masked_m, unsigned short* out,
                             int masked_cap, cudaStream_t stream) {
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
  // A: K-major [m*groups, k] folded 2D, smem box [block_k, load_block_m].
  const auto tma_a = deep_gemm::make_tma_2d_desc_raw(
      const_cast<unsigned char*>(a), 1, deep_gemm::DgDtype::Float8_e4m3, k,
      masked_cap * groups, 128, 64, k, 128);
  // B: K-major [n*groups, k], smem box [block_k, load_block_n].
  const auto tma_b = deep_gemm::make_tma_2d_desc_raw(
      const_cast<unsigned char*>(b), 1, deep_gemm::DgDtype::Float8_e4m3, k,
      n * groups, 128, 128, k, 128);
  // C/D: [m*groups, n], store box [store_block_n, store_block_m]; the raw
  // builder replaces smem_inner with swizzle/elem_size (=64 elems bf16).
  const auto tma_cd = deep_gemm::make_tma_2d_desc_raw(
      out, 2, deep_gemm::DgDtype::BFloat16, n, masked_cap * groups, 128, 16, n,
      128);
  // SF: MN-major packed UE8M0 i32. Inner dim = mn (contiguous, stride 1),
  // outer = ceil(k/512)*groups packed columns, outer stride = mn elements.
  const auto tma_sfa = deep_gemm::make_tma_2d_desc_raw(
      const_cast<int*>(a_scale), 4, deep_gemm::DgDtype::Int, masked_cap,
      (k / 512) * groups, 128, 1, masked_cap, 0);
  const auto tma_sfb = deep_gemm::make_tma_2d_desc_raw(
      const_cast<int*>(b_scale), 4, deep_gemm::DgDtype::Int, n,
      (k / 512) * groups, 128, 1, n, 0);

  // Cluster (2,1,1) (the A-side TMA multicast pair) + PDL, per DeepGEMM's
  // own launch config. The attrs array is per-call stack storage.
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
      const_cast<CUtensorMap*>(&tma_a),  const_cast<CUtensorMap*>(&tma_b),
      const_cast<CUtensorMap*>(&tma_sfa), const_cast<CUtensorMap*>(&tma_sfb),
      const_cast<CUtensorMap*>(&tma_cd),
  };
  return map_cuda_error(cudaLaunchKernelExC(&config, func, args));
}

template <uint32_t GROUPS, uint32_t NUM_SMS>
CUresult launch_masked_sm100_groups(int operand_kind, const unsigned char* a,
                                    const int* a_scale, const unsigned char* b,
                                    const int* b_scale, const int* masked_m,
                                    unsigned short* out, int n, int k,
                                    int masked_cap, cudaStream_t stream) {
  if (operand_kind == kKindW13 && n == kW13N && k == kW13K) {
    return launch_masked_sm100<
        MaskedGemmSm100<kW13N, kW13K, GROUPS, NUM_SMS>>(
        a, a_scale, b, b_scale, masked_m, out, masked_cap, stream);
  }
  if (operand_kind == kKindW2 && n == kW2N && k == kW2K) {
    return launch_masked_sm100<
        MaskedGemmSm100<kW2N, kW2K, GROUPS, NUM_SMS>>(
        a, a_scale, b, b_scale, masked_m, out, masked_cap, stream);
  }
  return CUDA_ERROR_INVALID_VALUE;
}

template <uint32_t NUM_SMS>
CUresult launch_masked_sm100_dispatch(
    int operand_kind, const unsigned char* a, const int* a_scale,
    const unsigned char* b, const int* b_scale, const int* masked_m,
    unsigned short* out, int groups, int n, int k, int masked_cap,
    cudaStream_t stream) {
  switch (groups) {
    case 64:
      return launch_masked_sm100_groups<64, NUM_SMS>(
          operand_kind, a, a_scale, b, b_scale, masked_m, out, n, k,
          masked_cap, stream);
    case 32:
      return launch_masked_sm100_groups<32, NUM_SMS>(
          operand_kind, a, a_scale, b, b_scale, masked_m, out, n, k,
          masked_cap, stream);
    case 16:
      return launch_masked_sm100_groups<16, NUM_SMS>(
          operand_kind, a, a_scale, b, b_scale, masked_m, out, n, k,
          masked_cap, stream);
    case 8:
      return launch_masked_sm100_groups<8, NUM_SMS>(
          operand_kind, a, a_scale, b, b_scale, masked_m, out, n, k,
          masked_cap, stream);
    case 4:
      return launch_masked_sm100_groups<4, NUM_SMS>(
          operand_kind, a, a_scale, b, b_scale, masked_m, out, n, k,
          masked_cap, stream);
    case 2:
      return launch_masked_sm100_groups<2, NUM_SMS>(
          operand_kind, a, a_scale, b, b_scale, masked_m, out, n, k,
          masked_cap, stream);
    default:
      return CUDA_ERROR_INVALID_VALUE;
  }
}

}  // namespace

extern "C" {

CUresult glm52_deepgemm_sm100_masked_grouped_fp8_launch_cuda(
    int operand_kind, const unsigned char* a, const int* a_scale,
    const unsigned char* b, const int* b_scale, const int* masked_m,
    unsigned short* out, int groups, int n, int k, int masked_cap, int num_sms,
    cudaStream_t stream) {
  if (a == nullptr || a_scale == nullptr || b == nullptr ||
      b_scale == nullptr || masked_m == nullptr || out == nullptr ||
      masked_cap <= 0 || masked_cap % 128 != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  switch (num_sms) {
    case kB200Sms:
      return launch_masked_sm100_dispatch<kB200Sms>(
          operand_kind, a, a_scale, b, b_scale, masked_m, out, groups, n, k,
          masked_cap, stream);
    case kGb300Sms:
      return launch_masked_sm100_dispatch<kGb300Sms>(
          operand_kind, a, a_scale, b, b_scale, masked_m, out, groups, n, k,
          masked_cap, stream);
    default:
      return CUDA_ERROR_NOT_SUPPORTED;
  }
}

}  // extern "C"

#else  // !GLM52_DEEPGEMM_GROUPED_SM100F

extern "C" {

CUresult glm52_deepgemm_sm100_masked_grouped_fp8_launch_cuda(
    int /*operand_kind*/, const unsigned char* /*a*/, const int* /*a_scale*/,
    const unsigned char* /*b*/, const int* /*b_scale*/,
    const int* /*masked_m*/, unsigned short* /*out*/, int /*groups*/,
    int /*n*/, int /*k*/, int /*masked_cap*/, int /*num_sms*/,
    cudaStream_t /*stream*/) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

}  // extern "C"

#endif  // GLM52_DEEPGEMM_GROUPED_SM100F
