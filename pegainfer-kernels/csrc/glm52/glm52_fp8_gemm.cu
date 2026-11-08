#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#ifdef GLM52_FP8_GEMM_SM100A
#include <flashinfer/gemm/gemm_groupwise_sm100.cuh>
#include <flashinfer/gemm/group_gemm_fp8_groupwise_sm100.cuh>
#endif

namespace {

// Grouped-GEMM workspace split. FlashInfer's grouped entry takes two buffers:
// an int_buffer it carves with AlignedAllocator into eleven per-group argument
// arrays (problem_sizes 12 B, five pointer arrays 8 B, three cute packed
// strides <= 24 B, two SF layouts <= 128 B per group, each array 16-aligned
// — well under 512 B/group), and a float_buffer for the CUTLASS kernel
// workspace (Gemm::get_workspace_size: grouped scheduler state + per-SM
// tensormaps, a few hundred KiB). 1 MiB on the int side covers 2048 groups
// with headroom (routed experts per rank are <= 256); everything past it
// feeds CUTLASS. The AlignedAllocator throws on overflow, which the FFI
// guard converts to an error instead of corrupting the split.
constexpr size_t kGroupedIntBufferBytes = size_t{1} << 20;
constexpr int kGroupedMaxGroups = 2048;

}  // namespace

#ifdef GLM52_FP8_GEMM_SM100A
namespace {

// Local replica of flashinfer::group_gemm::CutlassFP8GroupwiseScaledGroup-
// GEMMSM100<1, 128, 128, /*ScaleMajorK=*/true, /*MmaSM=*/1, e4m3, bf16>
// (group_gemm_fp8_groupwise_sm100.cuh) with the PDL pair disabled. The
// header's compute_sm100_cutlass_group_gemm_args kernel executes
//   asm volatile("griddepcontrol.wait;");
//   asm volatile("griddepcontrol.launch_dependents;");
// at the TOP of the kernel, BEFORE it writes the per-group problem_sizes/
// ptr/stride/layout arrays, and the header tail then launches the CUTLASS
// grouped kernel with launch_with_pdl=true — so the dependent grid's
// griddepcontrol.wait can pass while the args arrays are still being
// written. That race produced intermittent CUDA_ERROR_ILLEGAL_ADDRESS under
// load on GB300 (worth an upstream FlashInfer report). Here the args kernel
// gets a plain launch (its griddepcontrol asm is a no-op without the
// programmatic-stream-serialization attribute) and gemm.run passes
// launch_with_pdl=false, restoring the launch-order dependency through the
// stream. Types below are copied verbatim from the header so the
// compute_sm100_cutlass_group_gemm_args instantiation matches exactly; drop
// this copy once a flashinfer bump fixes the ordering upstream.
cudaError_t grouped_gemm_sm100_no_pdl(void* int_buffer,
                                      size_t int_buffer_size_in_bytes,
                                      void* float_buffer,
                                      size_t float_buffer_size_in_bytes,
                                      cutlass::float_e4m3_t* A,
                                      cutlass::float_e4m3_t* B, float* SFA,
                                      float* SFB, cutlass::bfloat16_t* D,
                                      int* m_indptr, int max_m, int n, int k,
                                      int num_groups, cudaStream_t stream) {
  using namespace cute;
  constexpr int ScaleGranularityM = 1;
  constexpr int ScaleGranularityN = 128;
  constexpr int ScaleGranularityK = 128;
  constexpr bool ScaleMajorK = true;
  constexpr int MmaSM = 1;

  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = cutlass::bfloat16_t;
  using LayoutD = cutlass::layout::RowMajor;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementC = void;
  using LayoutC = void;
  constexpr int AlignmentC = 0;

  using ElementAccumulator = float;
  using ElementCompute = float;

  using MmaTileShape_MNK = Shape<cute::Int<MmaSM * 128>, _128, _128>;
  using ClusterShape_MNK = Shape<cute::Int<MmaSM>, _1, _1>;

  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      ScaleGranularityM, ScaleGranularityN, ScaleGranularityK, UMMA::Major::K,
      UMMA::Major::K>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
          MmaTileShape_MNK, ClusterShape_MNK,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementCompute, ElementC, LayoutC*, AlignmentC, ElementD, LayoutD*,
          AlignmentD, EpilogueSchedule>::CollectiveOp;

  using MainloopSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA,
          cute::tuple<LayoutA*, LayoutSFA*>, AlignmentA, ElementB,
          cute::tuple<LayoutB*, LayoutSFB*>, AlignmentB, ElementAccumulator,
          MmaTileShape_MNK, ClusterShape_MNK,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  static_assert(
      cute::is_same_v<
          typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA,
          LayoutSFA>);
  static_assert(
      cute::is_same_v<
          typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB,
          LayoutSFB>);

  flashinfer::AlignedAllocator allocator(int_buffer, int_buffer_size_in_bytes);

  auto problem_sizes =
      allocator.aligned_alloc<typename ProblemShape::UnderlyingProblemShape>(
          num_groups * sizeof(typename ProblemShape::UnderlyingProblemShape),
          16, "sm100_groupwise_group_gemm_problem_sizes");
  auto A_ptr = allocator.aligned_alloc<const typename Gemm::ElementA*>(
      num_groups * sizeof(const typename Gemm::ElementA*), 16,
      "sm100_groupwise_group_gemm_A_ptr");
  auto B_ptr = allocator.aligned_alloc<const typename Gemm::ElementB*>(
      num_groups * sizeof(const typename Gemm::ElementB*), 16,
      "sm100_groupwise_group_gemm_B_ptr");
  auto D_ptr =
      allocator.aligned_alloc<typename Gemm::EpilogueOutputOp::ElementOutput*>(
          num_groups * sizeof(typename Gemm::EpilogueOutputOp::ElementOutput*),
          16, "sm100_groupwise_group_gemm_D_ptr");
  auto SFA_ptr = allocator.aligned_alloc<const ElementAccumulator*>(
      num_groups * sizeof(const ElementAccumulator*), 16,
      "sm100_groupwise_group_gemm_SFA_ptr");
  auto SFB_ptr = allocator.aligned_alloc<const ElementAccumulator*>(
      num_groups * sizeof(const ElementAccumulator*), 16,
      "sm100_groupwise_group_gemm_SFB_ptr");

  auto stride_A = allocator.aligned_alloc<StrideA>(
      num_groups * sizeof(StrideA), 16, "sm100_groupwise_group_gemm_stride_A");
  auto stride_B = allocator.aligned_alloc<StrideB>(
      num_groups * sizeof(StrideB), 16, "sm100_groupwise_group_gemm_stride_B");
  auto stride_D = allocator.aligned_alloc<StrideD>(
      num_groups * sizeof(StrideD), 16, "sm100_groupwise_group_gemm_stride_D");
  auto layout_SFA = allocator.aligned_alloc<LayoutSFA>(
      num_groups * sizeof(LayoutSFA), 16,
      "sm100_groupwise_group_gemm_layout_SFA");
  auto layout_SFB = allocator.aligned_alloc<LayoutSFB>(
      num_groups * sizeof(LayoutSFB), 16,
      "sm100_groupwise_group_gemm_layout_SFB");

  int num_threads = std::min(num_groups, 1024);
  int num_blocks = (num_groups + num_threads - 1) / num_threads;

  // PLAIN launch — deliberately no cudaLaunchAttributeProgrammaticStream-
  // Serialization: without the attribute the kernel's griddepcontrol asm is
  // a no-op and the CUTLASS grouped kernel below (launch_with_pdl=false)
  // only starts after these writes retire in stream order.
  auto prepare_args_kernel =
      flashinfer::group_gemm::compute_sm100_cutlass_group_gemm_args<
          ScaleConfig, ElementA, float, ElementD,
          typename ProblemShape::UnderlyingProblemShape, StrideA, StrideB,
          StrideD, LayoutSFA, LayoutSFB, ScaleMajorK>;
  prepare_args_kernel<<<num_blocks, num_threads, 0, stream>>>(
      A, B, SFA, SFB, D, m_indptr, max_m, n, k, num_groups, ScaleGranularityM,
      ScaleGranularityN, ScaleGranularityK, problem_sizes, A_ptr, B_ptr,
      SFA_ptr, SFB_ptr, D_ptr, stride_A, stride_B, stride_D, layout_SFA,
      layout_SFB);
  FLASHINFER_CUDA_CALL(cudaGetLastError());

  thread_local int const sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = sm_count;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_groups, problem_sizes, /*problem_sizes_host=*/nullptr},
      {
          A_ptr,
          stride_A,
          B_ptr,
          stride_B,
          SFA_ptr,
          layout_SFA,
          SFB_ptr,
          layout_SFB,
      },
      {
          {},       // epilogue.thread
          nullptr,  // C_ptr
          nullptr,  // stride_C
          D_ptr,
          stride_D,
      },
      hw_info};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  flashinfer::AlignedAllocator float_allocator(float_buffer,
                                               float_buffer_size_in_bytes);
  auto workspace_ptr = float_allocator.aligned_alloc<void>(
      workspace_size, 16, "sm100_groupwise_group_gemm_float_workspace");

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr));
  CUTLASS_CHECK(gemm.run(stream, /*cuda_adapter=*/nullptr,
                         /*launch_with_pdl=*/false));
  return cudaSuccess;
}

}  // namespace
#endif  // GLM52_FP8_GEMM_SM100A

extern "C" CUresult glm52_fp8_groupwise_gemm_sm100_cuda(
    const unsigned char* activation, const float* activation_scale,
    const unsigned char* weight, const float* weight_scale,
    __nv_bfloat16* output, void* workspace, size_t workspace_bytes, int m,
    int n, int k, CUstream stream) {
  if (!activation || !activation_scale || !weight || !weight_scale || !output ||
      !workspace || workspace_bytes == 0 || m <= 0 || n <= 0 || k <= 0 ||
      (m % 4) != 0 || (k % 128) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  PEGAINFER_FFI_GUARD_BEGIN
#ifdef GLM52_FP8_GEMM_SM100A
  auto status =
      flashinfer::gemm::CutlassGroupwiseScaledGEMMSM100<
          1, 128, 128, true, 2, cutlass::float_e4m3_t,
          cutlass::bfloat16_t>(
          workspace, workspace_bytes,
          reinterpret_cast<cutlass::float_e4m3_t*>(
              const_cast<unsigned char*>(activation)),
          reinterpret_cast<cutlass::float_e4m3_t*>(
              const_cast<unsigned char*>(weight)),
          const_cast<float*>(activation_scale),
          const_cast<float*>(weight_scale),
          reinterpret_cast<cutlass::bfloat16_t*>(output), m, n, k, 1,
          reinterpret_cast<cudaStream_t>(stream));
  return map_cuda_error(status);
#else
  return CUDA_ERROR_NOT_SUPPORTED;
#endif
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

// Grouped (per-expert) variant over a banked weight: group g multiplies
// activation rows m_indptr[g]..m_indptr[g+1] (device-side, so total_m cannot
// be host-validated) against weight bank slab g. Per-group m has no
// alignment requirement. k % 128: scale granularity K + 16-element fp8 TMA
// alignment. n % 128 is NOT a CUTLASS requirement (the bf16 TMA output only
// needs n % 8), but the banked weight-scale stride [num_groups, n/128,
// k/128] must be exact or groups would read each other's scales.
extern "C" CUresult glm52_fp8_grouped_gemm_sm100_cuda(
    const unsigned char* activation, const float* activation_scale,
    const unsigned char* weight, const float* weight_scale,
    __nv_bfloat16* output, const int* m_indptr, int max_m, int n, int k,
    int num_groups, void* workspace, size_t workspace_bytes, CUstream stream) {
  if (!activation || !activation_scale || !weight || !weight_scale ||
      !output || !m_indptr || !workspace ||
      workspace_bytes <= kGroupedIntBufferBytes || max_m <= 0 || n <= 0 ||
      k <= 0 || num_groups <= 0 || num_groups > kGroupedMaxGroups ||
      (n % 128) != 0 || (k % 128) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  PEGAINFER_FFI_GUARD_BEGIN
#ifdef GLM52_FP8_GEMM_SM100A
  // grouped_gemm_sm100_no_pdl replaces the header's host wrapper: same body,
  // PDL disabled — see the comment on the helper for the args-kernel
  // early-launch_dependents race it works around.
  void* int_buffer = workspace;
  void* float_buffer =
      static_cast<unsigned char*>(workspace) + kGroupedIntBufferBytes;
  auto status = grouped_gemm_sm100_no_pdl(
      int_buffer, kGroupedIntBufferBytes, float_buffer,
      workspace_bytes - kGroupedIntBufferBytes,
      reinterpret_cast<cutlass::float_e4m3_t*>(
          const_cast<unsigned char*>(activation)),
      reinterpret_cast<cutlass::float_e4m3_t*>(
          const_cast<unsigned char*>(weight)),
      const_cast<float*>(activation_scale), const_cast<float*>(weight_scale),
      reinterpret_cast<cutlass::bfloat16_t*>(output),
      const_cast<int*>(m_indptr), max_m, n, k, num_groups,
      reinterpret_cast<cudaStream_t>(stream));
  return map_cuda_error(status);
#else
  return CUDA_ERROR_NOT_SUPPORTED;
#endif
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}
