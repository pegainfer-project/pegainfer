// Qwen3.8-Flash-Next routed experts as one grouped GEMM per leg.
//
// Prefill used to issue one cuBLAS GEMM per expert per leg — 1024 launches and
// a device-to-host copy of `indptr` at 2048 tokens — because each expert's
// token slots are a different row range. CUTLASS's grouped GEMM takes the same
// work as one launch with a device-side array of per-group problems, pointers
// and leading dimensions, so the host never reads `indptr` and the launch count
// drops from `2 * experts` to two, plus one argument builder.
//
// Local replica, not the vendored entry: FlashInfer's
// `group_gemm::CutlassSegmentGEMMRun` (include/flashinfer/gemm/group_gemm.cuh)
// hardcodes `threadblock_count = 4`, which would run 512 groups on four CTAs.
// The kernel below is that header's GemmKernel instantiation — Sm80 tag, 32-wide
// K tile, f32 accumulator, `GemmBatchedIdentityThreadblockSwizzle`, the
// 8-element LinearCombination epilogue — with the threadblock count taken from
// `GemmGrouped::sufficient()`, and it fixes the operand roles to this block's
// layout instead of templating on them:
//
//   A [M, K] row-major   the gathered activation rows, one group per expert
//   B [N, K] column-major the expert's weight slab (the GEMM's logical B is
//                        [K, N], so the bank is column-major)
//   D [M, N] row-major   the expert output rows, in the slot order `indptr`
//                        already put them in
//
// so `D` is exactly the `[pairs, N]` buffer the per-expert loop wrote.
//
// The tile is 64 rows by 128 columns. At 2048 tokens a prefill group averages
// 40 rows (20480 pairs over 512 experts), so a 64-row tile covers it in one
// tile per expert — the expert's weight slab is read once — while a 128-row
// tile would spend half its multiply-accumulates on padding rows. A sweep over
// tile/stage configs at 200 tokens (narrow geometry, both legs) put 64x128x4 at
// 197/156 us, 64x128x3 at 194/158 (level), 64x128x2 at 230/138 (mixed) and
// 128x128x4 at 388/310 (2x worse), so only 64x128x4 ships; the CTA count sweep
// (13/26/52/104/208/416) showed the occupancy-based count (52 on 26 SMs) best.
//
// Numerics: CUTLASS accumulates each output element over K in f32 like cuBLAS,
// but the split/order of the accumulation and the tile's epilogue order differ,
// so the grouped result is a bound against the per-expert path, not a bitwise
// match. The gate measures the bound at real geometry.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

namespace {

using Element = cutlass::bfloat16_t;
using Problem = cutlass::gemm::GemmCoord;
/// CUTLASS's `LayoutA::Stride::LongIndex` for both row-major operands here.
using LongIndex = int64_t;

/// One grouped-GEMM kernel: Sm80 tensor-op mma with an f32 accumulator.
using GroupedKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8, Element,
    cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8, Element,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 4>::GemmKernel;

/// The per-group argument arrays, in the order the layout below carves them.
struct GroupArgs {
  Problem* problems;
  Element** a;
  Element** b;
  Element** c;
  Element** d;
  LongIndex* lda;
  LongIndex* ldb;
  LongIndex* ldc;
  LongIndex* ldd;
};

constexpr size_t kArgAlign = 16;

size_t align_up(size_t bytes) { return (bytes + kArgAlign - 1) & ~(kArgAlign - 1); }

/// Bytes one grouped launch needs for its per-group arrays. `experts` groups of
/// a 12 B problem, four pointer arrays and four `int64_t` leading dimensions.
size_t group_args_bytes(int experts) {
  return align_up(sizeof(Problem) * experts) + 4 * align_up(sizeof(void*) * experts) +
         4 * align_up(sizeof(LongIndex) * experts);
}

GroupArgs carve_args(void* base, int experts) {
  unsigned char* at = static_cast<unsigned char*>(base);
  GroupArgs args;
  args.problems = reinterpret_cast<Problem*>(at);
  at += align_up(sizeof(Problem) * experts);
  auto* ptr_rows = reinterpret_cast<void**>(at);
  at += align_up(sizeof(void*) * experts);
  auto* ptr_rows_next = reinterpret_cast<void**>(at);
  at += align_up(sizeof(void*) * experts);
  auto* ptr_rows_c = reinterpret_cast<void**>(at);
  at += align_up(sizeof(void*) * experts);
  auto* ptr_rows_d = reinterpret_cast<void**>(at);
  at += align_up(sizeof(void*) * experts);
  auto* ld_a = reinterpret_cast<LongIndex*>(at);
  at += align_up(sizeof(LongIndex) * experts);
  auto* ld_b = reinterpret_cast<LongIndex*>(at);
  at += align_up(sizeof(LongIndex) * experts);
  auto* ld_c = reinterpret_cast<LongIndex*>(at);
  at += align_up(sizeof(LongIndex) * experts);
  auto* ld_d = reinterpret_cast<LongIndex*>(at);
  args.a = reinterpret_cast<Element**>(ptr_rows);
  args.b = reinterpret_cast<Element**>(ptr_rows_next);
  args.c = reinterpret_cast<Element**>(ptr_rows_c);
  args.d = reinterpret_cast<Element**>(ptr_rows_d);
  args.lda = ld_a;
  args.ldb = ld_b;
  args.ldc = ld_c;
  args.ldd = ld_d;
  return args;
}

// The device-side argument builder: group `e` is the row range
// `indptr[e]..indptr[e + 1)` of `x` against weight slab `e`. A zero-row group is
// written as a legal (0, N, K) problem whose pointers point at its own empty
// range; the grouped scheduler gives it no tiles.
__global__ void qwen4_moe_grouped_args_kernel(const int* __restrict__ indptr,
                                              Element* __restrict__ x, Element* __restrict__ w,
                                              Element* __restrict__ y, GroupArgs args, int experts,
                                              int n, int k) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= experts) return;
  const int first = indptr[e];
  const int rows = indptr[e + 1] - first;
  args.problems[e] = Problem(rows, n, k);
  args.a[e] = x + static_cast<int64_t>(first) * k;
  args.b[e] = w + static_cast<int64_t>(e) * n * k;
  // C is never read (beta 0) but CUTLASS keeps the pointer array non-null.
  args.c[e] = y + static_cast<int64_t>(first) * n;
  args.d[e] = args.c[e];
  args.lda[e] = k;
  args.ldb[e] = k;
  args.ldc[e] = n;
  args.ldd[e] = n;
}

cudaError_t run_grouped(GroupArgs const& args, int groups, int threadblocks, cudaStream_t stream) {
  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GroupedKernel>;
  using EpilogueOp = typename GroupedKernel::Epilogue::OutputOp;
  typename EpilogueOp::Params epilogue(1.0f, 0.0f);
  typename GemmGrouped::Arguments arguments(args.problems, groups, threadblocks, epilogue, args.a,
                                            args.b, args.c, args.d, args.lda, args.ldb, args.ldc,
                                            args.ldd);
  GemmGrouped gemm;
  const cutlass::Status init = gemm.initialize(arguments, /*workspace=*/nullptr, stream);
  if (init != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
  const cutlass::Status ran = gemm.run(stream);
  if (ran != cutlass::Status::kSuccess) return cudaErrorLaunchFailure;
  return cudaGetLastError();
}

/// The CTA count CUTLASS would schedule this kernel with: the device's SM count
/// times the kernel's own max active blocks per SM. Queried by the wrapper once
/// per grouped block, not on the host per call.
int sufficient_threadblocks() {
  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GroupedKernel>;
  return GemmGrouped::sufficient(/*problem_sizes=*/nullptr, /*problem_count=*/0);
}

}  // namespace

extern "C" {

/// Bytes [`qwen4_moe_grouped_gemm_cuda`]'s `args` buffer needs for `experts`
/// groups.
size_t qwen4_moe_grouped_args_bytes(int experts) {
  if (experts <= 0) return 0;
  return group_args_bytes(experts);
}

/// The CTA count the shipped kernel config schedules with on this device.
int qwen4_moe_grouped_threadblocks() { return sufficient_threadblocks(); }

/// `out[first_e..] = x[first_e..] @ w[e]^T` for every expert `e`, where
/// `first_e = indptr[e]` and `x` is the gathered `[pairs, k]` activation,
/// `w` the `[experts * n, k]` expert bank and `out` the `[pairs, n]` slot
/// output. `args` is `qwen4_moe_grouped_args_bytes(experts)` bytes of scratch;
/// the per-group arrays are built on the device, so `indptr` never reaches the
/// host.
CUresult qwen4_moe_grouped_gemm_cuda(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                     __nv_bfloat16* out, const int* indptr, void* args,
                                     size_t args_bytes, int experts, int n, int k,
                                     int threadblocks, cudaStream_t stream) {
  if (x == nullptr || w == nullptr || out == nullptr || indptr == nullptr || args == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (experts <= 0 || n <= 0 || k <= 0 || threadblocks <= 0 || threadblocks > (1 << 20)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // The 8-element epilogue and 8-wide operand granularity: both operand row
  // strides and the output row stride must be whole vectors.
  if (n % 8 != 0 || k % 8 != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (args_bytes < group_args_bytes(experts)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const GroupArgs arrays = carve_args(args, experts);
  {
    const int threads = 256;
    const int blocks = (experts + threads - 1) / threads;
    qwen4_moe_grouped_args_kernel<<<blocks, threads, 0, stream>>>(
        indptr, reinterpret_cast<Element*>(const_cast<__nv_bfloat16*>(x)),
        reinterpret_cast<Element*>(const_cast<__nv_bfloat16*>(w)),
        reinterpret_cast<Element*>(out), arrays, experts, n, k);
    const CUresult status = consume_last_cuda_error();
    if (status != CUDA_SUCCESS) return status;
  }
  const cudaError_t status = run_grouped(arrays, experts, threadblocks, stream);
  return map_cuda_error(status == cudaSuccess ? cudaGetLastError() : status);
}

}  // extern "C"
