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
// The kernel below keeps that header's Sm80 tag, 32-wide K tile, f32
// accumulator, `GemmBatchedIdentityThreadblockSwizzle` and 8-element
// LinearCombination epilogue; only the threadblock shape and stage count differ
// (the tile is the next paragraph), the threadblock count comes from
// `GemmGrouped::sufficient()`, and the operand roles are fixed to this block's
// layout rather than templated:
//
//   A [M, K] row-major   the gathered activation rows, one group per expert
//   B [N, K] column-major the expert's weight slab (the GEMM's logical B is
//                        [K, N], so the bank is column-major)
//   D [M, N] row-major   the expert output rows, in the slot order `indptr`
//                        already put them in
//
// so `D` is exactly the `[pairs, N]` buffer the per-expert loop wrote.
//
// The tile is 64 rows by 128 columns with 4 stages: a prefill group averages
// ~40 rows (20480 pairs over 512 experts at 2048 tokens), so one 64-row tile
// spans M and the expert's weight slab is read once across its N tiles, while
// a 128-row tile would spend most of its rows on padding. This was the best of
// the tile/stage configs swept; the CTA count comes from occupancy. The
// measured sweeps are recorded in `docs/models/qwen4/bring-up.md`.
//
// Numerics: CUTLASS accumulates each output element over K in f32 like cuBLAS,
// but the split/order of the accumulation and the tile's epilogue order differ,
// so the grouped result is a bound against the per-expert path, not a bitwise
// match. The gate measures the bound at real geometry.
//
// The gate_up leg carries a second kernel on the same mainloop whose epilogue
// folds the SiLU-mul into the store. The block ran it as three steps: gate_up
// GEMM -> `[pairs, 1280]` bf16 in global memory, a SiLU-mul kernel that read
// that back, and the down GEMM on its `[pairs, 640]` output. At 2048 tokens the
// intermediate is 52 MB each way, and the fused leg writes only the 26 MB the
// down GEMM needs. The fold needs the expert's gate and up halves adjacent:
// `MoeBlock::gate_up` is stored split (rows `0..640` gate, `640..1280` up), so
// the weights are interleaved to `[g0, u0, g1, u1, ...]` — see
// `qwen4_moe_gate_up_interleave_kernel` below.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include <cutlass/arch/memory.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/epilogue/threadblock/epilogue.h>
#include <cutlass/epilogue/threadblock/predicated_tile_iterator.h>
#include <cutlass/epilogue/warp/fragment_iterator_tensor_op.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/matrix_shape.h>

namespace {

using Element = cutlass::bfloat16_t;
using Problem = cutlass::gemm::GemmCoord;
/// CUTLASS's `LayoutA::Stride::LongIndex` for both row-major operands here.
using LongIndex = int64_t;

using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
constexpr int kStages = 4;
/// The epilogue moves 8-element (16 B) vectors: both operand row strides and
/// the output row stride have to be whole vectors.
constexpr int kElementsPerAccess = 8;

/// One grouped-GEMM kernel: Sm80 tensor-op mma with an f32 accumulator.
using GroupedKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8, Element,
    cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8, Element,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape,
    cutlass::epilogue::thread::LinearCombination<Element, kElementsPerAccess, float, float>,
    ThreadblockSwizzle, kStages>::GemmKernel;

////////////////////////////////////////////////////////////////////////////////
// SiLU-mul epilogue: 8 adjacent gate_up columns -> 4 activated columns
////////////////////////////////////////////////////////////////////////////////

/// Folds `silu(gate) * up` into the gate_up leg's store.
///
/// The accumulator access covers 8 adjacent columns of the *interleaved* expert
/// slab, which are the pairs `(g_j, u_j), (g_j+1, u_j+1), ...`: gate on the even
/// column, up on the odd one. The results go to the first 4 slots of the output
/// access and the upper 4 are dead — `HalvedOutputTileIterator` stores exactly
/// those 4, at output column `c / 2`.
///
/// The rounding is the reference's (`csrc/shared/fused_proj.cu`)
/// `silu_mul_fused_kernel`, which reads the GEMM's gate and up back as bf16
/// before the activation: round gate and up to bf16, SiLU in f32, round the SiLU
/// to bf16, multiply by the bf16 up, round the product. Rounding the accumulator
/// down to bf16 first reproduces that intermediate store; skipping it changes
/// the result by up to a full bf16 quantum on the product.
struct SiluMulEpilogue {
  using ElementOutput = Element;
  using ElementAccumulator = float;
  using ElementCompute = float;

  struct Params {
    CUTLASS_HOST_DEVICE
    Params(float alpha = 1.0f, float beta = 0.0f) {
      (void)alpha;
      (void)beta;
    }
  };

  CUTLASS_HOST_DEVICE
  SiluMulEpilogue(Params const&) {}

  /// `D` is written whole; there is no source to accumulate onto.
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return false; }

  CUTLASS_DEVICE
  cutlass::Array<ElementOutput, kElementsPerAccess> operator()(
      cutlass::Array<ElementAccumulator, kElementsPerAccess> const& accum) const {
    cutlass::Array<ElementOutput, kElementsPerAccess> out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess / 2; ++i) {
      __nv_bfloat16 gate = __float2bfloat16(accum[2 * i]);
      __nv_bfloat16 up = __float2bfloat16(accum[2 * i + 1]);
      float gate_f = __bfloat162float(gate);
      float silu = gate_f / (1.0f + expf(-gate_f));
      float silu_bf16 = __bfloat162float(__float2bfloat16(silu));
      out[i] = Element(__float2bfloat16(silu_bf16 * __bfloat162float(up)));
    }
    // Never stored: the iterator writes a half access.
    CUTLASS_PRAGMA_UNROLL
    for (int i = kElementsPerAccess / 2; i < kElementsPerAccess; ++i) {
      out[i] = Element(0.0f);
    }
    return out;
  }

  /// Unreachable (`is_source_needed()` is false) but not dead code: the
  /// epilogue's source/no-source dispatch is a runtime `if`, so the form that
  /// takes an addend has to compile. `D` is never accumulated onto here, so the
  /// addend is ignored.
  CUTLASS_DEVICE
  cutlass::Array<ElementOutput, kElementsPerAccess> operator()(
      cutlass::Array<ElementAccumulator, kElementsPerAccess> const& accum,
      cutlass::Array<ElementOutput, kElementsPerAccess> const& addend) const {
    (void)addend;
    return (*this)(accum);
  }
};

/// The epilogue's output tile iterator, writing one column per pair.
///
/// The thread map, the fragment layout, the shared-memory staging and the
/// predication are CUTLASS's (`PredicatedTileIterator`'s, for this
/// threadblock/warp/MMA shape). Only the global store changes: a thread's
/// 8-element access lands on 8 adjacent gate_up columns `c..c+7`, i.e. the pairs
/// `c/2..c/2+3`, so the access stores 4 bf16 to output column `c/2` and the
/// output row is `n / 2` wide.
///
/// Two conveniences fall out of the element sizes: 4 bf16 is 8 B and one output
/// column is 2 B, so the byte offset of output column `c/2` is exactly `c` —
/// the column arithmetic is CUTLASS's unchanged — and only the row stride
/// halves. `c` is always a multiple of 8 (the thread map's column offsets are
/// whole accesses), so the halves never straddle a pair.
template <typename ThreadMap_>
class HalvedOutputTileIterator {
 public:
  using ThreadMap = ThreadMap_;
  using Element = cutlass::bfloat16_t;
  using Layout = cutlass::layout::RowMajor;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = cutlass::MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  /// Fragment object, laid out exactly as the shared-load iterator's.
  using Fragment = cutlass::Array<
      Element, ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow *
                   ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster *
                   ThreadMap::kElementsPerAccess>;

  /// What one access stores: the half of the accumulator access that survives.
  using StoreType = cutlass::Array<Element, ThreadMap::kElementsPerAccess / 2>;

  struct Params {
    /// Bytes between output rows.
    LongIndex stride;

    CUTLASS_HOST_DEVICE
    Params() : stride(0) {}

    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : stride(layout.stride(0) * LongIndex(cutlass::sizeof_bits<Element>::value / 8)) {}
  };

  struct Mask {
    static int const kCount = ThreadMap::Iterations::kColumn;
    bool predicates[kCount];

    CUTLASS_HOST_DEVICE
    Mask() { enable(); }

    CUTLASS_HOST_DEVICE
    void clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) predicates[i] = false;
    }

    CUTLASS_HOST_DEVICE
    void enable() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) predicates[i] = true;
    }
  };

 private:
  Params params_;
  uint8_t* store_byte_pointer_;
  Mask mask_;
  Index extent_row_;
  Index extent_column_;
  Index thread_start_row_;
  Index thread_start_column_;
  int state_[3];

 public:
  CUTLASS_DEVICE
  HalvedOutputTileIterator(Params const& params, Element* pointer, TensorCoord extent, int thread_idx,
                           TensorCoord threadblock_offset = TensorCoord()) : params_(params) {
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    extent_row_ = extent.row();
    extent_column_ = extent.column();
    thread_start_row_ = thread_offset.row();
    thread_start_column_ = thread_offset.column();

    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
      mask_.predicates[c] =
          ((thread_offset.column() + ThreadMap::Delta::kColumn * c) < extent.column());
    }
    if (!pointer) mask_.clear();

    // Column `c` sits at byte `c` of the halved row (see the class comment).
    store_byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                          LongIndex(thread_offset.row()) * LongIndex(params_.stride) +
                          LongIndex(thread_offset.column());

    state_[0] = state_[1] = state_[2] = 0;
  }

  /// Adds a pointer offset in units of Element.
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    store_byte_pointer_ += pointer_offset * cutlass::sizeof_bits<Element>::value / 8;
  }

  CUTLASS_DEVICE
  void store(Fragment const& frag) const {
    StoreType const* frag_ptr = reinterpret_cast<StoreType const*>(&frag);
    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int frag_row_idx =
              row + ThreadMap::Iterations::kRow *
                        (group + ThreadMap::Iterations::kGroup * cluster);
          LongIndex row_offset = row * LongIndex(ThreadMap::Delta::kRow) +
                                 group * LongIndex(ThreadMap::Delta::kGroup) +
                                 cluster * LongIndex(ThreadMap::Delta::kCluster);
          bool row_guard = (thread_start_row_ + row_offset) < extent_row_;
          uint8_t* row_pointer = store_byte_pointer_ + row_offset * params_.stride;

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            bool guard = row_guard && mask_.predicates[column];
            // `store_byte_pointer_` already carries the thread's column offset,
            // as `PredicatedTileIterator`'s does; only the access step is added.
            int column_offset = column * ThreadMap::Delta::kColumn;
            // Two store accesses per accumulator access: the live half is the
            // first (the functor filled it), the second is dead.
            cutlass::arch::global_store<StoreType, sizeof(StoreType)>(
                frag_ptr[2 * (frag_row_idx * ThreadMap::Iterations::kColumn + column)],
                static_cast<void*>(row_pointer + column_offset), guard);
          }
        }
      }
    }
  }

  /// Reads a fragment back, the inverse of `store`: one half access into the
  /// live half of each accumulator access, zeros into the dead half. Unreachable
  /// — `D` is written, never accumulated onto — but the epilogue's dispatch is a
  /// runtime `if`, so it has to compile.
  CUTLASS_DEVICE
  void load(Fragment& frag) const {
    StoreType* frag_ptr = reinterpret_cast<StoreType*>(&frag);
    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int frag_row_idx =
              row + ThreadMap::Iterations::kRow *
                        (group + ThreadMap::Iterations::kGroup * cluster);
          LongIndex row_offset = row * LongIndex(ThreadMap::Delta::kRow) +
                                 group * LongIndex(ThreadMap::Delta::kGroup) +
                                 cluster * LongIndex(ThreadMap::Delta::kCluster);
          bool row_guard = (thread_start_row_ + row_offset) < extent_row_;
          uint8_t const* row_pointer = store_byte_pointer_ + row_offset * params_.stride;

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            bool guard = row_guard && mask_.predicates[column];
            int column_offset = column * ThreadMap::Delta::kColumn;
            int chunk = frag_row_idx * ThreadMap::Iterations::kColumn + column;
            StoreType value;
            if (guard) value = *reinterpret_cast<StoreType const*>(row_pointer + column_offset);
            frag_ptr[2 * chunk] = value;
            frag_ptr[2 * chunk + 1].clear();
          }
        }
      }
    }
  }

  /// Advances one row band of the tile, as `PredicatedTileIterator` does.
  CUTLASS_HOST_DEVICE
  HalvedOutputTileIterator& operator++() {
    ++state_[0];
    store_byte_pointer_ += params_.stride * ThreadMap::Shape::kRow;
    thread_start_row_ += ThreadMap::Shape::kRow;

    if (state_[0] == ThreadMap::Count::kRow) {
      state_[0] = 0;
      ++state_[1];
      store_byte_pointer_ += params_.stride * (ThreadMap::Shape::kGroup - 1) *
                             ThreadMap::Shape::kRow * ThreadMap::Count::kRow;
      thread_start_row_ +=
          (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {
        state_[1] = 0;
        ++state_[2];
        store_byte_pointer_ += params_.stride * ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
                               ThreadMap::Count::kRow * ThreadMap::Shape::kRow;
        thread_start_row_ += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
                             ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;
          store_byte_pointer_ += params_.stride * ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow *
                                 ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile;
          thread_start_row_ += ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow *
                               ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile;
        }
      }
    }
    return *this;
  }

  CUTLASS_DEVICE
  void clear_mask() { mask_.clear(); }

  CUTLASS_DEVICE
  void enable_mask() { mask_.enable(); }
};

/// The epilogue the gate_up leg runs with: CUTLASS's default bf16 epilogue's
/// thread map and shared-memory staging, our functor and store.
using SiluOutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    ThreadblockShape, WarpShape, 1, Element, kElementsPerAccess>::Type;

using SiluAccumulatorFragmentIterator = cutlass::epilogue::warp::FragmentIteratorTensorOp<
    WarpShape, InstructionShape,
    typename GroupedKernel::Epilogue::WarpMmaOperator::Policy::Operator::ElementC,
    typename GroupedKernel::Epilogue::WarpMmaOperator::Policy::Operator::FragmentC,
    cutlass::layout::RowMajor>;

using SiluIterators = cutlass::epilogue::threadblock::detail::DefaultIteratorsTensorOp<
    Element, float, kElementsPerAccess, ThreadblockShape, WarpShape, InstructionShape,
    typename SiluOutputTileThreadMap::CompactedThreadMap>;

using SiluEpilogue = cutlass::epilogue::threadblock::Epilogue<
    ThreadblockShape, typename GroupedKernel::Epilogue::WarpMmaOperator, 1,
    HalvedOutputTileIterator<SiluOutputTileThreadMap>, SiluAccumulatorFragmentIterator,
    typename SiluIterators::WarpTileIterator, typename SiluIterators::SharedLoadIterator,
    SiluMulEpilogue,
    cutlass::MatrixShape<0, 64 / cutlass::sizeof_bits<float>::value * 4>,
    SiluIterators::kFragmentsPerIteration>;

using SiluGroupedKernel =
    cutlass::gemm::kernel::GemmGrouped<typename GroupedKernel::Mma, SiluEpilogue,
                                       ThreadblockSwizzle, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
                                       /*Transposed=*/false>;

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
// range; the grouped scheduler gives it no tiles. `y_stride` is the output row
// stride in elements — `n` itself, except on the fused leg, whose epilogue
// stores one column per pair (`n / 2`).
__global__ void qwen4_moe_grouped_args_kernel(const int* __restrict__ indptr,
                                              Element* __restrict__ x, Element* __restrict__ w,
                                              Element* __restrict__ y, GroupArgs args, int experts,
                                              int n, int k, int y_stride) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= experts) return;
  const int first = indptr[e];
  const int rows = indptr[e + 1] - first;
  args.problems[e] = Problem(rows, n, k);
  args.a[e] = x + static_cast<int64_t>(first) * k;
  args.b[e] = w + static_cast<int64_t>(e) * n * k;
  // C is never read (beta 0) but CUTLASS keeps the pointer array non-null.
  args.c[e] = y + static_cast<int64_t>(first) * y_stride;
  args.d[e] = args.c[e];
  args.lda[e] = k;
  args.ldb[e] = k;
  args.ldc[e] = y_stride;
  args.ldd[e] = y_stride;
}

template <typename Kernel>
cudaError_t run_grouped(GroupArgs const& args, int groups, int threadblocks, cudaStream_t stream) {
  using GemmGrouped = cutlass::gemm::device::GemmGrouped<Kernel>;
  using EpilogueOp = typename Kernel::Epilogue::OutputOp;
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
/// times the kernel's own max active blocks per SM. Queried once when the
/// wrapper builds its scratch, not per launch.
int sufficient_threadblocks() {
  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GroupedKernel>;
  return GemmGrouped::sufficient(/*problem_sizes=*/nullptr, /*problem_count=*/0);
}

/// Interleaves the gate and up halves of every expert's `gate_up` slab:
/// `dst[2j] = src[j]` (gate), `dst[2j + 1] = src[half + j]` (up), one 16 B
/// vector per thread. The output is a different slab, so this is a pure gather
/// with one writer per vector; nothing reads what another thread writes.
__global__ void qwen4_moe_gate_up_interleave_kernel(const uint4* __restrict__ src,
                                                    uint4* __restrict__ dst, int experts, int half,
                                                    int vectors_per_row) {
  const int64_t rows = static_cast<int64_t>(experts) * 2 * half;
  const int64_t total = rows * vectors_per_row;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t dst_row = idx / vectors_per_row;
    const int64_t vector = idx % vectors_per_row;
    const int64_t expert = dst_row / (2 * half);
    const int64_t row_in_block = dst_row % (2 * half);
    // Even destination rows carry the gate half, odd rows the up half.
    const int64_t src_row =
        (row_in_block & 1) ? (half + row_in_block / 2) : (row_in_block / 2);
    dst[idx] = src[(expert * 2 * half + src_row) * vectors_per_row + vector];
  }
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
  if (n % kElementsPerAccess != 0 || k % kElementsPerAccess != 0) {
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
        reinterpret_cast<Element*>(out), arrays, experts, n, k, n);
    const CUresult status = consume_last_cuda_error();
    if (status != CUDA_SUCCESS) return status;
  }
  const cudaError_t status = run_grouped<GroupedKernel>(arrays, experts, threadblocks, stream);
  return map_cuda_error(status == cudaSuccess ? cudaGetLastError() : status);
}

/// The gate_up leg with the SiLU-mul folded into the epilogue:
/// `out[first_e..] = silu_mul(x[first_e..] @ w[e]^T)`, where `w` is the
/// `[experts * 2 * n, k]` expert bank **interleaved** to `[g0, u0, g1, u1, ...]`
/// (`qwen4_moe_gate_up_interleave_cuda`) and `out` is the `[pairs, n]`
/// activated slot output — half the width the unfused leg wrote. `n` is the
/// output width; the GEMM still computes `2 * n` columns internally.
CUresult qwen4_moe_gate_up_silu_cuda(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                     __nv_bfloat16* out, const int* indptr, void* args,
                                     size_t args_bytes, int experts, int n, int k,
                                     int threadblocks, cudaStream_t stream) {
  if (x == nullptr || w == nullptr || out == nullptr || indptr == nullptr || args == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (experts <= 0 || n <= 0 || k <= 0 || threadblocks <= 0 || threadblocks > (1 << 20)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n % kElementsPerAccess != 0 || k % kElementsPerAccess != 0) {
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
        reinterpret_cast<Element*>(out), arrays, experts, 2 * n, k, n);
    const CUresult status = consume_last_cuda_error();
    if (status != CUDA_SUCCESS) return status;
  }
  const cudaError_t status = run_grouped<SiluGroupedKernel>(arrays, experts, threadblocks, stream);
  return map_cuda_error(status == cudaSuccess ? cudaGetLastError() : status);
}

/// Interleaves the gate and up halves of every expert's `[2 * half, width]`
/// `gate_up` slab in `src` into `dst`: `dst[2j] = src[j]`, `dst[2j + 1] =
/// src[half + j]`, every 16 B vector. `src` and `dst` must not overlap, and
/// `width` must be a multiple of 8.
CUresult qwen4_moe_gate_up_interleave_cuda(const __nv_bfloat16* src, __nv_bfloat16* dst,
                                           int experts, int half, int width,
                                           cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || src == dst) return CUDA_ERROR_INVALID_VALUE;
  if (experts <= 0 || half <= 0 || width <= 0) return CUDA_ERROR_INVALID_VALUE;
  if (width % kElementsPerAccess != 0) return CUDA_ERROR_INVALID_VALUE;
  const int64_t vectors = static_cast<int64_t>(experts) * 2 * half * (width / kElementsPerAccess);
  const int threads = 256;
  const int blocks = static_cast<int>(
      std::min<int64_t>((vectors + threads - 1) / threads, static_cast<int64_t>(1) << 20));
  qwen4_moe_gate_up_interleave_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint4*>(src), reinterpret_cast<uint4*>(dst), experts, half,
      width / kElementsPerAccess);
  return map_cuda_error(cudaGetLastError());
}

}  // extern "C"
