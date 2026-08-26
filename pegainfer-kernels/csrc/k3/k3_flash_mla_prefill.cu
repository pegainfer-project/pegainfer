// Kimi-K3 chunked-prefill MLA attention: C ABI over FlashMLA's SM100 dense
// FMHA forward (third_party/FlashMLA/csrc/sm100/prefill/dense — NVIDIA's
// CUTLASS FMHA contributed upstream, MIT/BSD per the FlashMLA repo).
//
// The prefill recipe is vLLM's: the paged latent stays the only persistent
// storage; per chunk the cached latent rows are gathered into a fixed
// workspace, expanded through `kv_b` into per-head K/V, and one dense
// varlen-shaped MHA call serves the whole `[context | chunk]` span with a
// bottom-right-aligned causal mask (`CausalMask<false>`: Q rows sit at the
// END of the KV axis, so chunk token `i` sees `context + i + 1` keys — no
// second pass and no LSE merge while the workspace covers the full context).
//
// One instantiation, matching pegainfer-k3's shapes: MLA head layout
// (d_qk = 128 nope + 64 rope = 192, d_vo = 128), bf16 in/out, b = 1
// non-varlen, causal, non-persistent scheduler (upstream picks the
// non-persistent path for causal). The gather and K-assembly helpers below
// are plain kernels that ride in the same TU so the whole prefill path is
// stubbed together off-Blackwell.
//
// Contract notes (mirroring the upstream torch binding, which is not vendored):
//   - q is [t_q, h, 192] bf16, per-head [nope | rope] — the raw `q_b` output
//     (K3 is NoPE: nothing is ever rotated).
//   - k is [t_kv, h, 192] bf16, assembled by `k3_mla_prefill_expand_k`.
//   - v is a strided view: [t_kv, h, 128] bf16 living inside the kv_b
//     expansion output [t_kv, h, 256] at element offset 128 (nope | value).
//   - out is [t_q, h, 128] bf16. lse_out, when non-null, receives the f32
//     per-row log-sum-exp `[h, t_q]` (natural log, softmax scale absorbed) —
//     the merge currency of the windowed context walk.
//   - scale is the f32 softmax scale; pass the engine's bf16-rounded
//     `qk_head_dim ** -0.5` so decode and prefill agree on the constant.
//
// Two mask instantiations share the launch core: the bottom-right causal one
// (`k3_flash_mla_prefill_fwd`) serves the window holding the queries, and a
// full-visibility one (`k3_flash_mla_prefill_fwd_dense`) serves windows that
// lie entirely in the queries' past — there `t_q` and `t_kv` are unrelated,
// so the causal entry's `t_q <= t_kv` guard does not apply. Window results
// combine through `k3_mla_prefill_lse_merge` (the log-sum-exp identity, exact
// up to summation order) and leave the f32 accumulator through
// `k3_mla_prefill_o_finalize`.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>

#ifdef K3_FLASH_MLA_SM100F
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_mla_fwd_mainloop_tma_warpspecialized.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "device/fmha.hpp"
#include "kernel/fmha_causal_tile_scheduler.hpp"
#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
#endif

// Not in an anonymous namespace: nvcc's device-stub generation trips over an
// anonymous-namespace __global__ when an included header (cute) opens its own.
static CUresult k3_flash_mla_map_cuda_error(cudaError_t err) {
    if (err == cudaSuccess) return CUDA_SUCCESS;
    if (err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
    if (err == cudaErrorNotSupported) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_ERROR_LAUNCH_FAILED;
}

static CUresult k3_flash_mla_consume_last_cuda_error() {
    return k3_flash_mla_map_cuda_error(cudaGetLastError());
}

// The paged latent row: post-norm kv latent | shared per-token rope half.
static constexpr int K3_FMP_LATENT = 512;
static constexpr int K3_FMP_ROPE = 64;
static constexpr int K3_FMP_ROW = K3_FMP_LATENT + K3_FMP_ROPE;  // 576
static constexpr int K3_FMP_NOPE = 128;
static constexpr int K3_FMP_QK = K3_FMP_NOPE + K3_FMP_ROPE;   // 192
static constexpr int K3_FMP_NV = K3_FMP_NOPE + 128;           // kv_b out: nope | value
static constexpr int K3_FMP_PAGE_TOKENS = 64;

// uint4 = 8 bf16 lanes; every span above is a multiple of 8.
static constexpr int K3_FMP_VEC = 8;

// Walk the block table and split each cached 576-wide latent row into the
// dense [t, 512] latent (kv_b's GEMM input) and [t, 64] rope halves.
static __global__ void k3_mla_prefill_gather_kernel(
    const uint4* __restrict__ slab,
    const int* __restrict__ table,
    long long page_stride_vec,
    long long layer_offset_vec,
    uint4* __restrict__ latent_out,
    uint4* __restrict__ rope_out,
    int t_total
) {
    constexpr int ROW_VEC = K3_FMP_ROW / K3_FMP_VEC;        // 72
    constexpr int LATENT_VEC = K3_FMP_LATENT / K3_FMP_VEC;  // 64
    constexpr int ROPE_VEC = K3_FMP_ROPE / K3_FMP_VEC;      // 8
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)t_total * ROW_VEC;
    if (idx >= total) {
        return;
    }
    int t = (int)(idx / ROW_VEC);
    int c = (int)(idx - (long long)t * ROW_VEC);
    int page = table[t / K3_FMP_PAGE_TOKENS];
    long long src = (long long)page * page_stride_vec + layer_offset_vec
        + (long long)(t % K3_FMP_PAGE_TOKENS) * ROW_VEC + c;
    uint4 value = slab[src];
    if (c < LATENT_VEC) {
        latent_out[(long long)t * LATENT_VEC + c] = value;
    } else {
        rope_out[(long long)t * ROPE_VEC + (c - LATENT_VEC)] = value;
    }
}

// Assemble the per-head K rows: nope from the kv_b expansion, the shared
// per-token rope half broadcast across heads. V needs no copy — the caller
// hands the FMHA a strided view into the expansion output.
static __global__ void k3_mla_prefill_expand_k_kernel(
    const uint4* __restrict__ nope_v,
    const uint4* __restrict__ rope,
    uint4* __restrict__ k_out,
    int t_total,
    int heads
) {
    constexpr int QK_VEC = K3_FMP_QK / K3_FMP_VEC;      // 24
    constexpr int NOPE_VEC = K3_FMP_NOPE / K3_FMP_VEC;  // 16
    constexpr int NV_VEC = K3_FMP_NV / K3_FMP_VEC;      // 32
    constexpr int ROPE_VEC = K3_FMP_ROPE / K3_FMP_VEC;  // 8
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)t_total * heads * QK_VEC;
    if (idx >= total) {
        return;
    }
    int c = (int)(idx % QK_VEC);
    long long th = idx / QK_VEC;
    uint4 value;
    if (c < NOPE_VEC) {
        value = nope_v[th * NV_VEC + c];
    } else {
        int t = (int)(th / heads);
        value = rope[(long long)t * ROPE_VEC + (c - NOPE_VEC)];
    }
    k_out[idx] = value;
}

extern "C" CUresult k3_mla_prefill_gather(
    const void* slab,
    const int* table,
    long long page_stride,   // elements between pages in the slab
    long long layer_offset,  // this MLA layer's row shift inside a page, elements
    int t_total,
    void* latent_out,
    void* rope_out,
    CUstream stream
) {
    if (slab == nullptr || table == nullptr || latent_out == nullptr || rope_out == nullptr
        || t_total <= 0 || page_stride % K3_FMP_VEC != 0 || layer_offset % K3_FMP_VEC != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    long long total = (long long)t_total * (K3_FMP_ROW / K3_FMP_VEC);
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    k3_mla_prefill_gather_kernel<<<(unsigned)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const uint4*)slab, table, page_stride / K3_FMP_VEC, layer_offset / K3_FMP_VEC,
        (uint4*)latent_out, (uint4*)rope_out, t_total);
    return k3_flash_mla_consume_last_cuda_error();
}

// Fold one window's FMHA output into the running f32 accumulator via the
// log-sum-exp identity. One block per (q, head) row: only this block touches
// its lse entry, so the read-old/write-new sequence needs no cross-block
// coordination. `reset` starts a fresh accumulation (no -inf seeding).
static __global__ void k3_mla_prefill_lse_merge_kernel(
    const __nv_bfloat16* __restrict__ o_win,
    const float* __restrict__ lse_win,
    float* __restrict__ o_acc,
    float* __restrict__ lse_acc,
    int t_q,
    int heads,
    int reset
) {
    int q = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    long long lse_idx = (long long)h * t_q + q;
    __shared__ float w_acc, w_win;
    if (d == 0) {
        float lw = lse_win[lse_idx];
        if (reset) {
            w_acc = 0.0f;
            w_win = 1.0f;
            lse_acc[lse_idx] = lw;
        } else {
            float la = lse_acc[lse_idx];
            float m = fmaxf(la, lw);
            float merged = m + logf(expf(la - m) + expf(lw - m));
            w_acc = expf(la - merged);
            w_win = expf(lw - merged);
            lse_acc[lse_idx] = merged;
        }
    }
    __syncthreads();
    long long o_idx = ((long long)q * heads + h) * K3_FMP_NOPE + d;
    float prev = reset ? 0.0f : o_acc[o_idx];
    o_acc[o_idx] = prev * w_acc + __bfloat162float(o_win[o_idx]) * w_win;
}

// Leave the f32 accumulator: out[t_q, h, 128] bf16.
static __global__ void k3_mla_prefill_o_finalize_kernel(
    const float* __restrict__ o_acc,
    __nv_bfloat16* __restrict__ out,
    long long total
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    out[idx] = __float2bfloat16(o_acc[idx]);
}

extern "C" CUresult k3_mla_prefill_lse_merge(
    const void* o_win,
    const void* lse_win,
    void* o_acc,
    void* lse_acc,
    int t_q,
    int heads,
    int reset,
    CUstream stream
) {
    if (o_win == nullptr || lse_win == nullptr || o_acc == nullptr || lse_acc == nullptr
        || t_q <= 0 || heads <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    dim3 grid((unsigned)t_q, (unsigned)heads);
    k3_mla_prefill_lse_merge_kernel<<<grid, K3_FMP_NOPE, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)o_win, (const float*)lse_win, (float*)o_acc, (float*)lse_acc,
        t_q, heads, reset);
    return k3_flash_mla_consume_last_cuda_error();
}

extern "C" CUresult k3_mla_prefill_o_finalize(
    const void* o_acc,
    void* out,
    int t_q,
    int heads,
    CUstream stream
) {
    if (o_acc == nullptr || out == nullptr || t_q <= 0 || heads <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    long long total = (long long)t_q * heads * K3_FMP_NOPE;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    k3_mla_prefill_o_finalize_kernel<<<(unsigned)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const float*)o_acc, (__nv_bfloat16*)out, total);
    return k3_flash_mla_consume_last_cuda_error();
}

extern "C" CUresult k3_mla_prefill_expand_k(
    const void* nope_v,
    const void* rope,
    void* k_out,
    int t_total,
    int heads,
    CUstream stream
) {
    if (nope_v == nullptr || rope == nullptr || k_out == nullptr || t_total <= 0 || heads <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    long long total = (long long)t_total * heads * (K3_FMP_QK / K3_FMP_VEC);
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    k3_mla_prefill_expand_k_kernel<<<(unsigned)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const uint4*)nope_v, (const uint4*)rope, (uint4*)k_out, t_total, heads);
    return k3_flash_mla_consume_last_cuda_error();
}

#ifdef K3_FLASH_MLA_SM100F

// Named (not anonymous) namespace: these types parameterize a __global__
// kernel through cutlass::device_kernel, same nvcc pitfall as above.
namespace k3_flash_mla_prefill {

using namespace cute;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha::kernel;

using Element = cutlass::bfloat16_t;
using ElementAcc = float;

// The upstream FwdRunner's MLA configuration (fmha_cutlass_fwd_sm100.cuh),
// specialized: non-varlen problem shape, parameterized over the mask/scheduler
// pair. The causal pair is the mask upstream selects for causal +
// h % TileH == 0; the dense pair serves fully-visible context windows.
using HeadDim = Shape<_128, _64>;
using TileShape = Shape<_256, _128, HeadDim>;
using ProblemShape =
    cute::tuple<int, int, cute::tuple<int, int>, cute::tuple<cute::tuple<int, int>, int>>;
using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
using StrideV = StrideK;
using StrideO = StrideQ;
using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;

template <class Mask, class Scheduler>
struct Config {
    using Mainloop = Sm100MlaFwdMainloopTmaWarpspecialized<
        Element, ElementAcc, ElementAcc, TileShape, StrideQ, StrideK, StrideV, Mask,
        Shape<_2, _1, _1>, cute::false_type>;
    using Epilogue = Sm100FmhaFwdEpilogueTmaWarpspecialized<
        Element, ElementAcc, typename Mainloop::TileShapePV, StrideO, StrideLSE,
        cute::false_type>;
    using Operation = cutlass::fmha::device::FMHA<Sm100FmhaFwdKernelTmaWarpspecialized<
        ProblemShape, Mainloop, Epilogue, Scheduler, Sm100MlaFwdCtxKernelWarpspecializedSchedule>>;
};

using CausalConfig = Config<CausalMask</*kIsQBegin=*/false>, CausalIndividualTileScheduler>;
using DenseConfig = Config<ResidualMask, IndividualTileScheduler>;

// The causal scheduler asserts h % TileH == 0; keep both entries honest.
constexpr int kTileH = CausalIndividualTileScheduler::TileH;

}  // namespace k3_flash_mla_prefill

// The FMHA's max-dynamic-shared-memory attribute is per-device state, but the
// upstream device wrapper guards the cudaFuncSetAttribute behind a
// process-wide one-shot (`static bool initialized`). This process runs one
// executor per GPU, so the first rank to prefill would consume the one-shot
// for its device and leave every sibling launching ~200KB of dynamic smem
// against the default 48KB cap — cudaLaunchKernelExC then fails with
// `invalid argument`. Set the attribute ourselves, once per device; the
// upstream one-shot becomes a harmless no-op after the first rank.
template <class Kernel>
static cudaError_t k3_flash_mla_prefill_ensure_smem_attr(int device) {
    constexpr int kMaxDevices = 64;
    static std::atomic<bool> g_smem_attr_done[kMaxDevices];
    if (device < 0 || device >= kMaxDevices) return cudaErrorInvalidValue;
    if (g_smem_attr_done[device].load(std::memory_order_acquire)) return cudaSuccess;
    int smem_size = Kernel::SharedStorageSize;
    if (smem_size >= (48 << 10)) {
        cudaError_t err = cudaFuncSetAttribute(
            (const void*)cutlass::device_kernel<Kernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        if (err != cudaSuccess) {
            (void)cudaGetLastError();  // clear so the failure is not sticky
            return err;
        }
    }
    // Concurrent first calls on the same device both set the attribute; the
    // set is idempotent, so the race is benign.
    g_smem_attr_done[device].store(true, std::memory_order_release);
    return cudaSuccess;
}

template <class Cfg>
static CUresult k3_flash_mla_prefill_run(
    const void* q,
    long long q_stride_tok,
    long long q_stride_head,
    const void* k,
    long long k_stride_tok,
    long long k_stride_head,
    const void* v,
    long long v_stride_tok,
    long long v_stride_head,
    void* out,
    long long o_stride_tok,
    long long o_stride_head,
    void* lse_out,
    int t_q,
    int t_kv,
    int heads,
    float scale,
    CUstream stream
) {
    namespace fmp = k3_flash_mla_prefill;
    using Operation = typename Cfg::Operation;
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return k3_flash_mla_map_cuda_error(err);
    err = k3_flash_mla_prefill_ensure_smem_attr<typename Operation::Kernel>(device);
    if (err != cudaSuccess) return k3_flash_mla_map_cuda_error(err);
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);

    // ((h_r = 1, h_k = heads), b = 1): every q head has its own expanded
    // K/V head, so the within-group K stride is the compiled _0.
    fmp::ProblemShape problem_shape = cute::make_tuple(
        t_q, t_kv, cute::make_tuple(K3_FMP_NOPE, K3_FMP_ROPE),
        cute::make_tuple(cute::make_tuple(1, heads), 1));

    // b = 1 (ProblemShape above): the batch coordinate is always 0, so these
    // int batch strides never offset. They carry 0 because the true extent
    // t_kv * stride_tok overflows int32 past ~116k tokens at heads = 96.
    fmp::StrideQ stride_q = cute::make_stride(
        (int)q_stride_tok, cute::_1{},
        cute::make_stride(cute::make_stride((int)q_stride_head, (int)q_stride_head), 0));
    fmp::StrideK stride_k = cute::make_stride(
        (int)k_stride_tok, cute::_1{},
        cute::make_stride(cute::make_stride(cute::_0{}, (int)k_stride_head), 0));
    fmp::StrideV stride_v = cute::make_stride(
        (int)v_stride_tok, cute::_1{},
        cute::make_stride(cute::make_stride(cute::_0{}, (int)v_stride_head), 0));
    fmp::StrideO stride_o = cute::make_stride(
        (int)o_stride_tok, cute::_1{},
        cute::make_stride(cute::make_stride((int)o_stride_head, (int)o_stride_head), 0));
    // LSE (natural log, softmax scale absorbed) lands as f32 [heads, t_q]
    // when lse_out is non-null; the stride must be well-formed either way.
    fmp::StrideLSE stride_lse =
        cute::make_stride(cute::_1{}, cute::make_stride(cute::make_stride(t_q, t_q), t_q));

    typename Operation::Arguments arguments{
        problem_shape,
        {{static_cast<const fmp::Element*>(q), stride_q, static_cast<const fmp::Element*>(k),
          stride_k, static_cast<const fmp::Element*>(v), stride_v},
         scale},
        {static_cast<fmp::Element*>(out), stride_o, static_cast<fmp::ElementAcc*>(lse_out),
         stride_lse},
        hw_info};

    Operation op;
    if (op.can_implement(arguments) != cutlass::Status::kSuccess) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (op.initialize(arguments, nullptr) != cutlass::Status::kSuccess) {
        return CUDA_ERROR_LAUNCH_FAILED;
    }
    if (op.run((cudaStream_t)stream) != cutlass::Status::kSuccess) {
        return CUDA_ERROR_LAUNCH_FAILED;
    }
    return k3_flash_mla_consume_last_cuda_error();
}

extern "C" CUresult k3_flash_mla_prefill_fwd(
    const void* q,
    long long q_stride_tok,
    long long q_stride_head,
    const void* k,
    long long k_stride_tok,
    long long k_stride_head,
    const void* v,
    long long v_stride_tok,
    long long v_stride_head,
    void* out,
    long long o_stride_tok,
    long long o_stride_head,
    void* lse_out,
    int t_q,
    int t_kv,
    int heads,
    float scale,
    CUstream stream
) {
    namespace fmp = k3_flash_mla_prefill;
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || t_q <= 0 || t_kv <= 0
        || heads <= 0 || heads % fmp::kTileH != 0 || t_q > t_kv) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return k3_flash_mla_prefill_run<fmp::CausalConfig>(
        q, q_stride_tok, q_stride_head, k, k_stride_tok, k_stride_head, v, v_stride_tok,
        v_stride_head, out, o_stride_tok, o_stride_head, lse_out, t_q, t_kv, heads, scale,
        stream);
}

// Full-visibility window: every query row attends every key. `t_q` and `t_kv`
// are unrelated (a ragged context tail can be far narrower than the chunk).
extern "C" CUresult k3_flash_mla_prefill_fwd_dense(
    const void* q,
    long long q_stride_tok,
    long long q_stride_head,
    const void* k,
    long long k_stride_tok,
    long long k_stride_head,
    const void* v,
    long long v_stride_tok,
    long long v_stride_head,
    void* out,
    long long o_stride_tok,
    long long o_stride_head,
    void* lse_out,
    int t_q,
    int t_kv,
    int heads,
    float scale,
    CUstream stream
) {
    namespace fmp = k3_flash_mla_prefill;
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || t_q <= 0 || t_kv <= 0
        || heads <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return k3_flash_mla_prefill_run<fmp::DenseConfig>(
        q, q_stride_tok, q_stride_head, k, k_stride_tok, k_stride_head, v, v_stride_tok,
        v_stride_head, out, o_stride_tok, o_stride_head, lse_out, t_q, t_kv, heads, scale,
        stream);
}

#else  // !K3_FLASH_MLA_SM100F

extern "C" CUresult k3_flash_mla_prefill_fwd(
    const void*, long long, long long, const void*, long long, long long, const void*, long long,
    long long, void*, long long, long long, void*, int, int, int, float, CUstream
) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult k3_flash_mla_prefill_fwd_dense(
    const void*, long long, long long, const void*, long long, long long, const void*, long long,
    long long, void*, long long, long long, void*, int, int, int, float, CUstream
) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

#endif  // K3_FLASH_MLA_SM100F
