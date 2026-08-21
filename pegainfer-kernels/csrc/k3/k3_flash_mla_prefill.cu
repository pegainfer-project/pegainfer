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
//   - out is [t_q, h, 128] bf16. LSE is not written (single-call recipe).
//   - scale is the f32 softmax scale; pass the engine's bf16-rounded
//     `qk_head_dim ** -0.5` so decode and prefill agree on the constant.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <climits>

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
// specialized: non-varlen problem shape, causal mask, and the non-persistent
// causal tile scheduler upstream selects for causal + h % TileH == 0.
using HeadDim = Shape<_128, _64>;
using TileShape = Shape<_256, _128, HeadDim>;
using ProblemShape =
    cute::tuple<int, int, cute::tuple<int, int>, cute::tuple<cute::tuple<int, int>, int>>;
using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
using StrideV = StrideK;
using StrideO = StrideQ;
using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;
using Mask = CausalMask</*kIsQBegin=*/false>;
using Scheduler = CausalIndividualTileScheduler;

using Mainloop = Sm100MlaFwdMainloopTmaWarpspecialized<
    Element, ElementAcc, ElementAcc, TileShape, StrideQ, StrideK, StrideV, Mask,
    Shape<_2, _1, _1>, cute::false_type>;
using Epilogue = Sm100FmhaFwdEpilogueTmaWarpspecialized<
    Element, ElementAcc, typename Mainloop::TileShapePV, StrideO, StrideLSE, cute::false_type>;
using Operation = cutlass::fmha::device::FMHA<Sm100FmhaFwdKernelTmaWarpspecialized<
    ProblemShape, Mainloop, Epilogue, Scheduler, Sm100MlaFwdCtxKernelWarpspecializedSchedule>>;

}  // namespace k3_flash_mla_prefill

// The FMHA's max-dynamic-shared-memory attribute is per-device state, but the
// upstream device wrapper guards the cudaFuncSetAttribute behind a
// process-wide one-shot (`static bool initialized`). This process runs one
// executor per GPU, so the first rank to prefill would consume the one-shot
// for its device and leave every sibling launching ~200KB of dynamic smem
// against the default 48KB cap — cudaLaunchKernelExC then fails with
// `invalid argument`. Set the attribute ourselves, once per device; the
// upstream one-shot becomes a harmless no-op after the first rank.
static cudaError_t k3_flash_mla_prefill_ensure_smem_attr(int device) {
    constexpr int kMaxDevices = 64;
    static std::atomic<bool> g_smem_attr_done[kMaxDevices];
    if (device < 0 || device >= kMaxDevices) return cudaErrorInvalidValue;
    if (g_smem_attr_done[device].load(std::memory_order_acquire)) return cudaSuccess;
    using Kernel = k3_flash_mla_prefill::Operation::Kernel;
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
    int t_q,
    int t_kv,
    int heads,
    float scale,
    CUstream stream
) {
    namespace fmp = k3_flash_mla_prefill;
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || t_q <= 0 || t_kv <= 0
        || heads <= 0 || heads % fmp::Scheduler::TileH != 0 || t_q > t_kv) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // b = 1, but the batch stride slots are still materialized as ints.
    if ((long long)t_q * q_stride_tok > INT_MAX || (long long)t_kv * k_stride_tok > INT_MAX
        || (long long)t_kv * v_stride_tok > INT_MAX || (long long)t_q * o_stride_tok > INT_MAX) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return k3_flash_mla_map_cuda_error(err);
    err = k3_flash_mla_prefill_ensure_smem_attr(device);
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

    fmp::StrideQ stride_q = cute::make_stride(
        (int)q_stride_tok, cute::_1{},
        cute::make_stride(cute::make_stride((int)q_stride_head, (int)q_stride_head),
                          t_q * (int)q_stride_tok));
    fmp::StrideK stride_k = cute::make_stride(
        (int)k_stride_tok, cute::_1{},
        cute::make_stride(cute::make_stride(cute::_0{}, (int)k_stride_head),
                          t_kv * (int)k_stride_tok));
    fmp::StrideV stride_v = cute::make_stride(
        (int)v_stride_tok, cute::_1{},
        cute::make_stride(cute::make_stride(cute::_0{}, (int)v_stride_head),
                          t_kv * (int)v_stride_tok));
    fmp::StrideO stride_o = cute::make_stride(
        (int)o_stride_tok, cute::_1{},
        cute::make_stride(cute::make_stride((int)o_stride_head, (int)o_stride_head),
                          t_q * (int)o_stride_tok));
    // LSE is not written; the stride only has to be well-formed.
    fmp::StrideLSE stride_lse =
        cute::make_stride(cute::_1{}, cute::make_stride(cute::make_stride(t_q, t_q), t_q));

    typename fmp::Operation::Arguments arguments{
        problem_shape,
        {{static_cast<const fmp::Element*>(q), stride_q, static_cast<const fmp::Element*>(k),
          stride_k, static_cast<const fmp::Element*>(v), stride_v},
         scale},
        {static_cast<fmp::Element*>(out), stride_o, nullptr, stride_lse},
        hw_info};

    fmp::Operation op;
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

#else  // !K3_FLASH_MLA_SM100F

extern "C" CUresult k3_flash_mla_prefill_fwd(
    const void*, long long, long long, const void*, long long, long long, const void*, long long,
    long long, void*, long long, long long, int, int, int, float, CUstream
) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

#endif  // K3_FLASH_MLA_SM100F
