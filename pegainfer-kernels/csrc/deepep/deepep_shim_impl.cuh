// DeepEP elastic shim body: AOT kernel instantiations + torch-free host layer.
//
// Replaces upstream's NVRTC JIT (csrc/jit/) and torch host orchestration
// (csrc/elastic/buffer.hpp) for one baked single-node config:
//   - kernel template specializations instantiated at build time
//   - launches via cudaLaunchKernelExC with upstream's exact grid / cluster /
//     cooperative / PDL attributes (jit/launch_runtime.hpp, jit/handle.hpp)
//   - NCCL device-comm + symmetric-window context (backend/nccl.cu, pure-GPU
//     ncclMemAlloc path of backend/symmetric.hpp)
//   - dispatch/combine host flows (elastic/buffer.hpp), deterministic mode,
//     do_expand=true; decode without CPU sync, prefill with it
//
// Included exactly once per instantiation TU, which must define first:
//   DEEPEP_SHIM_CFG       — config namespace (see deepep_config_derived.cuh)
//   DEEPEP_SHIM_CTX       — the opaque context struct tag of this instance
//   DEEPEP_SHIM_FN(name)  — extern "C" symbol prefixing (e.g. deepep_##name)
// and include the matching ABI header (DeepEpInfo) + config header.
//
// Warp counts and capacities come from the config namespace; ctx_create
// runtime-asserts the constexpr mirrors against the real layout classes.

#ifndef DEEPEP_SHIM_HAS_PREFILL
#error "instantiation must define DEEPEP_SHIM_HAS_PREFILL"
#endif

#include <algorithm>
#if DEEPEP_SHIM_HAS_PREFILL
#include <atomic>
#include <chrono>
#endif
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/impls/barrier.cuh>
#include <deep_ep/impls/combine.cuh>
#include <deep_ep/impls/combine_reduce_epilogue.cuh>
#include <deep_ep/impls/dispatch.cuh>
#include <deep_ep/impls/dispatch_copy_epilogue.cuh>
#include <deep_ep/impls/dispatch_deterministic_prologue.cuh>

namespace {

using namespace deep_ep;
using namespace deep_ep::elastic;
namespace cfg = DEEPEP_SHIM_CFG;

// The C ABI hands top-k indices over as i32 (the routers' native output
// width); build with -DEP_NUM_TOPK_IDX_BITS=32. DeepEP's wire format stores
// indices as 32-bit regardless — this only types the input arrays.
static_assert(sizeof(topk_idx_t) == sizeof(int32_t),
              "compile with -DEP_NUM_TOPK_IDX_BITS=32 to match the C ABI");

// ---------------------------------------------------------------------------
// Error plumbing: extern "C" boundary must not unwind. EP_HOST_ASSERT /
// NCCL_CHECK / CUDA_RUNTIME_CHECK throw EPException; we catch everything.
// ---------------------------------------------------------------------------

thread_local std::string g_last_error;

void set_last_error(const char* what) { g_last_error = what; }

#define SHIM_API_BEGIN try {
#define SHIM_API_END                  \
    return 0;                         \
    }                                 \
    catch (const std::exception& e) { \
        set_last_error(e.what());     \
        return -1;                    \
    }                                 \
    catch (...) {                     \
        set_last_error("deepep shim: unknown error"); \
        return -1;                    \
    }

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("deepep shim: ") + what + ": " +
                                 cudaGetErrorString(err));
}

void check_nccl(ncclResult_t res, const char* what) {
    if (res != ncclSuccess)
        throw std::runtime_error(std::string("deepep shim: ") + what + ": " +
                                 ncclGetErrorString(res));
}

void require(bool cond, const char* what) {
    if (!cond) throw std::runtime_error(std::string("deepep shim: ") + what);
}

// ---------------------------------------------------------------------------
// Launch helper: replicates jit::LaunchArgs + construct_launch_config.
// Attributes are stack-local (upstream's static array is unsafe with one
// launching thread per rank).
// ---------------------------------------------------------------------------

void launch(const void* func, int grid, int threads, int smem_bytes, int cluster_dim,
            bool cooperative, bool pdl, cudaStream_t stream, void** args) {
    if (smem_bytes > 0)
        check_cuda(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_bytes),
                   "cudaFuncSetAttribute");

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(static_cast<unsigned>(grid), 1, 1);
    config.blockDim = dim3(static_cast<unsigned>(threads), 1, 1);
    config.dynamicSmemBytes = static_cast<size_t>(smem_bytes);
    config.stream = stream;

    cudaLaunchAttribute attrs[3];
    int num_attrs = 0;
    if (cooperative) {
        attrs[num_attrs].id = cudaLaunchAttributeCooperative;
        attrs[num_attrs].val.cooperative = 1;
        ++num_attrs;
    }
    if (cluster_dim > 1) {
        attrs[num_attrs].id = cudaLaunchAttributeClusterDimension;
        attrs[num_attrs].val.clusterDim = {static_cast<unsigned>(cluster_dim), 1, 1};
        ++num_attrs;
    }
    if (pdl) {
        attrs[num_attrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[num_attrs].val.programmaticStreamSerializationAllowed = 1;
        ++num_attrs;
    }
    config.attrs = attrs;
    config.numAttrs = num_attrs;

    check_cuda(cudaLaunchKernelExC(&config, func, args), "cudaLaunchKernelExC");
}

// ---------------------------------------------------------------------------
// Per-path kernel specializations. Both paths are deterministic
// (kReuseSlotIndices=true + slot prologue) and expanded (do_expand=true).
// ---------------------------------------------------------------------------

template <int kMaxTokens, int kNumSms, bool kCpuSync>
struct PathKernels {
    static constexpr int kGridSms = kNumSms;
    static constexpr int kDispatchWarps = cfg::dispatch_warps(kNumSms);
    static constexpr int kDispatchThreads = (cfg::kNumNotifyWarps + kDispatchWarps) * 32;
    // "Make cluster dim 2 to overlap with clustered computation kernels"
    // (dispatch.hpp / combine.hpp).
    static constexpr int kClusterDim = 2 - (kNumSms % 2);

    static const void* prologue() {
        return reinterpret_cast<const void*>(
            &dispatch_deterministic_prologue_impl<cfg::kDeviceSms, cfg::kPrologueWarps,
                                                  cfg::kNumRanks, kMaxTokens,
                                                  cfg::kNumExperts, cfg::kNumTopk>);
    }

    static const void* dispatch() {
        return reinterpret_cast<const void*>(
            &dispatch_impl</*kIsScaleupNVLink=*/true, kCpuSync, /*kReuseSlotIndices=*/true,
                           kNumSms, cfg::kNumNotifyWarps, kDispatchWarps, cfg::kNumRanks,
                           cfg::kHiddenBytes, /*kNumSFPacks=*/0, kMaxTokens,
                           cfg::kNumExperts, cfg::kNumTopk, cfg::kExpertAlignment,
                           cfg::kKernelQPs, cfg::kTimeoutCycles>);
    }

    static const void* copy_epilogue() {
        return reinterpret_cast<const void*>(
            &dispatch_copy_epilogue_impl</*kDoExpand=*/true, /*kCachedMode=*/false,
                                         cfg::kDeviceSms, /*kNumChannels=*/1,
                                         cfg::kCopyEpilogueWarps,
                                         /*kNumScaleoutRanks=*/1, cfg::kNumRanks,
                                         cfg::kHiddenBytes, /*kNumSFPacks=*/0, kMaxTokens,
                                         cfg::kNumExperts, cfg::kNumTopk>);
    }

    static const void* combine() {
        return reinterpret_cast<const void*>(
            &combine_impl</*kIsScaleupNVLink=*/true, /*kUseExpandedLayout=*/true,
                          /*kAllowMultipleReduction=*/true, kNumSms, cfg::kCombineWarps,
                          cfg::kNumRanks, cfg::kHidden, kMaxTokens, cfg::kNumExperts,
                          cfg::kNumTopk, cfg::kKernelQPs, cfg::kTimeoutCycles>);
    }

    static const void* reduce_epilogue() {
        return reinterpret_cast<const void*>(
            &combine_reduce_epilogue_impl</*kUseExpandedLayout=*/true,
                                          /*kAllowMultipleReduction=*/true, cfg::kDeviceSms,
                                          cfg::kReduceEpilogueWarps,
                                          /*kNumScaleoutRanks=*/1, cfg::kNumRanks,
                                          cfg::kHidden, kMaxTokens, cfg::kNumExperts,
                                          cfg::kNumTopk>);
    }
};

using Decode = PathKernels<cfg::kDecodeMaxTokens, cfg::kDecodeNumSms, /*kCpuSync=*/false>;
#if DEEPEP_SHIM_HAS_PREFILL
using Prefill = PathKernels<cfg::kPrefillMaxTokens, cfg::kPrefillNumSms, /*kCpuSync=*/true>;
#endif

const void* barrier_kernel() {
    return reinterpret_cast<const void*>(
        &barrier_impl</*kIsScaleupNVLink=*/true, /*kNumSMs=*/1, cfg::kBarrierThreads,
                      /*kNumScaleoutRanks=*/1, cfg::kNumRanks, cfg::kTimeoutCycles>);
}

// ---------------------------------------------------------------------------
// Buffer sizing via the real layout classes (buffer.hpp's direct-mode
// formulas with is_scaleup_nvlink=true and allow_multiple_reduction=true).
// ---------------------------------------------------------------------------

int64_t dispatch_buffer_bytes(int max_tokens) {
    const auto token_layout =
        layout::TokenLayout(cfg::kHiddenBytes, 0, cfg::kNumTopk, /*with_metadata=*/true);
    // NVLink scaleup: no send buffer (0 ranks).
    const auto send = layout::BufferLayout<false>(token_layout, 0, max_tokens);
    const auto recv = layout::BufferLayout<false>(token_layout, cfg::kNumRanks, max_tokens);
    return send.get_num_bytes() + recv.get_num_bytes();
}

int64_t combine_buffer_bytes(int max_tokens) {
    const auto token_layout =
        layout::TokenLayout(cfg::kHiddenBytes, 0, cfg::kNumTopk, /*with_metadata=*/false);
    const auto send = layout::BufferLayout<false>(token_layout, 0, max_tokens);
    const auto recv = layout::BufferLayout<false>(
        token_layout, cfg::min_i(cfg::kNumRanks, cfg::kNumTopk), max_tokens);
    return send.get_num_bytes() + recv.get_num_bytes();
}

constexpr int64_t kSymAlignment = 2097152;  // symmetric::kNumAlignmentBytes

int64_t aligned_workspace_bytes() {
    return math::align<int64_t>(layout::WorkspaceLayout::get_num_bytes(), kSymAlignment);
}

int64_t data_buffer_bytes() {
    int64_t bytes = 0;
    bytes = std::max(bytes, dispatch_buffer_bytes(cfg::kDecodeMaxTokens));
    bytes = std::max(bytes, combine_buffer_bytes(cfg::kDecodeMaxTokens));
#if DEEPEP_SHIM_HAS_PREFILL
    bytes = std::max(bytes, dispatch_buffer_bytes(cfg::kPrefillMaxTokens));
    bytes = std::max(bytes, combine_buffer_bytes(cfg::kPrefillMaxTokens));
#endif
    return math::align<int64_t>(bytes, kSymAlignment);
}

}  // namespace

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct DEEPEP_SHIM_CTX {
    int rank_idx = -1;
    int num_ranks = 0;

    ncclComm_t comm = nullptr;
    ncclDevComm_t dev_comm = {};
    ncclWindow_t window = nullptr;
    void* sym_base = nullptr;  // ncclMemAlloc'd: [workspace][data buffer]

    void* workspace = nullptr;  // LSA-mapped pointers into the window
    void* buffer = nullptr;

#if DEEPEP_SHIM_HAS_PREFILL
    void* host_workspace = nullptr;         // pinned CPU-sync counters
    void* mapped_host_workspace = nullptr;  // device mapping of host_workspace
#endif
};

namespace {

void launch_barrier_on(DEEPEP_SHIM_CTX* ctx, cudaStream_t stream) {
    int scaleout_rank_idx = 0;
    int scaleup_rank_idx = ctx->rank_idx;
    void* args[] = {&ctx->dev_comm, &ctx->window, &ctx->workspace, &scaleout_rank_idx,
                    &scaleup_rank_idx};
    launch(barrier_kernel(), /*grid=*/1, cfg::kBarrierThreads, /*smem=*/0,
           /*cluster=*/1, /*cooperative=*/true, /*pdl=*/false, stream, args);
}

#if DEEPEP_SHIM_HAS_PREFILL
layout::WorkspaceLayout host_workspace_layout(DEEPEP_SHIM_CTX* ctx) {
    return layout::WorkspaceLayout(ctx->host_workspace, /*num_scaleout_ranks=*/1,
                                   cfg::kNumRanks, cfg::kNumExperts);
}
#endif

// Shared dispatch-side launch sequence: deterministic slot prologue, then the
// dispatch kernel. The copy epilogue differs between decode (fixed worst case)
// and prefill (after the CPU count sync), so it is not part of this.
template <typename Path>
void dispatch_send_impl(DEEPEP_SHIM_CTX* ctx, cudaStream_t stream, const void* x,
                        const int32_t* topk_idx, const float* topk_weights,
                        int32_t num_tokens, int32_t* rank_count_scratch,
                        int32_t* dst_slot_scratch, int32_t* psum_rank,
                        int32_t* psum_expert, int max_tokens) {
    require(num_tokens >= 0 && num_tokens <= max_tokens,
            "dispatch: num_tokens out of range for this path");

    // Deterministic prologue: assign destination buffer slots in source order.
    {
        void* px = const_cast<void*>(static_cast<const void*>(topk_idx));
        int rank_idx = ctx->rank_idx;
        void* args[] = {&px, &rank_count_scratch, &dst_slot_scratch, &num_tokens, &rank_idx};
        launch(Path::prologue(), cfg::kDeviceSms, cfg::kPrologueWarps * 32, cfg::kSmemBytes,
               /*cluster=*/1, /*cooperative=*/true, /*pdl=*/false, stream, args);
    }

    // Dispatch kernel (direct mode argument order, dispatch.hpp launch_impl).
    {
        void* px = const_cast<void*>(x);
        sf_pack_t* sf = nullptr;
        auto* tk = const_cast<int32_t*>(topk_idx);
        auto* tw = const_cast<float*>(topk_weights);
        topk_idx_t* copied_topk_idx = nullptr;
        int* cumulative_stats = nullptr;
        int sf_token_stride = 0, sf_hidden_stride = 0;
        int rank_idx = ctx->rank_idx;
#if DEEPEP_SHIM_HAS_PREFILL
        void* mapped_host_workspace = ctx->mapped_host_workspace;
#else
        void* mapped_host_workspace = nullptr;  // unused when Path::kCpuSync is false
#endif
        void* args[] = {&px,
                        &sf,
                        &tk,
                        &tw,
                        &copied_topk_idx,
                        &cumulative_stats,
                        &psum_rank,
                        &psum_expert,
                        &dst_slot_scratch,
                        &num_tokens,
                        &sf_token_stride,
                        &sf_hidden_stride,
                        &ctx->dev_comm,
                        &ctx->window,
                        &ctx->buffer,
                        &ctx->workspace,
                        &mapped_host_workspace,
                        &rank_idx};
        launch(Path::dispatch(), Path::kGridSms, Path::kDispatchThreads, cfg::kSmemBytes,
               Path::kClusterDim, /*cooperative=*/true, /*pdl=*/false, stream, args);
    }
}

template <typename Path>
void copy_epilogue_impl(DEEPEP_SHIM_CTX* ctx, cudaStream_t stream, int32_t num_recv_tokens,
                        const int32_t* psum_rank, const int32_t* psum_expert, void* recv_x,
                        float* recv_topk_weights, int32_t* recv_src_metadata) {
    auto* pr = const_cast<int32_t*>(psum_rank);
    auto* pe = const_cast<int32_t*>(psum_expert);
    sf_pack_t* recv_sf = nullptr;
    topk_idx_t* recv_topk_idx = nullptr;  // expand mode: no per-slot topk idx
    int* channel_linked_list = nullptr;
    int recv_sf_token_stride = 0, recv_sf_hidden_stride = 0;
    int scaleout_rank_idx = 0;
    int scaleup_rank_idx = ctx->rank_idx;
    void* args[] = {&ctx->buffer,
                    &ctx->workspace,
                    &pr,
                    &pe,
                    &recv_x,
                    &recv_sf,
                    &recv_topk_idx,
                    &recv_topk_weights,
                    &recv_src_metadata,
                    &channel_linked_list,
                    &num_recv_tokens,
                    &recv_sf_token_stride,
                    &recv_sf_hidden_stride,
                    &scaleout_rank_idx,
                    &scaleup_rank_idx};
    launch(Path::copy_epilogue(), cfg::kDeviceSms, cfg::kCopyEpilogueWarps * 32,
           cfg::kSmemBytes, /*cluster=*/1, /*cooperative=*/false, /*pdl=*/true, stream, args);
}

template <typename Path>
void combine_impl_call(DEEPEP_SHIM_CTX* ctx, cudaStream_t stream, const void* x,
                       const int32_t* src_metadata, const int32_t* psum_rank,
                       int32_t num_reduced_tokens, const int32_t* combined_topk_idx,
                       int32_t num_tokens, void* combined_x) {
    // Combine: push expanded expert outputs into the symmetric buffer.
    {
        auto* px = const_cast<void*>(x);
        float* topk_weights = nullptr;  // expanded layout carries no weights
        auto* sm = const_cast<int32_t*>(src_metadata);
        auto* pr = const_cast<int32_t*>(psum_rank);
        int rank_idx = ctx->rank_idx;
        void* args[] = {&px,
                        &topk_weights,
                        &sm,
                        &pr,
                        &ctx->dev_comm,
                        &ctx->window,
                        &ctx->buffer,
                        &ctx->workspace,
                        &rank_idx,
                        &num_reduced_tokens};
        launch(Path::combine(), Path::kGridSms, cfg::kCombineWarps * 32, cfg::kSmemBytes,
               Path::kClusterDim, /*cooperative=*/true, /*pdl=*/false, stream, args);
    }

    // Reduce epilogue: sum each source token's slots into combined_x.
    // Direct mode reduces straight from the data buffer (launch_combine
    // returns `buffer` when num_scaleout_ranks == 1).
    {
        float* combined_topk_weights = nullptr;
        auto* ci = const_cast<int32_t*>(combined_topk_idx);
        void* bias_0 = nullptr;
        void* bias_1 = nullptr;
        int scaleout_rank_idx = 0;
        int scaleup_rank_idx = ctx->rank_idx;
        void* args[] = {&combined_x,    &combined_topk_weights,
                        &ci,            &ctx->buffer,
                        &bias_0,        &bias_1,
                        &num_tokens,    &scaleout_rank_idx,
                        &scaleup_rank_idx};
        launch(Path::reduce_epilogue(), cfg::kDeviceSms, cfg::kReduceEpilogueWarps * 32,
               cfg::kSmemBytes, /*cluster=*/1, /*cooperative=*/false, /*pdl=*/true, stream,
               args);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

extern "C" {

const char* DEEPEP_SHIM_FN(last_error)(void) { return g_last_error.c_str(); }

void DEEPEP_SHIM_FN(info)(DeepEpInfo* out) {
    out->num_ranks = cfg::kNumRanks;
    out->num_experts = cfg::kNumExperts;
    out->num_local_experts = cfg::kNumLocalExperts;
    out->num_topk = cfg::kNumTopk;
    out->hidden = cfg::kHidden;
    out->expert_alignment = cfg::kExpertAlignment;
    out->decode_max_tokens_per_rank = cfg::kDecodeMaxTokens;
    out->decode_worst_recv_tokens = cfg::kDecodeWorstRecvTokens;
    out->decode_worst_expanded_tokens = cfg::kDecodeWorstExpandedTokens;
#if DEEPEP_SHIM_HAS_PREFILL
    out->prefill_max_tokens_per_rank = cfg::kPrefillMaxTokens;
    out->prefill_worst_recv_tokens = cfg::kPrefillWorstRecvTokens;
#else
    out->prefill_max_tokens_per_rank = 0;
    out->prefill_worst_recv_tokens = 0;
#endif
    out->prologue_rank_count_len = cfg::kDeviceSms * cfg::kNumRanks;
    out->buffer_bytes = data_buffer_bytes();
    out->workspace_bytes = aligned_workspace_bytes();
}

int DEEPEP_SHIM_FN(unique_id)(uint8_t out[128]) {
    SHIM_API_BEGIN
    static_assert(sizeof(ncclUniqueId) == 128, "ncclUniqueId size changed");
    ncclUniqueId id;
    check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
    std::memcpy(out, &id, sizeof(id));
    SHIM_API_END
}

int DEEPEP_SHIM_FN(ctx_create)(const uint8_t unique_id[128], int32_t num_ranks,
                               int32_t rank_idx, DEEPEP_SHIM_CTX** out) {
    SHIM_API_BEGIN
    require(num_ranks == cfg::kNumRanks, "ctx_create: num_ranks != baked config");
    require(rank_idx >= 0 && rank_idx < num_ranks, "ctx_create: bad rank_idx");

    // NCCL device API (ncclDevComm / windows / GIN) needs >= 2.30.4.
    int nccl_version = 0;
    check_nccl(ncclGetVersion(&nccl_version), "ncclGetVersion");
    require(nccl_version >= 23004, "ctx_create: NCCL >= 2.30.4 required for the device API");

    // The baked grid/smem template parameters must fit the actual device.
    int device_idx = -1;
    check_cuda(cudaGetDevice(&device_idx), "cudaGetDevice");
    cudaDeviceProp prop = {};
    check_cuda(cudaGetDeviceProperties(&prop, device_idx), "cudaGetDeviceProperties");
    require(prop.multiProcessorCount >= cfg::kDeviceSms,
            "ctx_create: device has fewer SMs than the baked kDeviceSms");
    require(static_cast<int>(prop.sharedMemPerBlockOptin) >= cfg::kSmemBytes,
            "ctx_create: device smem-per-block below the baked kSmemBytes");

    // Pin the constexpr token-layout mirrors to the real layout classes.
    {
        const auto dispatch_tl =
            layout::TokenLayout(cfg::kHiddenBytes, 0, cfg::kNumTopk, /*with_metadata=*/true);
        const auto combine_tl =
            layout::TokenLayout(cfg::kHiddenBytes, 0, cfg::kNumTopk, /*with_metadata=*/false);
        require(dispatch_tl.get_num_bytes<true>() == cfg::kDispatchTokenSmem,
                "ctx_create: dispatch token layout drifted from constexpr mirror");
        require(combine_tl.get_num_bytes<true>() == cfg::kCombineTokenSmem,
                "ctx_create: combine token layout drifted from constexpr mirror");
        const auto notify_smem =
            math::align(cfg::kNumRanks + cfg::kNumExperts, cfg::kNumNotifyWarps * 32) *
            static_cast<int>(sizeof(int));
        require(notify_smem == cfg::kNotifySmemBytes,
                "ctx_create: notify smem drifted from constexpr mirror");
    }

    auto ctx = std::make_unique<DEEPEP_SHIM_CTX>();
    ctx->rank_idx = rank_idx;
    ctx->num_ranks = num_ranks;

    ncclUniqueId id;
    std::memcpy(&id, unique_id, sizeof(id));
    check_nccl(ncclCommInitRank(&ctx->comm, num_ranks, id, rank_idx), "ncclCommInitRank");

    // GIN device contexts, mirroring backend/nccl.cu for non-hybrid mode.
    // Single-node NVLink traffic does not go through GIN, but the device
    // handles are still constructed by the kernels; EP_DISABLE_GIN matches
    // the upstream escape hatch for NIC-less machines.
    const char* disable_gin_env = std::getenv("EP_DISABLE_GIN");
    const bool gin_disabled = disable_gin_env != nullptr && std::atoi(disable_gin_env) != 0;
    if (!gin_disabled) {
        ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
        check_nccl(ncclCommQueryProperties(ctx->comm, &props), "ncclCommQueryProperties");
        require(props.ginType != NCCL_GIN_TYPE_NONE,
                "ctx_create: NCCL GIN unavailable (set EP_DISABLE_GIN=1 on NIC-less nodes)");
    }
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    if (!gin_disabled) {
        reqs.ginContextCount = cfg::kAllocatedQPs;
        reqs.ginExclusiveContexts = true;
        reqs.ginQueueDepth = 1024;
        reqs.ginTrafficClass = 3;  // upstream sl_idx default
        // Customized RDMA barrier needs extra signals (backend/nccl.cu).
        reqs.ginSignalCount = num_ranks + 2 * 2;
        reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    }
    check_nccl(ncclDevCommCreate(ctx->comm, &reqs, &ctx->dev_comm), "ncclDevCommCreate");

    // The kernels are instantiated for a single NVLink scaleup domain.
    require(ctx->dev_comm.lsaSize == num_ranks && ctx->dev_comm.lsaRank == rank_idx,
            "ctx_create: ranks are not one NVLink (LSA) domain");

    // Symmetric memory: [workspace][data buffer], window-registered.
    const int64_t workspace_bytes = aligned_workspace_bytes();
    const int64_t total_bytes = workspace_bytes + data_buffer_bytes();
    check_nccl(ncclMemAlloc(&ctx->sym_base, total_bytes), "ncclMemAlloc");
    // Collective; internally barriers across ranks.
    check_nccl(ncclCommWindowRegister(ctx->comm, ctx->sym_base, total_bytes, &ctx->window,
                                      NCCL_WIN_DEFAULT),
               "ncclCommWindowRegister");
    void* mapped = nullptr;
    check_nccl(ncclGetLsaDevicePointer(ctx->window, 0, rank_idx, &mapped),
               "ncclGetLsaDevicePointer");
    ctx->workspace = mapped;
    ctx->buffer = static_cast<uint8_t*>(mapped) + workspace_bytes;
    check_cuda(cudaMemset(ctx->workspace, 0, workspace_bytes), "workspace memset");

#if DEEPEP_SHIM_HAS_PREFILL
    // Pinned host workspace for the CPU-sync counters (prefill path).
    check_cuda(cudaMallocHost(&ctx->host_workspace, layout::WorkspaceLayout::get_num_bytes(),
                              cudaHostAllocMapped),
               "cudaMallocHost");
    check_cuda(cudaHostGetDevicePointer(&ctx->mapped_host_workspace, ctx->host_workspace, 0),
               "cudaHostGetDevicePointer");
    std::memset(ctx->host_workspace, 0, layout::WorkspaceLayout::get_num_bytes());
#endif

    // Workspace zeroing must be visible to peers before any dispatch; the
    // window registration above already barriered after the memset's
    // submission order, but the memset is async — sync and barrier once.
    check_cuda(cudaDeviceSynchronize(), "ctx_create sync");
    launch_barrier_on(ctx.get(), /*stream=*/nullptr);
    check_cuda(cudaDeviceSynchronize(), "ctx_create barrier sync");

    *out = ctx.release();
    SHIM_API_END
}

int DEEPEP_SHIM_FN(ctx_destroy)(DEEPEP_SHIM_CTX* ctx) {
    SHIM_API_BEGIN
    check_cuda(cudaDeviceSynchronize(), "ctx_destroy pre-sync");
    launch_barrier_on(ctx, /*stream=*/nullptr);
    check_cuda(cudaDeviceSynchronize(), "ctx_destroy barrier sync");

#if DEEPEP_SHIM_HAS_PREFILL
    check_cuda(cudaFreeHost(ctx->host_workspace), "cudaFreeHost");
#endif
    check_nccl(ncclCommWindowDeregister(ctx->comm, ctx->window), "ncclCommWindowDeregister");
    check_nccl(ncclDevCommDestroy(ctx->comm, &ctx->dev_comm), "ncclDevCommDestroy");
    check_nccl(ncclMemFree(ctx->sym_base), "ncclMemFree");
    check_nccl(ncclCommDestroy(ctx->comm), "ncclCommDestroy");
    delete ctx;
    SHIM_API_END
}

int DEEPEP_SHIM_FN(decode_dispatch)(DEEPEP_SHIM_CTX* ctx, void* stream, const void* x,
                                    const int32_t* topk_idx, const float* topk_weights,
                                    int32_t num_tokens, int32_t* rank_count_scratch,
                                    int32_t* dst_slot_scratch, int32_t* psum_rank,
                                    int32_t* psum_expert, void* recv_x,
                                    float* recv_topk_weights, int32_t* recv_src_metadata) {
    SHIM_API_BEGIN
    auto s = static_cast<cudaStream_t>(stream);
    dispatch_send_impl<Decode>(ctx, s, x, topk_idx, topk_weights, num_tokens,
                               rank_count_scratch, dst_slot_scratch, psum_rank, psum_expert,
                               cfg::kDecodeMaxTokens);
    copy_epilogue_impl<Decode>(ctx, s, cfg::kDecodeWorstRecvTokens, psum_rank, psum_expert,
                               recv_x, recv_topk_weights, recv_src_metadata);
    SHIM_API_END
}

int DEEPEP_SHIM_FN(decode_combine)(DEEPEP_SHIM_CTX* ctx, void* stream, const void* x,
                                   const int32_t* src_metadata, const int32_t* psum_rank,
                                   const int32_t* combined_topk_idx, int32_t num_tokens,
                                   void* combined_x) {
    SHIM_API_BEGIN
    require(num_tokens >= 0 && num_tokens <= cfg::kDecodeMaxTokens,
            "decode_combine: num_tokens out of range");
    combine_impl_call<Decode>(ctx, static_cast<cudaStream_t>(stream), x, src_metadata,
                              psum_rank, cfg::kDecodeWorstRecvTokens, combined_topk_idx,
                              num_tokens, combined_x);
    SHIM_API_END
}

#if DEEPEP_SHIM_HAS_PREFILL
int DEEPEP_SHIM_FN(prefill_dispatch_send)(DEEPEP_SHIM_CTX* ctx, void* stream, const void* x,
                                          const int32_t* topk_idx, const float* topk_weights,
                                          int32_t num_tokens, int32_t* rank_count_scratch,
                                          int32_t* dst_slot_scratch, int32_t* psum_rank,
                                          int32_t* psum_expert) {
    SHIM_API_BEGIN
    // Zero the pinned counters the dispatch kernel will publish into
    // (encoded as -count-1, so zero reads as "not ready").
    const auto host_layout = host_workspace_layout(ctx);
    std::fill_n(host_layout.get_scaleup_rank_count_ptr<false>(),
                cfg::kNumRanks + cfg::kNumLocalExperts, 0);
    std::atomic_thread_fence(std::memory_order_seq_cst);

    dispatch_send_impl<Prefill>(ctx, static_cast<cudaStream_t>(stream), x, topk_idx,
                                topk_weights, num_tokens, rank_count_scratch,
                                dst_slot_scratch, psum_rank, psum_expert,
                                cfg::kPrefillMaxTokens);
    SHIM_API_END
}

int DEEPEP_SHIM_FN(prefill_wait_counts)(DEEPEP_SHIM_CTX* ctx, int32_t* num_recv_tokens,
                                        int32_t* num_expanded_tokens) {
    SHIM_API_BEGIN
    const auto host_layout = host_workspace_layout(ctx);
    const volatile int64_t* rank_counts = host_layout.get_scaleup_rank_count_ptr<false>();
    const volatile int64_t* expert_counts = host_layout.get_scaleup_expert_count_ptr<false>();

    const auto start = std::chrono::steady_clock::now();
    int rank_i = 0, expert_i = 0;
    int64_t recv = 0, expanded = 0;
    while (rank_i < cfg::kNumRanks || expert_i < cfg::kNumLocalExperts) {
        bool ready = true;
        while (rank_i < cfg::kNumRanks && ready) {
            const auto count = math::encode_decode_positive(rank_counts[rank_i]);
            if ((ready = math::is_decoded_positive_ready(count))) {
                recv += count;
                ++rank_i;
            }
        }
        while (expert_i < cfg::kNumLocalExperts && ready) {
            // Already aligned to kExpertAlignment by the dispatch kernel.
            const auto count = math::encode_decode_positive(expert_counts[expert_i]);
            if ((ready = math::is_decoded_positive_ready(count))) {
                expanded += count;
                ++expert_i;
            }
        }
        if (!ready &&
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() -
                                                             start)
                    .count() > 120)
            throw std::runtime_error("deepep shim: prefill dispatch CPU wait timed out");
    }
    *num_recv_tokens = static_cast<int32_t>(recv);
    *num_expanded_tokens = static_cast<int32_t>(expanded);
    SHIM_API_END
}

int DEEPEP_SHIM_FN(prefill_dispatch_recv)(DEEPEP_SHIM_CTX* ctx, void* stream,
                                          int32_t num_recv_tokens, const int32_t* psum_rank,
                                          const int32_t* psum_expert, void* recv_x,
                                          float* recv_topk_weights,
                                          int32_t* recv_src_metadata) {
    SHIM_API_BEGIN
    require(num_recv_tokens >= 0 && num_recv_tokens <= cfg::kPrefillWorstRecvTokens,
            "prefill_dispatch_recv: num_recv_tokens out of range");
    copy_epilogue_impl<Prefill>(ctx, static_cast<cudaStream_t>(stream), num_recv_tokens,
                                psum_rank, psum_expert, recv_x, recv_topk_weights,
                                recv_src_metadata);
    SHIM_API_END
}

int DEEPEP_SHIM_FN(prefill_combine)(DEEPEP_SHIM_CTX* ctx, void* stream, const void* x,
                                    const int32_t* src_metadata, const int32_t* psum_rank,
                                    int32_t num_recv_tokens, const int32_t* combined_topk_idx,
                                    int32_t num_tokens, void* combined_x) {
    SHIM_API_BEGIN
    require(num_tokens >= 0 && num_tokens <= cfg::kPrefillMaxTokens,
            "prefill_combine: num_tokens out of range");
    require(num_recv_tokens >= 0 && num_recv_tokens <= cfg::kPrefillWorstRecvTokens,
            "prefill_combine: num_recv_tokens out of range");
    combine_impl_call<Prefill>(ctx, static_cast<cudaStream_t>(stream), x, src_metadata,
                               psum_rank, num_recv_tokens, combined_topk_idx, num_tokens,
                               combined_x);
    SHIM_API_END
}
#endif

}  // extern "C"
