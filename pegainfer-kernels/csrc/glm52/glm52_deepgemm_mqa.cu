// GLM5.2 DSA indexer: DeepGEMM paged MQA logits, AOT-instantiated (no JIT).
//
// The decode path's codegen parameters are all compile-time constants
// (next_n=1, 32 heads, head_dim 128, block_kv 64, split_kv 256, bf16 logits,
// batch <= 64, 132 SMs), so both kernels are instantiated here directly from
// DeepGEMM's device headers and launched with cudaLaunchKernelExC. This
// removes DeepGEMM's runtime JIT entirely — its compiler, include parser and
// launch-config helpers keep unsynchronized global state that the DP8
// coordinator's 8 concurrent rank threads corrupt (include-parser
// assertions, per-context CUfunction handles, a shared static attrs array),
// and the per-launch codegen + code hashing cost ~0.4 ms per serialized
// call. It also drops the PEGAINFER_DEEPGEMM_ROOT / CUDA_HOME runtime
// requirements — nothing is compiled at runtime anymore.
//
// Requires sm_100f on Blackwell (GLM5.2 is Blackwell-only). DG_NO_TORCH is
// defined via build.rs.

#include "../common.cuh"

#include <cuda.h>
#include <cstdint>
#include <cstdio>

// Without an sm_100f nvcc target the architecture-specific device code
// cannot be assembled; build.rs then omits this define and entry points
// compile as NOT_SUPPORTED stubs.
#ifdef GLM52_DEEPGEMM_MQA_SM100F

#include <jit_kernels/impls/runtime_utils.hpp>

#include <deep_gemm/impls/sm100_mqa_logits.cuh>
#include <deep_gemm/layout/mqa_logits.cuh>

namespace {

constexpr int kSM100SmemCapacity = 232448;
constexpr int kSplitKv = 256;
constexpr int kSplitsPerChunk = 16;
constexpr int kNumSpecializedThreads = 128;
constexpr int kNumMathThreads = 2 * 128;
constexpr int kNumQStages = 3;
constexpr int kNumKVStages = 5;

// AOT instantiation shape: the GLM5.2 DSA decode indexer.
constexpr int kAotNextN = 1;
constexpr int kAotNumHeads = 32;
constexpr int kAotHeadDim = 128;
constexpr int kAotBlockQ = 128 / kAotNumHeads;
constexpr int kAotBlockKv = 64;
constexpr int kAotNumSms = 132;
// Launcher guard only — batch is a runtime argument of both kernels (the
// grid-stride scheduler walks q atoms); no template arg depends on it.
// 64 covers the verify-span row ceiling (48, #812).
constexpr int kAotAlignedBatchSize = 96;

const auto kMetadataKernel = &deep_gemm::sched::sm100_paged_mqa_logits_metadata<
    kAotNextN, /*kIsContextLens2D=*/false, /*kIsVarlen=*/false,
    /*BLOCK_Q=*/1, kSplitKv, kAotNumSms>;

// Upstream #377 re-parameterized the MQA-logits templates: the leading
// `kIsFP4` flag became a trailing `qk_dtype_t` type argument plus a `kIsMXSF`
// flag (MX scale-factor path, shared by MXFP4 and MXFP8). The GLM5.2 indexer
// is plain FP8 e4m3 with per-KV f32 scales, so `kIsMXSF=false` and
// `qk_dtype_t = cutlass::float_e4m3_t` reproduce the previous behaviour.
const auto kLogitsKernel = &deep_gemm::sm100_paged_mqa_logits<
    kAotNextN, kAotNumHeads, kAotHeadDim, kAotBlockKv,
    /*kIsMXSF=*/false, /*kIsContextLens2D=*/false, /*kIsVarlen=*/false,
    kNumQStages, kNumKVStages, kSplitKv, kSplitsPerChunk,
    kNumSpecializedThreads, kNumMathThreads, cutlass::float_e4m3_t,
    cutlass::bfloat16_t, float>;

// Unpaged (contiguous-KV) prefill instantiation: DeepGEMM's `fp8_mqa_logits`
// path (the kernel vLLM uses for the DeepSeek-V3.2 indexer prefill).
//  - kIsCompressedLogits=false: logits column n is the ABSOLUTE kv token
//    index (not relative to `cu_seqlen_ks`). Columns inside a scheduled
//    256-wide split but outside a row's [ks, ke) range receive unmasked
//    garbage scores; the consumer must mask by [ks, ke) (DeepGEMM's own API
//    runs `smxx_clean_logits` for this).
//  - logits dtype float (fp32), matching DeepGEMM's `fp8_mqa_logits`
//    (`logits_dtype = torch::kFloat`).
//  - kNumSMs is a template constant consumed by the grid-stride scheduler;
//    the grid MUST be exactly kAotNumSms blocks.
//  - No schedule-metadata kernel: the contiguous-KV scheduler derives its
//    work list on the fly from `cu_seqlen_ks/ke` inside the kernel.
const auto kUnpagedLogitsKernel = &deep_gemm::sm100_mqa_logits<
    kAotNumHeads, kAotHeadDim,
    /*kIsMXSF=*/false, /*kIsCompressedLogits=*/false,
    /*BLOCK_Q=*/kAotBlockQ, kSplitKv,
    kNumQStages, kNumKVStages,
    /*kNumSMs=*/kAotNumSms,
    kNumSpecializedThreads, kNumMathThreads,
    /*qk_dtype_t=*/cutlass::float_e4m3_t,
    /*logits_dtype_t=*/float, /*reduce_dtype_t=*/float>;

CUresult launch_aot(const void* func, dim3 grid_dim, dim3 block_dim, int smem_size,
                    cudaStream_t stream, void** args) {
    if (smem_size > 0) {
        const cudaError_t attr_err = cudaFuncSetAttribute(
            func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        if (attr_err != cudaSuccess) {
            fprintf(stderr, "glm52_deepgemm_mqa: cudaFuncSetAttribute failed: %s\n",
                    cudaGetErrorString(attr_err));
            return CUDA_ERROR_LAUNCH_FAILED;
        }
    }

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config = {};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = static_cast<size_t>(smem_size);
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 1;

    const cudaError_t err = cudaLaunchKernelExC(&config, func, args);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm52_deepgemm_mqa: launch failed: %s\n", cudaGetErrorString(err));
        return CUDA_ERROR_LAUNCH_FAILED;
    }
    return CUDA_SUCCESS;
}

// Plain-stream-order launch (no programmatic stream serialization): the
// prefill-side unpaged logits launch takes normal stream semantics so it can
// never begin before its predecessors (K gather, ke upload) fully complete.
// The decode-side paged chain keeps the PDL attribute above.
CUresult launch_aot_plain(const void* func, dim3 grid_dim, dim3 block_dim, int smem_size,
                          cudaStream_t stream, void** args) {
    if (smem_size > 0) {
        const cudaError_t attr_err = cudaFuncSetAttribute(
            func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        if (attr_err != cudaSuccess) {
            fprintf(stderr, "glm52_deepgemm_mqa: cudaFuncSetAttribute failed: %s\n",
                    cudaGetErrorString(attr_err));
            return CUDA_ERROR_LAUNCH_FAILED;
        }
    }
    cudaLaunchConfig_t config = {};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = static_cast<size_t>(smem_size);
    config.stream = stream;
    const cudaError_t err = cudaLaunchKernelExC(&config, func, args);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm52_deepgemm_mqa: launch failed: %s\n", cudaGetErrorString(err));
        return CUDA_ERROR_LAUNCH_FAILED;
    }
    return CUDA_SUCCESS;
}

} // namespace

extern "C" {

CUresult glm52_deepgemm_paged_mqa_metadata_cuda(
    int* context_lens,
    int* schedule_metadata,
    int batch_size,
    int next_n,
    int block_kv,
    int num_sms,
    bool is_context_lens_2d,
    bool is_varlen,
    const int* indices_ptr,
    cudaStream_t stream
) {
    if (!context_lens || !schedule_metadata || batch_size <= 0 || block_kv <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (batch_size > kAotAlignedBatchSize || next_n != kAotNextN ||
        block_kv != kAotBlockKv || num_sms != kAotNumSms ||
        is_context_lens_2d || is_varlen || indices_ptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const int smem_size = 2 * batch_size * static_cast<int>(sizeof(int));
    if (smem_size > kSM100SmemCapacity) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const uint32_t arg_num_requests = static_cast<uint32_t>(batch_size);
    const uint32_t arg_num_q_tokens_total = static_cast<uint32_t>(batch_size * next_n);
    const uint32_t* arg_context_lens = reinterpret_cast<const uint32_t*>(context_lens);
    const uint32_t* arg_indices = nullptr;
    uint32_t* arg_schedule = reinterpret_cast<uint32_t*>(schedule_metadata);
    void* args[] = {
        const_cast<uint32_t*>(&arg_num_requests),
        const_cast<uint32_t*>(&arg_num_q_tokens_total),
        &arg_context_lens,
        &arg_indices,
        &arg_schedule,
    };
    return launch_aot(reinterpret_cast<const void*>(kMetadataKernel),
                      dim3(1, 1, 1), dim3(256, 1, 1), smem_size, stream, args);
}

CUresult glm52_deepgemm_paged_mqa_logits_cuda(
    const void* q,
    const void* kv_cache,
    int64_t kv_cache_stride_bytes,
    const void* weights,
    const int* context_lens,
    void* logits,
    const int* block_table,
    const int* indices,
    int* schedule_meta,
    int batch_size,
    int next_n,
    int num_heads,
    int head_dim,
    int num_kv_blocks,
    int block_kv,
    bool is_context_lens_2d,
    bool is_varlen,
    int logits_stride,
    int block_table_stride,
    int num_sms,
    int q_elem_size,
    int kv_elem_size,
    int weights_elem_size,
    int kv_scales_elem_size,
    cudaStream_t stream
) {
    if (!q || !kv_cache || !weights || !context_lens ||
        !logits || !block_table || !schedule_meta || batch_size <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (batch_size > kAotAlignedBatchSize || next_n != kAotNextN ||
        num_heads != kAotNumHeads || head_dim != kAotHeadDim ||
        block_kv != kAotBlockKv || num_sms != kAotNumSms ||
        is_context_lens_2d || is_varlen || indices) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (q_elem_size != 1 || kv_elem_size != 1 ||
        weights_elem_size != static_cast<int>(sizeof(float)) ||
        kv_scales_elem_size != static_cast<int>(sizeof(float))) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const int64_t min_stride = static_cast<int64_t>(block_kv) * (head_dim + 4);
    if (kv_cache_stride_bytes < min_stride || logits_stride % kSplitKv != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const auto tensor_map_q = deep_gemm::make_tma_2d_desc_raw(
        const_cast<void*>(q), q_elem_size, deep_gemm::DgDtype::Float8_e4m3,
        head_dim, batch_size * next_n * num_heads,
        head_dim, kAotBlockQ * num_heads,
        head_dim,
        head_dim);

    const float* kv_cache_scales = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(kv_cache) +
        static_cast<size_t>(block_kv) * head_dim);

    const auto tensor_map_kv = deep_gemm::make_tma_3d_desc_raw(
        const_cast<void*>(kv_cache), kv_elem_size, deep_gemm::DgDtype::Float8_e4m3,
        head_dim, block_kv, num_kv_blocks,
        head_dim, block_kv, 1,
        head_dim,
        static_cast<int>(kv_cache_stride_bytes / kv_elem_size),
        head_dim);

    const int aligned_block_kv = deep_gemm::get_tma_aligned_size(block_kv, kv_scales_elem_size);
    const auto tensor_map_kv_scales = deep_gemm::make_tma_2d_desc_raw(
        const_cast<void*>(static_cast<const void*>(kv_cache_scales)),
        kv_scales_elem_size, deep_gemm::DgDtype::Float,
        aligned_block_kv, num_kv_blocks,
        block_kv, 1,
        static_cast<int>(kv_cache_stride_bytes / kv_scales_elem_size),
        0);

    const auto tensor_map_weights = deep_gemm::make_tma_2d_desc_raw(
        const_cast<void*>(weights), weights_elem_size, deep_gemm::DgDtype::Float,
        num_heads, batch_size * next_n,
        num_heads, kAotBlockQ,
        num_heads,
        0);

    constexpr int smem_size = static_cast<int>(sizeof(
        deep_gemm::layout::MQALogitsSharedStorage<
            kAotNumHeads, kAotHeadDim, /*kIsMXSF=*/false, kAotBlockQ, kSplitKv,
            kNumQStages, kNumKVStages, 3, cutlass::float_e4m3_t, float>));
    static_assert(smem_size <= kSM100SmemCapacity);

    const uint32_t arg_num_q_tokens_total = static_cast<uint32_t>(batch_size * next_n);
    const uint32_t arg_logits_stride = static_cast<uint32_t>(logits_stride);
    const uint32_t arg_block_table_stride = static_cast<uint32_t>(block_table_stride);
    const uint32_t* arg_context_lens = reinterpret_cast<const uint32_t*>(context_lens);
    cutlass::bfloat16_t* arg_logits = static_cast<cutlass::bfloat16_t*>(logits);
    const uint32_t* arg_block_table = reinterpret_cast<const uint32_t*>(block_table);
    const uint32_t* arg_indices = nullptr;
    const uint32_t* arg_schedule = reinterpret_cast<const uint32_t*>(schedule_meta);
    void* args[] = {
        const_cast<uint32_t*>(&arg_num_q_tokens_total),
        const_cast<uint32_t*>(&arg_logits_stride),
        const_cast<uint32_t*>(&arg_block_table_stride),
        &arg_context_lens,
        &arg_logits,
        &arg_block_table,
        &arg_indices,
        &arg_schedule,
        const_cast<CUtensorMap*>(&tensor_map_q),
        const_cast<CUtensorMap*>(&tensor_map_kv_scales),
        const_cast<CUtensorMap*>(&tensor_map_kv),
        const_cast<CUtensorMap*>(&tensor_map_kv_scales),
        const_cast<CUtensorMap*>(&tensor_map_weights),
    };
    return launch_aot(reinterpret_cast<const void*>(kLogitsKernel),
                      dim3(static_cast<unsigned>(num_sms), 1, 1),
                      dim3(static_cast<unsigned>(kNumSpecializedThreads + kNumMathThreads), 1, 1),
                      smem_size, stream, args);
}

// DSA indexer prefill: unpaged (contiguous-KV) FP8 MQA logits.
//
// Contract (from sm100_mqa_logits.cuh + DeepGEMM's fp8_mqa_logits host code):
//  - q_fp8:    [seq_q, 32, 128] fp8 e4m3, contiguous.
//  - k_fp8:    [seq_kv, 128] fp8 e4m3, contiguous (compact, pre-gathered from
//              the paged indexer cache by glm52_indexer_k_gather_cuda).
//  - k_scale:  [seq_kv] f32 per-token scales. The TMA descriptor rounds the
//              inner dim up to align(seq_kv, 4); allocate the buffer padded
//              to a multiple of 4 floats (16 bytes).
//  - weights:  [seq_q, 32] f32, per-token per-head factors with q_scale
//              folded in. Applied in-kernel fused with ReLU:
//              logit = (sum_h max(q_h . k, 0) * w_h) * k_scale[n].
//  - cu_seqlen_ks/ke: [seq_q] i32 ABSOLUTE kv range [ks, ke) per query token
//              (kernel clamps both to seq_kv).
//  - logits:   f32, row stride `logits_stride` elements. Row stride must be
//              1024-byte aligned (logits_stride % 256 == 0) and cover the
//              trailing split overshoot (>= seq_kv + 256). The buffer must
//              have align(seq_q, 4) rows: the tail Q-block pads rows past
//              seq_q by re-processing the last query token and writes them.
//              Column n is the ABSOLUTE kv index; columns inside a scheduled
//              split but outside [ks, ke) hold garbage the caller must mask.
//  - No seq_q/seq_kv multiple-of alignment is required; the scheduler aligns
//    each Q-block's kv base down to 4 tokens internally (16B TMA alignment
//    for the f32 scale loads).
//  - No schedule-metadata pre-pass exists for this kernel (unlike the paged
//    variant); this is a single launch.
CUresult glm52_deepgemm_mqa_logits_unpaged_cuda(
    const unsigned char* q_fp8,
    const unsigned char* k_fp8,
    const float* k_scale,
    const float* weights,
    const int* cu_seqlen_ks,
    const int* cu_seqlen_ke,
    void* logits,
    int seq_q,
    int seq_kv,
    int logits_stride,
    CUstream stream
) {
    if (!q_fp8 || !k_fp8 || !k_scale || !weights ||
        !cu_seqlen_ks || !cu_seqlen_ke || !logits) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (seq_q <= 0 || seq_kv <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // fp32 rows must be 1024-byte aligned (DeepGEMM's stride contract) and
    // long enough for the last 256-wide split to overshoot seq_kv.
    constexpr int kLogitsStrideAlignment = 1024 / static_cast<int>(sizeof(float));
    if (logits_stride % kLogitsStrideAlignment != 0 ||
        logits_stride < seq_kv + kSplitKv) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // TMA descriptor for q: [seq_q * 32, 128] fp8 rows, 128B swizzle; the
    // smem box covers one Q-block ([BLOCK_Q * 32, 128]).
    const auto tensor_map_q = deep_gemm::make_tma_2d_desc_raw(
        const_cast<unsigned char*>(q_fp8), 1, deep_gemm::DgDtype::Float8_e4m3,
        kAotHeadDim, seq_q * kAotNumHeads,
        kAotHeadDim, kAotBlockQ * kAotNumHeads,
        kAotHeadDim,
        kAotHeadDim);

    // TMA descriptor for the compact kv: [seq_kv, 128] fp8 rows, box is one
    // 256-token split. Out-of-bounds rows of a trailing split are zero-filled
    // by TMA.
    const auto tensor_map_kv = deep_gemm::make_tma_2d_desc_raw(
        const_cast<unsigned char*>(k_fp8), 1, deep_gemm::DgDtype::Float8_e4m3,
        kAotHeadDim, seq_kv,
        kAotHeadDim, kSplitKv,
        kAotHeadDim,
        kAotHeadDim);

    // TMA descriptor for kv scales: 1D-as-2D [align(seq_kv, 4)] f32, box of
    // one 256-token split, no swizzle (mirrors DeepGEMM's fp8 branch).
    const auto tensor_map_kv_scales = deep_gemm::make_tma_2d_desc_raw(
        const_cast<float*>(k_scale), static_cast<int>(sizeof(float)),
        deep_gemm::DgDtype::Float,
        deep_gemm::get_tma_aligned_size(seq_kv, static_cast<int>(sizeof(float))), 1,
        kSplitKv, 1,
        0,
        0);

    // TMA descriptor for weights: [seq_q, 32] f32, box is one Q-block.
    const auto tensor_map_weights = deep_gemm::make_tma_2d_desc_raw(
        const_cast<float*>(weights), static_cast<int>(sizeof(float)),
        deep_gemm::DgDtype::Float,
        kAotNumHeads, seq_q,
        kAotNumHeads, kAotBlockQ,
        kAotNumHeads,
        0);

    // Same shared-storage shape as the paged instantiation (identical
    // H/D/BLOCK_Q/SPLIT_KV/stage template arguments).
    constexpr int smem_size = static_cast<int>(sizeof(
        deep_gemm::layout::MQALogitsSharedStorage<
            kAotNumHeads, kAotHeadDim, /*kIsMXSF=*/false, kAotBlockQ, kSplitKv,
            kNumQStages, kNumKVStages, 3, cutlass::float_e4m3_t, float>));
    static_assert(smem_size <= kSM100SmemCapacity);

    const uint32_t arg_num_q_tokens = static_cast<uint32_t>(seq_q);
    const uint32_t arg_num_kv_tokens = static_cast<uint32_t>(seq_kv);
    const uint32_t arg_logits_stride = static_cast<uint32_t>(logits_stride);
    const uint32_t* arg_ks = reinterpret_cast<const uint32_t*>(cu_seqlen_ks);
    const uint32_t* arg_ke = reinterpret_cast<const uint32_t*>(cu_seqlen_ke);
    float* arg_logits = static_cast<float*>(logits);
    void* args[] = {
        const_cast<uint32_t*>(&arg_num_q_tokens),
        const_cast<uint32_t*>(&arg_num_kv_tokens),
        const_cast<uint32_t*>(&arg_logits_stride),
        &arg_ks,
        &arg_ke,
        &arg_logits,
        const_cast<CUtensorMap*>(&tensor_map_q),
        // FP8 leaves the sf_q descriptor slot unused; fill it with the kv
        // scales descriptor exactly like DeepGEMM's host wrapper does.
        const_cast<CUtensorMap*>(&tensor_map_kv_scales),
        const_cast<CUtensorMap*>(&tensor_map_kv),
        const_cast<CUtensorMap*>(&tensor_map_kv_scales),
        const_cast<CUtensorMap*>(&tensor_map_weights),
    };
    // Grid MUST equal the kNumSMs template constant (scheduler grid stride).
    return launch_aot_plain(reinterpret_cast<const void*>(kUnpagedLogitsKernel),
                      dim3(static_cast<unsigned>(kAotNumSms), 1, 1),
                      dim3(static_cast<unsigned>(kNumSpecializedThreads + kNumMathThreads), 1, 1),
                      smem_size, stream, args);
}

} // extern "C"

#else // !GLM52_DEEPGEMM_MQA_SM100F

extern "C" {

CUresult glm52_deepgemm_paged_mqa_metadata_cuda(
    int*, int*, int, int, int, int, bool, bool, const int*, cudaStream_t) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult glm52_deepgemm_paged_mqa_logits_cuda(
    const void*, const void*, int64_t, const void*, const int*, void*,
    const int*, const int*, int*, int, int, int, int, int, int, bool, bool,
    int, int, int, int, int, int, int, cudaStream_t) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult glm52_deepgemm_mqa_logits_unpaged_cuda(
    const unsigned char*, const unsigned char*, const float*, const float*,
    const int*, const int*, void*, int, int, int, CUstream) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

} // extern "C"

#endif // GLM52_DEEPGEMM_MQA_SM100F
