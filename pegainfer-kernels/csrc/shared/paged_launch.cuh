#pragma once

#include "ffi_guard.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>

using namespace flashinfer;

using DType  = __nv_bfloat16;
using IdType = int32_t;
using PrefillParamsT = SinglePrefillParams<DType, DType, DType>;
using Variant = DefaultAttention</*custom_mask=*/false,
                                 /*sliding_window=*/false,
                                 /*logits_soft_cap=*/false,
                                 /*alibi=*/false>;

// `window_left` is an inclusive distance: an N-token window passes N - 1, and
// -1 degrades to full attention.
using WindowVariant = DefaultAttention</*custom_mask=*/false,
                                       /*sliding_window=*/true,
                                       /*logits_soft_cap=*/false,
                                       /*alibi=*/false>;

template <typename KvT = DType>
static paged_kv_t<KvT, IdType> make_paged_kv(
    void*    kv_data,
    int64_t  k_offset_elems,
    int64_t  v_offset_elems,
    int32_t* page_indices,
    int32_t* page_indptr,
    int32_t* last_page_len_d,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  page_size,
    int32_t  batch_size,
    int64_t  stride_page)
{
    KvT* k_data = reinterpret_cast<KvT*>(kv_data) + k_offset_elems;
    KvT* v_data = reinterpret_cast<KvT*>(kv_data) + v_offset_elems;

    // kv_strides[0] = stride_page, [1] = stride for NHD-n, [2] = stride for NHD-h
    int64_t kv_strides[3] = {
        stride_page,
        static_cast<int64_t>(num_kv_heads) * head_dim,
        static_cast<int64_t>(head_dim),
    };

    return paged_kv_t<KvT, IdType>(
        num_kv_heads, page_size, head_dim, batch_size,
        QKVLayout::kNHD,
        k_data, v_data, kv_strides,
        page_indices, page_indptr, last_page_len_d,
        /*rope_pos_offset=*/nullptr);
}

template <uint32_t HEAD_DIM, typename VariantT, typename KvT = DType>
static int decode_launch(
    // Q and output
    void*    q,                    // [num_qo_heads * head_dim] bf16, device
    void*    output,               // [num_qo_heads * head_dim] bf16, device
    // KV pool buffer (entire pool)
    void*    kv_data,
    int64_t  k_offset_elems,       // element offset: base → layer's K in page 0
    int64_t  v_offset_elems,       // element offset: base → layer's V in page 0
    // Paged KV metadata (GPU arrays)
    int32_t* page_indices,         // [num_pages_this_request]
    int32_t* page_indptr,          // [batch_size + 1]
    int32_t* last_page_len_d,      // [batch_size]
    // Plan metadata (GPU arrays, one slot per padded request)
    int32_t* request_indices,      // [padded_batch_size]
    int32_t* kv_tile_indices,      // [padded_batch_size]
    int32_t* kv_chunk_size_ptr,    // GPU ptr → per-request kv lengths
    // Dimensions
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  page_size,
    int32_t  batch_size,
    int64_t  stride_page,          // KvLayout.page_stride
    float    sm_scale,             // typically 1/sqrt(head_dim)
    int32_t  window_left,
    // Stream
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    using ParamsT = BatchDecodeParams<DType, KvT, DType, IdType>;
    auto paged_kv = make_paged_kv<KvT>(
        kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indptr, last_page_len_d,
        num_kv_heads, head_dim, page_size, batch_size, stride_page);

    ParamsT params(
        reinterpret_cast<DType*>(q),
        /*q_rope_offset=*/nullptr,
        paged_kv,
        reinterpret_cast<DType*>(output),
        /*lse=*/nullptr,
        /*maybe_alibi_slopes=*/nullptr,
        num_qo_heads,
        /*q_stride_n=*/num_qo_heads * head_dim,
        /*q_stride_h=*/head_dim,
        window_left,
        /*logits_soft_cap=*/0.0f,
        sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e6f);

    params.padded_batch_size = batch_size;
    params.request_indices   = request_indices;
    params.kv_tile_indices   = kv_tile_indices;
    params.o_indptr          = nullptr;
    params.kv_chunk_size_ptr = kv_chunk_size_ptr;
    params.block_valid_mask  = nullptr;
    params.partition_kv      = false;

    // tmp_v = nullptr → non-partition path (no merge step)
    return static_cast<int>(
        BatchDecodeWithPagedKVCacheDispatched<
            HEAD_DIM,
            PosEncodingMode::kNone,
            VariantT,
            ParamsT>(
            params,
            /*tmp_v=*/nullptr,
            /*tmp_s=*/nullptr,
            /*enable_pdl=*/false,
            reinterpret_cast<cudaStream_t>(stream)));
  PEGAINFER_FFI_GUARD_END(-1)
}

// ---------------------------------------------------------------------------
// Paged attention decode with partition-KV / split-K.
//
// The caller supplies a split-K plan:
//   request_indices[slot]  -> original batch slot
//   kv_tile_indices[slot]  -> KV chunk id for that request
//   o_indptr[request]      -> active split-slot range for merge
//   block_valid_mask[slot] -> false for graph-stability padding slots
//
// tmp_v/tmp_s hold per-chunk partial states and are merged by FlashInfer.
// ---------------------------------------------------------------------------
template <uint32_t HEAD_DIM, typename VariantT, typename KvT = DType>
static int decode_split_kv_launch(
    void*    q,                    // [batch_size * num_qo_heads * head_dim] bf16, device
    void*    output,               // [batch_size * num_qo_heads * head_dim] bf16, device
    void*    kv_data,
    int64_t  k_offset_elems,
    int64_t  v_offset_elems,
    int32_t* page_indices,
    int32_t* page_indptr,
    int32_t* last_page_len_d,
    int32_t* request_indices,
    int32_t* kv_tile_indices,
    int32_t* kv_chunk_size_ptr,    // GPU ptr -> 1 int32 chunk size
    int32_t* o_indptr,             // [batch_size + 1]
    uint8_t* block_valid_mask,     // [padded_batch_size], 0/1
    void*    tmp_v,                // [padded_batch_size * num_qo_heads * head_dim] bf16
    float*   tmp_s,                // [padded_batch_size * num_qo_heads]
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  page_size,
    int32_t  batch_size,
    int32_t  padded_batch_size,    // split slots, including masked padding slots
    int64_t  stride_page,
    float    sm_scale,
    int32_t  window_left,
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    using ParamsT = BatchDecodeParams<DType, KvT, DType, IdType>;
    auto paged_kv = make_paged_kv<KvT>(
        kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indptr, last_page_len_d,
        num_kv_heads, head_dim, page_size, batch_size, stride_page);

    ParamsT params(
        reinterpret_cast<DType*>(q),
        /*q_rope_offset=*/nullptr,
        paged_kv,
        reinterpret_cast<DType*>(output),
        /*lse=*/nullptr,
        /*maybe_alibi_slopes=*/nullptr,
        num_qo_heads,
        /*q_stride_n=*/num_qo_heads * head_dim,
        /*q_stride_h=*/head_dim,
        window_left,
        /*logits_soft_cap=*/0.0f,
        sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e6f);

    params.padded_batch_size = padded_batch_size;
    params.request_indices   = request_indices;
    params.kv_tile_indices   = kv_tile_indices;
    params.o_indptr          = o_indptr;
    params.kv_chunk_size_ptr = kv_chunk_size_ptr;
    params.block_valid_mask  = reinterpret_cast<bool*>(block_valid_mask);

    return static_cast<int>(
        BatchDecodeWithPagedKVCacheDispatched<
            HEAD_DIM,
            PosEncodingMode::kNone,
            VariantT,
            ParamsT>(
            params,
            reinterpret_cast<DType*>(tmp_v),
            tmp_s,
            /*enable_pdl=*/false,
            reinterpret_cast<cudaStream_t>(stream)));
  PEGAINFER_FFI_GUARD_END(-1)
}

static uint32_t resolve_prefill_cta_tile_q(
    int64_t packed_qo_len,
    int32_t head_dim,
    int32_t cta_tile_q_override)
{
    if (cta_tile_q_override == 0) {
        return FA2DetermineCtaTileQ(packed_qo_len, head_dim);
    }
    if (cta_tile_q_override == 16 ||
        cta_tile_q_override == 64 ||
        cta_tile_q_override == 128) {
        return static_cast<uint32_t>(cta_tile_q_override);
    }
    return 0;
}

template <uint32_t HEAD_DIM, typename VariantT, typename KvT = DType>
static int prefill_paged_launch(
    // Q and output (HiddenStates col-major: [q_dim, total_seq_len])
    void*    q,
    void*    output,
    // KV pool buffer (entire pool)
    void*    kv_data,
    int64_t  k_offset_elems,
    int64_t  v_offset_elems,
    // Paged KV metadata (GPU arrays, concatenated across requests)
    int32_t* page_indices,
    int32_t* page_indptr,
    int32_t* last_page_len_d,
    // Batch prefill plan metadata (GPU arrays, pre-allocated by Rust)
    int32_t* q_indptr,             // [batch_size+1]: CSR token boundaries
    int32_t* request_indices,      // [num_tiles]: tile → request mapping
    int32_t* qo_tile_indices,      // [num_tiles]: tile → local Q offset
    int32_t* kv_tile_indices,      // [num_tiles]: all zeros (no KV partition)
    int32_t* kv_chunk_size_ptr,    // [batch_size]: per-request kv_len
    uint32_t* total_num_rows,      // [1]: total Q tokens
    // Dimensions
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  page_size,
    int32_t  seq_len,              // total Q tokens across all requests
    int32_t  batch_size,           // number of requests
    int32_t  padded_batch_size,    // = total num_tiles
    int64_t  stride_page,
    float    sm_scale,
    int32_t  cta_tile_q_override,
    int32_t  window_left,
    // Stream
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    using BatchPrefillParamsT = BatchPrefillPagedParams<DType, KvT, DType, IdType>;
    auto paged_kv = make_paged_kv<KvT>(
        kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indptr, last_page_len_d,
        num_kv_heads, head_dim, page_size, batch_size, stride_page);

    uint32_t q_stride_n = num_qo_heads * head_dim;
    uint32_t q_stride_h = head_dim;

    BatchPrefillParamsT params(
        reinterpret_cast<DType*>(q),
        paged_kv,
        /*maybe_custom_mask=*/nullptr,
        q_indptr,
        /*maybe_mask_indptr=*/nullptr,
        /*maybe_q_rope_offset=*/nullptr,
        reinterpret_cast<DType*>(output),
        /*lse=*/nullptr,
        /*maybe_alibi_slopes=*/nullptr,
        num_qo_heads,
        q_stride_n,
        q_stride_h,
        window_left,
        /*logits_soft_cap=*/0.0f,
        sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e6f);

    params.request_indices   = request_indices;
    params.qo_tile_indices   = qo_tile_indices;
    params.kv_tile_indices   = kv_tile_indices;
    params.merge_indptr      = nullptr;
    params.o_indptr          = q_indptr;  // non-partition: rows land at their request boundaries
    params.block_valid_mask  = nullptr;
    params.kv_chunk_size_ptr = kv_chunk_size_ptr;
    params.max_total_num_rows = seq_len;
    params.total_num_rows    = total_num_rows;
    params.padded_batch_size = padded_batch_size;
    params.partition_kv      = false;

    // Determine CTA tile size and dispatch
    uint32_t group_size = num_qo_heads / num_kv_heads;
    int64_t packed_qo_len = static_cast<int64_t>(seq_len) * group_size;
    uint32_t cta_tile_q = resolve_prefill_cta_tile_q(
        packed_qo_len, head_dim, cta_tile_q_override);
    if (cta_tile_q == 0) {
        pegainfer_ffi_set_last_error("invalid cta_tile_q override");
        return -1;
    }

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int result = 0;
    DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
        result = static_cast<int>(
            BatchPrefillWithPagedKVCacheDispatched<
                CTA_TILE_Q,
                /*HEAD_DIM_QK=*/HEAD_DIM,
                /*HEAD_DIM_VO=*/HEAD_DIM,
                PosEncodingMode::kNone,
                /*USE_FP16_QK_REDUCTION=*/false,
                MaskMode::kCausal,
                VariantT,
                BatchPrefillParamsT>(
                params,
                /*tmp_v=*/nullptr,
                /*tmp_s=*/nullptr,
                /*enable_pdl=*/false,
                s));
    });
    return result;
  PEGAINFER_FFI_GUARD_END(-1)
}

template <uint32_t HEAD_DIM, typename VariantT>
static int single_prefill_launch(
    void* q,
    void* output,
    void* k_cache,
    void* v_cache,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t seq_len,
    int32_t kv_len,
    int32_t max_seq_len,
    float sm_scale,
    void* stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    uint32_t q_stride_n = num_qo_heads * head_dim;
    uint32_t q_stride_h = head_dim;
    uint32_t kv_stride_n = head_dim;
    uint32_t kv_stride_h = static_cast<uint32_t>(max_seq_len) * static_cast<uint32_t>(head_dim);

    PrefillParamsT params(
        reinterpret_cast<DType*>(q),
        reinterpret_cast<DType*>(k_cache),
        reinterpret_cast<DType*>(v_cache),
        /*maybe_custom_mask=*/nullptr,
        reinterpret_cast<DType*>(output),
        /*lse=*/nullptr,
        /*maybe_alibi_slopes=*/nullptr,
        num_qo_heads,
        num_kv_heads,
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(kv_len),
        q_stride_n,
        q_stride_h,
        kv_stride_n,
        kv_stride_h,
        static_cast<uint32_t>(head_dim),
        /*window_left=*/-1,
        /*logits_soft_cap=*/0.0f,
        sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e6f);

    return static_cast<int>(
        SinglePrefillWithKVCacheDispatched<
            HEAD_DIM,
            HEAD_DIM,
            PosEncodingMode::kNone,
            /*USE_FP16_QK_REDUCTION=*/false,
            MaskMode::kCausal,
            VariantT,
            PrefillParamsT>(
            params,
            /*tmp=*/nullptr,
            reinterpret_cast<cudaStream_t>(stream)));
  PEGAINFER_FFI_GUARD_END(-1)
}
