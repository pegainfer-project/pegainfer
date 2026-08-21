// FlashInfer attention instantiations at HEAD_DIM=512 (Gemma 4 global
// attention layers). Separate TU so the long FlashInfer template builds
// compile in parallel with paged_attention.cu.

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
using ParamsT = BatchDecodeParams<DType, DType, DType, IdType>;
using Variant = DefaultAttention</*custom_mask=*/false,
                                 /*sliding_window=*/false,
                                 /*logits_soft_cap=*/false,
                                 /*alibi=*/false>;

using BatchPrefillParamsT = BatchPrefillPagedParams<DType, DType, DType, IdType>;
using PrefillParamsT = SinglePrefillParams<DType, DType, DType>;

// Helper: build paged_kv_t from our page-first layout.
//
// Our KvPool stores all layers interleaved in one buffer. For a given layer L:
//   k_data = base + L * layer_stride
//   v_data = base + L * layer_stride + kv_block_len
//   stride_page = page_stride  (spans all layers — jumps to same layer in next page)
//   NHD within-block: stride_n = num_kv_heads * head_dim, stride_h = head_dim
static paged_kv_t<DType, IdType> make_paged_kv(
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
    DType* k_data = reinterpret_cast<DType*>(kv_data) + k_offset_elems;
    DType* v_data = reinterpret_cast<DType*>(kv_data) + v_offset_elems;

    // kv_strides[0] = stride_page, [1] = stride for NHD-n, [2] = stride for NHD-h
    int64_t kv_strides[3] = {
        stride_page,
        static_cast<int64_t>(num_kv_heads) * head_dim,
        static_cast<int64_t>(head_dim),
    };

    return paged_kv_t<DType, IdType>(
        num_kv_heads, page_size, head_dim, batch_size,
        QKVLayout::kNHD,
        k_data, v_data, kv_strides,
        page_indices, page_indptr, last_page_len_d,
        /*rope_pos_offset=*/nullptr);
}

extern "C" {

int single_prefill_cuda_hd512(
    void*    q,
    void*    output,
    void*    k_cache,
    void*    v_cache,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  seq_len,          // number of Q tokens (qo_len)
    int32_t  kv_len,           // total KV length (start_pos + seq_len)
    int32_t  max_seq_len,      // allocated cache rows (for HND stride)
    float    sm_scale,
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    uint32_t q_stride_n  = num_qo_heads * 512;
    uint32_t q_stride_h  = 512;

    uint32_t kv_stride_n = 512;
    uint32_t kv_stride_h = static_cast<uint32_t>(max_seq_len) * 512;

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
        /*head_dim=*/512U,
        /*window_left=*/-1,
        /*logits_soft_cap=*/0.0f,
        sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e6f);

    return static_cast<int>(
        SinglePrefillWithKVCacheDispatched<
            /*HEAD_DIM_QK=*/512,
            /*HEAD_DIM_VO=*/512,
            PosEncodingMode::kNone,
            /*USE_FP16_QK_REDUCTION=*/false,
            MaskMode::kCausal,
            Variant,
            PrefillParamsT>(
            params,
            /*tmp=*/nullptr,
            reinterpret_cast<cudaStream_t>(stream)));
  PEGAINFER_FFI_GUARD_END(-1)
}

// Takes no cta_tile_q override parameter because hd512 only ever selects 16|32
// via FA2DetermineCtaTileQ (the shared override helper accepts 16|64|128 only).
int batch_prefill_paged_cuda_hd512(
    void*    q,
    void*    output,
    void*    kv_data,
    int64_t  k_offset_elems,
    int64_t  v_offset_elems,
    int32_t* page_indices,
    int32_t* page_indptr,
    int32_t* last_page_len_d,
    int32_t* q_indptr,
    int32_t* request_indices,
    int32_t* qo_tile_indices,
    int32_t* kv_tile_indices,
    int32_t* kv_chunk_size_ptr,
    uint32_t* total_num_rows,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  page_size,
    int32_t  seq_len,
    int32_t  batch_size,
    int32_t  padded_batch_size,
    int64_t  stride_page,
    float    sm_scale,
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    auto paged_kv = make_paged_kv(
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
        /*window_left=*/-1,
        /*logits_soft_cap=*/0.0f,
        sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e6f);

    params.request_indices   = request_indices;
    params.qo_tile_indices   = qo_tile_indices;
    params.kv_tile_indices   = kv_tile_indices;
    params.merge_indptr      = nullptr;
    params.o_indptr          = q_indptr;
    params.block_valid_mask  = nullptr;
    params.kv_chunk_size_ptr = kv_chunk_size_ptr;
    params.max_total_num_rows = seq_len;
    params.total_num_rows    = total_num_rows;
    params.padded_batch_size = padded_batch_size;
    params.partition_kv      = false;

    uint32_t group_size = num_qo_heads / num_kv_heads;
    int64_t packed_qo_len = static_cast<int64_t>(seq_len) * group_size;
    uint32_t cta_tile_q = FA2DetermineCtaTileQ(packed_qo_len, head_dim);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int result = 0;
    DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
        result = static_cast<int>(
            BatchPrefillWithPagedKVCacheDispatched<
                CTA_TILE_Q,
                /*HEAD_DIM_QK=*/512,
                /*HEAD_DIM_VO=*/512,
                PosEncodingMode::kNone,
                /*USE_FP16_QK_REDUCTION=*/false,
                MaskMode::kCausal,
                Variant,
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

} // extern "C"
