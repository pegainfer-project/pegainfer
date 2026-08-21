// Thin C wrappers around FlashInfer's attention kernels.
//
// We include FlashInfer headers (header-only C++) and instantiate only the
// template variants needed: bf16 Q/KV/O, NHD layout, no RoPE, at HEAD_DIM 128
// and 256, the latter both with and without a sliding-window mask.
//
// FlashInfer's dispatchers internally instantiate multiple GQA group sizes
// (1,2,3,4,8) — this covers both Qwen3-4B (GQA=4) and Qwen3.5-4B (GQA=8).

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

// `window_left` is an inclusive distance: an N-token window passes N - 1, and
// -1 degrades to full attention.
using WindowVariant = DefaultAttention</*custom_mask=*/false,
                                       /*sliding_window=*/true,
                                       /*logits_soft_cap=*/false,
                                       /*alibi=*/false>;

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

// ---------------------------------------------------------------------------
// Paged attention decode — wraps FlashInfer BatchDecodeWithPagedKVCache.
//
// Reads Q from `q`, reads K/V from the paged cache, writes output to `output`.
// No RoPE applied inside (caller does RoPE beforehand).
// No partition-KV (single block per batch×head) for Phase 1 simplicity.
// ---------------------------------------------------------------------------
template <uint32_t HEAD_DIM, typename VariantT>
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
    // Plan metadata (GPU arrays — trivial for non-partition bs=1)
    int32_t* request_indices,      // [padded_batch_size], e.g. [0]
    int32_t* kv_tile_indices,      // [padded_batch_size], e.g. [0]
    int32_t* kv_chunk_size_ptr,    // GPU ptr → 1 int32 (kv_len)
    // Dimensions
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  page_size,
    int32_t  batch_size,           // 1 for Phase 1
    int64_t  stride_page,          // KvLayout.page_stride
    float    sm_scale,             // typically 1/sqrt(head_dim)
    int32_t  window_left,
    // Stream
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    auto paged_kv = make_paged_kv(
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
template <uint32_t HEAD_DIM, typename VariantT>
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
    auto paged_kv = make_paged_kv(
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

extern "C" {

int paged_attention_decode_cuda(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* request_indices, int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t batch_size, int64_t stride_page,
    float sm_scale, void* stream)
{
  return decode_launch</*HEAD_DIM=*/128, Variant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d,
      request_indices, kv_tile_indices, kv_chunk_size_ptr,
      num_qo_heads, num_kv_heads, head_dim, page_size, batch_size,
      stride_page, sm_scale, /*window_left=*/-1, stream);
}

int paged_attention_decode_split_kv_cuda(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* request_indices, int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr,
    int32_t* o_indptr, uint8_t* block_valid_mask,
    void* tmp_v, float* tmp_s,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t batch_size, int32_t padded_batch_size,
    int64_t stride_page, float sm_scale, void* stream)
{
  return decode_split_kv_launch</*HEAD_DIM=*/128, Variant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d,
      request_indices, kv_tile_indices, kv_chunk_size_ptr,
      o_indptr, block_valid_mask, tmp_v, tmp_s,
      num_qo_heads, num_kv_heads, head_dim, page_size,
      batch_size, padded_batch_size, stride_page, sm_scale,
      /*window_left=*/-1, stream);
}

// ---------------------------------------------------------------------------
// Paged KV append — writes one K and one V token per request to paged cache.
//
// Must be called AFTER RMSNorm + RoPE on K, and BEFORE the attention decode.
// V is appended as-is (no norm/RoPE).
// ---------------------------------------------------------------------------
int paged_kv_append_cuda(
    void*    kv_data,
    int64_t  k_offset_elems,
    int64_t  v_offset_elems,
    int32_t* page_indices,
    int32_t* page_indptr,
    int32_t* last_page_len_d,
    void*    key,                  // [batch_size * num_kv_heads * head_dim] bf16
    void*    value,                // [batch_size * num_kv_heads * head_dim] bf16
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  page_size,
    int32_t  batch_size,
    int64_t  stride_page,
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    auto paged_kv = make_paged_kv(
        kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indptr, last_page_len_d,
        num_kv_heads, head_dim, page_size, batch_size, stride_page);

    return static_cast<int>(AppendPagedKVCacheDecode(
        paged_kv,
        reinterpret_cast<DType*>(key),
        reinterpret_cast<DType*>(value),
        reinterpret_cast<cudaStream_t>(stream)));
  PEGAINFER_FFI_GUARD_END(-1)
}

// ---------------------------------------------------------------------------
// Scatter contiguous KV cache into paged layout (one layer at a time).
//
// Source layout (HND per layer): k[head, pos, dim]
//   stride_n = head_dim, stride_h = max_seq_len * head_dim
//
// Called once after prefill to bridge contiguous → paged.
// ---------------------------------------------------------------------------
int paged_kv_scatter_cuda(
    void*    kv_data,
    int64_t  k_offset_elems,
    int64_t  v_offset_elems,
    int32_t* page_indices,
    int32_t* page_indptr,
    int32_t* last_page_len_d,
    void*    src_k,                // contiguous K for this layer [num_kv_heads, max_seq, head_dim]
    void*    src_v,                // contiguous V for this layer [num_kv_heads, max_seq, head_dim]
    int32_t* batch_indices,        // [nnz] = [0, 0, ..., 0]
    int32_t* positions,            // [nnz] = [0, 1, 2, ..., seq_len-1]
    int32_t  nnz,                  // = seq_len
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  page_size,
    int64_t  stride_page,
    int64_t  src_stride_n,         // = head_dim
    int64_t  src_stride_h,         // = max_seq_len * head_dim
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    auto paged_kv = make_paged_kv(
        kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indptr, last_page_len_d,
        num_kv_heads, head_dim, page_size, /*batch_size=*/1, stride_page);

    return static_cast<int>(AppendPagedKVCache(
        paged_kv,
        reinterpret_cast<DType*>(src_k),
        reinterpret_cast<DType*>(src_v),
        batch_indices,
        positions,
        static_cast<uint32_t>(nnz),
        static_cast<size_t>(src_stride_n),
        static_cast<size_t>(src_stride_h),
        static_cast<size_t>(src_stride_n),   // V has same layout as K
        static_cast<size_t>(src_stride_h),
        reinterpret_cast<cudaStream_t>(stream)));
  PEGAINFER_FFI_GUARD_END(-1)
}

// ---------------------------------------------------------------------------
// Batch prefill with paged KV cache — wraps FlashInfer BatchPrefillWithPagedKVCache.
//
// Reads Q from col-major [q_dim, seq_len] layout (= HiddenStates).
// Reads K/V from paged layout (page-first, NHD within each block).
// No RoPE inside (caller does RoPE beforehand via qk_norm_rope_batched_decode_cuda).
// Causal mask, no split-KV (partition_kv=false).
//
// Plan metadata (request_indices, qo_tile_indices, etc.) is pre-computed by Rust
// and passed as GPU arrays. This avoids per-call GPU allocations.
// ---------------------------------------------------------------------------
using BatchPrefillParamsT = BatchPrefillPagedParams<DType, DType, DType, IdType>;

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

// Return the number of Q tiles for given dimensions (needed to size plan arrays).
int32_t batch_prefill_paged_num_tiles(
    int32_t  seq_len,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim)
{
    uint32_t group_size = num_qo_heads / num_kv_heads;
    int64_t packed_qo_len = static_cast<int64_t>(seq_len) * group_size;
    uint32_t cta_tile_q = FA2DetermineCtaTileQ(packed_qo_len, head_dim);
    return static_cast<int32_t>((packed_qo_len + cta_tile_q - 1) / cta_tile_q);
}

int32_t batch_prefill_paged_num_tiles_with_cta_tile_q(
    int32_t  seq_len,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  cta_tile_q_override)
{
    uint32_t group_size = num_qo_heads / num_kv_heads;
    int64_t packed_qo_len = static_cast<int64_t>(seq_len) * group_size;
    uint32_t cta_tile_q = resolve_prefill_cta_tile_q(
        packed_qo_len, head_dim, cta_tile_q_override);
    if (cta_tile_q == 0) {
        pegainfer_ffi_set_last_error("invalid cta_tile_q override");
        return -1;
    }
    return static_cast<int32_t>((packed_qo_len + cta_tile_q - 1) / cta_tile_q);
}

// Return the CTA tile size for batch prefill planning.
// Rust needs this to compute per-request tile counts that are consistent
// with the kernel dispatch.
int32_t batch_prefill_cta_tile_q(
    int32_t  total_seq_len,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim)
{
    uint32_t group_size = num_qo_heads / num_kv_heads;
    int64_t packed_qo_len = static_cast<int64_t>(total_seq_len) * group_size;
    return static_cast<int32_t>(FA2DetermineCtaTileQ(packed_qo_len, head_dim));
}

int32_t batch_prefill_cta_tile_q_with_override(
    int32_t  total_seq_len,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  cta_tile_q_override)
{
    uint32_t group_size = num_qo_heads / num_kv_heads;
    int64_t packed_qo_len = static_cast<int64_t>(total_seq_len) * group_size;
    return static_cast<int32_t>(resolve_prefill_cta_tile_q(
        packed_qo_len, head_dim, cta_tile_q_override));
}

} // extern "C"

template <uint32_t HEAD_DIM, typename VariantT>
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
        window_left,
        /*logits_soft_cap=*/0.0f,
        sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e6f);

    params.request_indices   = request_indices;
    params.qo_tile_indices   = qo_tile_indices;
    params.kv_tile_indices   = kv_tile_indices;
    params.merge_indptr      = nullptr;
    params.o_indptr          = q_indptr;  // same as q_indptr for non-partition: [0, seq_len]
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

extern "C" {

int batch_prefill_paged_cuda_with_cta_tile_q(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* q_indptr, int32_t* request_indices, int32_t* qo_tile_indices,
    int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr, uint32_t* total_num_rows,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t seq_len, int32_t batch_size,
    int32_t padded_batch_size, int64_t stride_page, float sm_scale,
    int32_t cta_tile_q_override, void* stream)
{
  return prefill_paged_launch</*HEAD_DIM=*/128, Variant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d, q_indptr,
      request_indices, qo_tile_indices, kv_tile_indices, kv_chunk_size_ptr,
      total_num_rows, num_qo_heads, num_kv_heads, head_dim, page_size,
      seq_len, batch_size, padded_batch_size, stride_page, sm_scale,
      cta_tile_q_override, /*window_left=*/-1, stream);
}

int batch_prefill_paged_cuda(
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
    return batch_prefill_paged_cuda_with_cta_tile_q(
        q, output, kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indptr, last_page_len_d, q_indptr,
        request_indices, qo_tile_indices, kv_tile_indices,
        kv_chunk_size_ptr, total_num_rows, num_qo_heads, num_kv_heads,
        head_dim, page_size, seq_len, batch_size, padded_batch_size,
        stride_page, sm_scale, /*cta_tile_q_override=*/0, stream);
  PEGAINFER_FFI_GUARD_END(-1)
}

// ---------------------------------------------------------------------------
// Single-request prefill — wraps FlashInfer SinglePrefillWithKVCache.
//
// Reads Q from col-major [q_dim, seq_len] layout (= HiddenStates).
// Reads K/V from contiguous HND cache: k[head, pos, dim].
// No RoPE inside (caller does RoPE beforehand via prefill_attention_prep_cuda).
// Causal mask, no split-KV (tmp=nullptr).
// ---------------------------------------------------------------------------
using PrefillParamsT = SinglePrefillParams<DType, DType, DType>;

int single_prefill_cuda(
    // Q and output (HiddenStates col-major: [q_dim, seq_len])
    void*    q,
    void*    output,
    // Contiguous KV cache (HND per-layer: k[head, pos, dim])
    void*    k_cache,
    void*    v_cache,
    // Dimensions
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  seq_len,          // number of Q tokens (qo_len)
    int32_t  kv_len,           // total KV length (start_pos + seq_len)
    int32_t  max_seq_len,      // allocated cache rows (for HND stride)
    float    sm_scale,
    // Stream
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    // Q/O strides: col-major [q_dim, seq_len]
    uint32_t q_stride_n  = num_qo_heads * head_dim;   // stride between tokens
    uint32_t q_stride_h  = head_dim;                   // stride between heads

    // K/V strides: HND layout k[head, pos, dim]
    uint32_t kv_stride_n = head_dim;                   // stride between positions
    uint32_t kv_stride_h = max_seq_len * head_dim;     // stride between heads

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
            /*HEAD_DIM_QK=*/128,
            /*HEAD_DIM_VO=*/128,
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

int single_prefill_nhd_noncausal_cuda(
    // Q and output (HiddenStates token-major: [seq_len, q_dim])
    void*    q,
    void*    output,
    // Contiguous KV cache (HiddenStates token-major: [max_seq_len, kv_dim])
    void*    k_cache,
    void*    v_cache,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  seq_len,
    int32_t  kv_len,
    int32_t  max_seq_len,
    float    sm_scale,
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    if (q == nullptr || output == nullptr || k_cache == nullptr || v_cache == nullptr ||
        num_qo_heads <= 0 || num_kv_heads <= 0 || head_dim != 128 ||
        seq_len <= 0 || kv_len <= 0 || max_seq_len < kv_len) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    uint32_t q_stride_n  = num_qo_heads * head_dim;
    uint32_t q_stride_h  = head_dim;
    uint32_t kv_stride_n = num_kv_heads * head_dim;
    uint32_t kv_stride_h = head_dim;

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
            /*HEAD_DIM_QK=*/128,
            /*HEAD_DIM_VO=*/128,
            PosEncodingMode::kNone,
            /*USE_FP16_QK_REDUCTION=*/false,
            MaskMode::kNone,
            Variant,
            PrefillParamsT>(
            params,
            /*tmp=*/nullptr,
            reinterpret_cast<cudaStream_t>(stream)));
  PEGAINFER_FFI_GUARD_END(-1)
}

// Identical to single_prefill_nhd_noncausal_cuda but dispatched with
// HEAD_DIM_QK/VO=64 — the GLM5.2 DSpark drafter's block attention (64 MHA
// heads x head_dim 64).
int single_prefill_nhd_noncausal_cuda_hd64(
    void*    q,
    void*    output,
    void*    k_cache,
    void*    v_cache,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  seq_len,
    int32_t  kv_len,
    int32_t  max_seq_len,
    float    sm_scale,
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    if (q == nullptr || output == nullptr || k_cache == nullptr || v_cache == nullptr ||
        num_qo_heads <= 0 || num_kv_heads <= 0 || head_dim != 64 ||
        seq_len <= 0 || kv_len <= 0 || max_seq_len < kv_len) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    uint32_t q_stride_n  = num_qo_heads * head_dim;
    uint32_t q_stride_h  = head_dim;
    uint32_t kv_stride_n = num_kv_heads * head_dim;
    uint32_t kv_stride_h = head_dim;

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
            /*HEAD_DIM_QK=*/64,
            /*HEAD_DIM_VO=*/64,
            PosEncodingMode::kNone,
            /*USE_FP16_QK_REDUCTION=*/false,
            MaskMode::kNone,
            Variant,
            PrefillParamsT>(
            params,
            /*tmp=*/nullptr,
            reinterpret_cast<cudaStream_t>(stream)));
  PEGAINFER_FFI_GUARD_END(-1)
}

// Causal variant of single_prefill_nhd_noncausal_cuda: identical NHD token-major
// layout (q/output [seq, q_dim], k/v [max_seq, kv_dim]) but a causal mask, so the
// N query tokens at the tail of the cache attend only positions <= their own.
// Used for EAGLE-3's teacher-forced prefill in one batched forward (query i at
// absolute position kv_len - seq_len + i, FlashInfer's causal alignment).
int single_prefill_nhd_causal_cuda(
    void*    q,
    void*    output,
    void*    k_cache,
    void*    v_cache,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  seq_len,
    int32_t  kv_len,
    int32_t  max_seq_len,
    float    sm_scale,
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    if (q == nullptr || output == nullptr || k_cache == nullptr || v_cache == nullptr ||
        num_qo_heads <= 0 || num_kv_heads <= 0 || head_dim != 128 ||
        num_qo_heads % num_kv_heads != 0 ||
        seq_len <= 0 || kv_len <= 0 || max_seq_len < kv_len || seq_len > kv_len) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    uint32_t q_stride_n  = num_qo_heads * head_dim;
    uint32_t q_stride_h  = head_dim;
    uint32_t kv_stride_n = num_kv_heads * head_dim;
    uint32_t kv_stride_h = head_dim;

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
            /*HEAD_DIM_QK=*/128,
            /*HEAD_DIM_VO=*/128,
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

// ---------------------------------------------------------------------------
// Single-query DECODE over the EAGLE-3 draft's contiguous NHD KV cache.
//
// The chain drafter advances one token per step, so each draft attention is a
// pure decode: exactly ONE query attends the whole [0, kv_len) prefix. This uses
// FlashInfer's dedicated single-query decode path (GEMV-style over the KV),
// which is *structurally* single-query — SingleDecodeParams::get_qo_len() is
// hard-wired to 1 — so, unlike single_prefill_nhd_noncausal_cuda (a prefill
// template forced to qo_len==1), it cannot be silently misused for a multi-query
// batch (a footgun once the draft chain is batched). Same NHD token-major layout
// as the *_nhd_* prefill pair: q/output [1, q_dim], k/v [max_seq_len, kv_dim].
// No RoPE inside (the caller applies eagle3_rope first).
// ---------------------------------------------------------------------------
using DecodeParamsT = SingleDecodeParams<DType, DType, DType>;

int single_decode_nhd_cuda(
    void*    q,            // [1, q_dim] token-major — the single decode query
    void*    output,       // [1, q_dim]
    void*    k_cache,      // [max_seq_len, kv_dim] NHD (k[pos, head, dim])
    void*    v_cache,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  kv_len,       // positions to attend: [0, kv_len)
    int32_t  max_seq_len,  // allocated cache rows (validation parity with the *_nhd_* pair)
    float    sm_scale,
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    if (q == nullptr || output == nullptr || k_cache == nullptr || v_cache == nullptr ||
        num_qo_heads <= 0 || num_kv_heads <= 0 || head_dim != 128 ||
        num_qo_heads % num_kv_heads != 0 ||
        kv_len <= 0 || max_seq_len < kv_len) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    DecodeParamsT params(
        reinterpret_cast<DType*>(q),
        reinterpret_cast<DType*>(k_cache),
        reinterpret_cast<DType*>(v_cache),
        reinterpret_cast<DType*>(output),
        /*maybe_alibi_slopes=*/nullptr,
        /*seq_len(=kv_len)=*/static_cast<uint32_t>(kv_len),
        static_cast<uint32_t>(num_qo_heads),
        static_cast<uint32_t>(num_kv_heads),
        QKVLayout::kNHD,
        static_cast<uint32_t>(head_dim),
        /*window_left=*/-1,
        /*logits_soft_cap=*/0.0f,
        sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e6f);

    return static_cast<int>(
        SingleDecodeWithKVCacheDispatched<
            /*HEAD_DIM=*/128,
            PosEncodingMode::kNone,
            Variant,
            DecodeParamsT>(
            params,
            /*tmp=*/nullptr,
            reinterpret_cast<cudaStream_t>(stream)));
  PEGAINFER_FFI_GUARD_END(-1)
}

// ---------------------------------------------------------------------------
// Single-request prefill for HEAD_DIM=256 — wraps FlashInfer SinglePrefillWithKVCache.
//
// Identical to single_prefill_cuda but instantiated with HEAD_DIM_QK/VO=256.
// Reads Q from col-major [q_dim, seq_len] (HiddenStates layout).
// Reads K/V from contiguous HND cache: k[head, pos, dim].
// No RoPE inside (caller does QK norm + partial RoPE beforehand).
// Causal mask, no split-KV.
//
// Used by Qwen3.5-4B multi-token prefill.  Single-token decode still routes to
// the Triton AOT path (CUDA-Graph safe) until Phase 2d introduces paged decode.
// ---------------------------------------------------------------------------
int single_prefill_cuda_hd256(
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
    uint32_t q_stride_n  = num_qo_heads * 256;
    uint32_t q_stride_h  = 256;

    uint32_t kv_stride_n = 256;
    uint32_t kv_stride_h = static_cast<uint32_t>(max_seq_len) * 256;

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
        /*head_dim=*/256U,
        /*window_left=*/-1,
        /*logits_soft_cap=*/0.0f,
        sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e6f);

    return static_cast<int>(
        SinglePrefillWithKVCacheDispatched<
            /*HEAD_DIM_QK=*/256,
            /*HEAD_DIM_VO=*/256,
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

// ---------------------------------------------------------------------------
// HEAD_DIM=256 paged entry points.  Qwen3.5-4B uses the full-attention pair;
// the windowed pair applies the sliding-window mask for Gemma 4's local layers.
// ---------------------------------------------------------------------------
int paged_attention_decode_cuda_hd256(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* request_indices, int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t batch_size, int64_t stride_page,
    float sm_scale, void* stream)
{
  return decode_launch</*HEAD_DIM=*/256, Variant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d,
      request_indices, kv_tile_indices, kv_chunk_size_ptr,
      num_qo_heads, num_kv_heads, head_dim, page_size, batch_size,
      stride_page, sm_scale, /*window_left=*/-1, stream);
}

int paged_attention_decode_window_cuda_hd256(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* request_indices, int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t batch_size, int64_t stride_page,
    float sm_scale, int32_t window_left, void* stream)
{
  return decode_launch</*HEAD_DIM=*/256, WindowVariant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d,
      request_indices, kv_tile_indices, kv_chunk_size_ptr,
      num_qo_heads, num_kv_heads, head_dim, page_size, batch_size,
      stride_page, sm_scale, window_left, stream);
}

// Split-KV decode at head_dim 512 — the Gemma global family's decode
// read. The non-partitioned grid is (pseudo-requests, kv heads) CTAs and
// starves the device; chunking the KV brings the grid to occupancy.
int paged_attention_decode_split_kv_cuda_hd512(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* request_indices, int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr,
    int32_t* o_indptr, uint8_t* block_valid_mask,
    void* tmp_v, float* tmp_s,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t batch_size, int32_t padded_batch_size,
    int64_t stride_page, float sm_scale, void* stream)
{
  return decode_split_kv_launch</*HEAD_DIM=*/512, Variant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d,
      request_indices, kv_tile_indices, kv_chunk_size_ptr,
      o_indptr, block_valid_mask, tmp_v, tmp_s,
      num_qo_heads, num_kv_heads, head_dim, page_size,
      batch_size, padded_batch_size, stride_page, sm_scale,
      /*window_left=*/-1, stream);
}

int batch_prefill_paged_cuda_hd256(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* q_indptr, int32_t* request_indices, int32_t* qo_tile_indices,
    int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr, uint32_t* total_num_rows,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t seq_len, int32_t batch_size,
    int32_t padded_batch_size, int64_t stride_page, float sm_scale,
    void* stream)
{
  return prefill_paged_launch</*HEAD_DIM=*/256, Variant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d, q_indptr,
      request_indices, qo_tile_indices, kv_tile_indices, kv_chunk_size_ptr,
      total_num_rows, num_qo_heads, num_kv_heads, head_dim, page_size,
      seq_len, batch_size, padded_batch_size, stride_page, sm_scale,
      /*cta_tile_q_override=*/0, /*window_left=*/-1, stream);
}

int batch_prefill_paged_window_cuda_hd256(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* q_indptr, int32_t* request_indices, int32_t* qo_tile_indices,
    int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr, uint32_t* total_num_rows,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t seq_len, int32_t batch_size,
    int32_t padded_batch_size, int64_t stride_page, float sm_scale,
    int32_t window_left, void* stream)
{
  return prefill_paged_launch</*HEAD_DIM=*/256, WindowVariant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d, q_indptr,
      request_indices, qo_tile_indices, kv_tile_indices, kv_chunk_size_ptr,
      total_num_rows, num_qo_heads, num_kv_heads, head_dim, page_size,
      seq_len, batch_size, padded_batch_size, stride_page, sm_scale,
      /*cta_tile_q_override=*/0, window_left, stream);
}

} // extern "C"
