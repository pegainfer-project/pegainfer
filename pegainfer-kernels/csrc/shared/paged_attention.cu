// Thin C wrappers around FlashInfer's attention kernels.
//
// We include FlashInfer headers (header-only C++) and instantiate only the
// template variants needed: bf16 Q/KV/O, NHD layout, no RoPE, at HEAD_DIM 128
// and 256 — the latter both with and without a sliding-window mask — plus the
// hd512 split-KV decode entry the global family reads through.
//
// FlashInfer's dispatchers internally instantiate multiple GQA group sizes
// (1,2,3,4,8) — this covers both Qwen3-4B (GQA=4) and Qwen3.5-4B (GQA=8).

#include "paged_launch.cuh"

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
  return single_prefill_launch</*HEAD_DIM=*/128, Variant>(
      q, output, k_cache, v_cache, num_qo_heads, num_kv_heads,
      head_dim, seq_len, kv_len, max_seq_len,
      sm_scale, stream);
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
  return single_prefill_launch</*HEAD_DIM=*/256, Variant>(
      q, output, k_cache, v_cache, num_qo_heads, num_kv_heads,
      /*head_dim=*/256, seq_len, kv_len, max_seq_len,
      sm_scale, stream);
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
