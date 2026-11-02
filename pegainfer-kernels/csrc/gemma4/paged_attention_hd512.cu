// FlashInfer attention instantiations at HEAD_DIM=512 (Gemma 4 global
// attention layers). Separate TU so the long FlashInfer template builds
// compile in parallel with paged_attention.cu.

#include "../shared/paged_launch.cuh"

extern "C" {

// Split-KV decode — the global family's decode read. The non-partitioned
// grid is (pseudo-requests, kv heads) CTAs and starves the device; chunking
// the KV brings the grid to occupancy.
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
  return prefill_paged_launch</*HEAD_DIM=*/512, Variant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d, q_indptr,
      request_indices, qo_tile_indices, kv_tile_indices, kv_chunk_size_ptr,
      total_num_rows, num_qo_heads, num_kv_heads, head_dim, page_size,
      seq_len, batch_size, padded_batch_size, stride_page, sm_scale,
      /*cta_tile_q_override=*/0, /*window_left=*/-1, stream);
}

} // extern "C"
