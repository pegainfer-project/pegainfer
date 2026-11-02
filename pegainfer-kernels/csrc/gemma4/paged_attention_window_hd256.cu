// FlashInfer paged prefill at HEAD_DIM=256 with the sliding-window mask
// (Gemma 4 local attention layers), over a bf16 pool. The e4m3 pool has its
// own entry in local_kv_fp8.cu.

#include "../shared/paged_launch.cuh"

extern "C" {

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
