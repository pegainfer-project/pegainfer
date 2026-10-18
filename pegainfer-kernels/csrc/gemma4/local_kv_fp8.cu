#include "../shared/paged_launch.cuh"

#include <cuda_fp8.h>

extern "C" {

int gemma4_batch_prefill_paged_window_hd256_fp8kv_cuda(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* q_indptr, int32_t* request_indices, int32_t* qo_tile_indices,
    int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr, uint32_t* total_num_rows,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t seq_len, int32_t batch_size,
    int32_t padded_batch_size, int64_t stride_page, float sm_scale,
    int32_t cta_tile_q_override, int32_t window_left, void* stream)
{
  return prefill_paged_launch<256, WindowVariant, __nv_fp8_e4m3>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d, q_indptr,
      request_indices, qo_tile_indices, kv_tile_indices, kv_chunk_size_ptr,
      total_num_rows, num_qo_heads, num_kv_heads, head_dim, page_size,
      seq_len, batch_size, padded_batch_size, stride_page, sm_scale,
      cta_tile_q_override, window_left, stream);
}

} // extern "C"
