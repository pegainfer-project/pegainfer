#pragma once

#include "common.cuh"

__device__ __forceinline__ void apply_rope_pair(
    __nv_bfloat16& x0,
    __nv_bfloat16& x1,
    __nv_bfloat16 cos_val,
    __nv_bfloat16 sin_val) {
    float fx0 = __bfloat162float(x0);
    float fx1 = __bfloat162float(x1);
    float fc = __bfloat162float(cos_val);
    float fs = __bfloat162float(sin_val);
    x0 = __float2bfloat16(fx0 * fc - fx1 * fs);
    x1 = __float2bfloat16(fx0 * fs + fx1 * fc);
}

template <int HEAD_DIM>
__device__ __forceinline__ int64_t paged_kv_offset(
    int page_id,
    int64_t block_offset_elems,
    int64_t stride_page,
    int page_size,
    int num_kv_heads,
    int pos,
    int kv_head,
    int d) {
    int offset_in_page = pos % page_size;
    return static_cast<int64_t>(page_id) * stride_page
        + block_offset_elems
        + static_cast<int64_t>(offset_in_page) * num_kv_heads * HEAD_DIM
        + static_cast<int64_t>(kv_head) * HEAD_DIM
        + d;
}
