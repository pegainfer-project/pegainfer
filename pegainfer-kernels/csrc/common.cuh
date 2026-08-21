#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

#define WARP_SIZE 32

// Warp-level sum reduction (fp32)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level max reduction (fp32)
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// One request's page row out of a CSR window, validated before any pointer is
// formed. `row_len` receives the window's length so the caller can bound its
// row index.
__device__ __forceinline__ const int* csr_page_row_checked(
    const int* __restrict__ page_indices,
    int page_indices_len,
    const int* __restrict__ page_indptr,
    int token,
    int* row_len) {
    int begin = page_indptr[token];
    int end = page_indptr[token + 1];
    if (begin < 0 || end < begin || end > page_indices_len) __trap();
    *row_len = end - begin;
    return page_indices + begin;
}

// The row a position lands on inside a resident window. The released front is
// subtracted only once both sides are known non-negative.
__device__ __forceinline__ int resident_row_checked(int pos, int page_size, int origin) {
    if (page_size <= 0 || origin < 0) __trap();
    int page_of_pos = pos / page_size;
    if (page_of_pos < origin) __trap();
    return page_of_pos - origin;
}
