// QK-norm + partial RoPE prep for head_dim 512 (Gemma 4 global layers).
// Differs from the hd256 sibling in three ways that are choices, not
// oversights: plain w rather than the 1+w offset, no gate, and no separate
// V input — V is the weightless RMS of the same raw row K reduces, so the
// kernel reuses inv_rms and writes V = x * inv_rms into the pool's V block
// alongside K.
//
// rotary_dim is a runtime argument, checked at the launcher for positive,
// even and <= HD512. Evenness is load-bearing: with half_rotary floored, an
// odd value leaves index rotary_dim - 1 written by neither branch.

//
// Positions and page ids are trapped on device — checking either on the
// host would require a D2H synchronization.

#include "common.cuh"
#include "ffi_guard.cuh"
#include "qk_prep.cuh"

#define HD512 512
#define THREADS_HD512 512
#define NUM_WARPS_HD512 (THREADS_HD512 / WARP_SIZE)

__device__ __forceinline__ __nv_bfloat16 rms_norm_elem_hd512(
    __nv_bfloat16 x, float rms_inv, __nv_bfloat16 weight) {
    float w = __bfloat162float(weight);
    return __float2bfloat16(__bfloat162float(x) * rms_inv * w);
}

// PER_TOKEN_META = true is the batched-decode form: token t is its own
// request, so its absolute position, its page-table window
// (page_indices + page_indptr[t]) and its window's first absolute page
// (page_origins[t]) ride per-token arrays. The global family never
// front-releases, so callers may compress a row's window to the single
// page holding its position by setting origin = pos / page_size.
template <bool PER_TOKEN_META>
__global__ void qk_norm_partial_rope_paged_prefill_hd512_kernel(
    const __nv_bfloat16* __restrict__ q_batch,      // [q_dim, seq_len]
    const __nv_bfloat16* __restrict__ k_batch,      // [kv_dim, seq_len]
    const __nv_bfloat16* __restrict__ q_norm_weight, // [HD512]
    const __nv_bfloat16* __restrict__ k_norm_weight, // [HD512]
    const __nv_bfloat16* __restrict__ cos_cache,    // [max_seq * rotary_dim]
    const __nv_bfloat16* __restrict__ sin_cache,
    __nv_bfloat16* __restrict__ q_batch_out,        // [q_dim, seq_len]
    __nv_bfloat16* __restrict__ kv_data,            // paged KV pool
    int64_t k_offset_elems,
    int64_t v_offset_elems,
    const int* __restrict__ page_indices,           // request page row(s)
    int page_indices_len,                           // bound for the CSR window
    int num_q_heads,
    int num_kv_heads,
    int start_pos,                                  // host base position
    int cos_max_pos,                                // rows in cos/sin tables
    int rotary_dim,
    float rms_eps,
    int page_size,
    int num_pages,                                  // pool capacity in pages
    int64_t stride_page,
    const int* __restrict__ positions,              // [seq_len] absolute (per-token form)
    const int* __restrict__ page_indptr,            // [seq_len + 1] into page_indices
    const int* __restrict__ page_origins            // [seq_len] window-start pages (per-token form)
) {
    // seq_len is mapped onto grid.x (limit ~2^31) and the head index onto
    // grid.y so prompts longer than the 65535 grid.y limit still launch.
    int token = blockIdx.x;
    int head_global = blockIdx.y;
    int d = threadIdx.x;

    bool is_q = head_global < num_q_heads;
    int head_local = is_q ? head_global : (head_global - num_q_heads);
    int q_dim = num_q_heads * HD512;
    int kv_dim = num_kv_heads * HD512;

    int src_offset = is_q
        ? token * q_dim + head_local * HD512 + d
        : token * kv_dim + head_local * HD512 + d;
    __nv_bfloat16 x = is_q ? q_batch[src_offset] : k_batch[src_offset];
    const __nv_bfloat16* norm_w = is_q ? q_norm_weight : k_norm_weight;

    float sq = __bfloat162float(x);
    sq *= sq;
    float sq_sum = warp_reduce_sum(sq);

    int warp_id = d / WARP_SIZE;
    int lane_id = d % WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS_HD512];
    __shared__ float inv_rms;
    __shared__ __nv_bfloat16 smem[HD512];

    if (lane_id == 0) warp_sums[warp_id] = sq_sum;
    __syncthreads();

    if (d == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS_HD512; i++) total += warp_sums[i];
        inv_rms = 1.0f / sqrtf(total / HD512 + rms_eps);
    }
    __syncthreads();

    smem[d] = rms_norm_elem_hd512(x, inv_rms, norm_w[d]);
    __syncthreads();

    int pos = PER_TOKEN_META ? positions[token] : start_pos + token;
    // Reject before reading the cos/sin tables.
    if (pos < 0 || pos >= cos_max_pos) __trap();
    // Check the device-resident page id before the first pool write. Q
    // threads never touch the pool.
    int page_id = -1;
    if (!is_q) {
        // Only the per-token form needs the window its indptr entry spans.
        int row_len = page_indices_len;
        const int* pages = page_indices;
        if (PER_TOKEN_META) {
            pages = csr_page_row_checked(
                page_indices, page_indices_len, page_indptr, token, &row_len);
        }
        // The global family never releases its front; a per-token origin
        // only compresses the row's window (single page per row).
        int origin = PER_TOKEN_META ? page_origins[token] : 0;
        int row = resident_row_checked(pos, page_size, origin);
        if (row >= row_len) __trap();
        page_id = pages[row];
        if (page_id < 0 || page_id >= num_pages) __trap();
        // V is the K=V fork: the weightless norm of the same raw vector,
        // sharing inv_rms. No RoPE, no weight.
        int64_t v_dst = paged_kv_offset<HD512>(
            page_id, v_offset_elems, stride_page, page_size,
            num_kv_heads, pos, head_local, d);
        kv_data[v_dst] = __float2bfloat16(__bfloat162float(x) * inv_rms);
    }
    int half_rotary = rotary_dim / 2;

    if (d < half_rotary) {
        __nv_bfloat16 lo = smem[d];
        __nv_bfloat16 hi = smem[d + half_rotary];
        apply_rope_pair(
            lo,
            hi,
            cos_cache[pos * rotary_dim + d],
            sin_cache[pos * rotary_dim + d]
        );

        if (is_q) {
            int dst = token * q_dim + head_local * HD512;
            q_batch_out[dst + d] = lo;
            q_batch_out[dst + d + half_rotary] = hi;
        } else {
            int64_t dst = paged_kv_offset<HD512>(
                page_id, k_offset_elems, stride_page, page_size,
                num_kv_heads, pos, head_local, d);
            kv_data[dst] = lo;
            kv_data[dst + half_rotary] = hi;
        }
    }

    if (d >= rotary_dim) {
        if (is_q) {
            int dst = token * q_dim + head_local * HD512;
            q_batch_out[dst + d] = smem[d];
        } else {
            int64_t dst = paged_kv_offset<HD512>(
                page_id, k_offset_elems, stride_page, page_size,
                num_kv_heads, pos, head_local, d);
            kv_data[dst] = smem[d];
        }
    }
}

// Batched decode prep: Q is written to contiguous q_batch_out; K is updated
// in place. Paged scatter is caller-owned.
__global__ void qk_norm_partial_rope_batched_decode_hd512_kernel(
    const __nv_bfloat16* __restrict__ q_batch,      // [q_dim, batch]
    __nv_bfloat16* __restrict__ k_batch,            // [kv_dim, batch] in-place
    const __nv_bfloat16* __restrict__ q_norm_weight, // [HD512]
    const __nv_bfloat16* __restrict__ k_norm_weight, // [HD512]
    const __nv_bfloat16* __restrict__ cos_cache,    // [max_seq * rotary_dim]
    const __nv_bfloat16* __restrict__ sin_cache,
    const int* __restrict__ positions,              // [batch]
    int cos_max_pos,                                // rows in cos/sin tables
    __nv_bfloat16* __restrict__ q_batch_out,        // [q_dim, batch]
    int num_q_heads,
    int num_kv_heads,
    int batch_size,
    int rotary_dim,
    float rms_eps
) {
    int head_global = blockIdx.x;
    int token = blockIdx.y;
    int d = threadIdx.x;

    bool is_q = head_global < num_q_heads;
    int head_local = is_q ? head_global : (head_global - num_q_heads);
    int q_dim = num_q_heads * HD512;
    int kv_dim = num_kv_heads * HD512;

    int src_offset = is_q
        ? token * q_dim + head_local * HD512 + d
        : token * kv_dim + head_local * HD512 + d;

    __nv_bfloat16 x = is_q ? q_batch[src_offset] : k_batch[src_offset];
    const __nv_bfloat16* norm_w = is_q ? q_norm_weight : k_norm_weight;

    float sq = __bfloat162float(x);
    sq *= sq;
    float sq_sum = warp_reduce_sum(sq);

    int warp_id = d / WARP_SIZE;
    int lane_id = d % WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS_HD512];
    __shared__ float inv_rms;
    __shared__ __nv_bfloat16 smem[HD512];

    if (lane_id == 0) warp_sums[warp_id] = sq_sum;
    __syncthreads();

    if (d == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS_HD512; i++) total += warp_sums[i];
        inv_rms = 1.0f / sqrtf(total / HD512 + rms_eps);
    }
    __syncthreads();

    smem[d] = rms_norm_elem_hd512(x, inv_rms, norm_w[d]);
    __syncthreads();

    int pos = positions[token];
    // Reject before reading the cos/sin tables.
    if (pos < 0 || pos >= cos_max_pos) __trap();
    int half_rotary = rotary_dim / 2;

    if (d < half_rotary) {
        __nv_bfloat16 lo = smem[d];
        __nv_bfloat16 hi = smem[d + half_rotary];
        apply_rope_pair(
            lo,
            hi,
            cos_cache[pos * rotary_dim + d],
            sin_cache[pos * rotary_dim + d]
        );

        if (is_q) {
            int dst = token * q_dim + head_local * HD512;
            q_batch_out[dst + d] = lo;
            q_batch_out[dst + d + half_rotary] = hi;
        } else {
            int dst = token * kv_dim + head_local * HD512;
            k_batch[dst + d] = lo;
            k_batch[dst + d + half_rotary] = hi;
        }
    }

    if (d >= rotary_dim) {
        if (is_q) {
            int dst = token * q_dim + head_local * HD512;
            q_batch_out[dst + d] = smem[d];
        } else {
            int dst = token * kv_dim + head_local * HD512;
            k_batch[dst + d] = smem[d];
        }
    }
}

extern "C" {

int qk_norm_partial_rope_paged_prefill_hd512_cuda(
    const __nv_bfloat16* q_batch,
    const __nv_bfloat16* k_batch,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    __nv_bfloat16* q_batch_out,
    __nv_bfloat16* kv_data,
    int64_t k_offset_elems,
    int64_t v_offset_elems,
    const int* page_indices,
    int page_indices_len,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int start_pos,
    int cos_max_pos,
    int rotary_dim,
    float rms_eps,
    int page_size,
    int num_pages,
    int64_t stride_page,
    cudaStream_t stream
) {
    PEGAINFER_FFI_GUARD_BEGIN
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > HD512) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_paged_prefill_hd512_cuda: rotary_dim must be "
            "positive, even and <= 512");
        return -1;
    }
    if (q_batch == nullptr || k_batch == nullptr || q_norm_weight == nullptr ||
        k_norm_weight == nullptr || cos_cache == nullptr || sin_cache == nullptr ||
        q_batch_out == nullptr || kv_data == nullptr || page_indices == nullptr) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_paged_prefill_hd512_cuda: null pointer argument");
        return -1;
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0 || seq_len <= 0 || page_size <= 0 ||
        num_pages <= 0) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_paged_prefill_hd512_cuda: num_q_heads, "
            "num_kv_heads, seq_len, page_size and num_pages must be positive");
        return -1;
    }
    if (start_pos < 0 || start_pos + seq_len > cos_max_pos) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_paged_prefill_hd512_cuda: start_pos + seq_len "
            "must be <= cos_max_pos");
        return -1;
    }
    dim3 prep_grid(seq_len, num_q_heads + num_kv_heads);
    qk_norm_partial_rope_paged_prefill_hd512_kernel<false>
        <<<prep_grid, THREADS_HD512, 0, stream>>>(
        q_batch,
        k_batch,
        q_norm_weight,
        k_norm_weight,
        cos_cache,
        sin_cache,
        q_batch_out,
        kv_data,
        k_offset_elems,
        v_offset_elems,
        page_indices,
        page_indices_len,
        num_q_heads,
        num_kv_heads,
        start_pos,
        cos_max_pos,
        rotary_dim,
        rms_eps,
        page_size,
        num_pages,
        stride_page,
        nullptr,
        nullptr,
        nullptr
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        pegainfer_ffi_set_last_error(cudaGetErrorString(err));
        return -1;
    }
    return 0;
    PEGAINFER_FFI_GUARD_END(-1)
}

int qk_norm_partial_rope_batched_decode_hd512_cuda(
    const __nv_bfloat16* q_batch,
    __nv_bfloat16* k_batch,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    const int* positions,
    int cos_max_pos,
    __nv_bfloat16* q_batch_out,
    int num_q_heads,
    int num_kv_heads,
    int batch_size,
    int rotary_dim,
    float rms_eps,
    cudaStream_t stream
) {
    PEGAINFER_FFI_GUARD_BEGIN
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > HD512) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_batched_decode_hd512_cuda: rotary_dim must be "
            "positive, even and <= 512");
        return -1;
    }
    if (q_batch == nullptr || k_batch == nullptr || q_norm_weight == nullptr ||
        k_norm_weight == nullptr || cos_cache == nullptr || sin_cache == nullptr ||
        positions == nullptr || q_batch_out == nullptr) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_batched_decode_hd512_cuda: null pointer argument");
        return -1;
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0 || batch_size <= 0) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_batched_decode_hd512_cuda: num_q_heads, "
            "num_kv_heads and batch_size must be positive");
        return -1;
    }
    if (cos_max_pos <= 0) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_batched_decode_hd512_cuda: cos_max_pos must be positive");
        return -1;
    }
    dim3 grid(num_q_heads + num_kv_heads, batch_size);
    qk_norm_partial_rope_batched_decode_hd512_kernel<<<grid, THREADS_HD512, 0, stream>>>(
        q_batch,
        k_batch,
        q_norm_weight,
        k_norm_weight,
        cos_cache,
        sin_cache,
        positions,
        cos_max_pos,
        q_batch_out,
        num_q_heads,
        num_kv_heads,
        batch_size,
        rotary_dim,
        rms_eps
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        pegainfer_ffi_set_last_error(cudaGetErrorString(err));
        return -1;
    }
    return 0;
    PEGAINFER_FFI_GUARD_END(-1)
}

int qk_norm_partial_rope_paged_decode_hd512_cuda(
    const __nv_bfloat16* q_batch,
    const __nv_bfloat16* k_batch,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    __nv_bfloat16* q_batch_out,
    __nv_bfloat16* kv_data,
    int64_t k_offset_elems,
    int64_t v_offset_elems,
    const int* page_indices,
    int page_indices_len,
    const int* page_indptr,
    const int* page_origins,
    const int* positions,
    int num_q_heads,
    int num_kv_heads,
    int batch,
    int cos_max_pos,
    int rotary_dim,
    float rms_eps,
    int page_size,
    int num_pages,
    int64_t stride_page,
    cudaStream_t stream
) {
    PEGAINFER_FFI_GUARD_BEGIN
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > HD512) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_paged_decode_hd512_cuda: rotary_dim must be "
            "positive, even and <= 512");
        return -1;
    }
    if (q_batch == nullptr || k_batch == nullptr ||
        q_norm_weight == nullptr || k_norm_weight == nullptr ||
        cos_cache == nullptr || sin_cache == nullptr ||
        q_batch_out == nullptr || kv_data == nullptr ||
        page_indices == nullptr || page_indptr == nullptr ||
        page_origins == nullptr || positions == nullptr) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_paged_decode_hd512_cuda: null pointer argument");
        return -1;
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0 || batch <= 0 ||
        page_size <= 0 || num_pages <= 0 || cos_max_pos <= 0) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_paged_decode_hd512_cuda: num_q_heads, "
            "num_kv_heads, batch, page_size, num_pages and cos_max_pos must "
            "be positive");
        return -1;
    }
    dim3 prep_grid(batch, num_q_heads + num_kv_heads);
    qk_norm_partial_rope_paged_prefill_hd512_kernel<true>
        <<<prep_grid, THREADS_HD512, 0, stream>>>(
        q_batch,
        k_batch,
        q_norm_weight,
        k_norm_weight,
        cos_cache,
        sin_cache,
        q_batch_out,
        kv_data,
        k_offset_elems,
        v_offset_elems,
        page_indices,
        page_indices_len,
        num_q_heads,
        num_kv_heads,
        0,
        cos_max_pos,
        rotary_dim,
        rms_eps,
        page_size,
        num_pages,
        stride_page,
        positions,
        page_indptr,
        page_origins
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        pegainfer_ffi_set_last_error(cudaGetErrorString(err));
        return -1;
    }
    return 0;
    PEGAINFER_FFI_GUARD_END(-1)
}

} // extern "C"
