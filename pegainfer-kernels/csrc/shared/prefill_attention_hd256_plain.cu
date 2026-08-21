// QK-norm + RoPE prep at head_dim 256 with the plain-w norm (Gemma 4 local
// layers). The hd256 sibling under csrc/qwen35/ is not this kernel: it
// computes the 1+w offset norm, assumes a gated Q layout twice as wide, and
// ships no v_norm.
//
// Two entry points share the math: the contiguous oracle form and the
// paged serving form.
//
// rotary_dim is a runtime argument, checked at the launcher for positive,
// even and <= HD256. Gemma 4 local layers rotate the full head (256); at
// that width the pass-through tail is empty. Evenness is load-bearing: with
// half_rotary floored, an odd value leaves index rotary_dim - 1 written by
// neither branch.
//
// Positions derive from host-known start_pos, so the launcher rejects any
// out-of-range window before launch; the device trap before the cos read is
// a second layer, not the contract. Page ids are device data, so for them
// the paged form's trap is the only check.

#include "common.cuh"
#include "ffi_guard.cuh"

#define HD256_PLAIN 256
#define THREADS_HD256_PLAIN 256
#define NUM_WARPS_HD256_PLAIN (THREADS_HD256_PLAIN / WARP_SIZE)

__device__ __forceinline__ __nv_bfloat16 rms_norm_elem_hd256_plain(
    __nv_bfloat16 x, float rms_inv, __nv_bfloat16 weight) {
    float w = __bfloat162float(weight);
    return __float2bfloat16(__bfloat162float(x) * rms_inv * w);
}

__device__ __forceinline__ void apply_rope_pair_hd256_plain(
    __nv_bfloat16& x0, __nv_bfloat16& x1,
    __nv_bfloat16 cos_val, __nv_bfloat16 sin_val) {
    float fx0 = __bfloat162float(x0);
    float fx1 = __bfloat162float(x1);
    float fc = __bfloat162float(cos_val);
    float fs = __bfloat162float(sin_val);
    x0 = __float2bfloat16(fx0 * fc - fx1 * fs);
    x1 = __float2bfloat16(fx0 * fs + fx1 * fc);
}

__global__ void qk_norm_rope_prefill_hd256_plain_kernel(
    const __nv_bfloat16* __restrict__ q_batch,      // [q_dim, seq_len]
    const __nv_bfloat16* __restrict__ k_batch,      // [kv_dim, seq_len]
    const __nv_bfloat16* __restrict__ q_norm_weight, // [HD256_PLAIN]
    const __nv_bfloat16* __restrict__ k_norm_weight, // [HD256_PLAIN]
    const __nv_bfloat16* __restrict__ cos_cache,    // [max_seq * rotary_dim]
    const __nv_bfloat16* __restrict__ sin_cache,
    __nv_bfloat16* __restrict__ q_batch_out,        // [q_dim, seq_len]
    __nv_bfloat16* __restrict__ k_batch_out,        // [kv_dim, seq_len]
    int num_q_heads,
    int num_kv_heads,
    int start_pos,                                  // host base position
    int cos_max_pos,                                // rows in cos/sin tables
    int rotary_dim,
    float rms_eps
) {
    // seq_len is mapped onto grid.x (limit ~2^31) and the head index onto
    // grid.y so prompts longer than the 65535 grid.y limit still launch.
    int token = blockIdx.x;
    int head_global = blockIdx.y;
    int d = threadIdx.x;

    bool is_q = head_global < num_q_heads;
    int head_local = is_q ? head_global : (head_global - num_q_heads);
    int q_dim = num_q_heads * HD256_PLAIN;
    int kv_dim = num_kv_heads * HD256_PLAIN;

    int src_offset = is_q
        ? token * q_dim + head_local * HD256_PLAIN + d
        : token * kv_dim + head_local * HD256_PLAIN + d;
    __nv_bfloat16 x = is_q ? q_batch[src_offset] : k_batch[src_offset];
    const __nv_bfloat16* norm_w = is_q ? q_norm_weight : k_norm_weight;

    float sq = __bfloat162float(x);
    sq *= sq;
    float sq_sum = warp_reduce_sum(sq);

    int warp_id = d / WARP_SIZE;
    int lane_id = d % WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS_HD256_PLAIN];
    __shared__ float inv_rms;
    __shared__ __nv_bfloat16 smem[HD256_PLAIN];

    if (lane_id == 0) warp_sums[warp_id] = sq_sum;
    __syncthreads();

    if (d == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS_HD256_PLAIN; i++) total += warp_sums[i];
        inv_rms = 1.0f / sqrtf(total / HD256_PLAIN + rms_eps);
    }
    __syncthreads();

    smem[d] = rms_norm_elem_hd256_plain(x, inv_rms, norm_w[d]);
    __syncthreads();

    int pos = start_pos + token;
    // Reject before reading the cos/sin tables.
    if (pos < 0 || pos >= cos_max_pos) __trap();
    int half_rotary = rotary_dim / 2;

    if (d < half_rotary) {
        __nv_bfloat16 lo = smem[d];
        __nv_bfloat16 hi = smem[d + half_rotary];
        apply_rope_pair_hd256_plain(
            lo,
            hi,
            cos_cache[pos * rotary_dim + d],
            sin_cache[pos * rotary_dim + d]
        );

        if (is_q) {
            int dst = token * q_dim + head_local * HD256_PLAIN;
            q_batch_out[dst + d] = lo;
            q_batch_out[dst + d + half_rotary] = hi;
        } else {
            int dst = token * kv_dim + head_local * HD256_PLAIN;
            k_batch_out[dst + d] = lo;
            k_batch_out[dst + d + half_rotary] = hi;
        }
    }

    if (d >= rotary_dim) {
        if (is_q) {
            int dst = token * q_dim + head_local * HD256_PLAIN;
            q_batch_out[dst + d] = smem[d];
        } else {
            int dst = token * kv_dim + head_local * HD256_PLAIN;
            k_batch_out[dst + d] = smem[d];
        }
    }
}

__device__ __forceinline__ int64_t paged_kv_offset_hd256_plain(
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
        + static_cast<int64_t>(offset_in_page) * num_kv_heads * HD256_PLAIN
        + static_cast<int64_t>(kv_head) * HD256_PLAIN
        + d;
}

// Paged serving prep. grid.y carries three bands: [0, num_q_heads) Q,
// then num_kv_heads K, then num_kv_heads V. Q and K are plain-w normed
// and rotated; V is weightless-normed over its own head vector (v_proj
// output — a separate reduction, unlike the hd512 K=V fork) and never
// rotated. K and V write straight into the pool's per-layer K/V blocks.
// PER_TOKEN_META = true is the batched-decode form: token t is its own
// request, so its absolute position, its page-table window (page_indices +
// page_indptr[t]) and its released-front origin ride per-token arrays and
// the scalar start_pos/page_origin are ignored.
template <bool PER_TOKEN_META>
__global__ void qkv_norm_rope_paged_prefill_hd256_plain_kernel(
    const __nv_bfloat16* __restrict__ q_batch,      // [q_dim, seq_len]
    const __nv_bfloat16* __restrict__ k_batch,      // [kv_dim, seq_len]
    const __nv_bfloat16* __restrict__ v_batch,      // [kv_dim, seq_len]
    const __nv_bfloat16* __restrict__ q_norm_weight, // [HD256_PLAIN]
    const __nv_bfloat16* __restrict__ k_norm_weight, // [HD256_PLAIN]
    const __nv_bfloat16* __restrict__ cos_cache,    // [max_seq * rotary_dim]
    const __nv_bfloat16* __restrict__ sin_cache,
    __nv_bfloat16* __restrict__ q_batch_out,        // [q_dim, seq_len]
    __nv_bfloat16* __restrict__ kv_data,            // paged KV pool
    int64_t k_offset_elems,
    int64_t v_offset_elems,
    const int* __restrict__ page_indices,           // resident page row(s)
    int page_indices_len,                           // bound for the CSR window
    int page_origin,                                // absolute page of row[0]
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
    const int* __restrict__ page_origins            // [seq_len] released-front pages
) {
    int token = blockIdx.x;
    int band = blockIdx.y;
    int d = threadIdx.x;

    bool is_q = band < num_q_heads;
    bool is_k = !is_q && band < num_q_heads + num_kv_heads;
    int head_local = is_q ? band
        : is_k ? band - num_q_heads
               : band - num_q_heads - num_kv_heads;
    int q_dim = num_q_heads * HD256_PLAIN;
    int kv_dim = num_kv_heads * HD256_PLAIN;

    int src_offset = is_q
        ? token * q_dim + head_local * HD256_PLAIN + d
        : token * kv_dim + head_local * HD256_PLAIN + d;
    __nv_bfloat16 x = is_q ? q_batch[src_offset]
        : is_k ? k_batch[src_offset]
               : v_batch[src_offset];

    float sq = __bfloat162float(x);
    sq *= sq;
    float sq_sum = warp_reduce_sum(sq);

    int warp_id = d / WARP_SIZE;
    int lane_id = d % WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS_HD256_PLAIN];
    __shared__ float inv_rms;
    __shared__ __nv_bfloat16 smem[HD256_PLAIN];

    if (lane_id == 0) warp_sums[warp_id] = sq_sum;
    __syncthreads();

    if (d == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS_HD256_PLAIN; i++) total += warp_sums[i];
        inv_rms = 1.0f / sqrtf(total / HD256_PLAIN + rms_eps);
    }
    __syncthreads();

    int pos = PER_TOKEN_META ? positions[token] : start_pos + token;
    // Reject before reading the cos/sin tables or the page list.
    if (pos < 0 || pos >= cos_max_pos) __trap();
    // Check the device-resident page id before the first pool write. Q
    // blocks never touch the pool.
    // The resident window starts page-aligned, so the in-page offset is
    // position-invariant and only the row index shifts.
    int page_id = -1;
    if (!is_q) {
        int row_len = page_indices_len;
        const int* pages = page_indices;
        if (PER_TOKEN_META) {
            pages = csr_page_row_checked(
                page_indices, page_indices_len, page_indptr, token, &row_len);
        }
        int origin = PER_TOKEN_META ? page_origins[token] : page_origin;
        int row = resident_row_checked(pos, page_size, origin);
        if (row >= row_len) __trap();
        page_id = pages[row];
        if (page_id < 0 || page_id >= num_pages) __trap();
    }

    if (!is_q && !is_k) {
        // V band: weightless norm, no RoPE — the whole block exits here.
        int64_t dst = paged_kv_offset_hd256_plain(
            page_id, v_offset_elems, stride_page, page_size,
            num_kv_heads, pos, head_local, d);
        kv_data[dst] = __float2bfloat16(__bfloat162float(x) * inv_rms);
        return;
    }

    smem[d] = rms_norm_elem_hd256_plain(
        x, inv_rms, is_q ? q_norm_weight[d] : k_norm_weight[d]);
    __syncthreads();

    int half_rotary = rotary_dim / 2;

    if (d < half_rotary) {
        __nv_bfloat16 lo = smem[d];
        __nv_bfloat16 hi = smem[d + half_rotary];
        apply_rope_pair_hd256_plain(
            lo,
            hi,
            cos_cache[pos * rotary_dim + d],
            sin_cache[pos * rotary_dim + d]
        );

        if (is_q) {
            int dst = token * q_dim + head_local * HD256_PLAIN;
            q_batch_out[dst + d] = lo;
            q_batch_out[dst + d + half_rotary] = hi;
        } else {
            int64_t dst = paged_kv_offset_hd256_plain(
                page_id, k_offset_elems, stride_page, page_size,
                num_kv_heads, pos, head_local, d);
            kv_data[dst] = lo;
            kv_data[dst + half_rotary] = hi;
        }
    }

    if (d >= rotary_dim) {
        if (is_q) {
            int dst = token * q_dim + head_local * HD256_PLAIN;
            q_batch_out[dst + d] = smem[d];
        } else {
            int64_t dst = paged_kv_offset_hd256_plain(
                page_id, k_offset_elems, stride_page, page_size,
                num_kv_heads, pos, head_local, d);
            kv_data[dst] = smem[d];
        }
    }
}

extern "C" {

int qk_norm_rope_prefill_hd256_plain_cuda(
    const __nv_bfloat16* q_batch,
    const __nv_bfloat16* k_batch,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    __nv_bfloat16* q_batch_out,
    __nv_bfloat16* k_batch_out,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int start_pos,
    int cos_max_pos,
    int rotary_dim,
    float rms_eps,
    cudaStream_t stream
) {
    PEGAINFER_FFI_GUARD_BEGIN
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > HD256_PLAIN) {
        pegainfer_ffi_set_last_error(
            "qk_norm_rope_prefill_hd256_plain_cuda: rotary_dim must be "
            "positive, even and <= 256");
        return -1;
    }
    if (q_batch == nullptr || k_batch == nullptr || q_norm_weight == nullptr ||
        k_norm_weight == nullptr || cos_cache == nullptr || sin_cache == nullptr ||
        q_batch_out == nullptr || k_batch_out == nullptr) {
        pegainfer_ffi_set_last_error(
            "qk_norm_rope_prefill_hd256_plain_cuda: null pointer argument");
        return -1;
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0 || seq_len <= 0) {
        pegainfer_ffi_set_last_error(
            "qk_norm_rope_prefill_hd256_plain_cuda: num_q_heads, num_kv_heads "
            "and seq_len must be positive");
        return -1;
    }
    if (start_pos < 0 || start_pos + seq_len > cos_max_pos) {
        pegainfer_ffi_set_last_error(
            "qk_norm_rope_prefill_hd256_plain_cuda: start_pos + seq_len must "
            "be <= cos_max_pos");
        return -1;
    }
    dim3 prep_grid(seq_len, num_q_heads + num_kv_heads);
    qk_norm_rope_prefill_hd256_plain_kernel<<<prep_grid, THREADS_HD256_PLAIN, 0, stream>>>(
        q_batch,
        k_batch,
        q_norm_weight,
        k_norm_weight,
        cos_cache,
        sin_cache,
        q_batch_out,
        k_batch_out,
        num_q_heads,
        num_kv_heads,
        start_pos,
        cos_max_pos,
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

int qkv_norm_rope_paged_prefill_hd256_plain_cuda(
    const __nv_bfloat16* q_batch,
    const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch,
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
    int page_origin,
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
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > HD256_PLAIN) {
        pegainfer_ffi_set_last_error(
            "qkv_norm_rope_paged_prefill_hd256_plain_cuda: rotary_dim must be "
            "positive, even and <= 256");
        return -1;
    }
    if (page_origin < 0 || page_origin * page_size > start_pos) {
        pegainfer_ffi_set_last_error(
            "qkv_norm_rope_paged_prefill_hd256_plain_cuda: page_origin must be "
            ">= 0 and at or before start_pos");
        return -1;
    }
    if (q_batch == nullptr || k_batch == nullptr || v_batch == nullptr ||
        q_norm_weight == nullptr || k_norm_weight == nullptr ||
        cos_cache == nullptr || sin_cache == nullptr ||
        q_batch_out == nullptr || kv_data == nullptr ||
        page_indices == nullptr) {
        pegainfer_ffi_set_last_error(
            "qkv_norm_rope_paged_prefill_hd256_plain_cuda: null pointer argument");
        return -1;
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0 || seq_len <= 0 ||
        page_size <= 0 || num_pages <= 0) {
        pegainfer_ffi_set_last_error(
            "qkv_norm_rope_paged_prefill_hd256_plain_cuda: num_q_heads, "
            "num_kv_heads, seq_len, page_size and num_pages must be positive");
        return -1;
    }
    if (start_pos < 0 || start_pos + seq_len > cos_max_pos) {
        pegainfer_ffi_set_last_error(
            "qkv_norm_rope_paged_prefill_hd256_plain_cuda: start_pos + seq_len "
            "must be <= cos_max_pos");
        return -1;
    }
    dim3 prep_grid(seq_len, num_q_heads + 2 * num_kv_heads);
    qkv_norm_rope_paged_prefill_hd256_plain_kernel<false>
        <<<prep_grid, THREADS_HD256_PLAIN, 0, stream>>>(
        q_batch,
        k_batch,
        v_batch,
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
        page_origin,
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

int qkv_norm_rope_paged_decode_hd256_plain_cuda(
    const __nv_bfloat16* q_batch,
    const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch,
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
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > HD256_PLAIN) {
        pegainfer_ffi_set_last_error(
            "qkv_norm_rope_paged_decode_hd256_plain_cuda: rotary_dim must be "
            "positive, even and <= 256");
        return -1;
    }
    if (q_batch == nullptr || k_batch == nullptr || v_batch == nullptr ||
        q_norm_weight == nullptr || k_norm_weight == nullptr ||
        cos_cache == nullptr || sin_cache == nullptr ||
        q_batch_out == nullptr || kv_data == nullptr ||
        page_indices == nullptr || page_indptr == nullptr ||
        page_origins == nullptr || positions == nullptr) {
        pegainfer_ffi_set_last_error(
            "qkv_norm_rope_paged_decode_hd256_plain_cuda: null pointer argument");
        return -1;
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0 || batch <= 0 ||
        page_size <= 0 || num_pages <= 0 || cos_max_pos <= 0) {
        pegainfer_ffi_set_last_error(
            "qkv_norm_rope_paged_decode_hd256_plain_cuda: num_q_heads, "
            "num_kv_heads, batch, page_size, num_pages and cos_max_pos must "
            "be positive");
        return -1;
    }
    dim3 prep_grid(batch, num_q_heads + 2 * num_kv_heads);
    qkv_norm_rope_paged_prefill_hd256_plain_kernel<true>
        <<<prep_grid, THREADS_HD256_PLAIN, 0, stream>>>(
        q_batch,
        k_batch,
        v_batch,
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
        0,
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
