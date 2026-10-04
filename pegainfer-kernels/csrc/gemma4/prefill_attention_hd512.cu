// QK-norm + partial RoPE prep for head_dim 512 (Gemma 4 global layers).
// Differs from the hd256 sibling in three ways that are choices, not
// oversights: plain w rather than the 1+w offset, no gate, and no separate
// V input — V is the weightless RMS of the same raw row K reduces, so the
// kernel reuses inv_rms and writes V = x * inv_rms into the pool alongside K.
//
// The rotation is the engine's proportional one: rotate_half pairs
// (d, d + 256) over cos/sin tables 512 wide whose entries past the live
// angles are the identity. `row_width` is the pool row's columns per
// (token, kv head) and `fold_rotary` how many of K's columns rotate.
//
// fold_rotary == 0: K and V are two 512-wide blocks at k_offset_elems and
// v_offset_elems. Otherwise one row of 512 + fold_rotary columns holds
// [K_rot | V_identity | V_rot]: K only where it rotates, V everywhere, and
// K's identity columns not at all, since there K is V times the norm weight
// and that weight rides the query. The column order is `KvFormat::permute`
// in paged_kv.rs, restated here as folded_col; the two must agree.
//
// Positions and page ids are trapped on device — checking either on the
// host would require a D2H synchronization.

#include "common.cuh"
#include "../shared/ffi_guard.cuh"
#include "../shared/qk_prep.cuh"

#define HD512 512
#define HALF_HD512 (HD512 / 2)
#define THREADS_HD512 512
#define NUM_WARPS_HD512 (THREADS_HD512 / WARP_SIZE)
// Tokens one paged-prep block carries, for the reason the hd256 prep gives:
// thread d keeps element d of every token, so each token's squares reduce
// through the tree they always did, while the block count drops eightfold
// and eight loads a thread are in flight.
#define HD512_PREP_TOKENS 8

// Where head column d lands in a folded row's first HD512 columns: the rh
// live pairs' columns first, in order, then the rest. rh == 0 is the
// identity, which is the split format's row.
__device__ __forceinline__ int folded_col(int d, int rh) {
    if (d < rh) return d;
    if (d >= HALF_HD512 && d < HALF_HD512 + rh) return d - HALF_HD512 + rh;
    if (d < HALF_HD512) return d + rh;
    return d;
}

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
    const __nv_bfloat16* __restrict__ q_batch,      // [seq_len, q_stride], Q in the first q_dim
    const __nv_bfloat16* __restrict__ k_batch,      // [seq_len, k_stride], K in the first kv_dim
    int q_stride,                                   // row strides: the projection's own width,
    int k_stride,                                   // or the fused row's when they share one
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
    int n_tokens,                                   // rows of the batch
    int start_pos,                                  // host base position
    int cos_max_pos,                                // rows in cos/sin tables
    int row_width,                                  // pool columns per (token, kv head)
    int fold_rotary,                                // K's rotated columns kept; 0 = split
    float rms_eps,
    int page_size,
    int num_pages,                                  // pool capacity in pages
    int64_t stride_page,
    const int* __restrict__ positions,              // [seq_len] absolute (per-token form)
    const int* __restrict__ page_indptr,            // [seq_len + 1] into page_indices
    const int* __restrict__ page_origins            // [seq_len] window-start pages (per-token form)
) {
    // Token blocks are mapped onto grid.x (limit ~2^31) and the head index
    // onto grid.y so prompts longer than the 65535 grid.y limit still launch.
    int token0 = blockIdx.x * HD512_PREP_TOKENS;
    int head_global = blockIdx.y;
    int d = threadIdx.x;
    int n = min(HD512_PREP_TOKENS, n_tokens - token0);

    bool is_q = head_global < num_q_heads;
    int head_local = is_q ? head_global : (head_global - num_q_heads);
    int q_dim = num_q_heads * HD512;

    const __nv_bfloat16* src = is_q ? q_batch : k_batch;
    int src_stride = is_q ? q_stride : k_stride;
    __nv_bfloat16 x[HD512_PREP_TOKENS];
    #pragma unroll
    for (int t = 0; t < HD512_PREP_TOKENS; t++) {
        x[t] = t < n
            ? src[(int64_t)(token0 + t) * src_stride + head_local * HD512 + d]
            : __float2bfloat16(0.0f);
    }
    const __nv_bfloat16* norm_w = is_q ? q_norm_weight : k_norm_weight;

    int warp_id = d / WARP_SIZE;
    int lane_id = d % WARP_SIZE;
    __shared__ float warp_sums[HD512_PREP_TOKENS][NUM_WARPS_HD512];
    __shared__ float inv_rms[HD512_PREP_TOKENS];
    __shared__ int pos_s[HD512_PREP_TOKENS];
    __shared__ int page_s[HD512_PREP_TOKENS];
    __shared__ __nv_bfloat16 smem[HD512_PREP_TOKENS][HD512];

    #pragma unroll
    for (int t = 0; t < HD512_PREP_TOKENS; t++) {
        float sq = __bfloat162float(x[t]);
        sq *= sq;
        float sq_sum = warp_reduce_sum(sq);
        if (lane_id == 0) warp_sums[t][warp_id] = sq_sum;
    }
    __syncthreads();

    if (d < n) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS_HD512; i++) total += warp_sums[d][i];
        inv_rms[d] = 1.0f / sqrtf(total / HD512 + rms_eps);
        int token = token0 + d;
        int pos = PER_TOKEN_META ? positions[token] : start_pos + token;
        if (pos < 0 || pos >= cos_max_pos) __trap();
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
        }
        pos_s[d] = pos;
        page_s[d] = page_id;
    }
    __syncthreads();

    __nv_bfloat16 w = norm_w[d];
    #pragma unroll
    for (int t = 0; t < HD512_PREP_TOKENS; t++) {
        if (t < n) smem[t][d] = rms_norm_elem_hd512(x[t], inv_rms[t], w);
    }
    __syncthreads();

    // The row's bands: with rh live pairs, head column d is rotated when it
    // is in one. The split format has no live pair here and sends every
    // pair through the table, whose identity beyond the live angles makes
    // the whole head K.
    const int rh = fold_rotary / 2;
    const bool folded = fold_rotary > 0;
    const bool rotated = d < rh || (d >= HALF_HD512 && d < HALF_HD512 + rh);
    const int col = folded_col(d, rh);
    if (!is_q) {
        // V is the K=V fork: the weightless norm of the same raw vector,
        // sharing inv_rms. No RoPE, no weight. A folded row keeps V's
        // rotated columns past the head.
        #pragma unroll
        for (int t = 0; t < HD512_PREP_TOKENS; t++) {
            if (t >= n) break;
            int64_t v_dst = paged_kv_row_offset(
                page_s[t], v_offset_elems, stride_page, page_size, num_kv_heads,
                row_width, pos_s[t], head_local, rotated ? HD512 + col : col);
            kv_data[v_dst] = __float2bfloat16(__bfloat162float(x[t]) * inv_rms[t]);
        }
    }

    const int pair_span = folded ? rh : HALF_HD512;
    if (d < pair_span) {
        const int col_hi = folded_col(d + HALF_HD512, rh);
        #pragma unroll
        for (int t = 0; t < HD512_PREP_TOKENS; t++) {
            if (t >= n) break;
            int pos = pos_s[t];
            __nv_bfloat16 lo = smem[t][d];
            __nv_bfloat16 hi = smem[t][d + HALF_HD512];
            apply_rope_pair(
                lo,
                hi,
                cos_cache[pos * HD512 + d],
                sin_cache[pos * HD512 + d]
            );

            if (is_q) {
                int64_t dst = (int64_t)(token0 + t) * q_dim + head_local * HD512;
                q_batch_out[dst + col] = lo;
                q_batch_out[dst + col_hi] = hi;
            } else {
                int64_t dst = paged_kv_row_offset(
                    page_s[t], k_offset_elems, stride_page, page_size, num_kv_heads,
                    row_width, pos, head_local, 0);
                kv_data[dst + col] = lo;
                kv_data[dst + col_hi] = hi;
            }
        }
    } else if (folded && !rotated && is_q) {
        // An identity column of the folded row: the pool holds V there and
        // K's norm weight rides the query, so the query carries both.
        float ww = __bfloat162float(q_norm_weight[d]) * __bfloat162float(k_norm_weight[d]);
        #pragma unroll
        for (int t = 0; t < HD512_PREP_TOKENS; t++) {
            if (t >= n) break;
            int64_t dst = (int64_t)(token0 + t) * q_dim + head_local * HD512;
            q_batch_out[dst + col] = __float2bfloat16(
                __bfloat162float(x[t]) * inv_rms[t] * ww);
        }
    }
}

extern "C" {

int qk_norm_partial_rope_paged_prefill_hd512_cuda(
    const __nv_bfloat16* q_batch,
    const __nv_bfloat16* k_batch,
    int q_stride,
    int k_stride,
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
    int row_width,
    int fold_rotary,
    float rms_eps,
    int page_size,
    int num_pages,
    int64_t stride_page,
    cudaStream_t stream
) {
    PEGAINFER_FFI_GUARD_BEGIN
    if (fold_rotary < 0 || (fold_rotary & 1) != 0 || fold_rotary > HD512 ||
        row_width != HD512 + fold_rotary) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_paged_prefill_hd512_cuda: row_width must be "
            "512 + fold_rotary, with fold_rotary even and in [0, 512]");
        return -1;
    }
    if (fold_rotary > 0 && k_offset_elems != v_offset_elems) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_paged_prefill_hd512_cuda: a folded row holds "
            "K and V in one block");
        return -1;
    }
    if (q_stride < num_q_heads * HD512 || k_stride < num_kv_heads * HD512) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_paged_prefill_hd512_cuda: a source row stride is "
            "narrower than its projection");
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
    dim3 prep_grid(
        (seq_len + HD512_PREP_TOKENS - 1) / HD512_PREP_TOKENS,
        num_q_heads + num_kv_heads);
    qk_norm_partial_rope_paged_prefill_hd512_kernel<false>
        <<<prep_grid, THREADS_HD512, 0, stream>>>(
        q_batch,
        k_batch,
        q_stride,
        k_stride,
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
        seq_len,
        start_pos,
        cos_max_pos,
        row_width,
        fold_rotary,
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

int qk_norm_partial_rope_paged_decode_hd512_cuda(
    const __nv_bfloat16* q_batch,
    const __nv_bfloat16* k_batch,
    int q_stride,
    int k_stride,
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
    int row_width,
    int fold_rotary,
    float rms_eps,
    int page_size,
    int num_pages,
    int64_t stride_page,
    cudaStream_t stream
) {
    PEGAINFER_FFI_GUARD_BEGIN
    if (fold_rotary < 0 || (fold_rotary & 1) != 0 || fold_rotary > HD512 ||
        row_width != HD512 + fold_rotary) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_paged_decode_hd512_cuda: row_width must be "
            "512 + fold_rotary, with fold_rotary even and in [0, 512]");
        return -1;
    }
    if (fold_rotary > 0 && k_offset_elems != v_offset_elems) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_paged_decode_hd512_cuda: a folded row holds "
            "K and V in one block");
        return -1;
    }
    if (q_stride < num_q_heads * HD512 || k_stride < num_kv_heads * HD512) {
        pegainfer_ffi_set_last_error(
            "qk_norm_partial_rope_paged_decode_hd512_cuda: a source row stride is "
            "narrower than its projection");
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
    dim3 prep_grid(
        (batch + HD512_PREP_TOKENS - 1) / HD512_PREP_TOKENS,
        num_q_heads + num_kv_heads);
    qk_norm_partial_rope_paged_prefill_hd512_kernel<true>
        <<<prep_grid, THREADS_HD512, 0, stream>>>(
        q_batch,
        k_batch,
        q_stride,
        k_stride,
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
        batch,
        0,
        cos_max_pos,
        row_width,
        fold_rotary,
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
