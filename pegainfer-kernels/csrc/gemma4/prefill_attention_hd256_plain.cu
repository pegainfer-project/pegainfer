// QK-norm + RoPE prep at head_dim 256 with the plain-w norm (Gemma 4 local
// layers). The hd256 sibling under csrc/qwen35/ is not this kernel: it
// computes the 1+w offset norm, assumes a gated Q layout twice as wide, and
// ships no v_norm.
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
// the kernel's trap is the only check.

#include "common.cuh"
#include "../shared/ffi_guard.cuh"
#include "../shared/qk_prep.cuh"
#include <cuda_fp8.h>

#define HD256_PLAIN 256
#define THREADS_HD256_PLAIN 256
#define NUM_WARPS_HD256_PLAIN (THREADS_HD256_PLAIN / WARP_SIZE)
// Tokens one paged-prep block carries. A block that held one token spent its
// life waiting on one load and three barriers; eight tokens a block keep
// eight loads in flight per thread and cut the block count eightfold, while
// thread d still holds element d of every token, so each token's squares
// reduce through the very tree they did before.
#define HD256_PREP_TOKENS 8

__device__ __forceinline__ __nv_bfloat16 rms_norm_elem_hd256_plain(
    __nv_bfloat16 x, float rms_inv, __nv_bfloat16 weight) {
    float w = __bfloat162float(weight);
    return __float2bfloat16(__bfloat162float(x) * rms_inv * w);
}

// Paged prep. grid.y carries three bands: [0, num_q_heads) Q,
// then num_kv_heads K, then num_kv_heads V. Q and K are plain-w normed
// and rotated; V is weightless-normed over its own head vector (v_proj
// output — a separate reduction, unlike the hd512 K=V fork) and never
// rotated. K and V write straight into the pool's per-layer K/V blocks.
template <typename KvT>
__device__ __forceinline__ KvT kv_store_cast(__nv_bfloat16 x);
template <>
__device__ __forceinline__ __nv_bfloat16 kv_store_cast(__nv_bfloat16 x) {
    return x;
}
template <>
__device__ __forceinline__ __nv_fp8_e4m3 kv_store_cast(__nv_bfloat16 x) {
    return __nv_fp8_e4m3(__bfloat162float(x));
}

// PER_TOKEN_META = true is the batched-decode form: token t is its own
// request, so its absolute position, its page-table window (page_indices +
// page_indptr[t]) and its released-front origin ride per-token arrays and
// the scalar start_pos/page_origin are ignored.
template <bool PER_TOKEN_META, typename KvT>
__global__ void qkv_norm_rope_paged_prefill_hd256_plain_kernel(
    const __nv_bfloat16* __restrict__ q_batch,      // [seq_len, q_stride], Q in the first q_dim
    const __nv_bfloat16* __restrict__ k_batch,      // [seq_len, k_stride], K in the first kv_dim
    const __nv_bfloat16* __restrict__ v_batch,      // [seq_len, v_stride], V in the first kv_dim
    int q_stride,                                   // row strides: the projection's own width,
    int k_stride,                                   // or the fused row's when they share one
    int v_stride,
    const __nv_bfloat16* __restrict__ q_norm_weight, // [HD256_PLAIN]
    const __nv_bfloat16* __restrict__ k_norm_weight, // [HD256_PLAIN]
    const __nv_bfloat16* __restrict__ cos_cache,    // [max_seq * rotary_dim]
    const __nv_bfloat16* __restrict__ sin_cache,
    __nv_bfloat16* __restrict__ q_batch_out,        // [q_dim, seq_len]
    KvT* __restrict__ kv_data,                      // paged KV pool
    int64_t k_offset_elems,
    int64_t v_offset_elems,
    const int* __restrict__ page_indices,           // resident page row(s)
    int page_indices_len,                           // bound for the CSR window
    int page_origin,                                // absolute page of row[0]
    int num_q_heads,
    int num_kv_heads,
    int n_tokens,                                   // rows of the batch
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
    int token0 = blockIdx.x * HD256_PREP_TOKENS;
    int band = blockIdx.y;
    int d = threadIdx.x;
    int n = min(HD256_PREP_TOKENS, n_tokens - token0);

    bool is_q = band < num_q_heads;
    bool is_k = !is_q && band < num_q_heads + num_kv_heads;
    int head_local = is_q ? band
        : is_k ? band - num_q_heads
               : band - num_q_heads - num_kv_heads;
    int q_dim = num_q_heads * HD256_PLAIN;

    const __nv_bfloat16* src = is_q ? q_batch : is_k ? k_batch : v_batch;
    int src_stride = is_q ? q_stride : is_k ? k_stride : v_stride;
    __nv_bfloat16 x[HD256_PREP_TOKENS];
    #pragma unroll
    for (int t = 0; t < HD256_PREP_TOKENS; t++) {
        x[t] = t < n
            ? src[(int64_t)(token0 + t) * src_stride + head_local * HD256_PLAIN + d]
            : __float2bfloat16(0.0f);
    }

    int warp_id = d / WARP_SIZE;
    int lane_id = d % WARP_SIZE;
    __shared__ float warp_sums[HD256_PREP_TOKENS][NUM_WARPS_HD256_PLAIN];
    __shared__ float inv_rms[HD256_PREP_TOKENS];
    __shared__ int pos_s[HD256_PREP_TOKENS];
    __shared__ int page_s[HD256_PREP_TOKENS];
    __shared__ __nv_bfloat16 smem[HD256_PREP_TOKENS][HD256_PLAIN];

    #pragma unroll
    for (int t = 0; t < HD256_PREP_TOKENS; t++) {
        float sq = __bfloat162float(x[t]);
        sq *= sq;
        float sq_sum = warp_reduce_sum(sq);
        if (lane_id == 0) warp_sums[t][warp_id] = sq_sum;
    }
    __syncthreads();

    // The resident window starts page-aligned, so the in-page offset is
    // position-invariant and only the row index shifts.
    if (d < n) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS_HD256_PLAIN; i++) total += warp_sums[d][i];
        inv_rms[d] = 1.0f / sqrtf(total / HD256_PLAIN + rms_eps);
        int token = token0 + d;
        int pos = PER_TOKEN_META ? positions[token] : start_pos + token;
        if (pos < 0 || pos >= cos_max_pos) __trap();
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
        pos_s[d] = pos;
        page_s[d] = page_id;
    }
    __syncthreads();

    if (!is_q && !is_k) {
        // V band: weightless norm, no RoPE — the whole block exits here.
        #pragma unroll
        for (int t = 0; t < HD256_PREP_TOKENS; t++) {
            if (t < n) {
                int64_t dst = paged_kv_offset<HD256_PLAIN>(
                    page_s[t], v_offset_elems, stride_page, page_size,
                    num_kv_heads, pos_s[t], head_local, d);
                kv_data[dst] = kv_store_cast<KvT>(
                    __float2bfloat16(__bfloat162float(x[t]) * inv_rms[t]));
            }
        }
        return;
    }

    __nv_bfloat16 w = is_q ? q_norm_weight[d] : k_norm_weight[d];
    #pragma unroll
    for (int t = 0; t < HD256_PREP_TOKENS; t++) {
        if (t < n) smem[t][d] = rms_norm_elem_hd256_plain(x[t], inv_rms[t], w);
    }
    __syncthreads();

    int half_rotary = rotary_dim / 2;

    if (d < half_rotary) {
        #pragma unroll
        for (int t = 0; t < HD256_PREP_TOKENS; t++) {
            if (t >= n) break;
            int pos = pos_s[t];
            __nv_bfloat16 lo = smem[t][d];
            __nv_bfloat16 hi = smem[t][d + half_rotary];
            apply_rope_pair(
                lo,
                hi,
                cos_cache[pos * rotary_dim + d],
                sin_cache[pos * rotary_dim + d]
            );

            if (is_q) {
                int64_t dst = (int64_t)(token0 + t) * q_dim + head_local * HD256_PLAIN;
                q_batch_out[dst + d] = lo;
                q_batch_out[dst + d + half_rotary] = hi;
            } else {
                int64_t dst = paged_kv_offset<HD256_PLAIN>(
                    page_s[t], k_offset_elems, stride_page, page_size,
                    num_kv_heads, pos, head_local, d);
                kv_data[dst] = kv_store_cast<KvT>(lo);
                kv_data[dst + half_rotary] = kv_store_cast<KvT>(hi);
            }
        }
    }

    if (d >= rotary_dim) {
        #pragma unroll
        for (int t = 0; t < HD256_PREP_TOKENS; t++) {
            if (t >= n) break;
            if (is_q) {
                int64_t dst = (int64_t)(token0 + t) * q_dim + head_local * HD256_PLAIN;
                q_batch_out[dst + d] = smem[t][d];
            } else {
                int64_t dst = paged_kv_offset<HD256_PLAIN>(
                    page_s[t], k_offset_elems, stride_page, page_size,
                    num_kv_heads, pos_s[t], head_local, d);
                kv_data[dst] = kv_store_cast<KvT>(smem[t][d]);
            }
        }
    }
}

template <typename KvT>
static int qkv_prep_paged_prefill_launch(
    const __nv_bfloat16* q_batch,
    const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch,
    int q_stride,
    int k_stride,
    int v_stride,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    __nv_bfloat16* q_batch_out,
    void* kv_data,
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
    if (q_stride < num_q_heads * HD256_PLAIN ||
        k_stride < num_kv_heads * HD256_PLAIN ||
        v_stride < num_kv_heads * HD256_PLAIN) {
        pegainfer_ffi_set_last_error(
            "qkv_norm_rope_paged_prefill_hd256_plain_cuda: a source row stride is "
            "narrower than its projection");
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
    dim3 prep_grid(
        (seq_len + HD256_PREP_TOKENS - 1) / HD256_PREP_TOKENS,
        num_q_heads + 2 * num_kv_heads);
    qkv_norm_rope_paged_prefill_hd256_plain_kernel<false, KvT>
        <<<prep_grid, THREADS_HD256_PLAIN, 0, stream>>>(
        q_batch,
        k_batch,
        v_batch,
        q_stride,
        k_stride,
        v_stride,
        q_norm_weight,
        k_norm_weight,
        cos_cache,
        sin_cache,
        q_batch_out,
        reinterpret_cast<KvT*>(kv_data),
        k_offset_elems,
        v_offset_elems,
        page_indices,
        page_indices_len,
        page_origin,
        num_q_heads,
        num_kv_heads,
        seq_len,
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

template <typename KvT>
static int qkv_prep_paged_decode_launch(
    const __nv_bfloat16* q_batch,
    const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch,
    int q_stride,
    int k_stride,
    int v_stride,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    __nv_bfloat16* q_batch_out,
    void* kv_data,
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
    if (q_stride < num_q_heads * HD256_PLAIN ||
        k_stride < num_kv_heads * HD256_PLAIN ||
        v_stride < num_kv_heads * HD256_PLAIN) {
        pegainfer_ffi_set_last_error(
            "qkv_norm_rope_paged_decode_hd256_plain_cuda: a source row stride is "
            "narrower than its projection");
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
    dim3 prep_grid(
        (batch + HD256_PREP_TOKENS - 1) / HD256_PREP_TOKENS,
        num_q_heads + 2 * num_kv_heads);
    qkv_norm_rope_paged_prefill_hd256_plain_kernel<true, KvT>
        <<<prep_grid, THREADS_HD256_PLAIN, 0, stream>>>(
        q_batch,
        k_batch,
        v_batch,
        q_stride,
        k_stride,
        v_stride,
        q_norm_weight,
        k_norm_weight,
        cos_cache,
        sin_cache,
        q_batch_out,
        reinterpret_cast<KvT*>(kv_data),
        k_offset_elems,
        v_offset_elems,
        page_indices,
        page_indices_len,
        0,
        num_q_heads,
        num_kv_heads,
        batch,
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

extern "C" {

int qkv_norm_rope_paged_prefill_hd256_plain_cuda(
    const __nv_bfloat16* q_batch, const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch, int q_stride, int k_stride, int v_stride,
    const __nv_bfloat16* q_norm_weight, const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache, const __nv_bfloat16* sin_cache,
    __nv_bfloat16* q_batch_out, __nv_bfloat16* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    const int* page_indices, int page_indices_len, int page_origin,
    int num_q_heads, int num_kv_heads, int seq_len, int start_pos,
    int cos_max_pos, int rotary_dim, float rms_eps,
    int page_size, int num_pages, int64_t stride_page, cudaStream_t stream)
{
    return qkv_prep_paged_prefill_launch<__nv_bfloat16>(
        q_batch, k_batch, v_batch, q_stride, k_stride, v_stride,
        q_norm_weight, k_norm_weight, cos_cache, sin_cache,
        q_batch_out, kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indices_len, page_origin,
        num_q_heads, num_kv_heads, seq_len, start_pos,
        cos_max_pos, rotary_dim, rms_eps, page_size, num_pages,
        stride_page, stream);
}

// E4m3 KV twin.
int qkv_norm_rope_paged_prefill_hd256_plain_fp8kv_cuda(
    const __nv_bfloat16* q_batch, const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch, int q_stride, int k_stride, int v_stride,
    const __nv_bfloat16* q_norm_weight, const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache, const __nv_bfloat16* sin_cache,
    __nv_bfloat16* q_batch_out, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    const int* page_indices, int page_indices_len, int page_origin,
    int num_q_heads, int num_kv_heads, int seq_len, int start_pos,
    int cos_max_pos, int rotary_dim, float rms_eps,
    int page_size, int num_pages, int64_t stride_page, cudaStream_t stream)
{
    return qkv_prep_paged_prefill_launch<__nv_fp8_e4m3>(
        q_batch, k_batch, v_batch, q_stride, k_stride, v_stride,
        q_norm_weight, k_norm_weight, cos_cache, sin_cache,
        q_batch_out, kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indices_len, page_origin,
        num_q_heads, num_kv_heads, seq_len, start_pos,
        cos_max_pos, rotary_dim, rms_eps, page_size, num_pages,
        stride_page, stream);
}

int qkv_norm_rope_paged_decode_hd256_plain_cuda(
    const __nv_bfloat16* q_batch, const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch, int q_stride, int k_stride, int v_stride,
    const __nv_bfloat16* q_norm_weight, const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache, const __nv_bfloat16* sin_cache,
    __nv_bfloat16* q_batch_out, __nv_bfloat16* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    const int* page_indices, int page_indices_len,
    const int* page_indptr, const int* page_origins, const int* positions,
    int num_q_heads, int num_kv_heads, int batch,
    int cos_max_pos, int rotary_dim, float rms_eps,
    int page_size, int num_pages, int64_t stride_page, cudaStream_t stream)
{
    return qkv_prep_paged_decode_launch<__nv_bfloat16>(
        q_batch, k_batch, v_batch, q_stride, k_stride, v_stride,
        q_norm_weight, k_norm_weight, cos_cache, sin_cache,
        q_batch_out, kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indices_len, page_indptr, page_origins,
        positions, num_q_heads, num_kv_heads, batch,
        cos_max_pos, rotary_dim, rms_eps, page_size, num_pages,
        stride_page, stream);
}

// E4m3 KV twin.
int qkv_norm_rope_paged_decode_hd256_plain_fp8kv_cuda(
    const __nv_bfloat16* q_batch, const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch, int q_stride, int k_stride, int v_stride,
    const __nv_bfloat16* q_norm_weight, const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache, const __nv_bfloat16* sin_cache,
    __nv_bfloat16* q_batch_out, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    const int* page_indices, int page_indices_len,
    const int* page_indptr, const int* page_origins, const int* positions,
    int num_q_heads, int num_kv_heads, int batch,
    int cos_max_pos, int rotary_dim, float rms_eps,
    int page_size, int num_pages, int64_t stride_page, cudaStream_t stream)
{
    return qkv_prep_paged_decode_launch<__nv_fp8_e4m3>(
        q_batch, k_batch, v_batch, q_stride, k_stride, v_stride,
        q_norm_weight, k_norm_weight, cos_cache, sin_cache,
        q_batch_out, kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indices_len, page_indptr, page_origins,
        positions, num_q_heads, num_kv_heads, batch,
        cos_max_pos, rotary_dim, rms_eps, page_size, num_pages,
        stride_page, stream);
}

} // extern "C"
