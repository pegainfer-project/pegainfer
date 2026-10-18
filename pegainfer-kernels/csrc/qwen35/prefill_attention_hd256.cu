#include "common.cuh"

#define HD256 256
#define THREADS_HD256 256
#define NUM_WARPS_HD256 (THREADS_HD256 / WARP_SIZE)

__device__ __forceinline__ __nv_bfloat16 rms_norm_elem_offset_hd256(
    __nv_bfloat16 x, float rms_inv, __nv_bfloat16 weight) {
    float w = 1.0f + __bfloat162float(weight);
    return __float2bfloat16(__bfloat162float(x) * rms_inv * w);
}

__device__ __forceinline__ void apply_rope_pair_hd256(
    __nv_bfloat16& x0, __nv_bfloat16& x1,
    __nv_bfloat16 cos_val, __nv_bfloat16 sin_val) {
    float fx0 = __bfloat162float(x0);
    float fx1 = __bfloat162float(x1);
    float fc = __bfloat162float(cos_val);
    float fs = __bfloat162float(sin_val);
    x0 = __float2bfloat16(fx0 * fc - fx1 * fs);
    x1 = __float2bfloat16(fx0 * fs + fx1 * fc);
}

__device__ __forceinline__ int64_t qwen35_paged_kv_offset_hd256(
    int page_id,
    int64_t layer_offset_elems,
    int64_t stride_page,
    int page_size,
    int num_kv_heads,
    int pos,
    int kv_head,
    int d) {
    int offset_in_page = pos % page_size;
    return static_cast<int64_t>(page_id) * stride_page
        + layer_offset_elems
        + static_cast<int64_t>(offset_in_page) * num_kv_heads * HD256
        + static_cast<int64_t>(kv_head) * HD256
        + d;
}

__global__ void prefill_qk_norm_rope_hd256_paged_kernel(
    const __nv_bfloat16* __restrict__ q_full_batch,  // [q_full_dim, seq_len]
    const __nv_bfloat16* __restrict__ k_batch,       // [kv_dim, seq_len]
    const __nv_bfloat16* __restrict__ q_norm_weight, // [HD256]
    const __nv_bfloat16* __restrict__ k_norm_weight, // [HD256]
    const __nv_bfloat16* __restrict__ cos_cache,     // [max_seq * rotary_dim]
    const __nv_bfloat16* __restrict__ sin_cache,
    __nv_bfloat16* __restrict__ q_batch_out,         // [q_dim, seq_len]
    __nv_bfloat16* __restrict__ kv_data,             // paged KV pool
    int64_t k_offset_elems,
    const int* __restrict__ page_indices,            // request page list
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    const int* __restrict__ start_pos_ptr,           // device pointer to this prefill's base position
    int rotary_dim,
    float rms_eps,
    int page_size,
    int64_t stride_page
) {
    // seq_len is mapped onto grid.x (limit ~2^31) and the head index onto
    // grid.y so prompts longer than the 65535 grid.y limit still launch.
    int start_pos = *start_pos_ptr;
    int token = blockIdx.x;
    int head_global = blockIdx.y;
    int d = threadIdx.x;

    bool is_q = head_global < num_q_heads;
    int head_local = is_q ? head_global : (head_global - num_q_heads);
    int q_full_dim = num_q_heads * HD256 * 2;
    int q_dim = num_q_heads * HD256;
    int kv_dim = num_kv_heads * HD256;

    int src_offset = is_q
        ? token * q_full_dim + head_local * 2 * HD256 + d
        : token * kv_dim + head_local * HD256 + d;
    __nv_bfloat16 x = is_q ? q_full_batch[src_offset] : k_batch[src_offset];
    const __nv_bfloat16* norm_w = is_q ? q_norm_weight : k_norm_weight;

    float sq = __bfloat162float(x);
    sq *= sq;
    float sq_sum = warp_reduce_sum(sq);

    int warp_id = d / WARP_SIZE;
    int lane_id = d % WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS_HD256];
    __shared__ float inv_rms;
    __shared__ __nv_bfloat16 smem[HD256];

    if (lane_id == 0) warp_sums[warp_id] = sq_sum;
    __syncthreads();

    if (d == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS_HD256; i++) total += warp_sums[i];
        inv_rms = 1.0f / sqrtf(total / HD256 + rms_eps);
    }
    __syncthreads();

    smem[d] = rms_norm_elem_offset_hd256(x, inv_rms, norm_w[d]);
    __syncthreads();

    int pos = start_pos + token;
    int half_rotary = rotary_dim / 2;

    if (d < half_rotary) {
        __nv_bfloat16 lo = smem[d];
        __nv_bfloat16 hi = smem[d + half_rotary];
        apply_rope_pair_hd256(
            lo,
            hi,
            cos_cache[pos * rotary_dim + d],
            sin_cache[pos * rotary_dim + d]
        );

        if (is_q) {
            int dst = token * q_dim + head_local * HD256;
            q_batch_out[dst + d] = lo;
            q_batch_out[dst + d + half_rotary] = hi;
        } else {
            int page_id = page_indices[pos / page_size];
            int64_t dst = qwen35_paged_kv_offset_hd256(
                page_id, k_offset_elems, stride_page, page_size,
                num_kv_heads, pos, head_local, d);
            kv_data[dst] = lo;
            kv_data[dst + half_rotary] = hi;
        }
    }

    if (d >= rotary_dim) {
        if (is_q) {
            int dst = token * q_dim + head_local * HD256;
            q_batch_out[dst + d] = smem[d];
        } else {
            int page_id = page_indices[pos / page_size];
            int64_t dst = qwen35_paged_kv_offset_hd256(
                page_id, k_offset_elems, stride_page, page_size,
                num_kv_heads, pos, head_local, d);
            kv_data[dst] = smem[d];
        }
    }
}

__global__ void prefill_v_cache_write_hd256_paged_kernel(
    const __nv_bfloat16* __restrict__ v_batch,  // [kv_dim, seq_len]
    __nv_bfloat16* __restrict__ kv_data,        // paged KV pool
    int64_t v_offset_elems,
    const int* __restrict__ page_indices,       // request page list
    int num_kv_heads,
    int seq_len,
    const int* __restrict__ start_pos_ptr,      // device pointer to this prefill's base position
    int page_size,
    int64_t stride_page
) {
    // seq_len on grid.x, kv-head on grid.y (see prep kernel note).
    int start_pos = *start_pos_ptr;
    int token = blockIdx.x;
    int kv_head = blockIdx.y;
    int d = threadIdx.x;

    int pos = start_pos + token;
    int kv_dim = num_kv_heads * HD256;
    int src = token * kv_dim + kv_head * HD256 + d;
    int page_id = page_indices[pos / page_size];
    int64_t dst = qwen35_paged_kv_offset_hd256(
        page_id, v_offset_elems, stride_page, page_size,
        num_kv_heads, pos, kv_head, d);
    kv_data[dst] = v_batch[src];
}

__global__ void attention_gate_batch_hd256_kernel(
    const __nv_bfloat16* __restrict__ q_full_batch,  // [q_full_dim, seq_len]
    __nv_bfloat16* __restrict__ attn_out,            // [q_dim, seq_len]
    int num_q_heads,
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_dim = num_q_heads * HD256;
    int total = q_dim * seq_len;
    if (idx >= total) return;

    int token = idx / q_dim;
    int q_offset = idx - token * q_dim;
    int q_head = q_offset / HD256;
    int dim = q_offset % HD256;
    int q_full_dim = q_dim * 2;
    int gate_idx = token * q_full_dim + q_head * 2 * HD256 + HD256 + dim;

    float gate = __bfloat162float(q_full_batch[gate_idx]);
    float sig_gate = 1.0f / (1.0f + expf(-gate));
    float out = __bfloat162float(attn_out[idx]);
    attn_out[idx] = __float2bfloat16(out * sig_gate);
}

// Batched decode prep for Qwen3.5 full attention:
// - read Q from interleaved q_full [q, gate]
// - RMSNorm with (1+w) offset
// - apply partial RoPE using per-request positions
// - write prepared Q to compact NHD batch
// - normalize/apply partial RoPE to K in-place
__global__ void qk_norm_partial_rope_batched_decode_hd256_kernel(
    const __nv_bfloat16* __restrict__ q_full_batch,  // [q_full_dim, batch]
    __nv_bfloat16* __restrict__ k_batch,             // [kv_dim, batch] in-place
    const __nv_bfloat16* __restrict__ q_norm_weight, // [HD256]
    const __nv_bfloat16* __restrict__ k_norm_weight, // [HD256]
    const __nv_bfloat16* __restrict__ cos_cache,     // [max_seq * rotary_dim]
    const __nv_bfloat16* __restrict__ sin_cache,
    const int* __restrict__ positions,               // [batch]
    __nv_bfloat16* __restrict__ q_batch_out,         // [q_dim, batch]
    int num_q_heads,
    int num_kv_heads,
    int batch_size,
    int rotary_dim,
    float rms_eps
) {
    int head_global = blockIdx.x;
    int token = blockIdx.y;
    int d = threadIdx.x;
    if (token >= batch_size) return;

    bool is_q = head_global < num_q_heads;
    int head_local = is_q ? head_global : (head_global - num_q_heads);
    int q_dim = num_q_heads * HD256;
    int q_full_dim = q_dim * 2;
    int kv_dim = num_kv_heads * HD256;

    int src_offset = is_q
        ? token * q_full_dim + head_local * 2 * HD256 + d
        : token * kv_dim + head_local * HD256 + d;

    __nv_bfloat16 x = is_q ? q_full_batch[src_offset] : k_batch[src_offset];
    const __nv_bfloat16* norm_w = is_q ? q_norm_weight : k_norm_weight;

    float sq = __bfloat162float(x);
    sq *= sq;
    float sq_sum = warp_reduce_sum(sq);

    int warp_id = d / WARP_SIZE;
    int lane_id = d % WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS_HD256];
    __shared__ float inv_rms;
    __shared__ __nv_bfloat16 smem[HD256];

    if (lane_id == 0) warp_sums[warp_id] = sq_sum;
    __syncthreads();

    if (d == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS_HD256; i++) total += warp_sums[i];
        inv_rms = 1.0f / sqrtf(total / HD256 + rms_eps);
    }
    __syncthreads();

    smem[d] = rms_norm_elem_offset_hd256(x, inv_rms, norm_w[d]);
    __syncthreads();

    int pos = positions[token];
    int half_rotary = rotary_dim / 2;

    if (d < half_rotary) {
        __nv_bfloat16 lo = smem[d];
        __nv_bfloat16 hi = smem[d + half_rotary];
        apply_rope_pair_hd256(
            lo,
            hi,
            cos_cache[pos * rotary_dim + d],
            sin_cache[pos * rotary_dim + d]
        );

        if (is_q) {
            int dst = token * q_dim + head_local * HD256;
            q_batch_out[dst + d] = lo;
            q_batch_out[dst + d + half_rotary] = hi;
        } else {
            int dst = token * kv_dim + head_local * HD256;
            k_batch[dst + d] = lo;
            k_batch[dst + d + half_rotary] = hi;
        }
    }

    if (d >= rotary_dim) {
        if (is_q) {
            int dst = token * q_dim + head_local * HD256;
            q_batch_out[dst + d] = smem[d];
        } else {
            int dst = token * kv_dim + head_local * HD256;
            k_batch[dst + d] = smem[d];
        }
    }
}

extern "C" {

void qk_norm_partial_rope_batched_decode_hd256_cuda(
    const __nv_bfloat16* q_full_batch,
    __nv_bfloat16* k_batch,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    const int* positions,
    __nv_bfloat16* q_batch_out,
    int num_q_heads,
    int num_kv_heads,
    int batch_size,
    int rotary_dim,
    float rms_eps,
    cudaStream_t stream
) {
    dim3 grid(num_q_heads + num_kv_heads, batch_size);
    qk_norm_partial_rope_batched_decode_hd256_kernel<<<grid, THREADS_HD256, 0, stream>>>(
        q_full_batch,
        k_batch,
        q_norm_weight,
        k_norm_weight,
        cos_cache,
        sin_cache,
        positions,
        q_batch_out,
        num_q_heads,
        num_kv_heads,
        batch_size,
        rotary_dim,
        rms_eps
    );
}

void prefill_attention_hd256_prep_paged_cuda(
    const __nv_bfloat16* q_full_batch,
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
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    const int* start_pos_ptr,
    int rotary_dim,
    float rms_eps,
    int page_size,
    int64_t stride_page,
    cudaStream_t stream
) {
    dim3 prep_grid(seq_len, num_q_heads + num_kv_heads);
    prefill_qk_norm_rope_hd256_paged_kernel<<<prep_grid, THREADS_HD256, 0, stream>>>(
        q_full_batch,
        k_batch,
        q_norm_weight,
        k_norm_weight,
        cos_cache,
        sin_cache,
        q_batch_out,
        kv_data,
        k_offset_elems,
        page_indices,
        num_q_heads,
        num_kv_heads,
        seq_len,
        start_pos_ptr,
        rotary_dim,
        rms_eps,
        page_size,
        stride_page
    );

    dim3 v_grid(seq_len, num_kv_heads);
    prefill_v_cache_write_hd256_paged_kernel<<<v_grid, THREADS_HD256, 0, stream>>>(
        v_batch,
        kv_data,
        v_offset_elems,
        page_indices,
        num_kv_heads,
        seq_len,
        start_pos_ptr,
        page_size,
        stride_page
    );
}

void attention_gate_batch_hd256_cuda(
    const __nv_bfloat16* q_full_batch,
    __nv_bfloat16* attn_out,
    int num_q_heads,
    int seq_len,
    cudaStream_t stream
) {
    int total = num_q_heads * HD256 * seq_len;
    int block = 256;
    int grid = (total + block - 1) / block;
    attention_gate_batch_hd256_kernel<<<grid, block, 0, stream>>>(
        q_full_batch,
        attn_out,
        num_q_heads,
        seq_len
    );
}

} // extern "C"
