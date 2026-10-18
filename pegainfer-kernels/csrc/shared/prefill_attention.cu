#include "common.cuh"
#include <cstdio>
#include <cstdlib>

#define HEAD_DIM 128

// Always-on invariant check (unlike <cassert>, fires under -DNDEBUG too).
#define QK_ASSERT(cond)                                                       \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "%s:%d: QK invariant failed: %s\n",          \
                         __FILE__, __LINE__, #cond);                          \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

// ============================================================================
// Kernel 1: warp-per-token QK RMSNorm + RoPE (in-place on Q and K batches).
//
// One warp owns one (head, token); each lane holds 4 contiguous dims as a
// float2 bf16 load. RMSNorm reduces with a shuffle butterfly, RoPE exchanges
// the rotate-half partner with `__shfl_xor_sync(.., 16)` (lane^16 == dim±half),
// so there is no shared memory and QK_TOKENS_PER_BLOCK warps pack into one
// block. Hard-wired to head_dim == 128 (32 lanes × 4 dims, half == 16 lanes);
// the launcher asserts it.
//
// The kernel is DRAM-latency-bound at ~88% occupancy (ncu, sm_120), so the
// lever is memory-level parallelism per warp, not more warps: all four
// independent loads (q/k, norm_w, cos, sin) are issued UP FRONT so they
// overlap in flight. Writing them in dependency order (as the partner shuffle
// would suggest) lets the `__trap` pos-check fence cos/sin behind the reduce;
// hoisting by hand measured −5% duration / +3pt DRAM SOL on a 5090.
//
// `positions[token]` is each row's absolute RoPE position (the single source
// of truth): a request resuming from a cached prefix has non-contiguous
// positions and still rotates at its true offset.
// ============================================================================
#define QK_TOKENS_PER_BLOCK 4
__global__ void prefill_qk_norm_rope_warp_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    const int* __restrict__ positions,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, int q_dim, int kv_dim,
    float eps, int cos_max_pos
) {
    const int head_global = blockIdx.x;
    const int token = blockIdx.y * QK_TOKENS_PER_BLOCK + threadIdx.y;
    if (token >= seq_len) return;
    const int lane = threadIdx.x;          // 0..31
    const int half = head_dim / 2;         // 64

    const bool is_q = (head_global < num_q_heads);
    const int head_local = is_q ? head_global : (head_global - num_q_heads);
    __nv_bfloat16* data = is_q ? q : k;
    const int dim_stride = is_q ? q_dim : kv_dim;
    const __nv_bfloat16* norm_w = is_q ? q_norm_weight : k_norm_weight;

    const int d0 = lane * 4;
    const int base = token * dim_stride + head_local * head_dim + d0;

    // Issue every independent global load up front (each is one float2/LDG.64;
    // the 4 contiguous dims a lane needs are adjacent in memory).
    __nv_bfloat16 raw_bf[4];
    *reinterpret_cast<float2*>(raw_bf) = *reinterpret_cast<const float2*>(&data[base]);
    __nv_bfloat16 wb[4];
    *reinterpret_cast<float2*>(wb) = *reinterpret_cast<const float2*>(&norm_w[d0]);
    const int pos = __ldg(positions + token);
    if (pos < 0 || pos >= cos_max_pos) __trap();
    const bool low = (d0 < half);
    const int cbase = pos * head_dim + (low ? d0 : d0 - half);
    __nv_bfloat16 cb[4], sb[4];
    *reinterpret_cast<float2*>(cb) = *reinterpret_cast<const float2*>(&cos_cache[cbase]);
    *reinterpret_cast<float2*>(sb) = *reinterpret_cast<const float2*>(&sin_cache[cbase]);

    float val[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) val[r] = __bfloat162float(raw_bf[r]);

    // RMSNorm sum of squares: 4-per-lane, then shuffle butterfly across the warp.
    float sq = 0.f;
    #pragma unroll
    for (int r = 0; r < 4; r++) sq += val[r] * val[r];
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        sq += __shfl_xor_sync(0xffffffff, sq, o);
    const float inv_rms = rsqrtf(sq / head_dim + eps);

    float normed[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        __nv_bfloat16 n = __float2bfloat16(val[r] * inv_rms);
        float nf = __bfloat162float(n) * __bfloat162float(wb[r]);
        normed[r] = __bfloat162float(__float2bfloat16(nf));
    }

    // Rotate-half partner: lane^16 maps d0 <-> d0 ± half.
    float partner[4];
    #pragma unroll
    for (int r = 0; r < 4; r++)
        partner[r] = __shfl_xor_sync(0xffffffff, normed[r], 16);

    __nv_bfloat16 out[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        float c = __bfloat162float(cb[r]);
        float s = __bfloat162float(sb[r]);
        if (low) {
            float lo_cos = __bfloat162float(__float2bfloat16(normed[r] * c));
            float hi_sin = __bfloat162float(__float2bfloat16(partner[r] * s));
            out[r] = __float2bfloat16(lo_cos - hi_sin);
        } else {
            float lo_sin = __bfloat162float(__float2bfloat16(partner[r] * s));
            float hi_cos = __bfloat162float(__float2bfloat16(normed[r] * c));
            out[r] = __float2bfloat16(lo_sin + hi_cos);
        }
    }
    *reinterpret_cast<float2*>(&data[base]) = *reinterpret_cast<float2*>(out);
}

__global__ void dflash_qk_norm_rope_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int q_len,
    int k_len,
    int q_start_pos,
    int k_start_pos,
    float eps,
    int cos_max_pos
) {
    int head_global = blockIdx.x;
    int token = blockIdx.y;
    int d = threadIdx.x;

    bool is_q = (head_global < num_q_heads);
    int local_heads = is_q ? num_q_heads : num_kv_heads;
    int seq_len = is_q ? q_len : k_len;
    if (token >= seq_len) return;

    int head_local = is_q ? head_global : (head_global - num_q_heads);
    if (head_local >= local_heads) return;

    __nv_bfloat16* data = is_q ? q : k;
    int dim_stride = local_heads * head_dim;
    const __nv_bfloat16* norm_w = is_q ? q_norm_weight : k_norm_weight;
    int pos = (is_q ? q_start_pos : k_start_pos) + token;
    if (pos < 0 || pos >= cos_max_pos) __trap();

    int offset = token * dim_stride + head_local * head_dim + d;
    float val = __bfloat162float(data[offset]);

    float sq = warp_reduce_sum(val * val);
    int warp_id = d / WARP_SIZE;
    int lane_id = d % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;  // head_dim / 32: 2 (dim 64) or 4 (dim 128)
    __shared__ float warp_sums[4];
    if (lane_id == 0) warp_sums[warp_id] = sq;
    __syncthreads();

    __shared__ float s_inv_rms;
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        float total = warp_reduce_sum(v);
        if (lane_id == 0) s_inv_rms = rsqrtf(total / head_dim + eps);
    }
    __syncthreads();

    __nv_bfloat16 normed = __float2bfloat16(val * s_inv_rms);
    float normed_f = __bfloat162float(normed) * __bfloat162float(norm_w[d]);

    __shared__ __nv_bfloat16 smem[HEAD_DIM];
    smem[d] = __float2bfloat16(normed_f);
    __syncthreads();

    int half = head_dim / 2;
    __nv_bfloat16 result;
    if (d < half) {
        float lo = __bfloat162float(smem[d]);
        float hi = __bfloat162float(smem[d + half]);
        float c = __bfloat162float(cos_cache[pos * head_dim + d]);
        float s = __bfloat162float(sin_cache[pos * head_dim + d]);
        float lo_cos = __bfloat162float(__float2bfloat16(lo * c));
        float hi_sin = __bfloat162float(__float2bfloat16(hi * s));
        result = __float2bfloat16(lo_cos - hi_sin);
    } else {
        int pair_d = d - half;
        float lo = __bfloat162float(smem[pair_d]);
        float hi = __bfloat162float(smem[d]);
        float c = __bfloat162float(cos_cache[pos * head_dim + pair_d]);
        float s = __bfloat162float(sin_cache[pos * head_dim + pair_d]);
        float lo_sin = __bfloat162float(__float2bfloat16(lo * s));
        float hi_cos = __bfloat162float(__float2bfloat16(hi * c));
        result = __float2bfloat16(lo_sin + hi_cos);
    }

    data[offset] = result;
}

// Plain RoPE (no QK-norm) for EAGLE-3, whose attention has no per-head q/k norm.
__global__ void eagle3_rope_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int q_len,
    int k_len,
    int q_start_pos,
    int k_start_pos,
    int cos_max_pos
) {
    int head_global = blockIdx.x;
    int token = blockIdx.y;
    int d = threadIdx.x;

    bool is_q = (head_global < num_q_heads);
    int local_heads = is_q ? num_q_heads : num_kv_heads;
    int seq_len = is_q ? q_len : k_len;
    if (token >= seq_len) return;

    int head_local = is_q ? head_global : (head_global - num_q_heads);
    if (head_local >= local_heads) return;

    __nv_bfloat16* data = is_q ? q : k;
    int dim_stride = local_heads * head_dim;
    int pos = (is_q ? q_start_pos : k_start_pos) + token;
    if (pos < 0 || pos >= cos_max_pos) __trap();

    int offset = token * dim_stride + head_local * head_dim + d;

    // No QK-norm: the raw value goes straight into the RoPE rotation.
    __shared__ __nv_bfloat16 smem[HEAD_DIM];
    smem[d] = data[offset];
    __syncthreads();

    int half = head_dim / 2;
    __nv_bfloat16 result;
    if (d < half) {
        float lo = __bfloat162float(smem[d]);
        float hi = __bfloat162float(smem[d + half]);
        float c = __bfloat162float(cos_cache[pos * head_dim + d]);
        float s = __bfloat162float(sin_cache[pos * head_dim + d]);
        float lo_cos = __bfloat162float(__float2bfloat16(lo * c));
        float hi_sin = __bfloat162float(__float2bfloat16(hi * s));
        result = __float2bfloat16(lo_cos - hi_sin);
    } else {
        int pair_d = d - half;
        float lo = __bfloat162float(smem[pair_d]);
        float hi = __bfloat162float(smem[d]);
        float c = __bfloat162float(cos_cache[pos * head_dim + pair_d]);
        float s = __bfloat162float(sin_cache[pos * head_dim + pair_d]);
        float lo_sin = __bfloat162float(__float2bfloat16(lo * s));
        float hi_cos = __bfloat162float(__float2bfloat16(hi * c));
        result = __float2bfloat16(lo_sin + hi_cos);
    }

    data[offset] = result;
}

extern "C" {

// ============================================================================
// Batched QK norm + RoPE with per-token positions from a GPU array.
//
// Serves both decode (one token per request) and paged prefill (the plan's
// positions array carries each token's absolute position, so requests
// resuming from a cached prefix rotate at their true offsets).
//
// Q layout: [q_dim, batch_size], K layout: [kv_dim, batch_size]
// Grid: (num_q_heads + num_kv_heads, batch_size), Block: head_dim
// ============================================================================
void qk_norm_rope_batched_decode_cuda(
    __nv_bfloat16* q,                    // [q_dim * batch_size] in-place
    __nv_bfloat16* k,                    // [kv_dim * batch_size] in-place
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    const int* positions,                // [batch_size] per-request positions on GPU
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int batch_size,
    float rms_eps,
    int cos_max_pos,
    cudaStream_t stream
) {
    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;

    // blockIdx.y = batch index; positions[batch_idx] supplies each row's
    // absolute RoPE position (prefix-cache-hit suffixes rotate at their true
    // offset). The warp kernel is hard-wired to head_dim == 128 — the only
    // caller is Qwen3, whose head_dim is 128 under any TP degree (TP shards
    // head count, not head_dim). Anything else is a wiring bug, not a runtime
    // case to silently absorb.
    QK_ASSERT(head_dim == HEAD_DIM);

    int tiles = (batch_size + QK_TOKENS_PER_BLOCK - 1) / QK_TOKENS_PER_BLOCK;
    dim3 grid(num_q_heads + num_kv_heads, tiles);
    dim3 block(WARP_SIZE, QK_TOKENS_PER_BLOCK);
    prefill_qk_norm_rope_warp_kernel<<<grid, block, 0, stream>>>(
        q, k, q_norm_weight, k_norm_weight,
        cos_cache, sin_cache, positions,
        num_q_heads, num_kv_heads, head_dim,
        /*seq_len=*/batch_size, q_dim, kv_dim,
        rms_eps, cos_max_pos
    );
}

int dflash_qk_norm_rope_cuda(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int q_len,
    int k_len,
    int q_start_pos,
    int k_start_pos,
    float rms_eps,
    int cos_max_pos,
    cudaStream_t stream
) {
    // One thread per dim element, whole warps only, and the shared buffers
    // (warp_sums[4], smem[HEAD_DIM]) are sized for head_dim <= 128. Serves
    // Qwen3 DFlash (head_dim 128) and the GLM5.2 DSpark drafter (head_dim 64).
    if (q == nullptr || k == nullptr || q_norm_weight == nullptr ||
        k_norm_weight == nullptr || cos_cache == nullptr || sin_cache == nullptr ||
        num_q_heads <= 0 || num_kv_heads <= 0 ||
        head_dim % WARP_SIZE != 0 || head_dim <= 0 || head_dim > HEAD_DIM ||
        q_len <= 0 || k_len <= 0 || q_start_pos < 0 || k_start_pos < 0 ||
        q_start_pos + q_len > cos_max_pos || k_start_pos + k_len > cos_max_pos) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    dim3 grid(num_q_heads + num_kv_heads, q_len > k_len ? q_len : k_len);
    dflash_qk_norm_rope_kernel<<<grid, head_dim, 0, stream>>>(
        q, k, q_norm_weight, k_norm_weight, cos_cache, sin_cache,
        num_q_heads, num_kv_heads, head_dim, q_len, k_len,
        q_start_pos, k_start_pos, rms_eps, cos_max_pos);
    return static_cast<int>(cudaGetLastError());
}

// Plain RoPE (no QK-norm) for EAGLE-3. Launches eagle3_rope_kernel.
int eagle3_rope_cuda(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int q_len,
    int k_len,
    int q_start_pos,
    int k_start_pos,
    int cos_max_pos,
    cudaStream_t stream
) {
    if (q == nullptr || k == nullptr || cos_cache == nullptr || sin_cache == nullptr ||
        num_q_heads <= 0 || num_kv_heads <= 0 || head_dim != HEAD_DIM ||
        q_len <= 0 || k_len <= 0 || q_start_pos < 0 || k_start_pos < 0 ||
        q_start_pos + q_len > cos_max_pos || k_start_pos + k_len > cos_max_pos) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    dim3 grid(num_q_heads + num_kv_heads, q_len > k_len ? q_len : k_len);
    eagle3_rope_kernel<<<grid, head_dim, 0, stream>>>(
        q, k, cos_cache, sin_cache,
        num_q_heads, num_kv_heads, head_dim, q_len, k_len,
        q_start_pos, k_start_pos, cos_max_pos);
    return static_cast<int>(cudaGetLastError());
}

} // extern "C"
