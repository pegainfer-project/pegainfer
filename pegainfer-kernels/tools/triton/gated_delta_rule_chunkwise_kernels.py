import triton
import triton.language as tl

# Portions of this file adapt chunk-wise Gated Delta Rule kernels from
# Flash Linear Attention (FLA): https://github.com/fla-org/flash-linear-attention
# License: Apache-2.0
#
# Main upstream references:
# - fla/ops/gated_delta_rule/chunk.py
# - fla/ops/common/chunk_delta_h.py
# - fla/ops/common/chunk_o.py
# - fla/ops/gated_delta_rule/wy_fast.py
#
# pegainfer-specific changes:
# - fixed Qwen3.5 dims (batch=1, K=128, V=128, chunk_size=64); head count is a runtime arg
# - Triton AOT-friendly surface and wrapper contracts
# - no backward / varlen / generic autotune surface
# - decode-compatible final-state layout contract [H, V, K]
# - fused prepare stage for q/k expansion, q/k normalization, and g/beta generation


QWEN35_GDR_CHUNK_SIZE = 64
QWEN35_GDR_KEY_DIM = 128
QWEN35_GDR_VALUE_DIM = 128
QWEN35_GDR_KEY_BLOCK = 64


@triton.jit
def gdr_prepare_qkv_gbeta_qwen35_kernel(
    qkv_ptr,         # [seq_len, qkv_dim] bf16, token-major
    b_ptr,           # [seq_len, H] bf16
    a_ptr,           # [seq_len, H] bf16
    dt_bias_ptr,     # [H] bf16
    a_log_ptr,       # [H] fp32
    q_out_ptr,       # [seq_len, H, K] bf16
    k_out_ptr,       # [seq_len, H, K] bf16
    v_out_ptr,       # [seq_len, H, V] bf16
    g_out_ptr,       # [seq_len, H] fp32
    beta_out_ptr,    # [seq_len, H] fp32
    num_key_heads,
    num_value_heads,
    qkv_dim,
    seq_len,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
):
    """Prepare normalized q/k, raw v, and raw g/beta for Qwen3.5 chunk-wise GDR."""
    token_idx = tl.program_id(0)
    v_head = tl.program_id(1)
    if token_idx >= seq_len:
        return

    k_head = (v_head * num_key_heads) // num_value_heads
    qk_dim_total = num_key_heads * KEY_DIM
    v_offset = qk_dim_total * 2 + v_head * VALUE_DIM
    token_qkv = qkv_ptr + token_idx * qkv_dim
    offs_k = tl.arange(0, KEY_DIM)
    offs_v = tl.arange(0, VALUE_DIM)

    q = tl.load(token_qkv + k_head * KEY_DIM + offs_k).to(tl.float32)
    k = tl.load(token_qkv + qk_dim_total + k_head * KEY_DIM + offs_k).to(tl.float32)
    v = tl.load(token_qkv + v_offset + offs_v)

    q *= tl.rsqrt(tl.sum(q * q, axis=0) + 1e-12)
    k *= tl.rsqrt(tl.sum(k * k, axis=0) + 1e-12)

    q_out = q_out_ptr + (token_idx * num_value_heads + v_head) * KEY_DIM + offs_k
    k_out = k_out_ptr + (token_idx * num_value_heads + v_head) * KEY_DIM + offs_k
    v_out = v_out_ptr + (token_idx * num_value_heads + v_head) * VALUE_DIM + offs_v
    tl.store(q_out, q.to(tl.bfloat16))
    tl.store(k_out, k.to(tl.bfloat16))
    tl.store(v_out, v)

    a_val = tl.load(a_ptr + token_idx * num_value_heads + v_head).to(tl.float32)
    b_val = tl.load(b_ptr + token_idx * num_value_heads + v_head).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + v_head).to(tl.float32)
    a_log = tl.load(a_log_ptr + v_head).to(tl.float32)
    x = a_val + dt_bias
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g_val = -tl.exp(a_log) * softplus_x
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    tl.store(g_out_ptr + token_idx * num_value_heads + v_head, g_val)
    tl.store(beta_out_ptr + token_idx * num_value_heads + v_head, beta_val)


@triton.jit
def gdr_prepare_qk_gbeta_qwen35_kernel(
    qkv_ptr,         # [seq_len, qkv_dim] bf16, token-major
    b_ptr,           # [seq_len, H] bf16
    a_ptr,           # [seq_len, H] bf16
    dt_bias_ptr,     # [H] bf16
    a_log_ptr,       # [H] fp32
    q_out_ptr,       # [seq_len, H, K] bf16
    k_out_ptr,       # [seq_len, H, K] bf16
    g_out_ptr,       # [seq_len, H] fp32
    beta_out_ptr,    # [seq_len, H] fp32
    num_key_heads,
    num_value_heads,
    qkv_dim,
    seq_len,
    KEY_DIM: tl.constexpr,
):
    """Prepare head-expanded q/k plus raw g/beta for Qwen3.5 chunk-wise GDR."""
    token_idx = tl.program_id(0)
    v_head = tl.program_id(1)
    if token_idx >= seq_len:
        return

    k_head = (v_head * num_key_heads) // num_value_heads
    qk_dim_total = num_key_heads * KEY_DIM
    token_qkv = qkv_ptr + token_idx * qkv_dim
    offs_k = tl.arange(0, KEY_DIM)

    q = tl.load(token_qkv + k_head * KEY_DIM + offs_k).to(tl.float32)
    k = tl.load(token_qkv + qk_dim_total + k_head * KEY_DIM + offs_k).to(tl.float32)

    q *= tl.rsqrt(tl.sum(q * q, axis=0) + 1e-12)
    k *= tl.rsqrt(tl.sum(k * k, axis=0) + 1e-12)

    q_out = q_out_ptr + (token_idx * num_value_heads + v_head) * KEY_DIM + offs_k
    k_out = k_out_ptr + (token_idx * num_value_heads + v_head) * KEY_DIM + offs_k
    tl.store(q_out, q.to(tl.bfloat16))
    tl.store(k_out, k.to(tl.bfloat16))

    a_val = tl.load(a_ptr + token_idx * num_value_heads + v_head).to(tl.float32)
    b_val = tl.load(b_ptr + token_idx * num_value_heads + v_head).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + v_head).to(tl.float32)
    a_log = tl.load(a_log_ptr + v_head).to(tl.float32)
    x = a_val + dt_bias
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g_val = -tl.exp(a_log) * softplus_x
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    tl.store(g_out_ptr + token_idx * num_value_heads + v_head, g_val)
    tl.store(beta_out_ptr + token_idx * num_value_heads + v_head, beta_val)


@triton.jit
def gdr_chunk_local_cumsum_qwen35_kernel(
    g_in_ptr,        # [seq_len, H] fp32
    g_out_ptr,       # [seq_len, H] fp32
    seq_len,
    num_value_heads,
    BLOCK_T: tl.constexpr,
):
    """Chunk-local prefix sum over the gate tensor for fixed chunk size."""
    chunk_idx = tl.program_id(0)
    v_head = tl.program_id(1)

    offs_t = chunk_idx * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < seq_len
    g = tl.load(g_in_ptr + offs_t * num_value_heads + v_head, mask=mask_t, other=0.0).to(tl.float32)
    g = tl.cumsum(g, axis=0)
    tl.store(g_out_ptr + offs_t * num_value_heads + v_head, g, mask=mask_t)


@triton.jit
def gdr_chunk_scaled_dot_kkt_qwen35_kernel(
    k_ptr,           # [seq_len, H, K] bf16
    g_ptr,           # [seq_len, H] fp32
    beta_ptr,        # [seq_len, H] fp32
    a_ptr,           # [seq_len, H, BLOCK_T] fp32
    seq_len,
    num_value_heads,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KEY_DIM: tl.constexpr,
):
    """Build the chunk-local strict lower-triangular A block."""
    chunk_idx = tl.program_id(0)
    v_head = tl.program_id(1)

    acc = tl.zeros([BLOCK_T, BLOCK_T], dtype=tl.float32)
    for k_block in range(0, KEY_DIM, BLOCK_K):
        p_k = tl.make_block_ptr(
            base=k_ptr + v_head * KEY_DIM,
            shape=(seq_len, KEY_DIM),
            strides=(num_value_heads * KEY_DIM, 1),
            offsets=(chunk_idx * BLOCK_T, k_block),
            block_shape=(BLOCK_T, BLOCK_K),
            order=(1, 0),
        )
        k = tl.load(p_k, boundary_check=(0, 1))
        acc += tl.dot(k, tl.trans(k))

    offs_t = chunk_idx * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < seq_len
    g = tl.load(g_ptr + offs_t * num_value_heads + v_head, mask=mask_t, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_t * num_value_heads + v_head, mask=mask_t, other=0.0).to(tl.float32)

    acc *= beta[:, None]
    acc *= tl.exp(g[:, None] - g[None, :])

    local_t = tl.arange(0, BLOCK_T)
    causal = (local_t[:, None] > local_t[None, :]) & (mask_t[:, None] & mask_t[None, :])
    acc = tl.where(causal, acc, 0.0)

    p_a = tl.make_block_ptr(
        base=a_ptr + v_head * BLOCK_T,
        shape=(seq_len, BLOCK_T),
        strides=(num_value_heads * BLOCK_T, 1),
        offsets=(chunk_idx * BLOCK_T, 0),
        block_shape=(BLOCK_T, BLOCK_T),
        order=(1, 0),
    )
    tl.store(p_a, acc, boundary_check=(0, 1))


@triton.jit
def gdr_solve_tril_64_qwen35_kernel(
    a_ptr,           # [seq_len, H, 64] fp32, strict lower triangular rows
    ai_ptr,          # [seq_len, H, 64] bf16, output inverse rows
    seq_len,
    num_value_heads,
):
    """Fixed-size BT=64 solve_tril for batch=1, fixed-length chunk-wise GDR.

    Adapted from FLA's chunk-wise WY / solve_tril path, specialized down to the
    Qwen3.5 runtime shape and Triton AOT constraints used by pegainfer.
    """
    chunk_idx = tl.program_id(0)
    v_head = tl.program_id(1)

    offs16 = tl.arange(0, 16)
    lower = offs16[:, None] > offs16[None, :]
    eye = offs16[:, None] == offs16[None, :]
    base_a = a_ptr + v_head * 64
    base_ai = ai_ptr + v_head * 64

    p_a_11 = tl.make_block_ptr(base_a, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64, 0), (16, 16), (1, 0))
    p_a_22 = tl.make_block_ptr(base_a, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 16, 16), (16, 16), (1, 0))
    p_a_33 = tl.make_block_ptr(base_a, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 32, 32), (16, 16), (1, 0))
    p_a_44 = tl.make_block_ptr(base_a, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 48, 48), (16, 16), (1, 0))
    b_ai_11 = -tl.where(lower, tl.load(p_a_11, boundary_check=(0, 1)).to(tl.float32), 0.0)
    b_ai_22 = -tl.where(lower, tl.load(p_a_22, boundary_check=(0, 1)).to(tl.float32), 0.0)
    b_ai_33 = -tl.where(lower, tl.load(p_a_33, boundary_check=(0, 1)).to(tl.float32), 0.0)
    b_ai_44 = -tl.where(lower, tl.load(p_a_44, boundary_check=(0, 1)).to(tl.float32), 0.0)

    for i in range(2, 16):
        row = chunk_idx * 64 + i
        a_row = -tl.load(base_a + row * num_value_heads * 64 + offs16, mask=row < seq_len, other=0.0).to(tl.float32)
        a_row = tl.where(offs16 < i, a_row, 0.0)
        a_row += tl.sum(a_row[:, None] * b_ai_11, axis=0)
        b_ai_11 = tl.where((offs16 == i)[:, None], a_row, b_ai_11)
    for i in range(18, 32):
        row = chunk_idx * 64 + i
        a_row = -tl.load(base_a + row * num_value_heads * 64 + offs16 + 16, mask=row < seq_len, other=0.0).to(tl.float32)
        a_row = tl.where(offs16 < i - 16, a_row, 0.0)
        a_row += tl.sum(a_row[:, None] * b_ai_22, axis=0)
        b_ai_22 = tl.where((offs16 == i - 16)[:, None], a_row, b_ai_22)
    for i in range(34, 48):
        row = chunk_idx * 64 + i
        a_row = -tl.load(base_a + row * num_value_heads * 64 + offs16 + 32, mask=row < seq_len, other=0.0).to(tl.float32)
        a_row = tl.where(offs16 < i - 32, a_row, 0.0)
        a_row += tl.sum(a_row[:, None] * b_ai_33, axis=0)
        b_ai_33 = tl.where((offs16 == i - 32)[:, None], a_row, b_ai_33)
    for i in range(50, 64):
        row = chunk_idx * 64 + i
        a_row = -tl.load(base_a + row * num_value_heads * 64 + offs16 + 48, mask=row < seq_len, other=0.0).to(tl.float32)
        a_row = tl.where(offs16 < i - 48, a_row, 0.0)
        a_row += tl.sum(a_row[:, None] * b_ai_44, axis=0)
        b_ai_44 = tl.where((offs16 == i - 48)[:, None], a_row, b_ai_44)

    b_ai_11 += eye
    b_ai_22 += eye
    b_ai_33 += eye
    b_ai_44 += eye

    p_a_21 = tl.make_block_ptr(base_a, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 16, 0), (16, 16), (1, 0))
    p_a_31 = tl.make_block_ptr(base_a, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 32, 0), (16, 16), (1, 0))
    p_a_32 = tl.make_block_ptr(base_a, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 32, 16), (16, 16), (1, 0))
    p_a_41 = tl.make_block_ptr(base_a, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 48, 0), (16, 16), (1, 0))
    p_a_42 = tl.make_block_ptr(base_a, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 48, 16), (16, 16), (1, 0))
    p_a_43 = tl.make_block_ptr(base_a, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 48, 32), (16, 16), (1, 0))
    b_a_21 = tl.load(p_a_21, boundary_check=(0, 1)).to(tl.float32)
    b_a_31 = tl.load(p_a_31, boundary_check=(0, 1)).to(tl.float32)
    b_a_32 = tl.load(p_a_32, boundary_check=(0, 1)).to(tl.float32)
    b_a_41 = tl.load(p_a_41, boundary_check=(0, 1)).to(tl.float32)
    b_a_42 = tl.load(p_a_42, boundary_check=(0, 1)).to(tl.float32)
    b_a_43 = tl.load(p_a_43, boundary_check=(0, 1)).to(tl.float32)

    b_ai_21 = -tl.dot(tl.dot(b_ai_22, b_a_21, input_precision="ieee"), b_ai_11, input_precision="ieee")
    b_ai_32 = -tl.dot(tl.dot(b_ai_33, b_a_32, input_precision="ieee"), b_ai_22, input_precision="ieee")
    b_ai_43 = -tl.dot(tl.dot(b_ai_44, b_a_43, input_precision="ieee"), b_ai_33, input_precision="ieee")
    b_ai_31 = -tl.dot(
        b_ai_33,
        tl.dot(b_a_31, b_ai_11, input_precision="ieee") +
        tl.dot(b_a_32, b_ai_21, input_precision="ieee"),
        input_precision="ieee",
    )
    b_ai_42 = -tl.dot(
        b_ai_44,
        tl.dot(b_a_42, b_ai_22, input_precision="ieee") +
        tl.dot(b_a_43, b_ai_32, input_precision="ieee"),
        input_precision="ieee",
    )
    b_ai_41 = -tl.dot(
        b_ai_44,
        tl.dot(b_a_41, b_ai_11, input_precision="ieee") +
        tl.dot(b_a_42, b_ai_21, input_precision="ieee") +
        tl.dot(b_a_43, b_ai_31, input_precision="ieee"),
        input_precision="ieee",
    )

    p_ai_11 = tl.make_block_ptr(base_ai, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64, 0), (16, 16), (1, 0))
    p_ai_22 = tl.make_block_ptr(base_ai, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 16, 16), (16, 16), (1, 0))
    p_ai_33 = tl.make_block_ptr(base_ai, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 32, 32), (16, 16), (1, 0))
    p_ai_44 = tl.make_block_ptr(base_ai, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 48, 48), (16, 16), (1, 0))
    p_ai_21 = tl.make_block_ptr(base_ai, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 16, 0), (16, 16), (1, 0))
    p_ai_31 = tl.make_block_ptr(base_ai, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 32, 0), (16, 16), (1, 0))
    p_ai_32 = tl.make_block_ptr(base_ai, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 32, 16), (16, 16), (1, 0))
    p_ai_41 = tl.make_block_ptr(base_ai, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 48, 0), (16, 16), (1, 0))
    p_ai_42 = tl.make_block_ptr(base_ai, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 48, 16), (16, 16), (1, 0))
    p_ai_43 = tl.make_block_ptr(base_ai, (seq_len, 64), (num_value_heads * 64, 1), (chunk_idx * 64 + 48, 32), (16, 16), (1, 0))
    tl.store(p_ai_11, b_ai_11.to(p_ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_ai_22, b_ai_22.to(p_ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_ai_33, b_ai_33.to(p_ai_33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_ai_44, b_ai_44.to(p_ai_44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_ai_21, b_ai_21.to(p_ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_ai_31, b_ai_31.to(p_ai_31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_ai_32, b_ai_32.to(p_ai_32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_ai_41, b_ai_41.to(p_ai_41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_ai_42, b_ai_42.to(p_ai_42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_ai_43, b_ai_43.to(p_ai_43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.jit
def gdr_recompute_w_u_qwen35_kernel(
    k_ptr,           # [seq_len, H, K] bf16
    v_ptr,           # [seq_len, H, V] bf16
    beta_ptr,        # [seq_len, H] fp32
    w_ptr,           # [seq_len, H, K] bf16
    u_ptr,           # [seq_len, H, V] bf16
    ai_ptr,          # [seq_len, H, BLOCK_T] bf16
    g_ptr,           # [seq_len, H] fp32
    seq_len,
    num_value_heads,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Fixed-shape fused recompute of chunk-wise w/u.

    Adapted from FLA's `recompute_w_u_fwd_kernel`, specialized for the
    Qwen3.5 batch=1 fixed-shape runtime path.
    """
    chunk_idx = tl.program_id(0)
    v_head = tl.program_id(1)

    p_beta = tl.make_block_ptr(
        beta_ptr + v_head,
        (seq_len,),
        (num_value_heads,),
        (chunk_idx * BLOCK_T,),
        (BLOCK_T,),
        (0,),
    )
    beta = tl.load(p_beta, boundary_check=(0,))
    p_ai = tl.make_block_ptr(
        ai_ptr + v_head * BLOCK_T,
        (seq_len, BLOCK_T),
        (num_value_heads * BLOCK_T, 1),
        (chunk_idx * BLOCK_T, 0),
        (BLOCK_T, BLOCK_T),
        (1, 0),
    )
    ai = tl.load(p_ai, boundary_check=(0, 1))

    for v_tile in range(0, VALUE_DIM, BLOCK_V):
        p_v = tl.make_block_ptr(
            v_ptr + v_head * VALUE_DIM,
            (seq_len, VALUE_DIM),
            (num_value_heads * VALUE_DIM, 1),
            (chunk_idx * BLOCK_T, v_tile),
            (BLOCK_T, BLOCK_V),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u_ptr + v_head * VALUE_DIM,
            (seq_len, VALUE_DIM),
            (num_value_heads * VALUE_DIM, 1),
            (chunk_idx * BLOCK_T, v_tile),
            (BLOCK_T, BLOCK_V),
            (1, 0),
        )
        v = tl.load(p_v, boundary_check=(0, 1))
        vb = v * beta[:, None].to(v.dtype)
        u = tl.dot(ai, vb, allow_tf32=False)
        tl.store(p_u, u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    p_g = tl.make_block_ptr(
        g_ptr + v_head,
        (seq_len,),
        (num_value_heads,),
        (chunk_idx * BLOCK_T,),
        (BLOCK_T,),
        (0,),
    )
    g = tl.exp(tl.load(p_g, boundary_check=(0,)))
    for k_tile in range(0, KEY_DIM, BLOCK_K):
        p_k = tl.make_block_ptr(
            k_ptr + v_head * KEY_DIM,
            (seq_len, KEY_DIM),
            (num_value_heads * KEY_DIM, 1),
            (chunk_idx * BLOCK_T, k_tile),
            (BLOCK_T, BLOCK_K),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w_ptr + v_head * KEY_DIM,
            (seq_len, KEY_DIM),
            (num_value_heads * KEY_DIM, 1),
            (chunk_idx * BLOCK_T, k_tile),
            (BLOCK_T, BLOCK_K),
            (1, 0),
        )
        k = tl.load(p_k, boundary_check=(0, 1))
        kb = k * beta[:, None].to(k.dtype)
        kb *= g[:, None].to(k.dtype)
        w = tl.dot(ai, kb)
        tl.store(p_w, w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def gdr_chunk_state_qwen35_kernel(
    k_ptr,              # [seq_len, H, K] bf16/fp16, token-major, already head-expanded + normalized
    w_ptr,              # [seq_len, H, K] bf16/fp16
    u_ptr,              # [seq_len, H, V] bf16/fp16
    g_ptr,              # [seq_len, H] fp32 cumulative gate
    initial_state_ptr,  # [H, K, V] fp32 (V contiguous, matches FLA/decode convention)
    chunk_state_ptr,    # [num_chunks, H, K, V] fp32 scratch
    v_new_ptr,          # [seq_len, H, V] bf16/fp16 scratch
    final_state_ptr,    # [H, K, V] fp32 (V contiguous, matches FLA/decode convention)
    seq_len,
    num_value_heads,
    BLOCK_V: tl.constexpr,
    BLOCK_T: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    KEY_BLOCK: tl.constexpr,
):
    """Chunk-wise recurrent-state update for Qwen3.5 GDR prefill.

    Adapted from FLA's chunk-state / delta-h recurrence kernels, then reshaped
    for pegainfer's fixed Qwen3.5 runtime and decode-state contract.

    This stage assumes `g`, `w`, and `u` are already prepared. It stores one
    per-chunk state snapshot in `[K, V]` scratch layout, writes token-level
    `v_new`, and updates the final decode-compatible state `[H, K, V]`.
    """
    v_tile = tl.program_id(0)
    v_head = tl.program_id(1)

    offs_v = v_tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < VALUE_DIM

    # State layout: [K, V] per head — V contiguous, matching FLA/decode convention.
    # h_lo/h_hi are [KEY_BLOCK, BLOCK_V] tiles loaded directly (no transpose).
    p_h0_lo = tl.make_block_ptr(
        base=initial_state_ptr + v_head * KEY_DIM * VALUE_DIM,
        shape=(KEY_DIM, VALUE_DIM),
        strides=(VALUE_DIM, 1),
        offsets=(0, v_tile * BLOCK_V),
        block_shape=(KEY_BLOCK, BLOCK_V),
        order=(1, 0),
    )
    p_h0_hi = tl.make_block_ptr(
        base=initial_state_ptr + v_head * KEY_DIM * VALUE_DIM,
        shape=(KEY_DIM, VALUE_DIM),
        strides=(VALUE_DIM, 1),
        offsets=(KEY_BLOCK, v_tile * BLOCK_V),
        block_shape=(KEY_BLOCK, BLOCK_V),
        order=(1, 0),
    )

    h_lo = tl.load(p_h0_lo, boundary_check=(0, 1)).to(tl.float32)
    h_hi = tl.load(p_h0_hi, boundary_check=(0, 1)).to(tl.float32)

    num_chunks = tl.cdiv(seq_len, BLOCK_T)
    t_offsets = tl.arange(0, BLOCK_T)

    for chunk_idx in range(num_chunks):
        chunk_base = chunk_state_ptr + (chunk_idx * num_value_heads + v_head) * KEY_DIM * VALUE_DIM
        p_chunk_lo = tl.make_block_ptr(
            base=chunk_base,
            shape=(KEY_DIM, VALUE_DIM),
            strides=(VALUE_DIM, 1),
            offsets=(0, v_tile * BLOCK_V),
            block_shape=(KEY_BLOCK, BLOCK_V),
            order=(1, 0),
        )
        p_chunk_hi = tl.make_block_ptr(
            base=chunk_base,
            shape=(KEY_DIM, VALUE_DIM),
            strides=(VALUE_DIM, 1),
            offsets=(KEY_BLOCK, v_tile * BLOCK_V),
            block_shape=(KEY_BLOCK, BLOCK_V),
            order=(1, 0),
        )
        tl.store(p_chunk_lo, h_lo.to(p_chunk_lo.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_chunk_hi, h_hi.to(p_chunk_hi.dtype.element_ty), boundary_check=(0, 1))

        p_w_lo = tl.make_block_ptr(
            base=w_ptr + v_head * KEY_DIM,
            shape=(seq_len, KEY_DIM),
            strides=(num_value_heads * KEY_DIM, 1),
            offsets=(chunk_idx * BLOCK_T, 0),
            block_shape=(BLOCK_T, KEY_BLOCK),
            order=(1, 0),
        )
        p_w_hi = tl.make_block_ptr(
            base=w_ptr + v_head * KEY_DIM,
            shape=(seq_len, KEY_DIM),
            strides=(num_value_heads * KEY_DIM, 1),
            offsets=(chunk_idx * BLOCK_T, KEY_BLOCK),
            block_shape=(BLOCK_T, KEY_BLOCK),
            order=(1, 0),
        )
        p_u = tl.make_block_ptr(
            base=u_ptr + v_head * VALUE_DIM,
            shape=(seq_len, VALUE_DIM),
            strides=(num_value_heads * VALUE_DIM, 1),
            offsets=(chunk_idx * BLOCK_T, v_tile * BLOCK_V),
            block_shape=(BLOCK_T, BLOCK_V),
            order=(1, 0),
        )

        w_lo = tl.load(p_w_lo, boundary_check=(0, 1))
        w_hi = tl.load(p_w_hi, boundary_check=(0, 1))
        v_new = tl.load(p_u, boundary_check=(0, 1)).to(tl.float32)
        v_new -= tl.dot(w_lo, h_lo.to(w_lo.dtype))
        v_new -= tl.dot(w_hi, h_hi.to(w_hi.dtype))

        token_idx = chunk_idx * BLOCK_T + t_offsets
        mask_t = token_idx < seq_len
        g_last_idx = tl.minimum((chunk_idx + 1) * BLOCK_T, seq_len) - 1
        g_last = tl.load(g_ptr + g_last_idx * num_value_heads + v_head).to(tl.float32)
        g_chunk = tl.load(
            g_ptr + token_idx * num_value_heads + v_head,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)
        gate = tl.where(mask_t, tl.exp(g_last - g_chunk), 0.0)
        decay = tl.exp(g_last)

        h_lo *= decay
        h_hi *= decay

        p_v_new = tl.make_block_ptr(
            base=v_new_ptr + v_head * VALUE_DIM,
            shape=(seq_len, VALUE_DIM),
            strides=(num_value_heads * VALUE_DIM, 1),
            offsets=(chunk_idx * BLOCK_T, v_tile * BLOCK_V),
            block_shape=(BLOCK_T, BLOCK_V),
            order=(1, 0),
        )
        tl.store(p_v_new, v_new.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
        v_new *= gate[:, None]

        p_k_lo = tl.make_block_ptr(
            base=k_ptr + v_head * KEY_DIM,
            shape=(KEY_DIM, seq_len),
            strides=(1, num_value_heads * KEY_DIM),
            offsets=(0, chunk_idx * BLOCK_T),
            block_shape=(KEY_BLOCK, BLOCK_T),
            order=(0, 1),
        )
        p_k_hi = tl.make_block_ptr(
            base=k_ptr + v_head * KEY_DIM,
            shape=(KEY_DIM, seq_len),
            strides=(1, num_value_heads * KEY_DIM),
            offsets=(KEY_BLOCK, chunk_idx * BLOCK_T),
            block_shape=(KEY_BLOCK, BLOCK_T),
            order=(0, 1),
        )
        k_lo = tl.load(p_k_lo, boundary_check=(0, 1))
        k_hi = tl.load(p_k_hi, boundary_check=(0, 1))
        v_new_mma = v_new.to(k_lo.dtype)

        h_lo += tl.dot(k_lo, v_new_mma)
        h_hi += tl.dot(k_hi, v_new_mma)

    # Store final state in [K, V] layout — no transpose, matching decode convention.
    p_ht_lo = tl.make_block_ptr(
        base=final_state_ptr + v_head * KEY_DIM * VALUE_DIM,
        shape=(KEY_DIM, VALUE_DIM),
        strides=(VALUE_DIM, 1),
        offsets=(0, v_tile * BLOCK_V),
        block_shape=(KEY_BLOCK, BLOCK_V),
        order=(1, 0),
    )
    p_ht_hi = tl.make_block_ptr(
        base=final_state_ptr + v_head * KEY_DIM * VALUE_DIM,
        shape=(KEY_DIM, VALUE_DIM),
        strides=(VALUE_DIM, 1),
        offsets=(KEY_BLOCK, v_tile * BLOCK_V),
        block_shape=(KEY_BLOCK, BLOCK_V),
        order=(1, 0),
    )
    tl.store(p_ht_lo, h_lo.to(p_ht_lo.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_ht_hi, h_hi.to(p_ht_hi.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def gdr_chunk_o_qwen35_kernel(
    q_ptr,            # [seq_len, H, K] bf16/fp16, token-major, already head-expanded + normalized
    k_ptr,            # [seq_len, H, K] bf16/fp16, token-major, already head-expanded + normalized
    v_new_ptr,        # [seq_len, H, V] bf16/fp16
    chunk_state_ptr,  # [num_chunks, H, K, V] fp32 scratch
    g_ptr,            # [seq_len, H] fp32 cumulative gate
    output_ptr,       # [seq_len, H, V] bf16/fp16
    seq_len,
    num_value_heads,
    scale,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_T: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
):
    """Chunk-wise output stage for Qwen3.5 GDR prefill.

    Adapted from FLA's chunk-wise output kernel, then specialized for the
    fixed-shape Qwen3.5 runtime path used by pegainfer.

    This stage consumes normalized q/k, token-level `v_new`, and per-chunk
    state snapshots to produce the final token-major output.
    """
    v_tile = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    v_head = tl.program_id(2)

    offs_t = chunk_idx * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < seq_len

    q_head = q_ptr + v_head * KEY_DIM
    k_head = k_ptr + v_head * KEY_DIM
    v_head_ptr = v_new_ptr + v_head * VALUE_DIM
    out_head_ptr = output_ptr + v_head * VALUE_DIM
    h_chunk_ptr = chunk_state_ptr + (chunk_idx * num_value_heads + v_head) * KEY_DIM * VALUE_DIM

    acc_o = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)
    acc_a = tl.zeros([BLOCK_T, BLOCK_T], dtype=tl.float32)

    for k_block in range(0, KEY_DIM, BLOCK_K):
        p_q = tl.make_block_ptr(
            base=q_head,
            shape=(seq_len, KEY_DIM),
            strides=(num_value_heads * KEY_DIM, 1),
            offsets=(chunk_idx * BLOCK_T, k_block),
            block_shape=(BLOCK_T, BLOCK_K),
            order=(1, 0),
        )
        p_k = tl.make_block_ptr(
            base=k_head,
            shape=(KEY_DIM, seq_len),
            strides=(1, num_value_heads * KEY_DIM),
            offsets=(k_block, chunk_idx * BLOCK_T),
            block_shape=(BLOCK_K, BLOCK_T),
            order=(0, 1),
        )
        p_h = tl.make_block_ptr(
            base=h_chunk_ptr,
            shape=(KEY_DIM, VALUE_DIM),
            strides=(VALUE_DIM, 1),
            offsets=(k_block, v_tile * BLOCK_V),
            block_shape=(BLOCK_K, BLOCK_V),
            order=(1, 0),
        )

        q = tl.load(p_q, boundary_check=(0, 1))
        k = tl.load(p_k, boundary_check=(0, 1))
        h = tl.load(p_h, boundary_check=(0, 1))

        acc_o += tl.dot(q, h.to(q.dtype))
        acc_a += tl.dot(q, k)

    g = tl.load(g_ptr + offs_t * num_value_heads + v_head, mask=mask_t, other=0.0).to(tl.float32)
    acc_o *= tl.exp(g)[:, None]
    acc_a *= tl.exp(g[:, None] - g[None, :])

    local_t = tl.arange(0, BLOCK_T)
    causal = (local_t[:, None] >= local_t[None, :]) & (mask_t[:, None] & mask_t[None, :])
    acc_a = tl.where(causal, acc_a, 0.0)

    p_v = tl.make_block_ptr(
        base=v_head_ptr,
        shape=(seq_len, VALUE_DIM),
        strides=(num_value_heads * VALUE_DIM, 1),
        offsets=(chunk_idx * BLOCK_T, v_tile * BLOCK_V),
        block_shape=(BLOCK_T, BLOCK_V),
        order=(1, 0),
    )
    p_out = tl.make_block_ptr(
        base=out_head_ptr,
        shape=(seq_len, VALUE_DIM),
        strides=(num_value_heads * VALUE_DIM, 1),
        offsets=(chunk_idx * BLOCK_T, v_tile * BLOCK_V),
        block_shape=(BLOCK_T, BLOCK_V),
        order=(1, 0),
    )

    v_new = tl.load(p_v, boundary_check=(0, 1))
    out = (acc_o + tl.dot(acc_a.to(v_new.dtype), v_new)) * scale
    tl.store(p_out, out.to(p_out.dtype.element_ty), boundary_check=(0, 1))
