"""Torch reference for the fp8 weight-only GEMV (semantic-level net only).

The production kernel's f32 association order is NOT reproduced here (torch
cannot): the reference dequants the e4m3 block-scale weight to f32 and runs
one f32 matmul. The manifest tolerance absorbs the reordering — the gate is
"kernel distance to f32 <= bf16 store floor", not bit-identity.
"""
from __future__ import annotations

from kernel_lab.data import FP8_BLOCK


def e4m3_decode_torch(weight_u8):
    """Vectorized e4m3 -> f32 (sign/exp/mantissa decode matching the kernels'
    bit-level half conversion chain)."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    b = weight_u8.to(torch.int32)
    sign = torch.where(b & 0x80 != 0, -1.0, 1.0)
    exp = (b >> 3) & 0xF
    man = (b & 0x7).to(torch.float32) / 8.0
    subnormal = sign * man * 2.0**-6
    normal = sign * (1.0 + man) * torch.exp2((exp - 7).to(torch.float32))
    return torch.where(exp == 0, subnormal, normal)


def fp8_weight_only_gemv_ref(act_bf16, weight_u8, scales_f32, n: int, k: int):
    """out[bs, n] = act[bs, k] @ deq(weight[n, k])^T accumulated in f32.

    deq(W) = e4m3(W) * weight_scale_inv per 128x128 block — the same scale
    association as the kernel (`scale[(n0>>7) * (k/128) + (kk>>7)]`).
    """
    from kernel_lab.loader import require_torch

    torch = require_torch()
    if n % FP8_BLOCK or k % FP8_BLOCK:
        raise ValueError(f"reference needs {FP8_BLOCK}-divisible n/k, got {n}/{k}")
    w = e4m3_decode_torch(weight_u8).view(
        n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK
    )
    s = scales_f32.view(n // FP8_BLOCK, 1, k // FP8_BLOCK, 1)
    deq = (w * s).reshape(n, k)
    return act_bf16.to(torch.float32) @ deq.t()
