"""Torch reference for the fp8 groupwise SM100 GEMM (semantic-level net only).

`glm52_fp8_groupwise_gemm_sm100_cuda` (csrc/glm52/glm52_fp8_gemm.cu) runs the
FlashInfer/CUTLASS tcgen05 block-scaled MMA with a fixed hardware accumulation
order the torch side cannot reproduce: the reference dequants both e4m3
block-scale operands to f32 and runs one f32 matmul, and the manifest
tolerance absorbs the reordering.
"""
from __future__ import annotations

from kernel_lab.loader import require_torch
from kernel_lab.refs.fp8_gemv import e4m3_decode_torch


def groupwise_gemm_ref(act_q, act_scales, weight_q, weight_scales, m: int, n: int, k: int):
    """out[m, n] = deq(act_q) @ deq(weight_q)^T, f32 accumulate, bf16 store.

    act_q:      e4m3 uint8 [m, k]     act_scales:    f32 [m, k/128]
    weight_q:   e4m3 uint8 [n, k]     weight_scales: f32 [n/128, k/128]
    Scale broadcast mirrors the block recipe: 128 consecutive k columns share
    one activation scalar; one 128x128 weight block shares one weight scalar.
    """
    torch = require_torch()
    a = e4m3_decode_torch(act_q.view(m, k))
    a = a * act_scales.repeat_interleave(128, dim=1)
    w = e4m3_decode_torch(weight_q.view(n, k))
    w = w * weight_scales.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    out = torch.matmul(a, w.t())
    # The production epilogue stores bf16; mirror the rounding so the
    # residual is pure accumulation-order noise.
    return out.to(torch.bfloat16)
