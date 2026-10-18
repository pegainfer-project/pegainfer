"""Seeded input factories + e4m3 codec (mirrors the Rust smoke-test semantics).

Seeds are purpose-named: every tensor family derives its own stream from the
run seed, so regenerating one input never reshuffles another. Scalar codec
helpers stay torch-free for the CPU tests; the tensor factories import torch
lazily.
"""
from __future__ import annotations

import hashlib

E4M3_MAX = 448.0  # S.1111.110; 0x7f/0xff are NaN and never generated
FP8_BLOCK = 128


def derive_seed(base: int, purpose: str) -> int:
    """Deterministic per-purpose seed: sha256("kernel_lab:<base>:<purpose>")."""
    digest = hashlib.sha256(f"kernel_lab:{base}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def e4m3_to_f32(byte: int) -> float:
    """e4m3 byte -> f32, matching `__nv_fp8_e4m3` semantics (NaN encodings are
    excluded from generated data — same contract as the Rust smoke test)."""
    sign = -1.0 if byte & 0x80 else 1.0
    exp = (byte >> 3) & 0xF
    man = float(byte & 0x7)
    if exp == 0:
        return sign * (man / 8.0) * 2.0**-6
    return sign * (1.0 + man / 8.0) * 2.0 ** (exp - 7)


def e4m3_codebook() -> list[tuple[float, int]]:
    """Sorted unique (value, byte) pairs with NaN encodings (0x7f/0xff) dropped.
    +0.0/-0.0 collapse to one entry (byte 0x00)."""
    seen: dict[float, int] = {}
    for b in range(256):
        if b & 0x7F == 0x7F:
            continue
        seen.setdefault(e4m3_to_f32(b), b)
    return sorted(seen.items())


def normal_bf16(shape: tuple[int, ...], seed: int, device: str = "cuda"):
    """N(0,1) activations in bf16. Generated on CPU with a fixed seed so the
    stream is identical across machines, then moved to `device`."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=gen, dtype=torch.float32)
    return x.to(torch.bfloat16).to(device)


def normal_quantized_fp8(n: int, k: int, seed: int, device: str = "cuda"):
    """Normal-distributed weight quantized per 128x128 block to e4m3 + f32 scale,
    the checkpoint `weight_scale_inv` recipe. Uniform-random e4m3 bytes are NOT
    used on purpose: they blow up the tensor-core accumulators (lesson of the
    fp8 line, docs/models/glm52/fp8-blockwise-gemm-lab.md).

    Returns (weight uint8 [n, k], scales f32 [n/128, k/128]) on `device`,
    matching the kernel's scale indexing `scale[(n0>>7) * (k/128) + (kk>>7)]`.
    """
    from kernel_lab.loader import require_torch

    torch = require_torch()
    if n % FP8_BLOCK or k % FP8_BLOCK:
        raise ValueError(f"normal_quantized_fp8 needs {FP8_BLOCK}-divisible n/k, got {n}/{k}")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn((n, k), generator=gen, dtype=torch.float32)
    wb = w.view(n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK)
    absmax = wb.abs().amax(dim=(1, 3)).clamp_min(1e-12)
    scales = absmax / E4M3_MAX  # [n/128, k/128]
    q = wb / scales.view(-1, 1, wb.shape[2], 1)  # |q| <= 448 by construction

    values, encodings = zip(*e4m3_codebook())
    table_v = torch.tensor(values, dtype=torch.float32)
    table_b = torch.tensor(encodings, dtype=torch.uint8)
    midpoints = (table_v[:-1] + table_v[1:]) / 2.0
    idx = torch.searchsorted(midpoints, q.reshape(-1))  # round-to-nearest
    weight = table_b[idx].view(n, k).contiguous()
    return weight.to(device), scales.contiguous().to(device)
