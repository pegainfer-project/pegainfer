"""Torch reference + stdlib spec for the GLM5.2 FP8 per-token-group quant twins.

Production kernels (csrc/glm52/glm52_moe_quant.cu; FFI mirror
pegainfer-kernels/src/ffi/glm52.rs:286-304; ops wrapper
pegainfer-kernels/src/ops/glm52/moe_quant.rs), per (row, 128-group):

- `glm52_fp8_per_token_group_quant_bf16_cuda` —
  amax = max(|x|) f32, scale = fmaxf(amax, 1e-10) / 448,
  out = e4m3(fminf(fmaxf(x / scale, -448), 448)).
- `glm52_fp8_per_token_group_quant_bf16_ue8m0_cuda` — same, then the scale is
  rounded UP to the next power of two via exact bit manipulation
  (`(bits + 0x7FFFFF) & 0x7F800000`, no log2f). FlashMLA V3.2 fp8 sparse
  KV-cache contract: the sm100 decode kernel truncates stored f32 scales to
  e8m0 (round-toward-zero) for its block-scaled MMA, so only power-of-two
  scales read identically on sm90 and sm100
  (docs/lessons/flashmla-sm100-ue8m0-kv-scales.md).

Bit-exactness argument (target: byte-identical out + identical scale bits):

- bf16 -> f32 widening and |x|: exact on both sides.
- amax: f32 max-reduce — order-independent, so the kernel's shared-memory tree
  and torch's amax agree exactly.
- scale: one f32 clamp + one f32 IEEE division (div.rn on both sides); the
  1e-10 / 448.0 literals parse to identical f32 constants on both sides.
  PITFALL (GB300 FAIL 2026-07-29; independent of the datacenter arch — any
  later Blackwell part reproduces it): torch lowers
  tensor / CPU-scalar to multiply-by-reciprocal (BinaryDivTrueKernel.cu
  is_cpu_scalar fast path), and rn(1/448) is inexact — the reference must
  divide by a DEVICE tensor of 448.0 to get the kernel's true div.rn. The
  ue8m0 twin is immune (the pow2 bump absorbs a sub-ulp difference, and
  exactly-representable quotients round back exactly); the plain path is not.
- encode: the kernel's `__nv_cvt_float_to_fp8(__NV_SATFINITE, __NV_E4M3)`
  lowers to PTX `cvt.rn.satfinite.e4m3x2.f32` — round-to-nearest-EVEN with
  saturation to finite. torch's f32 -> `torch.float8_e4m3fn` cast is RNE as
  well (c10 `fp8e4m3fn_from_fp32_value`), and inputs are pre-clamped to +-448
  on both sides, so the satfinite-vs-NaN overflow distinction never fires and
  midpoint ties land on the same (even-significand) side.
- ue8m0: identical integer bit-twiddle on both sides.

The scalar helpers are the torch-free spec mirror of the kernels; the tensor
functions lazily import torch per the kernel_lab contract.
"""
from __future__ import annotations

import bisect
import math
import struct

from kernel_lab import data
from kernel_lab.loader import require_torch

GROUP_SIZE = 128  # kernel kGroupSize; the FFI rejects any other group_size
E4M3_MAX = data.E4M3_MAX  # 448.0
QUANT_EPS = 1.0e-10  # kernel kPerTokenGroupQuantEps (f32 literal on both sides)
_F32_EXP_MASK = 0x7F80_0000
_F32_MAN_MASK = 0x007F_FFFF

# --- torch-free scalar spec (CPU-testable) -----------------------------------


def f32_to_bits(x: float) -> int:
    """Round a python float to the nearest f32 and return its IEEE bits."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF_FFFF))[0]


def ue8m0_ceil_pow2_bits(bits: int) -> int:
    """Next power of two >= the positive f32 with these bits — the kernel's
    `(bits + 0x7FFFFF) & 0x7F800000` verbatim. Powers of two are unchanged;
    f32-max would bump to +inf, which is unreachable for bf16-sourced inputs
    (amax/448 <= ~7.6e35) and mirrored here only as spec."""
    return (bits + _F32_MAN_MASK) & _F32_EXP_MASK


def group_scale_f32(amax: float, ue8m0: bool = False) -> float:
    """f32 group scale: fmaxf(amax, eps)/448 [+ ue8m0 bump].

    `amax` must already be an exact f32 value (it is a max of f32 inputs).
    The eps clamp compares f32 bit patterns (positive floats order like their
    bits), mirroring fmaxf exactly. The division is emulated as f64 quotient
    rounded to f32 — one double-rounding step vs the kernel's single f32
    div.rn, so CPU tests must use amax values with exactly-representable
    quotients; the tensor path (fp8_per_token_group_quant_ref) performs the
    real f32 division and carries no such caveat.
    """
    eps = bits_to_f32(f32_to_bits(QUANT_EPS))
    s = amax if f32_to_bits(amax) > f32_to_bits(eps) else eps
    s = bits_to_f32(f32_to_bits(s / E4M3_MAX))
    if ue8m0:
        s = bits_to_f32(ue8m0_ceil_pow2_bits(f32_to_bits(s)))
    return s


# Positive e4m3 magnitudes, ascending: (value, byte) with byte < 0x80 —
# 127 codes 0x00..0x7E (NaN 0x7F excluded by contract on both sides).
_POS_CODEBOOK = tuple((v, b) for v, b in data.e4m3_codebook() if b < 0x80)
_POS_VALUES = tuple(v for v, _ in _POS_CODEBOOK)


def e4m3_encode_rne(value: float) -> int:
    """Software round-to-nearest-even e4m3 encode of a finite value, with the
    kernel's satfinite clamp folded in (|value| > 448 encodes as +-448). This
    is the spec for both PTX `cvt.rn.satfinite.e4m3x2.f32` (kernel) and the
    torch float8_e4m3fn cast (reference). Midpoint ties resolve to the even
    significand LSB, which is the even byte LSB: adjacent codes alternate
    parity, including across the subnormal/normal boundary (0x07 <-> 0x08).
    Signed zero is preserved (cvt keeps the sign bit: -0.0 -> 0x80)."""
    if math.isnan(value):
        raise ValueError("NaN is out of contract for the quant kernels")
    sign = 0x80 if math.copysign(1.0, value) < 0.0 else 0x00
    a = min(abs(value), E4M3_MAX)
    i = bisect.bisect_left(_POS_VALUES, a)
    if i == 0:
        chosen = _POS_CODEBOOK[0]
    elif i >= len(_POS_CODEBOOK):
        chosen = _POS_CODEBOOK[-1]
    else:
        lo, hi = _POS_CODEBOOK[i - 1], _POS_CODEBOOK[i]
        d_lo, d_hi = a - lo[0], hi[0] - a
        if d_lo < d_hi:
            chosen = lo
        elif d_hi < d_lo:
            chosen = hi
        else:
            chosen = lo if lo[1] & 1 == 0 else hi
    return sign | chosen[1]


def packed_surface_len(rows: int, hidden: int) -> int:
    """Byte length of the packed comparison surface: e4m3 bytes followed by
    the raw little-endian f32 scale bytes. Torch-free; shared by the adapters
    and the CPU tests."""
    return rows * hidden + rows * (hidden // GROUP_SIZE) * 4


# --- torch tensor reference (lazy torch) --------------------------------------


def ue8m0_ceil_pow2_f32(scale_f32):
    """Tensor twin of ue8m0_ceil_pow2_bits — int32 bit math, no log2f."""
    torch = require_torch()
    bits = scale_f32.contiguous().view(torch.int32)
    return ((bits + _F32_MAN_MASK) & _F32_EXP_MASK).view(torch.float32)


def fp8_per_token_group_quant_ref(act_bf16, ue8m0: bool = False):
    """(out e4m3 uint8 [rows, hidden], scales f32 [rows, hidden/128]).

    Bit-exact mirror of the production kernel per the module docstring: every
    step is either exact (widening, abs, max, bit-twiddle) or the identical
    IEEE/RNE op on both sides (clamp, div.rn, e4m3 encode).
    """
    torch = require_torch()
    rows, hidden = act_bf16.shape
    if hidden % GROUP_SIZE:
        raise ValueError(f"hidden {hidden} not divisible by group {GROUP_SIZE}")
    x = act_bf16.to(torch.float32).view(rows, hidden // GROUP_SIZE, GROUP_SIZE)
    amax = x.abs().amax(dim=-1)  # exact f32 max, order-independent
    # True f32 div.rn, matching the kernel: the divisor must be a DEVICE
    # tensor. torch lowers tensor / CPU-scalar to multiply-by-reciprocal
    # (BinaryDivTrueKernel.cu is_cpu_scalar fast path) and rn(1/448) is
    # inexact, which put the reference scale a systematic +1 ulp above the
    # kernel's div.rn(amax, 448) on ~57% of groups (GB300 FAIL rel_l2=4.19e-4,
    # 2026-07-29; arch-independent). The ue8m0 twin only survived because
    # the pow2 bump absorbs a sub-ulp difference.
    scale = torch.clamp(amax, min=QUANT_EPS) / torch.full_like(amax, E4M3_MAX)
    if ue8m0:
        scale = ue8m0_ceil_pow2_f32(scale)
    q = torch.clamp(x / scale.unsqueeze(-1), -E4M3_MAX, E4M3_MAX)
    out = q.to(torch.float8_e4m3fn).view(torch.uint8).view(rows, hidden)
    return out, scale


def assert_scales_power_of_two(scales_f32) -> None:
    """UE8M0 contract assertion for the check path: every kernel-written scale
    must be a positive finite power of two (sign clear, exponent 1..254,
    mantissa zero). Raises AssertionError with the violation count — a
    non-power-of-two scale reads up to 2x too small after the sm100 FlashMLA
    e8m0 truncation."""
    torch = require_torch()
    bits = scales_f32.detach().contiguous().view(torch.int32)
    ok = (bits > 0) & (bits < _F32_EXP_MASK) & ((bits & _F32_MAN_MASK) == 0)
    bad = int((~ok).sum().item())
    if bad:
        raise AssertionError(
            f"ue8m0 quant wrote {bad}/{bits.numel()} non-power-of-two scales "
            "(sm100 FlashMLA e8m0 truncation would read them up to 2x too small)"
        )


def pack_quant_surface(out_u8, scales_f32):
    """Flatten (e4m3 bytes, scales as raw LE bytes) into one f32 tensor of
    byte values. compute_metrics casts the packed uint8 `out` buffer to f32,
    so the reference returns the expected byte values as f32 of equal length:
    rel_l2 == 0 iff the kernel is bit-identical in both bytes and scale bits.
    """
    torch = require_torch()
    flat = torch.cat(
        [out_u8.reshape(-1), scales_f32.contiguous().view(torch.uint8).reshape(-1)]
    )
    return flat.to(torch.float32)
