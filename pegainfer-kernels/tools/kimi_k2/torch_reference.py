#!/usr/bin/env python3
"""Kimi-K2 routed expert INT4 grouped GEMM reference fixture.

Generates a self-contained fixture for validating
``kimi_int4_grouped_w1_w3_cuda`` / ``kimi_int4_grouped_w2_swiglu_cuda``:

* random signed INT4 weights packed via the canonical compressed-tensors
  ``pack_to_int32`` routine, then re-laid out the same way vLLM keeps them at
  runtime ([num_experts, out_dim, in_dim / 2] uint8, low nibble = even input
  column, high nibble = odd input column);
* BF16 group scales [num_experts, out_dim, in_dim / 32] matching the
  per-output-channel × group-of-32-along-K layout;
* BF16 expert-major routed activations and the expert_indptr that drives
  the per-expert problem shape;
* torch reference outputs for W1/W3 grouped GEMM, fused SwiGLU activation,
  and W2 grouped GEMM, computed via dequant + bmm.

The default shape is small and CPU-runnable; pass ``--kimi`` for the real
EP8 shape (E=48, hidden=7168, intermediate=2048) — that path needs a GPU.

Outputs land in ``<out-dir>/kimi_int4_grouped.safetensors`` with a sidecar
``metadata.json`` describing shapes/dtypes/contract attributes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    unpack_from_int32,
)
from safetensors.torch import save_file


GROUP_SIZE = 32
NIBBLE_OFFSET = 8  # signed INT4 in [-8, 7] stored as unsigned [0, 15] via +8


def quantize_int4_per_group(weight: torch.Tensor, group_size: int):
    """Quantize a BF16 weight ``[out, in]`` per (out_row, in_group) to signed INT4.

    Returns ``(int4_signed [out, in] int8, scale [out, in/group_size] bf16)``.
    The reverse is ``dequant(int4_signed) = int4_signed.to(bf16) * scale``.
    """
    assert weight.ndim == 2, weight.shape
    out_dim, in_dim = weight.shape
    assert in_dim % group_size == 0, (in_dim, group_size)
    grouped = weight.reshape(out_dim, in_dim // group_size, group_size).to(torch.float32)
    absmax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = absmax / 7.0  # signed INT4 max magnitude
    q = torch.round(grouped / scale).clamp(-8, 7).to(torch.int8)
    scale_bf16 = scale.squeeze(-1).to(torch.bfloat16)  # [out, in/group_size]
    int4_signed = q.reshape(out_dim, in_dim)
    return int4_signed, scale_bf16


def pack_signed_int4_to_vllm_uint8(int4_signed: torch.Tensor) -> torch.Tensor:
    """Pack signed-INT4 weight ``[out, in]`` int8 → vLLM ``[out, in/2]`` uint8.

    Goes through compressed-tensors' canonical packer (signed → +8 → unsigned
    int32 packed along ``packed_dim=1`` with 8 nibbles per int32) and then
    applies vLLM's ``process_weights_after_loading`` reshape (view as uint8 →
    [out, in/2]).
    """
    packed_int32 = pack_to_int32(int4_signed, num_bits=4, packed_dim=1)
    # packed_int32 shape: [out, in / 8] int32
    return packed_int32.contiguous().view(torch.uint8)  # [out, in / 2] uint8 (LE)


def dequant_from_vllm_uint8(
    packed_uint8: torch.Tensor,
    scale_bf16: torch.Tensor,
    in_dim: int,
    group_size: int,
) -> torch.Tensor:
    """Round-trip: ``[out, in/2]`` uint8 + ``[out, in/group]`` bf16 → ``[out, in]`` bf16.

    Independent of the packer; uses byte-wise nibble extraction matching our
    Rust/CUDA contract (low nibble = even input column).
    """
    out_dim = packed_uint8.shape[0]
    assert packed_uint8.shape[1] == in_dim // 2
    bytes_ = packed_uint8.to(torch.int32)  # widen so shift is sign-safe
    low = (bytes_ & 0xF) - NIBBLE_OFFSET  # signed int4 at even input cols
    high = ((bytes_ >> 4) & 0xF) - NIBBLE_OFFSET  # signed int4 at odd input cols
    # Interleave low/high along in dim: [out, in/2, 2] -> [out, in]
    interleaved = torch.stack([low, high], dim=-1).reshape(out_dim, in_dim)
    signed = interleaved.to(torch.float32)
    # Broadcast group scale along in dim
    scale = scale_bf16.to(torch.float32).repeat_interleave(group_size, dim=1)
    return (signed * scale).to(torch.bfloat16)


def grouped_matmul_bf16(
    hidden: torch.Tensor,
    weight_bf16: torch.Tensor,
    expert_indptr: torch.Tensor,
) -> torch.Tensor:
    """Per-expert routed matmul.

    ``hidden`` is expert-major ``[routed_tokens, in_dim]`` BF16.
    ``weight_bf16`` is ``[num_experts, out_dim, in_dim]`` BF16.
    ``expert_indptr`` is ``[num_experts + 1]`` u32 prefix-sum.
    Returns ``[routed_tokens, out_dim]`` BF16.
    """
    num_experts = weight_bf16.shape[0]
    out_dim = weight_bf16.shape[1]
    routed = hidden.shape[0]
    out = torch.zeros((routed, out_dim), dtype=torch.bfloat16, device=hidden.device)
    for e in range(num_experts):
        start = int(expert_indptr[e].item())
        end = int(expert_indptr[e + 1].item())
        if end <= start:
            continue
        x = hidden[start:end].to(torch.float32)
        w = weight_bf16[e].to(torch.float32)
        out[start:end] = (x @ w.T).to(torch.bfloat16)
    return out


def silu_mul_bf16(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Matches the rounding of ``silu_mul_triton_aot_cuda``: silu computed in
    f32, cast to bf16, then multiplied with up via f32 product, cast back."""
    g = gate.to(torch.float32)
    u = up.to(torch.float32)
    silu_g = g / (1.0 + torch.exp(-g))
    silu_bf = silu_g.to(torch.bfloat16).to(torch.float32)
    return (silu_bf * u).to(torch.bfloat16)


def build_fixture(
    num_experts: int,
    hidden: int,
    intermediate: int,
    tokens_per_expert: list[int],
    group_size: int,
    seed: int,
):
    torch.manual_seed(seed)
    indptr = [0]
    for t in tokens_per_expert:
        indptr.append(indptr[-1] + t)
    routed_tokens = indptr[-1]
    expert_indptr = torch.tensor(indptr, dtype=torch.uint32)

    # Expert-major routed activation
    expert_hidden = (torch.randn(routed_tokens, hidden) * 0.1).to(torch.bfloat16)

    # Master BF16 weights (the reference we quantize from)
    w1_bf16 = (torch.randn(num_experts, intermediate, hidden) * 0.05).to(torch.bfloat16)
    w3_bf16 = (torch.randn(num_experts, intermediate, hidden) * 0.05).to(torch.bfloat16)
    w2_bf16 = (torch.randn(num_experts, hidden, intermediate) * 0.05).to(torch.bfloat16)

    # Per-expert quantize → pack
    w1_packed = torch.empty(num_experts, intermediate, hidden // 2, dtype=torch.uint8)
    w3_packed = torch.empty_like(w1_packed)
    w2_packed = torch.empty(num_experts, hidden, intermediate // 2, dtype=torch.uint8)
    w1_scale = torch.empty(num_experts, intermediate, hidden // group_size, dtype=torch.bfloat16)
    w3_scale = torch.empty_like(w1_scale)
    w2_scale = torch.empty(num_experts, hidden, intermediate // group_size, dtype=torch.bfloat16)

    # Materialize the BF16 weight we will compute against — round-tripped through
    # quantize → pack → unpack so the reference is what the kernel can attain.
    w1_dequant = torch.empty_like(w1_bf16)
    w3_dequant = torch.empty_like(w3_bf16)
    w2_dequant = torch.empty_like(w2_bf16)

    for e in range(num_experts):
        for src, packed_buf, scale_buf, dequant_buf, out_dim, in_dim in (
            (w1_bf16[e], w1_packed[e], w1_scale[e], w1_dequant[e], intermediate, hidden),
            (w3_bf16[e], w3_packed[e], w3_scale[e], w3_dequant[e], intermediate, hidden),
            (w2_bf16[e], w2_packed[e], w2_scale[e], w2_dequant[e], hidden, intermediate),
        ):
            q, scale = quantize_int4_per_group(src, group_size)
            packed_buf.copy_(pack_signed_int4_to_vllm_uint8(q))
            scale_buf.copy_(scale)
            dequant_buf.copy_(dequant_from_vllm_uint8(packed_buf, scale_buf, in_dim, group_size))

    # weight_shape: [num_experts, 2] u32 = (out_dim, in_dim) per expert
    w1_shape = torch.tensor([[intermediate, hidden]] * num_experts, dtype=torch.uint32).reshape(-1)
    w3_shape = w1_shape.clone()
    w2_shape = torch.tensor([[hidden, intermediate]] * num_experts, dtype=torch.uint32).reshape(-1)

    # Forward reference
    gate = grouped_matmul_bf16(expert_hidden, w1_dequant, expert_indptr.to(torch.int64))
    up = grouped_matmul_bf16(expert_hidden, w3_dequant, expert_indptr.to(torch.int64))
    activated = silu_mul_bf16(gate, up)
    expert_output = grouped_matmul_bf16(activated, w2_dequant, expert_indptr.to(torch.int64))

    tensors = {
        "expert_hidden": expert_hidden.contiguous(),
        "expert_indptr": expert_indptr.contiguous(),
        "w1_weight_packed": w1_packed.contiguous(),
        "w1_weight_scale": w1_scale.contiguous(),
        "w1_weight_shape": w1_shape.contiguous(),
        "w3_weight_packed": w3_packed.contiguous(),
        "w3_weight_scale": w3_scale.contiguous(),
        "w3_weight_shape": w3_shape.contiguous(),
        "w2_weight_packed": w2_packed.contiguous(),
        "w2_weight_scale": w2_scale.contiguous(),
        "w2_weight_shape": w2_shape.contiguous(),
        # Reference outputs.
        "ref_gate": gate.contiguous(),
        "ref_up": up.contiguous(),
        "ref_activated": activated.contiguous(),
        "ref_expert_output": expert_output.contiguous(),
        # For optional dequant-format probes.
        "ref_w1_dequant": w1_dequant.contiguous(),
        "ref_w3_dequant": w3_dequant.contiguous(),
        "ref_w2_dequant": w2_dequant.contiguous(),
    }
    metadata = {
        "num_experts": num_experts,
        "hidden": hidden,
        "intermediate": intermediate,
        "group_size": group_size,
        "tokens_per_expert": tokens_per_expert,
        "routed_tokens": routed_tokens,
        "weight_encoding": "signed_symmetric",
        "nibble_order": "low_then_high",
        "packed_dtype": "uint8",
        "scale_dtype": "bfloat16",
        "shape_dtype": "uint32",
        "activation": "silu_gate_mul_up",
        "accumulator_dtype": "float32",
        "seed": seed,
    }
    return tensors, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--kimi",
        action="store_true",
        help="use the full EP8 shape (48 experts, hidden=7168, intermediate=2048)",
    )
    parser.add_argument("--seed", type=int, default=0xC0FFEE)
    parser.add_argument(
        "--tokens-per-expert",
        type=int,
        nargs="+",
        help="override per-expert routed token counts (default: small balanced schedule)",
    )
    args = parser.parse_args()

    if args.kimi:
        num_experts = 48
        hidden = 7168
        intermediate = 2048
        default_tokens = [4, 0, 2, 1, 3, 0, 5, 2] + [1] * (num_experts - 8)
    else:
        num_experts = 4
        hidden = 128
        intermediate = 64
        default_tokens = [2, 0, 3, 1]

    tokens_per_expert = args.tokens_per_expert or default_tokens
    if len(tokens_per_expert) != num_experts:
        parser.error(
            f"--tokens-per-expert must have {num_experts} entries, got {len(tokens_per_expert)}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tensors, metadata = build_fixture(
        num_experts=num_experts,
        hidden=hidden,
        intermediate=intermediate,
        tokens_per_expert=tokens_per_expert,
        group_size=GROUP_SIZE,
        seed=args.seed,
    )

    fixture_path = args.out_dir / "kimi_int4_grouped.safetensors"
    save_file(tensors, fixture_path)
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"FIXTURE={fixture_path}")
    print(f"METADATA={args.out_dir / 'metadata.json'}")
    print(f"ROUTED_TOKENS={metadata['routed_tokens']}")


if __name__ == "__main__":
    main()
