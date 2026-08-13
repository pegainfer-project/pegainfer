#!/usr/bin/env python3
"""Analyze Higgs/Qwen3 q/k RMSNorm drift against a rich trace golden.

This is a diagnostic tool, not a fixer. It uses the trace as an oracle and
recomputes q/k RMSNorm from the dumped q/k projection tensors plus checkpoint
weights to separate:

1. projection/input drift that is amplified by RMSNorm, from
2. an actual implementation mismatch in the q/k RMSNorm kernel.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


@dataclass
class DriftStats:
    name: str
    shape: list[int]
    max_abs: float
    mean_abs: float
    p99_abs: float
    rmse: float
    cosine: float


@dataclass
class VariantReport:
    tensor: str
    variant: str
    recompute_from_golden_proj_vs_golden_norm: DriftStats
    recompute_from_actual_proj_vs_actual_norm: DriftStats
    recompute_from_actual_proj_vs_golden_norm: DriftStats


def stats(name: str, left: torch.Tensor, right: torch.Tensor) -> DriftStats:
    left = left.float()
    right = right.float()
    delta = left.flatten() - right.flatten()
    abs_delta = delta.abs()
    left_flat = left.flatten()
    right_flat = right.flatten()
    if left_flat.numel() == 0:
        cosine = 1.0
    elif float(torch.linalg.vector_norm(left_flat)) == 0.0 and float(
        torch.linalg.vector_norm(right_flat)
    ) == 0.0:
        cosine = 1.0
    else:
        cosine = float(torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=0))
    return DriftStats(
        name=name,
        shape=list(left.shape),
        max_abs=float(abs_delta.max()) if abs_delta.numel() else 0.0,
        mean_abs=float(abs_delta.mean()) if abs_delta.numel() else 0.0,
        p99_abs=float(torch.quantile(abs_delta, 0.99)) if abs_delta.numel() else 0.0,
        rmse=float(torch.sqrt(torch.mean(delta * delta))) if delta.numel() else 0.0,
        cosine=cosine,
    )


def last_token_from_trace(
    tensors: dict[str, torch.Tensor],
    name: str,
    actual_shape: torch.Size,
) -> torch.Tensor:
    tensor = tensors[name]
    if tuple(tensor.shape) == tuple(actual_shape):
        return tensor
    lengths = tensors["prompt.lengths"]
    if tensor.ndim >= 3:
        rows = [tensor[row, int(length) - 1] for row, length in enumerate(lengths.tolist())]
        last = torch.stack(rows, dim=0)
        if tuple(last.shape) == tuple(actual_shape):
            return last
        flat = last.reshape(last.shape[0], -1)
        if tuple(flat.shape) == tuple(actual_shape):
            return flat
    raise ValueError(f"cannot align {name}: golden={tuple(tensor.shape)} actual={tuple(actual_shape)}")


def checkpoint_weight(model_file: Path, layer_idx: int, q_or_k: str) -> torch.Tensor:
    key = f"body.layers.{layer_idx}.self_attn.{q_or_k}_norm.weight"
    with safe_open(str(model_file), framework="pt", device="cpu") as reader:
        if key not in reader.keys():
            raise KeyError(f"checkpoint is missing {key}")
        return reader.get_tensor(key).to(torch.bfloat16).contiguous()


def rmsnorm_variants(x: torch.Tensor, weight: torch.Tensor, head_dim: int, eps: float) -> dict[str, torch.Tensor]:
    x = x.reshape(x.shape[0], -1, head_dim)
    w_bf16 = weight.to(torch.bfloat16).reshape(1, 1, head_dim)
    w_f32 = weight.float().reshape(1, 1, head_dim)
    xf = x.float()
    inv_rms = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    norm_f32 = xf * inv_rms
    return {
        # Mirrors Transformers Qwen3RMSNorm: normalize in fp32, cast to input dtype,
        # then multiply by bf16 weight. PyTorch bf16 * bf16 yields bf16.
        "hf_like_bf16_mid": (norm_f32.to(torch.bfloat16) * w_bf16)
        .to(torch.bfloat16)
        .reshape(x.shape[0], -1),
        # One final round only. If this wins, the CUDA kernel is over-rounding.
        "single_round_fp32_weight": (norm_f32 * w_f32).to(torch.bfloat16).reshape(x.shape[0], -1),
        # Round normalized activations first, but multiply with fp32 weight.
        "bf16_mid_fp32_weight": (norm_f32.to(torch.bfloat16).float() * w_f32)
        .to(torch.bfloat16)
        .reshape(x.shape[0], -1),
        # Multiply before casting either operand back to bf16.
        "fp32_no_mid_round": (norm_f32 * w_bf16.float()).to(torch.bfloat16).reshape(x.shape[0], -1),
    }


def head_mean_abs(left: torch.Tensor, right: torch.Tensor, head_dim: int) -> list[float]:
    delta = (left.float() - right.float()).abs().reshape(left.shape[0], -1, head_dim)
    return [float(x) for x in delta.mean(dim=(0, 2)).tolist()]


def analyze_tensor(
    tensor: str,
    *,
    golden: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    weight: torch.Tensor,
    layer_idx: int,
    head_dim: int,
    eps: float,
) -> tuple[list[VariantReport], dict[str, object]]:
    proj_actual_name = f"layer{layer_idx}.{tensor}_proj.bf16"
    norm_actual_name = f"layer{layer_idx}.{tensor}_norm.bf16"
    proj_trace_name = f"layer.{layer_idx:02}.self_attn.{tensor}_proj.output.bf16"
    norm_trace_name = f"layer.{layer_idx:02}.self_attn.{tensor}_norm.output.bf16"

    actual_proj = actual[proj_actual_name].cpu().to(torch.bfloat16).contiguous()
    actual_norm = actual[norm_actual_name].cpu().to(torch.bfloat16).contiguous()
    golden_proj = last_token_from_trace(golden, proj_trace_name, actual_proj.shape).cpu().to(torch.bfloat16)
    golden_norm = last_token_from_trace(golden, norm_trace_name, actual_norm.shape).cpu().to(torch.bfloat16)

    reports = []
    golden_variants = rmsnorm_variants(golden_proj, weight, head_dim, eps)
    actual_variants = rmsnorm_variants(actual_proj, weight, head_dim, eps)
    for variant in sorted(golden_variants):
        reports.append(
            VariantReport(
                tensor=tensor,
                variant=variant,
                recompute_from_golden_proj_vs_golden_norm=stats(
                    f"{tensor}.{variant}.golden_proj_vs_golden_norm",
                    golden_variants[variant],
                    golden_norm,
                ),
                recompute_from_actual_proj_vs_actual_norm=stats(
                    f"{tensor}.{variant}.actual_proj_vs_actual_norm",
                    actual_variants[variant],
                    actual_norm,
                ),
                recompute_from_actual_proj_vs_golden_norm=stats(
                    f"{tensor}.{variant}.actual_proj_vs_golden_norm",
                    actual_variants[variant],
                    golden_norm,
                ),
            )
        )

    direct = {
        "proj_actual_vs_golden": asdict(stats(f"{tensor}.proj_actual_vs_golden", actual_proj, golden_proj)),
        "norm_actual_vs_golden": asdict(stats(f"{tensor}.norm_actual_vs_golden", actual_norm, golden_norm)),
        "norm_head_mean_abs": head_mean_abs(actual_norm, golden_norm, head_dim),
        "proj_head_mean_abs": head_mean_abs(actual_proj, golden_proj, head_dim),
    }
    return reports, direct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True, help="rich trace golden safetensors")
    parser.add_argument("--actual", required=True, help="PegaInfer layer-stage dump safetensors")
    parser.add_argument("--model-safetensors", required=True, help="Higgs model.safetensors")
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    golden = load_file(args.golden)
    actual = load_file(args.actual)
    all_reports: list[VariantReport] = []
    direct: dict[str, object] = {}
    for tensor in ("q", "k"):
        reports, tensor_direct = analyze_tensor(
            tensor,
            golden=golden,
            actual=actual,
            weight=checkpoint_weight(Path(args.model_safetensors), args.layer_idx, tensor),
            layer_idx=args.layer_idx,
            head_dim=args.head_dim,
            eps=args.eps,
        )
        all_reports.extend(reports)
        direct[tensor] = tensor_direct

    payload = {
        "golden": args.golden,
        "actual": args.actual,
        "model_safetensors": args.model_safetensors,
        "layer_idx": args.layer_idx,
        "head_dim": args.head_dim,
        "eps": args.eps,
        "direct": direct,
        "variants": [asdict(report) for report in all_reports],
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"layer={args.layer_idx} head_dim={args.head_dim} eps={args.eps}")
    for tensor, tensor_direct in direct.items():
        proj = tensor_direct["proj_actual_vs_golden"]
        norm = tensor_direct["norm_actual_vs_golden"]
        print(
            f"{tensor}: proj_mean={proj['mean_abs']:.8f} norm_mean={norm['mean_abs']:.8f} "
            f"proj_p99={proj['p99_abs']:.8f} norm_p99={norm['p99_abs']:.8f}"
        )
    print("variant ranking by actual_proj_vs_actual_norm mean_abs:")
    for report in sorted(
        all_reports,
        key=lambda item: item.recompute_from_actual_proj_vs_actual_norm.mean_abs,
    ):
        a = report.recompute_from_actual_proj_vs_actual_norm
        g = report.recompute_from_golden_proj_vs_golden_norm
        propagated = report.recompute_from_actual_proj_vs_golden_norm
        print(
            f"{report.tensor}.{report.variant}: "
            f"actual_kernel_mean={a.mean_abs:.8f} golden_formula_mean={g.mean_abs:.8f} "
            f"propagated_vs_golden_mean={propagated.mean_abs:.8f}"
        )


if __name__ == "__main__":
    main()
