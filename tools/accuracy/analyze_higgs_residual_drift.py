#!/usr/bin/env python3
"""Analyze Higgs/Qwen3 residual add and fused-add-RMSNorm drift.

This diagnostic checks the two residual boundaries in a decoder layer:

1. input_hidden + o_proj -> post_attn_norm
2. rounded_attn_residual + down_proj -> output_hidden

The trace/golden tensors remain oracles only; recomputation uses the recorded
stage inputs and checkpoint weights to explain drift without injecting golden
values into runtime code.
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
    variant: str
    recompute_golden_post_attn_norm_vs_golden: DriftStats
    recompute_actual_post_attn_norm_vs_actual: DriftStats
    recompute_actual_post_attn_norm_vs_golden: DriftStats


def stats(name: str, left: torch.Tensor, right: torch.Tensor) -> DriftStats:
    left = left.float().cpu()
    right = right.float().cpu()
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


def stage(layer_idx: int, suffix: str) -> str:
    return f"layer{layer_idx}.{suffix}.bf16"


def checkpoint_vec(model_file: Path, layer_idx: int, suffix: str, device: str) -> torch.Tensor:
    key = f"body.layers.{layer_idx}.{suffix}"
    with safe_open(str(model_file), framework="pt", device="cpu") as reader:
        if key not in reader.keys():
            raise KeyError(f"checkpoint is missing {key}")
        value = reader.get_tensor(key).to(torch.bfloat16).contiguous()
    return value.to(device=device)


def rmsnorm_variants(x_bf16: torch.Tensor, weight_bf16: torch.Tensor, eps: float) -> dict[str, torch.Tensor]:
    x = x_bf16.float()
    w_bf16 = weight_bf16.to(torch.bfloat16)
    w_f32 = weight_bf16.float()
    inv_rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    norm_f32 = x * inv_rms
    return {
        # Mirrors Transformers Qwen3RMSNorm.
        "hf_like_bf16_mid": (norm_f32.to(torch.bfloat16) * w_bf16).to(torch.bfloat16),
        # Mirrors pegainfer::norm::FusedAddRMSNormRoundKernel's visible formula:
        # rounded add feeds RMS reduction, then norm * weight is rounded at store.
        "fused_round_formula": (norm_f32 * w_f32).to(torch.bfloat16),
        "bf16_mid_fp32_weight": (norm_f32.to(torch.bfloat16).float() * w_f32).to(torch.bfloat16),
    }


def residual_sum_variants(left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
    left = left.to(torch.bfloat16)
    right = right.to(torch.bfloat16)
    return {
        "bf16_add": (left + right).to(torch.bfloat16),
        "fp32_add_then_bf16": (left.float() + right.float()).to(torch.bfloat16),
    }


def analyze_side(
    prefix: str,
    tensors: dict[str, torch.Tensor],
    *,
    layer_idx: int,
    post_weight: torch.Tensor,
    eps: float,
    device: str,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    input_hidden = tensors[stage(layer_idx, "input_hidden")].to(device=device, dtype=torch.bfloat16)
    o_proj = tensors[stage(layer_idx, "o_proj")].to(device=device, dtype=torch.bfloat16)
    down_proj = tensors[stage(layer_idx, "down_proj")].to(device=device, dtype=torch.bfloat16)

    residual_variants = residual_sum_variants(input_hidden, o_proj)
    post_norm_variants = {}
    output_variants = {}
    for add_name, attn_residual in residual_variants.items():
        for norm_name, post_norm in rmsnorm_variants(attn_residual, post_weight, eps).items():
            post_norm_variants[f"{add_name}+{norm_name}"] = post_norm.cpu().contiguous()
        for mlp_add_name, output_hidden in residual_sum_variants(attn_residual, down_proj).items():
            output_variants[f"{add_name}+{mlp_add_name}"] = output_hidden.cpu().contiguous()
    return {
        f"{prefix}_attn_residual": {k: v.cpu().contiguous() for k, v in residual_variants.items()},
        f"{prefix}_post_norm": post_norm_variants,
        f"{prefix}_output_hidden": output_variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True, help="full layer-stage golden safetensors")
    parser.add_argument("--actual", required=True, help="PegaInfer layer-stage actual safetensors")
    parser.add_argument("--model-safetensors", required=True)
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} requested but torch.cuda.is_available() is false")

    golden = load_file(args.golden)
    actual = load_file(args.actual)
    post_weight = checkpoint_vec(
        Path(args.model_safetensors),
        args.layer_idx,
        "post_attention_layernorm.weight",
        args.device,
    )
    golden_recompute = analyze_side(
        "golden", golden, layer_idx=args.layer_idx, post_weight=post_weight, eps=args.eps, device=args.device
    )
    actual_recompute = analyze_side(
        "actual", actual, layer_idx=args.layer_idx, post_weight=post_weight, eps=args.eps, device=args.device
    )
    golden_post = golden[stage(args.layer_idx, "post_attn_norm")].cpu().to(torch.bfloat16)
    actual_post = actual[stage(args.layer_idx, "post_attn_norm")].cpu().to(torch.bfloat16)
    golden_output = golden[stage(args.layer_idx, "output_hidden")].cpu().to(torch.bfloat16)
    actual_output = actual[stage(args.layer_idx, "output_hidden")].cpu().to(torch.bfloat16)

    post_reports = []
    for variant in sorted(golden_recompute["golden_post_norm"]):
        post_reports.append(
            VariantReport(
                variant=variant,
                recompute_golden_post_attn_norm_vs_golden=stats(
                    f"{variant}.golden_post_norm_vs_golden",
                    golden_recompute["golden_post_norm"][variant],
                    golden_post,
                ),
                recompute_actual_post_attn_norm_vs_actual=stats(
                    f"{variant}.actual_post_norm_vs_actual",
                    actual_recompute["actual_post_norm"][variant],
                    actual_post,
                ),
                recompute_actual_post_attn_norm_vs_golden=stats(
                    f"{variant}.actual_post_norm_vs_golden",
                    actual_recompute["actual_post_norm"][variant],
                    golden_post,
                ),
            )
        )

    output_reports = []
    for variant in sorted(golden_recompute["golden_output_hidden"]):
        output_reports.append(
            {
                "variant": variant,
                "recompute_golden_output_vs_golden": asdict(
                    stats(
                        f"{variant}.golden_output_vs_golden",
                        golden_recompute["golden_output_hidden"][variant],
                        golden_output,
                    )
                ),
                "recompute_actual_output_vs_actual": asdict(
                    stats(
                        f"{variant}.actual_output_vs_actual",
                        actual_recompute["actual_output_hidden"][variant],
                        actual_output,
                    )
                ),
                "recompute_actual_output_vs_golden": asdict(
                    stats(
                        f"{variant}.actual_output_vs_golden",
                        actual_recompute["actual_output_hidden"][variant],
                        golden_output,
                    )
                ),
            }
        )

    direct = {
        "o_proj_actual_vs_golden": asdict(
            stats(stage(args.layer_idx, "o_proj"), actual[stage(args.layer_idx, "o_proj")], golden[stage(args.layer_idx, "o_proj")])
        ),
        "post_attn_norm_actual_vs_golden": asdict(stats(stage(args.layer_idx, "post_attn_norm"), actual_post, golden_post)),
        "down_proj_actual_vs_golden": asdict(
            stats(stage(args.layer_idx, "down_proj"), actual[stage(args.layer_idx, "down_proj")], golden[stage(args.layer_idx, "down_proj")])
        ),
        "output_hidden_actual_vs_golden": asdict(stats(stage(args.layer_idx, "output_hidden"), actual_output, golden_output)),
    }
    payload = {
        "golden": args.golden,
        "actual": args.actual,
        "model_safetensors": args.model_safetensors,
        "layer_idx": args.layer_idx,
        "eps": args.eps,
        "device": args.device,
        "direct": direct,
        "post_attn_norm_variants": [asdict(report) for report in post_reports],
        "output_hidden_variants": output_reports,
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"layer={args.layer_idx} device={args.device} eps={args.eps}")
    print(
        "post_variant                         golden_vs_golden  actual_vs_actual  "
        "actual_vs_golden"
    )
    for report in sorted(
        post_reports,
        key=lambda item: item.recompute_actual_post_attn_norm_vs_actual.mean_abs,
    ):
        print(
            f"{report.variant:36} "
            f"{report.recompute_golden_post_attn_norm_vs_golden.mean_abs:16.8f} "
            f"{report.recompute_actual_post_attn_norm_vs_actual.mean_abs:16.8f} "
            f"{report.recompute_actual_post_attn_norm_vs_golden.mean_abs:16.8f}"
        )
    print("output_variant                       golden_vs_golden  actual_vs_actual  actual_vs_golden")
    for report in sorted(output_reports, key=lambda item: item["recompute_actual_output_vs_actual"]["mean_abs"]):
        print(
            f"{report['variant']:36} "
            f"{report['recompute_golden_output_vs_golden']['mean_abs']:16.8f} "
            f"{report['recompute_actual_output_vs_actual']['mean_abs']:16.8f} "
            f"{report['recompute_actual_output_vs_golden']['mean_abs']:16.8f}"
        )


if __name__ == "__main__":
    main()
