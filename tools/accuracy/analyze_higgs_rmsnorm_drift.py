#!/usr/bin/env python3
"""Analyze Higgs/Qwen3 plain RMSNorm drift.

This diagnostic checks RMSNorm boundaries that are not covered by the
fused-add-RMSNorm helper:

* decoder layer input RMSNorm from a full-stage layer dump
* final model RMSNorm from the last decoder output to final_hidden

The golden tensors stay read-only oracles. Recompute variants use dumped inputs
and checkpoint weights to decide whether drift is explained by input drift or by
a real RMSNorm rounding semantic mismatch.
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


def stats(name: str, left: torch.Tensor, right: torch.Tensor) -> DriftStats:
    left = left.float().cpu()
    right = right.float().cpu()
    delta = left.flatten() - right.flatten()
    abs_delta = delta.abs()
    if left.numel() == 0:
        cosine = 1.0
    elif float(torch.linalg.vector_norm(left.flatten())) == 0.0 and float(
        torch.linalg.vector_norm(right.flatten())
    ) == 0.0:
        cosine = 1.0
    else:
        cosine = float(torch.nn.functional.cosine_similarity(left.flatten(), right.flatten(), dim=0))
    return DriftStats(
        name=name,
        shape=list(left.shape),
        max_abs=float(abs_delta.max()) if abs_delta.numel() else 0.0,
        mean_abs=float(abs_delta.mean()) if abs_delta.numel() else 0.0,
        p99_abs=float(torch.quantile(abs_delta, 0.99)) if abs_delta.numel() else 0.0,
        rmse=float(torch.sqrt(torch.mean(delta * delta))) if delta.numel() else 0.0,
        cosine=cosine,
    )


def checkpoint_vec(model_file: Path, key: str, device: str) -> torch.Tensor:
    with safe_open(str(model_file), framework="pt", device="cpu") as reader:
        if key not in reader.keys():
            raise KeyError(f"checkpoint is missing {key}")
        value = reader.get_tensor(key).to(torch.bfloat16).contiguous()
    return value.to(device=device)


def rmsnorm_variants(x_bf16: torch.Tensor, weight_bf16: torch.Tensor, eps: float) -> dict[str, torch.Tensor]:
    x = x_bf16.to(torch.bfloat16)
    w_bf16 = weight_bf16.to(torch.bfloat16)
    w_f32 = weight_bf16.float()
    xf = x.float()
    inv_rms = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    norm_f32 = xf * inv_rms
    return {
        "hf_like_bf16_mid": (norm_f32.to(torch.bfloat16) * w_bf16).to(torch.bfloat16),
        "single_round_fp32_weight": (norm_f32 * w_f32).to(torch.bfloat16),
        "bf16_mid_fp32_weight": (norm_f32.to(torch.bfloat16).float() * w_f32).to(torch.bfloat16),
    }


def add_variant_reports(
    payload: dict[str, object],
    *,
    boundary: str,
    golden_input: torch.Tensor,
    golden_output: torch.Tensor,
    actual_input: torch.Tensor,
    actual_output: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    reports = []
    golden_variants = rmsnorm_variants(golden_input, weight, eps)
    actual_variants = rmsnorm_variants(actual_input, weight, eps)
    for variant in sorted(golden_variants):
        reports.append(
            {
                "variant": variant,
                "golden_recompute_vs_golden_output": asdict(
                    stats(f"{boundary}.{variant}.golden_vs_golden", golden_variants[variant], golden_output)
                ),
                "actual_recompute_vs_actual_output": asdict(
                    stats(f"{boundary}.{variant}.actual_vs_actual", actual_variants[variant], actual_output)
                ),
                "actual_recompute_vs_golden_output": asdict(
                    stats(f"{boundary}.{variant}.actual_vs_golden", actual_variants[variant], golden_output)
                ),
            }
        )
    payload[boundary] = {
        "direct_input_actual_vs_golden": asdict(stats(f"{boundary}.input_actual_vs_golden", actual_input, golden_input)),
        "direct_output_actual_vs_golden": asdict(
            stats(f"{boundary}.output_actual_vs_golden", actual_output, golden_output)
        ),
        "variants": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-stage", help="HF full-stage golden safetensors for --layer-idx")
    parser.add_argument("--actual-stage", help="PegaInfer full-stage actual safetensors for --layer-idx")
    parser.add_argument("--golden-one-step", help="one-step or rich-trace golden with final_hidden.bf16")
    parser.add_argument("--actual-one-step", help="PegaInfer one-step actual with final_hidden.bf16")
    parser.add_argument("--model-safetensors", required=True)
    parser.add_argument("--layer-idx", type=int)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} requested but torch.cuda.is_available() is false")

    model_file = Path(args.model_safetensors)
    payload: dict[str, object] = {
        "model_safetensors": args.model_safetensors,
        "layer_idx": args.layer_idx,
        "eps": args.eps,
        "device": args.device,
    }

    if args.golden_stage and args.actual_stage:
        if args.layer_idx is None:
            raise ValueError("--layer-idx is required with stage dumps")
        golden_stage = load_file(args.golden_stage)
        actual_stage = load_file(args.actual_stage)
        layer = args.layer_idx
        weight = checkpoint_vec(model_file, f"body.layers.{layer}.input_layernorm.weight", args.device)
        add_variant_reports(
            payload,
            boundary=f"layer{layer}.input_norm",
            golden_input=golden_stage[f"layer{layer}.input_hidden.bf16"].to(args.device),
            golden_output=golden_stage[f"layer{layer}.input_norm.bf16"].to(args.device),
            actual_input=actual_stage[f"layer{layer}.input_hidden.bf16"].to(args.device),
            actual_output=actual_stage[f"layer{layer}.input_norm.bf16"].to(args.device),
            weight=weight,
            eps=args.eps,
        )

    if args.golden_stage and args.actual_stage and args.golden_one_step and args.actual_one_step:
        if args.layer_idx is None:
            raise ValueError("--layer-idx is required with final norm inputs")
        golden_stage = load_file(args.golden_stage)
        actual_stage = load_file(args.actual_stage)
        golden_one = load_file(args.golden_one_step)
        actual_one = load_file(args.actual_one_step)
        layer = args.layer_idx
        weight = checkpoint_vec(model_file, "body.norm.weight", args.device)
        add_variant_reports(
            payload,
            boundary="final_norm",
            golden_input=golden_stage[f"layer{layer}.output_hidden.bf16"].to(args.device),
            golden_output=golden_one["final_hidden.bf16"].to(args.device),
            actual_input=actual_stage[f"layer{layer}.output_hidden.bf16"].to(args.device),
            actual_output=actual_one["final_hidden.bf16"].to(args.device),
            weight=weight,
            eps=args.eps,
        )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    for boundary, report in payload.items():
        if not isinstance(report, dict) or "variants" not in report:
            continue
        direct = report["direct_output_actual_vs_golden"]
        print(
            f"{boundary}: direct_output_mean={direct['mean_abs']:.8f} "
            f"direct_output_cos={direct['cosine']:.9f}"
        )
        for variant in sorted(
            report["variants"],
            key=lambda item: item["actual_recompute_vs_actual_output"]["mean_abs"],
        ):
            actual_fit = variant["actual_recompute_vs_actual_output"]
            golden_fit = variant["golden_recompute_vs_golden_output"]
            actual_to_golden = variant["actual_recompute_vs_golden_output"]
            print(
                f"  {variant['variant']:<24} "
                f"golden_fit={golden_fit['mean_abs']:.8f} "
                f"actual_fit={actual_fit['mean_abs']:.8f} "
                f"actual_vs_golden={actual_to_golden['mean_abs']:.8f}"
            )


if __name__ == "__main__":
    main()
