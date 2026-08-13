#!/usr/bin/env python3
"""Analyze Higgs/Qwen3 projection drift against full layer-stage goldens.

The trace is an oracle only. This tool recomputes linear projections from
golden and actual stage inputs plus checkpoint weights, then checks whether
PegaInfer's actual GEMM outputs are explained by the same weight/input pair or
whether the projection kernel/storage boundary itself is suspicious.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
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
class ProjectionReport:
    projection: str
    input_stage: str
    output_stage: str
    weight_name: str
    input_actual_vs_golden: DriftStats
    output_actual_vs_golden: DriftStats
    recompute_golden_input_vs_golden_output: DriftStats
    recompute_actual_input_vs_actual_output: DriftStats
    recompute_actual_input_vs_golden_output: DriftStats
    amplification_mean_abs: float


PROJECTIONS = {
    "q_proj": ("input_norm", "q_proj", "self_attn.q_proj.weight"),
    "k_proj": ("input_norm", "k_proj", "self_attn.k_proj.weight"),
    "v_proj": ("input_norm", "v_proj", "self_attn.v_proj.weight"),
    "o_proj": ("attn_output", "o_proj", "self_attn.o_proj.weight"),
    "gate_proj": ("post_attn_norm", "gate_proj", "mlp.gate_proj.weight"),
    "up_proj": ("post_attn_norm", "up_proj", "mlp.up_proj.weight"),
    "down_proj": ("silu_mul", "down_proj", "mlp.down_proj.weight"),
}


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


def checkpoint_weight(model_file: Path, layer_idx: int, suffix: str, device: str) -> tuple[str, torch.Tensor]:
    key = f"body.layers.{layer_idx}.{suffix}"
    with safe_open(str(model_file), framework="pt", device="cpu") as reader:
        if key not in reader.keys():
            raise KeyError(f"checkpoint is missing {key}")
        weight = reader.get_tensor(key).to(torch.bfloat16).contiguous()
    return key, weight.to(device=device)


def stage_name(layer_idx: int, suffix: str) -> str:
    return f"layer{layer_idx}.{suffix}.bf16"


def linear_bf16(input_row: torch.Tensor, weight: torch.Tensor, device: str) -> torch.Tensor:
    x = input_row.to(device=device, dtype=torch.bfloat16).contiguous()
    y = F.linear(x, weight)
    return y.to(torch.bfloat16).cpu().contiguous()


def analyze_projection(
    name: str,
    *,
    layer_idx: int,
    golden: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    model_file: Path,
    device: str,
) -> ProjectionReport:
    input_suffix, output_suffix, weight_suffix = PROJECTIONS[name]
    input_stage = stage_name(layer_idx, input_suffix)
    output_stage = stage_name(layer_idx, output_suffix)
    weight_name, weight = checkpoint_weight(model_file, layer_idx, weight_suffix, device)

    golden_input = golden[input_stage].cpu().to(torch.bfloat16).contiguous()
    actual_input = actual[input_stage].cpu().to(torch.bfloat16).contiguous()
    golden_output = golden[output_stage].cpu().to(torch.bfloat16).contiguous()
    actual_output = actual[output_stage].cpu().to(torch.bfloat16).contiguous()

    recompute_golden = linear_bf16(golden_input, weight, device)
    recompute_actual = linear_bf16(actual_input, weight, device)

    input_drift = stats(f"{name}.input_actual_vs_golden", actual_input, golden_input)
    output_drift = stats(f"{name}.output_actual_vs_golden", actual_output, golden_output)
    input_mean = max(input_drift.mean_abs, 1e-12)
    return ProjectionReport(
        projection=name,
        input_stage=input_stage,
        output_stage=output_stage,
        weight_name=weight_name,
        input_actual_vs_golden=input_drift,
        output_actual_vs_golden=output_drift,
        recompute_golden_input_vs_golden_output=stats(
            f"{name}.recompute_golden_input_vs_golden_output",
            recompute_golden,
            golden_output,
        ),
        recompute_actual_input_vs_actual_output=stats(
            f"{name}.recompute_actual_input_vs_actual_output",
            recompute_actual,
            actual_output,
        ),
        recompute_actual_input_vs_golden_output=stats(
            f"{name}.recompute_actual_input_vs_golden_output",
            recompute_actual,
            golden_output,
        ),
        amplification_mean_abs=output_drift.mean_abs / input_mean,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True, help="full layer-stage golden safetensors")
    parser.add_argument("--actual", required=True, help="PegaInfer layer-stage actual safetensors")
    parser.add_argument("--model-safetensors", required=True)
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--projection", action="append", choices=sorted(PROJECTIONS))
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} requested but torch.cuda.is_available() is false")

    golden = load_file(args.golden)
    actual = load_file(args.actual)
    selected = args.projection or list(PROJECTIONS)
    reports = [
        analyze_projection(
            projection,
            layer_idx=args.layer_idx,
            golden=golden,
            actual=actual,
            model_file=Path(args.model_safetensors),
            device=args.device,
        )
        for projection in selected
    ]

    payload = {
        "golden": args.golden,
        "actual": args.actual,
        "model_safetensors": args.model_safetensors,
        "layer_idx": args.layer_idx,
        "device": args.device,
        "reports": [asdict(report) for report in reports],
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"layer={args.layer_idx} device={args.device}")
    print(
        "projection           input_mean   output_mean  recompute_actual_vs_actual  "
        "recompute_golden_vs_golden  amp"
    )
    for report in reports:
        print(
            f"{report.projection:18} "
            f"{report.input_actual_vs_golden.mean_abs:10.8f} "
            f"{report.output_actual_vs_golden.mean_abs:11.8f} "
            f"{report.recompute_actual_input_vs_actual_output.mean_abs:26.8f} "
            f"{report.recompute_golden_input_vs_golden_output.mean_abs:26.8f} "
            f"{report.amplification_mean_abs:6.2f}"
        )


if __name__ == "__main__":
    main()
