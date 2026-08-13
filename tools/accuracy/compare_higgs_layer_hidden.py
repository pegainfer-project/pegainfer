#!/usr/bin/env python3
"""Compare Higgs/Qwen3 per-layer hidden snapshots from HF and PegaInfer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

NUM_LAYERS = 36


@dataclass
class DriftStats:
    name: str
    max_abs: float
    mean_abs: float
    p99_abs: float
    rmse: float
    cosine: float


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            left.float().flatten(), right.float().flatten(), dim=0
        )
    )


def quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), q))


def stats(name: str, golden: torch.Tensor, actual: torch.Tensor) -> DriftStats:
    if tuple(golden.shape) != tuple(actual.shape):
        raise ValueError(f"{name} shape mismatch: golden {tuple(golden.shape)} actual {tuple(actual.shape)}")
    delta = actual.float().flatten() - golden.float().flatten()
    abs_delta = delta.abs()
    return DriftStats(
        name=name,
        max_abs=float(abs_delta.max()),
        mean_abs=float(abs_delta.mean()),
        p99_abs=quantile(abs_delta, 0.99),
        rmse=float(torch.sqrt(torch.mean(delta * delta))),
        cosine=cosine(golden, actual),
    )


def prompt_exact(golden: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> bool:
    for name in ("prompt.input_ids_padded", "prompt.attention_mask", "prompt.lengths"):
        if not torch.equal(golden[name], actual[name]):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--mean-alert", type=float, default=0.003)
    parser.add_argument("--cosine-alert", type=float, default=0.9998)
    args = parser.parse_args()

    golden = load_file(str(args.golden), device="cpu")
    actual = load_file(str(args.actual), device="cpu")
    print(f"prompt_exact={prompt_exact(golden, actual)}")
    print("name                         max_abs   mean_abs    p99_abs       rmse      cosine")

    rows: list[DriftStats] = []
    rows.append(stats("embedding.last_hidden.bf16", golden["embedding.last_hidden.bf16"], actual["embedding.last_hidden.bf16"]))
    for layer_idx in range(NUM_LAYERS):
        name = f"layer.{layer_idx:02}.last_hidden.bf16"
        rows.append(stats(name, golden[name], actual[name]))
    rows.append(stats("final_hidden.bf16", golden["final_hidden.bf16"], actual["final_hidden.bf16"]))

    first_mean_alert = None
    first_cos_alert = None
    for row in rows:
        print(
            f"{row.name:28} {row.max_abs:8.6f} {row.mean_abs:10.6f} "
            f"{row.p99_abs:10.6f} {row.rmse:10.6f} {row.cosine:12.9f}"
        )
        if first_mean_alert is None and row.mean_abs > args.mean_alert:
            first_mean_alert = row.name
        if first_cos_alert is None and row.cosine < args.cosine_alert:
            first_cos_alert = row.name

    print("summary:")
    print(f"  first_mean_abs_gt_{args.mean_alert:.6f}={first_mean_alert or 'none'}")
    print(f"  first_cosine_lt_{args.cosine_alert:.9f}={first_cos_alert or 'none'}")
    worst_mean = max(rows, key=lambda row: row.mean_abs)
    worst_cosine = min(rows, key=lambda row: row.cosine)
    print(f"  worst_mean_abs={worst_mean.name}:{worst_mean.mean_abs:.6f}")
    print(f"  worst_cosine={worst_cosine.name}:{worst_cosine.cosine:.9f}")


if __name__ == "__main__":
    main()
