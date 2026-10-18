#!/usr/bin/env python3
"""Compare PegaInfer Kimi-K2 logits against an external HF logits fixture.

The reference directory must be produced by hf_logits_reference.py. The
candidate safetensors file must contain a full-vocab FP32 logits tensor, by
default named logits_f32. This script is deliberately not a reference generator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-tensor", default="logits_f32")
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--max-abs-diff", type=float)
    parser.add_argument("--mean-abs-diff", type=float)
    parser.add_argument("--require-argmax", action="store_true")
    parser.add_argument("--require-topk-order", action="store_true")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    metadata_path = args.reference_dir / "metadata.json"
    reference_path = args.reference_dir / "reference.safetensors"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    if not args.candidate.exists():
        raise FileNotFoundError(args.candidate)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("engine") != "hf_remote_code":
        raise ValueError(
            f"raw logits comparison requires hf_remote_code reference, got {metadata.get('engine')}"
        )

    reference = load_file(str(reference_path), device="cpu")
    candidate = load_file(str(args.candidate), device="cpu")
    if args.candidate_tensor not in candidate:
        raise KeyError(f"candidate tensor {args.candidate_tensor!r} not found")

    ref_logits = reference["logits_f32"].float().reshape(-1)
    cand_logits = candidate[args.candidate_tensor].float().reshape(-1)
    if ref_logits.shape != cand_logits.shape:
        raise ValueError(
            f"logits shape mismatch: reference={tuple(ref_logits.shape)} "
            f"candidate={tuple(cand_logits.shape)}"
        )

    vocab_size = int(ref_logits.numel())
    top_k = min(args.top_k, vocab_size)
    ref_top_vals, ref_top_ids = torch.topk(ref_logits, k=top_k)
    cand_top_vals, cand_top_ids = torch.topk(cand_logits, k=top_k)
    diff = (cand_logits - ref_logits).abs()
    ref_argmax = int(ref_top_ids[0].item())
    cand_argmax = int(cand_top_ids[0].item())
    overlap = len(set(ref_top_ids.tolist()) & set(cand_top_ids.tolist()))
    top_diff_ids = torch.topk(diff, k=min(32, vocab_size)).indices

    result = {
        "reference_dir": str(args.reference_dir),
        "candidate": str(args.candidate),
        "candidate_tensor": args.candidate_tensor,
        "seq_len": metadata.get("seq_len"),
        "vocab_size": vocab_size,
        "top_k": top_k,
        "argmax": {
            "reference": ref_argmax,
            "candidate": cand_argmax,
            "match": ref_argmax == cand_argmax,
        },
        "topk": {
            "order_match": ref_top_ids.tolist() == cand_top_ids.tolist(),
            "overlap": overlap,
            "reference_ids": [int(v) for v in ref_top_ids.tolist()],
            "candidate_ids": [int(v) for v in cand_top_ids.tolist()],
            "reference_logits": [float(v) for v in ref_top_vals.tolist()],
            "candidate_logits": [float(v) for v in cand_top_vals.tolist()],
        },
        "diff": {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "top_abs_diff_token_ids": [int(v) for v in top_diff_ids.tolist()],
            "top_abs_diff_values": [float(diff[i].item()) for i in top_diff_ids],
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    failures: list[str] = []
    if args.require_argmax and not result["argmax"]["match"]:
        failures.append("argmax mismatch")
    if args.require_topk_order and not result["topk"]["order_match"]:
        failures.append("top-k order mismatch")
    if args.max_abs_diff is not None and result["diff"]["max_abs"] > args.max_abs_diff:
        failures.append(
            f"max_abs_diff {result['diff']['max_abs']} > threshold {args.max_abs_diff}"
        )
    if args.mean_abs_diff is not None and result["diff"]["mean_abs"] > args.mean_abs_diff:
        failures.append(
            f"mean_abs_diff {result['diff']['mean_abs']} > threshold {args.mean_abs_diff}"
        )
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
