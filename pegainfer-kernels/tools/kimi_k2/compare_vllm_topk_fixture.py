#!/usr/bin/env python3
"""Compare PegaInfer Kimi-K2 full-vocab logits with a vLLM top-logprobs fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def candidate_path_for_case(candidate_root: Path, case_id: str) -> Path:
    cases_path = candidate_root / "cases.json"
    if cases_path.exists():
        cases = load_json(cases_path).get("cases")
        if isinstance(cases, list):
            for item in cases:
                if isinstance(item, dict) and item.get("id") == case_id:
                    candidate = item.get("candidate")
                    if isinstance(candidate, str):
                        return Path(candidate)
                    candidate_dir = item.get("candidate_dir")
                    if isinstance(candidate_dir, str):
                        return Path(candidate_dir) / "candidate.safetensors"
    return candidate_root / case_id / "candidate.safetensors"


def compare_one(
    reference_dir: Path,
    candidate_path: Path,
    *,
    candidate_tensor: str,
    top_k: int,
) -> dict[str, Any]:
    metadata_path = reference_dir / "metadata.json"
    reference_path = reference_dir / "reference.safetensors"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)

    metadata = load_json(metadata_path)
    if metadata.get("engine") != "vllm":
        raise ValueError(f"vLLM comparison requires engine=vllm, got {metadata.get('engine')}")

    reference = load_file(str(reference_path), device="cpu")
    candidate = load_file(str(candidate_path), device="cpu")
    if candidate_tensor not in candidate:
        raise KeyError(f"candidate tensor {candidate_tensor!r} not found in {candidate_path}")

    ref_ids = reference["top_logprob_ids"].to(torch.int64).reshape(-1)
    ref_logprobs = reference["top_logprobs_f32"].float().reshape(-1)
    generated = reference["generated_token_ids"].to(torch.int64).reshape(-1)
    cand_logits = candidate[candidate_tensor].float().reshape(-1)
    if ref_ids.numel() != ref_logprobs.numel():
        raise ValueError("reference top_logprob_ids/top_logprobs_f32 length mismatch")
    if generated.numel() == 0:
        raise ValueError("reference generated_token_ids is empty")
    if ref_ids.numel() == 0:
        raise ValueError("reference top_logprob_ids is empty")

    vocab_size = int(cand_logits.numel())
    top_k = min(top_k, vocab_size, int(ref_ids.numel()))
    cand_top_vals, cand_top_ids = torch.topk(cand_logits, k=top_k)
    ref_top_ids = ref_ids[:top_k]
    ref_top_logprobs = ref_logprobs[:top_k]
    generated_token = int(generated[0].item())
    cand_argmax = int(cand_top_ids[0].item())
    cand_logprobs = cand_logits - torch.logsumexp(cand_logits, dim=0)
    cand_at_ref_logprobs = cand_logprobs[ref_top_ids]
    logprob_diff = (cand_at_ref_logprobs - ref_top_logprobs).abs()
    overlap = len(set(int(v) for v in ref_top_ids.tolist()) & set(int(v) for v in cand_top_ids.tolist()))

    return {
        "reference_dir": str(reference_dir),
        "candidate": str(candidate_path),
        "candidate_tensor": candidate_tensor,
        "case_id": metadata.get("case_id"),
        "seq_len": metadata.get("seq_len"),
        "vocab_size": vocab_size,
        "top_k": top_k,
        "argmax": {
            "vllm_generated": generated_token,
            "vllm_top": int(ref_ids[0].item()),
            "candidate": cand_argmax,
            "match": generated_token == cand_argmax,
        },
        "topk": {
            "overlap": overlap,
            "order_match": ref_top_ids.tolist() == cand_top_ids.tolist(),
            "vllm_ids": [int(v) for v in ref_top_ids.tolist()],
            "candidate_ids": [int(v) for v in cand_top_ids.tolist()],
            "candidate_logits": [float(v) for v in cand_top_vals.tolist()],
            "vllm_logprobs": [float(v) for v in ref_top_logprobs.tolist()],
            "candidate_logprobs_at_vllm_ids": [
                float(v) for v in cand_at_ref_logprobs.tolist()
            ],
        },
        "logprob_diff_at_vllm_topk": {
            "max_abs": float(logprob_diff.max().item()),
            "mean_abs": float(logprob_diff.mean().item()),
        },
    }


def fail_if_needed(
    results: list[dict[str, Any]],
    *,
    require_argmax: bool,
    min_overlap: int | None,
    max_ref_logprob_abs_diff: float | None,
) -> None:
    failures: list[str] = []
    for result in results:
        case = result.get("case_id") or result["reference_dir"]
        if require_argmax and not result["argmax"]["match"]:
            failures.append(f"{case}: argmax mismatch")
        if min_overlap is not None and result["topk"]["overlap"] < min_overlap:
            failures.append(
                f"{case}: top-k overlap {result['topk']['overlap']} < {min_overlap}"
            )
        if (
            max_ref_logprob_abs_diff is not None
            and result["logprob_diff_at_vllm_topk"]["max_abs"] > max_ref_logprob_abs_diff
        ):
            failures.append(
                f"{case}: max top-k logprob diff "
                f"{result['logprob_diff_at_vllm_topk']['max_abs']} "
                f"> {max_ref_logprob_abs_diff}"
            )
    if failures:
        raise SystemExit("; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--candidate-tensor", default="logits_f32")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--require-argmax", action="store_true")
    parser.add_argument("--min-overlap", type=int)
    parser.add_argument("--max-ref-logprob-abs-diff", type=float)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    batch_mode = args.reference_root is not None or args.candidate_root is not None
    single_mode = args.reference_dir is not None or args.candidate is not None
    if batch_mode == single_mode:
        raise ValueError(
            "use either --reference-dir/--candidate or --reference-root/--candidate-root"
        )

    if single_mode:
        if args.reference_dir is None or args.candidate is None:
            raise ValueError("--reference-dir and --candidate must be provided together")
        results = [
            compare_one(
                args.reference_dir,
                args.candidate,
                candidate_tensor=args.candidate_tensor,
                top_k=args.top_k,
            )
        ]
    else:
        if args.reference_root is None or args.candidate_root is None:
            raise ValueError("--reference-root and --candidate-root must be provided together")
        cases_payload = load_json(args.reference_root / "cases.json")
        cases = cases_payload.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("reference-root cases.json must contain a non-empty cases list")
        results = []
        for item in cases:
            if not isinstance(item, dict):
                raise ValueError("each reference case must be an object")
            case_id = item.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"invalid case id in {item!r}")
            reference_dir = Path(item.get("reference_dir", args.reference_root / case_id))
            candidate = candidate_path_for_case(args.candidate_root, case_id)
            results.append(
                compare_one(
                    reference_dir,
                    candidate,
                    candidate_tensor=args.candidate_tensor,
                    top_k=args.top_k,
                )
            )

    summary = {
        "cases": results,
        "all_argmax_match": all(result["argmax"]["match"] for result in results),
        "min_overlap": min(result["topk"]["overlap"] for result in results),
        "max_logprob_abs_diff_at_vllm_topk": max(
            result["logprob_diff_at_vllm_topk"]["max_abs"] for result in results
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fail_if_needed(
        results,
        require_argmax=args.require_argmax,
        min_overlap=args.min_overlap,
        max_ref_logprob_abs_diff=args.max_ref_logprob_abs_diff,
    )


if __name__ == "__main__":
    main()
