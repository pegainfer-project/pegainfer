#!/usr/bin/env python3
"""Analyze a Higgs Audio one-step PegaInfer actual dump.

This is a diagnostic companion to `higgs_compare_one_step`: it explains where
the current actual-vs-golden drift comes from instead of only saying pass/fail.
In particular it separates:

* Qwen3 body hidden-state drift.
* Audio-head implementation drift from CPU fp32 dot vs CUDA bf16 F.linear.
* Top-k instability vs stable argmax decisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

AUDIO_HEAD_KEY = "tied.embedding.modality_embeddings.0.embedding.weight"
CODEBOOKS = 8
VOCAB = 1026


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), q))


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            left.float().flatten(), right.float().flatten(), dim=0
        )
    )


def print_stats(name: str, golden: torch.Tensor, actual: torch.Tensor) -> None:
    delta = (actual.float() - golden.float()).flatten()
    abs_delta = delta.abs()
    print(
        f"{name}: "
        f"max={float(abs_delta.max()):.6f} "
        f"mean={float(abs_delta.mean()):.6f} "
        f"p99={_quantile(abs_delta, 0.99):.6f} "
        f"rmse={float(torch.sqrt(torch.mean(delta * delta))):.6f} "
        f"cos={_cosine(golden, actual):.9f}"
    )


def load_audio_head(model_dir: Path) -> torch.Tensor:
    shard = model_dir / "model.safetensors"
    with safe_open(str(shard), framework="pt", device="cpu") as tensors:
        if AUDIO_HEAD_KEY not in tensors.keys():
            raise KeyError(f"missing {AUDIO_HEAD_KEY} in {shard}")
        weight = tensors.get_tensor(AUDIO_HEAD_KEY)
    expected = (CODEBOOKS * VOCAB, 2560)
    if tuple(weight.shape) != expected:
        raise ValueError(f"{AUDIO_HEAD_KEY} shape {tuple(weight.shape)} != {expected}")
    return weight


def topk_overlap(golden_ids: torch.Tensor, actual_ids: torch.Tensor) -> list[int]:
    overlaps: list[int] = []
    for cb in range(golden_ids.shape[1]):
        golden_set = set(int(v) for v in golden_ids[0, cb].tolist())
        actual_set = set(int(v) for v in actual_ids[0, cb].tolist())
        overlaps.append(len(golden_set & actual_set))
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    golden = load_file(str(args.golden), device="cpu")
    actual = load_file(str(args.actual), device="cpu")
    audio_head = load_audio_head(args.model_dir)

    print("schema:")
    for name in (
        "prompt.input_ids_padded",
        "prompt.attention_mask",
        "prompt.lengths",
        "final_hidden.bf16",
        "audio_logits.f32",
        "audio_argmax.ids",
    ):
        print(
            f"  {name}: golden={tuple(golden[name].shape)} {golden[name].dtype} "
            f"actual={tuple(actual[name].shape)} {actual[name].dtype}"
        )

    print("\nprimary drift:")
    print_stats("final_hidden.bf16", golden["final_hidden.bf16"], actual["final_hidden.bf16"])
    print_stats("audio_logits.f32", golden["audio_logits.f32"], actual["audio_logits.f32"])

    hidden_delta = (
        actual["final_hidden.bf16"].float().flatten()
        - golden["final_hidden.bf16"].float().flatten()
    )
    top_idx = torch.topk(hidden_delta.abs(), 12).indices
    print("\nfinal_hidden top abs deltas:")
    for idx in top_idx.tolist():
        g = float(golden["final_hidden.bf16"].float().flatten()[idx])
        a = float(actual["final_hidden.bf16"].float().flatten()[idx])
        print(f"  idx={idx:4d} golden={g:10.6f} actual={a:10.6f} delta={a - g:10.6f}")

    print("\naudio-head dtype attribution:")
    golden_hidden = golden["final_hidden.bf16"].to(torch.bfloat16)
    actual_hidden = actual["final_hidden.bf16"].to(torch.bfloat16)
    golden_logits = golden["audio_logits.f32"].reshape(1, CODEBOOKS, VOCAB)

    cpu_f32_from_golden = (golden_hidden.float() @ audio_head.float().T).reshape(
        1, CODEBOOKS, VOCAB
    )
    cpu_f32_from_actual = (actual_hidden.float() @ audio_head.float().T).reshape(
        1, CODEBOOKS, VOCAB
    )
    print_stats(
        "cpu_f32_from_golden_hidden_vs_golden_logits",
        golden_logits,
        cpu_f32_from_golden,
    )
    print_stats(
        "cpu_f32_from_actual_hidden_vs_golden_logits",
        golden_logits,
        cpu_f32_from_actual,
    )

    if args.device.startswith("cuda") and torch.cuda.is_available():
        weight_cuda = audio_head.to(args.device, dtype=torch.bfloat16)
        cuda_bf16_from_golden = torch.nn.functional.linear(
            golden_hidden.to(args.device), weight_cuda
        ).reshape(1, CODEBOOKS, VOCAB).cpu().float()
        cuda_bf16_from_actual = torch.nn.functional.linear(
            actual_hidden.to(args.device), weight_cuda
        ).reshape(1, CODEBOOKS, VOCAB).cpu().float()
        print_stats(
            "cuda_bf16_from_golden_hidden_vs_golden_logits",
            golden_logits,
            cuda_bf16_from_golden,
        )
        print_stats(
            "cuda_bf16_from_actual_hidden_vs_golden_logits",
            golden_logits,
            cuda_bf16_from_actual,
        )
        print_stats(
            "actual_hidden_effect_cuda_bf16",
            cuda_bf16_from_golden,
            cuda_bf16_from_actual,
        )
    else:
        print("cuda_bf16 attribution skipped: CUDA is unavailable")

    print("\ntop-k structure:")
    overlaps = topk_overlap(golden["audio_top64.ids"], actual["audio_top64.ids"])
    print(f"  top64_overlap_by_codebook={overlaps}")
    print(f"  top64_overlap_min={min(overlaps)} mean={sum(overlaps) / len(overlaps):.2f}")
    for cb in range(CODEBOOKS):
        golden_row = golden["audio_logits.f32"][0, cb]
        actual_row = actual["audio_logits.f32"][0, cb]
        top2 = golden_row.topk(2).values
        golden_argmax = int(golden_row.argmax())
        actual_argmax = int(actual_row.argmax())
        print(
            f"  cb={cb} argmax={golden_argmax}/{actual_argmax} "
            f"gold_gap={float(top2[0] - top2[1]):.6f} "
            f"delta_at_gold_top={float(actual_row[golden_argmax] - golden_row[golden_argmax]):.6f}"
        )


if __name__ == "__main__":
    main()
