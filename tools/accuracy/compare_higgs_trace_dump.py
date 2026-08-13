#!/usr/bin/env python3
"""Compare Higgs-Audio rich trace safetensors dumps.

The trace golden is intentionally wider than the default one-step fixture. This
tool compares the common tensor surface so it can be used while the PegaInfer
actual trace is still growing stage by stage.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

PROMPT_TENSORS = (
    "prompt.input_ids_padded",
    "prompt.attention_mask",
    "prompt.lengths",
)
NUM_LAYERS = 36
FINAL_LAYER_IDX = NUM_LAYERS - 1
FINAL_LAYER_HIDDEN_TRACE_KEYS = {
    f"layer.{FINAL_LAYER_IDX:02}.last_hidden.bf16",
    f"layer.{FINAL_LAYER_IDX:02}.sequence_hidden.bf16",
}
STAGE_SUFFIX_ALIASES = {
    "input_norm": "input_layernorm.output",
    "q_proj": "self_attn.q_proj.output",
    "k_proj": "self_attn.k_proj.output",
    "v_proj": "self_attn.v_proj.output",
    "q_norm": "self_attn.q_norm.output",
    "k_norm": "self_attn.k_norm.output",
    "o_proj": "self_attn.o_proj.output",
    "post_attn_norm": "post_attention_layernorm.output",
    "gate_proj": "mlp.gate_proj.output",
    "up_proj": "mlp.up_proj.output",
    "down_proj": "mlp.down_proj.output",
}



STAGE_EXECUTION_ORDER = {
    "input_hidden": 0,
    "input_norm": 1,
    "q_proj": 2,
    "k_proj": 3,
    "v_proj": 4,
    "q_norm": 5,
    "k_norm": 6,
    "q_norm_rope": 7,
    "k_norm_rope": 8,
    "attn_output": 9,
    "o_proj": 10,
    "post_attn_norm": 11,
    "gate_proj": 12,
    "up_proj": 13,
    "silu_mul": 14,
    "down_proj": 15,
    "output_hidden": 16,
}


def stage_actual_order(name: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"layer(\d+)\.([A-Za-z0-9_]+)\.bf16", name)
    if match is None:
        return (10_000, 10_000, name)
    return (int(match.group(1)), STAGE_EXECUTION_ORDER.get(match.group(2), 9_999), name)

def stage_actual_to_trace_name(actual_name: str) -> str | None:
    match = re.fullmatch(r"layer(\d+)\.([A-Za-z0-9_]+)\.bf16", actual_name)
    if match is None:
        return None
    layer_idx = int(match.group(1))
    suffix = match.group(2)
    if suffix == "input_hidden":
        if layer_idx == 0:
            return "embedding.sequence_hidden.bf16"
        return f"layer.{layer_idx - 1:02}.sequence_hidden.bf16"
    if suffix == "output_hidden":
        if layer_idx == FINAL_LAYER_IDX:
            return None
        return f"layer.{layer_idx:02}.sequence_hidden.bf16"
    trace_suffix = STAGE_SUFFIX_ALIASES.get(suffix)
    if trace_suffix is None:
        return None
    return f"layer.{layer_idx:02}.{trace_suffix}.bf16"



@dataclass
class TensorStats:
    name: str
    dtype: str
    shape: list[int]
    max_abs: float | None
    mean_abs: float | None
    p99_abs: float | None
    rmse: float | None
    cosine: float | None
    exact: bool
    alert: bool
    reason: str


@dataclass
class CompareItem:
    display_name: str
    golden_name: str
    actual_name: str


def natural_key(name: str) -> tuple:
    parts: list[object] = []
    for piece in re.split(r"(\d+)", name):
        if piece.isdigit():
            parts.append(int(piece))
        else:
            parts.append(piece)
    return tuple(parts)


def trace_order(name: str) -> tuple:
    if name.startswith("prompt."):
        group = 0
    elif name.startswith("embedding."):
        group = 1
    elif name.startswith("layer."):
        group = 2
    elif name.startswith("final_hidden."):
        group = 3
    elif name.startswith("audio_head."):
        group = 4
    elif name.startswith("audio_"):
        group = 5
    else:
        group = 9
    return (group, natural_key(name))


def quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), q))


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.float().flatten()
    right_flat = right.float().flatten()
    if left_flat.numel() == 0:
        return 1.0
    if float(torch.linalg.vector_norm(left_flat)) == 0.0 and float(torch.linalg.vector_norm(right_flat)) == 0.0:
        return 1.0
    return float(torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=0))


def last_token_from_sequence(sequence: torch.Tensor, prompt_lengths: torch.Tensor) -> torch.Tensor:
    if sequence.ndim < 3:
        return sequence
    rows = []
    for batch_idx, prompt_len in enumerate(prompt_lengths.tolist()):
        rows.append(sequence[batch_idx, int(prompt_len) - 1])
    return torch.stack(rows, dim=0)


def align_golden_to_actual(
    golden: torch.Tensor,
    actual: torch.Tensor,
    prompt_lengths: torch.Tensor | None,
) -> torch.Tensor:
    if tuple(golden.shape) == tuple(actual.shape):
        return golden
    if prompt_lengths is not None and golden.ndim >= 3:
        last = last_token_from_sequence(golden, prompt_lengths)
        if tuple(last.shape) == tuple(actual.shape):
            return last
        if actual.ndim == 2 and last.ndim > 2 and last.shape[0] == actual.shape[0]:
            flattened = last.reshape(last.shape[0], -1)
            if tuple(flattened.shape) == tuple(actual.shape):
                return flattened
    return golden


def compare_tensor(
    name: str,
    golden: torch.Tensor,
    actual: torch.Tensor,
    *,
    prompt_lengths: torch.Tensor | None,
    mean_alert: float,
    cosine_alert: float,
    max_alert: float | None,
) -> TensorStats:
    golden = align_golden_to_actual(golden, actual, prompt_lengths)
    if tuple(golden.shape) != tuple(actual.shape):
        return TensorStats(
            name=name,
            dtype=f"{golden.dtype}/{actual.dtype}",
            shape=list(golden.shape),
            max_abs=None,
            mean_abs=None,
            p99_abs=None,
            rmse=None,
            cosine=None,
            exact=False,
            alert=True,
            reason=f"shape_mismatch golden={tuple(golden.shape)} actual={tuple(actual.shape)}",
        )

    exact = bool(torch.equal(golden, actual))
    if not (torch.is_floating_point(golden) or torch.is_floating_point(actual)):
        return TensorStats(
            name=name,
            dtype=str(golden.dtype),
            shape=list(golden.shape),
            max_abs=None,
            mean_abs=None,
            p99_abs=None,
            rmse=None,
            cosine=None,
            exact=exact,
            alert=not exact,
            reason="ok" if exact else "integer_mismatch",
        )

    delta = actual.float().flatten() - golden.float().flatten()
    abs_delta = delta.abs()
    max_abs = float(abs_delta.max()) if abs_delta.numel() else 0.0
    mean_abs = float(abs_delta.mean()) if abs_delta.numel() else 0.0
    p99_abs = quantile(abs_delta, 0.99)
    rmse = float(torch.sqrt(torch.mean(delta * delta))) if delta.numel() else 0.0
    cos = cosine(golden, actual)
    reasons = []
    if mean_abs > mean_alert:
        reasons.append(f"mean_abs>{mean_alert}")
    if cos < cosine_alert:
        reasons.append(f"cosine<{cosine_alert}")
    if max_alert is not None and max_abs > max_alert:
        reasons.append(f"max_abs>{max_alert}")
    return TensorStats(
        name=name,
        dtype=str(golden.dtype),
        shape=list(golden.shape),
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        rmse=rmse,
        cosine=cos,
        exact=exact,
        alert=bool(reasons),
        reason=";".join(reasons) if reasons else "ok",
    )


def prompt_exact(golden: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> bool:
    for name in PROMPT_TENSORS:
        if name not in golden or name not in actual:
            return False
        if not torch.equal(golden[name], actual[name]):
            return False
    return True


def select_names(
    golden: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    include_regex: str,
) -> tuple[list[str], list[str], list[str]]:
    pattern = re.compile(include_regex) if include_regex else None
    golden_names = set(golden) - FINAL_LAYER_HIDDEN_TRACE_KEYS
    actual_names = set(actual) - FINAL_LAYER_HIDDEN_TRACE_KEYS
    if pattern is not None:
        golden_names = {name for name in golden_names if pattern.search(name)}
        actual_names = {name for name in actual_names if pattern.search(name)}
    common = sorted(golden_names & actual_names, key=trace_order)
    missing_from_actual = sorted(golden_names - actual_names, key=trace_order)
    extra_actual = sorted(actual_names - golden_names, key=trace_order)
    return common, missing_from_actual, extra_actual


def select_items(
    golden: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    include_regex: str,
    alias_set: str,
) -> tuple[list[CompareItem], list[str], list[str]]:
    if alias_set == "none":
        common, missing_from_actual, extra_actual = select_names(golden, actual, include_regex)
        return [CompareItem(name, name, name) for name in common], missing_from_actual, extra_actual
    if alias_set not in {"layer0-stage", "layer-stage"}:
        raise ValueError(f"unknown alias set: {alias_set}")

    pattern = re.compile(include_regex) if include_regex else None
    items = []
    missing_from_golden = []
    seen_golden = set()
    actual_names = sorted(actual, key=stage_actual_order)
    for actual_name in actual_names:
        golden_name = stage_actual_to_trace_name(actual_name)
        if golden_name is None:
            continue
        if alias_set == "layer0-stage" and not actual_name.startswith("layer0."):
            continue
        display_name = f"{actual_name} -> {golden_name}"
        if pattern is not None and not (pattern.search(actual_name) or pattern.search(golden_name)):
            continue
        if golden_name not in golden:
            missing_from_golden.append(golden_name)
            continue
        seen_golden.add(golden_name)
        items.append(CompareItem(display_name, golden_name, actual_name))
    missing_from_actual = []
    if alias_set == "layer0-stage":
        for actual_name in (
            "layer0.input_hidden.bf16",
            "layer0.input_norm.bf16",
            "layer0.q_proj.bf16",
            "layer0.k_proj.bf16",
            "layer0.v_proj.bf16",
            "layer0.q_norm.bf16",
            "layer0.k_norm.bf16",
            "layer0.o_proj.bf16",
            "layer0.post_attn_norm.bf16",
            "layer0.gate_proj.bf16",
            "layer0.up_proj.bf16",
            "layer0.down_proj.bf16",
            "layer0.output_hidden.bf16",
        ):
            if actual_name not in actual:
                missing_from_actual.append(actual_name)
    return items, missing_from_actual, sorted(set(missing_from_golden), key=trace_order)


def format_float(value: float | None) -> str:
    if value is None:
        return "       n/a"
    return f"{value:10.6f}"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--mean-alert", type=float, default=0.003)
    parser.add_argument("--cosine-alert", type=float, default=0.9998)
    parser.add_argument("--max-alert", type=float, default=None)
    parser.add_argument("--include-regex", default="")
    parser.add_argument(
        "--alias-set",
        choices=("none", "layer0-stage", "layer-stage"),
        default="none",
        help="Optional built-in mapping from a partial actual dump schema to the trace golden schema.",
    )
    parser.add_argument("--top", type=int, default=120)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--fail-on-alert", action="store_true")
    args = parser.parse_args()

    golden = load_file(str(args.golden), device="cpu")
    actual = load_file(str(args.actual), device="cpu")
    items, missing_from_actual, extra_actual = select_items(
        golden, actual, args.include_regex, args.alias_set
    )
    if not items:
        raise RuntimeError("no comparable tensors to compare")
    prompt_lengths = golden.get("prompt.lengths")

    rows = [
        compare_tensor(
            item.display_name,
            golden[item.golden_name],
            actual[item.actual_name],
            prompt_lengths=prompt_lengths,
            mean_alert=args.mean_alert,
            cosine_alert=args.cosine_alert,
            max_alert=args.max_alert,
        )
        for item in items
    ]
    alerts = [row for row in rows if row.alert]
    first_alert = alerts[0].name if alerts else "none"
    floating_rows = [row for row in rows if row.mean_abs is not None and row.cosine is not None]
    worst_mean = max(floating_rows, key=lambda row: row.mean_abs or 0.0, default=None)
    worst_cosine = min(floating_rows, key=lambda row: row.cosine if row.cosine is not None else 1.0, default=None)

    print(f"prompt_exact={prompt_exact(golden, actual)}")
    print(f"common_tensors={len(items)}")
    print(f"missing_from_actual={len(missing_from_actual)}")
    print(f"extra_actual={len(extra_actual)}")
    print("name                                                       max_abs   mean_abs    p99_abs       rmse      cosine  exact  alert  reason")
    for row in rows[: args.top]:
        print(
            f"{row.name:58} {format_float(row.max_abs)} {format_float(row.mean_abs)} "
            f"{format_float(row.p99_abs)} {format_float(row.rmse)} {format_float(row.cosine)} "
            f"{str(row.exact):>5} {str(row.alert):>6}  {row.reason}"
        )
    if len(rows) > args.top:
        print(f"... truncated {len(rows) - args.top} row(s); use --top to print more")
    print("summary:")
    print(f"  compared={len(rows)}")
    print(f"  alerts={len(alerts)}")
    print(f"  first_alert={first_alert}")
    print(f"  worst_mean_abs={(worst_mean.name + ':' + f'{worst_mean.mean_abs:.6f}') if worst_mean else 'none'}")
    print(f"  worst_cosine={(worst_cosine.name + ':' + f'{worst_cosine.cosine:.9f}') if worst_cosine else 'none'}")
    if missing_from_actual[:10]:
        print(f"  missing_from_actual_first10={missing_from_actual[:10]}")
    if extra_actual[:10]:
        print(f"  extra_actual_first10={extra_actual[:10]}")

    if args.json_out is not None:
        write_json(
            args.json_out,
            {
                "golden": str(args.golden),
                "actual": str(args.actual),
                "prompt_exact": prompt_exact(golden, actual),
                "common_tensors": len(items),
                "alias_set": args.alias_set,
                "items": [asdict(item) for item in items],
                "missing_from_actual": missing_from_actual,
                "extra_actual": extra_actual,
                "alerts": len(alerts),
                "first_alert": first_alert,
                "worst_mean_abs": asdict(worst_mean) if worst_mean else None,
                "worst_cosine": asdict(worst_cosine) if worst_cosine else None,
                "rows": [asdict(row) for row in rows],
            },
        )

    if args.require_all and missing_from_actual:
        raise SystemExit(2)
    if args.fail_on_alert and alerts:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
