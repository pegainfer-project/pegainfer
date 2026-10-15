#!/usr/bin/env python3
"""Compare fitted candidates and report held-out residual diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

from fit import (
    FitError,
    _error_metrics,
    evaluate_additive,
    evaluate_cached_surface,
    evaluate_piecewise,
    evaluate_surface,
)


SCHEMA_VERSION = 1
SPLITS = ("train", "threshold_validation", "held_out_mixed", "sealed_test")


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _percentiles(values: list[float]) -> dict[str, float | None]:
    return {
        f"p{int(quantile * 100)}": _percentile(values, quantile)
        for quantile in (0.50, 0.95, 0.99)
    }


def _cell_summary(
    actual: list[float],
    predicted: list[float],
    *,
    ood: list[dict[str, Any]],
    uncertain: list[dict[str, Any]],
    cell: dict[str, Any],
) -> dict[str, Any]:
    if actual:
        status = "measured"
    elif ood:
        status = "out_of_domain"
    elif uncertain:
        status = "uncertain"
    else:
        status = "no_observations"
    return {
        "status": status,
        "cell_id": cell.get("cell_id"),
        "name": cell.get("name"),
        "split": cell.get("split"),
        "kind": cell.get("kind"),
        "sample_count": len(actual),
        "actual": _percentiles(actual),
        "predicted": _percentiles(predicted),
        "errors": _error_metrics(actual, predicted),
        "ood": ood,
        "uncertain": uncertain,
        "source": cell.get("raw") or cell.get("metadata") or {},
    }


def _candidate(fit_document: dict[str, Any], name: str) -> dict[str, Any] | None:
    for candidate in fit_document.get("candidates", []):
        if isinstance(candidate, dict) and candidate.get("name") == name:
            return candidate
    return None


def _observations(document: dict[str, Any], *, kinds: set[str], splits: set[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for cell in document.get("cells", []):
        if not isinstance(cell, dict) or cell.get("status") != "measured":
            continue
        if cell.get("kind") not in kinds or cell.get("split") not in splits:
            continue
        metadata = cell.get("metadata") if isinstance(cell.get("metadata"), dict) else {}
        for observation in cell.get("observations", []):
            if isinstance(observation, dict):
                result.append({"cell": cell, "observation": observation, "metadata": metadata})
    return result


def _direct_prediction(candidate: dict[str, Any], observation: dict[str, Any]) -> float:
    if candidate.get("status") != "fitted":
        raise FitError(f"candidate is {candidate.get('status', 'missing')}")
    if candidate["name"] == "cold_prefill_piecewise_linear":
        return evaluate_piecewise(candidate, observation.get("prefill_input_tokens", observation["input_tokens"]))
    if candidate["name"] == "decode_additive":
        requests = observation["concurrency"]
        kv_tokens = observation.get("sum_decode_ctx_tokens")
        if kv_tokens is None:
            kv_tokens = observation["input_tokens"] * requests
        return evaluate_additive(candidate, requests, kv_tokens)
    if candidate["name"] == "decode_2d_surface":
        requests = observation["concurrency"]
        kv_tokens = observation.get("sum_decode_ctx_tokens")
        if kv_tokens is None:
            kv_tokens = observation["input_tokens"] * requests
        return evaluate_surface(candidate, requests, kv_tokens)
    if candidate["name"] == "cached_prefill_surface":
        prefix_tokens = observation.get("prefix_tokens", observation.get("cached_tokens"))
        if prefix_tokens is None:
            raise FitError("cached-prefill observation has no prefix_tokens")
        new_tokens = observation["input_tokens"] - prefix_tokens
        if new_tokens <= 0:
            raise FitError("cached-prefill observation has no positive new_tokens")
        return evaluate_cached_surface(candidate, prefix_tokens, new_tokens)
    raise FitError(f"no direct evaluator for {candidate['name']}")


def _comparison(
    name: str,
    candidates: dict[str, dict[str, Any] | None],
    entries: list[dict[str, Any]],
    *,
    actual_metric: str,
    prediction: Callable[[dict[str, Any]], float],
    candidate_names: list[str],
) -> dict[str, Any]:
    unavailable = [
        name
        for name in candidate_names
        if candidates.get(name) is None or candidates[name].get("status") != "fitted"
    ]
    if unavailable:
        reason = "; ".join(
            f"{name}: {(candidates.get(name) or {}).get('reason', 'candidate is unavailable')}"
            for name in unavailable
        )
        by_split: dict[str, dict[str, Any]] = {}
        source_cells = sorted(
            {
                entry["observation"].get("cell_id")
                for entry in entries
                if entry["observation"].get("cell_id")
            }
        )
        for split in SPLITS:
            split_entries = [entry for entry in entries if entry["cell"].get("split") == split]
            attribution: dict[str, int] = {}
            by_cell: dict[str, dict[str, Any]] = {}
            for entry in split_entries:
                observation = entry["observation"]
                label = observation.get("attribution", {}).get(actual_metric, "missing")
                attribution[label] = attribution.get(label, 0) + 1
                cell_id = observation.get("cell_id") or entry["cell"].get("name") or "<unknown>"
                record = by_cell.setdefault(
                    cell_id,
                    _cell_summary([], [], ood=[], uncertain=[], cell=entry["cell"]),
                )
                record["status"] = "candidate_unsupported"
                record["reason"] = reason
                record["uncertain"].append({"cell_id": cell_id, "reason": reason})
            by_split[split] = {
                "status": "candidate_unsupported",
                "reason": reason,
                "errors": _error_metrics([], []),
                "sample_count": 0,
                "ood": [],
                "uncertain": [],
                "attribution": attribution,
                "by_cell": by_cell,
            }
        return {
            "name": name,
            "status": "unsupported",
            "reason": reason,
            "candidate_names": candidate_names,
            "fit_policy": "candidates trained only on train; validation and held_out splits are report-only",
            "coverage": {
                candidate_name: candidates[candidate_name].get("coverage")
                if candidates.get(candidate_name)
                else None
                for candidate_name in candidate_names
            },
            "dispersion": {
                candidate_name: candidates[candidate_name].get("source", {}).get("dispersion")
                if candidates.get(candidate_name)
                else None
                for candidate_name in candidate_names
            },
            "candidate_provenance": {
                candidate_name: candidates[candidate_name].get("source")
                if candidates.get(candidate_name)
                else None
                for candidate_name in candidate_names
            },
            "source_cells": source_cells,
            "errors": _error_metrics([], []),
            "by_split": by_split,
            "ood_count": 0,
            "uncertain_count": sum(
                1
                for entry in entries
                if entry["observation"].get("attribution", {}).get(actual_metric) != "black_box_effective"
            ),
            "attribution_status": "paired_request_delta",
        }
    by_split: dict[str, dict[str, Any]] = {}
    source_cells: set[str] = set()
    all_actual: list[float] = []
    all_predicted: list[float] = []
    total_ood = 0
    total_uncertain = 0
    for split in SPLITS:
        actual: list[float] = []
        predicted: list[float] = []
        ood: list[dict[str, Any]] = []
        uncertain: list[dict[str, Any]] = []
        attribution: dict[str, int] = {}
        per_cell: dict[str, dict[str, Any]] = {}
        for entry in entries:
            cell = entry["cell"]
            observation = entry["observation"]
            if cell.get("split") != split:
                continue
            cell_id = observation.get("cell_id")
            if cell_id:
                source_cells.add(cell_id)
            label = observation.get("attribution", {}).get(actual_metric, "missing")
            attribution[label] = attribution.get(label, 0) + 1
            cell_key = cell_id or cell.get("name") or "<unknown>"
            cell_actual = per_cell.setdefault(cell_key, {"actual": [], "predicted": [], "ood": [], "uncertain": [], "cell": cell})
            value = observation.get("effective", {}).get(actual_metric)
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            if (
                entry["cell"].get("kind") == "mixed"
                and metadata.get("requires_overlap") is True
                and metadata.get("overlap_status") != "validated"
            ):
                detail = {"cell_id": cell_id, "reason": "mixed overlap evidence is not validated"}
                uncertain.append(detail)
                cell_actual["uncertain"].append(detail)
                continue
            if value is None or label != "black_box_effective":
                detail = {"cell_id": cell_id, "reason": "missing or uncertain attribution"}
                uncertain.append(detail)
                cell_actual["uncertain"].append(detail)
                continue
            try:
                estimate = prediction(observation)
            except FitError as exc:
                detail = {"cell_id": cell_id, "reason": str(exc)}
                ood.append(detail)
                cell_actual["ood"].append(detail)
                continue
            actual.append(float(value))
            predicted.append(float(estimate))
            cell_actual["actual"].append(float(value))
            cell_actual["predicted"].append(float(estimate))
        if actual:
            all_actual.extend(actual)
            all_predicted.extend(predicted)
        total_ood += len(ood)
        total_uncertain += len(uncertain)
        by_split[split] = {
            "status": "measured" if actual else "no_in_domain_observations",
            "errors": _error_metrics(actual, predicted),
            "sample_count": len(actual),
            "ood": ood,
            "uncertain": uncertain,
            "attribution": attribution,
            "by_cell": {
                cell_id: _cell_summary(
                    values["actual"],
                    values["predicted"],
                    ood=values["ood"],
                    uncertain=values["uncertain"],
                    cell=values["cell"],
                )
                for cell_id, values in sorted(per_cell.items())
            },
        }
    return {
        "name": name,
        "status": "fitted" if all_actual else "unsupported",
        "candidate_names": candidate_names,
        "fit_policy": "candidates trained only on train; validation and held_out splits are report-only",
        "coverage": {
            name: candidates[name].get("coverage") if candidates.get(name) else None for name in candidate_names
        },
        "dispersion": {
            name: candidates[name].get("source", {}).get("dispersion") if candidates.get(name) else None
            for name in candidate_names
        },
        "candidate_provenance": {
            name: candidates[name].get("source") if candidates.get(name) else None for name in candidate_names
        },
        "source_cells": sorted(source_cells),
        "errors": _error_metrics(all_actual, all_predicted),
        "by_split": by_split,
        "ood_count": total_ood,
        "uncertain_count": total_uncertain,
        "attribution_status": "paired_request_delta",
    }


def _composition_prediction(
    prefill: dict[str, Any] | None,
    decode: dict[str, Any] | None,
    *,
    marginal: bool,
) -> Callable[[dict[str, Any]], float]:
    def predict(observation: dict[str, Any]) -> float:
        if prefill is None or decode is None:
            raise FitError("prefill or decode candidate is missing")
        prefill_tokens = observation.get("prefill_input_tokens", observation["input_tokens"])
        prefill_ms = evaluate_piecewise(prefill, prefill_tokens)
        if marginal:
            decode_ms = _direct_prediction(decode, observation)
        else:
            kv_tokens = observation.get("sum_decode_ctx_tokens")
            if kv_tokens is None:
                if observation.get("kind") == "mixed":
                    raise FitError("mixed observation has no role-specific sum_decode_ctx_tokens")
                kv_tokens = observation["input_tokens"] * observation["concurrency"]
            decode_ms = evaluate_additive(
                decode,
                observation["concurrency"],
                kv_tokens,
            )
        return prefill_ms + decode_ms * float(observation["output_tokens"])

    return predict


def _residual_candidate(
    composition: dict[str, Any],
    entries: list[dict[str, Any]],
    prediction: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    excluded: list[dict[str, Any]] = []
    for entry in entries:
        cell = entry["cell"]
        if cell.get("split") != "threshold_validation" or cell.get("kind") != "mixed":
            continue
        metadata = entry["metadata"]
        if metadata.get("scheduler_mismatch") or metadata.get("frontend_mismatch"):
            excluded.append({"cell_id": entry["observation"].get("cell_id"), "reason": "scheduler/frontend mismatch"})
            continue
        observation = entry["observation"]
        if observation.get("attribution", {}).get("e2e_ms") != "black_box_effective":
            continue
        try:
            residual = float(observation["effective"]["e2e_ms"]) - prediction(observation)
        except (FitError, TypeError, ValueError):
            continue
        key = (observation.get("input_tokens"), observation.get("output_tokens"), observation.get("concurrency"))
        groups.setdefault(key, []).append(residual)
    stable: dict[str, float] = {}
    for key, residuals in groups.items():
        if len(residuals) < 2:
            continue
        if all(value > 0.0 for value in residuals) or all(value < 0.0 for value in residuals):
            stable[json.dumps(key, separators=(",", ":"))] = statistics.median(residuals)
    if not stable:
        return {
            "name": "sparse_residual_correction",
            "status": "unsupported",
            "reason": "no repeated same-direction validation residual after mismatch exclusion",
            "excluded": excluded,
            "source_cells": composition.get("source_cells", []),
        }
    return {
        "name": "sparse_residual_correction",
        "status": "candidate",
        "fit_split": "threshold_validation",
        "residuals_ms": stable,
        "excluded": excluded,
        "source_cells": composition.get("source_cells", []),
        "provenance": "derived only from repeated validation residuals; held_out_mixed excluded",
    }


def build_report(
    normalized: dict[str, Any],
    fitted: dict[str, Any],
    *,
    normalized_digest: str | None = None,
    fit_digest: str | None = None,
) -> dict[str, Any]:
    if fitted.get("schema_version") != 1:
        raise ValueError("fit artifact schema_version must be 1")
    entries = _observations(
        normalized,
        kinds={"cold_prefill", "steady_decode", "cached_prefill", "mixed"},
        splits=set(SPLITS),
    )
    candidates = {name: _candidate(fitted, name) for name in (
        "cold_prefill_piecewise_linear", "decode_additive", "decode_2d_surface", "cached_prefill_surface"
    )}
    comparisons: list[dict[str, Any]] = []
    cold_entries = [entry for entry in entries if entry["cell"].get("kind") == "cold_prefill"]
    decode_entries = [entry for entry in entries if entry["cell"].get("kind") == "steady_decode"]
    cached_entries = [entry for entry in entries if entry["cell"].get("kind") == "cached_prefill"]
    mixed_entries = [entry for entry in entries if entry["cell"].get("kind") == "mixed"]
    if candidates["cold_prefill_piecewise_linear"]:
        comparisons.append(
            _comparison(
                "cold_prefill_piecewise_linear",
                candidates,
                cold_entries,
                actual_metric="ttft_ms",
                prediction=lambda observation: _direct_prediction(candidates["cold_prefill_piecewise_linear"], observation),
                candidate_names=["cold_prefill_piecewise_linear"],
            )
        )
    for name in ("decode_additive", "decode_2d_surface"):
        if candidates[name]:
            comparisons.append(
                _comparison(
                    name,
                    candidates,
                    decode_entries,
                    actual_metric=candidates[name].get("metric", "tpot_ms"),
                    prediction=lambda observation, selected=name: _direct_prediction(candidates[selected], observation),
                    candidate_names=[name],
            )
        )
    if candidates["cached_prefill_surface"]:
        comparisons.append(
            _comparison(
                "cached_prefill_surface",
                candidates,
                cached_entries,
                actual_metric="ttft_ms",
                prediction=lambda observation: _direct_prediction(candidates["cached_prefill_surface"], observation),
                candidate_names=["cached_prefill_surface"],
            )
        )
    additive_prediction = _composition_prediction(
        candidates["cold_prefill_piecewise_linear"], candidates["decode_additive"], marginal=False
    )
    marginal_prediction = _composition_prediction(
        candidates["cold_prefill_piecewise_linear"], candidates["decode_2d_surface"], marginal=True
    )
    additive_report = _comparison(
        "composition_additive",
        candidates,
        mixed_entries,
        actual_metric="e2e_ms",
        prediction=additive_prediction,
        candidate_names=["cold_prefill_piecewise_linear", "decode_additive"],
    )
    marginal_report = _comparison(
        "composition_marginal_decode",
        candidates,
        mixed_entries,
        actual_metric="e2e_ms",
        prediction=marginal_prediction,
        candidate_names=["cold_prefill_piecewise_linear", "decode_2d_surface"],
    )
    comparisons.extend((additive_report, marginal_report))
    residual = _residual_candidate(additive_report, mixed_entries, additive_prediction)
    source = normalized.get("source", {}) if isinstance(normalized.get("source"), dict) else {}
    failures = normalized.get("failures", []) if isinstance(normalized.get("failures"), list) else []
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": normalized.get("campaign_id"),
        "status": "succeeded",
        "provenance": {
            "normalized_sha256": normalized_digest,
            "fit_sha256": fit_digest,
            "manifest_sha256": source.get("source_digests", {}).get("manifest_sha256"),
            "artifact_index_sha256": source.get("source_digests", {}).get("artifact_index_sha256"),
            "target_identity": source.get("target_identity"),
            "cache_policy": source.get("cache_policy"),
        },
        "splits": {"fit": ["train"], "validation": ["threshold_validation"], "held_out": list(("held_out_mixed", "sealed_test"))},
        "comparisons": comparisons,
        "held_out_ttft": {
            "status": "replay_required",
            "metric": "ttft_ms",
            "fit_excluded": True,
            "basis": "functional_profile_replay",
            "reason": "report.py fit comparisons intentionally remain TPOT; use replay.queueing_ttft for scheduler TTFT errors",
        },
        "residual_candidates": [residual],
        "failures": failures,
        "missing_invalid_ood": {
            "normalized_failures": failures,
            "comparison_ood": {item["name"]: item["ood_count"] for item in comparisons},
            "no_zero_fill": True,
        },
    }


def write_document(path: Path, document: dict[str, Any]) -> None:
    rendered = json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"refusing to overwrite a different report artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        normalized, normalized_digest = _read_json(args.normalized)
        fitted, fit_digest = _read_json(args.fit)
        report = build_report(normalized, fitted, normalized_digest=normalized_digest, fit_digest=fit_digest)
        write_document(args.output, report)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"report error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"campaign_id": report["campaign_id"], "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
