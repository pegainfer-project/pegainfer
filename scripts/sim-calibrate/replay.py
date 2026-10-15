#!/usr/bin/env python3
"""Profile validation and replay reporting rules retained until C7 migration.

C2 owns request execution and evidence storage through campaign/plan_driver.
This module performs no HTTP sampling and reads no execution artifact index.
Its report input/metric contracts still await C3-C7 migration; it is not wired
as a consumer of measurement v3 and does not adapt old execution artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from manifest import (
    ManifestError,
    canonical_cache_identity,
    canonical_client_identity,
    canonical_scheduler_identity,
    canonical_workload_identity,
)
from metrics import summarize_measurements
from pairing import (
    canonical_pair_key,
    index_metric_measurements,
    pair_metric_measurements,
    pairing_key_text,
)


SCHEMA_VERSION = 1
METRICS = ("ttft_ms", "stream_chunk_tpot_ms", "e2e_ms")
QUANTILES = (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
PURE_COVERAGE_KINDS = ("cold_prefill", "steady_decode")


class ReplayError(ValueError):
    """Raised when a profile replay cannot preserve campaign lineage."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayError(f"JSON artifact must be an object: {path}")
    return value, _sha256(payload)


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ReplayError(f"{field} must be finite")
    return float(value)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _percentiles(values: list[float]) -> dict[str, float | None]:
    return {label: _percentile(values, quantile) for label, quantile in QUANTILES}


def _error_metrics(observed: list[float], predicted: list[float]) -> dict[str, Any]:
    absolute: list[float] = []
    relative: list[float] = []
    for actual, estimate in zip(observed, predicted):
        actual = _finite(actual, "observed")
        estimate = _finite(estimate, "predicted")
        absolute.append(abs(estimate - actual))
        if actual > 0.0:
            relative.append(abs(estimate - actual) / actual)
    return {
        "count": len(absolute),
        "absolute_ms": {label: _percentile(absolute, quantile) for label, quantile in QUANTILES},
        "relative": {label: _percentile(relative, quantile) for label, quantile in QUANTILES},
    }


def _paired_error_metrics(pairs: list[tuple[Any, float, float]]) -> dict[str, Any]:
    """Calculate request-level errors after an exact identity join."""
    absolute: list[float] = []
    relative: list[float] = []
    for _key, actual, predicted in pairs:
        actual = _finite(actual, "observed")
        predicted = _finite(predicted, "predicted")
        absolute.append(abs(predicted - actual))
        if actual > 0.0:
            relative.append(abs(predicted - actual) / actual)
    return {
        "count": len(pairs),
        "absolute_count": len(absolute),
        "relative_count": len(relative),
        "relative_undefined_count": len(pairs) - len(relative),
        "absolute_ms": _percentiles(absolute),
        "relative": _percentiles(relative),
    }


def _safe_profile_artifact(profile_path: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ReplayError(f"publication artifact path must be relative: {relative!r}")
    root = profile_path.parent.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ReplayError(f"publication artifact path escapes profile bundle: {relative!r}") from exc
    return candidate


def _validate_publication(
    profile_path: Path,
    profile_digest: str,
    publication_path: Path,
    calibration: dict[str, Any],
    selection: dict[str, Any],
) -> str:
    publication, publication_digest = _read_json(publication_path)
    if publication.get("status") != "passed":
        raise ReplayError("published profile status is not passed")
    profile_ref = publication.get("profile")
    if not isinstance(profile_ref, dict) or profile_ref.get("sha256") != profile_digest:
        raise ReplayError("published profile digest does not match profile bytes")
    target = selection.get("target")
    if publication.get("target_identity") != target:
        raise ReplayError("published profile target identity does not match selection")
    for name, expected_status in (("tolerance", "frozen"), ("signoff", "passed")):
        reference = publication.get(name)
        if not isinstance(reference, dict):
            raise ReplayError(f"published profile is missing {name} reference")
        relative = reference.get("path")
        expected_digest = reference.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_digest, str):
            raise ReplayError(f"published profile {name} reference is incomplete")
        artifact_path = _safe_profile_artifact(profile_path, relative)
        artifact, actual_digest = _read_json(artifact_path)
        if actual_digest != expected_digest:
            raise ReplayError(f"published profile {name} digest does not match bytes")
        if artifact.get("status") != expected_status:
            raise ReplayError(f"published profile {name} status is not {expected_status}")
        if artifact.get("target_identity") != target:
            raise ReplayError(f"published profile {name} target identity does not match selection")
        if artifact.get("profile", {}).get("sha256") != profile_digest:
            raise ReplayError(f"published profile {name} does not reference this profile")
    return publication_digest


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = result.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("percentiles_ms"), dict):
        return summary
    measurements = result.get("measurements")
    if not isinstance(measurements, list):
        raise ReplayError("variant result has neither summary nor measurements")
    return summarize_measurements(measurements)


def _summary_value(result: dict[str, Any], metric: str, quantile: str = "p50") -> float | None:
    values = _summary(result).get("percentiles_ms", {}).get(metric)
    if not isinstance(values, dict):
        return None
    value = values.get(quantile)
    return None if value is None else _finite(value, f"summary.{metric}.{quantile}")


def _dispersion(result: dict[str, Any], metric: str) -> dict[str, Any]:
    measurements = result.get("measurements")
    values = []
    if isinstance(measurements, list):
        for item in measurements:
            if not isinstance(item, dict) or item.get("phase") != "measured" or item.get("ok") is not True:
                continue
            value = item.get(metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                values.append(float(value))
    if not values:
        return {"count": 0, "median": None, "mad": None, "min": None, "max": None}
    median = statistics.median(values)
    return {
        "count": len(values),
        "median": median,
        "mad": statistics.median(abs(value - median) for value in values),
        "min": min(values),
        "max": max(values),
    }


def _request_keys(result: dict[str, Any], cell_id: str | None = None) -> list[str]:
    measurements = result.get("measurements")
    if not isinstance(measurements, list):
        return []
    resolved_cell_id = cell_id or result.get("cell_id")
    keys: list[str] = []
    for item in measurements:
        if not isinstance(item, dict):
            continue
        if isinstance(resolved_cell_id, str) and resolved_cell_id:
            try:
                keys.append(pairing_key_text(canonical_pair_key(item, resolved_cell_id)))
                continue
            except ValueError:
                pass
        keys.append(f"{item.get('phase')}:{item.get('repeat')}:{item.get('index')}:{item.get('request_id')}")
    return sorted(keys)


def _bracket(axis: list[int], value: int) -> tuple[int, int, float] | None:
    if not axis or value < axis[0] or value > axis[-1]:
        return None
    for index in range(len(axis) - 1):
        lower, upper = axis[index], axis[index + 1]
        if lower <= value <= upper:
            fraction = 0.0 if upper == lower else (value - lower) / (upper - lower)
            return index, index + 1, fraction
    return len(axis) - 1, len(axis) - 1, 0.0


def _piecewise(model: dict[str, Any], value: int) -> float | None:
    axis = [int(item) for item in model.get("input_tokens", [])]
    durations = [float(item) / 1000.0 for item in model.get("duration_us", [])]
    if len(axis) != len(durations):
        raise ReplayError("prefill profile axis/value lengths do not match")
    bracket = _bracket(axis, value)
    if bracket is None:
        return None
    lower, upper, fraction = bracket
    return durations[lower] + (durations[upper] - durations[lower]) * fraction


def _surface(model: dict[str, Any], requests: int, context: int) -> float | None:
    request_axis = [int(item) for item in model.get("decode_reqs", [])]
    context_axis = [int(item) for item in model.get("sum_decode_ctx_tokens", [])]
    values = [float(item) / 1000.0 for item in model.get("duration_us", [])]
    request_bracket = _bracket(request_axis, requests)
    context_bracket = _bracket(context_axis, context)
    if request_bracket is None or context_bracket is None:
        return None
    r0, r1, rf = request_bracket
    c0, c1, cf = context_bracket
    width = len(context_axis)

    def at(request_index: int, context_index: int) -> float:
        return values[request_index * width + context_index]

    low = at(r0, c0) + (at(r0, c1) - at(r0, c0)) * cf
    high = at(r1, c0) + (at(r1, c1) - at(r1, c0)) * cf
    return low + (high - low) * rf


def _intercept_ms(mixed: dict[str, Any]) -> float:
    value = mixed.get("intercept_us")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ReplayError("mixed.intercept_us must be finite")
    if float(value) < 0.0:
        raise ReplayError("mixed.intercept_us must be non-negative")
    return float(value) / 1000.0


def _cell_shape(cell: dict[str, Any]) -> dict[str, Any]:
    kind = cell.get("kind")
    concurrency = int(cell.get("concurrency", 1))
    input_tokens = int(cell["input_tokens"])
    output_tokens = int(cell["output_tokens"])
    max_decode_context = (input_tokens + max(output_tokens - 2, 0)) * concurrency
    if kind == "cold_prefill":
        return {
            "decode_reqs": 1,
            "sum_decode_ctx_tokens": input_tokens,
            "max_sum_decode_ctx_tokens": max_decode_context,
            "prefill_tokens_in_step": input_tokens,
        }
    if kind == "steady_decode":
        return {
            "decode_reqs": concurrency,
            "sum_decode_ctx_tokens": input_tokens * concurrency,
            "max_sum_decode_ctx_tokens": max_decode_context,
            "prefill_tokens_in_step": input_tokens,
        }
    if kind == "mixed":
        mixed_shape = cell.get("mixed_shape")
        foreground = mixed_shape.get("foreground") if isinstance(mixed_shape, dict) else None
        background = mixed_shape.get("background") if isinstance(mixed_shape, dict) else None
        if not isinstance(foreground, dict) or not isinstance(background, dict):
            raise ReplayError("mixed cell requires role-specific mixed_shape")
        background_requests = int(background.get("requests", 0))
        background_context = int(background.get("context_tokens", 0))
        background_output = int(background.get("output_tokens", 0))
        if background_requests <= 0 or background_context <= 0 or background_output <= 1:
            raise ReplayError("mixed background shape is invalid")
        foreground_input = int(foreground.get("input_tokens", 0))
        if foreground_input <= 0:
            raise ReplayError("mixed foreground input_tokens must be positive")
        return {
            "decode_reqs": background_requests,
            "sum_decode_ctx_tokens": int(cell.get("sum_decode_ctx_tokens", background_requests * background_context)),
            "max_sum_decode_ctx_tokens": background_requests * (background_context + max(background_output - 2, 0)),
            "prefill_tokens_in_step": int(cell.get("prefill_input_tokens", foreground_input)),
            "decode_output_tokens": background_output,
        }
    raise ReplayError(f"unsupported cell kind {kind!r}")


def _predict_cell(profile: dict[str, Any], cell: dict[str, Any]) -> tuple[str, str | None, dict[str, float]]:
    model = profile.get("predictor", {}).get("model", {})
    shape = _cell_shape(cell)
    prefill = model.get("prefill")
    decode = model.get("decode")
    if not isinstance(prefill, dict) or not isinstance(decode, dict):
        return "unsupported", "functional profile is missing prefill or decode", {}
    prefill_ms = _piecewise(prefill, shape["prefill_tokens_in_step"])
    decode_steps = max(int(shape.get("decode_output_tokens", cell["output_tokens"])) - 1, 0)
    decode_durations = [
        _surface(
            decode,
            shape["decode_reqs"],
            shape["sum_decode_ctx_tokens"] + step * shape["decode_reqs"],
        )
        for step in range(decode_steps)
    ]
    if prefill_ms is None or any(value is None for value in decode_durations):
        return "out_of_domain", "step shape is outside the fitted model axes", {}
    decode_values = [float(value) for value in decode_durations]
    if cell.get("kind") == "mixed":
        mixed = model.get("mixed")
        if not isinstance(mixed, dict):
            return "unsupported", "mixed predictor is not present", {}
        mixed_decode = _surface(
            decode,
            shape["decode_reqs"],
            shape["sum_decode_ctx_tokens"],
        )
        if mixed_decode is None:
            return "out_of_domain", "mixed step shape is outside the fitted model axes", {}
        mixed_step_ms = prefill_ms + float(mixed_decode) + _intercept_ms(mixed)
        decode_ms = sum(decode_values) / len(decode_values) if decode_values else 0.0
        e2e_ms = mixed_step_ms + sum(decode_values)
        return "in_domain", None, {
            "ttft_ms": mixed_step_ms,
            "stream_chunk_tpot_ms": decode_ms,
            "e2e_ms": e2e_ms,
            "throughput_tokens_per_s": (int(cell["output_tokens"]) / e2e_ms) * 1000.0,
        }
    decode_ms = sum(decode_values) / len(decode_values) if decode_values else 0.0
    return "in_domain", None, {
        "ttft_ms": prefill_ms,
        "stream_chunk_tpot_ms": decode_ms,
        "e2e_ms": prefill_ms + sum(decode_values),
        "throughput_tokens_per_s": (int(cell["output_tokens"]) / (prefill_ms + sum(decode_values))) * 1000.0,
    }


def _identity_matches(
    campaign: dict[str, Any],
    calibration: dict[str, Any],
    selection: dict[str, Any],
    *,
    allow_independent_campaign: bool = False,
) -> None:
    target = campaign.get("target")
    if not isinstance(target, dict):
        raise ReplayError("campaign target identity is missing")
    expected = {
        "target_engine": target.get("engine"),
        "engine_version": target.get("engine_version"),
        "model_id": target.get("model"),
        "model_revision": target.get("model_revision"),
        "model_config_sha256": target.get("model_config_sha256"),
    }
    for field, value in expected.items():
        if calibration.get(field) != value:
            raise ReplayError(f"profile calibration identity mismatch for {field}")
    runtime = campaign.get("runtime")
    if not isinstance(runtime, dict):
        raise ReplayError("campaign runtime identity is missing")
    environment = campaign.get("environment")
    if not isinstance(environment, dict):
        raise ReplayError("campaign environment identity is missing")
    try:
        expected_scheduler = canonical_scheduler_identity(runtime.get("scheduler"))
        expected_cache = canonical_cache_identity(runtime.get("cache"))
        expected_client = canonical_client_identity(campaign.get("client"))
        expected_workload, expected_split = canonical_workload_identity(campaign)
    except ManifestError as exc:
        raise ReplayError(f"campaign identity is invalid: {exc}") from exc
    for field, expected_value in (
        ("backend", runtime.get("backend") or "unknown"),
        ("driver", environment.get("driver") or "unknown"),
        ("cuda", environment.get("cuda") or "unknown"),
        ("parallelism", runtime.get("parallelism") or "unknown"),
        ("quantization", runtime.get("quantization") or "none"),
        ("gpu", environment.get("cuda_visible_devices") or "unbound"),
    ):
        if calibration.get(field) != expected_value:
            raise ReplayError(f"profile calibration identity mismatch for {field}")
    if calibration.get("scheduler") != expected_scheduler:
        raise ReplayError("profile calibration identity mismatch for scheduler")
    if calibration.get("cache") != expected_cache:
        raise ReplayError("profile calibration identity mismatch for cache")
    actual_client = calibration.get("client")
    if not isinstance(actual_client, dict):
        raise ReplayError("profile calibration client identity is missing")
    for field, expected_value in expected_client.items():
        if field == "tokenizer_config_sha256" and "tokenizer_config_sha256" not in campaign.get("client", {}):
            continue
        if actual_client.get(field) != expected_value:
            raise ReplayError(f"profile calibration identity mismatch for client.{field}")
    if calibration.get("server_flags") != campaign.get("client", {}).get("server_flags"):
        raise ReplayError("profile calibration identity mismatch for server_flags")
    if calibration.get("workload") != expected_workload:
        raise ReplayError("profile calibration identity mismatch for workload")
    if not allow_independent_campaign and calibration.get("split") != expected_split:
        raise ReplayError("profile calibration identity mismatch for split")
    selected_target = selection.get("target")
    if selected_target != target:
        raise ReplayError("selection target identity does not match campaign target")


def validate_profile(
    profile_path: Path,
    campaign: dict[str, Any],
    *,
    selection_path: Path | None = None,
    allow_independent_campaign: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    profile, profile_digest = _read_json(profile_path)
    if profile.get("schema_version") != 3:
        raise ReplayError("profile schema_version must be 3")
    calibration_ref = profile.get("calibration")
    if not isinstance(calibration_ref, dict) or calibration_ref.get("manifest") != "calibration.manifest.json":
        raise ReplayError("profile calibration manifest reference is invalid")
    manifest_path = profile_path.parent / calibration_ref["manifest"]
    calibration, manifest_digest = _read_json(manifest_path)
    if calibration_ref.get("sha256") != manifest_digest:
        raise ReplayError("profile calibration manifest digest does not match bytes")
    if profile.get("predictor", {}).get("model", {}).get("kind") != "functional":
        raise ReplayError("selected replay requires a functional profile")
    selection_path = selection_path or profile_path.parent / "selection.json"
    selection, selection_digest = _read_json(selection_path)
    if not allow_independent_campaign and selection.get("campaign_id") != campaign.get("campaign_id"):
        raise ReplayError("profile selection campaign_id does not match campaign")
    _identity_matches(
        campaign,
        calibration,
        selection,
        allow_independent_campaign=allow_independent_campaign,
    )
    publication_path = profile_path.parent / "publication.json"
    publication_digest = None
    if publication_path.exists():
        publication_digest = _validate_publication(
            profile_path,
            profile_digest,
            publication_path,
            calibration,
            selection,
        )
    return profile, calibration, selection, {
        "profile_sha256": profile_digest,
        "calibration_manifest_sha256": manifest_digest,
        "selection_sha256": selection_digest,
        **({"publication_sha256": publication_digest} if publication_digest else {}),
    }


def _effective(target: dict[str, Any], zero_cost: dict[str, Any], metric: str) -> float | None:
    target_value = _summary_value(target, metric)
    zero_value = _summary_value(zero_cost, metric)
    if target_value is None or zero_value is None:
        return None
    return target_value - zero_value


def _add_error(
    bucket: dict[str, dict[str, list[float]]],
    family: str,
    metric: str,
    observed: float | None,
    predicted: float | None,
) -> None:
    if observed is None or predicted is None:
        return
    bucket.setdefault(family, {}).setdefault(metric, [[], []])
    bucket[family][metric][0].append(observed)
    bucket[family][metric][1].append(predicted)


def _add_error_samples(
    bucket: dict[str, dict[str, list[float]]],
    family: str,
    metric: str,
    observed: list[float],
    predicted: list[float],
) -> None:
    """Add all paired request samples so aggregate percentiles stay request-level."""
    if not observed or not predicted:
        return
    bucket.setdefault(family, {}).setdefault(metric, [[], []])
    bucket[family][metric][0].extend(observed)
    bucket[family][metric][1].extend(predicted)


def _request_metric_values(result: dict[str, Any] | None, metric: str) -> dict[str, float]:
    """Return measured request values keyed by the stable pairing identity."""
    if not isinstance(result, dict) or not isinstance(result.get("measurements"), list):
        return {}
    cell_id = result.get("cell_id")
    if isinstance(cell_id, str) and cell_id:
        indexed = index_metric_measurements(result, cell_id=cell_id, metric=metric)
        return {pairing_key_text(key): value for key, value in indexed["values"].items()}
    values: dict[str, float] = {}
    for item in result["measurements"]:
        if not isinstance(item, dict) or item.get("phase") != "measured" or item.get("ok") is not True:
            continue
        value = item.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        key = f"{item.get('phase')}:{item.get('repeat')}:{item.get('index')}:{item.get('request_id')}"
        values[key] = float(value)
    return values


def _request_error_samples(
    target: dict[str, Any],
    zero_cost: dict[str, Any],
    sim: dict[str, Any] | None,
    metric: str,
    fit_value: float | None,
) -> dict[str, tuple[list[float], list[float]]]:
    """Build paired request samples, leaving summary fallback to the caller."""
    target_values = _request_metric_values(target, metric)
    zero_values = _request_metric_values(zero_cost, metric)
    sim_values = _request_metric_values(sim, metric)
    effective_keys = sorted(set(target_values) & set(zero_values))
    result: dict[str, tuple[list[float], list[float]]] = {}
    if fit_value is not None and effective_keys:
        effective = [target_values[key] - zero_values[key] for key in effective_keys]
        result["effective_vs_fitted"] = (effective, [fit_value] * len(effective))
        result["target_vs_baseline_plus_fitted"] = (
            [target_values[key] for key in effective_keys],
            [zero_values[key] + fit_value for key in effective_keys],
        )
    sim_keys = sorted(set(target_values) & set(sim_values))
    if sim_keys:
        result["target_vs_sim"] = (
            [target_values[key] for key in sim_keys],
            [sim_values[key] for key in sim_keys],
        )
    return result


def _render_errors(bucket: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    return {
        family: {metric: _error_metrics(values[0], values[1]) for metric, values in metrics.items()}
        for family, metrics in bucket.items()
    }


def audit_publication_coverage(
    cells: list[dict[str, Any]],
    *,
    required_splits: tuple[str, ...] = ("sealed_test",),
    required_kinds: tuple[str, ...] = PURE_COVERAGE_KINDS,
) -> dict[str, Any]:
    """Audit the in-domain replay scope that may support a publication/signoff."""
    scoped = [
        cell
        for cell in cells
        if cell.get("split") in required_splits and cell.get("kind") in required_kinds
    ]
    failures: list[dict[str, Any]] = []
    counts = {
        "cells": len(scoped),
        "in_domain": 0,
        "out_of_domain": 0,
        "unsupported": 0,
        "sim_failed": 0,
        "pairing_failed": 0,
    }
    for cell in scoped:
        cell_id = cell.get("cell_id")
        coverage = cell.get("coverage") or {}
        coverage_status = coverage.get("status")
        if coverage_status in ("in_domain", "out_of_domain", "unsupported"):
            counts[coverage_status] += 1
        else:
            failures.append({"cell_id": cell_id, "reason": "missing coverage status"})
        sim = cell.get("sim") or {}
        if sim.get("status") != "succeeded":
            counts["sim_failed"] += 1
        pairing = cell.get("pairing") or {}
        if pairing.get("target_zero_cost") is not True or pairing.get("target_sim") is not True:
            counts["pairing_failed"] += 1
        if coverage_status != "in_domain":
            failures.append({"cell_id": cell_id, "reason": coverage.get("reason") or coverage_status})
        if sim.get("status") != "succeeded":
            failures.append({"cell_id": cell_id, "reason": "sim replay did not succeed"})
        if pairing.get("target_zero_cost") is not True or pairing.get("target_sim") is not True:
            failures.append({"cell_id": cell_id, "reason": "request attribution pairing failed"})
    missing_splits = [
        split
        for split in required_splits
        if not any(cell.get("split") == split and cell.get("kind") in required_kinds for cell in cells)
    ]
    for split in missing_splits:
        failures.append({"split": split, "reason": "no required pure prefill/decode cells"})
    return {
        "status": "passed" if not failures else "failed",
        "required_splits": list(required_splits),
        "required_kinds": list(required_kinds),
        "counts": counts,
        "missing_splits": missing_splits,
        "failures": failures,
    }


def require_publication_coverage(
    report: dict[str, Any],
    *,
    required_splits: tuple[str, ...] = ("sealed_test",),
    required_kinds: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if required_kinds is None:
        configured = report.get("coverage_required_kinds")
        required_kinds = (
            tuple(item for item in configured if isinstance(item, str))
            if isinstance(configured, list) and configured
            else PURE_COVERAGE_KINDS
        )
    audit = audit_publication_coverage(
        report.get("cells", []),
        required_splits=required_splits,
        required_kinds=required_kinds,
    )
    if audit["status"] != "passed":
        raise ReplayError(
            "publication coverage gate failed: "
            + json.dumps(audit, ensure_ascii=True, sort_keys=True)
        )
    return audit


def _empty_queueing_percentiles() -> dict[str, None]:
    return {label: None for label, _quantile in QUANTILES}


def _queueing_ttft_record(
    cell: dict[str, Any],
    cell_id: str,
    target: dict[str, Any],
    sim: dict[str, Any] | None,
    coverage_status: str,
    coverage_reason: str | None,
) -> dict[str, Any]:
    """Build an independent TTFT record with fail-closed request pairing."""
    concurrency = cell.get("concurrency")
    base = {
        "cell_id": cell_id,
        "name": cell.get("name"),
        "kind": cell.get("kind"),
        "split": cell.get("split"),
        "concurrency": concurrency,
        "actual": _empty_queueing_percentiles(),
        "predicted": _empty_queueing_percentiles(),
        "errors": None,
    }
    if isinstance(concurrency, int) and not isinstance(concurrency, bool) and concurrency <= 1:
        return {
            **base,
            "status": "not_applicable",
            "reason": "queueing TTFT requires concurrency > 1",
            "pairing": {"status": "not_applicable", "complete": False, "key_schema": ["cell_id", "phase", "repeat", "index"]},
        }
    if coverage_status != "in_domain":
        return {
            **base,
            "status": "out_of_domain",
            "reason": coverage_reason or coverage_status,
            "pairing": {"status": "not_evaluated", "complete": False, "key_schema": ["cell_id", "phase", "repeat", "index"]},
        }
    if not isinstance(sim, dict):
        return {
            **base,
            "status": "missing_prediction",
            "reason": "simulator TTFT measurements are missing",
            "pairing": {"status": "missing_prediction", "complete": False, "key_schema": ["cell_id", "phase", "repeat", "index"]},
        }
    paired = pair_metric_measurements(
        target,
        sim,
        cell=cell,
        cell_id=cell_id,
        metric="ttft_ms",
        positive=True,
    )
    pairing = paired["pairing"]
    if not pairing["complete"]:
        return {
            **base,
            "status": "pairing_failed",
            "reason": "target and simulator TTFT measurements do not exactly match the manifest request plan",
            "pairing": pairing,
        }
    pairs = paired["pairs"]
    actual = [actual for _key, actual, _predicted in pairs]
    predicted = [predicted for _key, _actual, predicted in pairs]
    return {
        **base,
        "status": "measured",
        "pairing": pairing,
        "actual": _percentiles(actual),
        "predicted": _percentiles(predicted),
        "errors": _paired_error_metrics(pairs),
        "_pairs": pairs,
    }


def build_report(
    campaign: dict[str, Any],
    profile: dict[str, Any],
    selection: dict[str, Any],
    source_digests: dict[str, str],
    raw_variants: dict[str, dict[str, dict[str, Any]]],
    sim_results: dict[str, dict[str, Any] | None],
    *,
    independent_campaign: bool = False,
) -> dict[str, Any]:
    errors_by_split: dict[str, dict[str, dict[str, list[float]]]] = {}
    errors_by_split_kind: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    cells: list[dict[str, Any]] = []
    coverage_counts = {"in_domain": 0, "out_of_domain": 0, "unsupported": 0}
    failures: list[dict[str, Any]] = []
    queueing_ttft: list[dict[str, Any]] = []
    queueing_samples_by_split: dict[str, list[tuple[Any, float, float]]] = {}
    for cell in campaign.get("workload", {}).get("cells", []):
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str):
            raise ReplayError("workload cell has no cell_id")
        variants = raw_variants.get(cell_id)
        if not variants or "target" not in variants or "zero_cost" not in variants:
            failures.append({"cell_id": cell_id, "reason": "target or zero-cost artifact missing"})
            continue
        target = variants["target"]
        zero_cost = variants["zero_cost"]
        coverage_status, coverage_reason, fitted = _predict_cell(profile, cell)
        coverage_counts[coverage_status] += 1
        sim = sim_results.get(cell_id)
        split = str(cell.get("split"))
        kind = str(cell.get("kind"))
        split_bucket = errors_by_split.setdefault(split, {})
        kind_bucket = errors_by_split_kind.setdefault(split, {}).setdefault(kind, {})
        if kind == "steady_decode" and split in ("threshold_validation", "held_out_mixed", "sealed_test"):
            queueing_record = _queueing_ttft_record(
                cell,
                cell_id,
                target,
                sim,
                coverage_status,
                coverage_reason,
            )
            queueing_samples_by_split.setdefault(split, []).extend(queueing_record.pop("_pairs", []))
            queueing_ttft.append(queueing_record)
        cell_errors: dict[str, Any] = {}
        effective: dict[str, float] = {}
        baseline_plus_fit: dict[str, float] = {}
        for metric in METRICS:
            effective_value = _effective(target, zero_cost, metric)
            fit_value = fitted.get(metric)
            baseline = _summary_value(zero_cost, metric)
            target_value = _summary_value(target, metric)
            sim_value = (
                _summary_value(sim, metric)
                if isinstance(sim, dict) and sim.get("status") == "succeeded"
                else None
            )
            if effective_value is not None:
                effective[metric] = effective_value
            if fit_value is not None and baseline is not None:
                baseline_plus_fit[metric] = baseline + fit_value
            request_samples = _request_error_samples(target, zero_cost, sim, metric, fit_value)
            request_errors: dict[str, dict[str, Any]] = {}
            for family, (observed, predicted) in request_samples.items():
                _add_error_samples(split_bucket, family, metric, observed, predicted)
                _add_error_samples(kind_bucket, family, metric, observed, predicted)
                request_errors[family] = _error_metrics(observed, predicted)
            effective_error = request_errors.get("effective_vs_fitted")
            baseline_error = request_errors.get("target_vs_baseline_plus_fitted")
            sim_error = request_errors.get("target_vs_sim")
            # Older hand-built reports contain summaries only.  Keep their
            # scalar behavior while real artifacts always take the branch above.
            if effective_error is None and effective_value is not None and fit_value is not None:
                _add_error(split_bucket, "effective_vs_fitted", metric, effective_value, fit_value)
                _add_error(kind_bucket, "effective_vs_fitted", metric, effective_value, fit_value)
                effective_error = _error_metrics([effective_value], [fit_value])
            if baseline_error is None and target_value is not None and baseline_plus_fit.get(metric) is not None:
                _add_error(
                    split_bucket, "target_vs_baseline_plus_fitted", metric, target_value, baseline_plus_fit[metric]
                )
                _add_error(
                    kind_bucket, "target_vs_baseline_plus_fitted", metric, target_value, baseline_plus_fit[metric]
                )
            if sim_error is None and target_value is not None and sim_value is not None:
                _add_error(split_bucket, "target_vs_sim", metric, target_value, sim_value)
                _add_error(kind_bucket, "target_vs_sim", metric, target_value, sim_value)
                sim_error = _error_metrics([target_value], [sim_value])
            cell_errors.setdefault("effective_vs_fitted", {})[metric] = effective_error
            cell_errors.setdefault("target_vs_baseline_plus_fitted", {})[metric] = baseline_error
            cell_errors.setdefault("target_vs_sim", {})[metric] = sim_error
        if isinstance(sim, dict) and sim.get("status") not in ("succeeded", "not_replayed"):
            failures.extend({"cell_id": cell_id, "variant": "sim", **failure} for failure in sim.get("failures", []))
        cells.append(
            {
                "cell_id": cell_id,
                "name": cell.get("name"),
                "kind": cell.get("kind"),
                "split": split,
                "shape": _cell_shape(cell),
                "coverage": {"status": coverage_status, "reason": coverage_reason},
                "pairing": {
                    "target_zero_cost": _request_keys(target, cell_id) == _request_keys(zero_cost, cell_id),
                    "target_sim": None if sim is None else _request_keys(target, cell_id) == _request_keys(sim, cell_id),
                },
                "target": {"status": target.get("status"), "summary": _summary(target), "dispersion": {metric: _dispersion(target, metric) for metric in METRICS}},
                "zero_cost": {"status": zero_cost.get("status"), "summary": _summary(zero_cost), "dispersion": {metric: _dispersion(zero_cost, metric) for metric in METRICS}},
                "effective": {"p50_ms": effective, "attribution": "target_minus_zero_cost"},
                "fitted": {"status": "available" if fitted else "unsupported", "metrics": fitted, "baseline_plus_fitted_p50_ms": baseline_plus_fit},
                "sim": ({"status": "not_replayed", "reason": coverage_reason} if sim is None else {"status": sim.get("status"), "summary": _summary(sim), "failures": sim.get("failures", [])}),
                "errors": cell_errors,
            }
        )
    rendered_aggregates = {split: _render_errors(bucket) for split, bucket in errors_by_split.items()}
    rendered_family_aggregates = {
        split: {kind: _render_errors(bucket) for kind, bucket in kinds.items()}
        for split, kinds in errors_by_split_kind.items()
    }
    all_bucket: dict[str, dict[str, list[float]]] = {}
    for bucket in errors_by_split.values():
        for family, metrics in bucket.items():
            for metric, values in metrics.items():
                all_bucket.setdefault(family, {}).setdefault(metric, [[], []])
                all_bucket[family][metric][0].extend(values[0])
                all_bucket[family][metric][1].extend(values[1])
    sim_failures = any(item.get("variant") == "sim" for item in failures)
    queueing_failures = [
        {
            "cell_id": item.get("cell_id"),
            "variant": "sim",
            "reason": item.get("reason", "queueing TTFT pairing failed"),
        }
        for item in queueing_ttft
        if item.get("status") in ("pairing_failed", "missing_prediction")
    ]
    failures.extend(queueing_failures)
    required_kinds = list(PURE_COVERAGE_KINDS)
    if profile.get("predictor", {}).get("model", {}).get("mixed") is not None:
        required_kinds.append("mixed")
    coverage_audit = {
        split: audit_publication_coverage(cells, required_splits=(split,), required_kinds=tuple(required_kinds))
        for split in ("threshold_validation", "sealed_test")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign.get("campaign_id"),
        "status": "partial" if sim_failures or queueing_failures else "succeeded",
        "provenance": {
            **source_digests,
            "campaign_manifest_sha256": source_digests.get("campaign_manifest_sha256"),
            "artifact_index_sha256": source_digests.get("artifact_index_sha256"),
            "profile_selection_status": selection.get("status"),
            "profile_campaign_id": selection.get("campaign_id"),
            "independent_campaign": independent_campaign,
            "target_identity": campaign.get("target"),
        },
        "coverage": coverage_counts,
        "coverage_required_kinds": required_kinds,
        "coverage_audit": coverage_audit,
        "queueing_ttft": {
            "metric": "ttft_ms",
            "basis": "functional_profile_replay",
            "splits": ["threshold_validation", "held_out_mixed", "sealed_test"],
            "fit_excluded": True,
            "cells": queueing_ttft,
            "aggregates": {
                split: {
                    "actual": _percentiles([actual for _key, actual, _predicted in pairs]),
                    "predicted": _percentiles([predicted for _key, _actual, predicted in pairs]),
                    "errors": _paired_error_metrics(pairs),
                }
                for split, pairs in queueing_samples_by_split.items()
            },
        },
        "cells": cells,
        "aggregates": {
            "by_split": rendered_aggregates,
            "by_split_family": rendered_family_aggregates,
            "all": _render_errors(all_bucket),
        },
        "failures": failures,
        "signoff": {"status": "characterization_only", "reason": "selected profile uses characterization evidence" if selection.get("status") == "characterization" else "not a W5 signoff"},
    }


def _write_document(path: Path, document: dict[str, Any]) -> None:
    rendered = json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise ReplayError(f"refusing to overwrite a different report artifact: {path}")
    if not path.exists():
        path.write_text(rendered, encoding="utf-8")
