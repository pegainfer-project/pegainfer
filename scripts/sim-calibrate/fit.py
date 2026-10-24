#!/usr/bin/env python3
"""Fit bounded calibration candidates from normalized paired observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from steady_state import StatisticsError, aggregate_request_samples


SCHEMA_VERSION = 2
FIT_SPLITS = ("train",)
VALIDATION_SPLITS = ("threshold_validation",)
HELD_OUT_SPLITS = ("held_out_mixed", "sealed_test")
WEIGHT_EPSILON_MS = 1e-3
WEIGHT_MIN = 0.01
WEIGHT_MAX = 100.0


class FitError(ValueError):
    """Raised when a candidate cannot be fit or evaluated safely."""


def _finite(value: Any, field: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FitError(f"{field} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered):
        raise FitError(f"{field} must be finite")
    if nonnegative and rendered < 0.0:
        raise FitError(f"{field} must be non-negative")
    return rendered


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _median_mad(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return {"count": 0, "median": None, "mad": None, "min": None, "max": None}
    center = statistics.median(clean)
    return {
        "count": len(clean),
        "median": center,
        "mad": statistics.median(abs(value - center) for value in clean),
        "min": clean[0],
        "max": clean[-1],
    }


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    position = (len(clean) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    return clean[lower] + (clean[upper] - clean[lower]) * (position - lower)


def _error_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    absolute: list[float] = []
    relative: list[float] = []
    for observed, estimate in zip(actual, predicted):
        observed = _finite(observed, "observed")
        estimate = _finite(estimate, "prediction")
        error = abs(estimate - observed)
        absolute.append(error)
        if observed > 0.0:
            relative.append(error / observed)
    return {
        "count": len(absolute),
        "absolute_ms": {f"p{int(q * 100)}": _percentile(absolute, q) for q in (0.50, 0.95, 0.99)},
        "relative": {f"p{int(q * 100)}": _percentile(relative, q) for q in (0.50, 0.95, 0.99)},
    }


def _training_observations(document: dict[str, Any], *, kind: str, metric: str) -> list[dict[str, Any]]:
    """Only fully verified current training cells reach numerical algorithms."""
    result = []
    for cell in document['cells']:
        if cell['kind'] != kind or cell['split'] != 'train' or cell['status'] != 'measured':
            continue
        for row in cell['observations']:
            if metric in row['fit_policy']['excluded_metrics']:
                continue
            expected = 'fixed_batch_decode_window' if kind == 'steady_decode' else 'paired_request_ttft'
            if row['metric_source'] != expected:
                raise FitError('incorrect training metric source')
            result.append({
                'cell_id': row['cell_id'], 'split': row['split'], 'repeat': row['repeat'], 'index': row['index'],
                'input_tokens': row['input_tokens'], 'requests': row['concurrency'],
                'kv_tokens': row['sum_decode_ctx_tokens'],
                'value': _finite(row['effective'][metric], 'paired effective', nonnegative=False),
                'metric': metric, 'dispersion': cell['training']['dispersion'],
                'timing_semantics': {metric: expected}, 'token_timing_valid': {},
            })
    return result


def _source_metadata(samples: Sequence[dict[str, Any]], *, metric: str) -> dict[str, Any]:
    cells = sorted({sample.get("cell_id") for sample in samples if sample.get("cell_id")})
    dispersions = [sample["dispersion"].get("mad") for sample in samples if sample.get("dispersion", {}).get("mad") is not None]
    timing = sorted(
        {
            str(value)
            for sample in samples
            for value in sample.get("timing_semantics", {}).values()
            if value is not None
        }
    )
    token_flags = sorted(
        {
            str(value)
            for sample in samples
            for value in sample.get("token_timing_valid", {}).values()
            if value is not None
        }
    )
    return {
        "metric": metric,
        "source_cells": cells,
        "observation_count": sum(int(sample.get("repeat_count", 1)) for sample in samples),
        "request_count": sum(int(sample.get("request_count", 1)) for sample in samples),
        "fit_point_count": len(samples),
        "dispersion": _median_mad(dispersions),
        "timing_semantics": timing,
        "token_timing_valid": token_flags,
        "attribution": "black_box_effective",
    }


def _piecewise_fit(samples: Sequence[dict[str, Any]], *, metric: str) -> dict[str, Any]:
    by_x: dict[float, list[float]] = {}
    for sample in samples:
        by_x.setdefault(sample["input_tokens"], []).append(sample["value"])
    knots = [
        {"x": x, "y": statistics.median(values), "dispersion": _median_mad(values)}
        for x, values in sorted(by_x.items())
    ]
    if not knots:
        raise FitError("no measured samples for cold prefill")
    source = _source_metadata(samples, metric=metric)
    return {
        "name": "cold_prefill_piecewise_linear",
        "status": "fitted",
        "kind": "cold_prefill",
        "metric": metric,
        "timing_semantics": "black_box_effective",
        "interpolation": "piecewise_linear_inside_measured_domain",
        "knots": knots,
        "coverage": {
            "input_tokens": {"min": knots[0]["x"], "max": knots[-1]["x"], "points": len(knots)},
            "ood_policy": "fail",
        },
        "source": source,
    }


def evaluate_piecewise(candidate: dict[str, Any], input_tokens: float) -> float:
    x = _finite(input_tokens, "input_tokens")
    knots = candidate.get("knots")
    if not isinstance(knots, list) or not knots:
        raise FitError("piecewise candidate has no knots")
    if x < knots[0]["x"] or x > knots[-1]["x"]:
        raise FitError(f"input_tokens={x:g} is outside measured domain")
    for left, right in zip(knots, knots[1:]):
        if x <= right["x"]:
            if right["x"] == left["x"]:
                return _finite(right["y"], "piecewise prediction")
            fraction = (x - left["x"]) / (right["x"] - left["x"])
            return _finite(left["y"] + fraction * (right["y"] - left["y"]), "piecewise prediction")
    return _finite(knots[-1]["y"], "piecewise prediction")


def _dispersion_weight(mad: float | None) -> float:
    if mad is None or not math.isfinite(float(mad)):
        return 1.0
    raw = 1.0 / max(float(mad), WEIGHT_EPSILON_MS) ** 2
    return min(WEIGHT_MAX, max(WEIGHT_MIN, raw))


def _aggregate_decode_samples(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for sample in samples:
        key = (sample["input_tokens"], sample["requests"], sample["kv_tokens"])
        grouped.setdefault(key, []).append(sample)
    aggregated: list[dict[str, Any]] = []
    for _, rows in sorted(grouped.items()):
        try:
            summary = aggregate_request_samples(rows)
        except StatisticsError as exc:
            raise FitError(str(exc)) from exc
        cells = sorted({row.get("cell_id") for row in rows if row.get("cell_id")})
        representative = {**rows[0], **summary,
                          "fit_weight": _dispersion_weight(summary["dispersion"]["mad"]),
                          "source_cells": cells}
        aggregated.append(representative)
    return aggregated


def _nnls_additive(samples: Sequence[dict[str, Any]]) -> tuple[float, float, float]:
    """Weighted coordinate-descent non-negative fit for d0+d_requests+d_kv."""

    features = [(1.0, sample["requests"] - 1.0, sample["kv_tokens"]) for sample in samples]
    values = [sample["value"] for sample in samples]
    weights = [float(sample.get("fit_weight", 1.0)) for sample in samples]
    coefficients = [max(0.0, statistics.median(values)), 0.0, 0.0]
    for _ in range(256):
        changed = 0.0
        for index in range(3):
            numerator = 0.0
            denominator = 0.0
            for row, value, weight in zip(features, values, weights):
                residual = value - sum(row[j] * coefficients[j] for j in range(3) if j != index)
                numerator += weight * row[index] * residual
                denominator += weight * row[index] * row[index]
            updated = max(0.0, numerator / denominator) if denominator else 0.0
            changed = max(changed, abs(updated - coefficients[index]))
            coefficients[index] = updated
        if changed < 1e-10:
            break
    if not all(math.isfinite(value) and value >= 0.0 for value in coefficients):
        raise FitError("additive fit produced invalid coefficients")
    return tuple(coefficients)  # type: ignore[return-value]


def _additive_fit(samples: Sequence[dict[str, Any]], *, metric: str) -> dict[str, Any]:
    if not samples:
        raise FitError("no measured samples for decode additive candidate")
    aggregated = _aggregate_decode_samples(samples)
    d0, d_requests, d_kv = _nnls_additive(aggregated)
    source = _source_metadata(aggregated, metric=metric)
    source["fit_weighting"] = {
        "method": "inverse_mad_squared",
        "center": "repeat_mean",
        "epsilon_ms": WEIGHT_EPSILON_MS,
        "min": WEIGHT_MIN,
        "max": WEIGHT_MAX,
    }
    return {
        "name": "decode_additive",
        "status": "fitted",
        "kind": "steady_decode",
        "metric": metric,
        "timing_semantics": "fixed_batch_decode_window",
        "formula": "d0 + d_requests * (requests - 1) + d_kv * kv_tokens",
        "coefficients": {"d0_ms": d0, "d_requests_ms": d_requests, "d_kv_ms_per_token": d_kv},
        "coverage": {
            "requests": {"min": min(sample["requests"] for sample in samples), "max": max(sample["requests"] for sample in samples)},
            "kv_tokens": {"min": min(sample["kv_tokens"] for sample in samples), "max": max(sample["kv_tokens"] for sample in samples)},
            "ood_policy": "fail",
        },
        "source": source,
    }


def evaluate_additive(candidate: dict[str, Any], requests: float, kv_tokens: float) -> float:
    requests = _finite(requests, "requests")
    kv_tokens = _finite(kv_tokens, "kv_tokens")
    coverage = candidate.get("coverage", {})
    for field, value in (("requests", requests), ("kv_tokens", kv_tokens)):
        bounds = coverage.get(field, {})
        if value < bounds.get("min", value) or value > bounds.get("max", value):
            raise FitError(f"{field}={value:g} is outside measured domain")
    coefficients = candidate["coefficients"]
    return _finite(
        coefficients["d0_ms"] + coefficients["d_requests_ms"] * (requests - 1.0) + coefficients["d_kv_ms_per_token"] * kv_tokens,
        "additive prediction",
    )


def _surface_fit(samples: Sequence[dict[str, Any]], *, metric: str) -> dict[str, Any]:
    if not samples:
        raise FitError("no measured samples for decode surface candidate")
    aggregated = _aggregate_decode_samples(samples)
    requests = sorted({sample["requests"] for sample in aggregated})
    kv_tokens = sorted({sample["kv_tokens"] for sample in aggregated})
    if len(requests) < 2 or len(kv_tokens) < 2:
        raise FitError("decode surface requires at least two request and kv points")
    values = {(sample["requests"], sample["kv_tokens"]): sample["value"] for sample in aggregated}
    missing = [(request, kv) for request in requests for kv in kv_tokens if (request, kv) not in values]
    if missing:
        raise FitError(f"decode surface grid is incomplete: {missing[:4]}")
    grid = [[values[(request, kv)] for kv in kv_tokens] for request in requests]
    return {
        "name": "decode_2d_surface",
        "status": "fitted",
        "kind": "steady_decode",
        "metric": metric,
        "timing_semantics": "fixed_batch_decode_window",
        "interpolation": "bilinear_inside_measured_domain",
        "axes": {"requests": requests, "kv_tokens": kv_tokens},
        "values_ms": grid,
        "coverage": {
            "requests": {"min": requests[0], "max": requests[-1]},
            "kv_tokens": {"min": kv_tokens[0], "max": kv_tokens[-1]},
            "ood_policy": "fail",
        },
        "source": _source_metadata(aggregated, metric=metric),
    }


def _cached_observations(document: dict[str, Any], *, metric: str) -> tuple[list[dict[str, Any]], list[str]]:
    samples: list[dict[str, Any]] = []
    rejected: list[str] = []
    for cell in document.get("cells", []):
        if not isinstance(cell, dict) or cell.get("status") != "measured":
            continue
        if cell.get("kind") != "cached_prefill" or cell.get("split") not in FIT_SPLITS:
            continue
        for observation in cell.get("observations", []):
            if not isinstance(observation, dict):
                continue
            reason: str | None = None
            evidence = observation.get("cache_hit_evidence", {}).get("target")
            if not isinstance(evidence, dict):
                reason = "missing positive cache hit evidence"
            elif not isinstance(evidence.get("cached_tokens"), int) or evidence.get("cached_tokens", 0) <= 0:
                reason = "cached_tokens evidence must be positive"
            elif not any(
                isinstance(evidence.get(field), str) and evidence.get(field)
                for field in ("shared_prefix_id", "shared_prefix_hash", "prefix_hash")
            ):
                reason = "missing shared-prefix identity evidence"
            elif not isinstance(observation.get("shared_prefix"), dict):
                reason = "missing manifest shared-prefix identity"
            elif evidence.get("shared_prefix_id") != observation["shared_prefix"].get("identity"):
                reason = "shared-prefix identity evidence does not match manifest"
            elif not (
                evidence.get("cache_hit") is True
                or evidence.get("prefix_cache_hit") is True
                or evidence.get("hit") is True
            ):
                reason = "cache hit evidence is not positive"
            elif evidence.get("duration_includes_uncached_tail") is not True:
                reason = "duration lacks explicit uncached-tail accounting"
            if reason is not None:
                rejected.append(f"{observation.get('cell_id', '<unknown>')}: {reason}")
                continue
            if observation.get("attribution", {}).get(metric) != "black_box_effective":
                rejected.append(f"{observation.get('cell_id', '<unknown>')}: attribution is uncertain")
                continue
            try:
                cached_tokens = _finite(evidence["cached_tokens"], "cached_tokens")
                input_tokens = _finite(observation.get("input_tokens"), "input_tokens")
                value = _finite(observation.get("effective", {}).get(metric), f"effective.{metric}")
            except FitError as exc:
                rejected.append(f"{observation.get('cell_id', '<unknown>')}: {exc}")
                continue
            new_tokens = input_tokens - cached_tokens
            if new_tokens <= 0.0:
                rejected.append(f"{observation.get('cell_id', '<unknown>')}: uncached new_tokens must be positive")
                continue
            if observation.get("prefix_tokens") != int(cached_tokens):
                rejected.append(f"{observation.get('cell_id', '<unknown>')}: metadata prefix_tokens disagrees with evidence")
                continue
            if evidence.get("expected_cached_tokens") != int(cached_tokens):
                rejected.append(f"{observation.get('cell_id', '<unknown>')}: expected cached token count disagrees with evidence")
                continue
            samples.append(
                {
                    "cell_id": observation.get("cell_id"),
                    "split": observation.get("split"),
                    "new_tokens": new_tokens,
                    "prefix_tokens": cached_tokens,
                    "value": value,
                    "dispersion": cell.get("summaries", {}).get(metric, {}).get("effective", {}),
                    "cache_evidence": evidence,
                    "timing_semantics": observation.get("timing_semantics", {}),
                    "token_timing_valid": observation.get("token_timing_valid", {}),
                }
            )
    return samples, rejected


def _cached_surface_fit(
    samples: Sequence[dict[str, Any]],
    *,
    metric: str,
    document: dict[str, Any],
) -> dict[str, Any]:
    if not samples:
        raise FitError("no cached-prefill observation has complete positive cache evidence")
    new_axis = sorted({sample["new_tokens"] for sample in samples})
    prefix_axis = sorted(
        {
            sample.get("prefix_tokens", sample.get("cached_tokens"))
            for sample in samples
            if sample.get("prefix_tokens", sample.get("cached_tokens")) is not None
        }
    )
    if len(new_axis) < 2 or len(prefix_axis) < 2:
        raise FitError("cached-prefill surface requires at least two new-token and prefix-token points")
    values: dict[tuple[float, float], list[float]] = {}
    for sample in samples:
        prefix_tokens = sample.get("prefix_tokens", sample.get("cached_tokens"))
        if prefix_tokens is None:
            raise FitError("cached-prefill sample has no prefix_tokens")
        values.setdefault((prefix_tokens, sample["new_tokens"]), []).append(sample["value"])
    missing = [(prefix, new) for prefix in prefix_axis for new in new_axis if (prefix, new) not in values]
    if missing:
        raise FitError(f"cached-prefill surface grid is incomplete: {missing[:4]}")
    grid = [[statistics.median(values[(prefix, new)]) for new in new_axis] for prefix in prefix_axis]
    grid_dispersion = [[_median_mad(values[(prefix, new)]) for new in new_axis] for prefix in prefix_axis]
    normalized_source = document.get("source", {})
    cache_policy = normalized_source.get("cache_policy") if isinstance(normalized_source, dict) else None
    return {
        "name": "cached_prefill_surface",
        "status": "fitted",
        "kind": "cached_prefill",
        "metric": metric,
        "timing_semantics": "black_box_effective",
        "duration_semantics": "effective_wall_including_uncached_tail",
        "formula": "prefix_cache(prefix_tokens, new_tokens)",
        "interpolation": "bilinear_inside_measured_domain",
        "axes": {"prefix_tokens": prefix_axis, "new_tokens": new_axis},
        "values_ms": grid,
        "dispersion": grid_dispersion,
        "coverage": {
            "new_tokens": {"min": new_axis[0], "max": new_axis[-1]},
            "prefix_tokens": {"min": prefix_axis[0], "max": prefix_axis[-1]},
            "ood_policy": "fail",
        },
        "source": {
            **_source_metadata(samples, metric=metric),
            "cache_policy": cache_policy,
            "block_alignment": (cache_policy or {}).get("block_alignment") if isinstance(cache_policy, dict) else None,
            "target_identity": normalized_source.get("target_identity") if isinstance(normalized_source, dict) else None,
            "source_digests": normalized_source.get("source_digests") if isinstance(normalized_source, dict) else None,
            "cache_evidence": "positive_shared_prefix_identity_and_uncached_tail",
        },
    }


def _bracket(axis: Sequence[float], value: float, name: str) -> tuple[int, int, float]:
    if value < axis[0] or value > axis[-1]:
        raise FitError(f"{name}={value:g} is outside measured domain")
    for index, right in enumerate(axis[1:], start=1):
        if value <= right:
            left = axis[index - 1]
            fraction = 0.0 if right == left else (value - left) / (right - left)
            return index - 1, index, fraction
    last = len(axis) - 1
    return last, last, 0.0


def evaluate_surface(candidate: dict[str, Any], requests: float, kv_tokens: float) -> float:
    axes = candidate["axes"]
    request_axis = axes["requests"]
    kv_axis = axes["kv_tokens"]
    left_request, right_request, request_fraction = _bracket(request_axis, _finite(requests, "requests"), "requests")
    left_kv, right_kv, kv_fraction = _bracket(kv_axis, _finite(kv_tokens, "kv_tokens"), "kv_tokens")
    values = candidate["values_ms"]
    lower = values[left_request][left_kv] + kv_fraction * (values[left_request][right_kv] - values[left_request][left_kv])
    upper = values[right_request][left_kv] + kv_fraction * (values[right_request][right_kv] - values[right_request][left_kv])
    return _finite(lower + request_fraction * (upper - lower), "surface prediction")


def evaluate_cached_surface(candidate: dict[str, Any], prefix_tokens: float, new_tokens: float) -> float:
    axes = candidate["axes"]
    prefix_axis = axes.get("prefix_tokens", axes.get("cached_tokens"))
    new_axis = axes["new_tokens"]
    if not isinstance(prefix_axis, list):
        raise FitError("cached-prefill candidate has no prefix_tokens axis")
    left_prefix, right_prefix, prefix_fraction = _bracket(
        prefix_axis, _finite(prefix_tokens, "prefix_tokens"), "prefix_tokens"
    )
    left_new, right_new, new_fraction = _bracket(new_axis, _finite(new_tokens, "new_tokens"), "new_tokens")
    values = candidate["values_ms"]
    lower = values[left_prefix][left_new] + new_fraction * (
        values[left_prefix][right_new] - values[left_prefix][left_new]
    )
    upper = values[right_prefix][left_new] + new_fraction * (
        values[right_prefix][right_new] - values[right_prefix][left_new]
    )
    return _finite(lower + prefix_fraction * (upper - lower), "cached-prefill prediction")


def _fit_or_unsupported(name: str, kind: str, metric: str, fit_fn: Any, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    try:
        return fit_fn(samples, metric=metric)
    except FitError as exc:
        return {
            "name": name,
            "status": "unsupported",
            "kind": kind,
            "metric": metric,
            "timing_semantics": "fixed_batch_decode_window" if kind == "steady_decode" else "black_box_effective",
            "reason": str(exc),
            "source": _source_metadata(samples, metric=metric),
        }


def fit_normalized(document: dict[str, Any], *, source_digest: str | None = None) -> dict[str, Any]:
    from normalize import SCHEMA_VERSION as NORMALIZED_SCHEMA, verify_normalized
    if (document.get('kind') != 'normalized_campaign' or document.get('schema_version') != NORMALIZED_SCHEMA
            or document.get('scope') != 'train' or document.get('status') != 'succeeded'
            or document.get('fit_ready') is not True):
        raise FitError('fitting requires a complete qualified current train artifact')
    try:
        verify_normalized(document)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise FitError(f'training evidence verification failed: {exc}') from exc
    prefill = _training_observations(document, kind='cold_prefill', metric='ttft_ms')
    decode = _training_observations(document, kind='steady_decode', metric='tpot_ms')
    candidates = [
        _fit_or_unsupported('cold_prefill_piecewise_linear', 'cold_prefill', 'ttft_ms', _piecewise_fit, prefill),
        _fit_or_unsupported('decode_additive', 'steady_decode', 'tpot_ms', _additive_fit, decode),
        _fit_or_unsupported('decode_2d_surface', 'steady_decode', 'tpot_ms', _surface_fit, decode),
        {'name': 'cached_prefill_surface', 'kind': 'cached_prefill', 'metric': 'ttft_ms',
         'status': 'unsupported', 'reason': 'cached training integration deferred to C8'},
    ]
    failures = [c['name'] for c in candidates if c['status'] != 'fitted'
                and ((c['kind'] == 'steady_decode' and decode) or (c['kind'] == 'cold_prefill' and prefill))]
    normalized_source = document['source']
    fitter_sources = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                     for name in ('fit.py', 'steady_state.py')}
    if source_digest is None:
        source_digest = hashlib.sha256((json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": document.get("campaign_id"),
        "status": "invalid" if failures else "succeeded",
        "failures": failures,
        "source": {
            "normalized_sha256": source_digest,
            "normalized_campaign_id": document.get("campaign_id"),
            "normalized_schema_version": document['schema_version'],
            "training": normalized_source,
            "fitter_sources": fitter_sources,
            "fit_splits": list(FIT_SPLITS),
            "validation_splits": list(VALIDATION_SPLITS),
            "held_out_splits": list(HELD_OUT_SPLITS),
            "held_out_fit_policy": "excluded",
        },
        "candidates": candidates,
        "validation": {"status": "pending", "split": list(VALIDATION_SPLITS)},
    }


def verify_fit(document: dict, normalized: dict, *, normalized_digest: str) -> None:
    """Recompute candidates from pinned qualified training; hashes alone are not proof."""
    from manifest import canonical_json
    expected = fit_normalized(normalized, source_digest=normalized_digest)
    if expected['status'] != 'succeeded' or canonical_json(document) != canonical_json(expected):
        raise FitError('fit differs from verified train evidence or fitter implementation')


def write_document(path: Path, document: dict[str, Any]) -> None:
    rendered = json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FitError(f"refusing to overwrite a different fit artifact: {path}")
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = args.normalized.read_bytes()
        document = json.loads(payload.decode("utf-8"))
        if not isinstance(document, dict):
            raise FitError("normalized artifact must be a JSON object")
        fitted = fit_normalized(document, source_digest=hashlib.sha256(payload).hexdigest())
        write_document(args.output, fitted)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, FitError) as exc:
        print(f"fit error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: fitted[key] for key in ("campaign_id", "status")}, sort_keys=True))
    return 0 if fitted['status'] == 'succeeded' else 1


if __name__ == "__main__":
    raise SystemExit(main())
