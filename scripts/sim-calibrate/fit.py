#!/usr/bin/env python3
"""Fit bounded calibration candidates from normalized paired observations."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from steady_state import StatisticsError, aggregate_request_samples


SCHEMA_VERSION = 4
FIT_SPLITS = ("train",)
VALIDATION_SPLITS = ("threshold_validation",)
HELD_OUT_SPLITS = ("held_out_mixed", "sealed_test")


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


def _training_observations(document: dict[str, Any], *, kind: str, metric: str) -> list[dict[str, Any]]:
    """Use qualified normalized training cells for the numerical algorithms."""
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
        representative = {**rows[0], **summary, "source_cells": cells}
        aggregated.append(representative)
    return aggregated


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


def _cached_observations(document: dict[str, Any], *, metric: str) -> list[dict[str, Any]]:
    """Use the already-qualified cached TTFT differences."""
    samples = []
    for cell in document['cells']:
        if cell['kind'] != 'cached_prefill' or cell['split'] != 'train':
            continue
        if cell['status'] != 'measured' or cell['training']['qualified'] is not True:
            raise FitError('cached-prefill training cell is incomplete or unqualified')
        for row in cell['observations']:
            if metric in row['fit_policy']['excluded_metrics']:
                raise FitError('required cached-prefill TTFT cannot be excluded from fitting')
            if row['metric_source'] != 'paired_request_ttft' or row['reference_interval_ms'] != 0:
                raise FitError('cached-prefill requires paired TTFT against the full zero-cost reference')
            samples.append({
                'cell_id': row['cell_id'], 'split': row['split'], 'repeat': row['repeat'], 'index': row['index'],
                'prefix_tokens': row['prefix_tokens'], 'new_tokens': row['new_tokens'],
                'value': _finite(row['effective'][metric], 'paired effective', nonnegative=False),
                'dispersion': cell['training']['dispersion'],
                'timing_semantics': {metric: 'paired_request_ttft'}, 'token_timing_valid': {},
            })
    return samples


def _cached_surface_fit(
    samples: Sequence[dict[str, Any]], *, metric: str, document: dict[str, Any],
) -> dict[str, Any]:
    """Fit only the complete compiled grid, without inferring axes from survivors."""
    required = document.get('cached_requirements')
    if not isinstance(required, list) or not required:
        raise FitError('cached-prefill surface requires the compiled grid declaration')
    declared = {row['cell_id']: row for row in required}
    if len(declared) != len(required):
        raise FitError('duplicate cached-prefill compiled cell identity')
    families = {}
    for row in required:
        if row['concurrency'] != 1 or type(row['repeats']) is not int or row['repeats'] < 1:
            raise FitError('cached-prefill requires concurrency=1 and positive measured repeats')
        families.setdefault(row['matrix_name'], []).append(row)
    for name, rows in families.items():
        axes = rows[0]['axes']
        prefixes = set(axes['prefix_tokens'])
        tails = set(axes['new_tokens'])
        points = {(row['prefix_tokens'], row['new_tokens']) for row in rows}
        if (any(row['axes'] != axes for row in rows) or len(prefixes) < 2 or len(tails) < 2
                or len(points) != len(rows) or points != {(p, n) for p in prefixes for n in tails}):
            raise FitError(f'cached-prefill compiled grid is incomplete for matrix {name}')
    new_axis = sorted({row['new_tokens'] for row in required})
    prefix_axis = sorted({row['prefix_tokens'] for row in required})
    expected = {(row['cell_id'], repeat, 0) for row in required for repeat in range(row['repeats'])}
    observed = set()
    values = {}
    for sample in samples:
        identity = (sample['cell_id'], sample['repeat'], sample['index'])
        if identity in observed or identity not in expected:
            raise FitError('duplicate or unexpected cached-prefill measured sample')
        observed.add(identity)
        row = declared[sample['cell_id']]
        point = (sample['prefix_tokens'], sample['new_tokens'])
        if point != (row['prefix_tokens'], row['new_tokens']):
            raise FitError('cached-prefill sample coordinates differ from the compiled grid')
        values.setdefault(point, []).append(_finite(sample['value'], 'paired effective', nonnegative=False))
    if observed != expected:
        raise FitError('cached-prefill grid is incomplete: missing required measured samples')
    missing = [(prefix, new) for prefix in prefix_axis for new in new_axis if (prefix, new) not in values]
    if missing:
        raise FitError(f'cached-prefill surface grid is incomplete: {missing[:4]}')
    grid = [[_finite(statistics.median(values[(prefix, new)]), 'cached cell representative')
             for new in new_axis] for prefix in prefix_axis]
    dispersion = [[_median_mad(values[(prefix, new)]) for new in new_axis] for prefix in prefix_axis]
    domain = {'prefix_tokens': {'min': prefix_axis[0], 'max': prefix_axis[-1]},
              'new_tokens': {'min': new_axis[0], 'max': new_axis[-1]}, 'ood_policy': 'fail'}
    return {
        'name': 'cached_prefill_surface', 'status': 'fitted', 'kind': 'cached_prefill', 'metric': metric,
        'timing_semantics': 'black_box_effective',
        'duration_semantics': 'paired_ttft_difference_including_uncached_tail',
        'formula': 'prefix_cache(prefix_tokens, new_tokens)',
        'interpolation': 'bilinear_inside_measured_domain',
        'aggregation': 'per_request_target_minus_zero_reference_then_measured_repeat_median',
        'axes': {'prefix_tokens': prefix_axis, 'new_tokens': new_axis},
        'values_ms': grid, 'dispersion': dispersion, 'coverage': domain,
        'runtime_enabled': False, 'validation_status': 'offline_fit_only',
        'source': {**_source_metadata(samples, metric=metric), 'required_cells': required,
                   'cache_evidence': 'exact_requested_observed_prefix_and_uncached_tail'},
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
    prefix_axis = axes["prefix_tokens"]
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


def fit_normalized(document: dict[str, Any]) -> dict[str, Any]:
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
    cached = _cached_observations(document, metric='ttft_ms')
    candidates = [
        _fit_or_unsupported('cold_prefill_piecewise_linear', 'cold_prefill', 'ttft_ms', _piecewise_fit, prefill),
        _fit_or_unsupported('decode_2d_surface', 'steady_decode', 'tpot_ms', _surface_fit, decode),
        _fit_or_unsupported('cached_prefill_surface', 'cached_prefill', 'ttft_ms',
                            lambda samples, metric: _cached_surface_fit(samples, metric=metric, document=document), cached),
    ]
    failures = [c['name'] for c in candidates if c['status'] != 'fitted'
                and ((c['kind'] == 'steady_decode' and decode) or (c['kind'] == 'cold_prefill' and prefill)
                     or (c['kind'] == 'cached_prefill' and document['cached_requirements']))]
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": document.get("campaign_id"),
        "status": "invalid" if failures else "succeeded",
        "failures": failures,
        "source": {
            "fit_splits": list(FIT_SPLITS),
            "validation_splits": list(VALIDATION_SPLITS),
            "held_out_splits": list(HELD_OUT_SPLITS),
            "held_out_fit_policy": "excluded",
        },
        "candidates": candidates,
        "validation": {"status": "pending", "split": list(VALIDATION_SPLITS)},
    }


def verify_fit(document: dict) -> None:
    """Read the direct fit contract without rerunning normalization or fitting."""
    if document.get('schema_version') != SCHEMA_VERSION or document.get('status') != 'succeeded':
        raise FitError('requires a succeeded current fit artifact')


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
        fitted = fit_normalized(document)
        write_document(args.output, fitted)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, FitError) as exc:
        print(f"fit error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: fitted[key] for key in ("campaign_id", "status")}, sort_keys=True))
    return 0 if fitted['status'] == 'succeeded' else 1


if __name__ == "__main__":
    raise SystemExit(main())
