#!/usr/bin/env python3
"""Normalize sealed paired campaign artifacts without mutating raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

from pairing import canonical_pair_key, expected_measured_keys, pairing_key_text


SCHEMA_VERSION = 1
PAIRED_SCHEMA_VERSIONS = (2, 3)
VARIANTS = ("target", "zero_cost", "replay")
METRICS = ("ttft_ms", "tpot_ms", "e2e_ms", "stream_chunk_tpot_ms")
MAX_TOKEN_COUNT = 2**31 - 1


class NormalizeError(ValueError):
    """Raised when a campaign lineage cannot be safely normalized."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NormalizeError(f"failed to read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise NormalizeError(f"JSON artifact must be an object: {path}")
    return value, payload


def _safe_artifact_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise NormalizeError(f"artifact path escapes artifact root: {relative!r}") from exc
    return candidate


def _finite_number(value: Any, field: str, *, nonnegative: bool = True) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NormalizeError(f"{field} must be numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise NormalizeError(f"{field} must be finite")
    if nonnegative and result < 0.0:
        raise NormalizeError(f"{field} must be non-negative")
    return result


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NormalizeError(f"{field} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise NormalizeError(f"{field} must be {'non-negative' if allow_zero else 'positive'}")
    if value > MAX_TOKEN_COUNT:
        raise NormalizeError(f"{field} exceeds supported integer range")
    return value


def _median_mad(values: Iterable[float]) -> dict[str, float | int | None]:
    clean: list[float] = []
    for value in values:
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        rendered = float(value)
        if math.isfinite(rendered):
            clean.append(rendered)
    if not clean:
        return {"count": 0, "median": None, "mad": None, "min": None, "max": None}
    center = statistics.median(clean)
    deviations = [abs(value - center) for value in clean]
    return {
        "count": len(clean),
        "median": center,
        "mad": statistics.median(deviations),
        "min": min(clean),
        "max": max(clean),
    }


def _measurement_key(measurement: dict[str, Any], cell_id: str | None = None) -> str:
    phase = measurement.get("phase")
    repeat = measurement.get("repeat")
    index = measurement.get("index")
    request_id = measurement.get("request_id")
    if phase != "measured":
        raise NormalizeError(f"normalization received non-measured request phase {phase!r}")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
        raise NormalizeError("measurement.repeat must be a non-negative integer")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise NormalizeError("measurement.index must be a non-negative integer")
    if not isinstance(request_id, str) or not request_id:
        raise NormalizeError("measurement.request_id must be a non-empty string")
    if cell_id is None:
        return f"{phase}:{repeat}:{index}"
    derived = pairing_key_text(canonical_pair_key(measurement, cell_id, measured_only=True))
    explicit = measurement.get("pairing_key")
    if explicit is not None and explicit != derived:
        raise NormalizeError(f"measurement.pairing_key does not match canonical key {derived!r}")
    return derived


def _validate_measurement(measurement: Any, *, variant: str) -> dict[str, Any]:
    if not isinstance(measurement, dict):
        raise NormalizeError(f"{variant} measurement must be an object")
    if measurement.get("ok") is not True:
        raise NormalizeError(
            f"{variant} measurement {_measurement_key(measurement)!r} is not successful: "
            f"{measurement.get('error', 'unknown failure')}"
        )
    _measurement_key(measurement)
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        _positive_int(measurement.get(field), f"{variant}.{field}")
    for field in METRICS:
        _finite_number(measurement.get(field), f"{variant}.{field}")
    if "token_timing_valid" in measurement and measurement["token_timing_valid"] not in (True, False, None):
        raise NormalizeError(f"{variant}.token_timing_valid must be boolean or null")
    semantics = measurement.get("timing_semantics")
    if semantics is not None and (not isinstance(semantics, str) or not semantics):
        raise NormalizeError(f"{variant}.timing_semantics must be a non-empty string or null")
    return measurement


def _load_variant_result(
    root: Path,
    entry: dict[str, Any],
    *,
    cell_id: str,
    variant: str,
    expected_campaign_id: str,
    indexed_file: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    if entry.get("status") != "sealed":
        raise NormalizeError(f"{cell_id}:{variant} is not sealed")
    relative = entry.get("result_path")
    expected_digest = entry.get("result_sha256")
    if not isinstance(relative, str) or not relative:
        raise NormalizeError(f"{cell_id}:{variant} has no result_path")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise NormalizeError(f"{cell_id}:{variant} has no valid result_sha256")
    path = _safe_artifact_path(root, relative)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise NormalizeError(f"failed to read raw result {path}: {exc}") from exc
    actual_digest = _sha256(payload)
    if actual_digest.lower() != expected_digest.lower():
        raise NormalizeError(
            f"raw result digest mismatch for {cell_id}:{variant}: expected {expected_digest}, got {actual_digest}"
        )
    if indexed_file is None or indexed_file.get("sha256", "").lower() != actual_digest:
        raise NormalizeError(f"artifact index digest mismatch for {cell_id}:{variant}")
    try:
        record = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NormalizeError(f"failed to parse raw result {path}: {exc}") from exc
    if not isinstance(record, dict) or record.get("campaign_id") != expected_campaign_id:
        raise NormalizeError(f"raw result {path} has invalid record envelope")
    record_cell = record.get("cell")
    if (
        record.get("variant") != variant
        or not isinstance(record_cell, dict)
        or record_cell.get("cell_id") != cell_id
    ):
        raise NormalizeError(f"raw result identity mismatch for {cell_id}:{variant}")
    result = record.get("result")
    if not isinstance(result, dict) or result.get("status") != "succeeded":
        raise NormalizeError(f"raw result {path} is not a succeeded result")
    return record, actual_digest


def _variant_measurements(
    record: dict[str, Any], variant: str, *, cell_id: str
) -> tuple[dict[str, dict[str, Any]], int, bool]:
    result = record["result"]
    raw_measurements = result.get("measurements")
    if not isinstance(raw_measurements, list) or not raw_measurements:
        raise NormalizeError(f"{variant} result has no measurements")
    measured: dict[str, dict[str, Any]] = {}
    warmup_count = 0
    explicit_pairing = True
    for item in raw_measurements:
        if not isinstance(item, dict):
            raise NormalizeError(f"{variant} measurement must be an object")
        if item.get("phase") == "warmup":
            warmup_count += 1
            continue
        if not isinstance(item.get("pairing_key"), str) or not item.get("pairing_key"):
            explicit_pairing = False
        measurement = _validate_measurement(item, variant=variant)
        key = _measurement_key(measurement, cell_id)
        if key in measured:
            raise NormalizeError(f"duplicate measured request key {key!r} in {variant}")
        measured[key] = measurement
    if not measured:
        raise NormalizeError(f"{variant} result has no measured requests after warmup exclusion")
    return measured, warmup_count, explicit_pairing


def _metric_value(measurement: dict[str, Any], metric: str) -> float | None:
    value = measurement.get(metric)
    return _finite_number(value, metric)


def _build_cell(
    root: Path,
    manifest: dict[str, Any],
    index: dict[str, Any],
    cell: dict[str, Any],
) -> dict[str, Any]:
    cell_id = cell.get("cell_id")
    if not isinstance(cell_id, str) or not cell_id:
        raise NormalizeError("manifest workload cell has no cell_id")
    entries = index.get("variant_cells", {})
    if not isinstance(entries, dict):
        raise NormalizeError("artifact index variant_cells must be an object")
    raw_refs: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, dict[str, Any]]] = {}
    variant_results: dict[str, dict[str, Any]] = {}
    warmup_excluded: dict[str, int] = {}
    legacy_pairing: dict[str, bool] = {}
    missing = [variant for variant in VARIANTS if f"{cell_id}:{variant}" not in entries]
    if missing:
        return {
            "cell_id": cell_id,
            "name": cell.get("name"),
            "kind": cell.get("kind"),
            "split": cell.get("split"),
            "status": "missing",
            "reason": f"missing sealed variant cells: {', '.join(missing)}",
            "raw": {},
            "observations": [],
        }
    try:
        for variant in VARIANTS:
            entry = entries[f"{cell_id}:{variant}"]
            indexed_file = index.get("files", {}).get(entry.get("result_path"))
            record, digest = _load_variant_result(
                root,
                entry,
                cell_id=cell_id,
                variant=variant,
                expected_campaign_id=manifest["campaign_id"],
                indexed_file=indexed_file if isinstance(indexed_file, dict) else None,
            )
            measurements, excluded, explicit_pairing = _variant_measurements(record, variant, cell_id=cell_id)
            raw_refs[variant] = {
                "result_path": entry["result_path"],
                "result_sha256": digest,
            }
            records[variant] = measurements
            result = record.get("result")
            if isinstance(result, dict):
                variant_results[variant] = result
            warmup_excluded[variant] = excluded
            legacy_pairing[variant] = not explicit_pairing
    except NormalizeError as exc:
        return {
            "cell_id": cell_id,
            "name": cell.get("name"),
            "kind": cell.get("kind"),
            "split": cell.get("split"),
            "status": "invalid",
            "reason": str(exc),
            "raw": raw_refs,
            "observations": [],
        }
    cell_metadata = dict(cell.get("metadata") or {})
    if cell.get("kind") == "mixed":
        overlaps = [variant_results.get(variant, {}).get("mixed_overlap") for variant in VARIANTS]
        if any(not isinstance(overlap, dict) or overlap.get("status") != "validated" for overlap in overlaps):
            return {
                "cell_id": cell_id,
                "name": cell.get("name"),
                "kind": cell.get("kind"),
                "split": cell.get("split"),
                "status": "invalid",
                "reason": "mixed cell is missing validated overlap evidence in one or more variants",
                "raw": raw_refs,
                "observations": [],
            }
        cell_metadata.update({"overlap_status": "validated", "overlap": overlaps[0]})
    if cell.get("kind") == "cached_prefill":
        target_evidence = [item.get("cache_hit_evidence") for item in records["target"].values()]
        if any(
            not isinstance(evidence, dict)
            or evidence.get("cached_tokens", 0) <= 0
            or not any(evidence.get(field) is True for field in ("cache_hit", "prefix_cache_hit", "hit"))
            or evidence.get("duration_includes_uncached_tail") is not True
            for evidence in target_evidence
        ):
            return {
                "cell_id": cell_id,
                "name": cell.get("name"),
                "kind": cell.get("kind"),
                "split": cell.get("split"),
                "status": "invalid",
                "reason": "cached-prefill target is missing positive cache-hit and uncached-tail evidence",
                "raw": raw_refs,
                "observations": [],
            }
    key_sets = {variant: set(values) for variant, values in records.items()}
    common_keys = set.intersection(*key_sets.values())
    all_keys = set.union(*key_sets.values())
    try:
        expected_keys = {
            pairing_key_text(key)
            for key in expected_measured_keys(cell, cell_id)
        }
    except ValueError as exc:
        return {
            "cell_id": cell_id,
            "name": cell.get("name"),
            "kind": cell.get("kind"),
            "split": cell.get("split"),
            "status": "invalid",
            "reason": f"invalid manifest pairing contract: {exc}",
            "raw": raw_refs,
            "observations": [],
        }
    if any(keys != common_keys for keys in key_sets.values()) or common_keys != expected_keys:
        return {
            "cell_id": cell_id,
            "name": cell.get("name"),
            "kind": cell.get("kind"),
            "split": cell.get("split"),
            "status": "invalid",
            "reason": "variant request-key sets do not match the manifest request plan",
            "raw": raw_refs,
            "warmup_excluded": warmup_excluded,
            "request_key_count": {variant: len(keys) for variant, keys in key_sets.items()},
            "expected_request_key_count": len(expected_keys),
            "unpaired_request_keys": sorted(all_keys - common_keys),
            "missing_expected_request_keys": sorted(expected_keys - common_keys),
            "observations": [],
        }
    request_id_mismatches = sorted(
        key
        for key in common_keys
        if len({records[variant][key].get("request_id") for variant in VARIANTS}) > 1
    )
    observations: list[dict[str, Any]] = []
    attribution_labels: set[str] = set()
    for key in sorted(common_keys):
        target = records["target"][key]
        zero_cost = records["zero_cost"][key]
        replay = records["replay"][key]
        effective: dict[str, float | None] = {}
        labels: dict[str, str] = {}
        for metric in METRICS:
            target_value = _metric_value(target, metric)
            baseline_value = _metric_value(zero_cost, metric)
            if target_value is None or baseline_value is None:
                labels[metric] = "attribution_uncertain"
                effective[metric] = None
            else:
                delta = target_value - baseline_value
                if not math.isfinite(delta) or delta < 0.0:
                    labels[metric] = "attribution_uncertain"
                    effective[metric] = None
                else:
                    labels[metric] = "black_box_effective"
                    effective[metric] = delta
            attribution_labels.add(labels[metric])
        observations.append(
            {
                "key": key,
                "cell_id": cell_id,
                "name": cell.get("name"),
                "kind": cell.get("kind"),
                "split": cell.get("split"),
                "input_tokens": cell.get("input_tokens"),
                "output_tokens": cell.get("output_tokens"),
                "concurrency": cell.get("concurrency"),
                "sum_decode_ctx_tokens": cell.get("sum_decode_ctx_tokens"),
                "prefill_input_tokens": cell.get("prefill_input_tokens", cell.get("input_tokens")),
                "decode_requests": (
                    cell.get("mixed_shape", {}).get("background", {}).get("requests")
                    if isinstance(cell.get("mixed_shape"), dict)
                    else cell.get("concurrency")
                ),
                "mixed_shape": cell.get("mixed_shape"),
                "prefix_tokens": cell.get("prefix_tokens", 0),
                "shared_prefix": cell.get("shared_prefix"),
                "metadata": cell_metadata,
                "repeat": target["repeat"],
                "index": target["index"],
                "request_id": target["request_id"],
                "target": {metric: target.get(metric) for metric in METRICS},
                "zero_cost": {metric: zero_cost.get(metric) for metric in METRICS},
                "replay": {metric: replay.get(metric) for metric in METRICS},
                "effective": effective,
                "attribution": labels,
                "timing_semantics": {
                    variant: records[variant][key].get("timing_semantics") for variant in VARIANTS
                },
                "token_timing_valid": {
                    variant: records[variant][key].get("token_timing_valid") for variant in VARIANTS
                },
                "cache_hit_evidence": {
                    variant: records[variant][key].get("cache_hit_evidence") for variant in VARIANTS
                },
            }
        )
    # A paired delta is only attributable when repeated requests at the same
    # index agree. Large cross-repeat MAD is retained as evidence but excluded
    # from candidate fitting instead of being silently clamped.
    for metric in METRICS:
        # Cold-prefill TTFT is the curve being fit at concurrency=1.  Its
        # repeated measurements can include scheduler/GPU startup jitter; the
        # piecewise fitter takes a robust median per input knot, so dropping an
        # entire input length when its paired deltas are noisy would create an
        # artificial OOD hole.  Keep those deltas while retaining the stricter
        # instability rejection for decode and for all other metrics.
        if cell.get("kind") == "cold_prefill" and metric == "ttft_ms":
            continue
        by_index: dict[int, list[float]] = {}
        for observation in observations:
            value = observation["effective"][metric]
            if value is None:
                continue
            by_index.setdefault(observation["index"], []).append(float(value))
        unstable_indices: set[int] = set()
        for index, values in by_index.items():
            if len(values) < 2:
                continue
            center = statistics.median(values)
            mad = statistics.median(abs(value - center) for value in values)
            if center <= 0.0 or mad > abs(center) * 0.5:
                unstable_indices.add(index)
        for observation in observations:
            if observation["index"] in unstable_indices and observation["effective"][metric] is not None:
                observation["effective"][metric] = None
                observation["attribution"][metric] = "attribution_uncertain"
                attribution_labels.add("attribution_uncertain")
    summaries: dict[str, Any] = {}
    for metric in METRICS:
        summaries[metric] = {
            "target": _median_mad(_metric_value(records["target"][key], metric) for key in common_keys),
            "zero_cost": _median_mad(_metric_value(records["zero_cost"][key], metric) for key in common_keys),
            "replay": _median_mad(_metric_value(records["replay"][key], metric) for key in common_keys),
            "effective": _median_mad(
                observation["effective"][metric]
                for observation in observations
                if observation["effective"][metric] is not None
            ),
        }
    return {
        "cell_id": cell_id,
        "name": cell.get("name"),
        "kind": cell.get("kind"),
        "split": cell.get("split"),
        "status": "measured",
        "raw": raw_refs,
        "warmup_excluded": warmup_excluded,
        "request_key_count": len(common_keys),
        "pairing": {
            "key_schema": ["cell_id", "phase", "repeat", "index"],
            "legacy": any(legacy_pairing.values()),
            "legacy_variants": [variant for variant, legacy in legacy_pairing.items() if legacy],
            "request_id_mismatches": request_id_mismatches,
        },
        "summaries": summaries,
        "attribution_labels": sorted(attribution_labels),
        "metadata": cell_metadata,
        "prefill_input_tokens": cell.get("prefill_input_tokens", cell.get("input_tokens")),
        "sum_decode_ctx_tokens": cell.get("sum_decode_ctx_tokens"),
        "mixed_shape": cell.get("mixed_shape"),
        "observations": observations,
    }


def normalize_campaign(root: Path) -> dict[str, Any]:
    """Validate a sealed artifact store and return a new normalized document."""

    root = root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    index_path = root / "artifact_index.json"
    manifest, manifest_bytes = _read_json(manifest_path)
    index, index_bytes = _read_json(index_path)
    campaign_id = manifest.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise NormalizeError("manifest.campaign_id must be non-empty")
    if manifest.get("schema_version") not in PAIRED_SCHEMA_VERSIONS:
        raise NormalizeError("normalization requires paired manifest schema_version=2 or 3")
    if index.get("campaign_id") != campaign_id:
        raise NormalizeError("artifact index campaign_id does not match manifest")
    indexed_manifest = index.get("files", {}).get("manifest.json")
    actual_manifest_digest = _sha256(manifest_bytes)
    if not isinstance(indexed_manifest, dict) or indexed_manifest.get("sha256", "").lower() != actual_manifest_digest:
        raise NormalizeError("artifact index manifest digest does not match manifest.json")
    cells = manifest.get("workload", {}).get("cells")
    if not isinstance(cells, list) or not cells:
        raise NormalizeError("manifest.workload.cells must be a non-empty list")
    normalized_cells = [_build_cell(root, manifest, index, cell) for cell in cells]
    observations = [
        observation
        for cell in normalized_cells
        for observation in cell.get("observations", [])
    ]
    failures = [
        {
            "cell_id": cell["cell_id"],
            "status": cell["status"],
            "reason": cell.get("reason"),
        }
        for cell in normalized_cells
        if cell["status"] != "measured"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "status": "succeeded" if not failures else "invalid",
        "attribution_mode": "paired_request_delta",
        "manifest_sha256": actual_manifest_digest,
        "artifact_index_sha256": _sha256(index_bytes),
        "source": {
            "artifact_root": str(root),
            "manifest": "manifest.json",
            "artifact_index": "artifact_index.json",
            "target_identity": manifest.get("target"),
            "runtime": manifest.get("runtime"),
            "cache_policy": manifest.get("runtime", {}).get("cache")
            if isinstance(manifest.get("runtime"), dict)
            else None,
            "source_digests": {
                "manifest_sha256": actual_manifest_digest,
                "artifact_index_sha256": _sha256(index_bytes),
            },
        },
        "cell_count": len(normalized_cells),
        "measured_cell_count": sum(cell["status"] == "measured" for cell in normalized_cells),
        "failure_count": len(failures),
        "failures": failures,
        "cells": normalized_cells,
        "observations": observations,
    }


def write_document(path: Path, document: dict[str, Any]) -> None:
    """Write an immutable normalized document, allowing an identical retry."""

    rendered = json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise NormalizeError(f"refusing to overwrite a different normalized artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        document = normalize_campaign(args.artifact_root)
        write_document(args.output, document)
    except (OSError, NormalizeError) as exc:
        print(f"normalization error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: document[key] for key in ("campaign_id", "status", "cell_count", "measured_cell_count", "failure_count")}, sort_keys=True))
    return 0 if document["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
