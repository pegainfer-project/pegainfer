#!/usr/bin/env python3
"""Canonical request identity and fail-closed metric pairing helpers."""

from __future__ import annotations

import math
from typing import Any


PAIRING_KEY_SCHEMA = ("cell_id", "phase", "repeat", "index")
PairingKey = tuple[str, str, int, int]


class PairingContractError(ValueError):
    """Raised when a campaign cannot establish a canonical request identity."""


def canonical_pair_key(measurement: dict[str, Any], cell_id: str, *, measured_only: bool = False) -> PairingKey:
    """Return the variant-independent identity for one request measurement."""
    if not isinstance(cell_id, str) or not cell_id:
        raise PairingContractError("cell_id must be a non-empty string")
    phase = measurement.get("phase")
    if phase not in ("warmup", "measured"):
        raise PairingContractError(f"measurement.phase must be warmup or measured, got {phase!r}")
    if measured_only and phase != "measured":
        raise PairingContractError(f"measurement.phase must be measured, got {phase!r}")
    repeat = measurement.get("repeat")
    index = measurement.get("index")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
        raise PairingContractError("measurement.repeat must be a non-negative integer")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise PairingContractError("measurement.index must be a non-negative integer")
    return (cell_id, phase, repeat, index)


def pairing_key_text(key: PairingKey) -> str:
    """Render a key for JSON diagnostics and compatibility with old artifacts."""
    return ":".join(str(part) for part in key)


def expected_measured_keys(cell: dict[str, Any], cell_id: str) -> set[PairingKey]:
    """Build the measured request set declared by a manifest cell."""
    repeats = cell.get("repeats")
    concurrency = cell.get("concurrency")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise PairingContractError("cell.repeats must be a positive integer")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise PairingContractError("cell.concurrency must be a positive integer")
    return {
        (cell_id, "measured", repeat, index)
        for repeat in range(repeats)
        for index in range(concurrency)
    }


def _finite_metric(value: Any, metric: str, *, positive: bool) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    rendered = float(value)
    if positive and rendered <= 0.0:
        return None
    return rendered


def index_metric_measurements(
    result: dict[str, Any] | None,
    *,
    cell_id: str,
    metric: str,
    positive: bool = False,
) -> dict[str, Any]:
    """Index measured values without overwriting duplicate logical requests."""
    keys: set[PairingKey] = set()
    values: dict[PairingKey, float] = {}
    request_ids: dict[PairingKey, str] = {}
    seen_request_ids: dict[str, PairingKey] = {}
    duplicate_keys: set[PairingKey] = set()
    duplicate_request_ids: set[str] = set()
    invalid_records: list[str] = []
    invalid_metrics: list[str] = []
    invalid_request_ids: list[str] = []

    measurements = result.get("measurements") if isinstance(result, dict) else None
    if not isinstance(measurements, list):
        invalid_records.append("measurements is not a list")
        measurements = []
    for ordinal, item in enumerate(measurements):
        if not isinstance(item, dict):
            invalid_records.append(f"ordinal:{ordinal}")
            continue
        if item.get("phase") == "warmup":
            continue
        try:
            key = canonical_pair_key(item, cell_id, measured_only=True)
        except PairingContractError as exc:
            invalid_records.append(f"ordinal:{ordinal}:{exc}")
            continue
        key_text = pairing_key_text(key)
        explicit_pairing_key = item.get("pairing_key")
        if explicit_pairing_key is not None and explicit_pairing_key != key_text:
            invalid_records.append(f"{key_text}:pairing_key does not match canonical key")
        if key in keys:
            duplicate_keys.add(key)
        keys.add(key)

        request_id = item.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            invalid_request_ids.append(key_text)
        elif request_id in seen_request_ids:
            duplicate_request_ids.add(request_id)
        else:
            seen_request_ids[request_id] = key
            request_ids[key] = request_id

        if item.get("ok") is not True:
            invalid_records.append(f"{key_text}:measurement is not successful")
            continue
        value = _finite_metric(item.get(metric), metric, positive=positive)
        if value is None:
            invalid_metrics.append(key_text)
            continue
        if key not in values:
            values[key] = value

    return {
        "keys": keys,
        "values": values,
        "request_ids": request_ids,
        "duplicate_keys": duplicate_keys,
        "duplicate_request_ids": duplicate_request_ids,
        "invalid_records": invalid_records,
        "invalid_metrics": invalid_metrics,
        "invalid_request_ids": invalid_request_ids,
        "result_status": result.get("status") if isinstance(result, dict) else None,
    }


def pair_metric_measurements(
    target: dict[str, Any] | None,
    predicted: dict[str, Any] | None,
    *,
    cell: dict[str, Any],
    cell_id: str,
    metric: str,
    positive: bool = False,
) -> dict[str, Any]:
    """Pair two variants against the manifest's exact measured request set."""
    try:
        expected = expected_measured_keys(cell, cell_id)
        contract_error = None
    except PairingContractError as exc:
        expected = set()
        contract_error = str(exc)
    target_index = index_metric_measurements(target, cell_id=cell_id, metric=metric, positive=positive)
    predicted_index = index_metric_measurements(predicted, cell_id=cell_id, metric=metric, positive=positive)
    target_keys = target_index["keys"]
    predicted_keys = predicted_index["keys"]
    matched_keys = set(target_index["values"]) & set(predicted_index["values"])
    request_id_mismatches = sorted(
        pairing_key_text(key)
        for key in matched_keys
        if target_index["request_ids"].get(key) is not None
        and predicted_index["request_ids"].get(key) is not None
        and target_index["request_ids"][key] != predicted_index["request_ids"][key]
    )
    missing_target = expected - target_keys
    missing_predicted = expected - predicted_keys
    extra_target = target_keys - expected
    extra_predicted = predicted_keys - expected
    complete = (
        contract_error is None
        and target_index["result_status"] == "succeeded"
        and predicted_index["result_status"] == "succeeded"
        and target_keys == expected
        and predicted_keys == expected
        and len(target_index["values"]) == len(expected)
        and len(predicted_index["values"]) == len(expected)
        and not target_index["duplicate_keys"]
        and not predicted_index["duplicate_keys"]
        and not target_index["duplicate_request_ids"]
        and not predicted_index["duplicate_request_ids"]
        and not target_index["invalid_records"]
        and not predicted_index["invalid_records"]
        and not target_index["invalid_metrics"]
        and not predicted_index["invalid_metrics"]
        and not target_index["invalid_request_ids"]
        and not predicted_index["invalid_request_ids"]
    )
    diagnostics = {
        "status": "complete" if complete else "pairing_failed",
        "complete": complete,
        "key_schema": list(PAIRING_KEY_SCHEMA),
        "expected_count": len(expected),
        "target_count": len(target_keys),
        "predicted_count": len(predicted_keys),
        "matched_count": len(matched_keys),
        "missing_target": sorted(pairing_key_text(key) for key in missing_target),
        "missing_predicted": sorted(pairing_key_text(key) for key in missing_predicted),
        "extra_target": sorted(pairing_key_text(key) for key in extra_target),
        "extra_predicted": sorted(pairing_key_text(key) for key in extra_predicted),
        "duplicate_keys": {
            "target": sorted(pairing_key_text(key) for key in target_index["duplicate_keys"]),
            "predicted": sorted(pairing_key_text(key) for key in predicted_index["duplicate_keys"]),
        },
        "duplicate_request_ids": {
            "target": sorted(target_index["duplicate_request_ids"]),
            "predicted": sorted(predicted_index["duplicate_request_ids"]),
        },
        "invalid_records": {
            "target": target_index["invalid_records"],
            "predicted": predicted_index["invalid_records"],
        },
        "invalid_metrics": {
            "target": target_index["invalid_metrics"],
            "predicted": predicted_index["invalid_metrics"],
        },
        "invalid_request_ids": {
            "target": target_index["invalid_request_ids"],
            "predicted": predicted_index["invalid_request_ids"],
        },
        "request_id_mismatches": request_id_mismatches,
    }
    if contract_error is not None:
        diagnostics["contract_error"] = contract_error
    pairs = []
    if complete:
        pairs = [
            (key, target_index["values"][key], predicted_index["values"][key])
            for key in sorted(expected)
        ]
    return {"pairing": diagnostics, "pairs": pairs}
