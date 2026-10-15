#!/usr/bin/env python3
"""Freeze target-specific replay tolerances from an independent validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from replay import audit_publication_coverage


SCHEMA_VERSION = 1
METRICS = ("ttft_ms", "stream_chunk_tpot_ms", "e2e_ms")
PURE_FAMILY_KINDS = {"prefill": "cold_prefill", "decode": "steady_decode"}
MIXED_FAMILY = {"mixed": "mixed"}
FAMILY_METRICS = {
    "prefill": ("ttft_ms", "e2e_ms"),
    # Concurrent decode TTFT/E2E include admission and queueing behavior. The
    # decode service-cost gate is the per-token stream timing; replay reports
    # retain the other metrics as diagnostics.
    "decode": ("stream_chunk_tpot_ms",),
    # Mixed TTFT/E2E include the composed first step; chunk timing remains a
    # diagnostic because it describes the subsequent pure decode intervals.
    "mixed": ("ttft_ms", "e2e_ms"),
}
FAMILY_ERROR_CHANNEL = {
    "prefill": "target_vs_sim",
    "decode": "target_vs_sim",
    "mixed": "effective_vs_fitted",
}
QUEUEING_FAMILY = "scheduler_queueing"
LIMIT_QUANTILES = ("p95", "p99")


class ToleranceError(ValueError):
    """Raised when validation evidence cannot freeze a tolerance artifact."""


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ToleranceError(f"failed to read {path}: {exc}") from exc


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ToleranceError(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ToleranceError(f"JSON artifact must be an object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ToleranceError(f"{field} must be finite")
    return float(value)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _dispersion(cells: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    target_mad: list[float] = []
    sim_mad: list[float] = []
    target_p50: list[float] = []
    for cell in cells:
        target = cell.get("target") or {}
        target_dispersion = (target.get("dispersion") or {}).get(metric) or {}
        sim = cell.get("sim") or {}
        sim_dispersion = (sim.get("dispersion") or {}).get(metric) or {}
        if target_dispersion.get("mad") is not None:
            target_mad.append(_finite(target_dispersion["mad"], f"target.{metric}.mad"))
        if sim_dispersion.get("mad") is not None:
            sim_mad.append(_finite(sim_dispersion["mad"], f"sim.{metric}.mad"))
        target_p50_value = ((target.get("summary") or {}).get("percentiles_ms") or {}).get(metric, {}).get("p50")
        if target_p50_value is not None:
            target_p50.append(_finite(target_p50_value, f"target.{metric}.p50"))
    target_mad_median = _median(target_mad)
    sim_mad_median = _median(sim_mad)
    absolute_margin = target_mad_median + sim_mad_median
    relative_margin = absolute_margin / max(_median(target_p50), 1.0)
    return {
        "target_mad_ms": {"median": target_mad_median, "samples": len(target_mad)},
        "sim_mad_ms": {"median": sim_mad_median, "samples": len(sim_mad)},
        "absolute_margin_ms": absolute_margin,
        "relative_margin": relative_margin,
        "target_p50_ms": _median(target_p50),
    }


def _errors_for_family(report: dict[str, Any], split: str, kind: str, channel: str) -> dict[str, Any]:
    family = (
        report.get("aggregates", {})
        .get("by_split_family", {})
        .get(split, {})
        .get(kind, {})
    )
    errors = family.get(channel)
    if not isinstance(errors, dict):
        raise ToleranceError(f"validation split has no {channel} errors for {kind}")
    return errors


def _queueing_errors(report: dict[str, Any], split: str) -> dict[str, Any]:
    """Return exact-paired TTFT errors for a validation split."""
    queueing = report.get("queueing_ttft") or {}
    cells = [
        cell
        for cell in queueing.get("cells", [])
        if cell.get("split") == split
        and cell.get("kind") == "steady_decode"
        and isinstance(cell.get("concurrency"), int)
        and cell.get("concurrency") > 1
    ]
    if not cells:
        raise ToleranceError(f"validation split has no concurrency>1 queueing TTFT cells for {split}")
    invalid = [
        cell.get("cell_id")
        for cell in cells
        if cell.get("status") != "measured" or (cell.get("pairing") or {}).get("complete") is not True
    ]
    if invalid:
        raise ToleranceError(f"queueing TTFT pairing is incomplete for {split}: {invalid}")
    aggregate = (queueing.get("aggregates") or {}).get(split)
    errors = aggregate.get("errors") if isinstance(aggregate, dict) else None
    if not isinstance(errors, dict) or errors.get("count", 0) <= 0:
        raise ToleranceError(f"queueing TTFT aggregate is missing for {split}")
    return errors


def freeze_tolerance(
    validation_report_path: Path,
    profile_path: Path,
    output_path: Path,
    *,
    validation_split: str = "threshold_validation",
) -> dict[str, Any]:
    report, report_digest = _load_json(validation_report_path)
    profile, profile_digest = _load_json(profile_path)
    selection_path = profile_path.parent / "selection.json"
    selection, selection_digest = _load_json(selection_path)
    if report.get("status") != "succeeded":
        raise ToleranceError("validation replay report is not succeeded")
    if selection.get("status") != "selected":
        raise ToleranceError("tolerance freeze requires a non-characterization profile selection")
    campaign_id = report.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ToleranceError("validation replay report campaign_id must be non-empty")
    report_provenance = report.get("provenance")
    if not isinstance(report_provenance, dict):
        raise ToleranceError("validation replay report provenance is missing")
    if report_provenance.get("profile_sha256") != profile_digest:
        raise ToleranceError("validation report profile digest does not match profile bytes")
    if report_provenance.get("selection_sha256") != selection_digest:
        raise ToleranceError("validation report selection digest does not match selection bytes")
    calibration_ref = profile.get("calibration")
    if not isinstance(calibration_ref, dict) or not isinstance(calibration_ref.get("sha256"), str):
        raise ToleranceError("profile calibration manifest digest is missing")
    if report_provenance.get("calibration_manifest_sha256") != calibration_ref["sha256"]:
        raise ToleranceError("validation report calibration manifest does not match profile")
    if report_provenance.get("profile_campaign_id") != selection.get("campaign_id"):
        raise ToleranceError("validation report profile campaign does not match selection")
    if report_provenance.get("target_identity") != selection.get("target"):
        raise ToleranceError("validation report target identity does not match selection")
    audit = (report.get("coverage_audit") or {}).get(validation_split)
    if not isinstance(audit, dict):
        audit = audit_publication_coverage(report.get("cells", []), required_splits=(validation_split,))
    if audit.get("status") != "passed":
        raise ToleranceError(f"validation coverage is not publishable: {json.dumps(audit, sort_keys=True)}")
    family_kinds = dict(PURE_FAMILY_KINDS)
    if profile.get("predictor", {}).get("model", {}).get("mixed") is not None:
        family_kinds.update(MIXED_FAMILY)
    cells = [
        cell
        for cell in report.get("cells", [])
        if cell.get("split") == validation_split
        and cell.get("kind") in family_kinds.values()
        and (cell.get("coverage") or {}).get("status") == "in_domain"
    ]
    if not cells:
        raise ToleranceError("validation replay has no in-domain required cells")
    families: dict[str, Any] = {}
    for family, kind in family_kinds.items():
        family_cells = [cell for cell in cells if cell.get("kind") == kind]
        if not family_cells:
            raise ToleranceError(f"validation replay has no in-domain {family} cells")
        error_channel = FAMILY_ERROR_CHANNEL[family]
        errors = _errors_for_family(report, validation_split, kind, error_channel)
        metrics: dict[str, Any] = {}
        for metric in FAMILY_METRICS[family]:
            metric_errors = errors.get(metric)
            if not isinstance(metric_errors, dict) or metric_errors.get("count", 0) <= 0:
                raise ToleranceError(f"validation replay has no {metric} errors for {family}")
            absolute = metric_errors.get("absolute_ms") or {}
            relative = metric_errors.get("relative") or {}
            dispersion = _dispersion(family_cells, metric)
            absolute_margin = dispersion["absolute_margin_ms"]
            relative_margin = dispersion["relative_margin"]
            metrics[metric] = {
                "basis": error_channel,
                "validation_errors": metric_errors,
                "repeat_dispersion": dispersion,
                "limits": {
                    "absolute_ms": {
                        quantile: _finite(absolute[quantile], f"{family}.{metric}.absolute.{quantile}") + absolute_margin
                        for quantile in LIMIT_QUANTILES
                    },
                    "relative": {
                        quantile: _finite(relative[quantile], f"{family}.{metric}.relative.{quantile}") + relative_margin
                        for quantile in LIMIT_QUANTILES
                    },
                },
            }
        families[family] = {"kind": kind, "cell_count": len(family_cells), "metrics": metrics}
    try:
        queueing_errors = _queueing_errors(report, validation_split)
    except ToleranceError:
        # Older reports predate the independent queueing section. They remain
        # readable, but a new validation run cannot freeze scheduler evidence
        # without concurrency>1 paired TTFT data.
        if "queueing_ttft" not in report:
            queueing_errors = None
        else:
            raise
    if queueing_errors is not None:
        absolute = queueing_errors.get("absolute_ms") or {}
        relative = queueing_errors.get("relative") or {}
        families[QUEUEING_FAMILY] = {
            "kind": "steady_decode",
            "cell_count": sum(
                1
                for cell in (report.get("queueing_ttft") or {}).get("cells", [])
                if cell.get("split") == validation_split
                and cell.get("kind") == "steady_decode"
                and isinstance(cell.get("concurrency"), int)
                and cell.get("concurrency") > 1
            ),
            "metrics": {
                "ttft_ms": {
                    "basis": "queueing_ttft",
                    "validation_errors": queueing_errors,
                    "limits": {
                        "absolute_ms": {
                            quantile: _finite(absolute[quantile], f"{QUEUEING_FAMILY}.ttft_ms.absolute.{quantile}")
                            for quantile in LIMIT_QUANTILES
                        },
                        "relative": {
                            quantile: _finite(relative[quantile], f"{QUEUEING_FAMILY}.ttft_ms.relative.{quantile}")
                            for quantile in LIMIT_QUANTILES
                        },
                    },
                }
            },
        }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "basis_split": validation_split,
        "validation_campaign_id": campaign_id,
        "target_identity": report_provenance.get("target_identity"),
        "profile": {
            "path": str(profile_path.resolve()),
            "sha256": profile_digest,
            "selection_sha256": selection_digest,
        },
        "validation_report": {
            "path": str(validation_report_path.resolve()),
            "sha256": report_digest,
            "profile_sha256": report_provenance.get("profile_sha256"),
            "selection_sha256": report_provenance.get("selection_sha256"),
            "calibration_manifest_sha256": report_provenance.get("calibration_manifest_sha256"),
            "campaign_manifest_sha256": report_provenance.get("campaign_manifest_sha256"),
            "artifact_index_sha256": report_provenance.get("artifact_index_sha256"),
        },
        "families": families,
        "scope": {
            "included_kinds": list(family_kinds.values()),
            "excluded_kinds": ["cached_prefill"],
            "policy": "validation_p95_p99_plus_repeat_mad_margin; scheduler_queueing_exact_pair",
        },
    }
    rendered = json.dumps(artifact, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_text(encoding="utf-8") != rendered:
        raise ToleranceError(f"refusing to overwrite a different tolerance artifact: {output_path}")
    if not output_path.exists():
        output_path.write_text(rendered, encoding="utf-8")
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-split", default="threshold_validation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        artifact = freeze_tolerance(
            args.validation_report,
            args.profile,
            args.output,
            validation_split=args.validation_split,
        )
    except (OSError, ToleranceError) as exc:
        print(f"tolerance error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": artifact["status"], "validation_campaign_id": artifact["validation_campaign_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
