#!/usr/bin/env python3
"""Sign off an independent replay campaign against a frozen tolerance artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from replay import require_publication_coverage


SCHEMA_VERSION = 1
LIMIT_QUANTILES = ("p95", "p99")
SIGNOFF_KINDS = ("cold_prefill", "steady_decode", "mixed")
QUEUEING_FAMILY = "scheduler_queueing"


class SignoffError(ValueError):
    """Raised when an independent fidelity signoff cannot be completed."""


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise SignoffError(f"failed to read {path}: {exc}") from exc


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SignoffError(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SignoffError(f"JSON artifact must be an object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SignoffError(f"{field} must be finite")
    return float(value)


def _valid_campaign_id(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def _check_limit(
    observed: dict[str, Any],
    limit: dict[str, Any],
    *,
    family: str,
    metric: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "passed", "observed": observed, "limit": limit, "failures": []}
    for dimension, field in (("absolute_ms", "absolute_ms"), ("relative", "relative")):
        observed_values = observed.get(dimension) or {}
        limit_values = limit.get(dimension) or {}
        for quantile in LIMIT_QUANTILES:
            observed_value = observed_values.get(quantile)
            limit_value = limit_values.get(quantile)
            if observed_value is None or limit_value is None:
                result["status"] = "failed"
                result["failures"].append({"family": family, "metric": metric, "dimension": field, "quantile": quantile, "reason": "missing value"})
                continue
            observed_value = _finite(observed_value, f"{family}.{metric}.{field}.{quantile}")
            limit_value = _finite(limit_value, f"limit.{family}.{metric}.{field}.{quantile}")
            if observed_value > limit_value:
                result["status"] = "failed"
                result["failures"].append(
                    {
                        "family": family,
                        "metric": metric,
                        "dimension": field,
                        "quantile": quantile,
                        "observed": observed_value,
                        "limit": limit_value,
                        "reason": "tolerance exceeded",
                    }
                )
    return result


def _cell_failures(report: dict[str, Any], *, kinds: tuple[str, ...]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    scoped_cells = [
        cell
        for cell in report.get("cells", [])
        if cell.get("split") == "sealed_test" and cell.get("kind") in kinds
    ]
    seen_kinds = {cell.get("kind") for cell in scoped_cells}
    for kind in kinds:
        if kind not in seen_kinds:
            failures.append({"kind": kind, "reason": "sealed report is missing a required kind"})
    cell_ids = [cell.get("cell_id") for cell in scoped_cells]
    if len(cell_ids) != len(set(cell_ids)):
        failures.append({"reason": "sealed report contains duplicate cell ids"})
    for cell in report.get("cells", []):
        if cell.get("split") != "sealed_test" or cell.get("kind") not in kinds:
            continue
        cell_id = cell.get("cell_id")
        if (cell.get("target") or {}).get("status") != "succeeded":
            failures.append({"cell_id": cell_id, "reason": "target result did not succeed"})
        if (cell.get("zero_cost") or {}).get("status") != "succeeded":
            failures.append({"cell_id": cell_id, "reason": "zero-cost result did not succeed"})
        if (cell.get("sim") or {}).get("status") != "succeeded":
            failures.append({"cell_id": cell_id, "reason": "sim result did not succeed"})
        pairing = cell.get("pairing") or {}
        if pairing.get("target_zero_cost") is not True or pairing.get("target_sim") is not True:
            failures.append({"cell_id": cell_id, "reason": "request/token pairing failed"})
    for failure in report.get("failures", []):
        failures.append({"source": "report", **failure})
    return failures


def _queueing_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Require complete sealed concurrency>1 TTFT pairing when gated."""
    cells = [
        cell
        for cell in (report.get("queueing_ttft") or {}).get("cells", [])
        if cell.get("split") == "sealed_test"
        and cell.get("kind") == "steady_decode"
        and isinstance(cell.get("concurrency"), int)
        and cell.get("concurrency") > 1
    ]
    if not cells:
        return [{"reason": "sealed report has no concurrency>1 queueing TTFT cells"}]
    failures: list[dict[str, Any]] = []
    for cell in cells:
        if cell.get("status") != "measured":
            failures.append({"cell_id": cell.get("cell_id"), "reason": "sealed queueing TTFT cell is not measured"})
        if (cell.get("pairing") or {}).get("complete") is not True:
            failures.append({"cell_id": cell.get("cell_id"), "reason": "sealed queueing TTFT pairing is incomplete"})
    aggregate = ((report.get("queueing_ttft") or {}).get("aggregates") or {}).get("sealed_test")
    if not isinstance(aggregate, dict) or not isinstance(aggregate.get("errors"), dict):
        failures.append({"reason": "sealed report is missing queueing TTFT aggregate"})
    return failures


def signoff(
    profile_path: Path,
    tolerance_path: Path,
    validation_report_path: Path,
    sealed_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    profile, profile_digest = _load_json(profile_path)
    selection, selection_digest = _load_json(profile_path.parent / "selection.json")
    tolerance, tolerance_digest = _load_json(tolerance_path)
    validation_report, validation_digest = _load_json(validation_report_path)
    sealed_report, sealed_digest = _load_json(sealed_report_path)
    signoff_kinds = list(SIGNOFF_KINDS[:2])
    if profile.get("predictor", {}).get("model", {}).get("mixed") is not None:
        signoff_kinds.append("mixed")
    signoff_kinds = tuple(signoff_kinds)
    failures: list[dict[str, Any]] = []
    if tolerance.get("status") != "frozen":
        failures.append({"reason": "tolerance artifact is not frozen"})
    if tolerance.get("profile", {}).get("sha256") != profile_digest:
        failures.append({"reason": "tolerance profile digest does not match profile bytes"})
    if tolerance.get("profile", {}).get("selection_sha256") != selection_digest:
        failures.append({"reason": "tolerance selection digest does not match selection bytes"})
    if tolerance.get("validation_report", {}).get("sha256") != validation_digest:
        failures.append({"reason": "tolerance validation report digest does not match bytes"})
    if tolerance.get("validation_campaign_id") != validation_report.get("campaign_id"):
        failures.append({"reason": "tolerance validation campaign does not match validation report"})
    validation_campaign_id = validation_report.get("campaign_id")
    sealed_campaign_id = sealed_report.get("campaign_id")
    if not _valid_campaign_id(validation_campaign_id):
        failures.append({"reason": "validation campaign id is invalid"})
    if not _valid_campaign_id(sealed_campaign_id):
        failures.append({"reason": "sealed campaign id is invalid"})
    if sealed_report.get("status") != "succeeded":
        failures.append({"reason": "sealed replay report is not succeeded"})
    if sealed_campaign_id == validation_campaign_id:
        failures.append({"reason": "sealed signoff campaign must be independent of validation campaign"})
    provenance = sealed_report.get("provenance") or {}
    if provenance.get("independent_campaign") is not True:
        failures.append({"reason": "sealed replay was not marked independent_campaign"})
    target_identity = tolerance.get("target_identity")
    if target_identity != provenance.get("target_identity"):
        failures.append({"reason": "sealed target identity differs from frozen tolerance target"})
    profile_calibration = profile.get("calibration") or {}
    if provenance.get("profile_sha256") != profile_digest:
        failures.append({"reason": "sealed replay profile digest does not match profile bytes"})
    if provenance.get("selection_sha256") != selection_digest:
        failures.append({"reason": "sealed replay selection digest does not match selection bytes"})
    if selection.get("status") != "selected":
        failures.append({"reason": "profile selection is characterization-only"})
    try:
        coverage = require_publication_coverage(
            sealed_report,
            required_splits=("sealed_test",),
            required_kinds=signoff_kinds,
        )
    except Exception as exc:
        coverage = sealed_report.get("coverage_audit", {}).get("sealed_test")
        failures.append({"reason": str(exc)})
    comparisons: dict[str, Any] = {}
    by_family = (sealed_report.get("aggregates") or {}).get("by_split_family", {}).get("sealed_test", {})
    for family, family_artifact in (tolerance.get("families") or {}).items():
        kind = family_artifact.get("kind")
        if family == QUEUEING_FAMILY:
            observed_family = {
                "queueing_ttft": (((sealed_report.get("queueing_ttft") or {}).get("aggregates") or {}).get("sealed_test") or {}).get("errors", {})
            }
        else:
            observed_family = by_family.get(kind, {})
        for metric, metric_artifact in (family_artifact.get("metrics") or {}).items():
            error_channel = metric_artifact.get("basis", "target_vs_sim")
            observed_errors = observed_family.get(error_channel) or {}
            observed = observed_errors.get(metric) or {}
            comparison = _check_limit(
                observed,
                metric_artifact.get("limits") or {},
                family=family,
                metric=metric,
            )
            comparison["basis"] = error_channel
            comparisons[f"{family}:{metric}"] = comparison
            failures.extend(comparison["failures"])
    failures.extend(_cell_failures(sealed_report, kinds=signoff_kinds))
    if QUEUEING_FAMILY in (tolerance.get("families") or {}):
        failures.extend(_queueing_failures(sealed_report))
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not failures else "failed",
        "profile": {"path": str(profile_path.resolve()), "sha256": profile_digest, "selection_sha256": selection_digest},
        "tolerance": {"path": str(tolerance_path.resolve()), "sha256": tolerance_digest},
        "validation": {"path": str(validation_report_path.resolve()), "sha256": validation_digest, "campaign_id": validation_report.get("campaign_id")},
        "sealed": {"path": str(sealed_report_path.resolve()), "sha256": sealed_digest, "campaign_id": sealed_report.get("campaign_id")},
        "target_identity": target_identity,
        "coverage": coverage,
        "comparisons": comparisons,
        "failures": failures,
        "scope": {
            "included_kinds": list(signoff_kinds),
            "excluded_kinds": ["cached_prefill"],
            "profile_calibration_manifest": profile_calibration.get("manifest"),
        },
        "digest_chain": {
            "profile_sha256": profile_digest,
            "selection_sha256": selection_digest,
            "tolerance_sha256": tolerance_digest,
            "validation_report_sha256": validation_digest,
            "sealed_report_sha256": sealed_digest,
        },
    }
    rendered = json.dumps(artifact, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_text(encoding="utf-8") != rendered:
        raise SignoffError(f"refusing to overwrite a different signoff artifact: {output_path}")
    if not output_path.exists():
        output_path.write_text(rendered, encoding="utf-8")
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--tolerance", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--sealed-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        artifact = signoff(
            args.profile,
            args.tolerance,
            args.validation_report,
            args.sealed_report,
            args.output,
        )
    except (OSError, SignoffError) as exc:
        print(f"signoff error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": artifact["status"], "sealed_campaign_id": artifact["sealed"]["campaign_id"]}, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
