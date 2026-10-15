#!/usr/bin/env python3
"""Tolerance-freeze and independent-signoff tests."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from signoff import _queueing_failures, signoff  # noqa: E402
from tolerance import freeze_tolerance  # noqa: E402
from tolerance import _queueing_errors  # noqa: E402
from publish import publish_profile  # noqa: E402
from replay import _validate_publication  # noqa: E402


def _metric_error(value: float) -> dict:
    return {
        "count": 1,
        "absolute_ms": {"p50": value, "p95": value, "p99": value},
        "relative": {"p50": value / 100.0, "p95": value / 100.0, "p99": value / 100.0},
    }


def _cell(cell_id: str, split: str, kind: str, error: float) -> dict:
    summary = {
        "percentiles_ms": {
            "ttft_ms": {"p50": 100.0},
            "stream_chunk_tpot_ms": {"p50": 10.0},
            "e2e_ms": {"p50": 120.0},
        }
    }
    dispersion = {
        metric: {"count": 2, "median": value, "mad": 0.5, "min": value - 0.5, "max": value + 0.5}
        for metric, value in (("ttft_ms", 100.0), ("stream_chunk_tpot_ms", 10.0), ("e2e_ms", 120.0))
    }
    errors = {metric: _metric_error(error) for metric in ("ttft_ms", "stream_chunk_tpot_ms", "e2e_ms")}
    return {
        "cell_id": cell_id,
        "kind": kind,
        "split": split,
        "coverage": {"status": "in_domain"},
        "target": {"status": "succeeded", "summary": summary, "dispersion": dispersion},
        "zero_cost": {"status": "succeeded", "summary": summary, "dispersion": dispersion},
        "sim": {"status": "succeeded", "summary": summary, "failures": []},
        "pairing": {"target_zero_cost": True, "target_sim": True},
        "errors": {"effective_vs_fitted": errors, "target_vs_sim": errors},
    }


def _report(
    campaign_id: str,
    split: str,
    error: float,
    *,
    profile_sha256: str = "p" * 64,
    selection_sha256: str = "s" * 64,
    profile_campaign_id: str = "q" * 64,
    calibration_manifest_sha256: str = "m" * 64,
    include_mixed: bool = True,
) -> dict:
    cells = [_cell(f"{split}-cold", split, "cold_prefill", error), _cell(f"{split}-decode", split, "steady_decode", error)]
    if include_mixed:
        cells.append(_cell(f"{split}-mixed", split, "mixed", error))
    by_family = {
        "cold_prefill": {"target_vs_sim": {metric: _metric_error(error) for metric in ("ttft_ms", "stream_chunk_tpot_ms", "e2e_ms")}},
        "steady_decode": {"target_vs_sim": {metric: _metric_error(error) for metric in ("ttft_ms", "stream_chunk_tpot_ms", "e2e_ms")}},
    }
    if include_mixed:
        by_family["mixed"] = {
            "effective_vs_fitted": {metric: _metric_error(error) for metric in ("ttft_ms", "stream_chunk_tpot_ms", "e2e_ms")},
            "target_vs_sim": {metric: _metric_error(error) for metric in ("ttft_ms", "stream_chunk_tpot_ms", "e2e_ms")},
        }
    required_kinds = ["cold_prefill", "steady_decode"]
    if include_mixed:
        required_kinds.append("mixed")
    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "status": "succeeded",
        "provenance": {
            "profile_sha256": profile_sha256,
            "selection_sha256": selection_sha256,
            "calibration_manifest_sha256": calibration_manifest_sha256,
            "profile_campaign_id": profile_campaign_id,
            "independent_campaign": split == "sealed_test",
            "target_identity": {"model": "test-model", "engine": "vllm"},
        },
        "coverage_audit": {
            split: {
                "status": "passed",
                "required_splits": [split],
                "required_kinds": required_kinds,
                "counts": {"cells": len(cells), "in_domain": len(cells), "out_of_domain": 0, "unsupported": 0, "sim_failed": 0, "pairing_failed": 0},
                "missing_splits": [],
                "failures": [],
            }
        },
        "cells": cells,
        "aggregates": {"by_split_family": {split: by_family}},
        "failures": [],
    }


class FidelityTests(unittest.TestCase):
    def test_queueing_gate_requires_complete_sealed_concurrency_pairing(self) -> None:
        report = _report("b" * 64, "sealed_test", 1.0)
        report["queueing_ttft"] = {
            "cells": [
                {
                    "cell_id": "decode-c8",
                    "kind": "steady_decode",
                    "split": "sealed_test",
                    "concurrency": 8,
                    "status": "pairing_failed",
                    "pairing": {"complete": False},
                }
            ],
            "aggregates": {},
        }
        failures = _queueing_failures(report)
        self.assertTrue(any(item["reason"] == "sealed queueing TTFT cell is not measured" for item in failures))
        self.assertTrue(any(item["reason"] == "sealed queueing TTFT pairing is incomplete" for item in failures))

    def test_queueing_validation_requires_concurrency_pairing_and_aggregate(self) -> None:
        report = _report("a" * 64, "threshold_validation", 1.0)
        report["queueing_ttft"] = {
            "cells": [
                {
                    "cell_id": "decode-c8",
                    "kind": "steady_decode",
                    "split": "threshold_validation",
                    "concurrency": 8,
                    "status": "measured",
                    "pairing": {"complete": True},
                }
            ],
            "aggregates": {
                "threshold_validation": {
                    "errors": _metric_error(1.0),
                }
            },
        }
        errors = _queueing_errors(report, "threshold_validation")
        self.assertEqual(errors["count"], 1)

    def test_freeze_and_signoff_require_an_independent_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            selection_path = root / "selection.json"
            validation_path = root / "validation.json"
            sealed_path = root / "sealed.json"
            tolerance_path = root / "tolerance.json"
            signoff_path = root / "signoff.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "calibration": {"manifest": "calibration.manifest.json", "sha256": "m" * 64},
                        "predictor": {"model": {"mixed": {"intercept_us": 0}}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            selection_path.write_text(
                    json.dumps({"status": "selected", "campaign_id": "q" * 64, "target": {"model": "test-model", "engine": "vllm"}})
                + "\n",
                encoding="utf-8",
            )
            profile_digest = hashlib.sha256(profile_path.read_bytes()).hexdigest()
            selection_digest = hashlib.sha256(selection_path.read_bytes()).hexdigest()
            validation_path.write_text(
                json.dumps(
                    _report(
                        "a" * 64,
                        "threshold_validation",
                        1.0,
                        profile_sha256=profile_digest,
                        selection_sha256=selection_digest,
                        profile_campaign_id="q" * 64,
                    ),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            artifact = freeze_tolerance(validation_path, profile_path, tolerance_path)
            self.assertEqual(artifact["status"], "frozen")
            sealed = _report(
                "b" * 64,
                "sealed_test",
                1.0,
                profile_sha256=artifact["profile"]["sha256"],
                selection_sha256=artifact["profile"]["selection_sha256"],
                profile_campaign_id="q" * 64,
            )
            sealed["provenance"]["profile_sha256"] = artifact["profile"]["sha256"]
            sealed["provenance"]["selection_sha256"] = artifact["profile"]["selection_sha256"]
            rejected_sealed_path = root / "rejected-sealed.json"
            sealed["status"] = "partial"
            sealed_path.write_text(json.dumps(sealed, sort_keys=True) + "\n", encoding="utf-8")
            rejected = signoff(profile_path, tolerance_path, validation_path, sealed_path, rejected_sealed_path)
            self.assertEqual(rejected["status"], "failed")
            self.assertIn("sealed replay report is not succeeded", [item["reason"] for item in rejected["failures"]])
            sealed["status"] = "succeeded"
            missing_kind_path = root / "missing-kind-sealed.json"
            sealed["cells"] = [cell for cell in sealed["cells"] if cell["kind"] != "mixed"]
            sealed_path.write_text(json.dumps(sealed, sort_keys=True) + "\n", encoding="utf-8")
            missing_kind = signoff(profile_path, tolerance_path, validation_path, sealed_path, missing_kind_path)
            self.assertEqual(missing_kind["status"], "failed")
            self.assertTrue(any(item.get("reason") == "sealed report is missing a required kind" for item in missing_kind["failures"]))
            sealed = _report(
                "b" * 64,
                "sealed_test",
                1.0,
                profile_sha256=artifact["profile"]["sha256"],
                selection_sha256=artifact["profile"]["selection_sha256"],
                profile_campaign_id="q" * 64,
            )
            sealed["provenance"]["profile_sha256"] = artifact["profile"]["sha256"]
            sealed["provenance"]["selection_sha256"] = artifact["profile"]["selection_sha256"]
            sealed_path.write_text(json.dumps(sealed, sort_keys=True) + "\n", encoding="utf-8")
            result = signoff(profile_path, tolerance_path, validation_path, sealed_path, signoff_path)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["comparisons"]["mixed:ttft_ms"]["basis"], "effective_vs_fitted")
            self.assertTrue(signoff_path.exists())
            publication = publish_profile(profile_path, tolerance_path, signoff_path)
            self.assertEqual(publication["status"], "passed")
            _validate_publication(
                profile_path,
                artifact["profile"]["sha256"],
                root / "publication.json",
                {},
                {"target": {"model": "test-model", "engine": "vllm"}},
            )

if __name__ == "__main__":
    unittest.main()
