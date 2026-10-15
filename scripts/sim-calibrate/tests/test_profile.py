#!/usr/bin/env python3
"""Focused profile identity materialization tests."""

from __future__ import annotations

import sys
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from manifest import canonical_workload_identity  # noqa: E402
from profile import ProfileError, _comparison_evidence, _validate_artifact_lineage  # noqa: E402


class ProfileIdentityTests(unittest.TestCase):
    def test_workload_identity_accepts_heterogeneous_measurement_repeats(self) -> None:
        campaign = {
            "workload": {
                "cells": [
                    {
                        "name": "train",
                        "kind": "cold_prefill",
                        "input_tokens": 128,
                        "output_tokens": 8,
                        "concurrency": 1,
                        "warmup_repeats": 1,
                        "repeats": 2,
                    },
                    {
                        "name": "heldout",
                        "kind": "steady_decode",
                        "input_tokens": 1024,
                        "output_tokens": 8,
                        "concurrency": 4,
                        "warmup_repeats": 1,
                        "repeats": 3,
                    },
                ]
            },
        }
        identity, _ = canonical_workload_identity(campaign)
        self.assertEqual(identity["warmup_repeats"], 1)
        self.assertEqual(identity["repeats"], 3)

    def test_selection_ignores_report_cells_outside_candidate_source(self) -> None:
        comparison = {
            "name": "candidate",
            "candidate_provenance": {"candidate": {"source_cells": ["fit-cell"]}},
            "by_split": {
                "train": {
                    "status": "measured",
                    "sample_count": 1,
                    "errors": {"relative": {"p95": 0.1}},
                    "by_cell": {
                        "fit-cell": {
                            "status": "measured",
                            "errors": {"relative": {"p95": 0.1}},
                        },
                        "out-of-domain-report-cell": {
                            "status": "uncertain",
                            "errors": {"relative": {"p95": None}},
                        },
                    },
                }
            },
        }
        evidence = _comparison_evidence(comparison)
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(evidence[0], "train")
        self.assertEqual(evidence[2]["cell_count"], 1)

    def test_parent_digest_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = {"campaign_id": "c" * 64}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(campaign, sort_keys=True) + "\n", encoding="utf-8")
            manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            normalized = {
                "campaign_id": campaign["campaign_id"],
                "status": "succeeded",
                "source": {"source_digests": {"manifest_sha256": manifest_digest, "artifact_index_sha256": "a" * 64}},
            }
            normalized_path = root / "normalized.json"
            normalized_path.write_text(json.dumps(normalized, sort_keys=True) + "\n", encoding="utf-8")
            normalized_digest = hashlib.sha256(normalized_path.read_bytes()).hexdigest()
            fit_path = root / "fit.json"
            fit_path.write_text(json.dumps({"campaign_id": campaign["campaign_id"], "status": "succeeded", "source": {"normalized_sha256": normalized_digest}}, sort_keys=True) + "\n", encoding="utf-8")
            fit_digest = hashlib.sha256(fit_path.read_bytes()).hexdigest()
            report_path = root / "report.json"
            report_path.write_text(json.dumps({
                "campaign_id": campaign["campaign_id"],
                "status": "succeeded",
                "provenance": {
                    "normalized_sha256": normalized_digest,
                    "fit_sha256": "0" * 64,
                    "manifest_sha256": manifest_digest,
                    "artifact_index_sha256": "a" * 64,
                },
            }, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ProfileError, "report fit digest does not match"):
                _validate_artifact_lineage(root, campaign, normalized_path, fit_path, report_path)


if __name__ == "__main__":
    unittest.main()
