#!/usr/bin/env python3
"""Mixed workload construction and overlap-proof tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mixed_orchestrator import (  # noqa: E402
    MixedError,
    build_mixed_workload,
    mark_overlap_metadata,
    merge_sidecars,
    validate_overlap,
)
from fit import fit_normalized  # noqa: E402
from report import build_report  # noqa: E402
from test_fit import _document  # noqa: E402


class MixedOrchestratorTests(unittest.TestCase):
    def test_no_overlap_is_unsupported(self) -> None:
        evidence = {
            "requests": [
                {"role": "background_decode", "clock_domain": "unix", "started_at_unix_s": 0.0, "finished_at_unix_s": 1.0},
                {"role": "foreground_prefill", "clock_domain": "unix", "started_at_unix_s": 1.1, "finished_at_unix_s": 2.0},
            ]
        }
        with self.assertRaisesRegex(MixedError, "no background/foreground interval overlap"):
            validate_overlap(evidence)

    def test_valid_overlap_can_mark_mixed_cell(self) -> None:
        evidence = {
            "requests": [
                {"role": "background_decode", "clock_domain": "unix", "started_at_unix_s": 0.0, "finished_at_unix_s": 2.0},
                {"role": "foreground_prefill", "clock_domain": "unix", "started_at_unix_s": 1.0, "finished_at_unix_s": 1.5},
            ]
        }
        overlap = validate_overlap(evidence)
        marked = mark_overlap_metadata({"kind": "mixed", "metadata": {}}, overlap)
        self.assertEqual(marked["metadata"]["overlap_status"], "validated")

    def test_held_out_mixed_overlap_is_not_used_for_correction(self) -> None:
        normalized = _document()
        normalized["cells"].append(
            {
                "cell_id": "held-out-mixed",
                "name": "held-out-mixed",
                "kind": "mixed",
                "split": "held_out_mixed",
                "status": "measured",
                "metadata": {"requires_overlap": True, "overlap_status": "validated"},
                "observations": [
                    {
                        "cell_id": "held-out-mixed",
                        "kind": "mixed",
                        "split": "held_out_mixed",
                        "input_tokens": 64,
                        "output_tokens": 4,
                        "concurrency": 1,
                        "repeat": repeat,
                        "index": 0,
                        "effective": {"e2e_ms": 20.0 + repeat},
                        "attribution": {"e2e_ms": "black_box_effective"},
                    }
                    for repeat in (0, 1)
                ],
            }
        )
        report = build_report(normalized, fit_normalized(normalized))
        self.assertEqual(report["residual_candidates"][0]["status"], "unsupported")
        self.assertNotIn("residuals_ms", report["residual_candidates"][0])


if __name__ == "__main__":
    unittest.main()
