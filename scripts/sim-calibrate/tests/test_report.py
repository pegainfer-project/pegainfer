#!/usr/bin/env python3
"""Composition and held-out residual report tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit import fit_normalized  # noqa: E402
from report import build_report  # noqa: E402
from test_fit import _cell, _document  # noqa: E402


def _mixed_observation(cell_id: str, repeat: int, value: float, *, input_tokens: int = 64) -> dict:
    return {
        "cell_id": cell_id,
        "kind": "mixed",
        "split": "threshold_validation",
        "input_tokens": input_tokens,
        "output_tokens": 4,
        "concurrency": 1,
        "prefill_input_tokens": input_tokens,
        "sum_decode_ctx_tokens": input_tokens,
        "mixed_shape": {
            "foreground": {"input_tokens": input_tokens, "output_tokens": 4},
            "background": {"requests": 1, "context_tokens": input_tokens, "output_tokens": 4},
        },
        "repeat": repeat,
        "index": 0,
        "request_id": f"{cell_id}-{repeat}",
        "effective": {"e2e_ms": value},
        "attribution": {"e2e_ms": "black_box_effective"},
        "timing_semantics": {"target": "chunk_count_matches_usage"},
        "token_timing_valid": {"target": True},
    }


def _mixed_cell(cell_id: str, split: str, observations: list[dict], *, input_tokens: int = 64) -> dict:
    for observation in observations:
        observation["split"] = split
    return {
        "cell_id": cell_id,
        "name": cell_id,
        "kind": "mixed",
        "split": split,
        "status": "measured",
        "metadata": {},
        "prefill_input_tokens": input_tokens,
        "sum_decode_ctx_tokens": input_tokens,
        "mixed_shape": {
            "foreground": {"input_tokens": input_tokens, "output_tokens": 4},
            "background": {"requests": 1, "context_tokens": input_tokens, "output_tokens": 4},
        },
        "observations": observations,
    }


class ReportTests(unittest.TestCase):
    def test_repeated_same_direction_validation_residual_forms_candidate(self) -> None:
        normalized = _document()
        normalized["cells"].append(
            _mixed_cell(
                "mixed-validation",
                "threshold_validation",
                [_mixed_observation("mixed-validation", 0, 20.0), _mixed_observation("mixed-validation", 1, 21.0)],
            )
        )
        fitted = fit_normalized(normalized)
        report = build_report(normalized, fitted)
        residual = report["residual_candidates"][0]
        self.assertEqual(residual["status"], "candidate")
        self.assertEqual(len(residual["residuals_ms"]), 1)
        self.assertEqual(residual["fit_split"], "threshold_validation")

    def test_ood_cell_has_reason_without_fabricated_prediction(self) -> None:
        normalized = _document()
        normalized["cells"].append(
            _cell(
                "cold-ood",
                "cold_prefill",
                "threshold_validation",
                [
                    {
                        **normalized["cells"][0]["observations"][0],
                        "cell_id": "cold-ood",
                        "split": "threshold_validation",
                        "input_tokens": 99999,
                        "effective": {"ttft_ms": 8.0},
                        "attribution": {"ttft_ms": "black_box_effective"},
                    }
                ],
            )
        )
        fitted = fit_normalized(normalized)
        report = build_report(normalized, fitted)
        cold_report = next(item for item in report["comparisons"] if item["name"] == "cold_prefill_piecewise_linear")
        record = cold_report["by_split"]["threshold_validation"]["by_cell"]["cold-ood"]
        self.assertEqual(record["status"], "out_of_domain")
        self.assertIsNone(record["predicted"]["p50"])
        self.assertTrue(record["ood"])

    def test_aggregate_errors_remain_consistent_with_cell_samples(self) -> None:
        normalized = _document()
        normalized["cells"].append(
            _cell(
                "cold-validation-2",
                "cold_prefill",
                "threshold_validation",
                [
                    {
                        **normalized["cells"][0]["observations"][0],
                        "cell_id": "cold-validation-2",
                        "split": "threshold_validation",
                        "input_tokens": 64,
                        "effective": {"ttft_ms": 6.0},
                        "attribution": {"ttft_ms": "black_box_effective"},
                    }
                ],
            )
        )
        fitted = fit_normalized(normalized)
        report = build_report(normalized, fitted)
        comparison = next(item for item in report["comparisons"] if item["name"] == "cold_prefill_piecewise_linear")
        split = comparison["by_split"]["threshold_validation"]
        self.assertEqual(split["sample_count"], sum(item["sample_count"] for item in split["by_cell"].values()))
        self.assertEqual(split["errors"]["count"], split["sample_count"])


if __name__ == "__main__":
    unittest.main()
