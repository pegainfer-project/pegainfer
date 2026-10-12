#!/usr/bin/env python3
"""Numeric cost-model tests independent of campaign and measurement schemas."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from fit import (  # noqa: E402
    FitError,
    _additive_fit,
    _cached_surface_fit,
    _piecewise_fit,
    _surface_fit,
    evaluate_cached_surface,
    evaluate_piecewise,
    evaluate_surface,
)


class FitTests(unittest.TestCase):
    def test_piecewise_retains_cliff_and_rejects_ood(self) -> None:
        samples = [
            {"cell_id": f"cold-{tokens}", "input_tokens": tokens, "value": value}
            for tokens, value in ((64, 5.0), (128, 20.0), (256, 21.0))
        ]
        candidate = _piecewise_fit(samples, metric="ttft_ms")
        self.assertEqual([knot["x"] for knot in candidate["knots"]], [64, 128, 256])
        self.assertEqual(evaluate_piecewise(candidate, 128), 20.0)
        self.assertAlmostEqual(evaluate_piecewise(candidate, 96), 12.5)
        with self.assertRaisesRegex(FitError, "outside measured domain"):
            evaluate_piecewise(candidate, 512)

    def test_decode_additive_uses_mad_weight_and_records_policy(self) -> None:
        samples = []
        for value in (1.0, 1.0):
            samples.append({"cell_id": "stable-10", "input_tokens": 10.0, "requests": 1.0, "kv_tokens": 10.0, "value": value})
        for value in (2.0, 2.0):
            samples.append({"cell_id": "stable-20", "input_tokens": 20.0, "requests": 1.0, "kv_tokens": 20.0, "value": value})
        for value in (2.5, 2.5):
            samples.append({"cell_id": "stable-batch", "input_tokens": 10.0, "requests": 2.0, "kv_tokens": 20.0, "value": value})
        samples.extend(
            [
                {"cell_id": "noisy", "input_tokens": 15.0, "requests": 2.0, "kv_tokens": 30.0, "value": 3.5},
                {"cell_id": "noisy", "input_tokens": 15.0, "requests": 2.0, "kv_tokens": 30.0, "value": 30.0},
            ]
        )
        candidate = _additive_fit(samples, metric="tpot_ms")
        coefficients = candidate["coefficients"]
        self.assertAlmostEqual(coefficients["d_kv_ms_per_token"], 0.1, delta=0.02)
        self.assertAlmostEqual(coefficients["d_requests_ms"], 0.5, delta=0.2)
        self.assertEqual(candidate["source"]["fit_weighting"]["center"], "repeat_median")
        self.assertEqual(candidate["source"]["fit_weighting"]["method"], "inverse_mad_squared")
        self.assertEqual(candidate["source"]["observation_count"], 8)

    def test_cached_prefill_bilinear_surface_uses_prefix_and_new_axes(self) -> None:
        samples = [
            {"cell_id": f"cached-{prefix}-{new}", "prefix_tokens": prefix,
             "new_tokens": new, "value": 3.0 + 0.25 * prefix + 0.5 * new}
            for prefix in (16, 32) for new in (4, 8)
        ]
        candidate = _cached_surface_fit(samples, metric="ttft_ms", document={})
        self.assertEqual(candidate["axes"], {"prefix_tokens": [16, 32], "new_tokens": [4, 8]})
        self.assertAlmostEqual(evaluate_cached_surface(candidate, 24, 6), 12.0)
        with self.assertRaisesRegex(FitError, "outside measured domain"):
            evaluate_cached_surface(candidate, 40, 6)
        with self.assertRaisesRegex(FitError, "grid is incomplete"):
            _cached_surface_fit(samples[:-1], metric="ttft_ms", document={})

    def test_decode_surface_requires_complete_total_kv_grid(self) -> None:
        samples = [
            {"cell_id": f"decode-{count}-{kv}", "requests": count, "kv_tokens": kv,
             "input_tokens": kv // count, "value": 2.0 + 0.5 * count + 0.01 * kv}
            for count in (1, 2) for kv in (16, 32)
        ]
        candidate = _surface_fit(samples, metric="tpot_ms")
        self.assertEqual(candidate["axes"], {"requests": [1, 2], "kv_tokens": [16, 32]})
        self.assertAlmostEqual(evaluate_surface(candidate, 2, 24), 3.24)
        with self.assertRaisesRegex(FitError, "outside measured domain"):
            evaluate_surface(candidate, 4, 24)
        with self.assertRaisesRegex(FitError, "grid is incomplete"):
            _surface_fit(samples[:-1], metric="tpot_ms")


if __name__ == "__main__":
    unittest.main()
