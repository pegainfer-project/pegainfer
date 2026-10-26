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
        for repeat, value in enumerate((1.0, 1.0)):
            samples.append({"repeat": repeat, "cell_id": "stable-10", "input_tokens": 10.0, "requests": 1.0, "kv_tokens": 10.0, "value": value})
        for repeat, value in enumerate((2.0, 2.0)):
            samples.append({"repeat": repeat, "cell_id": "stable-20", "input_tokens": 20.0, "requests": 1.0, "kv_tokens": 20.0, "value": value})
        for repeat, value in enumerate((2.5, 2.5)):
            samples.append({"repeat": repeat, "cell_id": "stable-batch", "input_tokens": 10.0, "requests": 2.0, "kv_tokens": 20.0, "value": value})
        samples.extend(
            [
                {"cell_id": "noisy", "input_tokens": 15.0, "requests": 2.0, "kv_tokens": 30.0, "value": 3.5},
                {"cell_id": "noisy", "input_tokens": 15.0, "requests": 2.0, "kv_tokens": 30.0, "value": 30.0},
            ]
        )
        # Numeric rows are expanded into complete fixed batches; two repeats per cell.
        for repeat, row in enumerate(samples[-2:]):
            row['repeat'] = repeat
        samples = [{**row, 'index': index} for row in samples for index in range(int(row['requests']))]
        candidate = _additive_fit(samples, metric="tpot_ms")
        coefficients = candidate["coefficients"]
        self.assertAlmostEqual(coefficients["d_kv_ms_per_token"], 0.1, delta=0.02)
        self.assertAlmostEqual(coefficients["d_requests_ms"], 0.5, delta=0.2)
        self.assertEqual(candidate["source"]["fit_weighting"]["center"], "repeat_mean")
        self.assertEqual(candidate["source"]["fit_weighting"]["method"], "inverse_mad_squared")
        self.assertEqual(candidate["source"]["observation_count"], 8)

    def test_cached_prefill_bilinear_surface_uses_prefix_and_new_axes(self) -> None:
        samples = [
            {"cell_id": f"cached-{prefix}-{new}", "prefix_tokens": prefix,
             "new_tokens": new, "value": 3.0 + 0.25 * prefix + 0.5 * new,
             "repeat": 0, "index": 0}
            for prefix in (16, 32) for new in (4, 8)
        ]
        document = {'cached_requirements': [
            {key: sample[key] for key in ('cell_id', 'prefix_tokens', 'new_tokens')} |
            {'matrix_name': 'cached', 'repeats': 1, 'concurrency': 1,
             'axes': {'prefix_tokens': [16, 32], 'new_tokens': [4, 8]}}
            for sample in samples]}
        candidate = _cached_surface_fit(samples, metric="ttft_ms", document=document)
        self.assertEqual(candidate["axes"], {"prefix_tokens": [16, 32], "new_tokens": [4, 8]})
        self.assertAlmostEqual(evaluate_cached_surface(candidate, 24, 6), 12.0)
        with self.assertRaisesRegex(FitError, "outside measured domain"):
            evaluate_cached_surface(candidate, 40, 6)
        with self.assertRaisesRegex(FitError, "grid is incomplete"):
            _cached_surface_fit(samples[:-1], metric="ttft_ms", document=document)
        # Losing a whole axis must not turn the remaining samples into a smaller domain.
        with self.assertRaisesRegex(FitError, "grid is incomplete"):
            _cached_surface_fit(samples[:2], metric="ttft_ms", document=document)
        with self.assertRaisesRegex(FitError, "compiled grid declaration"):
            _cached_surface_fit(samples, metric="ttft_ms", document={})
        with self.assertRaisesRegex(FitError, "duplicate"):
            _cached_surface_fit([*samples, samples[0]], metric="ttft_ms", document=document)
        # A negative request is retained when its paired-repeat median is nonnegative.
        for cell in document['cached_requirements']:
            cell['repeats'] = 2
        signed = [{**row, 'repeat': repeat, 'value': value}
                  for row in samples for repeat, value in enumerate((-10, 10))]
        candidate = _cached_surface_fit(signed, metric='ttft_ms', document=document)
        self.assertEqual(candidate['values_ms'], [[0, 0], [0, 0]])
        self.assertEqual(candidate['dispersion'][0][0]['min'], -10)
        with self.assertRaisesRegex(FitError, 'non-negative'):
            _cached_surface_fit([{**row, 'value': -1} for row in signed], metric='ttft_ms', document=document)

    def test_decode_surface_requires_complete_total_kv_grid(self) -> None:
        samples = [
            {"cell_id": f"decode-{count}-{kv}", "requests": count, "kv_tokens": kv,
             "repeat": repeat, "index": index, "input_tokens": kv // count, "value": 2.0 + 0.5 * count + 0.01 * kv + offset}
            for count in (1, 2) for kv in (16, 32) for repeat, offset in enumerate((0, 0, 9)) for index in range(count)
        ]
        candidate = _surface_fit(samples, metric="tpot_ms")
        self.assertEqual(candidate["axes"], {"requests": [1, 2], "kv_tokens": [16, 32]})
        self.assertEqual(candidate["source"]["observation_count"], 12)
        self.assertEqual(candidate["source"]["request_count"], 18)
        self.assertAlmostEqual(evaluate_surface(candidate, 2, 24), 6.24)
        with self.assertRaisesRegex(FitError, "outside measured domain"):
            evaluate_surface(candidate, 4, 24)
        with self.assertRaisesRegex(FitError, "grid is incomplete"):
            _surface_fit([row for row in samples if (row["requests"], row["kv_tokens"]) != (2, 32)], metric="tpot_ms")


if __name__ == "__main__":
    unittest.main()
