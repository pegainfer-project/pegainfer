#!/usr/bin/env python3
"""Candidate fit tests using deterministic normalized observations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from fit import (  # noqa: E402
    FitError,
    _additive_fit,
    evaluate_cached_surface,
    evaluate_piecewise,
    fit_normalized,
)


def _observation(
    cell_id: str,
    kind: str,
    split: str,
    input_tokens: int,
    concurrency: int,
    value: float,
    index: int = 0,
) -> dict:
    metric = "ttft_ms" if kind == "cold_prefill" else "tpot_ms"
    return {
        "cell_id": cell_id,
        "kind": kind,
        "split": split,
        "input_tokens": input_tokens,
        "output_tokens": 4,
        "concurrency": concurrency,
        "repeat": 0,
        "index": index,
        "request_id": f"{cell_id}-{index}",
        "effective": {metric: value},
        "attribution": {metric: "black_box_effective"},
        "timing_semantics": {"target": "chunk_count_matches_usage", "zero_cost": "stream_chunk"},
        "token_timing_valid": {"target": True, "zero_cost": True},
    }


def _cell(cell_id: str, kind: str, split: str, observations: list[dict]) -> dict:
    metric = "ttft_ms" if kind in ("cold_prefill", "cached_prefill") else "tpot_ms"
    return {
        "cell_id": cell_id,
        "name": cell_id,
        "kind": kind,
        "split": split,
        "status": "measured",
        "summaries": {metric: {"effective": {"mad": 0.25}}},
        "observations": observations,
    }


def _document() -> dict:
    cells: list[dict] = []
    cells.append(
        _cell("cold-64", "cold_prefill", "train", [_observation("cold-64", "cold_prefill", "train", 64, 1, 5.0)])
    )
    cells.append(
        _cell("cold-128", "cold_prefill", "train", [_observation("cold-128", "cold_prefill", "train", 128, 1, 20.0)])
    )
    cells.append(
        _cell("cold-256", "cold_prefill", "train", [_observation("cold-256", "cold_prefill", "train", 256, 1, 21.0)])
    )
    cells.append(
        _cell(
            "cold-validation",
            "cold_prefill",
            "threshold_validation",
            [_observation("cold-validation", "cold_prefill", "threshold_validation", 192, 1, 999.0)],
        )
    )
    # Use two input lengths for each request count so the surface has a complete grid.
    for requests, input_lengths in ((1, (32, 64)), (2, (16, 32))):
        for input_tokens in input_lengths:
            kv_tokens = input_tokens * requests
            value = 2.0 + 0.5 * (requests - 1) + 0.01 * kv_tokens
            cell_id = f"decode-{requests}-{input_tokens}"
            cells.append(
                _cell(
                    cell_id,
                    "steady_decode",
                    "train",
                    [_observation(cell_id, "steady_decode", "train", input_tokens, requests, value)],
                )
            )
    return {"schema_version": 1, "campaign_id": "campaign-fit", "status": "succeeded", "cells": cells}


def _cached_cell(cell_id: str, split: str, prefix_tokens: int, new_tokens: int, value: float) -> dict:
    observation = {
        "cell_id": cell_id,
        "kind": "cached_prefill",
        "split": split,
        "input_tokens": prefix_tokens + new_tokens,
        "prefix_tokens": prefix_tokens,
        "output_tokens": 4,
        "concurrency": 1,
        "repeat": 0,
        "index": 0,
        "request_id": f"{cell_id}-0",
        "effective": {"ttft_ms": value},
        "attribution": {"ttft_ms": "black_box_effective"},
        "timing_semantics": {"target": "chunk_count_matches_usage"},
        "token_timing_valid": {"target": True},
        "shared_prefix": {"identity": "prefix-v1", "cached_tokens": prefix_tokens},
        "cache_hit_evidence": {
            "target": {
                "cached_tokens": prefix_tokens,
                "shared_prefix_id": "prefix-v1",
                "shared_prefix_hash": f"hash-{prefix_tokens}",
                "cache_hit": True,
                "expected_cached_tokens": prefix_tokens,
                "duration_includes_uncached_tail": True,
            }
        },
    }
    return {
        "cell_id": cell_id,
        "name": cell_id,
        "kind": "cached_prefill",
        "split": split,
        "status": "measured",
        "summaries": {"ttft_ms": {"effective": {"mad": 0.1}}},
        "observations": [observation],
    }


class FitTests(unittest.TestCase):
    def setUp(self) -> None:
        result = fit_normalized(_document(), source_digest="normalized-digest")
        self.candidates = {candidate["name"]: candidate for candidate in result["candidates"]}

    def test_piecewise_retains_cliff_and_rejects_ood(self) -> None:
        candidate = self.candidates["cold_prefill_piecewise_linear"]
        self.assertEqual(candidate["status"], "fitted")
        self.assertEqual([knot["x"] for knot in candidate["knots"]], [64.0, 128.0, 256.0])
        self.assertEqual(evaluate_piecewise(candidate, 128), 20.0)
        self.assertAlmostEqual(evaluate_piecewise(candidate, 96), 12.5)
        with self.assertRaisesRegex(FitError, "outside measured domain"):
            evaluate_piecewise(candidate, 512)

    def test_decode_falls_back_to_explicit_stream_chunk_metric(self) -> None:
        document = _document()
        for cell in document["cells"]:
            if cell["kind"] != "steady_decode":
                continue
            for observation in cell["observations"]:
                value = observation["effective"].pop("tpot_ms")
                observation["effective"]["stream_chunk_tpot_ms"] = value
                observation["attribution"]["tpot_ms"] = "attribution_uncertain"
                observation["attribution"]["stream_chunk_tpot_ms"] = "black_box_effective"
        result = fit_normalized(document)
        additive = next(item for item in result["candidates"] if item["name"] == "decode_additive")
        self.assertEqual(additive["metric"], "stream_chunk_tpot_ms")
        self.assertEqual(additive["timing_semantics"], "black_box_effective_stream_chunk")

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
        document = _document()
        for prefix_tokens in (16, 32):
            for new_tokens in (4, 8):
                value = 3.0 + 0.25 * prefix_tokens + 0.5 * new_tokens
                document["cells"].append(
                    _cached_cell(
                        f"cached-{prefix_tokens}-{new_tokens}",
                        "train",
                        prefix_tokens,
                        new_tokens,
                        value,
                    )
                )
        document["source"] = {"cache_policy": {"policy": "on", "block_alignment": 16}}
        fitted = fit_normalized(document)
        candidate = next(item for item in fitted["candidates"] if item["name"] == "cached_prefill_surface")
        self.assertEqual(candidate["status"], "fitted")
        self.assertEqual(candidate["formula"], "prefix_cache(prefix_tokens, new_tokens)")
        self.assertEqual(candidate["axes"], {"prefix_tokens": [16.0, 32.0], "new_tokens": [4.0, 8.0]})
        self.assertAlmostEqual(evaluate_cached_surface(candidate, 24, 6), 12.0)
        with self.assertRaisesRegex(FitError, "outside measured domain"):
            evaluate_cached_surface(candidate, 40, 6)

    def test_decode_surface_accepts_complete_total_kv_grid(self) -> None:
        document = _document()
        document["cells"] = [cell for cell in document["cells"] if cell["kind"] != "steady_decode"]
        for requests in (1, 2):
            for kv_tokens in (16, 32):
                input_tokens = kv_tokens // requests
                cell_id = f"surface-{requests}-{kv_tokens}"
                observation = _observation(cell_id, "steady_decode", "train", input_tokens, requests, 2.0)
                observation["sum_decode_ctx_tokens"] = kv_tokens
                document["cells"].append(_cell(cell_id, "steady_decode", "train", [observation]))
        fitted = fit_normalized(document)
        candidate = next(item for item in fitted["candidates"] if item["name"] == "decode_2d_surface")
        self.assertEqual(candidate["status"], "fitted")
        self.assertEqual(candidate["axes"], {"requests": [1.0, 2.0], "kv_tokens": [16.0, 32.0]})


if __name__ == "__main__":
    unittest.main()
