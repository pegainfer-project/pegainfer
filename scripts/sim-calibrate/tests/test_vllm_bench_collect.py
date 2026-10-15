#!/usr/bin/env python3
"""Request identity and detailed-result conversion tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from vllm_bench_collect import VllmBenchCollectError, convert_vllm_bench  # noqa: E402
from vllm_bench_sidecar import build_sidecar  # noqa: E402


def _cell(concurrency: int = 2) -> dict:
    return {
        "cell_id": "decode-cell",
        "name": "decode-cell",
        "kind": "steady_decode",
        "input_tokens": 128,
        "output_tokens": 4,
        "concurrency": concurrency,
        "warmup_repeats": 0,
        "repeats": 2,
    }


def _result(sidecar: dict, *, order: list[int] | None = None) -> dict:
    rows = sidecar["requests"]
    order = order or list(range(len(rows)))
    return {
        "backend": "openai",
        "completed": len(order),
        "failed": 0,
        "request_ids": [rows[index]["request_id"] for index in order],
        "input_lens": [128] * len(order),
        "output_lens": [4] * len(order),
        "ttfts": [0.010 + index * 0.001 for index in range(len(order))],
        "latencies": [0.016 + index * 0.001 for index in range(len(order))],
        "itls": [[0.002, 0.002, 0.002] for _ in order],
        "errors": [None] * len(order),
    }


class VllmBenchCollectTests(unittest.TestCase):
    def test_sidecar_pairs_responses_after_concurrent_reordering(self) -> None:
        sidecar = build_sidecar(_cell(), ordering="stable_ordinal")
        result = _result(sidecar, order=[3, 2, 1, 0])
        converted = convert_vllm_bench(result, sidecar, cell=_cell(), variant="target")
        self.assertEqual(
            [item["request_id"] for item in converted["measurements"]],
            [sidecar["requests"][index]["request_id"] for index in (3, 2, 1, 0)],
        )
        self.assertEqual(converted["measurements"][0]["repeat"], 1)
        self.assertEqual(converted["measurements"][0]["index"], 1)

    def test_missing_identity_fails_closed_when_order_is_not_proven(self) -> None:
        sidecar = build_sidecar(_cell(), ordering="unknown")
        result = _result(sidecar)
        result.pop("request_ids")
        with self.assertRaisesRegex(VllmBenchCollectError, "does not prove stable"):
            convert_vllm_bench(result, sidecar, cell=_cell(), variant="target")

    def test_forwarded_body_digest_is_used_after_token_id_rewrite(self) -> None:
        cell = _cell(concurrency=1)
        cell["repeats"] = 1
        sidecar = build_sidecar(cell, body_values=[{"prompt": [1, 2, 3]}])
        result = _result(sidecar)
        result.pop("request_ids")
        result["body_sha256"] = [sidecar["requests"][0]["body_sha256"]]
        converted = convert_vllm_bench(result, sidecar, cell=cell, variant="target")
        self.assertEqual(converted["measurements"][0]["body_sha256"], sidecar["requests"][0]["body_sha256"])

    def test_chunked_itl_stream_is_retained_without_invalidating_request(self) -> None:
        cell = _cell(concurrency=1)
        cell["repeats"] = 1
        sidecar = build_sidecar(cell, ordering="stable_ordinal")
        result = _result(sidecar)
        result["itls"] = [[0.002]]
        converted = convert_vllm_bench(result, sidecar, cell=cell, variant="target")
        measurement = converted["measurements"][0]
        self.assertFalse(measurement["token_timing_valid"])
        self.assertEqual(measurement["itl_count"], 1)
        self.assertIsNotNone(measurement["tpot_ms"])

    def test_short_stream_completion_is_retained_with_unproven_length(self) -> None:
        cell = _cell(concurrency=1)
        cell["repeats"] = 1
        sidecar = build_sidecar(cell, ordering="stable_ordinal")
        result = _result(sidecar)
        result["output_lens"] = [3]
        result["itls"] = [[0.002, 0.002]]
        converted = convert_vllm_bench(result, sidecar, cell=cell, variant="zero_cost")
        measurement = converted["measurements"][0]
        self.assertFalse(measurement["completion_length_valid"])
        self.assertEqual(measurement["observed_completion_tokens"], 3)
        self.assertIsNone(measurement["tpot_ms"])
        self.assertIsNotNone(measurement["stream_chunk_tpot_ms"])


if __name__ == "__main__":
    unittest.main()
