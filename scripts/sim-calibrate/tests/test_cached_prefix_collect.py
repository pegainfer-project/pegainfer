#!/usr/bin/env python3
"""Tests for the independent vllm-bench cached-prefix converter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from cached_prefix_collect import CachedPrefixError, convert_prefix_repetition  # noqa: E402


def _cell() -> dict:
    return {
        "name": "cached-512-128",
        "kind": "cached_prefill",
        "split": "train",
        "prefix_tokens": 512,
        "new_tokens": 128,
        "input_tokens": 640,
        "output_tokens": 4,
        "concurrency": 1,
        "ignore_eos": True,
        "shared_prefix": {"identity": "prefix-a", "cached_tokens": 512},
    }


def _result(count: int = 2) -> dict:
    return {
        "backend": "vllm",
        "completed": count,
        "failed": 0,
        "input_lens": [640] * count,
        "output_lens": [4] * count,
        "ttfts": [0.010 + 0.001 * index for index in range(count)],
        "latencies": [0.016 + 0.001 * index for index in range(count)],
        "itls": [[0.002, 0.002, 0.002] for _ in range(count)],
        "errors": [None] * count,
    }


def _evidence(cell: dict, count: int = 2, *, hit: bool = True) -> dict:
    requests = []
    for index in range(count):
        requests.append(
            {
                "ordinal": index,
                "request_id": f"req-{index}",
                "phase": "measured",
                "status": 200,
                "input_tokens": 640,
                "output_tokens": 4,
                "finish_reason": "length",
                "usage": {
                    "prompt_tokens": 640,
                    "completion_tokens": 4,
                    "total_tokens": 644,
                },
                "cache_hit_evidence": {
                    "cached_tokens": 512 if hit else 0,
                    "cache_hit": hit,
                    "shared_prefix_id": "prefix-a",
                    "shared_prefix_hash": "hash-prefix-a",
                    "expected_cached_tokens": 512,
                    "duration_includes_uncached_tail": True,
                },
            }
        )
    import hashlib
    import json

    digest = hashlib.sha256(
        json.dumps({"cell": cell, "completed": count}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "evidence_source": "test-recorder",
        "ordering": "serial_arrival",
        "workload_digest": digest,
        "requests": requests,
    }


class CachedPrefixCollectTests(unittest.TestCase):
    def test_conversion_requires_positive_request_level_hit_evidence(self) -> None:
        cell = _cell()
        result = _result()
        evidence = _evidence(cell, hit=False)
        with self.assertRaisesRegex(CachedPrefixError, "cached_tokens must be positive"):
            convert_prefix_repetition(result, evidence, cell=cell, variant="target")

    def test_conversion_rejects_partial_hit_for_exact_prefix_cell(self) -> None:
        cell = _cell()
        evidence = _evidence(cell)
        evidence["requests"][0]["cache_hit_evidence"]["cached_tokens"] = 256
        with self.assertRaisesRegex(CachedPrefixError, "must cover prefix_tokens"):
            convert_prefix_repetition(_result(), evidence, cell=cell, variant="target")

    def test_conversion_rejects_request_identity_or_shape_mismatch(self) -> None:
        cell = _cell()
        evidence = _evidence(cell)
        evidence["requests"][1]["request_id"] = evidence["requests"][0]["request_id"]
        with self.assertRaisesRegex(CachedPrefixError, "duplicate request_id"):
            convert_prefix_repetition(_result(), evidence, cell=cell, variant="target")

        evidence = _evidence(cell)
        result = _result()
        result["input_lens"][1] = 641
        with self.assertRaisesRegex(CachedPrefixError, "input length mismatch"):
            convert_prefix_repetition(result, evidence, cell=cell, variant="target")


if __name__ == "__main__":
    unittest.main()
