#!/usr/bin/env python3
"""Coverage tests for the vllm-bench cold/decode matrices."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from vllm_bench_matrix import decode_matrix, decode_surface_axes  # noqa: E402
from manifest import _normalise_cell  # noqa: E402


class VllmBenchMatrixTests(unittest.TestCase):
    def test_surface_cells_form_complete_total_kv_grid(self) -> None:
        cells = decode_matrix(
            {
                "concurrency": {"start": 1, "stop": 4, "step": 1},
                "context_tokens": {"values": [128]},
                "surface_requests": {"values": [1, 2, 4]},
                "surface_kv_tokens": {"values": [256, 512]},
                "output_tokens": 8,
                "repeats": 2,
                "warmup_repeats": 1,
            }
        )
        requests, kv_tokens = decode_surface_axes(cells)
        self.assertEqual(requests, [1, 2, 4])
        self.assertEqual(kv_tokens, [256, 512])
        self.assertEqual(len([cell for cell in cells if cell["metadata"]["family"] == "surface"]), 6)

    def test_decode_total_kv_axis_survives_manifest_normalization(self) -> None:
        cell = decode_matrix(
            {
                "concurrency": {"values": [4]},
                "context_tokens": {"values": [128]},
                "surface_requests": {"values": [4]},
                "surface_kv_tokens": {"values": [512]},
                "output_tokens": 8,
                "repeats": 2,
                "warmup_repeats": 1,
            }
        )[0]
        normalized = _normalise_cell(cell, 0, {})
        self.assertEqual(normalized["sum_decode_ctx_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
