#!/usr/bin/env python3
"""Canonical request pairing tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from pairing import pair_metric_measurements  # noqa: E402


def _cell(*, concurrency: int = 2, repeats: int = 2) -> dict:
    return {"concurrency": concurrency, "repeats": repeats}


def _result(
    cell_id: str,
    *,
    values: dict[tuple[int, int], float],
    order: list[tuple[int, int]] | None = None,
    request_prefix: str = "request",
) -> dict:
    order = order or list(values)
    measurements = [
        {
            "pairing_key": f"{cell_id}:measured:{repeat}:{index}",
            "request_id": f"{request_prefix}-{repeat}-{index}",
            "phase": "measured",
            "repeat": repeat,
            "index": index,
            "ok": True,
            "ttft_ms": values[(repeat, index)],
        }
        for repeat, index in order
    ]
    return {"cell_id": cell_id, "status": "succeeded", "measurements": measurements}


class PairingTests(unittest.TestCase):
    def test_duplicate_key_fails_without_overwriting_value(self) -> None:
        cell_id = "decode-cell"
        result = _result(
            cell_id,
            values={(0, 0): 10.0, (0, 1): 20.0, (1, 0): 30.0, (1, 1): 40.0},
        )
        result["measurements"].append(dict(result["measurements"][0], ttft_ms=999.0))
        paired = pair_metric_measurements(
            result,
            _result(
                cell_id,
                values={(0, 0): 11.0, (0, 1): 21.0, (1, 0): 31.0, (1, 1): 41.0},
            ),
            cell=_cell(),
            cell_id=cell_id,
            metric="ttft_ms",
            positive=True,
        )
        self.assertFalse(paired["pairing"]["complete"])
        self.assertEqual(paired["pairing"]["duplicate_keys"]["target"], ["decode-cell:measured:0:0"])
        self.assertEqual(paired["pairs"], [])

    def test_invalid_ttft_is_not_dropped_from_the_denominator(self) -> None:
        cell_id = "decode-cell"
        target = _result(
            cell_id,
            values={(0, 0): 0.0, (0, 1): 20.0, (1, 0): 30.0, (1, 1): 40.0},
        )
        paired = pair_metric_measurements(
            target,
            _result(
                cell_id,
                values={(0, 0): 1.0, (0, 1): 21.0, (1, 0): 31.0, (1, 1): 41.0},
            ),
            cell=_cell(),
            cell_id=cell_id,
            metric="ttft_ms",
            positive=True,
        )
        self.assertFalse(paired["pairing"]["complete"])
        self.assertEqual(paired["pairing"]["invalid_metrics"]["target"], ["decode-cell:measured:0:0"])

    def test_explicit_pairing_key_mismatch_fails_closed(self) -> None:
        cell_id = "decode-cell"
        target = _result(
            cell_id,
            values={(0, 0): 10.0, (0, 1): 20.0, (1, 0): 30.0, (1, 1): 40.0},
        )
        target["measurements"][0]["pairing_key"] = "wrong-key"
        paired = pair_metric_measurements(
            target,
            _result(
                cell_id,
                values={(0, 0): 11.0, (0, 1): 21.0, (1, 0): 31.0, (1, 1): 41.0},
            ),
            cell=_cell(),
            cell_id=cell_id,
            metric="ttft_ms",
            positive=True,
        )
        self.assertFalse(paired["pairing"]["complete"])
        self.assertTrue(paired["pairing"]["invalid_records"]["target"])


if __name__ == "__main__":
    unittest.main()
