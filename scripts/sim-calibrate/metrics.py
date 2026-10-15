#!/usr/bin/env python3
"""Descriptive summaries for raw benchmark request measurements."""

from __future__ import annotations

import math
from typing import Any, Iterable


def percentile(values: Iterable[float | int], quantile: float) -> float | None:
    clean = sorted(
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    if not clean:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    fraction = position - lower
    return clean[lower] + (clean[upper] - clean[lower]) * fraction


def summarize_measurements(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in measurements if item.get("ok") is True]
    failed = [item for item in measurements if item.get("ok") is not True]

    def values(field: str) -> list[float]:
        return [float(item[field]) for item in successful if isinstance(item.get(field), (int, float))]

    itl = [
        float(value)
        for item in successful
        for value in item.get("itl_ms", [])
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        "requests": len(measurements),
        "successful": len(successful),
        "failed": len(failed),
        "timeouts": sum(1 for item in failed if item.get("timed_out") is True),
        "token_timing_valid": sum(1 for item in successful if item.get("token_timing_valid") is True),
        "token_timing_unknown": sum(1 for item in successful if item.get("token_timing_valid") is None),
        "percentiles_ms": {
            field: {
                label: percentile(values(field), quantile)
                for label, quantile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
            }
            for field in ("ttft_ms", "tpot_ms", "e2e_ms", "stream_chunk_tpot_ms")
        },
        "itl_ms": {
            label: percentile(itl, quantile)
            for label, quantile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
        },
        "input_tokens": sum(int(item.get("prompt_tokens", 0)) for item in successful),
        "output_tokens": sum(int(item.get("completion_tokens", 0)) for item in successful),
    }

