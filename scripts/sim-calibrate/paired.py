#!/usr/bin/env python3
"""Cross-role collection assessment retained for C3 integration.

C2 owns sending, readiness, checkpoints and execution-artifact verification.
This module retains the old collector's final key/success/seal assessment as
a pure function. It does not read old artifacts or send requests. C3 still
needs to bind measurement v3, the immutable plan and actual workload digests.
"""

from __future__ import annotations

from typing import Any

from pairing import canonical_pair_key, pairing_key_text


PAIRED_VARIANTS = ("target", "zero_cost", "sim")


def _request_keys(result: dict[str, Any]) -> list[str]:
    cell_id = result.get("cell_id")
    keys: list[str] = []
    for item in result.get("measurements", []):
        if isinstance(cell_id, str) and isinstance(item, dict):
            try:
                keys.append(pairing_key_text(canonical_pair_key(item, cell_id)))
                continue
            except ValueError:
                pass
        keys.append(
            f"{item.get('phase')}:{item.get('repeat')}:{item.get('index')}:{item.get('request_id')}"
        )
    return sorted(keys)


def summarize_paired_cell(
    manifest: dict[str, Any],
    cell: dict[str, Any],
    variant_results: dict[str, dict[str, Any]],
    entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assess already-collected roles without taking over their execution.

    The result/entry arguments are domain inputs, not measurement-v3 envelopes.
    This retains the previous assessment; it is not the C3 content validator.
    """
    results = {variant: variant_results[variant] for variant in PAIRED_VARIANTS}
    measured = [result for result in results.values() if result.get("status") == "succeeded"]
    request_keys = {variant: _request_keys(result) for variant, result in results.items()}
    paired = len(measured) == len(PAIRED_VARIANTS) and len(set(tuple(keys) for keys in request_keys.values())) == 1
    return {
        "campaign_id": manifest["campaign_id"],
        "cell_id": cell["cell_id"],
        "variants": entries,
        "request_keys": request_keys,
        "paired": paired,
        "status": "sealed" if paired and all(entries[variant]["status"] == "sealed" for variant in PAIRED_VARIANTS) else "invalid",
    }
