#!/usr/bin/env python3
"""Collect one fixed-config cell for target, zero-cost, and replay variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from artifacts import ArtifactError, ArtifactStore
from benchmark import run_stream_cell
from manifest import ManifestError
from pairing import canonical_pair_key, pairing_key_text
from probe import ProbeError, probe_endpoint


PAIRED_VARIANTS = ("target", "zero_cost", "replay")


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


def _stored_result(store: ArtifactStore, entry: dict[str, Any]) -> dict[str, Any]:
    path = store.root / entry["result_path"]
    try:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"failed to load sealed variant result {path}: {exc}") from exc
    result = record.get("result") if isinstance(record, dict) else None
    if not isinstance(result, dict):
        raise ArtifactError(f"sealed variant result has no result object: {path}")
    return result


def collect_paired_cell(
    manifest: dict[str, Any],
    cell: dict[str, Any],
    store: ArtifactStore,
    *,
    collector_argv: list[str],
    timeout: float = 30.0,
) -> dict[str, Any]:
    if manifest.get("schema_version") not in (2, 3):
        raise ManifestError("paired collection requires manifest schema_version=2 or 3")
    store.attach_manifest(manifest)
    variant_results: dict[str, dict[str, Any]] = {}
    entries: dict[str, dict[str, Any]] = {}
    for variant in PAIRED_VARIANTS:
        if store.variant_status(cell["cell_id"], variant) == "sealed":
            entries[variant] = store.index["variant_cells"][f"{cell['cell_id']}:{variant}"]
            variant_results[variant] = _stored_result(store, entries[variant])
            continue
        identity = manifest["variants"][variant]["identity"]
        probe = probe_endpoint(
            identity["endpoint"],
            expected_model=identity["model"],
            timeout=min(timeout, 10.0),
            attempts=2,
        )
        store.begin_variant_cell(cell, variant)
        try:
            result = run_stream_cell(manifest, cell, variant=variant, timeout=timeout)
        except Exception as exc:  # retain unexpected collector failure as a failed attempt
            result = {
                "cell_id": cell["cell_id"],
                "name": cell["name"],
                "variant": variant,
                "status": "failed",
                "measurements": [],
                "summary": {},
                "failures": [{"kind": type(exc).__name__, "error": str(exc)[:500]}],
            }
        variant_results[variant] = result
        entries[variant] = store.record_variant_cell(
            cell,
            variant,
            result,
            collector_argv=collector_argv,
            probe=probe,
        )
        if result["status"] == "succeeded":
            store.seal_variant_cell(cell["cell_id"], variant)
            entries[variant] = store.index["variant_cells"][f"{cell['cell_id']}:{variant}"]
    measured = [result for result in variant_results.values() if result.get("status") == "succeeded"]
    request_keys = {_variant: _request_keys(result) for _variant, result in variant_results.items()}
    paired = len(measured) == len(PAIRED_VARIANTS) and len(set(tuple(keys) for keys in request_keys.values())) == 1
    return {
        "campaign_id": manifest["campaign_id"],
        "cell_id": cell["cell_id"],
        "variants": entries,
        "request_keys": request_keys,
        "paired": paired,
        "status": "sealed" if paired and all(entry["status"] == "sealed" for entry in entries.values()) else "invalid",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cell", action="append", dest="cells")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with args.manifest.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        store = ArtifactStore(args.artifact_root, manifest["campaign_id"])
        selected = [
            cell
            for cell in manifest["workload"]["cells"]
            if args.cells is None or cell["name"] in set(args.cells)
        ]
        if not selected:
            raise ManifestError("no workload cells selected")
        results = [
            collect_paired_cell(
                manifest,
                cell,
                store,
                collector_argv=sys.argv,
                timeout=args.timeout,
            )
            for cell in selected
        ]
    except (OSError, json.JSONDecodeError, ManifestError, ProbeError, ArtifactError) as exc:
        print(f"paired collection error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(results, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if all(result["status"] == "sealed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
