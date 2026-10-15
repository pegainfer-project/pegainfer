#!/usr/bin/env python3
"""Write independent cached-prefix bundles into the paired artifact store.

This adapter only joins the vllm-bench detailed result and recorder evidence,
then delegates persistence to ``ArtifactStore``.  It does not import or call
the legacy request collector, benchmark, or streaming modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from artifacts import ArtifactError, ArtifactStore
from cached_prefix_collect import CachedPrefixError, _read_json, convert_prefix_repetition


VARIANTS = ("target", "zero_cost", "replay")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _invalid_result(cell: dict[str, Any], variant: str, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "collector": "vllm-bench-prefix-repetition",
        "variant": variant,
        "cell": dict(cell),
        "status": "invalid",
        "measurements": [],
        "failures": [{"kind": type(error).__name__, "error": str(error)[:500]}],
    }


def record_prefix_variant(
    store: ArtifactStore,
    cell: dict[str, Any],
    variant: str,
    *,
    result_path: Path,
    evidence_path: Path,
    collector_argv: list[str],
    probe: dict[str, Any],
    require_hit: bool | None = None,
) -> dict[str, Any]:
    """Convert and persist one variant, sealing it only after validation."""

    if variant not in VARIANTS:
        raise CachedPrefixError(f"unsupported variant: {variant}")
    if require_hit is None:
        require_hit = variant == "target"
    store.begin_variant_cell(cell, variant)
    source_probe = {
        **probe,
        "inputs": {
            "vllm_bench_result": {"name": result_path.name, "sha256": _file_sha256(result_path)},
            "recorder_evidence": {"name": evidence_path.name, "sha256": _file_sha256(evidence_path)},
        },
    }
    try:
        result = convert_prefix_repetition(
            _read_json(result_path),
            _read_json(evidence_path),
            cell=cell,
            variant=variant,
            require_hit=require_hit,
        )
    except (CachedPrefixError, OSError, ValueError) as exc:
        result = _invalid_result(cell, variant, exc)
    entry = store.record_variant_cell(
        cell,
        variant,
        result,
        collector_argv=collector_argv,
        probe=source_probe,
    )
    if result["status"] == "succeeded":
        store.seal_variant_cell(cell["cell_id"], variant)
        entry = store.index["variant_cells"][f"{cell['cell_id']}:{variant}"]
    return entry


def record_prefix_cell(
    manifest: dict[str, Any],
    cell: dict[str, Any],
    store: ArtifactStore,
    *,
    bundles: dict[str, tuple[Path, Path]],
    collector_argv: list[str],
    probe: dict[str, Any],
) -> dict[str, Any]:
    """Persist all three paired variants from detailed/evidence file pairs."""

    if manifest.get("schema_version") != 3:
        raise CachedPrefixError("cached-prefix artifact writing requires paired manifest schema_version=3")
    if cell.get("kind") != "cached_prefill":
        raise CachedPrefixError("cached-prefix artifact writing requires a cached_prefill cell")
    if set(bundles) != set(VARIANTS):
        raise CachedPrefixError("bundles must contain target, zero_cost, and replay")
    store.attach_manifest(manifest)
    entries: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        result_path, evidence_path = bundles[variant]
        entries[variant] = record_prefix_variant(
            store,
            cell,
            variant,
            result_path=result_path,
            evidence_path=evidence_path,
            collector_argv=collector_argv,
            probe=probe,
        )
    sealed = all(entry.get("status") == "sealed" for entry in entries.values())
    return {
        "campaign_id": manifest["campaign_id"],
        "cell_id": cell["cell_id"],
        "variants": entries,
        "status": "sealed" if sealed else "invalid",
    }


def _mapping(values: list[str], field: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        variant, separator, path = value.partition("=")
        if separator != "=" or variant not in VARIANTS or not path or variant in parsed:
            raise ValueError(f"{field} must contain one VARIANT=PATH entry for each variant")
        parsed[variant] = Path(path)
    if set(parsed) != set(VARIANTS):
        raise ValueError(f"{field} must contain target, zero_cost, and replay")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cell", required=True, help="workload cell name")
    parser.add_argument("--result", action="append", required=True, help="VARIANT=path to vllm-bench detailed JSON")
    parser.add_argument("--evidence", action="append", required=True, help="VARIANT=path to recorder evidence JSON")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        manifest = _read_json(args.manifest)
        cells = [item for item in manifest.get("workload", {}).get("cells", []) if item.get("name") == args.cell]
        if len(cells) != 1:
            raise CachedPrefixError(f"workload cell is not unique: {args.cell}")
        result_paths = _mapping(args.result, "--result")
        evidence_paths = _mapping(args.evidence, "--evidence")
        store = ArtifactStore(args.artifact_root, manifest["campaign_id"])
        outcome = record_prefix_cell(
            manifest,
            cells[0],
            store,
            bundles={variant: (result_paths[variant], evidence_paths[variant]) for variant in VARIANTS},
            collector_argv=sys.argv,
            probe={"kind": "external_cached_prefix_recorder"},
        )
    except (OSError, KeyError, TypeError, ValueError, ArtifactError, CachedPrefixError) as exc:
        print(f"cached-prefix artifact error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outcome, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if outcome["status"] == "sealed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
