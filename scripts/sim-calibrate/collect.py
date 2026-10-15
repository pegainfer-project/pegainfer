#!/usr/bin/env python3
"""Run fixed synthetic workload cells against an OAI-compatible endpoint.

This module intentionally returns structured results instead of writing raw
artifacts.  The checkpoint/artifact boundary is implemented in
``artifacts.py`` so a failed request cannot be mistaken for a sealed sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from manifest import ManifestError, load_spec
from artifacts import ArtifactError, ArtifactStore
from probe import Endpoint, ProbeError, request, probe_endpoint


class SampleValidationError(ValueError):
    """Raised when a successful HTTP response is not comparable evidence."""


def _prompt_tokens(
    cell_id: str,
    request_index: int,
    count: int,
    *,
    prefix_tokens: int = 0,
    prefix_identity: str | None = None,
) -> list[int]:
    if prefix_tokens <= 0:
        seed = hashlib.sha256(f"{cell_id}:{request_index}".encode("utf-8")).digest()
        return [1000 + ((seed[index % len(seed)] + index * 17) % 30000) for index in range(count)]
    if prefix_identity is None or prefix_tokens >= count:
        raise ValueError("cached prompt requires a prefix identity and an uncached suffix")
    prefix_seed = hashlib.sha256(f"prefix:{prefix_identity}".encode("utf-8")).digest()
    suffix_seed = hashlib.sha256(f"suffix:{cell_id}:{request_index}".encode("utf-8")).digest()
    prefix = [1000 + ((prefix_seed[index % len(prefix_seed)] + index * 17) % 30000) for index in range(prefix_tokens)]
    suffix = [1000 + ((suffix_seed[index % len(suffix_seed)] + index * 31) % 30000) for index in range(count - prefix_tokens)]
    return prefix + suffix


def _shared_prefix_hash(prompt: list[int], prefix_tokens: int) -> str | None:
    if prefix_tokens <= 0:
        return None
    payload = json.dumps(prompt[:prefix_tokens], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _request_id(cell_id: str, phase: str, repeat: int, index: int) -> str:
    return hashlib.sha256(f"{cell_id}:{phase}:{repeat}:{index}".encode("utf-8")).hexdigest()[:24]


def _validate_response(
    payload: Any,
    *,
    prompt_tokens: int,
    expected_output_tokens: int,
    ignore_eos: bool,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SampleValidationError("completion response must be an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise SampleValidationError("completion response must contain choices[0]")
    finish_reason = choices[0].get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise SampleValidationError("choices[0].finish_reason is missing")
    if ignore_eos and finish_reason != "length":
        raise SampleValidationError(
            f"finish_reason must be length for ignore_eos=true, got {finish_reason!r}"
        )
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise SampleValidationError("completion response must contain usage")
    expected = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": expected_output_tokens,
        "total_tokens": prompt_tokens + expected_output_tokens,
    }
    for field, wanted in expected.items():
        actual = usage.get(field)
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise SampleValidationError(f"usage.{field} must be an integer")
        if actual != wanted:
            raise SampleValidationError(f"usage.{field} expected {wanted}, got {actual}")
    return {"finish_reason": finish_reason, "usage": expected}


def _one_request(
    endpoint: Endpoint,
    manifest: dict[str, Any],
    cell: dict[str, Any],
    *,
    phase: str,
    repeat: int,
    index: int,
    timeout: float,
) -> dict[str, Any]:
    shared_prefix = cell.get("shared_prefix") or {}
    prompt = _prompt_tokens(
        cell["cell_id"],
        index,
        cell["input_tokens"],
        prefix_tokens=cell.get("prefix_tokens", 0),
        prefix_identity=shared_prefix.get("identity"),
    )
    body = {
        "model": manifest["target"]["model"],
        "prompt": prompt,
        "max_tokens": cell["output_tokens"],
        "temperature": cell["sampling"].get("temperature", 0.0),
        "top_p": cell["sampling"].get("top_p", 1.0),
        "ignore_eos": cell["ignore_eos"],
        "seed": manifest["client"].get("seed"),
        "stream": False,
    }
    body = {key: value for key, value in body.items() if value is not None}
    started_wall = time.time()
    started = time.perf_counter()
    try:
        response = request(endpoint, "POST", "/v1/completions", body=body, timeout=timeout)
        payload = response.json()
        validation = _validate_response(
            payload,
            prompt_tokens=len(prompt),
            expected_output_tokens=cell["output_tokens"],
            ignore_eos=cell["ignore_eos"],
        )
        return {
            "request_id": _request_id(cell["cell_id"], phase, repeat, index),
            "pairing_key": f"{cell['cell_id']}:{phase}:{repeat}:{index}",
            "phase": phase,
            "repeat": repeat,
            "index": index,
            "ok": True,
            "status": response.status,
            "started_at_unix_s": started_wall,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "prompt_tokens": len(prompt),
            "completion_tokens": validation["usage"]["completion_tokens"],
            "total_tokens": validation["usage"]["total_tokens"],
            "finish_reason": validation["finish_reason"],
            "response_body_sha256": hashlib.sha256(response.body).hexdigest(),
        }
    except (ProbeError, SampleValidationError, json.JSONDecodeError) as exc:
        return {
            "request_id": _request_id(cell["cell_id"], phase, repeat, index),
            "pairing_key": f"{cell['cell_id']}:{phase}:{repeat}:{index}",
            "phase": phase,
            "repeat": repeat,
            "index": index,
            "ok": False,
            "status": getattr(exc, "status", None),
            "started_at_unix_s": started_wall,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "prompt_tokens": len(prompt),
            "expected_output_tokens": cell["output_tokens"],
            "error": str(exc)[:500],
        }


def run_cell(
    manifest: dict[str, Any],
    cell: dict[str, Any],
    *,
    timeout: float = 30.0,
    endpoint_override: str | None = None,
) -> dict[str, Any]:
    """Run warmup and measured repeats for one manifest cell."""

    endpoint_url = endpoint_override or manifest["target"]["endpoint"]
    endpoint = Endpoint.parse(endpoint_url)
    measurements: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    request_rate = cell.get("request_rate")

    for phase, count in (("warmup", cell["warmup_repeats"]), ("measured", cell["repeats"])):
        for repeat in range(count):
            started_batch = time.perf_counter()
            with ThreadPoolExecutor(max_workers=cell["concurrency"]) as executor:
                futures = []
                for index in range(cell["concurrency"]):
                    if request_rate and index:
                        time.sleep(1.0 / float(request_rate))
                    futures.append(
                        executor.submit(
                            _one_request,
                            endpoint,
                            manifest,
                            cell,
                            phase=phase,
                            repeat=repeat,
                            index=index,
                            timeout=timeout,
                        )
                    )
                batch = [future.result() for future in as_completed(futures)]
            batch.sort(key=lambda result: result["index"])
            measurements.extend(batch)
            failures.extend(result for result in batch if not result["ok"])
            if failures:
                return {
                    "cell_id": cell["cell_id"],
                    "name": cell["name"],
                    "status": "invalid",
                    "measurements": measurements,
                    "failures": failures,
                    "batch_elapsed_ms": (time.perf_counter() - started_batch) * 1000.0,
                }
    return {
        "cell_id": cell["cell_id"],
        "name": cell["name"],
        "status": "succeeded",
        "measurements": measurements,
        "failures": [],
    }


def run_campaign(
    manifest: dict[str, Any],
    *,
    cell_names: set[str] | None = None,
    timeout: float = 30.0,
    probe_attempts: int = 1,
    endpoint_override: str | None = None,
) -> dict[str, Any]:
    endpoint = endpoint_override or manifest["target"]["endpoint"]
    if endpoint_override is not None and endpoint.rstrip("/") != manifest["target"]["endpoint"].rstrip("/"):
        raise ManifestError("endpoint override must match manifest target.endpoint")
    probe = probe_endpoint(
        endpoint,
        expected_model=manifest["target"]["model"],
        timeout=min(timeout, 10.0),
        attempts=probe_attempts,
    )
    selected = [
        cell
        for cell in manifest["workload"]["cells"]
        if cell_names is None or cell["name"] in cell_names
    ]
    if not selected:
        raise ManifestError("no workload cells selected")
    results = [
        run_cell(manifest, cell, timeout=timeout, endpoint_override=endpoint_override)
        for cell in selected
    ]
    return {
        "campaign_id": manifest["campaign_id"],
        "probe": probe,
        "cells": results,
        "status": "succeeded" if all(item["status"] == "succeeded" for item in results) else "invalid",
    }


def collect_to_store(
    manifest: dict[str, Any],
    store: ArtifactStore,
    *,
    collector_argv: list[str],
    cell_names: set[str] | None = None,
    timeout: float = 30.0,
    probe_attempts: int = 1,
    endpoint_override: str | None = None,
) -> dict[str, Any]:
    """Probe once, checkpoint each cell, and seal only validated results."""

    store.attach_manifest(manifest)
    endpoint = endpoint_override or manifest["target"]["endpoint"]
    if endpoint_override is not None and endpoint.rstrip("/") != manifest["target"]["endpoint"].rstrip("/"):
        raise ManifestError("endpoint override must match manifest target.endpoint")
    probe = probe_endpoint(
        endpoint,
        expected_model=manifest["target"]["model"],
        timeout=min(timeout, 10.0),
        attempts=probe_attempts,
    )
    selected = [
        cell
        for cell in manifest["workload"]["cells"]
        if cell_names is None or cell["name"] in cell_names
    ]
    if not selected:
        raise ManifestError("no workload cells selected")
    results = []
    for cell in selected:
        if store.status(cell["cell_id"]) == "sealed":
            results.append({"cell_id": cell["cell_id"], "name": cell["name"], "status": "sealed", "skipped": True})
            continue
        store.begin_cell(cell)
        try:
            result = run_cell(manifest, cell, timeout=timeout, endpoint_override=endpoint_override)
        except Exception as exc:  # preserve an unexpected collector failure as an invalid cell
            result = {
                "cell_id": cell["cell_id"],
                "name": cell["name"],
                "status": "failed",
                "measurements": [],
                "failures": [{"error": str(exc)[:500], "kind": type(exc).__name__}],
            }
        entry = store.record_cell(
            cell,
            result,
            collector_argv=collector_argv,
            probe=probe,
        )
        if result["status"] == "succeeded":
            store.seal_cell(cell["cell_id"])
            entry = store.index["cells"][cell["cell_id"]]
        results.append(entry)
    return {
        "campaign_id": manifest["campaign_id"],
        "probe": probe,
        "cells": results,
        "status": "succeeded" if all(item["status"] == "sealed" for item in results) else "invalid",
        "artifact_index": str(store.index_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell", action="append", dest="cells", help="cell name; repeat to select multiple")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--probe-attempts", type=int, default=3)
    parser.add_argument("--endpoint", help="override the manifest endpoint for a smoke run")
    parser.add_argument("--artifact-root", type=Path, help="external directory for immutable raw artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with args.manifest.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if args.artifact_root is None:
            result = run_campaign(
                manifest,
                cell_names=set(args.cells) if args.cells else None,
                timeout=args.timeout,
                probe_attempts=args.probe_attempts,
                endpoint_override=args.endpoint,
            )
        else:
            store = ArtifactStore(args.artifact_root, manifest["campaign_id"])
            result = collect_to_store(
                manifest,
                store,
                collector_argv=sys.argv,
                cell_names=set(args.cells) if args.cells else None,
                timeout=args.timeout,
                probe_attempts=args.probe_attempts,
                endpoint_override=args.endpoint,
            )
    except (OSError, json.JSONDecodeError, ManifestError, ProbeError, ArtifactError) as exc:
        print(f"collection error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
