#!/usr/bin/env python3
"""Integration tests for the independent cached-prefix artifact adapter."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from artifacts import ArtifactStore  # noqa: E402
from cached_prefix_artifacts import record_prefix_cell  # noqa: E402
from normalize import normalize_campaign  # noqa: E402
from manifest import build_manifest, sha256_json  # noqa: E402


DIGEST = "a" * 64


def _manifest() -> dict[str, Any]:
    target = {
        "endpoint": "http://127.0.0.1:18080",
        "model": "target-model",
        "engine": "vllm",
        "engine_version": "1",
        "model_revision": "target-revision",
        "model_config_sha256": DIGEST,
    }
    variants = {}
    for name, port in (("zero_cost", 18081), ("replay", 18082)):
        variants[name] = {
            "identity": {
                **target,
                "endpoint": f"http://127.0.0.1:{port}",
                "model": f"{name}-model",
                "engine": name,
            },
            "runtime": {"profile": name},
            "server_flags": [f"--{name}"],
            "profile_sha256": DIGEST,
        }
    return build_manifest(
        {
            "target": target,
            "runtime": {
                "scheduler": {"max_model_len": 2048},
                "cache": {"policy": "on", "block_size": 16, "alignment": 16},
            },
            "client": {
                "name": "vllm-bench-prefix",
                "version": "1",
                "revision": "test",
                "seed": 7,
                "argv": ["vllm-bench", "prefix_repetition"],
                "server_flags": ["--target"],
            },
            "workload": {
                "cells": [
                    {
                        "name": "cached-prefix",
                        "kind": "cached_prefill",
                        "split": "train",
                        "input_tokens": 640,
                        "output_tokens": 4,
                        "concurrency": 1,
                        "prefix_tokens": 512,
                        "shared_prefix": {"identity": "prefix-a", "suffix_mode": "request_index"},
                        "warmup_repeats": 0,
                        "repeats": 2,
                        "ignore_eos": True,
                    }
                ]
            },
            "variants": variants,
            "provenance": {
                "collector_revision": "independent-prefix-test",
                "collection_command": "vllm-bench prefix_repetition",
                "artifact_uri": "file:///tmp/prefix-test",
                "source_digest": "source-digest",
            },
        },
        created_at="2026-09-26T00:00:00+00:00",
    )


def _result() -> dict[str, Any]:
    return {
        "backend": "vllm",
        "completed": 2,
        "failed": 0,
        "input_lens": [640, 640],
        "output_lens": [4, 4],
        "ttfts": [0.010, 0.011],
        "latencies": [0.016, 0.017],
        "itls": [[0.002, 0.002, 0.002], [0.002, 0.002, 0.002]],
        "errors": [None, None],
    }


def _evidence(cell: dict[str, Any], *, hit: bool) -> dict[str, Any]:
    requests = []
    for index in range(2):
        request: dict[str, Any] = {
            "ordinal": index,
            "request_id": f"bench-{index}",
            "phase": "measured",
            "status": 200,
            "input_tokens": 640,
            "output_tokens": 4,
            "finish_reason": "length",
            "usage": {"prompt_tokens": 640, "completion_tokens": 4, "total_tokens": 644},
            "shared_prefix_hash": "hash-prefix-a",
        }
        if hit:
            request["cache_hit_evidence"] = {
                "cached_tokens": 512,
                "cache_hit": True,
                "shared_prefix_id": "prefix-a",
                "shared_prefix_hash": "hash-prefix-a",
                "expected_cached_tokens": 512,
                "duration_includes_uncached_tail": True,
            }
        requests.append(request)
    return {
        "schema_version": 1,
        "evidence_source": "test-recorder",
        "ordering": "serial_arrival",
        "workload_digest": sha256_json({"cell": cell, "completed": 2}),
        "requests": requests,
    }


class CachedPrefixArtifactTests(unittest.TestCase):
    def test_three_variant_bundle_normalizes_with_stable_pair_ids(self) -> None:
        manifest = _manifest()
        cell = manifest["workload"]["cells"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundles: dict[str, tuple[Path, Path]] = {}
            for variant in ("target", "zero_cost", "replay"):
                result_path = root / f"{variant}-result.json"
                evidence_path = root / f"{variant}-evidence.json"
                result_path.write_text(json.dumps(_result()), encoding="utf-8")
                evidence_path.write_text(
                    json.dumps(_evidence(cell, hit=variant == "target")), encoding="utf-8"
                )
                bundles[variant] = (result_path, evidence_path)
            outcome = record_prefix_cell(
                manifest,
                cell,
                ArtifactStore(root / "artifacts", manifest["campaign_id"]),
                bundles=bundles,
                collector_argv=["cached_prefix_artifacts.py"],
                probe={"kind": "test"},
            )
            document = normalize_campaign(root / "artifacts")
            self.assertEqual(outcome["status"], "sealed")
            self.assertEqual(document["status"], "succeeded")
            self.assertEqual(len(document["observations"]), 2)
            index = json.loads((root / "artifacts" / "artifact_index.json").read_text())
            target_entry = index["variant_cells"][f"{cell['cell_id']}:target"]
            target_record = json.loads(
                (root / "artifacts" / target_entry["result_path"]).read_text()
            )
            inputs = target_record["probe"]["inputs"]
            self.assertEqual(len(inputs["vllm_bench_result"]["sha256"]), 64)
            self.assertEqual(len(inputs["recorder_evidence"]["sha256"]), 64)
            self.assertEqual(
            {item["request_id"] for item in document["observations"]},
            {f"{cell['cell_id']}:measured:0:0", f"{cell['cell_id']}:measured:1:0"},
        )
        self.assertTrue(
            all(item["cache_hit_evidence"]["target"]["cached_tokens"] == 512 for item in document["observations"])
        )


if __name__ == "__main__":
    unittest.main()
