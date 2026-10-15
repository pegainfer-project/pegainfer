#!/usr/bin/env python3
"""Strict normalization tests for sealed paired artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from artifacts import ArtifactStore  # noqa: E402
from manifest import build_manifest  # noqa: E402
from normalize import normalize_campaign  # noqa: E402


DIGEST = "a" * 64


def _spec(cell_specs: list[dict[str, Any]]) -> dict[str, Any]:
    target = {
        "endpoint": "http://127.0.0.1:18080",
        "model": "target-model",
        "engine": "target-engine",
        "engine_version": "1",
        "model_revision": "target-revision",
        "model_config_sha256": DIGEST,
    }
    variants = {
        name: {
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
        for name, port in (("zero_cost", 18081), ("replay", 18082))
    }
    return {
        "target": target,
        "runtime": {"scheduler": {"max_model_len": 512}},
        "client": {
            "name": "normalize-test",
            "version": "1",
            "revision": "test-revision",
            "seed": 7,
            "argv": ["normalize-test"],
            "server_flags": ["--target"],
        },
        "workload": {"cells": cell_specs},
        "variants": variants,
        "provenance": {
            "collector_revision": "collector-revision",
            "collection_command": "python paired.py",
            "artifact_uri": "file:///tmp/normalize-test",
            "source_digest": "source-digest",
        },
    }


def _measurement(cell: dict[str, Any], variant: str, *, repeat: int, index: int, phase: str = "measured") -> dict[str, Any]:
    target_e2e = 20.0 if variant == "target" else 18.0
    if variant == "replay":
        target_e2e = 19.0
    return {
        "request_id": f"{cell['cell_id']}-{phase}-{repeat}-{index}",
        "pairing_key": f"{cell['cell_id']}:measured:{repeat}:{index}" if phase == "measured" else f"{cell['cell_id']}:warmup:{repeat}:{index}",
        "variant": variant,
        "phase": phase,
        "repeat": repeat,
        "index": index,
        "ok": True,
        "prompt_tokens": cell["input_tokens"],
        "completion_tokens": cell["output_tokens"],
        "total_tokens": cell["input_tokens"] + cell["output_tokens"],
        "ttft_ms": target_e2e / 2.0,
        "tpot_ms": 2.0,
        "e2e_ms": target_e2e,
        "stream_chunk_tpot_ms": 2.0,
        "token_timing_valid": True,
        "timing_semantics": "token_interval" if variant == "target" else "stream_chunk_interval",
        "cache_hit_evidence": None,
    }


def _store(root: Path, manifest: dict[str, Any], cells: list[dict[str, Any]], *, mutate: dict[str, Any] | None = None) -> None:
    store = ArtifactStore(root, manifest["campaign_id"])
    store.attach_manifest(manifest)
    for cell in cells:
        for variant in ("target", "zero_cost", "replay"):
            measurements = [
                _measurement(cell, variant, repeat=0, index=index, phase="warmup")
                for index in range(cell["concurrency"])
            ]
            measurements.extend(
                _measurement(cell, variant, repeat=repeat, index=index)
                for repeat in range(cell["repeats"])
                for index in range(cell["concurrency"])
            )
            result: dict[str, Any] = {
                "cell_id": cell["cell_id"],
                "name": cell["name"],
                "variant": variant,
                "status": "succeeded",
                "measurements": measurements,
            }
            if mutate and mutate.get("variant") in (variant, "all") and mutate.get("cell_id") == cell["cell_id"]:
                if "measurement_index" in mutate:
                    selected = result["measurements"][mutate["measurement_index"]]
                    selected.update(mutate.get("updates", {}))
                drop = mutate.get("drop")
                if isinstance(drop, dict):
                    result["measurements"] = [
                        item
                        for item in result["measurements"]
                        if not (
                            item.get("phase") == "measured"
                            and item.get("repeat") == drop.get("repeat")
                            and item.get("index") == drop.get("index")
                        )
                    ]
            store.begin_variant_cell(cell, variant)
            entry = store.record_variant_cell(cell, variant, result, collector_argv=["test"], probe={"ok": True})
            store.seal_variant_cell(cell["cell_id"], variant)


class NormalizeTests(unittest.TestCase):
    def _cell(self, name: str, *, split: str = "train", kind: str = "cold_prefill") -> dict[str, Any]:
        return {
            "name": name,
            "kind": kind,
            "split": split,
            "input_tokens": 8,
            "output_tokens": 2,
            "concurrency": 1,
            "warmup_repeats": 1,
            "repeats": 2,
            "ignore_eos": True,
        }

    def test_digest_mismatch_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_manifest(_spec([self._cell("cold")]), created_at="2026-09-23T00:00:00+00:00")
            root = Path(directory)
            _store(root, manifest, manifest["workload"]["cells"])
            entry = json.loads((root / "artifact_index.json").read_text())
            path = root / entry["variant_cells"][f"{manifest['workload']['cells'][0]['cell_id']}:target"]["result_path"]
            path.write_text(path.read_text() + "\n")
            document = normalize_campaign(root)
        self.assertEqual(document["status"], "invalid")
        self.assertIn("digest mismatch", document["failures"][0]["reason"])

    def test_request_id_difference_does_not_break_logical_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_manifest(_spec([self._cell("cold")]), created_at="2026-09-23T00:00:00+00:00")
            root = Path(directory)
            cell = manifest["workload"]["cells"][0]
            _store(root, manifest, [cell], mutate={"variant": "replay", "cell_id": cell["cell_id"], "measurement_index": 1, "updates": {"request_id": "different-request"}})
            document = normalize_campaign(root)
        self.assertEqual(document["status"], "succeeded")
        cell_document = document["cells"][0]
        self.assertEqual(cell_document["pairing"]["legacy"], False)
        self.assertEqual(len(cell_document["pairing"]["request_id_mismatches"]), 1)

    def test_shared_missing_request_is_invalid_against_manifest_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_manifest(_spec([self._cell("cold")]), created_at="2026-09-23T00:00:00+00:00")
            root = Path(directory)
            cell = manifest["workload"]["cells"][0]
            _store(
                root,
                manifest,
                [cell],
                mutate={
                    "variant": "all",
                    "cell_id": cell["cell_id"],
                    "drop": {"repeat": 0, "index": 0},
                },
            )
            document = normalize_campaign(root)
        self.assertEqual(document["status"], "invalid")
        self.assertIn("manifest request plan", document["failures"][0]["reason"])

    def test_cross_campaign_raw_record_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_manifest(_spec([self._cell("cold")]), created_at="2026-09-23T00:00:00+00:00")
            root = Path(directory)
            _store(root, manifest, manifest["workload"]["cells"])
            index_path = root / "artifact_index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            cell_id = manifest["workload"]["cells"][0]["cell_id"]
            entry = index["variant_cells"][f"{cell_id}:target"]
            result_path = root / entry["result_path"]
            record = json.loads(result_path.read_text(encoding="utf-8"))
            record["campaign_id"] = "b" * 64
            payload = (json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
            result_path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            entry["result_sha256"] = digest
            entry["size"] = len(payload)
            index["files"][entry["result_path"]] = {"sha256": digest, "size": len(payload)}
            index_path.write_text(json.dumps(index, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            document = normalize_campaign(root)
        self.assertEqual(document["status"], "invalid")
        self.assertIn("invalid record envelope", document["failures"][0]["reason"])

    def test_cross_repeat_attribution_instability_is_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_manifest(_spec([self._cell("cold")]), created_at="2026-09-23T00:00:00+00:00")
            root = Path(directory)
            cell = manifest["workload"]["cells"][0]
            _store(
                root,
                manifest,
                [cell],
                mutate={
                    "variant": "target",
                    "cell_id": cell["cell_id"],
                    "measurement_index": 1,
                    "updates": {"e2e_ms": 100.0},
                },
            )
            document = normalize_campaign(root)
        self.assertEqual(document["status"], "succeeded")
        observations = document["observations"]
        self.assertTrue(all(item["attribution"]["e2e_ms"] == "attribution_uncertain" for item in observations))
        self.assertTrue(all(item["effective"]["e2e_ms"] is None for item in observations))


if __name__ == "__main__":
    unittest.main()
