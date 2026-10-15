#!/usr/bin/env python3
"""Manifest validation smoke tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from manifest import ManifestError, build_manifest  # noqa: E402


def make_spec(endpoint: str) -> dict[str, Any]:
    return {
        "target": {
            "endpoint": endpoint,
            "model": "fake-model",
            "engine": "fake-engine",
            "engine_version": "fake-1",
            "model_revision": "model-rev-1",
            "model_config_sha256": "config-digest",
        },
        "runtime": {
            "gpu": "cpu",
            "backend": "fake",
            "parallelism": "1",
            "quantization": "none",
            "scheduler": {"max_num_seqs": 2, "max_num_batched_tokens": 32},
            "cache": {"policy": "off"},
        },
        "client": {
            "name": "sim-calibrate-test",
            "version": "1",
            "revision": "client-rev-1",
            "argv": ["sim-calibrate", "--model", "fake-model"],
            "server_flags": ["--max-num-seqs", "2"],
        },
        "workload": {
            "cells": [
                {
                    "name": "cold-prefill-small",
                    "kind": "cold_prefill",
                    "split": "train",
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "concurrency": 2,
                    "warmup_repeats": 1,
                    "repeats": 1,
                    "ignore_eos": True,
                }
            ]
        },
        "provenance": {
            "collector_revision": "collector-rev-1",
            "collection_command": "python collect.py --manifest manifest.json",
            "artifact_uri": "file:///tmp/sim-calibration",
            "source_digest": "source-digest-1",
        },
    }


class CalibrationSmokeTests(unittest.TestCase):
    def test_json_matrices_expand_into_concrete_cells(self) -> None:
        spec = make_spec("http://127.0.0.1:1")
        spec["workload"] = {
            "matrices": {
                "cold_prefill": {
                    "input_tokens": {"values": [256, 1024]},
                    "output_tokens": 8,
                    "repeats": 2,
                    "warmup_repeats": 1,
                },
                "steady_decode": {
                    "concurrency": {"start": 1, "stop": 2, "step": 1},
                    "context_tokens": {"values": [128]},
                    "surface_requests": {"values": [1]},
                    "surface_kv_tokens": {"values": [256]},
                    "output_tokens": 8,
                    "repeats": 2,
                    "warmup_repeats": 1,
                },
                "mixed": {
                    "cells": [
                        {
                            "name": "mixed-small",
                            "kind": "mixed",
                            "split": "held_out_mixed",
                            "input_tokens": 512,
                            "output_tokens": 4,
                            "concurrency": 2,
                            "prefill_input_tokens": 512,
                            "sum_decode_ctx_tokens": 1024,
                            "mixed_shape": {
                                "foreground": {"input_tokens": 512, "output_tokens": 4},
                                "background": {"requests": 2, "context_tokens": 512, "output_tokens": 4},
                            },
                            "warmup_repeats": 1,
                            "repeats": 2,
                            "metadata": {"workload_file": "mixed-small.jsonl"},
                        }
                    ]
                },
            }
        }
        manifest = build_manifest(spec, created_at="2026-09-29T00:00:00+00:00")
        self.assertEqual(manifest["workload"]["cell_count"], 6)
        self.assertEqual(manifest["workload"]["matrix_spec"]["steady_decode"]["concurrency"], [1, 2])
        self.assertEqual(manifest["workload"]["cells"][-1]["name"], "mixed-small")

    def test_paired_cached_prefill_requires_cache_enabled(self) -> None:
        spec = make_spec("http://127.0.0.1:1")
        spec["runtime"]["scheduler"]["max_model_len"] = 64
        spec["workload"] = {
            "cached_prefill": {
                "prefix_tokens": [16, 32],
                "new_tokens": [4, 8],
                "shared_prefix": {"identity": "fixed-prefix-v1"},
            }
        }
        digest = "a" * 64
        spec["target"]["model_config_sha256"] = digest
        spec["runtime"]["cache"] = {"policy": "off"}
        spec["client"]["seed"] = 7
        spec["variants"] = {
            name: {
                "identity": {
                    **spec["target"],
                    "endpoint": f"http://127.0.0.1:{port}",
                    "engine": "sim",
                },
                "runtime": {"backend": "sim"},
                "server_flags": ["sim", name],
                "profile_sha256": digest,
            }
            for name, port in (("zero_cost", 2), ("replay", 3))
        }
        spec["provenance"]["source_digest"] = "cached-test"
        with self.assertRaisesRegex(ManifestError, "runtime.cache.policy=on"):
            build_manifest(spec, created_at="2026-09-21T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
