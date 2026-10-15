#!/usr/bin/env python3
"""Functional profile replay prediction tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from copy import deepcopy


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from replay import ReplayError, _identity_matches, _predict_cell, _queueing_ttft_record, _sim_manifest  # noqa: E402


def _profile(*, mixed: dict | None = None) -> dict:
    return {
        "predictor": {
            "model": {
                "prefill": {"input_tokens": [64, 128], "duration_us": [10_000, 20_000]},
                "decode": {
                    "decode_reqs": [1, 2],
                    "sum_decode_ctx_tokens": [64, 128],
                    "duration_us": [3_000, 5_000, 7_000, 9_000],
                },
                "mixed": mixed,
            }
        }
    }


def _cell(*, kind: str = "mixed", input_tokens: int = 64, output_tokens: int = 3) -> dict:
    value = {
        "kind": kind,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "concurrency": 1,
    }
    if kind == "mixed":
        value.update(
            {
                "prefill_input_tokens": input_tokens,
                "sum_decode_ctx_tokens": input_tokens,
                "mixed_shape": {
                    "foreground": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                    "background": {"requests": 1, "context_tokens": input_tokens, "output_tokens": output_tokens},
                },
            }
        )
    return value


class ReplayPredictionTests(unittest.TestCase):
    def test_queueing_ttft_reports_request_level_errors_after_pairing(self) -> None:
        cell = {
            "cell_id": "decode-cell",
            "name": "decode-cell",
            "kind": "steady_decode",
            "split": "held_out_mixed",
            "concurrency": 2,
            "repeats": 1,
        }
        target = {
            "status": "succeeded",
            "measurements": [
                {"phase": "measured", "repeat": 0, "index": 0, "request_id": "target-0", "ok": True, "ttft_ms": 100.0},
                {"phase": "measured", "repeat": 0, "index": 1, "request_id": "target-1", "ok": True, "ttft_ms": 200.0},
            ],
        }
        sim = {
            "status": "succeeded",
            "measurements": [
                {"phase": "measured", "repeat": 0, "index": 1, "request_id": "sim-1", "ok": True, "ttft_ms": 180.0},
                {"phase": "measured", "repeat": 0, "index": 0, "request_id": "sim-0", "ok": True, "ttft_ms": 110.0},
            ],
        }
        record = _queueing_ttft_record(cell, cell["cell_id"], target, sim, "in_domain", None)
        self.assertEqual(record["status"], "measured")
        self.assertEqual(record["pairing"]["matched_count"], 2)
        self.assertEqual(record["errors"]["count"], 2)
        self.assertEqual(record["errors"]["absolute_ms"]["p50"], 15.0)

    def test_queueing_ttft_pairing_failure_has_no_error(self) -> None:
        cell = {
            "cell_id": "decode-cell",
            "name": "decode-cell",
            "kind": "steady_decode",
            "split": "sealed_test",
            "concurrency": 2,
            "repeats": 1,
        }
        result = {
            "status": "succeeded",
            "measurements": [
                {"phase": "measured", "repeat": 0, "index": 0, "request_id": "request-0", "ok": True, "ttft_ms": 100.0}
            ],
        }
        record = _queueing_ttft_record(cell, cell["cell_id"], result, result, "in_domain", None)
        self.assertEqual(record["status"], "pairing_failed")
        self.assertIsNone(record["errors"])
        self.assertEqual(record["pairing"]["missing_target"], ["decode-cell:measured:0:1"])

    def test_sim_manifest_uses_profile_target_model_identity(self) -> None:
        campaign = {
            "target": {"model": "target-model"},
            "variants": {"replay": {"identity": {"model": "sim-model"}}},
        }
        manifest = _sim_manifest(campaign, {"model_config_sha256": "c" * 64}, "http://sim", "target-model")
        self.assertEqual(manifest["variants"]["sim"]["identity"]["model"], "target-model")

    def test_profile_identity_mismatch_fails_closed(self) -> None:
        target = {
            "endpoint": "http://127.0.0.1:18080",
            "model": "test-model",
            "engine": "vllm",
            "engine_version": "0.30.0",
            "model_revision": "revision",
            "model_config_sha256": "a" * 64,
        }
        campaign = {
            "target": target,
            "runtime": {
                "backend": "vllm",
                "parallelism": "1",
                "quantization": "none",
                "scheduler": {
                    "policy": "vllm_v1",
                    "max_num_seqs": 2,
                    "max_num_batched_tokens": 16,
                    "max_model_len": 32,
                    "prefill": {"mode": "chunked", "max_chunk_tokens": 8},
                },
                "cache": {"policy": "off"},
            },
            "environment": {"cuda_visible_devices": "0", "driver": "550", "cuda": "12.4"},
            "client": {
                "name": "bench",
                "version": "1",
                "seed": 7,
                "server_flags": ["--model", "test-model"],
            },
            "workload": {
                "cells": [
                    {
                        "name": "cold",
                        "kind": "cold_prefill",
                        "input_tokens": 8,
                        "output_tokens": 2,
                        "concurrency": 1,
                        "arrival_pattern": "closed_loop",
                        "warmup_repeats": 1,
                        "repeats": 2,
                    }
                ]
            },
            "splits": {"train": ["cold"], "threshold_validation": [], "held_out_mixed": [], "sealed_test": []},
        }
        calibration = {
            "target_engine": "vllm",
            "engine_version": "0.30.0",
            "model_id": "test-model",
            "model_revision": "revision",
            "model_config_sha256": "a" * 64,
            "gpu": "0",
            "backend": "vllm",
            "driver": "550",
            "cuda": "12.4",
            "parallelism": "1",
            "quantization": "none",
            "scheduler": campaign["runtime"]["scheduler"],
            "cache": {"enabled": False, "policy": "off", "block_size": 0, "alignment": 0, "hit_semantics": "disabled"},
            "client": {"name": "bench", "version": "1", "seed": 7, "sampling": "temperature=0,top_p=1"},
            "server_flags": ["--model", "test-model"],
            "workload": {
                "cells": [
                    {
                        "id": "cold",
                        "family": "cold_prefill",
                        "input_tokens": 8,
                        "output_tokens": 2,
                        "concurrency": 1,
                        "arrival_pattern": "closed_loop",
                        "attribution": "black_box_effective",
                    }
                ],
                "warmup_repeats": 1,
                "repeats": 2,
            },
            "split": {"train": ["cold"], "threshold_validation": [], "held_out_mixed": [], "sealed_test": []},
        }
        selection = {"target": target}
        _identity_matches(campaign, calibration, selection)
        for field, value in (
            ("backend", "other"),
            ("cache", {"enabled": True}),
            ("scheduler", {"prefill": {"mode": "whole"}}),
            ("workload", {"cells": []}),
        ):
            mismatched = deepcopy(calibration)
            mismatched[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ReplayError):
                    _identity_matches(campaign, mismatched, selection)

    def test_mixed_without_predictor_fails_closed(self) -> None:
        status, reason, fitted = _predict_cell(_profile(), _cell())
        self.assertEqual(status, "unsupported")
        self.assertEqual(reason, "mixed predictor is not present")
        self.assertEqual(fitted, {})

    def test_mixed_base_does_not_depend_on_request_output_length(self) -> None:
        profile = _profile(mixed={"intercept_us": 0})
        first_status, first_reason, first = _predict_cell(profile, _cell(output_tokens=2))
        second_status, second_reason, second = _predict_cell(profile, _cell(output_tokens=3))
        self.assertEqual((first_status, first_reason), ("in_domain", None))
        self.assertEqual((second_status, second_reason), ("in_domain", None))
        self.assertEqual(first["ttft_ms"], second["ttft_ms"])
        self.assertGreater(second["e2e_ms"], first["e2e_ms"])


if __name__ == "__main__":
    unittest.main()
