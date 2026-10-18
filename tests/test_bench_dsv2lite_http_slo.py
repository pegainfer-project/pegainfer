#!/usr/bin/env python3
"""Regression tests for scripts/bench_dsv2lite_http_slo.py."""

from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "bench_dsv2lite_http_slo.py"
)
BENCHMARKING_DOC = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "models"
    / "deepseek-v2-lite"
    / "benchmarking.md"
)
SCRIPTS_DIR = SCRIPT_PATH.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location("bench_dsv2lite_http_slo", SCRIPT_PATH)
assert SPEC and SPEC.loader
bench_dsv2lite_http_slo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_dsv2lite_http_slo
SPEC.loader.exec_module(bench_dsv2lite_http_slo)
TEST_COMMIT = (
    bench_dsv2lite_http_slo.subprocess.check_output(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    )
    .strip()
    .lower()
)


def valid_summary(
    backend: str,
    profile: bench_dsv2lite_http_slo.SloProfile,
    *,
    commit: str = TEST_COMMIT,
) -> dict[str, object]:
    def metric() -> dict[str, object]:
        return {
            percentile: [1.0] * profile.repeats for percentile in ("p50", "p95", "p99")
        } | {
            f"{percentile}_summary": {
                "median": 1.0,
                "min": 1.0,
                "max": 1.0,
                "samples": profile.repeats,
            }
            for percentile in ("p50", "p95", "p99")
        }

    trace = {
        "traced_requests": profile.num_requests,
        "attached_server_records": profile.num_requests,
        "server_error_records": 0,
        "missing_traces": [],
        "missing_server_records": [],
        "coverage_ratio": 1.0,
        "server_record_coverage_ratio": 1.0,
        "active_set_coverage_ratio": 1.0,
        "active_set_size_max": 1,
        "decode_batch_coverage_ratio": 1.0,
        "decode_batch_size_max": 1,
        "token_timing_coverage_ratio": 1.0,
        "token_timing_mismatches": [],
        "token_timing_unknown": [],
    }
    prompt_shape: int | list[int]
    if profile.mixed_prompt_shape:
        prompt_shape = list(profile.prompt_words)
    else:
        prompt_shape = profile.prompt_words[0]
    rows = [
        {
            "backend": backend,
            "prompt_words": prompt_shape,
            "concurrency": concurrency,
            "max_tokens": max_tokens,
            "repeats": profile.repeats,
            "tail_sample_sufficient": profile.num_requests >= 30,
            "passed": True,
            "noisy_cell": "insufficient_repeats" if profile.repeats == 1 else "stable",
            "output_evidence_present": True,
            "trace_coverage_passed": True,
            "metric_sample_coverage_passed": True,
            "benchmark_commands_passed": True,
            "hash_manifests_consistent": True,
            "hash_stability_checked": True,
            "stable_per_request_hashes": True,
            "output_hash_distribution": {
                "hash": profile.num_requests * profile.repeats
            },
            "request_output_hashes_by_repeat": [
                ["hash"] * profile.num_requests for _ in range(profile.repeats)
            ],
            "greedy_output_hashes_by_repeat": [[] for _ in range(profile.repeats)],
            "sampled_output_hashes_by_repeat": [[] for _ in range(profile.repeats)],
            "combined_output_hashes": [
                bench_dsv2lite_http_slo.combined_output_hash(
                    ["hash"] * profile.num_requests
                )
                for _ in range(profile.repeats)
            ],
            "qps": [1.0] * profile.repeats,
            "qps_summary": {
                "median": 1.0,
                "min": 1.0,
                "max": 1.0,
                "samples": profile.repeats,
            },
            "output_tokens_per_s_summary": {
                "median": 1.0,
                "min": 1.0,
                "max": 1.0,
                "samples": profile.repeats,
            },
            "output_tokens_per_s": [1.0] * profile.repeats,
            "ttft_ms": metric(),
            "tpot_ms": metric(),
            "itl_ms": metric(),
            "completed": [profile.num_requests] * profile.repeats,
            "failed": [0] * profile.repeats,
            "timeouts": [0] * profile.repeats,
            "trace_coverage": [copy.deepcopy(trace) for _ in range(profile.repeats)],
            "metric_sample_counts": {
                "ttft": [profile.num_requests] * profile.repeats,
                "tpot": [profile.num_requests] * profile.repeats,
                "itl": [profile.num_requests * (max_tokens - 1)] * profile.repeats,
            },
        }
        for concurrency in profile.concurrency
        for max_tokens in profile.max_tokens
    ]
    return {
        "schema_version": 1,
        "kind": "openai_http_completions_sweep",
        "report_intent": "http_serving_slo",
        "model": "DeepSeek-V2-Lite",
        "backend": backend,
        "contract": {
            "name": profile.name,
            "backend": backend,
            "description": profile.description,
            "required_trace_coverage_ratio": profile.required_trace_coverage_ratio,
            "claim_boundary": profile.claim_boundary,
        },
        "metadata": {
            "commit": commit,
            "model_path": "models/DeepSeek-V2-Lite",
            "server_command": "pegainfer --model-path models/DeepSeek-V2-Lite",
            "source_revision": commit,
            "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
            "model_fingerprint": {
                "config.json": "a" * 64,
                "model.safetensors.index.json": "b" * 64,
                "tokenizer.json": "d" * 64,
            },
            "server_binary_sha256": "c" * 64,
            "backend_runtime_version": "2.26.2" if backend == "nccl" else "host-staged",
            "hardware_toolchain": {
                "gpu": ["NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 5090"],
                "gpu_identity_sha256": ["1" * 64, "2" * 64],
                "nvcc_version": "Cuda compilation tools, release 12.8",
            },
        },
        "workload": profile.workload_contract()
        | {
            "sampling_profiles": {
                "single": {
                    "label": "single",
                    "temperature": 0.0,
                    "top_k": -1,
                    "top_p": 1.0,
                }
            }
        },
        "metric_definitions": {
            "percentile_method": "R7 linear interpolation over sorted samples",
            "ttft": "request start to first non-empty streamed text chunk",
            "tpot": "requires streamed text chunk count to equal server completion_tokens",
            "itl": "requires streamed text chunk count to equal server completion_tokens",
        },
        "latency_budget": {
            "configured": False,
            "passed": None,
            "reason": "No production latency budget.",
        },
        "correctness_gate": {"passed": True},
        "leaf_artifacts": [
            {
                "artifact": (
                    f"artifacts/{backend}/{profile.name}/"
                    f"pw{'-'.join(str(value) for value in profile.prompt_words)}_"
                    f"c{concurrency}_mt{max_tokens}_r{repeat}.json"
                ),
                "sha256": "f" * 64,
            }
            for concurrency in profile.concurrency
            for max_tokens in profile.max_tokens
            for repeat in range(profile.repeats)
        ],
        "run_errors": [],
        "rows": rows,
    }


def all_summaries() -> list[tuple[Path, dict[str, object]]]:
    return [
        (
            Path(backend) / profile.name / "sweep_summary.json",
            valid_summary(backend, profile),
        )
        for backend in bench_dsv2lite_http_slo.BACKENDS
        for profile in bench_dsv2lite_http_slo.PROFILES.values()
    ]


class BenchDsv2LiteHttpSloTests(unittest.TestCase):
    def test_commit_validation_rejects_non_head_commit(self) -> None:
        with self.assertRaises(SystemExit):
            bench_dsv2lite_http_slo.validate_commit("000000000000")

    def test_leaf_artifact_verification_aggregates_child_failures(self) -> None:
        summaries = [(Path("summary.json"), {"leaf_artifacts": []})]
        with mock.patch.object(
            bench_dsv2lite_http_slo,
            "verify_summary_leaf_artifacts",
            return_value={"passed": True, "checked": 3, "failures": []},
        ):
            passed = bench_dsv2lite_http_slo.verify_leaf_artifacts(summaries)
        with mock.patch.object(
            bench_dsv2lite_http_slo,
            "verify_summary_leaf_artifacts",
            return_value={
                "passed": False,
                "checked": 3,
                "failures": ["rows.recomputed"],
            },
        ):
            failed = bench_dsv2lite_http_slo.verify_leaf_artifacts(summaries)

        self.assertTrue(passed["passed"])
        self.assertEqual(passed["checked"], 3)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["failures"][0]["failures"], ["rows.recomputed"])

    def test_documented_run_templates_include_required_provenance_flags(self) -> None:
        text = BENCHMARKING_DOC.read_text(encoding="utf-8")
        marker = "python3 scripts/bench_dsv2lite_http_slo.py run"
        blocks = [section.split("\n\n", 1)[0] for section in text.split(marker)[1:]]

        self.assertEqual(len(blocks), 1)
        for flag in (
            "--profile",
            "--backend",
            "--model-path",
            "--server-command",
            "--commit",
            "--model-revision",
            "--server-binary",
        ):
            self.assertIn(flag, blocks[0])
        for profile in bench_dsv2lite_http_slo.PROFILES:
            self.assertIn(profile, text)

    def test_profile_command_locks_complete_workload(self) -> None:
        args = SimpleNamespace(
            profile="dsv2-lite-long-prompt-smoke",
            backend="nccl",
            base_url="http://127.0.0.1:8000",
            model="DeepSeek-V2-Lite",
            model_path="models/DeepSeek-V2-Lite",
            server_command="pegainfer --model-path models/DeepSeek-V2-Lite",
            model_revision="604d5664dddd88a0433dbae533b7fe9472482de0",
            server_binary=Path("target/release/pegainfer"),
            backend_runtime_version="2.26.2",
            server_log=Path("server.log"),
            commit=TEST_COMMIT,
            record_startup_failure=None,
            out_dir=Path("artifacts/long"),
        )

        command = bench_dsv2lite_http_slo.build_sweep_command(args)

        self.assertEqual(command[command.index("--prompt-words") + 1], "2048")
        self.assertEqual(command[command.index("--max-tokens") + 1], "64")
        self.assertEqual(command[command.index("--num-requests") + 1], "1")
        self.assertEqual(command[command.index("--concurrency") + 1], "1")
        self.assertEqual(command[command.index("--repeats") + 1], "1")
        self.assertEqual(command[command.index("--timeout") + 1], "900.0")
        self.assertEqual(command[command.index("--source-revision") + 1], TEST_COMMIT)
        self.assertEqual(
            command[command.index("--model-revision") + 1],
            "604d5664dddd88a0433dbae533b7fe9472482de0",
        )
        self.assertEqual(
            command[command.index("--backend-runtime-version") + 1], "2.26.2"
        )
        self.assertEqual(
            bench_dsv2lite_http_slo.PROFILES[
                "dsv2-lite-short-decode-heavy"
            ].num_requests,
            32,
        )

    def test_retained_report_requires_all_fixed_children(self) -> None:
        summaries = all_summaries()

        report = bench_dsv2lite_http_slo.build_retained_slo_report(
            "DeepSeek-V2-Lite", summaries
        )
        missing = bench_dsv2lite_http_slo.build_retained_slo_report(
            "DeepSeek-V2-Lite", summaries[:-1]
        )

        self.assertTrue(report["coverage_gate"]["passed"])
        self.assertEqual(report["coverage_gate"]["retained_children"], 6)
        self.assertEqual(report["metadata"]["commit"], TEST_COMMIT)
        self.assertEqual(len(report["profile_spec_sha256"]), 64)
        self.assertFalse(report["latency_budget"]["configured"])
        self.assertIsNone(report["latency_budget"]["passed"])
        self.assertFalse(missing["coverage_gate"]["passed"])
        self.assertEqual(len(missing["coverage_gate"]["missing"]), 1)

    def test_retained_report_rejects_workload_drift(self) -> None:
        summaries = all_summaries()
        drifted = copy.deepcopy(summaries)
        drifted[0][1]["workload"]["timeout_s"] = 241.0

        report = bench_dsv2lite_http_slo.build_retained_slo_report(
            "DeepSeek-V2-Lite", drifted
        )

        self.assertFalse(report["coverage_gate"]["passed"])
        self.assertIn(
            "workload.timeout_s",
            report["coverage_gate"]["invalid_children"][0]["errors"],
        )

    def test_retained_report_rejects_empty_metric_payload(self) -> None:
        summaries = all_summaries()
        summaries[0][1]["rows"][0]["ttft_ms"] = {}

        report = bench_dsv2lite_http_slo.build_retained_slo_report(
            "DeepSeek-V2-Lite", summaries
        )

        self.assertFalse(report["coverage_gate"]["passed"])
        self.assertIn(
            "rows.success.ttft_ms.p50",
            report["coverage_gate"]["invalid_children"][0]["errors"],
        )

    def test_retained_report_rejects_forged_success_counts(self) -> None:
        summaries = all_summaries()
        summaries[0][1]["rows"][0]["failed"] = [1] * 3

        report = bench_dsv2lite_http_slo.build_retained_slo_report(
            "DeepSeek-V2-Lite", summaries
        )

        self.assertFalse(report["coverage_gate"]["passed"])
        self.assertIn(
            "rows.success.failed",
            report["coverage_gate"]["invalid_children"][0]["errors"],
        )

    def test_retained_report_rejects_non_object_child(self) -> None:
        summaries = all_summaries()
        summaries[0] = (summaries[0][0], [])

        report = bench_dsv2lite_http_slo.build_retained_slo_report(
            "DeepSeek-V2-Lite", summaries
        )

        self.assertFalse(report["coverage_gate"]["passed"])
        self.assertEqual(
            report["coverage_gate"]["invalid_children"][0]["errors"],
            ["document must be an object"],
        )

    def test_retained_report_rejects_duplicate_cell(self) -> None:
        summaries = all_summaries()
        summaries[0][1]["rows"].append(copy.deepcopy(summaries[0][1]["rows"][0]))

        report = bench_dsv2lite_http_slo.build_retained_slo_report(
            "DeepSeek-V2-Lite", summaries
        )

        self.assertFalse(report["coverage_gate"]["passed"])
        self.assertIn(
            "rows.duplicate_cells",
            report["coverage_gate"]["invalid_children"][0]["errors"],
        )

    def test_retained_report_rejects_forged_repeat_summary(self) -> None:
        summaries = all_summaries()
        summaries[0][1]["rows"][0]["ttft_ms"]["p50_summary"]["median"] = 2.0

        report = bench_dsv2lite_http_slo.build_retained_slo_report(
            "DeepSeek-V2-Lite", summaries
        )

        self.assertFalse(report["coverage_gate"]["passed"])
        self.assertIn(
            "rows.success.ttft_ms.p50_summary.median_mismatch",
            report["coverage_gate"]["invalid_children"][0]["errors"],
        )

    def test_retained_report_recomputes_noisy_marker(self) -> None:
        summaries = all_summaries()
        row = summaries[0][1]["rows"][0]
        row["qps"] = [1.0, 1.0, 2.0]
        row["qps_summary"] = {
            "median": 1.0,
            "min": 1.0,
            "max": 2.0,
            "samples": 3,
        }

        report = bench_dsv2lite_http_slo.build_retained_slo_report(
            "DeepSeek-V2-Lite", summaries
        )

        self.assertFalse(report["coverage_gate"]["passed"])
        self.assertIn(
            "rows.success.noisy_cell",
            report["coverage_gate"]["invalid_children"][0]["errors"],
        )

    def test_retained_report_recomputes_output_hash_evidence(self) -> None:
        summaries = all_summaries()
        row = summaries[0][1]["rows"][0]
        row["output_hash_distribution"] = {"forged": 96}
        row["combined_output_hashes"] = ["forged"] * 3

        report = bench_dsv2lite_http_slo.build_retained_slo_report(
            "DeepSeek-V2-Lite", summaries
        )

        self.assertFalse(report["coverage_gate"]["passed"])
        errors = report["coverage_gate"]["invalid_children"][0]["errors"]
        self.assertIn("rows.success.output_hash_distribution", errors)
        self.assertIn("rows.success.combined_output_hashes", errors)

    def test_combine_removes_stale_output_before_read_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid = root / "invalid.json"
            invalid.write_text("not json\n", encoding="utf-8")
            output = root / "retained.json"
            output.write_text('{"stale": true}\n', encoding="utf-8")
            args = SimpleNamespace(
                model="DeepSeek-V2-Lite", summary=[invalid], out=output
            )

            with self.assertRaises(Exception):
                bench_dsv2lite_http_slo.combine(args)

            self.assertFalse(output.exists())

    def test_retained_report_rejects_mixed_commits(self) -> None:
        summaries = all_summaries()
        summaries[-1][1]["metadata"]["commit"] = "different"

        report = bench_dsv2lite_http_slo.build_retained_slo_report(
            "DeepSeek-V2-Lite", summaries
        )

        self.assertFalse(report["coverage_gate"]["passed"])
        self.assertFalse(report["coverage_gate"]["commit_consistent"])
        self.assertIsNone(report["metadata"]["commit"])

    def test_retained_report_rejects_provenance_drift(self) -> None:
        summaries = all_summaries()
        summaries[-1][1]["metadata"]["model_revision"] = "different-model-revision"

        report = bench_dsv2lite_http_slo.build_retained_slo_report(
            "DeepSeek-V2-Lite", summaries
        )

        self.assertFalse(report["coverage_gate"]["passed"])
        self.assertFalse(report["coverage_gate"]["provenance_consistent"])

    def test_detects_loaded_nccl_version_from_server_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server_log = Path(tmp) / "server.log"
            server_log.write_text(
                "DeepSeek-V2-Lite NCCL backend loaded: version=2.26.2, version_code=22602\n",
                encoding="utf-8",
            )

            version = bench_dsv2lite_http_slo.detect_backend_runtime_version(
                server_log, "nccl"
            )

        self.assertEqual(version, "2.26.2")

    def test_loaded_nccl_version_takes_precedence_over_generic_log_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server_log = Path(tmp) / "server.log"
            server_log.write_text(
                "DeepSeek-V2-Lite NCCL backend loaded: version=2.26.2, version_code=22602\n"
                "NCCL version 2.30.7\n",
                encoding="utf-8",
            )

            version = bench_dsv2lite_http_slo.detect_backend_runtime_version(
                server_log, "nccl"
            )

        self.assertEqual(version, "2.26.2")

    def test_startup_failure_extracts_rejected_nccl_version_without_loaded_log(
        self,
    ) -> None:
        args = SimpleNamespace(
            profile="dsv2-lite-short-decode-heavy",
            backend="nccl",
            base_url="http://127.0.0.1:8000",
            model="DeepSeek-V2-Lite",
            model_path="models/DeepSeek-V2-Lite",
            server_command="pegainfer --model-path models/DeepSeek-V2-Lite",
            model_revision="604d5664dddd88a0433dbae533b7fe9472482de0",
            server_binary=Path("target/release/pegainfer"),
            backend_runtime_version=None,
            server_log=Path("missing-server.log"),
            commit=TEST_COMMIT,
            record_startup_failure=(
                "DeepSeek-V2-Lite NCCL EP2 on sm_120 requires NCCL >= 2.26.2, "
                "loaded 2.25.1"
            ),
            out_dir=Path("artifacts/startup-failure"),
        )

        command = bench_dsv2lite_http_slo.build_sweep_command(args)

        self.assertEqual(
            command[command.index("--backend-runtime-version") + 1], "2.25.1"
        )
        self.assertIn("--record-startup-failure", command)


if __name__ == "__main__":
    unittest.main()
