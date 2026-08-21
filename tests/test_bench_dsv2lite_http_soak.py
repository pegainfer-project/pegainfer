#!/usr/bin/env python3
"""Regression tests for scripts/bench_dsv2lite_http_soak.py."""

from __future__ import annotations

import importlib.util
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "bench_dsv2lite_http_soak.py"
)
SCRIPTS_DIR = SCRIPT_PATH.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location("bench_dsv2lite_http_soak", SCRIPT_PATH)
assert SPEC and SPEC.loader
bench_dsv2lite_http_soak = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_dsv2lite_http_soak
SPEC.loader.exec_module(bench_dsv2lite_http_soak)


def args(**overrides: object) -> SimpleNamespace:
    values = {
        "backend": "host-staged",
        "base_url": "http://127.0.0.1:18000",
        "model": "DeepSeek-V2-Lite",
        "model_path": "models/DeepSeek-V2-Lite",
        "server_command": "pegainfer --model-path models/DeepSeek-V2-Lite",
        "server_log": Path("server.log"),
        "server_pid": None,
        "commit": "a" * 12,
        "model_revision": "model-rev",
        "server_binary": Path("target/release/pegainfer"),
        "backend_runtime_version": "host-staged",
        "duration_s": 60.0,
        "bucket_s": 30.0,
        "num_requests": 8,
        "concurrency": [4, 8],
        "prompt_words": [64],
        "max_tokens": [64],
        "timeout": 240.0,
        "max_buckets": None,
        "required_trace_coverage": 1.0,
        "contract_name": "dsv2-lite-http-soak",
        "contract_description": "soak",
        "claim_boundary": bench_dsv2lite_http_soak.DEFAULT_CLAIM_BOUNDARY,
        "no_ignore_eos": False,
        "skip_clean_followup": False,
        "stop_on_failure": False,
        "out_dir": Path("artifacts/soak"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def resource(wall_s: float, rss_kib: int, gpu_mib: list[int]):
    return bench_dsv2lite_http_soak.ResourceSample(
        wall_s=wall_s,
        rss_kib=rss_kib,
        gpu_memory_used_mib=gpu_mib,
        gpu_memory_scope="device_total",
    )


def mark_artifact_loaded(record: dict[str, object], digest: str) -> dict[str, object]:
    record["sha256"] = digest
    for artifact in record["leaf_artifacts"]:
        artifact["sha256"] = digest
    return record


def leaf_report(
    *,
    completed: int = 8,
    failed: int = 0,
    timeouts: int = 0,
    qps: float = 2.0,
    output_tokens_per_s: float = 128.0,
) -> dict[str, object]:
    requests = [
        {
            "request_id": f"ok-{index}",
            "ok": True,
            "timed_out": False,
            "error": None,
            "ttft_ms": 10.0 + index,
            "tpot_ms": 1.0,
            "itl_ms": [1.0, 1.0],
            "output_hash": "hash",
            "token_timing_valid": True,
            "server_trace": {
                "terminal_reason": "completed_length",
                "active_set_size": 4,
                "active_set_size_max": 4,
                "pending_queue_size_max": 1,
                "decode_batch_size_max": 3,
                "completion_tokens": 2,
            },
        }
        for index in range(completed)
    ]
    requests.extend(
        {
            "request_id": f"failed-{index}",
            "ok": False,
            "timed_out": index < timeouts,
            "error": "boom",
            "ttft_ms": None,
            "tpot_ms": None,
            "itl_ms": [],
            "output_hash": "",
            "token_timing_valid": False,
            "server_trace": {
                "terminal_reason": "error",
                "active_set_size": 4,
                "active_set_size_max": 4,
                "pending_queue_size_max": 1,
                "decode_batch_size_max": 3,
            },
        }
        for index in range(failed)
    )
    return {
        "summary": {
            "completed": completed,
            "failed": failed,
            "timeouts": timeouts,
            "wall_s": 4.0,
            "qps": qps,
            "input_tokens_total": completed * 80,
            "output_tokens_total": completed * 2,
            "input_tokens_per_s": 512.0,
            "output_tokens_per_s": output_tokens_per_s,
            "output_hash_distribution": {"hash": completed},
            "combined_output_hash": "combined",
        },
        "metrics": {
            "ttft": {"p50_ms": 10.0, "p95_ms": 20.0, "p99_ms": 30.0},
            "tpot": {"p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 3.0},
            "itl": {"p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 3.0},
        },
        "server_trace": {
            "coverage_ratio": 1.0,
            "server_record_coverage_ratio": 1.0,
            "active_set_coverage_ratio": 1.0,
            "decode_batch_coverage_ratio": 1.0,
            "token_timing_coverage_ratio": 1.0,
            "missing_traces": [],
            "missing_server_records": [],
        },
        "requests": requests,
    }


def bucket_record(
    *,
    concurrency: int = 4,
    completed: int = 8,
    failed: int = 0,
    timeouts: int = 0,
    returncode: int = 0,
) -> dict[str, object]:
    record = bench_dsv2lite_http_soak.bucket_record(
        bucket_index=0,
        concurrency=concurrency,
        report_path=Path(f"bucket-c{concurrency}.json"),
        report=leaf_report(completed=completed, failed=failed, timeouts=timeouts),
        returncode=returncode,
        resource_before=resource(0.0, 100, [1000, 1000]),
        resource_after=resource(10.0, 110, [1100, 1100]),
    )
    return mark_artifact_loaded(record, "f" * 64)


def clean_followup_record(
    *, completed: int = 1, failed: int = 0, timeouts: int = 0, returncode: int = 0
) -> dict[str, object]:
    record = bench_dsv2lite_http_soak.bucket_record(
        bucket_index=-1,
        concurrency=1,
        report_path=Path("clean_followup.json"),
        report=leaf_report(completed=completed, failed=failed, timeouts=timeouts),
        returncode=returncode,
        resource_before=resource(10.0, 110, [1100, 1100]),
        resource_after=resource(11.0, 111, [1101, 1101]),
    )
    return mark_artifact_loaded(record, "e" * 64)


def passing_summary(
    *, backend: str = "host-staged", runtime: str = "host-staged"
) -> dict[str, object]:
    return bench_dsv2lite_http_soak.build_summary(
        args(
            backend=backend,
            backend_runtime_version=runtime,
            concurrency=[4],
            duration_s=4.0,
            required_trace_coverage=1.0,
        ),
        buckets=[bucket_record(concurrency=4)],
        resources=[resource(0.0, 100, [1000, 1000])],
        clean_followup=clean_followup_record(),
        run_errors=[],
        started_wall_s=0.0,
        ended_wall_s=11.0,
    )


class BenchDsv2LiteHttpSoakTests(unittest.TestCase):
    def test_leaf_command_locks_soak_workload_and_provenance(self) -> None:
        command = bench_dsv2lite_http_soak.build_leaf_command(
            args(),
            out=Path("bucket.json"),
            concurrency=8,
            num_requests=32,
            prompt_words=64,
            max_tokens=64,
            contract_name="dsv2-lite-http-soak",
            contract_description="soak",
        )

        self.assertEqual(command[command.index("--model") + 1], "DeepSeek-V2-Lite")
        self.assertEqual(command[command.index("--num-requests") + 1], "32")
        self.assertEqual(command[command.index("--concurrency") + 1], "8")
        self.assertEqual(command[command.index("--prompt-words") + 1], "64")
        self.assertEqual(command[command.index("--max-tokens") + 1], "64")
        self.assertEqual(command[command.index("--temperature") + 1], "0.0")
        self.assertEqual(command[command.index("--top-k") + 1], "-1")
        self.assertEqual(command[command.index("--top-p") + 1], "1.0")
        self.assertIn("--server-command", command)
        self.assertIn("--model-revision", command)
        self.assertIn("--backend-runtime-version", command)
        self.assertIn("--required-trace-coverage", command)

    def test_bucket_record_preserves_trace_resources_and_terminal_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bucket.json"
            path.write_text('{"ok": true}\n', encoding="utf-8")

            record = bench_dsv2lite_http_soak.bucket_record(
                bucket_index=0,
                concurrency=4,
                report_path=path,
                report=leaf_report(),
                returncode=0,
                resource_before=resource(0.0, 100, [1000, 1001]),
                resource_after=resource(4.0, 120, [1100, 1110]),
            )

        self.assertTrue(record["report_loaded"])
        self.assertEqual(record["completed"], 8)
        self.assertEqual(record["active_set_size_max"], 4)
        self.assertEqual(record["pending_queue_size_max"], 1)
        self.assertEqual(record["decode_batch_size_max"], 3)
        self.assertEqual(record["terminal_reasons"], {"completed_length": 8})
        self.assertEqual(record["resource_after"]["gpu_memory_used_mib"], [1100, 1110])
        self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")

    def test_bucket_window_aggregates_multiple_leaf_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first_path = Path(tmp) / "bucket0000_leaf000.json"
            second_path = Path(tmp) / "bucket0000_leaf001.json"
            first_path.write_text('{"leaf": 0}\n', encoding="utf-8")
            second_path.write_text('{"leaf": 1}\n', encoding="utf-8")

            record = bench_dsv2lite_http_soak.aggregate_bucket_record(
                bucket_index=0,
                concurrency=4,
                leaf_runs=[
                    {
                        "report_path": first_path,
                        "report": leaf_report(),
                        "returncode": 0,
                    },
                    {
                        "report_path": second_path,
                        "report": leaf_report(),
                        "returncode": 0,
                    },
                ],
                resource_before=resource(0.0, 100, [1000, 1001]),
                resource_after=resource(10.0, 120, [1100, 1110]),
            )

        self.assertTrue(record["report_loaded"])
        self.assertEqual(record["leaf_count"], 2)
        self.assertEqual(len(record["leaf_artifacts"]), 2)
        self.assertEqual(record["completed"], 16)
        self.assertEqual(record["failed"], 0)
        self.assertEqual(record["timeouts"], 0)
        self.assertEqual(record["qps"], 1.6)
        self.assertEqual(record["output_tokens_per_s"], 3.2)
        self.assertEqual(record["ttft_ms"]["p50"], 13.5)
        self.assertEqual(record["trace_coverage"]["coverage_ratio"], 1.0)
        self.assertEqual(record["trace_coverage"]["active_set_coverage_ratio"], 1.0)
        self.assertEqual(record["trace_coverage"]["decode_batch_coverage_ratio"], 1.0)
        self.assertEqual(record["trace_coverage"]["token_timing_coverage_ratio"], 1.0)
        self.assertEqual(record["active_set_size_max"], 4)
        self.assertEqual(record["pending_queue_size_max"], 1)
        self.assertEqual(record["decode_batch_size_max"], 3)
        self.assertEqual(record["terminal_reasons"], {"completed_length": 16})

    def test_backend_summary_reports_drift_and_clean_followup_gate(self) -> None:
        first = bench_dsv2lite_http_soak.bucket_record(
            bucket_index=0,
            concurrency=4,
            report_path=Path("bucket0.json"),
            report=leaf_report(qps=4.0, output_tokens_per_s=256.0),
            returncode=0,
            resource_before=resource(0.0, 100, [1000, 1000]),
            resource_after=resource(10.0, 110, [1100, 1100]),
        )
        mark_artifact_loaded(first, "a" * 64)
        last = bench_dsv2lite_http_soak.bucket_record(
            bucket_index=1,
            concurrency=4,
            report_path=Path("bucket1.json"),
            report=leaf_report(qps=2.0, output_tokens_per_s=128.0),
            returncode=0,
            resource_before=resource(10.0, 110, [1100, 1100]),
            resource_after=resource(20.0, 120, [1200, 1200]),
        )
        mark_artifact_loaded(last, "b" * 64)
        followup = bench_dsv2lite_http_soak.bucket_record(
            bucket_index=-1,
            concurrency=1,
            report_path=Path("clean_followup.json"),
            report=leaf_report(completed=1, qps=1.0, output_tokens_per_s=64.0),
            returncode=0,
            resource_before=resource(20.0, 120, [1200, 1200]),
            resource_after=resource(21.0, 121, [1201, 1201]),
        )
        mark_artifact_loaded(followup, "c" * 64)

        summary = bench_dsv2lite_http_soak.build_summary(
            args(concurrency=[4], duration_s=8.0, required_trace_coverage=1.0),
            buckets=[first, last],
            resources=[
                resource(0.0, 100, [1000, 1000]),
                resource(20.0, 120, [1200, 1200]),
            ],
            clean_followup=followup,
            run_errors=[],
            started_wall_s=100.0,
            ended_wall_s=121.0,
        )

        self.assertTrue(summary["soak_gate"]["passed"])
        self.assertTrue(summary["soak_gate"]["bucket_coverage_passed"])
        self.assertTrue(summary["soak_gate"]["duration_coverage_passed"])
        self.assertTrue(summary["soak_gate"]["leaf_commands_passed"])
        self.assertEqual(summary["summary"]["completed"], 16)
        self.assertEqual(summary["summary"]["terminal_reasons"], {"completed_length": 16})
        self.assertEqual(
            summary["drift_by_concurrency"]["4"]["qps"]["median_delta_pct"],
            -50.0,
        )
        self.assertTrue(summary["clean_followup"]["passed"])
        self.assertIn("not direct decode attribution", summary["claim_boundary"])

    def test_backend_summary_fails_missing_trace_coverage(self) -> None:
        bucket = bucket_record()
        bucket["trace_coverage"]["token_timing_coverage_ratio"] = 0.5

        summary = bench_dsv2lite_http_soak.build_summary(
            args(concurrency=[4], required_trace_coverage=1.0),
            buckets=[bucket],
            resources=[],
            clean_followup=clean_followup_record(),
            run_errors=[],
            started_wall_s=0.0,
            ended_wall_s=11.0,
        )

        self.assertFalse(summary["soak_gate"]["passed"])
        self.assertFalse(summary["soak_gate"]["trace_coverage_passed"])

    def test_backend_summary_fails_leaf_command_error(self) -> None:
        summary = bench_dsv2lite_http_soak.build_summary(
            args(concurrency=[4], required_trace_coverage=1.0),
            buckets=[bucket_record(returncode=1)],
            resources=[],
            clean_followup=clean_followup_record(),
            run_errors=[],
            started_wall_s=0.0,
            ended_wall_s=11.0,
        )

        self.assertFalse(summary["soak_gate"]["passed"])
        self.assertFalse(summary["soak_gate"]["leaf_commands_passed"])

    def test_backend_summary_fails_empty_or_missing_bucket(self) -> None:
        summary = bench_dsv2lite_http_soak.build_summary(
            args(concurrency=[4, 8], required_trace_coverage=None),
            buckets=[bucket_record(concurrency=4)],
            resources=[],
            clean_followup=None,
            run_errors=[],
            started_wall_s=0.0,
            ended_wall_s=1.0,
        )

        self.assertFalse(summary["soak_gate"]["passed"])
        self.assertFalse(summary["soak_gate"]["bucket_coverage_passed"])

    def test_backend_summary_rejects_non_retained_gate_shapes(self) -> None:
        cases = [
            (
                "short_duration",
                args(concurrency=[4], duration_s=60.0, required_trace_coverage=1.0),
                clean_followup_record(),
                "duration_coverage_passed",
            ),
            (
                "capped_duration",
                args(
                    concurrency=[4],
                    duration_s=10.0,
                    max_buckets=1,
                    required_trace_coverage=1.0,
                ),
                clean_followup_record(),
                "duration_coverage_passed",
            ),
            (
                "missing_clean_followup",
                args(concurrency=[4], duration_s=10.0, required_trace_coverage=1.0),
                None,
                "clean_followup_passed",
            ),
            (
                "failed_clean_followup",
                args(concurrency=[4], duration_s=10.0, required_trace_coverage=1.0),
                clean_followup_record(completed=0, failed=1, returncode=1),
                "clean_followup_passed",
            ),
        ]
        for name, case_args, clean_followup, failed_gate in cases:
            with self.subTest(name=name):
                summary = bench_dsv2lite_http_soak.build_summary(
                    case_args,
                    buckets=[bucket_record(concurrency=4)],
                    resources=[],
                    clean_followup=clean_followup,
                    run_errors=[],
                    started_wall_s=0.0,
                    ended_wall_s=11.0,
                )

                self.assertFalse(summary["soak_gate"]["passed"])
                self.assertFalse(summary["soak_gate"][failed_gate])

    def test_backend_summary_fails_missing_leaf_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first_path = Path(tmp) / "bucket0000_leaf000.json"
            missing_path = Path(tmp) / "bucket0000_leaf001.json"
            first_path.write_text('{"leaf": 0}\n', encoding="utf-8")
            bucket = bench_dsv2lite_http_soak.aggregate_bucket_record(
                bucket_index=0,
                concurrency=4,
                leaf_runs=[
                    {
                        "report_path": first_path,
                        "report": leaf_report(),
                        "returncode": 0,
                    },
                    {
                        "report_path": missing_path,
                        "report": None,
                        "returncode": 0,
                    },
                ],
                resource_before=resource(0.0, 100, [1000, 1001]),
                resource_after=resource(10.0, 120, [1100, 1110]),
            )

        summary = bench_dsv2lite_http_soak.build_summary(
            args(concurrency=[4], duration_s=10.0, required_trace_coverage=1.0),
            buckets=[bucket],
            resources=[],
            clean_followup=clean_followup_record(),
            run_errors=[],
            started_wall_s=0.0,
            ended_wall_s=11.0,
        )

        self.assertFalse(bucket["report_loaded"])
        self.assertFalse(summary["soak_gate"]["passed"])
        self.assertFalse(summary["soak_gate"]["leaf_commands_passed"])

    def test_drift_summary_reports_per_device_gpu_growth(self) -> None:
        first = {"resource_after": {"gpu_memory_used_mib": [10000, 8000]}}
        last = {"resource_after": {"gpu_memory_used_mib": [10000, 9000]}}

        drift = bench_dsv2lite_http_soak.drift_summary([first, last])

        self.assertEqual(drift["gpu_memory_max_mib"]["median_delta_pct"], 0.0)
        self.assertEqual(drift["gpu_memory_total_mib"]["median_delta_pct"], 1000 / 18000 * 100)
        self.assertEqual(drift["gpu_memory_by_device_mib"]["1"]["median_delta_pct"], 12.5)

    def test_combined_report_requires_host_and_nccl(self) -> None:
        host = passing_summary()
        nccl = passing_summary(backend="nccl", runtime="NCCL 2.26.2")

        missing = bench_dsv2lite_http_soak.build_combined_report(
            "DeepSeek-V2-Lite", [(Path("host.json"), host)]
        )
        full = bench_dsv2lite_http_soak.build_combined_report(
            "DeepSeek-V2-Lite", [(Path("host.json"), host), (Path("nccl.json"), nccl)]
        )

        self.assertFalse(missing["coverage_gate"]["passed"])
        self.assertEqual(missing["coverage_gate"]["missing"], ["nccl"])
        self.assertTrue(full["coverage_gate"]["passed"])
        self.assertEqual(full["coverage_gate"]["required_backends"], 2)
        self.assertEqual(
            full["metadata"]["runtime_boundaries"]["nccl"],
            "NCCL 2.26.2",
        )

    def test_combined_report_rejects_failed_child_gate(self) -> None:
        host = passing_summary()
        nccl = passing_summary(backend="nccl", runtime="NCCL 2.26.2")
        nccl["soak_gate"]["passed"] = False

        report = bench_dsv2lite_http_soak.build_combined_report(
            "DeepSeek-V2-Lite", [(Path("host.json"), host), (Path("nccl.json"), nccl)]
        )

        self.assertFalse(report["coverage_gate"]["passed"])
        self.assertFalse(report["coverage_gate"]["child_gates"]["nccl"])

    def test_combined_report_rejects_provenance_drift(self) -> None:
        host = passing_summary()
        nccl = passing_summary(backend="nccl", runtime="NCCL 2.26.2")
        nccl["metadata"]["model_revision"] = "different-model-rev"

        report = bench_dsv2lite_http_soak.build_combined_report(
            "DeepSeek-V2-Lite", [(Path("host.json"), host), (Path("nccl.json"), nccl)]
        )

        self.assertFalse(report["coverage_gate"]["passed"])
        self.assertFalse(report["coverage_gate"]["provenance_consistent"])

    def test_combined_report_rejects_workload_drift(self) -> None:
        host = passing_summary()
        nccl = passing_summary(backend="nccl", runtime="NCCL 2.26.2")
        nccl = copy.deepcopy(nccl)
        nccl["workload"]["duration_s"] = 120.0

        report = bench_dsv2lite_http_soak.build_combined_report(
            "DeepSeek-V2-Lite", [(Path("host.json"), host), (Path("nccl.json"), nccl)]
        )

        self.assertFalse(report["coverage_gate"]["passed"])
        self.assertFalse(report["coverage_gate"]["contract_consistent"])
        self.assertIsNone(report["metadata"]["contract"])

    def test_combined_report_rejects_missing_runtime_boundary(self) -> None:
        host = passing_summary()
        nccl = passing_summary(backend="nccl", runtime="NCCL 2.26.2")
        nccl = copy.deepcopy(nccl)
        nccl["metadata"]["backend_runtime_version"] = "nccl"

        report = bench_dsv2lite_http_soak.build_combined_report(
            "DeepSeek-V2-Lite", [(Path("host.json"), host), (Path("nccl.json"), nccl)]
        )

        self.assertFalse(report["coverage_gate"]["passed"])
        self.assertFalse(report["coverage_gate"]["runtime_boundaries_present"])
        self.assertEqual(report["coverage_gate"]["missing_runtime_boundaries"], ["nccl"])


if __name__ == "__main__":
    unittest.main()
