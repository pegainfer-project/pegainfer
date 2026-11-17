"""Regression tests for scripts/bench/sweep.py."""

from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "sweep.py"
# sweep.py imports common.py from its own directory.
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("sweep", SCRIPT_PATH)
assert SPEC is not None
sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sweep
assert SPEC.loader is not None
SPEC.loader.exec_module(sweep)


def args_for(sampling_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        base_url="http://127.0.0.1:8000",
        model="fake-model",
        backend=None,
        contract_name=None,
        contract_description=None,
        claim_boundary=None,
        required_trace_coverage=None,
        commit="abcdef123456",
        model_path=None,
        server_command=None,
        source_revision=None,
        model_revision=None,
        server_binary=None,
        backend_runtime_version=None,
        num_requests=2,
        warmup=0,
        prompt_words=[8],
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        sampling_mode=sampling_mode,
        sample_temperature=0.8,
        sample_top_k=40,
        sample_top_p=0.95,
        no_ignore_eos=False,
        timeout=5.0,
        concurrency=[2],
        max_tokens=[4],
        repeats=2,
    )


def report_with_hashes(
    hashes: list[str],
    *,
    output_chunks: int = 1,
    labels: list[str] | None = None,
) -> dict[str, object]:
    if labels is None:
        labels = ["single"] * len(hashes)
    assert len(labels) == len(hashes)
    return {
        "workload": {
            "prompt_words": 8,
            "concurrency": 2,
            "max_tokens": 4,
        },
        "summary": {
            "completed": len(hashes),
            "failed": 0,
            "timeouts": 0,
            "combined_output_hash": sweep.combined_output_hash(hashes),
            "output_hash_distribution": sweep.value_counts(hashes),
            "qps": 1.0,
            "input_tokens_per_s": 2.0,
            "output_tokens_per_s": 3.0,
        },
        "metrics": {
            "ttft": {
                "avg_ms": 4.0,
                "p50_ms": 4.0,
                "p95_ms": 4.0,
                "p99_ms": 4.0,
                "samples": len(hashes),
            },
            "tpot": {
                "avg_ms": 5.0,
                "p50_ms": 5.0,
                "p95_ms": 5.0,
                "p99_ms": 5.0,
                "samples": len(hashes),
            },
            "itl": {
                "avg_ms": 6.0,
                "p50_ms": 6.0,
                "p95_ms": 6.0,
                "p99_ms": 6.0,
                "samples": len(hashes) * 3,
            },
        },
        "server_trace": {
            "phases_ms": {},
            "coverage_ratio": 0.0,
            "active_set_coverage_ratio": 0.0,
            "decode_batch_coverage_ratio": 0.0,
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
        "metadata": {"benchmark_returncode": 0},
        "requests": [
            {
                "ok": True,
                "max_tokens": 4,
                "output_chunks": output_chunks,
                "output_hash": value,
                "sampling_label": label,
                "ttft_ms": 4.0,
                "tpot_ms": 5.0,
                "itl_ms": [6.0, 6.0, 6.0],
            }
            for value, label in zip(hashes, labels)
        ],
    }


def retained_leaf_report(
    args: SimpleNamespace, hashes: list[str], artifact: Path
) -> dict[str, object]:
    report = report_with_hashes(hashes)
    report.update(
        {
            "schema_version": 1,
            "kind": "openai_http_completions_stream_benchmark",
            "report_intent": "http_serving_slo",
            "base_url": args.base_url,
            "model": args.model,
            "backend": args.backend,
            "contract": {
                "name": args.contract_name,
                "backend": args.backend,
                "description": args.contract_description,
                "required_trace_coverage_ratio": args.required_trace_coverage,
                "claim_boundary": args.claim_boundary,
            },
        }
    )
    report["workload"].update(
        {
            "num_requests": args.num_requests,
            "warmup": args.warmup,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "sampling_mode": args.sampling_mode,
            "sampling_profiles": sweep.sampling_profiles(args),
            "ignore_eos": not args.no_ignore_eos,
            "timeout_s": args.timeout,
            "timeout_kind": "absolute_request_deadline",
        }
    )
    report["metadata"].update(
        {
            "commit": args.commit,
            "backend": args.backend,
            "contract_name": args.contract_name,
            "model_path": args.model_path,
            "server_command": args.server_command,
            "source_revision": args.source_revision,
            "model_revision": args.model_revision,
            "model_fingerprint": {},
            "server_binary_sha256": None,
            "backend_runtime_version": args.backend_runtime_version,
            "hardware_toolchain": sweep.detect_hardware_toolchain(),
            "benchmark_artifact": str(artifact),
        }
    )
    return report


class BenchHttpSweepTests(unittest.TestCase):
    def test_prepare_summary_path_removes_stale_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            stale = out_dir / "sweep_summary.json"
            stale_cell = out_dir / "pw8_c2_mt4_r0.json"
            unrelated = out_dir / "notes.json"
            stale.write_text('{"passed": true}\n', encoding="utf-8")
            stale_cell.write_text('{"passed": true}\n', encoding="utf-8")
            unrelated.write_text("{}\n", encoding="utf-8")

            summary_path = sweep.prepare_summary_path(out_dir)

            self.assertEqual(summary_path, stale)
            self.assertFalse(stale.exists())
            self.assertFalse(stale_cell.exists())
            self.assertTrue(unrelated.exists())

    def test_run_one_does_not_reuse_stale_child_artifact(self) -> None:
        args = args_for("single")
        args.base_url = "http://127.0.0.1:1"
        args.server_log = None
        with tempfile.TemporaryDirectory() as tmp:
            args.out_dir = Path(tmp)
            stale = args.out_dir / "pw8_c2_mt4_r0.json"
            stale.write_text('{"stale": true}\n', encoding="utf-8")
            with mock.patch.object(
                sweep.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=1),
            ):
                with self.assertRaises(FileNotFoundError):
                    sweep.run_one(args, 8, 2, 4, 0)

            self.assertFalse(stale.exists())

    def test_summary_hashes_each_leaf_artifact(self) -> None:
        args = args_for("single")
        with tempfile.TemporaryDirectory() as tmp:
            reports = []
            for repeat in range(2):
                report = report_with_hashes(["a", "b"])
                artifact = Path(tmp) / f"repeat-{repeat}.json"
                report["metadata"]["benchmark_artifact"] = str(artifact)
                sweep.write_json(artifact, report)
                reports.append(report)

            summary = sweep.build_summary(args, reports)

        self.assertEqual(len(summary["leaf_artifacts"]), 2)
        self.assertTrue(
            all(len(artifact["sha256"]) == 64 for artifact in summary["leaf_artifacts"])
        )

    def test_leaf_verifier_recomputes_rows_after_manifest_rehash(self) -> None:
        args = args_for("single")
        args.backend = "host-staged"
        args.contract_name = "retained-test"
        args.contract_description = "Retained test contract."
        args.claim_boundary = "HTTP test evidence only."
        with tempfile.TemporaryDirectory() as tmp:
            reports = []
            paths = []
            for repeat in range(args.repeats):
                path = Path(tmp) / f"repeat-{repeat}.json"
                report = retained_leaf_report(args, ["a", "b"], path)
                sweep.write_json(path, report)
                reports.append(report)
                paths.append(path)
            summary = sweep.build_summary(args, reports)

            passed = sweep.verify_summary_leaf_artifacts(summary)
            reports[0]["summary"]["qps"] = 99.0
            sweep.write_json(paths[0], reports[0])
            summary["leaf_artifacts"][0]["sha256"] = sweep.sha256_file(
                paths[0]
            )
            failed = sweep.verify_summary_leaf_artifacts(summary)

        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])
        self.assertIn("rows.recomputed", failed["failures"])

    def test_leaf_verifier_rejects_provenance_drift_after_manifest_rehash(self) -> None:
        args = args_for("single")
        args.backend = "host-staged"
        args.contract_name = "retained-test"
        args.contract_description = "Retained test contract."
        args.claim_boundary = "HTTP test evidence only."
        with tempfile.TemporaryDirectory() as tmp:
            reports = []
            paths = []
            for repeat in range(args.repeats):
                path = Path(tmp) / f"repeat-{repeat}.json"
                report = retained_leaf_report(args, ["a", "b"], path)
                sweep.write_json(path, report)
                reports.append(report)
                paths.append(path)
            summary = sweep.build_summary(args, reports)

            reports[0]["metadata"]["server_binary_sha256"] = "f" * 64
            sweep.write_json(paths[0], reports[0])
            summary["leaf_artifacts"][0]["sha256"] = sweep.sha256_file(
                paths[0]
            )
            failed = sweep.verify_summary_leaf_artifacts(summary)

        self.assertFalse(failed["passed"])
        self.assertIn(
            "leaf_artifacts[0].metadata.server_binary_sha256", failed["failures"]
        )

    def test_metric_samples_allow_valid_early_eos(self) -> None:
        report = report_with_hashes(["a", "b"])
        report["metrics"]["tpot"]["samples"] = 0
        report["metrics"]["itl"]["samples"] = 0
        for request in report["requests"]:
            request["tpot_ms"] = None
            request["itl_ms"] = []

        self.assertTrue(sweep.metric_sample_counts_pass(report))

    def test_nonzero_child_returncode_fails_cell(self) -> None:
        args = args_for("single")
        report = report_with_hashes(["a", "b"])
        report["metadata"]["benchmark_returncode"] = 1

        summary = sweep.build_summary(args, [report, copy.deepcopy(report)])

        self.assertFalse(summary["correctness_gate"]["passed"])
        self.assertFalse(summary["rows"][0]["benchmark_commands_passed"])
        self.assertEqual(summary["rows"][0]["noisy_cell"], "benchmark_error")

    def test_missing_child_returncode_fails_cell(self) -> None:
        args = args_for("single")
        report = report_with_hashes(["a", "b"])
        del report["metadata"]["benchmark_returncode"]

        summary = sweep.build_summary(args, [report, copy.deepcopy(report)])

        self.assertFalse(summary["correctness_gate"]["passed"])
        self.assertFalse(summary["rows"][0]["benchmark_commands_passed"])
        self.assertEqual(summary["rows"][0]["noisy_cell"], "benchmark_error")

    def test_summary_records_repeat_noise_hashes_and_trace_gate(self) -> None:
        args = args_for("single")
        args.backend = "nccl"
        args.contract_name = "fixed-contract"
        args.contract_description = "fixed workload"
        args.claim_boundary = "HTTP only"
        args.required_trace_coverage = 1.0
        first = report_with_hashes(["hash-a", "hash-b"])
        second = copy.deepcopy(first)
        first["metrics"]["ttft"]["p95_ms"] = 100.0
        second["metrics"]["ttft"]["p95_ms"] = 130.0
        first["summary"]["output_tokens_per_s"] = 100.0
        second["summary"]["output_tokens_per_s"] = 70.0
        for report in (first, second):
            report["server_trace"].update(
                {
                    "traced_requests": 2,
                    "coverage_ratio": 1.0,
                    "active_set_coverage_ratio": 1.0,
                    "decode_batch_coverage_ratio": 1.0,
                    "active_set_size_max": 2,
                    "decode_batch_size_max": 2,
                    "token_timing_coverage_ratio": 1.0,
                    "token_timing_mismatches": [],
                    "token_timing_unknown": [],
                }
            )

        summary = sweep.build_summary(args, [first, second])

        row = summary["rows"][0]
        self.assertTrue(summary["correctness_gate"]["passed"])
        self.assertEqual(row["noisy_cell"], "noisy")
        self.assertEqual(row["output_hash_distribution"], {"hash-a": 2, "hash-b": 2})
        self.assertEqual(
            row["request_output_hashes_by_repeat"],
            [["hash-a", "hash-b"], ["hash-a", "hash-b"]],
        )
        self.assertTrue(row["hash_manifests_consistent"])
        self.assertTrue(row["trace_coverage_passed"])
        self.assertEqual(row["ttft_ms"]["p95_summary"]["median"], 115.0)
        self.assertEqual(
            sweep.noisy_cell_marker([first]),
            "insufficient_repeats",
        )

    def test_startup_failure_summary_retains_contract(self) -> None:
        args = args_for("single")
        args.backend = "nccl"
        args.contract_name = "fixed-contract"
        args.contract_description = "fixed workload"
        args.claim_boundary = "HTTP only"
        args.required_trace_coverage = 1.0
        args.server_log = Path("server.log")
        args.mixed_prompt_shape = False

        summary = sweep.build_startup_failure_summary(
            args, "server failed before communicator creation"
        )

        self.assertEqual(summary["report_intent"], "http_serving_slo_startup_failure")
        self.assertFalse(summary["correctness_gate"]["passed"])
        self.assertEqual(summary["rows"][0]["noisy_cell"], "startup_failure")
        self.assertEqual(summary["rows"][0]["failed"], [2, 2])
        self.assertEqual(summary["run_errors"][0]["phase"], "server_startup")

    def test_mixed_sampling_does_not_require_repeat_hash_stability(self) -> None:
        summary = sweep.build_summary(
            args_for("mixed-greedy-sampled"),
            [
                report_with_hashes(
                    ["greedy", "sampled-a"], labels=["greedy", "sampled"]
                ),
                report_with_hashes(
                    ["greedy", "sampled-b"], labels=["greedy", "sampled"]
                ),
            ],
        )

        self.assertTrue(summary["correctness_gate"]["passed"])
        self.assertFalse(summary["rows"][0]["hash_stability_checked"])
        self.assertTrue(summary["rows"][0]["greedy_hash_stability_checked"])
        self.assertTrue(summary["rows"][0]["output_evidence_present"])
        self.assertFalse(summary["rows"][0]["stable_per_request_hashes"])
        self.assertTrue(summary["rows"][0]["stable_greedy_hashes"])
        self.assertTrue(summary["rows"][0]["sampled_hashes_present"])
        self.assertEqual(
            summary["workload"]["sampling_profiles"]["sampled"]["top_k"], 40
        )

    def test_mixed_sampling_requires_sampled_requests(self) -> None:
        summary = sweep.build_summary(
            args_for("mixed-greedy-sampled"),
            [
                report_with_hashes(["greedy-a"], labels=["greedy"]),
                report_with_hashes(["greedy-a"], labels=["greedy"]),
            ],
        )

        self.assertFalse(summary["correctness_gate"]["passed"])
        self.assertTrue(summary["rows"][0]["greedy_hashes_present"])
        self.assertFalse(summary["rows"][0]["sampled_hashes_present"])
        self.assertFalse(summary["rows"][0]["passed"])

    def test_mixed_sampling_requires_greedy_hash_stability(self) -> None:
        summary = sweep.build_summary(
            args_for("mixed-greedy-sampled"),
            [
                report_with_hashes(
                    ["greedy-a", "sampled-a"], labels=["greedy", "sampled"]
                ),
                report_with_hashes(
                    ["greedy-b", "sampled-b"], labels=["greedy", "sampled"]
                ),
            ],
        )

        self.assertFalse(summary["correctness_gate"]["passed"])
        self.assertTrue(summary["rows"][0]["greedy_hash_stability_checked"])
        self.assertFalse(summary["rows"][0]["stable_greedy_hashes"])
        self.assertFalse(summary["rows"][0]["passed"])

    def test_mixed_sampling_still_requires_output_evidence(self) -> None:
        summary = sweep.build_summary(
            args_for("mixed-greedy-sampled"),
            [
                report_with_hashes(["greedy", "sampled"], labels=["greedy", "sampled"]),
                report_with_hashes(
                    ["", ""], output_chunks=0, labels=["greedy", "sampled"]
                ),
            ],
        )

        self.assertFalse(summary["correctness_gate"]["passed"])
        self.assertFalse(summary["rows"][0]["output_evidence_present"])
        self.assertFalse(summary["rows"][0]["passed"])

    def test_single_sampling_requires_repeat_hash_stability(self) -> None:
        summary = sweep.build_summary(
            args_for("single"),
            [report_with_hashes(["a", "b"]), report_with_hashes(["c", "d"])],
        )

        self.assertFalse(summary["correctness_gate"]["passed"])
        self.assertTrue(summary["rows"][0]["hash_stability_checked"])
        self.assertFalse(summary["rows"][0]["passed"])

    def test_leaf_hash_summary_must_match_request_hashes(self) -> None:
        args = args_for("single")
        report = report_with_hashes(["a", "b"])
        report["summary"]["output_hash_distribution"] = {"forged": 2}

        summary = sweep.build_summary(args, [report, copy.deepcopy(report)])

        self.assertFalse(summary["correctness_gate"]["passed"])
        self.assertFalse(summary["rows"][0]["hash_manifests_consistent"])


if __name__ == "__main__":
    unittest.main()
