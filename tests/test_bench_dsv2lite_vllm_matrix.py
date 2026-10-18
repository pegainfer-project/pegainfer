"""Regression tests for scripts/bench_dsv2lite_vllm_matrix.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import ArgumentTypeError
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench_dsv2lite_vllm_matrix.py"
SPEC = importlib.util.spec_from_file_location("bench_dsv2lite_vllm_matrix", SCRIPT_PATH)
assert SPEC is not None
bench_matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_matrix
assert SPEC.loader is not None
SPEC.loader.exec_module(bench_matrix)

HTTP_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench_http_serving.py"
HTTP_SPEC = importlib.util.spec_from_file_location("bench_http_serving_for_matrix", HTTP_SCRIPT_PATH)
assert HTTP_SPEC is not None
bench_http_serving = importlib.util.module_from_spec(HTTP_SPEC)
sys.modules[HTTP_SPEC.name] = bench_http_serving
assert HTTP_SPEC.loader is not None
HTTP_SPEC.loader.exec_module(bench_http_serving)


class BenchDsv2LiteMatrixTests(unittest.TestCase):
    def summarize_existing_without_metadata_probe(self, args):
        with mock.patch.object(bench_matrix, "metadata", return_value={"test": True}):
            return bench_matrix.summarize_existing(args)

    def base_args(self, root: Path, **overrides):
        values = {
            "summarize_only": root,
            "baseline_summary": None,
            "noisy_threshold": 0.05,
            "model_path": Path("models/DeepSeek-V2-Lite"),
            "model_id": "DeepSeek-V2-Lite",
            "input_len": 64,
            "output_len": 64,
            "num_prompts": 32,
            "num_warmups": 4,
            "concurrency": [1],
            "request_rate": "inf",
            "temperature": 0.0,
            "ignore_eos": True,
            "repeats": 1,
            "hf_python": sys.executable,
            "vllm_cmd": "vllm",
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def valid_trace_payload(self, concurrency=1, *, trace_overrides=None):
        trace_overrides = trace_overrides or {}
        measured = []
        for index in range(32):
            trace = {
                "prompt_tokens": 84 + index % 4,
                "completion_tokens": 64,
                "active_set_size": concurrency,
                "decode_batch_size_max": concurrency,
                "queue_wait_ms": 12.0,
                "prefill_ms": 30.0,
                "first_decode_ms": 4.0,
                "decode_mean_ms": 6.0,
                "decode_total_ms": 384.0,
                "scheduled_to_first_token_ms": 46.0,
                "scheduled_to_terminal_ms": 420.0,
                "stream_flush_ms": 2.0,
                "decode_step_count": 2,
                "batch_decode_steps": 0 if concurrency == 1 else 1,
            }
            trace.update(trace_overrides)
            measured.append(
                bench_http_serving.RequestResult(
                    index=index,
                    request_id=f"bench-{index}",
                    prompt_words=64,
                    max_tokens=64,
                    ok=True,
                    status=200,
                    error=None,
                    timed_out=False,
                    start_s=0.0,
                    start_wall_s=0.0,
                    first_token_s=0.1,
                    first_token_wall_s=0.1,
                    end_s=1.0,
                    end_wall_s=1.0,
                    latency_ms=1000.0,
                    ttft_ms=100.0,
                    tpot_ms=14.0,
                    itl_ms=[14.0],
                    output_chunks=64,
                    output_chars=64,
                    output_hash=f"hash-{index}",
                    text_prefix="x",
                    server_trace=trace,
                )
            )
        args = SimpleNamespace(
            base_url="http://127.0.0.1:8000",
            model="DeepSeek-V2-Lite",
            num_requests=32,
            concurrency=concurrency,
            warmup=0,
            prompt_words=[64],
            max_tokens=[64],
            temperature=0.0,
            ignore_eos=True,
            timeout=900.0,
        )
        return bench_http_serving.build_report(args, measured, wall_s=102.4)

    def valid_http_payload(self, **overrides):
        payload = {
            "model_id": "DeepSeek-V2-Lite",
            "num_prompts": 32,
            "max_concurrency": 1,
            "request_rate": "inf",
            f"{bench_matrix.HTTP_METADATA_PREFIX}input_len": "64",
            f"{bench_matrix.HTTP_METADATA_PREFIX}output_len": "64",
            f"{bench_matrix.HTTP_METADATA_PREFIX}temperature": "0.0",
            f"{bench_matrix.HTTP_METADATA_PREFIX}ignore_eos": "true",
            "num_completed_requests": 32,
            "num_failed_requests": 0,
            "num_timeouts": 0,
            "total_output_tokens": 2048,
            "duration": 64.0,
            "generated_texts": [f"output-{index}" for index in range(32)],
        }
        payload.update(overrides)
        return payload

    def minimal_summary(self, *, noisy=False, http_failed=False):
        http_row = {
            "engine": "vllm-tp2",
            "claim_bucket": bench_matrix.CLAIM_FAILED,
            "passed": False,
            "error": "server_start_failed: old setup failure",
            "startup_failure": "server_start_failed: old setup failure",
            "cells": [],
        } if http_failed else {
            "engine": "vllm-tp2",
            "claim_bucket": bench_matrix.CLAIM_HTTP,
            "passed": True,
            "cells": [],
            "summary_by_concurrency": [
                {
                    "concurrency": 1,
                    "completed": [32],
                    "failed": [0],
                    "timeouts": [0],
                    "output_text_sha256": ["hash-a"],
                    "mean_tpot_ms": {
                        "values": [10.0],
                        "median": 10.0,
                        "min": 10.0,
                        "max": 10.0,
                        "spread_ratio": 0.0,
                        "noisy": False,
                    },
                    "output_tok_s": {
                        "values": [100.0],
                        "median": 100.0,
                        "min": 100.0,
                        "max": 100.0,
                        "spread_ratio": 0.0,
                        "noisy": False,
                    },
                    "noisy": noisy,
                }
            ],
        }
        return {
            "schema_version": 1,
            "kind": "deepseek_v2_lite_vllm_tp2_ep2_benchmark_matrix",
            "metadata": {
                "benchmark_contract": {"input_len": 64, "output_len": 64, "num_prompts": 32},
                "model": {"path": "models/DeepSeek-V2-Lite", "config_sha256": "cfg", "tokenizer_sha256": "tok"},
                "versions": {
                    "nvidia_smi": {"stdout": "NVIDIA GPU, 580.95.05, 12.0"},
                    "nvcc": {"stdout": "Cuda compilation tools, release 12.8"},
                    "nccl": {"available": True, "exit_code": 0, "stdout": "2.30.4"},
                    "vllm": {"stdout": "vllm 0.23.0"},
                },
            },
            "correctness_gate": {
                "claim_bucket": bench_matrix.CLAIM_CORRECTNESS,
                "passed": True,
                "comparison": {"classification": "all_token_text_exact", "warnings": []},
            },
            "direct_diagnostic_batch": [
                {
                    "claim_bucket": bench_matrix.CLAIM_DIRECT,
                    "backend": "host-staged",
                    "batch_size": 1,
                    "passed": True,
                    "token_sha256": "token-hash",
                    "text_sha256": "text-hash",
                    "tpot_ms": 1.0,
                    "output_tok_s": 2.0,
                }
            ],
            "http_concurrency_pressure": [http_row],
            "pegainfer_trace_pass": [
                {
                    "engine": "pegainfer-host-staged",
                    "claim_bucket": bench_matrix.CLAIM_HTTP,
                    "passed": True,
                    "cells": [
                        {
                            "concurrency": 1,
                            "completed": 8,
                            "failed": 0,
                            "timeouts": 0,
                            "missing_trace_count": 0,
                            "trace": {"active_set_size_max": 1, "decode_batch_size_max": 1},
                        }
                    ],
                }
            ],
        }

    def test_display_path_keeps_symlinked_target_repo_relative(self) -> None:
        path = bench_matrix.REPO_ROOT / "target" / "benchmarks" / "deepseek-v2-lite-vllm-tp2-ep2"

        self.assertEqual(
            bench_matrix.display_path(path),
            "target/benchmarks/deepseek-v2-lite-vllm-tp2-ep2",
        )

    def test_error_text_strips_repo_absolute_prefix(self) -> None:
        err = RuntimeError(
            f"failed at {bench_matrix.REPO_ROOT.absolute()}/target/benchmarks/result.json"
        )

        self.assertEqual(
            bench_matrix.error_text(err),
            "failed at target/benchmarks/result.json",
        )

    def test_error_text_redacts_home_absolute_prefix(self) -> None:
        err = RuntimeError(f"failed at {Path.home()}/models/DeepSeek-V2-Lite")

        self.assertEqual(
            bench_matrix.error_text(err),
            "failed at ~/models/DeepSeek-V2-Lite",
        )

    def test_redact_text_strips_repo_and_home_prefixes(self) -> None:
        text = (
            f"{bench_matrix.REPO_ROOT.absolute()}/target/run.log "
            f"{bench_matrix.REPO_ROOT.absolute()} "
            f"{Path.home()}/models/DeepSeek-V2-Lite "
            f"{Path.home()} "
            f"/{'root'}/miniconda3/lib/python3.12/site-packages/vllm "
            "/home/runner/project "
            f"~/{'auto'}{'dl'}-tmp/hf-accuracy-venv/bin/python"
        )

        self.assertEqual(
            bench_matrix.redact_text(text),
            "target/run.log <repo> ~/models/DeepSeek-V2-Lite ~ "
            "~/miniconda3/lib/python3.12/site-packages/vllm ~/project "
            "~/tmp/hf-accuracy-venv/bin/python",
        )

    def test_redact_text_masks_common_sensitive_env_values(self) -> None:
        token_key = "HF" + "_" + "TOKEN"
        hub_key = "HUGGINGFACE" + "_HUB" + "_TOKEN"
        public_key_name = "API" + "_KEY"
        pass_key = "PASS" + "WORD"
        text = (
            f"{token_key}=value_abc {hub_key}='value_def' "
            f"{public_key_name}=live_key {pass_key}=pw"
        )

        redacted = bench_matrix.redact_text(text)

        self.assertIn(f"{token_key}=<redacted>", redacted)
        self.assertIn(f"{hub_key}=<redacted>", redacted)
        self.assertIn(f"{public_key_name}=<redacted>", redacted)
        self.assertIn(f"{pass_key}=<redacted>", redacted)
        self.assertNotIn("value_abc", redacted)
        self.assertNotIn("value_def", redacted)
        self.assertNotIn("live_key", redacted)

    def test_redact_command_applies_text_redaction_per_argument(self) -> None:
        redacted = bench_matrix.redact_command([f"/{'root'}/venv/bin/python", "--version"])

        self.assertEqual(redacted, ["~/venv/bin/python", "--version"])

    def test_redact_payload_recurses_into_lists_and_dicts(self) -> None:
        payload = {
            "command": [f"/{'root'}/venv/bin/python", "--version"],
            "nested": {"path": "/home/runner/project"},
        }

        self.assertEqual(
            bench_matrix.redact_payload(payload),
            {"command": ["~/venv/bin/python", "--version"], "nested": {"path": "~/project"}},
        )

    def test_public_path_hides_external_absolute_model_path(self) -> None:
        path = Path("/private/machines/user/models/DeepSeek-V2-Lite")

        self.assertEqual(
            bench_matrix.public_path(path),
            "<external>/DeepSeek-V2-Lite",
        )

    def test_pegainfer_server_command_keeps_default_features(self) -> None:
        args = SimpleNamespace(model_path=Path("models/DeepSeek-V2-Lite"), model_id="DeepSeek-V2-Lite")
        spec = bench_matrix.ENGINES[0]

        cmd = bench_matrix.server_command(args, spec, 8000)

        self.assertNotIn("--no-default-features", cmd)
        self.assertIn("--features", cmd)
        self.assertIn("deepseek-v2-lite", cmd)

    def test_parse_args_allows_dash_prefixed_vllm_extra_args(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "bench",
                "--plan-only",
                "--vllm-serve-extra-args",
                "--max-num-seqs",
                "16",
            ],
        ):
            args = bench_matrix.parse_args()

        self.assertEqual(args.vllm_serve_extra_args, ["--max-num-seqs", "16"])

    def test_parse_args_allows_separator_before_vllm_extra_args(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "bench",
                "--plan-only",
                "--vllm-serve-extra-args",
                "--",
                "--max-num-seqs",
                "16",
            ],
        ):
            args = bench_matrix.parse_args()

        self.assertEqual(args.vllm_serve_extra_args, ["--max-num-seqs", "16"])

    def test_parse_args_defaults_vllm_extra_args_when_omitted(self) -> None:
        with mock.patch.object(sys, "argv", ["bench", "--plan-only"]):
            args = bench_matrix.parse_args()

        self.assertEqual(args.vllm_serve_extra_args, bench_matrix.default_vllm_extra_args())

    def test_vllm_bench_command_leaves_warmups_to_separate_call(self) -> None:
        args = SimpleNamespace(
            vllm_cmd="vllm",
            model_id="DeepSeek-V2-Lite",
            model_path=Path("models/DeepSeek-V2-Lite"),
            input_len=64,
            output_len=64,
            num_warmups=4,
            request_rate="inf",
            temperature=0.0,
            ignore_eos=True,
        )

        cmd = bench_matrix.vllm_bench_command(
            args,
            port=8000,
            num_prompts=32,
            result_dir=Path("target/results"),
            result_filename="result.json",
            max_concurrency=8,
        )

        self.assertNotIn("--num-warmups", cmd)
        self.assertIn("--save-detailed", cmd)

    def test_metadata_records_custom_hf_python_version_command(self) -> None:
        args = SimpleNamespace(
            model_path=Path("models/DeepSeek-V2-Lite"),
            model_id="DeepSeek-V2-Lite",
            input_len=64,
            output_len=64,
            num_prompts=32,
            num_warmups=4,
            concurrency=[1, 4, 8],
            request_rate="inf",
            temperature=0.0,
            ignore_eos=True,
            repeats=3,
            noisy_threshold=0.05,
            hf_python="/tmp/hf-python",
            vllm_cmd="vllm",
        )

        with mock.patch.object(
            bench_matrix,
            "try_command",
            side_effect=lambda cmd: {"command": cmd, "available": False},
        ):
            meta = bench_matrix.metadata(args)

        self.assertEqual(
            meta["versions"]["hf_python"]["command"],
            ["/tmp/hf-python", "--version"],
        )
        self.assertTrue(meta["versions"]["hf_python_explicit"])

    def test_metadata_records_hf_python_default_as_not_explicit(self) -> None:
        args = SimpleNamespace(
            model_path=Path("models/DeepSeek-V2-Lite"),
            model_id="DeepSeek-V2-Lite",
            input_len=64,
            output_len=64,
            num_prompts=32,
            num_warmups=4,
            concurrency=[1, 4, 8],
            request_rate="inf",
            temperature=0.0,
            ignore_eos=True,
            repeats=3,
            noisy_threshold=0.05,
            hf_python=None,
            vllm_cmd="vllm",
        )

        with mock.patch.object(
            bench_matrix,
            "try_command",
            side_effect=lambda cmd: {"command": cmd, "available": False},
        ):
            meta = bench_matrix.metadata(args)

        self.assertEqual(meta["versions"]["hf_python"]["command"], [sys.executable, "--version"])
        self.assertFalse(meta["versions"]["hf_python_explicit"])
        self.assertIn("--hf-python", meta["versions"]["hf_python_note"])

    def test_decode_nccl_version_code(self) -> None:
        self.assertEqual(
            bench_matrix.decode_nccl_version_code(23007),
            {"version_code": 23007, "version": "2.30.7"},
        )
        self.assertEqual(
            bench_matrix.decode_nccl_version_code(2804),
            {"version_code": 2804, "version": "2.8.4"},
        )
        self.assertIsNone(bench_matrix.decode_nccl_version_code(0))
        self.assertIsNone(bench_matrix.decode_nccl_version_code(True))

    def test_nccl_version_from_library_queries_exact_path(self) -> None:
        class FakeGetVersion:
            def __call__(self, version_ptr):
                version_ptr._obj.value = 23007
                return 0

        fake_library = SimpleNamespace(ncclGetVersion=FakeGetVersion())
        with mock.patch.object(bench_matrix.ctypes, "CDLL", return_value=fake_library):
            probe = bench_matrix.nccl_version_from_library("/wheel/lib/libnccl.so.2")

        self.assertEqual(
            probe,
            {
                "library": "/wheel/lib/libnccl.so.2",
                "available": True,
                "exit_code": 0,
                "version_code": 23007,
                "version": "2.30.7",
            },
        )

    def test_process_nccl_runtime_uses_server_mapped_library(self) -> None:
        maps = (
            "7f00-7f10 r-xp 00000000 00:00 0 /wheel/lib/libnccl.so.2\n"
            "7f10-7f20 r--p 00000000 00:00 0 /wheel/lib/libnccl.so.2\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            (proc_root / "1234").mkdir()
            (proc_root / "5678").mkdir()
            (proc_root / "1234" / "maps").write_text("", encoding="utf-8")
            (proc_root / "5678" / "maps").write_text(maps, encoding="utf-8")
            with mock.patch.object(
                bench_matrix.os,
                "getpgid",
                side_effect=lambda pid: 77 if pid in {1234, 5678} else 88,
            ), mock.patch.object(
                bench_matrix,
                "nccl_version_from_library",
                return_value={
                    "library": "/wheel/lib/libnccl.so.2",
                    "available": True,
                    "exit_code": 0,
                    "version_code": 23007,
                    "version": "2.30.7",
                },
            ) as probe:
                runtime = bench_matrix.process_nccl_runtime(1234, proc_root)

        probe.assert_called_once_with("/wheel/lib/libnccl.so.2")
        self.assertTrue(runtime["available"])
        self.assertEqual(runtime["source"], "server_process_group_maps")
        self.assertEqual(runtime["process_group_pids"], [1234, 5678])
        self.assertEqual(runtime["mapped_pids"], [5678])
        self.assertEqual(runtime["library"], "libnccl.so.2")
        self.assertEqual(runtime["version"], "2.30.7")

    def test_process_nccl_runtime_rejects_missing_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            (proc_root / "1234").mkdir()
            (proc_root / "1234" / "maps").write_text("", encoding="utf-8")
            with mock.patch.object(bench_matrix.os, "getpgid", return_value=77):
                runtime = bench_matrix.process_nccl_runtime(1234, proc_root)

        self.assertFalse(runtime["available"])
        self.assertEqual(runtime["mapped_library_count"], 0)
        self.assertIn("expected exactly one NCCL library", runtime["error"])

    def test_process_nccl_runtime_rejects_multiple_libraries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            for pid, library in (
                ("1234", "/wheel-a/lib/libnccl.so.2"),
                ("5678", "/wheel-b/lib/libnccl.so.2"),
            ):
                (proc_root / pid).mkdir()
                (proc_root / pid / "maps").write_text(
                    f"7f00-7f10 r-xp 00000000 00:00 0 {library}\n",
                    encoding="utf-8",
                )
            with mock.patch.object(bench_matrix.os, "getpgid", return_value=77):
                runtime = bench_matrix.process_nccl_runtime(1234, proc_root)

        self.assertFalse(runtime["available"])
        self.assertEqual(runtime["mapped_library_count"], 2)
        self.assertIn("found 2", runtime["error"])

    def test_process_nccl_runtime_rejects_unreadable_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            (proc_root / "1234").mkdir()
            with mock.patch.object(bench_matrix.os, "getpgid", return_value=77):
                runtime = bench_matrix.process_nccl_runtime(1234, proc_root)

        self.assertFalse(runtime["available"])
        self.assertEqual(runtime["mapped_library_count"], 0)
        self.assertEqual(len(runtime["map_errors"]), 1)

    def test_process_nccl_runtime_rejects_partial_maps_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            (proc_root / "1234").mkdir()
            (proc_root / "5678").mkdir()
            (proc_root / "1234" / "maps").write_text(
                "7f00-7f10 r-xp 00000000 00:00 0 /wheel/lib/libnccl.so.2\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                bench_matrix.os,
                "getpgid",
                return_value=77,
            ), mock.patch.object(
                bench_matrix,
                "nccl_version_from_library",
            ) as probe:
                runtime = bench_matrix.process_nccl_runtime(1234, proc_root)

        probe.assert_not_called()
        self.assertFalse(runtime["available"])
        self.assertEqual(runtime["mapped_library_count"], 1)
        self.assertEqual(len(runtime["map_errors"]), 1)
        self.assertIn("every server process-group maps file", runtime["error"])

    def test_try_command_records_redacted_error_without_raising(self) -> None:
        with mock.patch.object(bench_matrix.shutil, "which", return_value="/bin/tool"):
            with mock.patch.object(
                bench_matrix,
                "run_capture",
                side_effect=RuntimeError(f"timeout in /{'root'}/venv/bin/tool"),
            ):
                result = bench_matrix.try_command(["tool", "--version"])

        self.assertTrue(result["available"])
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("~/venv/bin/tool", result["error"])

    def test_run_correctness_gate_uses_custom_hf_python_only_for_hf_dump(self) -> None:
        args = SimpleNamespace(
            model_path=Path("models/DeepSeek-V2-Lite"),
            hf_python="/opt/hf-venv/bin/python",
            command_timeout_s=30,
            keep_going=True,
        )
        calls: list[list[str]] = []

        def fake_run_capture(cmd, **_kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def fake_load_json(_path):
            return {"classification": "all_token_text_exact", "warnings": []}

        with tempfile.TemporaryDirectory() as tmp:
            correctness = Path(tmp) / "correctness"
            correctness.mkdir()
            (correctness / "comparison.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(bench_matrix, "run_capture", side_effect=fake_run_capture):
                with mock.patch.object(bench_matrix, "load_json", side_effect=fake_load_json):
                    result = bench_matrix.run_correctness_gate(args, Path(tmp))

        self.assertTrue(result["passed"])
        self.assertEqual(calls[0][0], "/opt/hf-venv/bin/python")
        self.assertEqual(calls[3][0], sys.executable)

    def test_plan_records_custom_hf_python_in_correctness_commands(self) -> None:
        args = SimpleNamespace(
            model_path=Path("models/DeepSeek-V2-Lite"),
            model_id="DeepSeek-V2-Lite",
            out_dir=Path("target/benchmarks/deepseek-v2-lite-vllm-tp2-ep2"),
            port=8000,
            hf_python="/opt/hf-venv/bin/python",
            vllm_cmd="vllm",
            vllm_serve_extra_args=bench_matrix.default_vllm_extra_args(),
            cuda_visible_devices=None,
            input_len=64,
            output_len=64,
            num_prompts=32,
            num_warmups=4,
            concurrency=[1, 4, 8],
            direct_batches=[1, 4, 8],
            repeats=3,
            request_rate="inf",
            temperature=0.0,
            ignore_eos=True,
            noisy_threshold=0.05,
        )

        with mock.patch.object(bench_matrix, "metadata", return_value={"test": True}):
            plan = bench_matrix.plan(args)

        self.assertEqual(
            plan["correctness_commands"][0]["command"][0],
            "/opt/hf-venv/bin/python",
        )
        self.assertEqual(
            plan["correctness_commands"][1]["command"][0],
            bench_matrix.redact_text(sys.executable),
        )

    def test_plan_skips_version_probes(self) -> None:
        args = SimpleNamespace(
            model_path=Path("models/DeepSeek-V2-Lite"),
            model_id="DeepSeek-V2-Lite",
            out_dir=Path("target/benchmarks/deepseek-v2-lite-vllm-tp2-ep2"),
            port=8000,
            hf_python="/opt/hf-venv/bin/python",
            vllm_cmd="vllm",
            vllm_serve_extra_args=bench_matrix.default_vllm_extra_args(),
            cuda_visible_devices=None,
            input_len=64,
            output_len=64,
            num_prompts=32,
            num_warmups=4,
            concurrency=[1, 4, 8],
            direct_batches=[1, 4, 8],
            repeats=3,
            request_rate="inf",
            temperature=0.0,
            ignore_eos=True,
            noisy_threshold=0.05,
        )

        with mock.patch.object(bench_matrix, "try_command") as try_command:
            plan = bench_matrix.plan(args)

        try_command.assert_not_called()
        self.assertTrue(plan["metadata"]["versions"]["probes_skipped"])

    def test_vllm_bench_command_records_workload_metadata(self) -> None:
        args = self.base_args(Path("."))

        cmd = bench_matrix.vllm_bench_command(
            args,
            port=8000,
            num_prompts=32,
            result_dir=Path("results"),
            result_filename="result.json",
            max_concurrency=4,
        )

        metadata_values = [
            cmd[index + 1]
            for index, value in enumerate(cmd)
            if value == "--metadata"
        ]
        self.assertIn(f"{bench_matrix.HTTP_METADATA_PREFIX}input_len=64", metadata_values)
        self.assertIn(f"{bench_matrix.HTTP_METADATA_PREFIX}output_len=64", metadata_values)
        self.assertIn(f"{bench_matrix.HTTP_METADATA_PREFIX}temperature=0.0", metadata_values)
        self.assertIn(f"{bench_matrix.HTTP_METADATA_PREFIX}ignore_eos=True", metadata_values)

    def test_benchmark_server_env_records_rollback_switches(self) -> None:
        self.assertEqual(
            bench_matrix.benchmark_server_env(
                {
                    "PEGAINFER_DSV2_LITE_EP_BACKEND": "nccl",
                    "PEGAINFER_DSV2_LITE_HOST_STAGED_EXPERT_BATCH": "serial",
                    "PEGAINFER_DSV2_LITE_NCCL_EXPERT_BATCH": "grouped",
                    "PEGAINFER_DSV2_LITE_NCCL_ROUTER": "device",
                    "CUDA_VISIBLE_DEVICES": "0,1",
                    "UNRELATED": "ignored",
                }
            ),
            {
                "PEGAINFER_DSV2_LITE_EP_BACKEND": "nccl",
                "PEGAINFER_DSV2_LITE_HOST_STAGED_EXPERT_BATCH": "serial",
                "PEGAINFER_DSV2_LITE_NCCL_EXPERT_BATCH": "grouped",
                "PEGAINFER_DSV2_LITE_NCCL_ROUTER": "device",
                "CUDA_VISIBLE_DEVICES": "0,1",
            },
        )

    def test_wait_for_server_fails_fast_when_process_exits(self) -> None:
        class FakeServer:
            log_path = Path("server.log")

            def poll(self) -> int:
                return 1

            def log_tail(self) -> str:
                return "boom"

        with self.assertRaisesRegex(RuntimeError, "server exited before readiness"):
            bench_matrix.wait_for_server(
                FakeServer(),
                bench_matrix.ENGINES[0],
                9,
                "DeepSeek-V2-Lite",
                60.0,
            )

    def test_parse_direct_artifact_reports_tpot_and_backend_counters(self) -> None:
        parsed = bench_matrix.parse_direct_artifact(
            {
                "config": {"batch_size": 4},
                "timing": {"per_token_decode_stats": {"mean_us": 2000.0}},
                "accuracy": {
                    "token_sha256": "tok",
                    "text_sha256": "txt",
                    "same_prompt_rows_exact": True,
                },
                "gpu_timing": {"sample_count": 7, "failure_count": 0},
                "ep": {"dispatch_calls": 11, "nccl_exchange_calls": 3},
                "cuda_graph_readiness": {"full_decode_capture_ready": False},
            }
        )

        self.assertEqual(parsed["tpot_ms"], 2.0)
        self.assertEqual(parsed["output_tok_s"], 2000.0)
        self.assertEqual(parsed["token_sha256"], "tok")
        self.assertTrue(parsed["same_prompt_rows_exact"])
        self.assertEqual(parsed["gpu_event_samples"], 7)
        self.assertEqual(parsed["ep"]["nccl_exchange_calls"], 3)
        self.assertEqual(
            parsed["backend_counters"],
            {"host_dispatch_calls": 11, "nccl_exchange_calls": 3},
        )

    def test_parse_vllm_bench_artifact_uses_duration_fallback_for_output_rate(self) -> None:
        parsed = bench_matrix.parse_vllm_bench_artifact(
            {
                "num_completed_requests": 24,
                "num_failed_requests": 0,
                "total_output_tokens": 384,
                "duration": 12.0,
                "mean_tpot_ms": 41.0,
                "mean_ttft_ms": 120.0,
                "generated_texts": [f"output-{index}" for index in range(24)],
            }
        )

        self.assertEqual(parsed["completed"], 24)
        self.assertEqual(parsed["failed"], 0)
        self.assertEqual(parsed["output_tok_s"], 32.0)
        self.assertEqual(parsed["mean_tpot_ms"], 41.0)
        self.assertEqual(parsed["mean_ttft_ms"], 120.0)
        self.assertTrue(parsed["passed"])

    def test_parse_vllm_bench_artifact_marks_failed_requests_failed(self) -> None:
        parsed = bench_matrix.parse_vllm_bench_artifact(
            {
                "num_completed_requests": 31,
                "num_failed_requests": 1,
                "num_timeouts": 0,
            }
        )

        self.assertFalse(parsed["passed"])

    def test_parse_vllm_bench_artifact_rejects_empty_payload(self) -> None:
        parsed = bench_matrix.parse_vllm_bench_artifact(
            {},
            bench_matrix.expected_http_workload(self.base_args(Path(".")), 1),
        )

        self.assertFalse(parsed["passed"])
        self.assertIsNone(parsed["completed"])

    def test_parse_vllm_bench_artifact_requires_full_completion_and_contract(self) -> None:
        expected = bench_matrix.expected_http_workload(self.base_args(Path(".")), 1)
        partial = bench_matrix.parse_vllm_bench_artifact(
            self.valid_http_payload(num_completed_requests=31),
            expected,
        )
        wrong_contract = bench_matrix.parse_vllm_bench_artifact(
            self.valid_http_payload(
                **{f"{bench_matrix.HTTP_METADATA_PREFIX}input_len": "32"}
            ),
            expected,
        )

        self.assertFalse(partial["passed"])
        self.assertFalse(partial["full_completion"])
        self.assertEqual(partial["expected_completed"], 32)
        self.assertFalse(wrong_contract["passed"])
        self.assertEqual(wrong_contract["workload_mismatches"], ["input_len"])

    def test_parse_vllm_bench_artifact_rejects_partial_completion_with_matching_outputs(self) -> None:
        expected = bench_matrix.expected_http_workload(self.base_args(Path(".")), 1)
        payload = self.valid_http_payload(
            num_completed_requests=31,
            total_output_tokens=1984,
            generated_texts=[f"output-{index}" for index in range(31)],
        )

        parsed = bench_matrix.parse_vllm_bench_artifact(payload, expected)

        self.assertFalse(parsed["passed"])
        self.assertTrue(parsed["detailed_outputs_valid"])
        self.assertFalse(parsed["full_completion"])
        self.assertEqual(parsed["completed"], 31)
        self.assertEqual(parsed["expected_completed"], 32)

    def test_parse_vllm_bench_artifact_requires_detailed_output_hashes(self) -> None:
        expected = bench_matrix.expected_http_workload(self.base_args(Path(".")), 1)
        payload = self.valid_http_payload()
        payload.pop("generated_texts")

        parsed = bench_matrix.parse_vllm_bench_artifact(payload, expected)

        self.assertFalse(parsed["passed"])
        self.assertEqual(parsed["output_text_count"], 0)
        self.assertIsNone(parsed["output_text_sha256"])

    def test_parse_vllm_bench_artifact_rejects_fabricated_output_coverage(self) -> None:
        expected = bench_matrix.expected_http_workload(self.base_args(Path(".")), 1)
        empty_outputs = bench_matrix.parse_vllm_bench_artifact(
            self.valid_http_payload(generated_texts=[""] * 32),
            expected,
        )
        details = [
            {"response": {"choices": [{"text": f"output-{index}"}]}}
            for index in range(30)
        ]
        details.append(
            {
                "response": {
                    "choices": [
                        {"text": "output-30"},
                        {"text": "output-31"},
                    ]
                }
            }
        )
        multi_choice_coverage = self.valid_http_payload(details=details)
        multi_choice_coverage.pop("generated_texts")
        flattened_outputs = bench_matrix.parse_vllm_bench_artifact(
            multi_choice_coverage,
            expected,
        )

        self.assertFalse(empty_outputs["passed"])
        self.assertFalse(empty_outputs["detailed_outputs_valid"])
        self.assertEqual(empty_outputs["output_text_count"], 32)
        self.assertFalse(flattened_outputs["passed"])
        self.assertFalse(flattened_outputs["detailed_outputs_valid"])
        self.assertEqual(flattened_outputs["output_text_count"], 32)

    def test_summarize_http_rows_excludes_failed_repeats_from_perf_medians(self) -> None:
        rows = bench_matrix.summarize_http_rows(
            [
                {
                    "engine": "pegainfer-host-staged",
                    "cells": [
                        {
                            "concurrency": 1,
                            "passed": True,
                            "completed": 32,
                            "failed": 0,
                            "timeouts": 0,
                            "mean_tpot_ms": 10.0,
                            "mean_ttft_ms": 100.0,
                            "mean_itl_ms": 9.0,
                            "output_tok_s": 100.0,
                            "output_text_sha256": "good",
                        },
                        {
                            "concurrency": 1,
                            "passed": False,
                            "completed": 31,
                            "failed": 0,
                            "timeouts": 0,
                            "mean_tpot_ms": 1.0,
                            "mean_ttft_ms": 1.0,
                            "mean_itl_ms": 1.0,
                            "output_tok_s": 1000.0,
                            "output_text_sha256": "partial",
                        },
                    ],
                }
            ],
            0.05,
        )

        summary = rows[0]["summary_by_concurrency"][0]
        self.assertEqual(summary["repeat_count"], 2)
        self.assertEqual(summary["passed_repeat_count"], 1)
        self.assertEqual(summary["cell_passed"], [True, False])
        self.assertEqual(summary["mean_tpot_ms"]["median"], 10.0)
        self.assertEqual(summary["mean_ttft_ms"]["median"], 100.0)
        self.assertEqual(summary["mean_itl_ms"]["median"], 9.0)
        self.assertEqual(summary["output_tok_s"]["median"], 100.0)

    def test_output_text_hash_handles_openai_and_detail_shapes(self) -> None:
        parsed = bench_matrix.output_text_hash(
            {
                "details": [
                    {"response": {"choices": [{"text": "alpha"}]}},
                    {"generated_text": "beta"},
                ],
            }
        )

        self.assertEqual(parsed["count"], 2)
        self.assertIsNotNone(parsed["sha256"])

    def test_summarize_values_handles_empty_zero_and_noise(self) -> None:
        empty = bench_matrix.summarize_values([], 0.05)
        zero = bench_matrix.summarize_values([0.0, 0.0], 0.05)
        noisy = bench_matrix.summarize_values([10.0, 11.0], 0.05)

        self.assertIsNone(empty["median"])
        self.assertFalse(empty["noisy"])
        self.assertEqual(zero["spread_ratio"], None)
        self.assertFalse(zero["noisy"])
        self.assertEqual(noisy["median"], 10.5)
        self.assertTrue(noisy["noisy"])

    def test_parse_int_list_rejects_empty_and_non_positive_values(self) -> None:
        self.assertEqual(bench_matrix.parse_int_list("1,4,8"), [1, 4, 8])
        with self.assertRaises(ArgumentTypeError):
            bench_matrix.parse_int_list("")
        with self.assertRaises(ArgumentTypeError):
            bench_matrix.parse_int_list("1,0")

    def test_batch_size_from_path_parses_only_positive_batch_files(self) -> None:
        self.assertEqual(bench_matrix.batch_size_from_path(Path("batch8.json")), 8)
        self.assertIsNone(bench_matrix.batch_size_from_path(Path("batch.json")))
        self.assertIsNone(bench_matrix.batch_size_from_path(Path("batch-1.json")))
        self.assertIsNone(bench_matrix.batch_size_from_path(Path("batch0.json")))

    def test_trace_missing_count_prefers_missing_traces_length(self) -> None:
        self.assertEqual(bench_matrix.trace_missing_count({"missing_traces": []}), 0)
        self.assertEqual(
            bench_matrix.trace_missing_count({"missing_traces": ["a", "b"]}),
            2,
        )
        self.assertEqual(bench_matrix.trace_missing_count({"missing_trace_count": 3}), 3)

    def test_correctness_passed_requires_exact_classification_without_warnings(self) -> None:
        self.assertTrue(
            bench_matrix.correctness_passed(
                {"classification": "all_token_text_exact", "warnings": []}
            )
        )
        self.assertFalse(
            bench_matrix.correctness_passed(
                {"classification": "all_token_text_exact", "warnings": ["hash warning"]}
            )
        )
        self.assertFalse(
            bench_matrix.correctness_passed(
                {"classification": "token_mismatch", "warnings": []}
            )
        )
        self.assertFalse(
            bench_matrix.correctness_passed(
                {"classification": "all_token_text_exact"}
            )
        )

    def test_summarize_http_rows_marks_noisy_cells(self) -> None:
        rows = bench_matrix.summarize_http_rows(
            [
                {
                    "engine": "vllm-tp2",
                    "cells": [
                        {
                            "concurrency": 4,
                            "passed": True,
                            "mean_tpot_ms": 40.0,
                            "output_tok_s": 80.0,
                            "completed": 24,
                            "failed": 0,
                        },
                        {
                            "concurrency": 4,
                            "passed": True,
                            "mean_tpot_ms": 60.0,
                            "output_tok_s": 100.0,
                            "completed": 24,
                            "failed": 0,
                        },
                    ],
                }
            ],
            noisy_threshold=0.05,
        )

        row = rows[0]["summary_by_concurrency"][0]
        self.assertEqual(row["concurrency"], 4)
        self.assertTrue(row["noisy"])
        self.assertEqual(row["mean_tpot_ms"]["median"], 50.0)
        self.assertEqual(row["output_tok_s"]["median"], 90.0)

    def test_summarize_existing_rebuilds_summary_from_raw_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct_diagnostic_batch" / "nccl" / "batch1.json"
            direct.parent.mkdir(parents=True)
            direct.write_text(
                json.dumps(
                    {
                        "backend": "nccl",
                        "config": {"batch_size": 1},
                        "timing": {"per_token_decode_stats": {"mean_us": 5000}},
                        "accuracy": {"token_sha256": "tok", "text_sha256": "txt"},
                    }
                ),
                encoding="utf-8",
            )
            http = root / "http_raw" / "vllm-tp2" / "c8" / "r0" / "result.json"
            http.parent.mkdir(parents=True)
            http.write_text(
                json.dumps(
                    self.valid_http_payload(
                        num_completed_requests=32,
                        total_output_tokens=2048,
                        duration=2.0,
                        mean_tpot_ms=40.0,
                        mean_ttft_ms=110.0,
                        max_concurrency=8,
                    )
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=1,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertEqual(summary["kind"], "deepseek_v2_lite_vllm_tp2_ep2_benchmark_matrix")
        self.assertEqual(summary["direct_diagnostic_batch"][0]["backend"], "nccl")
        self.assertEqual(summary["direct_diagnostic_batch"][0]["batch_size"], 1)
        self.assertEqual(summary["http_concurrency_pressure"][0]["engine"], "vllm-tp2")
        self.assertEqual(
            summary["http_concurrency_pressure"][0]["summary_by_concurrency"][0]["output_tok_s"]["median"],
            1024.0,
        )

    def test_summarize_existing_rejects_http_workload_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            http = root / "http_raw" / "pegainfer-host-staged" / "c8" / "r0" / "result.json"
            http.parent.mkdir(parents=True)
            payload = self.valid_http_payload(max_concurrency=8)
            payload[f"{bench_matrix.HTTP_METADATA_PREFIX}input_len"] = "32"
            http.write_text(json.dumps(payload), encoding="utf-8")
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=1,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        row = summary["http_concurrency_pressure"][0]
        self.assertFalse(row["passed"])
        self.assertEqual(row["cells"][0]["workload_mismatches"], ["input_len"])

    def test_summarize_existing_writes_manifest_and_regression_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct_diagnostic_batch" / "host-staged" / "batch1.json"
            direct.parent.mkdir(parents=True)
            direct.write_text(
                json.dumps(
                    {
                        "backend": "host-staged",
                        "config": {"batch_size": 1},
                        "timing": {"per_token_decode_stats": {"mean_us": 2500}},
                        "accuracy": {"token_sha256": "tok", "text_sha256": "txt"},
                    }
                ),
                encoding="utf-8",
            )
            http = root / "http_raw" / "vllm-tp2" / "c1" / "r0" / "result.json"
            http.parent.mkdir(parents=True)
            http.write_text(
                json.dumps(
                    {
                        "num_completed_requests": 32,
                        "num_failed_requests": 0,
                        "total_output_tokens": 2048,
                        "duration": 64.0,
                        "mean_tpot_ms": 40.0,
                        "mean_ttft_ms": 120.0,
                    }
                ),
                encoding="utf-8",
            )

            self.summarize_existing_without_metadata_probe(self.base_args(root))

            manifest = bench_matrix.load_json(root / "artifact_manifest.json")
            regression = bench_matrix.load_json(root / "regression_summary.json")

        self.assertEqual(manifest["kind"], "deepseek_v2_lite_benchmark_artifact_manifest")
        self.assertIn("artifact_bundle_sha256", manifest)
        direct_records = [
            row for row in manifest["artifact_paths"]
            if row["kind"] == "direct_diagnostic_batch"
        ]
        self.assertEqual(direct_records[0]["path"], "direct_diagnostic_batch/host-staged/batch1.json")
        self.assertEqual(direct_records[0]["path_root"], "artifact_bundle")
        self.assertIsNotNone(direct_records[0]["sha256"])
        self.assertEqual(
            regression["comparability"]["claim_marker"],
            "no directional claim",
        )
        self.assertIn("baseline_missing", regression["comparability"]["reasons"])
        self.assertIn("no directional claim", regression["docs_summary"])

    def test_regression_summary_reports_resolved_failed_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_path = root / "summary.json"
            baseline_path = root / "baseline-summary.json"
            current = self.minimal_summary(http_failed=False)
            baseline = self.minimal_summary(http_failed=True)
            current_path.write_text(json.dumps(current), encoding="utf-8")
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

            regression = bench_matrix.build_regression_summary(
                current,
                baseline,
                current_summary_path=current_path,
                baseline_summary_path=baseline_path,
                noisy_threshold=0.05,
            )

        self.assertEqual(
            regression["failed_setup_rows"]["resolved"][0]["key"],
            "http_concurrency_pressure:vllm-tp2",
        )
        self.assertIn("failed_setup_rows_changed", regression["comparability"]["reasons"])
        self.assertTrue(regression["comparability"]["no_directional_claim"])

    def test_regression_summary_marks_preserved_failed_setup_no_directional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_path = root / "summary.json"
            baseline_path = root / "baseline-summary.json"
            current = self.minimal_summary(http_failed=True)
            baseline = self.minimal_summary(http_failed=True)
            current_path.write_text(json.dumps(current), encoding="utf-8")
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

            regression = bench_matrix.build_regression_summary(
                current,
                baseline,
                current_summary_path=current_path,
                baseline_summary_path=baseline_path,
                noisy_threshold=0.05,
            )

        self.assertEqual(
            regression["failed_setup_rows"]["preserved"][0]["key"],
            "http_concurrency_pressure:vllm-tp2",
        )
        self.assertIn("failed_setup_rows_preserved", regression["comparability"]["reasons"])
        self.assertEqual(regression["comparability"]["claim_marker"], "no directional claim")

    def test_regression_summary_ignores_dynamic_gpu_probe_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_path = root / "summary.json"
            baseline_path = root / "baseline-summary.json"
            current = self.minimal_summary()
            baseline = self.minimal_summary()
            current["metadata"]["versions"]["nvidia_smi"]["stdout"] = (
                "NVIDIA GeForce RTX 5090, 580.95.05, 12.0, 63, 2550\n"
                "NVIDIA GeForce RTX 5090, 580.95.05, 12.0, 61, 2490"
            )
            baseline["metadata"]["versions"]["nvidia_smi"]["stdout"] = (
                "NVIDIA GeForce RTX 5090, 580.95.05, 12.0, 42, 2100\n"
                "NVIDIA GeForce RTX 5090, 580.95.05, 12.0, 40, 1980"
            )
            current_path.write_text(json.dumps(current), encoding="utf-8")
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

            regression = bench_matrix.build_regression_summary(
                current,
                baseline,
                current_summary_path=current_path,
                baseline_summary_path=baseline_path,
                noisy_threshold=0.05,
            )

        self.assertNotIn("gpu_probe_changed", regression["comparability"]["reasons"])
        self.assertEqual(regression["comparability"]["claim_marker"], "directional comparison allowed")

    def test_regression_summary_detects_stable_gpu_probe_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_path = root / "summary.json"
            baseline_path = root / "baseline-summary.json"
            current = self.minimal_summary()
            baseline = self.minimal_summary()
            current["metadata"]["versions"]["nvidia_smi"]["stdout"] = (
                "NVIDIA GeForce RTX 5090, 580.95.05, 12.0, 63, 2550"
            )
            baseline["metadata"]["versions"]["nvidia_smi"]["stdout"] = (
                "NVIDIA GeForce RTX 4090, 580.95.05, 8.9, 42, 2100"
            )
            current_path.write_text(json.dumps(current), encoding="utf-8")
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

            regression = bench_matrix.build_regression_summary(
                current,
                baseline,
                current_summary_path=current_path,
                baseline_summary_path=baseline_path,
                noisy_threshold=0.05,
            )

        self.assertIn("gpu_probe_changed", regression["comparability"]["reasons"])
        self.assertEqual(regression["comparability"]["claim_marker"], "no directional claim")

    def test_regression_summary_detects_nccl_version_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_path = root / "summary.json"
            baseline_path = root / "baseline-summary.json"
            current = self.minimal_summary()
            baseline = self.minimal_summary()
            current["metadata"]["versions"]["nccl"]["stdout"] = "2.31.0"
            baseline["metadata"]["versions"]["nccl"]["stdout"] = "2.30.4"
            current_path.write_text(json.dumps(current), encoding="utf-8")
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

            regression = bench_matrix.build_regression_summary(
                current,
                baseline,
                current_summary_path=current_path,
                baseline_summary_path=baseline_path,
                noisy_threshold=0.05,
            )

        self.assertIn("nccl_version_changed", regression["comparability"]["reasons"])
        self.assertEqual(regression["comparability"]["claim_marker"], "no directional claim")

    def test_regression_summary_marks_noisy_http_cell_no_directional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_path = root / "summary.json"
            baseline_path = root / "baseline-summary.json"
            current = self.minimal_summary(noisy=True)
            baseline = self.minimal_summary(noisy=False)
            current_path.write_text(json.dumps(current), encoding="utf-8")
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

            regression = bench_matrix.build_regression_summary(
                current,
                baseline,
                current_summary_path=current_path,
                baseline_summary_path=baseline_path,
                noisy_threshold=0.05,
            )

        self.assertIn("current_noisy_http_cell:vllm-tp2/c1", regression["comparability"]["reasons"])
        self.assertEqual(regression["comparability"]["claim_marker"], "no directional claim")

    def test_regression_summary_marks_missing_benchmark_rows_no_directional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_path = root / "summary.json"
            baseline_path = root / "baseline-summary.json"
            current = self.minimal_summary()
            baseline = self.minimal_summary()
            baseline["direct_diagnostic_batch"].append({
                "claim_bucket": bench_matrix.CLAIM_DIRECT,
                "backend": "nccl",
                "batch_size": 1,
                "passed": True,
                "token_sha256": "nccl-token-hash",
                "text_sha256": "nccl-text-hash",
                "tpot_ms": 1.5,
                "output_tok_s": 2.5,
            })
            baseline["http_concurrency_pressure"].append({
                "engine": "pegainfer-nccl",
                "claim_bucket": bench_matrix.CLAIM_HTTP,
                "passed": True,
                "cells": [],
                "summary_by_concurrency": [
                    {
                        "concurrency": 1,
                        "completed": [32],
                        "failed": [0],
                        "timeouts": [0],
                        "output_text_sha256": ["nccl-hash"],
                        "mean_tpot_ms": {
                            "median": 12.0,
                            "min": 12.0,
                            "max": 12.0,
                            "noisy": False,
                        },
                        "output_tok_s": {
                            "median": 90.0,
                            "min": 90.0,
                            "max": 90.0,
                            "noisy": False,
                        },
                        "noisy": False,
                    }
                ],
            })
            baseline["pegainfer_trace_pass"].append({
                "engine": "pegainfer-nccl",
                "claim_bucket": bench_matrix.CLAIM_HTTP,
                "passed": True,
                "cells": [
                    {
                        "concurrency": 1,
                        "completed": 8,
                        "failed": 0,
                        "timeouts": 0,
                        "missing_trace_count": 0,
                        "trace": {"active_set_size_max": 1, "decode_batch_size_max": 1},
                    }
                ],
            })
            current_path.write_text(json.dumps(current), encoding="utf-8")
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

            regression = bench_matrix.build_regression_summary(
                current,
                baseline,
                current_summary_path=current_path,
                baseline_summary_path=baseline_path,
                noisy_threshold=0.05,
            )

        self.assertEqual(regression["direct_diagnostic_batch"]["missing"], ["nccl/batch1"])
        self.assertEqual(regression["http_concurrency_pressure"]["missing"], ["pegainfer-nccl/c1"])
        self.assertEqual(regression["pegainfer_trace_pass"]["missing"], ["pegainfer-nccl/c1"])
        self.assertIn(
            "direct_diagnostic_batch_missing:nccl/batch1",
            regression["comparability"]["reasons"],
        )
        self.assertIn(
            "http_concurrency_pressure_missing:pegainfer-nccl/c1",
            regression["comparability"]["reasons"],
        )
        self.assertIn(
            "pegainfer_trace_pass_missing:pegainfer-nccl/c1",
            regression["comparability"]["reasons"],
        )
        self.assertEqual(regression["comparability"]["claim_marker"], "no directional claim")

    def test_summarize_existing_marks_warned_correctness_artifact_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            correctness = root / "correctness" / "comparison.json"
            correctness.parent.mkdir(parents=True)
            correctness.write_text(
                json.dumps(
                    {
                        "classification": "all_token_text_exact",
                        "warnings": ["case_0: hash warning"],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertFalse(summary["correctness_gate"]["passed"])
        self.assertEqual(summary["correctness_gate"]["claim_bucket"], "failed_setup")

    def test_summarize_existing_preserves_correctness_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            correctness = root / "correctness" / "comparison.json"
            correctness.parent.mkdir(parents=True)
            correctness.write_text(
                json.dumps({"classification": "all_token_text_exact", "warnings": []}),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "correctness_gate": {
                            "artifacts": {
                                "hf": "correctness/hf.json",
                                "host_staged": "correctness/host-staged.json",
                                "nccl": "correctness/nccl.json",
                            },
                            "commands": [{"label": "hf", "command": ["python", "hf_dump.py"]}],
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        gate = summary["correctness_gate"]
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["artifacts"]["hf"], "correctness/hf.json")
        self.assertEqual(gate["artifacts"]["host_staged"], "correctness/host-staged.json")
        self.assertEqual(gate["artifacts"]["nccl"], "correctness/nccl.json")
        self.assertEqual(gate["commands"][0]["label"], "hf")

    def test_summarize_existing_falls_back_to_path_for_direct_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct_diagnostic_batch" / "host-staged" / "batch8.json"
            direct.parent.mkdir(parents=True)
            direct.write_text(
                json.dumps(
                    {
                        "timing": {"per_token_decode_stats": {"mean_us": 5000}},
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertEqual(summary["direct_diagnostic_batch"][0]["backend"], "host-staged")
        self.assertEqual(summary["direct_diagnostic_batch"][0]["batch_size"], 8)

    def test_summarize_existing_prefers_direct_json_identity_over_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct_diagnostic_batch" / "host-staged" / "batch8.json"
            direct.parent.mkdir(parents=True)
            direct.write_text(
                json.dumps(
                    {
                        "backend": "nccl",
                        "config": {"batch_size": 4},
                        "timing": {"per_token_decode_stats": {"mean_us": 5000}},
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertEqual(summary["direct_diagnostic_batch"][0]["backend"], "nccl")
        self.assertEqual(summary["direct_diagnostic_batch"][0]["batch_size"], 4)
        self.assertTrue(summary["direct_diagnostic_batch"][0]["passed"])

    def test_summarize_existing_marks_invalid_direct_identity_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct_diagnostic_batch" / "host-staged" / "batch8.json"
            direct.parent.mkdir(parents=True)
            direct.write_text(
                json.dumps(
                    {
                        "config": {"batch_size": 0},
                        "timing": {"per_token_decode_stats": {"mean_us": 5000}},
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertEqual(summary["direct_diagnostic_batch"][0]["batch_size"], 0)
        self.assertFalse(summary["direct_diagnostic_batch"][0]["passed"])
        self.assertEqual(summary["direct_diagnostic_batch"][0]["claim_bucket"], "failed_setup")

    def test_summarize_existing_keeps_unresolved_failed_setup_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct_diagnostic_batch" / "host-staged" / "batch1.json"
            direct.parent.mkdir(parents=True)
            direct.write_text(
                json.dumps(
                    {
                        "backend": "host-staged",
                        "config": {"batch_size": 1},
                        "timing": {"per_token_decode_stats": {"mean_us": 5000}},
                    }
                ),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "direct_diagnostic_batch": [
                            {
                                "claim_bucket": "failed_setup",
                                "backend": "nccl",
                                "batch_size": 1,
                                "passed": False,
                                "error": "nccl init failed",
                            }
                        ],
                        "http_concurrency_pressure": [
                            {
                                "claim_bucket": "failed_setup",
                                "engine": "vllm-tp2",
                                "passed": False,
                                "error": "server_start_failed",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        rows_by_backend = {
            row["backend"]: row for row in summary["direct_diagnostic_batch"]
        }
        self.assertTrue(rows_by_backend["host-staged"]["passed"])
        self.assertFalse(rows_by_backend["nccl"]["passed"])
        self.assertEqual(len(summary["direct_diagnostic_batch"]), 2)
        self.assertEqual(summary["http_concurrency_pressure"][0]["engine"], "vllm-tp2")
        self.assertFalse(summary["http_concurrency_pressure"][0]["passed"])

    def test_summarize_existing_replaces_resolved_failed_setup_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct_diagnostic_batch" / "nccl" / "batch1.json"
            direct.parent.mkdir(parents=True)
            direct.write_text(
                json.dumps(
                    {
                        "backend": "nccl",
                        "config": {"batch_size": 1},
                        "timing": {"per_token_decode_stats": {"mean_us": 5000}},
                    }
                ),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "direct_diagnostic_batch": [
                            {
                                "claim_bucket": "failed_setup",
                                "backend": "nccl",
                                "batch_size": 1,
                                "passed": False,
                                "error": "nccl init failed",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertEqual(len(summary["direct_diagnostic_batch"]), 1)
        self.assertEqual(summary["direct_diagnostic_batch"][0]["backend"], "nccl")
        self.assertTrue(summary["direct_diagnostic_batch"][0]["passed"])
        self.assertEqual(summary["direct_diagnostic_batch"][0]["claim_bucket"], "direct_diagnostic_batch")

    def test_summarize_existing_replaces_resolved_http_failed_setup_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            http = root / "http_raw" / "vllm-tp2" / "c1" / "r0" / "result.json"
            http.parent.mkdir(parents=True)
            http.write_text(
                json.dumps(
                    self.valid_http_payload()
                ),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "http_concurrency_pressure": [
                            {
                                "claim_bucket": "failed_setup",
                                "engine": "vllm-tp2",
                                "passed": False,
                                "error": "old startup failure",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=1,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertEqual(len(summary["http_concurrency_pressure"]), 1)
        row = summary["http_concurrency_pressure"][0]
        self.assertEqual(row["engine"], "vllm-tp2")
        self.assertTrue(row["passed"])
        self.assertEqual(row["claim_bucket"], "http_pressure")
        self.assertEqual(row["resolved_failed_setup_rows"][0]["error"], "old startup failure")

    def test_summarize_existing_preserves_http_context_and_engine_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for engine in ("vllm-tp2", "pegainfer-host-staged"):
                http = root / "http_raw" / engine / "c1" / "r0" / "result.json"
                http.parent.mkdir(parents=True)
                http.write_text(
                    json.dumps(
                        {
                            "num_completed_requests": 32,
                            "num_failed_requests": 0,
                            "total_output_tokens": 2048,
                            "duration": 64.0,
                        }
                    ),
                    encoding="utf-8",
                )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "http_concurrency_pressure": [
                            {
                                "engine": "vllm-tp2",
                                "label": "vLLM TP2",
                                "family": "vllm",
                                "server_command": ["vllm", "serve"],
                                "cells": [
                                    {
                                        "concurrency": 1,
                                        "repeat": 0,
                                        "artifact": str(
                                            root / "http_raw" / "vllm-tp2" / "c1" / "r0" / "result.json"
                                        ),
                                        "command": ["vllm", "bench"],
                                    }
                                ],
                            },
                            {
                                "engine": "pegainfer-host-staged",
                                "label": "PegaInfer host-staged",
                                "family": "pegainfer",
                                "server_command": ["cargo", "run"],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=1,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        rows = summary["http_concurrency_pressure"]
        self.assertEqual([row["engine"] for row in rows], ["pegainfer-host-staged", "vllm-tp2"])
        self.assertEqual(rows[0]["label"], "PegaInfer host-staged")
        self.assertEqual(rows[0]["server_command"], ["cargo", "run"])
        self.assertEqual(rows[1]["label"], "vLLM TP2")
        self.assertEqual(rows[1]["cells"][0]["command"], ["vllm", "bench"])

    def test_summarize_existing_does_not_copy_cell_context_by_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            http = root / "http_raw" / "vllm-tp2" / "c1" / "r0" / "result.json"
            http.parent.mkdir(parents=True)
            http.write_text(
                json.dumps(
                    {
                        "num_completed_requests": 32,
                        "num_failed_requests": 0,
                        "total_output_tokens": 2048,
                        "duration": 64.0,
                    }
                ),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "http_concurrency_pressure": [
                            {
                                "engine": "vllm-tp2",
                                "cells": [
                                    {
                                        "concurrency": 1,
                                        "repeat": 0,
                                        "artifact": "renamed/http_raw/vllm-tp2/c1/r0/result.json",
                                        "command": ["stale", "bench"],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=1,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        cell = summary["http_concurrency_pressure"][0]["cells"][0]
        self.assertNotIn("command", cell)

    def test_summarize_existing_marks_empty_http_engine_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "http_raw" / "vllm-tp2").mkdir(parents=True)
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        row = summary["http_concurrency_pressure"][0]
        self.assertEqual(row["engine"], "vllm-tp2")
        self.assertFalse(row["passed"])
        self.assertEqual(row["claim_bucket"], "failed_setup")
        self.assertIn("no HTTP benchmark result artifacts", row["error"])

    def test_summarize_existing_marks_missing_http_cells_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            http = root / "http_raw" / "vllm-tp2" / "c1" / "r0" / "result.json"
            http.parent.mkdir(parents=True)
            http.write_text(
                json.dumps(
                    {
                        "num_completed_requests": 32,
                        "num_failed_requests": 0,
                        "total_output_tokens": 2048,
                        "duration": 64.0,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=2,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        row = summary["http_concurrency_pressure"][0]
        self.assertFalse(row["passed"])
        self.assertEqual(row["claim_bucket"], "failed_setup")
        self.assertEqual(len(row["missing_result_cells"]), 3)

    def test_summarize_existing_preserves_previous_failed_setup_when_still_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "http_raw" / "vllm-tp2").mkdir(parents=True)
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "http_concurrency_pressure": [
                            {
                                "claim_bucket": "failed_setup",
                                "engine": "vllm-tp2",
                                "passed": False,
                                "error": "old startup failure",
                                "server_command": ["vllm", "serve"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        row = summary["http_concurrency_pressure"][0]
        self.assertFalse(row["passed"])
        self.assertIn("no HTTP benchmark result artifacts", row["error"])
        self.assertEqual(row["previous_failed_setup_rows"][0]["error"], "old startup failure")

    def test_summarize_existing_infers_failed_http_rows_from_server_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_key = "HF" + "_" + "TOKEN"
            log = root / "server_logs" / "vllm-tp2.log"
            log.parent.mkdir(parents=True)
            log.write_text(
                f"{token_key}=value_from_log\n"
                "RuntimeError: Engine core initialization failed\n"
                "ValueError: could not determine the shape of object type "
                "'torch.storage.UntypedStorage'\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertEqual(summary["http_concurrency_pressure"][0]["engine"], "vllm-tp2")
        self.assertFalse(summary["http_concurrency_pressure"][0]["passed"])
        self.assertIn("UntypedStorage", summary["http_concurrency_pressure"][0]["error"])
        self.assertIn("UntypedStorage", summary["http_concurrency_pressure"][0]["startup_failure"])
        self.assertIn(f"{token_key}=<redacted>", summary["http_concurrency_pressure"][0]["server_log_tail"])
        self.assertNotIn("value_from_log", summary["http_concurrency_pressure"][0]["server_log_tail"])

    def test_summarize_existing_rebuilds_pegainfer_trace_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "pegainfer_trace" / "pegainfer-host-staged" / "c8.json"
            trace.parent.mkdir(parents=True)
            trace.write_text(
                json.dumps(
                    self.valid_trace_payload(
                        concurrency=8,
                        trace_overrides={
                            "decode_batch_size_max": 5,
                        },
                    )
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        trace_rows = summary["pegainfer_trace_pass"]
        self.assertEqual(trace_rows[0]["engine"], "pegainfer-host-staged")
        self.assertTrue(trace_rows[0]["passed"])
        self.assertEqual(trace_rows[0]["cells"][0]["trace"]["decode_batch_size_max"], 5)
        self.assertEqual(trace_rows[0]["cells"][0]["active_set_size_max"], 8)
        self.assertEqual(trace_rows[0]["cells"][0]["decode_batch_size_max"], 5)
        self.assertEqual(
            trace_rows[0]["cells"][0]["decode_steps"]["batched_request_steps_total"],
            32,
        )
        self.assertEqual(trace_rows[0]["cells"][0]["phase_ms"]["decode_total"]["p50_ms"], 384.0)
        self.assertEqual(trace_rows[0]["cells"][0]["phase_ms"]["scheduled_to_terminal"]["p50_ms"], 420.0)

    def test_summarize_existing_preserves_unresolved_trace_failed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "pegainfer_trace" / "pegainfer-host-staged" / "c1.json"
            trace.parent.mkdir(parents=True)
            trace.write_text(json.dumps(self.valid_trace_payload()), encoding="utf-8")
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "pegainfer_trace_pass": [
                            {
                                "engine": "pegainfer-nccl",
                                "claim_bucket": "failed_setup",
                                "passed": False,
                                "error": "old trace setup failed",
                                "cells": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        rows_by_engine = {row["engine"]: row for row in summary["pegainfer_trace_pass"]}
        self.assertTrue(rows_by_engine["pegainfer-host-staged"]["passed"])
        self.assertFalse(rows_by_engine["pegainfer-nccl"]["passed"])
        self.assertEqual(rows_by_engine["pegainfer-nccl"]["error"], "old trace setup failed")

    def test_trace_cell_errors_fail_closed(self) -> None:
        args = self.base_args(Path("."))
        payload = self.valid_trace_payload()
        valid = {
            "completed": payload["summary"]["completed"],
            "failed": payload["summary"]["failed"],
            "timeouts": payload["summary"]["timeouts"],
            **bench_matrix.trace_summary_for_payload(payload),
        }
        self.assertEqual(bench_matrix.trace_cell_errors(valid, args, 1), [])

        mutations = {
            "missing trace": lambda cell: cell.update(traced_requests=31),
            "missing output hash": lambda cell: cell.update(output_hash_count=31),
            "wrong request count": lambda cell: cell.update(num_requests=8),
            "invalid prompt tokens": lambda cell: cell.update(
                prompt_tokens={"min": 0, "max": 87, "total": 31 * 85, "samples": 31}
            ),
            "truncated output": lambda cell: cell.update(
                completion_tokens={"min": 63, "max": 64, "total": 2047, "samples": 32}
            ),
            "missing phase": lambda cell: cell["phase_ms"].pop("decode_total"),
            "missing decode breakdown": lambda cell: cell["decode_steps"].pop(
                "singleton_request_steps_total"
            ),
            "missing active attribution": lambda cell: cell.update(active_set_size_max=None),
            "missing decode attribution": lambda cell: cell.update(decode_batch_size_max=None),
            "wrong workload": lambda cell: cell["workload"].update(prompt_words=1),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                cell = json.loads(json.dumps(valid))
                mutate(cell)
                self.assertTrue(bench_matrix.trace_cell_errors(cell, args, 1))

        concurrent_payload = self.valid_trace_payload(concurrency=4)
        concurrent = {
            "completed": concurrent_payload["summary"]["completed"],
            "failed": concurrent_payload["summary"]["failed"],
            "timeouts": concurrent_payload["summary"]["timeouts"],
            **bench_matrix.trace_summary_for_payload(concurrent_payload),
        }
        self.assertEqual(bench_matrix.trace_cell_errors(concurrent, args, 4), [])
        concurrent["decode_steps"]["batched_request_steps_total"] = 0
        concurrent["decode_steps"]["singleton_request_steps_total"] = 64
        concurrent["decode_steps"]["request_step_batched_share"] = 0.0
        self.assertIn(
            "no batched request-step evidence",
            bench_matrix.trace_cell_errors(concurrent, args, 4),
        )

    def test_summarize_existing_marks_empty_trace_engine_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pegainfer_trace" / "pegainfer-host-staged").mkdir(parents=True)
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        row = summary["pegainfer_trace_pass"][0]
        self.assertEqual(row["engine"], "pegainfer-host-staged")
        self.assertFalse(row["passed"])
        self.assertEqual(row["claim_bucket"], "failed_setup")
        self.assertIn("no PegaInfer trace result artifacts", row["error"])

    def test_summarize_existing_marks_missing_trace_concurrency_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "pegainfer_trace" / "pegainfer-host-staged" / "c1.json"
            trace.parent.mkdir(parents=True)
            trace.write_text(
                json.dumps(
                    {
                        "summary": {"completed": 8, "failed": 0, "output_tokens_per_s": 20.0},
                        "server_trace": {"decode_batch_size_max": 1, "missing_traces": []},
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        row = summary["pegainfer_trace_pass"][0]
        self.assertFalse(row["passed"])
        self.assertEqual(row["claim_bucket"], "failed_setup")
        self.assertEqual(row["missing_trace_concurrency"], [4])

    def test_classify_server_start_failure_prefers_specific_missing_ninja(self) -> None:
        log = (
            "RuntimeError: Engine core initialization failed\n"
            "FileNotFoundError: [Errno 2] No such file or directory: 'ninja'\n"
        )

        self.assertEqual(
            bench_matrix.classify_server_start_failure(log),
            "server_start_failed: missing ninja",
        )

    def test_classify_server_start_failure_specific_branches(self) -> None:
        cases = [
            ("group_end failed (ncclUnhandledCudaError)", "server_start_failed: ncclUnhandledCudaError"),
            (
                "ValueError: could not determine the shape of object type 'torch.storage.UntypedStorage'",
                "server_start_failed: safetensors UntypedStorage shape inference",
            ),
            (
                "Failed to get device capability: SM 12.x requires CUDA >= 12.9",
                "server_start_failed: FlashInfer SM120 CUDA compatibility",
            ),
            (
                "RuntimeError: FlashInfer requires GPUs with sm75 or higher",
                "server_start_failed: FlashInfer GPU capability detection",
            ),
            (
                "RuntimeError: Engine core initialization failed",
                "server_start_failed: vLLM engine core initialization failed",
            ),
            (
                "Novel launcher failure without traceback",
                "server_start_failed: Novel launcher failure without traceback",
            ),
        ]

        for log, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(bench_matrix.classify_server_start_failure(log), expected)


if __name__ == "__main__":
    unittest.main()
