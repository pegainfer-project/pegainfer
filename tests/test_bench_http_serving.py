#!/usr/bin/env python3
"""Regression tests for scripts/bench_http_serving.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import unittest
import urllib.parse
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench_http_serving.py"
SCRIPTS_DIR = SCRIPT_PATH.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location("bench_http_serving", SCRIPT_PATH)
assert SPEC and SPEC.loader
bench_http_serving = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_http_serving
SPEC.loader.exec_module(bench_http_serving)


class DoneOnlyHandler(BaseHTTPRequestHandler):
    response_body = b"data: [DONE]\n\n"
    response_chunks: list[bytes] | None = None
    chunk_delay_s = 0.0
    request_bodies: list[dict[str, object]] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b""
        if raw_body:
            self.request_bodies.append(json.loads(raw_body.decode("utf-8")))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if self.response_chunks is None:
            self.wfile.write(self.response_body)
            return
        for chunk in self.response_chunks:
            time.sleep(self.chunk_delay_s)
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
            except OSError:
                return

    def log_message(self, format: str, *args: object) -> None:
        return


class BenchHttpServingTests(unittest.TestCase):
    def setUp(self) -> None:
        DoneOnlyHandler.response_body = b"data: [DONE]\n\n"
        DoneOnlyHandler.response_chunks = None
        DoneOnlyHandler.chunk_delay_s = 0.0
        DoneOnlyHandler.request_bodies = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), DoneOnlyHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.url = urllib.parse.urlparse(f"http://{host}:{port}")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_percentile_uses_r7_linear_interpolation(self) -> None:
        self.assertEqual(
            bench_http_serving.percentile([1.0, 2.0, 100.0, 101.0], 50), 51.0
        )
        self.assertAlmostEqual(
            bench_http_serving.percentile([1.0, 2.0, 100.0, 101.0], 95),
            100.85,
        )

    def test_absolute_timeout_rejects_trickle_stream(self) -> None:
        DoneOnlyHandler.response_chunks = [
            b'data: {"choices":[{"text":"a","finish_reason":null}]}\n\n',
            b'data: {"choices":[{"text":"b","finish_reason":null}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        DoneOnlyHandler.chunk_delay_s = 0.08

        result = bench_http_serving.request_once(
            index=0,
            request_id="req-trickle-timeout",
            url=self.url,
            model="fake-model",
            prompt_words=1,
            prompt="hello",
            max_tokens=2,
            temperature=0.0,
            timeout=0.12,
            ignore_eos=True,
        )

        self.assertFalse(result.ok)
        self.assertTrue(result.timed_out)
        self.assertLess(result.latency_ms, 220.0)

    def test_absolute_timeout_interrupts_byte_trickle_inside_one_line(self) -> None:
        line = b'data: {"choices":[{"text":"slow","finish_reason":null}]}\n\n'
        DoneOnlyHandler.response_chunks = [
            line[index : index + 1] for index in range(len(line))
        ]
        DoneOnlyHandler.chunk_delay_s = 0.01

        result = bench_http_serving.request_once(
            index=0,
            request_id="req-byte-trickle-timeout",
            url=self.url,
            model="fake-model",
            prompt_words=1,
            prompt="hello",
            max_tokens=1,
            temperature=0.0,
            timeout=0.12,
            ignore_eos=True,
        )

        self.assertFalse(result.ok)
        self.assertTrue(result.timed_out)
        self.assertLess(result.latency_ms, 250.0)

    def test_token_timing_requires_chunk_token_parity(self) -> None:
        result = bench_http_serving.RequestResult(
            index=0,
            request_id="req-coalesced",
            prompt_words=1,
            max_tokens=3,
            ok=True,
            status=200,
            error=None,
            timed_out=False,
            start_s=0.0,
            start_wall_s=0.0,
            first_token_s=0.1,
            first_token_wall_s=0.1,
            end_s=0.3,
            end_wall_s=0.3,
            latency_ms=300.0,
            ttft_ms=100.0,
            tpot_ms=None,
            itl_ms=[],
            output_chunks=2,
            output_chars=3,
            output_hash="hash",
            text_prefix="abc",
            stream_chunk_tpot_ms=100.0,
            stream_chunk_itl_ms=[100.0],
            server_trace={"completion_tokens": 3},
        )

        bench_http_serving.finalize_token_timing(result)

        self.assertFalse(result.token_timing_valid)
        self.assertIsNone(result.tpot_ms)
        self.assertEqual(result.itl_ms, [])

    def test_retention_gate_rejects_incomplete_token_timing_coverage(self) -> None:
        report = {
            "contract": {"required_trace_coverage_ratio": 1.0},
            "workload": {"num_requests": 1},
            "summary": {"completed": 1, "failed": 0, "timeouts": 0},
            "server_trace": {
                "coverage_ratio": 1.0,
                "active_set_coverage_ratio": 1.0,
                "decode_batch_coverage_ratio": 1.0,
                "token_timing_coverage_ratio": 0.0,
            },
        }

        gate = bench_http_serving.build_retention_gate(report)

        self.assertFalse(gate["passed"])
        self.assertFalse(gate["trace_coverage_passed"])

    def test_done_only_stream_fails_when_tokens_requested(self) -> None:
        result = bench_http_serving.request_once(
            index=0,
            request_id="req-empty",
            url=self.url,
            model="fake-model",
            prompt_words=1,
            prompt="hello",
            max_tokens=1,
            temperature=0.0,
            timeout=5,
            ignore_eos=True,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, 200)
        self.assertIn("without streamed text chunks", result.error or "")
        self.assertEqual(result.output_chunks, 0)

    def test_text_stream_without_done_marker_fails(self) -> None:
        DoneOnlyHandler.response_body = (
            b'data: {"choices":[{"text":"complete","finish_reason":"length"}]}\n\n'
        )

        result = bench_http_serving.request_once(
            index=0,
            request_id="req-truncated-stream",
            url=self.url,
            model="fake-model",
            prompt_words=1,
            prompt="hello",
            max_tokens=1,
            temperature=0.0,
            timeout=5,
            ignore_eos=True,
        )

        self.assertFalse(result.ok)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.status, 200)
        self.assertIn("ended before [DONE]", result.error or "")
        self.assertEqual(result.output_chunks, 0)

    def test_finish_reason_error_stream_fails_even_after_text_chunk(self) -> None:
        DoneOnlyHandler.response_body = (
            b'data: {"choices":[{"text":"partial","finish_reason":null}]}\n\n'
            b'data: {"choices":[{"text":"","finish_reason":"error"}]}\n\n'
            b"data: [DONE]\n\n"
        )

        result = bench_http_serving.request_once(
            index=0,
            request_id="req-finish-error",
            url=self.url,
            model="fake-model",
            prompt_words=1,
            prompt="hello",
            max_tokens=1,
            temperature=0.0,
            timeout=5,
            ignore_eos=True,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, 200)
        self.assertIn("finish_reason=error", result.error or "")

    def test_error_payload_stream_fails(self) -> None:
        DoneOnlyHandler.response_body = (
            b'data: {"error":{"message":"generation failed"}}\n\ndata: [DONE]\n\n'
        )

        result = bench_http_serving.request_once(
            index=0,
            request_id="req-payload-error",
            url=self.url,
            model="fake-model",
            prompt_words=1,
            prompt="hello",
            max_tokens=1,
            temperature=0.0,
            timeout=5,
            ignore_eos=True,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, 200)
        self.assertIn("SSE error: generation failed", result.error or "")

    def test_mixed_sampling_payload_alternates_greedy_and_sampled_profiles(
        self,
    ) -> None:
        DoneOnlyHandler.response_body = (
            b'data: {"choices":[{"text":"x","finish_reason":null}]}\n\ndata: [DONE]\n\n'
        )
        args = type(
            "Args",
            (),
            {
                "base_url": f"http://{self.url.hostname}:{self.url.port}",
                "model": "fake-model",
                "num_requests": 4,
                "concurrency": 4,
                "warmup": 0,
                "prompt_words": [1],
                "max_tokens": [1],
                "temperature": 0.0,
                "top_k": -1,
                "top_p": 1.0,
                "sampling_mode": "mixed-greedy-sampled",
                "sample_temperature": 0.8,
                "sample_top_k": 40,
                "sample_top_p": 0.95,
                "ignore_eos": True,
                "timeout": 5.0,
            },
        )()

        results, _wall_s = bench_http_serving.run_batch(args, measured=True)
        bodies = sorted(
            DoneOnlyHandler.request_bodies, key=lambda body: str(body["request_id"])
        )

        self.assertTrue(all(result.ok for result in results))
        self.assertEqual(
            [
                (result.sampling_label, result.temperature, result.top_k, result.top_p)
                for result in results
            ],
            [
                ("greedy", 0.0, -1, 1.0),
                ("sampled", 0.8, 40, 0.95),
                ("greedy", 0.0, -1, 1.0),
                ("sampled", 0.8, 40, 0.95),
            ],
        )
        self.assertEqual(
            [(body["temperature"], body["top_k"], body["top_p"]) for body in bodies],
            [(0.0, 0, 1.0), (0.8, 40, 0.95), (0.0, 0, 1.0), (0.8, 40, 0.95)],
        )
        self.assertTrue(all("seed" not in body for body in bodies))

    def test_server_trace_log_is_attached_by_vllm_completion_id_prefix(self) -> None:
        result = bench_http_serving.RequestResult(
            index=0,
            request_id="bench-1",
            prompt_words=16,
            max_tokens=2,
            ok=True,
            status=200,
            error=None,
            timed_out=False,
            start_s=1.0,
            start_wall_s=100.0,
            first_token_s=1.2,
            first_token_wall_s=100.25,
            end_s=1.4,
            end_wall_s=100.4,
            latency_ms=400.0,
            ttft_ms=200.0,
            tpot_ms=20.0,
            itl_ms=[20.0],
            output_chunks=2,
            output_chars=4,
            output_hash="abcd",
            text_prefix="text",
        )
        line = (
            'INFO pegainfer_http_trace {"request_id":"cmpl-bench-1-generated",'
            '"queued_at_unix_s":100.01,"scheduled_at_unix_s":100.03,'
            '"first_token_emit_unix_s":100.20,"prefill_ms":170.0,'
            '"first_decode_ms":28.0}\\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(line, encoding="utf-8")
            traces = bench_http_serving.load_server_traces(path)
            bench_http_serving.attach_server_traces([result], traces)

        self.assertIsNotNone(result.server_trace)
        assert result.server_trace is not None
        self.assertAlmostEqual(
            result.server_trace["admission_queue_ms"], 20.0, places=3
        )
        self.assertAlmostEqual(result.server_trace["stream_flush_ms"], 50.0, places=3)
        self.assertAlmostEqual(
            result.server_trace["frontend_to_queue_ms"], 10.0, places=3
        )

    def test_server_trace_loader_ignores_lines_before_measured_offset(self) -> None:
        stale = (
            'INFO pegainfer_http_trace {"request_id":"cmpl-bench-0-stale",'
            '"queued_at_unix_s":50.0}\n'
        )
        current = (
            'INFO pegainfer_http_trace {"request_id":"cmpl-bench-0-current",'
            '"queued_at_unix_s":100.0}\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(stale, encoding="utf-8")
            offset = bench_http_serving.server_log_offset(path)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(current)
            traces = bench_http_serving.load_server_traces(
                path,
                start_offset=offset,
            )

        self.assertNotIn("cmpl-bench-0-stale", traces)
        self.assertIn("cmpl-bench-0-current", traces)

    def test_run_batch_uses_unique_request_prefix_per_args_instance(self) -> None:
        DoneOnlyHandler.response_body = (
            b'data: {"choices":[{"text":"x","finish_reason":null}]}\n\ndata: [DONE]\n\n'
        )

        def make_args() -> SimpleNamespace:
            return SimpleNamespace(
                base_url=f"http://{self.url.hostname}:{self.url.port}",
                model="fake-model",
                num_requests=1,
                concurrency=1,
                warmup=0,
                prompt_words=[1],
                max_tokens=[1],
                temperature=0.0,
                top_k=-1,
                top_p=1.0,
                sampling_mode="single",
                ignore_eos=True,
                timeout=5.0,
            )

        with mock.patch.object(bench_http_serving.uuid, "uuid4") as uuid4:
            uuid4.side_effect = [
                SimpleNamespace(hex="first"),
                SimpleNamespace(hex="second"),
            ]
            first_results, _ = bench_http_serving.run_batch(
                make_args(),
                measured=True,
            )
            second_results, _ = bench_http_serving.run_batch(
                make_args(),
                measured=True,
            )

        self.assertEqual(
            first_results[0].request_id,
            "pegainfer-bench-first-measured-0",
        )
        self.assertEqual(
            second_results[0].request_id,
            "pegainfer-bench-second-measured-0",
        )

    def test_server_stream_error_log_marks_request_failed(self) -> None:
        result = bench_http_serving.RequestResult(
            index=0,
            request_id="bench-0",
            prompt_words=16,
            max_tokens=16,
            ok=True,
            status=200,
            error=None,
            timed_out=False,
            start_s=1.0,
            start_wall_s=100.0,
            first_token_s=1.2,
            first_token_wall_s=100.25,
            end_s=1.4,
            end_wall_s=100.4,
            latency_ms=400.0,
            ttft_ms=200.0,
            tpot_ms=None,
            itl_ms=[],
            output_chunks=1,
            output_chars=80,
            output_hash="abcd",
            text_prefix="text",
        )
        lines = (
            "ERROR vllm_engine_core_client::client::stream: stream.rs:90 "
            "request failed with an internal error during generation "
            'self.request_id="cmpl-bench-0-generated"\n'
            'INFO pegainfer_http_trace {"request_id":"cmpl-bench-0-generated",'
            '"queued_at_unix_s":100.01,"terminal_unix_s":100.30,'
            '"completion_tokens":1,"finish_reason":"length"}\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(lines, encoding="utf-8")
            traces = bench_http_serving.load_server_traces(path)
            bench_http_serving.attach_server_traces([result], traces)

        self.assertFalse(result.ok)
        self.assertIn("server generation error", result.error or "")
        assert result.server_trace is not None
        self.assertIn("server_error", result.server_trace)

    def test_stale_server_trace_is_not_attached(self) -> None:
        result = bench_http_serving.RequestResult(
            index=0,
            request_id="bench-0",
            prompt_words=16,
            max_tokens=16,
            ok=True,
            status=200,
            error=None,
            timed_out=False,
            start_s=1.0,
            start_wall_s=1000.0,
            first_token_s=1.2,
            first_token_wall_s=1000.2,
            end_s=1.4,
            end_wall_s=1000.4,
            latency_ms=400.0,
            ttft_ms=200.0,
            tpot_ms=None,
            itl_ms=[],
            output_chunks=1,
            output_chars=80,
            output_hash="abcd",
            text_prefix="text",
        )
        stale = {
            "cmpl-bench-0-stale": {
                "request_id": "cmpl-bench-0-stale",
                "queued_at_unix_s": 10.0,
                "terminal_unix_s": 11.0,
            }
        }

        bench_http_serving.attach_server_traces([result], stale)

        self.assertIsNone(result.server_trace)

    def test_server_trace_zero_completion_tokens_marks_request_failed(self) -> None:
        result = bench_http_serving.RequestResult(
            index=0,
            request_id="bench-0",
            prompt_words=16,
            max_tokens=16,
            ok=True,
            status=200,
            error=None,
            timed_out=False,
            start_s=1.0,
            start_wall_s=100.0,
            first_token_s=1.2,
            first_token_wall_s=100.25,
            end_s=1.4,
            end_wall_s=100.4,
            latency_ms=400.0,
            ttft_ms=200.0,
            tpot_ms=None,
            itl_ms=[],
            output_chunks=1,
            output_chars=80,
            output_hash="abcd",
            text_prefix="text",
        )
        line = (
            'INFO pegainfer_http_trace {"request_id":"cmpl-bench-0-generated",'
            '"queued_at_unix_s":100.01,"terminal_unix_s":100.30,'
            '"completion_tokens":0}\\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(line, encoding="utf-8")
            traces = bench_http_serving.load_server_traces(path)
            bench_http_serving.attach_server_traces([result], traces)

        self.assertFalse(result.ok)
        self.assertIn("completion_tokens=0", result.error or "")

    def test_mixed_workload_report_records_input_and_output_tokens(self) -> None:
        results = [
            bench_http_serving.RequestResult(
                index=0,
                request_id="bench-0",
                prompt_words=16,
                max_tokens=4,
                ok=True,
                status=200,
                error=None,
                timed_out=False,
                start_s=0.0,
                start_wall_s=0.0,
                first_token_s=0.1,
                first_token_wall_s=0.1,
                end_s=0.2,
                end_wall_s=0.2,
                latency_ms=200.0,
                ttft_ms=100.0,
                tpot_ms=30.0,
                itl_ms=[30.0, 30.0, 30.0],
                output_chunks=4,
                output_chars=8,
                output_hash="aaaa",
                text_prefix="text",
                sampling_label="greedy",
                temperature=0.0,
                top_k=-1,
                top_p=1.0,
                server_trace={"prompt_tokens": 22, "completion_tokens": 4},
            ),
            bench_http_serving.RequestResult(
                index=1,
                request_id="bench-1",
                prompt_words=128,
                max_tokens=8,
                ok=True,
                status=200,
                error=None,
                timed_out=False,
                start_s=0.0,
                start_wall_s=0.0,
                first_token_s=0.2,
                first_token_wall_s=0.2,
                end_s=0.4,
                end_wall_s=0.4,
                latency_ms=400.0,
                ttft_ms=200.0,
                tpot_ms=25.0,
                itl_ms=[25.0] * 7,
                output_chunks=8,
                output_chars=16,
                output_hash="bbbb",
                text_prefix="more",
                sampling_label="sampled",
                temperature=0.8,
                top_k=40,
                top_p=0.95,
                server_trace={"prompt_tokens": 165, "completion_tokens": 8},
            ),
        ]
        args = type(
            "Args",
            (),
            {
                "base_url": "http://127.0.0.1:8000",
                "model": "fake-model",
                "num_requests": 2,
                "concurrency": 2,
                "warmup": 0,
                "prompt_words": [16, 128],
                "max_tokens": [4, 8],
                "temperature": 0.0,
                "top_k": -1,
                "top_p": 1.0,
                "sampling_mode": "mixed-greedy-sampled",
                "sample_temperature": 0.8,
                "sample_top_k": 40,
                "sample_top_p": 0.95,
                "ignore_eos": True,
                "timeout": 5.0,
            },
        )()
        report = bench_http_serving.build_report(args, results, wall_s=2.0)

        self.assertEqual(report["summary"]["input_tokens_total"], 187)
        self.assertEqual(report["summary"]["output_tokens_total"], 12)
        self.assertAlmostEqual(report["summary"]["input_tokens_per_s"], 93.5)
        self.assertAlmostEqual(report["summary"]["output_tokens_per_s"], 6.0)
        self.assertEqual(
            report["workload"]["mixed_shapes"],
            {
                "prompt_words=16,max_tokens=4": 1,
                "prompt_words=128,max_tokens=8": 1,
            },
        )
        self.assertEqual(report["workload"]["sampling_mode"], "mixed-greedy-sampled")
        self.assertEqual(
            report["workload"]["sampling_counts"], {"greedy": 1, "sampled": 1}
        )
        self.assertEqual(
            report["summary"]["completed_sampling_counts"], {"greedy": 1, "sampled": 1}
        )
        self.assertEqual(report["summary"]["failed_sampling_counts"], {})
        self.assertEqual(
            report["workload"]["sampling_profiles"]["greedy"]["temperature"], 0.0
        )
        self.assertEqual(
            report["workload"]["sampling_profiles"]["sampled"]["top_k"], 40
        )
        self.assertEqual(report["requests"][0]["sampling_label"], "greedy")
        self.assertEqual(report["requests"][1]["temperature"], 0.8)

    def test_output_token_throughput_is_unavailable_without_server_trace(self) -> None:
        result = bench_http_serving.RequestResult(
            index=0,
            request_id="bench-0",
            prompt_words=16,
            max_tokens=16,
            ok=True,
            status=200,
            error=None,
            timed_out=False,
            start_s=0.0,
            start_wall_s=0.0,
            first_token_s=0.1,
            first_token_wall_s=0.1,
            end_s=0.2,
            end_wall_s=0.2,
            latency_ms=200.0,
            ttft_ms=100.0,
            tpot_ms=None,
            itl_ms=[],
            output_chunks=1,
            output_chars=80,
            output_hash="aaaa",
            text_prefix="text",
            server_trace=None,
        )
        args = type(
            "Args",
            (),
            {
                "base_url": "http://127.0.0.1:8000",
                "model": "fake-model",
                "num_requests": 1,
                "concurrency": 1,
                "warmup": 0,
                "prompt_words": [16],
                "max_tokens": [16],
                "temperature": 0.0,
                "ignore_eos": True,
                "timeout": 5.0,
            },
        )()
        report = bench_http_serving.build_report(args, [result], wall_s=2.0)

        self.assertIsNone(report["summary"]["output_tokens_total"])
        self.assertIsNone(report["summary"]["output_tokens_per_s"])
        self.assertEqual(report["summary"]["output_token_count_source"], "unavailable")
        self.assertEqual(report["summary"]["output_token_count_coverage_ratio"], 0.0)

    def test_server_trace_summary_includes_decode_step_breakdown(self) -> None:
        first = SimpleNamespace(
            request_id="bench-0",
            server_trace={
                "active_set_size": 2,
                "decode_batch_size_max": 2,
                "queue_wait_ms": 10.0,
                "prefill_ms": 20.0,
                "first_decode_ms": 4.0,
                "decode_mean_ms": 5.0,
                "decode_total_ms": 10.0,
                "scheduled_to_first_token_ms": 24.0,
                "scheduled_to_terminal_ms": 60.0,
                "decode_step_count": 2,
                "batch_decode_steps": 1,
            },
        )
        second = SimpleNamespace(
            request_id="bench-1",
            server_trace={
                "active_set_size": 4,
                "decode_batch_size_max": 4,
                "queue_wait_ms": 30.0,
                "prefill_ms": 40.0,
                "first_decode_ms": 6.0,
                "decode_mean_ms": 7.0,
                "decode_total_ms": 21.0,
                "scheduled_to_first_token_ms": 46.0,
                "scheduled_to_terminal_ms": 90.0,
                "decode_step_count": 3,
                "batch_decode_steps": 3,
            },
        )

        summary = bench_http_serving.summarize_trace_ms([first, second])

        self.assertEqual(summary["active_set_size_max"], 4)
        self.assertEqual(summary["decode_batch_size_max"], 4)
        self.assertEqual(summary["phases_ms"]["queue_wait_ms"]["samples"], 2)
        self.assertEqual(summary["phases_ms"]["decode_total_ms"]["max_ms"], 21.0)
        self.assertEqual(summary["decode_steps"]["per_request"]["min"], 2)
        self.assertEqual(summary["decode_steps"]["per_request"]["max"], 3)
        self.assertEqual(summary["decode_steps"]["per_request"]["total"], 5)
        self.assertEqual(summary["decode_steps"]["batched_request_steps_total"], 4)
        self.assertEqual(summary["decode_steps"]["singleton_request_steps_total"], 1)
        self.assertAlmostEqual(
            summary["decode_steps"]["request_step_batched_share"], 0.8
        )

    def test_server_trace_summary_marks_partial_decode_breakdown_unknown(self) -> None:
        legacy = SimpleNamespace(
            request_id="legacy",
            server_trace={"batch_decode_steps": 4},
        )

        summary = bench_http_serving.summarize_trace_ms([legacy])

        self.assertIsNone(summary["decode_steps"]["batched_request_steps_total"])
        self.assertIsNone(summary["decode_steps"]["singleton_request_steps_total"])
        self.assertIsNone(summary["decode_steps"]["request_step_batched_share"])

    def test_server_trace_summary_infers_singleton_steps_from_total(self) -> None:
        trace = SimpleNamespace(
            request_id="with-total",
            server_trace={
                "decode_step_count": 5,
                "batch_decode_steps": 3,
            },
        )

        summary = bench_http_serving.summarize_trace_ms([trace])

        self.assertEqual(summary["decode_steps"]["batched_request_steps_total"], 3)
        self.assertEqual(summary["decode_steps"]["singleton_request_steps_total"], 2)
        self.assertAlmostEqual(
            summary["decode_steps"]["request_step_batched_share"], 0.6
        )

    def test_contract_report_records_hashes_and_full_trace_gate(self) -> None:
        result = bench_http_serving.RequestResult(
            index=0,
            request_id="bench-0",
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
            end_s=0.4,
            end_wall_s=0.4,
            latency_ms=400.0,
            ttft_ms=100.0,
            tpot_ms=5.0,
            itl_ms=[5.0, 6.0],
            output_chunks=64,
            output_chars=128,
            output_hash="hash-a",
            text_prefix="text",
            server_trace={
                "queued_at_unix_s": 0.0,
                "prompt_tokens": 64,
                "completion_tokens": 64,
                "active_set_size": 1,
                "decode_batch_size_max": 1,
                "decode_step_count": 63,
                "batch_decode_steps": 0,
            },
        )
        args = SimpleNamespace(
            base_url="http://127.0.0.1:8000",
            model="DeepSeek-V2-Lite",
            backend="host-staged",
            contract_name="dsv2-lite-short-decode-heavy",
            contract_description="fixed short contract",
            claim_boundary="HTTP SLO only",
            required_trace_coverage=1.0,
            commit="abcdef123456",
            model_path="models/DeepSeek-V2-Lite",
            server_command="pegainfer --model-path models/DeepSeek-V2-Lite",
            num_requests=1,
            concurrency=1,
            warmup=0,
            prompt_words=[64],
            max_tokens=[64],
            temperature=0.0,
            top_k=-1,
            top_p=1.0,
            sampling_mode="single",
            sample_temperature=0.8,
            sample_top_k=40,
            sample_top_p=0.95,
            ignore_eos=True,
            timeout=240.0,
        )

        report = bench_http_serving.build_report(args, [result], wall_s=2.0)

        self.assertEqual(report["report_intent"], "http_serving_slo")
        self.assertEqual(report["contract"]["backend"], "host-staged")
        self.assertEqual(report["contract"]["description"], "fixed short contract")
        self.assertEqual(report["summary"]["output_hash_distribution"], {"hash-a": 1})
        self.assertEqual(report["server_trace"]["coverage_ratio"], 1.0)
        self.assertEqual(report["server_trace"]["active_set_coverage_ratio"], 1.0)
        self.assertEqual(report["server_trace"]["decode_batch_coverage_ratio"], 1.0)
        self.assertTrue(report["retention_gate"]["passed"])
        self.assertFalse(report["latency_budget"]["configured"])
        self.assertIsNone(report["latency_budget"]["passed"])

    def test_server_error_record_is_not_full_trace_coverage(self) -> None:
        result = SimpleNamespace(
            request_id="bench-error",
            server_trace={"server_error": "request failed"},
        )

        trace = bench_http_serving.summarize_trace_ms([result])

        self.assertEqual(trace["attached_server_records"], 1)
        self.assertEqual(trace["server_error_records"], 1)
        self.assertEqual(trace["traced_requests"], 0)
        self.assertEqual(trace["coverage_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
