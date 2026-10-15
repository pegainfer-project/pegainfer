#!/usr/bin/env python3
"""Tests for the transparent cached-prefix evidence recorder."""

from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from cached_prefix_recorder import (  # noqa: E402
    CachedPrefixError,
    RecorderConfig,
    RecorderServer,
    _RecorderState,
    _parse_prefix_cache_metrics,
    _validate_bench_command,
)


class _UpstreamHandler(BaseHTTPRequestHandler):
    queries = 0
    hits = 0

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        prompt = body["prompt"]
        output = int(body["max_tokens"])
        include_cache_evidence = self.headers.get("x-no-cache-evidence") != "1"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        if include_cache_evidence:
            self.send_header("x-prefix-cache-hit", "true")
        self.end_headers()
        for index in range(output):
            event = {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
        usage: dict[str, Any] = {
            "prompt_tokens": 4,
            "completion_tokens": output,
            "total_tokens": 4 + output,
        }
        if include_cache_evidence:
            usage["prompt_tokens_details"] = {"cached_tokens": 4}
        event = {"choices": [{"delta": {}, "finish_reason": "length"}], "usage": usage}
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        type(self).queries += 1
        type(self).hits += 4

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        payload = (
            "# HELP vllm_prefix_cache_queries Prefix cache queries\n"
            f"vllm:prefix_cache_queries{{engine=\"0\"}} {type(self).queries}\n"
            f"vllm:prefix_cache_hits{{engine=\"0\"}} {type(self).hits}\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _Upstream:
    def __init__(self) -> None:
        _UpstreamHandler.queries = 0
        _UpstreamHandler.hits = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self) -> "_Upstream":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class CachedPrefixRecorderTests(unittest.TestCase):
    def test_formal_runner_requires_detailed_serial_prefix_workload(self) -> None:
        command = [
            ".venv/bin/vllm-bench",
            "--dataset-name",
            "prefix_repetition",
            "--max-concurrency",
            "1",
            "--prefix-repetition-num-prefixes",
            "1",
            "--save-result",
            "--save-detailed",
            "--disable-shuffle",
            "--num-warmups",
            "0",
        ]
        _validate_bench_command(command)

    def test_proxy_preserves_request_and_records_usage_and_header_evidence(self) -> None:
        with _Upstream() as upstream:
            config = RecorderConfig(
                upstream=upstream.url,
                evidence_path=Path("/tmp/unused-evidence.json"),
                prefix_tokens=4,
                prefix_identity="prefix-a",
            )
            with RecorderServer(config) as recorder:
                import urllib.request

                request = urllib.request.Request(
                    recorder.url + "/v1/completions",
                    data=json.dumps({"prompt": [1, 2, 3, 4], "max_tokens": 3}).encode(),
                    headers={"Content-Type": "application/json", "x-request-id": "req-0"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    response.read()
                evidence = recorder.state.finalize(
                    {
                        "name": "cached",
                        "prefix_tokens": 4,
                        "new_tokens": 1,
                    }
                )
            self.assertEqual(len(evidence["requests"]), 1)
            record = evidence["requests"][0]
            self.assertEqual(record["request_id"], "req-0")
            self.assertEqual(record["usage"]["prompt_tokens"], 4)
            self.assertEqual(record["cache_hit_evidence"]["cached_tokens"], 4)
            self.assertEqual(record["cache_hit_evidence"]["shared_prefix_id"], "prefix-a")
            self.assertTrue(record["shared_prefix_hash"])

    def test_proxy_uses_post_response_metrics_without_usage_cache_fields(self) -> None:
        with _Upstream() as upstream:
            config = RecorderConfig(
                upstream=upstream.url,
                evidence_path=Path("/tmp/unused-evidence.json"),
                prefix_tokens=4,
                prefix_identity="prefix-a",
            )
            with RecorderServer(config) as recorder:
                import urllib.request

                request = urllib.request.Request(
                    recorder.url + "/v1/completions",
                    data=json.dumps({"prompt": [1, 2, 3, 4], "max_tokens": 3}).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "x-request-id": "req-metrics",
                        "x-no-cache-evidence": "1",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    response.read()
                evidence = recorder.state.finalize(
                    {"name": "cached", "prefix_tokens": 4, "new_tokens": 1}
                )
            record = evidence["requests"][0]
            self.assertEqual(record["cache_hit_evidence"]["source"], "prometheus.vllm_v1_canonical.delta")
            self.assertEqual(record["cache_hit_evidence"]["cached_tokens"], 4)

    def test_metrics_parser_sums_engine_labels(self) -> None:
        parsed = _parse_prefix_cache_metrics(
            "vllm:prefix_cache_queries{engine=\"0\"} 2\n"
            "vllm:prefix_cache_queries{engine=\"1\"} 3\n"
            "vllm:prefix_cache_hits{engine=\"0\"} 1\n"
            "vllm:prefix_cache_hits{engine=\"1\"} 4\n"
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["queries_total"], 5.0)
        self.assertEqual(parsed["hits_total"], 5.0)


if __name__ == "__main__":
    unittest.main()
