#!/usr/bin/env python3
"""Exercise the P/D smoke CLI against local HTTP response fixtures."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "glm52_pd_smoke.py"


def event(text: str = "", finish_reason: str | None = None) -> bytes:
    return (
        "data: "
        + json.dumps({"choices": [{"text": text, "finish_reason": finish_reason}]})
        + "\n\n"
    ).encode()


class PdSmokeTests(unittest.TestCase):
    def run_smoke(self, body: bytes) -> subprocess.CompletedProcess[str]:
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP handler API.
                payload = self.rfile.read(int(self.headers["Content-Length"]))
                requests.append((self.path, json.loads(payload)))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        )
        thread.start()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base-url",
                    f"http://127.0.0.1:{server.server_port}/",
                    "--model",
                    "fixture-model",
                    "--prompt",
                    "fixture prompt",
                    "--max-tokens",
                    "16",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        self.assertEqual(
            requests,
            [
                (
                    "/v1/completions",
                    {
                        "model": "fixture-model",
                        "prompt": "fixture prompt",
                        "max_tokens": 16,
                        "temperature": 0,
                        "stream": True,
                    },
                )
            ],
        )
        return result

    def assert_failed(
        self, result: subprocess.CompletedProcess[str], error: str
    ) -> None:
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout, "", result.stdout)
        self.assertIn(error, result.stderr)

    def test_successful_stop_and_length_completions(self) -> None:
        for reason in ("stop", "length"):
            with self.subTest(reason=reason):
                result = self.run_smoke(
                    b": keepalive\n\n"
                    + event("first ")
                    + event("second")
                    + event(finish_reason=reason)
                    + b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n'
                    + b"data: [DONE]\n\n"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertTrue(report["ok"])
                self.assertEqual(report["text"], "first second")
                self.assertEqual(report["finish_reason"], reason)
                self.assertGreaterEqual(report["ttft_ms"], 0)
                self.assertGreaterEqual(report["total_ms"], report["ttft_ms"])

    def test_text_and_terminal_reason_without_done_fail(self) -> None:
        result = self.run_smoke(event("partial", "length"))
        self.assert_failed(result, "router stream ended before [DONE]")

    def test_text_and_done_without_terminal_reason_fail(self) -> None:
        result = self.run_smoke(event("partial") + b"data: [DONE]\n\n")
        self.assert_failed(result, "without a successful finish_reason")

    def test_unsuccessful_terminal_reason_fails_after_text(self) -> None:
        for reason in ("error", "abort", "unexpected"):
            with self.subTest(reason=reason):
                result = self.run_smoke(
                    event("partial") + event(finish_reason=reason) + b"data: [DONE]\n\n"
                )
                self.assert_failed(result, f"finish_reason={reason!r}")

    def test_error_envelope_fails_after_text(self) -> None:
        result = self.run_smoke(
            event("partial")
            + b'data: {"error":{"message":"handoff failed"}}\n\n'
            + b"data: [DONE]\n\n"
        )
        self.assert_failed(result, "handoff failed")

    def test_malformed_event_fails_after_text(self) -> None:
        result = self.run_smoke(event("partial") + b"data: {broken\n\ndata: [DONE]\n\n")
        self.assert_failed(result, "JSONDecodeError")

    def test_complete_stream_without_text_fails(self) -> None:
        result = self.run_smoke(event(finish_reason="stop") + b"data: [DONE]\n\n")
        self.assert_failed(result, "without emitting text")


if __name__ == "__main__":
    unittest.main()
