#!/usr/bin/env python3
"""Artifact behavior tests using a local streaming OAI endpoint."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from artifacts import ArtifactStore  # noqa: E402
from manifest import build_manifest  # noqa: E402
from paired import collect_paired_cell  # noqa: E402


class _StreamingHandler(BaseHTTPRequestHandler):
    models = ["target-model", "zero-model", "replay-model"]

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._json(200, {"data": [{"id": model} for model in self.models]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/completions":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        prompt_tokens = len(body["prompt"])
        output_tokens = int(body["max_tokens"])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("x-prefix-cache-hit", "true")
        self.end_headers()
        for _index in range(output_tokens):
            self.wfile.write(
                b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}\n\n'
            )
            self.wfile.flush()
            time.sleep(0.001)
        self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n')
        usage: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "prompt_tokens_details": {"cached_tokens": prompt_tokens},
        }
        self.wfile.write(
            f"data: {json.dumps({'choices': [], 'usage': usage}, separators=(',', ':'))}\n\n".encode()
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class StreamingEndpoint:
    def __init__(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StreamingHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self) -> "StreamingEndpoint":
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def make_manifest(endpoint: str) -> dict[str, Any]:
    digest = "a" * 64
    cell = {
        "name": "steady_decode",
        "kind": "steady_decode",
        "split": "train",
        "input_tokens": 4,
        "output_tokens": 3,
        "concurrency": 1,
        "prefix_tokens": 0,
        "warmup_repeats": 0,
        "repeats": 2,
        "ignore_eos": True,
    }
    target = {
        "endpoint": endpoint,
        "model": "target-model",
        "engine": "target-engine",
        "engine_version": "1",
        "model_revision": "target-revision",
        "model_config_sha256": digest,
    }
    variants = {}
    for name, model in (("zero_cost", "zero-model"), ("replay", "replay-model")):
        variants[name] = {
            "identity": {**target, "endpoint": endpoint, "model": model, "engine": name},
            "runtime": {"profile": name},
            "server_flags": [f"--{name}"],
            "profile_sha256": digest,
        }
    return build_manifest(
        {
            "target": target,
            "runtime": {"scheduler": {"max_model_len": 128}},
            "client": {
                "name": "artifact-test-client",
                "version": "1",
                "revision": "client-revision",
                "seed": 7,
                "argv": ["artifact-test-client"],
                "server_flags": ["--target"],
            },
            "workload": {"cells": [cell]},
            "variants": variants,
            "provenance": {
                "collector_revision": "collector-revision",
                "collection_command": "python paired.py",
                "artifact_uri": "file:///tmp/artifacts",
                "source_digest": "source-digest",
            },
        },
        created_at="2026-09-22T00:00:00+00:00",
    )


class ArtifactTests(unittest.TestCase):
    def test_streaming_pairing_writes_independent_percentile_artifacts(self) -> None:
        with StreamingEndpoint() as endpoint, tempfile.TemporaryDirectory() as directory:
            manifest = make_manifest(endpoint.url)
            store = ArtifactStore(Path(directory), manifest["campaign_id"])
            result = collect_paired_cell(
                manifest,
                manifest["workload"]["cells"][0],
                store,
                collector_argv=["paired.py"],
                timeout=5,
            )
            self.assertTrue(result["paired"])
            self.assertEqual(result["status"], "sealed")
            self.assertEqual(set(result["variants"]), {"target", "zero_cost", "replay"})
            for entry in result["variants"].values():
                self.assertEqual(entry["status"], "sealed")
                self.assertEqual(len(entry["attempts"]), 1)
                record = json.loads((Path(directory) / entry["result_path"]).read_text(encoding="utf-8"))
                percentiles = record["result"]["summary"]["percentiles_ms"]
                self.assertIsNotNone(percentiles["ttft_ms"]["p50"])
                self.assertIsNotNone(percentiles["e2e_ms"]["p95"])
                self.assertNotIn("delta", record["result"])
                evidence = record["result"]["measurements"][0]["cache_hit_evidence"]
                self.assertEqual(evidence["cached_tokens"], 4)
                self.assertIn("headers", evidence)

if __name__ == "__main__":
    unittest.main()
