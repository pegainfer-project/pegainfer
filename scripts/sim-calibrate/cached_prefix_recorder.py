#!/usr/bin/env python3
"""Transparent evidence recorder for vllm-bench prefix-repetition runs.

The recorder is an external HTTP observer.  It forwards the request body and
sampling parameters unchanged, captures request identity plus response usage /
cache evidence, and writes a sealed evidence bundle after the benchmark exits.
It is deliberately separate from the legacy calibration collector.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from cached_prefix_collect import CachedPrefixError, _sha256_json, _read_json


_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _body_hash(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _parse_json_body(body: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _header_value(headers: dict[str, str], *names: str) -> str | None:
    lowered = {key.lower(): value for key, value in headers.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _usage_from_sse(payload: bytes) -> tuple[dict[str, int] | None, str | None, dict[str, Any] | None]:
    usage: dict[str, int] | None = None
    finish_reason: str | None = None
    cache_evidence: dict[str, Any] | None = None
    for raw_event in payload.split(b"\n\n"):
        for raw_line in raw_event.splitlines():
            if not raw_line.startswith(b"data:"):
                continue
            data = raw_line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            try:
                event = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            value = event.get("usage")
            if isinstance(value, dict):
                fields = ("prompt_tokens", "completion_tokens", "total_tokens")
                if all(isinstance(value.get(field), int) and not isinstance(value.get(field), bool) for field in fields):
                    usage = {field: int(value[field]) for field in fields}
                details = value.get("prompt_tokens_details")
                if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
                    cache_evidence = {
                        "source": "usage.prompt_tokens_details.cached_tokens",
                        "cached_tokens": int(details["cached_tokens"]),
                        "cache_hit": int(details["cached_tokens"]) > 0,
                    }
            choices = event.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str):
                        finish_reason = choice["finish_reason"]
    return usage, finish_reason, cache_evidence


def _cache_from_headers(headers: dict[str, str]) -> dict[str, Any] | None:
    cached = _header_value(headers, "cached-tokens", "x-cached-tokens")
    if cached is None:
        return None
    try:
        cached_tokens = int(cached)
    except ValueError:
        return None
    hit = _header_value(headers, "prefix-cache-hit", "x-prefix-cache-hit", "x-cache-hit")
    return {
        "source": "response_headers",
        "cached_tokens": cached_tokens,
        "cache_hit": hit.lower() == "true" if hit is not None else cached_tokens > 0,
    }


def _parse_prefix_cache_metrics(payload: str) -> dict[str, Any] | None:
    """Parse vLLM's cumulative local prefix-cache counters."""

    schemas = (
        ("vllm_v1_canonical", "vllm:prefix_cache_queries", "vllm:prefix_cache_hits"),
        ("legacy_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"),
    )
    totals: dict[str, dict[str, float]] = {}
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.rsplit(None, 1)
        if len(fields) != 2:
            continue
        name = re.sub(r"\{.*\}\Z", "", fields[0])
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if value < 0.0:
            continue
        for schema, query_name, hit_name in schemas:
            if name == query_name:
                totals.setdefault(schema, {}).setdefault("queries_total", 0.0)
                totals[schema]["queries_total"] += value
            elif name == hit_name:
                totals.setdefault(schema, {}).setdefault("hits_total", 0.0)
                totals[schema]["hits_total"] += value
    for schema, query_name, hit_name in schemas:
        values = totals.get(schema, {})
        if "queries_total" in values and "hits_total" in values:
            return {
                "queries_total": values["queries_total"],
                "hits_total": values["hits_total"],
                "metric_schema": schema,
                "metric_names": {"queries_total": query_name, "hits_total": hit_name},
            }
    return None


def _read_prefix_cache_metrics(upstream: str) -> dict[str, Any] | None:
    parsed = urlsplit(upstream)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        return None
    if parsed.scheme == "https":
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            parsed.hostname, parsed.port or 443, timeout=5
        )
    else:
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=5)
    try:
        path = parsed.path.rstrip("/") + "/metrics"
        if not path.startswith("/"):
            path = "/" + path
        connection.request("GET", path, headers={"Accept": "text/plain"})
        response = connection.getresponse()
        if not 200 <= response.status < 300:
            response.read(4096)
            return None
        return _parse_prefix_cache_metrics(response.read().decode("utf-8", errors="replace"))
    except (OSError, socket.timeout):
        return None
    finally:
        connection.close()


@dataclass(frozen=True)
class RecorderConfig:
    upstream: str
    evidence_path: Path
    prefix_tokens: int
    prefix_identity: str
    tokenizer: str | None = None
    warmups: int = 0
    rewrite_prompts_to_token_ids: bool = False


class _RecorderState:
    def __init__(self, config: RecorderConfig) -> None:
        self.config = config
        self.lock = threading.Lock()
        self.idle = threading.Condition(self.lock)
        self.active = 0
        self.records: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.metrics_last: dict[str, Any] | None = None
        self.tokenizer: Any = None
        if config.tokenizer:
            try:
                tokenizer_path = Path(config.tokenizer)
                if config.rewrite_prompts_to_token_ids:
                    # This script directory contains the calibration module
                    # ``profile.py``, which would shadow Python's stdlib
                    # profile module while Transformers imports torch.
                    script_dir = Path(__file__).resolve().parent
                    sys.path = [
                        entry
                        for entry in sys.path
                        if Path(entry or os.getcwd()).resolve() != script_dir
                    ]
                    shadowed_profile = sys.modules.get("profile")
                    if shadowed_profile is not None and getattr(shadowed_profile, "__file__", None):
                        if Path(shadowed_profile.__file__).resolve().parent == script_dir:
                            sys.modules.pop("profile", None)
                    from transformers import AutoTokenizer

                    self.tokenizer = AutoTokenizer.from_pretrained(
                        os.fspath(tokenizer_path), use_fast=True
                    )
                else:
                    from tokenizers import Tokenizer

                    if tokenizer_path.is_dir():
                        tokenizer_path = tokenizer_path / "tokenizer.json"
                    if not tokenizer_path.is_file():
                        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_path}")
                    self.tokenizer = Tokenizer.from_file(os.fspath(tokenizer_path))
            except Exception as exc:  # pragma: no cover - exercised by real model setup
                raise RuntimeError(f"failed to load recorder tokenizer {config.tokenizer}: {exc}") from exc

    def begin_request(self) -> None:
        with self.idle:
            self.active += 1

    def end_request(self) -> None:
        with self.idle:
            self.active -= 1
            self.idle.notify_all()

    def wait_for_idle(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        with self.idle:
            while self.active:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise CachedPrefixError("recorder did not finish forwarding all completion requests")
                self.idle.wait(remaining)

    def _prompt_ids(self, prompt: Any) -> list[int] | None:
        if isinstance(prompt, list) and all(isinstance(value, int) and not isinstance(value, bool) for value in prompt):
            return [int(value) for value in prompt]
        if isinstance(prompt, str) and self.tokenizer is not None:
            encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
            ids = encoded.ids if hasattr(encoded, "ids") else encoded
            if all(isinstance(value, int) for value in ids):
                return [int(value) for value in ids]
        return None

    def record(
        self,
        *,
        headers: dict[str, str],
        body: dict[str, Any] | None,
        status: int,
        response_headers: dict[str, str],
        response_body: bytes,
        metrics_after: dict[str, Any] | None = None,
    ) -> None:
        request_id = _header_value(headers, "x-request-id", "request-id")
        request_body_sha256 = (
            _body_hash(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
            if body is not None
            else None
        )
        request_id_source = "request_header"
        if not request_id:
            # vLLM 0.30's prefix_repetition dataset does not propagate the
            # CLI request-id prefix into HTTP headers.  Serial collection
            # still gives us a stable, collision-resistant request identity.
            with self.lock:
                ordinal = len(self.records)
            request_id = f"bench-ordinal-{ordinal}-{(request_body_sha256 or 'empty')[:16]}"
            request_id_source = "ordinal_body_hash"
        usage, finish_reason, cache_evidence = _usage_from_sse(response_body)
        if cache_evidence is None:
            cache_evidence = _cache_from_headers(response_headers)
        with self.lock:
            metrics_before = self.metrics_last
            self.metrics_last = metrics_after
        if cache_evidence is None and metrics_before is not None and metrics_after is not None:
            queries_delta = metrics_after["queries_total"] - metrics_before["queries_total"]
            hits_delta = metrics_after["hits_total"] - metrics_before["hits_total"]
            if queries_delta >= 0.0 and hits_delta >= 0.0:
                cache_evidence = {
                    "source": f"prometheus.{metrics_after['metric_schema']}.delta",
                    "attribution": "serial_request_delta",
                    "cached_tokens": int(hits_delta),
                    "cache_hit": hits_delta > 0.0,
                    "prefix_cache_hit": hits_delta > 0.0,
                    "prefix_cache_queries_delta": int(queries_delta),
                    "prefix_cache_hits_delta": int(hits_delta),
                }
        if usage is None and isinstance(body, dict):
            prompt = body.get("prompt")
            prompt_len = len(prompt) if isinstance(prompt, list) else None
            output_len = body.get("max_tokens")
            if isinstance(prompt_len, int) and isinstance(output_len, int):
                usage = {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": output_len,
                    "total_tokens": prompt_len + output_len,
                }
        prompt = body.get("prompt") if isinstance(body, dict) else None
        prompt_ids = self._prompt_ids(prompt)
        prefix_hash = None
        if prompt_ids is not None and len(prompt_ids) >= self.config.prefix_tokens:
            prefix_hash = _sha256_json(prompt_ids[: self.config.prefix_tokens])
        record = {
            "ordinal": len(self.records),
            "request_id": request_id,
            "request_id_source": request_id_source,
            "phase": "raw",
            "status": status,
            "input_tokens": usage.get("prompt_tokens") if usage else None,
            "output_tokens": usage.get("completion_tokens") if usage else None,
            "finish_reason": finish_reason,
            "usage": usage,
            "cache_hit_evidence": cache_evidence,
            "shared_prefix_hash": prefix_hash,
            "request_body_sha256": request_body_sha256,
            "response_body_sha256": _body_hash(response_body),
        }
        with self.lock:
            self.records.append(record)

    def record_error(self, message: str) -> None:
        with self.lock:
            self.errors.append(message)

    def finalize(self, cell: dict[str, Any]) -> dict[str, Any]:
        self.wait_for_idle()
        with self.lock:
            raw = list(self.records)
            errors = list(self.errors)
        if errors:
            raise CachedPrefixError(f"recorder observed invalid responses: {errors[:3]}")
        if self.config.warmups < 0 or self.config.warmups > len(raw):
            raise CachedPrefixError(
                f"warmups={self.config.warmups} is incompatible with recorded requests={len(raw)}"
            )
        measured = raw[self.config.warmups :]
        output: list[dict[str, Any]] = []
        for ordinal, item in enumerate(measured):
            record = dict(item)
            record["ordinal"] = ordinal
            record["phase"] = "measured"
            evidence = record.get("cache_hit_evidence")
            if isinstance(evidence, dict):
                evidence = dict(evidence)
                evidence.update(
                    {
                        "expected_cached_tokens": self.config.prefix_tokens,
                        "shared_prefix_id": self.config.prefix_identity,
                        "shared_prefix_hash": record.get("shared_prefix_hash")
                        or _sha256_json(
                            {
                                "shared_prefix_id": self.config.prefix_identity,
                                "cached_tokens": self.config.prefix_tokens,
                            }
                        ),
                        "shared_prefix_hash_source": (
                            "tokenizer_prompt_prefix"
                            if record.get("shared_prefix_hash")
                            else "declared_identity_fallback"
                        ),
                        "duration_includes_uncached_tail": True,
                    }
                )
                record["cache_hit_evidence"] = evidence
            output.append(record)
        return {
            "schema_version": 1,
            "evidence_source": "cached-prefix-transparent-recorder",
            "upstream": self.config.upstream,
            "ordering": "serial_arrival",
            "warmups_excluded": self.config.warmups,
            "workload_digest": _sha256_json({"cell": cell, "completed": len(output)}),
            "requests": output,
        }


class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: _RecorderState
    upstream: tuple[str, str, int]

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _forward(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body_bytes = self.rfile.read(length) if length else b""
        body = _parse_json_body(body_bytes)
        forward_body = body
        forward_body_bytes = body_bytes
        if self.state.config.rewrite_prompts_to_token_ids and isinstance(body, dict):
            prompt = body.get("prompt")
            prompt_ids = self.state._prompt_ids(prompt)
            if isinstance(prompt, str) and prompt_ids is None:
                self.state.record_error("prompt could not be tokenized for token-id rewrite")
            elif isinstance(prompt, str):
                forward_body = dict(body)
                forward_body["prompt"] = prompt_ids
                forward_body_bytes = json.dumps(
                    forward_body, ensure_ascii=True, separators=(",", ":")
                ).encode("utf-8")
        parsed_body = urlsplit(self.state.config.upstream)
        path = self.path
        connection: http.client.HTTPConnection
        if parsed_body.scheme == "https":
            connection = http.client.HTTPSConnection(parsed_body.hostname, parsed_body.port or 443, timeout=60)
        else:
            connection = http.client.HTTPConnection(parsed_body.hostname, parsed_body.port or 80, timeout=60)
        request_headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _HOP_BY_HOP and key.lower() not in {"host", "content-length"}
        }
        request_headers["Host"] = parsed_body.netloc
        request_headers["Content-Length"] = str(len(forward_body_bytes))
        response_body = bytearray()
        track_completion = self.command == "POST" and self.path.endswith("/completions")
        if track_completion:
            self.state.begin_request()
        try:
            connection.request(self.command, path, body=forward_body_bytes, headers=request_headers)
            response = connection.getresponse()
            response_headers = {key: value for key, value in response.getheaders()}
            self.send_response(response.status, response.reason)
            for key, value in response_headers.items():
                if key.lower() in _HOP_BY_HOP or key.lower() in {"content-length", "connection"}:
                    continue
                self.send_header(key, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                response_body.extend(chunk)
                self.wfile.write(f"{len(chunk):x}\r\n".encode("ascii"))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            # The terminating chunk makes the benchmark's response complete;
            # metrics sampling below happens after this boundary and therefore
            # cannot contribute to client-observed TTFT/E2E.
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            if track_completion:
                metrics_after = _read_prefix_cache_metrics(self.state.config.upstream)
                try:
                    self.state.record(
                        headers={key: value for key, value in self.headers.items()},
                        body=forward_body,
                        status=response.status,
                        response_headers=response_headers,
                        response_body=bytes(response_body),
                        metrics_after=metrics_after,
                    )
                except CachedPrefixError as exc:
                    self.state.record_error(str(exc))
        except (OSError, socket.timeout) as exc:
            try:
                self.send_error(502, f"upstream request failed: {exc}")
            except OSError:
                pass
        finally:
            if track_completion:
                self.state.end_request()
            connection.close()

    def do_GET(self) -> None:  # noqa: N802
        self._forward()

    def do_POST(self) -> None:  # noqa: N802
        self._forward()


class RecorderServer:
    def __init__(self, config: RecorderConfig) -> None:
        self.state = _RecorderState(config)
        parsed = urlsplit(config.upstream)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("upstream must be an http(s) URL")
        handler = type("RecorderHandler", (_ProxyHandler,), {})
        handler.state = self.state
        handler.upstream = (parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self) -> "RecorderServer":
        self.state.metrics_last = _read_prefix_cache_metrics(self.state.config.upstream)
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def write_evidence(self, path: Path, cell: dict[str, Any]) -> dict[str, Any]:
        evidence = self.state.finalize(cell)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return evidence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, help="real OpenAI-compatible endpoint")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--prefix-tokens", type=int, required=True)
    parser.add_argument("--prefix-identity", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument(
        "--rewrite-prompts-to-token-ids",
        action="store_true",
        help="rewrite string prompts with the recorder tokenizer before forwarding",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="vllm-bench command; use {proxy_url} in --base-url")
    return parser.parse_args()


def _flag_value(command: list[str], flag: str) -> str | None:
    for index, value in enumerate(command[:-1]):
        if value == flag:
            return command[index + 1]
    return None


def _validate_bench_command(command: list[str]) -> None:
    dataset_name = _flag_value(command, "--dataset-name")
    if dataset_name not in {"prefix_repetition", "custom"}:
        raise CachedPrefixError("recorder requires --dataset-name prefix_repetition or custom")
    if dataset_name == "custom" and _flag_value(command, "--dataset-path") is None:
        raise CachedPrefixError("custom recorder workload requires --dataset-path")
    if _flag_value(command, "--max-concurrency") != "1":
        raise CachedPrefixError(
            "recorder requires explicit --max-concurrency 1; concurrent evidence is not positionally pairable"
        )
    if "--disable-shuffle" not in command:
        raise CachedPrefixError("recorder requires --disable-shuffle for serial workload identity")
    if dataset_name == "prefix_repetition" and _flag_value(command, "--prefix-repetition-num-prefixes") != "1":
        raise CachedPrefixError(
            "recorder requires --prefix-repetition-num-prefixes 1; one cell has one shared-prefix identity"
        )
    if "--save-result" not in command or "--save-detailed" not in command:
        raise CachedPrefixError("recorder requires --save-result and --save-detailed")
    if _flag_value(command, "--num-warmups") != "0":
        raise CachedPrefixError(
            "recorder requires explicit --num-warmups 0; the first dataset request is the cache bootstrap"
        )


def main() -> int:
    args = _parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("recorder requires a vllm-bench command", file=sys.stderr)
        return 2
    if not any("{proxy_url}" in arg for arg in command):
        print("vllm-bench command must contain {proxy_url} in --base-url", file=sys.stderr)
        return 2
    try:
        _validate_bench_command(command)
        if args.warmups < 1:
            raise CachedPrefixError("recorder requires --warmups >= 1 to drop the cache bootstrap request")
        cell = _read_json(args.cell)
        config = RecorderConfig(
            upstream=args.upstream,
            evidence_path=args.evidence,
            prefix_tokens=args.prefix_tokens,
            prefix_identity=args.prefix_identity,
            tokenizer=args.tokenizer,
            warmups=args.warmups,
            rewrite_prompts_to_token_ids=args.rewrite_prompts_to_token_ids,
        )
        with RecorderServer(config) as recorder:
            rendered = [arg.replace("{proxy_url}", recorder.url) for arg in command]
            completed = subprocess.run(rendered, check=False)
            recorder.write_evidence(args.evidence, cell)
        return completed.returncode
    except (OSError, ValueError, CachedPrefixError, RuntimeError) as exc:
        print(f"cached-prefix recorder error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
