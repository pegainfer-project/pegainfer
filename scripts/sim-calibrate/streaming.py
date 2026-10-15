#!/usr/bin/env python3
"""Streaming OpenAI completion capture with explicit timing semantics."""

from __future__ import annotations

import hashlib
import http.client
import json
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Any

from probe import Endpoint


class StreamError(RuntimeError):
    """Raised when a streaming response is not valid benchmark evidence."""

    def __init__(self, message: str, *, timed_out: bool = False) -> None:
        super().__init__(message)
        self.timed_out = timed_out


@dataclass(frozen=True)
class StreamCapture:
    status: int
    headers: dict[str, str]
    started_at_unix_s: float
    first_event_at_unix_s: float | None
    finished_at_unix_s: float
    elapsed_ms: float
    ttft_ms: float | None
    inter_event_ms: list[float]
    event_count: int
    finish_reason: str | None
    usage: dict[str, int] | None
    token_timing_valid: bool | None
    timing_semantics: str
    cache_hit_evidence: dict[str, Any] | None
    body_sha256: str


def _parse_prefix_cache_metrics(payload: str) -> dict[str, Any]:
    """Parse one complete supported vLLM prefix-cache counter schema."""

    schemas = (
        (
            "vllm_v1_canonical",
            {
                "queries_total": "vllm:prefix_cache_queries",
                "hits_total": "vllm:prefix_cache_hits",
            },
        ),
        (
            "legacy_total",
            {
                "queries_total": "vllm:prefix_cache_queries_total",
                "hits_total": "vllm:prefix_cache_hits_total",
            },
        ),
    )
    parsed: dict[str, dict[str, float]] = {}
    available: set[str] = set()
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.rsplit(None, 1)
        if len(fields) != 2:
            continue
        name = fields[0].split("{", 1)[0]
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if value < 0.0:
            continue
        available.add(name)
        for schema_name, metric_names in schemas:
            for output_name, metric_name in metric_names.items():
                if name == metric_name:
                    parsed.setdefault(schema_name, {}).setdefault(output_name, 0.0)
                    parsed[schema_name][output_name] += value

    for schema_name, metric_names in schemas:
        if all(metric_name in available for metric_name in metric_names.values()):
            totals = parsed[schema_name]
            return {
                "queries_total": totals["queries_total"],
                "hits_total": totals["hits_total"],
                "metric_schema": schema_name,
                "metric_names": metric_names,
            }
    expected = ", ".join(
        f"{names['queries_total']}/{names['hits_total']}" for _, names in schemas
    )
    raise StreamError(f"unsupported prefix-cache metrics schema; expected {expected}")


def prefix_cache_metrics(endpoint: Endpoint, *, timeout: float) -> dict[str, Any] | None:
    """Read vLLM's cumulative local prefix-cache counters for one delta sample.

    vLLM 0.30 exposes cache accounting through ``/metrics`` rather than the
    OpenAI completion usage object. Counters are summed across labels so this
    remains valid for a single isolated endpoint with one or more engine labels.
    """

    conn = _connection(endpoint, timeout)
    try:
        conn.request("GET", endpoint.path("/metrics"), headers={"Accept": "text/plain"})
        response = conn.getresponse()
        if not 200 <= response.status < 300:
            response.read(4096)
            return None
        payload = response.read().decode("utf-8", errors="replace")
    except (OSError, socket.timeout):
        return None
    finally:
        conn.close()

    return _parse_prefix_cache_metrics(payload)


def _connection(endpoint: Endpoint, timeout: float) -> http.client.HTTPConnection:
    if endpoint.scheme == "https":
        return http.client.HTTPSConnection(
            endpoint.host,
            endpoint.port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
    return http.client.HTTPConnection(endpoint.host, endpoint.port, timeout=timeout)


def _usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    if any(isinstance(value.get(field), bool) or not isinstance(value.get(field), int) for field in fields):
        return None
    return {field: int(value[field]) for field in fields}


def _cache_evidence(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    details = value.get("prompt_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
        return {
            "source": "usage.prompt_tokens_details.cached_tokens",
            "cached_tokens": details["cached_tokens"],
            "cache_hit": details["cached_tokens"] > 0,
        }
    for key in ("cached_tokens", "cache_hit", "prefix_cache_hit"):
        if key in value and isinstance(value[key], (bool, int, str)):
            return {"source": f"usage.{key}", key: value[key]}
    return None


def stream_completion(
    endpoint: Endpoint,
    body: dict[str, Any],
    *,
    timeout: float,
) -> StreamCapture:
    """Capture SSE timings without retaining response text or token IDs."""

    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    started_perf = time.perf_counter()
    deadline_perf = started_perf + timeout
    started_wall = time.time()
    conn = _connection(endpoint, timeout)
    digest = hashlib.sha256()
    first_event_wall: float | None = None
    previous_event_perf: float | None = None
    inter_event_ms: list[float] = []
    event_count = 0
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    cache_hit_evidence: dict[str, Any] | None = None
    done_count = 0
    try:
        conn.request(
            "POST",
            endpoint.path("/v1/completions"),
            body=encoded,
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
            },
        )
        response = conn.getresponse()
        headers = {key.lower(): value for key, value in response.getheaders()}
        if not 200 <= response.status < 300:
            error_body = response.read(4096)
            digest.update(error_body)
            raise StreamError(
                f"HTTP {response.status}; body_sha256={digest.hexdigest()}"
            )
        while True:
            remaining = deadline_perf - time.perf_counter()
            if remaining <= 0.0:
                raise StreamError("stream request timed out", timed_out=True)
            if conn.sock is not None:
                conn.sock.settimeout(remaining)
            raw = response.readline()
            if not raw:
                break
            digest.update(raw)
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                done_count += 1
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                raise StreamError("stream emitted invalid JSON") from exc
            if not isinstance(payload, dict):
                raise StreamError("stream event must be a JSON object")
            if isinstance(payload.get("error"), (str, dict)):
                raise StreamError("stream emitted an error event")
            event_usage = _usage(payload.get("usage"))
            if event_usage is not None:
                usage = event_usage
            cache_hit_evidence = _cache_evidence(payload.get("usage")) or cache_hit_evidence
            choices = payload.get("choices")
            if not isinstance(choices, list):
                raise StreamError("stream event choices must be a list")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise StreamError("stream event choice must be an object")
                candidate_finish = choice.get("finish_reason")
                if isinstance(candidate_finish, str) and candidate_finish:
                    finish_reason = candidate_finish
                delta = choice.get("delta", choice)
                # Count only events carrying actual generated content. A role-only
                # event and the final usage event must not affect timing counts.
                if isinstance(delta, dict) and isinstance(delta.get("content", delta.get("text")), str):
                    now_perf = time.perf_counter()
                    now_wall = time.time()
                    event_count += 1
                    if first_event_wall is None:
                        first_event_wall = now_wall
                    if previous_event_perf is not None:
                        inter_event_ms.append((now_perf - previous_event_perf) * 1000.0)
                    previous_event_perf = now_perf
        if done_count != 1:
            raise StreamError(f"stream must contain exactly one [DONE], got {done_count}")
        if finish_reason is None:
            raise StreamError("stream did not contain a finish_reason")
        header_evidence = {
            key: value
            for key, value in headers.items()
            if (key.startswith("x-") and ("cache" in key or "prefix" in key))
            or key in {"cached-tokens", "prefix-cache-hit"}
        }
        if header_evidence:
            cache_hit_evidence = {
                **(cache_hit_evidence or {}),
                "source": "response_headers",
                "headers": header_evidence,
            }
        finished_perf = time.perf_counter()
        finished_wall = time.time()
        token_timing_valid = None
        if usage is not None:
            token_timing_valid = event_count == usage["completion_tokens"]
        return StreamCapture(
            status=response.status,
            headers=headers,
            started_at_unix_s=started_wall,
            first_event_at_unix_s=first_event_wall,
            finished_at_unix_s=finished_wall,
            elapsed_ms=(finished_perf - started_perf) * 1000.0,
            ttft_ms=(None if first_event_wall is None else (first_event_wall - started_wall) * 1000.0),
            inter_event_ms=inter_event_ms,
            event_count=event_count,
            finish_reason=finish_reason,
            usage=usage,
            token_timing_valid=token_timing_valid,
            timing_semantics=("stream_chunk" if token_timing_valid is not True else "chunk_count_matches_usage"),
            cache_hit_evidence=cache_hit_evidence,
            body_sha256=digest.hexdigest(),
        )
    except socket.timeout as exc:
        raise StreamError("stream request timed out", timed_out=True) from exc
    except OSError as exc:
        raise StreamError(f"stream request failed: {exc}") from exc
    finally:
        conn.close()
