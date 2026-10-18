#!/usr/bin/env python3
"""OpenAI-compatible HTTP serving benchmark for pegainfer.

The harness talks to /v1/completions over HTTP so it measures the real
serving path (frontend + bridge + engine). It records streaming TTFT/ITL/TPOT,
request latency, QPS, error rate, timeout rate, and deterministic output hashes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import random
import re
import socket
import statistics
import sys
import threading
import time
import urllib.parse
import uuid
from dataclasses import asdict, dataclass, field as dataclass_field
from itertools import product
from pathlib import Path
from typing import Any

from bench_http_common import (
    artifact_command,
    combined_output_hash,
    current_commit,
    detect_hardware_toolchain,
    model_fingerprint,
    sha256_file,
    value_counts,
    write_json,
)


DEFAULT_PROMPT_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
    "nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
).split()


@dataclass
class RequestResult:
    index: int
    request_id: str
    prompt_words: int
    max_tokens: int
    ok: bool
    status: int | None
    error: str | None
    timed_out: bool
    start_s: float
    start_wall_s: float
    first_token_s: float | None
    first_token_wall_s: float | None
    end_s: float
    end_wall_s: float
    latency_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    itl_ms: list[float]
    output_chunks: int
    output_chars: int
    output_hash: str
    text_prefix: str
    stream_chunk_tpot_ms: float | None = None
    stream_chunk_itl_ms: list[float] = dataclass_field(default_factory=list)
    token_timing_valid: bool | None = None
    sampling_label: str = "single"
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    server_trace: dict[str, Any] | None = None


@dataclass(frozen=True)
class SamplingProfile:
    label: str
    temperature: float
    top_k: int
    top_p: float


@dataclass
class StreamCapture:
    chunks: list[str] = dataclass_field(default_factory=list)
    inter_chunk_ms: list[float] = dataclass_field(default_factory=list)
    first_chunk_s: float | None = None
    first_chunk_wall_s: float | None = None
    last_chunk_s: float | None = None
    done_received: bool = False


def arg_value(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def sampling_mode(args: argparse.Namespace) -> str:
    return arg_value(args, "sampling_mode", "single")


def single_profile(args: argparse.Namespace) -> SamplingProfile:
    return SamplingProfile(
        label="single",
        temperature=float(arg_value(args, "temperature", 0.0)),
        top_k=int(arg_value(args, "top_k", -1)),
        top_p=float(arg_value(args, "top_p", 1.0)),
    )


def greedy_profile() -> SamplingProfile:
    return SamplingProfile(label="greedy", temperature=0.0, top_k=-1, top_p=1.0)


def sampled_profile(args: argparse.Namespace) -> SamplingProfile:
    return SamplingProfile(
        label="sampled",
        temperature=float(arg_value(args, "sample_temperature", 0.8)),
        top_k=int(arg_value(args, "sample_top_k", 40)),
        top_p=float(arg_value(args, "sample_top_p", 0.95)),
    )


def sampling_profile_for(
    args: argparse.Namespace, global_index: int
) -> SamplingProfile:
    if sampling_mode(args) == "mixed-greedy-sampled":
        return greedy_profile() if global_index % 2 == 0 else sampled_profile(args)
    return single_profile(args)


def sampling_profiles_for_report(
    args: argparse.Namespace,
) -> dict[str, dict[str, float | int | str]]:
    if sampling_mode(args) == "mixed-greedy-sampled":
        profiles = [greedy_profile(), sampled_profile(args)]
    else:
        profiles = [single_profile(args)]
    return {profile.label: asdict(profile) for profile in profiles}


def count_sampling(results: list[RequestResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.sampling_label] = counts.get(result.sampling_label, 0) + 1
    return counts


def wire_top_k(top_k: int) -> int:
    return 0 if top_k <= 0 else top_k


def validate_top_p(name: str, value: float) -> None:
    if value <= 0.0 or value > 1.0:
        raise SystemExit(f"{name} must be in (0, 1]")


def validate_sampling_args(args: argparse.Namespace) -> None:
    validate_top_p("--top-p", args.top_p)
    validate_top_p("--sample-top-p", args.sample_top_p)
    if args.sampling_mode == "mixed-greedy-sampled" and args.sample_temperature <= 0.0:
        raise SystemExit(
            "--sample-temperature must be positive in mixed-greedy-sampled mode"
        )


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "avg_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "samples": 0,
        }
    sorted_values = sorted(values)
    return {
        "avg_ms": statistics.fmean(sorted_values),
        "p50_ms": percentile(sorted_values, 50),
        "p95_ms": percentile(sorted_values, 95),
        "p99_ms": percentile(sorted_values, 99),
        "max_ms": sorted_values[-1],
        "samples": len(sorted_values),
    }


def summarize_counts(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {
            "min": None,
            "max": None,
            "total": None,
            "samples": 0,
        }
    return {
        "min": min(values),
        "max": max(values),
        "total": sum(values),
        "samples": len(values),
    }


def build_retention_gate(report: dict[str, Any]) -> dict[str, Any]:
    required_trace = report.get("contract", {}).get("required_trace_coverage_ratio")
    if required_trace is None:
        return {"required": False, "passed": None}
    required_trace = float(required_trace)
    trace = report["server_trace"]
    outcomes_passed = (
        report["summary"]["completed"] == report["workload"]["num_requests"]
        and report["summary"]["failed"] == 0
        and report["summary"]["timeouts"] == 0
    )
    trace_passed = all(
        float(trace.get(field) or 0.0) >= required_trace
        for field in (
            "coverage_ratio",
            "active_set_coverage_ratio",
            "decode_batch_coverage_ratio",
            "token_timing_coverage_ratio",
        )
    )
    return {
        "required": True,
        "passed": outcomes_passed and trace_passed,
        "request_outcomes_passed": outcomes_passed,
        "trace_coverage_passed": trace_passed,
        "required_trace_coverage_ratio": required_trace,
    }


def is_full_server_trace(trace: dict[str, Any] | None) -> bool:
    if trace is None:
        return False
    trace_fields = {
        "queued_at_unix_s",
        "scheduled_at_unix_s",
        "first_token_emit_unix_s",
        "prompt_tokens",
        "completion_tokens",
        "active_set_size",
        "decode_batch_size_max",
        "decode_step_count",
        "batch_decode_steps",
    }
    return any(field in trace for field in trace_fields)


def summarize_trace_ms(measured: list[RequestResult]) -> dict[str, Any]:
    fields = [
        "frontend_to_queue_ms",
        "admission_queue_ms",
        "queue_wait_ms",
        "prefill_ms",
        "first_decode_ms",
        "decode_mean_ms",
        "decode_total_ms",
        "scheduled_to_first_token_ms",
        "scheduled_to_terminal_ms",
        "stream_flush_ms",
    ]
    phase_summary: dict[str, Any] = {}
    for field in fields:
        values = [
            float(result.server_trace[field])
            for result in measured
            if is_full_server_trace(result.server_trace)
            and isinstance(result.server_trace.get(field), (int, float))
        ]
        phase_summary[field] = summarize(values)
    attached = [result for result in measured if result.server_trace is not None]
    traced = [
        result for result in measured if is_full_server_trace(result.server_trace)
    ]
    server_error_records = [
        result
        for result in measured
        if result.server_trace is not None
        and isinstance(result.server_trace.get("server_error"), str)
    ]
    token_timing_requests = [
        result
        for result in measured
        if getattr(result, "token_timing_valid", None) is True
    ]
    token_timing_mismatches = [
        result.request_id
        for result in measured
        if getattr(result, "token_timing_valid", None) is False
    ]
    token_timing_unknown = [
        result.request_id
        for result in measured
        if getattr(result, "token_timing_valid", None) is None
    ]
    prompt_tokens = [
        int(result.server_trace["prompt_tokens"])
        for result in traced
        if isinstance(result.server_trace.get("prompt_tokens"), int)
    ]
    completion_tokens = [
        int(result.server_trace["completion_tokens"])
        for result in traced
        if isinstance(result.server_trace.get("completion_tokens"), int)
    ]
    active_set_sizes = [
        int(result.server_trace["active_set_size"])
        for result in traced
        if isinstance(result.server_trace.get("active_set_size"), int)
    ]
    decode_batch_sizes = [
        int(result.server_trace["decode_batch_size_max"])
        for result in traced
        if isinstance(result.server_trace.get("decode_batch_size_max"), int)
    ]
    decode_step_counts: list[int] = []
    complete_breakdowns: list[tuple[int, int]] = []
    breakdown_complete = bool(traced)
    for result in traced:
        trace = result.server_trace
        assert trace is not None
        count = (
            int(trace["decode_step_count"])
            if isinstance(trace.get("decode_step_count"), int)
            else None
        )
        batched = (
            int(trace["batch_decode_steps"])
            if isinstance(trace.get("batch_decode_steps"), int)
            else None
        )
        if count is not None:
            decode_step_counts.append(count)
        if count is None or batched is None:
            breakdown_complete = False
            continue

        singleton = count - batched
        if batched < 0 or singleton < 0:
            breakdown_complete = False
            continue
        complete_breakdowns.append((batched, singleton))
    if breakdown_complete:
        batched_total = sum(batched for batched, _ in complete_breakdowns)
        singleton_total = sum(singleton for _, singleton in complete_breakdowns)
        total_decode_steps = batched_total + singleton_total
        batched_share = (
            batched_total / total_decode_steps if total_decode_steps else None
        )
    else:
        batched_total = None
        singleton_total = None
        batched_share = None
    active_set_counts = value_counts([str(value) for value in active_set_sizes])
    decode_batch_counts = value_counts([str(value) for value in decode_batch_sizes])
    return {
        "source": "server log lines matching `pegainfer_http_trace`; frontend_to_queue includes HTTP ingress, tokenization, and vLLM submit before engine queue",
        "traced_requests": len(traced),
        "attached_server_records": len(attached),
        "server_error_records": len(server_error_records),
        "missing_traces": [
            result.request_id
            for result in measured
            if not is_full_server_trace(result.server_trace)
        ],
        "missing_server_records": [
            result.request_id for result in measured if result.server_trace is None
        ],
        "coverage_ratio": len(traced) / len(measured) if measured else 0.0,
        "server_record_coverage_ratio": len(attached) / len(measured)
        if measured
        else 0.0,
        "prompt_tokens": summarize_counts(prompt_tokens),
        "completion_tokens": summarize_counts(completion_tokens),
        "phases_ms": phase_summary,
        "active_set_sizes": active_set_counts,
        "active_set_coverage_ratio": len(active_set_sizes) / len(measured)
        if measured
        else 0.0,
        "active_set_size_max": max(active_set_sizes) if active_set_sizes else None,
        "decode_batch_sizes": decode_batch_counts,
        "decode_batch_coverage_ratio": len(decode_batch_sizes) / len(measured)
        if measured
        else 0.0,
        "decode_batch_size_max": max(decode_batch_sizes)
        if decode_batch_sizes
        else None,
        "token_timing_coverage_ratio": (
            len(token_timing_requests) / len(measured) if measured else 0.0
        ),
        "token_timing_mismatches": token_timing_mismatches,
        "token_timing_unknown": token_timing_unknown,
        "decode_steps": {
            "per_request": summarize_counts(decode_step_counts),
            "batched_request_steps_total": batched_total,
            "singleton_request_steps_total": singleton_total,
            "request_step_batched_share": batched_share,
        },
    }


def make_prompt(index: int, prompt_words: int) -> str:
    words = [
        DEFAULT_PROMPT_WORDS[(index + offset) % len(DEFAULT_PROMPT_WORDS)]
        for offset in range(prompt_words)
    ]
    return " ".join(words)


def load_prompt_pool(args: argparse.Namespace) -> list[str] | None:
    """Deterministic prompt pool from a ShareGPT-style JSON (--prompt-file).

    Takes each conversation's first human turn, filters by length, then
    seed-samples --prompt-count of them — the same protocol as the DSpark/DFlash
    closing A/Bs (30 ShareGPT first-turn prompts), reproducible via
    --prompt-seed. Synthetic --prompt-words prompts overstate speculative
    accept rates (repetitive text drafts too well); use real text for any
    spec-on number that will be quoted.
    """
    path = arg_value(args, "prompt_file", None)
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    lo = int(arg_value(args, "prompt_min_chars", 32))
    hi = int(arg_value(args, "prompt_max_chars", 2000))
    firsts = []
    for item in data:
        conversations = item.get("conversations") or []
        if not conversations:
            continue
        first = conversations[0]
        if first.get("from") != "human":
            continue
        text = (first.get("value") or "").strip()
        if lo <= len(text) <= hi:
            firsts.append(text)
    count = int(arg_value(args, "prompt_count", 30))
    if len(firsts) < count:
        raise SystemExit(f"prompt file has only {len(firsts)} usable first turns, need {count}")
    rng = random.Random(int(arg_value(args, "prompt_seed", 512)))
    return rng.sample(firsts, count)


def parse_int_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("values must be positive integers")
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def single_or_list(values: list[int]) -> int | list[int]:
    return values[0] if len(values) == 1 else values


def workload_shapes(
    prompt_words: list[int], max_tokens: list[int]
) -> list[tuple[int, int]]:
    return list(product(prompt_words, max_tokens))


def parse_sse_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return choice.get("text") or ""
    delta = choice.get("delta") or {}
    return delta.get("content") or ""


def parse_sse_finish_reason(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices") or []
    if not choices:
        return None
    finish_reason = choices[0].get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else None


def parse_sse_error(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
        return json.dumps(error, sort_keys=True)
    return None


def set_deadline_timeout(conn: http.client.HTTPConnection, deadline_s: float) -> None:
    remaining_s = deadline_s - time.perf_counter()
    if remaining_s <= 0.0:
        raise TimeoutError("request exceeded its absolute timeout")
    conn.timeout = remaining_s
    if conn.sock is not None:
        conn.sock.settimeout(remaining_s)


def iter_response_lines(
    response: http.client.HTTPResponse,
    conn: http.client.HTTPConnection,
    deadline_s: float,
):
    buffered = b""
    while True:
        set_deadline_timeout(conn, deadline_s)
        chunk = response.read1(4096)
        if not chunk:
            if buffered:
                yield buffered
            return
        buffered += chunk
        while b"\n" in buffered:
            line, buffered = buffered.split(b"\n", 1)
            yield line


def parse_sse_line(raw: bytes) -> tuple[bool, str]:
    line = raw.decode("utf-8", errors="replace").strip()
    if not line or not line.startswith("data:"):
        return False, ""
    data = line.removeprefix("data:").strip()
    if data == "[DONE]":
        return True, ""
    payload = json.loads(data)
    stream_error = parse_sse_error(payload)
    if stream_error is not None:
        raise RuntimeError(f"SSE error: {stream_error}")
    if parse_sse_finish_reason(payload) == "error":
        raise RuntimeError("SSE finish_reason=error")
    return False, parse_sse_text(payload)


def consume_sse_stream(
    response: http.client.HTTPResponse,
    conn: http.client.HTTPConnection,
    deadline_s: float,
) -> StreamCapture:
    capture = StreamCapture()
    for raw in iter_response_lines(response, conn, deadline_s):
        done, text = parse_sse_line(raw)
        if done:
            capture.done_received = True
            break
        if not text:
            continue
        now = time.perf_counter()
        if capture.first_chunk_s is None:
            capture.first_chunk_s = now
            capture.first_chunk_wall_s = time.time()
        if capture.last_chunk_s is not None:
            capture.inter_chunk_ms.append((now - capture.last_chunk_s) * 1000.0)
        capture.last_chunk_s = now
        capture.chunks.append(text)
    if not capture.done_received:
        raise RuntimeError("SSE stream ended before [DONE]")
    return capture


def start_deadline_watchdog(
    conn: http.client.HTTPConnection,
    deadline_s: float,
    expired: threading.Event,
) -> threading.Timer:
    def interrupt_at_deadline() -> None:
        expired.set()
        if conn.sock is None:
            return
        try:
            conn.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    timer = threading.Timer(
        max(0.0, deadline_s - time.perf_counter()), interrupt_at_deadline
    )
    timer.daemon = True
    timer.start()
    return timer


def successful_result(
    *,
    index: int,
    request_id: str,
    prompt_words: int,
    max_tokens: int,
    sampling_label: str,
    temperature: float,
    top_k: int,
    top_p: float,
    status: int,
    start_s: float,
    start_wall_s: float,
    capture: StreamCapture,
) -> RequestResult:
    end_s = time.perf_counter()
    text = "".join(capture.chunks)
    chunk_tpot_ms = None
    if (
        capture.first_chunk_s is not None
        and capture.last_chunk_s is not None
        and len(capture.chunks) > 1
    ):
        chunk_tpot_ms = (
            (capture.last_chunk_s - capture.first_chunk_s)
            * 1000.0
            / (len(capture.chunks) - 1)
        )
    return RequestResult(
        index=index,
        request_id=request_id,
        prompt_words=prompt_words,
        max_tokens=max_tokens,
        ok=True,
        status=status,
        error=None,
        timed_out=False,
        start_s=start_s,
        start_wall_s=start_wall_s,
        first_token_s=capture.first_chunk_s,
        first_token_wall_s=capture.first_chunk_wall_s,
        end_s=end_s,
        end_wall_s=time.time(),
        latency_ms=(end_s - start_s) * 1000.0,
        ttft_ms=(
            None
            if capture.first_chunk_s is None
            else (capture.first_chunk_s - start_s) * 1000.0
        ),
        tpot_ms=None,
        itl_ms=[],
        output_chunks=len(capture.chunks),
        output_chars=len(text),
        output_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        text_prefix=text[:80],
        stream_chunk_tpot_ms=chunk_tpot_ms,
        stream_chunk_itl_ms=capture.inter_chunk_ms,
        sampling_label=sampling_label,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


def request_once(
    index: int,
    request_id: str,
    url: urllib.parse.ParseResult,
    model: str,
    prompt_words: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    ignore_eos: bool,
    top_k: int = -1,
    top_p: float = 1.0,
    sampling_label: str = "single",
) -> RequestResult:
    start = time.perf_counter()
    deadline = start + timeout
    start_wall = time.time()
    status: int | None = None
    conn: http.client.HTTPConnection | None = None
    deadline_expired = threading.Event()
    deadline_timer: threading.Timer | None = None

    try:
        conn_cls = (
            http.client.HTTPSConnection
            if url.scheme == "https"
            else http.client.HTTPConnection
        )
        port = url.port
        conn = conn_cls(url.hostname, port=port, timeout=timeout)

        deadline_timer = start_deadline_watchdog(conn, deadline, deadline_expired)
        set_deadline_timeout(conn, deadline)
        path = (url.path.rstrip("/") or "") + "/v1/completions"
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": wire_top_k(top_k),
            "top_p": top_p,
            "stream": True,
            "ignore_eos": ignore_eos,
            "request_id": request_id,
        }
        conn.request(
            "POST",
            path,
            body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        set_deadline_timeout(conn, deadline)
        response = conn.getresponse()
        status = response.status
        if status != 200:
            error_body = response.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {status}: {error_body}")

        capture = consume_sse_stream(response, conn, deadline)
        if deadline_expired.is_set() or time.perf_counter() > deadline:
            raise TimeoutError("request exceeded its absolute timeout")
        if max_tokens > 0 and not capture.chunks:
            raise RuntimeError("stream completed without streamed text chunks")
        return successful_result(
            index=index,
            request_id=request_id,
            prompt_words=prompt_words,
            max_tokens=max_tokens,
            sampling_label=sampling_label,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            status=status,
            start_s=start,
            start_wall_s=start_wall,
            capture=capture,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark reports the error string.
        end = time.perf_counter()
        timed_out = deadline_expired.is_set() or isinstance(
            exc, (TimeoutError, socket.timeout)
        )
        return failed_result(
            index,
            request_id,
            prompt_words,
            max_tokens,
            sampling_label,
            temperature,
            top_k,
            top_p,
            status,
            start,
            start_wall,
            end,
            str(exc),
            timed_out=timed_out,
        )
    finally:
        if deadline_timer is not None:
            deadline_timer.cancel()
        if conn is not None:
            conn.close()


def failed_result(
    index: int,
    request_id: str,
    prompt_words: int,
    max_tokens: int,
    sampling_label: str,
    temperature: float,
    top_k: int,
    top_p: float,
    status: int | None,
    start: float,
    start_wall: float,
    end: float,
    error: str,
    timed_out: bool,
) -> RequestResult:
    end_wall = time.time()
    return RequestResult(
        index=index,
        request_id=request_id,
        prompt_words=prompt_words,
        max_tokens=max_tokens,
        ok=False,
        status=status,
        error=error,
        timed_out=timed_out,
        start_s=start,
        start_wall_s=start_wall,
        first_token_s=None,
        first_token_wall_s=None,
        end_s=end,
        end_wall_s=end_wall,
        latency_ms=(end - start) * 1000.0,
        ttft_ms=None,
        tpot_ms=None,
        itl_ms=[],
        output_chunks=0,
        output_chars=0,
        output_hash="",
        text_prefix="",
        sampling_label=sampling_label,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


TRACE_RE = re.compile(r"pegainfer_http_trace\s+(\{.*\})")
STREAM_ERROR_RE = re.compile(r'request failed .*self\.request_id="([^"]+)"')
TRACE_MATCH_SLOP_S = 5.0


def server_log_offset(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return path.stat().st_size


def load_server_traces(
    path: Path | None,
    *,
    start_offset: int = 0,
) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    if start_offset < 0:
        raise ValueError("start_offset must be non-negative")
    traces: dict[str, dict[str, Any]] = {}
    with path.open("rb") as handle:
        handle.seek(start_offset)
        log_text = handle.read().decode("utf-8", errors="replace")
    for line in log_text.splitlines():
        stream_error_match = STREAM_ERROR_RE.search(line)
        if stream_error_match:
            request_id = stream_error_match.group(1)
            traces.setdefault(request_id, {"request_id": request_id})[
                "server_error"
            ] = line.strip()
            continue
        match = TRACE_RE.search(line)
        if not match:
            continue
        try:
            trace = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        request_id = trace.get("request_id")
        if isinstance(request_id, str):
            existing = traces.get(request_id)
            if existing is not None and isinstance(existing.get("server_error"), str):
                trace["server_error"] = existing["server_error"]
            traces[request_id] = trace
    return traces


def attach_server_traces(
    results: list[RequestResult], traces: dict[str, dict[str, Any]]
) -> None:
    for result in results:
        trace = find_server_trace(
            result.request_id,
            result.start_wall_s,
            result.end_wall_s,
            traces,
        )
        if trace is None:
            continue
        result.server_trace = trace
        if result.ok and result.first_token_wall_s is not None:
            emit_at = trace.get("first_token_emit_unix_s")
            if isinstance(emit_at, (int, float)):
                trace["stream_flush_ms"] = max(
                    0.0, (result.first_token_wall_s - float(emit_at)) * 1000.0
                )
            queued_at = trace.get("queued_at_unix_s")
            if isinstance(queued_at, (int, float)):
                trace["frontend_to_queue_ms"] = max(
                    0.0, (float(queued_at) - result.start_wall_s) * 1000.0
                )
            scheduled_at = trace.get("scheduled_at_unix_s")
            if isinstance(queued_at, (int, float)) and isinstance(
                scheduled_at, (int, float)
            ):
                trace["admission_queue_ms"] = max(
                    0.0, (float(scheduled_at) - float(queued_at)) * 1000.0
                )
        apply_server_error_gate(result)
        finalize_token_timing(result)


def finalize_token_timing(result: RequestResult) -> None:
    trace = result.server_trace
    if trace is None or not isinstance(trace.get("completion_tokens"), int):
        return
    completion_tokens = int(trace["completion_tokens"])
    result.token_timing_valid = result.output_chunks == completion_tokens
    if result.token_timing_valid:
        if result.stream_chunk_tpot_ms is not None:
            result.tpot_ms = result.stream_chunk_tpot_ms
        if result.stream_chunk_itl_ms:
            result.itl_ms = list(result.stream_chunk_itl_ms)
        return
    result.tpot_ms = None
    result.itl_ms = []


def apply_server_error_gate(result: RequestResult) -> None:
    if not result.ok or result.server_trace is None:
        return
    server_error = result.server_trace.get("server_error")
    if isinstance(server_error, str):
        result.ok = False
        result.error = f"server generation error: {server_error}"
        return
    finish_reason = result.server_trace.get("finish_reason")
    if finish_reason == "error":
        result.ok = False
        result.error = "server generation error: finish_reason=error"
        return
    completion_tokens = result.server_trace.get("completion_tokens")
    if result.max_tokens > 0 and completion_tokens == 0:
        result.ok = False
        result.error = "server generation error: completion_tokens=0"


def find_server_trace(
    request_id: str,
    start_wall_s: float,
    end_wall_s: float,
    traces: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    prefix = f"cmpl-{request_id}-"
    matches = [
        trace
        for trace_id, trace in traces.items()
        if trace_id == request_id
        or trace_id == f"cmpl-{request_id}"
        or trace_id.startswith(prefix)
    ]
    timed_matches = []
    for trace in matches:
        timestamps = [
            float(trace[field])
            for field in ("queued_at_unix_s", "terminal_unix_s")
            if isinstance(trace.get(field), (int, float))
        ]
        if any(
            start_wall_s - TRACE_MATCH_SLOP_S
            <= timestamp
            <= end_wall_s + TRACE_MATCH_SLOP_S
            for timestamp in timestamps
        ):
            timed_matches.append(trace)
    if len(timed_matches) == 1:
        return timed_matches[0]
    if len(timed_matches) > 1:
        return min(
            timed_matches,
            key=lambda trace: abs(
                float(trace.get("queued_at_unix_s", trace.get("terminal_unix_s")))
                - start_wall_s
            ),
        )
    return None


def run_batch(
    args: argparse.Namespace, measured: bool
) -> tuple[list[RequestResult], float]:
    url = urllib.parse.urlparse(args.base_url)
    if url.scheme not in {"http", "https"} or not url.hostname:
        raise SystemExit(f"invalid --base-url: {args.base_url}")

    offset = args.warmup if measured else 0
    count = args.num_requests if measured else args.warmup
    label = "measured" if measured else "warmup"
    request_id_prefix = arg_value(args, "request_id_prefix", None)
    if request_id_prefix is None:
        request_id_prefix = f"pegainfer-bench-{uuid.uuid4().hex}"
        args.request_id_prefix = request_id_prefix
    shapes = workload_shapes(args.prompt_words, args.max_tokens)
    prompt_pool = load_prompt_pool(args)
    workloads = []
    for idx in range(count):
        global_index = offset + idx
        prompt_words, max_tokens = shapes[global_index % len(shapes)]
        if prompt_pool is not None:
            # Paired design under mixed-greedy-sampled: the profile alternates
            # by global_index parity, so indexing prompts by global_index //
            # 2 gives each prompt to BOTH labels (greedy g=2k and sampled
            # g=2k+1 share prompt k). Indexing by raw global_index would, with
            # an even pool, permanently assign each prompt to one label and
            # confound per-label comparisons with prompt content.
            pidx = (
                global_index // 2
                if sampling_mode(args) == "mixed-greedy-sampled"
                else global_index
            )
            prompt = prompt_pool[pidx % len(prompt_pool)]
            prompt_words = len(prompt.split())
        else:
            prompt = make_prompt(global_index, prompt_words)
        workloads.append(
            (
                global_index,
                prompt_words,
                max_tokens,
                prompt,
                sampling_profile_for(args, global_index),
            )
        )
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                request_once,
                idx,
                f"{request_id_prefix}-{label}-{offset + idx}",
                url,
                args.model,
                prompt_words,
                prompt,
                max_tokens,
                profile.temperature,
                args.timeout,
                args.ignore_eos,
                top_k=profile.top_k,
                top_p=profile.top_p,
                sampling_label=profile.label,
            )
            for idx, (
                _global_index,
                prompt_words,
                max_tokens,
                prompt,
                profile,
            ) in enumerate(workloads)
        ]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
    ended = time.perf_counter()
    results.sort(key=lambda result: result.index)
    return results, ended - started


def build_report(
    args: argparse.Namespace, measured: list[RequestResult], wall_s: float
) -> dict[str, Any]:
    backend = getattr(args, "backend", None)
    contract_name = getattr(args, "contract_name", None)
    contract_description = getattr(args, "contract_description", None)
    claim_boundary = getattr(args, "claim_boundary", None)
    required_trace_coverage = getattr(args, "required_trace_coverage", None)
    for result in measured:
        finalize_token_timing(result)
    successes = [result for result in measured if result.ok]
    failures = [result for result in measured if not result.ok]
    latencies = [result.latency_ms for result in successes]
    ttfts = [result.ttft_ms for result in successes if result.ttft_ms is not None]
    tpots = [result.tpot_ms for result in successes if result.tpot_ms is not None]
    itls: list[float] = []
    stream_chunk_tpots = [
        result.stream_chunk_tpot_ms
        for result in successes
        if result.stream_chunk_tpot_ms is not None
    ]
    stream_chunk_itls = [
        value for result in successes for value in result.stream_chunk_itl_ms
    ]
    output_chunks = [result.output_chunks for result in successes]
    output_chars = [result.output_chars for result in successes]
    hashes = [result.output_hash for result in successes]
    input_tokens = [
        int(result.server_trace["prompt_tokens"])
        if result.server_trace is not None
        and isinstance(result.server_trace.get("prompt_tokens"), int)
        else result.prompt_words
        for result in successes
    ]
    output_tokens = [
        int(result.server_trace["completion_tokens"])
        for result in successes
        if result.server_trace is not None
        and isinstance(result.server_trace.get("completion_tokens"), int)
    ]
    output_token_counts_complete = len(output_tokens) == len(successes)
    output_tokens_total = sum(output_tokens) if output_token_counts_complete else None
    shape_counts: dict[str, int] = {}
    for result in measured:
        key = f"prompt_words={result.prompt_words},max_tokens={result.max_tokens}"
        shape_counts[key] = shape_counts.get(key, 0) + 1

    for result in successes:
        itls.extend(result.itl_ms)

    report = {
        "schema_version": 1,
        "kind": "openai_http_completions_stream_benchmark",
        "report_intent": "http_serving_slo"
        if contract_name
        else "http_serving_benchmark",
        "base_url": args.base_url,
        "model": args.model,
        "backend": backend,
        "contract": {
            "name": contract_name,
            "backend": backend,
            "description": contract_description,
            "required_trace_coverage_ratio": required_trace_coverage,
            "claim_boundary": claim_boundary
            or "HTTP streaming benchmark evidence only; direct, profiler, soak, and production-readiness claims require separate artifacts.",
        },
        "metadata": {
            "commit": getattr(args, "commit", None) or current_commit(),
            "backend": backend,
            "contract_name": contract_name,
            "model_path": getattr(args, "model_path", None),
            "server_command": getattr(args, "server_command", None),
            "source_revision": getattr(args, "source_revision", None),
            "model_revision": getattr(args, "model_revision", None),
            "model_fingerprint": model_fingerprint(getattr(args, "model_path", None)),
            "server_binary_sha256": sha256_file(Path(getattr(args, "server_binary")))
            if getattr(args, "server_binary", None)
            else None,
            "backend_runtime_version": getattr(args, "backend_runtime_version", None),
            "benchmark_command": artifact_command(sys.argv),
            "hardware_toolchain": detect_hardware_toolchain(),
        },
        "workload": {
            "num_requests": args.num_requests,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "prompt_words": single_or_list(args.prompt_words),
            "max_tokens": single_or_list(args.max_tokens),
            "mixed_shapes": shape_counts,
            "temperature": args.temperature,
            "top_k": int(arg_value(args, "top_k", -1)),
            "top_p": float(arg_value(args, "top_p", 1.0)),
            "sampling_mode": sampling_mode(args),
            "sampling_profiles": sampling_profiles_for_report(args),
            "sampling_counts": count_sampling(measured),
            "ignore_eos": args.ignore_eos,
            "timeout_s": args.timeout,
            "timeout_kind": "absolute_request_deadline",
        },
        "summary": {
            "wall_s": wall_s,
            "completed": len(successes),
            "failed": len(failures),
            "timeouts": sum(1 for result in failures if result.timed_out),
            "sampling_mode": sampling_mode(args),
            "completed_sampling_counts": count_sampling(successes),
            "failed_sampling_counts": count_sampling(failures),
            "qps": len(successes) / wall_s if wall_s > 0 else 0.0,
            "input_tokens_total": sum(input_tokens),
            "output_tokens_total": output_tokens_total,
            "input_tokens_per_s": sum(input_tokens) / wall_s if wall_s > 0 else 0.0,
            "output_tokens_per_s": (
                output_tokens_total / wall_s
                if output_tokens_total is not None and wall_s > 0
                else None
            ),
            "output_token_count_source": (
                "server_trace.completion_tokens"
                if output_token_counts_complete
                else "unavailable"
            ),
            "output_token_count_coverage_ratio": (
                len(output_tokens) / len(successes) if successes else 0.0
            ),
            "error_rate": len(failures) / args.num_requests
            if args.num_requests
            else 0.0,
            "timeout_rate": (
                sum(1 for result in failures if result.timed_out) / args.num_requests
                if args.num_requests
                else 0.0
            ),
            "output_chunks_total": sum(output_chunks),
            "output_chars_total": sum(output_chars),
            "unique_output_hashes": len(set(hashes)),
            "output_hash_distribution": value_counts(hashes),
            "combined_output_hash": combined_output_hash(hashes),
        },
        "metrics": {
            "latency": summarize(latencies),
            "ttft": summarize(ttfts),
            "tpot": summarize(tpots),
            "itl": summarize(itls),
            "stream_chunk_tpot": summarize(stream_chunk_tpots),
            "stream_chunk_itl": summarize(stream_chunk_itls),
        },
        "metric_definitions": {
            "percentile_method": "R7 linear interpolation over sorted samples",
            "ttft": "request start to first non-empty streamed text chunk",
            "tpot": "mean inter-token time per request, emitted only when streamed text chunk count equals server completion_tokens",
            "itl": "inter-token latency, emitted only when streamed text chunk count equals server completion_tokens",
            "stream_chunk_tpot": "mean inter-chunk time per request; diagnostic when a chunk can contain multiple tokens",
            "stream_chunk_itl": "inter-chunk latency; diagnostic when a chunk can contain multiple tokens",
            "output_tokens": "server trace completion_tokens only; unavailable when any successful request lacks a trusted token count",
        },
        "server_trace": summarize_trace_ms(measured),
        "requests": [asdict(result) for result in measured],
    }
    report["retention_gate"] = build_retention_gate(report)
    report["latency_budget"] = {
        "configured": False,
        "passed": None,
        "reason": "This retained report records latency distributions but does not define a production latency budget.",
    }
    return report


def write_report(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if args.out:
        rendered = write_json(args.out, report)
    else:
        rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--warmup", type=int)
    parser.add_argument(
        "--prompt-words",
        type=parse_int_list,
        default=None,
        help="Prompt word count, or comma-separated counts for a mixed workload.",
    )
    parser.add_argument(
        "--max-tokens",
        type=parse_int_list,
        default=None,
        help="Completion token count, or comma-separated counts for a mixed workload.",
    )
    parser.add_argument("--contract-name", help="Stable report contract name.")
    parser.add_argument(
        "--contract-description", help="Contract description written into JSON."
    )
    parser.add_argument(
        "--claim-boundary", help="Claim boundary written into the JSON report."
    )
    parser.add_argument(
        "--required-trace-coverage",
        type=float,
        help="Require request, active-set, and decode-batch trace coverage at this ratio.",
    )
    parser.add_argument(
        "--backend", help="Backend label, for example host-staged or nccl."
    )
    parser.add_argument("--model-path", help="Server model path used for this run.")
    parser.add_argument(
        "--server-command", help="Server launch command used for this run."
    )
    parser.add_argument(
        "--commit", help="PegaInfer commit for this run; defaults to git HEAD."
    )
    parser.add_argument(
        "--source-revision", help="Exact source-tree revision identifier."
    )
    parser.add_argument(
        "--model-revision", help="Model snapshot or revision identifier."
    )
    parser.add_argument(
        "--server-binary", type=Path, help="Server binary used for this run."
    )
    parser.add_argument(
        "--backend-runtime-version", help="Loaded backend runtime version."
    )
    parser.add_argument(
        "--prompt-file",
        help="ShareGPT-style JSON; first human turns become the prompt pool (overrides --prompt-words)",
    )
    parser.add_argument("--prompt-count", type=int, default=30)
    parser.add_argument("--prompt-seed", type=int, default=512)
    parser.add_argument("--prompt-min-chars", type=int, default=32)
    parser.add_argument("--prompt-max-chars", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--sampling-mode",
        choices=["single", "mixed-greedy-sampled"],
        default="single",
        help=(
            "single uses --temperature/--top-k/--top-p for every request; "
            "mixed-greedy-sampled alternates greedy and sampled profiles by global request index."
        ),
    )
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-k", type=int, default=40)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=float)
    parser.add_argument(
        "--ignore-eos", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--server-log",
        type=Path,
        help="Optional pegainfer server log containing pegainfer_http_trace lines for TTFT phase attribution.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.prompt_words is None:
        args.prompt_words = [16]
    if args.max_tokens is None:
        args.max_tokens = [16]
    if args.timeout is None:
        args.timeout = 120.0
    if args.warmup is None:
        args.warmup = 1
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.contract_name and not args.backend:
        raise SystemExit("--backend is required when --contract-name is set")
    if args.required_trace_coverage is not None:
        if args.required_trace_coverage <= 0.0 or args.required_trace_coverage > 1.0:
            raise SystemExit("--required-trace-coverage must be in (0, 1]")
        if args.server_log is None:
            raise SystemExit("--server-log is required with --required-trace-coverage")

    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.num_requests <= 0:
        raise SystemExit("--num-requests must be positive")
    if args.timeout <= 0.0:
        raise SystemExit("--timeout must be positive")
    validate_sampling_args(args)


def run_warmup(args: argparse.Namespace) -> None:
    if args.warmup <= 0:
        return
    warmup_results, _ = run_batch(args, measured=False)
    attach_server_traces(warmup_results, load_server_traces(args.server_log))
    if any(not result.ok for result in warmup_results):
        report = build_report(args, warmup_results, wall_s=0.0)
        report["report_intent"] = "http_serving_slo_warmup_failure"
        report["warmup_failed"] = True
        write_report(args, report)
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_warmup(args)

    trace_offset = server_log_offset(args.server_log)
    measured, wall_s = run_batch(args, measured=True)
    attach_server_traces(
        measured, load_server_traces(args.server_log, start_offset=trace_offset)
    )
    report = build_report(args, measured, wall_s)
    write_report(args, report)

    if report["summary"]["failed"] or report["retention_gate"].get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
