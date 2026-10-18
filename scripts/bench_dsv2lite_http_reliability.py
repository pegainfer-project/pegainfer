#!/usr/bin/env python3
"""DeepSeek-V2-Lite HTTP reliability gate for /v1/completions.

The gate intentionally uses the real OpenAI-compatible HTTP path and consumes
`pegainfer_http_trace` server logs. It fails when terminal traces are missing,
state does not retire back to a healthy baseline, clean follow-up requests fail,
success hashes drift, or an expected cancel/disconnect/reject outcome is not
visible in traces.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import re
import socket
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TRACE_RE = re.compile(r"pegainfer_http_trace\s+(\{.*\})")
DSV2_LITE_ACTIVE_CAP = 8
PROMPT_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
    "nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
).split()


@dataclass
class RequestResult:
    request_id: str
    kind: str
    ok: bool
    status: int | None
    terminal_reason: str
    client_observed_outcome: str
    error: str | None = None
    timed_out: bool = False
    start_wall_s: float = 0.0
    end_wall_s: float = 0.0
    latency_ms: float = 0.0
    first_token_seen: bool = False
    output_chunks: int = 0
    output_chars: int = 0
    output_hash: str = ""
    trace: dict[str, Any] | None = None


@dataclass
class ScenarioReport:
    name: str
    passed: bool
    failures: list[str]
    counts: dict[str, int]
    output_hashes: dict[str, str]
    trace_coverage: dict[str, Any]
    trace_maxima: dict[str, int | None]
    terminal_reasons: dict[str, int]
    final_healthy_baseline: bool
    clean_follow_up: RequestResult
    requests: list[RequestResult] = field(default_factory=list)


def make_prompt(index: int, words: int) -> str:
    return " ".join(PROMPT_WORDS[(index + offset) % len(PROMPT_WORDS)] for offset in range(words))


def parse_sse_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return choice.get("text") or ""
    delta = choice.get("delta") or {}
    return delta.get("content") or ""


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


def completion_path(base: urllib.parse.ParseResult) -> str:
    prefix = base.path.rstrip("/")
    return f"{prefix}/v1/completions" if prefix else "/v1/completions"


def connect(base: urllib.parse.ParseResult, timeout: float) -> http.client.HTTPConnection:
    conn_cls = http.client.HTTPSConnection if base.scheme == "https" else http.client.HTTPConnection
    return conn_cls(base.hostname, port=base.port, timeout=timeout)


def request_body(
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: str,
    **overrides: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "stream": True,
        "ignore_eos": True,
        "request_id": request_id,
    }
    body.update(overrides)
    return body


def run_stream_request(
    base: urllib.parse.ParseResult,
    model: str,
    request_id: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    kind: str,
    close_after_first_token: bool = False,
    close_after_headers: bool = False,
    **overrides: Any,
) -> RequestResult:
    started = time.time()
    chunks: list[str] = []
    first_token_seen = False
    status: int | None = None
    conn: http.client.HTTPConnection | None = None
    try:
        conn = connect(base, timeout)
        body = request_body(model, prompt, max_tokens, request_id, **overrides)
        conn.request(
            "POST",
            completion_path(base),
            body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        status = response.status
        if status != 200:
            error_body = response.read(4096).decode("utf-8", errors="replace")
            return finish_result(
                request_id,
                kind,
                False,
                status,
                started,
                "rejected",
                error=f"HTTP {status}: {error_body}",
            )
        if close_after_headers:
            conn.close()
            return finish_result(request_id, kind, False, status, started, "disconnected")
        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            payload = json.loads(data)
            stream_error = parse_sse_error(payload)
            if stream_error is not None:
                return finish_result(
                    request_id,
                    kind,
                    False,
                    status,
                    started,
                    "failed",
                    error=f"SSE error: {stream_error}",
                )
            text = parse_sse_text(payload)
            if not text:
                continue
            first_token_seen = True
            chunks.append(text)
            if close_after_first_token:
                conn.close()
                return finish_result(
                    request_id,
                    kind,
                    False,
                    status,
                    started,
                    "cancelled",
                    first_token_seen=True,
                    output_chunks=len(chunks),
                    output_text="".join(chunks),
                )
        text = "".join(chunks)
        if max_tokens > 0 and not chunks:
            return finish_result(
                request_id,
                kind,
                False,
                status,
                started,
                "failed",
                error="stream completed without text chunks",
            )
        return finish_result(
            request_id,
            kind,
            True,
            status,
            started,
            "completed",
            first_token_seen=first_token_seen,
            output_chunks=len(chunks),
            output_text=text,
        )
    except (TimeoutError, socket.timeout) as exc:
        return finish_result(
            request_id,
            kind,
            False,
            status,
            started,
            "timeout",
            timed_out=True,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - reliability report keeps raw failure.
        return finish_result(
            request_id,
            kind,
            False,
            status,
            started,
            "failed",
            error=str(exc),
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def finish_result(
    request_id: str,
    kind: str,
    ok: bool,
    status: int | None,
    started: float,
    terminal_reason: str,
    *,
    error: str | None = None,
    timed_out: bool = False,
    first_token_seen: bool = False,
    output_chunks: int = 0,
    output_text: str = "",
) -> RequestResult:
    ended = time.time()
    return RequestResult(
        request_id=request_id,
        kind=kind,
        ok=ok,
        status=status,
        terminal_reason=terminal_reason,
        client_observed_outcome=terminal_reason,
        error=error,
        timed_out=timed_out,
        start_wall_s=started,
        end_wall_s=ended,
        latency_ms=(ended - started) * 1000.0,
        first_token_seen=first_token_seen,
        output_chunks=output_chunks,
        output_chars=len(output_text),
        output_hash=hashlib.sha256(output_text.encode("utf-8")).hexdigest()[:16]
        if output_text
        else "",
    )


def load_server_traces(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    traces: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TRACE_RE.search(line)
        if not match:
            continue
        try:
            trace = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        request_id = trace.get("request_id")
        if isinstance(request_id, str):
            traces[request_id] = trace
    return traces


def find_trace(request_id: str, traces: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    prefix = f"cmpl-{request_id}-"
    for trace_id, trace in traces.items():
        if trace_id == request_id or trace_id == f"cmpl-{request_id}" or trace_id.startswith(prefix):
            return trace
    return None


def attach_traces(results: list[RequestResult], traces: dict[str, dict[str, Any]]) -> None:
    for result in results:
        trace = find_trace(result.request_id, traces)
        if trace is None:
            continue
        result.trace = trace
        terminal_reason = trace.get("terminal_reason")
        if isinstance(terminal_reason, str):
            result.terminal_reason = normalize_trace_terminal_reason(terminal_reason)


def normalize_trace_terminal_reason(reason: str) -> str:
    if reason in {"completed", "completed_length", "completed_stop"}:
        return "completed"
    if reason in {"cancelled", "disconnected", "rejected"}:
        return reason
    return "failed"


def count_results(results: list[RequestResult]) -> dict[str, int]:
    counts = {
        "completed": 0,
        "failed": 0,
        "rejected": 0,
        "cancelled": 0,
        "disconnected": 0,
        "timeout": 0,
    }
    for result in results:
        if result.timed_out or result.client_observed_outcome == "timeout":
            counts["timeout"] += 1
        elif result.terminal_reason in counts:
            counts[result.terminal_reason] += 1
        elif result.ok:
            counts["completed"] += 1
        else:
            counts["failed"] += 1
    return counts


def trace_max(results: list[RequestResult], field: str) -> int | None:
    values = [
        int(result.trace[field])
        for result in results
        if result.trace is not None and isinstance(result.trace.get(field), int)
    ]
    return max(values) if values else None


def terminal_reason_counts(results: list[RequestResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        reason = result.trace.get("terminal_reason") if result.trace is not None else result.terminal_reason
        if not isinstance(reason, str):
            reason = "missing_trace"
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def stable_success_hashes(results: list[RequestResult]) -> tuple[bool, dict[str, str]]:
    hashes: dict[str, str] = {}
    stable = True
    for result in results:
        if not result.ok:
            continue
        key = result.kind
        previous = hashes.setdefault(key, result.output_hash)
        if previous != result.output_hash:
            stable = False
    return stable, hashes


def clean_follow_up(base: urllib.parse.ParseResult, args: argparse.Namespace, name: str) -> RequestResult:
    return run_stream_request(
        base,
        args.model,
        f"dsv2-rel-{name}-follow-up",
        make_prompt(97, 16),
        args.follow_up_tokens,
        args.timeout,
        "clean_follow_up",
    )


def build_scenario(
    name: str,
    results: list[RequestResult],
    follow_up: RequestResult,
    *,
    required_reasons: set[str],
    require_pending_pressure: bool = False,
    require_active_pressure: bool = False,
    allow_http_guard_rejections: bool = False,
) -> ScenarioReport:
    all_results = [*results, follow_up]
    counts = count_results(all_results)
    stable_hashes, hashes = stable_success_hashes(all_results)
    traced = [result for result in all_results if result.trace is not None]
    missing = [
        result.request_id
        for result in all_results
        if result.trace is None and not trace_optional(result, allow_http_guard_rejections)
    ]
    reasons = terminal_reason_counts(all_results)
    failures: list[str] = []

    if missing:
        failures.append(f"missing terminal traces: {missing}")
    if not follow_up.ok:
        failures.append(f"clean follow-up failed: {follow_up.error or follow_up.terminal_reason}")
    if not stable_hashes:
        failures.append("successful output hashes are not stable by request kind")
    for reason in sorted(required_reasons):
        if counts.get(reason, 0) <= 0:
            failures.append(f"expected terminal reason {reason!r} was not observed")
    for result in all_results:
        if result.timed_out or result.client_observed_outcome == "timeout":
            failures.append(f"request timed out without terminal recovery evidence: {result.request_id}")
        elif result.terminal_reason == "failed":
            detail = result.error or result.trace.get("error") if result.trace is not None else result.error
            failures.append(f"request failed unexpectedly: {result.request_id}: {detail or 'unknown error'}")
        if result.trace is None or trace_optional(result, allow_http_guard_rejections):
            continue
        missing_fields = [
            field
            for field in (
                "terminal_reason",
                "active_set_size_max",
                "pending_queue_size_max",
                "decode_batch_size_max",
                "active_set_size_at_terminal",
                "pending_queue_size_at_terminal",
                "healthy_baseline_after_terminal",
            )
            if field not in result.trace
        ]
        if missing_fields:
            failures.append(f"trace for {result.request_id} is missing fields: {missing_fields}")
        elif result.trace.get("healthy_baseline_after_terminal") is not True and result.kind == "clean_follow_up":
            failures.append(f"clean follow-up did not retire to baseline: {result.request_id}")

    final_healthy = bool(
        follow_up.trace is not None and follow_up.trace.get("healthy_baseline_after_terminal") is True
    )
    if not final_healthy:
        failures.append("clean follow-up trace did not return to healthy baseline")

    active_max = trace_max(all_results, "active_set_size_max")
    pending_max = trace_max(all_results, "pending_queue_size_max")
    if require_active_pressure and (active_max is None or active_max < 2):
        failures.append("active-set pressure was not visible in trace")
    if active_max is not None and active_max > DSV2_LITE_ACTIVE_CAP:
        failures.append(
            f"active_set_size_max exceeded DSV2-Lite active cap {DSV2_LITE_ACTIVE_CAP}: {active_max}"
        )
    if require_pending_pressure and (pending_max is None or pending_max < 1):
        failures.append("pending-queue pressure was not visible in trace")

    return ScenarioReport(
        name=name,
        passed=not failures,
        failures=failures,
        counts=counts,
        output_hashes=hashes,
        trace_coverage={
            "traced": len(traced),
            "total": len(all_results),
            "missing_request_ids": missing,
            "http_guard_rejections_without_trace": [
                result.request_id
                for result in all_results
                if result.trace is None and trace_optional(result, allow_http_guard_rejections)
            ],
        },
        trace_maxima={
            "active_set_size_max": active_max,
            "pending_queue_size_max": pending_max,
            "decode_batch_size_max": trace_max(all_results, "decode_batch_size_max"),
        },
        terminal_reasons=reasons,
        final_healthy_baseline=final_healthy,
        clean_follow_up=follow_up,
        requests=all_results,
    )


def trace_optional(result: RequestResult, allow_http_guard_rejections: bool) -> bool:
    return (
        allow_http_guard_rejections
        and result.terminal_reason == "rejected"
        and result.status is not None
        and result.status != 200
    )


def run_cancel_disconnect(base: urllib.parse.ParseResult, args: argparse.Namespace) -> ScenarioReport:
    specs = [
        ("dsv2-rel-cancel-after-token", "cancel_after_first_token", make_prompt(0, 16), {"close_after_first_token": True}),
        ("dsv2-rel-disconnect-before-token", "disconnect_before_token", make_prompt(1, 16), {"close_after_headers": True}),
        ("dsv2-rel-neighbor-ok", "neighbor_success", make_prompt(2, 16), {}),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = []
        for request_id, kind, prompt, options in specs:
            futures.append(
                pool.submit(
                    run_stream_request,
                    base,
                    args.model,
                    request_id,
                    prompt,
                    args.max_tokens,
                    args.timeout,
                    kind,
                    **options,
                )
            )
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    results.sort(key=lambda item: item.request_id)
    return build_scenario(
        "cancel_disconnect",
        results,
        clean_follow_up(base, args, "cancel-disconnect"),
        required_reasons={"cancelled", "disconnected", "completed"},
    )


def run_invalid_requests(base: urllib.parse.ParseResult, args: argparse.Namespace) -> ScenarioReport:
    over_context_words = args.over_context_words
    specs = [
        ("dsv2-rel-invalid-sampling", "invalid_non_greedy", make_prompt(3, 16), {"temperature": 0.8}),
        ("dsv2-rel-invalid-logprobs", "invalid_logprobs", make_prompt(4, 16), {"logprobs": 1}),
        ("dsv2-rel-invalid-empty", "invalid_empty_prompt", "", {}),
        ("dsv2-rel-invalid-context", "invalid_over_context", make_prompt(5, over_context_words), {}),
        ("dsv2-rel-invalid-neighbor-ok", "neighbor_success", make_prompt(6, 16), {}),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = [
            pool.submit(
                run_stream_request,
                base,
                args.model,
                request_id,
                prompt,
                args.max_tokens,
                args.timeout,
                kind,
                **options,
            )
            for request_id, kind, prompt, options in specs
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    results.sort(key=lambda item: item.request_id)
    return build_scenario(
        "invalid_requests",
        results,
        clean_follow_up(base, args, "invalid"),
        required_reasons={"rejected", "completed"},
        allow_http_guard_rejections=True,
    )


def run_overload(base: urllib.parse.ParseResult, args: argparse.Namespace) -> ScenarioReport:
    count = args.overload_concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        futures = [
            pool.submit(
                run_stream_request,
                base,
                args.model,
                f"dsv2-rel-overload-{idx}",
                make_prompt(20, 16),
                args.max_tokens,
                args.timeout,
                "overload_success",
            )
            for idx in range(count)
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    results.sort(key=lambda item: item.request_id)
    return build_scenario(
        "overload_active_cap",
        results,
        clean_follow_up(base, args, "overload"),
        required_reasons={"completed"},
        require_pending_pressure=True,
        require_active_pressure=True,
    )


def run_mixed_faults(base: urllib.parse.ParseResult, args: argparse.Namespace) -> ScenarioReport:
    specs = [
        ("dsv2-rel-mixed-short-ok-0", "mixed_short_success", make_prompt(10, 16), {}),
        ("dsv2-rel-mixed-long-ok-0", "mixed_long_success", make_prompt(11, 128), {}),
        ("dsv2-rel-mixed-cancel", "mixed_cancel", make_prompt(12, 16), {"close_after_first_token": True}),
        ("dsv2-rel-mixed-logprobs", "mixed_rejected", make_prompt(13, 16), {"logprobs": 1}),
        ("dsv2-rel-mixed-short-ok-1", "mixed_short_success", make_prompt(10, 16), {}),
        ("dsv2-rel-mixed-long-ok-1", "mixed_long_success", make_prompt(11, 128), {}),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.mixed_concurrency) as pool:
        futures = []
        for request_id, kind, prompt, options in specs:
            options = dict(options)
            close_after_first_token = bool(options.pop("close_after_first_token", False))
            futures.append(
                pool.submit(
                    run_stream_request,
                    base,
                    args.model,
                    request_id,
                    prompt,
                    args.mixed_max_tokens,
                    args.timeout,
                    kind,
                    close_after_first_token=close_after_first_token,
                    **options,
                )
            )
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    results.sort(key=lambda item: item.request_id)
    return build_scenario(
        "mixed_short_long_with_failures",
        results,
        clean_follow_up(base, args, "mixed"),
        required_reasons={"cancelled", "rejected", "completed"},
        require_active_pressure=True,
    )


def attach_scenario_traces(scenarios: list[ScenarioReport], traces: dict[str, dict[str, Any]]) -> None:
    for scenario in scenarios:
        attach_traces(scenario.requests, traces)
        refreshed = build_scenario(
            scenario.name,
            scenario.requests[:-1],
            scenario.requests[-1],
            required_reasons=required_reasons_for(scenario.name),
            require_pending_pressure=scenario.name == "overload_active_cap",
            require_active_pressure=scenario.name in {"overload_active_cap", "mixed_short_long_with_failures"},
            allow_http_guard_rejections=scenario.name == "invalid_requests",
        )
        scenario.passed = refreshed.passed
        scenario.failures = refreshed.failures
        scenario.counts = refreshed.counts
        scenario.output_hashes = refreshed.output_hashes
        scenario.trace_coverage = refreshed.trace_coverage
        scenario.trace_maxima = refreshed.trace_maxima
        scenario.terminal_reasons = refreshed.terminal_reasons
        scenario.final_healthy_baseline = refreshed.final_healthy_baseline
        scenario.clean_follow_up = refreshed.clean_follow_up


def required_reasons_for(name: str) -> set[str]:
    return {
        "cancel_disconnect": {"cancelled", "disconnected", "completed"},
        "invalid_requests": {"rejected", "completed"},
        "overload_active_cap": {"completed"},
        "mixed_short_long_with_failures": {"cancelled", "rejected", "completed"},
    }[name]


def dry_run_report(args: argparse.Namespace) -> dict[str, Any]:
    synthetic_wall_s = 0.0

    def dry_finish(
        request_id: str,
        kind: str,
        ok: bool,
        terminal_reason: str,
        output_text: str = "",
        output_chunks: int = 0,
    ) -> RequestResult:
        result = finish_result(
            request_id,
            kind,
            ok,
            200,
            synthetic_wall_s,
            terminal_reason,
            output_text=output_text,
            output_chunks=output_chunks,
        )
        result.end_wall_s = synthetic_wall_s
        result.latency_ms = 0.0
        return result

    def trace(request_id: str, reason: str, active: int, pending: int, decode: int) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "terminal_reason": reason,
            "active_set_size_max": active,
            "pending_queue_size_max": pending,
            "decode_batch_size_max": decode,
            "active_set_size_at_terminal": 0,
            "pending_queue_size_at_terminal": 0,
            "healthy_baseline_after_terminal": True,
            "prompt_tokens": 16,
            "completion_tokens": 4,
        }

    scenarios: list[ScenarioReport] = []
    samples = {
        "cancel_disconnect": [("a", "cancelled"), ("b", "disconnected"), ("c", "completed")],
        "invalid_requests": [("a", "rejected"), ("b", "rejected"), ("c", "completed")],
        "overload_active_cap": [("a", "completed"), ("b", "completed"), ("c", "completed")],
        "mixed_short_long_with_failures": [
            ("a", "completed"),
            ("b", "cancelled"),
            ("c", "rejected"),
            ("d", "completed"),
        ],
    }
    for scenario_name, rows in samples.items():
        results = []
        for suffix, reason in rows:
            request_id = f"dry-{scenario_name}-{suffix}"
            ok = reason == "completed"
            result = dry_finish(
                request_id,
                "dry",
                ok,
                normalize_trace_terminal_reason(reason),
                output_text="stable" if ok else "",
                output_chunks=1 if ok else 0,
            )
            result.trace = trace(
                request_id,
                "completed_length" if reason == "completed" else reason,
                DSV2_LITE_ACTIVE_CAP if scenario_name == "overload_active_cap" else 3,
                2 if scenario_name == "overload_active_cap" else 0,
                2,
            )
            results.append(result)
        follow = dry_finish(
            f"dry-{scenario_name}-follow",
            "clean_follow_up",
            True,
            "completed",
            output_text="follow",
            output_chunks=1,
        )
        follow.trace = trace(follow.request_id, "completed_length", 1, 0, 1)
        scenarios.append(
            build_scenario(
                scenario_name,
                results,
                follow,
                required_reasons=required_reasons_for(scenario_name),
                require_pending_pressure=scenario_name == "overload_active_cap",
                require_active_pressure=scenario_name
                in {"overload_active_cap", "mixed_short_long_with_failures"},
            )
        )
    return render_report(args, scenarios, dry_run=True)


def render_report(args: argparse.Namespace, scenarios: list[ScenarioReport], *, dry_run: bool) -> dict[str, Any]:
    passed = all(scenario.passed for scenario in scenarios)
    report = {
        "schema_version": 1,
        "kind": "deepseek_v2_lite_http_reliability_gate",
        "passed": passed,
        "dry_run": dry_run,
        "base_url": args.base_url,
        "model": args.model,
        "workload": {
            "max_tokens": args.max_tokens,
            "follow_up_tokens": args.follow_up_tokens,
            "overload_concurrency": args.overload_concurrency,
            "mixed_concurrency": args.mixed_concurrency,
            "mixed_max_tokens": args.mixed_max_tokens,
            "over_context_words": args.over_context_words,
            "timeout_s": args.timeout,
        },
        "summary": {
            "passed": passed,
            "scenario_count": len(scenarios),
            "passed_scenarios": sum(1 for scenario in scenarios if scenario.passed),
            "failed_scenarios": [scenario.name for scenario in scenarios if not scenario.passed],
        },
        "scenarios": [scenario_to_json(scenario) for scenario in scenarios],
    }
    return report


def scenario_to_json(scenario: ScenarioReport) -> dict[str, Any]:
    data = asdict(scenario)
    data["requests"] = [asdict(result) for result in scenario.requests]
    data["clean_follow_up"] = asdict(scenario.clean_follow_up)
    return data


def run_live(args: argparse.Namespace) -> dict[str, Any]:
    base = urllib.parse.urlparse(args.base_url)
    if base.scheme not in {"http", "https"} or not base.hostname:
        raise SystemExit(f"invalid --base-url: {args.base_url}")
    scenarios = [
        run_cancel_disconnect(base, args),
        run_invalid_requests(base, args),
        run_overload(base, args),
        run_mixed_faults(base, args),
    ]
    attach_scenario_traces(scenarios, load_server_traces(args.server_log))
    return render_report(args, scenarios, dry_run=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v2-lite")
    parser.add_argument("--server-log", type=Path, help="server log with pegainfer_http_trace lines")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--follow-up-tokens", type=int, default=8)
    parser.add_argument("--overload-concurrency", type=int, default=12)
    parser.add_argument("--mixed-concurrency", type=int, default=6)
    parser.add_argument("--mixed-max-tokens", type=int, default=16)
    parser.add_argument("--over-context-words", type=int, default=9000)
    parser.add_argument("--dry-run", action="store_true", help="validate schema and pass/fail logic without HTTP")
    args = parser.parse_args()

    if args.max_tokens <= 0 or args.follow_up_tokens <= 0 or args.mixed_max_tokens <= 0:
        raise SystemExit("token counts must be positive")
    if args.overload_concurrency <= 8:
        raise SystemExit("--overload-concurrency must exceed the DSV2-Lite active cap of 8")
    if args.mixed_concurrency <= 1:
        raise SystemExit("--mixed-concurrency must be > 1")

    report = dry_run_report(args) if args.dry_run else run_live(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
