#!/usr/bin/env python3
"""Run and combine sustained DeepSeek-V2-Lite HTTP soak evidence."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bench_http_common import (
    artifact_command,
    combined_output_hash,
    current_commit,
    detect_hardware_toolchain,
    model_fingerprint,
    numeric_summary,
    sha256_file,
    value_counts,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BENCH = SCRIPT_DIR / "bench_http_serving.py"
BACKENDS = ("host-staged", "nccl")
DEFAULT_CONTRACT_NAME = "dsv2-lite-http-soak"
DEFAULT_CONTRACT_DESCRIPTION = (
    "DeepSeek-V2-Lite sustained HTTP soak contract for EP2 serving."
)
DEFAULT_CLAIM_BOUNDARY = (
    "Sustained HTTP soak evidence for a fixed DeepSeek-V2-Lite EP2 serving "
    "contract. This is not direct decode attribution, vLLM parity, multi-node "
    "recovery, or production-readiness evidence by itself."
)
RESOURCE_SAMPLE_VERSION = 1
GENERIC_RUNTIME_BOUNDARIES = {"", "unknown", "n/a", "na", "none"}


@dataclass(frozen=True)
class ResourceSample:
    wall_s: float
    rss_kib: int | None
    gpu_memory_used_mib: list[int]
    gpu_memory_scope: str


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must not be empty")
    parsed = [int(item) for item in items]
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    return parsed


def normalized_runtime_boundary(value: Any) -> str:
    return str(value or "").strip()


def runtime_boundary_is_present(value: Any) -> bool:
    return normalized_runtime_boundary(value).lower() not in GENERIC_RUNTIME_BOUNDARIES


def run_text(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def ps_rss_kib(pid: int | None) -> int | None:
    if pid is None:
        return None
    output = run_text(["ps", "-o", "rss=", "-p", str(pid)])
    if not output:
        return None
    try:
        return int(output.splitlines()[0].strip())
    except (IndexError, ValueError):
        return None


def gpu_memory_used_mib() -> list[int]:
    output = run_text(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    values = []
    for line in output.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    return values


def sample_resources(args: argparse.Namespace, started_wall_s: float) -> ResourceSample:
    return ResourceSample(
        wall_s=time.time() - started_wall_s,
        rss_kib=ps_rss_kib(args.server_pid),
        gpu_memory_used_mib=gpu_memory_used_mib(),
        gpu_memory_scope="device_total",
    )


def validate_commit(commit: str) -> None:
    if not commit:
        raise SystemExit("--commit is required")
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit("soak runs require a Git worktree with a valid HEAD") from exc
    if commit not in {head, head[:12]}:
        raise SystemExit(f"--commit {commit} does not match current HEAD {head}")


def required_files(args: argparse.Namespace) -> list[Path]:
    paths = [
        args.server_log,
        Path(args.model_path) / "config.json",
        Path(args.model_path) / "model.safetensors.index.json",
    ]
    if args.server_binary is not None:
        paths.append(args.server_binary)
    return paths


def validate_run_args(args: argparse.Namespace) -> None:
    if args.duration_s <= 0.0:
        raise SystemExit("--duration-s must be positive")
    if args.bucket_s <= 0.0:
        raise SystemExit("--bucket-s must be positive")
    if args.num_requests <= 0:
        raise SystemExit("--num-requests must be positive")
    if args.timeout <= 0.0:
        raise SystemExit("--timeout must be positive")
    if args.required_trace_coverage is not None and not (
        0.0 < args.required_trace_coverage <= 1.0
    ):
        raise SystemExit("--required-trace-coverage must be in (0, 1]")
    if args.max_buckets is not None and args.max_buckets <= 0:
        raise SystemExit("--max-buckets must be positive")
    runtime_boundary = normalized_runtime_boundary(args.backend_runtime_version)
    if not runtime_boundary:
        raise SystemExit("--backend-runtime-version must describe the runtime boundary")
    if args.backend == "nccl" and runtime_boundary.lower() == "nccl":
        raise SystemExit(
            "--backend-runtime-version for NCCL must include the NCCL version "
            "or selector boundary"
        )
    args.backend_runtime_version = runtime_boundary
    missing = [str(path) for path in required_files(args) if not path.is_file()]
    if missing:
        raise SystemExit(f"required soak-run files are missing: {', '.join(missing)}")
    validate_commit(args.commit)


def optional_arg(flag: str, value: Any) -> list[str]:
    if value is None:
        return []
    return [flag, str(value)]


def build_leaf_command(
    args: argparse.Namespace,
    *,
    out: Path,
    concurrency: int,
    num_requests: int,
    prompt_words: int,
    max_tokens: int,
    contract_name: str,
    contract_description: str,
) -> list[str]:
    command = [
        sys.executable,
        str(BENCH),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--num-requests",
        str(num_requests),
        "--concurrency",
        str(concurrency),
        "--warmup",
        "0",
        "--prompt-words",
        str(prompt_words),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        "0.0",
        "--top-k",
        "-1",
        "--top-p",
        "1.0",
        "--timeout",
        str(args.timeout),
        "--server-log",
        str(args.server_log),
        "--contract-name",
        contract_name,
        "--contract-description",
        contract_description,
        "--claim-boundary",
        args.claim_boundary,
        "--backend",
        args.backend,
        "--model-path",
        args.model_path,
        "--server-command",
        args.server_command,
        "--commit",
        args.commit,
        "--source-revision",
        args.commit,
        "--model-revision",
        args.model_revision,
        "--backend-runtime-version",
        args.backend_runtime_version,
        "--out",
        str(out),
    ]
    if args.server_binary is not None:
        command.extend(["--server-binary", str(args.server_binary)])
    if args.required_trace_coverage is not None:
        command.extend(
            ["--required-trace-coverage", str(args.required_trace_coverage)]
        )
    if args.no_ignore_eos:
        command.append("--no-ignore-eos")
    return command


def read_report(path: Path) -> dict[str, Any] | None:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return report if isinstance(report, dict) else None


def trace_max(report: dict[str, Any], field: str) -> int | None:
    values = []
    for request in report.get("requests", []):
        if not isinstance(request, dict):
            continue
        trace = request.get("server_trace")
        if isinstance(trace, dict) and isinstance(trace.get(field), int):
            values.append(int(trace[field]))
    if values:
        return max(values)
    trace = report.get("server_trace")
    if isinstance(trace, dict) and isinstance(trace.get(field), int):
        return int(trace[field])
    return None


def terminal_reason_counts(report: dict[str, Any]) -> dict[str, int]:
    reasons = []
    for request in report.get("requests", []):
        if not isinstance(request, dict):
            continue
        trace = request.get("server_trace")
        if isinstance(trace, dict) and isinstance(trace.get("terminal_reason"), str):
            reasons.append(trace["terminal_reason"])
        elif request.get("ok") is True:
            reasons.append("completed_without_terminal_trace")
        elif request.get("timed_out") is True:
            reasons.append("client_timeout")
        else:
            reasons.append("client_failure")
    return value_counts(reasons)


def error_counts(report: dict[str, Any]) -> dict[str, int]:
    values = []
    for request in report.get("requests", []):
        if not isinstance(request, dict) or request.get("ok") is True:
            continue
        error = request.get("error")
        values.append(str(error)[:160] if error else "unknown")
    return value_counts(values)


def metric_percentile(report: dict[str, Any], metric: str, percentile: str) -> float | None:
    value = (
        ((report.get("metrics") or {}).get(metric) or {}).get(f"{percentile}_ms")
    )
    return float(value) if isinstance(value, (int, float)) else None


def percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def metric_percentiles(values: list[float | int | None]) -> dict[str, float | None]:
    clean = sorted(
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    return {
        percentile_name: percentile(clean, pct)
        for percentile_name, pct in (("p50", 50), ("p95", 95), ("p99", 99))
    }


def successful_requests(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        request
        for report in reports
        for request in report.get("requests", [])
        if isinstance(request, dict) and request.get("ok") is True
    ]


def measured_requests(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        request
        for report in reports
        for request in report.get("requests", [])
        if isinstance(request, dict)
    ]


def full_server_trace(trace: dict[str, Any] | None) -> bool:
    if trace is None:
        return False
    trace_fields = {
        "queued_at_unix_s",
        "scheduled_at_unix_s",
        "first_token_emit_unix_s",
        "prompt_tokens",
        "completion_tokens",
        "active_set_size",
        "active_set_size_max",
        "pending_queue_size_max",
        "decode_batch_size_max",
        "decode_step_count",
        "batch_decode_steps",
    }
    return any(field in trace for field in trace_fields)


def aggregate_trace_coverage(requests: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(requests)
    traces = [
        request.get("server_trace")
        for request in requests
        if isinstance(request.get("server_trace"), dict)
    ]
    full_traces = [trace for trace in traces if full_server_trace(trace)]
    active_set_values = [
        int(trace["active_set_size"])
        for trace in full_traces
        if isinstance(trace.get("active_set_size"), int)
    ]
    decode_batch_values = [
        int(trace["decode_batch_size_max"])
        for trace in full_traces
        if isinstance(trace.get("decode_batch_size_max"), int)
    ]
    token_timing_requests = [
        request
        for request in requests
        if request.get("token_timing_valid") is True
    ]
    return {
        "coverage_ratio": len(full_traces) / total if total else 0.0,
        "server_record_coverage_ratio": len(traces) / total if total else 0.0,
        "active_set_coverage_ratio": len(active_set_values) / total if total else 0.0,
        "decode_batch_coverage_ratio": len(decode_batch_values) / total
        if total
        else 0.0,
        "token_timing_coverage_ratio": len(token_timing_requests) / total
        if total
        else 0.0,
        "missing_traces": [
            request.get("request_id")
            for request in requests
            if not full_server_trace(request.get("server_trace"))
        ],
        "missing_server_records": [
            request.get("request_id")
            for request in requests
            if not isinstance(request.get("server_trace"), dict)
        ],
    }


def aggregate_trace_max(requests: list[dict[str, Any]], field: str) -> int | None:
    values = []
    for request in requests:
        trace = request.get("server_trace")
        if isinstance(trace, dict) and isinstance(trace.get(field), int):
            values.append(int(trace[field]))
    return max(values) if values else None


def aggregate_trace_max_any(
    requests: list[dict[str, Any]], fields: tuple[str, ...]
) -> int | None:
    values = [
        value
        for field in fields
        if (value := aggregate_trace_max(requests, field)) is not None
    ]
    return max(values) if values else None


def bucket_record(
    *,
    bucket_index: int,
    concurrency: int,
    report_path: Path,
    report: dict[str, Any] | None,
    returncode: int,
    resource_before: ResourceSample,
    resource_after: ResourceSample,
) -> dict[str, Any]:
    digest = sha256_file(report_path)
    report_loaded = isinstance(report, dict)
    record: dict[str, Any] = {
        "bucket_index": bucket_index,
        "concurrency": concurrency,
        "artifact": str(report_path),
        "sha256": digest,
        "leaf_artifacts": [
            {
                "artifact": str(report_path),
                "sha256": digest,
                "benchmark_returncode": returncode,
                "report_loaded": report_loaded,
            }
        ],
        "leaf_count": 1,
        "benchmark_returncode": returncode,
        "resource_before": asdict(resource_before),
        "resource_after": asdict(resource_after),
    }
    if report is None:
        record["report_loaded"] = False
        return record
    summary = report.get("summary") or {}
    trace = report.get("server_trace") or {}
    record.update(
        {
            "report_loaded": True,
            "completed": int(summary.get("completed") or 0),
            "failed": int(summary.get("failed") or 0),
            "timeouts": int(summary.get("timeouts") or 0),
            "wall_s": summary.get("wall_s"),
            "qps": summary.get("qps"),
            "input_tokens_per_s": summary.get("input_tokens_per_s"),
            "output_tokens_per_s": summary.get("output_tokens_per_s"),
            "output_hash_distribution": summary.get("output_hash_distribution") or {},
            "combined_output_hash": summary.get("combined_output_hash"),
            "ttft_ms": {
                percentile: metric_percentile(report, "ttft", percentile)
                for percentile in ("p50", "p95", "p99")
            },
            "tpot_ms": {
                percentile: metric_percentile(report, "tpot", percentile)
                for percentile in ("p50", "p95", "p99")
            },
            "itl_ms": {
                percentile: metric_percentile(report, "itl", percentile)
                for percentile in ("p50", "p95", "p99")
            },
            "trace_coverage": {
                "coverage_ratio": trace.get("coverage_ratio"),
                "server_record_coverage_ratio": trace.get(
                    "server_record_coverage_ratio"
                ),
                "active_set_coverage_ratio": trace.get("active_set_coverage_ratio"),
                "decode_batch_coverage_ratio": trace.get(
                    "decode_batch_coverage_ratio"
                ),
                "token_timing_coverage_ratio": trace.get(
                    "token_timing_coverage_ratio"
                ),
                "missing_traces": trace.get("missing_traces") or [],
                "missing_server_records": trace.get("missing_server_records") or [],
            },
            "active_set_size_max": trace_max(report, "active_set_size_max"),
            "pending_queue_size_max": trace_max(report, "pending_queue_size_max"),
            "decode_batch_size_max": trace_max(report, "decode_batch_size_max"),
            "terminal_reasons": terminal_reason_counts(report),
            "errors": error_counts(report),
        }
    )
    return record


def aggregate_bucket_record(
    *,
    bucket_index: int,
    concurrency: int,
    leaf_runs: list[dict[str, Any]],
    resource_before: ResourceSample,
    resource_after: ResourceSample,
) -> dict[str, Any]:
    reports = [
        run["report"]
        for run in leaf_runs
        if isinstance(run.get("report"), dict)
    ]
    requests = measured_requests(reports)
    successes = successful_requests(reports)
    wall_s = max(0.0, resource_after.wall_s - resource_before.wall_s)
    total_counts = {"completed": 0, "failed": 0, "timeouts": 0}
    input_tokens_total = 0
    output_tokens_total = 0
    output_tokens_complete = True
    output_hashes = []
    terminal_reasons: dict[str, int] = {}
    errors: dict[str, int] = {}
    for report in reports:
        summary = report.get("summary") or {}
        total_counts["completed"] += int(summary.get("completed") or 0)
        total_counts["failed"] += int(summary.get("failed") or 0)
        total_counts["timeouts"] += int(summary.get("timeouts") or 0)
        input_tokens_total += int(summary.get("input_tokens_total") or 0)
        if isinstance(summary.get("output_tokens_total"), int):
            output_tokens_total += int(summary["output_tokens_total"])
        else:
            output_tokens_complete = False
        terminal_reasons = add_counts(terminal_reasons, terminal_reason_counts(report))
        errors = add_counts(errors, error_counts(report))
    output_hashes = [
        str(request.get("output_hash"))
        for request in successes
        if request.get("output_hash")
    ]
    returncodes = [run.get("returncode") for run in leaf_runs]
    leaf_reports_loaded = bool(leaf_runs) and all(
        isinstance(run.get("report"), dict)
        and sha256_file(run["report_path"]) is not None
        for run in leaf_runs
    )
    leaf_commands_passed = bool(leaf_runs) and all(
        isinstance(code, int) and code == 0 for code in returncodes
    )
    leaf_artifacts = [
        {
            "artifact": str(run["report_path"]),
            "sha256": sha256_file(run["report_path"]),
            "benchmark_returncode": run.get("returncode"),
            "report_loaded": isinstance(run.get("report"), dict),
        }
        for run in leaf_runs
    ]
    return {
        "bucket_index": bucket_index,
        "concurrency": concurrency,
        "artifact": str(leaf_runs[0]["report_path"]) if leaf_runs else None,
        "sha256": sha256_file(leaf_runs[0]["report_path"]) if leaf_runs else None,
        "leaf_artifacts": leaf_artifacts,
        "leaf_count": len(leaf_runs),
        "report_loaded": leaf_reports_loaded,
        "benchmark_returncode": 0 if leaf_commands_passed else 1,
        "resource_before": asdict(resource_before),
        "resource_after": asdict(resource_after),
        "wall_s": wall_s,
        "completed": total_counts["completed"],
        "failed": total_counts["failed"],
        "timeouts": total_counts["timeouts"],
        "qps": total_counts["completed"] / wall_s if wall_s > 0 else 0.0,
        "input_tokens_per_s": input_tokens_total / wall_s if wall_s > 0 else 0.0,
        "output_tokens_per_s": (
            output_tokens_total / wall_s
            if output_tokens_complete and wall_s > 0
            else None
        ),
        "output_hash_distribution": value_counts(output_hashes),
        "combined_output_hash": combined_output_hash(output_hashes),
        "ttft_ms": metric_percentiles(
            [request.get("ttft_ms") for request in successes]
        ),
        "tpot_ms": metric_percentiles(
            [request.get("tpot_ms") for request in successes]
        ),
        "itl_ms": metric_percentiles(
            [
                value
                for request in successes
                for value in (request.get("itl_ms") or [])
            ]
        ),
        "trace_coverage": aggregate_trace_coverage(requests),
        "active_set_size_max": aggregate_trace_max_any(
            requests, ("active_set_size_max", "active_set_size")
        ),
        "pending_queue_size_max": aggregate_trace_max(
            requests, "pending_queue_size_max"
        ),
        "decode_batch_size_max": aggregate_trace_max(requests, "decode_batch_size_max"),
        "terminal_reasons": terminal_reasons,
        "errors": errors,
    }


def add_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, 0) + int(value)
    return dict(sorted(result.items()))


def numeric_bucket_values(
    buckets: list[dict[str, Any]], path: tuple[str, ...]
) -> list[float | int | None]:
    values = []
    for bucket in buckets:
        current: Any = bucket
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        values.append(current if isinstance(current, (int, float)) else None)
    return values


def delta_pct(first: float | int | None, last: float | int | None) -> float | None:
    if first is None or last is None or float(first) == 0.0:
        return None
    return (float(last) - float(first)) / float(first) * 100.0


def quartile_span(buckets: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not buckets:
        return [], []
    width = max(1, math.ceil(len(buckets) / 4))
    return buckets[:width], buckets[-width:]


def drift_summary(buckets: list[dict[str, Any]]) -> dict[str, Any]:
    first, last = quartile_span(buckets)
    fields = {
        "qps": ("qps",),
        "output_tokens_per_s": ("output_tokens_per_s",),
        "ttft_p95_ms": ("ttft_ms", "p95"),
        "tpot_p95_ms": ("tpot_ms", "p95"),
        "itl_p95_ms": ("itl_ms", "p95"),
        "rss_kib": ("resource_after", "rss_kib"),
        "gpu_memory_max_mib": ("resource_after", "gpu_memory_used_mib"),
    }
    result = {}
    for label, path in fields.items():
        if label == "gpu_memory_max_mib":
            first_values = [
                max(sample)
                for bucket in first
                if isinstance(
                    sample := (
                        (bucket.get("resource_after") or {}).get(
                            "gpu_memory_used_mib"
                        )
                    ),
                    list,
                )
                and sample
            ]
            last_values = [
                max(sample)
                for bucket in last
                if isinstance(
                    sample := (
                        (bucket.get("resource_after") or {}).get(
                            "gpu_memory_used_mib"
                        )
                    ),
                    list,
                )
                and sample
            ]
        else:
            first_values = numeric_bucket_values(first, path)
            last_values = numeric_bucket_values(last, path)
        first_summary = numeric_summary(first_values)
        last_summary = numeric_summary(last_values)
        result[label] = {
            "first_quartile": first_summary,
            "last_quartile": last_summary,
            "median_delta_pct": delta_pct(
                first_summary.get("median"), last_summary.get("median")
            ),
        }
    first_gpu_vectors = gpu_memory_vectors(first)
    last_gpu_vectors = gpu_memory_vectors(last)
    result["gpu_memory_total_mib"] = drift_field_summary(
        [sum(sample) for sample in first_gpu_vectors],
        [sum(sample) for sample in last_gpu_vectors],
    )
    result["gpu_memory_by_device_mib"] = {}
    device_count = max(
        [len(sample) for sample in first_gpu_vectors + last_gpu_vectors] or [0]
    )
    for device_index in range(device_count):
        first_values = [
            sample[device_index]
            for sample in first_gpu_vectors
            if device_index < len(sample)
        ]
        last_values = [
            sample[device_index]
            for sample in last_gpu_vectors
            if device_index < len(sample)
        ]
        result["gpu_memory_by_device_mib"][str(device_index)] = drift_field_summary(
            first_values, last_values
        )
    return result


def gpu_memory_vectors(buckets: list[dict[str, Any]]) -> list[list[int]]:
    vectors = []
    for bucket in buckets:
        sample = (bucket.get("resource_after") or {}).get("gpu_memory_used_mib")
        if isinstance(sample, list) and sample:
            values = [int(value) for value in sample if isinstance(value, int)]
            if values:
                vectors.append(values)
    return vectors


def drift_field_summary(
    first_values: list[float | int | None],
    last_values: list[float | int | None],
) -> dict[str, Any]:
    first_summary = numeric_summary(first_values)
    last_summary = numeric_summary(last_values)
    return {
        "first_quartile": first_summary,
        "last_quartile": last_summary,
        "median_delta_pct": delta_pct(
            first_summary.get("median"), last_summary.get("median")
        ),
    }


def resource_summary(samples: list[ResourceSample]) -> dict[str, Any]:
    rss = [sample.rss_kib for sample in samples]
    gpu_max = [
        max(sample.gpu_memory_used_mib)
        for sample in samples
        if sample.gpu_memory_used_mib
    ]
    gpu_total = [
        sum(sample.gpu_memory_used_mib)
        for sample in samples
        if sample.gpu_memory_used_mib
    ]
    gpu_device_count = max(
        [len(sample.gpu_memory_used_mib) for sample in samples] or [0]
    )
    gpu_by_device = {
        str(device_index): numeric_summary(
            [
                sample.gpu_memory_used_mib[device_index]
                for sample in samples
                if device_index < len(sample.gpu_memory_used_mib)
            ]
        )
        for device_index in range(gpu_device_count)
    }
    return {
        "schema_version": RESOURCE_SAMPLE_VERSION,
        "samples": [asdict(sample) for sample in samples],
        "rss_kib": numeric_summary(rss),
        "gpu_memory_max_mib": numeric_summary(gpu_max),
        "gpu_memory_total_mib": numeric_summary(gpu_total),
        "gpu_memory_by_device_mib": gpu_by_device,
        "gpu_memory_scope": "device_total",
    }


def build_clean_followup(
    args: argparse.Namespace, started_wall_s: float
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if args.skip_clean_followup:
        return None, None
    out = args.out_dir / "clean_followup.json"
    before = sample_resources(args, started_wall_s)
    command = build_leaf_command(
        args,
        out=out,
        concurrency=1,
        num_requests=1,
        prompt_words=min(args.prompt_words),
        max_tokens=min(args.max_tokens),
        contract_name=f"{args.contract_name}-clean-followup",
        contract_description="Post-soak clean follow-up request.",
    )
    completed = subprocess.run(command, check=False)
    after = sample_resources(args, started_wall_s)
    report = read_report(out)
    record = bucket_record(
        bucket_index=-1,
        concurrency=1,
        report_path=out,
        report=report,
        returncode=completed.returncode,
        resource_before=before,
        resource_after=after,
    )
    return report, record


def build_summary(
    args: argparse.Namespace,
    *,
    buckets: list[dict[str, Any]],
    resources: list[ResourceSample],
    clean_followup: dict[str, Any] | None,
    run_errors: list[dict[str, Any]],
    started_wall_s: float,
    ended_wall_s: float,
) -> dict[str, Any]:
    by_concurrency = {
        str(concurrency): [
            bucket for bucket in buckets if bucket.get("concurrency") == concurrency
        ]
        for concurrency in args.concurrency
    }
    duration_coverage = {
        str(concurrency): {
            "required_s": args.duration_s,
            "observed_bucket_wall_s": sum(
                float(bucket.get("wall_s") or 0.0)
                for bucket in concurrency_buckets
            ),
            "bucket_count": len(concurrency_buckets),
        }
        for concurrency, concurrency_buckets in by_concurrency.items()
    }
    for record in duration_coverage.values():
        record["passed"] = (
            args.max_buckets is None
            and record["observed_bucket_wall_s"] >= record["required_s"]
        )
    total_counts = {"completed": 0, "failed": 0, "timeouts": 0}
    terminal_reasons: dict[str, int] = {}
    errors: dict[str, int] = {}
    output_hashes = []
    for bucket in buckets:
        for field in total_counts:
            total_counts[field] += int(bucket.get(field) or 0)
        terminal_reasons = add_counts(
            terminal_reasons, bucket.get("terminal_reasons") or {}
        )
        errors = add_counts(errors, bucket.get("errors") or {})
        output_hashes.extend(
            output_hash
            for output_hash, count in (
                bucket.get("output_hash_distribution") or {}
            ).items()
            for _ in range(int(count))
        )

    clean_summary = None
    if clean_followup is not None:
        clean_summary = {
            "passed": (
                clean_followup.get("benchmark_returncode") == 0
                and clean_followup.get("completed") == 1
                and clean_followup.get("failed") == 0
                and clean_followup.get("timeouts") == 0
            ),
            "bucket": clean_followup,
        }

    required_trace = args.required_trace_coverage
    trace_passed = True
    if required_trace is not None:
        for bucket in buckets:
            coverage = bucket.get("trace_coverage") or {}
            trace_passed = trace_passed and all(
                isinstance(coverage.get(field), (int, float))
                and float(coverage[field]) >= required_trace
                for field in (
                    "coverage_ratio",
                    "active_set_coverage_ratio",
                    "decode_batch_coverage_ratio",
                    "token_timing_coverage_ratio",
                )
            )

    bucket_coverage_passed = bool(buckets) and all(
        by_concurrency[str(concurrency)] for concurrency in args.concurrency
    )
    duration_coverage_passed = bool(duration_coverage) and all(
        bool(record["passed"]) for record in duration_coverage.values()
    )
    clean_followup_passed = (
        clean_summary is not None and clean_summary["passed"] is True
    )
    leaf_commands_passed = bucket_coverage_passed and all(
        bucket.get("benchmark_returncode") == 0
        and bucket.get("report_loaded") is True
        and bucket_leaf_artifacts_loaded(bucket)
        for bucket in buckets
    )
    passed = (
        not run_errors
        and leaf_commands_passed
        and duration_coverage_passed
        and total_counts["failed"] == 0
        and total_counts["timeouts"] == 0
        and trace_passed
        and clean_followup_passed
    )

    return {
        "schema_version": 1,
        "kind": "deepseek_v2_lite_http_soak_backend",
        "report_intent": "http_soak",
        "model": args.model,
        "backend": args.backend,
        "metadata": {
            "commit": args.commit or current_commit(),
            "backend": args.backend,
            "contract_name": args.contract_name,
            "model_path": args.model_path,
            "server_command": args.server_command,
            "source_revision": args.commit,
            "model_revision": args.model_revision,
            "model_fingerprint": model_fingerprint(args.model_path),
            "server_binary_sha256": sha256_file(args.server_binary)
            if args.server_binary
            else None,
            "backend_runtime_version": args.backend_runtime_version,
            "benchmark_command": artifact_command(sys.argv),
            "hardware_toolchain": detect_hardware_toolchain(),
        },
        "contract": {
            "name": args.contract_name,
            "backend": args.backend,
            "description": args.contract_description,
            "required_trace_coverage_ratio": args.required_trace_coverage,
            "claim_boundary": args.claim_boundary,
        },
        "workload": {
            "duration_s": args.duration_s,
            "bucket_s": args.bucket_s,
            "num_requests_per_leaf": args.num_requests,
            "prompt_words": args.prompt_words,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 1.0,
            "ignore_eos": not args.no_ignore_eos,
            "timeout_s": args.timeout,
            "timeout_kind": "absolute_request_deadline",
        },
        "summary": {
            "started_wall_s": started_wall_s,
            "ended_wall_s": ended_wall_s,
            "elapsed_s": ended_wall_s - started_wall_s,
            "bucket_count": len(buckets),
            "completed": total_counts["completed"],
            "failed": total_counts["failed"],
            "timeouts": total_counts["timeouts"],
            "terminal_reasons": terminal_reasons,
            "errors": errors,
            "output_hash_distribution": value_counts(output_hashes),
            "combined_output_hash": combined_output_hash(output_hashes),
        },
        "soak_gate": {
            "passed": passed,
            "bucket_coverage_passed": bucket_coverage_passed,
            "duration_coverage_passed": duration_coverage_passed,
            "leaf_commands_passed": leaf_commands_passed,
            "zero_failures": total_counts["failed"] == 0,
            "zero_timeouts": total_counts["timeouts"] == 0,
            "trace_coverage_passed": trace_passed,
            "clean_followup_passed": clean_followup_passed,
            "run_errors": run_errors,
            "rule": (
                "This gate checks request completion, optional trace coverage, "
                "declared-duration coverage, leaf command success, complete leaf "
                "artifacts, and clean follow-up. Numeric drift is reported but not "
                "a hard budget until production limits are ratified."
            ),
        },
        "duration_coverage": duration_coverage,
        "resource_summary": resource_summary(resources),
        "drift_by_concurrency": {
            concurrency: drift_summary(concurrency_buckets)
            for concurrency, concurrency_buckets in by_concurrency.items()
        },
        "buckets": buckets,
        "clean_followup": clean_summary,
        "claim_boundary": args.claim_boundary,
    }


def bucket_leaf_artifacts_loaded(bucket: dict[str, Any]) -> bool:
    artifacts = bucket.get("leaf_artifacts")
    if isinstance(artifacts, list) and artifacts:
        return all(
            isinstance(artifact, dict)
            and artifact.get("report_loaded") is True
            and bool(artifact.get("sha256"))
            for artifact in artifacts
        )
    return bucket.get("report_loaded") is True and bool(bucket.get("sha256"))


def run_backend(args: argparse.Namespace) -> int:
    validate_run_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "soak_summary.json"
    summary_path.unlink(missing_ok=True)
    for stale in args.out_dir.glob("c*/bucket*.json"):
        stale.unlink()
    (args.out_dir / "clean_followup.json").unlink(missing_ok=True)

    started_wall_s = time.time()
    started_perf = time.perf_counter()
    resources: list[ResourceSample] = [sample_resources(args, started_wall_s)]
    buckets: list[dict[str, Any]] = []
    run_errors: list[dict[str, Any]] = []

    for concurrency in args.concurrency:
        bucket_index = 0
        concurrency_deadline = time.perf_counter() + args.duration_s
        while bucket_index == 0 or time.perf_counter() < concurrency_deadline:
            if args.max_buckets is not None and bucket_index >= args.max_buckets:
                break
            bucket_dir = args.out_dir / f"c{concurrency}"
            bucket_started = sample_resources(args, started_wall_s)
            resources.append(bucket_started)
            bucket_deadline = min(
                concurrency_deadline, time.perf_counter() + args.bucket_s
            )
            leaf_runs: list[dict[str, Any]] = []
            leaf_index = 0
            while leaf_index == 0 or time.perf_counter() < bucket_deadline:
                prompt_index = bucket_index + leaf_index
                prompt_words = args.prompt_words[prompt_index % len(args.prompt_words)]
                max_tokens = args.max_tokens[prompt_index % len(args.max_tokens)]
                out = bucket_dir / f"bucket{bucket_index:04d}_leaf{leaf_index:03d}.json"
                before = sample_resources(args, started_wall_s)
                resources.append(before)
                command = build_leaf_command(
                    args,
                    out=out,
                    concurrency=concurrency,
                    num_requests=args.num_requests,
                    prompt_words=prompt_words,
                    max_tokens=max_tokens,
                    contract_name=args.contract_name,
                    contract_description=args.contract_description,
                )
                completed = subprocess.run(command, check=False)
                after = sample_resources(args, started_wall_s)
                resources.append(after)
                report = read_report(out)
                leaf_runs.append(
                    {
                        "report_path": out,
                        "report": report,
                        "returncode": completed.returncode,
                    }
                )
                if completed.returncode != 0:
                    run_errors.append(
                        {
                            "phase": "leaf",
                            "concurrency": concurrency,
                            "bucket_index": bucket_index,
                            "leaf_index": leaf_index,
                            "returncode": completed.returncode,
                            "artifact": str(out),
                        }
                    )
                    if args.stop_on_failure:
                        break
                leaf_index += 1

            bucket_ended = sample_resources(args, started_wall_s)
            resources.append(bucket_ended)
            record = aggregate_bucket_record(
                bucket_index=bucket_index,
                concurrency=concurrency,
                leaf_runs=leaf_runs,
                resource_before=bucket_started,
                resource_after=bucket_ended,
            )
            buckets.append(record)
            if args.stop_on_failure and record.get("benchmark_returncode") != 0:
                break
            bucket_index += 1
            if time.perf_counter() - started_perf >= args.duration_s * len(
                args.concurrency
            ):
                break

    _followup_report, clean_followup = build_clean_followup(args, started_wall_s)
    if clean_followup is not None:
        resources.append(
            ResourceSample(
                wall_s=float(
                    (clean_followup.get("resource_after") or {}).get("wall_s", 0.0)
                ),
                rss_kib=(clean_followup.get("resource_after") or {}).get("rss_kib"),
                gpu_memory_used_mib=(clean_followup.get("resource_after") or {}).get(
                    "gpu_memory_used_mib", []
                ),
                gpu_memory_scope="device_total",
            )
        )
        if clean_followup.get("benchmark_returncode") != 0:
            run_errors.append(
                {
                    "phase": "clean_followup",
                    "returncode": clean_followup.get("benchmark_returncode"),
                    "artifact": clean_followup.get("artifact"),
                }
            )
    ended_wall_s = time.time()
    summary = build_summary(
        args,
        buckets=buckets,
        resources=resources,
        clean_followup=clean_followup,
        run_errors=run_errors,
        started_wall_s=started_wall_s,
        ended_wall_s=ended_wall_s,
    )
    print(write_json(summary_path, summary))
    return 0 if summary["soak_gate"]["passed"] else 1


def load_summary(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return report


def combined_contract_document(summary: dict[str, Any]) -> str:
    contract = summary.get("contract") or {}
    workload = summary.get("workload") or {}
    comparable_contract = {
        "contract": {
            "name": contract.get("name"),
            "description": contract.get("description"),
            "required_trace_coverage_ratio": contract.get(
                "required_trace_coverage_ratio"
            ),
            "claim_boundary": contract.get("claim_boundary"),
        },
        "workload": workload,
    }
    return json.dumps(comparable_contract, sort_keys=True, separators=(",", ":"))


def build_combined_report(
    model: str, summaries: list[tuple[Path, dict[str, Any]]]
) -> dict[str, Any]:
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    invalid = []
    duplicates = []
    for path, summary in summaries:
        backend = summary.get("backend")
        if (
            summary.get("kind") != "deepseek_v2_lite_http_soak_backend"
            or summary.get("report_intent") != "http_soak"
            or summary.get("model") != model
            or backend not in BACKENDS
        ):
            invalid.append({"artifact": str(path), "errors": ["identity"]})
            continue
        if backend in found:
            duplicates.append({"backend": backend, "artifact": str(path)})
            continue
        found[str(backend)] = (path, summary)
    missing = sorted(set(BACKENDS) - set(found))
    ordered = [found[backend] for backend in sorted(found)]
    commits = [
        str((summary.get("metadata") or {}).get("commit"))
        for _, summary in ordered
        if (summary.get("metadata") or {}).get("commit")
    ]
    commit_counts = value_counts(commits)
    commit_consistent = len(commit_counts) == 1 and len(commits) == len(found)
    provenance_documents = []
    for _, summary in ordered:
        metadata = summary.get("metadata") or {}
        provenance_documents.append(
            json.dumps(
                {
                    "source_revision": metadata.get("source_revision"),
                    "model_revision": metadata.get("model_revision"),
                    "model_fingerprint": metadata.get("model_fingerprint"),
                    "model_path": metadata.get("model_path"),
                    "server_binary_sha256": metadata.get("server_binary_sha256"),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    provenance_consistent = (
        len(set(provenance_documents)) == 1
        and len(provenance_documents) == len(found)
        and bool(provenance_documents)
    )
    contract_documents = [
        combined_contract_document(summary) for _, summary in ordered
    ]
    contract_consistent = (
        len(set(contract_documents)) == 1
        and len(contract_documents) == len(found)
        and bool(contract_documents)
    )
    child_gates = {
        backend: child_gate_passed(summary)
        for backend, (_path, summary) in found.items()
    }
    runtime_boundaries = {}
    missing_runtime_boundaries = []
    for backend, (_path, summary) in found.items():
        metadata = summary.get("metadata") or {}
        runtime_boundary = normalized_runtime_boundary(
            metadata.get("backend_runtime_version")
        )
        if runtime_boundary_is_present(runtime_boundary) and not (
            backend == "nccl" and runtime_boundary.lower() == "nccl"
        ):
            runtime_boundaries[backend] = runtime_boundary
        else:
            missing_runtime_boundaries.append(backend)
    missing_runtime_boundaries = sorted(missing_runtime_boundaries)
    runtime_boundaries_present = (
        len(runtime_boundaries) == len(found)
        and not missing_runtime_boundaries
        and bool(runtime_boundaries)
    )
    passed = (
        not missing
        and not invalid
        and not duplicates
        and all(child_gates.values())
        and commit_consistent
        and provenance_consistent
        and contract_consistent
        and runtime_boundaries_present
    )
    return {
        "schema_version": 1,
        "kind": "deepseek_v2_lite_http_soak_report",
        "report_intent": "http_soak",
        "model": model,
        "backends": list(BACKENDS),
        "metadata": {
            "commit": commits[0] if commit_consistent else None,
            "benchmark_command": artifact_command(sys.argv),
            "child_commits": commit_counts,
            "provenance": json.loads(provenance_documents[0])
            if provenance_consistent
            else None,
            "contract": json.loads(contract_documents[0])
            if contract_consistent
            else None,
            "runtime_boundaries": runtime_boundaries,
        },
        "coverage_gate": {
            "passed": passed,
            "required_backends": len(BACKENDS),
            "retained_backends": len(found),
            "missing": missing,
            "invalid_children": invalid,
            "duplicates": duplicates,
            "child_gates": child_gates,
            "commit_consistent": commit_consistent,
            "provenance_consistent": provenance_consistent,
            "contract_consistent": contract_consistent,
            "runtime_boundaries_present": runtime_boundaries_present,
            "missing_runtime_boundaries": missing_runtime_boundaries,
        },
        "reports": [
            {
                "artifact": str(path),
                "sha256": sha256_file(path),
                "backend": summary["backend"],
                "metadata": summary["metadata"],
                "workload": summary["workload"],
                "summary": summary["summary"],
                "soak_gate": summary["soak_gate"],
                "duration_coverage": summary.get("duration_coverage"),
                "resource_summary": summary["resource_summary"],
                "drift_by_concurrency": summary["drift_by_concurrency"],
                "clean_followup": summary["clean_followup"],
            }
            for path, summary in ordered
        ],
        "claim_boundary": DEFAULT_CLAIM_BOUNDARY,
    }


def child_gate_passed(summary: dict[str, Any]) -> bool:
    gate = summary.get("soak_gate") or {}
    required_true_fields = (
        "passed",
        "bucket_coverage_passed",
        "duration_coverage_passed",
        "leaf_commands_passed",
        "zero_failures",
        "zero_timeouts",
        "trace_coverage_passed",
        "clean_followup_passed",
    )
    return all(gate.get(field) is True for field in required_true_fields)


def combine(args: argparse.Namespace) -> int:
    if any(path.resolve() == args.out.resolve() for path in args.summary):
        raise SystemExit("--out must not overwrite an input --summary")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.unlink(missing_ok=True)
    summaries = [(path, load_summary(path)) for path in args.summary]
    report = build_combined_report(args.model, summaries)
    print(write_json(args.out, report))
    return 0 if report["coverage_gate"]["passed"] else 1


def add_run_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    parser = subparsers.add_parser("run", help="Run one backend soak contract.")
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="DeepSeek-V2-Lite")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--server-command", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--server-binary", type=Path)
    parser.add_argument("--backend-runtime-version", required=True)
    parser.add_argument("--duration-s", type=float, default=1800.0)
    parser.add_argument("--bucket-s", type=float, default=300.0)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--concurrency", type=parse_int_list, default=[4, 8])
    parser.add_argument("--prompt-words", type=parse_int_list, default=[64])
    parser.add_argument("--max-tokens", type=parse_int_list, default=[64])
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-buckets", type=int)
    parser.add_argument("--required-trace-coverage", type=float)
    parser.add_argument(
        "--contract-name",
        default=DEFAULT_CONTRACT_NAME,
        help="Stable contract name written into leaf and summary artifacts.",
    )
    parser.add_argument(
        "--contract-description",
        default=DEFAULT_CONTRACT_DESCRIPTION,
    )
    parser.add_argument("--claim-boundary", default=DEFAULT_CLAIM_BOUNDARY)
    parser.add_argument(
        "--ignore-eos", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--skip-clean-followup", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_run_parser(subparsers)
    combine_parser = subparsers.add_parser(
        "combine", help="Combine host-staged and NCCL soak summaries."
    )
    combine_parser.add_argument("--model", default="DeepSeek-V2-Lite")
    combine_parser.add_argument("--summary", action="append", type=Path, required=True)
    combine_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        args.no_ignore_eos = not args.ignore_eos
        raise SystemExit(run_backend(args))
    raise SystemExit(combine(args))


if __name__ == "__main__":
    main()
