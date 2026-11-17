#!/usr/bin/env python3
"""Run a reproducible HTTP serving sweep over concurrency and max_tokens."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import (
    artifact_command,
    combined_output_hash,
    current_commit,
    detect_hardware_toolchain,
    model_fingerprint,
    numeric_summary,
    repeat_noise_marker,
    sha256_file,
    value_counts,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BENCH = SCRIPT_DIR / "serving.py"
PERCENTILES = ("p50", "p95", "p99")
TRACE_COVERAGE_FIELDS = (
    "traced_requests",
    "attached_server_records",
    "server_error_records",
    "missing_traces",
    "missing_server_records",
    "coverage_ratio",
    "server_record_coverage_ratio",
    "active_set_coverage_ratio",
    "active_set_size_max",
    "decode_batch_coverage_ratio",
    "decode_batch_size_max",
    "token_timing_coverage_ratio",
    "token_timing_mismatches",
    "token_timing_unknown",
)
LEAF_METADATA_FIELDS = (
    "commit",
    "backend",
    "contract_name",
    "model_path",
    "server_command",
    "source_revision",
    "model_revision",
    "model_fingerprint",
    "server_binary_sha256",
    "backend_runtime_version",
    "hardware_toolchain",
)
LEAF_WORKLOAD_FIELDS = (
    "num_requests",
    "warmup",
    "temperature",
    "top_k",
    "top_p",
    "sampling_mode",
    "sampling_profiles",
    "ignore_eos",
    "timeout_s",
    "timeout_kind",
)


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must not be empty")
    parsed = [int(item) for item in items]
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    return parsed


def request_hashes(
    report: dict[str, Any], sampling_label: str | None = None
) -> list[str]:
    return [
        request["output_hash"]
        for request in report["requests"]
        if request["ok"]
        and (sampling_label is None or request.get("sampling_label") == sampling_label)
    ]


def report_hash_manifest_passes(report: dict[str, Any]) -> bool:
    hashes = request_hashes(report)
    summary = report.get("summary") or {}
    return summary.get("output_hash_distribution") == value_counts(
        hashes
    ) and summary.get("combined_output_hash") == combined_output_hash(hashes)


def successful_requests_have_output_evidence(report: dict[str, Any]) -> bool:
    for request in report["requests"]:
        if not request["ok"]:
            continue
        if request.get("max_tokens", 0) <= 0:
            continue
        if not request.get("output_hash") or request.get("output_chunks", 0) <= 0:
            return False
    return True


def sampling_profiles(
    args: argparse.Namespace,
) -> dict[str, dict[str, float | int | str]]:
    if args.sampling_mode == "mixed-greedy-sampled":
        return {
            "greedy": {
                "label": "greedy",
                "temperature": 0.0,
                "top_k": -1,
                "top_p": 1.0,
            },
            "sampled": {
                "label": "sampled",
                "temperature": args.sample_temperature,
                "top_k": args.sample_top_k,
                "top_p": args.sample_top_p,
            },
        }
    return {
        "single": {
            "label": "single",
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        }
    }


def sampling_cli_args(args: argparse.Namespace) -> list[str]:
    return [
        "--sampling-mode",
        args.sampling_mode,
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--sample-temperature",
        str(args.sample_temperature),
        "--sample-top-k",
        str(args.sample_top_k),
        "--sample-top-p",
        str(args.sample_top_p),
    ]


def nested_value(document: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = document
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_nested_value(reports: list[dict[str, Any]], *paths: tuple[str, ...]) -> Any:
    for report in reports:
        for path in paths:
            value = nested_value(report, path)
            if value is not None:
                return value
    return None


def noisy_cell_marker(reports: list[dict[str, Any]]) -> str:
    if any(
        (report.get("metadata") or {}).get("benchmark_returncode") != 0
        for report in reports
    ):
        return "benchmark_error"
    if any(
        report["summary"]["failed"] or report["summary"]["timeouts"]
        for report in reports
    ):
        return "failed_or_timeout"
    return repeat_noise_marker(
        [
            [report["metrics"][metric]["p95_ms"] for report in reports]
            for metric in ("ttft", "tpot", "itl")
        ]
        + [
            [report["summary"]["qps"] for report in reports],
            [report["summary"]["output_tokens_per_s"] for report in reports],
        ],
        len(reports),
    )


def report_trace_gate_passes(
    report: dict[str, Any], required_trace_coverage: float | None
) -> bool:
    if required_trace_coverage is None:
        return True
    trace = report.get("server_trace") or {}
    return all(
        float(trace.get(field) or 0.0) >= required_trace_coverage
        for field in (
            "coverage_ratio",
            "active_set_coverage_ratio",
            "decode_batch_coverage_ratio",
            "token_timing_coverage_ratio",
        )
    )


def metric_repeat_summary(reports: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for percentile in PERCENTILES:
        values = [report["metrics"][metric][f"{percentile}_ms"] for report in reports]
        result[percentile] = values
        result[f"{percentile}_summary"] = numeric_summary(values)
    return result


def trace_coverage(report: dict[str, Any]) -> dict[str, Any]:
    trace = report.get("server_trace") or {}
    return {field: trace.get(field) for field in TRACE_COVERAGE_FIELDS}


def empty_metric_repeat_summary() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for percentile in PERCENTILES:
        result[percentile] = []
        result[f"{percentile}_summary"] = numeric_summary([])
    return result


def empty_trace_coverage() -> dict[str, Any]:
    return {
        "traced_requests": 0,
        "attached_server_records": 0,
        "server_error_records": 0,
        "missing_traces": [],
        "missing_server_records": [],
        "coverage_ratio": 0.0,
        "server_record_coverage_ratio": 0.0,
        "active_set_coverage_ratio": 0.0,
        "active_set_size_max": None,
        "decode_batch_coverage_ratio": 0.0,
        "decode_batch_size_max": None,
        "token_timing_coverage_ratio": 0.0,
        "token_timing_mismatches": [],
        "token_timing_unknown": [],
    }


def prepare_summary_path(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "sweep_summary.json"
    summary_path.unlink(missing_ok=True)
    for cell_path in out_dir.glob("pw*_c*_mt*_r*.json"):
        cell_path.unlink()
    return summary_path


def leaf_artifact_manifest(reports: list[dict[str, Any]]) -> list[dict[str, str]]:
    artifacts = []
    for report in reports:
        artifact = (report.get("metadata") or {}).get("benchmark_artifact")
        if not isinstance(artifact, str):
            continue
        digest = sha256_file(Path(artifact))
        if digest is not None:
            artifacts.append({"artifact": artifact, "sha256": digest})
    return sorted(artifacts, key=lambda item: item["artifact"])


def build_benchmark_command(
    args: argparse.Namespace,
    prompt_words: int | list[int],
    concurrency: int,
    max_tokens: int,
    repeat: int,
) -> tuple[list[str], Path]:
    prompt_words_arg = (
        ",".join(str(value) for value in prompt_words)
        if isinstance(prompt_words, list)
        else str(prompt_words)
    )
    prompt_words_slug = prompt_words_arg.replace(",", "-")
    out = (
        args.out_dir
        / f"pw{prompt_words_slug}_c{concurrency}_mt{max_tokens}_r{repeat}.json"
    )
    cmd = [
        sys.executable,
        str(BENCH),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--num-requests",
        str(args.num_requests),
        "--concurrency",
        str(concurrency),
        "--warmup",
        str(args.warmup),
        "--prompt-words",
        prompt_words_arg,
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(args.temperature),
        "--timeout",
        str(args.timeout),
        "--out",
        str(out),
    ]
    cmd.extend(sampling_cli_args(args))
    optional_values = (
        ("--server-log", args.server_log),
        ("--contract-name", args.contract_name),
        ("--contract-description", args.contract_description),
        ("--backend", args.backend),
        ("--model-path", args.model_path),
        ("--server-command", args.server_command),
        ("--commit", args.commit),
        ("--source-revision", args.source_revision),
        ("--model-revision", args.model_revision),
        ("--server-binary", args.server_binary),
        ("--backend-runtime-version", args.backend_runtime_version),
        ("--claim-boundary", args.claim_boundary),
        ("--required-trace-coverage", args.required_trace_coverage),
    )
    cmd.extend(
        item
        for flag, value in optional_values
        if value is not None
        for item in (flag, str(value))
    )
    if args.no_ignore_eos:
        cmd.append("--no-ignore-eos")
    return cmd, out


def run_one(
    args: argparse.Namespace,
    prompt_words: int | list[int],
    concurrency: int,
    max_tokens: int,
    repeat: int,
) -> dict[str, Any]:
    cmd, out = build_benchmark_command(
        args, prompt_words, concurrency, max_tokens, repeat
    )
    out.unlink(missing_ok=True)
    completed = subprocess.run(cmd, check=False)
    if not out.exists():
        raise FileNotFoundError(
            f"benchmark command exited {completed.returncode} without writing {out}"
        )
    report = json.loads(out.read_text(encoding="utf-8"))
    metadata = report.setdefault("metadata", {})
    metadata["benchmark_returncode"] = completed.returncode
    metadata["benchmark_artifact"] = str(out)
    write_json(out, report)
    return report


def metric_sample_counts_pass(report: dict[str, Any]) -> bool:
    successful = [request for request in report["requests"] if request["ok"]]
    expected = {
        "ttft": sum(request.get("ttft_ms") is not None for request in successful),
        "tpot": sum(request.get("tpot_ms") is not None for request in successful),
        "itl": sum(len(request.get("itl_ms") or []) for request in successful),
    }
    return all(
        report["metrics"][metric]["samples"] == samples
        for metric, samples in expected.items()
    )


def cell_evidence(
    args: argparse.Namespace, reports: list[dict[str, Any]]
) -> dict[str, Any]:
    hash_groups = [request_hashes(report) for report in reports]
    greedy_groups = [
        request_hashes(report, sampling_label="greedy") for report in reports
    ]
    sampled_groups = [
        request_hashes(report, sampling_label="sampled") for report in reports
    ]
    hashes_stable = all(group == hash_groups[0] for group in hash_groups[1:])
    greedy_hashes_stable = all(group == greedy_groups[0] for group in greedy_groups[1:])
    evidence = {
        "zero_failures": all(
            not report["summary"]["failed"] and not report["summary"]["timeouts"]
            for report in reports
        ),
        "hashes_stable": hashes_stable,
        "greedy_hashes_stable": greedy_hashes_stable,
        "greedy_hashes_present": all(bool(group) for group in greedy_groups),
        "sampled_hashes_present": all(bool(group) for group in sampled_groups),
        "output_evidence_present": all(
            successful_requests_have_output_evidence(report) for report in reports
        ),
        "trace_coverage_passed": all(
            report_trace_gate_passes(report, args.required_trace_coverage)
            for report in reports
        ),
        "metric_sample_coverage_passed": all(
            metric_sample_counts_pass(report) for report in reports
        ),
        "benchmark_commands_passed": all(
            (report.get("metadata") or {}).get("benchmark_returncode") == 0
            for report in reports
        ),
        "hash_manifests_consistent": all(
            report_hash_manifest_passes(report) for report in reports
        ),
        "request_output_hashes_by_repeat": hash_groups,
        "greedy_output_hashes_by_repeat": greedy_groups,
        "sampled_output_hashes_by_repeat": sampled_groups,
    }
    single_mode = args.sampling_mode == "single"
    evidence["passed"] = all(
        evidence[field]
        for field in (
            "zero_failures",
            "output_evidence_present",
            "trace_coverage_passed",
            "metric_sample_coverage_passed",
            "benchmark_commands_passed",
            "hash_manifests_consistent",
        )
    ) and (
        hashes_stable
        if single_mode
        else evidence["greedy_hashes_present"]
        and greedy_hashes_stable
        and evidence["sampled_hashes_present"]
    )
    return evidence


def build_cell_row(
    args: argparse.Namespace,
    reports: list[dict[str, Any]],
    backend: str | None,
    prompt_words: int | list[int],
    concurrency: int,
    max_tokens: int,
) -> dict[str, Any]:
    evidence = cell_evidence(args, reports)
    hash_groups = evidence["request_output_hashes_by_repeat"]
    return {
        "backend": backend
        or first_nested_value(
            reports, ("backend",), ("contract", "backend"), ("metadata", "backend")
        ),
        "concurrency": concurrency,
        "prompt_words": prompt_words,
        "max_tokens": max_tokens,
        "repeats": len(reports),
        "tail_sample_sufficient": args.num_requests >= 30,
        "passed": evidence["passed"],
        "noisy_cell": noisy_cell_marker(reports),
        "output_evidence_present": evidence["output_evidence_present"],
        "trace_coverage_passed": evidence["trace_coverage_passed"],
        "metric_sample_coverage_passed": evidence["metric_sample_coverage_passed"],
        "benchmark_commands_passed": evidence["benchmark_commands_passed"],
        "hash_manifests_consistent": evidence["hash_manifests_consistent"],
        "hash_stability_checked": args.sampling_mode == "single",
        "stable_per_request_hashes": evidence["hashes_stable"],
        "greedy_hash_stability_checked": args.sampling_mode == "mixed-greedy-sampled",
        "stable_greedy_hashes": evidence["greedy_hashes_stable"],
        "greedy_hashes_present": evidence["greedy_hashes_present"],
        "sampled_hashes_present": evidence["sampled_hashes_present"],
        "request_output_hashes_by_repeat": hash_groups,
        "greedy_output_hashes_by_repeat": evidence["greedy_output_hashes_by_repeat"],
        "sampled_output_hashes_by_repeat": evidence["sampled_output_hashes_by_repeat"],
        "output_hash_distribution": value_counts(
            [output_hash for group in hash_groups for output_hash in group]
        ),
        "combined_output_hashes": [
            combined_output_hash(group) for group in hash_groups
        ],
        "qps": [report["summary"]["qps"] for report in reports],
        "qps_summary": numeric_summary(
            [report["summary"]["qps"] for report in reports]
        ),
        "input_tokens_per_s": [
            report["summary"]["input_tokens_per_s"] for report in reports
        ],
        "output_tokens_per_s": [
            report["summary"]["output_tokens_per_s"] for report in reports
        ],
        "output_tokens_per_s_summary": numeric_summary(
            [report["summary"]["output_tokens_per_s"] for report in reports]
        ),
        "ttft_ms": metric_repeat_summary(reports, "ttft"),
        "tpot_ms": metric_repeat_summary(reports, "tpot"),
        "itl_ms": metric_repeat_summary(reports, "itl"),
        "ttft_avg_ms": [report["metrics"]["ttft"]["avg_ms"] for report in reports],
        "tpot_avg_ms": [report["metrics"]["tpot"]["avg_ms"] for report in reports],
        "itl_avg_ms": [report["metrics"]["itl"]["avg_ms"] for report in reports],
        "completed": [report["summary"]["completed"] for report in reports],
        "failed": [report["summary"]["failed"] for report in reports],
        "timeouts": [report["summary"]["timeouts"] for report in reports],
        "trace_coverage": [trace_coverage(report) for report in reports],
        "metric_sample_counts": {
            metric: [report["metrics"][metric]["samples"] for report in reports]
            for metric in ("ttft", "tpot", "itl")
        },
        "trace_phases_avg_ms": [
            {
                phase: stats["avg_ms"]
                for phase, stats in (
                    (report.get("server_trace") or {}).get("phases_ms", {})
                ).items()
            }
            for report in reports
        ],
    }


def build_rows(
    args: argparse.Namespace,
    reports: list[dict[str, Any]],
    backend: str | None,
) -> list[dict[str, Any]]:
    cells: dict[tuple[tuple[int, ...], int, int], list[dict[str, Any]]] = {}
    for report in reports:
        workload = report["workload"]
        raw_prompt_words = workload["prompt_words"]
        prompt_word_key = (
            tuple(raw_prompt_words)
            if isinstance(raw_prompt_words, list)
            else (int(raw_prompt_words),)
        )
        key = (prompt_word_key, workload["concurrency"], workload["max_tokens"])
        cells.setdefault(key, []).append(report)

    rows = []
    for (prompt_word_key, concurrency, max_tokens), cell_reports in sorted(
        cells.items()
    ):
        prompt_words: int | list[int] = (
            list(prompt_word_key) if len(prompt_word_key) > 1 else prompt_word_key[0]
        )
        rows.append(
            build_cell_row(
                args,
                cell_reports,
                backend,
                prompt_words,
                concurrency,
                max_tokens,
            )
        )
    return rows


def build_summary(
    args: argparse.Namespace, reports: list[dict[str, Any]]
) -> dict[str, Any]:
    summary_backend = args.backend or first_nested_value(
        reports,
        ("backend",),
        ("contract", "backend"),
        ("metadata", "backend"),
    )
    summary_contract_name = args.contract_name or first_nested_value(
        reports,
        ("contract", "name"),
        ("metadata", "contract_name"),
    )
    summary_claim_boundary = args.claim_boundary or first_nested_value(
        reports,
        ("contract", "claim_boundary"),
    )
    metric_definitions = first_nested_value(reports, ("metric_definitions",)) or {}
    metric_definitions_consistent = bool(reports) and all(
        report.get("metric_definitions") == metric_definitions for report in reports
    )
    latency_budget = first_nested_value(reports, ("latency_budget",)) or {}
    latency_budget_consistent = bool(reports) and all(
        report.get("latency_budget") == latency_budget for report in reports
    )
    rows = build_rows(args, reports, summary_backend)

    correctness_ok = (
        metric_definitions_consistent
        and latency_budget_consistent
        and all(row["passed"] for row in rows)
    )

    return {
        "schema_version": 1,
        "kind": "openai_http_completions_sweep",
        "report_intent": "http_serving_slo"
        if summary_contract_name
        else "http_serving_benchmark",
        "base_url": args.base_url,
        "model": args.model,
        "backend": summary_backend,
        "metadata": {
            "commit": args.commit or current_commit(),
            "backend": summary_backend,
            "contract_name": summary_contract_name,
            "model_path": args.model_path,
            "server_command": args.server_command,
            "source_revision": args.source_revision,
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
            "name": summary_contract_name,
            "backend": summary_backend,
            "description": args.contract_description,
            "required_trace_coverage_ratio": args.required_trace_coverage,
            "claim_boundary": summary_claim_boundary
            or "HTTP streaming sweep evidence only; direct, profiler, soak, and production-readiness claims require separate artifacts.",
        },
        "metric_definitions": metric_definitions,
        "latency_budget": latency_budget,
        "workload": {
            "num_requests": args.num_requests,
            "warmup": args.warmup,
            "prompt_words": args.prompt_words,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "sampling_mode": args.sampling_mode,
            "sampling_profiles": sampling_profiles(args),
            "ignore_eos": not args.no_ignore_eos,
            "timeout_s": args.timeout,
            "timeout_kind": "absolute_request_deadline",
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "tail_sample_sufficient": args.num_requests >= 30,
        },
        "correctness_gate": {
            "passed": correctness_ok,
            "metric_definitions_consistent": metric_definitions_consistent,
            "latency_budget_consistent": latency_budget_consistent,
            "rule": (
                "all modes: failed=0, timeout=0, successful requests have output evidence, and retained "
                "profiles meet request/active-set/decode-batch trace coverage; single mode also requires "
                "the per-request output_hash list to stay stable across repeats for each cell; "
                "mixed-greedy-sampled mode requires "
                "both greedy and sampled requests are present; greedy output_hash lists stay "
                "stable across repeats, while sampled output hashes are reported but not required stable"
            ),
        },
        "leaf_artifacts": leaf_artifact_manifest(reports),
        "rows": rows,
    }


def verify_summary_leaf_artifacts(summary: dict[str, Any]) -> dict[str, Any]:
    manifest = summary.get("leaf_artifacts")
    if not isinstance(manifest, list):
        return {"passed": False, "checked": 0, "failures": ["leaf_artifacts"]}

    failures: list[str] = []
    reports: list[dict[str, Any]] = []
    seen_paths = set()
    summary_metadata = summary.get("metadata")
    summary_workload = summary.get("workload")
    summary_contract = summary.get("contract")
    if not isinstance(summary_metadata, dict):
        failures.append("metadata")
        summary_metadata = {}
    if not isinstance(summary_workload, dict):
        failures.append("workload")
        summary_workload = {}
    if not isinstance(summary_contract, dict):
        failures.append("contract")
        summary_contract = {}

    for index, item in enumerate(manifest):
        prefix = f"leaf_artifacts[{index}]"
        if not isinstance(item, dict):
            failures.append(prefix)
            continue
        raw_path = item.get("artifact")
        expected_digest = item.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            failures.append(f"{prefix}.artifact")
            continue
        if raw_path in seen_paths:
            failures.append(f"{prefix}.duplicate")
            continue
        seen_paths.add(raw_path)
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            failures.append(f"{prefix}.sha256")
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append(f"{prefix}.json")
            continue
        if not isinstance(report, dict):
            failures.append(f"{prefix}.document")
            continue
        reports.append(report)

        if report.get("schema_version") != 1:
            failures.append(f"{prefix}.schema_version")
        if report.get("kind") != "openai_http_completions_stream_benchmark":
            failures.append(f"{prefix}.kind")
        for field in ("report_intent", "base_url", "model", "backend"):
            if report.get(field) != summary.get(field):
                failures.append(f"{prefix}.{field}")
        if report.get("contract") != summary_contract:
            failures.append(f"{prefix}.contract")
        if report.get("metric_definitions") != summary.get("metric_definitions"):
            failures.append(f"{prefix}.metric_definitions")
        if report.get("latency_budget") != summary.get("latency_budget"):
            failures.append(f"{prefix}.latency_budget")

        metadata = report.get("metadata")
        if not isinstance(metadata, dict):
            failures.append(f"{prefix}.metadata")
        else:
            for field in LEAF_METADATA_FIELDS:
                if metadata.get(field) != summary_metadata.get(field):
                    failures.append(f"{prefix}.metadata.{field}")
            if metadata.get("benchmark_artifact") != raw_path:
                failures.append(f"{prefix}.metadata.benchmark_artifact")
            if metadata.get("benchmark_returncode") != 0:
                failures.append(f"{prefix}.metadata.benchmark_returncode")

        workload = report.get("workload")
        if not isinstance(workload, dict):
            failures.append(f"{prefix}.workload")
        else:
            for field in LEAF_WORKLOAD_FIELDS:
                if workload.get(field) != summary_workload.get(field):
                    failures.append(f"{prefix}.workload.{field}")
            if workload.get("concurrency") not in summary_workload.get(
                "concurrency", []
            ):
                failures.append(f"{prefix}.workload.concurrency")
            if workload.get("max_tokens") not in summary_workload.get("max_tokens", []):
                failures.append(f"{prefix}.workload.max_tokens")

    expected_count = sum(
        row.get("repeats", 0)
        for row in summary.get("rows", [])
        if isinstance(row, dict) and isinstance(row.get("repeats"), int)
    )
    if len(manifest) != expected_count:
        failures.append("leaf_artifacts.count")

    if reports:
        verification_args = argparse.Namespace(
            sampling_mode=summary_workload.get("sampling_mode"),
            num_requests=summary_workload.get("num_requests"),
            required_trace_coverage=summary_contract.get(
                "required_trace_coverage_ratio"
            ),
        )
        try:
            rebuilt_rows = build_rows(
                verification_args, reports, summary.get("backend")
            )
        except (KeyError, TypeError, ValueError):
            failures.append("rows.rebuild")
        else:
            if rebuilt_rows != summary.get("rows"):
                failures.append("rows.recomputed")

    return {
        "passed": bool(reports) and not failures,
        "checked": len(reports),
        "failures": list(dict.fromkeys(failures)),
    }


def build_startup_failure_summary(
    args: argparse.Namespace, error: str
) -> dict[str, Any]:
    if args.mixed_prompt_shape and len(args.prompt_words) > 1:
        prompt_word_cells: list[int | list[int]] = [args.prompt_words]
    else:
        prompt_word_cells = args.prompt_words

    rows = []
    for prompt_words in prompt_word_cells:
        for max_tokens in args.max_tokens:
            for concurrency in args.concurrency:
                rows.append(
                    {
                        "backend": args.backend,
                        "concurrency": concurrency,
                        "prompt_words": prompt_words,
                        "max_tokens": max_tokens,
                        "repeats": args.repeats,
                        "tail_sample_sufficient": args.num_requests >= 30,
                        "passed": False,
                        "noisy_cell": "startup_failure",
                        "output_evidence_present": False,
                        "trace_coverage_passed": False,
                        "metric_sample_coverage_passed": False,
                        "benchmark_commands_passed": False,
                        "hash_manifests_consistent": False,
                        "hash_stability_checked": args.sampling_mode == "single",
                        "stable_per_request_hashes": False,
                        "greedy_hash_stability_checked": args.sampling_mode
                        == "mixed-greedy-sampled",
                        "stable_greedy_hashes": False,
                        "greedy_hashes_present": False,
                        "sampled_hashes_present": False,
                        "output_hash_distribution": {},
                        "request_output_hashes_by_repeat": [],
                        "greedy_output_hashes_by_repeat": [],
                        "sampled_output_hashes_by_repeat": [],
                        "combined_output_hashes": [],
                        "qps": [],
                        "qps_summary": numeric_summary([]),
                        "input_tokens_per_s": [],
                        "output_tokens_per_s": [],
                        "output_tokens_per_s_summary": numeric_summary([]),
                        "ttft_ms": empty_metric_repeat_summary(),
                        "tpot_ms": empty_metric_repeat_summary(),
                        "itl_ms": empty_metric_repeat_summary(),
                        "ttft_avg_ms": [],
                        "tpot_avg_ms": [],
                        "itl_avg_ms": [],
                        "completed": [0 for _ in range(args.repeats)],
                        "failed": [args.num_requests for _ in range(args.repeats)],
                        "timeouts": [0 for _ in range(args.repeats)],
                        "trace_coverage": [
                            empty_trace_coverage() for _ in range(args.repeats)
                        ],
                        "metric_sample_counts": {
                            "ttft": [0 for _ in range(args.repeats)],
                            "tpot": [0 for _ in range(args.repeats)],
                            "itl": [0 for _ in range(args.repeats)],
                        },
                        "trace_phases_avg_ms": [{} for _ in range(args.repeats)],
                    }
                )

    return {
        "schema_version": 1,
        "kind": "openai_http_completions_sweep",
        "report_intent": "http_serving_slo_startup_failure"
        if args.contract_name
        else "http_serving_benchmark_startup_failure",
        "base_url": args.base_url,
        "model": args.model,
        "backend": args.backend,
        "metadata": {
            "commit": args.commit or current_commit(),
            "backend": args.backend,
            "contract_name": args.contract_name,
            "model_path": args.model_path,
            "server_command": args.server_command,
            "source_revision": args.source_revision,
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
            "claim_boundary": args.claim_boundary
            or "HTTP streaming sweep evidence only; direct, profiler, soak, and production-readiness claims require separate artifacts.",
        },
        "metric_definitions": {
            "percentile_method": "R7 linear interpolation over sorted samples",
            "ttft": "request start to first non-empty streamed text chunk",
            "tpot": "requires streamed text chunk count to equal server completion_tokens",
            "itl": "requires streamed text chunk count to equal server completion_tokens",
        },
        "latency_budget": {
            "configured": False,
            "passed": None,
            "reason": "The server did not start, and this retained contract does not define a production latency budget.",
        },
        "workload": {
            "num_requests": args.num_requests,
            "warmup": args.warmup,
            "prompt_words": args.prompt_words,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "sampling_mode": args.sampling_mode,
            "sampling_profiles": sampling_profiles(args),
            "ignore_eos": not args.no_ignore_eos,
            "timeout_s": args.timeout,
            "timeout_kind": "absolute_request_deadline",
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "tail_sample_sufficient": args.num_requests >= 30,
        },
        "startup_failure": {
            "error": error,
            "server_log": str(args.server_log) if args.server_log else None,
        },
        "correctness_gate": {
            "passed": False,
            "rule": "server must start before retained HTTP contract requests can be issued",
        },
        "leaf_artifacts": [],
        "rows": rows,
        "run_errors": [
            {
                "phase": "server_startup",
                "error": error,
            }
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--prompt-words", type=parse_int_list, default=[16])
    parser.add_argument(
        "--mixed-prompt-shape",
        action="store_true",
        help="Treat all --prompt-words values as one alternating mixed-shape cell.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--sampling-mode",
        choices=["single", "mixed-greedy-sampled"],
        default="single",
    )
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-k", type=int, default=40)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--no-ignore-eos", action="store_true")
    parser.add_argument("--concurrency", type=parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--max-tokens", type=parse_int_list, default=[16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--contract-name")
    parser.add_argument("--contract-description")
    parser.add_argument("--claim-boundary")
    parser.add_argument("--required-trace-coverage", type=float)
    parser.add_argument("--backend")
    parser.add_argument("--model-path")
    parser.add_argument("--server-command")
    parser.add_argument("--commit")
    parser.add_argument("--source-revision")
    parser.add_argument("--model-revision")
    parser.add_argument("--server-binary", type=Path)
    parser.add_argument("--backend-runtime-version")
    parser.add_argument(
        "--record-startup-failure",
        help="Write a retained sweep_summary.json for a server startup failure without issuing HTTP requests.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.contract_name and args.backend is None:
        raise SystemExit("--backend is required when --contract-name is set")
    if args.required_trace_coverage is not None:
        if args.required_trace_coverage <= 0.0 or args.required_trace_coverage > 1.0:
            raise SystemExit("--required-trace-coverage must be in (0, 1]")
        if args.server_log is None:
            raise SystemExit("--server-log is required with --required-trace-coverage")

    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.timeout <= 0.0:
        raise SystemExit("--timeout must be positive")
    if args.top_p <= 0.0 or args.top_p > 1.0:
        raise SystemExit("--top-p must be in (0, 1]")
    if args.sample_top_p <= 0.0 or args.sample_top_p > 1.0:
        raise SystemExit("--sample-top-p must be in (0, 1]")
    if args.sampling_mode == "mixed-greedy-sampled" and args.sample_temperature <= 0.0:
        raise SystemExit(
            "--sample-temperature must be positive in mixed-greedy-sampled mode"
        )


def prompt_word_cells(args: argparse.Namespace) -> list[int | list[int]]:
    if args.mixed_prompt_shape and len(args.prompt_words) > 1:
        return [args.prompt_words]
    return args.prompt_words


def run_sweep(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reports = []
    run_errors = []
    for prompt_words in prompt_word_cells(args):
        for max_tokens in args.max_tokens:
            for concurrency in args.concurrency:
                for repeat in range(args.repeats):
                    try:
                        reports.append(
                            run_one(args, prompt_words, concurrency, max_tokens, repeat)
                        )
                    except Exception as exc:  # noqa: BLE001 - retain partial sweep.
                        run_errors.append(
                            {
                                "prompt_words": prompt_words,
                                "concurrency": concurrency,
                                "max_tokens": max_tokens,
                                "repeat": repeat,
                                "error": str(exc),
                            }
                        )
    return reports, run_errors


def main() -> None:
    args = parse_args()
    validate_args(args)
    summary_path = prepare_summary_path(args.out_dir)
    if args.record_startup_failure:
        summary = build_startup_failure_summary(args, args.record_startup_failure)
        print(write_json(summary_path, summary))
        raise SystemExit(1)

    reports, run_errors = run_sweep(args)
    summary = build_summary(args, reports)
    summary["run_errors"] = run_errors
    if run_errors:
        summary["correctness_gate"]["passed"] = False
    print(write_json(summary_path, summary))
    if not summary["correctness_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
