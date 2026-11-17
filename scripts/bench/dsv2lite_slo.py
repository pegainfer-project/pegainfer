#!/usr/bin/env python3
"""Run and combine retained DeepSeek-V2-Lite HTTP SLO contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common import (
    artifact_command,
    combined_output_hash,
    numeric_summary,
    repeat_noise_marker,
    value_counts,
    write_json,
)
from sweep import verify_summary_leaf_artifacts


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SWEEP = SCRIPT_DIR / "sweep.py"
BACKENDS = ("host-staged", "nccl")
PROFILE_SCHEMA_VERSION = 1
NCCL_VERSION_PATTERNS = (
    re.compile(
        r"DeepSeek-V2-Lite NCCL backend loaded: version=([0-9]+\.[0-9]+\.[0-9]+)"
    ),
    re.compile(r"NCCL version ([0-9]+\.[0-9]+\.[0-9]+)"),
)
REPORT_CLAIM_BOUNDARY = (
    "Retained HTTP pressure/SLO evidence for the six fixed DeepSeek-V2-Lite "
    "host-staged/NCCL contracts. This report is not direct decode attribution, "
    "sustained soak, vLLM parity, multi-node recovery, or production-readiness evidence."
)


@dataclass(frozen=True)
class SloProfile:
    name: str
    description: str
    prompt_words: tuple[int, ...]
    max_tokens: tuple[int, ...]
    num_requests: int
    concurrency: tuple[int, ...]
    repeats: int
    warmup: int
    timeout_s: float
    mixed_prompt_shape: bool
    required_trace_coverage_ratio: float
    claim_boundary: str

    def workload_contract(self) -> dict[str, Any]:
        return {
            "num_requests": self.num_requests,
            "warmup": self.warmup,
            "prompt_words": list(self.prompt_words),
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 1.0,
            "sampling_mode": "single",
            "ignore_eos": True,
            "timeout_s": self.timeout_s,
            "timeout_kind": "absolute_request_deadline",
            "concurrency": list(self.concurrency),
            "max_tokens": list(self.max_tokens),
            "repeats": self.repeats,
        }


PROFILES = {
    profile.name: profile
    for profile in (
        SloProfile(
            name="dsv2-lite-short-decode-heavy",
            description="DeepSeek-V2-Lite retained short decode-heavy HTTP contract.",
            prompt_words=(64,),
            max_tokens=(64,),
            num_requests=32,
            concurrency=(1, 4, 8),
            repeats=3,
            warmup=0,
            timeout_s=240.0,
            mixed_prompt_shape=False,
            required_trace_coverage_ratio=1.0,
            claim_boundary=(
                "HTTP pressure/SLO evidence for a fixed short decode-heavy contract; "
                "not a direct decode attribution result, soak result, or production-readiness claim."
            ),
        ),
        SloProfile(
            name="dsv2-lite-mixed-prompt-shape",
            description="DeepSeek-V2-Lite retained mixed short/long prompt HTTP contract.",
            prompt_words=(64, 512),
            max_tokens=(64,),
            num_requests=32,
            concurrency=(1, 4, 8),
            repeats=3,
            warmup=0,
            timeout_s=240.0,
            mixed_prompt_shape=True,
            required_trace_coverage_ratio=1.0,
            claim_boundary=(
                "HTTP mixed-shape SLO evidence; trace coverage and tails may identify long-prefill "
                "risk but this does not close sustained soak or production-readiness gates."
            ),
        ),
        SloProfile(
            name="dsv2-lite-long-prompt-smoke",
            description="DeepSeek-V2-Lite retained long-prompt HTTP smoke contract.",
            prompt_words=(2048,),
            max_tokens=(64,),
            num_requests=1,
            concurrency=(1,),
            repeats=1,
            warmup=0,
            timeout_s=900.0,
            mixed_prompt_shape=False,
            required_trace_coverage_ratio=1.0,
            claim_boundary=(
                "Long-prompt HTTP smoke evidence only. Use it to retain a boundary row and route "
                "findings to long-prefill work; do not treat it as broad long-context or soak proof."
            ),
        ),
    )
}


def profile_spec_sha256() -> str:
    payload = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profiles": {
            name: asdict(profile) for name, profile in sorted(PROFILES.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_commit(commit: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{12}|[0-9a-f]{40}", commit):
        raise SystemExit(
            "--commit must be a 12- or 40-character lowercase Git object id"
        )
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(
            "retained runs require a Git worktree with a valid HEAD"
        ) from exc
    if commit not in {head, head[:12]}:
        raise SystemExit(f"--commit {commit} does not match current HEAD {head}")


def csv(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def detect_backend_runtime_version(server_log: Path, backend: str) -> str:
    if backend == "host-staged":
        return "host-staged"
    try:
        log_text = server_log.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SystemExit(f"cannot read --server-log for NCCL version: {exc}") from exc
    loaded_versions = NCCL_VERSION_PATTERNS[0].findall(log_text)
    if loaded_versions:
        unique_loaded = set(loaded_versions)
        if len(unique_loaded) != 1:
            raise SystemExit(
                f"--server-log contains multiple loaded NCCL versions: {sorted(unique_loaded)}"
            )
        return loaded_versions[-1]
    fallback_versions = set(NCCL_VERSION_PATTERNS[1].findall(log_text))
    if not fallback_versions:
        raise SystemExit("--server-log does not contain the loaded NCCL version")
    if len(fallback_versions) != 1:
        raise SystemExit(
            f"--server-log contains multiple NCCL INFO versions: {sorted(fallback_versions)}"
        )
    return next(iter(fallback_versions))


def startup_failure_runtime_version(error: str, backend: str) -> str:
    if backend == "host-staged":
        return "startup-failed"
    match = re.search(r"loaded ([0-9]+\.[0-9]+\.[0-9]+)", error)
    return match.group(1) if match else "unavailable-before-init"


def build_sweep_command(args: argparse.Namespace) -> list[str]:
    profile = PROFILES[args.profile]
    backend_runtime_version = getattr(args, "backend_runtime_version", None)
    if backend_runtime_version is None:
        if args.record_startup_failure:
            backend_runtime_version = startup_failure_runtime_version(
                args.record_startup_failure, args.backend
            )
        else:
            backend_runtime_version = detect_backend_runtime_version(
                args.server_log, args.backend
            )
    command = [
        sys.executable,
        str(SWEEP),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--num-requests",
        str(profile.num_requests),
        "--warmup",
        str(profile.warmup),
        "--prompt-words",
        csv(profile.prompt_words),
        "--max-tokens",
        csv(profile.max_tokens),
        "--concurrency",
        csv(profile.concurrency),
        "--repeats",
        str(profile.repeats),
        "--timeout",
        str(profile.timeout_s),
        "--server-log",
        str(args.server_log),
        "--contract-name",
        profile.name,
        "--contract-description",
        profile.description,
        "--claim-boundary",
        profile.claim_boundary,
        "--required-trace-coverage",
        str(profile.required_trace_coverage_ratio),
        "--backend",
        args.backend,
        "--model-path",
        args.model_path,
        "--server-command",
        args.server_command,
        "--source-revision",
        args.commit,
        "--model-revision",
        args.model_revision,
        "--server-binary",
        str(args.server_binary),
        "--backend-runtime-version",
        backend_runtime_version,
        "--out-dir",
        str(args.out_dir),
    ]
    if profile.mixed_prompt_shape:
        command.append("--mixed-prompt-shape")
    if args.commit:
        command.extend(["--commit", args.commit])
    if args.record_startup_failure:
        command.extend(["--record-startup-failure", args.record_startup_failure])
    return command


def expected_rows(profile: SloProfile) -> set[tuple[str, int, int, int]]:
    prompt_shape = csv(profile.prompt_words)
    return {
        (prompt_shape, concurrency, max_tokens, profile.repeats)
        for concurrency in profile.concurrency
        for max_tokens in profile.max_tokens
    }


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_numeric_summary(
    value: Any,
    path: str,
    expected_samples: int,
    raw_values: Any | None = None,
) -> list[str]:
    if not isinstance(value, dict):
        return [path]
    errors = []
    if value.get("samples") != expected_samples:
        errors.append(f"{path}.samples")
    samples = [value.get(field) for field in ("min", "median", "max")]
    if not all(is_finite_number(sample) and float(sample) >= 0.0 for sample in samples):
        errors.append(f"{path}.values")
    elif not float(samples[0]) <= float(samples[1]) <= float(samples[2]):
        errors.append(f"{path}.ordering")
    if raw_values is not None:
        if (
            not isinstance(raw_values, list)
            or len(raw_values) != expected_samples
            or not all(
                is_finite_number(sample) and float(sample) >= 0.0
                for sample in raw_values
            )
        ):
            errors.append(f"{path}.raw_values")
        else:
            expected = numeric_summary(raw_values)
            for field in ("min", "median", "max"):
                actual_value = value.get(field)
                expected_value = expected[field]
                if (
                    not is_finite_number(actual_value)
                    or not is_finite_number(expected_value)
                    or not math.isclose(
                        float(actual_value),
                        float(expected_value),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    errors.append(f"{path}.{field}_mismatch")
    return list(dict.fromkeys(errors))


def validate_metric_summary(value: Any, path: str, repeats: int) -> list[str]:
    if not isinstance(value, dict):
        return [path]
    errors = []
    for percentile in ("p50", "p95", "p99"):
        samples = value.get(percentile)
        if (
            not isinstance(samples, list)
            or len(samples) != repeats
            or not all(
                is_finite_number(sample) and float(sample) >= 0.0 for sample in samples
            )
        ):
            errors.append(f"{path}.{percentile}")
        errors.extend(
            validate_numeric_summary(
                value.get(f"{percentile}_summary"),
                f"{path}.{percentile}_summary",
                repeats,
                samples,
            )
        )
    percentile_samples = [value.get(percentile) for percentile in ("p50", "p95", "p99")]
    if all(
        isinstance(samples, list) and len(samples) == repeats
        for samples in percentile_samples
    ):
        for sample_index in range(repeats):
            values = [samples[sample_index] for samples in percentile_samples]
            if all(is_finite_number(sample) for sample in values) and not (
                float(values[0]) <= float(values[1]) <= float(values[2])
            ):
                errors.append(f"{path}.percentile_ordering")
    return errors


def validate_output_hash_evidence(
    row: dict[str, Any], profile: SloProfile
) -> list[str]:
    prefix = "rows.success"
    groups = row.get("request_output_hashes_by_repeat")
    if (
        not isinstance(groups, list)
        or len(groups) != profile.repeats
        or any(
            not isinstance(group, list)
            or len(group) != profile.num_requests
            or any(not isinstance(value, str) or not value for value in group)
            for group in groups
        )
    ):
        return [f"{prefix}.request_output_hashes_by_repeat"]

    stable = all(group == groups[0] for group in groups[1:])
    errors = []
    if row.get("stable_per_request_hashes") is not stable or not stable:
        errors.append(f"{prefix}.stable_per_request_hashes")
    flattened = [value for group in groups for value in group]
    if row.get("output_hash_distribution") != value_counts(flattened):
        errors.append(f"{prefix}.output_hash_distribution")
    expected_combined = [combined_output_hash(group) for group in groups]
    if row.get("combined_output_hashes") != expected_combined:
        errors.append(f"{prefix}.combined_output_hashes")
    if row.get("hash_manifests_consistent") is not True:
        errors.append(f"{prefix}.hash_manifests_consistent")
    return errors


def validate_success_flags(
    row: dict[str, Any], backend: str, profile: SloProfile
) -> list[str]:
    prefix = "rows.success"
    errors = []
    if row.get("backend") != backend:
        errors.append(f"{prefix}.backend")
    if row.get("repeats") != profile.repeats:
        errors.append(f"{prefix}.repeats")
    if row.get("passed") is not True:
        errors.append(f"{prefix}.passed")
    if row.get("tail_sample_sufficient") != (profile.num_requests >= 30):
        errors.append(f"{prefix}.tail_sample_sufficient")
    for field in (
        "output_evidence_present",
        "trace_coverage_passed",
        "metric_sample_coverage_passed",
        "benchmark_commands_passed",
        "hash_stability_checked",
        "stable_per_request_hashes",
        "hash_manifests_consistent",
    ):
        if row.get(field) is not True:
            errors.append(f"{prefix}.{field}")

    expected_marker = repeat_noise_marker(
        [
            row.get(metric, {}).get("p95", [])
            for metric in ("ttft_ms", "tpot_ms", "itl_ms")
        ]
        + [row.get("qps", []), row.get("output_tokens_per_s", [])],
        profile.repeats,
    )
    if row.get("noisy_cell") != expected_marker:
        errors.append(f"{prefix}.noisy_cell")

    for field, expected in (
        ("completed", profile.num_requests),
        ("failed", 0),
        ("timeouts", 0),
    ):
        values = row.get(field)
        if not isinstance(values, list) or values != [expected] * profile.repeats:
            errors.append(f"{prefix}.{field}")

    return errors


def validate_success_metrics(row: dict[str, Any], profile: SloProfile) -> list[str]:
    prefix = "rows.success"
    errors = validate_numeric_summary(
        row.get("qps_summary"),
        f"{prefix}.qps_summary",
        profile.repeats,
        row.get("qps"),
    )
    errors.extend(
        validate_numeric_summary(
            row.get("output_tokens_per_s_summary"),
            f"{prefix}.output_tokens_per_s_summary",
            profile.repeats,
            row.get("output_tokens_per_s"),
        )
    )
    for metric in ("ttft_ms", "tpot_ms", "itl_ms"):
        errors.extend(
            validate_metric_summary(
                row.get(metric), f"{prefix}.{metric}", profile.repeats
            )
        )

    sample_counts = row.get("metric_sample_counts")
    expected_tpot_samples = profile.num_requests if row.get("max_tokens", 0) > 1 else 0
    expected_counts = {
        "ttft": [profile.num_requests] * profile.repeats,
        "tpot": [expected_tpot_samples] * profile.repeats,
        "itl": [profile.num_requests * max(0, int(row.get("max_tokens", 0)) - 1)]
        * profile.repeats,
    }
    if not isinstance(sample_counts, dict) or any(
        sample_counts.get(metric) != expected
        for metric, expected in expected_counts.items()
    ):
        errors.append(f"{prefix}.metric_sample_counts")

    return errors


def validate_trace_evidence(row: dict[str, Any], profile: SloProfile) -> list[str]:
    prefix = "rows.success"
    traces = row.get("trace_coverage")
    if not isinstance(traces, list) or len(traces) != profile.repeats:
        return [f"{prefix}.trace_coverage"]
    errors = []
    for trace in traces:
        errors.extend(validate_trace_record(trace, profile, prefix))
    return errors


def validate_trace_record(trace: Any, profile: SloProfile, prefix: str) -> list[str]:
    if not isinstance(trace, dict):
        return [f"{prefix}.trace_coverage"]
    errors = []
    expected_values = {
        "traced_requests": profile.num_requests,
        "server_error_records": 0,
        "missing_traces": [],
        "missing_server_records": [],
        "token_timing_mismatches": [],
        "token_timing_unknown": [],
    }
    errors.extend(
        f"{prefix}.trace_coverage.{field}"
        for field, expected in expected_values.items()
        if trace.get(field) != expected
    )
    coverage_fields = (
        "coverage_ratio",
        "active_set_coverage_ratio",
        "decode_batch_coverage_ratio",
        "token_timing_coverage_ratio",
    )
    errors.extend(
        f"{prefix}.trace_coverage.{field}"
        for field in coverage_fields
        if not is_finite_number(trace.get(field))
        or float(trace[field]) < profile.required_trace_coverage_ratio
    )
    errors.extend(
        f"{prefix}.trace_coverage.{field}"
        for field in ("active_set_size_max", "decode_batch_size_max")
        if not isinstance(trace.get(field), int)
        or isinstance(trace.get(field), bool)
        or trace[field] <= 0
    )
    return errors


def validate_success_row(
    row: dict[str, Any], backend: str, profile: SloProfile
) -> list[str]:
    return list(
        dict.fromkeys(
            validate_success_flags(row, backend, profile)
            + validate_success_metrics(row, profile)
            + validate_trace_evidence(row, profile)
            + validate_output_hash_evidence(row, profile)
        )
    )


def validate_child_identity(
    summary: dict[str, Any], model: str, backend: str, profile: SloProfile
) -> list[str]:
    errors = []
    contract = summary.get("contract")
    if not isinstance(contract, dict):
        errors.append("contract")
        contract = {}
    if summary.get("schema_version") != 1:
        errors.append("schema_version")
    if summary.get("kind") != "openai_http_completions_sweep":
        errors.append("kind")
    if summary.get("model") != model:
        errors.append("model")
    if summary.get("backend") != backend or contract.get("backend") != backend:
        errors.append("backend")
    if contract.get("name") != profile.name:
        errors.append("contract.name")
    if contract.get("description") != profile.description:
        errors.append("contract.description")
    if contract.get("claim_boundary") != profile.claim_boundary:
        errors.append("contract.claim_boundary")
    if (
        contract.get("required_trace_coverage_ratio")
        != profile.required_trace_coverage_ratio
    ):
        errors.append("contract.required_trace_coverage_ratio")

    return errors


def validate_child_workload(summary: dict[str, Any], profile: SloProfile) -> list[str]:
    errors = []
    workload = summary.get("workload")
    if not isinstance(workload, dict):
        errors.append("workload")
        workload = {}
    for field, expected in profile.workload_contract().items():
        if workload.get(field) != expected:
            errors.append(f"workload.{field}")
    expected_sampling_profiles = {
        "single": {
            "label": "single",
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 1.0,
        }
    }
    if workload.get("sampling_profiles") != expected_sampling_profiles:
        errors.append("workload.sampling_profiles")
    return errors


def validate_child_metric_contract(summary: dict[str, Any]) -> list[str]:
    errors = []
    metric_definitions = summary.get("metric_definitions")
    if (
        not isinstance(metric_definitions, dict)
        or metric_definitions.get("percentile_method")
        != "R7 linear interpolation over sorted samples"
        or "completion_tokens" not in str(metric_definitions.get("tpot"))
        or "completion_tokens" not in str(metric_definitions.get("itl"))
    ):
        errors.append("metric_definitions")
    latency_budget = summary.get("latency_budget")
    if (
        not isinstance(latency_budget, dict)
        or latency_budget.get("configured") is not False
        or latency_budget.get("passed") is not None
        or not isinstance(latency_budget.get("reason"), str)
        or not latency_budget["reason"]
    ):
        errors.append("latency_budget")
    return errors


def validate_child_leaf_artifacts(
    summary: dict[str, Any], profile: SloProfile
) -> list[str]:
    artifacts = summary.get("leaf_artifacts")
    expected_count = len(expected_rows(profile)) * profile.repeats
    if not isinstance(artifacts, list) or len(artifacts) != expected_count:
        return ["leaf_artifacts"]
    paths = []
    errors = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            errors.append("leaf_artifacts.items")
            continue
        path = artifact.get("artifact")
        digest = artifact.get("sha256")
        if not isinstance(path, str) or not path:
            errors.append("leaf_artifacts.artifact")
        else:
            paths.append(path)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append("leaf_artifacts.sha256")
    if len(paths) != len(set(paths)):
        errors.append("leaf_artifacts.duplicates")
    prompt_slug = csv(profile.prompt_words).replace(",", "-")
    expected_names = {
        f"pw{prompt_slug}_c{concurrency}_mt{max_tokens}_r{repeat}.json"
        for concurrency in profile.concurrency
        for max_tokens in profile.max_tokens
        for repeat in range(profile.repeats)
    }
    if {Path(path).name for path in paths} != expected_names:
        errors.append("leaf_artifacts.contract_cells")
    return errors


def validate_child_contract(
    summary: dict[str, Any], model: str, backend: str, profile: SloProfile
) -> list[str]:
    return (
        validate_child_identity(summary, model, backend, profile)
        + validate_child_workload(summary, profile)
        + validate_child_metric_contract(summary)
        + validate_child_leaf_artifacts(summary, profile)
    )


def validate_child_rows(
    summary: dict[str, Any], backend: str, profile: SloProfile
) -> list[str]:
    errors = []
    rows = summary.get("rows")
    if not isinstance(rows, list):
        errors.append("rows")
    elif any(not isinstance(row, dict) for row in rows):
        errors.append("rows.items")
    else:
        actual_rows = {
            (
                csv(tuple(row.get("prompt_words", [])))
                if isinstance(row.get("prompt_words"), list)
                else str(row.get("prompt_words")),
                row.get("concurrency"),
                row.get("max_tokens"),
                row.get("repeats"),
            )
            for row in rows
        }
        if actual_rows != expected_rows(profile):
            errors.append("rows.contract_cells")
        if len(rows) != len(actual_rows):
            errors.append("rows.duplicate_cells")
        required_row_fields = {
            "noisy_cell",
            "output_hash_distribution",
            "request_output_hashes_by_repeat",
            "combined_output_hashes",
            "hash_manifests_consistent",
            "qps_summary",
            "output_tokens_per_s_summary",
            "ttft_ms",
            "tpot_ms",
            "itl_ms",
            "completed",
            "failed",
            "timeouts",
            "trace_coverage",
            "metric_sample_counts",
            "passed",
            "benchmark_commands_passed",
            "tail_sample_sufficient",
        }
        if any(not required_row_fields.issubset(row) for row in rows):
            errors.append("rows.required_metrics")
        if summary.get("report_intent") == "http_serving_slo":
            for row in rows:
                errors.extend(validate_success_row(row, backend, profile))
    return errors


def validate_source_metadata(metadata: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(metadata.get("commit"), str) or not re.fullmatch(
        r"[0-9a-f]{12}|[0-9a-f]{40}", metadata["commit"]
    ):
        errors.append("metadata.commit")
    if not metadata.get("model_path"):
        errors.append("metadata.model_path")
    if not metadata.get("server_command"):
        errors.append("metadata.server_command")
    if metadata.get("source_revision") != metadata.get("commit"):
        errors.append("metadata.source_revision")
    return errors


def validate_model_metadata(metadata: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(metadata.get("model_revision"), str) or not metadata["model_revision"]:
        errors.append("metadata.model_revision")
    fingerprint = metadata.get("model_fingerprint")
    if (
        not isinstance(fingerprint, dict)
        or not isinstance(fingerprint.get("config.json"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", fingerprint["config.json"])
        or not isinstance(fingerprint.get("model.safetensors.index.json"), str)
        or not re.fullmatch(
            r"[0-9a-f]{64}", fingerprint["model.safetensors.index.json"]
        )
        or not isinstance(fingerprint.get("tokenizer.json"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", fingerprint["tokenizer.json"])
    ):
        errors.append("metadata.model_fingerprint")
    if not isinstance(metadata.get("server_binary_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", metadata["server_binary_sha256"]
    ):
        errors.append("metadata.server_binary_sha256")
    return errors


def validate_backend_metadata(
    summary: dict[str, Any], metadata: dict[str, Any], backend: str
) -> list[str]:
    backend_runtime_version = metadata.get("backend_runtime_version")
    startup_failure = summary.get("report_intent") == "http_serving_slo_startup_failure"
    if backend == "host-staged":
        allowed = {"host-staged"}
        if startup_failure:
            allowed.add("startup-failed")
        if backend_runtime_version not in allowed:
            return ["metadata.backend_runtime_version"]
    elif backend_runtime_version != "unavailable-before-init" and (
        not isinstance(backend_runtime_version, str)
        or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", backend_runtime_version)
    ):
        return ["metadata.backend_runtime_version"]
    return []


def validate_hardware_metadata(metadata: dict[str, Any]) -> list[str]:
    hardware = metadata.get("hardware_toolchain")
    if not isinstance(hardware, dict):
        return ["metadata.hardware_toolchain"]
    gpu = hardware.get("gpu")
    gpu_identity = hardware.get("gpu_identity_sha256")
    if (
        not isinstance(gpu, list)
        or len(gpu) != 2
        or not all(isinstance(item, str) and item for item in gpu)
        or not isinstance(gpu_identity, list)
        or len(gpu_identity) != 2
        or not all(
            isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item)
            for item in gpu_identity
        )
        or not isinstance(hardware.get("nvcc_version"), str)
        or not hardware["nvcc_version"]
    ):
        return ["metadata.hardware_toolchain"]
    return []


def validate_child_metadata(summary: dict[str, Any], backend: str) -> list[str]:
    metadata = summary.get("metadata")
    if not isinstance(metadata, dict):
        return ["metadata"]
    return (
        validate_source_metadata(metadata)
        + validate_model_metadata(metadata)
        + validate_backend_metadata(summary, metadata, backend)
        + validate_hardware_metadata(metadata)
    )


def child_validation_errors(
    summary: dict[str, Any], model: str, backend: str, profile: SloProfile
) -> list[str]:
    return list(
        dict.fromkeys(
            validate_child_contract(summary, model, backend, profile)
            + validate_child_rows(summary, backend, profile)
            + validate_child_metadata(summary, backend)
        )
    )


def build_retained_slo_report(
    model: str, summaries: list[tuple[Path, dict[str, Any]]]
) -> dict[str, Any]:
    required = {(backend, profile) for backend in BACKENDS for profile in PROFILES}
    found: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    duplicates = []
    invalid_children = []

    for path, summary in summaries:
        if not isinstance(summary, dict):
            invalid_children.append(
                {"artifact": str(path), "errors": ["document must be an object"]}
            )
            continue
        contract = summary.get("contract")
        if not isinstance(contract, dict):
            invalid_children.append({"artifact": str(path), "errors": ["contract"]})
            continue
        backend = summary.get("backend")
        profile_name = contract.get("name")
        if not isinstance(backend, str) or not isinstance(profile_name, str):
            invalid_children.append({"artifact": str(path), "errors": ["contract key"]})
            continue
        key = (backend, profile_name)
        if key not in required:
            invalid_children.append({"artifact": str(path), "errors": ["contract key"]})
            continue
        errors = child_validation_errors(
            summary, model, backend, PROFILES[profile_name]
        )
        if errors:
            invalid_children.append({"artifact": str(path), "errors": errors})
            continue
        if key in found:
            duplicates.append(
                {"backend": backend, "profile": profile_name, "artifact": str(path)}
            )
            continue
        found[key] = (path, summary)

    missing = [
        {"backend": backend, "profile": profile}
        for backend, profile in sorted(required - set(found))
    ]
    failed_children = [
        {"backend": backend, "profile": profile, "artifact": str(path)}
        for (backend, profile), (path, summary) in sorted(found.items())
        if summary.get("report_intent") != "http_serving_slo"
        or (summary.get("correctness_gate") or {}).get("passed") is not True
        or bool(summary.get("run_errors"))
        or any(row.get("passed") is not True for row in summary.get("rows", []))
    ]
    ordered = [found[key] for key in sorted(found)]
    child_commits = [
        str((summary.get("metadata") or {}).get("commit"))
        for _, summary in ordered
        if (summary.get("metadata") or {}).get("commit")
    ]
    commit_counts = value_counts(child_commits)
    commit_consistent = len(commit_counts) == 1 and len(child_commits) == len(required)
    child_hardware = [
        (summary.get("metadata") or {}).get("hardware_toolchain")
        for _, summary in ordered
    ]
    hardware_documents = [
        json.dumps(hardware, sort_keys=True, separators=(",", ":"))
        for hardware in child_hardware
        if hardware
    ]
    hardware_consistent = len(set(hardware_documents)) == 1 and len(
        hardware_documents
    ) == len(required)
    provenance_documents = []
    backend_runtime_versions = []
    server_commands_by_backend: dict[str, set[str]] = {
        backend: set() for backend in BACKENDS
    }
    for _, summary in ordered:
        metadata = summary.get("metadata") or {}
        provenance = {
            field: metadata.get(field)
            for field in (
                "source_revision",
                "model_revision",
                "model_fingerprint",
                "model_path",
                "server_binary_sha256",
            )
        }
        provenance_documents.append(
            json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        )
        backend = summary["backend"]
        backend_runtime_versions.append(
            f"{backend}:{metadata.get('backend_runtime_version')}"
        )
        server_commands_by_backend[backend].add(str(metadata.get("server_command")))
    provenance_consistent = len(set(provenance_documents)) == 1 and len(
        provenance_documents
    ) == len(required)
    backend_runtime_consistent = len(backend_runtime_versions) == len(required) and all(
        len(
            {
                value
                for value in backend_runtime_versions
                if value.startswith(f"{backend}:")
            }
        )
        == 1
        for backend in BACKENDS
    )
    server_commands_consistent = all(
        len(commands) == 1 for commands in server_commands_by_backend.values()
    )
    passed = (
        not missing
        and not duplicates
        and not invalid_children
        and not failed_children
        and commit_consistent
        and hardware_consistent
        and provenance_consistent
        and backend_runtime_consistent
        and server_commands_consistent
    )

    return {
        "schema_version": 1,
        "kind": "deepseek_v2_lite_retained_http_slo_report",
        "report_intent": "http_serving_slo",
        "model": model,
        "backends": list(BACKENDS),
        "profiles": sorted(PROFILES),
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "profile_spec_sha256": profile_spec_sha256(),
        "profile_specs": {
            name: asdict(profile) for name, profile in sorted(PROFILES.items())
        },
        "latency_budget": {
            "configured": False,
            "passed": None,
            "reason": "The retained report records latency distributions but does not define a production latency budget.",
        },
        "metadata": {
            "commit": child_commits[0] if commit_consistent else None,
            "benchmark_command": artifact_command(sys.argv),
            "hardware_toolchain": child_hardware[0] if hardware_consistent else None,
            "child_commits": commit_counts,
            "provenance": json.loads(provenance_documents[0])
            if provenance_consistent
            else None,
            "backend_runtime_versions": value_counts(backend_runtime_versions),
            "server_commands": {
                backend: sorted(commands)
                for backend, commands in server_commands_by_backend.items()
            },
        },
        "coverage_gate": {
            "passed": passed,
            "required_children": len(required),
            "retained_children": len(found),
            "missing": missing,
            "duplicates": duplicates,
            "invalid_children": invalid_children,
            "failed_children": failed_children,
            "commit_consistent": commit_consistent,
            "hardware_consistent": hardware_consistent,
            "provenance_consistent": provenance_consistent,
            "backend_runtime_consistent": backend_runtime_consistent,
            "server_commands_consistent": server_commands_consistent,
        },
        "reports": [
            {
                "artifact": str(path),
                "backend": summary["backend"],
                "profile": summary["contract"]["name"],
                "correctness_gate": summary["correctness_gate"],
                "workload": summary["workload"],
                "rows": summary["rows"],
                "metadata": summary["metadata"],
                "metric_definitions": summary["metric_definitions"],
                "latency_budget": summary["latency_budget"],
                "leaf_artifacts": summary["leaf_artifacts"],
                "claim_boundary": summary["contract"]["claim_boundary"],
            }
            for path, summary in ordered
        ],
        "claim_boundary": REPORT_CLAIM_BOUNDARY,
    }


def verify_leaf_artifacts(
    summaries: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    checked = 0
    failures = []
    for path, summary in summaries:
        if not isinstance(summary, dict):
            failures.append(
                {"summary": str(path), "failures": ["document must be an object"]}
            )
            continue
        verification = verify_summary_leaf_artifacts(summary)
        checked += verification["checked"]
        if not verification["passed"]:
            failures.append(
                {"summary": str(path), "failures": verification["failures"]}
            )
    return {
        "passed": not failures and checked > 0,
        "checked": checked,
        "failures": failures,
    }


def run_profile(args: argparse.Namespace) -> int:
    required_paths = (
        args.server_log,
        args.server_binary,
        Path(args.model_path) / "config.json",
        Path(args.model_path) / "model.safetensors.index.json",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise SystemExit(
            f"required retained-run files are missing: {', '.join(missing)}"
        )
    validate_commit(args.commit)
    return subprocess.run(build_sweep_command(args), check=False).returncode


def combine(args: argparse.Namespace) -> int:
    if any(path.resolve() == args.out.resolve() for path in args.summary):
        raise SystemExit("--out must not overwrite an input --summary")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.unlink(missing_ok=True)
    summaries = [
        (path, json.loads(path.read_text(encoding="utf-8"))) for path in args.summary
    ]
    report = build_retained_slo_report(args.model, summaries)
    leaf_artifact_verification = verify_leaf_artifacts(summaries)
    report["leaf_artifact_verification"] = leaf_artifact_verification
    report["coverage_gate"]["leaf_artifacts_verified"] = leaf_artifact_verification[
        "passed"
    ]
    report["coverage_gate"]["passed"] = (
        report["coverage_gate"]["passed"]
        and leaf_artifact_verification["passed"]
    )
    print(write_json(args.out, report))
    return 0 if report["coverage_gate"]["passed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run one fixed backend/profile contract."
    )
    run_parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    run_parser.add_argument("--backend", choices=BACKENDS, required=True)
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    run_parser.add_argument("--model", default="DeepSeek-V2-Lite")
    run_parser.add_argument("--model-path", required=True)
    run_parser.add_argument("--server-command", required=True)
    run_parser.add_argument("--server-log", type=Path, required=True)
    run_parser.add_argument("--commit", required=True)
    run_parser.add_argument("--model-revision", required=True)
    run_parser.add_argument("--server-binary", type=Path, required=True)
    run_parser.add_argument("--record-startup-failure")
    run_parser.add_argument("--out-dir", type=Path, required=True)

    combine_parser = subparsers.add_parser(
        "combine", help="Combine all six retained summaries."
    )
    combine_parser.add_argument("--model", default="DeepSeek-V2-Lite")
    combine_parser.add_argument("--summary", action="append", type=Path, required=True)
    combine_parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "run":
        raise SystemExit(run_profile(args))
    raise SystemExit(combine(args))


if __name__ == "__main__":
    main()
