#!/usr/bin/env python3
"""Run and summarize the DeepSeek-V2-Lite vLLM TP2/EP2 benchmark matrix.

The script keeps three evidence buckets separate:

* HF / host-staged / NCCL correctness gate.
* PegaInfer direct same-prompt diagnostic batch attribution.
* HTTP concurrency pressure driven by `vllm bench serve`.

It intentionally does not turn HTTP concurrency into an internal batch-size
claim. Use the optional PegaInfer trace pass for `decode_batch_size_max`.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import http.client
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_PATH = Path("models/DeepSeek-V2-Lite")
DEFAULT_MODEL_ID = "DeepSeek-V2-Lite"
DEFAULT_CONCURRENCY = [1, 4, 8]
DEFAULT_DIRECT_BATCHES = [1, 4, 8]
CLAIM_CORRECTNESS = "correctness"
CLAIM_DIRECT = "direct_diagnostic_batch"
CLAIM_HTTP = "http_pressure"
CLAIM_FAILED = "failed_setup"
HTTP_METADATA_PREFIX = "pegainfer_contract_"


@dataclass(frozen=True)
class EngineSpec:
    name: str
    family: str
    claim_label: str
    ep_backend: str | None
    enable_expert_parallel: bool = False


ENGINES = [
    EngineSpec(
        name="pegainfer-host-staged",
        family="pegainfer",
        claim_label="PegaInfer host-staged",
        ep_backend="host-staged",
    ),
    EngineSpec(
        name="pegainfer-nccl",
        family="pegainfer",
        claim_label="PegaInfer NCCL",
        ep_backend="nccl",
    ),
    EngineSpec(
        name="vllm-tp2",
        family="vllm",
        claim_label="vLLM TP2",
        ep_backend=None,
    ),
    EngineSpec(
        name="vllm-tp2-ep2",
        family="vllm",
        claim_label="vLLM TP2+EP2",
        ep_backend=None,
        enable_expert_parallel=True,
    ),
]


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    """Return repo-relative paths for JSON artifacts when possible."""
    try:
        return str(path.absolute().relative_to(REPO_ROOT.absolute()))
    except ValueError:
        return str(path)


def public_path(path: Path) -> str:
    """Return a repo-relative or basename-only path for public benchmark metadata."""
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.absolute().relative_to(REPO_ROOT.absolute()))
    except ValueError:
        name = path.name or "path"
        return f"<external>/{name}"


def unix_s() -> int:
    return int(time.time())


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_capture(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit={completed.returncode}: {render_cmd(cmd)}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    return completed


def try_command(cmd: list[str], timeout_s: float = 15.0) -> dict[str, Any]:
    executable = cmd[0]
    if shutil.which(executable) is None and not Path(executable).exists():
        return {"command": redact_command(cmd), "available": False}
    try:
        completed = run_capture(cmd, timeout=timeout_s, check=False)
    except Exception as exc:  # noqa: BLE001 - metadata probe should not fail the benchmark.
        return {
            "command": redact_command(cmd),
            "available": True,
            "exit_code": 1,
            "error": error_text(exc),
        }
    return {
        "command": redact_command(cmd),
        "available": True,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def decode_nccl_version_code(version_code: int) -> dict[str, Any] | None:
    if isinstance(version_code, bool) or not isinstance(version_code, int) or version_code <= 0:
        return None
    scale = 1000 if version_code < 10000 else 10000
    major = version_code // scale
    minor = (version_code % scale) // 100
    patch = version_code % 100
    return {"version_code": version_code, "version": f"{major}.{minor}.{patch}"}


def nccl_version_from_library(library_name: str) -> dict[str, Any]:
    try:
        library = ctypes.CDLL(library_name)
    except OSError as exc:
        return {"library": library_name, "available": False, "error": redact_text(str(exc))}
    try:
        get_version = library.ncclGetVersion
        get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_version.restype = ctypes.c_int
        version_code = ctypes.c_int()
        exit_code = int(get_version(ctypes.byref(version_code)))
    except Exception as exc:  # noqa: BLE001 - metadata probe should not fail the benchmark.
        return {
            "library": library_name,
            "available": True,
            "exit_code": 1,
            "error": redact_text(str(exc)),
        }
    result: dict[str, Any] = {"library": library_name, "available": True, "exit_code": exit_code}
    if exit_code != 0:
        return result
    decoded = decode_nccl_version_code(version_code.value)
    if decoded is None:
        result["error"] = f"invalid NCCL version code: {version_code.value}"
        return result
    result.update(decoded)
    return result


def mapped_nccl_libraries(maps_text: str) -> list[str]:
    libraries = []
    seen = set()
    for line in maps_text.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 6:
            continue
        mapped_path = parts[5].removesuffix(" (deleted)")
        if not Path(mapped_path).name.startswith("libnccl.so"):
            continue
        if mapped_path not in seen:
            libraries.append(mapped_path)
            seen.add(mapped_path)
    return libraries


def process_group_pids(pid: int, proc_root: Path = Path("/proc")) -> tuple[int, list[int]]:
    process_group_id = os.getpgid(pid)
    members = []
    for candidate in proc_root.iterdir():
        if not candidate.name.isdigit():
            continue
        candidate_pid = int(candidate.name)
        try:
            if os.getpgid(candidate_pid) == process_group_id:
                members.append(candidate_pid)
        except (OSError, ProcessLookupError):
            continue
    return process_group_id, sorted(set(members))


def process_nccl_runtime(pid: int, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    try:
        process_group_id, pids = process_group_pids(pid, proc_root)
    except Exception as exc:  # noqa: BLE001 - metadata probe should not fail benchmark execution.
        return {
            "source": "server_process_group_maps",
            "available": False,
            "error": redact_text(str(exc)),
        }

    libraries: list[str] = []
    map_errors: list[dict[str, Any]] = []
    mapped_pids: list[int] = []
    for member_pid in pids:
        maps_path = proc_root / str(member_pid) / "maps"
        try:
            text = maps_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            map_errors.append({"pid": member_pid, "error": redact_text(str(exc))})
            continue
        mapped = mapped_nccl_libraries(text)
        if mapped:
            mapped_pids.append(member_pid)
            libraries.extend(mapped)

    unique_libraries = sorted(set(libraries))
    base: dict[str, Any] = {
        "source": "server_process_group_maps",
        "process_group_id": process_group_id,
        "process_group_pids": pids,
        "mapped_pids": mapped_pids,
        "mapped_library_count": len(unique_libraries),
        "map_errors": map_errors,
    }
    if map_errors:
        base["available"] = False
        base["error"] = "expected every server process-group maps file to be readable"
        return base
    if len(unique_libraries) != 1:
        base["available"] = False
        base["error"] = f"expected exactly one NCCL library, found {len(unique_libraries)}"
        base["libraries"] = [redact_text(library) for library in unique_libraries]
        return base
    probed = nccl_version_from_library(unique_libraries[0])
    available = bool(probed.get("available") and probed.get("exit_code") == 0)
    return {
        **base,
        **probed,
        "available": available,
        "library": Path(unique_libraries[0]).name,
        "library_path": redact_text(unique_libraries[0]),
    }


def render_cmd(cmd: list[str]) -> str:
    return " ".join(shell_quote(part) for part in cmd)


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:@%")
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()


def stable_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def redact_text(text: str) -> str:
    text = text.replace(str(REPO_ROOT.absolute()), "<repo>")
    home = Path.home()
    if str(home) != "/":
        text = text.replace(str(home), "~")
    text = re.sub(r"(?<![A-Za-z0-9_~.-])/(?:root|home/[A-Za-z0-9._-]+)(?=/|\b)", "~", text)
    provider_tmp = "~/" + "auto" + "dl-tmp/"
    text = text.replace(provider_tmp, "~/tmp/")
    text = text.replace("<repo>/", "")
    text = re.sub(
        r"(?i)\b((?:HF|HUGGINGFACE(?:_HUB)?|API|ACCESS|SECRET|PASSWORD|TOKEN)[A-Z0-9_]*"
        r"\s*=\s*)(?:'[^']*'|\"[^\"]*\"|[^\s'\";,]+)",
        r"\1<redacted>",
        text,
    )
    return text


def redact_command(cmd: list[str]) -> list[str]:
    return [redact_text(part) for part in cmd]


def redact_payload(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_payload(item) for key, item in value.items()}
    return value


def error_text(exc: Exception) -> str:
    return redact_text(str(exc))


def resolved_hf_python(args: argparse.Namespace) -> str:
    return args.hf_python or sys.executable


def first_number(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value: Any = payload
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if isinstance(value, (int, float)):
            return float(value)
    return None


def coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def coerce_int(value: Any) -> int | None:
    number = coerce_number(value)
    return None if number is None else int(number)


def first_int(payload: dict[str, Any], *keys: str) -> int | None:
    value = first_number(payload, *keys)
    return None if value is None else int(value)


def summarize_values(values: list[float], noisy_threshold: float) -> dict[str, Any]:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return {
            "values": [],
            "median": None,
            "min": None,
            "max": None,
            "spread_ratio": None,
            "noisy": False,
        }
    median = statistics.median(cleaned)
    min_value = min(cleaned)
    max_value = max(cleaned)
    spread = None if median == 0 else (max_value - min_value) / abs(median)
    return {
        "values": cleaned,
        "median": median,
        "min": min_value,
        "max": max_value,
        "spread_ratio": spread,
        "noisy": bool(spread is not None and spread > noisy_threshold),
    }


def model_snapshot(model_path: Path) -> dict[str, Any]:
    resolved = repo_path(model_path)
    return {
        "path": public_path(model_path),
        "exists": resolved.exists(),
        "config_sha256": sha256_file(resolved / "config.json"),
        "tokenizer_sha256": sha256_file(resolved / "tokenizer.json"),
    }


def metadata(
    args: argparse.Namespace,
    *,
    probe_versions: bool = True,
    nccl_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    git_head = run_capture(["git", "rev-parse", "HEAD"], check=False)
    git_status = run_capture(["git", "status", "--porcelain"], check=False)
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "hf_python_explicit": bool(args.hf_python),
        "hf_python_note": None if args.hf_python else (
            "defaulted to the benchmark script Python; pass --hf-python when the HF "
            "oracle needs a different Transformers environment"
        ),
    }
    if probe_versions:
        versions.update(
            {
                "hf_python": try_command([resolved_hf_python(args), "--version"]),
                "cargo": try_command(["cargo", "--version"]),
                "rustc": try_command(["rustc", "--version"]),
                "nvidia_smi": try_command(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version,compute_cap,temperature.gpu,clocks.sm",
                        "--format=csv,noheader",
                    ]
                ),
                "nvcc": try_command(["nvcc", "--version"]),
                "nccl": try_command(["pkg-config", "--modversion", "nccl"]),
                "vllm": try_command([args.vllm_cmd, "--version"]),
            }
        )
    else:
        versions.update(
            {
                "probes_skipped": True,
                "hf_python": {"command": redact_command([resolved_hf_python(args), "--version"])},
                "cargo": {"command": ["cargo", "--version"]},
                "rustc": {"command": ["rustc", "--version"]},
                "nvidia_smi": {
                    "command": [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version,compute_cap,temperature.gpu,clocks.sm",
                        "--format=csv,noheader",
                    ]
                },
                "nvcc": {"command": ["nvcc", "--version"]},
                "nccl": {"command": ["pkg-config", "--modversion", "nccl"]},
                "vllm": {"command": redact_command([args.vllm_cmd, "--version"])},
            }
        )
    if nccl_runtime is not None:
        versions["nccl_runtime"] = redact_payload(nccl_runtime)
    return {
        "schema_version": 1,
        "created_unix_s": unix_s(),
        "repo": {
            "worktree": REPO_ROOT.name,
            "git_commit": git_head.stdout.strip() if git_head.returncode == 0 else None,
            "dirty": bool(git_status.stdout.strip()) if git_status.returncode == 0 else None,
            "dirty_files": git_status.stdout.splitlines() if git_status.returncode == 0 else [],
        },
        "model": model_snapshot(args.model_path),
        "benchmark_contract": {
            "model_id": args.model_id,
            "prompt_source": "vllm bench serve --dataset-name random",
            "input_len": args.input_len,
            "output_len": args.output_len,
            "num_prompts": args.num_prompts,
            "num_warmups": args.num_warmups,
            "max_concurrency": args.concurrency,
            "request_rate": args.request_rate,
            "temperature": args.temperature,
            "ignore_eos": args.ignore_eos,
            "repeats": args.repeats,
            "noisy_threshold": args.noisy_threshold,
        },
        "versions": versions,
    }


def run_correctness_gate(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    gate_dir = out_dir / "correctness"
    hf_json = gate_dir / "hf.json"
    host_json = gate_dir / "host-staged.json"
    nccl_json = gate_dir / "nccl.json"
    comparison_json = gate_dir / "comparison.json"
    gate_dir.mkdir(parents=True, exist_ok=True)
    model = str(args.model_path)
    hf_json_arg = display_path(hf_json)
    host_json_arg = display_path(host_json)
    nccl_json_arg = display_path(nccl_json)
    comparison_json_arg = display_path(comparison_json)
    case_set = "test_data/deepseek-v2-lite-ep2-cases.json"
    commands: list[dict[str, Any]] = []

    steps = [
        (
            [resolved_hf_python(args), "tools/accuracy/hf_dump_dsv2_lite_ep2_greedy.py",
             "--model-path", model, "--case-set-json", case_set, "--out", hf_json_arg],
            {},
            "hf",
        ),
        (
            ["cargo", "test", "--release", "-p", "pegainfer-deepseek-v2-lite",
             "--features", "deepseek-v2-lite", "--test", "e2e_ep2", "--", "--nocapture"],
            {
                "PEGAINFER_TEST_MODEL_PATH": model,
                "PEGAINFER_DSV2_LITE_E2E_CASE_SET": case_set,
                "PEGAINFER_DSV2_LITE_E2E_JSON_OUT": host_json_arg,
            },
            "host-staged",
        ),
        (
            ["cargo", "test", "--release", "-p", "pegainfer-deepseek-v2-lite",
             "--features", "deepseek-v2-lite", "--test", "e2e_ep2", "--", "--nocapture"],
            {
                "PEGAINFER_TEST_MODEL_PATH": model,
                "PEGAINFER_DSV2_LITE_E2E_CASE_SET": case_set,
                "PEGAINFER_DSV2_LITE_EP_BACKEND": "nccl",
                "PEGAINFER_DSV2_LITE_E2E_JSON_OUT": nccl_json_arg,
            },
            "nccl",
        ),
        (
            [sys.executable, "tools/accuracy/compare_dsv2_lite_ep2_outputs.py",
             "--hf", hf_json_arg, "--host-staged", host_json_arg, "--nccl", nccl_json_arg,
             "--out", comparison_json_arg, "--require-all-exact"],
            {},
            "compare",
        ),
    ]

    passed = True
    for cmd, extra_env, label in steps:
        env = os.environ.copy()
        env.update(extra_env)
        record = {"label": label, "command": redact_command(cmd), "env": sorted(extra_env)}
        try:
            completed = run_capture(cmd, env=env, timeout=args.command_timeout_s)
            record.update({"exit_code": completed.returncode})
        except Exception as exc:  # noqa: BLE001 - benchmark artifact records setup failures.
            passed = False
            record.update({"exit_code": 1, "error": error_text(exc), "claim_bucket": CLAIM_FAILED})
            commands.append(record)
            if not args.keep_going:
                break
        else:
            commands.append(record)

    comparison = load_json(comparison_json) if comparison_json.exists() else {}
    passed = passed and correctness_passed(comparison)
    return {
        "claim_bucket": CLAIM_CORRECTNESS if passed else CLAIM_FAILED,
        "passed": passed,
        "artifacts": {
            "hf": display_path(hf_json),
            "host_staged": display_path(host_json),
            "nccl": display_path(nccl_json),
            "comparison": display_path(comparison_json),
        },
        "commands": commands,
        "comparison": comparison,
    }


def run_direct_diagnostic(args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    direct_dir = out_dir / "direct_diagnostic_batch"
    rows = []
    for backend in ("host-staged", "nccl"):
        for batch_size in args.direct_batches:
            artifact = direct_dir / backend / f"batch{batch_size}.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "cargo", "run", "--release", "-p", "pegainfer-deepseek-v2-lite",
                "--features", "deepseek-v2-lite", "--bin", "dsv2_lite_ep2_decode_attribution",
                "--", "--model-path", str(args.model_path), "--batch-size", str(batch_size),
                "--out", display_path(artifact),
            ]
            env = os.environ.copy()
            env["PEGAINFER_DSV2_LITE_EP_BACKEND"] = backend
            row = {
                "claim_bucket": CLAIM_DIRECT,
                "backend": backend,
                "batch_size": batch_size,
                "artifact": display_path(artifact),
                "command": redact_command(cmd),
                "env": ["PEGAINFER_DSV2_LITE_EP_BACKEND"],
            }
            try:
                run_capture(cmd, env=env, timeout=args.command_timeout_s)
                payload = load_json(artifact)
                row.update(parse_direct_artifact(payload))
                row["passed"] = True
            except Exception as exc:  # noqa: BLE001
                row.update({"passed": False, "claim_bucket": CLAIM_FAILED, "error": error_text(exc)})
                if not args.keep_going:
                    rows.append(row)
                    return rows
            rows.append(row)
    return rows


def parse_direct_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    timing = payload.get("timing", {})
    stats = timing.get("per_token_decode_stats", {}) if isinstance(timing, dict) else {}
    mean_us = first_number(stats, "mean_us") if isinstance(stats, dict) else None
    batch_size = first_int(payload, "config.batch_size") or 1
    output_tok_s = None if not mean_us else batch_size * 1_000_000.0 / mean_us
    accuracy = payload.get("accuracy", {})
    ep = payload.get("ep", {})
    graph = payload.get("cuda_graph_readiness", {})
    return {
        "tpot_ms": None if mean_us is None else mean_us / 1000.0,
        "output_tok_s": output_tok_s,
        "token_sha256": accuracy.get("token_sha256") if isinstance(accuracy, dict) else None,
        "text_sha256": accuracy.get("text_sha256") if isinstance(accuracy, dict) else None,
        "same_prompt_rows_exact": accuracy.get("same_prompt_rows_exact")
        if isinstance(accuracy, dict) else None,
        "gpu_event_samples": first_int(payload, "gpu_timing.sample_count"),
        "gpu_timing_failures": first_int(payload, "gpu_timing.failure_count"),
        "backend_counters": {
            "host_dispatch_calls": ep.get("dispatch_calls"),
            "nccl_exchange_calls": ep.get("nccl_exchange_calls"),
        } if isinstance(ep, dict) else {},
        "ep": ep if isinstance(ep, dict) else {},
        "cuda_graph_readiness": graph if isinstance(graph, dict) else {},
    }


def trace_missing_count(trace: Any) -> int | None:
    if not isinstance(trace, dict):
        return None
    missing = trace.get("missing_traces")
    if isinstance(missing, list):
        return len(missing)
    value = trace.get("missing_trace_count")
    return int(value) if isinstance(value, (int, float)) else None


def trace_summary_for_payload(payload: dict[str, Any]) -> dict[str, Any]:
    trace = payload.get("server_trace", {})
    workload = payload.get("workload", {})
    requests = payload.get("requests", [])
    phases = trace.get("phases_ms", {}) if isinstance(trace, dict) else {}
    if not isinstance(phases, dict):
        phases = {}
    output_hash_count = sum(
        isinstance(request, dict)
        and isinstance(request.get("output_hash"), str)
        and bool(request["output_hash"])
        for request in requests
    ) if isinstance(requests, list) else 0
    return {
        "trace": trace,
        "workload": workload if isinstance(workload, dict) else {},
        "num_requests": workload.get("num_requests") if isinstance(workload, dict) else None,
        "traced_requests": trace.get("traced_requests") if isinstance(trace, dict) else None,
        "missing_trace_count": trace_missing_count(trace),
        "output_hash_count": output_hash_count,
        "prompt_tokens": trace.get("prompt_tokens") if isinstance(trace, dict) else None,
        "completion_tokens": trace.get("completion_tokens") if isinstance(trace, dict) else None,
        "active_set_size_max": trace.get("active_set_size_max") if isinstance(trace, dict) else None,
        "decode_batch_size_max": trace.get("decode_batch_size_max") if isinstance(trace, dict) else None,
        "decode_steps": trace.get("decode_steps") if isinstance(trace, dict) else None,
        "phase_ms": {
            "queue_wait": phases.get("queue_wait_ms"),
            "prefill": phases.get("prefill_ms"),
            "first_decode": phases.get("first_decode_ms"),
            "decode_mean": phases.get("decode_mean_ms"),
            "decode_total": phases.get("decode_total_ms"),
            "scheduled_to_first_token": phases.get("scheduled_to_first_token_ms"),
            "scheduled_to_terminal": phases.get("scheduled_to_terminal_ms"),
            "stream_flush": phases.get("stream_flush_ms"),
        },
    }


def trace_cell_errors(
    cell: dict[str, Any],
    args: argparse.Namespace,
    concurrency: int,
) -> list[str]:
    expected = args.num_prompts
    workload = cell.get("workload")
    prompt_tokens = cell.get("prompt_tokens")
    completion_tokens = cell.get("completion_tokens")
    decode_steps = cell.get("decode_steps")
    phase_ms = cell.get("phase_ms")
    errors = []
    expected_workload = {
        "num_requests": expected,
        "concurrency": concurrency,
        "warmup": 0,
        "prompt_words": args.input_len,
        "max_tokens": args.output_len,
        "temperature": args.temperature,
        "ignore_eos": args.ignore_eos,
    }
    if not isinstance(workload, dict) or any(
        workload.get(key) != value for key, value in expected_workload.items()
    ):
        errors.append("workload mismatch")
    if cell.get("completed") != expected or cell.get("num_requests") != expected:
        errors.append("request count mismatch")
    if cell.get("failed") != 0 or cell.get("timeouts") != 0:
        errors.append("request failures")
    if cell.get("missing_trace_count") != 0 or cell.get("traced_requests") != expected:
        errors.append("incomplete trace coverage")
    if cell.get("output_hash_count") != expected:
        errors.append("incomplete output hash coverage")
    if not (
        isinstance(prompt_tokens, dict)
        and prompt_tokens.get("samples") == expected
        and isinstance(prompt_tokens.get("min"), int)
        and prompt_tokens["min"] > 0
        and isinstance(prompt_tokens.get("max"), int)
        and prompt_tokens["max"] >= prompt_tokens["min"]
        and isinstance(prompt_tokens.get("total"), int)
        and prompt_tokens["min"] * expected
        <= prompt_tokens["total"]
        <= prompt_tokens["max"] * expected
    ):
        errors.append("invalid prompt token coverage")
    if not (
        isinstance(completion_tokens, dict)
        and completion_tokens.get("samples") == expected
        and isinstance(completion_tokens.get("min"), int)
        and completion_tokens["min"] > 0
        and isinstance(completion_tokens.get("max"), int)
        and completion_tokens["max"] >= completion_tokens["min"]
        and isinstance(completion_tokens.get("total"), int)
        and completion_tokens["min"] * expected
        <= completion_tokens["total"]
        <= completion_tokens["max"] * expected
    ):
        errors.append("invalid completion token coverage")
    elif args.ignore_eos and not (
        completion_tokens["min"] == args.output_len
        and completion_tokens["max"] == args.output_len
        and completion_tokens["total"] == args.output_len * expected
    ):
        errors.append("incomplete fixed-length generation")
    batch_attribution_valid = (
        isinstance(cell.get("active_set_size_max"), int)
        and cell["active_set_size_max"] > 0
        and isinstance(cell.get("decode_batch_size_max"), int)
        and cell["decode_batch_size_max"] > 0
    )
    if not batch_attribution_valid:
        errors.append("missing active/decode batch attribution")
    elif concurrency > 1 and (
        cell["active_set_size_max"] <= 1 or cell["decode_batch_size_max"] <= 1
    ):
        errors.append("no concurrent decode batch evidence")
    required_phases = (
        "queue_wait",
        "prefill",
        "first_decode",
        "decode_mean",
        "decode_total",
        "scheduled_to_first_token",
        "scheduled_to_terminal",
    )
    if not isinstance(phase_ms, dict) or any(
        not isinstance(phase_ms.get(name), dict)
        or phase_ms[name].get("samples") != expected
        for name in required_phases
    ):
        errors.append("incomplete phase attribution")
    per_request = decode_steps.get("per_request") if isinstance(decode_steps, dict) else None
    batched = (
        decode_steps.get("batched_request_steps_total")
        if isinstance(decode_steps, dict)
        else None
    )
    singleton = (
        decode_steps.get("singleton_request_steps_total")
        if isinstance(decode_steps, dict)
        else None
    )
    batched_share = (
        decode_steps.get("request_step_batched_share")
        if isinstance(decode_steps, dict)
        else None
    )
    request_step_total = per_request.get("total") if isinstance(per_request, dict) else None
    request_steps_valid = (
        isinstance(per_request, dict)
        and per_request.get("samples") == expected
        and isinstance(per_request.get("min"), int)
        and per_request["min"] > 0
        and isinstance(per_request.get("max"), int)
        and per_request["max"] >= per_request["min"]
        and isinstance(request_step_total, int)
        and per_request["min"] * expected
        <= request_step_total
        <= per_request["max"] * expected
        and isinstance(batched, int)
        and batched >= 0
        and isinstance(singleton, int)
        and singleton >= 0
        and batched + singleton == request_step_total
        and isinstance(batched_share, (int, float))
        and abs(batched_share - batched / request_step_total) <= 1.0e-12
    )
    if not request_steps_valid:
        errors.append("invalid request-step attribution")
    elif concurrency > 1 and batched == 0:
        errors.append("no batched request-step evidence")
    return errors


def correctness_passed(comparison: dict[str, Any]) -> bool:
    return (
        comparison.get("classification") == "all_token_text_exact"
        and comparison.get("warnings") == []
    )


class ManagedServer:
    def __init__(self, cmd: list[str], env: dict[str, str], log_path: Path) -> None:
        self.cmd = cmd
        self.env = env
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any | None = None

    def __enter__(self) -> "ManagedServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        self.log_handle.write(f"$ {render_cmd(self.cmd)}\n")
        self.log_handle.flush()
        self.process = subprocess.Popen(
            self.cmd,
            cwd=REPO_ROOT,
            env=self.env,
            text=True,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            if self.process is not None and self.process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGINT)
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(self.process.pid, signal.SIGTERM)
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(self.process.pid, signal.SIGKILL)
                        with contextlib.suppress(subprocess.TimeoutExpired):
                            self.process.wait(timeout=10)
        finally:
            if self.log_handle is not None:
                self.log_handle.close()

    def poll(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def log_tail(self, limit: int = 4000) -> str:
        if self.log_handle is not None:
            self.log_handle.flush()
        if not self.log_path.exists():
            return ""
        return self.log_path.read_text(encoding="utf-8", errors="replace")[-limit:]


def wait_for_server(server: ManagedServer, spec: EngineSpec, port: int, model_id: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = "not checked"
    while time.time() < deadline:
        exit_code = server.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"{spec.name} server exited before readiness with exit={exit_code}\n"
                f"log tail:\n{server.log_tail()}"
            )
        try:
            if spec.family == "vllm":
                http_json("GET", port, "/v1/models")
            else:
                http_json(
                    "POST",
                    port,
                    "/v1/completions",
                    {
                        "model": model_id,
                        "prompt": "hi",
                        "max_tokens": 1,
                        "temperature": 0,
                        "ignore_eos": True,
                    },
                )
            return
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(2)
    raise TimeoutError(f"{spec.name} did not become ready on port {port}: {last_error}")


def http_json(method: str, port: int, path: str, body: dict[str, Any] | None = None) -> Any:
    conn = http.client.HTTPConnection("127.0.0.1", port=port, timeout=10)
    try:
        if body is None:
            conn.request(method, path)
        else:
            conn.request(
                method,
                path,
                body=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
        response = conn.getresponse()
        raw = response.read(4096)
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}: {raw.decode('utf-8', errors='replace')}")
        if not raw:
            return None
        return json.loads(raw.decode("utf-8", errors="replace"))
    finally:
        conn.close()


def server_command(args: argparse.Namespace, spec: EngineSpec, port: int) -> list[str]:
    if spec.family == "pegainfer":
        return [
            "cargo", "run", "--release", "-p", "pegainfer-server",
            "--features", "deepseek-v2-lite", "--bin", "pegainfer",
            "--", "--model-path", str(args.model_path), "--served-model-name", args.model_id,
            "--port", str(port), "--cuda-graph=false",
        ]
    cmd = [
        args.vllm_cmd, "serve", str(args.model_path), "--host", "127.0.0.1",
        "--port", str(port), "--served-model-name", args.model_id,
        "--tensor-parallel-size", "2", "--trust-remote-code",
    ]
    if spec.enable_expert_parallel:
        cmd.append("--enable-expert-parallel")
    cmd.extend(args.vllm_serve_extra_args)
    return cmd


def engine_env(args: argparse.Namespace, spec: EngineSpec) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("RUST_LOG", "info")
    if spec.ep_backend is not None:
        env["PEGAINFER_DSV2_LITE_EP_BACKEND"] = spec.ep_backend
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


BENCHMARK_SERVER_ENV_KEYS = (
    "PEGAINFER_DSV2_LITE_EP_BACKEND",
    "PEGAINFER_DSV2_LITE_HOST_STAGED_EXPERT_BATCH",
    "PEGAINFER_DSV2_LITE_NCCL_EXPERT_BATCH",
    "PEGAINFER_DSV2_LITE_NCCL_ROUTER",
    "CUDA_VISIBLE_DEVICES",
)


def benchmark_server_env(env: dict[str, str]) -> dict[str, str]:
    return {
        key: env[key]
        for key in BENCHMARK_SERVER_ENV_KEYS
        if key in env
    }


def vllm_bench_command(
    args: argparse.Namespace,
    *,
    port: int,
    num_prompts: int,
    result_dir: Path | None,
    result_filename: str | None,
    max_concurrency: int,
) -> list[str]:
    cmd = [
        args.vllm_cmd, "bench", "serve",
        "--backend", "openai",
        "--endpoint", "/v1/completions",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model", args.model_id,
        "--tokenizer", str(args.model_path),
        "--dataset-name", "random",
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(args.output_len),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(max_concurrency),
        "--request-rate", str(args.request_rate),
        "--temperature", str(args.temperature),
    ]
    for key, value in {
        f"{HTTP_METADATA_PREFIX}input_len": args.input_len,
        f"{HTTP_METADATA_PREFIX}output_len": args.output_len,
        f"{HTTP_METADATA_PREFIX}temperature": args.temperature,
        f"{HTTP_METADATA_PREFIX}ignore_eos": args.ignore_eos,
    }.items():
        cmd.extend(["--metadata", f"{key}={value}"])
    if args.ignore_eos:
        cmd.append("--ignore-eos")
    if result_dir is not None and result_filename is not None:
        cmd.extend(
            [
                "--save-result",
                "--save-detailed",
                "--result-dir",
                display_path(result_dir),
                "--result-filename",
                result_filename,
            ]
        )
    return cmd


def run_http_matrix(args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for spec_index, spec in enumerate(ENGINES):
        port = args.port + spec_index
        log_path = out_dir / "server_logs" / f"{spec.name}.log"
        cmd = server_command(args, spec, port)
        env = engine_env(args, spec)
        engine_row = {
            "engine": spec.name,
            "label": spec.claim_label,
            "family": spec.family,
            "claim_bucket": CLAIM_HTTP,
            "server_command": redact_command(cmd),
            "server_log": display_path(log_path),
            "server_env": benchmark_server_env(env),
            "cells": [],
        }
        try:
            with ManagedServer(cmd, env, log_path) as server:
                wait_for_server(server, spec, port, args.model_id, args.server_ready_timeout_s)
                if spec.ep_backend == "nccl" and server.process is not None:
                    engine_row["nccl_runtime"] = process_nccl_runtime(server.process.pid)
                for concurrency in args.concurrency:
                    for repeat in range(args.repeats):
                        cell = run_http_cell(args, out_dir, spec, port, concurrency, repeat)
                        engine_row["cells"].append(cell)
                        if not cell.get("passed") and not args.keep_going:
                            raise RuntimeError(cell.get("error", "HTTP benchmark cell failed"))
            engine_row["passed"] = bool(engine_row["cells"]) and all(
                cell.get("passed") for cell in engine_row["cells"]
            )
            if not engine_row["cells"]:
                engine_row.update({
                    "claim_bucket": CLAIM_FAILED,
                    "error": "no HTTP benchmark result artifacts found",
                })
        except Exception as exc:  # noqa: BLE001
            engine_row.update({
                "passed": False,
                "claim_bucket": CLAIM_FAILED,
                "error": error_text(exc),
            })
            if log_path.exists():
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                # Keep the exception text in `error` and the log-derived startup class separately.
                engine_row["startup_failure"] = classify_server_start_failure(log_text)
                engine_row["server_log_tail"] = redact_text(log_text[-4000:])
            if not args.keep_going:
                rows.append(engine_row)
                break
        if rows and rows[-1] is engine_row:
            continue
        rows.append(engine_row)
    return summarize_http_rows(rows, args.noisy_threshold)


def run_http_cell(
    args: argparse.Namespace,
    out_dir: Path,
    spec: EngineSpec,
    port: int,
    concurrency: int,
    repeat: int,
) -> dict[str, Any]:
    result_dir = out_dir / "http_raw" / spec.name / f"c{concurrency}" / f"r{repeat}"
    result_filename = "result.json"
    result_path = result_dir / result_filename
    warmup_cmd = None
    if args.num_warmups > 0:
        warmup_cmd = vllm_bench_command(
            args,
            port=port,
            num_prompts=args.num_warmups,
            result_dir=None,
            result_filename=None,
            max_concurrency=concurrency,
        )
        run_capture(warmup_cmd, timeout=args.command_timeout_s)
    bench_cmd = vllm_bench_command(
        args,
        port=port,
        num_prompts=args.num_prompts,
        result_dir=result_dir,
        result_filename=result_filename,
        max_concurrency=concurrency,
    )
    cell = {
        "concurrency": concurrency,
        "repeat": repeat,
        "command": redact_command(bench_cmd),
        "warmup_command": None if warmup_cmd is None else redact_command(warmup_cmd),
        "artifact": display_path(result_path),
    }
    try:
        completed = run_capture(bench_cmd, timeout=args.command_timeout_s)
        cell.update({"exit_code": completed.returncode})
        if not result_path.exists():
            raise FileNotFoundError(f"vLLM bench result not found: {result_path}")
        cell.update(
            parse_vllm_bench_artifact(
                load_json(result_path),
                expected_http_workload(args, concurrency),
            )
        )
    except Exception as exc:  # noqa: BLE001
        cell.update({"passed": False, "claim_bucket": CLAIM_FAILED, "error": error_text(exc)})
    return cell


def expected_http_workload(args: argparse.Namespace, concurrency: int) -> dict[str, Any]:
    return {
        "model_id": args.model_id,
        "num_prompts": args.num_prompts,
        "max_concurrency": concurrency,
        "request_rate": args.request_rate,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "temperature": args.temperature,
        "ignore_eos": args.ignore_eos,
    }


def parse_vllm_bench_artifact(
    payload: dict[str, Any],
    expected_workload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completed = first_int(payload, "completed", "num_completed_requests", "successful_requests")
    failed = first_int(
        payload,
        "failed",
        "num_failed_requests",
        "num_failures",
        "failed_requests",
        "errors",
    ) or 0
    timeouts = first_int(payload, "timeouts", "num_timeouts", "timeout_requests", "timed_out") or 0
    total_output_tokens = first_number(payload, "total_output_tokens", "output_tokens")
    duration = first_number(payload, "duration", "benchmark_duration_s", "total_time_s")
    output_tok_s = first_number(payload, "output_throughput", "output_tokens_per_s")
    if output_tok_s is None and total_output_tokens is not None and duration:
        output_tok_s = total_output_tokens / duration
    output_hash = output_text_hash(payload)
    output_count = output_hash["count"]
    detailed_outputs_valid = (
        completed is not None
        and output_count == completed
        and output_count > 0
        and output_hash["sha256"] is not None
        and output_hash.get("valid") is True
    )
    workload_mismatches: list[str] = []
    expected_completed = None
    if expected_workload is not None:
        expected_completed = expected_workload.get("num_prompts")
        ignore_eos = payload.get(f"{HTTP_METADATA_PREFIX}ignore_eos", payload.get("ignore_eos"))
        if isinstance(ignore_eos, str):
            ignore_eos = ignore_eos.lower() == "true"
        observed = {
            "model_id": payload.get("model_id") or payload.get("model"),
            "num_prompts": coerce_int(payload.get("num_prompts")),
            "max_concurrency": coerce_int(payload.get("max_concurrency")),
            "request_rate": str(payload.get("request_rate")) if payload.get("request_rate") is not None else None,
            "input_len": coerce_int(payload.get(f"{HTTP_METADATA_PREFIX}input_len", payload.get("input_len"))),
            "output_len": coerce_int(payload.get(f"{HTTP_METADATA_PREFIX}output_len", payload.get("output_len"))),
            "temperature": coerce_number(payload.get(f"{HTTP_METADATA_PREFIX}temperature", payload.get("temperature"))),
            "ignore_eos": ignore_eos,
        }
        for key, expected in expected_workload.items():
            if observed.get(key) != expected:
                workload_mismatches.append(key)
    full_completion = (
        completed is not None
        and completed > 0
        and (expected_completed is None or completed == expected_completed)
    )
    passed = (
        full_completion
        and failed == 0
        and timeouts == 0
        and detailed_outputs_valid
        and not workload_mismatches
    )
    return {
        "claim_bucket": CLAIM_HTTP,
        "passed": passed,
        "completed": completed,
        "failed": failed,
        "timeouts": timeouts,
        "total_input_tokens": first_int(payload, "total_input_tokens", "input_tokens"),
        "total_output_tokens": None if total_output_tokens is None else int(total_output_tokens),
        "duration_s": duration,
        "request_throughput": first_number(payload, "request_throughput", "requests_per_s"),
        "output_tok_s": output_tok_s,
        "mean_ttft_ms": first_number(payload, "mean_ttft_ms", "ttft.mean_ms"),
        "median_ttft_ms": first_number(payload, "median_ttft_ms", "ttft.median_ms"),
        "mean_tpot_ms": first_number(payload, "mean_tpot_ms", "tpot.mean_ms"),
        "median_tpot_ms": first_number(payload, "median_tpot_ms", "tpot.median_ms"),
        "mean_itl_ms": first_number(payload, "mean_itl_ms", "itl.mean_ms"),
        "median_itl_ms": first_number(payload, "median_itl_ms", "itl.median_ms"),
        "output_text_sha256": output_hash["sha256"],
        "output_text_count": output_count,
        "detailed_outputs_valid": detailed_outputs_valid,
        "workload_mismatches": workload_mismatches,
        "expected_completed": expected_completed,
        "full_completion": full_completion,
    }


def output_text_hash(payload: dict[str, Any]) -> dict[str, Any]:
    texts: list[str] = []
    detailed_outputs_valid = False
    for key in (
        "generated_texts",
        "generated_outputs",
        "outputs",
        "responses",
        "request_outputs",
        "request_results",
        "details",
        "per_request",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            source_valid = True
            for item in value:
                item_texts = extract_output_texts(item)
                texts.extend(item_texts)
                if len(item_texts) != 1 or not item_texts[0]:
                    source_valid = False
            detailed_outputs_valid = source_valid
            break
    if not texts:
        return {"sha256": None, "count": 0, "valid": False}
    digest = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"sha256": digest, "count": len(texts), "valid": detailed_outputs_valid}


def extract_output_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, dict):
        return []
    for key in ("generated_text", "output_text", "output", "text"):
        candidate = value.get(key)
        if isinstance(candidate, str):
            return [candidate]
    response = value.get("response")
    if isinstance(response, str):
        return [response]
    if isinstance(response, dict):
        return extract_openai_response_texts(response)
    return extract_openai_response_texts(value)


def extract_openai_response_texts(payload: dict[str, Any]) -> list[str]:
    texts = []
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if isinstance(choice.get("text"), str):
                texts.append(choice["text"])
                continue
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                texts.append(message["content"])
    return texts


def summarize_http_rows(engine_rows: list[dict[str, Any]], noisy_threshold: float) -> list[dict[str, Any]]:
    summarized = []
    for engine in engine_rows:
        cells = engine.get("cells") or []
        grouped: dict[int, list[dict[str, Any]]] = {}
        for cell in cells:
            if isinstance(cell, dict) and "concurrency" in cell:
                grouped.setdefault(int(cell["concurrency"]), []).append(cell)
        summary_rows = []
        for concurrency, repeats in sorted(grouped.items()):
            passed_repeats = [cell for cell in repeats if cell.get("passed") is True]
            tpot = [
                cell["mean_tpot_ms"]
                for cell in passed_repeats
                if cell.get("mean_tpot_ms") is not None
            ]
            output = [
                cell["output_tok_s"]
                for cell in passed_repeats
                if cell.get("output_tok_s") is not None
            ]
            failed = [cell.get("failed", 0) for cell in repeats]
            timeouts = [cell.get("timeouts", 0) for cell in repeats]
            tpot_summary = summarize_values(tpot, noisy_threshold)
            output_summary = summarize_values(output, noisy_threshold)
            summary_rows.append(
                {
                    "concurrency": concurrency,
                    "repeat_count": len(repeats),
                    "passed_repeat_count": len(passed_repeats),
                    "completed": [cell.get("completed") for cell in repeats],
                    "failed": failed,
                    "timeouts": timeouts,
                    "cell_passed": [cell.get("passed") is True for cell in repeats],
                    "output_text_sha256": [cell.get("output_text_sha256") for cell in repeats],
                    "mean_tpot_ms": tpot_summary,
                    "output_tok_s": output_summary,
                    "mean_ttft_ms": summarize_values(
                        [
                            cell["mean_ttft_ms"]
                            for cell in passed_repeats
                            if cell.get("mean_ttft_ms") is not None
                        ],
                        noisy_threshold,
                    ),
                    "mean_itl_ms": summarize_values(
                        [
                            cell["mean_itl_ms"]
                            for cell in passed_repeats
                            if cell.get("mean_itl_ms") is not None
                        ],
                        noisy_threshold,
                    ),
                    "noisy": tpot_summary["noisy"] or output_summary["noisy"],
                }
            )
        copied = dict(engine)
        copied["summary_by_concurrency"] = summary_rows
        summarized.append(copied)
    return summarized


def run_pegainfer_trace_pass(args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for spec in ENGINES:
        if spec.family != "pegainfer":
            continue
        port = args.port + 20 + len(rows)
        log_path = out_dir / "trace_server_logs" / f"{spec.name}.log"
        cmd = server_command(args, spec, port)
        env = engine_env(args, spec)
        row = {
            "engine": spec.name,
            "claim_bucket": CLAIM_HTTP,
            "server_command": redact_command(cmd),
            "server_log": display_path(log_path),
            "cells": [],
        }
        try:
            with ManagedServer(cmd, env, log_path) as server:
                wait_for_server(server, spec, port, args.model_id, args.server_ready_timeout_s)
                for concurrency in args.concurrency:
                    out = out_dir / "pegainfer_trace" / spec.name / f"c{concurrency}.json"
                    trace_cmd = [
                        sys.executable, "scripts/bench_http_serving.py",
                        "--base-url", f"http://127.0.0.1:{port}",
                        "--model", args.model_id,
                        "--num-requests", str(args.num_prompts),
                        "--concurrency", str(concurrency),
                        "--warmup", "0",
                        "--prompt-words", str(args.input_len),
                        "--max-tokens", str(args.output_len),
                        "--temperature", str(args.temperature),
                        "--timeout", str(args.command_timeout_s),
                        "--server-log", display_path(log_path),
                        "--out", display_path(out),
                    ]
                    if not args.ignore_eos:
                        trace_cmd.append("--no-ignore-eos")
                    run_capture(trace_cmd, timeout=args.command_timeout_s)
                    payload = load_json(out)
                    cell = {
                        "concurrency": concurrency,
                        "artifact": display_path(out),
                        "completed": payload["summary"]["completed"],
                        "failed": payload["summary"]["failed"],
                        "timeouts": payload["summary"]["timeouts"],
                        "output_tok_s": payload["summary"]["output_tokens_per_s"],
                    }
                    cell.update(trace_summary_for_payload(payload))
                    cell["validation_errors"] = trace_cell_errors(cell, args, concurrency)
                    cell["passed"] = not cell["validation_errors"]
                    row["cells"].append(cell)
            row["passed"] = bool(row["cells"]) and all(
                cell.get("passed") is True for cell in row["cells"]
            )
            if row["cells"] and not row["passed"]:
                row.update({
                    "claim_bucket": CLAIM_FAILED,
                    "error": "trace cells failed strict coverage validation",
                })
            if not row["cells"]:
                row.update({
                    "claim_bucket": CLAIM_FAILED,
                    "error": "no PegaInfer trace result artifacts found",
                })
        except Exception as exc:  # noqa: BLE001
            row.update({"passed": False, "claim_bucket": CLAIM_FAILED, "error": error_text(exc)})
            if not args.keep_going:
                rows.append(row)
                break
        if rows and rows[-1] is row:
            continue
        rows.append(row)
    return rows


def build_summary(args: argparse.Namespace, out_dir: Path, sections: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "deepseek_v2_lite_vllm_tp2_ep2_benchmark_matrix",
        "metadata": metadata(args),
        "artifacts_root": display_path(out_dir),
        "correctness_gate": sections.get("correctness_gate"),
        "direct_diagnostic_batch": sections.get("direct_diagnostic_batch", []),
        "http_concurrency_pressure": sections.get("http_concurrency_pressure", []),
        "pegainfer_trace_pass": sections.get("pegainfer_trace_pass", []),
        "claim_boundary": (
            "This matrix separates correctness, direct same-prompt diagnostic batch, "
            "HTTP concurrency pressure, and failed setup rows. It does not claim vLLM parity "
            "or production DeepSeek-V2-Lite EP2 serving readiness."
        ),
    }


def emit_benchmark_artifacts(args: argparse.Namespace, out_dir: Path, summary: dict[str, Any]) -> None:
    summary_path = out_dir / "summary.json"
    baseline_summary, baseline_path = load_baseline_summary(args)
    regression_path = out_dir / "regression_summary.json"
    regression = build_regression_summary(
        summary,
        baseline_summary,
        current_summary_path=summary_path,
        baseline_summary_path=baseline_path,
        noisy_threshold=args.noisy_threshold,
    )
    write_json(regression_path, regression)
    manifest = build_artifact_manifest(summary, out_dir, summary_path, regression_path)
    write_json(out_dir / "artifact_manifest.json", manifest)


def load_baseline_summary(args: argparse.Namespace) -> tuple[dict[str, Any] | None, Path | None]:
    baseline_arg = getattr(args, "baseline_summary", None)
    if baseline_arg is None:
        return None, None
    path = baseline_arg if isinstance(baseline_arg, Path) else Path(str(baseline_arg))
    path = path if path.is_absolute() else repo_path(path)
    return load_json(path), path


def build_artifact_manifest(
    summary: dict[str, Any],
    out_dir: Path,
    summary_path: Path,
    regression_path: Path,
) -> dict[str, Any]:
    artifacts = collect_manifest_artifacts(summary, out_dir)
    rows = collect_claim_rows(summary)
    commands = collect_manifest_commands(summary)
    summary_artifacts = [
        manifest_artifact_record(out_dir, summary_path, kind="summary", claim_bucket="summary"),
        manifest_artifact_record(
            out_dir,
            regression_path,
            kind="regression_summary",
            claim_bucket="regression_summary",
        ),
    ]
    return {
        "schema_version": 1,
        "kind": "deepseek_v2_lite_benchmark_artifact_manifest",
        "generated_unix_s": unix_s(),
        "artifacts_root": artifact_path_label(out_dir, out_dir)["path"],
        "metadata": redact_payload(summary.get("metadata", {})),
        "benchmark_contract": redact_payload(
            summary.get("metadata", {}).get("benchmark_contract", {})
            if isinstance(summary.get("metadata"), dict) else {}
        ),
        "model": redact_payload(
            summary.get("metadata", {}).get("model", {}) if isinstance(summary.get("metadata"), dict) else {}
        ),
        "artifact_paths": artifacts,
        "summary_artifacts": summary_artifacts,
        "summary_sha256": summary_artifacts[0]["sha256"],
        "regression_summary_sha256": summary_artifacts[1]["sha256"],
        "artifact_bundle_sha256": stable_json_sha256([
            {
                "path": artifact.get("path"),
                "path_root": artifact.get("path_root"),
                "sha256": artifact.get("sha256"),
                "bytes": artifact.get("bytes"),
            }
            for artifact in artifacts + summary_artifacts
        ]),
        "claim_rows": rows,
        "backend_commands": commands,
        "claim_boundary": summary.get("claim_boundary"),
    }


def collect_manifest_artifacts(summary: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    gate = summary.get("correctness_gate")
    if isinstance(gate, dict):
        for label, path_value in sorted((gate.get("artifacts") or {}).items()):
            append_manifest_artifact(
                artifacts,
                out_dir,
                path_value,
                kind=f"correctness_{label}",
                claim_bucket=gate.get("claim_bucket", CLAIM_CORRECTNESS),
                label=label,
            )
    for row in summary.get("direct_diagnostic_batch", []) or []:
        if not isinstance(row, dict):
            continue
        append_manifest_artifact(
            artifacts,
            out_dir,
            row.get("artifact"),
            kind="direct_diagnostic_batch",
            claim_bucket=row.get("claim_bucket", CLAIM_DIRECT),
            backend=row.get("backend"),
            batch_size=row.get("batch_size"),
        )
    for row in summary.get("http_concurrency_pressure", []) or []:
        if not isinstance(row, dict):
            continue
        append_manifest_artifact(
            artifacts,
            out_dir,
            row.get("server_log"),
            kind="http_server_log",
            claim_bucket=row.get("claim_bucket", CLAIM_HTTP),
            engine=row.get("engine"),
        )
        for cell in row.get("cells", []) or []:
            if not isinstance(cell, dict):
                continue
            append_manifest_artifact(
                artifacts,
                out_dir,
                cell.get("artifact"),
                kind="http_bench_result",
                claim_bucket=cell.get("claim_bucket", row.get("claim_bucket", CLAIM_HTTP)),
                engine=row.get("engine"),
                concurrency=cell.get("concurrency"),
                repeat=cell.get("repeat"),
            )
    for row in summary.get("pegainfer_trace_pass", []) or []:
        if not isinstance(row, dict):
            continue
        append_manifest_artifact(
            artifacts,
            out_dir,
            row.get("server_log"),
            kind="trace_server_log",
            claim_bucket=row.get("claim_bucket", CLAIM_HTTP),
            engine=row.get("engine"),
        )
        for cell in row.get("cells", []) or []:
            if not isinstance(cell, dict):
                continue
            append_manifest_artifact(
                artifacts,
                out_dir,
                cell.get("artifact"),
                kind="pegainfer_trace_result",
                claim_bucket=cell.get("claim_bucket", row.get("claim_bucket", CLAIM_HTTP)),
                engine=row.get("engine"),
                concurrency=cell.get("concurrency"),
            )
    artifacts.sort(key=lambda item: (str(item.get("path_root")), str(item.get("path")), str(item.get("kind"))))
    return artifacts


def append_manifest_artifact(
    artifacts: list[dict[str, Any]],
    out_dir: Path,
    path_value: Any,
    *,
    kind: str,
    claim_bucket: Any,
    **identity: Any,
) -> None:
    if not isinstance(path_value, str) or not path_value:
        return
    path = resolve_reported_path(path_value, out_dir)
    record = manifest_artifact_record(out_dir, path, kind=kind, claim_bucket=claim_bucket)
    for key, value in identity.items():
        if value is not None:
            record[key] = redact_payload(value)
    artifacts.append(record)


def manifest_artifact_record(
    out_dir: Path,
    path: Path,
    *,
    kind: str,
    claim_bucket: Any,
) -> dict[str, Any]:
    label = artifact_path_label(path, out_dir)
    exists = path.exists()
    return {
        "kind": kind,
        "claim_bucket": claim_bucket,
        "path": label["path"],
        "path_root": label["path_root"],
        "exists": exists,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size if exists and path.is_file() else None,
    }


def resolve_reported_path(path_value: str, out_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    bundle_candidate = out_dir / path
    if bundle_candidate.exists():
        return bundle_candidate
    return repo_candidate


def artifact_path_label(path: Path, out_dir: Path) -> dict[str, str]:
    absolute = path.absolute()
    try:
        return {"path": str(absolute.relative_to(out_dir.absolute())), "path_root": "artifact_bundle"}
    except ValueError:
        pass
    try:
        return {"path": str(absolute.relative_to(REPO_ROOT.absolute())), "path_root": "repo"}
    except ValueError:
        pass
    return {"path": public_path(absolute), "path_root": "external"}


def collect_manifest_commands(summary: dict[str, Any]) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    gate = summary.get("correctness_gate")
    if isinstance(gate, dict):
        for command in gate.get("commands", []) or []:
            if isinstance(command, dict):
                commands.append({
                    "section": "correctness_gate",
                    "claim_bucket": command.get("claim_bucket", gate.get("claim_bucket", CLAIM_CORRECTNESS)),
                    "command": redact_payload(command),
                })
    for row in summary.get("direct_diagnostic_batch", []) or []:
        if isinstance(row, dict) and "command" in row:
            commands.append({
                "section": "direct_diagnostic_batch",
                "claim_bucket": row.get("claim_bucket", CLAIM_DIRECT),
                "backend": row.get("backend"),
                "batch_size": row.get("batch_size"),
                "command": redact_payload(row.get("command")),
                "env": redact_payload(row.get("env", [])),
            })
    for section in ("http_concurrency_pressure", "pegainfer_trace_pass"):
        for row in summary.get(section, []) or []:
            if not isinstance(row, dict):
                continue
            if "server_command" in row:
                commands.append({
                    "section": section,
                    "claim_bucket": row.get("claim_bucket", CLAIM_HTTP),
                    "engine": row.get("engine"),
                    "command": redact_payload(row.get("server_command")),
                    "env": redact_payload(row.get("server_env", {})),
                })
            for cell in row.get("cells", []) or []:
                if not isinstance(cell, dict) or "command" not in cell:
                    continue
                commands.append({
                    "section": section,
                    "claim_bucket": cell.get("claim_bucket", row.get("claim_bucket", CLAIM_HTTP)),
                    "engine": row.get("engine"),
                    "concurrency": cell.get("concurrency"),
                    "repeat": cell.get("repeat"),
                    "command": redact_payload(cell.get("command")),
                    "warmup_command": redact_payload(cell.get("warmup_command")),
                })
    return commands


def collect_claim_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gate = summary.get("correctness_gate")
    if isinstance(gate, dict):
        comparison = gate.get("comparison") if isinstance(gate.get("comparison"), dict) else {}
        rows.append({
            "section": "correctness_gate",
            "claim_bucket": gate.get("claim_bucket", CLAIM_CORRECTNESS),
            "passed": gate.get("passed"),
            "classification": comparison.get("classification"),
            "warning_count": len(comparison.get("warnings", [])) if isinstance(comparison.get("warnings"), list) else None,
        })
    for row in summary.get("direct_diagnostic_batch", []) or []:
        if not isinstance(row, dict):
            continue
        rows.append({
            "section": "direct_diagnostic_batch",
            "claim_bucket": row.get("claim_bucket", CLAIM_DIRECT),
            "backend": row.get("backend"),
            "batch_size": row.get("batch_size"),
            "passed": row.get("passed"),
            "token_sha256": row.get("token_sha256"),
            "text_sha256": row.get("text_sha256"),
            "tpot_ms": row.get("tpot_ms"),
            "output_tok_s": row.get("output_tok_s"),
        })
    for row in summary.get("http_concurrency_pressure", []) or []:
        if not isinstance(row, dict):
            continue
        for cell_summary in row.get("summary_by_concurrency", []) or []:
            if not isinstance(cell_summary, dict):
                continue
            rows.append({
                "section": "http_concurrency_pressure",
                "claim_bucket": row.get("claim_bucket", CLAIM_HTTP),
                "engine": row.get("engine"),
                "passed": row.get("passed"),
                "concurrency": cell_summary.get("concurrency"),
                "completed": cell_summary.get("completed"),
                "failed": cell_summary.get("failed"),
                "timeouts": cell_summary.get("timeouts"),
                "output_text_sha256": cell_summary.get("output_text_sha256"),
                "noisy": cell_summary.get("noisy"),
                "mean_tpot_ms": metric_triplet(cell_summary.get("mean_tpot_ms")),
                "output_tok_s": metric_triplet(cell_summary.get("output_tok_s")),
            })
        if not row.get("summary_by_concurrency"):
            rows.append({
                "section": "http_concurrency_pressure",
                "claim_bucket": row.get("claim_bucket", CLAIM_HTTP),
                "engine": row.get("engine"),
                "passed": row.get("passed"),
                "error": row.get("error"),
                "startup_failure": row.get("startup_failure"),
            })
    for row in summary.get("pegainfer_trace_pass", []) or []:
        if not isinstance(row, dict):
            continue
        for cell in row.get("cells", []) or []:
            if not isinstance(cell, dict):
                continue
            trace = cell.get("trace") if isinstance(cell.get("trace"), dict) else {}
            rows.append({
                "section": "pegainfer_trace_pass",
                "claim_bucket": row.get("claim_bucket", CLAIM_HTTP),
                "engine": row.get("engine"),
                "passed": row.get("passed"),
                "concurrency": cell.get("concurrency"),
                "completed": cell.get("completed"),
                "failed": cell.get("failed"),
                "timeouts": cell.get("timeouts"),
                "output_tok_s": cell.get("output_tok_s"),
                "missing_trace_count": cell.get("missing_trace_count"),
                "active_set_size_max": trace.get("active_set_size_max"),
                "decode_batch_size_max": trace.get("decode_batch_size_max"),
            })
        if not row.get("cells"):
            rows.append({
                "section": "pegainfer_trace_pass",
                "claim_bucket": row.get("claim_bucket", CLAIM_HTTP),
                "engine": row.get("engine"),
                "passed": row.get("passed"),
                "error": row.get("error"),
                "startup_failure": row.get("startup_failure"),
            })
    return rows


def metric_triplet(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "median": value.get("median"),
        "min": value.get("min"),
        "max": value.get("max"),
        "noisy": value.get("noisy"),
    }


def build_regression_summary(
    summary: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    current_summary_path: Path,
    baseline_summary_path: Path | None,
    noisy_threshold: float,
) -> dict[str, Any]:
    current_sha = sha256_file(current_summary_path)
    baseline_sha = sha256_file(baseline_summary_path) if baseline_summary_path else None
    comparison: dict[str, Any] = {
        "schema_version": 1,
        "kind": "deepseek_v2_lite_benchmark_regression_summary",
        "generated_unix_s": unix_s(),
        "current_summary": {
            "path": artifact_path_label(current_summary_path, current_summary_path.parent)["path"],
            "sha256": current_sha,
        },
        "baseline_summary": None if baseline_summary_path is None else {
            "path": public_path(baseline_summary_path.absolute()),
            "sha256": baseline_sha,
        },
        "noisy_threshold": noisy_threshold,
    }
    reasons: list[str] = []
    if baseline is None:
        reasons.append("baseline_missing")
        comparison.update(no_baseline_regression_sections(summary))
    else:
        comparison["correctness"] = compare_single_projection(
            correctness_projection(summary),
            correctness_projection(baseline),
            ("passed", "classification", "warnings"),
        )
        comparison["direct_diagnostic_batch"] = compare_projection_maps(
            direct_projection_map(summary),
            direct_projection_map(baseline),
            ("passed", "token_sha256", "text_sha256", "tpot_ms", "output_tok_s"),
        )
        comparison["http_concurrency_pressure"] = compare_projection_maps(
            http_projection_map(summary),
            http_projection_map(baseline),
            (
                "row_passed",
                "completed",
                "failed",
                "timeouts",
                "output_text_sha256",
                "noisy",
                "mean_tpot_ms",
                "output_tok_s",
            ),
        )
        comparison["pegainfer_trace_pass"] = compare_projection_maps(
            trace_projection_map(summary),
            trace_projection_map(baseline),
            (
                "row_passed",
                "completed",
                "failed",
                "timeouts",
                "missing_trace_count",
                "active_set_size_max",
                "decode_batch_size_max",
                "trace_sha256",
            ),
        )
        reasons.extend(structural_projection_reasons(
            "direct_diagnostic_batch",
            comparison["direct_diagnostic_batch"],
        ))
        reasons.extend(structural_projection_reasons(
            "http_concurrency_pressure",
            comparison["http_concurrency_pressure"],
        ))
        reasons.extend(structural_projection_reasons(
            "pegainfer_trace_pass",
            comparison["pegainfer_trace_pass"],
        ))
        contract_reasons = comparability_reasons(summary, baseline)
        reasons.extend(contract_reasons)
    comparison["failed_setup_rows"] = compare_failed_setup(summary, baseline)
    reasons.extend(noise_reasons(summary, "current"))
    if baseline is not None:
        reasons.extend(noise_reasons(baseline, "baseline"))
    setup_changes = comparison["failed_setup_rows"]
    if setup_changes["added"] or setup_changes["resolved"]:
        reasons.append("failed_setup_rows_changed")
    if setup_changes["preserved"]:
        reasons.append("failed_setup_rows_preserved")
    no_directional_claim = bool(reasons)
    comparison["comparability"] = {
        "comparable": not no_directional_claim,
        "no_directional_claim": no_directional_claim,
        "claim_marker": "no directional claim" if no_directional_claim else "directional comparison allowed",
        "reasons": sorted(set(reasons)),
    }
    comparison["docs_summary"] = regression_docs_summary(comparison)
    return comparison


def no_baseline_regression_sections(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "correctness": {"state": "no_baseline", "current": correctness_projection(summary)},
        "direct_diagnostic_batch": {"state": "no_baseline", "current_rows": sorted(direct_projection_map(summary))},
        "http_concurrency_pressure": {"state": "no_baseline", "current_cells": sorted(http_projection_map(summary))},
        "pegainfer_trace_pass": {"state": "no_baseline", "current_cells": sorted(trace_projection_map(summary))},
    }


def correctness_projection(summary: dict[str, Any]) -> dict[str, Any]:
    gate = summary.get("correctness_gate")
    if not isinstance(gate, dict):
        return {"missing": True}
    comparison = gate.get("comparison") if isinstance(gate.get("comparison"), dict) else {}
    return {
        "missing": False,
        "passed": gate.get("passed"),
        "classification": comparison.get("classification"),
        "warnings": comparison.get("warnings", []),
    }


def direct_projection_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in summary.get("direct_diagnostic_batch", []) or []:
        if not isinstance(row, dict):
            continue
        key = f"{row.get('backend')}/batch{row.get('batch_size')}"
        rows[key] = {
            "passed": row.get("passed"),
            "claim_bucket": row.get("claim_bucket"),
            "token_sha256": row.get("token_sha256"),
            "text_sha256": row.get("text_sha256"),
            "tpot_ms": row.get("tpot_ms"),
            "output_tok_s": row.get("output_tok_s"),
            "error": row.get("error"),
        }
    return rows


def http_projection_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cells = {}
    for row in summary.get("http_concurrency_pressure", []) or []:
        if not isinstance(row, dict):
            continue
        if not row.get("summary_by_concurrency"):
            cells[f"{row.get('engine')}/setup"] = {
                "row_passed": row.get("passed"),
                "claim_bucket": row.get("claim_bucket"),
                "error": row.get("error"),
                "startup_failure": row.get("startup_failure"),
            }
            continue
        for cell in row.get("summary_by_concurrency", []) or []:
            if not isinstance(cell, dict):
                continue
            cells[f"{row.get('engine')}/c{cell.get('concurrency')}"] = {
                "row_passed": row.get("passed"),
                "claim_bucket": row.get("claim_bucket"),
                "completed": cell.get("completed"),
                "failed": cell.get("failed"),
                "timeouts": cell.get("timeouts"),
                "output_text_sha256": sorted(str(value) for value in cell.get("output_text_sha256", [])),
                "noisy": cell.get("noisy"),
                "mean_tpot_ms": metric_triplet(cell.get("mean_tpot_ms")),
                "output_tok_s": metric_triplet(cell.get("output_tok_s")),
            }
    return cells


def trace_projection_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cells = {}
    for row in summary.get("pegainfer_trace_pass", []) or []:
        if not isinstance(row, dict):
            continue
        if not row.get("cells"):
            cells[f"{row.get('engine')}/setup"] = {
                "row_passed": row.get("passed"),
                "claim_bucket": row.get("claim_bucket"),
                "error": row.get("error"),
                "startup_failure": row.get("startup_failure"),
            }
            continue
        for cell in row.get("cells", []) or []:
            if not isinstance(cell, dict):
                continue
            trace = cell.get("trace") if isinstance(cell.get("trace"), dict) else {}
            cells[f"{row.get('engine')}/c{cell.get('concurrency')}"] = {
                "row_passed": row.get("passed"),
                "claim_bucket": row.get("claim_bucket"),
                "completed": cell.get("completed"),
                "failed": cell.get("failed"),
                "timeouts": cell.get("timeouts"),
                "missing_trace_count": cell.get("missing_trace_count"),
                "active_set_size_max": trace.get("active_set_size_max"),
                "decode_batch_size_max": trace.get("decode_batch_size_max"),
                "trace_sha256": stable_json_sha256(trace),
            }
    return cells


def compare_single_projection(
    current: dict[str, Any],
    baseline: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    if current.get("missing") and baseline.get("missing"):
        return {"state": "missing_both", "changed_fields": []}
    if current.get("missing"):
        return {"state": "missing_current", "changed_fields": list(fields), "baseline": baseline}
    if baseline.get("missing"):
        return {"state": "missing_baseline", "changed_fields": list(fields), "current": current}
    changed = [field for field in fields if current.get(field) != baseline.get(field)]
    return {
        "state": "changed" if changed else "unchanged",
        "changed_fields": changed,
        "current": current,
        "baseline": baseline,
    }


def compare_projection_maps(
    current: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    current_keys = set(current)
    baseline_keys = set(baseline)
    changed = []
    for key in sorted(current_keys & baseline_keys):
        changed_fields = [field for field in fields if current[key].get(field) != baseline[key].get(field)]
        if changed_fields:
            changed.append({
                "key": key,
                "changed_fields": changed_fields,
                "current": current[key],
                "baseline": baseline[key],
            })
    return {
        "state": "changed" if changed or current_keys != baseline_keys else "unchanged",
        "added": sorted(current_keys - baseline_keys),
        "missing": sorted(baseline_keys - current_keys),
        "changed": changed,
        "unchanged": sorted((current_keys & baseline_keys) - {row["key"] for row in changed}),
    }


def structural_projection_reasons(section: str, comparison: dict[str, Any]) -> list[str]:
    reasons = []
    for direction in ("added", "missing"):
        for key in comparison.get(direction, []) or []:
            reasons.append(f"{section}_{direction}:{key}")
    return reasons


def comparability_reasons(summary: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    reasons = []
    current_meta = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
    baseline_meta = baseline.get("metadata") if isinstance(baseline.get("metadata"), dict) else {}
    for label, key_path in (
        ("benchmark_contract_changed", ("benchmark_contract",)),
        ("model_snapshot_changed", ("model", "config_sha256")),
        ("tokenizer_snapshot_changed", ("model", "tokenizer_sha256")),
        ("cuda_probe_changed", ("versions", "nvcc", "stdout")),
        ("vllm_version_changed", ("versions", "vllm", "stdout")),
    ):
        if nested_get(current_meta, key_path) != nested_get(baseline_meta, key_path):
            reasons.append(label)
    if version_probe_projection(current_meta, "nccl") != version_probe_projection(
        baseline_meta,
        "nccl",
    ):
        reasons.append("nccl_version_changed")
    if gpu_probe_projection(current_meta) != gpu_probe_projection(baseline_meta):
        reasons.append("gpu_probe_changed")
    return reasons


def version_probe_projection(metadata_payload: dict[str, Any], probe_name: str) -> Any:
    versions = metadata_payload.get("versions") if isinstance(metadata_payload, dict) else {}
    probe = versions.get(probe_name) if isinstance(versions, dict) else None
    if not isinstance(probe, dict):
        return None
    return {
        "available": probe.get("available"),
        "exit_code": probe.get("exit_code"),
        "stdout": probe.get("stdout"),
    }


def gpu_probe_projection(metadata_payload: dict[str, Any]) -> Any:
    versions = metadata_payload.get("versions") if isinstance(metadata_payload, dict) else {}
    probe = versions.get("nvidia_smi") if isinstance(versions, dict) else None
    if not isinstance(probe, dict):
        return None
    if probe.get("available") is False:
        return {"available": False}
    stdout = probe.get("stdout")
    if not isinstance(stdout, str):
        return None
    rows = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) >= 3:
            rows.append({
                "name": cells[0],
                "driver_version": cells[1],
                "compute_cap": cells[2],
            })
        else:
            rows.append({"raw": line.strip()})
    if rows:
        return rows
    return stdout.strip()


def nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def noise_reasons(summary: dict[str, Any], label: str) -> list[str]:
    reasons = []
    for key, row in http_projection_map(summary).items():
        if row.get("noisy"):
            reasons.append(f"{label}_noisy_http_cell:{key}")
    return reasons


def compare_failed_setup(
    summary: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    current = failed_setup_map(summary)
    previous = failed_setup_map(baseline or {})
    added = sorted(set(current) - set(previous))
    resolved = sorted(set(previous) - set(current))
    preserved = sorted(set(current) & set(previous))
    return {
        "added": [{"key": key, **current[key]} for key in added],
        "resolved": [{"key": key, **previous[key]} for key in resolved],
        "preserved": [{"key": key, **current[key]} for key in preserved],
    }


def failed_setup_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    gate = summary.get("correctness_gate")
    if isinstance(gate, dict) and (gate.get("claim_bucket") == CLAIM_FAILED or gate.get("passed") is False):
        rows["correctness_gate"] = {"section": "correctness_gate", "error": gate.get("error")}
    for row in summary.get("direct_diagnostic_batch", []) or []:
        if isinstance(row, dict) and (row.get("claim_bucket") == CLAIM_FAILED or row.get("passed") is False):
            key = f"direct:{row.get('backend')}/batch{row.get('batch_size')}"
            rows[key] = {
                "section": "direct_diagnostic_batch",
                "error": row.get("error"),
                "claim_bucket": row.get("claim_bucket"),
            }
    for section in ("http_concurrency_pressure", "pegainfer_trace_pass"):
        for row in summary.get(section, []) or []:
            if isinstance(row, dict) and (row.get("claim_bucket") == CLAIM_FAILED or row.get("passed") is False):
                key = f"{section}:{row.get('engine')}"
                rows[key] = {
                    "section": section,
                    "error": row.get("error"),
                    "startup_failure": row.get("startup_failure"),
                    "claim_bucket": row.get("claim_bucket"),
                }
    return rows


def regression_docs_summary(regression: dict[str, Any]) -> list[str]:
    lines = []
    correctness = regression.get("correctness", {})
    lines.append(f"correctness: {correctness.get('state', 'unknown')}")
    direct = regression.get("direct_diagnostic_batch", {})
    lines.append(
        "direct diagnostics: "
        f"{len(direct.get('changed', []))} changed, "
        f"{len(direct.get('missing', []))} missing, "
        f"{len(direct.get('added', []))} added"
    )
    http = regression.get("http_concurrency_pressure", {})
    lines.append(
        "HTTP pressure: "
        f"{len(http.get('changed', []))} changed, "
        f"{len(http.get('missing', []))} missing, "
        f"{len(http.get('added', []))} added"
    )
    trace = regression.get("pegainfer_trace_pass", {})
    lines.append(
        "trace: "
        f"{len(trace.get('changed', []))} changed, "
        f"{len(trace.get('missing', []))} missing, "
        f"{len(trace.get('added', []))} added"
    )
    failed = regression.get("failed_setup_rows", {})
    lines.append(
        "failed setup rows: "
        f"{len(failed.get('added', []))} added, "
        f"{len(failed.get('resolved', []))} resolved, "
        f"{len(failed.get('preserved', []))} preserved"
    )
    marker = regression.get("comparability", {}).get("claim_marker")
    if marker:
        lines.append(marker)
    return lines


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = repo_path(args.out_dir) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    sections: dict[str, Any] = {}
    if not args.skip_correctness:
        sections["correctness_gate"] = run_correctness_gate(args, out_dir)
    if not args.skip_direct:
        sections["direct_diagnostic_batch"] = run_direct_diagnostic(args, out_dir)
    if not args.skip_http:
        sections["http_concurrency_pressure"] = run_http_matrix(args, out_dir)
    if args.pegainfer_trace_pass:
        sections["pegainfer_trace_pass"] = run_pegainfer_trace_pass(args, out_dir)
    summary = build_summary(args, out_dir, sections)
    write_json(out_dir / "summary.json", summary)
    emit_benchmark_artifacts(args, out_dir, summary)
    return summary


def summarize_existing(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = repo_path(args.summarize_only)
    sections: dict[str, Any] = {}
    existing_summary_path = out_dir / "summary.json"
    existing_summary = load_json(existing_summary_path) if existing_summary_path.exists() else {}
    correctness = out_dir / "correctness" / "comparison.json"
    if correctness.exists():
        comparison = load_json(correctness)
        passed = correctness_passed(comparison)
        existing_gate = existing_summary.get("correctness_gate", {})
        existing_artifacts = existing_gate.get("artifacts", {}) if isinstance(existing_gate, dict) else {}
        artifacts = redact_payload(existing_artifacts) if isinstance(existing_artifacts, dict) else {}
        artifacts["comparison"] = display_path(correctness)
        sections["correctness_gate"] = {
            "claim_bucket": CLAIM_CORRECTNESS if passed else CLAIM_FAILED,
            "passed": passed,
            "artifacts": artifacts,
            "comparison": comparison,
        }
        if isinstance(existing_gate, dict) and "commands" in existing_gate:
            sections["correctness_gate"]["commands"] = redact_payload(existing_gate["commands"])
    elif existing_summary.get("correctness_gate"):
        sections["correctness_gate"] = existing_summary["correctness_gate"]
    direct_rows = []
    for artifact in sorted((out_dir / "direct_diagnostic_batch").glob("*/*.json")):
        payload = load_json(artifact)
        backend = payload.get("backend") or artifact.parent.name
        parsed_batch_size = first_int(payload, "config.batch_size")
        batch_size = parsed_batch_size if parsed_batch_size is not None else batch_size_from_path(artifact)
        row = {
            "claim_bucket": CLAIM_DIRECT,
            "backend": backend,
            "batch_size": batch_size,
            "artifact": display_path(artifact),
            "passed": bool(backend and batch_size),
        }
        row.update(parse_direct_artifact(payload))
        if not row["passed"]:
            row["claim_bucket"] = CLAIM_FAILED
            row["error"] = "direct diagnostic artifact is missing backend or positive batch size"
        direct_rows.append(row)
    direct_rows = merge_preserved_failed_rows(
        merge_existing_row_context(
            direct_rows,
            existing_summary.get("direct_diagnostic_batch", []),
            ("backend", "batch_size"),
            ("command", "env"),
        ),
        preserved_failed_rows(existing_summary.get("direct_diagnostic_batch", [])),
        ("backend", "batch_size"),
    )
    sections["direct_diagnostic_batch"] = direct_rows

    http_rows: list[dict[str, Any]] = []
    for engine_dir in sorted((out_dir / "http_raw").glob("*")):
        engine = {
            "engine": engine_dir.name,
            "claim_bucket": CLAIM_HTTP,
            "cells": [],
        }
        for artifact in sorted(engine_dir.glob("c*/r*/result.json")):
            payload = load_json(artifact)
            cell = {
                "artifact": display_path(artifact),
                "concurrency": int(artifact.parents[1].name.removeprefix("c")),
                "repeat": int(artifact.parent.name.removeprefix("r")),
            }
            cell.update(
                parse_vllm_bench_artifact(
                    payload,
                    expected_http_workload(args, cell["concurrency"]),
                )
            )
            engine["cells"].append(cell)
        expected_cells = {
            (concurrency, repeat)
            for concurrency in args.concurrency
            for repeat in range(args.repeats)
        }
        observed_cells = {
            (cell.get("concurrency"), cell.get("repeat"))
            for cell in engine["cells"]
            if isinstance(cell, dict)
        }
        missing_cells = sorted(expected_cells - observed_cells) if engine["cells"] else []
        engine["passed"] = bool(engine["cells"]) and not missing_cells and all(
            cell.get("passed") for cell in engine["cells"]
        )
        if not engine["cells"]:
            engine.update({
                "claim_bucket": CLAIM_FAILED,
                "error": "no HTTP benchmark result artifacts found",
            })
        elif missing_cells:
            engine["missing_result_cells"] = [
                {"concurrency": concurrency, "repeat": repeat}
                for concurrency, repeat in missing_cells
            ]
            engine.update({
                "claim_bucket": CLAIM_FAILED,
                "error": "missing HTTP benchmark result artifacts",
            })
        http_rows.append(engine)
    http_rows.extend(infer_failed_http_rows_from_logs(out_dir, http_rows))
    summarized_http_rows = summarize_http_rows(http_rows, args.noisy_threshold)
    summarized_http_rows = merge_preserved_failed_rows(
        merge_existing_row_context(
            summarized_http_rows,
            existing_summary.get("http_concurrency_pressure", []),
            ("engine",),
            ("label", "family", "server_command", "server_log", "server_env"),
        ),
        preserved_failed_rows(existing_summary.get("http_concurrency_pressure", [])),
        ("engine",),
    )
    summarized_http_rows.sort(key=engine_order_key)
    sections["http_concurrency_pressure"] = summarized_http_rows
    trace_rows: list[dict[str, Any]] = []
    for engine_dir in sorted((out_dir / "pegainfer_trace").glob("*")):
        row = {
            "engine": engine_dir.name,
            "claim_bucket": CLAIM_HTTP,
            "cells": [],
        }
        for artifact in sorted(engine_dir.glob("c*.json")):
            payload = load_json(artifact)
            cell = {
                "concurrency": int(artifact.stem.removeprefix("c")),
                "artifact": display_path(artifact),
                "completed": payload.get("summary", {}).get("completed"),
                "failed": payload.get("summary", {}).get("failed"),
                "timeouts": payload.get("summary", {}).get("timeouts"),
                "output_tok_s": payload.get("summary", {}).get("output_tokens_per_s"),
            }
            cell.update(trace_summary_for_payload(payload))
            cell["validation_errors"] = trace_cell_errors(
                cell,
                args,
                cell["concurrency"],
            )
            cell["passed"] = not cell["validation_errors"]
            row["cells"].append(cell)
        observed_concurrency = {
            cell.get("concurrency")
            for cell in row["cells"]
            if isinstance(cell, dict)
        }
        missing_concurrency = sorted(set(args.concurrency) - observed_concurrency) if row["cells"] else []
        row["passed"] = bool(row["cells"]) and not missing_concurrency and all(
            cell.get("passed") is True for cell in row["cells"]
        )
        if not row["cells"]:
            row.update({
                "claim_bucket": CLAIM_FAILED,
                "error": "no PegaInfer trace result artifacts found",
            })
        elif missing_concurrency:
            row["missing_trace_concurrency"] = missing_concurrency
            row.update({
                "claim_bucket": CLAIM_FAILED,
                "error": "missing PegaInfer trace result artifacts",
            })
        elif not row["passed"]:
            row.update({
                "claim_bucket": CLAIM_FAILED,
                "error": "trace cells failed strict coverage validation",
            })
        trace_rows.append(row)
    trace_rows.extend(infer_failed_trace_rows_from_logs(out_dir, trace_rows))
    trace_rows = merge_preserved_failed_rows(
        merge_existing_row_context(
            trace_rows,
            existing_summary.get("pegainfer_trace_pass", []),
            ("engine",),
            ("server_command", "server_log"),
        ),
        preserved_failed_rows(existing_summary.get("pegainfer_trace_pass", [])),
        ("engine",),
    )
    if trace_rows:
        sections["pegainfer_trace_pass"] = trace_rows
    elif existing_summary.get("pegainfer_trace_pass"):
        sections["pegainfer_trace_pass"] = [
            {
                **row,
                "passed": False,
                "claim_bucket": CLAIM_FAILED,
                "error": "raw PegaInfer trace artifacts are missing",
            }
            for row in existing_summary["pegainfer_trace_pass"]
            if isinstance(row, dict)
        ]
    summary = build_summary(args, out_dir, sections)
    if existing_summary.get("metadata"):
        summary["metadata"] = redact_payload(existing_summary["metadata"])
    write_json(out_dir / "summary.json", summary)
    emit_benchmark_artifacts(args, out_dir, summary)
    return summary


def engine_order_key(row: dict[str, Any]) -> tuple[int, int, str]:
    engine = row.get("engine")
    known = {spec.name: index for index, spec in enumerate(ENGINES)}
    passed_rank = 1 if row.get("passed") is True else 0
    return (known.get(engine, len(known)), passed_rank, str(engine))


def merge_existing_row_context(
    rows: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    context_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    existing_by_key = {
        tuple(row.get(field) for field in key_fields): row
        for row in existing_rows
        if isinstance(row, dict)
    }
    for row in rows:
        existing = existing_by_key.get(tuple(row.get(field) for field in key_fields))
        if not isinstance(existing, dict):
            continue
        for field in context_fields:
            if field in existing and field not in row:
                row[field] = redact_payload(existing[field])
        merge_existing_cell_context(row, existing)
    return rows


def merge_existing_cell_context(row: dict[str, Any], existing: dict[str, Any]) -> None:
    cells = row.get("cells")
    existing_cells = existing.get("cells")
    if not isinstance(cells, list) or not isinstance(existing_cells, list):
        return
    by_artifact = {
        cell.get("artifact"): cell
        for cell in existing_cells
        if isinstance(cell, dict) and cell.get("artifact")
    }
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        existing_cell = by_artifact.get(cell.get("artifact"))
        if not isinstance(existing_cell, dict):
            continue
        for field in ("command", "warmup_command", "exit_code"):
            if field in existing_cell and field not in cell:
                cell[field] = redact_payload(existing_cell[field])


def preserved_failed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if isinstance(row, dict) and (
            row.get("claim_bucket") == CLAIM_FAILED or row.get("passed") is False
        )
    ]


def infer_failed_http_rows_from_logs(
    out_dir: Path,
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_engines = {
        row.get("engine")
        for row in existing_rows
        if isinstance(row, dict) and row.get("engine")
    }
    inferred = []
    for log_path in sorted((out_dir / "server_logs").glob("*.log")):
        engine = log_path.stem
        if engine in existing_engines:
            continue
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        tail = log_text[-4000:]
        startup_failure = classify_server_start_failure(log_text)
        inferred.append(
            {
                "engine": engine,
                "claim_bucket": CLAIM_FAILED,
                "passed": False,
                "server_log": display_path(log_path),
                "error": startup_failure,
                "startup_failure": startup_failure,
                "server_log_tail": redact_text(tail),
                "cells": [],
            }
        )
    return inferred


def infer_failed_trace_rows_from_logs(
    out_dir: Path,
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_engines = {
        row.get("engine")
        for row in existing_rows
        if isinstance(row, dict) and row.get("engine")
    }
    inferred = []
    for log_path in sorted((out_dir / "trace_server_logs").glob("*.log")):
        engine = log_path.stem
        if engine in existing_engines:
            continue
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        tail = log_text[-4000:]
        startup_failure = classify_server_start_failure(log_text)
        inferred.append(
            {
                "engine": engine,
                "claim_bucket": CLAIM_FAILED,
                "passed": False,
                "server_log": display_path(log_path),
                "error": startup_failure,
                "startup_failure": startup_failure,
                "server_log_tail": redact_text(tail),
                "cells": [],
            }
        )
    return inferred


def classify_server_start_failure(log_text: str) -> str:
    if "ncclUnhandledCudaError" in log_text:
        return "server_start_failed: ncclUnhandledCudaError"
    if "could not determine the shape of object type 'torch.storage.UntypedStorage'" in log_text:
        return "server_start_failed: safetensors UntypedStorage shape inference"
    if "No such file or directory: 'ninja'" in log_text or "No such file or directory: \"ninja\"" in log_text:
        return "server_start_failed: missing ninja"
    if "SM 12.x requires CUDA >= 12.9" in log_text:
        return "server_start_failed: FlashInfer SM120 CUDA compatibility"
    if "FlashInfer requires GPUs with sm75 or higher" in log_text:
        return "server_start_failed: FlashInfer GPU capability detection"
    if "Engine core initialization failed" in log_text:
        return "server_start_failed: vLLM engine core initialization failed"
    if "Traceback" in log_text or "Error:" in log_text:
        return "server_start_failed: see server log"
    first_line = next((line.strip() for line in log_text.splitlines() if line.strip()), "")
    if first_line:
        return f"server_start_failed: {redact_text(first_line)[:120]}"
    return "server_start_failed: no HTTP benchmark result artifacts found"


def merge_preserved_failed_rows(
    rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows_by_key = {
        tuple(row.get(field) for field in key_fields): row
        for row in rows
    }
    unresolved_failed_rows = []
    for failed_row in failed_rows:
        key = tuple(failed_row.get(field) for field in key_fields)
        resolved_row = rows_by_key.get(key)
        if resolved_row is None:
            unresolved_failed_rows.append(failed_row)
            continue
        if resolved_row.get("passed") is True:
            resolved_row.setdefault("resolved_failed_setup_rows", []).append(redact_payload(failed_row))
        elif resolved_row.get("passed") is False:
            resolved_row.setdefault("previous_failed_setup_rows", []).append(redact_payload(failed_row))
    return rows + unresolved_failed_rows


def batch_size_from_path(path: Path) -> int | None:
    stem = path.stem
    if stem.startswith("batch") and stem[5:].isdigit():
        value = int(stem[5:])
        return value if value > 0 else None
    return None


def default_vllm_extra_args() -> list[str]:
    return [
        "--dtype", "bfloat16",
        "--enforce-eager",
        "--max-model-len", "512",
        "--gpu-memory-utilization", "0.70",
    ]


def normalize_vllm_serve_extra_args(raw: list[str] | None) -> list[str]:
    if raw is None:
        return default_vllm_extra_args()
    if raw and raw[0] == "--":
        return raw[1:]
    return raw


def split_vllm_serve_extra_args(argv: list[str]) -> tuple[list[str], list[str] | None]:
    marker = "--vllm-serve-extra-args"
    if marker not in argv:
        return argv, None
    index = argv.index(marker)
    return argv[:index], normalize_vllm_serve_extra_args(argv[index + 1:])


def parse_args() -> argparse.Namespace:
    script_argv, vllm_serve_extra_args = split_vllm_serve_extra_args(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--out-dir", type=Path, default=Path("target/benchmarks/deepseek-v2-lite-vllm-tp2-ep2"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--hf-python",
        help=(
            "Python executable for the HF correctness dump. If omitted, the script's "
            "Python is used; pass an explicit HF accuracy venv when vLLM and HF "
            "need different Transformers versions. Compare still uses this script's Python."
        ),
    )
    parser.add_argument("--vllm-cmd", default="vllm")
    parser.add_argument(
        "--vllm-serve-extra-args",
        nargs=argparse.REMAINDER,
        default=argparse.SUPPRESS,
        help=(
            "Extra arguments appended to vLLM serve. This passthrough option must be "
            "last; use '--vllm-serve-extra-args -- --flag value' if you prefer an "
            "explicit separator."
        ),
    )
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--num-warmups", type=int, default=4)
    parser.add_argument("--concurrency", type=parse_int_list, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--direct-batches", type=parse_int_list, default=DEFAULT_DIRECT_BATCHES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noisy-threshold", type=float, default=0.05)
    parser.add_argument("--server-ready-timeout-s", type=float, default=900)
    parser.add_argument("--command-timeout-s", type=float, default=1800)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-http", action="store_true")
    parser.add_argument("--pegainfer-trace-pass", action="store_true")
    parser.add_argument("--summarize-only", type=Path)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        help=(
            "Optional previous summary.json to compare against when writing "
            "regression_summary.json. Omit to emit a no-directional-claim summary."
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(script_argv)
    args.vllm_serve_extra_args = normalize_vllm_serve_extra_args(vllm_serve_extra_args)
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.num_prompts <= 0:
        raise SystemExit("--num-prompts must be positive")
    if args.num_warmups < 0:
        raise SystemExit("--num-warmups must be non-negative")
    return args


def plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "metadata": metadata(args, probe_versions=False),
        "baseline_summary": None if getattr(args, "baseline_summary", None) is None
        else str(args.baseline_summary),
        "warmup_policy": {
            "num_warmups": args.num_warmups,
            "mode": "separate vllm bench serve call before each measured repeat",
        },
        "correctness_commands": [
            {
                "label": "hf",
                "command": redact_command([
                    resolved_hf_python(args),
                    "tools/accuracy/hf_dump_dsv2_lite_ep2_greedy.py",
                    "--model-path",
                    str(args.model_path),
                    "--case-set-json",
                    "test_data/deepseek-v2-lite-ep2-cases.json",
                    "--out",
                    "<artifact-dir>/correctness/hf.json",
                ]),
            },
            {
                "label": "compare",
                "command": redact_command([
                    sys.executable,
                    "tools/accuracy/compare_dsv2_lite_ep2_outputs.py",
                    "--hf",
                    "<artifact-dir>/correctness/hf.json",
                    "--host-staged",
                    "<artifact-dir>/correctness/host-staged.json",
                    "--nccl",
                    "<artifact-dir>/correctness/nccl.json",
                    "--out",
                    "<artifact-dir>/correctness/comparison.json",
                    "--require-all-exact",
                ]),
            },
        ],
        "direct_commands": [
            {
                "backend": backend,
                "batch_size": batch,
                "command": redact_command([
                    "cargo", "run", "--release", "-p", "pegainfer-deepseek-v2-lite",
                    "--features", "deepseek-v2-lite", "--bin", "dsv2_lite_ep2_decode_attribution",
                    "--", "--model-path", str(args.model_path), "--batch-size", str(batch),
                ]),
            }
            for backend in ("host-staged", "nccl")
            for batch in args.direct_batches
        ],
        "servers": [
            {
                "engine": spec.name,
                "port": args.port + spec_index,
                "command": redact_command(server_command(args, spec, args.port + spec_index)),
            }
            for spec_index, spec in enumerate(ENGINES)
        ],
        "http_bench_template": redact_command(vllm_bench_command(
            args,
            port=args.port,
            num_prompts=args.num_prompts,
            result_dir=Path("<result-dir>"),
            result_filename="result.json",
            max_concurrency=args.concurrency[0],
        )),
    }


def main() -> None:
    args = parse_args()
    if args.plan_only:
        print(json.dumps(plan(args), indent=2, sort_keys=True))
        return
    if args.summarize_only:
        summary = summarize_existing(args)
    else:
        summary = run_matrix(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    socket.setdefaulttimeout(30)
    main()
