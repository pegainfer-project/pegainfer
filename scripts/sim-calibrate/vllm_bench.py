#!/usr/bin/env python3
"""Drive the installed vllm-bench binary without owning request generation.

The calibration pipeline treats vllm-bench as an external workload/timing
producer.  This module deliberately keeps the command line and raw output
outside the manifest's sensitive identity fields; request-level identity is
added by the sidecar adapter in a later stage.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class VllmBenchError(RuntimeError):
    """Raised when vllm-bench cannot be used as a calibration driver."""


@dataclass(frozen=True)
class VllmBenchIdentity:
    """Stable executable identity recorded with raw campaign artifacts."""

    path: str
    sha256: str
    version: str
    help_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "version": self.version,
            "help_sha256": self.help_sha256,
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise VllmBenchError(f"failed to read vllm-bench binary {path}: {exc}") from exc


def discover_binary(*, repo_root: Path | None = None, explicit: Path | None = None) -> Path:
    """Find the project virtualenv binary, then a PATH fallback.

    The virtualenv is intentionally checked first so a globally installed
    vllm-bench cannot silently change a campaign.  Callers can pass an
    explicit path for a pinned CI artifact.
    """

    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise VllmBenchError(f"vllm-bench is not an executable file: {candidate}")
        return candidate
    root = (repo_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    virtualenv = root / ".venv" / "bin" / "vllm-bench"
    if virtualenv.is_file() and os.access(virtualenv, os.X_OK):
        return virtualenv
    fallback = shutil.which("vllm-bench")
    if fallback:
        return Path(fallback).resolve()
    raise VllmBenchError(
        f"vllm-bench was not found at {virtualenv}; install it in the project .venv"
    )


def inspect_binary(binary: Path) -> VllmBenchIdentity:
    """Read version/help once and return an immutable executable identity."""

    binary = binary.expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise VllmBenchError(f"vllm-bench is not an executable file: {binary}")
    try:
        version_run = subprocess.run(
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        help_run = subprocess.run(
            [str(binary), "--help"],
            check=False,
            capture_output=True,
            text=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise VllmBenchError(f"failed to inspect vllm-bench {binary}: {exc}") from exc
    if version_run.returncode != 0:
        raise VllmBenchError(
            f"vllm-bench --version failed with {version_run.returncode}: "
            f"{version_run.stderr.strip()[:300]}"
        )
    if help_run.returncode != 0:
        raise VllmBenchError(
            f"vllm-bench --help failed with {help_run.returncode}: "
            f"{help_run.stderr.decode(errors='replace').strip()[:300]}"
        )
    version = (version_run.stdout or version_run.stderr).strip().splitlines()
    rendered_version = version[0].strip() if version else "unknown"
    return VllmBenchIdentity(
        path=str(binary),
        sha256=_sha256_file(binary),
        version=rendered_version,
        help_sha256=_sha256_bytes(help_run.stdout),
    )


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VllmBenchError(f"{field} must be a positive integer")
    return value


def _cell_id(cell: dict[str, Any]) -> str:
    value = cell.get("cell_id") or cell.get("name")
    if not isinstance(value, str) or not value:
        raise VllmBenchError("workload cell requires cell_id or name")
    return value


def _common_command(
    binary: Path,
    cell: dict[str, Any],
    *,
    endpoint: str,
    model: str,
    result_dir: Path,
    result_filename: str,
    seed: int,
    request_id_prefix: str | None,
    tokenizer: str | None,
) -> list[str]:
    input_tokens = _positive_int(cell.get("input_tokens"), "cell.input_tokens")
    output_tokens = _positive_int(cell.get("output_tokens"), "cell.output_tokens")
    concurrency = _positive_int(cell.get("concurrency"), "cell.concurrency")
    repeats = _positive_int(cell.get("repeats", 1), "cell.repeats")
    warmups = cell.get("warmup_repeats", 0)
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise VllmBenchError("cell.warmup_repeats must be a non-negative integer")
    measured_prompts = repeats * concurrency
    warmup_requests = warmups * concurrency
    argv = [
        str(binary),
        "--backend",
        "openai",
        "--base-url",
        endpoint.rstrip("/"),
        "--model",
        model,
        "--num-prompts",
        str(measured_prompts),
        "--max-concurrency",
        str(concurrency),
        "--seed",
        str(seed),
        "--num-warmups",
        str(warmup_requests),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
        "--disable-shuffle",
        "--ignore-eos",
        # Keep the generated prompt as token IDs.  Decoding random IDs to
        # text and tokenizing again can change the requested length (notably
        # for Qwen vocabularies), which breaks per-cell pairing.
        "--prompt-token-ids",
    ]
    if request_id_prefix:
        argv.extend(("--request-id-prefix", request_id_prefix))
    if tokenizer:
        argv.extend(("--tokenizer", tokenizer))
    if cell.get("request_rate") is not None:
        argv.extend(("--request-rate", str(cell["request_rate"])))
    else:
        argv.extend(("--request-rate", "inf"))
    return argv


def render_command(
    cell: dict[str, Any],
    *,
    binary: Path,
    endpoint: str,
    model: str,
    result_dir: Path,
    result_filename: str | None = None,
    seed: int = 0,
    request_id_prefix: str | None = None,
    tokenizer: str | None = None,
    workload_path: Path | None = None,
) -> list[str]:
    """Render one deterministic vllm-bench invocation for a manifest cell."""

    kind = cell.get("kind")
    if kind not in {"cold_prefill", "steady_decode", "cached_prefill", "mixed"}:
        raise VllmBenchError(f"unsupported workload kind: {kind!r}")
    cell_name = _cell_id(cell)
    result_filename = result_filename or f"{cell_name}.json"
    argv = _common_command(
        binary,
        cell,
        endpoint=endpoint,
        model=model,
        result_dir=result_dir,
        result_filename=result_filename,
        seed=seed,
        request_id_prefix=request_id_prefix or f"sim-calibrate-{cell_name}",
        tokenizer=tokenizer,
    )
    if kind in {"cold_prefill", "steady_decode"}:
        argv.extend(
            (
                "--dataset-name",
                "random",
                "--random-input-len",
                str(_positive_int(cell.get("input_tokens"), "cell.input_tokens")),
                "--random-output-len",
                str(_positive_int(cell.get("output_tokens"), "cell.output_tokens")),
            )
        )
    elif kind == "cached_prefill":
        prefix = _positive_int(cell.get("prefix_tokens"), "cell.prefix_tokens")
        suffix = _positive_int(cell.get("new_tokens", cell.get("input_tokens", 0) - prefix), "cell.new_tokens")
        argv.extend(
            (
                "--dataset-name",
                "prefix_repetition",
                "--prefix-repetition-prefix-len",
                str(prefix),
                "--prefix-repetition-suffix-len",
                str(suffix),
                "--prefix-repetition-num-prefixes",
                "1",
                "--prefix-repetition-output-len",
                str(_positive_int(cell.get("output_tokens"), "cell.output_tokens")),
            )
        )
    else:
        if workload_path is None:
            raise VllmBenchError("mixed workload requires a custom JSONL workload_path")
        workload_path = workload_path.expanduser().resolve()
        if not workload_path.is_file():
            raise VllmBenchError(f"mixed workload file does not exist: {workload_path}")
        argv.extend(
            (
                "--dataset-name",
                "custom",
                "--dataset-path",
                str(workload_path),
                "--custom-output-len",
                "-1",
            )
        )
    if kind == "steady_decode":
        argv.extend(
            (
                "--steady-state-threshold",
                str(cell.get("steady_state_threshold", 0.95)),
            )
        )
        if cell.get("steady_state_min_window") is not None:
            argv.extend(("--steady-state-min-window", str(cell["steady_state_min_window"])))
    return argv


@dataclass(frozen=True)
class RunOutput:
    """Raw process output and paths, suitable for an external artifact index."""

    argv: tuple[str, ...]
    returncode: int
    result_path: Path
    stdout_path: Path
    stderr_path: Path
    stdout_sha256: str
    stderr_sha256: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and self.result_path.is_file()


def run_command(
    argv: Sequence[str],
    *,
    result_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout: float | None = None,
    cwd: Path | None = None,
) -> RunOutput:
    """Run one already-rendered command and persist every raw stream."""

    if not argv:
        raise VllmBenchError("vllm-bench argv cannot be empty")
    for path in (result_path, stdout_path, stderr_path):
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd.expanduser().resolve()) if cwd else None,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise VllmBenchError(f"vllm-bench command failed to start: {exc}") from exc
    stdout_path.write_bytes(completed.stdout)
    stderr_path.write_bytes(completed.stderr)
    return RunOutput(
        argv=tuple(argv),
        returncode=completed.returncode,
        result_path=result_path.expanduser().resolve(),
        stdout_path=stdout_path.expanduser().resolve(),
        stderr_path=stderr_path.expanduser().resolve(),
        stdout_sha256=_sha256_bytes(completed.stdout),
        stderr_sha256=_sha256_bytes(completed.stderr),
    )


def provenance(identity: VllmBenchIdentity, output: RunOutput) -> dict[str, Any]:
    """Return a JSON-safe raw-driver provenance record."""

    return {
        "driver": "vllm-bench",
        "binary": identity.as_dict(),
        "argv": list(output.argv),
        "returncode": output.returncode,
        "raw": {
            "result_path": str(output.result_path),
            "stdout_path": str(output.stdout_path),
            "stderr_path": str(output.stderr_path),
            "stdout_sha256": output.stdout_sha256,
            "stderr_sha256": output.stderr_sha256,
        },
    }
