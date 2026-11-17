#!/usr/bin/env python3
"""Shared metadata and summary helpers for HTTP benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_FINGERPRINT_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def write_json(path: Path, document: Any) -> str:
    """Atomically replace a retained JSON artifact and return its rendered form."""
    rendered = json.dumps(document, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(rendered + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return rendered


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def model_fingerprint(model_path: str | None) -> dict[str, str]:
    if not model_path:
        return {}
    root = Path(model_path)
    return {
        name: digest
        for name in MODEL_FINGERPRINT_FILES
        if (digest := sha256_file(root / name)) is not None
    }


def current_commit() -> str | None:
    return run_text(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT)


def shell_command(argv: list[str]) -> str:
    """Render argv so artifact commands can be pasted back into a POSIX shell."""
    return shlex.join(argv)


def artifact_command(argv: list[str]) -> str:
    """Render argv without leaking absolute paths inside the source checkout."""
    normalized = []
    for arg in argv:
        path = Path(arg)
        if path.is_absolute():
            try:
                arg = path.resolve().relative_to(REPO_ROOT).as_posix()
            except ValueError:
                pass
        normalized.append(arg)
    return shell_command(normalized)


def run_text(command: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def nvcc_version() -> str | None:
    commands = [["nvcc", "--version"]]
    roots = [
        Path(value)
        for name in ("CUDA_HOME", "CUDA_PATH")
        if (value := os.environ.get(name))
    ]
    roots.extend([Path("/usr/local/cuda"), *sorted(Path("/usr/local").glob("cuda-*"))])
    for root in roots:
        command = [str(root / "bin" / "nvcc"), "--version"]
        if command not in commands:
            commands.append(command)
    for command in commands:
        if value := run_text(command):
            return value
    return None


def detect_hardware_toolchain() -> dict[str, Any]:
    nvidia_smi = run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    gpu_uuids = run_text(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"])
    return {
        "gpu": nvidia_smi.splitlines() if nvidia_smi else [],
        "gpu_identity_sha256": [
            hashlib.sha256(uuid.strip().encode("utf-8")).hexdigest()
            for uuid in gpu_uuids.splitlines()
            if uuid.strip()
        ]
        if gpu_uuids
        else [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvcc_version": nvcc_version(),
        "rustc_version": run_text(["rustc", "--version"]),
        "cargo_version": run_text(["cargo", "--version"]),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def numeric_summary(
    values: list[float | int | None],
) -> dict[str, float | int | None]:
    clean = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    if not clean:
        return {"median": None, "min": None, "max": None, "samples": 0}
    return {
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
        "samples": len(clean),
    }


def repeat_noise_marker(
    value_groups: list[list[float | int | None]], repeats: int
) -> str:
    if repeats < 2:
        return "insufficient_repeats"
    for values in value_groups:
        summary = numeric_summary(values)
        median = summary["median"]
        if median is None or summary["samples"] < 2:
            continue
        spread = float(summary["max"]) - float(summary["min"])
        if float(median) == 0.0:
            if spread > 0.0:
                return "noisy"
            continue
        if spread / float(median) > 0.10:
            return "noisy"
    return "stable"


def value_counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def combined_output_hash(values: list[str]) -> str:
    return hashlib.sha256("".join(values).encode("utf-8")).hexdigest()[:16]
