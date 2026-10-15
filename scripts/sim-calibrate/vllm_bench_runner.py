#!/usr/bin/env python3
"""Run one manifest cell/variant through vllm-bench and retain its evidence.

The vllm-bench executable owns request generation and timing.  This adapter
owns campaign identity, sidecar creation, resumable attempt directories, and
conversion into the calibration measurement contract.  Raw files are kept in
the external :class:`ArtifactStore` root; the variant result envelope records
their relative paths and digests in its probe metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from artifacts import ArtifactError, ArtifactStore
from manifest import ManifestError
from vllm_bench import (
    RunOutput,
    VllmBenchError,
    discover_binary,
    inspect_binary,
    render_command,
    run_command,
)
from vllm_bench_collect import VllmBenchCollectError, convert_vllm_bench
from cached_prefix_collect import CachedPrefixError, convert_files as convert_cached_files
from mixed_orchestrator import (
    MixedError,
    build_mixed_workload,
    mark_overlap_metadata,
    two_process_commands,
    validate_overlap,
    write_custom_jsonl,
)
from vllm_bench_sidecar import SidecarError, build_sidecar, write_sidecar


VARIANTS = ("target", "zero_cost", "replay")
_RAW_RESULT_NAME = "vllm-bench-result.json"
_STDOUT_NAME = "stdout.log"
_STDERR_NAME = "stderr.log"
_SIDECAR_NAME = "sidecar.json"
_CONVERTED_NAME = "converted.json"
_COMMAND_NAME = "command.json"
_EVIDENCE_NAME = "evidence.json"
_OVERLAP_NAME = "overlap-evidence.json"


class VllmBenchRunnerError(RuntimeError):
    """Raised when a campaign cell cannot be planned or persisted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise VllmBenchRunnerError(f"failed to hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    rendered = (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_bytes(path, rendered)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VllmBenchRunnerError(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VllmBenchRunnerError(f"JSON artifact must be an object: {path}")
    return value


def _campaign_id(manifest: dict[str, Any]) -> str:
    value = manifest.get("campaign_id")
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise VllmBenchRunnerError("manifest.campaign_id must be a 64-character hexadecimal digest")
    return value


def _select_cell(manifest: dict[str, Any], selector: str) -> dict[str, Any]:
    workload = manifest.get("workload")
    cells = workload.get("cells") if isinstance(workload, dict) else None
    if not isinstance(cells, list):
        raise VllmBenchRunnerError("manifest.workload.cells must be an array")
    matches = [
        item
        for item in cells
        if isinstance(item, dict) and selector in {item.get("name"), item.get("cell_id")}
    ]
    if len(matches) != 1:
        raise VllmBenchRunnerError(f"workload cell is not unique: {selector!r}")
    return dict(matches[0])


def _variant_identity(manifest: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise VllmBenchRunnerError(f"unsupported variant: {variant!r}")
    variants = manifest.get("variants")
    if variants is None:
        if variant != "target":
            raise VllmBenchRunnerError(
                f"manifest without variants can only run target, not {variant}"
            )
        identity = manifest.get("target")
    else:
        if not isinstance(variants, dict) or not isinstance(variants.get(variant), dict):
            raise VllmBenchRunnerError(f"manifest has no identity for variant {variant}")
        identity = variants[variant].get("identity")
    if not isinstance(identity, dict):
        raise VllmBenchRunnerError(f"variant {variant} identity must be an object")
    endpoint = identity.get("endpoint")
    model = identity.get("model")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise VllmBenchRunnerError(f"variant {variant} identity.endpoint is required")
    if not isinstance(model, str) or not model.strip():
        raise VllmBenchRunnerError(f"variant {variant} identity.model is required")
    return dict(identity)


def _seed(manifest: dict[str, Any], override: int | None) -> int:
    value = override
    if value is None:
        client = manifest.get("client")
        value = client.get("seed", 0) if isinstance(client, dict) else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VllmBenchRunnerError("client seed must be a non-negative integer")
    return value


def _tokenizer(manifest: dict[str, Any], identity: dict[str, Any], override: str | None) -> str | None:
    value = override
    if value is None:
        client = manifest.get("client", {})
        if isinstance(client, dict):
            for key in ("tokenizer", "tokenizer_path"):
                if client.get(key) is not None:
                    value = client[key]
                    break
    if value is None:
        for key in ("tokenizer", "tokenizer_path"):
            if identity.get(key) is not None:
                value = identity[key]
                break
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise VllmBenchRunnerError("tokenizer must be a non-empty path when present")
    return value


def _next_attempt(store: ArtifactStore, cell_id: str, variant: str) -> int:
    key = f"{cell_id}:{variant}"
    current = store.index.get("variant_cells", {}).get(key, {})
    attempts = current.get("attempts", []) if isinstance(current, dict) else []
    attempt = len(attempts) + 1
    # Mirror ArtifactStore.record_variant_cell's numbering rule so the raw
    # paths we prepare are the same attempt that the envelope will receive.
    # A killed process can leave an attempt directory without result.json;
    # treating that directory as free would make resume collide with it.
    while (store.cell_dir / cell_id / variant / f"attempt-{attempt:04d}").exists():
        attempt += 1
    return attempt


def _relative_artifact(store: ArtifactStore, path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        relative = path.relative_to(store.root).as_posix()
    except ValueError as exc:
        raise VllmBenchRunnerError(f"artifact path escapes artifact root: {path}") from exc
    if not path.is_file():
        raise VllmBenchRunnerError(f"artifact file is missing: {path}")
    return {"path": relative, "sha256": _sha256(path), "size": path.stat().st_size}


def _failure_result(
    cell: dict[str, Any],
    variant: str,
    *,
    status: str,
    error: Exception | str,
) -> dict[str, Any]:
    message = str(error)[:1000]
    return {
        "schema_version": 1,
        "collector": "vllm-bench",
        "variant": variant,
        "cell": dict(cell),
        "status": status,
        "measurements": [],
        "pairing": {},
        "failures": [{"kind": type(error).__name__ if isinstance(error, Exception) else "RunnerError", "error": message}],
    }


def _plan(
    manifest: dict[str, Any],
    cell: dict[str, Any],
    variant: str,
    *,
    binary: Path,
    result_dir: Path,
    seed: int,
    tokenizer: str | None,
    workload_path: Path | None,
) -> list[str]:
    identity = _variant_identity(manifest, variant)
    try:
        return render_command(
            cell,
            binary=binary,
            endpoint=identity["endpoint"],
            model=identity["model"],
            result_dir=result_dir,
            result_filename=_RAW_RESULT_NAME,
            seed=seed,
            request_id_prefix=f"sim-calibrate-{cell['cell_id']}-{variant}",
            tokenizer=tokenizer,
            workload_path=workload_path,
        )
    except (KeyError, TypeError, VllmBenchError) as exc:
        raise VllmBenchRunnerError(f"failed to render vllm-bench command: {exc}") from exc


def plan_cell_variant(
    manifest: dict[str, Any],
    *,
    cell_selector: str,
    variant: str,
    binary: Path | None = None,
    artifact_root: Path | None = None,
    seed: int | None = None,
    tokenizer: str | None = None,
    workload_path: Path | None = None,
) -> dict[str, Any]:
    """Render a one-cell command and sidecar without contacting the endpoint."""

    campaign_id = _campaign_id(manifest)
    cell = _select_cell(manifest, cell_selector)
    identity = _variant_identity(manifest, variant)
    resolved_binary = discover_binary(explicit=binary) if binary is not None else discover_binary()
    resolved_seed = _seed(manifest, seed)
    resolved_tokenizer = _tokenizer(manifest, identity, tokenizer)
    if artifact_root is None:
        result_dir = Path.cwd() / "raw"
    else:
        result_dir = artifact_root.expanduser().resolve() / "cells" / cell["cell_id"] / variant / "attempt-0001"
    argv = _plan(
        manifest,
        cell,
        variant,
        binary=resolved_binary,
        result_dir=result_dir,
        seed=resolved_seed,
        tokenizer=resolved_tokenizer,
        workload_path=workload_path,
    )
    try:
        sidecar = build_sidecar(cell, ordering="stable_ordinal")
    except (KeyError, TypeError, SidecarError) as exc:
        raise VllmBenchRunnerError(f"failed to build request sidecar: {exc}") from exc
    return {
        "campaign_id": campaign_id,
        "cell_id": cell["cell_id"],
        "cell_name": cell.get("name"),
        "variant": variant,
        "binary": str(resolved_binary),
        "endpoint": identity["endpoint"],
        "model": identity["model"],
        "seed": resolved_seed,
        "argv": argv,
        "sidecar": sidecar,
        "result_dir": str(result_dir),
    }


def run_cell_variant(
    manifest: dict[str, Any],
    cell: dict[str, Any],
    variant: str,
    store: ArtifactStore,
    *,
    binary: Path | None = None,
    seed: int | None = None,
    tokenizer: str | None = None,
    workload_path: Path | None = None,
    timeout: float | None = None,
    dry_run: bool = False,
    resume: bool = True,
    collector_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run and persist one cell/variant, or return its deterministic dry-run plan."""

    campaign_id = _campaign_id(manifest)
    if store.campaign_id != campaign_id:
        raise VllmBenchRunnerError("artifact store campaign_id does not match manifest")
    if not isinstance(cell, dict) or not cell.get("cell_id"):
        raise VllmBenchRunnerError("cell must contain a cell_id")
    selected = _select_cell(manifest, str(cell["cell_id"]))
    if selected.get("cell_id") != cell.get("cell_id"):
        raise VllmBenchRunnerError("cell does not match the manifest workload identity")
    cell = selected
    identity = _variant_identity(manifest, variant)
    resolved_seed = _seed(manifest, seed)
    resolved_tokenizer = _tokenizer(manifest, identity, tokenizer)
    resolved_binary = discover_binary(explicit=binary) if binary is not None else discover_binary()

    existing = store.variant_status(cell["cell_id"], variant)
    if dry_run:
        attempt = _next_attempt(store, cell["cell_id"], variant)
        result_dir = store.cell_dir / cell["cell_id"] / variant / f"attempt-{attempt:04d}"
        argv = _plan(
            manifest,
            cell,
            variant,
            binary=resolved_binary,
            result_dir=result_dir,
            seed=resolved_seed,
            tokenizer=resolved_tokenizer,
            workload_path=workload_path,
        )
        sidecar = build_sidecar(cell, ordering="stable_ordinal")
        return {
            "campaign_id": campaign_id,
            "cell_id": cell["cell_id"],
            "variant": variant,
            "status": "dry_run",
            "would_skip": existing == "sealed" and resume,
            "attempt": attempt,
            "binary": str(resolved_binary),
            "argv": argv,
            "sidecar": sidecar,
        }
    if existing == "sealed":
        if resume:
            return {
                "campaign_id": campaign_id,
                "cell_id": cell["cell_id"],
                "variant": variant,
                "status": "sealed",
                "skipped": True,
                "entry": store.index["variant_cells"][f"{cell['cell_id']}:{variant}"],
            }
        raise VllmBenchRunnerError(
            f"variant cell is already sealed: {cell['cell_id']}:{variant}; use resume to skip it"
        )

    if cell.get("kind") == "cached_prefill" and variant == "target":
        return _run_cached_target(
            manifest, cell, variant, store,
            binary=resolved_binary, seed=resolved_seed, tokenizer=resolved_tokenizer,
            timeout=timeout, dry_run=dry_run, campaign_id=campaign_id,
        )
    if cell.get("kind") == "mixed":
        return _run_mixed_cell(
            manifest, cell, variant, store,
            binary=resolved_binary, seed=resolved_seed, tokenizer=resolved_tokenizer, timeout=timeout,
            dry_run=dry_run, campaign_id=campaign_id,
        )

    try:
        binary_identity = inspect_binary(resolved_binary)
    except VllmBenchError:
        raise

    store.attach_manifest(manifest)
    store.begin_variant_cell(cell, variant)
    attempt = _next_attempt(store, cell["cell_id"], variant)
    attempt_dir = store.cell_dir / cell["cell_id"] / variant / f"attempt-{attempt:04d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    result_path = attempt_dir / _RAW_RESULT_NAME
    stdout_path = attempt_dir / _STDOUT_NAME
    stderr_path = attempt_dir / _STDERR_NAME
    sidecar_path = attempt_dir / _SIDECAR_NAME
    converted_path = attempt_dir / _CONVERTED_NAME
    command_path = attempt_dir / _COMMAND_NAME
    try:
        sidecar = build_sidecar(cell, ordering="stable_ordinal")
        write_sidecar(sidecar_path, sidecar)
        argv = _plan(
            manifest,
            cell,
            variant,
            binary=resolved_binary,
            result_dir=attempt_dir,
            seed=resolved_seed,
            tokenizer=resolved_tokenizer,
            workload_path=workload_path,
        )
        _write_json(command_path, {"argv": argv, "variant": variant, "cell_id": cell["cell_id"]})
    except (KeyError, TypeError, SidecarError, VllmBenchRunnerError) as exc:
        _write_bytes(stdout_path, b"")
        _write_bytes(stderr_path, str(exc).encode("utf-8", errors="replace"))
        if not sidecar_path.is_file():
            _write_json(
                sidecar_path,
                {
                    "schema_version": 1,
                    "collector": "vllm-bench",
                    "cell_id": cell["cell_id"],
                    "ordering": "unavailable",
                    "requests": [],
                    "request_count": 0,
                    "warmup_requests": 0,
                    "measured_requests": 0,
                    "includes_warmups": False,
                },
            )
        _write_json(command_path, {"argv": [], "variant": variant, "cell_id": cell["cell_id"], "error": str(exc)})
        converted = _failure_result(cell, variant, status="failed", error=exc)
        _write_json(converted_path, converted)
        probe = {
            "kind": "vllm-bench",
            "status": "not_started",
            "variant": variant,
            "inputs": {
                "sidecar": _relative_artifact(store, sidecar_path),
                "stdout": _relative_artifact(store, stdout_path),
                "stderr": _relative_artifact(store, stderr_path),
                "converted": _relative_artifact(store, converted_path),
            },
        }
        entry = store.record_variant_cell(cell, variant, converted, collector_argv=list(collector_argv or sys.argv), probe=probe)
        return _outcome(campaign_id, cell, variant, entry, attempt, probe)

    run_output: RunOutput | None = None
    run_error: Exception | None = None
    try:
        run_output = run_command(
            argv,
            result_path=result_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=timeout,
        )
    except Exception as exc:  # retain diagnostics as a failed attempt
        run_error = exc
        _write_bytes(stdout_path, b"")
        _write_bytes(stderr_path, str(exc).encode("utf-8", errors="replace"))

    status = "succeeded"
    conversion_error: Exception | None = None
    if run_error is not None:
        status = "failed"
        converted = _failure_result(cell, variant, status=status, error=run_error)
    elif run_output is None or run_output.returncode != 0:
        status = "failed"
        detail = f"vllm-bench exited with return code {run_output.returncode if run_output else 'unknown'}"
        converted = _failure_result(cell, variant, status=status, error=detail)
    elif not result_path.is_file():
        status = "failed"
        converted = _failure_result(cell, variant, status=status, error="vllm-bench did not write a result JSON")
    else:
        try:
            converted = convert_vllm_bench(
                _read_json(result_path),
                _read_json(sidecar_path),
                cell=cell,
                variant=variant,
            )
        except (VllmBenchCollectError, VllmBenchRunnerError, OSError, ValueError) as exc:
            status = "invalid"
            conversion_error = exc
            converted = _failure_result(cell, variant, status=status, error=exc)
    _write_json(converted_path, converted)

    inputs: dict[str, dict[str, Any]] = {
        "stdout": _relative_artifact(store, stdout_path),
        "stderr": _relative_artifact(store, stderr_path),
        "sidecar": _relative_artifact(store, sidecar_path),
        "converted": _relative_artifact(store, converted_path),
        "command": _relative_artifact(store, command_path),
    }
    if result_path.is_file():
        inputs["vllm_bench_result"] = _relative_artifact(store, result_path)
    probe = {
        "kind": "vllm-bench",
        "status": "completed" if run_error is None else "failed_to_start",
        "variant": variant,
        "binary": binary_identity.as_dict(),
        "returncode": run_output.returncode if run_output is not None else None,
        "conversion_error": str(conversion_error)[:1000] if conversion_error else None,
        "inputs": inputs,
    }
    entry = store.record_variant_cell(
        cell,
        variant,
        converted,
        collector_argv=list(collector_argv or sys.argv),
        probe=probe,
    )
    if status == "succeeded":
        store.seal_variant_cell(cell["cell_id"], variant)
        entry = store.index["variant_cells"][f"{cell['cell_id']}:{variant}"]
    return _outcome(campaign_id, cell, variant, entry, attempt, probe)


def _outcome(
    campaign_id: str,
    cell: dict[str, Any],
    variant: str,
    entry: dict[str, Any],
    attempt: int,
    probe: dict[str, Any],
) -> dict[str, Any]:
    return {
        "campaign_id": campaign_id,
        "cell_id": cell["cell_id"],
        "cell_name": cell.get("name"),
        "variant": variant,
        "status": entry.get("status"),
        "attempt": attempt,
        "entry": entry,
        "artifacts": probe.get("inputs", {}),
    }


def _record_special_failure(
    store: ArtifactStore,
    cell: dict[str, Any],
    variant: str,
    attempt: int,
    campaign_id: str,
    attempt_dir: Path,
    error: Exception | str,
    *,
    collector: str,
    inputs: dict[str, dict[str, Any]],
    command: list[str] | None = None,
) -> dict[str, Any]:
    converted_path = attempt_dir / _CONVERTED_NAME
    converted = _failure_result(cell, variant, status="failed", error=error)
    converted["collector"] = collector
    _write_json(converted_path, converted)
    inputs = dict(inputs)
    inputs["converted"] = _relative_artifact(store, converted_path)
    if command is not None:
        command_path = attempt_dir / _COMMAND_NAME
        _write_json(command_path, {"argv": command, "variant": variant, "cell_id": cell["cell_id"]})
        inputs["command"] = _relative_artifact(store, command_path)
    probe = {"kind": collector, "status": "failed", "variant": variant, "inputs": inputs}
    entry = store.record_variant_cell(
        cell,
        variant,
        converted,
        collector_argv=list(sys.argv),
        probe=probe,
    )
    return _outcome(campaign_id, cell, variant, entry, attempt, probe)


def _run_cached_target(
    manifest: dict[str, Any],
    cell: dict[str, Any],
    variant: str,
    store: ArtifactStore,
    *,
    binary: Path,
    seed: int,
    tokenizer: str | None,
    timeout: float | None,
    dry_run: bool,
    campaign_id: str,
) -> dict[str, Any]:
    """Run the target cached-prefix cell through the transparent recorder."""

    existing = store.variant_status(cell["cell_id"], variant)
    if dry_run:
        attempt = _next_attempt(store, cell["cell_id"], variant)
        result_dir = store.cell_dir / cell["cell_id"] / variant / f"attempt-{attempt:04d}"
        recorder_cell = {**cell, "warmup_repeats": 0, "repeats": int(cell.get("repeats", 1)) + 1}
        bench = _plan(
            manifest, recorder_cell, variant, binary=binary, result_dir=result_dir,
            seed=seed, tokenizer=tokenizer, workload_path=None,
        )
        return {
            "campaign_id": campaign_id, "cell_id": cell["cell_id"], "variant": variant,
            "status": "dry_run", "would_skip": existing == "sealed", "attempt": attempt,
            "collector": "cached-prefix-transparent-recorder", "argv": bench,
        }
    store.attach_manifest(manifest)
    store.begin_variant_cell(cell, variant)
    attempt = _next_attempt(store, cell["cell_id"], variant)
    attempt_dir = store.cell_dir / cell["cell_id"] / variant / f"attempt-{attempt:04d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    result_path = attempt_dir / _RAW_RESULT_NAME
    stdout_path = attempt_dir / _STDOUT_NAME
    stderr_path = attempt_dir / _STDERR_NAME
    sidecar_path = attempt_dir / _SIDECAR_NAME
    evidence_path = attempt_dir / _EVIDENCE_NAME
    cell_path = attempt_dir / "cell.json"
    custom_workload_path = attempt_dir / "cached-prefix.jsonl"
    # The first prefix-repetition request bootstraps the cache and is removed
    # by the recorder; ask vllm-bench for one extra request so the manifest's
    # measured repeat count remains intact.
    recorder_cell = {**cell, "warmup_repeats": 0, "repeats": int(cell.get("repeats", 1)) + 1}
    inputs: dict[str, dict[str, Any]] = {}
    try:
        sidecar = build_sidecar(cell, ordering="stable_ordinal")
        write_sidecar(sidecar_path, sidecar)
        _write_json(cell_path, cell)
        bench = _plan(
            manifest, recorder_cell, variant, binary=binary, result_dir=attempt_dir,
            seed=seed, tokenizer=tokenizer, workload_path=None,
        )
        identity = _variant_identity(manifest, variant)
        if "qwen35" in str(identity.get("model", "")).lower():
            if not tokenizer:
                raise VllmBenchRunnerError("Qwen3.5 cached custom workload requires a tokenizer path")
            from tokenizers import Tokenizer

            tokenizer_path = Path(tokenizer)
            if tokenizer_path.is_dir():
                tokenizer_path = tokenizer_path / "tokenizer.json"
            encoder = Tokenizer.from_file(str(tokenizer_path))
            prefix_tokens = int(cell["prefix_tokens"])
            new_tokens = int(cell.get("new_tokens", int(cell["input_tokens"]) - prefix_tokens))
            prefix_text = "a" + " a" * (prefix_tokens - 1)
            rows: list[dict[str, Any]] = []
            for index in range(int(recorder_cell["repeats"])):
                suffix_seed = f" sample{index:05d}"
                suffix_words = max(new_tokens - 2, 1)
                for _ in range(12):
                    suffix = suffix_seed + " a" * (suffix_words - 1)
                    prompt = prefix_text + suffix
                    measured = len(encoder.encode(prompt, add_special_tokens=False).ids)
                    if measured == prefix_tokens + new_tokens:
                        break
                    suffix_words = max(suffix_words + prefix_tokens + new_tokens - measured, 1)
                else:
                    raise VllmBenchRunnerError("Qwen3.5 cached custom prompt could not reach exact token length")
                rows.append({"prompt": prompt, "output_tokens": int(cell["output_tokens"])})
            write_custom_jsonl(custom_workload_path, rows)
            dataset_index = bench.index("--dataset-name")
            bench = [arg for arg in bench[:dataset_index] if arg != "--prompt-token-ids"] + [
                "--dataset-name", "custom", "--dataset-path", str(custom_workload_path),
                "--custom-output-len", "-1",
            ]
        recorder = Path(__file__).resolve().with_name("cached_prefix_recorder.py")
        prefix = cell.get("shared_prefix")
        prefix_id = prefix.get("identity") if isinstance(prefix, dict) else None
        if not isinstance(prefix_id, str) or not prefix_id:
            raise VllmBenchRunnerError("cached-prefix cell has no shared_prefix.identity")
        command = [
            sys.executable, str(recorder), "--upstream", identity["endpoint"],
            "--evidence", str(evidence_path), "--cell", str(cell_path),
            "--prefix-tokens", str(cell["prefix_tokens"]), "--prefix-identity", prefix_id,
            "--warmups", "1",
        ]
        if tokenizer:
            # Load the tokenizer for prefix hashing, but preserve vLLM's
            # native string request path. Qwen3.5 applies model-specific
            # prompt normalization when a list of ids is forwarded.
            command.extend(("--tokenizer", tokenizer))
        command.extend(("--", *["{proxy_url}" if arg == identity["endpoint"] else arg for arg in bench]))
        _write_json(attempt_dir / _COMMAND_NAME, {"argv": command, "variant": variant, "cell_id": cell["cell_id"]})
        output = run_command(
            command, result_path=result_path, stdout_path=stdout_path, stderr_path=stderr_path,
            timeout=timeout,
        )
        inputs.update({
            "stdout": _relative_artifact(store, stdout_path),
            "stderr": _relative_artifact(store, stderr_path),
            "sidecar": _relative_artifact(store, sidecar_path),
            "command": _relative_artifact(store, attempt_dir / _COMMAND_NAME),
        })
        if custom_workload_path.is_file():
            inputs["workload"] = _relative_artifact(store, custom_workload_path)
        if evidence_path.is_file():
            inputs["evidence"] = _relative_artifact(store, evidence_path)
        if output.returncode != 0 or not result_path.is_file() or not evidence_path.is_file():
            return _record_special_failure(
                store, cell, variant, attempt, campaign_id, attempt_dir,
                f"cached-prefix recorder exited with {output.returncode}",
                collector="cached-prefix-transparent-recorder", inputs=inputs, command=command,
            )
        converted = convert_cached_files(
            result_path, evidence_path, attempt_dir / _CONVERTED_NAME,
            cell=cell, variant=variant, require_hit=True,
        )
        converted["collector"] = "vllm-bench-prefix-repetition"
        _write_json(attempt_dir / _CONVERTED_NAME, converted)
        inputs["converted"] = _relative_artifact(store, attempt_dir / _CONVERTED_NAME)
        probe = {
            "kind": "cached-prefix-transparent-recorder", "status": "completed",
            "variant": variant, "returncode": output.returncode, "inputs": inputs,
            "cache_hit_required": True,
        }
        entry = store.record_variant_cell(cell, variant, converted, collector_argv=list(sys.argv), probe=probe)
        store.seal_variant_cell(cell["cell_id"], variant)
        entry = store.index["variant_cells"][f"{cell['cell_id']}:{variant}"]
        return _outcome(campaign_id, cell, variant, entry, attempt, probe)
    except Exception as exc:
        for path in (stdout_path, stderr_path):
            if not path.exists():
                _write_bytes(path, str(exc).encode("utf-8", errors="replace"))
        for path, key in ((sidecar_path, "sidecar"), (evidence_path, "evidence")):
            if path.is_file() and key not in inputs:
                inputs[key] = _relative_artifact(store, path)
        if (attempt_dir / _COMMAND_NAME).is_file():
            inputs["command"] = _relative_artifact(store, attempt_dir / _COMMAND_NAME)
        return _record_special_failure(
            store, cell, variant, attempt, campaign_id, attempt_dir, exc,
            collector="cached-prefix-transparent-recorder", inputs=inputs,
        )


def _run_mixed_cell(
    manifest: dict[str, Any],
    cell: dict[str, Any],
    variant: str,
    store: ArtifactStore,
    *,
    binary: Path,
    seed: int,
    tokenizer: str | None,
    timeout: float | None,
    dry_run: bool,
    campaign_id: str,
) -> dict[str, Any]:
    """Run role-specific mixed processes and require measured interval overlap."""

    existing = store.variant_status(cell["cell_id"], variant)
    shape = cell.get("mixed_shape")
    foreground = shape.get("foreground") if isinstance(shape, dict) else None
    background = shape.get("background") if isinstance(shape, dict) else None
    if not isinstance(foreground, dict) or not isinstance(background, dict):
        raise VllmBenchRunnerError("mixed cell requires mixed_shape.foreground/background")
    repeats = int(cell.get("repeats", 1))
    bg_requests = int(background["requests"])
    bg_context = int(background["context_tokens"])
    fg_input = int(foreground["input_tokens"])
    bg_output = int(background["output_tokens"])
    fg_output = int(foreground["output_tokens"])
    def prompt_text(token_count: int, ordinal: int) -> str:
        if not tokenizer:
            raise VllmBenchRunnerError("mixed role-specific workloads require a tokenizer path")
        try:
            from tokenizers import Tokenizer

            tokenizer_path = Path(tokenizer)
            if tokenizer_path.is_dir():
                tokenizer_path = tokenizer_path / "tokenizer.json"
            encoder = Tokenizer.from_file(str(tokenizer_path))
            suffix = f" sample{ordinal}"
            prefix_words = max(token_count - 2, 1)
            for _ in range(8):
                value = "a" + " a" * (prefix_words - 1) + suffix
                measured = len(encoder.encode(value, add_special_tokens=False).ids)
                if measured == token_count:
                    return value
                prefix_words = max(prefix_words + token_count - measured, 1)
            raise VllmBenchRunnerError(
                f"tokenizer could not construct exact mixed prompt length {token_count}"
            )
        except (OSError, ValueError) as exc:
            raise VllmBenchRunnerError(f"failed to load mixed tokenizer {tokenizer}: {exc}") from exc

    bg_prompts = [prompt_text(bg_context, index) for index in range(repeats * bg_requests)]
    fg_prompts = [prompt_text(fg_input, 10000 + index) for index in range(repeats)]
    rows, orchestrator_sidecar = build_mixed_workload(
        cell["cell_id"], background_prompts=bg_prompts, foreground_prompts=fg_prompts,
        background_output_tokens=bg_output, foreground_output_tokens=fg_output,
    )
    bg_count = len(bg_prompts)
    bg_rows, fg_rows = rows[:bg_count], rows[bg_count:]
    bg_cell = {
        **cell, "cell_id": f"{cell['cell_id']}:background", "name": f"{cell.get('name', cell['cell_id'])}:background",
        "input_tokens": bg_context, "output_tokens": bg_output, "concurrency": bg_requests,
        "warmup_repeats": 0, "repeats": repeats,
    }
    fg_cell = {
        **cell, "cell_id": f"{cell['cell_id']}:foreground", "name": f"{cell.get('name', cell['cell_id'])}:foreground",
        "input_tokens": fg_input, "output_tokens": fg_output, "concurrency": 1,
        "warmup_repeats": 0, "repeats": repeats,
    }
    bg_sidecar = build_sidecar(bg_cell, body_values=[row["prompt"] for row in bg_rows])
    fg_sidecar = build_sidecar(fg_cell, body_values=[row["prompt"] for row in fg_rows])
    if dry_run:
        result_dir = store.cell_dir / cell["cell_id"] / variant / "attempt-0001"
        bg_path, fg_path = result_dir / "mixed-background.jsonl", result_dir / "mixed-foreground.jsonl"
        return {
            "campaign_id": campaign_id, "cell_id": cell["cell_id"], "variant": variant,
            "status": "dry_run", "would_skip": existing == "sealed", "collector": "vllm-bench-mixed-orchestrator",
            "role_counts": {"background": len(bg_rows), "foreground": len(fg_rows)},
            "workloads": [str(bg_path), str(fg_path)],
        }
    store.attach_manifest(manifest)
    store.begin_variant_cell(cell, variant)
    attempt = _next_attempt(store, cell["cell_id"], variant)
    attempt_dir = store.cell_dir / cell["cell_id"] / variant / f"attempt-{attempt:04d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    bg_path, fg_path = attempt_dir / "mixed-background.jsonl", attempt_dir / "mixed-foreground.jsonl"
    bg_result = attempt_dir / "mixed-background.json"
    fg_result = attempt_dir / "mixed-foreground.json"
    bg_stdout, fg_stdout = attempt_dir / "background.stdout.log", attempt_dir / "foreground.stdout.log"
    bg_stderr, fg_stderr = attempt_dir / "background.stderr.log", attempt_dir / "foreground.stderr.log"
    overlap_path = attempt_dir / _OVERLAP_NAME
    combined_sidecar_path = attempt_dir / _SIDECAR_NAME
    try:
        write_custom_jsonl(bg_path, bg_rows)
        write_custom_jsonl(fg_path, fg_rows)
        write_sidecar(attempt_dir / "background-sidecar.json", bg_sidecar)
        write_sidecar(attempt_dir / "foreground-sidecar.json", fg_sidecar)
        _write_json(combined_sidecar_path, orchestrator_sidecar)
        identity = _variant_identity(manifest, variant)
        commands = two_process_commands(
            bg_cell, fg_cell, binary=binary, endpoint=identity["endpoint"], model=identity["model"],
            result_dir=attempt_dir, background_workload=bg_path, foreground_workload=fg_path, seed=seed,
            tokenizer=tokenizer,
        )
        _write_json(attempt_dir / _COMMAND_NAME, commands)
        starts: dict[str, float] = {}
        finishes: dict[str, float] = {}
        with bg_stdout.open("wb") as bg_out, bg_stderr.open("wb") as bg_err, fg_stdout.open("wb") as fg_out, fg_stderr.open("wb") as fg_err:
            starts["background_decode"] = time.time()
            bg_proc = subprocess.Popen(commands["background"], stdout=bg_out, stderr=bg_err)
            starts["foreground_prefill"] = time.time()
            fg_proc = subprocess.Popen(commands["foreground"], stdout=fg_out, stderr=fg_err)
            bg_code = bg_proc.wait(timeout=timeout)
            fg_code = fg_proc.wait(timeout=timeout)
            finishes["background_decode"] = time.time()
            finishes["foreground_prefill"] = time.time()
        overlap_evidence = {
            "clock_domain": "unix_wall_clock",
            "requests": [
                {"role": role, "clock_domain": "unix_wall_clock", "started_at_unix_s": starts[role], "finished_at_unix_s": finishes[role]}
                for role in ("background_decode", "foreground_prefill")
            ],
            "process_returncodes": {"background": bg_code, "foreground": fg_code},
        }
        overlap = validate_overlap(overlap_evidence)
        overlap_evidence["validation"] = overlap
        _write_json(overlap_path, overlap_evidence)
        if bg_code != 0 or fg_code != 0:
            raise VllmBenchRunnerError(f"mixed role exited with background={bg_code}, foreground={fg_code}")
        bg_converted = convert_vllm_bench(
            _read_json(bg_result), bg_sidecar, cell=bg_cell, variant=variant,
        )
        fg_converted = convert_vllm_bench(
            _read_json(fg_result), fg_sidecar, cell=fg_cell, variant=variant,
        )
        converted = {
            "schema_version": 1, "collector": "vllm-bench-mixed-orchestrator", "variant": variant,
            "cell": dict(cell), "status": "succeeded",
            "measurements": bg_converted["measurements"] + fg_converted["measurements"],
            "pairing": {"identity": "role-specific-canonical-sidecar", "ordering": "stable_ordinal"},
            "mixed_overlap": overlap, "mixed_overlap_evidence": overlap_evidence,
            "mixed_roles": {"background": bg_cell, "foreground": fg_cell},
        }
        converted_path = attempt_dir / _CONVERTED_NAME
        _write_json(converted_path, converted)
        inputs = {
            key: _relative_artifact(store, path)
            for key, path in {
                "stdout_background": bg_stdout, "stderr_background": bg_stderr,
                "stdout_foreground": fg_stdout, "stderr_foreground": fg_stderr,
                "background_result": bg_result, "foreground_result": fg_result,
                "background_sidecar": attempt_dir / "background-sidecar.json",
                "foreground_sidecar": attempt_dir / "foreground-sidecar.json",
                "sidecar": combined_sidecar_path, "overlap": overlap_path,
                "command": attempt_dir / _COMMAND_NAME, "converted": converted_path,
            }.items()
            if path.is_file()
        }
        probe = {"kind": "vllm-bench-mixed-orchestrator", "status": "completed", "variant": variant,
                 "inputs": inputs, "overlap": overlap}
        entry = store.record_variant_cell(cell, variant, converted, collector_argv=list(sys.argv), probe=probe)
        store.seal_variant_cell(cell["cell_id"], variant)
        entry = store.index["variant_cells"][f"{cell['cell_id']}:{variant}"]
        return _outcome(campaign_id, cell, variant, entry, attempt, probe)
    except Exception as exc:
        inputs = {
            key: _relative_artifact(store, path)
            for key, path in {
                "stdout_background": bg_stdout, "stderr_background": bg_stderr,
                "stdout_foreground": fg_stdout, "stderr_foreground": fg_stderr,
                "overlap": overlap_path, "sidecar": combined_sidecar_path,
            }.items() if path.is_file()
        }
        return _record_special_failure(
            store, cell, variant, attempt, campaign_id, attempt_dir, exc,
            collector="vllm-bench-mixed-orchestrator", inputs=inputs,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="planner-produced campaign manifest")
    parser.add_argument("--artifact-root", type=Path, required=True, help="external ArtifactStore directory")
    parser.add_argument("--cell", required=True, help="cell name or cell_id")
    parser.add_argument("--variant", choices=VARIANTS, default="target")
    parser.add_argument("--binary", type=Path, help="explicit vllm-bench executable")
    parser.add_argument("--tokenizer", help="local tokenizer path passed to vllm-bench")
    parser.add_argument("--workload-path", type=Path, help="custom JSONL path for mixed cells")
    parser.add_argument("--seed", type=int, help="override manifest client seed")
    parser.add_argument("--timeout", type=float, help="subprocess timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="render the command and sidecar without running it")
    parser.add_argument("--resume", dest="resume", action="store_true", help="skip an already sealed variant (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="fail instead of skipping a sealed variant")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        manifest = _read_json(args.manifest)
        store = ArtifactStore(args.artifact_root, _campaign_id(manifest))
        cell = _select_cell(manifest, args.cell)
        outcome = run_cell_variant(
            manifest,
            cell,
            args.variant,
            store,
            binary=args.binary,
            seed=args.seed,
            tokenizer=args.tokenizer,
            workload_path=args.workload_path,
            timeout=args.timeout,
            dry_run=args.dry_run,
            resume=args.resume,
            collector_argv=sys.argv,
        )
    except (
        OSError,
        ManifestError,
        ArtifactError,
        VllmBenchError,
        SidecarError,
        CachedPrefixError,
        MixedError,
        VllmBenchCollectError,
        VllmBenchRunnerError,
        ValueError,
    ) as exc:
        print(f"vllm-bench runner error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outcome, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if outcome.get("status") in {"sealed", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
