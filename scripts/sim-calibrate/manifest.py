#!/usr/bin/env python3
"""Versioned campaign manifest and workload-cell planning primitives.

The manifest is the identity boundary for a calibration campaign.  Its digest
is derived from target identity, client invocation, environment, workload
cells, split assignments, and provenance, but not from timestamps or the
digest field itself.  This keeps retries and dry-runs reproducible while still
retaining collection time as metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable

from vllm_bench_matrix import MatrixError, axis_values, cold_prefill_matrix, decode_matrix


SCHEMA_VERSION = 1
PAIRED_SCHEMA_VERSION = 2
CACHED_PAIRED_SCHEMA_VERSION = 3
PAIRED_VARIANTS = ("target", "zero_cost", "replay")
SPLITS = ("train", "threshold_validation", "held_out_mixed", "sealed_test")
WORKLOAD_KINDS = ("cold_prefill", "steady_decode", "cached_prefill", "mixed")
MIXED_SPLITS = ("threshold_validation", "held_out_mixed", "sealed_test")

_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|credential|password|passwd|prompt|secret|token[_-]?ids?|authorization)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE = re.compile(
    r"(?:bearer\s+\S+|sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{12,})",
    re.IGNORECASE,
)
_PRIVATE_PATH = re.compile(
    r"(?<![A-Za-z0-9_~.-])/(?:home|root|tmp|mnt|workspace)(?:/[^\s\"'<>`]+)+"
)
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


class ManifestError(ValueError):
    """Raised when a campaign spec cannot be made immutable and safe."""


def canonical_json(value: Any) -> str:
    """Render JSON deterministically for IDs and digest comparisons."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_matrix_spec(value: Any, field: str = "workload.matrices") -> Any:
    """Normalize range objects so equivalent JSON spellings share an identity."""

    axis_fields = {
        "input_tokens",
        "required_input_tokens",
        "concurrency",
        "context_tokens",
        "surface_requests",
        "surface_kv_tokens",
        "prefix_tokens",
        "new_tokens",
    }
    if isinstance(value, list) and field.rsplit(".", 1)[-1] in axis_fields:
        try:
            allow_empty = field.endswith(".surface_requests") or field.endswith(".surface_kv_tokens")
            return list(axis_values(value, field, allow_empty=allow_empty))
        except MatrixError as exc:
            raise ManifestError(str(exc)) from exc
    if isinstance(value, dict):
        keys = set(value)
        if keys == {"values"} or keys == {"start", "stop", "step"}:
            try:
                allow_empty = field.endswith(".surface_requests") or field.endswith(".surface_kv_tokens")
                return list(axis_values(value, field, allow_empty=allow_empty))
            except MatrixError as exc:
                raise ManifestError(str(exc)) from exc
        return {
            key: _canonical_matrix_spec(value[key], f"{field}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, list):
        return [_canonical_matrix_spec(item, f"{field}[{index}]") for index, item in enumerate(value)]
    return value


def canonical_scheduler_identity(scheduler: dict[str, Any]) -> dict[str, Any]:
    """Normalize the scheduler fields shared by campaign and profile contracts."""

    if not isinstance(scheduler, dict):
        raise ManifestError("runtime.scheduler must be an object")
    policy = _non_empty(scheduler.get("policy", "vllm_v1"), "runtime.scheduler.policy")
    max_num_seqs = _positive_int(scheduler.get("max_num_seqs"), "runtime.scheduler.max_num_seqs")
    max_num_batched_tokens = _positive_int(
        scheduler.get("max_num_batched_tokens"), "runtime.scheduler.max_num_batched_tokens"
    )
    max_model_len = _positive_int(scheduler.get("max_model_len"), "runtime.scheduler.max_model_len")
    prefill = scheduler.get("prefill")
    if prefill is None:
        prefill = {
            "mode": scheduler.get("prefill_mode", "whole"),
            "max_chunk_tokens": scheduler.get("max_chunk_tokens"),
        }
    if not isinstance(prefill, dict):
        raise ManifestError("runtime.scheduler.prefill must be an object")
    mode = _non_empty(prefill.get("mode", "whole"), "runtime.scheduler.prefill.mode")
    if mode == "whole":
        if prefill.get("max_chunk_tokens") is not None:
            raise ManifestError("whole prefill cannot define max_chunk_tokens")
        normalized_prefill = {"mode": "whole"}
    elif mode == "chunked":
        max_chunk_tokens = _positive_int(
            prefill.get("max_chunk_tokens"), "runtime.scheduler.prefill.max_chunk_tokens"
        )
        if max_chunk_tokens > max_num_batched_tokens:
            raise ManifestError("runtime.scheduler.prefill.max_chunk_tokens exceeds max_num_batched_tokens")
        normalized_prefill = {"mode": "chunked", "max_chunk_tokens": max_chunk_tokens}
    else:
        raise ManifestError("runtime.scheduler.prefill.mode must be whole or chunked")
    return {
        "policy": policy,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_model_len": max_model_len,
        "prefill": normalized_prefill,
    }


def canonical_cache_identity(cache: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize cache fields shared by campaign and profile contracts."""

    value = cache if isinstance(cache, dict) else {}
    policy = str(value.get("policy") or "unknown")
    enabled = policy == "on"
    block_size = value.get("block_size", value.get("block_alignment", 0)) if enabled else 0
    alignment = value.get("alignment", value.get("block_alignment", 0)) if enabled else 0
    for field, item in (("block_size", block_size), ("alignment", alignment)):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ManifestError(f"runtime.cache.{field} must be a non-negative integer")
    hit_semantics = value.get(
        "hit_semantics",
        "positive_shared_prefix_identity_and_uncached_tail" if enabled else "disabled",
    )
    hit_semantics = _non_empty(hit_semantics, "runtime.cache.hit_semantics")
    return {
        "enabled": enabled,
        "policy": policy,
        "block_size": block_size,
        "alignment": alignment,
        "hit_semantics": hit_semantics,
    }


def canonical_client_identity(client: dict[str, Any], *, tokenizer_config_sha256: str | None = None) -> dict[str, Any]:
    """Normalize the client fields represented by the profile contract."""

    if not isinstance(client, dict):
        raise ManifestError("client must be an object")
    name = _non_empty(client.get("name"), "client.name")
    version = _non_empty(client.get("version"), "client.version")
    seed = client.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ManifestError("client.seed must be a non-negative integer when present")
    sampling = client.get("sampling", {"temperature": 0.0, "top_p": 1.0})
    if not isinstance(sampling, dict):
        raise ManifestError("client.sampling must be an object")
    sampling_text = (
        "temperature=0,top_p=1"
        if sampling.get("temperature", 0.0) == 0.0 and sampling.get("top_p", 1.0) == 1.0 and len(sampling) <= 2
        else canonical_json(sampling)
    )
    identity = {"name": name, "version": version, "seed": seed, "sampling": sampling_text}
    if tokenizer_config_sha256 is not None:
        if not _SHA256.fullmatch(tokenizer_config_sha256):
            raise ManifestError("client.tokenizer_config_sha256 must be SHA-256")
        identity["tokenizer_config_sha256"] = tokenizer_config_sha256
    return identity


def canonical_workload_identity(campaign: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Normalize workload cells and split assignments for profile identity."""

    workload = campaign.get("workload") if isinstance(campaign, dict) else None
    cells = workload.get("cells") if isinstance(workload, dict) else None
    if not isinstance(cells, list) or not cells:
        raise ManifestError("workload.cells must be a non-empty list")
    normalized_cells: list[dict[str, Any]] = []
    warmups: set[int] = set()
    repeats: set[int] = set()
    names: set[str] = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise ManifestError(f"workload.cells[{index}] must be an object")
        name = _non_empty(cell.get("name"), f"workload.cells[{index}].name")
        if name in names:
            raise ManifestError(f"workload cell name is duplicated: {name}")
        names.add(name)
        kind = _non_empty(cell.get("kind"), f"workload.cells[{index}].kind")
        if kind not in WORKLOAD_KINDS:
            raise ManifestError(f"workload.cells[{index}].kind is unsupported: {kind}")
        normalized_cell = {
            "id": name,
            "family": kind,
            "input_tokens": _positive_int(cell.get("input_tokens"), f"workload.cells[{index}].input_tokens"),
            "output_tokens": _positive_int(cell.get("output_tokens"), f"workload.cells[{index}].output_tokens"),
            "concurrency": _positive_int(cell.get("concurrency"), f"workload.cells[{index}].concurrency"),
            "arrival_pattern": _non_empty(
                cell.get("arrival_pattern", "closed_loop"),
                f"workload.cells[{index}].arrival_pattern",
            ),
            "attribution": "black_box_effective",
        }
        normalized_cells.append(normalized_cell)
        warmups.add(_positive_int(cell.get("warmup_repeats", 0), f"workload.cells[{index}].warmup_repeats", allow_zero=True))
        repeats.add(_positive_int(cell.get("repeats", 1), f"workload.cells[{index}].repeats"))
    if not warmups or not repeats:
        raise ManifestError("profile workload identity requires workload repetition metadata")
    # Repetition counts describe measurement effort, not the serving workload
    # shape.  Campaigns may use more repeats for held-out cells than for train
    # cells; retain one deterministic compatibility scalar for the Rust profile
    # schema while the concrete cell counts remain in the campaign artifacts.
    raw_splits = campaign.get("splits")
    if raw_splits is None:
        split = {name: [] for name in SPLITS}
        for cell in cells:
            split[cell.get("split", "train")].append(cell["name"])
    elif isinstance(raw_splits, dict):
        unknown_splits = set(raw_splits) - set(SPLITS)
        if unknown_splits:
            raise ManifestError(f"splits contains unsupported names: {sorted(unknown_splits)}")
        split = {}
        for name in SPLITS:
            values = raw_splits.get(name, [])
            if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
                raise ManifestError(f"splits.{name} must be a list of strings")
            split[name] = sorted(values)
    else:
        raise ManifestError("splits must be an object when present")
    if set(name for values in split.values() for name in values) != names:
        raise ManifestError("workload split identity does not cover exactly the workload cells")
    return (
        {
            "cells": sorted(normalized_cells, key=lambda item: item["id"]),
            "warmup_repeats": max(warmups),
            "repeats": max(repeats),
        },
        split,
    )


def _redact_string(value: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    home = Path.home()
    redacted = value.replace(str(repo_root), "<repo>")
    if str(home) != "/":
        redacted = redacted.replace(str(home), "<home>")
    return _PRIVATE_PATH.sub("<redacted>", redacted)


def _validate_safe(value: Any, path: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _SENSITIVE_KEY.search(str(key)):
                raise ManifestError(f"sensitive field is not allowed at {path}.{key}")
            _validate_safe(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_safe(item, f"{path}[{index}]")
        return
    if isinstance(value, str) and _SENSITIVE_VALUE.search(value):
        raise ManifestError(f"credential-like value is not allowed at {path}")


def _validate_argv(argv: Iterable[str], field: str) -> list[str]:
    rendered = []
    for index, arg in enumerate(argv):
        if not isinstance(arg, str) or not arg:
            raise ManifestError(f"{field}[{index}] must be a non-empty string")
        if re.search(r"(?:^|[-_])(prompt|token[-_]?ids?|api[-_]?key|password|secret)(?:=|$)", arg, re.I):
            raise ManifestError(f"sensitive argument is not allowed at {field}[{index}]")
        if _SENSITIVE_VALUE.search(arg):
            raise ManifestError(f"credential-like argument is not allowed at {field}[{index}]")
        rendered.append(_redact_string(arg))
    return rendered


def _non_empty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{field} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ManifestError(f"{field} must be {'non-negative' if allow_zero else 'positive'}")
    return value


def _normalise_cell(raw: dict[str, Any], index: int, defaults: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ManifestError(f"workload.cells[{index}] must be an object")
    cell = {**defaults, **raw}
    kind = _non_empty(cell.get("kind"), f"workload.cells[{index}].kind")
    if kind not in WORKLOAD_KINDS:
        raise ManifestError(f"workload.cells[{index}].kind must be one of {WORKLOAD_KINDS}")
    split = _non_empty(cell.get("split", "train"), f"workload.cells[{index}].split")
    if split not in SPLITS:
        raise ManifestError(f"workload.cells[{index}].split must be one of {SPLITS}")
    name = _non_empty(cell.get("name", f"{kind}-{index:03d}"), f"workload.cells[{index}].name")
    input_tokens = _positive_int(cell.get("input_tokens"), f"workload.cells[{index}].input_tokens")
    output_tokens = _positive_int(cell.get("output_tokens"), f"workload.cells[{index}].output_tokens")
    concurrency = _positive_int(cell.get("concurrency", 1), f"workload.cells[{index}].concurrency")
    sum_decode_ctx_tokens = cell.get("sum_decode_ctx_tokens")
    if sum_decode_ctx_tokens is not None:
        sum_decode_ctx_tokens = _positive_int(
            sum_decode_ctx_tokens,
            f"workload.cells[{index}].sum_decode_ctx_tokens",
        )
    warmup = _positive_int(cell.get("warmup_repeats", 0), f"workload.cells[{index}].warmup_repeats", allow_zero=True)
    repeats = _positive_int(cell.get("repeats", 1), f"workload.cells[{index}].repeats")
    request_rate = cell.get("request_rate")
    if request_rate is not None and (isinstance(request_rate, bool) or not isinstance(request_rate, (int, float)) or request_rate <= 0):
        raise ManifestError(f"workload.cells[{index}].request_rate must be positive when present")
    arrival = _non_empty(cell.get("arrival_pattern", "closed_loop"), f"workload.cells[{index}].arrival_pattern")
    prefix_tokens = _positive_int(cell.get("prefix_tokens", 0), f"workload.cells[{index}].prefix_tokens", allow_zero=True)
    if kind == "cached_prefill" and prefix_tokens == 0:
        raise ManifestError(f"workload.cells[{index}] cached_prefill requires prefix_tokens > 0")
    shared_prefix = cell.get("shared_prefix")
    if kind == "cached_prefill":
        if not isinstance(shared_prefix, dict):
            raise ManifestError(f"workload.cells[{index}].cached_prefill requires shared_prefix")
        identity = _non_empty(shared_prefix.get("identity"), f"workload.cells[{index}].shared_prefix.identity")
        if prefix_tokens >= input_tokens:
            raise ManifestError(f"workload.cells[{index}] cached_prefill requires uncached suffix tokens")
        suffix_mode = _non_empty(
            shared_prefix.get("suffix_mode", "request_index"),
            f"workload.cells[{index}].shared_prefix.suffix_mode",
        )
        shared_prefix = {
            "identity": identity,
            "cached_tokens": prefix_tokens,
            "suffix_mode": suffix_mode,
        }
    elif shared_prefix is not None:
        raise ManifestError(f"workload.cells[{index}].shared_prefix is only valid for cached_prefill")
    if kind == "mixed" and split not in MIXED_SPLITS:
        raise ManifestError(f"mixed cell {name!r} must use one of {MIXED_SPLITS}")
    prefill_input_tokens = cell.get("prefill_input_tokens")
    if prefill_input_tokens is not None:
        prefill_input_tokens = _positive_int(
            prefill_input_tokens, f"workload.cells[{index}].prefill_input_tokens"
        )
    mixed_shape = cell.get("mixed_shape")
    if kind == "mixed":
        if not isinstance(mixed_shape, dict):
            raise ManifestError(f"workload.cells[{index}] mixed requires mixed_shape")
        foreground = mixed_shape.get("foreground")
        background = mixed_shape.get("background")
        if not isinstance(foreground, dict) or not isinstance(background, dict):
            raise ManifestError(f"workload.cells[{index}].mixed_shape requires foreground/background")
        for field in ("input_tokens", "output_tokens"):
            _positive_int(foreground.get(field), f"workload.cells[{index}].mixed_shape.foreground.{field}")
        for field in ("requests", "context_tokens", "output_tokens"):
            _positive_int(background.get(field), f"workload.cells[{index}].mixed_shape.background.{field}")
        if prefill_input_tokens is None:
            prefill_input_tokens = _positive_int(
                foreground["input_tokens"], f"workload.cells[{index}].prefill_input_tokens"
            )
        if sum_decode_ctx_tokens is None:
            sum_decode_ctx_tokens = background["requests"] * background["context_tokens"]
    elif mixed_shape is not None:
        raise ManifestError(f"workload.cells[{index}].mixed_shape is only valid for mixed")
    if kind == "steady_decode" and input_tokens <= 0:
        raise ManifestError(f"workload.cells[{index}] steady_decode requires input_tokens > 0")
    sampling = cell.get("sampling", {"temperature": 0.0, "top_p": 1.0})
    if not isinstance(sampling, dict):
        raise ManifestError(f"workload.cells[{index}].sampling must be an object")
    for field in ("temperature", "top_p"):
        if field in sampling and (
            isinstance(sampling[field], bool) or not isinstance(sampling[field], (int, float))
        ):
            raise ManifestError(f"workload.cells[{index}].sampling.{field} must be numeric")
    result = {
        "name": name,
        "kind": kind,
        "split": split,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "concurrency": concurrency,
        "arrival_pattern": arrival,
        "request_rate": request_rate,
        "prefix_tokens": prefix_tokens,
        "warmup_repeats": warmup,
        "repeats": repeats,
        "ignore_eos": bool(cell.get("ignore_eos", True)),
        "sampling": sampling,
        "cache_evidence_required": kind == "cached_prefill",
        "fit_policy": {
            "status": "held_out" if kind == "mixed" or split == "held_out_mixed" else "measured",
            "excluded_metrics": ["ttft_ms"] if concurrency > 1 else [],
        },
    }
    if sum_decode_ctx_tokens is not None:
        result["sum_decode_ctx_tokens"] = sum_decode_ctx_tokens
    elif kind == "steady_decode":
        result["sum_decode_ctx_tokens"] = input_tokens * concurrency
    if shared_prefix is not None:
        result["shared_prefix"] = shared_prefix
    if prefill_input_tokens is not None:
        result["prefill_input_tokens"] = prefill_input_tokens
    if mixed_shape is not None:
        result["mixed_shape"] = mixed_shape
    if "metadata" in cell:
        result["metadata"] = cell["metadata"]
    return result


def _axis_values(value: Any, field: str, *, allow_empty: bool = False) -> list[int]:
    try:
        return list(axis_values(value, field, allow_empty=allow_empty))
    except MatrixError as exc:
        raise ManifestError(str(exc)) from exc


def _expand_cached_prefill(
    workload: dict[str, Any],
    start_index: int,
    *,
    raw: Any = None,
    require_explicit_scalars: bool = False,
) -> list[dict[str, Any]]:
    raw = workload.get("cached_prefill") if raw is None else raw
    matrices = raw if isinstance(raw, list) else [raw]
    if not matrices or any(not isinstance(matrix, dict) for matrix in matrices):
        raise ManifestError("workload.cached_prefill must be an object or a list of objects")

    expanded: list[dict[str, Any]] = []
    for matrix_index, matrix in enumerate(matrices):
        defaults = matrix.get("defaults", {})
        if not isinstance(defaults, dict):
            raise ManifestError(f"workload.cached_prefill[{matrix_index}].defaults must be an object")
        split = _non_empty(
            matrix.get("split", defaults.get("split", "train")),
            f"workload.cached_prefill[{matrix_index}].split",
        )
        if split not in SPLITS:
            raise ManifestError(f"workload.cached_prefill[{matrix_index}].split must be one of {SPLITS}")
        prefix_axis = _axis_values(
            matrix.get("prefix_tokens"),
            f"workload.cached_prefill[{matrix_index}].prefix_tokens",
        )
        new_axis = _axis_values(
            matrix.get("new_tokens"),
            f"workload.cached_prefill[{matrix_index}].new_tokens",
        )
        shared_prefix = matrix.get("shared_prefix", defaults.get("shared_prefix"))
        if not isinstance(shared_prefix, dict):
            raise ManifestError(f"workload.cached_prefill[{matrix_index}].shared_prefix must be an object")
        identity = _non_empty(
            shared_prefix.get("identity"),
            f"workload.cached_prefill[{matrix_index}].shared_prefix.identity",
        )
        suffix_mode = _non_empty(
            shared_prefix.get("suffix_mode", "request_index"),
            f"workload.cached_prefill[{matrix_index}].shared_prefix.suffix_mode",
        )
        output_value = matrix.get("output_tokens", defaults.get("output_tokens"))
        if output_value is None and not require_explicit_scalars:
            output_value = 4
        output_tokens = _positive_int(
            output_value,
            f"workload.cached_prefill[{matrix_index}].output_tokens",
        )
        name_prefix = _non_empty(
            matrix.get("name_prefix", f"cached-prefill-{split}"),
            f"workload.cached_prefill[{matrix_index}].name_prefix",
        )
        metadata = matrix.get("metadata", defaults.get("metadata", {}))
        if not isinstance(metadata, dict):
            raise ManifestError(f"workload.cached_prefill[{matrix_index}].metadata must be an object")

        cell_defaults = dict(defaults)
        repeats_value = matrix.get("repeats", defaults.get("repeats"))
        warmup_value = matrix.get("warmup_repeats", defaults.get("warmup_repeats"))
        if repeats_value is None and not require_explicit_scalars:
            repeats_value = 3
        if warmup_value is None and not require_explicit_scalars:
            warmup_value = 1
        cell_defaults.update(
            {
                "kind": "cached_prefill",
                "split": split,
                "output_tokens": output_tokens,
                "concurrency": matrix.get("concurrency", defaults.get("concurrency", 1)),
                "warmup_repeats": warmup_value,
                "repeats": repeats_value,
                "ignore_eos": matrix.get("ignore_eos", defaults.get("ignore_eos", True)),
                "arrival_pattern": matrix.get("arrival_pattern", defaults.get("arrival_pattern", "closed_loop")),
                "request_rate": matrix.get("request_rate", defaults.get("request_rate")),
                "sampling": matrix.get("sampling", defaults.get("sampling", {"temperature": 0.0, "top_p": 1.0})),
            }
        )
        for prefix_tokens, new_tokens in product(prefix_axis, new_axis):
            cell_metadata = dict(metadata)
            cell_metadata["cached_prefill_matrix"] = {
                "prefix_tokens": prefix_tokens,
                "new_tokens": new_tokens,
                "shared_prefix_identity": identity,
            }
            expanded.append(
                _normalise_cell(
                    {
                        "name": f"{name_prefix}-prefix{prefix_tokens}-new{new_tokens}",
                        "input_tokens": prefix_tokens + new_tokens,
                        "prefix_tokens": prefix_tokens,
                        "shared_prefix": {
                            "identity": identity,
                            "suffix_mode": suffix_mode,
                        },
                        "metadata": cell_metadata,
                    },
                    start_index + len(expanded),
                    cell_defaults,
                )
            )
    return expanded


def _expand_matrices(workload: dict[str, Any], start_index: int = 0) -> list[dict[str, Any]]:
    """Expand the explicit JSON matrix section into concrete workload cells."""

    matrices = workload.get("matrices")
    if not isinstance(matrices, dict) or not matrices:
        raise ManifestError("workload.matrices must be a non-empty object")
    supported = {"cold_prefill", "decode", "steady_decode", "cached_prefill", "mixed"}
    unknown = set(matrices) - supported
    if unknown:
        raise ManifestError(f"workload.matrices contains unsupported kinds: {sorted(unknown)}")

    expanded: list[dict[str, Any]] = []

    def append_generated(raw_cells: list[dict[str, Any]], field: str) -> None:
        for raw_cell in raw_cells:
            try:
                expanded.append(_normalise_cell(raw_cell, start_index + len(expanded), {}))
            except ManifestError as exc:
                raise ManifestError(f"invalid {field} matrix cell: {exc}") from exc

    if "cold_prefill" in matrices:
        try:
            append_generated(cold_prefill_matrix(matrices["cold_prefill"]), "cold_prefill")
        except MatrixError as exc:
            raise ManifestError(str(exc)) from exc

    if "decode" in matrices and "steady_decode" in matrices:
        raise ManifestError("workload.matrices cannot define both decode and steady_decode")
    decode_key = "steady_decode" if "steady_decode" in matrices else "decode" if "decode" in matrices else None
    if decode_key is not None:
        try:
            append_generated(decode_matrix(matrices[decode_key]), decode_key)
        except MatrixError as exc:
            raise ManifestError(str(exc)) from exc

    if "cached_prefill" in matrices:
        append_generated(
            _expand_cached_prefill(
                {"cached_prefill": matrices["cached_prefill"]},
                start_index + len(expanded),
                require_explicit_scalars=True,
            ),
            "cached_prefill",
        )

    if "mixed" in matrices:
        raw_mixed = matrices["mixed"]
        if isinstance(raw_mixed, dict):
            defaults = raw_mixed.get("defaults", {})
            raw_cells = raw_mixed.get("cells")
            if not isinstance(defaults, dict):
                raise ManifestError("workload.matrices.mixed.defaults must be an object")
        elif isinstance(raw_mixed, list):
            defaults = {}
            raw_cells = raw_mixed
        else:
            raise ManifestError("workload.matrices.mixed must be an object or a list")
        if not isinstance(raw_cells, list) or not raw_cells:
            raise ManifestError("workload.matrices.mixed.cells must be a non-empty list")
        mixed_cells: list[dict[str, Any]] = []
        for index, raw_cell in enumerate(raw_cells):
            if not isinstance(raw_cell, dict):
                raise ManifestError(f"workload.matrices.mixed.cells[{index}] must be an object")
            cell = {**defaults, **raw_cell}
            if cell.get("kind") != "mixed":
                raise ManifestError(f"workload.matrices.mixed.cells[{index}].kind must be mixed")
            mixed_cells.append(cell)
        append_generated(mixed_cells, "mixed")

    if not expanded:
        raise ManifestError("workload.matrices must define at least one matrix")
    return expanded


def _expand_cells(workload: dict[str, Any]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    if "matrices" in workload:
        extra = set(workload) - {"matrices"}
        if extra:
            raise ManifestError(
                f"workload.matrices cannot be combined with other workload keys: {sorted(extra)}"
            )
        expanded.extend(_expand_matrices(workload))
    elif "cells" in workload:
        cells = workload["cells"]
        if not isinstance(cells, list) or not cells:
            raise ManifestError("workload.cells must be a non-empty list")
        expanded.extend(_normalise_cell(cell, index, {}) for index, cell in enumerate(cells))
    elif "dimensions" in workload:
        dimensions = workload.get("dimensions")
        if not isinstance(dimensions, dict) or not dimensions:
            raise ManifestError("workload.dimensions must be a non-empty object")
        defaults = workload.get("defaults", {})
        if not isinstance(defaults, dict):
            raise ManifestError("workload.defaults must be an object")
        names = sorted(dimensions)
        values: list[list[Any]] = []
        for name in names:
            options = dimensions[name]
            if not isinstance(options, list) or not options:
                raise ManifestError(f"workload.dimensions.{name} must be a non-empty list")
            values.append(options)
        for index, combination in enumerate(product(*values)):
            cell = dict(zip(names, combination))
            expanded.append(_normalise_cell(cell, index, defaults))
    elif "cached_prefill" not in workload:
        raise ManifestError("workload requires cells, dimensions, or cached_prefill")
    if "cached_prefill" in workload:
        expanded.extend(_expand_cached_prefill(workload, len(expanded)))
    if not expanded:
        raise ManifestError("workload must expand to at least one cell")
    return expanded


def _environment() -> dict[str, Any]:
    """Capture stable, non-secret environment facts for provenance."""

    commit = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_revision": commit,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "proxy_mode": "ambient_env_present"
        if any(os.environ.get(name) for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"))
        else "ambient_env_absent",
    }


def _split_assignments(cells: list[dict[str, Any]], raw: Any) -> dict[str, list[str]]:
    by_name = {cell["name"]: cell for cell in cells}
    assignments: dict[str, list[str]] = {split: [] for split in SPLITS}
    if raw is None:
        for cell in cells:
            assignments[cell["split"]].append(cell["name"])
        return assignments
    if not isinstance(raw, dict):
        raise ManifestError("splits must be an object")
    seen: set[str] = set()
    for split in SPLITS:
        names = raw.get(split, [])
        if not isinstance(names, list):
            raise ManifestError(f"splits.{split} must be a list")
        for name in names:
            if not isinstance(name, str) or name not in by_name:
                raise ManifestError(f"splits.{split} references unknown cell {name!r}")
            if name in seen:
                raise ManifestError(f"cell {name!r} appears in multiple splits")
            seen.add(name)
            assignments[split].append(name)
    missing = set(by_name) - seen
    if missing:
        raise ManifestError(f"cells missing split assignment: {sorted(missing)}")
    for split, names in assignments.items():
        for name in names:
            if by_name[name]["split"] != split:
                raise ManifestError(f"cell {name!r} split disagrees with splits.{split}")
    return assignments


def _paired_variants(spec: dict[str, Any], target: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    supplied = spec["variants"]
    if not isinstance(supplied, dict) or set(supplied) != {"zero_cost", "replay"}:
        raise ManifestError("variants must contain exactly zero_cost and replay; target comes from target/runtime")
    variants: dict[str, Any] = {"target": {"identity": target, "runtime": runtime}}
    for name in ("zero_cost", "replay"):
        value = supplied[name]
        if not isinstance(value, dict) or not isinstance(value.get("identity"), dict):
            raise ManifestError(f"variants.{name}.identity must be an object")
        identity = dict(value["identity"])
        for field in ("endpoint", "model", "engine", "engine_version", "model_revision", "model_config_sha256"):
            _non_empty(identity.get(field), f"variants.{name}.identity.{field}")
        identity["endpoint"] = identity["endpoint"].rstrip("/")
        if identity["model_config_sha256"].lower() != target["model_config_sha256"].lower():
            raise ManifestError(f"variants.{name} model config digest differs from target")
        variant_runtime = value.get("runtime")
        if not isinstance(variant_runtime, dict) or not variant_runtime:
            raise ManifestError(f"variants.{name}.runtime must be a non-empty object")
        server_flags = _validate_argv(value.get("server_flags", []), f"variants.{name}.server_flags")
        if not server_flags:
            raise ManifestError(f"variants.{name}.server_flags must be complete and non-empty")
        _non_empty(value.get("profile_sha256"), f"variants.{name}.profile_sha256")
        if not _SHA256.fullmatch(value["profile_sha256"]):
            raise ManifestError(f"variants.{name}.profile_sha256 must be SHA-256")
        variants[name] = {
            "identity": identity,
            "runtime": variant_runtime,
            "server_flags": server_flags,
            "profile_sha256": value["profile_sha256"],
        }
    variants["target"]["server_flags"] = spec["client"]["server_flags"]
    return variants


def build_manifest(spec: dict[str, Any], *, created_at: str | None = None) -> dict[str, Any]:
    """Validate a campaign spec and return its immutable manifest document."""

    if not isinstance(spec, dict):
        raise ManifestError("campaign spec must be an object")
    _validate_safe(spec)
    target = spec.get("target")
    runtime = spec.get("runtime", {})
    client = spec.get("client")
    provenance = spec.get("provenance", {})
    if not isinstance(target, dict):
        raise ManifestError("target must be an object")
    if not isinstance(runtime, dict) or not isinstance(client, dict) or not isinstance(provenance, dict):
        raise ManifestError("runtime, client, and provenance must be objects")
    target = dict(target)
    for field in ("endpoint", "model", "engine", "engine_version", "model_revision", "model_config_sha256"):
        _non_empty(target.get(field), f"target.{field}")
    target["endpoint"] = _non_empty(target["endpoint"], "target.endpoint").rstrip("/")
    client = dict(client)
    for field in ("name", "version", "revision"):
        _non_empty(client.get(field), f"client.{field}")
    client["argv"] = _validate_argv(client.get("argv", []), "client.argv")
    client["server_flags"] = _validate_argv(client.get("server_flags", []), "client.server_flags")
    cells = _expand_cells(spec.get("workload", {}))
    names = [cell["name"] for cell in cells]
    if len(names) != len(set(names)):
        raise ManifestError("workload cell names must be unique")
    splits = _split_assignments(cells, spec.get("splits"))
    paired = "variants" in spec
    if paired:
        if not _SHA256.fullmatch(target["model_config_sha256"]):
            raise ManifestError("paired target.model_config_sha256 must be SHA-256")
        scheduler = runtime.get("scheduler")
        if not isinstance(scheduler, dict):
            raise ManifestError("paired runtime.scheduler must be an object")
        max_model_len = _positive_int(scheduler.get("max_model_len"), "runtime.scheduler.max_model_len")
        if not client["argv"] or not client["server_flags"]:
            raise ManifestError("paired campaign requires complete client.argv and server_flags")
        if isinstance(client.get("seed"), bool) or not isinstance(client.get("seed"), int):
            raise ManifestError("paired campaign requires an integer client.seed")
        if any(cell["kind"] == "cached_prefill" for cell in cells):
            cache = runtime.get("cache")
            if not isinstance(cache, dict) or cache.get("policy") != "on":
                raise ManifestError("cached_prefill paired campaigns require runtime.cache.policy=on")
        cache = runtime.get("cache")
        if isinstance(cache, dict) and "endpoint_isolated" in cache and not isinstance(cache["endpoint_isolated"], bool):
            raise ManifestError("runtime.cache.endpoint_isolated must be boolean when present")
        for cell in cells:
            if cell["repeats"] < 2:
                raise ManifestError(f"paired cell {cell['name']!r} requires at least two repeats")
            if cell["kind"] == "cold_prefill" and cell["concurrency"] != 1:
                raise ManifestError(f"cold prefill cell {cell['name']!r} requires concurrency=1")
            if cell["kind"] == "cached_prefill" and cell["concurrency"] != 1:
                raise ManifestError(f"cached prefill cell {cell['name']!r} requires concurrency=1")
            if cell["kind"] == "steady_decode" and cell["output_tokens"] < 2:
                raise ManifestError(f"steady decode cell {cell['name']!r} requires at least two output tokens")
            if cell["input_tokens"] + cell["output_tokens"] > max_model_len:
                raise ManifestError(f"cell {cell['name']!r} exceeds target scheduler.max_model_len")
    for field in ("collector_revision", "collection_command", "artifact_uri", "source_digest"):
        _non_empty(provenance.get(field), f"provenance.{field}")
    environment = spec.get("environment", _environment())
    if not isinstance(environment, dict):
        raise ManifestError("environment must be an object")
    environment = {key: _redact_string(value) if isinstance(value, str) else value for key, value in environment.items()}
    workload_document: dict[str, Any] = {
        "cells": cells,
        "cell_count": len(cells),
    }
    if "matrices" in spec.get("workload", {}):
        workload_document["matrix_spec"] = _canonical_matrix_spec(spec["workload"]["matrices"])
    base = {
        "schema_version": (
            CACHED_PAIRED_SCHEMA_VERSION
            if paired and any(cell["kind"] == "cached_prefill" for cell in cells)
            else PAIRED_SCHEMA_VERSION
            if paired
            else SCHEMA_VERSION
        ),
        "target": target,
        "runtime": runtime,
        "client": client,
        "workload": workload_document,
        "splits": splits,
        "provenance": provenance,
        "environment": environment,
    }
    if paired:
        base["variants"] = _paired_variants(spec, target, runtime)
        base["pairing"] = {
            "variant_order": list(PAIRED_VARIANTS),
            "request_key": ["cell_id", "phase", "repeat", "index"],
            "percentile_subtraction": "prohibited",
        }
    campaign_id = sha256_json(base)
    stamped_cells = []
    for cell in cells:
        cell_id = sha256_json({"campaign_id": campaign_id, "cell": cell})
        stamped_cells.append({"cell_id": cell_id, **cell})
    base["workload"]["cells"] = stamped_cells
    base["campaign_id"] = campaign_id
    base["created_at_utc"] = created_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return base


def load_spec(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"failed to read spec {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ManifestError("campaign spec must be a JSON object")
    return document


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a manifest, allowing only an identical existing document."""

    rendered = json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ManifestError(f"failed to read existing manifest {path}: {exc}") from exc
        if existing != rendered:
            raise ManifestError(f"refusing to overwrite a different manifest: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
