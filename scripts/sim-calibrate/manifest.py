#!/usr/bin/env python3
"""Compile named campaign matrices and canonical serving/profile identities."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from itertools import product
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from measurement_schema import DECODE_STATISTICS_POLICY, REFERENCE_POLICIES, reference_policy, MeasurementError, contract as measurement_contract, digest, integer, number, text


CAMPAIGN_SPEC_SCHEMA_VERSION = 8
COMPILED_CAMPAIGN_SCHEMA_VERSION = 8
SPLITS = ("train", "threshold_validation", "held_out_mixed", "sealed_test")
WORKLOAD_KINDS = ("cold_prefill", "steady_decode", "cached_prefill", "mixed")

_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


class ManifestError(ValueError):
    """Raised when a campaign spec cannot be made immutable and safe."""


def canonical_json(value: Any) -> str:
    """Render JSON deterministically for IDs and digest comparisons."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _axis_values(value: Any, field: str) -> tuple[int, ...]:
    """Canonical positive integer axes, with inclusive ranges and no duplicates."""
    if isinstance(value, dict):
        if set(value) == {"values"}:
            value = value["values"]
        elif set(value) == {"start", "stop", "step"}:
            start, stop, step = (value[name] for name in ("start", "stop", "step"))
            for name, item in (("start", start), ("stop", stop), ("step", step)):
                integer(item, f"{field}.{name}", positive=True)
            if stop < start:
                raise ManifestError(f"{field}.stop must be >= {field}.start")
            value = list(range(start, stop + 1, step))
        else:
            raise ManifestError(f"{field} must use values or start/stop/step")
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{field} must be a non-empty array or range object")
    for item in value:
        integer(item, field, positive=True)
    if len(set(value)) != len(value):
        raise ManifestError(f"{field} must not contain duplicates")
    return tuple(sorted(value))


def _matrix_cells(name: str, matrix: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a validated named recipe; experiment policy stays on each cell."""
    kind = matrix["kind"]
    axes = {"input_tokens", "num_ctx", "concurrency", "prefix_tokens", "new_tokens",
            "required_input_tokens"}
    policy = {key: value for key, value in matrix.items() if key not in axes}
    cells = []
    if kind == "cold_prefill":
        if matrix["concurrency"] != 1:
            raise ManifestError(f"{name}.concurrency must be 1 for cold_prefill")
        inputs = _axis_values(matrix.get("input_tokens"), f"{name}.input_tokens")
        if matrix.get("required_input_tokens") is not None:
            required = _axis_values(matrix["required_input_tokens"], f"{name}.required_input_tokens")
            missing = sorted(set(required) - set(inputs))
            if missing:
                raise ManifestError(f"{name}.input_tokens is missing required values: {missing}")
        metadata = {**matrix.get("metadata", {}), "matrix": kind, "input_axis": list(inputs)}
        cells = [{"name": f"{name}-{tokens}", "input_tokens": tokens,
                  "concurrency": 1, "metadata": dict(metadata)} for tokens in inputs]
    elif kind == "steady_decode":
        if matrix["output_tokens"] not in (8, 16):
            raise ManifestError(f"{name}.output_tokens must be 8 or 16")
        requests = _axis_values(matrix.get("concurrency"), f"{name}.concurrency")
        contexts = _axis_values(matrix.get("num_ctx"), f"{name}.num_ctx")
        for count, total in product(requests, contexts):
            if total % count:
                raise ManifestError(f"{name}.num_ctx must be divisible by every concurrency value")
            cells.append({
                "name": f"{name}-c{count}-ctx{total}", "input_tokens": total // count,
                "concurrency": count, "num_ctx": total, "sum_decode_ctx_tokens": total,
                "decode_statistics": dict(DECODE_STATISTICS_POLICY),
                "metadata": {**matrix.get("metadata", {}), "matrix": kind},
            })
    elif kind == "cached_prefill":
        if matrix["concurrency"] != 1:
            raise ManifestError(f"{name}.concurrency must be 1 for cached_prefill")
        prefixes = _axis_values(matrix.get("prefix_tokens"), f"{name}.prefix_tokens")
        tails = _axis_values(matrix.get("new_tokens"), f"{name}.new_tokens")
        cells = [{"name": f"{name}-prefix{prefix}-new{tail}", "input_tokens": prefix + tail,
                  "prefix_tokens": prefix, "new_tokens": tail, "concurrency": 1,
                  "metadata": {**matrix.get("metadata", {}), "matrix": kind,
                               "prefix_axis": list(prefixes), "new_token_axis": list(tails)}}
                 for prefix, tail in product(prefixes, tails)]
    return [{**policy, **cell, "matrix_name": name} for cell in cells]


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


def _merge_configuration(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_configuration(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _fields(value: Any, allowed: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{field} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ManifestError(f"{field} contains unknown fields: {sorted(unknown)}")
    return value


def canonical_sampling_configuration(value: Any, field: str) -> dict[str, Any]:
    """Validate and normalize sampling on cells, plans and logical requests."""
    sampling = _fields(value, {"temperature", "top_p", "top_k", "ignore_eos"}, field)
    temperature = number(sampling.get("temperature"), "sampling.temperature")
    top_p = number(sampling.get("top_p"), "sampling.top_p")
    if not 0 < top_p <= 1:
        raise ManifestError("sampling.top_p must be in (0,1]")
    if "top_k" in sampling:
        integer(sampling["top_k"], "sampling.top_k", positive=True)
    if not isinstance(sampling.get("ignore_eos"), bool):
        raise ManifestError("sampling.ignore_eos must be boolean")
    return {**sampling, "temperature": temperature, "top_p": top_p}


def canonical_execution_configuration(value: dict[str, Any], field: str) -> dict[str, Any]:
    """Validate generation, sampling and timing shared by campaign and plan."""
    integer(value.get("seed"), f"{field}.seed", maximum=2**32 - 1)
    sampling = canonical_sampling_configuration(value.get("sampling"), f"{field}.sampling")
    generation = _fields(value.get("generation"), {"dataset", "shuffle", "prompt_representation", "seed_policy"}, f"{field}.generation")
    if generation.get("dataset") not in ("random", "custom"):
        raise ManifestError("generation.dataset must be random or custom")
    if generation.get("shuffle") is not False:
        raise ManifestError("generation.shuffle must be false for fixed-batch calibration")
    if generation.get("prompt_representation") not in ("token_ids", "text"):
        raise ManifestError("generation.prompt_representation must be token_ids or text")
    if generation.get("seed_policy") != "phase_repeat":
        raise ManifestError("generation.seed_policy must be phase_repeat")
    from evidence_proxy import TIMING_CONTRACT
    timing = value.get("timing")
    if timing != TIMING_CONTRACT:
        raise ManifestError("timing must use the explicit proxy token-ID/text contract")
    normalized = copy.deepcopy(value)
    normalized["sampling"] = sampling
    return normalized


def _matrix_configuration(name: str, matrix: Any, defaults: dict[str, Any]) -> dict[str, Any]:
    """Resolve common experiment configuration before expanding workload axes."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ManifestError("matrix names must be safe non-empty identifiers")
    common_fields = {"kind", "split", "concurrency", "output_tokens", "warmup_repeats", "repeats", "seed", "sampling", "generation", "timing", "arrival", "cache_lifecycle", "metadata"}
    kind_fields = {
        "cold_prefill": {"input_tokens", "required_input_tokens"},
        "steady_decode": {"num_ctx"},
        "cached_prefill": {"prefix_tokens", "new_tokens", "shared_prefix", "bootstrap_output_tokens"},
    }
    if not isinstance(matrix, dict) or not isinstance(matrix.get("kind"), str) or matrix["kind"] not in kind_fields:
        raise ManifestError(f"matrix {name}.kind must be cold_prefill, steady_decode or cached_prefill")
    _fields(matrix, common_fields | kind_fields[matrix["kind"]], f"matrix {name}")
    if "metadata" in matrix and not isinstance(matrix["metadata"], dict):
        raise ManifestError(f"matrix {name}.metadata must be an object")
    if matrix.get("split") not in SPLITS:
        raise ManifestError(f"matrix {name}.split is required and must be a known split")
    if matrix.get("arrival") != {"mode": "fixed_batch"}:
        raise ManifestError(f"matrix {name}.arrival must explicitly declare fixed_batch")
    for field in ("output_tokens", "repeats"):
        integer(matrix.get(field), f"matrix {name}.{field}", positive=True, maximum=2**32-1)
    integer(matrix.get("warmup_repeats"), f"matrix {name}.warmup_repeats", maximum=2**32-1)
    execution = canonical_execution_configuration(_merge_configuration(defaults, {k: matrix[k] for k in defaults if k in matrix}), f"matrix {name}")
    lifecycle = matrix.get("cache_lifecycle")
    if lifecycle is not None:
        _fields(lifecycle, {"namespace", "reset_before", "bootstrap_repeats", "recovery"}, "cache_lifecycle")
        text(lifecycle.get("namespace"), "cache_lifecycle.namespace")
        if lifecycle.get("reset_before") not in ("cell", "repeat"):
            raise ManifestError("cache_lifecycle.reset_before must be cell or repeat")
        integer(lifecycle.get("bootstrap_repeats"), "bootstrap_repeats", maximum=2**32-1)
        if lifecycle.get("recovery", "reset") not in ("reset", "restart"):
            raise ManifestError("cache_lifecycle.recovery must be reset or restart")
    return {**matrix, **execution}


def _finalize_cell(cell: dict[str, Any], variants: dict[str, Any], assets: dict[str, Any]) -> dict[str, Any]:
    """Validate shared request limits and freeze the compiled cell identity."""
    for field in ("input_tokens", "output_tokens", "concurrency"):
        integer(cell[field], field, positive=True, maximum=2**32-1)
    for variant in variants.values():
        maximum = max(cell["input_tokens"] + cell["output_tokens"], cell.get("prefix_tokens", 0) + cell.get("bootstrap_output_tokens", 0))
        if maximum > variant["runtime"]["scheduler"]["max_model_len"]:
            raise ManifestError(f"cell {cell['name']} exceeds a variant context ceiling")
    if cell["kind"] in REFERENCE_POLICIES:
        cell = {**cell, "reference_policy": reference_policy(cell["kind"], cell["split"])}
    cell = {**cell, "fit_policy": {"train": cell["split"] == "train", "excluded_metrics": ["ttft_ms"] if cell["concurrency"] > 1 else []}}
    cell["cell_id"] = sha256_json({"schema_version": COMPILED_CAMPAIGN_SCHEMA_VERSION, "client_assets": assets, "cell": cell})
    return cell


def compile_manifest(spec: dict[str, Any], assets: dict[str, Any]) -> dict[str, Any]:
    """Compile the new input contract; historical manifest consumers stay separate."""
    try:
        _fields(spec, {"kind", "schema_version", "client", "variants", "workload", "output"}, "campaign spec")
        version = spec.get("schema_version", CAMPAIGN_SPEC_SCHEMA_VERSION)
        if spec.get("kind", "campaign_spec") != "campaign_spec" or type(version) is not int or version != CAMPAIGN_SPEC_SCHEMA_VERSION:
            raise ManifestError("unsupported campaign spec kind/schema_version")
        output = _fields(spec.get("output"), {"artifact_root"}, "output")
        text(output.get("artifact_root"), "output.artifact_root")
        client = _fields(spec.get("client"), {"name", "version", "binary", "tokenizer", "seed", "sampling", "generation", "timing", "retry"}, "client")
        for field in ("name", "version", "binary", "tokenizer"):
            text(client.get(field), f"client.{field}")
        defaults = canonical_execution_configuration({key: client.get(key) for key in ("seed", "sampling", "generation", "timing")}, "client")
        client = {**client, **defaults}
        from evidence_proxy import RETRY_CONTRACT
        if client.get("retry") != RETRY_CONTRACT:
            raise ManifestError("client.retry must declare at-most-once proxy forwarding and unavailable client retry control")
        supplied = _fields(spec.get("variants"), {"target", "zero_cost", "sim"}, "variants")
        if set(supplied) != {"target", "zero_cost", "sim"}:
            raise ManifestError("variants must contain target, zero_cost and sim")
        variants: dict[str, Any] = {}
        target_runtime = supplied["target"].get("runtime") if isinstance(supplied["target"], dict) else None
        if not isinstance(target_runtime, dict):
            raise ManifestError("variants.target.runtime must be an object")
        for name in ("target", "zero_cost", "sim"):
            raw = _fields(supplied[name], {"identity", "runtime", "server_flags", "profile_path", "launch"}, f"variants.{name}")
            identity = copy.deepcopy(_fields(raw.get("identity"), {"endpoint", "model", "engine", "engine_version", "model_revision", "model_config_sha256"}, f"variants.{name}.identity"))
            for field in ("endpoint", "model", "engine", "engine_version", "model_revision"):
                text(identity.get(field), f"variants.{name}.identity.{field}")
            endpoint = urlsplit(identity["endpoint"])
            if endpoint.scheme not in ("http", "https") or not endpoint.netloc or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment or endpoint.path not in ("", "/"):
                raise ManifestError(f"variants.{name}.identity.endpoint must be an HTTP URL without credentials")
            identity["endpoint"] = identity["endpoint"].rstrip("/")
            digest(identity.get("model_config_sha256"), f"variants.{name}.model_config_sha256")
            raw_runtime = _fields(raw.get("runtime"), {"gpu", "backend", "driver", "cuda", "parallelism", "quantization", "scheduler", "cache"}, f"variants.{name}.runtime")
            runtime = _merge_configuration(target_runtime, raw_runtime)
            runtime["backend"] = raw_runtime.get("backend", identity["engine"])
            for field in ("gpu", "backend", "driver", "cuda", "parallelism", "quantization"):
                if field in runtime:
                    text(runtime[field], f"variants.{name}.runtime.{field}")
            scheduler = canonical_scheduler_identity(runtime.get("scheduler"))
            for field in ("max_num_seqs", "max_num_batched_tokens", "max_model_len"):
                integer(scheduler[field], f"scheduler.{field}", positive=True, maximum=2**32-1)
            if scheduler["policy"] != "vllm_v1" or scheduler["max_num_seqs"] > scheduler["max_num_batched_tokens"]:
                raise ManifestError("scheduler requires vllm_v1 and max_num_seqs <= max_num_batched_tokens")
            runtime["scheduler"] = scheduler
            cache = canonical_cache_identity(runtime.get("cache"))
            if cache["policy"] not in ("on", "off") or (cache["enabled"] and (cache["block_size"] == 0 or cache["alignment"] == 0)):
                raise ManifestError("cache policy must be on/off with positive enabled block size/alignment")
            flags = raw.get("server_flags")
            if not isinstance(flags, list) or any(not isinstance(flag, str) or not flag for flag in flags):
                raise ManifestError(f"variants.{name}.server_flags must be an array of non-empty strings")
            if any(flag == "--profile" or flag.startswith("--profile=") for flag in flags):
                raise ManifestError("declare profile_path instead of duplicating --profile in server_flags")
            if name == "zero_cost" and "profile_path" in raw:
                raise ManifestError("zero_cost profile is prepared after target collection, not supplied at compile time")
            if name == "target" and "profile_path" in raw:
                raise ManifestError("target cannot declare a simulator profile_path")
            variants[name] = {"identity": identity, "runtime": runtime, "server_flags": list(flags)}
            if "launch" in raw:
                if name != "target":
                    raise ManifestError("only target may declare launch")
                launch = _fields(raw["launch"], {"argv", "environment", "startup_timeout", "executable"}, "target.launch")
                argv = launch.get("argv")
                if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg for arg in argv) or not Path(argv[0]).is_absolute():
                    raise ManifestError("target.launch.argv must start with an absolute executable")
                environment = launch.get("environment", {})
                if not isinstance(environment, dict) or any(not isinstance(key, str) or not key or "=" in key or not isinstance(value, str) for key, value in environment.items()):
                    raise ManifestError("target.launch.environment must contain string values")
                timeout = number(launch.get("startup_timeout", 90), "target.launch.startup_timeout")
                if timeout <= 0:
                    raise ManifestError("target.launch.startup_timeout must be positive")
                executable = _fields(launch.get('executable'), {'sha256', 'size'}, 'target.launch.executable')
                digest(executable.get('sha256'), 'target.launch.executable.sha256')
                integer(executable.get('size'), 'target.launch.executable.size', positive=True)
                variants[name]["launch"] = {"argv": list(argv), "environment": dict(environment), "startup_timeout": timeout, 'executable': dict(executable)}
            if "profile_path" in raw:
                variants[name]["profile_path"] = text(raw["profile_path"], f"variants.{name}.profile_path")
        if variants["zero_cost"]["runtime"]["scheduler"] != variants["sim"]["runtime"]["scheduler"] or canonical_cache_identity(variants["zero_cost"]["runtime"].get("cache")) != canonical_cache_identity(variants["sim"]["runtime"].get("cache")):
            raise ManifestError("zero_cost and sim must have identical scheduler/cache configuration")
        if any(v["identity"]["model_config_sha256"] != variants["target"]["identity"]["model_config_sha256"] for v in variants.values()):
            raise ManifestError("variant model config digests differ")
        workload = _fields(spec.get("workload"), {"matrices"}, "workload")
        matrices = workload.get("matrices")
        if not isinstance(matrices, dict) or not matrices:
            raise ManifestError("workload.matrices must be a non-empty named object")
        cells = []
        for name in sorted(matrices):
            matrix = _matrix_configuration(name, matrices[name], defaults)
            if matrix["kind"] != "steady_decode":
                integer(matrix.get("concurrency"), f"matrix {name}.concurrency", positive=True)
            if matrix.get("kind") == "cached_prefill":
                if matrix["generation"]["dataset"] != "custom" or matrix["generation"]["prompt_representation"] != "text":
                    raise ManifestError("cached_prefill requires custom text generation")
                integer(matrix.get("bootstrap_output_tokens"), "bootstrap_output_tokens", positive=True, maximum=2**32-1)
                prefix = _fields(matrix.get("shared_prefix"), {"identity", "suffix_mode"}, "shared_prefix")
                text(prefix.get("identity"), "shared_prefix.identity")
                if prefix.get("suffix_mode") != "request_index":
                    raise ManifestError("shared_prefix.suffix_mode must be request_index")
            elif matrix["generation"]["dataset"] != "random":
                raise ManifestError("cold/decode require random generation")
            elif matrix["generation"]["prompt_representation"] != "token_ids":
                raise ManifestError("cold/decode require token-ID prompts")
            lifecycle = matrix.get("cache_lifecycle")
            if lifecycle is not None and matrix["kind"] != "cached_prefill" and lifecycle["bootstrap_repeats"]:
                raise ManifestError("bootstrap is only supported for cached_prefill")
            if matrix.get("kind") == "cached_prefill" and (lifecycle is None or not lifecycle["bootstrap_repeats"]):
                raise ManifestError("cached_prefill requires an explicit bootstrap lifecycle")
            for cell in _matrix_cells(name, matrix):
                if cell["kind"] == "cached_prefill":
                    if lifecycle.get("recovery", "reset") == "restart" and "launch" not in variants["target"]:
                        raise ManifestError("cached restart recovery requires target.launch")
                    cache = canonical_cache_identity(variants["target"]["runtime"].get("cache"))
                    if not cache["enabled"] or cell["prefix_tokens"] % cache["alignment"]:
                        raise ManifestError("cached prefix must align to the enabled target cache")
                if cell["kind"] == "steady_decode" and cell["split"] == "train" and any(cell["concurrency"] > v["runtime"]["scheduler"]["max_num_seqs"] for v in variants.values()):
                    raise ManifestError("decode train concurrency exceeds scheduler capacity")
                cells.append(_finalize_cell(cell, variants, assets))
        # Output locations and profile bindings are storage metadata, not workload identity.
        variant_identity = {name: {k: v for k, v in value.items() if k != "profile_path"} for name, value in variants.items()}
        client_identity = {key: value for key, value in client.items() if key not in ("binary", "tokenizer")}
        identity = {"kind": "campaign_spec", "schema_version": CAMPAIGN_SPEC_SCHEMA_VERSION, "client": {**client_identity, **assets}, "variants": variant_identity, "workload": {"cells": cells}}
        campaign_id = sha256_json(identity)
        splits = {split: [cell["cell_id"] for cell in cells if cell["split"] == split] for split in SPLITS}
        return {
            "kind": "compiled_campaign", "schema_version": COMPILED_CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": campaign_id, "source_digest": campaign_id,
            "client": {**copy.deepcopy(client), **assets}, "variants": variants,
            "workload": {"cells": cells, "cell_count": len(cells)}, "splits": splits,
            "required_scope": {"cells": [cell["cell_id"] for cell in cells], "sha256": sha256_json(splits)},
            "measurement_contract": measurement_contract(),
        }
    except (MeasurementError, ValueError, TypeError) as exc:
        if isinstance(exc, ManifestError):
            raise
        raise ManifestError(str(exc)) from exc
