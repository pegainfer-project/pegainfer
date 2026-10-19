"""Immutable fixed-batch request identities; vllm-bench owns prompt generation."""

from __future__ import annotations

import copy
import re
from typing import Any

from manifest import canonical_execution_configuration, canonical_sampling_configuration, sha256_json
from measurement_schema import DECODE_STATISTICS_POLICY, REFERENCE_POLICIES, reference_policy, MeasurementError, digest, integer, number, text

SCHEMA_VERSION = 4
KIND = "request_plan"


class PlanError(ValueError):
    """A logical request plan cannot be executed without guessing its meaning."""


def _seed(base: int, identity: str, phase: str, repeat: int) -> int:
    return int(sha256_json([SCHEMA_VERSION, base, identity, phase, repeat])[:8], 16)


def _plan_fields(value: Any, allowed: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanError(f"{field} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise PlanError(f"{field} contains unknown fields: {sorted(unknown)}")
    return value


def build_plan(cell: dict[str, Any], client: dict[str, Any]) -> dict[str, Any]:
    try:
        digest(cell.get("cell_id"), "cell_id")
        concurrency = integer(cell.get("concurrency"), "concurrency", positive=True)
        repeats = integer(cell.get("repeats"), "repeats", positive=True)
        warmups = integer(cell.get("warmup_repeats"), "warmup_repeats")
        base = integer(cell.get("seed"), "seed", maximum=2**32 - 1)
        if cell.get("arrival") != {"mode": "fixed_batch"}:
            raise PlanError("request plan requires fixed_batch")
        lifecycle = cell.get("cache_lifecycle", {})
        bootstrap_repeats = lifecycle.get("bootstrap_repeats", 0)
        batches = []
        requests = []
        bootstrap_index = 0
        prefix_seed = None
        if cell["kind"] == "cached_prefill":
            prefix_seed = _seed(base, cell["shared_prefix"]["identity"], "prefix", 0)

        def append(phase: str, repeat: int, *, before_phase: str | None = None, before_repeat: int | None = None) -> None:
            batch = {
                "phase": phase, "repeat": repeat,
                "seed": _seed(base, cell["cell_id"], phase, repeat),
                "request_keys": [],
            }
            if before_phase is not None:
                batch.update({"before_phase": before_phase, "before_repeat": before_repeat})
            if prefix_seed is not None:
                batch["prefix_seed"] = prefix_seed
            for index in range(concurrency):
                key = f"{cell['cell_id']}:{phase}:{repeat}:{index}"
                row = {
                    "pairing_key": key, "phase": phase, "repeat": repeat, "index": index,
                    "input_tokens": cell["input_tokens"], "output_tokens": cell["output_tokens"],
                    "sampling": copy.deepcopy(cell["sampling"]), "dispatch_offset_ms": 0,
                }
                batch["request_keys"].append(key)
                requests.append(row)
            batches.append(batch)

        for phase, count in (("warmup", warmups), ("measured", repeats)):
            for repeat in range(count):
                if bootstrap_repeats and (lifecycle.get("reset_before") == "repeat" or not batches):
                    for _ in range(bootstrap_repeats):
                        append("bootstrap", bootstrap_index, before_phase=phase, before_repeat=repeat)
                        bootstrap_index += 1
                append(phase, repeat)
        plan = {
            "kind": KIND, "schema_version": SCHEMA_VERSION, "cell_id": cell["cell_id"],
            "matrix_name": cell["matrix_name"], "split": cell["split"], "workload_kind": cell["kind"],
            "concurrency": concurrency, "warmup_repeats": warmups, "repeats": repeats,
            "client": {"driver": "vllm-bench", "version": client["version"],
                       "binary_sha256": client["binary_sha256"], "tokenizer_sha256": client["tokenizer_sha256"]},
            "generation": {**copy.deepcopy(cell["generation"]), "seed": base},
            "timing": copy.deepcopy(cell["timing"]),
            "retry": copy.deepcopy(client["retry"]),
            "observer": copy.deepcopy(client["observer"]),
            "shape": {"input_tokens": cell["input_tokens"], "output_tokens": cell["output_tokens"]},
            "sampling": copy.deepcopy(cell["sampling"]),
            "arrival": {"mode": "fixed_batch", "clock": "monotonic"},
            "batches": batches, "logical_requests": requests,
        }
        for field in ("cache_lifecycle", "shared_prefix", "prefix_tokens", "new_tokens", "num_ctx", "decode_statistics", "reference_policy"):
            if field in cell:
                plan[field] = copy.deepcopy(cell[field])
        plan["plan_sha256"] = sha256_json(plan)
        validate_plan(plan)
        return plan
    except (MeasurementError, KeyError, TypeError) as exc:
        raise PlanError(str(exc)) from exc


def validate_plan(plan: dict[str, Any]) -> None:
    """Validate a loaded plan independently of campaign compilation."""
    try:
        if not isinstance(plan, dict) or plan.get("kind") != KIND or type(plan.get("schema_version")) is not int or plan["schema_version"] != SCHEMA_VERSION:
            raise PlanError("unsupported request plan kind/schema_version")
        _plan_fields(plan, {"kind", "schema_version", "cell_id", "matrix_name", "split", "workload_kind", "concurrency", "warmup_repeats", "repeats", "client", "generation", "timing", "shape", "sampling", "retry", "observer", "arrival", "batches", "logical_requests", "plan_sha256", "cache_lifecycle", "shared_prefix", "prefix_tokens", "new_tokens", "num_ctx", "decode_statistics", "reference_policy"}, "plan")
        digest(plan.get("cell_id"), "cell_id")
        digest(plan.get("plan_sha256"), "plan_sha256")
        if sha256_json({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
            raise PlanError("request plan digest mismatch")
        matrix_name = text(plan.get("matrix_name"), "matrix_name")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", matrix_name):
            raise PlanError("matrix_name must be a safe identifier")
        from manifest import SPLITS
        if plan.get("split") not in SPLITS:
            raise PlanError("unknown plan split")
        workload = plan.get("workload_kind")
        if workload not in ("cold_prefill", "steady_decode", "cached_prefill"):
            raise PlanError("unsupported workload_kind")
        client = _plan_fields(plan.get("client"), {"driver", "version", "binary_sha256", "tokenizer_sha256"}, "client")
        if client.get("driver") != "vllm-bench":
            raise PlanError("client.driver must be vllm-bench")
        text(client.get("version"), "client.version")
        for field in ("binary_sha256", "tokenizer_sha256"):
            digest(client.get(field), f"client.{field}")
        generation = _plan_fields(plan.get("generation"), {"dataset", "shuffle", "prompt_representation", "seed_policy", "seed"}, "generation")
        base_seed = integer(generation.get("seed"), "generation.seed", maximum=2**32 - 1)
        canonical_execution_configuration({
            "seed": base_seed, "sampling": plan.get("sampling"), "timing": plan.get("timing"),
            "generation": {key: value for key, value in generation.items() if key != "seed"},
        }, "plan")
        from evidence_proxy import RETRY_CONTRACT, OBSERVER_CONFIG, OBSERVER_SOURCES
        if plan.get("retry") != RETRY_CONTRACT or not isinstance(plan.get("observer"), dict):
            raise PlanError("missing observer/retry contract")
        observer = _plan_fields(plan["observer"], {"config", "dependencies", "sources", "sha256"}, "observer")
        digest(observer.get("sha256"), "observer.sha256")
        if observer.get("config") != OBSERVER_CONFIG or not isinstance(observer.get("dependencies"), dict) or not isinstance(observer.get("sources"), dict):
            raise PlanError("invalid observer configuration")
        if set(observer['sources']) != set(OBSERVER_SOURCES):
            raise PlanError('incomplete observer sources; recompile campaign')
        for name, checksum in observer['sources'].items():
            digest(checksum, f'observer.sources.{name}')
        if observer["sha256"] != sha256_json([observer["config"], observer["dependencies"], observer["sources"]]):
            raise PlanError("observer identity digest mismatch")
        c = integer(plan.get("concurrency"), "concurrency", positive=True, maximum=2**32 - 1)
        repeats = integer(plan.get("repeats"), "repeats", positive=True, maximum=2**32 - 1)
        warmups = integer(plan.get("warmup_repeats"), "warmup_repeats", maximum=2**32 - 1)
        shape = _plan_fields(plan.get("shape"), {"input_tokens", "output_tokens"}, "shape")
        for field in ("input_tokens", "output_tokens"):
            integer(shape.get(field), f"shape.{field}", positive=True, maximum=2**32 - 1)
        integer(shape["input_tokens"] + shape["output_tokens"], "shape.total_tokens", maximum=2**32 - 1)
        lifecycle = _plan_fields(plan.get("cache_lifecycle", {}), {"namespace", "reset_before", "bootstrap_repeats"}, "cache_lifecycle")
        boots = 0
        if "cache_lifecycle" in plan:
            text(lifecycle.get("namespace"), "cache_lifecycle.namespace")
            if lifecycle.get("reset_before") not in ("cell", "repeat"):
                raise PlanError("cache_lifecycle.reset_before must be cell or repeat")
            boots = integer(lifecycle.get("bootstrap_repeats"), "cache_lifecycle.bootstrap_repeats", maximum=2**32 - 1)
        prefix_seed = None
        if workload == "cached_prefill":
            if c != 1 or generation["dataset"] != "prefix_repetition" or not boots:
                raise PlanError("cached_prefill requires concurrency=1, prefix_repetition and bootstrap")
            prefix = integer(plan.get("prefix_tokens"), "prefix_tokens", positive=True, maximum=2**32 - 1)
            tail = integer(plan.get("new_tokens"), "new_tokens", positive=True, maximum=2**32 - 1)
            if prefix + tail != shape["input_tokens"]:
                raise PlanError("cached prefix/new tokens do not match input shape")
            shared = _plan_fields(plan.get("shared_prefix"), {"identity", "suffix_mode"}, "shared_prefix")
            text(shared.get("identity"), "shared_prefix.identity")
            if shared.get("suffix_mode") != "request_index":
                raise PlanError("shared_prefix.suffix_mode must be request_index")
            prefix_seed = _seed(base_seed, shared["identity"], "prefix", 0)
        else:
            if generation["dataset"] != "random" or boots:
                raise PlanError("cold/decode require random generation without bootstrap")
            if any(field in plan for field in ("shared_prefix", "prefix_tokens", "new_tokens")):
                raise PlanError("prefix fields are only supported for cached_prefill")
            if workload == "cold_prefill" and c != 1:
                raise PlanError("cold_prefill requires concurrency=1")
        if workload == "steady_decode":
            total = integer(plan.get("num_ctx"), "num_ctx", positive=True, maximum=2**32 - 1)
            if total != c * shape["input_tokens"] or shape["output_tokens"] not in (8, 16):
                raise PlanError("steady_decode requires num_ctx = concurrency * input_tokens and output_tokens 8 or 16")
            if plan.get("decode_statistics") != DECODE_STATISTICS_POLICY:
                raise PlanError("invalid fixed decode statistics policy")
        elif any(field in plan for field in ("num_ctx", "decode_statistics")):
            raise PlanError("decode fields are only supported for steady_decode")
        if workload in REFERENCE_POLICIES and plan.get("reference_policy") != reference_policy(workload, plan["split"]):
            raise PlanError("reference policy differs from the frozen workload contract")
        if plan.get("arrival") != {"mode": "fixed_batch", "clock": "monotonic"}:
            raise PlanError("invalid fixed_batch arrival contract")
        batches = plan.get("batches")
        rows = plan.get("logical_requests")
        if not isinstance(batches, list) or not isinstance(rows, list):
            raise PlanError("request plan must contain batches and logical_requests")
        regular_count = warmups + repeats
        bootstrap_count = boots * (regular_count if lifecycle.get("reset_before") == "repeat" else 1)
        if len(batches) != regular_count + bootstrap_count or len(rows) != len(batches) * c:
            raise PlanError("batch/request count does not match phase/repeat contract")
        by_key = {}
        for row in rows:
            _plan_fields(row, {"pairing_key", "phase", "repeat", "index", "input_tokens", "output_tokens", "sampling", "dispatch_offset_ms"}, "logical request")
            text(row.get("pairing_key"), "pairing_key")
            for field in ("repeat", "index", "input_tokens", "output_tokens"):
                integer(row.get(field), f"logical request.{field}", positive=field.endswith("tokens"), maximum=2**32 - 1)
            number(row.get("dispatch_offset_ms"), "dispatch_offset_ms")
            canonical_sampling_configuration(row.get("sampling"), "logical request.sampling")
            if row["pairing_key"] in by_key:
                raise PlanError("duplicate pairing_key")
            by_key[row["pairing_key"]] = row
        expected_batches = []
        bootstrap = 0
        for phase, count in (("warmup", warmups), ("measured", repeats)):
            for repeat in range(count):
                if lifecycle.get("bootstrap_repeats", 0) and (lifecycle.get("reset_before") == "repeat" or not expected_batches):
                    for _ in range(lifecycle["bootstrap_repeats"]):
                        expected_batches.append(("bootstrap", bootstrap, phase, repeat))
                        bootstrap += 1
                expected_batches.append((phase, repeat, None, None))
        if len(batches) != len(expected_batches):
            raise PlanError("batch count does not match phase/repeat contract")
        seen = []
        for batch, (phase, repeat, before_phase, before_repeat) in zip(batches, expected_batches):
            _plan_fields(batch, {"phase", "repeat", "seed", "request_keys", "before_phase", "before_repeat", "prefix_seed"}, "batch")
            integer(batch.get("repeat"), "batch.repeat")
            integer(batch.get("seed"), "batch.seed", maximum=2**32 - 1)
            if (batch.get("phase"), batch.get("repeat")) != (phase, repeat):
                raise PlanError("batch order does not match phase/repeat contract")
            if before_phase is not None and (batch.get("before_phase"), batch.get("before_repeat")) != (before_phase, before_repeat):
                raise PlanError("bootstrap belongs to a different batch")
            if before_phase is not None:
                integer(batch.get("before_repeat"), "batch.before_repeat")
            elif "before_phase" in batch or "before_repeat" in batch:
                raise PlanError("only bootstrap batches can declare before_phase/before_repeat")
            keys = [f"{plan['cell_id']}:{phase}:{repeat}:{i}" for i in range(c)]
            if batch.get("request_keys") != keys:
                raise PlanError("fixed batch must contain exactly concurrency keys")
            if batch.get("seed") != _seed(base_seed, plan["cell_id"], phase, repeat):
                raise PlanError("batch seed differs from phase_repeat mapping")
            if prefix_seed is not None:
                integer(batch.get("prefix_seed"), "batch.prefix_seed", maximum=2**32 - 1)
            if batch.get("prefix_seed") != prefix_seed or (prefix_seed is None and "prefix_seed" in batch):
                raise PlanError("batch prefix seed differs from shared prefix identity")
            for index, key in enumerate(keys):
                row = by_key.get(key)
                if row is None or (row.get("phase"), row.get("repeat"), row.get("index")) != (phase, repeat, index):
                    raise PlanError("missing or mismatched logical request identity")
                if row.get("input_tokens") != shape["input_tokens"] or row.get("output_tokens") != shape["output_tokens"] or row.get("sampling") != plan["sampling"] or row.get("dispatch_offset_ms") != 0:
                    raise PlanError("logical request differs from declared shape/sampling/arrival")
            seen.extend(keys)
        if seen != [row["pairing_key"] for row in rows]:
            raise PlanError("request order or key set differs from batches")
    except (MeasurementError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, PlanError):
            raise
        raise PlanError(str(exc)) from exc
