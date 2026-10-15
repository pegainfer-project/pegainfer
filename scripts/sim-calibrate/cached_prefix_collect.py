#!/usr/bin/env python3
"""Convert vllm-bench prefix-repetition output plus external evidence.

This module is intentionally independent from the legacy calibration collector.
It never constructs requests and never imports ``collect``, ``benchmark`` or
``streaming``.  vllm-bench owns request generation/timing; an external recorder
owns request identity and cache evidence.  A result is accepted only when both
parts describe the same measured request sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
REQUIRED_RESULT_ARRAYS = ("input_lens", "output_lens", "ttfts", "latencies", "itls", "errors")


class CachedPrefixError(ValueError):
    """Raised when a prefix-repetition bundle is not trustworthy evidence."""


def _finite(value: Any, field: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CachedPrefixError(f"{field} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or (nonnegative and rendered < 0.0):
        raise CachedPrefixError(f"{field} must be finite and non-negative")
    return rendered


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CachedPrefixError(f"{field} must be a positive integer")
    return value


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CachedPrefixError(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CachedPrefixError(f"JSON root must be an object: {path}")
    return value


def _require_list(document: dict[str, Any], field: str) -> list[Any]:
    value = document.get(field)
    if not isinstance(value, list):
        raise CachedPrefixError(f"vllm-bench result.{field} must be an array")
    return value


def _require_exact_usage(evidence: dict[str, Any], *, input_tokens: int, output_tokens: int) -> None:
    usage = evidence.get("usage")
    if not isinstance(usage, dict):
        raise CachedPrefixError("evidence.usage is required")
    expected = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if usage != expected:
        raise CachedPrefixError(f"usage mismatch: expected {expected}, got {usage}")


def _cache_evidence(
    evidence: dict[str, Any],
    *,
    prefix_tokens: int,
    prefix_identity: str,
    require_hit: bool,
) -> dict[str, Any] | None:
    raw = evidence.get("cache_hit_evidence")
    if not isinstance(raw, dict):
        if not require_hit:
            return None
        raise CachedPrefixError("cache_hit_evidence is required")
    cached_tokens = raw.get("cached_tokens")
    if isinstance(cached_tokens, bool) or not isinstance(cached_tokens, int) or cached_tokens < 0:
        raise CachedPrefixError("cache_hit_evidence.cached_tokens must be a non-negative integer")
    if require_hit and cached_tokens <= 0:
        raise CachedPrefixError("cache_hit_evidence.cached_tokens must be positive")
    # vLLM exposes cumulative block counters on some releases. A serialized
    # request can therefore attribute a small aligned superset of the shared
    # prefix; it is still positive hit evidence as long as it covers the
    # requested prefix. The fit axis remains the manifest prefix_tokens.
    if require_hit and cached_tokens < prefix_tokens:
        raise CachedPrefixError(
            f"cache_hit_evidence.cached_tokens must cover prefix_tokens={prefix_tokens}, got {cached_tokens}"
        )
    if require_hit and not any(raw.get(field) is True for field in ("cache_hit", "prefix_cache_hit", "hit")):
        raise CachedPrefixError("cache_hit_evidence must explicitly report a positive hit")
    if raw.get("expected_cached_tokens") != prefix_tokens:
        raise CachedPrefixError("cache evidence disagrees with prefix_tokens")
    if raw.get("shared_prefix_id") != prefix_identity:
        raise CachedPrefixError("cache evidence disagrees with shared prefix identity")
    if not isinstance(raw.get("shared_prefix_hash"), str) or not raw["shared_prefix_hash"]:
        raise CachedPrefixError("cache evidence must include shared_prefix_hash")
    if raw.get("duration_includes_uncached_tail") is not True:
        raise CachedPrefixError("cache evidence must account for the uncached tail")
    return dict(raw)


def _evidence_records(document: dict[str, Any]) -> list[dict[str, Any]]:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise CachedPrefixError("unsupported evidence schema_version")
    if document.get("ordering") != "serial_arrival":
        raise CachedPrefixError("evidence ordering must be serial_arrival for positional vllm-bench pairing")
    records = document.get("requests")
    if not isinstance(records, list) or not records:
        raise CachedPrefixError("evidence.requests must be a non-empty array")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, item in enumerate(records):
        if not isinstance(item, dict):
            raise CachedPrefixError(f"evidence.requests[{ordinal}] must be an object")
        request_id = item.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise CachedPrefixError(f"evidence.requests[{ordinal}].request_id is required")
        if request_id in seen:
            raise CachedPrefixError(f"duplicate request_id in evidence: {request_id}")
        seen.add(request_id)
        if item.get("ordinal") != ordinal:
            raise CachedPrefixError(f"evidence request ordinal mismatch at index {ordinal}")
        result.append(item)
    return result


def _tpot_ms(latency_s: float, ttft_s: float, output_tokens: int) -> float | None:
    if output_tokens <= 1:
        return None
    value = (latency_s - ttft_s) * 1000.0 / float(output_tokens - 1)
    if not math.isfinite(value) or value < 0.0:
        raise CachedPrefixError("derived TPOT is not finite and non-negative")
    return value


def convert_prefix_repetition(
    result: dict[str, Any],
    evidence: dict[str, Any],
    *,
    cell: dict[str, Any],
    variant: str,
    require_hit: bool = True,
) -> dict[str, Any]:
    """Convert one vllm-bench run to the independent raw measurement schema."""

    if variant not in {"target", "zero_cost", "replay"}:
        raise CachedPrefixError(f"unsupported variant: {variant}")
    prefix_tokens = _positive_int(cell.get("prefix_tokens"), "cell.prefix_tokens")
    input_tokens = _positive_int(cell.get("input_tokens"), "cell.input_tokens")
    raw_new_tokens = cell.get("new_tokens", input_tokens - prefix_tokens)
    new_tokens = _positive_int(raw_new_tokens, "cell.new_tokens")
    output_tokens = _positive_int(cell.get("output_tokens"), "cell.output_tokens")
    prefix = cell.get("shared_prefix")
    if not isinstance(prefix, dict) or not isinstance(prefix.get("identity"), str) or not prefix["identity"]:
        raise CachedPrefixError("cell.shared_prefix.identity is required")
    prefix_identity = prefix["identity"]
    expected_input = prefix_tokens + new_tokens

    if result.get("backend") not in {"vllm", "openai", "openai-chat"}:
        raise CachedPrefixError("vllm-bench result must use an OpenAI-compatible generation backend")
    if result.get("failed") != 0:
        raise CachedPrefixError("vllm-bench result contains failed requests")
    completed = _positive_int(result.get("completed"), "result.completed")
    if cell.get("concurrency") is not None and cell.get("concurrency") != 1:
        raise CachedPrefixError("cached-prefix conversion requires cell.concurrency=1")
    arrays = {field: _require_list(result, field) for field in REQUIRED_RESULT_ARRAYS}
    if any(len(values) != completed for values in arrays.values()):
        raise CachedPrefixError("vllm-bench detailed arrays do not have completed length")
    records = _evidence_records(evidence)
    warmups_excluded = evidence.get("warmups_excluded", 0)
    if isinstance(warmups_excluded, bool) or not isinstance(warmups_excluded, int) or warmups_excluded < 0:
        raise CachedPrefixError("evidence.warmups_excluded must be a non-negative integer")
    if completed < warmups_excluded or len(records) != completed - warmups_excluded:
        raise CachedPrefixError(
            "evidence request count and warmups_excluded do not match vllm-bench completed count"
        )
    if cell.get("repeats") is not None and cell.get("repeats") != len(records):
        raise CachedPrefixError(
            f"measured evidence count must equal cell.repeats={cell['repeats']}, got {len(records)}"
        )
    measured_arrays = {field: values[warmups_excluded:] for field, values in arrays.items()}
    if any(error not in (None, "") for error in measured_arrays["errors"]):
        raise CachedPrefixError("vllm-bench detailed result contains request errors")
    if evidence.get("workload_digest") != _sha256_json({"cell": cell, "completed": len(records)}):
        raise CachedPrefixError("evidence workload_digest does not match cell")
    measurements: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if record.get("phase") != "measured":
            raise CachedPrefixError("evidence requests must contain measured requests only")
        input_tokens = _positive_int(measured_arrays["input_lens"][index], f"result.input_lens[{index}]")
        actual_output = _positive_int(measured_arrays["output_lens"][index], f"result.output_lens[{index}]")
        if input_tokens != expected_input:
            raise CachedPrefixError(
                f"input length mismatch at {index}: expected {expected_input}, got {input_tokens}"
            )
        if actual_output != output_tokens:
            raise CachedPrefixError(
                f"output length mismatch at {index}: expected {output_tokens}, got {actual_output}"
            )
        if record.get("input_tokens") != input_tokens or record.get("output_tokens") != actual_output:
            raise CachedPrefixError(f"evidence token lengths disagree at request {index}")
        _require_exact_usage(record, input_tokens=input_tokens, output_tokens=actual_output)
        if cell.get("ignore_eos", True) and record.get("finish_reason") != "length":
            raise CachedPrefixError(f"finish_reason must be length at request {index}")
        cache_evidence = _cache_evidence(
            record,
            prefix_tokens=prefix_tokens,
            prefix_identity=prefix_identity,
            require_hit=require_hit,
        )
        status = record.get("status")
        if isinstance(status, bool) or not isinstance(status, int) or not 200 <= status < 300:
            raise CachedPrefixError(f"request {index} response status is not successful: {status!r}")
        ttft_s = _finite(measured_arrays["ttfts"][index], f"result.ttfts[{index}]")
        latency_s = _finite(measured_arrays["latencies"][index], f"result.latencies[{index}]")
        itls = measured_arrays["itls"][index]
        if not isinstance(itls, list) or any(_finite(value, f"result.itls[{index}][]") < 0.0 for value in itls):
            raise CachedPrefixError(f"result.itls[{index}] must be a numeric array")
        itl_ms = [_finite(value, f"result.itls[{index}][]") * 1000.0 for value in itls]
        # Streaming clients may coalesce multiple generated tokens into one
        # chunk. Preserve the observed ITLs and mark token timing invalid
        # below instead of rejecting otherwise valid TTFT/E2E/cache evidence.
        observed_request_id = record["request_id"]
        pair_id = cell.get("cell_id") or cell.get("name")
        if not isinstance(pair_id, str) or not pair_id:
            raise CachedPrefixError("cell.cell_id or cell.name is required for paired request identity")
        measurements.append(
            {
                "request_id": f"{pair_id}:measured:{index}:0",
                "observed_request_id": observed_request_id,
                "variant": variant,
                "phase": "measured",
                # prefix_repetition is deliberately serialized with one
                # request in flight, so each completed request is one repeat
                # at index zero in the paired artifact contract.
                "repeat": index,
                "index": 0,
                "ok": True,
                "prompt_tokens": input_tokens,
                "completion_tokens": actual_output,
                "total_tokens": input_tokens + actual_output,
                "finish_reason": record["finish_reason"],
                "ttft_ms": ttft_s * 1000.0,
                "tpot_ms": _tpot_ms(latency_s, ttft_s, actual_output),
                "e2e_ms": latency_s * 1000.0,
                "itl_ms": itl_ms,
                "stream_chunk_tpot_ms": (
                    statistics.fmean(itl_ms) if itl_ms else None
                ),
                "token_timing_valid": len(itls) in {actual_output - 1, actual_output},
                "timing_semantics": "vllm_bench_detailed_seconds",
                "cache_hit_evidence": cache_evidence,
                "shared_prefix_hash": (
                    cache_evidence.get("shared_prefix_hash")
                    if isinstance(cache_evidence, dict)
                    else record.get("shared_prefix_hash")
                ),
                "prefix_tokens": prefix_tokens,
                "shared_prefix": dict(prefix),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "collector": "vllm-bench-prefix-repetition",
        "variant": variant,
        "cell": dict(cell),
        "status": "succeeded",
        "measurements": measurements,
        "evidence": {
            "source": evidence.get("evidence_source", "external_recorder"),
            "request_count": len(measurements),
            "cache_hit_required": require_hit,
        },
    }


def convert_files(
    result_path: Path,
    evidence_path: Path,
    output_path: Path,
    *,
    cell: dict[str, Any],
    variant: str,
    require_hit: bool = True,
) -> dict[str, Any]:
    converted = convert_prefix_repetition(
        _read_json(result_path),
        _read_json(evidence_path),
        cell=cell,
        variant=variant,
        require_hit=require_hit,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(converted, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return converted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="vllm-bench --save-detailed JSON")
    parser.add_argument("--evidence", type=Path, required=True, help="external per-request evidence JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("target", "zero_cost", "replay"), required=True)
    parser.add_argument("--cell", type=Path, required=True, help="JSON file containing one workload cell")
    parser.add_argument("--allow-miss", action="store_true", help="accept non-hit controls; never use for cached fit")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        cell = _read_json(args.cell)
        convert_files(
            args.result,
            args.evidence,
            args.output,
            cell=cell,
            variant=args.variant,
            require_hit=not args.allow_miss,
        )
    except CachedPrefixError as exc:
        print(f"cached-prefix conversion error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
