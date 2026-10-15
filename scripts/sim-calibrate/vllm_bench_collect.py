#!/usr/bin/env python3
"""Convert vllm-bench detailed output into the paired measurement contract."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from vllm_bench_sidecar import SCHEMA_VERSION, SidecarError, canonical_request_id


class VllmBenchCollectError(ValueError):
    """Raised when detailed output cannot be paired unambiguously."""


REQUIRED_ARRAYS = ("input_lens", "output_lens", "ttfts", "latencies", "itls", "errors")


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VllmBenchCollectError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise VllmBenchCollectError(f"{field} must be finite and non-negative")
    return result


def _positive(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VllmBenchCollectError(f"{field} must be a positive integer")
    return value


def _sidecar_requests(sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    if sidecar.get("schema_version") != SCHEMA_VERSION:
        raise VllmBenchCollectError("unsupported sidecar schema_version")
    requests = sidecar.get("requests")
    if not isinstance(requests, list) or not requests:
        raise VllmBenchCollectError("sidecar.requests must be a non-empty array")
    seen_ids: set[str] = set()
    seen_ordinals: set[int] = set()
    for ordinal, request in enumerate(requests):
        if not isinstance(request, dict):
            raise VllmBenchCollectError(f"sidecar.requests[{ordinal}] must be an object")
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise VllmBenchCollectError(f"sidecar.requests[{ordinal}] has no request_id")
        if request_id in seen_ids:
            raise VllmBenchCollectError(f"duplicate sidecar request_id: {request_id}")
        seen_ids.add(request_id)
        if request.get("ordinal") != ordinal:
            raise VllmBenchCollectError(f"sidecar ordinal mismatch at {ordinal}")
        if ordinal in seen_ordinals:
            raise VllmBenchCollectError(f"duplicate sidecar ordinal: {ordinal}")
        seen_ordinals.add(ordinal)
    return requests


def _array(result: dict[str, Any], field: str, count: int) -> list[Any]:
    values = result.get(field)
    if not isinstance(values, list) or len(values) != count:
        raise VllmBenchCollectError(f"result.{field} must contain {count} entries")
    return values


def _identity_index(
    result: dict[str, Any],
    sidecar_requests: list[dict[str, Any]],
    *,
    concurrency: int,
    sidecar: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve result positions to sidecar rows using explicit identity first."""

    result_ids = result.get("request_ids")
    result_digests = result.get("body_sha256") or result.get("request_body_sha256")
    if result_ids is not None:
        if not isinstance(result_ids, list) or len(result_ids) != len(sidecar_requests):
            raise VllmBenchCollectError("result.request_ids length does not match sidecar")
        by_id = {request["request_id"]: request for request in sidecar_requests}
        resolved: list[dict[str, Any]] = []
        for value in result_ids:
            if value not in by_id:
                raise VllmBenchCollectError(f"result request_id is absent from sidecar: {value!r}")
            resolved.append(by_id[value])
        if len({item["request_id"] for item in resolved}) != len(resolved):
            raise VllmBenchCollectError("result.request_ids contains duplicates")
        return resolved
    if result_digests is not None:
        if not isinstance(result_digests, list) or len(result_digests) != len(sidecar_requests):
            raise VllmBenchCollectError("result body digest array length does not match sidecar")
        by_digest: dict[str, dict[str, Any]] = {}
        for request in sidecar_requests:
            digest = request.get("body_sha256")
            if not isinstance(digest, str) or not digest:
                continue
            if digest in by_digest:
                raise VllmBenchCollectError("duplicate body digest cannot provide unique pairing")
            by_digest[digest] = request
        resolved = []
        for digest in result_digests:
            if digest not in by_digest:
                raise VllmBenchCollectError(f"result body digest is absent from sidecar: {digest!r}")
            resolved.append(by_digest[digest])
        if len({item["request_id"] for item in resolved}) != len(resolved):
            raise VllmBenchCollectError("result body digests contain duplicates")
        return resolved
    ordering = sidecar.get("ordering")
    if ordering != "stable_ordinal":
        raise VllmBenchCollectError(
            "vllm-bench result has no request identity and sidecar does not prove stable ordinal ordering"
        )
    if concurrency > 1 and sidecar.get("body_digest_unique") is False:
        raise VllmBenchCollectError("concurrent sidecar has non-unique body identity")
    return list(sidecar_requests)


def _evidence_by_identity(evidence: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if evidence is None:
        return {}
    records = evidence.get("requests")
    if records is None:
        return {}
    if not isinstance(records, list):
        raise VllmBenchCollectError("evidence.requests must be an array")
    result: dict[str, dict[str, Any]] = {}
    for ordinal, item in enumerate(records):
        if not isinstance(item, dict):
            raise VllmBenchCollectError(f"evidence.requests[{ordinal}] must be an object")
        request_id = item.get("request_id")
        digest = item.get("body_sha256") or item.get("request_body_sha256")
        key = request_id or digest
        if not isinstance(key, str) or not key:
            if item.get("ordinal") != ordinal:
                raise VllmBenchCollectError("evidence without identity must provide stable ordinals")
            key = f"ordinal:{ordinal}"
        if key in result:
            raise VllmBenchCollectError(f"duplicate evidence identity: {key}")
        result[key] = item
    return result


def _tpot_ms(latency_s: float, ttft_s: float, output_tokens: int) -> float | None:
    if output_tokens <= 1:
        return None
    result = (latency_s - ttft_s) * 1000.0 / float(output_tokens - 1)
    if not math.isfinite(result) or result < 0.0:
        raise VllmBenchCollectError("derived tpot_ms is not finite and non-negative")
    return result


def convert_vllm_bench(
    result: dict[str, Any],
    sidecar: dict[str, Any],
    *,
    cell: dict[str, Any],
    variant: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one detailed result while retaining warmup rows for normalize."""

    if variant not in {"target", "zero_cost", "replay"}:
        raise VllmBenchCollectError(f"unsupported variant: {variant}")
    if result.get("backend") not in {"vllm", "openai", "openai-chat"}:
        raise VllmBenchCollectError("result is not from an OpenAI-compatible backend")
    if result.get("failed", 0) != 0:
        raise VllmBenchCollectError("vllm-bench result contains failed requests")
    completed = _positive(result.get("completed"), "result.completed")
    sidecar_rows = _sidecar_requests(sidecar)
    if completed != len(sidecar_rows):
        raise VllmBenchCollectError(
            f"completed request count {completed} does not match sidecar count {len(sidecar_rows)}"
        )
    arrays = {field: _array(result, field, completed) for field in REQUIRED_ARRAYS}
    concurrency = _positive(cell.get("concurrency"), "cell.concurrency")
    rows = _identity_index(result, sidecar_rows, concurrency=concurrency, sidecar=sidecar)
    evidence_by_key = _evidence_by_identity(evidence)
    measurements: list[dict[str, Any]] = []
    for position, sidecar_row in enumerate(rows):
        error = arrays["errors"][position]
        if error not in (None, ""):
            raise VllmBenchCollectError(f"vllm-bench request {position} failed: {error}")
        input_tokens = _positive(arrays["input_lens"][position], f"result.input_lens[{position}]")
        output_tokens = _positive(arrays["output_lens"][position], f"result.output_lens[{position}]")
        expected_input = sidecar_row.get("input_tokens")
        expected_output = sidecar_row.get("output_tokens")
        if expected_input is not None and input_tokens != expected_input:
            raise VllmBenchCollectError(f"input length mismatch at result position {position}")
        if expected_output is not None and output_tokens > expected_output:
            raise VllmBenchCollectError(f"output length exceeds request at result position {position}")
        # OpenAI streaming clients may count only non-empty stream chunks.  A
        # frontend can therefore report fewer observed chunks/tokens than the
        # requested max_tokens even when the request completed successfully
        # (the detailed usage counter is not present in vllm-bench JSON). Keep
        # the timing row, but mark the completion-length contract as unproven;
        # fitting can use stream_chunk_tpot_ms while excluding derived TPOT.
        completion_length_valid = expected_output is None or output_tokens == expected_output
        timing_output_tokens = expected_output if expected_output is not None else output_tokens
        ttft_s = _finite(arrays["ttfts"][position], f"result.ttfts[{position}]")
        latency_s = _finite(arrays["latencies"][position], f"result.latencies[{position}]")
        itls = arrays["itls"][position]
        if not isinstance(itls, list):
            raise VllmBenchCollectError(f"result.itls[{position}] must be an array")
        itl_ms = [_finite(value, f"result.itls[{position}][]") * 1000.0 for value in itls]
        # vllm-bench records stream chunks, not necessarily one entry per
        # generated token.  Long-prefill responses can legally coalesce
        # several tokens into one chunk, so retain TTFT/E2E/derived TPOT while
        # marking the per-token stream as incomplete.
        token_timing_valid = completion_length_valid and len(itls) in {output_tokens - 1, output_tokens}
        evidence_row = None
        for key in (sidecar_row["request_id"], sidecar_row.get("body_sha256"), f"ordinal:{position}"):
            if key and key in evidence_by_key:
                evidence_row = evidence_by_key[key]
                break
        measurement = {
            "request_id": sidecar_row["request_id"],
            "pairing_key": (
                f"{cell['cell_id']}:{sidecar_row['phase']}:{sidecar_row['repeat']}:{sidecar_row['index']}"
            ),
            "observed_request_id": (
                evidence_row.get("request_id") if isinstance(evidence_row, dict) else None
            ),
            "variant": variant,
            "phase": sidecar_row["phase"],
            "repeat": sidecar_row["repeat"],
            "index": sidecar_row["index"],
            "ordinal": sidecar_row["ordinal"],
            "ok": True,
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "observed_completion_tokens": output_tokens,
            "completion_length_valid": completion_length_valid,
            "finish_reason": (
                evidence_row.get("finish_reason") if isinstance(evidence_row, dict) else None
            ),
            "ttft_ms": ttft_s * 1000.0,
            "tpot_ms": (
                _tpot_ms(latency_s, ttft_s, timing_output_tokens)
                if completion_length_valid
                else None
            ),
            "e2e_ms": latency_s * 1000.0,
            "itl_ms": itl_ms,
            "stream_chunk_tpot_ms": statistics.fmean(itl_ms) if itl_ms else None,
            "token_timing_valid": token_timing_valid,
            "itl_count": len(itls),
            "expected_itl_count": timing_output_tokens - 1,
            "timing_semantics": "vllm_bench_detailed_seconds",
            "body_sha256": sidecar_row.get("body_sha256"),
        }
        if isinstance(evidence_row, dict):
            usage = evidence_row.get("usage")
            if usage is not None:
                measurement["usage"] = usage
        measurements.append(measurement)
    return {
        "schema_version": 1,
        "collector": "vllm-bench",
        "variant": variant,
        "cell": dict(cell),
        "status": "succeeded",
        "measurements": measurements,
        "pairing": {
            "identity": "canonical_sidecar_request_id",
            "ordering": sidecar.get("ordering"),
            "evidence_attached": bool(evidence_by_key),
        },
    }


def convert_files(
    result_path: Path,
    sidecar_path: Path,
    output_path: Path,
    *,
    cell: dict[str, Any],
    variant: str,
    evidence_path: Path | None = None,
) -> dict[str, Any]:
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        evidence = (
            json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path is not None else None
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VllmBenchCollectError(f"failed to read vllm-bench inputs: {exc}") from exc
    converted = convert_vllm_bench(result, sidecar, cell=cell, variant=variant, evidence=evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(converted, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return converted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("target", "zero_cost", "replay"), required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        cell = json.loads(args.cell.read_text(encoding="utf-8"))
        convert_files(
            args.result,
            args.sidecar,
            args.output,
            cell=cell,
            variant=args.variant,
            evidence_path=args.evidence,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, VllmBenchCollectError, SidecarError) as exc:
        print(f"vllm-bench collection error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
