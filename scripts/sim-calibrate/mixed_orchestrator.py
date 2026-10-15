#!/usr/bin/env python3
"""Build and validate vllm-bench mixed-workload sidecars.

vllm-bench can generate heterogeneous custom requests, but a single finite
run does not prove that a long decode was live while a foreground prefill
arrived.  The default plan therefore uses two externally coordinated
processes and requires explicit interval-overlap evidence before report code
attributes a mixed cell to the scheduler.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from vllm_bench import render_command


class MixedError(ValueError):
    """Raised when a mixed workload cannot establish its phase contract."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _request_id(cell_id: str, role: str, ordinal: int) -> str:
    return f"{cell_id}:mixed:{role}:{ordinal}"


@dataclass(frozen=True)
class MixedRequest:
    request_id: str
    role: str
    prompt: str | list[int]
    output_tokens: int
    ordinal: int

    def jsonl(self) -> dict[str, Any]:
        # vllm-bench custom JSONL intentionally receives only its supported
        # fields; role/identity stay in the sidecar rather than the prompt.
        return {"prompt": self.prompt, "output_tokens": self.output_tokens}

    def sidecar(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "role": self.role,
            "phase": "measured",
            "repeat": 0,
            "index": self.ordinal,
            "ordinal": self.ordinal,
            "input_tokens": len(self.prompt) if isinstance(self.prompt, list) else None,
            "output_tokens": self.output_tokens,
            "body_sha256": _digest(self.jsonl()),
        }


def build_mixed_workload(
    cell_id: str,
    *,
    background_prompts: Iterable[str | list[int]],
    foreground_prompts: Iterable[str | list[int]],
    background_output_tokens: int,
    foreground_output_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return custom JSONL rows and a sidecar with disjoint role identities."""

    if not isinstance(cell_id, str) or not cell_id:
        raise MixedError("cell_id must be a non-empty string")
    if background_output_tokens <= 1 or foreground_output_tokens <= 0:
        raise MixedError("background output must be >=2 and foreground output positive")
    requests: list[MixedRequest] = []
    ordinal = 0
    for prompt in background_prompts:
        if not _valid_prompt(prompt):
            raise MixedError("background prompts must be non-empty strings or token-id arrays")
        requests.append(MixedRequest(_request_id(cell_id, "background_decode", ordinal), "background_decode", prompt, background_output_tokens, ordinal))
        ordinal += 1
    foreground_start = ordinal
    for prompt in foreground_prompts:
        if not _valid_prompt(prompt):
            raise MixedError("foreground prompts must be non-empty strings or token-id arrays")
        requests.append(MixedRequest(_request_id(cell_id, "foreground_prefill", ordinal), "foreground_prefill", prompt, foreground_output_tokens, ordinal))
        ordinal += 1
    if not requests or foreground_start == ordinal:
        raise MixedError("mixed workload requires both background and foreground requests")
    return (
        [request.jsonl() for request in requests],
        {
            "schema_version": 1,
            "collector": "vllm-bench",
            "cell_id": cell_id,
            "ordering": "stable_ordinal",
            "roles": {"background": "background_decode", "foreground": "foreground_prefill"},
            "requests": [request.sidecar() for request in requests],
            "request_count": len(requests),
            "body_digest_unique": len({request.sidecar()["body_sha256"] for request in requests}) == len(requests),
            "overlap_required": True,
        },
    )


def _valid_prompt(prompt: Any) -> bool:
    return bool(
        (isinstance(prompt, str) and prompt)
        or (
            isinstance(prompt, list)
            and bool(prompt)
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in prompt)
        )
    )


def write_custom_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows)
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise MixedError(f"refusing to overwrite a different mixed JSONL file: {path}")
    if not path.exists():
        path.write_text(rendered, encoding="utf-8")


def merge_sidecars(*sidecars: dict[str, Any]) -> dict[str, Any]:
    """Merge independent process sidecars while rejecting identity collisions."""

    if not sidecars:
        raise MixedError("at least one sidecar is required")
    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sidecar in sidecars:
        if sidecar.get("schema_version") != 1:
            raise MixedError("unsupported mixed sidecar schema_version")
        for request in sidecar.get("requests", []):
            request_id = request.get("request_id") if isinstance(request, dict) else None
            if not isinstance(request_id, str) or not request_id:
                raise MixedError("mixed sidecar request has no request_id")
            if request_id in seen:
                raise MixedError(f"mixed sidecar request identity collision: {request_id}")
            seen.add(request_id)
            requests.append(dict(request))
    for ordinal, request in enumerate(requests):
        request["ordinal"] = ordinal
        request["index"] = ordinal
    return {
        "schema_version": 1,
        "collector": "vllm-bench",
        "cell_id": sidecars[0].get("cell_id"),
        "ordering": "stable_ordinal",
        "requests": requests,
        "request_count": len(requests),
        "body_digest_unique": len({request.get("body_sha256") for request in requests}) == len(requests),
        "overlap_required": True,
    }


def two_process_commands(
    background_cell: dict[str, Any],
    foreground_cell: dict[str, Any],
    *,
    binary: Path,
    endpoint: str,
    model: str,
    result_dir: Path,
    background_workload: Path,
    foreground_workload: Path,
    seed: int = 0,
    tokenizer: str | None = None,
) -> dict[str, list[str]]:
    """Render the two commands that an external runner starts with a barrier."""

    return {
        "background": render_command(
            {**background_cell, "kind": "mixed"},
            binary=binary,
            endpoint=endpoint,
            model=model,
            result_dir=result_dir,
            result_filename="mixed-background.json",
            seed=seed,
            tokenizer=tokenizer,
            workload_path=background_workload,
        ),
        "foreground": render_command(
            {**foreground_cell, "kind": "mixed"},
            binary=binary,
            endpoint=endpoint,
            model=model,
            result_dir=result_dir,
            result_filename="mixed-foreground.json",
            seed=seed + 1,
            tokenizer=tokenizer,
            workload_path=foreground_workload,
        ),
    }


def validate_overlap(evidence: dict[str, Any]) -> dict[str, Any]:
    """Validate explicit background/foreground interval overlap evidence."""

    records = evidence.get("requests") if isinstance(evidence, dict) else None
    if not isinstance(records, list) or not records:
        raise MixedError("mixed overlap evidence.requests must be non-empty")
    clock_domains = {record.get("clock_domain") for record in records if isinstance(record, dict)}
    if len(clock_domains) != 1 or None in clock_domains:
        raise MixedError("mixed overlap evidence must use one explicit clock domain")
    intervals: dict[str, list[tuple[float, float]]] = {"background_decode": [], "foreground_prefill": []}
    for record in records:
        if not isinstance(record, dict) or record.get("role") not in intervals:
            raise MixedError("mixed overlap evidence has an unsupported role")
        started = record.get("started_at_unix_s")
        finished = record.get("finished_at_unix_s")
        if not isinstance(started, (int, float)) or not isinstance(finished, (int, float)) or finished <= started:
            raise MixedError("mixed overlap evidence contains an invalid interval")
        intervals[record["role"]].append((float(started), float(finished)))
    overlaps = []
    for background in intervals["background_decode"]:
        for foreground in intervals["foreground_prefill"]:
            start = max(background[0], foreground[0])
            end = min(background[1], foreground[1])
            if end > start:
                overlaps.append({"start": start, "end": end, "duration_s": end - start})
    if not overlaps:
        raise MixedError("mixed workload has no background/foreground interval overlap")
    return {
        "status": "validated",
        "clock_domain": next(iter(clock_domains)),
        "overlap_count": len(overlaps),
        "max_overlap_s": max(item["duration_s"] for item in overlaps),
    }


def mark_overlap_metadata(cell: dict[str, Any], overlap: dict[str, Any]) -> dict[str, Any]:
    if overlap.get("status") != "validated":
        raise MixedError("only validated overlap evidence can mark a mixed cell")
    result = dict(cell)
    metadata = dict(result.get("metadata") or {})
    metadata.update({"requires_overlap": True, "overlap_status": "validated", "overlap": dict(overlap)})
    result["metadata"] = metadata
    return result
