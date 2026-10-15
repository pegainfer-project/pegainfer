#!/usr/bin/env python3
"""Canonical request sidecars for vllm-bench detailed results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


class SidecarError(ValueError):
    """Raised when a sidecar cannot provide a unique request identity."""


def canonical_request_id(cell_id: str, phase: str, repeat: int, index: int) -> str:
    if not isinstance(cell_id, str) or not cell_id:
        raise SidecarError("cell_id must be a non-empty string")
    if phase not in {"warmup", "measured"}:
        raise SidecarError(f"unsupported request phase: {phase!r}")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
        raise SidecarError("repeat must be a non-negative integer")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise SidecarError("index must be a non-negative integer")
    return f"{cell_id}:{phase}:{repeat}:{index}"


def _positive(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SidecarError(f"{field} must be a positive integer")
    return value


def _body_digest(value: Any) -> str:
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SidecarRequest:
    request_id: str
    phase: str
    repeat: int
    index: int
    ordinal: int
    input_tokens: int | None
    output_tokens: int | None
    body_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "request_id": self.request_id,
            "pairing_key": f"{self.request_id}",
            "phase": self.phase,
            "repeat": self.repeat,
            "index": self.index,
            "ordinal": self.ordinal,
        }
        if self.input_tokens is not None:
            value["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            value["output_tokens"] = self.output_tokens
        if self.body_sha256 is not None:
            value["body_sha256"] = self.body_sha256
        return value


def build_sidecar(
    cell: dict[str, Any],
    *,
    body_values: Iterable[Any] | None = None,
    ordering: str = "stable_ordinal",
    include_warmups: bool = False,
) -> dict[str, Any]:
    """Create the request index expected by all paired vllm-bench runs.

    ``body_values`` is optional because random/prefix-repetition datasets are
    generated inside vllm-bench.  Custom JSONL callers can supply one value per
    request to retain a body digest without recording the prompt itself.
    """

    cell_id = cell.get("cell_id") or cell.get("name")
    if not isinstance(cell_id, str) or not cell_id:
        raise SidecarError("cell requires cell_id or name")
    concurrency = _positive(cell.get("concurrency"), "cell.concurrency")
    repeats = _positive(cell.get("repeats", 1), "cell.repeats")
    warmups = cell.get("warmup_repeats", 0)
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise SidecarError("cell.warmup_repeats must be a non-negative integer")
    input_tokens = cell.get("input_tokens")
    output_tokens = cell.get("output_tokens")
    if input_tokens is not None:
        input_tokens = _positive(input_tokens, "cell.input_tokens")
    if output_tokens is not None:
        output_tokens = _positive(output_tokens, "cell.output_tokens")
    values = list(body_values) if body_values is not None else []
    expected = (warmups + repeats) * concurrency if include_warmups else repeats * concurrency
    if values and len(values) != expected:
        raise SidecarError(f"body_values must contain {expected} entries, got {len(values)}")
    requests: list[dict[str, Any]] = []
    ordinal = 0
    phases = (("warmup", warmups), ("measured", repeats)) if include_warmups else (("measured", repeats),)
    for phase, count in phases:
        for repeat in range(count):
            for index in range(concurrency):
                digest = _body_digest(values[ordinal]) if values else None
                request = SidecarRequest(
                    request_id=canonical_request_id(cell_id, phase, repeat, index),
                    phase=phase,
                    repeat=repeat,
                    index=index,
                    ordinal=ordinal,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    body_sha256=digest,
                )
                requests.append(request.as_dict())
                ordinal += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "collector": "vllm-bench",
        "cell_id": cell_id,
        "ordering": ordering,
        "requests": requests,
        "request_count": len(requests),
        "warmup_requests": warmups * concurrency,
        "measured_requests": repeats * concurrency,
        "includes_warmups": include_warmups,
        "body_digest_unique": len({item["body_sha256"] for item in requests if item.get("body_sha256")})
        == len([item for item in requests if item.get("body_sha256")]),
    }


def write_sidecar(path: Path, document: dict[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise SidecarError("unsupported sidecar schema_version")
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise SidecarError(f"refusing to overwrite a different sidecar: {path}")
    if not path.exists():
        path.write_text(rendered, encoding="utf-8")
