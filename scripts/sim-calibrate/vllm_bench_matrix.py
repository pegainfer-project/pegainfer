#!/usr/bin/env python3
"""Deterministic workload matrices expanded from an explicit campaign JSON."""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable, Mapping


class MatrixError(ValueError):
    """Raised when a requested matrix cannot represent valid token shapes."""


def axis_values(value: Any, field: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    """Expand one JSON axis into canonical, sorted integer values.

    An axis is either ``{"values": [...]}`` or an inclusive integer range
    ``{"start": N, "stop": M, "step": S}``.  There are intentionally no
    implicit calibration defaults here: a campaign must describe every axis
    it wants to measure.
    """

    if isinstance(value, Mapping):
        keys = set(value)
        if keys == {"values"}:
            value = value["values"]
        elif keys == {"start", "stop", "step"}:
            start, stop, step = (value[name] for name in ("start", "stop", "step"))
            for name, item in (("start", start), ("stop", stop), ("step", step)):
                if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                    raise MatrixError(f"{field}.{name} must be a positive integer")
            if stop < start:
                raise MatrixError(f"{field}.stop must be >= {field}.start")
            value = list(range(start, stop + 1, step))
        else:
            raise MatrixError(f"{field} must use values or start/stop/step")
    if not isinstance(value, (list, tuple)):
        raise MatrixError(f"{field} must be an array or range object")
    rendered = tuple(value)
    if not rendered and not allow_empty:
        raise MatrixError(f"{field} must contain at least one value")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in rendered):
        raise MatrixError(f"{field} must contain positive integers")
    if len(set(rendered)) != len(rendered):
        raise MatrixError(f"{field} must not contain duplicates")
    return tuple(sorted(rendered))


def _required_int(config: Mapping[str, Any], name: str, field: str, *, allow_zero: bool = False) -> int:
    if name not in config:
        raise MatrixError(f"{field} is required")
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise MatrixError(f"{field} must be a {qualifier} integer")
    return value


def _split(config: Mapping[str, Any], field: str) -> str:
    value = config.get("split", "train")
    if not isinstance(value, str) or not value.strip():
        raise MatrixError(f"{field}.split must be a non-empty string")
    return value.strip()


def _metadata(config: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = config.get("metadata", {})
    if not isinstance(value, dict):
        raise MatrixError(f"{field}.metadata must be an object")
    return dict(value)


def cold_prefill_matrix(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return concurrency-one cold cells from an explicit JSON matrix."""

    if not isinstance(config, Mapping):
        raise MatrixError("cold_prefill matrix must be an object")
    inputs = axis_values(config.get("input_tokens"), "cold_prefill.input_tokens")
    output_tokens = _required_int(config, "output_tokens", "cold_prefill.output_tokens")
    repeats = _required_int(config, "repeats", "cold_prefill.repeats")
    warmup_repeats = _required_int(
        config, "warmup_repeats", "cold_prefill.warmup_repeats", allow_zero=True
    )
    split = _split(config, "cold_prefill")
    required = config.get("required_input_tokens")
    if required is not None:
        required_values = axis_values(required, "cold_prefill.required_input_tokens")
        missing = sorted(set(required_values) - set(inputs))
        if missing:
            raise MatrixError(f"cold_prefill.input_tokens is missing required values: {missing}")
    metadata = _metadata(config, "cold_prefill")
    metadata.update({"matrix": "cold_prefill", "input_axis": list(inputs)})
    name_prefix = config.get("name_prefix", "cold-prefill")
    if not isinstance(name_prefix, str) or not name_prefix.strip():
        raise MatrixError("cold_prefill.name_prefix must be a non-empty string")
    return [
        {
            "name": f"{name_prefix.strip()}-{tokens}",
            "kind": "cold_prefill",
            "split": split,
            "input_tokens": tokens,
            "output_tokens": output_tokens,
            "concurrency": 1,
            "warmup_repeats": warmup_repeats,
            "repeats": repeats,
            "metadata": dict(metadata),
        }
        for tokens in inputs
    ]


def decode_matrix(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build additive coverage plus an explicit surface subset."""

    if not isinstance(config, Mapping):
        raise MatrixError("steady_decode matrix must be an object")
    requests = axis_values(config.get("concurrency"), "steady_decode.concurrency")
    contexts = axis_values(config.get("context_tokens"), "steady_decode.context_tokens")
    surface_request_axis = axis_values(
        config.get("surface_requests"), "steady_decode.surface_requests", allow_empty=True
    )
    surface_kv_axis = axis_values(
        config.get("surface_kv_tokens"), "steady_decode.surface_kv_tokens", allow_empty=True
    )
    if bool(surface_request_axis) != bool(surface_kv_axis):
        raise MatrixError(
            "steady_decode surface_requests and surface_kv_tokens must be both empty or non-empty"
        )
    output_tokens = _required_int(config, "output_tokens", "steady_decode.output_tokens")
    repeats = _required_int(config, "repeats", "steady_decode.repeats")
    warmup_repeats = _required_int(
        config, "warmup_repeats", "steady_decode.warmup_repeats", allow_zero=True
    )
    if output_tokens < 2:
        raise MatrixError("steady_decode.output_tokens must be >=2")
    split = _split(config, "steady_decode")
    base_metadata = _metadata(config, "steady_decode")
    name_prefix = config.get("name_prefix", "decode")
    if not isinstance(name_prefix, str) or not name_prefix.strip():
        raise MatrixError("steady_decode.name_prefix must be a non-empty string")
    cells: list[dict[str, Any]] = []
    # Keep explicit surface cells even when their shape also appears in
    # additive coverage: the families carry different fit intent.
    seen: set[tuple[str, int, int]] = set()

    def add(request_count: int, input_count: int, *, family: str, kv_count: int) -> None:
        key = (family, request_count, input_count)
        if key in seen:
            return
        seen.add(key)
        metadata = {
            **base_metadata,
            "matrix": "steady_decode",
            "family": family,
            "sum_decode_ctx_tokens": kv_count,
        }
        cells.append(
            {
                "name": f"{name_prefix.strip()}-{family}-c{request_count}-ctx{input_count}",
                "kind": "steady_decode",
                "split": split,
                "input_tokens": input_count,
                "output_tokens": output_tokens,
                "concurrency": request_count,
                "sum_decode_ctx_tokens": kv_count,
                "warmup_repeats": warmup_repeats,
                "repeats": repeats,
                "metadata": metadata,
            }
        )

    for request_count, context in product(requests, contexts):
        add(request_count, context, family="additive", kv_count=request_count * context)
    for request_count, kv_count in product(surface_request_axis, surface_kv_axis):
        if kv_count % request_count:
            raise MatrixError(
                "steady_decode surface_kv_tokens must be divisible by every surface request count"
            )
        add(request_count, kv_count // request_count, family="surface", kv_count=kv_count)
    return cells


def decode_surface_axes(cells: Iterable[dict[str, Any]]) -> tuple[list[int], list[int]]:
    """Return axes represented by surface-family cells after completeness checks."""

    surface = [cell for cell in cells if cell.get("metadata", {}).get("family") == "surface"]
    requests = sorted({int(cell["concurrency"]) for cell in surface})
    kv_tokens = sorted({int(cell["sum_decode_ctx_tokens"]) for cell in surface})
    expected = {(request, kv) for request in requests for kv in kv_tokens}
    actual = {(int(cell["concurrency"]), int(cell["sum_decode_ctx_tokens"])) for cell in surface}
    if expected != actual:
        raise MatrixError("surface cells do not form a complete request x total-KV grid")
    return requests, kv_tokens
