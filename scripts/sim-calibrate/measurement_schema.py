"""Versioned client evidence contract shared by compilation and collection."""

from __future__ import annotations

import math
import re
from typing import Any

SCHEMA_VERSION = 3
ENVELOPE_KIND = "measurement_envelope"
ROW_KIND = "measurement_row"
PHASES = ("warmup", "bootstrap", "measured")
STATUSES = ("succeeded", "failed", "unsupported")
CONTENT_DIGESTS = ("prompt_sha256", "token_sha256", "body_sha256", "wire_body_sha256")
TIMESTAMPS = ("actual_dispatch_timestamp", "first_token_timestamp", "last_token_timestamp", "stream_end_timestamp")
METRICS = ("ttft_ms", "tpot_ms", "first_text_latency_ms", "e2e_ms")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

DECODE_STATISTICS_POLICY = {
    "source": "fixed_batch_decode_window",
    "window": "full_common_generation_events",
    "interval": "both_endpoints_inside_inclusive",
    "token_denominator": "right_event_output_token_count",
    "qualification": "positive_duration_and_tokens_per_measured_request",
    "aggregation": "request_then_batch_then_repeat_equal_mean",
    "coordinate": "initial_sum_decode_ctx_tokens",
}


REFERENCE_POLICIES = {
    "cold_prefill": {"kind": "cold_zero", "interval_source": "none", "encoding": "integer_us_round_nearest"},
    "steady_decode": {"kind": "target_decode_mean", "interval_source": "qualified_train_target_cell_mean",
                      "encoding": "integer_us_round_nearest"},
}


def reference_policy(kind: str, split: str) -> dict:
    """Validation keeps the existing zero-control diagnostic, never a train interval."""
    if split != 'train':
        return {"kind": "validation_zero", "interval_source": "none", "encoding": "integer_us_round_nearest"}
    return dict(REFERENCE_POLICIES[kind])


class MeasurementError(ValueError):
    """Evidence does not satisfy the declared client measurement contract."""


def digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise MeasurementError(f"{field} must be a lowercase SHA-256 digest")
    return value


def integer(value: Any, field: str, *, positive: bool = False, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise MeasurementError(f"{field} must be a {'positive' if positive else 'non-negative'} integer")
    if maximum is not None and value > maximum:
        raise MeasurementError(f"{field} exceeds {maximum}")
    return value


def number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise MeasurementError(f"{field} must be finite and non-negative")
    return float(value)


def text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MeasurementError(f"{field} must be a non-empty string")
    return value


def contract() -> dict[str, Any]:
    """Compilation declares this contract before a collector can emit rows."""
    return {
        "kind": ENVELOPE_KIND,
        "schema_version": SCHEMA_VERSION,
        "row_kind": ROW_KIND,
        "timestamp_unit": "seconds",
        "timestamp_clock": "monotonic",
        "duration_unit": "milliseconds",
        "phases": list(PHASES),
        "statuses": list(STATUSES),
        "measurement_origin": "http_sse_proxy",
        "text_fallback": "disabled",
        "decode_fit_source": "fixed_batch_decode_window",
        "content_digests": list(CONTENT_DIGESTS),
    }


def validate_row(row: dict[str, Any]) -> None:
    if not isinstance(row, dict):
        raise MeasurementError("measurement row must be an object")
    if row.get("kind") != ROW_KIND or type(row.get("schema_version")) is not int or row["schema_version"] != SCHEMA_VERSION:
        raise MeasurementError("unsupported measurement row kind/schema_version")
    key = text(row.get("pairing_key"), "pairing_key")
    cell_id = digest(row.get("cell_id"), "cell_id")
    digest(row.get("plan_sha256"), "plan_sha256")
    phase = row.get("phase")
    if phase not in PHASES:
        raise MeasurementError("unknown measurement phase")
    repeat = integer(row.get("repeat"), "repeat")
    index = integer(row.get("index"), "index")
    if key != f"{cell_id}:{phase}:{repeat}:{index}":
        raise MeasurementError("pairing_key does not match cell/phase/repeat/index")
    if row.get("status") not in STATUSES:
        raise MeasurementError("unknown measurement status")
    if row.get("variant") not in ("target", "zero_cost", "sim"):
        raise MeasurementError("unknown measurement variant")
    if "transport_request_id" not in row:
        raise MeasurementError("missing transport_request_id (use null when unavailable)")
    if row["transport_request_id"] is not None:
        text(row["transport_request_id"], "transport_request_id")
    if not isinstance(row.get("extensions", {}), dict):
        raise MeasurementError("measurement extensions must be an object")
    if "token_timing_valid" not in row:
        raise MeasurementError("missing token_timing_valid (use null when unproven)")
    if row.get("token_timing_valid") not in (True, False, None) or (
        row.get("token_timing_valid") is not None and not isinstance(row["token_timing_valid"], bool)
    ):
        raise MeasurementError("token_timing_valid must be boolean or null")
    number(row.get("planned_dispatch_offset_ms"), "planned_dispatch_offset_ms")
    for field in (*TIMESTAMPS, "first_text_timestamp", *METRICS):
        if field not in row:
            raise MeasurementError(f"missing measurement field {field}")
        if row[field] is not None:
            number(row[field], field)
    present = [row[field] for field in TIMESTAMPS]
    if all(value is not None for value in present) and present != sorted(present):
        raise MeasurementError("dispatch/first/last timestamps are out of order")
    events = row.get("generation_event_timestamps")
    if not isinstance(events, list):
        raise MeasurementError("generation_event_timestamps must be an array")
    for value in events:
        number(value, "generation event timestamp")
    if events != sorted(events):
        raise MeasurementError("generation events are out of order")
    counts = row.get("interval_token_counts")
    if not isinstance(counts, list):
        raise MeasurementError("interval_token_counts must be an array")
    for count in counts:
        integer(count, "interval token count", positive=True)
    if row.get("token_timing_valid") is True:
        first_count = integer(row.get("first_event_token_count"), "first_event_token_count", positive=True)
        if not events or len(counts) != len(events) - 1:
            raise MeasurementError("validated token timing requires attributable event intervals")
        if first_count + sum(counts) != row.get("completion_tokens"):
            raise MeasurementError("event token counts disagree with completion usage")
        if row.get("first_token_timestamp") != events[0] or row.get("last_token_timestamp") != events[-1]:
            raise MeasurementError("first/last token timestamps differ from generation events")
    if row.get("measurement_origin") != "http_sse_proxy":
        raise MeasurementError("measurement must identify proxy observation origin")
    first_text = row["first_text_timestamp"]
    if (first_text is None) != (row["first_text_latency_ms"] is None):
        raise MeasurementError("text timestamp/latency must both be present or null")
    if first_text is not None:
        start, end = row['actual_dispatch_timestamp'], row['stream_end_timestamp']
        if start is None or first_text < start or (end is not None and first_text > end):
            raise MeasurementError("first text is outside the stream")
        if not math.isclose(row['first_text_latency_ms'], (first_text-start)*1000, abs_tol=1e-5):
            raise MeasurementError("first text latency disagrees with timestamps")
    if events and ((present[0] is not None and events[0] < present[0]) or
                   (present[-1] is not None and events[-1] > present[-1])):
        raise MeasurementError("generation events are outside the stream")
    if row["status"] == "succeeded":
        for field in (*CONTENT_DIGESTS, "workload_evidence_sha256"):
            digest(row.get(field), field)
        text(row.get("request_instance_id"), "request_instance_id")
        text(row.get("transport_request_id"), "transport_request_id")
        if row.get("event_source") != "proxy_upstream_read":
            raise MeasurementError("successful event source must be proxy_upstream_read")
        for field in ("prompt_tokens", "completion_tokens"):
            integer(row.get(field), field, positive=True)
        if any(row[field] is None for field in TIMESTAMPS) or any(row[field] is None for field in ("ttft_ms", "e2e_ms")):
            raise MeasurementError("successful measurement is missing timing evidence")
        if row["token_timing_valid"] is not True or not events or events[0] != present[1] or events[-1] != present[2]:
            raise MeasurementError("successful measurement requires first/last generation events")
        if row.get("error") is not None:
            raise MeasurementError("successful measurement cannot contain an error")
        for metric, elapsed in (("ttft_ms", present[1] - present[0]), ("e2e_ms", present[3] - present[0])):
            if not math.isclose(row[metric], elapsed * 1000, rel_tol=1e-6, abs_tol=1e-5):
                raise MeasurementError(f"{metric} disagrees with proxy timestamps")
        denominator = sum(counts)
        expected_tpot = (events[-1] - events[0]) * 1000 / denominator if denominator else None
        if expected_tpot is None:
            if row['tpot_ms'] is not None:
                raise MeasurementError("no interval tokens for TPOT")
        elif row['tpot_ms'] is None or not math.isclose(row['tpot_ms'], expected_tpot, abs_tol=1e-5):
            raise MeasurementError("TPOT disagrees with right-event token counts")
    else:
        text(row.get("error"), "error")


def validate_envelope(envelope: dict[str, Any]) -> None:
    if not isinstance(envelope, dict) or envelope.get("kind") != ENVELOPE_KIND or type(envelope.get("schema_version")) is not int or envelope["schema_version"] != SCHEMA_VERSION:
        raise MeasurementError("unsupported measurement envelope kind/schema_version")
    digest(envelope.get("plan_sha256"), "plan_sha256")
    if envelope.get("variant") not in ("target", "zero_cost", "sim"):
        raise MeasurementError("unknown measurement envelope variant")
    identity = envelope.get("variant_identity")
    if not isinstance(identity, dict):
        raise MeasurementError("measurement variant_identity must be an object")
    for field in ("endpoint", "model", "engine", "engine_version", "model_revision"):
        text(identity.get(field), f"variant_identity.{field}")
    digest(identity.get("model_config_sha256"), "variant_identity.model_config_sha256")
    provenance = envelope.get("provenance")
    if not isinstance(provenance, dict):
        raise MeasurementError("measurement provenance must be an object")
    for field in ("binary_sha256", "tokenizer_sha256", "observer_sha256", "raw_bundle_index_sha256", "source_command_sha256"):
        digest(provenance.get(field), field)
    for field in ("client_name", "client_version"):
        text(provenance.get(field), field)
    rows = envelope.get("measurements")
    if not isinstance(rows, list) or not rows:
        raise MeasurementError("measurement envelope must contain rows")
    seen: set[str] = set()
    instances: set[str] = set()
    transport_ids: set[str] = set()
    for row in rows:
        validate_row(row)
        if row["plan_sha256"] != envelope["plan_sha256"] or row["variant"] != envelope["variant"]:
            raise MeasurementError("measurement row belongs to a different plan/variant")
        if row["pairing_key"] in seen:
            raise MeasurementError("duplicate measurement pairing_key")
        seen.add(row["pairing_key"])
        for field, identities in (("request_instance_id", instances), ("transport_request_id", transport_ids)):
            value = row.get(field)
            if value is not None:
                if value in identities:
                    raise MeasurementError(f"duplicate measurement {field}")
                identities.add(value)
