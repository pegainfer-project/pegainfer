#!/usr/bin/env python3
"""Run one manifest cell with streaming per-request timing evidence."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from collect import _prompt_tokens, _request_id, _shared_prefix_hash
from manifest import ManifestError
from metrics import summarize_measurements
from probe import Endpoint
from streaming import StreamError, prefix_cache_metrics, stream_completion


def _cache_request_evidence(
    raw: dict[str, Any] | None,
    cell: dict[str, Any],
    prompt: list[int],
) -> dict[str, Any] | None:
    if not cell["cache_evidence_required"]:
        return raw
    shared_prefix = cell.get("shared_prefix") or {}
    prefix_tokens = int(cell.get("prefix_tokens", 0))
    evidence = dict(raw or {})
    evidence["shared_prefix_id"] = shared_prefix.get("identity")
    evidence["shared_prefix_hash"] = _shared_prefix_hash(prompt, prefix_tokens)
    evidence["expected_cached_tokens"] = prefix_tokens
    evidence["duration_includes_uncached_tail"] = True
    return evidence


def _metrics_cache_evidence(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    *,
    endpoint_isolated: bool,
) -> dict[str, Any] | None:
    if before is None or after is None:
        return None
    queries_delta = after["queries_total"] - before["queries_total"]
    hits_delta = after["hits_total"] - before["hits_total"]
    if queries_delta < 0.0 or hits_delta < 0.0:
        return None
    return {
        "source": f"prometheus.{after.get('metric_schema', 'unknown')}.delta",
        "attribution": "isolated_endpoint_delta" if endpoint_isolated else "endpoint_global_delta",
        "cached_tokens": int(hits_delta),
        "cache_hit": hits_delta > 0.0,
        "prefix_cache_hit": hits_delta > 0.0,
        "prefix_cache_queries_delta": int(queries_delta),
        "prefix_cache_hits_delta": int(hits_delta),
    }


def _cache_evidence_failure(evidence: dict[str, Any] | None, variant: str) -> str | None:
    if variant != "target":
        return None
    cached_tokens = evidence.get("cached_tokens") if isinstance(evidence, dict) else None
    cache_hit = evidence.get("cache_hit") if isinstance(evidence, dict) else None
    prefix_hit = evidence.get("prefix_cache_hit") if isinstance(evidence, dict) else None
    if evidence is None or cached_tokens is None or cached_tokens <= 0 or cache_hit is False or prefix_hit is False:
        return "cached-prefill response did not provide positive hit evidence"
    if evidence.get("attribution") == "endpoint_global_delta":
        return "endpoint-global prefix-cache metrics require runtime.cache.endpoint_isolated=true"
    return None


def _one_stream_request(
    endpoint: Endpoint,
    manifest: dict[str, Any],
    variant: str,
    cell: dict[str, Any],
    *,
    phase: str,
    repeat: int,
    index: int,
    timeout: float,
) -> dict[str, Any]:
    request_id = _request_id(cell["cell_id"], phase, repeat, index)
    shared_prefix = cell.get("shared_prefix") or {}
    prefix_tokens = int(cell.get("prefix_tokens", 0))
    # Keep the declared prefix stable, but make each request's suffix unique.
    # Otherwise repeated cells can cache the entire prompt and over-report the
    # requested shared-prefix length in vLLM's cumulative hit counter.
    prompt_index = index + (repeat + (1 if phase == "measured" else 0)) * cell["concurrency"]
    prompt = _prompt_tokens(
        cell["cell_id"],
        prompt_index,
        cell["input_tokens"],
        prefix_tokens=prefix_tokens,
        prefix_identity=shared_prefix.get("identity"),
    )
    shared_prefix_hash = _shared_prefix_hash(prompt, prefix_tokens)
    variant_info = manifest["variants"][variant]
    model = variant_info["identity"]["model"]
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": cell["output_tokens"],
        "temperature": cell["sampling"].get("temperature", 0.0),
        "top_p": cell["sampling"].get("top_p", 1.0),
        "ignore_eos": cell["ignore_eos"],
        "seed": manifest["client"]["seed"],
        "request_id": request_id,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.time()
    prompt_hash = hashlib.sha256(json.dumps(prompt).encode("utf-8")).hexdigest()
    try:
        metrics_before = None
        cache_runtime = (manifest.get("runtime") or {}).get("cache") or {}
        endpoint_isolated = cache_runtime.get("endpoint_isolated") is True
        if variant == "target" and cell["cache_evidence_required"]:
            metrics_before = prefix_cache_metrics(endpoint, timeout=min(timeout, 5.0))
        capture = stream_completion(endpoint, body, timeout=timeout)
        metrics_after = None
        if variant == "target" and cell["cache_evidence_required"]:
            metrics_after = prefix_cache_metrics(endpoint, timeout=min(timeout, 5.0))
        usage = capture.usage
        if usage is None:
            raise StreamError("stream did not include usage evidence")
        expected = {
            "prompt_tokens": len(prompt),
            "completion_tokens": cell["output_tokens"],
            "total_tokens": len(prompt) + cell["output_tokens"],
        }
        if usage != expected:
            raise StreamError(f"usage mismatch: expected {expected}, got {usage}")
        if cell["ignore_eos"] and capture.finish_reason != "length":
            raise StreamError(f"finish_reason must be length, got {capture.finish_reason!r}")
        inter = capture.inter_event_ms
        token_valid = capture.token_timing_valid is True
        tpot = (sum(inter) / len(inter)) if token_valid and inter else None
        e2e = capture.elapsed_ms
        raw_evidence = capture.cache_hit_evidence or _metrics_cache_evidence(
            metrics_before,
            metrics_after,
            endpoint_isolated=endpoint_isolated,
        )
        evidence = _cache_request_evidence(raw_evidence, cell, prompt)
        measurement = {
            "request_id": request_id,
            "pairing_key": f"{cell['cell_id']}:{phase}:{repeat}:{index}",
            "variant": variant,
            "phase": phase,
            "repeat": repeat,
            "index": index,
            "ok": True,
            "status": capture.status,
            "started_at_unix_s": started,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "finish_reason": capture.finish_reason,
            "ttft_ms": capture.ttft_ms,
            "tpot_ms": tpot,
            "e2e_ms": e2e,
            "itl_ms": inter if token_valid else [],
            "stream_chunk_tpot_ms": (sum(inter) / len(inter)) if inter else None,
            "stream_chunk_count": capture.event_count,
            "token_timing_valid": capture.token_timing_valid,
            "timing_semantics": capture.timing_semantics,
            "prompt_token_hash": prompt_hash,
            "response_body_sha256": capture.body_sha256,
            "cache_hit_evidence": evidence,
            "shared_prefix_hash": shared_prefix_hash,
        }
        if cell["cache_evidence_required"] and phase == "measured":
            failure = _cache_evidence_failure(evidence, variant)
            if failure is not None:
                measurement.update(
                    ok=False,
                    error=failure,
                    cache_evidence_status="unsupported",
                )
        return measurement
    except (StreamError, ValueError) as exc:
        return {
            "request_id": request_id,
            "pairing_key": f"{cell['cell_id']}:{phase}:{repeat}:{index}",
            "variant": variant,
            "phase": phase,
            "repeat": repeat,
            "index": index,
            "ok": False,
            "status": None,
            "started_at_unix_s": started,
            "prompt_tokens": len(prompt),
            "expected_output_tokens": cell["output_tokens"],
            "timed_out": isinstance(exc, StreamError) and exc.timed_out,
            "prompt_token_hash": prompt_hash,
            "error": str(exc)[:500],
            "cache_evidence_status": "invalid" if cell["cache_evidence_required"] and variant == "target" else "not_required",
        }


def run_stream_cell(
    manifest: dict[str, Any],
    cell: dict[str, Any],
    *,
    variant: str = "target",
    timeout: float = 30.0,
) -> dict[str, Any]:
    if manifest.get("schema_version") not in (2, 3):
        raise ManifestError("streaming benchmark cells require paired manifest schema_version=2 or 3")
    if variant not in manifest.get("variants", {}):
        raise ManifestError(f"unknown campaign variant: {variant}")
    endpoint = Endpoint.parse(manifest["variants"][variant]["identity"]["endpoint"])
    measurements: list[dict[str, Any]] = []
    for phase, count in (("warmup", cell["warmup_repeats"]), ("measured", cell["repeats"])):
        for repeat in range(count):
            futures = []
            with ThreadPoolExecutor(max_workers=cell["concurrency"]) as executor:
                for index in range(cell["concurrency"]):
                    request_rate = cell.get("request_rate")
                    if request_rate and index:
                        time.sleep(1.0 / float(request_rate))
                    futures.append(
                        executor.submit(
                            _one_stream_request,
                            endpoint,
                            manifest,
                            variant,
                            cell,
                            phase=phase,
                            repeat=repeat,
                            index=index,
                            timeout=timeout,
                        )
                    )
                batch = [future.result() for future in as_completed(futures)]
            batch.sort(key=lambda item: item["index"])
            measurements.extend(batch)
            if any(not item["ok"] for item in batch):
                return {
                    "cell_id": cell["cell_id"],
                    "name": cell["name"],
                    "variant": variant,
                    "status": "invalid",
                    "measurements": measurements,
                    "summary": summarize_measurements(measurements),
                    "fit_policy": cell["fit_policy"],
                    "failures": [item for item in measurements if not item["ok"]],
                }
    return {
        "cell_id": cell["cell_id"],
        "name": cell["name"],
        "variant": variant,
        "status": "succeeded",
        "measurements": measurements,
        "summary": summarize_measurements(measurements),
        "fit_policy": cell["fit_policy"],
        "failures": [],
    }
