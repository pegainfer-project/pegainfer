#!/usr/bin/env python3
"""Direct OpenAI-compatible endpoint probing for calibration campaigns."""

from __future__ import annotations

import http.client
import json
import socket
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

TARGET_ENV_PREFIXES = ('PEGAINFER_', 'CUDA_', 'NCCL_', 'VLLM_', 'OMP_', 'MKL_')


def target_runtime_environment(environment: dict[str, str]) -> dict[str, str]:
    """Select runtime configuration for an owned target, excluding credentials."""
    return {key: value for key, value in environment.items()
            if (key.startswith(TARGET_ENV_PREFIXES) or key == 'RUST_LOG')
            and not key.endswith(('_KEY', '_TOKEN', '_SECRET', '_PASSWORD'))}


class ProbeError(RuntimeError):
    """Raised when an endpoint is unavailable or fails the identity contract."""


@dataclass(frozen=True)
class Endpoint:
    scheme: str
    host: str
    port: int
    base_path: str

    @classmethod
    def parse(cls, base_url: str) -> "Endpoint":
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ProbeError("endpoint must be an http(s) URL with a host")
        base_path = parsed.path.rstrip("/")
        if parsed.query or parsed.fragment:
            raise ProbeError("endpoint must not contain a query or fragment")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return cls(parsed.scheme, parsed.hostname, port, base_path)

    def path(self, suffix: str) -> str:
        return f"{self.base_path}/{suffix.lstrip('/')}"


def profile_endpoint(base_url: str) -> Endpoint:
    """The static endpoint contract shared by live and saved profile bindings."""
    try:
        endpoint = Endpoint.parse(base_url)
    except ProbeError as exc:
        raise ValueError(str(exc)) from exc
    if endpoint.host != '127.0.0.1':
        raise ValueError('profile PID binding requires explicit IPv4 127.0.0.1; use that address instead of localhost')
    return endpoint


def profile_listener_matches(endpoint: Endpoint, listener: dict) -> bool:
    return (listener.get('family') == 'IPv4' and listener.get('address') in ('127.0.0.1', '0.0.0.0')
            and listener.get('port') == endpoint.port)


def validate_profile_binding(base_url: str, binding: dict) -> None:
    """Check saved address/listener consistency without consulting /proc."""
    endpoint = profile_endpoint(base_url)
    if binding.get('endpoint') != {'address': endpoint.host, 'port': endpoint.port}:
        raise ValueError('saved profile endpoint differs from configured endpoint')
    listeners = binding.get('listeners')
    if not isinstance(listeners, list) or not listeners or any(
            not isinstance(row, dict) or not profile_listener_matches(endpoint, row) for row in listeners):
        raise ValueError('profile listeners do not attest the configured endpoint')


def requires_cache_reset(plan: dict, variant: str, runtime: dict, batch: dict) -> bool:
    """Keep acquisition and offline receipt requirements on the same lifecycle."""
    if variant != 'target' or runtime.get('cache', {}).get('policy') != 'on':
        return False
    lifecycle = plan.get('cache_lifecycle', {})
    if not lifecycle:
        raise ValueError('target with cache enabled requires explicit reset lifecycle')
    if plan['workload_kind'] == 'cached_prefill':
        if batch['phase'] != 'bootstrap':
            return False
        if lifecycle['reset_before'] == 'cell':
            return batch == plan['batches'][0]
        group = (batch['before_phase'], batch['before_repeat'])
        return batch == next(b for b in plan['batches'] if b['phase'] == 'bootstrap'
                             and (b['before_phase'], b['before_repeat']) == group)
    return lifecycle['reset_before'] == 'repeat' or batch == plan['batches'][0]


def validate_reset_receipt(receipt: dict) -> None:
    """Require the existing HTTP 200 plus explicit JSON success response."""
    try:
        value = json.loads(receipt['body'])
    except (ValueError, TypeError, KeyError):
        value = None
    if receipt.get('status') != 200 or not (value is True or isinstance(value, dict) and value.get('success') is True):
        raise ValueError('target reset did not prove success; cold collection unsupported')


def validate_cache_recovery_receipt(receipt: dict, plan: dict, batch: dict, *,
                                    variant_identity: dict | None = None, launch: dict | None = None) -> dict:
    """Revalidate indexed recovery evidence before its prefix-only bootstrap."""
    method = plan['cache_lifecycle'].get('recovery', 'reset')
    group = {k: batch[k] for k in ('before_phase', 'before_repeat')}
    if receipt.get('method') != method or receipt.get('group') != group:
        raise ValueError('cache recovery method/group differs from the frozen plan')
    start, ready = receipt.get('started_ns'), receipt.get('ready_ns')
    if type(start) is not int or type(ready) is not int or not 0 <= start <= ready:
        raise ValueError('cache recovery has invalid monotonic timestamps')
    identity = variant_identity or receipt.get('variant_identity')
    if receipt.get('variant_identity') != identity or not isinstance(identity, dict):
        raise ValueError('cache recovery belongs to another target')
    readiness = receipt.get('readiness', {})
    if (readiness.get('endpoint') != identity['endpoint']
            or identity['model'] not in readiness.get('models', {}).get('model_ids', [])):
        raise ValueError('cache recovery lacks non-generating model readiness evidence')
    if method == 'reset':
        validate_reset_receipt(receipt.get('reset', {}))
    elif method == 'restart':
        if not launch or receipt.get('launch') != launch:
            raise ValueError('restart differs from the frozen target launch')
        binding = receipt.get('process', {})
        validate_profile_binding(identity['endpoint'], binding)
        if (type(binding.get('server_pid')) is not int or binding['server_pid'] <= 0
                or type(binding.get('process_start_ticks')) is not int
                or binding.get('executable') != str(Path(launch['argv'][0]).resolve())
                or binding.get('argv') != launch['argv']):
            raise ValueError('restart process does not attest the frozen executable/argv')
    else:
        raise ValueError('unsupported cache recovery method')
    return receipt


@dataclass(frozen=True)
class HttpResult:
    status: int
    headers: dict[str, str]
    body: bytes
    elapsed_ms: float

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProbeError(f"HTTP {self.status} returned non-JSON data") from exc


def _connection(endpoint: Endpoint, timeout: float) -> http.client.HTTPConnection:
    if endpoint.scheme == "https":
        return http.client.HTTPSConnection(
            endpoint.host,
            endpoint.port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
    return http.client.HTTPConnection(endpoint.host, endpoint.port, timeout=timeout)


def _get(
    endpoint: Endpoint,
    path: str,
    *,
    timeout: float,
) -> HttpResult:
    headers = {"Accept": "application/json"}
    started = time.perf_counter()
    conn = _connection(endpoint, timeout)
    try:
        conn.request("GET", endpoint.path(path), headers=headers)
        response = conn.getresponse()
        payload = response.read()
        return HttpResult(
            response.status,
            {key.lower(): value for key, value in response.getheaders()},
            payload,
            (time.perf_counter() - started) * 1000.0,
        )
    except (OSError, socket.timeout) as exc:
        raise ProbeError(f"GET {path} failed: {exc}") from exc
    finally:
        conn.close()


def _require_success(result: HttpResult, path: str) -> None:
    if not 200 <= result.status < 300:
        raise ProbeError(
            f"{path} returned HTTP {result.status}"
        )


def probe_endpoint(
    base_url: str,
    *,
    expected_model: str | None = None,
    timeout: float = 5.0,
    attempts: int = 1,
    retry_delay_s: float = 0.25,
) -> dict[str, Any]:
    """Probe readiness and model identity without using ambient proxies."""

    if attempts <= 0:
        raise ProbeError("attempts must be positive")
    endpoint = Endpoint.parse(base_url)
    last_error: ProbeError | None = None
    for attempt in range(attempts):
        try:
            models = _get(endpoint, "/v1/models", timeout=timeout)
            _require_success(models, "/v1/models")
            document = models.json()
            entries = document.get("data") if isinstance(document, dict) else None
            if not isinstance(entries, list):
                raise ProbeError("/v1/models response must contain a data list")
            model_ids = []
            selected: dict[str, Any] | None = None
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                    continue
                model_id = entry["id"]
                model_ids.append(model_id)
                if expected_model is None or model_id == expected_model:
                    selected = {
                        key: entry[key]
                        for key in ("id", "owned_by", "root", "created")
                        if key in entry and isinstance(entry[key], (str, int, float, type(None)))
                    }
                    if expected_model is not None:
                        break
            if expected_model is not None and selected is None:
                raise ProbeError(
                    f"model {expected_model!r} not found in /v1/models: {sorted(model_ids)}"
                )
            return {
                "proxy_mode": "direct_http_client",
                "endpoint": base_url.rstrip("/"),
                "models": {
                    "status": models.status,
                    "elapsed_ms": models.elapsed_ms,
                    "model_ids": model_ids,
                    "selected": selected,
                },
            }
        except ProbeError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(max(0.0, retry_delay_s))
    assert last_error is not None
    raise last_error

