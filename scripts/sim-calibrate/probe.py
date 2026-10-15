#!/usr/bin/env python3
"""Direct OpenAI-compatible endpoint probing for calibration campaigns."""

from __future__ import annotations

import hashlib
import http.client
import json
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


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


def request(
    endpoint: Endpoint,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float,
) -> HttpResult:
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if encoded is not None:
        headers.update({"Content-Type": "application/json", "Content-Length": str(len(encoded))})
    started = time.perf_counter()
    conn = _connection(endpoint, timeout)
    try:
        conn.request(method, endpoint.path(path), body=encoded, headers=headers)
        response = conn.getresponse()
        payload = response.read()
        return HttpResult(
            response.status,
            {key.lower(): value for key, value in response.getheaders()},
            payload,
            (time.perf_counter() - started) * 1000.0,
        )
    except (OSError, socket.timeout) as exc:
        raise ProbeError(f"{method} {path} failed: {exc}") from exc
    finally:
        conn.close()


def _safe_body_digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _require_success(result: HttpResult, path: str) -> None:
    if not 200 <= result.status < 300:
        raise ProbeError(
            f"{path} returned HTTP {result.status}; body_sha256={_safe_body_digest(result.body)}"
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
            health = request(endpoint, "GET", "/health", timeout=timeout)
            _require_success(health, "/health")
            models = request(endpoint, "GET", "/v1/models", timeout=timeout)
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
                "health": {"status": health.status, "elapsed_ms": health.elapsed_ms},
                "models": {
                    "status": models.status,
                    "elapsed_ms": models.elapsed_ms,
                    "model_ids": model_ids,
                    "selected": selected,
                    "body_sha256": _safe_body_digest(models.body),
                },
            }
        except ProbeError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(max(0.0, retry_delay_s))
    assert last_error is not None
    raise last_error

