#!/usr/bin/env python3
"""Materialize one selected functional simulator profile from calibration artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from manifest import (  # noqa: E402
    ManifestError,
    canonical_cache_identity,
    canonical_client_identity,
    canonical_scheduler_identity,
    canonical_workload_identity,
)


class ProfileError(ValueError):
    """Raised when calibration evidence cannot produce a safe profile."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mixed_model(mixed_validation: dict[str, Any]) -> dict[str, Any] | None:
    """Materialize only the step-observable mixed composition base."""

    if not isinstance(mixed_validation, dict):
        return None
    if mixed_validation.get("status") != "measured":
        return None
    mixed_errors = mixed_validation.get("errors") or {}
    if not isinstance(mixed_errors.get("absolute_ms"), dict) or mixed_errors.get("count", 0) <= 0:
        return None
    if mixed_validation.get("ood") or mixed_validation.get("uncertain"):
        return None
    return {
        "kind": "composition_additive",
        "intercept_us": 0,
    }


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise ProfileError(f"failed to read {path}: {exc}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    value, _ = _load_json_with_digest(path)
    return value


def _load_json_with_digest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProfileError(f"{path} must contain a JSON object")
    return value, _sha256_bytes(payload)


def _select_artifact(
    campaign_root: Path,
    directory: str,
    pattern: str,
    explicit: Path | None,
    label: str,
) -> Path:
    root = campaign_root.resolve()
    if explicit is not None:
        candidate = explicit if explicit.is_absolute() else root / explicit
        candidate = candidate.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ProfileError(f"{label} path escapes campaign root: {candidate}") from exc
        if not candidate.is_file():
            raise ProfileError(f"{label} artifact does not exist: {candidate}")
        return candidate
    candidates = sorted((root / directory).glob(pattern))
    if not candidates:
        raise ProfileError(f"campaign must contain {directory}/{pattern}")
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates)
        raise ProfileError(f"ambiguous {label} artifacts; specify one explicitly: {names}")
    return candidates[0].resolve()


def _validate_artifact_lineage(
    campaign_root: Path,
    campaign: dict[str, Any],
    normalized_path: Path,
    fit_path: Path,
    report_path: Path,
) -> dict[str, str]:
    manifest_path = campaign_root / "manifest.json"
    manifest, manifest_digest = _load_json_with_digest(manifest_path)
    normalized, normalized_digest = _load_json_with_digest(normalized_path)
    fit, fit_digest = _load_json_with_digest(fit_path)
    report, report_digest = _load_json_with_digest(report_path)
    campaign_id = campaign.get("campaign_id")
    if manifest.get("campaign_id") != campaign_id:
        raise ProfileError("campaign manifest identity changed while materializing profile")
    for label, artifact in (("normalized", normalized), ("fit", fit), ("report", report)):
        if artifact.get("campaign_id") != campaign_id:
            raise ProfileError(f"{label} campaign_id does not match manifest")
        if artifact.get("status") != "succeeded":
            raise ProfileError(f"{label} artifact is not succeeded")
    normalized_source = normalized.get("source")
    source_digests = normalized_source.get("source_digests") if isinstance(normalized_source, dict) else None
    if not isinstance(source_digests, dict) or source_digests.get("manifest_sha256") != manifest_digest:
        raise ProfileError("normalized artifact is not bound to the campaign manifest")
    fit_source = fit.get("source")
    if not isinstance(fit_source, dict) or fit_source.get("normalized_sha256") != normalized_digest:
        raise ProfileError("fit artifact normalized digest does not match bytes")
    report_provenance = report.get("provenance")
    if not isinstance(report_provenance, dict):
        raise ProfileError("report artifact provenance is missing")
    if report_provenance.get("normalized_sha256") != normalized_digest:
        raise ProfileError("report normalized digest does not match bytes")
    if report_provenance.get("fit_sha256") != fit_digest:
        raise ProfileError("report fit digest does not match bytes")
    if report_provenance.get("manifest_sha256") != manifest_digest:
        raise ProfileError("report manifest digest does not match bytes")
    normalized_index_digest = source_digests.get("artifact_index_sha256")
    report_index_digest = report_provenance.get("artifact_index_sha256")
    if normalized_index_digest != report_index_digest:
        raise ProfileError("report artifact index digest does not match normalized lineage")
    return {
        "manifest_sha256": manifest_digest,
        "normalized_sha256": normalized_digest,
        "fit_sha256": fit_digest,
        "report_sha256": report_digest,
    }


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ProfileError(f"{context} is missing {key}")
    return mapping[key]


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProfileError(f"{context} must be a finite number")
    return float(value)


def _ms_to_us(value: Any, context: str) -> int:
    milliseconds = _finite_number(value, context)
    if milliseconds < 0:
        raise ProfileError(f"{context} must be non-negative")
    return int(round(milliseconds * 1000.0))


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProfileError(f"{context} must be a positive integer")
    return value


def _comparison_evidence(comparison: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    # Sealed and held-out evidence is reserved for validation/signoff, never selection.
    for split in ("threshold_validation", "train"):
        record = comparison.get("by_split", {}).get(split)
        if not isinstance(record, dict):
            continue
        if record.get("status") != "measured" or int(record.get("sample_count", 0)) <= 0:
            continue
        if record.get("ood"):
            continue
        errors = record.get("errors")
        if not isinstance(errors, dict):
            continue
        relative = errors.get("relative")
        if not isinstance(relative, dict) or relative.get("p95") is None:
            continue
        by_cell = record.get("by_cell")
        if isinstance(by_cell, dict):
            # Reports may retain extra train observations outside a candidate's
            # fitted domain for OOD characterization.  The candidate's source
            # cells are the contract for profile selection: every fitted point
            # must be measured, while unrelated report cells must not veto it.
            provenance = comparison.get("candidate_provenance")
            candidate_provenance = (
                provenance.get(comparison.get("name"))
                if isinstance(provenance, dict)
                else None
            )
            source_cells = (
                candidate_provenance.get("source_cells")
                if isinstance(candidate_provenance, dict)
                else None
            )
            if isinstance(source_cells, list) and source_cells:
                selected_cells = [by_cell.get(str(cell_id)) for cell_id in source_cells]
                if any(not isinstance(item, dict) for item in selected_cells):
                    continue
            else:
                selected_cells = list(by_cell.values())
            measured_cells = [
                item
                for item in selected_cells
                if isinstance(item, dict) and item.get("status") == "measured"
            ]
            if not measured_cells or len(measured_cells) != len(selected_cells):
                # Aggregate errors cannot hide an unsupported fitted point.
                continue
            cell_p95 = [
                item.get("errors", {}).get("relative", {}).get("p95")
                for item in measured_cells
            ]
            if any(value is None for value in cell_p95):
                continue
            fidelity = {
                "cell_count": len(measured_cells),
                "worst_cell_relative_p95": max(float(value) for value in cell_p95),
                "selection_metric": "worst_cell_relative_p95",
            }
        else:
            # Keep old reports readable, but new reports always carry by_cell.
            fidelity = {
                "cell_count": int(record.get("sample_count", 0)),
                "worst_cell_relative_p95": float(relative["p95"]),
                "selection_metric": "aggregate_relative_p95_legacy",
            }
        return split, record, fidelity
    return None


def _select_comparison(
    report: dict[str, Any],
    names: tuple[str, ...],
    *,
    allow_characterization: bool,
) -> tuple[dict[str, Any], str, bool, dict[str, Any]]:
    candidates = [
        item
        for item in report.get("comparisons", [])
        if isinstance(item, dict) and item.get("name") in names and item.get("status") == "fitted"
    ]
    scored: list[tuple[tuple[float, float, float, int], dict[str, Any], str, bool, dict[str, Any]]] = []
    complexity = {name: index for index, name in enumerate(names)}
    for candidate in candidates:
        evidence = _comparison_evidence(candidate)
        if evidence is None:
            continue
        split, record, fidelity = evidence
        if split == "train" and not allow_characterization:
            continue
        relative = record["errors"]["relative"]
        scored.append(
            (
                (
                    float(fidelity["worst_cell_relative_p95"]),
                    float(relative["p95"]),
                    float(relative.get("p50") if relative.get("p50") is not None else float("inf")),
                    complexity.get(candidate["name"], len(names)),
                ),
                candidate,
                split,
                split == "train",
                fidelity,
            )
        )
    if not scored:
        mode = "characterization" if allow_characterization else "held-out"
        raise ProfileError(
            f"no usable {mode} evidence for candidates {', '.join(names)}"
        )
    _, candidate, split, characterization, fidelity = min(scored, key=lambda item: item[0])
    return candidate, split, characterization, fidelity


def _fit_candidate(fit: dict[str, Any], name: str) -> dict[str, Any]:
    for candidate in fit.get("candidates", []):
        if isinstance(candidate, dict) and candidate.get("name") == name:
            if candidate.get("status") != "fitted":
                raise ProfileError(f"fit candidate {name} is not fitted")
            return candidate
    raise ProfileError(f"fit artifact does not contain fitted candidate {name}")


def _prefill_model(candidate: dict[str, Any]) -> dict[str, Any]:
    knots = candidate.get("knots")
    if not isinstance(knots, list) or not knots:
        raise ProfileError("cold prefill candidate has no knots")
    inputs: list[int] = []
    durations: list[int] = []
    for index, knot in enumerate(knots):
        if not isinstance(knot, dict):
            raise ProfileError(f"cold prefill knot {index} is not an object")
        x = _finite_number(_required(knot, "x", f"cold prefill knot {index}"), f"knot {index}.x")
        if x < 0 or int(x) != x:
            raise ProfileError(f"cold prefill knot {index}.x must be a non-negative integer")
        inputs.append(int(x))
        durations.append(_ms_to_us(_required(knot, "y", f"cold prefill knot {index}"), f"knot {index}.y"))
    return {"input_tokens": inputs, "duration_us": durations}


def _decode_model(candidate: dict[str, Any]) -> dict[str, Any]:
    kind = candidate.get("name")
    coverage = _required(candidate, "coverage", f"decode candidate {kind}")
    requests = coverage.get("requests")
    kv_tokens = coverage.get("kv_tokens")
    if not isinstance(requests, dict) or not isinstance(kv_tokens, dict):
        raise ProfileError(f"decode candidate {kind} has incomplete coverage")
    request_min = int(_finite_number(requests["min"], "decode requests.min"))
    request_max = int(_finite_number(requests["max"], "decode requests.max"))
    kv_min = int(_finite_number(kv_tokens["min"], "decode kv_tokens.min"))
    kv_max = int(_finite_number(kv_tokens["max"], "decode kv_tokens.max"))
    if kind == "decode_2d_surface":
        axes = _required(candidate, "axes", "decode surface")
        values = _required(candidate, "values_ms", "decode surface")
        request_axis = [int(_finite_number(value, "decode requests axis")) for value in axes["requests"]]
        kv_axis = [int(_finite_number(value, "decode kv axis")) for value in axes["kv_tokens"]]
        duration_us = [
            [_ms_to_us(value, "decode surface duration") for value in row]
            for row in values
        ]
    elif kind == "decode_additive":
        coefficients = _required(candidate, "coefficients", "decode additive")
        d0 = _finite_number(coefficients["d0_ms"], "decode d0_ms")
        d_kv = _finite_number(coefficients["d_kv_ms_per_token"], "decode d_kv_ms_per_token")
        d_requests = _finite_number(coefficients["d_requests_ms"], "decode d_requests_ms")
        request_axis = [request_min, request_max]
        kv_axis = [kv_min, kv_max]
        duration_us = [
            [
                _ms_to_us(
                    d0 + d_requests * (request_count - 1) + d_kv * kv_count,
                    "decode additive duration",
                )
                for kv_count in kv_axis
            ]
            for request_count in request_axis
        ]
    else:
        raise ProfileError(f"unsupported decode candidate {kind}")
    if len(request_axis) < 2 or len(kv_axis) < 2:
        raise ProfileError(f"decode candidate {kind} needs at least two points per axis")
    if len(duration_us) != len(request_axis) or any(len(row) != len(kv_axis) for row in duration_us):
        raise ProfileError(f"decode candidate {kind} has inconsistent surface dimensions")
    return {
        "decode_reqs": request_axis,
        "sum_decode_ctx_tokens": kv_axis,
        "duration_us": [value for row in duration_us for value in row],
    }


def _rust_manifest(
    campaign: dict[str, Any],
    source_artifacts: list[str],
    source_digest: str,
    tokenizer_config_sha256: str,
    calibration_driver: str,
) -> dict[str, Any]:
    target = _required(campaign, "target", "campaign")
    runtime = _required(campaign, "runtime", "campaign")
    scheduler = _required(runtime, "scheduler", "campaign.runtime")
    client = _required(campaign, "client", "campaign")
    provenance = _required(campaign, "provenance", "campaign")
    environment = campaign.get("environment") or {}
    cache_runtime = runtime.get("cache") or {}
    try:
        scheduler_identity = canonical_scheduler_identity(scheduler)
        cache_identity = canonical_cache_identity(cache_runtime)
        client_identity = canonical_client_identity(
            client,
            tokenizer_config_sha256=tokenizer_config_sha256,
        )
        workload_identity, split_identity = canonical_workload_identity(campaign)
    except ManifestError as exc:
        raise ProfileError(str(exc)) from exc
    return {
        "schema_version": 2,
        "target_engine": _required(target, "engine", "campaign.target"),
        "engine_version": _required(target, "engine_version", "campaign.target"),
        "model_id": _required(target, "model", "campaign.target"),
        "model_revision": _required(target, "model_revision", "campaign.target"),
        "model_config_sha256": _required(target, "model_config_sha256", "campaign.target"),
        "gpu": environment.get("cuda_visible_devices") or "unbound",
        "server_flags": _required(client, "server_flags", "campaign.client"),
        "collection_command": _required(provenance, "collection_command", "campaign.provenance"),
        "source_artifacts": source_artifacts,
        "backend": runtime.get("backend") or "unknown",
        "driver": environment.get("driver") or "unknown",
        "cuda": environment.get("cuda") or "unknown",
        "parallelism": runtime.get("parallelism") or "unknown",
        "quantization": runtime.get("quantization") or "none",
        "scheduler": scheduler_identity,
        "cache": cache_identity,
        "client": client_identity,
        "workload": workload_identity,
        "split": split_identity,
        "provenance": {
            "collector_revision": _required(provenance, "collector_revision", "campaign.provenance"),
            "fitter_revision": "sim-calibrate-fit-schema-1",
            "artifact_uri": _required(provenance, "artifact_uri", "campaign.provenance"),
            "source_digest": source_digest,
        },
    }


def materialize_profile(
    campaign_root: Path,
    output_root: Path,
    *,
    tokenizer_config: Path,
    allow_characterization: bool = False,
    normalized_artifact: Path | None = None,
    fit_artifact: Path | None = None,
    report_artifact: Path | None = None,
) -> dict[str, Any]:
    campaign_root = campaign_root.expanduser().resolve()
    campaign = _load_json(campaign_root / "manifest.json")
    normalized_path = _select_artifact(
        campaign_root,
        ".",
        "normalized.json",
        normalized_artifact,
        "normalized",
    )
    fit_path = _select_artifact(campaign_root, "fit", "candidates-*.json", fit_artifact, "fit")
    report_path = _select_artifact(campaign_root, "report", "comparison-*.json", report_artifact, "report")
    lineage = _validate_artifact_lineage(campaign_root, campaign, normalized_path, fit_path, report_path)
    normalized = _load_json(normalized_path)
    fit = _load_json(fit_path)
    report = _load_json(report_path)
    tokenizer_digest = _sha256_file(tokenizer_config)
    prefill_comparison, prefill_split, prefill_characterization, prefill_fidelity = _select_comparison(
        report,
        ("cold_prefill_piecewise_linear",),
        allow_characterization=allow_characterization,
    )
    decode_comparison, decode_split, decode_characterization, decode_fidelity = _select_comparison(
        report,
        ("decode_2d_surface", "decode_additive"),
        allow_characterization=allow_characterization,
    )
    prefill_candidate = _fit_candidate(fit, prefill_comparison["name"])
    decode_candidate = _fit_candidate(fit, decode_comparison["name"])
    prefill = _prefill_model(prefill_candidate)
    decode = _decode_model(decode_candidate)
    mixed_comparison = next(
        (
            comparison
            for comparison in report.get("comparisons", [])
            if comparison.get("name") == "composition_additive"
            and comparison.get("status") == "fitted"
        ),
        None,
    )
    mixed_validation = (
        mixed_comparison.get("by_split", {}).get("threshold_validation", {})
        if isinstance(mixed_comparison, dict)
        else {}
    )
    mixed_errors = mixed_validation.get("errors", {}) if isinstance(mixed_validation, dict) else {}
    mixed_is_validated = (
        isinstance(mixed_validation, dict)
        and mixed_validation.get("status") == "measured"
        and isinstance(mixed_errors.get("absolute_ms"), dict)
        and mixed_errors.get("count", 0) > 0
        and not mixed_validation.get("ood")
        and not mixed_validation.get("uncertain")
    )
    mixed = _mixed_model(mixed_validation) if mixed_is_validated else None
    max_prefill = max(prefill["input_tokens"])
    max_decode_reqs = max(decode["decode_reqs"])
    max_decode_ctx = max(decode["sum_decode_ctx_tokens"])
    source_paths = [
        str((campaign_root / "manifest.json").resolve()),
        str(normalized_path.resolve()),
        str(fit_path.resolve()),
        str(report_path.resolve()),
    ]
    source_digest = _sha256_bytes(
        "\n".join(_sha256_file(Path(path)) for path in source_paths).encode("utf-8")
    )
    normalized_source = normalized.get("source") if isinstance(normalized.get("source"), dict) else {}
    campaign_provenance = campaign.get("provenance") if isinstance(campaign.get("provenance"), dict) else {}
    collection_command = str(campaign_provenance.get("collection_command", ""))
    calibration_driver = str(
        normalized_source.get("collector")
        or campaign_provenance.get("collector")
        or ("vllm-bench" if "vllm-bench" in collection_command else "unknown")
    )
    rust_manifest = _rust_manifest(
        campaign,
        source_paths,
        source_digest,
        tokenizer_digest,
        calibration_driver,
    )
    manifest_bytes = json.dumps(
        rust_manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    profile = {
        "schema_version": 3,
        "calibration": {
            "manifest": "calibration.manifest.json",
            "sha256": manifest_sha256,
        },
        "scheduler": rust_manifest["scheduler"],
        "predictor": {
            "coverage": {
                "decode_reqs": [0, max_decode_reqs],
                "sum_decode_ctx_tokens": [0, max_decode_ctx],
                "prefill_tokens_in_step": [0, max_prefill],
            },
            "model": {
                "kind": "functional",
                "prefill": prefill,
                "decode": decode,
                "mixed": mixed,
            },
        },
    }
    residual_candidate = next(
        (
            item
            for item in report.get("residual_candidates", [])
            if isinstance(item, dict)
            and item.get("name") == "sparse_residual_correction"
            and item.get("status") == "candidate"
        ),
        None,
    )
    residual_count = (
        len(residual_candidate.get("residuals_ms", {}))
        if isinstance(residual_candidate, dict)
        and isinstance(residual_candidate.get("residuals_ms"), dict)
        else 0
    )
    selection = {
        "status": (
            "characterization"
            if prefill_characterization or decode_characterization
            else "selected"
        ),
        "campaign_id": campaign["campaign_id"],
        "target": campaign["target"],
        "fit_artifact": str(fit_path.resolve()),
        "report_artifact": str(report_path.resolve()),
        "artifact_digests": lineage,
        "calibration_driver": calibration_driver,
        "fidelity_report": {
            "sha256": lineage["report_sha256"],
            "per_cell": any(
                isinstance(item, dict)
                and any(
                    isinstance(split.get("by_cell"), dict)
                    for split in item.get("by_split", {}).values()
                    if isinstance(split, dict)
                )
                for item in report.get("comparisons", [])
            ),
        },
        "prefill": {
            "candidate": prefill_comparison["name"],
            "evidence_split": prefill_split,
            "source": prefill_candidate.get("source"),
            "fidelity": prefill_fidelity,
        },
        "decode": {
            "candidate": decode_comparison["name"],
            "evidence_split": decode_split,
            "source": decode_candidate.get("source"),
            "fidelity": decode_fidelity,
        },
        "cached_prefill": {
            "status": "fit_report_only",
            "runtime_predictor": "excluded",
            "candidate": "cached_prefill_surface",
            "report_status": next(
                (
                    comparison.get("status")
                    for comparison in report.get("comparisons", [])
                    if comparison.get("name") == "cached_prefill_surface"
                ),
                "unsupported",
            ),
        },
        "mixed": (
            {
                "status": "selected",
                "candidate": "composition_additive",
                "evidence_split": "threshold_validation",
                "intercept_us": 0,
                "residual_correction": {
                    "status": "offline_only" if residual_candidate is not None else "unsupported",
                    "candidate": "sparse_residual_correction" if residual_candidate is not None else None,
                    "fit_split": residual_candidate.get("fit_split") if isinstance(residual_candidate, dict) else None,
                    "count": residual_count,
                },
            }
            if mixed_is_validated and mixed is not None
            else {"status": "unsupported", "reason": "no in-domain mixed evidence"}
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "calibration.manifest.json").write_bytes(manifest_bytes)
    (output_root / "profile.json").write_text(
        json.dumps(profile, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--normalized", type=Path)
    parser.add_argument("--fit-artifact", type=Path)
    parser.add_argument("--report-artifact", type=Path)
    parser.add_argument("--allow-characterization", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        selection = materialize_profile(
            args.campaign_root,
            args.output_root,
            tokenizer_config=args.tokenizer_config,
            allow_characterization=args.allow_characterization,
            normalized_artifact=args.normalized,
            fit_artifact=args.fit_artifact,
            report_artifact=args.report_artifact,
        )
    except ProfileError as exc:
        print(f"profile error: {exc}")
        return 2
    print(json.dumps(selection, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
