#!/usr/bin/env python3
"""Materialize one selected functional simulator profile from calibration artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from artifacts import (_read_json, _render, checked_path, file_reference, fitted_bundle_paths,
                       load_compiled_bundle, write_fitted_bundle, write_derived_document)
from manifest import (
    canonical_cache_identity,
    canonical_client_identity,
    canonical_scheduler_identity,
    canonical_json,
)


CALIBRATION_PATH = 'derived/cached_prefill/calibration.json'


class ProfileError(ValueError):
    """Raised when calibration evidence cannot produce a safe profile."""


def _calibration_manifest(campaign: dict, variant: str, cells: list[dict], source_digest: str,
                          source_artifacts: list[str], fitter_revision: str) -> dict:
    target = campaign["variants"]["target"]
    serving = campaign["variants"][variant]
    identity, runtime, provenance = target["identity"], target["runtime"], campaign["provenance"]
    return {
        "schema_version": 2,
        "target_engine": identity["engine"], "engine_version": identity["engine_version"],
        "model_id": serving["identity"]["model"], "model_revision": identity["model_revision"],
        "model_config_sha256": identity["model_config_sha256"], "gpu": runtime.get("gpu", "unbound"),
        "server_flags": target["server_flags"], "collection_command": canonical_json(provenance["compile_argv"]),
        "source_artifacts": source_artifacts,
        "backend": runtime.get("backend", identity["engine"]), "driver": runtime.get("driver", "not_recorded"),
        "cuda": runtime.get("cuda", "not_recorded"), "parallelism": runtime.get("parallelism", "1"),
        "quantization": runtime.get("quantization", "none"),
        "scheduler": canonical_scheduler_identity(serving["runtime"]["scheduler"]),
        "cache": canonical_cache_identity(serving["runtime"].get("cache")),
        "client": canonical_client_identity(campaign["client"], tokenizer_config_sha256=campaign["client"]["tokenizer_config_sha256"]),
        "workload": {
            "cells": [{"id": cell["cell_id"], "family": cell["kind"], "input_tokens": cell["input_tokens"],
                       "output_tokens": cell["output_tokens"], "concurrency": cell["concurrency"],
                       "arrival_pattern": "fixed_batch", "attribution": "black_box_effective"} for cell in cells],
            "warmup_repeats": max(cell["warmup_repeats"] for cell in cells),
            "repeats": max(cell["repeats"] for cell in cells),
        },
        "split": {name: [cell["cell_id"] for cell in cells if name == cell["split"]] for name in campaign["splits"]},
        "provenance": {
            "collector_revision": provenance["compiler_revision"], "fitter_revision": fitter_revision,
            "artifact_uri": provenance["artifact_uri"], "source_digest": source_digest,
        },
    }


def reference_bundle(campaign: dict[str, Any], cell: dict, *, interval_us: int,
                     source_digest: str, source_artifacts: list[str], manifest_name: str = "manifest.json") -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a known functional reference, independent of fit/report artifacts."""
    manifest = _calibration_manifest(campaign, "zero_cost", [cell], source_digest, source_artifacts,
                                     "not_applicable_known_reference")
    scheduler = manifest["scheduler"]
    manifest_bytes = _render(manifest)
    profile = {
        "schema_version": 3,
        "calibration": {"manifest": manifest_name, "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        "scheduler": scheduler,
        "predictor": {
            "coverage": {
                "decode_reqs": [0, scheduler["max_num_seqs"]],
                "sum_decode_ctx_tokens": [0, scheduler["max_num_seqs"] * scheduler["max_model_len"]],
                "prefill_tokens_in_step": [0, scheduler["max_num_batched_tokens"]],
            },
            "model": {
                "kind": "functional",
                "prefill": {"input_tokens": [0, scheduler["max_num_batched_tokens"]], "duration_us": [0, 0]},
                "decode": {"decode_reqs": [0, scheduler["max_num_seqs"]],
                           "sum_decode_ctx_tokens": [0, scheduler["max_num_seqs"] * scheduler["max_model_len"]],
                           "duration_us": [interval_us] * 4},
                "mixed": {"kind": "composition_additive", "intercept_us": 0},
            },
        },
    }
    return manifest, profile


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
    scaled = milliseconds * 1000.0
    if not math.isfinite(scaled):
        raise ProfileError(f"{context} exceeds u64 microseconds")
    encoded = round(scaled)
    if encoded > 2**64 - 1:
        raise ProfileError(f"{context} exceeds u64 microseconds")
    return encoded


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
    _axis(inputs, "prefill.input_tokens", 2**32 - 1)
    return {"input_tokens": inputs, "duration_us": durations}


def _axis(values: list, label: str, maximum: int) -> list[int]:
    if (not isinstance(values, list) or len(values) < 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or x < 0 or x > maximum
                   or not math.isfinite(x) or int(x) != x for x in values)
            or any(a >= b for a, b in zip(values, values[1:]))):
        raise ProfileError(f"{label} must be a strictly increasing integer axis with at least two points")
    return [int(x) for x in values]


def _decode_model(candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate.get("name") != "decode_2d_surface":
        raise ProfileError("C6 requires decode_2d_surface; no candidate fallback")
    axes = candidate["axes"]
    requests = _axis(axes["requests"], "decode.requests", 2**32 - 1)
    contexts = _axis(axes["kv_tokens"], "decode.kv_tokens", 2**64 - 1)
    values = candidate["values_ms"]
    if len(values) != len(requests) or any(len(row) != len(contexts) for row in values):
        raise ProfileError("decode surface dimensions do not match axes")
    return {"decode_reqs": requests, "sum_decode_ctx_tokens": contexts,
            "duration_us": [_ms_to_us(value, "decode surface duration") for row in values for value in row]}


def _verified_training(root: Path, campaign: dict, normalized_path: Path, fit_path: Path) -> tuple[dict, dict, dict, dict]:
    """Validate frozen acquisition, paired training and numerical recomputation."""
    from fit import verify_fit
    from normalize import NORMALIZER_SOURCES
    normalized, fitted = _read_json(normalized_path), _read_json(fit_path)
    if (normalized.get('campaign_id') != campaign['campaign_id']
            or Path(normalized['source']['artifact_root']).resolve() != root):
        raise ProfileError('profile training belongs to another campaign/root')
    frozen_sources = campaign['client']['observer']['sources']
    if normalized['source']['normalizer_sources'] != {name: frozen_sources[name] for name in NORMALIZER_SOURCES}:
        raise ProfileError('training normalizer differs from frozen acquisition sources')
    verify_fit(fitted, normalized, normalized_digest=file_reference(normalized_path)['sha256'])
    sources = {'normalized': {'path': str(normalized_path), **file_reference(normalized_path)},
               'fit': {'path': str(fit_path), **file_reference(fit_path)}}
    generator_sources = {'profile.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         **fitted['source']['fitter_sources']}
    return normalized, fitted, sources, generator_sources


def _calibration_document(root: Path, campaign: dict, normalized_path: Path, fit_path: Path) -> dict:
    normalized, fitted, sources, generator_sources = _verified_training(root, campaign, normalized_path, fit_path)
    surface = _fit_candidate(fitted, 'cached_prefill_surface')
    source = {'training': sources, 'generator_sources': generator_sources,
              'compiler_index': file_reference(root / 'artifact_index.json'),
              'plans': [{'cell_id': cell['cell_id'], **cell['plan']} for cell in campaign['workload']['cells']
                        if cell['split'] == 'train' and cell['kind'] == 'cached_prefill'],
              'sealed_training': normalized['source']}
    return {'kind': 'cached_prefill_calibration', 'schema_version': 1,
            'campaign_id': campaign['campaign_id'], 'status': 'fitted',
            'target_identity': campaign['variants']['target']['identity'],
            'cache_policy': campaign['variants']['target']['runtime']['cache'],
            'surface': surface, 'source': source,
            'source_sha256': hashlib.sha256(_render(source)).hexdigest(),
            'runtime_enabled': False, 'validation_status': 'offline_fit_only'}


def verify_calibration(root: Path, campaign: dict) -> dict:
    """Recompute the offline surface and all bindings from sealed source bytes."""
    path = checked_path(root, CALIBRATION_PATH)
    document = _read_json(path)
    if document.get('kind') != 'cached_prefill_calibration' or document.get('schema_version') != 1:
        raise ProfileError('unsupported cached-prefill calibration artifact')
    sources = document['source']['training']
    expected = _calibration_document(root, campaign, Path(sources['normalized']['path']), Path(sources['fit']['path']))
    if path.read_bytes() != _render(expected):
        raise ProfileError('cached-prefill calibration differs from verified training')
    return {'path': CALIBRATION_PATH, **file_reference(path)}


def materialize_calibration(root: Path, *, normalized_artifact: Path, fit_artifact: Path) -> dict:
    """Publish the cached-only endpoint; no runtime curves or replay are required."""
    from evidence_proxy import observer_identity
    root = root.expanduser().resolve()
    campaign = load_compiled_bundle(root)
    if campaign['client']['observer'] != observer_identity():
        raise ProfileError('frozen materializer changed; compile and collect into a new root')
    document = _calibration_document(root, campaign, normalized_artifact.expanduser().resolve(), fit_artifact.expanduser().resolve())
    write_derived_document(root / CALIBRATION_PATH, document, root)
    verify_calibration(root, campaign)
    return document


def _fitted_documents(root: Path, campaign: dict, normalized_path: Path, fit_path: Path) -> dict:
    """Reconstruct the entire runtime bundle from verified, pinned training bytes."""
    normalized, fitted, sources, generator_sources = _verified_training(root, campaign, normalized_path, fit_path)
    prefill_candidate = _fit_candidate(fitted, 'cold_prefill_piecewise_linear')
    decode_candidate = _fit_candidate(fitted, 'decode_2d_surface')
    prefill, decode = _prefill_model(prefill_candidate), _decode_model(decode_candidate)
    cached = ({'status': 'fitted', 'candidate': 'cached_prefill_surface', 'runtime_enabled': False,
               'validation_status': 'offline_fit_only', 'artifact': verify_calibration(root, campaign)}
              if normalized['cached_requirements'] else {'status': 'not_requested', 'runtime_enabled': False})
    if normalized['cached_requirements']:
        sources['cached_prefill'] = {**cached['artifact'], 'path': str(root / cached['artifact']['path'])}
    source_digest = hashlib.sha256(_render({'training': sources, 'generator_sources': generator_sources})).hexdigest()
    manifest = _calibration_manifest(campaign, "sim", campaign["workload"]["cells"], source_digest,
                                     [value["path"] for value in sources.values()],
                                     "sim-calibrate-fit-schema-3:" + generator_sources["fit.py"])
    scheduler = manifest["scheduler"]
    max_requests, max_context, max_prefill = decode["decode_reqs"][-1], decode["sum_decode_ctx_tokens"][-1], prefill["input_tokens"][-1]
    if (max_requests > scheduler["max_num_seqs"] or max_prefill > scheduler["max_num_batched_tokens"]
            or max_context > scheduler["max_num_seqs"] * scheduler["max_model_len"]):
        raise ProfileError("fitted domain exceeds scheduler capacity")
    paths = fitted_bundle_paths(root, campaign)
    profile = {
        "schema_version": 3,
        "calibration": {"manifest": "calibration.manifest.json",
                        "sha256": hashlib.sha256(_render(manifest)).hexdigest()},
        "scheduler": scheduler,
        "predictor": {
            # Zero denotes an absent component. Positive work still has to be
            # covered by its actual curve; no zero knots or extrapolation.
            "coverage": {"decode_reqs": [0, max_requests], "sum_decode_ctx_tokens": [0, max_context],
                         "prefill_tokens_in_step": [0, max_prefill]},
            "model": {"kind": "functional", "prefill": prefill, "decode": decode,
                      "mixed": {"kind": "composition_additive", "intercept_us": 0}},
        },
    }
    documents = {paths["profile"].name: profile, paths["manifest"].name: manifest}
    selection = {
        "kind": "fitted_profile_selection", "schema_version": 1, "status": "selected",
        "campaign_id": campaign["campaign_id"], "target": campaign["variants"]["target"]["identity"],
        "training": sources, "generator_sources": generator_sources,
        "validation_status": "pending", "selection_policy": "fixed_train_surface",
        "prefill": {"candidate": prefill_candidate["name"], "evidence_split": "train", "source": prefill_candidate["source"]},
        "decode": {"candidate": decode_candidate["name"], "evidence_split": "train", "source": decode_candidate["source"]},
        "mixed": {"candidate": "composition_additive", "intercept_us": 0, "validation_status": "pending"},
        "cached_prefill": cached,
        "files": {name: {"sha256": hashlib.sha256(_render(value)).hexdigest(), "size": len(_render(value))}
                  for name, value in documents.items()},
    }
    return {**documents, paths["selection"].name: selection}


def verify_fitted_bundle(root: Path, campaign: dict) -> dict:
    """Offline proof of the selected model, not merely self-reported digests."""
    paths = fitted_bundle_paths(root, campaign)
    selection = _read_json(paths["selection"])
    if selection.get("kind") != "fitted_profile_selection" or selection.get("schema_version") != 1:
        raise ProfileError("sim requires a current complete fitted profile bundle")
    expected = _fitted_documents(root, campaign, Path(selection["training"]["normalized"]["path"]),
                                 Path(selection["training"]["fit"]["path"]))
    for path in paths.values():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != _render(expected[path.name]):
            raise ProfileError(f"fitted bundle differs from verified training: {path.name}")
    return file_reference(paths["selection"])


def materialize_profile(root: Path, *, normalized_artifact: Path, fit_artifact: Path) -> dict:
    from evidence_proxy import observer_identity
    root = root.expanduser().resolve()
    campaign = load_compiled_bundle(root)
    if campaign["client"]["observer"] != observer_identity():
        raise ProfileError("frozen materializer changed; compile and collect into a new root")
    if any(cell['split'] == 'train' and cell['kind'] == 'cached_prefill' for cell in campaign['workload']['cells']):
        materialize_calibration(root, normalized_artifact=normalized_artifact, fit_artifact=fit_artifact)
    documents = _fitted_documents(root, campaign, normalized_artifact.expanduser().resolve(), fit_artifact.expanduser().resolve())
    write_fitted_bundle(root, campaign, documents)
    verify_fitted_bundle(root, campaign)
    return documents["selection.json"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--fit-artifact", type=Path, required=True)
    args = parser.parse_args()
    try:
        selection = materialize_profile(args.artifact_root, normalized_artifact=args.normalized, fit_artifact=args.fit_artifact)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"profile error: {exc}")
        return 2
    print(json.dumps({key: selection[key] for key in ("campaign_id", "status", "validation_status")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
