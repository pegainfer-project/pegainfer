#!/usr/bin/env python3
"""Load and materialize an explicit JSON calibration campaign."""

from __future__ import annotations

from pathlib import Path

from manifest import ManifestError, build_manifest, load_spec


def load_json_campaign(path: Path) -> dict:
    """Load a campaign spec and reject implicit Python matrix definitions."""

    spec = load_spec(path)
    workload = spec.get("workload")
    if not isinstance(workload, dict) or "matrices" not in workload:
        raise ManifestError("campaign JSON requires workload.matrices; implicit Python matrices are unsupported")
    return spec


def build_json_manifest(path: Path) -> dict:
    """Build an immutable manifest from one explicit campaign JSON file."""

    return build_manifest(load_json_campaign(path))
