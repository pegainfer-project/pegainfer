#!/usr/bin/env python3
"""Publish a profile bundle by linking its frozen tolerance and passed signoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


class PublicationError(ValueError):
    """Raised when a profile bundle cannot be published safely."""


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PublicationError(f"JSON artifact must be an object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _relative(root: Path, path: Path, name: str) -> str:
    root = root.resolve()
    path = path.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise PublicationError(f"{name} must live inside the profile bundle: {path}") from exc


def publish_profile(
    profile_path: Path,
    tolerance_path: Path,
    signoff_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    profile, profile_digest = _load_json(profile_path)
    selection, selection_digest = _load_json(profile_path.parent / "selection.json")
    tolerance, tolerance_digest = _load_json(tolerance_path)
    signoff, signoff_digest = _load_json(signoff_path)
    if tolerance.get("status") != "frozen":
        raise PublicationError("tolerance artifact is not frozen")
    if signoff.get("status") != "passed":
        raise PublicationError("signoff artifact is not passed")
    if tolerance.get("profile", {}).get("sha256") != profile_digest:
        raise PublicationError("tolerance does not reference this profile")
    if signoff.get("profile", {}).get("sha256") != profile_digest:
        raise PublicationError("signoff does not reference this profile")
    if signoff.get("tolerance", {}).get("sha256") != tolerance_digest:
        raise PublicationError("signoff does not reference this tolerance artifact")
    target = selection.get("target")
    if not isinstance(target, dict):
        raise PublicationError("selection target identity is missing")
    if tolerance.get("target_identity") != target or signoff.get("target_identity") != target:
        raise PublicationError("publication target identity does not match selection")
    root = profile_path.parent.resolve()
    publication = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "profile": {"path": _relative(root, profile_path, "profile"), "sha256": profile_digest},
        "tolerance": {"path": _relative(root, tolerance_path, "tolerance"), "sha256": tolerance_digest},
        "signoff": {"path": _relative(root, signoff_path, "signoff"), "sha256": signoff_digest},
        "target_identity": target,
        "selection_sha256": selection_digest,
        "digest_chain": {
            "profile_sha256": profile_digest,
            "selection_sha256": selection_digest,
            "tolerance_sha256": tolerance_digest,
            "signoff_sha256": signoff_digest,
        },
    }
    output_path = output_path or root / "publication.json"
    rendered = json.dumps(publication, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    output_path = output_path.expanduser().resolve()
    if output_path.parent != root:
        raise PublicationError("publication output must live beside profile.json")
    if output_path.exists() and output_path.read_text(encoding="utf-8") != rendered:
        raise PublicationError(f"refusing to overwrite a different publication artifact: {output_path}")
    if not output_path.exists():
        output_path.write_text(rendered, encoding="utf-8")
    return publication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--tolerance", type=Path, required=True)
    parser.add_argument("--signoff", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        publication = publish_profile(args.profile, args.tolerance, args.signoff, args.output)
    except (OSError, PublicationError) as exc:
        print(f"publication error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": publication["status"], "profile_sha256": publication["profile"]["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
