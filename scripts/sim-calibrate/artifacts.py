#!/usr/bin/env python3
"""Immutable raw-artifact index and resumable cell checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
INDEX_SCHEMA_VERSION = 2


class ArtifactError(RuntimeError):
    """Raised when an artifact store cannot preserve campaign lineage."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _render(document: Any) -> bytes:
    return (json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_atomic(path: Path, document: Any) -> str:
    payload = _render(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _sha256_bytes(payload)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"failed to read artifact JSON {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ArtifactError(f"artifact JSON must be an object: {path}")
    return document


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_compiled_bundle(root: Path, documents: dict[str, Any], campaign_id: str) -> dict[str, Any]:
    """Publish immutable compiler outputs; the index is written last.

    Check all conflicts before writing anything. Exclusive hard-link creation
    also prevents another compiler from replacing a file after that check.
    An interrupted write can be resumed with the identical document bytes.
    """
    root = root.expanduser().resolve()
    if root.is_relative_to(_repo_root()):
        raise ArtifactError("compiled artifact root must be outside the repository")
    payloads = {name: _render(document) for name, document in documents.items()}
    index = {
        "kind": "compiled_artifact_index", "schema_version": 3, "campaign_id": campaign_id,
        "files": {name: {"sha256": _sha256_bytes(data), "size": len(data)} for name, data in sorted(payloads.items())},
    }
    payloads["artifact_index.json"] = _render(index)
    for name, data in payloads.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not name:
            raise ArtifactError(f"invalid compiler artifact path: {name}")
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ArtifactError(f"compiler artifact escapes root: {name}")
        if path.exists() and (not path.is_file() or path.read_bytes() != data):
            raise ArtifactError(f"refusing to overwrite a different compiler artifact: {name}")
    for name, data in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ArtifactError(f"compiler artifact changed while publishing: {name}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return index


class ArtifactStore:
    """A campaign-scoped store whose raw files live outside the repository."""

    def __init__(self, root: Path, campaign_id: str) -> None:
        root = root.expanduser().resolve()
        repo = _repo_root()
        try:
            root.relative_to(repo)
        except ValueError:
            pass
        else:
            raise ArtifactError(f"raw artifact root must be outside the repository: {root}")
        if not campaign_id or any(character not in "0123456789abcdef" for character in campaign_id.lower()):
            raise ArtifactError("campaign_id must be a non-empty hexadecimal digest")
        self.root = root
        self.campaign_id = campaign_id
        self.index_path = root / "artifact_index.json"
        self.manifest_path = root / "manifest.json"
        self.checkpoint_dir = root / "checkpoints"
        self.cell_dir = root / "cells"
        self._index = self._load_index()

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {
                "schema_version": INDEX_SCHEMA_VERSION,
                "campaign_id": self.campaign_id,
                "created_at_utc": _now(),
                "updated_at_utc": _now(),
                "cells": {},
                "variant_cells": {},
                "files": {},
            }
        index = _read_json(self.index_path)
        if index.get("schema_version") not in (SCHEMA_VERSION, INDEX_SCHEMA_VERSION) or index.get("campaign_id") != self.campaign_id:
            raise ArtifactError("artifact index schema or campaign_id mismatch")
        if not isinstance(index.get("cells"), dict) or not isinstance(index.get("files"), dict):
            raise ArtifactError("artifact index cells/files must be objects")
        index.setdefault("variant_cells", {})
        return index

    @property
    def index(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._index))

    def _save_index(self) -> None:
        self._index["updated_at_utc"] = _now()
        _write_atomic(self.index_path, self._index)

    def attach_manifest(self, manifest: dict[str, Any]) -> None:
        if manifest.get("campaign_id") != self.campaign_id:
            raise ArtifactError("manifest campaign_id does not match artifact store")
        if self.manifest_path.exists():
            existing = _read_json(self.manifest_path)
            if existing != manifest:
                raise ArtifactError("refusing to replace a different campaign manifest")
        else:
            _write_atomic(self.manifest_path, manifest)
        self._index["files"]["manifest.json"] = {
            "sha256": _sha256_bytes(self.manifest_path.read_bytes()),
            "size": self.manifest_path.stat().st_size,
        }
        self._save_index()

    def status(self, cell_id: str) -> str:
        entry = self._index["cells"].get(cell_id)
        return entry.get("status", "pending") if entry else "pending"

    def begin_cell(self, cell: dict[str, Any]) -> None:
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ArtifactError("cell must have a cell_id")
        current = self._index["cells"].get(cell_id)
        if current and current.get("status") == "sealed":
            raise ArtifactError(f"sealed cell cannot be overwritten: {cell_id}")
        if current and current.get("name") != cell.get("name"):
            raise ArtifactError(f"cell identity changed for {cell_id}")
        self._index["cells"][cell_id] = {
            "cell_id": cell_id,
            "name": cell.get("name"),
            "split": cell.get("split"),
            "status": "running",
            "updated_at_utc": _now(),
        }
        _write_atomic(self.checkpoint_dir / f"{cell_id}.json", self._index["cells"][cell_id])
        self._save_index()

    def record_cell(
        self,
        cell: dict[str, Any],
        result: dict[str, Any],
        *,
        collector_argv: list[str],
        probe: dict[str, Any],
    ) -> dict[str, Any]:
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ArtifactError("cell must have a cell_id")
        if self.status(cell_id) == "sealed":
            raise ArtifactError(f"sealed cell cannot be overwritten: {cell_id}")
        status = result.get("status")
        if status not in ("succeeded", "invalid", "failed"):
            raise ArtifactError(f"cell result has unsupported status: {status!r}")
        record = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "cell": cell,
            "probe": probe,
            "collector_argv": collector_argv,
            "result": result,
            "recorded_at_utc": _now(),
        }
        result_path = self.cell_dir / cell_id / "result.json"
        digest = _write_atomic(result_path, record)
        relative = result_path.relative_to(self.root).as_posix()
        self._index["cells"][cell_id] = {
            "cell_id": cell_id,
            "name": cell.get("name"),
            "split": cell.get("split"),
            "status": status,
            "result_path": relative,
            "result_sha256": digest,
            "size": result_path.stat().st_size,
            "updated_at_utc": _now(),
        }
        self._index["files"][relative] = {"sha256": digest, "size": result_path.stat().st_size}
        _write_atomic(self.checkpoint_dir / f"{cell_id}.json", self._index["cells"][cell_id])
        self._save_index()
        return self._index["cells"][cell_id]

    def seal_cell(self, cell_id: str) -> None:
        entry = self._index["cells"].get(cell_id)
        if not entry:
            raise ArtifactError(f"cannot seal unknown cell: {cell_id}")
        if entry.get("status") == "sealed":
            return
        if entry.get("status") != "succeeded":
            raise ArtifactError(f"only succeeded cells can be sealed: {cell_id}")
        result_path = self.root / entry["result_path"]
        if not result_path.exists():
            raise ArtifactError(f"sealed cell result is missing: {result_path}")
        actual_digest = _sha256_bytes(result_path.read_bytes())
        if actual_digest != entry.get("result_sha256"):
            raise ArtifactError(f"cell result digest changed before sealing: {cell_id}")
        entry["status"] = "sealed"
        entry["sealed_at_utc"] = _now()
        _write_atomic(self.checkpoint_dir / f"{cell_id}.json", entry)
        self._save_index()

    def variant_status(self, cell_id: str, variant: str) -> str:
        entry = self._index["variant_cells"].get(f"{cell_id}:{variant}")
        return entry.get("status", "pending") if entry else "pending"

    def begin_variant_cell(self, cell: dict[str, Any], variant: str) -> None:
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ArtifactError("cell must have a cell_id")
        key = f"{cell_id}:{variant}"
        current = self._index["variant_cells"].get(key)
        if current and current.get("status") == "sealed":
            raise ArtifactError(f"sealed variant cell cannot be overwritten: {key}")
        if current and current.get("name") != cell.get("name"):
            raise ArtifactError(f"variant cell identity changed for {key}")
        entry = {
            "cell_id": cell_id,
            "variant": variant,
            "name": cell.get("name"),
            "split": cell.get("split"),
            "status": "running",
            "attempts": current.get("attempts", []) if current else [],
            "updated_at_utc": _now(),
        }
        self._index["variant_cells"][key] = entry
        _write_atomic(self.checkpoint_dir / f"{cell_id}.{variant}.json", entry)
        self._save_index()

    def record_variant_cell(
        self,
        cell: dict[str, Any],
        variant: str,
        result: dict[str, Any],
        *,
        collector_argv: list[str],
        probe: dict[str, Any],
    ) -> dict[str, Any]:
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ArtifactError("cell must have a cell_id")
        key = f"{cell_id}:{variant}"
        current = self._index["variant_cells"].get(key)
        if current is None:
            raise ArtifactError(f"variant cell was not started: {key}")
        if current.get("status") == "sealed":
            raise ArtifactError(f"sealed variant cell cannot be overwritten: {key}")
        status = result.get("status")
        if status not in ("succeeded", "invalid", "failed"):
            raise ArtifactError(f"variant cell result has unsupported status: {status!r}")
        attempt = len(current.get("attempts", [])) + 1
        while (self.cell_dir / cell_id / variant / f"attempt-{attempt:04d}" / "result.json").exists():
            attempt += 1
        record = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "cell": cell,
            "variant": variant,
            "attempt": attempt,
            "probe": probe,
            "collector_argv": collector_argv,
            "result": result,
            "recorded_at_utc": _now(),
        }
        result_path = self.cell_dir / cell_id / variant / f"attempt-{attempt:04d}" / "result.json"
        digest = _write_atomic(result_path, record)
        relative = result_path.relative_to(self.root).as_posix()
        attempt_entry = {
            "attempt": attempt,
            "status": status,
            "result_path": relative,
            "result_sha256": digest,
            "size": result_path.stat().st_size,
            "recorded_at_utc": record["recorded_at_utc"],
        }
        current["attempts"] = [*current.get("attempts", []), attempt_entry]
        current.update({"status": status, "result_path": relative, "result_sha256": digest, "size": result_path.stat().st_size, "updated_at_utc": _now()})
        self._index["files"][relative] = {"sha256": digest, "size": result_path.stat().st_size}
        _write_atomic(self.checkpoint_dir / f"{cell_id}.{variant}.json", current)
        self._save_index()
        return current

    def seal_variant_cell(self, cell_id: str, variant: str) -> None:
        key = f"{cell_id}:{variant}"
        entry = self._index["variant_cells"].get(key)
        if not entry:
            raise ArtifactError(f"cannot seal unknown variant cell: {key}")
        if entry.get("status") == "sealed":
            return
        if entry.get("status") != "succeeded":
            raise ArtifactError(f"only succeeded variant cells can be sealed: {key}")
        result_path = self.root / entry["result_path"]
        if not result_path.exists():
            raise ArtifactError(f"sealed variant result is missing: {result_path}")
        actual_digest = _sha256_bytes(result_path.read_bytes())
        if actual_digest != entry.get("result_sha256"):
            raise ArtifactError(f"variant result digest changed before sealing: {key}")
        entry["status"] = "sealed"
        entry["sealed_at_utc"] = _now()
        _write_atomic(self.checkpoint_dir / f"{cell_id}.{variant}.json", entry)
        self._save_index()
