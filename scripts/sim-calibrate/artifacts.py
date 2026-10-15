#!/usr/bin/env python3
"""Immutable raw-artifact index and resumable cell checkpoints."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
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
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
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


def checked_path(root: Path, relative: str) -> Path:
    """Resolve an artifact reference without permitting escape through symlinks."""
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or not relative:
        raise ArtifactError(f'unsafe artifact reference: {relative}')
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ArtifactError(f'artifact escapes root: {relative}')
    return result


def file_reference(path: Path) -> dict[str, Any]:
    return {'sha256': _sha256_bytes(path.read_bytes()), 'size': path.stat().st_size}


def verify_files(root: Path, files: dict[str, Any]) -> None:
    if not isinstance(files, dict) or not files:
        raise ArtifactError('artifact index has no files')
    for relative, expected in files.items():
        path = checked_path(root, relative)
        if not path.is_file() or file_reference(path) != expected:
            raise ArtifactError(f'artifact bytes changed or missing: {relative}')


def load_compiled_bundle(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    index = _read_json(root / 'artifact_index.json')
    if index.get('kind') != 'compiled_artifact_index' or index.get('schema_version') != 3:
        raise ArtifactError('not a current compiled bundle')
    verify_files(root, index.get('files'))
    campaign = _read_json(root / 'compiled_campaign.json')
    if campaign.get('kind') != 'compiled_campaign' or campaign.get('schema_version') != 5:
        raise ArtifactError('unsupported compiled campaign; compile a new experiment')
    if campaign.get('campaign_id') != index.get('campaign_id'):
        raise ArtifactError('compiler campaign/index identity mismatch')
    from request_plan import build_plan
    for cell in campaign['workload']['cells']:
        plan_path = cell['plan']['path']
        if plan_path not in index['files']:
            raise ArtifactError('plan is not in compiler index')
        actual = _read_json(checked_path(root, plan_path))
        if actual != build_plan(cell, campaign['client']) or actual['plan_sha256'] != cell['plan']['plan_sha256']:
            raise ArtifactError('plan does not match compiled cell/client')
    return campaign


@contextmanager
def execution_snapshot(root: Path, campaign_id: str):
    """Read a consistent index under the driver's existing lock, without resuming it."""
    root = root.resolve()
    with (root / '.execution.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArtifactError('execution is active; retry normalization after the driver exits') from exc
        path = root / 'execution_index.json'
        payload = path.read_bytes()
        index = json.loads(payload)
        if (not isinstance(index, dict) or index.get('kind') != 'execution_index' or index.get('schema_version') != 1
                or index.get('campaign_id') != campaign_id or not isinstance(index.get('cells'), dict)):
            raise ArtifactError('execution index identity/schema mismatch')
        yield index, _sha256_bytes(payload)


def sealed_reference(entry: dict[str, Any]) -> dict[str, Any]:
    """Pin the selected attempt, excluding mutable execution history."""
    if entry.get('status') != 'sealed':
        raise ArtifactError('required execution is missing or not sealed')
    fields = ('status', 'attempt', 'path', 'plan_sha256', 'binding',
              'measurement', 'raw_index', 'workload_evidence')
    if any(field not in entry for field in fields):
        raise ArtifactError('sealed execution is missing an attempt reference')
    return json.loads(json.dumps({field: entry[field] for field in fields}))


def _validate_measurement_links(envelope: dict, workload: dict, plan: dict, variant: str,
                                binding: dict, raw_index_sha256: str) -> None:
    """The same provenance links must hold at sealing and at later reads."""
    from manifest import sha256_json
    from measurement_schema import validate_envelope

    validate_envelope(envelope)
    if (envelope['plan_sha256'] != plan['plan_sha256'] or envelope['variant'] != variant
            or envelope['variant_identity'] != binding.get('variant_identity')
            or envelope['provenance']['observer_sha256'] != binding.get('observer_sha256')
            or envelope['provenance']['raw_bundle_index_sha256'] != raw_index_sha256):
        raise ArtifactError('measurement plan/role/binding/raw-index mismatch')
    if (not isinstance(workload.get('requests'), list)
            or workload.get('sha256') != sha256_json(workload['requests'])
            or any(row.get('workload_evidence_sha256') != workload['sha256']
                   for row in envelope['measurements'])):
        raise ArtifactError('measurement workload evidence binding mismatch')


def read_sealed_attempt(root: Path, plan: dict, variant: str, reference: dict) -> dict:
    """Verify pinned file links offline; live PID attestation belongs to the driver."""
    if reference != sealed_reference(reference) or reference['plan_sha256'] != plan['plan_sha256']:
        raise ArtifactError('sealed attempt plan/reference mismatch')
    attempt = reference['attempt']
    expected_path = f"raw/{variant}/{plan['cell_id']}/attempt-{attempt}"
    if type(attempt) is not int or attempt < 1 or reference['path'] != expected_path:
        raise ArtifactError('sealed attempt path/role/cell mismatch')
    directory = checked_path(root, reference['path'])
    actual_files = set()
    for path in directory.rglob('*'):
        if path.is_symlink():
            raise ArtifactError(f'symlink in sealed raw evidence: {path.relative_to(directory)}')
        if path.is_file():
            actual_files.add(path.relative_to(directory).as_posix())
    for name, field in (('raw_bundle_index.json', 'raw_index'),
                        ('measurements.json', 'measurement'), ('workload_evidence.json', 'workload_evidence')):
        if file_reference(checked_path(directory, name)) != reference[field]:
            raise ArtifactError(f'sealed artifact bytes changed: {reference["path"]}/{name}')
    raw = _read_json(directory / 'raw_bundle_index.json')
    if raw.get('kind') != 'raw_bundle_index' or raw.get('schema_version') != 1:
        raise ArtifactError('unsupported raw bundle index')
    verify_files(directory, raw.get('files'))
    derived = {'raw_bundle_index.json', 'measurements.json', 'workload_evidence.json'}
    if actual_files - derived != set(raw['files']) - derived:
        raise ArtifactError('sealed raw directory contains unindexed or missing files')
    for required in ('binding.json', 'observer.json', 'attempt.json', 'probe.json'):
        if required not in raw['files']:
            raise ArtifactError(f'required raw file is not indexed: {required}')
    if _read_json(directory / 'binding.json') != reference['binding']:
        raise ArtifactError('raw binding differs from selected attempt')
    envelope = _read_json(directory / 'measurements.json')
    workload = _read_json(directory / 'workload_evidence.json')
    _validate_measurement_links(envelope, workload, plan, variant, reference['binding'], reference['raw_index']['sha256'])
    return {'directory': directory, 'raw_index': raw, 'envelope': envelope, 'workload': workload}


def write_derived_document(path: Path, document: dict, root: Path) -> None:
    """Publish exclusively and idempotently, without replacing source evidence."""
    root, path = root.resolve(), path.expanduser().resolve()
    index = _read_json(root / 'artifact_index.json')
    protected = {checked_path(root, name) for name in index['files']}
    protected.update(root / name for name in ('artifact_index.json', 'execution_index.json', '.execution.lock'))
    if (path in protected or any(path.is_relative_to(root / folder) for folder in ('raw', 'plans', 'profiles'))
            or path.is_relative_to(_repo_root())):
        raise ArtifactError('normalized output would overwrite protected source evidence or repository files')
    payload = _render(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ArtifactError(f'refusing to overwrite a different normalized artifact: {path}')
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class ExecutionStore:
    """Current execution state never mutates the immutable compiler index.

    The campaign lock spans a complete driver invocation. Attempt directories
    are allocated exclusively, including after a killed process; old raw bytes
    are never reused or overwritten.
    """

    def __init__(self, root: Path, campaign_id: str):
        self.root = root.resolve()
        if self.root.is_relative_to(_repo_root()):
            raise ArtifactError('execution artifacts must be outside repository')
        self.campaign_id = campaign_id
        self.path = self.root / 'execution_index.json'
        self.lock = None

    def __enter__(self):
        self.lock = (self.root / '.execution.lock').open('a+b')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.index = _read_json(self.path) if self.path.exists() else {
                'kind': 'execution_index', 'schema_version': 1, 'campaign_id': self.campaign_id, 'cells': {}}
            if self.index.get('kind') != 'execution_index' or self.index.get('schema_version') != 1 or self.index.get('campaign_id') != self.campaign_id:
                raise ArtifactError('execution index identity/schema mismatch')
            return self
        except BaseException:
            self.lock.close()
            raise

    def __exit__(self, *_):
        self.lock.close()

    def _save(self):
        _write_atomic(self.path, self.index)

    def sealed(self, plan: dict, variant: str, binding: dict) -> dict | None:
        entry = self.index['cells'].get(f"{plan['cell_id']}:{variant}")
        if not entry or entry['status'] != 'sealed':
            return None
        if entry.get('plan_sha256') != plan['plan_sha256'] or entry.get('binding') != binding:
            raise ArtifactError('sealed execution binding changed')
        read_sealed_attempt(self.root, plan, variant, sealed_reference(entry))
        return entry

    def begin(self, plan: dict, variant: str, binding: dict) -> tuple[Path, int]:
        key = f"{plan['cell_id']}:{variant}"
        old = self.index['cells'].get(key, {})
        if old.get('status') == 'sealed':
            raise ArtifactError('sealed execution cannot be overwritten')
        history = old.get('attempts', [])
        for previous in history:
            if previous['status'] == 'running':
                previous.update(status='interrupted', error='previous driver terminated before sealing')
        parent = self.root / 'raw' / variant / plan['cell_id']
        parent.mkdir(parents=True, exist_ok=True)
        attempt = max([a['attempt'] for a in history], default=0) + 1
        while True:
            directory = parent / f'attempt-{attempt}'
            try:
                directory.mkdir()
                break
            except FileExistsError:
                attempt += 1
        entry = {'attempt': attempt, 'path': directory.relative_to(self.root).as_posix(),
                 'status': 'running', 'started_at_utc': _now()}
        self.index['cells'][key] = {'status': 'running', 'plan_sha256': plan['plan_sha256'],
                                    'binding': binding, 'attempts': [*history, entry]}
        self._save()
        return directory, attempt

    def raw_index(self, directory: Path) -> str:
        files = {}
        for path in sorted(directory.rglob('*')):
            if path.is_symlink():
                raise ArtifactError('symlink in raw evidence')
            if path.is_file() and path.name not in ('raw_bundle_index.json', 'measurements.json'):
                files[path.relative_to(directory).as_posix()] = file_reference(path)
        return _write_atomic(directory / 'raw_bundle_index.json', {
            'kind': 'raw_bundle_index', 'schema_version': 1, 'files': files})

    def finish(self, plan: dict, variant: str, directory: Path, *, error: str | None = None) -> dict:
        key = f"{plan['cell_id']}:{variant}"
        entry = self.index['cells'][key]
        current = entry['attempts'][-1]
        status = 'failed'
        if error is None:
            envelope = _read_json(directory / 'measurements.json')
            raw = _read_json(directory / 'raw_bundle_index.json')
            verify_files(directory, raw['files'])
            workload = _read_json(directory / 'workload_evidence.json')
            _validate_measurement_links(envelope, workload, plan, variant, entry['binding'],
                                        file_reference(directory / 'raw_bundle_index.json')['sha256'])
            if [r['pairing_key'] for r in envelope['measurements']] != [r['pairing_key'] for r in plan['logical_requests']]:
                raise ArtifactError('measurement key set differs from plan')
            rows = envelope['measurements']
            status = ('sealed' if all(r['status'] == 'succeeded' for r in rows) else
                      'unsupported' if any(r['status'] == 'unsupported' for r in rows) else 'failed')
            current.update(measurement=file_reference(directory / 'measurements.json'),
                           raw_index=file_reference(directory / 'raw_bundle_index.json'),
                           workload_evidence=file_reference(directory / 'workload_evidence.json'))
        current.update(status=status, finished_at_utc=_now(), error=error)
        entry.update({k: v for k, v in current.items() if k != 'attempts'})
        self._save()
        return entry
