#!/usr/bin/env python3
"""Prepare, package, and validate the single production GDN AOT candidate."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


# One project-owned frozen contract is read by both generation and the independent
# Rust pre-link validator. Artifact values never supply their own expected pins.
_FROZEN_LOCK = json.loads(Path(__file__).with_name("source-lock.json").read_text())
FROZEN_CONTRACT = _FROZEN_LOCK["contract"]
VARIANT = FROZEN_CONTRACT["variant"]
TARGET_ARCH = FROZEN_CONTRACT["target"]["arch"]
FROZEN_FLASHINFER_COMMIT = _FROZEN_LOCK["flashinfer_commit"]
GEOMETRY = FROZEN_CONTRACT["geometry"]
PINNED_TOOLCHAIN = FROZEN_CONTRACT["toolchain"]
KERNEL_SOURCE = "flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py"
ARTIFACT_FILES = {
    "header": "kernel.h",
    "object": "kernel.o",
    "native_runtime": "libcuda_dialect_runtime_static.a",
}
FORBIDDEN_TMA_CLUSTER_LOAD = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
    "mbarrier::complete_tx::bytes.L2::cache_hint"
)
class ContractError(RuntimeError):
    """An artifact or source contract is invalid."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def source_lock_path() -> Path:
    return Path(__file__).with_name("source-lock.json")


def requirements_lock_path() -> Path:
    return Path(__file__).with_name("requirements-cu13.lock")


def compiler_path() -> Path:
    return Path(__file__).with_name("compile_sm120.py")


def load_source_lock() -> tuple[dict[str, Any], str]:
    path = source_lock_path()
    lock = read_json(path)
    patches = lock.get("patches")
    if not isinstance(patches, list) or len(patches) != 1:
        raise ContractError("source lock must contain exactly one HKV patch")
    patch = patches[0]
    if not isinstance(patch, dict):
        raise ContractError("source lock patch entry must be an object")
    patch_relative = patch.get("path")
    if patch_relative != "patches/0001-openinfer-hkv-state-layout.patch":
        raise ContractError("source lock HKV patch path mismatch")
    patch_path = path.parent / patch_relative
    if not patch_path.is_file():
        raise ContractError(f"source lock patch is missing: {patch_path}")
    if patch.get("sha256") != sha256_file(patch_path):
        raise ContractError("source lock HKV patch hash mismatch")
    patched_kernel_sha256 = lock.get("patched_kernel_sha256")
    if not isinstance(patched_kernel_sha256, str) or len(patched_kernel_sha256) != 64:
        raise ContractError("source lock patched kernel hash is missing")
    return lock, sha256_file(path)


def run_git(flashinfer_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(flashinfer_dir), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ContractError(f"git {' '.join(args)} failed for {flashinfer_dir}: {detail}")
    return result.stdout.strip()


def verify_flashinfer_base(flashinfer_dir: Path) -> str:
    flashinfer_dir = flashinfer_dir.resolve()
    commit = run_git(flashinfer_dir, "rev-parse", "HEAD")
    if commit != FROZEN_FLASHINFER_COMMIT:
        raise ContractError(
            f"FlashInfer SHA mismatch: expected {FROZEN_FLASHINFER_COMMIT}, got {commit}"
        )
    dirty = run_git(flashinfer_dir, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ContractError("FlashInfer tracked source is dirty")
    return commit


def inspect_kernel_source(source_dir: Path, commit: str, *, upstream_layout: bool = False) -> dict[str, Any]:
    kernel_path = source_dir / KERNEL_SOURCE
    if not kernel_path.is_file():
        raise ContractError(f"patched GDN kernel is missing: {kernel_path}")
    lock, source_lock_sha256 = load_source_lock()
    kernel_sha256 = sha256_file(kernel_path)
    expected_hash = lock["upstream_export_kernel_sha256" if upstream_layout else "patched_kernel_sha256"]
    _require_equal(kernel_sha256, expected_hash, "GDN kernel source hash")
    return {
        "flashinfer_commit": commit,
        "kernel_source_sha256": kernel_sha256,
        "source_lock_sha256": source_lock_sha256,
    }


def prepare_flashinfer_source(
    flashinfer_dir: Path, destination: Path, *, upstream_layout: bool = False
) -> dict[str, Any]:
    commit = verify_flashinfer_base(flashinfer_dir)
    lock, _ = load_source_lock()
    if destination.exists():
        raise ContractError(f"refusing to overwrite prepared source: {destination}")
    shutil.copytree(flashinfer_dir / "flashinfer", destination / "flashinfer")
    for patch in lock["patches"]:
        patch_path = source_lock_path().parent / patch["path"]
        patch_text = patch_path.read_text()
        if upstream_layout:
            # Keep only export type annotations. Both layout hunks are omitted,
            # so the oracle retains the pinned upstream HVK representation.
            hunks = re.split(r"(?=^@@ )", patch_text, flags=re.MULTILINE)
            patch_text = "".join(hunk for hunk in hunks if "order=" not in hunk)
        result = subprocess.run(
            ["git", "apply", "--unsafe-paths", "-"],
            input=patch_text,
            cwd=destination,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ContractError(f"failed to apply HKV patch: {detail}")
    return inspect_kernel_source(destination, commit, upstream_layout=upstream_layout)


def verify_prepared_flashinfer_source(
    source_dir: Path, flashinfer_dir: Path, *, upstream_layout: bool = False
) -> dict[str, Any]:
    commit = verify_flashinfer_base(flashinfer_dir)
    return inspect_kernel_source(source_dir, commit, upstream_layout=upstream_layout)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ContractError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def validate_compile_metadata(
    metadata: dict[str, Any], source: dict[str, Any]
) -> None:
    _require_equal(
        set(metadata),
        {
            "flashinfer_commit",
            "kernel_source_sha256",
            "source_lock_sha256",
            "generator_sha256",
            "requirements_lock_sha256",
            "toolchain",
            "aot",
        },
        "compile metadata keys",
    )
    for key in ("flashinfer_commit", "kernel_source_sha256", "source_lock_sha256"):
        _require_equal(metadata.get(key), source[key], f"compile metadata {key}")
    _require_equal(
        metadata.get("generator_sha256"),
        sha256_file(compiler_path()),
        "compile metadata generator hash",
    )
    _require_equal(
        metadata.get("requirements_lock_sha256"),
        sha256_file(requirements_lock_path()),
        "compile metadata requirements lock hash",
    )
    aot = metadata.get("aot")
    if not isinstance(aot, dict):
        raise ContractError("compile metadata is missing AOT export metadata")
    toolchain = metadata.get("toolchain")
    if not isinstance(toolchain, dict):
        raise ContractError("compile metadata is missing toolchain")
    _require_equal(toolchain, PINNED_TOOLCHAIN, "compile metadata toolchain")


def build_manifest(
    *,
    artifacts: dict[str, bytes],
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        **FROZEN_CONTRACT,
        "source": {
            **source,
            "generator_sha256": sha256_file(compiler_path()),
            "requirements_lock_sha256": sha256_file(requirements_lock_path()),
        },
        "artifact": {
            "format": "elf_relocatable_with_embedded_cubin",
            **{
                name: {"sha256": sha256_bytes(artifacts[name]), "size_bytes": len(artifacts[name])}
                for name in ARTIFACT_FILES
            },
        },
    }


def package_candidate(
    *,
    raw_aot_dir: Path,
    compile_metadata_path: Path,
    output_dir: Path,
    source: dict[str, Any],
) -> Path:
    if output_dir.exists():
        raise ContractError(f"refusing to overwrite existing output directory: {output_dir}")
    metadata = read_json(compile_metadata_path)
    validate_compile_metadata(metadata, source)

    aot = metadata["aot"]
    _require_equal(
        aot["function_prefix"], FROZEN_CONTRACT["abi"]["function_prefix"],
        "generated function prefix",
    )
    paths = {
        "header": raw_aot_dir / aot["header"],
        "object": raw_aot_dir / aot["object"],
        "native_runtime": Path(aot["native_runtime"]),
    }
    artifacts = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ContractError(f"AOT {name} is missing: {path}")
        artifacts[name] = path.read_bytes()
    manifest = build_manifest(artifacts=artifacts, source=source)
    for name in ARTIFACT_FILES:
        for field in ("sha256", "size_bytes"):
            _require_equal(
                aot[f"{name}_{field}"], manifest["artifact"][name][field],
                f"AOT {name} {field}",
            )

    output_dir.mkdir(parents=True)
    for name, filename in ARTIFACT_FILES.items():
        (output_dir / filename).write_bytes(artifacts[name])
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def validate_manifest(
    manifest_path: Path,
    *,
    flashinfer_dir: Path,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    lock, lock_hash = load_source_lock()
    verify_flashinfer_base(flashinfer_dir)
    for path in (manifest_path.parent, *manifest_path.parent.parents):
        if path.is_symlink():
            raise ContractError(f"symlink in candidate path: {path}")
    if ".." in manifest_path.parts:
        raise ContractError("candidate path traversal")
    artifacts = {}
    for name, filename in {"manifest": "manifest.json", **ARTIFACT_FILES}.items():
        path = manifest_path.parent / filename
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"candidate must be a regular file: {path}")
        if name != "manifest":
            data = path.read_bytes()
            if not data:
                raise ContractError(f"empty candidate artifact: {path}")
            artifacts[name] = data
    # Reconstruct the whole document from project pins and actual artifact bytes,
    # so extra/missing fields cannot be supplied by the candidate itself.
    expected = build_manifest(artifacts=artifacts, source={
        "flashinfer_commit": FROZEN_FLASHINFER_COMMIT,
        "kernel_source_sha256": lock["patched_kernel_sha256"],
        "source_lock_sha256": lock_hash,
    })
    _require_equal(manifest, expected, "complete candidate contract")
    return manifest


def default_flashinfer_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "flashinfer"
