#!/usr/bin/env python3
"""Package and validate FlashInfer CuTe GDN SM120 native AOT artifacts.

This module intentionally uses only the Python standard library. CuTe and
PyTorch are generation-time dependencies isolated in ``compile_sm120.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
TARGET_ARCH = "sm_120a"
FROZEN_FLASHINFER_COMMIT = "a0efa0adfe49bb836ab1a147d6572980b870f3d4"
SUPPORTED_GEOMETRIES = {
    "qwen35_4b_candidate": {"h_q": 16, "h_k": 16, "h_v": 32, "head_dim": 128},
}
DTYPES = {
    "q": "bfloat16",
    "k": "bfloat16",
    "v": "bfloat16",
    "o": "bfloat16",
    "alpha": "float32",
    "beta": "float32",
    "state": "float32",
    "cu_seqlens": "int64",
    "workspace": "uint8",
}
PINNED_TOOLCHAIN = {
    "python": "3.12.3",
    "ptx_compiler_release": "13.1",
    "ptx_compiler_version": "13.1.66",
    "ptx_isa": "9.1",
    "cutlass_dsl": "4.5.0",
    "cutlass_dsl_libs_base": "4.5.0",
    "torch": "2.7.1",
    "cuda_python": "13.0.1",
    "cuda_bindings": "13.0.3",
    "cuda_pathfinder": "1.6.0",
}
WORKSPACE_SOURCE = (
    "flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py"
)
FORBIDDEN_TMA_CLUSTER_LOAD = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
    "mbarrier::complete_tx::bytes.L2::cache_hint"
)
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?:^|[\s\"'=])/(?:home|mnt|tmp|Users|workspace|build)/[^\s\"']+"),
    re.compile(r"[A-Za-z]:\\[^\s\"']+"),
)
ENTRY_RE = re.compile(
    r"(?:\.visible\s+)?\.entry\s+([A-Za-z_$][A-Za-z0-9_$.]*)\s*\(",
    re.MULTILINE,
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


def load_source_lock(path: Path | None = None) -> tuple[dict[str, Any], str]:
    path = path or source_lock_path()
    lock = read_json(path)
    if lock.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("source lock schema_version mismatch")
    if lock.get("flashinfer_commit") != FROZEN_FLASHINFER_COMMIT:
        raise ContractError("source lock FlashInfer commit mismatch")
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
    hkv = lock.get("hkv_state_index_patch")
    expected_hkv = {
        "applied": True,
        "state_layout": "openinfer_hkv_v_contiguous",
        "ordered_layout": [1, 0, 2, 3],
    }
    if hkv != expected_hkv:
        raise ContractError("HKV state-index patch metadata mismatch")
    expected_export = {
        "grid_x": "cutlass.Int32",
        "stream": "cuda.CUstream",
        "purpose": "host-only type annotations required by official export_to_c",
    }
    if lock.get("aot_export_patch") != expected_export:
        raise ContractError("AOT export annotation metadata mismatch")
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


def inspect_kernel_source(source_dir: Path, commit: str) -> dict[str, Any]:
    kernel_path = source_dir / WORKSPACE_SOURCE
    source = kernel_path.read_text(encoding="utf-8")
    workspace_match = re.search(
        r"workspace_size\s*=\s*get_device_sm_count\(q\.device\)\s*\*\s*(\d+)",
        source,
    )
    alignment_match = re.search(
        r"from_dlpack\(tensormaps_t,\s*assumed_align\s*=\s*(\d+)\)",
        source,
    )
    target_match = re.search(r'cute\.GPUArch\("([^"]+)"\)', source)
    if not workspace_match or not alignment_match or not target_match:
        raise ContractError("cannot derive workspace/target metadata from frozen kernel source")
    target = target_match.group(1)
    if target != TARGET_ARCH:
        raise ContractError(f"kernel source target mismatch: expected {TARGET_ARCH}, got {target}")

    return {
        "flashinfer_commit": commit,
        "kernel_source_sha256": sha256_file(kernel_path),
        "workspace": {
            "kind": "per_sm",
            "bytes_per_sm": int(workspace_match.group(1)),
            "alignment_bytes": int(alignment_match.group(1)),
            "formula": "sm_count * bytes_per_sm",
            "source": WORKSPACE_SOURCE,
        },
        "target_arch": target,
    }


def prepare_flashinfer_source(flashinfer_dir: Path, destination: Path) -> dict[str, Any]:
    commit = verify_flashinfer_base(flashinfer_dir)
    lock, _ = load_source_lock()
    if destination.exists():
        raise ContractError(f"refusing to overwrite prepared source: {destination}")
    shutil.copytree(flashinfer_dir / "flashinfer", destination / "flashinfer")
    for patch in lock["patches"]:
        patch_path = source_lock_path().parent / patch["path"]
        result = subprocess.run(
            ["git", "apply", "--unsafe-paths", str(patch_path)],
            cwd=destination,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ContractError(f"failed to apply HKV patch: {detail}")
    source = inspect_kernel_source(destination, commit)
    _require_equal(
        source["kernel_source_sha256"],
        lock["patched_kernel_sha256"],
        "prepared HKV kernel hash",
    )
    kernel_text = (destination / WORKSPACE_SOURCE).read_text(encoding="utf-8")
    if kernel_text.count("order=(1, 0, 2, 3)") != 2:
        raise ContractError("prepared source does not contain both HKV ordered layouts")
    return source


def verify_prepared_flashinfer_source(
    source_dir: Path, flashinfer_dir: Path
) -> dict[str, Any]:
    commit = verify_flashinfer_base(flashinfer_dir)
    lock, _ = load_source_lock()
    source = inspect_kernel_source(source_dir, commit)
    _require_equal(
        source["kernel_source_sha256"],
        lock["patched_kernel_sha256"],
        "prepared HKV kernel hash",
    )
    return source


def verify_flashinfer_source(flashinfer_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="openinfer-gdn-hkv-source-") as temp_name:
        return prepare_flashinfer_source(flashinfer_dir, Path(temp_name) / "patched")


def normalize_ptx(ptx: str) -> str:
    """Normalize harmless path/debug text without changing PTX instructions."""
    normalized_lines: list[str] = []
    file_directive = re.compile(r'^(\s*\.file\s+\d+\s+")([^"]+)(".*)$')
    for raw_line in ptx.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.rstrip()
        match = file_directive.match(line)
        if match:
            name = Path(match.group(2).replace("\\", "/")).name
            line = f"{match.group(1)}{name}{match.group(3)}"
        normalized_lines.append(line)
    return "\n".join(normalized_lines) + "\n"


def leaked_absolute_paths(text: str) -> list[str]:
    leaks: set[str] = set()
    for pattern in ABSOLUTE_PATH_PATTERNS:
        leaks.update(match.group(0).lstrip(" \t\"'=") for match in pattern.finditer(text))
    return sorted(leaks)


def parse_entry_symbols(ptx: str) -> list[str]:
    return sorted(set(ENTRY_RE.findall(ptx)))


def expected_spec(variant: str) -> dict[str, Any]:
    try:
        geometry = SUPPORTED_GEOMETRIES[variant]
    except KeyError as exc:
        raise ContractError(f"unknown artifact variant: {variant}") from exc
    return {
        "variant": variant,
        "target_arch": TARGET_ARCH,
        "geometry": dict(geometry),
        "dtypes": dict(DTYPES),
        "tokens": {"extent": "dynamic", "minimum": 1, "divisibility": 1},
    }


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ContractError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def validate_compile_metadata(
    metadata: dict[str, Any], variant: str, source: dict[str, Any]
) -> None:
    spec = expected_spec(variant)
    for key in ("variant", "target_arch", "geometry", "dtypes", "tokens"):
        _require_equal(metadata.get(key), spec[key], f"compile metadata {key}")
    _require_equal(
        metadata.get("flashinfer_commit"),
        FROZEN_FLASHINFER_COMMIT,
        "compile metadata FlashInfer SHA",
    )
    _require_equal(
        metadata.get("kernel_source_sha256"),
        source["kernel_source_sha256"],
        "compile metadata kernel source hash",
    )
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
    _require_equal(metadata.get("workspace"), source["workspace"], "workspace metadata")
    aot = metadata.get("aot")
    if not isinstance(aot, dict):
        raise ContractError("compile metadata is missing AOT export metadata")
    expected_prefix = f"pegainfer_qwen35_gdn_{variant}"
    _require_equal(aot.get("function_prefix"), expected_prefix, "AOT function prefix")
    toolchain = metadata.get("toolchain")
    if not isinstance(toolchain, dict):
        raise ContractError("compile metadata is missing toolchain")
    _require_equal(toolchain, PINNED_TOOLCHAIN, "compile metadata toolchain")


def build_manifest(
    *,
    variant: str,
    header_name: str,
    header_bytes: bytes,
    object_name: str,
    object_bytes: bytes,
    runtime_name: str,
    runtime_bytes: bytes,
    compile_metadata: dict[str, Any],
    source: dict[str, Any],
    patch_set_sha256: str,
) -> dict[str, Any]:
    lock, _ = load_source_lock()
    patch_sha256 = lock["patches"][0]["sha256"]
    spec = expected_spec(variant)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "flashinfer_cute_gdn_prefill_aot_object",
        "variant": variant,
        "target": {"arch": TARGET_ARCH, "code_object": "embedded_cubin"},
        "geometry": spec["geometry"],
        "dtypes": spec["dtypes"],
        "tokens": spec["tokens"],
        "abi": {
            "version": 1,
            "function_prefix": compile_metadata["aot"]["function_prefix"],
            "geometry_binding": "stable_project_c_wrapper",
            "q_view": {"shape": ["T", 128, spec["geometry"]["h_q"]], "stride": [spec["geometry"]["h_q"] * 128, 1, 128]},
            "k_view": {"shape": [128, "T", spec["geometry"]["h_k"]], "stride": [1, spec["geometry"]["h_k"] * 128, 128]},
            "v_view": {"shape": [128, "T", spec["geometry"]["h_v"]], "stride": [1, spec["geometry"]["h_v"] * 128, 128]},
            "o_view": {"shape": [128, "T", spec["geometry"]["h_v"]], "stride": [1, spec["geometry"]["h_v"] * 128, 128]},
            "state_layout": "openinfer_hkv_v_contiguous",
        },
        "workspace": source["workspace"],
        "source": {
            "flashinfer_commit": FROZEN_FLASHINFER_COMMIT,
            "kernel_source_sha256": source["kernel_source_sha256"],
            "generator_sha256": compile_metadata["generator_sha256"],
            "requirements_lock_sha256": compile_metadata["requirements_lock_sha256"],
            "patch_set_sha256": patch_set_sha256,
            "hkv_state_index_patch_sha256": patch_sha256,
            "hkv_state_index_patch_applied": True,
        },
        "toolchain": compile_metadata["toolchain"],
        "artifact": {
            "format": "elf_relocatable_with_embedded_cubin",
            "header": {
                "file": header_name,
                "sha256": sha256_bytes(header_bytes),
                "size_bytes": len(header_bytes),
            },
            "object": {
                "file": object_name,
                "sha256": sha256_bytes(object_bytes),
                "size_bytes": len(object_bytes),
            },
            "native_runtime": {
                "file": runtime_name,
                "sha256": sha256_bytes(runtime_bytes),
                "size_bytes": len(runtime_bytes),
            },
        },
        "distribution": {
            "strategy": "release_bundle",
            "serving_requires_python": False,
            "serving_requires_cute_dsl": False,
            "cuda_driver_jit_required": False,
            "cute_runtime_linkage": "static",
            "production_eligible": True,
        },
    }


def package_variant(
    *,
    variant: str,
    raw_aot_dir: Path,
    compile_metadata_path: Path,
    output_dir: Path,
    flashinfer_dir: Path,
) -> Path:
    if output_dir.exists():
        raise ContractError(f"refusing to overwrite existing output directory: {output_dir}")
    source = verify_flashinfer_source(flashinfer_dir)
    _, patch_set_sha256 = load_source_lock()
    metadata = read_json(compile_metadata_path)
    validate_compile_metadata(metadata, variant, source)

    aot = metadata["aot"]
    header_path = raw_aot_dir / aot["header"]
    object_path = raw_aot_dir / aot["object"]
    if not header_path.is_file() or not object_path.is_file():
        raise ContractError("AOT export header/object is missing")
    header_bytes = header_path.read_bytes()
    object_bytes = object_path.read_bytes()
    runtime_path = Path(aot["native_runtime"])
    if not runtime_path.is_file():
        raise ContractError("CuTe static runtime archive is missing")
    runtime_bytes = runtime_path.read_bytes()
    _require_equal(aot["header_sha256"], sha256_bytes(header_bytes), "AOT header hash")
    _require_equal(aot["object_sha256"], sha256_bytes(object_bytes), "AOT object hash")
    _require_equal(aot["object_size_bytes"], len(object_bytes), "AOT object size")
    _require_equal(
        aot["native_runtime_sha256"],
        sha256_bytes(runtime_bytes),
        "CuTe static runtime hash",
    )
    _require_equal(
        aot["native_runtime_size_bytes"],
        len(runtime_bytes),
        "CuTe static runtime size",
    )

    output_dir.mkdir(parents=True)
    header_name = "kernel.h"
    object_name = "kernel.o"
    runtime_name = "libcuda_dialect_runtime_static.a"
    (output_dir / header_name).write_bytes(header_bytes)
    (output_dir / object_name).write_bytes(object_bytes)
    (output_dir / runtime_name).write_bytes(runtime_bytes)
    manifest = build_manifest(
        variant=variant,
        header_name=header_name,
        header_bytes=header_bytes,
        object_name=object_name,
        object_bytes=object_bytes,
        runtime_name=runtime_name,
        runtime_bytes=runtime_bytes,
        compile_metadata=metadata,
        source=source,
        patch_set_sha256=patch_set_sha256,
    )
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    validate_manifest(manifest_path, flashinfer_dir=flashinfer_dir)
    return manifest_path


def validate_manifest(
    manifest_path: Path,
    *,
    flashinfer_dir: Path | None = None,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    _require_equal(manifest.get("schema_version"), SCHEMA_VERSION, "schema_version")
    variant = expected_variant or manifest.get("variant")
    if not isinstance(variant, str):
        raise ContractError("manifest variant is missing")
    spec = expected_spec(variant)
    _require_equal(manifest.get("variant"), variant, "variant")
    _require_equal(manifest.get("target"), {"arch": TARGET_ARCH, "code_object": "embedded_cubin"}, "target")
    _require_equal(manifest.get("geometry"), spec["geometry"], "geometry")
    _require_equal(manifest.get("dtypes"), spec["dtypes"], "dtypes")
    _require_equal(manifest.get("tokens"), spec["tokens"], "dynamic token contract")

    source_manifest = manifest.get("source")
    if not isinstance(source_manifest, dict):
        raise ContractError("manifest source is missing")
    _require_equal(source_manifest.get("flashinfer_commit"), FROZEN_FLASHINFER_COMMIT, "FlashInfer SHA")
    lock, patch_set_sha256 = load_source_lock()
    _require_equal(source_manifest.get("patch_set_sha256"), patch_set_sha256, "patch-set hash")
    _require_equal(
        source_manifest.get("hkv_state_index_patch_sha256"),
        lock["patches"][0]["sha256"],
        "HKV patch hash",
    )
    _require_equal(source_manifest.get("hkv_state_index_patch_applied"), True, "HKV patch state")
    _require_equal(source_manifest.get("generator_sha256"), sha256_file(compiler_path()), "generator hash")
    _require_equal(
        source_manifest.get("requirements_lock_sha256"),
        sha256_file(requirements_lock_path()),
        "requirements lock hash",
    )

    if flashinfer_dir is not None:
        source = verify_flashinfer_source(flashinfer_dir)
        _require_equal(source_manifest.get("kernel_source_sha256"), source["kernel_source_sha256"], "kernel source hash")
        _require_equal(manifest.get("workspace"), source["workspace"], "workspace")
    workspace = manifest.get("workspace")
    if not isinstance(workspace, dict):
        raise ContractError("workspace is missing")
    if workspace.get("kind") != "per_sm" or workspace.get("formula") != "sm_count * bytes_per_sm":
        raise ContractError("workspace must be expressed as per-SM generation metadata")
    if workspace.get("bytes_per_sm") == 256 * 128:
        raise ContractError("workspace bytes_per_sm must not be the old guessed 256*128 allocation")

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise ContractError("artifact metadata is missing")
    _require_equal(artifact.get("format"), "elf_relocatable_with_embedded_cubin", "artifact format")
    for component in ("header", "object", "native_runtime"):
        entry = artifact.get(component)
        if not isinstance(entry, dict):
            raise ContractError(f"artifact {component} metadata is missing")
        name = entry.get("file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ContractError(f"artifact {component} file must be a relative basename")
        path = manifest_path.parent / name
        if not path.is_file():
            raise ContractError(f"artifact {component} file is missing: {path}")
        data = path.read_bytes()
        _require_equal(entry.get("size_bytes"), len(data), f"artifact {component} size")
        _require_equal(entry.get("sha256"), sha256_bytes(data), f"artifact {component} hash")
    manifest_toolchain = manifest.get("toolchain")
    if not isinstance(manifest_toolchain, dict):
        raise ContractError("manifest toolchain is missing")
    _require_equal(manifest_toolchain, PINNED_TOOLCHAIN, "manifest toolchain")
    abi = manifest.get("abi")
    if not isinstance(abi, dict):
        raise ContractError("ABI metadata is missing")
    _require_equal(abi.get("version"), 1, "stable C ABI version")
    _require_equal(abi.get("function_prefix"), f"pegainfer_qwen35_gdn_{variant}", "AOT function prefix")
    _require_equal(
        abi.get("geometry_binding"),
        "stable_project_c_wrapper",
        "geometry binding",
    )
    _require_equal(
        abi.get("state_layout"),
        "openinfer_hkv_v_contiguous",
        "state layout",
    )

    distribution = manifest.get("distribution")
    if not isinstance(distribution, dict):
        raise ContractError("distribution metadata is missing")
    for key in ("serving_requires_python", "serving_requires_cute_dsl"):
        _require_equal(distribution.get(key), False, f"distribution {key}")
    _require_equal(distribution.get("production_eligible"), True, "production eligibility")
    _require_equal(distribution.get("cuda_driver_jit_required"), False, "driver JIT policy")
    _require_equal(distribution.get("cute_runtime_linkage"), "static", "CuTe runtime linkage")
    _require_equal(distribution.get("strategy"), "release_bundle", "distribution strategy")
    return manifest


def validate_bundle(bundle_dir: Path, flashinfer_dir: Path | None = None) -> None:
    expected = set(SUPPORTED_GEOMETRIES)
    present = {path.parent.name for path in bundle_dir.glob("*/manifest.json")}
    _require_equal(present, expected, "bundle variants")
    manifests = [
        validate_manifest(
            bundle_dir / variant / "manifest.json",
            flashinfer_dir=flashinfer_dir,
            expected_variant=variant,
        )
        for variant in sorted(expected)
    ]
    if {manifest["tokens"]["extent"] for manifest in manifests} != {"dynamic"}:
        raise ContractError("all bundle variants must use dynamic T")
    bundle_path = bundle_dir / "bundle.json"
    bundle = read_json(bundle_path)
    _require_equal(bundle.get("schema_version"), SCHEMA_VERSION, "bundle schema_version")
    expected_entries = {
        variant: {
            "manifest": f"{variant}/manifest.json",
            "manifest_sha256": sha256_file(bundle_dir / variant / "manifest.json"),
        }
        for variant in sorted(expected)
    }
    _require_equal(bundle.get("variants"), expected_entries, "bundle manifest index")


def default_flashinfer_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "flashinfer"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source_parser = subparsers.add_parser("verify-source")
    source_parser.add_argument("--flashinfer-dir", type=Path, default=default_flashinfer_dir())
    manifest_parser = subparsers.add_parser("validate-manifest")
    manifest_parser.add_argument("manifest", type=Path)
    manifest_parser.add_argument("--flashinfer-dir", type=Path)
    bundle_parser = subparsers.add_parser("validate-bundle")
    bundle_parser.add_argument("bundle", type=Path)
    bundle_parser.add_argument("--flashinfer-dir", type=Path)
    args = parser.parse_args()

    try:
        if args.command == "verify-source":
            source = verify_flashinfer_source(args.flashinfer_dir)
            _, patch_hash = load_source_lock()
            print(json.dumps({**source, "patch_set_sha256": patch_hash}, indent=2, sort_keys=True))
        elif args.command == "validate-manifest":
            validate_manifest(args.manifest, flashinfer_dir=args.flashinfer_dir)
            print(f"validated {args.manifest}")
        else:
            validate_bundle(args.bundle, flashinfer_dir=args.flashinfer_dir)
            print(f"validated {args.bundle}")
    except ContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
