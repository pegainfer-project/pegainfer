#!/usr/bin/env python3
"""Generate the final production FlashInfer GDN SM120 AOT bundle."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import re
import sys
import tempfile
import subprocess
import types
from pathlib import Path

from artifact_contract import (
    FORBIDDEN_TMA_CLUSTER_LOAD,
    FROZEN_CONTRACT,
    GEOMETRY,
    ARTIFACT_FILES,
    PINNED_TOOLCHAIN,
    ContractError,
    build_manifest,
    prepare_flashinfer_source,
    validate_manifest,
    TARGET_ARCH,
    VARIANT,
    _COMPILER_PATH,
    _REQUIREMENTS_LOCK_PATH,
    sha256_file,
    write_json,
)


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required generation package is missing: {distribution}") from exc


def ptx_metadata(ptx: str) -> dict[str, str]:
    compiler_match = re.search(
        r"Cuda compilation tools, release\s+([0-9.]+),\s+V([0-9.]+)", ptx
    )
    isa_match = re.search(r"^\.version[ \t]+([0-9.]+)[ \t\r]*$", ptx, re.MULTILINE)
    if not compiler_match or not isa_match:
        raise RuntimeError("cannot derive CUDA compiler/PTX ISA from generated PTX")
    return {
        "ptx_compiler_release": compiler_match.group(1),
        "ptx_compiler_version": compiler_match.group(2),
        "ptx_isa": isa_match.group(1),
    }


def read_compiled_ptx(compiled: object) -> str:
    artifact = getattr(compiled, "__ptx__", None)
    if isinstance(artifact, str) and ".version" in artifact:
        return artifact
    if isinstance(artifact, str) and Path(artifact).is_file():
        return Path(artifact).read_text(encoding="utf-8")
    raise RuntimeError("CuTe compile did not expose a readable PTX artifact")


def find_static_cuda_dialect_runtime() -> Path:
    """Locate the runtime archive shipped by the pinned CuTe DSL wheel."""
    import cutlass

    cutlass_file = Path(cutlass.__file__).resolve()
    roots = {
        Path(entry).resolve()
        for entry in sys.path
        if entry and ("site-packages" in entry or "dist-packages" in entry)
    }
    # Restrict runtime discovery to the installed wheel/package trees.
    roots.update((cutlass_file.parent, cutlass_file.parent.parent))
    matches: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        matches.extend(root.glob("**/libcuda_dialect_runtime_static.a"))
    unique = sorted({path.resolve() for path in matches if path.is_file()})
    if len(unique) != 1:
        raise RuntimeError(
            "expected exactly one libcuda_dialect_runtime_static.a in the pinned "
            f"generation environment, found {[str(path) for path in unique]}"
        )
    return unique[0]


def import_frozen_kernel(flashinfer_dir: Path):
    """Import only the frozen kernel package, without FlashInfer's top-level API."""
    package_paths = {
        "flashinfer": flashinfer_dir / "flashinfer",
        "flashinfer.gdn_kernels": flashinfer_dir / "flashinfer" / "gdn_kernels",
        "flashinfer.gdn_kernels.delta_rule_dsl": (
            flashinfer_dir / "flashinfer" / "gdn_kernels" / "delta_rule_dsl"
        ),
    }
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package

    # delta_rule_sm120 imports these helpers for its public Torch wrapper. The
    # offline fake-tensor compiler never calls them, so avoid importing the
    # rest of FlashInfer and its unrelated pynvml/JIT dependencies.
    utils = types.ModuleType("flashinfer.utils")

    def generation_only_stub(*_args, **_kwargs):
        raise RuntimeError("runtime-only FlashInfer helper called by offline compiler")

    utils.get_device_sm_count = generation_only_stub
    utils._get_cache_buf = generation_only_stub
    sys.modules["flashinfer.utils"] = utils

    cache_module = importlib.import_module(
        "flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache"
    )
    kernel_module = importlib.import_module(
        "flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120"
    )
    return cache_module.cached_compile, kernel_module._FullyFusedDeltaRuleSm120


def compile_kernel(flashinfer_dir: Path) -> tuple[object, str]:
    import cutlass
    import cutlass.cute as cute

    cached_compile, kernel_type = import_frozen_kernel(flashinfer_dir)

    h_q = GEOMETRY["h_q"]
    h_k = GEOMETRY["h_k"]
    h_v = GEOMETRY["h_v"]
    d = GEOMETRY["head_dim"]
    t = cute.sym_int()
    flat_tokens = cute.sym_int()
    workspace_bytes = cute.sym_int()
    cu_count = cute.sym_int()

    q = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (t, d, h_q), stride=(h_q * d, 1, d), assumed_align=16
    )
    k = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (d, t, h_k), stride=(1, h_k * d, d), assumed_align=16
    )
    v = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (d, t, h_v), stride=(1, h_v * d, d), assumed_align=16
    )
    o = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (d, t, h_v), stride=(1, h_v * d, d), assumed_align=16
    )
    alpha = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (flat_tokens,), assumed_align=16)
    beta = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (flat_tokens,), assumed_align=16)
    state = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (h_v * d * d,), assumed_align=16)
    init_state = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (h_v * d * d,), assumed_align=16)
    workspace = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (workspace_bytes,), assumed_align=128)
    cu_seqlens = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (cu_count,), assumed_align=8)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = kernel_type(
        needs_alpha=True,
        needs_beta=True,
        needs_init_state=True,
        needs_checkpointing=False,
        dtype=cutlass.BFloat16,
    )
    args = (
        q,
        k,
        v,
        o,
        alpha,
        beta,
        state,
        init_state,
        None,
        None,
        workspace,
        cu_seqlens,
        cutlass.Float32(1.0 / (d**0.5)),
        cutlass.Int32(h_q),
        cutlass.Int32(h_k),
        cutlass.Int32(h_v),
        cutlass.Int32(max(h_q, h_v)),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(0),
        cutlass.Int32(max(h_q, h_v)),
        stream,
    )
    compiled = cached_compile(kernel, *args, compile_options=(cute.GPUArch(TARGET_ARCH),))
    ptx = read_compiled_ptx(compiled)
    if FORBIDDEN_TMA_CLUSTER_LOAD in ptx:
        raise RuntimeError("upstream SM120 TMA workaround was not applied")
    return compiled, ptx


def export_kernel(
    flashinfer_dir: Path, work: Path, *, upstream_layout: bool = False
) -> tuple[dict[str, Path], dict]:
    source_dir = work / "source"
    source = prepare_flashinfer_source(flashinfer_dir, source_dir, upstream_layout=upstream_layout)
    compiled, ptx = compile_kernel(source_dir)
    toolchain = {
        "python": sys.version.split()[0],
        **ptx_metadata(ptx),
        "cutlass_dsl": package_version("nvidia-cutlass-dsl"),
        "cutlass_dsl_libs_base": package_version("nvidia-cutlass-dsl-libs-base"),
        "torch": package_version("torch"),
        "cuda_python": package_version("cuda-python"),
        "cuda_bindings": package_version("cuda-bindings"),
        "cuda_pathfinder": package_version("cuda-pathfinder"),
    }
    if toolchain != PINNED_TOOLCHAIN:
        raise ContractError(f"generation toolchain mismatch: expected {PINNED_TOOLCHAIN}, got {toolchain}")
    prefix = ("pegainfer_qwen35_gdn_upstream_hvk" if upstream_layout
              else FROZEN_CONTRACT["abi"]["function_prefix"])
    raw = work / "aot"
    raw.mkdir()
    compiled.export_to_c(str(raw), prefix, prefix)
    paths = {
        "header": raw / f"{prefix}.h",
        "object": raw / f"{prefix}.o",
        "native_runtime": find_static_cuda_dialect_runtime(),
    }
    for name, path in paths.items():
        if not path.is_file() or not path.stat().st_size:
            raise ContractError(f"missing or empty AOT {name}: {path}")
    provenance = {
        **source,
        "generator_sha256": sha256_file(_COMPILER_PATH),
        "requirements_lock_sha256": sha256_file(_REQUIREMENTS_LOCK_PATH),
        "toolchain": toolchain,
        "aot": {
            "function_prefix": prefix,
            **{name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
               for name, path in paths.items()},
        },
    }
    return paths, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.absolute()
    try:
        if output.exists() or output.is_symlink():
            raise ContractError(f"refusing to overwrite existing output directory: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".gdn-aot-", dir=output.parent) as temporary:
            work = Path(temporary)
            paths, observed = export_kernel(args.flashinfer_dir.resolve(), work)
            artifacts = {name: path.read_bytes() for name, path in paths.items()}
            manifest = build_manifest(artifacts=artifacts, source={
                key: observed[key]
                for key in ("flashinfer_commit", "kernel_source_sha256", "source_lock_sha256")
            })
            if manifest["artifact"] != {"format": "elf_relocatable_with_embedded_cubin",
                                        **{name: observed["aot"][name] for name in ARTIFACT_FILES}}:
                raise ContractError("exported artifacts changed before packaging")
            staged = work / "candidate"
            staged.mkdir()
            for name, filename in ARTIFACT_FILES.items():
                (staged / filename).write_bytes(artifacts[name])
            write_json(staged / "manifest.json", manifest)
            validate_manifest(staged / "manifest.json", flashinfer_dir=args.flashinfer_dir)
            if output.exists() or output.is_symlink():
                raise ContractError(f"output appeared during generation: {output}")
            staged.rename(output)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"error: generation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"candidate": str(output), "variant": VARIANT}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
