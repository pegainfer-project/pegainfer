#!/usr/bin/env python3
"""AOT-export the production FlashInfer GDN specialization to a C header/object."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import re
import sys
import types
from pathlib import Path

from artifact_contract import (
    FORBIDDEN_TMA_CLUSTER_LOAD,
    TARGET_ARCH,
    VARIANT,
    expected_spec,
    compiler_path,
    normalize_ptx,
    requirements_lock_path,
    sha256_file,
    verify_prepared_flashinfer_source,
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
    isa_match = re.search(r"^\.version\s+([0-9.]+)$", ptx, re.MULTILINE)
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
    # Stay inside the installed wheel/package tree. `Path.parents` eventually
    # reaches `/`; recursively globbing that root made generation appear hung.
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


def compile_variant(variant: str, flashinfer_dir: Path) -> tuple[object, str]:
    spec = expected_spec(variant)
    geometry = spec["geometry"]
    import cutlass
    import cutlass.cute as cute

    cached_compile, kernel_type = import_frozen_kernel(flashinfer_dir)

    h_q = geometry["h_q"]
    h_k = geometry["h_k"]
    h_v = geometry["h_v"]
    d = geometry["head_dim"]
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
    ptx = normalize_ptx(read_compiled_ptx(compiled))
    if FORBIDDEN_TMA_CLUSTER_LOAD in ptx:
        raise RuntimeError("upstream SM120 TMA workaround was not applied")
    return compiled, ptx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=(VARIANT,))
    parser.add_argument("--flashinfer-dir", required=True, type=Path)
    parser.add_argument("--base-flashinfer-dir", required=True, type=Path)
    parser.add_argument("--aot-out", required=True, type=Path)
    parser.add_argument("--metadata-out", required=True, type=Path)
    args = parser.parse_args()

    source = verify_prepared_flashinfer_source(
        args.flashinfer_dir, args.base_flashinfer_dir
    )
    compiled, ptx = compile_variant(args.variant, args.flashinfer_dir.resolve())
    prefix = f"pegainfer_qwen35_gdn_{args.variant}"
    args.aot_out.mkdir(parents=True, exist_ok=True)
    compiled.export_to_c(str(args.aot_out), prefix, prefix)
    header = args.aot_out / f"{prefix}.h"
    object_file = args.aot_out / f"{prefix}.o"
    if not header.is_file() or not object_file.is_file():
        raise RuntimeError("CuTe export_to_c did not produce the expected .h/.o pair")
    runtime_archive = find_static_cuda_dialect_runtime()
    metadata = {
        "flashinfer_commit": source["flashinfer_commit"],
        "kernel_source_sha256": source["kernel_source_sha256"],
        "source_lock_sha256": source["source_lock_sha256"],
        "generator_sha256": sha256_file(compiler_path()),
        "requirements_lock_sha256": sha256_file(requirements_lock_path()),
        "toolchain": {
            "python": sys.version.split()[0],
            **ptx_metadata(ptx),
            "cutlass_dsl": package_version("nvidia-cutlass-dsl"),
            "cutlass_dsl_libs_base": package_version("nvidia-cutlass-dsl-libs-base"),
            "torch": package_version("torch"),
            "cuda_python": package_version("cuda-python"),
            "cuda_bindings": package_version("cuda-bindings"),
            "cuda_pathfinder": package_version("cuda-pathfinder"),
        },
        "aot": {
            "function_prefix": prefix,
            "header": header.name,
            "header_sha256": sha256_file(header),
            "header_size_bytes": header.stat().st_size,
            "object": object_file.name,
            "object_sha256": sha256_file(object_file),
            "object_size_bytes": object_file.stat().st_size,
            "native_runtime": str(runtime_archive),
            "native_runtime_sha256": sha256_file(runtime_archive),
            "native_runtime_size_bytes": runtime_archive.stat().st_size,
        },
    }
    write_json(args.metadata_out, metadata)
    print(json.dumps({"variant": args.variant, "aot": str(args.aot_out), "metadata": str(args.metadata_out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
