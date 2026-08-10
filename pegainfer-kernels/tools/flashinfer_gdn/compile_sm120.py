#!/usr/bin/env python3
"""Offline-compile one frozen FlashInfer GDN specialization to patched PTX."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

from artifact_contract import (
    DTYPES,
    FORBIDDEN_TMA_CLUSTER_LOAD,
    TARGET_ARCH,
    expected_spec,
    normalize_ptx,
    parse_entry_symbols,
    compiler_path,
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


def executable_version(executable: Path) -> str:
    result = subprocess.run(
        [str(executable), "--version"], check=True, capture_output=True, text=True
    )
    combined = result.stdout + result.stderr
    match = re.search(r"release\s+([0-9.]+)", combined)
    if not match:
        raise RuntimeError(f"cannot parse CUDA version from {executable}")
    return match.group(1)


def host_cuda_toolkit_version(cuda_root: Path) -> str:
    nvcc = cuda_root / "bin" / "nvcc"
    return executable_version(nvcc)


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


def validate_with_ptxas(ptx: str, ptxas: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="openinfer-gdn-ptxas-") as temp_name:
        temp = Path(temp_name)
        ptx_path = temp / "kernel.ptx"
        cubin_path = temp / "kernel.cubin"
        ptx_path.write_text(ptx, encoding="utf-8")
        subprocess.run(
            [str(ptxas), "-arch=sm_120a", str(ptx_path), "-o", str(cubin_path)],
            check=True,
        )
        if not cubin_path.is_file() or cubin_path.stat().st_size == 0:
            raise RuntimeError("ptxas did not produce a non-empty validation cubin")
    return executable_version(ptxas)


def read_compiled_ptx(compiled: object) -> str:
    artifact = getattr(compiled, "_flat_patched_ptx", None)
    if artifact is None:
        artifact = getattr(compiled, "__ptx__", None)
    if isinstance(artifact, str) and os.path.isfile(artifact):
        return Path(artifact).read_text(encoding="utf-8")
    if isinstance(artifact, str) and ".version" in artifact:
        return artifact
    raise RuntimeError("CuTe compile did not expose a readable PTX artifact")


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


def compile_variant(variant: str, flashinfer_dir: Path) -> str:
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
        max(h_q, h_v),
        stream,
    )
    compiled = cached_compile(kernel, *args, compile_options=(cute.GPUArch(TARGET_ARCH),))
    ptx = normalize_ptx(read_compiled_ptx(compiled))
    if FORBIDDEN_TMA_CLUSTER_LOAD in ptx:
        raise RuntimeError("upstream SM120 TMA workaround was not applied")
    symbols = parse_entry_symbols(ptx)
    if len(symbols) != 1:
        raise RuntimeError(f"expected exactly one PTX entry symbol, got {symbols}")
    return ptx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("qwen35_4b_candidate", "operator_hv48"))
    parser.add_argument("--flashinfer-dir", required=True, type=Path)
    parser.add_argument("--base-flashinfer-dir", required=True, type=Path)
    parser.add_argument("--cuda-root", required=True, type=Path)
    parser.add_argument("--ptxas", required=True, type=Path)
    parser.add_argument("--ptx-out", required=True, type=Path)
    parser.add_argument("--metadata-out", required=True, type=Path)
    args = parser.parse_args()

    source = verify_prepared_flashinfer_source(
        args.flashinfer_dir, args.base_flashinfer_dir
    )
    spec = expected_spec(args.variant)
    ptx = compile_variant(args.variant, args.flashinfer_dir.resolve())
    ptxas_version = validate_with_ptxas(ptx, args.ptxas)
    args.ptx_out.parent.mkdir(parents=True, exist_ok=True)
    args.ptx_out.write_text(ptx, encoding="utf-8")
    metadata = {
        **spec,
        "flashinfer_commit": source["flashinfer_commit"],
        "kernel_source_sha256": source["kernel_source_sha256"],
        "generator_sha256": sha256_file(compiler_path()),
        "requirements_lock_sha256": sha256_file(requirements_lock_path()),
        "workspace": source["workspace"],
        "toolchain": {
            "python": sys.version.split()[0],
            "host_cuda_toolkit": host_cuda_toolkit_version(args.cuda_root),
            "ptxas": ptxas_version,
            **ptx_metadata(ptx),
            "cutlass_dsl": package_version("nvidia-cutlass-dsl"),
            "cutlass_dsl_libs_base": package_version("nvidia-cutlass-dsl-libs-base"),
            "cuda_nvcc_package": package_version("nvidia-cuda-nvcc-cu12"),
            "torch": package_version("torch"),
            "cuda_python": package_version("cuda-python"),
            "cuda_bindings": package_version("cuda-bindings"),
        },
    }
    write_json(args.metadata_out, metadata)
    print(json.dumps({"variant": args.variant, "ptx": str(args.ptx_out), "metadata": str(args.metadata_out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
