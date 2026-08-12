#!/usr/bin/env python3
"""Generate an unpatched upstream-HVK SM120 artifact for Stage 7 A/B only.

This intentionally stays separate from ``generate.py``: production artifacts
must retain the pinned OpenInfer HKV patch, while this artifact answers whether
an observed numeric tail is already present in the frozen upstream kernel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from artifact_contract import (
    ABSOLUTE_PATH_PATTERNS,
    DTYPES,
    FROZEN_FLASHINFER_COMMIT,
    PINNED_TOOLCHAIN,
    TARGET_ARCH,
    expected_spec,
    inspect_kernel_source,
    normalize_ptx,
    parse_entry_symbols,
    sha256_bytes,
    sha256_file,
    verify_flashinfer_base,
    write_json,
)
from compile_sm120 import (
    compile_variant,
    host_cuda_toolkit_version,
    package_version,
    ptx_metadata,
    validate_with_ptxas,
)


UPSTREAM_KERNEL_SHA256 = "dafd93ceeafeee0ac024a8405f40da69edae33b7f99fc6b97f670b41a85e8cc6"
ZERO_SHA256 = "0" * 64


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-dir", required=True, type=Path)
    parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda-12.8"))
    parser.add_argument("--ptxas", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output directory: {output}")

    flashinfer_dir = args.flashinfer_dir.resolve()
    commit = verify_flashinfer_base(flashinfer_dir)
    source = inspect_kernel_source(flashinfer_dir, commit)
    if source["kernel_source_sha256"] != UPSTREAM_KERNEL_SHA256:
        raise RuntimeError(
            "unpatched upstream kernel hash mismatch: "
            f"expected {UPSTREAM_KERNEL_SHA256}, got {source['kernel_source_sha256']}"
        )
    kernel_text = (flashinfer_dir / source["workspace"]["source"]).read_text(
        encoding="utf-8"
    )
    if kernel_text.count("order=(0, 1, 2, 3)") != 2:
        raise RuntimeError("upstream source does not contain both frozen HVK layouts")

    variant = "operator_hv48"
    spec = expected_spec(variant)
    ptx = normalize_ptx(compile_variant(variant, flashinfer_dir))
    ptxas_version = validate_with_ptxas(ptx, args.ptxas)
    if any(pattern.search(ptx) for pattern in ABSOLUTE_PATH_PATTERNS):
        raise RuntimeError("diagnostic PTX contains an absolute build path")
    symbols = parse_entry_symbols(ptx)
    if len(symbols) != 1:
        raise RuntimeError(f"expected one PTX entry symbol, got {symbols}")

    toolchain = {
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
    }
    if toolchain != PINNED_TOOLCHAIN:
        raise RuntimeError(
            "diagnostic generation toolchain differs from the production artifact: "
            f"expected {PINNED_TOOLCHAIN}, got {toolchain}"
        )

    ptx_bytes = ptx.encode("utf-8")
    geometry = spec["geometry"]
    manifest = {
        "schema_version": 1,
        "artifact_kind": "flashinfer_cute_gdn_prefill_ptx",
        "variant": variant,
        "target": {"arch": TARGET_ARCH, "driver_jit_target": "compute_120a"},
        "geometry": geometry,
        "dtypes": DTYPES,
        "tokens": {"extent": "dynamic", "minimum": 1, "divisibility": 1},
        "abi": {
            "entry_symbol": symbols[0],
            "geometry_binding": "manifest_guarded_runtime_head_parameters",
            "q_view": {
                "shape": ["T", 128, geometry["h_q"]],
                "stride": [geometry["h_q"] * 128, 1, 128],
            },
            "k_view": {
                "shape": [128, "T", geometry["h_k"]],
                "stride": [1, geometry["h_k"] * 128, 128],
            },
            "v_view": {
                "shape": [128, "T", geometry["h_v"]],
                "stride": [1, geometry["h_v"] * 128, 128],
            },
            "o_view": {
                "shape": [128, "T", geometry["h_v"]],
                "stride": [1, geometry["h_v"] * 128, 128],
            },
            "state_layout": "upstream_hvk_k_contiguous",
        },
        "workspace": source["workspace"],
        "source": {
            "flashinfer_commit": FROZEN_FLASHINFER_COMMIT,
            "kernel_source_sha256": UPSTREAM_KERNEL_SHA256,
            "generator_sha256": sha256_file(Path(__file__)),
            "requirements_lock_sha256": sha256_file(
                Path(__file__).with_name("requirements-cu13.lock")
            ),
            "patch_set_sha256": ZERO_SHA256,
            "hkv_state_index_patch_sha256": ZERO_SHA256,
            "hkv_state_index_patch_applied": False,
        },
        "toolchain": toolchain,
        "artifact": {
            "file": "kernel.ptx",
            "format": "ptx",
            "sha256": sha256_bytes(ptx_bytes),
            "size_bytes": len(ptx_bytes),
            "entry_symbols": symbols,
            "absolute_path_scan": "passed",
        },
        "distribution": {
            "strategy": "stage7_upstream_hvk_diagnostic",
            "serving_requires_python": False,
            "serving_requires_cute_dsl": False,
            "cuda_driver_jit_required": True,
            "production_candidate_geometry": False,
            "production_eligible": False,
            "production_blocker": "diagnostic-only unpatched upstream HVK state layout",
        },
    }

    output.mkdir(parents=True)
    (output / "kernel.ptx").write_bytes(ptx_bytes)
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "manifest": str(output / "manifest.json"),
                "ptx_sha256": manifest["artifact"]["sha256"],
                "state_layout": manifest["abi"]["state_layout"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
