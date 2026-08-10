#!/usr/bin/env python3
"""Generate and package both frozen FlashInfer GDN SM120 PTX variants."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from artifact_contract import (
    SUPPORTED_GEOMETRIES,
    ContractError,
    default_flashinfer_dir,
    package_variant,
    prepare_flashinfer_source,
    sha256_file,
    validate_bundle,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--flashinfer-dir", type=Path, default=default_flashinfer_dir())
    parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda-12.8"))
    parser.add_argument("--ptxas", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("target/flashinfer-gdn-sm120"))
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        print(f"error: refusing to overwrite existing output directory: {output}", file=sys.stderr)
        return 2
    try:
        with tempfile.TemporaryDirectory(prefix="openinfer-gdn-sm120-") as temp_name:
            temp = Path(temp_name)
            prepared = temp / "patched-flashinfer"
            prepare_flashinfer_source(args.flashinfer_dir, prepared)
            staged = temp / "bundle"
            compiler = Path(__file__).with_name("compile_sm120.py")
            for variant in sorted(SUPPORTED_GEOMETRIES):
                raw_dir = temp / "raw" / variant
                ptx_path = raw_dir / "kernel.ptx"
                metadata_path = raw_dir / "compile-metadata.json"
                subprocess.run(
                    [
                        str(args.python),
                        str(compiler),
                        "--variant",
                        variant,
                        "--flashinfer-dir",
                        str(prepared),
                        "--base-flashinfer-dir",
                        str(args.flashinfer_dir),
                        "--cuda-root",
                        str(args.cuda_root),
                        "--ptxas",
                        str(args.ptxas),
                        "--ptx-out",
                        str(ptx_path),
                        "--metadata-out",
                        str(metadata_path),
                    ],
                    check=True,
                )
                package_variant(
                    variant=variant,
                    raw_ptx_path=ptx_path,
                    compile_metadata_path=metadata_path,
                    output_dir=staged / variant,
                    flashinfer_dir=args.flashinfer_dir,
                )

            bundle = {
                "schema_version": 1,
                "variants": {
                    variant: {
                        "manifest": f"{variant}/manifest.json",
                        "manifest_sha256": sha256_file(staged / variant / "manifest.json"),
                    }
                    for variant in sorted(SUPPORTED_GEOMETRIES)
                },
            }
            write_json(staged / "bundle.json", bundle)
            validate_bundle(staged, flashinfer_dir=args.flashinfer_dir)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged), output)
    except (ContractError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: generation failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"bundle": str(output), "variants": sorted(SUPPORTED_GEOMETRIES)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
