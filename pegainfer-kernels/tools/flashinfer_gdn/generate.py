#!/usr/bin/env python3
"""Generate and package the production FlashInfer GDN SM120 AOT object."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from artifact_contract import (
    VARIANT,
    ContractError,
    default_flashinfer_dir,
    package_candidate,
    prepare_flashinfer_source,
    validate_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--flashinfer-dir", type=Path, default=default_flashinfer_dir())
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
            source = prepare_flashinfer_source(args.flashinfer_dir, prepared)
            staged = temp / "candidate"
            compiler = Path(__file__).with_name("compile_sm120.py")
            raw_dir = temp / "raw"
            metadata_path = raw_dir / "compile-metadata.json"
            subprocess.run(
                [
                    str(args.python),
                    str(compiler),
                    "--variant",
                    VARIANT,
                    "--flashinfer-dir",
                    str(prepared),
                    "--base-flashinfer-dir",
                    str(args.flashinfer_dir),
                    "--aot-out",
                    str(raw_dir),
                    "--metadata-out",
                    str(metadata_path),
                ],
                check=True,
            )
            package_candidate(
                raw_aot_dir=raw_dir,
                compile_metadata_path=metadata_path,
                output_dir=staged,
                source=source,
            )
            validate_manifest(
                staged / "manifest.json", flashinfer_dir=args.flashinfer_dir
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged), output)
    except (ContractError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: generation failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"candidate": str(output), "variant": VARIANT}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
