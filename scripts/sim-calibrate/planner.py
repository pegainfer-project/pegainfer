#!/usr/bin/env python3
"""Build and print a deterministic JSON-matrix calibration campaign manifest.

Example:
    python3 scripts/sim-calibrate/planner.py --spec campaign.json \
        --output /tmp/campaign/manifest.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from campaign import build_json_manifest
from manifest import ManifestError, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="JSON campaign specification")
    parser.add_argument("--output", type=Path, help="write the manifest to this path")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without contacting a target")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_json_manifest(args.spec)
        if args.output is not None:
            write_manifest(args.output, manifest)
        if args.dry_run or args.output is None:
            print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
        else:
            print(f"campaign_id={manifest['campaign_id']}")
            print(f"cells={manifest['workload']['cell_count']}")
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
