#!/usr/bin/env python3
"""Fetch only the model snapshots frozen in the project lock."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"], required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
lock = json.loads(Path(__file__).with_name("environment.lock.json").read_text())
revision = lock["models"][args.model]
snapshot_download(args.model, revision=revision, local_dir=args.output,
                  allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
Path(args.output, "pegainfer-provenance.json").write_text(
    json.dumps({"repo": args.model, "revision": revision}, indent=2) + "\n")
print(json.dumps({"model": args.model, "revision": revision, "path": args.output}))
