#!/usr/bin/env python3
"""Compile a named JSON campaign into immutable plans and a zero-cost bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from artifacts import ArtifactError, write_compiled_bundle
from manifest import ManifestError, compile_manifest, sha256_json
from profile import ProfileError
from request_plan import PlanError, build_plan
from vllm_bench import VllmBenchError, inspect_binary

TOKENIZER_FILES = (
    "tokenizer_config.json", "tokenizer.json", "tokenizer.model", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt", "config.json", "chat_template.jinja",
)


def _file_digest(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            checksum = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
            return checksum.hexdigest()
    except OSError as exc:
        raise ManifestError(f"cannot read compiler input {path}: {exc}") from exc


def load_json_campaign(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for name, value in pairs:
            if name in result:
                raise ManifestError(f"duplicate campaign JSON key: {name}")
            result[name] = value
        return result

    try:
        with path.open(encoding="utf-8") as stream:
            spec = json.load(stream, object_pairs_hook=unique_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"failed to read campaign {path}: {exc}") from exc
    if not isinstance(spec, dict):
        raise ManifestError("campaign JSON must be an object")
    if not isinstance(spec.get("workload"), dict) or "matrices" not in spec["workload"]:
        raise ManifestError("campaign JSON requires workload.matrices")
    return spec


def _assets(spec: dict[str, Any], base: Path) -> dict[str, Any]:
    client = spec.get("client")
    if not isinstance(client, dict):
        raise ManifestError("client must be an object")
    paths = {}
    for field in ("binary", "tokenizer"):
        value = client.get(field)
        if not isinstance(value, str) or not value:
            raise ManifestError(f"client.{field} must name a readable local path")
        path = Path(value).expanduser()
        paths[field] = (base / path).resolve() if not path.is_absolute() else path.resolve()
        client[field] = str(paths[field])
    tokenizer = paths["tokenizer"]
    if not tokenizer.is_dir() or not (tokenizer / "tokenizer_config.json").is_file():
        raise ManifestError("client.tokenizer must contain tokenizer_config.json")
    if not any((tokenizer / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt")):
        raise ManifestError("client.tokenizer is missing vocabulary/tokenizer data")
    files = {name: _file_digest(tokenizer / name) for name in TOKENIZER_FILES if (tokenizer / name).is_file()}
    binary = inspect_binary(paths["binary"])
    launcher = inspect_binary(Path("/usr/bin/setpriv"))
    if client.get('version') != binary.version:
        raise ManifestError(f"client.version must equal actual --version: {binary.version}")
    from evidence_proxy import observer_identity
    return {
        "binary_sha256": binary.sha256,
        "binary_help_sha256": binary.help_sha256,
        "process_launcher": {"path": launcher.path, "sha256": launcher.sha256, "version": launcher.version, "help_sha256": launcher.help_sha256},
        "observer": observer_identity(),
        "tokenizer_sha256": sha256_json(files),
        "tokenizer_config_sha256": files["tokenizer_config.json"],
        "tokenizer_files": files,
    }


def _profile_destination(root: Path, configured: str | None, default: str) -> str:
    path = Path(configured or default).expanduser()
    destination = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not destination.is_relative_to(root) or destination.suffix != ".json":
        raise ManifestError("profile_path must be a JSON file within output.artifact_root")
    return destination.relative_to(root).as_posix()


def compile_campaign(spec_path: Path, *, output: Path | None = None, argv: list[str] | None = None, write: bool = True) -> dict[str, Any]:
    spec_path = spec_path.expanduser().resolve()
    spec = load_json_campaign(spec_path)
    assets = _assets(spec, spec_path.parent)
    compiled = compile_manifest(spec, assets)
    root = output or Path(spec["output"]["artifact_root"]).expanduser()
    if not root.is_absolute():
        root = spec_path.parent / root
    root = root.resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ArtifactError("compiled artifact root must be outside the repository")
    documents = {}
    for cell in compiled["workload"]["cells"]:
        plan = build_plan(cell, compiled["client"])
        name = f"plans/{cell['cell_id']}/request_plan.json"
        documents[name] = plan
        cell["plan"] = {"path": name, "plan_sha256": plan["plan_sha256"]}
    module_dir = Path(__file__).resolve().parent
    compiler_files = {name: assets['observer']['sources'][name] for name in (
        "campaign.py", "manifest.py", "request_plan.py", "measurement_schema.py", "profile.py", "artifacts.py", "steady_state.py",
    )}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=module_dir, capture_output=True, text=True, timeout=5, check=False)
    compiled["provenance"] = {
        "compiler_revision": revision.stdout.strip() if revision.returncode == 0 else "unversioned",
        "compiler_sha256": sha256_json(compiler_files), "compiler_files": compiler_files,
        "source_spec_sha256": _file_digest(spec_path),
        "compile_argv": argv or ["campaign.py", "compile", "--spec", str(spec_path)],
        "artifact_uri": root.as_uri(),
    }
    variants = compiled["variants"]
    variants["zero_cost"]["reference_preparation"] = "references/preparation.json"
    relative = _profile_destination(root, variants["sim"].get("profile_path"), "profiles/sim/profile.json")
    if relative in documents or relative in ("compiled_campaign.json", "artifact_index.json"):
        raise ManifestError("sim profile output collides with a compiled artifact")
    variants["sim"]["profile_path"] = relative
    variants["sim"]["profile_binding"] = {"source": "pending_fit"}
    documents["compiled_campaign.json"] = compiled
    if write:
        write_compiled_bundle(root, documents, compiled["campaign_id"])
    return compiled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    compile_parser = subcommands.add_parser("compile", help="compile configuration without starting servers")
    compile_parser.add_argument("--spec", required=True, type=Path)
    compile_parser.add_argument("--output", type=Path, help="override output.artifact_root")
    compile_parser.add_argument("--dry-run", action="store_true", help="validate and print without writing artifacts")
    prepare_parser = subcommands.add_parser("prepare-reference", help="freeze references from complete train-target evidence")
    prepare_parser.add_argument("--artifact-root", type=Path, required=True)
    run_parser = subcommands.add_parser("run", help="execute frozen cold/decode plans through the evidence proxy")
    run_parser.add_argument("--artifact-root", type=Path, required=True)
    run_parser.add_argument("--variant", choices=("target", "zero_cost", "sim"), action="append", required=True)
    run_parser.add_argument("--cell", action="append", help="exact compiled cell name or ID; repeatable")
    run_parser.add_argument("--timeout", type=float, default=120.0, help="whole bench batch deadline in seconds")
    run_parser.add_argument("--server-pid", action="append", default=[], metavar="VARIANT=PID", help="bind an existing local profile server; repeat for zero_cost and sim")
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare-reference':
            from baseline import prepare_references
            prepared = prepare_references(args.artifact_root)
            print(json.dumps({'preparation_id': prepared['preparation_id'], 'cells': len(prepared['cells'])}, sort_keys=True))
            return 0
        if args.command == 'run':
            import asyncio
            from plan_driver import run_campaign
            pids = {}
            for binding in args.server_pid:
                name, value = binding.split('=', 1)
                if name not in ('zero_cost', 'sim') or name in pids:
                    raise ValueError('server-pid must bind each profile variant once')
                pids[name] = int(value)
            result = asyncio.run(run_campaign(args.artifact_root, variants=args.variant,
                                              selectors=args.cell, timeout=args.timeout, server_pids=pids))
            print(json.dumps(result, sort_keys=True))
            return 0 if all(entry['status'] == 'sealed' for entry in result) else 1
        compiled = compile_campaign(args.spec, output=args.output, write=not args.dry_run, argv=[str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])
        print(json.dumps(compiled if args.dry_run else {"campaign_id": compiled["campaign_id"], "cells": compiled["workload"]["cell_count"]}, sort_keys=True))
        return 0
    except (ManifestError, ArtifactError, PlanError, ProfileError, VllmBenchError, OSError, ValueError, RuntimeError) as exc:
        print(f"campaign {args.command} error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
