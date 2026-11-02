#!/usr/bin/env python3
"""Compile immutable campaigns and execute the train-to-held-out lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
import platform
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from artifacts import (ArtifactError, _read_json, checked_path, execution_snapshot,
                       load_compiled_bundle, read_sealed_attempt, sealed_reference, write_compiled_bundle)
from manifest import ManifestError, compile_manifest
from profile import ProfileError
from request_plan import PlanError, build_plan
from vllm_bench import VllmBenchError, inspect_assets

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


def _profile_destination(root: Path, configured: str | None, default: str) -> str:
    path = Path(configured or default).expanduser()
    destination = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not destination.is_relative_to(root) or destination.suffix != ".json":
        raise ManifestError("profile_path must be a JSON file within output.artifact_root")
    return destination.relative_to(root).as_posix()


def compile_campaign(spec_path: Path, *, output: Path | None = None, argv: list[str] | None = None, write: bool = True) -> dict[str, Any]:
    spec_path = spec_path.expanduser().resolve()
    spec = load_json_campaign(spec_path)
    launch = spec.get('variants', {}).get('target', {}).get('launch')
    if launch is not None:
        if (not isinstance(launch, dict) or not isinstance(launch.get('argv'), list) or not launch['argv']
                or any(not isinstance(arg, str) or not arg for arg in launch['argv'])):
            raise ManifestError('target launch requires a nonempty argv')
        executable = Path(launch['argv'][0]).expanduser()
        if not executable.is_absolute():
            executable = spec_path.parent / executable
        launch['argv'][0] = str(executable.absolute())
        from probe import target_runtime_environment
        environment = launch.get('environment', {})
        if not isinstance(environment, dict):
            raise ManifestError('target launch environment must be an object')
        launch['environment'] = {**target_runtime_environment(dict(os.environ)), **environment}
    assets = inspect_assets(spec["client"], spec_path.parent)
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
        if cell['kind'] == 'cached_prefill':
            from cached_prefill import workload_documents
            documents.update(workload_documents(plan))
        cell["plan"] = {"path": name}
    from cached_prefill import validate_workload_collection
    validate_workload_collection([documents[c['plan']['path']] for c in compiled['workload']['cells']])
    compiled["provenance"] = {
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
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                  capture_output=True, text=True).stdout.strip()
        packages = {d.metadata['Name']: d.version for d in distributions()}
        environment = {"python": platform.python_version(), "git_revision": revision, "packages": packages}
        diagnostics = root / 'diagnostics'
        diagnostics.mkdir(exist_ok=True)
        (diagnostics / 'environment.json').write_text(json.dumps(environment, indent=2, sort_keys=True) + '\n')
    return compiled


async def execute_campaign(root: Path, *, sim_binary: Path, model_path: Path,
                           timeout: float = 120, startup_timeout: float = 60,
                           through_fit: bool = False) -> dict:
    """Run a complete cold/decode scope against an already-running target.

    Profile servers are owned here because references differ by cell. Every
    acquisition still uses the existing driver. Recovery reuses sealed attempts
    without needing their historical processes alive.
    """
    import math
    from baseline import prepare_references, resolve_reference, select_reference
    from fit import fit_normalized, write_document as write_fit
    from normalize import normalize_campaign, verify_normalized, write_document as write_normalized
    from pairing import campaign_scope
    from plan_driver import profile_server, run_campaign
    from profile import materialize_calibration, materialize_profile, verify_fitted_bundle
    from artifacts import fitted_bundle_paths
    from report import report_campaign
    from report import write_document as write_report

    if any(not math.isfinite(value) or value <= 0 for value in (timeout, startup_timeout)):
        raise ArtifactError('campaign timeouts must be finite and positive')
    root = root.expanduser().resolve()
    campaign = load_compiled_bundle(root)
    if not sim_binary.expanduser().resolve().is_file():
        raise ArtifactError('simulator binary is missing')
    train, train_deferred, _, _ = campaign_scope(campaign, 'train')
    heldout, heldout_deferred, _, _ = campaign_scope(campaign, 'validation')
    cached_only = bool(train) and all(c['kind'] == 'cached_prefill' for c in train)
    through_fit = through_fit or (cached_only and not heldout and not heldout_deferred)
    if train_deferred or (heldout_deferred and not through_fit):
        raise ArtifactError('required cached online simulator validation is unsupported')
    if not train or (not through_fit and (not heldout or not any(c['concurrency'] > 1 for c in heldout))):
        raise ArtifactError('full-stage execution requires train and required C>1 held-out cells')

    def pending(cells: list[dict], role: str) -> list[dict]:
        remaining = []
        if not (root / 'execution_index.json').exists():
            return cells
        with execution_snapshot(root, campaign['campaign_id']) as execution:
            for cell in cells:
                entry = execution['cells'].get(f"{cell['cell_id']}:{role}", {})
                if entry.get('status') != 'sealed':
                    remaining.append(cell)
                    continue
                plan = _read_json(checked_path(root, cell['plan']['path']))
                read_sealed_attempt(root, plan, role, sealed_reference(entry))
        return remaining

    async def collect(cells: list[dict], role: str, pid: int | None = None) -> None:
        if not cells:
            return
        results = await run_campaign(root, variants=[role], selectors=[c['cell_id'] for c in cells],
                                     timeout=timeout, server_pids={role: pid} if pid else None)
        if len(results) != len(cells) or any(result['status'] != 'sealed' for result in results):
            raise ArtifactError(f'{role} acquisition failed; inspect execution_index.json before an explicit resume')

    async def references(cells: list[dict]) -> None:
        for cell in pending(cells, 'zero_cost'):
            binding = select_reference(root, campaign, cell['cell_id'])
            variant = {**campaign['variants']['zero_cost'], **resolve_reference(root, campaign, binding, cell['cell_id'])}
            async with profile_server(root, variant, binary=sim_binary, model_path=model_path,
                                      timeout=startup_timeout) as pid:
                await collect([cell], 'zero_cost', pid)

    await collect(pending(train, 'target'), 'target')
    prepare_references(root)
    await references(train)
    if through_fit:
        normalized_path, fit_path = root / 'derived/train.normalized.json', root / 'derived/fit.json'
        normalized = normalize_campaign(root, scope='train')
        if normalized['status'] != 'succeeded':
            raise ArtifactError('complete qualified training is required before fit')
        write_normalized(normalized_path, normalized)
        fitted = fit_normalized(normalized)
        if fitted['status'] != 'succeeded':
            raise ArtifactError('required training candidates could not be fitted')
        write_fit(fit_path, fitted)
        result = {'campaign_id': campaign['campaign_id'], 'status': 'succeeded',
                  'stage': 'offline_calibration', 'runtime_enabled': False,
                  'normalized': {'path': str(normalized_path)},
                  'fit': {'path': str(fit_path)}}
        if any(c['kind'] == 'cached_prefill' for c in train):
            result['calibration'] = materialize_calibration(root, normalized_artifact=normalized_path, fit_artifact=fit_path)
        return result
    if fitted_bundle_paths(root, campaign)['selection'].exists():
        verify_fitted_bundle(root, campaign)
    else:
        normalized_path, fit_path = root / 'derived/train.normalized.json', root / 'derived/fit.json'
        if normalized_path.exists():
            normalized = _read_json(normalized_path)
            verify_normalized(normalized, root=root)
        else:
            normalized = normalize_campaign(root, scope='train')
        if normalized['status'] != 'succeeded':
            raise ArtifactError('complete qualified training is required before fit')
        write_normalized(normalized_path, normalized)
        fitted = fit_normalized(normalized)
        if fitted['status'] != 'succeeded':
            raise ArtifactError('required training candidates could not be fitted')
        write_fit(fit_path, fitted)
        materialize_profile(root, normalized_artifact=normalized_path, fit_artifact=fit_path)
    await collect(pending(heldout, 'target'), 'target')
    await references(heldout)
    remaining = pending(heldout, 'sim')
    if remaining:
        async with profile_server(root, campaign['variants']['sim'], binary=sim_binary,
                                  model_path=model_path, timeout=startup_timeout) as pid:
            await collect(remaining, 'sim', pid)
    report = report_campaign(root)
    report_path = root / 'reports' / ('validation.json' if report['status'] == 'succeeded'
                                    else f"{report['status']}-{uuid.uuid4().hex}.json")
    write_report(report_path, report)
    return {'campaign_id': campaign['campaign_id'], 'status': report['status'],
            'report': {'path': str(report_path)},
            'expected_count': report['expected_count'], 'matched_count': report['matched_count']}


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
    execute_parser = subcommands.add_parser('execute', help='run all cold/decode stages against an already-running target')
    execute_parser.add_argument('--artifact-root', type=Path, required=True)
    execute_parser.add_argument('--sim-binary', type=Path, required=True)
    execute_parser.add_argument('--model-path', type=Path, required=True)
    execute_parser.add_argument('--timeout', type=float, default=120)
    execute_parser.add_argument('--startup-timeout', type=float, default=60)
    execute_parser.add_argument('--through-fit', action='store_true', help='stop after verified offline training and calibration publication')
    args = parser.parse_args(argv)
    try:
        if args.command == 'execute':
            import asyncio
            result = asyncio.run(execute_campaign(args.artifact_root, sim_binary=args.sim_binary,
                                                 model_path=args.model_path, timeout=args.timeout,
                                                 startup_timeout=args.startup_timeout, through_fit=args.through_fit))
            print(json.dumps(result, sort_keys=True))
            return 0 if result['status'] == 'succeeded' else 1
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
