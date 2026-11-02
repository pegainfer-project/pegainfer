#!/usr/bin/env python3
"""Collect held-out cached requests and validate the offline calibration surface."""
from __future__ import annotations

import argparse
import asyncio
import math
import statistics
from pathlib import Path

from artifacts import (_read_json, checked_path, execution_snapshot,
                       load_compiled_bundle, read_sealed_attempt, sealed_reference, write_derived_document)
from baseline import resolve_reference, select_reference
from fit import evaluate_cached_surface
from pairing import campaign_scope, load_role, pair_plan_measurements
from profile import CALIBRATION_PATH, read_calibration
from report import _percentiles

KIND = 'cached_prefill_validation_report'
SCHEMA_VERSION = 3
ROLES = ('target', 'zero_cost')
DEFAULT_REPORT = 'reports/cached_validation.json'


class CachedValidationError(ValueError):
    """The complete declared cached held-out experiment cannot be verified."""


def _scope(campaign: dict) -> list[dict]:
    # Reuse the canonical required-scope checks; this report does not promote
    # deferred cached cells into the three-role online report.
    campaign_scope(campaign, 'train')
    campaign_scope(campaign, 'validation')
    cells = [c for c in campaign['workload']['cells']
             if c['kind'] == 'cached_prefill' and c['split'] != 'train']
    if not cells:
        raise CachedValidationError('required cached held-out scope is empty')
    return cells


def _selected(root: Path, campaign: dict, cells: list[dict], pinned: list | None = None) -> list[dict]:
    required = [(cell['cell_id'], role) for cell in cells for role in ROLES]
    if pinned is None:
        with execution_snapshot(root, campaign['campaign_id']) as execution:
            pinned = [{'cell_id': cell_id, 'variant': role,
                       **sealed_reference(execution['cells'].get(f'{cell_id}:{role}', {}))}
                      for cell_id, role in required]
    if (not isinstance(pinned, list) or any(not isinstance(ref, dict) for ref in pinned)
            or [(ref.get('cell_id'), ref.get('variant')) for ref in pinned] != required):
        raise CachedValidationError('selected held-out attempts do not cover the exact required roles/cells')
    return pinned


def _open(root: Path) -> tuple[dict, list[dict], dict]:
    campaign = load_compiled_bundle(root)
    cells = _scope(campaign)
    calibration = read_calibration(root, campaign)
    for cell in cells:
        prediction = evaluate_cached_surface(calibration['surface'], cell['prefix_tokens'], cell['new_tokens'])
        if not math.isfinite(prediction) or prediction < 0:
            raise CachedValidationError('cached prediction must be finite and nonnegative')
    return campaign, cells, calibration


def _sample(logical: dict, measurements: dict, predicted: float) -> dict:
    target, zero = (measurements[role]['ttft_ms'] for role in ROLES)
    if any(type(value) not in (float, int) or not math.isfinite(value) or value < 0
           for value in (target, zero, predicted)):
        raise CachedValidationError('cached TTFT/prediction must be finite and nonnegative')
    effective = target - zero
    signed = predicted - effective
    rebuilt = predicted + zero
    result = {'key': logical['pairing_key'], 'repeat': logical['repeat'], 'index': logical['index'],
            'target_ttft_ms': target, 'zero_cost_ttft_ms': zero, 'effective_ms': effective,
            'predicted_ms': predicted, 'signed_error_ms': signed, 'absolute_error_ms': abs(signed),
            'relative_error': abs(signed) / effective if effective > 0 else None,
            'reconstructed_target_ttft_ms': rebuilt, 'reconstructed_signed_error_ms': rebuilt-target,
            'reconstructed_absolute_error_ms': abs(rebuilt-target),
            'reconstructed_relative_error': abs(rebuilt-target) / target if target > 0 else None}
    if any(value is not None and not math.isfinite(value)
           for key, value in result.items() if key.endswith(('_ms', '_error'))):
        raise CachedValidationError('paired cached errors must be finite when defined')
    return result


def _primary_summary(rows: list[dict]) -> dict:
    relative = [row['relative_error'] for row in rows if row['relative_error'] is not None]
    return {'count': len(rows),
            **{field: _percentiles([row[field] for row in rows])
               for field in ('effective_ms', 'predicted_ms', 'signed_error_ms', 'absolute_error_ms')},
            'relative_error': _percentiles(relative), 'relative_count': len(relative),
            'relative_undefined_count': len(rows)-len(relative)}


def _summary(rows: list[dict]) -> dict:
    result = _primary_summary(rows)
    result.update({field: _percentiles([row[field] for row in rows]) for field in (
        'target_ttft_ms', 'zero_cost_ttft_ms', 'reconstructed_target_ttft_ms',
        'reconstructed_signed_error_ms', 'reconstructed_absolute_error_ms')})
    relative = [row['reconstructed_relative_error'] for row in rows
                if row['reconstructed_relative_error'] is not None]
    result.update(reconstructed_relative_error=_percentiles(relative),
                  reconstructed_relative_count=len(relative),
                  reconstructed_relative_undefined_count=len(rows)-len(relative))
    return result


def _median(samples: list[dict], prediction: float) -> dict:
    effective = statistics.median(row['effective_ms'] for row in samples)
    signed = prediction-effective
    return {'effective_ms': effective, 'predicted_ms': prediction, 'signed_error_ms': signed,
            'absolute_error_ms': abs(signed), 'relative_error': abs(signed)/effective if effective > 0 else None}


def _build(root: Path, context: tuple, pinned: list | None = None, *, replay_raw: bool = False) -> dict:
    campaign, cells, calibration = context
    selected = _selected(root, campaign, cells, pinned)
    indexed = {(ref['cell_id'], ref['variant']): ref for ref in selected}
    results = []
    for cell in cells:
        plan = _read_json(checked_path(root, cell['plan']['path']))
        references = {role: {key: value for key, value in indexed[cell['cell_id'], role].items()
                             if key not in ('cell_id', 'variant')} for role in ROLES}
        # Exactly one load of each held-out role per cell, then all repeats share it.
        attempts = {role: load_role(root, campaign, plan, role, references[role], replay_raw=replay_raw) for role in ROLES}
        paired = pair_plan_measurements(plan, attempts, ROLES)
        prediction = evaluate_cached_surface(calibration['surface'], plan['prefix_tokens'], plan['new_tokens'])
        samples = [_sample(logical, rows, prediction) for logical, rows in paired['pairs']]
        if len(samples) != cell['repeats'] or cell['concurrency'] != 1:
            raise CachedValidationError('cached measured repeats differ from the required plan')
        results.append({'cell_id': cell['cell_id'], 'name': cell['name'], 'kind': cell['kind'], 'split': cell['split'],
                        'prefix_tokens': plan['prefix_tokens'], 'new_tokens': plan['new_tokens'],
                        'plan': cell['plan'], 'expected_count': paired['expected_count'],
                        'matched_count': paired['matched_count'], 'expected_keys': [row['key'] for row in samples],
                        'predicted_ms': prediction, 'samples': samples, 'summary': _summary(samples),
                        'median': _median(samples, prediction),
                        'reference_policy': cell['reference_policy']})
    all_samples = [row for cell in results for row in cell['samples']]
    report = {'kind': KIND, 'schema_version': SCHEMA_VERSION, 'status': 'succeeded',
              'campaign_id': campaign['campaign_id'], 'scope': 'cached_heldout',
              'basis': 'offline_surface_vs_paired_effective_ttft_ms', 'runtime_enabled': False,
              'reconstruction_basis': 'offline_prediction_plus_same_key_zero_cost_ttft',
              'accuracy_acceptance': {'status': 'not_assessed', 'thresholds': None},
              'required_roles': list(ROLES), 'required_cell_ids': [cell['cell_id'] for cell in cells],
              'campaign_required_scope': campaign['required_scope'],
              'cell_count': len(results), 'expected_count': len(all_samples), 'matched_count': len(all_samples),
              'cells': results, 'aggregates': {'all': _summary(all_samples),
                                              'cell_medians': _primary_summary([c['median'] for c in results])},
              'source': {'artifact_root': str(root), 'calibration': CALIBRATION_PATH,
                         'selected_attempts': selected}}
    return report


def validate_cached_campaign(root: Path, *, output: Path | None = None) -> dict:
    root = root.expanduser().resolve()
    report = _build(root, _open(root))
    if output is not None:
        write_derived_document(output, report, root)
    return report


def verify_report(document: dict, *, root: Path | None = None) -> None:
    if document.get('kind') != KIND or document.get('schema_version') != SCHEMA_VERSION:
        raise CachedValidationError('current cached validation report schema 3 is required')
    root = (root or Path(document['source']['artifact_root'])).expanduser().resolve()
    expected = _build(root, _open(root), document['source']['selected_attempts'], replay_raw=True)
    if document != expected:
        raise CachedValidationError('cached report differs from selected evidence, scope or statistics')


def write_report(path: Path, document: dict) -> None:
    """Verify an arbitrary report before publication; generated reports can use output=."""
    verify_report(document)
    write_derived_document(path, document, Path(document['source']['artifact_root']))


async def execute_cached_validation(root: Path, *, sim_binary: Path, model_path: Path,
                                    timeout: float = 120, startup_timeout: float = 60,
                                    output: Path | None = None) -> dict:
    from plan_driver import profile_server, run_campaign
    if any(not math.isfinite(value) or value <= 0 for value in (timeout, startup_timeout)):
        raise CachedValidationError('cached collection timeouts must be finite and positive')
    root = root.expanduser().resolve()
    context = _open(root)
    campaign, cells, _ = context

    def pending(role: str) -> list[dict]:
        remaining = []
        with execution_snapshot(root, campaign['campaign_id']) as execution:
            for cell in cells:
                entry = execution['cells'].get(f"{cell['cell_id']}:{role}", {})
                if entry.get('status') != 'sealed':
                    remaining.append(cell)
                    continue
                plan = _read_json(checked_path(root, cell['plan']['path']))
                read_sealed_attempt(root, plan, role, sealed_reference(entry))
        return remaining

    async def collect(chosen: list[dict], role: str, pid: int | None = None) -> None:
        if not chosen:
            return
        result = await run_campaign(root, variants=[role], selectors=[cell['cell_id'] for cell in chosen],
                                    timeout=timeout, server_pids={role: pid} if pid else None)
        if len(result) != len(chosen) or any(entry['status'] != 'sealed' for entry in result):
            raise CachedValidationError(f'cached held-out {role} collection failed; retain raw and explicitly resume')

    await collect(pending('target'), 'target')
    for cell in pending('zero_cost'):
        binding = select_reference(root, campaign, cell['cell_id'])
        variant = {**campaign['variants']['zero_cost'], **resolve_reference(root, campaign, binding, cell['cell_id'])}
        async with profile_server(root, variant, binary=sim_binary, model_path=model_path,
                                  timeout=startup_timeout) as pid:
            await collect([cell], 'zero_cost', pid)
    report = _build(root, context)
    if output is not None:
        write_derived_document(output, report, root)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('collect', 'validate', 'verify'):
        child = commands.add_parser(command)
        child.add_argument('--artifact-root', type=Path, required=True)
        if command == 'verify':
            child.add_argument('--report', type=Path)
        else:
            child.add_argument('--output', type=Path)
        if command == 'collect':
            child.add_argument('--sim-binary', type=Path, required=True)
            child.add_argument('--model-path', type=Path, required=True)
            child.add_argument('--timeout', type=float, default=120)
            child.add_argument('--startup-timeout', type=float, default=60)
    args = parser.parse_args()
    root = args.artifact_root.expanduser().resolve()
    try:
        if args.command == 'verify':
            report = _read_json(args.report or root/DEFAULT_REPORT)
            verify_report(report, root=root)
        elif args.command == 'collect':
            report = asyncio.run(execute_cached_validation(
                root, sim_binary=args.sim_binary, model_path=args.model_path,
                timeout=args.timeout, startup_timeout=args.startup_timeout,
                output=args.output or root/DEFAULT_REPORT))
        else:
            report = validate_cached_campaign(root, output=args.output or root/DEFAULT_REPORT)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f'cached validation error: {exc}')
        return 2
    import json
    print(json.dumps({key: report[key] for key in ('campaign_id', 'status', 'cell_count', 'expected_count', 'matched_count')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
