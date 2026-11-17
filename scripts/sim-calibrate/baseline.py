#!/usr/bin/env python3
"""Collect target statistics, prepare references, and derive paired training values."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from artifacts import (ArtifactError, _read_json, checked_path, execution_snapshot,
                       load_compiled_bundle, sealed_reference, write_derived_document)
from manifest import canonical_scheduler_identity
from measurement_schema import DECODE_STATISTICS_POLICY, REFERENCE_POLICIES, reference_policy, sha256_json
from profile import reference_bundle
from steady_state import aggregate_request_samples, summary

KIND = 'prepared_references'
SCHEMA_VERSION = 3
PREPARATION_PATH = 'references/preparation.json'
TARGET_STATISTICS_PATH = 'derived/train_target_statistics.json'
TARGET_STATISTICS_SCHEMA = 3


def training_cells(campaign: dict) -> list[dict]:
    from pairing import campaign_scope
    return campaign_scope(campaign, 'train')[0]


def collect_target_statistics(root: Path) -> dict:
    """Read train-target decode attempts once, before preparing any reference."""
    from pairing import load_role
    root = root.expanduser().resolve()
    campaign = load_compiled_bundle(root)
    cells = [cell for cell in training_cells(campaign) if cell['kind'] == 'steady_decode']
    results, selected = [], []
    with execution_snapshot(root, campaign['campaign_id']) as execution:
        for cell in cells:
            try:
                reference = sealed_reference(execution['cells'].get(f"{cell['cell_id']}:target", {}))
                selected.append({'cell_id': cell['cell_id'], 'variant': 'target', **reference})
                plan = _read_json(checked_path(root, cell['plan']['path']))
                attempt = load_role(root, campaign, plan, 'target', reference)
                results.append({'cell_id': cell['cell_id'], 'name': cell['name'], 'status': 'qualified',
                                'statistics': attempt['statistics']})
            except (ArtifactError, ValueError, OSError, KeyError, TypeError) as exc:
                results.append({'cell_id': cell['cell_id'], 'name': cell['name'], 'status': 'invalid',
                                'reason': f'{type(exc).__name__}: {exc}'})
    failures = [{'cell_id': cell['cell_id'], 'reason': cell['reason']}
                for cell in results if cell['status'] != 'qualified']
    if not cells:
        failures.append({'cell_id': None, 'reason': 'train steady_decode scope is empty'})
    return {'kind': 'train_target_statistics', 'schema_version': TARGET_STATISTICS_SCHEMA,
            'campaign_id': campaign['campaign_id'], 'status': 'invalid' if failures else 'succeeded',
            'scope': 'train', 'required_roles': ['target'],
            'required_cell_ids': [cell['cell_id'] for cell in cells],
            'policy': dict(DECODE_STATISTICS_POLICY), 'fit_ready': False,
            'cell_count': len(cells), 'qualified_cell_count': sum(c['status'] == 'qualified' for c in results),
            'failure_count': len(failures), 'failures': failures, 'cells': results,
            'source': {'artifact_root': str(root), 'selected_attempts': selected}}


def verify_target_statistics(document: dict) -> None:
    """Read the direct target-statistics contract without replaying its attempts."""
    if (document.get('kind') != 'train_target_statistics'
            or document.get('schema_version') != TARGET_STATISTICS_SCHEMA
            or document.get('status') != 'succeeded'):
        raise ArtifactError('requires a succeeded current train-target statistics artifact')


def read_target_statistics(path: Path) -> dict:
    document = _read_json(path)
    verify_target_statistics(document)
    return document


def write_document(path: Path, document: dict) -> None:
    write_derived_document(path, document, Path(document['source']['artifact_root']))


def _construct(campaign: dict, selected: list[dict], stats: dict) -> tuple[dict, dict]:
    intervals = {cell['cell_id']: round(stats[cell['cell_id']]['summary']['mean'] * 1000)
                 if cell['kind'] == 'steady_decode' and cell['split'] == 'train' else 0
                 for cell in campaign['workload']['cells'] if cell['kind'] in REFERENCE_POLICIES}
    preparation_id = sha256_json(intervals)
    cells, documents = {}, {}
    for cell in campaign['workload']['cells']:
        if cell['kind'] not in REFERENCE_POLICIES:
            continue
        interval_us = intervals[cell['cell_id']]
        if interval_us < 0 or interval_us > 2**64 - 1 or (cell['kind'] == 'steady_decode' and cell['split'] == 'train' and interval_us == 0):
            raise ArtifactError('reference interval cannot be encoded as a valid u64 microsecond value')
        policy = reference_policy(cell['kind'], cell['split'])
        folder = f"references/{preparation_id}/{cell['cell_id']}"
        manifest, profile = reference_bundle(
            campaign, cell, interval_us=interval_us,
            source_digest=sha256_json({'cell_id': cell['cell_id'], 'interval_us': interval_us}),
            source_artifacts=[TARGET_STATISTICS_PATH] if cell['cell_id'] in stats else [])
        paths = {'manifest_path': folder + '/manifest.json', 'profile_path': folder + '/profile.json'}
        documents[paths['manifest_path']] = manifest
        documents[paths['profile_path']] = profile
        cells[cell['cell_id']] = {**paths, 'cell_id': cell['cell_id'], 'policy': policy,
                                 'interval_us': interval_us, 'encoded_interval_ms': interval_us / 1000,
                                 'requested_interval_ms': stats[cell['cell_id']]['summary']['mean']
                                 if cell['cell_id'] in stats else 0.0}
    return {'kind': KIND, 'schema_version': SCHEMA_VERSION, 'campaign_id': campaign['campaign_id'],
            'preparation_id': preparation_id,
            'inputs': {'selected_targets': selected, 'target_statistics': TARGET_STATISTICS_PATH if stats else None},
            'cells': cells}, documents


def _read_reference(root: Path, campaign: dict, cell_id: str, reference: dict) -> dict:
    """Read the actual known-reference curve used for subtraction."""
    profile = _read_json(checked_path(root, reference['profile_path']))
    manifest = _read_json(checked_path(root, reference['manifest_path']))
    serving = campaign['variants']['zero_cost']
    if (manifest['model_id'] != serving['identity']['model']
            or profile['scheduler'] != canonical_scheduler_identity(serving['runtime']['scheduler'])):
        raise ArtifactError('reference model or scheduler does not match the campaign')
    model = profile['predictor']['model']
    durations = model['decode']['duration_us']
    interval_us = durations[0]
    cell = next(cell for cell in campaign['workload']['cells'] if cell['cell_id'] == cell_id)
    if (any(value != interval_us for value in durations)
            or any(model['prefill']['duration_us']) or model['mixed']['intercept_us'] != 0
            or (cell['kind'] != 'steady_decode' or cell['split'] != 'train') and interval_us != 0):
        raise ArtifactError('reference must have zero prefill and a constant policy interval')
    return {**reference, 'interval_us': interval_us, 'encoded_interval_ms': interval_us / 1000}


def prepare_references(root: Path) -> dict:
    from pairing import load_role
    root = root.expanduser().resolve()
    campaign = load_compiled_bundle(root)
    output = root / PREPARATION_PATH
    if output.exists():
        return _read_json(output)
    cells = training_cells(campaign)
    target = collect_target_statistics(root) if any(c['kind'] == 'steady_decode' for c in cells) else None
    stats = {}
    if target is not None:
        verify_target_statistics(target)
        write_document(root / TARGET_STATISTICS_PATH, target)
        stats = {cell['cell_id']: cell['statistics'] for cell in target['cells']}
    selected = []
    with execution_snapshot(root, campaign['campaign_id']) as execution:
        for cell in cells:
            ref = sealed_reference(execution['cells'].get(f"{cell['cell_id']}:target", {}))
            if cell['kind'] != 'steady_decode':
                plan = _read_json(checked_path(root, cell['plan']['path']))
                load_role(root, campaign, plan, 'target', ref)
            selected.append({'cell_id': cell['cell_id'], 'variant': 'target', **ref})
    document, files = _construct(campaign, selected, stats)
    for path, value in files.items():
        write_derived_document(root / path, value, root)
    write_derived_document(output, document, root)
    return document


def select_reference(cell_id: str) -> dict:
    return {'path': PREPARATION_PATH, 'cell_id': cell_id}


def resolve_reference(root: Path, campaign: dict, binding: dict, cell_id: str) -> dict:
    if binding['cell_id'] != cell_id:
        raise ArtifactError('reference selection belongs to another cell')
    document = _read_json(checked_path(root, binding['path']))
    if document.get('schema_version') != SCHEMA_VERSION:
        raise ArtifactError('unsupported reference preparation')
    return _read_reference(root, campaign, cell_id, document['cells'][cell_id])


def derive_cell(cell: dict, observations: list[dict], reference: dict) -> dict:
    """Keep signed request values; qualify only the complete cell representative."""
    decode = cell['kind'] == 'steady_decode'
    metric = 'tpot_ms' if decode else 'ttft_ms'
    interval_ms = reference['interval_us'] / 1000
    if cell['kind'] == 'cached_prefill' and interval_ms != 0:
        raise ArtifactError('cached-prefill requires a zero-prefill, zero-decode reference')
    values, residuals, errors = [], [], []
    for row in observations:
        if decode:
            target = row['decode_window_tpot_ms']['target']
            control = row['decode_window_tpot_ms']['zero_cost']
        else:
            target = row['measurements']['target']['ttft_ms']
            control = row['measurements']['zero_cost']['ttft_ms']
        residual = control - interval_ms if decode else control
        effective = target - residual
        finite = math.isfinite(residual) and math.isfinite(effective)
        row.update(reference_interval_ms=interval_ms, baseline_residual={metric: residual if math.isfinite(residual) else None},
                   effective={metric: effective if math.isfinite(effective) else None},
                   metric_source='fixed_batch_decode_window' if decode else 'paired_request_ttft')
        if not finite:
            errors.append(f"{row['key']}: non-finite paired value")
        values.append({'cell_id': cell['cell_id'], 'repeat': row['repeat'], 'index': row['index'],
                       'requests': cell['concurrency'], 'value': effective})
        residuals.append({**values[-1], 'value': residual})
    if not values:
        errors.append('empty measured training cell')
    representative = None
    dispersion, residual_dispersion, batches = None, None, []
    if not errors:
        if decode:
            aggregate = aggregate_request_samples(values)
            representative, dispersion = aggregate['value'], aggregate['dispersion']
            residual_aggregate = aggregate_request_samples(residuals)
            residual_dispersion = residual_aggregate['dispersion']
            batches = [{'repeat': effective['repeat'], 'effective_mean': effective['mean'],
                        'residual_mean': residual['mean'], 'request_count': effective['request_count']}
                       for effective, residual in zip(aggregate['batches'], residual_aggregate['batches'])]
        else:
            representative = statistics.median(row['value'] for row in values)
            dispersion = summary([row['value'] for row in values])
            residual_dispersion = summary([row['value'] for row in residuals])
            batches = [{'repeat': row['repeat'], 'effective_mean': row['value'],
                        'residual_mean': residual['value'], 'request_count': 1}
                       for row, residual in zip(values, residuals)]
        if not math.isfinite(representative) or representative < 0 or (decode and representative == 0):
            errors.append('non-finite or invalid cell representative for nonnegative cost model')
    return {'qualified': not errors, 'representative': representative if representative is not None and math.isfinite(representative) else None,
            'metric': metric, 'center': 'repeat_mean' if decode else 'paired_repeat_median',
            'repeat_count': cell['repeats'], 'request_count': len(values), 'dispersion': dispersion,
            'residual_dispersion': residual_dispersion, 'batches': batches,
            'errors': errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path)
    mode.add_argument('--verify', type=Path)
    args = parser.parse_args()
    try:
        if args.verify:
            document = read_target_statistics(args.verify)
        else:
            document = collect_target_statistics(args.artifact_root)
            write_document(args.output, document)
    except (ArtifactError, ValueError, OSError, KeyError, TypeError) as exc:
        print(f'target statistics error: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({key: document[key] for key in ('campaign_id', 'status', 'cell_count', 'failure_count')}, sort_keys=True))
    return 0 if document['status'] == 'succeeded' else 1


if __name__ == '__main__':
    raise SystemExit(main())
