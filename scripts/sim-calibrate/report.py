#!/usr/bin/env python3
"""Report paired held-out raw TTFT; explicitly replay selected evidence on verify."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from artifacts import (ArtifactError, _read_json, checked_path, execution_snapshot, file_reference,
                       fitted_bundle_paths, load_compiled_bundle, sealed_reference, write_derived_document)
from manifest import execution_identity
from pairing import campaign_scope, load_role, pair_plan_measurements

KIND = 'held_out_replay_report'
SCHEMA_VERSION = 4


def _percentiles(values: list[float]) -> dict:
    values = sorted(values)
    result = {}
    for name, quantile in (('p50', .50), ('p95', .95), ('p99', .99)):
        if not values:
            result[name] = None
            continue
        position = (len(values)-1) * quantile
        lower = int(position)
        upper = min(lower+1, len(values)-1)
        result[name] = values[lower] + (values[upper]-values[lower]) * (position-lower)
    return result


def summarize_samples(samples: list[dict]) -> dict:
    """The sole percentile owner; errors are computed before aggregation."""
    return {'count': len(samples),
            'actual': _percentiles([row['actual_ms'] for row in samples]),
            'predicted': _percentiles([row['predicted_ms'] for row in samples]),
            'errors': {'count': len(samples),
                       'absolute_ms': _percentiles([row['absolute_error_ms'] for row in samples]),
                       'relative': _percentiles([row['relative_error'] for row in samples])}}


def _sample(observation: dict) -> dict:
    measurements = observation['measurements']
    actual, predicted = (measurements[role]['ttft_ms'] for role in ('target', 'sim'))
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
           for value in (actual, predicted)) or actual <= 0 or predicted < 0:
        raise ValueError('raw TTFT requires positive target and finite nonnegative simulator values')
    absolute = abs(actual-predicted)
    relative = absolute/actual
    if not math.isfinite(relative):
        raise ValueError('raw TTFT relative error is non-finite')
    return {'key': observation['key'], 'repeat': observation['repeat'], 'index': observation['index'],
            'actual_ms': actual, 'predicted_ms': predicted,
            'absolute_error_ms': absolute, 'relative_error': relative}


def _percentiles_complete(values) -> bool:
    return isinstance(values, dict) and all(
        type(values.get(key)) in (int, float) and math.isfinite(values[key]) and values[key] >= 0
        for key in ('p50', 'p95', 'p99'))


def _summary_complete(summary: dict, count: int) -> bool:
    return (summary.get('count') == count and summary.get('errors', {}).get('count') == count
            and all(_percentiles_complete(values) for values in (
                summary.get('actual'), summary.get('predicted'),
                summary.get('errors', {}).get('absolute_ms'), summary.get('errors', {}).get('relative'))))


def assess_report(document: dict) -> dict:
    """Never infer campaign success from a subset or from aggregate values alone."""
    failures = []
    required = document['required_cell_ids']
    cells = document['cells']
    identities = [cell['cell_id'] for cell in cells]
    if not required or len(set(required)) != len(required) or identities != required:
        failures.append('required cell set is empty, duplicated or incomplete')
    if document['required_roles'] != ['target', 'zero_cost', 'sim']:
        failures.append('validation requires all three roles')
    if not document['required_queueing_cell_ids']:
        failures.append('required concurrency > 1 held-out scope is empty')
    if document['required_queueing_cell_ids'] != [cell['cell_id'] for cell in cells if cell['concurrency'] > 1]:
        failures.append('required queueing scope differs from report cells')
    if document['relative_denominator'] != 'positive_target_raw_ttft_ms':
        failures.append('undefined relative-error denominator')
    if document['deferred_cells']:
        failures.append('required cached-prefill runtime validation is unsupported')
    for cell in cells:
        cell_id = cell['cell_id']
        if cell['status'] != 'measured' or cell['coverage']['status'] != 'in_domain':
            failures.append(f'{cell_id}: missing, unsupported or out-of-domain execution')
        if cell['pairing'].get('complete') is not True or any(cell[role]['status'] != 'succeeded'
                                                            for role in ('target', 'zero_cost', 'sim')):
            failures.append(f'{cell_id}: incomplete role pairing')
        count = cell['expected_count']
        if (type(count) is not int or count <= 0 or cell['matched_count'] != count
                or cell['count'] != count or len(cell['samples']) != count
                or cell['errors']['count'] != count or cell['pairing']['matched_count'] != count):
            failures.append(f'{cell_id}: expected/matched/statistic count mismatch')
        for values in (cell['actual'], cell['predicted'], cell['errors']['absolute_ms'], cell['errors']['relative']):
            if not _percentiles_complete(values):
                failures.append(f'{cell_id}: missing actual/predicted/error percentiles')
    if (document['cell_count'] != len(cells)
            or document['measured_cell_count'] != sum(cell['status'] == 'measured' for cell in cells)
            or document['expected_count'] != sum(cell['expected_count'] for cell in cells)
            or document['matched_count'] != sum(cell['matched_count'] for cell in cells)
            or document['matched_count'] != document['expected_count']
            or document['aggregates']['all']['count'] != document['matched_count']):
        failures.append('aggregate and cell counts disagree')
    if cells and not _summary_complete(document['aggregates']['all'], document['matched_count']):
        failures.append('overall actual/predicted/error statistics are incomplete')
    if set(document['aggregates']['by_split']) != {cell['split'] for cell in cells}:
        failures.append('aggregate split scope differs from cells')
    for split, aggregate in document['aggregates']['by_split'].items():
        count = sum(cell['matched_count'] for cell in cells if cell['split'] == split)
        if not _summary_complete(aggregate, count):
            failures.append(f'{split}: aggregate and cell counts disagree')
    queueing = document['queueing_ttft']
    expected_queueing = [cell for cell in cells if cell['concurrency'] > 1]
    if queueing['cells'] != expected_queueing or set(queueing['aggregates']) != {cell['split'] for cell in cells}:
        failures.append('queueing cells/aggregate scope differs from required cells')
    for split, aggregate in queueing['aggregates'].items():
        count = sum(cell['matched_count'] for cell in expected_queueing if cell['split'] == split)
        if count and not _summary_complete(aggregate, count):
            failures.append(f'{split}: queueing aggregate statistics are incomplete')
        elif aggregate.get('count') != count:
            failures.append(f'{split}: queueing aggregate count disagrees')
    if not document['provenance']['profile']:
        failures.append('verified fitted profile is missing')
    return {'status': 'failed' if failures else 'passed', 'failures': failures,
            'basis': 'evidence_scope_pairing_statistics', 'accuracy_assessed': False}


def _heldout(root: Path, campaign: dict, cells: list[dict], roles: tuple,
             selected: list[dict] | None, replay_raw: bool = False) -> tuple[list[dict], list[dict]]:
    if selected is None:
        execution = {}
        if (root/'execution_index.json').exists():
            with execution_snapshot(root, campaign['campaign_id']) as index:
                execution = index['cells']
        selected = []
        for cell in cells:
            for role in roles:
                entry = execution.get(f"{cell['cell_id']}:{role}", {})
                if entry.get('status') == 'sealed':
                    selected.append({'cell_id': cell['cell_id'], 'variant': role, **sealed_reference(entry)})
    indexed = {(ref['cell_id'], ref['variant']): ref for ref in selected}
    results = []
    for cell in cells:
        plan = _read_json(checked_path(root, cell['plan']['path']))
        attempts, sources, observations = {}, {}, []
        reason = None
        try:
            for role in roles:
                ref = indexed[cell['cell_id'], role]
                sources[role] = {key: value for key, value in ref.items() if key not in ('cell_id', 'variant')}
                attempts[role] = load_role(root, campaign, plan, role, sources[role], replay_raw=replay_raw)
            paired = pair_plan_measurements(plan, attempts, roles)
            observations = [{'key': logical['pairing_key'], 'repeat': logical['repeat'],
                             'index': logical['index'], 'measurements': rows}
                            for logical, rows in paired['pairs']]
        except (ArtifactError, ValueError, OSError, KeyError, TypeError) as exc:
            reason = f'{type(exc).__name__}: {exc}'
        results.append({'cell_id': cell['cell_id'], 'expected_count': sum(
            row['phase'] == 'measured' for row in plan['logical_requests']),
                        'observations': observations, 'sources': sources, 'reason': reason})
    return results, selected


def _build_report(root: Path, campaign: dict, selected: list[dict] | None = None, *, replay_raw: bool = False) -> dict:
    cells, deferred, roles, _ = campaign_scope(campaign, 'validation')
    profile_sources, profile_failure = {}, None
    try:
        for name, path in fitted_bundle_paths(root, campaign).items():
            profile_sources[name] = {'path': str(path)}
            if name in ('profile', 'manifest'):
                profile_sources[name].update(file_reference(path))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        profile_failure = f'{type(exc).__name__}: {exc}'
    paired_cells, selected = _heldout(root, campaign, cells, roles, selected, replay_raw)
    results = []
    for cell, paired in zip(cells, paired_cells):
        samples, failures = [], []
        if paired['reason']:
            failures.append(paired['reason'])
        else:
            try:
                samples = [_sample(row) for row in paired['observations']]
            except (ValueError, KeyError, TypeError) as exc:
                failures.append(str(exc))
        if profile_failure:
            failures.append('fitted profile: ' + profile_failure)
        sources = paired['sources']
        complete = paired['reason'] is None
        result = {'cell_id': cell['cell_id'], 'name': cell['name'], 'kind': cell['kind'], 'split': cell['split'],
                  'concurrency': cell['concurrency'], 'plan': cell['plan'],
                  'status': 'measured' if not failures else 'not_measured',
                  'expected_count': paired['expected_count'], 'matched_count': len(samples),
                  'pairing': {'complete': complete, 'target_zero_cost': complete, 'target_sim': complete,
                              'expected_count': paired['expected_count'], 'matched_count': len(samples)},
                  'coverage': {'status': 'in_domain' if not failures else 'unverified',
                               'basis': 'successful_profiled_execution',
                               'reason': '; '.join(failures) if failures else None},
                  **{role: {'status': 'succeeded' if complete else 'missing'} for role in roles},
                  'reference_diagnostic': {'policy': cell['reference_policy'], 'fit_eligible': False,
                                           'binding': sources.get('zero_cost', {}).get('binding')},
                  'samples': samples, **summarize_samples(samples), 'failures': failures}
        results.append(result)
    all_samples = [row for cell in results for row in cell['samples']]
    split_names = sorted({cell['split'] for cell in results})
    by_split = {split: summarize_samples([row for cell in results if cell['split'] == split for row in cell['samples']])
                for split in split_names}
    queueing = [cell for cell in results if cell['concurrency'] > 1]
    report = {
        'kind': KIND, 'schema_version': SCHEMA_VERSION, 'campaign_id': campaign['campaign_id'],
        'scope': 'validation',
        'required_roles': list(roles), 'required_cell_ids': [cell['cell_id'] for cell in cells],
        'required_queueing_cell_ids': [cell['cell_id'] for cell in cells if cell['concurrency'] > 1],
        'campaign_required_scope': campaign['required_scope'], 'deferred_cells': deferred,
        'coverage_required_kinds': sorted({cell['kind'] for cell in cells}),
        'metric': 'ttft_ms', 'basis': 'paired_raw_target_sim',
        'relative_denominator': 'positive_target_raw_ttft_ms',
        'accuracy_acceptance': {'status': 'not_assessed', 'thresholds': None},
        'cell_count': len(results), 'measured_cell_count': sum(cell['status'] == 'measured' for cell in results),
        'expected_count': sum(cell['expected_count'] for cell in results), 'matched_count': len(all_samples),
        'cells': results, 'aggregates': {'by_split': by_split, 'all': summarize_samples(all_samples)},
        'queueing_ttft': {'metric': 'ttft_ms', 'basis': 'paired_raw_target_sim', 'fit_excluded': True,
                          'cells': queueing,
                          'aggregates': {split: summarize_samples([row for cell in queueing if cell['split'] == split
                                                                  for row in cell['samples']]) for split in split_names}},
        'provenance': {'artifact_root': str(root), 'profile': profile_sources,
                       'target_identity': execution_identity(campaign['variants']['target']['identity'])},
        'source': {'selected_attempts': selected},
    }
    report['acceptance'] = assess_report(report)
    report['status'] = ('succeeded' if report['acceptance']['status'] == 'passed' else
                        'partial' if any(ref['variant'] == 'sim' for ref in selected) else 'replay_required')
    return report


def report_campaign(root: Path) -> dict:
    root = root.expanduser().resolve()
    return _build_report(root, load_compiled_bundle(root))


def verify_report(document: dict, *, root: Path | None = None) -> None:
    """Explicitly recompute a report from its fixed held-out attempts."""
    if document.get('kind') != KIND or document.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('current held-out replay report schema 4 is required')
    root = (root or Path(document['provenance']['artifact_root'])).expanduser().resolve()
    expected = _build_report(root, load_compiled_bundle(root), document['source']['selected_attempts'], replay_raw=True)
    if document != expected:
        raise ValueError('held-out report differs from selected evidence, statistics or completeness')


def write_document(path: Path, document: dict) -> None:
    write_derived_document(path, document, Path(document['provenance']['artifact_root']))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        document = report_campaign(args.artifact_root)
        write_document(args.output, document)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(f'report error: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({key: document[key] for key in ('campaign_id', 'status', 'cell_count', 'matched_count')}, sort_keys=True))
    return 0 if document['status'] == 'succeeded' else 1


if __name__ == '__main__':
    raise SystemExit(main())
