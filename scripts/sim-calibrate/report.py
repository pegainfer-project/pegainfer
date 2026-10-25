#!/usr/bin/env python3
"""Report paired held-out target/sim raw TTFT, with pinned source verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from acceptance import assess_report
from artifacts import (_read_json, checked_path, file_reference, fitted_bundle_paths, load_compiled_bundle,
                       write_derived_document)
from manifest import sha256_json
from normalize import (NORMALIZER_SOURCES, _assemble_document, _cell_result,
                       normalize_campaign, verify_normalized)
from pairing import campaign_scope
from profile import verify_fitted_bundle

KIND = 'held_out_replay_report'
SCHEMA_VERSION = 2
REPORT_SOURCES = ('report.py', 'replay.py', 'acceptance.py')


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


def _build_report(root: Path, campaign: dict, normalized: dict) -> dict:
    frozen = campaign['client']['observer']['sources']
    reporter_sources = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                        for name in REPORT_SOURCES}
    if reporter_sources != {name: frozen.get(name) for name in REPORT_SOURCES}:
        raise ValueError('frozen report generator changed; compile and collect into a new root')
    if normalized['source']['normalizer_sources'] != {name: frozen.get(name) for name in NORMALIZER_SOURCES}:
        raise ValueError('validation normalizer differs from frozen acquisition sources')
    cells, deferred, roles, definition = campaign_scope(campaign, 'validation')
    # Final report success covers required evidence, including unsupported train
    # components. These stay out of held-out samples and their request counts.
    deferred = [*campaign_scope(campaign, 'train')[1], *deferred]
    profile_sources, profile_failure = {}, None
    try:
        selection = verify_fitted_bundle(root, campaign)
        paths = fitted_bundle_paths(root, campaign)
        profile_sources = {name: {'path': str(path), **file_reference(path)} for name, path in paths.items()}
        if profile_sources['selection'] != {'path': str(paths['selection']), **selection}:
            raise ValueError('selected fitted bundle changed')
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        profile_failure = f'{type(exc).__name__}: {exc}'
    indexed = {cell['cell_id']: cell for cell in normalized['cells']}
    results = []
    for cell in cells:
        paired = indexed[cell['cell_id']]
        samples, failures = [], []
        if paired['status'] == 'paired':
            try:
                samples = [_sample(row) for row in paired['observations']]
            except (ValueError, KeyError, TypeError) as exc:
                failures.append(str(exc))
        else:
            failures.append(paired.get('reason', 'complete held-out role pairing is missing'))
        if profile_failure:
            failures.append('fitted profile: ' + profile_failure)
        sources = paired['observations'][0]['sources'] if paired['observations'] else {}
        result = {'cell_id': cell['cell_id'], 'name': cell['name'], 'kind': cell['kind'], 'split': cell['split'],
                  'concurrency': cell['concurrency'], 'plan': cell['plan'],
                  'status': 'measured' if not failures else 'not_measured',
                  'expected_count': paired['expected_count'], 'matched_count': len(samples),
                  'pairing': {'complete': paired['status'] == 'paired',
                              'target_zero_cost': paired['status'] == 'paired',
                              'target_sim': paired['status'] == 'paired',
                              'expected_count': paired['expected_count'], 'matched_count': paired['matched_count']},
                  'coverage': {'status': 'in_domain' if not failures else 'unverified',
                               'basis': 'successful_profiled_execution',
                               'reason': '; '.join(failures) if failures else None},
                  **{role: {'status': 'succeeded' if role in sources else 'missing',
                            'source': sources.get(role)} for role in roles},
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
        'scope': 'validation', 'scope_sha256': sha256_json(definition),
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
        'provenance': {'artifact_root': str(root), 'normalized_sha256': sha256_json(normalized),
                       'source': normalized['source'], 'profile': profile_sources,
                       'target_identity': campaign['variants']['target']['identity'], 'client': campaign['client'],
                       'reporter_sources': reporter_sources},
        'normalized': normalized,
    }
    report['acceptance'] = assess_report(report)
    report['status'] = ('succeeded' if report['acceptance']['status'] == 'passed' else
                        'partial' if any(ref['variant'] == 'sim' for ref in normalized['source']['selected_attempts'])
                        else 'replay_required')
    return report


def _validation_normalization(root: Path, campaign: dict) -> dict:
    try:
        return normalize_campaign(root, scope='validation')
    except FileNotFoundError as exc:
        if Path(exc.filename) != root/'execution_index.json' or (root/'execution_index.json').exists():
            raise
        # No execution index exists yet. Reuse normalization's cell/document
        # contracts, explicitly recording the absent source rather than a digest.
        cells = campaign_scope(campaign, 'validation')[0]
        results = []
        for cell in cells:
            result = _cell_result(cell, _read_json(checked_path(root, cell['plan']['path'])))
            result['reason'] = 'execution has not started: execution index is absent'
            results.append(result)
        sources = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                   for name in NORMALIZER_SOURCES}
        source = {'artifact_root': str(root), 'compiler_index': file_reference(root/'artifact_index.json'),
                  'execution_index_sha256_at_read': None, 'selected_attempts': [],
                  'selected_attempts_sha256': sha256_json([]),
                  'normalizer_sources': sources, 'normalizer_sha256': sha256_json(sources)}
        return _assemble_document(campaign, 'validation', results, source)


def report_campaign(root: Path) -> dict:
    root = root.expanduser().resolve()
    campaign = load_compiled_bundle(root)
    normalized = _validation_normalization(root, campaign)
    if normalized['status'] == 'succeeded':
        verify_normalized(normalized, root=root)
    return _build_report(root, campaign, normalized)


def verify_report(document: dict, *, root: Path | None = None) -> None:
    if document.get('kind') != KIND or document.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('current held-out replay report schema 2 is required')
    root = (root or Path(document['provenance']['artifact_root'])).expanduser().resolve()
    normalized = document['normalized']
    if normalized.get('status') == 'succeeded':
        verify_normalized(normalized, root=root)
    else:
        # Incomplete snapshots may never be promoted by editing their status.
        if sha256_json(_validation_normalization(root, load_compiled_bundle(root))) != sha256_json(normalized):
            raise ValueError('incomplete report no longer matches current evidence; regenerate it')
    expected = _build_report(root, load_compiled_bundle(root), normalized)
    if sha256_json(document) != sha256_json(expected):
        raise ValueError('held-out report differs from verified sources, statistics or completeness')


def write_document(path: Path, document: dict) -> None:
    verify_report(document)
    root = Path(document['provenance']['artifact_root'])
    protected = [Path(ref['path']).resolve() for ref in document['provenance']['profile'].values()]
    selection_ref = document['provenance']['profile'].get('selection')
    if selection_ref:
        selection = _read_json(Path(selection_ref['path']))
        protected.extend(Path(ref['path']).resolve() for ref in selection['training'].values())
    source = document['normalized']['source']
    for ref in source['selected_attempts']:
        protected.append((root/ref['path']).resolve())
    if (path.expanduser().resolve().is_relative_to(root/'references') or
            any(path.expanduser().resolve() == source_path or path.expanduser().resolve().is_relative_to(source_path)
                for source_path in protected)):
        raise ValueError('report output would overwrite source evidence')
    write_derived_document(path, document, root)


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
