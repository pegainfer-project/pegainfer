#!/usr/bin/env python3
"""Verify exact scopes and derive metric-specific paired training observations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from artifacts import (ArtifactError, _read_json, checked_path, execution_snapshot,
                       load_compiled_bundle, sealed_reference, write_derived_document)
from baseline import derive_cell, resolve_reference
from pairing import campaign_scope as _scope, pair_plan_measurements, load_role

SCHEMA_VERSION = 7


class NormalizeError(ValueError):
    """A declared evidence scope cannot be normalized without guessing."""


def _cell_result(cell: dict, plan: dict) -> dict:
    return {'cell_id': cell['cell_id'], 'name': cell['name'], 'kind': cell['kind'], 'split': cell['split'],
            'status': 'invalid', 'expected_count': sum(row['phase'] == 'measured' for row in plan['logical_requests']),
            'matched_count': 0, 'observed_count': {}, 'observations': []}


def _paired_cell(root: Path, campaign: dict, cell: dict, plan: dict, attempts: dict, sources: dict, scope: str) -> dict:
    paired = pair_plan_measurements(plan, attempts, tuple(sources))
    request_tpot = {role: {request['key']: request['tpot_ms']
                           for batch in attempt['statistics']['batches'] if batch['phase'] == 'measured'
                           for request in batch['requests']}
                    for role, attempt in attempts.items()} if cell['kind'] == 'steady_decode' else {}
    observations = [{
        'key': logical['pairing_key'], 'cell_id': cell['cell_id'], 'kind': cell['kind'],
        'split': cell['split'], 'repeat': logical['repeat'], 'index': logical['index'],
        'input_tokens': logical['input_tokens'], 'output_tokens': logical['output_tokens'],
        'concurrency': cell['concurrency'], 'sum_decode_ctx_tokens': cell.get('sum_decode_ctx_tokens'),
        'fit_policy': cell['fit_policy'], 'metadata': cell.get('metadata', {}),
        **({'prefix_tokens': cell['prefix_tokens'], 'new_tokens': cell['new_tokens']}
           if cell['kind'] == 'cached_prefill' else {}),
        'measurements': measurements,
        'decode_window_tpot_ms': {role: values[logical['pairing_key']] for role, values in request_tpot.items()}
                                 if cell['kind'] == 'steady_decode' else None,
    } for logical, measurements in paired['pairs']]
    result = {**_cell_result(cell, plan), 'status': 'paired', 'matched_count': paired['matched_count'],
              'observations': observations, 'observed_count': paired['observed_count'], 'sources': sources,
              'decode_statistics': {role: attempt['statistics'] for role, attempt in attempts.items()}
              if cell['kind'] == 'steady_decode' else None}
    if scope == 'train':
        reference = resolve_reference(root, campaign, sources['zero_cost']['binding']['reference'], cell['cell_id'])
        result['baseline'] = reference
        result['training'] = derive_cell(cell, observations, reference)
        result['status'] = 'measured' if result['training']['qualified'] else 'invalid'
        if result['status'] == 'invalid':
            result['reason'] = '; '.join(result['training']['errors'])
    return result


def normalize_campaign(root: Path, *, scope: str) -> dict[str, Any]:
    """Consume a complete declared scope; train succeeds only with qualified paired effective values."""
    root = root.expanduser().resolve()
    campaign = load_compiled_bundle(root)
    cells, _, roles, _ = _scope(campaign, scope)
    results, selected = [], []
    with execution_snapshot(root, campaign['campaign_id']) as execution:
        for cell in cells:
            plan = _read_json(checked_path(root, cell['plan']['path']))
            result = _cell_result(cell, plan)
            role = None
            try:
                attempts, sources = {}, {}
                for role in roles:
                    reference = sealed_reference(execution['cells'].get(f"{cell['cell_id']}:{role}", {}))
                    selected.append({'cell_id': cell['cell_id'], 'variant': role, **reference})
                    attempt = load_role(root, campaign, plan, role, reference)
                    result['observed_count'][role] = len(attempt['envelope']['measurements'])
                    attempts[role], sources[role] = attempt, reference
                result = _paired_cell(root, campaign, cell, plan, attempts, sources, scope)
            except (ArtifactError, ValueError, OSError, KeyError, TypeError) as exc:
                result.update(reason=f'{role or "pairing"}: {type(exc).__name__}: {exc}')
            results.append(result)
    source = {'artifact_root': str(root), 'selected_attempts': selected}
    return _assemble_document(campaign, scope, results, source)


def _assemble_document(campaign: dict, scope: str, results: list, source: dict) -> dict:
    """One result contract for normalization and verification of saved output."""
    cells, deferred, roles, _ = _scope(campaign, scope)
    expected_status = 'measured' if scope == 'train' else 'paired'
    failures = [{'cell_id': cell['cell_id'], 'reason': cell['reason']} for cell in results if cell['status'] != expected_status]
    if not cells:
        failures.append({'cell_id': None, 'reason': 'required evidence scope is empty'})
    if scope == 'validation' and deferred:
        failures.extend({'cell_id': cell['cell_id'], 'reason': cell['reason']} for cell in deferred)
    observations = [row for cell in results for row in cell['observations']]
    return {
        'kind': 'normalized_campaign', 'schema_version': SCHEMA_VERSION, 'campaign_id': campaign['campaign_id'],
        'status': 'invalid' if failures else 'succeeded', 'scope': scope,
        'required_roles': list(roles), 'required_cell_ids': [cell['cell_id'] for cell in cells],
        'campaign_required_scope': campaign['required_scope'], 'deferred_cells': deferred,
        'cached_requirements': [
            {**{field: cell[field] for field in ('cell_id', 'matrix_name', 'prefix_tokens', 'new_tokens', 'repeats', 'concurrency')},
             'axes': {'prefix_tokens': cell['metadata']['prefix_axis'],
                      'new_tokens': cell['metadata']['new_token_axis']}}
            for cell in cells if cell['kind'] == 'cached_prefill'],
        'fit_ready': scope == 'train' and not failures,
        'fit_pending': [] if scope == 'train' and not failures else ['complete qualified train scope required'],
        'cell_count': len(cells), 'paired_cell_count': sum(cell['status'] == expected_status for cell in results),
        'expected_count': sum(cell['expected_count'] for cell in results),
        'matched_count': len(observations), 'failure_count': len(failures), 'failures': failures,
        'source': source,
        'cells': results, 'observations': observations,
    }


def verify_normalized_sources(document: dict, *, root: Path | None = None) -> dict:
    """Explicitly replay the saved attempt selection for offline comparison."""
    verify_normalized(document)
    source = document['source']
    root = (root or Path(source['artifact_root'])).expanduser().resolve()
    campaign = load_compiled_bundle(root)
    cells, _, roles, _ = _scope(campaign, document['scope'])
    indexed = {(ref['cell_id'], ref['variant']): ref for ref in source['selected_attempts']}
    results = []
    for cell in cells:
        plan = _read_json(checked_path(root, cell['plan']['path']))
        attempts, sources = {}, {}
        for role in roles:
            ref = {k: v for k, v in indexed[cell['cell_id'], role].items() if k not in ('cell_id', 'variant')}
            sources[role] = ref
            attempts[role] = load_role(root, campaign, plan, role, ref, replay_raw=True)
        results.append(_paired_cell(root, campaign, cell, plan, attempts, sources, document['scope']))
    return _assemble_document(campaign, document['scope'], results, source)


def verify_normalized(document: dict, *, root: Path | None = None) -> None:
    """Check the direct normalized input; training reads do not replay raw evidence."""
    if (document.get('kind') != 'normalized_campaign' or document.get('schema_version') != SCHEMA_VERSION
            or document.get('status') != 'succeeded'):
        raise NormalizeError('requires a succeeded current normalized artifact')
    if (not document['cells'] or document['failure_count'] or
            document['matched_count'] != document['expected_count']):
        raise NormalizeError('normalized scope is incomplete')
    if document['scope'] == 'train' and (not document['fit_ready'] or
            any(cell['split'] != 'train' or cell['status'] != 'measured' for cell in document['cells'])):
        raise NormalizeError('training requires qualified train cells')


def write_document(path: Path, document: dict) -> None:
    write_derived_document(path, document, Path(document['source']['artifact_root']))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--scope', choices=('train', 'validation'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        document = normalize_campaign(args.artifact_root, scope=args.scope)
        write_document(args.output, document)
    except (ArtifactError, ValueError, OSError, KeyError, TypeError) as exc:
        print(f'normalization error: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({key: document[key] for key in ('campaign_id', 'status', 'scope', 'cell_count',
                                                   'matched_count', 'failure_count')}, sort_keys=True))
    return 0 if document['status'] == 'succeeded' else 1


if __name__ == '__main__':
    raise SystemExit(main())
