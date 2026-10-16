#!/usr/bin/env python3
"""Normalize exact current-format evidence scopes, without pricing or fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from artifacts import (ArtifactError, _read_json, checked_path, execution_snapshot, file_reference,
                       load_compiled_bundle, read_sealed_attempt, sealed_reference, write_derived_document)
from manifest import SPLITS, sha256_json
from measurement_schema import digest
from pairing import pair_plan_measurements, validate_attempt_evidence
from probe import validate_profile_binding

SCHEMA_VERSION = 3
SUPPORTED_KINDS = ('cold_prefill', 'steady_decode')
NORMALIZER_SOURCES = ('normalize.py', 'pairing.py', 'artifacts.py', 'manifest.py', 'request_plan.py',
                      'measurement_schema.py', 'evidence_proxy.py', 'sse_evidence.py',
                      'vllm_bench.py', 'vllm_bench_collect.py', 'probe.py', 'steady_state.py')


class NormalizeError(ValueError):
    """A declared evidence scope cannot be normalized without guessing."""


def _scope(campaign: dict, scope: str) -> tuple[list, list, tuple, dict]:
    if scope not in ('train', 'validation'):
        raise NormalizeError('scope must be explicitly train or validation')
    cells, deferred = [], []
    all_cells = campaign['workload']['cells']
    if (len({cell['cell_id'] for cell in all_cells}) != len(all_cells)
            or campaign['workload']['cell_count'] != len(all_cells)):
        raise NormalizeError('duplicate compiled cell identity or inconsistent cell count')
    splits = {split: [cell['cell_id'] for cell in all_cells if cell['split'] == split] for split in SPLITS}
    if (splits != campaign['splits'] or campaign['required_scope'] != {
            'cells': [cell['cell_id'] for cell in all_cells], 'sha256': sha256_json(splits)}):
        raise NormalizeError('compiled required scope/splits mismatch')
    for cell in all_cells:
        if cell['split'] not in SPLITS or cell['kind'] not in (*SUPPORTED_KINDS, 'cached_prefill'):
            raise NormalizeError('unknown compiled kind/split')
        if (cell['split'] == 'train') != (scope == 'train'):
            continue
        if cell['kind'] == 'cached_prefill':
            deferred.append({'cell_id': cell['cell_id'], 'kind': cell['kind'], 'split': cell['split'],
                             'reason': 'cached-prefill acquisition/validation deferred to C8'})
        else:
            cells.append(cell)
    roles = ('target', 'zero_cost') if scope == 'train' else ('target', 'zero_cost', 'sim')
    definition = {'scope': scope, 'required_roles': list(roles),
                  'plans': [{'cell_id': cell['cell_id'], 'plan_sha256': cell['plan']['plan_sha256']}
                            for cell in cells]}
    return cells, deferred, roles, definition


def _profile_sources(root: Path, campaign: dict, variant: str, reference: dict) -> None:
    """Verify saved profile file bindings without consulting a current process."""
    if variant == 'target':
        return
    configured, binding = campaign['variants'][variant], reference['binding']
    validate_profile_binding(configured['identity']['endpoint'], binding)
    profile_path = checked_path(root, configured['profile_path'])
    if file_reference(profile_path) != binding.get('profile'):
        raise NormalizeError(f'{variant}: saved profile bytes changed')
    profile = _read_json(profile_path)
    manifest_path = checked_path(profile_path.parent, profile['calibration']['manifest'])
    if (file_reference(manifest_path) != binding.get('manifest')
            or file_reference(manifest_path)['sha256'] != profile['calibration']['sha256']):
        raise NormalizeError(f'{variant}: saved calibration manifest bytes changed')
    if variant == 'zero_cost' and binding['profile']['sha256'] != configured['profile_binding']['sha256']:
        raise NormalizeError('zero_cost: profile differs from compiler-generated control')


def load_role(root: Path, campaign: dict, plan: dict, role: str, reference: dict) -> dict:
    _profile_sources(root, campaign, role, reference)
    attempt = read_sealed_attempt(root, plan, role, reference)
    configured = campaign['variants'][role]
    validate_attempt_evidence(plan, attempt, role, campaign['client'], configured['identity'], runtime=configured['runtime'])
    if _read_json(attempt['directory'] / 'attempt.json')['attempt'] != reference['attempt']:
        raise NormalizeError(f'{role}: raw attempt number differs from selected attempt')
    return attempt


def _cell_result(cell: dict, plan: dict) -> dict:
    return {'cell_id': cell['cell_id'], 'name': cell['name'], 'kind': cell['kind'], 'split': cell['split'],
            'status': 'invalid', 'expected_count': sum(row['phase'] == 'measured' for row in plan['logical_requests']),
            'matched_count': 0, 'observed_count': {}, 'observations': []}


def _paired_cell(cell: dict, plan: dict, attempts: dict, sources: dict) -> dict:
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
        'measurements': measurements, 'sources': sources,
        'decode_window_tpot_ms': {role: values[logical['pairing_key']] for role, values in request_tpot.items()}
                                 if cell['kind'] == 'steady_decode' else None,
    } for logical, measurements in paired['pairs']]
    return {**_cell_result(cell, plan), 'status': 'paired', 'matched_count': paired['matched_count'],
            'observations': observations, 'observed_count': paired['observed_count'],
            'decode_statistics': {role: attempt['statistics'] for role, attempt in attempts.items()}
            if cell['kind'] == 'steady_decode' else None}


def normalize_campaign(root: Path, *, scope: str) -> dict[str, Any]:
    """Consume a complete declared scope; succeeded means paired, never fit-ready."""
    root = root.expanduser().resolve()
    campaign = load_compiled_bundle(root)
    cells, _, roles, _ = _scope(campaign, scope)
    results, selected = [], []
    with execution_snapshot(root, campaign['campaign_id']) as (execution, execution_digest):
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
                result = _paired_cell(cell, plan, attempts, sources)
            except (ArtifactError, ValueError, OSError, KeyError, TypeError) as exc:
                result.update(reason=f'{role or "pairing"}: {type(exc).__name__}: {exc}')
            results.append(result)
    source_files = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                    for name in NORMALIZER_SOURCES}
    source = {'artifact_root': str(root), 'compiler_index': file_reference(root / 'artifact_index.json'),
              'execution_index_sha256_at_read': execution_digest,
              'selected_attempts': selected, 'selected_attempts_sha256': sha256_json(selected),
              'normalizer_sources': source_files, 'normalizer_sha256': sha256_json(source_files)}
    return _assemble_document(campaign, scope, results, source)


def _assemble_document(campaign: dict, scope: str, results: list, source: dict) -> dict:
    """One result contract for normalization and verification of saved output."""
    cells, deferred, roles, definition = _scope(campaign, scope)
    failures = [{'cell_id': cell['cell_id'], 'reason': cell['reason']} for cell in results if cell['status'] != 'paired']
    if not cells:
        failures.append({'cell_id': None, 'reason': 'required cold/decode scope is empty'})
    observations = [row for cell in results for row in cell['observations']]
    return {
        'kind': 'normalized_campaign', 'schema_version': SCHEMA_VERSION, 'campaign_id': campaign['campaign_id'],
        'status': 'invalid' if failures else 'succeeded', 'scope': scope, 'scope_sha256': sha256_json(definition),
        'required_roles': list(roles), 'required_cell_ids': [cell['cell_id'] for cell in cells],
        'campaign_required_scope': campaign['required_scope'], 'deferred_cells': deferred,
        'fit_ready': False, 'fit_pending': ['C5 validated paired baseline subtraction'],
        'cell_count': len(cells), 'paired_cell_count': sum(cell['status'] == 'paired' for cell in results),
        'expected_count': sum(cell['expected_count'] for cell in results),
        'matched_count': len(observations), 'failure_count': len(failures), 'failures': failures,
        'source': source,
        'cells': results, 'observations': observations,
    }


def verify_normalized_sources(document: dict, *, root: Path | None = None) -> dict:
    """Recheck pinned sources and return canonical results, without checking saved observations."""
    if (document.get('kind') != 'normalized_campaign' or document.get('schema_version') != SCHEMA_VERSION
            or document.get('status') != 'succeeded'):
        raise NormalizeError('source verification requires succeeded normalized schema 3')
    source = document['source']
    root = (root or Path(source['artifact_root'])).expanduser().resolve()
    campaign = load_compiled_bundle(root)
    cells, deferred, roles, definition = _scope(campaign, document['scope'])
    if (not cells or campaign['campaign_id'] != document['campaign_id']
            or file_reference(root / 'artifact_index.json') != source['compiler_index']
            or sha256_json(definition) != document['scope_sha256']
            or document['required_roles'] != list(roles)
            or document['required_cell_ids'] != [cell['cell_id'] for cell in cells]
            or document['campaign_required_scope'] != campaign['required_scope'] or document['deferred_cells'] != deferred):
        raise NormalizeError('normalized compiler/scope binding mismatch')
    selected = source['selected_attempts']
    if sha256_json(selected) != source['selected_attempts_sha256']:
        raise NormalizeError('selected attempt snapshot digest mismatch')
    indexed = {(ref['cell_id'], ref['variant']): ref for ref in selected}
    if (len(indexed) != len(selected)
            or set(indexed) != {(cell['cell_id'], role) for cell in cells for role in roles}):
        raise NormalizeError('selected attempt snapshot scope mismatch')
    results = []
    for cell in cells:
        plan = _read_json(checked_path(root, cell['plan']['path']))
        attempts, sources = {}, {}
        for role in roles:
            ref = {k: v for k, v in indexed[cell['cell_id'], role].items() if k not in ('cell_id', 'variant')}
            sources[role] = ref
            attempts[role] = load_role(root, campaign, plan, role, ref)
        results.append(_paired_cell(cell, plan, attempts, sources))
    return _assemble_document(campaign, document['scope'], results, source)


def verify_normalized(document: dict, *, root: Path | None = None) -> None:
    """Validate a complete succeeded output against its pinned source evidence."""
    expected = verify_normalized_sources(document, root=root)
    source = document['source']
    digest(source['execution_index_sha256_at_read'], 'execution_index_sha256_at_read')
    source_files = source['normalizer_sources']
    if not isinstance(source_files, dict) or not source_files:
        raise NormalizeError('normalizer source map must be nonempty')
    for name, value in source_files.items():
        digest(value, f'normalizer_sources.{name}')
    if sha256_json(source_files) != source['normalizer_sha256']:
        raise NormalizeError('normalizer source map digest mismatch')
    if sha256_json(document) != sha256_json(expected):
        raise NormalizeError('normalized results differ from pinned source evidence')


def write_document(path: Path, document: dict) -> None:
    if document.get('status') == 'succeeded':
        verify_normalized(document)
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
