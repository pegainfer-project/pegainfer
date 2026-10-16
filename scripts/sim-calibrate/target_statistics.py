#!/usr/bin/env python3
"""Read verified train-target attempts into a distinct, immutable statistics artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from artifacts import (ArtifactError, _read_json, checked_path, execution_snapshot, file_reference,
                       load_compiled_bundle, sealed_reference, write_derived_document)
from manifest import sha256_json
from measurement_schema import DECODE_STATISTICS_POLICY, digest
from normalize import NORMALIZER_SOURCES, _scope, load_role

KIND = 'train_target_statistics'
SCHEMA_VERSION = 1
SOURCES = (*NORMALIZER_SOURCES, 'target_statistics.py')


def _cells(campaign: dict) -> list[dict]:
    # Reuse the compiler scope checks; this stage has its own target-only role contract.
    cells, _, _, _ = _scope(campaign, 'train')
    return [cell for cell in cells if cell['kind'] == 'steady_decode']


def _result(root: Path, campaign: dict, cell: dict, reference: dict) -> dict:
    plan = _read_json(checked_path(root, cell['plan']['path']))
    attempt = load_role(root, campaign, plan, 'target', reference)
    return {'cell_id': cell['cell_id'], 'name': cell['name'], 'status': 'qualified',
            'plan_sha256': plan['plan_sha256'], 'statistics': attempt['statistics']}


def _assemble(campaign: dict, results: list[dict], source: dict) -> dict:
    cells = _cells(campaign)
    failures = [{'cell_id': cell['cell_id'], 'reason': cell['reason']}
                for cell in results if cell['status'] != 'qualified']
    if not cells:
        failures.append({'cell_id': None, 'reason': 'train steady_decode scope is empty'})
    return {'kind': KIND, 'schema_version': SCHEMA_VERSION, 'campaign_id': campaign['campaign_id'],
            'status': 'invalid' if failures else 'succeeded', 'scope': 'train', 'required_roles': ['target'],
            'required_cell_ids': [cell['cell_id'] for cell in cells],
            'campaign_required_scope': campaign['required_scope'], 'policy': dict(DECODE_STATISTICS_POLICY),
            'fit_ready': False, 'fit_pending': ['C5 validated paired baseline subtraction'],
            'cell_count': len(cells), 'qualified_cell_count': sum(c['status'] == 'qualified' for c in results),
            'failure_count': len(failures), 'failures': failures, 'cells': results, 'source': source}


def collect_target_statistics(root: Path) -> dict:
    root = root.expanduser().resolve()
    campaign = load_compiled_bundle(root)
    results, selected = [], []
    with execution_snapshot(root, campaign['campaign_id']) as (execution, execution_digest):
        for cell in _cells(campaign):
            try:
                reference = sealed_reference(execution['cells'].get(f"{cell['cell_id']}:target", {}))
                selected.append({'cell_id': cell['cell_id'], 'variant': 'target', **reference})
                results.append(_result(root, campaign, cell, reference))
            except (ArtifactError, ValueError, OSError, KeyError, TypeError) as exc:
                results.append({'cell_id': cell['cell_id'], 'name': cell['name'], 'status': 'invalid',
                                'reason': f'{type(exc).__name__}: {exc}'})
    files = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in SOURCES}
    source = {'artifact_root': str(root), 'compiler_index': file_reference(root / 'artifact_index.json'),
              'execution_index_sha256_at_read': execution_digest,
              'selected_attempts': selected, 'selected_attempts_sha256': sha256_json(selected),
              'statistics_sources': files, 'statistics_sources_sha256': sha256_json(files)}
    return _assemble(campaign, results, source)


def verify_target_statistics(document: dict, *, root: Path | None = None) -> None:
    """Recompute the pinned attempts, independent of later execution index changes."""
    if (document.get('kind') != KIND or document.get('schema_version') != SCHEMA_VERSION
            or document.get('status') != 'succeeded'):
        raise ArtifactError('requires a succeeded current train-target statistics artifact')
    source = document['source']
    root = (root or Path(source['artifact_root'])).expanduser().resolve()
    campaign = load_compiled_bundle(root)
    if file_reference(root / 'artifact_index.json') != source['compiler_index']:
        raise ArtifactError('target statistics compiler binding changed')
    digest(source['execution_index_sha256_at_read'], 'execution_index_sha256_at_read')
    files = source['statistics_sources']
    if not isinstance(files, dict) or set(files) != set(SOURCES):
        raise ArtifactError('target statistics source map incomplete')
    for name, value in files.items():
        digest(value, f'statistics_sources.{name}')
    if sha256_json(files) != source['statistics_sources_sha256']:
        raise ArtifactError('target statistics source map digest mismatch')
    selected = source['selected_attempts']
    if sha256_json(selected) != source['selected_attempts_sha256']:
        raise ArtifactError('selected target attempts digest mismatch')
    indexed = {(ref['cell_id'], ref['variant']): ref for ref in selected}
    cells = _cells(campaign)
    if len(indexed) != len(selected) or set(indexed) != {(cell['cell_id'], 'target') for cell in cells}:
        raise ArtifactError('selected target attempts scope mismatch')
    results = [_result(root, campaign, cell, {k: v for k, v in indexed[cell['cell_id'], 'target'].items()
                                             if k not in ('cell_id', 'variant')}) for cell in cells]
    if sha256_json(document) != sha256_json(_assemble(campaign, results, source)):
        raise ArtifactError('target statistics differ from pinned source evidence')


def read_target_statistics(path: Path, *, root: Path | None = None) -> dict:
    document = _read_json(path)
    verify_target_statistics(document, root=root)
    return document


def write_document(path: Path, document: dict) -> None:
    if document['status'] == 'succeeded':
        verify_target_statistics(document)
    write_derived_document(path, document, Path(document['source']['artifact_root']))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path)
    mode.add_argument('--verify', type=Path)
    args = parser.parse_args()
    try:
        if args.verify:
            document = read_target_statistics(args.verify, root=args.artifact_root)
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
