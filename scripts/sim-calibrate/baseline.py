"""Target-first immutable references and paired, metric-specific training values."""
from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path

from artifacts import (ArtifactError, _read_json, _render, checked_path, execution_snapshot,
                       file_reference, load_compiled_bundle, sealed_reference, write_derived_document)
from manifest import sha256_json
from measurement_schema import REFERENCE_POLICIES, reference_policy
from profile import reference_bundle
from steady_state import aggregate_request_samples, summary

KIND = 'prepared_references'
SCHEMA_VERSION = 1
PREPARATION_PATH = 'references/preparation.json'
GENERATOR_SOURCES = ('baseline.py', 'profile.py', 'manifest.py', 'measurement_schema.py',
                     'artifacts.py', 'pairing.py', 'normalize.py', 'target_statistics.py',
                     'request_plan.py', 'steady_state.py')


def training_cells(campaign: dict) -> list[dict]:
    from pairing import campaign_scope
    return campaign_scope(campaign, 'train')[0]


def _reference(ref: dict) -> dict:
    return {k: v for k, v in ref.items() if k not in ('cell_id', 'variant')}


def _validate_inputs(root: Path, campaign: dict, inputs: dict) -> dict:
    from pairing import load_role
    from target_statistics import verify_target_statistics
    if inputs['compiler_index'] != file_reference(root / 'artifact_index.json'):
        raise ArtifactError('reference compiler binding changed')
    cells = training_cells(campaign)
    selected = inputs['selected_targets']
    indexed = {ref['cell_id']: ref for ref in selected}
    if (not cells or len(indexed) != len(selected) or set(indexed) != {c['cell_id'] for c in cells}
            or any(ref['variant'] != 'target' for ref in selected)):
        raise ArtifactError('reference preparation requires the complete train-target scope')
    expected_sources = {name: campaign['client']['observer']['sources'][name] for name in GENERATOR_SOURCES}
    if inputs['generator_sources'] != expected_sources:
        raise ArtifactError('reference generator differs from frozen acquisition sources')
    for cell in cells:
        plan = _read_json(checked_path(root, cell['plan']['path']))
        load_role(root, campaign, plan, 'target', _reference(indexed[cell['cell_id']]))
    decode = [cell for cell in cells if cell['kind'] == 'steady_decode']
    document = inputs['target_statistics']
    stats = {}
    if decode:
        verify_target_statistics(document, root=root)
        refs = {ref['cell_id']: _reference(ref) for ref in document['source']['selected_attempts']}
        if refs != {cell['cell_id']: _reference(indexed[cell['cell_id']]) for cell in decode}:
            raise ArtifactError('reference interval uses a different target attempt')
        stats = {cell['cell_id']: cell['statistics'] for cell in document['cells']}
    elif document is not None:
        raise ArtifactError('cold-only preparation does not consume decode statistics')
    return stats


def _construct(campaign: dict, inputs: dict, stats: dict) -> tuple[dict, dict]:
    preparation_id = sha256_json(inputs)
    indexed = {ref['cell_id']: _reference(ref) for ref in inputs['selected_targets']}
    cells, documents = {}, {}
    for cell in campaign['workload']['cells']:
        if cell['kind'] not in REFERENCE_POLICIES:
            continue
        policy = reference_policy(cell['kind'], cell['split'])
        if cell['reference_policy'] != policy:
            raise ArtifactError('compiled reference policy changed')
        requested = stats[cell['cell_id']]['summary']['mean'] if cell['kind'] == 'steady_decode' and cell['split'] == 'train' else 0.0
        if not math.isfinite(requested) or requested < 0 or requested > (2**64-1)/1000:
            raise ArtifactError('reference interval cannot be encoded as u64 microseconds')
        interval_us = round(requested * 1000)
        if cell['kind'] == 'steady_decode' and cell['split'] == 'train' and interval_us <= 0:
            raise ArtifactError('encoded decode interval must be positive')
        folder = f"references/{preparation_id}/{cell['cell_id']}"
        source = {'preparation_id': preparation_id, 'cell_id': cell['cell_id'],
                  'plan_sha256': cell['plan']['plan_sha256'], 'target': indexed.get(cell['cell_id']),
                  'policy': policy, 'interval_us': interval_us}
        manifest, profile = reference_bundle(
            campaign, cell, interval_us=interval_us, source_digest=sha256_json(source),
            source_artifacts=([indexed[cell['cell_id']]['path']] if cell['cell_id'] in indexed else []) + [cell['plan']['path']])
        paths = {'manifest_path': folder + '/manifest.json', 'profile_path': folder + '/profile.json'}
        documents[paths['manifest_path']] = manifest
        documents[paths['profile_path']] = profile
        refs = {name: {'sha256': hashlib.sha256(_render(value)).hexdigest(), 'size': len(_render(value))}
                for name, value in (('manifest', manifest), ('profile', profile))}
        cells[cell['cell_id']] = {**paths, **source, **refs, 'requested_interval_ms': requested,
                                 'encoded_interval_ms': interval_us / 1000}
    return {'kind': KIND, 'schema_version': SCHEMA_VERSION, 'campaign_id': campaign['campaign_id'],
            'preparation_id': preparation_id, 'inputs': inputs, 'cells': cells}, documents


def verify_preparation(root: Path, campaign: dict, document: dict) -> None:
    if document.get('kind') != KIND or document.get('schema_version') != SCHEMA_VERSION:
        raise ArtifactError('unsupported reference preparation')
    stats = _validate_inputs(root, campaign, document['inputs'])
    expected, files = _construct(campaign, document['inputs'], stats)
    if sha256_json(expected) != sha256_json(document):
        raise ArtifactError('reference preparation differs from pinned inputs')
    for path, value in files.items():
        if checked_path(root, path).read_bytes() != _render(value):
            raise ArtifactError(f'reference artifact differs from its declared policy: {path}')


def prepare_references(root: Path) -> dict:
    from evidence_proxy import observer_identity
    from pairing import load_role
    from target_statistics import collect_target_statistics
    root = root.expanduser().resolve()
    campaign = load_compiled_bundle(root)
    if observer_identity() != campaign['client']['observer']:
        raise ArtifactError('reference generator changed; compile a new campaign')
    output = root / PREPARATION_PATH
    if output.exists():
        document = _read_json(output)
        verify_preparation(root, campaign, document)
        return document
    cells = training_cells(campaign)
    target = collect_target_statistics(root) if any(c['kind'] == 'steady_decode' for c in cells) else None
    selected = []
    with execution_snapshot(root, campaign['campaign_id']) as (execution, _):
        for cell in cells:
            ref = sealed_reference(execution['cells'].get(f"{cell['cell_id']}:target", {}))
            plan = _read_json(checked_path(root, cell['plan']['path']))
            load_role(root, campaign, plan, 'target', ref)
            selected.append({'cell_id': cell['cell_id'], 'variant': 'target', **ref})
    inputs = {'compiler_index': file_reference(root / 'artifact_index.json'), 'selected_targets': selected,
              'target_statistics': target,
              'generator_sources': {name: campaign['client']['observer']['sources'][name] for name in GENERATOR_SOURCES}}
    stats = _validate_inputs(root, campaign, inputs)
    document, files = _construct(campaign, inputs, stats)
    for path, value in files.items():
        write_derived_document(root / path, value, root)
    # Publish the immutable selection last; partially written bundles cannot be selected.
    write_derived_document(output, document, root)
    verify_preparation(root, campaign, document)
    return document


def select_reference(root: Path, campaign: dict, cell_id: str) -> dict:
    path = root / PREPARATION_PATH
    binding = {'path': PREPARATION_PATH, **file_reference(path), 'cell_id': cell_id}
    resolve_reference(root, campaign, binding, cell_id)
    return binding


def resolve_reference(root: Path, campaign: dict, binding: dict, cell_id: str) -> dict:
    if (set(binding) != {'path', 'sha256', 'size', 'cell_id'} or binding['cell_id'] != cell_id
            or binding['path'] != campaign['variants']['zero_cost']['reference_preparation']):
        raise ArtifactError('reference selection belongs to another cell or preparation')
    path = checked_path(root, binding['path'])
    if file_reference(path) != {k: binding[k] for k in ('sha256', 'size')}:
        raise ArtifactError('selected reference preparation bytes changed')
    document = _read_json(path)
    verify_preparation(root, campaign, document)
    if cell_id not in document['cells']:
        raise ArtifactError('no train-derived reference for this cell')
    return document['cells'][cell_id]


def derive_cell(cell: dict, observations: list[dict], reference: dict) -> dict:
    """Keep signed request values; qualify only the complete cell representative."""
    decode = cell['kind'] == 'steady_decode'
    metric = 'tpot_ms' if decode else 'ttft_ms'
    interval_ms = reference['interval_us'] / 1000
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
