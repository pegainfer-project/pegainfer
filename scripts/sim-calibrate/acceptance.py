"""Held-out report completeness; numerical accuracy has no implicit threshold."""
from __future__ import annotations

import math


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
    if document['normalized']['status'] != 'succeeded':
        failures.append('complete source-verified validation normalization is missing')
    return {'status': 'failed' if failures else 'passed', 'failures': failures,
            'basis': 'evidence_scope_pairing_statistics', 'accuracy_assessed': False}


def require_complete(document: dict) -> None:
    result = assess_report(document)
    if result['status'] != 'passed':
        raise ValueError('held-out completeness failed: ' + '; '.join(result['failures']))
    from report import verify_report
    verify_report(document)
    if document['status'] != 'succeeded':
        raise ValueError('held-out report is not complete')
