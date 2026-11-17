"""One complete-batch decode statistic, shared by sealing and offline consumers."""
from __future__ import annotations

import statistics

from measurement_schema import DECODE_STATISTICS_POLICY, validate_envelope
from request_plan import validate_plan

KIND = 'decode_window_statistics'
SCHEMA_VERSION = 2


class StatisticsError(ValueError):
    """A decode statistic cannot be established from the frozen evidence."""


def requires_qualified_window(plan: dict) -> bool:
    """Only frozen train plans require A-qualified windows for consumption.

    Held-out plans retain the same recomputable statistics as diagnostics;
    their measured metric is complete request TTFT, including zero controls.
    """
    return plan['split'] == 'train'


def summary(values: list[float]) -> dict:
    """Dispersion describes repeat values; mean is the sole representative."""
    if not values:
        raise StatisticsError('no measured repeats')
    median = statistics.median(values)
    return {'count': len(values), 'mean': statistics.mean(values),
            'min': min(values), 'max': max(values),
            'stddev': statistics.stdev(values) if len(values) > 1 else None,
            'mad': statistics.median(abs(value - median) for value in values)}


def _batch(batch: dict, rows: dict) -> dict:
    keys = batch['request_keys']
    errors = []
    result = {'phase': batch['phase'], 'repeat': batch['repeat'], 'keys': keys,
              'window_start': None, 'window_end': None, 'window_ms': None,
              'requests': [], 'mean_tpot_ms': None, 'errors': errors, 'qualified': False}
    if any(key not in rows or rows[key]['status'] != 'succeeded' for key in keys):
        errors.append('missing or unsuccessful request in complete batch')
        return result
    start = max(rows[key]['first_token_timestamp'] for key in keys)
    end = min(rows[key]['last_token_timestamp'] for key in keys)
    result.update(window_start=start, window_end=end, window_ms=(end-start)*1000)
    if end <= start:
        errors.append('non-positive common decode window')
    for key in keys:
        row = rows[key]
        times = row['generation_event_timestamps']
        counts = row['interval_token_counts']
        selected = [j for j in range(1, len(times)) if start <= times[j-1] <= times[j] <= end]
        duration = sum(times[j] - times[j-1] for j in selected) * 1000
        tokens = sum(counts[j-1] for j in selected)
        distinct = {times[k] for j in selected for k in (j-1, j)}
        tpot = duration / tokens if duration > 0 and tokens > 0 else None
        result['requests'].append({
            'key': key, 'index': row['index'], 'retained_event_indices': selected,
            'event_interval_count': len(selected), 'interval_token_count': tokens,
            'distinct_timestamp_count': len(distinct), 'duration_ms': duration, 'tpot_ms': tpot,
        })
        if tpot is None:
            errors.append(f'{key}: no complete interval with positive total duration and token denominator')
    if not errors:
        result.update(qualified=True, mean_tpot_ms=statistics.mean(r['tpot_ms'] for r in result['requests']))
    return result


def compute_statistics(plan: dict, envelope: dict) -> dict:
    validate_plan(plan)
    validate_envelope(envelope)
    if (plan['workload_kind'] != 'steady_decode'
            or plan.get('decode_statistics') != DECODE_STATISTICS_POLICY
            or envelope['cell_id'] != plan['cell_id']):
        raise StatisticsError('decode statistics plan/measurement contract mismatch')
    rows = {row['pairing_key']: row for row in envelope['measurements']}
    if set(rows) != {row['pairing_key'] for row in plan['logical_requests']}:
        raise StatisticsError('statistics require the complete plan request set')
    batches = [_batch(batch, rows) for batch in plan['batches']]
    measured = [batch for batch in batches if batch['phase'] == 'measured']
    qualified = len(measured) == plan['repeats'] and all(b['qualified'] for b in measured)
    return {
        'kind': KIND, 'schema_version': SCHEMA_VERSION, 'cell_id': plan['cell_id'],
        'variant': envelope['variant'],
        'policy': dict(DECODE_STATISTICS_POLICY), 'qualified': qualified,
        'coordinate': {'requests': plan['concurrency'], 'num_ctx': plan['num_ctx']},
        'repeat_count': len(measured), 'request_count': len(measured)*plan['concurrency'],
        'batches': batches,
        'summary': summary([b['mean_tpot_ms'] for b in measured]) if qualified else None,
        'failures': [{'repeat': b['repeat'], 'errors': b['errors']} for b in measured if not b['qualified']],
    }


def verify_statistics(plan: dict, envelope: dict, document: dict) -> None:
    if document != compute_statistics(plan, envelope):
        raise StatisticsError('decode statistics differ from pinned measurement evidence')


def aggregate_request_samples(samples: list[dict]) -> dict:
    """Aggregate complete request rows into repeats, without treating rows as repeats."""
    batches = {}
    for row in samples:
        if type(row.get('repeat')) is not int or type(row.get('index')) is not int:
            raise StatisticsError('decode samples require explicit repeat/index')
        batches.setdefault((row['cell_id'], row['repeat']), []).append(row)
    means, repeats = [], []
    for (cell_id, repeat), rows in batches.items():
        size = rows[0]['requests']
        if (size != int(size) or len(rows) != size or {r['index'] for r in rows} != set(range(int(size)))
                or any(r['requests'] != size for r in rows)):
            raise StatisticsError('incomplete or duplicate decode batch samples')
        mean = statistics.mean(row['value'] for row in rows)
        means.append(mean)
        repeats.append({'cell_id': cell_id, 'repeat': repeat, 'request_count': len(rows), 'mean': mean})
    dispersion = summary(means)
    return {'value': dispersion['mean'], 'repeat_count': len(means),
            'request_count': len(samples), 'dispersion': dispersion, 'batches': repeats}
