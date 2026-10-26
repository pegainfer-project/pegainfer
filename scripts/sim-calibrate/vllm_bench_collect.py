"""Convert actual proxy evidence; bench JSON is diagnostic only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cached_prefill import CachedPrefillError, capture_cached_evidence, server_prompt_events, validate_cached_measurement
from manifest import sha256_json
from measurement_schema import ROW_KIND, SCHEMA_VERSION, validate_envelope


class VllmBenchCollectError(ValueError):
    """Raw evidence is missing, inconsistent, or cannot be attributed."""


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def convert_evidence(plan: dict, variant: str, identity: dict, root: Path, *,
                     raw_index_sha256: str, client: dict, batch_errors: list[str]) -> dict[str, Any]:
    """Require a complete plan key set, retaining explicit missing/failed rows."""
    records = {}
    for path in sorted((root / 'batches').glob('*/requests/*/terminal.json')):
        record = json.loads(path.read_text())
        key = record['pairing_key']
        if key in records:
            raise VllmBenchCollectError('duplicate proxy terminal pairing key')
        records[key] = (record, path.parent)
    expected = {r['pairing_key'] for r in plan['logical_requests']}
    if set(records) - expected:
        raise VllmBenchCollectError('unknown proxy pairing key')
    evidence = []
    measurements = []
    for logical in plan['logical_requests']:
        record, directory = records.get(logical['pairing_key'], ({}, None))
        start = record.get('upstream_dispatch_ns')
        end = record.get('stream_end_ns')
        row = {
            'kind': ROW_KIND, 'schema_version': SCHEMA_VERSION, 'cell_id': plan['cell_id'],
            'plan_sha256': plan['plan_sha256'], 'pairing_key': logical['pairing_key'],
            **{k: logical[k] for k in ('phase', 'repeat', 'index')}, 'variant': variant,
            'status': 'failed', 'error': record.get('error') or 'request not observed',
            'planned_dispatch_offset_ms': logical['dispatch_offset_ms'],
            'measurement_origin': 'http_sse_proxy', 'event_source': 'proxy_upstream_read',
            'actual_dispatch_timestamp': start / 1e9 if start is not None else None,
            'stream_end_timestamp': end / 1e9 if end is not None else None,
            'transport_request_id': record.get('transport_request_id'),
            'request_instance_id': record.get('request_instance_id'),
            'first_token_timestamp': None, 'last_token_timestamp': None, 'first_text_timestamp': None,
            'ttft_ms': None, 'tpot_ms': None, 'first_text_latency_ms': None, 'e2e_ms': None,
            'generation_event_timestamps': [], 'interval_token_counts': [], 'first_event_token_count': None,
            'token_timing_valid': None, 'prompt_tokens': None, 'completion_tokens': None,
            'usage': None, 'extensions': {'terminal': record},
        }
        if start is not None and end is not None:
            row['e2e_ms'] = (end-start) / 1e6
        if directory is not None and (directory / 'request.json').is_file():
            body = json.loads((directory / 'request.json').read_text())
            prompt = body.get('prompt')
            normalized = {k: v for k, v in body.items() if k != 'model'}
            cached = plan['workload_kind'] == 'cached_prefill'
            events = _jsonl(directory / 'events.jsonl')
            prompt_events = server_prompt_events(events) if cached else []
            actual_ids = prompt_events[-1]['token_ids'] if cached and prompt_events else prompt
            row.update(prompt_sha256=sha256_json(prompt), token_sha256=sha256_json(actual_ids),
                       body_sha256=sha256_json(normalized),
                       wire_body_sha256=hashlib.sha256((directory/'request-body.bin').read_bytes()).hexdigest())
            row['prompt_tokens'] = len(actual_ids) if isinstance(actual_ids, list) else None
            row['extensions']['prompt_count_source'] = ('server_prompt_token_ids' if cached
                                                         else 'actual_request_token_ids')
            generation = [e for e in events if e['output_token_ids']]
            texts = [e for e in events if e['text'] != '']
            usages = [e['usage'] for e in events if e.get('usage') is not None]
            usage = usages[-1] if usages else None
            row['usage'] = usage
            if texts and start is not None:
                row['first_text_timestamp'] = texts[0]['timestamp_ns'] / 1e9
                row['first_text_latency_ms'] = (texts[0]['timestamp_ns']-start) / 1e6
            count = sum(len(e['output_token_ids']) for e in generation)
            if generation:
                times = [e['timestamp_ns'] / 1e9 for e in generation]
                counts = [len(e['output_token_ids']) for e in generation]
                row.update(generation_event_timestamps=times, interval_token_counts=counts[1:],
                           first_event_token_count=counts[0], first_token_timestamp=times[0],
                           last_token_timestamp=times[-1], completion_tokens=count)
                if start is not None:
                    row['ttft_ms'] = (generation[0]['timestamp_ns']-start) / 1e6
                if sum(counts[1:]):
                    row['tpot_ms'] = (generation[-1]['timestamp_ns']-generation[0]['timestamp_ns'])/1e6/sum(counts[1:])
            complete = (record.get('http_status') == 200 and record.get('eof') is True and
                        record.get('done') is True and record.get('finished') is True and
                        record.get('downstream_end_ns') is not None and record.get('error') is None)
            problem = None
            finishes = [e['finish_reason'] for e in events if e['finish_reason'] is not None]
            row['finish_reason'] = finishes[-1] if finishes else None
            if complete and (len(finishes) != 1 or plan['sampling']['ignore_eos'] and finishes[0] != 'length'):
                problem = 'finish reason differs from generation contract'

            if usage is not None:
                if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
                        for k in ('prompt_tokens','completion_tokens','total_tokens')):
                    problem = 'malformed raw usage'
                elif usage['prompt_tokens'] != row['prompt_tokens'] or usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
                    problem = 'prompt/total usage disagrees with actual request'
                elif generation and usage['completion_tokens'] != count:
                    problem = 'usage/output token-ID count mismatch'
                details = usage.get('prompt_tokens_details', {}) if isinstance(usage, dict) else {}
                if not cached and isinstance(details, dict) and details.get('cached_tokens', 0) != 0:
                    problem = 'cold/decode request reports cached input tokens'
            for event in events:
                actual_prompt = event.get('payload', {}).get('prompt_token_ids')
                if not cached and actual_prompt is not None and actual_prompt != prompt:
                    problem = 'server prompt token IDs disagree with request'
            if generation and ((plan['sampling']['ignore_eos'] and count != logical['output_tokens']) or count > logical['output_tokens']):
                problem = 'actual completion length differs from fixed-length contract'
            if complete and problem is None:
                if not generation:
                    row.update(status='unsupported', error='endpoint provided no output token IDs')
                else:
                    row.update(status='succeeded', error=None, token_timing_valid=True)
            elif problem:
                row.update(error=problem, token_timing_valid=False)
            if cached:
                row['extensions']['cached_prefill'] = capture_cached_evidence(row, events)
                if row['status'] == 'succeeded':
                    try:
                        validate_cached_measurement(plan, row, logical)
                    except CachedPrefillError as exc:
                        row.update(status='failed', error=str(exc), token_timing_valid=False)
            if batch_errors:
                row.update(status='failed', error='attempt invalid: ' + '; '.join(batch_errors), token_timing_valid=False)
            evidence.append({k: row.get(k) for k in ('pairing_key','transport_request_id','request_instance_id',
                                                    'prompt_sha256','token_sha256','body_sha256','wire_body_sha256')})
        measurements.append(row)
    workload_digest = sha256_json(evidence)
    for row in measurements:
        row['workload_evidence_sha256'] = workload_digest
    result = {'kind': 'measurement_envelope', 'schema_version': SCHEMA_VERSION,
              'plan_sha256': plan['plan_sha256'], 'variant': variant, 'variant_identity': identity,
              'provenance': {'client_name': 'vllm-bench', 'client_version': client['version'],
                             'binary_sha256': client['binary_sha256'], 'tokenizer_sha256': client['tokenizer_sha256'],
                             'observer_sha256': client['observer']['sha256'],
                             'raw_bundle_index_sha256': raw_index_sha256,
                             'source_command_sha256': sha256_json([
                                 json.loads(p.read_text()) for p in sorted((root/'batches').glob('*/command.json'))])},
              'workload_evidence': {'requests': evidence, 'sha256': workload_digest},
              'measurements': measurements}
    validate_envelope(result)
    return result
