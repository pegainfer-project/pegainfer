"""Shared cached-prefill qualification from attributable HTTP/SSE evidence."""
from __future__ import annotations

import math

from manifest import sha256_json


class CachedPrefillError(ValueError):
    """A cached request cannot establish the frozen input and exact cache hit."""


def server_prompt_events(events: list[dict]) -> list[dict]:
    """Retain both OpenAI/vLLM prompt-ID locations, without selecting a fallback."""
    observed = []
    for event in events:
        payload = event.get('payload', {})
        choices = payload.get('choices', [])
        candidates = [('payload.prompt_token_ids', payload.get('prompt_token_ids'))]
        if choices:
            candidates.append(('payload.choices[0].prompt_token_ids', choices[0].get('prompt_token_ids')))
        for source, ids in candidates:
            if ids is not None:
                observed.append({'sequence': event['sequence'], 'timestamp_ns': event['timestamp_ns'],
                                 'source': source, 'token_ids': ids})
    return observed


def capture_cached_evidence(row: dict, events: list[dict]) -> dict:
    """Retain observations only; never infer token IDs or hits from the plan."""
    terminal = row['extensions']['terminal']
    generation = [event for event in events if event['output_token_ids']]
    return {
        'source': 'request_scoped_http_sse', 'unit': 'tokens',
        **{field: row.get(field) for field in ('pairing_key', 'transport_request_id', 'request_instance_id')},
        'prompt_events': server_prompt_events(events),
        'usage_events': [
            {'sequence': event['sequence'], 'timestamp_ns': event['timestamp_ns'], 'usage': event['usage']}
            for event in events if event.get('usage') is not None
        ],
        'ttft': {'source': 'proxy_upstream_dispatch_to_first_output_token_ids',
                 'start_ns': terminal.get('upstream_dispatch_ns'),
                 'end_ns': generation[0]['timestamp_ns'] if generation else None,
                 'includes_uncached_tail': True},
    }


def validate_cached_measurement(plan: dict, row: dict, logical: dict | None = None) -> None:
    """Qualify cached rows identically during collection, pairing and fitting.

    Bootstrap proves its prefix input and full completion. Only real target
    warmup/measured requests must prove an exact P-token hit; the all-zero
    reference has no cache implementation or TPOT requirement.
    """
    if plan.get('workload_kind') != 'cached_prefill':
        return
    if row.get('status') != 'succeeded':
        raise CachedPrefillError('cached measurement did not succeed')
    if logical is None:
        logical = next((item for item in plan['logical_requests']
                        if item['pairing_key'] == row.get('pairing_key')), None)
    if logical is None or row.get('pairing_key') != logical.get('pairing_key'):
        raise CachedPrefillError('cached request has no exact logical identity')
    if row.get('variant') not in ('target', 'zero_cost'):
        raise CachedPrefillError('cached online sim validation is unsupported')
    for field in ('phase', 'repeat', 'index'):
        if row.get(field) != logical.get(field):
            raise CachedPrefillError('cached request phase/repeat/index differs from plan')
    prefix, tail = plan['prefix_tokens'], plan['new_tokens']
    expected = logical.get('token_ids')
    expected_count = prefix if logical['phase'] == 'bootstrap' else prefix + tail
    if (not isinstance(expected, list) or len(expected) != expected_count
            or any(type(token) is not int or token < 0 for token in expected)
            or logical.get('input_tokens') != expected_count or row.get('prompt_tokens') != expected_count):
        raise CachedPrefillError('cached input does not match frozen phase P/N shape')
    if row.get('prompt_sha256') != sha256_json(logical.get('prompt')) or row.get('token_sha256') != sha256_json(expected):
        raise CachedPrefillError('cached prompt/token digests differ from frozen content')
    extensions = row.get('extensions', {})
    if not isinstance(extensions, dict):
        raise CachedPrefillError('cached measurement extensions must be an object')
    terminal, evidence = extensions.get('terminal'), extensions.get('cached_prefill')
    if not isinstance(terminal, dict) or not isinstance(evidence, dict):
        raise CachedPrefillError('cached request is missing attributable SSE evidence')
    if evidence.get('source') != 'request_scoped_http_sse' or evidence.get('unit') != 'tokens':
        raise CachedPrefillError('cached evidence source/unit is invalid')
    for field in ('pairing_key', 'transport_request_id', 'request_instance_id'):
        if (not row.get(field) or evidence.get(field) != row[field] or terminal.get(field) != row[field]):
            raise CachedPrefillError('cached evidence request attribution differs from measurement')
    if (terminal.get('http_status') != 200 or terminal.get('upstream_attempts') != 1
            or any(terminal.get(field) is not True for field in ('done', 'eof', 'finished'))
            or terminal.get('error') is not None or terminal.get('downstream_end_ns') is None):
        raise CachedPrefillError('cached request did not completely terminate')
    start, end = terminal.get('upstream_dispatch_ns'), terminal.get('stream_end_ns')
    if (type(start) is not int or type(end) is not int or type(terminal['downstream_end_ns']) is not int
            or not start <= end <= terminal['downstream_end_ns']):
        raise CachedPrefillError('cached request has invalid dispatch/end timestamps')

    def observed_events(name: str) -> list[dict]:
        values = evidence.get(name)
        if not isinstance(values, list) or not values:
            raise CachedPrefillError(f'cached request is missing server {name}')
        previous, seen = -1, set()
        for event in values:
            if (not isinstance(event, dict) or type(event.get('sequence')) is not int
                    or event['sequence'] < previous or type(event.get('timestamp_ns')) is not int
                    or not isinstance(event.get('source', ''), str)
                    or not start <= event['timestamp_ns'] <= end):
                raise CachedPrefillError(f'cached {name} have invalid sequence/time attribution')
            identity = event['sequence'], event.get('source')
            if identity in seen:
                raise CachedPrefillError(f'cached {name} have duplicate event attribution')
            seen.add(identity)
            previous = event['sequence']
        return values

    for event in observed_events('prompt_events'):
        actual = event.get('token_ids')
        if (event.get('source') not in ('payload.prompt_token_ids', 'payload.choices[0].prompt_token_ids')
                or not isinstance(actual, list) or any(type(token) is not int or token < 0 for token in actual)
                or actual != expected):
            raise CachedPrefillError('server prompt token IDs differ from frozen cached input')
    usages = observed_events('usage_events')
    for event in usages:
        usage = event.get('usage')
        if (not isinstance(usage, dict) or any(type(usage.get(field)) is not int or usage[field] < 0
                for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
                or usage['prompt_tokens'] != expected_count
                or usage['completion_tokens'] != row.get('completion_tokens')
                or usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']):
            raise CachedPrefillError('cached usage disagrees with actual input/output token counts')
        details = usage.get('prompt_tokens_details')
        cached = details.get('cached_tokens') if isinstance(details, dict) else None
        if row['variant'] == 'target' and logical['phase'] != 'bootstrap':
            if type(cached) is not int or cached != prefix:
                raise CachedPrefillError(f'cached target observed_cached_tokens must equal P={prefix}, got {cached}')
        if cached is not None and (type(cached) is not int or not 0 <= cached <= expected_count):
            raise CachedPrefillError('cached token evidence is outside token bounds')
    if row.get('usage') != usages[-1]['usage']:
        raise CachedPrefillError('cached selected usage differs from observed SSE usage')
    timing = evidence.get('ttft', {})
    if not isinstance(timing, dict):
        raise CachedPrefillError('cached TTFT evidence must be an object')
    finish = timing.get('end_ns')
    if (timing.get('source') != 'proxy_upstream_dispatch_to_first_output_token_ids'
            or timing.get('includes_uncached_tail') is not True or timing.get('start_ns') != start
            or type(finish) is not int or not start <= finish <= end
            or row.get('token_timing_valid') is not True
            or not isinstance(row.get('ttft_ms'), (int, float)) or isinstance(row.get('ttft_ms'), bool)
            or not math.isfinite(row['ttft_ms'])
            or not math.isclose(row['ttft_ms'], (finish-start)/1e6, rel_tol=1e-6, abs_tol=1e-5)
            or row.get('first_token_timestamp') != finish/1e9
            or row.get('actual_dispatch_timestamp') != start/1e9):
        raise CachedPrefillError('cached TTFT does not include complete P+N request processing')
