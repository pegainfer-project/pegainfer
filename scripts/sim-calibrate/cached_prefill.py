"""Nested cached text preparation and attributable HTTP/SSE qualification."""
from __future__ import annotations

import math
from pathlib import Path

from measurement_schema import canonical_json, sha256_json


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


GENERATOR = {"name": "nested_ascii_words", "version": 1, "add_special_tokens": False,
             "alphabet": "abcdefghijklmnopqrstuvwxyz", "suffix_mode": "request_index"}


def _words(seed: int, count: int) -> list[str]:
    alphabet = GENERATOR['alphabet']
    return [alphabet[int(sha256_json([GENERATOR['version'], seed, index])[:8], 16) % len(alphabet)]
            for index in range(count)]


def suffix_seed(plan: dict, row: dict) -> int:
    batch = next(batch for batch in plan['batches']
                 if (batch['phase'], batch['repeat']) == (row['phase'], row['repeat']))
    return int(sha256_json(['stable_coordinates_v1', batch['seed'], row['index'], 'suffix'])[:8], 16)


def batch_payload(plan: dict, batch: dict) -> bytes:
    """Exact importer bytes, in logical request order, without oversampling."""
    rows = {row['pairing_key']: row for row in plan['logical_requests']}
    return ''.join(canonical_json({'prompt': rows[key]['prompt'],
                                   'output_tokens': rows[key]['output_tokens']}) + '\n'
                   for key in batch['request_keys']).encode('utf-8')


def _batch_manifest(plan: dict, batch: dict) -> dict:
    return {'phase': batch['phase'], 'repeat': batch['repeat'],
            'path': f"workloads/{plan['cell_id']}/{batch['phase']}-{batch['repeat']}.jsonl",
            'rows': [{'line': index + 1, 'pairing_key': key}
                     for index, key in enumerate(batch['request_keys'])]}


def workload_documents(plan: dict) -> dict[str, bytes]:
    """Canonical compiled files. The driver copies the same bytes into each batch."""
    if plan['workload_kind'] != 'cached_prefill':
        return {}
    return {_batch_manifest(plan, batch)['path']: batch_payload(plan, batch)
            for batch in plan['batches']}


def prepare_plan_workload(plan: dict, client: dict) -> None:
    """Attach actual text, expected IDs and exact row/file mappings."""
    from tokenizers import Tokenizer
    path = Path(client['tokenizer']) / 'tokenizer.json'
    try:
        tokenizer = Tokenizer.from_file(str(path))
    except (OSError, ValueError) as exc:
        raise ValueError(f'cached custom preparation requires a readable local tokenizer.json: {exc}') from exc
    prefix_length = plan['prefix_tokens']
    prefix_seed = plan['batches'][0]['prefix_seed']
    prefix = ' '.join(_words(prefix_seed, prefix_length))
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
    if len(prefix_ids) != prefix_length:
        raise ValueError('tokenizer cannot represent this canonical nested prefix at the requested length')
    seen_suffixes = set()
    for row in plan['logical_requests']:
        if row['phase'] == 'bootstrap':
            prompt = prefix
        else:
            row['suffix_seed'] = suffix_seed(plan, row)
            suffix = ' '.join(_words(row['suffix_seed'], plan['new_tokens']))
            if suffix in seen_suffixes:
                raise ValueError('independent suffixes collided; choose a new seed or larger new_tokens')
            seen_suffixes.add(suffix)
            prompt = prefix + ' ' + suffix
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        if len(ids) != row['input_tokens'] or ids[:prefix_length] != prefix_ids:
            raise ValueError('complete custom string does not preserve the frozen prefix/input token shape')
        if tokenizer.decode(ids, skip_special_tokens=False) != prompt:
            raise ValueError('cached custom text must round-trip through the actual tokenizer')
        row['prompt'], row['token_ids'] = prompt, ids
    plan['cached_workload'] = {
        'generator': dict(GENERATOR), 'tokenizer': {'implementation': 'tokenizers'},
        'prefix_seed': prefix_seed, 'prefix_token_ids': prefix_ids,
        'batches': [_batch_manifest(plan, batch) for batch in plan['batches']],
    }


def validate_cached_workload(plan: dict) -> None:
    """Validate serialized content and mappings without requiring a live model."""
    workload = plan.get('cached_workload')
    expected_fields = {'generator', 'tokenizer', 'prefix_seed', 'prefix_token_ids', 'batches'}
    if not isinstance(workload, dict) or set(workload) != expected_fields or workload['generator'] != GENERATOR:
        raise ValueError('missing or unsupported cached workload generation contract')
    if workload['tokenizer'] != {'implementation': 'tokenizers'}:
        raise ValueError('invalid cached tokenizer implementation')
    prefix_ids = workload['prefix_token_ids']
    if not isinstance(prefix_ids, list) or len(prefix_ids) != plan['prefix_tokens']:
        raise ValueError('cached canonical prefix IDs/length mismatch')
    if workload['prefix_seed'] != plan['batches'][0]['prefix_seed']:
        raise ValueError('cached workload prefix seed mismatch')
    expected_prefix = ' '.join(_words(workload['prefix_seed'], plan['prefix_tokens']))
    seen_suffixes = set()
    for row in plan['logical_requests']:
        ids = row.get('token_ids')
        if not isinstance(ids, list) or len(ids) != row['input_tokens'] or any(type(token) is not int or token < 0 for token in ids) or ids[:plan['prefix_tokens']] != prefix_ids:
            raise ValueError('cached expected IDs do not match phase shape/canonical prefix')
        if row['phase'] == 'bootstrap':
            if 'suffix_seed' in row or row.get('prompt') != expected_prefix:
                raise ValueError('bootstrap must contain only the canonical prefix')
        else:
            seed = suffix_seed(plan, row)
            suffix = ' '.join(_words(seed, plan['new_tokens']))
            if row.get('suffix_seed') != seed or row.get('prompt') != expected_prefix + ' ' + suffix:
                raise ValueError('cached independent suffix/content differs from frozen seed')
            if suffix in seen_suffixes:
                raise ValueError('independent suffixes collided')
            seen_suffixes.add(suffix)
    if workload['batches'] != [_batch_manifest(plan, batch) for batch in plan['batches']]:
        raise ValueError('cached custom JSONL row mapping mismatch')


def validate_workload_collection(plans: list[dict]) -> None:
    """Enforce actual token nesting across P axes and equal content at equal P."""
    families = {}
    for plan in plans:
        if plan['workload_kind'] != 'cached_prefill':
            continue
        key = (plan['shared_prefix']['identity'], plan['generation']['seed'])
        entries = families.setdefault(key, {})
        ids = plan['cached_workload']['prefix_token_ids']
        for other in entries.values():
            width = min(len(ids), len(other))
            if ids[:width] != other[:width]:
                raise ValueError('cached actual token prefixes violate canonical nested slicing')
        entries[len(ids)] = ids
