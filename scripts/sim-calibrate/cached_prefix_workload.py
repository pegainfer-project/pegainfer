"""Frozen nested-prefix custom text for the original bench's JSONL importer.

The compiler tokenizes complete strings. Expected IDs describe the planned
content; only IDs returned by the server count as observed request evidence.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path

from manifest import canonical_json, sha256_json

GENERATOR = {"name": "nested_ascii_words", "version": 1, "add_special_tokens": False,
             "alphabet": "abcdefghijklmnopqrstuvwxyz", "suffix_mode": "request_index"}


def _words(seed: int, count: int) -> list[str]:
    alphabet = GENERATOR['alphabet']
    return [alphabet[int(sha256_json([GENERATOR['version'], seed, index])[:8], 16) % len(alphabet)]
            for index in range(count)]


def suffix_seed(plan: dict, row: dict) -> int:
    return int(sha256_json([plan['generation']['seed'], row['pairing_key'], 'suffix'])[:8], 16)


def batch_payload(plan: dict, batch: dict) -> bytes:
    """Exact importer bytes, in logical request order, without oversampling."""
    rows = {row['pairing_key']: row for row in plan['logical_requests']}
    return ''.join(canonical_json({'prompt': rows[key]['prompt'],
                                   'output_tokens': rows[key]['output_tokens']}) + '\n'
                   for key in batch['request_keys']).encode('utf-8')


def _batch_manifest(plan: dict, batch: dict) -> dict:
    payload = batch_payload(plan, batch)
    return {'phase': batch['phase'], 'repeat': batch['repeat'],
            'path': f"workloads/{plan['cell_id']}/{batch['phase']}-{batch['repeat']}.jsonl",
            'sha256': hashlib.sha256(payload).hexdigest(), 'size': len(payload),
            'rows': [{'line': index + 1, 'pairing_key': key}
                     for index, key in enumerate(batch['request_keys'])]}


def workload_documents(plan: dict) -> dict[str, bytes]:
    """Canonical compiled files. The driver copies the same bytes into each batch."""
    if plan['workload_kind'] != 'cached_prefill':
        return {}
    return {_batch_manifest(plan, batch)['path']: batch_payload(plan, batch)
            for batch in plan['batches']}


def prepare_plan_workload(plan: dict, client: dict) -> None:
    """Attach text, expected IDs, tokenizer identity and exact row/file mappings."""
    from tokenizers import Tokenizer
    path = Path(client['tokenizer']) / 'tokenizer.json'
    try:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != client['tokenizer_files'].get('tokenizer.json'):
            raise ValueError('tokenizer.json changed during compilation')
        tokenizer = Tokenizer.from_str(payload.decode('utf-8'))
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
        'generator': dict(GENERATOR), 'tokenizer': {'implementation': 'tokenizers',
            'version': version('tokenizers'), 'tokenizer_sha256': client['tokenizer_sha256'],
            'files': dict(client['tokenizer_files'])},
        'prefix_seed': prefix_seed, 'prefix_token_ids': prefix_ids,
        'prefix_sha256': sha256_json(prefix_ids),
        'batches': [_batch_manifest(plan, batch) for batch in plan['batches']],
    }
    validate_cached_workload(plan)


def validate_cached_workload(plan: dict) -> None:
    """Validate serialized content and mappings without requiring a live model."""
    workload = plan.get('cached_workload')
    expected_fields = {'generator', 'tokenizer', 'prefix_seed', 'prefix_token_ids', 'prefix_sha256', 'batches'}
    if not isinstance(workload, dict) or set(workload) != expected_fields or workload['generator'] != GENERATOR:
        raise ValueError('missing or unsupported cached workload generation contract')
    identity = workload['tokenizer']
    if not isinstance(identity, dict) or set(identity) != {'implementation', 'version', 'tokenizer_sha256', 'files'}:
        raise ValueError('invalid cached tokenizer identity')
    if identity['implementation'] != 'tokenizers' or not isinstance(identity['version'], str) or not identity['version']:
        raise ValueError('invalid cached tokenizer implementation/version')
    if not isinstance(identity['files'], dict) or identity['tokenizer_sha256'] != plan['client']['tokenizer_sha256'] or sha256_json(identity['files']) != identity['tokenizer_sha256'] or 'tokenizer.json' not in identity['files']:
        raise ValueError('cached tokenizer files differ from client identity')
    from measurement_schema import digest
    for name, checksum in identity['files'].items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('invalid cached tokenizer filename')
        digest(checksum, 'cached tokenizer file')
    prefix_ids = workload['prefix_token_ids']
    if not isinstance(prefix_ids, list) or len(prefix_ids) != plan['prefix_tokens'] or sha256_json(prefix_ids) != workload['prefix_sha256']:
        raise ValueError('cached canonical prefix IDs/digest mismatch')
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
        raise ValueError('cached custom JSONL digest/row mapping mismatch')


def validate_workload_collection(plans: list[dict]) -> None:
    """Enforce actual token nesting across P axes and equal content at equal P."""
    families = {}
    for plan in plans:
        if plan['workload_kind'] != 'cached_prefill':
            continue
        validate_cached_workload(plan)
        key = (plan['shared_prefix']['identity'], plan['generation']['seed'],
               plan['client']['tokenizer_sha256'])
        entries = families.setdefault(key, {})
        ids = plan['cached_workload']['prefix_token_ids']
        for other in entries.values():
            width = min(len(ids), len(other))
            if ids[:width] != other[:width]:
                raise ValueError('cached actual token prefixes violate canonical nested slicing')
        entries[len(ids)] = ids
