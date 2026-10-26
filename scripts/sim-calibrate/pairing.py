#!/usr/bin/env python3
"""Canonical request identity and fail-closed metric pairing helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from manifest import SPLITS, sha256_json


class PairingContractError(ValueError):
    """Raised when a campaign cannot establish a canonical request identity."""


def campaign_scope(campaign: dict, scope: str) -> tuple[list, list, tuple, dict]:
    if scope not in ('train', 'validation'):
        raise PairingContractError('scope must be explicitly train or validation')
    cells, deferred = [], []
    all_cells = campaign['workload']['cells']
    if (len({cell['cell_id'] for cell in all_cells}) != len(all_cells)
            or campaign['workload']['cell_count'] != len(all_cells)):
        raise PairingContractError('duplicate compiled cell identity or inconsistent cell count')
    splits = {split: [cell['cell_id'] for cell in all_cells if cell['split'] == split] for split in SPLITS}
    if (splits != campaign['splits'] or campaign['required_scope'] != {
            'cells': [cell['cell_id'] for cell in all_cells], 'sha256': sha256_json(splits)}):
        raise PairingContractError('compiled required scope/splits mismatch')
    for cell in all_cells:
        if cell['split'] not in SPLITS or cell['kind'] not in ('cold_prefill', 'steady_decode', 'cached_prefill'):
            raise PairingContractError('unknown compiled kind/split')
        if (cell['split'] == 'train') != (scope == 'train'):
            continue
        if cell['kind'] == 'cached_prefill' and scope == 'validation':
            deferred.append({'cell_id': cell['cell_id'], 'kind': cell['kind'], 'split': cell['split'],
                             'reason': 'cached-prefill runtime validation is unsupported'})
        else:
            cells.append(cell)
    roles = ('target', 'zero_cost') if scope == 'train' else ('target', 'zero_cost', 'sim')
    definition = {'scope': scope, 'required_roles': list(roles),
                  'plans': [{'cell_id': cell['cell_id'], 'plan_sha256': cell['plan']['plan_sha256']}
                            for cell in cells]}
    return cells, deferred, roles, definition


def index_plan_measurements(plan: dict, envelope: dict, workload: dict, variant: str) -> dict[str, dict]:
    """Exact current-schema identities; no metric qualification or ordinal fallback."""
    from manifest import sha256_json
    from measurement_schema import validate_envelope
    from request_plan import validate_plan

    validate_plan(plan)
    for row in envelope.get('measurements', []):
        if isinstance(row, dict) and row.get('status') == 'succeeded' and any(
                not row.get(field) for field in ('prompt_sha256', 'token_sha256', 'body_sha256',
                                                'wire_body_sha256', 'transport_request_id', 'request_instance_id')):
            raise PairingContractError(f'{variant}: unsupported_pairing: incomplete content/identity evidence')
    validate_envelope(envelope)
    if envelope['plan_sha256'] != plan['plan_sha256'] or envelope['variant'] != variant:
        raise PairingContractError(f'{variant}: measurement plan/role mismatch')
    expected = {row['pairing_key']: row for row in plan['logical_requests']}
    rows = {row['pairing_key']: row for row in envelope['measurements']}
    evidence = workload.get('requests', [])
    if (not isinstance(evidence, list) or any(not isinstance(row, dict) for row in evidence)
            or workload.get('sha256') != sha256_json(evidence)):
        raise PairingContractError(f'{variant}: invalid workload evidence')
    by_key = {row.get('pairing_key'): row for row in evidence}
    if len(by_key) != len(evidence) or set(rows) != set(expected) or set(by_key) != set(expected):
        raise PairingContractError(f'{variant}: duplicate/missing/extra plan or workload keys')
    for key, row in rows.items():
        logical = expected[key]
        if row['status'] != 'succeeded':
            raise PairingContractError(f'{variant}:{key}: request did not succeed')
        if (row['cell_id'] != plan['cell_id'] or row['planned_dispatch_offset_ms'] != logical['dispatch_offset_ms']
                or row['prompt_tokens'] != logical['input_tokens']):
            raise PairingContractError(f'{variant}:{key}: request shape/arrival differs from plan')
        for field, value in by_key[key].items():
            if row.get(field) != value:
                raise PairingContractError(f'{variant}:{key}: workload/measurement {field} mismatch')
        required = ('transport_request_id', 'request_instance_id', 'prompt_sha256', 'token_sha256',
                    'body_sha256', 'wire_body_sha256')
        if any(not by_key[key].get(field) for field in required):
            raise PairingContractError(f'{variant}:{key}: unsupported_pairing: incomplete content/identity evidence')
        if row['workload_evidence_sha256'] != workload['sha256']:
            raise PairingContractError(f'{variant}:{key}: workload digest mismatch')
        if plan['workload_kind'] == 'cached_prefill':
            from cached_prefill import validate_cached_measurement
            validate_cached_measurement(plan, row, logical)
    return rows


def pair_plan_measurements(plan: dict, attempts: dict[str, dict], required_roles: tuple[str, ...]) -> dict:
    """Pair only the complete declared role/key sets; repeated content is valid."""
    if not required_roles or len(set(required_roles)) != len(required_roles):
        raise PairingContractError('pairing requires distinct explicit roles')
    if set(attempts) != set(required_roles):
        raise PairingContractError('required pairing roles are missing or unexpected')
    indexed = {role: index_plan_measurements(plan, attempts[role]['envelope'], attempts[role]['workload'], role)
               for role in required_roles}
    pairs = []
    for logical in plan['logical_requests']:
        key = logical['pairing_key']
        rows = {role: indexed[role][key] for role in required_roles}
        for field in ('prompt_sha256', 'token_sha256', 'body_sha256'):
            if len({row[field] for row in rows.values()}) != 1:
                raise PairingContractError(f'{key}: cross-role {field} mismatch')
        if logical['phase'] == 'measured':
            pairs.append((logical, rows))
    return {'pairs': pairs, 'expected_count': len(pairs), 'matched_count': len(pairs),
            'observed_count': {role: len(rows) for role, rows in indexed.items()}}


def validate_attempt_evidence(plan: dict, attempt: dict, variant: str, client: dict, identity: dict, *, runtime: dict,
                              launch: dict | None = None) -> dict:
    """Bind wire requests, read/event evidence and frozen commands to plan keys."""
    from artifacts import _read_json, checked_path
    from evidence_proxy import validate_request_body
    from probe import requires_cache_reset, validate_reset_receipt, validate_cache_recovery_receipt
    from sse_evidence import SSEParser
    from vllm_bench import render_command
    from vllm_bench_collect import convert_evidence

    directory, raw = attempt['directory'], attempt['raw_index']
    envelope, workload = attempt['envelope'], attempt['workload']
    rows = index_plan_measurements(plan, envelope, workload, variant)
    logical_rows = {row['pairing_key']: row for row in plan['logical_requests']}

    def required_path(name):
        if name not in raw['files']:
            raise PairingContractError(f'{variant}: required raw file is not indexed: {name}')
        return checked_path(directory, name)

    def read(name):
        return _read_json(required_path(name))

    if envelope['variant_identity'] != identity or read('observer.json') != client['observer']:
        raise PairingContractError(f'{variant}: frozen variant/observer identity mismatch')
    provenance = envelope['provenance']
    for field, expected in (('client_name', 'vllm-bench'), ('client_version', client['version']),
                            ('binary_sha256', client['binary_sha256']), ('tokenizer_sha256', client['tokenizer_sha256']),
                            ('observer_sha256', client['observer']['sha256'])):
        if provenance[field] != expected:
            raise PairingContractError(f'{variant}: frozen {field} mismatch')
    attempt_record = read('attempt.json')
    if (attempt_record.get('cell_id') != plan['cell_id'] or attempt_record.get('variant') != variant
            or attempt_record.get('errors') != []):
        raise PairingContractError(f'{variant}: invalid attempt outcome')

    expected_terminals, expected_commands, expected_recoveries = set(), set(), set()
    previous_end = -1
    for batch in plan['batches']:
        batch_path = f"batches/{batch['phase']}-{batch['repeat']}"
        ready_ns = -1
        if requires_cache_reset(plan, variant, runtime, batch):
            if plan['workload_kind'] == 'cached_prefill':
                recovery_path = f'{batch_path}/recovery.json'
                expected_recoveries.add(recovery_path)
                receipt = read(recovery_path)
                ready_ns = validate_cache_recovery_receipt(receipt, plan, batch, variant_identity=identity, launch=launch)['ready_ns']
                if receipt['method'] == 'reset':
                    reset_path = f'{batch_path}/reset.json'
                    expected_recoveries.add(reset_path)
                    if read(reset_path) != receipt['reset']:
                        raise PairingContractError(f'{variant}:{batch_path}: HTTP reset receipt differs from restoration')
                if receipt['started_ns'] < previous_end:
                    raise PairingContractError(f'{variant}:{batch_path}: cache restoration overlaps the preceding request')
            else:
                validate_reset_receipt(read(f'{batch_path}/reset.json'))
        command_path = f'{batch_path}/command.json'
        expected_commands.add(command_path)
        command = read(command_path)
        argv, prefix = command['argv'], command['request_id_prefix']
        if command.get('batch') != batch or command.get('target_identity') != identity:
            raise PairingContractError(f'{variant}:{batch_path}: command batch/identity mismatch')
        if not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv):
            raise PairingContractError(f'{variant}:{batch_path}: invalid command argv')
        try:
            endpoint = argv[argv.index('--base-url') + 1]
        except (ValueError, IndexError) as exc:
            raise PairingContractError(f'{variant}:{batch_path}: missing proxy endpoint') from exc
        expected_argv = render_command(plan, batch, binary=Path(client['binary']),
                                       endpoint=endpoint, model=identity['model'],
                                       result_dir=directory / batch_path, request_id_prefix=prefix,
                                       tokenizer=client['tokenizer'])
        # Result paths are execution locations; copied evidence keeps its original argv.
        result_index = expected_argv.index('--result-dir') + 1
        if len(argv) != len(expected_argv):
            raise PairingContractError(f'{variant}:{batch_path}: command length differs from frozen plan')
        expected_argv[result_index] = argv[result_index]
        if plan['workload_kind'] == 'cached_prefill':
            from cached_prefix_workload import batch_payload
            dataset_index = expected_argv.index('--dataset-path') + 1
            # Preserve the original absolute invocation path in copied bundles,
            # but require its exact relationship to the recorded result directory.
            if argv[dataset_index] != str(Path(argv[result_index]) / 'custom.jsonl'):
                raise PairingContractError(f'{variant}:{batch_path}: custom dataset path differs from the batch')
            expected_argv[dataset_index] = argv[dataset_index]
            if required_path(f'{batch_path}/custom.jsonl').read_bytes() != batch_payload(plan, batch):
                raise PairingContractError(f'{variant}:{batch_path}: custom dataset differs from frozen content')
        if argv != expected_argv or not isinstance(prefix, str) or not prefix:
            raise PairingContractError(f'{variant}:{batch_path}: command differs from frozen plan')
        process = read(f'{batch_path}/process.json')
        if (process.get('returncode') != 0 or process.get('timed_out') is not False
                or process.get('result_present') is not True
                or process.get('invocation') != [client['process_launcher']['path'], '--pdeathsig', 'KILL', '--', *argv]):
            raise PairingContractError(f'{variant}:{batch_path}: unsuccessful or different bench invocation')
        required_path(f'{batch_path}/bench-result.json')
        ends = []
        for index, key in enumerate(batch['request_keys']):
            row = rows[key]
            transport = f'{prefix}{index}'
            instance = hashlib.sha256(transport.encode()).hexdigest()
            request_path = f'{batch_path}/requests/{instance}'
            terminal_path = f'{request_path}/terminal.json'
            expected_terminals.add(terminal_path)
            terminal = read(terminal_path)
            if (terminal != row['extensions'].get('terminal') or terminal.get('pairing_key') != key
                    or terminal.get('directory') != request_path or terminal.get('request_instance_id') != instance
                    or terminal.get('transport_request_id') != transport or row['transport_request_id'] != transport
                    or row['request_instance_id'] != instance or terminal.get('upstream_attempts') != 1
                    or any(terminal.get(field) != logical_value for field, logical_value in logical_rows[key].items())):
                raise PairingContractError(f'{variant}:{key}: terminal/request mapping differs from plan')
            start, end = terminal['upstream_dispatch_ns'], terminal['downstream_end_ns']
            if (type(start) is not int or type(end) is not int or start < max(previous_end, ready_ns)
                    or not start <= terminal['stream_end_ns'] <= end):
                raise PairingContractError(f'{variant}:{key}: batches overlap or terminal times are invalid')
            ends.append(end)
            body = read(f'{request_path}/request.json')
            wire = required_path(f'{request_path}/request-body.bin').read_bytes()
            if body != json.loads(wire):
                raise PairingContractError(f'{variant}:{key}: wire body differs from request.json')
            validate_request_body(body, plan, identity['model'], logical_rows[key])
            payload = required_path(f'{request_path}/response-body.bin').read_bytes()
            reads = [json.loads(line) for line in required_path(f'{request_path}/reads.jsonl').read_text().splitlines()]
            events = [json.loads(line) for line in required_path(f'{request_path}/events.jsonl').read_text().splitlines()]
            parser, recovered, offset, timestamp = SSEParser(), [], 0, start
            for chunk in reads:
                finish, observed = chunk['byte_end'], chunk['timestamp_ns']
                if (chunk['byte_start'] != offset or type(finish) is not int or not offset < finish <= len(payload)
                        or type(observed) is not int or not timestamp <= observed <= terminal['stream_end_ns']):
                    raise PairingContractError(f'{variant}:{key}: invalid raw read bounds/clock')
                recovered.extend(parser.feed(payload[offset:finish], observed))
                offset, timestamp = finish, observed
            recovered.extend(parser.eof())
            if offset != len(payload) or recovered != events:
                raise PairingContractError(f'{variant}:{key}: SSE events differ from raw reads')
        previous_end = max(ends)
    if ({name for name in raw['files'] if name.endswith('/terminal.json')} != expected_terminals
            or {name for name in raw['files'] if name.endswith('/command.json')} != expected_commands):
        raise PairingContractError(f'{variant}: extra/missing raw request or batch identities')
    if plan['workload_kind'] == 'cached_prefill' and {
            name for name in raw['files'] if name.endswith(('/recovery.json', '/reset.json'))} != expected_recoveries:
        raise PairingContractError(f'{variant}: extra/missing cache restoration breaks the bootstrap lifecycle')
    reconstructed = convert_evidence(plan, variant, identity, directory,
                                     raw_index_sha256=provenance['raw_bundle_index_sha256'],
                                     client=client, batch_errors=[])
    reconstructed_workload = reconstructed.pop('workload_evidence')
    if reconstructed != envelope or reconstructed_workload != workload:
        raise PairingContractError(f'{variant}: measurement/workload differs from raw conversion')
    return rows


def _profile_sources(root: Path, campaign: dict, variant: str, reference: dict) -> None:
    """Verify saved profile file bindings without consulting a current process."""
    if variant == 'target':
        return
    from artifacts import _read_json, checked_path, file_reference
    from probe import validate_profile_binding
    configured, binding = campaign['variants'][variant], reference['binding']
    if variant == 'zero_cost':
        from baseline import resolve_reference
        configured = {**configured, **resolve_reference(root, campaign, binding['reference'], binding['reference']['cell_id'])}
    if variant == 'sim':
        from profile import verify_fitted_bundle
        if verify_fitted_bundle(root, campaign) != binding.get('selection'):
            raise PairingContractError('saved fitted selection differs from verified training')

    validate_profile_binding(configured['identity']['endpoint'], binding)
    profile_path = checked_path(root, configured['profile_path'])
    if file_reference(profile_path) != binding.get('profile'):
        raise PairingContractError(f'{variant}: saved profile bytes changed')
    profile = _read_json(profile_path)
    manifest_path = checked_path(profile_path.parent, profile['calibration']['manifest'])
    if (file_reference(manifest_path) != binding.get('manifest')
            or file_reference(manifest_path)['sha256'] != profile['calibration']['sha256']):
        raise PairingContractError(f'{variant}: saved calibration manifest bytes changed')



def load_role(root: Path, campaign: dict, plan: dict, role: str, reference: dict) -> dict:
    from artifacts import _read_json, read_sealed_attempt
    if role == 'zero_cost' and reference['binding'].get('reference', {}).get('cell_id') != plan['cell_id']:
        raise PairingContractError('reference binding belongs to another cell')
    _profile_sources(root, campaign, role, reference)
    attempt = read_sealed_attempt(root, plan, role, reference)
    configured = campaign['variants'][role]
    validate_attempt_evidence(plan, attempt, role, campaign['client'], configured['identity'], runtime=configured['runtime'],
                              launch=configured.get('launch'))
    if _read_json(attempt['directory'] / 'attempt.json')['attempt'] != reference['attempt']:
        raise PairingContractError(f'{role}: raw attempt number differs from selected attempt')
    return attempt
