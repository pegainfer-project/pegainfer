#!/usr/bin/env python3
"""Canonical request identity and fail-closed metric pairing helpers."""

from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
from typing import Any

from manifest import SPLITS, sha256_json


PAIRING_KEY_SCHEMA = ("cell_id", "phase", "repeat", "index")
PairingKey = tuple[str, str, int, int]


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


def canonical_pair_key(measurement: dict[str, Any], cell_id: str, *, measured_only: bool = False) -> PairingKey:
    """Return the variant-independent identity for one request measurement."""
    if not isinstance(cell_id, str) or not cell_id:
        raise PairingContractError("cell_id must be a non-empty string")
    phase = measurement.get("phase")
    if phase not in ("warmup", "measured"):
        raise PairingContractError(f"measurement.phase must be warmup or measured, got {phase!r}")
    if measured_only and phase != "measured":
        raise PairingContractError(f"measurement.phase must be measured, got {phase!r}")
    repeat = measurement.get("repeat")
    index = measurement.get("index")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
        raise PairingContractError("measurement.repeat must be a non-negative integer")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise PairingContractError("measurement.index must be a non-negative integer")
    return (cell_id, phase, repeat, index)


def pairing_key_text(key: PairingKey) -> str:
    """Render a key for JSON diagnostics and compatibility with old artifacts."""
    return ":".join(str(part) for part in key)


def expected_measured_keys(cell: dict[str, Any], cell_id: str) -> set[PairingKey]:
    """Build the measured request set declared by a manifest cell."""
    repeats = cell.get("repeats")
    concurrency = cell.get("concurrency")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise PairingContractError("cell.repeats must be a positive integer")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise PairingContractError("cell.concurrency must be a positive integer")
    return {
        (cell_id, "measured", repeat, index)
        for repeat in range(repeats)
        for index in range(concurrency)
    }


def _finite_metric(value: Any, metric: str, *, positive: bool) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    rendered = float(value)
    if positive and rendered <= 0.0:
        return None
    return rendered


def index_metric_measurements(
    result: dict[str, Any] | None,
    *,
    cell_id: str,
    metric: str,
    positive: bool = False,
) -> dict[str, Any]:
    """Index measured values without overwriting duplicate logical requests."""
    keys: set[PairingKey] = set()
    values: dict[PairingKey, float] = {}
    request_ids: dict[PairingKey, str] = {}
    seen_request_ids: dict[str, PairingKey] = {}
    duplicate_keys: set[PairingKey] = set()
    duplicate_request_ids: set[str] = set()
    invalid_records: list[str] = []
    invalid_metrics: list[str] = []
    invalid_request_ids: list[str] = []

    measurements = result.get("measurements") if isinstance(result, dict) else None
    if not isinstance(measurements, list):
        invalid_records.append("measurements is not a list")
        measurements = []
    for ordinal, item in enumerate(measurements):
        if not isinstance(item, dict):
            invalid_records.append(f"ordinal:{ordinal}")
            continue
        if item.get("phase") == "warmup":
            continue
        try:
            key = canonical_pair_key(item, cell_id, measured_only=True)
        except PairingContractError as exc:
            invalid_records.append(f"ordinal:{ordinal}:{exc}")
            continue
        key_text = pairing_key_text(key)
        explicit_pairing_key = item.get("pairing_key")
        if explicit_pairing_key is not None and explicit_pairing_key != key_text:
            invalid_records.append(f"{key_text}:pairing_key does not match canonical key")
        if key in keys:
            duplicate_keys.add(key)
        keys.add(key)

        request_id = item.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            invalid_request_ids.append(key_text)
        elif request_id in seen_request_ids:
            duplicate_request_ids.add(request_id)
        else:
            seen_request_ids[request_id] = key
            request_ids[key] = request_id

        if item.get("ok") is not True:
            invalid_records.append(f"{key_text}:measurement is not successful")
            continue
        value = _finite_metric(item.get(metric), metric, positive=positive)
        if value is None:
            invalid_metrics.append(key_text)
            continue
        if key not in values:
            values[key] = value

    return {
        "keys": keys,
        "values": values,
        "request_ids": request_ids,
        "duplicate_keys": duplicate_keys,
        "duplicate_request_ids": duplicate_request_ids,
        "invalid_records": invalid_records,
        "invalid_metrics": invalid_metrics,
        "invalid_request_ids": invalid_request_ids,
        "result_status": result.get("status") if isinstance(result, dict) else None,
    }


def pair_metric_measurements(
    target: dict[str, Any] | None,
    predicted: dict[str, Any] | None,
    *,
    cell: dict[str, Any],
    cell_id: str,
    metric: str,
    positive: bool = False,
) -> dict[str, Any]:
    """Pair two variants against the manifest's exact measured request set."""
    try:
        expected = expected_measured_keys(cell, cell_id)
        contract_error = None
    except PairingContractError as exc:
        expected = set()
        contract_error = str(exc)
    target_index = index_metric_measurements(target, cell_id=cell_id, metric=metric, positive=positive)
    predicted_index = index_metric_measurements(predicted, cell_id=cell_id, metric=metric, positive=positive)
    target_keys = target_index["keys"]
    predicted_keys = predicted_index["keys"]
    matched_keys = set(target_index["values"]) & set(predicted_index["values"])
    request_id_mismatches = sorted(
        pairing_key_text(key)
        for key in matched_keys
        if target_index["request_ids"].get(key) is not None
        and predicted_index["request_ids"].get(key) is not None
        and target_index["request_ids"][key] != predicted_index["request_ids"][key]
    )
    missing_target = expected - target_keys
    missing_predicted = expected - predicted_keys
    extra_target = target_keys - expected
    extra_predicted = predicted_keys - expected
    complete = (
        contract_error is None
        and target_index["result_status"] == "succeeded"
        and predicted_index["result_status"] == "succeeded"
        and target_keys == expected
        and predicted_keys == expected
        and len(target_index["values"]) == len(expected)
        and len(predicted_index["values"]) == len(expected)
        and not target_index["duplicate_keys"]
        and not predicted_index["duplicate_keys"]
        and not target_index["duplicate_request_ids"]
        and not predicted_index["duplicate_request_ids"]
        and not target_index["invalid_records"]
        and not predicted_index["invalid_records"]
        and not target_index["invalid_metrics"]
        and not predicted_index["invalid_metrics"]
        and not target_index["invalid_request_ids"]
        and not predicted_index["invalid_request_ids"]
    )
    diagnostics = {
        "status": "complete" if complete else "pairing_failed",
        "complete": complete,
        "key_schema": list(PAIRING_KEY_SCHEMA),
        "expected_count": len(expected),
        "target_count": len(target_keys),
        "predicted_count": len(predicted_keys),
        "matched_count": len(matched_keys),
        "missing_target": sorted(pairing_key_text(key) for key in missing_target),
        "missing_predicted": sorted(pairing_key_text(key) for key in missing_predicted),
        "extra_target": sorted(pairing_key_text(key) for key in extra_target),
        "extra_predicted": sorted(pairing_key_text(key) for key in extra_predicted),
        "duplicate_keys": {
            "target": sorted(pairing_key_text(key) for key in target_index["duplicate_keys"]),
            "predicted": sorted(pairing_key_text(key) for key in predicted_index["duplicate_keys"]),
        },
        "duplicate_request_ids": {
            "target": sorted(target_index["duplicate_request_ids"]),
            "predicted": sorted(predicted_index["duplicate_request_ids"]),
        },
        "invalid_records": {
            "target": target_index["invalid_records"],
            "predicted": predicted_index["invalid_records"],
        },
        "invalid_metrics": {
            "target": target_index["invalid_metrics"],
            "predicted": predicted_index["invalid_metrics"],
        },
        "invalid_request_ids": {
            "target": target_index["invalid_request_ids"],
            "predicted": predicted_index["invalid_request_ids"],
        },
        "request_id_mismatches": request_id_mismatches,
    }
    if contract_error is not None:
        diagnostics["contract_error"] = contract_error
    pairs = []
    if complete:
        pairs = [
            (key, target_index["values"][key], predicted_index["values"][key])
            for key in sorted(expected)
        ]
    return {"pairing": diagnostics, "pairs": pairs}


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


def validate_attempt_evidence(plan: dict, attempt: dict, variant: str, client: dict, identity: dict, *, runtime: dict) -> dict:
    """Bind wire requests, read/event evidence and frozen commands to plan keys."""
    from artifacts import _read_json, checked_path
    from evidence_proxy import validate_request_body
    from probe import requires_cache_reset, validate_reset_receipt
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

    expected_terminals, expected_commands = set(), set()
    previous_end = -1
    for batch in plan['batches']:
        batch_path = f"batches/{batch['phase']}-{batch['repeat']}"
        if requires_cache_reset(plan, variant, runtime, batch):
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
            if (type(start) is not int or type(end) is not int or start < previous_end
                    or not start <= terminal['stream_end_ns'] <= end):
                raise PairingContractError(f'{variant}:{key}: batches overlap or terminal times are invalid')
            ends.append(end)
            body = read(f'{request_path}/request.json')
            wire = required_path(f'{request_path}/request-body.bin').read_bytes()
            if body != json.loads(wire):
                raise PairingContractError(f'{variant}:{key}: wire body differs from request.json')
            validate_request_body(body, plan, identity['model'])
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
    validate_attempt_evidence(plan, attempt, role, campaign['client'], configured['identity'], runtime=configured['runtime'])
    if _read_json(attempt['directory'] / 'attempt.json')['attempt'] != reference['attempt']:
        raise PairingContractError(f'{role}: raw attempt number differs from selected attempt')
    return attempt
