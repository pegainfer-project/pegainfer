#!/usr/bin/env python3
"""Original vllm-bench: one fixed batch per subprocess, logs persisted live."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import asyncio
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cached_prefill import CachedPrefillError, capture_cached_evidence, server_prompt_events, validate_cached_measurement
from measurement_schema import ROW_KIND, SCHEMA_VERSION, validate_envelope, sha256_json, observer_contract
from manifest import execution_identity


class VllmBenchError(RuntimeError):
    """Raised when vllm-bench cannot be used as a calibration driver."""


@dataclass(frozen=True)
class VllmBenchIdentity:
    """Executable location and reported version for acquisition metadata."""

    path: str
    version: str


def inspect_binary(binary: Path) -> VllmBenchIdentity:
    binary = binary.expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise VllmBenchError(f"vllm-bench is not an executable file: {binary}")
    version_run = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=30)
    if version_run.returncode != 0:
        raise VllmBenchError(f"vllm-bench --version failed with {version_run.returncode}: {version_run.stderr.strip()[:300]}")
    version = (version_run.stdout or version_run.stderr).strip().splitlines()
    return VllmBenchIdentity(path=str(binary), version=version[0].strip() if version else "unknown")


def inspect_assets(client: dict[str, Any], base: Path) -> dict[str, Any]:
    from manifest import ManifestError
    for field in ("binary", "tokenizer"):
        value = client.get(field)
        if not isinstance(value, str) or not value:
            raise ManifestError(f"client.{field} must name a readable local path")
        path = Path(value).expanduser()
        client[field] = str((base / path).resolve() if not path.is_absolute() else path.resolve())
    binary = inspect_binary(Path(client['binary']))
    client['version'] = binary.version
    launcher = inspect_binary(Path('/usr/bin/setpriv'))
    return {"process_launcher": {"path": launcher.path, "version": launcher.version},
            "observer": observer_contract()}


def render_command(plan: dict, batch: dict, *, binary: Path, endpoint: str, model: str,
                   result_dir: Path, request_id_prefix: str, tokenizer: str) -> list[str]:
    """Render an explicitly registered batch; no cell/repeat policy lives here."""
    cached = plan['workload_kind'] == 'cached_prefill'
    if cached:
        from cached_prefill import batch_payload
        dataset = result_dir / 'custom.jsonl'
        try:
            if dataset.read_bytes() != batch_payload(plan, batch):
                raise VllmBenchError('custom dataset differs from frozen batch content/row order')
        except OSError as exc:
            raise VllmBenchError(f'cannot read frozen custom dataset {dataset}: {exc}') from exc
    elif plan['generation']['prompt_representation'] != 'token_ids':
        raise VllmBenchError('cold/decode require token-ID prompts')
    argv = [str(binary), '--backend', 'openai', '--base-url', endpoint, '--model', model,
            '--tokenizer', tokenizer, '--dataset-name', 'custom' if cached else 'random',
            '--num-prompts', str(plan['concurrency']), '--max-concurrency', str(plan['concurrency'])]
    if cached:
        argv.extend(('--dataset-path', str(dataset), '--custom-output-len', '-1', '--no-oversample',
                     '--skip-chat-template'))
    else:
        argv.extend(('--random-input-len', str(plan['shape']['input_tokens']),
                     '--random-output-len', str(plan['shape']['output_tokens']), '--random-range-ratio', '0',
                     '--prompt-token-ids'))
    argv.extend(('--seed', str(batch['seed']), '--num-warmups', '0', '--ready-check-timeout-sec', '0',
            '--request-rate', 'inf', '--disable-shuffle', '--no-steady-state',
            '--save-result', '--save-detailed', '--result-dir', str(result_dir),
            '--result-filename', 'bench-result.json', '--request-id-prefix', request_id_prefix))
    sampling = plan['sampling']
    if sampling['ignore_eos']:
        argv.append('--ignore-eos')
    for key in ('temperature', 'top_p', 'top_k'):
        if key in sampling:
            argv.extend(('--' + key.replace('_', '-'), str(sampling[key])))
    argv.extend(('--extra-body', json.dumps({'return_token_ids': True, 'echo': False, 'ignore_eos': sampling['ignore_eos']}, sort_keys=True)))
    return argv


async def run_command(argv: list[str], *, root: Path, timeout: float, launcher: dict) -> dict[str, Any]:
    """Bound the whole process group; preserve stdout/stderr even on cancellation."""
    invocation = [launcher['path'], '--pdeathsig', 'KILL', '--', *argv]
    root.mkdir(parents=True, exist_ok=True)
    process = None
    timed_out = False
    with (root / 'stdout.log').open('xb') as stdout, (root / 'stderr.log').open('xb') as stderr:
        try:
            env = {k: v for k, v in os.environ.items() if k.lower() not in
                   ('http_proxy', 'https_proxy', 'all_proxy', 'openai_api_key')}
            env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            process = await asyncio.create_subprocess_exec(*invocation, stdout=stdout, stderr=stderr,
                                                           env=env, start_new_session=True)
            try:
                await asyncio.wait_for(process.wait(), timeout)
            except asyncio.TimeoutError:
                timed_out = True
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
            for stream in (stdout, stderr):
                stream.flush()
                os.fsync(stream.fileno())
    return {'invocation': invocation, 'returncode': process.returncode, 'timed_out': timed_out,
            'result_present': (root / 'bench-result.json').is_file()}


class VllmBenchCollectError(ValueError):
    """Raw evidence is missing, inconsistent, or cannot be attributed."""


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def convert_evidence(plan: dict, variant: str, identity: dict, root: Path, *,
                     client: dict, batch_errors: list[str]) -> dict[str, Any]:
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
            'pairing_key': logical['pairing_key'],
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
              'cell_id': plan['cell_id'], 'variant': variant, 'variant_identity': execution_identity(identity),
              'provenance': {'client_name': 'vllm-bench', 'client_version': client['version'],
                             'observer': client['observer']},
              'workload_evidence': {'requests': evidence, 'sha256': workload_digest},
              'measurements': measurements}
    validate_envelope(result)
    return result
