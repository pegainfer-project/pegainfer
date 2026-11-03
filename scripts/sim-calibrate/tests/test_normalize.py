"""Current-format evidence fixtures and production-entry contract/fault checks.

These synthetic streams exercise lineage, not model timing or GPU performance.
"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import ArtifactError, ExecutionStore, _read_json, _write_atomic, file_reference
from campaign import compile_campaign
from fit import fit_normalized
from normalize import normalize_campaign, verify_normalized, verify_normalized_sources, write_document
from probe import requires_cache_reset
from request_plan import build_plan
from sse_evidence import SSEParser
from test_campaign import make_input
from test_evidence_proxy import event
from vllm_bench import render_command
from vllm_bench import convert_evidence
from steady_state import compute_statistics
from baseline import prepare_references, resolve_reference, select_reference
from profile import materialize_profile
from manifest import execution_identity


def make_bundle(base: Path, *, cached=False, reset_before=None, full_train=False):
    """Compile current inputs; evidence is deliberately deterministic and tiny."""
    path, spec = make_input(base)
    cached_matrix = spec['workload']['matrices']['cached-train']
    matrix = {'kind': 'steady_decode', 'split': 'train', 'concurrency': [2], 'num_ctx': [16],
              'output_tokens': 8, 'warmup_repeats': 1, 'repeats': 2, 'arrival': {'mode': 'fixed_batch'},
              'cache_lifecycle': {'namespace': 'decode', 'reset_before': 'repeat', 'bootstrap_repeats': 0}}
    if reset_before:
        matrix = {'kind': 'cold_prefill', 'split': 'train', 'concurrency': 1, 'input_tokens': [8],
                  'output_tokens': 4, 'warmup_repeats': 1, 'repeats': 2, 'arrival': {'mode': 'fixed_batch'},
                  'cache_lifecycle': {'namespace': 'cold', 'reset_before': reset_before, 'bootstrap_repeats': 0}}
    spec['workload']['matrices'] = {'decode': matrix, 'heldout': {**matrix, 'split': 'sealed_test'}}
    if full_train:
        for variant in spec['variants'].values():
            variant['runtime']['cache'] = {'policy': 'off'}
        spec['workload']['matrices'] = {
            'cold': {**matrix, 'kind': 'cold_prefill', 'concurrency': 1, 'input_tokens': [8, 64]},
            'decode': {**matrix, 'concurrency': [1, 2], 'num_ctx': [16, 64]},
            'heldout': {**matrix, 'num_ctx': [32], 'split': 'sealed_test'},
        }
        spec['workload']['matrices']['cold'].pop('num_ctx')
    if cached:
        cached_matrix.update(warmup_repeats=1, repeats=2, output_tokens=1)
        spec['workload']['matrices']['cached'] = cached_matrix
    path.write_text(json.dumps(spec))
    campaign = compile_campaign(path)
    root = base / 'artifacts'
    return root, campaign


def collect_fixture(root: Path, campaign: dict, roles=('target', 'zero_cost'), *, split='train', collapsed_phases=(), event_interval_ms=5):
    """Build raw evidence through the real converter and ExecutionStore."""
    if 'zero_cost' in roles:
        if 'target' in roles:
            collect_fixture(root, campaign, ('target',), split=split, collapsed_phases=collapsed_phases,
                            event_interval_ms=event_interval_ms)
            roles = tuple(role for role in roles if role != 'target')
        prepare_references(root)
    if 'sim' in roles:
        normalized_path, fit_path = root.parent / 'profile-train.json', root.parent / 'profile-fit.json'
        if not fit_path.exists():
            normalized = normalize_campaign(root, scope='train')
            write_document(normalized_path, normalized)
            from fit import write_document as write_fit
            write_fit(fit_path, fit_normalized(normalized))
        materialize_profile(root, normalized_artifact=normalized_path, fit_artifact=fit_path)
    for role in roles:
        with ExecutionStore(root, campaign['campaign_id']) as store:
            for cell in campaign['workload']['cells']:
                if cell['split'] != split:
                    continue
                plan = build_plan(cell, campaign['client'])
                identity, client = execution_identity(campaign['variants'][role]['identity']), campaign['client']
                binding = {'variant_identity': identity}
                if role != 'target':
                    configured = campaign['variants'][role]
                    if role == 'zero_cost':
                        binding['reference'] = select_reference(root, campaign, cell['cell_id'])
                        configured = resolve_reference(root, campaign, binding['reference'], cell['cell_id'])
                    profile = root / configured['profile_path']
                    endpoint = urlsplit(identity['endpoint'])
                    manifest_path = profile.parent / _read_json(profile)['calibration']['manifest']
                    binding.update(profile=file_reference(profile), manifest=file_reference(manifest_path),
                                   endpoint={'address': endpoint.hostname, 'port': endpoint.port},
                                   listeners=[{'family': 'IPv4', 'address': '0.0.0.0', 'port': endpoint.port}])
                if store.sealed(plan, role, binding):
                    continue
                directory, number = store.begin(plan, role, binding)
                for name, value in [('binding', binding), ('observer', client['observer']), ('probe', {'model': identity['model']}),
                                    ('attempt', {'attempt': number, 'errors': [], 'cell_id': cell['cell_id'], 'variant': role})]:
                    _write_atomic(directory / f'{name}.json', value)
                logical = {row['pairing_key']: row for row in plan['logical_requests']}
                for b, batch in enumerate(plan['batches']):
                    relative = f"batches/{batch['phase']}-{batch['repeat']}"
                    batch_root = directory / relative
                    if requires_cache_reset(plan, role, campaign['variants'][role]['runtime'], batch):
                        if cell['kind'] == 'cached_prefill':
                            _write_atomic(batch_root / 'reset.json', {'status': 200, 'body': '{"success": true}'})
                            _write_atomic(batch_root / 'recovery.json', {
                                'method': 'reset', 'group': {key: batch[key] for key in ('before_phase', 'before_repeat')},
                                'started_ns': b * 1_000_000_000 + 100_000_000,
                                'ready_ns': b * 1_000_000_000 + 200_000_000, 'variant_identity': identity,
                                'readiness': {'endpoint': identity['endpoint'], 'models': {'model_ids': [identity['model']]}},
                                'reset': {'status': 200, 'body': '{"success": true}'}})
                        else:
                            _write_atomic(batch_root / 'reset.json', {'status': 200, 'body': '{"success": true}'})
                    prefix = f'fixture-{role}-{b}-'
                    if cell['kind'] == 'cached_prefill':
                        from cached_prefill import batch_payload
                        batch_root.mkdir(parents=True, exist_ok=True)
                        (batch_root / 'custom.jsonl').write_bytes(batch_payload(plan, batch))
                    argv = render_command(plan, batch, binary=Path(client['binary']), endpoint='http://127.0.0.1:1',
                                          model=identity['model'], result_dir=batch_root, request_id_prefix=prefix,
                                          tokenizer=client['tokenizer'])
                    _write_atomic(batch_root / 'command.json', {'argv': argv, 'request_id_prefix': prefix,
                                                              'batch': batch, 'target_identity': identity})
                    _write_atomic(batch_root / 'process.json', {'returncode': 0, 'timed_out': False, 'result_present': True,
                                  'invocation': [client['process_launcher']['path'], '--pdeathsig', 'KILL', '--', *argv]})
                    _write_atomic(batch_root / 'bench-result.json', {'diagnostic': True})
                    for i, key in enumerate(batch['request_keys']):
                        transport = prefix + str(i)
                        instance = hashlib.sha256(transport.encode()).hexdigest()
                        request_relative = f'{relative}/requests/{instance}'
                        request = directory / request_relative
                        request.mkdir(parents=True)
                        request_shape = logical[key]
                        body = {'model': identity['model'], 'prompt': request_shape.get('prompt', [1] * request_shape['input_tokens']),
                                'max_tokens': request_shape['output_tokens'], 'stream': True, 'return_token_ids': True, 'echo': False,
                                'stream_options': {'include_usage': True}, **plan['sampling']}
                        _write_atomic(request / 'request.json', body)
                        (request / 'request-body.bin').write_bytes(json.dumps(body).encode())
                        count = request_shape['output_tokens']
                        chunks = [event({'index': 0, 'token_ids': list(range(1, min(count, 2) + 1)), 'text': '',
                                         'finish_reason': 'length' if count <= 2 else None},
                                        **({'prompt_token_ids': request_shape['token_ids']} if cell['kind'] == 'cached_prefill' else {}))]
                        chunks.extend(event({'index': 0, 'token_ids': [j], 'text': 'x',
                                             'finish_reason': 'length' if j == count else None})
                                      for j in range(3, count + 1))
                        chunks.append(event(usage={'prompt_tokens': request_shape['input_tokens'],
                                                  'completion_tokens': count,
                                                  'total_tokens': request_shape['input_tokens'] + count,
                                                  **({'prompt_tokens_details': {'cached_tokens': cell['prefix_tokens']
                                                       if role == 'target' and batch['phase'] != 'bootstrap' else 0}}
                                                     if cell['kind'] == 'cached_prefill' else {})}) + b'data: [DONE]\r\n\r\n')
                        payload = b''.join(chunks)
                        start = 1_000_000_000 + b * 1_000_000_000 + i * 2_000_000
                        parser = SSEParser()
                        events, reads, offset = [], [], 0
                        for j, chunk in enumerate(chunks):
                            timestamp = start + 1_000_000 + (0 if batch['phase'] in collapsed_phases else j * event_interval_ms * 1_000_000)
                            events.extend(parser.feed(chunk, timestamp))
                            reads.append({'timestamp_ns': timestamp, 'byte_start': offset, 'byte_end': offset+len(chunk)})
                            offset += len(chunk)
                        events.extend(parser.eof())
                        (request / 'response-body.bin').write_bytes(payload)
                        (request / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
                        (request / 'reads.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in reads))
                        terminal = {**logical[key], 'transport_request_id': transport, 'request_instance_id': instance,
                                    'directory': request_relative, 'upstream_dispatch_ns': start,
                                    'stream_end_ns': start + (len(chunks) * event_interval_ms + 10) * 1_000_000,
                                    'downstream_end_ns': start + (len(chunks) * event_interval_ms + 11) * 1_000_000,
                                    'upstream_attempts': 1, 'http_status': 200, 'eof': True, 'done': True,
                                    'finished': True, 'error': None}
                        _write_atomic(request / 'terminal.json', terminal)
                store.raw_index(directory)
                envelope = convert_evidence(plan, role, identity, directory, client=client, batch_errors=[])
                _write_atomic(directory / 'workload_evidence.json', envelope.pop('workload_evidence'))
                _write_atomic(directory / 'measurements.json', envelope)
                if plan['workload_kind'] == 'steady_decode':
                    _write_atomic(directory / 'decode_statistics.json', compute_statistics(plan, envelope))
                store.finish(plan, role, directory)


def refresh_attempt(root: Path, campaign: dict, role='target'):
    """Regenerate selected request measurements after semantic fixture changes."""
    index = _read_json(root / 'execution_index.json')
    cell = next(c for c in campaign['workload']['cells'] if c['split'] == 'train' and c['kind'] != 'cached_prefill')
    entry = index['cells'][cell['cell_id'] + ':' + role]
    directory = root / entry['path']
    with ExecutionStore(root, campaign['campaign_id']) as store:
        store.raw_index(directory)
    plan = build_plan(cell, campaign['client'])
    envelope = convert_evidence(plan, role, execution_identity(campaign['variants'][role]['identity']),
                                directory, client=campaign['client'], batch_errors=[])
    _write_atomic(directory / 'workload_evidence.json', envelope.pop('workload_evidence'))
    _write_atomic(directory / 'measurements.json', envelope)
    if cell['kind'] == 'steady_decode':
        _write_atomic(directory / 'decode_statistics.json', compute_statistics(plan, envelope))


class NormalizeTests(unittest.TestCase):
    def test_current_train_scope_and_repeated_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, campaign = make_bundle(Path(tmp), cached=True)
            collect_fixture(root, campaign)
            document = normalize_campaign(root, scope='train')
            self.assertEqual(document['status'], 'succeeded', document['failures'])
            self.assertEqual(document['matched_count'], 12)
            self.assertEqual(len(document['deferred_cells']), 0)
            self.assertEqual(document['required_roles'], ['target', 'zero_cost'])
            self.assertTrue(document['fit_ready'])
            for observation in document['observations']:
                self.assertEqual(observation['split'], 'train')
                if observation['kind'] == 'cached_prefill':
                    self.assertEqual(observation['effective']['ttft_ms'], 0)
                    continue
                self.assertAlmostEqual(observation['baseline_residual']['tpot_ms'], 0)
                self.assertAlmostEqual(observation['effective']['tpot_ms'], 5)
                self.assertEqual(observation['reference_interval_ms'], 5)
                self.assertGreater(observation['decode_window_tpot_ms']['zero_cost'], 0)
            verify_normalized_sources(document)
        for reset_before in ('cell', 'repeat'):
            with self.subTest(reset_before=reset_before), tempfile.TemporaryDirectory() as tmp:
                root, campaign = make_bundle(Path(tmp), reset_before=reset_before)
                collect_fixture(root, campaign)
                document = normalize_campaign(root, scope='train')
                self.assertEqual(document['status'], 'succeeded', document['failures'])
                resets = list((root / 'raw' / 'target').rglob('reset.json'))
                self.assertEqual(len(resets), 1 if reset_before == 'cell' else 3)
                self.assertFalse(list((root / 'raw' / 'zero_cost').rglob('reset.json')))
                verify_normalized(document)

    def test_explicit_raw_replay_rejects_bad_evidence(self):
        for mode in ('raw', 'workload', 'measurement', 'wire', 'shape', 'sampling', 'command', 'mapping',
                     'overlap', 'events', 'observer', 'unindexed',
                     'binding_endpoint', 'binding_listener', 'reset_missing', 'reset_false', 'decode_statistics'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root, campaign = make_bundle(Path(tmp), reset_before='repeat' if mode.startswith('reset_') else None)
                collect_fixture(root, campaign)
                index = _read_json(root / 'execution_index.json')
                role = 'zero_cost' if mode.startswith('binding_') else 'target'
                entry = next(e for k, e in index['cells'].items() if k.endswith(':' + role))
                directory = root / entry['path']
                request = next((directory / 'batches' / 'measured-0' / 'requests').iterdir())
                if mode == 'raw':
                    (request / 'response-body.bin').write_bytes(b'corrupt')
                elif mode == 'workload':
                    (directory / 'workload_evidence.json').write_bytes(b'{}')
                elif mode == 'measurement':
                    envelope = _read_json(directory / 'measurements.json')
                    envelope['measurements'].pop()
                    _write_atomic(directory / 'measurements.json', envelope)
                elif mode == 'decode_statistics':
                    statistics = _read_json(directory / 'decode_statistics.json')
                    statistics['summary']['mean'] += 1
                    _write_atomic(directory / 'decode_statistics.json', statistics)
                else:
                    if mode in ('wire', 'shape', 'sampling'):
                        body = _read_json(request / 'request.json')
                        if mode == 'shape': body['prompt'].append(1)
                        elif mode == 'sampling': body['temperature'] = 0.5
                        else: body['prompt'][0] = 3
                        _write_atomic(request / 'request.json', body)
                        if mode != 'wire':
                            (request / 'request-body.bin').write_bytes(json.dumps(body).encode())
                    elif mode == 'command':
                        p = request.parent.parent / 'command.json'
                        command = _read_json(p)
                        command['argv'][command['argv'].index('--seed') + 1] = '0'
                        _write_atomic(p, command)
                    elif mode in ('mapping', 'overlap'):
                        terminal = _read_json(request / 'terminal.json')
                        terminal['transport_request_id' if mode == 'mapping' else 'upstream_dispatch_ns'] = 'wrong' if mode == 'mapping' else 1
                        _write_atomic(request / 'terminal.json', terminal)
                    elif mode == 'events':
                        events = (request / 'events.jsonl').read_text().replace('"text": "x"', '"text": "changed"')
                        (request / 'events.jsonl').write_text(events)
                    elif mode == 'observer':
                        observer = _read_json(directory / 'observer.json')
                        observer['config']['writer_open_files'] += 1
                        _write_atomic(directory / 'observer.json', observer)
                    elif mode.startswith('binding_'):
                        binding = entry['binding']
                        if mode == 'binding_endpoint':
                            binding['endpoint']['address'] = '127.0.0.2'
                        else:
                            binding['listeners'][0]['port'] += 1
                        _write_atomic(directory / 'binding.json', binding)
                        _write_atomic(root / 'execution_index.json', index)
                    elif mode.startswith('reset_'):
                        receipt = request.parent.parent / 'reset.json'
                        if mode == 'reset_missing':
                            receipt.unlink()
                        else:
                            _write_atomic(receipt, {'status': 200, 'body': 'false'})
                    refresh_attempt(root, campaign, role)
                    if mode == 'unindexed':
                        raw_path = directory / 'raw_bundle_index.json'
                        raw = _read_json(raw_path)
                        raw['files'].remove((request / 'request-body.bin').relative_to(directory).as_posix())
                        _write_atomic(raw_path, raw)
                from artifacts import sealed_reference
                from pairing import load_role
                index = _read_json(root / 'execution_index.json')
                entry = next(e for k, e in index['cells'].items() if k.endswith(':' + role))
                cell = next(c for c in campaign['workload']['cells'] if c['split'] == 'train')
                plan = build_plan(cell, campaign['client'])
                with self.assertRaises((ArtifactError, ValueError, KeyError)):
                    load_role(root, campaign, plan, role, sealed_reference(entry), replay_raw=True)
