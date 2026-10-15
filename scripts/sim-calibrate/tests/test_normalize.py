"""Current-format evidence fixtures and production-entry contract/fault checks.

These synthetic streams exercise lineage, not model timing or GPU performance.
"""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import ArtifactError, ExecutionStore, _read_json, _write_atomic, file_reference
from campaign import compile_campaign
from fit import FitError, fit_normalized
from normalize import NormalizeError, normalize_campaign, verify_normalized, verify_normalized_sources, write_document
from probe import requires_cache_reset
from request_plan import build_plan
from sse_evidence import SSEParser
from test_campaign import make_input
from test_sse_evidence import event
from vllm_bench import render_command
from vllm_bench_collect import convert_evidence


def make_bundle(base: Path, *, cached=False, reset_before=None):
    """Compile current inputs; evidence is deliberately deterministic and tiny."""
    path, spec = make_input(base)
    cached_matrix = spec['workload']['matrices']['cached-train']
    matrix = {'kind': 'steady_decode', 'split': 'train', 'concurrency': [2], 'context_tokens': [8],
              'surface_requests': [], 'surface_kv_tokens': [],
              'output_tokens': 4, 'warmup_repeats': 1, 'repeats': 2, 'arrival': {'mode': 'fixed_batch'}}
    if reset_before:
        matrix = {'kind': 'cold_prefill', 'split': 'train', 'concurrency': 1, 'input_tokens': [8],
                  'output_tokens': 4, 'warmup_repeats': 1, 'repeats': 2, 'arrival': {'mode': 'fixed_batch'},
                  'cache_lifecycle': {'namespace': 'cold', 'reset_before': reset_before, 'bootstrap_repeats': 0}}
    spec['workload']['matrices'] = {'decode': matrix, 'heldout': {**matrix, 'split': 'sealed_test'}}
    if cached:
        spec['workload']['matrices']['cached'] = cached_matrix
    path.write_text(json.dumps(spec))
    campaign = compile_campaign(path)
    root = base / 'artifacts'
    zero = root / campaign['variants']['zero_cost']['profile_path']
    sim = root / campaign['variants']['sim']['profile_path']
    sim.parent.mkdir(parents=True)
    sim.write_bytes(zero.read_bytes())
    (sim.parent / 'manifest.json').write_bytes((zero.parent / 'manifest.json').read_bytes())
    return root, campaign


def collect_fixture(root: Path, campaign: dict, roles=('target', 'zero_cost'), *, split='train'):
    """Build raw evidence through the real converter and ExecutionStore."""
    with ExecutionStore(root, campaign['campaign_id']) as store:
        for cell in campaign['workload']['cells']:
            if cell['split'] != split or cell['kind'] == 'cached_prefill':
                continue
            plan = build_plan(cell, campaign['client'])
            for role in roles:
                identity, client = campaign['variants'][role]['identity'], campaign['client']
                binding = {'variant_identity': identity, 'observer_sha256': client['observer']['sha256']}
                if role != 'target':
                    profile = root / campaign['variants'][role]['profile_path']
                    endpoint = urlsplit(identity['endpoint'])
                    binding.update(profile=file_reference(profile), manifest=file_reference(profile.parent / 'manifest.json'),
                                   endpoint={'address': endpoint.hostname, 'port': endpoint.port},
                                   listeners=[{'family': 'IPv4', 'address': '0.0.0.0', 'port': endpoint.port}])
                directory, number = store.begin(plan, role, binding)
                for name, value in [('binding', binding), ('observer', client['observer']), ('probe', {'model': identity['model']}),
                                    ('attempt', {'attempt': number, 'errors': [], 'cell_id': cell['cell_id'], 'variant': role})]:
                    _write_atomic(directory / f'{name}.json', value)
                logical = {row['pairing_key']: row for row in plan['logical_requests']}
                for b, batch in enumerate(plan['batches']):
                    relative = f"batches/{batch['phase']}-{batch['repeat']}"
                    batch_root = directory / relative
                    if requires_cache_reset(plan, role, campaign['variants'][role]['runtime'], batch):
                        _write_atomic(batch_root / 'reset.json', {'status': 200, 'body': '{"success": true}'})
                    prefix = f'fixture-{role}-{b}-'
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
                        body = {'model': identity['model'], 'prompt': [1] * plan['shape']['input_tokens'],
                                'max_tokens': 4, 'stream': True, 'return_token_ids': True, 'echo': False,
                                'stream_options': {'include_usage': True}, **plan['sampling']}
                        _write_atomic(request / 'request.json', body)
                        (request / 'request-body.bin').write_bytes(json.dumps(body).encode())
                        payload = (event({'index': 0, 'token_ids': [1, 2], 'text': ''}) +
                                   event({'index': 0, 'token_ids': [3, 4], 'text': 'x', 'finish_reason': 'length'}) +
                                   event(usage={'prompt_tokens': 8, 'completion_tokens': 4, 'total_tokens': 12}) +
                                   b'data: [DONE]\r\n\r\n')
                        start = 1_000_000_000 + b * 100_000_000 + i * 2_000_000
                        parser = SSEParser()
                        events = parser.feed(payload, start + 1_000_000) + parser.eof()
                        (request / 'response-body.bin').write_bytes(payload)
                        (request / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
                        (request / 'reads.jsonl').write_text(json.dumps({'timestamp_ns': start + 1_000_000,
                                                                      'byte_start': 0, 'byte_end': len(payload)}) + '\n')
                        terminal = {**logical[key], 'transport_request_id': transport, 'request_instance_id': instance,
                                    'directory': request_relative, 'upstream_dispatch_ns': start,
                                    'stream_end_ns': start + 2_000_000, 'downstream_end_ns': start + 3_000_000,
                                    'upstream_attempts': 1, 'http_status': 200, 'eof': True, 'done': True,
                                    'finished': True, 'error': None}
                        _write_atomic(request / 'terminal.json', terminal)
                digest = store.raw_index(directory)
                envelope = convert_evidence(plan, role, identity, directory, raw_index_sha256=digest,
                                            client=client, batch_errors=[])
                _write_atomic(directory / 'workload_evidence.json', envelope.pop('workload_evidence'))
                _write_atomic(directory / 'measurements.json', envelope)
                store.finish(plan, role, directory)


def refresh_attempt(root: Path, campaign: dict, role='target'):
    """Rehash local links for semantic fault injection, leaving compiler inputs fixed."""
    index = _read_json(root / 'execution_index.json')
    cell = next(c for c in campaign['workload']['cells'] if c['split'] == 'train' and c['kind'] != 'cached_prefill')
    entry = index['cells'][cell['cell_id'] + ':' + role]
    directory = root / entry['path']
    with ExecutionStore(root, campaign['campaign_id']) as store:
        store.raw_index(directory)
    # Match acquisition: workload is published after the raw index, not part of it.
    raw = _read_json(directory / 'raw_bundle_index.json')
    raw['files'].pop('workload_evidence.json', None)
    raw_digest = _write_atomic(directory / 'raw_bundle_index.json', raw)
    envelope = convert_evidence(build_plan(cell, campaign['client']), role, campaign['variants'][role]['identity'],
                                directory, raw_index_sha256=raw_digest, client=campaign['client'], batch_errors=[])
    _write_atomic(directory / 'workload_evidence.json', envelope.pop('workload_evidence'))
    _write_atomic(directory / 'measurements.json', envelope)
    for name, field in [('raw_bundle_index.json', 'raw_index'), ('measurements.json', 'measurement'),
                        ('workload_evidence.json', 'workload_evidence')]:
        entry[field] = file_reference(directory / name)
    _write_atomic(root / 'execution_index.json', index)


class NormalizeTests(unittest.TestCase):
    def test_current_train_scope_repeated_content_and_fitter_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, campaign = make_bundle(Path(tmp), cached=True)
            collect_fixture(root, campaign)
            document = normalize_campaign(root, scope='train')
            self.assertEqual(document['status'], 'succeeded', document['failures'])
            self.assertEqual(document['matched_count'], 4)
            self.assertEqual(len(document['deferred_cells']), 4)
            self.assertEqual(document['required_roles'], ['target', 'zero_cost'])
            self.assertFalse(document['fit_ready'])
            for observation in document['observations']:
                self.assertEqual(observation['split'], 'train')
                self.assertNotIn('effective', observation)
                self.assertEqual(observation['measurements']['zero_cost']['tpot_ms'], 0)
            with self.assertRaises(FitError):
                fit_normalized(document)
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

    def test_validation_requires_complete_roles_and_empty_scope_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, campaign = make_bundle(Path(tmp))
            collect_fixture(root, campaign, split='sealed_test')
            self.assertEqual(normalize_campaign(root, scope='validation')['status'], 'invalid')
            collect_fixture(root, campaign, roles=('sim',), split='sealed_test')
            result = normalize_campaign(root, scope='validation')
            self.assertEqual(result['status'], 'succeeded', result['failures'])
            self.assertTrue(all(o['split'] == 'sealed_test' for o in result['observations']))
            verify_normalized_sources(result)
        with tempfile.TemporaryDirectory() as tmp:
            path, spec = make_input(Path(tmp))
            spec['workload']['matrices'] = {}
            # A nonempty campaign with no train cold/decode still cannot succeed vacuously.
            original = json.loads((Path(__file__).parent / 'fixtures/campaign-compile.json').read_text())
            spec['workload']['matrices']['cached'] = original['workload']['matrices']['cached-train']
            path.write_text(json.dumps(spec))
            campaign = compile_campaign(path)
            root = Path(tmp) / 'artifacts'
            with ExecutionStore(root, campaign['campaign_id']):
                _write_atomic(root / 'execution_index.json', {'kind': 'execution_index', 'schema_version': 1,
                              'campaign_id': campaign['campaign_id'], 'cells': {}})
            self.assertEqual(normalize_campaign(root, scope='train')['status'], 'invalid')

    def test_snapshot_survives_later_collection_and_offline_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, campaign = make_bundle(Path(tmp))
            collect_fixture(root, campaign)
            document = normalize_campaign(root, scope='train')
            output = Path(tmp) / 'normalized.json'
            write_document(output, document)
            before = output.read_bytes()
            collect_fixture(root, campaign, roles=('sim',), split='sealed_test')
            with patch('evidence_proxy.observer_identity', side_effect=AssertionError('no live source check')):
                verify_normalized(document)
            self.assertEqual(output.read_bytes(), before)
            for mode in ('metric', 'observations', 'cells', 'counts', 'normalizer_digest', 'fit_ready'):
                with self.subTest(mode=mode):
                    changed = copy.deepcopy(document)
                    if mode == 'metric':
                        changed['observations'][0]['measurements']['target']['ttft_ms'] += 1
                        changed['cells'][0]['observations'] = copy.deepcopy(changed['observations'])
                    elif mode in ('observations', 'cells'):
                        changed[mode] = []
                    elif mode == 'counts':
                        changed['expected_count'] = changed['matched_count'] = 999
                    elif mode == 'normalizer_digest':
                        changed['source']['normalizer_sha256'] = '0' * 64
                    else:
                        changed['fit_ready'] = True
                    with self.assertRaises(NormalizeError):
                        verify_normalized(changed)
                    with self.assertRaises(NormalizeError):
                        write_document(Path(tmp) / f'{mode}.json', changed)
                    self.assertFalse((Path(tmp) / f'{mode}.json').exists())
            damaged = root / document['source']['selected_attempts'][0]['path'] / 'binding.json'
            damaged.write_bytes(damaged.read_bytes() + b' ')
            with self.assertRaises(ArtifactError):
                verify_normalized_sources(document)

    def test_recursive_bytes_and_semantics_fail_closed(self):
        for mode in ('raw', 'workload', 'measurement', 'wire', 'shape', 'sampling', 'command', 'mapping',
                     'overlap', 'events', 'observer', 'unindexed', 'path', 'symlink',
                     'binding_endpoint', 'binding_listener', 'reset_missing', 'reset_false'):
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
                    entry['measurement'] = file_reference(directory / 'measurements.json')
                    _write_atomic(root / 'execution_index.json', index)
                else:
                    if mode == 'symlink':
                        (request / 'outside.json').symlink_to(root / 'compiled_campaign.json')
                    elif mode in ('wire', 'shape', 'sampling'):
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
                        observer['sha256'] = '0' * 64
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
                    if mode != 'symlink':
                        refresh_attempt(root, campaign, role)
                    if mode in ('unindexed', 'path'):
                        raw_path = directory / 'raw_bundle_index.json'
                        raw = _read_json(raw_path)
                        if mode == 'unindexed':
                            del raw['files'][(request / 'request-body.bin').relative_to(directory).as_posix()]
                        else:
                            raw['files']['../outside.json'] = {'sha256': '0' * 64, 'size': 0}
                        _write_atomic(raw_path, raw)
                        index = _read_json(root / 'execution_index.json')
                        entry = next(e for k, e in index['cells'].items() if k.endswith(':target'))
                        entry['raw_index'] = file_reference(raw_path)
                        envelope = _read_json(directory / 'measurements.json')
                        envelope['provenance']['raw_bundle_index_sha256'] = entry['raw_index']['sha256']
                        _write_atomic(directory / 'measurements.json', envelope)
                        entry['measurement'] = file_reference(directory / 'measurements.json')
                        _write_atomic(root / 'execution_index.json', index)
                result = normalize_campaign(root, scope='train')
                self.assertEqual(result['status'], 'invalid', mode)
                self.assertEqual(result['matched_count'], 0)
                self.assertTrue(result['failures'])

    def test_immutable_output_paths_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, campaign = make_bundle(Path(tmp))
            collect_fixture(root, campaign)
            document = normalize_campaign(root, scope='train')
            output = Path(tmp) / 'normalized.json'
            write_document(output, document)
            write_document(output, document)
            with self.assertRaises(ArtifactError):
                write_document(output, {**document, 'status': 'invalid'})
            for path in (root / 'execution_index.json', root / 'raw' / 'out.json', root / 'plans' / 'out.json'):
                with self.subTest(path=path), self.assertRaises(ArtifactError):
                    write_document(path, document)
            run = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'normalize.py'),
                                  '--artifact-root', str(root), '--scope', 'train', '--output', str(output)],
                                 capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            plan = root / campaign['workload']['cells'][0]['plan']['path']
            plan.write_bytes(plan.read_bytes() + b' ')
            with self.assertRaises(ArtifactError):
                normalize_campaign(root, scope='train')
