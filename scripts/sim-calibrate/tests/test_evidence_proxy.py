"""HTTP fault injection for the observer; diagnostic endpoint, not model performance evidence."""
import asyncio
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
import aiohttp
from aiohttp import web
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from campaign import compile_campaign
from evidence_proxy import EvidenceProxy
from cached_prefill import CachedPrefillError, validate_cached_measurement
from manifest import sha256_json
from request_plan import build_plan
from test_campaign import make_input
from test_sse_evidence import event
from vllm_bench_collect import convert_evidence


class DiagnosticEndpoint:
    def __init__(self, mode='normal'):
        self.mode = mode
        self.received = []
        self.completed = []
        self.models = 0

    async def __aenter__(self):
        app = web.Application()
        app.router.add_get('/v1/models', self.get_models)
        app.router.add_post('/v1/completions', self.complete)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        self.url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
        return self

    async def __aexit__(self, *_):
        await self.runner.cleanup()

    async def get_models(self, request):
        self.models += 1
        return web.json_response({'data':[{'id':'test-model'}]})

    async def complete(self, request):
        body = await request.json()
        rid = request.headers.get('x-request-id')
        self.received.append((rid, body))
        if self.mode == 'http_error':
            return web.json_response({'error':'injected'}, status=503)
        response = web.StreamResponse(headers={'Content-Type':'text/event-stream'})
        await response.prepare(request)
        try:
            if self.mode == 'hang':
                await asyncio.sleep(2)
            else:
                await asyncio.sleep(0.01 if rid and rid.endswith('1') else 0.03)
            await response.write(event({'index':0, 'text':'', 'token_ids':[]}, prompt_token_ids=body['prompt']))
            count = body['max_tokens']
            first = count if self.mode == 'burst' else min(2, count)
            choice = {'index':0, 'text':'' if self.mode != 'no_ids' else 'a', 'finish_reason':None}
            if self.mode != 'no_ids':
                choice['token_ids'] = list(range(first))
            if first == count:
                choice['finish_reason'] = 'length'
            await response.write(event(choice))
            for position in range(first, count):
                await asyncio.sleep(0.01)
                choice = {'index':0, 'text':' ', 'finish_reason':'length' if position == count-1 else None}
                if self.mode != 'no_ids':
                    choice['token_ids'] = [position]
                await response.write(event(choice))
            if self.mode == 'drop':
                request.transport.close()
                return response
            await asyncio.sleep(0.02)
            if self.mode != 'no_usage':
                usage = {'prompt_tokens':len(body['prompt']), 'completion_tokens':count,
                         'total_tokens':len(body['prompt'])+count}
                if self.mode == 'bad_usage':
                    usage['completion_tokens'] += 1
                    usage['total_tokens'] += 1
                await response.write(event(usage=usage))
            await response.write(b'data: [DONE]\n\n')
            await response.write_eof()
            self.completed.append(rid)
        except (ConnectionResetError, aiohttp.ClientConnectionError):
            pass
        return response


def compiled_plan(root: Path):
    path, spec = make_input(root)
    spec['workload']['matrices'] = {'decode':{
        'kind':'steady_decode','split':'train','concurrency':[2],'num_ctx':[16],
        'output_tokens':8,
        'warmup_repeats':0,'repeats':1,'arrival':{'mode':'fixed_batch'}}}
    path.write_text(json.dumps(spec))
    campaign = compile_campaign(path, write=False)
    cell = campaign['workload']['cells'][0]
    return campaign, build_plan(cell, campaign['client'])


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_identity_never_reaches_upstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            _, plan=compiled_plan(root)
            async with DiagnosticEndpoint() as upstream:
                async with EvidenceProxy(upstream.url,'test-model',plan,root/'raw',timeout=2) as proxy:
                    proxy.register(plan['batches'][0],'p-',root/'raw'/'batches'/'measured-0')
                    async with aiohttp.ClientSession() as session:
                        async with session.post(proxy.url+'/v1/completions',json={},headers={'x-request-id':'unknown'}) as r:
                            self.assertEqual(r.status,409)
                    self.assertEqual(upstream.received,[])

    async def test_ambiguous_header_and_invalid_shape_are_not_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            _,plan=compiled_plan(root)
            async with DiagnosticEndpoint() as upstream:
                async with EvidenceProxy(upstream.url,'test-model',plan,root/'raw',timeout=2) as proxy:
                    proxy.register(plan['batches'][0],'p-',root/'raw'/'batch')
                    async with aiohttp.ClientSession() as session:
                        for headers in ([('x-request-id','p-0'),('x-request-id','p-1')],[('x-request-id','p-0')]):
                            async with session.post(proxy.url+'/v1/completions',json={},headers=headers) as r:
                                self.assertEqual(r.status,409)
                    self.assertEqual(upstream.received,[])


class CachedDiagnosticEndpoint(DiagnosticEndpoint):
    """Fault injection for request-scoped cache facts, not performance evidence."""
    async def complete(self, request):
        body = await request.json()
        self.received.append((request.headers.get('x-request-id'), body))
        ids = [2, 3] if body['prompt'] == 'αβ' else [2, 3, 4, 5]
        observed = ids if self.mode not in ('wrong_ids', 'dual_wrong_ids') else ids[:-1] + [99]
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        prompt_choice = {'index': 0, 'token_ids': [], 'text': ''}
        if self.mode != 'no_prompt_ids':
            prompt_choice['prompt_token_ids'] = observed
        extra = {'prompt_token_ids': ids} if self.mode in ('dual_prompt_ids', 'dual_wrong_ids') else {}
        await response.write(event(prompt_choice, **extra))
        await asyncio.sleep(0.001)
        await response.write(event({'index': 0, 'token_ids': [7], 'text': '', 'finish_reason': 'length'}))
        if self.mode != 'no_usage':
            usage = {'prompt_tokens': len(ids), 'completion_tokens': 1, 'total_tokens': len(ids) + 1}
            if len(ids) == 4:
                usage['prompt_tokens_details'] = {'cached_tokens': {'short': 1, 'extra': 3, 'zero': 0}.get(self.mode, 2)}
            await response.write(event(usage=usage))
        await response.write(b'data: [DONE]\n\n')
        await response.write_eof()
        return response


class CachedProxyTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, root, mode='normal', variant='target', fault=None):
        cell = 'a' * 64
        rows = [{'pairing_key': f'{cell}:{phase}:0:0', 'phase': phase, 'repeat': 0, 'index': 0,
                 'input_tokens': len(ids), 'output_tokens': 1, 'dispatch_offset_ms': 0,
                 'prompt': prompt, 'token_ids': ids}
                for phase, prompt, ids in [('bootstrap', 'αβ', [2, 3]), ('measured', 'αβδε', [2, 3, 4, 5])]]
        plan = {'workload_kind': 'cached_prefill', 'cell_id': cell, 'plan_sha256': 'b' * 64,
                'concurrency': 1, 'prefix_tokens': 2, 'new_tokens': 2,
                'shape': {'input_tokens': 4, 'output_tokens': 1}, 'sampling': {'ignore_eos': True},
                'logical_requests': rows}
        async with CachedDiagnosticEndpoint(mode) as upstream:
            async with EvidenceProxy(upstream.url, 'test-model', plan, root, timeout=2) as proxy:
                async with aiohttp.ClientSession() as session:
                    for logical in rows:
                        batch = {'phase': logical['phase'], 'repeat': 0, 'request_keys': [logical['pairing_key']]}
                        proxy.register(batch, logical['phase'] + '-', root / 'batches' / f"{logical['phase']}-0")
                        body = {'model': 'test-model', 'prompt': logical['prompt'], 'max_tokens': 1,
                                'stream': True, 'return_token_ids': True, 'echo': False,
                                'stream_options': {'include_usage': True}, **plan['sampling']}
                        if fault == 'prompt' and logical['phase'] == 'measured':
                            body['prompt'] += 'x'
                        if fault == 'bootstrap_shape' and logical['phase'] == 'bootstrap':
                            body['max_tokens'] = 2
                        wire = json.dumps(body, ensure_ascii=False).encode()
                        async with session.post(proxy.url + '/v1/completions', data=wire,
                                                headers={'x-request-id': logical['phase'] + '-0',
                                                         'content-type': 'application/json'}) as response:
                            await response.read()
                        await proxy.drain()
                errors = list(proxy.errors)
            received = list(upstream.received)
        identity = {'endpoint': 'http://localhost:1', 'model': 'test-model', 'engine': 'diagnostic',
                    'engine_version': '1', 'model_revision': 'test', 'model_config_sha256': 'c' * 64}
        client = {'version': '1', 'binary_sha256': 'd' * 64, 'tokenizer_sha256': 'e' * 64,
                  'observer': {'sha256': 'f' * 64}}
        envelope = convert_evidence(plan, variant, identity, root, raw_index_sha256='0' * 64,
                                    client=client, batch_errors=errors)
        return plan, envelope['measurements'], received

    async def test_cached_text_is_forwarded_and_phase_evidence_is_qualified(self):
        for role, mode in [('target', 'normal'), ('target', 'dual_prompt_ids'), ('zero_cost', 'zero')]:
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan, rows, received = await self.collect(root, mode, role)
                self.assertEqual([row['status'] for row in rows], ['succeeded', 'succeeded'])
                self.assertEqual([body['prompt'] for _, body in received], ['αβ', 'αβδε'])
                self.assertEqual([row['prompt_tokens'] for row in rows], [2, 4])
                self.assertIsNone(rows[1]['tpot_ms'])
                self.assertEqual(rows[1]['token_sha256'], sha256_json([2, 3, 4, 5]))
                validate_cached_measurement(plan, rows[0])  # Bootstrap has no cached-token assertion.
                validate_cached_measurement(plan, rows[1])
                for field, value in [('unit', 'blocks'), ('pairing_key', rows[0]['pairing_key'])]:
                    altered = copy.deepcopy(rows[1])
                    altered['extensions']['cached_prefill'][field] = value
                    with self.assertRaises(CachedPrefillError):
                        validate_cached_measurement(plan, altered)
                for request in root.glob('batches/*/requests/*'):
                    self.assertEqual(json.loads((request / 'request-body.bin').read_bytes()),
                                     json.loads((request / 'request.json').read_text()))

    async def test_cache_faults_never_qualify_and_custom_mismatch_is_not_forwarded(self):
        for fault in ['short', 'extra', 'wrong_ids', 'dual_wrong_ids', 'no_prompt_ids', 'no_usage']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                _, rows, _ = await self.collect(Path(tmp), fault)
                self.assertEqual(rows[1]['status'], 'failed')
        for fault in ['prompt', 'bootstrap_shape']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                _, rows, received = await self.collect(Path(tmp), fault=fault)
                self.assertTrue(all(row['status'] == 'failed' for row in rows))
                self.assertEqual(len(received), 1)
