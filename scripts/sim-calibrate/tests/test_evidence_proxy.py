"""HTTP observer request validation and diagnostic endpoint fixtures."""
import asyncio
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
from request_plan import build_plan
from test_campaign import make_input


def event(choice=None, **extra):
    return ('data: ' + json.dumps({'choices': [] if choice is None else [choice], **extra}, ensure_ascii=False) + '\r\n\r\n').encode()


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
        'warmup_repeats':0,'repeats':1,'arrival':{'mode':'fixed_batch'},
        'cache_lifecycle':{'namespace':'decode','reset_before':'repeat','bootstrap_repeats':0}}}
    path.write_text(json.dumps(spec))
    campaign = compile_campaign(path, write=False)
    cell = campaign['workload']['cells'][0]
    return campaign, build_plan(cell, campaign['client'])


class ProxyTests(unittest.IsolatedAsyncioTestCase):
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
