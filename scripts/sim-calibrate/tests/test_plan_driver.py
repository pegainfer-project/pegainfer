"""Original installed bench integration against controlled HTTP fault injection."""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from campaign import compile_campaign
from plan_driver import run_campaign
from artifacts import ArtifactError
from vllm_bench import inspect_binary
from test_campaign import make_input
from test_evidence_proxy import DiagnosticEndpoint

BENCH = Path(__file__).resolve().parents[3]/'.venv/bin/vllm-bench'
TOKENIZER = Path('/home/mgj/qwen3weights/Qwen3-4B')


def real_input(root: Path, endpoint: str, repeats=2, warmups=1):
    path,spec=make_input(root)
    spec['client'].update(binary=str(BENCH),version=inspect_binary(BENCH).version,tokenizer=str(TOKENIZER))
    spec['variants']['target']['identity']['endpoint']=endpoint
    spec['variants']['target']['runtime']['cache']['policy']='off'
    spec['workload']['matrices']={'decode':{
        'kind':'steady_decode','split':'train','concurrency':[2],'context_tokens':[8],
        'surface_requests':[],'surface_kv_tokens':[],'output_tokens':4,
        'warmup_repeats':warmups,'repeats':repeats,'arrival':{'mode':'fixed_batch'}}}
    path.write_text(json.dumps(spec))
    return compile_campaign(path),root/'artifacts'


@unittest.skipUnless(BENCH.is_file() and TOKENIZER.is_dir(),'installed original bench/tokenizer unavailable')
class DriverTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_batches_reordered_completion_and_sealed_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            async with DiagnosticEndpoint() as upstream:
                compiled,root=real_input(Path(tmp),upstream.url)
                before=(root/'artifact_index.json').read_bytes()
                result=await run_campaign(root,variants=['target'],timeout=10)
                self.assertEqual(result[0]['status'],'sealed',result)
                self.assertEqual(len(upstream.received),6)
                self.assertEqual(upstream.models,1)  # Readiness generated no extra completion.
                envelope=json.loads((root/result[0]['path']/'measurements.json').read_text())
                rows=envelope['measurements']
                self.assertEqual([r['phase'] for r in rows],['warmup']*2+['measured']*4)
                for row in rows:
                    self.assertEqual(row['first_event_token_count'],2)
                    self.assertEqual(row['interval_token_counts'],[2])
                    self.assertGreater(row['first_text_latency_ms'],row['ttft_ms'])
                    self.assertGreater(row['e2e_ms'],(row['last_token_timestamp']-row['actual_dispatch_timestamp'])*1000+10)
                skipped=await run_campaign(root,variants=['target'],timeout=10)
                self.assertTrue(skipped[0]['skipped'])
                self.assertEqual(len(upstream.received),6)
                self.assertEqual((root/'artifact_index.json').read_bytes(),before)
                raw=next((root/result[0]['path']).rglob('response-body.bin'))
                raw.write_bytes(raw.read_bytes()+b'corrupt')
                with self.assertRaisesRegex(ArtifactError,'changed'):
                    await run_campaign(root,variants=['target'],timeout=10)

    async def test_missing_ids_partial_usage_mismatch_and_recovery(self):
        for mode,expected in [('no_ids','unsupported'),('drop','failed'),('bad_usage','failed'),('no_usage','sealed')]:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                async with DiagnosticEndpoint(mode) as upstream:
                    _,root=real_input(Path(tmp),upstream.url,repeats=1,warmups=0)
                    result=await run_campaign(root,variants=['target'],timeout=5)
                    self.assertEqual(result[0]['status'],expected,result)
                    if mode=='no_ids':
                        rows=json.loads((root/result[0]['path']/'measurements.json').read_text())['measurements']
                        self.assertTrue(all(r['ttft_ms'] is None and r['first_text_latency_ms'] is not None for r in rows))
                    if expected!='sealed':
                        upstream.mode='normal'
                        recovered=await run_campaign(root,variants=['target'],timeout=5)
                        self.assertEqual(recovered[0]['status'],'sealed',recovered)
                        self.assertEqual(len(recovered[0]['attempts']),2)
                        self.assertTrue((root/result[0]['path']).is_dir())

    async def test_timeout_keeps_logs_and_next_attempt_can_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            async with DiagnosticEndpoint('hang') as upstream:
                _,root=real_input(Path(tmp),upstream.url,repeats=1,warmups=0)
                result=await run_campaign(root,variants=['target'],timeout=0.2)
                self.assertEqual(result[0]['status'],'failed',result)
                self.assertTrue(list((root/result[0]['path']).rglob('stderr.log')))
                upstream.mode='normal'
                result=await run_campaign(root,variants=['target'],timeout=5)
                self.assertEqual(result[0]['status'],'sealed',result)

    async def test_original_bench_retry_cannot_forward_the_identity_twice(self):
        from unittest.mock import patch
        from aiohttp import web
        from evidence_proxy import EvidenceProxy
        from vllm_bench import render_command, run_command
        from request_plan import build_plan
        with tempfile.TemporaryDirectory() as tmp:
            async with DiagnosticEndpoint() as upstream:
                campaign,root=real_input(Path(tmp),upstream.url,repeats=1,warmups=0)
                cell=campaign['workload']['cells'][0]
                plan=build_plan(cell,campaign['client'])
                raw=root/'retry-injection'
                async with EvidenceProxy(upstream.url,'test-model',plan,raw,timeout=5) as proxy:
                    proxy.register(plan['batches'][0],'retry-',raw/'batch')
                    original=web.StreamResponse.prepare
                    dropped=False
                    async def lose_headers(response,request):
                        nonlocal dropped
                        if request.app is proxy.runner.app and response.status==200 and not dropped:
                            dropped=True
                            import socket, struct
                            sock = request.transport.get_extra_info('socket')
                            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
                            request.transport.abort()
                        return await original(response,request)
                    argv=render_command(plan,plan['batches'][0],binary=BENCH,endpoint=proxy.url,
                                        model='test-model',result_dir=raw,request_id_prefix='retry-',tokenizer=str(TOKENIZER))
                    with patch.object(web.StreamResponse,'prepare',lose_headers):
                        await run_command(argv,root=raw,timeout=10,launcher=campaign['client']['process_launcher'])
                        await proxy.drain()
                    self.assertTrue(dropped)
                    self.assertTrue(any('duplicate' in e for e in proxy.errors),proxy.errors)
                    self.assertEqual(len(upstream.received),2)
                    self.assertEqual(len({rid for rid,body in upstream.received}),2)

    async def test_killed_driver_does_not_leave_bench_and_recovers_to_new_attempt(self):
        import os, signal, time
        with tempfile.TemporaryDirectory() as tmp:
            async with DiagnosticEndpoint('hang') as upstream:
                _,root=real_input(Path(tmp),upstream.url,repeats=1,warmups=0)
                command=[sys.executable,str(Path(__file__).resolve().parents[1]/'campaign.py'),
                         'run','--artifact-root',str(root),'--variant','target','--timeout','10']
                with (root/'driver.stdout').open('wb') as stdout,(root/'driver.stderr').open('wb') as stderr:
                    process=await asyncio.create_subprocess_exec(*command,stdout=stdout,stderr=stderr)
                    child=None
                    try:
                        deadline=time.monotonic()+10
                        while len(upstream.received)<2:
                            self.assertIsNone(process.returncode)
                            self.assertLess(time.monotonic(),deadline)
                            await asyncio.sleep(0.02)
                        children=Path(f'/proc/{process.pid}/task/{process.pid}/children').read_text().split()
                        self.assertEqual(len(children),1)
                        child=int(children[0])
                        process.kill()
                        await process.wait()
                        deadline=time.monotonic()+3
                        while Path(f'/proc/{child}/stat').exists():
                            state=Path(f'/proc/{child}/stat').read_text().rsplit(')',1)[1].split()[0]
                            if state=='Z':
                                break
                            self.assertLess(time.monotonic(),deadline,'bench still running after driver death')
                            await asyncio.sleep(0.02)
                        upstream.mode='normal'
                        outcome=await run_campaign(root,variants=['target'],timeout=5)
                        self.assertEqual(outcome[0]['status'],'sealed',outcome)
                        self.assertEqual(outcome[0]['attempt'],2)
                        self.assertEqual(outcome[0]['attempts'][0]['status'],'interrupted')
                    finally:
                        if process.returncode is None:
                            process.kill()
                            await process.wait()
                        if child is not None:
                            try: os.killpg(child,signal.SIGKILL)
                            except ProcessLookupError: pass
