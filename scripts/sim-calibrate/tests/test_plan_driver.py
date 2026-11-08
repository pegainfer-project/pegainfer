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
from vllm_bench import inspect_binary
from test_campaign import make_input
from test_evidence_proxy import DiagnosticEndpoint

BENCH = Path(__file__).resolve().parents[3]/'.venv/bin/vllm-bench'


def real_input(root: Path, endpoint: str, repeats=2, warmups=1):
    path,spec=make_input(root)
    spec['client'].update(binary=str(BENCH),version=inspect_binary(BENCH).version)
    spec['variants']['target']['identity']['endpoint']=endpoint
    spec['variants']['target']['runtime']['cache']['policy']='off'
    spec['workload']['matrices']={'decode':{
        'kind':'steady_decode','split':'train','concurrency':[2],'num_ctx':[16],
        'output_tokens':8,
        'warmup_repeats':warmups,'repeats':repeats,'arrival':{'mode':'fixed_batch'}}}
    path.write_text(json.dumps(spec))
    return compile_campaign(path),root/'artifacts'


@unittest.skipUnless(BENCH.is_file(),'installed original bench unavailable')
class DriverTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_ids_partial_usage_mismatch_and_recovery(self):
        for mode,expected in [('burst','failed'),('no_ids','unsupported'),('drop','failed'),('bad_usage','failed'),('no_usage','sealed')]:
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
