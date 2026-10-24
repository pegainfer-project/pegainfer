"""Exercise rerun checkpoint recovery through the public CLI and loopback HTTP."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'eval_rerun_truncated.py'
BENCHMARK = 'mmlu_redux'


class CompletionHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(payload)
        if payload['messages'][0]['content'] == self.server.held_prompt:
            self.server.release.wait(60)
        body = json.dumps({
            'choices': [{'message': {'content': self.server.answer},
                         'finish_reason': 'stop'}],
            'usage': {'completion_tokens': 1},
        }).encode()
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # An interrupted CLI has already closed its HTTP connection.

    def log_message(self, *_args):
        pass


class RerunRecoveryCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='pegainfer-eval-resume-')
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.server = self.start_server()
        self.env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                        PYTHONDONTWRITEBYTECODE='1')

    def start_server(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), CompletionHandler)
        server.requests = []
        server.answer = 'A'
        server.held_prompt = None
        server.release = threading.Event()
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.release.set)
        return server

    def path(self, suffix):
        return self.out / f'{BENCHMARK}_{suffix}'

    def read(self, suffix):
        return json.loads(self.path(suffix).read_text())

    def write(self, suffix, value):
        self.path(suffix).write_text(json.dumps(value))

    def source(self, rows=None):
        if rows is None:
            rows = [self.row(0, 'Question zero', 'A')]
        self.write('samples.json', rows)
        self.write('summary.json', {'benchmark': BENCHMARK, 'model': 'local-model',
                                   'temperature': 0.0, 'n': len(rows), 'acc': 0,
                                   'api_errors': 0, 'max_tokens': 8})
        return rows

    @staticmethod
    def row(idx, prompt, gold):
        return {'idx': idx, 'prompt': prompt, 'gold': gold, 'output': '',
                'reasoning': 'unfinished thinking', 'pred': '', 'correct': False,
                'finish_reason': 'length', 'completion_tokens': 8,
                'subject': 'current subject'}

    def command(self, *, server=None, max_tokens=16):
        server = server or self.server
        return [sys.executable, str(SCRIPT), BENCHMARK, '--out-dir', str(self.out),
                '--base-url', f'http://127.0.0.1:{server.server_port}/v1',
                '--max-tokens', str(max_tokens), '--concurrency', '2', '--timeout', '5']

    def cli(self, **kwargs):
        result = subprocess.run(self.command(**kwargs), env=self.env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_interrupted_same_run_recovers_without_duplicate_completed_request(self):
        rows = self.source([self.row(0, 'Question zero', 'A'),
                            self.row(1, 'Question held', 'A')])
        self.server.held_prompt = rows[1]['prompt']
        with tempfile.TemporaryFile(mode='w+') as output:
            process = subprocess.Popen(self.command(), env=self.env, stdout=output,
                                       stderr=subprocess.STDOUT, text=True)
            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    checkpoint = self.path('rerun.partial.jsonl')
                    if (checkpoint.exists() and checkpoint.read_text().count('\n') == 1
                            and len(self.server.requests) == 2):
                        break
                    if process.poll() is not None:
                        output.seek(0)
                        self.fail(f'CLI exited before interruption: {output.read()}')
                    time.sleep(.05)
                else:
                    self.fail('CLI did not persist the first completed response')
            finally:
                process.kill()
                process.wait(timeout=10)
        self.server.held_prompt = None
        self.server.release.set()
        self.assertFalse(self.path('samples_merged.json').exists())
        result = self.cli()
        prompts = [r['messages'][0]['content'] for r in self.server.requests]
        self.assertEqual(prompts.count('Question zero'), 1)
        self.assertEqual(prompts.count('Question held'), 2)
        self.assertIn('recovered 1 completed rows', result.stdout)
        self.assertEqual(self.read('summary.json')['acc_merged'], 100.0)

    def test_changed_source_does_not_reuse_partial_by_index(self):
        self.source()
        self.cli()
        self.path('samples_merged.json').unlink()
        current = self.row(0, 'Different question with the same index', 'B')
        self.write('samples.json', [current])
        self.cli()
        result = self.read('samples_merged.json')[0]
        self.assertEqual(result['prompt'], current['prompt'])
        self.assertEqual(result['gold'], 'B')
        self.assertFalse(result['correct'])
        self.assertEqual(len(self.server.requests), 2)
        self.assertEqual(self.read('summary.json')['acc_merged'], 0.0)

    def test_corrected_gold_and_source_metadata_survive_merged_recovery(self):
        self.source()
        self.cli()
        rows = self.read('samples.json')
        rows[0].update(gold='B', subject='corrected subject', options=['new option'])
        self.write('samples.json', rows)
        self.cli()
        result = self.read('samples_merged.json')[0]
        self.assertEqual(result['gold'], 'B')
        self.assertEqual(result['subject'], 'corrected subject')
        self.assertEqual(result['options'], ['new option'])
        self.assertFalse(result['correct'])
        self.assertEqual(self.read('summary.json')['acc_merged'], 0.0)

    def test_checkpoint_cannot_replace_current_scoring_metadata(self):
        self.source()
        self.cli()
        self.path('samples_merged.json').unlink()
        row = json.loads(self.path('rerun.partial.jsonl').read_text())
        row.update(gold='B', subject='obsolete subject', options=['obsolete option'])
        self.path('rerun.partial.jsonl').write_text(json.dumps(row) + '\n')
        self.cli()
        result = self.read('samples_merged.json')[0]
        self.assertEqual(result['gold'], 'A')
        self.assertEqual(result['subject'], 'current subject')
        self.assertNotIn('options', result)
        self.assertTrue(result['correct'])
        self.assertEqual(len(self.server.requests), 1)

    def test_changed_generation_settings_invalidate_completed_results(self):
        for setting in ('max_tokens', 'model', 'temperature', 'base_url'):
            with self.subTest(setting=setting):
                for path in self.out.iterdir():
                    path.unlink()
                self.server.requests.clear()
                self.server.answer = 'A'
                self.source()
                self.cli()
                kwargs = {}
                server = self.server
                if setting == 'max_tokens':
                    kwargs['max_tokens'] = 32
                elif setting == 'base_url':
                    server = self.start_server()
                    kwargs['server'] = server
                else:
                    summary = self.read('summary.json')
                    summary[setting] = 'new-model' if setting == 'model' else 0.7
                    self.write('summary.json', summary)
                server.answer = 'B'
                self.cli(**kwargs)
                self.assertEqual(self.read('samples_merged.json')[0]['output'], 'B')
                self.assertEqual(len(server.requests), 1 if setting == 'base_url' else 2)
                payload = server.requests[-1]
                self.assertEqual(payload['max_tokens'], kwargs.get('max_tokens', 16))
                if setting in ('model', 'temperature'):
                    self.assertEqual(payload[setting], summary[setting])
                self.assertEqual(self.read('summary.json')['acc_merged'], 0.0)

    def test_legacy_partial_without_provenance_is_not_reused(self):
        self.source()
        self.cli()
        self.path('samples_merged.json').unlink()
        self.path('rerun_config.json').unlink(missing_ok=True)
        self.server.answer = 'B'
        self.cli()
        self.assertEqual(self.read('samples_merged.json')[0]['output'], 'B')
        self.assertEqual(len(self.server.requests), 2)

    def test_unchanged_completed_run_reuses_results_without_http_calls(self):
        self.source()
        self.cli()
        provenance_mtime = self.path('rerun_config.json').stat().st_mtime_ns
        self.cli()
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(self.read('samples_merged.json')[0]['output'], 'A')
        self.assertEqual(self.path('rerun_config.json').stat().st_mtime_ns,
                         provenance_mtime)

    def test_complete_or_empty_replacement_source_rescores_stale_summary(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                self.source()
                self.cli()
                self.assertEqual(self.read('summary.json')['acc_merged'], 100.0)
                rows = self.read('samples.json')
                rows[0].update(output='B', reasoning='', completion_tokens=1,
                               finish_reason='stop')
                rows = [] if empty else rows
                self.write('samples.json', rows)
                request_count = len(self.server.requests)
                self.cli()
                self.assertEqual(len(self.server.requests), request_count)
                self.assertEqual(self.read('summary.json')['acc_merged'], 0.0)
                self.assertEqual(self.read('summary.json')['rerun_n'], 0)
                self.assertEqual(len(self.read('samples_merged.json')), len(rows))
                if rows:
                    self.assertEqual(self.read('samples_merged.json')[0]['output'], 'B')


if __name__ == '__main__':
    unittest.main()
