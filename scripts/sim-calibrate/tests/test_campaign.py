"""Canonical experiment identities and named matrix configuration tests."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))
from campaign import compile_campaign

FIXTURE = Path(__file__).parent / 'fixtures/campaign-compile.json'


def make_input(root: Path) -> tuple[Path, dict]:
    spec = json.loads(FIXTURE.read_text())
    model = root / 'model'
    model.mkdir()
    (model / 'tokenizer_config.json').write_text('{"tokenizer_class":"PreTrainedTokenizerFast"}\n')
    (model / 'tokenizer.json').write_text('{"version":"1.0","model":{"type":"WordLevel","vocab":{"x":0},"unk_token":"x"}}\n')
    (model / 'config.json').write_text('{"model_type":"qwen3","vocab_size":1}\n')
    config_digest = hashlib.sha256((model / 'config.json').read_bytes()).hexdigest()
    binary = root / 'bench'
    binary.write_text("#!/usr/bin/env python3\nimport sys\nprint('test' if '--version' in sys.argv else 'test help')\n")
    binary.chmod(0o755)
    spec['client']['binary'] = str(binary)
    spec['client']['tokenizer'] = str(model)
    for variant in spec['variants'].values():
        variant['identity']['model_config_sha256'] = config_digest
        variant['identity']['model_revision'] = config_digest
        # The compiler records these flags; it does not start a server.
        variant['server_flags'] = []
    spec['output']['artifact_root'] = str(root / 'artifacts')
    path = root / 'spec.json'
    path.write_text(json.dumps(spec))
    return path, spec


class CampaignCompileTests(unittest.TestCase):

    def test_output_relocation_and_key_order_do_not_change_workload_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, spec = make_input(root)
            first = compile_campaign(path, write=False)
            spec['output']['artifact_root'] = str(root / 'elsewhere')
            spec['workload']['matrices'] = dict(reversed(list(spec['workload']['matrices'].items())))
            spec['client']['sampling'].update(temperature=0.0, top_p=1.0)
            relocated_model = root / 'relocated-model'
            relocated_model.mkdir()
            for original in (root / 'model').iterdir():
                (relocated_model / original.name).write_bytes(original.read_bytes())
            spec['client']['tokenizer'] = str(relocated_model)
            path.write_text(json.dumps(spec, sort_keys=True))
            second = compile_campaign(path, write=False)
            self.assertEqual(first['campaign_id'], second['campaign_id'])
            self.assertEqual([c['plan']['plan_sha256'] for c in first['workload']['cells']], [c['plan']['plan_sha256'] for c in second['workload']['cells']])
            spec['client']['seed'] += 1
            path.write_text(json.dumps(spec))
            self.assertNotEqual(first['campaign_id'], compile_campaign(path, write=False)['campaign_id'])

    def test_matrix_overrides_are_resolved_before_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            path, spec = make_input(Path(directory))
            matrix = spec['workload']['matrices']['decode-heldout']
            matrix['sampling'] = {'temperature': 0.5}
            matrix['seed'] = 9
            path.write_text(json.dumps(spec))
            compiled = compile_campaign(path, write=False)
            cell = next(c for c in compiled['workload']['cells'] if c['matrix_name'] == 'decode-heldout')
            self.assertEqual(cell['sampling'], {'temperature': 0.5, 'top_p': 1, 'ignore_eos': True})
            self.assertEqual(cell['seed'], 9)
            self.assertFalse(cell['fit_policy']['train'])
            self.assertEqual(cell['fit_policy']['excluded_metrics'], ['ttft_ms'])

    def test_complete_surface_grid_preserves_total_kv_and_fit_families(self):
        with tempfile.TemporaryDirectory() as directory:
            path, spec = make_input(Path(directory))
            spec['workload']['matrices'] = {
                'decode-grid': {
                    'kind': 'steady_decode', 'split': 'train',
                    'concurrency': {'start': 1, 'stop': 4, 'step': 1},
                    'context_tokens': {'values': [128]},
                    'surface_requests': {'values': [1, 2, 4]},
                    'surface_kv_tokens': {'values': [256, 512]},
                    'output_tokens': 8, 'warmup_repeats': 1, 'repeats': 2,
                    'arrival': {'mode': 'fixed_batch'},
                },
            }
            path.write_text(json.dumps(spec))
            compiled = compile_campaign(path)
            cells = compiled['workload']['cells']
            surface = [c for c in cells if c['metadata']['family'] == 'surface']
            self.assertEqual(len(cells), 10)
            self.assertEqual({(c['concurrency'], c['sum_decode_ctx_tokens']) for c in surface},
                             {(count, kv) for count in (1, 2, 4) for kv in (256, 512)})
            for cell in cells:
                self.assertEqual(cell['sum_decode_ctx_tokens'], cell['concurrency'] * cell['input_tokens'])
                self.assertEqual(cell['metadata']['sum_decode_ctx_tokens'], cell['sum_decode_ctx_tokens'])
                plan = json.loads((path.parent / 'artifacts' / cell['plan']['path']).read_text())
                self.assertEqual(plan['shape']['input_tokens'], cell['input_tokens'])
            overlapping = [c for c in cells if c['concurrency'] == 4 and c['input_tokens'] == 128]
            self.assertEqual({c['metadata']['family'] for c in overlapping}, {'additive', 'surface'})
            self.assertEqual(len({c['cell_id'] for c in overlapping}), 2)

    def test_equivalent_axes_preserve_identity_and_fixed_batch_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path, spec = make_input(Path(directory))
            spec['workload']['matrices'] = {
                'decode-axis': {
                    'kind': 'steady_decode', 'split': 'train',
                    'concurrency': {'values': [2, 1]}, 'context_tokens': {'values': [128]},
                    'surface_requests': [], 'surface_kv_tokens': [],
                    'output_tokens': 8, 'warmup_repeats': 1, 'repeats': 2,
                    'arrival': {'mode': 'fixed_batch'},
                },
            }
            path.write_text(json.dumps(spec))
            first = compile_campaign(path, write=False)
            spec['workload']['matrices']['decode-axis']['concurrency'] = {'start': 1, 'stop': 2, 'step': 1}
            path.write_text(json.dumps(spec))
            second = compile_campaign(path, write=False)
            self.assertEqual(first['campaign_id'], second['campaign_id'])
            self.assertEqual(first['workload']['cells'], second['workload']['cells'])


if __name__ == '__main__':
    unittest.main()
