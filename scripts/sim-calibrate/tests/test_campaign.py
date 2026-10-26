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
    from tokenizers import Tokenizer, models, pre_tokenizers
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, **{chr(97 + i): i + 1 for i in range(26)}}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(model / 'tokenizer.json'))
    (model / 'config.json').write_text('{"model_type":"qwen3","vocab_size":27}\n')
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
            cell = next(c for c in compiled['workload']['cells'] if c['matrix_name'] == 'decode-heldout' and c['concurrency'] == 2)
            self.assertEqual(cell['sampling'], {'temperature': 0.5, 'top_p': 1, 'ignore_eos': True})
            self.assertEqual(cell['seed'], 9)
            self.assertFalse(cell['fit_policy']['train'])
            self.assertEqual(cell['fit_policy']['excluded_metrics'], ['ttft_ms'])

    def test_decode_grid_is_unique_and_preserves_initial_total_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = make_input(Path(directory))
            compiled = compile_campaign(path)
            cells = [c for c in compiled['workload']['cells'] if c['kind'] == 'steady_decode']
            train = [c for c in cells if c['split'] == 'train']
            heldout = [c for c in cells if c['split'] == 'sealed_test']
            self.assertEqual(len(train), 12)
            self.assertEqual(len(heldout), 11)
            self.assertEqual({(c['concurrency'], c['num_ctx']) for c in train},
                             {(b, ctx) for b in (1,2,4) for ctx in (256,1024,3072,6144)})
            self.assertEqual(len({(c['concurrency'], c['num_ctx']) for c in cells}), 23)
            for cell in cells:
                self.assertEqual(cell['num_ctx'], cell['concurrency'] * cell['input_tokens'])
                self.assertEqual(cell['num_ctx'], cell['sum_decode_ctx_tokens'])
                self.assertEqual((cell['warmup_repeats'], cell['repeats'], cell['output_tokens']), (1,2,8))
                plan = json.loads((path.parent / 'artifacts' / cell['plan']['path']).read_text())
                self.assertEqual(plan['num_ctx'], cell['num_ctx'])
                self.assertEqual(plan['shape']['input_tokens'], cell['input_tokens'])

    def test_invalid_decode_axes_and_retired_fields_are_rejected(self):
        from manifest import ManifestError
        for change in ({'num_ctx': [257]}, {'num_ctx': [32768]}, {'output_tokens': 4},
                       {'context_tokens': [128]}, {'surface_requests': []}, {'steady_state': {}}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path, spec = make_input(Path(directory))
                spec['workload']['matrices']['decode-train'].update(change)
                path.write_text(json.dumps(spec))
                with self.assertRaises(ManifestError):
                    compile_campaign(path, write=False)

    def test_output16_is_a_new_frozen_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path, spec = make_input(Path(directory))
            first = compile_campaign(path, write=False)
            spec['workload']['matrices']['decode-train']['output_tokens'] = 16
            path.write_text(json.dumps(spec))
            second = compile_campaign(path, write=False)
            self.assertNotEqual(first['campaign_id'], second['campaign_id'])
            before = {c['plan']['plan_sha256'] for c in first['workload']['cells'] if c['split'] == 'train' and c['kind'] == 'steady_decode'}
            after = {c['plan']['plan_sha256'] for c in second['workload']['cells'] if c['split'] == 'train' and c['kind'] == 'steady_decode'}
            self.assertFalse(before & after)

    def test_equivalent_axes_preserve_identity_and_fixed_batch_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path, spec = make_input(Path(directory))
            spec['workload']['matrices'] = {
                'decode-axis': {
                    'kind': 'steady_decode', 'split': 'train',
                    'concurrency': {'values': [2, 1]}, 'num_ctx': {'values': [256]},
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
