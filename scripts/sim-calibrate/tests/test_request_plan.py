"""Fixed-batch identities, bootstrap scheduling, and corruption rejection."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from campaign import compile_campaign
from manifest import sha256_json
from request_plan import PlanError, build_plan, validate_plan
from test_campaign import make_input


class RequestPlanTests(unittest.TestCase):
    def test_phase_counts_and_prefix_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = make_input(Path(directory))
            compiled = compile_campaign(path, write=False)
            cached_prefix_seeds = set()
            for cell in compiled['workload']['cells']:
                plan = build_plan(cell, compiled['client'])
                validate_plan(json.loads(json.dumps(plan)))
                measured = [b for b in plan['batches'] if b['phase'] == 'measured']
                self.assertEqual(len(measured), cell['repeats'])
                for batch in plan['batches']:
                    self.assertEqual(len(batch['request_keys']), cell['concurrency'])
                keys = [r['pairing_key'] for r in plan['logical_requests']]
                self.assertEqual(len(keys), len(set(keys)))
                if cell['kind'] != 'cached_prefill':
                    self.assertTrue(all('prompt' not in r and 'prompt_uri' not in r for r in plan['logical_requests']))
                if cell['kind'] == 'cached_prefill':
                    self.assertEqual([b['phase'] for b in plan['batches']], ['bootstrap', 'measured'] * 3)
                    cached_prefix_seeds.update(b['prefix_seed'] for b in plan['batches'])
            self.assertEqual(len(cached_prefix_seeds), 1)

    def test_cached_custom_inputs_preserve_token_nesting_and_phase_shape(self):
        from cached_prefix_workload import batch_payload, validate_workload_collection, workload_documents
        from tokenizers import Tokenizer
        with tempfile.TemporaryDirectory() as directory:
            path, spec = make_input(Path(directory))
            spec['workload']['matrices']['cached-train']['warmup_repeats'] = 1
            path.write_text(json.dumps(spec))
            compiled = compile_campaign(path, write=False)
            plans = [build_plan(cell, compiled['client']) for cell in compiled['workload']['cells']
                     if cell['kind'] == 'cached_prefill']
            validate_workload_collection(plans)
            tokenizer = Tokenizer.from_file(str(path.parent / 'model/tokenizer.json'))
            self.assertEqual(len({p['cached_workload']['prefix_seed'] for p in plans}), 1)
            prefixes = {}
            for plan in plans:
                prefix = plan['cached_workload']['prefix_token_ids']
                prefixes.setdefault(len(prefix), prefix)
                self.assertEqual(prefixes[len(prefix)], prefix)
                documents = workload_documents(plan)
                for batch in plan['batches']:
                    payload = batch_payload(plan, batch)
                    self.assertIn(payload, documents.values())
                    imported = [json.loads(line) for line in payload.splitlines()]
                    self.assertEqual(len(imported), 1)
                    logical = next(row for row in plan['logical_requests'] if row['pairing_key'] == batch['request_keys'][0])
                    actual = tokenizer.encode(imported[0]['prompt'], add_special_tokens=False).ids
                    self.assertEqual(actual, logical['token_ids'])
                    self.assertEqual(actual[:len(prefix)], prefix)
                    self.assertEqual(len(actual), len(prefix) if batch['phase'] == 'bootstrap' else len(prefix) + plan['new_tokens'])
                    self.assertEqual(imported[0]['output_tokens'], 1 if batch['phase'] == 'bootstrap' else 8)
            short, long = (prefixes[p] for p in sorted(prefixes))
            self.assertEqual(short, long[:len(short)])

    def test_rehashed_invalid_plan_cannot_bypass_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = make_input(Path(directory))
            compiled = compile_campaign(path, write=False)
            cell = next(c for c in compiled['workload']['cells'] if c['kind'] == 'steady_decode' and c['concurrency'] == 4)
            original = build_plan(cell, compiled['client'])
            changes = [
                lambda p: p.update(num_ctx=1),
                lambda p: p['decode_statistics'].update(window='central_80_percent'),
                lambda p: p['logical_requests'].pop(),
                lambda p: p['logical_requests'].append(copy.deepcopy(p['logical_requests'][0])),
                lambda p: p['batches'][0]['request_keys'].pop(),
                lambda p: p['batches'][0].update(seed=0),
                lambda p: p['batches'][0].update(prefix_seed=1),
                lambda p: p['client'].update(binary_sha256='unproven'),
                lambda p: p['logical_requests'][0].update(input_tokens=1),
                lambda p: p['logical_requests'][0].update(prompt_uri='external.json'),
            ]
            for change in changes:
                with self.subTest(change=change):
                    plan = copy.deepcopy(original)
                    change(plan)
                    plan['plan_sha256'] = sha256_json({k: v for k, v in plan.items() if k != 'plan_sha256'})
                    with self.assertRaises(PlanError):
                        validate_plan(plan)
            original['logical_requests'][0]['output_tokens'] += 1
            with self.assertRaisesRegex(PlanError, 'digest mismatch'):
                validate_plan(original)

    def test_shared_prefix_seed_cannot_be_changed_by_rehashing(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = make_input(Path(directory))
            compiled = compile_campaign(path, write=False)
            cell = next(c for c in compiled['workload']['cells'] if c['kind'] == 'cached_prefill')
            plan = build_plan(cell, compiled['client'])
            plan['batches'][0]['prefix_seed'] += 1
            plan['plan_sha256'] = sha256_json({k: v for k, v in plan.items() if k != 'plan_sha256'})
            with self.assertRaisesRegex(PlanError, 'prefix seed'):
                validate_plan(plan)


if __name__ == '__main__':
    unittest.main()
