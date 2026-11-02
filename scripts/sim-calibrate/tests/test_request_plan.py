"""Cached custom-input token nesting and phase shapes."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from campaign import compile_campaign
from request_plan import build_plan
from test_campaign import make_input


class RequestPlanTests(unittest.TestCase):
    def test_cached_custom_inputs_preserve_token_nesting_and_phase_shape(self):
        from cached_prefill import batch_payload, validate_workload_collection, workload_documents
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


if __name__ == '__main__':
    unittest.main()
