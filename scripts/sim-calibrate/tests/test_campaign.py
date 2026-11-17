"""Named matrix sampling and seed overrides."""
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
from manifest import ManifestError
from probe import requires_cache_reset
from request_plan import build_plan

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
    spec['variants']['target']['identity']['model_config_sha256'] = config_digest
    for variant in spec['variants'].values():
        variant['identity']['model_revision'] = 'local'
        # The compiler records these flags; it does not start a server.
        variant['server_flags'] = []
    spec['output']['artifact_root'] = str(root / 'artifacts')
    path = root / 'spec.json'
    path.write_text(json.dumps(spec))
    return path, spec


class CampaignCompileTests(unittest.TestCase):
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
            plan = build_plan(cell, compiled['client'])
            self.assertTrue(requires_cache_reset(plan, 'target', compiled['variants']['target']['runtime'], plan['batches'][0]))
            for matrix_name in ('prefill-train', 'decode-train'):
                lifecycle = spec['workload']['matrices'][matrix_name]['cache_lifecycle']
                lifecycle['recovery'] = 'restart'
                path.write_text(json.dumps(spec))
                with self.subTest(matrix=matrix_name), self.assertRaisesRegex(ManifestError, 'restart.*cached_prefill'):
                    compile_campaign(path, write=False)
                del lifecycle['recovery']
            del matrix['cache_lifecycle']
            path.write_text(json.dumps(spec))
            with self.subTest(mode='decode_cache_without_lifecycle'), self.assertRaisesRegex(ManifestError, 'cache lifecycle'):
                compile_campaign(path, write=False)
            del plan['cache_lifecycle']
            with self.assertRaisesRegex(ValueError, 'reset lifecycle'):
                requires_cache_reset(plan, 'target', compiled['variants']['target']['runtime'], plan['batches'][0])


if __name__ == '__main__':
    unittest.main()
