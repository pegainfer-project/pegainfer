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


if __name__ == '__main__':
    unittest.main()
