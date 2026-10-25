"""Held-out replay entry contract checks."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import replay_campaign
from report import verify_report
from test_normalize import make_bundle
from test_campaign import make_input
from campaign import compile_campaign


class ReplayTests(unittest.TestCase):
    def test_missing_acquisition_remains_replay_required(self):
        for cached_train in (False, True):
            with self.subTest(cached_train=cached_train), tempfile.TemporaryDirectory() as tmp:
                if cached_train:
                    path, _ = make_input(Path(tmp))
                    compile_campaign(path)
                    root = Path(tmp)/'artifacts'
                else:
                    root, _ = make_bundle(Path(tmp), full_train=True)
                document = replay_campaign(root)
                self.assertEqual(document['status'], 'replay_required')
                self.assertEqual(document['matched_count'], 0)
                self.assertEqual(document['acceptance']['status'], 'failed')
                self.assertEqual(bool(document['deferred_cells']), cached_train)
                self.assertTrue(all(cell['split'] == 'train' for cell in document['deferred_cells']))
                verify_report(document)
