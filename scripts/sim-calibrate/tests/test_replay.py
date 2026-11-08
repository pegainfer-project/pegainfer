"""Held-out replay entry contract checks."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report import report_campaign, verify_report
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
                document = report_campaign(root)
                self.assertEqual(document['status'], 'replay_required')
                self.assertEqual(document['matched_count'], 0)
                self.assertEqual(document['acceptance']['status'], 'failed')
                # Cached training is supported offline; missing acquisition
                # still cannot satisfy the existing online replay gate.
                self.assertEqual(document['deferred_cells'], [])
                verify_report(document)
