"""Raw TTFT statistics and full current evidence report verification.

Raw SSE fixtures validate lineage and gate behavior, not model accuracy.
"""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import ArtifactError
from acceptance import require_complete
from report import _sample, report_campaign, summarize_samples, verify_report
from test_normalize import collect_fixture, make_bundle


class ReportStatisticsTests(unittest.TestCase):
    def test_raw_request_errors_precede_percentiles_without_baseline_subtraction(self):
        rows = [{'key': str(i), 'repeat': i, 'index': 0,
                 'measurements': {'target': {'ttft_ms': actual}, 'sim': {'ttft_ms': predicted},
                                  'zero_cost': {'ttft_ms': 900}}}
                for i, (actual, predicted) in enumerate(((10, 14), (20, 12)))]
        result = summarize_samples([_sample(row) for row in rows])
        self.assertEqual(result['actual']['p50'], 15)
        self.assertEqual(result['predicted']['p50'], 13)
        self.assertEqual(result['errors']['absolute_ms']['p50'], 6)
        self.assertAlmostEqual(result['errors']['relative']['p99'], .4)
        rows[0]['measurements']['target']['ttft_ms'] = 0
        with self.assertRaises(ValueError):
            _sample(rows[0])


class ReportEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root, cls.campaign = make_bundle(Path(cls.temporary.name), full_train=True)
        collect_fixture(cls.root, cls.campaign)
        collect_fixture(cls.root, cls.campaign, roles=('target',), split='sealed_test')
        # The diagnostic held-out reference may have no computable decode window.
        collect_fixture(cls.root, cls.campaign, roles=('zero_cost',), split='sealed_test',
                        collapsed_phases=('warmup', 'measured'))
        collect_fixture(cls.root, cls.campaign, roles=('sim',), split='sealed_test')
        cls.document = report_campaign(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_report_binds_sources_and_rejects_statistics_or_scope_tampering(self):
        self.assertEqual(self.document['status'], 'succeeded')
        self.assertEqual(self.document['expected_count'], 4)
        self.assertFalse(self.document['normalized']['cells'][0]['decode_statistics']['zero_cost']['qualified'])
        require_complete(self.document)
        for mode in ('percentile', 'scope', 'count', 'profile', 'attestation'):
            document = copy.deepcopy(self.document)
            if mode == 'percentile':
                document['cells'][0]['actual']['p99'] += 1
            elif mode == 'scope':
                document['required_cell_ids'].clear()
            elif mode == 'count':
                document['aggregates']['all']['count'] += 1
            elif mode == 'profile':
                document['provenance']['profile']['profile']['sha256'] = '0'*64
            else:
                document['normalized']['source']['selected_attempts'][2]['binding']['profile']['sha256'] = '0'*64
            with self.subTest(mode=mode), self.assertRaises((ValueError, ArtifactError)):
                verify_report(document)
        ref = self.document['normalized']['source']['selected_attempts'][0]
        evidence = self.root/ref['path']/'binding.json'
        original = evidence.read_bytes()
        try:
            evidence.write_bytes(original+b'\n')
            with self.assertRaises((ValueError, ArtifactError)):
                verify_report(self.document)
        finally:
            evidence.write_bytes(original)
