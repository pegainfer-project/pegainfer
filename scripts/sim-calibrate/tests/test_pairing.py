"""Exact current-plan pairing: identity, completeness and repeated content."""
import copy
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import _read_json, read_sealed_attempt, sealed_reference
from manifest import sha256_json
from pairing import index_plan_measurements, pair_plan_measurements
from request_plan import build_plan
from test_normalize import collect_fixture, make_bundle


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root, campaign = make_bundle(Path(self.tmp.name))
        collect_fixture(self.root, campaign)
        cell = next(c for c in campaign['workload']['cells'] if c['split'] == 'train')
        self.plan = build_plan(cell, campaign['client'])
        index = _read_json(self.root / 'execution_index.json')
        self.attempts = {role: read_sealed_attempt(self.root, self.plan, role,
                         sealed_reference(index['cells'][cell['cell_id'] + ':' + role]))
                         for role in ('target', 'zero_cost')}

    def test_repeated_content_distinct_ids_and_reordered_rows_pair_by_key(self):
        self.attempts['zero_cost']['envelope']['measurements'].reverse()
        paired = pair_plan_measurements(self.plan, self.attempts, ('target', 'zero_cost'))
        self.assertEqual(paired['matched_count'], 4)
        self.assertEqual(paired['observed_count'], {'target': 6, 'zero_cost': 6})
        for logical, rows in paired['pairs']:
            self.assertEqual(logical['phase'], 'measured')
            self.assertEqual(rows['target']['body_sha256'], rows['zero_cost']['body_sha256'])
            self.assertNotEqual(rows['target']['transport_request_id'], rows['zero_cost']['transport_request_id'])
        self.assertEqual(len({row['body_sha256'] for row in self.attempts['target']['envelope']['measurements']}), 1)

    def test_missing_duplicate_extra_keys_and_invalid_identity_never_fall_back(self):
        for fault in ('missing', 'duplicate', 'extra', 'implicit', 'wrong_key', 'failed', 'instance', 'workload_missing', 'shape'):
            with self.subTest(fault=fault):
                attempt = copy.deepcopy(self.attempts['target'])
                rows = attempt['envelope']['measurements']
                row = rows[-1]
                if fault == 'missing': rows.pop()
                elif fault == 'duplicate': rows.append(copy.deepcopy(row))
                elif fault == 'extra':
                    extra = copy.deepcopy(row)
                    extra['index'] = 99
                    extra['pairing_key'] = f"{self.plan['cell_id']}:measured:{extra['repeat']}:99"
                    rows.append(extra)
                elif fault == 'implicit': del row['pairing_key']
                elif fault == 'wrong_key': row['pairing_key'] += 'bad'
                elif fault == 'failed': row.update(status='failed', error='injected')
                elif fault == 'instance': row['request_instance_id'] = rows[0]['request_instance_id']
                elif fault == 'shape': row['prompt_tokens'] += 1
                else:
                    attempt['workload']['requests'].pop()
                    attempt['workload']['sha256'] = sha256_json(attempt['workload']['requests'])
                with self.assertRaises((ValueError, KeyError)):
                    index_plan_measurements(self.plan, attempt['envelope'], attempt['workload'], 'target')

    def test_content_mismatch_cannot_pass_with_locally_consistent_digests(self):
        attempt = self.attempts['zero_cost']
        row = attempt['envelope']['measurements'][-1]
        key = row['pairing_key']
        evidence = next(item for item in attempt['workload']['requests'] if item['pairing_key'] == key)
        row['body_sha256'] = evidence['body_sha256'] = '1' * 64
        attempt['workload']['sha256'] = sha256_json(attempt['workload']['requests'])
        for item in attempt['envelope']['measurements']:
            item['workload_evidence_sha256'] = attempt['workload']['sha256']
        with self.assertRaisesRegex(ValueError, 'cross-role body_sha256'):
            pair_plan_measurements(self.plan, self.attempts, ('target', 'zero_cost'))
        with self.assertRaisesRegex(ValueError, 'roles'):
            pair_plan_measurements(self.plan, {'target': self.attempts['target']}, ('target', 'zero_cost'))
