"""Reference preparation and signed paired training through production readers."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import ArtifactError, _read_json, _write_atomic
from baseline import derive_cell, prepare_references, resolve_reference, select_reference, verify_preparation
from campaign import compile_campaign
from fit import FitError, fit_normalized
from normalize import normalize_campaign, verify_normalized
from test_campaign import make_input
from test_normalize import collect_fixture, make_bundle


class BaselineTests(unittest.TestCase):
    def test_signed_requests_use_encoded_interval_and_hierarchical_mean(self):
        cell = {'kind': 'steady_decode', 'cell_id': 'decode', 'concurrency': 2, 'repeats': 2}
        rows = [{'key': str(i), 'repeat': i // 2, 'index': i % 2,
                 'decode_window_tpot_ms': {'target': target, 'zero_cost': reference}}
                for i, (target, reference) in enumerate(zip((19, 21, 18, 22), (21, 19, 42, 22)))]
        result = derive_cell(cell, rows, {'interval_us': 20000})
        self.assertTrue(result['qualified'])
        self.assertEqual([row['baseline_residual']['tpot_ms'] for row in rows], [1, -1, 22, 2])
        self.assertEqual([row['effective']['tpot_ms'] for row in rows], [18, 22, -4, 20])
        self.assertEqual([row['effective_mean'] for row in result['batches']], [20, 8])
        self.assertEqual(result['representative'], 14)
        self.assertEqual(result['dispersion']['mad'], 6)
        self.assertEqual(result['request_count'], 4)
        # Actual encoded microseconds, rather than the unrounded target mean, are added back.
        derive_cell(cell, rows, {'interval_us': 20001})
        self.assertAlmostEqual(rows[2]['effective']['tpot_ms'], -3.999)

    def test_cell_numeric_qualification_preserves_signed_samples(self):
        for kind, effective, qualified in (
                ('steady_decode', [0, 0], False), ('steady_decode', [-1, -2], False),
                ('steady_decode', [-10, 30], True), ('steady_decode', [float('inf'), 1], False),
                ('cold_prefill', [-10, 10], True), ('cold_prefill', [-10, 8], False),
                ('cold_prefill', [float('nan'), 2], False)):
            with self.subTest(kind=kind, effective=effective):
                cell = {'kind': kind, 'cell_id': 'cell', 'concurrency': 1, 'repeats': 2}
                rows = [{'key': str(i), 'repeat': i, 'index': 0,
                         'decode_window_tpot_ms': {'target': value + 1, 'zero_cost': 2},
                         'measurements': {'target': {'ttft_ms': value + 1}, 'zero_cost': {'ttft_ms': 1}}}
                        for i, value in enumerate(effective)]
                result = derive_cell(cell, rows, {'interval_us': 1000})
                self.assertEqual(result['qualified'], qualified)
                self.assertEqual(len(rows), 2)
                json.dumps(result, allow_nan=False)
                json.dumps([{'effective': r['effective'], 'residual': r['baseline_residual']} for r in rows], allow_nan=False)
                if kind == 'cold_prefill' and qualified:
                    self.assertEqual(result['representative'], 0)

    def test_preparation_requires_targets_and_freezes_profiles_without_heldout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, campaign = make_bundle(Path(tmp))
            self.assertFalse((root / 'profiles/zero_cost/profile.json').exists())
            with self.assertRaises((ArtifactError, OSError)):
                prepare_references(root)
            collect_fixture(root, campaign, roles=('target',))
            prepared = prepare_references(root)
            self.assertEqual(prepared, prepare_references(root))
            self.assertEqual(len(prepared['inputs']['selected_targets']), 1)
            for cell in campaign['workload']['cells']:
                reference = resolve_reference(root, campaign, select_reference(root, campaign, cell['cell_id']), cell['cell_id'])
                self.assertEqual(reference['interval_us'], 5000 if cell['split'] == 'train' else 0)
                profile = _read_json(root / reference['profile_path'])
                self.assertEqual(profile['predictor']['model']['prefill']['duration_us'], [0, 0])
                self.assertEqual(profile['predictor']['model']['decode']['duration_us'], [reference['interval_us']] * 4)
                self.assertEqual(profile['predictor']['model']['mixed'], {'kind': 'composition_additive', 'intercept_us': 0})
            train = next(c for c in campaign['workload']['cells'] if c['split'] == 'train')
            binding = select_reference(root, campaign, train['cell_id'])
            with self.assertRaises(ArtifactError):
                resolve_reference(root, campaign, binding, 'another-cell')
            for field in ('interval_us', 'target'):
                changed = copy.deepcopy(prepared)
                changed['cells'][train['cell_id']][field] = 0 if field == 'interval_us' else None
                with self.assertRaises(ArtifactError):
                    verify_preparation(root, campaign, changed)
            profile_path = root / prepared['cells'][train['cell_id']]['profile_path']
            profile = _read_json(profile_path)
            profile['predictor']['model']['decode']['duration_us'][0] += 1
            _write_atomic(profile_path, profile)
            with self.assertRaises(ArtifactError):
                resolve_reference(root, campaign, binding, train['cell_id'])

    def test_invalid_cell_blocks_complete_training_without_dropping_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, campaign = make_bundle(Path(tmp))
            collect_fixture(root, campaign, roles=('target',))
            collect_fixture(root, campaign, roles=('zero_cost',), event_interval_ms=15)
            normalized = normalize_campaign(root, scope='train')
            self.assertEqual(normalized['status'], 'invalid')
            self.assertFalse(normalized['fit_ready'])
            cell = normalized['cells'][0]
            self.assertEqual(cell['matched_count'], 4)
            self.assertLess(cell['training']['representative'], 0)
            self.assertEqual(len(cell['observations']), 4)
            self.assertTrue(all(row['effective']['tpot_ms'] < 0 for row in cell['observations']))
            with self.assertRaises(FitError):
                fit_normalized(normalized)

    def test_cold_single_output_and_complete_decode_train_reach_fit(self):
        for cold_only in (True, False):
            with self.subTest(cold_only=cold_only), tempfile.TemporaryDirectory() as tmp:
                path, spec = make_input(Path(tmp))
                for variant in spec['variants'].values():
                    variant['runtime']['cache'] = {'policy': 'off'}
                common = {'split': 'train', 'warmup_repeats': 1, 'repeats': 2, 'arrival': {'mode': 'fixed_batch'}}
                spec['workload']['matrices'] = {
                    'cold': {**common, 'kind': 'cold_prefill', 'concurrency': 1, 'input_tokens': [8, 32], 'output_tokens': 1}}
                if not cold_only:
                    spec['workload']['matrices']['decode'] = {
                        **common, 'kind': 'steady_decode', 'concurrency': [1, 2], 'num_ctx': [16, 64], 'output_tokens': 8}
                path.write_text(json.dumps(spec))
                campaign = compile_campaign(path)
                root = Path(tmp) / 'artifacts'
                collect_fixture(root, campaign)
                prepared = prepare_references(root)
                self.assertEqual(prepared['inputs']['target_statistics'] is None, cold_only)
                normalized = normalize_campaign(root, scope='train')
                self.assertTrue(normalized['fit_ready'], normalized['failures'])
                verify_normalized(normalized)
                fitted = fit_normalized(normalized)
                self.assertEqual(fitted['status'], 'succeeded', fitted['failures'])
                for candidate in fitted['candidates'][:1 if cold_only else 3]:
                    self.assertEqual(candidate['status'], 'fitted')
                self.assertEqual(fitted['source']['training'], normalized['source'])
                for field in ('effective', 'reference_interval_ms', 'baseline_residual'):
                    changed = copy.deepcopy(normalized)
                    row = changed['cells'][0]['observations'][0]
                    if field == 'reference_interval_ms':
                        row[field] += 0.001
                    else:
                        row[field][next(iter(row[field]))] += 1
                    with self.assertRaises(FitError):
                        fit_normalized(changed)


if __name__ == '__main__':
    unittest.main()
