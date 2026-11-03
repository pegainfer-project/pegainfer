"""Training-only materialization and immutable fitted-bundle contract tests.

Transport fixtures exercise evidence verification; actual GPU/HTTP acceptance
is recorded separately in real calibration campaigns.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import ArtifactError, _read_json, _write_atomic, fitted_bundle_paths
from fit import FitError, _piecewise_fit, fit_normalized, write_document as write_fit
from normalize import normalize_campaign, write_document
from pairing import PairingContractError, load_role
from profile import ProfileError, materialize_calibration, materialize_profile
from test_normalize import collect_fixture, make_bundle


def training(base: Path):
    root, campaign = make_bundle(base, full_train=True)
    collect_fixture(root, campaign)
    normalized = normalize_campaign(root, scope='train')
    normalized_path, fit_path = base / 'normalized.json', base / 'fit.json'
    write_document(normalized_path, normalized)
    fitted = fit_normalized(normalized)
    write_fit(fit_path, fitted)
    return root, campaign, normalized_path, fit_path


class ProfileTests(unittest.TestCase):
    def test_materialization_requires_selected_train_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, campaign, normalized_path, fit_path = training(Path(tmp))
            normalized_bytes, fit_bytes = normalized_path.read_bytes(), fit_path.read_bytes()
            for mode in ('candidate', 'missing_surface', 'scope', 'normalized_campaign', 'fit_campaign'):
                with self.subTest(mode=mode):
                    normalized, fitted = json.loads(normalized_bytes), json.loads(fit_bytes)
                    if mode == 'candidate':
                        fitted['candidates'][1]['values_ms'][0][0] = -1
                    elif mode == 'missing_surface':
                        fitted['candidates'].pop(1)
                    elif mode == 'scope':
                        normalized['scope'] = 'validation'
                    elif mode == 'normalized_campaign':
                        normalized['campaign_id'] = 'other-campaign'
                    elif mode == 'fit_campaign':
                        fitted['campaign_id'] = 'other-campaign'
                    _write_atomic(normalized_path, normalized)
                    _write_atomic(fit_path, fitted)
                    with self.assertRaises((ProfileError, FitError, ArtifactError)):
                        materialize_profile(root, normalized_artifact=normalized_path, fit_artifact=fit_path)
                    self.assertFalse(fitted_bundle_paths(root, campaign)['selection'].exists())
                    if mode.endswith('_campaign'):
                        with self.assertRaisesRegex(ProfileError, 'different campaign'):
                            materialize_calibration(root, normalized_artifact=normalized_path, fit_artifact=fit_path)
            with self.subTest(mode='single_cold_point'), self.assertRaisesRegex(FitError, 'at least two distinct'):
                _piecewise_fit([{'input_tokens': 8, 'value': 1}, {'input_tokens': 8, 'value': 2}], metric='ttft_ms')

    def test_offline_sim_reader_binds_actual_profile_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, campaign, _, _ = training(Path(tmp))
            collect_fixture(root, campaign, roles=('sim',), split='sealed_test')
            entry = next(value for key, value in _read_json(root / 'execution_index.json')['cells'].items()
                         if key.endswith(':sim'))
            from artifacts import sealed_reference
            cell = next(c for c in campaign['workload']['cells'] if c['split'] == 'sealed_test')
            from request_plan import build_plan
            plan = build_plan(cell, campaign['client'])
            load_role(root, campaign, plan, 'sim', sealed_reference(entry))
            paths = fitted_bundle_paths(root, campaign)
            profile = _read_json(paths['profile'])
            profile['predictor']['model']['decode']['duration_us'][0] += 1
            _write_atomic(paths['profile'], profile)
            with self.assertRaises(PairingContractError):
                load_role(root, campaign, plan, 'sim', sealed_reference(entry))
