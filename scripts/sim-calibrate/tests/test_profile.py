"""Training-only materialization and immutable fitted-bundle contract tests.

Transport fixtures exercise evidence verification; actual GPU/HTTP acceptance
is recorded separately in the C6 production smoke.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import ArtifactError, _read_json, _write_atomic, file_reference, fitted_bundle_paths
from fit import FitError, fit_normalized, write_document as write_fit
from normalize import normalize_campaign, write_document
from pairing import load_role
from profile import ProfileError, materialize_profile
from test_normalize import collect_fixture, make_bundle


def training(base: Path):
    root, campaign = make_bundle(base, full_train=True)
    collect_fixture(root, campaign)
    normalized = normalize_campaign(root, scope='train')
    normalized_path, fit_path = base / 'normalized.json', base / 'fit.json'
    write_document(normalized_path, normalized)
    fitted = fit_normalized(normalized, source_digest=file_reference(normalized_path)['sha256'])
    write_fit(fit_path, fitted)
    return root, campaign, normalized_path, fit_path


class ProfileTests(unittest.TestCase):
    def test_corrupt_fit_or_training_never_publishes_even_with_updated_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, campaign, normalized_path, fit_path = training(Path(tmp))
            normalized_bytes, fit_bytes = normalized_path.read_bytes(), fit_path.read_bytes()
            for mode in ('candidate', 'missing_surface', 'training_value', 'scope', 'campaign', 'fitter_sources'):
                with self.subTest(mode=mode):
                    normalized, fitted = json.loads(normalized_bytes), json.loads(fit_bytes)
                    if mode == 'candidate':
                        fitted['candidates'][2]['values_ms'][0][0] += 1
                    elif mode == 'missing_surface':
                        fitted['candidates'].pop(2)
                    elif mode == 'training_value':
                        normalized['observations'][0]['effective']['ttft_ms'] += 1
                    elif mode == 'scope':
                        normalized['scope'] = 'validation'
                    elif mode == 'campaign':
                        normalized['campaign_id'] = 'other'
                    else:
                        fitted['source']['fitter_sources']['fit.py'] = '0' * 64
                    _write_atomic(normalized_path, normalized)
                    fitted['source']['normalized_sha256'] = file_reference(normalized_path)['sha256']
                    _write_atomic(fit_path, fitted)
                    with self.assertRaises((ProfileError, FitError, ArtifactError)):
                        materialize_profile(root, normalized_artifact=normalized_path, fit_artifact=fit_path)
                    self.assertFalse(fitted_bundle_paths(root, campaign)['selection'].exists())

    def test_offline_sim_reader_binds_selection_and_revalidates_training(self):
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
            selection = _read_json(paths['selection'])
            selection['files'][paths['profile'].name] = file_reference(paths['profile'])
            _write_atomic(paths['selection'], selection)
            with self.assertRaises(ProfileError):
                load_role(root, campaign, plan, 'sim', sealed_reference(entry))
