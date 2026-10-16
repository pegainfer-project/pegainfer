"""Target-only lineage, immutable publishing and the C4/C5 consumer boundary."""
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import ArtifactError, _read_json, _write_atomic, file_reference
from fit import FitError, fit_normalized
from normalize import normalize_campaign
from target_statistics import (collect_target_statistics, read_target_statistics,
                               verify_target_statistics, write_document)
from test_normalize import make_bundle, collect_fixture


class TargetStatisticsTests(unittest.TestCase):
    def test_target_without_reference_and_later_execution_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, campaign = make_bundle(Path(tmp))
            collect_fixture(root, campaign, roles=('target',))
            document = collect_target_statistics(root)
            self.assertEqual(document['status'], 'succeeded', document['failures'])
            self.assertEqual(document['required_roles'], ['target'])
            self.assertFalse(document['fit_ready'])
            self.assertEqual(document['cell_count'], 1)
            self.assertEqual(normalize_campaign(root, scope='train')['status'], 'invalid')
            with self.assertRaises(FitError):
                fit_normalized(document)
            output = Path(tmp) / 'target.json'
            write_document(output, document)
            write_document(output, document)
            collect_fixture(root, campaign, roles=('target',), split='sealed_test')
            self.assertEqual(read_target_statistics(output), document)
            for flag in ('--output', '--verify'):
                result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'target_statistics.py'),
                                         '--artifact-root', str(root), flag, str(output if flag == '--verify' else Path(tmp)/'later.json')],
                                        text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_or_unqualified_target_and_mutations_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, campaign = make_bundle(Path(tmp))
            collect_fixture(root, campaign, roles=('zero_cost',))
            self.assertEqual(collect_target_statistics(root)['status'], 'invalid')
            collect_fixture(root, campaign, roles=('target',), collapsed_phases=('measured',))
            self.assertEqual(collect_target_statistics(root)['status'], 'invalid')
            collect_fixture(root, campaign, roles=('target',))
            document = collect_target_statistics(root)
            for mode in ('mean', 'window', 'fit_ready', 'sources', 'snapshot', 'scope', 'counts'):
                with self.subTest(mode=mode):
                    changed = copy.deepcopy(document)
                    if mode == 'mean': changed['cells'][0]['statistics']['summary']['mean'] += 1
                    elif mode == 'window': changed['cells'][0]['statistics']['batches'][1]['window_start'] += 1
                    elif mode == 'fit_ready': changed['fit_ready'] = True
                    elif mode == 'sources': changed['source']['statistics_sources'].pop('steady_state.py')
                    elif mode == 'snapshot': changed['source']['selected_attempts'] = []
                    elif mode == 'scope': changed['required_cell_ids'] = []
                    else: changed['qualified_cell_count'] = 999
                    with self.assertRaises((ArtifactError, ValueError)):
                        verify_target_statistics(changed)
                    with self.assertRaises((ArtifactError, ValueError)):
                        write_document(Path(tmp)/f'{mode}.json', changed)
            with self.assertRaises(ArtifactError):
                write_document(root/'raw'/'out.json', document)
            # Rehashed sidecar edits still fail the shared canonical recomputation.
            index = _read_json(root/'execution_index.json')
            entry = next(e for key, e in index['cells'].items() if key.endswith(':target'))
            path = root/entry['path']/'decode_statistics.json'
            stats = _read_json(path)
            stats['summary']['mean'] += 1
            _write_atomic(path, stats)
            entry['statistics'] = file_reference(path)
            _write_atomic(root/'execution_index.json', index)
            self.assertEqual(collect_target_statistics(root)['status'], 'invalid')
