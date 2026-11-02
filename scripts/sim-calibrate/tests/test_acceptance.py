"""Report gate fault checks; these records are not production timing evidence."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report import assess_report, require_complete
from report import summarize_samples


def complete_record():
    samples = [{'key': 'a', 'actual_ms': 10, 'predicted_ms': 12,
                'absolute_error_ms': 2, 'relative_error': .2}]
    summary = summarize_samples(samples)
    cell = {'cell_id': 'c', 'split': 'sealed_test', 'concurrency': 2, 'status': 'measured',
            'coverage': {'status': 'in_domain'}, 'expected_count': 1, 'matched_count': 1,
            'pairing': {'complete': True, 'matched_count': 1}, 'samples': samples, **summary,
            **{role: {'status': 'succeeded'} for role in ('target', 'zero_cost', 'sim')}}
    return {'required_cell_ids': ['c'], 'required_queueing_cell_ids': ['c'],
            'required_roles': ['target', 'zero_cost', 'sim'], 'deferred_cells': [],
            'relative_denominator': 'positive_target_raw_ttft_ms', 'cells': [cell],
            'cell_count': 1, 'measured_cell_count': 1, 'expected_count': 1, 'matched_count': 1,
            'aggregates': {'all': summary, 'by_split': {'sealed_test': summary}},
            'queueing_ttft': {'cells': [cell], 'aggregates': {'sealed_test': summary}},
            'provenance': {'profile': {'profile': {'sha256': 'bound'}}}}


class AcceptanceTests(unittest.TestCase):
    def test_missing_ood_statistics_counts_scope_and_roles_never_pass(self):
        document = complete_record()
        self.assertEqual(assess_report(document)['status'], 'passed')
        mutations = (
            lambda d: d['required_cell_ids'].append('missing'),
            lambda d: d['required_queueing_cell_ids'].clear(),
            lambda d: d['cells'][0]['coverage'].update(status='out_of_domain'),
            lambda d: d['cells'][0]['coverage'].update(status='unsupported'),
            lambda d: d['cells'][0]['sim'].update(status='missing'),
            lambda d: d['cells'][0]['errors']['relative'].update(p99=None),
            lambda d: d['cells'][0].update(matched_count=0),
            lambda d: d['aggregates']['all'].update(count=2),
            lambda d: d['aggregates']['by_split'].clear(),
            lambda d: d['deferred_cells'].append({'kind': 'cached_prefill'}),
            lambda d: d.update(relative_denominator='effective_ttft'),
        )
        for mutate in mutations:
            changed = copy.deepcopy(document)
            mutate(changed)
            with self.subTest(changed=changed):
                self.assertEqual(assess_report(changed)['status'], 'failed')
                with self.assertRaises(ValueError):
                    require_complete(changed)
