"""Decode window arithmetic and complete-repeat aggregation edge cases."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steady_state import StatisticsError, _batch, aggregate_request_samples


def batch(times, counts):
    rows = {str(i): {'status': 'succeeded', 'index': i, 'first_token_timestamp': ts[0],
                     'last_token_timestamp': ts[-1], 'generation_event_timestamps': ts,
                     'interval_token_counts': cs} for i, (ts, cs) in enumerate(zip(times, counts))}
    return _batch({'request_keys': list(rows), 'phase': 'measured', 'repeat': 0}, rows)


class SteadyStateTests(unittest.TestCase):
    def test_inclusive_full_window_right_event_tokens_and_zero_duration(self):
        # Window [2, 5]. Zero-time events count toward tokens; the first event is excluded.
        result = batch([[1,2,2,3,5,6], [2,3,5]], [[7,2,3,1,9], [2,2]])
        self.assertTrue(result['qualified'])
        a, b = result['requests']
        self.assertEqual((result['window_start'], result['window_end']), (2,5))
        self.assertEqual(a['retained_event_indices'], [2,3,4])
        self.assertEqual((a['event_interval_count'], a['interval_token_count'], a['distinct_timestamp_count']), (3,6,3))
        self.assertEqual(a['tpot_ms'], 500)
        self.assertEqual(b['tpot_ms'], 750)
        self.assertEqual(result['mean_tpot_ms'], 625)

    def test_one_full_interval_is_sufficient_and_unsplittable_events_are_excluded(self):
        result = batch([[1,2,4,6], [2,4]], [[100,2,100], [4]])
        self.assertTrue(result['qualified'])
        self.assertEqual(result['requests'][0]['retained_event_indices'], [2])
        self.assertEqual(result['requests'][0]['interval_token_count'], 2)
        self.assertEqual(result['mean_tpot_ms'], 750)

    def test_hierarchical_mean_dispersion_and_actual_repeat_count(self):
        rows = [{'cell_id': 'cell', 'repeat': r, 'index': i, 'requests': 2, 'value': value}
                for r, values in enumerate(((1,3), (2,8), (4,20))) for i, value in enumerate(values)]
        result = aggregate_request_samples(rows)
        self.assertAlmostEqual(result['value'], 19/3)
        self.assertEqual((result['repeat_count'], result['request_count']), (3,6))
        self.assertEqual(result['dispersion']['mad'], 3)
        with self.assertRaises(StatisticsError):
            aggregate_request_samples(rows[:-1])
        with self.assertRaises(StatisticsError):
            aggregate_request_samples([*rows, rows[0]])
