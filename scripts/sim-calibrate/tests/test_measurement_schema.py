"""Validate the client evidence envelope before any fit can consume it."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measurement_schema import MeasurementError, SCHEMA_VERSION, validate_envelope, validate_row


def row() -> dict:
    cell = 'a' * 64
    return {
        'kind': 'measurement_row', 'schema_version': SCHEMA_VERSION,
        'cell_id': cell, 'pairing_key': f'{cell}:measured:0:0', 'plan_sha256': 'b' * 64,
        'phase': 'measured', 'repeat': 0, 'index': 0, 'variant': 'target', 'status': 'succeeded',
        'planned_dispatch_offset_ms': 0, 'actual_dispatch_timestamp': 1.0,
        'first_token_timestamp': 1.01, 'last_token_timestamp': 1.03,
        'ttft_ms': 10, 'tpot_ms': 10, 'stream_chunk_tpot_ms': 10, 'e2e_ms': 31,
        'prompt_tokens': 128, 'completion_tokens': 3, 'token_timing_valid': True,
        'generation_event_timestamps': [1.01, 1.02, 1.03], 'interval_token_counts': [1, 1],
        'first_event_token_count': 1, 'event_source': 'client_sse_generated_content',
        'request_instance_id': 'request-task-0', 'transport_request_id': None, 'error': None,
        **{field: 'c' * 64 for field in ('prompt_sha256', 'token_sha256', 'body_sha256', 'wire_body_sha256', 'workload_evidence_sha256')},
    }


def envelope() -> dict:
    return {
        'kind': 'measurement_envelope', 'schema_version': SCHEMA_VERSION,
        'plan_sha256': 'b' * 64, 'variant': 'target', 'measurements': [row()],
        'variant_identity': {'endpoint': 'http://127.0.0.1:8000', 'model': 'qwen3',
                             'engine': 'vllm', 'engine_version': 'test', 'model_revision': 'local',
                             'model_config_sha256': 'e' * 64},
        'provenance': {'client_name': 'vllm-bench', 'client_version': 'test',
                       **{key: 'd' * 64 for key in ('binary_sha256', 'tokenizer_sha256', 'raw_bundle_index_sha256', 'source_command_sha256')}},
    }


class MeasurementSchemaTests(unittest.TestCase):
    def test_json_roundtrip_and_duplicate_content_with_distinct_keys(self):
        data = envelope()
        other = copy.deepcopy(data['measurements'][0])
        other['index'] = 1
        other['pairing_key'] = f"{other['cell_id']}:measured:0:1"
        other['request_instance_id'] = 'request-task-1'
        data['measurements'].append(other)
        validate_envelope(json.loads(json.dumps(data)))

    def test_missing_or_inconsistent_evidence_is_rejected(self):
        changes = [
            lambda r: r.pop('body_sha256'),
            lambda r: r.update(pairing_key='wrong-key'),
            lambda r: r.pop('transport_request_id'),
            lambda r: r.pop('token_timing_valid'),
            lambda r: r.update(request_instance_id=''),
            lambda r: r.update(ttft_ms=11),
            lambda r: r.update(last_token_timestamp=0),
            lambda r: r.update(interval_token_counts=[1]),
            lambda r: r.update(completion_tokens=4),
            lambda r: r.update(token_timing_valid=1),
            lambda r: r.update(ttft_ms=float('nan')),
            lambda r: r.update(schema_version=1),
            lambda r: r.update(schema_version=2.0),
            lambda r: r.update(variant='replay'),
        ]
        for change in changes:
            with self.subTest(change=change):
                data = row()
                change(data)
                with self.assertRaises(MeasurementError):
                    validate_row(data)
        data = envelope()
        data['measurements'].append(copy.deepcopy(data['measurements'][0]))
        with self.assertRaisesRegex(MeasurementError, 'duplicate'):
            validate_envelope(data)
        data = envelope()
        data['measurements'][0]['plan_sha256'] = 'e' * 64
        with self.assertRaisesRegex(MeasurementError, 'different plan'):
            validate_envelope(data)
        for field in ('client_version', 'tokenizer_sha256'):
            data = envelope()
            data['provenance'].pop(field)
            with self.assertRaises(MeasurementError):
                validate_envelope(data)

    def test_failed_evidence_stays_explicitly_missing(self):
        data = row()
        data.update(status='failed', error='request timeout', token_timing_valid=None,
                    generation_event_timestamps=[], interval_token_counts=[])
        for field in ('first_token_timestamp', 'last_token_timestamp', 'ttft_ms', 'tpot_ms', 'stream_chunk_tpot_ms', 'e2e_ms'):
            data[field] = None
        validate_row(data)
        data['error'] = None
        with self.assertRaises(MeasurementError):
            validate_row(data)


if __name__ == '__main__':
    unittest.main()
