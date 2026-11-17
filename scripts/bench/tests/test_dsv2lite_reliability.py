"""Regression tests for scripts/bench/dsv2lite_reliability.py."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "dsv2lite_reliability.py"
SPEC = importlib.util.spec_from_file_location("dsv2lite_reliability", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
dsv2lite_reliability = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dsv2lite_reliability
SPEC.loader.exec_module(dsv2lite_reliability)


def _result(request_id: str, kind: str, ok: bool, reason: str):
    return dsv2lite_reliability.finish_result(
        request_id,
        kind,
        ok,
        200,
        time.time(),
        reason,
        output_text="stable" if ok else "",
        output_chunks=1 if ok else 0,
    )


def _trace(request_id: str, reason: str, active: int = 2, pending: int = 0, decode: int = 1):
    return {
        "request_id": request_id,
        "terminal_reason": reason,
        "active_set_size_max": active,
        "pending_queue_size_max": pending,
        "decode_batch_size_max": decode,
        "active_set_size_at_terminal": 0,
        "pending_queue_size_at_terminal": 0,
        "healthy_baseline_after_terminal": True,
        "prompt_tokens": 16,
        "completion_tokens": 4,
    }


class ReliabilityGateTests(unittest.TestCase):
    def test_trace_terminal_reason_overrides_client_disconnect_label(self):
        disconnected = _result("disconnect", "disconnect_early_bytes", False, "disconnected")
        disconnected.trace = _trace("disconnect", "error")
        disconnected.terminal_reason = dsv2lite_reliability.normalize_trace_terminal_reason(
            "error"
        )
        completed = _result("ok", "neighbor_success", True, "completed")
        completed.trace = _trace("ok", "completed_length")
        follow = _result("follow", "clean_follow_up", True, "completed")
        follow.trace = _trace("follow", "completed_length", active=1)

        scenario = dsv2lite_reliability.build_scenario(
            "cancel_disconnect",
            [disconnected, completed],
            follow,
            required_reasons={"disconnected", "completed"},
        )

        self.assertFalse(scenario.passed)
        self.assertEqual(scenario.counts["disconnected"], 0)
        self.assertIn("expected terminal reason 'disconnected' was not observed", scenario.failures)
        self.assertTrue(any("request failed unexpectedly: disconnect" in item for item in scenario.failures))

    def test_missing_trace_fields_fail_gate(self):
        completed = _result("ok", "neighbor_success", True, "completed")
        completed.trace = {
            "request_id": "ok",
            "terminal_reason": "completed_length",
            "active_set_size_max": 1,
        }
        follow = _result("follow", "clean_follow_up", True, "completed")
        follow.trace = _trace("follow", "completed_length", active=1)

        scenario = dsv2lite_reliability.build_scenario(
            "invalid_requests",
            [completed],
            follow,
            required_reasons={"completed"},
        )

        self.assertFalse(scenario.passed)
        self.assertTrue(any("trace for ok is missing fields" in item for item in scenario.failures))

    def test_dry_run_report_stays_green(self):
        args = argparse.Namespace(
            base_url="http://127.0.0.1:8000",
            model="deepseek-v2-lite",
            max_tokens=16,
            follow_up_tokens=8,
            overload_concurrency=12,
            mixed_concurrency=6,
            mixed_max_tokens=16,
            over_context_words=9000,
            timeout=180.0,
        )

        report = dsv2lite_reliability.dry_run_report(args)

        self.assertTrue(report["passed"])
        self.assertTrue(report["summary"]["passed"])
        self.assertEqual(report["summary"]["passed_scenarios"], 4)


if __name__ == "__main__":
    unittest.main()
