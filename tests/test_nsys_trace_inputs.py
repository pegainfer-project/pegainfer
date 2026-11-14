"""Exercise trace preservation through the public Nsight report command."""

import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "nsys_tail_stats.py"


class NsysTraceInputTests(unittest.TestCase):
    def create_trace(self, path):
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO StringIds VALUES (1, 'cuGraphLaunch')")
            conn.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
                "(start INTEGER, end INTEGER, nameId INTEGER)"
            )
            conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0, 120000, 1)")

    def run_report(self, trace, output):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(trace), "--out", str(output)],
            capture_output=True,
            text=True,
        )

    def test_missing_trace_does_not_create_a_database_or_report(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "missing.sqlite"
            output = Path(directory) / "report.md"
            result = self.run_report(trace, output)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertFalse(trace.exists())
            self.assertFalse(output.exists())

    def test_report_cannot_overwrite_input_or_its_aliases(self):
        for alias in ("same", "symlink", "hardlink"):
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as directory:
                trace = Path(directory) / "trace.sqlite"
                self.create_trace(trace)
                original = trace.read_bytes()
                output = trace
                if alias == "symlink":
                    output = Path(directory) / "report.md"
                    output.symlink_to(trace)
                elif alias == "hardlink":
                    output = Path(directory) / "report.md"
                    os.link(trace, output)
                result = self.run_report(trace, output)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("input SQLite trace", result.stderr)
                self.assertEqual(trace.read_bytes(), original)
                with sqlite3.connect(trace) as conn:
                    self.assertEqual(
                        conn.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone(),
                        (1,),
                    )

    def test_read_only_trace_with_uri_characters_produces_report(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace #1? cpu.sqlite"
            output = Path(directory) / "report.md"
            self.create_trace(trace)
            original = trace.read_bytes()
            trace.chmod(0o444)
            try:
                result = self.run_report(trace, output)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = output.read_text()
                self.assertIn("| cuGraphLaunch | 1 | 0.120 |", report)
                self.assertEqual(result.stdout, report + "\n")
                self.assertEqual(trace.read_bytes(), original)
            finally:
                trace.chmod(0o644)

    def test_existing_separate_report_can_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.sqlite"
            output = Path(directory) / "report.md"
            self.create_trace(trace)
            original = trace.read_bytes()
            output.write_text("Previous report\n")
            result = self.run_report(trace, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("| cuGraphLaunch | 1 | 0.120 |", output.read_text())
            self.assertEqual(trace.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
