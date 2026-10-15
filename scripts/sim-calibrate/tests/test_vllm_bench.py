#!/usr/bin/env python3
"""Tests for the external vllm-bench command driver."""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from vllm_bench import (  # noqa: E402
    VllmBenchError,
    discover_binary,
    inspect_binary,
    render_command,
    run_command,
)


def _cell(kind: str = "cold_prefill") -> dict:
    return {
        "cell_id": f"cell-{kind}",
        "name": f"cell-{kind}",
        "kind": kind,
        "input_tokens": 128,
        "output_tokens": 8,
        "new_tokens": 64,
        "prefix_tokens": 64,
        "concurrency": 2 if kind == "steady_decode" else 1,
        "warmup_repeats": 1,
        "repeats": 3,
        "steady_state_threshold": 0.9,
    }


class VllmBenchTests(unittest.TestCase):
    def _fake_binary(self, root: Path) -> Path:
        path = root / ".venv" / "bin" / "vllm-bench"
        path.parent.mkdir(parents=True)
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "if '--version' in sys.argv:\n"
            "    print('vllm-bench-test 1.2.3')\n"
            "elif '--help' in sys.argv:\n"
            "    print('Usage: vllm-bench [OPTIONS]')\n"
            "else:\n"
            "    raise SystemExit(0)\n",
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def test_records_version_and_binary_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = self._fake_binary(Path(directory))
            identity = inspect_binary(binary)
            self.assertEqual(identity.version, "vllm-bench-test 1.2.3")
            self.assertEqual(len(identity.sha256), 64)
            self.assertEqual(len(identity.help_sha256), 64)

    def test_render_rejects_mixed_without_sidecar_workload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = self._fake_binary(Path(directory))
            with self.assertRaisesRegex(VllmBenchError, "custom JSONL"):
                render_command(
                    _cell("mixed"),
                    binary=binary,
                    endpoint="http://127.0.0.1:8000/v1",
                    model="model",
                    result_dir=Path(directory) / "raw",
                )

if __name__ == "__main__":
    unittest.main()
