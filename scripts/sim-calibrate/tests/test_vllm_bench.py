#!/usr/bin/env python3
"""Identity checks for the external vllm-bench executable."""

from __future__ import annotations

import stat
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from vllm_bench import inspect_binary  # noqa: E402


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

if __name__ == "__main__":
    unittest.main()
