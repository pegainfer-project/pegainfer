#!/usr/bin/env python3
"""Focused persistence and lifecycle tests for the vllm-bench runner."""

from __future__ import annotations

import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from artifacts import ArtifactStore  # noqa: E402
from vllm_bench_runner import run_cell_variant  # noqa: E402


def _manifest() -> tuple[dict, dict]:
    cell = {
        "cell_id": "a" * 64,
        "name": "cold-smoke",
        "kind": "cold_prefill",
        "split": "train",
        "input_tokens": 4,
        "output_tokens": 3,
        "concurrency": 1,
        "warmup_repeats": 0,
        "repeats": 1,
    }
    manifest = {
        "campaign_id": "b" * 64,
        "target": {
            "endpoint": "http://127.0.0.1:8000/v1",
            "model": "test-model",
        },
        "client": {"seed": 7},
        "workload": {"cells": [cell]},
    }
    return manifest, cell


class VllmBenchRunnerTests(unittest.TestCase):
    def _fake_binary(self, root: Path, *, fail: bool = False) -> Path:
        path = root / "vllm-bench"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "import pathlib\n"
            "import sys\n"
            "log = pathlib.Path(sys.argv[0]).with_name('invocations.log')\n"
            "with log.open('a', encoding='utf-8') as handle:\n"
            "    handle.write(' '.join(sys.argv[1:]) + '\\n')\n"
            "if '--version' in sys.argv:\n"
            "    print('vllm-bench-test 1.0')\n"
            "    raise SystemExit(0)\n"
            "if '--help' in sys.argv:\n"
            "    print('help')\n"
            "    raise SystemExit(0)\n"
            f"if {fail!r}:\n"
            "    print('benchmark failed', file=sys.stderr)\n"
            "    raise SystemExit(3)\n"
            "def value(flag):\n"
            "    return sys.argv[sys.argv.index(flag) + 1]\n"
            "result = pathlib.Path(value('--result-dir')) / value('--result-filename')\n"
            "result.parent.mkdir(parents=True, exist_ok=True)\n"
            "result.write_text(json.dumps({\n"
            "    'backend': 'vllm', 'completed': 1, 'failed': 0,\n"
            "    'input_lens': [4], 'output_lens': [3], 'ttfts': [0.01],\n"
            "    'latencies': [0.02], 'itls': [[0.005, 0.005, 0.005]],\n"
            "    'errors': [None]\n"
            "}), encoding='utf-8')\n"
            "print('benchmark stdout')\n"
            "print('benchmark stderr', file=sys.stderr)\n",
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def test_success_persists_raw_sidecar_and_converted_result(self) -> None:
        manifest, cell = _manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = self._fake_binary(root)
            store = ArtifactStore(root / "artifacts", manifest["campaign_id"])
            outcome = run_cell_variant(manifest, cell, "target", store, binary=binary)

            self.assertEqual(outcome["status"], "sealed")
            self.assertEqual(outcome["attempt"], 1)
            self.assertEqual(set(outcome["artifacts"]), {"stdout", "stderr", "sidecar", "converted", "command", "vllm_bench_result"})
            for item in outcome["artifacts"].values():
                self.assertTrue((root / "artifacts" / item["path"]).is_file())
            envelope_path = root / "artifacts" / outcome["entry"]["result_path"]
            envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
            self.assertEqual(envelope["result"]["status"], "succeeded")
            sidecar = json.loads((root / "artifacts" / outcome["artifacts"]["sidecar"]["path"]).read_text(encoding="utf-8"))
            self.assertEqual(sidecar["measured_requests"], 1)
            converted = json.loads((root / "artifacts" / outcome["artifacts"]["converted"]["path"]).read_text(encoding="utf-8"))
            self.assertEqual(converted["measurements"][0]["request_id"], f"{cell['cell_id']}:measured:0:0")

    def test_resume_skips_a_sealed_variant_without_a_second_attempt(self) -> None:
        manifest, cell = _manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = self._fake_binary(root)
            store = ArtifactStore(root / "artifacts", manifest["campaign_id"])
            first = run_cell_variant(manifest, cell, "target", store, binary=binary)
            second = run_cell_variant(manifest, cell, "target", store, binary=binary, resume=True)

            self.assertEqual(first["status"], "sealed")
            self.assertTrue(second["skipped"])
            self.assertEqual(len(second["entry"]["attempts"]), 1)
            self.assertEqual(store.variant_status(cell["cell_id"], "target"), "sealed")


if __name__ == "__main__":
    unittest.main()
