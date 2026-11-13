#!/usr/bin/env python3
"""Freeze source, binary, checkpoint and runtime identities after all gates pass."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

root = Path.cwd()
output = Path(os.environ["PEGAINFER_QUALIFICATION_DIR"])
lock = json.loads(Path("scripts/amd/environment.lock.json").read_text())
expected_checks = {"build", "device", "native", "host", "server", "cli", "wrappers", "golden-0.6B", "golden-4B", "http-0.6B", "http-4B", "fmt", "clippy"}
assert set((output / "passed-checks.txt").read_text().splitlines()) == expected_checks


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


names = set(command("git", "diff", "HEAD", "--name-only").splitlines())
names.update(command("git", "ls-files", "--others", "--exclude-standard").splitlines())
source_hashes = {name: digest(root / name) for name in sorted(names) if (root / name).is_file()}
patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"])
(output / "tracked.patch").write_bytes(patch)
models = {}
fixtures = {}
for size in ["0.6B", "4B"]:
    model = Path(os.environ.get("PEGAINFER_MODEL_ROOT", root / "models")) / f"Qwen3-{size}"
    provenance = model / "pegainfer-provenance.json"
    models[size] = {"provenance": json.loads(provenance.read_text()) if provenance.is_file() else {"revision": None, "source": "user-provided checkpoint; revision unverified"},
                    "files": {p.name: digest(p) for p in sorted(model.iterdir()) if p.is_file()}}
    fixture = root / "test_data" / f"qwen3-{size.lower()}-hf-golden.safetensors"
    with fixture.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        metadata = json.loads(handle.read(length)).get("__metadata__", {})
    fixtures[size] = {"sha256": digest(fixture), "metadata": metadata,
                      "limitation": "upstream fixture does not pin checkpoint revision"}
environment = os.environ.get("PEGAINFER_ENVIRONMENT_REPORT")
report = {
    "passed": True, "recorded_at": datetime.now(timezone.utc).isoformat(),
    "profile": lock, "environment": json.loads(Path(environment).read_text()) if environment else None,
    "git_head": command("git", "rev-parse", "HEAD"),
    "branch": command("git", "branch", "--show-current"),
    "dirty_source_sha256": source_hashes,
    "tracked_patch_sha256": hashlib.sha256(patch).hexdigest(),
    "binary_sha256": digest(Path(os.environ["CARGO_TARGET_DIR"]) / "release/pegainfer"),
    "cargo_lock_sha256": digest("Cargo.lock"),
    "rustc": command("rustc", "--version"), "cargo": command("cargo", "--version"),
    "hipcc": command(os.environ.get("PEGAINFER_HIPCC", "hipcc"), "--version"),
    "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
    "logical_device": 0, "kernel": command("uname", "-r"),
    "models": models, "fixtures": fixtures,
    "http": {size: json.loads((output / f"http-{size}" / "report.json").read_text()) for size in ["0.6B", "4B"]},
    "logs": {p.name: digest(p) for p in sorted(output.glob("*.log"))},
    "limits": ["RDNA3 runtime untested", "CUDA runtime regression untested",
               "greedy TP1 eager only", "no long-duration stability or performance parity claim"],
}
assert all(item["passed"] and item["server_exit_code"] == 0 for item in report["http"].values())
(output / "qualification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"qualification": str(output / "qualification.json"), "passed": True}))
