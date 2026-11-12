#!/usr/bin/env python3
"""Freeze source, binary, checkpoint and runtime identities after all gates pass."""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

root = Path.cwd()
output = Path(os.environ["PEGAINFER_QUALIFICATION_DIR"])
lock = json.loads(Path("scripts/amd/environment.lock.json").read_text())
expected_checks = {"build", "device", "native", "host", "server", "cli", "golden-0.6B", "golden-4B", "http-0.6B", "http-4B", "fmt", "clippy"}
assert set((output / "passed-checks.txt").read_text().splitlines()) == expected_checks


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


names = set(command("git", "diff", "--name-only").splitlines())
names.update(command("git", "ls-files", "--others", "--exclude-standard").splitlines())
source_hashes = {name: digest(root / name) for name in sorted(names) if (root / name).is_file()}
patch = subprocess.check_output(["git", "diff", "--binary"])
(output / "tracked.patch").write_bytes(patch)
hip = ctypes.CDLL(str(Path(os.environ["PEGAINFER_ROCM_CORE"]) / "lib/libamdhip64.so.7"))
hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
bus = ctypes.create_string_buffer(64)
assert hip.hipDeviceGetPCIBusId(bus, len(bus), 0) == 0
assert bus.value.decode() == os.environ["PEGAINFER_EXPECTED_PCI_BUS_ID"]
assert bus.value.decode() not in lock["device_selection"]["excluded_pci_bus_ids"]
blas = ctypes.CDLL(str(Path(os.environ["PEGAINFER_ROCM_LIBRARIES"]) / "lib/librocblas.so.5"))
blas.rocblas_get_version_string.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
version = ctypes.create_string_buffer(256)
assert blas.rocblas_get_version_string(version, len(version)) == 0
models = {}
fixtures = {}
for size in ["0.6B", "4B"]:
    model = root.parent / "models" / f"Qwen3-{size}"
    models[size] = {"provenance": json.loads((model / "pegainfer-provenance.json").read_text()),
                    "files": {p.name: digest(p) for p in sorted(model.iterdir()) if p.is_file()}}
    fixture = root / "test_data" / f"qwen3-{size.lower()}-hf-golden.safetensors"
    with fixture.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        metadata = json.loads(handle.read(length)).get("__metadata__", {})
    fixtures[size] = {"sha256": digest(fixture), "metadata": metadata,
                      "limitation": "upstream fixture does not pin checkpoint revision"}
foundation = Path(os.environ["DEEPINFER_ENVIRONMENT_REPORT"])
shutil.copyfile(foundation, output / "foundation-environment.json")
report = {
    "passed": True, "recorded_at": datetime.now(timezone.utc).isoformat(),
    "profile": lock, "foundation_report": json.loads(foundation.read_text()),
    "git_head": command("git", "rev-parse", "HEAD"),
    "branch": command("git", "branch", "--show-current"),
    "dirty_source_sha256": source_hashes,
    "tracked_patch_sha256": hashlib.sha256(patch).hexdigest(),
    "binary_sha256": digest(Path(os.environ["CARGO_TARGET_DIR"]) / "release/pegainfer"),
    "cargo_lock_sha256": digest("Cargo.lock"),
    "rustc": command("rustc", "--version"), "cargo": command("cargo", "--version"),
    "hipcc": command("hipcc", "--version"), "rocblas": version.value.decode(),
    "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
    "logical_device": 0, "pci_bus_id": bus.value.decode(), "kernel": command("uname", "-r"),
    "models": models, "fixtures": fixtures,
    "http": {size: json.loads((output / f"http-{size}" / "report.json").read_text()) for size in ["0.6B", "4B"]},
    "logs": {p.name: digest(p) for p in sorted(output.glob("*.log"))},
    "limits": ["RDNA3 runtime untested", "CUDA runtime regression untested",
               "greedy TP1 eager only", "no long-duration stability or performance parity claim"],
}
assert all(item["passed"] and item["server_exit_code"] == 0 for item in report["http"].values())
(output / "qualification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"qualification": str(output / "qualification.json"), "passed": True}))
