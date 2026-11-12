#!/usr/bin/env python3
"""Verify the project lock after the shared foundation verifier has passed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

lock = json.loads(Path(__file__).with_name("environment.lock.json").read_text())
assert os.environ.get("DEEPINFER_IMAGE_ID") == lock["image_id"], "project image lock mismatch"
assert os.environ.get("DEEPINFER_ROCM_RELEASE") == lock["rocm_release"], "ROCm release mismatch"
for file, expected in lock["runtime_files"].items():
    actual_path = file.replace("/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core", os.environ["PEGAINFER_ROCM_CORE"])
    actual_path = actual_path.replace("/opt/venv/lib/python3.13/site-packages/_rocm_sdk_libraries", os.environ["PEGAINFER_ROCM_LIBRARIES"])
    actual_path = actual_path.replace("/opt/rocm/core-10.1/include", os.environ.get("PEGAINFER_ROCM_DEV_INCLUDE", "/opt/rocm/core-10.1/include"))
    with open(actual_path, "rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    assert actual == expected, f"project SDK/runtime mismatch: {file}"
compiler = subprocess.check_output([os.environ.get("PEGAINFER_HIPCC", "hipcc"), "--version"], text=True)
assert lock["hip_build"] in compiler and lock["clang_revision"] in compiler, "HIP compiler identity mismatch"
print(json.dumps({"project_profile": lock["profile"], "passed": True,
                  "build_arch": os.environ["PEGAINFER_ROCM_ARCH"]}), flush=True)
