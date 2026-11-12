#!/usr/bin/env python3
"""Production CLI rejection checks, without a running HTTP listener."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile

binary = str(Path(os.environ["CARGO_TARGET_DIR"]) / "release/pegainfer")
model = Path("../models/Qwen3-0.6B")
checks = []
for flags, expected in [
    (["--tp-size", "2"], "requires --tp-size=1"),
    (["--cuda-graph=true"], "first release is eager"),
    (["--rocm-slots", "0"], "slots must be 1..16"),
    (["--rocm-max-context", "8193"], "context exceeds"),
]:
    result = subprocess.run([binary, "--model-path", str(model), *flags],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and expected in result.stdout + result.stderr, result
    checks.append({"flags": flags, "expected": expected, "passed": True})
with tempfile.TemporaryDirectory() as directory:
    for file in model.glob("*.json"):
        shutil.copyfile(file, Path(directory, file.name))
    config = json.loads((model / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "gptq"}
    Path(directory, "config.json").write_text(json.dumps(config))
    result = subprocess.run([binary, "--model-path", directory],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "requires unquantized BF16" in result.stdout + result.stderr, result
    checks.append({"quantized_checkpoint": "rejected", "passed": True})
print(json.dumps(checks, indent=2))
