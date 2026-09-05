#!/usr/bin/env python3
"""Generate the isolated HVK reference consumed by the production Rust ABI gate."""

from __future__ import annotations

import argparse
import ctypes
import os
import shlex
import subprocess
import sys
from pathlib import Path

from artifact_contract import (
    ContractError,
    prepare_flashinfer_source, read_json, sha256_file,
    validate_compile_metadata, validate_manifest, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = validate_manifest(args.candidate / "manifest.json", flashinfer_dir=args.flashinfer_dir)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    tools = Path(__file__).resolve().parent
    # The runner supplies a previously generated candidate. Contract validation
    # and Rust's linked-object hash check bind the reference to those exact bytes;
    # the hash does not independently establish how the candidate was generated.
    source_dir = root / "upstream-source"
    source = prepare_flashinfer_source(args.flashinfer_dir, source_dir, upstream_layout=True)
    raw = root / "upstream-aot"
    metadata_path = raw / "compile-metadata.json"
    subprocess.run([sys.executable, str(tools / "compile_sm120.py"),
                    "--upstream-layout", "--flashinfer-dir", str(source_dir),
                    "--base-flashinfer-dir", str(args.flashinfer_dir.resolve()),
                    "--aot-out", str(raw), "--metadata-out", str(metadata_path)], check=True)
    metadata = read_json(metadata_path)
    validate_compile_metadata(metadata, source)
    cuda = Path(os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda")))
    library = root / "upstream-oracle.so"
    subprocess.run([*shlex.split(os.environ.get("CC", "cc")), "-shared", "-fPIC", "-O3",
                    "-std=c11", str(tools / "upstream_adapter.c"),
                    str(raw / metadata["aot"]["object"]), metadata["aot"]["native_runtime"],
                    "-I", str(raw), "-isystem", str(cuda / "include"),
                    "-L", str(cuda / "lib64"), "-Wl,-rpath," + str(cuda / "lib64"),
                    "-lcudart", "-lcuda", "-ldl", "-lpthread", "-lstdc++", "-o", str(library)], check=True)

    import torch
    if sys.byteorder != "little" or torch.cuda.get_device_capability(0) != (12, 0):
        raise ContractError("layout reference requires little-endian SM120")
    torch.cuda.set_device(0)
    oracle_library = ctypes.CDLL(str(library))
    launch = oracle_library.upstream_in_place
    launch.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32, ctypes.c_void_p]
    launch.restype = ctypes.c_int
    workspace_bytes = torch.cuda.get_device_properties(0).multi_processor_count * 128
    workspace = torch.zeros(workspace_bytes, dtype=torch.uint8, device="cuda")

    def save(path: Path, tensor: torch.Tensor) -> None:
        # Viewing bytes also handles BF16, which NumPy has no native dtype for.
        path.write_bytes(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())

    def run(inputs: dict[str, torch.Tensor], initial: torch.Tensor, chunks: list[int]):
        state = initial.transpose(-1, -2).contiguous().cuda()  # HKV -> upstream HVK
        outputs, first_state, start = [], None, 0
        for count in chunks:
            q, k, v, alpha, beta = [inputs[name][start:start + count].cuda()
                                     for name in ("q", "k", "v", "alpha", "beta")]
            output = torch.empty_like(v)
            cu = torch.tensor([0, count], dtype=torch.int64, device="cuda")
            pointers = (ctypes.c_void_p * 9)(*[tensor.data_ptr() for tensor in
                (q, k, v, output, alpha, beta, state, workspace, cu)])
            rc = launch(count, pointers, workspace_bytes, torch.cuda.current_stream().cuda_stream)
            if rc:
                raise ContractError(f"upstream in-place launch failed: {rc}")
            torch.cuda.synchronize()
            outputs.append(output.cpu())
            if first_state is None:
                first_state = state.transpose(-1, -2).contiguous().cpu()
            start += count
        return torch.cat(outputs), state.transpose(-1, -2).contiguous().cpu(), first_state

    for count in (1, 63, 64, 65, 128):
        rng = torch.Generator().manual_seed(691 + count)
        def values(shape, low, high):
            return torch.empty(shape).uniform_(low, high, generator=rng)
        inputs = {
            "q": torch.nn.functional.normalize(values((count, 16, 128), -1, 1), dim=-1).bfloat16(),
            "k": torch.nn.functional.normalize(values((count, 16, 128), -1, 1), dim=-1).bfloat16(),
            "v": values((count, 32, 128), -0.25, 0.25).bfloat16(),
            "alpha": values((count, 32), -0.4, -0.01).exp(),
            "beta": values((count, 32), 0.1, 0.9),
        }
        initial = values((32, 128, 128), -0.01, 0.01)
        for name, chunks in [(f"t{count}", [count])] + ([("resumed64", [64, 64])] if count == 128 else []):
            case = root / name
            case.mkdir()
            for tensor_name, tensor in inputs.items():
                save(case / f"{tensor_name}.{'bf16' if tensor.dtype == torch.bfloat16 else 'f32'}", tensor)
            save(case / "initial-state.f32", initial)
            output, state, first_state = run(inputs, initial, chunks)
            if not torch.isfinite(output).all() or not torch.isfinite(state).all():
                raise ContractError(f"non-finite upstream reference: {name}")
            save(case / "output.bf16", output)
            save(case / "state.f32", state)
            if len(chunks) > 1:
                save(case / "first-state.f32", first_state)
    (root / "patched-object.sha256").write_text(candidate["artifact"]["object"]["sha256"] + "\n")
    write_json(root / "reference.json", {
        "seed": 691, "tokens": [1, 63, 64, 65, 128], "continuation": [64, 64],
        "comparison": {"atol": 0, "rtol": 0, "output": "full", "state": "full"},
        "patched": candidate, "upstream": metadata,
        "adapter_sha256": sha256_file(tools / "upstream_adapter.c"),
    })
    print(f"upstream HVK reference ready for production HKV Rust gate: {root}")


if __name__ == "__main__":
    main()
