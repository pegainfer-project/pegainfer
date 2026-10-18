# Triton AOT Integration

`pegainfer` uses Triton AOT for the Qwen3.5 GDR chunkwise prefill kernels. The
whole pipeline is gated behind the `qwen35` feature of `pegainfer-kernels` —
a default build (Qwen3 only) never runs Triton and needs no Python.

`pegainfer-kernels` can also expose selected generated CUBIN launchers through
TVM FFI under `pegainfer_kernels::triton_cubin` when the
`tvm-ffi-triton-cubin` feature is enabled. This is the DSL-facing wrapper layer;
the generated C stubs remain the low-level CUDA launch owner.

## What this covers

- Build-time generation of Triton AOT cubins for `gated_delta_rule_chunkwise_kernels.py` (with `--features qwen35`)
- Generated C wrappers linked into the normal Rust build
- Native CUDA covers basic ops (`add`, `silu_mul`, `embedding`) and decode-critical paths

## Prerequisites

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

The TVM FFI bridge is optional. Only install the TVM FFI runtime when building
with `--features tvm-ffi-triton-cubin`; in that mode `tvm-ffi-config` must be on
`PATH` and `libtvm_ffi` must be discoverable during build and runtime.

Bootstrap a repo-local Triton Python once:

```bash
uv venv .venv
uv pip install -p .venv/bin/python triton
```

Then either point the build to that interpreter explicitly:

```bash
export PEGAINFER_TRITON_PYTHON=$PWD/.venv/bin/python
```

or let `build.rs` auto-probe `.venv/bin/python` before trying `python3` / `python`.

If `nvidia-smi` is unavailable where you build, also set the target SM manually.

```bash
export PEGAINFER_CUDA_SM=120
```

`PEGAINFER_CUDA_SM` also drives the explicit Triton AOT compile target, so it is the default escape hatch when the build environment cannot query a live GPU.

### Windows

Official Triton does not ship Windows wheels. Use [`triton-windows`](https://github.com/woct0rdho/triton-windows) instead:

```powershell
uv venv .venv --python 3.12
uv pip install "triton-windows<3.7"
$env:PEGAINFER_TRITON_PYTHON = ".venv\Scripts\python.exe"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
```

Requires CUDA 12+, Python 3.9–3.12, and an NVIDIA GPU with compute capability 7.5+ (GTX 16xx or newer).

## Build

```bash
cargo build --release --features qwen35
```

Generated Triton artifacts are written to Cargo `OUT_DIR`, typically under:

```text
target/release/build/pegainfer-kernels-*/out/triton_aot/
```

## TVM FFI wrapper example

```bash
cargo run --release -p pegainfer-kernels --features tvm-ffi-triton-cubin --example triton_cubin_tvm_ffi
```

The registered names use the `pegainfer.triton_cubin.qwen35.*` prefix. Pointer
and stream arguments are packed as TVM integers or opaque pointers; scalar launch
arguments use TVM integers. The wrapper returns `()` on CUDA success and a TVM
`RuntimeError` if the underlying CUBIN launcher returns a non-success CUDA result.

## Validation

Run the focused GPU tests for the active Triton-backed paths:

```bash
cargo test --release -p pegainfer-qwen35 --features qwen35 recurrent::tests::conv1d_prefill_handoff_matches_single_prefill -- --nocapture
PEGAINFER_TEST_MODEL_PATH=/path/to/Qwen3.5-4B cargo test --release -p pegainfer-qwen35 --features qwen35 --test e2e_scheduler -- --nocapture
```

## Common failures

- `Could not find a Python interpreter with Triton installed`
  - Set `PEGAINFER_TRITON_PYTHON`, or bootstrap `.venv` with `uv`.
- `GPU detection failed`
  - Set `PEGAINFER_CUDA_SM` explicitly if `nvidia-smi` is not available during build.
- `Triton AOT generator failed`
  - Re-run the build and inspect the generator stderr printed by `build.rs`; the generator accepts an explicit `cuda:<sm>:32` target derived from `PEGAINFER_CUDA_SM`.
- `CUDA_ERROR_NO_BINARY_FOR_GPU` or similar runtime load errors
  - Rebuild on the target GPU environment; the generated Triton cubin is target-specific.
