# TVM FFI Triton CUBIN Wrapper

> **TL;DR:** `pegainfer-kernels` has an optional `tvm-ffi-triton-cubin` bridge for the Qwen3.5 GDR solve Triton AOT CUBIN launcher, with unit coverage for wrapper registration and packed-ABI diagnostics.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - routed this task to the kernels subsystem.
  - `docs/subsystems/kernels/pegainfer-kernels-boundary.md` - confirmed DSL/kernel integration belongs at the kernels boundary rather than in model runtimes.
  - `docs/subsystems/kernels/kernel-op-reports.md` - confirmed Triton/CuTe tooling is already feature-scoped in kernel infrastructure.
  - `pegainfer-kernels/tools/triton/README.md` - described the current Triton AOT CUBIN generation and validation path.
  - `pegainfer-kernels/build.rs` - showed generated Triton AOT C stubs and wrapper symbols.
  - `pegainfer-kernels/src/ffi/qwen35.rs` and `pegainfer-kernels/src/ffi/shared.rs` - showed the existing C ABI launch symbols used by Rust model code.
  - Local `tvm-ffi` crate source - confirmed typed callbacks only cover up to 8 arguments, so Triton launchers need packed TVM FFI wrappers.
- **Relevant history**:
  - GitHub issue `#191` proposed TVM FFI as the DSL interface direction.
  - Draft PR `#202` kept TVM FFI optional/test-only; PR `#315` keeps the bridge optional behind `tvm-ffi-triton-cubin` while focusing it on Triton CUBIN launch wrappers.
- **Plan**:
  1. Add `tvm-ffi` as an optional dependency of `pegainfer-kernels` behind `tvm-ffi-triton-cubin`.
  2. Add a `triton_cubin` module that exposes a current Qwen3.5 Triton AOT CUBIN launcher as a packed TVM FFI function.
  3. Keep existing C ABI and Rust call sites available; the TVM FFI layer is an additional DSL boundary, not a production scheduler/model migration.
  4. Add a small example that registers the wrapper and prints the function contract.
  5. Validate formatting and the strongest local build/test checks available.
- **Risks / open questions**:
  - The `tvm-ffi-triton-cubin` feature means `tvm-ffi-config` and `libtvm_ffi` are build prerequisites only for the optional bridge path.
  - The wrapper depends on `qwen35` because the wrapped Triton AOT symbol is only generated with that feature.
  - The current wrapper accepts raw device pointer and stream handles as TVM integers or opaque pointers; a future DLPack/tensor-handle wrapper can sit on top once the DSL artifact contract is stable.

## Execution Log

### Step 1: Optional dependency and wrapper surface
- Added optional `tvm-ffi = "0.1.0-alpha.0"` to `pegainfer-kernels` behind `tvm-ffi-triton-cubin`.
- Made `tvm-ffi-triton-cubin` imply `qwen35`, since the current wrapper targets a Qwen3.5 Triton AOT CUBIN symbol.
- Added `pegainfer_kernels::triton_cubin`, which exposes metadata plus a packed TVM FFI callback for the generated Qwen3.5 GDR solve Triton AOT launcher.
- Kept existing CUDA C ABI symbols and model call sites unchanged.

### Step 2: Small example
- Added `pegainfer-kernels/examples/triton_cubin_tvm_ffi.rs` to register the TVM FFI global function and print the launch contract.

### Step 3: Unit test coverage
- Added wrapper unit tests for:
  - known/unknown wrapper lookup;
  - global TVM FFI registry round-trip;
  - accepted raw handle encodings (`i64`, `u64`, and opaque pointer);
  - accepted TVM `i64` scalar launch dimensions;
  - missing-argument diagnostics before CUDA launch;
  - handle and scalar type diagnostics before CUDA launch.
- Kept tests on pre-launch validation paths so they do not require valid device memory or actually launch the Triton CUBIN.

### Step 4: Review fixes
- Addressed xiaguan's requested changes on PR `#315`:
  - made `tvm-ffi` optional behind `tvm-ffi-triton-cubin` so normal `pegainfer-kernels` builds do not require `tvm-ffi-config` / `libtvm_ffi`;
  - replaced `expect_err(...)` in tests with explicit `Result` matching because `tvm_ffi::Any` does not implement `Debug`;
  - updated the example and docs to require/pass the feature.
- Addressed automated inline feedback by accepting TVM FFI packed integers as `i64` for pointer handles and scalar launch dimensions, with range checks before casting.

### Step 5: Rebase onto main
- Rebasing onto `origin/main` renamed the kernel crate from `pegainfer-kernels` to `pegainfer-kernels` and added the `qwen35` Triton feature gate.
- Adapted the TVM bridge to the renamed crate, `pegainfer_kernels` Rust import path, `pegainfer.triton_cubin.*` TVM global prefix, and `PEGAINFER_*` docs.
- Rebase validation:
  - `cargo fmt --all --check` passed.
  - `cargo metadata --no-deps --format-version 1` passed.
  - `cargo tree -p pegainfer-kernels -e normal --no-default-features --depth 1` shows no `tvm-ffi`.
  - `cargo tree -p pegainfer-kernels -e normal --features tvm-ffi-triton-cubin --depth 1` shows `tvm-ffi` only with the bridge feature enabled.
  - `cargo check --release -p pegainfer-kernels` and `PATH=/home/ziyang/gpu_memory_profiling/.venv/bin:$PATH cargo test --release -p pegainfer-kernels --features tvm-ffi-triton-cubin triton_cubin --lib -- --nocapture` both stop in the existing CUDA build before Rust checks run: FlashInfer `v0.6.12` headers require CUDA symbols not available from this local CUDA 12.8 toolchain (`cuda::fast_mod_div`, `cuda::maximum`, `cuda::minimum`).

## Debrief

- **Outcome**: Added optional TVM FFI dependency wiring plus a real Triton CUBIN wrapper MVP for the Qwen3.5 GDR solve launcher, with unit tests covering wrapper discovery, registry registration, packed handle conversion, and pre-launch diagnostics.
- **Pitfalls encountered**:
  - `apply_patch` and normal shell commands were blocked by the sandbox namespace failure, so edits were applied with scoped scripts/patches.
  - TVM FFI is now a real build prerequisite only when `tvm-ffi-triton-cubin` is enabled; hosts using that feature need `tvm-ffi-config` on `PATH`.
  - Local full kernel-crate validation is currently blocked by the pinned FlashInfer headers failing under the local CUDA 12.8 toolchain, not by the TVM FFI code.
- **Lessons learned**:
  - TVM FFI typed callbacks currently cover only up to 8 arguments, while Triton/CUDA launchers can exceed that, so the wrapper should use packed TVM FFI callbacks for launch surfaces.
- **Follow-ups**:
  - Add packed TVM FFI wrappers for the remaining generated Triton AOT launchers once the FlashInfer/CUDA toolchain gate is green.
  - Consider a higher-level DLPack/tensor-handle wrapper above the raw pointer/stream packed ABI once the DSL artifact contract is stable.
