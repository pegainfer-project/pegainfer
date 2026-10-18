# Qwen3 Kernels Crate Extraction

**Created**: 2026-05-03
**Status**: complete
**TL;DR**: Phase 1 now extracts the Qwen3-4B dense full-attention kernel surface into `crates/pegainfer-kernels`, with a compact kernel index so future LLM sessions can jump from model DAG nodes to Rust wrappers, FFI symbols, CUDA/Triton sources, and shape constraints. `KvPool`, `PagePool`, and `SamplingParams` stay in the root runtime. Local metadata/format checks pass; GPU release build, release test-target compilation, release clippy, Qwen3-4B e2e, and `bench_serving snapshot` pass.

## Preparation

- **Read**:
  - `docs/index.md` - confirmed the relevant architecture, kernel, TP, benchmarking, and Qwen3 history docs.
  - `docs/subsystems/kernels/pegainfer-kernels-boundary.md` - recorded the per-model engine direction, but its near-term ordering needs to be corrected from ledger-first to crate-first.
  - `docs/models/qwen3/tp-design.md` - confirmed Qwen3-4B TP constraints and runtime hazards around per-thread CUDA/cuBLAS state.
  - `src/model/qwen3/*`, `src/ops/*`, `src/ffi.rs`, `src/tensor.rs`, `src/kv_pool.rs`, `src/page_pool.rs`, and `build.rs` - mapped the current Qwen3-4B kernel calls, tensor/runtime dependencies, paged KV metadata, and CUDA/Triton build pipeline.
- **Relevant history**:
  - `docs/models/qwen3/tp-design.md` shows that Qwen3 execution is already rank-local and step-oriented, so the kernel crate must not hide device binding or TP collective points.
- **Plan**:
  1. Convert the repository into a Cargo workspace while keeping the root `pegainfer` package as the server/control-plane crate.
  2. Create `crates/pegainfer-kernels` with the Qwen3-4B kernel surface: kernel ABI tensor helpers, Qwen3-used `ops`, FFI declarations, CUDA/Triton build support, and Qwen3 paged-attention layout metadata helpers.
  3. Move Qwen3 call sites to import `pegainfer_kernels::{ops, tensor}` and remove direct Qwen3 dependence on root-local `ops`, `ffi`, and `tensor` modules.
  4. Preserve repository build health. If Qwen3.5 still requires symbols from the old combined CUDA library, either keep those symbols as compatibility exports in the kernels crate or explicitly document and gate any temporary Qwen3-only limitation before making code changes.
  5. Add a kernel index for LLM navigation under the new crate:
     - `KERNELS.md`: short human/LLM routing table from `qwen3_4b::<phase>::<op>` to Rust wrapper, FFI symbol, source file, backend, shape/layout constraints, and status.
     - Machine-readable model DAG metadata should wait for the Qwen3-4B model crate, where it can be generated or validated from model code instead of hand-maintained in the generic kernels crate.
  6. Update `docs/subsystems/kernels/pegainfer-kernels-boundary.md` and `docs/index.md` so the recorded next step is crate-first, with ledger/trace/simulator as metadata products of the crate boundary.
  7. Verify with `cargo test --release` or, if the local environment blocks full release tests, at least `cargo check --release` and report the exact blocker.
- **Risks / open questions**:
  - A strict Qwen3-only CUDA extraction can conflict with the current default binary because Qwen3.5 still compiles in the same root crate and references some shared FFI symbols. The safest implementation may need to move the link/build owner to `pegainfer-kernels` while only stabilizing and indexing the Qwen3 API first.
  - `kv_pool` and `page_pool` sit between model state and kernel metadata. For Phase 1, only the kernel-facing layout/descriptor pieces should move if needed; scheduler-owned allocation policy should remain in the root crate unless compilation forces a narrower split.
  - Build-script path handling is fragile when moving kernel source into `crates/pegainfer-kernels/`. The plan should prefer one build owner and avoid compiling the same C symbols in both root and dependency crates.

## Execution Log

### Step 1: Create kernels crate and move build ownership
- Converted the repository into a Cargo workspace with `crates/pegainfer-kernels`.
- Added `pegainfer-kernels` as a root dependency.
- Moved CUDA source from root `csrc/` to `crates/pegainfer-kernels/csrc/`.
- Moved Triton AOT files from root `tools/triton/` to `crates/pegainfer-kernels/tools/triton/`.
- Moved the FlashInfer submodule path from `third_party/flashinfer` to `crates/pegainfer-kernels/third_party/flashinfer`.
- Replaced the root `build.rs` with an intentionally empty build script; `crates/pegainfer-kernels/build.rs` now owns CUDA/Triton compilation.

- Moved kernel-owned ABI and operator code into `crates/pegainfer-kernels/src/`: `ffi`, tensor helpers, paged-KV geometry metadata, and the Qwen3-used `ops` modules.
- Kept `KvPool`, `PagePool`, and `SamplingParams` in the root crate because they are runtime allocation/policy state, not kernels.
- Replaced root `src/ffi.rs` and `src/tensor.rs` with compatibility re-exports.
- Replaced root `src/ops.rs` with re-exports from `pegainfer-kernels` plus thin root adapters for sampling, paged prefill planning, paged attention layout conversion, and the remaining Qwen3.5 recurrent wrapper.
- Removed duplicate root `src/ops/{attention,elementwise,embedding,linear,norm,sampling}.rs`.
- Kept `src/ops/recurrent.rs` in root for now because it depends on Qwen3.5's model-local `GdrChunkwiseScratch35`; moving that would expand Phase 1 beyond Qwen3-4B.

### Step 3: Add kernel index for LLM navigation
- Added `crates/pegainfer-kernels/KERNELS.md`.
- The index maps each Qwen3-4B op ID to phase, Rust wrapper, FFI symbol, source file, backend, and shape/layout notes.
- Removed the initial `kernel_manifest/qwen3_4b.toml` idea from the kernels crate. A hand-maintained machine-readable manifest in the generic kernel crate would drift; the right place is the future Qwen3-4B model crate, where the manifest can describe the model DAG and be generated or checked against code.

### Step 4: Documentation updates
- Updated `CLAUDE.md`, `README.md`, and `docs/playbooks/developer-onboarding.md` to point CUDA/Triton paths at `crates/pegainfer-kernels/`.
- Updated `docs/subsystems/kernels/pegainfer-kernels-boundary.md` to record crate-first ordering before ledger/simulator work.

### Step 5: Verification
- `cargo metadata --no-deps --format-version 1` succeeded and showed both workspace packages: root `pegainfer` and `pegainfer-kernels`.
- `cargo fmt --all` applied formatting, then `cargo fmt --all --check` passed.
- `PEGAINFER_CUDA_SM=120 cargo check --release` reached the `pegainfer-kernels` build script and failed at `nvcc` execution because this machine has no `nvcc`.

### Step 6: GPU release compile
- Avoided overwriting `<validation-checkout>` because that validation checkout has unrelated uncommitted work.
- Synced the local working tree to `<validation-worktree>` with `rsync`, excluding `.git/`, `target/`, `.venv/`, and `models/`.
- Copied the existing validation FlashInfer submodule contents from `<validation-checkout>/third_party/flashinfer` into `crates/pegainfer-kernels/third_party/flashinfer` inside the build directory.
- `PEGAINFER_CUDA_SM=120 cargo build --release` passed on the CUDA validation host. First pass exposed two Rust warnings from this split (`SamplingParams::is_greedy` unused and root `PrefillPagedPlan` visibility too wide); both were cleaned up.
- Re-synced and reran `PEGAINFER_CUDA_SM=120 cargo build --release`; it passed in 14.16s with only build-script informational warnings.
- `PEGAINFER_CUDA_SM=120 cargo test --release --no-run` passed in 12.28s and compiled all unit, binary, e2e, paged-attention, and regen test targets.

### Step 7: GPU e2e and serving benchmark
- Ran Qwen3-4B e2e on the same validation build directory:
  - `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=<model-path> cargo test --release --test e2e -- --nocapture`
  - Result: pass, 1 test passed in 9.36s.
  - Covered greedy golden outputs, multi-request generation, and consumer-drop scheduler survival.
- Ran the standard in-process serving snapshot:
  - `RUST_LOG=warn PEGAINFER_CUDA_SM=120 cargo run --release --bin bench_serving -- --model-path <model-path> snapshot`
  - Result: pass.
  - RTX 5090 Qwen3-4B snapshot:
    - `prefill_heavy (10000,1)`: TTFT p50 `501.93ms`, p99 `503.75ms`.
    - `decode_heavy (1024,256)`: TPOT p50 `7.40ms`, p99 `7.46ms`.
  - Snapshot was written on the validation build dir at `bench_snapshots/rtx-5090/qwen3-4b.json`.
  - Pulled the snapshot back into the local repo as `bench_snapshots/rtx-5090/qwen3-4b.json` so it can be committed with the crate split.
  - The isolated rsync build directory intentionally excludes `.git/`, so the generated `commit` field was `unknown`; after pulling it back, set it to the current local `HEAD` short hash `3448f87`.
- Checked `<qwen35-model-path>`; it is not present on this CUDA validation host, so no Qwen3.5 e2e was run.

### Step 8: GPU clippy and final local checks
- Ran local `cargo fmt --all --check`: pass.
- Ran local `cargo metadata --no-deps --format-version 1`: pass.
- Synced the current working tree to `<validation-worktree>`.
- Ran `PEGAINFER_CUDA_SM=120 cargo clippy --release --all-targets -- -D warnings` on the CUDA validation host: pass in 1m42s.

### Unexpected
- Local `cargo check --release` reached `pegainfer-kernels` build script but failed because this machine does not have `nvcc`; the user will provide a GPU build machine for compilation.
- A second `cargo check --release -p pegainfer-kernels --lib` without `PEGAINFER_CUDA_SM` failed earlier at GPU SM detection, which is expected on this local machine without `nvidia-smi`.
- The validation checkout was dirty, so verification used a separate validation build directory instead of modifying that checkout.
- The validation build directory does not include `.git/`, so `bench_serving snapshot` reports `commit: unknown`.

## Debrief

- **Outcome**: Implemented and validated the crate-first Phase 1 split. Kernel source, Triton source, FlashInfer submodule ownership, CUDA/Triton build script, FFI, kernel ABI tensor helpers, paged-KV layout metadata, and Qwen3-used Rust ops now live under `crates/pegainfer-kernels`. Root `pegainfer` keeps server/model code, `KvPool`, `PagePool`, `SamplingParams`, and thin compatibility adapters. The split passes local format/metadata checks, GPU release build/test-target compilation, release clippy, Qwen3-4B e2e, and the standard Qwen3-4B `bench_serving snapshot`.
- **Pitfalls encountered**:
  - Root `src/ops/recurrent.rs` cannot be moved cleanly in this pass because it takes Qwen3.5's `GdrChunkwiseScratch35` type. Moving it would pull hybrid-model scratch ownership into the kernels crate, which is outside the Qwen3-4B Phase 1 scope.
  - Initially moved `KvPool`, `PagePool`, and `SamplingParams` into the kernels crate. That was too broad; those belong to runtime policy and have been moved back to root.
  - Local compile verification is blocked by missing `nvcc`, so GPU compile verification should happen on a CUDA build host.
- **Lessons learned**:
  - The kernel crate should own source and build artifacts physically, not only re-export copied Rust wrappers. Keeping `csrc/`, `tools/triton/`, and `third_party/flashinfer` in root creates exactly the duplicate context we are trying to remove.
  - The human/LLM routing index belongs beside the kernels crate because it helps edit reusable kernels. Machine-readable model DAG manifests should not live there unless they are generated or validated; they belong with the model crate that owns the DAG.
- **Follow-ups**:
  - Phase 2 can extract the Qwen3 model crate on top of `pegainfer-kernels`.
  - In the Qwen3 model crate, define the model-owned kernel DAG and decide whether any TOML/JSON manifest is generated from Rust code, validated against wrappers, or avoided entirely in favor of trace IDs emitted directly from the executor.
  - Run Qwen3.5 e2e separately on a box with `<qwen35-model-path>` if later changes touch the compatibility kernels or recurrent wrappers.
