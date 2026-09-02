# Qwen3.5-4B Model Crate

**Created**: 2026-05-05
**TL;DR**: `pegainfer-qwen35` now owns Qwen3.5 config, weights, prefill/decode/unified forward, recurrent state, scheduler, recurrent op wrappers, scheduler integration tests, and Qwen3.5 op benches. The whole crate is behind the `qwen35` feature (`--features qwen35` on `pegainfer-server`) because its GDR prefill kernels are Triton AOT-generated — this keeps the default Qwen3 build Python-free. Root `pegainfer` loads Qwen3.5 through `pegainfer_qwen35::start_engine(...)` / `Engine` (`LaunchedEngine::Stepped`); root no longer exposes `pegainfer::model::Qwen35Model` or `pegainfer::scheduler_qwen35`. The original exact-text e2e/regen tests described in this migration record were later retired by the HF logits gate in `docs/models/qwen35/accuracy.md`.
**Last touched**: 2026-08

## Feature gate (2026-06)

Qwen3.5 is the only model line whose kernels need Python at build time (Triton
AOT for the GDR chunkwise prefill). To make the stock build pure Rust + CUDA,
the crate is feature-gated end to end:

- `pegainfer-kernels/qwen35` gates the Triton AOT build step and the GDR
  chunk FFI declarations — without it, `build.rs` never probes for Python.
- `pegainfer-qwen35` compiles to an empty crate without its `qwen35`
  feature (crate-root `#![cfg]`), so `cargo test --workspace --lib` stays
  Python-free; its tests/benches carry `required-features` and fail with an
  actionable message instead of a link error.
- `pegainfer-server` defaults to `qwen3` only; serve Qwen3.5 with
  `cargo run --release --features qwen35 -- --model-path models/Qwen3.5-4B`.

The unused Triton HD256 prefill kernel (replaced by the native paged
`batch_prefill_paged_cuda_hd256`) was deleted in the same change.

## Preparation

- **Read**:
  - `docs/index.md` - identified the existing core split, Qwen3 model crate split, and Qwen3.5 accuracy/optimization docs.
  - `docs/models/qwen3/model-crate.md` - Qwen3 already owns its scheduler, executor/runtime API, tests, benches, and root-facing `EngineHandle` entry.
  - `docs/models/qwen35/accuracy.md` - at the time of this migration, Qwen3.5 e2e tests were regression guards against `test_data/Qwen3.5-4B.json`; current accuracy coverage is the HF logits gate recorded there.
  - `docs/models/qwen35/optimization.md` - Qwen3.5 should keep its hybrid linear/full-attention scheduler/state architecture.
  - GitHub issue #79 - acceptance criteria require `pegainfer-qwen35`, removal of root `pegainfer::model::Qwen35Model` and `pegainfer::scheduler_qwen35`, generic root `bench_serving`, and CUDA validation.
  - `Cargo.toml`, `src/lib.rs`, `src/main.rs`, `src/ops.rs`, `src/scheduler.rs`, `src/model/qwen35.rs`, and `pegainfer-qwen3/src/lib.rs` - mapped the current root Qwen3.5 surface and the Qwen3 crate interface to copy.
- **Relevant history**:
  - `docs/models/qwen3/model-crate.md` - root should load model crates through `EngineHandle`; model-owned execution details should move behind crate-local modules.
- **Plan**:
  1. Add `pegainfer-qwen35` to the workspace with dependencies mirroring the Qwen3 crate plus the root dependencies Qwen3.5 currently uses.
  2. Move `src/model/qwen35.rs`, `src/model/qwen35/*`, `src/scheduler_qwen35.rs`, and Qwen3.5 recurrent op wrappers into the new crate, keeping CUDA/Triton kernel sources and FFI in `pegainfer-kernels`.
  3. Rewrite imports so the new crate depends on `pegainfer-core` and `pegainfer-kernels`, not on root `pegainfer`.
  4. Expose `start_engine` and a deliberate `runtime` module from `pegainfer-qwen35`.
  5. Update root `main.rs` and `src/bin/bench_serving.rs` to call `pegainfer_qwen35::start_engine`.
  6. Move Qwen3.5 e2e tests and regen test into the model crate; adjust model/test-data paths after the move.
  7. Remove root Qwen3.5 modules and compatibility exports, then audit root with `rg`.
  8. Verify with `cargo fmt --all --check`, `cargo metadata --no-deps --format-version 1`, and the CUDA-capable build/test commands available on this machine.
- **Risks / open questions**:
  - Some root operator tests cover Qwen3.5 recurrent wrappers; they may need to move with the wrappers or be split so root no longer imports model-specific scratch types.
  - Accuracy docs reference historical `qwen35_dump_*` and `tools/accuracy/*` files that are not present in the current tree; this migration can document the current test locations but cannot move absent tools.

## Execution Log

### Step 1: Add model crate and move Qwen3.5 runtime
- Added `pegainfer-qwen35` to the workspace and root dependencies.
- Moved Qwen3.5-owned runtime files out of root:
  - `src/model/qwen35.rs`
  - `src/model/qwen35/*`
  - `src/scheduler_qwen35.rs`
  - `src/ops/recurrent.rs`
- The new crate exposes:
  - `start_engine(model_path, EngineLoadOptions, max_batch, max_prefill_tokens) -> Result<EngineHandle>`
  - `runtime::{Qwen35Model, MAX_BATCH}` for model-local tests/debugging
  - `runtime_ops` for Qwen3.5-local operator benches.

### Step 2: Move tests and benches
- Moved root Qwen3.5 tests to the model crate at the time:
  - `pegainfer-qwen35/tests/e2e.rs`
  - `pegainfer-qwen35/tests/e2e_scheduler.rs`
  - `pegainfer-qwen35/tests/regen_test_data.rs`
- The exact-text `e2e.rs` and `regen_test_data.rs` were later removed by the Qwen3.5 HF logits gate work; `e2e_scheduler.rs` remains as request-flow coverage.
- Moved Qwen3.5-specific op benches to `pegainfer-qwen35/benches/qwen35_ops.rs`.
- Moved the `conv1d_prefill_handoff_matches_single_prefill` operator test into `pegainfer-qwen35/src/recurrent.rs`, next to the wrapper it validates.
- Removed Qwen3.5-specific GEMV shapes from the root generic `ops_bench`; the model-specific benches now live with Qwen3.5.

### Step 3: Remove root Qwen3.5 compatibility surface
- Removed root exports/modules:
  - `pub mod model`
  - `pub mod scheduler_qwen35`
  - `src/model.rs`
  - `src/ffi.rs`
  - `src/kv_pool.rs`
- Root `main.rs` now calls `pegainfer_qwen35::start_engine(...)` for Qwen3.5.
- Root `bench_serving` now calls `pegainfer_qwen35::start_engine(...)` and still benchmarks via generic `EngineHandle`.
- The Qwen3.5 engine entry honors a single `EngineLoadOptions.device_ordinals` value and rejects multi-device input, matching the current single-GPU implementation instead of silently ignoring the option.
- `rg` confirms there are no root references to `pegainfer::model::Qwen35Model`, `pegainfer::scheduler_qwen35`, or `src/model/qwen35`.

### Step 4: Validation
- Passed:
  - `cargo metadata --no-deps --format-version 1`
  - `cargo fmt --all --check`
  - `PEGAINFER_CUDA_SM=120 cargo check --release --workspace --all-targets`
  - `PEGAINFER_CUDA_SM=120 cargo clippy --release --workspace --all-targets -- -D warnings`
  - `PEGAINFER_CUDA_SM=120 cargo build --release`
  - `PEGAINFER_CUDA_SM=120 cargo test --release -p pegainfer-qwen35 recurrent::tests::conv1d_prefill_handoff_matches_single_prefill -- --nocapture`
  - `PEGAINFER_CUDA_SM=120 cargo run --release --bin bench_serving -- --model-path $LOCAL_PEGAINFER_DIR/models/Qwen3.5-4B request --prompt-len 1 --output-len 1 --warmup 0 --iters 1`
- Initial Qwen3.5 e2e failure:
  - `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=$LOCAL_PEGAINFER_DIR/models/Qwen3.5-4B cargo test --release -p pegainfer-qwen35 --test e2e -- --nocapture`
  - `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=$LOCAL_PEGAINFER_DIR/models/Qwen3.5-4B cargo test --release -p pegainfer-qwen35 --test e2e_scheduler -- --nocapture`
  - Both initially produced all-case gibberish-output mismatches.
- Control run:
  - A temporary old-HEAD worktree at `$RESULT_ROOT/pegainfer-head` ran `PEGAINFER_CUDA_SM=120 PEGAINFER_TRITON_PYTHON=$LOCAL_PEGAINFER_DIR/.venv/bin/python PEGAINFER_TEST_MODEL_PATH=$LOCAL_PEGAINFER_DIR/models/Qwen3.5-4B CARGO_TARGET_DIR=$RESULT_ROOT/pegainfer-head-target cargo test --release --test e2e_qwen35 -- --nocapture`.
  - Old HEAD failed the same way on all 10 Qwen3.5 cases, so the e2e mismatch predated this crate split.
- Follow-up fix:
  - `docs/lessons/exact-match-gate-thread-cublas.md` identified the first gibberish commit as `6a5b826`, fixed Qwen3.5 scheduler thread CUDA/cuBLAS binding, kept greedy sampling on FlashInfer top1, and refreshed the exact Qwen3.5 golden for the default engine shape.
  - After that fix, both Qwen3.5 e2e commands above pass.

## Debrief

- **Outcome**: Qwen3.5 is now an independent model crate with the same root-facing engine style as Qwen3-4B. Root retains model detection/frontend/bench orchestration, but not Qwen3.5 model internals. The follow-up e2e corruption fix restored the then-current exact-text e2e and scheduler e2e; the exact-text gate was later retired in favor of the HF logits gate.
- **Pitfalls encountered**:
  - The first e2e run used a relative `PEGAINFER_TEST_MODEL_PATH`; package tests execute with a crate-oriented working directory, so absolute model paths are safer for crate-local tests.
  - Qwen3.5 e2e initially looked like a crate-split regression, but git history showed the corruption started earlier when cuBLAS handles became thread-local without equivalent Qwen3.5 scheduler thread binding.
  - Moving recurrent wrappers out of root exposed stale root compatibility re-exports (`src/ffi.rs`, `src/kv_pool.rs`, and root Qwen3.5 ops bench shapes), which were removed.
- **Lessons learned**:
  - Model-local benches need a deliberate public surface. `runtime_ops` is intentionally narrow and only exposes the Qwen3.5 operator wrappers needed by Qwen3.5 benches.
  - Qwen3.5 test docs should use absolute `PEGAINFER_TEST_MODEL_PATH` examples when run from the workspace, because package test working directories can make relative paths misleading.
