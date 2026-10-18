# Typed Forward Pipeline Macro

> **TL;DR:** Build a reusable typed tensor pipeline macro in `pegainfer-kernels` so model crates can express common `typed_ops` forward chains without model-specific wrapper macros.
>
> **Last touched:** 2026-05

## Preparation

- **Read**:
  - `docs/index.md` - routed this task to the kernels subsystem with Kimi-K2 as the first consumer.
  - `docs/models/kimi-k2/bringup-history.md` - showed the Kimi decode/prefill hot paths and the CUDA Graph boundary constraints that the macro must preserve.
  - `docs/subsystems/kernels/pegainfer-kernels-boundary.md` - confirmed reusable kernel/runtime helpers belong in `pegainfer-kernels`, while model-specific execution remains in model crates.
  - `pegainfer-kernels/src/forward_pass.rs` - found the existing `typed_forward_pass!` DSL covering a subset of typed ops.
  - `pegainfer-kernels/src/typed_ops.rs` - confirmed the typed op surface available to the macro.
  - `pegainfer-kimi-k2/src/runner/worker.rs` - identified repeated typed op chains in MLA, dense MLP, and MoE paths.
- **Relevant history**:
  - `docs/models/kimi-k2/bringup-history.md` records that decode graph capture requires stable pointers and no decode-step allocation; macro expansion must not hide allocations inside decode paths.
- **Plan**:
  1. Extend `pegainfer-kernels/src/forward_pass.rs` into a generic typed pipeline macro with explicit `ctx`, `eps`, optional `seq_len`, and configurable GEMM mode.
  2. Support reusable statements for typed tensor allocation, `rms_norm`, `gemm`, `silu_mul`, `add`, `swap`, bf16/f32 conversion, and escape hatches for arbitrary calls that return `Result`.
  3. Replace repeated typed op chains in `pegainfer-kimi-k2/src/runner/worker.rs` with the generic macro while leaving Kimi-specific kernels explicit.
  4. Run formatting and the narrowest compile checks that exercise the touched crates.
- **Risks / open questions**:
  - Macro grammar must remain readable at call sites; too much DSL would make CUDA graph and borrow ordering harder to audit.
  - Existing uncommitted changes touch the same files, so edits must preserve current worktree content.

## Execution Log

### Step 1: Replace the local forward macro with a reusable pipeline

- Reworked `pegainfer-kernels/src/forward_pass.rs` into `typed_pipeline!`.
- Added pipeline statements for tensor allocation, typed GEMM/RMSNorm/add/SiLU, bf16/f32 conversion, swaps, and explicit `try`/`call` escapes for model kernels.
- Kept decode graph paths allocation-free by requiring `gemm = prefill` for any `tensor` allocation statement.

### Step 2: Make typed ops require typed weights

- Removed the untyped weight adapter layer from `pegainfer-kernels/src/typed_ops.rs`.
- `gemm_*_into` now accepts `GpuWeight<OUT, IN>` only.
- RMSNorm helpers now accept `NormWeight<DIM>` only.
- Runtime-row vocab weights use `GpuTensor<DIM>` so embedding/lm-head keep hidden width static while vocab rows remain runtime.
- Kimi router and MLA wrappers now receive typed weight owners where the dimensions are fixed.

### Step 3: Move Kimi forward paths onto the generic pipeline

- Replaced repeated op chains in `pegainfer-kimi-k2/src/runner/worker/forward.rs` and `state.rs` with `typed_pipeline!`.
- Converted fixed Kimi decode/prefill weights to `GpuWeight`/`NormWeight` at load/package boundaries.
- Converted token embedding and lm-head cache entries to `GpuTensor<KIMI_K2_HIDDEN>`.
- Tightened MLA wrappers with seq_len/cache metadata validation and updated kernel-report measurement paths to call typed MLA APIs.
- Removed the unused `typed_forward.rs` experiment after the real worker path was using typed weights and typed scratch directly.

### Step 4: Verify

- `PEGAINFER_CUDA_SM=90a PEGAINFER_TRITON_PYTHON=$LOCAL_PEGAINFER_DIR/.venv/bin/python3 cargo check --release -p pegainfer-kimi-k2 --features kimi-k2 --tests` passed.
- `PEGAINFER_CUDA_SM=90a PEGAINFER_TRITON_PYTHON=$LOCAL_PEGAINFER_DIR/.venv/bin/python3 cargo check --release -p pegainfer-kimi-k2 --lib` passed after gating runtime/weights exports behind `kimi-k2`.
- `PEGAINFER_CUDA_SM=90a PEGAINFER_TRITON_PYTHON=$LOCAL_PEGAINFER_DIR/.venv/bin/python3 cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bins` passed after migrating report harness inputs to typed Kimi kernel wrappers.
- Final audit replaced the remaining worker load `debug_assert_eq!` rank/report checks with release `ensure!` errors and confirmed the Kimi worker/MLA typed path has no `AsGemmWeight`, `AsNormWeight`, `typed_forward`, or `TypedDecodeScratch` remnants.

## Debrief

- **Outcome**: Kimi fixed weights and decode scratch now participate in the typed tensor path instead of going through untyped adapters or detached example code.
- **Pitfalls encountered**:
  - Allowing `seq_len` in the default macro header made it too easy to allocate inside graphsafe call sites, so allocation is now tied to `gemm = prefill`.
  - The default Kimi crate build exposed an unclear feature boundary: runtime/weights modules depended on Kimi kernels even when `kimi-k2` was disabled.
- **Lessons learned**:
  - Reusable typed macros should keep model kernels explicit and be strict about allocation sites.
  - Model crates with feature-gated CUDA kernels should gate runtime exports, not compile half a runtime with missing kernel symbols.
