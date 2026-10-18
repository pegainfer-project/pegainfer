# Kimi-K2 Source Layout

> **TL;DR:** Kimi-K2 source files over 1k lines were split by responsibility; the largest Rust file under `pegainfer-kimi-k2/src` is now `layers/attention.rs` at 950 lines.
>
> **Last touched:** 2026-05

## Preparation

- **Read**:
  - `docs/index.md` - routed the cleanup to the Kimi-K2 model docs.
  - `docs/models/kimi-k2/bringup-history.md` - confirmed `worker.rs` owns decode arena, forward, routing, and sampling paths.
  - `pegainfer-kimi-k2/src/layers/attention.rs` - found tensor-view wrappers and validation helpers mixed into the attention header API.
  - `pegainfer-kimi-k2/src/layers/experts.rs` - found tests embedded at the end of the expert header API.
  - `pegainfer-kimi-k2/src/runner/worker.rs` - found rank worker ownership, state command handling, arena/cache logic, forward kernels, load helpers, runtime helpers, and tests in one file.
- **Relevant history**:
  - `docs/models/kimi-k2/bringup-history.md` records CUDA Graph and decode arena constraints; splits must preserve pointer-stable decode behavior and not change allocation sites.
- **Plan**:
  1. List Rust files under `pegainfer-kimi-k2/src` over 1k lines.
  2. Split low-risk header/API files first: attention tensor wrappers/validation helpers and expert tests.
  3. Split `runner/worker.rs` by runtime responsibility: state command handling, cache/arena ownership, forward kernels, load helpers, and runtime helpers.
  4. Run formatting and Kimi feature compile checks.

## Execution Log

### Step 1: List oversized files

- Ran `find pegainfer-kimi-k2/src -name '*.rs' -type f -print0 | xargs -0 wc -l`.
- Files over 1k lines before splitting:
  - `pegainfer-kimi-k2/src/runner/worker.rs` - 2799 lines.
  - `pegainfer-kimi-k2/src/layers/attention.rs` - 1146 lines.
  - `pegainfer-kimi-k2/src/layers/experts.rs` - 1008 lines.

### Step 2: Split header/API modules

- Moved attention tensor view wrappers to `pegainfer-kimi-k2/src/layers/attention/tensors.rs`.
- Moved attention validation helpers to `pegainfer-kimi-k2/src/layers/attention/validation.rs`.
- Moved expert tests to `pegainfer-kimi-k2/src/layers/experts/tests.rs`.

### Step 3: Split rank worker

- Moved `KimiRankThreadState` command handling to `pegainfer-kimi-k2/src/runner/worker/state.rs`.
- Moved decode cache/arena/scratch impls to `pegainfer-kimi-k2/src/runner/worker/cache.rs`.
- Moved forward kernel paths to `pegainfer-kimi-k2/src/runner/worker/forward.rs`.
- Moved weight-cache loading and shape checks to `pegainfer-kimi-k2/src/runner/worker/load.rs`.
- Moved collectives, RoPE helpers, sampling helpers, and decode scalar helpers to `pegainfer-kimi-k2/src/runner/worker/runtime.rs`.

### Step 4: Verify

- `cargo fmt --all --check` passed.
- `PEGAINFER_CUDA_SM=90a PEGAINFER_TRITON_PYTHON=$LOCAL_PEGAINFER_DIR/.venv/bin/python3 cargo check --release -p pegainfer-kimi-k2 --features kimi-k2 --tests` passed.
- `PEGAINFER_CUDA_SM=90a PEGAINFER_TRITON_PYTHON=$LOCAL_PEGAINFER_DIR/.venv/bin/python3 cargo check --release -p pegainfer-kimi-k2 --lib` passed after gating Kimi runtime/weights exports behind the crate `kimi-k2` feature.

## Debrief

- **Outcome**: All Rust files under `pegainfer-kimi-k2/src` are now below 1k lines; the worker split preserved the Kimi feature compile gate and the default config/tokenizer build.
- **Pitfalls encountered**:
  - Rust module visibility needed explicit promotion for methods moved under `runner/worker/*`.
  - The default feature check exposed that Kimi runtime/weights exports were visible without the `kimi-k2` kernel feature.
- **Lessons learned**:
  - `worker.rs` has at least five natural ownership boundaries: rank state, cache/arena, forward kernels, load/shape checks, and runtime helpers.
  - Kimi default-feature compilation should retain config/tokenizer probing while keeping kernel-backed runtime modules behind `kimi-k2`.
