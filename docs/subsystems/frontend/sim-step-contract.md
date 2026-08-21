# pegainfer-sim onto the step contract

> **TL;DR:** `pegainfer-sim` now launches as `LaunchedEngine::Stepped`: one `SimScheduler` per partition, driven by the contract's polling loop. Legacy `EngineHandle` / `TokenEvent` is gone from this crate.
>
> **Last touched:** 2026-08

## Preparation

- **Read**:
  - `docs/index.md` — sim lives under subsystems/frontend; architecture doc is the contract source of truth.
  - `docs/subsystems/frontend/frontend-architecture.md` — onboarding checklist: implement `Scheduler`, `spawn_scheduler`, return `LaunchedEngine::Stepped`. Echo-server and qwen3 `frontend_adapter.rs` are the references.
  - `docs/subsystems/frontend/simulated-inference-engine.md` — sim is a CPU-only frontend/bench harness with fixed TTFT/TPOT; still that, different wire.
  - `docs/conventions/migration-defense.md` — every old defensive structure needs an heir.
- **Relevant history**:
  - Qwen3 already migrated (`pegainfer-qwen3/src/frontend_adapter.rs`).
  - `pegainfer-frontend/examples/echo-server.rs` is the minimal Scheduler.
  - Sim e2e currently injects metrics through `EngineHandle::with_metrics_watches` (legacy watch publisher). Stepped stamps stats onto step batches; there is no watch to inject into.
- **Plan**:
  1. Rewrite `pegainfer-sim/src/lib.rs` as a `Scheduler`: admit queued, emit one token per due request per step, finish/stop as today.
  2. `start_engine` returns `Engine`; CLI and tests pass `engine.into()` (`LaunchedEngine::Stepped`).
  3. Timing stays deadline-based inside `step` (sleep until next due token) so a CPU-only sim does not spin a core the way a GPU scheduler can.
  4. Rewrite unit tests to collect `StepOutputs` / `Terminal`. Keep HTTP/chat/tool-call e2e. Replace watch-injection metrics tests with stepped-path gauges. Drop the closed-feed test (no feed). Update partition-mismatch to the Stepped error string.
  5. Update living docs.
- **Risks / open questions**:
  - Stepped `scheduler_stats_from` still drops `spec_decode`. That is a frontend gap (qwen3 already publishes counters the stepped bridge does not stamp). Sim will not pretend to close it.

## Execution Log

### Step 1: Rewrite `pegainfer-sim` as a `Scheduler`
- `src/lib.rs` is now `SimScheduler` + `start_engine` / `start_engine_with_partitions` returning `Engine`.
- Timing is deadline-based inside `step`; parks at most 1ms so admission is not stalled for a full TPOT.
- `src/main.rs` passes `engine.into()` (`LaunchedEngine::Stepped`).
- Result: compile-ready; unit tests drive the step stream instead of `TokenEvent`.

### Step 2: Rewrite HTTP e2e
- Dropped watch-injection (`publish_metrics`, closed-feed, spec-decode inject). Those tested the legacy handle publisher; frontend unit tests in `bridge/tests.rs` still cover that path.
- Kept two-engine idle-gauge settle, added a slow-request test that the running gauge rises then drains.
- Partition-mismatch now matches the Stepped error string.
- Result: e2e follows the contract the crate actually speaks.

### Step 3: Verify
- `cargo clippy --release -p pegainfer-sim --all-targets -- -D warnings` clean.
- `cargo test --release -p pegainfer-sim`: lib 6, frontend_e2e 12, tool_call_roundtrip 3.

### Unexpected
- `pegainfer-sim/Cargo.toml` already had `pegainfer-frontend = { workspace = true }` before this cut. The missing piece was the engine contract, not the Cargo edge.

## Debrief

- **Outcome**: `pegainfer-sim` launches only as `LaunchedEngine::Stepped`. Legacy `EngineHandle` / `TokenEvent` is gone from this crate.
- **Pitfalls encountered**:
  - Metrics e2e used `EngineHandle::with_metrics_watches` to inject snapshots. Stepped has no watch; injecting would have been a second fake contract. Replaced with occupancy the scheduler actually reports.
- **Lessons learned**:
  - The Cargo dep was already there; "切到 frontend" here means the step contract, not adding a crate edge.
  - Stepped `scheduler_stats_from` still drops `spec_decode`. That is a frontend gap (qwen3 already publishes counters). Do not fake it in sim.
- **Follow-ups**:
  - Stamp `spec_decode` on the stepped bridge when a line that actually drafts is ready to prove it.
