# Qwen3.5 onto the step contract

> **TL;DR:** `pegainfer-qwen35` launches only as `LaunchedEngine::Stepped`. `Qwen35Scheduler` implements the step contract in `scheduler/`; the legacy `EngineHandle` / `TokenEvent` path is gone. GPU gates passed on RTX 5070 Ti against `/data/models/Qwen3.5-4B`.
>
> **Last touched:** 2026-08

## Preparation

- **Read**:
  - `docs/index.md` — Qwen3.5 is still listed as a legacy-handle line; frontend architecture says glm52 is next, then qwen35. User asked for qwen35 now, and to drop the old path.
  - `docs/subsystems/frontend/frontend-architecture.md` — onboarding checklist: implement `Scheduler`, `spawn_scheduler`, return `Stepped`. Echo-server / K3 `scheduler/mod.rs` / Qwen3 `frontend_adapter.rs` are the references. Legacy modules stay in the frontend crate until every line migrates.
  - `docs/subsystems/frontend/sim-step-contract.md` — sim already cut over with no `EngineHandle` leftover; tests drive `StepOutputs` / `Terminal`.
  - `docs/conventions/migration-defense.md` — every old defensive structure needs an heir (inherit / replace / argue impossible).
  - `docs/lessons/exact-match-gate-thread-cublas.md` — scheduler thread must rebind CUDA context and init thread-local cuBLAS. Today's `bind_model_thread` must land on the contract driver thread, not the load thread.
  - `docs/models/qwen35/load-snapshot.md` — today's snapshot is drain → prune → publish → admission. The step driver publishes *after* `step()`, same as Qwen3 already does.
  - `docs/models/qwen35/model-crate.md` — crate still documents `start_engine` / `EngineHandle` as the root-facing API.
  - `docs/models/qwen35/unified-prefill-overlap.md` — Shared-SM overlap: at most one inflight prefill; when decode retires first, wait on the CUDA event instead of parking on submit.
  - `pegainfer-k3/src/model_line.rs` + `pegainfer-k3/src/scheduler/mod.rs` — target shape: `submit` parks, `step` writes the ledger, `start_with_executors` → `Engine`, `launch` → `Stepped`. Fake-executor contract tests in `scheduler/tests.rs`.
  - `pegainfer-qwen3/src/frontend_adapter.rs` + `tests/common/harness.rs` — closest *behavioral* reference for chunked prefill / overlap / abort-as-flag. GPU tests use a step-stream harness, not `TokenSink`. We are **not** copying Qwen3's extra `frontend_adapter.rs` layer.
  - `pegainfer-qwen35/src/model_line.rs`, `lib.rs`, `scheduler.rs` — `launch` currently maps to `LaunchedEngine::Handle`. `scheduler_loop` owns submit drain, TokenSink dispatch, idle `blocking_recv`, and a watch metrics publisher.
- **Relevant history**:
  - Qwen3 and `pegainfer-sim` already migrated; glm52/qwen35/kimi/dsv2/gemma4 still launch through `Handle`.
  - Qwen3 kept `scheduler.rs` as contract-free mechanics and put the ledger writes in `frontend_adapter.rs`. K3 folded both into `scheduler/`. Qwen3.5 already has `scheduler/plan.rs` as the mechanics split, so K3's layout is the closer fit.
  - #830-style risk: `prune_closed_requests` (channel-close = cancel), `terminal_scheduler_shutdown` (fail every in-flight request), `bind_model_thread`, and the overlap "don't park while inflight" rule are the defenses that must not vanish.
- **Plan**:
  1. Reshape `pegainfer-qwen35` like K3:
     - `scheduler/mod.rs` implements `Scheduler` (`submit` / `step` / `metrics`) and `start_with_backend` → `Engine`.
     - Keep `scheduler/plan.rs` as admission / chunk / KV-budget mechanics.
     - Split the current 2.6k-line `scheduler.rs` into `scheduler/{mod,backend}.rs` (single-GPU + TP backends stay behind one enum; GPU execute/overlap stay south of the ledger).
     - `model_line::launch` returns `LaunchedEngine::Stepped` only.
  2. Delete the old path from this crate — no `EngineHandle`, no `TokenSink` / `TokenEvent` / `GenerateRequest`, no `start_engine*`, no private `scheduler_loop` / submit channel / watch publisher. Tests and `runtime` re-exports follow the new `Engine`.
  3. Wire replacements (defense table in execution):
     - Token dispatch → `RequestLedger::push_tokens` / `finish` / `fail` / `reject` / `retire`.
     - Channel-close cancel → `ledger.is_aborted` on every touch (same as K3 `finish_or_retire`).
     - Idle `blocking_recv` → contract driver spin (overlap wait on the CUDA event stays *inside* `step`, matching today's "don't park while inflight").
     - Load watch → `Scheduler::metrics` after prune; driver publishes once per iteration.
     - Fatal shutdown → `step` returns `Err`; driver `fail_all`s open accounts (drop-bomb covers the rest).
     - `bind_model_thread` + cuBLAS init on first `step` of the driver thread.
  4. Rewrite tests onto the contract:
     - Protocol: fake executor in `scheduler/tests.rs` (K3 pattern) — abort, reject, stop/length, overlap idle, fatal fail-all.
     - GPU: copy Qwen3's `EngineHarness` into `tests/common/`; rewrite `e2e_scheduler`, `sampling_behavior`, `chunked_prefill`, `serving_tp2` off `EngineHandle`.
     - Keep plan.rs unit tests; they do not speak the wire.
  5. Docs: this file's execution log; `frontend-architecture.md` (qwen35 is stepped); `model-crate.md` / `load-snapshot.md` TL;DRs if they would mislead.
  6. Verify: `cargo fmt` / clippy `-p pegainfer-qwen35 --features qwen35 --all-targets -- -D warnings`; lib + scheduler unit tests; GPU gates `e2e_scheduler` / `sampling_behavior` / `chunked_prefill` when a card is available.
- **Risks / open questions**:
  - Metrics cadence moves from pre-admission to post-`step` (Qwen3 already lives with this). Load-snapshot tests must assert the new boundary, not the old one.
  - Shared-SM overlap currently parks on a CUDA event when decode empties first. That wait belongs inside `step`, not as a driver idle. Easy to get wrong and stall admission.
  - cuBLAS thread-local handles: skipping `bind_model_thread` on the driver thread reproduces the historical gibberish bug.
  - Frontend crate still carries the legacy contract for glm52/kimi/dsv2/gemma4. This task does not delete those modules.

## Execution Log

### GPU tests `sampling_behavior` + `chunked_prefill` onto EngineHarness
- Copied `pegainfer-qwen3/tests/common/harness.rs` into `pegainfer-qwen35/tests/common/harness.rs` (identical). `tests/common/mod.rs` already exports `harness`.
- Rewrote `tests/sampling_behavior.rs` and `tests/chunked_prefill.rs` onto `EngineHarness` / `Request` / `Terminal`. Deleted `EngineHandle`, `GenerateRequest`, `TokenSink`, `TokenEvent`, `TokenStreamReceiver`. Sampling params stay on `Request.params`. `ignore_eos` still forces a length finish, so `outcome.tokens` is exactly `max_tokens` (no stop/EOS token to drop).
- Coverage kept: greedy determinism, `top_k=1` / tiny `top_p` collapse, hot-temp diversity, chunked vs unchunked greedy token match + `FinishReason::Length`.
- Not in these two files before, so not added here: mixed-batch (lives in `e2e_scheduler.rs`, untouched), abort/cancel, `Terminal::Rejected`, timeouts.
- Result: files formatted; compile/GPU run wait on `start_engine` returning `Engine`.

### GPU test `serving_tp2` onto EngineHarness
- Rewrote `tests/serving_tp2.rs` off `LaunchedEngine::Handle` / HTTP `vllm::serve`. Live generate now: `start_engine_with_capacity` → `EngineHarness::new` → `submit` / `expect_finished`. CUDA Graph + TP still asserts `start_engine_with_capacity(...).err()` contains `"eager execution only"` (now a non-ignored pre-load test; no weights/GPU).
- Coverage kept: greedy `ignore_eos` length stop, completion-token count, finite logprobs, two concurrent prompts, TP2 + CUDA Graph fail-closed before load.
- Dropped: OpenAI `/v1/models` + `/v1/completions` HTTP (streaming SSE `[DONE]`, usage JSON, health wait). That was this file's unique serving smoke; in-process harness does not replace the frontend wire.
- Removed unused `reqwest` / `tokio-util` from `pegainfer-qwen35` dev-dependencies (HTTP client was only used here). `lib.rs` launch-validation tests already match `.err()` and do not name `EngineHandle`.
- Result: files formatted; compile waits on `start_engine_with_capacity` returning `Engine`.

### HTTP TP2 serving gate inherited onto Stepped
- Restored `qwen35_tp2_serves_openai_completions_over_http` from `HEAD:pegainfer-qwen35/tests/serving_tp2.rs`. The generate harness and CUDA Graph + TP pre-load reject tests stay; the HTTP gate was not abolished.
- `spawn_ready_server` still `spawn_blocking`s `start_engine_with_capacity`, then hands the engine to `pegainfer_frontend::vllm::serve(std::future::ready(Ok(LaunchedEngine::Stepped(engine))), …)` — same model path / port / `CancellationToken` args as HEAD, `Handle` swapped for `Stepped`. No `EngineHandle` / `TokenSink`.
- Assertions inherited: `/v1/models` advertises `qwen35-tp2-serving-smoke`; non-streaming `/v1/completions` (non-empty text, `finish_reason=length`, usage JSON, finite logprobs); SSE streaming including `data: [DONE]` and a choice payload; concurrent completions; `/health` wait before the first request; 30s shutdown. `#[ignore]` text and `PEGAINFER_TEST_FRONTEND_MODEL_PATH` fallback (`common::model_fixture::frontend_model_path_or_skip`) match HEAD. CUDA Graph fail-closed stays the standalone pre-load test (not re-nested in HTTP).
- Restored `reqwest` (`features = ["json"]`) and `tokio-util` in `pegainfer-qwen35` dev-dependencies, same as HEAD.
- Result: files formatted; HTTP compile waits on `start_engine_with_capacity` returning `Engine`.

### Crate-internal scheduler unit tests onto the step contract

Rewrote `pegainfer-qwen35/src/scheduler/tests.rs` off TokenSink / TokenEvent. No dual path in the test file. Production `scheduler.rs` still owns the legacy loop (rewritten in parallel); tests now call `echo_refusal` / `contract_reject_reason` / `prefill_drop_expectation` / `logical_load_counts` helpers added there. `plan.rs` tests were not touched. GPU tests under `pegainfer-qwen35/tests/` were not touched by this slice.

`cargo test --release -p pegainfer-qwen35 --features qwen35 --lib -- scheduler::tests`: 7 passed, 1 ignored (`tp2_scheduler_runs_forced_mixed_steps`). `start_tp_with_capacity` still returns `EngineHandle`; the Length-terminal assertions live in `assert_forced_mixed_steps(Engine)` and become live once start returns `Engine`.

#### Defense table (crate-internal tests)

| Old defense | Failure mode | Heir |
| --- | --- | --- |
| `send_rejection` KvBudget TokenEvent message | client does not see lifetime KV tokens | **inherit**: `contract_reject_reason` → `RejectReason::KvBudget { worst_case_tokens }` Display (`max_request_tokens=80`) |
| `send_rejection` ContextLength TokenEvent message | client does not see window + requested length | **inherit**: `contract_reject_reason` → `RejectReason::ContextLength` Display |
| `reject_unsupported_echo` + `UNSUPPORTED_ECHO_MESSAGE` | echo request reaches backend admission | **replace**: `echo_refusal` → `RejectReason::EchoPrefillTokens { limit: 0 }`; assert contract Display, not the old TokenEvent string |
| `tp_engine_rejects_cuda_graph_before_model_load` | TP+CUDA Graph starts loading weights | **inherit**: `start_engine_with_capacity` still fail-closes before load (`eager execution only`); Ok type is irrelevant because the test asserts `Err` |
| `tp2_scheduler_runs_forced_mixed_steps` TokenSink collection | TP mixed decode+chunk-prefill does not finish at Length | **replace**: ignored GPU test launches via `start_tp_with_capacity`; step-stream assertions in `assert_forced_mixed_steps(Engine)` (K3 StepCollector). TokenSink collection abolished |
| `closed_pending_work_is_pruned_before_admission` | cancelled queued request is admitted | **replace**: cannot mint `RequestLedger` here. Heir is K3 `aborted_request_retires_silently_and_frees_its_slot` (`ledger.is_aborted` on admit). Comment in the test file |
| `closed_resident_work_is_absent_from_post_prune_load` | cancelled resident still in load snapshot | **replace**: abort prune needs a ledger (K3). Load formula heir: `logical_load_counts` still counts inflight as running (`overlap_wait_policy_is_inside_step`) |
| `closed_resident_frees_capacity_for_same_tick_admission` | cancelled decode slot not reused this tick | **replace**: same K3 abort+admit path; not unit-testable here without a ledger |
| `closed_materialized_prefill_requires_existing_worker_state` | drop missing TP worker state / drop unmaterialized state | **inherit**: `prefill_drop_expectation(cursor)` — `MustBeAbsent` at cursor 0, `MustExist` after |
| `prune_drop_failure_preserves_pending_for_terminal_fanout` | prune fatal starves pending of an error | **abolish**: `step` returns `Err`; driver `fail_all` (`pegainfer-frontend/src/engine/driver.rs` `fatal_step_fails_in_flight_requests_with_the_error`) |
| `decode_eos_waits_for_drop_before_finished` | client sees Finished before KV drop | **abolish**: one `RequestUpdate` committed after the whole step; drop always happens before the client sees the terminal |
| `decode_length_waits_for_drop_before_token_and_finished` | Token then Finished before drop | **abolish**: same structure |
| `non_tp_decode_preserves_publish_before_retire_order` | TokenEvent order vs retire | **abolish**: same structure |
| `decode_completion_drop_failure_publishes_only_terminal_error` | drop fail publishes Token then Error | **abolish**: `step` `Err` → driver `fail_all`; no TokenEvent fan-out |
| `immediate_prefill_completion_waits_for_drop` | prefill Length before drop | **abolish**: same one-update-per-step structure |
| `immediate_prefill_drop_failure_publishes_only_terminal_error` | remaining scheduled prefill not failed | **abolish**: driver `fail_all` covers every open account |
| `terminal_shutdown_closes_drains_and_errors_every_owner_once` | shutdown misses an owner / double Error / load not zeroed | **abolish**: frontend `drive()` `fail_all` + drop-bomb; already tested in `pegainfer-frontend/src/engine/driver.rs` |
| `FatalSchedulerError.transient` + TokenEvent::Error fan-out | fatal loses in-flight requests | **abolish**: the ledger holds an account for every unanswered request; `fail_all` writes them off |
| `should_block_on_submit` after last decode retires | scheduler parks on submit while inflight prefill is the only work | **replace**: wait is inside `step` (overlap_wait). Untestable without GPU; `overlap_wait_policy_is_inside_step` asserts `logical_load_counts` still counts inflight as running so the driver cannot see idle |

Admission/chunk tests that never spoke TokenSink stay in `scheduler/plan.rs`.

### Production cutover: `Qwen35Scheduler` on the step contract

No dual path. `scheduler.rs` is now `scheduler/{mod,backend,plan,tests}.rs`. `Qwen35Scheduler` implements `submit` / `step` / `metrics`. `start_with_capacity` / `start_with_capacity_and_policy` / `start_tp_with_capacity` return `Engine` via `spawn_scheduler` (`qwen35-scheduler` / `qwen35-scheduler-tp`). `model_line::launch` returns `LaunchedEngine::Stepped` only. `start_engine*` keep their names but return `Engine`. GPU tests and HTTP TP2 were already on `Engine` / `Stepped`; they compile against this cutover. `assert_forced_mixed_steps` is now called from the ignored TP2 lib test.

`bind_model_thread` + `CublasThreadGuard` run on first `step` of the driver thread (single-GPU only; stored on the scheduler). TP workers still bind themselves.

#### Defense table (loop / TokenSink / bind / overlap / metrics / shutdown)

| Old defense | Failure mode | Heir |
| --- | --- | --- |
| Own `scheduler_loop` + tokio submit + idle `blocking_recv` | second loop disagrees with the driver; parks while GPU work is in flight | **replace**: contract `drive()` drains submit, calls `step`, publishes metrics, commits. Idle is `spin_loop` in the driver |
| `TokenSink` / `TokenEvent` dispatch | per-request send, ordering by convention, send-fail = cancel | **replace**: `ledger.admit` / `reject` / `push_tokens` / `finish` / `fail` / `retire`. Stop token is not pushed (bridge appends EOS for usage) |
| `token_tx.is_closed` prune (`prune_closed_requests`) | cancelled work is admitted or stays in load | **replace**: `ledger.is_aborted` on every touch; `finish_or_retire` on every finish path |
| `should_block_on_submit` (`owned_work_empty && !inflight`) | parks on submit while inflight prefill is the only work | **replace**: if inflight and active empty, `step` waits on the CUDA event (`overlap_wait`); if active nonempty, decode while polling. Never park on submit. Inflight counts as running so the driver cannot see idle |
| Watch `LoadSnapshot` publish before admission | cancelled residents appear as running; waiting misses same-tick submits | **replace**: `Scheduler::metrics` after `step` (same cadence as Qwen3). Inflight in running; deferred/queued in waiting. Driver publishes once per iteration |
| `FatalSchedulerError` + TokenEvent Error fan-out + `terminal_scheduler_shutdown` | fatal misses an in-flight owner / double Error | **abolish**: `step` `Err` → driver `fail_all` + drop-bomb. Single-GPU execute failures `ledger.fail` the touched requests and keep serving (`Ok`) |
| `completion_requires_drop_ack` TokenEvent-vs-drop order | client sees Finished before TP KV drop | **abolish**: drop backend state, then `ledger.finish`; one `RequestUpdate` committed after the whole step — structurally drop-before-visible-terminal |
| `bind_model_thread` before `scheduler_loop` on a private thread | first `step` of `spawn_scheduler` runs without CUDA/cuBLAS binding → gibberish logits | **inherit**: first `step` of the driver thread calls `bind_model_thread`; `CublasThreadGuard` lives on `Qwen35Scheduler` for the thread lifetime. Single-GPU only (TP workers bind themselves) |

### GPU gates `e2e_scheduler` / `sampling_behavior` / `chunked_prefill`

- Weights are not under the repo `models/` (that directory does not exist). Fixture is `/data/models/Qwen3.5-4B` (`config.json` `model_type=qwen3_5`, `text_config.model_type=qwen3_5_text`; ~9.3 GiB safetensors). Duplicate copy at `/data/openclaw-data/workspace/models/Qwen3.5-4B` not used.
- `nvidia-smi` before run: RTX 5070 Ti, 34 MiB / 16303 MiB, 0% util, no processes. One GPU, so TP2 ignored.
- Command:

```
PEGAINFER_TEST_MODEL_PATH=/data/models/Qwen3.5-4B cargo test --release -p pegainfer-qwen35 --features qwen35 --test e2e_scheduler --test sampling_behavior --test chunked_prefill -- --test-threads=1
```

- First run: `chunked_prefill` passed; `test_e2e_qwen35_scheduler` OOM'd at load (`from_safetensors_with_options` sizes recurrent state to MAX_BATCH=64: 6288 MB + 3248 MB scratch vs 7227 MB free). Pre-existing vs HEAD, but the test comment already wanted 8 slots for 16GB. Aligned TP1 onto `start_engine_with_capacity(..., 8)` like TP2 in the same file. Not a bind_model_thread / EngineHandle leftover.
- Second run: **4 passed, 0 failed, 1 ignored** (`test_e2e_qwen35_scheduler_tp2`). `chunked_prefill` 1/1, `e2e_scheduler` 2/2 (+1 ignored), `sampling_behavior` 1/1. No gibberish / cuBLAS first-step failure.

## Debrief

- **Outcome**: `pegainfer-qwen35` production launch is on the step contract only. `Qwen35Scheduler` lives in `scheduler/`, `launch` returns `LaunchedEngine::Stepped`, `start_engine*` return `Engine`. Lib tests: 90 passed, 7 ignored, 0 failed. Clippy `-D warnings --all-targets` passed. GPU gates on RTX 5070 Ti + `/data/models/Qwen3.5-4B`: 4 passed, 0 failed, 1 ignored (TP2).
- **Pitfalls encountered**: `bind_model_thread` used to run before the private loop and `start_*` waited on it. The contract driver thread is a different spawn, so binding had to move to first `step` or the historical cuBLAS gibberish bug returns. Overlap wait must stay *inside* `step`; returning idle to the driver while inflight would spin-loop instead of waiting on the CUDA event. `test_e2e_qwen35_scheduler` still loaded via `from_safetensors_with_options` (64 slots) then `start_with_capacity(8)` — that OOM'd on 16GB before any step ran; the 8-slot comment only applied after load.
- **Lessons learned**: K3's crate shape (Scheduler in `scheduler/`, ledger writes next to start) fits Qwen3.5 better than Qwen3's extra `frontend_adapter.rs`, because `plan.rs` was already the mechanics split. TokenEvent drop-vs-publish tests are structurally gone; keep the failure-mode comments. Repo `models/` is absent on this machine; the Qwen3.5-4B fixture lives at `/data/models/Qwen3.5-4B`. GPU tests must load through `start_engine*` with an explicit small `max_batch` so recurrent-state reservation matches the 16GB budget.
- **Follow-ups**: `serving_tp2` / `tp2_scheduler_runs_forced_mixed_steps` still skipped — only one GPU. A fake-backend protocol suite (abort/reject/stop) still cannot mint `RequestLedger` from this crate.
