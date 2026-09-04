# Frontend architecture: pegainfer-frontend and the engine boundary

**TL;DR:** `pegainfer-frontend` owns everything north of the model schedulers: the engine contract, the vLLM protocol stack, and the `ModelLine` dispatch trait. The contract now has two generations living side by side: the **step contract** (`StepOutputs` wire + `RequestLedger` lifecycle + a contract-owned polling driver — Qwen3, Gemma 4 and `pegainfer-sim` are migrated) and the **legacy handle contract** (`EngineHandle` + `TokenEvent` per-request events — glm52/qwen35/kimi-k2/deepseek-v2-lite still launch through it). **Next step: migrate glm52, then delete the legacy contract.**

Last touched: 2026-09

## The boundary, in one sentence

An engine is a set of schedulers, each a `Scheduler` implementation driven by the contract's polling loop: *the frontend submits `Request`s into a scheduler and receives one `StepOutputs` message per scheduler step, in which every touched request has exactly one flat `RequestUpdate`; every request ends in exactly one terminal.* Tokenizer, chat templates, HTTP, metrics, LoRA routing live north of the scheduler; KV, batching, CUDA live south. The contract contains no CUDA types, and the frontend holds no model-layer structs — everything it touches is a contract type (channels, `JoinHandle<()>`, POD info). How many schedulers an engine exposes and what each one means (DP rank, P/D role) is the model line's decision; the contract attaches no rank semantics, and TP/EP lockstep is encapsulated below the `Scheduler` impl. Placement across schedulers is frontend policy over the load feed.

## The step contract

```
pegainfer-frontend/src/engine/
├── step.rs              # the wire: RequestId, Request, QueuedRequest,
│                        #   StepOutputs { Vec<RequestUpdate> },
│                        #   RequestUpdate { scheduled, tokens, logprobs, cached_tokens,
│                        #   prompt_echo, kv_transfer, terminal }, Terminal
├── request_lifecycle.rs # submission envelope, abort control and step sender plumbing;
│                        #   DeferredFinish remains available for P/D handoff
├── ledger.rs            # RequestLedger: admit/reject/push/finish/fail/retire,
│                        #   prompt/completion tallies, one merged update per touched id
├── wiring.rs            # scheduler_pair, SchedulerHandle (submit/take_steps/load),
│                        #   Engine { schedulers, info, lora }, LiveScheduler,
│                        #   EngineInfo, LaunchedEngine { Handle | Stepped }
├── control.rs           # LoRA capability — outside the contract: LoraControl vocabulary,
│                        #   LoraClient (the `Engine.lora: Option<LoraClient>` capability)
└── driver.rs            # trait Scheduler { submit, step, metrics } + spawn_scheduler:
                         #   the contract-owned pure-polling drive loop
```

Design decisions worth knowing before touching it:

- **Step-batched wire.** One message per scheduler step, not one channel per request: the scheduler's natural output unit is the step batch, and per-request channels were tried and rejected (the scheduler for-loop over N channels was the bottleneck). The protocol stack demuxes.
- **Flat `RequestUpdate`.** All facts a step produced for one request travel in one struct, so intra-request ordering is structure, not convention. The ledger merges every write for an id into that one record before the driver commits the step.
- **Ledger lifecycle.** `RequestLedger` owns one account for every unanswered submission. A scheduler receives only `QueuedRequest { id, request }` and writes every lifecycle transition by id. The ledger rejects touches after closure, derives completion counts from `push_tokens`, and writes off every open account when an engine-fatal `step` error ends the driver.
- **Ledger as single writer.** Schedulers never touch the step channel; they call ledger methods. `admit` stamps `ScheduledInfo` from the registered prompt length, `push_tokens` tallies completions, and `commit_step` publishes the merged statement once per driver iteration.
- **Polling driver, scheduler-owned park.** `spawn_scheduler` owns the serve loop: drain submissions, `Scheduler::step`, publish metrics, commit. An idle iteration ends in a `spin_loop` hint. Gemma 4 has one deliberate park inside this policy: while async prefill is the only remaining work, its scheduler drains the lane and joins it rather than hot-polling the completion (a drain failure is engine-fatal); when decode or queued work exists it keeps polling the lane without blocking.
- **Abort is a flag, not channel teardown.** `SchedulerHandle::submit` returns a `RequestControl`; the frontend flips its boolean abort flag and the scheduler retires the request silently on its next touch (no terminal — the frontend already dropped its state for that id).
- **Channels:** the submit channel is crossbeam (sync consumer on the scheduler thread), steps are tokio mpsc (async consumer in the bridge); load is a shared cell read via `SchedulerHandle::load()` — pull-only by design, "notify me on load change" is deliberately unrepresentable (the driver busy-polls, so a subscription edge would fire per spin). All channels unbounded on purpose — admission control is the scheduler's job, expressed as `Rejected`, never as backpressure on submit.
- **Control plane lives outside the contract.** `Scheduler` has no control method and the contract carries no control channel. A capability like LoRA is a private channel the model crate mints *before* `spawn_scheduler` — the scheduler closes over the receiver, the `LoraClient` sender surfaces as `Engine.lora: Option<LoraClient>`, and the `Option` *is* the capability (no `bool` flag, no registry until a second capability exists). The vocabulary (`LoraControl`, `LoraClient`) is still defined in the frontend crate because the frontend must speak it without holding model structs; only the wiring is the model's business.

### Onboarding checklist for a new model line

1. Implement `Scheduler` (see `Qwen3Scheduler` in `pegainfer-qwen3/src/frontend_adapter.rs` — the whole adaptation is deliberately in one file):
   - `submit(queued)` — ownership transfer only: park `QueuedRequest { id, request }`. No verdict here by design — `step` is the single ledger-writing site, and the driver commits once per iteration.
   - `step(ledger)` — one scheduling step: admit/reject, execute, push tokens, finish/fail/retire. Return `Err` only for engine-fatal states.
   - `metrics()` — KV occupancy + running/waiting counts for routers and shutdown draining.
   - Extra capabilities (LoRA etc.) are not trait methods: mint the private channel before spawning, close the scheduler over the receiver, drain it inside `step`.
2. `spawn_scheduler(name, scheduler)` per scheduler; return `Engine { schedulers, info: EngineInfo { kv_capacity, servable_len }, lora }` — the required-metadata fields are the checklist, and `lora: Some(client)` only when the line actually serves adapter control.
3. `ModelLine::launch` returns `LaunchedEngine::Stepped(engine)`. `pegainfer-sim` is the CPU-only reference (`SimScheduler` in `pegainfer-sim/src/lib.rs`); it has no `ModelLine` and hands the `Engine` to `vllm::serve` directly.

The contract's own invariants are tested in `ledger.rs`/`driver.rs` tests; the qwen3 adapter's contract tests (`frontend_adapter/tests.rs`) are the CPU reference for testing a model's protocol behaviour end to end with a fake executor. Gemma 4's ignored GPU lifecycle gates drive the same `spawn_scheduler`/`StepOutputs` surface.

## The legacy handle contract (migration pending)

`request.rs`/`event.rs`/`sink.rs`/`kv.rs`/`handle.rs` still carry the previous generation: `launch -> EngineHandle`, per-request `TokenSink` events (`Scheduled … Token* … terminal` by convention), send-failure-as-cancellation. glm52, qwen35, kimi-k2 and deepseek-v2-lite launch through it (`LaunchedEngine::Handle`), and the vllm stack keeps both bridge paths (`bridge.rs` for handles, `bridge/stepped.rs` for step engines). KV-prefix resolution (`KvPrefix`, `submit_resolved`) currently exists only on the legacy path; step schedulers can refuse only the features represented by `Request`, including non-null `kv_transfer_params`.

## Crate layout

```
pegainfer-frontend
├── engine/            # both contract generations, re-exported flat
├── sampler.rs         # SamplingParams
├── parallel.rs        # ParallelConfig
├── tracing_state.rs   # global tracing on/off flag (frontend + schedulers both read it)
├── model_line.rs      # ModelLine trait + ModelLineRegistry + SharedArgs (dispatch seam)
└── vllm/              # protocol stack #1: vLLM EngineCore impersonation over ZMQ
    ├── mod.rs         #   serve_* entry points; engine_task fans out per LaunchedEngine arm
    ├── bridge.rs      #   LocalEngineBridge (legacy handle path) + shared BridgeLink
    ├── bridge/stepped.rs  # SteppedEngineBridge: StepOutputs -> EngineCore messages
    ├── wire.rs        #   EngineCoreSamplingParams <-> SamplingParams translation
    ├── lora.rs        #   /v1/{load,unload}_lora_adapter over LoraClient
    └── request_contract.rs  # GLM5.2 prefill-only route guard
```

Dependency direction: `pegainfer-frontend ← pegainfer-core ← model crates ← pegainfer-server` (thin bin). The frontend never depends on core or model crates — that keeps the contract CUDA-free and leaves the server binary as the only model dispatch point.

## ModelLine: what a new model provides

`model_line.rs` defines the dispatch seam. A model crate implements `ModelLine` (each crate has a `model_line.rs` exporting `pub static MODEL_LINE`):

- `name()` — family name for logs and errors.
- `probe(config_json) -> Result<(), String>` — claim or reject the model directory by its identity fields; exactly one registered line must claim a config.
- `augment_cli(cmd)` — the line's *exclusive* CLI section. Shared flags (`--tp-size`, `--kv-offload`, …) live in `SharedArgs`; a line opts into each via `consumed_shared_args()`.
- `validate(ctx, provided)` — the line's cross-flag rules, after the registry's consume-or-reject pass.
- `serve_plan(ctx)` — what the HTTP frontend must know before the engine finishes loading (scheduler count, prefill-only mode, LoRA route enablement).
- `launch(ctx) -> anyhow::Result<LaunchedEngine>` — assemble options, start the engine, return `Stepped` (step contract) or `Handle` (legacy).

All six lines are onboarded. Adding a model line = write `model_line.rs` in the crate + one registry entry + one Cargo feature.

## Protocol stacks

**`vllm` (current default, fleet-proven).** Impersonates a vLLM EngineCore process over in-process ZMQ/msgpack because upstream `vllm-server` assumes the engine is a separate process. HTTP routes, OpenAI types, tokenizer, chat templates, Prometheus live in the external `vllm-server`/`vllm-metrics`/`vllm-text` crates. `SteppedEngineBridge` translates each `RequestUpdate` 1:1 into an EngineCore output (wall-clock timestamps are reconstructed from the contract's `Instant`s via a per-bridge unix anchor; a `Finished{Stop}` appends the stop sentinel token, which is how usage keeps counting the suppressed EOS).

**`dynamo` (planned second stack).** dynamo's `lib/llm` in-process path removes the wire protocol entirely (`EngineConfig::InProcessTokens` + `run_input`). The step contract was shaped so this stack can consume `StepOutputs` directly without impersonation overhead. Decision gate: prototype, A/B against the vllm stack, let TTFT/step-overhead numbers pick the default.

## What was deleted, and the heirs

| Deleted | Heir |
| --- | --- |
| `pegainfer-engine`, `pegainfer-vllm-frontend`, dynamo crates, `bench_serving` | see git history of this doc for the pre-2026-08 consolidation table |
| Qwen3 `scheduler_loop` / `start_qwen3*` in `scheduler.rs` (double loop + sink plumbing) | contract driver (`driver.rs`) + `frontend_adapter.rs`; `scheduler.rs` is now contract-free mechanics |
| `TokenSink` send-failure-as-cancellation (stepped paths) | `RequestControl::abort` flag, observed through `RequestLedger::is_aborted` and closed with `retire` |
| `EngineCommand` + `EngineHandle` LoRA control methods | `LoraClient` over the model crate's private pre-spawn channel (`Engine.lora: Option<LoraClient>` is the capability); idle-drain policy lives in the qwen3 adapter (`pending_control` + `post_control_deferred`) |
| Per-request event-order convention (`Scheduled` before tokens, one terminal, enforced by hand) | `RequestLedger`: invalid transitions panic at the scheduler call site, and fatal driver teardown fails every open account instead of leaving clients hanging |
| KV block-event feed (`KvBlockEvent`/`KvStoredBlock`, `EngineHandle::take_kv_events`, qwen3 `ExecutorKvEvents` + `enable_kv_events` plumbing) | none — it never had a consumer. Rebuild from git history when a cache-aware router lands; the natural re-home is a `RequestUpdate`/load-feed sibling on the step contract |
| Qwen3 `scheduler/kv_events.rs` pump | same as above |

## Next step

Migrate glm52 onto the step contract (the multi-scheduler pilot; brings P/D and EP requirements), then qwen35/kimi-k2/deepseek-v2-lite, then delete the legacy contract modules and `LaunchedEngine::Handle`.
