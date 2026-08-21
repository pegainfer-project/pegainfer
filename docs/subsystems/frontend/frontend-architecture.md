# Frontend architecture: pegainfer-frontend and the engine boundary

**TL;DR:** `pegainfer-frontend` owns everything north of the model schedulers: the engine contract, the vLLM protocol stack, and the `ModelLine` dispatch trait. The contract now has two generations living side by side: the **step contract** (`StepOutputs` wire + typestate request handles + a contract-owned polling driver — Qwen3 and `pegainfer-sim` are migrated) and the **legacy handle contract** (`EngineHandle` + `TokenEvent` per-request events — glm52/qwen35/kimi-k2/deepseek-v2-lite/gemma4 still launch through it). **Next step: migrate glm52, then delete the legacy contract.**

Last touched: 2026-08

## The boundary, in one sentence

An engine is a set of schedulers, each a `Scheduler` implementation driven by the contract's polling loop: *the frontend submits `Request`s into a scheduler and receives one `StepOutputs` message per scheduler step, in which every touched request has exactly one flat `RequestUpdate`; every request ends in exactly one terminal.* Tokenizer, chat templates, HTTP, metrics, LoRA routing live north of the scheduler; KV, batching, CUDA live south. The contract contains no CUDA types, and the frontend holds no model-layer structs — everything it touches is a contract type (channels, `JoinHandle<()>`, POD info). How many schedulers an engine exposes and what each one means (DP rank, P/D role) is the model line's decision; the contract attaches no rank semantics, and TP/EP lockstep is encapsulated below the `Scheduler` impl. Placement across schedulers is frontend policy over the load feed.

## The step contract

```
pegainfer-frontend/src/engine/
├── step.rs              # the wire: RequestId, Request, StepOutputs { Vec<RequestUpdate> },
│                        #   RequestUpdate { scheduled, tokens, logprobs, cached_tokens,
│                        #   prompt_echo, kv_transfer, terminal }, Terminal
├── request_lifecycle.rs # typestate handles: QueuedRequest ─admit→
│                        #   ActiveRequest ─finish/fail/defer→ consumed; every
│                        #   transition is by-move, a dropped handle emits Failed
│                        #   (drop bomb); DeferredFinish; RequestControl
├── emitter.rs           # StepEmitter: the single writer of the per-step buffer; stamps
│                        #   timestamps, tallies prompt/completion counts, folds each
│                        #   request's step into one RequestUpdate; commit_step sends once
├── wiring.rs            # scheduler_pair, SchedulerHandle (submit/take_steps/load),
│                        #   Engine { schedulers, info, lora }, LiveScheduler,
│                        #   EngineInfo, LaunchedEngine { Handle | Stepped }
├── control.rs           # LoRA capability — outside the contract: LoraControl vocabulary,
│                        #   LoraClient (the `Engine.lora: Option<LoraClient>` capability)
└── driver.rs            # trait Scheduler { submit, step, load } + spawn_scheduler:
                         #   the contract-owned pure-polling drive loop
```

Design decisions worth knowing before touching it:

- **Step-batched wire.** One message per scheduler step, not one channel per request: the scheduler's natural output unit is the step batch, and per-request channels were tried and rejected (the scheduler for-loop over N channels was the bottleneck). The protocol stack demuxes.
- **Flat `RequestUpdate`.** All facts a step produced for one request travel in one struct, so intra-request ordering is structure, not convention. This is what makes `defer_finish` safe: a P/D prefill executor can withhold a request's `Finished` until its KV saves are peer-visible and send it later from any thread — the deferred message carries the request's entire buffered update, so late delivery cannot reorder.
- **Typestate lifecycle.** `QueuedRequest` (queued) and `ActiveRequest` (streaming) are owned tokens; admit/reject/retire/finish/fail consume them, so "terminal exactly once, nothing after it" cannot be miscoded — it does not compile. A handle dropped without a transition emits `Failed` from its `Drop`, which is also how a crashed scheduler answers every in-flight request: the driver drops the scheduler, the handles fall, the terminals ship.
- **Emitter as single writer.** Schedulers never touch the channel; they call `StepEmitter` methods against their handles. The emitter stamps `ScheduledInfo` at admission, tallies token counts (terminal counts derive from the tally, never from model-side arithmetic), and `commit_step` publishes the whole step in one send.
- **Pure polling driver.** `spawn_scheduler` owns the serve loop: drain submissions, `Scheduler::step`, publish load, commit. No idle/park distinction — the scheduler owns the GPU and spinning on it costs nothing anyone else could use; async KV I/O (prefetch, decode-overlap prefill) is naturally absorbed by polling. An idle iteration ends in a `spin_loop` hint (relaxes the core's issue slots, no latency cost — busy iterations never pause). The loop exits when the frontend drops the handle and the queue drains.
- **Abort is a flag, not channel teardown.** `SchedulerHandle::submit` returns a `RequestControl`; the frontend flips its boolean abort flag and the scheduler retires the request silently on its next touch (no terminal — the frontend already dropped its state for that id).
- **Channels:** the submit channel is crossbeam (sync consumer on the scheduler thread), steps are tokio mpsc (async consumer in the bridge); load is a shared cell read via `SchedulerHandle::load()` — pull-only by design, "notify me on load change" is deliberately unrepresentable (the driver busy-polls, so a subscription edge would fire per spin). All channels unbounded on purpose — admission control is the scheduler's job, expressed as `Rejected`, never as backpressure on submit.
- **Control plane lives outside the contract.** `Scheduler` has no control method and the contract carries no control channel. A capability like LoRA is a private channel the model crate mints *before* `spawn_scheduler` — the scheduler closes over the receiver, the `LoraClient` sender surfaces as `Engine.lora: Option<LoraClient>`, and the `Option` *is* the capability (no `bool` flag, no registry until a second capability exists). The vocabulary (`LoraControl`, `LoraClient`) is still defined in the frontend crate because the frontend must speak it without holding model structs; only the wiring is the model's business.

### Onboarding checklist for a new model line

1. Implement `Scheduler` (see `Qwen3Scheduler` in `pegainfer-qwen3/src/frontend_adapter.rs` — the whole adaptation is deliberately in one file):
   - `submit(req)` — ownership transfer only: take the payload (`req.take_request()`), mint your internal id, park the handle in a registry. No emitter here by design — `step` is the single emission site, and the driver commits once per iteration so nothing is gained by emitting earlier.
   - `step(emitter)` — one scheduling step: admit (consume queued handles via `emitter.admit`/`reject`), execute, push tokens, finish/fail/retire. Return `Err` only for engine-fatal states.
   - `load()` — KV occupancy + running/waiting counts for routers.
   - Extra capabilities (LoRA etc.) are not trait methods: mint the private channel before spawning, close the scheduler over the receiver, drain it inside `step`.
2. `spawn_scheduler(name, scheduler)` per scheduler; return `Engine { schedulers, info: EngineInfo { kv_capacity, servable_len }, lora }` — the required-metadata fields are the checklist, and `lora: Some(client)` only when the line actually serves adapter control.
3. `ModelLine::launch` returns `LaunchedEngine::Stepped(engine)`. `pegainfer-sim` is the CPU-only reference (`SimScheduler` in `pegainfer-sim/src/lib.rs`); it has no `ModelLine` and hands the `Engine` to `vllm::serve` directly.

The contract's own invariants are tested in `emitter.rs`/`driver.rs` tests; the qwen3 adapter's contract tests (`frontend_adapter/tests.rs`) are the reference for testing a model's protocol behaviour end to end with a fake executor. GPU integration tests drive the contract through `pegainfer-qwen3/tests/common/harness.rs`.

## The legacy handle contract (migration pending)

`request.rs`/`event.rs`/`sink.rs`/`kv.rs`/`handle.rs` still carry the previous generation: `launch -> EngineHandle`, per-request `TokenSink` events (`Scheduled … Token* … terminal` by convention), send-failure-as-cancellation. glm52, qwen35, kimi-k2, deepseek-v2-lite, and gemma4 launch through it (`LaunchedEngine::Handle`), and the vllm stack keeps both bridge paths (`bridge.rs` for handles, `bridge/stepped.rs` for step engines). KV-prefix resolution (`KvPrefix`, `submit_resolved`) currently exists only on the legacy path; fold it into the step contract when the first offload-capable line migrates.

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
| `TokenSink` send-failure-as-cancellation (qwen3 path) | `RequestControl::abort` flag, observed via `QueuedRequest/ActiveRequest::is_aborted` |
| `EngineCommand` + `EngineHandle` LoRA control methods | `LoraClient` over the model crate's private pre-spawn channel (`Engine.lora: Option<LoraClient>` is the capability); idle-drain policy lives in the qwen3 adapter (`pending_control` + `post_control_deferred`) |
| Per-request event-order convention (`Scheduled` before tokens, one terminal, enforced by hand) | typestate handles + emitter: illegal orders don't compile; drop bombs turn scheduler bugs into `Failed` terminals instead of client hangs |
| KV block-event feed (`KvBlockEvent`/`KvStoredBlock`, `EngineHandle::take_kv_events`, qwen3 `ExecutorKvEvents` + `enable_kv_events` plumbing) | none — it never had a consumer. Rebuild from git history when a cache-aware router lands; the natural re-home is a `RequestUpdate`/load-feed sibling on the step contract |
| Qwen3 `scheduler/kv_events.rs` pump | same as above |

## Next step

Migrate glm52 onto the step contract (second pilot; brings P/D and EP multi-scheduler requirements), then qwen35/kimi-k2/deepseek-v2-lite, then delete the legacy contract modules and `LaunchedEngine::Handle`.
