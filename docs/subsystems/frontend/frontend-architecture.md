# Frontend architecture: pegainfer-frontend and the engine boundary

**TL;DR:** One crate — `pegainfer-frontend` — owns everything north of the model schedulers: the engine request/event contract (formerly `pegainfer-engine`), the vLLM protocol stack (formerly `pegainfer-vllm-frontend`, now the `vllm` module), and the `ModelLine` dispatch trait. All six model lines implement `ModelLine` in their own crates; `pegainfer-server` is a ~250-line pure-dispatch binary (the `ModelType` enum, `detect_model_type`, the `load_engine` match, and the `consumed_args` table are gone). **Next step: prototype a `dynamo` module as a second protocol stack.**

Last touched: 2026-08

## The boundary, in one sentence

An engine is a function `launch(model_path, options) -> EngineHandle` whose semantics are: *give me a channel I can push `(GenerateRequest, KvPrefix)` into, and I guarantee every request's `TokenSink` receives a well-formed `Scheduled … terminal` event sequence.* Tokenizer, chat templates, HTTP protocol, metrics, LoRA routing all live north of the channel; KV, batching, CUDA all live south. The contract contains no CUDA types — `KvPrefix`'s anti-eviction hold is a `Box<dyn Any + Send>` for exactly this reason.

## Crate layout

```
pegainfer-frontend
├── engine/            # the contract, split by cohesion and re-exported flat
│   │                  #   from `engine/mod.rs` (`engine::X` paths unchanged):
│   ├── request.rs     #   GenerateRequest (one request in)
│   ├── event.rs       #   TokenEvent, FinishReason, TokenLogprob, RequestTag, unix_now_s
│   ├── sink.rs        #   TokenSink / RequestAbortReason over the shared tagged channel
│   ├── kv.rs          #   KvPrefix, SubmittedRequest, KvCapacity, KvStoredBlock, KvBlockEvent
│   ├── control.rs     #   LoRA control plane (EngineCommand & friends)
│   └── handle.rs      #   EngineLoadOptions/EpBackend (launch), EngineHandle: routing,
│                      #     LoadSnapshot feeds, join-on-drop
├── sampler.rs         # SamplingParams
├── parallel.rs        # ParallelConfig
├── tracing_state.rs   # global tracing on/off flag (frontend + schedulers both read it)
├── model_line.rs      # ModelLine trait + ModelLineRegistry + SharedArgs (dispatch seam)
└── vllm/              # protocol stack #1: vLLM EngineCore impersonation over ZMQ
    ├── mod.rs         #   serve_* entry points, vllm_server::Config assembly
    ├── bridge.rs      #   LocalEngineBridge: handshake, intake, burst demux, stats
    ├── wire.rs        #   EngineCoreSamplingParams <-> SamplingParams translation
    ├── lora.rs        #   /v1/{load,unload}_lora_adapter + adapter-name rewrite layer
    └── request_contract.rs  # GLM5.2 prefill-only route guard
```

Dependency direction: `pegainfer-frontend ← pegainfer-core ← model crates ← pegainfer-server` (thin bin). The frontend never depends on core or on model crates — that is what keeps the contract CUDA-free and lets the server binary hold the only model dispatch. Every model crate imports the contract as `pegainfer_frontend::engine` (the old `pegainfer_core::engine` re-export shims are gone).

Trade-off accepted knowingly: model crates now pull the vllm-server/axum/zeromq tree into their build graph. That is the price of "one frontend crate"; if model-crate compile times become painful, the escape hatch is splitting the contract back out, not feature-gating the stacks.

## How a request crosses the boundary

**Downstream (request in).** The protocol stack calls `EngineHandle::submit` / `submit_resolved`. The handle is a router plus metadata bag: it picks a scheduler partition (a resolved `KvPrefix` binds the request to the rank holding its blocks) and pushes into that partition's unbounded submit channel. **The model crate creates this channel in `launch`**, keeps the receiver in its scheduler thread, and hands the senders to `EngineHandle::new_with_join_handles`. Submission never blocks and never fails for capacity reasons — admission control is the scheduler's job, expressed as a `Rejected` event, not a submit error.

**Upstream (tokens out).** Direction of ownership flips: **the protocol stack creates the event channel** — one shared `TokenStreamReceiver` for all requests — and wraps a per-request `TokenSink` (tag + shared sender + abort flag) into each `GenerateRequest`. The stack demuxes by `RequestTag` (vllm: `dispatch_burst` folds each scheduler step into one ZMQ message).

**Cancellation** is not channel teardown: the stack flips the sink's `RequestAbortReason` (`AtomicU8`), the scheduler polls `is_cancelled()` and retires the request on its next step.

### Event-sequence contract (currently by convention, not by type)

Per request, the scheduler must emit:

1. `Scheduled` first — carries queued/scheduled timestamps and `cached_tokens`; the metrics path depends on it arriving before any token.
2. `PromptTokens` (echo only), then `Token`\*.
3. Exactly one terminal event: `Finished` | `Error` | `Rejected`. Nothing after it.

Both existing translators (vllm `wire.rs`; and dynamo `convert.rs`, now deleted — see below) fold streams assuming this order. It is enforced by hand at every `Finished` call site today; if a scheduler double-terminates, the failure shows up as protocol corruption in the frontend, not at the source. A `debug_assert` in `TokenSink` (terminal-then-anything panics) is the cheap hardening when it next bites.

## ModelLine: what a new model provides

`model_line.rs` defines the dispatch seam. A model crate implements `ModelLine` (each crate has a `model_line.rs` exporting `pub static MODEL_LINE`):

- `name()` — family name for logs and errors.
- `probe(config_json) -> Result<(), String>` — claim or reject the model directory by its identity fields plus the crate's `probe_config_json`; exactly one registered line must claim a config.
- `augment_cli(cmd)` — the line's *exclusive* CLI section (a private `#[derive(clap::Args)]` struct). The registry diffs the command before/after to learn which arg ids belong to the line, so ownership needs no separate table. Flags shared by several lines (`--tp-size`, `--kv-offload`, …) live in `SharedArgs` in the frontend; a line opts into each via `consumed_shared_args()`.
- `validate(ctx, provided) -> Result<(), CliError>` — the line's cross-flag rules (e.g. GLM5.2's topology/dp/tp matrix, Qwen3's batch-invariant exclusions). Runs after the registry's consume-or-reject pass.
- `serve_plan(ctx) -> Result<ServePlan, CliError>` — what the HTTP frontend must know *before the engine finishes loading*: scheduler partition count (one engine identity per partition is registered during load; checked post-launch against `EngineHandle::scheduler_partition_count`), prefill-only mode, and LoRA route enablement.
- `launch(ctx) -> anyhow::Result<EngineHandle>` — assemble the crate's option struct from `SharedArgs` + its own flags, spawn scheduler threads, attach handle metadata, return the handle.

Errors at this boundary are typed (`thiserror`): `DetectError::{NoMatch, Conflict}` — the server branches on `NoMatch` to append a "rebuild with --features X" hint — and `CliError::{UnconsumedFlag, Rule}`. `launch` stays `anyhow`: its failures are deep context chains nobody branches on.

All six lines are onboarded. `pegainfer-server/src/main.rs` is the whole server: build registry → merged clap command → detect from config.json → validate (registry consume-or-reject, `SharedArgs::validate`, line rules) → `serve_plan` → `launch` on a blocking thread → pick the serve path (LoRA / prefill-only / normal) from the plan. Adding a model line = write `model_line.rs` in the crate + one registry entry + one Cargo feature.

## Protocol stacks

**`vllm` (current default, fleet-proven).** Impersonates a vLLM EngineCore process over in-process ZMQ/msgpack because upstream `vllm-server` assumes the engine is a separate process. HTTP routes, OpenAI types, tokenizer, chat templates, and Prometheus all live in the external `vllm-server`/`vllm-metrics`/`vllm-text` crates. The per-step msgpack round-trip and the impersonation handshake are pure overhead for our single-process deployment — tolerated because the stack is validated at EP16/EP32 scale.

**`dynamo` (planned second stack).** dynamo's `lib/llm` has an in-process path that removes the wire protocol entirely: `EngineConfig::InProcessTokens` + `run_input(drt, Input::Http, …)` gives axum → preprocessor (chat template + tokenize) → *your engine as a function call* (`AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, _>`) → detokenize → streaming tool-call/reasoning parsing → SSE. `DistributedConfig::process_local()` runs it with no etcd/NATS. The deleted `pegainfer-dynamo-backend/src/convert.rs` (see git history of this branch's parent) already contained the full `PreprocessedRequest → GenerateRequest` / `TokenEvent → LLMEngineOutput` translation — resurrect it as the adapter core. Known risks before committing: the `InProcessTokens` rename is recent (crates.io releases may still say `StaticCore`), the in-process path is not dynamo's flagship (maintained via tests + the python echo example), and the dep tree compiles fat (unconditional tonic/swagger/object_store; needs libzmq + protoc). Decision gate: prototype, run vllm-bench A/B against the vllm stack, let TTFT/step-overhead numbers pick the default.

## What was deleted, and the heirs

| Deleted | Heir |
| --- | --- |
| `pegainfer-engine`, `pegainfer-vllm-frontend` | merged into `pegainfer-frontend` (git mv, history preserved) |
| `pegainfer-core` re-export shims (`engine`/`sampler`/`parallel`), `pegainfer-server` shims (`scheduler`/`sampler`/`vllm_frontend`, incl. the `SchedulerHandle` alias) | direct `pegainfer_frontend::` imports everywhere |
| `pegainfer-dynamo-frontend`, `pegainfer-dynamo-backend` | future `dynamo` module in `pegainfer-frontend`; `convert.rs` translation logic recoverable from git |
| `bench_serving` bin (3.1k lines, in-process, bypassed the serving path) | HTTP-level benching: `scripts/bench_http_serving.py` + external vllm-bench; see [bench-regression](../../conventions/bench-regression.md) for the retired snapshot gate |
| `glm52_step_bench` bin, `scripts/run_snapshot_benchmark.sh`, `scripts/sweep_mb8.sh` | none — step-level microbenching lives in `pegainfer-glm52/benches/` (its `kernel_lab` docstring still mentions the old bin; harmless) |

## Next step

Prototype the `dynamo` in-process protocol stack (`EngineConfig::InProcessTokens`, resurrecting the deleted `convert.rs` translation logic from git history) as a second module beside `vllm`, then run a vllm-bench A/B to decide the default.
