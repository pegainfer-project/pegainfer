# Bump vllm-* frontend crates

> **TL;DR:** Bumped `vllm-*` from `cc706b05` to `d3e2888c`. Mac sim 3200/c=320: 0-fail both sides; steady-state req/s 402 vs 401; TPOT p50 still 12.11ms. Handshake/output/config field fills only.
>
> **Last touched:** 2026-08

## Preparation

- **Read**:
  - `docs/index.md` — frontend docs live under `subsystems/frontend/`.
  - `docs/subsystems/frontend/frontend-architecture.md` — HTTP/tokenizer/metrics live in upstream `vllm-server` / `vllm-text`; pegainfer only impersonates EngineCore.
  - `docs/subsystems/frontend/simulated-inference-engine.md` — sim is a CPU-only step-contract harness; TTFT/TPOT are configured delays.
  - `docs/subsystems/frontend/sim-high-concurrency-bench.md` — rust `vllm-bench` against sim, `--ignore-eos` + `--extra-body '{"min_tokens":null}'`.
- **Relevant history**:
  - Workspace pins all five crates to the same vllm.git rev (`cc706b05`). Mixing revs splits type identity.
- **Plan**:
  1. Baseline: the user's 128-in / 64-out / 3200 prompts / c=320 `vllm-bench` against the already-running sim on :8732; save JSON.
  2. Bump all five `vllm-*` workspace git deps to the same latest rust-crate-touching rev; `cargo update`.
  3. Rebuild `pegainfer-sim`, restart on :8732, replay the same bench, `vllm-bench --compare`.
  4. If compile + 0-fail + no surprising latency cliff: open a bump PR.
- **Risks / open questions**:
  - Upstream EngineCore / sampling / tokenizer API drift will fail `pegainfer-frontend` compile.
  - Sim TPOT is a 12ms floor, so this A/B mostly measures frontend+bridge overhead in TTFT / req/s, not kernel time.

## Execution Log

### Baseline (cc706b05, sim already running on :8732)

Command: 128-in / 64-out / 3200 prompts / c=320, `--ignore-eos --extra-body '{"min_tokens":null}'`. Saved `/tmp/pegainfer-sim-bench/baseline-cc706b05.json`.

| | |
| --- | --- |
| ok/fail | 3200 / 0 |
| duration | 9.01s |
| req/s | 355 (steady-state 402) |
| out tok/s | 22.7k (steady-state 26.8k) |
| TTFT p50 / p90 / p99 | 9.18 / 12.32 / 18.93 ms |
| TPOT p50 | 12.11 ms (matches `--tpot-ms 12`) |
| E2EL p50 | 772 ms |

### Upstream delta: `cc706b05` → vllm `main` (`d3e2888c`, 2026-08-25)

~6 weeks of `rust/` commits. Crate names we pin are unchanged (`vllm-{server,text,chat,tokenizer,engine-core-client}`); workspace gained `vllm-bench`, `vllm-build-info`, `vllm-tracing`.

**Will fail our compile (EngineCore impersonation):**

- `EngineCoreReadyResponse` grew required discovery fields: `effective_data_parallel_size` (serde-rename of `data_parallel_size`), `tensor_parallel_size`, `pipeline_parallel_size`, `decode_context_parallel_size`, `data_parallel_rank`, `max_num_seqs`, `max_num_batched_tokens`, `instance_id`, `supports_lora`, `max_loras`, plus optional KV-events / RL / sleep / draft-weight flags (`#52575`, `#53204`, `#52031`, `#50033`).
- `EngineCoreOutput` added defaulted `mm_cache_miss_hashes`, `new_sampling_mask`, `spec_decode_metrics` (`#46747`, `#49577`, `#48915`). Same pattern as `#898`'s `ec_transfer_params: None`.
- `EngineCoreRequest` added `session_id` (`#48048`). Our destructure uses `..`, so receive path is fine.

**Frontend behaviour we actually serve through:**

- Kimi K3: rust frontend land (`#50104`), tool rendering (`#50540`), reserved-marker strip (`#52889`), `reasoning_effort="none"` (`#53043`), conversation-unique tool-call ids (`#50420`).
- GLM-5.2 chat-template parity (`#51426`); Qwen parser auto-detect (`#51169`); DeepSeek V4 reasoning-effort prompts (`#50580`).
- Tokenizer: ordinary-text encoding (`#49992`); MiniJinja 2.0 → 2.24 (`#51235`); standalone renderer (`#50289`); `--max-model-len` stays engine-owned (`#49944`).
- OpenAI surface: per-request spec-decode accept stats (`#48915`); request priority HTTP header (`#51089`); `--generation-config vllm` (`#53044`); `n > 1` rejected on `/inference/v1/generate` (`#52844`).
- Build: `protoc` → pure-Rust `protox` (`#52892`) — relevant on Mac.

**Mostly unused by pegainfer (gRPC / pooling / RL / multimodal):** world-size over gRPC, DP-rank routing, LoRA capability advertise, abort RPC, image inference, pooling bench endpoints, in-tree `vllm-bench`.

### After (`d3e2888c`)

Same command. Saved `/tmp/pegainfer-sim-bench/after-d3e2888c.json`.

`vllm-bench --compare` (A=baseline, B=after): 3200/0 both sides. Steady-state req/s 401.67 vs 401.47. TPOT p50 12.11 both. Median TTFT 9.18 both. Headline req/s +5.7% is ramp-up duration noise (9.01s → 8.52s); P99 TTFT +27% (18.9 → 24.1ms) is a one-run tail — steady-state P99 TTFT 16.0 → 14.1ms.

Compile glue: handshake discovery fields, `EngineCoreOutput` three new defaults, `PrefillStats.num_cache_creation_tokens=0`, `Config.{generation_config,limit_mm_per_prompt}`, `TransportMode::Bootstrapped.data_parallel_size`, `EngineId` now `u16`.

Tests: `pegainfer-frontend --lib` 65, `pegainfer-sim --lib` 6, `frontend_e2e` 13.

## Debrief

- **Outcome**: Bump compiles and serves. Sim A/B is flat at the 12ms TPOT floor; no frontend cliff.
- **Pitfalls encountered**: Not lockfile-only. Handshake grew required fields; `EngineId::from_engine_index` narrowed to `u16`.
- **Lessons learned**: Next bump, diff `EngineCoreReadyResponse` / `Config` / `PrefillStats` first — that is where #898's `ec_transfer_params` pattern repeats.
- **Follow-ups**: Handshake reports `supports_lora: false` / `max_loras: 0`. HTTP LoRA routes are still ours; gRPC LoRA advertise is unused. Revisit if a client starts reading the ready payload.
