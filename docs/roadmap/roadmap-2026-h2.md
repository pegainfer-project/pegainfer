# Roadmap 2026-H2

> **TL;DR:** Second-half 2026 plan. Supersedes the five-workstream roadmap issue (#203), most of which has landed (prefix cache, KV offload, speculative decoding) or drifted. Priorities: make the website the real product surface (recipes-style per-model pages, CLI reference, hardware matrix), formalize model maturity tiers with Qwen3-4B as the first Stable line, and finally wire observability. GLM5.2 large-MoE bring-up is the active mainline and doubles as the boundary test for shared MoE infrastructure. P/D disaggregation (NIXL-compatible) is design-first, Later.
>
> `direction.md` (why per-model engines, where the shared boundary sits) and `execution.md` (what is in flight this week) remain the day-to-day source of truth. This file is the half-year view above them. Entries carry **Now / Next / Later** and move or get deleted as reality changes. No dates.

## Who this is for

pegainfer's near-term market is narrow and deliberate: **small-to-mid models, single node, coding-agent backend, fast startup, small footprint**. That matches both our measured strengths (771 MB binary vs multi-GB images; ~3 s vs ~70 s cold start; Qwen3-4B beating vLLM across the QPS sweep on consumer Blackwell) and our contributor base, which is individual developers with one good consumer GPU. The large-MoE / multi-node story is being built right now through the GLM5.2 mainline — but until that line ships, we position the project by the individual-developer lane.

---

## Now

### 1. Website as the product surface

Today the site has three pages of real content while the engine's actual capabilities live scattered in repo docs. The website (pegainfer-project/website) becomes the canonical user-facing surface, in the style of [recipes.vllm.ai](https://recipes.vllm.ai/): one verified, copy-pasteable recipe per model line, kept current as flags change.

- **Per-model recipe pages.** For each served line: exact launch command, required hardware, the CLI flags that matter for that model (`--kv-offload`, `--dflash-draft-model-path`, TP flags, …), expected TTFT/TPOT on the tested GPUs, and a benchmark snapshot. Only Qwen3-4B has a page today; Qwen3.5 and DeepSeek-V2-Lite are next. Every command on a page is run before it is committed — same rule as repo docs.
- **CLI / server reference.** A maintained page for `pegainfer-server` args and env vars (`PEGAINFER_CUDA_SM`, `PEGAINFER_TRITON_PYTHON`, …). This must not rot: once the page exists, a flag change without a website update is an incomplete PR.
- **Hardware support matrix.** Which GPUs are tested, expected performance. The numbers already exist in `docs/benchmarks/`; they are just not published.
- **Troubleshooting page** distilled from pitfalls already recorded in repo docs: driver floor, the cuBLAS 12.9 N=1025 cliff, build with CUDA ≥ 13, SM autodetection failures.
- **Install path stays source-build** for now. A prebuilt runtime Docker image is deliberately *not* on this roadmap — a per-SM binary matrix is a maintenance treadmill we don't want yet. Instead: a **dev Dockerfile / devcontainer** (rust toolchain + nvcc + Triton venv) so a contributor's first `cargo build --release` cannot fail on missing prerequisites, and a getting-started page good enough that source build is a ten-minute path, not an afternoon.

*Done when:* a stranger with a supported GPU goes from the website to a served completion in under ten minutes, and the recipe pages are trusted enough that we link them in issue replies instead of re-typing commands.

### 2. Model maturity tiers, with Qwen3-4B as the first Stable line

Formalize what "supported" means, so users know which model to run and every line has a concrete, checkable next milestone:

| Tier | Contract | Lines today |
| --- | --- | --- |
| **Stable** | Accuracy gate always green; bench-regression tracking per `conventions/bench-regression.md`; serving flags don't break without notice; a website recipe page; the reference implementation new models copy from | Qwen3-4B |
| **Maturing** | Accuracy gate exists and passes; perf ledger active; serving contract may still move | Qwen3.5-4B, DeepSeek-V2-Lite, Kimi-K2 |
| **Bring-up** | Under construction; correctness fixtures may be partial; no serving promises | GLM5.2 |

Qwen3-4B earns Stable on evidence: golden-gate accuracy, full-sweep serving wins vs vLLM, prefix cache + KV offload + speculative decoding + TP all landed, regression tracking in place. Being Stable also means it is the **scaffolding reference**: a new model starts by copying the closest crate (per `direction.md`), and Qwen3-4B is the canonical one to copy.

Tier definitions land in `CONTRIBUTING.md`; each model's README and website page states its tier. Promotion criteria must be objective and checkable — "take a line from Maturing to Stable" should be a self-contained goal someone can pursue and finish.

Supporting mechanics for contributors: label every open issue with its hardware floor (`hw:none` / `hw:1-gpu` / `hw:2-gpu` / `hw:8-gpu`) — most of Qwen3/Qwen3.5, the whole frontend, and everything on `pegainfer-sim` (the CPU-only simulated engine) needs at most one consumer GPU. We do not add another 8-GPU model line this half; the existing three already exceed what anyone but the maintainer can verify end-to-end.

### 3. Observability: metrics + request tracing

The oldest unpaid debt from #203, a well-bounded subsystem, and mostly `hw:1-gpu` work. `prometheus`, `opentelemetry-otlp`, and `tracing-opentelemetry` are declared in the workspace `Cargo.toml` and wired into nothing.

- **`/metrics` endpoint** (Prometheus): TTFT, TPOT, queue depth, running/waiting batch size, KV-block utilization, prefix-cache hit rate, per-phase step time.
- **OTLP request tracing:** spans from frontend → scheduler → forward step, low-overhead, off by default.
- **A committed Grafana dashboard JSON** so the metrics are usable on day one.

Process rule (unchanged from #203): **agree the metric names in a short issue before the wiring PR.** Metric names are a public interface.

This is also the tracing leg of the `direction.md` ledger → simulator → tracing loop — the same spans that serve a dashboard feed the simulator-vs-measured gap analysis later.

*Done when:* a `vllm bench serve` run against Qwen3-4B is fully explainable from the dashboard alone.

### 4. GLM5.2 large-MoE mainline

The original DP1/EP8 bring-up campaign is complete and has grown into Blackwell-only EP4/EP8/one-domain EP-N decode, TP4 prefill-only, and cross-engine P/D serving paths (Hopper and decode TP/LL removed). Current capabilities and promotion blockers live in `docs/models/glm52/serving-status.md`.

It is also the deliberate **boundary test** for shared infrastructure: large MoE forces the question of which DeepGEMM / DeepEP / FlashMLA substrate is genuinely cross-model (`pegainfer-kernels::moe`) versus model-local. We do not refactor scaffolding speculatively — the boundary moves when GLM5.2 provides evidence it must (per `direction.md`). Any extraction of shared MoE/MLA primitives falls out of this line, not out of a standalone "clean up the scaffolding" project.

---

## Next

### 5. Coding-agent frontend: verified tool-calling round-trip

Carried over from #203 W3, still the right goal: an pegainfer endpoint should be a drop-in OpenAI-compatible backend for Claude Code, opencode, and similar. The frontend already routes `/v1/chat/completions` and carries a `tool_call_parser`; the gap is verification, not construction.

- An integration test that drives a **real tool-call round-trip** in the formats these agents actually emit (streamed included). Protocol layer testable on `pegainfer-sim`.
- **Structured / guided output** (JSON-schema / grammar logits masking, xgrammar class) — the hard dependency for reliable tool calls from small models.
- Sampling parity (#490: `min_p`, penalties, per-request `seed`, `n>1`) folds in here.
- A "point your agent at localhost" recipe page on the website.

None of this needs more than one GPU.

### 6. One new single-GPU model line

The next model we add should be runnable by contributors, so the model roster matches the hardware people actually have. Candidate: a small MoE (Qwen3-30B-A3B class — also gives the shared-MoE substrate a single-GPU testbed). The higgs-audio line (#395–#400, #864) moved to pega-omni, which serves speech models. A new line needs an accuracy gate and a bench from day one (per the model-roadmap template in #203's appendix).

### 7. Kernel ledger → simulator MVP

Carried from `execution.md`. ~~Qwen3 and Qwen3.5 already have `kernel_plan()` descriptors~~ (the descriptors had no readers and were removed in #743); the cross-model ledger format and a Qwen3-4B simulator MVP come after observability lands, because the tracing spans are the measurement side of the loop. Value is *explaining* a TTFT/TPOT number, not predicting it.

---

## Later

### 8. P/D disaggregation (NIXL-compatible), design-first

Prefill/decode disaggregation with KV transfer between workers, speaking the same NIXL semantics vLLM/Dynamo use, so pegainfer workers drop into that ecosystem instead of inventing a transport protocol. PegaFlow remains the data plane.

Design-first for the same reason the KV-cache crate was: it is load-bearing and under-specified. **One design issue** — not two parallel efforts (an earlier per-model P/D handoff design doc was retired with its model line; its page-ownership/lease ideas live on in the issue). Implementation needs multi-GPU/multi-node verification, so it stays Later until the design is settled and the GLM5.2 line supplies a real workload.

Related positioning: **multi-instance concerns (routing, P/D orchestration) are delegated to Dynamo; pegainfer is a first-class worker.** The `pegainfer-dynamo-backend`/`-frontend` crates and the verified KV-aware routing result (follow-up-turn TTFT ~45 ms vs ~165 ms round-robin) already point this way.

### 9. Foundations carried from #203 W5

- **GPU CI runner** for accuracy/e2e gates (CPU CI already covers fmt/clippy/pure-logic tests; the correctness signal contributors see is still manual).
- **Typed errors + panic policy** on `pegainfer-core` public surfaces (replace `anyhow` on library boundaries; panic on our invariants, typed errors for caller preconditions).

---

## What we are explicitly not doing

- A prebuilt runtime Docker image (per-SM binary maintenance treadmill; revisit if demand shows up).
- Chasing vLLM/SGLang on model breadth or ecosystem surface. Depth in named lanes over coverage.
- Weight quantization as a general feature (BF16 stays the default; Marlin INT4 exists where a model line demands it, e.g. Kimi-K2).
- Multimodal/VLM (speech and image models are served by pega-omni), pipeline parallel, embedding/rerank endpoints — unless a concrete need lands.
- A universal scheduler or KV abstraction. The per-model boundary from `direction.md` stands; it moves on evidence from new model lines, not on aesthetics.

## Tracking issues to open

- Website tracking: recipe pages, CLI reference, hardware matrix, troubleshooting — `hw:none`
- Dev Dockerfile / devcontainer — `hw:none`
- Tier definitions → `CONTRIBUTING.md` PR
- Observability: metric-name agreement (short, interface-only), then wiring
- P/D + NIXL design (absorbs the existing V4 handoff doc) — design discussion, no PR
- New single-GPU model line

Close #203 with a pointer here once this file merges.
