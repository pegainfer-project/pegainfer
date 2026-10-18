# Direction

> **TL;DR:** pegainfer is not chasing a super abstraction that swallows every model. The next-stage shape is a set of reusable, stable infrastructure plus per-model engines with clear boundaries. Share what is stable; let go of what will truly fork. Optimize for keeping each model's context coherent for both humans and LLMs to read in one pass.
>
> **Origin:** "One Size Can't Fit All" (2026-05-05).

## Why we resist the super-abstraction reflex

Supporting a new model is not the hard part — it's config, weights, ops, accuracy, perf, in some order. The hard part is what happens after: every model added breaks parts of the previous shape, because at the code level the models *are* genuinely different.

- Qwen3 — dense full attention
- Qwen3.5 — 24 linear + 8 full attention, with recurrent state
- GLM5.2 / Kimi-K2 — MLA compressed KV, MoE routing, FP8 / Marlin INT4 GEMM, EP communication

They're all "LLM inference" at the marketing level. At the code level, state layout, scheduler, kernel DAG, and communication pattern all diverge. A single super-scheduler that subsumes them ends up as everyone's branches inside one shared blob, and no one dares touch it.

So the working principle is **share the stable parts; allow the fork-prone parts to be per-model**.

## Where the boundary goes

### Shared infrastructure (stable, cross-model)

- **Frontend** — HTTP/OpenAI surface, tokenizer, chat template, stop sequences, streaming, logprobs, usage. Bridged via `vllm-frontend-rs` plus a local engine-core adapter. The frontend should never know what model, parallel strategy, or attention pattern is behind it.
- **Runtime primitives** — `pegainfer-core` owns the generation contract (`ModelForward`, `GenerationState`), sampler, CUDA Graph state, weight loader, page/KV pool primitives. Per-model crates depend on it, not on root.
- **Kernel layer** — `pegainfer-kernels` owns kernel sources, FFI, and wrappers. cuBLAS, FlashInfer, Triton AOT, handwritten CUDA all live here. Open to extension; we don't fork third-party kernels lightly.
- **Data plane** — PegaFlow stays the KV data plane (RDMA transfer, SSD offload, prefix dedup), not folded into model internals.
- **Tooling** — benchmarks, profiling, eventual tracing/simulator infrastructure.

### Per-model (fork-allowed, model crate owns)

- Config interpretation, weight loading, state layout
- Prefill / decode / unified-step execution
- Scheduler — batching policy, KV admission, recurrent-state handling, slot compaction
- Kernel DAG — which kernel for which shape on which GPU
- TP / EP / PP / P-D disaggregation strategy
- Accuracy alignment with reference (HF / official)
- Per-model benchmarks

The rule of thumb: if a concept exists in only one model's mental model (`RecurrentState`, MLA compressed KV, EP routing), it does not earn a place in the shared layer.

## Why per-model crates beat one universal scheduler

A "general scheduler" that absorbs every model accumulates per-model branches anyway — full attention, linear attention, MoE, P-D, RL workload — and the result is a giant shared context where everyone is afraid to make changes. Maintenance becomes a coordination problem instead of a code problem.

Per-model crates make the trade explicit:

- Duplication is OK when the duplicated thing is genuinely different — three similar lines beats a premature abstraction with three escape hatches.
- Each model's full context — config, weights, scheduler, kernel selection, accuracy story — fits in one place. Humans and LLMs can both load it in one read.
- New models start from a working reference (another model crate), not a fresh runtime. Reused infrastructure stays small enough that the new model's complexity is contained inside its own crate.

This is the inverse of the historical instinct ("avoid duplicate code at all cost"). With cheap LLM-assisted authoring and large context windows, the more important property is **a self-contained, coherent per-model story**, not a maximally-DRY framework.

## The long-term loop: ledger → simulator → tracing

A kernel library answers "what kernels exist." It does not answer "which kernel for this model, this shape, this GPU" — and that question dominates performance and correctness for every model we add. The intended closed loop:

1. **Kernel ledger** — for each kernel: supported dtype/layout/SM/shapes, measured latency and bandwidth on specific GPU/CUDA/commit, correctness reference and tolerance, and where it sits in each model's DAG (phase, layer, op instance). Maintained machine-readably so an LLM can navigate from config to kernel without guessing.
2. **Simulator** — given a model's kernel DAG and the ledger, predict TTFT/TPOT/throughput offline. Primary value is *explaining* performance, not predicting an exact number. "TPOT is 20ms" is meaningless; "the 20ms breaks down into N ms of GEMV memory bandwidth, M ms of attention compute, K ms of launch overhead" is actionable. Also useful for finding workload sweet spots.
3. **Request tracing** — low-overhead online traces from frontend through scheduler through GPU steps. When the simulator says 100ms and a real batch returns 500ms, tracing tells you which span explains the gap. Long-tail p99 spikes typically expose structural problems (topology, NUMA, scheduler pathology) more than they expose hot-kernel bugs.

The closed loop matters: offline expectation (simulator) + online measurement (tracing) + the bridge between them (kernel ledger) is what lets the engine stay honest as model coverage grows.

## What this is not

- Not "Rust rewrite of vLLM." vLLM optimizes within Python; SGLang aspires to Rust but hasn't started. pegainfer's structural choice — per-model engines on shared infrastructure — is orthogonal to language.
- Not "general-purpose LLM serving stack." We are good at specific model families with clear ownership; that is the unit of scaling, not "support every model on every backend."
- Not a permanent freeze on shared layers. The boundary moves when a new model genuinely demands it (e.g., MoE all-to-all may pull communication primitives into the shared layer). It does not move because abstraction feels nicer.

## Next-step intent

Direction, not task list. Concrete current work lives in `execution.md`.

- **Per-model crate is the unit of model work.** Qwen3 and Qwen3.5 already live as independent crates; Kimi-K2 and GLM5.2 follow the same pattern. New models start by copying the closest existing crate as a reference, not by extending a shared abstraction.
- **Build the ledger → simulator → tracing loop incrementally.** Start with model-owned kernel plans (Qwen3 has one); extend across models; then a cross-model index; then a simulator MVP against one model; then tracing.
- **Treat each new model as a boundary test.** If something keeps re-appearing per-model, it might belong in the shared layer. If a "shared" thing keeps needing per-model escape hatches, it should fork. The boundary moves on evidence, not aesthetics.
