# Execution

> **TL;DR:** Current state and immediate next steps. No timeline — entries move through `In progress → Next → Open` (or get deleted) as work lands. Vision lives in `direction.md`; this file only tracks what's actually being worked on.

## Cross-model infrastructure

These are the shared layers — frontend, runtime, kernels, ledger/simulator/tracing. Each model crate consumes them.

### In progress

- **Model-owned kernel plans.** ~~Qwen3 already carries a light `kernel_plan` mapping prefill/decode/unified phases → Rust wrappers, FFI symbols, and CUDA/Triton/cuBLAS backends.~~ The `kernel_plan` descriptors (qwen3 + qwen35) never acquired a reader and were removed in the hawk dead-public sweep (#743); the self-describing-model idea continues via `kernel_manifests/` + kernel-op reports instead.
- **Frontend polish.** `vllm-frontend-rs` is the default OpenAI surface, talking to pegainfer via a local engine-core IPC bridge. Outstanding: logprobs / prompt-logprobs translation, usage accounting, and a deliberate decision on whether the served-model-id should decouple from the tokenizer path.
- **KV data plane (pegaflow).** First concrete tier landed: `pegainfer-kv-offload::OffloadEngine` wraps in-process `pegaflow-core` as a host-tier ("L2") KV backend (mechanism), with per-model schedulers owning the residency policy (no universal connector trait — policy/mechanism split per `direction.md`). Shipped on Qwen3-4B full-attn (#316). Next candidate: Kimi-K2 MLA (layout zero-impedance, reuses the same connector pattern). SSD/RDMA tiers and DeepSeek sparse remain unrealized scope. See `subsystems/runtime/pegaflow-offload-integration.md`.

### Next

- **Cross-model kernel ledger.** Per-kernel record: dtype/layout/SM/shape support, measured latency/bandwidth (GPU + CUDA + commit pinned), correctness reference and tolerance, and DAG position in each consuming model. Machine-readable so LLMs and humans both navigate from `config.json` to kernel without guessing.
- **Simulator MVP.** Given a kernel DAG plus the ledger, predict TTFT/TPOT/throughput offline. Primary value is *explaining* a measured number, not predicting it exactly. Pick one model (Qwen3-4B) as the first target.
- **Request tracing.** Low-overhead spans from frontend → scheduler → forward steps. Bridges to the simulator: when measured ≠ predicted, tracing shows which span owns the gap. OTLP export so it plugs into existing collectors.

### Open

- Whether kernel ledger lives entirely per-model (denormalized) or also gets a cross-model index. Probably both, but the index format is not designed yet.

## Models

Each model crate owns its own scheduler, kernels, accuracy story, and benchmarks. The boundary with shared infrastructure is `pegainfer-core` (runtime contract) + `pegainfer-kernels` (op layer).

### Qwen3.5-4B

**Status:** at parity with vLLM 0.18 (TTFT 234ms vs 229ms, TPOT 11.77ms vs 11.67ms). Accuracy: 11/13 exact HF match, two small-logit-drift cases remaining.

**In progress:**
- Prefill full-paged migration. Decode is fully paged through the scheduler; prefill still scatters from contiguous HND staging into paged KV before attention. Mirrors Qwen3's step-2 migration but with HD256 kernels and partial RoPE (`rotary_dim=64` of `head_dim=256`) wrinkles.

**Next:**
- Close the last two HF parity cases (small-logit drift).
- Bring Qwen3.5 into the kernel-plan shape Qwen3 uses, so the ledger has both models covered.

**Open:**
- TP support for Qwen3.5 (no current driver — the hybrid linear/full state layout adds complexity Qwen3 didn't have).

### Qwen3-4B

**Goal:** stay ahead of vLLM on serving experience while expanding the parallel-strategy and scheduling surface.

**Done:** single-request and continuous batching ahead of vLLM (decode TPOT wins at all concurrencies; QPS=2 within 2% throughput while leading TTFT, TPOT, and latency stability). TP=2 brought up end-to-end on one machine; TP=8 smoke-tested on 8×4090. Issue #85 KV-pressure hang at QPS=2 is fixed. Batched greedy decode sampling landed (#307 — one `argmax_batch_bf16_into` launch per step, not O(batch)); mixed greedy/non-greedy sampling landed in the same batched selector path (#284), with HTTP mixed-sampling evidence tracked separately by #412. **In-process pegaflow KV offload shipped (#316)** — host-tier "L2" save/restore wired into executor + scheduler, server CLI (`--kv-offload` / `--kv-offload-host-gib` / `--no-prefix-cache`) on both plain and LoRA paths; pure-L2 TTFT 195→40 ms measured. See `subsystems/runtime/pegaflow-offload-integration.md`.

**Next:**
- Explore pipeline parallelism (PP) as a complement to TP — particularly for larger model fits and multi-node layouts.
- Chunked prefill — closes the remaining ITL p99 tail under mixed prefill+decode load (pure decode already beats vLLM).
- Further perf tuning surfaces as they appear: TP=8 systematic correctness pass and scheduler-level cleanup against the controller/worker model in `models/qwen3/tp-design.md`.
