# Qwen3-4B Roadmap

> **TL;DR:** Qwen3-4B is the maturity bar of the project — continuous batching, TP=2, default-on prefix cache (#216), and the HF logits golden gate are all live — so its roadmap is sharpening, not bring-up. The #220 RoPE OOB bug is fixed (cos/sin cache sized from `max_position_embeddings`, admission rejects past the window, kernel traps an out-of-range position; gated by both an oversized-reject and an in-window >4096 IT). Since the 2026-06-04 open-set review, batched greedy sampling (#307), mixed greedy/non-greedy batched sampling (#284), and in-process pegaflow KV offload / L2 host tier (#316, pure-L2 TTFT 195→40ms) have landed in the table below. Remaining open set: zero TP correctness coverage, LoRA gated only by a zero-adapter smoke, prefix-cache observability dropped at the scheduler boundary, stale docs, and the YaRN #8 follow-up for rope-scaled checkpoints.
>
> **Last touched:** 2026-08

Tracking issue: see the `[Model] Qwen3-4B roadmap` GitHub issue. Cross-model items stay in `docs/roadmap/execution.md`; this doc owns the qwen3 line.

## Where the line stands

| Area | State | Evidence |
| --- | --- | --- |
| Batching | ✓ continuous, KV full-lifetime admission, rejection (#85 fix) | `scheduler.rs:478-510` |
| Prefix cache | ✓ default-on full-block kvbm matching (#216); 4 cache-hit replay passes in the golden gate | `executor.rs:750-751`, `tests/hf_golden_gate.rs` |
| Accuracy gate | ✓ HF bf16 golden, bs=1/batched/graph + cached replays; single-GPU, ≤256-token prompts | `tests/hf_golden_gate.rs:451` |
| Long context | ✓ fixed: RoPE cache sized from `max_position_embeddings`, admission rejects past the window, kernel traps OOB; gated by reject + in-window >4096 ITs. YaRN #8 still open for scaled checkpoints | `weights.rs:310-318`, `tests/context_window.rs`, `tests/context_window_in_window.rs` |
| Batch sampling | ✓ #307/#284: greedy rows use indexed batched argmax; non-greedy rows compact into one FlashInfer batched sampling call per step; top1 scratch sizing now comes from the kernel | `pegainfer-core/src/ops/sampling.rs`, `pegainfer-kernels/csrc/shared/flashinfer_sampling.cu` |
| KV offload (L2) | ✓ in-process pegaflow host-tier save/restore (#316); CLI `--kv-offload`/`--no-prefix-cache`, plain + LoRA; pure-L2 TTFT 195→40ms | `subsystems/runtime/pegaflow-offload-integration.md` |
| TP correctness | ✗ zero automated coverage — every test runs `device_ordinals: vec![0]` | grep `tests/` |
| LoRA | ⚠ load/unload/TP/request-level all built; only test uses a **zero adapter** | `lora.rs`, `tests/lora_smoke.rs:91-130` |
| Non-greedy sampling | ⚠ qwen3/qwen35 model behavior gates now cover `temperature` / `top_k` / `top_p`; penalties/min_p are still absent from `SamplingParams` | `tests/sampling_behavior.rs`, `pegainfer-kernels/src/ops/sampling.rs` |
| Bench snapshots | ⚠ exist but ~187 commits stale; not refreshed by #216; no mixed-load ITL profile | `bench_snapshots/` |
| PP | greenfield (aspiration only) | — |

## Roadmap

### Now

1. **YaRN for rope-scaled checkpoints (#8).** The #220 RoPE OOB fix landed scope (a): the cos/sin cache is sized from `config.max_position_embeddings`, admission crash-early rejects past the window (distinct context-length vs KV-budget reasons), the kernel `__trap`s an out-of-range position as a last-resort backstop, and the gate now covers both an oversized reject and an in-window >4096 case (`tests/context_window.rs`, `tests/context_window_in_window.rs`). That precompute is correct *only because this checkpoint has `rope_scaling: null`*. Scope (b) remains open: #8 YaRN is the prerequisite for any rope-scaled checkpoint — the precompute length must come from the scaled schedule, coordinated with the qwen3.5 sibling fix so both crates share the pattern.
2. **Batched decode sampling.** ✓ #307/#284 route all-greedy batches through indexed batched argmax and mixed batches through one compact FlashInfer batched sampling pass for non-greedy rows. Shared `pegainfer-core/kernels` work covers qwen35 too; keep the HF gate and nsys no-per-row-sampler check as the regression surface.
3. **Sampling correctness coverage.** Shared sampler tests cover seed determinism and temperature/top_k/top_p behavior, and qwen3/qwen35 each now have a model-level non-greedy behavior gate. Keep auditing the frontend for silently-dropped params (penalties, min_p are absent from `SamplingParams` entirely) — the kimi-k2 silent-greedy bug (#237) shows this class is real.
4. **Prefix-cache observability.** `cached_tokens` is computed (`executor.rs:751`) and dies at the scheduler boundary; the frontend hardcodes `num_cached_tokens: 0`. Thread it through `TokenEvent::Scheduled` into usage; log hit rate. Adjacent: #78 (streaming usage discards completion_tokens) — same usage-accounting surface.

### Next

5. **Mixed-load ITL profile, then the chunked-prefill decision.** A long prompt admitted mid-decode runs as one unbounded prefill in the unified step (no per-step token budget exists) — the documented +38% ITL p99 tail vs vLLM. Maintainer stance on chunked prefill is *conditional* (`scheduler.md`: varied-length workloads break waves naturally); so the gate is a tracked mixed-load benchmark profile first, implementation only if the tail matters for a real workload. Refresh the stale bench snapshots in the same pass.
6. **TP correctness pass.** Run the golden gate over `device_ordinals [0,1]` (skip when <2 GPUs) so the tolerances also guard sharding + all-reduce; TP=8 systematic pass after. A reduction-order or shard-offset bug is currently invisible to every gate.
7. **LoRA real-adapter accuracy gate.** The last open #173 acceptance criterion: teacher-force one real PEFT adapter against an HF reference with the golden-gate tolerances. Today base==(base+zero·LoRA) is all that's proven. The salt-isolation of the prefix cache also deserves a pinning test (adapter A's blocks must not hit for adapter B).
8. **Eviction behavioral test.** Evict-then-remiss is never exercised: register a prefix, release it, pressure the pool until eviction, assert truncated/zero match + correct recompute. kvbm-logical layer needs no GPU.
9. **Disconnect block-pinning.** After #216, a disconnected request pins its cache blocks (strong Arcs) until the next failed send — #215 is now also a KV-budget problem. Scheduler half: proactive `token_tx.is_closed()` sweep per step; folds into the server-wide #215.

### Later

- **Pipeline parallelism** — greenfield, no code; revisit when a multi-node driver appears.
- **Vocab-parallel embedding/lm_head + TP CUDA-graph** — the real open remainder of `tp-design.md`.

## Cleanup ledger

- **Issue hygiene:** #188 references a test target deleted in #194 — close as superseded by the golden gate. #203 §1 still claims qwen3 has no prefix reuse — stale since #216.
- **Dead code:** ~~`batch_decode_trace.rs` `HIDDEN_SIZE`/`INTERMEDIATE_SIZE` consts; qwen3 `probe_model()`+`ModelInfo`~~ — removed in the hawk dead-public sweep (#743).
- **File size:** `executor.rs` (1435) and `scheduler.rs` (1420, ~826 of them inline tests) breach the 1k-line redline. ~~`kernel_bench.rs` (1112)~~ — the report harness left the library in #944; it now lives under `src/bin/report_support/`, split across five modules and compiled only by the `kernel-report` binaries.
- **Docs:** `model-crate.md` TL;DR advertises a deleted `qwen3_kernel_snapshot` bench and, with `kernels-crate.md`, uses the obsolete `crates/` layout in every command — collapse both into one slim layout doc. `tp-design.md` describes the implemented controller/worker runtime as future direction — rewrite to past tense, promote the 3 real open items. `kv-pressure-hang.md` — lift the KV-lifetime-reservation lessons to `docs/lessons/`, then delete. `execution.md` Done list predates #216.

## Done criteria

- No admitted request can read past the RoPE cache; long-context behavior is gated.
- A mixed greedy/non-greedy decode step issues one indexed argmax pass for greedy rows and one batched FlashInfer sampling pass for non-greedy rows, not O(batch) per-row sampling.
- TP and LoRA paths sit under the same golden-gate tolerances as the single-GPU path.
- Usage reporting (cached tokens, streaming completion tokens) is truthful.
- The docs describe the crate that exists.
