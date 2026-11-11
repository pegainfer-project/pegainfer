# Qwen3-4B prefix cache

**TL;DR**: Prefix caching is on by default for Qwen3-4B. Full-block token-hash matching via the vendored kvbm radix tree, wired at the executor level. Repeated ~1900-token prompt: TTFT 141.8ms → 16.3ms p50 (8.7×); warm TTFT ≈ 1 decode TPOT (11.4ms) + ~5ms request-setup/eager-launch overhead. Accuracy gated by `hf_golden_gate` cached-replay surfaces (warm deltas at the cold bf16 floor).

Last touched: 2026-11

## How it works

- `Qwen3Executor::execute_prefill`/`execute_unified` call `RequestKv::match_and_add_prefix` before scheduling. Matched blocks advance `prefill_position`/`kv_position`; only the uncached suffix is forwarded (`PrefillStepItem::as_slice()` skips `cached_tokens`).
- Matching is full-block only (block = 16 tokens), keyed on kvbm `SequenceHash` (positional lineage hash of token content). Partial tail blocks never match.
- **Full-block cap**: `SchedulableSequence::match_and_add_prefix` caps matches at `(num_input_tokens - 1) / block_size` — a fully-cached prompt still recomputes ≥ 1 token (vLLM does the same). Without the cap the schedule/apply state machine deadlocks: prefill is "complete" with no forward pass to emit the first token, and the dangling-block count is wrong.
- **Echo requests skip matching** — prompt logprobs need every position forwarded.
- **LoRA-scoped**: the active adapter name is folded into the block-hash chain as a salt (`compute_salt_hash`, upstream router-parity recipe), so KV computed under one adapter (or the base model) never matches a request running under another. Same tokens + same adapter still share.
- Eviction safety: matched blocks are held as `ImmutableBlock` (strong Arc) in the sequence's assignments for the request lifetime; the inactive-pool LRU cannot reclaim them mid-request.
- **Pages are counted by reference, not by physical block.** N views holding the same cached prefix each list those page ids again, so anything sized off the pool's *physical* block count under-allocates under sharing — admission won't catch it, it also charges physical blocks. `BatchDecodeBuffers::page_indices_d` hit exactly this (#403 fault 1: cudarc copy assert → dead worker → wedged engine; deterministic with N concurrent same-prompt requests once the prefix is sealed). It is now sized for `max_batch × full-context views` and `sync_paged_meta` fails loud instead of tripping the assert. Any future consumer of concatenated per-request page lists must use the by-reference bound.
- Toggle: `Qwen3Executor::set_prefix_cache_enabled(false)` — used by tests that need cold determinism; there is no server flag.

## With speculative decoding (DFlash / DSpark)

A prefix hit skips the forward that produces the drafter's target hidden rows. DFlash instead stores the **draft K/V** in a paged pool indexed by the target KV's block IDs; the target radix cache's block lifetime covers it, so a prefix hit already has the drafter's K/V.

- **Pool:** per draft layer, K and V are `[kv_dim, total_blocks × 16]` bf16, addressed by `page_indices[position / 16] * 16 + position % 16`. Draft K/V is 20,480 B/token on Qwen3-4B (5 layers × 2 × 1024 × 2), ~14% of target KV; on the L20 profile (8,556 pool blocks) the pool is ~2.8 GB. Only real pool blocks participate (padding/scratch excluded). The pool is allocated only while prefix caching is enabled; `--no-prefix-cache` keeps the request's dense mirror and skips the 2.8 GB side pool.
- **Materialize:** immediately after target prefill/verify, before the block can be matched, the lane fuses the captured target rows once (`fc + hidden_norm`), does per-layer K/V + K norm + RoPE at the absolute positions, and writes the rows into the paged slots and the request's dense mirror. Gated/plain verify also materializes. Every registered block therefore has all of its rows forwarded; kvbm stages a block completed by the verify's sampled bonus and registers it only after the next forward.
- **Tags:** the executor keeps one `Option<SequenceHash>` per target block, set after the block is registered and its rows were materialized. A prefix match whose blocks are not tagged for the same content (a logprob request that never captured, or a reused block) keeps the target KV hit but plain-decodes; it never reads uninitialized draft K/V.
- **Restore:** a prefix match hands the matched page IDs to the executor, which checks the tags; the lane gathers those blocks from the pool into the request's dense mirror. There is no capture hash table, no prefix projection, and no first-round rebuild. The suffix prefill materializes only its own rows.
- **Lifetime:** the draft pool uses the target block IDs, so target refcount/eviction reclaims both and block reuse overwrites the draft rows. The reservation covers the pool plus the per-request mirror (20,480 B/token); the L20 budget ceiling is ~108 requests at 4k tokens.
- **Scope:** DFlash/DSpark only. EAGLE-3 has no prefix-restorable draft state yet and keeps `spec-on ⇒ APC off`.
- **Batch:** the step decision is all-or-nothing; a refused request plain-decodes the batch and drops the restored mirrors. With the smaller mirror, 64 shared-prefix requests now fit and speculate: the manual gate reports 64/64 KV hits and **2 → 130 drafts in 8.4 s**.

The warm-hit gate is `dflash_warm_prefix_keeps_drafting` in `tests/dflash_speculative_gate.rs`: cold fill, plain baseline, then a warm request that must hit the prefix, stay lossless up to bf16 ties, and keep the cold accept rate.

**L20 run record** (2026-11, Qwen3-4B, `deepseek-ai/dflash_qwen3_4b_block7`, release): the warm gate reports **4 draft rounds / 27 accepted on both the cold and warm requests**; the prefill-only-turn gate warms the prefix and drafts; the exact-boundary gate matches one block short and drafts; the untagged-prefix gate keeps its KV hit and plain-decodes; the finish gate passes; the lever gate (`MAX_BATCH=1`, `MAX_DRAFTS=4/2`, `MAX_BATCH=3`) passes with the gated and trimmed rows. The full gate file is 12 passed / 2 ignored in 168.80 s; the manual 64-request shared-prefix gate is 8.08 s with 64/64 KV hits and 2 → 130 drafts. Run on one L20.

## Numbers (RTX 5090, 2026-06, drained-stream measurement)

| Metric | p50 | p99 |
|---|---|---|
| cold TTFT (unique ~1900-token prompts) | 141.8ms | 205.8ms |
| warm TTFT (repeated prompt) | 16.3ms | 16.7ms |
| decode TPOT (bs=1) | 11.4ms | — |

Warm TTFT theory: the suffix is 1..16 tokens (cap + prompt length mod 16), and at 4B the forward is weight-bandwidth bound, so GPU time ≈ one decode step. The ~5.4ms gap over TPOT decomposes (measured by varying one factor at a time):

| Component | Cost | How isolated |
|---|---|---|
| tokenize ~1900-token prompt | ~2.9ms | warm TTFT − cold 16-token-prompt TTFT (16.8 − 13.9); HF tokenizers on the same string measures 3.2ms standalone, so radix match + block-table H2D inside this bucket are sub-ms |
| request setup (HTTP parse, admission, FlashInfer prefill-plan build, sampling readback, SSE) | ~1.7ms | cold 16-token-prompt TTFT − graph TPOT − eager-launch share |
| eager per-layer launches (suffix prefill is not graph-captured) | ~0.8ms | eager-decode TPOT (`--cuda-graph=false`) 12.18ms − graph TPOT 11.40ms |

Tokenization is the single largest term — for cached long prompts the CPU tokenizer costs more than everything the scheduler and kernels add on top of the forward pass. So the next lever for warm TTFT is the tokenizer, not the GPU path; graph-capturing the suffix prefill only buys ~0.8ms.

Supporting measurements behind the decomposition (all drained-stream, n=20, same session):

| Measurement | p50 |
|---|---|
| TTFT, 1-token prompt, cold | 13.4ms |
| TTFT, 16-token prompt, cold | 13.9ms |
| TTFT, ~1900-token prompt, warm | 16.8ms |
| TPOT, graph decode | 11.40ms |
| TPOT, eager decode (`--cuda-graph=false`) | 12.18ms |
| tokenize 1500-word prompt standalone (HF `tokenizers`, 1502 tokens) | 3.2ms |

Negative result: registering 40 extra unique ~1900-token prompts did not move warm TTFT p50 at all — radix-tree matching cost is flat at this registry scale (~5k blocks), so don't reach for match-cost optimizations without first reproducing a regression.

## Pitfalls

- **RoPE scalar-path corruption (fixed on this branch).** `prefill_attention_paged_into` had a `batch_size == 1` fast path that used a scalar `start_pos` (always 0 from the qwen3 call site) instead of the plan's per-token positions array. Cold runs were unaffected (start really is 0); warm bs=1 prefills rotated the suffix K/Q from position 0 and scattered the mis-rotated K into KV pages — decode drift up to 3.3 nat in `hf_golden_gate` cached replay. The scalar branch and its dead kernel wrapper (`prefill_qk_norm_rope_only_cuda`) are deleted; all callers go through the positions array, which is bit-identical for cold runs (both C entry points launched the same kernel). Lesson: a "fast path" that duplicates position logic is a fork that only breaks when positions stop being trivial.
- **TTFT measurement: drain the stream.** The server has no abort-on-disconnect — a client that hangs up after the first token leaves the request decoding its remaining tokens, and the *next* request's prefill contends with that zombie decode. Early-disconnect measurement inflated warm TTFT 16.3 → 25.5ms p50. Always read SSE to `[DONE]` between samples.
- **Cache-off runs do not seed later warm replays.** Blocks are still registered while a request executes, but request teardown marks their canonical primaries reset-on-release. Re-enabling the cache therefore starts cold and tests must seed retained blocks explicitly before asserting a warm hit.

## Tests

- `pegainfer-qwen3/tests/prefix_cache.rs` — behavioral contract: exact cached-token counts (3-block hit + tail recompute, extension match, full-block cap edge), mixed cold+warm batch in one plan, unified prefill+decode path, warm-vs-cold logit bounds (regret + mean, golden-gate methodology).
- `pegainfer-qwen3/tests/hf_golden_gate.rs` — cached-replay surfaces (sequential bs=1 eager, batched eager, batched cuda-graph) vs the HF golden: warm mean 0.0316 / p99 0.1215 vs cold floor 0.0317 / 0.1196.

```bash
PEGAINFER_TEST_MODEL_PATH=/data/models/Qwen3-4B cargo test --release -p pegainfer-qwen3 --test prefix_cache
PEGAINFER_TEST_MODEL_PATH=/data/models/Qwen3-4B cargo test --release -p pegainfer-qwen3 --test hf_golden_gate
```

## Next

- Abort-on-disconnect filed as #215 (zombie decode wastes GPU and pollutes TTFT measurement).
- Qwen3.5-4B adoption: the executor wiring pattern transfers, but the GDR linear-attention state is positional — cached prefix reuse needs the recurrent state snapshot, not just KV pages. Needs its own design note before attempting.
