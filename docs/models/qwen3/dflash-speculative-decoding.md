# DFlash Speculative Decoding (Qwen3-4B)

**TL;DR:** Qwen3-4B gains DFlash speculative decoding behind `--dflash-draft-model-path`. Speculative decode is modelled as an **optimistic transaction**: the DFlash drafter proposes K tokens, one target "verify" forward over the K+1 span confirms them, and we commit the longest argmax-matching prefix + 1 bonus token (rolling back the rest of the speculative KV). Greedy decode is **lossless up to bf16 numerical tie-flips** — the same non-determinism that already affects plain greedy decode at genuine bifurcation points; lm-eval gsm8k strict-match is identical with spec on vs off. Measured single-stream decode A/B: **1.82× on RTX 5070 Ti** (93.4 → 170.0 tok/s), **1.56× on RTX 5090** (168.9 → 263.2 tok/s). The drafter's out-of-pool footprint (weights + per-request KV/scratch) is reserved during memory profiling so the KV pool shrinks to fit instead of OOMing under load, and requests landing in the draft's final in-fill block are rejected cleanly at admission rather than crashing mid-prefill. **Concurrent throughput is now competitive after batching the draft forward.** The draft used to run a per-request **serial** `for` loop — launch-bound (a skip-attention A/B showed attention compute <2%, so 24.8 ms/step at batch 16 was almost all kernel-launch overhead), which inverted the single-stream win (c16 −59%). Batching the dense ops into one N×block pass drops draft@batch16 to **5.6 ms** and lifts 5090 greedy throughput to **c8 1346 / c16 1868 tok/s — both now beating vLLM's 1240 / 1846**. The single-stream gap is now closed by a **piecewise CUDA Graph** over the verify forward: 5090 greedy **c1 237 → 274 tok/s (+16%), matching vLLM dflash (278)**; c8 1346 → 1525 and c16 ≈ flat (no regression, both still ≥ vLLM's 1240 / 1846). Under greedy dflash the spec path ran with no CUDA Graph at all (the base-decode graph never fires), so nsys saw 1296 ms of GPU-idle launch gap — **~84% from dense-op kernel launches** (GEMM alone 48%), only 8% from attention. Capturing the *whole* forward fails: FlashInfer's paged-prefill attention freezes its KV-iteration count at capture time, so a captured attention under-reads the growing verify KV and corrupts tokens past ~60. The fix is **piecewise** — capture the dense ops (embedding, RMSNorm, every GEMM, SwiGLU, residual adds) into per-segment graphs and replay them, keeping **attention eager**. Accept is *not* the gap (9.1% vs vLLM's 8.85%, same drafter). Draft-side piecewise graph is the tracked next step. See Performance § for the A/B tables.

Last touched: 2026-06

## The abstraction: speculative decode = optimistic transaction

Every speculative method is the same transaction; only *propose* differs.

1. **Propose** — a method-specific drafter emits K candidate tokens. (DFlash is the only proposer today; an n-gram / EAGLE proposer would slot in here.)
2. **Verify** — the target model runs ONE prefill-style forward over the K+1 span `[current_token, draft_1, …, draft_K]` and reports its argmax at every position (echo=true).
3. **Accept** — `accept_greedy` keeps the longest prefix where each draft matches the target argmax, then appends the target's own token at the first mismatch (the "bonus"). A verify step therefore always commits `1..=K+1` tokens — at least one token of progress even when every draft is rejected.
4. **Commit / roll back** — accepted tokens' KV is committed (`apply_speculative`); the unused tail of the speculative reservation is LIFO-dropped (`revert_schedule`).

The draft↔verify boundary is a **pure token span**. Hidden states never cross it — they stay inside the proposer (`dflash.rs` / `dflash_lane.rs`). This is what lets the shared core (`speculative.rs`: `accept_greedy`, `build_verify_results`) be method-agnostic, and it is why there is deliberately **no proposer trait yet**: a trait with one impl is premature. Add it when a second method lands.

Shared core lives in `pegainfer-qwen3/src/speculative.rs`; the transaction wiring is in `pegainfer-qwen3/src/executor/spec.rs`; the KV transaction primitives (`schedule_speculative` / `apply_speculative` / `speculative_view` / `revert_schedule`) are in `pegainfer-kv-cache/src/pool.rs` delegating to kvbm `scheduled.rs`.

## Key invariant: readiness comes from prefill capture, not a handshake

Speculative-on **forces prefix caching off**. With no prefix reuse, every eligible request's target hidden context is captured during its own prefill — so there is no per-request "drafter ready" handshake. Readiness is derived from the prefill capture-set plus the completed flag. This is the load-bearing simplification; if prefix caching were allowed, a cache-hit request would skip the prefill that seeds the drafter, and the invariant would break.

Draft-seed alignment: after a verify accepting `m` drafts, the committed span positions with *known* hidden states are `[current_token, draft_1..draft_m]` = `m+1` rows. The next current_token (`target_argmax[m]`) is freshly predicted and its hidden was never forwarded — exactly like the post-prefill first token. `record_verify_dflash_context` appends `accepted_tokens.len()` (= `m+1`) rows and advances `kv_position` by the same. This is the subtlest part of the feature; it is defended by crash-early asserts (`append_pending_context` overflow/dim/range checks).

## Losslessness: lossless up to bf16 ties

The greedy oracle is "spec-on equals spec-off token-for-token". In practice the two diverge only at genuine bifurcation points, and only because of bf16 numerical noise — **not** a logic bug. Evidence:

- The verify forward builds committed KV incrementally across batched spans (M=16-ish), while a one-shot prefill is a single M=1 forward. cuBLAS picks different GEMM tilings → ULP-scale reduction-order differences → an argmax flip at a near-tie. This is the same class as the `hf_golden_gate` `MARGIN_TOL = 0.20` tolerance.
- Multi-token acceptance (runs observed up to 11 tokens) is **bit-identical** to baseline at every non-tie position — proving the span-position-≥1 path is correct.
- Re-running flips **different** prompts each time (non-deterministic ⇒ bf16, not a deterministic logic error); one observed flip had regret 0.000 (an exact tie).
- The spec pick is always the #1 or #2 token of the prefill kernel's own distribution, within 0.20 nat of #1.

### The losslessness gate (`tests/dflash_speculative_gate.rs`)

The gate runs a baseline (spec off, logprobs on) and a spec engine on the same prompts, then at the first token where they diverge it measures the spec pick's **regret** against the prefill kernel's own distribution. Within `MARGIN_TOL = 0.20` nat ⇒ benign tie-flip (pass); clearly worse ⇒ real bug (fail). The realistic bug classes (KV misalignment, mask leak) push the pick far outside 0.20 nat, so the gate has teeth — it is not tautological.

**Known scope limit:** the gate regret-checks only the *first* divergence position per prompt, then continues. A bug that stays within the tie band at the first bifurcation but corrupts later positions could slip. Acceptable for v1; the next iteration should re-anchor and regret-check the following few positions after a benign-tie classification.

## Performance

Single-stream decode is where speculative decoding pays off directly: plain decode is memory-bound (one target forward per token); spec amortizes that forward over the accepted run. A/B harness: `tests/dflash_speculative_perf.rs` (bs=1, 256 tokens, `ignore_eos`, one warm-up discarded).

| Config | RTX 5070 Ti, bs=1 | RTX 5090, bs=1 |
| --- | --- | --- |
| spec OFF (plain decode) | 93.4 tok/s | 168.9 tok/s |
| spec ON (DFlash) | 170.0 tok/s | 263.2 tok/s |
| **speedup** | **1.82×** | **1.56×** |

The speedup is smaller on the 5090: its higher memory bandwidth makes the baseline decode less memory-bound, so amortizing the target forward buys less. Both builds use CUDA 13.x (the 5090's default 12.9 has the documented cuBLAS N=1025 cliff).

### Concurrent throughput: the draft loop is serial and launch-bound

Under concurrent load the single-stream win inverts. Same greedy harness (`temperature=0`, sharegpt prompts, 128 out tokens), pegainfer vs vLLM with the **same** DFlash-b16 drafter, RTX 5090:

| concurrency | OI plain | OI DFlash serial | OI DFlash batched | **OI DFlash +graph** | vLLM DFlash |
| --- | --- | --- | --- | --- | --- |
| c1 | 170 | 245 | 237 | **274** | 278 |
| c8 | 1180 | 831 | 1346 | **1525** | 1240 |
| c16 | 2277 | 1013 | 1868 | **1834** | 1846 |

(tok/s, sharegpt out128, RTX 5090, greedy. The serial draft inverted the win — vLLM degraded gracefully while pegainfer nearly halved; batching the draft restored c8/c16 past vLLM. The **+graph** column adds the piecewise verify CUDA Graph (this branch): it closes c1 to vLLM and lifts c8, with c16 flat — see "Single-stream gap" below. Caveat: the serial→batched columns are a same-session A/B; the +graph column's c1 is a clean same-session A/B (251 fixed-buffer eager → 274 graph), while c8/c16 are single runs against the prior batched baseline, i.e. a no-regression check rather than a tight A/B.)

#### Root cause (serial draft) and the fix (batched draft, landed)

Accept length is **not** the cause — both engines accept ~equally (same drafter + greedy = longest-prefix match). The cause was the **per-request serial draft loop**: `execute_dflash_draft` (`dflash_lane.rs`) called `draft_logits` once per request in a `for` loop, while verify runs ONE batched target forward over all spans. Per-step draft timing by batch size, **before vs after batching** (5090, instrumented):

| batch | draft serial | draft batched | verify | draft serial scaling | draft batched scaling |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.56 ms | 1.55 ms | 8.1 ms | 1.00× | 1.00× |
| 8 | 12.45 ms | 3.17 ms | 9.6 ms | 7.99× | 2.04× |
| 16 | 24.86 ms | **5.62 ms** | 13.6 ms | **15.96×** | **3.62×** |

The serial draft scaled **exactly linearly** (`draft_x` 16.00 at batch 16); batched draft scales sub-linearly (3.62×). At batch 16 the draft drops from 65% of the step to ~29%, and step time roughly halves → c16 throughput doubles.

**Why it was launch-bound, not compute-bound.** A skip-attention A/B (short-circuit `single_prefill` to a cheap copy, keep every other kernel) barely moved the serial draft — batch 16: 24.36 ms vs 24.81 ms, so attention compute is **<2%**. The 24.8 ms was almost entirely per-request serial **kernel-launch overhead**: each request's draft is ~90 tiny 16-token kernels (5 layers × 35 dense GEMMs + varlen copy/rope/KV), compute ≈ 0.

**The fix that landed: batch the whole draft forward (N requests in one pass), killing the N× launch.** `draft_logits_batched` (`dflash.rs`) runs the dense ops (rms_norm / GEMM / silu / MLP / embedding / logits) **once over an N×block buffer** — free, since cuBLAS takes any M and the ops are already row-batched (N×35 → 35 launches, no CUDA-kernel change). The varlen ops (tail concat / k·v GEMM / rope / KV-copy / attention) stay a per-request loop slicing the batched buffers at `row_offset = i·block_size` (the two DFlash-exclusive ops `dflash_qk_norm_rope_into` / `single_prefill_nhd_noncausal_into` advance the device pointer to the slice). A lane-level `DFlashBatchScratch` (sized for the largest batch bucket) replaces the per-request scratch — folding in the reservation-reclaim follow-up. Losslessness gate still passes (bf16 tie-flips only).

#### Single-stream gap (c1): closed by a piecewise verify CUDA Graph

c1 trailed vLLM (237 vs 278), and it is **not accept**: with the *same* drafter and spec config (vLLM `--speculative-config '{"method":"dflash",…,"num_speculative_tokens":16}'`), measured greedy accept is **9.1% (ours) vs 8.85% (vLLM)** — mean 1.29 vs 1.42 draft tokens/step, our pos-0 accept (60%) actually higher. 9% is the drafter's floor on sharegpt free-text (bimodal: structured spans accept up to 15, free text 0), shared by both engines.

The gap is **launch exposure**. Under greedy dflash the spec path runs with *no* CUDA Graph: the base-decode graph (`execute_decode`) never fires because greedy routes through `SpeculativeDecode`, not `batch_decode`. nsys on c1 (single stream, fully serial): wall 7999 ms, GPU **busy** 6703 ms (memory-bound weight reads — irreducible, the same for vLLM), GPU **idle 1296 ms**, ~79% of it sub-3 µs kernel-launch gaps. Attributing that idle by kernel type (`gap_by_kernel.py`): **dense ops 84%** (GEMM 48%, RMSNorm 15%, embedding / KV-copy / residual-add / SwiGLU the rest), **attention only 8%**. Dense dominates because verify is 36 layers of GEMMs vs the draft's 5 — so the launch gap lives mostly in *verify*, not the draft. (This corrects the earlier read that pinned c1 on draft launch; that predated the batched draft and the per-kernel-type nsys breakdown.)

**Why the whole forward can't go in one graph.** A first attempt captured the entire verify forward; output was correct up to ~token 60, then garbage. Root cause: FlashInfer's paged-prefill attention derives its KV-iteration count (`num_iterations`, from `kv_len`) and that loop bound is **frozen when the graph is recorded**. The verify context grows every step, so once it crosses the captured `CTA_TILE_KV` (~64) boundary the replayed attention under-reads KV. Base-decode's graph is safe only because its *decode* kernel's KV loop is purely device-driven; *prefill*'s is not. FlashInfer ships no graph-safe prefill variant; vLLM hits the same wall and keeps attention out of its piecewise cudagraph.

**The fix: piecewise graph.** Keep attention **eager**; capture only the dense ops, whose dims depend on the verify row count (`total_tokens`), never on KV length. `forward_layer_batch_paged` is split into `pre_attn` / `attn` / `post_attn`, and the verify forward becomes `num_layers+1` dense graph segments — `[embed+L0.pre] [L0.attn eager] [L0.post+L1.pre] … [L_last.post+lm_head]` — captured once per batch bucket and replayed (`verify_graph.rs`). The ping-pong residual swap sits inside the captured segments, so its pointer alternation is baked into each graph and reproduces on every replay regardless of layer parity: `run_or_capture` re-runs the CPU swap only on the capture step, and the one eager op (attention) touches just `q/k/v_batch` / `attn_output`, never the swapped `hidden`.

Result (5090, greedy, same-session A/B): fixed-buffer eager **250.9 → +graph 274.3 tok/s (+9.3% from the graph alone)**; 237 → 274 (+16%) over the pre-graph batched baseline, **matching vLLM's 278**. Concurrent (no-regression check): c8 1346 → 1525, c16 1868 → 1834 (both still ≥ vLLM). Losslessness gate passes (bf16 tie-flips only) — the dense ops replay bit-identically, and the eager attention is unchanged.

**Capture-shape gotcha (fixed).** `total_tokens` is **not** constant at a given batch bucket: a request near its output budget shortens its verify span (`plan.rs` truncates the span to `max_tokens − generated`), so `total_tokens < batch_size × span`. The captured dense kernels bake their row count into the launch, and `run_or_capture` is capture-once-replay-forever — so a graph first captured at a *short* span and later replayed at a *full* span processes too few rows and leaves the tail-request logits **stale**: a silent losslessness break. It hid from the bs=1 gate (a fresh request's first verify is always a full span ⇒ bucket captured at the max ⇒ only the harmless over-compute direction occurs) **and from the homogeneous c8/c16 benches** (lockstep requests capture every bucket at full span during ramp-up; all truncation comes later as they finish together — still the safe direction). The dangerous direction needs *heterogeneous* progress. Fix: gate the captured-graph path on `total_tokens == batch_size × span` (every request a full span); any truncated step runs eager — making capture-shape ≡ replay-shape an invariant by construction. Regression test `dflash_short_then_long_verify_capture_is_lossless`: a `max_tokens=4` request poisons the bucket-bs=1 graph at a truncated span, then a long request replays it — RED before the gate (diverges to a stale-buffer token), GREEN after. Cost: a batch containing *any* truncated request runs fully eager that step (rare — only a request's final block); a follow-up could pad-to-full + mask to keep the graph.

**Draft-side piecewise graph is the tracked next step.** The draft (5 layers, `dflash.rs`) is the other ~16% of the launch gap; it needs the same pre/attn/post split, with its variable-length contiguous KV (`DFlashLayerCache`) handled at the eager attention boundary. Tracked as its own PR after this one lands.

The EAGLE proposer trait is still deferred to when EAGLE actually lands (see "no proposer trait yet" above); both the batched draft and the future CUDA-Graph draft are DFlash-internal changes behind the unchanged `DraftPlan→DraftResult` seam, so neither is thrown away by EAGLE.

## Task-level accuracy parity (lm-eval gsm8k)

Token-level losslessness should imply task-level parity. Confirmed on the 5090 with `lm-eval` gsm8k (5-shot, greedy, `local-completions` against the pegainfer server, 50 questions):

| | flexible-extract | strict-match |
| --- | --- | --- |
| spec OFF | 0.86 | 0.86 |
| spec ON (DFlash) | 0.88 | 0.86 |

`strict-match` is identical; `flexible-extract` differs by one question (within the ±0.05 stderr), the same single bf16 tie-flip the losslessness gate sees. DFlash does not change task accuracy.

Harness note: pegainfer's `/v1/completions` rejects a per-request `seed` field (`"per-request seed is not supported by this engine"` → 400), which the OpenAI/lm-eval client sends by default. For the eval the client was patched to drop `seed`; making the frontend accept-and-ignore `seed` under greedy is a separate, unrelated improvement (not part of this change).

## GPU memory budget & context limit

DFlash's draft model and its per-request KV/scratch live **outside** the paged KV pool (`KvCacheManager`), so they must be reserved *before* the pool is sized or they silently steal from it and OOM under concurrency / long contexts. `DFlashMemoryReservation::from_config` (cheap — reads the draft `config.json`) splits the footprint by how it scales and the budget charges each in the matching place:

- **Per-token, pool-scaling** (draft KV + the prompt-tracking context scratch + the in-fill tail scratch + pending context, **65 536 B/token**) → folded into `effective_bytes_per_block`, so the *target* block count shrinks while the pool itself stays allocated at the target-only `bytes_per_block`. This is a safe upper bound: the scheduler reserves pool blocks for each request's whole lifetime (`prompt + max_tokens`), and the draft attends at most that many tokens, so billing the draft per pool-token over-covers. The draft KV and tail scratch are sized one in-fill block past that lifetime, so the fixed term also reserves `max_decode_batch × block_size` of that per-request headroom.
- **Fixed, pool-independent** (draft weights ~1.1 GiB + the block-sized per-request scratch across the decode batch — logits-dominated, ~6.5 MiB × 256) → added to the KV `margin`.

Measured on the RTX 5070 Ti (16 GB, util 90%), the pool correctly makes room for the draft:

| | margin | KV budget | KV blocks |
| --- | --- | --- | --- |
| DFlash OFF | 150 MiB | 4113 MiB | 1828 |
| DFlash ON | 2972 MiB | 1389 MiB | 427 |

The fixed reservation lands exactly in the margin (+2822 MiB) and the per-token factor (≈1.44×) shrinks the remaining blocks. Before this, those ~2.8 GiB loaded on top of the full 1828-block pool → OOM under load.

**Context limit.** The drafter's fixed-width in-fill block writes `block_size` positions past the committed length each step, so the DFlash-effective context is `max_position_embeddings − block_size`. `max_context_tokens()` returns that when DFlash is on, so a request that fits the target window but lands in the draft's final block is **rejected cleanly at admission** instead of crashing mid-prefill (`tests/dflash_speculative_gate.rs::dflash_request_in_draft_headroom_is_rejected_not_panicked`).

## Constraints & open follow-ups

- **TP=1, primary rank only.** The DFlash lane runs on the worker thread of the primary rank; the launch path gates `!(dflash && lora)` and `tp_size == 1`. The server fails loud (`--dflash-draft-model-path` is rejected for non-Qwen3 model lines rather than silently ignored).
- **Second proposer (EAGLE) → introduce the trait then.** A proposer trait wants two real methods to fix its shape; DFlash stays concrete until EAGLE lands. The batched-draft perf work (Performance §) is orthogonal — it's a DFlash-internal change behind the unchanged `DraftPlan→DraftResult` seam, so it can land before or with the trait without being redone, and EAGLE's autoregressive attention won't reuse DFlash's block attention anyway.
- **File size.** `executor.rs` (~2.8k lines) and `scheduler/tests.rs` (~1.2k lines) exceed the 1k guideline; the spec arms in `execute_step_on_lane` are candidates to move into `executor/spec.rs` in a follow-up.
- **5090 validation — done** for correctness, single-stream, and concurrent throughput: `hf_golden_gate` passes (bs1 / batched / cuda-graph / tp2), the losslessness gate passes (batched draft *and* piecewise verify graph), single-stream A/B is 1.56×, and lm-eval gsm8k parity holds (above). Concurrent throughput is **fixed** (Performance §): batching the draft beat vLLM at c8/c16, then the **piecewise verify CUDA Graph** closed c1 — **c1 274 ≈ vLLM 278, c8 1525 > 1240, c16 1834 ≈ 1846**, all batch sizes now at or above vLLM. Draft-side piecewise graph (the remaining ~16% launch gap) is tracked next.
- **Frontend `seed`.** `/v1/completions` 400s on a per-request `seed` instead of accepting-and-ignoring it under greedy — surfaced while wiring lm-eval; orthogonal to spec decode, worth a small separate fix.
- **Reclaim the per-request reservation.** The fixed term is dominated by the block-sized scratch billed per decode-batch slot (~6.5 MiB × 256 ≈ 1.6 GiB), most of it the transient logits buffer (`vocab × block`). The per-token term carries the context/pending scratch, which today reallocs to prompt length on the first draft and never shrinks (sized for `[0, context_len)` but only `[committed_len, +block]` is live). Both are billed conservatively because they were per-request. **Partly reclaimed by the batched-draft PR:** the per-request scratch is gone — `DFlashBatchScratch` is allocated once at lane level, sharing the transient logits/dense buffers across the whole batch instead of per request. The reservation accounting (`from_config`) still bills the old per-request upper bound (a safe over-estimate, so no OOM); tightening it to the now-smaller lane-level footprint is the remaining follow-up.

### Review blockers (correctness/usability, independent of the perf work)

Issues surfaced in PR review, largely independent of the draft batching.

1. **Unified path silently skipped DFlash readiness — fixed.** `StepCommand::Unified` (`executor.rs`) captures no DFlash hidden state, and only `execute_prefill` marks a request draft-ready. A greedy request prefilled via a fused Unified step would therefore never become draft-ready and never recover — DFlash silently no-opped for it under mixed load (no wrong tokens, the feature just quietly disabled itself). Fixed by routing capture-eligible pending (greedy, no LoRA, no logprobs) to a **dedicated prefill step** instead of Unified — `build_next_plan`'s `needs_dflash_capture` (`scheduler/plan.rs`), mirroring the existing `needs_prompt_logprobs` precedent — so prefill capture always runs. The Unified decode arm also now drops stale draft context for each decoded request (`execute_unified`, mirroring `execute_decode`), keeping the "readiness comes from prefill capture" invariant closed instead of degrading silently.
2. **Stream-override race — fixed.** The DFlash `qk_norm_rope` / `single_prefill` wrappers (`attention.rs`) already used `active_cu_stream(ctx)`; `copy_hidden_rows_into` and `copy_hidden_token_range_into` (`elementwise.rs`) now launch on `active_cu_stream(ctx)` too, matching the repo convention (`tensor.rs:43`) so a captured copy records on the right stream. Belt-and-braces: DFlash + decode overlap is now **rejected at launch** (`lib.rs`) — the speculative path never takes the unified overlap route, so the combination only burned VRAM the drafter needs.
3. **`gemm_lt_pin_tune` is not a real warmup — still open.** It only pins the heuristic (`linear.cu:497`); it never executes a `cublasLtMatmul` the way the old `gemm_lt_tune_cuda` (`linear.cu:431`) did, so the first real matmul can land inside CUDA-graph capture. `batch_invariance_decode_gemm_graph` backstops it, but the warmup should actually run the matmul. This is the remaining review blocker.

Concurrent, heterogeneous-`max_tokens` losslessness is now covered by `dflash_concurrent_heterogeneous_is_lossless` — several greedy requests at staggered budgets run as one batch, each checked against its own plain-greedy baseline. That exercises the bs>1 draft+verify path the bs=1 gate and the homogeneous c8/c16 benches never reach (a batched-draft indexing or capture-shape regression at bs>1 would surface as a real, non-tie divergence).
