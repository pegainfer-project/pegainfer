# GLM5.2 continuous batching (D2 + D2.5)

> **TL;DR:** Execution record of PR-D2 (multi-slot admission + {1, 8} batch-bucket graphs) and PR-D2.5 (middle buckets → **{1, 2, 4, 8}**, smallest bucket covering the fullest rank, per-bucket state consolidated into `Glm52BucketState`). **Solo decode 22.4 ms/step** (D1's fixed 8-row batch was 39.2); D2.5 kills the D2 c9 cliff (see the A/B table). jz-38 gates all green across both PRs: oracles (EP8 layer gate replays every bucket's global-token count), solo byte-identical to the D2/D1/PR5c records, per-bucket mutual consistency at 9/17/64-way, 80-way queueing, ladder bucket-crossing, disconnect, SIGTERM, pinned slot-3 parity. **Known numerics property (root-caused, not a bug): each bucket is a distinct FP association** — a request whose lifetime spans buckets can greedy-diverge from its solo replay at a near-tie; deterministic given the same occupancy timeline. Bucket choice per step is a pure function over occupancy (`plan_step_shapes`).
>
> **Last touched:** 2026-07

## What changed

One commit on `feat/glm52-continuous-batching`: `pegainfer-glm52` only, zero CUDA changes.

- **Scheduler (`scheduler.rs`):** one-request-per-rank → up to `GLM52_MAX_BATCH_PER_RANK = 8` requests per rank, each owning one slot (and that slot's disjoint `max_model_len`-token cache region — hardcoded 4096 at the time, VRAM-derived at launch since #579). Admission is least-loaded rank first, lowest free slot; requests join/leave at step boundaries; beyond 64 active the queue holds. The admission and bucket decisions are pure functions over the occupancy (`admission_target` / `step_bucket`, unit-tested).
- **Batch bucket ({1, 8}):** the coordinator agrees a global bucket per step — 1 row per rank while every rank holds ≤ 1 request, the full 8-row batch as soon as any rank holds two. The MoE collectives require every rank to enter with the same global row count (8 vs 64), so the bucket is a coordinator decision, never per-rank. Attention tier stays per-rank (attention is rank-local).
- **Model (`model.rs`):** four whole-step CUDA graphs (2 tiers × 2 buckets), lazily captured — the mid-serving capture-safety argument from the PR5c tier crossing carries over. The 1-row bucket gets batch-1 FlashMLA plans, a batch-1 scratch arena (~1/8 the batch-8 arena), and a 1-row block table the prologue rewrites (dtod of the static table's row for the active slot) — the captured b1 graphs address whichever slot holds the request through device data, never a baked slot id. The 78-layer step body is shared verbatim by both buckets (`run_step_body`); only the plan, scratch, block table, and `global_tokens` differ.
- **Why zero CUDA:** the weight-only GEMV already supports exactly rows ∈ {1, 8} (the m=1 kernel and the D1 batched kernel), and every other kernel takes the row count as a launch parameter.

## Gates (jz-38 8×H200, 2026-07-04, `glm52_d2_gates.sh` / `glm52_d2_pin_slot.sh`, logs alongside)

| gate | result |
|---|---|
| oracle: MLA full / short tier / layer (grouped+gemv) / EP8 layer (replayed at BOTH global-token buckets, 64 and 8) | all PASS |
| solo determinism ×2 (24 + 128 tok), vs PR5c v7 record, vs D1 record | PASS — byte-identical to both |
| tier crossing 320-tok ×2 + prefix + post-short (b1 graphs) | PASS |
| 8-way identical (one per rank — stays bucket 1) | PASS |
| 9-way identical (first two-slot rank — bucket 8) | PASS |
| 64-way full occupancy | PASS (identical prompts, 1590 tok/s aggregate) |
| 80-way (> 64 slots → queueing) | PASS |
| mixed tiers + mixed lengths (12-way) | PASS |
| post-concurrency solo drain (slot reuse, back to b1) | PASS |
| disconnect mid-stream → 9-way traffic | PASS |
| SIGTERM mid-decode at bucket 8 | PASS |
| **pinned slot-3 (b1 bucket)** — hard-coded admission pin, byte vs slot-0 refs | PASS |
| **pinned slot-3 / slot-7 (forced 8-row bucket)** — the D1 acceptance item | PASS |

Slot-stride evidence beyond the pins: in the 9-way 320-tok probe, rank 0's two real rows (slots 0+1) and the single-row ranks produced **mutually byte-identical** outputs — co-resident real rows don't perturb each other and slot-1 addressing equals slot-0.

## The bucket-switch numerics finding

`GATE-BUCKET-CROSS` (a 320-tok request whose lifetime spans 1→8→1 bucket switches) showed the crossing request diverging from its solo replay while the 8 co-resident 128-tok requests stayed byte-identical to theirs. Root-caused with two discriminators:

- 8-way 320-tok (bucket 1 throughout): **identical to solo** → the b1 path and bucket switching corrupt no state.
- 9-way 320-tok (bucket 8 throughout): all 9 **mutually identical**, all differ from solo starting at output token 215 (position 221, inside the short tier); a strict-short-tier 240-tok replay reproduces it.

So batch-1 and batch-8 logits differ in bits from the start (cuBLAS picks different kernels for n=1 vs n=8; FlashMLA distributes its SM parts over 1 vs 8 rows → different split-combine association), and greedy first flips at a near-tie ~215 tokens in. D1's "byte-identical" evidence (24/128 tok) was argmax-visible equality inside the flip-free region — there was never a cross-batch bit-parity guarantee. This is the standard batch-variant-numerics property every batching engine has (vLLM included); the guarantees that DO hold, all gate-proven: determinism given the same occupancy timeline, slot/rank placement invariance within a bucket, and state integrity across switches.

## Measured performance (jz-38 8×H200, 2026-07-04)

Solo probe (133-step greedy, ×5): **22.4 ms/step dead stable** — D1 was 39.2, the PR5c 1-row record 22.5.

Closed-loop scaling, **diverse prompts** (random ~15-token prompts, 128-token outputs, non-streaming `/v1/completions`, `d2_engine_bench.py`):

| concurrency | ms/step p50 | ms/step p99 | aggregate out tok/s | note |
|---|---|---|---|---|
| 1 | 21.5 | 21.6 | 42 | 1-row bucket |
| 2 | 24.1 | 24.3 | 75 | 1-row bucket — per-step cost climbs gently with active ranks (more real dispatch in the collectives + request-boundary stagger) |
| 4 | 26.6 | 26.8 | 135 | |
| 7 | 29.3 | 29.7 | 215 | |
| 8 | 28.1 | 28.5 | 256 | last 1-row point; +25% over the identical-prompt 22.3 = routing diversity |
| **9** | **47.1** | **49.2** | **171** | **the cliff is non-monotonic**: the 9th request drops TOTAL throughput below c7 — everyone pays the 64-row step for 9 real rows |
| 12 | 49.0 | 49.5 | 220 | still below c8; break-even ≈ c14 |
| 16 | 51.1 | 52.3 | 280 | |
| 32 | 57.1 | 57.7 | 502 | |
| 64 | 62.8 | 63.6 | 911 | 22× the solo rate |

Poisson-arrival soak (random output 32–224, non-streaming): rate 4 → 600/600 ok, 493 tok/s, request e2e p50 8.8 s / p99 14.2 s; rate 8 (mild overload) → 600/600 ok, 782 tok/s, p50 15.6 s / p99 27.0 s (queueing as designed). Post-traffic solo parity holds.

Two measurement lessons this table encodes:

- **Identical prompts overstate MoE throughput badly at scale**: the 64-way identical-prompt gate reads 1586 tok/s vs 911 diverse — a 74% inflation (degenerate routing collapses the expert segments). The known ~7-15% figure from bs=1-per-rank measurements grows with rows per step. D1's "~40 ms full-slot step" projection was made with pad rows (token 0 → degenerate routing); the real diverse 64-row step is ~63 ms.
- **`vllm bench serve` (0.24) measurements against this server are client-distorted**: its streaming path reported ~39 ms/token at concurrency 1 where a streamed curl of the same request measures 23.0 (non-streamed 22.5 — so the server's SSE transport is fine, refuting an initial Nagle/TCP_NODELAY suspicion), and at concurrency 8 the client hung with every request complete and balanced on the server side (160/160 finished, all sockets drained, client parked in epoll). It also sends a non-zero default temperature, which the engine fast-rejects — pass `--temperature 0.0`. Engine numbers here therefore come from the non-streaming closed-loop harness; the vllm-bench client interaction is filed as a frontend follow-up.

## D2.5: middle buckets {1, 2, 4, 8}

Motivated by the D2 c9 cliff above, plus the long-context argument: with 1M-class contexts a rank realistically holds 1–2 requests, so the low buckets are the common case — 1 and 2 active/rank hit exactly, 3 pads one row. Powers of two; not all 8 batch sizes because the validation matrix (oracle replays, per-boundary e2e gates, GEMV instantiations) grows linearly per bucket while 5/6/7 active/rank barely occurs — extending is a one-line const change (`GLM52_DECODE_BUCKETS`) + one GEMV template instantiation, justified by its own A/B.

What changed (`feat/glm52-middle-bucket`):

- **Per-bucket state is one struct** (`Glm52BucketState`: batch-N FlashMLA plans per tier, scratch arena, whole-step graphs per tier, block table), `buckets: [Glm52BucketState; 4]` on the rank model — selecting the bucket selects everything coherently; a graph can't be restored into the wrong shape. Marginal cost per bucket ≈ rows × ~1 MB scratch (logits row dominates) + 2 lazily-captured graphs.
- **Coordinator picks the smallest bucket covering the fullest rank** (`plan_step_shapes`, pure function, unit-tested): each rank forwards its active slots first, free slots as padding up to the bucket. The full bucket forwards every slot and must be the identity mapping — its graphs read the static identity block table (`decode_step` asserts this; a non-identity full shape is a scheduler bug).
- **Partial buckets address arbitrary slots through device data**: the prologue dtod-gathers each forwarded row's block-table row from the static identity table, generalizing D2's 1-row `block_table_b1` mechanism to any partial bucket.
- **Batched weight-only GEMV gains batch-2/4 template instantiations**; the launcher switch-rejects anything outside {1, 2, 4, 8}, so a Rust-side bucket drift crashes at the launch boundary instead of computing garbage.
- **EP8 oracle gate replays its collectives at every bucket's global-token count** (g = 8/16/32/64, derived from the const).

### D2.5 gates (jz-38 8×H200, 2026-07-04, `glm52_d25_chain.sh`, log `d25_chain.log`)

| gate | result |
|---|---|
| oracles: MLA full/short tier, layer, EP8 layer at g=64/32/16/8 | all PASS |
| solo determinism + vs D2 refs (b1 path untouched) | PASS — byte-identical, 22.4 ms/step ×5 |
| 8-way (bucket 1) vs solo | PASS — byte-identical |
| 9-way → **bucket 2** / 17-way → **bucket 4** / 64-way → bucket 8 | PASS — each mutually byte-identical AND identical to solo (no near-tie flip within 128 tokens in any bucket regime) |
| ladder: 320-tok request rides buckets 1→2→4 mid-flight | PASS — full length, co-residents clean |
| 80-way queueing, drain, disconnect, SIGTERM mid-decode | PASS |
| pinned slot-3 (b1) — partial-bucket block-table gather addressing slot 3 | PASS — byte-identical to slot-0 refs |

### D2.5 A/B vs D2 (jz-38 8×H200, 2026-07-04; diverse prompts, closed-loop, non-streaming, 3 runs consistent)

| concurrency | D2 ms/step p50 (bucket) | D2.5 ms/step p50/p99 (bucket) | D2.5 tok/s (D2) |
|---|---|---|---|
| 1 | 21.5 (b1) | 21.4–22.3 / 22.4 (b1) | 42 (42) |
| 2 | 24.1 (b1) | 23.2–24.1 / 24.2 (b1) | 78 (75) |
| 4 | 26.6 (b1) | 25.7–26.6 / 27.1 (b1) | 140 (135) |
| 8 | 28.1 (b1) | 28.1–28.2 / 30.6 (b1) | 255 (256) |
| 9 | **47.1 / 49.2** (b8) | **31.8 / 32.3** (b2) | **254 (171)** |

The c9 cliff is gone: −32% step latency, +48% throughput, and the curve is monotonic again (D2's c9 dropped below c7's 215 tok/s). b1 points are flat — zero regression. Sweep points beyond c9 were dropped by decision: the target workload (1M-class contexts, P/D decode node) rarely exceeds one request per rank, and the c9 boundary is the cliff the middle buckets exist to remove.

### Open anomaly: one-off silent request drop (~1/3500, unreproduced)

During the first D2.5 bench attempt one c2 request hung the client forever: the frontend logged its arrival, but the engine never stepped it (GPU idle) while the coordinator stayed healthy (a later probe request completed instantly) — i.e. the request vanished between the frontend's `completions;` log line and admission, or was silently freed. Not reproduced since across >3500 requests (6× c2:200 + 3× full sweeps) with a lifecycle-instrumented build (eprintln at intake/admit/dead-skip/prefill-closed-free/emit-send-fail-free/finish); a 40-round instrumented soak keeps hunting it. No evidence it is D2.5-specific — D2's total bench volume never reached this event's rarity class. Tracked in a GitHub issue with the forensic signature; the instrumented repro harness is `glm52_d25_longsoak.sh` on jz-38.

## Next step

- ~~Expert-GEMM M-tile work (#542's swapAB lead)~~ DONE 2026-07-05: DeepGEMM masked grouped GEMM landed (the "swapAB" attribution was wrong — see whole-step-decode-graph.md); c64 diverse 1113 → 1475 tok/s, sweep −16..−24 % across buckets.
- P/D decode-node work (KV ingestion from a vLLM prefill) is the next campaign; prefill stays out of scope per the standing prefill-by-vLLM decision. (The static per-slot cap itself is no longer hardcoded 4096 — #579 made it VRAM-derived at launch with a `--max-model-len` override; the shared page pool landed — see `paged-kv-prefix-cache.md` for the true paged block table + prefix caching this campaign builds on.)
