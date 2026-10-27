# Qwen3-4B Speculative Decode on L20 — Gate Study (2026-10)

**TL;DR:** A load-aware speculation gate (`PEGAINFER_SPEC_MAX_BATCH`) and a static draft trim (`PEGAINFER_SPEC_MAX_DRAFTS`) were built and measured on one L20 (sm_89) with Qwen3-4B and the two DeepSpec block-7 drafters. The **noise floor of the interleaved A/B protocol is 0.3–2.2 % (paired median)**, so the 5090-era ±1–2 % acceptance rules were never valid here; the measured **spec-off crossover is c32 for DFlash-block7 and c64 for DSpark-block7** (plain decode beats speculation by 4.8 % / 27 % at c64). The gate **recovers +46.6 % output tok/s at c64** (2115 → 3099) while leaving every below-threshold cell untouched (±1.2 %, inside the noise floor), and it is **reversible**: a gated round keeps the DFlash capture, and the child-process gate verifies on a real run that a gated request later speculates and accepts (140 gated / 8 speculating rows on the threshold-3 batch). The trim tells the same story from the other end: its sign **flips with concurrency** (it loses at c8 — −3.6 % at K=4, −15.6 % at K=2 — and at c16 for K=2; c16 K=4 changes sign across rounds and is unresolved; it wins +9.1 % at c32 over the untrimmed drafter, which brings it to **parity** with plain decode, not above it; at c64 the span ordering is monotone — K=7 2113 < K=4 2405 < K=2 2572 < gate 3083). **The optimal verify span shrinks as the batch grows**, so a static K is a load-adaptive lever in disguise, and #718's adaptive window is the natural next step. What the gate does *not* do is reach the plain decode: the gated one-token **verify** step costs ~5 % more than a plain **decode** step, because it runs the prefill-style verify path instead of the tuned decode path.

## Setup

| Item | Value |
| --- | --- |
| Host | one **NVIDIA L20** (46 GB, sm_89), CUDA 12.9 |
| Target | `Qwen3-4B` (7.6 GB bf16), local-disk copy |
| Drafters | `dflash_qwen3_4b_block7` (2.6 GB) and `dspark_qwen3_4b_block7` (2.8 GB), DeepSpec format (`block_size=7`, `num_anchors=512` ⇒ anchor-first, `target_layer_ids=[1,9,17,25,33]`) |
| Harness | `scripts/bench_http_serving.py` (the repo's own HTTP harness), **greedy** (`--temperature 0`), `--ignore-eos`, 128 output tokens |
| Prompts | 64 ShareGPT first turns (`ShareGPT_V3_unfiltered_cleaned_split.json`, `--prompt-count 64 --prompt-seed 512`) — real text, because synthetic `--prompt-words` prompts overstate the speculative accept rate |
| Protocol | **arms alternate** (A/B/A/B…), one server start per arm per round, one harness run per cell; every arm is `--no-prefix-cache` symmetric (loading a drafter forces the prefix cache off, so the spec-off arm must too) |
| Metric | output tok/s = `qps × 128`. The qwen3 server emits no `pegainfer_http_trace` lines, so the harness cannot source per-request token counts; with `--ignore-eos` and a fixed `max_tokens` every successful request emits exactly 128 tokens. |

L20 is **not** the 5090: absolute tok/s and crossover positions do not transfer. The 5090's `c16 plain 2277` is ~2× this box's `c16 plain 1150`.

## 0a — Noise floor (interleaved A/B, two *identical* arms)

The control row: two arms with the same configuration, alternating, four rounds. The paired per-round difference (normalized by the pair's mean — the statistic every later section quotes as "paired") is the number an acceptance gate has to beat: the same pairing the real A/B uses, so the two are comparable.

| c | paired \|Δ\| median | paired \|Δ\| max | within-arm spread (max) |
| --- | --- | --- | --- |
| 1 | 0.27 % | 0.44 % | 0.9 % |
| 8 | 1.30 % | 4.43 % | 4.7 % |
| 16 | 2.17 % | 3.91 % | 4.8 % |
| 32 | 1.27 % | 7.06 % | 8.3 % |
| 64 | 0.47 % | 0.57 % | 1.1 % |

**Reading:** the *paired* noise floor is 0.3–2.2 % (median) but individual pairs reach 4–7 % at c8–c32. A gate verdict needs **≥ ~2 % at the median and a consistent sign across rounds**; a single pair below ~4 % proves nothing. (For reference, the 5090 `vllm-bench` harness that produced the ±4 % figure in `models/qwen3/dspark-integration.md` had a *within-arm* spread of the same order — the two numbers are not the same statistic.)

## 0b — Per-drafter spec-off baseline (4 interleaved rounds)

Output tok/s, median of 4 rounds:

| c | plain (spec off, no prefix cache) | DFlash block7 | DSpark block7 |
| --- | --- | --- | --- |
| 1 | 84.5 | 164.7 (+95 %) | 183.4 (+117 %) |
| 8 | 608.0 | 909.4 (+50 %) | 1027.2 (+69 %) |
| 16 | 1136.0 | 1439.9 (+27 %) | 1563.0 (+38 %) |
| 32 | 1958.8 | 1865.6 (**−4.8 %**) | 2110.4 (+7.7 %) |
| 64 | 3266.0 | 2108.0 (**−35.5 %**) | 2380.8 (**−27.1 %**) |

**Crossover:** plain decode overtakes speculation between c16 and c32 for DFlash-block7, and between c32 and c64 for DSpark-block7. The c32 DFlash gap (−4.8 %) is only ~1.4× the noise floor's paired median and inside its max — treat it as *marginal*; the c64 gaps (−35 %, −27 %) are unambiguous.

DSpark beats the matched DFlash by **+8.6 % … +13.1 %** at every concurrency — a bigger margin than the +3.6 % geomean measured on the 5090, consistent with the L20 being more bandwidth-starved (the Markov head's extra draft cost is amortized differently).

## 0c — The gate (`PEGAINFER_SPEC_MAX_BATCH=32`, DFlash block7)

Burst workload (64 requests, so the active batch decays as the requests finish): 4 interleaved rounds, output tok/s.

| c | plain | DFlash (no gate) | DFlash + gate 32 | gate vs. no gate | gate vs. plain |
| --- | --- | --- | --- | --- | --- |
| 8 | 608.9 | 905.1 | 904.3 | −0.1 % | +48.5 % |
| 16 | 1150.3 | 1410.9 | 1428.1 | +1.2 % | +24.2 % |
| 32 | 1980.1 | 1853.1 | 1859.8 | +0.4 % | −6.1 % |
| 64 | 3263.0 | 2114.6 | **3099.3** | **+46.6 %** | −5.0 % |

- **Below the threshold nothing changes** (≤1.2 %, inside the noise floor) — the gate is a no-op there, as it must be.
- **At c64 the gate is worth +46.6 %** (ratio of medians; the mean-normalized paired statistic of 0a is a median 37.9 %, max 38.7 % — far outside the noise floor).
- The gate lands **~5 % short of the plain decode**: the gated step is a one-token *verify* (prefill-style path, paged attention, all-position logits, DFlash capture) while the plain step is the tuned *decode* path (split-KV attention, decode graphs). Closing that 5 % would need the gated step to run the decode path and capture the target hidden state there — a different design, not a tuning knob.

## Reversibility diagnostic (the latch that must not exist)

The per-round trace marks a load-gated round **explicitly** (`gated=1`) and carries the request id, the number of drafts that entered the verify span (`verified_draft`) and the accepted count (`accepted_draft`). Neither of the latter two can classify a round on its own: a near-budget request's span is truncated to one token (`verified_draft=0`), and a speculative round whose drafts were all rejected also logs `accepted_draft=0`.

`dflash_spec_levers_keep_losslessness` (in `tests/dflash_speculative_gate.rs`) asserts the transition per request on a small batch: four staggered budgets with `PEGAINFER_SPEC_MAX_BATCH=3`, so the batch starts above the threshold and shrinks below it as requests finish. The run must contain gated rounds, and **a request that was gated must later speculate and accept at least one draft**; the child's own lossless assertions cover the output.

**Run record** (one L20, `PEGAINFER_TEST_LOG=1`, 86.5 s, gate passed `1 passed / 0 failed`). Counts are trace rows — one per request per round:

| child | env | gated rows | speculating rows | max drafts |
| --- | --- | ---: | ---: | ---: |
| `dflash_speculative_greedy_matches_plain_greedy` | `MAX_BATCH=1` | 315 | 0 | 0 |
| `dflash_speculative_greedy_matches_plain_greedy` | `MAX_DRAFTS=4` | 0 | 116 | 4 |
| `dflash_concurrent_heterogeneous_is_lossless` | `MAX_DRAFTS=4` | 0 | 64 | 4 |
| `dflash_speculative_greedy_matches_plain_greedy` | `MAX_DRAFTS=2` | 0 | 142 | 2 |
| `dflash_concurrent_heterogeneous_is_lossless` | `MAX_DRAFTS=2` | 0 | 79 | 2 |
| `dflash_concurrent_heterogeneous_is_lossless` | `MAX_BATCH=3` | 140 | 8 | — |

The `MAX_BATCH=3` row is the resume case: the threshold-3 batch gates first and speculates later, and the gate's per-request assertion (a gated request later speculates **and** accepts) passed. The `MAX_BATCH=1` row shows the one-token round is the only path exercised there, and the `MAX_DRAFTS` rows show the gate off (`gated=0`) with no round exceeding K.

*(The measurement pod's toolkit is CUDA 12.9 while this tree references CUDA 13's `cudaErrorLossyQuery`; the local build wraps that one error-mapping case in a version guard. It is outside the speculative path and does not touch the harness.)*

The earlier c64 reversal count (8142 / 29) was taken from `accepted_draft`, which cannot tell a gated round from a fully-rejected speculative one, and is withdrawn. Reversibility is asserted per request by `dflash_spec_levers_keep_losslessness` through the explicit `gated=` marker; the run record above is that gate.

## 0d — Steady-state gate (128 requests, so the batch holds at the concurrency limit)

The burst workload above lets the active batch decay as the requests finish, so the gate is only active while the count is above the threshold. A serving workload keeps the count at the concurrency limit (requests arrive as others finish), so the same sweep was repeated with 128 requests (the 64-prompt pool cycles): 3 interleaved rounds.

| c | plain | DFlash (no gate) | DFlash + gate 32 | gate vs. no gate | gate vs. plain |
| --- | --- | --- | --- | --- | --- |
| 32 | 1984.3 | 1846.8 | 1862.3 | +0.8 % | −6.2 % |
| 64 | 3306.9 | 2071.4 | **3101.4** | **+49.7 %** | −6.2 % |

The steady-state numbers reproduce the burst ones: the gate is worth ~+50 % at c64 (mean-normalized paired \|Δ\| median 39.8 %, max 40.2 %) and is a wash at c32 (+0.8 %, inside the noise floor). **The gate's effect does not depend on the tail behaviour of the workload** — it is the per-step decision that matters.

### Reading the three cells

- **c32 — a wash.** The gated step is ~8× cheaper in rows (32 rows vs. 32×8) but commits 1 token per request instead of ~2.5, and the two effects cancel. The plain decode is still ~6 % ahead of both, so at c32 *nothing* helps: speculation is already at its break-even.
- **c64 — the gate wins big.** Speculation's extra rows cost more than the extra tokens buy (the target forward is bandwidth-bound and the rows scale the attention and the logits GEMM), so the ungated engine loses 37 % to plain; the gate removes the extra rows and recovers all but ~6 %.
- **The residual ~6 %** is the gap between the gated **verify** step and the plain **decode** step — different code paths (paged prefill attention + all-position logits + the DFlash capture vs. the split-KV decode attention + decode graphs), not a tuning knob.

### Recommended thresholds (this box, this model, these drafters)

| Drafter | crossover (plain overtakes speculation) | gate threshold |
| --- | --- | --- |
| DFlash block7 | c32 (marginal, −4.8 %) → c64 (unambiguous, −35.5 %) | **32** — a wash at c32, +50 % at c64, and it survives a decaying batch (a threshold of 64 would stop tripping as soon as the first request finishes) |
| DSpark block7 | c64 (unambiguous, −27.1 %); DSpark still wins at c32 (+7.7 %) | **64** |

The threshold is a property of (model, GPU, drafter) — it is not a constant, and it must be re-measured when any of the three changes.

## 0e/0f — The threshold must sit at or above the crossover

Two extra sweeps pin down *where* the gate may be armed.

**0e — below the threshold, off-bucket concurrencies (c9 / c17 / c30; none is a batch bucket), gate at 32:** the gate is a no-op, because none of those batches reaches 32. The per-cell medians differ by +0.6 % / +0.5 % / +0.2 % (individual paired rounds swing up to ~3 %, the same order as the 0a pair noise at those shapes) — inside the noise floor. Speculation still beats plain at all three (−36 % / −23 % / −18 % for plain), so the eager verify path (a non-bucket batch cannot use the captured piecewise graph) is still a win there.

**0f — above a *misplaced* threshold (gate at 16, concurrency 17 / 30):** the gate fires, and it **loses**: 1413 → 1082 at c17 (−23.4 %) and 1760 → 1477 at c30 (−16.1 %). The gated step lands exactly on the plain decode (1082 vs plain 1097 at c17, 1477 vs 1433 at c30) — i.e. the one-row verify costs about what a plain decode costs, while the multi-row verify it replaced was *cheaper per token* than both, because at those shapes the step is launch-bound and the extra rows are nearly free.

**The rule this yields:** the gate trades verify rows for rounds, so it only pays where the rows actually cost something — at high concurrency, where the target forward's row work dominates its launch overhead. Below the crossover the rows are nearly free, speculation wins, and arming the gate there throws that win away.

⇒ **Arm the gate at or above the measured crossover, and at or below the serving concurrency.** A threshold under the crossover is not a conservative choice; it is the expensive one.



## Losslessness (the hard gate)

`cargo test --release -p pegainfer-qwen3 --test dflash_speculative_gate`, same L20, Qwen3-4B + DFlash block7:

| Run | Result |
| --- | --- |
| Gate OFF (default) — the whole file | **6 passed / 0 failed** (35 s) — the gate is a no-op by default; nothing else regressed |
| Gate ON at `PEGAINFER_SPEC_MAX_BATCH=1` (every step gated) — `dflash_speculative_greedy_matches_plain_greedy` | **passed** (17 s) — the gated one-token round reproduces plain greedy |
| Gate ON at 1 — `dflash_concurrent_heterogeneous_is_lossless` | **passed** (12 s) — the concurrent batch path stays lossless while gated |

`PEGAINFER_SPEC_MAX_BATCH=1` gates *every* step (the batch is always ≥ 1), so these two runs exercise the gated round exclusively: the one-token verify commits exactly the tokens the plain decode would, and the DFlash capture it advances stays consistent.

## Static draft trim (`PEGAINFER_SPEC_MAX_DRAFTS`) — block 7

The intended target was **`Qwen3-4B-DFlash-b16`** (block 16, the checkpoint with the bimodal accept distribution). That checkpoint was **not available on the benchmark image** and was not downloaded for this run — only the two DeepSpec block-7 drafters were — so the trim was run on block 7 instead, with the same mechanism:

- `PEGAINFER_SPEC_MAX_DRAFTS=<k>` caps the *proposed* candidates. The drafter still rolls its whole block out (its cache still writes the `block_size`-slot in-fill tail every round, so the headroom is unchanged); only the first `k` candidates enter the verify span, so the target forward carries `1 + k` rows per request instead of 8.
- The verify buffers are sized to the trimmed span, and the span stays uniform across the batch, so a full-span step still satisfies the capture-shape ≡ replay-shape invariant — the same invariant the one-token gated round depends on.

**Losslessness (the hard gate for a new span dimension):** with the trim on, the greedy output still matches the plain decode baseline.

| Trim | `dflash_speculative_greedy_matches_plain_greedy` | `dflash_concurrent_heterogeneous_is_lossless` |
| --- | --- | --- |
| `PEGAINFER_SPEC_MAX_DRAFTS=4` (span 5) | **passed** (15.0 s) | **passed** (11.0 s) |
| `PEGAINFER_SPEC_MAX_DRAFTS=2` (span 3) | **passed** (15.4 s) | **passed** (12.0 s) |

**Throughput sweep** (3 interleaved rounds, output tok/s, median):

| c | plain | DFlash (K=7, span 8) | trim K=4 (span 5) | trim K=2 (span 3) | gate (span 1) |
| --- | --- | --- | --- | --- | --- |
| 8 | 606.4 | 904.1 | 868.1 (−4.0 %) | 764.2 (−15.5 %) | — |
| 16 | 1148.6 | 1397.4 | 1338.6 (−4.2 %) | 1246.8 (−10.8 %) | — |
| 32 | 1974.5 | 1846.2 | **2014.5 (+9.1 %)** | 1939.5 (+5.1 %) | 1859.8 (+0.4 %) |
| 64 | 3263.0 | 2113.3 | 2405.1 (+13.8 %) | 2571.6 (+21.7 %) | **3082.6 (+45.9 %)** |

(The c32 gate cell is from 0c; the c64 trim cells are a separate 3-round sweep, 0h. Paired \|Δ\| medians at c64: trim-vs-untrimmed 13.2 % (K=4) / 19.6 % (K=2), gate-vs-untrimmed 37.2 % — all far outside the 0.5 % noise floor at c64.)

**The trim's sign flips with concurrency — and the optimum span shrinks with the batch.**

- **Below c32 the trim loses where the rounds agree.** At c8 K=4 is negative in all three rounds (−5.4 / −3.5 / −3.6 %, outside the 1.3 % floor) and K=2 more so (−15.6 %); at c16 K=2 stays negative (−10.6 %) but **K=4 changes sign across rounds (−3.4 / −7.4 / +5.3) and is unresolved**. At those shapes the step is launch-bound, the verify rows are nearly free, and cutting them only shortens the accepted run — so the trim throws the speculation win away for nothing. (The 0f sweep found the same mechanism from the gate's side.)
- **At c32 the trim wins**: K=4 gives **+9.1 %** over the untrimmed drafter (paired \|Δ\| median 7.5 % vs. a ~1.3 % noise floor) and moves the drafter from **6.5 % below plain decode to parity with it** (per-round +1.8 / +2.2 / +0.6 %, median +1.8 % — below the 2 % bar this study applies, so parity, not a win). The optimum is interior there (K=4 2014 > K=2 1940 > K=7 1846).
- **At c64 the ordering is monotone in the span** — the fewer rows, the better: K=7 2113 < K=4 2405 < K=2 2572 < **gate (span 1) 3083**. The gate is the extreme trim and it wins by +45.9 %; no intermediate K comes close.

⇒ **The optimal verify span is a function of the batch, not a constant.** The row cost of the target forward grows with `batch × span` (attention over the context plus the all-position logits), while the cost of a shorter accepted run does not depend on the batch — so the span that maximises committed tokens per second *shrinks as concurrency rises*: K=4 at c32, K=0 (the gate) at c64 on this L20.

⇒ The static K and the load gate are the two ends of one lever, and #718's "adaptive per-request verify windows under load" is the natural next step: a span that shrinks with the batch, rather than a span chosen per checkpoint. **A static K is a load-adaptive lever in disguise** — it needs a concurrency range, not a constant, and the same rule as the gate's applies: apply it at or above the crossover, never below it.

## What this does and does not settle

- **Settled:** the gate is worth tens of percent at saturation, is a no-op below its threshold, and is reversible — the child-process gate verifies per request that a gated request later speculates and accepts (run record above). The threshold is a *measured* per-(model, GPU, drafter) number — on this L20 it is **c32 for DFlash-block7, c64 for DSpark-block7** — and it must be armed **at or above the crossover**, because below it the rows are free and the gate (or a trim) only throws the speculation win away.
- **Settled:** the static trim is lossless (K=4 and K=2 both pass the greedy and the concurrent heterogeneous gates) and its sign flips with concurrency — it wins at c32 (+9.1 % at K=4 over the untrimmed drafter, bringing it from 6.5 % below plain decode to parity: paired median +1.8 %, below the 2 % bar) and loses at c8 for both caps; c16 K=4 is unresolved. A static K is therefore a *load-adaptive* lever in disguise: it needs a concurrency range, not a constant.
- **Not settled:** the per-shape cost table (a `(bucket, span) → {graph ms, eager ms}` grid). Its two consumers were answered empirically instead — the gate's row trade by the gate A/B (0c), and the trim's row trade by the K sweep (0g) — but the table itself is not built, so the *shape* of the row cost (which shapes are launch-bound, which are row-bound) is inferred from end-to-end throughput rather than measured per shape.
- **Not settled:** the trim on DFlash-b16. Its target checkpoint (**`Qwen3-4B-DFlash-b16`**, block 16, the bimodal accept distribution that motivated it) was **not available on the benchmark image** and was not downloaded for this run — only the two DeepSpec block-7 drafters were — so the trim was measured on block 7, where the block-7 cost model had predicted no money; the L20 run disagreed, which is exactly why the b16 target still needs its own sweep on a host with that checkpoint.
- **Not transferable:** every absolute number here. The 5090 sweep in `models/qwen3/dspark-integration.md` is a different box with a different harness; the *relative* orderings (plain vs. spec-on, DSpark vs. DFlash, trim vs. no-trim) are the portable part.
