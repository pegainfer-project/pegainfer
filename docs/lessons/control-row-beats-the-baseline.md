# A Control Row That Beats the Baseline Means Your Harness Is Noisy

> **TL;DR:** The most valuable row in an A/B sweep is the one that should have been *impossible* to win. Two headline rules, both learned the hard way on the DSpark confidence-truncation sweep (#717):
>
> - **A control that does strictly more work cannot legitimately be faster.** If it is, you have just measured your harness's noise floor, and it is the ceiling on every delta you are about to quote.
> - **Measure that floor before you design the acceptance gate, not after.** A 4.4% single-run control movement at c16 makes a 2% criterion unreadable there; a paired, interleaved floor is how you get below that.

Companion to [profile-diff-before-blaming-transport.md](profile-diff-before-blaming-transport.md), which is the profiling-discipline sibling. This one is about **how much a number is worth** before you write it down.

## The incident

The DSpark Phase 2 sweep tested **static-threshold confidence truncation**: cut the proposed draft block at the first position whose predicted survival probability falls below τ, so the verify span shrinks and fewer target rows are paid for. Four thresholds (τ = 0.40 / 0.50 / 0.60 / 0.90) plus a static baseline, at c1 / c8 / c16, greedy, 128 out tokens, 30 text+code prompts.

The sweep checked its two acceptance criteria and missed both. It also ran a **threshold-0 arm**: instrument the confidence head, compute every `c_k`, truncate nothing. That arm does *strictly more work* than the static baseline — one extra `[2816→1]` GEMM per draft block — so it cannot legitimately come out ahead.

It came out ahead anyway: the instrumented arm moved +0.03% / +1.2% / **+4.4%** at c1 / c8 / c16 (full table in [the Phase 2 section](../models/qwen3/dspark-integration.md#phase-2--confidence-head--static-threshold-draft-truncation--no-win-not-merged)).

## Failure 1 — the control was in the table and easy to read past

The threshold-0 row is the control, and reading the sweep without it is the failure mode this lesson is about. A row that *cannot* win and then wins is not a data point about the feature — it is a **measurement of the floor**. At c16 it says a 4.4% swing is just what happens between two runs; at c1 the same control moved 0.03%, so the large c1 regression is real.

That reframes the result. At c16 the mild-threshold deltas are inside the control's 4.4% movement, i.e. **no result at all**; at c8 only τ=0.40 is inside the 1.2% control. What survives cleanly is the harmful direction: the c1 regression is monotone in τ and far outside 0.03%, and τ≥0.50 at c8 / τ≥0.60 at c16 are outside their cells. What does not survive is the question the sweep was built to answer — whether *mild* truncation pays.

## Failure 2 — a missed acceptance bar is not a refutation

Missing both criteria is the record, and it is a result about this configuration. The defensible conclusion is narrower than a blanket verdict: **the c1 harm is real, and at c8/c16 only the mildest cells are inside the control's movement** (Failure 1). That is not the same as a refutation of confidence truncation — for those cells it is the absence of the resolution needed to test the mild end, and writing "negative result" for it is how a live hypothesis gets closed by accident.

The cost model does not contradict that reading: for block7 at c16 it predicts the mild thresholds lose (τ=0.5's ~4.3% row saving is outweighed by ~9.2% more rounds), which is what the sweep measured. The open question is a different configuration with a larger row share, not a different reading of this one.

## Failure 3 — the mechanism was attributed to the feature

At c1, the only retention number we have (91.6% at τ=0.5) alone predicts ~−8.4%, so **lost tokens are most of the c1 regression**. A **graph miss** is a plausible *extra* cost (a per-request variable span misses the captured shape and the step runs **eager**), but that part is inferred: the graph-vs-eager A/B in the Phase 2 section is a different checkpoint and throughput, so it cannot be added to this number.

c1 is launch-bound, where a graph miss is expensive and trimming buys nothing. At higher concurrency the row-marginal share grows, where trimming *could* pay. The failure was treating "c1 is bad" as "trimming is bad" — a generalization across a regime boundary that the mechanism, not the data, should have told you about. Split the result by regime before generalizing.

## The rules

1. **Put a control row in every sweep, and read it first.** The control does the *same work plus a little more* and must not win. When it does, that number is your noise floor.
2. **Report the floor before the acceptance gate.** A criterion of "±2%" is a claim about the harness's resolution. If a single-run control moves 4.4% at that concurrency, a 2% criterion is unreadable there; either pair and interleave repeats to shrink the floor (the L20 gate study got 0.3–2.2%) or require a bigger effect.
3. **Measure the floor the way the gate will be applied.** If acceptance is an interleaved same-session A/B, the floor must be an interleaved same-session A/B too. Same-arm repeats measure a different quantity (run-to-run drift) than paired differences (what the gate will see), and the two are not comparable.
4. **Distinguish "harmful" from "unresolvable".** "This hurts" and "we cannot see whether this helps" are different claims with different consequences. The second one means the experiment is unfinished, not that the idea is dead.
5. **Attribute a regression to a mechanism before generalizing it.** Ask what the change did to the *other* subsystems in the step (graph hit rate, kernel selection, launch count) before concluding what it did to the feature. A regression that tracks a graph miss is a graph-miss result.

## Where this landed

- `docs/models/qwen3/dspark-integration.md` — the Phase 2 section carries the control table, the per-cell control movement, and the 5090 cost model. A re-measurement on an L20 (`benchmarks/qwen3-4b-spec-decode-l20.md`) is why the cost model must be per-GPU rather than per-checkpoint.
