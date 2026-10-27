# A Control Row That Beats the Baseline Means Your Harness Is Noisy

> **TL;DR:** The most valuable row in an A/B sweep is the one that should have been *impossible* to win. Two headline rules, both learned the hard way on the DSpark confidence-truncation sweep (#717):
>
> - **A control that does strictly more work cannot legitimately be faster.** If it is, you have just measured your harness's noise floor, and it is the ceiling on every delta you are about to quote.
> - **Measure that floor before you design the acceptance gate, not after.** A ±4% floor makes every ≤2% criterion unsatisfiable — but you cannot know the floor's size until you measure it.

Companion to [profile-diff-before-blaming-transport.md](profile-diff-before-blaming-transport.md), which is the profiling-discipline sibling. This one is about **how much a number is worth** before you write it down.

## The incident

The DSpark Phase 2 sweep tested **static-threshold confidence truncation**: cut the proposed draft block at the first position whose predicted survival probability falls below τ, so the verify span shrinks and fewer target rows are paid for. Four thresholds (τ = 0.40 / 0.50 / 0.60 / 0.90) plus a static baseline, at c1 / c8 / c16, greedy, 128 out tokens, 30 text+code prompts.

The headline read was "truncation is harmful at c1 (−9.2% for τ=0.40) and roughly neutral at c8/c16 for the mild thresholds". That read was **built on an unexamined control**. The sweep also ran a **threshold-0 arm**: instrument the confidence head, compute every `c_k`, truncate nothing. That arm does *strictly more work* than the static baseline — one extra `[2816→1]` GEMM per draft block — so it cannot legitimately come out ahead.

It came out ahead anyway: the instrumented arm beat the baseline by **4.4% at c16** (full table in [the Phase 2 section](../models/qwen3/dspark-integration.md#phase-2--confidence-head--static-threshold-draft-truncation--no-win-not-merged)).

## Failure 1 — the control was in the table and nobody read it

The threshold-0 row was present, labelled, and still got read as "instrumentation is free" instead of "the instrumented arm beat the plain arm". A row that *cannot* win and then wins is not a data point about the feature — it is a **measurement of the floor**. It says: **on this harness, at this concurrency, a ±4% swing is just what happens between two runs.**

That reframes the whole result. The mild-threshold deltas are points inside a ±4% band, i.e. **no result at all**. What survives cleanly is the harmful direction: the c1 regression is monotone in τ and far outside the band, and τ≥0.60 is outside it at c8/c16 too. What does not survive is the question the sweep was built to answer — whether *mild* truncation (τ=0.40 at c8/c16, τ=0.50 at c16) pays.

## Failure 2 — the "negative result" was a "no result"

The sweep's conclusion was written up as *truncation does not pay*. The defensible conclusion is narrower and different: **every threshold is clearly harmful at c1, and τ≥0.60 is clearly harmful at c8/c16 (at c8 so is τ=0.50), while the mildest cells (τ=0.40 at c8/c16, τ=0.50 at c16) are unresolvable.** Those are not the same claim. The second one is not a refutation of confidence truncation — it is the absence of the measurement capability needed to test it. Writing "negative result" for the second is how a live hypothesis gets closed by accident.

The tell is the offline counterfactual recorded in the Phase 2 section: it was positive. The online account is confounded by a likely CUDA-graph miss. A claim of the form "the feature does not work" that is contradicted by the offline model is a claim about the *harness* until proven otherwise.

## Failure 3 — the mechanism was attributed to the feature

Once the c1 regression was real, it was read as "truncation loses accepted tokens". That is not a measured decomposition: a **graph miss** is the plausible mechanism, and the graph-vs-eager A/B in the Phase 2 section is on a different checkpoint and throughput, so the attribution is inferred. Truncating each request's span independently makes the batch's row count differ from `batch_size × span`, so the captured-shape lookup misses and the step runs **eager**; a uniform static trim now has its own captured variant.

c1 is launch-bound, where a graph miss is expensive and trimming buys nothing. At higher concurrency the row-marginal share grows, where trimming *could* pay. The failure was treating "c1 is bad" as "trimming is bad" — a generalization across a regime boundary that the mechanism, not the data, should have told you about. Split the result by regime before generalizing.

## The rules

1. **Put a control row in every sweep, and read it first.** The control does the *same work plus a little more* and must not win. When it does, that number is your noise floor.
2. **Report the floor before the acceptance gate.** A criterion of "±2%" is a claim about the harness's resolution. If the floor is ±4%, the criterion is unsatisfiable, and no amount of repetition fixes it — the harness must get quieter (or the effect must get bigger).
3. **Measure the floor the way the gate will be applied.** If acceptance is an interleaved same-session A/B, the floor must be an interleaved same-session A/B too. Same-arm repeats measure a different quantity (run-to-run drift) than paired differences (what the gate will see), and the two are not comparable.
4. **Distinguish "harmful" from "unresolvable".** "This hurts" and "we cannot see whether this helps" are different claims with different consequences. The second one means the experiment is unfinished, not that the idea is dead.
5. **Attribute a regression to a mechanism before generalizing it.** Ask what the change did to the *other* subsystems in the step (graph hit rate, kernel selection, launch count) before concluding what it did to the feature. A regression that tracks a graph miss is a graph-miss result.
6. **When the offline model and the online measurement disagree, suspect the online path first.** A positive offline counterfactual plus a negative online number is a harness or mechanism bug until proven otherwise.

## Where this landed

- `docs/models/qwen3/dspark-integration.md` — the Phase 2 section carries the corrected reading, the control table, the noise floor, and the 5090 cost model. A re-measurement on an L20 (`benchmarks/qwen3-4b-spec-decode-l20.md`) is why the cost model must be per-GPU rather than per-checkpoint.
- The sweep's own conclusion is recorded as **"every threshold harmful at c1; τ≥0.60 harmful at c8/c16; the mildest cells unresolved"** — not as a refutation of confidence truncation.
