# A Control Row That Beats the Baseline Means Your Harness Is Noisy

> **TL;DR:** The most valuable row in an A/B sweep is the one that should have been *impossible* to win. Two rules, both learned the hard way on the DSpark confidence-truncation sweep (#717):
>
> - **A control that does strictly more work cannot legitimately be faster.** If it is, you have just measured your harness's noise floor, and it is the ceiling on every delta you are about to quote.
> - **Measure that floor before you design the acceptance gate, not after.** A ±4% floor makes every ≤2% criterion unsatisfiable — but you cannot know the floor's size until you measure it.

Companion to [profile-diff-before-blaming-transport.md](profile-diff-before-blaming-transport.md), which is the profiling-discipline sibling. This one is about **how much a number is worth** before you write it down.

## The incident

The DSpark Phase 2 sweep tested **static-threshold confidence truncation**: cut the proposed draft block at the first position whose predicted survival probability falls below τ, so the verify span shrinks and fewer target rows are paid for. Four thresholds (τ = 0.40 / 0.50 / 0.90) plus a static baseline, at c1 / c8 / c16, greedy, 128 out tokens, 30 text+code prompts.

The headline read was "truncation is harmful at c1 (−9.2%) and roughly neutral at c8/c16". That read was **built on an unexamined control**. The sweep also ran a **threshold-0 arm**: instrument the confidence head, compute every `c_k`, truncate nothing. That arm does *strictly more work* than the static baseline — one extra `[2816→1]` GEMM per draft block — so it cannot legitimately come out ahead.

It came out ahead anyway:

| mode | c1 | c8 | c16 |
| --- | ---: | ---: | ---: |
| static block7 | 358.0 | 1847.7 | 2709.0 |
| **threshold 0 (instrument only, no truncation)** | **358.1 (+0.03%)** | **1869.3 (+1.2%)** | **2829.1 (+4.4%)** |
| τ=0.40 | 324.9 (−9.2%) | 1856.8 (+0.5%) | 2749.7 (+1.5%) |
| τ=0.50 | 306.2 (−14.5%) | 1733.8 (−6.2%) | 2671.4 (−1.4%) |
| τ=0.90 | 243.0 (−32.1%) | 1393.6 (−24.6%) | 2215.3 (−18.2%) |

## Failure 1 — the control was in the table and nobody read it

The threshold-0 row was present, labelled, and still got read as "instrumentation is free" instead of "the instrumented arm beat the plain arm by 4.4%". A row that *cannot* win and then wins is not a data point about the feature — it is a **measurement of the floor**. It says: **on this harness, at this concurrency, a ±4% swing is just what happens between two runs.**

That reframes the whole table. τ=0.40's +1.5% at c16 and τ=0.50's −1.4% are not "a small win" and "a small loss" — they are two points inside a ±4% band, i.e. **no result at all**. Only c1 survives: −9.2% → −14.5% → −32.1%, monotone in τ, an order of magnitude outside the band.

## Failure 2 — the "negative result" was a "no result"

The sweep's conclusion was written up as *truncation does not pay*. The defensible conclusion is narrower and different: **truncation is clearly harmful at c1, and unresolvable at c8/c16.** Those are not the same claim. The second one is not a refutation of confidence truncation — it is the absence of the measurement capability needed to test it. Writing "negative result" for the second is how a live hypothesis gets closed by accident.

The tell is the offline counterfactual, which said the opposite: at τ=0.5, truncation removes 47.8% of the verified rows while keeping 91.6% of the tokens. The offline account was positive; the online account was eaten by something else entirely (a CUDA-graph miss, below). A claim of the form "the feature does not work" that is contradicted by the offline model is a claim about the *harness* until proven otherwise.

## Failure 3 — the mechanism was attributed to the feature

Once the c1 regression was real, it was read as "truncation loses accepted tokens". It is not. Truncating the span makes the verify row count differ from `batch_size × span`, so the captured-graph `full_shape` gate fails and the step runs **eager**. The piecewise verify graph is worth **+9.3%** on its own at c1 (250.9 → 274.3 tok/s, same-session A/B), and τ=0.40 measured **−9.2%**. The c1 regression is therefore almost entirely **losing the graph**, with token loss contributing ≈0.7%.

c1 is launch-bound, where a graph miss is expensive and trimming buys nothing. c16 is compute-bound, where trimming *could* pay. The failure was treating "c1 is bad" as "trimming is bad" — a generalization across a regime boundary that the mechanism, not the data, should have told you about. Split the result by regime before generalizing.

## The rules

1. **Put a control row in every sweep, and read it first.** The control does the *same work plus a little more* and must not win. When it does, that number is your noise floor.
2. **Report the floor before the acceptance gate.** A criterion of "±2%" is a claim about the harness's resolution. If the floor is ±4%, the criterion is unsatisfiable, and no amount of repetition fixes it — the harness must get quieter (or the effect must get bigger).
3. **Measure the floor the way the gate will be applied.** If acceptance is an interleaved same-session A/B, the floor must be an interleaved same-session A/B too. Same-arm repeats measure a different quantity (run-to-run drift) than paired differences (what the gate will see), and the two are not comparable.
4. **Distinguish "harmful" from "unresolvable".** "This hurts" and "we cannot see whether this helps" are different claims with different consequences. The second one means the experiment is unfinished, not that the idea is dead.
5. **Attribute a regression to a mechanism before generalizing it.** Ask what the change did to the *other* subsystems in the step (graph hit rate, kernel selection, launch count) before concluding what it did to the feature. A regression that tracks a graph miss is a graph-miss result.
6. **When the offline model and the online measurement disagree, suspect the online path first.** A positive offline counterfactual plus a negative online number is a harness or mechanism bug until proven otherwise.

## Where this landed

- `docs/models/qwen3/dspark-integration.md` — the Phase 2 section now carries the corrected reading, the control table, the noise floor, and the 5090 cost model (row marginal ~9% of step time against a +18% round-count cost, so trimming block7 cannot pay *there*; a re-measurement on an L20 found the opposite at c32/c64, which is why the cost model must be per-GPU rather than per-checkpoint — see `benchmarks/qwen3-4b-spec-decode-l20.md`).
- The sweep's own conclusion is recorded as **"c1 harmful, c8/c16 unresolvable"** — not as a refutation of confidence truncation.
