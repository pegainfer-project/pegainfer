# Qwen3.5 Step Plan

> **TL;DR:** Every scheduler step is one `StepPlan`: a decode flag plus the
> prompt rows assigned to the FIFO front of the prefill queue, charged against
> one `--max-prefill-tokens` budget that defaults to a single 1024-row chunk.
> The step-kind enum and the `--qwen35-scheduler-policy` flag are gone; the
> executor reads the plan instead of re-deriving the step kind.
>
> **Last touched:** 2026-11

## Shape

The scheduler hands the executor facts, not a mode:

- `decode: bool` — live decode requests take this step.
- `prefill: Vec<usize>` — prompt rows for the first `prefill.len()` queued
  prompts, in FIFO order.

`plan_step(active_decodes, budget, prefilling, policy)` is the only place
the budget is divided, delegating prompt-row allocation to the pure function
`plan_prefill_rows`. The budget counts prompt rows; live decode requests
contribute one row each and do not consume it. `--max-prefill-tokens` overrides
the default on both single-GPU and TP launches, and the opt-in stall-bounded
policy (`--qwen35-prefill-stall-ms`) is documented in
`docs/models/qwen35/step-plan-policy.md`.

## Invariants

- Prefill rows come from the FIFO front. A prompt that crosses the budget is
  sliced; its tail is the first prompt resumed by the next plan.
- A plan never schedules more prompt rows than the budget.
- The budget must be positive. `plan_step` rejects zero, the CLI rejects it
  before loading, and `check_launch` rejects it again for callers that skip
  the CLI.
- The executor starts from the planned prompts and rows. The one shortening
  left is the prefix cache: `take_prefill_chunks` may trim a scheduled
  prompt's chunk to a 256-token snapshot boundary so a snapshot lands on a
  stride. A chunk that cannot reach the next boundary runs as planned.
  Telemetry reports the rows actually forwarded, not the planned count.

## Driver

Planning is one path; execution keeps the paths each backend requires.
`run_step` drains the planned prompts and dispatches on the data: prefill-only
goes to `prefill_batch`, decode-only to `decode_step`, both to the unified path
or the overlap lane when the backend offers it. The overlap lane is an
execution mode of a mixed plan, not a separate plan kind. While an async
prefill is in flight, the decode that keeps moving is a decode-only plan; it
runs through the same driver.

## Removed

The change is against `main`, where the scheduler had:

- `ExecutionPlan::{Prefill, Decode, Unified}` and `build_next_plan`
- `choose_prefill_budget` with the `0` sentinel and
  `DECODE_FINISH_WINDOW_TOKENS`
- `plan_prefill_chunks`, superseded by `plan_step`
- `Qwen35SchedulerPolicy` (`off` / `auto`) and the
  `--qwen35-scheduler-policy` CLI flag

Migration: drop `--qwen35-scheduler-policy` from launch commands; the default
step is one 1024-row chunk on every path, and `--max-prefill-tokens` still sets
the budget explicitly. The `auto` finishing tick and the `auto` + shared-SM
comparison test are gone with the policy; no other flag changes meaning.

## Tests

`cargo test --release -p pegainfer-qwen35 --features qwen35 --lib scheduler::plan` covers
the one-chunk default, FIFO packing at an explicit wider budget, slicing at
the budget, resuming a sliced tail, prefill-only and decode-only plans, an
unbounded budget, empty plans, and the zero-budget rejection.
`scheduler::chunked_prefill_tests` keeps the end-to-end gate that a chunked run
matches an unchunked run token for token.
