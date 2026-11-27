# Qwen3.5 Step-Plan Scheduling Policy

> **TL;DR:** Opt-in `StepPolicy::StallBounded` (`--qwen35-prefill-stall-ms <MS>`)
> adds a stateless scheduling policy (`plan_prefill_rows`) on top of `StepPlan`
> that bounds mixed prefill+decode step duration using a linear step-cost model
> fitted on H20 Qwen3.5-4B traces (`8.20 ms + 74.0 us/row + 106 us/decode`,
> `R^2 = 0.9972`). Because tighter stall bounds trade output throughput and
> deep-queue TTFT p99 for lower ITL p99/p99.9, `StepPolicy::Fixed` at
> `--max-prefill-tokens 1024` remains the default on both single-GPU and TP,
> and TP scratch remains derived solely from `--max-prefill-tokens`.
>
> **Last touched:** 2026-11

## Problem: Fixed Budgets Hide Long Stalls Behind ITL p99

When prefill and active decode share a mixed (`unified`) step on Qwen3.5-4B,
every active decode request sees an inter-token latency (ITL) equal to that
step's duration. Raising `--max-prefill-tokens` from `1024` to `4096` packs
four 1024-token prompts into one ~315 ms step instead of four ~85 ms steps:

- At concurrency 8 (64 prompts, 256 output tokens each = 16,384 decode tokens),
  the `1024` budget puts ~512/16,384 (3.1%) decode tokens on mixed steps, so
  `ITL p99` reports the ~83 ms step wall. At `4096`, fewer packed steps push
  the affected share below 1%, so `ITL p99` drops to `5.5 ms` while **`ITL
  p99.9` jumps to `235.7 ms` (max `316.4 ms`)** and `TTFT p50` more than
  doubles (`165.8 -> 395.8 ms`).
- At concurrency 16 and 16 req/s, `4096` raises `ITL p99.9` to `315.9 / 321.5 ms`
  and `TTFT p50` to `624.6 / 504.2 ms` for `+1.2% / +2.9%` throughput.

Reporting `ITL p99.9` alongside `ITL p99` prevents wide packing from hiding
multi-hundred-millisecond decode freezes below the 1% percentile line.

## Trace Fit: Prompt Rows to Step Duration

Measured on 1x NVIDIA H20 96 GB (driver 580.126.20, CUDA 13.0,
`PEGAINFER_CUDA_SM=90`, Triton 3.8.0, Qwen3.5-4B BF16) with
`PEGAINFER_ITL_DEBUG=1` across `--max-prefill-tokens` in
`{256, 512, 1024, 2048, 4096}` (`n = 1497` mixed steps):

| Step type | Prompt rows | Steps | Avg prompts/step | Avg active decodes | Step p50 (ms) | Step p95 (ms) | Step max (ms) | us / prompt token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure prefill | 256 | 12 | 1.00 | 0.0 | 22.82 | 32.28 | 40.96 | 89.1 |
| Pure prefill | 512 | 6 | 1.00 | 0.0 | 41.13 | 58.57 | 62.64 | 80.3 |
| Pure prefill | 1024 | 29 | 1.00 | 0.0 | 78.39 | 100.30 | 112.76 | 76.6 |
| Mixed (`unified`) | 256 | 756 | 1.00 | 14.3 | 28.33 | 30.57 | 42.92 | 110.6 |
| Mixed (`unified`) | 512 | 378 | 1.00 | 14.2 | 47.19 | 53.98 | 70.23 | 92.2 |
| Mixed (`unified`) | 1024 | 235 | 1.00 | 13.4 | 83.90 | 91.14 | 118.15 | 81.9 |
| Mixed (`unified`) | 2048 | 82 | 2.00 | 13.0 | 160.09 | 162.59 | 169.85 | 78.2 |
| Mixed (`unified`) | 3072 | 16 | 3.00 | 9.6 | 235.66 | 243.08 | 244.67 | 76.7 |
| Mixed (`unified`) | 4096 | 30 | 4.00 | 10.7 | 312.31 | 321.73 | 342.84 | 76.2 |

OLS linear fit over the 1,497 mixed steps:

```text
step_ms(P, D) = 8.20 + 0.0740 * P + 0.106 * D    (R^2 = 0.9972)
```

Decode-only step p50 scales from `3.56 ms` (`D = 1`) through `4.82 ms`
(`D = 8`) and `6.02 ms` (`D = 16`) to `6.81 ms` (`D = 32`).

## Policy Design (`pegainfer-qwen35/src/scheduler/plan.rs`)

All scheduling policy logic lives in pure, stateless types and functions in
`scheduler/plan.rs`:

- `StepCostModel { base_us, us_per_prefill_row, us_per_decode_row }` with
  `StepCostModel::H20_QWEN35_4B = { 8_200, 74, 106 }`.
  - `predict_step_us(P, D)` predicts mixed step duration in microseconds.
  - `min_reachable_stall_us()` returns `predict_step_us(1, 1)` (`8,380 us`,
    so `--qwen35-prefill-stall-ms` requires at least `9 ms`).
  - `max_prefill_rows_for_stall(max_stall_us, D)` inverts the linear model
    continuously in `D` (no decoder-width threshold), keeping a 1-row floor
    for liveness.
- `StepLoadSignals { pending_reqs, queued_prefill_tokens }` captures queue
  pressure right before `plan_step`.
- `StepPolicy::{Fixed, StallBounded { max_stall_us, cost_model }}` defaults to
  `Fixed`.
- `plan_prefill_rows(active_decodes, prefilling, remaining_budget, load, policy)`:
  - Returns `vec![]` when `remaining_budget == 0` or `prefilling.is_empty()`.
  - `Fixed`: unchanged FIFO packing/slicing up to `remaining_budget`.
  - `StallBounded`: computes `stall_cap = min(remaining_budget, max_prefill_rows_for_stall(max_stall_us, active_decodes))`.
    - **Pure prefill (`active_decodes == 0`)**: no live decoders exist to
      stall, so the front prompt may use up to `remaining_budget`. Followers
      pack only if they fit **whole** inside `stall_cap`.
    - **Mixed step (`active_decodes > 0`)**: hard row cap is `step_cap = stall_cap`.
      - If `first_rem <= step_cap`: pack **whole** FIFO followers inside
        `step_cap` (`pack_whole_followers`), never slicing a follower into a
        partial sliver.
      - If `first_rem > step_cap`: slice at `step_cap` only when a backlog is
        present and `tail + next_whole <= step_cap`; otherwise use balanced
        slicing `first_rem.div_ceil(first_rem.div_ceil(step_cap))` so the same
        step count yields a lower peak single-step stall.

### Why Partial Follower Slivers Are Disallowed

An earlier prototype packed leftover headroom greedily into the follower prompt
(for example, `[1024, 38]` at `stall_cap = 1062` for 90 ms, or `[503, 18]` at
`stall_cap = 521` for 50 ms). Because Qwen3.5's prefill path launches kernels
per request in the step, adding a second request for an 18–38 row sliver added
~6–9 ms of per-request launch overhead and split the follower across two steps.
Disallowing partial follower slivers improved `16 req/s` under the `50 ms` cap
from `TTFT p50 = 270.3 -> 150.9 ms` (-44%), `ITL p99 = 61.0 -> 54.5 ms`, and
throughput `1034.8 -> 1065.0 tok/s`, and brought the `90 ms / 4096 budget` arm
from `1097.3 -> 1149.3 tok/s` and `TTFT p50 = 295.0 -> 174.5 ms`.

### Validation & TP Invariants

- `--qwen35-prefill-stall-ms <MS>` defaults to unset (`StepPolicy::Fixed`).
- `check_launch` (invoked by both CLI validation and engine startup before
  weights load) rejects `0`, rejects targets below
  `cost_model.min_reachable_stall_us()` (`< 9 ms` on the built-in H20 4B fit),
  and rejects combining `--qwen35-prefill-stall-ms` with
  `--decode-overlap=stream`.
- Single-GPU and TP share the same default (`Fixed`, 1024 rows) and the same
  `plan_step` path. TP per-rank prefill scratch stays derived solely from
  `min(max_prefill_tokens, PREFILL_CHUNK_LEN)` in `tp_executor/worker.rs`.

## Same-Host H20 A/B Results

Hardware and build: 1x NVIDIA H20 96 GB (single device, no other workload),
driver 580.126.20, CUDA 13.0, Rust `1.99.0-nightly`, Triton 3.8.0, Qwen3.5-4B
BF16, release binary built with
`cargo build --release -p pegainfer-server --no-default-features --features qwen35`.

Workloads (all prompts 1024 tokens, greedy temperature 0, ignore-eos, seed 42,
`PEGAINFER_ITL_DEBUG` unset, 2 clean runs per cell, 0 failed requests):
- **Single stream (`c=1`)**: 20 prompts, 256 output tokens
- **Steady concurrency 8 (`c=8`)**: 64 prompts, 256 output tokens
- **Steady concurrency 16 (`c=16`)**: 64 prompts, 256 output tokens
- **Poisson 16 req/s (`qps=16`)**: 64 prompts, max concurrency 32, 128 output tokens

Two-run means (higher is better for throughput; lower is better for all latency columns):

| Workload | Policy configuration | Throughput (tok/s) | TPOT (ms) | TTFT p50 / p99 (ms) | ITL p50 / p99 / **p99.9** / max (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| `c=1` (1024/256) | **Default (`1024` fixed)** | 257.8 | 3.57 | 80.5 / 90.2 | 3.57 / 3.7 / **4.5** / 10.9 |
| | `4096` fixed | 257.7 | 3.57 | 80.9 / 90.2 | 3.57 / 3.7 / **4.6** / 10.8 |
| | `50 ms` stall cap (`1024` budget) | 257.8 | 3.57 | 80.6 / 91.8 | 3.57 / 3.7 / **5.1** / 10.4 |
| | `90 ms` stall cap (`4096` budget) | 257.3 | 3.58 | 80.6 / 101.1 | 3.57 / 3.7 / **4.6** / 12.3 |
| | `165 ms` stall cap (`4096` budget) | 257.3 | 3.57 | 80.9 / 101.5 | 3.57 / 3.7 / **4.7** / 10.7 |
| `c=8` (1024/256) | **Default (`1024` fixed)** | 1091.8 | 6.62 | 165.8 / 615.0 | 4.82 / 83.1 / **90.2** / 91.5 |
| | `4096` fixed | 1102.7 | 5.64 | 395.8 / 630.4 | 4.82 / 5.5 / **235.7** / 316.4 |
| | `50 ms` stall cap (`1024` budget) | 1062.9 | 6.88 | 141.2 / 663.9 | 4.83 / **49.5** / **55.4** / 56.5 |
| | `90 ms` stall cap (`4096` budget) | 1090.7 | 6.63 | 165.8 / 616.3 | 4.82 / 83.0 / **91.0** / 95.1 |
| | `165 ms` stall cap (`4096` budget) | 1097.3 | 6.22 | 317.4 / 607.6 | 4.82 / 82.5 / **160.5** / 172.9 |
| `c=16` (1024/256) | **Default (`1024` fixed)** | 1452.0 | 9.84 | 168.5 / 1286.9 | 6.02 / 84.8 / **91.3** / 94.4 |
| | `4096` fixed | 1469.6 | 8.62 | 624.6 / 1257.5 | 6.02 / 85.3 / **315.9** / 322.4 |
| | `50 ms` stall cap (`1024` budget) | 1374.0 | 10.41 | 154.1 / 1460.5 | 6.03 / **54.3** / **57.3** / 59.1 |
| | `90 ms` stall cap (`4096` budget) | 1449.6 | 9.82 | 172.1 / 1291.1 | 6.02 / 84.7 / **93.8** / 96.0 |
| | `165 ms` stall cap (`4096` budget) | 1462.6 | 9.39 | 320.9 / 1241.8 | 6.01 / 160.6 / **164.5** / 168.3 |
| `qps=16` (1024/128) | **Default (`1024` fixed)** | 1149.9 | 20.96 | 176.7 / 1111.1 | 6.84 / 92.1 / **93.8** / 94.3 |
| | `4096` fixed | 1183.1 | 19.07 | 504.2 / 1074.5 | 6.82 / 315.2 / **321.5** / 321.9 |
| | `50 ms` stall cap (`1024` budget) | 1065.0 | 22.27 | 150.9 / 1465.3 | 6.92 / **54.5** / **65.8** / 66.1 |
| | `90 ms` stall cap (`4096` budget) | 1149.3 | 20.97 | 174.5 / 1119.5 | 6.84 / 90.5 / **94.5** / 94.7 |
| | `165 ms` stall cap (`4096` budget) | 1169.5 | 20.15 | 323.6 / 1054.2 | 6.83 / 162.7 / **170.6** / 171.1 |

> Baseline stability note: a repeated control run on `Default (1024 fixed)`
> hit a single host-jitter spike on `qps=16` run 2 (`ITL max 587.6 ms`,
> `1065.2 tok/s`), while its `c=1/c=8/c=16` and `qps=16` run 1 matched the clean
> control pair above (`qps=16` `1150.1 / 1149.7 tok/s`, `ITL p99.9 93.9 / 93.8 ms`)
> within 1%.

### Decision: Keep Default `Fixed (1024)`

No single stall cap strictly dominates `Default (1024 fixed)` across all
metrics:

- **`50 ms` stall cap**: cuts mixed-load `ITL p99` by **35–41%** and `ITL p99.9`
  by **30–39%**, improves `TTFT p50` across `c=8/c=16/qps=16`, and leaves `c=1`
  unchanged — at the cost of `-2.6% / -5.4% / -7.4%` output throughput and
  `+8% / +13% / +32%` `TTFT p99`.
- **`90 ms` stall cap with `4096` budget**: eliminates raw `4096`'s
  `236–322 ms` `ITL p99.9` freezes and matches `1024` within `0.2%` throughput,
  while allowing solo long prompts (`active_decodes == 0`) to prefill up to
  `4096` tokens without chunk overhead.
- **`165 ms` stall cap with `4096` budget**: splits the difference between
  `1024` and `4096` (`ITL p99.9 ~161–171 ms`, `+0.5% to +1.7%` throughput,
  better `TTFT p99`, higher `TTFT p50`).

`StepPolicy::Fixed` (`--max-prefill-tokens 1024`) therefore remains the default
and `--qwen35-prefill-stall-ms <MS>` is an explicit opt-in flag.

## Claim Boundary

- `StepCostModel::H20_QWEN35_4B` is calibrated on single-GPU H20 with
  Qwen3.5-4B BF16. Other GPU architectures, larger model sizes (9B/27B), or TP
  launches have different per-row slopes and should supply a calibrated
  `StepCostModel` if strict millisecond accuracy is required.
- Verification covers unit tests (`scheduler::plan`, `model_line`, `lib`),
  `clippy -D warnings`, single-GPU and TP2 `scheduler::e2e_tests`,
  `scheduler::chunked_prefill_tests`, the 6 non-ignored single-GPU cases of
  `tests/prefix_cache` (the 2 `#[ignore]` TP2 prefix-cache cases were not run),
  and the 5-arm H20 serving matrix above.

## Verification Commands

```bash
# Unit tests and lint
cargo clippy --release -p pegainfer-qwen35 --features qwen35 --lib --tests -- -D warnings
cargo test --release -p pegainfer-qwen35 --features qwen35 --lib scheduler::plan
cargo test --release -p pegainfer-qwen35 --features qwen35 --lib model_line
cargo test --release -p pegainfer-frontend --lib

# GPU E2E gates (with PEGAINFER_TEST_MODEL_PATH=/path/to/Qwen3.5-4B)
CUDA_VISIBLE_DEVICES=0,1 cargo test --release -p pegainfer-qwen35 --features qwen35 --lib scheduler::e2e_tests -- --test-threads=1
CUDA_VISIBLE_DEVICES=0,1 cargo test --release -p pegainfer-qwen35 --features qwen35 --lib scheduler::e2e_tests::test_e2e_qwen35_scheduler_tp2 -- --ignored --test-threads=1
CUDA_VISIBLE_DEVICES=0 cargo test --release -p pegainfer-qwen35 --features qwen35 --lib scheduler::chunked_prefill_tests -- --test-threads=1
CUDA_VISIBLE_DEVICES=0,1 cargo test --release -p pegainfer-qwen35 --features qwen35 --test prefix_cache -- --test-threads=1
```
