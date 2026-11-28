# Qwen3.5 Step-Plan Scheduling Policy

> **TL;DR:** Opt-in `StepPolicy::StallBounded` (`--qwen35-prefill-stall-ms <MS>`)
> adds a stateless scheduling policy (`plan_prefill_rows`) on top of `StepPlan`
> that bounds mixed prefill+decode steps using a 5-term physical step-cost model
> calibrated on H20 Qwen3.5-4B traces across prompt counts (`1..=31` prompts/step)
> and attended KV lengths (`cursor 0..=64,512`, `R^2 = 0.9985`). Because tighter
> stall targets trade throughput and deep-queue TTFT p99 for lower ITL
> p99/p99.9, `StepPolicy::Fixed` at `--max-prefill-tokens 1024` remains the
> default on both single-GPU and TP, and TP scratch remains derived solely from
> `--max-prefill-tokens`.
>
> **Last touched:** 2026-11

## Problem: Fixed Budgets Hide Long Stalls Behind ITL p99

When prefill and active decode share a mixed (`unified`) step on Qwen3.5-4B,
every active decode request sees an inter-token latency (ITL) equal to that
step's duration. Raising `--max-prefill-tokens` from `1024` to `4096` packs
four 1024-token prompts into one ~315 ms step instead of four ~84 ms steps:

- At concurrency 8 (64 prompts, 256 output tokens each = 16,384 decode tokens),
  the `1024` budget puts ~512/16,384 (3.1%) decode tokens on mixed steps, so
  `ITL p99` reports the ~83 ms step wall. At `4096`, fewer packed steps push
  the affected share below 1%, so `ITL p99` drops to `5.2 ms` while **`ITL
  p99.9` jumps to `235.4 ms` (max `313.0 ms`)** and `TTFT p50` more than
  doubles (`165.8 -> 394.6 ms`).
- At concurrency 16 and 16 req/s, `4096` raises `ITL p99.9` to `313.2 / 315.3 ms`
  and `TTFT p50` to `624.5 / 500.8 ms` for `+1.2% / +2.3%` throughput.

Reporting `ITL p99.9` alongside `ITL p99` prevents wide packing from hiding
multi-hundred-millisecond decode freezes below the 1% percentile line.

## Calibration Trace: Prompt Count, Prompt Rows, and Attended KV Length

Measured on 1x NVIDIA H20 96 GB (driver 580.126.20, CUDA 13.0,
`PEGAINFER_CUDA_SM=90`, Triton 3.8.0, Qwen3.5-4B BF16) using
`PEGAINFER_ITL_DEBUG=1` across two calibration sweeps (`n = 174` mixed steps):

### 1. Per-prompt fixed overhead (short prompts packed under `--max-prefill-tokens 4096`)

Because the executor runs one prefill pass per prompt in the step, packing many
short prompts multiplies the per-prompt launch cost even when total rows are small:

| Prompt length | Packed prompts (`prefill_reqs`) | Total prompt rows | Active decodes (med) | Step p50 / max (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 16 | 26 | 12.31 / 12.34 |
| 16 | 5 | 80 | 27 | 36.32 / 36.34 |
| 16 | 24 | 384 | 8 | 149.30 / 149.31 |
| 32 | 31 | 992 | 1 | 203.25 / 203.25 |
| 64 | 30 | 1920 | 1 | 251.18 / 251.18 |
| 128 | 31 | 3968 | 1 | 400.52 / 400.62 |
| 256 | 15 | 3840 | 1 | 330.38 / 331.73 |
| 512 | 7 | 3584 | 1 | 279.44 / 279.97 |
| 1024 | 1 | 1024 | 3 | 82.02 / 82.08 |
| 1024 | 4 | 4096 | 1 | 310.01 / 310.44 |

### 2. Attended KV / cursor length scaling (`65,536`-token and `32,768`-token continuations, `decode_n = 1`)

Full-attention prefill reads paged KV through `cursor + rows`, so per-row work
grows linearly with `cursor`:

| Chunk rows | `cursor = 0` | `cursor = 7,168 / 7,680` | `cursor = 15,360 / 15,872` | `cursor = 31,744 / 32,256` | `cursor = 64,512` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 86.96 ms | 100.88 ms | 116.15 ms | 146.47 ms | 208.23 ms |
| 512 | 46.38 ms | 53.68 ms | 61.44 ms | 77.35 ms | — |

### 3. Five-term physical cost model

OLS fit over the 174 mixed steps (`R^2 = 0.9985`):

```text
step_us = base_us
        + us_per_decode_row * active_decodes
        + sum_i (us_per_prefill_prompt + us_per_prefill_row * rows_i + attn_us(cursor_i, rows_i))

attn_us(cursor, rows) = us_per_million_attn_pairs * rows * (2 * cursor + rows + 1) / 2_000_000
```

Built-in `StepCostModel::H20_QWEN35_4B` uses integer microsecond coefficients
(`base_us = 7_000`, `us_per_prefill_prompt = 4_800`, `us_per_prefill_row = 68`,
`us_per_decode_row = 100`, `us_per_million_attn_pairs = 1_930`), where
`us_per_prefill_prompt` conservatively covers the tile floor of 16-token
prompts (`24 x 16` predicts `149.1 ms` vs `149.3 ms` measured).

## Policy Design (`pegainfer-qwen35/src/scheduler/plan.rs`)

All scheduling policy logic lives in pure, stateless types and functions in
`scheduler/plan.rs`:

- `PrefillQueueState { cursor, remaining_tokens }` carries each admitted
  prompt's resident KV length (`cursor`) alongside its unprefilled tail.
- `StepCostModel`:
  - `fixed_step_us(active_decodes)` and `prompt_cost_us(cursor, rows)` (using
    `u128` arithmetic so high cursors cannot overflow).
  - `predict_step_us(active_decodes, &[(cursor, rows)])` sums the fixed cost
    plus every scheduled prompt's per-prompt, per-row, and attention cost.
  - `min_reachable_stall_us()` returns `fixed_step_us(1) + prompt_cost_us(0, 1)`
    (`11,968 us`, so `--qwen35-prefill-stall-ms` requires at least `12 ms`).
  - `max_rows_for_prompt(headroom_us, cursor, max_rows)` binary-searches the
    largest slice fitting `headroom_us` at `cursor`, with a 1-row liveness
    floor so deep continuations or wide decode batches cannot starve prefill.
- `StepPolicy::{Fixed, StallBounded { max_stall_us, cost_model }}` defaults to
  `Fixed`.
- `plan_prefill_rows(active_decodes, prefilling, remaining_budget, policy)`:
  - Returns `vec![]` when `remaining_budget == 0` or `prefilling.is_empty()`.
  - `Fixed`: unchanged FIFO packing/slicing up to `remaining_budget`.
  - `StallBounded`:
    - **Pure prefill (`active_decodes == 0`)**: no live decoders exist to
      stall, so the front prompt may use up to `remaining_budget`. Followers
      pack only if they fit **whole** inside both `remaining_budget` and
      `max_stall_us` (charging each follower's own per-prompt and `cursor` cost).
    - **Mixed step (`active_decodes > 0`)**: computes `first_cap = max_rows_for_prompt(headroom_us, first.cursor, remaining_budget)`.
      - If `first.remaining_tokens <= first_cap`: pack **whole** FIFO followers
        (`pack_whole_followers`) while total rows stay within `remaining_budget`
        and cumulative step cost stays within `max_stall_us`, never slicing a
        follower into a partial sliver.
      - If `first.remaining_tokens > first_cap`: the head needs more than one
        slice. The next step can merge the remaining head
        (`tail = first.remaining_tokens - first_cap`, starting at
        `first.cursor + first_cap`) with the whole follower only when that pair
        fits both `remaining_budget` and `max_stall_us`; then this step takes
        `first_cap`. A head that needs three or more slices leaves `tail` too
        large to pass, so the planner uses balanced slicing
        `first_rem.div_ceil(first_rem.div_ceil(first_cap))` for the same step
        count and a lower peak stall.

### Counterexample Verification on H20 (`PEGAINFER_ITL_DEBUG=1`)

- **Short multi-prompt burst** (`16`-token prompts, concurrency 32,
  `--qwen35-prefill-stall-ms 50`):
  - Without per-prompt cost: packed `23` prompts (`368` rows) -> **`149.56 ms`** actual GPU step.
  - With `StepCostModel::H20_QWEN35_4B`: capped at **`7` prompts** (`112` rows) -> **`40.16 ms` p50 / `44.12 ms` p95 / `48.67 ms` max**.
- **Long-context continuation** (`65,536`-token prompt, `--max-prefill-tokens 1024 --qwen35-prefill-stall-ms 90`, `decode_n = 1`):
  - Without `cursor` cost: `1024`-row chunks stalled up to **`208.23 ms`** at `cursor = 64,512`.
  - With `cursor` cost: chunks shrank automatically from `1024` (`cursor = 0`) to `398` (`cursor ~ 65k`) -> **`96.57 ms` p50 / `103.92 ms` p95 / `106.83 ms` max** (and on `32,768` tokens at `50 ms`: **`54.24 ms` p50 / `60.79 ms` p95 / `63.29 ms` max**).

### Validation & TP Invariants

- `--qwen35-prefill-stall-ms <MS>` defaults to unset (`StepPolicy::Fixed`).
- `check_launch` (invoked by both CLI validation and engine startup before
  weights load) rejects `0`, rejects targets below
  `cost_model.min_reachable_stall_us()` (`< 12 ms` on the built-in H20 4B fit),
  and rejects combining `--qwen35-prefill-stall-ms` with
  `--decode-overlap=stream`.
- Single-GPU and TP share the same default (`Fixed`, 1024 rows) and the same
  `plan_step` path. TP per-rank prefill scratch stays derived solely from
  `min(max_prefill_tokens, PREFILL_CHUNK_LEN)` in `tp_executor/worker.rs`.

## Same-Host H20 A/B Results

Hardware and build: 1x NVIDIA H20 96 GB (single idle device), driver
580.126.20, CUDA 13.0, Rust `1.99.0-nightly`, Triton 3.8.0, Qwen3.5-4B BF16,
release binary built with
`cargo build --release -p pegainfer-server --no-default-features --features qwen35`.

Workloads (all prompts 1,024 tokens, greedy temperature 0, ignore-eos, seed 42,
`PEGAINFER_ITL_DEBUG` unset, 2 clean runs per cell, 0 failed requests):
- **Single stream (`c=1`)**: 20 prompts, 256 output tokens
- **Steady concurrency 8 (`c=8`)**: 64 prompts, 256 output tokens
- **Steady concurrency 16 (`c=16`)**: 64 prompts, 256 output tokens
- **Poisson 16 req/s (`qps=16`)**: 64 prompts, max concurrency 32, 128 output tokens

Two-run means (higher is better for throughput; lower is better for all latency columns):

| Workload | Policy configuration | Throughput (tok/s) | TPOT (ms) | TTFT p50 / p99 (ms) | ITL p50 / p99 / **p99.9** / max (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| `c=1` (1024/256) | **Default (`1024` fixed)** | 258.3 | 3.57 | 80.0 / 85.8 | 3.57 / 3.6 / **3.8** / 6.2 |
| | `4096` fixed | 258.6 | 3.57 | 80.3 / 86.3 | 3.56 / 3.6 / **3.8** / 6.5 |
| | `50 ms` stall cap (`1024` budget) | 258.2 | 3.57 | 80.3 / 85.8 | 3.57 / 3.6 / **3.8** / 6.5 |
| | `90 ms` stall cap (`4096` budget) | 258.3 | 3.57 | 79.9 / 85.5 | 3.57 / 3.6 / **3.8** / 7.0 |
| | `165 ms` stall cap (`4096` budget) | 256.8 | 3.60 | 79.8 / 85.6 | 3.59 / 3.6 / **3.8** / 10.3 |
| `c=8` (1024/256) | **Default (`1024` fixed)** | 1095.7 | 6.60 | 165.8 / 606.8 | 4.83 / 83.0 / **84.0** / 84.6 |
| | `4096` fixed | 1102.4 | 5.65 | 394.6 / 631.2 | 4.83 / 5.2 / **235.4** / 313.0 |
| | `50 ms` stall cap (`1024` budget) | 1076.5 | 6.80 | 136.3 / 656.2 | 4.83 / **45.6** / **46.5** / 63.5 |
| | `90 ms` stall cap (`4096` budget) | 1096.8 | 6.60 | 165.4 / 606.3 | 4.82 / 82.8 / **83.6** / 84.1 |
| | `165 ms` stall cap (`4096` budget) | 1098.2 | 6.22 | 316.6 / 589.5 | 4.85 / 82.2 / **159.2** / 160.1 |
| `c=16` (1024/256) | **Default (`1024` fixed)** | 1455.4 | 9.83 | 168.1 / 1272.3 | 6.04 / 84.3 / **84.8** / 85.4 |
| | `4096` fixed | 1472.3 | 8.61 | 624.5 / 1255.8 | 6.02 / 85.5 / **313.2** / 314.5 |
| | `50 ms` stall cap (`1024` budget) | 1424.5 | 10.06 | 139.8 / 1385.9 | 6.02 / **46.8** / **47.3** / 47.9 |
| | `90 ms` stall cap (`4096` budget) | 1458.4 | 9.81 | 167.7 / 1272.2 | 6.02 / 84.2 / **84.7** / 86.2 |
| | `165 ms` stall cap (`4096` budget) | 1468.2 | 9.36 | 320.0 / 1228.8 | 6.02 / 160.1 / **160.8** / 161.8 |
| `qps=16` (1024/128) | **Default (`1024` fixed)** | 1158.5 | 20.80 | 170.0 / 1096.8 | 6.83 / 85.5 / **86.3** / 86.7 |
| | `4096` fixed | 1185.5 | 19.04 | 500.8 / 1069.6 | 6.82 / 314.6 / **315.3** / 315.6 |
| | `50 ms` stall cap (`1024` budget) | 1065.9 | 21.05 | 522.1 / 1465.4 | 6.93 / **47.8** / **95.5** / 95.6 |
| | `90 ms` stall cap (`4096` budget) | 1159.3 | 20.79 | 170.0 / 1093.0 | 6.83 / 85.6 / **90.9** / 91.2 |
| | `165 ms` stall cap (`4096` budget) | 1178.5 | 20.03 | 322.2 / 1042.2 | 6.82 / 161.7 / **162.1** / 162.3 |

### Decision: Keep Default `Fixed (1024)`

No single stall target strictly dominates `Default (1024 fixed)` across all
cells:

- **`50 ms` stall cap**: cuts `c=8 / c=16` `ITL p99` by **45%** (`83.0 -> 45.6 ms`, `84.3 -> 46.8 ms`) and `ITL p99.9` by **44–45%** (`84.0 -> 46.5 ms`, `84.8 -> 47.3 ms`), lowers `c=8 / c=16` `TTFT p50`, and cuts `qps=16` `ITL p99` by **44%** (`85.5 -> 47.8 ms`) — at the cost of `-1.7% / -2.1% / -8.0%` throughput and higher `qps=16` TTFT (where 30+ active decoders leave just enough headroom for the first 512-row half at `cursor = 0` while slicing the `cursor = 512` second half further).
- **`90 ms` stall cap with `4096` budget**: eliminates `4096`'s `235–315 ms` `ITL p99.9` freezes (holding `p99.9` at `83.6–90.9 ms`) and matches `1024`'s throughput within `0.2%`, while letting solo long prompts (`active_decodes == 0`) prefill up to `4096` tokens without chunking overhead.
- **`165 ms` stall cap with `4096` budget**: sits cleanly under `165 ms` even at `ITL max` (`160.1 / 161.8 / 162.3 ms`), trading higher `TTFT p50` for `+0.2% to +1.7%` throughput and lower `TTFT p99`.

`StepPolicy::Fixed` (`--max-prefill-tokens 1024`) therefore remains the default
and `--qwen35-prefill-stall-ms <MS>` is an explicit opt-in flag.

## Claim Boundary

- `--qwen35-prefill-stall-ms` is an **empirical cost-model target**, not a
  hard real-time kernel deadline. Non-tile-aligned long-context slices (e.g.,
  `398` rows at `cursor ~ 65k`) can exceed the target by ~7–18% on H20 (`96.6 ms`
  p50 / `106.8 ms` max vs `90 ms` target), and extreme cursors or wide batches
  fall back to the 1-row liveness floor so prefill never starves.
- `StepCostModel::H20_QWEN35_4B` is calibrated on single-GPU H20 with
  Qwen3.5-4B BF16. Other GPU architectures, larger model sizes (9B/27B), or TP
  launches have different slopes and should supply a calibrated `StepCostModel`
  if millisecond accuracy is needed.
- Verification covers `clippy -D warnings`, unit tests (`scheduler::plan` 48,
  `model_line` 15, `pegainfer-qwen35::tests` 9, `pegainfer-frontend` 109),
  single-GPU and TP2 `scheduler::e2e_tests`, `scheduler::chunked_prefill_tests`,
  the live H20 short-prompt and 32k/64k continuation checks, and the 5-arm H20
  serving matrix above.

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
```
