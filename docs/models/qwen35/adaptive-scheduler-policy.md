# Qwen3.5 Adaptive Scheduler Policy

> **TL;DR:** Issue #727 now lands Qwen3.5 scheduler policy plumbing with
> conservative defaults: `off` remains the default, `auto` is explicit opt-in,
> `--max-prefill-tokens` remains a hard per-step cap, and TP rejects `auto`
> instead of silently downgrading to `off`. With `--decode-overlap stream`,
> `auto` keeps prefill running through the finishing window (the overlapped
> chunk no longer stalls decode), trading a redundant QPS16 TPOT win for 31%
> TTFT and 14% throughput at an unchanged tail.
>
> **Last touched:** 2026-09

## Preparation

- **Read**:
  - `docs/index.md` - Qwen3.5 roadmap, scheduler, benchmark, and evidence docs are the relevant route.
  - `docs/models/qwen35/roadmap.md` - #469 is the current HTTP boundary; #470/#727 must keep mixed-load evidence separate from serving parity.
  - `docs/subsystems/scheduler/scheduler.md` - the scheduler is single-threaded GPU ownership with chunked prefill and unified prefill+decode.
  - GitHub #727 - asks for adaptive decode-priority policy work with explicit off mode and standard-cell regression protection.
  - GitHub PR #730 review - requires `off` as the default, `--max-prefill-tokens` as a hard cap, and explicit TP rejection for `auto`.
- **Relevant history**:
  - `docs/benchmarks/qwen35-4b-serving-vllm-rtx5090-2026-07.md` - standard HTTP cells and QPS pressure are the current serving regression boundary.
  - Prior stream-overlap exploration stayed out of scope; #727 reuses the existing unified/decode/chunk controls before adding any new execution stream.
- **Plan**:
  1. Add a small Qwen3.5 scheduler policy enum with default `off` and explicit opt-in `auto`.
  2. Move the adaptive decision into pure scheduler-plan helpers with unit tests for fixed, hard-cap, final-chunk, and decode-finish protection cases.
  3. Wire the policy through Qwen3.5 launch, the PegaInfer server CLI, and `bench_serving` so validation can compare opt-in `auto` vs default `off`.
  4. Run narrow Rust checks locally, then use the remote GPU host for Qwen3.5 build/test and representative benchmark cells.
- **Risks / open questions**:
  - The local Mac is not a CUDA validation host, so runtime evidence must come from the provided remote GPU environment.
  - The policy should not claim vLLM parity or production readiness; it is a scheduler-regression and mixed-load tail gate.

## Execution Log

### Step 1: Scheduler policy and pure decision helper

- Added `Qwen35SchedulerPolicy::{Auto, Off}` in `pegainfer-qwen35`, defaulting existing Qwen3.5 launch paths to `Off`.
- Added `choose_prefill_budget(...)` in `scheduler/plan.rs` so the adaptive decision is unit-testable outside the GPU loop.
- Policy rules:
  - `Off` preserves the fixed base prefill budget.
  - No active decode or no in-flight prefill keeps the fixed budget.
  - Active requests with at most 4 tokens remaining get one decode-priority tick before the FIFO-front prefill continues.
  - With `--decode-overlap stream` the finishing-window deferral is disabled: the overlapped chunk already runs off the decode step, so deferring only delays prefill. Measured on A100-40GB, single run (1024/128 QPS16): TTFT `1828 → 1264 ms`, output throughput `873 → 992 tok/s`, ITL p99 unchanged (`41.7 ms`), TPOT back to vLLM parity (`23.8 vs 23.6 ms`); see #727.
  - `Auto` never returns more than the configured base budget; `--max-prefill-tokens` stays a hard per-step cap.
  - Final chunks may shrink below the cap when fewer prompt tokens remain.

### Step 2: Runtime and benchmark wiring

- Threaded the policy through Qwen3.5 launch, `pegainfer` server CLI, and `bench_serving`:
  - `--qwen35-scheduler-policy auto|off` defaults to `off`.
  - Tensor-parallel Qwen3.5 rejects `auto` because TP Phase 1 does not run unified prefill+decode.
  - Qwen3.5 `--max-batch` now accepts `1..=MAX_DECODE_BATCH`; non-bucket requests such as `5` allocate the next graph bucket internally but admit only the requested slots.
- Added mixed-load report visibility for `max_batch` / `max_prefill_tokens` and warnings when `bg_concurrency >= max_batch`; `max_batch=4,bg=4` remains a starvation negative control, while retained mixed evidence used `max_batch=5,bg=4`.

### Step 3: Local checks

Commands run from the issue worktree:

```bash
cargo fmt --all -- --check
git diff --check
git diff --name-only -z | xargs -0 rg -n '<private-patterns>' || true
codex-style-check --no-fail docs/models/qwen35/adaptive-scheduler-policy.md docs/index.md
```

Result: format and diff whitespace passed. The private-data scan found no changed-file hits. `codex-style-check` only reported pre-existing `docs/index.md` Kimi rows, not this task doc.

### Step 4: Remote GPU build and tests

Validation host contract:

| Field | Value |
| --- | --- |
| GPU | 1x NVIDIA GeForce RTX 5090 |
| Driver / CUDA toolkit | NVIDIA driver `595.71.05`, `nvcc 12.8` |
| Rust | `rustc/cargo 1.99.0-nightly` |
| Source | upstream/main `8dd3953` plus this patch |
| Feature | `qwen35` |
| Model | `Qwen/Qwen3.5-4B` downloaded through ModelScope on 2026-07-20 |
| Model config | `model_type=qwen3_5`, `architectures=["Qwen3_5ForConditionalGeneration"]`, `config.json` sha256 `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670` |
| Build env | `PEGAINFER_CUDA_SM=120`, `CUDA_HOME` set to CUDA 12.8, `PEGAINFER_TRITON_PYTHON` set to a Triton 3.7.1 Python |

Remote checks:

```bash
cargo fmt --all -- --check
git diff --check
cargo test -p pegainfer-qwen35 --features qwen35 adaptive_prefill_budget -- --nocapture
cargo test -p pegainfer-server --features qwen35 qwen35 -- --nocapture
cargo build --release -p pegainfer-server --features qwen35
cargo build --release --bin bench_serving --features qwen35
PEGAINFER_TEST_MODEL_PATH=<absolute model path> \
  cargo test --release -p pegainfer-qwen35 --features qwen35 --test e2e_scheduler -- --nocapture
```

Result: all checks passed. `e2e_scheduler` passed the single-GPU test; the TP2 test remained ignored on the one-GPU host.

### Step 5: Pre-review benchmark cells

These cells were gathered before review narrowed the PR to default-off, cap-preserving behavior. They explain why the first draft tried whole-prefill for one low-pressure mixed cell, but they are not current default-policy evidence.

Benchmark flags shared by those pre-review cells:

- Engine: PegaInfer Qwen3.5 direct `bench_serving`, CUDA Graph enabled, feature `qwen35`.
- Source: upstream/main `8dd3953` plus this patch.
- Hardware/toolchain/model: same as the remote GPU contract above.
- Sampling: synthetic random prompts, greedy, fixed output.
- Standard cells: `1024/256`, warmup 1, iters 3, `--max-batch 16`.
- Mixed cells: `--max-batch 5`, `bg_concurrency=4`, background `512/2048`, injection `4096/1`, `qps=0.5`, 5 cold injections, warmup 1, `--skip-baseline`, with background and injection generated-token lengths/hashes retained in the JSON.
- Negative control: `--max-batch 4`, `bg_concurrency=4`, kept to prove the warning path; not used as improvement evidence.

Standard request A/B:

| Policy | Cell | TTFT p50 ms | steady TPOT p50/p99 ms | request tok/s | output length | hash0 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `auto` | 1024/256 c1 | 49.816 | 6.942 / 7.022 | 140.68 | 256-256 | `0827a7035c7b7a89` |
| `off` | 1024/256 c1 | 50.176 | 6.949 / 7.150 | 140.37 | 256-256 | `0827a7035c7b7a89` |
| `auto` | 1024/256 c16 | 508.187 | 9.763 / 58.321 | 76.73 | 256-256 | `0827a7035c7b7a89` |
| `off` | 1024/256 c16 | 508.343 | 9.770 / 58.255 | 76.76 | 256-256 | `0827a7035c7b7a89` |

Mixed-load A/B:

| Policy | all ITL p50/p99/max ms | steady p99/max ms | stall p50/p99/max ms | stall gaps | warnings |
| --- | ---: | ---: | ---: | ---: | --- |
| `auto` | 7.386 / 7.845 / 196.554 | 7.754 / 58.553 | 7.941 / 196.552 / 196.554 | 60 / 4944 | none |
| `off` | 7.389 / 57.211 / 65.232 | 7.794 / 58.536 | 57.141 / 65.229 / 65.232 | 120 / 4904 | none |

Mixed output sanity:

| Policy | background output length | background hash0 | injection output length | injection hash0 |
| --- | --- | --- | --- | --- |
| `auto` | 1236-1238 | `dea24e27083abe47` | 1-1 | `ec2064181e172bb6` |
| `off` | 1226-1228 | `40933b449d599567` | 1-1 | `ec2064181e172bb6` |

Interpretation: this pre-review whole-prefill variant improved p99 for one low-pressure 4k/1-token mixed cell but raised max ITL. Review correctly treated that as a tradeoff, so the landed policy no longer exceeds the configured prefill cap and does not use these cells to justify a default flip.

The starvation negative control (`max_batch=4,bg=4`) emitted the expected warning, plus QPS/background-length warnings caused by the intentionally saturated setup. It remains a measurement guard only.

## Debrief

- **Outcome**: #727's scheduler policy plumbing, explicit opt-in `auto`, default `off`, server/bench CLI wiring, TP `auto` rejection, non-bucket Qwen3.5 `max_batch`, and cap-preserving budget tests are implemented.
- **Pitfalls encountered**:
  - Triton 3.3.0 could not AOT for `cc120`; the validation host used Triton 3.7.1.
  - Hugging Face download was unavailable from the host; ModelScope provided the same public `Qwen/Qwen3.5-4B` model family, with config hash recorded above.
  - A relative `PEGAINFER_TEST_MODEL_PATH` failed for `e2e_scheduler` because the test process cwd differed; the rerun used an absolute path and passed.
  - `bench_serving mixed` still infers stall windows from request `[submit,last-token]`; the retained `max_batch=5,bg=4` capacity gate avoids the known starvation artifact, but this doc does not claim internal `decode_n` trace instrumentation.
- **Lessons learned**:
  - The adaptive path should stay opt-in until wider active-decode/QPS evidence chooses an SLA objective.
  - A `max_batch=4,bg=4` mixed cell is a negative control, not evidence of overlap.
  - Qwen3.5 TP should reject `auto` until TP supports unified mixed steps.
- **Follow-ups**:
  - A future mixed-load trace can add explicit per-step `prefill_tok` and `decode_n` fields; that would make #470-style overlap validity visible without relying on the capacity gate.
  - Wider QPS pressure and active-decode-width cells from #727 acceptance remain the next benchmark expansion before any default-readiness wording.
