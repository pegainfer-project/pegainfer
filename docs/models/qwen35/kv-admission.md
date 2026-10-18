# Qwen3.5 KV Admission

**Created**: 2026-06-06
**Status**: complete for issue #254

**TL;DR**: Issue #254 is implemented and RTX 5090-validated. Qwen3.5 admission now reserves each admitted request's full KV lifetime budget (`prompt_len + max_tokens - 1`), reserves only future page growth for active requests, keeps temporarily over-budget requests deferred in FCFS order, rejects requests that can never fit this model instance, and reports execution failures as explicit request errors. Real Qwen3.5 e2e passed, an HTTP over-capacity lifetime-reservation pressure run completed `100/100` admissible requests with a healthy post-pressure completion, and a direct in-process impossible request hit the new rejection path.

## Preparation

- **Read**:
  - `docs/index.md` - routed the task to Qwen3.5 roadmap, Qwen3 KV pressure history, and scheduler docs.
  - `docs/models/qwen3/kv-pressure-hang.md` - records the #85 fix shape: full-lifetime KV admission, waiting-queue deferral, impossible-request rejection, error semantics, and real HTTP pressure validation.
  - `docs/models/qwen35/roadmap.md` - lists admission overhaul as the next Qwen3.5 structural item and calls out prompt-only sizing plus missing rejected path.
  - `docs/subsystems/scheduler/scheduler.md` - explains paged KV, scheduler ownership, and why pressure evidence needs a real serving run plus post-pressure completion.
  - `docs/models/qwen35/model-crate.md` - confirms Qwen3.5 scheduler/runtime ownership and test paths.
  - GitHub issue #254 - desired outcome is full-lifetime admission, clean rejection for impossible requests, and no batch-wide abort from KV exhaustion.
  - `pegainfer-qwen35/src/scheduler.rs` - production scheduler currently calls prompt-only admission and reports execution errors as normal finishes in several paths.
  - `pegainfer-qwen35/src/scheduler/plan.rs` - CPU-testable admission seam currently reserves prompt pages only.
  - `pegainfer-qwen3/src/scheduler.rs` - reference implementation for `prompt_len + max_tokens - 1` KV accounting and impossible-request rejection.
  - `pegainfer-core/src/kv_pool.rs` - confirms `KvState::ensure_capacity` grows physical pages lazily and pool capacity includes the reserved padding page.
- **Relevant history**:
  - `docs/models/qwen3/kv-pressure-hang.md` - the original failure mode kept the server alive while completions hung, so validation must include both pressure result and a post-pressure completion.
  - `docs/models/qwen35/roadmap.md` - the current #255 scheduler seam should host policy changes so they remain CPU-testable.
- **Plan**:
  1. Update `scheduler/plan.rs` admission to use full-lifetime KV pages: `prompt_len + max_tokens - 1` for pending requests and remaining future pages for active requests.
  2. Add rejected requests to the admission outcome and emit `TokenEvent::Rejected` from the Qwen3.5 scheduler for requests larger than the usable KV pool.
  3. Convert Qwen3.5 scheduler execution failures from normal `Finished(Stop)` to explicit `TokenEvent::Error`.
  4. Extend CPU tests around admission, active future reservations, page-boundary math, FCFS deferral, and impossible-request rejection.
  5. Run local narrow tests and diff hygiene.
  6. Sync to the authorized remote GPU host and run Qwen3.5 e2e plus issue-shaped HTTP pressure and post-pressure completion.
- **Risks / open questions**:
  - `batch_decode_graph` returns a batch-level error without an offending request id. Full-lifetime admission should prevent KV exhaustion there; if another batch-level error appears, the scheduler can only report errors to all touched active requests.

## Execution Log

### Step 1: Qwen3.5 admission policy
- Updated `pegainfer-qwen35/src/scheduler/plan.rs` so pending requests are sized by full lifetime KV demand: `prompt_len + max_tokens - 1`.
- Added active-request budgeting with `ActiveKvBudget`. Active requests subtract only their remaining future page growth from `available_pages`, because `KvState` already holds their current pages through the shared `PagePool`.
- Preserved Qwen3.5's existing FCFS deferral policy after the first temporary budget miss. Requests larger than `max_request_pages` are rejected and do not block later fitting requests.
- Added a release assertion for the invariant that an active request's current KV pages cannot exceed its admitted lifetime pages.

### Step 2: Scheduler event semantics
- Updated `pegainfer-qwen35/src/scheduler.rs` to build active budgets, pass the usable single-request cap (`capacity_pages - 1`, excluding the CUDA Graph padding page), and emit `TokenEvent::Rejected` for impossible requests.
- Converted Qwen3.5 execution/sampling failure paths from fake `Finished(Stop)` to `TokenEvent::Error`, so request failures surface as errors instead of clean stops.
- Rejection message includes the prompt length and full lifetime request demand:
  - `request requires more KV pages than this model instance can provide: prompt_tokens=..., max_request_tokens=...`

### Step 3: Tests and review hardening
- Added and updated CPU tests for:
  - full generation-budget accounting;
  - active future-page accounting;
  - exact-fit single-request page cap admission;
  - FCFS deferral after a temporary miss;
  - impossible request rejection without blocking a later fitting request;
  - one-token completion at a page boundary;
  - direct `send_rejection` event shape.
- After PR review, removed the fake scheduler-loop seam and fake loop tests. The loop remains concrete; pure admission policy stays in `scheduler/plan.rs`, and runtime shell behavior is covered by e2e plus the direct bench rejection gate.
- Kept the frontend bridge test proving `TokenEvent::Rejected` maps to an error finish:
  - `cargo test --offline --release -p pegainfer-vllm-frontend rejected_request_is_reported_as_error --lib -- --nocapture` passed, `1 passed`.
- Ran read-only DeepSeek diff reviews. Useful findings were handled by adding the active future-page direct test, a release assertion, the padding-page comment, and direct rejection-event coverage. Two findings were rejected after source checks:
  - `TokenEvent::Error` is an existing engine contract and the frontend consumes it.
  - `active.drain(..)` drops `ActiveRequest35`, and its owned `KvState` returns pages by RAII; no KV page leak was found there.

### Step 4: Remote setup and narrow gates
- Remote validation host:
  - GPU: NVIDIA GeForce RTX 5090, driver `580.105.08`, 32607 MiB.
  - CUDA toolkit: `/usr/local/cuda-12.8`, `PEGAINFER_CUDA_SM=120`.
  - Triton AOT Python: validation venv with Triton 3.6 for `sm_120`.
  - Model: `models/Qwen3.5-4B`, HF revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Remote dependency fixes needed before validation:
  - vendored crates.io dependencies for offline cargo;
  - removed AppleDouble `._*` files from the uploaded tar/vendor tree;
  - restored FlashInfer CCCL headers from an existing CUDA 13 Python environment and used Triton 3.6 for `sm_120`.
- Commands passed:
  - `cargo fmt --check`
  - `cargo test --offline --release -p pegainfer-qwen35 --lib scheduler::plan -- --nocapture` - `14 passed`.
  - `cargo test --offline --release -p pegainfer-qwen35 --lib -- --nocapture` - `22 passed` before the fake-loop test deletion.
  - `cargo test --offline --release -p pegainfer-qwen35 send_rejection_reports_kv_lifetime_context --lib -- --nocapture` - `1 passed` before the rejection label rename and test rename.
  - `cargo test --offline --release -p pegainfer-vllm-frontend rejected_request_is_reported_as_error --lib -- --nocapture` - `1 passed`.
  - `cargo build --offline --release -p pegainfer-server` - passed with existing unused-import warnings in `pegainfer-server`.
- Review-fix validation on the H20 host after deleting the fake seam and renaming the rejection label:
  - `cargo fmt --check` passed with nightly Rust.
  - `cargo test --offline --release -p pegainfer-qwen35 send_rejection_reports_kv_lifetime_request_tokens --lib -- --nocapture` passed, `1 passed`.
  - `cargo test --offline --release -p pegainfer-qwen35 --lib -- --nocapture` passed, `20 passed`.
  - H20 real-model e2e was not rerun because the host had no `models/Qwen3.5-4B/config.json`, and downloading `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` failed with `Network is unreachable`.
  - `cargo test --offline --release -p pegainfer-vllm-frontend rejected_request_is_reported_as_error --lib -- --nocapture` was blocked by the local vendored vLLM `proto/vllm_grpc.proto` path, so the prior frontend bridge pass and GitHub CPU check remain the evidence for that surface.

### Step 5: Real Qwen3.5 e2e
- Command:
  - `PEGAINFER_TEST_MODEL_PATH=models/Qwen3.5-4B cargo test --offline --release -p pegainfer-qwen35 --test e2e_scheduler -- --nocapture`
- Result before the extra release-assert hardening:
  - `test_e2e_qwen35_scheduler ... ok`
  - `1 passed`, finished in `12.95s`.
- Result after the release-assert hardening:
  - `test_e2e_qwen35_scheduler ... ok`
  - `1 passed`, finished in `11.04s`.
### Step 6: HTTP pressure validation
- Started the real OpenAI-compatible server:
  - `./target/release/pegainfer --model-path models/Qwen3.5-4B --served-model-name issue254-qwen35 --port 18082`
- Server startup facts:
  - `Qwen3.5 KV cache: 33289 pages (16644 MB), prefill scratch reserve: 3873 MB`
  - scheduler `max_batch=64`
  - frontend `max_model_len=262144`
- Baseline HTTP smoke passed:
  - `/v1/models` returned model id `issue254-qwen35`.
  - `/v1/completions` with prompt `Hello`, `max_tokens=8`, `ignore_eos=true` returned HTTP 200 and `completion_tokens=8`.
- Rejected one invalid pressure shape:
  - 70 requests with `8704` token-id prompts and `max_tokens=8` caused a `Qwen3.5 unified step failed: Alloc failed: CUDA_ERROR_OUT_OF_MEMORY` prefill allocation error.
  - Result was `27` successful completions and `46` HTTP 500s, followed by a healthy post-pressure completion.
  - This is not used as the acceptance gate; it exercises prefill scratch pressure, not KV lifetime admission.
- Final issue-shaped pressure gate:
  - 100 concurrent admissible requests with prompt `Hello`, `max_tokens=10000`, `stop=[","]`, `ignore_eos=true`.
  - Each request reserves about `625` KV pages by lifetime budget, so `100` simultaneous requests exceed the `33288` usable-page pool and require deferral, while actual generation stops after 1 token.
  - Result: `100/100` admissible requests returned HTTP 200, all with `completion_tokens=1`, `finish_reason=stop`, `stop_reason=","`.
  - Latency summary: elapsed `1.265s`, min `0.077s`, p50 `0.588s`, p95 `1.248s`, max `1.251s`.
- Post-pressure checks passed:
  - `/v1/models` still returned `issue254-qwen35`.
  - `/v1/completions` with prompt `Hello`, `max_tokens=8`, `ignore_eos=true` returned HTTP 200 and `completion_tokens=8`.
- Stopped the server with SIGINT and confirmed `nvidia-smi` returned to `0MiB` used.

### Step 7: Direct impossible-request gate
- HTTP could not prove scheduler-level impossible rejection on this host because the frontend/model limits and stop handling keep frontend-visible requests from reaching the raw scheduler cap in a clean way.
- Ran an in-process scheduler path instead:
  - `./target/release/bench_serving --model-path models/Qwen3.5-4B request --prompt-len 1 --output-len 600000 --warmup 0 --iters 1`
- Expected result:
  - command exited non-zero through `bench_serving`'s panic-on-generation-failure path;
  - rejection was explicit and included both `prompt_tokens=1` and the full lifetime request demand.
  - PR review later named that lifetime-demand field `max_request_tokens` to avoid confusing it with the model window.
- GPU was free after the command.

## Debrief

- **Outcome**: Issue #254's core scheduler failure class is fixed for Qwen3.5. Admission now uses full-lifetime KV accounting, temporarily over-budget work waits instead of being admitted into later KV exhaustion, impossible scheduler requests are rejected with a clear event, and execution failures no longer masquerade as clean stops.
- **Pitfalls encountered**:
  - A long-prompt HTTP workload can hit Qwen3.5 prefill scratch allocation before it meaningfully tests KV admission. The valid pressure shape for this issue used high `max_tokens` with early stop to stress lifetime reservation while keeping real generation short.
  - On a 32GB 5090, Qwen3.5's usable KV pool is larger than the frontend `max_model_len`, so ordinary HTTP requests may be frontend-clamped or stop before exposing the raw scheduler impossible-request cap. The in-process bench path is the cleaner rejection gate.
  - The remote environment needed offline vendor, FlashInfer CCCL, and Triton 3.6 fixes before `sm_120` validation was usable.
- **Lessons learned**:
  - For scheduler/KV pressure bugs, the validation workload must isolate KV lifetime reservation from prefill scratch memory. Prompt-heavy pressure can be a different bottleneck.
  - Rejection evidence should include both the pure admission unit policy and a runtime path that actually reaches `TokenEvent::Rejected`; HTTP may be too high-level when frontend constraints are lower than raw KV capacity.
  - Error paths that drain `ActiveRequest35` rely on `KvState` RAII for page return. That is worth remembering when reviewing future Qwen3.5 recovery code.
- **Follow-ups**:
  - Prefill scratch pressure from very large batched prompts is a separate robustness topic; it should not be folded into issue #254 unless the desired policy expands beyond KV admission.
