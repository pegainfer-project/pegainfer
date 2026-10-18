# Qwen3 KV Pressure Hang

**Created**: 2026-05-15
**Status**: complete

**TL;DR**: Qwen3-4B scheduler admission now reserves each admitted request's full KV lifetime budget, keeps temporarily over-budget requests in the waiting queue, rejects only requests that can never fit this model instance, reports those rejects to vLLM as request errors, and releases request state on client-drop/execution-error paths. RTX 5090 issue #85 `vllm bench serve` QPS=2 now completes `500/500` with `0` failures, and post-pressure `/v1/completions` still returns.

## Preparation

- **Read**:
  - `docs/index.md` - routed this issue to Qwen3 batching, scheduler, and benchmark docs.
  - `docs/subsystems/scheduler/scheduler.md` - contains the exact `vllm bench serve` QPS=2 workload from issue #85 and the expected serving evidence shape.
  - `docs/subsystems/scheduler/scheduler.md` - explains Qwen3 scheduler, paged KV, and the page-pool design contract.
  - `docs/conventions/bench-regression.md` - gives benchmark evidence discipline and threshold language.
  - `.codex/harness/README.md` - confirms the verification ladder and safety boundaries.
  - `.codex/harness/commands.md` - provides Qwen3 e2e, server, and benchmark commands.
  - `.codex/harness/verification.md` - classifies this as serving/scheduler behavior needing a narrow repro plus HTTP/benchmark evidence.
  - `pegainfer-qwen3/src/scheduler.rs` - admission control currently defers requests under KV pressure.
  - `pegainfer-qwen3/src/scheduler/plan.rs` - execution plans currently consume pending requests before failures are handled.
  - `pegainfer-qwen3/src/scheduler/effects.rs` - successful finishes drop request state; scheduler execution errors do not.
  - `pegainfer-qwen3/src/executor.rs` - `drop_request` is the existing owner API for releasing per-request KV state.
  - `pegainfer-core/src/kv_pool.rs` and `pegainfer-core/src/page_pool.rs` - KV pages are RAII-returned only when request state is dropped.
  - GitHub issue #85 - observed server stays alive but completions hang after QPS=2 KV pressure.
- **Relevant history**:
  - `docs/subsystems/scheduler/scheduler.md` - QPS=2 varied workload is near capacity and already had some failed requests; the fix must handle pressure explicitly rather than claim higher throughput.
  - `docs/subsystems/scheduler/scheduler.md` - page-pool RAII is the intended cleanup mechanism; scheduler must call the owner drop path when abandoning a request.
- **Plan**:
  1. Add a scheduler-level regression using a fake executor so admission deadlock and execution-error cleanup are testable without GPU/model weights.
  2. Refactor Qwen3 scheduler admission into a small helper that rejects requests that can never fit in the KV pool and keeps temporarily deferred requests.
  3. Preserve touched request IDs for each execution plan; if a prefill/decode/unified step fails, send explicit errors and call `drop_request` for active plus plan-pending requests.
  4. Run `cargo fmt --check`, the targeted Qwen3 scheduler/lib tests, and `git diff --check` locally.
  5. Run a read-only DeepSeek diff review focused on missed cleanup/admission cases.
  6. Use the authorized remote GPU host for Qwen3-4B e2e and the issue #85 `vllm bench serve` workload; verify a post-pressure completion returns.
- **Risks / open questions**:
  - The real hang could include another path beyond leaked KV pages; the pressure test is the裁判.
  - The QPS=2 benchmark is long and may fail some requests by design; the claim boundary is recovery/no permanent hang, not zero failures or performance improvement.

## Execution Log

### Step 1: Scheduler cleanup and admission regression
- Made `start_with_executor`/`scheduler_loop` generic over `ModelExecutor` so scheduler behavior can be tested with a fake executor without GPU/model weights.
- Added fake-executor regression coverage for:
  - requests that can never fit being rejected without blocking later work;
  - temporary KV pressure keeping requests waiting until full KV budget is available;
  - decode errors surfacing as `TokenEvent::Error`, dropping request state, and allowing recovery;
  - client/receiver drop releasing request state.
- Changed `DecodeEffect::EmitAndContinue` send-failure handling to call `drop_request` before retiring the active request.
- Result: remote RTX 5090 `cargo test --release -p pegainfer-qwen3 --lib scheduler -- --nocapture` passed, `4 passed`.

### Step 2: Maintainer feedback refinement
- The maintainer clarified that the basic fix should keep requests that cannot get KV allocation in the waiting queue; preemption can be deferred.
- Updated scheduler admission from prefill-only accounting to full lifetime accounting:
  - active requests reserve the remaining pages they may need until `max_tokens`;
  - pending requests are admitted only if their prompt plus maximum generated-token KV footprint fits after those active reservations;
  - temporarily over-budget pending requests stay in `deferred`;
  - only requests larger than this model instance's total usable KV capacity are rejected to avoid permanent head-of-line deadlock.
- This is intentionally conservative: it may defer earlier than a preemption-capable scheduler would, but it prevents decode-time allocation failure for newly admitted batches without implementing preemption in this PR.

### Step 3: Build and static gates
- Remote environment:
  - GPU: NVIDIA GeForce RTX 5090, driver `580.76.05`, 32607 MiB.
  - CUDA: `nvcc` `13.0.88`, `PEGAINFER_CUDA_SM=120`.
  - Rust: `rustc 1.97.0-nightly (7c3c88f42 2026-05-14)`.
  - Model: `models/Qwen3-4B`, HF revision metadata `1cfa9a7208912126459214e8b04321603b3df60c`.
- Commands:
  - `cargo fmt --check` — passed.
  - `cargo test --release -p pegainfer-qwen3 --lib scheduler -- --nocapture` — passed, `4 passed`.
  - `cargo clippy --release -p pegainfer-qwen3 --lib -- -D warnings` — passed.
  - `cargo build --release -p pegainfer-server` — passed.
- Local command:
  - `~/.cargo/bin/cargo fmt --check` — passed.

### Step 4: E2E and serving pressure validation
- Installed `vllm 0.21.0` in the validation venv to run the issue's real `vllm bench serve` client.
- Ran a host-local exact e2e check against the validation model snapshot:
  - `PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release -p pegainfer-qwen3 --test e2e -- --nocapture`
  - Result after local fixture regeneration for that model snapshot: passed, `1 passed`.
  - PR review later found the regenerated fixture was not portable to the standard local model snapshot, so the repository `test_data/Qwen3-4B.json` change was reverted and this e2e result is not used as a merge gate.
- Ran a small issue-shaped benchmark first:
  - `vllm bench serve ... --num-prompts 20 --request-rate 2 ...`
  - Result: `20/20` successful, `0` failed.
- Ran the full issue #85 workload against the rebuilt server:
  - `vllm bench serve --backend openai --model models/Qwen3-4B --port 8000 --dataset-name random --random-input-len 2048 --random-output-len 128 --random-range-ratio 0.5 --num-prompts 500 --request-rate 2 --seed 42 --ignore-eos --temperature 0 --tokenizer models/Qwen3-4B`
  - Result: `500` successful, `0` failed, duration `250.89s`, peak concurrency `13`, throughput `1.99 req/s`, mean TTFT `129.27ms`, mean TPOT `12.54ms`.
- Post-pressure checks:
  - `/v1/models` returned model id `models/Qwen3-4B`.
  - `timeout 30 curl ... /v1/completions` returned HTTP completion text and usage with `completion_tokens=16`.
- Also ran an overload HTTP probe (`200` concurrent-ish long requests at `80 rps`); it returned explicit HTTP 500s quickly and a post-pressure completion still returned. This was not the acceptance gate, but it confirmed the server did not enter the old half-alive hang state under more aggressive pressure.

### Step 5: Compatibility fix encountered during validation
- Remote CUDA 13.0 initially failed with the existing `cudarc` `cuda-13010` feature because the driver/runtime lacked `cuDevSmResourceSplit`.
- Kept the workspace on `cuda-13010`; changing the shared `cudarc` feature would widen the PR's collaboration surface beyond issue #85.
- Fixed `qwen3_decode_context` test-target compilation by linking `cudaProfilerStart/Stop` directly from `cudart`; the symbols were not exposed through `pegainfer_core::ffi`.

### Step 6: Final diff hygiene
- `git diff --check` — passed.
- Confirmed the remote pegainfer server process was stopped after validation.

### Step 7: Maintainer-style review follow-up
- Re-reviewed the changed scheduler and bridge paths after the main fix.
- Found one API-contract issue: `TokenEvent::Rejected` was being translated to vLLM `EngineCoreFinishReason::Stop`, which would make an impossible KV request look like an empty successful response.
- Changed `pegainfer-server/src/vllm_frontend.rs` so `Rejected` maps to `EngineCoreFinishReason::Error` with the rejection message as `stop_reason`.
- Added `vllm_frontend::tests::rejected_request_is_reported_as_error`.
- Remote RTX 5090 command:
  - `cargo test --release -p pegainfer-server rejected_request_is_reported_as_error --lib` — passed, `1 passed`.

### Step 8: PR review comment follow-up
- Read PR #131 review comments from `gemini-code-assist`. The comments claimed the KV budget formulas should use `prompt_len + max_tokens` and `prompt_len + generated_count`.
- Source check showed that Qwen3 prefill writes only prompt tokens to KV; the sampled first output token is not appended until it is fed as a later decode input. Therefore a request returning `N` completion tokens occupies at most `prompt_len + N - 1` KV tokens.
- Kept the scheduler formula unchanged and added explicit boundary coverage for this contract:
  - helper-level assertions for current/max KV token counts;
  - a scheduler regression proving `prompt_len=page_size, max_tokens=1` fits in one prompt page and finishes without a decode KV page.

### Step 9: Maintainer review portability fixes
- Maintainer review on PR #131 reproduced the issue-shaped HTTP pressure workload successfully on PR head `6b5f963`, then requested two portability fixes before merge.
- Reverted `test_data/Qwen3-4B.json` to avoid carrying a non-portable exact-golden refresh in a scheduler/KV PR.
- Rewrote validation evidence to use checkout-neutral paths such as `models/Qwen3-4B` and "validation venv" instead of machine-local absolute paths.

### Unexpected
- The exact Qwen3-4B e2e initially failed because the checked-in golden text did not match the validation host's current HF revision/runtime output. This matches prior project history around Qwen3 greedy near-tie/golden drift. Maintainer review showed the regenerated fixture was not portable to the standard local model snapshot, so the fixture change was reverted and the scheduler/HTTP gates carry this PR.
- DeepSeek diff-review was attempted twice and timed out (`180s`, then `300s`), so no external advisor result is counted.

## Debrief

- **Outcome**: Issue #85's observed hang is addressed for the measured QPS=2 Qwen3-4B serving workload. Scheduler admission now keeps temporarily over-budget requests waiting instead of admitting them on prefill-only capacity, successful/client-dropped/error paths release request state, impossible requests surface as vLLM request errors, and the real `vllm bench serve` workload completed `500/500` with post-pressure completion still healthy.
- **Pitfalls encountered**:
  - Full lifetime KV accounting is the basic no-preemption fix. Prefill-only accounting can still allow decode-time allocation failure when many active requests grow into new pages together.
  - Exact text e2e depends on the model snapshot/golden pairing; do not refresh `test_data/Qwen3-4B.json` in scheduler PRs unless the regeneration contract is reproducible across the standard validation paths.
  - `vllm 0.21.0` installation pulled a large PyTorch/CUDA 13 stack. The install was slow but completed and enabled the real issue client.
  - CUDA feature selection mattered on the remote 5090: `cuda-13010` expects CUDA 13.1 driver API symbols, so validation hosts using CUDA 13.0 need a CUDA 13.1/compat runtime rather than a source-level downgrade.
- **Lessons learned**:
  - The scheduler should treat KV as a lifetime reservation until preemption exists. That is simpler and safer than relying on decode-time allocation errors.
  - KV budget math must count tokens actually written to KV, not sampled tokens already returned to the client. For Qwen3, the first sampled token does not occupy KV until the next decode step.
  - Requests larger than a single model instance's usable KV capacity need explicit rejection; otherwise they would wait forever and block the queue.
  - Serving regression evidence should include both the pressure client result and a post-pressure completion, because the original failure mode kept `/v1/models` alive while completions hung.
- **Follow-ups**:
  - Design real preemption/cancellation semantics for active requests when the scheduler wants to trade fairness/throughput against full lifetime reservation.
  - Decide whether exact Qwen3 greedy golden drift should get a stronger deterministic tie-breaking gate or remain a regenerated-snapshot fixture.
