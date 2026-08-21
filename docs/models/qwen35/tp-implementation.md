# Qwen3.5 TP Implementation Record

> **TL;DR:** Qwen3.5 TP Phase 1 and P2A are complete: TP2 now supports start-gated eager unified prefill+decode with strict ID-aligned artifacts, fail-closed lifecycle recovery, and pre-load CUDA ordinal validation; P2B GDR state sharding is next.
>
> **Last touched:** 2026-08

## Scope

This is the implementation record for Qwen3.5 tensor parallelism. The stable architecture contract lives in `docs/models/qwen35/tp-design.md`; this file records what actually landed, what was verified, and what should carry into later phases.

Keep Phase 1 and Phase 2 in this file until the Phase 2 implementation becomes large enough to split. The same state ownership risks continue across phases, so keeping the history together is useful: Phase 1 proves dense TP and worker-owned request state, while Phase 2 builds on that boundary for mixed prefill/decode and sharded recurrent state.

Out of scope for this file:

- local machine paths, NCCL symlink details, and temporary environment setup
- raw command transcripts unless they are part of retained evidence
- benchmark/performance claims
- prompt echo support; Qwen3.5 accepts completion-only requests

## Phase 1 Outcome

Phase 1 is complete as a correctness/runtime milestone.

Implemented:

- TP config validation for rank/world size, dense divisibility, and `TP > 1 && CUDA Graph` fail-closed startup.
- Dense TP weight loading for full-attention projections, full-attention KV heads, and MLP projections.
- Rank-local worker executor with worker-owned model shards, KV state, recurrent/conv state, CUDA context, cuBLAS, and NCCL comms.
- Eager TP prefill, chunked prefill, and eager decode.
- Scheduler TP backend that routes chunked prefill and eager decode through TP workers while keeping logical request/page accounting in the scheduler.
- Public multi-device Qwen3.5 engine path and server launch path for `tp_size > 1` with CUDA Graph disabled.
- Real HTTP TP2 serving smoke through the vLLM/OpenAI-compatible frontend.

Not implemented in Phase 1:

- TP CUDA Graph capture/replay.
- TP `RunUnifiedStep` mixed prefill+decode execution.
- Sharded linear-attention/GDR weights, kernels, conv state, or recurrent state.
- Vocab-parallel embedding or `lm_head`.
- Prefix-cache or recurrent-state snapshot support.
- Performance claims.

## Important Fixes

### Gated q projection layout

The major numeric blocker was the full-attention gated `q_proj` TP shard layout.

The wrong assumption was that `q_proj.weight` rows were physically arranged as:

```text
[all q rows][all gate rows]
```

The actual Qwen3.5 kernel contract is per-head interleaved:

```text
[head0 q][head0 gate][head1 q][head1 gate]...
```

For TP2, the fixed loader preserves contiguous head-interleaved ranges:

- rank 0 loads rows `0..4096`
- rank 1 loads rows `4096..8192`

The old loader gathered local q rows and local gate rows separately, then rebuilt a `[q][gate]` fused matrix. That corrupted the first full-attention contribution and failed the TP2 HF gate from prefill position `0`.

### Per-device Triton AOT handles

Real TP2 prefill exposed that Qwen3.5 GDR Triton AOT C stubs could not cache `CUmodule` / `CUfunction` in process-global state. With two CUDA devices, the rank that loaded a GDR kernel first could leave the other rank with an invalid function handle.

The generated stubs now cache module/function handles per CUDA device ordinal. This is an implementation constraint worth remembering for future multi-GPU users of generated Triton C stubs.

Follow-up review tightened this path: the generated stubs now fail closed before indexing the fixed per-device handle tables if `cuCtxGetDevice` returns an ordinal outside the table size. This preserves the Phase 1 static-table implementation while avoiding out-of-bounds writes on high CUDA ordinals.

### Worker-local NCCL setup

NCCL comms are initialized inside rank worker threads after each worker binds its CUDA context and initializes thread-local cuBLAS. Creating comms on the controller thread and moving them into workers led to invalid-handle symptoms and hangs.

This matches the design contract: TP workers own rank-local CUDA/NCCL execution resources.

### Current-main API compatibility

Rebasing Phase 1 onto current `main` required preserving the TP execution boundary while adopting newer shared contracts:

- Hybrid batch decode now builds `Vec<&mut RecurrentState>` from graph-owned slots before entering the common linear-attention helper. This keeps request state in place while satisfying the helper's mutable-reference slice contract.
- `pegainfer_sample::select_batch` now requires request-local sampling steps. Phase 1 TP still samples one row at a time and has no request-local sampling counter, so it passes step `0` and retains its existing per-row `sample_seed` offset. Do not substitute batch row indices for request-local steps: that would make seeded output depend on batch composition.
- Qwen3.5 launch and tests use the current `EngineLoadOptions` surface; the removed `enable_prefill_profile` field is no longer supplied.
- TP scheduler tests explicitly set the newer `GenerateRequest::data_parallel_rank` field to `None` because Phase 1 is TP-only, not DP.
- Synthetic TP config/loader fixtures include `tie_word_embeddings`, matching the current `Config35` contract without changing production config loading.
- TP2 short/long HF gates use `Golden::load_for(model_path, long)` and pass the complete `Golden` to metadata validation, matching the model-selected fixture flow used by TP1.

These are compatibility changes, not extensions of Phase 1 scope. In particular, full seeded-sampling replay under TP should add a request-local completion counter rather than overloading batch position.

## Validation Evidence

Phase 1 acceptance coverage:

- TP2 short HF logits gate passes:
  - sequential eager: `108` positions, mean `0.0258`, p99 `0.0801`, max `0.1298`
  - batched eager: `72` positions, mean `0.0257`, p99 `0.0809`, max `0.1298`
- TP2 long HF logits gate passes:
  - prompts `4097` and `8192`, sequential eager: `18` positions, mean `0.0232`, p99 `0.0792`, max `0.1035`
- TP2 scheduler e2e passes and covers:
  - context-window rejection
  - greedy/logprobs paths
  - sequential requests
  - repeated request reuse
  - concurrent mixed greedy/sampling requests
  - consumer drop
  - post-drop scheduler health
- TP2 HTTP serving smoke passes through `pegainfer_vllm_frontend::serve`:
  - `/v1/models`
  - non-streaming `/v1/completions`
  - streaming `/v1/completions`
  - concurrent completions
  - finite logprobs
  - chunked prefill forced with `max_prefill_tokens=1`
  - `TP2 + CUDA Graph` fail-closed startup
- TP1 regression gates pass after the TP2 additions:
  - TP1 short/long HF logits gates
  - TP1 scheduler e2e
- Current-main rebase verification passes:
  - formatting check
  - Qwen3.5 release compilation for all test targets
  - `pegainfer-server` release compilation with only the `qwen35` model feature

Known validation constraints:

- TP2 tests remain ignored by default because they require two CUDA devices, NCCL, and real Qwen3.5 weights.
- Long TP2 HF replay is GPU-memory-sensitive; choose a sufficiently free device pair.
- Qwen3.5 HF golden integration tests should run serially on memory-constrained hosts to avoid unrelated KV-capacity failures from concurrent model loads.

Stable test knobs:

- `PEGAINFER_TEST_MODEL_PATH`: real Qwen3.5 weights path for HF, scheduler, and serving tests.
- `PEGAINFER_TEST_TP_DEVICES`: comma-separated TP2 CUDA ordinals. Defaults to `0,1`; examples: `1,2`, `2,3`. TP2 tests require exactly two distinct ordinals.
- `PEGAINFER_TEST_FRONTEND_MODEL_PATH`: optional tokenizer/config metadata path for HTTP serving tests. Defaults to `PEGAINFER_TEST_MODEL_PATH` when unset.

## Phase 2 Progress

Phase 2 is locked in `docs/models/qwen35/tp-design.md` as two separate implementation series: P2a is eager mixed unified execution on the replicated Phase 1 GDR path; P2b shards the head-indexed linear-attention/GDR weight and state surface. P2a protocol/lifecycle gates are complete, so P2b can now change loader, kernel, and state shapes while preserving those contracts.

### P2a: TP mixed-step unified execution

P2A implements eager `RunUnifiedStep` under TP while retaining Phase 1's replicated linear-attention/GDR weights, kernels, conv state, recurrent state, and scratch shapes.

#### Step 1: lifecycle cleanup gates

Completed as test and fault-observation infrastructure before changing production lifecycle semantics.

Why it exists:

- A successful controller-side `DropRequest` return did not directly prove that every rank removed the same `RequestId` or released its KV/recurrent/conv ownership.
- Later cancellation, partial-dispatch, drop-acknowledgement, and unified-step work needs direct evidence of rank-local state before and after each transition.
- Cleanup must prove both capacity recovery and fresh numeric state; scheduler bookkeeping alone cannot detect a stale rank-local recurrent or KV allocation.

Implemented in `tp_executor.rs` under `#[cfg(test)]` only:

- `WorkerStateSnapshot { rank, request_count, requests }`, where each request entry carries its `RequestId` and `Prefilling`/`Decoding` phase.
- A healthy snapshot API that requires an unpoisoned executor and an unchecked test-only API that bypasses only the controller poison guard. The latter is reserved for later synthetic failure tests while every worker channel remains connected.
- Exact-rank collection that rejects duplicate or out-of-range ranks, payload/response rank disagreement, missing responses, wrong reply variants, and inconsistent request counts. Valid snapshots are returned in rank order.
- An ignored TP2 capacity gate that fills configured `max_batch=2`, observes every ID in `Decoding` on both ranks, drops every ID, requires zero requests on every rank, refills the complete capacity, executes decode, and proves the second cleanup is also empty.
- An ignored TP2 numeric gate that records a prompt's deterministic first token and five requested logprobs, drops the request, verifies every rank is empty, re-admits the same prompt under a new `RequestId`, and requires an exact artifact match.

Verification:

- Qwen3.5 release all-target check and clippy with `-D warnings` pass.
- The regular library suite passes: `71 passed`, `0 failed`, `8 ignored`.
- Both new TP2 GPU gates pass independently.
- Formatting and `git diff --check` pass.

This step deliberately adds no production command, scheduler state transition, drop semantics, or serving claim. It supplies the observability needed to prove the later P2A changes rather than trusting controller-visible acknowledgements alone.

#### Step 2: cancellation pruning before admission

Completed as the first production scheduler lifecycle change in P2A.

The shared TP1/TP scheduler tick now follows the fixed order `drain -> prune -> publish load -> admission -> plan`. It merges deferred and newly submitted work, removes requests whose `TokenSink` is already closed, publishes the resulting state, and only then computes slot/KV budgets and admits work. The idle wakeup path repeats drain/prune/publication after `blocking_recv()` because the waking request can close before admission and other submissions can race with the receive.

Change to the existing scheduler skeleton:

- Previously, the loop published the state left by the prior tick before draining `submit_rx`; `num_waiting_reqs` therefore represented only `deferred.len()`. It then drained submissions and entered admission.
- The loop now first takes `deferred`, drains every currently available submission into the same `pending` vector, and calls one backend-neutral `prune_closed_requests` helper across pending, active, and prefilling ownership.
- `publish_load` remains the same internal watch publication surface and `LoadSnapshot` retains the same public fields and types. A small `logical_load_counts` helper makes the post-prune running/waiting calculation directly testable; waiting is now `pending.len()` at the admission boundary.
- The empty-loop blocking skeleton is retained, but wakeup now has a second drain/prune/publish boundary before admission. If every received request is already closed, the loop returns to blocking without entering admission or planning.
- The admission and plan implementations are otherwise unchanged. This commit changes which settled state they consume, not their policies or the public `EngineHandle::load_watch()` API.

Effect on the pre-existing TP1 (`SchedulerBackend::Single`) path:

- A closed pending request is removed before `alloc_prefill_state`, so TP1 no longer allocates its KV/recurrent state and then discovers the disconnected consumer during token delivery.
- A closed prefilling request is removed before plan construction. Consuming its existing `PrefillBackendState::Single` drops the owned `KvState` and `RecurrentState` through their existing RAII path; no TP1 release routine or allocator rule was added.
- A closed active request is retired before the next model step through the existing `compact_single_slot` path. The same `swap_remove` and graph-slot copy used by EOS, length, and token-send failure are reused; only the retirement timing moves earlier, so the cancelled row no longer executes and samples one unnecessary decode step.
- TP1 page availability, slot budgets, and plan construction now observe that cleanup in the same tick. Consequently, the executed batch width and load samples can differ under cancellation, but KV accounting formulas, slot-compaction mechanics, admission policy, plan builders, kernels, and sampling implementation are unchanged.

Cleanup deliberately reuses existing backend ownership paths:

- pending cancellation uses stable `retain`, preserving FIFO order among live requests;
- active cancellation uses normal request retirement, including TP drop and single-GPU slot compaction;
- prefilling cancellation removes the entry and drops its backend state through the existing adapter.

This ordering makes cancellation visible to admission in the same tick. A closed resident is absent from the post-prune running/KV state, a live replacement is present in waiting, and the capacity released by the resident can admit that replacement immediately.

Verification:

- Three focused CPU tests pass for pending FIFO pruning, post-prune logical load, and same-tick capacity reuse.
- The ignored real TP1 `max_batch=1` cancellation/replacement gate passes on an RTX 3090. It observes the post-prune `running=0, waiting=1` boundary, completes the replacement, and returns running, waiting, and KV usage to zero.
- The existing real TP1 scheduler integration E2E passes (`1 passed`, 29.06 seconds).
- Release all-target clippy with `-D warnings` passes.
- The regular library suite passes: `74 passed`, `0 failed`, `9 ignored`.

This step does not strengthen the TP drop reply protocol or add scheduler-wide fail-closed propagation. It uses the Phase 1 cleanup calls as they exist; exact-rank `DropExpectation` acknowledgement belongs to step 3, and mandatory replica-fatal propagation belongs to step 4.

#### Step 3: TP worker protocol hardening

Completed as the distributed command/reply foundation for later scheduler fail-closed handling and unified execution.

State-mutating prefill, decode, drop, and future unified envelopes now share one `Pending -> Execute | Cancel` start gate. The controller enqueues an envelope for every rank before resolving `Execute`; if any enqueue fails, it resolves `Cancel` for the delivered prefix and poisons the executor. Workers wait on the gate before request-state mutation, kernel launch, or NCCL entry. Ping and test-only snapshots remain ungated because they do not mutate request state or enter collectives.

Response handling now separates timed transport collection from pure response-set validation. Ping, prefill, decode, and drop all require exactly one in-range response from every rank:

- ping requires `Ack` from every rank;
- prefill requires one `Prefill` result from rank 0 and `Ack` from every non-primary rank;
- decode requires one `Decode` result from rank 0 and `Ack` from every non-primary rank;
- drop requires `DropAck { existed }` from every rank and validates the complete existence vector against the scheduler's `DropExpectation`.

Duplicate, missing, out-of-range, wrong-variant, non-primary typed, or missing-primary responses poison the executor after dispatch. Drop accepts only exact-rank all-false for `MustBeAbsent` and exact-rank all-true for `MustExist`; mixed values and uniformly unexpected values are lifecycle divergence even though every rank is absent after the command.

The low-level `Qwen35TpExecutor::drop_request` API now requires `DropExpectation` and returns `Result<()>`; `DropExpectation` is re-exported through the model-local `runtime` module for tests and debugging. The server-facing `EngineHandle` API is unchanged. Scheduler callers derive expectations from owned lifecycle state:

- active and successfully dispatched/final-prefill state use `MustExist`;
- prefilling cancellation at `cursor == 0` uses `MustBeAbsent`;
- prefilling cancellation after progress uses `MustExist`;
- execution failure performs no healthy-path drop because the executor is already poisoned.

This commit intentionally retains the temporary scheduler adapter that logs a returned healthy-drop failure. Executor poison prevents subsequent normal commands, but step 4 still owns mandatory propagation into one scheduler-wide terminal path, fail-closed completion publication, and exactly-once unresolved-request error fan-out.

Verification:

- Release all-target check and clippy with `-D warnings` pass.
- Pure protocol/gate tests cover exact-rank reply matrices, lifecycle-expectation mismatch, controller structural rejection with zero dispatch, all-enqueued `Execute`, prefix-only `Cancel`, and poison preservation.
- The complete regular library suite passes on SM86: `82 passed`, `0 failed`, `12 ignored`.
- Three new real TP2 gates pass: prefix-only dispatch failure leaves every rank empty (214.38 seconds under unrelated four-GPU training contention), lifecycle expectations plus mixed-rank divergence pass (8.05 seconds), and a real receiver disconnect poisons without claiming an unavailable all-rank snapshot (6.96 seconds).
- Existing healthy TP2 prefill/decode/drop and scheduler chunked-prefill/decode smoke tests pass (6.45 and 6.54 seconds).

Implementation pitfall: the first release-only gate test hung because `start.execute()` was placed inside `debug_assert!`; release builds remove the complete assertion expression, including side effects. The state transition now executes unconditionally and only its boolean result is debug-asserted. Protocol state transitions must never live inside debug-only assertions.

#### Step 4: fail-closed TP scheduler lifecycle

Completed as the scheduler recovery and user-visible completion boundary required before TP unified execution.

The temporary step-3 logging adapter is gone. TP active retirement and prefill-state cleanup now return `Result`, and cancellation pruning, standalone prefill/decode execution, token dispatch, and final-prefill promotion propagate any returned TP lifecycle failure to the scheduler loop. A returned TP prefill/decode or post-execution artifact-alignment error follows the same fatal path. The scheduler does not retry per-request cleanup after poison.

Successful TP completion is now prepared and committed explicitly:

- decode EOS buffers only `Finished(Stop)`;
- decode length buffers the final `Token` followed by `Finished(Length)`;
- immediate final-prefill EOS/length uses the same buffering rule, so its first token is not exposed before cleanup;
- the candidate keeps logical request ownership while `DropRequest(MustExist)` runs, and publishes the buffered events only after exact-rank all-true `DropAck` succeeds;
- drop failure discards the buffered success events and transfers the unresolved candidate to terminal error fan-out.

Fatal errors carry any tick-local request ownership back to the scheduler loop instead of consuming it in a step helper. The single terminal helper closes `submit_rx`, drains every request whose send completed before close, consumes transient candidates/scheduled work plus active, prefilling, current pending, and deferred owners exactly once, attempts one `TokenEvent::Error` per request, publishes a zero-running/zero-waiting/zero-KV load snapshot, and exits. It does not deduplicate by TP or external request ID; exclusivity comes from moving each request out of its prior owner.

The existing single-GPU boundary remains explicit. TP completion requires cleanup acknowledgement before publication, while the non-TP decode path retains its original `Token`/`Finished` publication before slot retirement. Single-GPU prefill state still drops through its existing RAII ownership, and `EngineHandle` plus `TokenEvent` public types are unchanged.

Verification:

- Focused scheduler lifecycle coverage passes: `16 passed`, `0 failed`, `2 ignored`. It covers TP EOS/length and immediate-prefill event buffering, active/prefill drop failure with no leaked success event, remaining scheduled ownership preservation, prune failure propagation, non-TP publication order, and barrier-controlled submit close/drain fan-out across every owner class.
- Release all-target check and clippy with `-D warnings` pass; formatting and `git diff --check` pass.
- The complete regular library suite passes on SM86 outside the GPU-isolated sandbox: `90 passed`, `0 failed`, `12 ignored`.
- The real-weight TP2 scheduler chunked-prefill/decode smoke and the complete TP2 scheduler E2E pass on physical GPUs 1/2. The complete E2E covers context rejection, greedy/logprobs, sequential/repeated/concurrent requests, consumer drop, and post-drop health.
- A same-session TP1 scheduler E2E attempt could not pass model startup because an unrelated training process left insufficient memory for the loader's default 64-slot graph allocation. No training process was stopped. The non-TP ordering CPU gate and complete regular suite are green; the previously established real TP1 E2E evidence remains unchanged.

Step 4 does not add unified plans, change worker command payloads, alter TP1 sampling, or modify GDR weights/state/kernel shapes. Step 5 owns `RunUnifiedStep`, strict ID-aligned unified artifacts, ordinal validation, and the final TP1/TP2 regression ladder.

#### Step 5: TP unified execution

Completed as the final P2A execution-protocol milestone. When active decode and scheduled prefill coexist, TP now uses the same `plan::build_next_plan` decision as the single-GPU scheduler and emits one start-gated `RunUnifiedStep` to every rank. The canonical plan carries ordered prefill/decode items plus separate seeds. The scheduler selects the decode seed first and the prefill seed second to preserve its prior RNG order; workers execute prefill first and decode second on every rank. Rank 0 returns `TpUnifiedResult`, while every non-primary rank must return `Ack`.

Internal protocol/API changes:

- Added crate-private `TpUnifiedPlan`, `TpUnifiedResult`, the complete `RunUnifiedStep` payload, and `TpWorkerReply::Unified`. These are not re-exported through `runtime`; the public server `EngineHandle`, `TokenEvent`, and request contract are unchanged.
- Refactored worker prefill/decode into typed inner row operations so standalone and unified commands share the same eager kernels and sampling behavior. Existing public low-level `execute_prefill`/`execute_decode` results retain their shapes.
- Added controller structural validation before enqueue: both halves must be non-empty, row counts must fit scheduler capacity, prefill/decode IDs must be internally unique and mutually disjoint, and prefill chunks must be non-empty. Worker-local request existence, phase, and actual capacity are revalidated after gate release and remain replica-fatal rather than adding a two-phase validation protocol.
- Extended exact-rank reply validation so unified accepts exactly one rank-0 `Unified` result plus non-primary acknowledgements. Returned worker or reply-set failure continues through the step-4 poison and terminal scheduler path.

Change to the existing scheduler skeleton:

- Removed the TP-only `build_eager_only_plan` branch. Both backends now use the normal planner, so active decode plus scheduled prefill becomes `ExecutionPlan::Unified` instead of two serialized scheduler ticks.
- Added a TP-only execution adapter that builds the canonical plan, aligns returned artifacts by `RequestId`, processes decode results first, and only then promotes or requeues prefill. Decode completion can therefore release capacity through `DropRequest(MustExist)` before final-prefill promotion.
- Replaced TP prefill token-`0` placeholders and positional decode matching with explicit artifacts. Prefill alignment is a chunk-length `Vec<Option<PrefillArtifact>>`: outer `None` means non-final, while `Some { logprob: None, .. }` is a valid final artifact. Decode requires one artifact for every active ID. Shuffled valid results are accepted; unknown, duplicate, non-final, or missing IDs poison the replica after execution.
- The shared `promote_or_requeue` skeleton now accepts a backend-specific artifact wrapper. The single-GPU branch still consumes its original dense sampled tokens/logprobs, uses the same logits path and RNG calls, and retains its step-4 publish/retire semantics.

Startup now validates TP CUDA ordinals against the generated Triton AOT handle-table contract before model/filesystem/CUDA access. Ordinals must be distinct and below the named table length `16`; the generated wrapper's runtime bounds check remains defense in depth.

Verification:

- Release all-target check and clippy with `-D warnings`, formatting, and `git diff --check` pass.
- Complete non-ignored library suite passes on SM86: `96 passed`, `0 failed`, `14 ignored`.
- Pure tests cover ordinal bounds/duplicates, unified structural rejection plus subsequent healthy dispatch, worker-local missing/phase/capacity failures, exact-rank unified replies, shuffled artifact alignment, unknown/duplicate/missing IDs, and final-without-logprobs versus non-final absence.
- The real-weight TP2 executor mixed-step gate passes (`1 passed`, 7.27 seconds), confirms both ranks retain two decoding requests after the combined operation, and returns both ranks to empty after lifecycle cleanup.
- The deterministic real-weight TP2 scheduler case with `max_batch=2` and `max_prefill_tokens=1` passes (`1 passed`, 7.15 seconds).
- All `13` TP2/lifecycle ignored library gates pass together. The fourteenth ignored test is TP1-only and could not load under the unrelated training process: the loader reported `8101 MB` free versus `3248 MB` prefill scratch plus `6288 MB` recurrent state and minimal KV needs. No training process was stopped.
- Complete TP2 scheduler E2E passes (`1 passed`, 35.89 seconds).
- TP2 short/long HF golden gates pass (`2 passed`): short sequential mean `0.0260`, p99 `0.1081`; short batched mean `0.0267`, p99 `0.1167`; long sequential mean `0.0228`, p99 `0.0689`.
- TP2 OpenAI-compatible HTTP serving smoke passes (`1 passed`, 10.41 seconds), including streaming, non-streaming, concurrent completions, logprobs, and TP+CUDA Graph rejection.

P2A does not shard GDR weights/state/scratch, add a post-GDR all-reduce, enable TP CUDA Graph, or claim a performance improvement. TP1's earlier real-weight evidence remains applicable, but a same-session full TP1 model E2E rerun is still resource-blocked by external GPU occupancy.

#### Merge-gate test surface

The pre-merge review reduced P2A coverage to distributed behavior that cannot be
established by restating local helpers. Removed tests enumerated start-gate
booleans, planner choices, CUDA-ordinal bounds, unified-plan shapes, reply
variants, artifact-alignment variants, snapshot-collector variants, and basic
TP2 startup/prefill/chunk/decode smoke already exercised by higher-level gates.

The retained P2A library surface is seven real TP2 gates:

- partial dispatch cancels the delivered prefix before rank-local mutation;
- rank-local lifecycle divergence is detected and poisons the replica;
- a disconnected worker receiver fails closed;
- executor and scheduler mixed prefill/decode both complete;
- drop-all restores complete request capacity;
- re-admission under a fresh `RequestId` reproduces the clean first-token artifact.

Completion cleanup and terminal error fan-out remain covered by the focused
scheduler lifecycle tests. The production-facing regression ladder remains the
TP2 scheduler E2E, short/long HF logits gates, and OpenAI-compatible HTTP
serving lifecycle test.

On the local 2x RTX 3090 SM86 fixture, the complete explicit ignored run passed
`11/11`: seven library gates, one scheduler E2E, two HF gates, and one HTTP gate.
The regular release library suite passed `94` tests with those seven real TP2
tests ignored by default. The HF results remained within the established
tolerances: short sequential/batched mean deltas `0.0260`/`0.0267`, and long
sequential mean delta `0.0228`.

#### Qwen3.5 feature CI gate

The CUDA workflow now has separate matrix results for Qwen3.5 compile and
Clippy. Both install the repository's CUDA 13.0.2 toolchain, Python 3.10, and
pinned Triton 3.7.1, then build `pegainfer-qwen35` with the actual `qwen35`
feature enabled for all targets on `sm_80`:

```text
cargo check --release --locked -p pegainfer-qwen35 --features qwen35 --all-targets
cargo clippy --release --locked -p pegainfer-qwen35 --features qwen35 --all-targets -- -D warnings
```

This is a compile/lint merge gate, not a substitute for the ignored real-GPU
TP2 tests above. The same target surface passed locally on SM86 with nightly
2026-07-10 and Triton 3.4.0; both commands compiled the Qwen3.5 Triton AOT path.
The GitHub jobs provide the independent Ubuntu, SM80, and pinned Triton 3.7.1
verification.

#### Real TP2 multi-turn serving gate

The Phase 2A production-path gate starts the real OpenAI-compatible Qwen3.5
server on two RTX 3090 GPUs with TP=2, eager execution, `max_batch=2`, and
`max_prefill_tokens=64`. A pinned upstream Rust `vllm-bench` client runs
dependent `openai-chat` conversations at concurrency 4, so the workload exceeds
resident capacity and forces cleanup followed by fresh admission.

The primary workload completed 12/12 conversations and 44/44 measured turns
with zero failures. Conversation lengths varied from 2-5 turns, later prompts
carried accumulated history, and every turn returned its configured 24 tokens.
Without restarting the server, a second 4-conversation, 8-turn probe also
completed with zero failures. Graceful shutdown released both TP devices.

The exact client/server revisions, build and run commands, request counts,
concurrency, prompt/output lengths, per-turn history evidence, pass criteria,
limitations, and raw JSON results are published in
[`docs/benchmarks/qwen35-tp2-phase2a-multiturn.md`](../../benchmarks/qwen35-tp2-phase2a-multiturn.md).

#### Serving scope: prompt echo is unsupported

Qwen3.5 serving does not support `echo=true`. Both the legacy and stepped vLLM
bridges submit `echo: false`, so prompt echo has never been part of the current
HTTP serving contract. Direct engine callers can still construct the shared
request type with `echo=true`; the Qwen3.5 scheduler now rejects those requests
immediately after queue drain and cancellation pruning, before load accounting,
capacity admission, backend-state allocation, or TP command dispatch.

Qwen3.5 no longer emits `TokenEvent::PromptTokens` after prefill. The shared
event variant remains in `pegainfer-frontend` because other model lines still
use it, but it is sealed off from the Qwen3.5 scheduler path. A focused release
test requires a `Rejected` event with the original prompt length and proves the
request is absent from the vector eligible for backend admission.

Verification on the local SM86 environment passed:

- focused unsupported-echo release test: `1 passed`;
- complete Qwen3.5 release library suite: `95 passed`, `0 failed`, `7 ignored`;
- all retained real TP2 executor/scheduler gates on two RTX 3090 GPUs:
  `7 passed`, `0 failed`;
- release Clippy with `-D warnings` and formatting checks passed.

#### Model-backed test fixtures are explicit

Qwen3.5 model-backed tests no longer embed a developer-local absolute weights
path or fall back to a repository-relative model directory. The shared test
fixture resolver requires `PEGAINFER_TEST_MODEL_PATH`, reads and parses its
`config.json`, and confirms the Qwen3.5 model identity before any model or CUDA
initialization. A missing, non-UTF-8, empty, unreadable, malformed, or non-Qwen3.5
fixture prints one test-specific `SKIP` diagnostic and returns without GPU work.
The optional `PEGAINFER_TEST_FRONTEND_MODEL_PATH` uses the same validation when
set and otherwise reuses the validated engine fixture.

The resolver source lives under `tests/common` and is also included only for
crate unit-test builds, so the six retained TP executor gates, the mixed-step
scheduler gate, and the integration targets share the same contract without
adding a production API. Static review confirms that no Qwen3.5 test retains a
private model path or implicit `DEFAULT_MODEL_PATH` fallback.

Verification on the explicit public-compatible SM86 fixture passed:

- missing fixture, missing `config.json`, and malformed JSON probes emitted the
  expected diagnostics without GPU initialization;
- release all-target compile and Clippy with `-D warnings` passed;
- regular release library suite: `95 passed`, `0 failed`, `7 ignored`;
- retained real TP2 library gates: `7 passed`, `0 failed`;
- TP2 scheduler E2E, short/long HF gates, and HTTP serving gate passed; HF
  mean/p99 deltas remained `0.0260/0.1081` sequential, `0.0267/0.1167`
  batched, and `0.0228/0.0689` long;
- the pinned multi-turn gate rerun completed `12/12` conversations and `44/44`
  turns, followed without restart by `4/4` conversations and `8/8` turns, all
  with zero failures; graceful shutdown returned both TP GPUs to idle.

Delivered constraints:

- Support mixed prefill+decode scheduler steps under TP.
- Preserve deterministic collective ordering across ranks.
- Return mixed prefill/decode artifacts from the primary rank.
- Use the `RequestId`-keyed `UnifiedPlan` and artifact contract in `tp-design.md`; P2a does not introduce CUDA Graph padded-slot or compaction semantics.
- Validate finish/drop/client-disconnect cleanup under mixed-step execution.
- Keep TP CUDA Graph disabled unless a separate graph design is completed.
- Keep the fixed 16-device Triton AOT handle table and add startup-time validation for unsupported logical CUDA ordinals.

Why this should be separated from GDR sharding:

- Mixed-step scheduling is an execution-protocol problem.
- Sharded linear-attention/GDR is a model-state-shape problem.
- Combining them would make failures hard to attribute.

### P2b: sharded linear-attention/GDR state

Shard the Qwen3.5 linear-attention/GDR path after P2a establishes the mixed-step and state-lifecycle contract.

Expected work:

- shard linear-attention projection weights
- shard conv state and GDR recurrent state by local value/key heads
- adapt or regenerate GDR kernels for local state shapes
- keep recurrent/conv state rank-local and request-local
- all-reduce only after local linear-attention `out_proj`
- report matched Phase 1 TP2 versus P2b TP2 HBM/latency/throughput data before making a performance claim

Non-negotiable invariant:

- Never all-reduce GDR recurrent state or conv state. These states are owned by rank-local request state.

## Follow-Ups

- Design and implement P2B sharded linear-attention/GDR state without weakening the completed P2A lifecycle and ID contracts.
- Promote any stable contract changes discovered here back into `tp-design.md` through the design-doc branch.
- Decide whether Qwen3.5 server CLI should accept arbitrary TP device ordinals instead of only `0..tp_size`.
- Consider lifting the per-device Triton AOT handle lesson into a kernels or runtime subsystem doc if another model hits the same issue.
