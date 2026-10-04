# Qwen3.5 TP Implementation

> **TL;DR:** Qwen3.5 TP uses rank-local model state behind one step-contract scheduler, with start-gated mixed execution, ID-aligned results and acknowledged retirement. Decode is batched eager or pre-captured CUDA Graph replay when the decode GQA group is compiled. Remaining TP work: group-6 batch-decode kernels for 27B graphs, perf gates.
>
> **Last touched:** 2026-10

## Runtime ownership

The architecture contract is in [TP design](tp-design.md). The implementation
is split between `scheduler/{mod,backend,tp}.rs` and `tp_executor.rs`:

- `Qwen35Scheduler` owns logical admission, prefill/decode planning and dense
  decode-slot order. It writes request updates through `RequestLedger`; the
  shared driver publishes metrics and commits those updates.
- `TpSchedulerBackend` maps frontend request ownership to model-local
  `RequestId`s and builds commands for `Qwen35TpExecutor`. The two ID types
  serve different boundaries; client labels are not lifecycle keys.
- Rank workers own model shards, CUDA contexts, thread-local cuBLAS, NCCL
  comms, KV buffers and recurrent/conv tensors. They bind their contexts and
  initialize CUDA/NCCL resources on their own threads. The controller starts
  its scheduler after worker startup completes, without a second handshake.
- The controller's `KvCacheManager` owns logical request pages. Immutable
  `KvView`s describe the same page IDs on every rank; each worker writes its
  own KV shard. Capacity is capped by the smallest rank-local pool after
  validating consistent geometry and snapshot slots.

TP ordinal validation precedes model loading. Ordinals must be distinct and
fit the generated Triton AOT handle table. The generated stubs cache module
and function handles per device and check the device index before accessing
the table; process-global CUDA function handles are invalid across devices.

## Shards and execution

Full-attention and MLP projections use dense TP sharding. Gated `q_proj`
weights are interleaved by head (`[head0 q][head0 gate][head1 q][head1 gate]`),
so loaders preserve head ranges rather than rebuilding separate q/gate
halves. Embeddings and `lm_head` remain replicated.

Linear-attention/GDR weights, recurrent/conv state and scratch shapes are
rank-local. The local linear output is followed by an all-reduce. Recurrent
state and KV memory sizing use the configured `max_batch`, rather than
reserving an unrelated default number of slots.

Decode rows run as one batched forward on each rank, with sampling and
logprobs returned by rank 0. Eager GDR pointer tables are allocated once at
scheduler capacity and refilled from live rows each step: slot compaction
must not leave stale row addresses. Request-local sampling steps must remain
independent of a request's position in the batch.

When prefill and decode coexist, the common planner produces one
`RunUnifiedStep`. Every rank executes prefill before decode in the same
collective order. The scheduler consumes decode results before promoting
prefill results, allowing acknowledged retirement to free slots first.
TP does not support the single-GPU stream-overlap or adaptive `auto` policies.

## Dispatch and result contracts

State-mutating commands share a `Pending -> Execute | Cancel` start gate.
The controller enqueues to every rank before choosing `Execute`; a partial
enqueue failure cancels the delivered prefix and poisons the executor.
Workers cannot mutate request state or enter kernels/collectives before the
gate opens. Gate transitions execute unconditionally, never only inside a
`debug_assert!`.

Structural plan validation runs before dispatch. Unified plans require both
halves, fit the configured capacity and contain unique, mutually disjoint
prefill/decode IDs and non-empty chunks. Workers revalidate local existence,
phase and capacity after the gate opens. A post-dispatch inconsistency is
replica-fatal; it is not retried as an individual request failure.

Replies must cover every rank exactly once with the expected variant. Rank
0 supplies sampled prefill/decode/unified artifacts; other ranks acknowledge.
Drop uses `DropAck { existed }` from every rank. Duplicated or missing ranks,
out-of-range ranks, unexpected variants and inconsistent existence results
poison the executor.

The scheduler aligns returned artifacts by model-local `RequestId`, not row
position. Decode needs one result per active ID. Prefill needs a result only
for final chunks: a non-final chunk has no artifact, while a final artifact
may legitimately have no requested logprobs. Unknown, duplicated, non-final
or missing result IDs after execution poison the replica.

## Cancellation, completion and failure

Cancellation is the frontend abort flag observed through the ledger. Pending
requests retire before allocation; resident requests use normal backend
cleanup before planning, so released capacity is available to admission in
the same step.

Admission creates rank-local recurrent state even on a cold prefix miss.
Consequently, cancelling an admitted prefill at cursor zero still requires
`DropExpectation::MustExist`, as do active retirement and final-prefill
cleanup.
The exact-rank drop contract accepts all-true for `MustExist` and all-false
for `MustBeAbsent`; mixed or unexpected answers are lifecycle divergence.

Successful completion is recorded in the ledger only after backend
retirement succeeds. In TP this requires all-rank drop acknowledgement, so a
cleanup failure cannot leak a successful terminal or its final token.
Remaining open accounts are failed by the shared driver when the scheduler
returns an engine-fatal error. Poisoned workers are not retried through
healthy cleanup; backend teardown owns their remaining resources.

## Joint prefix cache

Prefix reuse requires full-attention KV and a complete recurrent/conv
snapshot at the same boundary. Key, pin and LRU metadata is centralized;
snapshot tensors remain rank-local. Publication reserves a common slot,
saves it on every rank and verifies the boundary before publishing the key.
Restore reports a hit only after every rank confirms that boundary.

The snapshot budget is reserved independently on each rank through
`qwen35-prefix-cache-mib`; zero keeps cold serving. See
[prefix-cache design](prefix-cache.md) for boundary and eviction rules.

## P2c — CUDA Graph under TP

Graph mode requires both the launch option and a compiled decode GQA group.
4B/9B TP2 can capture; group-6 27B execution stays eager until that kernel is
available. A graph request falling back to eager is logged at startup.

The scheduler assigns dense `slot_idx` values and supplies compaction moves;
workers do not infer slot ownership. Each graph worker owns fixed-address
state for the rounded batch capacity. A request's first decode row copies
its prefill recurrent state into its assigned slot. Retirement validates
occupancy and applies the explicit D2D compaction; a request retired before
its first decode legitimately has no materialized slot.

Startup warms NCCL collectives, captures and launches each required bucket,
then verifies all buckets are captured. Lazy NCCL connection inside capture
can hang, so warmup and the capture watchdog are required. Serving replays
existing graphs; it never captures during a live request. Sampling and
logprobs stay outside capture. Graph state is destroyed before the model's
NCCL resources during worker teardown.

Low-level convenience executor calls maintain their own guarded slot
tracker. The scheduler uses explicit-slot commands and must not also update
that tracker.

## Validation surface

The tests exercise different ownership boundaries:

- TP executor gates observe rank-local state to prove partial-dispatch
  cancellation, lifecycle divergence, disconnected-worker failure, mixed
  execution, full capacity recovery and clean readmission under a fresh ID.
- Scheduler lifecycle tests cover acknowledged completion, cancellation,
  request rejection and cleanup failures.
- `e2e_scheduler` covers eager and Graph request flow, mixed sampling and
  post-cancellation health. `prefix_cache` covers joint restore and eviction.
- `hf_golden_gate` checks short/long numeric replay, batched buckets and
  post-compaction replay. Fixtures match `config_sha256` and the checkpoint
  revision, including Qwen3.8; see [accuracy](accuracy.md).
- `serving_tp2` covers real HTTP model discovery, JSON/SSE completions,
  concurrency and logprobs. Historical dependent multi-turn evidence is
  retained in the [TP2 serving benchmark](../../benchmarks/qwen35-tp2-phase2a-multiturn.md).

The shared model-fixture resolver uses `PEGAINFER_TEST_MODEL_PATH`; an absent or invalid
fixture emits `SKIP`, which is not a GPU pass. TP2 tests are ignored by
default and use `PEGAINFER_TEST_TP_DEVICES` (two distinct ordinals, default
`0,1`). `PEGAINFER_TEST_FRONTEND_MODEL_PATH` optionally selects frontend
metadata; otherwise it uses the engine fixture. Hand-downloaded HF fixtures
need a resolvable revision or `PEGAINFER_TEST_MODEL_REVISION`.

Run GPU gates serially to avoid capacity pressure and cuBLASLt tuning
interference between independent executors.
