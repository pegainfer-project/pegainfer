# Qwen3.5 Tensor Parallelism Design

> **TL;DR:** Qwen3.5 TP Phase 2 is two separately delivered correctness milestones: P2a adds eager `RunUnifiedStep` with a shared ordered `RequestId` plan while retaining Phase 1 replicated GDR; P2b shards the head-indexed linear-attention/GDR surface and adds only the hidden all-reduce after local `out_proj`.
>
> **Last touched:** 2026-08

## Goal

Add tensor-parallel support for `Qwen3.5-4B` by reusing the Qwen3 TP runtime instead of designing a second parallel execution stack.

The implementation should be degree-parametric where the model dimensions divide cleanly. `TP=2` is the first validation target, not an architectural limit. Unsupported or indivisible degrees must fail closed before model load.

## Qwen3 Runtime Reuse

Reuse the Qwen3 TP shape:

- controller/worker broadcast execution model
- `RequestId` request identity
- coarse-grained prefill/decode/unified/drop step protocol
- rank-local worker-owned model state
- rank-local CUDA context, cuBLAS, and NCCL resources
- hidden all-reduce after row-parallel projections
- replicated embedding/lm_head as the first-pass simplification

Qwen3.5-specific design work should stay focused on model geometry and state ownership: hybrid layer layout, gated q projection, linear-attention conv state, and GDR recurrent state.

## Boundaries

This design does not cover multi-node TP, data parallelism, pipeline parallelism, vocab-parallel embedding/lm_head, or Qwen3.5 prefix-cache/recurrent-state snapshots.

Phase 1 does not shard linear attention or change GDR kernel shapes. Phase 2 does not change GDR math, does not all-reduce recurrent state, and does not move recurrent state ownership back into the scheduler.

## Settled Phase 1 Contract

These decisions are settled before implementation starts.

- `TP=1` must preserve the current single-GPU behavior.
- `TP=2` is the first correctness target. The implementation may stay degree-parametric, but unsupported or indivisible degrees must fail before model load.
- `TP > 1` is eager-only in Phase 1. CUDA Graph under TP must fail closed instead of silently falling back or partially capturing.
- Reuse the Qwen3 controller/worker broadcast execution model and avoid a second long-lived Qwen3.5-specific TP runtime shape.
- Shard dense full-attention and MLP operators.
- Replicate embedding and tied `lm_head`.
- Replicate linear-attention/GDR weights in Phase 1.
- Each rank worker owns and mutates its own full linear-attention conv state and GDR recurrent state copy.
- The scheduler owns logical request lifecycle and logical KV/page lifecycle only.
- Full-attention KV is physically rank-local and sharded by local KV heads, but one logical request/page assignment is mirrored across all ranks.
- `DropRequest`, finish cleanup, cancellation cleanup, and client disconnect must release or reset the corresponding rank-local KV/recurrent/conv state on every rank by `RequestId`.
- Qwen3.5 gated `q_proj` slicing is an explicit acceptance gate: every rank must receive both q rows and gate rows for its local query heads.
- MLP gate/up row sharding and down column sharding require explicit reconstruction or layout tests.

## Phase 2 Delivery Boundaries

P2a and P2b are separate implementation series. P2a must complete its protocol and lifecycle gates before P2b changes loader, kernel, or state shapes. This keeps worker-protocol failures distinguishable from local-head loader/kernel/state failures.

The following are separate follow-up RFCs, not Phase 2 deliverables:

- TP CUDA Graph capture/replay, including graph slots, padding, synchronized capture, and recurrent/conv D2D compaction.
- TP-aware prefix caching and recurrent-state snapshots.
- Vocabulary-parallel embedding or `lm_head`.
- Multi-node TP, data parallelism, and pipeline parallelism.

Phase 2 is a correctness and per-rank HBM-reduction milestone. It makes no speedup promise: P2b adds one hidden all-reduce in each of the 24 linear-attention layers. Report a matched Phase 1 TP2 versus P2b TP2 A/B before making any performance claim.

## Why Dense First, GDR Second

Qwen3.5 has two separable TP problems.

The dense part is already proven by Qwen3: full-attention head sharding, local KV heads, MLP intermediate sharding, all-reduce after row-parallel projections, and worker-thread CUDA/NCCL execution.

The linear-attention part is Qwen3.5-specific: conv state and GDR recurrent state are long-lived request state, current GDR AOT kernels are built for the global value-head shape, and `DropRequest` cleanup plus re-admission must preserve rank-local recurrent-state boundaries. If dense TP and GDR TP land together, failures are hard to attribute. Phase 1 narrows correctness debugging to runtime + dense sharding; Phase 2 then isolates the GDR/recurrent contract. CUDA Graph slot, padding, and compaction semantics remain a separate follow-up RFC.

## Architecture Summary

Qwen3.5-4B:

- 32 layers: 24 linear attention + 8 full attention
- full-attention layers: `3, 7, 11, 15, 19, 23, 27, 31`
- `hidden_size = 2560`
- `intermediate_size = 9216`
- tied embedding/lm_head
- `vocab_size = 248320`

Full attention:

- `num_attention_heads = 16`
- `num_key_value_heads = 4`
- `head_dim = 256`
- `q_dim = num_attention_heads * head_dim = 4096`
- `kv_dim = num_key_value_heads * head_dim = 1024`
- q projection includes an output gate, so gated q projection output dim is `2 * q_dim = 8192`

Linear attention:

- `linear_num_key_heads = 16`
- `linear_key_head_dim = 128`
- `linear_num_value_heads = 32`
- `linear_value_head_dim = 128`
- `linear_q_dim = linear_num_key_heads * linear_key_head_dim = 2048`
- `linear_k_dim = linear_q_dim`
- `linear_v_dim = linear_num_value_heads * linear_value_head_dim = 4096`
- `linear_qkv_dim = linear_q_dim + linear_k_dim + linear_v_dim = 8192`
- `linear_z_dim = linear_v_dim = 4096`
- recurrent state per linear layer: `[linear_num_value_heads, linear_key_head_dim, linear_value_head_dim] f32`
- conv state per linear layer: `linear_qkv_dim * (conv_kernel_dim - 1)` bf16

## Partition Contract

For any candidate `tp`, require:

- `num_attention_heads % tp == 0`
- `num_key_value_heads % tp == 0`
- `intermediate_size % tp == 0`
- Phase 2 additionally requires `linear_num_key_heads % tp == 0` and `linear_num_value_heads % tp == 0`

Full attention local dimensions:

- `local_q_heads = num_attention_heads / tp`
- `local_kv_heads = num_key_value_heads / tp`
- `local_q_dim = local_q_heads * head_dim`
- `local_kv_dim = local_kv_heads * head_dim`
- `local_gated_q_dim = 2 * local_q_dim`

Qwen3.5 full-attention `q_proj` must be sharded by head-local q/gate pairs. Each rank owns a contiguous query-head range, and for each owned head it must receive both that head's q rows and that head's gate rows. Do not reuse a naive contiguous row shard if the physical layout can split q rows from their gate rows.

MLP local dimensions:

- `local_intermediate = intermediate_size / tp`
- local fused `gate_up_proj` rows: `2 * local_intermediate`
- local `down_proj` input cols: `local_intermediate`

Linear-attention local dimensions for Phase 2:

- `local_linear_key_heads = linear_num_key_heads / tp`
- `local_linear_value_heads = linear_num_value_heads / tp`
- `local_linear_q_dim = local_linear_key_heads * linear_key_head_dim`
- `local_linear_k_dim = local_linear_q_dim`
- `local_linear_v_dim = local_linear_value_heads * linear_value_head_dim`
- `local_linear_qkv_dim = local_linear_q_dim + local_linear_k_dim + local_linear_v_dim`
- `local_linear_z_dim = local_linear_v_dim`
- local recurrent state: `[local_linear_value_heads, linear_key_head_dim, linear_value_head_dim] f32`
- local conv state: `local_linear_qkv_dim * (conv_kernel_dim - 1)` bf16

## Phase 1: Dense TP, Replicated Linear Attention

Shard:

- full-attention `q_proj`, `k_proj`, `v_proj`, `o_proj`
- full-attention KV cache over local KV heads
- MLP `gate_proj`, `up_proj`, `down_proj`

Replicate:

- embedding and tied lm_head
- all linear-attention weights
- all linear-attention conv state
- all GDR recurrent state
- existing GDR kernels and scratch shapes

Execution:

- full-attention: local q/k/v + local attention + local `o_proj`, then all-reduce hidden
- MLP: local gate/up + local activation + local `down_proj`, then all-reduce hidden
- linear attention: every rank runs the full layer and updates a full local recurrent-state copy; do not all-reduce replicated linear-attention output

State ownership:

- scheduler owns request admission, request identity, logical page allocation, streaming handles, sampling params, generation counters, and finish bookkeeping
- rank workers own rank-local model shards, rank-local physical KV buffers, rank-local decode buffers, and rank-local recurrent/conv state
- rank 0 is not special for state mutation; it follows the same worker command protocol as other ranks
- non-primary workers may return acknowledgement or step failure only, while the primary worker returns artifacts for scheduler-side result resolution
- all workers must observe the same ordered `RunPrefillChunks`, `RunDecodeStep`, `DropRequest`, and `Shutdown` commands

CUDA Graph:

- Phase 1 TP execution is eager-only
- `tp_size > 1` with CUDA Graph enabled must return an explicit startup/configuration error before serving requests
- TP graph capture is a follow-up because Qwen3.5 graph state includes recurrent slots, slot compaction, padding slots, and NCCL ordering questions

Validation scope:

- first validated degree: `TP=2`
- Qwen3.5 HF logits gate
- Qwen3.5 scheduler e2e
- long prompt / chunked prefill path
- finish, explicit drop, cancellation, and client-disconnect cleanup by `RequestId`
- subsequent admission with a new `RequestId` observes no stale KV, recurrent, or conv state
- gated `q_proj` head-local q/gate slicing test
- MLP gate/up shard and down shard reconstruction/layout test
- basic TP2 serving smoke
- startup fails closed for unsupported or indivisible degrees
- startup fails closed for `tp_size > 1` with CUDA Graph enabled

## P2a: Eager TP Unified Execution

P2a implements eager TP `RunUnifiedStep` while retaining the Phase 1 replicated linear-attention/GDR weights, kernels, conv state, recurrent state, and scratch shapes.

Every rank receives the same canonical `UnifiedPlan`: ordered prefill and decode items, each carrying a `RequestId` and the request-local execution inputs needed for that row. The order is the collective-order contract; every rank executes the actual plan rows in that order. P2a does not introduce CUDA Graph padded-slot semantics, D2D state movement, or a cross-rank slot-compaction protocol.

State and artifacts are keyed by `RequestId`:

- worker-local KV, conv, and recurrent state is found, created, promoted, and released by `RequestId`;
- the primary worker returns prefill and decode artifacts carrying their `RequestId`;
- the scheduler resolves artifacts by ID and rejects unknown, duplicate, or missing results instead of relying on returned row position;
- finish, explicit drop, cancellation, and client disconnect broadcast the same `DropRequest(RequestId)` lifecycle command to every rank;
- each worker returns `DropAck { existed: bool }`, where `existed` reports whether that rank actually found and removed the request state;
- every controller-side drop carries a `DropExpectation::{MustBeAbsent, MustExist}` derived from scheduler-owned lifecycle state, not from a separate `worker_state_materialized` flag;
- cancellation before the first successful prefill dispatch requires `MustBeAbsent` and an exact-rank all-false `DropAck` set; partially prefetched, active, disconnected, and completion-candidate requests require `MustExist` and an exact-rank all-true set;
- uniformity alone is insufficient. All-false for `MustExist`, all-true for `MustBeAbsent`, or mixed true/false values prove lifecycle divergence and poison the whole TP executor even if the drop leaves every rank absent. Dispatch failure, a missing acknowledgement, or a rank-local drop failure is likewise fatal; the scheduler must not warn and continue promotion or serving.

The scheduler allocates a TP `RequestId` before the first prefill command materializes worker state, so cancellation in that interval legitimately requires `MustBeAbsent`; every successfully prefetched or active request requires `MustExist`. If execution mutates state and then fails, the executor is already poisoned and no healthy-path drop is attempted. Worker replies prove cross-rank existence, while scheduler lifecycle determines which value is valid.

Successful request completion has a fail-closed commit boundary. After computing a terminal artifact, the scheduler keeps the request unresolved in a local completion candidate and withholds its user-visible terminal events. An EOS candidate buffers only `Finished`; a length-limited candidate buffers the final `Token` followed by `Finished`; completion on the first prefill token, including `max_tokens <= 1`, follows the same rule. A completion candidate is materialized, so the scheduler first requires a valid `MustExist` all-rank all-true `DropAck` contract, then removes the logical request from scheduler state, and only then publishes the buffered events in order. A client that observes `Finished` can therefore rely on consistent rank-local cleanup acknowledgement for that request.

If completion drop fails, the scheduler publishes neither the buffered final `Token` nor `Finished`, poisons the replica, and keeps the candidate unresolved for the complete terminal error fan-out. The client may already have received earlier streamed tokens, so this boundary makes only successful termination atomic; it does not make the complete streamed response transactional. The completion candidate is a scheduler-local prepare/commit abstraction, not a distributed `ValidateUnified -> ExecuteUnified` protocol. A process-level fail-stop may still prevent both the acknowledgement and the terminal `Error` from being delivered.

Controller response collection validates an exact rank set, not only a message count. Every response rank must be in `0..world_size` and appear exactly once. Ping requires `Ack` from every rank; drop requires a `DropAck` from every rank whose existence values match the controller's `DropExpectation`; prefill, decode, and unified execution require exactly one matching typed result from rank 0 and `Ack` from every non-primary rank. Missing, duplicate, out-of-range, wrong-variant, mixed drop-existence, or uniformly unexpected drop-existence responses are protocol failures. If workers may already have mutated state, any such response-set failure poisons the complete TP replica.

Cancellation cleanup is a scheduler-tick boundary, not a late planning cleanup. At the start of each tick the scheduler first merges deferred work with newly received submissions, then prunes every already-closed sink from active, prefilling, and pending work. Healthy active/prefilling TP removals complete their all-rank `DropRequest` before the scheduler publishes load or computes admission capacity; pending requests have no worker state and are discarded directly. The required order is `drain -> prune -> publish load -> admission -> plan`. This keeps cancelled requests out of running/waiting metrics, decode slots, future-KV reservation, and the current tick's prefill budget. A cancellation racing after the prune check is still retired by the existing token-send failure path.

Artifact alignment follows the ordered plan rather than returned row position. Prefill expects artifacts only for `finish_prefill == true` IDs and preserves explicit absence for non-final rows; decode expects one artifact for every row. Unknown, duplicate, or missing IDs are fatal after execution. This TP adapter contract does not require the TP1/single-GPU logits-and-sampling path to adopt sparse artifacts.

P2a separates recoverable plan rejection from fatal replica failure. Controller-provable structural errors reject before dispatch without poisoning. Worker-local existence, phase, materialization, or capacity mismatches are not protected by an all-rank validation barrier and are therefore replica-fatal, as is any CUDA/NCCL, response-set, artifact, or lifecycle failure after execution is released.

When fatal failure returns control, the scheduler closes and drains submissions, emits exactly one terminal `Error` for every unresolved stable or tick-local request, publishes an idle load snapshot, exits, and begins whole-executor teardown. Exactly-once fan-out comes from consuming mutually exclusive request owners; pre-admission requests do not yet have a TP `RequestId`, so that ID is not a global deduplication key. The scheduler does not retry per-request drops or claim rank-local cleanup after poison. Collective or teardown timeout remains process-level fail-stop, where client fan-out cannot be guaranteed; rollback, communicator recovery, and in-place restart remain out of scope.

The worker's internal request table may remove and reinsert entries as an implementation detail, but that is not CUDA Graph slot semantics and must not require state copying between request identities.

The fixed 16-device Triton AOT handle table remains. Before model loading or worker launch, TP startup validates every requested logical CUDA ordinal against that supported range; dynamic handle allocation is not a P2a prerequisite. `tp_size > 1` with CUDA Graph requested continues to fail closed.

P2a acceptance retains the Phase 1 TP1/TP2 gates and adds TP2 mixed chunked-prefill/decode, cleanup and re-admission, strict artifact-ID checks, and fatal-path coverage. Drop tests must accept only `MustBeAbsent`/all-false and `MustExist`/all-true exact-rank sets, rejecting mixed or uniformly unexpected existence and malformed rank/reply sets. Completion tests must withhold EOS, length, and immediate-prefill success events until `MustExist` cleanup succeeds; controller-rejected plans must preserve executor health, while worker-local or post-execution failures must stop the scheduler and fan out terminal errors to every accepted unresolved request. All work remains eager and uses the replicated GDR path.

Lifecycle observability, cancellation ordering, fail-closed cleanup, and unified execution remain separately reviewable implementation boundaries. Cancellation ordering fixes a pre-existing TP1 scheduler issue but stays in P2a because unified planning depends on the pruned state.

## P2b: Local-Head Linear Attention / GDR

P2b converts the 24 linear-attention layers from replicated execution to true TP execution. It additionally requires `linear_num_key_heads % tp == 0` and `linear_num_value_heads % tp == 0`; unsupported degrees and unsupported local kernel shapes fail before model loading.

Shard every head-indexed linear-attention/GDR surface by the local key/value-head ranges:

- `in_proj_qkv`, preserving local q/k/value channel layout;
- `in_proj_z`, `in_proj_b`, `in_proj_a`, `conv1d_weight`, `dt_bias`, and `A_log`;
- `out_proj` input columns;
- conv state, recurrent state, GDR scratch, and intermediate buffers.

The head-dimension `norm_weight` remains deliberately replicated because it is shared by every local value head; it is not a head-indexed state or collective surface. Embedding and tied `lm_head` also remain replicated.

Each rank runs local projections, convolution, GDR prefill/decode kernels, gated RMSNorm/output-gate work, and local `out_proj` against local dimensions. The only linear-attention collective is the hidden all-reduce after `out_proj`. Conv state and GDR recurrent state are request-local and rank-local for their full lifetime and are never all-reduced or centralized.

P2b acceptance requires loader reconstruction/layout tests, local AOT-kernel shape validation, rank-local allocation checks, Phase 1 and P2a regression gates, short and long TP2 HF replay, cleanup without stale request state, and a matched Phase 1 TP2 versus P2b TP2 HBM/latency/throughput report. The report is evidence, not a speedup threshold.

### vLLM Reference

Use vLLM's `Qwen3NextForCausalLM` / `QwenGatedDeltaNetAttention` as the reference contract, not as code to copy mechanically:

- GDN state shape depends on `tp_size`
- q/k/v/z projections are tensor-parallel column projections
- `out_proj` is row-parallel and reduces back to full hidden
- `dt_bias` and `A_log` are sharded over local value heads
- b/a projections are local-value-head aware; some quantized paths may replicate small projections and slice locally
- GDR prefill/decode kernels consume local head/state shapes

PegaInfer-specific work remains: worker-owned rank-local recurrent state, `RequestId` lifecycle, request-state removal and re-admission, `DropRequest` cleanup, and fail-closed kernel-shape validation.

Validation scope:

- Phase 1 gates still pass
- long HF logits replay under the validated degree
- request-state cleanup and re-admission replay
- recurrent-state cleanup on finish/drop/cancellation
- no stale local recurrent state after a new `RequestId` is admitted

## References

- `docs/models/qwen3/tp-design.md`
- `pegainfer-qwen3/src/config.rs`
- `pegainfer-qwen3/src/executor.rs`
- `pegainfer-qwen35/src/config.rs`
- `pegainfer-qwen35/src/weights.rs`
- `pegainfer-qwen35/src/recurrent_state.rs`
- `pegainfer-qwen35/src/batch_decode.rs`
- vLLM `Qwen3NextForCausalLM`
- vLLM `QwenGatedDeltaNetAttention`
