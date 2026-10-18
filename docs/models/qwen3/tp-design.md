# Qwen3 Tensor Parallelism Design

> **TL;DR:** Add `TP=2` support for `Qwen3-4B` as the first model-parallel milestone. The goal is correctness and a clean architectural foundation for larger dense models and future MoE work, with the runtime moving toward a controller-plus-workers broadcast execution model instead of scheduler-owned cross-thread mutable state.
>
> **Status:** Active. `Qwen3-4B` has now been brought up end-to-end with `TP=2` on a single machine. `TP=8` has also been smoke-tested on an 8x4090 host, but the implementation still carries first-pass runtime debt and has not yet gone through systematic correctness validation.

## Goal

Add tensor parallelism for `Qwen3-4B` with a narrow and explicit first target:

- support `TP=2`
- preserve `TP=1`
- keep the milestone focused on model-parallel correctness
- establish the right abstractions for later large-model and MoE work

This milestone is about making pegainfer capable of serving a single model replica across two GPUs. It is not about multi-replica throughput scaling.

## Why This Matters

Large-model serving and MoE serving both require model-parallel building blocks. For pegainfer, tensor parallelism is the first such building block.

The immediate value is:

- larger dense models become reachable without forcing a single-GPU fit
- the runtime starts carrying rank-local weights and rank-local execution state
- later MoE work can build on model-parallel foundations instead of retrofitting them into a single-GPU design

## Scope

This first pass is intentionally narrow:

- model: `Qwen3-4B`
- parallel degree: `TP=2`
- focus: correctness and architecture
- deployment target: a single machine

The first milestone does not need to solve every parallelism problem. It needs to prove that pegainfer can run one dense model replica across two GPUs without breaking correctness or making the architecture harder to evolve.

## Design Constraints

- Tensor parallelism must be model-parallel, not data-parallel in disguise.
- The design must serve future larger dense models and MoE, not only Qwen3-4B.
- `TP=1` must remain a supported and healthy path.
- The first version may simplify some weight handling, but those simplifications must not block later movement toward a more fully sharded design.
- The external user experience should stay simple. Tensor parallel support should not require the user to reason about a heavyweight distributed system.
- The design should avoid coupling the abstraction too tightly to one specific kernel path or one specific Qwen3 implementation detail.

## Explicit Non-Goals

The first TP milestone does not aim to do the following:

- support `Qwen3.5`
- support MoE expert parallelism
- support data parallelism
- support pipeline parallelism
- preserve current CUDA Graph behavior from day one
- introduce vocab-parallel embedding or vocab-parallel `lm_head` in the first pass
- optimize every path for peak throughput before the basic design is proven correct
- serve `--batch-invariant`, which the builder rejects for `tp_size > 1`: the prefill and unified all-reduces are eager and unbucketed, so NCCL can pick a different algorithm and protocol as the message size moves with batch composition, and that reduction order has never been audited

## Simplifications Allowed In First Pass

The first pass is allowed to trade some efficiency for speed of validation, as long as the trade does not poison the long-term design.

Allowed simplifications:

- tied embedding / `lm_head` may be replicated instead of sharded
- some paths may be brought up incrementally as long as the final milestone still has a clear acceptance target
- correctness and maintainable abstraction take priority over immediate performance parity

These are first-pass scope controls, not permanent architectural commitments.

## Tensor-Parallel Partitioning Spec

The first `Qwen3-4B` tensor-parallel milestone should follow the mainstream dense-model TP layout used by systems such as vLLM and SGLang:

- attention projections are partitioned by head
- MLP projections are partitioned by intermediate dimension
- layer outputs that rejoin the residual stream are reduced across ranks
- tied embedding / `lm_head` is replicated in the first pass

This is the intended layout for `TP=2`.

### Qwen3-4B Local Dimensions At TP=2

`Qwen3-4B` runtime dimensions:

- `hidden_size = 2560`
- `num_attention_heads = 32`
- `num_key_value_heads = 8`
- `head_dim = 128`
- `intermediate_size = 9728`

Under `TP=2`, the local dimensions per rank are:

- local query heads: `16`
- local KV heads: `4`
- local query projection dim: `16 * 128 = 2048`
- local KV projection dim: `4 * 128 = 512`
- local intermediate dim: `9728 / 2 = 4864`

The first pass should explicitly require divisibility for these dimensions. If a model does not divide cleanly across TP ranks, it is out of scope for this milestone.

### Attention Projection Layout

For Qwen3 attention, the fused `qkv_proj` should be partitioned by head-aligned output slices.

Global layout:

- `qkv_proj`: `[q_dim + 2 * kv_dim, hidden_size]`
- for Qwen3-4B: `[4096 + 1024 + 1024, 2560] = [6144, 2560]`

Local layout at `TP=2`:

- local `q_proj`: `[2048, 2560]`
- local `k_proj`: `[512, 2560]`
- local `v_proj`: `[512, 2560]`
- local fused `qkv_proj`: `[3072, 2560]`

This partitioning is semantic, not just row-chunking by index. Each rank owns a contiguous subset of query heads and KV heads.

### Attention Output Projection Layout

The output projection should follow the standard row-parallel pattern.

- each rank consumes its local attention output
- each rank produces a partial hidden-state contribution
- the partial hidden states are combined with an `all-reduce`

The residual stream after this reduction is logically full-width hidden state.

### MLP Projection Layout

The MLP should be partitioned by intermediate dimension.

For Qwen3-4B:

- global `intermediate_size = 9728`
- local `intermediate_size = 4864` at `TP=2`

Projection layout:

- `gate_up_proj` is column-parallel over intermediate dimension
- `down_proj` is row-parallel and its outputs are combined with an `all-reduce`

With the current fused MLP layout, each rank owns its local fused gate/up rows:

- global fused `gate_up_proj`: `[2 * 9728, 2560] = [19456, 2560]`
- local fused `gate_up_proj`: `[2 * 4864, 2560] = [9728, 2560]`

As with attention, the residual stream after the MLP output reduction is logically full-width hidden state.

### Embedding And LM Head

For the first pass:

- token embedding is replicated
- tied `lm_head` is replicated

This is a deliberate simplification for the initial milestone. It is acceptable because the goal of the first pass is to validate model-parallel execution and establish the TP boundary, not to fully optimize vocab-side memory layout on day one.

### Communication Points

For the first dense TP pass, the communication pattern should stay minimal.

Per transformer layer, the expected TP collectives are:

- one `all-reduce` after attention output projection
- one `all-reduce` after MLP output projection

No additional collective requirements are introduced by the first-pass embedding / `lm_head` choice.

### Runtime Bring-Up Notes

The first end-to-end `TP=2` bring-up exposed a few concrete runtime hazards that are worth recording because they are not obvious from the high-level TP partitioning design alone.

- `cuBLAS` handle and workspace state must not be process-global when TP ranks execute on different GPUs from different threads
- TP worker threads must explicitly bind both the CUDA runtime device and the driver context before using cuBLAS, FlashInfer, or NCCL

In practice, the initial TP implementation initially hit:

- an illegal memory access reported later in `paged_kv_scatter_cuda`
- intermittent hangs after some requests had already succeeded

The root cause was not the scheduler boundary. It was runtime state management.

These issues have now been fixed in the current bring-up implementation:

- `cuBLAS` handles and workspaces needed to become thread-local
- TP worker threads needed explicit per-thread device binding before GPU work
- request-scoped worker-thread cuBLAS resources needed explicit teardown so repeated TP requests did not accumulate unstable per-thread state

The later TP correctness pass exposed a separate decode-path bug that should be recorded explicitly:

- decode was using a specialized paged KV append path that did not stay aligned with the generic paged scatter semantics used by prefill
- this caused decode-built KV state to drift from a fresh prefill-built KV state for the same logical prefix
- the fix was to stop using the decode-only append path and route decode KV writes through the same explicit paged scatter path used elsewhere

That decode-state corruption bug is now fixed. The remaining TP correctness work is narrower than the original bring-up failures.

This means the first-pass TP executor is now correct enough to run end-to-end, but the runtime shape is still more fragile than the eventual target design.

### First-Pass Validity Constraints

The `Qwen3-4B, TP=2` first pass assumes:

- `num_attention_heads % tp_size == 0`
- `num_key_value_heads % tp_size == 0`
- `intermediate_size % tp_size == 0`

These constraints are part of the milestone definition, not an implementation accident.

## ModelExecutor Abstraction

The next architectural step should be extracting a synchronous `ModelExecutor` boundary from the current scheduler-owned execution path.

This is the key abstraction for future model-execution strategies:

- single GPU execution
- tensor parallel execution
- later tensor-parallel plus expert-parallel execution

It should be the execution abstraction for one logical model replica. It should not become the abstraction for request queueing, service-layer data parallelism, or cluster orchestration.

### Why This Is The Next Step

The current scheduler owns both:

- control-plane logic such as active/deferred request management, admission control, and token streaming
- execution-plane logic such as prefill, decode, and unified-step GPU execution

Tensor parallelism changes the execution plane much more than it changes the control plane.

So the right next step is not a new scheduler design. The right next step is to extract the execution plane behind a stable executor interface while keeping the scheduler responsible for request lifecycle, KV allocation, and batching policy.

### Control Plane Versus Execution Plane

The scheduler should remain the control plane.

The scheduler should continue to own:

- request queueing
- active / deferred request lifecycle
- admission control
- `KvPool`
- KV page allocation and recycling
- deciding whether the next step is prefill, decode, or unified
- sampling policy
- token streaming
- finish reasons
- HTTP / API semantics

The executor should become the execution plane.

The executor should own:

- model weights
- shared execution resources such as decode buffers, graph state, and TP communication state
- the implementation of batch-level prefill, decode, and unified-step execution

This means the scheduler decides what batch should run next, while the executor decides how to execute that batch on the underlying device topology.

### Request-Owned Versus Executor-Owned State

The scheduler should continue to own request lifecycle state.

Examples:

- KV allocation state
- page lists or page-table metadata
- request-local sequence lengths
- last-token bookkeeping and generation counters

The executor should own shared execution resources.

Examples:

- model weights
- shared decode buffers
- CUDA graph state
- TP communication state
- rank-local scratch buffers

This split keeps admission control and KV budgeting where they already belong, while still moving model execution out of the scheduler.

### Interface Shape

The interface should be batch-step oriented, not kernel oriented.

The scheduler should build a batch specification for one of the existing runtime step types:

- prefill batch
- decode batch
- unified step

The executor should synchronously execute that batch specification and return the outputs needed for scheduler-side post-processing.

The first version should stay synchronous.

The current runtime already serializes GPU ownership through the scheduler thread, and the next architectural goal is to separate responsibilities, not to introduce a second concurrency model.

### Batch Specification

The batch specification should describe one execution step, not a whole request lifecycle.

The important information is:

- which requests participate in the step
- whether the step is prefill, decode, or unified
- the input tokens for each request
- mutable references to request-owned execution state
- KV page metadata or equivalent scheduler-owned KV views needed by the kernels

This keeps the executor narrow: it consumes one scheduler-chosen batch plan and executes it.

### Three Runtime Boundaries

The point of the next refactor is not to make the scheduler look smaller on paper.

The point is to isolate three different responsibilities that already change independently:

- control-plane step selection
- model execution
- step-result resolution back into request lifecycle state

Those responsibilities should be represented by three explicit boundary types.

#### `ExecutionPlan`

`ExecutionPlan` is the boundary between the scheduler and the executor.

Its job is to describe what should run in this step.

It should contain:

- the step kind
- the participating prefill and decode requests
- the input tokens or prompt slices for those requests
- mutable references or views into scheduler-owned request execution state such as `KvState`
- any per-step ordering or indexing the executor needs

It should not contain:

- finish reasons
- token streaming semantics
- HTTP / API semantics
- admission policy
- executor-internal resource ownership

Ownership model:

- owned and constructed by the scheduler
- consumed by the executor
- valid only for one execution step

#### `ExecutionArtifacts`

`ExecutionArtifacts` is the boundary between the executor and the step-result resolver.

Its job is to describe what the executor produced, before those results are interpreted as request lifecycle outcomes.

Examples:

- prefill logits
- unified-step prefill and decode logits
- an executor-owned decode-step view over batched decode buffers

It should contain raw execution products and executor-owned views.

It should not contain:

- finish reasons
- retirement decisions
- request promotions
- token events

Ownership model:

- produced by the executor
- consumed by a scheduler-local result-resolution layer
- short-lived, step-scoped

#### `StepEffects`

`StepEffects` is the boundary between the step-result resolver and the scheduler's long-lived request state.

Its job is to describe how this step changes request lifecycle state.

Examples:

- which pending requests become active
- which active requests retire
- which token events should be emitted
- prompt echo payloads
- finish reasons
- updates to `last_token` and generation counters

It should contain scheduler-facing state transitions and event payloads.

It should not contain:

- raw executor buffer views
- CUDA graph state
- TP communication state
- executor-owned staging resources

Ownership model:

- produced by the step-result resolver
- applied by the scheduler
- step-scoped, but expressed in scheduler terms rather than executor terms

### Why These Boundaries Matter

This is not abstraction for its own sake.

These boundaries isolate three different change vectors:

- batching and admission policy change the scheduler
- TP and other model-parallel execution strategies change the executor
- sampling, logprobs, echo, and finish handling change step-result resolution

## Controller / Worker Broadcast Execution Model

The execution model for Qwen3 tensor parallelism should now be treated as part of the main TP design, not as a separate note.

The core idea is:

- one controller decides the next step
- one primary worker plus zero or more additional rank workers execute it
- an ordered broadcast command stream keeps worker-local state synchronized

This is the direction we want for the steady-state runtime. The earlier shape where the scheduler owned rank-local mutable state and worker threads borrowed into it was acceptable for bring-up, but it should not be the long-term architecture.

### What We Are Rejecting

We do not want the long-term design to rely on:

- the scheduler thread owning all rank-local KV or execution objects
- worker threads borrowing `&mut` access into those objects
- raw pointer wrappers and timing assumptions as the main cross-thread correctness mechanism

Tensor parallelism is a replicated local-state problem, not a shared-mutable-state problem.

### Controller Responsibilities

The controller is the only place that decides:

- which requests participate in the next step
- whether the next step is prefill, decode, or unified
- which high-level lifecycle transitions should happen
- which ordered command should be broadcast next

The controller should not directly execute GPU work, including rank 0 work. Rank 0 should execute on the primary worker thread under the same protocol as the additional ranks.

### Worker Responsibilities

Each worker owns its rank-local execution state, including:

- its local model shard
- its local decode buffers
- its local KV state
- its local per-request execution state keyed by request identity

The scheduler should continue to own user-facing lifecycle state such as streaming handles, sampling params, generation counters, and finish bookkeeping. This preserves the existing control-plane role while moving rank-local execution state to the workers.

### Request Identity

The broadcast protocol should use an explicit process-local request identity:

- `RequestId(u64)`

The controller assigns it from a monotonically increasing counter. Protocol messages should identify requests by `RequestId`, not by slot indices or by aligning multiple parallel vectors.

### Command Protocol

The protocol should stay coarse-grained and step-oriented. The primary commands are:

- `RunPrefillStep`
- `RunDecodeStep`
- `RunUnifiedStep`
- `DropRequest`
- `Shutdown`

We explicitly prefer this over exposing low-level commands such as `EnsureCapacity`, `Advance`, or `Reset` as first-class protocol messages. Internal KV mutation details should remain implementation details of one step whenever possible.

For step payloads, the command shape should stay request-oriented:

- `RunPrefillStep { requests: Vec<PrefillStepItem>, echo: bool }`
- `RunDecodeStep { requests: Vec<DecodeStepItem> }`
- `RunUnifiedStep { prefill: Vec<PrefillStepItem>, decode: Vec<DecodeStepItem> }`

At minimum:

- `PrefillStepItem` contains `request_id` and prompt tokens
- `DecodeStepItem` contains `request_id` and the decode token

### Synchronization Rule

The runtime should obey one simple rule:

- no worker starts command `N + 1` until all workers have finished command `N`

This gives deterministic ordering, simpler failure handling, and avoids reintroducing cross-thread mutable-borrow coupling through the side door.

### Execution Shape

Qwen3 should keep one executor shape:

- one `Qwen3Executor`
- one primary local lane
- zero or more additional rank workers

Under this model:

- single-GPU execution is the `tp_size == 1` case
- tensor parallel execution is the `tp_size > 1` case

The controller-side protocol should not split into unrelated single-GPU and TP command families. Both modes should use the same coarse-grained `StepCommand`, with broadcast fanout degenerating naturally to the single-worker case.

### Result Ownership

For now, result flow should stay asymmetric:

- non-primary workers return acknowledgement or step failure only
- the primary worker returns the step artifacts needed by the controller

This keeps workers responsible for local execution while the controller remains responsible for resolving execution artifacts into scheduler-visible effects.

### Sampling Ownership

Sampling is split into two responsibilities:

- the controller owns sampling policy and random input generation
- the primary worker executes GPU sampling and logprob extraction

Concretely, `SamplingParams` and per-step random values travel with the step items, worker threads run the GPU sampling path, and the controller consumes CPU-visible step artifacts after execution.

### Request Destruction

Request destruction should remain an explicit protocol action:

- `DropRequest { request_id }`

This keeps lifecycle transitions visible at the command layer instead of hiding them inside unrelated step commands.

### Near-Term Cleanup Direction

For the next cleanup passes, we should bias toward:

- worker-owned rank-local state
- broadcast command protocol
- explicit barrier semantics
- request-oriented payloads

We should bias away from:

- controller ownership of rank-local execution objects
- command payloads that smuggle `&mut` semantics through raw pointers
- designs that depend on multiple parallel vectors and positional alignment to identify one logical request

Without these boundaries, those changes accumulate in the same scheduler functions and the runtime becomes harder to extend in predictable ways.

### Scheduler-Local Result Resolution

Not all logic currently living in the scheduler should move into the executor.

There is a third layer that should remain scheduler-local, but should not stay inline inside the scheduler loop.

That layer is step-result resolution.

It should be responsible for:

- first-token handling after prefill or unified execution
- decode-token handling after decode or unified execution
- logprob assembly
- prompt-echo assembly
- EOS / max-length / consumer-drop retirement decisions
- promotion of newly-prefilled requests into the active set

This logic is scheduler policy, not model execution.

So the right direction is:

- keep execution in `ModelExecutor`
- keep request lifecycle ownership in the scheduler
- move step-result interpretation into a scheduler-local resolver layer

### Next Refactor Shape

The next step after introducing `ModelExecutor` should be to restructure the scheduler around these boundaries:

1. build an `ExecutionPlan`
2. execute it to produce `ExecutionArtifacts`
3. resolve those artifacts into `StepEffects`
4. apply those effects to scheduler-owned state

Conceptually:

```rust
loop {
    let plan = build_next_plan(...);
    let artifacts = executor.execute(plan)?;
    let effects = resolve_step(plan, artifacts, ...);
    apply_effects(effects, ...);
}
```

The primary value of this refactor is responsibility isolation, not reducing the line count of `scheduler.rs` by itself.

Current status after the latest TP cleanup:

- the scheduler no longer runs Qwen3 GPU execution directly
- `tp_size == 1` and `tp_size > 1` both go through the same coarse-grained step protocol
- rank 0 execution now happens on a primary worker thread rather than on the scheduler thread
- scheduler-side work is limited to planning, command submission, and effect application
- sampling policy remains controller-owned, but sampling execution and logprob extraction now run on the worker side

### Rust Sketch

The following sketch is intentionally narrow. It is meant to capture the boundary, not to freeze the final implementation.

```rust
use anyhow::Result;

use crate::kv_pool::KvState;
use crate::tensor::DeviceVec;

pub struct Qwen3RequestState {
    pub kv: KvState,
}

pub enum BatchKind {
    Prefill,
    Decode,
    Unified,
}

pub struct PrefillItem<'a> {
    pub prompt_tokens: &'a [u32],
    pub state: &'a mut Qwen3RequestState,
}

pub struct DecodeItem<'a> {
    pub token: u32,
    pub state: &'a mut Qwen3RequestState,
}

pub struct BatchSpec<'a> {
    pub kind: BatchKind,
    pub prefills: &'a mut [PrefillItem<'a>],
    pub decodes: &'a mut [DecodeItem<'a>],
}

pub struct BatchResult {
    pub prefill_logits: Vec<DeviceVec>,
    pub decode_logits: Vec<DeviceVec>,
}

pub trait ModelExecutor: Send {
    fn execute_batch(&mut self, spec: BatchSpec<'_>) -> Result<BatchResult>;
}
```

The exact output representation may later need adjustment.

In particular, decode output should be allowed to preserve a batched representation if that is important for throughput.

What matters here is the ownership and control boundary, not freezing `Vec<DeviceVec>` as the permanent decode API.

### Design Intent

The intent of this interface is:

- scheduler remains the control plane
- executor becomes the execution plane
- TP is hidden inside the executor, not leaked into scheduler logic
- `KvPool` and request admission stay in the scheduler

This keeps the future shape clean:

- `SingleGpuQwen3Executor`
- `TensorParallelQwen3Executor`

Both should be able to sit behind the same scheduler-facing interface.

### Relationship To DP

This abstraction is the right carrier for model-internal parallelism such as TP and later EP.

It is not the right abstraction for service-layer data parallelism across multiple model replicas.

If pegainfer later needs multiple model replicas, that should live above the executor layer. A `ModelExecutor` still represents one logical model replica, even if that replica internally spans multiple GPUs.

## What Success Looks Like

The milestone is successful when all of the following are true:

- `Qwen3-4B` runs correctly with `TP=2`
- existing single-GPU behavior remains intact with `TP=1`
- the runtime can be reasonably extended later toward larger dense models and MoE-related model parallelism
- the resulting design is understandable and does not create a one-off parallel path that the rest of the codebase must work around

## Acceptance Criteria

Primary acceptance criteria:

- `Qwen3-4B` under `TP=2` passes the existing end-to-end test suite, or passes an equivalent minimally adjusted `e2e` path if the test harness needs TP-aware setup
- generated outputs under `TP=2` match the `TP=1` baseline for the covered `e2e` cases
- `TP=1` continues to pass its existing `e2e` coverage
- runtime stability is acceptable: no hangs, no cross-device state corruption, no obvious lifecycle failures during model load or generation

## Current State And Remaining Issues

At this point, the implementation meets the basic smoke-test bar for `Qwen3-4B TP=2`:

- model load succeeds on two GPUs
- requests complete end-to-end through the existing OpenAI-compatible HTTP path
- generated outputs are sensible and clearly non-degenerate

The current implementation has also passed a narrower `TP=8` smoke test on an 8x4090 machine:

- eight-way weight load succeeds
- the server reaches `Scheduler ready` and starts listening
- simple completion requests return non-degenerate text

That means the executor and sharding path are no longer merely `TP=2`-shaped, but `TP=8` should still be treated as an experimental validated configuration rather than a fully qualified support target.

However, a few important engineering issues still remain open:

- TP-vs-TP=1 exact parity is still not fully settled, but the old decode-state corruption bug is no longer the main blocker
- embedding and `lm_head` are still replicated by design in this first pass
- some of the runtime fixes are pragmatic bring-up fixes rather than final abstractions, especially around thread-scoped CUDA runtime / cuBLAS setup and teardown

The next practical steps should be:

- keep the current TP path stable and avoid reopening the earlier decode append bug
- further unify the `tp=1` and `tp>1` scheduler / executor flow now that both paths are real and runnable
- keep the TP decode-graph startup pre-capture (every reachable bucket graph is captured in lock-step at init, so TP `batch_decode` steps are pure replay; mixed prefill+decode steps stay on the eager unified path) aligned with any bucket/attention-path changes
- then revisit vocab-side replication only after the execution and correctness story is stable

So the right reading of the current status is:

- the architecture direction is validated
- the TP path is real and runnable
- the implementation is still a first-pass runtime bring-up, not the final production shape
- TP CUDA Graph is now served via startup pre-capture (#498): every reachable (bucket, attention-path) decode graph is captured per rank at init while the ranks are trivially in lock-step, after every rank worker has spawned — uniform captured-ness across ranks is an invariant established at construction, not a runtime property
- the sweep is two-phase per bucket (dual-GPU measured requirement, not a nicety): every rank records + instantiates + uploads the bucket's graph before any rank launches it, because a launched captured collective spins on its peers while capture/instantiate/upload contend process-wide driver locks — overlapping the two deadlocks the driver; for the same reason the TP memory profile's decode exercise runs eager, and the NCCL warmup runs one eager all-reduce per bucket size (connections are established per size-selected algorithm)

The core bar for this milestone is straightforward:

- `TP=2` for `Qwen3-4B` must be real, correct, and regression-safe

## Out Of Scope Questions

The following questions are intentionally deferred until after the first TP milestone is proven:

- how far to push throughput optimization in the first TP implementation
- when to shard vocab-facing weights instead of replicating them
- whether later multi-GPU support should expand first toward larger dense models, MoE, or broader serving topology work

## Summary

This milestone should stay disciplined.

The job is not to build a full distributed inference platform in one step. The job is to make pegainfer capable of correct `TP=2` execution for `Qwen3-4B`, while establishing the architectural boundary that future large dense and MoE work can build on.
