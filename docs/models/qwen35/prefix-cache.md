# Qwen3.5-4B prefix cache

> **TL;DR:** Qwen3.5-4B uses a GPU-only, content-hashed joint prefix cache: full-attention KV is reusable only with a matching complete recurrent/conv snapshot at the same 256-token boundary, otherwise the request is cold. KV stays in kvbm's reclaimable inactive pool, while `SnapshotCache` owns snapshot lookup, pinning, LRU, and publication. TP1/TP2 correctness and serving tests pass, and a 4,160-token shared prefix cuts warm TTFT by 94.3% (TP1) / 95.1% (TP2).
>
> **Last touched:** 2026-09

## Preparation

- **Read**:
  - `docs/index.md` - identified the current Qwen3.5 roadmap and the related Qwen3 cache and scheduler docs.
  - `docs/models/qwen35/roadmap.md` - direct-paged writes and bounded chunked prefill are complete; issue #257 needs a joint KV/recurrent/conv cache.
  - Maintainer RFC discussion for issue #257 - narrowed the first version to a GPU snapshot cache with one consistency rule for KV and recurrent state.
  - `docs/models/qwen3/prefix-cache.md` - provides the existing rules for block hashes, adapter isolation, final-token recompute, and keeping matched KV alive.
  - `docs/subsystems/runtime/qwen3-kvbm-integration-spec.md` - describes the content-hashed `BlockPool`/`RequestKv` cache adopted by Qwen3.5.
  - `pegainfer-qwen35/src/{scheduler.rs,prefill.rs,recurrent_state.rs,weights.rs}` and `pegainfer-kv-cache/src/pool.rs` - confirmed request-state flow, snapshot layout, exact-boundary KV operations, and GPU memory reservation.
- **Relevant history**:
  - The first draft focused on CPU offload but did not say clearly who keeps KV and snapshots consistent. Review narrowed the first version to GPU allocation, lookup, lifetime, and whole-model snapshot creation.
  - The first draft also treated the 64-token GDR tile as a correctness boundary. Resumed prefill works with smaller scheduler chunks, so the safe boundary is a completed whole-model window, not an internal GDR tile.
- **Plan**:
  1. Add exact-boundary, non-mutating KV probe and attach support.
  2. Unify Qwen3.5 execution around `KvCacheManager`/`RequestKv` transactions before enabling reuse.
  3. Add a fixed-budget recurrent/conv snapshot pool, joint restore/publication, TP coordination, observability, and validation.
- **Risks / open questions**:
  - Snapshot copy and cold-prefill costs vary by model shape. The initial 256-token interval is a starting policy, not a proven optimum.
  - TP must publish and restore a snapshot only when every rank reports the same token boundary.

## Decisions

The first version is deliberately narrow:

- GPU-only recurrent snapshot allocator with a fixed load-time byte budget.
- One complete snapshot contains all linear layers' f32 GDR state, bf16 conv state, and the token position.
- Snapshots are eligible every 256 prompt tokens. Non-aligned request ends apply normally but do not create snapshots.
- A reusable boundary must have both registered full-attention KV and a GPU-resident recurrent snapshot for the same token prefix.
- `Qwen35PrefixCache` is the only interface used by the scheduler for joint prefix creation and restore.
- Restored state is copied into the request's own `RecurrentState`; active requests never modify or directly share a cache slot.
- Echo requests are rejected before cache lookup because cached positions would not produce their required logits.

CPU offload, a second LRU tier, snapshot compression, request-end snapshots, and cross-process transfer are deferred. They must keep the same rule that KV and recurrent state are restored together.

## Implemented state flow

The scheduler/executor now owns content-hashed request KV and recurrent state together:

```rust
enum PrefillBackendState {
    Single {
        kv: Box<RequestKv>,
        rec: RecurrentState,
    },
    // TP workers address controller-owned RequestKv by request id.
    Tp { request_id: RequestId },
}
```

`RequestKv::schedule_prefill` reserves pages and produces an immutable `KvView`. Prefill and decode kernels write through that view without changing logical KV. After the whole-model call succeeds, the scheduler applies the KV transaction and may publish a recurrent snapshot. After each successful prefill window:

```text
kv.kv_position() == rec.seq_len == cursor + step_chunk
```

Single-GPU serving, TP serving, and the low-level accuracy executor all use `KvCacheManager`/`RequestKv`, even when the snapshot budget is zero. With zero budget, release resets registered blocks so the disabled mode remains a true cold control.

Joint lookup adds these exact-boundary operations to the shared cache:

1. Probe the longest contiguous registered KV prefix without changing the new request.
2. Keep the probed KV blocks pinned while `SnapshotCache` is checked.
3. Expose complete KV-block boundaries and their canonical `SequenceHash` values in descending order.
4. Attach only the boundary with a matching snapshot; ignore any longer KV-only tail.
5. Advance the new request's KV position as part of the same attachment.
6. Keep at least one prompt token uncached so prefill can produce the first generated token.

`TokenEvent::Scheduled.cached_tokens` reports the selected joint boundary. A KV-only tail is never reported as a hit.

## Valid cache hit

Qwen3.5 stores KV and recurrent snapshots separately, but a request may reuse a boundary `N` only when all of the following are true:

1. Full-attention KV for `[0, N)` is registered and still on GPU.
2. A complete recurrent snapshot for `[0, N)` is still on GPU.
3. Both use the same `SequenceHash`, which includes token lineage and adapter/LoRA salt.
4. The KV position, snapshot position, recurrent `seq_len`, and scheduler cursor all equal `N`.

KV without a matching snapshot is not a Qwen3.5 prefix hit. A snapshot without matching KV is also not a hit. The scheduler never sees either one as partial reuse.

## Snapshot interval

A snapshot boundary is the token position after a scheduled prefill window has completed all model layers. The current interval is:

```text
SNAPSHOT_STRIDE_TOKENS = 256
```

The stride is a multiple of the 16-token KV block size, so every snapshot key references the lineage hash of a complete registered KV block. The scheduler clamps each request's next window to the next snapshot boundary. With a 900-token prompt, the resulting positions are `256 -> 512 -> 768 -> 900`; only the first three are snapshot candidates.

The GDR implementation internally tiles work in 64-token chunks, but this is not a snapshot correctness constraint. It handles a partial final tile and applies recurrent state for arbitrary positive sequence lengths. The invariant is “after a successful whole-model window,” not `position % 64 == 0`.

The 256-token interval limits how many large snapshots one prompt creates. Prefixes shorter than 256 tokens intentionally remain cold. Any later interval must still align to complete KV blocks.

## Snapshot contents and capacity

Each slot uses the same device layout as request-local `RecurrentState`:

- for every linear layer, `state: [num_value_heads, key_head_dim, value_head_dim]` f32;
- for every linear layer, `conv_state: [linear_attn_qkv_dim, conv_kernel_dim - 1]` bf16;
- host metadata recording the exact `seq_len` represented by the slot.

Capacity is derived from `recurrent_state::bytes_per_request(config)`, not a hard-coded model label. For Qwen3.5-4B, one slot is:

```text
per-layer GDR state = 32 * 128 * 128 * 4             = 2,097,152 bytes
per-layer conv      = 8,192 * (4 - 1) * 2            =    49,152 bytes
all 24 layers       = 24 * (2,097,152 + 49,152)      = 51,511,296 bytes
                                                        49.1 MiB
```

The allocator reserves a fixed number of whole slots at model load:

```rust
let bytes_per_slot = bytes_per_request(config);
let max_slots = snapshot_budget_bytes / bytes_per_slot;
```

Snapshot bytes are reserved before KV capacity is finalized, so snapshot allocation cannot consume memory that admission assumes belongs to KV. Zero configured MiB disables prefix reuse. A positive budget smaller than one complete slot is rejected instead of silently acting disabled.

Under TP the configured budget applies to each rank. Every rank must allocate the same number of physical slots.

## Cache ownership and pinning

`Qwen35PrefixCache` owns the full-attention KV manager and the joint-entry directory. Physical snapshot stores are separate so TP can use one metadata decision with identical slot numbers on every rank:

```rust
struct Qwen35PrefixCache {
    kv: KvCacheManager,
    snapshots: SnapshotCache,
    stats: PrefixCacheStats,
}

struct SnapshotCache {
    entries: HashMap<PrefixBoundaryKey, Arc<SnapshotEntry>>,
    free_slots: Vec<usize>,
    slot_count: usize,
    clock: u64,
}

struct SnapshotEntry {
    recurrent_slot: usize,
    last_used: AtomicU64,
}

struct SnapshotGuard {
    boundary: usize,
    entry: Arc<SnapshotEntry>,
}

struct SnapshotReservation {
    key: PrefixBoundaryKey,
    recurrent_slot: usize,
    replaced: bool,
}

struct RecurrentStateStore {
    slots: Vec<RecurrentState>,
}

struct PrefixBoundaryKey {
    sequence_hash: [u8; 16],
    boundary_tokens: usize,
}
```

`sequence_hash` is the canonical hash returned by the KV cache for the block ending at `boundary_tokens`. It already includes earlier block lineage and adapter salt. Storing `boundary_tokens` explicitly prevents reuse at the wrong token position.

`SnapshotEntry` owns one recurrent-state slot; a `SnapshotGuard` pins it by holding another `Arc`. `BlockPool` owns KV, with `PrefixProbe` and the request-local attachment keeping matched blocks alive through lookup and attach. TP publishes only after every worker saves the same slot at the same boundary, and reports a hit only after every worker confirms the restored boundary.

## Creating a snapshot

A snapshot is created after a whole-model window, not inside per-layer GDR scratch:

1. Clamp the scheduled window to the next 256-token boundary or prompt end.
2. Reserve KV blocks with `RequestKv::schedule_prefill`.
3. Run the full model, updating full-attention KV and request-local recurrent/conv state.
4. On failure, revert the KV schedule and publish nothing.
5. On success, apply KV with `apply_prefill_chunk` or final `apply_prefill`.
6. Verify that KV position and `rec.seq_len` equal the candidate boundary.
7. At an eligible boundary, call `reserve_snapshot`; if it returns a reservation, copy the complete `RecurrentState` into that slot.
8. Publish the `SnapshotEntry` after all rank-local copies succeed; abort the reservation on failure.

The GDR `chunk_state` scratch is per linear-layer call and cannot represent a whole-model snapshot. Only request-local `RecurrentState` after all layers finish contains the complete recurrent/conv state required for publication.

Running out of snapshot slots is a soft cache event: skip insertion and continue the request. A CUDA copy failure is an execution error and publishes no key. A duplicate key refreshes LRU without copying another immutable snapshot. If replacement copying fails, the reserved slot returns to the free list unpublished.

## Lookup and restore

Restore is two-phase so one directory can coordinate one or many physical ranks:

```rust
let (request_kv, restore) = prefix_cache.begin_request(...)?;
// Restore restore.recurrent_slot() on every physical rank.
let cached_tokens = prefix_cache.finish_restore(
    &request_kv,
    restore,
    &rank_positions,
)?;
```

`begin_request` performs the logical lookup and KV attach:

1. Probe the longest registered KV prefix while keeping candidate blocks alive.
2. Enumerate eligible 256-token boundaries from longest to shortest, leaving at least one prompt token to run.
3. Build `PrefixBoundaryKey` from the canonical `SequenceHash` and token position.
4. Pin the corresponding snapshot slot.
5. Select the first boundary with both resources; if none exists, return `0` without changing request state.
6. Attach exactly that KV boundary to request-local `RequestKv`.

The single-GPU or TP executor then restores `recurrent_slot` into the request-local state. `finish_restore` verifies all positions, records the hit, and releases the pin.

For example, a 768-token KV match with snapshots at 256 and 512 restores 512 tokens. The scheduler never receives “KV hit 768, snapshot hit 512” as separate facts.

If physical restore or position validation fails after KV attachment, request preparation fails and releases the prepared state. It is not treated as a normal cache miss.

After restore, suffix prefill operates on `tokens[cached_tokens..]`. When prefill completes, recurrent state is promoted into the normal decode state. Decode continues to schedule, forward, and apply KV one token at a time; it does not perform another prefix lookup.

## Lifetime and eviction

`Qwen35PrefixCache` owns recurrent snapshots; `BlockPool` owns KV lifetime:

- With caching enabled, released KV enters kvbm's inactive pool, where it remains reusable and counts toward `available_blocks()`.
- With caching disabled, released KV returns directly to the free pool.
- Snapshot slots are immutable while indexed and can be evicted only when no `SnapshotGuard` exists.
- `SnapshotGuard` protects a snapshot until D2D restore and position checks complete.
- Restored suffix prefill and decode mutate request-owned state, never the cached slot.
- If no free or unpinned snapshot slot exists, insertion is skipped rather than blocking or failing the request.

Snapshot eviction reuses its recurrent-state slot; KV reclamation remains independent. An aborted replacement returns the unpublished slot to the free list. Lookup can continue to the next shorter published boundary.

LRU selects an unpinned snapshot victim. Correctness depends on pinning and joint validation, not on LRU ordering.

## TP behavior

TP uses one logical cache decision and rank-local physical storage:

1. The controller owns the only `KvCacheManager`, `RequestKv` map, `SnapshotCache`, and LRU state.
2. Startup validates compatible KV geometry and snapshot-slot counts across ranks.
3. Logical capacity is capped by the smallest rank-local physical capacity.
4. The controller broadcasts identical `KvView` page IDs; every worker writes its local KV shard into its own `KvBuffer`.
5. Snapshot insertion reserves one common slot, saves it on every rank, verifies all returned positions, and only then publishes the key.
6. Restore uses the same slot on every rank and reports a hit only after every worker confirms the boundary.

This keeps admission, attachment, and eviction deterministic across ranks while leaving recurrent/conv tensors local to each GPU.

## Correctness rules

- `RequestKv::kv_position() == RecurrentState::seq_len == PrefixBoundaryKey::boundary_tokens` after insertion and restore.
- A snapshot contains both state tensors for every linear layer; GDR-only or conv-only snapshots are invalid.
- Snapshot contents are immutable after publication.
- A prefix hit always leaves at least one prompt token uncached so final prefill can emit the first generated token.
- Echo requests are rejected before prefix matching.
- Allocation pressure changes hit rate only, not request output.
- Failed forward or snapshot insertion never publishes a cache key.
- A KV-only or snapshot-only boundary is never reported as cached tokens.
- A snapshot may outlive its KV blocks; lookup treats missing KV as a cache miss.
- Disabling the feature preserves cold-serving behavior.

## Implementation order

1. Move Qwen3.5 KV management to `KvCacheManager` and `RequestKv`.
2. Add fixed-budget recurrent snapshot storage and joint prefix entry management.
3. Publish snapshots at committed 256-token prefill boundaries.
4. Add joint KV/snapshot lookup, restore, pinning, and LRU eviction for TP1 and TP.
5. Validate correctness, serving behavior, and cold/warm performance before default enablement.

## Validation

The implementation acceptance surface should include:

- cache-management tests for snapshot keys, shorter-boundary fallback, duplicate insertion, pinning, and unpinned eviction;
- scheduler tests for prompts below 256, exactly on boundaries, across multiple boundaries, and with non-aligned tails;
- availability tests for KV-only, snapshot-only, and shorter valid fallback;
- adapter salt isolation;
- mixed cold and warm requests in the same prefill/unified step;
- pool-full behavior proving insertion skip preserves cold output;
- real GPU cold-vs-warm HF logits gates, including resumed suffix prefill and decode-slot promotion;
- retained cache metrics for joint hits, hit length, misses, insertions, evictions, and slot occupancy.

The cold/warm measurements below guide future stride changes. They must not be replaced by the old RTX 4090 CPU-transfer estimates, which measured a deferred design and a different snapshot shape.

## Performance Result (2026-08-06)

- **Environment:** local Qwen3.5-4B, RTX 4090s only (GPU 1 for TP1; GPUs 1/2 for TP2). TP1 used CUDA Graphs; TP2 used `--cuda-graph=false`.

Qwen3.5 reuses the largest 256-token boundary strictly below the prompt length. This table compares cold and warm TTFT p50 for single-token generation (`cache off -> cache on`).

| Prompt tokens | Cached tokens | TP1 | TP1 reduction | TP2 | TP2 reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 320 | 256 | 29.22 -> 15.56 | 46.7% | 46.18 -> 18.75 | 59.4% |
| 576 | 512 | 46.85 -> 16.92 | 63.9% | 72.97 -> 18.82 | 74.2% |
| 1,088 | 1,024 | 88.75 -> 15.93 | 82.1% | 126.14 -> 19.36 | 84.7% |
| 2,112 | 2,048 | 161.00 -> 16.40 | 89.8% | 230.07 -> 19.82 | 91.4% |
| 4,160 | 4,096 | 308.64 -> 17.71 | 94.3% | 439.37 -> 21.68 | 95.1% |

This table shows how the same cache hit affects TTFT and end-to-end latency when generating 128 tokens; decode dominates the remaining latency.

| Prompt tokens | TP1 TTFT p50 off -> on | TP1 E2E p50 off -> on | TP2 TTFT p50 off -> on | TP2 E2E p50 off -> on |
| ---: | ---: | ---: | ---: | ---: |
| 1,088 | 94.58 -> 24.13 | 1,542.02 -> 1,452.78 | 127.05 -> 20.38 | 1,445.78 -> 1,324.97 |
| 2,112 | 166.25 -> 24.23 | 1,700.09 -> 1,535.84 | 231.56 -> 21.89 | 1,640.17 -> 1,421.95 |
| 4,160 | 313.51 -> 24.30 | 2,006.08 -> 1,696.04 | 442.01 -> 24.88 | 2,036.57 -> 1,606.08 |

This table summarizes behavior under concurrency, mixed load, and decode batching:

| Additional workload | Result |
| --- | --- |
| TP1 concurrency, 2,112 prompt + 128 output | At concurrency 1/4/8: TTFT p50 `24.67/94.41/152.25 ms`; request throughput `83.11/75.87/69.04 tok/s`; every request hit 2,048 cached tokens. |
| TP1 mixed load | Baseline ITL p50/p99 `12.03/12.25 ms`; mixed ITL `12.03/36.12 ms`. After initial insertion, 4,096-token injections hit 3,840 tokens with `41.42--51.00 ms` prefill and no warnings. |
| Decode TP1 vs TP2 | At context 4,096/batch 1, TP1/TP2 TPOT is `13.08/12.47 ms`; at batch 4 it is `13.74/48.76 ms`. TP2 batch decode needs a separate runtime optimization pass; this run disables CUDA Graphs. |

**Conclusion**

- **Core performance gains**
  - Warm TTFT improves with prefix length: at 4,160 prompt tokens it falls by 94.3% on TP1 and 95.1% on TP2.
  - For 128-token outputs, the same long prompt reduces E2E latency by 15.4% on TP1 and 21.1% on TP2; steady decode TPOT is effectively unchanged.
  - The benefit remains under TP1 concurrency and mixed load; warm injections hit 3,840 tokens with 41--51 ms prefill and no warnings.
  - TP1 HTTP warm TTFT remains about 19--29 ms for 320--4,160-token prompts with cache enabled.
- **Follow-up**
  - TP2 batch-4 decode has high TPOT while CUDA Graphs are disabled; profile and optimize this runtime path separately from prefix cache.


## Deferred work

- pinned-CPU snapshot offload and two-tier LRU;
- request-end snapshots for non-aligned prompt lengths;
- workload-adaptive or per-model snapshot stride;
- snapshot compression or reduced-precision state;
- cross-worker/P-D transfer of hybrid state;
- sharing snapshot infrastructure across other hybrid model lines.

Each extension must preserve the same logical rule: one reusable boundary restores all model state at one token position.
