# Gemma 4 KV cache contract

**TL;DR:** Gemma 4 caches at two head dims — 256 sliding, 512 full attention — and the two families
differ in lifetime: the sliding one wants to release pages the other still reads. The backends and
fused pool-write preps have landed, pinning two coordinate systems (absolute RoPE position vs
cache-relative slot; the never-conflate invariant is frozen, not a wire format). Storage is
declared along orthogonal axes — lifecycle, paged/slab live storage, reference/copy checkpoint
materialization — with observable semantics frozen rather than a closed pairing. Serving stages in
two steps: one unreclaimed pool first, the paged window with batching and long context. Checkpoint
hits freeze one query — the largest position at which every required family is exactly recoverable
— and a global prefix without its local window is worth nothing. Four behaviors hold throughout:
transactional families, complete hits only, exact admission accounting, fail-closed
prefix/offload/P-D (a default-off engine-internal prefix cache admitted by amendment).

Last touched: 2026-08

## Two geometries, two lifetimes

`KvLayout` applies one `num_kv_heads` and one `head_dim` to every layer, and `KvCacheManager` pairs
one such buffer with one `BlockPool`, so Gemma 4 cannot use that facade. Owning buffers outside it
is routine: `BlockPool` holds no GPU memory, and glm52's arenas register one width per layer plus a
narrower one on a layer subset. Copy that shape. What is new is that the two groups want different
lifetimes — the sliding group to release pages the full-attention group still reads.

Per-token cost is `layers × kv_heads × head_dim × 2 (K and V) × 2 (bf16)`, from the published
configs:

| | 12B | 26B-A4B | 31B |
| --- | --- | --- | --- |
| layers, sliding / full | 40 / 8 | 25 / 5 | 50 / 10 |
| sliding, KV heads × head_dim | 8 × 256 | 8 × 256 | 16 × 256 |
| full, KV heads × head_dim | 1 × 512 | 2 × 512 | 4 × 512 |
| sliding, per token | 320 KiB | 200 KiB | 800 KiB |
| full, per token | 16 KiB | 20 KiB | 80 KiB |
| sliding, one 1024-token window | 320 MiB | 200 MiB | 800 MiB |
| full, at 262144 positions | 4 GiB | 5 GiB | 20 GiB |

The sliding group is 20× the full group per token at 12B. That ratio, not the head dims, drives the
sizing decisions here — and it flips with context: the sliding group is bounded by the window while
the full group grows linearly, so past roughly 20k positions the full group is the larger term.
Full-attention layers sit at indices 5, 11, 17, …, which the model probe expresses as
`i % 6 == 5 || i == last_index`; all three published layer counts are divisible by 6, so the
last-layer clause never fires on them.

## The attention backends and the fused preps — landed

**The sliding-window instantiation.** Sliding layers must not attend past 1024 positions, so this
was a correctness prerequisite regardless of memory, and its conversion is part of the contract:
**`flashinfer_window_left = config.sliding_window - 1`**, i.e. 1023. Gemma's mask keeps
`kv_idx > q_idx - sliding_window`, so 1024 means the current token plus 1023 predecessors;
FlashInfer's `window_left` is an inclusive distance — `W` means the last `W + 1` KV entries.
Passing the config value straight
through attends to one token too many, and an off-by-one here changes output without failing
anything — the distinction between a 1024- and a 1025-token window needs its own gate.

**Paged attention at head_dim 512.** Each kernel family is instantiated at a compile-time width, so
Gemma 4 needed an HD512 backend covering single prefill, paged batch prefill and paged batch
decode. One property is load-bearing: FlashInfer's decode dispatcher instantiates a fixed set of
GQA groups, and at TP1 the 12B full-attention group — 16 query heads over one KV head — is outside
it, so that group's decode rides the paged-prefill kernel instead (a dispatch-coverage gap with a
working path, not a correctness bug). The ratio is **per rank, not per config**: query heads shard
with the world size while a single KV head can only be replicated, so the same group is 16 at TP1,
8 at TP2, 4 at TP4, and lands back inside the compiled decode set above TP1. Backend selection, GQA
validation and graph capture must all key off the resolved per-rank mapping.

**The fused pool-write preps.** The write side landed as Gemma-specific prep entries that fuse
normalisation, rotation and the pool scatter: the hd256 serving form writes Q contiguous while
scattering normed+rotated K and weightless-normed V straight into the layer's pool blocks, and the
hd512 prep writes the K=V fork's V block alongside K in the same pass — one read of the raw K, no
D2D fork copy, no intermediate scatter. Three of their properties are contracts:

- **The slot contract.** Slots derive from `start_pos + token` — the absolute position — and
  absolute and cache-relative coordinates coincide **only below the sliding window**. The wrappers
  cannot detect a violation, because below the window there is nothing to detect; a window-crossing
  caller must map request positions to pool slots *before* the prep call — the coordinate section
  below owns that mapping.
- **The layout is never trusted.** `PagedKvLayout` carries public fields, so every prep and
  windowed attention wrapper re-derives the block/stride chain with overflow-checked arithmetic and
  rejects a layout that disagrees with itself; both K and V offsets come from that one derivation,
  which also closes the aliasing footgun noted under K=V below.
- **Defence splits by what the host can see.** Everything host-visible fails as `Err`; page ids are
  device data, so for them the kernel's `__trap()` is the only check — host validation would cost a
  D2H synchronization per launch. A trap rather than a clamp is deliberate: a clamped page id turns
  an addressing bug into silently wrong numerics, the trap into a dead context at the fault site.

## Two coordinate systems, first-class

Once the sliding window's page table is truncated to a trailing suffix, every token has two
coordinates, and the state must carry both rather than derive one from the other:

- **`absolute_position`** — what RoPE consumes; grows without bound.
- **cache-relative slot position** — what the paged scatter consumes; wraps within the window.

They coincide only while every page row starts at position 0; conflated, a scatter at position
5000 indexes page 312 of a 65-entry row — an out-of-bounds **write**. Request state therefore
carries `absolute_position` plus the local family's `resident_origin`/`resident_len`, and what is
frozen is the **invariant**, not a wire format: the two meanings never mix, and every
cache-relative slot comes through the resident state. For contiguous tokens over a page-aligned
origin — the only case that exists — the kernels derive the slot from two scalars,
`row = pos / page_size - page_origin`; a per-token slot array becomes necessary only if a
non-contiguous scatter ever does.

Dual-family scheduling is transactional (frozen behavior 1): a step schedules on both families or
reverts both, and a request whose apply succeeded on one family and failed on the other is a fatal
state corruption — the two positions can never be reconciled afterwards.

## Capacity: what reclaiming the window buys

All figures are per rank at TP1; the tensor-parallelism section says how they scale.

With the backends landed and no reclamation, pages stay resident and the sliding group costs its
full 320 KiB per token. That is correct, and budget-limited rather than window-limited: one 12B
request costs 336 KiB per token across both groups, so a 20 GiB budget reaches roughly 62k
positions for one request, or 7.8k each at eight concurrent. The declared context is what it cannot
reach — at 262144 positions one unreclaimed request wants **84 GiB** (55 GiB at 26B, 220 GiB at
31B), while capping the sliding group at its residency drops it to about 4.6 GiB. The trade is
"cannot serve one at the declared maximum" versus "serves four": a mask-only engine is shippable
with a budget-derived context ceiling.

**Sliding residency is a window plus a shared prefill burst.** Prefill is append-then-attend, so a
request forwarding a chunk of `C` tokens needs `C + window` resident, not `window`. But the prefill
budget caps the **step's total** forwarded tokens across all requests, so the burst is bought once
for the whole step, and the pool decomposes:

```
sliding pages = concurrency × (ceil(window / page_size) + 1)   steady residency, per request
              + ceil(max_prefill_tokens / page_size)           in-flight burst, shared by the step
              + (concurrency - 1)                              per-row chunk rounding
```

The third term is not optional: every active row rounds its own frontier, so summing rounded
chunks exceeds rounding the sum — eight rows taking `[1003, 3 × 7]` at page size 16 need 590 pages
where the first two terms give 584, and the failure mode is admission accepting a request the pool
cannot schedule (hence frozen behavior 3: per-request frontiers and chunks, never window-only). At
12B, page size 16, `max_prefill_tokens` 1024: 129 pages (645 MiB) peak for one request, 591 pages
(2.89 GiB) for eight — against 1032 if the burst were charged per request.

Reclamation granularity is one page, which does **not** require the page size to divide the window:
the window's left edge lands mid-page at almost every position regardless. The mechanism is to
retain the frontier page whole and let the mask exclude the expired tokens inside it — what
`window_left` does and what FlashInfer's page-count arithmetic assumes — at a cost of at most
`page_size - 1` over-retained tokens per request, which the `+ 1` above carries.

## Storage is a declared dimension

Each KV family declares its shape along three axes, of which the first is independent and the
last two are orthogonal to each other:

- **lifecycle**: `AppendOnly`, or `Sliding { window }` — the active token domain is `[0, t)` or
  `[t - W, t)`.
- **live storage**: paged (front-releasable page rows, shareable while live under refcounts) or
  slab (in-place rewrite, unshareable while live).
- **checkpoint materialization and restore**: by *reference* (pages seal immutable when full, for
  free; a checkpoint is a set of page references; restore assembles a row by borrowing) or by
  *copy* (a paid D2D snapshot into cache-owned pages or sealed artifacts; restore copies into
  request-private state).

The last two axes are genuinely independent — three combinations are real today. Paged live with
reference checkpoints is the paged window's end state. Slab live with copy checkpoints is the shape
in-place linear states force. And paged live with **copy** checkpoints — snapshotting D2D into
cache-owned reservations and restoring by copy — is what the engine-internal prefix cache already
runs. What is frozen is the observable semantics (complete hits only, invalidate before release,
seal cost and live shareability stated per combination), not a closed enum.

Gemma 4's local family choice — ring slab versus paged window — is the live-storage axis made
explicit, with reference checkpoints as the paged arm's destination. Serving stages in two steps:
**first, one shared pool with nothing reclaimed** — correct while nothing is released, the first
milestone; **the paged window comes with batching and long context**. The slab's last advantage — a
fixed-shape table for graph capture — dissolves natively on the paged arm: below the window the
local row is always the trailing ~65 page references, so decode rotates *references* through a
fixed `64 + 1`-slot row whose table updates outside the graph. Fixed shape for capture, shared
pages for dedup — the slab arm keeps only its simplicity, and is the fallback.

One pool cannot survive reclamation, and not because of a race: a request holds one assignment
list, so releasing the sliding group's oldest block releases it for the full-attention group too —
"held for one geometry, released for the other" is not a state one pool can represent. Two families
are the precondition for independent reclamation. Logical positions stay aligned across them by two
things only — the shared tokens-per-page and the two `kv_position`s advancing together; physical
block ids and allocation history are independent by design, and an implementation forcing physical
lockstep has misread this.

## Sharing economics, revised to transient

The shared-prefix argument for the paged arm is real but **transient**. After c requests fork from
a shared checkpoint and each generates `d` tokens, the local family's physical footprint is roughly
`W + (c - 1) × min(d, W)`: the moment every request's window is its own generated content
(`d ≥ W`), the local sharing win is gone and paged and slab steady states cost the same. The
*persistent* sharing win lives in the **global family's prefix** — which past 20k positions is also
the larger memory term. The paged arm's remaining arguments are the transient window sharing (still
material for short replies), one uniform mechanism across both families, and checkpoint metadata at
page granularity — not a steady-state memory delta.

## Checkpoints: a frozen query semantics, not a structure

A hit must make resume possible, and resume needs **every** required family at the boundary. The
frozen query is:

```
resolve_checkpoint(g_end) -> the largest t <= g_end at which every
                             required family is exactly recoverable
```

Page alignment is a property of the free candidate set, not of the query: reference-materialized
seals produce page-aligned candidates for free, while a paid partial-page snapshot can place an
exact boundary anywhere — both are candidates under the same query.

The two materialization arms implement it differently. Reference checkpoints are **emergent** — no
manifest objects: both families share one prefix-hash chain; a sealed boundary's local page set is
directly computable from `t`; and a maintained interval index over the local family's contiguous
runs (same truth source as the block table, updated atomically with it) answers the query in
`O(log)` — an evicted block breaks an interval, so stale boundaries become unqueryable
structurally, with no invalidation ordering to get right. Copy checkpoints are **explicit**: a
manifest naming the boundary, the global prefix end, the local resident origin and one artifact per
family, invalidated atomically before its blocks are released.

The fallback ladder is part of the semantics: an incomplete window at `t` yields a smaller `t`;
nothing available means a full prefill from zero. One corollary drives eviction: **a global prefix
without its matching local window is worth nothing to resume** — a position's local K/V depends on
a cone of roughly `layers × window` entries through the layers below, so the window can be neither
skipped nor cheaply recomputed, and eviction must price the two families together at checkpoint
granularity. Rejected on the same grounds: position-independent window reuse (pre-RoPE K is already
position-polluted below — an approximation a correctness-first engine does not ship) and decode-side
window recompute (the same cone makes it a near-full prefill).

Two consequences of the reference arm are worth naming because they delete policy: resident hits
need no checkpoint cadence (every page-aligned boundary is naturally queryable), and the offload
write path becomes save-on-seal with content-addressed skips — adjacent windows overlap, so write
traffic converges to one upload per block, `O(n)` overall. Overlap sharing across checkpoints is a
**metadata** property, not a physical one: two retained windows `d` apart hold a page union of
roughly `W + d`, and retaining `K` turn boundaries holds history pages back from the free pool —
an explicit memory-for-TTFT trade, not a free lunch. Turn boundaries rarely align to pages: the
free immutable-seal path can only offer the last *sealed* boundary, so a free checkpoint rounds
**down** and resume recomputes at most `page_size - 1` tokens forward from it; an exact-boundary
checkpoint needs a paid partial-page snapshot — a policy-level choice to declare, either way.

## What reclamation breaks

**Page slot index is absolute position** in every list the pool builds today, and assignment lists
truncate from the *end*; dropping from the front is not expressible. **A request holds one
assignment list**, released whole. So reclamation needs three things, not one:

1. **The resident origin reaching the kernels** — the coordinate section's derivation; a per-token
   slot array only if scatter ever goes non-contiguous.
2. **Resident origin and length as first-class request state** — views must build from
   `resident_origin`/`resident_len`, not from the absolute position, or a front-trimmed page row
   trips exact-cover assertions on the next step.
3. **An ownership path for releasing a live request's oldest blocks.** Nothing exposes one today:
   the only partial drop is LIFO and idle-only. This is the piece with no existing analogue, and it
   lands with the paged-window stage together with the store-side elements it implies — the
   interval index, the dual-family checkpoint hold, and a host tier that can query and load
   non-contiguous block sets by content hash.

Reclamation happens between forward passes, never inside one — a pass needs its whole `C + window`
span resident — and it has two call sites that release different amounts: up to a chunk's worth
after a prefill step, one page every `page_size` tokens during decode. Wiring only the decode path
leaves long prompts holding everything.

## K and V share a projection weight, not a cache slot

`attention_k_eq_v` is true at all three sizes and means: full-attention layers ship no `v_proj` of
any form (the loader enforces the absence), and the architecture forks after the shared projection
— K takes `k_norm` then RoPE, V takes the weightless norm and no RoPE, the fork sitting **before**
`k_norm`, which the layer oracle pinned empirically against the reference. Two distinct tensors are
materialised per token and both must be cached; the fused hd512 prep writes both in one pass. It is
not a cache saving, and position 0 is especially misleading for anyone tempted to re-measure it,
because RoPE is the identity there.

The wrong version is one line away: the paged-KV assembly takes `k_offset_elems` and
`v_offset_elems` as separate offsets into one buffer, and passing the same offset twice aliases
them — wrong output rather than a crash. The Gemma entries close this by deriving both offsets from
one checked layout derivation; other callers still own it. So the cache stores both and
`layer_stride = 2 × kv_block_len` stays.

One tempting derivative is explicitly **not promised**: storing the pre-fork `k_proj` output
(8 KiB/token) on offload boundaries to halve the global family's checkpoint growth. The raw
projection is gone after the layer forward — resident post-norm K/V cannot losslessly reconstruct
it — so a save-time repack has no data source; capturing a raw sidecar during forward or a
producer-side pack-on-write interface would be new machinery. Research item, not an accounting
line.

## Scheduling and budget across two families

Admission does not allocate — it decrements a scalar budget, and blocks are taken later at schedule
time — so there is no partial grant to roll back there, and schedule-time allocation already
reverts LIFO. What is new is transactionality across families (frozen behavior 1, above) and the
reservation formula: lifetime reservation models occupancy as monotonically growing, which for a
reclaiming sliding family would demand 16384 pages for a request declaring the maximum length —
rejecting exactly the requests reclamation exists to serve. Its per-request reservation is the
steady cap alone, `min(lifetime_blocks, ceil(window / page_size) + 1)`; the prefill burst is one
step-wide allowance the scheduler holds out of the pool, not part of any request's reservation.

Pool sizing has two regimes. Without reclamation both families need the same page count and the
sliding one takes 95% of the bytes and sets the context ceiling. With reclamation the sliding pool
follows the three-term formula and is independent of context — 591 pages, 2.89 GiB, at 12B with
eight concurrent and page size 16 — while the full-attention pool takes the remainder and sets the
ceiling. `servable_len` and any single cache-usage ratio have to state which family they mean.

## Frozen behaviors

1. **Transactional families**: schedule/apply/revert covers both families atomically; a single-side
   apply success is fatal.
2. **Complete hits only**: a prefix hit adopts the largest common checkpoint at which both required
   families are complete — the resolve semantics above.
3. **Exact admission accounting**: per-request frontiers and chunks, never window-only.
4. **Fail closed** on everything that assumes one block-id space — prefix matching, KV offload, P/D
   disaggregation — until the multi-family store lands. **Amendment**: a full-semantics prefix
   cache may land engine-internal ahead of the store, under strict conditions — complete rather
   than partial (behaviors 1–3 all hold), default-off behind an environment opt-in with bytewise
   identical behavior when off, implemented against the frozen resolve semantics so the logic ports
   when the store arrives; offload and P/D stay fail-closed regardless.

**Executor boundary**: the serving stage may run one pool internally, but the model executor
consumes only a per-step plan (absolute positions, per-family page rows, and the local family's
resident origin) from day one — swapping the storage underneath later must not touch the forward interface.

## Tensor parallelism

The qwen3 sharding policy — refuse a world size that does not divide `num_key_value_heads`, shard
by integer division — cannot be reused: the full-attention group has fewer KV heads than a
plausible world size (1 at 12B, 2 at 26B, 4 at 31B), and integer division alone silently drops
heads. With `P` the world size and `Q`, `Kv`, `G` the attention, sliding-KV and global-KV head
counts, a legal `P` satisfies all three:

```
Q  % P == 0
Kv % P == 0
G  % P == 0   ||   P % G == 0
```

The first two stop the silent loss (12B at `P = 6` would cover 12 of 16 query heads without
erroring). The third is unwitnessed by the published sizes but pins the rank mapping: `G >= P`
shards contiguous runs of `G / P` heads; `G < P` replicates each head to `P / G` contiguous ranks,
one head per rank — from which point the group stops dividing and costs the same per rank at every
world size.

**Every capacity figure in this document is per rank**, and the tables are TP1. At 12B `G = 1`, so
the full group is at its floor already:

| 12B, one request at 262144, per rank | TP1 | TP2 | TP4 |
| --- | --- | --- | --- |
| sliding / full attention, KiB per token | 320 / 16 | 160 / 16 | 80 / 16 |
| total | 84 GiB | 44 GiB | 24 GiB |

TP buys less than naive division suggests: the group that stops shrinking is the one that scales
with context. It cuts the other way for kernels — the per-rank GQA group is 16 at TP1, 8 at TP2, 4
at TP4, so 12B's full-attention decode lands back inside the compiled dispatch set above TP1.

## CUDA graph capture

Decode is captured with pre-allocated buffers for pointer stability, and two families impose two
page-table contents with independent CSR offsets, two attention geometries, and two base pointers
with per-layer offsets. A shrinking page row is compatible with replay — sequence length, page
count and last-page length are derived device-side — and the fixed `64 + 1`-slot local row from the
ring synthesis keeps the captured shapes constant while its table updates outside the graph. One
question precedes layout: for GQA groups the decode kernel cannot instantiate, the existing
executors reroute decode to an uncaptured path, and at TP1 12B's full-attention group is exactly
such a group — whether it is captured at all differs per TP degree.

## RoPE precompute

The two groups need different tables — theta 10000 with full rotation at head_dim 256 for sliding
layers, theta 1000000 with the proportional scheme for full-attention layers. The core precompute
now separates the rotary width from the frequency denominator, and the sliding family is its
identity case. The global family's tables stay **full head width by design**: the proportional
pairing rotates `(d, d + head_dim/2)` across the whole head with the inactive band zero-padded, so
narrow tables are not an available optimisation — the cost at a given serving length is two
`max_seq_len × head_dim` bf16 buffers per family, and both families' tables are sized from an
explicit configured serving limit capped at `max_position_embeddings`. Deriving their length from
the KV budget is circular — the budget is measured after they are allocated.

## What the loader validates before building the layouts

Landed in the config/probe layer: the layer map's length against `num_hidden_layers` and its
6-periodic pattern, each rope family's declared `rope_type` against the implemented algorithm,
positive finite thetas, and the global rotary width derived exactly (`partial_rotary_factor ×
global_head_dim` must land on a positive even width within the head). Still owed with the serving
work: GQA divisibility per group, world-size legality by the three conditions above, and a
compiled-or-accepted decode path for each group's **resolved per-rank** mapping — the config ratio
is not the contract, the per-rank mapping is.

## The optimisation ladder, after correctness

1. ~~Fused prep + KV write~~ — landed: the preps write normalised/rotated K and weightless V
   straight into the pool, one raw-K read, no fork copy.
2. Per-family page size: local page 16 = 5 MiB; page 8 halves frontier waste and checkpoint
   granularity at the cost of table and offload op counts. Keep it configurable; measure before
   choosing.
3. A windowed split-KV decode: the landed windowed decode is non-partitioned, and at bs=1 the
   sliding group's CTA count is small enough that split-KV may still pay.
4. A native group-16 decode for TP1's full-attention group: via-prefill is correct; a dedicated
   instantiation may shave fixed per-layer overhead.
5. A V-only wire format: mathematically V is the weightless norm of raw K, so K might be
   reconstructed from V (reweight + rotate) — but bf16 intermediate rounding does not guarantee
   bit-identity with the original K, so this must pass the golden gates and parity checks before it
   is anything more than an experiment. Not presumed lossless.
6. FP8 for the global family: past 20k positions the global family is the dominant linear term, so
   quantising it outranks squeezing the local frontier further — independent correctness work after
   bf16 serving is solid.

## Excluded

- **Quantised KV at load**: the NVFP4 checkpoints declare an FP8 KV cache and this engine has no KV
  quantisation concept yet; an unsupported declared KV dtype fails closed, naming the scheme.
- **Cross-layer KV sharing**: `num_kv_shared_layers` is 0 at all three supported sizes.
