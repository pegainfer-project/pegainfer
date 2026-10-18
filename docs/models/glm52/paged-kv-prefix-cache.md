# GLM5.2 paged KV pool + prefix caching

**TL;DR**: The static per-slot KV partitions (slot `b` owns tokens `[b*cap, (b+1)*cap)`, identity block table) are replaced by a per-rank `pegainfer-kv-cache::BlockPool` of 64-token pages with content-hashed blocks — the Kimi #239 pattern, zero kernel changes. Admission reserves a request's full-lifetime page count (honor-or-reject), prefix caching is on by default (suffix-only prefill, `cached_tokens` reported), and the coordinator ships each step's page rows + write slots to the ranks as `Glm52StepKv`. DSpark and prefix caching are mutually exclusive (draft anchor-position assert). **jz-38 validated 2026-07-06: warm prefix TTFT 14.1 s → 0.84 s (16.9×) on a 1200-token varied prompt with byte-identical output; step-bench sweep dead even with the D5 main anchors (b1 25.4 / b8 43.2 ms, c64 1483 tok/s); oracle gates green; 9-way near-tie divergence adjudicated under the D2 cross-bucket contract.**

Last touched: 2026-08

## Why the kernels needed nothing

The attention stack was already page-table-driven; only the *host-side mapping*
was static:

- FlashMLA sparse decode reads the packed cache by DSA top-k **token slot
  indices**, not a dense walk — physical contiguity per request was never
  required.
- `local_topk_to_slots` already converts positions through the block table
  (`page = block_table[t, off/64]; slot = page*64 + off%64` —
  `glm52_indexer.cu:9`).
- The DeepGEMM paged MQA consumes `block_table` (BLOCK_KV=64) and bounds its
  walk by `seq_lens`.
- The cache-pack and index-K store kernels write at the flat token slot from
  `slot_mapping` — the value just changes from `slot*cap + pos` to
  `page*64 + pos%64`.

## Design

### One pool, two arenas, shared ids

Each rank's `BlockPool(page=64)` hands out block ids; the rank's per-layer
MLA packed caches (656 B/token) and index-K caches (full-indexer layers) are
both indexed by those ids, and the pool, `Glm52RankModel::finish_kv`, and the
scheduler all share the ONE launch-decided block count.

**Pool sizing is a measured launch decision, not a formula (#823).** The old
`slots*ceil((cap+1)/64) + 1` provisioning multiplied the servable cap into
the pool (a 200K cap forced slots×200K of KV — unaffordable — and the derived
cap shrank as slots grew). Now the workers build everything except the
pool-scaled slabs, report measured free VRAM, and launch min-reduces and
fills the pool from fact at the exact per-block slab cost: the legacy
`slots x cap` count remains the FLOOR where it fits (defaults unchanged, the
fill only adds), an explicit big cap falls back to a one-request floor
(`ceil((cap+1)/64) + 2`), and `GLM52_KV_POOL_BLOCKS` pins the count for A/B
runs (rejected below the one-request floor — a smaller pool would advertise a
cap no request can fit). The only estimate left is `GLM52_GRAPH_RESERVE_GIB`
(default 3) for lazy graph instantiation + runtime workspaces. One page is
the reserved **padding page**: padding rows and graph pre-capture write
there, nobody reads it meaningfully (Kimi's benign-garbage argument). The
`cap+1` (not `cap`) in the floors keeps max-shape requests
(`prompt + max_tokens = cap + 1`) admissible including the dangling-token
page — a toxic-review finding; the naive `ceil(cap/64)` variant silently
degraded max-shape concurrency.

Measured on GB300 (EP4, 16K cap): the fill lands ~1.35-1.44M pool tokens per
rank (2.6x the old 32-slot nominal), and admission's lifetime reservation
arbitrates mixed loads — 12 concurrent ~118K requests totalling 1.42M tokens
against a 1.35M pool queue the overflow and all complete. The win
is *sharing* — released requests' sealed blocks stay matchable as the prefix
cache instead of dying with a slot.

### Step protocol: `Glm52StepKv`

The coordinator (which owns the pools) computes per step, per rank:

- `pages: [bucket, table_width]` row-major page ids (span rows repeat their
  slot's row; tails and padding rows = padding page). The prologue uploads it
  H2D into the bucket's device block table (replacing the old per-row dtod
  gathers from the static identity table). ~36 KB at bucket 8 / cap 72576;
  the #812 verify-span buckets raise the worst case 6x (bucket 48).
- `slot_mapping[row] = pages[pos/64]*64 + pos%64` — the flat write slot for
  both caches. The prologue cross-checks it against the page row (drift =
  failed step, not silent cross-request corruption).

Whole-step CUDA graphs are untouched: arena base pointers are allocate-once,
page ids are device *data* rewritten outside the graph — the same
contents-change-pointers-don't contract as before.

### Admission: full-lifetime reservation

`lifetime_blocks = ceil((prompt + max_tokens)/64)` — one token more than the
last written position, because kvbm appends the final generated token and
provisions its page (the dangling-token off-by-one Kimi probed empirically).
`admission_target` picks the least-loaded rank among those with a free slot
AND `Σ active lifetimes + new ≤ pool usable`; no rank fits → the request
stays queued (never mid-decode starvation, never a livelock — an empty rank
fits any request the length validation admits). The reservation is
conservative (prefix-shared pages counted per holder): over-reserving can
only defer admission.

### Per-step KV bookkeeping (schedule/apply pairing)

The submit walk schedules each active span (`schedule_prefill(n)` /
`schedule_decode` / `schedule_speculative(n)`) and records a `SpanKind`; the
output walk applies the exact pairing (`apply_prefill_chunk` /
`apply_prefill(tok)` / `apply_decode(tok)` / `apply_speculative(&committed)`).
Every page row handed to a forward comes from `step_page_indices` (the
trimmed row — kvbm's eagerly allocated generation block must not reach the
kernels; Kimi's DP8-deadlock lesson). The span's first position is asserted
against `kv_position()` every step.

### Launch-ahead × paging

The feed kernel advances `slot_mapping += 1`, which is only valid inside the
current 64-token page. The lease gate adds `(position+1) % 64 != 0` for every
active row — streaks break at page boundaries (≤ 1/64 of steps lose the
launch-ahead win), which also bounds padding rows (reset to position 0 by
each full prologue) inside the padding page: an active row's boundary always
arrives within 63 leased steps. `decode_step_harvest` crash-early-asserts the
same invariant. On consumed steps the coordinator skips building the page
rows (the worker's prologue is skipped) but still runs the kvbm scheduling —
bookkeeping must advance every step.

### Prefix caching

`match_and_add_prefix` at admission (full 64-token blocks only, ≥1 prompt
token always uncached); `Glm52SlotState` starts `fed = cached_tokens`, so the
prefill spans cover only the suffix at their true absolute positions. Nothing
to re-materialize on a hit — GLM5.2 prefill *rides decode* through the same
paged FlashMLA read path, and k_pe is stored RoPE-applied (absolute positions
baked), so a warm hit is just a later span start. The index-K cache content
rides the same block ids. `cached_tokens` is reported in the `Scheduled`
event. Kill switch: `--no-prefix-cache`.

Known blind spot: admission places by least-loaded rank first and only then
matches against that rank's pool, so under concurrency a repeated prefix can
land on a rank that never saw it — worst-case hit rate ÷8. The c1 warm TTFT
A/B can't see this (solo always lands on the same rank). Cache-aware
placement (probe every rank's pool at admission, prefer the best hit) is the
follow-up lever.

### DSpark exclusion

The draft lane asserts `anchor_pos == committed + pending` context rows; a
skipped prefix never produces the aux-hidden captures ({7,22,38,54,69}) the
draft consumes, and recomputing them = running the target forward = no
savings. Drafter on → prefix matching off (logged). Follow-up if warm-hit ×
spec-decode matters: base-offset the draft context (draft KV starts at the
suffix; verify keeps losslessness, accept rate near the hit degrades).

## Determinism contract change (for the jz-38 gates)

- **Run-to-run** (same binary, fresh engine, same request sequence): pool
  allocation is deterministic from an empty pool → same page ids → outputs
  stay reproducible.
- **vs-main byte-parity**: physical page addresses differ from the per-slot
  layout, and decode has address-sensitive accumulation (Kimi #293) — cold
  solo output may flip at near-ties. Gates must move to the Kimi contract:
  identical token streams expected, near-tie flips adjudicated like the
  existing cross-bucket divergence contract (D2), warm-vs-cold A/B token
  equality + top-1 tolerance.

## jz-38 validation record (2026-07-06, branch bb7adff rebased on #586)

Gate scripts `glm52_pagedkv_gates.sh` / `glm52_pagedkv_gates2.sh` /
`glm52_pk_bench.sh` on jz-38 `~/develop/xingming`.

- **Oracles**: mla full/short, layer dense, EP8 layer, bookend — all PASS.
  indexer oracle FAILED on a missing fixture file — the #541 pre-existing
  reference drift; main's own gate suites (D2/D5) exclude it too.
- **GATE1** solo determinism + warm-vs-cold token parity: PASS byte-exact
  (a warm hit reuses the same physical pages — no address change, no flip).
- **GATE2** vs stored PR5c refs: near-tie divergence on both prompts
  (coherent continuations, split at a token boundary) — the predicted
  address-sensitivity contract change (Kimi #293 class), adjudicated.
- **GATE3b** warm prefix TTFT, varied 1200-token prompt + ignore_eos:
  cold 14.12 s → warm 0.84 s (**16.9×**), non-empty byte-identical output.
  (First attempt used a repeated-sentence prompt → first-token EOS → empty
  parity; the D3 trap in a new costume. Bench prompts must be varied prose.)
- **GATE4** 9-way identical concurrency: 8/9 byte-identical to the solo run
  over their full extent; the 1 outlier (admitted first, different bucket
  phase) diverges at a near-tie — the documented D2 cross-bucket contract.
- **GATE5/5b** 17-way diverse + long-prefill/decode mix: 17/17 and 7/7
  complete, zero server error lines.
- **GATE6** `--no-prefix-cache`: cold determinism PASS.
- **GATE7** DSpark round: prefix-cache self-disable logged, deterministic
  output, accept-incl-bonus 2.26 on the code prompt (in family with the D3
  record for that class).
- **Step bench** (in-process sweep, conc = 8×bucket): b1 25.44 / b2 31.01 /
  b4 36.80 / b8 43.16 ms p50, c64 1483 tok/s — vs the D5 main anchors
  25.4 / 30.9 / 36.4 / 43.5 and 1475 tok/s: **flat, zero regression**
  (the per-step H2D page-row upload and the coordinator's page bookkeeping
  are invisible at these step times; consume steps skip the page build).

## Next step

Merge, then: pegaflow offload PR (per-layer strided registration over these
arenas, qwen3 #316 connector pattern), the P/D KV-ingestion campaign on this
substrate, cache-aware admission placement (the ÷8 blind spot above), and
the scheduler.rs module split (toxic-review H3 chore).
