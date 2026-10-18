# Kimi-K2 KV cache: adopting the qwen3 paged stack (#239 → #230/#231)

**TL;DR**: Kimi's KV is *already paged* at the kernel level (FlashInfer-style page tables, paged MLA append/decode kernels) — the "fixed 2048-token arena" was only a static Rust-side slot→pages mapping. The qwen3 stack landed in **one PR with zero kernel changes**: a kvbm-managed `BlockPool` per rank replaces the static mapping, admission uses full-lifetime reservation (the qwen3 #85 pattern), over-cap requests get an explicit `Rejected` instead of poisoning the batch mid-decode, and the per-request cap rises 2048 → 8192 tokens (DP prompts stay ≤ 2048 — a PPLX fabric-buffer constraint, see below). Prefix caching (#230) now rides this substrate: kvbm full-block matching at admission + suffix-only prefill via latent gather → kv_b decompression → cached k/v assembly (see below); long context (#231) remains open. Landed on `feat/kimi-kv-pool`, 2026-06-06/07. 8×H200 verification caught a DP8 decode deadlock (kvbm's eager generation block vs the worker's exact page-count check — fixed via `step_page_indices`, see below); golden-gate parity is green; run-to-run bitwise logprob determinism is lost by design under dynamic paging (address-sensitive accumulation order, #293) — the det check now gates token-stream equality + a 0.25-nat top-1 tolerance.

Last touched: 2026-06

## Findings (verified against code)

### 1. The roadmap's "silent overrun" claim was stale

The worker always rejected out-of-range positions (`configure_slot_prefill`,
`configure_batch_decode`). There was no silent corruption. The real problem
was **blast radius and lateness**: a `prompt + max_tokens > 2048` request
prefilled fine, decoded normally until the append position hit 2048, then the
decode step errored — on TP8 the scheduler drained **the entire active batch**
with `TokenEvent::Error`; on DP the coordinator failed **all 8 slots of that
rank**. One over-budget request took down every neighbour mid-generation
after burning up to 2047 tokens of compute. Nothing validated
`prompt + max_tokens` anywhere.

### 2. Kimi KV is already paged — the kernels need nothing

`KimiMlaPagedKvLayout` (`pegainfer-kernels/src/ops/kimi_k2/mla.rs`), page
table buffers (`page_indices_d/page_indptr_d/last_page_len_d`), a paged append
kernel (`kimi_mla_paged_kv_append`) and a paged MLA decode kernel
(`kimi_flashinfer_batch_decode_mla`) were all in production. The "fixed arena"
was one line of host code: slot *i* statically owned physical pages
`[i*128, (i+1)*128)`. Replacing that identity mapping with a per-request block
table is pure host-side work.

CUDA graph compatibility is free: page tables are H2D-uploaded fresh each step
into fixed-address buffers before graph replay — contents change, pointers
don't.

kpe is stored **RoPE-applied** (positions baked at append time). Consequence
for prefix cache later: cache-hit suffix prefill must rotate from the cached
length — the same start-position contract qwen3 fixed in PR #216. Kimi already
threads explicit `positions_d`, so the fix shape is cleaner than qwen3's was.

## Design (as landed)

### Logical/physical split: `BlockPool` in `pegainfer-kv-cache`

The qwen3 `KvCacheManager` was split so MLA models reuse the logical layer
without inheriting the full-attention physical layout:

- **`BlockPool`** (`pegainfer-kv-cache/src/pool.rs`): kvbm `BlockManager` +
  the reserved padding block + `RequestKv` (the `SchedulableSequence` wrapper:
  `schedule_prefill/apply_prefill/schedule_decode/apply_decode`, RAII release).
  Owns **no GPU memory** — it hands out block IDs.
- **`KvCacheManager`** is now a thin facade `{ pool: BlockPool, buffer: KvBuffer }`
  for full-attention models (qwen3). Kimi consumes `BlockPool` directly and
  owns its MLA physical layout: per rank, one `ckv` + `kpe` buffer pair per
  layer (`KimiWorkerKvPool`), indexed by the pool's block IDs. The dual
  ckv/kpe segment never crosses the logical layer — kimi's kernels take
  separate base pointers sharing one page table, so no `KvLayout`
  generalization was needed.

A layout subtlety that makes one shared pool possible:
`KimiMlaPagedKvLayout::required_ckv_len()/required_kpe_len()` depend only on
`max_pages × page geometry`, **not** `batch_size` — so all decode bucket
arenas (1/2/4/…/64) share the rank's single pool buffers, each arena keeping
its own layout with the bucket's batch dimension.

### Page table plumbing: row-major CSR → slot-indexed scatter

The scheduler speaks **row-major CSR** (`KimiKvStepPages`: one page row per
batch row). The worker's FlashInfer metadata is indexed by **slot**
(`batch_size` rows, idle slots included). `build_slot_page_table`
(`worker/cache.rs`) scatters the row CSR to the slot table; idle slots and
CUDA-graph padding rows ride the pool's **padding page** (block 0,
leak-registered in `BlockPool` so it can never be handed to a request).
Concurrent garbage writes to the padding page are benign by construction.
Page counts are matched **exactly** (`pages == ceil(kv_tokens/16)`) — any
drift between scheduler accounting and the kernel's view crashes the step
instead of silently truncating attention.

**The `step_page_indices` contract** (found as a DP8 decode deadlock on
8×H200): kvbm's `schedule_decode` eagerly allocates the *next* generation
block whenever this step's token will fill the current block's last slot, so
the raw `RequestKv::page_indices()` holds one block more than the KV tokens
need at **every** block boundary (`kv_tokens ≡ 0 mod 16` — any request hits
it within 16 decode steps). Handing that raw list to the worker trips the
exact-match check above. Every page row given to a forward pass must come
from `RequestKv::step_page_indices(new_tokens)`, which trims to
`ceil((kv_position + new_tokens)/16)`; a regression test in
`pegainfer-kv-cache/src/pool.rs` sweeps prompt lengths × decode steps and
self-retires if kvbm stops over-allocating.

Why it surfaced as a *hang*, not an error: on DP the owning rank's
`forward_decode_batch` failed **before entering the PPLX collective**, the
other 7 ranks parked in `GdrFlag::wait` forever, and the coordinator blocked
collecting results in rank order — 600 s silent timeout, error invisible.
The coordinator is now hardened: in DP lock-step, any rank-level step `Err`
is unrecoverable by construction (peers are already inside EP collectives),
so `abort_poisoned_step` crashes the process loudly instead of logging and
continuing into a deadlock. Local single-GPU tests can never exercise this —
DP8 decode is the first code path where scheduler page rows meet the
exact-match check with a generation block pending.

### Admission: full-lifetime reservation, honor-or-reject

Two distinct per-request quantities — conflating them was a real bug caught
in review:

- **KV positions written**: `prompt + max_tokens − 1` (the final token is
  returned but never fed back, so its KV is never written — qwen3's
  dangling-token contract). Checked against `KIMI_MAX_REQUEST_TOKENS = 8192`
  (RoPE table / position bounds).
- **Pool blocks drawn**: `request_lifetime_blocks = ceil((prompt +
  max_tokens)/16)` — one token *more*, because kvbm appends the final
  dangling token to the sequence and provisions its block even though its
  KV is never written (probed empirically: prompt=16/max=17 peaks at 3
  blocks, not 2). Reservation must use this, or requests die mid-decode at
  boundary alignments. (qwen3's admission has the same latent off-by-one —
  follow-up issue.)

`validate_kv_capacity` (`scheduler/lifecycle.rs`) rejects at admission with
the limit spelled out when a request **can never fit**: per-request cap,
pool capacity, or a path-specific prompt cap. A request that fits but not
*right now* is **deferred**, never rejected:

- **TP8** (`scheduler.rs`): wave model — budget = `available_blocks` at batch
  start, deferred requests go back to the queue *front* (the wave drains
  fully, so the next wave starts from a full pool; FCFS preserved; the first
  valid request always fits ⇒ no livelock).
- **DP** (`scheduler/dp.rs`): live budget per rank — `rank_kv_budget =
  available − Σ future_blocks` over active slots, where `future_blocks =
  lifetime blocks − ceil((prompt + completion − 1)/16)` (a lower bound on
  blocks already drawn, computed from request fields — kvbm's own counters
  only see tokens already appended, never the future). `round_reserved`
  covers queued-but-not-installed admissions in the same scheduling round.

Decode-time block exhaustion is impossible by construction; a
`schedule_decode` failure is an accounting bug and fails loudly
("violated full-lifetime reservation").

### Capacity model and the PPLX 2048 constraint

| Path | Pool/rank | Prompt cap | Total sequence cap |
|---|---|---|---|
| TP8 (NCCL MoE prefill) | 8192 pages ≈ 9.2 GiB | 8192 | 8192 |
| DP (PPLX EP) | 1024 pages ≈ 1.15 GiB | **2048** | 8192 |

PPLX bootstrap sizes fabric buffers from `max_num_tokens` at startup; bumping
2048 → 8192 costs ~13 GiB extra per rank (`compute_sizing`: recv buffer scales
linearly with dispatch tokens × 14352 B). Decode dispatches are batch-many
tokens, so only the **prompt** is capped on DP
(`PPLX_MAX_DISPATCH_TOKENS = 2048`, `moe_pplx.rs`) — total sequence still
reaches 8192 through decode. TP8 prefill uses NCCL MoE with per-call
allocations (verified: no 2048 assumption), so TP8 prompts go to 8192.

Per-request 8192 is bounded by the per-arena YaRN RoPE table
(`KIMI_MAX_REQUEST_TOKENS × 64 × 2 × bf16 ≈ 4 MB`) — growing it further is
#231's problem (single-CTA MLA decode scan, prefill temp buffers), not the
pool's.

Pool buffers are allocated eagerly at weight load (crash early on OOM); the
stable base pointers keep captured CUDA graphs valid.

### Prefix caching (#230, landed on this branch)

Default on, no flag. Admission calls `match_and_add_prefix` (full-block
matching, ≥1 token always left uncached) and prefills only the suffix.

The interesting part is MLA-specific: kimi prefill attention never reads
the paged pool — it materializes contiguous k/v from the latent via the
kv_b GEMM and runs `single_prefill`. So a cache hit cannot just point the
page table at cached pages; it must rebuild the k/v rows the suffix
attends over:

1. **Gather** the cached ckv pages into a contiguous `[cached][512]`
   latent buffer (`gather_cached_ckv_kernel` — pure page-table scatter).
2. **Decompress** through the *same* `kv_b_proj` GEMM the cold path uses
   (same weights, same math — only the GEMM's M dimension differs).
3. **Assemble** k/v rows `0..cached` from the decompressed latent plus
   the pooled kpe pages (`assemble_cached_kv_kernel`). Pool kpe is stored
   post-RoPE: it is broadcast per head **verbatim** — re-rotating it would
   corrupt positions (the bug class the qwen3 prefix cache hit with its
   RoPE scalar path).
4. The suffix assembles at absolute positions `cached..cached+seq` (RoPE
   table sized to `kv_len`), and FlashInfer's bottom-right causal
   alignment makes `qo_len < kv_len` exactly the suffix causal mask.

DP plumbing: the owning rank forwards only the suffix; padding ranks and
PPLX prefill scratch size against the suffix (`ep_max_seq_len`), so a
warm hit also shrinks the EP collective, not just the GEMMs. The worker
page table covers `cached + suffix` tokens and the append positions start
at `cached` — the kernel-side append needed zero changes.

Cost honesty: a warm hit does **not** skip all prefix compute. Every
layer still gathers the cached latent and re-runs the kv_b decompression
GEMM over it, and the materialized k/v (and the attention's K side)
scale with `cached + suffix`. What a hit skips is everything upstream of
attention for the cached tokens — qkv_a/q_b projections, MoE/dense MLP,
the EP collectives — which dominates prefill, but a long cached prefix
with a tiny suffix still pays O(cached) decompress + attention per
layer. The alternative (persisting decompressed k/v) trades that compute
for `heads × 320/576` more cache memory per token — rejected; the latent
is the cache.

Accuracy: the golden gate's repeated prompts now exercise the cached
path under every bound (teacher-forced sweep requests share block-aligned
prefixes), and the det check doubles as the warm-vs-cold A/B: run A cold,
run B cache-hit, same tokens + top-1 |Δ| ≤ 0.25 nat.

## Run-to-run determinism under dynamic paging (#293)

Bitwise run-to-run logprob equality is **lost by design** with a dynamic
pool, and the golden gate's det check now asserts the real contract:
identical token streams + top-1 |Δlogprob| ≤ 0.25 nat (2× observed max).

What the 8×H200 investigation established (two nodes, 3/3 repro):

- Identical inputs → identical tokens, but every decode position's top-32
  logprob distribution wobbles 1-2 bf16 ULP (top-1 |Δ| mean ≈0.03, max
  0.12 nat). Prefill is bit-identical. `main` is bit-stable.
- The two runs get different physical KV pages (LRU churn): run A `[1]`,
  run B `[4]`. Prefill is placement-invariant; decode is not.
- Controlled experiment: pristine zero-initialized pool, det check only.
  Page *content* identical between runs, only the page IDs differ —
  **still diverges**. Garbage-content leakage is ruled out.
- Surviving mechanism: an accumulation in the decode path whose summation
  order is address/timing-sensitive (candidates, not isolated: cuBLASLt
  skinny GEMMs, PPLX combine arrival order, FlashInfer MLA decode). `main`
  is stable because the static arena gives bit-identical addresses every
  run — an incidental property, not a contract. Isolation tracked in #293.

## Next step

8×H200 verification, remaining items: golden gate on the prefix-cache
branch state (now covers warm path under all bounds); over-cap request →
explicit rejection (not batch poison); `prompt + max_tokens` boundary;
>2048-token prompt e2e on TP8 (4K–8K); DP prompt-cap rejection at 2049;
greedy bs64 TPOT p50 unchanged (~30 ms); warm-vs-cold TTFT p50/p99.
Golden-gate parity green pre-#230; det green under the #293 tolerance
contract.
