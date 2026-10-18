# GLM5.2 P/D native-MTP handoff

> **TL;DR:** TP4 prefill transfers the KV prefix and a five-token proposal
> to EP decode; gates from 89 tokens to 16K restore byte-identically over
> RDMA with `first_step=verify`. **The wire contract is v4 page-first**
> (#849/#850): the 101 per-layer arenas of v3 collapsed into one slab arena
> (`glm52.page`) whose page stride is the whole layout identity —
> `handoff_fingerprint()` now reads `glm52-native-mtp/4/page:<stride>/...`,
> and every restore lands as one contiguous copy per block instead of 101
> fragments (agent-trace A/B: ITL p99 156 → 74 ms, slow iters 57 → 0). End-to-end
> through a dual-endpoint router (P TP4 + D EP8 across two trays): parity
> gate byte-exact, GSM8K full 1,276/1,315 strict (0.970) at c32 — parity
> with the single-instance reference — and random-IO sweeps show the
> throughput knee at c32 (~1.6k tok/s out, ~14.1k tok/s in ceiling). Under
> sustained long-generation load the hard-coded 15 s handoff deadline is the
> first limit: with one prefill, c64 rejects ~24% on decode1, c128 ~83%; a
> GSM8K boundary case also exposed and fixed a fatal admission-capacity
> drift at `(input+output) ≡ 1 (mod 64)` (`d791dffc`). Scaling out: a second
> TP4 prefill (2P, round_robin router) holds GSM8K full at c64 with strict
> 1,273/1,319 (0.9718) and only a warm-up transient — the sustainable
> envelope doubles from c32 to c64. Multi-turn chat at c16 exposed a second
> reject cause: the stack script's default 8 GiB pegaflow host pool starves
> D-side 256 MiB fetch chunks into the same 15 s deadline rejects — 64 GiB
> clears it (TTFT p99 30 s → ≤6.3 s, +27% output). A standalone EP16 decode fleet
> (4 trays × 4 ranks, local prefill) serves GSM8K n200 c32 at 0.985 strict
> and ~944 tok/s out per endpoint decode-heavy (TPOT p50 29 ms), but
> co-located prefill costs ~55% of mid-workload throughput versus a
> dedicated P. **The P → EP16 handoff gate is now closed**: a Slurm-deployed
> bare-metal 4-node EP16 fleet admits handoffs `first_step=verify` on all
> nodes, survives multi-turn c64 with zero rejects (640/640, 1,158 tok/s),
> and exposes the next lever — without session-affinity routing, every turn
> full-history-refetches (TTFT grows per turn as fleet width dilutes
> rank-local KV reuse). The admission-coupled ITL p99 tail is **fixed**
> (PR #801, D-side #799): restores now park-and-poll instead of blocking the
> engine loop — multi-turn ITL p99 98 → 31 ms (c16) and 74 → 30 ms (c64)
> with zero rejects; the vLLM-compat P/D path is removed. PR #804 finishes
> the line (#799/#802/#805): every offload leg — query, load, save — is a
> pollable handle (pegaflow 0.23.5, `wait_for_full_prefix` on handoff
> queries), the P-side tail save detaches instead of blocking, and the TP4
> native-MTP proposal path broken by #797 is repaired (P dies on first
> handoff on any #797..#804 build without it). Same-fleet A/B: ITL p90 −27%
> / p99 −32% vs the pre-async binary; c64 full-stack 640/640 at 1,189 tok/s.
> **Prefix caching now works under native MTP** (constant v2 page salt —
> the v1 full-prompt salt had killed all cross-turn reuse) and the prefill
> pool retains released prefixes: multi-turn per-turn TTFT is flat instead
> of linear in history (c1 p50 ~270 ms at every turn; c8 late-turn p50
> 1,186 → 454 ms), with byte-identical greedy outputs.
>
> **Last touched:** 2026-08

## Preparation

- **Read**:
  - `docs/index.md` — GLM5.2 P/D state belongs with the model line, and the
    existing P/D execution record is authoritative for page naming, strict
    restore, and handoff failure semantics.
  - the retired vLLM-prefill target-only contract (#657, removed)
    transfers 78 MLA plus 21 index-K arenas, forwards the first target token,
    and admits D at `suffix == 1`; speculative state is explicitly absent.
  - `docs/models/glm52/tp4-prefill-only.md` — native TP4 prefill already emits
    the canonical 656-byte MLA and 132-byte index-K rows consumed by EP
    decode, but currently rejects external P/D and returns its predicted token
    without writing that token's target KV.
  - `docs/models/glm52/native-mtp-accuracy.md` — native MTP consumes the
    target's final-normalized hidden boundary and owns a separate layer-78 MLA
    plus index-K cache whose continuity affects acceptance.
- **Relevant history**:
  - the retired target-only path (#657) established that a strict D worker
    must never silently recompute missing prompt state and that transfer
    completion can lag the P response.
  - Native MTP is currently restricted to single-process EP8: its layer-78
    build path hard-codes EP8, its round uses the EP8 collective state, and
    its cache is not returned by `Glm52RankModel::kv_arenas`.
  - The desired boundary is stronger than the existing `suffix == 1`
    handoff: P returns an anchor plus initial draft token IDs, and D's first
    target step verifies that span directly.
  - Post-review found that the original admission log claimed
    `first_step=verify` without changing the slot out of its one-token prompt
    suffix. The historical hardware runs restored the intended bytes but did
    not verify the transferred proposal on their first target step.
- **Plan**:
  1. Specify and unit-test the handoff state machine: P transfers committed
     target pages, committed MTP layer-78 pages, `anchor`, initial draft token
     IDs, committed lengths, and page metadata; speculative MTP tail pages
     are not authoritative. D installs the proposal as its first
     `SpanKind::Speculative`, verifies it, then rebuilds MTP continuity from
     verifier hidden rows before making the next proposal.
  2. Generalize native layer 78 away from the EP8-only build boundary. Add a
     TP4 producer context path that consumes each target prefill chunk's
     final-normalized hidden rows and shifted prompt token IDs in batch,
     writes MTP MLA/index-K pages, and makes one initial proposal after the
     target anchor is sampled.
  3. Extend the PegaFlow registration contract from 99 target arenas to 101
     target-plus-MTP arenas. Give the MTP MLA/index-K pages stable names and
     page-first geometry, select one identical TP4 producer copy rather than
     concatenating rank shards, and keep incomplete restores fail-closed.
  4. Extend EP decode admission to restore the MTP arenas and seed the
     forwarded proposal so its first forward is verification. Preserve the
     existing target-only P/D mode when native MTP handoff metadata is absent.
  5. Gate CPU state transitions and arena geometry, release builds/tests, and
     hardware behavior. First validate TP4 P → EP4 D on GB300 with exact
     target output, `first_step=verify`, MTP committed-length continuity, and
     acceptance telemetry; repeat TP4 P → EP16 D when a 16-rank decode
     environment is available.
- **Risks / open questions**:
  - Chunk-boundary shifting must not create or omit the MTP row spanning the
    last token of one target chunk and the first token of the next.
  - The P-side TP4 layer-78 MoE produces proposals with different reduction
    numerics from EP decode. Target verification preserves output correctness,
    but first-round acceptance must be measured rather than assumed.
  - MTP MLA is logically replicated under TP4 even though query heads and MoE
    compute are sharded; this needs a byte-equality gate across producer ranks
    before registering only one copy.
  - The existing external producer is vLLM TP8. Native PegaInfer TP4 producer
    metadata must be a versioned extension, not an implicit reinterpretation
    of the merged target-only protocol.
  - EP16 requires multi-node collective and deployment validation beyond the
    four-GPU local development host; EP4 is the first executable consumer
    gate.

## Execution Log

### v4 wire contract: page-first slab replaces the 101-arena registration (2026-08-06, #849/#850)

The per-layer arena registration (99 target + 2 MTP, the v3 contract this
document was written against) is gone: each rank now registers **one**
pegaflow arena (`glm52.page`) in which block *b*'s page carries every
layer's slices at fixed offsets (78 MLA · 656 B + 21 index-K · 132 B + the
L78 MTP mirrors; content 3,502,592 B, stride 3,503,040 B). The handoff
fingerprint moved to v4 with the page stride as the layout identity — two
engines agreeing on the stride agree on the whole per-block byte layout, so
the `arenas:101/page:64` terms are retired. TP4's tensor-replicated KV is
expressed as pegaflow replica devices (worker 0 saves, loads fan out under
one shared query lease with a `world_size` consumer budget — the same
contract the vLLM MLA-TP connector uses). Restore-side effect measured on
the agent-trace replay A/B (r20 chunked vs r21 page-first + Direct): ITL
p99 156 → 74 ms, TPOT p99 84 → 43 ms, slow iters 57 → 0, fragment copies
2,437 → 0 — every restore is one contiguous copy-engine memcpy per block.

### P → EP16 handoff gate closed; Slurm bare-metal fleet deployment (2026-07-31)

The remaining deployment gate ("rerun the token-ID handoff contract against
a P → EP16/EP32 D fleet") is closed: TP4 P + metaserver feeding a 4-node ×
4-rank EP16 decode fleet through the dual-endpoint router (4 `--decode`
endpoints, round_robin), native MTP on both sides. All four fleet nodes
admitted handoffs with `first_step=verify drafts=5` on their local ranks
(0/4/8/12).

**Deployment moved from ssh+docker to Slurm, bare-metal.** The engine
binary resolves entirely against host libraries; the only gap is NCCL
(DeepEP needs 2.30.7 for `ncclCommQueryProperties`, hosts ship older) —
solved by one shared-filesystem copy of `libnccl.so.2.30.7` on
`LD_LIBRARY_PATH`. One sbatch job: node 0 starts first and must log
`serving DeepEP id` before the other nodes launch (their rendezvous connect
is not retried forever); two srun steps inside the allocation encode that
ordering. `scancel` is the whole teardown story. This removes the container
mount/NCCL-version/HF-snapshot-symlink failure class that cost three D
bring-up attempts in the same session.

Two operational facts any future fleet launch must respect:

- **Slurm state and GPU occupancy are separate ledgers on this cluster.**
  Nodes shown idle by `sinfo` carried full off-Slurm GPU loads (a vLLM TP4
  and another engine), and OOM'd the first fleet attempt at weight load;
  conversely this stack's own P/D trays look idle to Slurm. Verify
  `nvidia-smi` per node before submitting; do not trust partition state.
- **GPU-free ≠ node-clean**: idle leftover decode containers from prior
  fleet experiments still held the KV-P2P transfer port (50104) on three
  nodes and killed two more attempts with `Address already in use`. Sweep
  stale engine containers/ports before reusing trays.

**Multi-turn results on the EP16 fleet** (same 48×5-turn c16 workload as
the pool-sizing A/B below, plus a 128×5-turn c64 run; 64 GiB pool):

- c16: 240/240, zero failures, 948 tok/s out (EP4 same workload: 568),
  TPOT p50 ~11.5 ms (EP4: ~21) — 16 ranks at c16 sit mostly in bucket 1.
- c64: **640/640, zero failures, 1,158 tok/s out** — the historical 1P
  c64 saturation (steady ~24% deadline rejects on EP8, 8 GiB era) is gone;
  worst per-turn TTFT p99 8.5 s stays under the 15 s deadline.
- ITL: p50/p90 improve over EP4 (30.9/37.6 ms vs 41.8/47.6 at the same
  per-rank load) and barely move c16 → c64, but **p99 degrades** (52 →
  98 ms at c16, 73.7 at c64): the fixed-cadence DeepEP chain couples all
  16 ranks, so one rank pausing for an admission restore (the per-turn
  full-history refetch) taxes every rank's ITL tail. Session-affinity
  routing would shrink both this tail and the TTFT growth at once.
  **Confirmed by control run**: the same fleet under single-turn
  decode-heavy c16 (no mid-flight admissions) collapses ITL p99 98 → 40 ms
  with p90 at 32 — the tail is admission-coupled, not decode-plane. The
  admission-restore path still blocks the engine loop on the
  coordinator-era assumption ("every rank is joined, so blocking on the
  load is safe" — `scheduler/offload.rs`), which free-running invalidated;
  moving restore install off the engine thread is the code-side fix,
  session-affinity routing the traffic-side one.
- **New lever surfaced — per-turn TTFT grows with fleet width**: EP16 c16
  TTFT p50 climbs 455 → 912 ms across turns while EP4 stays flat. With 16
  ranks a conversation's next turn almost never lands on the rank holding
  its KV (~1/16), so every turn full-history-refetches over RDMA, growing
  linearly with history (~130 MB by turn 5). At c64 this compounds with
  the single-P queue into 2.9 → 7.8 s p50 by turn 5. Wide EP decode fleets
  need session-affinity / cache-aware routing (the qwen3 KV-aware-routing
  result) before this shape is production-shaped.

### Multi-turn chat bench: 8 GiB pegaflow pool is a deadline-reject root cause (2026-07-31)

First multi-turn (growing-history) load through the router: `vllm-bench`
`openai-chat`, 48 conversations × 5 turns, turn-1 input 1,024 + 256/turn,
output 128/turn `ignore_eos`, c16, temp 0, P TP4 tray03 + D EP4 tray04
(`85fcc386`). Artifacts
`bench-results/glm52-pd-multiturn-c16-1024-256-128*.json`.

- **Prefix cache on P confirmed under multi-turn**: with history growing
  1,152 → 2,672 tokens, turn-2+ TTFT p50 stays flat (~270–660 ms across all
  runs) — suffix-only prefill; 240/240 turns, zero failures, TPOT ~21–23 ms.
- **The stack script's `KV_OFFLOAD_HOST_GIB=8` default starves the D-side
  fetch path at this load**: pegaflow fetch chunks are 256 MiB, and pool
  exhaustion (`failed to allocate fetch chunk … on NUMA0`) drops the RDMA
  connection and cascades into 15 s-quantized handoff rejects
  ("retry via P") — turn-3/4/5 TTFT p99 hit 28.7/30.1/15.1 s and aggregate
  output throughput sagged to 437 tok/s. Earlier c64 GSM8K reject rates in
  this log likely include this cause, not just prefill capacity.
- **64 GiB pool (both sides) clears it**: zero allocation failures, zero
  deadline rejects, worst per-turn TTFT p99 ≤ 6.3 s, output 556–568 tok/s
  (+27%). Both trays have ~750 GiB free DRAM; 64 GiB pinned is cheap.
- **Decode plane is clean throughout; the jitter is TTFT-side.** ITL is
  flat across all three runs (mean 38–41 ms, p90 ~47, p99 ~52) — no decode
  tail even during the 8 GiB deadline storms. ITL p50 ~41 ms vs TPOT p50
  ~21 ms is native MTP's burst delivery (~2 accepted tokens per verify
  round on random prompts), not a discrepancy.
- **Remaining signal — TTFT p99 jitter**: an episodic multi-second stall
  (one wave of requests; 2.3 s p50 / 6.3 s p99 on the affected turn) moves
  between turns run to run (turn-2 in one run, turn-4 in the next) with
  clean D logs — consistent with the save/publish pipeline lever already
  named below, not with pool exhaustion. The c16 workload is fully
  phase-locked (temp 0, fixed lengths, no think time): all 16 conversations
  release + re-arrive simultaneously each turn, so a few-second publish or
  save-queue hiccup on P stalls a whole wave. Next probes: P-side
  save/publish timestamps around a stalled wave, per-conversation D rank
  placement / `cached_tokens` (rank-migration refetch), and a
  `--multi-turn-delay-ms` jitter A/B to break the phase lock.

### EP16 decode-only fleet and second-prefill (2P) scaling (2026-07-31)

Two scale-out experiments in the idle evening window (tray01/02/04/06/08
free; tray09–12 still held by the k3 job).

**EP16 standalone decode fleet** (tray01/02/04/08, 4 trays × 4 ranks,
rendezvous on tray01:19211, no P/metaserver/router — requests are served by
local prefill on the decode ranks):
`GLM52_PD_CONFIG=~/.config/pegainfer/glm52-ep16-decode.env
scripts/glm52_pd_stack.sh decode-only`.

- Bring-up incidents worth knowing about: jump-host disconnects killed
  script attempts 1–3 (every attempt restarts decode0, discarding peer
  progress), and tray02's 8-hour-old container lost GPU visibility
  (`Failed to initialize NVML: Unknown Error` → `CUDA_ERROR_NO_DEVICE`,
  host `nvidia-smi` fine) after a host-side driver event. `docker restart`
  on the container restored NVML; relaunching decode1 (ranks 4..8) by hand
  with the script's exact command let the in-flight fleet finish DeepEP
  init — all 4 endpoints healthy on the final attempt. Per-token decode
  speed was unaffected afterwards.
- 4-endpoint smoke: identical deterministic answers from all endpoints.
- **GSM8K (8-shot, temp 0, max_tokens 4,096, against the decode0 endpoint
  only): n200 c32 = 197/200 strict (0.985), 0 errors, wall 399 s** — parity
  with the EP8 P/D baseline (0.980). The throwaway harness in susun-dev
  needed a `cached_tokens` empty-list guard: standalone decode reports no
  prefix-cache hits.
- **vllm-bench against decode0 alone** (random IO, 512 prompts, c64, temp
  0, `ignore_eos`, 0 failed; artifacts
  `/mnt/shared/home/susun/bench-results/glm52-ep16-{decode,mid}-infqps-concurrency64-GLM-5.2-FP8-20260731-*.json`):
  - decode-heavy in=256/out=2,048: **943.7 tok/s out, TPOT p50 29.3 ms**,
    TTFT p50 75 s (local-prefill queueing at c64), 1,111 s wall.
  - mid in=1,024/out=512: **369.4 tok/s out, TPOT p50 29.1 ms**, TTFT p50
    73.6 s.
  - Read against the EP8 P/D stack: per decode endpoint the EP8 fleet did
    ~818 tok/s out at TPOT p50 38 ms on decode-heavy, so an EP16 endpoint
    is faster per token (native MTP, 29 ms) and per endpoint. On mid the
    EP16 endpoint collapses to 369 tok/s versus ~815/endpoint with a
    dedicated P — co-locating 1,024-token prefills with decode on the same
    4 ranks costs ~55% of output throughput. That is the cleanest
    quantification so far of what P/D separation buys.
- Not measured: fleet-aggregate throughput over all 4 endpoints (decode-only
  has no router; the bench targeted decode0 by design).

**Second prefill instance (2P)** behind the same router: P2 on tray06 (TP4
prefill-only, advertise 10.13.84.12:50103, same metaserver on tray03:50056),
router restarted with two `--prefill` endpoints round_robin over both P and
both D endpoints. Launched manually — `glm52_pd_stack.sh` still models a
single P.

- **GSM8K full 1,319 c64: strict 1,273 (0.9718), flex identical, 9 errors,
  finish stop 1,297 / length 13, wall 523 s** (artifact
  `bench-results/gsm8k-full-c64-2p-router.json`). The 9 errors were a P2
  warm-up transient in the first ~2 minutes. The same c64 load with 1P
  rejected ~24% of requests steady-state on decode1; 2P eliminates the
  saturation rejects and doubles the sustainable GSM8K-class envelope from
  c32 to c64 while holding accuracy.
- P2 bring-up gotchas (apply to any reused EP32-provisioned container):
  those containers were created without `--ulimit memlock=-1:-1`, so
  kv-offload RDMA MR registration fails — the tray06 container was rebuilt
  with the ulimit. Any container rebuild reverts libnccl to the image
  default; reinstall `libnccl2/libnccl-dev=2.30.7-1+cuda13.3` and verify
  `ncclCommQueryProperties` before starting (the stack script's
  `ensure_nccl` does this check).

### Full P/D-stack validation: router parity, GSM8K, load sweeps (2026-07-30)

Same deployment as the multi-process section above (P TP4 tray03, D EP8
tray13+tray14, vLLM router on tray03:10001, round_robin over both decode
endpoints), exercised end to end through the router.

- **Router parity gate**: for a prompt of N ids with anchor a,
  `router(ids, mt=6).text == D(ids, mt=1).text + D(ids+[a], mt=5).text` —
  byte-exact at N=148 and N=4,096 on BOTH decode endpoints (round_robin
  confirmed to hit decode0 rank=0 and decode1 rank=4, each admitting with
  `first_step=verify`). 4K router handoff ~352 ms versus ~29 s decode-local
  recompute. The gate script treats the anchor as the client's first
  generated token (`completion_tokens=6` includes it).
- **Fatal admission-capacity drift found by GSM8K and fixed** (`d791dffc`):
  `adopt_external_prefill_anchor` reclassifies the anchor from input to
  generated (`num_input_tokens -= 1`), so `RequestKv::lifetime_blocks()` lost
  one block of capacity exactly when `(input + output) ≡ 1 (mod 64)` —
  e.g. 1,601+4,096=5,697 → 90 blocks dropped to 89, tripping the
  admission `ensure` and fail-stopping the whole EP fleet (the second
  decode's `CUDA_ERROR_LAUNCH_FAILED` was collateral of the first decode's
  death). `RequestKv` now freezes its lifetime capacity at construction;
  regression test `external_prefill_anchor_promotion_keeps_lifetime_capacity`.
- **GSM8K (8-shot, temp 0, max_tokens 4,096, through the router)**:
  n200 c32 = 196/200 strict (0.980), 0 errors; full 1,319 c32 = 1,276/1,315
  strict (0.970), 4 errors. At parity with the single-instance
  admission-fix reference 1,280/1,315 (0.973). The 4 full-run errors were a
  startup transient: they were rejected while decode1 was still draining
  timed-out handoffs from the preceding c64 experiment; steady state at c32
  is clean.
- **Load ceiling: the 15 s handoff deadline is the first thing to bind.**
  `REMOTE_FETCH_DEADLINE` (scheduler/offload.rs) caps one request's
  remote-KV wait; a parked request that outlives it is rejected
  ("GLM5.2 native-MTP P/D handoff incomplete after 15.0s (full-page
  transfer)") and the client sees a 500 (the router runs
  `--disable-retries`). GSM8K-class load (sustained queue, long
  generations): c64 drove decode1 to a steady ~16 rejects/min (~24% of
  traffic, decode0 clean); c128 rejected ~83% of requests. One TP4 prefill
  cannot feed the handoff pipeline at those rates; decode1 (ranks 4..8)
  saturates before decode0. Sustainable envelope for that workload is
  ~c32. Raising the deadline only converts rejects into TTFT; the real
  lever is more prefill capacity (second P) or a faster save/publish
  pipeline.
- **vllm-bench random-IO sweeps through the router** (temperature 0,
  `ignore_eos`, 0 failed requests at every point; artifacts in
  `/mnt/shared/home/susun/bench-results/glm52-pd-*-20260730-165646.json`):
  - mid in=1,024/out=512: output throughput saturates past the knee —
    1,590 tok/s at c32 (TPOT p50 18.5 ms, TTFT p50 267 ms) versus 1,629
    tok/s at c64 (TPOT 37.5 ms, TTFT p99 7.4 s). c1: 164 tok/s, TPOT 5.1 ms.
  - prefill-heavy in=4,096/out=128: total (in+out) throughput plateaus at
    ~14.1k tok/s by c16–c64; TTFT p50 degrades 466 ms (c4) → 3.6 s (c16) →
    17.8 s (c64, queue-bound).
  - decode-heavy in=256/out=2,048: single stream 187 tok/s out (TPOT
    4.7 ms, native MTP on); fleet output ~1,640 tok/s at c64 with TPOT p50
    38 ms and TTFT p99 72 s. All 512 prompts completed.
  - The same c64 concurrencies that flooded GSM8K produced only ~2 rejects
    per bench run: the deadline binds under *sustained* long-generation
    load, not short random-IO bursts.
- **EP16/EP32 decode-scale attempt: blocked by cluster contention.** A
  4-node k3 job took tray09–12 and other tenants took tray04/08 within
  minutes of the free-machine scan (decode2/3 died on
  `CUDA_ERROR_OUT_OF_MEMORY` at `W13Weight` alloc). EP16 fleet was torn
  down; tray01/02/06 remain provisioned (`pegainfer-ep32-decode`, NCCL
  2.30.7 checked) for the next idle window. `scripts/glm52_pd_stack.sh`
  supports `D_TOPO=ep16/ep32` + `decode-only` for that rerun.

### Multi-process decode fleet: TP4 P → EP8 D across two trays (2026-07-30)

- First hardware run of native P/D against a **multi-process** decode fleet:
  P TP4 on tray03; D EP8 split tray13 (ranks 0..4) + tray14 (ranks 4..8)
  under `--glm52-rendezvous`; metaserver on tray03:50056; transfers over
  `mlx5_bond_0`; vLLM router v0.1.15 on tray03:10001. The bring-up used
  `scripts/glm52_pd_stack.sh`, extended for multi-process decode
  (`D_TOPO`/`D_HOSTS`, rendezvous gating, ssh-proxied health probes).
- **148-token gate byte-identical on both decode endpoints** (tray13
  handoff 194 ms vs 660 ms local baseline; tray14 220 ms vs 683 ms). The
  4,096-token/64-page gate is byte-identical on tray13 (104.5 ms vs
  14,607 ms local baseline, ~140×). Anchor, drafts, and tail key match the
  single-process EP4 run bit-for-bit.
- Both decode nodes restore independently: admissions logged
  `first_step=verify` with `committed_len=148/4096` on rank=1 (tray13) and
  rank=5 (tray14) — each D node registers only its local arenas and pulls
  rank-locally, hardware-verifying the free-running claim that cross-node
  KV offload needs no central arena registry.
- The 4K first verify round on EP8 D accepted **5 of 5 drafts plus the
  bonus token** (the single-process EP4 run took 4 of 5) — first observed
  full-proposal acceptance.
- Router-driven smoke through the P/D loop passes (`ok: true`, TTFT
  117 ms).

### Post-shell-split replay: free-running DP architecture (2026-07-30)

- Context: the DP coordinator was split into per-rank autonomous engines
  (`free-running-dp.md` migration step 2, branch
  `feat/glm52-free-running-dp-gates`, head `22d7d047`). The full P/D chain was
  replayed on that code to confirm the handoff contract survived the
  restructure: TP4 P on tray03 + EP4 D on tray04 + metaserver over
  `mlx5_bond_0` (RoCE), native MTP on both sides.
- **148-token gate: byte-identical.** P → D first 161 ms versus EP4
  local-prefill TTFT 680 ms.
- **4,096-token gate (64 pages): byte-identical.** The first verify round
  accepted 4 of the 5 forwarded drafts plus the bonus token (5 tokens total,
  `mean_accepted_drafts=4.000`); D log confirmed `first_step=verify`. P → D
  first 104 ms versus EP4 local-prefill TTFT 15,339 ms (~147×).
- One 75-token ambiguous prompt produced a deterministic fork (` point` vs
  ` paragraph`) between the P/D and local-prefill paths. This is the top-1
  near-tie class already declared in `native-mtp-accuracy.md` (logit-margin
  level, movable by bucket/topology numerics), not a handoff defect; all
  non-near-tie prompts matched byte-for-byte.
- This replay settles the earlier outstanding item ("hardware first-verify
  telemetry must be rerun after the state fix"): post-split, first-verify
  admission and initial-proposal acceptance are confirmed on hardware.

### Post-review first-verify and context-cap fixes

- Native D admission now marks the forwarded anchor as the current decode
  token, with the prompt fully fed and D-side completion still zero. The
  installed five-token proposal therefore makes the first target span
  speculative instead of a `PrefillBoundary` that clears the drafts.
- TP4 native-MTP prefill now reserves the four extra positions used after
  draft-1 by the fixed five-token proposal loop. Requests above
  `max_model_len - 4` are rejected at intake instead of failing the prefill
  engine inside proposal generation. Plain TP4 prefill keeps its original
  context limit.
- Release validation: the two focused regression tests passed, followed by
  the full GLM5.2 library suite (`97 passed`, `21 ignored`). The historical
  hardware TTFT rows remain useful transfer/latency measurements, but their
  initial-proposal acceptance claim is withdrawn until the EP4 handoff is
  rerun.

### TP4 P → EP4 D hardware handoff

- On the prefill node, TP4 prefill registered four rank-local PegaFlow
  instances with 101 arenas each under
  `pegainfer-glm52-l78-p64-mla656-idxk132-mtp1`. A five-token prompt returned
  target text ` Paris`, five draft IDs `[13, 576, 3283, 374, 1112]`,
  `committed_len=5`, `tail_len=5`, and a non-null 128-bit partial-page key.
- On the decode node, EP4 decode restored that page over RDMA from the prefill
  node's endpoint: one block, 101 arena slots, 3.3 MiB. Admission reported
  `cached_tokens=5` and
  `first_step=verify`; the six-token continuation
  `. Distance from Paris to Lyon` was byte-identical to an EP4 local-prefill
  baseline.
- The first cross-page gate found a producer-side bug before transfer:
  prompts longer than one 64-token page made the redundant small-M MTP
  boundary recomputation produce all-`-inf` argmax values and fail-stop the
  TP4 P engine. Both repetitive and ordinary prose reproduced it, so this is
  a page-boundary defect rather than an adversarial-text artifact.
- Staged finite-value logging localized the first invalid value to small-M MTP
  attention: `prepare` was finite, the historical MLA and index-K pages were
  populated, and the full indexer selected the correct physical slots, but
  attention returned NaN from element zero.
- Root cause: TP4's local FlashInfer decode backend consumes a statically
  quantized 576-byte cache row, while the P/D wire contract deliberately
  persists D-compatible `fp8_ds_mla` rows at 656 bytes. The original proposal
  loop interpreted the 656-byte transferable cache with 576-byte strides. It
  could appear to work inside the first page, then accumulated enough address
  error to fail at the first cross-page prompt.
- Fix: TP4 P owns two MTP MLA views. The 656-byte cache remains authoritative
  for PegaFlow/RDMA; a local 576-byte FlashInfer cache exists only for draft
  proposal. The large-M MTP pass fills both from the same raw layer-78 state,
  and synchronizes the layout-compatible index-K cache before proposal.
  Speculative writes and partial-page backup/restore touch only the local
  proposal view, so unverified state never contaminates transferable pages.
  Startup logs state both contracts explicitly, for example
  `execution_backend=FlashInferFp8 execution_bytes/token=576
  wire_layout=fp8_ds_mla wire_bytes/token=656`.
- The repaired 89-token gate returned target text `We`, drafts
  `[1184, 311, 387, 63141, 382]`, `committed_len=89`, `tail_len=25`, and 101
  arenas. EP4 D fetched one full block and the partial tail separately over
  RDMA (101 slots and 3.3 MiB each), admitted with `first_step=verify`, and
  accepted one draft in its first verify round. Its six-token result
  ` need to answer: "In` exactly matched a no-handoff EP4 baseline.
- After removing the staged finite-value diagnostics, the same TP4 producer
  gate was repeated with no debug environment variables and the normal
  CUDA-Graph-enabled server path. It returned the same target token, five
  drafts, committed length, arena count, and tail length.
- Completion text is not always a lossless way to construct the D request:
  appending the decoded anchor `We` to a prompt ending in `.` retokenized the
  pair as one `.We` token. The successful gate sent the original 89 prompt
  token IDs plus the anchor token ID, preserving the required
  `prompt = committed KV + anchor` length of 90. A router must forward token
  identity, not reconstruct the handoff boundary by concatenating text.

### Native P/D versus EP4 local-prefill TTFT

The A/B used a TP4 prefill node and EP4 decode node over the bonded RDMA
interface, with
native MTP enabled on both sides, a 4,096-token P chunk, and a 16,384-token
context cap. Each row is five deterministic, non-prefix-sharing token-ID
prompts after shape warmup at concurrency one and temperature zero. D was
restarted between the local-prefill and P/D phases so the P/D phase could not
hit baseline HBM or host cache. The 16K row uses 16,379 input tokens; its D
request asks for five rather than six output tokens to remain inside the
context cap.

`P first` is the first target token that a router can stream immediately.
`D handoff` starts when the router sends D the committed prompt, anchor,
proposal, and transfer metadata; it includes remote restore, admission, and
the first verify step. `P → D first` is the conservative latency if a router
waits for D's first new token before exposing any output.

| Input | EP4 local TTFT p50 | P first p50 | D handoff p50 | P → D first p50 | Combined delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 89 | 317.35 ms | 47.76 ms | 34.79 ms | 82.43 ms | −74.0% |
| 4,096 | 13,853.16 ms | 309.72 ms | 75.90 ms | 385.61 ms | −97.2% |
| 16,379 | 55,932.50 ms | 1,290.94 ms | 175.30 ms | 1,465.38 ms | −97.4% |

This is not evidence that disaggregation makes an optimized EP prefill
kernel faster: EP4 local prefill currently feeds the prompt through the
decode-oriented path. It does show the deployment-relevant result that TP4
large-M prefill plus transfer is dramatically cheaper than asking this EP4 D
worker to prefill locally. The additional latency after P has produced the
first token is 34.79/75.90/175.30 ms p50 for 89/4K/16K.

The transfer telemetry explains only part of that extra latency. At 4K, D
restored 63 full blocks (210.4 MiB across 101 arenas) in 11.2–11.7 ms, then
one 3.3 MiB explicit tail in 0.4–0.5 ms. At 16K, it restored 255 full blocks
(851.8 MiB) in 44.1–45.1 ms, then the tail in 0.4–1.1 ms. Every measured
request admitted with five drafts and `first_step=verify`.

The sweep found and fixed two long-context boundary bugs before producing the
table:

- **Exactly aligned committed length:** 4,096 initially advertised
  `tail_len=0`. PegaFlow fetched all 63 lineage-hashed full blocks in about
  11 ms, but D could only rematch 4,032 tokens and rejected after the strict
  15-second deadline. kvbm requires a dangling token after a sealed hash, so
  the final committed page remains an explicit tail even when it contains 64
  tokens. Native P/D now computes tail length in `1..=64`; the 4K metadata
  carries `tail_len=64` and a tail key. The
  `native_pd_tail_keeps_the_last_aligned_page_explicit` regression test covers
  both sides of the boundary.
- **Context-cap block table:** the first 16K P run panicked while copying 257
  page IDs into a 256-entry MTP attention table. kvbm may eagerly own one
  dangling generation page at the cap, but no valid position can address it.
  MTP now validates the requested logical page and copies at most the
  max-model-length table width. A 16,379-token TP4 P → EP4 D hardware gate and
  all five measured runs pass after the fix.

### Step 1: audit native-MTP physical page identity

- Created `feat/glm52-pd-mtp-arenas`.
- A first 101-arena registration passed the release library suite, but deeper
  inspection found it was semantically invalid and it was reverted before
  commit: target cache pages use BlockPool physical IDs, while native MTP
  currently uses fixed `1 + slot * pages_per_slot` IDs. Registering both under
  one content hash would therefore attach unrelated MTP bytes to a target
  page.
- Correct page-first storage needs two MTP regions: committed rows addressed
  by the target request's BlockPool page table, plus per-slot speculative
  scratch pages for proposal rows that target verification has not allocated
  or committed yet. A proposal crossing a 64-token boundary makes the scratch
  separation mandatory.
- Restore must also install MTP committed lengths; cache bytes alone do not
  establish continuity.
- Validation:
  - `cargo fmt --all -- --check` passed.
  - `cargo test --release -p pegainfer-glm52 --lib` passed in the development
    container
    with NCCL 2.30 from the installed Python wheel: 88 passed, 21
    GPU-dependent tests ignored.

### Implementation state

- The committed half of the page-addressing refactor is implemented:
  `RequestKv::current_page_indices` exposes only pages covering committed KV;
  the scheduler attaches that table to every MTP context append; layer 78's
  first pass now writes through those target BlockPool IDs. Focused release
  MTP scheduler tests pass.
- Finish the other half by replacing proposal-time fixed-slot addressing with
  explicit scratch pages. This is now implemented: two pages per slot live
  beyond the transferable BlockPool range; partial committed pages are copied
  before drafting, and aligned/unaligned boundary tests pass.
- The two layer-78 arenas now register only the committed allocation prefix,
  producing 101 transferable arenas while excluding proposal scratch.
- Make MTP committed length an explicit restore/install state, then enable
  the 101-arena PegaFlow path behind a native P/D contract.
- D-side reset/resume now derives the installed committed length from the
  first restored append position, after the layer-78 bytes have been restored
  under the same BlockPool pages.
- TP4 producer weight loading now admits native MTP: the resident pass loads
  layer-78 bookends/attention/router, and the existing topology-aware TP slice
  loader includes layer 78 for all routed/shared experts. Focused weight-plan
  tests pass.
- The producer execution boundary is now explicit: TP4 cannot reuse the EP8
  decode-MTP buckets. It needs a large-M context pass over every prefill row,
  using shifted prompt tokens (and the sampled anchor at the boundary), target
  final-normalized hidden rows, layer-78 attention/indexer cache writes, and
  the existing TP4 expert-slice MoE path.
- The large-M TP4 layer-78 context pass is implemented and hardware-validated
  on 4xGB300. A real prefill-only request returned `Paris`; the same pass ran
  during kernel preflight and request execution while writing the committed
  layer-78 MLA/index-K rows through target BlockPool page IDs.
- The prefill result now separates the target anchor from native-MTP proposal
  metadata. Layer 78's boundary residual goes through
  `shared_head.norm + vocabulary head` to produce draft-1, and all four TP
  ranks fail closed unless both the target token and draft-1 match. The
  4xGB300 request gate passes; the release library suite is 90 passed /
  21 GPU-dependent ignored.
- Draft-1 now continues through four scratch-page iterations to form the
  complete five-token initial proposal. The HTTP response carries versioned
  native-P/D metadata, and strict D admission restores all 101 arenas, seeds
  the proposal, and begins with speculative verification.
- The cleaned implementation passes the release library suite: 92 passed,
  zero failed, and 21 GPU-dependent tests ignored.
- Remaining deployment gate: repeat the same contract on a real EP16 D fleet.
  The arena geometry and MTP launch restrictions are topology-independent,
  but multi-node collective startup and end-to-end EP16 restore have not yet
  been exercised. A post-cleanup EP4 decode-node replay also stopped before
  serving
  at DeepEP `ncclDevCommCreate` with a system error, including with unlimited
  memlock; no request reached the cleaned D code in that attempt. The earlier
  successful cross-page EP4 handoff remains the functional evidence, while
  the machine-level NCCL initialization needs a separate rerun.

### Async admission restore closes the ITL p99 tail (2026-07-31, PR #801)

The D-side half of #799 is implemented and verified on the same 4-node EP16
fleet (Slurm job 5268, same trays/router as the 5193 baseline): admission
restores now submit the pegaflow host→GPU load and poll `LoadHandle::poll()`
at step boundaries while the request parks at its queue front — the engine
loop never blocks, so one rank's restore no longer stalls the DeepEP chain.
Abandoned loads keep their destination pages on a scrap list until the DMA
settles; a parked front's held pages are credited back to the admission
budget (double-counting would wedge the queue). The vLLM-compat P/D prefill
path was removed in the same PR.

Multi-turn 1024/256/128 × 5 turns, temperature 0, same workload as the
baseline rows above:

| metric | c16 old → new | c64 old → new |
| --- | --- | --- |
| ITL p99 | 98.0 → **30.7 ms** | 73.7 → **30.4 ms** |
| ITL p90 | 38.7 → 29.9 | 37.6 → 29.2 |
| TPOT p50 | 11.6 → 9.6 | 11.5 → 9.9 |
| TTFT p50 | 524 → 645 ms | 5.18 → 5.24 s |
| TTFT p99 | 1.28 → 1.73 s | 8.41 → 8.73 s |
| out tok/s | 948 → 1,018 | 1,158 → 1,156 |

Turns completed 240/240 and 640/640, zero rejects. ITL p99 now sits within
~3 ms of p50 at both concurrencies — the admission-coupled tail is gone
entirely, and the whole decode plane runs smoother (TPOT −17%). The cost is
the predicted TTFT tax at low load (each restore leg pays one extra
step-boundary round trip, ~+120 ms p50 / +440 ms p99 at c16); at c64 the
queueing term dominates and TTFT/throughput are unchanged. The per-turn
TTFT growth from rank-migration refetch remains — that is the
session-affinity item, not this fix.

### Poll-everything offload + TP4 proposal repair (2026-07-31, PR #804)

PR #804 closes #799/#802/#805 in one change: pegaflow bumped to 0.23.5, the
shim rebuilt around pollable `OffloadHandle<T>` handles (`submit_query` /
`submit_save` / `load`), the D-side admission restore extended to poll the
*query* leg too (`FullQuery → FullLoad → TailQuery → TailLoad`, one settled
handle per admission attempt), and the P-side `save_native_tail` detached
(the finished request's KV parks on a tail-save list until the D2H settles).
Handoff queries pass pegaflow's new `wait_for_full_prefix=true`: D cannot
recompute a miss, so the query holds `Loading` until the full prefix is
fetched instead of surfacing a useless partial hit.

**#805 found and fixed in the process**: #797 removed the LL decode paths
and left the TP4 native-MTP proposal forward bailing — the P role died
fatally on its *first* handoff request on any current build. This went
unnoticed because P validation kept running a pre-#797 binary. Rebuilt on
the prefill machinery (attention o_proj partial + layer-78 MoE both cross
the NCCL prefill all-reduce; the TP4 proposal body runs eagerly so the
collectives stay out of CUDA graph capture). Verified: greedy outputs
byte-identical to the pre-#797 binary through the router, proposals produce
`drafts=5`, D admits `first_step=verify`.

Multi-turn 1024/256/128 × 5 turns, temp 0, TP4 P + 4-node EP16 D fleet,
whole stack on the PR binary (artifacts
`bench-results/glm52-pd-multiturn-*-ep16-asyncall-fullstack.json`):

| metric | c16 | c64 |
| --- | --- | --- |
| turns completed | 240/240 | 640/640 |
| ITL p50/p90/p99 | 31.7 / 32.5 / 32.9 ms | 32.4 / 33.3 / 33.7 ms |
| TTFT p50/p99 | 0.64 / 1.42 s | 5.11 / 8.26 s |
| out tok/s | 871 | **1,189** |

A same-fleet control (identical trays, pre-async D binary,
`*-asyncall-oldD-control.json`) measured ITL p50/p90/p99 =
36.0/44.4/48.7 ms — the PR binary improves the whole ITL distribution on
identical hardware (p90 −27%, p99 −32%) and c64 sets the best throughput
and TTFT/e2e p99 of any run of this workload. Do not compare ITL absolutes
against the #801-era rows above: one fleet node differs, and the
fixed-cadence DeepEP chain paces every rank at the slowest node (~+4 ms
ITL p50 fleet-to-fleet). The c16 TTFT park tax (~+0.1 s p50) persists —
restore pipelining (deferred in #802) is the code-side answer,
session-affinity routing the traffic-side one.

### Prefix cache under native MTP: multi-turn TTFT decoupled from history (2026-08-01)

TTFT diagnosis on a 2P (TP4) + 1D (EP4) stack, multi-turn 1024 + 256/turn ×
8 turns, 2-token outputs (`max_tokens=1` finishes at admission via anchor
replay — it exercises the restore but skips the verify step and its admit
log, so TTFT probes must use ≥2): per-turn TTFT p50 grew 243 → 374 ms at c1
and 432 → 1,186 ms at c8. Decomposition: bs=1 direct-P prefill is
`~80 ms + len/9.2K tok/s`, the handoff adds ~72 ms — the growth was P
re-prefilling the FULL history every turn, and the c8 blowup was P's
serial single-request pool. Root cause of both: the v1 native-MTP cache
salt hashed the whole prompt, giving every continuation its own cache
universe — zero cross-turn reuse anywhere (P radix, D radix, host tier).

Fix (same-day): constant v2 salt + prefix matching enabled under native
MTP + the prefill pool sized for the full slot count. Layer-78 KV rides
the same pool page ids as the main cache, so a radix hit reuses it for
free; the accepted alias (two prompts diverging exactly at a page boundary
share one shifted-token L78 row) affects draft quality only — target
verification rejects any draft it misleads. Verified on the same stack:

- greedy 2-turn goldens byte-identical across the flip;
- c1 per-turn TTFT p50 `[303, 271, 270, 270, 271, 271, 302, 281]` — flat
  (was 243 → 374 and linear in history; at 8K histories the projected gap
  is ~3×);
- c8 per-turn TTFT p50 turn-7/8 `454/374` ms (was `1,186/993`), p99 flat
  ~600–900 ms (was growing to 2,247) — concurrent prefills replace the
  head-of-line queue;
- 24/24 conversations, zero failures, D admits on every turn.

The bare-metal P deployment also surfaced #810 (two NCCL copies corrupt
the TP4 comm when the shared lib dir lacks the unversioned symlink) —
fixed operationally, hardening tracked there.

## Debrief

The transferable cache format and the producer's fastest local execution
format are different contracts. Keeping one buffer and relying on identical
logical dimensions hid a physical-stride mismatch until a page boundary.
Future P/D additions should log both wire and execution layouts at startup and
must either prove them byte-identical or make the conversion boundary explicit.

The post-review state bug also showed that a log describing the intended next
step is not evidence of the slot's actual span kind. First-verify gates must
assert the state transition or observed speculative span.

Next action: the P → EP16 handoff contract is now hardware-closed (see the
Slurm fleet section above), which also covers the fleet-aggregate
measurement through the router. What remains: (1) EP32 handoff rerun when
eight trays are actually free (Slurm-idle is not evidence — verify
`nvidia-smi` per node); (2) **#799/#802/#805 are closed by PR #804** (poll-everything section
above): D-side query+load park-and-poll, P-side tail-save detach, save
submit-depth audited, TP4 proposal path repaired;
(3) session-affinity / cache-aware decode routing — the per-turn
full-history refetch is the dominant multi-turn TTFT term on wide fleets
(tracked in #799's discussion as the traffic-side mitigation; restore
pipelining is the complementary code-side lever, deferred in #802); (4) fold multi-P and
the Slurm fleet launch into one operational entrypoint (the sbatch script
and `glm52_pd_stack.sh` are separate today).
