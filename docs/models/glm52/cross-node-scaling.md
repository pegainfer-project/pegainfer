# GLM5.2 cross-node scaling: DP16 design, and the road to DP32/64

> **TL;DR:** The NVL72 variant of the data plane is proven: on GB300 NVL72 the whole rack is
> ONE NVLink/IMEX domain, so the existing single-LSA DeepEP shim works cross-tray with
> **no GIN un-baking at all** — live-verified up to **EP32 across 8 trays**: bucket-1 solo p50
> is flat at ~23.3-23.6 ms from EP4 loopback through EP32 (intra-rack width tax ≈ nil),
> bucket-8 EP32 c256 9072 tok/s, greedy text coherent through the serving path (EP8). **EP32
> serving is production-shaped (2026-08, #812 stages 2+3):** 8-tray decode fleet + 3x TP4
> prefill + router, throughput profile `GLM52_DECODE_SLOTS=16 GLM52_MTP_DRAFTS=2` (same 48-row
> step as the 8x5+1 latency profile) — fleet c512 through the router: **626 tok/s per D-GPU,
> mean TPOT 19.8 ms, zero failures** (decode-only endpoints: 522 at c64/node, 559 at 1.5x
> oversubscription). Width remains a KV-capacity lever, not a throughput one. EP widths
> {4,8,16,32,64} each get a constexpr shim instantiation + `Glm52MoeTopo` variant; the
> weight-only expert chain is ABI-generic so a new width is a config header, not new code.
> The GIN scale-out sections below remain the design for IB/RoCE clusters beyond one rack.
> **架构方向更新(2026-07):** 本文的 Event plane(facts plane)与 SMR coordinator 方向被
> `free-running-dp.md` 取代——长期架构是删除 coordinator 的 per-rank 独立 engine;本文的
> NVL72 实测数据仍有效。**rank-host 已随 free-running 迁移第 2 步从树上删除
> (2026-07-30):跨节点 = 同一 binary 每节点一进程(`--glm52-ranks` +
> `--glm52-rendezvous`),framed-TCP 控制面不复存在;下文 rank-host 章节留作实测记录。**
> Pitfalls that cost real time: the IMEX channel device must be passed into containers
> (`--device /dev/nvidia-caps-imex-channels/channel0` — `--gpus` alone silently breaks
> cross-tray LSA), and a remote-node teardown must `shutdown()` the socket, not just drop
> the writer (a reader clone keeps the fd alive → no FIN → both sides hang).
>
> **Last touched:** 2026-08

## Scope and status

- **Status: control plane + NVL72 data plane shipped** (`feat/glm52-rank-host`); the IB/RoCE
  GIN data plane below stays design-only. Its gate is hardware (two GIN-capable
  IB/RoCE-connected 8×H200 nodes) plus the data-plane microbench below coming back with an
  acceptable number.
- **What scales: the EP path.** Cross-node DP-N means attention stays data-parallel per rank
  and MoE goes EP-N. The TP8 topology (`--moe-topo tp8`, the solo-latency winner) is pinned to
  one NVLink/LSA domain by construction and does not cross nodes; a cross-node deployment is
  `DP-N/EP-N`, extending today's EP8 high-throughput launch configuration.
- **Why bother vs. two independent DP8 islands** (the null hypothesis any DP16 bench must
  beat): (a) expert weights exist once instead of twice — the fleet streams roughly half the
  expert bytes for the same token load, and routed-expert weight streaming is the decode wall
  (see `moe-gemm` history); (b) halving per-GPU expert residency frees HBM for KV → bigger
  admissible batches per rank. Both wins grow with scale; both are throughput plays. Solo
  latency *loses* cross-node (added a2a hops), full stop.

## Shipped: rank-host on GB300 NVL72 (2026-07-12)

One NVL72 tray = 4 GPUs; the rack is a single NVLink/IMEX domain, so DeepEP's single-LSA
assert holds across trays and the shim needs nothing new. Verified 2-tray EP8 recipe
(all inside the `susun-dev` container, binaries + weights on the shared `/work` mount):

```bash
# remote tray: dumb rank-host, hosts ranks by the coordinator's instruction
CUDA_VISIBLE_DEVICES=0,1,2,3 EP_DISABLE_GIN=1 \
  pegainfer --glm52-rank-host 0.0.0.0:19000

# coordinator tray: 4 local ranks + 4 remote
glm52_step_bench --model-path <GLM-5.2-FP8> --moe-topo ep8 \
  --rank-hosts <tray04-ip>:19000=4 --buckets 1,8 --steps 96 --warmup-steps 24
```

Measured (GB300, weight-only chain, whole-step graphs), same command with only
`--moe-topo`/`--rank-hosts` widened; EP16 = 4 trays, EP32 = 8 trays:

| Topo | trays | bucket-1 p50/p99 ms | bucket-8 p50/p99 ms | bucket-8 tok/s |
|------|-------|--------------------|---------------------|----------------|
| EP4 (loopback) | 1 | 23.45 / – | – | – |
| EP8 | 2 | 23.58 / 23.68 | 34.85 / 71.03 | 1837 (c64) |
| EP16 | 4 | 23.29 / 23.91 | 28.90 / 57.83 | 4430 (c128) |
| EP32 | 8 | 23.31 / 24.29 | 28.22 / 55.46 | 9072 (c256) |

Bucket-1 p50 is flat across widths — intra-rack width tax ≈ nil (per-rank expert bytes
shrink with width and cancel the wider a2a). 32-rank DeepEP contexts come up in ~7 s.
Two open items: bucket-8 p99 has a reproducible ~2× bimodal tail at every width ≥ EP8,
unexplained; and the context-cap ledger doesn't count width-scaled DeepEP buffers
(`kNumRanks × kDecodeMaxTokens`), so wide-EP runs need an explicit `--max-model-len`
until the ledger learns about comm buffers.
`pegainfer --rank-hosts ...` serves the same topology over HTTP (greedy prose + code checked).

Operational contract:

- Weights load per CONNECTION (~30 s warm page cache, ~95 s cold): the rank-host derives its
  own load bundles from the wire hello; nothing model-specific is configured on it.
- Teardown: coordinator drop `shutdown()`s the socket → rank-host sees EOF → fail-stop
  worker teardown bounded by a 60 s watchdog (`exit(2)` if the collective destroy lacks
  peers). The rank-host serves ONE connection and exits: worker drop does not return all
  hosted GPU state (281 GiB/GPU stayed resident after a clean close), so process exit is
  the release mechanism — wrap in a restart loop for a persistent node.
- Container must receive the IMEX channel device (`dev-container.sh` does this): without
  `/dev/nvidia-caps-imex-channels/channel0`, NCCL init succeeds but DeepEP `ctx_create`
  fails cross-tray — some ranks error, others block in bootstrap.
- After any tray-side `git pull`, verify `git rev-parse HEAD`: the egress proxy can flake
  mid-pull and leave a stale checkout that builds fine (a chain-selection bug hid there for
  one full bench cycle).

## Topology

```
DP16 = 2 nodes × 8 H200
  attention: DP — each rank owns its slots' KV (8 slots/rank, GLM52_MAX_BATCH_PER_RANK)
  MoE:       EP16 — 256 routed experts / 16 ranks = 16 local experts per rank
  intra-node fabric: NVLink (NCCL LSA domain)
  inter-node fabric: IB/RoCE via NCCL GIN (GPU-initiated networking)
  global step:  ≤128 rows (16 ranks × 8 slots), one bucket, lock-step as today
```

The scheduler contract is unchanged: every rank enters every MoE dispatch/combine with the
same global row count, so every scheduling decision is global (see "Scheduler semantics").

## Data plane: DeepEP v2 GIN scale-out — the real wall

### What we actually run (correcting an easy misconception)

The shim (`pegainfer-kernels/csrc/deepep/deepep_shim_impl.cuh`) vendors the **DeepEP v2
(elastic) kernel family** — `deep_ep/impls/dispatch.cuh` / `combine.cuh` — on the **NCCL
device API** backend: `ncclDevComm`, symmetric windows (`ncclMemAlloc`), and GIN. v1's
split into intranode (IPC/NVLink) and internode (NVSHMEM/IBGDA) kernel families does not
exist in v2: one kernel family covers both fabrics, and cross-node traffic goes through GIN.
The transport dependency stays NCCL (≥ 2.30.4); NVSHMEM never enters the build.

The GIN plumbing is **already constructed today**: `ctx_create` requests GIN device contexts
(QP count, queue depth, signal budget, `NCCL_GIN_CONNECTION_FULL`) because the v2 kernels
take the handles even when all traffic is NVLink — that is why NIC-less machines need
`EP_DISABLE_GIN=1` to boot.

### The four single-node bakes to undo

| where | what | cross-node change |
| --- | --- | --- |
| `deepep_shim_impl.cuh:279,363,418` | `scaleout_rank_idx = 0` hardcoded at the dispatch / combine / epilogue launch sites | plumb the real scale-out rank |
| `deepep_shim_impl.cuh:289` | `WorkspaceLayout(..., num_scaleout_ranks=1, ...)` | derive from topology |
| `deepep_shim_impl.cuh:543` | `ctx_create` asserts all ranks form one LSA (NVLink) domain — the explicit "no cross-node" gate | accept hybrid: `lsaSize = 8` per node × `num_scaleout_ranks = N` |
| `deepep_shim_impl.cuh:412` | combine reduce-epilogue takes the `num_scaleout_ranks == 1` passthrough | enable the real reduce path |

Upstream `backend/nccl.cu` has the hybrid (LSA + GIN) mode we deliberately cut when the shim
was written ("mirroring backend/nccl.cu for **non-hybrid** mode" — the shim's own comment).
The work is un-cutting it, not inventing it.

### Per-scale constants that must be re-derived

`deepep_config_glm52.cuh` is a DP1/EP8 snapshot. Scaling touches:

- `kNumRanks`, `kNumLocalExperts` (256/16 = 16 at EP16; 8 at EP32; 4 at EP64);
- GIN budgets: `kAllocatedQPs` / signal counts scale with scale-out rank count;
- the masked-GEMM worst case: today's `[32, 64, k]` slab assumes ≤64 global tokens can all
  route one row to one expert. At DP16 the global step is ≤128 rows, so the per-expert
  worst-case row bound doubles while local expert count halves — the slab shape, `bound_rows`
  derivation, and `GLM52_DEEPGEMM_MASKED_CAP` all re-derive. `kDecodeMaxTokens = 128` already
  has DP16 headroom; DP32/64 does not fit it.
- **wire format**: the shim dispatches bf16 (12,288 B/token payload at hidden 6144). Intra-node
  that was a simplicity win; cross-node it doubles IB bytes vs. DeepEP's native fp8 wire.
  Envelope math, DP16 decode, per node per step: 64 resident tokens × ~3 expected remote-rank
  copies (topk 8 over 16 ranks, half remote) × 12 KiB × 75 MoE layers ≈ **170 MB egress per
  step** each way. On 8×400G rails that is ~0.5 ms of wire time per direction plus per-op GIN
  latency × 150 collectives — a few ms on top of today's ~15–25 ms steps. fp8 wire halves the
  bytes term. These are estimates; the microbench replaces them.

### The go/no-go gate

Before any control-plane code: a two-node **dispatch/combine microbench** on real hardware —
the shim's own collectives at decode shapes (≤128 rows, topk 8), measuring added latency per
layer vs. NVLink-only. That number × 75 layers is the decode-step tax and decides whether
DP16 clears the two-islands null hypothesis at target concurrency. If it doesn't, everything
below stays on paper.

Hardware prerequisite to resolve first: whether `EP_DISABLE_GIN=1` on the 8×H200
verification node means "no usable NIC" or "GIN bug there" — it determines whether existing
machines can host the microbench at all.

## Control plane: framed-TCP hub-and-spoke

### One scheduler, not N

The coordinator (`pegainfer-glm52/src/scheduler/mod.rs`, `run_dp8_coordinator`) stays the
**single global scheduler**. DP shards *state* (each rank's KV, slots, requests); it does not
shard *control*: the EP contract makes the step bucket, step/skip, and launch-ahead all
global quantities, so a per-rank scheduler would retain zero local discretion. The
per-rank-schedulers-plus-consensus shape (vLLM/SGLang's DP coordinator, dummy batches, wave
counters) is what this design deletes: one state machine computes the global answer and fans
out, instead of N state machines negotiating it every step. `scheduler/` (admission / plan /
slot) is the pure decision core and does not change; only the transport under
`Glm52RankWorker` does.

### Deployment shape

```
node A                                   node B
┌───────────────────────────┐            ┌─────────────────────────┐
│ frontend + coordinator    │            │ rank-host (dumb)        │
│  ├─ crossbeam ► ranks 0-7 │───TCP────► │  demux ► crossbeam ►    │
│  └─ RemoteRankWorker ×8   │ ◄──frames──│         ranks 8-15      │
└───────────────────────────┘            └─────────────────────────┘
```

- **`rank-host`** is a subcommand of the same binary: spawns 8 unmodified `Glm52RankWorker`
  threads, one blocking demux loop (frame → local crossbeam send; response → frame back). No
  scheduler, no HTTP, no decisions.
- **`RemoteRankWorker`** exposes the same typed surface as `Glm52RankWorker`
  (`step_async → Receiver<Result<T>>` etc.), so `submit_and_join_step` / `run_draft_round`
  don't change. Raw bytes live in one module: encode, FIFO response matching, event demux.

### Protocol

1. **Framing**: one persistent TCP connection **per node** (a node is one failure domain; a
   single connection's global FIFO is strictly stronger than 8 independent channels), frames
   are `len(u32) ‖ bincode(Frame)`, `TCP_NODELAY` on. Two frame kinds:
   - `Response { worker_id, body }` — command plane, strict FIFO, exactly-one-per-command
     (the existing `Glm52RankCommand` contract, serde-derived; there is no second schema to
     drift);
   - `Event { body }` — facts plane, unsolicited push, no sequence, no reply (next section).
2. **Handshake as door**: on connect exchange git hash + `moe_topo` + config digest; mismatch
   = reject. Both ends ship in one repo at one commit — schema evolution is a non-problem by
   construction, not a handled one.
3. **Fail-stop, no resilience**: connection reset, frame decode error, or per-step response
   timeout (5 s; the step cadence is the heartbeat) → `fail_step` → whole-engine teardown,
   exactly like a worker-thread death today. No retry, no reconnect, no buffering. The 5 s
   timeout surfaces root cause two orders of magnitude before DeepEP's ~100 s device trap.
4. **Rejected transports** (litigated once, kept for the record): *RDMA verbs* — µs latency
   and zero-copy buy nothing on a KB/s control plane; QP state machines and error semantics
   are strictly worse than a TCP reset for fail-stop. *gRPC* — schema evolution, discovery,
   LB, TLS for a fixed-membership pipe between one binary and itself; dead weight that hides
   frames inside HTTP/2 during debugging. *ZMQ* — auto-reconnect, silent buffering, and HWM
   drops are the exact inverse of fail-stop.
5. **No async runtime in the core.** The coordinator and workers are plain threads +
   crossbeam, deliberately; tokio stays in the HTTP frontend. A bare protocol confined to one
   module, carrying KB/s, with fail-stop semantics, has almost no failure surface — the
   danger in hand-rolled protocols is resilience logic, and there is none.
6. **Re-visit triggers** (the moment a framework earns its keep, written down so drift is a
   decision): runtime-variable membership; independently deployed endpoints; any requirement
   to retry instead of tear down.

### Command-surface fit (audited against `runner.rs:95`)

- Every `Glm52RankCommand` variant is plain data + one response — already wire-shaped.
- `SetupComm.unique_id` is a 128-byte `ncclUniqueId` — designed for cross-process broadcast;
  EP16's 16 ranks join one communicator with it unchanged. `tp_exchange` (intra-node pointer
  rendezvous) is `None` under EP topologies.
- `BuildModel`'s reply (`Vec<KvArena>`, device-pointer arena descriptors) must not cross the
  wire: the KV offload host becomes per-node; remote arenas register with the remote node's
  host locally, and the reply carries a serializable summary.
- Bring-up sequence is unchanged: `LoadWeights` all → `BuildModel` all → `SetupComm` all.

### Testing: loopback soak

Because `RemoteRankWorker` is interface-identical to `Glm52RankWorker`, the entire existing
single-node suite (contract tests, golden gates, e2e) runs unmodified with the 8 local
workers parked behind a `127.0.0.1` rank-host. The protocol layer gets soaked by real
workloads on one machine, before a second node exists. This is the first implementation
milestone, and it needs no new hardware.

## Facts plane: Event frames

Two kinds of traffic want different channels. The **command plane** is lock-step,
request-response, load-bearing. The **facts plane** is node-local asynchronous observations:
a D2H offload save landing (releasing pinned pages), a P/D KV transfer completing, a host-tier
restore finishing. Folding facts into step responses turns the hottest frame into a
god-message (every new async feature edits the step schema); giving facts their own protocol
stack is machinery for KB/s. The answer is the second *frame kind* on the same connection:

- **Producers push at the observation point.** The hooks exist: `SavePin::drop`
  (`scheduler/offload.rs:46`) *is* the D2H-landed callback; a load/transfer handle completing
  is the same shape. Pushing an `Event` frame is one write from that callback.
- **The coordinator drains a mailbox at step boundaries.** Under lock-step, an event arriving
  mid-step cannot take effect before the next boundary anyway, so a ≤1-step-stale mirror of
  remote facts is exactly as fresh as a local poll — machine 1 vs. machine 2 differ only in
  where the observation happens, not in effective latency. (One real exception: with
  launch-ahead planning, a mid-step event can catch step N+1 where a join-time piggyback only
  catches N+2 — push wins one step of TTFT.)
- **Idle engines stay reachable.** The coordinator's idle wait becomes a select over
  {new requests, event mailbox}; the "step cadence is the heartbeat, but there are no steps"
  hole closes itself.
- **Same connection, same fate.** A separate facts connection could die while the command
  connection lives — the coordinator would keep scheduling on a silently stale mirror. One
  connection makes staleness impossible and keeps fail-stop atomic.

Rule of thumb, stated once: **every node-local async fact travels as an Event; the step
response carries tokens and nothing else.**

## P/D disaggregation interplay

From the decode scheduler's view a P→D KV transfer is **a prefix hit that hasn't arrived
yet** — the async sibling of the M1 host-tier restore. The state machine grows exactly one
pre-admission state:

1. Request arrives with block hashes; the coordinator sends `StartKvPull { hashes }` to the
   target rank's node. The rank-host reserves target pages (`reserve_loaded_blocks` shape),
   initiates the pegaflow pull, acks **immediately** ("reserved, in flight"). Unlike M1's
   blocking `restore_host_prefix` (safe only because local D2H is fast and bounded), nothing
   ever waits: a stalled coordinator stalls every rank's decode under lock-step.
2. The request sits in a **waiting list separate from the per-rank `pending` queue** — no
   head-of-line blocking; later requests admit normally. Reserved pages hide from admission's
   full-lifetime budget via the `pinned_blocks` pattern (`scheduler/mod.rs:443`).
3. Completion pushes an `Event`; the next step boundary admits the request into a slot.
   Failure/timeout pushes an `Event` too: release the reservation, degrade to a cold local
   prefill. **Cache maintenance, never a correctness dependency** — the `offload.rs` posture
   survives the network hop, and a transfer failure is *not* fail-stop (only protocol/step
   failures are).

The prefill engine needs nothing new: it is an independent engine instance that publishes
block hashes into pegaflow and exits the story. P↔D request routing is the frontend's
business; the scheduler-visible contract is only "blocks with these hashes will appear".

## Scheduler semantics: unchanged at every scale

`plan_step_shapes`, `admission_target`, launch-ahead flags are pure functions over the
`[rank][slot]` global snapshot; 8 → 16 → 64 ranks is a constant, and planning cost
(microseconds, O(ranks × slots)) never approaches the step budget. Control fan-out/join has
never appeared in a step profile (the D7 dissection's culprit was `cuGraphLaunch`, not
channel joins). What actually varies per scale is below.

## Scaling to DP32 / DP64

DP32 = 4×8, DP64 = 8×8. Nothing above changes shape; four things change weight:

### 1. Data plane pressure (dominant)

- Per-expert worst-case rows grow with the global step (≤256 / ≤512 rows) while local
  experts shrink (8 / 4) — slab shapes, `bound_rows`, GIN QP/signal budgets re-derive per
  scale; `kDecodeMaxTokens = 128` must grow with them.
- **EPLB becomes load-bearing.** At 32 local experts, routing skew averages out; at 4 experts
  per rank, one hot expert makes one rank the whole fleet's critical path (lock-step means
  the slowest rank *is* the step time). Expert-placement rebalancing, deferred so far,
  graduates from optimization to prerequisite somewhere between EP16 and EP32.
- fp8 wire stops being optional; rail utilization and NIC-per-GPU mapping become first-class
  microbench axes.

### 2. Blast radius (operational ceiling)

Fail-stop over N nodes multiplies fleet-fatal events by N: any NIC flap, any single-node
maintenance, any step failure kills all 64 GPUs' state. Wide-EP deployments elsewhere share
the property; what changes is recovery economics — at DP64, warm-restart time (weights
reload ~180 s + KV loss) times event rate is the real availability number, and P/D + pegaflow
host tiers (warm KV survives engine death) become the mitigation, not Raft-style redundancy.

### 3. Batch economics

DP64 earns its keep only with ~512 concurrent decode rows of real traffic. Below that,
padding rows burn the collectives and two smaller islands win. The null hypothesis scales
with N: DP-2N must beat 2 × DP-N at the target concurrency, measured, every time N doubles.

### 4. Control plane: the SMR transition

Hub-and-spoke's per-step cost is one command/response RTT to each of N−1 remote nodes,
joined in parallel — max of N−1 RTTs, tail growing slowly with fan-out. At 2 nodes and
15–45 ms steps this is <1% and permanently fine. Two developments change the verdict: steps
optimized into single-digit ms (RTT tail becomes a visible fraction), or node count past
~4 (tail + head-of-line on the coordinator's NIC). The upgrade is **not** per-rank
schedulers — it is replicating the one deterministic scheduler:

> **SMR design (preserved from `dp16-smr-coordinator.md`).** The scheduler is already a
> deterministic state machine over exactly two input streams: the **admission stream**
> (which requests, what order, effective at which step boundary) and the **step-output
> stream** (every rank's tokens each step). Same code + same streams ⇒ byte-identical
> `slots`, shapes, and flags on every replica — the Calvin shape: *sequence first, execute
> with zero coordination.*
>
> - A **sequencer** (the frontend, demoted) stamps a total order on admissions and cancels;
>   it does not pick rank/slot — placement (`admission_target`) is computed identically
>   inside every replica. The broadcast must be *ordered*; admission takes effect at a
>   declared boundary ("R joins at step N+2") to stay off the critical path.
> - **Every replica simulates the whole fleet, executes only its node's 8 ranks.** Step
>   outputs are **allgathered** (≤8 u32/rank, piggybacking a GPU path that already runs 150
>   collectives per step), not routed through a master. Per-step cross-node control traffic:
>   zero. Coordination cost drops from per-step to per-request.
> - **Determinism discipline**: no replica may read node-local state on the decision path.
>   Current inventory of violations, each of which becomes a sequenced event under SMR:
>   `token_tx.is_closed()` disconnect probes (`scheduler/mod.rs:461,771` → sequenced cancel);
>   offload timing on the admission path — `pinned_blocks` (`scheduler/mod.rs:443`) and the
>   host-restore outcome (`scheduler/offload.rs`); P/D transfer-completion Events. The Event
>   frames of the facts plane are already event-shaped: SMR-ifying them is routing through
>   the sequencer instead of point-to-point.
> - **Buys**: zero per-step control RTT, and — for free — a **deterministic replay log**:
>   persist the two streams and any desync or "byte-deterministic garbage, no crash" incident
>   replays through the state machine on a laptop, no GPUs. **Does not buy availability**:
>   replicas exist for locality, not redundancy; a failed step still kills the fleet.
> - The replay-journal fragment stands alone at any scale, including single-node today:
>   journal (admission, outputs) behind a flag + a replay harness asserting shape determinism
>   de-risks the discipline before any network code exists.

### Decision table

| condition | control plane |
| --- | --- |
| 2 nodes, steps 15–45 ms (DP16 bring-up) | framed-TCP hub-and-spoke — permanently acceptable at this cadence |
| steps single-digit ms, or >4 nodes | SMR replicas (design above) |
| desync forensics wanted at any scale | replay journal fragment, stands alone |

## Open questions

- The 8×H200 verification node's mandatory `EP_DISABLE_GIN=1`: absent NIC or GIN defect?
  Decides microbench hosting.
- fp8 dispatch wire in the shim (halves cross-node bytes; touches re-quant placement).
- Masked-slab growth vs. `GLM52_DEEPGEMM_MASKED_CAP` at ≥128 global rows — does the
  64-row-slab GEMM shape survive, or does the masked path need a per-scale variant?
- EPLB design for ≤16 local experts (prerequisite by EP32).

## Next step

None scheduled. The gate is hardware + the two-node dispatch/combine microbench; until that
number exists, this doc is the whole deliverable. First implementation milestone when it
opens: the loopback rank-host soak (no second node required), then two-node bring-up behind
the DET/oracle gates, then the DP16-vs-two-islands bench at target concurrency.
