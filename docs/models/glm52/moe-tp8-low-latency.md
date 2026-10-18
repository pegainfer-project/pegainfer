# GLM5.2 decode MoE: EP8 → TP8-sharded persistent kernel + LL allreduce

> **TL;DR:** Replace the bucket-1 decode MoE path — the DeepEP EP8
> dispatch/grouped-GEMM/combine chain, at ~103 µs/layer kernel-resident / ~145 µs/layer
> wait-inclusive the single largest bucket of the 19.5 ms solo step — with one TP8-sharded
> whole-layer persistent kernel per layer plus a low-latency (LL) packet reduce-scatter.
> P0 + M0 + M1 code are on `feat/glm52-moe-tp8`. M1 pilot (8 layers dual-resident) is
> functionally green (smoke, determinism ×2, coherent text, TP8 layer oracle gate); the
> per-layer win is real (**−63 µs/layer**, in-situ kernel 56.5 µs vs EP8 segment 110.1 µs,
> confirmed by nsys segment walls + untraced layer-count-sweep slope) but a fixed
> **F ≈ +0.87 ms/step** once-per-step cost — the step's *first* TP8 chain waits ~924 µs on
> cross-rank arrival skew (rendezvous wall), invariant across mapping topology, fence
> shape, kernel fusion, and layer placement — makes the pilot-8 perf gate structurally
> unpassable (8 × 63 µs = 0.5 ms < F). **M2 (launch-time `--moe-topo tp8|ep8`, all 75
> layers, bucket-1-only low-latency mode) PASSED on jz-38 2026-07-07: solo untraced ITL
> 19.23 → 15.27 ms/step (−3.96 ms, 1.26×, ×2 runs, p99 15.38), 8-concurrent diverse
> bucket-1 24.56 → 20.92 ms (1.17×, 326 → 382 tok/s), determinism ×2, coherent text,
> TP8 layer oracle gate green** — landing within 0.1 ms of the 15.4 arithmetic
> prediction. EP8 is *not* deleted: expert sharding is
> chosen at load time (weights are repacked during H2D anyway), so EP8 remains the
> high-throughput launch configuration while TP8 becomes the low-latency one.
> **M4 attention-TP (replicated activations over 8-head shards + o_proj LL allreduce
> + pad want-mask) measured 2026-07-08, same-day A/B (table in the M4 section): solo
> plain 15.27 → 13.75 ms (−10%), MTP code 188.6 → 220.9 tok/s (+17%, steady-state
> run-2 pair; run 1: 186.1 → 197.3), MTP text lossless; c8 diverse regresses
> 20.6 → 22.7 ms (KV-replication trade, still beats EP8's 24.6).** Inspired by
> the latency-first executor design of TileRT (persistent kernels, communication as graph
> nodes, no NCCL in the hot path); everything here is measured and implemented
> independently in this repo.
>
> **Last touched:** 2026-07

## Why topology, not more kernel work

The EP8 wait structure has been mined out (evidence in
`whole-step-decode-graph.md`):

- The MoE block (collectives + quant/SiLU glue + masked GEMM) is ~8.3 ms of the
  19.5 ms solo step. Dispatch/combine alone are 4.75 ms kernel-median /
  7.73 ms wait-inclusive.
- **Dispatch is rank-arrival-wait bound, not byte bound**: shrinking the payload to
  fp8 measured perf-neutral.
- Fused-MoE via DeepGEMM MegaMoE was measured and rejected: ~200 µs/layer structural
  floor at decode payloads, ~2× slower than the current chain.
- `kDecodeNumSms` tuning is flat.

What's left is the topology itself. EP8 partitions *experts* across ranks, so tokens
must travel to their experts and every layer waits for the slowest rank's arrival.
TP8 instead stripes *every* expert across all ranks (each rank holds a 1/8
intermediate-dim slice of all 257 experts): every rank does identical, uniform work
per layer and partials are merged with a reduce-scatter. **Per-rank expert bytes read
are exactly equal between the two shardings** — the entire gap is wait structure,
kernel boundaries, and small-op floors. A structural bonus: the per-layer combine
straggler spread (~44 µs/layer under EP8) disappears by construction.

## What changes, what doesn't

Only the **routed-expert weight sharding** changes, and only for the bucket-1 decode
graph at first:

| Plane | Today (DP8×EP8×TP1) | After this slice |
|---|---|---|
| Request/KV data plane | DP8 lock-step, one graph per bucket | unchanged |
| Non-expert stack (attention/MLA/indexer/dense/lm_head, ~19.6 GiB) | replicated per rank | unchanged |
| Routed experts (~85.5 GiB/rank) | 32 whole experts per rank (`weights.rs:275-290`) | 1/8 row-slice of all 257 experts (bucket-1 graph; pilot = 8 layers) |
| Shared expert | aux-stream fork/join around the routed path | folded into the persistent kernel (its I=2048 sliced 256/rank the same way) |
| Buckets 2/4/8 | EP8 chain | unchanged until the M3 decision |

Per-layer target form (replaces the 9-kernel EP8 chain + aux-stream shared expert at
`model/mod.rs:972-1012`); lock-step bucket-1 has `global_tokens = 8` (7 pads,
`model/mod.rs:890`), so the TP kernel processes all 8 global tokens (m=8) with a
uniform shape on every rank:

```text
① LL allgather: each rank pushes its token's post-attn hidden (12 KB bf16) to all peers
   (pad tokens pushed too — uniform shape, no owner branches)
② one cooperative persistent TP-MoE kernel (m=8):
   phase A: 8× rmsnorm + router (noaux_tc semantics: sigmoid+bias top-8, renorm ×2.5,
            computed redundantly on every rank)
   phase B: gate|up GEMV over the 256-row I-slice for 8 tokens — each weight load
            serves 8 rows (8× arithmetic intensity) — + SiLU
   phase C: down GEMV (m=8) producing 8 partials
   epilogue: LL reduce-scatter — token j's partial goes only to rank j;
             rank j sums 8 contributions + residual
```

## Evidence (standalone prototype, same node, jz-38 8×H200)

Measured with a standalone microbench prototype using the exact GLM5.2 shapes
(H=6144, E=256+1 shared, per-rank I-slice 2048/8=256, per-128 fp8 scale semantics).
The prototype kernels land in-tree with the M0/M1 PRs.

- One TP-shard MoE layer as a single cooperative persistent kernel: **30.5 µs cold-L2**
  (vs 38.1 for the best fused-3-kernel graph variant, vs 52 for an 8-kernel chain);
  CPU golden comparison max rel err 3.6e-3.
- LL packet one-shot allreduce with the epoch tag embedded in each 128-bit packet
  (zero fences, zero separate flags): radix-8 marginal cost **5.8 µs/layer**
  (radix 2/4: 1.5/2.5). Epoch advance is a device-side monotonic counter with
  odd/even payload double-buffering, so graph replay advances it without changing
  kernel parameters; 8-rank lock-step numeric identity checks pass.
- Total ≈ **36 µs/layer** (m=1 form) vs the EP8 chain's 103–145 µs/layer.

Arithmetic expectation: 75 MoE layers × (65–105 µs) ≈ **−5 to −8 ms/step** solo.
Per this repo's own calibration discipline, all claims settle on e2e A/B via
`pegainfer-server/src/bin/glm52_step_bench.rs`, never on trace proportions
(`whole-step-decode-graph.md`, calibration section).

## Constraints and standing decisions

1. **No dual layout inside one process.** The expert slab is ~85.5 GiB/rank; two
   layouts cannot coexist in 141 GiB HBM (this is why the loader packs experts at
   H2D time, `weights.rs:31-37`). Consequences:
   - The **pilot** dual-resides only N=8 layers (~+9.7 GiB/rank — fits).
   - Full migration makes sharding a **launch-time configuration** (e.g.
     `--moe-topo tp8|ep8`): the loader picks the slicing during H2D repack. EP8 is
     retained as the high-throughput configuration — at large buckets its wait cost
     amortizes per token while TP's thin per-rank GEMM (I-slice 256) loses efficiency,
     so EP8 plausibly stays the right choice there. The M3 measurement decides the
     default, not a code deletion.
   - Cost accepted: two MoE backends means two numeric gates (EP8 oracle exists; TP
     oracle is new in M1) and dual e2e coverage until/unless M3 retires one.
2. **TP slices align with fp8 scale blocks.** I-slice 256 = 2×128, so the W13 row-slice
   and W2 column-slice each cover whole 128-blocks — no scale-boundary straddling.
3. **Numeric contract is a tolerance gate, not bit-parity.** TP partial-sum order
   differs from EP8 by construction. Precedent: the batch-4/8 mma GEMV was accepted on
   `max_rel < 2e-2` + ≥90% bf16-exact + e2e greedy coherence + DSpark accept-rate
   parity (`whole-step-decode-graph.md`). TP-MoE follows the same recipe, plus a TP
   layer oracle gate modeled on `oracle/layer_ep8.rs`. Near-tie router flips need the
   same bounded-allowance mechanism the EP8 oracle already has (2 known outliers).
4. **Cooperative launch under stream capture has no in-repo precedent** but is
   verified on the same node/driver (the prototype's graph timing captures
   `cudaLaunchCooperativeKernel` via `cudaStreamBeginCapture`), and the DeepEP shim
   already replays cooperative+cluster launches inside the production graph. Residual
   risk is plumbing the cudarc/driver-API path, not feasibility.
5. **Two iron rules for the LL protocol** (validated in the prototype; do not relax):
   the epoch only ever advances device-side (graph parameters are frozen), and
   payloads are double-buffered by epoch parity (a fast rank may lead by at most one
   iteration). Spins get an upper bound + `__trap` — crash early instead of letting a
   half-paired collective ride the ~100 s DeepEP-style device timeout.
6. **Comm/stage buffers follow the caller-owned resident-arena convention**
   (`glm52_moe_gemv.cu:281-287` style): pointer-stable across capture/replay.

## Work breakdown (each step has an independent accept/kill)

| # | Content | Accept / kill |
|---|---|---|
| **P0** | Mechanism probes, before any kernel work — these are the only kill-level unknowns: (a) LL buffer `cudaDeviceEnablePeerAccess` coexisting with NCCL ≥2.30 symmetric-window registration in the real process shape (single process, 8 threads, primary contexts); (b) cooperative launch under cudarc stream capture + graph replay | Both probes pass on jz-38. **Kill: either fails → redesign before spending on kernels**. **(a) PASSED 2026-07-07** — standalone probe (`p0_probe_peer_nccl.cu`, lands with the M1 PR), 8×H200 / NCCL 2.30.7: symmetric window + allreduce, peer-access LL mailboxes, and 50 interleaved NCCL+LL rounds all verify in **both** init orders (NCCL-first and peer-first). Fidelity caveat: probe drives the NCCL *host* API over the window; the DeepEP shim's device API (ncclDevComm/GIN) coexistence is re-checked implicitly in M1 when both run in-process. **(b) PASSED 2026-07-07** — `p0_probe_coop_graph.cu`: cooperative launch via `cudaLaunchKernelExC`+attr (the DeepEP-shim launch shape) captured with `cudaStreamBeginCapture`, instantiated, **50 replays enqueued back-to-back with zero host involvement** — epoch advances via a device-side counter, parity double-buffered LL packets across all 8 GPUs, kernel self-verifies every round; NCCL window allreduce healthy before capture and after the storm. Kernel-design lesson baked into the probe: **no block barrier may sit in a thread-divergent branch of a kernel that also calls `grid.sync()`** — threads parked at the grid barrier never release a `__syncthreads`, deadlocking the block (the probe's first version hung on every coop variant this way; isolation matrix + rerunning the R4 ground truth pinned it) |
| **M0** | Kernel extension m=1 → m=8 in the standalone prototype (no pegainfer changes) | **DONE 2026-07-07, PASSED**: solo **55.2 µs/layer** (1.54× the m=1 anchor's 35.7 — under the 2× kill line; **6.9 µs/token = 5.2× per-token amortization**), diverse (E_active=22) 74.8 µs; CPU golden 0.0030/0.0044 < 5e-3 both configs; grid 264. Final form differs from the plan sketch in two load-bearing ways: (1) the compute engine is the **batch-8 m16n8k16 mma port** from `glm52_moe_gemv.cu` (σ-permutation, fp8→bf16 lossless, f32 tensor-core accum) — the scalar 8-token-reuse form measured 171 µs (instruction wall + occupancy); (2) activations stay in **global memory (L2-hot), not smem** — 96 KB smem capped occupancy and the mma B-fragment reads pointers anyway. Phases run as a global warp-job pool (expert × proj × 16-row tile × k-slice) with f32 partial scratch and fixed-order epilogues (deterministic). Two portable traps recorded: mma k-slices MUST be multiples of 128 (scale-fold contract, now static_asserted), and per-token top-8 + union compaction must not serialize on one block (was 41.5 µs of grid.sync parking; block-parallel top-8 + ballot-scan → 8 µs). Remaining headroom: phase B 20.1 µs vs ~7 µs weight roofline. Details: skeleton_bench `M0_实验记录.md` |
| **M1** | Pilot integration, N=8 layers dual-resident: loader TP-slice path (pilot layers only), LL buffers + epoch counters in a resident arena, launch FFI, TP layer oracle gate | `glm52_step_bench` bucket-1 A/B: **pilot 8 layers ≥ −0.4 ms total** (~−55 µs/layer, generous margin); oracle + determinism ×2 green. **Kill: < −0.2 ms** → attribute before proceeding. **Status 2026-07-08: functionally DONE (oracle gate included); perf kill-line hit and root-caused to a once-per-step rendezvous wall that pilot scale cannot amortize — proceed to M2 by design decision.** Wiring: second-pass TP-slice loader (`pegainfer-glm52/src/moe_tp8.rs`, pilot layers dual-resident, ~+9.7 GiB/rank correctly charged by the VRAM probe: cap 72576 → 49600), kernel rides the EP8 arm's graph slot for bucket-1 only (`--moe-tp8-pilot-layers`). Functional gates green on jz-38: 96-step smoke across 4 buckets × 2 tiers, determinism ×2 byte-identical, text coherent (one near-tie divergence at byte 55/685 vs EP8 — expected, different partial-sum order). In-situ kernel median **56.5 µs/layer** (= M0's 55.2 reproduced) vs EP8 segment wall 110.1 µs; the **−60 µs/layer** win confirmed by two independent methods (nsys segment walls + untraced layer-count-sweep slope). But untraced solo A/B first read **+0.36 ms** (kill line). Sweep over pilot = 0/1/2/4/8 (19.23 / 20.04 / 20.02 / 19.84 / 19.59 ms) decomposed it: ITL fits **19.23 + F + n·(c−w)** with a **fixed F ≈ +0.87 ms/step paid from the first pilot layer** and c−w ≈ −63 µs/layer. Four falsification rounds all measured 19.57–19.62, i.e. F is *invariant* across: LL mapping topology (single peer-granted VA vs per-accessor VMM mapping), fence shape, fused whole-layer kernel vs 8 phase-split plain nodes (447b7f1), and pilot layer placement (head 1–8 vs mid-model 41–48). nsys located F: the step's **first** TP8 chain waits ~924 µs on cross-rank packet arrival (the remaining chains wait ~60 µs) — a once-per-step cross-rank rendezvous wall, not a per-layer cost. An earlier bisection with empty-kernel variants had blamed the *existence of a peer-access grant* (page-table side-effect tax on the memory-bound EP8 GEMMs); that data was **poisoned** — empty kernels leave `mlp_out` garbage, which raises downstream routing entropy and inflates the EP8 segments, so those absolute ITLs are not comparable — and the grant-tax conclusion is downgraded to unproven. The per-accessor VMM mapping (bccf874: one `cuMemCreate`, per-accessor `cuMemMap` + single-device `cuMemSetAccess`, the NCCL-window shape) is kept as harmless hygiene, not as a measured win. Residual open question: the untraced EP8 baseline absorbs the same step-head arrival skew ~0.9 ms more cheaply than the TP8 chain does (traced EP8 first-dispatch also waits ~1.1 ms); node-traced A/B *inverts* the verdict (pilot-8 looks faster traced) because per-kernel tracing overhead inflates the ~12-kernel EP8 chain more than the TP8 chain — only untraced e2e ITL decides. Also falsified (all measured): grid 132 vs 264, the cooperative-launch attribute (a software sense-reversing grid barrier measures identical — kept, since it also removes the coop-launch constraint), graph-topology/aux-fork parity, and `cudaGraphLaunch` +94 µs as a cause (symptom: API duration includes enqueue wait). Verdict: pilot scale cannot pass the gate (8 × 63 µs = 0.5 ms < F ≈ 0.87 ms); the win scales with layers and F does not → M2 |
| **M2** | Launch-time `--moe-topo tp8\|ep8`: tp8 loads TP8 slice banks for all 75 MoE layers *instead of* the EP8 expert slabs and serves as a **bucket-1-only low-latency configuration** (1 slot/rank, fleet ≤ 8 concurrent, prefill rides spans at 1 token/step — the TileRT-style ≤8-batch positioning); ep8 stays the default high-throughput configuration. DeepEP init is kept (idle comm buffers) to avoid touching the collective teardown contract. Scheduler clamps bucket=1 (plan cap + admission 1 slot/rank + precapture filter) | solo untraced ITL **≤ 15.5 ms/step** (arithmetic: 19.23 + F ≈ 0.87 − 75 × 63 µs ≈ 15.4); determinism ×2 + coherent text + 8-concurrent bucket-1 green. **PASSED 2026-07-07 (04179f2, jz-38): solo 15.27 ms ×2 (p99 15.38) vs ep8 19.23 (−3.96 ms, 1.26×); 8-concurrent diverse step-bench 20.92 vs 24.56 ms p50 (1.17×, 382 vs 326 tok/s); determinism ×2 byte-identical; text coherent (first-token near-tie flip vs EP8 — expected, all 75 layers change partial-sum order); tp8 load 180 s vs ep8 75 s (8× slice read amplification absorbed by page cache — streaming loader not needed).** The rendezvous wall F is fully amortized at 75 layers, exactly as the sweep model predicted |
| **M3** | Bucket decision by measurement: masked grouped GEMM on TP slices at m=16/32/64 (per-rank N thins to 512 gate|up) vs staying EP8 for buckets 2/4/8. Within one process the layouts are exclusive (constraint 1), so this sets the *default* of the launch-time switch and decides whether EP8 ever retires | Decision record written from M2 data + the masked-GEMM-on-slice microbench |
| **M4** | Follow-on (next phase): attention projections TP + o_proj allreduce reusing the same LL machinery; MTP verify path | separate doc when reached |

Benefit claims are reported **per workload**: solo (pads share routing, E_active≈9,
maximum win) and diverse bucket-1 (E_active≈57, per-rank ~268 MB/layer expert reads —
same bytes as EP8, but the persistent kernel's tile loop lengthens and dilutes the win).

## Risks, ranked

1. **m=8 E_active dilution** (above) — report solo/diverse separately; no blended claim.
2. ~~**Peer access × NCCL window registration interference**~~ — cleared by P0 probe (a), 2026-07-07 (host-API scope; shim device-API coexistence lands with M1).
3. **Cooperative kernel occupancy vs existing aux-stream overlap** — the shared expert
   folds into the kernel (no loss by construction), and indexer∥MLA overlap lives
   outside the MoE segment; confirm with an nsys graph trace anyway.
4. ~~**Grid size**~~ — measured flat in-situ (132 vs 264 identical, M1).
5. **Near-tie routing flips under the tolerance gate** — bounded allowance, as EP8.

## Code map (verified against main @ 9c169f9)

| Topic | Location |
|---|---|
| EP8 MoE chain (replacement target) | `pegainfer-glm52/src/moe_ep8.rs:193` (`glm52_moe_ep8_routed_forward`); call site `model/mod.rs:972-1012` |
| Router semantics | `moe_decode.rs:396-419` (`run_router_into` → `glm52_router_noaux_tc_launch`) |
| Shared expert (fold target) | `moe_decode.rs:169` (`forward_into`); aux-stream fork/join `model/mod.rs:980-1004` |
| Weight loader (gains the TP-slice path) | `weights.rs:93` (`expert_placement`), `weights.rs:275-290` (rank slicing), `weights/load.rs` |
| Per-layer MLP enum (new arm) | `layer.rs:59` (`Glm52LayerMlp`) |
| Graph capture infra | `pegainfer-core/src/cuda_graph.rs`; per-bucket state `model/mod.rs:285-297` |
| New-kernel plumbing | drop `.cu` under `pegainfer-kernels/csrc/glm52/` (auto-collected, `build.rs` `is_glm52_source`); FFI `src/ffi/glm52.rs`; wrappers `src/ops/glm52/`; register in `KERNELS.md` |
| GEMV conventions to match (dot/dequant/scale) | `csrc/glm52/glm52_moe_gemv.cu:75-148` |
| Multi-GPU contexts / comm buffers | single process, 8 threads (`runner.rs`), per-rank primary context (`weights/context.rs`); LL buffers = per-device `cudaMalloc` + peer access + pointer table |
| A/B anchor | `pegainfer-server/src/bin/glm52_step_bench.rs` (bucket-1 solo = the 19.5 ms reference) |
| Numeric gates | `pegainfer-glm52/src/oracle/layer_ep8.rs` (EP8 layer gate) + `oracle/layer_tp8.rs` (TP8 twin, passing on jz-38); e2e gates per `serving-status.md` |

## TP8 + MTP: the span row mapping (single-user speculative decode)

At bs=1 the bucket-1 dp8 mapping computes m=8 with 1 real row and 7 pads. The **span
mapping** turns those pads into work: all 8 rows belong to one owner rank (1 committed
token + 7 DSpark drafts, or 8 prefill positions), gathered from the owner instead of
one-per-rank — the MoE cost is unchanged (the compute phases are mapping-agnostic;
only the AG source and RS destination patterns differ). The owner id is read from
device memory (staged per step, like the epoch), so one captured bucket-8 span graph
serves any owner. The tp8 scheduler serves exactly two shapes: bucket-1 dp8 (any
concurrency) and bucket-8 span, planned only when the fleet holds exactly one active
request wanting >1 row. EP8's span-4 verdict (bucket-4 round beats span-8's tail)
inverts here, so tp8 feeds the drafter's full 7-draft proposal.

**Measured (jz-38, 8×H200, solo, greedy, 500-token completions):**

| workload | tok/s | round p50/p99 | accept incl. bonus |
|---|---|---|---|
| code ×2 | **186.4 / 189.0** | 25.5 / 29.1 ms | 4.80 (19/104 rounds accept all 7) |
| prose | **106.5** | 24.8 / 27.4 ms | 2.64 |
| counting | **136.2** | 24.6 / 28.0 ms | 3.37 |

Baselines: plain tp8 solo 65.5 tok/s (15.27 ms/tok), EP8+DSpark span-4 code ≈ 75 —
**2.5–2.9× on code**. Bonus from the same shape: solo prefill rides span-8 (8 prompt
tokens/step), long-prompt cold TTFT ~2400 tokens 5.37 s vs ~36 s at 1 token/step.
Cost: the RS LL region grows a row dimension ([parity][row][src][hidden], one layout
for both mappings since any rank can be the device-staged owner) — ~900 MiB/rank for
75 layers vs ~112 before, which the post-build VRAM probe deducts from KV capacity
(≈5.5k tokens' worth).
Correctness: span-vs-dp8 bit-identity gate (`oracle/layer_tp8.rs::layer_moe_tp8_span_matches_dp8`
— same 8 rows through both mappings must match exactly, plus the pad zero-fill
contract), determinism ×2, 2-concurrent clamp-back, clean shutdown, EP8 span-4
regression. Measurement trap for the record: one SSE event carries one verify round's
committed text, so event counting understates tok/s by the accept factor — count
completion tokens against stream wall time.

## Round profile: where the 25 ms goes (nsys, jz-38, 2026-07-08)

Two traces of solo tp8+dspark decode on merged main (c57481a), same probe prompts as
the table above. Methodology: `--cuda-graph-trace=graph` for honest wall numbers
(near-zero overhead re-confirmed: 183 tok/s traced vs 186 untraced; whole-graph rows
land in the `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` sqlite table, not `..._KERNEL`), then a
separate `--cuda-graph-trace=node` run for composition **proportions only** (it
inflates the round ~73%). Scripts: jz-38 `~/develop/xingming/tp8_mtp_profile.sh` +
`tp8_round_pair.py` (round decomposition), `tp8_node_profile.sh` +
`tp8_node_attrib.py` (verify-graph composition; draft windows carved out by
dflash/markov anchor clustering).

**Round decomposition (graph-trace, honest):**

| component | code (p50/p99) | prose (p50/p99) |
|---|---|---|
| round wall | 26.2 / 29.1 ms | 25.5 / 30.5 ms |
| verify span-8 graph replay (device) | **23.7 / 25.0 ms (90%)** | 22.9 / 24.2 ms |
| draft window (replay end → next replay start) | 2.6 / 3.2 ms (10%) | 2.5 / 4.1 ms |

The draft window itself: the drafter's 7 markov steps are already graph-replayed
(7 × 205 µs = 1.4 ms device), plus ~60 µs eager dflash kernels (aux attention, rope,
copies), plus ~0.7 ms host gaps between the 7 step replays, plus 0.5 ms
last-draft-end → verify-start (the big graph's launch cost partially exposed).
**Draft is not the lever**: a zero-cost draft would buy only ~11% tok/s.

**Verify-graph composition (node-trace, read proportions only; per-step device-busy
sums to ~24.9 ms inflated vs 23.7 honest):**

| segment | share | note |
|---|---|---|
| MoE TP8 chain (`tp8_*` ×75 + router + quant) | **~45%** | includes cross-rank wait spun inside `rs_recv` (13%) / `ag_*` — the F-wall lives here, so this over-reads pure compute (M0 anchor: diverse m=8 ≈ 75 µs/layer ≈ 5.6 ms) |
| FP8 weight-only batched GEMV (`glm52_gemv_batched_mma`, **360/step**) | **~30%** | all dense projections: MLA q/kv/o ×78 layers + indexer proj ×21 + dense-MLP ×3 — the single biggest non-MoE block |
| MLA attention core (sparse flash + combine + pack/assemble) | ~14% | |
| cuBLAS nvjet GEMMs + splitK | ~7% | |
| norms / elementwise | ~3% | |

Lever ranking this data supports: **M4 attention-TP first** — the FP8 GEMV block
(~30%) plus MLA core (~14%) is the round's non-MoE majority and both split 8-way;
draft-side work (adaptive span, draft-step host gaps) is capped at ~10% of the round.

## Attention-TP: replicated activations over head-sharded weights (M4)

`feat/glm52-attn-tp` extends the TP topology from the MoE to the whole layer:
q_b/kv_b/o_proj/indexer-q are head-sharded ×8 (8 heads/rank; W_UK/UV batch 64→8,
o_proj input columns split), and instead of gathering activations, **every rank
redundantly computes ALL bucket rows** — bit-identical across ranks, so the only
wire traffic is two per-layer partial-sum collectives: the o_proj epilogue rides a
new two-shot LL allreduce (`glm52_tp8_ar.cu`, 7.7 µs/layer graph-replayed), and the
MoE RS becomes an all-reduce (push broadcasts every row to every rank). The AG phase
— and with it the span/owner concept — is deleted: the scheduler runs one logical
rank with eight mirrored executors (outputs asserted bit-identical at join), serving
a single bucket-8 shape.

FlashMLA stays untouched: `splitkv_mla.cuh` asserts `h_q % 64 == 0`, so query/latent
keep the 64-wide layout with the 8 real heads in slots 0..8 and zero-query pads
(zero query → uniform softmax → finite latent, discarded). The MLA kernel itself
does not get cheaper from head sharding — and each rank now reads every request's
KV (replicated latent), which is the structural cost at high concurrency (below).

**Pad rows ride a device want-mask** (`active_rows`, staged per step like the
epoch; actives are a bucket prefix by plan construction): pads are excluded from
the expert union (solo union 17→9 entries — the B/C GEMMs nearly halve), skip both
LL collectives entirely, and get zero-filled outputs. Zero is the pre-capture shape
(push nothing, wait on nothing — capture pairs trivially).

**Measured (jz-38 8×H200, same day, same client, branch f8677a8 vs main 2669bb6):**

| workload | main tp8 | attn-TP | Δ |
|---|---|---|---|
| solo plain ITL p50 (p99) | 15.27 ms (15.38) | **13.75 ms** (13.85) | **−10%** |
| solo MTP code tok/s ×2 | 186.1 / 188.6 | 197.3 / **220.9** | **+17%** |
| solo MTP prose tok/s ×2 | 108.5 / 108.6 | 118.8 / 119.5 | **+10%** |
| c8 diverse ITL p50 | 20.56 / 20.58 | 22.65 / 22.68 | **+10% (regression)** |
| EP8+MTP code (regression) | 131–134 (historical) | 132.4 / 134.4 | untouched |

The c8 regression is the structural trade: the old mapping attended 1 row × 64
heads per rank against its own request's KV; replicated activations attend 8 rows
× 8 heads (same head-row product) against **all eight requests' KV** (8× KV bytes
per rank), plus the AR wire. Attention-TP still beats EP8 at c8 (22.7 vs 24.6 ms)
and tp8 remains the ≤8-concurrency low-latency mode — but at c8 the pre-M4 tp8 was
faster. Correctness: plain DET ×2, MTP DET, **MTP text byte-identical to plain**
(the lossless contract end-to-end through 78 sharded layers + AR + replicated MoE),
cross-rank bitwise identity in the twin/layer/AR oracle gates.

Solo per-step attribution (node-traced, both sides equally inflated): GEMV family
5158→3342 µs (the shard saving, realized), dense nvjet −516, AG deleted −730,
versus flash MLA +1155 (the KV-replication tax), rs broadcast and AR wire mostly
reclaimed by the want-mask. Net −1.5 ms/step solo.

## Next action

Attention-TP measured and gated (table above); single PR for `feat/glm52-attn-tp`.
Remaining levers: the **flash MLA sharding tax** (+1.2 ms/step — each rank reads
the full KV for 1/8 of the heads; unexplored: KV segmenting or head-group
reordering), M3 bucket decision for multi-user (masked grouped GEMM on TP slices
at m=16/32/64), adaptive span, tree drafts.
