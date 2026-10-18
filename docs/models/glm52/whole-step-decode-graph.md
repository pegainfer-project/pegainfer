# GLM5.2 whole-step decode CUDA graph (PR5c)

> **TL;DR:** Execution record of PR5c: the whole per-rank decode step (embed → 78 layers → lm_head → device argmax) is captured into one CUDA graph and replayed every step. Measured on jz-38 8×H200, single request: **200 → 37.5 ms/step from the graph alone** (byte-identical to the PR5b record), **→ 31.3** after switching every m=1 projection to the weight-only fp8 GEMV (activation quant removed — re-gated via the #499 oracle in the new `--precision gemv` mode), **→ 25.3** after grid-striding the capacity-sized MoE quant/SiLU launches (block *scheduling*, not arithmetic, was their cost), **→ 23.4** after packing gate|up into one GEMV, overlapping the shared expert with the MoE collectives on a second stream (fork/join events inside the graph), and staging the relayout's expert ranges in shared memory, **→ 22.9 single** after fusing each layer's closing add with the next layer's input norm (ping-ponged attn buffers, bit-identical `_round` kernel) and running the DSA indexer concurrently with the MLA front's q_b/kv_a, **→ 22.6 single / 22.3 at 8-way (~346 tok/s aggregate)** after two-tier attention graphs (short-context FlashMLA topk 256 — while `seq_len <= 256` the DSA top-256 IS the full token set, so the short-tier graph attends the same tokens at 1/8 the padded index walk; tier-crossing and mixed-tier concurrency e2e-gated, short tier oracle-gated), **→ 22.5 single** after swapping the device argmax to the shared two-stage split kernel (bit-identical; the single-CTA scan was the last serial kernel in the step). A step-timing probe puts the inter-step host gap at ~0.05 ms — the whole step lives inside the graph; **vLLM GLM5.2 DP8/EP8 measured on the same node/workload: steady-state 20.0 ms/step (TPOT 19.8)** — the remaining 2.6 ms gap sits mostly in the expert GEMM (their ~10.8 µs/instance vs our 64-row-M-tile TRTLLM grouped at 22.7 µs) and the collectives' wait structure (#542; fp8 dispatch payload is measured perf-NEUTRAL — dispatch is rank-arrival-wait-bound, not byte-bound). **→ 19.6 single (2026-07-05, below the vLLM reference)** after replacing the TRTLLM grouped expert GEMM with the DeepGEMM masked grouped GEMM (the "swapAB" attribution was wrong — retraction + A/B record in the masked-GEMM section; c64 diverse 1113 → 1475 tok/s, solo span-8 32.2 → 28.2 ms). **Launch-ahead decode (D8, 2026-07-05): the ~0.7 ms host-side cuGraphLaunch + step-boundary idle move off the critical path (inter-step device idle 641–804 → 43 µs; solo 19.5 ms/step, zero variance) — feed kernel + pinned D2H + coordinator-global speculation lease, byte-parity gated.** Indexer oracle reference drift is #541 (pre-existing on main).
>
> **Last touched:** 2026-07

## Why a whole-step graph works here

- **Every step has the same shape by construction.** The DP8 lock-step protocol (#537) forwards exactly `GLM52_DECODE_GLOBAL_TOKENS = 8` tokens per global step — idle ranks send padding. Unlike kimi's full-bucket-only replay (partial occupancy falls back to eager), GLM5.2 never has a partial bucket: capture once per rank, replay every step, no eager fallback path.
- **No cross-rank barrier needed for capture.** Stream capture records without executing; in lock-step all 8 ranks hit their first step (and therefore capture) on the same global step, then each rank's first graph launch executes the collectives together. The safety ceiling is the DeepEP device timeout (~100 s) against a capture+instantiate window of tens of ms — same argument as kimi (`pegainfer-kimi-k2/src/runner/worker/state.rs:261-271`).
- **The collective path is already graph-clean.** `decode_dispatch`/`decode_combine` are single stream-ordered calls with compile-time worst-case shapes, no count readback (`kCpuSync=false` — the host busy-wait path is prefill-only). The kimi decode graph replays the identical shim launch machinery (cooperative + cluster-dim 2 via `cudaLaunchKernelExC`) on H200 in production benches.

## Capture-blocker inventory (main @ 45e72ab)

The captured region is `Glm52RankModel::decode_step` (`pegainfer-glm52/src/model.rs:327-415`) minus its prologue.

**1. Per-step host-varying inputs — stay OUTSIDE the graph (kimi pattern: update device buffers, then launch).**
`model.rs:338-347`: rope cos/sin row `memcpy_dtod` (host-computed slice offset), `slot_mapping`/`seq_lens`/`token_id` `memcpy_htod`. All downstream consumers already read these buffers on-device, so the prologue stays eager and the graph starts after it.

**2. `position` as a kernel scalar inside the region — must read a device pointer instead.**
`glm52_mla_cache_pack_launch(…, position)` (`mla_decode.rs:391`) bakes the cache write slot into the capture. Change the kernel to read the slot from `slot_mapping` (already device-resident, already holds `position` as i64). The host `ensure!` bounds checks inside the layer loop (`mla_decode.rs:340-344`, `model.rs:334-337`) only validate the capture step — hoist them to the prologue so they hold for every replay.

**3. Indexer mid-step D2H readbacks — the only true syncs between embed and logits.**
`indexer.rs:292` (`weights_proj` output → host) and `indexer.rs:320-327` (quant scales → host, host fold `weights · q_scale · sm_scale · h^-1/2`, htod back). 21 full-indexer layers × 2 syncs per step. Replace with one fused 32-element device kernel.

**4. Per-step allocation churn — hoist into a persistent arena.**
Remaining after #535's MoE workspace: MLA attend scratch (`ql_nope/query/ckv_fp8/ckv_scales/latent/lse/lse_accum/o_accum/v`, `mla_decode.rs:353-413`), every `fp8_linear` call (`a_fp8/a_scale_plain/a_scale/out`, `fp8.rs:131-168`), indexer (~15 buffers/full layer, `indexer.rs:273-419`), `fp8_mlp` (`fp8.rs:218-227`), layer norm/residual scratch (`layer.rs:104-224`), router outputs (`moe_decode.rs:329-331`), bookends (`bookend.rs:33/53/79`), and the last MoE alloc (`combined`, `moe_ep8.rs:292`). All shapes are static (bs=1/rank, topk 2048, `bound_rows` 2080), and identical across layers — one shared scratch set, not per-layer.

**5. Egress.** `clone_dtoh(logits)` + host `greedy_argmax` (`model.rs:413-436`). Move argmax on-device (the repo already wraps FlashInfer top-k selection K=1: `csrc/shared/flashinfer_top1.cu`); the graph ends at the sampled-token device scalar, and the per-step D2H shrinks from the full vocab row to 4 bytes, outside the graph.

**Already capture-clean (verified):** DeepGEMM MQA AOT (PDL-only `cudaLaunchKernelExC`), FlashMLA sparse (caller-preallocated scratch, no sync), FlashInfer topk (`dsa_graph_safe=true`), all decode cuBLAS through the workspace-free `g_cublas_handle` (`csrc/shared/linear.cu:323-348`). No host branch depends on runtime state inside the loop — dense-vs-MoE and full-vs-shared-indexer are static per layer index.

## Execution (all measured on jz-38 8×H200, 133-step greedy request)

| stage | ms/step | notes |
|---|---|---|
| PR5b baseline (#537) | ~200 | 8 rank threads × ~4155 launches/step through one driver |
| whole-step graph (arena + host-quiet + capture) | **37.5** | byte-identical to the PR5b/PR5a record; capture on each rank's first step, warm+capture 493 ms |
| + weight-only fp8 GEMV for every m=1 projection | **31.3** | numerics change (activation quant removed) — re-gated via #499 oracle in `--precision gemv` mode |
| + device-bounded MoE recv chain (`expert_offsets[n_local]`) | 30.1 | bit-preserving (pad rows never read); small win — block scheduling still dominated |
| + grid-strided quant/SiLU (row grid capped at 256) | 25.3 | ~100k tiny blocks/launch → ~12k; work AND scheduling now scale with real rows |
| + gate\|up packed GEMV + shared-expert ∥ collectives + smem relayout ranges | 23.4 | per-row math unchanged; overlap = fork/join events inside the capture (kimi pattern) |
| + closing-add fused with next input-norm + indexer ∥ MLA front | 22.9 (8-way 22.6, ~341 tok/s) | bit-identical `_round` fusion; indexer forks after q_resid, joins before attend |
| + two-tier attention graphs (short-context topk 256) | 22.6 (8-way 22.3, ~346 tok/s) | same attended tokens below seq 256, 1/8 the FlashMLA padded walk; per-tier graph picked by position; indexer top-k narrows with the attend plan; new `mla_oracle_gate_short_tier` + tier-crossing/mixed-tier e2e gates (v7) |
| + two-stage device argmax (qwen split kernel) | **22.5** | bit-identical (global-index total order); the 154k-vocab single-CTA scan (0.22 ms traced) was the last serial kernel — e2e buys 0.1 ms, the rest was hidden |

8-way concurrency stays free throughout (same per-step wall as bs=1; ~308 tok/s aggregate at 128-tok requests). All e2e gates green each stage: determinism ×2, 8-way identical, mixed concurrency, disconnect, slot reuse, SIGTERM teardown ≤4 s.

Three latent findings shaken out on the way:

- **DeepEP cooperative+cluster launches replay fine under stream capture** — the kimi precedent held for the GLM-baked shim; no fallback path needed.
- **Block scheduling, not arithmetic, dominated the capacity-sized quant/SiLU launches** (2080×48 ≈ 100k 128-thread blocks ≈ 60 µs/layer): a device-bound early-return recovers nothing because retired blocks still get scheduled — the kernels are now grid-strided (row grid capped at 256, loop bounded by `expert_offsets[n_local]` on device).
- **The HF `glm_moe_dsa` indexer reference is a moving target** (#541): the 5.13.0 release regressed the RoPE-interleave fix (the oracle script now refuses such builds), and `5.13.0.dev0` drifted between snapshots — the indexer oracle gate needs a pinned or vllm-derived reference before its overlap threshold means anything. Not introduced by this branch (main scores the same ~1585/2048 against any freshly generated oracle).

## Where the 22.6 ms sits (final-state profile, jz-38 nsys 2026-07-04)

Methodology (read this before trusting any number): node-granularity trace
(`--cuda-graph-trace=node`) of the shipped build, 133-step greedy request,
totals divided by 1064 rank-steps. Small kernels inflate 30-50% under node
tracing, and the collective kernels' `avg` absorbs boundary stalls — so the
table uses **median × instances**, which sums to ~21 ms against the 22.6 e2e
wall; the difference is rank-arrival wait pooled into the dispatch/combine
side. Use this for magnitudes and ordering, and size any expected win with an
e2e A/B, never with trace ratios (the tier stage read ~1 ms in trace and
bought 0.3 ms e2e).

| bucket | ~ms/step | share | assessment |
|---|---|---|---|
| MoE collective wait (dispatch rank-arrival + combine device-bound spin) | ~5.6 | 25% | dispatch kernel proper is only 12.4 µs × 75; combine 47 µs × 75 + reduce-epilogue 9.5 µs × 75 — the wait lives inside the kernels. Engineering headroom, not physics (vLLM pays 29+36 µs/layer medians on plain NCCL AG+RS) |
| non-expert projection GEMVs (q_a/q_b/kv_a/o_proj × 78 + dense) | ~3.6 | 16% | weight-BW floor at ~75-80%; little headroom |
| expert grouped GEMM (TRTLLM, 2 × 75) | ~3.2 | 14% | 21.4 µs med/instance; **the biggest single movable item** — vLLM's DeepGEMM does it in 10.8 µs. RESOLVED 2026-07-05: replaced by the DeepGEMM masked grouped GEMM (section below; the "swapAB" attribution here was wrong — their kernel is MGroupedMasked with the same 64-row BLOCK_M) |
| FlashMLA short tier (splitkv 16.4 µs + combine 4.9 µs × 78) | ~1.7 | 7% | splitkv is the 4-index-block serial floor; combine collapsed 12.6→4.9 µs with 4 SM parts |
| quant / SiLU / relayout / metadata glue | ~1.7 | 7% | post-grid-stride steady state |
| shared expert + indexer (aux stream) | ~1.1 | — | overlapped with the collectives; mostly off the wall clock |
| absorb GEMMs (nvjet W_UK/W_UV × 78) | ~1.0 | 4% | cuBLAS already optimal for this shape (ncu-proven in the PP era) |
| router (gate GEMM + top-8 + splitK reduce) | ~1.0 | 4% | |
| norms (fused + standalone) | ~0.8 | 4% | |
| bookends (lm_head 0.45 + argmax + embed) | ~0.7 | 3% | lm_head is the 1.9 GB weight-read floor; the argmax was a serial single-CTA 0.22 ms until the two-stage split swap (below) |
| small kernels (cache-pack, assemble, adds, sort) | ~0.4 | 2% | |

Rough split: ~6.7 ms is weight-read physics (GEMVs + expert-GEMM bytes +
absorb + lm_head), ~5.6 ms is collective wait (engineering), the rest is
long-tail kernel time. The path to vLLM's measured 20.0 = the expert GEMM
swap (~1.5 ms; landed 2026-07-05, masked-GEMM section below) + whatever
shortening each layer's critical path recovers from the arrival wait.

A late find from this profile: the device argmax was the last serial
single-CTA kernel in the step (one 256-thread block walking the 154k-vocab
row, 0.22 ms/step); it now uses the shared two-stage split kernel (per-4096
-tile partials + finalize, the qwen3 dspark path) — bit-identical by
construction, e2e-parity-gated, 22.6 → 22.5 (the calibration rule above in
action again: 0.22 ms of traced serial time bought 0.1 ms of wall).

Attempted, proven correct, and measured perf-NEUTRAL: **FP8 dispatch payload** (source-rank quant commutes with byte-preserving dispatch → bit-identical; vendored SF-pack support wired through config/shim/wrapper/moe_ep8, plus a plain-copy SF patch after the `cp.async.mbarrier.arrive` accounting hang). Clean probe on an idle node: 22.9-23.0 ms/step, parity PASS — identical to bf16 dispatch. nsys shows `dispatch_impl` kernel time really drops (~0.4-0.5 ms/step) but wall time doesn't move: **dispatch is rank-arrival-wait-bound, not byte-bound, at DP8/bs=1**. Not landed (complexity without measured win); full diff preserved on `wip/glm52-fp8-dispatch` @ c275be8 for when per-rank batches make dispatch byte-bound. An earlier "26.2 regression" reading was a contaminated measurement (another user's vLLM run on the same GPUs).

Measured dead ends (all recorded in #542): the inter-step host gap is ~0.05 ms (step-timing probe — device self-feeding/pipelining buys nothing); `kDecodeNumSms` 32→16 AND 32→64 both exactly flat (combine's 42 µs/layer is protocol/NVLink latency, insensitive to SM count in either direction); FlashMLA `num_sm_parts` 32→4 REGRESSES at topk 2048 (25.1 — fewer splits make splitkv the long pole; the short tier uses 4 parts because it only walks 4 index blocks); L2-prefetching the next layer's MLA weights on the aux stream REGRESSES (24.7 — contends with the grouped GEMM and thrashes L2); the quant-style capacity-loop lever does not exist for dispatch/combine (already device-bounded via the vendored capacity-sentinel).

The same-node vLLM nsys diff (their trace: NCCL AllGather+ReduceScatter naive EP — NOT DeepEP — at 29+36 µs/layer medians, comparable to our 12+47) relocates the remaining 2.6 ms: (1) **expert GEMM** — vLLM's DeepGEMM runs ~10.8 µs/instance where our TRTLLM grouped takes 22.7 µs (resolved 2026-07-05: their kernel turned out to be MGroupedMasked, not swapAB — see the masked-GEMM section). (2) **collective wait structure** (dispatch is rank-arrival-bound; fp8 payload measured neutral). A calibration lesson from the tier stage: node-granularity nsys inflates small kernels — the attention diff read as ~1 ms in trace proportions but bought 0.3 ms e2e; size expected wins from e2e A/B, not trace ratios.

## Bucket-8 step attribution (jz-38 nsys 2026-07-05)

Why does a bucket-8 step cost 45.6 ms when bucket-1 costs 22.5? Same methodology
as above (node-granularity trace), two windows on the same binary: bucket-8 via
long-prompt span ingestion (one slot, 8 rows/rank; 1800 rank-steps sampled) vs a
bucket-1 solo control (2520 rank-steps). Total-kernel-time accounting closes:
27.2 → 50.1 ms/rank-step (+22.9) against the bare-metal +23.1 wall delta.

| component (per rank-step, total incl tails) | b1 | b8 | Δ |
|---|---|---|---|
| non-expert weight-only GEMV (all projections) | 6.5 | 21.6 | **+15.1** |
| expert grouped GEMM (TRTLLM) | 3.4 | 7.2 | +3.8 |
| MoE combine kernel | 3.6 | 5.4 | +1.8 |
| MLA splitkv attention (8 queries vs 1) | 1.3 | 2.6 | +1.3 |
| SiLU/quant glue | 1.2 | 1.8 | +0.6 |
| dispatch (incl rank-arrival wait) | 4.4 | 4.6 | +0.2 |
| everything else | ~6.8 | ~6.9 | ~0 |

Two-thirds of the delta sits in the kernel designed to be batch-free: the
batched weight-only GEMV reads each weight packet once for all 8 rows, but the
1-row/warp layout issues 16 activation LDGs per weight LDG — ncu shows 93%
L1TEX / 12% DRAM, i.e. the L1TEX **port** is saturated, not bandwidth or FMA.
Shared-memory staging cannot help (LDS shares the L1TEX pipe on Hopper — a
microbenched smem variant lost everywhere); registers are the only storage off
that port. The collectives are nearly innocent: dispatch wait is flat and
combine grows only +1.8 ms for 8× payload.

Fix (`perf(glm52): ROWS=4 register tile for the batched weight-only GEMV`):
each warp owns 4 output rows and reuses the register-held activation chunk
across them, cutting per-row activation loads 4× while keeping per-row bit
parity with the m=1 kernel (memcmp-gated microbench, all 10 shapes × batches
{2,4,8}). Batch 2 keeps the 1-row/warp shape (act traffic negligible there).
H200 microbench: o_proj 132→66 µs, dense_dn 99→51, q_b 42→26 at batch 8;
weighted per-step: bucket-8 GEMV 23.0→13.9 ms, bucket-4 13.4→9.5 ms. ROWS=8
spills registers; WARPS=4 keeps ~3 blocks/SM. **e2e measured (jz-38, dspark
stack + cherry-pick): bucket-8 span step 45.6 → 37.0 ms (−19%), 1621-token
prompt TTFT 9.25 → 7.55 s; bucket-1 solo 22.63 ms/step — flat, b1 path
untouched.** Matches the microbench-weighted −9.1 ms prediction. Artifacts:
traces + kern-sum CSVs `~/develop/xingming/b8span* b1ctrl*`, microbench
`~/develop/xingming/gemv_bench` on jz-38.

## Tensor-core GEMV for batches 4/8 (jz-38, 2026-07-05, #559)

Post-#558 re-attribution (same two-window method, fixed binary): the b8−b1
delta fell +22.9 → +15.2 ms and the GEMV was still the largest term:

| component (per rank-step, totals) | b1 | b8 post-#558 | Δ |
|---|---|---|---|
| weight-only GEMV | 6.4 | 13.2 | **+6.8** |
| expert grouped GEMM | 3.4 | 7.2 | +3.8 |
| MoE combine | 4.3 | 6.2 | +1.9 |
| glue | 5.8 | 7.0 | +1.2 |
| MLA splitkv | 2.1 | 3.3 | +1.2 |
| dispatch (incl wait) | 5.3 | 5.7 | +0.4 |

ncu on the ROWS=4 tile showed the wall had MOVED: L1TEX 46% (fixed), DRAM 24%,
Compute 65% — the BATCH×16-term f32 FMA chain itself. Computed CUDA-core
floor for o_proj at batch 8 ≈ 32 µs vs the 28.5 µs weight-read floor;
occupancy and cache-policy variants (KernelWiki NVFP4-GEMV levers) measured
flat or worse. Conclusion: the CUDA-core design space was exhausted — the fix
is `mma.m16n8k16.bf16` (fp8 e4m3 decodes losslessly to bf16).

Key trick: no weight repack. The mma k-slot→column map is a free permutation
when A and B agree, so σ(step s, tid, d) = tid·16+4s+d over a k64 super-chunk
reads the *original* row-major layout with one 16-byte LDG per owned row —
no second weight copy, m=1/batch-2 paths byte-untouched. KSPLIT k-slices
write f32 partials to a 12.6 MB per-device scratch (allocated by the
pre-capture bucket warm-up) and a fixed-order epilogue reduces them:
deterministic and replay-stable per bucket, but **not** bit-identical to m=1
(validated by tolerance: max_rel < 2e-2, ≥90 % of bf16 outputs exact, plus
e2e greedy coherence and dspark accept-rate parity).

A packed fragment layout is ~30 % faster still on the big shapes (o_proj 32 vs
46 µs) but needs either a second weight copy (+15 GB/rank — doesn't fit) or an
m=1 kernel rewrite that changes bucket-1 numerics; recorded as a non-goal.
The vendored TRT-LLM dense fp8 `gemm()` entry (w8a8 library route) is a no-op
in our AOT setup — its SM90 dense path expects the DeepGEMM JIT runtime.

**e2e measured (jz-38): solo span-8 ingest step 36.9 → 32.2 ms (−12.6 %),
c64 diverse decode 1046 → 1121 tok/s (−8 % step); b1/b2 flat.** Iteration ran
on the in-process `glm52_step_bench` (one weight load, ~3 min/cycle).

## DeepGEMM masked grouped expert GEMM (jz-38, 2026-07-05)

**Correction first: the "swapAB" attribution above was wrong.** Reading vLLM's
DeepGEMM JIT cache on this node (`/data/cache/glm52_mtp_local/vllm/deep_gemm`)
shows the exact instantiations its 10.8 µs/instance came from:
`sm90_fp8_gemm_1d2d_impl<..., GemmType::MGroupedMasked>` with **BLOCK_M=64**
— the same 64-row M-tile as our TRTLLM kernel, no operand swap anywhere
(vLLM's `deep_gemm` copy is vendored at
`site-packages/vllm/third_party/deep_gemm`). The gap was kernel quality
(persistent 132-SM scheduler, TMA multicast on B, 8/6-deep pipelines, no
scale-relayout kernel) plus their lower hit-expert count at single-request
decode. W13 (4096×6144): BLOCK_N=128, 8 stages; W2 (6144×2048): BLOCK_N=192,
6 stages; both TMA multicast 2 on B, 132 SMs.

Our vendored DeepGEMM tree already supports `MGroupedMasked` in the SM90 1d2d
impl, so the port is an AOT instantiation (the `glm52_deepgemm_mqa.cu`
pattern — no JIT, no torch, no Python) plus a layout bridge:

- the metadata kernel emits `masked_m` (real rows/expert) and a `row_map`
  (aligned recv row → masked slot, -1 on alignment gaps) next to the segment
  offsets it already computed;
- the re-quant and SwiGLU-quant kernels keep their aligned-row loop space and
  device row bound, but write through `row_map` into fixed-stride
  `[32, 64, k]` slabs, with per-row scales going **straight into the
  mn-major TMA layout** the GEMM's SFA descriptor reads — the
  offset-scale-relayout kernel is deleted, not moved;
- one new remap kernel puts the W2 output back into the aligned slots
  `decode_combine` addresses. DeepEP dispatch/combine untouched.

Standalone A/B first (same data, same masked_m distribution, same node —
`~/develop/xingming/dgmask_bench/bench.cu`): masked wins 1.51-1.93× over
TRTLLM+relayout at 2/7/27 hit-expert scenarios, numerics match an fp32
reference at bf16-rounding level (4e-3 rel-to-row-RMS). Two bench traps worth
keeping: uniform-random fp8 bytes blow up the QGMMA reduced-precision
accumulator far beyond real data (use normal-quantized fills), and the
production TRTLLM launch is capacity-proportional (m_capacity = bound_rows ≈
2080 at 8 global tokens), so a bench that sizes it by real rows flatters it.

e2e A/B (same day, same node, `glm52_step_bench` + gate suite):

| workload | TRTLLM | masked | Δ |
|---|---|---|---|
| sweep b1 (8 diverse, conc 8) | 30.4 ms | 25.4 | −16 % |
| sweep b2 | 39.0 | 30.9 | −21 % |
| sweep b4 | 47.3 | 36.4 | −23 % |
| sweep b8 (c64 diverse) | 57.5 ms / 1113 tok/s | 43.5 / **1475 tok/s** | −24 % |
| solo span-8 ingest (spec-8 verify step) | 32.2 | 28.2 | −13 % |

EP8 layer-6 oracle green at every global-token bucket (g=64/32/16/8): 62/64
probes with the same two known router-near-tie outliers as the EP1/TRTLLM
record. The masked capacity invariant lives at the protocol level
(`global_tokens ≤ 64`; each token contributes ≤1 row per expert), NOT at the
shim's `decode_max_tokens_per_rank=128` buffer capacity — the metadata kernel
device-traps as the backstop.

Ops lesson that cost this session hours: `cargo build … | tail` swallows the
exit code — a failed jz-38 build "passed", and a whole gate suite ran on
stale binaries whose numbers happened to look plausible. The gate script now
refuses to run unless `nm` finds the new kernel symbols; keep that pattern.
(Silver lining: the stale run produced the TRUE same-day baseline sweep.)

## MegaMoE (DeepGEMM PR #323) evaluated — NOT usable at decode payloads (jz-38, 2026-07-05)

MegaMoE is DeepGEMM's mega-kernel fusing dispatch → L1 GEMM → SwiGLU → L2
GEMM → combine into one launch over symmetric NVLink memory (official release
is SM100 FP8×FP4 only; [PR #323](https://github.com/deepseek-ai/DeepGEMM/pull/323)
is the community SM90 FP8×FP8 adaptation, actively tuned — its tip has a
"Tune SM90 MegaMoE decode heuristics for GLM5.2" commit from ByteDance).
Evaluated as the candidate replacement for our per-layer MoE chain.

Post-#567 re-profile first (same node-granularity methodology as above, 19.6
ms/step untraced): dispatch+combine kernels are now **4.75 ms/step
median×instances, 7.73 ms/step wait-inclusive** — the whole MoE block
(collectives + quant/SiLU glue + masked GEMM) is ~8.3 ms med×inst ≈ **103
µs/layer kernel-resident, ~145 µs/layer wait-inclusive**, the largest bucket
in the step. That is the prize any fusion has to beat.

Measured (PR #323 tip a444600, sgl tvm-ffi wheel built with CUDA 12.8, kexi
venv torch 2.11 read-only, isolated `--target` pydeps; accuracy suite
**28/28 PASS** at GLM5.2 shapes, diff ~6e-4 vs torch reference — the kernel
is *correct*):

```
cd ~/develop/xingming/megamoe-deepgemm && PYTHONPATH=~/develop/xingming/pydeps \
python -u sgl_deep_gemm/tests/test_mega_moe_hopper.py --fused-only-sweep \
  --batches 1 2 4 8 16 32 64 --hidden 6144 --intermediate-hidden 2048 \
  --num-experts 256 --num-topk 8 --num-max-tokens-per-rank 128 \
  --activation-clamp inf --num-processes 8
```

| tokens/rank | fused µs/call | HBM GB/s | our chain per layer |
|---|---|---|---|
| 1 | 201.7 | 749 | ~103 µs kernel-resident / ~145 µs wait-inclusive |
| 8 | 345.9 | 2732 | |
| 64 | 399.6 | 3051 | |

**~200 µs structural floor at decode payloads — ~2× slower than our existing
unfused chain.** 75 layers × 200 µs = 15 ms/step of MoE alone. Ruled out as
causes: swapAB off (`DG_SM90_FP8_SWAP_AB=0` → 189 µs, same), small symm
buffer (`--num-max-tokens-per-rank 8` → 200 µs, same). The floor is the
in-kernel NVLink handshake + expert-wave scheduling: our whole comm set
(dispatch 12.4 + combine 30 + reduce 9.6 + prologue/copy 6.4) is ~58 µs
kernel-resident for the same protocol work.

Why the community's "2.7× at bs=1 on H20" doesn't transfer: their baseline is
DeepEP v2 + grouped GEMM orchestrated per-kernel from Python — MegaMoE's win
there is launch/orchestration overhead we already removed with the whole-step
CUDA graph. Against a graph-replayed chain, only MegaMoE's floor remains.
Revisit only if the SM90 kernel's small-payload floor drops ~4× upstream, or
for Blackwell (official SM100 path is FP8×FP4 — a weight-format change with
its own accuracy question). Checkout preserved at
`~/develop/xingming/megamoe-deepgemm` (branch pr323).

## Arrival-stagger anatomy + the cuGraphLaunch discovery (jz-38, 2026-07-05)

Methodology upgrade first: `--cuda-graph-trace=graph` records one range per
whole-step replay per device at **near-zero overhead** (traced 2715 ms vs
untraced 2708 ms for the same request — vs +100% under node tracing), giving
honest per-rank step timelines. Analyzer: `graph_stagger.py` /
`stagger_analyze.py` on jz-38.

Where the "wait-inclusive minus kernel-median" ~3 ms/step actually sits (solo,
19.6–20.4 ms/step):

| component | measured | mechanism |
|---|---|---|
| **`cuGraphLaunch` on the critical path** | **~700 µs/step/rank** (steady p50 700, p90 930; API trace avg 766) | host-side launch cost of the ~2000-node step graph; `exec_start − launch_end ≈ −4 µs` proves the GPU idles through it. Userspace libcuda work (strace: no ioctl, 99% sched_yield) |
| step-boundary device idle | 641–810 µs/step | the launch above + argmax D2H → host token processing → coordinator round |
| cross-rank step-start spread | 364 µs median / max 965 | ranks' launches land at different times; absorbed entirely by the layer-0 dispatch barrier (interior dispatch arrival spread is 4 µs — the protocol itself is tight) |
| per-layer combine straggler | 44 µs/layer spread (p90 67) × 75 | uneven per-rank expert-hit chains; in solo mode partly ARTIFICIAL — 7/8 rows are pads all routing like token 0, hammering the same 8 experts every layer |

Refuted along the way (each with a measurement): deep C-state exit latency
(`/dev/cpu_dma_latency`=0 → no change), and — via a standalone microbench
(`~/develop/xingming/coop_bench/`) — every structural suspect for the launch
cost: node count (2025-node graph → 4.1 µs), cooperative, cluster, PDL,
fork/join event nodes, 200 KB dynamic smem, 8-thread simultaneous launches
(6–9 µs each), pageable-D2H lock contention, and per-node parameter bytes
(512 B/node → 4.2 µs). The microbench cannot reproduce 700 µs; the cause
lives in the production context state (8 peer-mapped contexts, ~45 real
cubins, NVLink/host-mapped memory), not in any single graph feature. An
earlier "inter-step host gap ~0.05 ms" note came from an in-process probe
that did not see the e2e server's launch path — superseded by these numbers.

**Double-replay probe (decisive): launch-ahead works on the real graph.** A
6-line jz-38-only patch launched the same step graph twice back-to-back
(inputs unchanged → the second replay recomputes the first bit-identically;
generation text unchanged). Graph-trace gaps went bimodal exactly as
predicted: **intra-step replay gap 5.4 µs median (p90 5.7)** — the second
launch's ~700 µs host cost fully hides under the first replay's 20 ms
execution — vs 810 µs at the true step boundary. Wall 39.9 ms/step = 2 ×
20.2 − 0.5. Patch reverted, clean binary rebuilt.

Validated lever: **device-fed decode loop** — a feed kernel copies the
argmax result into the next step's input-token buffer on device, per-step
scalars advance on device, and the host enqueues replay i+1 while i executes
(EOS/streaming trail one step; a discarded speculative step is harmless).
Expected: the 641–810 µs boundary idle AND the 364 µs cross-rank spread off
the critical path, ~1 ms/step at solo (≈5%), same absolute saving per
spec-8 verify step. Second lever, independent: mask pad-row routing
(`topk_idx = -1`) in partial buckets to kill the artificial combine
straggler and 7/8 of solo collective payload.

## Launch-ahead decode (D8, jz-38, 2026-07-05)

The stagger investigation (docs PR #568) found `cuGraphLaunch` costing
~700 µs of host-side driver work per rank per step **with the GPU idle
through it**, plus a ~364 µs cross-rank launch spread — together the
641–810 µs step-boundary idle. The double-replay probe proved the launch
hides completely under the previous replay's execution (5.4 µs intra-step
gap). This branch turns that into the serving path:

- a **feed kernel** advances `token_ids`/`positions`/`slot_mapping`/
  `seq_lens` in place from the step's device argmax; rope rows re-gather on
  device; the argmax D2H lands in per-bucket **pinned** buffers so the copy
  is truly asynchronous — the next step's replay is enqueued before the
  host blocks on this step's result. Zero graph changes: everything is
  eager launches stream-ordered behind the executing replay, so the leased
  path is byte-identical by construction.
- **the speculation is all-ranks-or-none**: a speculative replay is a full
  set of collectives. The coordinator decides both flags on global data —
  `lease` (next step repeats this shape; includes model-length headroom,
  and pad rows can never outrun active rows) and `consume` (this step IS
  the speculation; needs unchanged shapes AND no slot handoff, because a
  finish+admission can reuse a slot id under an identical shape). Rank-side
  checks are hard `ensure!`s, not silent fallbacks — the first gate run
  desynced exactly there (the 1/8 near-tie variant hit EOS early, that rank
  fell back while others consumed, and the collective pairing slipped into
  the ~100 s DeepEP device-timeout trap).
- **every bucket × tier graph is pre-captured at startup** (8 warm pad
  steps while the ranks are trivially in lock-step): captured-ness must be
  uniform for the same reason, and the old mid-serving capture stall goes
  away with it.

Measured (jz-38, same-day baselines):

| metric | main | launch-ahead |
|---|---|---|
| inter-step device idle (own-gap, graph trace) | 641–804 µs | **43.0 µs** (p90 44.8) |
| cross-rank step-start spread | 350–364 µs (max 965–1097) | **32.6 µs** (max 42.5) |
| `cuGraphLaunch` p50 | 700 µs on the critical path | 660 µs, fully overlapped |
| solo ms/step (5 runs) | 19.6–20.4, run-to-run wander | **19.5, zero variance** |

Gates: determinism + byte-parity vs the main record PASS (short + 128-tok
head), 8-way identical (7/8 + the known deterministic near-tie variant),
8-way 128-tok concurrent 391 tok/s aggregate, staggered-admission churn +
post-churn determinism PASS, post-disconnect + teardown PASS.

Pitfalls recorded: cudarc's `memcpy_dtoh` copies the DESTINATION's byte
count (an oversized pinned buffer reads past the device allocation —
CUDA_ERROR_INVALID_VALUE); and a speculative-consume decision made from
per-rank data (inputs, positions, captured-ness) desyncs the collectives —
every such condition must come from the coordinator or be an invariant.

## Next action

Follow-ups: #542 wait-structure — launch-ahead lands the boundary half
(section above); the per-layer combine straggler under diverse load is
expert imbalance (EPLB, deferred). Pad-row masked routing (`topk_idx = -1`)
is the next small lever. #559 residual (combine +1.9, MLA +1.2 at b8),
#541 indexer reference re-pin.
