# K3 EP4 decode step profile (MegaMoE, 2026-08)

**TL;DR**: nsys kernel profile of the full 93-layer K3 decode step at
`--k3-ep-size 4` on 4×GB300 (MegaMoE transport, eager, B=1 per rank, 4
concurrent streams). The step is **not** MoE-bound and **not** launch-bound:
52% of the step is the backbone's B=1 dense GEMMs on cuBLASLt (running at
~50% of the bandwidth floor via a splitK strategy), MegaMoE is 20%, and a
mis-tuned router top-k kernel wastes 5%. The two cheap levers are a
near-SOL B=1 GEMV for the dense projections (~10 ms) and a router kernel fix
(~2 ms); together they project ~50 → ~30 ms/step. Graph capture for the EP
path is a small lever (GPU is already ~fully occupied).

## Setup

- 224-expert checkpoint, 93 layers, EP4 (56 experts/rank, 189 GiB/rank),
  MegaMoE transport (zero host collectives), eager, `max_batch=16`,
  greedy decode.
- 4 concurrent single-stream requests (one per rank partition), so every
  rank runs live B=1 steps; prompts ~12 tokens, `max_tokens=110`
  (`max_ctx=128` cap), requests looped back-to-back.
- nsys 2025.5.2, CUDA trace only, 15 s capture in steady state,
  device 0 kernel sums. Steps captured are counted exactly from fused-MoE
  kernel instances (92 launches/step): 307.6 steps, ≈50 ms/step of kernel
  time, which matches the wall-clock ITL — the device is essentially fully
  busy (the aux stream overlaps slightly, so busy-sum can exceed wall).
- Includes the requests' prefill steps (~10% of steps); K3 prefill is
  decode-shaped, so the mix is representative.

## Per-step breakdown (device 0)

| Phase | ms/step | Share | Notes |
|---|---:|---:|---|
| Backbone dense GEMMs (cuBLASLt nvjet + splitKreduce) | 26.0 | 52% | 485 GEMMs + 741 splitK-reduce kernels per step |
| MegaMoE fused kernel (92 layers) | 10.2 | 20% | 110 µs/layer incl. dispatch, experts, combine, cross-rank barriers |
| TileLang glue (norms, landings, add2, conv, attn-res, situ) | 6.2 | 12% | ~67 µs/layer aggregate |
| KDA core (69 layers) | 3.5 | 7% | 51 µs/layer |
| Router top-k (92×) | 2.6 | 5% | 28.6 µs per `[1,224]` top-16 call — mis-tuned |
| Mega activation quant + routing write | 0.5 | 1% | |
| Row plumbing (extract/gather/copy/scaled-add) | 0.6 | 1% | |
| MLA attention (24 layers, ctx ≤ 128) | 0.35 | 0.7% | 15 µs/layer at bring-up context |
| memcpy/memset | 0.3 | 0.6% | |
| argmax | 0.01 | — | |

## Findings

1. **The bottleneck moved to the backbone, not MoE.** The dense projections
   read ~100.5 GiB of bf16 per rank per step; the bandwidth floor is
   ~12.6 ms, and cuBLASLt's B=1 splitK strategy delivers ~26 ms (~50% of
   SOL), paying an extra 3.8 ms in 741 splitK-reduce kernels. A dedicated
   near-SOL B=1 GEMV for the dense projections is the single largest lever
   (~10 ms). Note: the earlier "GEMV retired" decision was about the
   *expert* path (masked grouped GEMM wins there); the dense backbone is a
   different trade.
2. **Graphs are a small lever on this path.** The device is ~100% busy in
   eager mode, consistent with the single-rank measurement that graphs buy
   only ~4.5%. EP-path graph capture (with its captured-barrier discipline)
   can stay deferred.
3. **Router top-k is an easy 5%.** 28.6 µs for a 224-wide top-16 at B=1 is
   kernel inefficiency, not work.
4. MegaMoE at 110 µs/layer already beats the retired stepwise chain's
   expert path by a wide margin at B=1, and this is with the block config
   pinned to the protocol max (the traffic-invariance choice).
5. Beyond these: the next structural step for latency is an FP8 backbone,
   which halves the bandwidth floor (and is the prerequisite for EP8 full
   K3).
