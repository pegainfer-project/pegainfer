# K3 decode kernel A/B: PegaInfer EP4 vs vLLM TP4+EP (pruned 224-expert, GB300)

> **TL;DR:** Mined both engines' decode kernel streams on the same tray/checkpoint
> (`/mnt/shared/weights/kimi-k3-pruned-75pct`, bs ∈ {1,8,32}, CUPTI capture, eager
> both sides), then **ported the two biggest levers as capsule cubins**
> (`PEGAINFER_K3_CAPSULE=all`): vLLM v0.28.0's fused KDA decode kernel and its
> `single_group_topk` router selection, loaded from vendored cubins with a
> fail-closed ABI check — zero vLLM source in our build. E2E on the same serve:
> **+7% decode throughput at 4-concurrent (96.3 → 103.0 tok/s), +20% at
> 32-concurrent (495 → 594 tok/s)**, greedy text 4/8 byte-identical and 4/8
> diverging at a near-tie token with comparable quality (rounding-chain
> difference, per-kernel gates bound it). Remaining mined-but-not-ported: CuTe
> skinny GEMM (only covers M≤2; CuTe param-staging complexity), attn-res
> (structure mismatch — their NB=3 online kernel vs our 8-block walk — for ≤2%
> of step). MoE stays ours (MegaMoE more fused, zero collectives).
>
> **Last touched:** 2026-08

## Setup

- Same tray (tray03, 4xGB300 sm_103), same checkpoint (224-expert MXFP4 pruned dev,
  isomorphic per-rank to full 896-expert @EP16).
- vLLM: `vllm/vllm-openai:kimi-k3` image (0.1.dev19262, native kimi_k3 support),
  `--tensor-parallel-size 4 --enable-expert-parallel --enforce-eager`,
  FLASHINFER_MLA backend, `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1`. Note the
  parallel shape differs by design: vLLM shards attention TP4 (every rank sees all
  requests, M = full batch), we run attn-DP (requests round-robined, per-rank M =
  bs/4).
- PegaInfer: `--features k3` release, `--k3-ep-size 4`, single process hosting all
  4 ranks; `cuda_graph=false` in this EP configuration, so every launch is eager
  and CUPTI-visible, symmetric with the vLLM side.
- Capture: `tools/kernel-capture/` injection lib on both engines (same JSONL
  schema). Drive: 1/8/32 concurrent completions x 128 tokens, diverse prose
  prompts; phase boundaries recorded as `launches.jsonl` line offsets, census
  taken over the 60–95% tail of each phase (steady decode, ~45 steps).
- Artifacts (local disk, not committed): vLLM
  `/data/susun/kernel-capture/k3-pruned-ep4/pid*/` (per-rank, 4 processes),
  PegaInfer `/data/susun/pegainfer-dev-cache/target/nvidia_cuda_13.2.0-devel-ubuntu24.04/kernel-capture-selfrun/pid1/`
  (one process = all ranks), plus `phase_census.json` / `pega_phase_census.json`
  under `/data/susun/kernel-capture/`.

## Headline: launches per rank-step (steady decode)

| | PegaInfer EP4 | vLLM TP4+EP |
|---|---|---|
| launches per rank-step | ~4.3k | ~2.2k |
| top kernel | `cublasLt::splitKreduce` (17%) | rank-local allreduce (12%) |
| distinct symbols in window | 57–59 | 50–53 |

## Per-op dispatch table

| op | PegaInfer | vLLM | verdict |
|---|---|---|---|
| KDA decode (69 layers) | `k3_kda_core` + 3x `k3_conv_silu` + `k3_land_nt256` x2 + rms + nvjet projections per layer | **one** `kda_decode_fusion_many_heads_kernel` per layer (exactly 69/step) | port lever (rank 3) |
| attn-res (24 MLA layers x 8 nb-blocks) | `k3_attnres_scores` + `k3_attnres_mix` per nb-block = **16 launches/layer** | `sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel` (~7/layer incl. aux) | port lever (rank 4) |
| MLA decode | one `mla_paged_absorbed_attn_kernel`/layer (absorbed, single-pass) | `fusedKimiK3MLADecodeQConcatKVCacheKernel` + CuTe-DSL Blackwell MLA split-kv + occasional reduction | comparable; needs timing (and long-ctx split-kv check) |
| MoE (92 layers) | `mega_quant_x` + `mega_write_routing` + **one fused** `deep_gemm::sm100_fp8_fp4_mega_moe` = 3/layer | MXFP8 quantize + routing (2–3) + 2x MXFP4 `bmm_t128x8x512` + finalize ≈ 7/layer | **we are more fused; keep** |
| collectives | none (attn-DP, MegaMoE pairs ranks over NVLink) | `vllm::cross_device_reduce_1stage` ~278/step (TP4 attention) | structural win, keep |
| dense GEMM | same 5 narrow nvjet tiles (`tss_64x8`, `tss_32x64`, `tss_128x8`) with **bit-identical grid/smem at bs1/8/32**; `splitKreduce` = #1 launched kernel every phase | bs1: own CuTe-DSL skinny GEMM/dotprod family; bs8: splitK + `tst_64x8`; bs32: `tst_64x32`/`tst_128x16` 2-CTA fat tiles, splitK mostly gone | **port lever #1** (skinny GEMM at M=1; M-aware re-pick later) |
| MoE routing top-k | mis-tuned `[1,224]` top-16 (5% of step per profile) | `moe::dev::routing::routingIndices{Block,DynBlock,Cluster}Kernel`, picked by bs, same E=224 shape | **port lever #2** |

## vLLM's bs-dependent dispatch (the "if bs > N" question, answered)

Batch-invariant on their side: KDA fused decode, MLA kernels, MXFP4 MoE bmm tile
(`t128x8x512` at every bs — per-expert M stays tiny with 224 experts). What
switches with bs:

- **Linear-layer GEMM provider**: bs1 = vLLM's own CuTe-DSL skinny/dotprod kernels
  (8 variants, latency-tuned for M=1 — the same "vLLM wins bs=1" suspect class as
  the Qwen study); bs8 = CuTe splitK + narrow nvjet; bs32 = pure nvjet fat tiles.
- **MoE routing kernel**: `routingIndicesBlockKernel` (bs1) →
  `DynBlockKernel` (bs8) → `ClusterKernel` (bs32), `BlockScoresKernel` joins at 8+.
- Triton fused-MoE configs (not active for K3 but shipped) are keyed per
  M ∈ {1,2,4,...,4096}; cudagraph capture ladder is `[1,2,4]+range(8,256,8)+...`
  — so under graphs a single injected startup enumerates every bs bucket.

Mining discipline that follows: capture per bs bucket; manifest rows bind per
`(op, bucket)` — the axis the capsule catalog already planned.

## Port levers, ranked

Ranking is **timing-informed**: `benchmarks/k3-ep4-decode-profile.md` already
measured the same EP4 decode step (~50 ms) as 52% backbone B=1 dense GEMM at
~50% of the bandwidth floor, 20% MegaMoE, 12% TileLang glue, 7% KDA core, 5%
router top-k — and explicitly *not* launch-bound. The census above tells us what
vLLM does differently at each of those slots; the profile tells us which slots
pay. (License classes per kernel-mining.md: vLLM-tree CUDA and CuTe-DSL /
`moe::dev` TRT-LLM-family kernels are Apache-2.0 and committable; `nvjet_*` is
proprietary — steal the decision, never the cubin.)

1. **B=1 dense GEMM (52% of step, ~10 ms recoverable).** vLLM solved exactly
   this cliff by *leaving cuBLASLt* at M=1: their CuTe-DSL skinny-GEMM/dotprod
   family is the mined counterpart of the near-SOL B=1 GEMV the profile calls
   for. Cubins + full ABI are in the capture; shapes match our checkpoint.
   First candidate for the capsule loader path.
2. **Router top-k (5%, ~2.6 ms).** Our `[1,224]` top-16 call is mis-tuned;
   vLLM's `moe::dev::routing::routingIndicesBlockKernel` for the *same E=224
   shape* is captured, per-bs variants included (Block → DynBlock → Cluster).
3. **KDA-layer fusion (attacks the 7% core + a slice of the 12% glue).**
   `kda_decode_fusion_many_heads_kernel` collapses our core + 3 convs +
   land/rms chain into one launch per layer, 69 of 93 layers.
4. **attn-res fusion (rest of the glue).** `attn_res_fwd_online_v2_kernel`
   replaces our 16-launch scores/mix nb-walk. Worth a few ms at most; take it
   only if (3) is already being ported from the same source area.
5. Not worth porting now: MLA (0.7% at short ctx — revisit at long context with
   split-kv), MoE (we are already more fused than vLLM), M-aware nvjet re-pick
   for larger buckets (real, but attn-DP keeps per-rank M small; matters only
   when per-rank batch grows).

## Port log (2026-08-28): capsule cubins for KDA + router top-k

Shipped as the **capsule substrate**: the serving kernel is an external cubin
artifact (`pegainfer-kernels/cubin/k3/`, provenance + sha256 in its README),
embedded at build time and bound in `csrc/k3/k3_capsule.cu` via
`cuModuleLoadData` + a fail-closed `cuFuncGetParamInfo` walk against the ABI
recorded at capture. No vLLM source enters the build; the native kernels stay
as the reference twin and `PEGAINFER_K3_CAPSULE` (unset ⇒ byte-identical
serving; `all` or csv of `topk,kda`) flips ops per launch site.

- **Router top-k** (`single_group_topk_warp_kernel<f32,f32,i32,SIGMOID,512,22>`,
  offline single-instantiation build, 54 KB): drop-in at the `step.rs` router
  call; weights come back in descending-score order vs our selection order —
  consumers treat the pairs as unordered. Needs the host-scalar routed scale
  (`rs_host` beside the device `rs`). Gate `k3_capsule_topk_gate`: expert sets
  equal, weights ≤1e-5, b ∈ {1,3,8,32}.
- **KDA decode** (`kda_decode_fusion_many_heads_kernel`, h96 static-layout
  head-grid variant, 42 KB): one launch replaces conv_silu×3 + kda_core, and
  the four projection GEMMs collapse to one full `wbig` GEMM + two landings
  (`out_gate`, new packed `q|k|v` land config). Formulas are our exact
  spellings (`GATE_LOWER_BOUND=-5`, `scale=128^-0.5`, `RMS_EPS=1e-5`, same tap
  order, same `[head, v, k]` state layout) — the captured scalars decode to
  precisely our generator constants, and **no new weights are needed**
  (`cw_*`, `dt_bias`, `a_log`, `gamma_o` are layout-identical). Two structural
  deltas: (1) the kernel's conv-tap stride is compiled as `3*12288`, so
  capsule mode allocates a packed `[rows, 3 taps, q|k|v, 12288]` slab filled
  by `adopt_row` at the prefill→decode handover (the only state-flow boundary;
  continuations re-prefill from scratch); (2) conv + recurrent state update
  **in place**, so capsule decode pins recurrent parity slab 0 and skips the
  ping-pong — dspark and CP refuse to arm with the flag set. Gate
  `k3_capsule_kda_gate`: conv windows bitwise-equal, state ≤1.6% / out ≤6.3%
  rel err (bf16-rounding chains; native lands intermediates in bf16, vLLM
  keeps f32).

**E2E A/B** (same tray, same serve config, 128-token completions, diverse
prose): 4-concurrent 96.3 → 103.0 tok/s (+7%), 32-concurrent 495 → 594 tok/s
(+20%). Greedy 100-token texts: 4/8 byte-identical, 4/8 diverge at a
near-tie token and continue at comparable quality. Native path with the flag
unset is untouched (same launch sequence, byte-identical).

Offline-build recipe: standalone TUs (upstream `.cu` cut above the torch
host-launcher block, `namespace { ... }` reopened around an explicit template
instantiation of the captured flag set) live at
`/data/susun/kernel-capture/offline-build/{kda_tu.cu,topk_tu.cu}` (tray03)
next to the captured ABI manifest `port_abi.json`.

**Not ported, and why:** skinny GEMM/dotprod — captured cubins only
instantiate M ∈ {1,2} (M is a CuTe compile-time), the A/B buckets (4/32) never
hit them, and staging CuTe's 24-byte tensor-view params is the highest-effort
ABI in the set; attn-res — vLLM's online kernel is templated NB=3 vs our
8-block snapshot walk (structural mismatch, not a slot-for-slot swap) for ≤2%
of step. Both stay mined in the capture with full ABI if the calculus changes.

## Caveats

- The census is launch identity/ABI, not time; the profile above supplies time
  for our side. A per-op nsys A/B against vLLM's kernels (especially skinny
  GEMM vs our nvjet-splitK at M=1) is still the gate before any port lands.
- Contexts here are short (~11+128 tokens); split-kv and long-ctx MLA behavior
  unmeasured.
- Side observation, not chased: pruned-checkpoint greedy text from our EP4 serve
  degenerated quickly ("the gryl of the gryl of...") on a prose prompt; worth a
  spec_verify/accuracy pass someday, unrelated to kernel structure.
