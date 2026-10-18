# GLM5.2 EP1 Decode Forward (PR3)

> **TL;DR:** **BUILT + ALL GATES GREEN on jz-38 H200 (2026-07-03), branch `feat/glm52-ep1-forward`.** PR3 composes the merged MLA brick (#477) and DSA indexer forward (#521) with MoE / dense / bookend bricks — mostly cherry-picked from the abandoned `feat/glm52-pp8-decode` branch, re-gated through the self-contained #499 harness — into a full oracle-gated decoder layer, EP1 (all 256 experts local, single H200, layer-brick scale). Gate results: bookend embed/argmax exact + 64/64 logits; layer-0 dense 64/64; layer-6 MoE 62/64 on BOTH expert paths (identical outliers = measured router near-ties, bounded allowance); MLA regression 64/64 after the front/attend split. See Execution log below. Two hard design constraints from the start: (1) **every op is host-quiet with fixed-capacity buffers** — the decode step must be CUDA-graph-capturable as-is (进图 is the acceptance bar, not a PR5 retrofit); (2) **the MoE kernel chain follows the in-tree DeepEP v2 (elastic) shim contract** (expert-major aligned segments + device psum → grouped-GEMM metadata → in-place grouped GEMM), so PR4 swaps a local scatter/combine pair for `decode_dispatch`/`decode_combine` and changes nothing else. Full-model e2e generation moves to PR4 — the model does not fit on one GPU.
>
> **Last touched:** 2026-07

## Corrections carried into the current layer implementation

The original bring-up plan described this layer only as a milestone. The corrections below are the retained implementation contract.

1. **"Full EP1 decode forward with e2e bs=1 generation gate" is impossible.** Routed-expert weights total ~700 GiB fp8 (measured rank slabs: rank0 105.1 GiB + 7×85.5 GiB). One GPU cannot hold the model. PR3 is brick-level: single-decoder-layer oracle gates (a MoE layer's 256 experts ≈ 9.4 GiB — fits fine). The first e2e generation gate lands in PR4, the first point where the full model is resident across 8 ranks and all-to-all exists.
2. **Cross-layer top-k sharing was missing from the plan.** GLM5.2 config: `index_topk_freq=4`, `index_skip_topk_offset=3` (`config.rs:36-39` validates both). Per transformers' derivation, layer *i* is `full` iff `max(i-2, 0) % 4 == 0` → **21 of 78 layers run the indexer** ({0,1,2} ∪ {6,10,…,74}); the other 57 are `shared` and reuse the previous full layer's `topk_indices` verbatim (they have **no indexer weights** — the manifest's conditional indexer push in `weights.rs:218-227` matches). Reuse across layers is sound because `topk_indices` are global KV slots and all layers share one block table / slot mapping. The layer composition must thread `prev_topk` through the loop; this is also a large decode win (indexer runs 21×, not 78×).
3. **MoE data layout targets the in-tree DeepEP v2 shim, not vllm's `BatchedExperts`.** `csrc/deepep/deepep_shim.cu` is already an AOT instantiation of **DeepEP elastic (v2)** kernels (`using namespace deep_ep::elastic`) with a torch-free host layer — the exact "deepepv2" decode path, kimi-proven graph-quiet (`do_cpu_sync=false`, worst-case fixed buffers, `moe_deepep.rs:10-17`). Its `decode_dispatch` emits an expert-major recv buffer with per-expert aligned segments plus a device-side `psum_expert`; the already-merged `glm52_deepgemm_grouped` **metadata kernel** consumes exactly that psum to produce `expert_offsets`. PR3's EP1 path emulates dispatch/combine with trivial local kernels against the *same* contract.

## The graph bar (进图标准)

Every PR3 op must satisfy, from day one:

- **No D2H, no host branching on data.** Router top-k, expert counts, slot conversion all stay on device. (Kimi lesson: `cpu_sync=false` means auditing every adapter kernel for capacity-proportional cost — bound work by device-side counters, not host loops.)
- **Fixed shapes at capture time.** MoE token capacity = `topk(8) × EXPERT_ALIGNMENT(64)` slots, allocated once in a decode-state arena (mirror `KimiMoeDeepEpState`, `moe_deepep.rs:66-142`). Empty tail rows are skipped via device bounds, never via smaller launches.
- **Pool allocation only, no arena.** The forwards allocate scratch per call through cudarc's stream-ordered `cuMemAllocAsync` pool — the PP8 branch's `graph_alloc_probe` proved pool allocation inside capture replays correctly as graph memory nodes, so per-call allocs are NOT a capture blocker and the arena refactor the old plan called for is unnecessary complexity. What stays banned is host-side allocation *decisions* (sizes must not depend on data).
- **Stream-ordered only.** Use the `_graphsafe` gemm variants from `typed_ops` for bookends (the variants exist precisely for capture; `pegainfer-kimi-k2/src/runner/worker/forward.rs:109-114`).

PR5 then *captures* this shape; it must not have to *change* it.

## Model math (vllm-verified; vllm is the production reference)

GLM5.2 in vllm is `DeepseekV2ForCausalLM` verbatim (`deepseek_v2.py:1899` — `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM): pass`). Zero GLM-specific model code; only config values differ. Facts a correct port must respect:

- **Decoder layer = plain two-norm fused-add residual** (`deepseek_v2.py:1247-1325`): `input_layernorm(hidden, residual)` → attn → `post_attention_layernorm(hidden, residual)` → mlp/moe → residual carried to the next layer's fused add. **No sandwich norms** (verified — no post-mlp/post-attn extra norms). RMSNorm eps = 1e-5. The two-arg norm returns `(rmsnorm(hidden+residual), hidden+residual)`.
- **Router** (`grouped_topk_router.py:80-161`): fp32 logits (gate GEMM bf16×bf16→f32; **not** `CUBLAS_COMPUTE_32F_PEDANTIC` — the kimi router's 60×-off-roofline mistake) → `sigmoid` → `+e_score_correction_bias` **for selection only** → top-8 (GLM has `n_group=1, topk_group=1`, so the group-limited stage degenerates away — simpler than kimi) → **weights gathered from the unbiased scores** → renormalize (`norm_topk_prob=true`; transformers adds `1e-20` to the denominator) → routed scaling 2.5.
- **Scaling order** (`moe_runner.py:699-719`): `y = shared_mlp(x) + 2.5 * Σ w_e · expert_e(x)`. The shared expert is **not** scaled. We fold `w_e × 2.5` into the per-slot weight at the SwiGLU→fp8 step (`glm52_silu_and_mul_weighted_per_token_group_quant`, already merged) — valid by linearity of down_proj, same trick kimi uses (`moe_deepep.rs:20-23`).
- **Dense layers 0–2**: SwiGLU, intermediate 12288, fp8 block-scale — `fp8.rs::fp8_mlp` (already merged) covers this as-is. The MoE shared expert is the same shape at intermediate 2048.
- **Bookends**: embedding bf16 `[154880, 6144]` plain lookup (no scaling); final norm = fused-add RMSNorm consuming the trailing residual; lm_head bf16 untied, no logit cap. Contracts already validated in `weights.rs:291-296`.
- **FP8 contract**: weights 128×128 block-scale, activations per-token-group-128 dynamic — matches what `fp8.rs` and the vendored TRTLLM CUTLASS runner already implement.

## Decode MoE kernel chain (EP1 now, EP8 by substitution)

```
                         PR3 (EP1, 256 groups local)          PR4 (EP8, 32 groups/rank)
hidden[6144] bf16
  ├─ router: cuBLAS gate GEMM → f32[256]                      unchanged
  │          epilogue kernel: sigmoid+bias → top8,
  │          unbiased-gather, renorm, ×2.5
  │          → expert_ids[8] i32, slot_weights[8] f32          unchanged
  ├─ act quant: fp8 per-token-group-128 (once)                 unchanged
  ├─ scatter: token → expert-major aligned slots,      ⇄       deepep decode_dispatch
  │           build psum_expert on device                      (shim, GLM-baked config)
  ├─ metadata: psum_expert → expert_offsets  (✅ merged)       unchanged
  ├─ grouped GEMM w13 (gate|up): TRTLLM moeGemm                unchanged
  ├─ weighted SiLU·mul + fp8 requant (✅ merged)               unchanged
  ├─ grouped GEMM w2 (down): TRTLLM moeGemm                    unchanged
  ├─ combine: sum owned slots into token           ⇄           deepep decode_combine
  └─ + shared_expert(x) (fp8_mlp, overlappable) + residual     unchanged
```

**Almost none of this is greenfield.** The abandoned `feat/glm52-pp8-decode` branch built and H200-oracle-gated the entire MoE/dense/bookend surface during its Slice 5/6 work (its fixtures were the prototype-era npz dumps — irreproducible, which is why the bricks must be **re-gated** through the #499 self-contained harness, but the code is written and was proven against real layer-3 weights):

- `csrc/glm52/glm52_router.cu` + `ops/glm52/router.rs` — `glm52_router_noaux_tc_launch`, HF-exact (selection on sigmoid+bias, weights gathered from unbiased scores, renorm, `route_scale` param — **pass 2.5**, the in-file 1.0 default is a placeholder). Gate was green: expert sets exact, worst weight rel 4.2e-4.
- `csrc/glm52/glm52_moe_route.cu` — `route_offsets` (builds `expert_offsets[E+1]` directly on device, aligned running total), `scatter` (replicate quantized token into its 8 expert-major slots), `combine` (plain slot sum; weight folded by the weighted-SiLU). `m_capacity = topk×64 = 512` proven tight.
- `glm52_trtllm_grouped_fp8` safe wrapper (`ops/glm52/trtllm_grouped.rs`) + the runtime-groups refactor — **`moeGemm` has executed and passed a layer-3 oracle on H200** on that branch; "compiled but never run" is only true of main.
- `csrc/glm52/glm52_moe_gemv.cu` + the branch's final `moe_decode.rs` — a later evolution: at bs=1 the block-scale GEMM pads M 1→64 and runs compute-bound, so the branch switched to a **weight-only GEMV** (bf16 activation × on-the-fly fp8 dequant, f32 accum) that drops activation quant, route_offsets, scatter, and both scale relayouts entirely.
- `dense.rs`, `bookend.rs`, `fp8_mlp` extraction, layer/residual composition (`model.rs`/`decode.rs`) — all exist; bookends oracle-gated (embed gather exact, lm_head argmax exact ×8).

**Expert-GEMM decision: land both cherry-picks behind one forward signature; the winner is a PR5 measurement, but the branch's data already leans GEMV for decode.** The PP8 branch *measured* this fork at bs=1: the TRTLLM/CUTLASS grouped path pads every active expert's M to the 64-row tile (1 real row → 64× compute waste, ~29 ms of MoE GEMM), while the weight-only GEMV rewrite was the dominant lever of its 67.8→22.7 ms TPOT campaign (weight-memory-bound, 73–83% DRAM after de-staging). Note the M≪tile regime persists at EP8 serving batch: bs=64 global × top-8 / 256 experts ≈ 2 rows/expert. Grouped stays in PR3 because (a) it is the layout DeepEP dispatch delivers and the shape that scales if batches grow, and (b) PR4/PR5 must be able to A/B both without new bricks. Cherry-pick the branch's **final** kernel versions (staging-free GEMV, rank-count router) — they are bit-identical rewrites of the gated originals. Vendored DeepGEMM `m_grouped_fp8_gemm_nt_masked` (vllm's DeepEP-decode default) is a further alternative — the branch-era "JIT rules it out" verdict is stale now that #489 built DG_NO_TORCH C-ABI wrappers for the indexer MQA logits — but don't build it until a PR5 measurement asks for it.

Why not vllm's masked `BatchedExperts` layout: our shim's dispatch contract is expert-major-contiguous + psum (in-place GEMM, zero gather — kimi-proven), and the merged glm52 metadata kernel already implements it. Adopting vllm's layout would force a different dispatch epilogue for zero benefit.

## Scope

### Kernel ops (`pegainfer-kernels`) — cherry-picks from `feat/glm52-pp8-decode` unless noted

| op | file | what | source |
|---|---|---|---|
| `glm52_router_noaux_tc` | `ops/glm52/router.rs` + `csrc/glm52/glm52_router.cu` | gate GEMM + sigmoid+bias top-8 + unbiased gather + renorm + ×2.5 | cherry-pick |
| `glm52_moe_route` (offsets/scatter/combine) | `ops/glm52/moe_route.rs` + `csrc/glm52/glm52_moe_route.cu` | EP1 dispatch/combine stand-ins, expert-major slots, device-side offsets | cherry-pick |
| `glm52_trtllm_grouped` | `ops/glm52/trtllm_grouped.rs` | safe wrapper over `glm52_trtllm_grouped_fp8_launch_cuda` | cherry-pick (+ verify main's csrc copy matches the branch's runtime-groups refactor) |
| `glm52_moe_fp8_weight_only_gemv` | `ops/glm52/moe_gemv.rs` + `csrc/glm52/glm52_moe_gemv.cu` | bs=1 alternative expert path (see decision above) | cherry-pick |

Adaptation, not rewrite: the picks must land on main's refactored op surfaces (`fp8.rs` ProjWeight, `Glm52DeepGemmScaleLayout`, feature-gated build) — expect mechanical conflicts, not design work.

### Model crate (`pegainfer-glm52`)

- `moe_decode.rs` — cherry-pick base from the branch: `Glm52MoeLayerWeights` (gate bf16 `[256,6144]` + bias f32 `[256]` + experts packed expert-major `[E,n,k]` fp8 + scales + shared-expert ProjWeights) + `glm52_moe_decode_forward`. **Rework needed**: the branch's final version is GEMV-only and allocates per call — PR3's version fronts the grouped chain, takes an arena, and keeps GEMV behind the same signature. `from_host` only; `from_device` against the EP8 slab is PR4 (rank-slab repack decision lives there; the branch's `pack_loaded_expert_fp8_layers` in its `package.rs` is the packer reference — keep one packer shared between test and production paths).
- `dense.rs`, `bookend.rs` — cherry-pick (branch Slice 6); swap bookend GEMMs to the `_graphsafe` variants and device top-1 (`launch_local_top1_batch`), mirroring `kimi-k2 .../forward.rs:102-124`.
- `layer.rs` — new composition (the branch's `decode.rs` is the reference but is PP-stage-shaped): `glm52_decoder_layer_forward` = fused-add norm → MLA decode (+ indexer on `full` layers, `prev_topk` on `shared` layers — **new**, the branch deferred the indexer entirely) → fused-add norm → dense/moe. Threads `(hidden, residual, prev_topk)`. This is the unit PR4's 78-layer loop calls.
- No arena: per-call pool allocs throughout, matching the merged MLA/indexer bricks (see the graph bar above for why this is capture-safe).

### Not in PR3

- DeepEP shim GLM instantiation (256 experts / 32 local / topk 8 / hidden 6144 baked config) + fp8-dispatch question — PR4. Note: the shim's dispatch payload is currently **bf16-only** (`deepep.h:81-103`); GLM would either dispatch bf16 and requant per-rank, or extend the shim with fp8+scale payload (DeepEP elastic supports it). Decide in PR4 with a bandwidth measurement.
- `from_device` constructors / runner & coordinator changes — still rejecting; PR4.
- Scheduler, CUDA-graph capture itself, prefill path — PR5 (prefill rides decode token-by-token as before).
- MTP layer 78 — out of scope for the whole campaign so far.

## Oracle gates

Extends `tools/accuracy/glm52_oracle.py` + the `oracle/mla.rs` pattern (probe consts, input-digest-first, fp8sim precision, `--emit rust`). Note: layer 3 (the first MoE layer) is a **shared-indexer** layer, so the MoE gate uses **layer 6** — the first layer that is both MoE and `full`-indexer, making the decoder-layer oracle self-contained.

1. **`oracle/layer.rs::layer_moe_oracle_gate`** — layer 6, full decoder layer via `--stage layer` (official `GlmMoeDsaDecoderLayer`, fp8sim linears incl. a lazy-dequant `Fp8SimExperts`): probes on `layer_out`, run for BOTH expert paths (Grouped and Gemv) against the same constants; router last-position selection emitted as a debugging reference.
2. **`oracle/layer.rs::layer_dense_oracle_gate`** — layer 0 full-layer probes (residual/norm wiring around the already-gated MLA brick + `fp8_mlp` at 12288).
3. **`oracle/bookend.rs`** — `--stage bookend`: embed-rows digest (exact — bf16 gather), logits probes, per-position argmax (exact).
4. **Regression pins**: Rust-vs-Rust sha256 on `topk_ids` + probe RMS on `layer_out`, same GPU (the indexer gate precedent).

Gate env: jz38 H200 + `/data/models/GLM-5.2-FP8`, same build env as `oracle-harness.md` pitfalls section.

## Execution order

1. Cherry-pick the kernel ops (router, moe_route, trtllm_grouped wrapper, moe_gemv) onto main's op surfaces; synthetic smoke for the grouped chain (multi-group, non-uniform token counts — the scale-relayout layouts are where garbage hides).
2. `moe_decode.rs` (grouped spine + GEMV alternative behind one signature, arena-fed) + `oracle/layer.rs` (layer 3) via the self-contained harness.
3. Cherry-pick `dense.rs`/`bookend.rs`; new `layer.rs` composition with cross-layer top-k threading; layer-0 gate.
4. Bookend taps (embed/final-norm/logits/argmax).
5. fmt/clippy, toxic-review pass, PR.

## Risks / open questions

- **Old-branch gates don't transfer.** Every cherry-picked brick was validated against the prototype npz fixtures (`oracle-harness.md` explicitly retired them). Re-gate everything through `glm52_oracle.py` probes before claiming correctness; treat the picks as "written, plausible" until then.
- **Grouped-path scale relayout**: `fp8_linear` relays activation scales into TRTLLM's col-major TMA layout per GEMM (branch-pinned footgun: raw row-major scales walk off the buffer → inf/garbage); the grouped path needs `a_scale_trtllm` across `m_capacity` rows and per-expert `b_scale` stacked by group. The step-1 smoke must cover this, not just single-group shapes.
- **Grouped vs GEMV fork risk**: keeping two expert paths is only acceptable behind one forward signature with one packer and one oracle gate. If the GEMV path starts demanding its own weight layout, drop it — DeepEP alignment wins.
- **Router tie-break / renorm epsilon** vs torch: assert `topk_ids` exactly first; if tie flake appears, relax to set-overlap like the indexer gate.
- **Combine dtype**: `moeGemm` writes half-precision out; confirm bf16 (not fp16) end-to-end through combine — the oracle's `moe_out` tap will catch a mix-up immediately.

## Execution log (2026-07-03, jz-38 8×H200)

- **Cherry-picks landed as planned** with only mechanical adaptation; main's `glm52_trtllm_grouped_fp8.cu` / `glm52_moe_quant.cu` were already byte-identical to the branch (the runtime-groups refactor is on main), and only the `deepgemm_layout` grouped-offset relayout diff needed porting. `GLM52_ROUTER_WEIGHT_SCALE=1.0` placeholder deleted; `Glm52RouterConfig::glm52()` now defaults to the real 2.5.
- **`glm52_mla_decode_forward` split into `glm52_mla_front` / `glm52_mla_attend`** so the indexer consumes `q_resid` and produces the sparse top-k between the halves (the vllm structure). MLA gate re-run green (64/64) — no regression.
- **Gate design corrections found while wiring** (both are now in the harness):
  1. *Residual-passthrough tolerance*: `layer_out`'s rms is dominated by the unchanged residual, so tolerance must scale with `rms(layer_out - hidden)`.
  2. *bf16 ulp floor*: `layer_out` is stored bf16 at the hidden's magnitude; at input scale 1 the ulp (~0.4% of |hidden|) exceeds the whole layer delta. The layer stage seeds its input at `--input-scale 0.02` (rms_norm makes the delta scale-invariant) and the emitted tol is `max(rel_tol × delta_rms, 3 × ulp)`.
- **Gate results**:
  - bookend: embed rows digest exact, argmax exact ×8, logits 64/64.
  - layer 0 (dense + full indexer): **64/64**.
  - layer 6 (MoE + full indexer): **62/64 on BOTH Grouped and Gemv paths, failing the SAME two probes with the SAME values** — the deviation is upstream of the expert GEMMs. Measured cause: the divergent positions sit on 8th-vs-9th biased-score margins of 1.0–1.7e-4 (median 1.8e-3); the engine's fp8 router logits legitimately flip those near-tied picks vs the fp8sim oracle. The gate allows ≤4/64 outliers capped at 8×tol (dense: zero allowance). Analysis artifacts: `/tmp/layer6_taps.safetensors` + `/tmp/layer6_engine.f32` on jz-38 (`PEGAINFER_GLM52_LAYER_DUMP`).
  - TRTLLM grouped `moeGemm` executed correctly on this branch path (its first run outside the PP8 branch).
- **Environment pitfalls (jz-38)**:
  - `PEGAINFER_DEEPGEMM_ROOT` must point at **`.../DeepGEMM/deep_gemm`** (the python-package dir that contains `include/`), not the submodule root — the JIT `IncludeParser` resolves `$ROOT/include/deep_gemm/...`; the wrong root fails as an opaque `CUDA_ERROR_LAUNCH_FAILED` at the MQA metadata launch. Docs and gate comments corrected.
  - The repo venv's `nvidia/nccl` is back to 2.29.7 (a torch reinstall downgraded it); the `moe`-feature DeepEP shim needs ≥ 2.30.4 → `PEGAINFER_NCCL_ROOT=/root/develop/xingming/nccl-latest/nvidia/nccl` (nvidia-nccl-cu12 2.30.7 wheel).
  - transformers 5.13.0.dev0 is not on PyPI; run the harness with the repo venv python (`.venv/bin/python tools/accuracy/glm52_oracle.py ...`), not `uv run`.
- **Pre-existing breakage fixed en passant**: `tests/checkpoint.rs` used `{event:?}` but `TokenEvent` had no `Debug` (fixed by deriving it); `pegainfer-kernels/tests/glm52_indexer_smoke.rs` had a type error — `--tests` compile of both crates was red on main.
- **Toxic-review round (Request Changes → fixed, all gates re-run green)**: cherry-pick had smuggled `CUBLAS_COMPUTE_32F_PEDANTIC` back into the router (the exact kimi lesson this doc forbids — fixed to `CUBLAS_COMPUTE_32F`; at bs=1 the tie-flip outlier set did not move); scatter/combine validated `expert_offsets.len() > topk` while the kernels index by expert id (fixed to `> n_experts`, a silent-OOB trap); the caller-less plain weight-only GEMV path was deleted (~150 lines, re-cherry-pick from the PP8 branch if PR5 wants it); the post-attention boundary now uses `fused_add_rms_norm_round_batch_into` (bit-identical); contradictory pad-row invariant docs unified on the row-isolation argument (zeroing not required for correctness).

## Read

- `docs/models/glm52/indexer-forward.md`, `oracle-harness.md` — the retained indexer design and gate pattern.
- `vllm/vllm/model_executor/models/deepseek_v2.py:216-407,1154-1325` — MLP/MoE/decoder layer (authoritative model graph).
- `vllm/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:80-161` — routing math.
- `vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py` — the graph-first DeepEP v2 decode contract (what PR4 must preserve; `do_cpu_sync=False`, device-side counts, pow2 token cap).
- `pegainfer-kimi-k2/src/runner/moe_deepep.rs` — the in-repo graph-quiet DeepEP decode shape (state arena, stream discipline, worst-case buffers).
- `pegainfer-kernels/csrc/deepep/deepep_shim.cu` + `src/ops/deepep.rs` — the DeepEP v2 elastic shim PR4 re-bakes for GLM.
- `pegainfer-kernels/src/ops/glm52/{deepgemm_grouped.rs,moe_quant.rs}` + `csrc/glm52/glm52_trtllm_grouped_fp8.cu` — merged metadata/quant ops and the compiled-but-unwrapped grouped GEMM.
- `transformers/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py:456-621` — `GlmMoeDsaMLP/TopkRouter/Experts/MoE/DecoderLayer` (oracle source).
- `feat/glm52-pp8-decode:` `pegainfer-glm52/src/{moe_decode.rs,dense.rs,bookend.rs,decode.rs,weights/package.rs}` + `pegainfer-kernels/csrc/glm52/{glm52_router.cu,glm52_moe_route.cu,glm52_moe_gemv.cu}` + `ops/glm52/{router.rs,moe_route.rs,moe_gemv.rs,trtllm_grouped.rs}` — the cherry-pick sources. That branch's full PP8 forward decoded fluent English on 8×H200, so the composed math is known-good; only its PP spine and npz-fixture gates are discarded.
- `feat/glm52-pp8-decode:docs/models/glm52/pp-decode.md` — the branch's measured perf record. Facts that carry over to this composition: use `fused_add_rms_norm_round_batch_into` across the layer loop (bit-identical, −0.35 ms there); clamp FlashMLA `num_sm_parts` to 32 for bs=1 (the 132-way default over-splits the combine; U-curve measured); cuBLAS is near-optimal for the bs=1 small-N/short-k GEMVs (gate, absorb) — don't hand-roll those.
