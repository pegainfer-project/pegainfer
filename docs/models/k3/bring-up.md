# Kimi K3 bring-up

**TL;DR**: New model line (`--features k3`). Single-rank decode is wired
end-to-end: config probe + weight loader + step-contract scheduler + a batched
decode executor (buckets up to `B = 128`, per-bucket CUDA graphs default-on)
that token-matches a certified 4-layer greedy golden (38/40 exact, 2 steps
inside a structural ≤2-ULP noise floor — see below). Kernel surface: thirteen
batched TileLang decode families (every non-GEMM step, `B` a compile-time
bucket) + DeepGEMM masked FP8xFP4 grouped-GEMM AOT shim, both compiled behind
`pegainfer-kernels`'s `k3` feature; dense projections go to cuBLASLt. Next:
EP4 MoE (allreduce oracle first, MegaMoE swap-in after).

Last touched: 2026-08

## Architecture

93 layers, hidden 7168: 69 KDA linear-attention layers (fixed-size recurrent
state + 4-tap conv, replace-in-place) + 24 MLA layers (paged KV, NoPE, layers
`{3,7,…,91} ∪ {92}` zero-based); layer 0 dense; 92 latent-MoE layers (hidden
7168 → latent 3584 → routed experts inter 3072, top-16, situ activation →
RMSNorm → up) + 2 shared experts on hidden. Routed experts are MXFP4
(group-32 e8m0 scales); everything else bf16. No MTP. The checkpoint is a
multimodal wrapper (`KimiK3ForConditionalGeneration`, text tower under
`text_config`, weights prefixed `language_model.`); we serve text only.

Two published checkpoints differ only in `num_experts` (224 vs 896). The
224-expert variant at EP4 (56 experts/rank) is shape-isomorphic to the full
model at EP16 — that is the development vehicle.

## Decisions and why

- **Step contract, not the legacy Handle contract**: K3 starts on
  `pegainfer_frontend::engine::Scheduler` (qwen3 is the other user) since the
  legacy contract is slated for deletion.
- **MoE decode = glm52's stepwise chain with an FP4 B side** (DeepEP dispatch →
  masked FP8xFP4 grouped GEMM W13 → situ+requant → masked GEMM W2 → combine).
  The fused MegaMoE(situ) kernel exists in the DeepGEMM fork (`openinfer`
  branch) and is a planned phase-2 swap-in — the SM90-era rejection of mega
  MoE (see `docs/models/glm52/whole-step-decode-graph.md`) explicitly deferred
  the SM100 FP8xFP4 variant.
- **MXFP4 loads raw**: the checkpoint's packed-nibble format is
  byte-isomorphic to DeepGEMM's FP4 K-major B layout (verified numerically at
  1e-6 against a dequant reference on real weights), so the loader uploads
  payload and e8m0 scale bytes as-is; the SF relayout to DeepGEMM's packed
  i32 form is a device-side build step (`k3_fp4_sf_prepare`).
- **KV story is dual-pool**: paged KV (kv-store `BlockPool`) for the 24 MLA
  layers plus a qwen35-style fixed-size slot pool for KDA recurrent state.
  Prefix caching ships disabled: KDA state is not recomputable from tokens
  (`docs/subsystems/kv-cache/design.md`, bounded class).
- **TileLang kernels are generated at build time** from vendored, certified
  kernel definitions in `pegainfer-k3/kernels/` (three tiers: live
  generation → pre-generated dir → NOT_SUPPORTED stubs). The definitions'
  spelling is certified against the HF reference in a separate harness and
  must not drift. The set covers every non-GEMM step of a decode iteration —
  norms, the bf16 landings of the framework GEMMs, conv+silu, the KDA delta
  rule, MLA attention, router, situ, combine and the attention-residual mix —
  so the executor composes launches rather than reimplementing spellings.
  Dense projections run on cuBLASLt and the routed experts on the DeepGEMM
  masked grouped-GEMM chain, so no GEMV is generated. Every shape is a static
  compile dimension, batch size included: `B = 1` is a bucket whose per-row
  spelling is the certified single-row kernel, gated bitwise upstream, so
  single-stream and high-concurrency decode share one kernel set. See
  `pegainfer-kernels/KERNELS.md` for the instantiation matrix.

## Executor (single-rank decode, wired)

`pegainfer-k3/src/executor/` composes the certified kernels into the decode
step: `step.rs` is a line-by-line port of the certified reference engine's
launch sequence (dense projections on cuBLASLt with banded/offset landings,
the 13 batched TileLang families, the 7-step masked MoE chain), `buffers.rs`
owns the state pools, `mod.rs` owns graphs and the `StepExecutor` impl.

- **Batching**: seat `i` is row `i` of every state pool. Buckets
  `{1,2,4,8,16,32,48,64,96,128}`; one CUDA graph per `(bucket, parity)`
  (KDA recurrent state ping-pongs across two slabs, so parity is part of the
  graph identity). Graphs are default-on; `PEGAINFER_K3_CUDA_GRAPH=0` escapes
  to eager. H2D feed (`token_ids`, `context_len`, `cache_row`) and the single
  argmax D2H stay outside capture.
- **State contract**: a padding row is stepped like any live row, so its
  recurrent state advances — a seat's state is only meaningful while the seat
  is in *every* batch. The scheduler preserves this (running requests decode
  every step; `prefill` resets the seat at admission). Prefill runs on a
  separate one-row pool and hands its state over by row copy.
- **Bring-up flags**: `PEGAINFER_K3_LAYERS` (layer truncation),
  `PEGAINFER_K3_MAX_BATCH`, `PEGAINFER_K3_CUDA_GRAPH`.

### Gates and the noise floor

`tests/golden_decode.rs` replays a 4-layer greedy golden fixture (16-token
prompt + 24 greedy steps) against the 224-expert checkpoint
(`PEGAINFER_K3_TEST_224`): 38/40 argmax match exactly; the two misses are the
only two steps the reference itself decided by ≤1 bf16 ULP, and the sampled
token stays in the reference top-5. The floor is structural: the reference
computes routed experts as bf16×MXFP4 GEMV, while this engine quantizes
activations to FP8-e4m3 with UE8M0 group scales for the masked grouped GEMM
(median per-step max logit deviation 2 bf16 ULP). Within a bucket, results
are exact: row independence (padding rows and real neighbours don't change a
seat's stream), graph-vs-eager, and multi-slot staggered admission are all
gated bitwise. Across buckets, streams may diverge at ≤2-ULP coin-flip steps
(different cuBLASLt tile shapes ⇒ different summation order) — the
cross-bucket gate holds to the fixture with the same noise-floor rule.

4-layer, single-rank perf: B=1 ≈ 0.51 ms/layer with graphs (masked FP4 chain
dominates — known to lose to a GEMV below ~8 rows/expert); B=128 ≈ 12.3k
tok/s. Graphs buy ~4.5% (the step is not launch-bound).

## Next

EP4 MoE: first the replicated-batch expert-partial allreduce oracle (bitwise
scheme verified in the reference harness, zero new kernels, correctness only),
then the fused MegaMoE(situ) swap-in (AOT instantiation + symm-buffer
plumbing) with the stepwise chain kept as the A/B baseline. Launch-ahead
(overlap next step's feed with current compute) needs the scheduler to hand
the next batch over before reading this step's tokens — the executor side is
ready. Paged KV for MLA (current MLA cache is fixed `max_ctx = 128` per seat)
and real prefill come after.
