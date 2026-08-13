# Kimi K3 bring-up

**TL;DR**: New model line (`--features k3`), bring-up stage: config probe + weight
loader + step-contract scheduler skeleton land first; model execution is not
wired yet (requests fail with an explicit message). Kernel surface: thirteen
batched TileLang decode families (every non-GEMM step, `B` a compile-time
bucket) + DeepGEMM masked FP8xFP4 grouped-GEMM AOT shim, both compiled behind
`pegainfer-kernels`'s `k3` feature; dense projections go to cuBLASLt.

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

## Next

Wire the model executor: weights → device model build (SF prepare, kv_b-style
transforms), KDA/MLA decode steps with oracle gates, then the EP4 MoE chain.
The decode-step kernel surface is in place, so the executor's first job is
buffer ownership and the launch sequence, mirroring the certified engine.
Numerics gate #1: the masked FP8xFP4 GEMM TMA/barrier byte accounting (see the
shim header in `pegainfer-kernels/csrc/k3/`).
