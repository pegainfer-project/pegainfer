# Kimi K3 bring-up

**TL;DR**: New model line (`--features k3`). Decode is end-to-end: full
93-layer model serves at `--k3-ep-size 4` (free-running per-rank engines, a
fixed 4-collectives-per-MoE-layer NCCL chain as the only coupling, 189 GiB
per rank), producing coherent greedy text over `/v1/completions`, and the EP4
sharding is **bitwise-equal to single-rank** (all 40 gate tokens and all
163840 final logits, idle and busy peers alike). Single-rank executor:
buckets up to `B = 128`, per-bucket CUDA graphs default-on, token-matching a
certified 4-layer greedy golden (38/40 exact, 2 steps inside a structural
≤2-ULP noise floor — see below). Kernel surface: thirteen batched TileLang
decode families + DeepGEMM masked FP8xFP4 grouped-GEMM AOT shim behind
`pegainfer-kernels`'s `k3` feature; dense projections on cuBLASLt. Next:
MegaMoE(situ) swap-in.

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

## EP4: free-running fixed-chain oracle (wired, bitwise)

The full model does not fit one GPU (routed experts alone are ~354 GB of
MXFP4), so multi-GPU EP is the only serving shape. The architecture is the
free-running design glm52 migrated to (`docs/models/glm52/free-running-dp.md`)
— K3 adopts it from day one rather than building a coordinator and deleting
it later:

- **Each rank is an autonomous engine.** `start_with_executors` already gives
  one scheduler partition per rank; the frontend routes requests across
  partitions. There is no cross-rank host protocol of any kind.
- **The only coupling is the step's fixed collective chain.** The scheduler
  calls `decode()` unconditionally — an idle EP rank pads the step rather
  than skipping it, so collective pairing (which is by entry order) is a code
  structure guarantee. K3 gets the fixed-chain discipline for free: prefill
  is sequential decode-shaped steps, so every step of every rank runs the
  identical 4-collectives-per-MoE-layer sequence, and a rank's prefill step
  pairs against a peer's decode step with no negotiation. A per-step ledger
  `ensure!`s the collective count is the compile-time constant.
- **Numerics are bitwise by construction** (`executor/ep.rs`, gated by
  `tests/ep_oracle.rs`): protocol-max allgather of latents + router topk
  (padding rows constructively zero/`-1`), expert-windowed route metadata
  (entries outside the rank's window go inactive; the intra-expert compaction
  order is unchanged, so every computed row is byte-identical to the
  single-rank run), local masked chain over the global batch, dense scatter
  to entry-major staging (each entry owned by exactly one rank), bf16
  allreduce — disjoint support means every reduction is `0 + x`, exact in
  any order — and an entry combine whose accumulation loop is copied verbatim
  from the masked combine. Gate: EP4 rank 0 vs single-rank, all 40 greedy
  tokens and all 163840 final logits bit-identical, with idle peers and with
  peers running their own traffic.
- **Lessons imported from glm52/qwen3 instead of re-learned**: EP forces
  eager (collectives under graph capture need warmup-per-size, two-phase
  cross-rank pre-capture and an abort watchdog — deferred to the MegaMoE
  phase); NCCL comms are minted on each rank's own thread after a
  condvar-timeout id rendezvous; all ranks finish loading weights before any
  comm init; an EP step error is group-fatal (log + exit — plain NCCL has no
  device timeout, survivors would pair against the wrong step forever);
  `ep_size × max_batch ≤ masked_cap` is enforced at construction (worst case
  one expert claims every global token). GPU gates run one per process
  (comm/context lifetime is the process).

This chain is the correctness oracle and the fallback; the fused
MegaMoE(situ) kernel is the planned production swap-in, A/B'd against it.

Full-depth serve (93 layers, 4×GB300, `--k3-ep-size 4`,
`PEGAINFER_K3_MAX_BATCH=16`): each rank loads in ~68 s (backbone 100.5 GiB +
56 experts 84.2 GiB + 4.4 GiB other = 189.1 GiB), all four ranks join the
NCCL group at 368 collectives/step, and greedy completions come back as
coherent English. Four concurrent requests across the four partitions each
finish in ~2.9 s (40 tokens, ~72 ms/step — eager oracle chain, perf is not
its job); a 60 s idle hold of free-running padding steps followed by a fresh
request serves normally with zero error lines. The checkpoint must be
readable by the serving user — its ACLs may effectively be owner-only, which
surfaces as `Permission denied` on the first shard.

## Next

MegaMoE(situ) swap-in (AOT instantiation + symm-buffer plumbing), then
graphs-with-collectives (qwen3 recipe), launch-ahead, paged KV for MLA
(current MLA cache is fixed `max_ctx = 128` per seat) and real prefill.
