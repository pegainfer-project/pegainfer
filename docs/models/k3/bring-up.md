# Kimi K3 bring-up

**TL;DR**: New model line (`--features k3`). Decode is end-to-end: the full
93-layer model serves at `--k3-ep-size 4` with no flags set (free-running
per-rank engines, 189 GiB per rank), producing coherent greedy text over
`/v1/completions`. Routed experts are **one fused DeepGEMM MegaMoE(situ)
launch per MoE layer at every world size** — dispatch, both FP8xFP4 GEMMs, the
situ activation, the mid-quantization and the combine inside a single
persistent kernel that pairs the ranks over NVLink itself, so an EP step issues
**zero host collectives**. EP4 is bitwise-equal to the single-rank fused path
and bitwise-invariant to peer traffic. Single-rank executor: buckets up to
`B = 128`, per-bucket CUDA graphs default-on, token-matching a certified
4-layer greedy golden (39/40 exact under the fused kernel, 38/40 under the
masked-chain anchor, misses inside a structural ≤2-ULP noise floor — see
below). Kernel surface: twelve batched TileLang decode families + DeepGEMM
FP8xFP4 AOT shims (fused MegaMoE and the masked grouped GEMM) behind
`pegainfer-kernels`'s `k3` feature; dense projections on cuBLASLt. Next: CUDA
graphs over the EP4 fused path.

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
- **MoE decode = the fused MegaMoE(situ) kernel** from the DeepGEMM fork
  (`openinfer` branch), at every world size (see below). Bring-up went through
  glm52's stepwise masked chain first (masked FP8xFP4 grouped GEMM W13 →
  situ+requant → masked GEMM W2 → combine); that chain survives as the numerics
  anchor the fused kernel is A/B'd against, not as a transport. The SM90-era
  rejection of mega MoE (`docs/models/glm52/whole-step-decode-graph.md`)
  explicitly deferred the SM100 FP8xFP4 variant, which is this one.
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
the 12 batched TileLang families, the fused MoE launch), `buffers.rs` owns the
state pools, `mod.rs` owns graphs and the `StepExecutor` impl.

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
(`PEGAINFER_K3_TEST_224`), through **both** routed-expert transports: 39/40
argmax match exactly under the fused kernel, 38/40 under the masked-chain
anchor. Every miss is a step the reference itself decided by ≤1 bf16 ULP, and
the sampled token stays in the reference top-5. The floor is structural: the
reference computes routed experts as bf16×MXFP4 GEMV, while this engine
quantizes activations to FP8-e4m3 with UE8M0 group scales (median per-step max
logit deviation 2 bf16 ULP). Within a bucket, results
are exact: row independence (padding rows and real neighbours don't change a
seat's stream), graph-vs-eager, and multi-slot staggered admission are all
gated bitwise. Across buckets, streams may diverge at ≤2-ULP coin-flip steps
(different cuBLASLt tile shapes ⇒ different summation order) — the
cross-bucket gate holds to the fixture with the same noise-floor rule.

4-layer, single-rank perf: see the per-layer table below (the FP4 expert GEMMs
dominate — known to lose to a GEMV below ~8 rows/expert). Graphs buy ~4.5%;
the step is not launch-bound. Absolute numbers move a few percent with box
load, so compare within a session, not across.

## EP4: free-running ranks

The full model does not fit one GPU (routed experts alone are ~354 GB of
MXFP4), so multi-GPU EP is the only serving shape. The architecture is the
free-running design glm52 migrated to (`docs/models/glm52/free-running-dp.md`)
— K3 adopts it from day one rather than building a coordinator and deleting
it later:

- **Each rank is an autonomous engine.** `start_with_executors` gives one
  scheduler partition per rank; the frontend routes requests across
  partitions. There is no cross-rank host protocol of any kind.
- **The only coupling is inside the step**, and it is a compile-time constant:
  every rank launches the same sequence at the same shapes on every step it
  takes. The scheduler calls `decode()` unconditionally — an idle rank pads the
  step rather than skipping it — and prefill is sequential decode-shaped steps,
  so a rank's prefill step pairs against a peer's decode step with no
  negotiation.
- **A step error is group-fatal** (log + exit). A rank that skips a launch
  leaves every peer inside a device barrier it will never reach; there is no
  state from which the group can serve a correct next token. GPU gates run one
  per process (context lifetime is the process).
- **EP forces eager.** Capture works on the single-rank fused path, but a
  captured cross-rank launch has not been replayed here.

*History*: EP first shipped as a fixed four-collectives-per-MoE-layer NCCL
chain (allgather the dispatch, run the masked chain over the fleet's batch
through an expert window, scatter to entry-major staging, sum-allreduce,
combine). It was retired once the fused kernel covered `ep_size 4`, and the
code is gone. Three things it established are load-bearing and survive it: the
free-running architecture above; **bitwise equality to single rank as the
sharding criterion** (not a tolerance — expert windowing leaves each computed
row byte-identical, and disjoint support makes every cross-rank reduction
`0 + x`), now carried by `tests/ep_mega_oracle.rs`; and the fact that
`ep_size × max_batch ≤ masked_cap` was a *chain-era* constraint — the fleet's
whole batch had to fit one masked tile — which the fused kernel does not have,
so it is gone too. The chain's per-step collective ledger went with it for a
reason worth keeping: plain NCCL has no device-side timeout, so a mispaired
chain was a silent wrong answer and the ledger was the only detector. The fused
kernel's NVLink barrier times out at 60 s and asserts, so what remains is a
cheap host-side launch-count guard (92 launches per step at EP4) that names the
rank that fell behind instead of leaving its peers to time out anonymously.

## MegaMoE(situ): the routed-expert transport

The routed experts are one DeepGEMM MegaMoE launch per MoE layer, fusing
dispatch, both FP8xFP4 GEMMs, the situ activation, the mid-quantization and the
weighted combine. AOT-instantiated from the vendored device headers exactly
like the masked GEMM — no JIT, no torch (`csrc/k3/k3_mega_moe_sm100.cu`). It is
the default and the only production path; there is no flag.

- **The masked chain survives as the numerics anchor, test-only.** It is what
  `k3_moe_chain_gate` checks against an f32 reference and what `golden_decode`
  A/Bs the fused kernel against, and it is reachable only through
  `K3MoeTransport::masked_chain_for_tests()` — a `#[doc(hidden)]` constructor,
  deliberately not an env var, so no stray environment can change which
  arithmetic serves a request. Single rank only.
- **The two are not bit-equivalent, by construction.** MegaMoE multiplies the
  routing weight into the activation *before* the down projection (the chain
  applies it at combine) and mid-quantizes per 32 elements rather than per
  128. Each is held to the golden fixture rather than to the other.
- **One flat symmetric slab** (260 MiB at 224 experts) holds twelve
  differently-typed regions: the FP8 activation and its packed scales, the
  routing pair, and the L1/L2 ring buffers. Its size and offsets are pure host
  arithmetic over the shapes, the candidate `BLOCK_M` set and the SM count —
  and they are kernel *template parameters*, so a rounded-up allocation is
  wrong, not merely wasteful. At `ep_size 1` the slab is a plain device
  allocation (260 MiB): the kernel's cross-rank barriers compile down to
  grid-local synchronisation, so no IPC or NVSHMEM handle is involved. At
  `ep_size 4` each rank owns one (209.8 MiB, ring 17280 tokens / SF ring
  276480) on its own device and the world exchanges bare base pointers.
- **The expert bank carries one layout or the other, never both** — a rank
  holds 84-189 GiB of experts. `K3ExpertBankForm` picks the layout at build
  time: the mega form interleaves the fused gate|up rows at granularity 8 and
  adds the UTCCP transpose to both scale-factor tensors.
- **CUDA graphs stay on at `ep_size 1`.** The fused kernel's grid sync is
  atomics over the slab, not a cooperative launch, and its TMA descriptors are
  built from persistent pointers — capture and replay work unchanged, verified
  over every bucket. EP4 stays eager for now.
- **Bit-level parity gate.** `pegainfer-kernels/tests/k3_mega_parity_gate.rs`
  replays a fixture dumped from DeepGEMM's own Python path (inputs in
  checkpoint form plus the kernel's output *bits*) through the whole Rust
  pipeline — scale prepare, both weight transforms, activation quant, launch —
  and requires `y` to be bit-identical. That is what pins the twelve offsets,
  the packed-UE8M0 spelling and the block/stage/SM launch configuration to
  what the Python wrapper would have chosen. Point
  `PEGAINFER_K3_MEGA_FIXTURE` at the dump directory; unset, it skips.

Step-time snapshot (4-layer truncated model, `PEGAINFER_K3_LAYERS=4`, one
GB300, ms per layer):

| bucket | chain eager | mega eager | chain graphs | mega graphs |
|--------|-------------|------------|--------------|-------------|
| 1      | 0.538       | 0.508      | 0.513        | 0.483       |
| 16     | 0.997       | 0.919      | 0.970        | 0.895       |
| 128    | 2.635       | 2.507      | 2.607        | 2.482       |

### MegaMoE at ep_size 4

At four ranks a MoE layer is still one launch. A rank quantizes its own live
rows into its own slab and launches; the kernel dispatches across the world
over NVLink, computes every expert it owns for whoever sent it work, and
combines each token back to the rank that owns it. The host issues no
collective at all — an EP group builds no communicator — and the
write-then-launch ordering the inputs need is plain stream order on the rank's
own stream.

- **The rank count is a template parameter**, not a runtime dimension: it sets
  the ring capacities and the experts-per-rank divisor. `1` and `4` are
  instantiated (`K3_MEGA_EP_SIZES`); anything else is refused at construction.
- **One fixed block config for every rank and every step at EP4**, derived from
  the protocol maximum (`num_max_tokens_per_rank = 384`) rather than the live
  token count: BLOCK_M 192 / BLOCK_K 128. Two reasons. Nothing in the kernel
  forces the world to agree on a config, and heterogeneous tiling across a
  collective launch is unverified territory. And a fixed config makes a row's
  tile shape independent of how much traffic its peers are sending — which is
  what turns traffic invariance into a bitwise claim instead of a tolerance.
  The small-batch cost is accepted. `ep_size 1` keeps the live-config
  selection, because that is what keeps it bit-identical to upstream's Python
  path.
- **Cross-rank addressing is layout-only.** Every access a rank makes into a
  peer's slab targets a region whose offset and stride come from
  `MegaMoEBuffer(hidden, intermediate, ranks, experts, max_tokens, topk,
  ring_tokens, sf_ring_tokens)` — no BLOCK_M/BLOCK_N/stage terms anywhere. The
  block-config-dependent quantities (pool block indices, the L1/L2 ring
  counters, SF paging) are only ever used against the rank's OWN workspace and
  rings. A sender never needs the receiver's block config; that is what makes
  the paragraph above a choice rather than a requirement.
- **`num_tokens = 0` is a first-class case.** An idle rank's launch still
  serves its local experts for its peers' tokens (the pull loop iterates the
  sum over all ranks) and still meets both NVLink barriers; only the topk read
  and the final combine, both bounded by `num_tokens`, do nothing. The
  free-running contract is therefore unchanged: an empty batch still launches
  all 92 mega layers.
- **Peer access needs two grants, not one.** `cudaDeviceEnablePeerAccess` lets
  a context address a peer's memory; it explicitly does **not** cover
  stream-ordered memory-pool allocations, and every buffer here comes from
  `cudaMallocAsync`. The owner must also `cudaMemPoolSetAccess` its pool for
  each peer, *before* allocating the slab — the grant does not reliably reach
  allocations that predate it. Getting only the first grant reads as
  `CUDA_ERROR_ILLEGAL_ADDRESS` inside the kernel, not as an error at setup.
- **Startup is the rendezvous itself.** Each rank publishes its slab base only
  after the allocation is zeroed and synchronised; the first step blocks until
  the whole table is in. No launch can precede the last allocation.
- **Gates** — `pegainfer-k3/tests/ep_mega_oracle.rs` is THE EP oracle (one gate
  per process, four GPUs): EP4 rank 0 against a single-rank mega run is
  **bit-identical** — all 40 forced-replay tokens and all 163840 final logits —
  with three idle peers; and rank 0 with busy peers against rank 0 with idle
  peers is **bit-identical** too. The first was only required to clear the
  fixture's noise floor (the two world sizes pick different block configs);
  bitwise is what it measured, which says the MMA's K-accumulation order does
  not depend on BLOCK_K for these shapes.

Full-depth serve (93 layers, 4xGB300, `--k3-ep-size 4`,
`PEGAINFER_K3_MAX_BATCH=16`, no other flags): all four ranks load 189.1 GiB and pair over peer
access, greedy completions come back as coherent English, four concurrent
requests across the four partitions finish together, and a 60 s idle hold of
free-running padding steps followed by a fresh request serves normally with
zero error lines. Single stream, 76 decode-shaped steps (12-token prompt + 64
generated), median of three:

| EP4 MoE transport | ms/step | 4 concurrent, 69 steps |
|-------------------|---------|------------------------|
| NCCL chain, since retired (368 collectives/step) | 54.6 | 3.90 s |
| MegaMoE (0 collectives/step) | 43.3 | 3.03 s |

Both emitted the same greedy text. That A/B is why the chain was retired; it is
recorded here as the measurement, not as a configuration you can still select.

## Next

Graphs over the EP4 fused path (the ranks=1 path already captures), then
launch-ahead, paged KV for MLA (current MLA cache is fixed `max_ctx = 128` per
seat) and real prefill.
