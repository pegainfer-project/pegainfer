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
below). MLA KV is a **paged latent cache** — 576-wide `[kv_lora|rope]` rows in
64-token pages behind per-slot block tables, 27.6 KB/token — decoded by an
absorbed-MLA CUDA kernel with no compile-time context cap (the old 1.47
MB/token expanded slot cache and its `max_ctx = 128` are gone). Kernel surface:
eleven batched TileLang decode families + the hand-written paged-attention
kernel + DeepGEMM FP8xFP4 AOT shims (fused MegaMoE and the masked grouped GEMM)
behind `pegainfer-kernels`'s `k3` feature; dense projections on cuBLASLt.
Prefill is **chunked at the MegaMoE protocol width**: up to 4224 consecutive
prompt tokens per batched step (a 4096-token prompt is ONE step), with the
KDA recurrence crossing each chunk as one **vendored FlashKDA** chunkwise
forward per layer (MoonshotAI, MIT, `third_party/flash-kda`) and the MLA
layers served by **FlashMLA's SM100 dense FMHA** over kv_b-expanded K/V in
fixed workspace (vLLM's recipe; the paged latent stays the only persistent
storage). Chunk steps skip the batched epilogue — the boundary token is
sampled once, at one row, after the final chunk. 6x-247x TTFT over per-token
stepping at the 4-layer snapshot (2048 tokens: 6377 → 25.8 ms). Warm-cache
startup now has one opt-in fast path combining pinned double-buffer upload with
concurrent local-rank load/build; it cuts the full-depth EP4 process start →
HTTP ready **126.27 → 22.08 s (5.72x)**, and the same rank-local path serves
the full 896-expert EP16 fleet. Next: CUDA graphs over the EP4 fused path,
kv-store integration.

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

## Weight-loading startup

The K3 loader already shared GLM5.2's rank-resident load plan, shard-at-a-time
mmap lifetime events, expert-major placement, and pinned double-buffer
implementation. The missing production wiring was above that layer: the
pinned uploader was always passed `false`, and a process completed one local
rank's weight load, model build, executor allocation, and optional DSpark load
before starting the next rank. Mirroring GLM5.2, one production switch enables
the complete fast path:

- `--k3-weight-staging`: load/build every rank hosted by this process
  concurrently, with two 32 MiB pinned slots and four persistent fill workers
  per active rank loader. It operates on the hosted `--k3-ranks` slice, not
  the global EP width, so EP4/8/16/32/64 use the same path and map their local
  ranks to devices 0..N.

The 2026-08-25 same-binary EP4 A/B used the full 93-layer 224-expert
checkpoint (189.08 GiB and 33,372 tensors per rank), four GB300s, warm page
cache, one decode slot/rank, and a real HTTP completion after readiness. All
four arms produced the same completion bytes:

| Local-rank scheduling | Uploader | Local critical path | HTTP ready | Critical-path speedup |
| --- | --- | ---: | ---: | ---: |
| serial | pageable mmap | 125.16 s | 126.27 s | 1.00x |
| serial | pinned double buffer | 47.15 s | 48.12 s | 2.65x |
| parallel | pageable mmap | 44.54 s | 46.11 s | 2.81x |
| parallel | pinned double buffer | **20.96 s** | **22.08 s** | **5.97x** |

In the winning arm, the four rank weight loads were 15.96--18.58 s; model
build was below one second/rank. This rules out the per-MoE-layer weight
repack syncs as the current startup bottleneck: source-page materialization
and H2D dominate.

The CPU-affinity boundary is part of this result. A first Slurm run allocated
the whole 144-core node but left `CPUs/Task=1`, pinning the process and all 16
fill workers to CPU 0; the same parallel+pinned arm then took 100.61 s. The
correct launch used `--cpus-per-task=144 --cpu-bind=none`. Startup now warns
when the process affinity exposes fewer CPUs than the selected rank-loader +
pinned-fill teams can use, rather than silently making a starved run look like
a loader regression.

The EP-width invariant was then closed on the full 896-expert model at EP16:
four processes each hosted four ranks with parallel+pinned loading, all four
HTTP endpoints reached ready, and the same prompt returned byte-identical
text from every endpoint. Forced-cold/partially resident node critical paths
were 134--194 s. On the immediate restart, three nodes whose selected pages
survived reached 18.60/23.73/29.20 s; one node whose selected pages did not
survive took 198.65 s despite a large aggregate page cache. Therefore the fast
path remains opt-in: changing the default needs residency detection over
the actual rank load plan (for example `mincore`), not a host-level free/cache
memory heuristic.

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
- **KV story is dual-pool**: paged KV for the 24 MLA layers (executor-owned
  latent page pool today; kv-store `BlockPool` integration is a later
  milestone) plus a qwen35-style fixed-size slot pool for KDA recurrent state.
  Prefix caching ships disabled: KDA state is not recomputable from tokens
  (`docs/subsystems/kv-cache/design.md`, bounded class).
- **TileLang kernels are generated at build time** from vendored, certified
  kernel definitions in `pegainfer-k3/kernels/` (three tiers: live
  generation → pre-generated dir → NOT_SUPPORTED stubs). The definitions'
  spelling is certified against the HF reference in a separate harness and
  must not drift. The set covers every non-GEMM, non-attention step of a
  decode iteration — norms, the bf16 landings of the framework GEMMs,
  conv+silu, the KDA delta rule, router, situ, combine and the
  attention-residual mix — so the executor composes launches rather than
  reimplementing spellings. Dense projections run on cuBLASLt, the routed
  experts on the DeepGEMM masked grouped-GEMM chain, and MLA decode on the
  hand-written paged kernel below, so neither a GEMV nor an attention family
  is generated. Every shape is a static
  compile dimension, batch size included: `B = 1` is a bucket whose per-row
  spelling is the certified single-row kernel, gated bitwise upstream, so
  single-stream and high-concurrency decode share one kernel set. See
  `pegainfer-kernels/KERNELS.md` for the instantiation matrix.

## Executor (single-rank decode, wired)

`pegainfer-k3/src/executor/` composes the certified kernels into the decode
step: `step.rs` is a line-by-line port of the certified reference engine's
launch sequence (dense projections on cuBLASLt with banded/offset landings,
the 11 batched TileLang families, the paged MLA attention, the fused MoE
launch), `buffers.rs` owns the state pools and the paged KV pool, `mod.rs`
owns graphs and the `StepExecutor` impl.

- **Batching**: seat `i` is row `i` of every state pool. Buckets
  `{1,2,4,8,16,32,48,64,96,128}`; one CUDA graph per `(bucket, parity)`
  (KDA recurrent state ping-pongs across two slabs, so parity is part of the
  graph identity). Graphs are default-on; `PEGAINFER_K3_CUDA_GRAPH=0` escapes
  to eager. H2D feed (`token_ids`, `context_len`, `kv_row`, the KV block
  table) and the single argmax D2H stay outside capture.
- **State contract**: a padding row is stepped like any live row, so its
  recurrent state advances — a seat's state is only meaningful while the seat
  is in *every* batch. The scheduler preserves this (running requests decode
  every step; `prefill` resets the seat at admission). Prefill runs on a
  separate pool and hands its state over by row copy (see Chunked prefill).
- **Bring-up flags**: `PEGAINFER_K3_LAYERS` (layer truncation),
  `PEGAINFER_K3_MAX_BATCH`, `PEGAINFER_K3_CUDA_GRAPH`,
  `PEGAINFER_K3_MAX_CTX` (per-slot context ceiling, default 4096).

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

### Chunked prefill

A prompt is walked in chunks of up to `chunk_tokens` consecutive tokens; each
chunk is one batched step whose *rows are the chunk's tokens* (`executor/`:
`prefill_inner` → `step::k3_prefill_chunk_step`). The cap defaults to the
**MegaMoE protocol maximum (4224 rows**, clamped to `max_ctx`; the masked
chain stays at `max_batch` — its layout reserves 128 rows per expert): the
batched TileLang families carry five prefill-only buckets
(`256/512/1024/2048/4224`, every family except the decode-only `kda_core`),
the fused MegaMoE launch was already protocol-max, and the GEMMs take rows
at runtime. Chunk steps **skip the batched epilogue** entirely — a
chunk-wide lm_head would cost ~10 TFLOP and a 4 GB vocab buffer per step for
rows nobody reads — and the boundary token is sampled once after the final
chunk by `k3_prefill_boundary_sample`: collapse the last live row's
snapshots to row 0 (the same collapse the decode handover needs anyway),
then a `b = 1` pass of the ordinary epilogue. The vocab-wide scratch
therefore stays sized by the decode rows; the per-layer scratch spans the
chunk bucket (~7 GB extra at 4224 — the price of the wide step).
What made the chunking itself nearly free:

- **MLA is one dense FMHA call per layer** over FlashMLA's SM100 CUTLASS
  forward (`third_party/FlashMLA/csrc/sm100/prefill/dense`, shimmed torch-free
  in `csrc/k3/k3_flash_mla_prefill.cu`), following vLLM's chunked-prefill MLA
  recipe: gather the cached latent rows into fixed workspace (the chunk's own
  rows were just appended, so the cache holds the whole `[context | chunk]`
  span), expand through `kv_b` (one cuBLAS GEMM; V is read as a strided view
  into the expansion, K assembled by a small broadcast kernel), and let
  `CausalMask<false>`'s bottom-right alignment — Q rows sit at the *end* of
  the KV axis — give chunk token `i` exactly `context + i + 1` visible keys.
  While the workspace covers `max_ctx` (4096 at bring-up, ~341 MB scratch),
  vLLM's context loop and LSE merge degenerate to this single call; the
  W-chunked loop is the extension point if `max_ctx` outgrows the workspace.
  **The expanded K/V is per-chunk scratch, not a cache** — the retired 1.47
  MB/token expanded slot cache stays retired. The first cut of this phase
  reused the absorbed *decode* kernel per row (causal via per-row
  `context_len`), which was O(L²) latent reads per chunk — that was 95% of
  the 2048-token TTFT.
- **The conv is batched bitwise.** Window rows are prebuilt from the landed
  bf16 inputs themselves (`window[t][j] = x[t-3+j]`, carry from the previous
  chunk's state), so every window value equals what sequential stepping would
  have shifted through the slabs; one `k3_conv_silu` launch covers the chunk.
  This needed one new AOT land config (`(KDA_DIM, KDA_DIM, 0)`) — the
  sequential engine casts conv inputs inside the conv kernel, the chunk needs
  them landed *before* the window build.
- **The KDA delta rule is one chunkwise FlashKDA forward per layer**
  (`third_party/flash-kda` — MoonshotAI's own CUTLASS/CuTe chunkwise KDA
  kernel, MIT, vendored; see its PROVENANCE.md). Two launches replace the
  69-layers × tokens b=1 walk the first cut of this phase shipped. The C ABI
  shim (`csrc/k3/k3_flash_kda.cu`) pins D=128 / f32 state / one sequence per
  call; the gate math is the fused core's own formula applied in-kernel from
  the same pre-activation bf16 landing, beta sigmoid in-kernel, and the f32
  recurrent slab plugs in directly ([H, V, K] both sides). What is *not* in
  FlashKDA — the per-head o_norm × sigmoid output gate — is the new
  `k3_o_norm_gate` TileLang family, word-for-word the fused core's tail.
  Parity becomes a per-chunk double buffer (read one slab, land in the
  other) instead of a per-token flip. KDA state stays one row per pool
  (~929 MB/slot forbids chunk-wide state). Decode is untouched: it keeps the
  bit-matched fused TileLang core.
- **The prefill pool is asymmetric**: one row of KDA/conv state (the
  recurrence is sequential anyway), a full bucket of attention-residual
  snapshot rows and block-table rows (`attn_rows`). Handoff collapses the last
  live row's snapshots to row 0, then `adopt_row` copies as before. Prefill
  runs eagerly — the KDA walk makes the launch count depend on the token
  count, so there is no fixed body to capture per bucket.

Equivalence: the conv is bitwise against sequential stepping; FlashKDA
computes the same delta rule but with an f32 q/k l2norm chain (the TileLang
core mirrors the reference's bf16 chain) and chunkwise accumulation order,
and the GEMM bucket retiles — so chunked prefill is held to the fixture's
noise floor, and the state a slot adopts can legitimately send a greedy
continuation off the bit-exact baseline at a ≤2-ULP coin-flip step (observed:
one flip at a 1-ULP step). The two prefill gates therefore force-feed the
fixture feed and hold every step to the noise-floor excusal:
`prefill_then_decode…` (cap 1 — the degenerate single-token chunks) and
`chunked_prefill_crosses_its_bucket_boundaries` (cap 8 vs a 13-token prompt:
full chunk + ragged odd chunk + parity handling + padding rows). The EP
oracle's busy peers prefill through chunks.

TTFT snapshot (4-layer truncated, one GB300, `prefill_time_snapshot`),
against the retired per-token prefill; the truncation carries ~3 KDA / 1 MLA
layer where the full model is 69/24:

| prompt | per-token | chunked (cap 128) | + FlashKDA | + FlashMLA | + 4224 cap | total |
|--------|-----------|-------------------|------------|------------|------------|-------|
| 64     | 132 ms    | 21.2 ms           | 18.1 ms    | 27.1 ms    | 23.7 ms    | 5.6x  |
| 512    | 1176 ms   | 161 ms            | 114.5 ms   | 37.3 ms    | 25.8 ms    | 46x   |
| 2048   | 6377 ms   | 1508 ms           | 1312 ms    | 61.1 ms    | 25.8 ms    | 247x  |

The FlashMLA step killed the decode-shaped attention's O(L²) latent reads
(~1.05 s of the 1.31 s at 2048); widening the cap then collapsed 2048 tokens
from 16 chunks to one, paying the per-chunk fixed cost (per-layer FlashKDA /
gather / kv_b GEMM / FMHA setup, plus the step's own overhead) once — 512
and 2048 now cost the same wall clock. Chunk width changes GEMM bucket
retiling and FlashKDA call boundaries, so different caps agree to the noise
floor, not bitwise (verified: cap 64 vs 128 — both on the pre-existing
ladder — diverge the same way a wide cap does; each cap is individually
deterministic).

### MLA KV: paged latent cache + absorbed decode

The MLA cache holds the **latent**, not the expanded heads: one 576-wide bf16
row per token per MLA layer — the post-norm kv latent (512, `kv_lora_rank`)
next to the raw shared rope half (64; K3 is NoPE, nothing is ever rotated).
That is 27.6 KB/token against the 1.47 MB/token the expanded `[96×192 K |
96×128 V]` slot cache used to pin, and it is what lifted the fixed
`max_ctx = 128`.

- **Layout**: one slab per rank, `[page][layer][token][576]`, 64 tokens per
  page, every MLA layer's slice inside the same page (layer offset =
  `mla_index × 64 × 576` elements). Pages come from a plain free list —
  claimed when a slot's position crosses a 64 boundary, freed together when
  the slot retires; no content addressing, no reuse (kv-store integration is
  a later milestone). Per-slot block tables live host-side and ride to the
  device with the step feed, outside graph capture; the captured kernels read
  the device table by pointer. `K3ExecutorConfig::kv_pages` sizes the pool
  (`0` = full coverage, `max_batch × ceil(max_ctx/64)`).
- **Write path**: the step computes `kv_norm` and `rope` exactly as before
  and lands them into the mapped page row (`kv_row` is the device-fed
  destination index; `-1` rows are skipped). A page is zeroed when claimed
  and every position written once, so the indexed add is an exact indexed
  copy.
- **Decode is absorbed MLA** (`csrc/k3/k3_mla_paged_attn.cu`, hand CUDA —
  TileLang 0.1.12 cannot express the runtime-length page walk without a
  compile-time capacity, which is the thing being removed): per head,
  `score(t) = (W_UKᵀ q_nope)ᵀ c_t + q_ropeᵀ k_rope_t` and
  `o = W_UV (Σ p_t c_t)`, with `W_UK`/`W_UV` read straight out of the
  checkpoint's `w_kv_b` — no expansion GEMM, no expanded cache. One
  (row, head) block; softmax is a 3-sweep recompute (max / sum /
  probs+attend), so there is no O(ctx) storage and **no compile-time context
  cap**. The certified rounding chain is preserved landing for landing and
  documented step-by-step in the kernel header: q absorption lands bf16 (f32
  dot per latent column); f32 score dot over the 576 columns ascending; the
  dot lands bf16 and multiplies the bf16 scale in bf16; max and Σexp in f32
  fixed order; probabilities land bf16 after normalizing; the latent
  accumulation is f32 per column over ascending t, landing bf16; the W_UV
  expansion is an f32 dot landing bf16.
- **Physical page ids never enter the arithmetic** — the kernel walks the
  block table by logical position — so any page permutation is bit-identical,
  and an unmapped (`-1`) entry reads as zero latent, which is exactly the
  zeroed padding row of the old cache.

Gates (`pegainfer-k3/tests/paged_kv.rs`, plus the golden suite).
`scripts/k3_gates.sh` runs the whole checkpoint-backed battery on a tray with
the invocations below baked in (idle-tray check, prebuilt binaries, per-gate
logs) — the commands here remain the reference for running one gate by hand.
Either way the gates load weights through the pinned staged uploader by
default (`PEGAINFER_K3_WEIGHT_STAGING=0` restores serial pageable mmap; same
bytes, the whole battery drops to ~3 minutes on a warm page cache):

```bash
# Paged gates: page-permutation bitwise, long-context (2048-ctx, 1100 steps,
# rerun + graphs-vs-eager bitwise), on a GPU box with the checkpoint:
PEGAINFER_K3_TEST_224=<checkpoint> cargo test --release -p pegainfer-k3 \
  --test paged_kv -- --ignored --test-threads 1

# Absorbed-vs-expanded certification: dump per-step logits on the expanded
# revision (M1) and on this one, then compare per step in bf16 ULP:
PEGAINFER_K3_TEST_224=<checkpoint> PEGAINFER_K3_LOGIT_DUMP=/tmp/logits.bin \
  cargo test --release -p pegainfer-k3 --test paged_kv \
  dump_forced_replay_logits -- --ignored

# And the existing suites must stay green:
PEGAINFER_K3_TEST_224=<checkpoint> cargo test --release -p pegainfer-k3 \
  --test golden_decode -- --ignored --test-threads 1
```

Certified (4-layer golden inputs, absorbed vs expanded per step): argmax
identical on all 40 steps; deviation over the fixture-published top-5 logits
median 1.0 / max 3.0 bf16 ULP — inside the measured ≤2-ULP structural noise
floor of the FP8 expert path; whole-row max deviation at the top-logit
magnitude median 1.75 / max 4.0 ULP. (The absorbed kernel associates the
score dot as `(W_UKᵀ q)ᵀ c` rather than `qᵀ (W_UK c)`, so bit equality is not
expected — the floor is.) The golden fixture holds at 39/40 (fused) and 38/40
(masked chain) with misses only on the documented coin-flip steps; page
permutation (200 steps) and the long-context reruns and graphs-vs-eager
(1100 steps, ctx 2048) are bit-identical; the EP4 oracle is bitwise green.

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
  step rather than skipping it — and a prefill chunk step issues the same
  per-layer launch sequence as a decode step (the chunk's per-token KDA walk is
  rank-local), so a rank's prefill step pairs against a peer's decode step with
  no negotiation. What chunking *does* change is the step count a prompt
  spends: `ceil(len/cap)` steps instead of one per token — a peer that finishes
  earlier just pads, as ever.
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
- **One flat symmetric slab** holds twelve
  differently-typed regions: the FP8 activation and its packed scales, the
  routing pair, and the L1/L2 ring buffers. Its size and offsets are pure host
  arithmetic over the shapes, the candidate `BLOCK_M` set and the SM count —
  and they are kernel *template parameters*, so a rounded-up allocation is
  wrong, not merely wasteful. The slab is sized at the protocol maximum
  (`k3_mega_max_tokens_per_rank` = 4224 rows, the chunked-prefill ceiling),
  whatever the executor's live batch is. At `ep_size 1` the slab is a plain
  device allocation (940 MiB, ring 48000 tokens): the kernel's cross-rank
  barriers compile down to grid-local synchronisation, so no IPC or NVSHMEM
  handle is involved. At `ep_size 4` each rank owns one (1633 MiB, ring
  119424 tokens) on its own device and the world exchanges bare base pointers.
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
  the protocol maximum (`num_max_tokens_per_rank = 4224`) rather than the live
  token count: BLOCK_M 192 / BLOCK_K 128 (the same entry the old 384 maximum
  selected, so raising the ceiling for chunked prefill left EP4 decode tiles
  untouched — confirmed by an A/B step-time run within noise at every bucket). Two reasons. Nothing in the kernel
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

## The attnres stride bug: what "all gates green" did not cover

Found 2026-08-18, when chunked prefill made full-depth serve emit garbage on
every prompt while all 13 golden gates and both EP oracles stayed green. The
batched attention-residual kernels declared the snapshot slab as `(B, NB, H)`
— row stride `NB x H`, `NB` being the layer's candidate count — but the slab
is `[rows, block_count, H]` with `block_count = ceil(layers/12)`. Row 0 is
stride-independent, so every `b = 1` path was correct, and that turned out to
be every path any gate or any prior serve had ever run: the 4-layer fixture
sits at `block_count = 1 = NB` (strides agree by accident), EP4 serve
partitions decode one request each (`b = 1`), and pre-chunked prefill walked
prompts one token at a time. A prefill chunk is the first shipped shape with
`b > 1` past 12 layers, and rows ≥ 1 read other rows' snapshots on every mix.
Concurrent decode rows in one partition had the same latent corruption —
never shipped, never caught.

The fix pins the slab row stride at `K3_ATTNRES_MAX_BLOCKS` everywhere: the
kernels compile `Bl` as `(B, BC=8, H)` and `K3StatePool` always allocates and
copies at that stride.

Two lessons with teeth:

- A slab shared by kernels must have exactly one stride authority. The
  kernel's tensor declaration *is* an indexing contract; if the host lays the
  slab out with a different constant, only row 0 tells no tales.
- The debugging shape that found it is now a gate:
  `chunked_prefill_agrees_with_the_per_token_walk_at_depth` holds chunked
  prefill to the per-token decode walk at **any** depth
  (`PEGAINFER_K3_AB_LAYERS`, `PEGAINFER_K3_AB_CHUNK`) with no fixture in the
  loop — boundary-logit max |Δ| plus a forced-fed continuation. The bug's
  signature was Δ step-jumping 1.1 → 10.5 between 12 and 13 layers
  (`block_count` 1 → 2) while a cap-1 chunk stayed at Δ ≈ 1 at every depth.
  Text-based A/B at truncated depth is useless for this: a truncated model's
  logits are near-uniform, so any two correct implementations diverge within
  a few tokens anyway.

## Next

Chunked prefill has landed end-to-end at the protocol width: the chunkwise
KDA kernel (vendored FlashKDA), the dense MLA prefill attention (FlashMLA
SM100 FMHA), and the 4224-token chunk cap with the one-row boundary sample —
see Chunked prefill above. What remains on the prefill axis is a W-chunked
context loop (+ LSE merge) if `max_ctx` outgrows the fixed expansion
workspace, and trimming the wide step's f32 partial scratch if the ~7 GB
ever bites. Then graphs over the EP4
fused path (the ranks=1 path already captures), launch-ahead, kv-store
`BlockPool` integration (content addressing / reuse for the MLA pages), and a
perf pass on the paged attention kernel (the 3-sweep recompute reads the cache
three times; fine at bring-up depth, worth a fused pass at 24 MLA layers x
long contexts).
