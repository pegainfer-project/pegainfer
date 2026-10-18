# GLM5.2 TP4 on GB300

> **TL;DR:** GLM-5.2-FP8 TP4 on 4xGB300 via FlashInfer sparse MLA, compact decode buckets, SM-selected GEMV/MoE launches, and a vocabulary-sharded greedy tail. The 2026-07-10 matched pair beat vLLM pure TP4 at both fully warmed bs=1 shapes (`9.03` vs `9.21ms` at 1/256, `9.56` vs `9.60ms` at 1024/256); a 2026-07-11 re-validation on the same host reads higher on the PegaInfer path only — host-state drift, code exonerated by same-day A/B (see "2026-07-11 re-validation"). The short-context top-256 attention tier was then removed as non-production weight (it bought `~0.18ms/step` only at the synthetic 1/256 shape). A 2026-07-11 node-count campaign then fused the MLA front, the rows=1 attention AR, and the MoE gate|up+SiLU — all bit-exact — cutting the bucket-1 graph `2,334 -> 1,867` kernels and same-day TPOT `9.85 -> 9.38ms` / `10.20 -> 9.74ms` (see "Graph node-count reduction"). Functional gates are all green. A controlled vLLM EP4 run is slower (`9.47/9.84ms`), so an attention-TP4 + expert-EP4 *hybrid* is out of scope for this low-latency topology — the separate DP4/EP4 *throughput* topology landed later as `--moe-topo ep4` (see `ep4-gb300.md`).
>
> **Last touched:** 2026-07

## Scope

TP4 is a low-latency topology for the four-GB300 target. It is not an EP8 compatibility mode.

| Topology | Devices | TP / DP / EP | Expert placement | Attention heads/rank |
| --- | ---: | --- | --- | ---: |
| EP8 | 8 | 1 / 8 / 8 | 32 whole routed experts/rank | 64 |
| TP8 | 8 | 1 / 8 / 8 | 1/8 slice of all routed + shared experts | 8 |
| TP4 | 4 | 4 / 1 / 1 | 1/4 slice of all routed + shared experts | 16 |

TP4 launch requires `--tp-size 4 --moe-topo tp4`; omitted DP size resolves to one. Prefix-cache and DSpark behavior follows the existing tensor-replicated path. KV offload remains EP8-only because tensor-replicated ranks mirror the cache.

## Design

### Tensor-parallel runtime

- Every TP rank runs the same eight-row logical bucket with replicated activations and routing.
- Attention weights are head-sharded. MoE gate/up and down weights are sliced along the 2048-wide expert intermediate dimension.
- The shared expert is folded into bank slot 256, so one phase chain handles routed and shared outputs.
- The MoE chain is union, gate/up GEMM, SiLU, down GEMM, LL push, and fixed-order LL receive/reduce.
- Attention `o_proj` partials use a two-shot LL reduce-scatter/broadcast chain. Every rank receives a bit-identical result before redundant downstream routing.
- One device-side epoch tags parity-double-buffered 16-byte LL packets. Spins only wait on packets produced by a previous kernel node.
- VMM buffers use one accessor-specific VA per GPU and reject links without native P2P atomics. Broad peer grants measurably tax the memory-bound GEMMs.

The implementation is shared rather than copied:

- `pegainfer-kernels/csrc/glm52/glm52_moe_tp_impl.cuh` contains the MoE kernels and VMM protocol; TP4/TP8 `.cu` files only instantiate rank/slice/grid parameters and ABI names.
- `pegainfer-kernels/csrc/glm52/glm52_tp_ar_impl.cuh` and `glm52_tp_ll.cuh` contain the common attention collective and packet primitives.
- `pegainfer-kernels/src/ops/glm52/moe_tp.rs` and `tp_ar.rs` own topology-dependent shape validation and FFI dispatch.
- `pegainfer-glm52/src/moe_tp.rs` owns one topology-parameterized model runtime state and slice loader.

### MLA backend and cache contract

Kernel selection happens once at model build from the actual device capability and per-rank head shape. There is no `PEGAINFER_GLM52_MLA_BACKEND` override.

| Runtime shape | Backend | KV token layout |
| --- | --- | ---: |
| SM100/SM103 and 16 heads/rank | FlashInfer TRTLLM-generation sparse MLA | 576-byte standard E4M3 |
| Other attention-TP shards up to 16 heads/rank | Right-sized sparse MLA | 656-byte `fp8_ds_mla` |
| Full 64-head fallback | FlashMLA sparse decode | 656-byte `fp8_ds_mla` |
| Neither contract supported | Startup error | n/a |

The kernel and cache format are one immutable startup contract. Allocation stride, cache packing, query assembly, offload namespace, schedule state, and attention launch all derive from it. Backend-specific scratch and schedule metadata are enums; invalid `Option` combinations and dummy allocations are not representable.

FlashInfer's header-only runner needs seven checked-in SM100-family cubins for the `{1,2,4,8} x {256,2048}` selector closure. Two are selector seeds and may not appear as final Nsight symbols, but removing them makes the upstream selector reject otherwise-supported shapes. Provenance and hashes are in `pegainfer-kernels/cubin/glm52/README.md` and `trtllm_gen/flashInferMetaInfo.h`.

### Blackwell-specific paths

- FlashMLA fallback code is assembled as `sm_100f`; plain `sm_103` cannot encode its CTA-group/tensor-core instructions.
- DeepGEMM MQA has an SM100f instantiation rather than falling into the Hopper-only stub.
- FlashInfer uses standard E4M3 query/KV with static-token sparse indices and a 16MiB persistent workspace per bucket.
- TP4 MoE keeps the same math and fixed reduction order as TP8, but grid sizing is architecture-specific. Blackwell caps GEMM B at 2 CTA/SM and GEMM C at 3 CTA/SM; Hopper retains its occupancy-derived grid.

### Vocabulary-parallel greedy tail

TP4/TP8 decode copies one contiguous `lm_head` vocabulary shard per rank at model build (`38,720` rows for TP4, `19,360` for TP8). The full head remains resident for DSpark and non-greedy sampling.

Each rank computes compact shard logits and a local top-1. The candidate's bf16 value plus three exact bf16 token-id bytes are packed into rank-unique positions of a hidden-width scratch row, gathered through one reserved slot of the existing fixed-order attention TP all-reduce, and selected on-device with the same lowest-global-index tie break. This preserves launch-ahead: every rank has the same global next token before the next graph replay is enqueued. No host merge, new communication protocol, runtime environment variable, or full-logit exchange is added. Sampling steps recompute the full head eagerly outside the graph only when a non-greedy row exists.

## Performance

All serving rows are fully warmed, untraced, bs=1, concurrency 1, fixed input/output lengths, ignore EOS, temperature zero. TTFT is intentionally excluded from the decision.

| Shape | Original TP4 FlashMLA | FlashInfer MLA | + MoE grid | + compact/fused | + vocab shard | vLLM TP4 | Advantage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/256 | `12.30ms` | `10.83ms` | `10.68ms` | `9.28ms` | **`9.03ms`** | `9.21ms` | `0.18ms` (`2.0%`) |
| 1024/256 | `12.45ms` | `11.17ms` | `11.01ms` | `9.78ms` | **`9.56ms`** | `9.60ms` | `0.04ms` (`0.4%`) |

The MLA switch recovered `1.47ms` at 1/256 and `1.28ms` at 1024/256. The MoE grid tune recovered another `0.15-0.16ms/token`. Compact TP4 graph buckets, SM103 one-row dense GEMV launches, the paired `q_a+kv_a` projection, and removal of the compact MoE output bridge recovered the next `1.23-1.40ms/token`. Vocabulary sharding recovered the final `0.22-0.25ms/token`.

The compact/fused and vocab-shard 1/256 columns included a short-context top-256 attention tier that was later removed (see "Short-context tier removal" under the re-validation section); the 1024/256 columns never exercised it.

### EP4 topology check

The original comparison and PegaInfer implementation both use pure TP4 for MoE. To test whether expert parallelism was the missing route, vLLM was relaunched with the same checkpoint, TP4 attention, standard FP8 KV cache, max model length 4096, and `--enable-expert-parallel`. This places 64 whole experts on each rank and uses the FlashInfer TRTLLM FP8 MoE backend.

| Shape | vLLM pure TP4 | vLLM EP4 | EP4 delta |
| --- | ---: | ---: | ---: |
| 1/256 | `9.21ms` | `9.47ms` | `+2.8%` |
| 1024/256 | `9.60ms` | `9.84ms` | `+2.5%` |

EP4 is not a latency win for the bs=1 target. PegaInfer currently exposes EP8 and tensor-sliced TP4/TP8, not an attention-TP4 plus expert-EP4 hybrid; adding that topology would require 64-expert rank bundles, a four-rank DeepEP path, and composition with the TP4 attention collectives. The controlled vLLM result does not justify that PR expansion.

### Profile attribution

The accepted optimized node trace uses four ranks and exact per-layer instance counts. Totals are all-device kernel duration normalized by tokens and ranks; auxiliary-stream work may overlap.

| Family | Original FlashMLA | FlashInfer FP8 | Delta |
| --- | ---: | ---: | ---: |
| MoE + router + RS | `3.868ms` | `3.840ms` | `-0.028ms` |
| Projection GEMV + reduce | `3.743ms` | `3.765ms` | `+0.022ms` |
| Sparse MLA attention | `2.042ms` | `0.554ms` | **`-1.488ms`** |
| MLA/cache/query/quant glue | `0.734ms` | `0.562ms` | **`-0.172ms`** |
| Norm + fused residual | `0.905ms` | `0.906ms` | `+0.001ms` |
| Attention TP allreduce | `0.693ms` | `0.824ms` | `+0.131ms` |
| MLA absorb W_UK/W_UV | `0.554ms` | `0.545ms` | `-0.009ms` |
| Indexer + lm_head + other | `1.000ms` | `1.001ms` | `+0.002ms` |
| **All-kernel sum** | **`13.540ms`** | **`11.999ms`** | **`-1.541ms`** |

The attention-AR increase is one communication-wait outlier; per-kernel medians did not regress. The traced request measured `11.49ms` versus the untraced `10.83ms`, so the trace is composition evidence rather than the performance result.

The final vocabulary-sharded node trace measures the local LM head at `77.01us` per rank versus about `312us` for the old redundant full-vocabulary head. Pack/unpack cost `1.63/2.46us`; the one extra TP AR triplet averages `8.41us`. The predicted net saving is about `222us/token`, matching serving.

### Graph node-count reduction (2026-07-11)

Three bit-exact fuses drove the bucket-1 whole-step graph from `2,334` to
`1,867` kernels (`-20%`), each validated by a standalone randomized
byte-compare harness, a graph-dump node count, a greedy byte-identity smoke,
and a same-day TPOT A/B (the day's drifted baseline, not the 07-10 headline):

| Fuse | Nodes | 1/256 TPOT | 1024/256 TPOT |
| --- | ---: | ---: | ---: |
| (day baseline, post short-tier removal) | `2,334` | `9.85ms` | `10.20ms` |
| MLA front: ckv_split + kv_a norm + query/cache quant-pack → 1 | `-234` | `9.70ms` | `10.03ms` |
| Attention AR one-shot at rows=1 (push/reduce/recv → 1) | `-158` | `9.49ms` | `9.85ms` |
| MoE gate\|up mma + SiLU epilogue → 1 (bpart roundtrip deleted) | `-75` | **`9.38ms`** | **`9.74ms`** |

Bit-exactness discipline: a fuse must reproduce the exact f32 association of
the chain it replaces (FlashInfer's d=512 norm reduction; the AR's
ascending-src sums; the SiLU's per-slice-then-ks two-level reduce), verified
against the vendored FlashInfer templates or the production kernels in the
harness before integration. Candidates that cannot preserve bits (router on
cuBLAS tensor-op accumulation) are deferred until a deliberate re-baseline.

**Rejected with data — recv + FusedAddRMSNormRound:** the closing add+norm
needs one block per row, which concentrates the recv's `24,576` 16-byte LL
packet loads (4-byte payload — a 4x wire waste) onto a single SM at bucket 1.
Load-issue bound: `24.7us` naively, `13.4us` after coalescing the packet
reads (strided ownership + shared staging), vs `4.9us` for the unfused
recv+norm pair — and a 12-byte-payload wire format would still only reach
~`4.5us`. A push+recv single-kernel fuse was also rejected: its progress
guarantee needs every block of every rank resident simultaneously, a
deadlock risk on shared GPUs. The MoE chain therefore stays at five nodes.

### MoE resource gate and grid tune

The temporary standalone harness directly included the production TP4 `.cu` instantiation and used `UC=9`, one active row, production scratch shapes, and the production kernels. It was used for the measurements below but is intentionally excluded from the production PR.

| NCU metric | GEMM B, grid 456 | GEMM C, grid 608 |
| --- | ---: | ---: |
| Registers/thread | 80 | 56 |
| Theoretical occupancy | 37.50% | 50.00% |
| Achieved occupancy | 30.27% | 35.27% |
| Compute throughput | 33.60% | 20.07% |
| Tensor-pipe active | 9.01% | 6.44% |
| DRAM throughput | 29.62% | 21.31% |
| L1/TEX throughput | 62.83% | 72.89% |

The kernels were not near their compute, Tensor, or DRAM roofs. The occupancy-max grids created excess short-workload scheduling overhead on 152-SM GB300.

Five hot-cache runs used 100 warmups and 1,000 measured launches each:

| Exact kernel | Old grid median | Tuned grid median | Delta |
| --- | ---: | ---: | ---: |
| B | `13.847us` (456) | `13.095us` (304) | `-5.4%` |
| C | `8.912us` (608) | `7.596us` (456) | `-14.8%` |
| **B + C** | **`22.759us`** | **`20.691us`** | **`-2.068us` (`-9.1%`)** |

Across 75 MoE layers, the hot-cache delta predicts `0.155ms/token`, matching the measured serving recovery. Cache-flushed NCU gives an upper bound of `39.33 -> 34.93us` (`-11.2%`).

## Validation

- SM103 release server build passes.
- SM90a and SM103 standalone compilation passes for both TP4/TP8 MoE and attention-AR instantiations.
- FlashInfer sparse MLA numerical smoke passes all eight `batch={1,2,4,8} x topk={256,2048}` combinations.
- Focused GLM5.2 topology/slice tests pass.
- A 64-token greedy HTTP smoke retains the accepted `" Paris. Distance from ..."` prefix.
- A non-greedy temperature/top-p smoke passes through the eager full-head fallback.
- Vocabulary pack/unpack passes the checked-in device smoke (`pegainfer-kernels/tests/glm52_vocab_parallel_smoke.rs`) covering negative logits, cross-rank tie breaking, a global token id above 65,535, and the all-NaN row degrading to token 0; the translation unit also compiles for SM90a.
- Refactoring preserves TP4 B/C register counts (`80/56`) and launch grids. Clean post-refactor and compact/fused serving reruns pass for both target shapes.

### 2026-07-11 re-validation

A full re-run on the same host after the mainline rebase and review-hardening
pass. Every functional gate is green: lib tests (59), server config tests (7,
including the new tp-size/topology rejections), FlashInfer numerical smoke
(uniform + paged-ramp), the vocabulary pack/unpack device smoke, graph
pre-capture (`4 buckets`), greedy byte-determinism ×2 with the
accepted `" Paris. Distance from ..."` prefix, sampled fallback, and 4-way
concurrent greedy identity.

TPOT did not reproduce, and the cause is the host, not the code:

| p50 TPOT, c1 n20 warmed | 1/256 | 1024/256 |
| --- | ---: | ---: |
| PegaInfer record (2026-07-10) | `9.03ms` | `9.56ms` |
| PegaInfer pre-rebase source, rebuilt 07-11 | `9.57ms` | `10.10ms` |
| PegaInfer rebase + review fixes | `9.67ms` | `10.21ms` |
| PegaInfer HEAD (short tier removed) | `9.85ms` | `10.20ms` |
| vLLM record (2026-07-10) | `9.21ms` | `9.60ms` |
| vLLM re-run 07-11 (same launch config) | `9.27ms` | `9.65ms` |

The pre-rebase source (with its own pinned older nightly) rebuilt and
re-measured on 07-11 lands within `0.1ms` of HEAD, so neither the rebase nor
the review fixes cost anything. vLLM reproduces its record within noise. The
PegaInfer path alone pays a flat `+0.54ms/step` at BOTH shapes versus 07-10 —
a constant per-step host effect (clocks pinned at max, no throttle reasons,
no reboot, no MPS). Follow-up below.

### Short-context tier removal

The optimization pass had re-introduced a lossless top-256 attention tier
(dual MLA schedule and whole-step graph per bucket, engaged while every row's
context fit in 256 tokens) that mainline had previously deleted. It is
removed again, for mainline's original reason: production serving is agent
workloads whose contexts pass 256 tokens almost immediately, so the tier only
ever engaged on the synthetic 1/256 benchmark shape. Same-day A/B: removing
it costs `0.18ms/step` at 1/256 (`9.67 -> 9.85ms`) and nothing at 1024/256
(`10.21 -> 10.20ms`). It also halves the pre-captured graphs and drops one
MLA schedule per bucket. In record-day terms the 1/256 headline is therefore
~parity with vLLM (`~9.21` vs `9.21ms`) rather than `2.0%` ahead; the
1024/256 result is unaffected. Greedy byte-determinism, the sampled fallback,
and the `" Paris. Distance from ..."` prefix all still pass post-removal.

## Artifacts

- Serving JSON: `bench_results/glm52-tp4-moe-tune-20260710/moe-grid-in{1,1024}-out256-c1-n20.json`
- NCU reports: `bench_results/glm52-tp4-moe-util-20260710/tp4-gemm-{b,c}.ncu-rep`
- Optimized node trace: `bench_results/glm52-tp4-flashinfer-profile-20260710/pegainfer-flashinfer-node.{nsys-rep,sqlite}`
- Trace summary: `bench_results/glm52-tp4-flashinfer-profile-20260710/pegainfer-flashinfer-node-summary.md`
- Compact/fused serving JSON: `bench_results/glm52-tp4-fused-front-20260710/fused-front-in{1,1024}-out256-c1-n20.json`
- Compact node trace: `bench_results/glm52-tp4-compact-profile-20260710/pegainfer-compact-node.{nsys-rep,sqlite}`
- vLLM EP4 control: `bench_results/glm52-vllm-ep4-20260710/vllm-ep4-in{1,1024}-out256-c1-n20.json`
- Final serving JSON: `bench_results/glm52-tp4-vocab-parallel-20260710/vocab-parallel-in{1,1024}-out256-c1-n20.json`
- Final node trace: `bench_results/glm52-tp4-vocab-profile-20260710/pegainfer-vocab-node.{nsys-rep,sqlite}`
- Final trace summaries: `bench_results/glm52-tp4-vocab-profile-20260710/pegainfer-vocab-node-{summary,tail}.md`
- Post-tier-removal serving JSON: `bench_results/glm52-tp4-notier-20260711/notier-in{1,1024}-out256-c1-n20.json`

## Pitfalls

- TP8 historically hid `ranks == tokens == 8`. TP4 scratch and job counts must use the fixed eight-row bucket where appropriate, not the four-rank count.
- TP4 must not initialize the EP8 DeepEP path. It uses the tensor-replicated LL rendezvous.
- Backend labels do not imply cache-layout equality. vLLM's standard FP8 and the original PegaInfer `fp8_ds_mla` cache are different contracts.
- FlashInfer module selection/loading must complete before CUDA Graph capture.
- Node-trace sums include auxiliary-stream overlap and communication waits. Graph-level traces and untraced TPOT remain the wall-time truth.
- H200 results explain the original launch choices but are not a controlled GB300 baseline.
- A host-side merge of vocabulary candidates is invalid under launch-ahead: the global token must be identical on-device before every rank enqueues the speculative next replay.

## Lessons

- Kernel/backend selection belongs to SM and topology at startup — the
  `PEGAINFER_GLM52_MLA_BACKEND` override was removed in favor of the build-time
  contract above, and mutually exclusive scratch/schedule enums make an
  inconsistent backend pairing unrepresentable.
- Compact graph shapes matter at bs=1: buckets 1/2/4/8 plus the fused
  `q_a+kv_a` front recovered more than the MLA backend swap itself.
- Replicated bookend work must be audited separately from the layer stack in
  tensor-parallel profiles — attention/MoE weights were sharded while the full
  vocabulary head was still replicated on every rank.
- CUDA refactors need resource acceptance gates: the shared-header extraction
  was gated on exact register counts (`80/56` for GEMM B/C), launch grids, and
  serving TPOT, not on compilation alone.
- A fuse that shrinks the grid can starve SMs at bucket 1 even when total
  work is unchanged: the first gemm_b+SiLU cut (16 k-slices per 256-thread
  CTA) regressed TPOT `9.49 -> 10.13ms`; restoring the old ~2-slice serial
  depth per warp with 512-thread CTAs won on both shapes. Node count is not
  the objective function — the same-day TPOT A/B is.
- LL packet consumers must separate load issue from tag check: a check after
  each volatile load makes every packet a dependent memory round trip
  (`+14.6us/layer` measured); batch the loads, then check, then fall back to
  the spin only for misses. And per-element 16-byte packets are hostile to
  any single-block consumer — one SM issues one load per cycle, so 24,576
  packets cost `~13.6us` regardless of bandwidth.
- Simulating N LL ranks on one GPU needs staged launches (device sync between
  chain kernels): one GPU has no multi-device progress guarantee, so a spin
  can starve the peer kernel it waits on and trip the 200M-iteration trap —
  the production topology is immune (`tp_ar_oneshot_harness.cu`).

## Remaining Work

- Add a model-level TP4 golden/logit gate; HTTP prefix parity and focused kernel gates are narrower evidence.
- Re-run the retained TP8 runtime gates on H200 when that host is available; GB300 can prove compilation but not H200 performance.
- Root-cause the 07-11 flat `+0.54ms/step` host drift (see the re-validation
  table): it hits the PegaInfer serving path at both shapes while vLLM is
  unaffected, and it survives a rebuild of the pre-rebase source with its
  original toolchain. Suspect host state (P2P/VMM mapping latency, NUMA
  placement); re-baseline BOTH engines in one session on a quiet host and
  update the TL;DR table.
