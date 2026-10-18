# GLM5.2 EP8 DeepEP MoE + full-model forward (PR4)

> **TL;DR:** **BUILT + ALL GATES GREEN on jz-38 8×H200 (2026-07-03), branch `feat/glm52-ep8-deepep-moe` — the campaign's first full-model generation.** PR4 turns the oracle-gated single-layer bricks (PR1–PR3) into a running 8×H200 model: a GLM-baked instantiation of the DeepEP v2 elastic shim replaces PR3's local scatter/combine stand-ins, the weight loader places expert tensors into their **final packed layout at H2D time** (post-load repacking cannot fit: 2×85.5 GiB > 141 GiB), `from_device` constructors hand the resident buffers to the PR3 weight structs zero-copy, and a DP1 executor walks all 78 layers on rank 0 while ranks 1–7 run their 32 local experts through the collective dispatch/combine. Gate results: EP8 layer-6 oracle **62/64 with the outlier probes bit-identical to EP1** (the collective chain computes exactly the local chain's output); all PR1–PR3 gates re-run green; e2e greedy generation is fluent, factually correct, and byte-deterministic across runs (~109 ms/step bs=1 bring-up, unmeasured levers listed below). See Execution log.
>
> **Last touched:** 2026-07

## Corrections / decisions vs the plan doc's PR4 section

1. **Dispatch payload stays bf16; the fp8-payload question is answered "not now", not by a new measurement.** The shim's wire format is bf16-only today. At DP1 bs=1 decode the all-to-all is latency-dominated (kimi measured combine 59–88 µs against a 37 µs bandwidth theory at bs=64 — bandwidth is not the binding constraint even there), so halving payload bytes buys nothing measurable at this batch. Extending the shim with fp8+scale packs (`kNumSFPacks`) is real work with zero expected win until a measured bandwidth problem exists at a real serving batch — PR5's TPOT measurement is the trigger. Consequence: every rank re-quants its received rows to fp8 before the grouped GEMMs.
2. **Expert-GEMM path at EP8 is Grouped-only.** The dispatch's expert-major aligned recv layout *is* the grouped GEMM layout (that was the whole point of PR3's contract choice). The GEMV path stays EP1-only for now: at EP8 it would need a per-row expert-id map built from `psum_expert` — a new kernel with no correctness value. Wire it up when PR5's Grouped-vs-GEMV A/B asks for it.
3. **No CUDA-graph capture in PR4** (that is PR5), but the decode step stays host-quiet: dispatch/combine are the kimi-proven `do_cpu_sync=false` kernels, re-quant/SiLU launch at fixed worst-case capacity with row isolation, and nothing branches on device data.

## DeepEP shim: GLM instantiation

`csrc/deepep/deepep_shim.cu` is one TU with the Kimi config baked as `deepep_shim::cfg` constants and fixed `deepep_*` C symbols. GLM needs different constants **and** different symbols (both models can be linked into one binary):

- Split the shim body into `deepep_shim_impl.cuh`, parameterized by two macros: `DEEPEP_SHIM_CFG` (config namespace) and `DEEPEP_SHIM_FN(name)` (symbol prefix), plus a per-instance opaque ctx tag (`DeepEpCtx` vs `Glm52DeepEpCtx` — distinct tags, no ODR games).
- `deepep_shim.cu` becomes the Kimi instantiation (symbols unchanged — kimi code untouched).
- New `deepep_shim_glm52.cu` + `deepep_config_glm52.cuh` (compiled only under the `glm52` feature): ranks 8, experts 256, local 32, topk 8, hidden 6144 (bf16 → 12288 B), **expert_alignment 64** (= `GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT`, the TRTLLM M-tile — segments land pre-aligned for the grouped GEMM), decode max 128 tokens/rank, SM/smem/QP/timeout identical to kimi (same H200 node class). Worst-case expanded decode capacity: `align(8·128·8 + 63·32, 64) = 10240` rows.
- Rust: `ffi::glm52_deepep_*` decls + `ops/deepep.rs` generalized over a small ABI table so `DeepEp` (kimi, unchanged API) and `Glm52DeepEp` share the wrapper body.

## Weight residency: packed placement at load time

The #476 loader streams checkpoint tensors into one raw slab in shard order. PR3's `Glm52MoeLayerWeights` needs experts **packed expert-major** (`w13 = [expert][gate;up]`). Repacking after load is impossible on ranks 1–7: the slab is 85.5 GiB of experts and the packed copy is the same size — 171 GiB > 141 GiB HBM. So the loader learns placement:

- A pure layout function maps every tensor name → (region, offset). Expert tensors go into four per-layer packed regions (`w13_weight`, `w13_scale`, `w2_weight`, `w2_scale`, experts in local order, gate;up concatenated per expert — byte-identical to `from_host`'s packing, which stays as the oracle-side reference of the same layout). Non-expert tensors (rank 0) each get their own region.
- Regions are **individually owned** `CudaSlice` allocations (the single-slab alloc dissolves — a few thousand allocs replace one, irrelevant at load time). H2D copies go per-tensor into region sub-ranges; the shard-order mmap lifetime machinery stays.
- `from_device` constructors then *move* the owned regions into the PR3 structs (`ProjWeight`, `Glm52MoeExpertBank`, …) — zero copies, no duplicate residency. The one exception: MLA absorb factors `w_uk`/`w_uv` are derived (fp8 kv_b → bf16 dequant); they take a one-time D2H→host-dequant→H2D round trip (~29 MB/layer, 2.3 GiB total bf16 result), reusing PR3's host dequant so oracle and production share one code path.

## EP8 MoE decode chain (per rank, per MoE layer)

```
rank 0                                          ranks 1..7
router (256-wide, bs tokens)                    —
deepep decode_dispatch(normed bf16, topk)       decode_dispatch(dummy, 0 tokens)
        └→ recv_x bf16 [10240,6144], recv_topk_weights, psum_expert[33]
metadata kernel: psum i32 → expert_offsets i64 + masked_m + row_map (32 groups)
fp8 per-token-group re-quant → masked [32,64,k] slabs + mn-major TMA scales
DeepGEMM masked grouped FP8 W13 → weighted SiLU·quant (weights = recv_topk_weights)
DeepGEMM masked grouped FP8 W2 → remap masked→aligned slots
deepep decode_combine → combined[bs,6144]       decode_combine (0 tokens)
rank 0: + shared expert + residual              —
```

(2026-07-05: the TRTLLM grouped GEMMs + activation-scale relayout were
replaced by the DeepGEMM MGroupedMasked AOT instantiation — 1.5-1.9× on the
same data, c64 1113 → 1475 tok/s e2e; the masked-layout bridge and A/B record
live in whole-step-decode-graph.md. The aligned-segment recv layout and the
row-isolation invariant below are unchanged.)

PR3's `Glm52MoeLayerWeights` splits into `Glm52MoeRouterWeights` + shared-expert projections + `Glm52MoeExpertBank` (n_experts is 256 at EP1, 32 per rank at EP8); the EP1 gates recompose the same pieces, unchanged math.

Pad/garbage rows: recv rows beyond the real count hold stale bytes; re-quant of garbage (even NaN) is row-isolated, the GEMMs are row-independent, and combine only reads slots addressed by `src_metadata` — the PR3 row-isolation invariant carries over verbatim. The re-quant/SiLU cost is capacity-proportional (10240 rows regardless of bs) — a known PR5 measurement item, not a correctness issue.

## Executor (DP1/EP8, bs=1, no scheduler)

- Rank 0 owns everything non-expert: embed → 78 × (MLA + DSA indexer with cross-layer top-k carry + dense/MoE) → final norm → lm_head → greedy argmax (host-side for bring-up; device sampling is PR5).
- Ranks 1–7 run 75 collective pairs per token step (MoE layers 3..77), driven by a per-step command from the coordinator; every rank acks per step so errors surface immediately instead of via the DeepEP 100 s device timeout.
- DeepEP contexts are created collectively after weight load (rank 0 generates the NCCL unique id, workers call `ctx_create` concurrently).
- A minimal serial bs=1 coordinator replaces the rejecting one: prefill rides decode token-by-token (position i per prompt token), then greedy decode until eos/max_tokens, emitting real `TokenEvent`s. Batching, streaming scheduler, CUDA graphs: PR5.

## Gates (jz-38, 8×H200)

1. **EP8 layer-6 MoE oracle** — the PR3 layer gate re-run with the MoE half going through the real 8-GPU dispatch/combine (from_host weights split per rank). Same probe constants, same tolerance, same ≤4/64 tie-flip allowance; expected 62/64 with the *same* outlier positions as EP1 (the deviation is upstream of the expert path).
2. **Packed-placement pins** — proven piecewise instead of a GPU digest (which would need a full ~85 GiB rank load per run): a pure layout-parity unit test walks every expert tensor and asserts the placement offsets reproduce `from_host`'s packing gap-free, and the loader's per-region coverage counters fail the load if any packed byte is left unwritten. A placement bug that slipped both would scramble expert weights and fail gate 3 catastrophically.
3. **Full-model e2e generation** — first time the whole model runs: greedy decode on a short English prompt; assert determinism (two runs, identical token ids), finite logits, and eyeball fluency (the PP8 branch's bar). Full-model teacher-forced HF comparison is infeasible (~700 GiB bf16 reference) — per-layer oracles carry the numeric burden.

## Not in PR4

- Scheduler, batching, CUDA-graph capture, device-side sampling — PR5.
- fp8 dispatch payload, GEMV-at-EP8, re-quant cost reduction — PR5 measurement items.
- MTP layer 78 — out of campaign scope (weights resident, unused).

## Execution log (2026-07-03, jz-38 8×H200)

- **Gate results:**
  - All PR1–PR3 gates re-run green after the refactors (MLA 64/64, dense 64/64, EP1 MoE 62/64 both paths, bookend exact, indexer smoke).
  - **EP8 layer-6 oracle: 62/64, and the two outlier probes' engine values are bit-identical to the EP1 Grouped/Gemv runs** (0.021606 / 0.019165) — dispatch → re-quant → metadata → grouped GEMMs → combine reproduces the local chain exactly; the outliers stay the PR3-measured router near-ties.
  - **e2e generation** (pegainfer-server, `/v1/completions`, greedy): "The capital of France is" → " Paris. Distance from Paris to Lyon is 391 km, …" — two runs byte-identical; "Why is the sky blue?" → 96 tokens of correct Rayleigh-scattering prose; 中文 prompt answers 北京. Startup: 700 GiB across 8 ranks loads in ~122 s (placement coverage checks green on the real checkpoint), `from_device` build + DeepEP contexts ~25 s. Decode ≈ 109 ms/step at bs=1 (29 steps in 3.17 s) — bring-up number; the known levers (worst-case-capacity re-quant/SiLU rows, host argmax D2H, per-step rope H2D) are PR5 measurement items.
- **Fixed during bring-up:**
  1. *Collective teardown deadlock*: the DeepEP context drop barriers across ranks; join-then-drop (gate) / sequential worker shutdown (coordinator Drop order) left early ranks spinning in the destroy barrier until the ~100 s device timeout trapped every rank ("unspecified launch failure" ×3, EP8 gate 124 s → 20 s). Fix: rank 0 drops before joining; the coordinator broadcasts Shutdown to all ranks before any join.
  2. *Thread-local cuBLAS handle*: the kernels crate's handle is thread-local per device and initialized by `DeviceContext::new` — which the oracle gates use, masking that the rank workers (own `CudaContext`) never initialized one. First server request died with `CUBLAS_STATUS_NOT_INITIALIZED` in the layer-0 absorb GEMM. `device_context()` now does the idempotent per-thread setup.
  3. *Gate env*: the DeepGEMM JIT needs `CUDA_HOME` in addition to `PEGAINFER_DEEPGEMM_ROOT` — without it the MQA metadata launch fails as an opaque `CUDA_ERROR_INVALID_VALUE`.
- **Toxic-review round (Request Changes → fixed, gates re-run green):** the numeric chain / buffer sizing / loader placement / shim parameterization came back clean; both blockers were in the error paths of the collective protocol. (1) A failed step leaves the EP8 group permanently out of lockstep — the survivors' pending dispatch would pair with the NEXT request's first dispatch and every later layer would run against the wrong expert bank, producing deterministic garbage with no crash; the coordinator now treats any step error as fatal and tears the engine down. (2) BuildModel bundled the fallible per-rank build with the collective context creation, so one rank's build failure (rank-0 OOM being the likely one) would strand the other seven in `ncclCommInitRank` with no timeout; split into build (all ranks ack) → SetupComm (collective). Minors: EOS is now counted but suppressed from emission (the qwen3-asserted contract); the GLM shim's dead prefill path no longer reserves ~200 MB/rank of symmetric memory; rope/geometry constants deduped (model.rs owns them, gates import).
- **Chat template fix (follow-up, landed via lockfile bump):** the GLM5.2 checkpoint's `chat_template.jinja` uses `m.content.0.type` — minijinja ≤2.19 lexed `.0` as a float instead of a dotted integer index, so the vllm frontend failed startup while building the chat backend. Fixed by bumping `minijinja` 2.19 → 2.21 in `Cargo.lock` (2.20 changelog: "Fixed dotted integer lookup in the middle of attribute chains (e.g. `foo.0.bar`)"). e2e previously ran with a shadow model dir (symlinks to `/data/models/GLM-5.2-FP8` + a trivial chat template); `/v1/completions` never renders the template.

## Read

- `docs/models/glm52/ep1-forward.md` — PR3 record; the MoE contract this PR substitutes into.
- `pegainfer-kernels/csrc/deepep/{deepep_shim.cu,deepep_config.cuh,deepep.h}` — the shim being parameterized.
- `pegainfer-kimi-k2/src/runner/moe_deepep.rs` — the graph-quiet DeepEP decode shape (state arena, collective discipline).
- `pegainfer-glm52/src/weights/load.rs` — the loader being taught placement.
- `vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py` — upstream decode contract reference.
