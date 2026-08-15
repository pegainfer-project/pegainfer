# docs index

Organized by domain (model line / subsystem / playbook / lesson) instead of by lifecycle stage. A doc's freshness is recorded in its own header (TL;DR, and `Last touched` for active areas), not by which directory it lives in.

| Where it lives | What it is |
| --- | --- |
| `roadmap/` | Strategic plans and milestones — quarterly direction, product positioning. |
| `models/<line>/` | Per-model living docs: design, accuracy, perf, refactor records, gotchas. |
| `subsystems/<area>/` | Cross-cutting components (runtime / scheduler / frontend / kernels). |
| `playbooks/` | Reusable how-to: benching, profiling, accuracy debugging, onboarding. |
| `lessons/` | Tribal knowledge from research / other projects worth keeping. |
| `benchmarks/` | Standalone benchmark snapshots and eval reports. |
| `conventions/` | Ongoing standards (bench regression policy, coding style). |
| `private/` | Local-only notes (gitignored). |

## roadmap

| Path | TL;DR |
| --- | --- |
| `roadmap/roadmap-2026-h2.md` | 2026-H2 plan, supersedes issue #203. Now: website as product surface (recipes-style pages, no runtime Docker image), model maturity tiers (Qwen3-4B first Stable), observability wiring, GLM5.2 large-MoE mainline. Later: NIXL-compatible P/D, design-first. |
| `roadmap/direction.md` | One size can't fit all. Shared infrastructure (frontend, runtime primitives, kernels, data plane) + per-model engines with their own scheduler/kernel DAG/state. Long-term loop: kernel ledger → simulator → request tracing. |
| `roadmap/execution.md` | Current state and immediate next steps. No timeline — entries move through In progress → Next → Open. Covers cross-model infrastructure (kernel ledger, simulator, tracing, frontend polish) and per-model active work (Qwen3.5, Qwen3). |

## models / qwen3

| Path | TL;DR |
| --- | --- |
| `models/qwen3/cuda-graph-png.md` | `--dump-graph-png PATH` exports the live Qwen3 rank-0, batch-1 SplitKv decode graph as an unfolded detailed DOT and a 192-DPI folded PNG; Qwen3-4B yields 507 kernels and 506 edges, with CUDA Driver API 12.3 as the repository floor. |
| `models/qwen3/serving-performance.md` | **Authoritative Qwen3 serving perf numbers** (4B + 8B QPS sweep vs vLLM 0.24.0, footprint, DSpark/DFlash spec decode, warm prefix-cache TTFT, KV offload). |
| `models/qwen3/serving-perf-5090.md` | Tuning history behind the serving numbers: unified-step attention fusion, batched step tail (#345), chunked prefill, cuBLAS 12.9 N=1025 cliff, cublasLt per-shape tuning, split-KV ≤bs32. Latest data lives in `serving-performance.md`. |
| `models/qwen3/decode-attention.md` | Decode attention path (`NonPartition` vs `SplitKv`) is chosen by **batch (CTA-vs-SM), not context**: the old `max_seq_len>=1024` gate stranded bs=1 mid-context decode on the SM-starved NonPartition kernel — a tpot hump peaking ~ctx800, cliff-dropping at ctx1024. Removing it flattens bs=1 tpot (5090 −16% / 5070 Ti −7.5% @ctx800); kept `padded_bs<=32` (bs≤8 wins big, bs16 even, bs32 <1% loss). Also records the SplitKv chunk-size/grid policy (`Tuned` adaptive vs `Pin`/`PerToken` fixed-split batch-invariance, #435/#438). Two-card A/B + CUDA-graph capture + golden-gate verified. |
| `models/qwen3/green-ctx-sm-partition.md` | Green Context SM partition (`--decode-overlap green-ctx --decode-sm-pct 20`) runs prefill/decode on disjoint SMs so decode stops stalling behind co-scheduled prefill: 5090 mid-band ITL p99 ~halved, TPOT down (−22% @QPS12), but TTFT 2–4× worse (prefill deferred + fewer SMs) — a TTFT↔ITL/TPOT trade, not a free win. Two-graph change (decode CUDA graph captured on the green decode stream) adds ~5% ITL p99 / 1–4% TPOT on top. Mechanism, A/B table, Xid-31/gemm_lt pitfalls. |
| `models/qwen3/roadmap.md` | Qwen3-4B roadmap (2026-06 review): line is the maturity bar; #220 RoPE OOB, batched greedy sampling (#307), mixed greedy/non-greedy sampling (#284), and pegaflow KV offload (#316) are landed; open set is zero TP coverage, zero-adapter-only LoRA gate, dropped prefix-cache observability, stale docs, and YaRN #8 follow-up. |
| `models/qwen3/model-crate.md` | `pegainfer-qwen3` owns Qwen3 config/weights/executor/scheduler/tests/kernel plan; root sees generic `EngineHandle`; split-K decode gated on `padded_bs<=32` (64-token `Tuned` floor, cap 64 chunks; `Pin`/`PerToken` fixed 160-token split), 4k/64 serving TPOT p50 `6.46ms` on RTX 5090. |
| `models/qwen3/prefix-cache.md` | Prefix caching on by default for Qwen3-4B: full-block kvbm radix matching at the executor, suffix-only prefill. Repeated ~1900-token prompt TTFT 141.8 → 16.3ms p50 (8.7×); warm TTFT ≈ TPOT + ~5ms setup. Includes the RoPE scalar-path corruption fix and the drain-the-stream TTFT measurement pitfall. |
| `models/qwen3/dspark-integration.md` | DeepSeek **DSpark** Phase 1 is implemented for Qwen3-4B: DFlash backbone + rank-256 Markov head, anchor-first DeepSpec layout, one strided argmax-with-bias kernel, PDL polish, and one D2H per draft block. Greedy losslessness passes; 5090 block7 A/B vs matched DFlash shows DSpark +3.6% geomean output tok/s overall (+3–16% on text/code, random synthetic exception) and better accepted-draft distribution (2.52 vs 2.30 draft tokens/round). |
| `models/qwen3/dflash-speculative-decoding.md` | DFlash speculative decoding behind `--dflash-draft-model-path`, modelled as an optimistic transaction (propose K → verify K+1 span → accept longest argmax prefix + 1 bonus → commit/roll back KV). Lossless up to bf16 tie-flips (bit-identical multi-token accepts; lm-eval gsm8k strict-match identical spec on/off). Single-stream decode 1.82× on 5070 Ti, 1.56× on 5090. Concurrent throughput fixed by batching the draft forward, then a piecewise verify CUDA Graph (dense ops captured, attention eager) closed single-stream: 5090 greedy c1 274 ≈ vLLM 278, c8 1525 > 1240, c16 1834 ≈ 1846 — all batch sizes now ≥ vLLM. Accept measured equal (9.1% vs 8.85%, same drafter); draft-side piecewise graph tracked next. Proposer trait deferred to EAGLE. |
| `models/qwen3/accuracy-gate.md` | Qwen3 size-keyed logits golden gate (all six sizes 0.6B–32B committed) (`tests/hf_golden_gate.rs`): 48 teacher-forced sequences / 816 positions vs a stored HF bf16 golden, replayed over bs=1 / batched eager / CUDA-graph. Strict guards: regret check + mean ≤ 0.06 + p99 ≤ 0.20; absolute max printed but not asserted (coverage-unstable). Methodology in `subsystems/correctness/`. |
| `models/qwen3/kernels-crate.md` | Phase 1 split implemented and 5090-verified: Qwen3-4B kernel surface lives in `pegainfer-kernels`; release build, test-target compile, accuracy gate, and bench snapshot pass. |
| `models/qwen3/tp-design.md` | Qwen3 tensor-parallel design: `TP=2` milestone scope plus the controller/worker broadcast execution model, request identity, and coarse-grained step protocol for future TP/MoE work. |
| `models/qwen3/kv-pressure-hang.md` | Issue #85 Qwen3-4B KV pressure hang fixed by full-lifetime scheduler KV admission, waiting-queue deferral, cleanup on disconnect/error, impossible-request errors, scheduler/bridge gates, and real `vllm bench serve` QPS=2 `500/500` pass with post-pressure completion healthy. |
| `models/qwen3/pd-disaggregation-m2.md` | P/D 分离 M2 **已端到端验证**（单机 2×H200 + 400G IB）：Qwen3-8B 1P+1D，KV 经 pegaflow metaserver P2P（RDMA READ）流转，greedy 输出与单实例逐 token 一致，杀 metaserver/P 优雅退化；多轮并发压测已过（含 router `max_completion_tokens` 坑）。pegainfer `feat/pd-pegaflow-p2p` + pegaflow PR #381。RemoteFetch 状态机单测欠账；M3 layer-wise push 延后。 |

## models / qwen35

| Path | TL;DR |
| --- | --- |
| `models/qwen35/roadmap.md` | Qwen3.5 dense roadmap v2 (#654): core correctness/admission/chunked-prefill/sampling/step-tail gates are landed; current 4B HTTP boundary is the retained #469 RTX 5090 sweep, which completed with zero failed requests but trails vLLM at high concurrency. Next: HTTP gap attribution, mixed-load ITL (#470), lifecycle recovery (#471), joint-state prefix reuse (#257), and design-first TP (#446). |
| `models/qwen35/load-snapshot.md` | Issue #605 publishes Qwen3.5 logical running, waiting, and KV load through the shared single-GPU/TP scheduler backend and `EngineHandle::with_load_watch`. |
| `models/qwen35/prefix-cache.md` | Qwen3.5 prefix-cache design: a hit is valid only when full-attention KV and a complete recurrent/conv snapshot exist at the same 256-token boundary; the first version uses a fixed-budget GPU snapshot pool with joint lookup, pinning, and LRU eviction. |
| `models/qwen35/kv-admission.md` | Issue #254 complete: Qwen3.5 now uses full-lifetime KV admission, deferred pressure handling, impossible-request rejection, explicit error semantics, direct rejection-event coverage, RTX 5090 e2e, and real HTTP pressure/post-pressure validation. |
| `models/qwen35/optimization.md` | Hybrid 24 linear + 8 full attn optimization ledger. Decode-tuning refresh fuses MLP gate/up and tunes decode cublasLt buckets, improving direct TPOT by 2-3%; vLLM still leads 1024/256 HTTP decode. |
| `models/qwen35/accuracy.md` | Qwen3.5 HF bf16 logits goldens, size-keyed (0.8b/2b/4b/9b/27b all committed), through `past_key_values`: short replay covers sequential graph, bucket-straddling batched graph, and slot-compaction; long replay covers 4097/8192-token prompts; full GSM8K 8-shot now matches the HF baseline within 0.15 percentage points. |
| `models/qwen35/model-crate.md` | `pegainfer-qwen35` owns Qwen3.5 model/scheduler/recurrent ops/tests/benches; feature-gated behind `qwen35` (Triton AOT is the only Python build dependency); root loads it through `EngineHandle`. Build/check/clippy, root bench sanity check, historical Qwen3.5 e2e, and scheduler e2e records live here. |
| `models/qwen35/batched-step-tail.md` | Qwen3.5 issue #353 implementation record: final prefill tail is batched, decode/unified sample from batched logits, host full-vocab copies are logprobs-only, HF + scheduler e2e pass, and final serving A/B supports only the first-token/short-output TTFT claim. |
| `models/qwen35/tp-design.md` | Qwen3.5 TP design: Phase 1 is eager dense TP on Qwen3's controller/worker runtime; validate TP2 first, fail closed for indivisible degrees and TP+CUDA Graph, shard dense full-attention/MLP, and leave sharded linear/GDR state to follow-up. |
| `models/qwen35/tp-implementation.md` | Qwen3.5 TP Phase 1 implementation record: eager dense TP2 worker/scheduler path, short/long HF logits gates, scheduler e2e, and real OpenAI-compatible HTTP smoke pass; remaining TP work is kept as follow-up, not a Phase 1 claim. |
| `models/qwen35/mixed-load-itl-470.md` | Issue #470: full cold `--max-batch 8/bg=4` matrix on RTX 4090 (24/24 valid) + starvation negative control. Qwen3.5 is not immune; chunking bounds max/per-step stall but raises p99 at low QPS (~14→~80–92ms) and pulls p99/max back from the prefill wall to the chunk wall at high load; `qps·prefill_s≳1` is a throughput wall (chunking can't fix it, and ON's +15% TTFT can trip it earlier). The old "p99 immunity" was a slot-starvation artifact. |
| `models/qwen35/adaptive-scheduler-policy.md` | Issue #727 adaptive scheduler policy record: default `off`, opt-in `auto`, hard `--max-prefill-tokens` cap, TP `auto` rejection, and pre-review whole-prefill benchmark tradeoff retained as non-default evidence. |

## models / gemma4

| Path | TL;DR |
| --- | --- |
| `models/gemma4/tokenizer.md` | Gemma 4 tokenizer/chat-template contracts, gated against a Hugging Face reference (token ids plus all five chat renders, content flattened to strings): BOS comes only from the standalone `chat_template.jinja` (which opens a thought channel and accepts a native system role), EOS is declared three times with three values, the published defaults are sampled rather than greedy, image/audio tokens encode straight from user text so text-only serving must reject them at admission, and one divergence stays open — the server's default content format adds a trailing space to system turns. |
| `models/gemma4/hf-golden.md` | The Hugging Face reference for 12B: layer-boundary activations at both ends of both layer types plus top-64 logprobs, over a single-token, a nine-token and a 1024-token (exactly the sliding window) case. Pins three facts the forward path has to match — the embedding scale is bf16 62.0 rather than `sqrt(3840)`, text attention is causal, and `layer_scalar` applies to the layer output after both residual adds. Regeneration is byte-identical and checked with sha256. |
| `models/gemma4/kv-cache.md` | The two-family KV contract (256 sliding / 512 full attention, two lifetimes): two coordinate systems with the never-conflate invariant frozen — slots derive in-kernel from the resident origin; storage declared along orthogonal axes (lifecycle, paged/slab live storage, reference/copy checkpoint materialization) with observable semantics frozen, not a closed pairing; staged serving — unreclaimed pool first, paged window with batching; a frozen checkpoint query — the largest position at which every required family is exactly recoverable, and a global prefix without its local window is worthless; capacity and TP accounting; four frozen behaviors with a default-off prefix-cache amendment. |

## models / glm52

| Path | TL;DR |
| --- | --- |
| `models/glm52/cuda-graph-png.md` | `--dump-graph-png PATH` exports GLM5.2 rank 0's live EP whole-step graph (historical notes also cover retired TP8). |
| `models/glm52/dp-scheduler-metrics.md` | GLM5.2 maps logical scheduler partitions onto vLLM EngineCore identities (EP multi-engine / historical TP8 single-engine). |
| `models/glm52/moe-tp8-low-latency.md` | **Historical** Hopper TP8 LL decode record (path removed; not a launch contract). |
| `models/glm52/pd-native-mtp-handoff.md` | TP4 P transfers committed native-MTP KV + an initial proposal; D enters `first_step=verify`. P→EP16 gate closed (Slurm 4-node fleet, multi-turn c64 640/640); async admission restore fixed the ITL p99 tail (98→31 ms, PR #801); pool sizing and session-affinity findings live here. |
| `models/glm52/serving-status.md` | **Authoritative** GLM5.2 matrix: Blackwell-only; EP4/EP8/EP-N decode uses SM100 DeepGEMM masked grouped expert GEMMs; TP4 is prefill-only; Hopper/TP8/decode-TP are removed. Bring-up until indexer + lifecycle gates close. |
| `models/glm52/tp4-gb300-bringup.md` | Historical TP4 decode bring-up notes on 4×GB300 (decode TP LL removed; active TP4 contract is prefill-only). |
| `models/glm52/tp4-prefill-only.md` | Active TP4 contract: `--moe-topo=tp4 --glm52-prefill-only`, layer-outer NCCL bf16 AR on 4×GB300. |
| `models/glm52/ep4-gb300.md` | DP4/EP4 (`--moe-topo ep4`) uses 64 whole experts/rank, a four-rank DeepEP shim, and the shared SM100 DeepGEMM masked grouped chain. Current gates cover the EP4 oracle, automatic VRAM budget, and all graph buckets; same-session replacement A/B improved decode p50 3.64% and aggregate throughput 3.9%. Historical weight-only and pinned-staging records remain as explicitly labeled baselines. |
| `models/glm52/free-running-dp.md` | 设计文档:删除 DP coordinator——每个 DP rank 是独立 engine(scheduler/BlockPool 自持),loop 无条件全速跑,唯一耦合是固定节拍的 DeepEP collective 链;三条纪律(固定链/保守 bound/padding 即协议)。**五个 gate 全 GO(GB300 tray03,2026-07-30);迁移两步均已实装——per-rank bucket/协议 bound/MTP 固定链 + 拆壳(自治 engine 线程、always-consume lease、rank-host 退役、bootstrap rendezvous);双 tray EP8 首验通过(rendezvous 一次成功、多进程 MTP、fail-stop,tray03+tray13)。EP4 回归与 EP16 验证待跑。** 取代 cross-node-scaling.md 的 Event plane 与 SMR 方向。 |
| `models/glm52/cross-node-scaling.md` | NVL72 实测存档:one rack = one NVLink/IMEX domain, single-LSA DeepEP shim works cross-tray with no GIN un-baking — 2-tray EP8 solo p50 23.61 / p99 24.00 ms (≈ loopback), live-verified through EP32 × 8 trays; EP widths {4,8,16,32,64} each a constexpr shim instantiation. Pitfalls: containers need the IMEX channel device; teardown must shutdown() the socket. **framed-TCP rank-host 已随 free-running 拆壳删除(2026-07-30);跨节点 = 每节点一进程 + bootstrap rendezvous。** GIN scale-out sections remain the design for IB/RoCE beyond one rack. |
| `models/glm52/dspark-mtp.md` | DSpark speculative decoding (community `RedHatAI/GLM-5.2-speculator.dspark`, not native MTP): qwen3-arch 5-layer draft at hidden 6144 + rank-256 Markov head; verify span rides the decode buckets (span-4 default). M1 span-steps, M2 draft lane, M3 greedy round loop (sharegpt c1 1.52×), M4 sampled verify — non-greedy speculation via prefix-match over sampled tokens (c1 code 2.38× at temp 1; full rejection sampling probe-measured out at ≤ +1.5%). |
| `models/glm52/native-mtp-accuracy.md` | Native MTP must consume the target model's final-normalized hidden, matching official vLLM rather than the pre-final-norm residual. The fix raises matched c8 accepted length from `1.753` to `3.725` versus `3.786`; repeated matched c1 measures PegaInfer `7.749 ms` TPOT versus official vLLM `8.814 ms`. |
| `models/glm52/paged-kv-prefix-cache.md` | Static per-slot KV partitions → per-rank `BlockPool` of 64-token content-hashed pages (Kimi #239 pattern, zero kernel changes): full-lifetime admission reservation, coordinator-shipped `Glm52StepKv` page rows, prefix caching on by default (suffix-only prefill), DSpark × prefix-cache mutually exclusive, launch-ahead lease breaks at page boundaries. Merged as #588: all jz-38 gates green, warm prefix TTFT 14.12 s → 0.84 s (16.9×) byte-identical, step bench flat vs the D5 anchor. |
| `models/glm52/continuous-batching.md` | D2 + D2.5 execution record: multi-slot admission (8 requests/rank, least-loaded first) + {1,2,4,8} batch-bucket graphs (smallest bucket covering the fullest rank, per-bucket `Glm52BucketState`). Solo 22.4 ms/step; D2's c9 cliff killed (47.1 → 31.8 ms/step, 171 → 254 tok/s); poisson soaks clean; pinned slot-3/7 parity PASS. Known: buckets are distinct FP associations (bucket-crossing requests can greedy-diverge at near-ties); open anomaly: one-off silent request drop (#551). |
| `models/glm52/oracle-harness.md` | Self-contained accuracy oracle: `tools/accuracy/glm52_oracle.py` (pinned transformers 5.12.1 official `glm_moe_dsa`, fp8-precision-emulated) emits hardcodable Rust probe constants; `oracle/mla.rs` replays the seeded input and asserts. MLA gate green on jz38 (64/64 probes, diff RMS 1.8e-5), negative controls red. No MB fixtures in git. |
| `models/glm52/indexer-forward.md` | Indexer model-crate record: `glm52_indexer_forward` composes the #489 kernel ops and aligns with vLLM's `DeepseekV32Indexer` — k_norm is LayerNorm-with-bias, weights fold into DeepGEMM, and no Hadamard is applied. Merged as #521; oracle gate green (2013/2048 slot overlap). |
| `models/glm52/ep8-deepep-moe.md` | PR4: GLM-baked DeepEP v2 shim instantiation replaces PR3's local scatter/combine; loader places experts into their packed layout at H2D time (post-load repack cannot fit HBM); rank 0 runs the full 78-layer spine + bs=1 greedy coordinator, ranks 1..7 replay the 75 MoE collectives per step. Gates: EP8 layer-6 oracle 62/64 (same outliers as EP1), full-model e2e generation. |
| `models/glm52/ep1-forward.md` | PR3 built + all gates green on jz-38 H200 (2026-07-03): MoE/dense/bookend bricks (cherry-picked from the PP8 branch, re-gated via the #499 harness) + decoder-layer composition with cross-layer top-k sharing. MoE chain shaped to the DeepEP v2 elastic shim contract, Grouped + GEMV expert paths behind one signature; graph capturability as the bar. Gates: bookend exact, layer-0 dense 64/64, layer-6 MoE 62/64 both paths (measured router near-ties, bounded allowance). |
| `models/glm52/whole-step-decode-graph.md` | Whole-step decode graph execution record: 200 → **19.6 ms/step** on jz-38 (below the vLLM 20.0 reference) via CUDA graph + weight-only fp8 GEMV (tensor-core mma at batch 4/8) + DeepGEMM masked grouped expert GEMM (the earlier "swapAB" attribution retracted inside). Remaining lever = collective wait structure (#542); indexer oracle reference drift (#541). |
| `models/glm52/fp8-blockwise-gemm-lab.md` | kernel_lab fp8 blockwise GEMM lab line (8 units: production CUTLASS `fp8_gemm.*` + CuTe DSL tcgen05 `fp8_gemm_dsl_tc.*`, `sm_tcgen05_only` fail-closed): GB300 sm_103 DSL passes 24/24 and beats the CUTLASS production baselines at rows=64 by 1.37–1.97x (tuning round over tile-N/split-K adds −21~−40% more), cold-L2 recheck holds; production profit estimate and SASS/PTX evidence inside. |
| `models/glm52/kernel-lab-ops.md` | kernel_lab ops & maintenance manual: pipeline map (manifest -> registry -> adapter -> loader -> timing -> ledger), the add-a-CUTLASS-unit and add-a-python-native-DSL-unit lifecycles, capability-key semantics, warm vs `--cold-l2` protocol boundaries + the <11us event-timing floor, ledger discipline, and the GB300 tray remote-workflow record. |

## models / k3

| Doc | TL;DR |
|---|---|
| `models/k3/bring-up.md` | New model line (`--features k3`). Decode end-to-end: the full 93-layer model serves at `--k3-ep-size 4` (free-running per-rank engines, zero host collectives per step, 189 GiB/rank); routed experts are one fused DeepGEMM MegaMoE(situ) launch per layer at every world size, with the masked FP8xFP4 grouped-GEMM chain retained test-only as the numerics anchor. Single-rank: buckets to 128, per-bucket CUDA graphs on, token-matching a certified 4-layer golden. Dev vehicle: 224-expert checkpoint at EP4, shape-isomorphic to the full 896-expert model at EP16. |

## models / deepseek-v2-lite

| Path | TL;DR |
| --- | --- |
| `models/deepseek-v2-lite/status.md` | DeepSeek-V2-Lite EP2 status ledger: correctness, direct attribution, lifecycle reliability, and the #466 six-child host-staged/NCCL HTTP SLO report remain separate evidence buckets; no soak, parity, or production claim. |
| `models/deepseek-v2-lite/benchmarking.md` | Verification ladder and commands for DSV2-Lite correctness, direct decode diagnostics, retained host-staged/NCCL HTTP SLO profiles, and the separate soak/production boundary. |
| `models/deepseek-v2-lite/roadmap.md` | Single-node EP2 roadmap: #466 retained SLO reporting is an evidence layer; soak, device KV/attention, long-prefill scheduling, and Stable promotion remain separate gates. |
| `models/deepseek-v2-lite/benchmark-artifact-manifest.md` | Issue #467 implemented: the retained DeepSeek-V2-Lite benchmark matrix emits `artifact_manifest.json` and `regression_summary.json`, with CPU-only summarize-only tests. |
| `models/deepseek-v2-lite/hf-accuracy-gate.md` | DeepSeek-V2-Lite EP2 HF accuracy gate after PR #149/#150/#274: HF `generate(use_cache=true)`, host-staged EP2, and NCCL EP2 are compared across the committed small case set. |
| `models/deepseek-v2-lite/decode-attribution-gate.md` | Direct diagnostic benchmark for DeepSeek-V2-Lite EP2 `Hello`/16-token batch sizes 1/4/8: structured timing/counters and graph probes with no HTTP SLO or production claim. |
| `models/deepseek-v2-lite/source-layout.md` | DeepSeek-V2-Lite runtime layout refactor: `runtime.rs` split by responsibility, HF/host-staged/NCCL EP2 E2E exact on 2x RTX 5090; NCCL CUDA Graph smoke remains a diagnostic blocker on that host, independent of the passed correctness gate. |
| `models/deepseek-v2-lite/device-resident-nccl-combine.md` | Issue #275 record: NCCL decode combine uses reusable device-resident f32 scratch; current NCCL graph-readiness blockers live in `status.md`. |

## models / kimi-k2

| Path | TL;DR |
| --- | --- |
| `models/kimi-k2/roadmap.md` | Cross-cutting Kimi-K2 plan, re-verified 2026-06-08 on 8×H200. Decode leads vLLM on the active TP1/DP8 **DeepEP** line (bs64 graph TPOT `26.3 ms` p50 / `30.5` p99); M1 serving contract (sampling/EOS/admission) + M2 accuracy gate shipped and green teacher-forced. Live frontier = serving perf: the "+51% HTTP" (#225) was a **bench/metric artifact** (measured: identical prompts under-measure decode ~7–15% via the Marlin expert GEMM; transport ≈0) — floor ~34 ms, a2a ~30% GPU (#228); TTFT 4.5×/31× behind vLLM (#224). Open correctness debt: tests (#222), concurrent mispick (#286), graph-replay gate (#300). |
| `models/kimi-k2/accuracy-gate.md` | vLLM-golden accuracy gate (#223)：`tests/vllm_golden_gate.rs` + committed K2.6 fixture，teacher-forced regret sweep + free-greedy decode parity，走真实 serving path（TP1/DP8/EP8 PPLX）；两档 regret 规则（自信位 0.30 / 平分布位 1.25 且每 pass 限 2 个），缺模型/fixture 显式 fail。 |
| `models/kimi-k2/deepep-migration.md` | PPLX→DeepEP 迁移已实现：kimi 路径 PPLX 全删（moe_pplx.rs 没了，kimi crate 不再依赖 pegainfer-comm）；decode `expand=true`+`cpu_sync=false` 零 host 同步/分配（graph-ready，#227 capture 仍关）；Marlin 原地消费 recv buffer（alignment 8 == block size，identity routing + sentinel）；router scale 在 residual 处应用，combine 提前一步 bf16 取整。待 8×H200 数值 gate + serving bench。 |
| `models/kimi-k2/sampling.md` | Sampling param surface + design (#237)：TP1/DP8 上 temperature/top_k/top_p 经单次 batched FlashInfer pass 生效（greedy 行保持 in-graph argmax，零开销），TP8 显式拒绝非 greedy；OpenAI 参数表逐项标注 honored/rejected/ignored，无静默路径；8×H200 已验证 e2e + TPOT 无回归。 |
| `models/kimi-k2/kv-cache-design.md` | KV cache 接入 qwen3 paged 栈 (#239→#230/#231)，单 PR 落地：kimi kernel 层本就 paged，kernel 零改动；kvbm `BlockPool` per rank 取代静态 slot→pages 映射，full-lifetime reservation admission + 超界显式 Rejected，per-request cap 2048→8192（DP prompt 仍 ≤2048，PPLX fabric buffer 约束）；#230/#231 的 substrate，8×H200 验证待做。 |
| `models/kimi-k2/optimization.md` | Kimi-K2 model card + decode 优化主线。Active mainline 是 TP1+DP8+EP8 PPLX（decode batch cap 64，buckets `[1,2,4,8,16,32,64]`，bs64 output `1336 tok/s`）；下半篇的 TP8+EP8 NCCL bs4 graph TPOT `14.39ms` 路径是历史 bring-up 记录，保留以解释 MLA/MoE/collective kernel 结构。 |
| `models/kimi-k2/bringup-history.md` | Kimi-K2 text-only bring-up 压缩史（合并自旧 support-analysis/changelog/operator-todo trio）：HF probe → 文本 manifest → TP8/EP8 sliced loader → MLA + Marlin WNA16 routed expert → NCCL bridge → bs4 wave decode → 整段 CUDA Graph → vLLM top-20 gate。持有 still-load-bearing 的 checkpoint/INT4/Marlin layout facts 与 #234 tombstone（expert-major CUTLASS 删除、weight_shape 不再加载、bs4 cap → 64）。 |
| `models/kimi-k2/vllm-path-comparison.md` | Kimi-K2 decode 路径对照：vLLM-style fused qkv_a、MoE shared/routed compute overlap、shared/dense gate-up fusion、routed scaled-add 和 bridge microbench 已过 H20 gate；output64 avg/p50/p99 均在 `15ms` 内，vLLM TP-only MoE final all-reduce BF16/F32 两版均慢于当前 RS bridge。 |
| `models/kimi-k2/vllm-h20-baseline.md` | vLLM 0.19.0 H20 ×8 TP1+DP8+EP8 decode-heavy baseline：bs 1..256 扫描，bs=8 拐点 TPOT med `26.4ms` / aggregate `308 tok/s`，bs=256 拉到 `1131 tok/s`；同 client 下 pegainfer TP8+EP8 bs=4 TPOT `19.13ms` 比 vLLM 低 23%，但 HTTP 口径比 in-process 高 33%，frontend overhead 待查。 |
| `models/kimi-k2/pplx-ep-decode.md` | PPLX EP decode bs=1 TPOT 37ms → 17.94ms（−52%），超过 NCCL no-graph 18.52ms。根因是 expert_padding=64 导致 Marlin 98% 计算浪费 + <<<1,1>>> 串行 routing kernel。含完整优化 log、failed approaches、nsys 对比数据。 |
| `models/kimi-k2/pplx-ep-correctness.md` | TP8/EP8 PPLX correctness baseline：H20 64-token token trace 与 TP8/EP8 NCCL 完全一致，hash `4920f088c2338236`；记录 recv capacity、routed-row top-k weight、F32 combine 边界。 |
| `models/kimi-k2/tp1-dp8-ep8-performance.md` | TP1 DP8 EP8 性能优化 ledger：O1 prompt_len1 decode admission 过 vLLM bs64 gate；O2 落地 5 个 decode kernel cherry-pick（cuBLASLt fixed-shape GEMM、argmax split、router fusion），精度由 base-vs-opt prefill logits A/B 压在 bf16 ULP 底，PPLX Marlin small-N tile 因 `-inf`/SIGSEGV 被定性为原分支精度破坏点并拒绝；bs64 TPOT 噪声内持平（p50 `40.58→40.09ms`）。 |
| `models/kimi-k2/source-layout.md` | Kimi-K2 source files over 1k lines were split by responsibility; the largest Rust file under `pegainfer-kimi-k2/src` is now `layers/attention.rs` at 950 lines. |
| `models/kimi-k2/dp-design.md` | TP×DP 可配置并行：每 DP rank 是独立 decode engine，EP all-to-all 天然 sync，轻量 load balancer 做 request 路由。首批 TP1×DP8 + TP8×DP1。 |

## subsystems / kv-cache

| Path | TL;DR |
| --- | --- |
| `subsystems/kv-cache/design.md` | 异构 KV（full attn / MLA / SWA / linear state）统一为「组 + checkpoint」模型：两类封存语义（paged 原地封 / bounded seal-by-copy）、对齐组、单索引以最稀疏 checkpointer 为准。`pegainfer-kv-store` 收编 qwen3/glm52 手写 offload 编排（resolve/seal/retire），模型侧只声明 `KvModel`。迁移：qwen3 → glm52 D → P/D 租约 → qwen35 bounded。 |
| `subsystems/kv-cache/resolver-ownership.md` | #830 八轮评审复盘与后继设计：plain 路径合规、偏航在 native P/D 把权威分配放进 resolver（HeadroomLedger 即被警告的第二本账）；定形 radix-first + tail pad-to-boundary（keyed API 族删除、失败前置到 admission 前）、仲裁按池内 async reserve + prefetched 抵扣归位；#830 契约测试全集为后继验收。 |

## subsystems / runtime

| Path | TL;DR |
| --- | --- |
| `subsystems/router/kv-aware-routing.md` | Dynamo KV-aware routing on 8×Qwen3-4B (RTX 5090): cache-affinity routing keeps a multi-turn conversation on its home worker, so follow-up-turn TTFT stays flat ~45ms vs round-robin 160–170ms / random 165–180ms (all-turns p50 3.3–3.8× lower). Router prefix overlap 0.72 under KV, 0 under stateless policies; `kv_hit_rate>0` is the gate that the worker↔router block-hash bridge is actually matching. Includes the per-response `prompt_tokens_details.cached_tokens` signal. |
| `subsystems/runtime/runtime.md` | Runtime complexity is controlled by a shared `pegainfer-core` that owns the generation contract and orchestration; per-model crates implement `ModelForward` so prefill/decode and hybrid attention stay hidden from the caller. State (`&mut`) is separated from weights (`&self`) for future bs > 1. |
| `subsystems/runtime/kv-cache-design.md` | Dynamo 式 logical/physical 分层 KV cache：BlockManager 管 block 生命周期和 admission，PhysicalBackend trait 管 GPU 内存和布局（FullAttention / MLA）。支持 TP / DP。基于 vLLM/Dynamo/pegaflow 调研。 |
| `subsystems/runtime/pegaflow-offload-integration.md` | 把 `pegaflow-core` 当进程内 Rust 库做 KV 卸载物理后端（HBM→DRAM/SSD/RDMA），补 kvbm 没写的卸载层。**Qwen3-4B full-attn 首发，端到端已在真实 GPU 跑通并验证**（async SAVE+LOAD 接进 executor/scheduler，纯 CPU-hit 与 GPU+CPU 组合 hit 恢复后 logits 与冷算一致）。pegaflow 经 git rev pin（#331+#333）。默认关，server CLI 已接（#316：`--kv-offload`/`--no-prefix-cache`，plain+LoRA）。linear 排除，sparse 暂缓。 |

## subsystems / scheduler

| Path | TL;DR |
| --- | --- |
| `subsystems/scheduler/scheduler.md` | Single dedicated thread owns GPU; FCFS prefill-priority, paged KV, bucket CUDA Graphs, unified forward for mixed prefill+decode. Qwen3-4B at QPS=2 is within 2% of vLLM throughput while winning TTFT (-16%), TPOT (-3%), and latency stability. Open: ITL p99 tail, Qwen3.5 full-paged prefill, and high-concurrency wedge triage. |
| `subsystems/scheduler/output-dispatch.md` | GPU bubble study + token-dispatch redesign (**landed 2026-06**). Single-thread CPU↔GPU(sync) alternation idles the GPU through scheduling; bubble ≈3µs×batch (bs=128 → ~380µs, 2% of an 18ms step on 5070 Ti), dominated by N per-request `token_tx.send` wakeups. Fix shipped: `token_tx` is a `TokenSink` drop-in over one request-tagged channel + one bridge demux loop (N→1 wakeups/tasks/ZMQ msgs); cancellation rides an `Arc<AtomicBool>` flag, not a separate channel. Bubble target ~150µs (exec_cpu floor). Trigger: fast GPUs (→10–15%) or N≫128. |
| `subsystems/scheduler/qwen-batched-sampling.md` | Issue #284 record: Qwen3/Qwen3.5 mixed greedy/non-greedy token selection compacts non-greedy rows into one batched FlashInfer sampling call per step, with greedy rows staying on indexed batched argmax. |

## subsystems / sampling

| Path | TL;DR |
| --- | --- |
| `subsystems/sampling/pegainfer-sample.md` | `pegainfer-sample` is the one crate every model routes through for batched token selection (`select_batch`) and host logprobs (`token_logprob_from_row`, generic over f32/bf16). Replaces `core::ops::select_batch_tokens_into` + three copies of the logprob math. Kimi keeps its sharded-vocab greedy argmax (a DP concern the whole-vocab `select_batch` can't express) but shares the non-greedy sampler and the logprob math. |

## subsystems / frontend

| Path | TL;DR |
| --- | --- |
| `subsystems/frontend/frontend-architecture.md` | `pegainfer-frontend` owns everything north of the model schedulers. Two contract generations coexist: the step contract (`StepOutputs` wire + typestate handles + contract-owned polling driver; qwen3 fully migrated via `frontend_adapter.rs`) and the legacy `EngineHandle`/`TokenEvent` path (other five lines). Next: migrate glm52, then delete the legacy contract. |
| `subsystems/frontend/simulated-inference-engine.md` | CPU-only simulated model crate for vLLM/OpenAI frontend and `vllm bench serve` validation without CUDA, real model weights, or real-model performance claims. |
| `subsystems/frontend/cpu-profiling-baseline.md` | Frontend CPU profiling baseline using `pegainfer-sim` with fixed TTFT=5ms/TPOT=12ms: 200 req / concurrency=16 shows ~150ms TTFT overhead (no dominant hotspot), heap allocation ~10%, stream polling ~7.5%, IPC ~1%; reproducible benchmark command and perf evidence documented. |
| `subsystems/frontend/startup-time.md` | Qwen3-4B warm startup-to-ready: frontend tokenizer load runs concurrently with the engine load (HTTP still binds only after the engine registers); mmap teardown is paid at the end of load since #377; pinned-staging upload (2026-07) cuts warm ready 5.22s → 4.66s on sm_89, and the remaining floor is the engine's own post-load startup work. |
| `subsystems/frontend/prometheus-metrics.md` | `/metrics` request histograms work for every model; Qwen3, Qwen3.5, and GLM5.2 schedulers also publish running/waiting/KV engine gauges through `LoadSnapshot` watches. |
| `subsystems/frontend/dashboards/README.md` | Grafana 10.4-validated dashboard for PegaInfer's live `/metrics` surface: HTTP traffic, request outcomes, scheduler/KV state, token throughput, and request latency. |

## subsystems / correctness

| Path | TL;DR |
| --- | --- |
| `subsystems/correctness/logits-golden-gate.md` | Reusable pattern for guarding a model's logits against an HF bf16 golden without binding to one GPU's bits: teacher-force fixed sequences, assert a structural regret check on the argmax + mean/p99 of the logprob delta at the bf16 floor (never the absolute max — it grows with coverage). Replay bs=1 / batched eager / CUDA-graph for determinism / cross-request / padding surfaces. Qwen3-4B is the reference impl. |

## subsystems / kernels

| Path | TL;DR |
| --- | --- |
| `subsystems/kernels/pegainfer-kernels-boundary.md` | Architecture decision: reusable frontend/runtime/data-plane layers plus per-model engines; `pegainfer-kernels` keeps shared MoE/MLA substrate (`moe`: DeepEP/DeepGEMM/FlashMLA) separate from model-local surfaces such as the narrow GLM5.2 DeepGEMM/FlashMLA wrappers. |
| `subsystems/kernels/pegainfer-bench-hawk-audit.md` | Hawk retained `pegainfer-bench` on two report consumers, reduced workspace findings 576 → 361, and exposed dead state that was deleted; 332 remaining findings are in the excluded kvbm fork. |
| `subsystems/kernels/build-rs-submodule-init.md` | `pegainfer-kernels/build.rs` initializes missing git submodules automatically for first-time builds before checking vendored third-party kernel headers. |
| `subsystems/kernels/kernel-op-reports.md` | Qwen3 kernel/report tooling is feature-gated: `qwen3_kernel_report` covers per-op kernel reports, and `qwen3_model_report` emits runtime-traced eager-DAG decode operator rollups with TensorSpec `KernelCall`s, latency stats, tables, and Graphviz DOT; measured FA2 `CTA_TILE_Q=64` prefill default in place. |
| `subsystems/kernels/typed-forward-pipeline.md` | Reusable typed tensor pipeline macro in `pegainfer-kernels` so model crates can express common `typed_ops` chains without model-specific wrapper macros. |
| `subsystems/kernels/tvm-ffi-mvp.md` | Optional `tvm-ffi-triton-cubin` bridge in `pegainfer-kernels` plus a packed TVM wrapper for the Qwen3.5 GDR solve Triton AOT CUBIN launcher. |

## playbooks

| Path | TL;DR |
| --- | --- |
| `playbooks/developer-onboarding.md` | New-developer onboarding — toolchain, unified venv, build, tests, quick benchmark validation. |
| `playbooks/bench-vs-vllm.md` | pegainfer vs vLLM comparative benchmarking: method, workflow, typical configs, gotchas. |
| `playbooks/model-optimization-pipeline.md` | Per-model optimization methodology: 2 standard profiles, vLLM baseline, e2e dashboard + append-only optimization log. |
| `playbooks/profiling-guide.md` | GPU profiling playbook: nsys pitfalls, diagnostic paths, measured kernel comparisons. |
| `playbooks/accuracy-parity-playbook.md` | Accuracy debugging playbook: truth-source rules, first-diff workflow, bf16 rounding traps, and verified Qwen3.5 parity commands. |
| `playbooks/hawk-visibility-audit.md` | hawk workspace 可见性审计：必须 all-features（单 profile 的 dead_public 61% 假阳性），三个必需环境变量，kvbm fork 边界。 |

## lessons

| Path | TL;DR |
| --- | --- |
| `lessons/moe-bench-prompt-diversity.md` | MoE decode TPOT is routing-diversity-dependent: identical concurrent prompts route greedy streams to the same experts and under-measure decode TPOT by **~7–15%** (measured via a `--distinct-prompts` sweep, not the ~30% first claimed). Bench MoE+EP with seeded distinct prompts. nsys kernel diff proves the whole delta is the **Marlin expert GEMM** (per-launch ~2× K=1→64); the DeepEP all-to-all is flat → lever is grouped-GEMM tile efficiency, not a2a overlap (#228). Transport ≈0. |
| `lessons/flashmla-sm100-ue8m0-kv-scales.md` | FlashMLA V3.2 fp8 sparse decode on Blackwell truncates KV-cache scales to e8m0 (round-to-zero) — the cache writer MUST emit power-of-two (UE8M0) group scales or attention is silently ~0.67× on sm100/sm103 while sm90 looks fine. Bit us via the GLM5.2 EP4 oracle gate; weights-free kernel gate (`flashmla_sparse_vs_reference_gate`) now pins it per arch. Kimi-K2 shares the kernel — audit before any Blackwell move. |
| `lessons/profile-diff-before-blaming-transport.md` | Profiling discipline from the #225 misfire: when two profiles of one workload differ in wall-time, **diff `cuda_gpu_kern_sum` first** — transport can't change GPU kernel time, so a kernel delta means compute/data, full stop. I nsys'd both paths and missed a +15.6% Marlin delta in plain view. Also: pin the same metric both sides; chase tails, don't annotate them; a root cause without a number is a hypothesis. |
| `lessons/moe-dplb-decode-imbalance.md` | DPLB lesson for future PegaFlow/WiDeep MoE+EP serving: decode-side DP imbalance is a sticky KV-state problem; engines should emit raw progress while external router/proxy derive load and routing. |
| `lessons/moe-zero-prefill-long-prefill.md` | ZeRO-Prefill lesson for future long-prefill MoE serving: once a router selects long-P work, maximize batch throughput by preserving compute-bound execution, hiding expert-weight movement, respecting KV handoff boundaries, and measuring bottlenecks before committing to an AsyncEP-style backend. |
| `lessons/exact-match-gate-thread-cublas.md` | Two durable lessons from a Qwen3.5 e2e gibberish bug: worker threads that run a model must rebind the CUDA context and init thread-local cuBLAS handles, and exact-match greedy gates are sensitive to equal-logit top1 choices (keep a single FlashInfer selector). |
| `lessons/kimi-bringup-numerics.md` | Three MoE+TP greedy-parity / reporting lessons from Kimi-K2 bring-up, reusable on any MoE+TP decode engine gated on token-id parity: reduce hidden states in F32 not BF16 (BF16 bulk all-reduce silently breaks greedy); don't merge shared+routed expert reduce into one collective (breaks cold-batch greedy); always report p50+p99, never just mean (tail dominates on barrier-synced MoE+EP decode). |
| `lessons/cuda-green-contexts.md` | Local mirror of NVIDIA CUDA 13.1+ Green Contexts guide (§4.6): static SM/workqueue partitioning via runtime execution contexts; host-only changes, no kernel edits. Generated by `scripts/html_to_md.py`. |

## benchmarks

| Path | TL;DR |
| --- | --- |
| `benchmarks/qwen3-4b-serving-vllm-rtx5090.md` | **Deleted** — superseded by `models/qwen3/serving-performance.md`. |
| `benchmarks/k3-ep4-decode-profile.md` | K3 EP4 93-layer decode step profile (MegaMoE, GB300, 2026-08): 52% backbone B=1 dense GEMM at ~50% of SOL (cuBLASLt splitK), 20% fused MoE, 5% a mis-tuned router top-k; not launch-bound. Levers: near-SOL dense GEMV (~10 ms) + router fix (~2 ms) project ~50 → ~30 ms/step; FP8 backbone is the structural next step. |
| `benchmarks/deepseek-v2-lite-vllm-tp2-ep2.md` | DeepSeek-V2-Lite EP2 2026-06-28 snapshot: PegaInfer host-staged/NCCL passed correctness, direct diagnostics, HTTP pressure, and trace rows; stock vLLM TP2/TP2+EP2 are retained as FlashInfer SM120/CUDA 12.8 setup failures, with a separate FlashInfer-fixed vLLM validation and no parity claim. |
| `benchmarks/qwen35-4b-serving-vllm-rtx5090-2026-07.md` | Qwen3.5-4B vs vLLM 0.25.1 on 1x RTX 5090 for #469: correctness gates and retained HTTP matrix completed with zero failed requests, but PegaInfer does not reach vLLM parity; requested 1024/256 c16 is `17.36ms` / `807 tok/s` vs vLLM `9.34ms` / `1425 tok/s`, while direct c16 TPOT `9.14ms` points first to HTTP/frontend/scheduler attribution. |
| `benchmarks/qwen35-4b-serving-vllm-rtx5090.md` | Qwen3.5-4B TP1 vs vLLM 0.23.0 on RTX 5090: latest direct PegaInfer A/B improves TPOT by 2-3%; HTTP `vllm bench serve` shows prompt-len-1 decode close, but vLLM still leads 1024/256 TPOT and high-concurrency output tok/s. Includes Nsight Systems direct/HTTP gap notes. |
| `benchmarks/qwen-mixed-sampling-http.md` | Issue #412 HTTP mixed-sampling evidence: Qwen3-4B `/v1/completions` completed 64/64 with 32 greedy + 32 sampled requests, failed=0/timeouts=0, TTFT/TPOT/ITL/output tok/s retained; Qwen3.5-4B passed the same workload as supplemental evidence. |
| `benchmarks/bs1-4k64-vllm-pegainfer.md` | RTX 5090 single-concurrency probe: `input_len=4096`, `output_len=64`, no vLLM prefix cache. PegaInfer TTFT median `177ms` vs vLLM `198ms`; TPOT median `6.47ms` vs `6.36ms`; corrected output throughput `+6%` for PegaInfer. |
| `benchmarks/mixed-load-itl.md` | Qwen3-4B + Qwen3.5 mixed-load ITL (#244, #375): chunking-off sweeps via `bench_serving mixed`. Both freeze active decode for the full prefill. Qwen3 p99 blows up with prompt/QPS; the old Qwen3.5 “p99-immune” table is a **measurement artifact** (primary: hardcoded `max_batch=4` slot starvation — see #470 / `models/qwen35/mixed-load-itl-470.md`; secondary: short `bg_output_len`). Prefix reuse defeats it on Qwen3. |
| `benchmarks/accuracy-eval-results.md` | Phase 1 GSM8K: Qwen3-4B PASS (pegainfer 85.37% vs HF 85.82%, delta -0.45 pp). Qwen3.5-4B historical FAIL recovered by #250 (strict 79.38%, flexible 79.30% vs HF 79.45%). |
| `benchmarks/qwen3-8b-pd-vs-mix-h200.md` | Qwen3-8B 多轮负载三方 A/B（2×H200）：P/D 1P+1D vs mixed×2（会话亲和 LB）vs mixed×1。吞吐持平（47.8k vs 47.0k tok/s），P/D 赢在 decode 稳定性（TPOT p99 10.08 vs 12.77ms，turn2+ TTFT 恒定 ~107ms vs 爬升 71→132ms），冷 turn1 多付 ~200ms（M3 目标）。含 vllm-bench 命令与 `max_completion_tokens` 坑。 |

## conventions

| Path | TL;DR |
| --- | --- |
| `conventions/bench-regression.md` | **Retired 2026-08**: in-process `bench_serving` snapshot gate deleted with the frontend consolidation; HTTP-based benching (`scripts/bench_http_serving.py`, vllm-bench) replaces it. Doc keeps the conventions (per-GPU snapshots, p50-only thresholds) for a future rebuilt gate. |
| `conventions/coding-style.md` | Testing principle: prefer integration tests, don't test what E2E catches. |
| `conventions/migration-defense.md` | 重写/迁移 PR 红线：旧防御结构（序号/计数/等待/断言/guard）逐条注明接班人——继承、替代、或结构性论证废除；测试断言跟人走。出自 #830 复盘。 |
