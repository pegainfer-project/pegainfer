# DeepSeek-V2-Lite Status And Benchmark Ledger

> **TL;DR:** DeepSeek-V2-Lite keeps correctness, direct decode diagnostics, retained HTTP SLO reports, and sustained soak evidence as separate gates. HF/host-staged/NCCL exactness, HTTP lifecycle evidence, #466 SLO artifacts, and #465 host-staged/NCCL soak artifacts are retained without claiming production readiness.

Last touched: 2026-07

## Capability Contract

| Capability | Status | Evidence |
| --- | --- | --- |
| EP2 correctness bring-up | Available | PR #149 adds the model crate, EP2 expert ownership, rank1 expert-only loading, and the host-staged dispatch/combine baseline. |
| Naive NCCL backend | Available | PR #150 adds a dense correctness-first NCCL path. Host-staged remains the transport oracle. |
| HF token/text/hash gate | Available | PR #154 establishes the HF / host-staged / NCCL comparison; PR #176 refreshes it to Transformers `generate(..., use_cache=true)`. |
| HF widened case set | Available | Issue #274 adds a committed case set that keeps the HF / host-staged / NCCL oracle strict while adding additional prompts and diagnostic batch sizes `4` and `8`; the 2026-06-20 2x RTX 5090 run classified all 5 cases as `all_token_text_exact`. |
| Decode attribution | Available | PR #162 and PR #169 add CPU/GPU attribution, route counts, NCCL counters, CUDA event timing, and optional NVTX correlation. |
| Direct same-prompt diagnostic batch | Available | PR #184 and PR #196 cover batch sizes `1`, `4`, and `8` for the fixed same-prompt direct path. |
| Startup observability | Available | Load logs report safetensor shard count, mmap/deserialization timing, per-rank GPU model-load timing, backend, devices, and total EP2 startup time. |
| Device-resident NCCL combine | Available | Issue #275 keeps NCCL combine contributions/results on reusable f32 device scratch and preserves the HF / host-staged / NCCL exact gate on 2x RTX 5090. |
| Device-resident NCCL dense exchange | Available | Issue #276 reuses backend-owned bf16 dense-exchange scratch, clears rank1 zero-send every exchange, removes dense-exchange stream sync from the backend call, and preserves HF / host-staged / NCCL exactness on 2x RTX 5090. |
| NCCL route-plan replay | Available | Issue #277 builds a token-major host route plan once after top-k routing, replays that plan for NCCL expert launches and device contribution accumulation, keeps route counters visible, and preserves HF / host-staged / NCCL exactness on 2x RTX 5090. This remains the eager NCCL oracle path. |
| NCCL CUDA Graph readiness | Covered-shape diagnostic | Schema-2 `cuda_graph_readiness` now includes a fail-closed `full_decode_graph_probe`. The 2026-06-20 run reports capture, instantiate, replay, and verification success with `8/8` verified replays for the retained batch-1 NCCL decode step. |
| First mixed-request serving gate | Available | Issue #281 adds greedy-only request admission, FCFS deferral, explicit request-local rejection/error/finish events, and one owned `DecodeCache` per active request. The 2026-06-23 2x RTX 5090 run passed HF / host-staged / NCCL exactness and the mixed-serving E2E for host-staged and NCCL. |
| Long-shape NCCL collectives | Available | Issue #280 chunks large bf16 dense-exchange and f32 combine all-reduces. The 2026-06-24 2x RTX 5090 NCCL checks preserve HF / host-staged / NCCL exactness and complete 24/64/128-word direct long-shape probes. |
| HTTP trace and measured MoE throughput optimization | HTTP and direct evidence | Issue #280 logs DeepSeek-V2-Lite `pegainfer_http_trace` records and batches same-position decode subgroups. Issue #464 extends phase/decode-step attribution, groups host-staged and NCCL routes by stable `(owner_rank, global_expert)`, and moves the NCCL gate GEMM to a bitwise-matched CUDA logits kernel while keeping host top-k/softmax. Diagnostic serial/host rollback switches remain available for retained A/B and emergency rollback; they are not production tuning knobs. |
| HTTP reliability lifecycle gate | Available | Issue #453 adds `scripts/bench_dsv2lite_http_reliability.py`, which drives real streaming `/v1/completions` scenarios for client cancel/disconnect, unsupported params, active-cap overload, mixed short/long prompts with adjacent failures, and clean follow-up recovery. The 2026-07-04 2x RTX 5090 host-staged and NCCL runs both passed with terminal trace coverage, stable output hashes, active/pending/decode maxima, and healthy final scheduler baselines. |
| Retained HTTP serving SLO report | Retained HTTP evidence | Issue #466 adds model-owned short/mixed/long profiles on the shared HTTP benchmark scripts. The current retained run covered all six host-staged/NCCL children with zero failures/timeouts and full trace coverage. The #466 follow-up fixed NCCL no-selector readiness by discovering a compatible Python-wheel NCCL runtime from `PATH`; the short NCCL c1 HTTP smoke now reaches readiness and completes without startup failure or layer-1 illegal address. This is HTTP pressure/SLO evidence only; command details and artifact hashes live in `benchmarking.md`. |
| Sustained HTTP soak report | Retained soak evidence | Issue #465 adds `scripts/bench_dsv2lite_http_soak.py`, which reuses the shared HTTP leaf benchmark inside fixed time buckets, then retains backend-level and host-staged/NCCL combined JSON. The 2026-07-18 2x RTX 5090 run covered host-staged and NCCL with zero failures/timeouts, full required trace coverage, clean follow-up recovery, output hashes, resource samples, and first/last-quartile drift fields. The NCCL row used NCCL `2.26.2` with `NCCL_IB_DISABLE=1` and `NCCL_P2P_DISABLE=1`, so this is sustained HTTP soak evidence for that runtime contract, not a default-runtime speed claim or production-readiness claim. Current tooling also fails missing duration coverage, capped retained runs, missing leaf artifacts, missing clean follow-up, failed child gates, provenance drift, contract drift, and missing/generic runtime boundaries. |
| Retained vLLM comparison matrix | Snapshot complete with clean failed setup rows and supplemental validation rows | The retained matrix for tracking issue #279 keeps HF/host/NCCL correctness, PegaInfer direct diagnostic batch, `vllm bench serve` HTTP pressure, PegaInfer trace rows, and failed setup rows separate. The 2026-06-28 clean full matrix passed HF / host-staged / NCCL correctness plus PegaInfer host-staged/NCCL direct, HTTP pressure, and trace rows; stock vLLM TP2 and TP2+EP2 failed during setup on the target FlashInfer SM120 path. A separate FlashInfer #3633-equivalent validation completed vLLM TP2 and TP2+EP2 under the same HTTP client/workload contract. |
| vLLM production parity | Not claimed | The vLLM TP2 / TP2+EP2 rows are gap-finding evidence from a documented contract. The supplemental validation run is not serving parity or a stock-install claim. |

## Correctness Contract

The retained correctness gate is deliberately narrow:

- model: DeepSeek-V2-Lite;
- devices: single-node EP2 with two local GPUs;
- committed cases: `test_data/deepseek-v2-lite-ep2-cases.json` keeps the original `Hello` / 16-token case and widens the oracle with a few additional prompts plus batch sizes `4` and `8`;
- generation mode: greedy;
- backends: host-staged and `PEGAINFER_DSV2_LITE_EP_BACKEND=nccl`.

The comparison gate must be run on the same model snapshot for HF, host-staged, and NCCL outputs. Same-host comparison remains strict: HF, host-staged, and NCCL must be token-exact and text-exact for every committed case and every diagnostic batch row. Host-staged remains the baseline oracle for NCCL transport changes. The latest retained evidence is the 2026-06-28 2x RTX 5090 case-set run with `case_count=5`, top-level `classification=all_token_text_exact`, no comparison warnings, token hash `4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225`, and text hash `0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347`.

The mixed-request serving E2E computes sequential greedy token-id oracles with `DeepSeekV2LiteEp2Generator::generate_greedy`, then submits concurrent requests through `start_engine`. The retained 2026-06-23 run covers same-length mixed prompts for same-position batch decode, different-length mixed prompts for single-row decode fallback, and a valid request submitted beside an invalid `logprobs` request to prove explicit rejection does not poison the valid stream. Host-staged and NCCL both passed the mixed-serving E2E.

The HTTP reliability gate is intentionally separate from the mixed-serving E2E. It proves that the serving bridge and scheduler surface terminal states in a machine-readable way, then uses a clean follow-up request after every failure scenario to show state recovery. Rejected requests that fail in the HTTP/frontend guard may have no scheduler trace; rejected requests admitted to the DSV2-Lite scheduler must have `pegainfer_http_trace` terminal evidence. Cancelled and disconnected streams are classified separately by the request-local `TokenSink` cancellation flag versus a closed shared channel.

The Rust E2E accepts the known HF-confirmed RTX 5090 and A800 hash pairs for this narrow shape, because the same model snapshot has produced different exact greedy text on those hosts while still matching HF on each host. Do not use the static hash pair list as a substitute for the same-host HF comparison when changing accuracy-sensitive code.

`e2e_ep2.rs` is a correctness/integration gate. Its JSON uses `report_intent=correctness_integration`; timing percentiles, throughput, SLO budgets, and soak claims are intentionally absent. The direct attribution binary and HTTP report commands are mapped in `benchmarking.md`.

## Benchmark Ledger

### Issue #466 Retained HTTP SLO Report

The retained SLO layer uses model-owned `scripts/bench_dsv2lite_http_slo.py` over the generic `bench_http_serving.py` and `bench_http_sweep.py` harnesses. The model line owns three fixed contracts:

| Contract | Shape | Boundary |
| --- | --- | --- |
| short decode-heavy | `prompt_words=64`, `max_tokens=64`, 240 s timeout | Repeated short HTTP pressure/SLO evidence |
| mixed prompt shape | alternating `prompt_words=64,512`, `max_tokens=64`, 240 s timeout | Mixed-shape tails and trace evidence |
| long-prompt smoke | `prompt_words=2048`, `max_tokens=64`, 900 s timeout | One long boundary cell, no broad long-context claim |

Every retained profile fixes shape, timeout, request count, concurrency, repeats, greedy sampling, ignore-EOS, and full trace coverage. Repeat summaries report median/min/max and a stable/noisy/failure marker. `benchmarking.md` is the command and artifact-schema source of truth.

On Blackwell (`sm_120`), DSV2-Lite NCCL rejects runtimes older than `2.26.2` before communicator creation and reports how to select a compatible NCCL library. NCCL 2.26.2 is the first upstream release containing NVIDIA's recent-Blackwell shared-memory fix. This floor does not apply to non-SM120 GPUs.

The #466 follow-up readiness fix keeps this fail-closed floor but also scans Python executables on `PATH` for the `nvidia-nccl-cu12` wheel library. On the 2026-07-15 2x RTX 5090 run, `PATH=<conda-root>/bin:...` was enough to load NCCL `2.26.2` with no explicit selector. The same run preserved HF / host-staged / NCCL exactness, passed NCCL direct batch 1, passed NCCL short HTTP c1, and passed a host-staged short HTTP c1 no-regression smoke. This does not claim #465 soak, #452 long-prompt readiness, #635 device attention/KV readiness, #636 route-plan readiness, production readiness, or vLLM parity.

The current-source #466 aggregate is retained under `artifacts/bench/dsv2-lite/<run-id>/` (gitignored). `benchmarking.md` contains the command, artifact fields, aggregate hash, and claim boundary.

### Issue #465 Retained HTTP Soak Report

The retained soak layer uses model-owned `scripts/bench_dsv2lite_http_soak.py` over the generic `bench_http_serving.py` leaf runner. The retained 2026-07-18 rerun used commit `a5703d0424d917ce99b4bd8691b0b86eecde966f`, model revision `604d5664dddd88a0433dbae533b7fe9472482de0`, 2x RTX 5090, `prompt_words=64`, `max_tokens=64`, greedy sampling, ignore-EOS, concurrency `4,8`, 120-second duration, and 60-second buckets. Both backends passed `soak_gate.passed=true` under that run's gate. Later commits harden the retained gate semantics; regenerate artifacts before claiming final-head retained evidence. Command details and artifact hashes live in `benchmarking.md`.

| Backend | Runtime boundary | Completed | Failed/timeouts | Buckets / leafs | Combined output hash | Boundary |
| --- | --- | ---: | ---: | ---: | --- | --- |
| host-staged | `host-staged` | 112 | 0 / 0 | 4 / 14 | `6912777bed672f57` | Sustained HTTP soak evidence for the host-staged oracle path |
| NCCL | NCCL `2.26.2`, `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1` | 128 | 0 / 0 | 4 / 16 | `24f2db9fc47acc10` | Sustained HTTP soak evidence for this conservative single-node NCCL runtime |

The combined retained report passed child gates for both backends, commit consistency, model provenance consistency, and runtime-boundary retention. The soak JSON records first/last-quartile tail, throughput, RSS, and device-memory drift, but numeric drift is still reported evidence rather than a production budget. Current tooling also records per-device and total VRAM drift so one EP rank cannot hide behind the max-device summary. Treat long-duration and long/mixed-prompt soaks as follow-up contracts, not implied coverage from this short-shape baseline.

### Retained vLLM TP2/EP2 Matrix

The retained matrix lives in `docs/benchmarks/deepseek-v2-lite-vllm-tp2-ep2.md` and tracks [#279](https://github.com/pegainfer-project/pegainfer/issues/279). It is the current source for PegaInfer host-staged/NCCL versus vLLM TP2/TP2+EP2 under the `prompt_words=64`, `max_tokens=64`, `num_prompts=32`, `max_concurrency=1/4/8`, `temperature=0`, `ignore_eos=true` HTTP pressure contract. Prompt words are a workload-generator input, not a token count.

Latest 2026-06-28 result on 2x RTX 5090:

| Bucket | Result | Claim boundary |
| --- | --- | --- |
| Correctness | HF dump, PegaInfer host-staged E2E, and PegaInfer NCCL E2E all passed; comparison classified `all_token_text_exact` with no warnings. | Correctness bucket only; no HTTP serving claim. |
| Direct diagnostic batch | PegaInfer host-staged and NCCL batch `1/4/8` all passed with token hash `4fb4c8825fe4d2c4...`. | Direct same-prompt model-path evidence only; do not compare the backend TPOT rows as production performance. |
| HTTP pressure | Clean PegaInfer host-staged and NCCL completed all `1/4/8` concurrency cells; host-staged c4, NCCL c4, and NCCL c8 were noisy. Clean vLLM TP2 and vLLM TP2+EP2 failed server startup on the target FlashInfer SM120 path. | `--max-concurrency` is client pressure, not true internal batch size by itself. |
| Supplemental vLLM validation | A separate FlashInfer #3633-equivalent validation run completed vLLM TP2 and TP2+EP2 for all `1/4/8` concurrency cells. | Not a clean stock vLLM package-stack claim; it only shares the HTTP client/workload contract. |
| Trace pass | PegaInfer host-staged showed `decode_batch_size_max=1/4/5` and NCCL showed `1/2/5` for concurrency `1/4/8`. | PegaInfer-only trace evidence; no vLLM internal claim. |

### HTTP Trace And MoE Optimization

The corrected trace contract uses real `/v1/completions` traffic with `prompt_words=64`, observed prompt tokens `84..87`, `max_tokens=64`, concurrency `1/4/8`, greedy decoding, ignore-EOS, and no warmup. Host-staged and NCCL completed the retained trace cells with zero failures, timeouts, or missing traces. Active sets and batched decode steps formed, while decode time still grew with active rows, which ruled out scheduler admission as the final limiter.

Backend attribution found two costs: repeated outer expert replay for routes sharing `(owner_rank, global_expert)`, and the NCCL path copying full hidden states to the host for the gate GEMM. The optimized path groups routes, preserves token-major contribution order, and computes NCCL logits on CUDA with bitwise coverage against the host accumulator. Top-k and softmax remain on the host. The rollback env switches are diagnostic only, used to reproduce serial/host-router A/B rows or recover the old path if a retained gate fails. Direct batch GEMM remains rejected because it changed the retained token hash.

| Backend | c1 | c4 | c8 | Boundary |
| --- | ---: | ---: | ---: | --- |
| host-staged paired change | `+14.4%` | `+16.3%` | `+18.3%` | retained short-shape HTTP A/B |
| NCCL paired median change | `+23.4%` | `+21.7%` | `+29.5%` | retained short-shape HTTP A/B; multiple paired runs |

The retained A/B rows completed with zero failures, timeouts, or missing traces, and each paired baseline/optimized cell preserved its request output hash set. Direct `Hello`/16 attribution also preserved the exact retained token and text hashes.

The remaining profiler direction is attention/KV host-side work. Moving attention and KV state fully device-side changes cache ownership, long-context capacity, request retirement, and CUDA Graph semantics, so it remains a separate change with its own artifact contract.

These results cover direct `Hello`/16 and short fixed-shape HTTP traffic only. They do not prove mixed or long-prompt scaling, soak/SLO behavior, production readiness, multi-node EP, or vLLM parity.

### Direct Same-Prompt Diagnostic Batch

This path is useful for attribution and for avoiding the earlier row-loop TPOT measurement. It is separate from the first mixed-request serving gate and is not production continuous batching:

- every row uses the same prompt;
- prefill remains conservative;
- the direct benchmark path is not `/v1/completions` serving;
- it does not prove request admission, per-request KV ownership, fairness, or mixed-request scheduling.

Current retained direct snapshot from the issue #277 branch (`2f52ed6`, 2026-06-15, 2x RTX 5090 / SYS interconnect). Shape: `prompt="Hello"`, `output_len=16`, `warmup=5`, `iters=20`; every row produced token trace hash `ed0eab52473991fc`. `decode tok/s` is the benchmark report's aggregate `metrics.decode_tok_s`. This refresh replaces the older PR #184 row values for the current branch ledger, but it should not be read as an isolated route-plan speedup because the retained snapshot was rerun on a different validation environment.

| Batch | Backend | steady TPOT p50 ms | steady TPOT avg ms | decode tok/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | host-staged | 55.727 | 57.313 | 17.486 |
| 1 | NCCL | 181.795 | 188.420 | 5.321 |
| 4 | host-staged | 193.954 | 198.905 | 20.106 |
| 4 | NCCL | 303.389 | 311.621 | 12.821 |
| 8 | host-staged | 385.013 | 394.908 | 20.270 |
| 8 | NCCL | 472.045 | 483.538 | 16.517 |

PR #196 extends attribution for the same direct diagnostic shapes. The retained A800 attribution gate keeps `batch-size=1/4/8`, `prompt="Hello"`, `output_len=16`, host-staged, and NCCL exact against the same-host HF gate.

### HTTP Concurrency Pressure

The issue #277 branch was also run through `/v1/completions` with `vllm bench serve` used only as the common HTTP client. Shape: random input length `2`, output length `16`, `24` prompts, `temperature=0`, `ignore_eos`, `--max-concurrency 1/4/8`, PegaInfer `--cuda-graph=false`.

PegaInfer streaming currently makes the client-side TPOT fields near-zero in this shape, so this table reports output throughput and throughput-derived milliseconds per output token computed as `duration / total_output_tokens`. `--max-concurrency` should be read as concurrent request pressure, not as proof of true internal PegaInfer batch size.

| Backend | conc | completed | output tok/s | throughput-derived ms/output token | mean TTFT ms | median TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| host-staged | 1 | 24/24 | 20.912 | 47.820 | 764.471 | 740.296 |
| host-staged | 4 | 24/24 | 21.030 | 47.552 | 2838.390 | 3036.649 |
| host-staged | 8 | 24/24 | 20.964 | 47.700 | 5198.935 | 5991.506 |
| NCCL | 1 | 24/24 | 6.302 | 158.689 | 2538.216 | 2553.374 |
| NCCL | 4 | 24/24 | 6.326 | 158.083 | 9491.680 | 10097.244 |
| NCCL | 8 | 24/24 | 6.341 | 157.710 | 17242.941 | 20110.121 |

### Issue #280 HTTP Trace, Subgroup Decode, And Long Prompts

Retained 2026-06-24 evidence on 2x RTX 5090, NCCL EP2 with chunked large collectives, release `pegainfer-server --features deepseek-v2-lite`, `/v1/completions`, `temperature=0`, `ignore_eos=true`, `max_tokens=16`, `num_requests=8`, `repeats=3`, with server logs consumed by `scripts/bench_http_serving.py`.

This is HTTP serving evidence for request-level trace attribution, completed/failed/timeout accounting, output hash stability, and same-position decode subgroups. It does not prove vLLM parity, production EP readiness, or acceptable long-prompt latency.

Long-shape NCCL direct smoke after chunking:

| prompt words | prompt tokens | generated | token hash |
| ---: | ---: | ---: | --- |
| 24 | 32 | 16 | `78dfd3123da2ed54829027384682c6eb562a6d29b2a92ee96a7b26d7acc4e226` |
| 64 | 86 | 16 | `920f24edd016e8e16973f304e5cb909303812a930a9c6608694d0b47f2c48918` |
| 128 | 172 | 16 | `5fd2f30c1f1c4e4477791f233c30ce6c0148dba91737c358cb357a2065482861` |

HTTP 128-word smoke: `prompt_words=128`, `concurrency=1`, `num_requests=2`, `warmup=0`, actual prompt tokens `170,171`.

| completed | failed/timeouts | QPS | output tok/s | TTFT avg ms | TPOT/ITL avg ms | active max | decode batch max | traces | output hash |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2/2 | 0/0 | 0.331 | 5.295 | 2535.3 | 32.3 | 1 | 1 | 2/2 | `2299c1c50f50e819` |

Same-shape sweep: `prompt_words=16`, actual prompt tokens `20..23`.

| conc | completed | failed/timeouts | QPS avg | output tok/s avg | TTFT avg ms | TPOT/ITL avg ms | active max | decode batch max | traces | output hash |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 8/8 x3 | 0/0 | 2.068 | 33.083 | 240.5 | 16.2 | 1 | 1 | 8/8 x3 | `0989a10c5d842d8b` |
| 2 | 8/8 x3 | 0/0 | 2.248 | 35.967 | 332.8 | 36.8 | 2 | 2 | 8/8 x3 | `0989a10c5d842d8b` |
| 4 | 8/8 x3 | 0/0 | 2.243 | 35.890 | 481.1 | 85.1 | 4 | 2 | 8/8 x3 | `0989a10c5d842d8b` |
| 8 | 8/8 x3 | 0/0 | 2.290 | 36.635 | 985.9 | 163.1 | 8 | 4 | 8/8 x3 | `0989a10c5d842d8b` |

Interpretation: short-shape throughput improved at every concurrency point, while TTFT stayed queue-sensitive and moved a little in both directions. The trace fields still prove the scheduler did batch live decode rows (`decode_batch_size_max=4` at concurrency 8), so `--max-concurrency` is no longer being inferred as batch size from the client alone.

Mixed-shape proof: `prompt_words=16,128`, `num_requests=8` per repeat, four short and four long requests, `warmup=2`.

| conc | completed | failed/timeouts | QPS avg | output tok/s avg | TTFT avg ms | TPOT/ITL avg ms | active max | decode batch max | traces | output hash |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 4 | 8/8 x3 | 0/0 | 0.597 | 9.545 | 2756.2 | 260.2 | 4 | 2 | 8/8 x3 | `d53e286068a7cd5e` |
| 8 | 8/8 x3 | 0/0 | 0.602 | 9.634 | 5251.0 | 527.1 | 8 | 2 | 8/8 x3 | `d53e286068a7cd5e` |

Interpretation: the old long-prompt prefill failure is fixed for this HTTP contract, and the post-fastpath rerun lifts mixed 16/128 throughput a bit, but the row is still dominated by long-prompt prefill and admission queueing. The c4/c8 rows prove subgroup batching can happen with mixed prompt lengths (`decode_batch_size_max=2` here), yet the latency profile is not a production serving claim.

### Issue #453 No-Regression HTTP Benchmark

Retained 2026-07-04 no-regression benchmark for #453 used real `/v1/completions` traffic after the reliability gate. Both host-staged and NCCL completed every cell with `failed=0`, `timeouts=0`, and full trace coverage (`missing_traces=[]`). The short same-shape rows use `prompt_words=64`, `max_tokens=64`, and concurrency `1/4/8`; the mixed rows use `prompt_words=16,128`, `max_tokens=16`, and concurrency `4/8`.

| Backend | Shape | Completed | Failed/timeouts | Output tok/s | TTFT avg ms | TPOT/ITL avg ms | active max | decode batch max | traces | output hash |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| host-staged | short `prompt_words=64/max_tokens=64`, c1 | 8/8 | 0/0 | 22.886 | 1226.8 | 24.9 | 1 | 1 | 8/8 | `fed09e819c83762a` |
| host-staged | short `prompt_words=64/max_tokens=64`, c4 | 8/8 | 0/0 | 21.698 | 2612.8 | 145.1 | 4 | 3 | 8/8 | `fed09e819c83762a` |
| host-staged | short `prompt_words=64/max_tokens=64`, c8 | 8/8 | 0/0 | 21.830 | 5439.8 | 284.2 | 8 | 5 | 8/8 | `fed09e819c83762a` |
| host-staged | mixed `16/128`, c4 | 8/8 | 0/0 | 8.281 | 4543.3 | 210.0 | 4 | 2 | 8/8 | `88d8d31234d66978` |
| host-staged | mixed `16/128`, c8 | 8/8 | 0/0 | 8.288 | 7897.1 | 496.9 | 8 | 3 | 8/8 | `88d8d31234d66978` |
| NCCL | short `prompt_words=64/max_tokens=64`, c1 | 8/8 | 0/0 | 24.873 | 1034.2 | 24.4 | 1 | 1 | 8/8 | `fed09e819c83762a` |
| NCCL | short `prompt_words=64/max_tokens=64`, c4 | 8/8 | 0/0 | 23.678 | 2210.0 | 135.8 | 4 | 3 | 8/8 | `fed09e819c83762a` |
| NCCL | short `prompt_words=64/max_tokens=64`, c8 | 8/8 | 0/0 | 23.872 | 4594.4 | 266.1 | 8 | 6 | 8/8 | `fed09e819c83762a` |
| NCCL | mixed `16/128`, c4 | 8/8 | 0/0 | 9.186 | 2216.6 | 314.9 | 4 | 1 | 8/8 | `88d8d31234d66978` |
| NCCL | mixed `16/128`, c8 | 8/8 | 0/0 | 9.269 | 6920.1 | 453.0 | 8 | 3 | 8/8 | `88d8d31234d66978` |

This table is a no-regression and trace-coverage record. It is not a throughput optimization claim.

### Issue #453 HTTP Reliability Gate

The #453 runner is `scripts/bench_dsv2lite_http_reliability.py`. It uses standard-library HTTP streaming against `/v1/completions`, parses `pegainfer_http_trace` from the server log, and writes one JSON artifact with per-scenario counts, output hashes, trace coverage, active/pending/decode maxima, terminal reasons, final healthy-baseline status, and clean follow-up results.

Scenarios covered by the runner:

- `cancel_disconnect`: closes a streaming request after the first token, closes another connection after early bytes, keeps a neighboring request completing, then sends a clean follow-up.
- `invalid_requests`: sends non-greedy, `logprobs`, empty-prompt, and over-context requests beside a valid neighbor, then sends a clean follow-up.
- `overload_active_cap`: sends more concurrent requests than the DSV2-Lite active cap (`DEFAULT_MAX_ACTIVE_REQUESTS=8`) and requires pending-queue trace evidence plus clean recovery.
- `mixed_short_long_with_failures`: mixes 16-word and 128-word prompts with a cancelled stream and a rejected request, then sends a clean follow-up.

Strict failure rules:

- missing terminal trace fails, except for HTTP/frontend guard rejections that never reach the DSV2-Lite scheduler;
- a clean follow-up failure fails;
- success-hash drift within the same request kind fails;
- unexplained timeout fails;
- missing expected terminal reasons (`completed`, `rejected`, `cancelled`, `disconnected`) fail;
- missing active-set pressure, pending-queue pressure, or decode-batch evidence fails in scenarios that require it.

Validation completed for the runner schema, runner false-positive guards, scheduler accounting, the shared token-sink cancel/disconnect contract, host-staged/NCCL E2E exactness, and live host-staged/NCCL HTTP reliability:

```bash
python3 scripts/bench_dsv2lite_http_reliability.py --dry-run --out <artifact>.json
python3 -m py_compile scripts/bench_dsv2lite_http_reliability.py tests/test_bench_dsv2lite_http_reliability.py
python3 -m unittest tests.test_bench_dsv2lite_http_reliability
cargo test --release -p pegainfer-engine --lib token_sink -- --nocapture
cargo test --release -p pegainfer-vllm-frontend --lib abort -- --nocapture
cargo test --release -p pegainfer-deepseek-v2-lite --features deepseek-v2-lite --lib scheduler -- --nocapture
PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite PEGAINFER_DSV2_LITE_EP_BACKEND=host-staged cargo test --release -p pegainfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture
PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite PEGAINFER_DSV2_LITE_EP_BACKEND=nccl cargo test --release -p pegainfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture
```

The local dry-run passed all four scenarios and emits deterministic JSON. `tests/test_bench_dsv2lite_http_reliability.py` passed and verifies that the runner fails when a client-observed disconnect lacks a matching scheduler terminal reason or when trace fields are missing. `cargo test --release -p pegainfer-engine --lib token_sink -- --nocapture` passed and verifies explicit cancel versus closed receiver behavior. `cargo test --release -p pegainfer-vllm-frontend --lib abort -- --nocapture` passed and verifies that frontend aborts drop late tokens and classify disconnect before the first client-visible token separately from cancel after token output. The DSV2-Lite scheduler lifecycle subset passed `23 passed; 0 failed`; host-staged and NCCL `e2e_ep2` each passed `1 passed; 0 failed`. The SM120 NCCL validation used an NCCL 2.30.7 runtime library.

Retained 2026-07-04 live HTTP reliability artifacts from real `/v1/completions` traffic:

| Backend | Artifact | SHA-256 | Result | Scenario coverage |
| --- | --- | --- | --- | --- |
| host-staged | `http_reliability_host_staged.json` | `832d65a8e8b2b3a6ad0100c4a35f38475f040d6ffc192ec38a3b7384167187a5` | passed | cancel/disconnect, invalid requests, overload active-cap, mixed short/long with failures, clean follow-up after every scenario |
| NCCL | `http_reliability_nccl.json` | `53bedd98f19c5241df588a1ade8756e84a5e8c99225a589c4ed303e90fba38fa` | passed | cancel/disconnect, invalid requests, overload active-cap, mixed short/long with failures, clean follow-up after every scenario |

Scenario summaries:

| Backend | Scenario | Counts | Terminal reasons | Trace maxima | Final baseline |
| --- | --- | --- | --- | --- | --- |
| host-staged | `cancel_disconnect` | completed `2`, cancelled `1`, disconnected `1`, failed/rejected/timeout `0` | cancelled `1`, disconnected `1`, completed_length `2` | active `2`, pending `0`, decode `1` | healthy |
| host-staged | `invalid_requests` | completed `2`, rejected `4`, failed/timeout `0` | rejected and completed_length observed | active `1`, pending `0`, decode `1` | healthy |
| host-staged | `overload_active_cap` | completed `13`, failed/rejected/timeout `0` | completed_length `13` | active `8`, pending `4`, decode `7` | healthy |
| host-staged | `mixed_short_long_with_failures` | completed `5`, cancelled `1`, rejected `1`, failed/timeout `0` | cancelled, rejected, completed_length observed | active `4`, pending `0`, decode `2` | healthy |
| NCCL | `cancel_disconnect` | completed `2`, cancelled `1`, disconnected `1`, failed/rejected/timeout `0` | cancelled `1`, disconnected `1`, completed_length `2` | active `2`, pending `0`, decode `1` | healthy |
| NCCL | `invalid_requests` | completed `2`, rejected `4`, failed/timeout `0` | rejected and completed_length observed | active `1`, pending `0`, decode `1` | healthy |
| NCCL | `overload_active_cap` | completed `13`, failed/rejected/timeout `0` | completed_length `13` | active `8`, pending `4`, decode `7` | healthy |
| NCCL | `mixed_short_long_with_failures` | completed `5`, cancelled `1`, rejected `1`, failed/timeout `0` | cancelled, rejected, completed_length observed | active `5`, pending `0`, decode `2` | healthy |

### Interpretation

- older direct same-prompt snapshots showed NCCL behind host-staged; the #464 paired runs supersede that conclusion for the retained `Hello/16`, batch `1/4/8` shape only;
- the #280 HTTP trace proves active request sets and subgroup decode batches, but throughput still scales only weakly on NCCL EP2 and long prompts have high TTFT;
- the #464 trace refresh closes the phase/decode-step observability gap: batching forms, while decode mean/total grows with active rows, so scheduler admission is not the final throughput blocker;
- the #464 host-staged and NCCL route grouping reduces repeated expert replay; the NCCL device logits router also removes the CPU gate GEMM and full-hidden D2H copy;
- NCCL collectives are not the dominant measured limiter for the retained short shape; attention/KV host-side work requires a separate cache/ownership design;
- the #453 runner and trace fields make failure isolation auditable; host-staged and NCCL live HTTP artifacts now prove cancel/disconnect/reject/overload/mixed-failure cleanup and clean follow-up recovery on the retained 2-GPU validation contract;
- the 2026-06-28 clean matrix keeps stock vLLM startup failures visible because they are part of the reproducibility record;
- the supplemental vLLM validation shows the HTTP contract can run after the FlashInfer SM120/CUDA 12.8 path is fixed, but it should stay separate from stock-package rows;
- future performance claims should use the retained matrix contract, not older short-shape vLLM experiments.

## Claim Boundaries

Use these labels consistently:

| Label | Meaning | Do not infer |
| --- | --- | --- |
| `direct single-row` | In-process batch `1` decode. | HTTP serving throughput. |
| `direct same-prompt diagnostic batch` | Fixed same-prompt direct batch sizes `1/4/8`. | Production continuous batching or mixed-request scheduling. |
| `first mixed-request serving gate` | Greedy-only EP2 scheduler path with explicit admission/rejection/deferral, per-request host-side decode `DecodeCache`, active cap `8`, and exact sequential-oracle E2E. | vLLM parity, sparse dispatch, production EP readiness, HTTP throughput scaling, non-greedy sampling, or logprobs support. |
| `HTTP trace/subgroup evidence` | `/v1/completions` requests have per-request `pegainfer_http_trace` rows; HTTP sweeps show non-1 `active_set_size`, `decode_batch_size_max`, phase timing, and batched-vs-singleton decode-step counts. | Fair vLLM parity, long-prompt latency readiness, backend/kernel attribution, or a before/after percentage unless a paired baseline run is recorded. |
| `route-grouped MoE slice` | Separate paired host-staged and NCCL direct/HTTP A/B for the retained #464 shapes, with exact token/text hashes and contribution accumulation in original route order. | Fewer per-row GEMM launches, broad workload scaling, soak/SLO readiness, production EP readiness, or vLLM parity. |
| `NCCL device logits router slice` | The gate GEMM runs on CUDA with bitwise host-logit coverage; existing host top-k/softmax builds the route plan. | Fully device-resident routing, no routing D2H, general numerical equivalence, or non-NCCL improvement. |
| `HTTP reliability lifecycle gate` | `/v1/completions` cancel/disconnect/reject/overload/mixed-failure scenarios have terminal reason counts, trace coverage, active/pending/decode maxima, output hashes, and clean follow-up recovery evidence. | Production EP readiness, soak stability, SLO latency, vLLM parity, throughput improvement, sparse dispatch, or multi-node EP support. |
| `retained HTTP serving SLO` | Named short/mixed/long `/v1/completions` contracts report latency percentiles, throughput, outcomes, full trace coverage, hashes, and repeat spread for one backend/hardware/toolchain. | Direct attribution, sustained soak, production readiness, vLLM parity, or performance outside a matched contract. |
| `retained HTTP soak` | Fixed-shape `/v1/completions` contracts run inside time buckets and report declared-duration coverage, completion, failures/timeouts, trace coverage, output hashes, complete leaf artifacts, clean follow-up recovery, resource samples, and first/last-quartile drift. | Direct attribution, long/mixed-prompt readiness, production budgets, default-runtime speed, vLLM parity, or multi-node recovery. |
| `covered NCCL decode graph probe` | Probe-only batch-1 `Hello` decode step captured, instantiated, replayed, and token-verified under CUDA Graph. | Default serving graph coverage, multi-step graph replay, batch `4/8` graph coverage, or performance improvement. |
| `HTTP concurrency pressure` | `vllm bench serve --max-concurrency N` against an HTTP endpoint. | True PegaInfer batch size unless the engine path proves it. |
| `vLLM comparison from documented environment` | vLLM TP2 / TP2+EP2 from the retained matrix or the separate FlashInfer-fixed validation. | Stock vLLM install support, PegaInfer serving parity, or production readiness. |

Do not claim:

- production EP readiness;
- sparse dispatch readiness;
- multi-node EP support;
- vLLM serving parity;
- performance improvement outside a recorded paired benchmark contract.

## Next Gates

Issue #205 records the model roadmap. Maintainer feedback there calls out NCCL plus CUDA Graph as the likely best decode direction, with host staging possibly deprecated later. Treat that as a future direction, not as current evidence.

The graph-readiness diagnostic is fail-closed. `full_decode_capture_ready=true` is valid only when `full_decode_graph_probe` captures, instantiates, replays, and verifies the covered shape. The optional f32 NCCL graph smoke remains collective-only evidence. HF, host-staged, and NCCL still need token/text exactness for the committed case set.

The next implementation should be chosen from measured evidence:

1. Keep the widened HF / host-staged / NCCL case set current.
   - keep the committed cases and row-level comparison shape in sync with the accuracy docs;
   - treat the widened oracle as correctness evidence only, not serving evidence;
   - keep host-staged as the baseline oracle while it exists.

2. Decide whether to productize the probe-only CUDA Graph path.
   - keep HF / host-staged / NCCL exact before and after;
   - keep host-staged as the correctness baseline while it exists;
   - preserve attribution before and after the change;
   - keep the eager NCCL route-plan path as the serving oracle until the graph path is widened and measured;
   - keep the graph claim at batch `1`, `prompt="Hello"`, `output_len=16` until another probe covers more;
   - treat any future `failure_stage` as fail-closed evidence.

3. Keep a fair serving benchmark contract around future performance work.
   - PegaInfer host-staged.
   - PegaInfer NCCL.
   - vLLM TP2.
   - vLLM TP2+EP2 when supported.
   - default vLLM configuration plus a controlled configuration with cache/flag choices recorded.
   - keep host-staged and NCCL measurements separately attributable even when one PR changes both.

4. Widen the first mixed-request serving gate before broader throughput claims.
   - keep the fixed EP2 path and exact sequential oracle until a wider oracle replaces it;
   - keep greedy-only admission explicit until sampling/logprobs have their own gate;
   - keep direct same-prompt batch labeled diagnostic;
   - reduce long-prompt prefill and admission-queue TTFT before claiming long-prompt serving readiness;
   - require paired baseline runs for every future percentage speedup; #464 retains separate host-staged and NCCL paired results.

5. Keep the #453 HTTP reliability evidence retained.
   - rerun the reliability runner for host-staged and NCCL when scheduler or HTTP lifecycle code changes;
   - keep the JSON artifact hashes and server-log trace coverage in the PR evidence;
   - keep #452 long/mixed-prompt latency, the completed #465 soak baseline, the completed #466 SLO report layer, and #467 benchmark manifest separate.

6. Keep MoE internals readable.
   - routing, dispatch, expert execution, and combine should remain distinguishable in code and attribution;
   - avoid introducing a generic EP framework before the DeepSeek-V2-Lite EP2 path has a measured reason to need it.

7. Design the remaining attention optimization as a separate change.
   - preserve explicit KV/cache ownership and request retirement semantics;
   - prove long-context capacity before replacing the host path;
   - define CUDA Graph pointer-stability and replay behavior before enabling it;
   - keep exact HF / host-staged / NCCL gates and paired direct/HTTP evidence.
