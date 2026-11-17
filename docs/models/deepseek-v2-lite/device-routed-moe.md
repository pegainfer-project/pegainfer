# DeepSeek-V2-Lite Device-Routed MoE

> **TL;DR:** NCCL MoE forwards with at most eight rows keep routing, routed-expert projection, weighted reduction, and EP combine inputs on device. Each rank executes fixed route slots through pointer-batched `N=1` GEMMs; this removes the host route plan but is not an expert-sorted grouped GEMM.

Last touched: 2026-10

## Scope

The path covers NCCL MoE forwards with one to eight rows, including decode and short prefill. DeepSeek-V2-Lite has 64 routed experts with top-6 routing: rank 0 owns experts 0..31 and the model driver, while rank 1 owns experts 32..63 and replicated MoE gates. Larger forwards and the host-router or serial-expert rollback modes continue through the eager route-plan path.

## Design

The eager path copies router logits to the host, performs CPU softmax/top-k, builds a `MoeRoutePlan`, and replays many small expert calls. The new path keeps routing and expert execution on GPU:

```text
rank-local hidden
  -> router logits and top-6 on GPU
  -> route-slot pointer arrays on GPU
  -> pointer-batched gate/up
  -> SiLU x up
  -> pointer-batched down
  -> ordered rank-local f32 reduction
  -> NCCL f32 combine
  -> bf16 routed output on rank 0
```

Each token reserves six slots on each rank. Owned routes point to the selected expert; non-owned routes use a zero input and are discarded during reduction. This fixed layout avoids host coordination and dynamic packing. At eight rows, each projection submits 48 independent `N=1` problems through one `cublasGemmBatchedEx` call per rank. It is pointer-batched execution, not an expert-grouped GEMM: tokens selecting the same expert are not merged into a larger matrix.

The shared expert remains on rank 0 and is added after the routed result is combined.

## Runtime State

`DeviceRoutedMoeRuntime` owns per-layer expert pointer tables and reusable capacity-eight scratch for both ranks. The pointer tables reference model weights, so the runtime is destroyed before the two rank models. Mutex guards serialize scratch reuse and protect the NCCL send buffers while both rank-local reductions write them.

`begin_forward` clears route summaries once before the MoE layers; `finish_forward` reads them once afterward. The summaries check nonfinite values, route totals, ownership counts, and matching per-layer route hashes across ranks without adding a host synchronization to every layer. For finite logits, the private router producer constructs six in-range, distinct IDs in ascending order; the parity fixtures check that invariant before the consumer relies on it.

## Numerical Contract And Fallback

GPU routing preserves the host selection order: stable scan over 64 logits, softmax, top-6, then ascending expert ID for ordered f32 reduction. Batched cuBLAS may produce low-bit differences from the serial GEMM loop, so acceptance is based on identical route IDs and model output accuracy rather than intermediate bitwise equality.

Host-staged execution remains the correctness oracle. More than eight rows, host-router mode, and serial-expert mode automatically use the eager route-plan path.

## Correctness Validation

Validation was run on two A100 40 GB GPUs with release builds.

| Check | Result |
| --- | --- |
| DeepSeek-V2-Lite library tests | Passed |
| Router randomized / tie / near-tie fixtures | Exact route IDs and ordering at rows 1 / 4 / 8 |
| Compute Sanitizer memcheck | Passed with `0 errors` |
| HF / host-staged / NCCL case set | Token- and text-exact |
| Mixed-request NCCL E2E | Passed |
| Host-router + serial-expert fallback | Passed |
| HTTP lifecycle scenarios | Passed with healthy follow-up requests |

The direct A/B runs also produced matching token hashes. Batched cuBLAS projections are not bitwise-equal to the serial intermediate tensors, but the retained route checks and generated outputs passed.

## Performance Validation

The baseline uses GPU router logits followed by CPU softmax/top-k and a host route plan. Baseline and device-routed binaries were alternated on two A100 40 GB GPUs in the same environment. Each cell below is the median of three per-run decode-step means.

Measurements used `dsv2_lite_ep2_decode_attribution` with the same `Hello` prompt in each batch row, 16 output tokens, attribution enabled, and CUDA Graph disabled. See [Verification And Benchmarking](benchmarking.md#direct-diagnostic-benchmark) for the run command.

| Decode batch | Baseline | Device-routed | Improvement |
| ---: | ---: | ---: | ---: |
| 1 | 41.896 ms | 27.499 ms | 34.4% |
| 4 | 83.433 ms | 50.162 ms | 39.9% |
| 8 | 136.096 ms | 77.736 ms | 42.9% |

This path covers the current decode scheduler limit of eight live requests and short prefill with at most eight rows. The performance numbers above measure decode only. Any forward with more than eight rows continues through the eager route-plan path.

One same-binary HTTP A/B used 32 requests with exactly 64 input and 64 output tokens at each client concurrency. The baseline enabled the host-router rollback (`PEGAINFER_DSV2_LITE_NCCL_ROUTER=host`); the device-routed run removed only that setting. Each cell was measured once without warmup.

| Client concurrency | Baseline TPOT p50 | Device-routed TPOT p50 | TPOT change | Output tok/s change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 57.128 ms | 49.697 ms | -13.0% | +19.7% |
| 4 | 235.396 ms | 207.178 ms | -12.0% | +18.8% |
| 8 | 448.599 ms | 362.379 ms | -19.2% | +26.8% |

All HTTP cells completed without failures or timeouts, had complete server-trace coverage, and produced matching output hashes.

## Follow-Up Work

1. Split timing attribution into router, pointer construction, expert projection, reduction, and NCCL combine.
2. Treat larger decode batches and prefill separately. If they become performance targets, measure expert occupancy before choosing between wider fixed slots and a true expert-grouped GEMM.
