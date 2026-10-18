# GPU Profiling Guide

> **TL;DR:** For benchmarking, drive the HTTP server with `scripts/bench_http_serving.py` or vllm-bench (the in-process `bench_serving` bin is retired). For profiling, wrap the *server* in `nsys`, drive load over HTTP, analyze totals with `nsys stats`, and always inspect `std/p95/p99/max` tails before making optimization decisions. This document covers pitfalls and diagnostic paths, not CLI reference.
>
> **Status:** Active.

## Prerequisites

- `nsys`: ships with the CUDA Toolkit, or install the standalone [Nsight Systems CLI](https://developer.nvidia.com/nsight-systems). Verify: `nsys --version`.
- **Must use `--release`.** Debug builds slow GPU kernels by 10x+ due to debug info. Traces from debug builds do not reflect real behavior.

## Capturing a Trace

Wrap the server process in nsys, then drive load over HTTP from a second shell:

```bash
# Shell 1: server under nsys, export sqlite directly
nsys profile --force-overwrite=true --cuda-graph-trace=node \
  --export=sqlite -o target/profiling/trace \
  cargo run -r -- --model-path models/Qwen3-4B --port 8000

# Shell 2: drive the load, then Ctrl-C the server to finalize the trace
python3 scripts/bench_http_serving.py --base-url http://localhost:8000 \
  --model models/Qwen3-4B --num-requests 8 --concurrency 2 --max-tokens 256
```

Produces `target/profiling/trace.nsys-rep` + `target/profiling/trace.sqlite`. The sqlite file can be analyzed directly with `nsys stats` — no secondary conversion needed.

### Key Flags

**`--cuda-graph-trace=node`** — The single most important pitfall. Without this, CUDA Graph replay appears as an opaque block in the timeline — you cannot see which kernels run inside the graph. With it, every kernel inside replay is expanded. The tradeoff is overhead: absolute times in the trace are inflated. Use it only for composition and proportions, not for benchmarking.

**`--force-overwrite=true`** — Without this, nsys refuses to overwrite an existing output file and exits with an error.

### Other Useful nsys Options

```bash
# Capture only CUDA activity, skip OS scheduling noise (smaller trace, faster to open)
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node ...

# Delayed start (skip model loading, capture only inference)
nsys profile --delay=10 --duration=30 --cuda-graph-trace=node ...
```

## Analyzing a Trace

Use `nsys stats` to produce reports (reads sqlite directly, no conversion):

```bash
# Kernel time summary — the first report to look at
nsys stats --report cuda_gpu_kern_sum target/profiling/trace.sqlite

# CUDA API call summary — find sync / memcpy overhead
nsys stats --report cuda_api_sum target/profiling/trace.sqlite
```

`cuda_gpu_kern_sum` example output (Qwen3-4B, prompt=1, output=10, 2026-03-13):

```
 Time (%)  Total Time (ns)  Instances  Avg (ns)    Name
     41.4   86,781,589          720    120,530   fused_mlp_intermediate_kernel  ← MLP top half: gate+up, largest share
     32.7   68,628,022        2,900     23,665   gemv_handwritten_kernel        ← all projection GEMV
     21.2   44,380,185          720     61,639   fused_mlp_output_kernel        ← MLP bottom half: down projection
      2.2    4,595,007        1,440      3,191   fused_add_rms_norm_kernel      ← residual + norm, lightweight
      2.0    4,128,672          721      5,726   fused_attention_decode_kernel   ← decode attention
      0.3      549,600          721        762   attention_reduce_kernel         ← split-KV reduce
```

Normal pattern: MLP (intermediate + output) + GEMV account for >90%. Attention share is small at short context and grows with sequence length.

`cuda_api_sum` example output:

```
 Time (%)  Total Time (ns)  Num Calls   Name
     67.4  661,216,067          421    cuMemcpyHtoDAsync_v2      ← model weight loading, one-time at startup
     20.3  199,588,713           22    cuStreamSynchronize       ← host waiting for GPU
      5.9   58,181,025          518    cuMemAllocAsync           ← VRAM allocation
      1.2   11,474,056          310    cudaLaunchKernel          ← kernel launch overhead
      0.8    8,100,427           20    cuGraphLaunch             ← CUDA Graph replay
```

Normal pattern: `cuMemcpyHtoDAsync` is dominated by model loading (one-time). During inference, watch `cuStreamSynchronize` — if it far exceeds kernel launch + graph launch, the host is waiting for the GPU unnecessarily.

## Tail-Aware Kernel Stats

`nsys stats` is good for total time and average duration, but it can hide the most important decode regressions. For serving work, do not judge a kernel by p50 alone: collect `count`, `total`, `avg`, `std`, `p50`, `p95`, `p99`, and `max`.

The common failure mode is a kernel with a healthy p50 but a large p99/max. That usually points to route imbalance, rank arrival skew, allocator churn, stream drains, or rare launch/runtime stalls. In MoE decode, NCCL all-reduce and routed expert kernels are especially likely to look cheap at p50 while dominating p99.

Use the checked-in helper `tools/nsys_tail_stats.py` on exported sqlite traces when tail behavior matters. It emits separate kernel and CUDA API tables sorted by total time and by tail score, so a trace review can call out both steady-state cost and rare stalls.

For keep/revert decisions, a profile that only reports p50 or avg is incomplete. The minimum acceptable evidence is the tail table plus a clear statement of whether the profile includes prompt, steady decode, HTTP/frontend, or profiler overhead.

## Standard Optimization Profiles

Two load shapes isolate prefill and decode paths for per-model optimization
(server running; `--prompt-words` controls prompt length approximately):

```bash
# Prefill-heavy: TTFT dominates, decode negligible
python3 scripts/bench_http_serving.py --base-url http://localhost:8000 \
  --model models/Qwen3-4B --prompt-words 2048 --max-tokens 1

# Decode-heavy: TPOT dominates, prefill negligible
python3 scripts/bench_http_serving.py --base-url http://localhost:8000 \
  --model models/Qwen3-4B --prompt-words 1 --max-tokens 128
```

## Diagnosing Decode Degradation With Sequence Length

If TPOT degrades significantly as context grows (sweep `--prompt-words` while holding `--max-tokens` fixed), use comparative traces to pinpoint the offending kernel: capture two server traces as above (`-o target/profiling/ctx_short` / `ctx_long`), driving one with a short prompt and one with a long prompt while output tokens stay fixed:

```bash
# Against the ctx_short server run
python3 scripts/bench_http_serving.py --base-url http://localhost:8000 \
  --model models/Qwen3-4B --prompt-words 1 --max-tokens 128

# Against the ctx_long server run
python3 scripts/bench_http_serving.py --base-url http://localhost:8000 \
  --model models/Qwen3-4B --prompt-words 2048 --max-tokens 128
```

Compare the two `cuda_gpu_kern_sum` reports — the kernel with the largest avg time increase is the culprit.

Measured comparison (Qwen3-4B, 2026-03-13):

```
kernel                          prompt=1 avg    prompt=2048 avg    change
fused_attention_decode_kernel      6.1μs           41.5μs          6.8x  ← only kernel that scales with context
fused_mlp_intermediate_kernel    120.5μs          120.5μs          1.0x
gemv_handwritten_kernel           23.7μs           23.8μs          1.0x
fused_mlp_output_kernel           61.7μs           61.8μs          1.0x
```

MLP and GEMV are completely flat. Attention decode is O(seq_len). If other kernels also show growth, there is an unexpected context dependency to investigate.

## CUDA Toolkit Version Impact

nvcc version matters. Measured on RTX 5070 Ti (sm_120) with Qwen3.5-4B, same code, same GPU, only nvcc changed (2026-03-28):

| | CUDA 13.1 (V13.1.115) | CUDA 12.8 (V12.8.93) | regression |
|--|------------------------|----------------------|------------|
| Decode TPOT (1,128) | 11.81ms | 12.18ms | **+3.1%** |
| Prefill TTFT (2048,1) | 224.5ms | 229.4ms | **+2.2%** |

Decode is hit harder because the hotpath is custom CUDA kernels (GDR decode, handwritten GEMV, fused MLP) where nvcc codegen quality directly matters. Prefill is less affected because the heavy compute is Triton AOT and cuBLAS — nvcc only compiles auxiliary kernels.

**Practical rule:** always build with the latest CUDA toolkit that supports your SM target. For sm_120 (Blackwell), CUDA 13.1 is the native toolkit; 12.8 can target it but generates worse code.
