---
name: nsys-profiling
description: "Profile GPU workloads with NVIDIA Nsight Systems (nsys). Use this skill when the user says 'profile', 'nsys', 'capture a trace', 'GPU profiling', 'kernel breakdown', 'where is time spent', 'decode is slow', 'prefill is slow', 'TPOT regression', 'TTFT regression', or wants to understand where GPU time is going. Also trigger when the user pastes nsys output and asks for help interpreting it. This skill covers the full lifecycle: capturing traces with the right flags, analyzing kernel/API summaries, detecting tail latency problems, and diagnosing performance regressions."
---

# nsys Profiling

You are helping a developer profile GPU workloads in a Rust + CUDA inference engine using NVIDIA Nsight Systems (`nsys`). The goal is always to answer "where is time going and what should I optimize next?"

## Before you start

1. **Confirm the workload.** Ask which model, which path (prefill vs decode vs both), and what prompt/output shapes matter. If the user is vague, default to the two standard profiles:
   - Prefill-heavy: `--prompt-len 2048 --output-len 1`
   - Decode-heavy: `--prompt-len 1 --output-len 128`

2. **Check `--release`.** Debug builds slow GPU kernels by 10x+ and produce misleading traces. If you see `cargo run` without `-r` or `--release`, stop and fix this first.

3. **Check nsys is available.** Run `nsys --version`. If it fails, the user likely needs to add `/usr/local/cuda/bin` to PATH — a common gotcha in SSH or tmux sessions where `~/.bashrc` doesn't source. Fix: `export PATH="/usr/local/cuda/bin:$PATH"`, or use full path `/usr/local/cuda/bin/nsys`.

4. **Check profiling environment.** Run `nsys status -e` to verify the system can profile. It checks kernel paranoid level, perf_event support, and LBR availability. If paranoid level is too high (>1), CPU sampling won't work without root.

## Capturing a trace

### Standard capture command

```bash
nsys profile --force-overwrite=true --cuda-graph-trace=node \
  --export=sqlite -o target/profiling/<trace_name> \
  cargo run -r --bin bench_serving -- request \
    --prompt-len <PROMPT> --output-len <OUTPUT> --warmup 1 --iters 1
```

Build this command for the user based on their workload. Always include all four flags:

| Flag | Why |
|------|-----|
| `--force-overwrite=true` | nsys errors on existing files without this |
| `--cuda-graph-trace=node` | Without it, CUDA Graph replay is opaque — you can't see individual kernels. This is the single most common nsys pitfall. Default is `graph` on driver ≥11.7, which hides everything inside the graph. |
| `--export=sqlite` | Produces a `.sqlite` file that `nsys stats` reads directly |
| `-o target/profiling/<name>` | Keep traces organized in one place |

### Precise capture with cudaProfilerApi

If the `bench_serving` binary supports `--cuda-profiler-capture`, use it to capture only the measured iterations — this excludes model loading and warmup from the trace, producing a much cleaner profile:

```bash
nsys profile --force-overwrite=true --cuda-graph-trace=node \
  -c cudaProfilerApi --export=sqlite -o target/profiling/<trace_name> \
  cargo run -r --bin bench_serving -- --cuda-profiler-capture request \
    --prompt-len <PROMPT> --output-len <OUTPUT> --warmup 1 --iters 1
```

Use `--capture-range-end=repeat[:N]` to capture multiple separate ranges (e.g., N iterations) into the same trace. Each range produces its own segment.

### Delay/duration capture (when cudaProfilerApi isn't available)

For workloads without profiler API integration, use `--delay` to skip startup and `--duration` to limit capture:

```bash
nsys profile --delay=10 --duration=30 --cuda-graph-trace=node \
  --force-overwrite=true --export=sqlite -o target/profiling/<trace_name> \
  <command>
```

### Disabling CUDA Graph for cleaner kernel analysis

CUDA Graph replay groups many kernels into a single opaque launch. Even with `--cuda-graph-trace=node`, profiler overhead inflates times differently than ungraphed launches. For precise per-kernel timing (e.g., comparing short vs long context), disable CUDA Graph:

```bash
cargo run -r --bin bench_serving -- --cuda-graph false request --prompt-len 2048 --output-len 32 ...
```

Tradeoff: ungraphed traces show true individual kernel times but include ~5μs CPU launch overhead per kernel that the production (graphed) path avoids. Use graphed traces for real TPOT, ungraphed for kernel-level comparison.

### Interactive profiling (long-running servers)

For profiling a server that's already running, use the launch/start/stop workflow:

```bash
# Terminal 1: launch the server under nsys control, but don't start collecting yet
nsys launch --trace=cuda,nvtx --cuda-graph-trace=node \
  --session-new=my_session -- cargo run -r --bin pegainfer-server -- ...

# Terminal 2: start/stop collection on demand
nsys start --session=my_session
# ... send requests ...
nsys stop --session=my_session
```

Or use `--start-later=true` with `nsys profile` to get the same deferred-start behavior in a single command.

### Lighter traces

If the trace is too large or slow to process, limit tracing scope:

```bash
# Only CUDA + NVTX, skip OS scheduling noise — smaller trace, faster to open
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node ...
```

Other `--trace` options: `cublas` (trace cuBLAS API calls with parameters), `cudnn`, `osrt` (OS runtime like pthread), `mpi`, `python-gil`. Multiple values comma-separated.

### Output naming tricks

The `-o` flag supports pattern substitution:
- `%h` — hostname (useful for multi-node)
- `%p` — PID of target process
- `%n` — auto-incrementing number (avoids overwrite without `--force-overwrite`)
- `%q{ENV_VAR}` — environment variable expansion
- `%%` — literal `%`

Example: `-o traces/%h_rank%q{RANK}_%n` → `traces/gpu-node-01_rank0_1.nsys-rep`

### Reproducible configs with `--command-file`

Store nsys flags in a file to avoid long command lines:

```bash
# nsys_decode.cfg
--force-overwrite=true
--cuda-graph-trace=node
--export=sqlite
--trace=cuda,nvtx
```

Then: `nsys profile --command-file=nsys_decode.cfg -o target/profiling/trace <command>`

Command-line flags override file flags.

### One-shot stats with `--stats=true`

Skip the separate `nsys stats` step entirely:

```bash
nsys profile --stats=true --force-overwrite=true --cuda-graph-trace=node \
  --export=sqlite -o target/profiling/trace <command>
```

This automatically generates all default summary reports (kern_sum, api_sum, mem_sum, etc.) to the console after collection finishes. Convenient for quick checks, but the output is long — better for scripted workflows.

## Analyzing a trace

Run these analyses in sequence. Each answers a different question.

### Step 1: Kernel time breakdown

```bash
nsys stats --report cuda_gpu_kern_sum target/profiling/<trace>.sqlite
```

This tells you **which GPU kernels consume the most time**. Normal patterns:
- MLP + GEMV typically >90% of decode time
- Attention grows with sequence length (O(seq_len)) while MLP/GEMV stay flat
- RMSNorm and activation kernels are lightweight (<3%)

**Use `:base` to tame mangled names.** By default, kernel names are fully mangled C++ templates — unreadable. Append `:base` to strip template parameters:

```bash
nsys stats --report cuda_gpu_kern_sum:base target/profiling/<trace>.sqlite
```

This turns `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, ...>(T13)` into just `kernel`. Much easier to scan, but note that it merges kernels with different template instantiations. Use the full name when you need to distinguish shapes.

There's also `:mangled` if you need the raw linker symbol.

### Step 2: CUDA API overhead

```bash
nsys stats --report cuda_api_sum target/profiling/<trace>.sqlite
```

This tells you **where the CPU/host is spending time**. Watch for:
- `cuMemcpyHtoDAsync` dominated by model loading (normal if one-time)
- `cuStreamSynchronize` — if this is large during inference (not loading), the host is unnecessarily waiting for the GPU
- `cuMemAllocAsync` / `cuMemFreeAsync` — high counts or time means hot-path allocation churn
- `cudaLaunchKernel` — large count × ~5μs each = significant CPU launch overhead

### Step 3: Tail-aware stats

```bash
python3 tools/nsys_tail_stats.py target/profiling/<trace>.sqlite --limit 15
```

This is the team's custom tool that goes beyond nsys's built-in p50/avg reporting. It shows `p50, p95, p99, max` and a tail score for each kernel. The p50 can hide serious problems:

- **max/p50 > 2×**: Kernel has outlier invocations. Common causes: route imbalance in MoE, rank arrival skew in NCCL, allocator stalls.
- **p99/p50 > 1.5×**: Systematic tail — not a one-off spike. Look for stream contention or variable work per invocation.
- **High tail score but low total time**: The kernel is cheap on average but occasionally blocks everything. These are the sneakiest bottlenecks.

### Step 4: Bandwidth efficiency (for GEMV-bound workloads)

When GEMV dominates (>90% of GPU time, typical for bs=1 decode), compute memory bandwidth utilization to answer "is there room to optimize, or are we already at the hardware limit?"

```
BW_achieved = weight_bytes / kernel_time
BW_efficiency = BW_achieved / GPU_peak_BW
```

For each major GEMV shape, calculate weight size (M × K × sizeof(bf16)) and divide by the nsys avg kernel time. Compare against GPU peak memory bandwidth (e.g., RTX 5070 Ti = 896 GB/s, H100 = 3.35 TB/s). Large GEMVs (>20MB weights) typically reach 85-90% efficiency; small GEMVs (<5MB) drop to 60-70% due to launch overhead and tail effects.

If overall efficiency is >80%, GEMV optimization has limited headroom — quantization or batching is the path forward. If <70%, there's room for custom kernels.

### Step 5 (optional): Auto-analysis

```bash
nsys analyze target/profiling/<trace>.sqlite
```

Runs NVIDIA's built-in rules: `cuda_memcpy_async`, `cuda_memcpy_sync`, `cuda_memset_sync`, `cuda_api_sync`, `gpu_gaps`, `gpu_time_util`. It flags synchronous memcpy, unnecessary device syncs, and periods where the GPU is idle >500ms. Quick sanity check, but won't catch domain-specific problems.

Run a specific rule: `nsys analyze --rule cuda_api_sync <trace>.sqlite`

## Advanced analysis techniques

### Time-window filtering

Isolate just the inference phase by filtering out model loading:

```bash
# Skip first 5 seconds (model loading), analyze everything after
nsys stats --report cuda_gpu_kern_sum --filter-time "5s/" target/profiling/<trace>.sqlite

# Analyze only a specific 2-second window
nsys stats --report cuda_gpu_kern_sum --filter-time "5s/7s" target/profiling/<trace>.sqlite
```

Time units are composable: `1s2ms3us4ns` = 1,002,003,004ns. This is much more precise than `--delay`/`--duration` — it's post-hoc filtering on an already-captured trace.

You can also filter by NVTX range if the code has NVTX annotations:
```bash
nsys stats --report cuda_gpu_kern_sum --filter-nvtx "decode_step@inference" target/profiling/<trace>.sqlite
```

### Change time units

Default output is nanoseconds. Switch to milliseconds for readability:

```bash
nsys stats --report cuda_gpu_kern_sum --timeunit msec target/profiling/<trace>.sqlite
```

### Machine-readable output

For scripted analysis or piping into other tools:

```bash
# JSON — directly parseable by Python/jq
nsys stats --report cuda_gpu_kern_sum --format json target/profiling/<trace>.sqlite

# CSV — for spreadsheets or pandas
nsys stats --report cuda_gpu_kern_sum --format csv --output . target/profiling/<trace>.sqlite

# Pipe to grep for quick filtering
nsys stats --report cuda_api_sum --format table --output @"grep -E '(-|Name|cudaFree)'" target/profiling/<trace>.sqlite
```

### Launch-to-execution correlation

```bash
nsys stats --report cuda_kern_exec_sum target/profiling/<trace>.sqlite
```

This report shows three time columns per kernel:
- **AAvg** (API time) — how long the host-side launch API took
- **QAvg** (queue time) — how long the kernel sat in the stream queue before the GPU picked it up
- **KAvg** (kernel time) — actual GPU execution time

If QAvg is large compared to KAvg, kernels are queuing because the GPU is still busy with previous work, or the launch stream is congested.

### Per-launch GPU trace

```bash
nsys stats --report cuda_gpu_trace target/profiling/<trace>.sqlite
```

Shows every individual kernel launch with its timestamp, grid dimensions (`GrdX/Y/Z`), block dimensions (`BlkX/Y/Z`), register count (`Reg/Trd`), and shared memory usage. Useful for:
- Verifying launch configurations match what you expect
- Finding specific outlier launches (sort by Duration)
- Checking if grid dimensions vary across iterations (sign of dynamic shapes)

### Multi-report in one command

Generate multiple reports in a single pass:

```bash
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --report cuda_gpu_mem_time_sum \
  target/profiling/<trace>.sqlite
```

### GPU memory tracking

Capture GPU memory allocation/deallocation patterns over time:

```bash
nsys profile --cuda-memory-usage=true --force-overwrite=true --cuda-graph-trace=node \
  --export=sqlite -o target/profiling/trace <command>
```

Adds overhead but shows memory watermarks. Useful when debugging OOM or fragmentation.

### CUDA backtrace for slow API calls

Find which host code is issuing slow CUDA API calls:

```bash
nsys profile --cudabacktrace=kernel:500 --force-overwrite=true --cuda-graph-trace=node \
  --export=sqlite -o target/profiling/trace <command>
```

Collects CPU backtraces for CUDA kernel launches that take >500ns (the threshold). The backtrace tells you exactly which Rust/C++ function triggered the launch. Options: `all`, `kernel`, `memory`, `sync`, `other`, or comma-combined. Significant overhead — use selectively.

### GPU hardware metrics

```bash
nsys profile --gpu-metrics-devices=all --gpu-metrics-frequency=10000 \
  --force-overwrite=true --export=sqlite -o target/profiling/trace <command>
```

Samples SM utilization, memory throughput, and other hardware counters at 10kHz. Check available metric sets: `nsys profile --gpu-metrics-set=help`.

**Caveat**: on consumer GPUs (GeForce), this requires setting GPU performance counter access. If you see `ERR_NVGPUCTRPERM`, run `nsys status -e` and check NVIDIA's permission guide.

### Experimental plugins

```bash
# Power and temperature monitoring during profiling
nsys profile --enable=nvml_metrics --force-overwrite=true --export=sqlite -o target/profiling/trace <command>

# Network adapter metrics
nsys profile --enable=network_interface --force-overwrite=true --export=sqlite -o target/profiling/trace <command>
```

List all plugins: `nsys profile --enable=help`

### Export to other formats

```bash
# Parquet — for pandas/Spark analysis on large traces
nsys export --type=parquetdir -o target/profiling/trace_parquet target/profiling/trace.nsys-rep

# Arrow — for Apache Arrow consumers
nsys export --type=arrowdir -o target/profiling/trace_arrow target/profiling/trace.nsys-rep

# Export only specific tables (faster for large traces)
nsys export --type=sqlite --tables=CUPTI_ACTIVITY_KIND_KERNEL,StringIds \
  -o target/profiling/kernels_only target/profiling/trace.nsys-rep

# Time-filtered export — extract just the interesting window
nsys export --type=sqlite --times=5s/10s \
  -o target/profiling/inference_only target/profiling/trace.nsys-rep
```

### Recipes (multi-file analysis)

`nsys recipe` runs higher-level analyses across one or more trace files. Requires `pandas` — install with:
```bash
python3 /opt/nvidia/nsight-systems/*/target-linux-x64/python/packages/nsys_recipe/install.py
```

Useful recipes for this project:

| Recipe | What it does |
|--------|-------------|
| `diff` | Side-by-side comparison of two traces — kernel time, API time, memory ops |
| `nccl_sum` | NCCL collective summary (multi-GPU/multi-node) |
| `nccl_gpu_overlap_trace` | NCCL/compute overlap analysis — are collectives hiding behind compute? |
| `cuda_gpu_kern_pace` | Kernel pacing — are kernels launching at a steady rate or bursty? |
| `cuda_gpu_kern_hist` | Duration histogram per kernel — visualize the distribution, not just percentiles |
| `cuda_gpu_time_util_map` | GPU utilization heatmap over time |

Example:
```bash
# Diff two traces to see what changed
nsys recipe diff -i trace_before.sqlite trace_after.sqlite

# NCCL analysis for distributed profiling
nsys recipe nccl_sum -i rank0.sqlite rank1.sqlite rank2.sqlite rank3.sqlite
```

## Available report types (quick reference)

| Report | Purpose |
|--------|---------|
| `cuda_gpu_kern_sum[:base\|:mangled]` | Kernel time totals — first thing to look at |
| `cuda_api_sum` | Host-side CUDA API time |
| `cuda_gpu_trace` | Per-launch kernel trace with grid/block dims and registers |
| `cuda_kern_exec_sum` | Launch → queue → execute timing per kernel |
| `cuda_gpu_mem_size_sum` | Memory ops grouped by transfer size |
| `cuda_gpu_mem_time_sum` | Memory ops grouped by duration |
| `cuda_api_gpu_sum` | Combined API + kernel + memops view |
| `nvtx_sum` / `nvtx_pushpop_sum` | NVTX range summaries (if code has annotations) |
| `nvtx_kern_sum` | Kernels grouped by NVTX range |
| `osrt_sum` | OS runtime API summary (pthread, file I/O) |

Full list: `nsys stats --help-reports`

## Interpreting results

### What to report to the user

After analysis, present:

1. **Top 3-5 kernels by total time** — with percentage of total GPU time
2. **Any tail anomalies** — kernels where max/p50 > 2×
3. **Host overhead** — is `cuStreamSynchronize` or `cudaLaunchKernel` significant?
4. **Diagnosis** — which kernel family is the optimization target?
5. **Next action** — what to microbench or optimize

### Mapping kernel names to model operations

nsys kernel names are mangled C++ symbols. Map them to logical operations:
- `gemvx::kernel` → cuBLAS GEMV (matrix-vector multiply for all linear projections)
- `FusedAddRMSNormKernel` → residual connection + RMS normalization
- `BatchDecodeWithPagedKVCacheKernel` → FlashInfer paged attention decode
- `BatchPrefillWithPagedKVCacheKernel` → FlashInfer paged attention prefill
- `AppendPagedKVCacheKernel` → KV cache page append
- `RadixTopKKernel_Unified` → FlashInfer sampling (token selection)
- `silu_mul_fused_kernel` → SiLU activation × gate (MLP)
- `prefill_qk_norm_rope_kernel` → QK normalization + rotary position embedding
- `embedding_batched_kernel` → token embedding lookup

Use `:base` to get these short names directly: `nsys stats --report cuda_gpu_kern_sum:base`.

### Diagnosing sequence-length degradation

If TPOT grows with context length, run two traces with different prompt lengths:

```bash
# Short context
nsys profile ... -o target/profiling/ctx_short \
  cargo run -r --bin bench_serving -- request --prompt-len 1 --output-len 128 --warmup 1 --iters 1

# Long context
nsys profile ... -o target/profiling/ctx_long \
  cargo run -r --bin bench_serving -- request --prompt-len 2048 --output-len 128 --warmup 1 --iters 1
```

Compare `cuda_gpu_kern_sum` between the two. Only attention decode should scale with context — if other kernels also grow, investigate.

If `nsys recipe diff` is available (pandas installed), use it for an automated side-by-side:
```bash
nsys recipe diff -i target/profiling/ctx_short.sqlite target/profiling/ctx_long.sqlite
```

## Critical pitfalls (team knowledge)

These are real lessons from production profiling on this codebase:

1. **`--cuda-graph-trace=node` inflates absolute times by 30-60%.** Always measure actual TPOT with `bench_serving` *without* nsys, and use nsys only for kernel time *proportions* and *composition*. Never quote a TPOT number from an nsys trace as ground truth.

2. **Default `--cuda-graph-trace` is `graph`, not `node`.** On driver ≥11.7, nsys defaults to `graph` granularity — the entire CUDA Graph replay appears as a single opaque block. You MUST explicitly pass `--cuda-graph-trace=node` to see individual kernels. This is the #1 "my trace is useless" moment.

3. **p50 is not enough.** A kernel can look cheap at p50 while dominating p99. In MoE workloads, NCCL all-reduce and routed expert kernels are especially prone to this — the p50 reflects pure compute but the p99 includes rank arrival skew. Always run `nsys_tail_stats.py` when tail behavior matters.

4. **CPU launch overhead is multiplicative.** Each `cudaLaunchKernel` costs ~5μs on the host. A per-token loop launching 20,000 tiny kernels adds ~100ms of pure CPU overhead per decode step. If `cudaLaunchKernel` count is surprisingly high, look for kernel fusion or CUDA Graph opportunities. Use `cuda_kern_exec_sum` to see queue time vs execution time.

5. **`cuStreamSynchronize` during inference is a red flag.** During model loading it's normal. During decode it means the host is blocking — look for unnecessary D2H copies or sync points. Use `--filter-time` to isolate inference-only API stats.

6. **NCCL wait-inclusive time ≠ pure transfer time.** In distributed traces, if NCCL collectives look expensive, separate rank arrival skew from actual data movement. Two ranks may do the same collective but one "takes 1.2ms" only because the other arrived 1.18ms late. Use `nsys recipe nccl_gpu_overlap_trace` to analyze this.

7. **CUDA toolkit version affects kernel performance.** Same code, same GPU: CUDA 13.1 generates ~3% faster code than 12.8 for custom CUDA kernels on sm_120. Always note the toolkit version in profiling reports. Check with `nvcc --version`.

8. **Allocation churn on the hot path.** High `cuMemAllocAsync` / `cuMemFreeAsync` counts during inference (not loading) indicate per-step allocations that should be pre-allocated or pooled. Check the CUDA API tail candidates table — allocation stalls show up as high max/p50 ratios.

9. **PATH in non-interactive shells.** When profiling via SSH or detached tmux, `nsys` may not be in PATH because `~/.bashrc` is only sourced for interactive shells. Either export PATH explicitly or use the full path `/usr/local/cuda/bin/nsys`.

10. **GPU metrics require privilege on consumer GPUs.** `--gpu-metrics-devices=all` fails on GeForce cards with `ERR_NVGPUCTRPERM`. On datacenter GPUs (A100/H100/B200), it works out of the box. Run `nsys status -e` to check.
