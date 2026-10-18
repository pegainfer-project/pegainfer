# Qwen3 Serving Performance (RTX 5090)

**TL;DR:** pegainfer Qwen3-4B TP1 on one RTX 5090 (32 GB): beats vLLM 0.24.0 at every QPS point from QPS 8 up — at QPS 16 it's +17% throughput (1980 vs 1688 tok/s), 19× lower TTFT (204 ms vs 3832 ms), 40% lower TPOT (47 ms vs 79 ms). Qwen3-8B runs on the same binary, same GPU. DSpark speculative decoding drops single-stream TPOT from 6.5 ms to 3.7 ms (~2× decode speedup). Measured with the earlier fixed-seed harness (see the note under Reproduce).

Last touched: 2026-07

## Setup

| Item | Value |
| --- | --- |
| GPU | 1× NVIDIA GeForce RTX 5090 (32 GB), driver 590.48.01 |
| CUDA | 13.1 build (`CUDA_HOME=/usr/local/cuda-13.1`) |
| Model | Qwen3-4B / Qwen3-8B, BF16 safetensors, TP1 |
| pegainfer | main @ `70888b2`, release build, CUDA Graph on (default), prefix cache on |
| vLLM | 0.24.0 (PyPI), prefix cache on (default), `--max-model-len 8192` |
| Client | `vllm-bench` (Rust) on localhost, same host, same GPU |
| Workload | `vllm-bench --dataset-name random`, in=1024 / out=128, Poisson arrivals, seed 42, 60 s per QPS point, `--temperature 0` (greedy) |

## Qwen3-4B Serving Load

| QPS | pegainfer out tok/s | vLLM out tok/s | pegainfer TTFT p50 | vLLM TTFT p50 | pegainfer TPOT p50 | vLLM TPOT p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 126.3 | 126.2 | 45.2 ms | 54.9 ms | 6.53 ms | 6.71 ms |
| 2 | 252.3 | 252.2 | 30.3 ms | 38.4 ms | 6.93 ms | 7.08 ms |
| 4 | 504.1 | 503.3 | 48.8 ms | 38.7 ms | 8.30 ms | 7.95 ms |
| 8 | 1007.8 | 1006.9 | 51.1 ms | 66.9 ms | 11.39 ms | 11.97 ms |
| 10 | 1258.3 | 1256.3 | 53.4 ms | 76.3 ms | 13.55 ms | 14.11 ms |
| 12 | 1507.7 | 1506.2 | 60.0 ms | 106.0 ms | 16.75 ms | 18.36 ms |
| 16 | **1979.9** | 1687.9 | **203.8 ms** | 3832.3 ms | **46.92 ms** | 79.42 ms |

Low load (QPS 1–4) is comparable. At QPS 8+ pegainfer leads on both TTFT and TPOT. At QPS 16 both systems are overloaded, but pegainfer edges ahead on throughput (+17%) and stays 19× lower on TTFT.

## Qwen3-8B Serving Load

Same harness, Qwen3-8B BF16, single RTX 5090. The 8B model is 2× the weights of 4B; throughput scales accordingly until the GPU saturates around QPS 8. QPS 10+ severely overloaded, omitted.

| QPS | pegainfer out tok/s | vLLM out tok/s | pegainfer TTFT p50 | vLLM TTFT p50 | pegainfer TPOT p50 | vLLM TPOT p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 125.1 | 125.0 | 82.2 ms | 97.4 ms | 11.55 ms | 11.63 ms |
| 2 | 249.9 | 250.0 | 54.1 ms | 61.5 ms | 11.46 ms | 11.57 ms |
| 4 | 498.6 | 498.5 | 88.1 ms | 103.6 ms | 16.08 ms | 16.24 ms |
| 8 | 991.9 | 990.4 | 148.0 ms | 235.1 ms | 30.97 ms | 35.56 ms |

## Footprint

| Metric | pegainfer | vLLM 0.24.0 |
| --- | ---: | ---: |
| RSS before stress, loaded and idle | **771 MB** | 3814 MB |
| RSS after stress | **1064 MB** | 3863 MB |
| Startup to HTTP ready, cold | **2.99 s** | 70.0 s |
| Startup, warm compile cache | **~3.0 s** | 32.7 s |
| GPU memory, default utilization | 28832 MiB | 30290 MiB |

pegainfer is a single process; vLLM RSS is summed over its process tree. The pegainfer RSS peak during load is transient while reading safetensors through `mmap`; steady-state settles at 771 MB after load.

## DSpark Speculative Decoding

[DSpark](https://github.com/deepseek-ai/DeepSpec) adds a semi-autoregressive Markov head to a DFlash parallel drafter, raising accepted draft length by conditioning each block position on the previously sampled token. Greedy verify keeps output lossless.

Single-stream TPOT drops from 5.8 ms to 3.0 ms — roughly 2× decode speedup from amortizing target forwards over accepted drafts. Concurrency sweep, greedy, sharegpt + SPEED-Bench (coding) datasets:

**ShareGPT:**

| Concurrency | baseline tok/s | DSpark tok/s | baseline TPOT p50 | DSpark TPOT p50 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 170 | **381** | 5.83 ms | 2.96 ms |
| 4 | 576 | **1288** | 6.72 ms | 3.59 ms |

**SPEED-Bench (coding):**

| Concurrency | baseline tok/s | DSpark tok/s | baseline TPOT p50 | DSpark TPOT p50 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 164 | **314** | 5.87 ms | 3.07 ms |
| 4 | 574 | **988** | 6.73 ms | 3.77 ms |

DSpark roughly doubles throughput and halves TPOT across both datasets.

DSpark beats the matched DFlash block7 baseline by +3.6% geomean output tok/s overall (+3–16% on text/code), with better accepted-draft distribution (2.52 vs 2.30 draft tokens/round). See [dspark-integration.md](dspark-integration.md) for the full A/B and design.

DFlash (the non-Markov predecessor) is also supported via the same flag. Single-stream decode: 1.82× on 5070 Ti, 1.56× on 5090. See [dflash-speculative-decoding.md](dflash-speculative-decoding.md).

## Warm Prefix-Cache TTFT

For multi-turn chat and agent workloads, most of the prompt often lands as a warm prefix-cache hit. Same prompt sent cold once to populate GPU KV cache, then re-sent warm (1-token output to isolate TTFT):

| Input length | pegainfer cold | pegainfer warm p50 | pegainfer warm p99 | vLLM warm p50 | vLLM warm p99 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 16.2 ms | 8.5 ms | 8.8 ms | 14.5 ms | 19.1 ms |
| 512 | 24.6 ms | 8.6 ms | 8.8 ms | 16.0 ms | 16.4 ms |
| 1024 | 44.0 ms | 9.2 ms | 9.5 ms | 18.4 ms | 19.0 ms |
| 2048 | 92.0 ms | 10.4 ms | 10.8 ms | 23.7 ms | 24.4 ms |
| 4096 | 211.5 ms | 12.7 ms | 13.4 ms | 34.1 ms | 36.2 ms |
| 8192 | 460.0 ms | 21.6 ms | 22.8 ms | 58.6 ms | 59.9 ms |
| 16384 | 1143.9 ms | **26.3 ms** | 27.9 ms | 95.6 ms | 98.2 ms |

pegainfer wins warm TTFT at every length; the 16k warm-cache path is 3.6× faster than vLLM p50.

## KV Offload

With `--kv-offload`, sealed Qwen3 KV blocks can be restored from the pegaflow host tier instead of recomputing full prefill. The pure-L2 mode below disables cross-request HBM prefix reuse, so every prefix hit is restored from host DRAM:

| Input length | Cold full prefill | L2 warm p50, host restore | Speedup |
| ---: | ---: | ---: | ---: |
| 256 | 25.4 ms | 9.8 ms | 2.6× |
| 512 | 25.6 ms | 11.6 ms | 2.2× |
| 1024 | 45.3 ms | 15.4 ms | 2.9× |
| 2048 | 92.5 ms | 22.9 ms | 4.0× |
| 4096 | 211.1 ms | 37.5 ms | 5.6× |
| 8192 | 461.3 ms | 71.4 ms | 6.5× |
| 16384 | 1140.5 ms | **125.5 ms** | 9.1× |

At 16k: HBM hit 26 ms < host-tier L2 restore 126 ms ≪ cold prefill 1140 ms.

## Reproduce

Harness note: these tables were measured with the earlier fixed-seed harness — one seed (42) for every sweep point against one long-lived server with the prefix cache on, so later points can replay earlier points' prompts as warm-prefix hits and results are order-dependent. The script now seeds each point independently; re-runs are not directly comparable to the tables above.

The current workload entry point is:

```bash
# pegainfer Qwen3-4B QPS sweep
MODEL=/data/Qwen3-4B GPU=0 tools/bench/run_serving_bench.sh

# vLLM Qwen3-4B QPS sweep (for comparison)
ENGINE=vllm MODEL=/data/Qwen3-4B GPU=0 \
  VLLM=~/develop/xingming/.venv/bin/vllm tools/bench/run_serving_bench.sh

# pegainfer Qwen3-4B + DSpark concurrency sweep
MODEL=/data/Qwen3-4B DRAFT_MODEL=/data/dspark_qwen3_4b_block7 GPU=0 \
  QPS_LIST="" CONCURRENCY_LIST="1 4 8" tools/bench/run_serving_bench.sh

# Qwen3-8B
MODEL=/data/Qwen3-8B GPU=0 tools/bench/run_serving_bench.sh
```

Warm TTFT sweep: `tools/bench/warm_ttft_sweep.py`. Build with `CUDA_HOME=/usr/local/cuda-13.1` (cuBLAS 12.9 has a GEMM cliff at N=1025; see [serving-perf-5090.md](serving-perf-5090.md) for the tuning history).
