# Qwen3.5-4B serving: pegainfer vs vLLM on RTX 5090

**Created**: 2026-06-15

**Updated**: 2026-06

**TL;DR**: This Qwen3.5 decode-tuning refresh improves pegainfer's direct TPOT
by `2.1-3.2%`. HTTP decode is close on 1-token prompts, but vLLM 0.23.0 still
leads 1024/256 TPOT and high-concurrency output throughput. TTFT rows are
fixed-client timings because the longer prompts report different token totals.

Source benchmark for the README Qwen3.5 performance rows.

## Setup

| Item | Value |
| --- | --- |
| GPU | 1x NVIDIA GeForce RTX 5090 (32 GB), driver 580.105.08, same GPU for each sequential run |
| Model | Qwen3.5-4B, BF16 safetensors, TP1, text-only serving |
| pegainfer | branch based on upstream/main `a1846ca`, release build with `--features qwen35`, CUDA Graph decode on by default |
| vLLM | 0.23.0 from PyPI, latest stable release checked in June 2026 |
| pegainfer serve flags | `--no-prefix-cache`, CUDA Graph decode on by default |
| vLLM serve flags | `--language-model-only`, `--no-enable-prefix-caching`, `--max-model-len 8192`, `--gpu-memory-utilization 0.9` |
| vLLM env | `VLLM_USE_FLASHINFER_SAMPLER=0`; this SM120/CUDA 12.8 host hit a FlashInfer sampler startup error otherwise. Attention selected FlashAttention 2. |
| Client | `vllm bench serve` 0.23.0 on localhost, OpenAI `/v1/completions` backend |
| Profiler | Nsight Systems 2025.3.2 for PegaInfer direct and HTTP diagnostics |

Client flags for both HTTP engines:

| Field | Value |
| --- | --- |
| Dataset | random |
| Request count | `--num-prompts 64` |
| Warmup | `--num-warmups 2` |
| Request rate | `--request-rate inf` |
| Length control | `--random-range-ratio 0.0` |
| Decoding | `--temperature 0`, `--ignore-eos` |
| Seed | `--seed 20260618` |

## PegaInfer Direct A/B

Same binary interface, same model, same GPU. Baseline is upstream/main before
the Qwen3.5 gate/up MLP fusion and decode cublasLt tuning in this branch.

| Workload | Metric | upstream/main | tuned branch | Delta |
| --- | --- | ---: | ---: | ---: |
| 1 input / 256 output | steady TPOT avg | 6.524 ms | 6.386 ms | -2.1% |
| 1 input / 512 output | steady TPOT avg | 6.603 ms | 6.397 ms | -3.1% |
| 1024 input / 256 output | steady TPOT avg | 7.338 ms | 7.100 ms | -3.2% |
| 2048 input / 1 output | TTFT avg | 97.978 ms | 95.855 ms | -2.2% |

## HTTP Fixed Shapes

| Workload | Metric | pegainfer | vLLM 0.23.0 | Read |
| --- | --- | ---: | ---: | --- |
| 1 input / 256 output | completed | 64/64 | 64/64 | both clean |
| 1 input / 256 output | TTFT mean | 11.83 ms | 15.46 ms | pegainfer lower on this client path |
| 1 input / 256 output | TPOT mean | 6.282 ms | **6.214 ms** | vLLM 1.1% lower |
| 1 input / 256 output | output tok/s | 158.58 | **159.95** | vLLM 0.9% higher |
| 1 input / 512 output | completed | 64/64 | 64/64 | both clean |
| 1 input / 512 output | TTFT mean | 11.55 ms | 16.23 ms | pegainfer lower on this client path |
| 1 input / 512 output | TPOT mean | 6.381 ms | **6.221 ms** | vLLM 2.5% lower |
| 1 input / 512 output | output tok/s | 156.45 | **160.22** | vLLM 2.4% higher |
| 1024 input / 256 output | completed | 64/64 | 64/64 | both clean |
| 1024 input / 256 output | reported input tokens | 63,459 (991.5/request) | 65,536 (1,024.0/request) | prompt-token totals differ |
| 1024 input / 256 output | TTFT mean | 55.29 ms | 66.34 ms | fixed-client timing, not token-normalized prefill |
| 1024 input / 256 output | TPOT mean | 7.110 ms | **6.346 ms** | vLLM 10.8% lower |
| 1024 input / 256 output | output tok/s | 136.98 | **151.92** | vLLM 10.9% higher |
| 2048 input / 1 output | completed | 64/64 | 64/64 | both clean |
| 2048 input / 1 output | reported input tokens | 126,957 (1,983.7/request) | 131,072 (2,048.0/request) | prompt-token totals differ |
| 2048 input / 1 output | TTFT mean | 97.41 ms | 101.93 ms | fixed-client timing, not token-normalized prefill |
| 2048 input / 1 output | output tok/s | 10.24 | 9.78 | client-contract throughput; prompt-token totals differ |

## HTTP Concurrency Sweep

Workload: 1024 input / 256 output, `num_prompts=64`, random fixed-length
client probes, prefix cache disabled on both servers.

| Max concurrency | pegainfer TTFT mean | vLLM TTFT mean | pegainfer TPOT mean | vLLM TPOT mean | pegainfer output tok/s | vLLM output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 55.29 ms | 66.34 ms | 7.110 ms | **6.346 ms** | 136.98 | **151.92** |
| 2 | 82.07 ms | 97.64 ms | 8.146 ms | **7.148 ms** | 237.06 | **266.55** |
| 4 | 167.04 ms | 165.88 ms | 9.263 ms | **7.459 ms** | 404.76 | **494.85** |
| 8 | 352.18 ms | 232.45 ms | 11.333 ms | **8.650 ms** | 631.33 | **839.05** |
| 16 | 741.95 ms | 358.21 ms | 15.566 ms | **9.823 ms** | 868.63 | **1425.72** |

## Attribution

PegaInfer's measured in-process 1024/256 concurrency-16 run reported `9.320 ms`
steady TPOT avg, while the HTTP row is `15.566 ms`. Nsight Systems therefore
points the high-concurrency gap at serving/scheduler/event-sync overhead before
another model-kernel rewrite. The full HTTP trace includes startup and warmup,
so it is coarse attribution only.

## Caveats

- This is a same-host synthetic benchmark, not a production traffic trace.
- The 1-token-prompt decode rows have equal reported input/output token totals.
  The 1024/256 and 2048/1 rows do not, so TTFT is a fixed-client workload
  timing rather than token-normalized prefill throughput.
- Prefix cache was disabled on both servers for this refresh.
- vLLM startup was made serviceable on this host by disabling the FlashInfer
  sampler path. The measured decode was greedy (`temperature=0`); attention
  still selected FlashAttention 2.
- Nsight Systems direct traces are measured-range. HTTP traces are full-process
  diagnostics and should be treated as coarse attribution only.
- The current honest claim is narrower than vLLM parity: this branch improves
  pegainfer Qwen3.5 decode TPOT by a few percent, closes most of the prompt-len
  1 HTTP decode gap, and leaves the 1024/256 plus high-concurrency HTTP gap as
  follow-up work.
