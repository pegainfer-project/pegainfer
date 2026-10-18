# Qwen Mixed Sampling HTTP Benchmark

**Created**: 2026-06-18

**Last run**: 2026-06-21

**TL;DR**: Issue #412 now has HTTP `/v1/completions` mixed
greedy/sampled serving evidence. Qwen3-4B completed 64/64 requests with
failed=0/timeouts=0 and 32 greedy + 32 sampled requests in one concurrent run;
Qwen3.5-4B passed the same workload as supplemental evidence.

## Setup

| Item | Value |
| --- | --- |
| Issue | [#412](https://github.com/pegainfer-project/pegainfer/issues/412) |
| GPU | 1x NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.105.08 |
| Primary model | Qwen3-4B BF16 safetensors, TP1, text-only serving |
| Supplemental model | Qwen3.5-4B BF16 safetensors, TP1, text-only serving |
| Server | Existing PegaInfer release binary; `RUST_LOG=info`; not rebuilt during the 2026-06-21 rerun |
| Client | This branch's `scripts/bench_http_serving.py`, rebased onto `upstream/main` `b66f845` |

PR [#424](https://github.com/pegainfer-project/pegainfer/pull/424) is the
benchmark-snapshot format precedent; its multi-turn workload is separate. The
client uses the OpenAI-compatible `temperature`, `top_k`, and `top_p` request
fields used by `vllm bench serve`, without adding a vLLM dependency.

## Command

```bash
RUST_LOG=info pegainfer \
  --model-path <Qwen3-4B> \
  --served-model-name Qwen3-4B \
  --port 18080
```

```bash
python3 scripts/bench_http_serving.py \
  --base-url http://127.0.0.1:18080 \
  --model Qwen3-4B \
  --num-requests 64 \
  --concurrency 8 \
  --warmup 8 \
  --prompt-words 128 \
  --max-tokens 64 \
  --sampling-mode mixed-greedy-sampled \
  --server-log <server.log> \
  --out <raw.json>
```

`mixed-greedy-sampled` alternates by global request index:

| Label | Profile |
| --- | --- |
| greedy | `temperature=0.0`, `top_k=-1`, `top_p=1.0` |
| sampled | `temperature=0.8`, `top_k=40`, `top_p=0.95` |

The HTTP body does not send `seed`. Disabled top-k is sent as `top_k=0`
because the frontend wire type is unsigned; the report records the semantic
profile as `top_k=-1`. Qwen launch paths use a fixed engine seed, so same-binary
reruns can reproduce sampled hashes without per-request seeds.

## Results

Raw artifact labels from the 2026-06-21 rerun:
`qwen3-4b-latest-mixed-http-64c8.json` and
`qwen35-latest-mixed-http-64c8.json`. Remote storage paths are intentionally
omitted.

| Metric | Qwen3-4B | Qwen3.5-4B |
| --- | ---: | ---: |
| Completed / requested | 64 / 64 | 64 / 64 |
| Failed | 0 | 0 |
| Timeouts | 0 | 0 |
| Greedy / sampled completed | 32 / 32 | 32 / 32 |
| Wall time | 3.542 s | 6.139 s |
| QPS | 18.069 | 10.426 |
| Input tokens/s | 2312.77 | 1334.52 |
| Output tokens/s | 1156.38 | 667.26 |
| TTFT avg / p50 / p95 | 26.20 / 16.12 / 73.83 ms | 114.54 / 119.71 / 178.72 ms |
| TPOT avg / p50 / p95 | 6.582 / 6.549 / 6.672 ms | 10.336 / 10.139 / 11.655 ms |
| ITL avg / p50 / p99 | 6.582 / 6.499 / 9.012 ms | 10.336 / 10.082 / 12.772 ms |
| Output chunks | 4096 | 4096 |
| Combined output hash | `74933fbe0cff594b` | `b6a5478ea59ebefe` |
| Unique output hashes | 24 | 25 |

The raw report records per-request `sampling_label`, `temperature`, `top_k`,
and `top_p`, plus workload and summary sampling counts. This server build did
not emit `pegainfer_http_trace` lines, so TTFT/TPOT/ITL are client-observed
metrics, not server phase attribution. Qwen3.5-4B is supplemental evidence; the
table is not a performance comparison between model lines.

Both server logs were scanned after the run and had no `error`, `panic`,
`failed`, or `timeout` lines. The server was stopped after the benchmark, and
the GPU compute process list was empty.

## Claim Boundaries

- This is HTTP serving evidence, not a vLLM parity or production traffic claim.
- It fills #284's HTTP serving gap; it does not replace #284's correctness,
  direct benchmark, or nsys evidence.
