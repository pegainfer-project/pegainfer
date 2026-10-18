# Qwen3.5-4B serving vs vLLM on RTX 5090

> **TL;DR:** #469 is satisfied as a retained benchmark record, not as a parity claim. PegaInfer Qwen3.5-4B passed correctness gates, the HTTP matrix, QPS sweep, and overload recovery with zero failed benchmark requests, but vLLM 0.25.1 is faster across the retained serving envelope. The clearest gap is requested 1024/256 c16: PegaInfer `17.36ms` TPOT / `807 tok/s`; vLLM `9.34ms` / `1425 tok/s`. PegaInfer direct diagnostic for the same shape is `9.14ms` avg / `8.75ms` p50, so the next issue should trace HTTP/frontend/scheduler overhead first.

## Setup

| Field | Value |
| --- | --- |
| Date | 2026-07-15 |
| GPU | 1x NVIDIA GeForce RTX 5090, 32607 MiB |
| Driver / CUDA runtime | NVIDIA driver `595.71.05`, CUDA runtime `13.2` from `nvidia-smi` |
| CUDA toolkit | `nvcc 12.8.93`, `CUDA_HOME=/usr/local/cuda-12.8` |
| PegaInfer source | Snapshot of upstream/main `e2de05dbc52bedbbb3c213e648148b64cd12b3b4` |
| Rust | `rustc 1.99.0-nightly (af3d95584 2026-07-09)`, `cargo 1.99.0-nightly (59800466c 2026-07-07)` |
| Python / vLLM | Python 3.12.3, vLLM `0.25.1`, torch `2.11.0+cu130`, Triton `3.6.0` |
| Model | `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, config sha256 `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670` |
| Artifact | `target/bench-artifacts/qwen35-469-5090-20260715-1955/` on the benchmark host |

Remote Git fetch/clone was unreliable on the benchmark host. The artifact records that the source tree came from a local archive of upstream/main at `e2de05dbc52bedbbb3c213e648148b64cd12b3b4`. Remote-only dependency patches and third-party downloads were environment workarounds; they are not source changes.

## Flags

PegaInfer build and gates:

```bash
PEGAINFER_CUDA_SM=120 \
PEGAINFER_TRITON_PYTHON=<python-with-triton> \
CUDA_HOME=/usr/local/cuda-12.8 \
cargo build --release -p pegainfer-server --features qwen35

PEGAINFER_TEST_MODEL_PATH=$MODEL \
cargo test --release -p pegainfer-qwen35 --features qwen35 --test hf_golden_gate -- --nocapture

PEGAINFER_TEST_MODEL_PATH=$MODEL \
cargo test --release -p pegainfer-qwen35 --features qwen35 --test e2e_scheduler -- --nocapture
```

PegaInfer serve:

```bash
RUST_LOG=info target/release/pegainfer \
  --model-path "$MODEL" \
  --served-model-name Qwen3.5-4B \
  --port 8000
```

Qwen3.5 rejected the Qwen3-specific `--no-prefix-cache` and `--gpu-memory-utilization` flags. This run did not enable any PegaInfer product-mode prefix cache path.

vLLM serve:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "$MODEL" \
  --served-model-name Qwen3.5-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching
```

Relevant vLLM warnings:

- `Failed to get device capability: SM 12.x requires CUDA >= 12.9.`
- DeepGEMM import failed because `libnvrtc.so.13` was unavailable.
- The server still started, completed the matrix, and reported `enable_prefix_caching=False`.

## Benchmark Contract

- Client: `vllm bench serve --backend openai --endpoint /v1/completions`.
- Sampling: greedy, `--temperature 0`.
- Fixed-output cells: `--ignore-eos`.
- Prefix cache: disabled for vLLM; not enabled/exposed for PegaInfer Qwen3.5.
- Runs: 3 runs per cell, median reported below.
- Order: vLLM block first, then PegaInfer block. This is not ABBA-interleaved, so heat-state bias is possible.
- Output sanity: every cell records completed, failed, average input/output tokens, output hashes, and raw `output_lens`. No 0-token cell was counted as success.
- vLLM 0.25.1 random dataset cannot represent exact `input_len=1` because it may generate empty prompts. The 1-token cells use custom JSONL prompt `"Hello"` with `--skip-chat-template` and `--custom-output-len`.
- For random synthetic prompts, PegaInfer observed prompt token counts were lower than the requested length in several cells. The tables show observed average input tokens as `PegaInfer/vLLM`.

## Fixed / Concurrency

Median of 3 runs. `ok` and `fail` are total requests over all 3 runs.

| Cell | ok OI/vLLM | fail OI/vLLM | avg in OI/vLLM | avg out OI/vLLM | TPOT ms OI/vLLM | OI TPOT delta | out tok/s OI/vLLM | OI tok/s delta | ITL p99 OI/vLLM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1/256 c1 | 12/12 | 0/0 | 1.0/1.0 | 256.0/256.0 | 6.22/6.13 | +1.5% | 160/162 | -1.0% | 7.08/6.52 |
| 1/256 c4 | 12/12 | 0/0 | 1.0/1.0 | 256.0/256.0 | 8.15/7.08 | +15.2% | 484/558 | -13.2% | 8.86/9.61 |
| 1/256 c8 | 24/24 | 0/0 | 1.0/1.0 | 256.0/256.0 | 10.37/7.92 | +30.9% | 756/991 | -23.7% | 10.85/8.82 |
| 1/256 c16 | 48/48 | 0/0 | 1.0/1.0 | 256.0/256.0 | 15.21/8.17 | +86.1% | 1024/1896 | -46.0% | 18.03/10.07 |
| 1/512 c1 | 12/12 | 0/0 | 1.0/1.0 | 512.0/512.0 | 6.31/6.15 | +2.7% | 158/162 | -2.4% | 6.74/8.80 |
| 1/512 c4 | 12/12 | 0/0 | 1.0/1.0 | 512.0/512.0 | 8.22/7.06 | +16.4% | 483/563 | -14.1% | 9.79/7.74 |
| 1/512 c8 | 24/24 | 0/0 | 1.0/1.0 | 512.0/512.0 | 10.43/7.94 | +31.3% | 759/998 | -23.9% | 10.99/8.66 |
| 1/512 c16 | 48/48 | 0/0 | 1.0/1.0 | 512.0/512.0 | 15.29/8.23 | +85.6% | 1032/1914 | -46.1% | 15.97/11.96 |
| 1024/256 c1 | 12/12 | 0/0 | 973.5/1024.0 | 256.0/256.0 | 6.96/6.17 | +12.7% | 140/152 | -7.8% | 8.05/6.99 |
| 1024/256 c2 | 12/12 | 0/0 | 973.5/1024.0 | 256.0/256.0 | 7.84/6.94 | +12.9% | 246/265 | -7.1% | 8.63/9.97 |
| 1024/256 c4 | 12/12 | 0/0 | 973.5/1024.0 | 256.0/256.0 | 9.03/7.19 | +25.5% | 416/509 | -18.4% | 9.38/8.01 |
| 1024/256 c8 | 24/24 | 0/0 | 998.8/1024.0 | 256.0/256.0 | 11.67/8.39 | +39.1% | 624/825 | -24.4% | 58.49/11.85 |
| 1024/256 c16 | 48/48 | 0/0 | 970.9/1024.0 | 256.0/256.0 | 17.36/9.34 | +85.9% | 807/1425 | -43.3% | 77.05/64.98 |
| 2048/1 c1 | 12/12 | 0/0 | 1955.5/2048.0 | 1.0/1.0 | 0.00/0.00 | n/a | 10/9 | +15.3% | 0.04/0.00 |
| 2048/1 c4 | 12/12 | 0/0 | 1955.5/2048.0 | 1.0/1.0 | 0.00/0.00 | n/a | 10/11 | -13.3% | 0.03/0.00 |
| 2048/1 c8 | 24/24 | 0/0 | 2001.8/2048.0 | 1.0/1.0 | 0.00/0.00 | n/a | 10/12 | -20.2% | 0.02/0.00 |

The 2048/1 cells have TPOT `0.00` because there is only one generated token. Use output throughput and TTFT raw JSON for those cells.

## QPS Sweep

Workload: requested 1024/128, max concurrency 64, fixed output, Poisson/open-loop request rate.

| QPS | ok OI/vLLM | fail OI/vLLM | avg in OI/vLLM | TPOT ms OI/vLLM | out tok/s OI/vLLM | ITL p99 OI/vLLM |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 24/24 | 0/0 | 998.8/1024.0 | 7.34/6.53 | 115/115 | 8.15/9.53 |
| 2 | 24/24 | 0/0 | 998.8/1024.0 | 8.64/7.53 | 207/209 | 56.23/67.55 |
| 4 | 24/24 | 0/0 | 998.8/1024.0 | 11.28/9.18 | 321/345 | 59.54/79.74 |
| 8 | 48/48 | 0/0 | 970.9/1024.0 | 16.55/10.75 | 532/647 | 64.27/84.60 |
| 12 | 72/72 | 0/0 | 969.7/1024.0 | 24.46/11.91 | 678/999 | 82.79/74.37 |
| 16 | 96/96 | 0/0 | 971.1/1024.0 | 28.07/13.17 | 812/1286 | 85.35/73.99 |

PegaInfer kept every QPS cell alive with zero failures. Latency and throughput diverged at QPS 8 and above.

## Overload And Recovery

Overload cell: requested 1024/256, c32, request-rate `inf`, 3 runs.

| Cell | ok OI/vLLM | fail OI/vLLM | TPOT ms OI/vLLM | out tok/s OI/vLLM |
| --- | --- | --- | --- | --- |
| 1024/256 c32 | 96/96 | 0/0 | 25.27/11.04 | 1042/2193 |

Both servers survived the overload cell. A clean follow-up `Hello`, `max_tokens=8`, `temperature=0`, `ignore_eos=true` request succeeded after overload:

| Backend | completion tokens | output hash |
| --- | --- | --- |
| PegaInfer | 8 | `04048402ba653530` |
| vLLM | 8 | `83d4b3ded677dc41` |

## Direct Diagnostic

This is not serving evidence. It only decides where to look first.

| Workload | TTFT avg ms | steady TPOT avg/p50/p99 ms | request tok/s | decode tok/s |
| --- | --- | --- | --- | --- |
| 1024/256 c1 | 49.03 | 6.96/6.97/7.07 | 140 | 144 |
| 1024/256 c16 | 3827.11 | 9.14/8.75/9.10 | 41 | 108 |

Direct c1 TPOT matches HTTP c1. Direct c16 steady TPOT is close to vLLM HTTP c16 and far below PegaInfer HTTP c16. Direct c16 TTFT is high because the request benchmark reports all 16 requests and the in-process scheduler admitted work in waves. Start attribution in HTTP/frontend/scheduler/event timing.

## Claim Boundary

- This doc supports a retained #469 serving comparison on 1x RTX 5090 for the stack above.
- It does not support "SOTA", "vLLM parity", or "production ready".
- It does not include product-mode prefix cache results.
- It does not prove mixed-load ITL behavior. That remains a separate workload.
- It does not prove Qwen3.5 loses on every possible workload. It loses on the retained HTTP fixed-shape/QPS/overload envelope here. The closest cell is 1-token c1, where PegaInfer is within a few percent of vLLM.
- It does not prove a model-kernel root cause for the high-concurrency HTTP gap.

## Follow-Up Recommendation

Next work should start with HTTP/frontend/scheduler attribution:

1. Add per-step serving trace fields for Qwen3.5 HTTP: queue wait, scheduled-to-first-token, step type, decode batch size, unified-step duration, send/event overhead, and terminal latency.
2. Re-run 1024/256 c1/c16 and QPS 8/12/16 with the same output sanity fields.
3. Only move to model-kernel profiling if the trace shows GPU step time itself explains the HTTP c16 gap.
