# Qwen3.5-4B serving on A100-40GB: PegaInfer vs vLLM 0.27.0

**Created**: 2026-09

**TL;DR**: Same-session, same-client comparison of PegaInfer `upstream/main` + #1072 + #1073 against vLLM 0.27.0 on 1x A100-40GB, four cells, zero failed requests. PegaInfer beats vLLM's mean TPOT at bs1 (8.16 against 8.21 ms) and c8 (9.81 against 10.07), stays 6.1% behind at c16 (11.79 against 11.12) and 23% behind at qps16 (29.71 against 24.06), wins single-request TTFT by 2.5x, and — under the opt-in `--decode-overlap stream` + `--qwen35-scheduler-policy auto` pose, measured on the tree before #1073 — wins qps16 mean TPOT (20.89 against 24.06 ms), qps16 ITL p99 (39.0 against 97.0) and c16 ITL p99 (34.2 against 85.5). Output throughput is within 2% of vLLM at bs1, c8 and c16 and 7.3% behind at qps16. The prefill budget is a separate dial measured here: 4096 instead of the 1024 default improves TPOT, TTFT and throughput on all three cells and cuts the c8 and c16 tails five- to six-fold, but triples the qps16 p99 because the whole-run total falls while the longest step grows.

## Setup

| Item | Value |
| --- | --- |
| GPU | 1x NVIDIA A100-SXM4-40GB (sm_80), one engine at a time on device 1 |
| Model | Qwen3.5-4B, BF16, TP1, text-only serving, `/mnt/data/models/Qwen3.5-4B` |
| PegaInfer | `upstream/main` `8f455a18` + #1072 (split-KV decode) + #1073 (fused linear-attention projections), release build `--features qwen35` |
| vLLM | 0.27.0 (`~/vllm-omni-venv`), FLASH_ATTN + FLA Triton GDN + FlashInfer sampler, piecewise CUDA graphs |
| PegaInfer flags | default, and `--decode-overlap stream --max-batch 32 --qwen35-scheduler-policy auto` |
| vLLM flags | `--dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 --no-enable-prefix-caching` |
| Client | `vllm bench serve` from the same venv, OpenAI `/v1/completions`, `--dataset-name random --random-range-ratio 0 --temperature 0 --ignore-eos --seed 42` |
| Cells | bs1 and c8 and c16 at 1024 in / 256 out; qps16 at 1024 in / 128 out, `--request-rate 16`, 64 prompts |

## Results

Mean TPOT, and the tail and first-token metrics beside it. Lower is better everywhere except output tok/s. The default column is the current tree (#1072 and #1073, two runs a side); the auto+stream column was measured on the tree before #1073 and is kept as the record of that pose.

| cell | metric | vLLM 0.27 | PegaInfer default | PegaInfer auto+stream |
| --- | --- | --- | --- | --- |
| bs1 @1024/256 | TPOT | 8.21 | **8.16** | 8.55 |
| | TTFT | 198.3 | **78** | 88 |
| c8 @1024/256 | TPOT | 10.07 | **9.81** | 10.40 |
| | ITL p99 | 26.9 | **10.0** | 31.4 |
| | TTFT | 359.0 | 384 | 391 |
| | output tok/s | 696 | **707** | 652 |
| c16 @1024/256 | TPOT | **11.12** | 11.79 | 12.59 |
| | ITL p99 | 85.5 | **12.2** | 34.2 |
| | TTFT | **550.6** | 672 | 699 |
| | output tok/s | **1197** | 1107 | 990 |
| qps16 | TPOT | **24.06** | 29.71 | 20.89 |
| | ITL p99 | 97.0 | 264.1 | **39.0** |
| | TTFT | **228.2** | 415 | 1024 |
| | output tok/s | **1394** | 1293 | 1110 |

Every row completed with zero failed requests on both engines.

## What the numbers say

- **Step time is the deficit that never goes away, and the projection arrangement was the first half of it.** At c16 PegaInfer needs 11.79 ms per decode step against vLLM's 11.12, and output throughput tracks that ratio at the same concurrency. The kernel-level attribution is in `models/qwen35/decode-kernel-attribution.md`: full-attention paged decode is the largest single block, and the decode GEMM family is at per-kernel parity on every shape both engines run, so what was left there was vLLM's shape arrangement. #1073 fuses the linear-attention projections the way vLLM does and takes 0.30 ms off every decode step (−2.7% bs1, −3.0% c8, −2.5% c16, −1.0% qps16); the full-attention q+k+v fusion is the same change applied to the other four layers in eight.
- **The overlap pose is where the tails live.** `--decode-overlap stream --qwen35-scheduler-policy auto` takes c16 ITL p99 from 79.2 to 34.2 ms and qps16 ITL p99 from 97.2 to 39.0, both better than vLLM, and turns qps16 mean TPOT into a win. It is neutral at bs1 (8.55/8.59 ms against 8.57) and costs about 3% of c16 throughput. It is opt-in today, so the default posture leaves those three wins on the table.
- **Prefill admission is what the overlap pose costs.** qps16 TTFT goes 772 → 1024 ms and c16 stays 22% behind vLLM. vLLM holds both a low TTFT and a low TPOT, so it is overlapping without starving prefill. This is a scheduler question, not a kernel one.
- **The c8 tail is a separate loss.** 31.4 ms p99 against vLLM's 26.9 under the same pose, while c16 and qps16 are won. Whatever blocks decode during a prefill at concurrency 8 is not what blocks it at 16.

## The prefill budget is a latency/throughput dial, and it now ships at 4096

`--max-prefill-tokens` sets how many prompt tokens one step may prefill, and so how many admitted prompts ride in a single unified step. How finely prefill is chopped turns out to be the structural difference against vLLM on these cells: at the old 1024 default a 1024-token prompt consumes the whole budget and exactly one prompt is prefilled per step, so a ramp costs one step per request and c16 spends sixteen steps admitting sixteen requests. Per layer-step our GDN prefill is already the faster of the two — 247 µs against FLA's 433 — and the whole cost is in how many steps the same work is spread over.

Two runs a side, default pose, zero failed requests:

| cell | metric | 1024 | **4096 (new default)** |
| --- | --- | --- | --- |
| c8 | mean TPOT | 10.42 / 10.47 ms | **10.14 / 10.11 ms** |
| c8 | ITL p99 | 64.9 / 65.8 ms | **10.2 / 10.2 ms** |
| c8 | output throughput | 665 / 661 tok/s | **679 / 683 tok/s** |
| c16 | mean TPOT | 12.55 / 12.55 ms | **12.09 / 12.09 ms** |
| c16 | ITL p99 | 79.7 / 79.1 ms | **13.0 / 13.2 ms** |
| c16 | output throughput | 1038 / 1033 tok/s | **1089 / 1082 tok/s** |
| qps16 | mean TPOT | 31.99 / 32.00 ms | **29.89 / 29.97 ms** |
| qps16 | TTFT | 764 / 759 ms | **418 / 419 ms** |
| qps16 | output throughput | 1140 / 1140 tok/s | **1289 / 1287 tok/s** |
| qps16 | ITL p99 | 95.9 / 97.0 ms | 260.7 / 261.5 ms |

Every metric this snapshot is about improves, and the c8 and c16 tails improve sixfold with them. The one regression is qps16's ITL p99, and it is the same mechanism viewed from the other side: that cell really does queue several prompts, and a step carrying five of them is longer than a step carrying two. The scheduler trace at 4096 shows the step count falling 203 → 172 and the total step time 7130 → 6291 ms while the longest step rises 97.5 → 267.4 ms, so the whole-run total and the tail move in opposite directions. `--max-prefill-tokens 1024` restores the old behaviour where that tail matters more than the throughput.

Against vLLM 0.27 this moves c8 to parity (10.15 against 10.07 ms) with an ITL p99 2.5x better, c16 to −8.6% TPOT with a p99 6.8x better, and qps16 to −25% TPOT, half the TTFT and −7.8% throughput.

## Claim boundary

Two runs per PegaInfer configuration and one run per vLLM cell, zero failed requests on every row. The default column is four runs a side across two cards: two interleaved a/b/b/a sessions on device 1, and two more a side on device 0 against the same baseline binary, which reproduced the fusion's delta at every cell (bs1 −2.6%, c8 −3.0%, c16 −3.7%, qps16 −1.1%) with the absolute rows agreeing to 0.05 ms between the cards. The vLLM column is one same-session run per cell on device 1. These are single-host snapshot numbers: the c16 and qps16 rows are not parity claims and the run-to-run spread on this card is around 0.5% on mean TPOT at c16. bs1 TTFT is measured at `max-concurrency 1` and includes the engine's own startup of that request only.

## Next step

The step-time gap is the binding constraint at c16 and qps16. Three items are open and independent:

- **The full-attention projections.** #1073 fused the linear-attention half (qkv+z, beta+alpha) and took 0.30 ms off every decode step. The same change applied to `q`, `k` and `v` in the eight full-attention layers is worth the 160 µs/step those separate kernels cost and is the largest single remaining GEMM item. The HD256 attention kernel itself is the other half: vLLM runs flash-attention's `flash_fwd_splitkv`, and replacing our own split-KV kernel with a tensor-core form needs the per-position dependent chain broken first — the memory floor for the same access pattern at c16 is 54 µs per layer-step against 98.6 now.
- **Prefill.** Our prefill GEMM spends 566 ms of the c16 window against vLLM's 504 for the same tokens, and the `gdr_*` chunkwise kernels spend 0.57 ms/step against FLA's 0.29. Both bound how fast an admission ramp clears, which is what qps16 TTFT is made of.
- **Two scheduler items, independent of the kernels.** The overlap pose holds the c16 and qps16 tails and turns qps16 mean TPOT into a win, but costs qps16 TTFT (1024 ms against vLLM's 228); holding both needs a prefill-admission policy that does not starve prefill while it protects decode. And the prefill-budget section above shows that the width of a unified step trades the whole-run total against the p99 tail, so the budget wants a latency or work bound rather than a larger constant.
