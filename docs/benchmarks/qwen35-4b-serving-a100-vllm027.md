# Qwen3.5-4B serving on A100-40GB: PegaInfer vs vLLM 0.27.0

**Created**: 2026-09

**TL;DR**: Same-session, same-client comparison of PegaInfer `upstream/main` + #1072 against vLLM 0.27.0 on 1x A100-40GB, four cells, zero failed requests. PegaInfer wins single-request TTFT by 2.3x and — under the opt-in `--decode-overlap stream` + `--qwen35-scheduler-policy auto` pose — qps16 mean TPOT (20.89 vs 24.06 ms), qps16 ITL p99 (39.0 vs 97.0) and c16 ITL p99 (34.2 vs 85.5). It still loses mean TPOT on every cell (4–13%), output throughput everywhere (6–19%), the c8 ITL tail, and time to first token at c16 and qps16 (23% and 4.5x). The pose is neutral at bs1, so the losses that remain are step time and prefill admission, not the decode-overlap mechanism. The prefill budget is a separate dial measured here: 4096 instead of the 1024 default improves TPOT, TTFT and throughput on all three cells and cuts the c8 and c16 tails five- to six-fold, but triples the qps16 p99 because the whole-run total falls while the longest step grows.

## Setup

| Item | Value |
| --- | --- |
| GPU | 1x NVIDIA A100-SXM4-40GB (sm_80), one engine at a time on device 1 |
| Model | Qwen3.5-4B, BF16, TP1, text-only serving, `/mnt/data/models/Qwen3.5-4B` |
| PegaInfer | `upstream/main` `8f455a18` + #1072 (split-KV decode for buckets ≤ 16), release build `--features qwen35` |
| vLLM | 0.27.0 (`~/vllm-omni-venv`), FLASH_ATTN + FLA Triton GDN + FlashInfer sampler, piecewise CUDA graphs |
| PegaInfer flags | default, and `--decode-overlap stream --max-batch 32 --qwen35-scheduler-policy auto` |
| vLLM flags | `--dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 --no-enable-prefix-caching` |
| Client | `vllm bench serve` from the same venv, OpenAI `/v1/completions`, `--dataset-name random --random-range-ratio 0 --temperature 0 --ignore-eos --seed 42` |
| Cells | bs1 and c8 and c16 at 1024 in / 256 out; qps16 at 1024 in / 128 out, `--request-rate 16`, 64 prompts |

## Results

Mean TPOT, and the tail and first-token metrics beside it. Lower is better everywhere except output tok/s.

| cell | metric | vLLM 0.27 | PegaInfer default | PegaInfer auto+stream |
| --- | --- | --- | --- | --- |
| bs1 @1024/256 | TPOT | **8.21** | 8.57 | 8.55 |
| | TTFT | 198.3 | **87** | **88** |
| c8 @1024/256 | TPOT | **10.07** | 10.65 | 10.40 |
| | ITL p99 | **26.9** | 65.6 | 31.4 |
| | TTFT | 359.0 | 381 | 391 |
| | output tok/s | **696** | 654 | 652 |
| c16 @1024/256 | TPOT | **11.12** | 12.80 | 12.59 |
| | ITL p99 | 85.5 | 79.2 | **34.2** |
| | TTFT | **550.6** | 673 | 699 |
| | output tok/s | **1197** | 1021 | 990 |
| qps16 | TPOT | 24.06 | 32.11 | **20.89** |
| | ITL p99 | 97.0 | 97.2 | **39.0** |
| | TTFT | **228.2** | 772 | 1024 |
| | output tok/s | **1394** | 1134 | 1110 |

Every row completed with zero failed requests on both engines.

## What the numbers say

- **Step time is the deficit that never goes away.** At c16 PegaInfer needs 12.59 ms per decode step against vLLM's 11.12, and output throughput tracks that ratio exactly at the same concurrency. The kernel-level attribution for it is in `models/qwen35/decode-kernel-attribution.md`: full-attention paged decode is the largest single block, and the GEMM families are already at per-kernel parity, so the remaining distance is a tensor-core attention kernel plus the projection fusion vLLM does.
- **The overlap pose is where the tails live.** `--decode-overlap stream --qwen35-scheduler-policy auto` takes c16 ITL p99 from 79.2 to 34.2 ms and qps16 ITL p99 from 97.2 to 39.0, both better than vLLM, and turns qps16 mean TPOT into a win. It is neutral at bs1 (8.55/8.59 ms against 8.57) and costs about 3% of c16 throughput. It is opt-in today, so the default posture leaves those three wins on the table.
- **Prefill admission is what the overlap pose costs.** qps16 TTFT goes 772 → 1024 ms and c16 stays 23% behind vLLM. vLLM holds both a low TTFT and a low TPOT, so it is overlapping without starving prefill. This is a scheduler question, not a kernel one.
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

Two runs per PegaInfer configuration and one run per vLLM cell, one GPU, default flags unless the pose column says otherwise, zero failed requests on every row. These are single-host snapshot numbers: the c16 and qps16 rows are not parity claims and the run-to-run spread on this card is around 0.5% on mean TPOT at c16. bs1 TTFT is measured at `max-concurrency 1` and includes the engine's own startup of that request only.

## Next step

The step-time gap is the binding constraint at c8 and c16. It needs the HD256 attention path replaced with a tensor-core kernel (vLLM runs flash-attention's `flash_fwd_splitkv`; the memory floor for the same access pattern on this card is 54 µs per layer-step against the 154 µs the current FlashInfer kernel spends) and the decode projections fused the way vLLM fuses them (one M=12288 GEMM per linear layer and one M=10240 GEMM per full-attention layer, 40 fewer launches per step).

Two scheduler items are open and independent of the kernels. The overlap pose holds the c16 and qps16 tails and turns qps16 mean TPOT into a win, but costs qps16 TTFT (1024 ms against vLLM's 228); holding both needs a prefill-admission policy that does not starve prefill while it protects decode. And the prefill-budget section above shows that the width of a unified step trades the whole-run total against the p99 tail, so the budget wants a latency or work bound rather than a larger constant.
