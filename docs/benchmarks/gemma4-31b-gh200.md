# Gemma 4 31B on one GH200

**TL;DR:** The 31B checkpoint served bf16 at tensor parallel 1 on one GH200 (96 GiB), against a pinned vLLM on the same card. Single long-prompt requests: the folded-pool state (`PEGAINFER_GLOBAL_ATTN=tilelang640`) leads vLLM on every paired cell, E2EL −4.1% at a 10.6K prompt to −14.9% at 163K, TTFT −3 .. −16%, TPOT −4 .. −5%, 12 of 12 cells outside the ≤0.9% round spread, peaking 19 GiB lower on the card. The byte-identical default state trails vLLM by 8 .. 35% on the same cells and is kept as the numerical reference. At the default operating point all 32 concurrency and QPS cells complete with every output token and no error: ours leads on median latency and on throughput up to four in flight and at one request per second, and at eight and sixteen in flight matches vLLM's throughput and median end-to-end latency within 1.3% with half the admission latency and an 8 .. 10% slower wide-batch decode step, the open item at this size; the p99 tail is behind from two in flight on, which whole-prompt admission pays for. The envelope is measured at four operating points from 8192 × 16 slots (85.1 GiB idle) to 262144 × 1 (75.7 GiB), the maintained gate suite runs 45 of 45 on the 31B fixture set, and the regression thresholds are at the end.

## Platform and pins

| | |
| --- | --- |
| Card | one NVIDIA GH200 120GB (97871 MiB device memory, sm_90, aarch64), driver 565.57.01 |
| Checkpoint | `google/gemma-4-31B-it`, bf16, 60 layers (50 sliding hd256 with 16 KV heads, 10 global hd512 with 4 KV heads), vocabulary 262144, final logit softcap 30 |
| Ours | the commits this snapshot ships with; server built with `--features gemma4` and the TileLang-generated kernels present (`PEGAINFER_GEMMA4_TILELANG_PYTHON`). The performance tables were taken with TileLang 0.1.12 and the CUDA 12.6 toolchain; the correctness section was re-run whole on the shipped commits with TileLang 0.1.14 and the CUDA 13.1 toolchain, which emit different kernel bodies from the same definitions, so the two are not one run |
| vLLM | source build at `2cf0a6915` (reports `0.1.dev1+g2cf0a6915`), torch 2.13.0+cu126, FLASH_ATTN backend, bf16 KV, prefix caching off, `--language-model-only` |
| Client | `vllm bench serve`, random dataset, `--ignore-eos --temperature 0`, seeded per round |

Three of our states are one binary and one knob apart: `off` (the incumbent kernels on the split K|V pool, byte-identical serving), `tilelang` (the generated global kernel on the same pool) and `tilelang640` (the generated kernels on a folded 640-column global pool). All three read the sliding family through the same generated windowed prefill and decode kernels and project Q|K|V and gate|up with one GEMM each in the generated states.

## Single requests at four prompt lengths

Protocol: four arms booted in rotating order, four rounds, one request in flight; at each of four prompt lengths one discarded warm request then three kept, so 12 kept requests per cell; 256 output tokens; all four arms served at a 165888 ceiling with an 8192-token chunk (vLLM's own `max_num_batched_tokens`, applied to ours as `PEGAINFER_MIX_CHUNK_TOKENS`) and one decode slot; vLLM at `--gpu-memory-utilization 0.96`, ours with its pools sized by the ceiling. Paired deltas are same-seed, same-round.

Medians over the 12 kept requests:

| prompt | arm | TTFT s | TPOT ms | E2EL s | E2EL round spread |
| --- | --- | ---: | ---: | ---: | ---: |
| 10602 | vLLM | 1.360 | 20.69 | 6.64 | 0.5% |
| 10602 | tilelang640 | 1.332 | 19.73 | 6.36 | 0.3% |
| 10602 | tilelang | 1.336 | 19.76 | 6.37 | 0.5% |
| 10602 | off | 1.509 | 22.25 | 7.18 | 0.4% |
| 40002 | vLLM | 6.650 | 21.22 | 12.06 | 0.8% |
| 40002 | tilelang640 | 6.099 | 20.35 | 11.29 | 0.9% |
| 40002 | tilelang | 6.109 | 20.57 | 11.35 | 0.9% |
| 40002 | off | 7.902 | 23.28 | 13.84 | 0.9% |
| 81653 | vLLM | 17.686 | 22.00 | 23.29 | 0.9% |
| 81653 | tilelang640 | 15.531 | 21.08 | 20.91 | 0.9% |
| 81653 | tilelang | 15.616 | 21.55 | 21.11 | 0.6% |
| 81653 | off | 22.312 | 25.53 | 28.82 | 0.7% |
| 163336 | vLLM | 50.858 | 23.63 | 56.85 | 0.5% |
| 163336 | tilelang640 | 42.626 | 22.49 | 48.36 | 0.4% |
| 163336 | tilelang | 43.030 | 23.55 | 49.04 | 0.4% |
| 163336 | off | 69.173 | 30.05 | 76.84 | 0.3% |

Paired per-round deltas against vLLM (min .. max over the four rounds; negative is ours faster):

| prompt | arm | E2EL | TTFT | TPOT |
| --- | --- | --- | --- | --- |
| 10602 | tilelang640 | −4.2 .. −4.1% | −3.3 .. −2.0% | −4.7 .. −4.5% |
| 40002 | tilelang640 | −6.6 .. −6.3% | −8.7 .. −8.1% | −4.3 .. −3.7% |
| 81653 | tilelang640 | −10.3 .. −10.2% | −12.4 .. −12.1% | −4.6 .. −3.3% |
| 163336 | tilelang640 | −15.0 .. −14.9% | −16.3 .. −16.1% | −5.0 .. −3.7% |
| 10602 | tilelang | −4.1 .. −3.9% | −2.5 .. −1.4% | −4.5 .. −4.3% |
| 40002 | tilelang | −6.0 .. −5.8% | −8.4 .. −8.0% | −3.3 .. −2.7% |
| 81653 | tilelang | −9.6 .. −9.3% | −12.2 .. −11.6% | −2.4 .. −1.2% |
| 163336 | tilelang | −13.9 .. −13.7% | −15.5 .. −15.2% | −0.6 .. +0.9% |
| 10602 | off | +8.2 .. +8.3% | +10.0 .. +11.4% | +7.5 .. +7.7% |
| 40002 | off | +14.6 .. +14.9% | +18.3 .. +19.2% | +9.4 .. +10.1% |
| 81653 | off | +23.7 .. +23.9% | +25.8 .. +26.5% | +15.6 .. +17.1% |
| 163336 | off | +35.0 .. +35.4% | +35.9 .. +36.4% | +26.8 .. +28.7% |

Peak device memory over the whole campaign (500 ms samples): vLLM 97104 MiB (its utilization budget, not its need), `off` 82849, `tilelang` 82750, `tilelang640` 77886.

Output sanity: every arm returned its 256 tokens on every request.

Where the time goes at bs=1 (per decode step, same nsys instrument on both engines, 10.6K prompt): the weight-streaming GEMMs are 17.6 ms on both sides and sit at the memory roofline, so the lead is in everything around them. The lm-head is 2.82 GB of that stream (262144 × 5376 bf16), 4.9% of the 57.2 GiB of weights, about 0.9 ms of the step at the measured GEMM rate; its logits row is 1 MiB of f32 per decode slot and the softcap is a fused elementwise pass, so at this vocabulary the lm-head is a bandwidth term, not an allocation one.

## Concurrency and QPS cells

Both engines at the default operating point: an 8192 ceiling and 16 sequences (`--max-num-seqs 16` on vLLM to match the slot count; vLLM at `--gpu-memory-utilization 0.90`, which gave it a 30,828-token KV cache, 3.76 requests of the full ceiling, comfortably above what these cells hold), random 1024-token prompts with 256 output tokens, greedy, ignore-eos, one warm-up burst after each boot. Concurrency cells hold `c` requests in flight over `max(8, 4c)` prompts at an unbounded arrival rate; QPS cells send 64 prompts at a Poisson rate with no concurrency cap, so the server's own slots bound them. Two rounds in opposite boot order; the aggregator holds every cell to completed = requests, failed = 0 and output tokens = requests × 256, and the server logs are grepped for error finishes, since the bench counts an error-finished stream as successful.

All 32 cells clean, no error line in any server log. Medians over the two rounds (ms; throughput in requests and output tokens per second):

| cell | arm | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | E2EL p50 | E2EL p99 | req/s | out tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c=1 | tilelang640 | 121 | 123 | 19.5 | 19.5 | 5093 | 5096 | 0.20 | 50.3 |
| c=1 | vLLM | 121 | 123 | 20.5 | 20.5 | 5343 | 5345 | 0.19 | 47.9 |
| c=2 | tilelang640 | 161 | 249 | 20.3 | 20.3 | 5326 | 5425 | 0.38 | 96.1 |
| c=2 | vLLM | 195 | 242 | 21.0 | 21.2 | 5548 | 5551 | 0.36 | 92.3 |
| c=4 | tilelang640 | 289 | 510 | 21.3 | 21.9 | 5720 | 5864 | 0.70 | 178.2 |
| c=4 | vLLM | 461 | 467 | 21.4 | 22.7 | 5924 | 5930 | 0.68 | 172.8 |
| c=8 | tilelang640 | 414 | 1029 | 24.1 | 25.0 | 6555 | 7070 | 1.21 | 310.3 |
| c=8 | vLLM | 892 | 897 | 22.4 | 25.3 | 6596 | 6599 | 1.21 | 310.5 |
| c=16 | tilelang640 | 579 | 2094 | 29.4 | 31.2 | 8189 | 9386 | 1.94 | 495.8 |
| c=16 | vLLM | 1341 | 1800 | 26.6 | 31.3 | 8134 | 8207 | 1.96 | 502.4 |
| qps=1 | tilelang640 | 161 | 276 | 24.3 | 29.1 | 6362 | 7602 | 0.92 | 235.9 |
| qps=1 | vLLM | 159 | 579 | 25.5 | 31.7 | 6638 | 8479 | 0.92 | 235.1 |
| qps=2 | tilelang640 | 936 | 4324 | 30.0 | 31.4 | 8540 | 11866 | 1.63 | 418.0 |
| qps=2 | vLLM | 901 | 4244 | 30.1 | 30.9 | 8527 | 11822 | 1.63 | 416.5 |
| qps=4 | tilelang640 | 6601 | 13371 | 31.0 | 32.7 | 14515 | 20450 | 1.91 | 489.6 |
| qps=4 | vLLM | 6412 | 12548 | 30.7 | 31.1 | 14306 | 20236 | 1.92 | 491.4 |

Paired same-round ratios, ours over vLLM, median of the two rounds (below 1 is ours lower):

| cell | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | E2EL p50 | E2EL p99 | out tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c=1 | 0.998 | 1.000 | 0.952 | 0.952 | 0.953 | 0.953 | 1.049 |
| c=2 | 0.826 | 1.026 | 0.967 | 0.957 | 0.960 | 0.977 | 1.041 |
| c=4 | 0.627 | 1.090 | 0.996 | 0.966 | 0.966 | 0.989 | 1.031 |
| c=8 | 0.464 | 1.148 | 1.077 | 0.988 | 0.994 | 1.071 | 1.000 |
| c=16 | 0.432 | 1.163 | 1.104 | 0.995 | 1.007 | 1.144 | 0.987 |
| qps=1 | 1.009 | 0.726 | 0.956 | 0.923 | 0.959 | 0.905 | 1.004 |
| qps=2 | 0.987 | 1.019 | 0.997 | 1.013 | 1.001 | 1.003 | 1.004 |
| qps=4 | 1.031 | 1.069 | 1.009 | 1.054 | 1.015 | 1.010 | 0.996 |

The shape is the same at every cell: admission is faster, the wide-batch decode step is slower, and the tail pays for whole-prompt admission. Up to four in flight ours leads on TTFT p50 (0.63 .. 1.00), TPOT at both quantiles, E2EL at both quantiles and output throughput (1.03 .. 1.05), but its TTFT p99 is already behind from two in flight on (1.03 at c=2, 1.09 at c=4). At eight and sixteen in flight E2EL p50 and throughput are within 1.3% either way (E2EL p50 0.994 and 1.007, throughput 1.000 and 0.987), while the distribution differs: ours admits in less than half the time (TTFT p50 0.46 and 0.43) and decodes the wider batch 8 .. 10% slower per token (TPOT p50 24.1 against 22.4 ms at eight rows, 29.4 against 26.6 at sixteen), and both tails are behind (TTFT p99 1.15 and 1.16, E2EL p99 1.07 and 1.14). At one request per second ours leads on every column except TTFT p50, which ties. At two and four requests per second both engines are queue-bound (48 requests waiting on 16 slots at qps=4) and agree within 2% at p50; at p99 qps=4 is behind on TTFT and TPOT (1.069 and 1.054). Peak device memory: 85,417 MiB ours against 89,605 vLLM at its 0.90 budget.

The wide-batch decode step is the open item this snapshot names: at bs=1 the step is 4 .. 5% faster than vLLM's, at 16 rows 10% slower, so the batched decode kernels' efficiency across the bucket set is the next account to take at this scale.

Before the fix that ships with this snapshot the cells at c=16 and every QPS cell returned all 64 requests within milliseconds with 0 to 43 output tokens: the generated prefill kernels declared a plan bound of eight requests while a mixed step names every request holding a slot, the engine's mixed step failed on the kernel's refusal and wound the scheduler down, and the bench counted the error-finished streams as successful. The output-token check caught it, the server log named it, and the roster gate that now guards it is in the maintained suite.

## Memory envelope

Measured on the same card, `tilelang640` unless stated: what the process holds with no request in flight (`nvidia-smi` after the boot settles, which includes the CUDA context, the weights, both pools, the arena and the captured decode graphs), then one request of the ceiling's length (input = ceiling − 256, 256 output tokens, greedy) and the peak over it.

| operating point | slots | chunk | global pool | idle MiB | one request at the ceiling: TTFT / TPOT / E2EL | peak MiB |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| 8192 (default) | 16 | whole prompts | 2049 pages, 6.25 GiB | 85,127 | 7936 in: 0.97 s / 19.6 ms / 5.96 s | 87,431 |
| 8192 (default), `off` | 16 | whole prompts | 2049 pages, 10.00 GiB | 88,967 | 7936 in: 1.06 s / 21.5 ms / 6.54 s | 91,367 |
| 32768 | 8 | 2048 | 4097 pages, 12.50 GiB | 80,995 | 32512 in: 4.82 s / 20.1 ms / 9.96 s | 81,571 |
| 65536 | 4 | 2048 | 4097 pages, 12.50 GiB | 77,665 | 65280 in: 11.7 s / 20.7 ms / 16.9 s | 78,273 |
| 262144 | 1 | 2048 | 4097 pages, 12.50 GiB | 75,677 | 261888 in: 92.7 s / 24.2 ms / 98.9 s | 76,285 |

The weights are 57.19 GiB on the device (57.18 in the manifest; the 356 vision and audio tensors are skipped) and load in about 2.4 s from a warm page cache. The global pool is slots × ceiling rows at 640 columns per head on the folded pool, 1024 on the split one, which is the 3.75 GiB between the two default-point rows; the sliding pool is sized by the window and the chunk, so the raised ceilings hold less resident than the default point although they serve four to thirty-two times the context. At the default point a whole 8192-row prompt costs 2.3 GiB of transient scratch above idle; under the chunked walk the transient is bounded by the chunk and the peak sits 0.6 GiB above idle at every raised ceiling. Prefill at depth runs 0.35 ms per token at 262K where the first 64K cost 0.18 ms per token, the quadratic global-attention term having overtaken the linear one; decode at 262K of history is 24.2 ms per token against 19.6 at the default point.

Past the ceiling, at the default point: a 9001-token prompt is refused with HTTP 400 (`invalid_request_error`: "this model's maximum context length is 8192 tokens, but the prompt contains 9001 input tokens"), and so is a prompt of exactly 8192, since nothing is left for output; a 3-token prompt asking for `max_tokens` 8192 is served with `max_tokens` clamped to 8189 (usage 3 + 8189 = 8192, finish reason `length`), as vLLM's own frontend clamps it; the request after each of these is served normally. The bench client records the over-length prompt as one failed request ("Bad Request"), which is the shape the QPS cells would show if a cell ever crossed the ceiling.

## Correctness gates at 31B

The maintained suite (`scripts/gemma4_gates.sh`, `PEGAINFER_GEMMA4_FIXTURE_TAG=31b`) ran whole on the same card and build: 41 manifest gates, 45 executions (the four dual-profile serving contracts run once on the 31B dense checkpoint and once on the 26B routed one), every one completed, peak 97,871 MiB during the suite. The fixture set is the 12B set's four HF references dumped from `google/gemma-4-31B-it` at revision `842da3794eaa` (the window and long-context cases with the HF tower sharded over two GPUs) plus the chat-render reference; the runner holds them against the checkpoint's digests before the first load.

The window and long-context waypoints, teacher-forced against HF, max |Δ logprob| over the compared positions against the calibrated floor, top-1 agreement over the nine probed positions:

| case | max Δ | floor | top-1 |
| --- | ---: | ---: | --- |
| w1023 | 0.70 | 2.31 | 9/9 |
| w1024 | 0.74 | 2.81 | 9/9 |
| w1025 | 0.80 | 2.34 | 9/9 |
| w4096 | 2.77 | 4.29 | 9/9 |
| w4096, chunked 1024 | 2.766 | 4.29 | 9/9 |
| w16384 | 0.891 | 4.29 | 9/9 |
| w32768, chunked 2048 | 2.906 | 4.29 | 9/9 |
| w32768, whole prompt | skipped: the pass needs 11.7 GiB of scratch beside the stack and the card had 9.3 of 95.0 GiB free | | |

The whole-prompt 32K pass is the one case this card cannot hold beside 57 GiB of weights and a stack sized for the 32900-row ceiling; the gate prints that arithmetic and requires the chunked pass over the same prompt to have passed, which is the pass a raised ceiling serves through anyway.

The rest of the suite, in one line each: greedy generation matches HF generate; the mixed step matches its serial replay; the fp8 sliding pool's argmax agreement is 1.000 at both probed prompts (512/512 and 256/256) with the bf16 run-to-run reference at 1023/1023 and 2048/2048; the overlapped prefill is bit-equal to the sync step at 40 and 1500 prompt tokens with 13 greedy tokens equal; a prefix restore is bit-equal to the cold path at 200 of 264 and 1500 of 1564; the gathered walk does not depend on its batching and a ragged batch does not depend on row order; the raised ceiling and slots hold at the roster edge, the raise reaches the frontend and refuses without its prerequisites; admission is atomic across the two pools; the chat renders match the HF reference; with the incumbent bit-stable run to run, the generated global prefill differs from it by |Δ logit| 0.5 on the prompt row, and over nine decode cells (three fixture prompts at 512, 1500 and 3000 tokens, 16 steps each) the generated global decode differs by at most 5.625 and the folded pool from the split one by at most 5.9375, against a drift line of 12.0 that stands outside the measured spread rather than at its edge, with the argmax equal on every row of every cell; the routed block matches its reference formulas on the 26B checkpoint; and the device-only kernel contracts (norm fusions, suppression mask, router top-k, the fp8 sliding pool's byte layout and window reads) all pass.

## Regression thresholds

Same card only, per `docs/conventions/bench-regression.md`: gate on p50, a firing threshold means investigate.

| Cell | Metric | Threshold |
| --- | --- | --- |
| single request, each of the four prompt lengths, `tilelang640` | TPOT p50 | > 2% over this snapshot |
| single request, each of the four prompt lengths, `tilelang640` | TTFT p50 | > 3% over this snapshot |
| single request, `off` | TPOT and TTFT p50 | the same margins against its own row; it is the numerical reference, not the fast path |
| any single-request cell | peak device memory | > 80 GiB for `tilelang640`, > 85 GiB for `off` |
| concurrency cells at the default point | peak device memory | > 86 GiB for `tilelang640` |
| c=1 .. c=4 and qps=1 | E2EL p50 | > 2% over this snapshot |
| c=8 and c=16 | output tok/s | more than 3% under this snapshot |
| concurrency and QPS cells | failed requests | any |
| concurrency and QPS cells | output tokens | fewer than requests × 256 |

The round spread of every single-request cell was at most 0.9%, so a 2% TPOT margin is more than twice the noise and a 3% TTFT margin more than three times it.
