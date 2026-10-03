# Gemma 4 31B bf16 tensor-parallel-2 on two L20s

TL;DR: the stock 31B bf16 checkpoint serves on two 48 GiB L20s (sm_89, PCIe) as a two-rank eager engine — 29.91 GiB of weights per rank, an 8192-token ceiling at 8 decode slots, cold load 30 s, first token 57 ms, 49.4 ms per decode step, and 143 tok/s aggregate at 8 concurrent. The checkpoint does not fit one card (57 GiB), so there is no in-box single-rank baseline for it; numerical parity rests on the 31B-geometry synthetic gate (`models/gemma4/tp.md`).

## Rig

| | |
| --- | --- |
| checkpoint | `gemma-4-31b-it`, bf16, text tower only (356 vision/audio tensors skipped) |
| hardware | two NVIDIA L20 (sm_89, 44.99 GiB usable each), PCIe, one p8s node |
| build | `--features gemma4`, `PEGAINFER_CUDA_SM=89`, CUDA 12.9 |
| runtime | `--tp-size=2 --cuda-graph=false`, `NCCL_PROTO=LL128`, pod NCCL 2.18.3 |
| envelope | `PEGAINFER_MAX_CONTEXT=8192`, `PEGAINFER_DECODE_SLOTS=8` |
| checkpoint source | node-local disk (a PVC read is ~3.5 min per rank) |

## Envelope

| | per rank |
| --- | --- |
| weights (device) | 29.91 GiB |
| global KV pool | 2.50 GiB (1025 pages x 2.50 MiB) |
| resident after weights | 14.18 GiB free |
| load wall (cold) | 30 s (31.4 s in a second cold run) |

The 8192 x 16-slot default point (≈45.4 GiB) is over the card; 8 slots is the point that fits. `PEGAINFER_MAX_CONTEXT`/`PEGAINFER_DECODE_SLOTS` trade context against concurrency — see `models/gemma4/tp.md` for the other measured envelopes.

## Method

Raw `/v1/completions`, greedy, one connection per request. Time to first token is measured directly as the wall of a `max_tokens=1` request; per-step time is `(t(65) - t(1)) / 64`, which cancels the first-token cost. One warmup request is issued first to absorb the one-time buffer/pool warm; concurrency is a thread pool of `N` over `N` requests of 64 tokens each, wall-timed as a batch.

## Results

Prompt `The capital of France is`, 5 TTFT samples and 3 TPOT samples:

| metric | value |
| --- | --- |
| first token (TTFT) | 57 ms (58, 57, 57, 57, 57) |
| decode step (TPOT) | 49.4 ms (49.4, 49.41, 49.40) |

Throughput, 64 tokens per request:

| concurrency | wall | aggregate | per request |
| ---: | ---: | ---: | ---: |
| 1 | 3.17 s | 20.2 tok/s | 3.17 s |
| 4 | 3.47 s | 73.8 tok/s | 0.87 s |
| 8 | 3.57 s | 143.5 tok/s | 0.45 s |

A decode step costs ~49 ms for one row and ~56 ms for eight (3.57 s / 64 steps), so batch width is nearly free until the collective and the wide-batch GEMMs bite.

## Eager vs captured decode

The default is captured (`--cuda-graph=true`, as at TP1); both modes measured back to back on the pair:

| mode | TTFT | step | c8 aggregate |
| --- | ---: | ---: | ---: |
| eager (`--cuda-graph=false`) | 56.5 ms | 49.4 ms | 143.7 tok/s |
| captured (default) | 57.0 ms | 51.5 ms | 146.0 tok/s |

A wash: the collective, not launch count, dominates the step, so capture buys little and costs ~4% per step at c1. Both start, serve and shut down cleanly; the numbers in the tables above are the eager run.

## Correctness

The distribution gate at 31B geometry is exact (`models/gemma4/tp.md`): the synthetic checkpoint carrying this checkpoint's shapes is bit-identical between one rank and two (48/48 picks, worst picked-token gap `0.0000`). On the real checkpoint the sane answers come out through the chat template — `What is the capital of France?` → `Paris`; `Name three primary colors.` → `The three primary colors are red, yellow, and blue.` (the sampled defaults; greedy repeats, which is the model's published non-default behavior, not a TP artifact).

## Notes

- Raw completions without `<bos>` degenerate, as the tokenizer contract states; the chat endpoint (which carries the template's BOS) is the coherent path.
- The first GEMM of a cold two-rank process would hang on this environment's NCCL before the module-loading warm was added — see "Cold start" in `models/gemma4/tp.md`. The load figure above is with that warm in place.
- Every rank holds the whole 262144-entry vocabulary projection (~2.8 GiB at 31B), and the extra ranks compute a full batch of logits per decode step that is then discarded. The step numbers above include that waste; see "Known bounds" in `models/gemma4/tp.md`.
