# Kimi-K2 TP1 DP8 EP8 performance

> TL;DR: This ledger tracks pegainfer TP1+DP8+EP8 on 8x H20 against the vLLM TP1+DP8+EP8 bs64 target. The vLLM sustained bs64 `~106ms` TPOT is now explained by a DPLB/CUDA-graph bucket cliff: an uneven DP distribution such as `9,8,8,8,8,8,8,7` pads every rank from graph bucket 8 to 16 and doubles TPOT. O2 landed five production decode-kernel picks (cuBLASLt fixed-shape shared_gate_up / o_proj / MLA strided-batch, split-vocab argmax, fused router selector); accuracy held at the bf16 ULP floor by a base-vs-opt prefill logits A/B, and the PPLX Marlin small-N tile was identified as the messy branch's real accuracy break (`-inf` logits + SIGSEGV at small per-rank N) and rejected. bs64 TPOT is unchanged within noise (p50 `40.58 -> 40.09ms`): the per-kernel wins do not resolve above the ±1ms band at this shape. Every pegainfer optimization must start from a profile, state the expected gain, show a microbench or isolated measurement, then pass correctness and service-level gates before commit.
>
> Last touched: 2026-06-07

## Target

| Item | Target |
| --- | ---: |
| Hardware | 8x NVIDIA H20 |
| Model | `$MODEL_DIR` |
| Shape | TP1 DP8 EP8 |
| Workload | prompt_len=1, output_len=128, max_concurrency=64, num_prompts=256 |
| vLLM baseline | output `594.57 tok/s`, TTFT p50/p99 `161.30/303.20ms`, TPOT p50/p99 `107.20/109.20ms`, ITL p50 `108.92ms` |
| Gate | `256/256` success, TPOT p50 `< 107.20ms`, TPOT p99 `< 109.20ms`, output `> 594.57 tok/s` |

Baseline source: H20 rerun with explicit bs64 warmup on 2026-05-25:
`$RESULT_ROOT/kimi-vllm-dp8-warmup-20260525/measure_bs64_o128_after_warmup.json`.
The older sweep in `docs/models/kimi-k2/vllm-h20-baseline.md` recorded bs64 TPOT p50/p99
`109.00/109.76ms`; the warmup-after rerun is slightly faster but still the same
100ms-class H20 shape.

The gate above is the sustained `num_prompts=256, max_concurrency=64` client shape,
not a one-shot 64-request pure-decode wave. A separate command audit on 2026-05-25
showed that vLLM can report `~50ms` TPOT for a single 64-request wave, then return to
`~106ms` TPOT when the benchmark continuously refills another 192 requests. Treat these
as different workloads.

## Method

Performance work in this file follows this loop:

1. Profile: record the service JSON/log, in-process JSON, and nsys sqlite/tail report when profiling is needed.
2. Motivation and expected gain: name the bottleneck and estimate the target metric movement.
3. Microbench: isolate the changed stage, or explain why the service/in-process measurement is the smallest meaningful unit.
4. Correctness: keep generated-token hash distributions, mismatch counts, and any relaxed tolerance rationale.
5. Decision: keep, reject, defer, or revert; every kept optimization gets a commit.

For TP1 DP8, correctness checks must include uneven per-rank active rows and empty-rank EP participation, because PPLX collectives still require all ranks to enter each MoE layer in the same order.

## Unified Commands

Path placeholders:

```bash
export PEGAINFER_DIR=/path/to/pegainfer
export VLLM_DIR=/path/to/vllm_test
export MODEL_DIR=/path/to/Kimi-K2.5
export NCCL_LIB_DIR=/path/to/nccl-lib
export EVAL_VENV=/path/to/eval-venv
export RESULT_ROOT=/path/to/result-root
export TRITON_PYTHON=$PEGAINFER_DIR/.triton-venv/bin/python
```

Build on an <H20_NODE>:

```bash
cd "$PEGAINFER_DIR"
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH="$NCCL_LIB_DIR:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON="$TRITON_PYTHON" \
cargo build --release -p pegainfer-server \
  --features kimi-k2 --bin pegainfer --bin bench_serving
```

(The old `kimi-k2-pplx-ep` feature and `PEGAINFER_KIMI_PARALLEL` env existed only on the
pre-merge branch; on main the feature is `kimi-k2` and parallel shape is selected by the
`--tp-size/--dp-size/--ep-backend` CLI flags below. nvcc must also be on `PATH` — the
`pegainfer-comm` cc-rs build looks it up there, not via `$NVCC`.)

In-process bs64:

```bash
cd "$PEGAINFER_DIR"
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH="$NCCL_LIB_DIR:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON="$TRITON_PYTHON" \
target/release/bench_serving \
  --model-path "$MODEL_DIR" \
  --cuda-graph false \
  --tp-size 1 --dp-size 8 --ep-backend pplx \
  --format json \
  --out "$RESULT_ROOT/kimi-tp1dp8/tp1dp8_bs64_o128_${COMMIT}.json" \
  request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 1 --iters 1
```

Service bs64, same client shape as vLLM:

```bash
cd "$PEGAINFER_DIR"
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH="$NCCL_LIB_DIR:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON="$TRITON_PYTHON" \
target/release/pegainfer --model-path "$MODEL_DIR" --served-model-name kimi-k2.5 \
  --port 8124 --cuda-graph false --tp-size 1 --dp-size 8 --ep-backend pplx
```

```bash
source "$VLLM_DIR/.venv/bin/activate"
vllm bench serve \
  --backend openai \
  --model "$MODEL_DIR" \
  --tokenizer "$MODEL_DIR" \
  --trust-remote-code \
  --base-url http://127.0.0.1:8124 \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 1 \
  --random-output-len 128 \
  --random-range-ratio 0 \
  --num-prompts 256 \
  --max-concurrency 64 \
  --request-rate inf \
  --ignore-eos \
  --temperature 0 \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --save-result \
  --save-detailed \
  --result-dir "$RESULT_ROOT/kimi-tp1dp8-service" \
  --result-filename pegainfer_tp1dp8_bs64_${COMMIT}.json
```

GSM8K accuracy smoke, concurrent OpenAI `/v1/completions` path:

```bash
cd "$PEGAINFER_DIR"
source "$EVAL_VENV/bin/activate"
lm_eval run --model local-completions \
  --model_args "model=kimi-k2.5,base_url=http://127.0.0.1:8125/v1/completions,tokenizer_backend=huggingface,tokenizer=$MODEL_DIR,tokenized_requests=False,trust_remote_code=True,max_length=4096,max_gen_toks=256,num_concurrent=16,timeout=300" \
  --tasks gsm8k --num_fewshot 8 --batch_size 1 --limit 64 \
  --output_path "$RESULT_ROOT/kimi-tp1dp8-gsm8k-lm-eval-${COMMIT}-limit64-c16" \
  --log_samples
```

vLLM TP1 DP8 EP8 baseline server:

```bash
cd "$VLLM_DIR"
source .venv/bin/activate
vllm serve "$MODEL_DIR" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --api-server-count 1 \
  --served-model-name kimi-k2.5 \
  --port 8123 \
  --max-num-seqs 64 \
  --max-model-len 4096
```

Use the served model name on the client. vLLM 0.19.0 returns 404 for
`--model $MODEL_DIR` in the single-API-server setup.

```bash
cd "$VLLM_DIR"
source .venv/bin/activate
vllm bench serve \
  --backend openai \
  --model kimi-k2.5 \
  --tokenizer "$MODEL_DIR" \
  --trust-remote-code \
  --base-url http://127.0.0.1:8123 \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 1 \
  --random-output-len 128 \
  --random-range-ratio 0 \
  --num-prompts 256 \
  --max-concurrency 64 \
  --request-rate inf \
  --ignore-eos \
  --temperature 0 \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --save-result \
  --save-detailed \
  --result-dir "$RESULT_ROOT/kimi-vllm-dp8-cmdcheck-20260525" \
  --result-filename api1_maxseq64_measure_bs64_o128_after_warmup_modelname.json
```

nsys profile:

```bash
cd "$PEGAINFER_DIR"
mkdir -p "$RESULT_ROOT/kimi-profile"
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH="$NCCL_LIB_DIR:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON="$TRITON_PYTHON" \
nsys profile --force-overwrite=true --trace=cuda,nvtx \
  --cuda-graph-trace=node --export=sqlite \
  -o "$RESULT_ROOT/kimi-profile/tp1dp8_bs64_o128_${COMMIT}" \
  target/release/bench_serving \
    --model-path "$MODEL_DIR" \
    --tp-size 1 --dp-size 8 --ep-backend pplx \
    --cuda-graph false \
    --cuda-profiler-capture \
    --format json \
    --out "$RESULT_ROOT/kimi-profile/tp1dp8_bs64_o128_${COMMIT}.json" \
    request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 1 --iters 1

uv run --no-project python tools/nsys_tail_stats.py \
  "$RESULT_ROOT/kimi-profile/tp1dp8_bs64_o128_${COMMIT}.sqlite" \
  --out "$RESULT_ROOT/kimi-profile/tp1dp8_bs64_o128_${COMMIT}_tail.md"
```

## Optimization Log

### O1 - prompt_len=1 admission goes through decode

Status: keep. Baseline implementation: `8946078`. Safety follow-ups: `64192bb`, `0c23389`.

Profile:

- Code inspection showed TP1 DP8 uses `DpCoordinator`, not the TP8 `KimiK2Scheduler` prompt_len1 batch path.
- Old admission ran each prompt_len=1 request through `synchronized_prefill`, with `decode_batch_size=1`, and padding ranks doing dummy prefill. At bs64 that is 64 synchronized prefill waves.
- Old `MAX_BATCH_PER_DP=4` capped global active requests at 32, so bs64 could not occupy all requested rows.

Motivation and expected gain:

- prompt_len=1 is semantically a decode step at position 0: consume one token, append KV at position 0, produce the first generated token.
- Replace 64 serialized prompt prefill waves with one DP-wide decode admission wave.
- Raise per-DP slots to 8 so TP1 DP8 can hold the full bs64 workload.
- Expected gain: large TTFT reduction and service throughput improvement; TPOT should use rank-local bs8 instead of two bs32 waves.

Change:

- `pegainfer-kimi-k2/src/runner/engine.rs`
  - `MAX_BATCH_PER_DP: 4 -> 8`.
  - Added prompt_len1 admission batching in `DpCoordinator`.
  - For prompt_len1 requests, send `StepCommand::Decode { positions: vec![0], slots, decode_batch_size: MAX_BATCH_PER_DP }` instead of `Prefill`.
  - Empty ranks still run padding decode with the same arena capacity to preserve PPLX collective order.
  - Existing active rows are included in the same prompt_len1 admission decode command; padding rows can only use free slots.
  - Ordinary prefill padding ranks write the dummy token into a free slot, not fixed slot 0. If any rank lacks a safe padding slot, that request remains pending.

Correctness constraints:

- In TP1 DP8, `decode_batch_size` means decode arena capacity, not active row count. Keep it fixed at `MAX_BATCH_PER_DP` for decode, prompt_len1 admission, padding decode, and ordinary prefill.
- Slot IDs are decode arena row IDs. A request must keep the same arena bucket for prefill and all decode steps, otherwise its KV cache lives in a different arena.
- PPLX decode scratch capacity must be identical across ranks even when active row counts differ.
- Padding decode and padding prefill execute real kernels and can write KV. They may only target unoccupied slots.
- Every synchronized step must drain one result from every DP rank, including the error path, before the next command is sent.
- Padding prefill failures are request failures; the owner request must not become active unless every rank completed its synchronized prefill step.
- A missing rank forward thread is fatal for the process. Continuing with a partial DP command would leave surviving ranks inside unmatched PPLX collectives.
- prompt_len1 admission at `append_position=0` must install request state after the first token, or finish/error the request in the same result pass.

Microbench:

- Remote build passed on H20 node at `0c23389`.
- Smoke command:

```bash
target/release/bench_serving \
  --model-path $MODEL_DIR \
  --tp-size 1 --dp-size 8 --ep-backend pplx \
  --cuda-graph false \
  --format json \
  --out $RESULT_ROOT/kimi-tp1dp8/tp1dp8_bs64_o5_64192bb_smoke.json \
  request --prompt-len 1 --output-len 5 --concurrency 64 --warmup 0 --iters 1
```

- Smoke result after stable-arena safety fix: `64/64` success,
  `steady_tpot_ms` p50/p95/p99 `37.21/37.41/37.42ms`, first decode step p50 `38.47ms`.

Correctness:

- Smoke generated all 5 tokens for every request without PPLX collective mismatch or slot-state failure.
- bs8/o8 deterministic smoke generated `8/8` full traces with one hash,
  `$RESULT_ROOT/kimi-tp1dp8/prompt1_decode_admission_bs8_o8_correctness.json`.
- Scope: this proves scheduler/collective/slot safety for the prompt_len1 decode-admission path.
  It is not a full TP1 DP8 token-parity gate against vLLM or TP8 DP1; that reference trace still
  needs explicit mismatch counts before this shape becomes an accuracy baseline.
- GSM8K lm-eval smoke on H20 node at `f193af2`, TP1 DP8 service,
  `num_concurrent=16`, `limit=64`, `num_fewshot=8`, `max_gen_toks=256`:
  strict-match and flexible-extract both `55/64 = 0.8594` (`stderr 0.0438`).
  Artifacts:
  `$RESULT_ROOT/kimi-tp1dp8-gsm8k-lm-eval-f193af2-limit64-c16/kimi-k2.5/results_2026-05-25T16-02-38.986675.json`
  and `samples_gsm8k_2026-05-25T16-02-38.986675.jsonl`.
- Local coordinator tests cover sparse logical slots, prompt_len1 admission mixed with active rows,
  padding decode arena capacity, and ordinary prefill padding slot selection:

```bash
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
cargo test -r -p pegainfer-kimi-k2 --features pplx-ep runner::engine::tests --no-fail-fast
```

- Local result: `5 passed`.
- H20 result at `0c23389`: `5 passed`.
- Mixed-arrival service test, `$RESULT_ROOT/kimi-tp1dp8-service/pegainfer_tp1dp8_mixed_arrival_prompt1_o64_0c23389.json`:
  `64/64` success with `--request-rate 16`, peak concurrent requests `54`, TTFT p50/p99
  `58.10/110.88ms`, TPOT p50/p99 `35.91/37.63ms`. This covers prompt_len1
  admissions landing while existing decode slots are active.

Performance:

- In-process, `$RESULT_ROOT/kimi-tp1dp8/tp1dp8_bs64_o128_0c23389_w1_i1.json`:
  `64/64` success, TTFT p50/p99 `74.62/77.19ms`, first decode p50/p99
  `38.23/38.24ms`, steady TPOT p50/p95/p99 `40.10/43.32/43.72ms`.
- Service, same `vllm bench serve` client as vLLM,
  `$RESULT_ROOT/kimi-tp1dp8-service/pegainfer_tp1dp8_bs64_o128_0c23389_after_warmup.json`:
  `256/256` success, output `1336.35 tok/s`, TTFT p50/p99 `105.31/127.81ms`,
  TPOT p50/p95/p99 `47.34/47.70/47.71ms`, ITL p50/p99 `47.84/50.69ms`.
- vLLM warmup-after baseline,
  `$RESULT_ROOT/kimi-vllm-dp8-warmup-20260525/measure_bs64_o128_after_warmup.json`:
  `256/256` success, output `594.57 tok/s`, TTFT p50/p99 `161.30/303.20ms`,
  TPOT p50/p95/p99 `107.20/109.00/109.20ms`, ITL p50/p99 `108.92/116.35ms`.

vLLM baseline diagnosis, H20 node, vLLM `0.19.0`, NCCL/AgRs path:

- Startup sanity: `max_seq_len=4096` is confirmed in the log; active context is only
  about 129 tokens. Workers use `nccl==2.27.5` and `AgRsAll2AllManager`. `pplx` is
  removed/falls back in this vLLM, and DBO requires DeepEP backends, so neither is a
  valid NCCL baseline knob here.
- Command sanity: use `--api-server-count 1`, `--max-num-seqs 64`, and client
  `--model kimi-k2.5`. This removes API-process routing noise and lowers graph
  capture from largest `512` to `128`, but does not by itself fix sustained TPOT.
- Single-wave vs sustained: the same server reports `50.45/50.46ms` TPOT p50/p99
  for one 64-request wave, but `106.92/108.73ms` for sustained
  `num_prompts=256,max_concurrency=64`.

Pinned DP-rank controls explain the cliff:

| Run | DP-rank distribution | Global output | TPOT p50/p99 | Artifact |
| --- | --- | ---: | ---: | --- |
| balanced | `8,8,8,8,8,8,8,8` | `1192.22 tok/s` | `48.41/48.95ms` | `$RESULT_ROOT/kimi-vllm-dp8-dplb-20260525/balanced_8x8/` |
| one-rank over bucket | `9,8,8,8,8,8,8,7` | `640.94 tok/s` | `96.01/97.34ms` | `$RESULT_ROOT/kimi-vllm-dp8-dplb-20260525/skew_98888887/` |
| observed-like skew | `8,9,9,9,8,7,7,7` | `612.12 tok/s` | `99.80/99.99ms` | `$RESULT_ROOT/kimi-vllm-dp8-dplb-20260525/skew_89998777/` |

Mechanism:

- vLLM DPLB minimizes `waiting * 4 + running`
  (`vllm/v1/engine/core_client.py:1337-1360`) and refreshes local counts from
  coordinator stats (`core_client.py:1263-1274`). Sustained refill logs show small
  imbalances such as `8,9,9,9,8,7,7,7`, `11,7,7,7,7,8,9,8`, and
  `10,9,8,7,7,7,7,9`.
- CUDA Graph dispatch pads non-exact sizes to the next captured bucket
  (`vllm/v1/cudagraph_dispatcher.py:71-90,140-151`), so local batch `8` uses bucket
  8 and local batch `9` uses bucket 16.
- DP coordination pads every rank to the maximum padded size when CUDA Graph is
  active (`vllm/v1/worker/dp_utils.py:78-88,148-160`;
  `gpu_model_runner.py:3616-3637`, verified on an H20 node
  `$VLLM_DIR/.venv/lib/python3.10/site-packages/vllm/v1/worker/gpu_model_runner.py`).
  One rank at 9 therefore makes the whole DP group execute bucket 16.

Decision for vLLM interpretation: the surprising 2x TPOT is a DPLB plus DP CUDA
Graph padding cliff. Out-of-box sustained serving is correctly reported as
`~106ms`; balanced pinned capability at bs64 is `~48-50ms`. A vLLM-side fix needs
bucket-aware DP routing or explicit router/header assignment for controlled bs64
benchmarks.

Decision:

- Keep as the current H20 bs64 performance baseline. O1 moves prompt_len=1 onto the decode
  shape and clears the vLLM bs64 TPOT/output gate; full token-parity correctness remains a
  separate reference gate before using TP1 DP8 as an accuracy baseline. Follow-up profiles should
  focus on lowering pegainfer service TPOT from `47ms` toward the H200-reported 30ms-class
  expectation if that target is confirmed on comparable hardware.

### O2 - decode kernel cherry-pick: cuBLASLt fixed-shape GEMMs, argmax split, router fusion

Status: keep (5 commits). One candidate rejected as a real accuracy break (below).

Context: these kernels were developed together on one branch (`opt/kimi-tp1-dp8-decode`)
along with ~2.4k lines of microbench scaffolding, and that branch produced wrong output
with no obvious culprit. Salvage was done by production-only cherry-picks onto a clean
base (`3bec64f`, kimi code identical to main `927a00c`), validating after every pick;
docs and scaffolding were not picked.

Landed commits (microbench numbers from each commit message; H20, TP1 PPLX, bs=8, ctx=1):

| commit | opt | isolated microbench |
| --- | --- | --- |
| `257c9f4` | shared_gate_up cuBLASLt fixed-shape | `1.818ms -> 1.505ms` (1.21x) |
| `fc9327c` | attention o_proj cuBLASLt fixed-shape | `2.715ms -> 2.374ms` (1.14x) |
| `f77f729` | MLA absorb/v_up cuBLASLt strided batches | absorb `973.6us -> 748.5us` (1.30x) |
| `0d52e73` | final argmax split-vocab reduction | `125.3us -> 12.7us` (9.85x) |
| `8cf932f` | router post-GEMM fused score+topk selector | `3.655ms -> 3.514ms` (1.04x) |

Per-pick validation (in-process bench_serving, TP1 DP8 PPLX, cuda-graph false):

- c1 o128 determinism A/B vs base (concurrency=1 is bitwise reproducible; bs64 is not):
  all five picks stay coherent (len 128, degenerate 0/64). The greedy hash legitimately
  changes at the first cuBLASLt pick (bf16 rounding tie-breaks) and then stays at
  `cb1954bb77d652fb` — the MLA, argmax, and router picks changed nothing further.
- bs64 o128 steady TPOT p50/p99: base `40.58/44.06ms` -> cumulative 5-opt
  `40.09/42.87ms`. That is inside the ±1ms run-to-run band at warmup=1/iters=1, so the
  honest end-to-end claim is "no regression": the per-kernel wins do not resolve above
  noise at this shape.

Rejected: PPLX Marlin small-N tile (messy-branch `dd69876`) — the accuracy break.

- concurrency=1: `non-finite top logit -inf` on the serving rank -> panic (exit 101).
- bs64: SIGSEGV (exit 139) with `-inf` on multiple ranks. It never produced one valid bench.
- Why it hid: its isolated microbench (`250.64 -> 161.45us`, 1.50x) never checks numerics,
  and a small-N tile only activates at small per-rank token counts — exactly the decode
  regime, not the regime perf sweeps exercise hardest. Re-land only behind a real small-N
  decode numeric gate.

Accuracy gate: base-vs-opt prefill logits A/B. GSM8K-class evals are too coarse for
ULP-level kernel drift, so the gate follows `subsystems/correctness/logits-golden-gate.md`
with base-pegainfer itself as the reference at the same TP1 DP8 PPLX config: a throwaway
(uncommitted) hook after the prefill lm_head GEMM in `runner/worker/state.rs` dumps
full-vocab bf16 logits at every prompt position for 12 fixed raw prompts (en/zh/code/math,
1..90 tokens) sent through `/v1/completions` at `max_tokens=1`, identical patch on base
`3bec64f` and the 5-opt tip.

| metric | value | reading |
| --- | --- | --- |
| positions | 236 | 12 prompts, every prefill position scored |
| bit-identical positions | 145/236 | 10/12 prompts fully untouched -> router fusion is bit-exact |
| head-token abs-dlogprob mean / p50 | 0.0355 / 0.0000 | at the bf16 reduction-order floor (Qwen3-4B floor: 0.032) |
| top-1 token delta mean / max | 0.029 / 0.283 | drift confined to the 2 prompts whose GEMM shapes engage the cuBLASLt paths |
| argmax agreement | 233/236 | |
| worst regret | 0.125 nat | = 1 ULP at logit magnitude ~16; all 3 flips are genuine ties, under the 0.20 tie tolerance |

Coverage note: prefill exercises shared_gate_up / o_proj / router. The MLA strided-batch
pick is decode-absorb-only and is covered by the c1 greedy hash being bit-identical before
and after that pick (zero argmax flips over 128 decode steps); the argmax split changes
selection only, not logit values.

Decision: keep the five production commits; reject the small-N Marlin tile until it passes
a small-N decode numeric gate. The next perf step at this shape is not more skinny-GEMM
tuning — re-profile for the dominant decode cost (collectives / MoE path) and measure with
enough iters to resolve sub-millisecond moves.

## Open Questions

- The H20 vLLM TP1 DP8 EP8 sustained-vs-balanced discrepancy is explained by the
  DPLB/CUDA-graph bucket cliff above. The remembered 30ms-class TPOT is still not
  reproduced on H20; it may have been measured on H200 or with a different vLLM
  build/version/runtime flag set.
- TP1 DP8 prompt_len1 still needs a full reference token-parity run with mismatch counts.
  The current evidence is scheduler safety plus deterministic smoke.
- `vllm bench serve` can report `max_concurrent_requests=128` while the command uses
  `--max-concurrency 64`. Source inspection shows the client semaphore is real, but
  `max_concurrent_requests` is computed from one-second buckets and counts both
  requests ending and requests starting inside the same bucket. Treat this field as a
  coarse reporting artifact for refill-heavy runs; rely on the command shape, completed
  traces, throughput, and TPOT/ITL percentiles.
