# Qwen3.5 Unified Prefill Overlap

> **TL;DR:** Issue #715 adds opt-in, single-GPU `--decode-overlap stream` for
> Qwen3.5. One prefill chunk may run on a second CUDA stream while active decode
> continues. The default remains serial `off`. Stream mode is currently limited
> to `--max-batch <= 32` until the bucket-64 decode GEMMs have an independent
> per-stream cuBLAS route. The original HTTP table below predates that cap and
> used the server's implicit Qwen3.5 max-batch default, so treat it as pre-cap
> evidence until the HTTP cells are re-run with an explicit safe max batch.
>
> **Last touched:** 2026-08

## Preparation

- **Read**:
  - `docs/index.md` - routes Qwen3.5 scheduler, accuracy, mixed-load, and profiling evidence.
  - `docs/models/qwen35/adaptive-scheduler-policy.md` - `off` stays the default; `auto` is a separate opt-in policy and cannot be combined with overlap yet.
  - `docs/models/qwen35/mixed-load-itl-470.md` - valid mixed load needs spare admission capacity and an observed prefill/decode intersection.
  - `docs/models/qwen35/accuracy.md` - `hf_golden_gate` is the numerical oracle; generated-text hashes are sanity evidence.
  - `docs/playbooks/bench-vs-vllm.md` and `docs/playbooks/profiling-guide.md` - bind A/B numbers and profiler claims to a fixed environment and raw artifacts.
- **Relevant history**:
  - Qwen3 PR #695 shows that a kernel-only stream override leaves allocations, copies, frees, and temporary lifetimes unordered across streams.
  - Qwen3 issue #699 shows that an in-flight prefill must participate in the scheduler idle wait or the scheduler can park forever after the last decoder retires.
  - Historical prototype `8ed45964` is design evidence only. It predates current crate names and duplicates merged scheduler/trace work with about 3,000 added lines.
- **Plan**:
  1. Add default-off `Qwen35DecodeOverlap::{Off, SharedSm}` startup plumbing and reject TP, Green Context, and `Auto + SharedSm` before model loading.
  2. Keep at most one `InflightPrefill`; run its allocations, kernels, and stream-ordered drops on one owned prefill stream, then run active decode on the existing compute stream.
  3. Poll the completion event while decode is active; block on the event when it is the only remaining work; sample and promote exactly once after completion.
  4. Validate scheduler transitions, GPU correctness and lifecycle, HTTP A/B, mixed load, and Nsight on one RTX 5090.

## Execution Log

### Implementation

- Existing public Qwen3.5 launch APIs delegate to `Qwen35DecodeOverlap::Off`.
  The server and `bench_serving` expose `--decode-overlap off|stream`.
- Shared-SM mode owns one prefill stream and at most one `InflightPrefill`. The
  model context stream and `StreamOverrideGuard` move together for the complete
  prefill enqueue, including allocations, copies, kernels, and stream-ordered
  frees.
- Prefill and decode use separate cuBLAS handles. A stream override routes N=1
  projections away from the graph-safe decode handle.
- The scheduler polls the prefill event between decode ticks. If the last
  decoder retires first, it waits for the event rather than parking on the
  submit channel. Error, unwind, and shutdown paths drain the prefill stream
  before KV, recurrent, or convolution state can be released.
- `ITL_STEP` distinguishes `overlap_launch`, `overlap_decode`,
  `overlap_complete`, and `overlap_wait`. The aggregator counts each forwarded
  chunk once and keeps concurrent actions out of serial prefill-freeze
  statistics.

Unsupported combinations fail before model loading:

- Qwen3.5 TP plus overlap;
- `--qwen35-scheduler-policy auto --decode-overlap stream`;
- Qwen3.5 `--decode-overlap green-ctx`;
- Qwen3.5 `--decode-overlap stream` with `--max-batch > 32`.

### Review Fixes

- Lifetime review found that the first `InflightPrefill` layout released request
  state before its stream-draining owner during unwind. `AsyncPrefillOutput` is
  now the first field, so it drains before `ScheduledChunk` returns KV pages or
  releases recurrent/convolution buffers.
- Oracle review found that in-flight decode ticks were missing from
  `OPENINFER_ITL_DEBUG`, while the launch tick was mislabeled as a serial
  Unified stall. Mode-aware plan kinds now retain every overlap tick without
  classifying it as a freeze.
- Completion sampling and state promotion initially ran before the timed step.
  They now emit `overlap_complete` with the active decode width, so the trace
  includes host work that can delay the next decode action.
- The aggregator initially counted the same chunk again on every in-flight
  action. Exact plan classification now counts only `prefill`, `unified`, or
  `overlap_launch` as a forwarded chunk.
- The first RNG refactor changed default-Off sampled-output progression. The
  final path preserves the historical sequence: a successful single-GPU pure
  decode consumes the same two scheduler values as current main; TP consumes
  the first value.
- The lifecycle oracle needed three corrections:
  - It first assumed the direct scheduler always emits `Scheduled` before a
    token. A token may arrive first, so the receiver now accepts both orders.
  - A four-chunk prefill let serial Unified produce one decode token after each
    chunk, so token progress alone did not prove overlap. The final oracle uses
    one 8192-token chunk and requires two new decode tokens before the prefill's
    first token; serial Unified can produce only one decode token at that bound.
  - It also proves that cancelling the last decoder drains and promotes the
    in-flight prefill, a post-overlap request succeeds, and final-handle drop
    returns while another prefill is in flight. Every critical receive has a
    30-second deadline; the final debug gate observed two `overlap_wait`
    actions.
- Shared-SM aggregation initially omitted `decode_n` unless a serial Unified
  stall existed, which made the mixed-load validity grep report `<none>`.
  Prefill-associated actions now publish their own active-width distribution;
  a synthetic overlap launch/decode/complete/wait log reports `decode_n=4`
  while counting the 8192-token chunk once.
- RLCR found that treating every `StreamOverrideGuard` as prefill also disabled
  graph-safe N=1 GEMMs for existing Qwen3 decode/Green Context overrides. A
  nested, RAII-restored prefill marker now narrows that routing rule to Qwen3.5
  async prefill. Its unit test and affected Qwen3/Qwen3.5 builds passed, and the
  Shared-SM lifecycle gate still observed two `overlap_wait` actions.
- Follow-up review after the PegaInfer rename fixed the `bench_serving` Qwen3.5
  crate path, made the `MAX_SHARED_SM_DECODE_BATCH <= GEMM_LT_MAX_N` relationship
  a compile-time assertion, and added a back-reference at the decode GEMM tuning
  filter. It also removed the dead prefill cuBLAS workspace allocation: cuBLAS
  resets a handle's workspace when `cublasSetStream()` is called, so the safe
  bucket cap is about avoiding undocumented concurrent use of one cuBLAS handle
  across decode replay and async prefill until bucket-64 has a per-stream route.
- The HTTP A/B table is now explicitly marked pre-cap evidence. The archived
  server launch command was not retained, and the old source defaulted omitted
  Qwen3.5 `--max-batch` to 64. Current `stream` mode rejects that default, so
  cite the HTTP numbers only after a re-run with explicit `--max-batch <= 32`.

## Verification Contract

| Field | Value |
| --- | --- |
| GPU | 1x NVIDIA GeForce RTX 5090, 32 GB |
| Driver / CUDA toolkit | `595.71.05` / `12.8.93`, target `sm_120` |
| Rust / Triton build env | nightly 2026-07-10 / Triton 3.7.1 |
| Source | benchmark binaries: `c405b556` on upstream/main `e3f91120`; the RLCR follow-up only narrows generic-vs-prefill override classification and was separately GPU-lifecycle verified |
| Model | `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| HTTP client | vLLM 0.25.1 `bench serve` only; no vLLM server was used |
| Server binary sha256 | `5f19998277f4be7577846cf3e78bc3fef43151a5add0b0a56dd443a06d39b091` |
| `bench_serving` sha256 | `cf5040aa4bbad60011a3fa3799f23d8aa2bae8ef399f35bd5a7872a7d98c1936` |
| Current HTTP rerun requirement | Use `--decode-overlap stream --max-batch <= 32`; the server default is 64 for Qwen3.5 and is intentionally rejected in stream mode |

Qwen3.5 does not expose product prefix reuse or accept
`--no-prefix-cache`; the CLI rejects it as an unused option. The primary result
therefore contains no prefix-cache path, and the verified commands omit the
flag.

### Correctness and lifecycle

- Release Qwen3.5 library tests: 73 passed, 6 expected TP2 tests ignored.
- Shared-SM lifecycle e2e: 1 passed; the required ITL gate observed two
  `overlap_wait` actions.
- Default-Off scheduler e2e: 1 passed, 1 TP2 test ignored.
- `hf_golden_gate`: 2 passed, 2 TP2 tests ignored. Long-golden logprob delta
  mean/p99/max was `0.0225/0.0718/0.0968`; short graph mean was
  `0.0253-0.0281`, max `0.1394-0.2403`.
- Release server and `bench_serving` builds passed. Startup rejection probes
  passed for TP, `Auto + SharedSm`, and Green Context.

### HTTP A/B (pre-cap evidence; needs explicit max-batch rerun)

The same release binary served both modes. Each round ran `off` then `stream`
with a 15-second cooldown. Fixed-concurrency cells used request rate `inf`;
QPS cells used max concurrency 64. Shared client flags were:

```bash
vllm bench serve \
  --backend openai --endpoint /v1/completions \
  --model qwen35-715 --served-model-name qwen35-715 \
  --tokenizer <model> --dataset-name random \
  --random-input-len 1024 --random-output-len <256|128> \
  --num-prompts <count> --request-rate <inf|8|12|16> \
  --max-concurrency <1|16|64> --ignore-eos --temperature 0 --seed 42 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
  --save-result --save-detailed
```

The exact historical server launch command was not retained in the archived
notes. Inspecting the benchmark source commit shows Qwen3.5 server startup
filled an omitted `--max-batch` from Qwen3.5 `MAX_DECODE_BATCH`, so omitting
`--max-batch` meant the HTTP table used the implicit 64-slot default.
Current stream mode rejects that shape; use commands like these for a current
HTTP rerun:

```bash
cargo build --release -p pegainfer-server --features qwen35

target/release/pegainfer \
  --model-path <model> --served-model-name qwen35-715 --port <port> \
  --cuda-graph=true --qwen35-scheduler-policy off \
  --max-batch 32 --max-prefill-tokens 1024 \
  --decode-overlap <off|stream>
```

Median of three historical runs. `ok/fail` is the total across all three. TPOT is
the median run's mean TPOT; the latency columns retain p50/p99 from the median
run. These rows are retained to show the original direction and output sanity,
not as the current post-cap HTTP result.

| Cell | mode | ok/fail | observed avg in/out | TTFT p50/p99 ms | TPOT mean/p99 ms | ITL p50/p99 ms | output tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024/256 c1 | off | 12/0 | 980.2/256 | 53.06/55.58 | 6.99/7.03 | 6.97/8.01 | 139.1 |
| 1024/256 c1 | stream | 12/0 | 980.2/256 | 54.58/59.24 | 7.07/7.11 | 7.05/7.78 | 137.6 |
| 1024/256 c16 | off | 48/0 | 973.7/256 | 569.74/1061.79 | 11.48/13.17 | 9.81/72.01 | 1147.9 |
| 1024/256 c16 | stream | 48/0 | 973.7/256 | 558.49/1084.71 | 11.13/12.55 | 9.79/56.70 | 1149.4 |
| 1024/128 QPS 8 | off | 48/0 | 973.7/128 | 73.68/197.11 | 12.12/14.73 | 9.69/68.44 | 622.5 |
| 1024/128 QPS 8 | stream | 48/0 | 973.7/128 | 75.62/200.71 | 11.31/13.28 | 9.63/59.88 | 624.9 |
| 1024/128 QPS 12 | off | 72/0 | 983.9/128 | 142.59/388.82 | 14.85/18.93 | 9.79/73.60 | 848.8 |
| 1024/128 QPS 12 | stream | 72/0 | 983.9/128 | 153.35/405.02 | 13.70/16.92 | 9.85/63.12 | 846.4 |
| 1024/128 QPS 16 | off | 96/0 | 983.2/128 | 251.95/633.34 | 16.88/22.87 | 9.88/73.94 | 1057.8 |
| 1024/128 QPS 16 | stream | 96/0 | 983.2/128 | 267.50/667.90 | 15.52/20.34 | 10.02/62.65 | 1054.3 |

Every run completed with zero failures, timeouts, error strings, or zero-token
outputs. Every request produced exactly 256 or 128 tokens. The saved detailed
JSON contains all generated text; short sha256 values over each run's combined
text are retained below.

| Cell | off hashes, rounds 1/2/3 | stream hashes, rounds 1/2/3 |
| --- | --- | --- |
| c1 | `04bcd42f10baa890`, `3421ef078cf3d773`, `04bcd42f10baa890` | `412d5660f743ebe0`, `412d5660f743ebe0`, `04bcd42f10baa890` |
| c16 | `c75c098ef30a0ff6`, `2d4500ae44c7d0a2`, `7f2e24296c2ea7ed` | `ea3ebc8e9588a2b2`, `00ca54985f6f659a`, `af8f43ce6c7751fe` |
| QPS 8 | `195843022462af6c`, `87177a5fc918381e`, `1e77b081acf3afd2` | `123e3c0d8bfb50b6`, `cc45e46d1de7d057`, `195843022462af6c` |
| QPS 12 | `4b9032e8a6d680e3`, `fd6608b46bb8cbf3`, `92af3f26065940e1` | `edf4960eb3c9def5`, `cc14d9a7748adf81`, `98c99203f184f11d` |
| QPS 16 | `440c62bb49fd70f9`, `1d413287575f15f3`, `05ea0f131d26655e` | `72d18e2f184f7111`, `220b321e6b6d3c9f`, `24134af1bf0f9e5f` |

An independent 128-token HTTP sanity request completed in every server block.
Its two deterministic text variants had hashes `3a225a2290dd7ad1` and
`480c2617c7ed462e` in both modes.

In this pre-cap table, Stream improved c16 mean TPOT in all three paired rounds:
the three-run median was `11.48 -> 11.13 ms` (`-3.04%`). QPS 8/12/16 improved
`6.68%/7.74%/8.05%`. c1 moved `6.99 -> 7.07 ms` (`+1.20%`), and QPS TTFT p50
increased by about 2-16 ms. Re-run these cells with explicit `--max-batch <= 32`
before citing them as current HTTP performance evidence.

### Serving plan trace and direct diagnostic

Across the six HTTP server blocks, the Off trace contained 6,249 Decode, 252
Unified, and 27 Prefill ticks. Stream replaced the 252 Unified ticks with 252
`overlap_launch`, 369 `overlap_decode`, and 252 `overlap_complete` ticks; it had
zero serial Unified stalls. Every launch had exactly one completion. Real
active widths reached 32. Rounded CUDA Graph bucket counts were:

| mode | bs1 | bs2 | bs4 | bs8 | bs16 | bs32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 3800 | 117 | 138 | 297 | 1554 | 595 |
| stream | 3840 | 151 | 219 | 415 | 1651 | 556 |

A separate in-process 1024/256 c16 diagnostic, three measured iterations and
48 completed requests per mode, measured steady TPOT avg/p50/p99 at
`11.06/9.82/58.27 ms` Off and `10.77/9.86/29.12 ms` Stream. HTTP minus direct
mean TPOT was `0.42 ms` Off and `0.36 ms` Stream. This is an attribution check, not an
HTTP performance result; it shows that the overlap improvement did not create
a new frontend/output-path gap.

### Mixed-load regression gate

The valid gate used `max_batch=8`, four 512/4096 background decoders, ten cold
4096/1 injections at QPS 0.5, warmup 5, and three repetitions. Every injection
intersected `decode_n=4`, completed with one output token, and retained its hash.

| mode | baseline p50/p99 ms | mixed p50/p99/max ms | injection prefill median ms | serial stalls |
| --- | ---: | ---: | ---: | ---: |
| off | 7.83/8.59 | 7.80/58.23/60.89 | 237.41 | 52-53 per run |
| stream | 7.83/8.61 | 7.82/26.15/41.30 | 240.66 | 0 |

Stream reduced mixed p99 by 55.1% and median-run max by 32.2%, while injection
prefill rose 1.4%. The intentional `max_batch=4,bg=4` starvation control was
invalid as expected: `decode_n` reached only 1-3, never 4; one injection waited
33.9 seconds for a slot, and the tool emitted slot-starvation, pacing, and
background-exhaustion warnings. It is retained only as a measurement guard.

### Nsight Systems

Nsight Systems 2025.1.1 captured matched Off and Stream mixed-load runs with:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx \
  --cuda-graph-trace=node --export=sqlite -o <trace> \
  target/release/bench_serving --model-path <model> \
  --max-batch 8 --decode-overlap <off|stream> \
  --format json --out <result> mixed --bg-prompt-len 512 \
  --bg-concurrency 4 --bg-output-len 2048 --inj-prompt-len 4096 \
  --inj-output-len 1 --qps 0.5 --num-injections 3 \
  --inj-warm-frac 0 --warmup 3 --skip-baseline
```

Off placed prefill and decode on CUDA stream 13. Stream placed active prefill on
stream 14 and decode on stream 13. Within the prefill-active window, the SQLite
timeline contained 28,082 cross-stream kernel-overlap pairs totaling 321.4 ms;
examples include prefill GEMMs overlapping `conv1d_decode_batch_kernel` and
`gated_delta_rule_decode_batch_kernel`. This proves real simultaneous kernel
execution, beyond host launch interleave. Profiler timings are not used in the
HTTP tables because CUDA Graph node tracing inflates absolute time.

`cuda_gpu_kern_sum`, `cuda_api_sum`, and `tools/nsys_tail_stats.py` were all run
on the Stream SQLite. The top GPU totals remained projection GEMMs; the host
table was dominated by expected event synchronization and whole-process model
load traffic. No new kernel-first optimization is justified by this trace.

## Debrief

- **Outcome:** #715 has a compact opt-in Shared-SM path, explicit unsupported
  combinations, lifecycle-safe cleanup, mode-aware ITL diagnostics, real GPU
  correctness/lifecycle coverage, repeated HTTP improvement, a valid mixed-load
  gate, and profiler proof of two-stream kernel overlap.
- **Subtraction:** the implementation reuses the existing Unified scheduler,
  request state, paged KV ownership, recurrent/conv state, CUDA Graph decode,
  and `StreamOverrideGuard`. It does not add a second scheduler, duplicate model
  state, a new chunk policy, or a speculative abstraction.
- **Decision:** keep `off` as the default. Stream helps concurrent admission and
  QPS TPOT but slightly regresses c1 and increases loaded TTFT. `auto`, TP, and
  Green Context remain separate policies with explicit startup errors.
- **Claim boundary:** evidence covers this source, model revision, and one RTX
  5090. It does not claim vLLM parity, SOTA, production readiness, TP overlap,
  prefix caching, or broad hardware results.
