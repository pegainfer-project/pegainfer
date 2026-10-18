# Mixed-Load ITL — long prompts arriving into steady-state decode (Qwen3-4B + Qwen3.5)

**Created**: 2026-06-08

**TL;DR**: A long prompt admitted mid-decode freezes **every** active decode for
the whole prefill (the stalled inter-token gap ≈ prefill wall-time: 4k ≈490ms,
8k ≈1180ms, 12k ≈2230ms); gaps with no prefill in flight stay at baseline TPOT
(~14ms). Whether that reaches headline **ITL p99** is a *frequency* question — it
only does once stalls exceed ~1% of decode gaps, a fraction that grows with both
**QPS and prompt length**.

## Motivation

- [Issue #244](https://github.com/xiaguan/openinfer/issues/244). Qwen3-4B's
scheduler admits a pending prefill into a **unified step** when decodes are active, thus the long prefill and all active decode rows run in one forward pass, so that step's wall-time balloons to ~the prefill time and every decoding request stalls for that one inter-token gap.
- Current test in [scheduler.md](../subsystems/scheduler/scheduler.md) is a single QPS=2 random-length run; the maintainer stance is **chunked prefill is not automatically the fix — measure first**. This characterises the tail across the QPS × prompt-length × prefix-reuse space.

## Method

In-process, deterministic, single-GPU. `bench_serving mixed` is an **open-loop** driver:

- **Background (decode-heavy steady state):** N long-lived greedy decode streams
  (`ignore_eos`), kept alive for the whole run, timestamping every token.
- **Injector:** one thread submits a long prompt every `1/qps` s (greedy,
  `output_len=1` → prefill-dominated), draining each; `[submit, last-token]` is an
  *in-flight-prefill window*.
- **Metric:** background inter-token gaps, each tagged **stall** (overlaps a prefill
  window) or **steady**; reported as `mixed_itl.{all,steady,stall}` p50/p95/p99/max
  vs a decode-only **baseline**.
- **`--inj-warm-frac`** picks the cold/warm mix: cold = a distinct prompt per
  injection (real prefill; default-on prefix cache #216 would otherwise serve
  repeats as ~37ms hits and *hide* the stall); warm = a shared prompt (cache hit
  after the first). Evenly interleaved, so e.g. 0.5 = every other.

**Sizing (16 GB) for RTX5070Ti.** Admission reserves the **full** `prompt + max_tokens` KV per
request; the pool here is ~2332 blocks ≈ 18.6k tokens. So `bg_conc × (bg_prompt +
bg_out) + inj_prompt` must fit, and `bg_out` must outlast the run. **16k prompts
OOM** (prefill *activation* scratch, not KV) — **12k is the feasible ceiling** on
this card. The sweep holds background at **4-way / 512-prompt / 1024-out** so one
baseline covers every cell and 4k/8k/12k all fit.

**Thermal.** A 10k+ prefill is a heavy compute burst; a back-to-back sweep
throttles the GPU and *fabricates* saturation (12k prefill inflated 2235→4400ms).
The sweep script inserts inter-cell cooldowns(`sleep`) — the throttle-check table below
(cold prefill ≈ constant across QPS) confirms the numbers are clock-clean.

## Results — mixed-load stall, chunking off

Both lines swept with chunking **off** (whole prompt in one unified step — the
#244 condition), to characterize the raw stall. This section is the *before*:
#375's chunked prefill (cap each step at `--max-prefill-tokens`) is the fix that
bounds it — its measured effect is the per-step `max` drop noted in the
explanation, and the canonical chunked cell lives in the `mixed_itl` snapshot, not
a table here. `bench_serving mixed`,
4-way / 512-prompt / 1024-out background, greedy, RTX 5070 Ti. Cells are **ITL p99
(ms)**; `*` = saturated (`qps·prefill_s ≳ 1`: prefills run back-to-back and decode
never recovers — a throughput wall chunking can't fix; needs rate-limit / bigger
card).


### Qwen3-4B

Results — sweep (RTX 5070 Ti, Qwen3-4B, greedy, 4-way background)

**Baseline (decode-only): p50 13.7 / p99 14.7 ms.** Cells are **ITL p99 (ms)**;
`*` = saturated (prefill > 1/qps, prefills overlap, decode starves).

**qps = 0.25**
| prompt | cold | warm½ | warm |
|--------|-----:|------:|-----:|
| 4k  | 14.9 | 16.1 | 15.2 |
| 8k  | 15.1 | 14.5 | 14.8 |
| 12k | 19.4 | 19.4 | 14.9 |

**qps = 0.5**
| prompt | cold | warm½ | warm |
|--------|-----:|------:|-----:|
| 4k  | 16.7 | 14.9 | 15.2 |
| 8k  | **1161** | **28.8** | 14.6 |
| 12k | **3270\*** | **3812\*** | 19.5† |

**qps = 1.0**
| prompt | cold | warm½ | warm |
|--------|-----:|------:|-----:|
| 4k  | **482** | **482** | 24.7 |
| 8k  | **1175\*** | **1166\*** | 28.7† |
| 12k | **3796\*** | **2302\*** | 34.5† |

(warm½ = `--inj-warm-frac 0.5`; warm = 1.0. † = the only saturation is the first
injection's one-time cold cache-fill — the rest are hits, so p99 is clean.)

**Throttle-check — cold prefill median (ms), should be ≈ constant across QPS:**
| prompt | qps 0.25 | 0.5 | 1.0 |
|--------|---------:|----:|----:|
| 4k  | 494 | 497 | 493 |
| 8k  | 1213 | 1170 | 1167 |
| 12k | 2192 | 2180 | 2304 |


### Qwen3.5-4B

Results — sweep (RTX 5070 Ti, Qwen3.5-4B, greedy, 4-way background), chunking off.

**Baseline (decode-only): p50 15.1 / p99 15.7 ms.** Cells are **ITL p99 (ms)** —
every cell sits at baseline. Qwen3.5 **does** freeze decode per injection, same as
Qwen3: a pending prefill with active decodes runs as a unified step (serial prefill
then the decode graph, in one call), so each active decode eats one ~prefill-length
gap — `max` confirms it (459 / 906 / 1408ms ≈ the prefill wall, throttle-check
below).

p99 stays at baseline as a **measurement artifact**, not architectural immunity.
**Primary cause (issue [#470](https://github.com/pegainfer-project/pegainfer/issues/470)):**
`bench_serving` hardcoded `max_batch=4` for Qwen3.5, so `--bg-concurrency 4` filled
every slot and the injector could not admit until a background stream died —
injected prefill never overlapped a full decode batch (`ITL_STEP` `decode_n` never
reached 4). Secondary contributor on this older sweep: `--bg-output-len 1024` also
capped how many gaps could be stalled (~0.7%, under the p99 knee).

**Corrected post-chunking evidence** (RTX 4090, `--max-batch 8 --bg-concurrency 4`,
cold only, with starvation negative control) lives in
[`docs/models/qwen35/mixed-load-itl-470.md`](../models/qwen35/mixed-load-itl-470.md):
with a free slot, Qwen3.5 shows the same severity/frequency structure as Qwen3;
chunking bounds per-step max but can raise headline p99 at low QPS. The tables
below are retained as the historical 5070 Ti artifact sweep — do **not** cite them
as immunity.

**qps = 0.25**
| prompt | cold | warm½ | warm |
|--------|-----:|------:|-----:|
| 4k  | 16 | 15 | 15 |
| 8k  | 16 | 16 | 16 |
| 12k | 16 | 15 | 16 |

**qps = 0.5**
| prompt | cold | warm½ | warm |
|--------|-----:|------:|-----:|
| 4k  | 17 | 16 | 16 |
| 8k  | 16 | 16 | 16 |
| 12k | 16 | 15 | 16 |

**qps = 1.0**
| prompt | cold | warm½ | warm |
|--------|-----:|------:|-----:|
| 4k  | 16 | 16 | 15 |
| 8k  | 15 | 16 | 16 |
| 12k | 16 | 16 | 16 |

**Throttle-check — cold prefill median (ms):**
| prompt | qps 0.25 | 0.5 | 1.0 |
|--------|---------:|----:|----:|
| 4k  | 459 | 444 | 441 |
| 8k  | 906 | 898 | 900 |
| 12k | 1408 | 1381 | 1422 |

## Explanation

Two independent knobs explain every cell:

1. **Severity = the prefill wall-time** (throttle-check row): the one stalled gap ≈
   the entire prefill. Scales ~linearly with prompt (4k→8k→12k ≈ 0.5→1.2→2.2s).

2. **Frequency decides if it reaches p99** = stall-gap fraction
   `≈ qps / (qps + (1−qps·prefill_s)/TPOT)`. It rises with *both* QPS and prompt
   length (a long stall eats decode time, inflating its own share). p99 ≈ baseline
   while frac < ~1%, and climbs toward the per-event stall above it. So the
   **p99-break frontier moves left (lower QPS) as prompts grow**:
   - **4k** stays clean until **1 req/s**.
   - **8k** breaks by **0.5 req/s** (1161ms).
   - **12k** saturates by **0.5 req/s** (3.3s).
   - **qps 0.25 is clean at every length** — even a 12k stall only hits `max`.

3. **Prefix reuse defeats it universally** (`warm` column: 14.6–34.5ms everywhere) —
   a cache hit isn't a prefill. **warm½** only helps when halving the cold rate
   drops below the knee: rescues 8k@0.5 (1161→**29**) but not 12k@0.5 (the cold
   half alone saturates → 3.8s).

4. **Saturation** (`*`): when `qps·prefill_s ≳ 1`, prefills run back-to-back and
   decode never recovers (stall% → ~50–60%, even p50 rises). This is a throughput
   wall, not just a tail — chunked prefill can't add prefill FLOPs; needs
   rate-limit / bigger card.

## Decision for chunked prefill

Depends on the workload. The stall is a pure **tail** effect — `p50`/`steady` stay
at baseline in every cell — so it only matters under a hard ITL-**p99** SLA, and
only in the cold-prefix / high-QPS / long-prompt corner. Chunked prefill itself is
out of scope for #244; this measurement only records the go/no-go.

| Your workload | Chunked prefill? |
|---------------|------------------|
| Moderate prompts at low QPS, **or** prompts that share a prefix (warm) | **No** — ITL p99 already at baseline (~15ms) |
| Hard ITL-p99 SLA **and** sustained arrival (≥1 req/s) **or** routinely long cold prompts (≥8k) | **Yes** — this is the right fix |
| …and the cell is already saturated (`qps·prefill_s ≳ 1`) | **Yes, but** necessary-not-sufficient — also rate-limit / bigger card |

The trigger lives in the deployment's SLA and arrival pattern, not the engine: with
no hard per-token SLA, or long prompts that share prefixes, leave it unbuilt.


## Reproduce

```bash
# Build (RTX 5070 Ti / Arch WSL2; SM 120, absolute Triton python):
CUDA_HOME=/opt/cuda NVCC_PREPEND_FLAGS="-ccbin g++-13" \
  LIBRARY_PATH=/usr/lib/wsl/lib:/opt/cuda/lib64 \
  PEGAINFER_CUDA_SM=120 PEGAINFER_TRITON_PYTHON=/abs/.venv/bin/python \
  cargo build -r -p pegainfer-server --bin bench_serving

BIN=./target/release/bench_serving
BG="--bg-prompt-len 512 --bg-concurrency 4 --bg-output-len 1024"
# Chunking OFF (the sweeps above): --max-prefill-tokens ≥ the prompt forwards the
# whole prompt in one unified step. Omit the flag for the model default (1024).
sweep() {  # $1 = model path
  for q in 0.25 0.5 1.0; do for p in 4096 8192 12288; do for w in 0.0 0.5 1.0; do
    sleep 25
    $BIN --model-path "$1" --max-prefill-tokens 99999999 --format json --out /tmp/itl.json \
      mixed $BG --inj-prompt-len $p --inj-output-len 1 --qps $q \
      --num-injections 5 --warmup 5 --inj-warm-frac $w --skip-baseline >/dev/null 2>&1
    echo "$1 q=$q p=$p w=$w  $(python3 -c "import json;d=json.load(open('/tmp/itl.json'))['mixed_itl']['all'];print(f\"p99={d['p99_ms']:.0f} max={d['max_ms']:.0f}\")")"
  done; done; done
}
sweep models/Qwen3-4B
sweep models/Qwen3.5-4B

# Canonical mixed_itl cell folded into the regression snapshot:
CUDA_HOME=/opt/cuda LIBRARY_PATH=/usr/lib/wsl/lib:/opt/cuda/lib64 \
  $BIN --model-path models/Qwen3-4B snapshot --warmup 5 --iters 20
```

The canonical cell is folded into the `snapshot` subcommand as the `mixed_itl`
profile, so it refreshes with the prefill/decode profiles and its history lives in
git ([bench-regression.md](../conventions/bench-regression.md)). The wider
qps×warm×prompt sweep stays an **ad-hoc diagnostic** — the minimal loop above is
the whole of it; widen the loops for more cells. It's deliberately not a tracked
script: too noisy/expensive for the regression set.


## Next

If a deployment surfaces a hard ITL-p99 SLA under sustained/long-cold-prompt load,
re-run the sweep at its QPS/prompt mix and compare mixed vs baseline p99 to
re-open the chunked-prefill decision.
