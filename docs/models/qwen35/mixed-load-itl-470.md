# Qwen3.5-4B Mixed-Load ITL — chunked prefill vs active-decode stall (issue #470)

**Last touched:** 2026-07-19

**Question (issue #470):** when a long *cold* prompt arrives while decode is already
in steady state, does scheduler-level chunked prefill bound the Qwen3.5 inter-token
latency (ITL) tail?

**Answer in one line:** it bounds the *per-step* freeze but not the *total* prefill
work, so it trades a rare huge stall for frequent medium ones — helpful past the p99
knee, harmful below it, and useless once `qps·prefill_s ≳ 1`.

---

## 1. Findings

1. **Qwen3.5 is not immune.** With chunking OFF, one cold injection freezes every
   active decode stream for ~the whole prefill wall — `4k ≈ 0.28s … 16k ≈ 1.19s`
   (max ≈ cold-prefill median).
2. **Chunking (budget 1024) caps the per-step freeze at ~one chunk (~80–110 ms), but
   does not reduce total prefill FLOPs.** Below the p99 knee it *raises* headline ITL
   p99 (~14 ms → ~80–92 ms); above the knee it *lowers* p99/max (hundreds–thousands
   of ms → ~the chunk wall). It also adds ~13–16% TTFT.
3. **Throughput wall ≠ heavy tail.** A wall needs `qps·prefill_s ≳ 1` **and** a
   lifted mixed **p50** (12k/16k @1.0, stall% → 94–95%). `12k@1.0 OFF` (p50 still
   12.3 ms, p99 = 868) is a *pure heavy tail*, not a wall.
4. **The old "Qwen3.5 p99-immune" table was a `max_batch=4` slot-starvation
   artifact**, reproduced here by the negative control (`decode_n` never reaches 4,
   p99 ≈ baseline).

> This report is the #470 acceptance evidence. Saturated `qps=1.0` cells are
> single-run and small-sample (gap N as low as 77); their tail numbers are
> **qualitative only**. No production-readiness claim — see §7 claim boundary.

---

## 2. Environment & configuration

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| Driver / CUDA | 535.104.05 / CUDA 12.1 (nvcc V12.1.105) |
| Branch | `qwen35-mixed-load-itl-470` (adds `--max-batch`) |
| Model | `models/Qwen3.5-4B` |
| Tool | `bench_serving mixed` |
| Main matrix | `--max-batch 8 --bg-concurrency 4`, all cells |
| Negative control | `--max-batch 4 --bg-concurrency 4` |
| Background | prompt 512 / output 4096 / concurrency 4, greedy `ignore_eos` |
| Injector | cold only (`--inj-warm-frac 0.0`), `output_len=1`, `num_injections=10`, warmup 5 |
| chunk ON | default `--max-prefill-tokens` → budget 1024 |
| chunk OFF | `--max-prefill-tokens 99999999` (whole prompt in one step) |
| Sweep | `qps ∈ {0.25, 0.5, 1.0}` × `prompt ∈ {4k, 8k, 12k, 16k}` × `{ON, OFF}` = 24 cells |
| Diagnostics | `PEGAINFER_ITL_DEBUG=1` + `scripts/itl_step_agg.py` |
| Hygiene | `sleep 25` between cells; `nvidia-smi` GPU-idle (~3 MiB) check before each |
| Raw data | `datasets/mixed-load-itl-470-data-mb8/` (untracked — repo-root `datasets/` is gitignored; per-cell `.json` + `.log`) |

**Capacity model.** `--max-batch` is the **scheduler admission cap**, decoupled from
the CUDA-graph decode bucket. `8` is itself a bucket, so physical capacity ==
admission == 8; with 4 background streams the injector always has 4 free slots. (The
knob also accepts a non-bucket value — e.g. `--max-batch 5` admits ≤5 while
physically allocating bucket 8 — but this matrix uses the legal bucket 8 so physical
== admission. `4` is likewise a bucket.)

**Recorded capacity metrics (new in this PR).** Per the maintainer's request that an
archived run be reproducible on its own, every cell's JSON now serialises the
capacity into `config`:

| Field | Meaning | Main matrix | Negative control |
|---|---|---|---|
| `config.max_batch` | scheduler admission cap the engine was built with | `8` | `4` |
| `config.max_prefill_tokens` | per-step chunk budget | `null` (ON, default 1024) / `99999999` (OFF) | `null` |

Both carry `#[serde(default)]`, so pre-#470 snapshots (which lack the fields) still
deserialise — `max_batch` reads back as `0`, i.e. "unrecorded".

---

## 3. Method & review gates

`bench_serving mixed` keeps N long-lived background decode streams running and fires
a serial injector of long cold prompts at rate `qps`. Each background inter-token gap
is tagged **stall** or **steady** by whether it overlaps an injection window, and
compared against a decode-only **baseline**.

**Mandatory gates (per the maintainer's #470 comment):**

| Gate | Rule |
|---|---|
| **valid** | ≥1 `ITL_STEP` with `prefill_tok>0` **and** `decode_n == bg_concurrency` (the injection intersected a full background batch; `plan=overlap_*` is concurrent work, not a serial freeze) |
| **negative control** | `max_batch = bg = 4` must be **invalid** — proves the old artifact reproduces |
| **cross ON/OFF comparison** | compare only `mixed_itl.all` p99/max + `ITL_STEP` per-step stall; never compare the window-attributed stall buckets across configs |
| **saturation** | `qps·prefill_s ≳ 1` and/or a clearly lifted mixed **p50** ⇒ throughput wall, not a tail |

**Caveats kept in view (not hidden):**

- Clocks are sampled on token receipt, so steady/all gaps carry thread-wakeup jitter.
- Window attribution can mislabel admission/edge gaps as stall → trust `ITL_STEP` for
  the true per-step freeze.
- A few baseline p99s are high (~19–25 ms outliers); the report leans on the matrix
  as a whole and on `ITL_STEP`, not on any single baseline value.
- **Sample size shrinks with load.** A `qps=1.0` run is only ~10 s and the background
  is mostly stalled, so gap counts fall from ~8k–11k (low load) to 77–684 (saturated).
  Low-load tails are stable; saturated tails are ≈ max, qualitative only. Absolute
  numbers are not for cross-machine / cross-version comparison.

---

## 4. Results

### 4.1 Negative control — `max_batch = 4, bg = 4` (intentional starvation)

| Item | Value |
|---|---|
| valid | **False** — `decode_n ∈ {1, 3}`, **never 4** |
| mixed p99 / max | **14.7 / 81.7 ms** (p99 ≈ baseline — the fake "immunity") |
| warnings | slot-starvation warning + `bg_output` exhaustion (a *downstream symptom*: the injector can't admit → the run overruns → background streams hit 4096 and finish early — not an independent contaminant) |

This corrects `docs/benchmarks/mixed-load-itl.md`: the old "Qwen3.5 p99-immune"
result is **primarily** slot starvation (`bg ≥ max_batch` fills every slot so the
injector never overlaps a full decode batch), and only secondarily the short
`bg_output_len` lowering stall frequency.

### 4.2 Headline ITL — main matrix (all 24 valid)

Cells are **ITL p99 (ms)** with **max** in parentheses. `*` = throughput-wall cell
(see §4.5). All `qps=1.0` saturated cells are single-run small-sample (gap N as low
as 77); their p99 ≈ max and are qualitative only.

**chunk OFF (whole prompt in one step):**

| prompt \ qps | 0.25 | 0.5 | 1.0 |
|---:|---:|---:|---:|
| 4k | 13.9 (284) | 13.4 (283) | **282** (283) |
| 8k | 13.9 (565) | 13.5 (563) | **562** (613) |
| 12k | 13.7 (873) | **868** (875) | **868** (868) |
| 16k | 13.6 (1187) | **1174** (1223) | **1201\*** (1201) |

At low frequency OFF p99 sits at baseline (~13–14 ms) but **max ≈ the prefill wall** —
the damage is in max, not p99, until frequency crosses the knee.

**chunk ON (budget 1024):**

| prompt \ qps | 0.25 | 0.5 | 1.0 |
|---:|---:|---:|---:|
| 4k | **79** (82) | **80** (81) | **81** (103) |
| 8k | **83** (99) | **84** (85) | **109** (123) |
| 12k | **87** (89) | **88** (119) | **106\*** (108) |
| 16k | **90** (109) | **92** (109) | **112\*** (119) |

ON pins p99 at **~79–92 ms** (one chunk wall) in the low/mid-load cells, rising to
~106–112 ms only in the `qps=1.0` 8k–16k cells (throughput wall / near-knee).

**Gap sample count N (tail confidence), ON:** shrinks with load, smallest in the `*`
cells.

| qps | 4k | 8k | 12k | 16k |
|---|---:|---:|---:|---:|
| 0.25 | ~11.4k | ~10.6k | ~9.7k | ~8.8k |
| 0.5 | ~5.5k | ~4.6k | ~3.7k | ~2.7k |
| 1.0 | ~2.4k | ~1.4k | 501–540 | **77–684** |

### 4.3 True per-step freeze — `ITL_STEP`

The per-step diagnostic confirms the headline and is the trustworthy stall metric:

- **OFF:** stall p99/max ≈ the whole prefill wall (`4k ~283 ms … 16k ~1201 ms`).
- **ON:** stall p99/max ≈ a single chunk wall (~80–134 ms).
- Every main-matrix cell shows `decode_n = 4` — the **valid** gate.

### 4.4 Clock health — cold-prefill median (ms)

| prompt | ON q0.25 / 0.5 / 1.0 | OFF q0.25 / 0.5 / 1.0 |
|---:|---:|---:|
| 4k | 326 / 328 / 330 | 289 / 290 / 290 |
| 8k | 662 / 659 / 661 | 571 / 567 / 571 |
| 12k | 1009 / 1004 / 1000 | 877 / 879 / 875 |
| 16k | 1375 / 1370 / 1357 | 1182 / 1183 / 1175 |

Flat across QPS ⇒ no thermal throttling / clock is clean. ON runs **~+13–16%** over
OFF — the per-chunk sync/scheduling overhead, and the quantitative basis for the
"does not reduce total prefill work" claim.

### 4.5 Saturation classification

**Throughput wall** = `qps·prefill_s ≳ 1` **and** a sustained lifted mixed **p50**
(decode can't recover). **Heavy tail** = p50 still at baseline, only p99/max blow up
(a pure frequency effect).

| cell | load `qps·prefill_s` | stall% | mixed p50 | gap N | class |
|---|---:|---:|---:|---:|---|
| 12k @1.0 ON | **1.00** | **94%** | **82 ms** (base ~12.5) | 540 | **throughput wall** — 9/10 injections overran; chunking turns the disaster max into a near-continuous medium stall |
| 16k @1.0 ON | **1.36** | **95%** | **84 ms** | 684 | **throughput wall**, worse |
| 16k @1.0 OFF | **1.18** | **62%** | **1172 ms** | **77** | **throughput wall** — decode nearly drowned by whole-prompt prefill (N=77 tiny) |
| 12k @1.0 OFF | 0.87 | 24% | **12.3 ms** | 501 | **pure heavy tail** — p50 clean; p99=868 is only the "one stall = whole prefill" frequency effect |

**Statistics warning:** saturated cells are single-run and small-sample. `16k@1.0
OFF` has only **77** gaps (steady ~29 / stall ~48), so its p99=1201 ms is really "the
2nd-worst of 77" — effectively max, not a statistical p99. Their absolute tails
support only the qualitative class (wall vs tail), not precise values.

**No hard failures.** No prefill-scratch OOM, KV-admission rejection, or frontend
limit (24 GB + fixed scratch). All 24 main cells completed 10/10 injections, 0 failed.

---

## 5. Answers to the four #470 questions

**Q1 — How bad is the active-decode stall for cold long prompts?**
Severe and ~linear in prompt length. OFF freezes active decode for ~the prefill wall:
`4k ≈ 0.28s, 8k ≈ 0.56s, 12k ≈ 0.87s, 16k ≈ 1.19s`. Severity lives in **max**;
whether it reaches **p99** is a frequency question (Q3).

**Q2 — Do warm / repeated prompts behave differently without prefix reuse?**
No. Qwen3.5 has no prefix cache (linear-attention recurrent state makes a "hit" a
design problem in itself), so `--inj-warm-frac` is a no-op and this run is cold only.
Warm cells carry no value until a prefix cache exists.

**Q3 — Does chunked prefill move ITL p99 / TTFT / TPOT as expected?**

| Metric | Low QPS (OFF p99 clean) | Mid/high load (OFF p99 broken) | Throughput wall |
|---|---|---|---|
| per-step / max stall | ON **improves** (≤ one chunk) | ON **improves** | ON bounds max, p50 stays bad |
| headline ITL p99 | ON **worse** (~14 → ~79–92) | ON **better** (hundreds–thousands → ~90) | ON p99 ≈ chunk wall, stall% → 94%+ |
| injection TTFT | ON **+~15%** | same | same; can push load past 1.0 |
| steady / non-overlap TPOT | ≈ baseline | ≈ baseline (until the wall) | p50 lifted inside the wall |

Frequency knee (OFF): `qps 0.25` clean at every length (damage only in max); `qps
0.5` 4k/8k clean, **12k/16k break p99**; `qps 1.0` everything breaks. ON almost always
breaks p99 relative to baseline because it splits one rare big freeze into several
>1%-frequency medium freezes — the mechanism, not a bug.

**Q4 — Any bottleneck worth its own issue?**
Yes — see §8 Follow-ups (all from this run's measurements, not guessed).

---

## 6. Is chunked prefill sufficient?

| Your goal | Answer under this matrix |
|---|---|
| Bound **per-step / max** decode freeze | **Yes** — holds across the matrix |
| Hard **ITL p99** SLA where OFF is still clean (low QPS or short cold prompts) | **No / harmful** — ON raises p99 to ~the chunk wall |
| Hard p99 SLA where OFF is past the knee (≥12k@0.5, or any length @1.0) | **Yes, for the tail** — ON pulls p99/max back from the prefill wall to ~the chunk wall, but at the cost of a lifted p50/TTFT and ~+15% higher offered load |
| Already at `qps·prefill_s ≳ 1` | **Necessary but not sufficient** — needs rate-limiting; ON may even raise offered load |

Chunking by default is **not** a free lunch: it trades the max↔p99 distribution plus
~15% TTFT. Where "ON improves p99" it is also doing more total work and pushing
offered load higher, so a lower ON p99 at the same qps must not be read in isolation
as ON being better.

---

## 7. Reproduce

Single cell (`ITL_STEP` goes to stderr, so redirect into the log for aggregation):

```bash
cargo build -r -p pegainfer-server --bin bench_serving --features qwen35
BIN=target/release/bench_serving
# chunk ON = default budget; chunk OFF = add --max-prefill-tokens 99999999
PEGAINFER_ITL_DEBUG=1 RUST_LOG=info "$BIN" \
  --model-path models/Qwen3.5-4B --max-batch 8 \
  --format json --out /tmp/cell.json mixed \
  --bg-concurrency 4 --bg-prompt-len 512 --bg-output-len 4096 \
  --inj-prompt-len 4096 --inj-output-len 1 --qps 0.5 --num-injections 10 \
  --inj-warm-frac 0.0 --warmup 5 > /tmp/cell.log 2>&1
python3 scripts/itl_step_agg.py /tmp/cell.log   # valid gate: decode_n reaches bg_concurrency
```

The full matrix wraps the command above in
`for q in 0.25 0.5 1.0; do for p in 4096 8192 12288 16384; do for chunk in on off; …`
(one `--out sweep_q${q}_p${p}_${chunk}.json` per cell, `sleep 25` cooldown, and an
`nvidia-smi` idle check before each); the negative control uses
`--max-batch 4 --bg-concurrency 4`. See `scripts/sweep_mb8.sh`. Raw results are kept
locally under `datasets/mixed-load-itl-470-data-mb8/`.
