# Kimi-K2 PPLX EP Correctness

> **TL;DR:** TP8/DP1 PPLX decode is token-trace exact against the TP8/DP1 NCCL
> path under the same bs64 active-decode schedule on an 8×H200 node. Historical
> TP8/DP1 correctness baseline; the active serving line is now TP1/DP8/EP8 PPLX.
>
> **Ground truth rule:** compare PPLX against TP8 NCCL with the same scheduler
> shape. A single historical hash is not enough once admission changes make the
> decode batch truly active at 64 rows.
>
> **Last touched:** 2026-06

## Scope

Target comparison:

| Item | Value |
| --- | --- |
| Machine | 8×H200 node |
| Model | `$MODEL_DIR` |
| Reference path | `--tp-size 8 --dp-size 1 --ep-backend nccl`, feature `kimi-k2` |
| PPLX path | `--tp-size 8 --dp-size 1 --ep-backend pplx`, feature `kimi-k2` |
| Probe | `bench_serving request --prompt-len 1 --output-len 5 --concurrency 64 --warmup 0 --iters 1 --cuda-graph false` |

> CLI note: the parallel shape and EP backend are selected by the
> `--tp-size/--dp-size/--ep-backend` flags. The old `kimi-k2-pplx-ep` cargo
> feature and `PEGAINFER_KIMI_PARALLEL` env (used in the original 2026-05-25 run)
> have been removed; the feature is now just `kimi-k2`.

TP1/DP8 PPLX is intentionally not the baseline for this document. The current
repair first makes TP8/DP1 PPLX match TP8/DP1 NCCL.

## Validation Ledger

| Date | Path | Output | Result |
| --- | --- | --- | --- |
| 2026-05-25 | `cargo check --release -p pegainfer-server --features kimi-k2 --bin bench_serving` (PPLX selected via `--ep-backend pplx`) | clean build on 8×H200 node | Pass |
| 2026-05-25 | `cargo check --release -p pegainfer-server --features kimi-k2 --bin bench_serving` | clean build on 8×H200 node | Pass |
| 2026-05-25 | `cargo test --release -p pegainfer-comm --test pplx_roundtrip -- --nocapture` | 8 ranks dispatch+combine roundtrip, each rank received 512 tokens | Pass |
| 2026-05-25 | TP8 PPLX bs4, output 5, iters 3 | `$RESULT_ROOT/kimi_pplx_tp8_bs4_o5_final.json`: 12/12 traces hash `7c4c5d83355198fd` | Pass |
| 2026-05-25 | TP8 NCCL bs64 active decode | `$RESULT_ROOT/kimi_nccl_tp8_active64_o5_final.json`: `Counter({'7c4c5d83355198fd': 32, '9eecc1ca6fb3409d': 32})`, steady TPOT p50 `97.53ms` | Reference |
| 2026-05-25 | TP8 PPLX bs64 active decode | `$RESULT_ROOT/kimi_pplx_tp8_active64_o5_after_review.json`: `Counter({'7c4c5d83355198fd': 32, '9eecc1ca6fb3409d': 32})`, steady TPOT p50 `110.14ms` | Matches NCCL |
| 2026-05-25 | TP8 PPLX vs TP8 NCCL bs64 per-index traces | 0 mismatches across 64 requests | Pass |

The bs64 probe has two hashes because fully active 64-row decode has a different
schedule from the earlier split-wave runs. The correctness condition is
per-index trace equality between PPLX and NCCL for the same active scheduling.

## Repro Commands

Common environment:

```bash
cd $PEGAINFER_DIR
export CUDA_HOME=/usr/local/cuda
export NVCC=/usr/local/cuda/bin/nvcc
export LD_LIBRARY_PATH=$RESULT_ROOT/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PEGAINFER_CUDA_SM=90a
export PEGAINFER_TRITON_PYTHON=$PEGAINFER_DIR/.triton-venv/bin/python
```

NCCL reference (TP8/DP1):

```bash
cargo run --quiet --release -p pegainfer-server --features kimi-k2 --bin bench_serving -- \
  --model-path $MODEL_DIR \
  --tp-size 8 --dp-size 1 --ep-backend nccl \
  --cuda-graph false \
  --format json \
  --out $RESULT_ROOT/kimi_nccl_tp8_active64_o5_final.json \
  request --prompt-len 1 --output-len 5 --concurrency 64 --warmup 0 --iters 1
```

PPLX path (TP8/DP1):

```bash
cargo run --quiet --release -p pegainfer-server --features kimi-k2 --bin bench_serving -- \
  --model-path $MODEL_DIR \
  --tp-size 8 --dp-size 1 --ep-backend pplx \
  --cuda-graph false \
  --format json \
  --out $RESULT_ROOT/kimi_pplx_tp8_active64_o5_after_review.json \
  request --prompt-len 1 --output-len 5 --concurrency 64 --warmup 0 --iters 1
```

Trace comparison:

```bash
uv run --no-project python - <<'PY'
import json
from collections import Counter
from pathlib import Path

paths = {
    "nccl": Path("$RESULT_ROOT/kimi_nccl_tp8_active64_o5_final.json"),
    "pplx": Path("$RESULT_ROOT/kimi_pplx_tp8_active64_o5_after_review.json"),
}
traces = {}
for name, path in paths.items():
    data = json.loads(path.read_text())
    rows = data["metrics"]["generated_token_traces"]
    traces[name] = rows
    print(name, Counter(row["hash"] for row in rows))

mismatches = [
    idx for idx, (a, b) in enumerate(zip(traces["nccl"], traces["pplx"]))
    if a["tokens"] != b["tokens"]
]
print("mismatches", len(mismatches), mismatches[:16])
PY
```

## Fixed Invariants

### Active MoE Rows

Decode arenas are bucketed up to 64 rows, but a specific decode step may have
fewer active requests. PPLX MoE must route only active rows:

```text
arena seq_len = allocated bucket rows
active_len = token_ids.len()
PPLX MoE seq_len = active_len
```

`KimiWorkerDecodeScratch::set_moe_seq_len(active_len)` is applied only around
the PPLX MoE layer and restored afterward. This prevents stale arena padding
rows from entering PPLX dispatch and combine.

### TP8 Duplicate Source Canonicalization

In TP8/DP1 each TP rank has the same post-collective hidden rows and the same
router result. PPLX all-to-all still observes eight source ranks, so the compact
Marlin route must canonicalize duplicate source groups when the Kimi PPLX
backend opts in:

```text
transfer counts = total rows across sources
compute counts = max rows per canonical TP source group
padded row for duplicate sources = canonical padded row
```

The flag is `canonicalize_duplicate_sources`. It is enabled only when Kimi runs
TP8/DP1 PPLX, where TP sources are duplicate rows. TP1/DP8, lower-level PPLX
tools, and Python bindings keep the default `false`.

### NCCL-Layout Local Expert Compute For TP8

For TP8/DP1 PPLX decode, the current correctness path computes local experts
with the same NCCL-layout Marlin routing used by the NCCL path, then scatters
the global route rows into the compact PPLX combine layout. PPLX remains
responsible for the combine transfer.

This preserves:

```text
router top-k -> NCCL Marlin route-slot layout -> W2 applies router weight
-> BF16 expert row -> PPLX compact combine -> F32 routed output
```

The PPLX dispatch step is still executed to drive PPLX metadata and protocol.
Removing unnecessary TP8 duplicate payload movement is a performance item, not
part of the correctness baseline.

### Routed-Row Weights

NCCL applies the router top-k weight inside Marlin W2 before the BF16 expert row
is stored. PPLX must preserve the same rounding boundary. Kimi PPLX W2 receives
the real route weights, while `combine_recv` uses dummy all-ones weights because
the expert row is already weighted.

### No Silent Fallback

`--ep-backend pplx` must fail startup if PPLX bootstrap fails. A
silent fallback to NCCL would make correctness probes pass for the wrong path.
The runtime log should include:

```text
kimi-k2: pplx EP backends installed on all 8 ranks
```

## Dump Policy

Wide dumps are acceptable during repair, but production code must not retain
debug markers or dump helpers. The baseline code has no `KimiDebugDecodeMarker`,
`debug_dump_*`, `dump_point`, or `pplx_routed_out` leftovers.
