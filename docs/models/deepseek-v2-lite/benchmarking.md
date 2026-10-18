# DeepSeek-V2-Lite Verification And Benchmarking

> **TL;DR:** Use `e2e_ep2` for correctness, `dsv2_lite_ep2_decode_attribution` for direct decode diagnostics, and `bench_dsv2lite_http_slo.py` for retained HTTP SLO evidence. The #466 follow-up NCCL readiness fix lets the backend discover a compatible Python-wheel NCCL runtime from `PATH`; these artifacts still answer different questions, and none of them alone proves production readiness.
>
> Last touched: 2026-07

## Verification Ladder

| Layer | Entry point | Proves | Does not prove |
| --- | --- | --- | --- |
| Correctness / integration | `pegainfer-deepseek-v2-lite/tests/e2e_ep2.rs` | EP2 load, host-staged/NCCL generation, request isolation, output tokens/text/hashes, route and collective accounting | Latency, throughput, SLO, soak, production readiness |
| Direct diagnostic | `dsv2_lite_ep2_decode_attribution` | Fixed-shape CPU/CUDA section timing, route/collective counters, graph-readiness diagnostics | HTTP behavior, client pressure, serving SLO |
| HTTP serving SLO | `scripts/bench_dsv2lite_http_slo.py` over the shared HTTP harness | Streaming TTFT/TPOT/ITL, request/output throughput, failures/timeouts, server trace coverage, output hashes, repeat spread | Direct-kernel attribution, sustained soak, production readiness |
| Soak / production readiness | Separate sustained-run gate | Memory drift, long-duration tails, recovery and deployment limits | Out of scope for issue #466 |

## Correctness Gate

The following is a remote-GPU template. Replace `MODEL_PATH` and, on Blackwell, select an NCCL runtime that satisfies the startup check.

```bash
# Template: requires two GPUs and DeepSeek-V2-Lite weights.
PEGAINFER_TEST_MODEL_PATH=MODEL_PATH \
PEGAINFER_DSV2_LITE_EP_BACKEND=host-staged \
  cargo test --release -p pegainfer-deepseek-v2-lite \
  --features deepseek-v2-lite --test e2e_ep2 -- --nocapture

PEGAINFER_TEST_MODEL_PATH=MODEL_PATH \
PEGAINFER_DSV2_LITE_EP_BACKEND=nccl \
PEGAINFER_NCCL_LIB_DIR=NCCL_LIB_DIR \
  cargo test --release -p pegainfer-deepseek-v2-lite \
  --features deepseek-v2-lite --test e2e_ep2 -- --nocapture
```

The JSON emitted by this test uses `report_intent=correctness_integration` and carries an explicit no-performance claim boundary. Use the same-host HF comparison in `hf-accuracy-gate.md` for accuracy-sensitive changes.

On `sm_120`, the DSV2-Lite NCCL backend fails before communicator creation when the loaded NCCL is older than `2.26.2`. NCCL 2.26.2 contains NVIDIA's shared-memory fix for recent Blackwell GPUs ([NVIDIA/nccl#1637](https://github.com/NVIDIA/nccl/issues/1637)); older releases can exceed the device/function shared-memory limit when launching collectives. Set `PEGAINFER_NCCL_LIB_DIR` or `PEGAINFER_NCCL_PYTHON` to select a compatible runtime when the process environment does not already expose one. The backend also scans Python executables on `PATH` for `nvidia/nccl/lib/libnccl.so.2`, so a conda or venv Python with the `nvidia-nccl-cu12` wheel can satisfy the floor without an extra selector. This startup floor is specific to `sm_120`, not older GPU architectures.

## Direct Diagnostic Benchmark

This is a remote-GPU template:

```bash
# Template: direct/in-process diagnostic, no HTTP server involved.
PEGAINFER_DSV2_LITE_EP_BACKEND=nccl \
PEGAINFER_NCCL_LIB_DIR=NCCL_LIB_DIR \
  cargo run --release -p pegainfer-deepseek-v2-lite \
  --features deepseek-v2-lite \
  --bin dsv2_lite_ep2_decode_attribution \
  -- --model-path MODEL_PATH --commit COMMIT --batch-size 8 \
  --out artifacts/bench/dsv2-lite/direct/nccl-b8.json
```

The artifact has `kind=deepseek_v2_lite_direct_decode_attribution` plus model, backend, commit, hardware/toolchain, workload, metrics, coverage, output hashes, and `claim_boundary`. Keep `timing`, `by_section`, and `by_gpu_*` as attribution fields. Do not translate them into HTTP TTFT/TPOT or production throughput claims.

## Retained HTTP SLO Profiles

The profile definitions and six-child aggregate gate live in `scripts/bench_dsv2lite_http_slo.py`. The model runner calls the generic `bench_http_sweep.py`, which calls `bench_http_serving.py`; neither generic harness imports model-specific contracts.

| Profile | Workload | Default timeout | Intended use |
| --- | --- | ---: | --- |
| `dsv2-lite-short-decode-heavy` | 32 requests, `prompt_words=64`, `max_tokens=64` | 240 s/request | Short decode-heavy SLO rows with at least 30 samples per repeat |
| `dsv2-lite-mixed-prompt-shape` | 32 alternating `prompt_words=64,512` requests, `max_tokens=64` | 240 s/request | Short/long prompt interaction and trace tails with at least 30 samples per repeat |
| `dsv2-lite-long-prompt-smoke` | `prompt_words=2048`, `max_tokens=64` | 900 s/request | One explicit long-prompt boundary cell |

`prompt_words` is deterministic prompt-generator input. The artifact records actual server-side prompt-token counts when traces are attached.

Retained profiles lock greedy sampling, ignore-EOS, shape, absolute request deadline, warmup, request count, concurrency, repeats, and full trace coverage. Percentiles use R7 linear interpolation. The aggregate rejects missing traces, failed/time-out requests, duplicate cells, mixed commit/model/backend provenance, backend-version drift, and leaf-artifact SHA mismatches.

`coverage_gate.passed` means the retained report has complete HTTP evidence for the fixed contracts. It is not a numeric latency budget; reports keep `latency_budget.configured=false` until a production budget is ratified.

### Server

Run one backend at a time. These are remote-GPU templates:

```bash
# Template: host-staged server.
RUST_LOG=info \
PEGAINFER_DSV2_LITE_EP_BACKEND=host-staged \
  cargo run --release -p pegainfer-server \
  --features deepseek-v2-lite --bin pegainfer -- \
  --model-path MODEL_PATH --served-model-name DeepSeek-V2-Lite \
  --port 18000 --cuda-graph=false \
  > artifacts/bench/dsv2-lite/RUN_ID/host-staged/server.log 2>&1

# Template: NCCL server. Use the verified runtime selector on Blackwell.
RUST_LOG=info \
PEGAINFER_DSV2_LITE_EP_BACKEND=nccl \
PEGAINFER_NCCL_LIB_DIR=NCCL_LIB_DIR \
  cargo run --release -p pegainfer-server \
  --features deepseek-v2-lite --bin pegainfer -- \
  --model-path MODEL_PATH --served-model-name DeepSeek-V2-Lite \
  --port 18000 --cuda-graph=false \
  > artifacts/bench/dsv2-lite/RUN_ID/nccl/server.log 2>&1
```

### Sweep

These are remote-GPU templates. Replace metadata/path placeholders; `PROFILE` is one row from the profile table and `PROFILE_SLUG` is the matching output directory name such as `short`, `mixed`, or `long`.

```bash
# Template: run one retained profile/backend contract.
python3 scripts/bench_dsv2lite_http_slo.py run \
  --profile PROFILE --backend BACKEND \
  --base-url http://127.0.0.1:18000 \
  --server-log artifacts/bench/dsv2-lite/RUN_ID/BACKEND/server.log \
  --model-path MODEL_PATH --server-command "SERVER_COMMAND" --commit COMMIT \
  --model-revision MODEL_REVISION \
  --server-binary SERVER_BINARY \
  --out-dir artifacts/bench/dsv2-lite/RUN_ID/BACKEND/PROFILE_SLUG

# Template: combine the six passing backend/profile summaries into one report.
python3 scripts/bench_dsv2lite_http_slo.py combine \
  --summary artifacts/bench/dsv2-lite/RUN_ID/host-staged/short/sweep_summary.json \
  --summary artifacts/bench/dsv2-lite/RUN_ID/host-staged/mixed/sweep_summary.json \
  --summary artifacts/bench/dsv2-lite/RUN_ID/host-staged/long/sweep_summary.json \
  --summary artifacts/bench/dsv2-lite/RUN_ID/nccl/short/sweep_summary.json \
  --summary artifacts/bench/dsv2-lite/RUN_ID/nccl/mixed/sweep_summary.json \
  --summary artifacts/bench/dsv2-lite/RUN_ID/nccl/long/sweep_summary.json \
  --out artifacts/bench/dsv2-lite/RUN_ID/retained_slo_report.json
```

Each `sweep_summary.json` records:

- command, model, source, backend, hardware, and toolchain metadata;
- workload contract and latency-budget status;
- TTFT, TPOT, and ITL p50/p95/p99;
- request throughput and output tokens/s;
- completed, failed, and timeout counts;
- active-set, decode-batch, token-timing, and missing-trace coverage;
- output-hash distribution;
- repeat median/min/max plus `stable`, `noisy`, `insufficient_repeats`, `benchmark_error`, `failed_or_timeout`, or `startup_failure`.

A cell is `noisy` when repeat spread exceeds 10% of the median for TTFT/TPOT/ITL p95, request throughput, or output-token throughput.

## Local Tooling Gate

These commands were run on 2026-07-13:

```bash
python3 -m py_compile scripts/bench_http_common.py scripts/bench_http_serving.py scripts/bench_http_sweep.py scripts/bench_dsv2lite_http_slo.py
python3 -m unittest -v tests/test_bench_http_serving.py tests/test_bench_http_sweep.py tests/test_bench_dsv2lite_http_slo.py
cargo fmt --all --check
cargo metadata --locked --no-deps --format-version 1
```

## Current-Source Evidence

Regenerate the retained report when its profile, schema, or measurement contract changes. The current retained 2x RTX 5090 evidence completed all six host-staged/NCCL children with zero failures/timeouts and full required trace coverage. The gitignored local copy is under `artifacts/bench/dsv2-lite/<run-id>/`; aggregate SHA-256: `a7e677c63d1ce92ad0c069f83acfcc8b381e07d06bed9364ab899adecef8d317`.

### Issue #466 Follow-Up NCCL Readiness Smoke

The retained #466 report exposed a runtime blocker outside the report tooling: with `PEGAINFER_DSV2_LITE_EP_BACKEND=nccl` and no explicit NCCL selector, the server could load the system `libnccl.so.2` `2.25.1` on 2x RTX 5090 and fail before readiness. The focused fix keeps explicit `PEGAINFER_NCCL_*` selectors fail-fast, then scans executable Python binaries found on `PATH` for NCCL wheel roots before falling back to generic library names. If an auto-discovered PATH candidate loads but fails the sm_120 NCCL version floor, the loader records it and continues to the next auto candidate. On the validation host, that resolved `<conda-root>/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2` and loaded NCCL `2.26.2`.

Validation was run on `upstream/main@d083b745699f527186baed1e61225e4c86965486` plus the focused fix, with `PEGAINFER_CUDA_SM=120`, 2x RTX 5090, and no `PEGAINFER_NCCL_PYTHON`, `PEGAINFER_NCCL_LIB_DIR`, `PEGAINFER_NCCL_LIB`, `PEGAINFER_NCCL_LIBRARY_PATH`, `CONDA_PREFIX`, or `VIRTUAL_ENV` for the NCCL runs:

| Gate | Artifact basename | SHA-256 / key result | Boundary |
| --- | --- | --- | --- |
| HF / host-staged / NCCL correctness compare | `comparison.json` | `3e5e324f97171a35db682b4a2ebe4e08724b159e06516df19728ac4a482d0760`; `classification=all_token_text_exact`, `case_count=5`, `warnings=[]` | correctness only |
| NCCL direct diagnostic batch 1 | `nccl-b1.json` | `40753081bacbac3fbfa0da82a5c65961369138b46cedc26e7cf5170aca0f45e4`; token/text hash exact, `gpu_timing_failure_count=0` | direct Hello/16 attribution only |
| NCCL HTTP c1 short smoke | `pw64_c1_mt64_r0.json` | `9c84d60f889b4e90f805541870a45b22ab29737876b185b79f429d63fe38a3ff`; `completed=4`, `failed=0`, `timeouts=0`, `combined_output_hash=2e74c01cfdd4dc75`, `retention_gate.passed=true` | one-cell short smoke only |
| Host-staged HTTP c1 short smoke | `pw64_c1_mt64_r0.json` | `c8b83b72ca47829d943262a96f6b8ea898344f5c84e427c3232176b63778b36c`; `completed=4`, `failed=0`, `timeouts=0`, `combined_output_hash=2e74c01cfdd4dc75`, `retention_gate.passed=true` | host-staged no-regression smoke only |

The NCCL server log for the final HTTP smoke includes `DeepSeek-V2-Lite NCCL backend loaded: version=2.26.2, version_code=22602` and reached readiness in 17 seconds. This fixes the readiness/runtime blocker for the short 64-token NCCL HTTP path. It does not complete the #465 sustained soak, #452 long-prompt scheduler work, #635 device attention/KV work, or #636 device route plan.

## Claim Boundary

A passing retained report is HTTP pressure/SLO evidence for the named backend, model revision, hardware/toolchain, and workload. It can support comparisons between retained runs when their contracts match. It does not establish direct decode attribution, vLLM parity, sustained soak stability, multi-node recovery, or production readiness.
