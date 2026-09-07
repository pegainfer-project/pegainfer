# DeepSeek-V2-Lite Verification And Benchmarking

> **TL;DR:** Use `e2e_ep2` for correctness, `dsv2_lite_ep2_decode_attribution` for direct decode diagnostics, `bench_dsv2lite_http_slo.py` for retained HTTP SLO evidence, and `bench_dsv2lite_http_soak.py` for issue #465 sustained HTTP soak evidence. These artifacts still answer different questions, and none of them alone proves production readiness.
>
> Last touched: 2026-07

## Verification Ladder

| Layer | Entry point | Proves | Does not prove |
| --- | --- | --- | --- |
| Correctness / integration | `pegainfer-deepseek-v2-lite/tests/e2e_ep2.rs` | EP2 load, host-staged/NCCL generation, request isolation, output tokens/text/hashes, route and collective accounting | Latency, throughput, SLO, soak, production readiness |
| Direct diagnostic | `dsv2_lite_ep2_decode_attribution` | Fixed-shape CPU/CUDA section timing, route/collective counters, graph-readiness diagnostics | HTTP behavior, client pressure, serving SLO |
| HTTP serving SLO | `scripts/bench_dsv2lite_http_slo.py` over the shared HTTP harness | Streaming TTFT/TPOT/ITL, request/output throughput, failures/timeouts, server trace coverage, output hashes, repeat spread | Direct-kernel attribution, sustained soak, production readiness |
| Soak / production readiness | `scripts/bench_dsv2lite_http_soak.py` | Sustained request completion, first/last-quartile tail and throughput drift, RSS/VRAM drift, terminal reasons, clean follow-up recovery | Direct decode attribution, vLLM parity, multi-node recovery, or production readiness by itself |

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

## Sustained HTTP Soak

Issue #465 uses `scripts/bench_dsv2lite_http_soak.py`. It reuses the generic streaming `/v1/completions` leaf benchmark inside fixed time buckets, then writes a backend-level `soak_summary.json` and an optional host-staged/NCCL combined report. A failed soak is still useful evidence when the JSON keeps the failing leaf artifact, server log pointer, terminal reasons, and clean follow-up result.

The default retained shape is greedy, ignore-EOS, `prompt_words=64`, `max_tokens=64`, concurrency `4,8`, and no production latency budget. Tune `--duration-s`, `--bucket-s`, and `--num-requests` for the validation host; `--bucket-s` controls the target wall-clock window per bucket, while `--num-requests` controls each leaf chunk launched continuously inside that window. The summary records actual elapsed time, bucket count, and leaf count.

```bash
# Template: run one backend soak after the server is ready.
python3 scripts/bench_dsv2lite_http_soak.py run \
  --backend BACKEND \
  --base-url http://127.0.0.1:18000 \
  --server-log artifacts/bench/dsv2-lite/RUN_ID/BACKEND/server.log \
  --model-path MODEL_PATH \
  --server-command "SERVER_COMMAND" \
  --commit COMMIT \
  --model-revision MODEL_REVISION \
  --server-binary SERVER_BINARY \
  --backend-runtime-version BACKEND_RUNTIME_VERSION \
  --duration-s 1800 \
  --bucket-s 300 \
  --num-requests 32 \
  --concurrency 4,8 \
  --prompt-words 64 \
  --max-tokens 64 \
  --out-dir artifacts/bench/dsv2-lite/RUN_ID/BACKEND/soak

# Template: combine host-staged and NCCL backend summaries.
python3 scripts/bench_dsv2lite_http_soak.py combine \
  --summary artifacts/bench/dsv2-lite/RUN_ID/host-staged/soak/soak_summary.json \
  --summary artifacts/bench/dsv2-lite/RUN_ID/nccl/soak/soak_summary.json \
  --out artifacts/bench/dsv2-lite/RUN_ID/retained_soak_report.json
```

Each backend summary records:

- leaf artifact path and SHA-256 for every leaf chunk inside each bucket;
- completed, failed, timeout, terminal-reason, and error counts;
- TTFT, TPOT, ITL, request throughput, and output-token throughput per bucket;
- first/last-quartile drift summaries for tails, throughput, RSS, per-device
  VRAM, and total VRAM;
- active-set, pending-queue, decode-batch, token-timing, and missing-trace coverage when server traces are available;
- output hash distribution and combined hash;
- process RSS plus per-device, total, and max device-memory samples;
- post-soak clean follow-up result.

`soak_gate.passed` means every requested concurrency has loaded bucket evidence for the declared duration, every leaf artifact is readable and hashed, leaf commands completed successfully with zero failures/timeouts, optional trace coverage passed for every bucket, and the clean follow-up completed. Runs capped by `--max-buckets` are smoke/debug evidence and do not pass the retained gate. Numeric drift is reported but not a hard budget until deployment limits are ratified.

The combined host-staged/NCCL report hard-fails when either backend is missing, a child `soak_gate` fails any required sub-gate, the child commits differ, model/server provenance differs, the soak contract differs, or a backend runtime boundary is missing or generic. This keeps NCCL selector evidence attached to the soak result instead of turning a conservative runtime run into a default-runtime claim.

## Local Tooling Gate

These commands were run on 2026-07-18:

```bash
python3 -m py_compile scripts/bench_http_common.py scripts/bench_http_serving.py scripts/bench_http_sweep.py scripts/bench_dsv2lite_http_slo.py scripts/bench_dsv2lite_http_soak.py
python3 -m unittest -v tests/test_bench_http_common.py tests/test_bench_http_serving.py tests/test_bench_http_sweep.py tests/test_bench_dsv2lite_http_slo.py tests/test_bench_dsv2lite_http_soak.py
cargo fmt --all --check
cargo metadata --locked --no-deps --format-version 1
```

## Current-Source Evidence

Regenerate retained reports when their profile, schema, or measurement contract changes. The gitignored local copies live under `artifacts/bench/dsv2-lite/<run-id>/`; keep public docs to artifact basenames, hashes, contract shape, and claim boundaries.

### Issue #465 Retained HTTP Soak

The retained #465 rerun used code commit `a5703d0424d917ce99b4bd8691b0b86eecde966f`, model revision `604d5664dddd88a0433dbae533b7fe9472482de0`, 2x RTX 5090, `prompt_words=64`, `max_tokens=64`, greedy sampling, ignore-EOS, concurrency `4,8`, `duration_s=120`, `bucket_s=60`, `num_requests=8`, and full required trace coverage. Later commits harden the retained gate semantics; regenerate the artifacts before claiming final-head evidence for the changed benchmark schema or runtime path. The combined report SHA-256 is `1c06e8825da70888277f1485f54d7f4fb9b2f61d617149d3ac7357cd5a03e7f1`.

| Artifact basename | Backend | SHA-256 | Result | Boundary |
| --- | --- | --- | --- | --- |
| `soak_summary.json` | host-staged | `3fe4f163024602a51f10cac0c15cc24a5feffd92bbcb93091e175cd63e49bd33` | `completed=112`, `failed=0`, `timeouts=0`, `soak_gate.passed=true`, clean follow-up passed, combined output hash `6912777bed672f57` | Host-staged sustained HTTP soak evidence |
| `soak_summary.json` | NCCL | `52c2a70d895fc0c080233961ea06824097fbf5b7241a0983e41c0ca6646e762f` | `completed=128`, `failed=0`, `timeouts=0`, `soak_gate.passed=true`, clean follow-up passed, combined output hash `24f2db9fc47acc10` | NCCL sustained HTTP soak evidence with NCCL `2.26.2`, `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1` |
| `retained_soak_report.json` | combined | `1c06e8825da70888277f1485f54d7f4fb9b2f61d617149d3ac7357cd5a03e7f1` | `coverage_gate.passed=true`, `child_gates={host-staged:true,nccl:true}`, commit/model provenance consistent, runtime boundaries present | Combined host-staged/NCCL #465 report |

Resource and drift fields are retained for diagnosis, not hard budgets. Host-staged max device memory stayed flat in both concurrency buckets, with output-token throughput drift `-1.5%` at c4 and `-0.8%` at c8; RSS moved `+1.6%` at c4 and `+12.1%` at c8. NCCL max device memory stayed flat, with output-token throughput drift `-1.0%` at c4 and `+1.1%` at c8; RSS moved `+1.6%` at c4 and `+3.8%` at c8. The NCCL server used a conservative single-node runtime selection and reached readiness after the NCCL communicator init path completed; do not turn this row into a default-runtime speed claim.

Reviewer evidence summary for issue/PR text:

- Combined artifact basename/SHA: `retained_soak_report.json` / `1c06e8825da70888277f1485f54d7f4fb9b2f61d617149d3ac7357cd5a03e7f1`.
- Child artifact basenames/SHA: host-staged `soak_summary.json` / `3fe4f163024602a51f10cac0c15cc24a5feffd92bbcb93091e175cd63e49bd33`; NCCL `soak_summary.json` / `52c2a70d895fc0c080233961ea06824097fbf5b7241a0983e41c0ca6646e762f`.
- Gate result: host-staged and NCCL child gates passed, combined gate passed, commit/model provenance consistent, runtime boundary retained.
- Claim boundary: short-shape sustained HTTP soak evidence only; no production-readiness, default-runtime NCCL, long/mixed-prompt, or vLLM-parity claim.

### Issue #466 Retained HTTP SLO Evidence

The current retained #466 2x RTX 5090 evidence completed all six host-staged/NCCL children with zero failures/timeouts and full required trace coverage. The gitignored local copy is under `artifacts/bench/dsv2-lite/<run-id>/`; aggregate SHA-256: `a7e677c63d1ce92ad0c069f83acfcc8b381e07d06bed9364ab899adecef8d317`.

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
