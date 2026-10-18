# DeepSeek-V2-Lite Benchmark Artifact Manifest

> **TL;DR:** Issue #467 is implemented: the DeepSeek-V2-Lite retained benchmark matrix emits `artifact_manifest.json` and `regression_summary.json`; PR #617 review follow-ups are addressed, including conservative comparability for setup failures, stable GPU metadata, and added/missing benchmark rows or cells.
>
> **Last touched:** 2026-07

## Implementation Summary

`scripts/bench_dsv2lite_vllm_matrix.py` now emits two audit artifacts beside the existing `summary.json` for both fresh benchmark runs and `--summarize-only` rebuilds:

| File | Role |
| --- | --- |
| `artifact_manifest.json` | Lists the run metadata, benchmark contract, model config/tokenizer hashes, version probes, backend commands, claim rows, artifact paths, artifact sha256 values, `summary_sha256`, `regression_summary_sha256`, and `artifact_bundle_sha256`. |
| `regression_summary.json` | Compares the run to an optional `--baseline-summary` and classifies correctness, direct diagnostic rows, HTTP pressure cells, trace cells, and failed setup rows. |

The manifest records correctness artifacts, direct diagnostic JSON, HTTP result JSON, server logs, trace result JSON, `summary.json`, and `regression_summary.json`. Paths are artifact-bundle-relative or repo-relative when possible; external absolute model paths are reduced to `<external>/<basename>`. Command/env/log payloads continue to use the script's redaction helpers.

The regression summary emits `comparability.claim_marker: "no directional claim"` when a speed/regression direction would be unsafe: no baseline, added/resolved/preserved setup rows, added/missing direct diagnostic rows, added/missing HTTP pressure cells, added/missing trace cells, changed contract/version/model probes, NCCL version drift, or noisy HTTP cells. GPU probe comparability uses stable GPU name, driver version, and compute capability from `nvidia-smi`, ignoring temperature/clocks drift.

## Validation

CPU-only checks:

```bash
python3 -m py_compile scripts/bench_dsv2lite_vllm_matrix.py
python3 -m unittest tests/test_bench_dsv2lite_vllm_matrix.py
python3 scripts/bench_dsv2lite_vllm_matrix.py --plan-only --baseline-summary target/benchmarks/previous/summary.json
```

The final unit run passed `59` tests. The new coverage includes `--summarize-only` manifest/regression emission, public-safe external model paths, artifact-bundle-relative manifest paths, no-baseline `no directional claim`, failed setup resolution and preservation, baseline-present/current-missing direct/http/trace rows, dynamic GPU telemetry drift, stable GPU identity changes, NCCL version drift, and noisy HTTP cells blocking directional claims.

## Execution Log

### PR #617 review follow-up

- `scripts/bench_dsv2lite_vllm_matrix.py`: added section-scoped comparability reasons for added/missing direct diagnostic rows, HTTP pressure cells, and PegaInfer trace cells.
- `tests/test_bench_dsv2lite_vllm_matrix.py`: added a CPU-only regression test for the maintainer-reported case where the baseline contains `nccl/batch1`, `pegainfer-nccl/c1` HTTP, and `pegainfer-nccl/c1` trace rows but the current summary omits them.
- Validation: `py_compile`, unit tests, and the documented plan-only command all passed locally.

## Debrief

- **Outcome**: PR #617's remaining maintainer blocker is fixed; structural benchmark row/cell changes now force `no directional claim`.
- **Pitfalls encountered**: `compare_projection_maps` already reported structural deltas, so the bug was only in the comparability reason aggregation.
- **Lessons learned**: Regression summaries need two layers of checks: value changes for reporting, and row-set changes for claim safety.

## Notes

- `summary.json` remains compatible. The script writes `summary.json`, then `regression_summary.json`, then `artifact_manifest.json` to avoid circular hashes.
- `--baseline-summary` is optional. Without it, the run is still auditable but not directionally comparable.
- After the next real retained GPU run, attach or commit the generated `artifact_manifest.json` and `regression_summary.json` if the team wants a public example.
