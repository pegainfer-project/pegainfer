# Benchmark Regression Tracking

> **TL;DR:** **Retired 2026-08** — the in-process `bench_serving` binary that generated and compared `bench_snapshots/{gpu-slug}/{model}.json` was deleted with the frontend consolidation (it bypassed the serving path entirely and forced `pegainfer-server` to depend on model-crate internals). Regression benching is now HTTP-based; a generic replacement gate has not been rebuilt yet.

## What replaced the tool

- `scripts/bench_http_serving.py` — OpenAI-compatible HTTP harness (streaming TTFT/ITL/TPOT, QPS, error rate, deterministic output hashes) against a running server.
- External `vllm-bench` for multi-turn load tests.

Both measure the *real* serving path (HTTP + bridge + engine), which the deleted in-process tool never did.

## What a rebuilt gate should keep from the old design

The old conventions are worth carrying into any HTTP-based successor:

- One snapshot per model **per GPU** (`bench_snapshots/{gpu-slug}/{model}.json`), git history as the timeline. The historical snapshots under `bench_snapshots/` remain valid as history.
- Standard profiles: prefill-heavy (TTFT-gated), decode-heavy (TPOT-gated), mixed-ITL (tracked, never gated — the stall tail is too noisy for thresholds).
- Gate on **p50 only**: TPOT p50 >2%, TTFT p50 >3%. p99 is shown for eyeballing. A firing threshold means "investigate", not "reject".
- Cross-GPU comparisons are meaningless; only same-GPU across commits counts.
