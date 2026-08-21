# DeepSeek-V2-Lite Serving Roadmap

> **TL;DR:** DeepSeek-V2-Lite has single-node EP2 correctness, request-lifecycle reliability, direct diagnostics, retained HTTP SLO reporting, and a retained #465 short-shape soak baseline. Production readiness still requires bounded device KV ownership, long-prefill scheduling, ratified budgets, and explicit deployment limits.
>
> Last touched: 2026-07

## Product Boundary

The target is a stable single-node, two-GPU EP2 serving line. Current roadmap work does not claim multi-node EP, transparent rank-loss recovery, sparse EP, broad API parity, or vLLM parity.

Evidence stays in four buckets:

1. Correctness and integration: HF/host-staged/NCCL exactness, EP accounting, request isolation.
2. Direct diagnostics: decode attribution, backend timing, CUDA event and graph-readiness probes.
3. HTTP serving SLO: fixed `/v1/completions` contracts, tails, throughput, failures/timeouts, trace coverage, hashes, repeat spread.
4. Soak and production readiness: sustained memory/tail drift, recovery, capacity, deployment and support limits.

The first three can be green while the fourth is only a retained baseline. A passing soak report records sustained behavior for a named contract; it does not define production budgets by itself.

## Current Gates

| Gate | State | Source of truth | Boundary |
| --- | --- | --- | --- |
| HF and EP2 correctness | Retained | `hf-accuracy-gate.md`, `e2e_ep2.rs` | Correctness only |
| Direct decode attribution | Retained | `decode-attribution-gate.md` | Direct diagnostic only |
| HTTP lifecycle reliability | Retained | `status.md`, issue #453 | Failure isolation and recovery scenarios, no long-duration claim |
| HTTP SLO report | Retained for #466 | `benchmarking.md`, `bench_dsv2lite_http_slo.py` | Fixed host-staged/NCCL HTTP contracts retained; no soak or production claim |
| Sustained soak | Retained baseline for #465 | `benchmarking.md`, `bench_dsv2lite_http_soak.py` | Fixed short-shape host-staged/NCCL soak retained; runtime/provenance gates are hard, drift is reported, not a ratified production budget |
| Long-prefill scheduling | Open | issue #452 | Current long smoke records the boundary; it does not close latency work |
| Device attention and KV | Open | issue #635 | Required for bounded device lifetime and stronger scaling |

## Issue #466 Position

Issue #466 is an evidence and reporting milestone. It provides named DSV2-Lite profiles for short decode-heavy, mixed prompt-shape, and long-prompt smoke workloads across host-staged and NCCL. The retained JSON carries the model/backend metadata, TTFT/TPOT/ITL tails, throughput, failures/timeouts, trace coverage, output hashes, repeat spread, and an HTTP-only claim boundary.

This closes the missing report layer. It does not optimize latency or throughput and does not close issue #465.

## Issue #465 Position

Issue #465 is the sustained evidence milestone. The retained 2026-07-18 run covered host-staged and NCCL under the fixed short-shape HTTP soak contract with zero failures/timeouts, full required trace coverage, clean follow-up recovery, output hashes, resource samples, and first/last-quartile drift fields. The NCCL row used NCCL `2.26.2` with `NCCL_IB_DISABLE=1` and `NCCL_P2P_DISABLE=1`; keep that runtime boundary attached to the evidence.

This closes the first retained soak baseline. It does not set hard latency or drift budgets, does not cover long-prefill scheduling, and does not claim production readiness.

The tooling now guards the evidence boundary directly: backend summaries fail on missing declared-duration coverage, capped retained runs, missing bucket coverage, unreadable leaf artifacts, leaf command errors, missing trace coverage, or failed/missing clean follow-up. The combined report fails on missing host-staged/NCCL children, failed child sub-gates, provenance drift, commit drift, contract drift, or missing/generic runtime boundaries. Long-duration and long/mixed-prompt soaks still need separate versioned contracts before Stable promotion.

## Sequence

### 1. Retain The Serving Contract

- Keep #466 short/mixed/long profiles stable unless a versioned schema or profile change is reviewed.
- Run both host-staged and NCCL after scheduler, frontend, trace, or backend changes.
- Fail retained runs on request failures, timeouts, missing traces, or missing active/decode coverage.
- Keep startup failures as structured artifacts instead of dropping failed cells.

### 2. Extend Sustained Availability

Primary issue: #465.

- Keep the retained host-staged/NCCL short-shape soak reproducible after scheduler, frontend, trace, or backend changes.
- Extend from the short-shape baseline into ratified long-duration and long/mixed-prompt soaks only when those contracts have explicit support limits.
- Calibrate latency, throughput, RSS, and VRAM budgets from repeated retained variance before promoting numeric thresholds to hard gates.
- Keep conservative NCCL runtime selectors in the retained evidence until default-runtime startup is separately proven on the supported hardware matrix.

### 3. Move Decode State To The Device

Primary issue: #635.

- Give each request explicit device KV ownership, capacity, retirement, and slot-reuse semantics.
- Move steady decode attention and compressed KV off the host path.
- Preserve exact output, cancellation cleanup, stable pointers, eager fallback, and graph diagnostics.
- Require paired HTTP evidence under #466 contracts before any performance claim.

### 4. Schedule Long Prefill

Primary issue: #452.

- Add bounded prefill work per scheduler step and protect active decode from starvation.
- Reserve capacity before admission and reject impossible contexts explicitly.
- Use #466 mixed and long rows as regression evidence, then add issue-specific long-context gates where needed.

### 5. Stable Promotion

Promotion requires all of the following:

- HF/host-staged/NCCL correctness remains green;
- request and KV capacity are bounded and observable;
- lifecycle and soak gates recover cleanly with no unexplained drift;
- short, mixed, and long retained SLO reports pass on the supported hardware/runtime matrix;
- long-duration and long/mixed-prompt soak profiles have ratified duration, shape, and drift budgets;
- backend startup failures give actionable version or configuration errors;
- API, topology, context, sampling, and recovery limits are documented;
- performance claims use matched repeated HTTP contracts, while direct and profiler data remain diagnostic.

## Deferred Work

Sparse all-to-all, expert replication, prefix caching, KV offload, multi-node EP, and rank restart should start only after current profiles show they are material and the device-KV ownership contract is stable. Host-staged deprecation requires NCCL correctness, SLO, and soak evidence on every supported hardware/runtime row.
