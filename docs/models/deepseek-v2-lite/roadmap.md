# DeepSeek-V2-Lite Serving Roadmap

> **TL;DR:** DeepSeek-V2-Lite has single-node EP2 correctness, request-lifecycle reliability, direct diagnostics, and retained HTTP SLO reporting. Production readiness still requires sustained soak, bounded device KV ownership, long-prefill scheduling, and explicit deployment limits.
>
> Last touched: 2026-07

## Product Boundary

The target is a stable single-node, two-GPU EP2 serving line. Current roadmap work does not claim multi-node EP, transparent rank-loss recovery, sparse EP, broad API parity, or vLLM parity.

Evidence stays in four buckets:

1. Correctness and integration: HF/host-staged/NCCL exactness, EP accounting, request isolation.
2. Direct diagnostics: decode attribution, backend timing, CUDA event and graph-readiness probes.
3. HTTP serving SLO: fixed `/v1/completions` contracts, tails, throughput, failures/timeouts, trace coverage, hashes, repeat spread.
4. Soak and production readiness: sustained memory/tail drift, recovery, capacity, deployment and support limits.

The first three can be green while the fourth remains open.

## Current Gates

| Gate | State | Source of truth | Boundary |
| --- | --- | --- | --- |
| HF and EP2 correctness | Retained | `hf-accuracy-gate.md`, `e2e_ep2.rs` | Correctness only |
| Direct decode attribution | Retained | `decode-attribution-gate.md` | Direct diagnostic only |
| HTTP lifecycle reliability | Retained | `status.md`, issue #453 | Failure isolation and recovery scenarios, no long-duration claim |
| HTTP SLO report | Retained for #466 | `benchmarking.md`, `bench_dsv2lite_http_slo.py` | Fixed host-staged/NCCL HTTP contracts retained; no soak or production claim |
| Sustained soak | Open | issue #465 | Required before Stable promotion |
| Long-prefill scheduling | Open | issue #452 | Current long smoke records the boundary; it does not close latency work |
| Device attention and KV | Done for NCCL decode | issue #635, `device-attention.md` | Prefill stays on the host |

## Issue #466 Position

Issue #466 is an evidence and reporting milestone. It provides named DSV2-Lite profiles for short decode-heavy, mixed prompt-shape, and long-prompt smoke workloads across host-staged and NCCL. The retained JSON carries the model/backend metadata, TTFT/TPOT/ITL tails, throughput, failures/timeouts, trace coverage, output hashes, repeat spread, and an HTTP-only claim boundary.

This closes the missing report layer. It does not optimize latency or throughput and does not close issue #465.

## Sequence

### 1. Retain The Serving Contract

- Keep #466 short/mixed/long profiles stable unless a versioned schema or profile change is reviewed.
- Run both host-staged and NCCL after scheduler, frontend, trace, or backend changes.
- Fail retained runs on request failures, timeouts, missing traces, or missing active/decode coverage.
- Keep startup failures as structured artifacts instead of dropping failed cells.

### 2. Close Sustained Availability

Primary issue: #465.

- Run host-staged and NCCL for ratified short and long durations.
- Track first/last-quartile tails and throughput, RSS/VRAM drift, active/pending state, terminal reasons, and clean follow-up recovery.
- Calibrate budgets from retained variance before promoting numeric thresholds to hard gates.

### 3. Move Decode State To The Device

Primary issue: #635. Done for NCCL steady decode; see `device-attention.md`.

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
- backend startup failures give actionable version or configuration errors;
- API, topology, context, sampling, and recovery limits are documented;
- performance claims use matched repeated HTTP contracts, while direct and profiler data remain diagnostic.

## Deferred Work

Sparse all-to-all, expert replication, prefix caching, KV offload, multi-node EP, and rank restart should start only after current profiles show they are material and the device-KV ownership contract is stable. Host-staged deprecation requires NCCL correctness, SLO, and soak evidence on every supported hardware/runtime row.
