# pegainfer-sim high-concurrency vllm-bench

> **TL;DR:** Same-session rust `vllm-bench` A/B: stepped sim vs `main` (legacy handle). Both 0-fail to c=1024. Feat TTFT is worse (serialized admit); TPOT is ~30–180× better. E2EL/throughput win at c=64 and c=1024; mid-band is a wash.
>
> **Last touched:** 2026-08

## Preparation

- **Read**:
  - `docs/index.md` — sim and frontend CPU baseline live under subsystems/frontend.
  - `docs/subsystems/frontend/simulated-inference-engine.md` — sim is a step-contract CPU harness; TTFT/TPOT are configured delays, not model time.
  - `docs/subsystems/frontend/cpu-profiling-baseline.md` — old path at c=16 added ~150ms TTFT over a 5ms floor; no single CPU hotspot.
  - `docs/subsystems/frontend/sim-step-contract.md` — sim now emits one token per request per step and parks ≤1ms.
- **Relevant history**:
  - Previous baseline used Python `bench_http_serving.py` at c=16, not rust vllm-bench, and ran on the legacy per-request tokio tasks.
- **Plan**:
  1. Start `pegainfer-sim` with TTFT=0 / TPOT=0 so any latency is frontend + scheduler-loop overhead.
  2. Smoke with rust `vllm-bench`.
  3. Scale concurrency 64 → 256 → 1024 on `/v1/completions`.
  4. Record TTFT/TPOT/failures and server CPU; name the limiter if one shows up.
- **Risks**:
  - Zero-delay sim still emits one token per step with a 1ms park when anything is waiting; that itself can cap output tok/s.

## Execution Log

### Setup
- `pegainfer-sim` on a loopback port, Qwen3-4B tokenizer only (no weights), `--base-ttft-ms 0 --tpot-ms 0` so any latency is frontend + scheduler loop.
- Client: rust `vllm-bench`, `--backend openai`, 128-in / 64-out, `--ignore-eos`, `--extra-body '{"min_tokens":null}'`.

### Smoke
- c=4, 8 prompts: 0 fail, TTFT p50 1.38ms, TPOT 0.01ms, 17k req/s (too short to trust throughput).

### Sweep (prompts = 4 × concurrency)

| c | ok/fail | req/s | out tok/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) |
| --- | --- | --- | --- | --- | --- |
| 64 | 256/0 | 743 | 47.6k | 35 / 190 / 262 | 0.05 |
| 256 | 1024/0 | 1591 | 102k | 138 / 168 / 245 | 0.04 |
| 512 | 2048/0 | 1493 | 96k | 327 / 403 / 418 | 0.05 |
| 1024 | 4096/0 | 1709 | 109k | 581 / 644 / 682 | 0.05 |

Throughput plateaus from c=256. TTFT ≈ 0.55 ms × concurrency. TPOT does not move.

### Thread sample at c=1024
- `pegainfer-sim-0` (the contract driver thread) is **100% even when idle** (`spin_loop`).
- Mid-burst still only that one thread at 100%. tokio workers and `vllm-zmq-*` stay ~1%.
- Process CPU ~1.2–1.5 cores, RSS peak ~690 MiB. 48-core box is not compute-bound.

### A/B vs `main` (same session, same client, 128/64, zero-delay)

`main` = per-request tokio tasks + `EngineHandle`. Feat = `SimScheduler` + stepped driver. Both binaries built `--release`; feat re-run immediately after main so the numbers are paired.

| c | side | req/s | out tok/s | TTFT p50 | TPOT p50 | E2EL p50 |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | main | 639 | 41k | 3.7 | 1.31 | 86 |
| 64 | feat | 1713 | 110k | 31.3 | 0.04 | 35 |
| 256 | main | 1662 | 106k | 11.5 | 1.61 | 116 |
| 256 | feat | 1725 | 110k | 134 | 0.05 | 137 |
| 512 | main | 1514 | 97k | 51 | 3.97 | 292 |
| 512 | feat | 1570 | 100k | 305 | 0.05 | 309 |
| 1024 | main | 1358 | 87k | 287 | 7.49 | 733 |
| 1024 | feat | 1691 | 108k | 571 | 0.04 | 574 |

`vllm-bench --compare` (A=main, B=feat): c=64 throughput **+168%**, E2EL p50 **−60%**, TTFT p50 **+757%**. c=1024 throughput **+25%**, E2EL p50 **−22%**, TTFT p50 **+99%**. Failures 0/0 both sides.

CPU: feat pins `pegainfer-sim-0` at 100% even idle; main idles cheap (tokio workers ~7% residual after the sweep). Burst CPU similar (~1.3–1.6 cores).

### Unexpected
- `perf` is blocked (`perf_event_paranoid=4`). Attribution is from `ps -L` + the latency shape, not a flamegraph.
- Idle driver spin is by design for GPU engines; on a CPU-only sim it just burns a core.

## Debrief

- **Outcome**: Versus `main`, stepped sim **trades TTFT for TPOT**. Decode is ~30–180× cheaper and stays flat; first token is serialized on the driver and grows ~linear with C. Net E2EL/throughput: feat wins the ends (c=64, c=1024), mid-band is a wash.
- **Pitfalls encountered**: Zero-delay sim still emits one token per request per step; a burst of N admits + first tokens is O(N) work on one thread, which is exactly the linear TTFT. Main spreads that across tokio tasks, so TTFT stays low until the per-request sleep/wake tax shows up as TPOT.
- **Lessons learned**: Do not call the stepped path a frontend win from throughput alone — look at TTFT and E2EL together. The idle `spin_loop` is a sim-only tax.
- **Follow-ups**:
  - Park the driver when idle if sim stays a bench harness.
  - Measure qwen3 the same way before treating the TTFT slope as a production bottleneck.
