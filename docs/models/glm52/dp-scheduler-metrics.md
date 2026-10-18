# GLM5.2 DP scheduler metrics

**TL;DR:** GLM5.2 now maps each logical scheduler partition to vLLM's existing EngineCore identity: EP8/DP8 exposes eight rank-local running/waiting/KV series and uses frontend least-load routing, while TP8 exposes one series. Endpoint-level gates passed on 8x H200, and a matched three-run serving A/B found no measurable regression; upstream vLLM required no change.

Last touched: 2026-07

## Preparation

### Request and acceptance boundary

Expose the GLM5.2 scheduler gauges already supported by the vLLM Rust frontend:

- `num_running_reqs`
- `num_waiting_reqs`
- `kv_cache_usage`

The representation must match the real scheduling topology. EP8/DP8 must expose eight rank-local scheduler series; TP8 has one logical rank and must expose one series. Validate the result on 8x H200, measure matched performance before review, and submit the PegaInfer change as a PR. An upstream vLLM change is allowed only when the existing frontend cannot express this topology, and must be reviewed separately before implementation.

### Sources read

- `docs/index.md`
- `docs/subsystems/frontend/prometheus-metrics.md`
- `docs/models/glm52/continuous-batching.md`
- `docs/models/glm52/serving-status.md`
- `docs/conventions/bench-regression.md`
- `pegainfer-engine/src/engine.rs`
- `pegainfer-vllm-frontend/src/{lib,bridge}.rs`
- `pegainfer-glm52/src/{lib,scheduler/mod,scheduler/admission}.rs`
- vLLM latest `main` at public commit `c241c7a2b015f8168d1c75e80cee15e45c18ba94` (2026-07-10), especially its Rust engine-core client, metrics recorder, Python DP coordinator, EngineCore, scheduler, and metrics logger
- The vLLM revision pinned by this workspace, confirming the required multi-engine routing and per-engine metrics mechanisms are already available to PegaInfer

### Findings

vLLM DP8 already has the desired model:

1. It launches eight EngineCore instances.
2. Each EngineCore owns one scheduler, waiting queue, and KV manager.
3. Each engine emits one `SchedulerStats`; the enclosing output carries `engine_index`.
4. The frontend registers `engine=0..7`, records eight Prometheus series, and routes new requests using a least-load score based on waiting and running counts.
5. Aggregated text logging sums running/waiting and averages KV usage, but Prometheus retains the eight rank-local series.

No upstream vLLM change is required. PegaInfer is the mismatch:

- The local frontend config hard-codes `engine_count=1`.
- The bridge hard-codes `engine_index=0` and reports `data_parallel_size=1`.
- GLM5.2 chooses a rank inside its coordinator from one global pending queue.
- GLM5.2 exposes no load watches.

Sending eight stats-only batches while keeping one registered engine would not work: the frontend drops stats for unknown engine indexes. Directly writing Prometheus families would also leave request routing and waiting ownership inconsistent. The truthful integration is to register the logical DP ranks as the engines the frontend already knows how to route.

### Invariants

- EP8/DP8 registers exactly eight engine identities; TP8 registers exactly one.
- A request selected for frontend engine `r` remains bound to logical rank `r`; the coordinator cannot silently reassign it.
- Each rank owns its waiting count, running count, KV used blocks, and KV total blocks.
- `sum(num_waiting_reqs)` equals the coordinator's total queued requests; `sum(num_running_reqs)` equals occupied request slots.
- KV usage is `rank_used_blocks / rank_total_blocks`, never a fleet average presented as a rank value.
- TP8's eight mirrored workers remain one scheduler partition and one KV series.
- The central coordinator still chooses global step shapes and drives every worker in lock-step; only request ownership moves to the existing frontend least-load boundary.
- Direct `EngineHandle` callers without a selected DP rank retain deterministic least-load placement.
- Other model lines remain single-engine unless they explicitly opt into partitioned registration.
- An engine-count mismatch between the frontend declaration and the resolved handle fails startup rather than degrading to aggregate metrics.

## Implementation

1. The shared engine contract carries explicit scheduler partitions:
   - a request may carry an optional target DP rank;
   - an `EngineHandle` may publish one load watch per partition;
   - existing single-engine constructors and callers remain the one-partition case.
2. The local vLLM bridge supports multiple engine identities:
   - accept a launch-time engine count without waiting for model loading, preserving concurrent tokenizer/model startup;
   - start one bridge identity per partition on the shared transport;
   - use that identity for requests, outputs, ready metadata, and its rank-local stats watch;
   - validate the handle exposes the declared number of partitions.
   - supervise the output sender and scheduler-stats publisher as part of the bridge lifecycle; a closed load feed, output failure, or scheduler-submit failure tears down the endpoint instead of leaving a registered but unusable rank.
   - keep request validation failures local: malformed public request extensions receive a terminal error, while transport, output, scheduler-submit, and load-feed failures remain endpoint-fatal.
3. The GLM5.2 coordinator owns one pending queue per logical rank:
   - HTTP requests arrive already bound by the vLLM frontend's least-load choice;
   - direct unbound requests are assigned once at intake;
   - admission stays FIFO within a rank and preserves full-lifetime KV reservation;
   - global step planning continues over all rank slots.
4. The coordinator publishes one snapshot per logical rank at idle and every scheduler boundary. KV usage comes from that rank's pool and excludes its reserved padding block.
5. Focused tests prove:
   - eight bridge identities produce eight independent stats series;
   - rank-bound requests are not reassigned;
   - running/waiting sums are exact across queue/admit/finish transitions;
   - KV accounting is per pool;
   - TP8 exposes one partition;
   - scheduler metrics update live and return to zero, while a closed rank-local load feed fails the endpoint;
   - a malformed request extension rejects only that request and a following valid request still completes;
   - single-engine model behavior is unchanged.
6. The endpoint refuses to start when the declared engine count and the resolved handle's scheduler partition count disagree.

The public HTTP surface remains one endpoint. The eight EP8 identities are internal routing and metrics identities connected through the existing shared vLLM transport; they are not eight HTTP servers.

## Validation

### Release gates

- `pegainfer-engine`: 10 passed.
- `pegainfer-vllm-frontend`: 22 passed.
- CPU frontend E2E, including live per-engine series, closed-feed lifecycle, request-local validation failure, and fail-fast topology mismatch coverage: 9 passed.
- `pegainfer-glm52`: 56 passed, 14 hardware oracle tests ignored by their explicit annotations.
- GLM5.2-enabled server release check and release build passed.

### 8x H200 endpoint gates

EP8/DP8 exposed exactly eight series for each requested gauge, labelled `engine="0"` through `engine="7"`:

| Workload state | Per-engine evidence |
| --- | --- |
| Idle | running=0, waiting=0, KV=0 for all eight engines |
| One long request | engine 0 running=1 and KV=0.011538; the other seven remained zero |
| Eight concurrent requests | every engine running=1 and KV=0.009615 |
| 72 concurrent requests | every engine running=8 and waiting=1; KV stayed rank-local |
| Drained | all 72 requests succeeded; all three gauges returned to zero on every engine |

TP8 exposed exactly one `engine="0"` series. A long request moved it to running=1 and KV=0.019231, then both returned to zero. This confirms the eight mirrored GPU workers remain one logical scheduler/KV partition.

A 10 Hz diagnostic trace across 451 busy samples observed no rank-routing skew: the maximum difference between any two engines' running+waiting counts was one, and no sample queued while the 64-request workload was at its configured concurrency limit.

### Performance A/B

The performance gate used vllm-bench `cfa8044c` against the real OpenAI completions endpoint on the same 8x H200 host. Each run used 256 requests, exact 64-token inputs, exact 256-token outputs, concurrency 64, 16 warmups, seed 20260710, greedy sampling, and EOS suppression. Both variants completed 768/768 measured requests across three runs.

| Three-run median, steady state | Main baseline | Per-rank metrics | Delta |
| --- | ---: | ---: | ---: |
| Output throughput | 1268.58 tok/s | 1264.82 tok/s | -0.30% |
| TPOT p50 | 41.76 ms | 41.35 ms | -0.97% |
| TTFT p50 | 2353.64 ms | 2349.74 ms | -0.17% |
| TTFT p99 | 2378.98 ms | 2373.87 ms | -0.21% |

The throughput delta is below run-to-run noise, while token latency is unchanged to slightly lower. One candidate run had four late completions; an additional instrumented run completed normally at 1269.89 tok/s and the per-engine trace stayed balanced, so the isolated tail was not attributed to the change. vllm-bench's coarse top-level `max_concurrent_requests=128` was not used: its inclusive one-second buckets overlap adjacent 64-request waves, while its exact steady-state event window correctly reported 64.

## Execution log

- 2026-07-10: plan approved. Implementation started on `feat/glm52-metrics`; upstream vLLM remains unchanged.
- 2026-07-10: local release gates passed, followed by EP8 and TP8 endpoint validation on 8x H200.
- 2026-07-10: matched three-run serving A/B passed the no-regression gate; evidence recorded before review.
- 2026-07-10: review found that detached bridge output/stats tasks could fail without unregistering their engine identity. The bridge now supervises both tasks and the CPU E2E drives live snapshots, verifies drain-to-zero, and proves a closed load feed tears down the endpoint.
- 2026-07-10: follow-up review found request validation and infrastructure errors shared one propagation path. Malformed request extensions now terminate only their request; an E2E proves the endpoint accepts the next valid request.
- 2026-07-10: final toxic review passed after both findings were fixed; release tests, targeted `-D warnings` lint gates, formatting, and diff checks are green.

## Debrief

The important boundary is scheduler ownership, not endpoint count. EP8 has eight independent admission queues and KV pools even though clients see one HTTP endpoint, so it needs eight `SchedulerStats` identities. TP8 has one request stream mirrored across eight workers, so it needs one identity. Reusing vLLM's EngineCore registry keeps routing, request ownership, and Prometheus labels consistent; adding metrics as a separate aggregation layer would have made those three views disagree.

No upstream vLLM change was needed. Its pinned Rust frontend already provides per-engine registration, frontend-local in-flight least-load routing, scheduler-stat gauges, and explicit DP-rank routing. PegaInfer only needed to expose its real logical partitions to that contract.
