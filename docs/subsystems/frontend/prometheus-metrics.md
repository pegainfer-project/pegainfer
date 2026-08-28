# Prometheus /metrics via the vLLM frontend

**TL;DR:** `/metrics` exposes request histograms for every model and engine gauges for schedulers that publish `LoadSnapshot`: Qwen3 and Qwen3.5 use one logical engine, while GLM5.2 EP8/DP8 uses eight rank-local engines and GLM5.2 TP8 uses one logical engine. The bridge forwards each partition's stats under the same identity the vLLM frontend uses for least-load routing.

Last touched: 2026-08

## How the numbers flow

Two independent paths feed the upstream Prometheus registry (`vllm-metrics`, served by `vllm-server` at `/metrics` with its HTTP middleware counters):

1. **Per-request path (works for every model crate).** The bridge stamps each request's first output with `Queued`/`Scheduled` timestamps and `PrefillStats` (prompt/computed/cached token split). The upstream `RequestMetricsTracker` turns those into `time_to_first_token_seconds`, `inter_token_latency_seconds`, `request_queue_time_seconds`, `prompt_tokens_total`, `generation_tokens_total`, `request_success_total`, `prompt_tokens_by_source_total`, … unconditionally — `disable_log_stats` only gates the periodic *text* logger, not Prometheus.
2. **Engine-gauge path (needs one `LoadSnapshot` watch per scheduler partition).** The scheduler publishes `LoadSnapshot { kv_used_blocks, kv_total_blocks, num_running_reqs, num_waiting_reqs }` at scheduler boundaries; one bridge identity per partition forwards its snapshot as a stats-only `RequestBatchOutputs`. The enclosing `engine_index` is both the routing identity and the Prometheus `engine` label. Watches coalesce to ≤1 message per scheduler step, and the scheduler's final idle publish settles the gauges back to 0.

One logical request stream is one engine. Qwen3 and Qwen3.5 publish `Scheduler::metrics` after each `step` (single-GPU and TP share that one stream); the stepped bridge pulls the snapshot once per driver iteration. Legacy-handle lines still attach `EngineHandle::with_load_watch`. A partitioned scheduler uses one watch per partition, and the frontend launch declares the same engine count; a mismatch fails startup. GLM5.2 EP8 therefore registers engines 0–7, each bound to its own pending queue and KV pool. TP8 registers only engine 0 because its eight workers mirror one logical request stream.

Measured cost is noise in both covered configurations:

- Qwen3 TPOT: 10.6387 ms (main) vs 10.6395 ms (metrics branch) over 828 tokens.
- GLM5.2 EP8, three-run median at concurrency 64: 1268.58 vs 1264.82 output tok/s (-0.30%); TPOT p50 41.76 vs 41.35 ms.

## What deliberately reads zero (state at capture time)

- `prefix_cache_queries/hits` and the by-reason waiting split (`reason="deferred"` is driven by a skipped-request counter we don't report; all waiting shows as `reason="capacity"`).
- Spec-decode counters, per-GPU FLOPs/bytes estimates, KV-block residency histograms, cudagraph stats — the bridge sends `SchedulerStats::default()` for these fields.
- Every model crate whose scheduler doesn't publish a `LoadSnapshot` watch (currently deepseek and kimi) gets path 1 only; its engine gauges are absent, not lying-zero — the bridge skips the stats task for that partition when no watch exists.

## Validated coverage and next step

Qwen3.5 single-GPU live RTX 5090 validation confirmed that running and KV gauges rise during generation, waiting rises under batch-slot pressure, and all three return to zero after drain and recovery. The commands and metric samples are recorded in [Qwen3.5 Scheduler LoadSnapshot](../../models/qwen35/load-snapshot.md#validation-boundary). TP uses the same scheduler publication path but was not part of that live run.

Next, wire the DeepSeek-V2-Lite and Kimi-K2 schedulers using the same recipe, and report real prefix-cache query/hit counters instead of zeros. A future partitioned model must expose its logical scheduler partitions instead of averaging them behind engine 0.
