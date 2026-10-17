# Prometheus /metrics via the vLLM frontend

**TL;DR:** `/metrics` exposes request histograms for every model and engine gauges for schedulers that publish load: Qwen3, Qwen3.5 and Gemma 4 use one logical engine, while GLM5.2 EP8/DP8 uses eight rank-local engines and GLM5.2 TP8 uses one logical engine. The bridge forwards each partition's stats under the same identity the vLLM frontend uses for least-load routing. Qwen3 reports real token-granular prefix-cache query/hit counters, split into local KV and connector-restored tokens; the other lines still publish zero.

Last touched: 2026-10

## How the numbers flow

Two independent paths feed the upstream Prometheus registry (`vllm-metrics`, served by `vllm-server` at `/metrics` with its HTTP middleware counters):

1. **Per-request path (works for every model crate).** The bridge stamps each request's first output with `Queued`/`Scheduled` timestamps and `PrefillStats` (prompt/computed/cached token split). The upstream `RequestMetricsTracker` turns those into `time_to_first_token_seconds`, `inter_token_latency_seconds`, `request_queue_time_seconds`, `prompt_tokens_total`, `generation_tokens_total`, `request_success_total`, `prompt_tokens_by_source_total`, … unconditionally — `disable_log_stats` only gates the periodic *text* logger, not Prometheus.
2. **Engine-gauge path.** Step-contract schedulers publish `SchedulerMetrics { kv_used_blocks, kv_total_blocks, num_running_reqs, num_waiting_reqs }` through the shared driver's post-step metrics cell. The bridge reads the latest cell at startup and when forwarding a step; it is not a snapshot bound to that step's tokens. Metric-only changes send an empty step, so cancellation of the last request can reset the gauges. Unchanged idle iterations send nothing. Legacy schedulers use `LoadSnapshot` watches. Both bridges forward stats in `RequestBatchOutputs`, whose `engine_index` is both the routing identity and the Prometheus `engine` label.

Qwen3.5 exposes one step-contract scheduler for both single-GPU and TP. For legacy engines, `EngineHandle::with_load_watch` keeps the original one-engine contract. A partitioned scheduler uses `with_load_watches`, and the frontend launch declares the same engine count; a mismatch fails startup. GLM5.2 EP8 therefore registers engines 0–7, each bound to its own pending queue and KV pool. TP8 registers only engine 0 because its eight workers mirror one logical request stream.

Historical watch-path measurements found noise-level cost in both covered configurations:

- Qwen3 TPOT: 10.6387 ms (main) vs 10.6395 ms (metrics branch) over 828 tokens.
- GLM5.2 EP8, three-run median at concurrency 64: 1268.58 vs 1264.82 output tok/s (-0.30%); TPOT p50 41.76 vs 41.35 ms.

## Prefix-cache counters

Qwen3 publishes token-granular prefix-cache counters, in four families:

- `vllm:prefix_cache_queries_total` / `..._hits_total` — local KV. A request
  adds its prompt length to `queries` once, on its first prefill chunk, and
  `hits` counts the tokens of that prompt already cached locally.
- `vllm:external_prefix_cache_queries_total` / `..._hits_total` — the
  connector (CPU offload / P2P restore). The query denominator is the prompt
  minus the local hits, so a locally cached token never inflates it, and the
  family is absent when no connector was consulted at all.

A connector miss is counted: `external_queries` advances while
`external_hits` stays flat. Only a request whose connector was never
consulted (`None`, not `Some(0)`) contributes nothing to the external
family, which keeps the hit rate honest instead of reporting only successful
restores.

Qwen3.5, GLM5.2, Kimi-K2 and DeepSeek-V2-Lite have not wired these counters
yet; they still report zero.

## What deliberately reads zero (state at capture time)

- The by-reason waiting split: `reason="deferred"` is driven by a skipped-request counter we don't report, so all waiting shows as `reason="capacity"`.
- Per-GPU FLOPs/bytes estimates, KV-block residency histograms, cudagraph stats — the bridge sends `SchedulerStats::default()` for these fields.
- Every legacy model crate whose scheduler doesn't publish a `LoadSnapshot` watch (currently deepseek and kimi) gets path 1 only; its engine gauges are absent, not lying-zero — the bridge skips the stats task for that partition when no watch exists.

## Next step

Next, wire the DeepSeek-V2-Lite and Kimi-K2 schedulers using the same recipe, and extend the prefix-cache counters to the remaining lines (Qwen3.5, GLM5.2, Kimi and DeepSeek report zero today). A future partitioned model must expose its logical scheduler partitions instead of averaging them behind engine 0.
