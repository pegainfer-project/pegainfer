# Prometheus /metrics via the vLLM frontend

**TL;DR:** `/metrics` exposes request histograms for every model and engine gauges for schedulers that publish load: Qwen3, Qwen3.5 and Gemma 4 use one logical engine, while GLM5.2 EP8/DP8 uses eight rank-local engines and GLM5.2 TP8 uses one logical engine. The bridge forwards each partition's stats under the same identity the vLLM frontend uses for least-load routing.

Last touched: 2026-10

## How the numbers flow

Two independent paths feed the upstream Prometheus registry (`vllm-metrics`, served by `vllm-server` at `/metrics` with its HTTP middleware counters):

1. **Per-request path (works for every model crate).** The bridge stamps each request's first output with `Queued`/`Scheduled` timestamps and `PrefillStats` (prompt/computed/cached token split). The upstream `RequestMetricsTracker` turns those into `time_to_first_token_seconds`, `inter_token_latency_seconds`, `request_queue_time_seconds`, `prompt_tokens_total`, `generation_tokens_total`, `request_success_total`, `prompt_tokens_by_source_total`, … unconditionally — `disable_log_stats` only gates the periodic *text* logger, not Prometheus.
2. **Engine-gauge path.** Step-contract schedulers publish `SchedulerMetrics { kv_used_blocks, kv_total_blocks, num_running_reqs, num_waiting_reqs }` through the shared driver's post-step metrics cell. The bridge reads the latest cell at startup and when forwarding a step; it is not a snapshot bound to that step's tokens. Metric-only changes send an empty step, so cancellation of the last request can reset the gauges. Unchanged idle iterations send nothing. Legacy schedulers use `LoadSnapshot` watches. Both bridges forward stats in `RequestBatchOutputs`, whose `engine_index` is both the routing identity and the Prometheus `engine` label.

Cumulative counters ride the same snapshot and the bridge diffs consecutive snapshots into the per-interval deltas vLLM `inc_by`s. `spec_decode` feeds `vllm:spec_decode_*`. `prefix_cache: Option<PrefixCacheCounters>` (prompt tokens queried and hit, plus connector lookups and the hits it restored) is split by tier the way vLLM's own offload connector reports it: hits not restored through the connector (the device tier) go through `SchedulerStats::prefix_cache_stats` into `vllm:prefix_cache_{queries,hits}_total`, connector lookups and hits through `connector_prefix_cache_stats` into `vllm:external_prefix_cache_{queries,hits}_total`, and `prompt_tokens_cached_by_source` gets `device` for the former and, for the latter, `host` when the engine knows the hits came from host memory (`host_hits`) or `external_unspecified` when it cannot say (`external_unspecified_hits`), so it sums to `prefix_cache_hits + external_prefix_cache_hits`. Only hits carry a source; the external query count is source-agnostic. A total that goes backwards (engine restart) restarts the delta from zero. Pitfall: the pinned vLLM frontend credits every local hit to `source="device"` and ignores the local `hits_by_source`, which is why host hits must not ride the local stats.

Qwen3.5 exposes one step-contract scheduler for both single-GPU and TP. For legacy engines, `EngineHandle::with_load_watch` keeps the original one-engine contract. A partitioned scheduler uses `with_load_watches`, and the frontend launch declares the same engine count; a mismatch fails startup. GLM5.2 EP8 therefore registers engines 0–7, each bound to its own pending queue and KV pool. TP8 registers only engine 0 because its eight workers mirror one logical request stream.

Historical watch-path measurements found noise-level cost in both covered configurations:

- Qwen3 TPOT: 10.6387 ms (main) vs 10.6395 ms (metrics branch) over 828 tokens.
- GLM5.2 EP8, three-run median at concurrency 64: 1268.58 vs 1264.82 output tok/s (-0.30%); TPOT p50 41.76 vs 41.35 ms.

## Qwen3 prefix-cache counters

Qwen3 fills `SchedulerMetrics::prefix_cache` with token-granular totals, counted once per request on its first prefill chunk:

- `vllm:prefix_cache_{queries,hits}_total` — local KV. `queries` adds the prompt length, `hits` the tokens of that prompt already cached locally. A request whose lookup never ran (prefix cache off, or prompt scoring, which forwards the prompt whole) adds nothing, so the rate is not diluted by lookups that did not happen.
- `vllm:external_prefix_cache_{queries,hits}_total` — the offload connector (CPU tier, P2P, SSD). The denominator is the prompt minus the local hits **at the moment the connector was asked**: a request queries the connector once and may sit deferred while another request registers the same prefix locally, and that growth must neither shrink the query already sent nor be counted as a restore. Only blocks a connector load actually committed are external hits.
- A connector miss, a query error and a released lease still count their query (`external_queries` advances while `external_hits` stays flat); only a request whose connector was never consulted adds nothing to the external family. Cancellation settles the same way, so a lookup is neither lost nor counted twice.
- Connector hits are exported as `source="external_unspecified"`, never `host`. A connector query returns a block count only — the blocks can come from host memory, a P2P peer or SSD, and nothing says which — so labeling them `host` would misattribute the peer and disk restores. Carrying the real tier through needs it reported by the offload layer first.

## What deliberately reads zero (state at capture time)

- `prefix_cache_queries/hits` and the external family for every in-tree model line except Qwen3 (the others leave `SchedulerMetrics::prefix_cache` at `None`; an out-of-tree `Scheduler` can fill it), and the by-reason waiting split (`reason="deferred"` is driven by a skipped-request counter we don't report; all waiting shows as `reason="capacity"`).
- Per-GPU FLOPs/bytes estimates, KV-block residency histograms, cudagraph stats — the bridge sends `SchedulerStats::default()` for these fields.
- Every legacy model crate whose scheduler doesn't publish a `LoadSnapshot` watch (currently deepseek and kimi) gets path 1 only; its engine gauges are absent, not lying-zero — the bridge skips the stats task for that partition when no watch exists.

## Next step

Next, wire the DeepSeek-V2-Lite and Kimi-K2 schedulers using the same recipe, and have the remaining in-tree engines with a prefix cache (Gemma 4's `PEGAINFER_PREFIX_CACHE`, Qwen3.5, GLM5.2) fill `SchedulerMetrics::prefix_cache`, and have the offload layer report which tier answered so Qwen3 can use `host`, `p2p` and `disk` instead of `external_unspecified`. A future partitioned model must expose its logical scheduler partitions instead of averaging them behind engine 0.
