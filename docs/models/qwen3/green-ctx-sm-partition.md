# Qwen3-4B Green Context SM partition (concurrent prefill/decode)

> **TL;DR:** Enabling Green Context SM partitioning (`--decode-overlap green-ctx --decode-sm-pct 20`) runs prefill and decode on disjoint SM partitions (decode 32 SM / prefill 138 SM on a 5090's 170 SM) so a decode no longer stalls behind a co-scheduled prefill in the unified step. On the 5090 mid-band (`vllm bench serve` random in=1024/out=128, QPS 8–12) this **halves ITL p99** (44.8→22.5 / 46.7→24.7 / 56.6→26.9 ms) and cuts TPOT mean (up to −22% @QPS12) — at the cost of a **2–4× TTFT regression** (prefill loses SMs *and* is deferred to the next scheduler iteration). Restoring decode's CUDA graph under the partition (the "two-graph" change) adds a further **~5% ITL p99 / 1–4% TPOT** with zero correctness cost. Net: this is a TTFT↔ITL/TPOT trade, not a free win — turn it on only when steady-state token smoothness matters more than first-token latency.
>
> **Last touched:** 2026-07

Select the prefill/decode overlap mode with one CLI flag (`--decode-overlap`, off by default):

| `--decode-overlap` | what it does |
|---|---|
| `off` | single stream; prefill and decode serialize. Lowest TTFT. |
| `stream` | two CUDA streams sharing all SMs — concurrency, no SM partition. |
| `green-ctx` | two streams pinned to disjoint Green Context SM partitions; `--decode-sm-pct <N>` (default 20) sets decode's share. |

`green-ctx` fails loudly: if the driver/VRAM combo rejects SM-pinned streams (or the requested split isn't viable for the SM granularity), startup aborts rather than silently degrading to the shared path, so a benchmark always measures the mode you asked for. The stream-creation failure suggests `--decode-overlap stream`; a non-viable split reports the SM counts so you can retune `--decode-sm-pct`. The public knob is the `DecodeOverlap` enum (`green_ctx.rs`), threaded through `start_engine_with_offload`. Implementation: `green_ctx.rs` (`OverlapStreams` — partition + streams), `executor.rs` `SplitConcurrent` (the split step), `scheduler.rs` (when it fires).

## When the split actually fires

`SplitConcurrent` is taken **only for a Unified step** — one that contains both prefill and decode work (the `has_decode_overlap() && Unified` gate in `scheduler.rs`). A lone request (prefill then pure decode) never splits; its decode runs on the full-SM path with the normal CUDA graph. So the partition only matters under mixed load — exactly the mid-band (QPS 8–12 here, ~1800 split steps over a 3×60 s sweep). Below that there is nothing to overlap; well above it the box is decode-throughput-bound and the picture changes again (not measured here).

In the split step the scheduler **syncs decode and returns immediately, deferring the prefill**: prefill kernels stay in flight on the prefill stream and are polled at the top of the next iteration (`executor.rs` `InflightPrefillState`). This is *why* TTFT regresses so hard at high QPS — prefill is both squeezed onto fewer SMs and pushed behind decode. The decode-side win and the TTFT loss have the same root cause (decode is prioritized), so they move together; tuning `decode_pct` rebalances but cannot remove the trade.

## The two-graph change: CUDA graph under the partition

Before: the split path forced `enable_cuda_graph = false`, so decode ran eager whenever it was co-scheduled with prefill — losing the ~0.8 ms/step graph-launch saving exactly in the regime the partition targets.

The fix rests on one CUDA rule (Programming Guide §4.6.5, mirrored in `docs/lessons/cuda-green-contexts.md`): **stream capture binds each kernel node to the execution context of the stream it is captured on.** A decode graph captured *on the green decode stream* therefore replays on the decode partition's SMs regardless of which stream launches it. So:

- `CudaGraphState` (pegainfer-core) now captures/replays on `active_cu_stream(ctx)` — the thread-local stream override, which is the green decode stream inside the split step and `ctx.stream` everywhere else. Rewritten on the raw driver API (`cuStreamBeginCapture_v2`/`cuStreamEndCapture`/`cuGraphInstantiateWithFlags`/`cuGraphLaunch`) because cudarc's `CudaGraph` is bound to a `CudaStream` object and the green stream is a bare `CUstream`. Behaviour with no override is identical to before, so qwen35 / kimi-k2 are unaffected.
- A graph captured on `ctx.stream` (full-SM, primary ctx) and one captured on the green decode stream are different objects, so `BatchDecodeBuffers` keeps **two** caches — `graphs` and `graphs_split` — selected by `has_stream_override()`. Same decode buffers back both, so pointer stability holds for both.

Why this is safe even though the per-step H2D (token_ids/positions/paged meta) lands on `ctx.stream` while the graph replays on the decode stream: a producer fence orders that H2D ahead of the decode stream's read (`fence_producers_before_override`, see Pitfalls). Graph replay submits the same kernels to the same stream, so it inherits the same ordering. The H2D sits outside the captured region (it runs before `run_or_capture`), so capture never sees a copy.

## A/B data (5090, Qwen3-4B, random in=1024/out=128, Poisson seed 42, 60 s/point, idle GPU 0)

| QPS | metric | baseline (no partition) | green-ctx graph OFF | green-ctx graph ON |
|---|---|---|---|---|
| 8 | TPOT mean | 11.03 | 10.49 | **10.07** |
|   | ITL p99 | 44.83 | 22.49 | **21.10** |
|   | TTFT mean | 62.48 | 102.72 | 88.39 |
| 10 | TPOT mean | 13.55 | 12.24 | **12.11** |
|    | ITL p99 | 46.67 | 24.65 | **23.45** |
|    | TTFT mean | 70.41 | 143.95 | 152.50 |
| 12 | TPOT mean | 18.88 | 14.75 | **14.54** |
|    | ITL p99 | 56.58 | 26.89 | **25.51** |
|    | TTFT mean | 92.33 | 327.20 | 372.36 |

Output throughput is within ±1% across all three arms (admission-bound, not kernel-bound). All requests succeeded; the graph-on server logged 0 Xid/capture failures over 1876 split steps.

Reading it: **partition vs baseline** is the big move (ITL p99 roughly halved, TPOT down, TTFT up 2–4×). **graph-on vs graph-off** is the incremental two-graph win — ~5% ITL p99 and 1–4% TPOT, consistent across all three points and above the ±0.3 ms cross-run drift. The magnitude matches the ~0.8 ms launch saving on a ~10–14 ms TPOT. The TTFT ± between the two partition arms is prefill-side run-to-run noise, unrelated to the decode graph.

### Power draw (GPU 0 raised to 600 W enforced limit for this run)

`nvidia-smi power.draw` sampled at 2 Hz across each arm's sweep, loaded samples only (util ~98%):

| arm | mean W | median W | p99 W |
|---|---|---|---|
| baseline | 548.7 | 559.9 | 606.8 |
| green-ctx graph OFF | 551.0 | 563.6 | 606.2 |
| green-ctx graph ON | 555.2 | 568.5 | 606.8 |

Power climbs with QPS: at QPS 8 the board still has headroom (draw oscillates ~510–590 W between request bursts), QPS 10 pushes toward the wall, and **QPS 12 fully saturates the 600 W wall** (p99 ~606 W is instantaneous board power; the limit is enforced on a moving average, so brief overshoot is expected). Across the whole sweep, partition and decode-graph state move average draw by ≤1.2% — graph-on is marginally *higher*, not less efficient: it sustains slightly more SM occupancy doing the same admission-bound work, and at QPS 12 that difference is pinned out by the wall. **The lever this feature pulls is the ITL/TPOT distribution, not the power envelope.** Figure rendered locally with seaborn (not committed; data is throwaway and the conclusion is the table above).

## Pitfalls

- **The Xid 31/43 hit during bring-up was a cross-stream buffer use-after-free; the supported success and checked-`Result` paths are fixed — not an open driver risk.** Prefill temporaries are allocated on `ctx.stream` but consumed by override-stream kernels; the fix is a producer fence (`green_ctx::fence_producers_before_override`, ordering each `ctx.stream` alloc/H2D ahead of its override consumer) plus RAII stream drains that hold each stream until its kernels finish before any buffer frees or KV page is released — `PrefillTempBin` (prefill) and `DecodeStreamGuard` (decode), each fail-closed (a sync failure aborts). An arbitrary panic that unwinds past a not-yet-parked prefill buffer is out of scope.
- **The "`cuGreenCtxStreamCreate` itself faults on driver 590 + >16 GB resident" theory did not reproduce.** This 5090 (driver 590.48.01, ~27 GB resident — 19.3 GB of it KV) ran green-ctx clean across both 2026-06-19 sweeps and the 2026-06-20 three-mode smoke (split path fired, 0 Xid).

- **`gemm_lt` is still disabled under the stream override** (`5af4fd5`, to avoid the cuBLASLt workspace Xid-31 path). So split-path decode keeps its CUDA graph but loses the per-shape Lt tuning — a remaining decode-side lever, not yet re-measured under the partition.
- **A single request never exercises the split path** — smoke-test with concurrent load or you are only testing the full-SM graph.
- **`pkill` from an ssh one-liner matches its own command line** — use `pkill -f "[t]arget/release/pegainfer"`, and kill/launch in separate ssh invocations.
- Build on the 5090 with `CUDA_HOME=/usr/local/cuda-13.1` (stale `/usr/local/cuda` → cuBLAS 12.9 N=1025 cliff; see `serving-perf-5090.md`). Verify `ldd target/release/pegainfer | grep cublas` shows `.so.13`.

## Next

- **The TTFT regression is the gating cost, not a bug.** It is structural (prefill deprioritized + fewer SMs). Before this feature ships on by default it needs either a `decode_pct` sweep to find a TTFT/ITL balance, or a scheduler change so prefill is not starved at high QPS. Not done — `decode_pct=20` is the only split measured.
- Re-enable / re-measure `gemm_lt` under the override now that decode is graphed again.
- Low/high-QPS tails (QPS 1/4/16) unmeasured; the mid-band is where the trade lives.
