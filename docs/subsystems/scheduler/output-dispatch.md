# Scheduler output dispatch — GPU bubble & redesign

> **TL;DR:** The scheduler is a single thread that strictly alternates CPU(schedule) → GPU(forward+sample+`ctx.sync()`) → CPU(dispatch), so the GPU sits idle through every non-`run_step` CPU phase. Measured on RTX 5070 Ti / Qwen3-4B (greedy): the bubble was **≈3µs × batch** (bs=128 → ~380µs, 2.0% of an 18ms step), dominated by **`apply_effects` per-request token dispatch (~250µs at bs=128)** — N individual `token_tx.send()` waking N distinct frontend tasks. **Landed (2026-06):** the per-request output fan-out (N channels + N consumer tasks) is collapsed into one request-tagged channel + one demux loop. `GenerateRequest.token_tx` is now a `TokenSink` (engine.rs) — a drop-in over a shared `UnboundedSender<(Arc<str>, TokenEvent)>` + an `Arc<AtomicBool>` cancel flag — so scheduler call sites are unchanged; the bridge runs one `dispatch_burst` demux that buckets a ready burst by request and ships **one** `EngineCoreOutputs` per burst (N→1 wakeups, tasks, ZMQ msgs). Cancellation is the flag, not a new channel: abort flips it + drops the bridge stream entry, the scheduler retires the request on its next emit. Measured A/B vs `main` on this GPU is throughput-neutral-to-slightly-positive (within run-to-run noise at c≤128, a consistent +0.9% at c=192) — exactly the predicted noise-floor win here; the payoff is fast decode GPUs (H20/H100, bubble → 10–15%) or N≫128. (The A/B also caught that the migration had left the `bench_serving` server bins uncompiled — fixed.)
>
> **Last touched:** 2026-06.

## What "GPU bubble" means here

Each scheduler step ends in `select_batch_tokens_into` → `ctx.sync()` (`pegainfer-core/src/ops/sampling.rs`), a full stream sync after sampling. So the worker thread blocks until the GPU finishes the whole forward+sample, and the GPU then goes idle until the *next* step's kernels are enqueued. Everything the scheduler thread does between two `run_step` calls — admission, batch formation, result resolution, token dispatch — runs with the GPU idle. Bubble = per-step CPU time outside the GPU forward.

DB analogy: a single-threaded transaction loop that blocks on each disk I/O (the GPU step) and does all bookkeeping serially around it; the disk (GPU) is idle during the bookkeeping.

## Measurement (how to reproduce)

Temporary instrumentation (reverted from the tree — re-add to re-measure):
1. In `Qwen3Executor::run_step` (executor.rs), bracket the worker round trip with `Instant` and accumulate to a global ns counter (this is "GPU busy" — the step ends in a sync, so this ≈ forward+sample).
2. In `scheduler_loop` (scheduler.rs), bracket per step: `pre` (top-of-loop → before `execute_plan`), `exec` (`execute_plan`), `resolve` (`resolve_step`), `apply` (`apply_effects`). Capture decode batch = `active.len()` before the step. Flush an `info!` summary every ~2s.

Drive load over HTTP: `scripts/bench_http_serving.py --concurrency 128 --num-requests 384 --prompt-words 32 --max-tokens 256 --temperature 0`.

`bubble = pre + exec_cpu + resolve + apply`, where `exec_cpu = exec − run_step(global)`.

## Measured decomposition (RTX 5070 Ti, Qwen3-4B, greedy)

| decode batch | period/step | GPU busy | bubble | bubble % | exec_cpu / resolve / apply |
|---:|---:|---:|---:|---:|---|
| 1   | 10.6 ms | 99.8% | 19 µs  | 0.2% | 7 / 1 / 9 µs |
| 32  | 12.0 ms | 99.3% | 83 µs  | 0.7% | 34 / 3 / 43 µs |
| 64  | 13.1 ms | 98.8% | 157 µs | 1.2% | 58 / 5 / 88 µs |
| 128 | 17.8 ms | 98.0% | 380 µs | 2.0% | 110 / 13 / **254** µs |
| 140 | 20.5 ms | 97.9% | 440 µs | 2.1% | 125 / 14 / **300** µs |

- Bubble ≈ **3µs × batch**, grows ~linearly with concurrency.
- `apply` (apply_effects) dominates: ~250µs of ~380µs at bs=128. Decomposes into ~240µs of N `token_tx.send()` (each wakes one frontend task, ~1.9µs) + ~14µs of the O(N²) `active.iter().position` scan and state writes. The O(N²) scans (`resolve` + `apply`) are negligible now (~14µs total) but quadratic — they bite at bs≳256.
- `exec_cpu` (~0.9µs/req): per-request `schedule_decode`/`apply_decode`/kv_views in the executor.
- **Bubble % is small *here* only because the decode step is huge** (bs=1 = 10.6ms ⇒ memory-bandwidth-bound, 7.67GB weights / ~0.75TB/s). The ~380µs absolute cost is fixed; on a ~4TB/s GPU (bs=128 decode ≈ 3ms) it is ~12%.
- Greedy-only profile. #284 later removed the Qwen per-row non-greedy sampler, so remeasure mixed sampling separately before using this bubble table for non-greedy workloads.

## Why pipelining is *not* the cheap answer (and what it actually costs)

The decode dependency is real: step N+1's forward input token = step N's sampled token. But the dependency does **not** create the bubble — two dependent GPU ops can be back-to-back on the stream with zero gap. The bubble comes from routing the dependency **through the host**: `d2h+sync → CPU decide → h2d`. Async/pipelined scheduling (keep the sampled token on-GPU, feed it to the next step's embedding gather; run host bookkeeping one step behind, overlapped) removes the host from the critical path. Costs: EOS-by-value can't gate the launch, so an EOS'ing request runs ≤1 extra (discarded) decode step (length-stop is count-based and still gates cleanly); sampler must stay on-device; bubble → residual launch gap, and only fully hidden when CPU/step < GPU/step. Real but heavyweight; not justified by a 2% bubble.

## Recommended architecture: single request-tagged output channel + one demux loop

Three facts (investigated 2026-06) reframe the dispatch fix:

1. **Explicit abort already exists.** `EngineCoreRequestType::Abort` → `bridge.rs:143` → `task.abort()` → `token_rx` drop → scheduler notices on its *next* send. Today's "consumer-drop is the cancellation signal" is a lazy 3-hop chain whose last leg is the per-request channel closure. Removing per-request channels means routing abort *directly* to the scheduler (1 hop) — more responsive, not a regression.
2. **The consumer/wire side is already batch-shaped.** `EngineCoreOutputs.outputs` is a `Vec<EngineCoreOutput>`, each already carrying `request_id: String`; everything funnels through one shared `output_tx` → one `output_loop` → one ZMQ socket. The N `run_request_stream` tasks' isolation is real for per-request state, **illusory for throughput** (single downstream socket). The per-request fan-out is vestigial.
3. **The 240µs is N distinct sleeping consumers = N wakeups.** tokio mpsc only wakes its receiver on the empty→non-empty transition. Collapsing N channels into one ⇒ ~1 wake + N cheap pushes — **even if the producer still emits one event at a time.** The lever is "one consumer," not "batch the producer API"; that makes per-model migration mechanical.

### Target shape

```
now:   scheduler ──N× token_tx──> N× run_request_stream task ──> 1 output_tx ──> 1 socket
                  (N wakeups=240µs)    (N tasks, vestigial)       (already single point)

target: scheduler ──1× (RequestId,TokenEvent)──> 1 demux loop ──> 1 socket
                   (1 wake + N pushes)   HashMap<id, RequestState>   coalesce into 1 EngineCoreOutputs
```

- **Producer (engine + each scheduler):** drop `GenerateRequest.token_tx`; carry the external `request_id`. Engine owns one `UnboundedSender<(RequestId, TokenEvent)>`. qwen3's `StepEffects`/`apply_effects` already aggregates per step — change "send each" to "push into the shared sender" (single-emit is enough; per-step batch-send is an optional P3 micro-opt). Keep the external id in `PendingRequest`/`ActiveRequestState` for tagging + cancel lookup.
- **Consumer (bridge):** one demux loop replacing N tasks, holding `HashMap<RequestId, RequestState>` (the existing `first_token_events` / `prefill_stats` / `has_sent_token_output` / `pending` fields). `recv` one + `try_recv`-drain the ready burst → one `EngineCoreOutputs { outputs: Vec<…> }` → one socket send (ZMQ msgs N→1/step). Remove the HashMap entry on terminal event + on abort — **the one place that must be leak-tight.**
- **Cancellation (as shipped):** *not* a new channel — a shared `Arc<AtomicBool>` per request, held by both the bridge stream entry and the request's `TokenSink`. Abort flips the flag (`Release`) and drops the bridge stream entry; the scheduler's next `TokenSink::send`/`is_closed` reads it (`Acquire`) and retires the request — the same reactive retirement the old consumer-drop gave, with no scheduler-side cancel drain and no external→internal id registry. A token already in flight for an aborted id finds no stream entry in the demux and is dropped. Organic disconnect already arrives as an explicit Abort from the in-process vLLM server, which also drops the request from its own tracking, so no post-abort terminal is required (matching the old `task.abort()` behavior).

### Why this over a dispatch thread (Option A)

A dispatch thread that keeps the N per-request channels only **relocates** the N wakeups off the critical path; it keeps N tasks, keeps N ZMQ msgs/step, and is thrown away the moment the real fix lands. The consumer already wants the batch shape, so go straight to it. Both land bubble ≈150µs (floored by `exec_cpu` ~110µs), but the single-channel design also removes ~N−1 wakeups of *total* work.

## Numbers and cost

- bs=128 bubble **380 → ~150µs** (`apply` 254 → ~22), new floor `exec_cpu` ~110µs (batch the KV bookkeeping next). Fast GPU (3ms step): 11% → ~5%.
- Blast radius: 86 `token_tx.send` sites, 5 crates, 12 files, ~400–500 LOC, no shared dispatch abstraction. Mostly mechanical; the only redesign is qwen3 `effects.rs` (~50%) and kimi `dp.rs` (`RequestState` carries `token_tx`, needs request_id tracking). ~1–2 weeks incl. tests.
- Risk: centralized demux concentrates both performance and failure (a per-request task isolates bugs); the `streams` HashMap cleanup must be airtight — entries are removed on the terminal event (demux) and on abort (`streams.remove`), and a request that only ever emits `Scheduled` is held until one of those fires. Cancellation needed no new channel or id registry — the shared `Arc<AtomicBool>` flag carries it.

## Staging — what landed

- **P1 (done):** `TokenSink` contract in `pegainfer-engine/src/engine.rs` + the bridge `dispatch_burst` demux + cancel-flag; qwen3 migrated as the reference. Bridge unit tests cover burst coalescing, token+finish in one burst vs. across bursts, first-token metadata flush-once, lone-Scheduled deferral, multi-request N→1 batching, and aborted-request late-token drop. End-to-end on RTX 5070 Ti / Qwen3-4B: single + streaming + concurrency-32 (0 errors) + mid-stream client disconnect (server stays healthy).
- **Build-fix caught during A/B (done):** the migration originally missed the two `pegainfer-server` `bench_serving` bins (`decode.rs`, `exec.rs`) — the `--lib` test runs never compiled the server binaries, so the default `cargo run --release` build was broken on the branch until those two call sites were swapped to `TokenSink::standalone()` and `pegainfer::scheduler` re-exported `TokenSink`. Lesson: gate on `cargo build --release` (bins), not just `--workspace --lib`.
- **A/B throughput (done, RTX 5070 Ti / Qwen3-4B, greedy, 32-word prompt):** branch vs. stashed `main`, identical KV pool (2473 blocks ≈ 39.5K tokens). End-to-end output tok/s, 3 runs each: c=96/out256 **5594 vs 5551** (+0.8%, within noise); c=128/out128 **7517 vs 7502** (+0.2%, within noise); c=192/out128 **7764 vs 7698** (+0.9%, *non-overlapping* ranges — small but consistent, and growing with concurrency as predicted). 0 failures on both. Note: c=128/out256 over-subscribes this box's 6.5 GB-free KV pool and errors ~245 tokens in **identically on branch and main** (pre-existing admission behavior, not a regression). Takeaway: throughput-neutral-to-slightly-positive on this memory-bound GPU, exactly the predicted ~1–2%-at-the-noise-floor; the payoff regime (fast decode GPU / N≫128) isn't reachable on this hardware.
- **P2 (done):** mechanically migrated qwen35, deepseek-v2-lite, kimi-k2, and the sim. Compiler-verified here: engine, vllm-frontend, qwen3, deepseek-v2-lite, sim. The Triton/NCCL-gated crates (qwen35, kimi-k2) couldn't link in this env (missing toolchains); their migration is the same uniform type-swap.
- **P3 (optional, not done):** per-step batch send (shaves the N pushes, ~6µs, marginal); attack the new `exec_cpu` floor (batch `schedule_decode`/`apply_decode`).

## Next

- End-to-end throughput A/B is done (see Staging — within noise to +0.9%, growing with concurrency). Still open: the *per-phase* bubble re-measurement (instrument as in "Measurement") to confirm `apply` 254 → ~22µs directly — the low-noise mechanism proof that end-to-end tok/s can't resolve on this memory-bound GPU.
- Build/verify qwen35, kimi-k2 once their toolchains are available (Triton venv / `PEGAINFER_NCCL_ROOT`).
- Adjacent: O(N²) `active.iter().position/find` in resolve/apply → `HashMap<RequestId, usize>` index (cheap hygiene, prevents bs≳256 blowup); per-row sampling redesign for `temperature>0` (separate `exec_cpu` source).
