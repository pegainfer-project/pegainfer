# Server startup-to-ready time

**TL;DR**: Qwen3-4B warm (page-cache-hot) startup-to-HTTP-ready cut 3.25s → ~1.45s by (1) loading the engine on a blocking thread so the vLLM frontend's ~1.15s tokenizer/chat-template load runs concurrently, and (2) moving the ~0.4s munmap page-table teardown of the touched source mmap off the allocation hot path (since #377 it is paid once at the end of the load function). Readiness semantics unchanged: HTTP binds only after the engine bridge registers. 2026-07: the engine path has since outgrown the frontend overlap, and the once-rejected pinned-staging upload is adopted (warm ready 5.22s → 4.66s on sm_89) — see below.

Last touched: 2026-07

## Warm-start breakdown (RTX 5070 Ti, Qwen3-4B, 8GB bf16)

Baseline 3.25s was three serial phases:

| Phase | Cost | Fix |
| --- | --- | --- |
| Engine load (CUDA ctx 0.18s + H2D upload ~1.1s @ ~7GB/s pageable) | 1.3s | pageable at the time; pinned staging adopted 2026-07 — see below |
| munmap of the 8GB touched source mmap | 0.42s | dropped at the end of the load function since #377 |
| vLLM frontend: tokenizer (11MB tokenizer.json) + text/chat backends | 1.15s | runs concurrently with engine load |

After: engine path ~1.35s and frontend path ~1.25s overlap; ready ≈ max of the two + process start ≈ 1.45s.

## The munmap trap

munmap of ~8GB of touched pages costs ~0.4s of kernel page-table teardown holding the process mmap lock. Dropping the mmaps "in the background" just moves the stall to whatever allocates next — cudaMalloc (KV cache), cuBLAS init on the worker thread, or first-decode graph capture all take the same lock. We measured the 0.4s gap migrating from `kv_budget` → `KvCacheManager::new` → `RankWorker::spawn` as the drop point moved. The 2026-06 fix kept the mapping alive for the model's lifetime; #377 later changed this to an explicit drop at the end of the load function (after the `GPU model loaded` timing log), so the teardown is paid once before profiling starts instead of at an arbitrary later allocation point.

## Concurrent startup invariant (pegainfer-vllm-frontend)

`serve()` takes `impl Future<Output = Result<EngineHandle>>`. vllm-server starts immediately (loads tokenizer, then waits in `EngineCoreClient::connect` for the engine to register); a spawned task awaits the engine and runs the `LocalEngineBridge`. Invariant chain: bridge registers (reporting `max_model_len` in the ready handshake) → vllm-server builds the router → HTTP binds. So a reachable port still means the engine can serve. Request-length limits are enforced by vllm-text natively from the reported `max_model_len` (prompt-too-long error + `max_tokens` clamp), and the scheduler rejects prompt+`max_tokens` overflow via `RejectReason::ContextLength` — there is no separate HTTP-layer guard.

Consequences to keep in mind:

- The bootstrapped-transport `ready_timeout` now bounds the **whole engine load** (it used to start after load). It is 30min so multi-GPU MoE loads and cold starts fit; load *failure* cancels the server immediately via the engine task, the timeout only catches a truly hung load.
- The graceful ctrl-c handler is installed only **after** the engine load resolves (`cancel_token_on_ctrl_c`). The blocking load can't be cancelled, so during load SIGINT keeps its default kill behavior — same as before the change. (When testing this: background jobs of non-interactive shells inherit SIGINT=SIG_IGN; use a spawner that restores default dispositions.)
- LoRA mode stays sequential: the LoRA routes need the handle when the router is built.

## Pinned-staging H2D upload (rejected 2026-06, adopted 2026-07, retuned 2026-07)

The 2026-06 rejection reasoned from the numbers above: upload ran at ~7GB/s pageable, and with the engine path (~1.35s) barely above the concurrent frontend path (~1.25s), a faster upload stood to gain ≲0.1s of HTTP-ready. By 2026-07 the engine path had grown far past that floor (warm HTTP-ready ~5.2s on sm_89), so the frontend overlap no longer hides the upload: the pinned double-buffer pipeline (`WeightStager` in pegainfer-core) cuts the warm Qwen3-4B load phase ~1.26s → ~0.69s and HTTP-ready 5.22s → 4.66s — the phase saving passes through in full. (The timed load phase ends at the `GPU model loaded` log, before the mmap teardown; HTTP-ready includes everything.) Cold ready stays storage-bound (5.39s → 5.33s on local NVMe). On dual-GH200 Qwen3-14B TP2 the picture is platform-dependent: pageable copies over NVLink-C2C are already fast, so the pipeline wins there only once the strided column-shard gather lands with it (warm rank-0 load 1.19s → 0.69s vs main; the intermediate gather-only state regresses to 1.33s), validated by the TP2 golden gate with the page cache warm and cold.

### Staging geometry and the fill team

The first rollout shipped 64 MiB slots and four workers spawned per chunk. The 64 MiB event-sync rationale did not survive the later sensitivity sweep, which found the constant flat from 16 to 64 MiB; the four workers did hold up — on sm_89 that is the smallest team whose fill saturates the H2D path. A ten-arm sweep (two-GPU GH200, Qwen3-32B, TP2, 144 cores, five interleaved repetitions) moved the shipping point to 32 MiB slots filled by a resident rayon team of eight, halving pinned memory to 64 MiB across both buffers.

The geometry alone is dominated by pooling at the old geometry: spawn 8/32 MiB reaches 8272 ms to ready at 28.84 s of CPU while pool 4/64 MiB reaches 8262 ms at 24.25 s, better on both axes with no constant touched. Pooling alone is not a dead arm — against main's 8903 ms at 21.20 s it buys 641 ms for 3.05 CPU-s. The resident team is what turns the smaller chunk from a loss into a gain: spawning at eight workers, 64→32 MiB costs 7993→8272 ms to ready; pooling at eight, it buys 7651→7356 ms. CPU is bought by the worker count, not the chunk size: at fixed chunk, four workers to eight costs 12.6–14.8% more CPU on the pooled arms; at fixed workers, 64→32 MiB costs 0.3–2.2%.

**`FILL_THREADS` is per rank**, but ranks do not load concurrently: `Qwen3Executor::from_runtime_with_lora_options` finishes one `Qwen3Model` before starting the next, and each rank's loader — which owns the fill pool — is dropped at the end of its own load. One team of eight is therefore the most that ever fills, at any world size. Concurrency knees at that eight (2/4/8/12 workers → 5482/3583/2720/2701 ms of the two-rank load marker, the last for 20.7% more CPU to ready), so dividing a host budget by `world_size` only shrinks the one active team: at a fixed budget of 16 — four workers per rank at TP4, two at TP8 — eight per rank wins HTTP-ready 5/5 at both widths, by paired medians of 268 ms and 1119 ms, at the cost of 0.31 s and 0.78 s more process CPU (eight-GPU sm_120 host, Qwen3-4B, five order-rotated pairs per width). A host-level bound becomes relevant only where rank-local fill pools run concurrently. GLM5.2 does that under `--glm52-weight-staging`, submitting every rank's load before receiving any result: four fill workers per local rank, so 16 on a single-host EP4 run, and none at all on its default pageable path. On multi-host EP the host total follows the ranks local to that host, not the global EP width.

On the post-#752 base, where uploads run back to back from `finish()`, five interleaved repetitions pair to medians of -259 ms of GPU-model load and -541 ms to ready for +4.94 s of process CPU — about nine CPU-seconds per second of startup saved. That load figure is the model's own `GPU model loaded in` line, not the two-rank marker above; the two are not comparable. On a single sm_89 GPU the change is flat to ready, measured on the pre-#752 base and not repeated since: that host is transfer-bound at a 23.3 GiB/s H2D roofline against 24.6 GiB/s of fill.

Halving the buffer also halves the column-shard ceiling `prepare_cols` enforces, from 33.5 M to 16.7 M columns per shard; no supported Qwen3 checkpoint approaches it.

## Next

The warm HTTP-ready floor is now the engine's own post-load startup work (profile, warmup, graph capture — ~3.7s of the ~4.7s), no longer the frontend path; that is where the next startup-time win lives. Cold-start stays bandwidth-bound (per roadmap out of scope).
