# Scheduler

> **TL;DR:** Single dedicated thread owns all GPU resources. Continuous batching with FCFS prefill-priority, paged KV cache, bucket CUDA Graphs for batch decode, and a unified forward pass when prefill and decode coexist. On Qwen3-4B (varied-length Poisson QPS=2, RTX 5070 Ti) within 2% of vLLM throughput while winning TTFT (−16%), TPOT (−3%), and latency stability across the board. Remaining gap is ITL p99 tail from prefill stalls.
>
> **Last touched:** 2026-05.

## Why this shape

Single-request decode is bandwidth-bound — ~91% of decode time is GEMV/MLP at 82–87% DRAM utilization. No single-operator optimization moves the needle. The only throughput lever is amortizing the 7.67GB weight read across N requests: bs=1 → ~85 tok/s, bs=8 → ~680 tok/s, bs=32 → ~2700 tok/s.

## Runtime architecture

```
HTTP handler (tokio async)        Scheduler thread (1 fixed thread, owns GPU)
─────────────────────────         ────────────────────────────────────────────
tokenizer.encode()                add → prefill_batch / unified_step
submit_tx.send(req) ─────────→
                                  decode loop: batch_decode (bucket CUDA Graph)
                                  sample → token per request
token_rx.recv()  ←──────────────  token_tx.send(token_id)
tokenizer.decode() + SSE
```

The scheduler thread owns the model, `BatchDecodeBuffers`, and `KvPool` exclusively. No `Mutex` — single-threaded GPU ownership. HTTP handlers and scheduler talk over channels only. `token_tx` is now a `TokenSink` over one shared request-tagged channel (see `output-dispatch.md`); `token_tx.send` returns `Err` when the request's cancel flag is flipped (abort) or the engine shuts down, which is the cancellation signal the scheduler retires on.

## Paged KV cache

Page-first single buffer: each physical page contains all layers' K/V interleaved for `page_size` tokens. One `cudaMalloc`; kernel addressing is `base + page_id × page_stride + layer × layer_kv_offset`.

| Type | Role |
|---|---|
| `PagePool` | Generic fixed-page allocator with RAII permits |
| `KvPool` | GPU-backed allocator over `PagePool`, owns the backing buffer |
| `KvState` | Per-request state: pages held, seq_len, capacity growth |
| `KvDesc` | Kernel-facing metadata: base ptr, strides, page indices, last-page len |

"Paged" never leaks to callers. Pointer arithmetic is hidden inside `KvDesc` construction. `KvState` parallels `RecurrentState` so the shared `GenerationState` shape stays uniform.

The bridge to FlashInfer's `paged_kv_t` (which expects separate `k_data`/`v_data` with custom strides) is purely stride math — `k_data = base + L × layer_stride`, `v_data = base + L × layer_stride + kv_block_len`. No transpose, no copy.

## Scheduling policy

- **FCFS prefill-priority.** Pending prefill always preempts continued decode at step boundaries.
- **Bucket CUDA Graphs.** Batch buckets `[1, 2, 4, 8, 16, 32, 64]`, padded up at runtime. One captured graph per bucket, capture-on-first-use, replay afterward.
- **Padding page.** `KvPool` reserves one page at construction. Padding batch slots point at it with `seq_len=1, last_page_len=1` — KV append writes harmless garbage, attention output is discarded.
- **Unified forward.** When both pending prefill and active decode exist, one forward pass handles all tokens. Decode rows ride free inside the compute-bound prefill GEMM (<2% FLOPs added). Pure decode (the common case) still uses CUDA Graph.
- **Admission.** Reject when `KvPool` full. No queue behind memory pressure.

## Performance (RTX 5070 Ti, Qwen3-4B, n=500, varied-length Poisson)

### QPS=2 head-to-head vs vLLM 0.18.x

| Metric | pegainfer | vLLM | Delta |
|---|---|---|---|
| TTFT median | 301ms | 359ms | **−16%** |
| TTFT p99 | 951ms | 1245ms | **−24%** |
| TPOT median | 25.2ms | 26.1ms | **−3.4%** |
| TPOT p99 | 50.0ms | 56.2ms | **−11%** |
| output tok/s | 251 | 256 | −1.9% |
| failed | 9 | **0** | — |
| ITL p99 | 291ms | **211ms** | +38% |

pegainfer wins 17/20 metrics. Std lower across the board (TTFT −20%, TPOT −22%). vLLM wins on robustness (zero failed) and ITL tail.

### Decode TPOT scaling (in=1, out=128)

| Concurrency | pegainfer | vLLM |
|---|---|---|
| 1 | 10.75ms | 11.31ms |
| 4 | 10.96ms | 11.52ms |
| 8 | 11.05ms | 11.42ms |

Beats vLLM at all concurrencies post-fusion. TPOT only grows 3% from c=1→c=8 (scheduling overhead, not GPU saturation).

## Key optimizations landed

- **Fused projections** (zero extra VRAM, weights concatenated at load time): `qkv_proj = [Q;K;V]` and `gate_up_proj = [gate;up]`. The K/V GEMMs were small (M=512, N=8, K=3584) with ~20% SM use; merging into one big GEMM and demultiplexing in a tiny kernel fills SMs. Drops TPOT from 11.75 → 11.05ms at c=8.
- **Batched argmax.** Per-step serial `(0..bs).map(extract_vec → argmax → sync → D2H)` replaced by one kernel over contiguous `[bs, vocab]` logits. `cuStreamSynchronize` calls dropped 8× per step → 1× per step.
- **Persistent CUDA Graph across requests.** Don't destroy `CudaGraphState` on `reset()` — graph topology is identical across requests; only metadata (positions, page indices) changes, updated via `memcpy_htod` before launch. Saves ~2.4ms per request.
- **Dynamic KV pool sizing.** 85% of free VRAM at startup; eliminates the old fixed `num_pages=128` that OOM'd beyond 2048 tokens.

## FlashInfer integration gotchas

- **`o_indptr` and `kv_chunk_size_ptr` cannot be null** even when `partition_kv=false`. The prefill kernel dereferences them unconditionally. Set `o_indptr = q_indptr` and use a 1-element scratch for `kv_chunk_size_ptr`.
- **`request_indices` / `kv_tile_indices` also unconditional** in non-partition decode — pass trivial `[0]` arrays.
- **bs=1 occupancy hole.** Non-partition decode grid is `(batch_size, num_kv_heads)` = `(1, 8)` ⇒ 8 blocks on 48 SMs, ~17% SM utilization. Costs ~800μs at ctx=1024. Goes away naturally at bs≥6; enabling `partition_kv=true` would fix it but needs `tmp_v`/`tmp_s` buffers, a merge kernel, and `block_valid_mask` for CUDA Graph stability. Deferred.

## Lesson worth keeping

**One variable at a time when migrating attention paths.** The first paged-prefill attempt changed 8 things simultaneously (QK norm fusion, RoPE source, KV layout, attention kernel, position encoding mode on both prefill and decode). Output was coherent but diverged on all 6 test prompts ("What is 2+2?" → "5...") with no way to bisect. Splitting into separate PRs — Triton FA2 → FlashInfer first, then contiguous→paged KV second — let each step verify greedy parity cleanly.

Related: FlashInfer's f32 fused RoPE rounds differently from a precomputed bf16 cos/sin cache. Greedy decoding amplifies the divergence across 36 layers — same algorithm, different output.

## Known issues / open work

- **ITL p99 tail (291 vs vLLM 211ms).** Large prefills block in-flight decode. Chunked prefill would fix it. Low priority — varied-length workloads break the waves naturally, and fixed-length ITL p99 already beats vLLM.
- **9 failures at QPS=2.** Needs root-cause — likely KV pressure or empty-prompt rejection from the random dataset.
- **Worker-panic wedge at high concurrency (RTX 5070 Ti, verified 2026-06).** Under sustained high load (c≈128–192, mixed prefill+decode), the GPU worker thread `qwen3-tp-rank-0` panics in a cudarc copy: `assertion failed: dst.len() >= src.len()` (`cudarc .../driver/safe/core.rs:1607`) — a pre-allocated buffer is too small for some batch/shape the step produces. Two distinct faults compound: (1) the undersized buffer (proximate crash); (2) **blast radius + no recovery** — `fail_touched_requests` clears the *entire* active batch on any step `Err`, the worker thread is never restarted, so every subsequent step returns "worker step channel closed" and **all** in-flight and future requests fail. `/health` still returns 200 (separate layer), so it reads as a silent permanent wedge until process restart. Not KV admission: a 160-request single burst defers cleanly with 0 errors, so full-lifetime admission works. Independent of the output-dispatch change (the panic is in the executor copy path; `main` fails identically). Next: `RUST_BACKTRACE=1` to capture the call site — prime suspects are the prefill/unified or batched step-tail logits/sampling readback (not the decode token H2D, which is sized to the 256 bucket).
- **Batched Qwen sampling regression guard.** #284 removes the Qwen per-row sampling path: greedy rows use indexed batched argmax, and non-greedy rows compact into one FlashInfer batched sampling call per step. Keep the release HF/nsys/TPOT gate when sampling params or decode batching change.
- **Qwen3.5 partial paged migration.** Decode is fully paged via the scheduler. Prefill still scatters from contiguous HND staging into paged KV before attention. Migration mirrors Qwen3's step 2 with HD256 + partial RoPE (rotary_dim=64 of head_dim=256) wrinkles.

## Next

- Qwen3.5 prefill fully paged (eliminate contiguous staging)
- Chunked prefill — only if ITL p99 becomes a hard requirement
- Preemption / priority queuing — deferred, no current pain
- **Output-dispatch redesign — landed 2026-06** (see `output-dispatch.md`): the per-request `token_tx` fan-out (N channels + N tasks, N wakeups/step ≈ 3µs×batch GPU bubble) is collapsed to one request-tagged channel + one demux loop; `token_tx` is a `TokenSink` drop-in and cancellation rides an `Arc<AtomicBool>` flag (no separate cancel channel). Remaining: re-measure the bubble on the migrated path; small win on this GPU, material on fast decode GPUs (H20/H100) or N≫128.
