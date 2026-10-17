# DeepSeek-V2-Lite Device Attention

> **TL;DR:** On the NCCL backend, steady decode keeps attention, the KV cache, and every RMSNorm on rank 0's GPU, with one slot of a fixed bf16 KV pool per request, so rows at different positions share one decode step. On the #466 short HTTP contract, TPOT drops 72-74% at concurrency 1/4/8 with identical output hashes; prefill still runs on the host (#452).

Last touched: 2026-10

## Scope

The device path covers NCCL steady decode for up to eight live requests. Prefill and the host-staged backend still use host attention, and host-staged remains the correctness oracle.

Before this change, every decode layer copied the q / kv_a / kv_b projections to the host, ran RMSNorm and attention on the CPU, and copied the result back. The host batch path also needed one shared position, so rows at different positions were split into groups.

## Design

```text
after host prefill:
  take a free slot, copy the prompt's K/V into it as bf16, free the host copy

each decode step:
  upload_decode_rows: slot_ids[row], positions[row]       (once per step)
  per layer on rank 0:
    device RMSNorm
    q = gemm(q_proj), kv_a = gemm(kv_a_proj)
    compressed = kv_norm(kv_a[:512])
    kv_b = gemm(kv_b_proj, compressed)
    decode_attention: append K/V at positions[row], attend over 0..=positions[row]
  advance every slot by one token
```

`DeviceKvPool` owns per-layer K and V buffers laid out `[slot][position][head][dim]`, with K at 192 dims per head (128 nope + 64 rope) and V at 128. It stores expanded per-head K/V. The 512-wide compressed latent is only an intermediate projection. Eight slots x 4096 tokens x 27 layers is about 8.4 GiB of bf16 on rank 0. The slot count is tied to the scheduler's active-request cap.

Startup builds the pool only if current usage plus the pool stays within 90% of rank 0's memory, the same default as Qwen3's `gpu_memory_utilization`; otherwise it logs a warning and leaves NCCL decode on host attention. Rank 0 weights take about 15.9 GiB, so by static estimate 24 GiB cards take the fallback and 32 GiB cards fit. The 90% margin is a convention, not a measured peak, and only 40 GiB cards have run the device path.

`DeviceKvSlot` is an RAII handle stored in the request's `DecodeCache`. Dropping the cache returns the slot to the free list, so finish, error, and cancellation share one release path. A reused slot is not cleared: the kernel only reads positions `0..=position`, and the new owner overwrites them from prefill on.

`decode_attention_kernel` launches one 256-thread block per (head, row). A block writes only its own head's K/V in its own row's slot, so blocks need no cross-block synchronization. Slot IDs come from owned handles, and the pool checks positions before launch. The kernel also guards invalid slot or position metadata by writing NaN instead of touching the pool.

## Numerical Contract

The device path must match the host path bit for bit, because a last-bit difference can flip a greedy near-tie. The first version used a tree reduction and `rsqrtf` in RMSNorm and flipped the `Hello` case (`19 year old` vs `20 year old`). Both kernels now follow host order:

- products and sums use `__fmul_rn` / `__fadd_rn`, so the compiler cannot fuse them into FMA;
- the RMSNorm square sum and the softmax denominator are sequential in index order;
- RMSNorm uses `__fsqrt_rn` and `__fdiv_rn`;
- scores, probabilities, and stored K/V round to bf16 at the same points as the host.

## Correctness Validation

Final validation ran on the current tree on two A100 40 GB GPUs with release builds. These runs build the pool, so they do not exercise the fallback.

| Check | Result |
| --- | --- |
| DeepSeek-V2-Lite library tests | Passed |
| Ragged rows (prefill lengths 1 / 5 / 17 / 64, four steps) vs host oracle | Bitwise-equal |
| Device RMSNorm vs host, hidden (2048) and compressed KV (512 of 576) | Bitwise-equal |
| Slot reuse, exclusivity, and full-slot refusal | Passed |
| `e2e_ep2` case set: host-staged vs NCCL device | 15 rows token-exact, 15 device decode steps per case |
| `e2e_ep2` mixed serving: host-staged vs NCCL device | 7 requests with identical token hashes |

`GenerationStats::device_decode_steps` counts steps that ran on the device. On NCCL, `e2e_ep2` accepts either every decode step on the device or a complete host fallback; on host-staged it requires zero device steps.

## Performance Validation

HTTP A/B compared main (`353cebf8`) with this branch on two idle A100 40 GB GPUs, NCCL, CUDA Graph off, using the #466 short contract (about 85 prompt tokens, 64 output tokens, 32 requests, three repeats; see [benchmarking.md](benchmarking.md#retained-http-slo-profiles)). Cells are medians of the repeats.

| Concurrency | Main TPOT p50 | Branch TPOT p50 | TPOT change | Main output tok/s | Branch output tok/s | Main TTFT p50 | Branch TTFT p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 54.8 ms | 15.3 ms | -72.1% | 13.1 | 28.1 | 1413 ms | 1319 ms |
| 4 | 228.1 ms | 58.9 ms | -74.2% | 14.8 | 36.3 | 2868 ms | 3319 ms |
| 8 | 440.6 ms | 116.5 ms | -73.6% | 15.5 | 38.2 | 5674 ms | 6032 ms |

All requests completed with identical output hashes on both binaries, and the #453 reliability runner passed on both. TTFT rose 6-16% at concurrency 4/8 even though prefill is unchanged; the cause was not isolated. The mixed and long profiles were not compared because host prefill (#452) dominates them.

## Follow-Up Work

1. Store the compressed 576-wide latent instead of expanded K/V. That cuts pool memory about ninefold and lets 24 GiB cards use the device path, but needs an absorbed MLA kernel.
2. Prefill still runs on the host and uploads its cache. Device prefill attention needs a causal kernel and belongs with #452.
3. Port the serving decode step to CUDA Graph; the slot metadata buffers already have stable pointers.
