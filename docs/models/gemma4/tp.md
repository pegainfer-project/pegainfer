# Gemma 4 tensor parallelism

TL;DR: dense Gemma 4 runs as one rank per device: a per-rank weight shard, one `all_reduce` after `o_proj` and one after `down`, and a per-rank KV bound to that rank's own pools, all driven lock-step from one scheduler thread. 12B bf16 TP2 is verified against a single-rank run; 31B bf16 serves at an 8192-token ceiling with 8 decode slots on the 48 GiB-class sm_89 pair (L20/L40S/RTX 6000 Ada), measured on two L20s: 57 ms first token, 49.4 ms per step, 143 tok/s at 8 concurrent (`benchmarks/gemma4-31b-tp2-l20.md`). The need is capacity, not the card — ~59.8 GiB of weights do not fit one device under ~60 GiB usable, whatever its model. A cold two-rank process must warm the cuBLAS kernels before the communicator exists, or the first GEMM deadlocks in the driver's module loader. Decode graphs work under TP: the sweep interleaves phases across ranks, and each rank releases its graphs before the communicators drop. Gate: `engine::lane_gates_tp::the_two_rank_engine_matches_one_rank`.

Last touched: 2026-10

## What is sharded, and what is not

The text tower's linears split the qwen3 way — output-sharded rows or input-sharded columns — but the head counts differ per layer kind, so the row ranges do too.

| tensor | shard | 12B TP2 (per rank) | 31B TP2 (per rank) |
| --- | --- | --- | --- |
| `q_proj` rows | contiguous query heads, `Q / P` | 8 × 256 | 16 × 256 sliding / 16 × 512 global |
| `k_proj` rows, sliding | `Kv / P` | 4 × 256 | 8 × 256 |
| `v_proj` rows (sliding only) | same as `k_proj` | 4 × 256 | 8 × 256 |
| `k_proj` rows, global | `G % P == 0` → shard, `P % G == 0` → replicate | 1 × 512 (replicated) | 2 × 512 (sharded) |
| `o_proj` cols | the same query-head run | 8 × 256 | 4096 sliding / 8192 global |
| `gate`, `up` rows | `intermediate / P` | 7680 | 10752 |
| `down` cols | `intermediate / P` | 7680 | 10752 |

Published per-rank counts, from each checkpoint's own `config.json` (31B: `Q` 32, sliding `Kv` 16, `G` 4, head dims 256/512, hidden 5376, intermediate 21504, 50 sliding + 10 global layers).

Replicated on every rank: the token embedding (and the tied head), all four layer norms, `q_norm` / `k_norm` (whole heads are held, so the vectors stay whole), `layer_scalar`, and both rope tables. The residual stream and every head width stay whole; only head *counts* and the MLP width shard.

## The world size a shard decision has to satisfy

`TensorParallelConfig::validate_for` refuses a launch that would silently drop or misalign heads. The three counts the kv-cache design doc states are necessary but not sufficient — they do not keep the **per-rank GQA group** integral (`Q = 8`, `Kv = 6`, `P = 2` clears them and yields group `4/3`), so that is checked as well:

```
Q  % P == 0
Kv % P == 0
G  % P == 0  ||  P % G == 0
(Q/P) % (Kv/P) == 0          # per-rank sliding group
(Q/P) % G_local == 0         # per-rank global group
intermediate % P == 0
```

A routed (MoE) checkpoint and a W4A16 one are refused under TP > 1: the first needs an expert-sharding design, the second projects whole matrices.

## How a step runs

One scheduler thread drives every rank. Each rank owns its `DeviceContext`, `GemmaServe` and `StepArena`; a request's KV is **one `GemmaKv` per rank**, each minted from that rank's pools (a `GemmaKv` is pool-bound by construction, so this is what `admit_tokens`' `belongs_to` check requires). `GemmaKv` derefs to rank 0's families, which is why the whole single-rank engine and serve code paths are unchanged.

That per-rank KV is why no host/device split was needed: each rank's entry point advances its own frontier, and because the pools have identical budgets and the admission sequence is identical, the page ids stay in step across ranks.

Per step, for every rank `r`: `activate_rank` (set the device, bind its context, make its thread-local cuBLAS handles current — creating them on first use), run that rank's segment, then restore rank 0 before the sampler, which runs on rank 0 alone. A prefill additionally drains each non-primary rank's stream, so a device fault on that rank surfaces by name instead of stalling the primary's collective forever.

The staged decode pipeline (ids written by the previous step's sampler) is disabled under TP: a non-primary rank has no sampler and no ids of its own, so every rank takes the explicit-token path.

## The collective

`all_reduce_in_place` on the live extent (`hidden_size × seq_len`) of the projection buffer, inserted in `attention_epilogue_into` after `o_proj` and after `down` — both are row-parallel sums, and both are reduced before the residual they feed is formed. One communicator per rank, built on that rank's compute stream so the reduction lands inside a captured graph once graphs are enabled.

## Cold start: warm the kernels before the communicator

Under the default `CUDA_MODULE_LOADING=LAZY` a fresh process enters the CUDA driver's module loader on the first GEMM of *each shape*. `Comm::from_devices` starts NCCL's proxy threads, which enter that same loader while the communicator comes up; a GEMM that materializes a shape for the first time after that point deadlocks — the engine thread spins at 100% CPU inside `cuLibraryGetModule` (reached through `cublasGemmEx` → `cublasLtTSTMatmul`), never issues rank 1's tower, and the request never returns. It only bites a **cold** process: any run that already did a single-rank pass in the same process (the gate, a warm box) reuses the kernels that pass loaded, which is why it can hide behind a passing gate.

The engine therefore runs one tower pass per rank at load, **before** the communicator is built, at a spread of row counts across cublasLt's kernel-selection regions (`1, 2, 3, 4, 8, …, 1024`, capped by the serving ceiling; larger prompts and every decode bucket reuse those kernels — verified by sweeping prompts of 5…6000 tokens). With no comm alive the loads happen single-threaded, and the serving path never touches the loader again. `CUDA_MODULE_LOADING=EAGER` is the equivalent sledgehammer and also works; the warm is preferred because it is per-rank and bounded by the rows the tower actually uses. The cost is a few extra tower passes folded into load — not measured on their own, but the real 31B's cold load is 30 s with them folded in (`benchmarks/gemma4-31b-tp2-l20.md`).

## Memory: the pools are per-layer arrays

Each family's pool is `layers × pages × bytes-per-page`, which is the number that decides whether a 31B configuration fits a 48 GiB card.

| | local family | global family |
| --- | --- | --- |
| bytes per token | `kv_heads × head_dim × 2 (K,V) × 2` | same |
| 31B TP2 per token | `8 × 256 × 4` = 8 KiB | `2 × 512 × 4` = 4 KiB |
| pages at ceiling 8192 / 16 slots | `15 × 17 + 128 + 1` = 384 | `16 × 128 + 1` = 2049 |
| pool bytes | 50 × 384 × 0.5 MiB ≈ 10.1 GiB | 10 × 2049 × 0.25 MiB ≈ 5.4 GiB |

31B TP2 therefore needs **29.91 GiB of weights plus ~15.4 GiB of pools ≈ 45.4 GiB per rank at the default point** — over a 48 GiB L20. The working envelopes measured on L20:

| ceiling | slots | pools | ≈ resident per rank |
| --- | ---: | ---: | ---: |
| 8192 | 8 | 8.6 GiB | 38.5 GiB |
| 4096 | 16 | 10.3 GiB | 40.2 GiB |
| 2048 | 2 | 1.4 GiB | 31.3 GiB |

The startup error names the page counts when a budget cannot be allocated, and the global pool's size is logged once it is.

The 8192 x 8 row is the one that has been served: 2.50 GiB global pool, 14.18 GiB free after weights, cold load 31.4 s, and 57 ms / 49.4 ms / 143 tok/s (TTFT / step / 8-way aggregate). The full run and method are in `benchmarks/gemma4-31b-tp2-l20.md`.

## What is refused under TP today

`PEGAINFER_ASYNC_PREFILL` (its lane stream cannot be lock-stepped across ranks), `PEGAINFER_GLOBAL_ATTN=tilelang*` (the generated kernels are compiled for the whole global family), and `PEGAINFER_PREFIX_CACHE`.

## Decode graphs under TP

`--cuda-graph=true` works under TP. The sweep interleaves **phase by phase across ranks** — for each bucket, each phase, each rank — because a `Warm`/`Launch` phase executes and enqueues its rank's all-reduce, whose peer call has to be in flight for the step to finish; `Capture` only records, and a recorded collective replays when its peer replays. One scheduler thread is enough because the phases, not whole sweeps, are the unit that has to line up.

Two environment constraints are baked into the code. The **kernels must be warm before the communicator exists** (see "Cold start"): a first-shape GEMM that materializes after `Comm::from_devices` wedges in the driver's module loader, which is what made the earlier capture attempts look like a capture defect. And **every rank must release its graphs before the communicators drop**: a captured collective bakes in NCCL kernel launches, and `ncclCommAbort` wedges while a graph that references them is still alive, so `EngineState::drop` releases them first, on each rank's own device.

Parity is the same gate, with `PEGAINFER_TP_GRAPH=1`: two ranks captured against one rank eager, `48/48` picks and worst gap `0.0000` on the 31B-geometry checkpoint. The server starts, serves and shuts down cleanly with `--tp-size=2 --cuda-graph=true`.

Captured against eager is a wash on two L20s at 31B (8192 x 8), back to back on a quiet box: first token 57.0 vs 56.5 ms, decode step 51.5 vs 49.4 ms (+4%), 8-way aggregate 146.0 vs 143.7 tok/s. The collective, not launch count, dominates the step here, so graphs buy little; the flag is on by default (as at TP1) and `--cuda-graph=false` is the marginally faster per-step choice at low concurrency.

## Verification

`engine::lane_gates_tp::the_two_rank_engine_matches_one_rank` starts a real engine twice — once with one rank, once with two — over the same 12B checkpoint, three prompts and one batch, and compares the requested top-8 logprobs.

It holds the *distributions*, not the greedy tokens: a two-rank reduction sums the same products in a different order and NCCL writes bf16 back, so the logits differ in their last bits and a near-tie can flip the pick. Measured at 12B on two L20s:

| claim | result |
| --- | --- |
| one-rank run twice (control) | bit-identical, so the comparison is not measuring harness noise |
| steps keeping the one-rank pick | 46 / 48 |
| worst picked-token logprob gap | 0.379 |
| a differing pick | always a genuine near-tie: each pick inside the other run's top-8 |

The **shard branch** (`G % P == 0`, which the published 12B never takes — its single global KV head is replicated) is gated with a synthetic checkpoint carrying the real 31B shapes (`Q` 32, `G` 4, head dims 256/512, hidden 5376, intermediate 21504) cut down to six layers, so the whole run is ~8 GiB and takes seconds on any pair. It is **bit-identical**: 48/48 picks, worst picked-token gap `0.0000`. Both numbers above are with `NCCL_PROTO=LL128` (see the notes below).

The gate serves its first prompt on its own and only then the rest as one batch, so the **solo** admission path (`step` + `prefill_extra_ranks`) — where the cold-start hazard above first surfaced, and the only path a lone short request takes — is compared on every run; `PEGAINFER_TP_PROMPTS` / `PEGAINFER_TP_PROMPT_TOKENS` still widen the set. There is **no automatic gate for the production shard path**: the numbers above are the 12B checkpoint (replicate branch) and the six-layer synthetic (shard branch), and the 60-layer real checkpoint is served by hand.

The single-rank suite is unchanged by the TP path (`cargo test --release -p pegainfer-gemma4 --features gemma4 --lib`).

The **real 31B checkpoint**, which no single card holds, is served end to end on two L20s; its load, envelope and serving numbers are in `benchmarks/gemma4-31b-tp2-l20.md`.

## Known bounds

- **A rank-0 failure mid-step has no cross-rank circuit breaker.** The extra ranks are now driven *unconditionally* after rank 0 has been started (see "How a step runs"), so a rank-0 failure cannot skip the peer and shift the collective pairing. What is not handled is a rank-0 failure that lands **before it issued any collective**: the peer then waits on a call that never comes and the drain waits with it, so the scheduler wedges instead of failing that request. The triggers are host-side (`prepare_single`'s checks, a per-request scratch allocation), not client-reachable, and no test injects one.
- **Page-id agreement is inspected, not enforced.** Every rank's pools must stay identical in free-page order — that is the whole basis for "identical page ids". A debug build compares free-page counts after every admission, and every prefill compares the ranks' frontiers; a release build only does the frontiers, and nothing repairs a divergence.
- **The vocabulary projection is replicated.** `embed_tokens` and the tied head are `Whole` on each rank (31B: 262144 × 5376 bf16, ~2.8 GiB per rank), and every extra rank computes a whole batch of `vocab × rows` logits per decode step that is then discarded. Dropping it needs a head-free tower for the extra ranks — in graph mode that means a second per-bucket capture, because the head sits inside `decode_gpu_body`. Not implemented, not measured.
- **`NCCL_PROTO=LL128` is a requirement the code reports rather than enforces.** Startup logs the NCCL version and the effective protocol and warns when it is not LL128; it does not refuse. On this container's stock NCCL the default protocol corrupts the all-reduce buffer (see the operational notes).

## Operational notes

- **A pair, not a card.** Two ranks are two devices; the gate runner claims both (`require_tp2`, `PEGAINFER_GATE_GPU=a,b`) and exports them as `CUDA_VISIBLE_DEVICES=a,b`, so the ranks inside the process are devices 0 and 1.
- **Set `NCCL_PROTO=LL128` in this container.** With the pod's stock NCCL 2.18.3 (2023, CUDA 12.2 vintage) on CUDA 12.9 and sm_89, the default `LL` protocol — and `SIMPLE` — make the all-reduce kernel write far outside its buffer (`compute-sanitizer` names `ncclKernel_AllReduce_RING_LL_Sum`, ~95 GB past the allocation). A non-primary rank faults asynchronously, so the visible symptom is the primary's collective stalling forever rather than an error. `LL128` is sound, and the repo's own L20 precedent ran NCCL 2.32.3. Prefer a newer NCCL over the variable where the environment allows it.
- **The checkpoint is read once per rank.** 62 GB does not stay in a 32 GB page cache, so a 31B load is ~3.5 min per rank from the PVC and ~15 s from node-local disk — copy it local for iteration.
- **In a container, disable the fabrics NCCL cannot reach** (`NCCL_IB_DISABLE=1`, and `NCCL_P2P_DISABLE=1` if P2P is unavailable). The log otherwise shows `Unable to open device mlx5_*` warnings.
- **Check the pair is healthy.** On a shared box, an uncorrectable DRAM fault shows up as a pair-dependent illegal memory access, which is not a TP defect.
