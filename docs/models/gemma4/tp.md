# Gemma 4 tensor parallelism

TL;DR: dense Gemma 4 runs as one rank per device: a per-rank weight shard, one `all_reduce` after `o_proj` and one after `down`, and a per-rank KV bound to that rank's own pools, all driven lock-step from one scheduler thread. 12B bf16 TP2 is verified against a single-rank run; 31B bf16 serves on the 48 GiB-class sm_89 pair (L20/L40S/RTX 6000 Ada) with 8 decode slots and a `PEGAINFER_MAX_CONTEXT` ceiling, measured on two L20s: 57 ms first token, 49.4 ms per step, 143 tok/s at 8 concurrent (`benchmarks/gemma4-31b-tp2-l20.md`). The need is capacity, not the card — ~59.8 GiB of weights do not fit one device under ~60 GiB usable, whatever its model. A cold two-rank process must warm the cuBLAS kernels before the communicator exists, or the first GEMM deadlocks in the driver's module loader. Decode graphs work under TP: the sweep interleaves phases across ranks, and each rank releases its graphs before the communicators drop. Gate: `engine::lane_gates_tp::the_two_rank_engine_matches_one_rank`.

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
P  > 0
rank < P
Q  % P == 0
Kv % P == 0
G  % P == 0  ||  P % G == 0
(Q/P) % (Kv/P) == 0          # per-rank sliding group
(Q/P) % G_local == 0         # per-rank global group, sharding branch only
intermediate % P == 0
```

The global check carries a branch condition because only the **sharding** branch can split a group: `G % P == 0` hands a rank `G / P` KV heads, so `(Q/P) / (G/P)` has to be integral. The replicate branch (`P % G == 0`) gives a rank one whole KV head, and `P % G == 0` already keeps its `Q / P` query heads inside that head's group — the `P / G` ranks holding that head split its `Q / G` query heads evenly, so `Q / P` is `(Q / G) / (P / G)`, not a multiple of `Q / G` — so there is nothing left to police, and the check would only compare against a group of one.

A routed (MoE) checkpoint and a W4A16 one are refused under TP > 1: the first needs an expert-sharding design, the second projects whole matrices.

## How a step runs

One scheduler thread owns every rank, but a step drives them **concurrently**: rank 0's segment runs on the scheduler thread and each extra rank's on its own scoped thread, then all are joined (`EngineState::drive_ranks`). One thread per rank is what keeps a host-blocking operation inside a rank's own segment from stranding the others — see "Known bounds" for the stall it fixes. Each rank owns its `DeviceContext`, `GemmaServe` and `StepArena`; a request's KV is **one `GemmaKv` per rank**, each minted from that rank's pools (a `GemmaKv` is pool-bound by construction, so this is what `admit_tokens`' `belongs_to` check requires). `GemmaKv` derefs to rank 0's families, which is why the whole single-rank engine and serve code paths are unchanged.

That per-rank KV is why no host/device split was needed: each rank's entry point advances its own frontier, and because the pools have identical budgets and the admission sequence is identical, the page ids stay in step across ranks.

Per step, each rank's segment calls `activate_rank` (set the device, bind its context, make its thread-local cuBLAS handles current — creating them on first use), runs that rank's segment, and drains it: the extras on their own threads, rank 0 on the scheduler thread before the sampler, which runs on rank 0 alone. The drain is what turns a device fault on a non-primary rank into a named error instead of the primary's collective stalling forever.

Why the extras get threads is the invariant the single-threaded driver could not hold: **a host blocking point must come after every rank's launch.** `all_reduce_rows` enqueues and returns, but a prefill's tower reaches host-side waits of its own — a lazy cuBLAS/cuBLASLt library load (`cuLibraryGetModule`, reached through `cublasGemmEx`) inside a projection GEMM, and rank 0's per-row readback — so with one thread driving every rank in turn, rank 0 could block on a collective whose peer call had not been issued yet and hang. Running each extra rank's segment on its own thread, and joining only after rank 0's has returned, makes that ordering impossible rather than something each call site has to arrange; it is also what lifted the old prompt ceiling (see "Known bounds").

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
| pool bytes | 50 × 384 × 0.5 MiB = 9600 MiB = 9.38 GiB | 10 × 2049 × 0.25 MiB = 5122.5 MiB = 5.00 GiB |

Every figure here and below is binary (1 GiB = 1024 MiB); the MiB columns are what the start-up log prints, so the two are directly comparable.

31B TP2 therefore needs **29.91 GiB of weights plus 14.38 GiB of pools = 44.29 GiB per rank at the default point**, before the ~0.90 GiB of CUDA context, step arena, cuBLAS workspace and NCCL buffers that the same run reports as the gap between 29.91 GiB of weights and the 14.18 GiB it leaves free on a 44.99 GiB L20. That is **≈45.2 GiB against 44.99 GiB usable**, so the default point does not quite fit — by ~0.2 GiB, not by a wide margin, which is why the reduced envelopes below are the ones that were served. The working envelopes measured on L20:

| ceiling | slots | pools | ≈ resident per rank |
| --- | ---: | ---: | ---: |
| 8192 | 8 | 8.6 GiB | 38.5 GiB |
| 4096 | 16 | 10.3 GiB | 40.2 GiB |
| 2048 | 2 | 1.4 GiB | 31.3 GiB |

The startup error names the page counts when a budget cannot be allocated, and the global pool's size is logged once it is.

The 8192 x 8 row is the one that has been served: 2.50 GiB global pool, 14.18 GiB free after weights, cold load 31.4 s, and 57 ms / 49.4 ms / 143 tok/s (TTFT / step / 8-way aggregate). The full run and method are in `benchmarks/gemma4-31b-tp2-l20.md`.

## What is refused under TP today

`PEGAINFER_ASYNC_PREFILL` (its lane stream cannot be lock-stepped across ranks), `PEGAINFER_GLOBAL_ATTN=tilelang*` (the generated kernels are compiled for the whole global family), `PEGAINFER_PREFIX_CACHE`, and `PEGAINFER_MIX_CHUNK_TOKENS` (a chunked walk's rounds gather across prompts, which the TP gates do not cover). The prompt ceiling that used to be refused here (`PEGAINFER_TP_MAX_PROMPT`) is gone with the hazard it guarded — see "Known bounds".

## Decode graphs under TP

`--cuda-graph=true` works under TP. The sweep interleaves **phase by phase across ranks** — for each bucket, each phase, each rank — because a `Warm`/`Launch` phase executes and enqueues its rank's all-reduce, whose peer call has to be in flight for the step to finish; `Capture` only records, and a recorded collective replays when its peer replays. One scheduler thread is enough because the phases, not whole sweeps, are the unit that has to line up.

Two environment constraints are baked into the code. The **kernels must be warm before the communicator exists** (see "Cold start"): a first-shape GEMM that materializes after `Comm::from_devices` wedges in the driver's module loader, which is what made the earlier capture attempts look like a capture defect. And **every rank must release its graphs before the communicators drop**: a captured collective bakes in NCCL kernel launches, and `ncclCommAbort` wedges while a graph that references them is still alive, so `EngineState::drop` releases them first, on each rank's own device.

Parity is the same gate, with `PEGAINFER_TP_GRAPH=1`: two ranks captured against one rank eager, `48/48` picks and worst gap `0.0000` on the 31B-geometry checkpoint. The server starts, serves and shuts down cleanly with `--tp-size=2 --cuda-graph=true`.

Captured against eager is a wash on two L20s at 31B (8192 x 8), back to back on a quiet box: first token 57.0 vs 56.5 ms, decode step 51.5 vs 49.4 ms (+4%), 8-way aggregate 146.0 vs 143.7 tok/s. The collective, not launch count, dominates the step here, so graphs buy little; the flag is on by default (as at TP1) and `--cuda-graph=false` is the marginally faster per-step choice at low concurrency.

## Verification

`engine::lane_gates_tp::the_two_rank_engine_matches_one_rank` starts a real engine twice — once with one rank, once with two — over the same 12B checkpoint, three prompts and one batch, and compares the requested top-8 logprobs.

It holds the *distributions*, not the greedy tokens: a two-rank reduction sums the same products in a different order and NCCL writes bf16 back, so the logits differ in their last bits and a near-tie can flip the pick. The gate prints the build and the device pair it ran on (`tp2: NCCL 2.18.3, devices 1,2`), because a reading is only comparable with the environment it came from — but nothing here claims *why* two readings from different environments differ. The line comes from this gate's own readings, one procedure, one head (`45eb4a68`, four runs of the 12B on two L20s, `NCCL_PROTO=LL128`):

| claim | result |
| --- | --- |
| one-rank run twice (control) | bit-identical, **asserted** token for token and logprob for logprob, so the comparison is not measuring harness noise |
| steps keeping the one-rank pick | 22 / 24, the same in all four runs |
| worst picked-token logprob gap | 0.5738 (top 0.2362), the same to four decimals in all four runs |
| a differing pick | always a genuine near-tie: each pick inside the other run's top-8, and both picks' own gaps folded into the bound above before the comparison stops |
| **one `all_reduce` skipped** — the fault the line is for | the gate fails: at request 0 step 0 the one-rank pick leaves the two-rank top-8 (the near-tie rule), and with that rule relaxed the top-token gap reads 2.1895 |

The fault is the `o_proj` reduction commented out in `layer.rs`: both ranks skip it, so the run still completes, and the single-rank control is untouched (at world size 1 the call is a no-op). `worst_pick` reads `0.0000` on that run — the two top-8 lists share no token, so the picked-token bound has nothing left to measure, which is the case the near-tie rule exists to catch — so the rule and the top-token bound are what catch a fault this size, and `LOGBROB_LINE` sits between the two readings instead of under the noise. `LOGBROB_LINE` is `1.0` — it was `0.5`, which sits below this floor and so fails a correct run; 1.0 is ~1.7x over the floor and ~2.2x under the fault, and it is the value `DRIFT_LINE` below already uses. The repo's one precedent for calibrating a quantity of this kind is `serve_oracle`'s `neutral_scale` (two algorithms over one context, 0.31..5.75 observed, line 12.0), i.e. roughly 2x the worst observed drift. What the line is for is a structural error — a wrong shard, a missing reduction, a rank out of step — which moves a logprob by many nats and flips most picks, not 2 of 24; reduction noise is policed by the near-tie rule beside it.

The **shard branch** (`G % P == 0`, which the published 12B never takes — its single global KV head is replicated) is gated with a synthetic checkpoint carrying the real 31B shapes (`Q` 32, `G` 4, head dims 256/512, hidden 5376, intermediate 21504) cut down to six layers, so the whole run is ~8 GiB and takes seconds on any pair. It **measured bit-identical**: 48/48 picks, worst picked-token gap `0.0000`. That is a measurement, not an assertion — the two-rank comparison asserts against `LOGBROB_LINE`, the same line for both branches, so a future run that drifts within the line passes while this sentence's `0.0000` no longer holds.

The gate serves its first prompt on its own and only then the rest as one batch, so the **solo** admission path (`step` + `prefill_extra_ranks`) — where the cold-start hazard above first surfaced, and the only path a lone short request takes — is compared on every run; `PEGAINFER_TP_PROMPTS` / `PEGAINFER_TP_PROMPT_TOKENS` still widen the set. Which branch the gate covers is the checkpoint's: point `PEGAINFER_TEST_MODEL_PATH` at the 12B for the replicate branch, at a 31B-geometry checkpoint for the shard one. **A full-depth 31B cannot be gated against a *single-rank* baseline here**, not for want of a fixture but because the gate needs a single-rank control and 57 GiB of weights do not fit one card — the six-layer synthetic is exactly that shape at a depth that fits. So the numbers above are the 12B (replicate) and the synthetic (shard); the 60-layer real checkpoint is gated instead against its own Hugging Face dump (below) and served end to end on two L20s.

The single-rank suite is unchanged by the TP path (`cargo test --release -p pegainfer-gemma4 --features gemma4 --lib`), and now also carries the prefill-fault gate named above, which runs on one GPU.

The **real 31B checkpoint**, which no single card holds, is served end to end on two L20s; its load, envelope and serving numbers are in `benchmarks/gemma4-31b-tp2-l20.md`.

### The real 31B against its own Hugging Face dump

`engine::lane_gates_tp::the_two_rank_engine_matches_the_hf_reference` compares the two-rank engine directly against a Hugging Face reference for the **60-layer 31B** — the checkpoint whose single-rank baseline one card cannot hold. The fixture is that checkpoint's own dump (`tools/accuracy/dump_gemma4_hf_golden.py`, pinned by file digests), compared over the teacher-forced prompt rows' top-64 logprobs. Measured on two L20s, nine-token prompt:

| claim | result |
| --- | --- |
| reference rows keeping the engine's pick | 7 / 8 |
| worst shared-token logprob gap | 0.6753 |
| the one differing row | a near-tie: the engine's pick sits inside the reference's top-64 |

So the real geometry (hidden 5376, `Q` 32 / `Kv` 16 / `G` 4, head dims 256/512, the 50 sliding + 10 global split) matches the reference over all 60 layers — with the same `NCCL_PROTO=LL128` requirement. The tolerance here is **not** the synthetic gate's: that one is bit-identical (`0.0000` against `LOGBROB_LINE`), this one is `0.6753` against a `1.0` line. `DRIFT_LINE` is a first cut — calibrating it from this comparison on the 12B at one rank is a follow-up.

## Known bounds

- **A rank-0 step failure stops the engine — but only with extra ranks.** Rank 0's tower runs on the scheduler thread while each extra rank's runs on its own (see "How a step runs"): on failure rank 0's graphs are released and its communicator detached *before* the join, so a peer that is already waiting on a call which will never come is unblocked instead of hanging, and every later step refuses — so the engine cannot serve a comm-less reduction that would return partial sums. A failure on an extra rank is fatal too. At **world size 1** there is no peer to unblock and no frontier to keep in step, so the same failure stays what it has always been: that one request fails and the driver keeps serving (`drive_ranks` marks the engine broken only when `more` is non-empty). `engine::lane_gates_logprobs::a_failed_prefill_costs_that_request_not_the_engine` pins the single-rank half on both prefill paths, using a token id outside the embedding — `prepare_single`'s `validate_tokens` refuses it before the tower allocates anything, so the fault needs no injection hook. **The multi-rank half is still untested**: no gate injects a rank-0 or extra-rank failure at TP2, so the abort itself — destroying every rank's captured graph execs and dropping its communicator, with no stream drained first — is argued from the teardown order rather than exercised.
- **The stall the prompt ceiling guarded was not a length bug — the ceiling is gone with it.** Rank 0's segment reaches a host-side wait inside the tower: captured at TP2 as a **lazy cuBLAS/cuBLASLt library load** (`cuLibraryGetModule`, reached through `cublasGemmEx`) in a projection GEMM, which is a *first touch* of a shape rather than a function of the prompt's length — that capture's prompt was **5 tokens**, with rank 0's device 100 % busy and rank 1's idle while the one driving thread waited on a reduction whose peer call it had not issued. Driven **concurrently** (`EngineState::drive_ranks`: rank 0 on the scheduler thread, one scoped thread per extra rank, then join), a host blocking point in one rank's segment cannot stop another rank's launches, so the class is removed rather than bounded and `PEGAINFER_TP_MAX_PROMPT` — with its refusal and its start-up warning — is deleted rather than left as dead scaffolding. Measured on the single-threaded driver with the ceiling lifted: a scored prefill of 50/64/100/128/300 tokens and a 401-token server prefill all complete (4-bit 60-layer geometry; the bf16 31B is not re-measured), so the ceiling was refusing prompts that serve. On this build the three TP gates pass (one-rank comparison `48/48` at gap `0.0000`, the scored probe, the HF reference); the cost is one thread spawn per rank per step.
- **Page-id agreement is inspected, not enforced.** Every rank's pools must stay identical in free-page order — that is the whole basis for "identical page ids". A debug build compares free-page counts after every admission, and every prefill compares the ranks' frontiers; a release build only does the frontiers, and nothing repairs a divergence. Three of the four admission sites self-heal, because the request's `GemmaKv` is dropped and both ranks' pages come back with it; the fourth (`ready_rows_pinned`, which keeps the row in the batch) cannot, so an extra rank that refuses there stops the engine instead of carrying the divergence into a step.
- **The chunked walk's TP code has no runtime.** `PEGAINFER_MIX_CHUNK_TOKENS` is refused under TP, so the walker's extra-rank admission and its `tp_broken` check are unreachable with a non-empty `more` — kept because the concurrent-driver follow-up re-enables the knob, untested until then.
- **The mixed step has no TP gate.** The ceiling was also what kept a TP mixed step's prompt rows inside it (`mixed_fits`, and the gather's `gather_budget`), and both went with it: any prompt length may now ride a live decode batch at two ranks (`PEGAINFER_MIX_GATHER_ROWS` still bounds the gathered rows, and the step rides the same shared segment at every rank). `serve::oracle::mixed_step_matches_serial` is single-rank, so that region is untested at two ranks rather than known-good.
- **The vocabulary projection is replicated.** `embed_tokens` and the tied head are `Whole` on each rank (31B: 262144 × 5376 bf16, ~2.8 GiB per rank), and every extra rank computes a whole batch of `vocab × rows` logits per decode step that is then discarded. Dropping it needs a head-free tower for the extra ranks — in graph mode that means a second per-bucket capture, because the head sits inside `decode_gpu_body`. Not implemented, not measured.
- **The 1024-token reference comparison is not gated yet — its drift is the known chaotic band.** With the concurrent driver a 1024-token (window-width) prompt no longer stalls, but compared against the HF dump its worst shared-token logprob gap is **8.37** and four rows have a pick outside the other's top-64. That magnitude is the shape-dependence the repo already measures: `serve_oracle`'s `neutral_scale` scores the same context two ways and records **0.31..5.75 raw logits**, with its line at **12.0** because the quantity is chaotic and a tighter one only flakes. A prefix-consistency check confirms it is that and not a page/position error: the same 512-token prefix scored in a 1024-token run differs by 4.05 on the 31B and 1.69 on the four-layer synthetic, from row 0 on. So the nine-token case is gated (7/8 rows, worst gap 0.6753) and the 1024-token case needs the tolerance-plus-top-1-share discipline of `serve_oracle` — its stored per-case tolerance and `top1_share` floor — rather than a strict top-k containment, which cannot hold at this depth; that calibration is the follow-up.
- **`NCCL_PROTO=LL128` is a requirement the code reports rather than enforces.** Startup logs the NCCL version and the effective protocol and warns when it is not LL128; it does not refuse. On this container's stock NCCL the default protocol corrupts the all-reduce buffer (see the operational notes).

## Operational notes

- **A pair, not a card.** Two ranks are two devices; the gate runner claims both (`require_devices 2`, `PEGAINFER_GATE_GPU=a,b`) and exports them as `CUDA_VISIBLE_DEVICES=a,b`, so the ranks inside the process are devices 0 and 1.
- **Set `NCCL_PROTO=LL128` in this container.** With the pod's stock NCCL 2.18.3 (2023, CUDA 12.2 vintage) on CUDA 12.9 and sm_89, the default `LL` protocol — and `SIMPLE` — make the all-reduce kernel write far outside its buffer (`compute-sanitizer` names `ncclKernel_AllReduce_RING_LL_Sum`, ~95 GB past the allocation). A non-primary rank faults asynchronously, so the visible symptom is the primary's collective stalling forever rather than an error. `LL128` is sound, and the repo's own L20 precedent ran NCCL 2.32.3. Prefer a newer NCCL over the variable where the environment allows it.
- **The checkpoint is read once per rank.** 62 GB does not stay in a 32 GB page cache, so a 31B load is ~3.5 min per rank from the PVC and ~15 s from node-local disk — copy it local for iteration.
- **In a container, disable the fabrics NCCL cannot reach** (`NCCL_IB_DISABLE=1`, and `NCCL_P2P_DISABLE=1` if P2P is unavailable). The log otherwise shows `Unable to open device mlx5_*` warnings.
- **Check the pair is healthy.** On a shared box, an uncorrectable DRAM fault shows up as a pair-dependent illegal memory access, which is not a TP defect.
