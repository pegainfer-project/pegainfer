# Qwen3-4B serving perf tuning record (RTX 5090)

**TL;DR**: Tuning history for the QPS sweep numbers in [serving-performance.md](serving-performance.md). The mid-band gap closed with unified-step attention fusion: decode rows enter the prefill plan as qo_len=1 entries, one varlen attention call per layer, and the dispatch honors the plan's cta_tile_q (the kernel silently re-deriving its own tile size cost ~3ms/step). Earlier fixes: batched step tail (#345), chunked prefill (default now 1024), cuBLAS ≥ 13 (12.9 has a 50–100% GEMM cliff at N=1025 — build with `CUDA_HOME=/usr/local/cuda-13.x`), cublasLt per-shape algo tuning (which also re-enabled buckets 8/16), split-KV decode attention ≤bs32, two-stage argmax. The per-shape tuning result can also be carried across boots by an opt-in file store (`PEGAINFER_GEMM_LT_CACHE`).

Last touched: 2026-09

## Setup

- RTX 5090, `/data/Qwen3-4B`, dev checkout `~/develop/xingming/pegainfer` on the 5090 box.
- Workload: `vllm bench serve`, random dataset, in=1024 / out=128, Poisson arrivals, seed 42, 60s per QPS point (`tools/bench/qps_sweep.sh`).
- Reference: vLLM 0.22.1, `--max-model-len 8192 --no-enable-prefix-caching`, defaults otherwise (max_num_seqs=256, max_num_batched_tokens=2048 chunked prefill, decode-priority). Verified from v0.22.1 source: vLLM does **not** lock batch size at 128; observed ~129 concurrency is KV-capacity-bound.
- Results: 5090 `/data/xingming/bench/20260611-tune/{vllm-ref,oi-chunk512,oi-chunk1024,oi-chunk2048,oi-final,oi-default2048}/`.

## Fixed: cuBLAS 12.9 GEMM cliff at N=1025 — deploy with CUDA ≥ 13

The unified (prefill+decode) step ran 60.8ms where pure prefill of the same tokens took 42.3ms. nsys attribution: +14.6ms was GEMM alone. Root cause: the unified step's GEMMs run at N=1024+decode_bs, and **cuBLAS 12.9's kernel selection collapses at N=1025** (standalone GemmEx microbench, bf16/COMPUTE_32F: gate 9728×2560 234→347µs, oproj 2560×4096 94→185µs, down 2560×9728 223→299µs going from N=1024 to N=1025; qproj immune). cuBLAS 13.2 has no cliff.

The trap: `pegainfer-kernels/build.rs` derives both nvcc and the cublas link path from `CUDA_HOME` → `CUDA_PATH` → `/usr/local/cuda`. On a box where `/usr/local/cuda` symlinks 12.9, PATH exports do nothing — the server silently links `libcublas.so.12`. **Build with `CUDA_HOME=/usr/local/cuda-13.x`** and verify with `ldd target/release/pegainfer | grep cublas`.

After the CUDA 13.1 rebuild: unified step 60.8 → 47.1ms, QPS1 TTFT 64 → 50ms (arrivals landing mid-decode prefill faster). This also invalidated the doc's earlier claim that mid-band was "bounded by prefill near the bf16 roofline" — the measured 45ms prefill included the 12.9 cliff.

## Fixed: QPS1 decode-path tuning (PR #366)

At QPS1 the TPOT integrand is the decode step itself (43% of steps run bs≥2 from Poisson overlap). Three fixes, decode step (ctx1024, p50): bs1 6.51→5.96ms, bs2 7.02→6.53, bs4 8.11→6.72:

- **cublasLt per-shape algo tuning**: cuBLAS's default heuristic leaves 4–6% bandwidth on the table for every small-N decode GEMM (in-graph kernel times match Lt heuristic[0] exactly; the best candidate is 1.40–1.52 TB/s vs default's 1.28–1.48). `gemm_lt_tune` times all candidates at executor startup — on real weights rotated across all 36 layers so the loop stays L2-cold — and caches the winner per (M,N,K); GEMMs with N ≤ `GEMM_LT_MAX_N` (32) consult the cache, untuned shapes fall back to the old paths. The table dies with the process; **Cross-process algo store** below carries it across boots.
- **Split-KV decode attention bs≤2 → bs≤4, chunk 256 → 64 tokens**: the non-partitioned kernel runs one CTA per request×head = 8 CTAs at bs1 on 170 SMs. 64-token chunks measured fastest (32 is past the merge-overhead knee).
- **Two-stage batched greedy argmax**: tile-parallel partials + per-row finalize; the single-block-per-row kernel cost 91–191µs/step over the 151936 vocab.

## Cross-process algo store (opt-in)

`gemm_lt_tune`'s table lives in the process, so every boot re-searches shapes an earlier boot already paid for. Setting `PEGAINFER_GEMM_LT_CACHE=<file>` carries a winner across processes. Unset, the tuning path is byte-identical to before and no file is touched.

- **Path**: a file, created if it does not exist (its directory is not). The variable is read once per process; the file is opened per shape — looked up before that shape's search starts, appended one line when it finishes. Failing to open it, for any reason, is a miss: it costs start-up time, never a result.
- **Sharing**: a record's key is GPU name, compute capability, SM count, CUDA runtime version, driver version, cublasLt version, `M/N/K`, dtype, layout, and the workspace cap the algo was chosen under. One file can therefore back many processes and several machines; a record written by other hardware or another toolchain simply does not match. Writers append whole lines with `O_APPEND` and the last matching record wins, so concurrent boots need no lock; a torn tail is rejected on read, and the next append starts a fresh line rather than gluing onto it. Append atomicity across a shared filesystem is that filesystem's to promise, not ours — but a record that lands garbled is rejected on read, so the worst it costs is a search that had already been paid for once.
- **Invalidation**: a CUDA, cublasLt, or driver upgrade changes the key, so records written before it become misses — with one gap worth knowing. The driver field comes from `cudaDriverGetVersion`, which reports the CUDA level the installed driver supports and not its build number, so a driver update that keeps the same level (`570.86` to `570.124`, say) leaves the key identical and the old winners in play. Nothing re-times them: a stored algo is checked for validity, never for being still the fastest. If you want the ranking re-derived after a driver update, delete the file — it costs one tuning pass. A record whose key still matches but whose algo no longer validates against the live descriptors — or no longer fits the workspace — prints `gemm_lt_tune: stored algo rejected for [<key>]; retuning` and re-tunes, appending the new winner; a stale file therefore costs one re-tune, not a wrong answer. Deleting the file is always safe.
- **Adoption is verified, not trusted**: the stored bytes are an opaque `cublasLtMatmulAlgo_t` another process wrote, so they are put through `cublasLtMatmulAlgoCheck` against this process's descriptors and launched once before being published — the same "the tuned kernel has already executed" precondition CUDA-Graph capture leans on.
- **Not consulted under `--batch-invariant`**: that selects `Pin`, which pins the heuristic top without timing.

Measured on Qwen3-4B, single sm_89 card, 40 stored records, warm `ready`, over five order-rotated pairs of one binary whose arms differ only in whether the store is there to be read: median −657 ms (−33.4%), 5/5, the present arm holding 1309–1316 ms against 1957–2070 ms absent. Against the flat-20 tuner two steps back, warm ready goes 2560 ms → ~1965 ms with the repeat budget → ~1312 ms here.

## #345 root cause: per-request step tail

nsys at bs≈120 (`--cuda-graph-trace=node`, cudaProfilerApi capture): prefill/unified steps spent ~57ms (~25% of GPU time) in a per-request loop of extract_vec → single-row RMSNorm → lm_head GEMV (0.48ms each) → single-row sample → blocking 1KB pageable DtoH. Linear in batch size, so it presented as "batch doesn't scale" after #341 raised the CUDA-graph buckets to 256. Decode kernels themselves scale fine and beat vLLM ≥bs32 (bs128 step: 13.7ms vs vLLM's 28.8ms implied).

Fix (`perf/qwen3-batched-step-tail`): gather last-token rows → batched RMSNorm → one GEMM → `select_batch_tokens_into` → one DtoH. Same audit found the pattern in other model crates — tracked in #353 (qwen35 is worst: full-vocab blocking DtoH per request per decode step).

## Chunked prefill (`--max-prefill-tokens`)

Prompts are split across steps under a per-step forwarded-token budget; mid-prompt requests live in a scheduler `prefilling` queue, the executor clamps each chunk at the request's `kv_position` (prefix-cache hits compose), non-final chunks apply KV without emitting a token (kvbm `apply_prefill(None)`). Echo never splits. Admission reserves KV + decode slots for mid-prefill requests; KV prefetch respects a reserve floor so it can't steal a queued chunk's blocks.

- **The budget is an ITL-tail vs throughput knob.** A unified step's duration scales with its prefill tokens and every decode request in it stalls for the whole step, so the budget bounds the stall.
- **Default is 1024**: vs 2048 it keeps the same mean TPOT (within ±0.3ms at QPS10–12) and halves mid-band ITL p99 (90→50ms @QPS10, 100→62 @QPS12) by never fusing two full 1024-token prompts into one step. 512 regresses *both*: the per-step fixed cost stops amortizing, prefill throughput falls behind arrivals, and TTFT queues up (median 187ms vs 86 at QPS12).
- **Chunking can't buy mean TPOT** — total prefill work is conserved; the knob only redistributes it across steps. The mean gains came from making the unified step itself cheaper (attention fusion below).

## Fixed: bs8–16 decode-step anomaly (cuBLAS skips split-K)

Fine sweep (ctx1024, CUDA graph, p50 ms): bs4 8.11, **bs8 9.42, bs12 9.47, bs16 9.52**, bs20 8.51 — non-monotonic, exactly the mid-band serving concurrency. nsys diff: bs20's GEMMs use split-K (GEMM med 16.8µs) while bs8's don't (med 23.2µs) — cuBLAS's GemmEx heuristic leaves ~20 CTAs on 170 SMs for batch∈[8,16]. Re-verified on cuBLAS 13.2: still present (bs8/16 steps 9.2/9.3ms vs bs20 7.9ms).

Fix history: first removed buckets 8/16 so batches 5–19 pad up to 20. Then the cublasLt tuning (`GEMM_LT_MAX_N=32`) turned out to have full-speed candidates at every small N — the hole is GemmEx-only — so the buckets came back tuned: bs8 9.22→6.87, bs16 9.32→7.30. Split-KV attention extended to the same range (cap 32; wins shrink with batch, noise past bs40).

## Fixed: unified-step attention fusion (mid-band TPOT)

The unified step ran prefill and decode attention as separate kernel families per layer (BatchPrefill for prefill rows + BatchDecode for decode rows, each with its own rope/scatter glue). Now decode requests enter the `PrefillPagedPlan` as qo_len=1 rows over their full KV history — the same shape a 1-token prefill chunk already exercises — so each layer runs one scatter and one varlen BatchPrefill call for everything, via the same `prefill_attention_paged_into` op the pure-prefill path uses.

Routing through that op also fixed a silent tile mismatch worth ~3ms/step: the plan laid out tiles for `cta_tile_q=64`, but the raw FFI call let the FlashInfer kernel re-derive its own tile size (128 for any step ≥17 tokens), leaving half the planned tiles empty. With the dispatch honoring the plan's tile, a 1024-token unified step is 45.1ms — *cheaper* than the old split path's 47.1ms and a pure prefill step of the same tokens.

Step-phase attribution (PHASELOG, QPS10): host work is irrelevant — scheduler bookkeeping, KvView builds, channel send, apply/seal together cost <10µs per step at dbs32; the step duration *is* GPU time. Mid-band TPOT decomposes as decode-step mix (~8ms) + unified-step rides: each of the ~1 prefill/request unified steps stalls every decode request in it for the full step, contributing ~6ms/token at QPS10. That made U-step duration (× count) the whole lever — hence the fusion, the tile fix, and the 1024 chunk budget.

## Current sweep (all fixes, cuBLAS 13.1; TPOT mean / ITL p99 ms, TTFT mean ms)

| QPS | pegainfer | vLLM 0.22.1 | TTFT oi / vllm |
|---|---|---|---|
| 1 | **6.51** / 6.9 | 6.86 / 12 | **47** / 56 |
| 8 | **10.70** / 46 | 12.09 / 47 | **57** / 68 |
| 10 | **13.82** / 48 | 14.62 / 82 | **72** / 79 |
| 12 | **19.65** / 58 | 20.49 / 103 | **96** / 123 |
| 16 | **61.65** / 257 (**1865 tok/s**) | 78.55 / 119 (1674) | **1355** / 4279 |

(QPS 2/4 unchanged from the pre-Lt sweep: 8.10 / **8.63** vs vLLM 7.42 / 8.89. QPS16 row predates the fusion; overload is decode-throughput-bound. Mid-band numbers are seed-42 single runs on an otherwise idle box; cross-run drift is ±0.3ms — an earlier sweep with co-tenant load read 3–4ms worse across the board, so check `nvidia-smi` before trusting a regression.)

QPS16 admission verdict (#345): no knee anymore — the old 90ms TPOT regression at QPS12+ was the per-request step tail, not admission policy. Current overload behavior is strictly better than vLLM's; no admission cap needed.

## Pitfalls

- **Stale `/usr/local/cuda` symlink silently links cuBLAS 12** — see the N=1025 cliff section. Always `CUDA_HOME=/usr/local/cuda-13.x cargo build` and check `ldd`.
- Microbenching GEMMs with a single weight buffer measures L2-hot timings (96MB L2 on GB202 holds entire decode weights) — rotate ≥4 weight copies or the ranking is fiction. Same for algo tuning, hence the all-layer rotation in `gemm_lt_tune`.
- `vllm bench serve` TPOT in overload follows the identity TPOT ≡ bs/throughput — admission policy changes TPOT without changing kernel speed. Compare out_tok/s alongside.
- nsys `--cuda-graph-trace=node` inflates step times 30–60%; use it for composition only, never absolute TPOT.
- pkill from an ssh one-liner matches its own command line — use `pkill -f "[t]arget/release/pegainfer"`.
