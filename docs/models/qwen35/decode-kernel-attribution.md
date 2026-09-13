# Qwen3.5 decode kernel attribution vs vLLM 0.27 (A100)

> **TL;DR:** nsys kernel-level attribution of the remaining serving gap (1×A100-40GB, upstream/main September 2026, vLLM 0.27.0, 1024-token prompts, `--cuda-graph-trace=node`, steady-decode session capture). Four findings: (1) at c16 total kernel-busy is 16.5 vs ~13.2 ms/step — GEMM family +3.5 ms/step (sm_80 decode buckets never tuned vs torch.compile-selected kernels); (2) at bs1 the GPU is 92% busy — kernel-busy 10.1 vs 8.2 ms/step, a genuine ~1.9 ms kernel-time deficit of which ~1.0 ms is FlashInfer `BatchDecodeWithPagedKVCache` (156 µs/layer-step vs `flash_fwd_splitkv` 26 µs, 6×); (3) per layer-step the GDN decode kernel is 97.0 µs vs vLLM's FLA `fused_recurrent` 43.6 µs (2.2×) and the FlashInfer paged decode 153.9 µs vs 64.6 µs (2.4×); (4) the once-per-step output-projection GEMM lands on a `cutlass_75_tensorop` align-1 kernel costing 1.67 ms/step (~12% of c16 TPOT) because the selection width is odd — fixed by aligning the selection width to the tile multiple (see the `perf(qwen35)` output-projection alignment change).
>
> **Last touched:** 2026-09

## Contract

- GPU 1× A100-SXM4-40GB (sm_80); model Qwen3.5-4B (Qwen/Qwen3.5-4B checkpoint); greedy, seed 42, random dataset (1024-token prompts).
- PegaInfer upstream/main (September 2026), default serve flags (serial, policy off — kernel composition is the subject, not scheduling).
- vLLM 0.27.0: FLASH_ATTN full attention + FLA Triton GDN + FlashInfer sampler, piecewise CUDA graphs.
- Capture: server under `nsys launch`/`start`/`stop` sessions with `--cuda-graph-trace=node`; capture armed after readiness + warmup, stopped after the bench; `--export=sqlite`; `nsys stats --report cuda_gpu_kern_sum`.
- Absolute times are node-trace inflated — used for **composition and cross-engine ratios at the same workload**, not as TPOT claims. Wall TPOT from the HTTP runs: PegaInfer bs1@1024 `9.82 ms`, vLLM `8.32`; c16 `14.26` vs `11.00`.

## Findings

### 1. c16 per-step kernel deficits: busy 16.5 vs ~13.2 ms/step (matches the 3.3 ms wall TPOT gap)

The decode shape itself is healthy: an `PEGAINFER_ITL_DEBUG` run shows 271 scheduler steps where `decode_n` ramps 0→16 over the first second (chunked prefill, one 1024-token request per unified step) and then stays **16/16 for the entire decode**; the clean capture agrees (270 implied steps, bucket-16 grids throughout).

The per-step kernel deficits at c16 (270 steps, bucket 16):

| family | PegaInfer | vLLM 0.27 | delta |
| --- | --- | --- | --- |
| GEMM family | 13.27 ms/step | ~9.78 | **+3.5** |
| GDN decode (24/step) | 2.33 (97.0 µs/layer-step) | 1.05 (43.6 µs) | **+1.3** |
| full-attn decode (8/step) | 1.23 (153.9 µs/layer-step) | ~0.53 (64.6 µs flash splitkv) | +0.7 |

vLLM numbers from its capture (251 steps × batch 16, `fused_recurrent_gated_delta_rule_packed` one launch per layer-step).

### 2. bs1: GPU 92% busy — the deficit is kernel time, concentrated in paged decode attention

Clean single-request capture (256 steps, kernel-busy 92% of wall): PegaInfer **10.1 ms/step** kernel-busy vs wall TPOT `9.82`; vLLM `8.2 ms/step` busy on a full-coverage capture. Composition: GEMM/GEMV `7.42 ms/step` (73%, ≈ vLLM's ~7.1 — both near the weight-read bandwidth floor, not the gap), **full-attn paged decode `1.25 ms/step` vs vLLM ~`0.21` — FlashInfer `BatchDecodeWithPagedKVCacheKernel` at 156 µs/layer-step vs `flash::flash_fwd_splitkv_kernel` 26 µs (6×)**, GDN decode 0.53 vs 0.21, rest small. The ~1.9 ms/step deficit ≈ attention (+1.0) + GDN (+0.3) + GEMM (+0.3).

### 3. Decode attention + GDN kernels: 2.1–2.4× per layer-step at batch 16 (bs1: up to 6×)

| kernel family | PegaInfer (batch 16) | vLLM (batch 16) | per layer-step |
| --- | --- | --- | --- |
| GDN decode | `gated_delta_rule_decode_batch_kernel` 97.0 µs ×24/step | `fused_recurrent_gated_delta_rule_packed` 43.6 µs ×24/step | **2.2×** |
| full-attn decode | FlashInfer `BatchDecodeWithPagedKVCacheKernel` 153.9 µs ×8/step | `flash::flash_fwd_splitkv_kernel` 64.6 µs ×8/step | **2.4×** |

At bs1@1024 the attention gap widens to 156 vs 26 µs/layer-step (**6×**): FlashInfer's paged decode is especially weak at tiny batch on sm_80. Our GDN kernel is one block per value head (grid = value_heads × batch, sequential k-head slices inside); FLA packs the work into one launch with a different tiling.

Neither kernel has had an sm_80 tuning pass (decode tuning history is sm_120/RTX 5090).

### 4. The once-per-step output-projection GEMM runs on an sm_75-era align-1 kernel (1.67 ms/step)

`cutlass_75_tensorop_bf16_s1688gemm_bf16_128x64_tn_align1` runs **once per decode step at 1.67 ms** (280 instances ≈ 270 steps) — an sm_75-era kernel at alignment 1, ~12% of c16 TPOT for a single launch. One GEMM per step points at the output projection (`selection_vocab × hidden` over the bounded vocab): the selection width is Qwen3.5-4B's tokenizer-decodable vocab **248077 — odd — so both the GEMM M and the logits leading dimension defeat cublasLt's vectorized ampere kernels**.

Follow-up: round the selection width to the 128-token tile multiple at the `bound_selection_vocab` boundary (still inside the 248,320-row checkpoint weight), so every downstream buffer and the sampler stay consistent.

### 5. What is NOT the problem

- bs1 GEMM/GEMV: ours 7.42 vs vLLM ~7.1 ms/step — both near the weight-read bandwidth floor.
- Host/launch overhead: the clean captures show 91–92% GPU busy with only ~0.75 ms/step of inter-kernel gaps.
- Admission/batching shape at c16: decode_n stays 16/16 after the ramp.
- Prefill: not captured in this pass (decode attribution only).

## Improvement queue (ordered by expected value)

1. Retune the decode GEMM family on sm_80 (+3.5 ms/step at c16 — cublasLt algo selection vs torch.compile's kernel choices; possible qkv/z/b/a projection fusion).
2. Replace or retune the full-attn paged decode path (FlashInfer `BatchDecodeWithPagedKVCache` → `flash_fwd_splitkv`-class: ~1.0 ms/step at bs1, 2.4× per layer-step at c16).
3. Retune the GDN decode kernel vs FLA `fused_recurrent` (2.2× per layer-step).
4. Fix the output-projection align-1 kernel (1.67 ms/step at c16): round the selection width to the tile multiple or pin a cublasLt algo.

## Claim boundary

Single runs per capture, one GPU, node-trace-inflated absolute times (composition/ratio claims only), kernel-family grouping with template args stripped. Aggregate-step counts are inferred from kernel instance counts (24 GDN layers/step, 8 full-attn layers/step). The vLLM bs1 capture covers its full run (span 4.61 s ≈ 512 steps). Two earlier readings from polluted or partial captures — "admission waves at c16" and "a 5.9 ms/step bs1 host gap" — were retracted during review; every number here comes from the clean captures listed above.
