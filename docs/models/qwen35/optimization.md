# Qwen3.5-4B Optimization

> **TL;DR:** Hybrid 24 linear + 8 full attn. This file is the historical optimization ledger; the current RTX 5090 comparison is in [Qwen3.5 serving: pegainfer vs vLLM on RTX 5090](../../benchmarks/qwen35-4b-serving-vllm-rtx5090.md). The decode-tuning refresh fuses MLP gate/up and tunes decode cuBLASLt buckets, improving direct TPOT by `2.1-3.2%`; vLLM still leads 1024/256 HTTP decode and high-concurrency throughput.
>
> **Last touched:** 2026-09. Qwen3.5 runtime code lives in top-level `pegainfer-qwen35`. Current accuracy coverage is the crate-local HF logits gate in [accuracy.md](accuracy.md); run `e2e_scheduler` when scheduler request-flow behavior changes. The old exact-text e2e/regen baseline was retired by the HF logits gate in `docs/models/qwen35/accuracy.md`.

Historical command logs below keep the command paths that were actually run at the time (the in-process `bench_serving` bin they reference is retired as of 2026-08). For new Qwen3.5 accuracy tests, follow [accuracy.md](accuracy.md); for serving benchmarks, use HTTP-level benching (`scripts/bench_http_serving.py` / vllm-bench, see [profiling-guide](../../playbooks/profiling-guide.md)).

## Goal

Close Qwen3.5's decode and serving gap against the current vLLM baseline on the
same GPU/workload. The current refresh is a measured incremental step: direct
PegaInfer decode gets a few-percent TPOT improvement, but HTTP 1024/256 and
high-concurrency serving still trail vLLM and remain active optimization work.

## Current Decode Refresh

Changes in the decode-tuning refresh:

- MLP gate/up weights are stacked at load time, so runtime uses one gate-up GEMM
  plus fused SiLU*mul.
- Decode cublasLt buckets are tuned before CUDA Graph capture.
- Token selection stays on the shared batched sampler path.

Same-host direct A/B:

| Workload | Metric | upstream/main | tuned branch | Delta |
| --- | --- | ---: | ---: | ---: |
| 1 input / 256 output | steady TPOT avg | 6.524 ms | 6.386 ms | -2.1% |
| 1 input / 512 output | steady TPOT avg | 6.603 ms | 6.397 ms | -3.1% |
| 1024 input / 256 output | steady TPOT avg | 7.338 ms | 7.100 ms | -3.2% |
| 2048 input / 1 output | TTFT avg | 97.978 ms | 95.855 ms | -2.2% |

Current vLLM boundary:

- Prompt-len-1 HTTP decode is close: 1/256 TPOT mean `6.282 ms` vs vLLM
  `6.214 ms`, and 1/512 TPOT mean `6.381 ms` vs vLLM `6.221 ms`.
- 1024/256 HTTP decode still trails: TPOT mean `7.110 ms` vs vLLM `6.346 ms`
  at concurrency 1, widening to `15.566 ms` vs `9.823 ms` at concurrency 16.
- 2048/1 and 1024/256 TTFT rows are fixed-client timings, not token-normalized
  prefill throughput, because the servers report different prompt-token totals.
- Nsight Systems measured the direct 1024/256 concurrency-16 path at `9.320 ms`
  steady TPOT avg; the HTTP row is `15.566 ms`. The next gap search is
  serving/scheduler/event-sync overhead.

## Known Caveat

The major chunk-wise correctness blocker has now been identified and fixed:

- root cause: `gdr_chunk_state_qwen35_kernel` wrote `v_new` after multiplying by `exp(g_last - g_t)`
- correct semantics: write ungated `v_new` to memory, and only use the gated form for the recurrent `h += k @ v_new_gated` update

After this fix:

- mixed-pipeline validation shows:
  - `v_new` stage diff vs FLA drops to `max ~1.95e-3`
  - `chunk_o` output stage diff vs FLA is `max ~1.22e-4`
  - final recurrent state remains exact after layout alignment
- historical `cargo test --release --test e2e_qwen35 -- --nocapture` passed again before the exact-text gate was retired

Historical exact-text baseline note:

- regenerated `test_data/Qwen3.5-4B.json` changes `6/13` prompts
- changed cases currently are:
  - `tell_story`
  - `python_prime`
  - `quantum_simple`
  - `math_multiply`
  - `chinese_capital`
  - `chinese_weather`

## Historical E2E Dashboard

GPU: RTX 5070 Ti, Model: Qwen3.5-4B, vLLM 0.18.0, single concurrency.
Both measured via `vllm bench serve` HTTP client (apples-to-apples). vLLM: torch.compile + CUDA Graph (`--max-num-seqs 1` to fit in 16 GB alongside desktop).

| Profile | Metric | pegainfer | vLLM | delta |
|---------|--------|-----------|------|-------|
| prefill-heavy (2048,1) | TTFT median | 234.21ms | 229.25ms | +2% |
| prefill-heavy (2048,1) | TTFT p99 | 375.65ms | 8822ms¹ | — |
| decode-heavy (1,128) | TPOT median | 11.77ms | 11.67ms | +1% |
| decode-heavy (1,128) | ITL p99 | 12.23ms | 12.05ms | +1% |

¹ vLLM P99 is dominated by torch.compile cold-start on the first request; steady-state latency = median.

Reference — Qwen3-4B on same GPU: TTFT(2048,1)=213ms, TPOT(1,128)≈10.6ms.

## Architecture

- Layers: 32 (24 linear attention + 8 full attention)
- Full attention layer indices: 3, 7, 11, 15, 19, 23, 27, 31 (every 4th)
- hidden_dim: 2560
- MLP intermediate_size: 9216
- RMSNorm: (1 + weight) offset variant, eps=1e-6
- Word embeddings: 4B ties them (embed_tokens doubles as the LM head); 9B/27B are untied with a top-level `lm_head.weight`
- Vocab size: 248,320

**Full attention (8 layers):**
- 16 q_heads, 4 kv_heads (GQA ratio 4), head_dim=256
- Partial RoPE: rotary_dim=64 (25% of head_dim), theta=1e7
- Q projection includes output gate: [8192, 2560] (q + gate interleaved)
- QK norm (per-head, broadcast)

**Linear attention (24 layers):**
- 16 q_heads (k_dim=128), 16 k_heads (k_dim=128), 32 v_heads (v_dim=128)
- Conv1d: kernel_dim=4, depthwise on QKV (dim=8192)
- Gated delta rule: recurrent state [32 x 128 x 128] f32 per layer
- Output gating: Z projection [4096, 2560] → SiLU gate on RMSNorm'd output
- Parameters: A_log [32] f32, dt_bias [32] bf16, norm_weight [128] f32

### Full Attention Layer — prefill

```
RMSNorm_offset [2560,seq]
  → GEMM Q [2560→8192,seq]     ← batched
  → GEMM K [2560→1024,seq]     ← batched
  → GEMM V [2560→1024,seq]     ← batched
  → prefill_attention_hd256_prep (Q/K norm, partial RoPE, KV write)   ← batched CUDA helper
  → flash_attention_prefill_hd256_into (attention core)               ← batched Triton
  → attention_gate_batch_hd256 (apply sigmoid(gate))                  ← batched CUDA helper
  → GEMM O [4096→2560,seq]     ← batched
  → Residual + RMSNorm_offset [2560,seq]
  → GEMM Gate+Up [2560→18432,seq]  ← batched
  → SiLU*Mul [9216,seq]
  → GEMM Down [9216→2560,seq]  ← batched
  → Residual
```

**Current state:** full-attention prefill is no longer a meaningful TTFT bottleneck. The current end-to-end profile shows the remaining cost sits in linear-attention prefill compute.

### Linear Attention Layer — prefill

```
RMSNorm_offset [2560,seq]
  → GEMM QKV [2560→8192,seq]   ← batched
  → GEMM Z [2560→4096,seq]     ← batched
  → GEMM B [2560→32,seq]       ← batched
  → GEMM A [2560→32,seq]       ← batched
  → conv1d_prefill [8192,seq]  ← one launch/layer, updates conv_state across seq
  → gdr_prepare_qkv_gbeta [Q/K expand + g/beta]     ← one launch/layer
  → gdr_chunk_local_cumsum                          ← one launch/layer
  → gdr_chunk_scaled_dot_kkt                        ← one launch/layer
  → gdr_solve_tril_64                               ← one launch/layer
  → gdr_recompute_w_u                               ← one launch/layer
  → gdr_chunk_state                                 ← one launch/layer
  → gdr_chunk_o                                     ← one launch/layer
  → rms_norm_gated [4096,seq]  ← one launch/layer
  → GEMM O [4096→2560,seq]     ← batched
  → Residual + RMSNorm_offset [2560,seq]
  → GEMM Gate+Up [2560→18432,seq]
  → SiLU*Mul [9216,seq]
  → GEMM Down [9216→2560,seq]
  → Residual
```

**Critical:** the recurrence is still real, but it has moved from token granularity to chunk granularity. The linear stack now pays a small fixed chunk-wise pipeline per layer instead of a host-side token loop, and `RMSNorm_offset` was already fixed in #1 so it is no longer part of the prefill hotspot.

### Full Attention Layer — decode

```
RMSNorm_offset [2560]
  → GEMV Q [2560→8192]
  → GEMV K [2560→1024]
  → GEMV V [2560→1024]
  → fused_attention_hd256_decode (QK norm, partial RoPE, KV write, split-KV attention, output gate)
  → GEMV O [4096→2560]
  → fused_add_rms_norm_offset (residual + next layer norm)
  → fused_mlp (gate+up+SiLU+down)
  → fused_add_rms_norm_offset
```

### Linear Attention Layer — decode

```
RMSNorm_offset [2560]
  → GEMV QKV [2560→8192]
  → GEMV Z [2560→4096]
  → GEMV B [2560→32]
  → GEMV A [2560→32]
  → conv1d_decode (update conv_state [8192×3], SiLU)
  → gated_delta_rule_decode (update state [32×128×128] f32)
  → rms_norm_gated (norm * SiLU(z))
  → GEMV O [4096→2560]
  → fused_add_rms_norm_offset (residual + next layer norm)
  → fused_mlp (gate+up+SiLU+down)
  → fused_add_rms_norm_offset
```

Decode is fully CUDA Graph'd. Zero GPU allocation after first token. conv1d and GDR are single-token operations — no per-token loop penalty.

## Operator Performance

### Decode (1,128) — nsys kernel breakdown per decode step (current, 2026-03-28)

Total GPU kernel time: ~11.8ms/step (matches TPOT 11.81ms — fully GPU-bound, near-zero CPU overhead thanks to CUDA Graph).

Architecture: prefill-as-decode (#40) with restored GDR decode kernel (#9). MLP uses separate cuBLAS GEMV + silu_mul (not fused handwritten kernel). Full attention uses flash_attention_prefill_hd256 (Triton).

| Kernel | Time/step | % | Count/step | Avg each | Notes |
|--------|-----------|---|------------|----------|-------|
| gemvx (cuBLAS, all projections) | 9.27ms | 78.6% | 248 | 37μs | QKV/Z/B/A/O + MLP gate/up/down |
| gemv_handwritten (LM head) | 1.53ms | 13.0% | 1 | 1.53ms | final logits projection |
| gated_delta_rule_decode | 0.33ms | 2.8% | 24 | 13.8μs | fused recurrent state update |
| flash_attention_prefill_hd256 | 0.19ms | 1.6% | 8 | 23.6μs | full attention decode via Triton |
| rms_norm_batched_offset | 0.16ms | 1.4% | 64 | 2.5μs | residual + norm |
| argmax | 0.10ms | 0.8% | 1 | 98μs | |
| conv1d_prefill | 0.05ms | 0.4% | 24 | 2.0μs | |
| add_kernel | 0.04ms | 0.4% | 64 | 0.7μs | residual add |
| silu_mul | 0.02ms | 0.2% | 32 | 0.7μs | MLP activation |
| rms_norm_gated | 0.03ms | 0.3% | 24 | 1.3μs | |
| other | ~0.03ms | ~0.3% | — | — | embedding + first norm + attn helpers |

GEMV + MLP (cuBLAS + LM head + silu_mul) dominate at 91.6%. GDR is 2.8% after the dedicated decode kernel restore (#9).

### Decode (1,128) — refreshed operator split (2026-03-27)

Command used:

```bash
PEGAINFER_TRITON_PYTHON=./.venv/bin/python \
nsys profile --force-overwrite=true --trace=cuda,nvtx --cuda-graph-trace=node \
  --export=sqlite -o target/profiling/qwen35_decode_1x128_20260327 \
  cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B \
  request --prompt-len 1 --output-len 128 --warmup 1 --iters 1
```

Reference no-trace bench from the same session:

```bash
cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B \
  request --prompt-len 1 --output-len 128 --warmup 1 --iters 3
```

Observed no-trace result:

- TTFT avg `12.28ms`
- first decode step avg `12.11ms`
- steady TPOT avg `12.53ms`, p50 `12.54ms`, p99 `12.94ms`

Trace note: the `nsys` capture includes one warmup run plus one measured run, so the decode kernel counts below are divided by `256` total decode steps. In this case the summed kernel time still comes out to `12.531ms/step`, so the operator split is directly usable.

| Operator family | Time/step | % | Count/step | Avg each | Notes |
|-----------------|-----------|---|------------|----------|-------|
| GEMV total | 4.97ms | 39.6% | 153 | 32.5μs | all non-MLP projections + LM head |
| fused_mlp_intermediate | 3.73ms | 29.8% | 32 | 116.5μs | gate+up projection and SiLU*mul |
| fused_mlp_output | 1.91ms | 15.2% | 32 | 59.7μs | down projection |
| gated_delta_rule | 1.10ms | 8.7% | 24 | 45.7μs | one linear-attn recurrent update per linear layer |
| fused_attention_hd256 | 0.48ms | 3.8% | 8 | 59.9μs | one full-attn decode kernel per full-attn layer |
| fused_add_rms_norm_offset | 0.22ms | 1.7% | 64 | 3.4μs | two residual+norm kernels per layer |
| conv1d_decode | 0.05ms | 0.4% | 24 | 2.1μs | one per linear layer |
| argmax | 0.05ms | 0.4% | 1 | 48.9μs | greedy selection is already inside the graph |
| rms_norm_gated | 0.03ms | 0.2% | 24 | 1.2μs | one per linear layer |
| first norm + embedding | <0.01ms | ~0% | 2 | — | negligible |

The GEMV family can be split further by launch shape (`gridX`) because each output width maps to a distinct projection class in the current Qwen3.5 decode path:

| GEMV subfamily | Time/step | % | Count/step | Avg each | Mapping |
|----------------|-----------|---|------------|----------|---------|
| Q / QKV (8192-dim) | 1.65ms | 13.1% | 32 | 51.5μs | 8 full-attn `q_proj` + 24 linear-attn `in_proj_qkv` |
| LM head (248320-dim) | 1.50ms | 12.0% | 1 | 1.50ms | final logits projection |
| O projection (2560-dim) | 0.84ms | 6.7% | 32 | 26.2μs | 8 full-attn `o_proj` + 24 linear-attn `out_proj` |
| Z projection (4096-dim) | 0.65ms | 5.2% | 24 | 27.1μs | 24 linear-attn `in_proj_z` |
| B / A projection (32-dim) | 0.20ms | 1.6% | 48 | 4.2μs | 24 linear-attn `in_proj_b` + 24 `in_proj_a` |
| K / V projection (1024-dim) | 0.13ms | 1.1% | 16 | 8.3μs | 8 full-attn `k_proj` + 8 `v_proj` |

Interpretation:

- The dominant decode cost is still matrix-vector bandwidth work: `GEMV + fused_mlp = 10.61ms/step = 84.6%` of TPOT.
- Within the plain GEMV bucket, the biggest items are `Q/QKV` and the `LM head`; together they are `3.14ms/step`, about one quarter of total TPOT.
- The hybrid-only recurrent path is visible but not dominant: `gated_delta_rule = 1.10ms/step`, versus `5.64ms/step` for the fused MLP pair.
- The 24 linear-attention layers add extra decode projection pressure (`QKV`, `Z`, `B`, `A`, `out_proj`) relative to dense-attention Qwen3-4B; that is the main reason Qwen3.5 sits above the Qwen3-4B `~10.6ms` TPOT reference on the same GPU.
- There is still no evidence that host-side decode orchestration is the limiter. The archived pure-GPU decode experiments remain consistent with this profile: the problem is kernel compute, not the CPU loop.

### Decode hotspot workflow note (2026-03-27)

This decode pass established a simple workflow worth reusing for future operator work:

1. Run a decode-heavy end-to-end profile first (`prompt_len=1, output_len=128`) and identify the largest kernel families from `nsys`.
2. Map those kernel families back to concrete model projections using launch shape and the decode code path.
3. Add only the hottest real model shapes to `ops_bench` and microbench them in isolation before attempting kernel rewrites.

For this pass, `ops_bench` was updated in-place rather than adding a new benchmark surface. The `gemv` bench now includes the Qwen3.5 decode-critical shapes:

- `8192x2560` (`q_proj` / `in_proj_qkv`): `~23.17us`
- `4096x2560` (`in_proj_z`): `~16.68us`
- `1024x2560` (`k_proj` / `v_proj`): `~10.77us`
- `32x2560` (`in_proj_b` / `in_proj_a`): `~9.73us`
- `2560x4096` (`o_proj` / `out_proj`): `~15.73us`
- `248320x2560` (LM head): `~1.505ms`

Command:

```bash
cargo bench --bench ops_bench -- gemv
```

Interpretation:

- The microbench ranking matches the decode trace: `Q/QKV` and `LM head` are the largest plain-GEMV costs, while `B/A` is inefficient per element but too small to matter much in TPOT.
- This makes the next decode optimization step concrete: focus on the large projection shapes first, not the generic CPU loop or tiny helper kernels.

### Triton GEMV reference pass (2026-03-27)

To check whether decode GEMV is mainly "missing a better implementation" versus "already near the limit of a bandwidth-bound shape", a temporary Triton JIT autotune probe was used during this pass. The probe was intentionally not kept in-tree after the experiment; the results below are the durable takeaway.

The probe autotuned a simple row-parallel GEMV family over:

- `BLOCK_M in {32, 64, 128, 256}`
- `BLOCK_K in {64, 128, 256}`
- `num_warps in {2, 4}`
- `num_stages in {2, 3}`

Results:

| Shape | Triton JIT autotune | Torch `mv` | Best config | Existing handwritten GEMV |
|-------|----------------------|------------|-------------|---------------------------|
| `8192x2560` (`Q/QKV`) | `59.94us` | `76.62us` | `BLOCK_M=128, BLOCK_K=256, warps=4, stages=3` | `~23.17us` |
| `2560x4096` (`O`) | `31.89us` | `51.85us` | `BLOCK_M=64, BLOCK_K=256, warps=4, stages=2` | `~15.73us` |
| `248320x2560` (LM head) | `3.33ms` | `3.44ms` | `BLOCK_M=32, BLOCK_K=256, warps=2, stages=3` | `~1.505ms` |

Interpretation:

- Triton improves clearly over the generic library reference (`torch.mv`) on the medium decode shapes, so it is useful as a kernel-research and autotune surface.
- However, this simple Triton GEMV family is still well behind the current handwritten CUDA kernel on the real Qwen3.5 hotspots: roughly `2.6x` slower on `Q/QKV`, `2.0x` slower on `O`, and `2.2x` slower on the LM head.
- The LM-head result is especially important: even after autotune, Triton only reaches rough parity with `torch.mv`, while the existing handwritten kernel is much faster. That strongly suggests the remaining performance comes from lower-level memory-system details, not just trying a different high-level DSL.
- Practical conclusion: Triton is worth keeping as a reference implementation and autotune probe, but it is not yet a drop-in replacement for the decode GEMV path. If decode GEMV remains the target, the main value of Triton here is helping bracket the problem, not solving it outright.

### Handwritten GEMV `ncu` spot check (2026-03-27)

Nsight Compute was used to inspect the current handwritten CUDA GEMV on two decode-critical shapes:

- `248320x2560` (LM head)
- `8192x2560` (`Q/QKV`)

The key takeaway is that both are already memory-bound, with very high achieved occupancy and no spill pathologies. The large LM-head shape is the clearest "pure DRAM streaming" case; the medium `Q/QKV` shape is similar but also shows a mild launch-tail effect.

| Shape | Time | DRAM Throughput | Memory Throughput | Compute Throughput | Achieved Occupancy | L2 Hit Rate | Notes |
|-------|------|-----------------|-------------------|--------------------|--------------------|-------------|-------|
| `248320x2560` (LM head) | `1.66ms` | `87.24%` | `769.84 GB/s` | `18.94%` | `97.56%` | `0.37%` | classic large streaming GEMV |
| `8192x2560` (`Q/QKV`) | `58.46us` | `82.27%` | `725.32 GB/s` | `17.76%` | `95.21%` | `1.74%` | still memory-bound; minor partial-wave tail |

Common observations:

- `Registers/thread = 40`, no local-memory spilling, no shared-memory spilling.
- Occupancy is already high (`95%+`) on both shapes, so there is no obvious launch-configuration or register-pressure failure to fix first.
- Compute utilization stays below `19%` while DRAM sits above `82%`, which confirms the decode GEMV hotspot is bandwidth-limited, not ALU-limited.

Shape-specific interpretation:

- `LM head` is close to the textbook bandwidth roofline case. It has near-zero L2 hit rate and very high DRAM throughput, so the current handwritten kernel is already close to the practical limit of a full-vocabulary streaming logits projection on this GPU.
- `Q/QKV` is also bandwidth-bound, but `ncu` reports an estimated `20%` speedup opportunity from a partial-wave tail: grid size `2048`, theoretical limit `6 blocks/SM`, four full waves plus one partial wave of `368` blocks. That does not change the overall conclusion, but it is the first concrete low-level hint that the medium GEMV shapes may still have some room left.

Practical conclusion:

- Do not expect a large win from rewriting the LM-head GEMV in another DSL alone; it is already close to the DRAM limit.
- If decode GEMV work continues, the better targets are the medium projection shapes (`Q/QKV`, then likely `O` and `Z`), where launch geometry and shape-specialized dispatch may still buy something measurable.

Follow-up experiment:

- A focused `Q/QKV` launch-geometry sweep was tried by changing `ROWS_PER_BLOCK` from the current `4` to `6` and `8` on the real `8192x2560` shape.
- Measured results were worse, not better: baseline `rows=4` stayed at `~23.31us`, while `rows=6` regressed to `~23.77us` and `rows=8` regressed to `~23.89us`.
- That is useful in itself: the `ncu` partial-wave warning is real, but it is not the main limiter for this kernel. Simply reducing the number of blocks per launch does not beat the current balance of memory parallelism and per-block work.
- Immediate implication: if `Q/QKV` is pursued further, the next experiment should target data movement (`x` staging / reuse, alternative vectorization, or a different medium-shape dispatch), not just `ROWS_PER_BLOCK`.

### Fused MLP spot check (2026-03-27)

`ops_bench` was extended with the real Qwen3.5-4B decode MLP shape (`2560 -> 9216 -> 2560`). The end-to-end fused MLP microbench lands at `~184.4us`, which is consistent with the decode trace split (`~116.5us` intermediate + `~59.7us` output).

Nsight Compute confirms that both MLP phases are also bandwidth-bound:

| Kernel | Time | DRAM Throughput | Memory Throughput | Compute Throughput | Achieved Occupancy | Registers/thread | L2 Hit Rate |
|--------|------|-----------------|-------------------|--------------------|--------------------|------------------|-------------|
| `fused_mlp_intermediate_kernel` | `130.69us` | `84.21%` | `742.98 GB/s` | `15.02%` | `76.69%` | `48` | `0.99%` |
| `fused_mlp_output_kernel` | `68.58us` | `80.73%` | `711.72 GB/s` | `11.28%` | `75.02%` | `40` | `3.37%` |

Interpretation:

- Both kernels are clearly memory-bound, not compute-bound: DRAM is `80%+` while SM-compute stays in the `11%–15%` range.
- Neither kernel shows spilling, so there is no obvious register-pressure failure to clean up first.
- The fused intermediate kernel is already doing the most important structural optimization: gate and up projections share the same pass over `x`.
- The output kernel shows low waves per SM (`0.76`) because the grid is only `320` blocks, so there is some occupancy / scheduling slack. However, it is still fundamentally a bandwidth kernel, not a compute kernel.

Practical conclusion:

- MLP is a large decode cost, but it does not look like "easy kernel engineering money". Like GEMV, it is already operating close to the memory roofline on this GPU.
- If decode work continues, pure CUDA-kernel iteration on MLP should be treated as lower priority than architecture-specific costs such as GDR, or broader decode-path changes that remove work instead of trying to execute the same work slightly faster.

### Prefill (128 tokens) — baseline before batched Triton full-attention wiring

Wall clock: 669ms. GPU kernel time: 222ms. **CPU kernel launch overhead: 447ms (67%).**

| Kernel | Time/prefill | % of GPU | Count/prefill | Avg each | Notes |
|--------|-------------|----------|---------------|----------|-------|
| gated_delta_rule | 138ms | 62.2% | 3,072 | 45μs | 128 tokens × 24 layers, sequential |
| fused_attention_hd256_single_token | 46ms | 20.8% | 1,024 | 45μs | 128 × 8 layers, old decode-style kernel reused during prefill |
| rms_norm_offset | 16ms | 7.1% | 8,193 | 1.9μs | per-token (no batched kernel) |
| cuBLAS GEMM (batched projections) | 12ms | 5.3% | ~250 | 48μs | well-utilized, not a bottleneck |
| conv1d_decode | 4.4ms | 2.0% | 3,072 | 1.4μs | |
| rms_norm_gated | 3.7ms | 1.7% | 3,072 | 1.2μs | |
| other | ~2ms | ~1% | — | — | silu_mul, add, embedding, argmax |

**Key insight (baseline):** GPU only works 222ms out of 669ms wall clock. The remaining 447ms is CPU overhead from launching ~20,000 tiny kernels in a sequential per-token loop. At seq=2048, this scales to ~160K kernel launches → 16.8s wall clock.

This section is retained as the pre-#3 baseline. Full-attention prefill has since been moved to a batched Triton path; the updated end-to-end `nsys` capture is summarized below.

### Prefill (2048 tokens) — before Triton GDR prefill (#5 historical snapshot)

`nsys` command used:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx --cuda-graph-trace=node \
  --export=sqlite -o target/profiling/qwen35_prefill_2048 \
  cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B \
  request --prompt-len 2048 --output-len 1 --warmup 1 --iters 1
```

Trace note: this capture includes one warmup request plus one measured request, so instance counts below are doubled relative to a single `(2048,1)` run. Percentages are still representative.

| Kernel | GPU% | Instances in trace | Avg each | Notes |
|--------|------|--------------------|----------|-------|
| gated_delta_rule_decode | 87.6% | 98,304 | 45.0us | 24 linear-attn layers × 2048 tokens × 2 runs |
| conv1d_decode | 2.8% | 98,304 | 1.44us | linear-attn sequential prefilter |
| rms_norm_gated | 2.4% | 98,304 | 1.21us | linear-attn post-GDR gating |
| batched GEMM kernels (CUTLASS) | ~6.3% | 416 | 122us to 1.04ms | projections + MLP, no longer dominant |
| flash_attention_prefill_hd256_kernel | 0.4% | 16 | 1.41ms | 8 full-attn layers × 2 runs |
| prefill_attention_hd256_prep + gate + v-cache helpers | <0.1% | 48 | 15us to 122us | Q/K norm + partial RoPE + gate are negligible |
| rms_norm_batched_offset | 0.0% | 128 | 12.5us | fixed in #1, no longer relevant |

**Kernel conclusion:** full-attention prefill is no longer the problem. The remaining GPU bottleneck is the linear-attention sequential loop, and specifically `gated_delta_rule_decode`.

`cuda_api_sum` from the same trace shows the host/runtime side is now dominated by bookkeeping rather than full-attention launches:

| API | Total calls in trace | Avg each | Interpretation |
|-----|----------------------|----------|----------------|
| cuMemsetD8Async | 689,163 | 4.65us | heavy temporary-buffer churn |
| cuMemcpyDtoDAsync_v2 | 491,522 | 5.92us | row extraction / writeback style device copies still dominate |
| cudaLaunchKernel | 295,094 | 5.25us | still a very large sequential launch count |
| cuMemAllocAsync | 689,545 | 0.97us | temporary allocations remain excessive |
| cuMemFreeAsync | 689,545 | 0.75us | same churn on the free side |

**API conclusion:** after deleting the old HD256 attention loop, prefill was still paying for a massive amount of linear-attention per-token orchestration: launches, DtoD copies, memsets, and alloc/free traffic. That is the exact problem addressed in #6 below.

### Prefill (2048 tokens) — after Triton GDR prefill (#6 current snapshot)

Stable request bench:

```bash
PEGAINFER_TRITON_PYTHON=./.venv/bin/python \
cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B \
  request --prompt-len 2048 --output-len 1 --warmup 1 --iters 3
```

Result:

- TTFT avg `377.89ms`
- TTFT p50 `377.20ms`
- TTFT p99 `379.85ms`

`nsys` command used:

```bash
PEGAINFER_TRITON_PYTHON=./.venv/bin/python \
nsys profile --force-overwrite=true --trace=cuda,nvtx --cuda-graph-trace=node \
  --export=sqlite -o target/profiling/qwen35_prefill_2048_gdr \
  cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B \
  request --prompt-len 2048 --output-len 1 --warmup 1 --iters 1
```

Trace note: this capture includes one warmup request plus one measured request, so instance counts below are doubled relative to a single `(2048,1)` run. The profile run measured TTFT `377.27ms`, matching the 3-iteration bench.

| Kernel | GPU% | Instances in trace | Avg each | Notes |
|--------|------|--------------------|----------|-------|
| gated_delta_rule_prefill | 48.2% | 48 | 7.39ms | 24 linear-attn layers × 2 runs; now the main compute kernel |
| batched GEMM kernels (CUTLASS) | ~44.1% | 496 | 14.9us to 1.04ms | projections + MLP are now co-dominant with GDR |
| flash_attention_prefill_hd256_kernel | 3.1% | 16 | 1.41ms | 8 full-attn layers × 2 runs |
| conv1d_prefill | 1.3% | 48 | 205us | one launch per linear-attn layer per run |
| rms_norm_gated | 0.6% | 48 | 95us | one launch per linear-attn layer per run |
| prefill qk/rope + gate + v-cache helpers | ~0.5% | 48 | 14.9us to 122us | still negligible |
| rms_norm_batched_offset | 0.2% | 128 | 12.6us | unchanged from #1; not relevant |

**Kernel conclusion:** the bottleneck has finally moved to genuine compute. Full-attention prefill is small, host-side per-token orchestration is gone, and the remaining cost is `gated_delta_rule_prefill_kernel` plus batched GEMM.

`cuda_api_sum` from the same trace:

| API | Total calls in trace | Avg each | Interpretation |
|-----|----------------------|----------|----------------|
| cudaLaunchKernel | 278 | 1.28ms | request-level launch count has collapsed from the old per-token regime |
| cuLaunchKernel | 754 | 15.5us | includes driver-level kernel entry points |
| cuMemcpyDtoDAsync_v2 | 2 | 14.4us | row extraction / writeback style copies are effectively gone |
| cuMemAllocAsync | 1,513 | 37.0us | far lower than the old per-token churn, though still not zero |
| cuMemFreeAsync | 1,513 | 1.12us | tracks the same temporary allocation pattern |
| cuMemsetD8Async | 1,131 | 9.44us | no longer a dominant cost center |

Note: `cuMemcpyHtoDAsync_v2` dominates total API time in this whole-process trace because it includes model-load / one-time setup traffic, so it is not the useful signal for request-level comparison here.

**API conclusion:** compared with the historical #5 snapshot, the big wins are structural: `cudaLaunchKernel` went from `~295k` to `278`, `cuMemcpyDtoDAsync_v2` from `~492k` to `2`, and `cuMemAllocAsync` from `~690k` to `1.5k`. Prefill is no longer spending its life in launch/copy/allocation bookkeeping.

### Prefill (2048 tokens) — after chunk-wise GDR prefill (#7 current snapshot)

Stable request bench:

```bash
PEGAINFER_TRITON_PYTHON=./.venv/bin/python \
cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B \
  request --prompt-len 2048 --output-len 1 --warmup 1 --iters 3
```

Result:

- TTFT avg `222.45ms`
- TTFT p50 `222.55ms`
- TTFT p99 `222.85ms`

Correctness refresh:

```bash
PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo test --release --test e2e_qwen35 -- --nocapture
PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo test --release --test gen_test_data_35 -- --nocapture
```

Result:

- `e2e_qwen35` passes again after fixing the chunk-state `v_new` writeback bug
- regenerated `test_data/Qwen3.5-4B.json` differs from the old `HEAD` baseline on `6/13` prompts
- this refreshed baseline is accepted for the chunk-wise path

**Interpretation:** the first fused-recurrent rewrite removed the host-side disaster and got TTFT down to `~378ms`; the chunk-wise rewrite is the follow-up step that actually gets Qwen3.5 prefill-heavy latency to parity level on this GPU. Remaining work is now normal tuning and cleanup, not feasibility.

## Optimization Log

### #9 Restore GDR decode kernel for single-token path (2026-03-28)

**Goal:** Recover the +4.2% decode TPOT regression introduced by the accuracy-parity refactor (#40), which replaced the dedicated GDR decode kernel with the 7-stage chunk-wise Triton pipeline for all token counts.

**Root cause:** The prefill-as-decode refactor (#40) deleted the dedicated `gated_delta_rule_decode_kernel` and routed single-token decode through `gated_delta_rule_prefill_chunkwise_into` (7 Triton kernel launches per linear layer). For seq_len=1, the chunk-wise pipeline adds ~33μs/layer of launch overhead vs ~14μs for the fused CUDA kernel.

`nsys` regression breakdown (per decode step):

| Component | Before (#40) | After (#40) | Delta |
|-----------|-------------|-------------|-------|
| GDR (1 fused kernel) | 0.36ms | — | — |
| GDR (7 chunk kernels) | — | 0.78ms | — |
| **GDR subtotal** | **0.36ms** | **0.78ms** | **+0.42ms** |
| Attention (flash_attn) | 0.48ms | 0.22ms | −0.27ms |
| GEMV + MLP | 10.61ms | 10.87ms | +0.26ms |
| rest | 0.33ms | 0.35ms | ~0 |
| **Total TPOT** | **~11.78ms** | **~12.28ms** | **+0.50ms** |

GDR was the dominant regression source (0.42ms of 0.50ms). The GEMV+MLP cost increased ~0.26ms because MLP switched from fused handwritten kernels to separate cuBLAS GEMV + silu_mul, but this was partly offset by faster attention (flash_attn prefill kernel is faster than the old fused_attention_hd256 for single token).

**Changes:**

1. Restored the j-loop parallel GDR decode kernel (`csrc/gated_delta_rule.cu`, 190 lines) — identical to the #8 version.
2. Re-added FFI declaration and `gated_delta_rule_decode_into()` ops wrapper accepting `HiddenStates` for the single-token path.
3. Replaced `gated_delta_rule_prefill_chunkwise_into` with `gated_delta_rule_decode_into` in `single_token_kernels()`.
4. Removed `GdrChunkwiseScratch35` from `SingleTokenBuffers` (saves ~2MB VRAM).

The multi-token prefill path still uses the chunk-wise pipeline (unchanged).

**Validated commands:**
- `cargo test --release --test e2e_qwen35 -- --nocapture`
- `cargo test --release --test e2e -- --nocapture`
- `cargo test --release --test gen_test_data_35 -- --nocapture`
- `cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 1 --output-len 128`
- `cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 2048 --output-len 1`

**Results:**
- Decode-heavy `(1,128)`: TPOT avg `11.78ms`, p50 `11.81ms`, p99 `11.85ms` (was `12.28ms` → **−3.8%**)
- Prefill-heavy `(2048,1)`: TTFT `224.5ms` (unchanged)
- `e2e_qwen35`: pass after baseline regeneration; FP accumulation order change from fused kernel vs chunk-wise causes some greedy output drift
- `e2e` (Qwen3): pass (no changes to Qwen3 path)

`nsys` kernel breakdown after fix (per decode step, 256 steps):

| Operator family | Time/step | % | Count/step | Avg each |
|-----------------|-----------|---|------------|----------|
| gemvx (cuBLAS, all projections) | 9.27ms | 78.6% | 248 | 37μs |
| gemv_handwritten (LM head) | 1.53ms | 13.0% | 1 | 1.53ms |
| gated_delta_rule_decode | 0.33ms | 2.8% | 24 | 13.8μs |
| flash_attention_prefill_hd256 | 0.19ms | 1.6% | 8 | 23.6μs |
| rms_norm_batched_offset | 0.16ms | 1.4% | 64 | 2.5μs |
| argmax | 0.10ms | 0.8% | 1 | 98μs |
| conv1d_prefill | 0.05ms | 0.4% | 24 | 2.0μs |
| add_kernel | 0.04ms | 0.4% | 64 | 0.7μs |
| silu_mul | 0.02ms | 0.2% | 32 | 0.7μs |
| rms_norm_gated | 0.03ms | 0.3% | 24 | 1.3μs |
| other | ~0.03ms | ~0.3% | — | — |

**Interpretation:** The decode kernel restore recovered essentially all of the regression (11.81ms vs pre-#40 11.78ms). The remaining ~0.03ms gap is from unfused MLP (cuBLAS gemvx vs fused handwritten kernel), but cuBLAS gemvx has better tiling for these shapes so re-fusing would likely not help. GEMV + MLP remain the dominant cost at 91.6% — bandwidth-limited, not kernel-optimization-limited.

### #8 GDR decode kernel j-loop parallelism (2026-03-27)

**Goal:** Close the +7% decode TPOT gap vs vLLM by optimizing the `gated_delta_rule_decode_kernel`, which was 8.7% of decode time (1.10ms/step, 45.7μs/layer from nsys, 37.1μs/layer from microbench without tracing overhead).

**Root cause:** The kernel launched 32 blocks × 128 threads = 4 warps/block. On an 80-SM GPU, each active SM had only 4 warps — far too few to hide the ~300-cycle DRAM latency. The kernel was latency-bound, not bandwidth-bound.

Initial attempt — state layout transpose for coalescing — had no measurable effect (37.1μs → 37.1μs, within noise). The bottleneck was occupancy, not coalescing.

**Changes:**

1. **J-loop parallelism:** Split the 128-iteration j-loop across `J_SLICES=4` thread groups. Thread mapping: `val_idx = threadIdx.x % 128`, `j_slice = threadIdx.x / 128`. Each slice handles 32 j-iterations. Partial kv_mem and output reductions via shared memory. Block size: 128 → 512 threads (4 → 16 warps/block).

2. **State layout transpose:** Changed per-head state from `[V, K]` to `[K, V]` (V contiguous), matching FLA convention. Adjacent threads now access adjacent memory (coalesced). Updated both the CUDA decode kernel and the Triton prefill `gdr_chunk_state_qwen35_kernel` (removed `tl.trans` on initial/final state load/store).

3. **Pass fusion:** Merged 4 separate state passes into 2: decay+kv_mem (pass 1), rank-1 update+output (pass 2). Eliminated the shared-memory `smem_delta` round-trip.

**Validated commands:**
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo bench --bench ops_bench -- gated_delta_rule_decode`
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo test --release --test e2e_qwen35 -- --nocapture`
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 1 --output-len 128 --warmup 3 --iters 5`
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 2048 --output-len 1 --warmup 1 --iters 3`
- `cargo test --release --test e2e -- --nocapture` (Qwen3 unaffected)

**Results:**
- GDR decode kernel microbench: `37.1μs` → `14.8μs` (−60%)
- Decode-heavy `(1,128)`: TPOT avg `12.53ms` → `11.77ms` (−6.1%), p50 `12.54ms` → `11.78ms`, p99 `12.94ms` → `12.18ms`
- Prefill-heavy `(2048,1)`: TTFT `222ms` → `222ms` (unchanged)
- `e2e_qwen35`: pass after baseline regeneration; `1/13` prompt output changed (`tell_story`) due to FP accumulation order change from j-slice split
- `e2e` (Qwen3): pass (no changes to Qwen3 path)

**Interpretation:** The dominant decode optimization lever was thread-level parallelism (occupancy), not memory coalescing. With 16 warps per block instead of 4, the SM can overlap enough memory requests to approach the bandwidth roofline. The remaining TPOT gap vs vLLM is ~0.1ms (+1%), which sits within the GEMV/MLP bandwidth-limited floor — further gains would require lower-level kernel tuning rather than architectural changes.

### #7 Chunk-wise GDR prefill for Qwen3.5 (2026-03-21)

**Goal:** Replace the fused-recurrent linear-prefill GDR path with a chunk-wise path that preserves decode-state semantics while materially reducing prefill-heavy TTFT.

**Changes:** Added explicit chunk-wise scratch buffers, added Triton AOT stages for `gdr_prepare_qkv_gbeta`, `chunk_local_cumsum`, `chunk_scaled_dot_kkt`, `solve_tril_64`, `recompute_w_u`, `chunk_state`, and `chunk_o`, rewired the real Qwen3.5 prefill path to launch this pipeline per linear-attention layer, and fixed the main correctness bug in `gdr_chunk_state_qwen35_kernel`: `v_new` must be written back ungated and only the recurrent update should use the gated form. The chunk-wise solve/recompute/state/output kernels are adapted from FLA and now carry explicit source attribution in `tools/triton/gated_delta_rule_chunkwise_kernels.py`.

**Validated commands:**
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo check --release`
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 2048 --output-len 1 --warmup 1 --iters 3`
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo test --release --test e2e_qwen35 -- --nocapture`
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo test --release --test gen_test_data_35 -- --nocapture`

**Results:**
- Prefill-heavy `(2048,1)`: TTFT avg `222.45ms`, p50 `222.55ms`, p99 `222.85ms`
- `e2e_qwen35`: pass
- refreshed `test_data/Qwen3.5-4B.json`: accepted, with `6/13` prompt outputs changed relative to the old `HEAD` baseline
- focused stage-level comparison against FLA after the `v_new` fix:
  - `v_new` diff: `max ~1.95e-3`
  - `chunk_o` output diff: `max ~1.22e-4`
  - final recurrent state diff after layout alignment: `0.0`

**Interpretation:** the chunk-wise path is now the real runtime path for Qwen3.5 prefill. It preserves decode compatibility, restores E2E correctness, and cuts the prefill-heavy TTFT from the fused-recurrent `~378ms` level to `~222ms`, which is effectively parity on this setup.

### #6 Triton GDR prefill for Qwen3.5 (2026-03-21)

**Goal:** Remove the Qwen3.5 linear-attention prefill host-side token loop by replacing per-token GDR launches with a single Triton fused-recurrent kernel per layer.

**Changes:** Added a temporary Triton fused-recurrent GDR prefill kernel and its AOT build wiring, exposed it through FFI, added batched `conv1d -> GDR -> gated_norm` operator plumbing in `ops.rs`, rewired `Qwen35Model` linear prefill to use that path, and added a standalone prefill microbench entry. This path was later superseded by the chunk-wise implementation in `#7`.

**Validated commands:**
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 2048 --output-len 1 --warmup 1 --iters 3`
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 1 --output-len 128 --warmup 1 --iters 3`
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo test --release --test gen_test_data_35 -- --nocapture`
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python nsys profile --force-overwrite=true --trace=cuda,nvtx --cuda-graph-trace=node --export=sqlite -o target/profiling/qwen35_prefill_2048_gdr cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 2048 --output-len 1 --warmup 1 --iters 1`
- `nsys stats --report cuda_gpu_kern_sum target/profiling/qwen35_prefill_2048_gdr.sqlite`
- `nsys stats --report cuda_api_sum target/profiling/qwen35_prefill_2048_gdr.sqlite`

**Results:**
- Prefill-heavy `(2048,1)`: TTFT avg `377.89ms`, p50 `377.20ms`, p99 `379.85ms`
- Decode-heavy `(1,128)`: TTFT avg `12.27ms`, steady TPOT avg `12.53ms`, p50 `12.54ms`, p99 `12.94ms`
- `gated_delta_rule_prefill_kernel`: `48.2%` of GPU time, `48` instances, `7.39ms` avg each
- Batched GEMM kernels: another `~44.1%` of GPU time
- Launch/copy/allocation churn collapsed:
  - `cudaLaunchKernel`: `~295k` -> `278`
  - `cuMemcpyDtoDAsync_v2`: `~492k` -> `2`
  - `cuMemAllocAsync`: `~690k` -> `1,513`

**Correctness note:** Regenerating `test_data/Qwen3.5-4B.json` changes `6/13` prompts relative to the old baseline. Most changes are small tail drift, but `python_prime` changes materially. The focused GDR-prefill-vs-decode reference test passes, so this looks like an existing integration-level drift rather than an obvious Triton kernel failure.

**Interpretation:** This is the first Qwen3.5 profile where prefill is no longer dominated by orchestration. TTFT is now within about `1.7x` of vLLM at `(2048,1)`, and the remaining work is real recurrent/GEMM compute rather than launch bookkeeping.

### #3 Batched Triton full-attention prefill for Qwen3.5 (2026-03-21)

**Goal:** Replace the Qwen3.5 HD256 full-attention prefill loop with a batched Triton path and remove the legacy single-token CUDA prefill implementation.

**Changes:** Added `prefill_attention_hd256_prep_cuda` and `attention_gate_batch_hd256_cuda`, wired `Qwen35Model` full-attention prefill to `prefill_attention_hd256_batch()`, updated the `qwen35_prefill_attn` microbench to measure the runtime path, regenerated the Qwen3.5 greedy baseline, and deleted the dead `fused_gqa_attention_hd256_single_token` CUDA implementation.

**Validated commands:**
- `cargo test --release test_prefill_attention_hd256_batch_matches_cpu_reference -- --ignored --nocapture`
- `cargo test --release --test e2e_qwen35 -- --nocapture`
- `cargo bench --bench ops_bench -- qwen35_prefill_attn`

**Microbench result (runtime subpath: prep + Triton attention + gate):**
- seq128: 50.7us
- seq512: 365us
- seq2048: 1.64ms

**Interpretation:** Qwen3.5 full-attention prefill is no longer dominated by per-token HD256 attention launches. That follow-up profile is now in #5 and confirms the remaining TTFT cost sits in the linear-attention sequential kernels.

### #4 End-to-end refresh after Triton prefill wiring (2026-03-21)

**Goal:** Measure real request-level impact before running another profiling pass.

**Validated commands:**
- `cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 2048 --output-len 1 --warmup 1 --iters 5`
- `cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 1 --output-len 128 --warmup 1 --iters 5`

**Results:**
- Prefill-heavy `(2048,1)`: TTFT avg `3889.14ms`, p50 `3889.25ms`, p99 `3890.67ms`
- Decode-heavy `(1,128)`: TTFT avg `12.28ms`, steady TPOT avg `12.53ms`, p50 `12.54ms`, p99 `12.94ms`

**Interpretation:** Removing the old HD256 per-token prefill kernel improved TTFT by about `3.7x` versus the previous `14.5s` baseline while leaving decode essentially unchanged. The remaining gap to vLLM is now much more likely to sit in the 24 linear-attention layers and their sequential prefill kernels.

### #5 nsys reprofile after Triton prefill wiring (2026-03-21)

**Goal:** Confirm where prefill time moved after removing the old HD256 per-token full-attention path.

**Validated commands:**
- `nsys profile --force-overwrite=true --trace=cuda,nvtx --cuda-graph-trace=node --export=sqlite -o target/profiling/qwen35_prefill_2048 cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 2048 --output-len 1 --warmup 1 --iters 1`
- `nsys stats --report cuda_gpu_kern_sum target/profiling/qwen35_prefill_2048.sqlite`
- `nsys stats --report cuda_api_sum target/profiling/qwen35_prefill_2048.sqlite`

**Key findings:**
- `gated_delta_rule_decode_kernel`: `87.6%` of GPU time
- `conv1d_decode_kernel` + `rms_norm_gated_kernel`: another `5.2%`
- Batched Triton full-attention core: `0.4%`
- HD256 prep/gate helpers combined: `<0.1%`
- API activity still includes `~295k` kernel launches, `~492k` DtoD copies, and `~690k` alloc/free pairs in the warmup+iter trace

**Interpretation:** the full-attention prefill rewrite worked. The dominant bottleneck is now the linear-attention prefill loop and the large amount of per-token runtime bookkeeping around it. The next meaningful win will come from batching or restructuring linear-attention prefill, especially GDR.

### #2 Standalone HD256 Triton attention (2026-03-21)

**Goal:** Validate the Qwen3.5 full-attention prefill core in Triton before touching model wiring, and drop the old CUDA-vs-Triton microbench duplication.

**Changes:** Added `tools/triton/flash_attention_prefill_hd256_kernel.py`, `flash_attention_prefill_hd256_into()` in Rust/FFI, a focused CPU-reference test, and a Triton-only `qwen35_prefill_attn` microbench. Removed the legacy CUDA attention microbench entries so attention coverage now tracks only the Triton path.

**Validated commands:**
- `cargo test --release test_flash_attention_prefill_hd256_matches_cpu_reference -- --ignored --nocapture`
- `cargo bench --bench ops_bench -- qwen35_prefill_attn`

**Microbench result (attention only; excludes QK norm, partial RoPE, KV write, and output gate):**
- seq128: 30.9us
- seq512: 151us
- seq2048: 1.43ms

**Interpretation:** The batched HD256 attention core is no longer the blocking concern. The remaining work is the Qwen3.5-specific prep and gate wiring around it.

### #1 Batched RMSNorm_offset (2026-03-21)

**Bottleneck:** rms_norm_offset — 8,193 kernel launches per prefill at seq=128 (~131K at seq=2048), costing ~2.3s of CPU launch overhead + per-token extract/write memcpys.

**Approach:** New `rms_norm_batched_offset_kernel` with `<<<seq_len, 256>>>` grid — one block per token, single launch. Replaced per-token loop in `Qwen35Model::batched_rms_norm_offset`.

**Changes:** `csrc/norm.cu`, `src/ffi.rs`, `src/ops.rs`, `src/qwen35_model.rs`

**Result:** Kernel time seq=2048: 38.7ms → 17.7μs (2186×). RMSNorm now 0% of GPU time (64 launches vs 131K).

**E2E impact:** TTFT (2048,1): 16,846ms → 14,500ms (−14%). Bottleneck shifts entirely to per-token attention (80%) and GDR (17%).

| Kernel | GPU% | Instances | Avg |
|--------|------|-----------|-----|
| fused_attention_hd256_single_token | 80.4% | 16,384 | 630μs |
| gated_delta_rule | 17.2% | 49,152 | 45μs |
| rms_norm_batched_offset ← new | 0.0% | 64 | 13μs |

### #0 Baseline (2026-03-14)

**E2E numbers:**
- Prefill-heavy (2048,1): pegainfer 16,846ms vs vLLM 222ms (**76× slower**)
- Decode-heavy (1,128): pegainfer 12.55ms vs vLLM 11.64ms (+8%)
- Supplementary: prefill (128,1) TTFT 669ms → extrapolation confirms superlinear scaling due to O(n) attention per token

**Decode verdict:** Close to parity. 12.55ms TPOT vs vLLM 11.64ms (+8%), fully GPU-bound, CUDA Graph'd. Slower than Qwen3-4B (10.6ms) due to 153 GEMV/step (vs ~109 for Qwen3-4B) — linear attention's extra projections (Z, B, A). GDR kernel adds 1.09ms (8.7%). Not worth optimizing until prefill is fixed.

**Prefill verdict:** Catastrophic. Two independent problems:

1. **CPU launch overhead (67% of wall time):** The per-token loop launches ~20K tiny kernels at seq=128. Each launch costs ~22μs CPU overhead. At seq=2048, this alone accounts for ~10s.

2. **No batched kernels for attention/recurrent ops (33% of wall time):** GDR, attention, conv1d, and rms_norm_offset all use single-token decode kernels. Even if launch overhead were zero, processing 2048 tokens through 32 layers of single-token kernels would take ~3.5s (vs 213ms for Qwen3-4B with FA2).

**Next steps:**

1. ~~**Batched RMSNorm_offset kernel**~~ — done in #1.
2. **FlashAttention-2 for full attention layers during prefill** — replace 16,384 single-token attention calls with 8 FA2 launches (already implemented for Qwen3-4B, needs adaptation for HD256 + partial RoPE + output gate). **This is now the dominant bottleneck (80% GPU time).**
3. **Chunk-parallel GDR prefill** — reduce 49,152 per-token GDR launches to ~24 (one per layer) via chunk-parallel recurrence. 17% GPU time. Note: WY decomposition tried and failed (FP rounding → incoherent output). Needs a different approach.
