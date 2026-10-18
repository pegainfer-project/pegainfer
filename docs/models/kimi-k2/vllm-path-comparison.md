# Kimi-K2 vLLM Path Comparison

> **TL;DR:** vLLM Kimi/DeepSeekV3 decode 和 PegaInfer decode 的最大结构差异已缩小到 MLA cache/metadata 与 collective bridge：PegaInfer 现在同样用 load-time `fused_qkv_a_proj` 合并 `q_a + kv_a`，decode 执行 `gemm_graphsafe(fused_qkv_a)` 后用 `kimi_mla_split_qkv_a` 一次拆出 `q_a/compressed_kv/k_rope`。MoE shared/main 与 routed compute/aux stream overlap、shared gate/up fused GEMM、dense layer0 gate/up fused GEMM、routed scale+residual add fused kernel、routed sum clear 与 Marlin locks clear 清理已通过 H20 correctness/perf gate；真实 fixture output16 steady TPOT p99 `14.26ms`，synthetic output64 steady TPOT avg `14.39ms` / p99 `14.83ms`。vLLM TP-only MoE final all-reduce cadence 已实测 BF16/F32 两版均慢于当前 RS bridge，因此保留 RS bridge。
>
> **Last touched:** 2026-05

## Source Map

- vLLM checkouts:
  - `$VLLM_DIR_ALT` on `H20 node` was readable from the main shell and matches the vLLM V0 files already used for Kimi fixture work.
  - A sub-agent also read `$LOCAL_VLLM_DIR`, whose V1 layout places MLA code under `vllm/model_executor/layers/mla.py` and `vllm/model_executor/layers/attention/mla_attention.py`. The operator structure is consistent, but file names differ.
- Kimi text model uses vLLM `DeepseekV2Model` / `DeepseekV3ForCausalLM`; Kimi-VL only wraps the language model and is out of scope.
- Main vLLM files:
  - `$VLLM_DIR_ALT/vllm/model_executor/models/deepseek_v2.py`
  - `$VLLM_DIR_ALT/vllm/attention/backends/mla/common.py`
  - `$VLLM_DIR_ALT/vllm/attention/backends/flashmla.py`
  - `$VLLM_DIR_ALT/vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`
  - `$VLLM_DIR_ALT/vllm/model_executor/layers/fused_moe/layer.py`
  - `$VLLM_DIR_ALT/csrc/cache_kernels.cu`
  - `$VLLM_DIR_ALT/csrc/moe/*`
- PegaInfer files:
  - `pegainfer-kimi-k2/src/direct/worker.rs`
  - `pegainfer-kimi-k2/src/batch_decode_trace.rs`
  - `pegainfer-kernels/src/ops/kimi_mla.rs`
  - `pegainfer-kernels/src/ops/kimi_router.rs`
  - `pegainfer-kernels/src/ops/kimi_experts.rs`

## vLLM Decode Operator List

This is the source-level list for Kimi/DeepSeekV3 decode, not an nsys trace. PyTorch, CUDA graph, and vLLM custom-op wrappers can fuse or hide individual CUDA kernels at runtime.

| Section | vLLM operator path | Source evidence |
| --- | --- | --- |
| Embedding | `get_input_embeddings(input_ids)` then model layers; TP vocab-parallel reduction is handled by vLLM parallel layers. | `deepseek_v2.py:704-716` |
| Attention input | `input_layernorm(hidden_states, residual)`; residual is carried by vLLM layer contract. | `deepseek_v2.py:609-616` |
| MLA q/kv down projection | `fused_qkv_a_proj = MergedReplicatedLinear(hidden_size, [q_lora_rank, kv_lora_rank + rope_dim])`; forward does one projection and splits into `q_c` and `kv_lora`. V1 small-batch code can route this through `min_latency_fused_qkv_a_proj` / `dsv3_fused_a_gemm`. | `deepseek_v2.py:410-417`, `deepseek_v2.py:505-510`; V1: `layers/mla.py`, `dsv3_fused_a_gemm` |
| MLA q branch | `q_a_layernorm(q_c)` then `q_b_proj(q_c)`; `q` is reshaped to local heads. | `deepseek_v2.py:425-433`, `deepseek_v2.py:511-522` |
| MLA kv branch | `kv_lora.split([kv_lora_rank, qk_rope_head_dim])`; `kv_a_layernorm(kv_c)`; `k_pe` goes through RoPE. | `deepseek_v2.py:517-526` |
| MLA cache append | `ops.concat_and_cache_mla(k_c_normed, k_pe, kv_cache, slot_mapping, ...)` writes latent KV and RoPE PE into MLA paged cache. | `common.py:1276-1285` |
| MLA q absorb | Decode path splits `q_nope/q_pe`, transposes `q_nope`, and runs `torch.bmm(decode_q_nope, W_UK_T)` to form `decode_ql_nope`. | `common.py:1297-1308` |
| MLA attention | FlashMLA path calls `_flashmla_C.fwd_kvcache_mla` via `flash_mla_with_kvcache`, passing block table, seq lens, tile scheduler metadata, and num splits. | `flashmla.py:212-225` |
| MLA v up | FlashMLA returns latent output; `_v_up_proj` runs per-head `torch.bmm(x, W_UV)` and reshapes to `num_heads * v_head_dim`. | `flashmla.py:227`, `common.py:1021-1027` |
| MLA output projection | `o_proj(attn_out)` is a row-parallel linear; TP reduction is handled inside vLLM parallel layer. | `deepseek_v2.py:528-534` |
| Dense layer 0 MLP | `DeepseekV2MLP`: vLLM V1 uses fused `gate_up_proj` GEMM where available, then SiLU multiply, then row-parallel down projection. The V0 path is still gate/up/down at the module level. | `deepseek_v2.py:590-645`; V1: `deepseek_v2.py:190-235` |
| MoE shared expert | `shared_experts = DeepseekV2MLP(..., reduce_results=self.experts.must_reduce_shared_expert_outputs())`. | `deepseek_v2.py:166-176` |
| MoE router | `gate(hidden_states)` then `grouped_topk(..., scoring_func=sigmoid, renormalize=True, num_expert_group, topk_group)` returns normalized top-k weights and ids. V1 has small-batch router GEMM specializations such as `dsv3_router_gemm` before grouped top-k. | `deepseek_v2.py:179-190`, `layer.py:1447-1461`; V1: `GateLinear`, `dsv3_router_gemm` |
| MoE route align | `moe_align_block_size(topk_ids, block_size_m, global_num_experts, expert_map)` produces `sorted_token_ids`, `expert_ids`, `num_tokens_post_padded`. | `fused_marlin_moe.py:99-109` |
| MoE W13 | `ops.moe_wna16_marlin_gemm(..., top_k=topk, mul_topk_weights=apply_router_weight_on_input, use_fp32_reduce=True)`; Kimi path uses WNA16 INT4 experts. | `fused_marlin_moe.py:133-159` |
| MoE activation | `torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, 2 * N))`. | `fused_marlin_moe.py:161-163`, `csrc/activation_kernels.cu` |
| MoE W2 | `ops.moe_wna16_marlin_gemm(..., top_k=1, mul_topk_weights=not apply_router_weight_on_input, use_fp32_reduce=True)`. | `fused_marlin_moe.py:175-201` |
| MoE route sum | `torch.sum(intermediate_cache3.view(...), dim=1, out=output)` sums the top-k route rows. | `fused_marlin_moe.py:203-205` |
| MoE scale and TP reduce | For BF16, routed output is multiplied by `routed_scaling_factor`, added with shared output, then `maybe_all_reduce_tensor_model_parallel`. | `deepseek_v2.py:187-208` |
| Final logits | Final RMSNorm then LM head; sampling/logprobs live in vLLM sampling path rather than model file. | `deepseek_v2.py:724-725` |

## PegaInfer Current Decode Operator List

This list follows the current worker implementation. The static trace is now source-aligned for these high-level operators after the MLA trace fix below.

| Section | PegaInfer actual operator path | Source evidence |
| --- | --- | --- |
| Embedding | `embedding_batch_vocab_shard` then TP all-reduce through BF16-via-F32 bridge. | `batch_decode_trace.rs:49-63` |
| Attention input | `rms_norm_batch_into(hidden, input_norm)`. | `worker.rs:1777-1783` |
| MLA q/kv down projection | `gemm_graphsafe(fused_qkv_a_proj)` then `kimi_mla_split_qkv_a` produces `q_a`, `compressed_kv`, and `k_rope`; q branch then runs `rms_norm_batch(q_a_norm)` and `gemm_graphsafe(q_b_proj)`, kv branch runs `rms_norm_batch(kv_a_norm)`. | `worker.rs:1784-1827` |
| MLA RoPE split | `kimi_mla_rope_split_decode(q_proj, k_rope, cos, sin, positions)` produces `q_nope`, `q_pe`, and `append_kpe`. | `worker.rs:1839-1849` |
| MLA q absorb | `kimi_mla_absorb_q_nope(kv_b_proj, q_nope)` uses preloaded `kv_b_proj` weight; this is the PegaInfer equivalent of vLLM `q_nope @ W_UK_T`. | `worker.rs:1850-1855` |
| MLA cache append | `kimi_mla_paged_kv_append(compressed_normed, append_kpe, page tables, positions)` writes worker-owned paged MLA KV. | `worker.rs:1856-1868` |
| MLA attention | `kimi_flashinfer_batch_decode_mla(q_abs_nope, q_pe, ckv_cache, kpe_cache, page tables, request_indices, kv metadata)`. | `worker.rs:1880-1895` |
| MLA v up | `kimi_mla_v_up(kv_b_proj, latent)`; this is the PegaInfer equivalent of vLLM `_v_up_proj`. | `worker.rs:1907-1912` |
| MLA output projection | `gemm_graphsafe(o_proj)` then TP all-reduce through BF16-via-F32 bridge, then residual add. | `worker.rs:1913-1934`, `batch_decode_trace.rs:279-291` |
| Dense layer 0 MLP | post-attn RMSNorm, separate gate/up GEMMs, `silu_mul_batch`, down GEMM, BF16-via-F32 TP all-reduce, residual add. | `batch_decode_trace.rs:294-327` |
| MoE shared expert | post-attn RMSNorm; load-time fused shared gate/up GEMM, `silu_mul_fused_batch_into`, shared down GEMM, BF16-via-F32 TP all-reduce. | `worker.rs:2201-2238` |
| MoE router | `kimi_router_noaux_tc_launch` with Kimi config, producing `router_topk_weight` and `router_topk_idx`. | `worker.rs:2262-2285` |
| MoE route align | `kimi_moe_marlin_align_block_size` builds local EP route metadata. | `worker.rs:2118-2127`, `batch_decode_trace.rs:360-377` |
| MoE W13 | `kimi_marlin_wna16_w13_gemm` using vLLM Marlin WNA16 package. | `worker.rs:2143-2153` |
| MoE activation | `kimi_marlin_w13_swiglu`. | `worker.rs:2154-2155` |
| MoE W2 | `kimi_marlin_wna16_w2_gemm` with top-k weights. | `worker.rs:2157-2166` |
| MoE route sum | `kimi_marlin_sum_topk_rows_f32`. | `worker.rs:2168-2169` |
| MoE combine | Current decode path uses `repeat_f32_for_reduce_scatter` + NCCL `reduce_scatter` for routed F32 bridge, then `scale_f32_in_place`, shared residual add, and `kimi_add_f32_bf16_to_bf16`. Older non-decode helper still has F32 all-reduce; decode trace should describe the decode path. | `batch_decode_trace.rs:410-460` |
| Final logits/top1 | final RMSNorm, LM head shard GEMM, `top1_batch`; worker reads local top1 ids/values back to host after graph replay and scheduler selects global max across ranks. | `batch_decode_trace.rs:74-96`, `worker.rs:797-824`, `scheduler.rs:528-604` |

## Count Snapshot

Current H20 static trace after fused `qkv_a` and shared gate/up:

```text
calls 1766
307 gemm_graphsafe
245 rms_norm_batch
123 all_reduce
122 add_batch
120 kimi_marlin_wna16_gemm
61  kimi_mla_split_qkv_a
61  kimi_mla_rope_split_decode
61  kimi_mla_absorb_q_nope
61  kimi_mla_paged_kv_append
61  kimi_flashinfer_batch_decode_mla
61  kimi_mla_v_up
61  silu_mul_batch
60  kimi_router_noaux_tc
60  kimi_moe_marlin_align_block_size
60  kimi_marlin_w13_swiglu
60  kimi_marlin_sum_topk_rows_f32
60  repeat_f32_for_reduce_scatter
60  reduce_scatter
60  kimi_scaled_add_f32_bf16_to_bf16
1   embedding_batch_vocab_shard
1   top1_batch
```

This count is source-aligned for the high-level worker operators. It still folds BF16-via-F32 collectives into one logical `all_reduce` and does not count CUDA memset/memcpy nodes.

## Trace Drift Fixed In This Session

`pegainfer-kimi-k2/src/batch_decode_trace.rs` differed from `worker.rs` in the first draft of this document:

| Trace item | Current trace | Actual worker path | Effect |
| --- | --- | --- | --- |
| q/kv down projection | now records one `fused_qkv_a` GEMM plus `kimi_mla_split_qkv_a` | worker uses load-time `DeviceMatrix::vstack(q_a_proj, kv_a_proj_with_mqa)` and one graph-safe GEMM | Fixed in the fused-qkv patch; removes one GEMM per layer and the old separate KV split path. |
| fused qkv split | now counted as `kimi_mla_split_qkv_a` | `kimi_mla_split_qkv_a` writes `q_a`, `compressed_kv`, `k_rope` directly | Provider added in `kernel_report.rs`; no second split kernel remains. |
| `kv_a_norm` | missing | `rms_norm_batch_into(compressed_kv, kv_a_norm)` | Fixed: RMSNorm count increased by 61. |
| decode `kv_b` GEMM | records `L*.attn.kv_b` as `gemm_graphsafe` | no full `kv_b` GEMM in decode; worker uses `kv_b_proj` weight in `absorb_q` and `v_up` custom kernels | Fixed: fake GEMM removed, GEMM count decreased by 61. |
| MLA cache append | missing | `kimi_mla_paged_kv_append` | Fixed: now counted once per layer. |
| all-reduce bridge | folded into one `all_reduce(dtype=bf16_via_f32)` | actual path is BF16-to-F32 kernel, NCCL F32 collective, F32-to-BF16 kernel | Fine for high-level op count, wrong for kernel launch count and CUDA graph node count. |
| top1 | `top1_batch` | kernel is `argmax_batch_bf16_cuda`; `ctx.sync()` + D2H id/value readback happen after graph body | The GPU op is counted, but graph-external host boundary is hidden. |

Patch range: `push_attention_layer` in `batch_decode_trace.rs` first removed the fake `kv_b` GEMM and added the missing MLA operators; the fused-qkv patch then replaced `q_a GEMM + kv_a GEMM + split_compressed_kv` with `fused_qkv_a GEMM + split_qkv_a`.

Validation:

```bash
cargo fmt --all --check
PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bins
PEGAINFER_CUDA_SM=90a cargo run --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_kernel_report -- \
  trace --source static --batch-size 4 --kv-len 1024 --out $RESULT_ROOT/kimi_decode_trace_fixed_bs4_kv1024.json
```

H20 validation for the fused-qkv patch used the same `cargo check` and static trace command under `$PEGAINFER_DIR` with `PEGAINFER_TRITON_PYTHON=$PEGAINFER_DIR/.venv-kimi/bin/python`; output was `calls=1886`, `gemm_graphsafe=367`, and `kimi_mla_split_qkv_a=61`.

Runtime model-report validation on H20:

```bash
LD_LIBRARY_PATH=$RESULT_ROOT/pegainfer-nccl-lib:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=$PEGAINFER_DIR/.venv-kimi/bin/python \
cargo run --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_model_report -- \
  decode --source runtime --batch-size 4 --kv-len 28 --iters 1 --format text \
  --out $RESULT_ROOT/kimi_runtime_model_report_bs4_kv28_fixed_trace_v2.json
```

The fused-qkv runtime report produced `total_schedule_calls=1886`, `measured_schedule_calls=1642`, `missing_schedule_calls=244`, and measured subset `136.549ms`. Missing providers are explicit:

- `all_reduce`: `123` calls, multi-rank H20 provider needed.
- `reduce_scatter`: `60` calls, multi-rank H20 provider needed.
- `kimi_mla_paged_kv_append`: `61` calls.

Measured subset top rows with `iters=1`: `kimi_marlin_wna16_gemm` `120` calls / `118.06ms`, `gemm_graphsafe` `367` calls / `5.73ms`, `kimi_router_noaux_tc` `60` calls / `2.61ms`, `rms_norm_batch` `245` calls / `2.03ms`, and `kimi_mla_split_qkv_a` `61` calls / `0.44ms`. This report is a corrected ledger gate, not a final TPOT number; it still lacks NCCL and `kimi_mla_paged_kv_append`.

H20 graph serving gates after fused-qkv:

- Synthetic `prompt-len=27`, `output-len=64`, `concurrency=4`, `--cuda-graph true`: steady TPOT avg `16.43ms`, p50 `16.48ms`, p95 `16.77ms`, p99 `16.82ms`; generated token hashes matched across all four rows.
- Real Kimi fixture rendered prompt, `output-len=16`, `concurrency=4`, `--cuda-graph true`: steady TPOT avg `16.15ms`, p50 `16.15ms`, p95 `16.29ms`, p99 `16.30ms`; all four rows matched the vLLM fixture prefix `[1008,2742,2531,414,19180,6082,1379,387,261,5216,63853,13,374,1765,11983,306]`.
- H20 GPU was released after the gates; `nvidia-smi --query-compute-apps` printed no active process.

## Path Differences That Matter

| Difference | vLLM | PegaInfer | Why it matters |
| --- | --- | --- | --- |
| MLA first projection | One `MergedReplicatedLinear` for `[q_lora_rank, kv_lora_rank + rope_dim]`. | Now one load-time fused `DeviceMatrix` plus one graph-safe GEMM and one split kernel. | This structural delta is closed in code. The keep/revert gate is H20 correctness plus TPOT/model-report improvement. |
| Dense gate/up | V1 can use fused `gate_up_proj`; V0 module-level path still exposes gate/up. | Dense layer still uses separate gate/up; MoE shared expert now uses load-time fused gate/up GEMM. | One dense layer only matters little; shared expert repeat cost is now closed at the high-level GEMM count. |
| Router GEMM | V1 has small-batch `dsv3_router_gemm` / `router_gemm_bf16_fp32` path before grouped top-k. | `kimi_router_noaux_tc_launch` is a single custom router/top-k kernel path. | Need compare microbench, not assume; router was ~3.7ms/step in old strong-sync profile. |
| MLA cache append and metadata | vLLM uses `concat_and_cache_mla`; FlashMLA prepares tile scheduler metadata and graph buffers. | PegaInfer uses `kimi_mla_paged_kv_append` and precomputed decode arena arrays. | Need compare metadata/cache append cost before changing attention kernels; trace currently hides this. |
| MLA q absorb/v up | vLLM uses `torch.bmm` with preprocessed `W_UK_T/W_UV`. | PegaInfer custom kernels `kimi_mla_absorb_q_nope` and `kimi_mla_v_up` over `kv_b_proj`. | Semantically aligned, but microbench should decide whether custom kernels or cuBLAS batched GEMM wins for bs1..4. |
| MoE WNA16 | Both use Marlin WNA16 route align, W13, SiLU, W2, sum. | PegaInfer has persistent workspace and explicit local EP route metadata. | Main MoE kernel choice is already aligned; next work is route histogram/tail and combine, not replacing WNA16. |
| Routed combine | vLLM EP path maps local experts via `expert_map`; final tensor-parallel reduce happens through vLLM distributed path. | PegaInfer currently uses NCCL bridge: local sum -> repeat -> reduce-scatter -> fused scale+residual add. | This is not PPLX EP; it is graph-capturable but likely still extra data movement. |
| TP collectives | vLLM parallel layers hide TP reductions; BF16 path does not visibly use our BF16-via-F32 bridge. | PegaInfer uses BF16-via-F32 bridge for hidden all-reduces because BF16 collective changed greedy output. | This is correctness-driven overhead; replacing it needs external vLLM greedy/top-k gate. |
| Sampling/top1 | vLLM sampling/logprobs is integrated with its sampler path. | PegaInfer graph body ends at local top1; worker D2H reads local top1 and scheduler CPU-selects across ranks. | This graph-external boundary is real, but prior profile says it is not the largest item; fix after trace/accounting is accurate. |

## Routed Bridge Probe

Historical `kimi_graph_probe --probe routed-bridge-compare` (since retired, see changelog) compared the current NCCL bridge against a direct F32 all-reduce in the same CUDA Graph replay setup. H20 `world=8,batch=4,hidden=7168,replay_iters=500` measured direct all-reduce at max-rank `33.46us` and current `repeat_f32_for_reduce_scatter + reduce_scatter` at max-rank `32.90us` (`0.983x`). This says the current bridge should not be reverted to direct all-reduce for speed; the real next communication change needs token ownership with true AG/RS or a PPLX dispatch/combine path.

## TP-Only MoE Cadence Probe

Hypatia 对 `$LOCAL_VLLM_DIR` 的 Kimi/DeepSeekV3 TP-only path 做了源码对照：vLLM decode 是 embedding `1` 次、attention `61` 次、dense layer0 `1` 次、MoE final `60` 次 BF16 all-reduce，总计 `123` 次 BF16 all-reduce，MoE TP-only path 不使用 reduce-scatter。PegaInfer 当前是同样 `123` 次 logical hidden all-reduce，再额外加 `60` 次 routed `repeat+RS` bridge。

把 PegaInfer decode MoE 临时改成 vLLM TP-only final all-reduce 后，H20 correctness 通过但性能回退：

| Variant | output16 steady | output64 steady | Decision |
| --- | --- | --- | --- |
| BF16 final all-reduce | avg `14.925ms`, p99 `18.285ms` | avg `15.048ms`, p99 `16.129ms` | Reverted |
| BF16-via-F32 final all-reduce | avg `14.730ms`, p99 `15.705ms` | avg `14.818ms`, p99 `15.227ms` | Reverted |
| Current RS bridge + dense gate/up fused | p99 `14.258ms` | avg `14.388ms`, p99 `14.834ms` | Kept |

Conclusion: source-level cadence parity alone is not a keep criterion. The next communication change must either implement real token ownership/AG-RS/PPLX semantics or show a measured graph/nsys win before touching the worker path.

## Next Actions

1. Profile any remaining p99/max tail under dense/shared gate-up fusion plus routed scaled-add fusion and Marlin locks clear removal: output64 avg/p50/p95/p99 are now around `14.4/14.5/14.9/14.8ms`, with p99 under `15ms` in the latest kept gate.
2. Revisit full shared/EP communication overlap only with a production-shaped NCCL probe; isolated two-comm graph replay wins, but worker two-comm init/capture is not stable enough to ship.
3. Next graph-safe local wins: keep Marlin output clears unless route metadata proves every consumed row is written, add `kimi_mla_paged_kv_append` provider coverage, and design a real AG/RS or PPLX EP combine path that removes the repeat-for-RS bridge.
4. Keep MoE WNA16 kernel path unchanged until the corrected report shows a measured win candidate; current vLLM/PegaInfer MoE compute path is already structurally close.
