# Kimi-K2 Bring-up History

> **TL;DR:** Compressed chronology + decision records of the Kimi-K2 text-only bring-up (2026-05), consolidated from the retired `support-analysis.md` / `changelog.md` / `operator-todo.md` trio. Path: HF probe → text-only manifest → TP8/EP8 sliced loader → MLA + router + Marlin WNA16 routed experts → NCCL-sum EP bridge → bs4 wave decode → full CUDA Graph → vLLM top-20 greedy gate. The current engine (TP1/DP8/EP8 PPLX serving, paged KV, accuracy gate in git) is documented in [roadmap.md](roadmap.md), [optimization.md](optimization.md), [tp1-dp8-ep8-performance.md](tp1-dp8-ep8-performance.md), [accuracy-gate.md](accuracy-gate.md), and [kv-cache-design.md](kv-cache-design.md). This doc exists to explain *why* the early decisions were made and to hold the still-load-bearing checkpoint/INT4 layout facts. Numeric lessons general enough for other model lines are lifted to [docs/lessons/kimi-bringup-numerics.md](../../lessons/kimi-bringup-numerics.md).
>
> **Last touched:** 2026-06

## Load-bearing facts (still true)

### Model

| 项 | 值 |
| --- | --- |
| 文本 HF 类 | `DeepseekV3ForCausalLM`（外壳 `KimiK25ForConditionalGeneration`，vision/projector 本项目不支持） |
| `text_config.model_type` | `kimi_k2` |
| layers | `61`（layer 0 dense，layer 1..60 MoE）；`first_k_dense_replace = 1` |
| hidden / vocab | `7168` / `163840` |
| `max_position_embeddings` | `262144` |
| MLA | `num_attention_heads=64`，`q_lora_rank=1536`，`kv_lora_rank=512`，`qk_nope/qk_rope=128/64`，`v_head_dim=128`，`q_head_dim=192`，`o_proj` in `8192` → out `7168` |
| YARN RoPE | `theta=50000`，`factor=64`，original `4096`，`beta_fast/slow=32/1`，partial dim `64` |
| MoE | `n_routed_experts=384`，top-`8`，`n_shared_experts=1`，`moe_intermediate_size=2048`，dense layer0 FFN `18432` |
| Router | `noaux_tc`（sigmoid + group top-k），`n_group=1`，`topk_group=1`，`norm_topk_prob=true`，`routed_scaling_factor=2.827` |
| Routed quant | compressed-tensors INT4，`group_size=32`；只有 routed experts 被 pack-quantized，attention/shared/dense/lm_head 是 BF16 |
| Index | 64 safetensors shard，`total_size ≈ 595GB` |

K2.5 与 K2.6 文本架构相同，K2.6 是继续训练版；shape/TP8/EP8 规划共用。

### INT4 checkpoint format (on disk)

- Per-linear on-disk tensors: `weight_packed`、`weight_scale`、`weight_shape`。
- `weight_packed`：safetensors `I32 [out, in/8]`，pack 沿 in 维 little-endian，element k 占 int32 bits `[4k, 4k+4)`。compressed-tensors 官方 `pack_to_int32` 接收 **signed** int4，落盘是 **offset-binary** nibble（`value + 8`）；dequant 必须 `signed = unsigned - 8`。view 成 bytes 后每 byte 含两个 in_col：**低 nibble = 偶数 in_col，高 nibble = 奇数 in_col**。
- `weight_scale`：BF16 `[out, in/32]`。
- `weight_shape`：`I32 [2]`，**仅落盘存在；运行时不加载到 GPU**（见下文 #234）。
- Per-expert tensor 不预 fuse；W1/W3 各自独立存。EP8 plan 阶段沿 expert 维 stack 成本地 48 个 expert。

### Runtime INT4 path: Marlin WNA16 (the only one)

- 运行时 routed expert backend 是 vLLM Marlin WNA16（INT4 + BF16 scale，`group_size=32`），按 H20 vLLM `0.19.0` ABI 接（`moe_wna16_marlin_gemm` 无 `is_ep`）。
- Marlin weight repack（`gptq_marlin_moe_repack`，no-actorder）：checkpoint offset-binary `[expert,out,K/8] int32` → Marlin `uint4b8` bias=8 `[expert,K/16,N*2] int32`，总字节不变，**不做 `xor 0x88`**（保留 unsigned nibble）。
- Marlin scale permute（`marlin_moe_permute_scales`）：checkpoint `[expert,out,in_group]` → group-major + 64-block `scale_perm` 的 `[expert,in_group,out]`。
- W13 必须在 load/package 阶段 fuse 成 `gate_then_up`（vLLM runtime ABI 不吃独立 gate/up）：fused W13 int32 view `[48,448,8192]`，scale `[48,224,4096]`；W2 packed `[48,128,14336]`，scale `[48,64,7168]`。常驻 package 是 fused W13 + W2，gate/up 只是 load-time 临时 buffer。
- 关键修复（Marlin atomic split-K）：vLLM `fused_marlin_moe` 对 W13/W2 都用 `use_atomic_add=False, use_fp32_reduce=True`，走 global F32 `c_tmp` 累加。PegaInfer 早期固定 `use_atomic_add=true` 且不传 `c_tmp`，split-K>1 时 BF16 `atomicAdd` 写 C，累加顺序非确定 → row-state 发散。修复为预分配 `c_tmp` F32 + 关 atomic。H20 单层 W13/route_output/final 对 vLLM reference `max_diff=0 / mean_diff=0`（real K2.5 rank0 layer1）。

### Router scale placement

vLLM `grouped_topk` 返回 **未乘** `routed_scaling_factor` 的 normalized topk weights；`DeepseekV2MoE.forward` 在 routed expert 总输出后整体乘 `2.827`。PegaInfer 早期把 `2.827` 提前乘进 router topk weight 再喂 W2，rounding boundary 与 vLLM 不一致 → 已改为 router 输出 unscaled weights，routed F32 sum/reduce 后整体乘 scale。

### Tokenizer / prompt contract

- `tiktoken.model` + `tokenizer_config.json` special tokens。
- chat template（`chat_template.jinja`）text-only：`<|im_system|>`/`<|im_user|>`/`<|im_assistant|>...<|im_middle|>`；assistant generation prompt 以 `<think>` 开始，instant mode 为 `<think></think>`，preserve-thinking 保留 `reasoning`/`reasoning_content` suffix。
- thinking 默认开启；README 推荐 thinking `temperature=1.0`、instant `temperature=0.6`、`top_p=0.95`。image/video content 显式拒绝。

## Removed / superseded (tombstones)

- **Expert-major INT4 / CUTLASS example69 path — removed in #234.** Bring-up first built a CUTLASS example69 (Hopper INT4×BF16 grouped GEMM) probe as the routed-expert backend. A focused H20 probe proved it could not express Kimi's `group_size=32` per-K-group scale: example69 reloads scale on a 64-wide K tile (`TileShapeK=64`), so col `32/33` reused group0 scale and col `64` used group1 scale; `TileShapeK=32` hits CUTLASS static assert `K_BLOCK_MAX >= 4`. The path was demoted to a limitation probe and then deleted in #234 — the CUTLASS-era projection kernels/probe (`weight_packed_cutlass_example69`, `weight_shape` tensor loading, the example69 launcher and FFI) are gone. Marlin WNA16 is the only runtime INT4 path. `KimiExpertMajorProjectionPlan` (`pegainfer-kimi-k2/src/weights/package.rs`) remains **live** — it describes the EP weight layout, not the dead CUTLASS kernel. `KimiExpertMajorRoute` outlived its callers (DeepEP routing replaced it) and was deleted in the post-#298 dead-code sweep.
- **`weight_shape` GPU load — removed in #234.** It was loaded for 60 MoE layers × 384 experts × 3 projections, validated to `[2]`, then never consumed by any kernel (dims come from manifest constants). Dropping it removes **8,640 tensors** from the load set (`pegainfer-kimi-k2/src/weights/tests.rs` asserts the count). The checkpoint still carries `weight_shape` on disk; the runtime simply no longer reads it.
- **`KIMI_RUNNER_MAX_BATCH = 4` hard-cap — superseded.** Bring-up locked decode at a fixed bs4 wave. The const is now `64` (`pegainfer-kimi-k2/src/runner/scheduler.rs`), with worker decode arenas bucketed `[1, 2, 4, 8, 16, 32, 64]` (`KIMI_DECODE_BATCH_BUCKETS`, `worker.rs`) and per-request cap `KIMI_MAX_REQUEST_TOKENS = 8192` (DP prompt cap `PPLX_MAX_DISPATCH_TOKENS = 2048`). Changing the cap is not a one-const edit: it ties arena count, every `decode_batch_size`-shaped scratch/router/Marlin shape, and per-bucket CUDA-graph capture.
- **`kimi-k2-pplx-ep` cargo feature + `PEGAINFER_KIMI_PARALLEL` env — removed.** Parallel shape and EP backend are now CLI flags: `--tp-size/--dp-size/--ep-backend`. The feature is just `kimi-k2`. Active line: `--tp-size 1 --dp-size 8 --ep-backend pplx`.
- **Internal H20 smoke/candidate/debug test entries — removed.** Direct worker/scheduler no longer carries `forward_prompt_smoke`, `ForwardOneTokenSmoke`, full-decode smoke, row-diff D2H instrumentation, or candidate-dump tests; only CPU unit tests (placement, page metadata) remain. Progress is gated end-to-end through `pegainfer-server` / `bench_serving` / OpenAI `/v1/completions`.

## Chronology (decision records)

The bring-up ran ~2026-05-20 to 2026-05-22 on an 8×H200 node against a vLLM `0.19.0` K2.5 reference. Greedy token-id parity vs a vLLM fixture was the gate throughout.

1. **HF probe + manifest.** Pulled non-weight files; confirmed text core is `DeepseekV3ForCausalLM`-style, only routed experts pack-quantized. Built a text-only safetensors index manifest and TP8/EP8 rank ownership (attention/dense/shared/lm_head TP-sharded; routed experts EP-sharded 48/rank; router replicated).
2. **Sliced loader + typed GPU view.** `load_rank_sliced_weights_to_gpu` does vocab row-slice, attention head row/col-slice, dense/shared MLP row/col-slice, local-48-expert enumeration — avoids each TP rank copying full BF16 tensors. Typed view enforces a dtype barrier (BF16 backbone, F32 router bias, INT4 routed triple). Caught at this stage: `weight_packed`/`weight_shape` are safetensors `I32`, not the catalog's earlier `U8/U32` guess — a loader dtype-contract issue, not a model difference.
3. **MLA + router + Marlin.** MLA prefill via FlashInfer `SinglePrefillWithKVCacheDispatched<192,128>`; YARN RoPE; `kimi_router_noaux_tc` (sigmoid + group top-k matching DeepSeekV3 semantics); Marlin WNA16 routed experts after the CUTLASS example69 dead end (#234, see tombstone). Marlin route alignment metadata (`kimi_moe_marlin_align_block_size`) is device-resident and ignores non-local experts.
4. **Row-state debug → Marlin atomic fix.** A fixed 4-concurrency `max_tokens=16` fixture intermittently diverged at row 1. Per-phase device first-diff narrowed the first dirty phase to layer1 routed local (`moe_routed_local`, after `sum_topk_rows_f32`, before NCCL). Root cause: the Marlin BF16 atomic split-K (see load-bearing facts). After the `c_tmp` + global-reduce fix, the fixed bs4 output16 gate showed `ROUTER/ROUTE_ROW/ROW diff = 0`.
5. **Decode diagnostic cleanup + RS bridge.** Removed the row-diff D2H, restored the F32 collective to a single contiguous all-reduce (the per-row loop was a diagnostic bridge that ×4'd each collective at bs4), stopped the decode CPU barrier (kept only the first-collective barrier — H20's first NCCL call has an independent stream-drain stability issue). Routed MoE combine changed to `local router/Marlin → repeat_f32_for_reduce_scatter → NCCL reduce_scatter` (no BF16 all-gather, no EP-world compute blowup). Warm output64 ~`114 → 144 tok/s`.
6. **bs4 wave decode + scratch pre-allocation.** Scheduler ran up to 4 requests per wave, slot-local prompt forward then `forward_decode_batch_next_tokens` from token 2. All decode scratch (dense MLP, shared expert, router, Marlin route/workspace, W13/W2 output, routed F32 reduce, top1) moved into a worker-owned arena. **Reuse requires device zero-fill** of Marlin locks/W13/W2 output/routed buffers: stale values on non-local/padding route rows caused divergence past token 2. Decode arenas went from a fixed bs1/bs4 split to `1..=4` selected by real wave size; `bs==1` special-casing was banned (a bs1 no-barrier experiment regressed correctness and masked rank/stream state).
7. **Full-segment CUDA Graph.** First capture hung at `max_tokens=2`/4-concurrency. `kimi_graph_probe` proved local kernel / cuBLAS / NCCL all-reduce / NCCL reduce-scatter each capture+replay fine; the hang was that 8 rank workers begin/end/launch independently with no cross-rank phase alignment. Fix: `CudaGraphState` sync phase hook + rank barriers around graph begin/enqueue/end/launch. `cuGraphLaunch count = 8 ranks × decode steps` confirmed replay. Steady TPOT collapsed from the ~35ms strong-sync step toward ~14ms (host-enqueue fold, not faster kernels).
8. **Kernel fusions (kept).** fused qkv_a (q/kv down-proj → one GEMM + split, `1947 → 1886` static calls); shared + dense gate/up vstack fusion; `kimi_scaled_add_f32_bf16_to_bf16` (scale + F32→BF16 residual add, parity-safe by rounding the contribution to BF16 first); removed redundant routed-sum memset and Marlin locks memset (workspace already zero / `barrier_release` resets). Synthetic output64 steady TPOT settled at `14.39ms` (avg) / `14.83ms` (p99). This was the TP8+EP8 keep/revert gate.
9. **vLLM top-20 greedy gate.** Multi-prompt (`hello/math_short/self_intro_zh/code_rust`) 4/4 argmax match, top-20 id overlap min `19/20`, via `compare_vllm_topk_fixture.py --require-argmax --min-overlap 16`. This was the bring-up accuracy bar; the current git-versioned per-position logprob gate is #223 (see [accuracy-gate.md](accuracy-gate.md)).

## Rejected approaches (don't repeat)

- **BF16 bulk hidden all-reduce** instead of the F32 bridge — breaks greedy parity, no perf win. (Lesson 1 in [docs/lessons/kimi-bringup-numerics.md](../../lessons/kimi-bringup-numerics.md).)
- **Merging shared + routed reduce** into one F32 all-reduce per layer — breaks cold-batch greedy and contaminates later requests. (Lesson 2, same file.)
- **vLLM TP-only MoE final all-reduce cadence** (`123` BF16 all-reduce, `0` RS): two variants both kept correctness but regressed TPOT/p99 vs the RS bridge. A microbench showed `repeat+RS` ≈ direct F32 all-reduce (ratio `0.983x`), so a single-point "match vLLM cadence" swap buys nothing — fewer collectives must come from real EP dispatch/combine.
- **Row-wise F32 collectives** (per-active-row all-reduce) — stabilized the short gate but did not remove the row-state diff and was slower; diagnostic bridge only.
- **CUTLASS example69 as routed backend** — see #234 tombstone.

## Reference tooling (off-repo fixtures)

- `pegainfer-kernels/tools/kimi_k2/hf_logits_reference.py` — HF raw full-logits reference (trust_remote_code + vision-tower stub; INT4-only reference, slow run_compressed load).
- `pegainfer-kernels/tools/kimi_k2/vllm_logits_reference.py` — vLLM serving top-logprob fixture (vLLM 0.19.0 caps sample `logprobs` at 20, so the bring-up gate used top-20). Supports `--prompt-set-json` for batched multi-prompt cases.
- `pegainfer-kernels/tools/kimi_k2/vllm_marlin_wna16_reference.py` — vLLM Marlin W13 / W2 / final BF16 reference; `--model-path ... --layer-idx 1 --rank 0` reads the real checkpoint's rank-local experts.
- `pegainfer-kernels/tools/kimi_k2/compare_vllm_topk_fixture.py` / `compare_logits_fixture.py` — candidate comparison (argmax / top-k overlap / logits diff).
- `pegainfer-kernels/tools/kimi_k2/torch_reference.py` — compressed-tensors official pack/dequant, bit-exact INT4 single-expert fixture (self-check `0-diff`).

A strict bit-level `h20_kimi_marlin_wna16_single_layer_matches_vllm_reference` gate is kept `#[ignore]` (red by design): vLLM Marlin's W2 atomic split-K accumulation order gives `route_output max_diff=96 / mean_diff=1.86` at BF16 magnitude ~7000 (< 0.03% relative, ~1.5 ULP) — not an algorithm bug. Turning it green requires either a `use_fp32_reduce` fixture or a ULP-relative tolerance.
