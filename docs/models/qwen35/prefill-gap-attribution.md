# Qwen3.5-4B prefill 吞吐归因：H20 上对 vLLM 0.29 的混合到达差距

**Last touched:** 2026-11

**TL;DR:** H20 单卡上，qps16 相对 vLLM 的时差全部落在 prefill：同一窗口的 decode 已与 vLLM 持平（2.14 对 2.08 s），prefill 是 5.07 对 3.81 s，折合每 token 82.8 µs 对 58.1-60.8 µs。GEMM 跑到 61% MFU（vLLM 端到端整体 51%），所以 1.33 倍差距分散在 GDN（15.6%）、逐元素与 attention 内核（11%）、launch 间隙（10.5%）上，单点修改各 1-3%。一步内的请求已经改成各占一条 stream 并行，拿回 6.0% 的 prefill 吞吐（第 7 节）；剩下的结构性修复是把一个 chunk 的 token 合成一次 prefill pass。

## 1. 基准契约

| 项目 | 值 |
| --- | --- |
| 模型 | Qwen3.5-4B，bf16 |
| 代码树 | 提交 `9ef13bc5`，`--features qwen35`，nvcc 目标 sm_90 |
| 机器 | H20 96 GB，单卡 |
| 服务端 | `target/release/pegainfer --model-path … --port 18099 --max-batch 64`（默认姿态） |
| 对照 | vLLM 0.29.0（系统 `python3`），`serve … --language-model-only --max-num-seqs 64 --max-model-len 16384 --gpu-memory-utilization 0.90` |
| 客户端 | `vllm bench serve --backend openai --dataset-name random --random-range-ratio 0 --temperature 0 --ignore-eos --seed 42 --tokenizer …` |
| 单元 | `pfonly` 1024 in/1 out/64 prompts/c32；`c32` 1024/128/64/c32；`qps16` 1024/128/64/c32 加 `--request-rate 16` |
| 通过条件 | `completed == num_prompts && failed == 0` |
| 原始产物 | 每步分解来自 `PEGAINFER_ITL_DEBUG=1` 的服务端日志；内核表来自 `nsys profile --trace=cuda` 的 `cuda_gpu_kern_sum`；客户端结果来自 `vllm bench serve --save-result` 的 JSON |

## 2. 时间花在哪里

qps16 与 c32 两个单元里，PegaInfer 与 vLLM 都是「prefill 时间 + decode 时间」顺序相加，没有重叠：

| 单元 | 引擎 | 总时长 | prefill | decode |
| --- | --- | --- | --- | --- |
| qps16 | PegaInfer 默认 | 7.35 s | 5072 ms（61614 token） | 2143 ms（276 步 × 8.26 ms @ n=32） |
| qps16 | vLLM 0.29 | 5.89 s | 3810 ms | ≈2080 ms（由总时长减 prefill 推得） |
| c32 | PegaInfer 默认 | 7.19 s | 5072 ms | 2120 ms |
| c32 | vLLM 0.29 | 5.57 s | 3810 ms | 1760 ms |

qps16 的 decode 部分两边基本持平，差距全在 prefill 的 1.33 倍；c32 下 decode 也比 vLLM 慢约 20%，但绝对差仍以 prefill 为主。prefill 侧的对照单元 `pfonly`（1024 in/1 out，64 prompts，c32）：PegaInfer 5.07-5.24 s（12.2-12.6 req/s，约 12.5k prompt token/s），vLLM 3.81 s（16.8 req/s，约 17.2k prompt token/s）。

## 3. prefill 每 token 的预算

`pfonly` 一次 5189 ms / 62638 token = 82.8 µs/token；79 个 request-pass（64 个 prompt 中 15 个跨两个 pass）。按内核家族折算到每 token：

| 类别 | µs/token | 占 prefill 墙钟 |
| --- | --- | --- |
| cuBLAS GEMM（nvjet） | 48.5 | 59% |
| GDN chunkwise（6 个 Triton 内核） | 11.8 | 14% |
| conv1d、silu_mul、rms_norm、add 等逐元素 | 7.1 | 9% |
| flashinfer paged prefill attention | 1.3 | 2% |
| 内核之间的 launch 间隙 | 7.9 | 10% |
| 设备侧 memset | 2.4 | 3% |

GDR 家族内部：`gdr_chunk_o_qwen35_kernel` 382.6 ms/1896 次（201.8 µs，24 层 × 79 pass 的主项），`gdr_recompute_w_u` 94.8 ms、`gdr_prepare_qkv_gbeta` 92.1 ms、`gdr_chunk_state` 69.3 ms、`gdr_solve_tril_64` 65.7 ms、`gdr_chunk_scaled_dot_kkt` 34.5 ms。主机侧每次 pass 约 583 次 `cuMemAllocAsync`/`cuMemFreeAsync`（prefill 的每个中间张量都是新分配），全程 46k 次，另有 152 ms 的 memset。

trace 内 GPU 忙碌率 90.5%（4728 ms busy / 5225 ms span），空闲里 70 个 1-5 ms 的间隙共 147.6 ms，基本都是 pass 边界（host 换 pass 约 2 ms），其余是内核之间的 3-4 µs 小间隙。

## 4. 为什么不是 GEMM

每 token ≈ 4.37 GFLOP（`hidden_size` 2560；24 个线性层每层 in_proj 20.97M、o_proj 10.49M、MLP 47.2M MAC，加 4 个 full-attn 层）。1024 token = 4.47 TFLOP，H20 BF16 峰值 148 TFLOPS 对应 30 ms，实测 78 ms/pass，即端到端 39% MFU；其中 GEMM 的 48.5 µs/token 折合 61% MFU。vLLM 端到端 58.1-60.8 µs/token 相当于 51% MFU，它的 GEMM 不可能比我们快出量级。

给 prefill 做 cublasLt plan 调优也不成立：`pegainfer-kernels/csrc/shared/linear.cu:611-641` 的 plan 表以精确 `(M,N,K)` 为键，除 1024 以外的 prompt 长度都会落回 `GEMM_LT_UNTUNED`，调出来的只会是基准形状本身。

## 5. 负结果

三处都在 H20 上实测过，都不作为默认值：

1. **`gdr_chunk_o` 调块**：BLOCK_V 32→64（grid 第一维 4→2）、`num_warps` 4→8，重建后 `pfonly` 5.23/5.08 s，与基线 5.24/5.07 s 一致（二进制尺寸变化确认重建生效）。分块形状不是这个内核的瓶颈。
2. **`--qwen35-scheduler-policy auto`**：qps16 上 TPOT 17.94/17.62 ms、ITL p99 97.0/89.8 ms（默认 20.58、326.6），但 TTFT 1139.9/975.0 ms（默认 534）、吞吐 −4%。它把每步冻结从 200 ms 缩到 78 ms，同时把 23 个 unified 步变成约 60 个，每步多付一次 decode 与 host 间隙。
3. **`--max-prefill-tokens 8192`**：prefill 吞吐 +3%（12.22→12.62 req/s），qps16 TPOT 19.04 ms（默认 20.58），但 ITL p99 528 ms（默认 337）。步更大、冻结更长。

## 6. 按收益排序的后续项

1. **一步内把一个 chunk 的请求合成一次 pass（结构性，最大）**：`pegainfer-qwen35/src/unified_forward.rs` 的注释写明 GDR chunkwise 没有 batched prefill 内核，投影与 MLP 仍按请求各跑一遍，M=1024 的 GEMM 只到 61% MFU。把各请求的 token 拼成一次 pass 跑投影与 MLP，只把 attention/GDN 按请求切分，能同时消掉重复读权重与 launch 间隙；这是唯一能追平 1.33 倍的项。
2. **GDN prefill 内核（739 ms，15.6%）**：`gdr_chunk_o` 每层 pass 202 µs，是 GDR 家族的 52%。
3. **prefill scratch 复用**：583 次分配/pass 与 152 ms memset，`pegainfer-qwen35/src/prefill.rs` 每层的 `HiddenStates::zeros` 与每 chunk 的 `GdrChunkwiseScratch35::new` 都可以提到 pass 之外。
4. **launch 间隙（约 8 µs/token）**：逐元素内核随之合批后自然收敛。

前三项各自 1-3%，合起来才有机会追上 1.33 倍；只改一项不会改变 qps16 的结论。

## 7. 一轮改动：一步内请求并行

`pegainfer-qwen35/src/unified_forward.rs` 现在让一个 chunk 的请求各占一条 stream 跑 prefill，跑完再汇回 context stream。串行时单条请求的 28 层依赖链在两次核函数之间留下空隙、最后一次 GEMM 还有尾波，四个独立请求并行正好把这些填上。同一次会话里串行与并行两份二进制背靠背各跑一遍（H20 单卡，vLLM 0.29 客户端）：

| 单元 | 串行 | 并行 | 变化 |
| --- | --- | --- | --- |
| prefill-only，64 prompts，c32 | 5.27 / 5.11 / 5.08 s，12.14 / 12.52 / 12.61 req/s | 4.98 / 4.80 / 4.82 s，12.86 / 13.35 / 13.27 req/s | 时长 −5.7%，prompts/s +6.0% |
| qps16，1024 in / 128 out | 1121 / 1126 tok/s，TPOT 20.29 / 20.23，ITL p99 335.9 / 321.9 | 1141 / 1169 tok/s，TPOT 19.90 / 19.80，ITL p99 326.6 / 297.6 | tok/s +2.8%，TPOT −2.0% |
| c32，1024 in / 128 out | 1139 tok/s，TPOT 19.96 | 1152 / 1173 tok/s，TPOT 19.71 / 19.44 | tok/s +2.1%，TPOT −2.3% |

正确性由 `pegainfer-qwen35` 的 lib 测试兜住：`unified_forward::tests::unified_step_decode_matches_graph_decode`（两个请求走并行路径，贪心 token 与 decode graph 路径一致）与 `pad_columns_are_suppressed_on_the_model_logits_paths`（单请求仍走串行路径）。剩下 1.3 倍只能靠第 6 节第 1 项：把 chunk 的请求合进一次 pass，并行流只能填平空隙，改不了 M=1024 的 GEMM 效率。

代码改动单独成提交、叠在 qwen35 分支上；本文只记录测量与归因。

