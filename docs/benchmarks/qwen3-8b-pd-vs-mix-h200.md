# Qwen3-8B P/D 分离 vs mixed 部署（2×H200，多轮对话负载）

> **TL;DR**: 同一多轮负载下三种部署 A/B（2026-07-03）：2 卡 P/D（1P+1D）与 2 卡 mixed（会话亲和 LB）总吞吐持平（47.8k vs 47.0k tok/s），P/D 的收益在 **decode 稳定性**——TPOT p99 10.08ms vs 12.77ms（-21%），且 turn2+ TTFT 不随轮数增长（~107ms 恒定 vs mixed 71→132ms 爬升）；代价是冷 turn1 TTFT 多 ~200ms（P→D 串行，M3 layer-wise push 的目标）。压测命令与坑（`max_completion_tokens`）见下。

环境：单机 2×NVIDIA H200，每 GPU 配一块 400G IB NIC（P2P KV 走 RDMA READ），Qwen3-8B bf16，`--kv-offload --kv-offload-host-gib 32` + pegaflow P2P mesh。pegainfer 分支 `feat/pd-pegaflow-p2p`，pegaflow 含 router `max_completion_tokens` 修复（`283c451`）。

## 1. 负载与压测方法

[vllm-bench](https://github.com/vllm-project/vllm-bench) 多轮对话，20 个会话、并发 10、每会话 5 轮，首轮 4096 tokens、每轮追加 1024 tokens 输入、每轮输出 128 tokens，greedy：

```bash
vllm-bench \
  --backend openai-chat \
  --base-url http://<endpoint> \
  --model <model-path> --tokenizer <model-path> \
  --dataset-name random \
  --multi-turn --multi-turn-num-turns 5 \
  --random-input-len 4096 --per-turn-input-len 1024 --random-output-len 128 \
  --num-prompts 20 --multi-turn-concurrency 10 \
  --extra-body '{"min_tokens":1}' \
  --temperature 0
```

**协议**：每组配置测前重启整栈（清 HBM 前缀缓存 + host tier + metaserver 目录），冷态起跑。三组共用同一对 GPU、同一构建。

**坑（必读）**：`openai-chat` backend 发送的是 `max_completion_tokens`（OpenAI chat 已弃用 `max_tokens`），engine 两者并存时优先前者。P/D router 若只钳 `max_tokens=1`，P 会做完整 decode——症状是 P 侧 GPU 满载、TTFT 秒级、看似 prefix cache 失效。诊断：看 P 日志 `output_tokens` 分布；修复见 pegaflow `283c451`。

## 2. 三组配置

| 配置 | 卡数 | 拓扑 |
|---|---|---|
| mixed ×1 | 1 | 单实例直连（prefill+decode 同实例） |
| mixed ×2 | 2 | 两个独立 mixed 实例 + 会话亲和轮询 LB（首条消息 hash 定实例，多轮保持前缀缓存局部性） |
| P/D 1P+1D | 2 | pegaflow-router：请求先发 P（`max_tokens=1`，flush-on-finish 屏障后返回 = KV-ready 信号）再发 D；D 经 metaserver 发现 + 单边 RDMA READ 拉 P 的 KV |

## 3. 结果

Overall（100 请求）：

| 指标 | mixed ×1 | mixed ×2 | P/D 1P+1D |
|---|---|---|---|
| TTFT mean / p99 (ms) | 159 / 1245 | 160 / 702 | 204 / 1293 |
| TPOT mean / p99 (ms) | 12.31 / 15.33 | 9.38 / 12.77 | **8.97 / 10.08** |
| 总吞吐 (tok/s) | 38,316 | 47,028 | **47,753** |
| E2E mean (ms) | 1722 | 1351 | **1343** |

分轮 TTFT mean（ms）：

| Turn | mixed ×1 | mixed ×2 | P/D |
|---|---|---|---|
| 1（冷，4096 tok） | 501 | 394 | 590 |
| 2 | 67 | 71 | 108 |
| 3 | 71 | 90 | 106 |
| 4 | 78 | 115 | 105 |
| 5 | 81 | 132 | 111 |

分轮 TPOT p99（ms）：

| Turn | mixed ×1 | mixed ×2 | P/D |
|---|---|---|---|
| 1 | 15.35 | 12.90 | **8.42** |
| 2 | 11.29 | 10.12 | **8.86** |
| 3 | 11.97 | 11.59 | **9.40** |
| 4 | 12.77 | 12.71 | **9.93** |
| 5 | 13.03 | 10.94 | **10.08** |

## 4. 读法

- **P/D 买到的是 decode 隔离**：D 上没有并发 prefill 打断 decode step，TPOT p99 全轮压在 <10.1ms；mixed 两组都在 11-15ms 波动（其它会话的 suffix prefill 挤进 unified step）。负载越重、prompt 越长，这个差距越大——本组 10 并发只是温和负载。
- **mixed 的 TTFT 随轮数爬升**（71→132ms）：turn 越深 suffix prefill 与在跑 decode 的互相干扰越强；P/D 的 turn2+ TTFT 恒定 ~107ms（P suffix prefill ~60ms + router 一跳 + D 侧 RDMA 拉新增块 ~15ms + suffix 重建）。
- **P/D 冷 turn1 多付 ~200ms**（590 vs 394）：P 全量 prefill → flush 屏障 → D 拉 64 块×16 tokens/块 → suffix。这是 M2 的已知代价，M3 layer-wise push（P prefill 与传输流水化）针对它。
- **总吞吐持平**：P/D 没有吞吐税（P 的算力换来了 D 的纯净 decode）。
- mixed ×1 行不是同卡数对比（1 卡 vs 2 卡），仅作单实例基线。

## 5. 关联

- P/D 架构与验收记录：`../models/qwen3/pd-disaggregation-m2.md`
- 混合负载 ITL 干扰的前期定量：`mixed-load-itl.md`
