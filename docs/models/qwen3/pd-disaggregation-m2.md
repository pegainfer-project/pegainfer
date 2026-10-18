# P/D 分离 M2：pegaflow metaserver P2P 数据面

> **TL;DR**: Qwen3-8B 1P+1D 双 pegainfer 实例 P/D 分离**已在单机 2×H200（每卡 1 块 400G IB NIC）端到端验证**：KV 经 pegaflow 内容寻址 P2P 从 P 流向 D（metaserver 发现 + 单边 RDMA READ + H2D restore），greedy 输出与单实例 baseline 逐 token 一致（3 档 prompt 长度），33/33 块 74.2 MiB 拉取 rdma_wait 仅 2.6ms，杀 metaserver / 杀 P 均优雅退化为本地 prefill。无 handle 协议——D 从同一 prompt 推出同一组 kvbm lineage hash 直接查询。**多轮并发压测已过**：turn2+ TTFT 恒定 ~107ms、TPOT p99 全轮 <10.1ms，与 mixed 部署的完整 A/B 见 `../../benchmarks/qwen3-8b-pd-vs-mix-h200.md`；先修掉 §4 的 `max_completion_tokens` 坑。pegainfer 分支 `feat/pd-pegaflow-p2p`，pegaflow 侧 PR [#381](https://github.com/novitalabs/pegaflow/pull/381)。
>
> Last touched: 2026-07

## 1. 架构

```
            pegaflow-router (:9299, 同步流程)
           /                                  \
   pegainfer-P (GPU0, :9200)          pegainfer-D (GPU1, :9201)
   embed PegaEngine                    embed PegaEngine
   + P2pTransferService (:51100)       + P2pTransferService (:51101)
   + flush-on-finish                   + RemoteFetch prefetch phase
           \                                  /
            pegaflow-metaserver (:51056, hash→owner 目录)
```

- **控制面**：router 收到请求 → 改 `max_tokens=1` 发 P → P prefill + 1 步 decode，该 step 的 `Finished` 事件被扣住，等 save+metaserver 注册对 peer 可见后由 offload runtime 异步释放（`flush_on_finish`，scheduler 线程不等屏障）→ P 的 HTTP 响应即 KV-ready 信号 → router 原样转发原请求给 D。
- **数据面**：D 的 `begin_kv_prefetch` 本地 miss → pegaflow 问 metaserver 谁有最长前缀 → gRPC 握手 + `QueryBlocksForTransfer`（owner 侧 transfer lock 防 evict）→ 单边 RDMA READ 拉进本地 pinned pool → `QueryOutcome::Loading` 期间请求 park 在 scheduler `loading` 队列逐 tick 重查询 → `Ready` 后走既有 H2D prefetch 路径 → suffix-only prefill。
- **无 handle**：kvbm `SequenceHash` 由 token 序列确定性推导，P/D 同 commit 必然同 key。namespace 含 `hidden_size/intermediate_size/vocab_size + KV 几何`——4B 和 8B 几何相同（36L/8H/128d）、tokenizer 相同，纯几何 namespace 会静默交叉命中（toxic review 抓出）。

## 2. 代码落点

| 仓 | 分支/PR | 内容 |
| --- | --- | --- |
| pegaflow | PR #381 | `MetaServerClient::flush()` 屏障（送达或丢弃语义）+ `PegaEngine::flush_saves_and_registrations` + `P2pTransferService`（3 个 P2P RPC + Health 的最小嵌入服务面）+ `logging::init` 改 `try_apply`（宿主已装 logger 时不再 panic——库嵌入形态才触发）+ cudarc 0.19.7 floor |
| pegainfer | `feat/pd-pegaflow-p2p` | kv-offload：`P2pConfig` + 嵌入 tonic 服务（bind 先行 fail-fast）+ 60s GC（transfer lock + stale prefetch 双清扫）+ `QueryOutcome::Loading` + flush 5s deadline（`flush_saves_then` 异步屏障，`Finished` 延迟释放，scheduler 不阻塞）；qwen3：prefetch 三相状态机（`RemoteFetch`/`Loading`/`Committed`，15s 超时退化）+ `reserve_floor` 穿透重查询路径 + `flush_on_finish`；server：`--kv-p2p-{metaserver-addr,advertise-addr,nics,flush-on-finish}` |

pegaflow #381 已合入 master（squash 为 `d46fd16`，含 router `max_completion_tokens` 修复）；全局 `pegaflow-core` pin 后续已推进到包含 #381 与 #395 的 `1473c53`。Qwen3 跨引擎 P/D 只支持已实测的 vLLM 0.23.0 默认 NHD 配置：KV 是 block-first、单 segment，与 PegaInfer page 对齐，因此不需要 #382；其他版本/layout 不在本 PR 范围。

## 3. 验收数据（单机 2×H200，2026-07）

| 门 | 结果 |
| --- | --- |
| 1. 正确性 | 3 档 prompt（<1 block / 跨块 / ~600 tok），temp=0，router P/D vs 直连 D baseline **逐字节一致** |
| 2. P2P 实证 | D 侧 `RDMA fetch summary: blocks=33/33 bytes_mib=74.2`；stages: connect 53ms（首次握手，后续 0）+ query 43ms + **rdma_wait 2.6ms**（连接复用后整体 3.4ms, 21.7 GiB/s）；P 侧 `P2P rdma_handshake accepted` |
| 3. 故障退化 | 杀 metaserver → 请求正常完成（`MetaServer query failed` WARN 后本地 prefill）；杀 P → D 冷请求正常完成；无 crash 无 hang |
| 4. TTFT 记录 | 长 prompt(~600 tok)+64 out：P/D 冷 534ms（P prefill 43ms + D 拉取/重建/suffix ~100ms + decode）vs P/D 暖 379ms vs D 本地冷 384ms。**单机上 P/D 冷路径多付 ~150ms**（首次握手 53ms 是一次性的）；M2 无性能目标，M3 layer-wise push 才是 TTFT 优化 |
| 5. 多轮并发 vs mixed | vllm-bench openai-chat，4k prompt + 1k/turn ×5 turns，20 会话并发 10，temp=0：P/D turn2+ TTFT 恒定 **105-111ms**、TPOT p99 <10.1ms；同卡数 mixed×2（会话亲和 LB）TTFT 随轮爬升 71→132ms、TPOT p99 10-13ms；总吞吐持平。P 前缀缓存逐 turn 全量命中（cached=7200/8240 @turn5），D 每 turn 只拉新增 suffix（64 块 ~15ms）。完整表格 + 压测命令：`../../benchmarks/qwen3-8b-pd-vs-mix-h200.md` |

## 4. 已知边界 / 下一步

- **router 必须同时钳 `max_tokens` 和 `max_completion_tokens`**（pegaflow `283c451` 修复）。chat 客户端（vllm-bench openai-chat、新版 OpenAI SDK）发的是 `max_completion_tokens`，engine 两者并存时优先后者——漏掉它 P 会做完整 decode，多轮 TTFT 从 ~110ms 劣化到 ~1.5s/turn，且症状极具迷惑性（GPU 满载、看似 prefill 慢/缓存失效）。诊断路径：P 日志 `output_tokens` 分布一眼定罪。
- **RemoteFetch 状态机缺单测**（超时 / drop-during-fetch / zero-hit 三用例)——需 GPU 环境，欠账在此记录。
- **P2P/RDMA 依赖未做 feature gate**（pegainfer #523）：`rdma` feature 无条件开，默认构建也拉 pegaflow-transfer + vendored rdma-core；运行时无影响（不带 `--kv-p2p-*` 不激活），是打包卫生欠账。
- P 侧冷 prompt 多付一轮 RemoteFetch 往返（本地全 miss 先 `Loading` 再空手 prefill）——设计使然。
- 单机验证 ≠ 跨机：跨机需确认 dma-buf/GID/路由；目标集群 GPU↔NIC 同构（8×400G 1:1 PIX）预期直接成立。
- 多 P 多 D 纯 router 事务（内容寻址保证任意 D 发现任意 P 的 KV），M2 架构无障碍。
- prefill-only 请求模式（省掉 max_tokens=1 的一步 decode）：`PendingEffect::EmitAndFinish` 缝上加,未做。
- M3（延后）：pd-rdma-push 逐层 GPU→GPU WRITE Rust 化,P prefill 与传输流水。

## 5. 复跑

内部部署机（位置见团队记录）上的三个脚本：`pd_stack.sh start|stop|status`（起 metaserver + P + D + router 全栈，端口避开同机生产 pegaflow）、`pd_accept.sh baseline|pd|compare|evidence`（**顺序**：baseline 后必须重启栈清缓存再跑 pd，否则 D 本地命中不走 P2P）、`pd_gate4.sh`（TTFT 分解）。多轮压测命令与 mixed 基线做法（会话亲和 LB）见 `../../benchmarks/qwen3-8b-pd-vs-mix-h200.md` §1-2。
