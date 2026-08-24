# K3 CP lane design — 固定 EP16 上的弹性 CP 与分布式上下文

**TL;DR**（拍板 2026-08-24）：模型拓扑永久固定（TP1 × attn-DP16 × EP16，权重/expert
ownership/graph 不动），**CP 是 per-sequence 的弹性执行 lane**：whale 长 prefill 变成
一个 CP gang（gang 内 BS=1）与其余 local lane（verify/decode/短 filler）在同一个
EP16 superstep 里共存，MoE 天然仍是 EP16 全宽——这取代 mix-engine-design 的独占
CP4×TP4 whale 步型（后者降级为 M1 A/B 的对照臂，TP 从设计中出场）。请求路由看两个
独立维度：**prefill 并行度由 extend_len 决定，context/KV 并行度由 total_context_len
决定**（长 ctx + 短 extend 走 distributed-context MLA decode，不走 CP prefill）。
KDA CP 采用 affine-summary + prefix-merge 的 KCP 算法（FLA PR #691 路线，本地
2026-08-24 实测 KCP4 比同后端 TP4 快 2.70%）；M0 先做 **contiguous（可不等长）切分，
zigzag 缓议**。Baseline = main（CP1 chunked prefill，attn-DP16×EP16）；**不建
TP4-DP2-EP8 baseline，不做 CP8-EP8 部署形态**。阶段：M0 pruned@EP4 算子正确性
（**DONE 2026-08-24：CP4 16k logits gate 绿，零 CUDA 改动，内部墙钟 2980→1026 ms
= 2.90x；HTTP e2e 同脚本实测 CP1 serving 3048 ms 落后 vLLM TP4 1181 ms ~2.6×，
CP4 e2e 预估 ~1100-1150 ms 仅险胜——真 e2e 待 serving 集成**）→ M0.5 CP serving
集成（gang 协调 + KV/state 归拢到 decode home rank）→ M1 full@EP16 交叉矩阵 +
lane-vs-独占步系统 A/B → M2 agent cache 闭环 → M3 弹性调度。

Last touched: 2026-08

## 0. 与 mix-engine-design 的关系

`mix-engine-design.md` 收束了 batch 形态（span 原语、稳态/whale 两步型、kernel 清单、
session KV 常驻），这些**全部保留**。本文取代其中一节：**whale 的并行形态**。

原方案：whale step 独占整个 fleet，瞬态重构成 CP4×TP4（tray 内 TP4 切头、tray 间
CP4 切行），步后回 TP1×DP16。它成立的前提是"whale 必须吃满 16 rank，而纯 CP16 有
FlashKDA 段长悬崖，所以用 TP4 凑"。

Lane 化拆掉了这个前提：whale 是一个 CP2/CP4 gang + 其余 12–14 个 local lane
**同一 superstep 共存**，attention 只需要 gang 宽度的并行；而 MoE——12–16k chunk
下 FLOPs 的大头——因为所有 lane 的 token 都进同一个 EP16 dispatch/combine，
**天然 16 路全宽**。于是 TP4 的最后一个存在理由消失，随之消失的还有双布局权重
loader、head-range kernel 参数化、瞬态重构协议这一整层工程。付出的代价是 §6 的
pre-EP 时间均衡问题——用一个调度问题换掉一个 kernel/权重工程问题。

这同时回答了 mix-engine-design 结尾未拍的设计点（"whale 期间 filler 是否共存"）：
共存，且是结构性的。独占步方案保留为 M1 系统 A/B 的对照臂；若 lane 化在 decode
p99 上输给独占步（预期不会，但要测），再回头。

## 1. 关键修正：两个独立维度

"后续 turn 命中 prefix 后新增 token 很短，就不需要并行"——这句话**只对 KDA 成立**。
KDA 命中后从 `S_cached` 继续递归，开销只随新增 token 数走；但 MLA 的新 query 仍要
读整个历史 latent KV：128k hit + 200 new 意味着 MLA 仍是 200×128k 的 attention，
decode 更是每 token 读全量。所以路由必须拆成两维：

| `L_ext` | `L_ctx` | 执行模式 |
| --- | ---: | --- |
| 短 | 短 | `LOCAL`：普通 lane，home_rank 本地 |
| 长 | 任意 | `CP_EXTEND`：CP gang prefill（有 hit 则从 `S_cached`/prefix KV 起步） |
| 短 | 长 | `DIST_CTX`：home_rank 出 query，striped MLA KV 远程并行扫 + LSE merge |

`DIST_CTX` 的通信量只与 query/head 输出成正比、与 ctx 长度无关；KDA 层没有这条
远程路径（`S_cached` 本地更新即可）。它的 kernel 基础就是 dense 路径的
W-chunked KV 循环 + LSE merge（serving-roadmap 已挂账）——分布式版只是把 merge
跨 rank 做一次。

## 2. 不变量

- **拓扑固定**：EP16、TP1、expert ownership、CUDA graph family 永不因请求变形。
  变形只发生在 sequence 的数据 lane。
- **CP group 预建**：启动时按 buddy 对齐建好 CP2×8 / CP4×4 / CP8×2 / CP16×1 的
  communicator，运行时不动态建组。
- **cp_degree 只在边界变**：turn 边界、prefill chunk 边界、cache commit 边界。
  层间不换布局（层间换布局 = 搬 `[T, hidden]` 激活，正是 KCP 要避免的成本）。
- **CP gang 内 BS=1**；整个 superstep 的全局 BS 不限。多条长请求放进互不重叠的
  多个 gang（M3）。上游 FLA PR #691 同样强制 CP 内 BS=1，这条独立印证。

## 3. KDA CP：KCP 算法与切分决策

算法即 FLA PR #691 / 本地 tray14 实验验证的形态：各 rank 对本段并行算 affine
summary（段效果 = 状态上的仿射映射，包 `[H, K, K+V]`），all-gather 紧凑状态包，
fp32 prefix merge 得到各段真实 `initial_state`，再并行本地 forward。通信量
`CP×H×K×(K+V)`，与 T 无关。同后端 100k 对照：KCP4 8.198ms vs TP4 8.425ms
（−2.70%，四卡效率 71.85% vs 69.91%，归档
`~/bench_results/2026-08-24-k3-kcp4-vs-tp4/`）——线性 recurrence 不需要跨 rank
串行，KDA 不再是反对 CP 的证据。

**切分拍板：M0 做 contiguous（允许不等长），zigzag 缓议。** 理由：

- zigzag 唯一目的是拉平 MLA causal 三角，KDA 是线性的不需要它，却要为它付
  2CP 合成链、双 conv1d halo、每 rank 两段 `initial_state` 的复杂度；
- 上游 PR 也只有 contiguous，zigzag 全部是自研增量；
- MLA 的不平衡可以用**不等长 contiguous 切分**解决：段 i 成本 ∝ prefix·L_i +
  段内三角，给后面的 rank 分更短的段即可拉平，KDA 链保持 CP 长、halo 一组；
- agent 场景长 extend 多带巨大 prefix hit，三角不平衡本来就被 prefix 读稀释。

zigzag 只在"冷启动长 prefill 实测不等长切分仍不平衡超阈值"时再上。

上游 PR 事实清单（对 M0 有约束力的）：contiguous only；CP 内 BS=1 assertion；
`compress_h0`/`expand_h0` 支持 initial_state 链式（`S_cached` 恢复无算法障碍）；
作者自认输出 lossy（fp32 merge 数值路径）——本地实测 rel L2 4.5e-4，所以 **M0
的 logits 门禁必须对 CP1 拿真实容差**，不能只看 cosine；其 76%/86% 加速数字是
H800+32K+训练场景，仅方向性参考。

## 4. MLA：CP prefill 与 distributed-context decode

- **CP prefill**：长 extend 在 gang 内按 §3 的（不等长）contiguous 切分；residual
  stream 整个 forward 保持该布局，MLA/KDA 层间不切换。prefix-hit 时各 rank 需要
  读 prefix latent KV——V1 允许 gang 内临时镜像 prefix 页（实现捷径），canonical
  是 §5 的 striped 布局 + 远程读。
- **DIST_CTX decode/append**：query 在 home_rank 生成 → 广播到 context group →
  各 rank 扫本地 KV 分片出 (local_max, local_sum, local_out) → online-softmax
  结合律 merge → hidden 回 home_rank。每层通信 = 一次 query 广播 + 一次定长
  partial 回收。

## 5. Context cache

每个 agent session 一个逻辑对象：

```rust
struct ContextHandle {
    home_rank: RankId,
    context_group: GroupId,       // 绑定的 buddy group
    mla_block_table: BlockTable,  // LOCAL 或 STRIPED 布局
    kda_state_handle: StateHandle, // 每层 S_cached
    committed_prefix_len: u64,
    commit_epoch: u64,
}
```

- **MLA KV**：短 ctx = LOCAL（全在 home_rank）；长 ctx = **STRIPED**（token 块
  条带分布在 context group）为 canonical 布局。V1 可以用"CP 期间镜像 prefix"
  做捷径，但长存 session 的 `CP × KV` 内存放大会吃掉容量，striped 是终态。
- **KDA state**：home_rank 持权威版本；CP extend 时广播，其他 rank 上是临时
  replica；turn / chunk 边界建不可变 checkpoint。branch 走 CoW（block table +
  state checkpoint 共享，分叉后各自 append）。
- **Commit 协议**：完整 chunk 成功后才原子更新 (prefix_len, block_table,
  kda_state, commit_epoch)；半途失败的 CP chunk 对 radix cache 不可见，杜绝
  "KV 写了一半、state 还是旧版"的污染。

这一节与 mix-engine-design 的"session KV 常驻"（69 层定长 state + 24 层 latent
页 ~5.5GB/session@200k，闲置 offload host）是同一个子系统的两面。

## 6. 调度：优化到达 EP collective 的时间，不是 token 数

`CP4 gang + 12 local lane` 的风险不是数学而是同步气泡：gang attention 8ms、
local lane 2ms，则 12 个 rank 在 all-to-all 前空等 6ms。目标函数：

```text
minimize max_rank(predicted_time_before_EP)
```

不同 token 成本完全不同（1M-ctx decode token ≠ 2k prefill token），不能按 token
计数配平。step planner 每步：decode 延迟预算先留 → 候选长 extend 评
`c ∈ {1,2,4,8}` 的 cost（job time + λ·induced_decode_delay + μ·cache_reshard +
ν·reserved_rank_ms）→ 短 prefill/decode 填其余 rank → 预测各 rank pre-EP 时间、
迭代到偏差可接受。

- 低并发：只有 `T_cp(c) < T_cp(1)` 才扩 CP；GPU 空闲不是扩 CP 的理由，
  允许 idle 胜过负收益并行。
- 高并发：CP degree 收缩，把 rank 还给 local lane，优化 goodput。
- 长请求 aging credit 防饿死。

## 7. Chunk 约束与合法 CP 集合

约束是双边的：

```text
local_chunk_tokens >= amortization_threshold(cp_degree)   # KDA 状态通信摊薄 + kernel 饱和
predicted_step_time <= decode_latency_quantum             # TPOT 保护
global_chunk = cp_degree × local_chunk（不等长时为其和）
```

组件数据的先验（待 M1 矩阵定案）：FlashKDA 段长扫描给出固定开销 ~26µs/call、
**T≈2k 才 90% 饱和**；expert tile 满载要 chunk ≳10.7k（chunk/56 ≥ BLOCK_M 192，
topk=16）。由此推：

- **CP2/CP4 是主力**（chunk 12–16k → 每 rank 3–8k，KDA 94–97% 理想吞吐）；
- **CP8 在边界上**（chunk 16k → 每 rank 2k，正踩饱和线；这也是 zigzag 缓议的
  原因之一——再砍半就掉下悬崖）；
- **CP16 大概率不在合法集合**：需要 chunk ≥32k 才躲开悬崖，多半破坏 decode
  quantum；唯一候选场景是 1M 冷启动这类 TTFT 压倒一切的请求。

若某 degree 下不存在同时满足两约束的 chunk，就降 degree，不硬开。

## 8. Baseline 拍板：不建 TP，不做 EP8

**Phase-1 的交叉不是 TP4 vs CP4，是 `T_cp(c)` vs `T_cp(1)`。** 不建
TP4-DP2-EP8 baseline，三个理由按分量：

1. **组件证据已把 TP 判到输或平**：GEMM 形状扫描（TP 窄切 K=384 → 1180TF、
   kv_a@TP8 447TF vs EP 2380TF）；同后端 KCP4 −2.70%；MLA decode 侧 TP 切头
   不切 latent 读，稳态 TPOT 一分不买。为预期会输的 baseline 从零建 MLA/KDA
   TP + collective + graph 桶是数周量级的 strawman。
2. **TP 的唯一动机本来就是"要个 baseline"，而外部 baseline 让 vLLM/sglang
   出即可**（拍板 2026-08-24）：它们部署 MLA 系模型本来就用 attn-DP + EP 而非
   attn-TP（latent 复制问题相同），所以哪怕请它们出场，出的也不是 TP 形态。
   spec-decode 验收已有 same-checkpoint sglang 对照的先例（`mtp-dspark.md`）。
   内部静态基线 = 现在的 main（TP1×DP16×EP16 + CP1 chunked prefill），免费。
3. "不喜欢 TP 跨机"在 NVL72 上其实不是硬约束（跨 tray ≈ tray 内带宽），但
   无所谓——lane 架构里 TP 根本不出场。

**CP8-EP8 不做为部署/设计形态。** full model @ EP8（112 expert/rank）内存放得下，
但那是没人会部署的中间态，交叉数字不转移。dev ladder：**pruned@EP4（单 tray，
与 full@EP16 同构 56 expert/rank）做 M0 正确性 → full@EP16 做一切性能定案**。
EP8 仅作"只抢得到 2 个 tray"时的 full-model 备胎平台，不进设计。

## 9. 阶段计划

### M0 — 算子正确性（pruned@EP4，单 tray）— DONE 2026-08-24

落地形态与计划的偏差：**FlashKDA 零 CUDA 改动**。KDA 递推对 state 是仿射的
（`S_out = S_in·M + D`，state 布局 `[head, v, k]`，转移作用在 k 轴 = 右乘），
所以 M/D 包用现有 kernel 的"配方操作数"跑出来：`v=0` + identity state → M；
真 v + zero state → D；rank 0 一次真调用即是自己的包。merge 是 per-head fp32
strided-batched GEMM（96×128³，新增 `gemm_strided_batched_f32`）。conv halo 由
接收方把上游 3 行 normed 过 wbig band 投影（bucket 4 + 零 pad 行）。MLA 交换
post-norm latent + rope，绕过 paged gather 直接拼 `mla_ctx`。进程内 barrier +
peer D2D 读（MegaMoE mempool 已开 peer access）。M0 约束：单 superstep，每段
≤ chunk_tokens（4224）。

实测（tray14，pruned 224-expert @EP4，`cp_prefill.rs`）：

- **logits gate**：全深度 16k，CP4 vs CP1 argmax/采样 token 一致，rel_l2
  2.52e-1。两路不 bitwise（切分不同 → bf16 累积噪声随深度 √L 扩散：1 层
  1.6e-2 → 4 层 4.2e-2 → 93 层 2.5e-1），bar 定为 `4e-2·√layers`；硬门禁是
  argmax + token。方向反的 merge（M·S）在 4 层就打出 2.3e-1 + argmax 翻转，
  与噪声可区分。
- **TTFT，内部墙钟（min-of-4，one-shot，forward 直测）**：

  | tokens | CP1 ms | CP4 ms | speedup |
  |-------:|-------:|-------:|--------:|
  | 2048 | 354.8 | 174.2 | 2.04x |
  | 4096 | 707.2 | 284.6 | 2.48x |
  | 8192 | 1442.4 | 517.0 | 2.79x |
  | 16384 | 2980.2 | 1026.2 | **2.90x** |

- **跨引擎口径 = HTTP e2e 同脚本打两边、卡数对等（4 卡对 4 卡）**。
  我们 serving（CP1 chunked prefill @EP4，tray14，`PEGAINFER_K3_MAX_CTX=32768`）
  与 vLLM pruned TP4（chunk 16k 档，存档 2026-08-17）同用一份
  `bench_vllm_ttft.py`，e2e min-of-4：

  | tokens | pegainfer CP1 e2e | vLLM TP4 e2e |
  |-------:|------:|------:|
  | 122 | 132 | 59 |
  | 258 | 173 | 64 |
  | 494 | 182 | 76 |
  | 1014 | 258 | 251 |
  | 2004 | 421 | 251 |
  | 4131 | 774 | 321 |
  | 8083 | 1506 | 595 |
  | 16395 | 3048 | **1181** |

  读法：单请求 TTFT 下 CP1 只有 1 卡在算 prefill（DP 的另 3 卡陪跑 padding），
  对 vLLM 4 卡切一个 prompt 慢 ~2.6×——这正是 CP lane 存在的理由。CP4 是
  我们的 4 卡形态：内部墙钟 1026 ms + 本引擎 ~70-130 ms 前端截距 ≈
  **~1100-1150 ms e2e 预估，对 1181 仅险胜个位数百分比**。真 CP4 e2e 要
  serving 集成后用同脚本实测（见 Next action）；固定截距我们比 vLLM 高
  （122-token 档 132 vs 59 ms），也是待办。
- 全量 fairness 账（2026-08-24 实算）：checkpoint 1453.7 GiB = expert
  1347.1 + dense 106.5；attn-DP 下 dense 每 rank 全复制 ⇒ **EP8 权重
  274.9 GiB/rank > GB300 可用 277.5 GiB 减 runtime，放不下；全量最少
  EP16（190.7 GiB/rank）**。vLLM 8 卡能起全量是 TP 连 dense 一起切
  （~182 GiB/rank）；全量对局 = 16 卡对 16 卡（vLLM TP×PP 或 sglang EP16）。
- microbench：`k3_flash_kda_bench`（FlashKDA 段长吞吐扫描）进
  `pegainfer-kernels` bench 系列。

### M1 — 交叉矩阵 + 系统 A/B（full@EP16，4 tray）

- **交叉矩阵**：`T_cp(c)`，c ∈ {1,2,4,8}，extend ∈ {8k,16k,32k,64k,128k}，
  冷启动 + prefix-hit 两组，chunk 扫描 → 产出 `amortization_threshold(c)` 表
  （§6 cost model 的查找表）+ 合法 CP 集合判决（§7 先验的证伪机会）；
- **系统 A/B**：whale-as-lane vs whale 独占步（对照臂按 mix-engine-design 原
  方案，可用 CP4 独占近似，不必真建 TP4），指标 = whale TTFT / decode TPOT
  p99 / goodput / EP barrier wait；
- 单 gang、gang 内 BS=1、暂无 prefix hit 进 CP 路径；
- 依赖：mega world >4224（roadmap 已挂账，whale chunk 12–16k 需要）。

### M2 — Agent cache 闭环

`ContextHandle`；KDA turn-level checkpoint + commit 协议；MLA striped KV；
DIST_CTX decode；长 hit + 长 extend 从 `S_cached`/prefix 恢复进 CP；
session home/group affinity。**做到 M2 才算支持 agent workload。**

### M3 — 弹性调度

多 gang 并存；c 动态选择（online latency model）；cache group 重分片；
work stealing。

## 10. Benchmark 清单

算子层：KDA context/extend {16k…1M} × CP {1,2,4,8} × 等长/不等长 × 局部段
{512…16k}；MLA prefill 不等长切分配平验证 + prefix-hit extend；MLA decode
ctx {16k…1M} × query {1,8,64,512} × local vs striped KV。

系统层用真实 agent trace（冷启动长度 / turn append 长度 / tool output 长度 /
hit ratio / ctx 生命周期 / 并发分布）。指标：TTFT p50/p99、TPOT p50/p99、
SLO 下 goodput、EP barrier wait、rank pre-EP 偏差、CP 通信时间、cache 迁移
字节、内存放大。

**验收不是 CP kernel 快几倍，而是：同 decode SLO 下，弹性 lane 的 agent
goodput 高于静态 DP16 与独占步方案，且长上下文请求无 OOM、无长期饥饿。**

## Next action

M0 已落地（见 §9）。**M0.5：CP prefill 接进 serving，拿真 HTTP e2e**——
(a) gang 协调：长 prompt 触发全体 rank 走 `prefill_cp`（scheduler 是
per-rank partition 自由跑，需要跨 partition 的 gang 招募）；(b) decode 归属
末 CP rank：KDA 终态与 conv 窗口天然在它那，MLA 上游行在 CP 交换时顺手写进
它的 pool（全局位置，~340 MB D2D），之后 decode 全本地，不需要 DIST_CTX；
(c) 用同一份 bench 脚本重测 e2e 表。顺带查前端固定截距（我们 132 ms vs
vLLM 59 ms @122 tokens）。然后 M1：full@EP16 交叉矩阵 + 系统 A/B（前置：
mega world >4224、多 superstep 段游走）。
