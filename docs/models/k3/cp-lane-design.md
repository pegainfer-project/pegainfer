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
（**DONE 2026-08-24：CP4 16k logits gate 绿，零 CUDA 改动**）→ M0.5 CP serving
集成（**DONE 2026-08-24：gang = free-running leveling loop，decode 归 owner；
HTTP e2e 同脚本 4 卡对 4 卡：16k CP4 1161 ms vs vLLM TP4 1181 ms 首次同口径
险胜，对内 CP1 3045 ms = 2.62×，交叉点 ~1k tokens，详表见 §M0.5。**16 卡全量
对局 DONE：EP16 CP4 16k 1072 ms，CP4/CP1 2.86×；1–2k 赢 vLLM TP16-MNNVL
1.3–1.5×，8k+ 输 0.68×（我们 CP 宽度钉在 4）——M1 加宽的立项数据，详表见
§16 卡对局**；**gap anatomy 2026-08-25：2.86× 的缺口 40% 是 MoE 不吃 CP 的
Amdahl 项，CP4 硬顶 3.49×，通信仅 0.6% 且纯 peer D2D 已快过 NCCL 地板，
详见 §CP4 gap anatomy；event 门控交换 + M+D 融合 kernel 双双落地后
CP4/CP1 = 2.95×，剩余缺口只剩 FMHA 三角与 Amdahl 项；**mega 协议上限
2026-08-25 直升 16896：CP4 单 superstep 盖 67.5k，64k 门禁绿 CP4 5.0s vs
CP1 13.9s，CP8/CP16 只差 gang 侧**）→ M1 full@EP16
交叉矩阵 + lane-vs-独占步系统 A/B
→ M2 agent cache 闭环 → M3 弹性调度。

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
≤ chunk_tokens（当时 4224；2026-08-25 mega 协议上限升 16896 后 CP4 单
superstep 盖到 67.5k，见 §Mega 协议上限）。

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
- 依赖：mega world >4224 —— **DONE 2026-08-25，直接 4 倍到 16896**（见
  §Mega 协议上限）。

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

### M0.5 — CP prefill 接进 serving（2026-08-24，DONE，e2e 实测落表）

实现形态（`scheduler/gang.rs` + executor trait 扩展）：

- **Gang = 贴任务板 + free-running 纪律**。admit 到长 prompt 的 partition 把
  `(prompt, poster)` 贴上板（`K3CpGang`），自己即 owner（CP rank `size-1`，
  末段——KDA 终态/conv 窗口天然是全 prompt 的）；其余 partition 每步开头查板，
  全员按**贴板顺序**执行（否则两个 gang 的 exchange 窗口会交叉）。
- **Leveling loop（唯一的会合原语）**：EP mega launch 跨 rank 按绝对序号
  **双边**配对——一个 step 的 device sync 只有在所有 peer 排队了同号 launch
  后才返回，free-running 下各 rank 天然被钉在 ±1 launch。gang 成员的纪律
  是**任何时刻不许安静等待**：每轮把自己（即将到达）的 launch 数写上板
  （pump 前先写 `count+1`，防 stale bid），人未齐或自己低于 `max(bids)` 就
  `pump_step`（阻塞到完成，即背压），只有"人齐且自己 == max"才进
  `prefill_cp`。人齐后 max 不再增长（只有低于 max 的才 pump 且 bid ≤ max），
  全员在**同一 launch 数**进 compute——这是 `prefill_cp` mid-step CPU
  exchange 的先决条件（不等长进入时，领先 rank 的 mid-step sync 等的
  peer launch 永远不会来）。
  灵感直接来自 `models/glm52/free-running-dp.md`：曾试过 arrive/bid/equalize
  三阶段 + condvar 纯等待，四连死锁（60s DeepGEMM grid-sync timeout）——
  最后到齐的 partition break 后停止发射，其余 partition 还堵在 pump 的
  `gpu.sync()` 里等它的下一个 launch。任何"到齐后改纯等待"的协议都有这个洞。
- **pump 不翻 parity**（关键坑）：plain decode 的不变量是"所有 running slot
  的 KDA 态/conv 窗口住在 `self.parity` 侧"。pump 可能在本 rank 有活跃
  slot 时跑；钉在当前 parity 上它只读 committed 侧、涂写 scratch 侧，下一
  个真 step 会整行覆写。翻了 parity 就是把 decode 指向 token-0 垃圾态。
- **decode 归属 owner**：owner 把 paged pool 按全局位置 stage（自己段走
  step 的正常 latent append；上游段在 MLA exchange 时从拼好的 `mla_ctx`
  经 `append_latent` 落 pool），epilogue 边界采样 + `adopt_row` 进 decode
  slot——之后 decode 与普通 prefill 后完全一致，全本地，无 DIST_CTX。
- 开关：`PEGAINFER_K3_CP`（须 = 本进程 local rank 数；fleet 上每进程自己
  一个 gang，远端 EP rank 以 padding 陪跑；dspark 互斥）、
  `PEGAINFER_K3_CP_MIN`（准入下限，默认 2048）。资格窗上限 = 每段 ≤
  chunk_tokens（M0 单 chunk 约束）。
- M0 gate 在 M0.5 executor 改动后复跑绿：16k rel_l2 2.52e-1（bar 0.386），
  argmax/token 一致。serving 验证：512-token 探针 48 greedy token 连贯；
  并发双请求 3 gang job 全执行、输出一致。

**e2e HTTP 实测**（2026-08-24，tray14 pruned@EP4，同一 `bench_vllm_ttft.py`
n=4 取 min，4 卡对 4 卡；CP 档用 `PEGAINFER_K3_CP_MIN=128` 全长度强制走
gang——生产默认 2048 以下走本地）：

| tokens | vLLM TP4 | 本引擎 CP1 | 本引擎 CP4 serving | CP4/CP1 | CP4 vs vLLM |
|---|---|---|---|---|---|
| 122 | 59 | 132 | 132（<128 不走 CP） | — | 0.45× |
| 258 | 64 | 173 | 230 | 0.75× | 0.28× |
| 494 | 76 | 182 | 238 | 0.76× | 0.32× |
| 1014 | 251 | 258 | 258 | 1.00× | 0.97× |
| 2004 | 251 | 420 | 307 | 1.37× | 0.82× |
| 4131 | 321 | 774 | 417 | 1.86× | 0.77× |
| 8083 | 595 | 1506 | 649 | 2.32× | 0.92× |
| 16395 | **1181** | 3045 | **1161** | **2.62×** | **1.02×** |

判读：CP4/CP1 交叉点 ~1k tokens（默认 `CP_MIN=2048` 合理）；16k 档 e2e
**首次同口径压过 vLLM TP4**。中段（2–8k）落后主要吃在前端固定截距
（我们 132 ms vs vLLM 59 ms）：剥离截距后 8k/16k 计算部分快 4–9%，
4k 慢 ~9%。截距是下一个杠杆，不在 CP lane 范围内。

### 16 卡对局（2026-08-24，DONE：vLLM TP16 vs 本引擎 EP16 全量）

4×GB300 tray（tray05/10/13/14）×4 卡，全量 1.5T checkpoint，同一
`bench_vllm_ttft.py` n=4 取 min。本引擎裸机 EP16（每进程 4 rank，
`--k3-ep-size 16`，MoE 走 NVLink fabric）；vLLM 容器化 ray TP16
（bring-up 12 连坑全录见 `bench_results/2026-08-24-k3-tp16-vs-ep16/`
存档 README——MNNVL/IMEX/GID/FlashInfer fusion 死锁）。CP4 档
`PEGAINFER_K3_CP=4 PEGAINFER_K3_CP_MIN=128` 全长度强制走 gang。

| tokens | vLLM TP16 (MNNVL) | vLLM TP16 (RoCE) | CP1 | CP4 | CP4/CP1 | CP4 vs MNNVL |
|---|---|---|---|---|---|---|
| 122 | 98 | 117 | 133 | 132 | — | 0.74× |
| 258 | 96 | 197 | 174 | 236 | 0.74× | 0.41× |
| 494 | 116 | 319 | 183 | 243 | 0.75× | 0.48× |
| 1014 | 391 | 576 | 261 | 259 | 1.01× | **1.51×** |
| 2004 | 406 | 1110 | 425 | 302 | 1.41× | **1.34×** |
| 4131 | 418 | 2193 | 782 | 441 | 1.77× | 0.95× |
| 8083 | 439 | 4385 | 1516 | 604 | 2.51× | 0.73× |
| 16395 | 728 | 8921 | 3061 | **1072** | **2.86×** | 0.68× |

判读：
- **CP4/CP1 的纵向账在全量 EP16 上完全兑现**（16k 2.86×，比 pruned@EP4
  的 2.62× 还好——全量 MoE 更重，CP 摊薄的 attention 份额相对更大）。
- **对 vLLM 的横向账分两段**：1–2k CP4 赢 1.3–1.5×（他们的 TP16 中段
  掉进 ~400ms 平台）；4k 基本打平；8k+ 输 0.68–0.73×——vLLM prefill 是
  16 路全切，我们 M0 的 CP 宽度被单 superstep 约束钉在 4（每段 ≤4224；
  2026-08-25 mega 升 16896 后该约束实际消失，CP8/CP16 只差 gang 侧）。
  **8k+ 的差距就是 M1 CP8/CP16 的立项理由**：CP4 已把
  CP1 的 3061 压到 1072，宽度每翻倍理论上还有 ~1.6–1.8× 空间。
- vLLM 两列的差距（16k 728 vs 8921，12×）是 fabric 的账不是引擎的账：
  任何一台 tray 的 nvidia-imex daemon 挂掉，整个 MNNVL 域的 fabric
  import 全崩（CUDA 801/600 报的是 import 方不是病灶方），NCCL 静默落
  RoCE。基线以 MNNVL 列为准。

### CP4 gap anatomy（2026-08-25：为什么是 2.86× 不是 4×）

nsys 剖析 `cp_prefill` gate（pruned@EP4，tray13，CP1/CP4 同进程背靠背
@16384/16383），全账 + NCCL/D2D 通信地板见
`~/code/bench_results/2026-08-25-k3-cp4-16k-profile/`。核账（kernel 墙钟
CP1 3159.8ms / CP4 1116.8ms = 2.83×，暖跑 2.88×，与 serving 2.86× 对上）：

- **完美缩放的部分**：elementwise 3.99×、dense GEMM 3.93×、DtoD 3.93×——
  4k 行的 kernel 效率没有损失。
- **+132ms：MoE 不吃 CP（Amdahl 项，缺口的 ~40%）**。专家算量每 rank 在
  CP 下不变（mega 本来就把 token 全 EP 分发）；CP4 的 mega 反而比 CP1 快
  （93 发 ×16k tokens 比 372 发 ×4k 批得好，391→~229ms），但永远到不了
  97.8。**硬天花板 = 2700/4+229 = 904ms → CP4 极限 3.49×**，4× 从一开始
  就不存在。
- **+141ms：临界 rank（dev3=owner）的空转**——57ms 一次性（首 gang 的
  peer grant + scratch 建立，暖跑消失）+ ~100ms 每层 0.5–2ms 的交换
  barrier 等待。
- **+32ms：FMHA 三角**。causal 使 rank r 的 context 是 r+1 段：每层
  766/1696/2605/3562µs（正好 4k/8k/12k/16k），总量守恒但墙钟付末 rank 的。
- KDA 3× doctored 调用（dev1/2 +68ms kernel 时间；rank0/rank3 各 1×，
  rank3 只等 merge）**大多被隐藏**：跑在等上游 state 的窗口里，只经 mega
  两侧配对的 skew 上墙。
- **通信不是缺口，也不是 NCCL**：CP 窗口零 NCCL kernel，纯 peer D2D。
  PtoP 字节与理论逐 rank 对上（KDA 包 6.29MB fp32 占大头，r3=2513MB
  实测 vs 2509 理论），拷贝引擎共 7.1ms = **墙钟 0.6%**。同 tray torch
  地板：raw D2D 779GB/s，NCCL 复刻整个交换 pattern 要 17.5ms——比我们
  现在还慢 2.5×（6.29MB 消息在 NCCL 协议开销区，91 vs 213GB/s）。
  MegaMoE 式融合没有字节可藏，可融的是 barrier 空转那 ~100ms。

**Event 门控交换已落地（2026-08-25，`f0399a6b`）**：per-window host
Barrier + 双 stream sync 换成四拍 CUDA event 协议（publish/consume event
跨卡 stream wait + enqueue 侧原子计数保证 record-before-wait；等待集合按
窗口收窄——halo 只挂前驱，upstream 扇出只挂真生产者）。host 线程整个
superstep free-run，只有真依赖 stall stream。同门禁复测：logits 逐位一致，
CP4 kernel 墙钟 1116.8→1071.9 ms（冷）/ 1054.9→1037.7（暖），
**CP4/CP1 2.83→2.92×**；非临界 rank 空转 93/91→12/10 ms。剩余空转全部
有名有姓：dev3 的 72.8 ms（×69 层）= 等 dev1/2 M/D doctored 调用的真依赖
（下一段：已被 M+D 融合 kernel 砍半）；dev0 的 53 ms 是 consume-wait
挂在窗口尾的保守放置（不在临界路径，修了墙钟不动，暂不动）。

**M+D 融合 kernel 已落地（2026-08-25）**：中间 rank 的两次 doctored 调用
（M：v=0+单位态；D：真 v+零态）本就共享整个 kernel-1（prepare 与 v/state
无关）且丢弃 token 输出，融成 `k3_flash_kda_fwd_md` —— 一次 K1 扫 + 双态
K2（砍 q-GEMM/Phase4/Phase5/out-store，每 tile 6 发 GEMM 对 10 发）。保持
上游 GEMM 形状与逐态累积顺序 → **与两次 doctored 调用逐位一致**
（`k3_flash_kda_md_equiv` gate，T∈{14,224,1056,4223,4224} 全 0 diff）；
微基准 pair 1.0165→0.5943 ms @4224（1.71×）。整机同门禁复测（tray13，
argmax/token 一致）：暖跑 CP4 forward 墙钟 **1037.7→1001.6 ms，CP4/CP1
2.92→2.95×**，−36 ms 正中 −28 ms×69 层预估区间。顺手删掉 `zero_v`/
`zero_state`/`identity` scratch，每 rank 省 ~117 MB。

杠杆排序（更新）：event 门控 DONE（2.92×）；M+D 单 kernel 融合 DONE
（2.95×）→ FMHA 条带化配段（≈3.1–3.2×）→ **3.49× 是 EP 共享 MoE 的硬顶，
再往上只有 M1 加宽**。

## Mega 协议上限 4224 → 16896（M1 的前半，2026-08-25）

`kProtocolMaxTokensPerRank` 直接 4 倍到 16896（=44×384 对齐；ring 容量随
之线性缩放，是 kernel 模板参数所以全 world 重实例化）。动机：**单
superstep 的 CP 天花板从 16.9k 直升 67.5k @CP4**（CP16@EP16 理论 256k+，
FlashKDA 264k 线性已证 kernel 侧免费），多 superstep 段游走在实用长度内
基本失去必要性；CP1 本地 chunked prefill 同时受益（16k = 单 chunk，mega
发射数 ÷4，GEMM 形状进高效区）。

代价是 symm slab 随协议上限线性长（`k3_mega_symm_buffer_layout` 实测）：
EP4/224 1.6→6.3 GiB，**EP16/896 5.1→19.6 GiB**——EP16 全量（权重 190.7
GiB/rank）后 HBM 收紧 ~15 GiB，fleet 复测时要盯。TileLang prefill bucket
梯子补 8448/16896 两档（AOT 全家 +24s 构建）。

验证链（tray14/06，pruned@EP4）：
- **16k 门禁与旧协议逐位同答案**（rel_l2 2.523e-1/4.511e-2 与 4224 时代
  完全一致——GEMM 行独立 + FlashKDA 跨 call 状态 bf16↔fp32 无损 + FMHA
  行内在线 softmax，chunking 对 CP1 逐位稳定）；CP4 墙钟不变（段仍 4096）。
- **64k 单 superstep CP4 落地**：argmax/token 与 CP1 一致，rel_l2
  9.4e-2/7.0e-2（bar 0.386）；暖跑墙钟 **CP1 13870 ms vs CP4 5006 ms =
  2.77×**。
- mega oracle 双绿且 **Gate 1 bitwise**（EP4 vs EP1 全 40 token +
  163840 logits 逐位一致），Gate 2 流量不变性 bitwise 保持。
- ttft sweep（tray06，min-of-4）：CP1 @16k 2980→2870 ms（单 chunk mega
  赚 ~110 ms），CP4 @16k 1003.5 持平（段仍 4096），CP4/CP1 2.86×；
  128–8k 档与旧表一致（128: 0.84× / 1k: 1.68× / 4k: 2.60× / 8k: 2.82×，
  交叉点 ~400 tokens 不动）。
- golden_decode 串行 13/13 绿（`--test-threads 1`，bring-up.md 的规定跑
  法；默认并行会 13 实例同租一卡 OOM——mega slab EP1 涨到 3.2 GiB 后更
  容易炸，跑法不对怪跑法）。

## Next action

M1 后半：CP8/CP16 gang（full@EP16 交叉矩阵 + 系统 A/B），8k+ 追平 vLLM
TP16 的 16 路切分；EP16 全量下 19.6 GiB slab 的 HBM 账实测。次优先：FMHA
条带化配段、前端固定截距（132 vs 59 ms @122 tokens）。
