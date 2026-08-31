# K3 mix engine design — batch 形态与并行形态收束

**TL;DR**: mix serving（P/D 不分离，agent 负载 avg ctx ~200k，draft 常驻 + 静态 k）
的引擎形态收束为：**两种步型**（稳态 = verify 预留 + defer 聚合的短 prefill filler；
whale = 占空比插入的大 chunk step），统一原语是 **span = 已提交前缀 + 投机尾巴**
（`K3KdaGroup{commit_rows, spec_rows}` 已是这个语言）；MLA 双 kernel（latent FMHA
稳态通吃 / dense FMHA 仅 whale）被折叠进"是否 whale"一个 bit。
**whale 并行形态已被 `cp-lane-design.md`（2026-08-24）取代**：whale = CP gang
lane 与 local lane 在同一 EP16 superstep 共存（MoE 天然全宽），TP 出场；本文原
瞬态 CP4×TP4 方案降级为 M1 系统 A/B 的对照臂。
三份测量（GEMM 形状扫描、FlashKDA 段长扫描、mega padding 地板）同指
whale chunk 12–16k 的组件工作点；四卡同后端 100k KDA 对照中 KCP4
为 8.198 ms、TP4 为 8.425 ms，KCP4 低 2.70%，证明 KDA 的 CP 不是串行 state
relay——这也是 lane 方案敢去掉 TP 的组件证据之一。

Last touched: 2026-08

## 前提（susun 拍板，2026-08-24）

- 最终方案是 **mix**：P/D 同一个 EP group，不做分离。
- **draft 常驻**：每个生成中的 slot 永远在 verify round 里；k（draft 数）是
  **静态**启动配置（值另议，仿 `GLM52_MTP_DRAFTS`），不做自适应——静态 k 使
  verify segment 形状可预留、可 graph 桶化。
- Agent 负载：avg ctx ~200k；prefill 两个种群——稳定到来的短 suffix（前提是
  session KV 常驻，见下）和偶发几十 k 的 whale。

## 统一 batch 原语：span

稳态下 "decode" 作为独立类别消失：每序列每步贡献一段连续 span（长度 n ≥ 1），
属性 = (commit_rows, spec_rows)。prefill chunk 是 `spec_rows=0` 的 span，
verify pack 是带投机尾巴的 span，plain decode 是 n=1 的退化 span（ctx-headroom
clamp 落到这里）。现有三个 `K3StepMode`（Decode/PrefillChunk/Verify）收敛成一个
group 化 step 原语——KDA 侧 `kda_attention_chunk` 今天已经是这个形态
（PrefillChunk 就是单 group 特例）。

KV 提交语义随之统一：正常 append，被拒 draft 靠页指针回卷。prefill 独立
asymmetric pool + `adopt_row` 的形态消失：prompt 从 admission 起住进 slot，
原地逐 chunk 消化，"prefill→decode 转换"不再存在。

## 两种步型（fleet 每步共识一个形态描述符）

| 步型 | 行组成 | 执行 | 频率 |
|---|---|---|---|
| 稳态 | verify packs（静态预留 `slots × 2(k+1)` 上界）+ defer 聚合的短 prefill spans（filler，cap ≤ q*） | filler=0 时 CUDA graph 捕获；有 filler eager | 绝大多数 |
| whale | 单长 prompt 大 chunk（目标 12–16k，见测量） | eager，占空比插入 | 偶发 |

- **发射节拍固定**：先发 prefill/filler segment → host 同时 propose（藏进计算，
  顺带消灭 EP16 round anatomy 里的 convoy seed）→ verify segment → epilogue
  （acceptance + boundary sample，row-mask 只对采样行）。
- **defer**：短 prefill admission 攒批（预算满或 O(10ms) 超时 flush），步型
  二极化——多数步保持 graph 捕获的纯 verify 形状；admission 时做 attn-DP
  placement 摊负载。
- **占空比** = whale TTFT vs TPOT p99 的 SLO 旋钮；whale 期间种群 A 的 defer
  窗口自然拉长。优先级：verify 预留 > filler > whale。
- **形态描述符的共识走 TCP 控制面**，不进数据面 collectives：描述符可提前
  几步流水裁决，不在 lockstep 关键路径上。复用 rendezvous 那条 TCP；fleet
  现在是 fail-stop 全体重启，rank0 当 sequencer 即可，无需选主——将来要
  成员变更/容错再考虑 raft（rust 有 `openraft`）。

## Kernel 清单

| 组件 | kernel | 覆盖 |
|---|---|---|
| KDA | FlashKDA chunkwise **varlen**（一层一发） | 全部 span，两步型通用 |
| MLA ① | **latent FMHA**（q-blocked absorbed，块内 causal，q=1 退化 decode） | 稳态一切：verify、退化 decode、短 filler span |
| MLA ② | **dense FMHA + 展开**（W-chunked KV 循环 + LSE merge） | 仅 whale |
| MoE | mega 单 instantiation | 行无感（whale 需更大 world，见 TODO） |
| 其余 | GEMM/norm/epilogue 行 batch | 通用 |

MLA 二象性是矩阵链结合律的本质（`q·(W_uk·c)` vs `(q·W_uk)·c`，FLOPs 粗算交叉点
q\*≈170，待实测），kernel 消不掉，但 mix 的 cap 选择使它整个折叠进"是否 whale"。
dense 路径在 200k ctx 下现有单发 `[ctx|chunk]` 的展开 workspace（~17GB）不可行
——chunked-prefill 计划里挂账的 W-chunked context loop + LSE merge 到期。

**varlen FlashKDA 不是 launch 数洁癖，但 9.1× 只是优先级证据**：段长扫描给出
16 个独立 T14 调用共 0.415ms，而单序列 T224 调用为 0.046ms；真正 varlen
仍有 16 份独立 recurrent state，不能把两者比值当作实测收益。这个数量级足以把
varlen instantiation 提到第一梯队，但收益必须在一发多序列实现落地后重测。

## 并行形态：稳态 TP1×DP×EP + whale 瞬态 CP4×TP4（**superseded**）

> **2026-08-24 更新**：本节方案被 `cp-lane-design.md` 取代——whale 不再独占
> fleet 重构成 CP4×TP4，而是作为 CP gang lane 与 local lane 共存于同一 EP16
> superstep；"CP16 有 FlashKDA 悬崖所以要 TP4 凑 16-way"的前提随 lane 化消失
> （whale attention 只需 gang 宽度，MoE 自动 EP16 全宽）。本节保留为独占步
> 对照臂的定义与当时的论证记录。

TP 是 **step 属性不是 rank 属性**：whale step 里整个 group 瞬态重构成一台
CP4×TP4 机器（tray 内 TP4 按头切，tray 间 CP4 按 chunk 行切；KDA 走 local
affine summary all-gather + fp32 prefix merge + local forward 的 KCP，不做段间串行
state 接力；MLA 前缀 latent 在 whale 期间各 rank 本地镜像），下一步回到
TP1×DP16。EP 切分与 mega slab 自始至终不动。

- **为什么能瞬态**：latent KV head-agnostic = TP-layout 不变量，whale 算出的
  latent 页直接服务之后的 TP1 decode，退出近零成本（只 gather 定长 KDA state）。
  GQA 架构做不到这一点。
- **权重成本**：每 rank 存 TP1 全量 + 1/4 TP4 shard（dense/attention 件），
  +25% dense 权重，无所谓。
- **TP 因子止步 4 是 GEMM 形状经济学**（非带宽）：TP8 切片 kv_a 447TF、
  专家 l2 K=384 封顶 1180TF vs EP 2380TF；NVL72 跨 tray ≈ tray 内带宽，
  拓扑不构成约束。
- **TP 买 session 容量 + whale 时延，不买稳态 TPOT**（稳态 MLA 是 latent
  带宽 bound，切头不切 latent 读）——这也是它只以瞬态形式存在的原因。

## 测量依据（组件证据指向 whale chunk 12–16k @ CP4×TP4）

1. **GEMM/TP 形状扫描**（susun，`~/code/bench_results/2026-08-17-k3-prefill-tp-vs-ep`）：
   t=4 定案；GB300 有效峰值 1.6PF bf16 / 2.8PF fp8；contiguous padding 地板
   → EP16 全模型 chunk/56 ≥ BLOCK_M 192 即 **chunk ≳10.7k 才吃满 expert tile**。
2. **FlashKDA 段长扫描**（`~/code/bench_results/2026-08-24-k3-flashkda-segment-sweep`，
   bench 在 `pegainfer-kernels/tests/k3_flash_kda_bench.rs`）：固定开销
   ~26µs/call，~90% 饱和 T≈2k；CP4 的局部段形状 @ chunk≥8k 为理想吞吐的
   94–97%，CP16 悬崖。这里只量局部 kernel shape，不含 recurrent state 依赖链。
3. chunk 16.9k 工作点：per-rank 段 4224——KDA 97%、GEMM M 肥、expert tile 满。
4. **100k KDA TP/CP 对照**（
   `~/bench_results/2026-08-24-k3-kcp4-vs-tp4/tray14-gpu0-3`）：同一 FLA
   KDA 后端下，TP4（每 rank H24×T100k）8.4250ms，KCP4（每 rank
   H96×T25k，含 summary all-gather/merge）8.1978ms；KCP4 低 2.70%，四卡
   效率 71.85% vs TP4 69.91%。这是组件实测，未含 projection、TP output
   collective、MLA/MoE 与调度，故全路径仍未定案。

## Session KV 常驻（短 prefill 种群的结构前提）

200k avg ctx 下"短 prefill"只因前缀活在 KV 里而短。K3 混合架构友好：69 层
KDA 定长 state（snapshot/restore 免费）+ 24 层 MLA latent 页
（~5.5GB/session@200k）→ 活跃 slot 在卡、闲置 offload host（kv-offload），
resume = page-in。这是一个子系统，不是优化项。

## TODO（依赖序）

1. **latent FMHA**（q-blocked absorbed，ctx-tiled）：三个身份合一——verify 的
   16× KV 重读修复、稳态统一 kernel、短 filler 路径。先在 GB300 上对测
   q ∈ {14,64,128,256,512} × ctx ∈ {2k,8k,32k} 拿真实 q\*。
2. **varlen FlashKDA instantiation**（vendored 裁剪加回，build 配置级）——
   c16 verify 一阶修复。
3. **span/group 化 step 原语重构**：K3StepMode 收敛、prefill pool 消失、
   prompt 直住 slot。
4. **调度器**：verify 预留 + filler 预算 + defer admission + 占空比。
5. **mega world 扩充**：whale 用 >4224 的 protocol max（≥10.7k 才吃满 EP16
   tile；lockstep 单 instantiation 约束下评估 whale 专用 world vs 全程换大）。
6. **dense 路径 W-chunked KV 循环 + LSE merge**（200k 必需）。
7. **形态描述符 TCP 控制面**（rank0 sequencer，提前流水裁决）。
8. ~~**瞬态 CP4×TP4**~~ — superseded by `cp-lane-design.md`：双布局权重与
   head-range 参数化不再需要；存活的子项（KDA KCP summary exchange、latent
   prefix 读取布局）并入该文档的 M0/M2。
9. **session KV 常驻 + offload 生命周期**；max_ctx 4096→200k 全套尺寸账
   （scratch、draft arena、drafter YaRN 窗口）。

"whale 期间 filler 是否共存"已由 `cp-lane-design.md` 拍板：结构性共存（lane 化）。
仍未拍：静态 k 的取值实验（acceptance-vs-k 按负载类别，全模型）。

## KDA TP4 vs KCP4 裁决实验（2026-08-24）

### Preparation

- **Read**:
  - `docs/index.md` — K3 mix-engine 文档是本次并行形态裁决的归档位置。
  - `docs/models/k3/mix-engine-design.md` — 段长扫描只回答 local kernel shape，
    未回答 KCP 的 summary exchange 与 prefix merge。
  - FLA 0.5.2 的 `fla.ops.cp` / KDA `cp_context` 路径 — KCP 先并行计算各段
    affine summary，再 all-gather、fp32 前缀合并，最后并行算本地输出。
- **Relevant history**:
  - 把 CP4 写成 `4 × t(H96,T25k)` 等价于假定 recurrent state 逐 rank 接力；
    这只能描述 naive relay，不能描述已有的 KCP 算法。
- **Plan**:
  1. 用同一个 FLA KDA 实现比较 TP4（四 rank 各 H24×T100k）与 KCP4（四 rank
     各 H96×T25k），避免跨后端污染。
  2. KCP4 计时覆盖 local summary、NCCL all-gather、fp32 prefix merge 与 local
     forward；每轮取四 rank 最大时间，2 次 warmup、7 次计时取中位数。
  3. 先用 T1024 对比单 rank reference 与 gather 后 KCP4 输出；只保留通过正确性
     门禁的最终运行及完整复跑脚本。
- **Risks / open questions**:
  - FLA Triton 不是生产 CUTLASS FlashKDA；本实验裁决并行算法，不代表生产融合
    kernel 的最终绝对延迟。
  - 这是 KDA forward-only 组件实验；TP output collective 和其余网络层不在范围内。

### Execution Log

- **环境**：tray14 GPU0–3 均为 0 MiB / 0%；容器
  `verl:vllm-nightly-8fdcb420-fla`，torch 2.13.0、Triton 3.7.1、FLA 0.5.2、
  NCCL 2.29.7。只分配随机 q/k/v/g/beta 张量，不加载 checkpoint 或权重。
- **正确性**：T1024 的 KCP4 gather 输出相对单 rank reference：relative L2
  `4.5466e-4`、cosine `0.99999982`、max abs `6.1035e-5`。
- **实测**（tray14 GPU0–3，同一后端，七轮中位数）：
  - 单 rank H96×T100k：23.5611ms；
  - TP4 H24×T100k/rank：8.4250ms，2.7966×，四卡效率 69.91%；
  - KCP4 H96×T25k/rank：8.1978ms，2.8741×，四卡效率 71.85%；
  - `KCP4 / TP4 = 0.97303`，即 KCP4 低 2.70%（0.2272ms）。
- **归档**：
  `~/bench_results/2026-08-24-k3-kcp4-vs-tp4/tray14-gpu0-3/`，含唯一
  `raw.log`、精确 Python 源码、环境清单和空卡门禁复跑脚本；旧 naive-relay
  结果不保留。

### Debrief

- **Outcome**：KCP 的 associative affine-summary 算法把 recurrent prefix 依赖
  并行化；在该 100k shape 上不仅没有 `4×segment` 串行代价，还略胜同后端 TP4。
- **Interpretation**：2.70% 差距不足以单独裁决整个 whale topology，但 KDA 已不再
  是反对 CP4 的证据；它与“CP 保留胖 GEMM tile”的方向一致。瞬态 CP4×TP4 是否
  继续需要 TP4，仍取决于多请求并发、MLA、TP collective 和实际 chunk 路径。
- **Pitfalls encountered**：第一轮曾把 production FlashKDA TP4 与 FLA KCP4
  做跨后端对照；最终运行改成同一 FLA 后端，并删除中间/旧 relay 结果。
- **Lessons learned**：线性 recurrence 不等于跨 rank 必须串行；只要每段可表示成
  可结合的 affine transform，就能先并行求 summary，再做小状态的 prefix merge。
  CP 基准必须测这一实现，不能以 `C × local_segment_time` 代替。
- **Follow-ups**：把 KCP 接进 production FlashKDA/引擎路径，并在真实 12–16k
  whale chunk 上同测 projection、MLA、MoE、TP collective 与最终 state 布局；
  本次 100k one-shot 只锁定 KDA component。
