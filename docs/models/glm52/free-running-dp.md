# GLM5.2 free-running DP:删除协调者的架构设计

> **TL;DR:** EP16 P/D 难支持暴露的不是 P/D 问题,是 DP 架构问题:现在的
> DP 只切了状态没切控制,每个 rank-local 异步事实(P/D pull、offload、断连)都要经协议送回中央
> coordinator 才能生效,于是每个新 feature 都在给中心发明新协议(rank-host、Event plane)。
> 本设计把 coordinator 删除:每个 DP rank 是完整独立 engine(自己的 scheduler/BlockPool/HTTP
> endpoint),loop 无条件全速跑,唯一耦合是固定节拍的 DeepEP collective 链本身。换来的义务是
> 三条静态纪律(固定链、保守 bound、padding 即协议)。**§8 五个 gate 已全部 GO
> (2026-07-30,GB300 tray03):跨 rank 流量不变性 bit-exact、异构 token 数 graph 回放
> bit-exact、保守 bound 税 = 0、padding 字节逐步恒定、MTP 固定链轨迹相同且空 round
> 开销 = 0。§10 两步迁移均已实装:第 1 步 per-rank bucket、协议最大 global_tokens、
> MTP 固定链(gate 1–5 验证);第 2 步拆壳完成——coordinator 删除,每 DP rank 一个自治
> engine 线程,lease 本地化(always-consume,stale-replay 路径消失),rank-host
> (remote.rs,978 行)退役,跨节点 = 同一 binary 每节点一进程 + 一次性 bootstrap
> rendezvous(rank0 进程分发 DeepEP unique id)。**双 tray EP8 首验通过
> (2026-07-30,tray03+tray13):rendezvous 一次成功(fetch attempt 1),两 endpoint
> greedy 输出字节一致,多进程 native MTP 正常运行(mean_accepted_drafts≈1.2,两侧
> 轨迹一致),fail-stop 实测符合设计——杀 peer 后首个请求触发 DeepEP NVLink barrier
> ~17s 超时 → fatal ERROR log → 进程退出。** 取代 `cross-node-scaling.md` 的
> Event plane 与 SMR 方向(该文档的 NVL72 实测数据仍有效)。
>
> **Last touched:** 2026-07

## 1. 问题:DP 只切了状态,没切控制

现状(`scheduler/mod.rs` 的 `run_dp8_coordinator`)是"一个大脑、N 只手":admission、bucket
规划、launch-ahead lease、MTP round 协商、输出应用、client 回复全部集中在一个线程,rank worker
只执行。DP 切分的只有 KV 和 slot 状态。

这个形态下,每个 rank-local 的异步事实都必须传送回中心才能生效:

- P/D pull 完成 → `StartKvPull` 命令 + Event frame 回报 + parked 队列 + all-idle 时 5ms
  sleep 节流(`scheduler/mod.rs:386-393`);
- offload save 落地 → `SavePin::drop` 回调 → 未来的 Event frame;
- client 断连 → coordinator 的 `token_tx.is_closed()` 探测。

`cross-node-scaling.md` 的 rank-host 协议和 Event plane 设计,本质是给中心化架构打的远程补丁
——**补丁的存在本身就是架构问题的证据**:如果 rank 天生独立,这些协议根本不需要发明。EP16 P/D
只是下一个穿不过 coordinator 循环的 feature,不会是最后一个(`run_dp8_coordinator` 已经 14 个
参数)。

Repo 里两条模型线其实各选了一边:kimi-k2(`models/kimi-k2/dp-design.md`)选了 per-rank
独立 engine + EP 天然 sync,glm52 选了中央 coordinator。本设计是把 kimi 的选择泛化到 glm52,
并补上 kimi 文档没处理的 graph/MTP/padding 细节。

## 2. 不可约同步分析

EP 约束下真正删不掉的同步,穷举后只有:

1. **Collective 步调**:每 rank 进入每层 dispatch/combine 的次数和顺序必须一致。错位
   (mispair)= 某 rank 的第 N 层和别人的第 N+k 层配对 → 字节确定性垃圾,不 crash
   (`fail_step` 注释描述的场景)。这是最恶性的失败模式。
2. **协议上界**:collective buffer 按 `num_ranks × GLM52_MAX_BATCH_PER_RANK` 的最大形状
   分配。注意——**这不等于"每 rank 相同 bucket"**:`moe_ep_wo.rs` 的 dispatch 传 rank-local
   `num_tokens`,`global_tokens` 只用于收紧 masked GEMM 的 tile bound,recv 侧真实行数从
   device 上的 `psum_expert` 读。"bucket 全局一致"是 `plan_step_shapes` 取 hungriest rank
   的**选择**,不是 DeepEP 的要求。
3. **Fate-sharing**:一步失败 collective group 无法重新对齐,全 fleet 一起死。物理事实,
   任何架构下都在。

由于每 step 的 collective 链由模型代码写死(75 层 MoE,顺序无自由度),第 1 条退化为:
**所有 rank 对 step 计数一致**。唯一的自由度是"step N 跑不跑"——把 loop 改成无条件跑,
这个自由度也消失,不变量从运行时协议保证变成代码结构保证。

其余一切——admission、bucket、KV pool、sampling、client 回复、prefix cache、offload、
P/D pull、launch-ahead lease——都是 rank-local 的。集中在 coordinator 是历史选择,不是必然。

Idle 协调明确不做:部署姿态是机器默认满负荷,空 rank 全速跑 padding step,不引入任何
全局活跃度协议。

## 3. 目标架构

```
                        ┌──────────────────────┐
                        │  外部 router(无状态)   │
                        │  KV 亲和 / least-load  │
                        └───┬────┬────────┬────┘
                            │    │        │      ← 普通 HTTP,每 rank 一个 endpoint
                   ┌────────┘    │        └────────────┐
                   ▼             ▼                     ▼
            ┌────────────┐ ┌────────────┐        ┌────────────┐
            │ Engine 0   │ │ Engine 1   │  ...   │ Engine N-1 │
            │ HTTP       │ │ HTTP       │        │ HTTP       │
            │ scheduler  │ │ scheduler  │        │ scheduler  │
            │ BlockPool  │ │ BlockPool  │        │ BlockPool  │
            │ offload/PD │ │ offload/PD │        │ offload/PD │
            │ GPU worker │ │ GPU worker │        │ GPU worker │
            └─────┬──────┘ └─────┬──────┘        └─────┬──────┘
                  │              │                     │
                  └──────────────┴─────────────────────┘
                      DeepEP collective 链(唯一的运行时耦合)
                      每 step 固定:75 层 dispatch/combine + 5 个 MTP forward
```

- **控制面:不存在。** 没有 coordinator、step bell、Event frame、rank-host 协议。同步就是
  collective 的 back-pressure 本身。
- **数据面:一条固定链。** 每 step 每 rank 跑同一条编译期确定的 collective 序列。
- **请求面:普通 HTTP。** router 无状态(dynamo 路由已有先例,
  `subsystems/router/kv-aware-routing.md`);engine 间对彼此的请求/KV/slot 一无所知。
  local/remote 的区分从概念里消失:跨节点部署 = 每节点起进程加入同一个 DeepEP communicator,
  没有中心要连。

### 单个 engine 的 loop

```
loop {
    drain HTTP 请求 → 本地 admission(BlockPool 全生命周期预留,honor-or-reject)
    P/D:带 hash 的请求 → 本地 reserve → 发起 pull → 本地 parked 队列
    pull 完成(本地回调)→ 下一轮 admit
    按本地 slot 状态选 bucket → 选对应 graph
    step:固定 collective 链(有活带活,没活 padding 进场)
    apply 输出 → 直接回自己的 client
}
```

唯一让它区别于单卡 engine 的规则:**forward 无条件、collective 不跳过**。除此之外它就是
一个普通的自治推理引擎。TP8/TP4 mirrored 拓扑本来就是这个架构的 N=1 特例,原样不动。

### Draft/verify 两条 lane

- **Verify 免费**:verify 是 target step 里的 `SpanKind::Speculative` span,改变行数不改变
  collective 条数。行数差异由 gate 1 覆盖。
- **DSpark 零改动**:drafter 是 5 层 dense,rank-local 无 collective(`run_draft_round`
  注释自证),原样保留。
- **Native MTP 是真正的手术点**:layer 78 是 MoE 层,draft forward 也是 EP collective。
  现在 `select_round_kind`(`scheduler/mtp.rs`)按全 fleet 状态协商 Reset/Context/Propose
  ——每步 collective 总条数是变量,这正是中心化的残留。改法:**每 step 无条件跑固定 5 个
  layer-78 forward**,没活的 rank 以 padding 进场。steady decode 下 Propose 本来就几乎每步
  跑,固定链在主流工况零额外成本;round kind 协商、`source_bucket` 一致性 ensure、bucket
  全局 max 全部删除。

### Launch-ahead

lease 的 all-ranks-or-none 是 bucket 全局一致的推论;bucket per-rank 化 + 步进无条件后,
lease 降级为 rank-local bit:rank A replay 投机步、rank B 跑普通步,collective 按计数照样
配对。

## 4. 三条纪律(写进 conventions,是本架构的承重墙)

1. **固定链纪律**:任何 feature 不得引入"有条件的 collective"。省 collective 的唯一正确
   姿势是让空 rank 以零负载进场、kernel 内部便宜地穿过去;跳过靠 kernel,不靠 host 协商。
   CI 守法:数一个 step 的 collective launch 次数,断言是常量。
2. **形状本地纪律**:任何进 collective 的 buffer 按协议最大值做保守 bound,不得依赖
   "别人此刻的真实行数"。
3. **Padding 即协议**:任何进 collective 的 dummy 行,其全部输入(token、position、
   seq_len、KV 页内容、MTP shifted token)必须构造性确定,且有 byte-stability gate 守护。
   禁止"输出会被丢弃所以输入无所谓"——输出被丢弃,但路由和字节已经上了 wire,影响的是
   别人的 step。

## 5. Padding corner cases(纪律 3 的展开)

Free-running 后 padding row 从"本地丢弃的废行"升级为**协议表面**:

- **空 rank 的进场姿势:选整 bucket dispatch(路 B),不选 `num_tokens=0`(路 A)。**
  路 A 语义干净但 `num_tokens` 是 kernel 实参,每步变化破坏 whole-step graph;路 B 是今天
  的实际行为(`global_tokens = ep_ranks * batch`),graph-safe,代价是空 rank 发真实 a2a
  流量——满负荷部署下是零头。`token: None` 路径保留给 prefill-only。
- **Padding row 路由必须确定**:现有 `GLM52_PADDING_STEP` 契约(固定 token、position 0、
  seq_len 1、写 padding page 位置 0)大概率已构造性确定,但 indexer sparse top-k 在
  seq_len=1 上的行为和 fp8 quant 两个环节未验证——gate 3 把它从"碰巧对"升级成契约。
- **Lease × padding 位置走漂**:leased replay 的 `slot_mapping += 1` 不得推进 padding row
  (现有"padding rows reset by each full prologue"在全局 lease 下成立,本地化后需重新确认
  连续多步 lease 的复位边界)。
- **MTP dummy round**:固定 5 forward 后,零 proposal 的 rank 跑 bucket-1 dummy forward,
  需要明确的 `MTP_PADDING_STEP` 契约(layer 78 消费 shifted token),不得复用 capture
  buffer 残值——固定链下残值读取会变成每步都发生的事。
- **输出侧**:padding row 的 argmax/sampling 输出本地丢弃,现状已对,零新语义。

## 6. 失败模型

**Engine 内部错误 → crash early(现有姿势);任何 rank 死 → 全 fleet 经 collective 超时
数秒内 fail-stop → router 摘流量 → 全体重启。** 没有部分存活,没有脑裂可能——没有需要
一致的共享状态。KV 温数据在 pegaflow host tier 等重启后 restore。

变化只在检测与收尸的去中心化:每 rank 自己的 step watchdog(超时 → fail 自己的请求 →
进程退出),router 健康检查摘除,不再有"负责宣布死亡"的线程。fate-sharing 靠超时传染完成。

启动期协调无法归零但一次性:DeepEP 的 `ncclUniqueId` 分发 + graph precapture 全员就位,
退化为最小 bootstrap rendezvous(单机进程内;跨节点约定 rank0 节点发 id),fail-stop,
与运行时控制面无关。

## 7. 代价清单(诚实版)

- **Prefill 延迟税还在**:fleet step 时间 = 最慢 rank,rank A 跑 prefill span 时全员 TPOT
  变差。per-rank bucket 删掉的是算力税(别人不再陪跑 bucket-8),延迟税是 EP 物理。
  **这正是本架构与 P/D 互相成就之处**:P/D decode fleet 没有 prefill,step 时间天然均匀
  ——架构最成立的部署形态恰好是 ep16 P/D decode 端,而 ep16 P/D 也只有在此架构下不需要
  Event plane。
- **Debug 变难**:单状态机 + contract tests 是现有资产;N 个独立状态机后,交织类问题复现
  变难。缓解:决策核心(admission/plan/slot)本就是纯函数,per-rank 复用后 contract tests
  原样保留;`cross-node-scaling.md` SMR 章节的 replay journal 片段单独实现,per-rank 挂
  本地 journal。
- **Load 不均从调度问题变 router 问题**:`lessons/moe-dplb-decode-imbalance.md` 已预言
  ——engine 吐原始 progress,router 负责均衡。
- **N 份 HTTP/tokenizer**:小钱,权重本来就 per-rank。

## 8. Go/no-go kernel gates(先于任何架构代码)

**结果(2026-07-30,GB300 tray03 单 tray 4 GPU,`susun-dev`,commit `16d95344`):
gate 1–3 全部 GO。** 前三个 gate 实现于 `pegainfer-glm52/src/oracle/freerun_ep4.rs`,
按 EP4 形状写(一个 GB300 NVL72 tray = 4 GPU,走 weight-only 链——正是 NVL72 上的
生产链)。运行(**每个 gate 必须单独一个进程**,见下面的 pitfall):

```bash
for g in freerun_hetero_traffic_gate freerun_hetero_graph_gate freerun_bound_tax_probe; do
  PEGAINFER_TEST_MODEL_PATH=/mnt/shared/weights/GLM-5.2-FP8 EP_DISABLE_GIN=1 \
    cargo test --release -p pegainfer-glm52 --lib "$g" -- --ignored --nocapture
done
```

1. **`freerun_hetero_traffic_gate` — 跨 rank 流量不变性。✅ PASS。** 同一组 DeepEP
   context 跑两遍 layer-6 oracle walk:pass A 旁路 rank 全 token-less,pass B 旁路
   rank 每 position 推 0..=8 变化 token 数。验收:两遍都过 oracle probes,且 rank 0
   两遍输出逐值 bit-identical。实测:quiet 与 hetero 各 63/64 probes(同一个已知
   router tie-flip outlier,与既有 EP4 oracle gate 一致),200×6144 个输出值零 bit
   抖动——"一个 rank 的行的计算与别人的流量无关"成立。
2. **`freerun_hetero_graph_gate` — 异构 token 数的 graph 回放。✅ PASS。** 4 个 rank
   各以不同 token 数(1/2/4/8)capture routed 链的 CUDA graph 并回放 16 次,每次
   combined 输出与 eager 参考 bit-identical。whole-step graph(含 attention/采样)的
   同类验证留到迁移第 1 步的 e2e gate。
3. **`freerun_bound_tax_probe` — 保守 bound 的性能税。✅ GO,税 = 0。** 每 rank
   1 token 的 steady-decode 形状,256 次均值:tight(`global_tokens=4`)180.9 µs/层,
   protocol-max(=32)180.2 µs/层——**差异在噪声内,方向还是反的**。整条 weight-only
   链对 `global_tokens` 不敏感(它只收紧 tiles kernel 的扫描上界,GEMM 工作量由
   device 侧 psum 决定)。原判读标准(≤0.5ms/step → go)以最强形式满足,"per-rank
   静态 bound 档位"退让方案不需要。EP16 复测仍保留(shim 常量不同),但 EP4 的零税
   使不同结论的先验概率很低。
4. **`freerun_padding_byte_constancy_gate` — padding 字节恒定。✅ PASS(2026-07-30,
   tray03,commit `7f7f4b93`)。** EP4 引擎真实 launch:rank 0 跑一条 sampled 请求
   (sampled 挡住 launch-ahead lease,每步走完整 prologue——被测对象正是
   `GLM52_PADDING_STEP` 契约),rank 1–3 全程空转。每步从生产 step 路径 D2H 最后一层
   routed MoE 的 `topk_idx`/`topk_weight`(probe 挂在 `runner::step`,
   `freerun_probe.rs`)。实测:460 个 route 快照,空 rank 每 bucket 分组内字节逐步
   恒定——indexer seq_len=1 与 fp8 quant 两个此前未验证环节均为构造性确定。
   (leased replay 的 padding 行按设计自喂,其 wire 字节演化由 lease 不变量守护,
   不在本 gate 范围。)
5. **`freerun_mtp_fixed_chain_gate` — MTP 固定链。✅ PASS(同上)。** 固定链已实装:
   `select_round_kind` 与全局 bucket 协商已删除,每 rank 每 round 无条件跑 context +
   4 个 proposal forward,空 rank 以 padding 进场(零 append 的 round 显式清零
   `previous` padding 行——capture buffer 残值不上 wire)。两阶段:A 阶段 rank 0 独自
   decode(rank 1–3 空,每 round 三个全 padding rank),B 阶段四 rank 全忙且 rank 0
   重复同一请求。实测:rank 0 两阶段轨迹逐 token 相同(gate 1 流量不变性的 whole-step
   版);空 round 开销 hetero 3.725 vs busy 3.809 ms/round,**delta −0.084 ms——
   在噪声内,验收线 0.5 ms 以最强形式满足**。

**Pitfall(实测踩中):DeepEP context 是一进程一次性的。** 三个 gate 在同一个 test
进程串行时,第二个 gate 的 `ctx_create` 撞 NVLink barrier timeout →
`unspecified launch failure`——与 rank-host 契约记录一致("worker drop does not
return all hosted GPU state; process exit is the release mechanism")。gate 必须
每个单独一个 `cargo test` 进程。这也是 free-running 架构的一条部署事实:engine
进程的生命周期 = DeepEP context 的生命周期,重启即换进程。

## 9. 代码映射(删多于加)

| 现在 | 之后 |
|---|---|
| ~~`run_dp8_coordinator`(14 参数,持全 fleet 状态)~~ | **已拆**:`Glm52Engine` per-rank 自治 loop(`scheduler/mod.rs`),`Vec<RankSlots>` → `RankSlots`,`for rank` 循环消失;EP 无条件全速 step,mirrored TP 是 N=1 退化(保留空闲阻塞) |
| ~~`plan_step_shapes`(hungriest rank)~~ | **已降维**:`plan_step_shape(&my_wants)`;函数仍纯,contract tests 原样迁移 |
| ~~`launch_ahead_flags`(all-ranks-or-none)+ stale-replay 重跑~~ | **已本地化**:`lease_flags` always-consume——lease 冻结 slot 集合一步(finish/断连延迟物理释放、新请求延迟一步 admit),`consume` 从决策变结构保证;空 rank 显式禁 lease(修掉 vacuous-true 下 padding row `slot_mapping` 漂出 padding page 的漏洞) |
| ~~`select_round_kind` + MTP 全局 bucket/ensure~~ | **已删**(固定 5-forward 链,迁移第 1 步) |
| ~~`remote.rs`(978 行)+ rank-host + `--rank-hosts`~~ | **已退役**:跨节点 = 同一 binary 每节点一进程加入同一 DeepEP communicator;新增一次性 bootstrap rendezvous(`rendezvous.rs`,rank0 进程分发 unique id,`--glm52-ranks 4..8` + `--glm52-rendezvous`) |
| ~~`VllmPdState`/`NativePdState`(coordinator 持有) | **已本地化**:engine 本地字段;parked 5ms sleep 节流消失(无条件 step 提供重试 cadence) |
| 请求路由(coordinator intake 绑秩) | **已上移**:`EngineHandle` 多 submit channel,按 `data_parallel_rank` 路由,无标请求 least-load(4x waiting)读 load watches;HTTP 端点先复用每进程单 endpoint(客户端可显式 pin rank),每 rank 独立 endpoint 待 router 对接时再做 |
| mirrored TP8/TP4 | 原样(本就是 N=1 特例,骑同一 engine loop) |
| ~~`fail_step` 全局收尸~~ | **已去中心化**:rank 本地 fatal → error log + 本地请求发 Error + `process::exit(1)`;fleet 经 collective 超时传染 fail-stop(§6)。per-rank step watchdog 仍是后续项 |

## 10. 迁移路径

不 big-bang。gates 绿后两步:

1. **协议先变,结构后变(已实装并验证)**:保留 coordinator 的壳,协议全部
   per-rank 化——per-rank bucket、协议最大 `global_tokens`、MTP 固定链;launch-ahead
   lease/consume 暂保持全局。gate 4/5(§8)在 whole-step 路径上验收本步。
2. **拆壳(已实装,2026-07-30)**:coordinator 循环拆成 N 个 engine 线程,lease
   本地化(always-consume),rank-host 退役,bootstrap rendezvous 收编启动期协调。
   实现要点:
   - **请求入口**:`EngineHandle` 扩展为 per-rank 多 submit channel(`new_with_join_handles`),
     `submit()` 按 `data_parallel_rank` 路由;vLLM frontend 与 server 零改动
     (`frontend_engine_count` = 本进程 rank 数)。
   - **启动屏障**:各 engine 按固定 bucket 顺序各自 precapture——collective 天然
     rendezvous;bootstrap 结果逐个上报 launcher,任一失败关闸全 join,fail-stop。
   - **优雅退出**:channel 断开(bridge 已 abort 在途请求)→ 各 engine drain 后退出
     loop → 各自 flush offload + shutdown 自己的 worker(DeepEP destroy barrier 自然
     配对)→ handle join 全部线程。
   - **跨节点**:`--glm52-ranks <start..end>` 指定本进程托管的全局 rank 段,
     `--glm52-rendezvous <addr:port>` 是一次性 id 分发(rank0 进程绑定并幂等服务,
     其余连接拉取;范围错 tile → DeepEP `ctx_create` 挂住 → 超时 → 各方 error log +
     退出)。**注意 `max_model_len` 现为各节点本地 VRAM 推导,跨节点可能发散**——
     不进 collective、不炸协议,但多机部署应显式 `--max-model-len` 保持 fleet 口径一致。
   - **KV offload 跨节点自然解锁**:每节点只注册本地 arena 到自己的 pegaflow host
     (老 blocker "remote arena 指针过不了 wire" 随 rank-host 一起消失),namespace
     推导确定、各节点一致。**native MTP 的单机限制已解除**(round 本就是固定链 +
     rank-local bucket,跨进程配对机制与目标步同构;双 tray EP8+MTP 已硬件验证,见
     Next step)。
   验收:全量单测 + clippy(本机,2026-07-30);双 tray EP8 rendezvous / 服务 /
   fail-stop 已验(见 Next step);EP4 五 gate 回归与 EP16 验证仍挂 GPU 执行。

## 与 cross-node-scaling.md 的关系

该文档的 NVL72 实测数据(EP4→EP32 bucket-1 p50 平坦、IMEX/teardown 坑)仍然有效且
load-bearing。被本设计取代并已随第 2 步删除的部分:framed-TCP rank-host 作为
**长期架构**(短期已 shipped 可用,2026-07-30 从树上删除)、Event plane(facts plane)
设计、SMR coordinator 方向。replay journal 片段被本设计吸收(第 7 节)。

## Next step

迁移两步均已实装。挂 GPU 的验收序列:
1. **EP4 五 gate 回归**(tray03):拆壳后单机路径不变的证明——逐 gate 单进程跑 §8 命令。
2. **e2e/golden 回归**:qwen 系 golden gate 与 glm52 既有 e2e。
3. **双 tray 首验:EP8 已完成(2026-07-30,tray03 ranks 0..4 + tray13 ranks 4..8,
   commit `22d7d047`)。** rendezvous 一次成功(peer checkin / fetch attempt 1 均有
   日志);两进程各 4 engine 起服务,两个 HTTP endpoint 的 greedy 输出字节一致;多进程
   native MTP 正常运行(56 round,mean_accepted_drafts=1.196,两侧进程轨迹一致——
   `is_mtp → !multi_process` 限制已随 `6549b9be` 解除)。fail-stop 实测:杀 tray13
   进程后,无流量时幸存进程不自知(per-rank watchdog 未做,见第 4 项),首个新请求
   触发 DeepEP NVLink barrier 超时(~17s)→ `ERROR ... rank 0 fatal; the engine
   process exits` → 进程退出,curl 空回——与 §6 设计一致,检测目前由流量触发。
   本轮修出一个多进程专属 bug:weight-load/build 的错误标签与校验把局部 worker 下标
   当全局 rank 用(`ranks.start != 0` 必炸),`22d7d047` 修复。**EP16 仍待首验**,
   连同 bound-tax 复测(EP16 shim 常量不同,§8 gate 3 注记)。
4. per-rank step watchdog(§6 的去中心化收尸的最后一块——实测确认无流量时 peer
   死亡不会被察觉)与每 rank 独立 HTTP endpoint(router 对接时)仍是后续项。
