# Kimi-K2 Data-Parallel Decode Design

> **TL;DR:** 可配置 TP×DP 并行。每个 DP rank 是独立的 decode engine——自己持有 request、KV cache、slot，自己回复 client。唯一的跨 rank 耦合是 MoE 层的 PPLX EP all-to-all，它同时充当天然 sync point，不需要显式 barrier 或中央 scheduler。前面放一个轻量 DP load balancer 做 request 路由。PPLX variable-batch（含空 rank）是 P0 硬 contract。首批支持 TP8×DP1（现状）+ TP1×DP8（新增）。
>
> **Last touched:** 2026-05

## Motivation

当前 TP8+EP8 单 token decode 每步做 123 次 NCCL all-reduce + 60 次 reduce-scatter。graph-replay 下 bs4 TPOT 14.39ms（~278 tok/s），strong-sync 下 ~35ms。Collective cadence 是 graph-replay 下 tail 的主要来源。

TP1+DP8+EP8 的核心收益：
- **Attention / dense / shared expert 零通信**。123 次 all-reduce 全部消失。
- **8 条 DP lane 并行上限**。8 个 DP rank 各自服务独立 request batch。实际吞吐是 collective 删除收益和 TP1 full GEMM 增量的净效果，不是线性 ×8。
- **PPLX EP 已验证**。bs=1 PPLX decode TPOT 17.94ms，超过 NCCL no-graph 18.52ms（见 [pplx-ep-decode.md](pplx-ep-decode.md)）。

vLLM H20 baseline（TP1+DP8+EP8）：bs=8 sweet spot TPOT 26.4ms / 308 tok/s，bs=256 峰值 1131 tok/s（见 [vllm-h20-baseline.md](vllm-h20-baseline.md)）。

---

## 1. Parallel Config

### 不变量

```
total_ranks = tp_world × dp_world = N  (单机 N=8)
ep_world    = N                        (全部 rank 参与 EP)
local_experts = 384 / ep_world         (按 ep_world 切)
```

### 数据结构

```rust
/// 纯并行拓扑，跟模型无关。可复用于 Qwen 等其他模型线。
/// 放 pegainfer-core。
pub struct ParallelConfig {
    pub tp_world: usize,
    pub dp_world: usize,
    pub ep_world: usize,       // = tp_world × dp_world
}

/// 一个 rank 在 TP×DP×EP 网格中的坐标。
/// 放 pegainfer-core。
pub struct RankCoord {
    pub global_rank: usize,
    pub tp_rank: usize,        // global_rank % tp_world
    pub dp_rank: usize,        // global_rank / tp_world
    pub ep_rank: usize,        // = global_rank
}

/// Kimi-K2 专属：从拓扑派生的模型维度。
/// 现有 KimiK2ParallelShape 的延续，留在 pegainfer-kimi-k2。
pub struct KimiK2ModelConfig {
    pub topo: ParallelConfig,
    pub heads_per_tp: usize,   // = 64 / tp_world
    pub local_experts: usize,  // = 384 / ep_world
    pub vocab_per_tp: usize,   // = 163840 / tp_world
}
```

TP group 从 `ParallelConfig` 直接算：group `g` = `[g * tp_world .. (g+1) * tp_world)`。不需要单独的 `RankGroups` struct（`ep_group` 恒等于全部 rank，冗余）。

### 支持的形态

| 形态 | tp_world | dp_world | TP 组 | 独立 engine 数 | TP all-reduce | EP all-to-all |
| --- | --- | --- | --- | --- | --- | --- |
| TP8 DP1 | 8 | 1 | {0..7} | 1 | 每层 attn + shared + dense | PPLX / NCCL RS |
| TP1 DP8 | 1 | 8 | 各 1 rank | 8 | **无** | PPLX |
| TP2 DP4 | 2 | 4 | {0,1}, {2,3}, ... | 4 | TP2 组内 | *design candidate, not v1* |
| TP4 DP2 | 4 | 2 | {0..3}, {4..7} | 2 | TP4 组内 | *design candidate, not v1* |

首批实现：**TP8×DP1**（现状）+ **TP1×DP8**（新增）。TP2/TP4 后续扩展（见 [§7](#7-extension-to-tp1)）。

---

## 2. Per-DP-Rank Decode Engine（主设计）

### 核心思想

**每个 DP rank 是一个独立的 decode engine。** 自己持有 request pool、KV cache、decode slot，自己做 prefill，自己跑 decode loop，自己回复 client。不存在中央 scheduler 构造 wave 再下发。

唯一的跨 rank 耦合：MoE 层的 PPLX EP all-to-all。它同时是 **天然 sync point**——各 rank 独立跑 decode loop，在每个 MoE 层的 `combine_recv` 处自动对齐，不需要显式 barrier。

```
                    ┌────────────────────┐
                    │  DP Load Balancer  │
                    │  (轻量 request 路由)│
                    └──────────┬─────────┘
                               │ route request → dp_rank
              ┌────────┬───────┼───────┬────────┐
              ▼        ▼       ▼       ▼        ▼
           DP Rank 0  DP Rank 1  ...          DP Rank 7
           ┌────────┐ ┌────────┐              ┌────────┐
           │slots   │ │slots   │              │slots   │
           │KV cache│ │KV cache│              │KV cache│
           │decode  │ │decode  │              │decode  │
           │loop    │ │loop    │              │loop    │
           │reply → │ │reply → │              │reply → │
           │client  │ │client  │              │client  │
           └────┬───┘ └────┬───┘              └────┬───┘
                │          │                       │
                └──────────┴───────────────────────┘
                     EP all-to-all (天然 sync point)
```

### Per-rank State

```rust
/// 每个 DP rank 自己维护的 decode state。
/// TP1: 1 个 rank 1 个 engine。
/// TP8: 1 个 TP group 共用 1 个 engine（dp_world=1）。
struct DpRankEngine {
    dp_rank: usize,
    slots: Vec<Option<RequestState>>,   // len = max_batch_per_dp
    kv_pool: KvPool,                    // 本 rank 独立的 paged KV pool
}

struct RequestState {
    slot: usize,
    token_tx: mpsc::UnboundedSender<TokenEvent>,
    prompt_len: usize,
    completion_tokens: usize,
    max_tokens: usize,
    last_token: u32,
}
```

Request state 和 KV cache 完全是 per-DP-rank 的。跨机时没有人试图分配其他机器的 KV。

> **TODO（work stealing）**：v1 不做运行时 KV 迁移。理论上 DP rank 间 load 不均时可以迁移 request + KV，但代价高、收益有限。标注为未来可选项，详见 [moe-dplb-decode-imbalance.md](../../lessons/moe-dplb-decode-imbalance.md)。

### Decode Loop

每个 DP rank 独立跑 decode loop：

```rust
// 每个 DP rank 独立运行（概念伪码）
fn dp_rank_decode_loop(engine: &mut DpRankEngine, executor: &mut impl ForwardExecutor) {
    loop {
        // 1. 确定本轮 active rows
        let active = engine.active_requests();
        let active_rows = active.len();

        // 2. 即使 active_rows == 0 也必须跑——EP 要求所有 rank 参与
        //    用 padding row 或 PPLX 零 token 语义（见 §3）
        let forward_rows = if active_rows == 0 {
            // 空 rank: padding 1 行 dummy，sampling 时 skip
            build_padding_rows()
        } else {
            build_active_rows(&active)
        };

        // 3. Forward: attention → MoE (EP sync) → ... → sampling
        //    MoE 层的 EP all-to-all 天然对齐所有 rank
        let results = executor.forward_decode_step(&forward_rows);

        // 4. 处理结果（空 rank skip）
        if active_rows > 0 {
            for (req, next_token) in active.iter_mut().zip(results) {
                req.completion_tokens += 1;
                req.last_token = next_token;
                let _ = req.token_tx.send(TokenEvent::Token { id: next_token, logprob: None });
                // EOS / max_length → retire
            }
        }

        // 5. 是否继续？见 idle coordination（§2.4）
    }
}
```

### EP 天然 Sync

各 rank 独立跑 decode loop，不需要显式 step barrier。同步由 MoE 层的 EP all-to-all 保证：

- Layer 0（dense）：各 rank 独立算，无同步。可能有微小时间漂移。
- Layer 1（第一个 MoE 层）：`dispatch_send` 非阻塞，但 `combine_recv` 必须等全部 rank 的 `combine_send`。**这里自然拉齐所有 rank。**
- Layer 2..60：每个 MoE 层再次对齐。

Layer 0 到 layer 1 之间的漂移来源：各 rank 的 attention seq_len 差异 + batch_size 差异。对 decode 单 token query，FlashInfer batch decode latency 差异 ~10s μs，可忽略。

### Idle Coordination

当所有 DP rank 都没有 active request 时，rank 不应空转（浪费 GPU + EP 带宽）。需要一个轻量机制让 rank 知道"全局是否有活"：

**方案：host-side shared atomic + barrier**

```rust
// 所有 rank 共享（单机场景，Arc<AtomicUsize>）
static GLOBAL_ACTIVE_COUNT: AtomicUsize;

// 每个 decode step 开始前
fn should_continue(engine: &DpRankEngine) -> bool {
    let my_active = engine.active_count();
    GLOBAL_ACTIVE_COUNT.fetch_add(my_active, Ordering::Relaxed);
    barrier.wait();  // 等全部 rank 到齐
    let total = GLOBAL_ACTIVE_COUNT.load(Ordering::Relaxed);
    barrier.wait();  // 等全部 rank 读完
    GLOBAL_ACTIVE_COUNT.store(0, Ordering::Relaxed);
    total > 0
}
```

如果 `total == 0`，全部 rank 进入 idle wait（condvar / channel），等 load balancer 送来新请求后唤醒。

跨机时可改成 NCCL all-reduce 1 个 int，或 PPLX sideband signal——但这是 multi-node 扩展的问题，单机先用 shared atomic。

### TP8 DP1 退化

DP1 时只有 1 个 DP rank（= 全部 8 个物理 rank 组成的 TP group）。整个 engine 跟现有 `KimiK2Scheduler` 行为一致：所有 rank 处理同一 batch，TP all-reduce 在每层执行。**现有 TP8 路径不删除，作为 `dp_world=1` 的特例保留。**

---

## 3. PPLX Variable-Batch Contract（P0）

各 DP rank 独立管理自己的 batch，EP 成立的前提是 PPLX 能处理 per-rank 不同数量的 active rows。

### 硬 Contract

1. **每个 rank 每 step 有 `0..max_batch_per_dp` 个 active rows。** Rank 之间的 active_rows 可以不同。
2. **`active_rows = 0` 的 rank 必须仍然调用 PPLX dispatch/combine API**——它可能不发送自己的 token，但必须参与 collective（接收其他 rank dispatch 过来的 token 做本地 expert 计算，以及 combine 回去）。如果 PPLX 不支持 `seq_len=0`，则 rank 必须发送 padding row（1 个 dummy token，sampling 时忽略）。
3. **combine_recv 只写 active rows 的位置。** Padding row 不进入 sampling。如果用 padding row 实现空 rank，sampling 阶段 skip 该 row。
4. **所有 rank 每层调用 PPLX API 的次数和顺序完全一致。** 60 个 MoE 层，每层恰好一组 `dispatch_send → dispatch_recv → expert GEMM → combine_send → combine_recv`。不允许某些 rank skip 或多调一次。

### Capacity Contract

每个 rank 的 PPLX scratch **不能按本 rank `active_rows` 估算**。空 rank（`active_rows=0`）本地不产 token，但仍持有 48 个 expert，其他 rank 的 token 可能路由到这里。所有 buffer 按全局 worst case 分配：

```
max_global_rows       = dp_world × max_batch_per_dp
max_route_rows        = max_global_rows × topk
max_padded_route_rows = max_route_rows + local_experts × (expert_padding - 1)
```

| Buffer | 大小依据 |
| --- | --- |
| recv_hidden (bf16) | `max_route_rows × hidden_dim` |
| expert-major routing metadata | `max_padded_route_rows × hidden_dim` |
| Marlin W13 out / activated | `max_padded_route_rows × expert_intermediate` |
| expert output (f32) | `max_padded_route_rows × hidden_dim` |
| combine send (f32) | `max_route_rows × hidden_dim` |
| send_hidden (bf16) | `max_batch_per_dp × topk × hidden_dim`（本 rank 发出）|

`max_padded_route_rows` 的 padding 尾巴来自 Marlin block_size 对齐。`expert_padding` 影响 PPLX recv/combine layout、Marlin routing/GEMM 实际处理 rows、scratch capacity 和 CUDA Graph shape；它不是模型语义常量，应作为 runtime tuning knob。当前 decode 小 batch 默认先用 `8`，后续调性能时再 sweep，不应长期写死成全局 const。

### 需要验证的点

- PPLX `dispatch_send` 当 `num_tokens=0` 时的行为：是否合法？是否需要传空 buffer？
- 如果不支持零 token，padding 策略：每 rank 始终 pad 到 `max(1, active_rows)`，dummy token 的 hidden state 为全零，expert 计算完直接丢弃

### 空 rank 场景

DP8 下只有 3 个请求分布在 3 个 rank，其余 5 个 rank 为空。空 rank：
- Attention/dense/shared: 无 compute（0 或 1 padding row）
- MoE: 必须参与 dispatch/combine（持有 48 个 expert，其他 rank 的 token 可能路由到这里）
- Sampling: 无 output（skip padding row）

这是 DP 的常态工况，不是边界 case。

---

## 4. DP Load Balancer

极轻量。只做 request → dp_rank 路由，不持有 request state，不参与 decode。

```rust
/// 路由策略：选空 slot 最多的 DP rank。
fn route_request(engines: &[DpRankEngine]) -> usize {
    engines
        .iter()
        .enumerate()
        .max_by_key(|(_, e)| e.free_slot_count())
        .map(|(dp_rank, _)| dp_rank)
        .unwrap()
}
```

Request 一经路由，整个交给目标 DP rank。Load balancer 不再跟踪该 request。

跨机时 load balancer 可以是一个独立 proxy 进程（跟 [DPLB lesson](../../lessons/moe-dplb-decode-imbalance.md) 里的外部 router 对上），只需要知道各 rank 的空 slot 数。

---

## 5. Scope 分层

### v1 Runtime Target: Decode Only

每个 DP rank engine 的 runtime 职责：**接收 request（含已 prefill 的 KV），跑 decode loop，返回 token。** KV cache 在 decode 开始前必须就绪。

### Server Integration: Load Balance + Prefill + Decode

端到端 server 生命周期：

```
HTTP Request
  → DP Load Balancer（选 dp_rank）
  → 目标 DP rank 接管：
      → 分配 slot，开始 prefill
         （TP1: 单 rank 独立执行；TP8: 整个 TP group 协作）
      → prefill 完成，KV 就绪
      → 进入 decode loop（循环直到 EOS/max_length）
      → 直接回复 client
```

Prefill 的 TP 协作方式由 executor 决定。关键约束：**prefill 完成后才能进入 decode loop**。

---

## 6. Executor 边界

### 设计原则

不在 forward 里散射 `if tp_world == 1`。TP8×DP1 和 TP1×DP8 的 forward 差异封装在 executor 里。每个 DP rank engine 持有一个 executor。

### Forward Executor

```rust
trait ForwardExecutor {
    /// 单个 decode step 的 forward。
    /// 接收本 DP rank 的 rows，返回 next token ids。
    /// TP1: 单 rank forward，直接 full-vocab sampling。
    /// TP8: 多 rank forward + TP all-reduce + cross-rank logit merge。
    fn forward_decode_step(&mut self, rows: &DecodeRows) -> Result<Vec<u32>>;
}
```

两个实现：

**`Tp8Dp1ForwardExecutor`**（现有路径封装）
- 8 个 rank worker 收到相同 batch → forward + TP all-reduce
- MoE routed: PPLX dispatch/combine 或 NCCL RS bridge
- Sampling: 8 个 rank 各出 vocab shard logit → cross-rank max-logit merge

**`Tp1Dp8ForwardExecutor`**（新增）
- 单 rank forward，零 TP communication
- MoE routed: PPLX dispatch/combine（必选）
- Sampling: full-vocab logit，直接 top-1

### Executor ↔ Worker 边界

Executor 不直接操作 GPU。通过 `KimiRankWorker` 的 command channel 驱动 rank worker：

```
ForwardExecutor                 Worker (per-rank thread)
    │                                │
    │  ForwardDecodeStep {           │
    │    active_rows, token_ids,     │
    │    positions, slots            │
    │  }                             │
    ├───────────────────────────────►│
    │                                ├── forward kernels
    │                                ├── (MoE: PPLX EP sync)
    │  Vec<next_token_id>            │
    │◄───────────────────────────────┤
```

TP1 executor 给 1 个 worker 发 rows。TP8 executor 给 8 个 worker 发相同 rows + 聚合结果。Worker 不感知 DP 语义。

### 代码组织

现有 `runner/` 结构（来自 [source-layout.md](source-layout.md) 的 worker 拆分）：

```
runner/
├── scheduler.rs        ← 现有 TP8 scheduler
├── worker.rs           ← rank worker spawn/command
├── worker/
│   ├── state.rs        ← rank thread state
│   ├── cache.rs        ← KV arena
│   ├── forward.rs      ← decode forward kernels（TP8 路径）
│   ├── load.rs         ← weight loading
│   └── runtime.rs      ← collectives, RoPE, sampling
├── moe_pplx.rs         ← PPLX EP forward
├── affinity.rs         ← thread pinning
└── config.rs           ← runner config
```

改造后的目标结构：

```
runner/
├── config.rs                ← ParallelConfig, KimiK2ModelConfig
├── engine.rs                ← DpRankEngine, decode loop, idle coordination
├── load_balancer.rs         ← DP request routing
├── executor/
│   ├── mod.rs               ← ForwardExecutor trait
│   ├── tp8_dp1.rs           ← Tp8Dp1ForwardExecutor（现有 TP8 路径）
│   └── tp1_dp8.rs           ← Tp1Dp8ForwardExecutor（新增）
├── worker.rs                ← rank worker spawn/command（保留）
├── worker/
│   ├── state.rs             ← rank thread state（保留）
│   ├── cache.rs             ← KV arena（保留）
│   ├── forward.rs           ← forward kernels（保留，被 executor 调用）
│   ├── load.rs              ← weight loading（保留）
│   └── runtime.rs           ← collectives, RoPE, sampling（保留）
├── moe_pplx.rs              ← PPLX EP forward（保留）
└── affinity.rs              ← thread pinning（保留）
```

Worker command 边界保持，内部 forward/cache 会按 executor 路径扩展（TP1 的 full-head MLA scratch、full vocab embedding/lm_head、PPLX 空 rank 语义都会碰 worker/cache/forward）。

---

## 7. Extension to TP>1

TP2 DP4 和 TP4 DP2 引入一个新问题：**TP 组内的 rank 在 MoE 层前有相同的 hidden states（经过 attention all-reduce 后），但 EP all-to-all 需要避免重复 dispatch。**

### 方案选项

**A. EP world = dp_world（TP 组内 MoE replicated）**

- EP 只在 DP 维度做：每 TP 组选一个 primary rank 参与 EP
- 组内其他 TP rank 在 MoE 层 idle（或做 shared expert）
- 优点：简单，EP communicator 跟 TP 解耦
- 缺点：浪费 expert memory（TP 组内 expert 权重 replicated）

**B. EP world = N（全 rank 参与，expert 权重按全局 rank 切）**

- 每 rank 拥有 384/N 个 expert
- TP 组内的 2 个 rank 拥有不同的 expert
- Dispatch 时需要特殊处理避免重复
- 优点：expert memory 不浪费
- 缺点：dispatch 语义更复杂

**C. 分阶段混合**

- Attention: TP all-reduce within TP group
- MoE: 先 TP all-reduce 得到完整 hidden，再由 primary rank 做 EP dispatch/combine，最后 broadcast 回 TP 组

Recommendation：首批只做 TP1 DP8 和 TP8 DP1。TP2/TP4 的 executor 独立实现，MoE-TP 交互在有真实性能数据后再选方案。

---

## 8. Forward Path Reference

保留作为 executor 实现的参考。

### TP1 Decode Forward (Layer 1..60 MoE)

```
RMSNorm input
  → fused_qkv_a GEMM [7168 → 2112]
  → kimi_mla_split_qkv_a
  → RMSNorm(q_a)
  → q_b GEMM [1536 → 12288]                     ← TP1: 完整 12288（TP8: 1536）
  → RMSNorm(compressed_kv)
  → kimi_mla_rope_split_decode                   ← 64 heads（TP8: 8 heads）
  → kimi_mla_absorb_q_nope                       ← 64 heads
  → kimi_mla_paged_kv_append
  → kimi_flashinfer_batch_decode_mla             ← 64 heads
  → kimi_mla_v_up                                ← 64 heads
  → o_proj GEMM [8192 → 7168]                   ← TP1: 完整 GEMM
  → [TP1: 无 all-reduce] [TP8: all-reduce]
  → residual add + post-attention RMSNorm
  → kimi_router_noaux_tc
  ┌───────────────────────┐              ┌──────────────────────────────────┐
  │ shared expert path    │              │ routed expert path (PPLX EP)     │
  │ shared gate/up GEMM   │              │ PPLX dispatch_send               │
  │   [7168 → 4096] 完整   │              │ PPLX dispatch_recv               │
  │ silu_mul_fused        │              │ Marlin W13 + SwiGLU + W2          │
  │ shared down GEMM      │              │ PPLX combine_send                │
  │   [2048 → 7168] 完整   │              │ PPLX combine_recv                │
  │ [TP1: 无 all-reduce]  │              │ [TP1: 无 RS bridge]              │
  └───────────┬───────────┘              └──────────────┬───────────────────┘
              │                                          │
              └────────────► kimi_scaled_add ◄───────────┘
```

### TP1 vs TP8 对比

| 当前调用 (TP8) | 次数/step | TP1 后 |
| --- | --- | --- |
| `all_reduce_in_place` (BF16-via-F32) | 123 | 全部删除 |
| `reduce_scatter` (routed combine) | 60 | 全部删除 |
| `repeat_f32_for_reduce_scatter_into` | 60 | 全部删除 |
| `embedding_batch_vocab_shard` + all-reduce | 1 | → `embedding_batch` (full vocab) |
| logits all-gather | 1 | 删除（full vocab 直接 sampling） |
| **Total removed** | **245 calls/step** | |

### Weight Distribution

| 组件 | Per-rank TP8 | Per-rank TP1 | 变化 |
| --- | --- | --- | --- |
| Attention (q_b, kv_b, o_proj × 61) | ~3.2 GB | ~12.2 GB | TP shard → full |
| Shared expert (× 60) | ~0.7 GB | ~4.9 GB | TP shard → full |
| Dense + embed + lm_head | ~0.7 GB | ~5.0 GB | TP shard → full |
| RMSNorm + replicated | ~1.9 GB | ~1.9 GB | 不变 |
| Routed experts (48 × 60, INT4) | ~63 GB | ~63 GB | 不变 (EP8) |
| **Total** | **~69 GB** | **~87 GB** | +18 GB |

H200 (141 GB) 剩 ~54 GB 给 KV + scratch。

---

## 9. Correctness Gate

跟 TP8 现有标准一致：

1. vLLM K2.5 fixture（27 tok prompt）`max_tokens=1/2/8/16` greedy token ids 一致
2. 多 prompt vLLM gate（hello / math_short / self_intro_zh / code_rust）argmax match
3. 每个 DP rank 独立过 greedy parity（不同 rank 可以跑不同 prompt）
4. **Variable-batch 专项**：不均匀 active_rows pattern `[4, 0, 2, 1, 0, 3, 0, 1]`，验证：
   - 所有非空 rank 的 token 与 vLLM / TP8 baseline 对齐
   - 空 rank 无 output 但仍正常完成 EP collective
   - 非空 rank 的结果不受空 rank 存在的影响（同一 prompt 在满载 vs 有空 rank 时 output 一致）

---

## 10. Open

1. **Prefill bridge 实现**：v1 runtime 不含 prefill。Server integration 需要在 owning DP rank（TP1: 单 rank；TP8: TP group）上执行 prefill 后进入 decode loop。
2. **CUDA Graph**：TP1 路径需要新的 graph topology（无 NCCL，有 PPLX）。PPLX graph 兼容性独立跟进。
3. **Multi-node DP16 EP16**：单机跑通后扩展。Attention 不跨机，EP all-to-all 需 RDMA。PPLX 已支持多节点。Idle coordination 从 shared atomic 改成 NCCL all-reduce 或 PPLX sideband。
