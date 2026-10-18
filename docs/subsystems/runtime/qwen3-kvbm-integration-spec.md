# Qwen3-4B kvbm-logical 接入 Spec

> **TL;DR**: 用 kvbm-logical 的 `BlockManager` + `SchedulableSequence` 替换 Qwen3-4B 当前的 `PagePool`/`KvState` 体系。逻辑层管生命周期，物理层仍然是现有的单块 GPU buffer + page-first layout。验通后直接搬 K2。
>
> Last touched: 2026-05

## 目标

- 用 kvbm-logical 的 type-state block lifecycle 替换手写的 `PagePool` + `OwnedPagePermit` 分配
- Scheduler admission 改为查询 `BlockManager::available_blocks()`
- 为后续 prefix caching（inactive pool 复用）留好接口，但本轮不启用
- 不改动 attention kernel（FlashInfer BatchPrefill / BatchDecode）
- E2E 测试全绿，TPOT 无退化（允许 ±3%）

## 不做

- 前缀缓存命中（`match_blocks` / `scan_matches`）
- 事件发布（`EventsManager`）
- Prometheus metrics 对接
- kvbm-physical（物理层仍用现有 `CudaSlice<bf16>` buffer）
- Speculative decode

---

## 当前架构

```
Frontend
  ↓ GenerateRequest
Scheduler (scheduler.rs)
  ├─ admit_deferred_requests():  available_pages - active_future_pages ≥ max_pages_needed?
  ├─ build_next_plan():          Prefill / Decode / Unified
  └─ execute_plan() → resolve_step() → apply_effects()
      ↓
Executor (executor.rs)
  ├─ RequestStateStore:  HashMap<RequestId, KvState>
  │   ensure_with() → pool.alloc()
  │   take_batch() / restore_batch() → KvState 暂借给 forward
  │   drop_request() → KvState::drop → OwnedPagePermit::drop → pages 归还
  ├─ LocalQwen3Lane:
  │   execute_prefill() → kv.ensure_capacity(prompt_len); kv.advance(prompt_len)
  │   execute_decode()  → kv.ensure_capacity(seq_len+1); kv.advance(1)
  └─ KvPool → PagePool → OwnedPagePermit (RAII free-list)
```

**关键观察**:
- `KvState` 同时持有逻辑状态（seq_len）和物理句柄（OwnedPagePermit → page indices）
- 物理 page index 直接用于 GPU metadata upload（CSR page_indices_d, last_page_len_d）
- Admission 在 scheduler 层按 max_tokens 预估，executor 层 ensure_capacity() 再做实际分配
- CUDA Graph 要求 buffer base pointer 和 metadata buffer pointer 在 capture 后不变

---

## 目标架构

```
Frontend
  ↓ GenerateRequest
Scheduler (scheduler.rs)
  ├─ BlockManager::available_blocks() ≥ blocks_needed?
  ├─ SchedulableSequence::schedule_prefill / schedule_decode
  └─ execute_plan() → apply_prefill / apply_decode → register/advance
      ↓
Executor (executor.rs)
  ├─ SequenceStore:  HashMap<RequestId, SchedulableSequence<()>>
  │   schedule 阶段分配 MutableBlock
  │   apply 阶段 stage → register → ImmutableBlock
  │   drop_request() → SchedulableSequence::release()
  ├─ BlockId == PageId (1:1，无翻译层)
  └─ KvBuffer: CudaSlice<bf16> (不变)
```

### 核心改动

| 组件 | 之前 | 之后 |
|------|------|------|
| 逻辑分配 | `PagePool::try_acquire_many` | `BlockManager::allocate_blocks` |
| Per-request 状态 | `KvState { permit, seq_len }` | `SchedulableSequence<()>` |
| 生命周期释放 | `OwnedPagePermit::drop` | `SchedulableSequence::release()` → blocks 归还 |
| Admission | `available_pages - future_pages ≥ max_pages` | `available_blocks ≥ blocks_needed` |
| GPU metadata | `kv.page_indices_i32()` | `seq.block_ids()` 直接作为 page indices（BlockId == PageId） |

---

## 分层设计

### Layer 1: kvbm-logical（照搬，不改）

- `BlockManager<()>` — `()` 作为 `BlockMetadata`（无 per-block 自定义数据）
- `SchedulableSequence<()>` — 管 token→block 映射、schedule/apply 状态机
- Builder 参数：
  - `block_count`: 对应当前 `num_pages`
  - `block_size`: 对应当前 `page_size`（Qwen3-4B 为 16）
  - `with_lru_backend()`: 最简单的 inactive 策略（本轮 inactive pool 不会被命中，但需要一个 backend）
  - `duplication_policy`: `Allow`

### Layer 2: BlockId == PageId（无翻译层）

kvbm-logical 的 `BlockId` 是 `usize`，范围 `0..block_count`。GPU kernel 需要 `i32` page index。

本轮 **BlockId == PageId**，直接 `block_id as i32`，不需要任何翻译数据结构：
- `block_count` == 物理 page 总数
- 每个 BlockId 对应 buffer 内的固定 offset（跟当前 `PagePool` 返回 `PageId(n)` 等价）
- 分配/释放只改逻辑状态，物理位置不变

**后续物理层**：KV offload（GPU→CPU→SSD）和 P/D disaggregation 走 pegaflow 的传输层（pegaflow-transfer RDMA、pegaflow-core 三级存储），不走 Dynamo 的 kvbm-physical。kvbm-logical 的逻辑层（block lifecycle、sequence tracking）与物理传输实现无关，可以对接任意物理后端。

### Layer 3: KvBuffer（改造 KvPool）

`KvPool` 拆成：
- **`KvBuffer`** — 只持有 `CudaSlice<bf16>` + `KvLayout`。不再有 `PagePool`。
- 保留 `KvLayout`、`padding_page_id`（为 CUDA Graph padding slot 留一个固定 block）。
- `KvDesc` 保持不变：kernel-facing metadata bundle，接收 page indices、seq_len、layout。

```rust
pub struct KvBuffer {
    buffer: CudaSlice<bf16>,
    layout: KvLayout,
    padding_block_id: BlockId,  // 预留给 CUDA Graph padding
}
```

---

## 详细改动清单

### 1. pegainfer-core: 改造 KvPool → KvBuffer

**文件**: `pegainfer-core/src/kv_pool.rs` → 重命名为 `kv_buffer.rs`

- 删除 `PagePool` 依赖，删除 `KvState`
- `KvPool::new()` → `KvBuffer::new()`: 分配 `num_blocks × page_stride` 的 GPU buffer
- 保留 `KvLayout`、`KvDesc`
- 新增 `KvBuffer::padding_block_id()` → 返回 block 0（约定 block 0 是 padding）
- `KvDesc` 不变：仍然接收 `&[i32]` page indices + seq_len

**文件**: `pegainfer-core/Cargo.toml`

- 添加 `kvbm-logical = { workspace = true }`

### 2. pegainfer-qwen3/src/executor.rs: SequenceStore 替换 RequestStateStore

**删除**:
- `RequestStateStore`
- `RequestStateBatch`
- `kv_state_refs()`
- `ensure_with()` / `take_batch()` / `restore_batch()`

**新增**:
```rust
use kvbm_logical::{BlockManager, SchedulableSequence, DecodeOutcome};

struct SequenceStore {
    sequences: HashMap<RequestId, SchedulableSequence<()>>,
}
```

**BlockId → page index**:

当 forward 需要 `KvDesc` 时（prefill/decode），从 `SchedulableSequence` 提取已分配的 block IDs，直接 cast 为 `i32` page indices（BlockId == PageId），构建 `KvDesc`。

```rust
fn build_kv_desc<'a>(
    seq: &SchedulableSequence<()>,
    buffer: &'a KvBuffer,
) -> KvDesc<'a> {
    let page_indices: Vec<i32> = (0..seq.assigned_blocks())
        .filter_map(|i| seq.inner().assignments().get_assigned(i))
        .map(|(block_id, _)| block_id as i32)
        .collect();
    KvDesc::from_parts(
        buffer.layout(),
        buffer.buffer(),
        &page_indices,
        seq.kv_position(),
    )
}
```

> `KvDesc::from_parts` 是新增的构造函数，接受 `Vec<i32>` 而不是 `&[PageId]`。

**Qwen3Executor 改动**:
- `kv_pools: Vec<KvPool>` → `kv_buffers: Vec<KvBuffer>`
- 新增 `block_manager: BlockManager<()>` — scheduler 和 executor 共享（单线程，不需要 Arc）
- `alloc_kv()` → 在 `SchedulableSequence::new()` 时创建
- `drop_request()` → `seq.release()` + remove from SequenceStore

### 3. pegainfer-qwen3/src/scheduler.rs: Admission 改用 BlockManager

**删除**:
- `pages_needed()`, `max_request_tokens()`, `max_active_tokens()`, `current_active_tokens()`, `active_future_pages()`
- 当前的 budget 逻辑

**替换**:
```rust
fn admit_deferred_requests(
    deferred: Vec<PendingRequest>,
    block_manager: &BlockManager<()>,
    block_size: usize,
    max_blocks_per_request: usize,
) -> AdmissionOutcome {
    let mut pending = Vec::new();
    let mut still_deferred = Vec::new();
    let mut rejected = Vec::new();

    for req in deferred {
        let max_tokens = req.prompt_tokens.len() + req.max_tokens.saturating_sub(1);
        let blocks_needed = max_tokens.div_ceil(block_size);

        if blocks_needed > max_blocks_per_request {
            rejected.push(req);
            continue;
        }

        if blocks_needed <= block_manager.available_blocks() {
            pending.push(req);
        } else {
            still_deferred.push(req);
        }
    }

    AdmissionOutcome { pending, deferred: still_deferred, rejected }
}
```

**注意**: `available_blocks()` 返回 reset + inactive 池的总量。kvbm-logical 内部的 `allocate_blocks` 做 atomic all-or-nothing，所以 admission 只需要检查总量，不需要手动扣减 budget。但当前的 scheduler 串行处理一批 deferred requests，每 admit 一个就需要 "预扣" blocks。两种做法：

**方案 A（推荐）**: Admission 仍然做 budget 减法（跟现在一样），但改成 blocks 计数：
```rust
let mut budget = block_manager.available_blocks();
// 减去已 active 的 future growth
for active_seq in active_sequences.values() {
    budget -= active_seq.remaining_blocks();
}
```

**方案 B**: Admission 只做简单检查，依赖 `schedule_prefill` 返回 `ScheduleError::AllocationFailed` 后 defer。更简单但 latency spike 更大（prefill 时才发现分配不了）。

### 4. Forward path: prefill / decode / unified

**`execute_prefill_on_lane`**:
```
之前: ensure_with → pool.alloc() → take_batch → kv.ensure_capacity → kv.advance → forward → restore
之后: seq.schedule_prefill(prompt_len) → build_kv_desc → forward → seq.apply_prefill(first_token)
```

**`execute_decode_on_lane`**:
```
之前: take_batch → kv.ensure_capacity(seq_len+1) → kv.advance(1) → forward → restore
之后: seq.schedule_decode() → build_kv_desc → forward → seq.apply_decode(token)
```

**`execute_unified_on_lane`**: 同理，prefill 和 decode 各自走 SchedulableSequence 的 schedule/apply。

**关键**: `schedule_prefill` / `schedule_decode` 在 BlockManager 上做 `allocate_blocks`。如果失败（不应该，因为 admission 已经保证了 budget），返回 `ScheduleError::AllocationFailed`，propagate 成 anyhow error。

### 5. batch_decode_buffers.rs: sync_paged_meta 适配

当前 `sync_paged_meta` 从 `&[&KvState]` 提取 page indices。改为接收 translated page indices：

```rust
pub fn sync_paged_meta(
    &mut self,
    requests: &[RequestKvMeta],  // { page_indices: Vec<i32>, seq_len: usize }
    padded_bs: usize,
    padding_page_id: i32,
) -> Result<AttentionPath> {
    // 跟原来逻辑一样，只是数据源从 KvState 变成 RequestKvMeta
}
```

### 6. CUDA Graph 兼容

- `KvBuffer` base pointer 不变 → graph capture 后仍然有效
- Padding slot 用 `block_manager` 预留的 `padding_block_id` → 通过 `page_table.translate()` 得到 padding page index
- metadata upload（page_indices_d 等）在 graph replay 前执行，pointer 不变 → 无影响

### 7. TP 支持

当前每个 rank 有独立的 `KvPool`（相同 layout，独立 buffer）。改造后：
- 每个 rank 有独立的 `KvBuffer`（不变）
- `BlockManager<()>` 只需要一个（逻辑层对所有 rank 统一分配 block ID）
- BlockId == PageId，rank 无关

---

## Block 0 Padding 约定

当前 `KvPool` 在构造时 acquire 1 page 作为 padding。kvbm-logical 下，等价操作：

```rust
// 构造 BlockManager 时 block_count = num_physical_pages
// 立即分配 block 0 作为 padding，永远不释放
let padding_blocks = block_manager.allocate_blocks(1)
    .expect("must have at least 1 block for padding");
let padding_block_id = padding_blocks[0].block_id();
// stage + register 让它进入 ImmutableBlock 状态，RAII 持有
let padding_immutable = block_manager.register_block(
    padding_blocks.into_iter().next().unwrap()
        .stage(SequenceHash::default(), block_size)
        .unwrap()
);
// padding_immutable 存在 KvBuffer 里，lifetime = KvBuffer lifetime
```

---

## 接入顺序

1. **pegainfer-core**: 改造 `kv_pool.rs` → `kv_buffer.rs`（保留 `KvLayout`、`KvDesc`，删 `PagePool` 依赖）。保留旧 `kv_pool.rs` 不删，Qwen3.5 等其他模型仍用旧路径。
2. **pegainfer-qwen3/executor.rs**: `RequestStateStore` → `SequenceStore`，引入 `BlockManager`。
3. **pegainfer-qwen3/scheduler.rs**: Admission 改为 block 计数。
4. **pegainfer-qwen3/{prefill,batch_decode,unified_forward}.rs**: Forward path 适配 `SchedulableSequence` + `KvDesc` from block IDs。
5. **pegainfer-qwen3/batch_decode_buffers.rs**: `sync_paged_meta` 接口改造。
6. **E2E 测试验证**: `cargo test --release -p pegainfer-qwen3 --test e2e`
7. **TPOT 回归测试**: bench 跑一遍确认无退化。

每一步编译通过 + 已有 UT 通过后再进入下一步。

---

## 验收标准

- [ ] `cargo check --workspace` 通过
- [ ] `cargo test --release -p pegainfer-qwen3 --test e2e` 全绿
- [ ] `cargo test --release -p kvbm-logical --lib` 全绿（port 没被改坏）
- [ ] 其他模型 crate 不受影响（仍然用 `KvPool`）
- [ ] 单请求 TPOT bench ±3% 以内

---

## 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| `SchedulableSequence` 的 schedule/apply 状态机 overhead | 每 decode step 多一层函数调用 | kvbm-logical 内部全是 in-place mutation，无 allocation；可 bench 验证 |
| BlockId == PageId 假设在 offload / P-D 时不成立 | 需要引入 block→physical 映射层 | 对接 pegaflow 的物理传输；kvbm-logical 的逻辑层不受影响 |
| `available_blocks()` 获取需要锁 BlockStore mutex | Scheduler 热路径上多一次 mutex acquire | 单线程 scheduler，无竞争；实测 uncontended mutex < 50ns |
| 保留旧 `kv_pool.rs` 导致两套并存 | 维护成本 | 其他模型迁移后统一删除 |
