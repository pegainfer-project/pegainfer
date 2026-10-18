# KV Cache Management Design

**TL;DR**: 生产级 paged KV cache。Dynamo 式 logical/physical 分层：BlockManager 管逻辑生命周期和 admission，PhysicalBackend 管 GPU 内存和布局。Block 是管理单元，有状态机。Scheduler 只看 BlockManager，不碰物理层。

**Last touched**: 2026-05

---

## 调研驱动

三个系统的核心 takeaway：

**Dynamo (NVIDIA)** — 最重要的参考

- **Logical/physical 分层**：`kvbm-logical` 管 block identity、ownership、state machine、registry；`kvbm-physical` 管 GPU 内存、RDMA 传输、layout。两层独立演化。这是最值得采纳的设计。
- **Block 状态机**：`Reset → Partial → Complete → Registered`。Block 不是 "free 或 allocated"，而是有完整生命周期。
- **三池模型**：`ResetPool`(free) → `ActivePool`(in-use) → `InactivePool`(evictable)。第三个池是 prefix cache 的前置条件——block 完成后不立刻回收，而是保留内容等待复用。即使现在不做 prefix cache，把 inactive 状态设计进来意味着将来不需要重构。
- **Leader/Worker**：`InstanceLeader` 做 placement 决策，`PhysicalWorker` 执行。天然映射到 TP rank 0 = leader。
- **Attachments**：block 上的可扩展元数据（不改 struct 就能挂新信息），避免 breaking change。

**pegaflow** — 补充视角

- **内容寻址 block**：`BlockKey = namespace + content_hash`。即使不做 prefix cache，block 有稳定 identity 意味着可以 trace、log、compare，而不只是 "page index 3"。
- **Sealed vs Inflight**：block 正在写（mutable）和写完（immutable/sealed）是两种完全不同的类型。所有权在 seal 时转移。
- **Segment 化布局**：block 不绑定 "K 和 V 交替" 的布局——而是持有若干 segment，segment 的含义由 backend 定义。MLA 的 ckv/kpe 就是两个 segment。

**vLLM** — admission 参考

- **preemption-free admission**：worst-case page budget，admit 就保证跑完。简单有效。
- **Multi-group**：不同注意力类型（full / sliding / MLA）独立管理 block group。

## 设计原则

1. **Block 是管理单元**。不是 page，不是 slot。Block 持有 `block_size` 个 token 在所有层上的 KV 数据。Block 有 identity、state、owner。
2. **Logical 和 physical 分离**。BlockManager 管谁拥有哪些 block；PhysicalBackend 管 block 在 GPU 上长什么样。两层的接口是 `BlockId`。
3. **Block 有状态机**。不是 free/allocated 二态，而是有明确的生命周期转移。
4. **Scheduler 只看 logical 层**。问 "有多少 free block" 做 admission，拿到 `RequestBlocks` handle 传给 executor。不碰 GPU 指针。
5. **Physical backend 是 trait**。Full attention 和 MLA 是不同的 backend 实现。加新的注意力类型 = 加新的 backend，不改 BlockManager。

## 架构

```
Scheduler                      BlockManager (logical)           PhysicalBackend (physical)
─────────                      ──────────────────────           ─────────────────────────

"can I admit 3 requests?" ──→  free_blocks() → 450             GPU memory owner
                               block_size() → 16               ┌──────────────────────┐
"admit request R1,             ──────────────────────           │ FullAttentionBackend │
 prompt=100, max=200" ────→    allocate(R1, 19 blocks)         │ [K|V|K|V|...] / page │
                               → RequestBlocks {               └──────────────────────┘
                                   blocks: [B3,B7,B12,...],    ┌──────────────────────┐
                                   seq_len: 0,                 │ MlaBackend           │
                                   state: Reserved             │ ckv_buf + kpe_buf    │
                                 }                             │ shared page index    │
                                                               └──────────────────────┘
executor.prefill(              blocks[i].state: Writing
  &request_blocks) ──────→     → advance(100)
                               blocks[i].state: Active

executor.decode(               → grow_if_needed()
  &request_blocks) ──────→     → advance(1)

request finished ──────────→   release(R1)
                               blocks → Free pool
```

### Block 状态机

```
                 allocate()        prefill/decode writes       request done
    ┌──────┐    ──────────→    ┌──────────┐    ────────→    ┌────────┐    ──────→    ┌──────┐
    │ Free │                   │ Reserved │                 │ Active │              │ Free │
    └──────┘                   └──────────┘                 └────────┘              └──────┘
                                                                │
                                                                │ (future: prefix cache)
                                                                ▼
                                                           ┌──────────┐
                                                           │ Inactive │
                                                           └──────────┘
```

- **Free**: 无 owner，可分配。在 free pool 中。
- **Reserved**: 已分配给某个 request，但尚未写入 KV 数据（prefill 还没开始，或 decode 还没到这个 block）。
- **Active**: 包含有效 KV 数据，由 active request 持有。不可分配给其他 request。
- **Inactive** (仅将来 prefix cache 需要): 包含有效 KV 数据，但 owner request 已结束。可被新 request 复用（prefix hit）或回收（eviction）。

当前实现只需要 Free / Reserved / Active 三态。Inactive 是设计预留——block 释放时目前直接回到 Free，将来改为先进 Inactive 即可启用 prefix cache，不动 BlockManager 核心。

## 核心类型

### BlockId — block 的 identity

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BlockId(u32);
```

纯数字 index。当前够用。将来加 prefix cache 时，在 BlockManager 里给 block 关联 `SequenceHash`（Dynamo 做法）或 `BlockKey`（pegaflow 做法），BlockId 本身不变。

### BlockState — block 生命周期

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockState {
    Free,
    Reserved { owner: RequestId },
    Active { owner: RequestId },
}
```

### BlockManager — 逻辑层

管理所有 block 的 ownership 和 state。不持有 GPU 内存。

```rust
pub struct BlockManager {
    block_size: usize,                         // tokens per block
    states: Vec<BlockState>,                   // indexed by BlockId
    free_list: Vec<BlockId>,                   // LIFO for locality
    request_blocks: HashMap<RequestId, Vec<BlockId>>,
    // telemetry
    total_blocks: usize,
    stats: BlockManagerStats,
}

pub struct BlockManagerStats {
    pub free_count: usize,
    pub reserved_count: usize,
    pub active_count: usize,
    pub total_allocations: u64,
    pub total_releases: u64,
    pub total_rejections: u64,
    pub peak_active: usize,
}

impl BlockManager {
    pub fn new(total_blocks: usize, block_size: usize) -> Self;

    // ── Capacity queries (scheduler 用) ──
    pub fn free_blocks(&self) -> usize;
    pub fn block_size(&self) -> usize;
    pub fn total_blocks(&self) -> usize;

    // ── Allocation ──
    /// 为 request 预留 n 个 block。
    /// 全部能分配才分配（atomic：不会部分分配后 OOM）。
    pub fn allocate(&mut self, request_id: RequestId, n: usize) -> Result<()>;

    /// 请求跑着跑着需要更多 block（decode 跨 block 边界）。
    /// 申请 1 个追加 block。
    pub fn grow(&mut self, request_id: RequestId) -> Result<BlockId>;

    // ── Release ──
    /// 请求结束，释放所有 block。
    pub fn release(&mut self, request_id: RequestId);

    // ── Query ──
    /// 获取 request 当前持有的 block ID 列表。
    pub fn blocks_for(&self, request_id: RequestId) -> &[BlockId];

    // ── State transition ──
    /// 标记一批 block 从 Reserved → Active（prefill 完成后）。
    pub fn activate(&mut self, request_id: RequestId);

    // ── Telemetry ──
    pub fn stats(&self) -> &BlockManagerStats;
}
```

**为什么 allocate 用 RequestId 而非返回 handle？**

handle 模式（Dynamo 的 `MutableBlock`/`CompleteBlock`）适合跨进程、多 worker 场景。pegainfer 是单进程、单 scheduler 线程模型——block ownership 在 `BlockManager` 内部用 `HashMap<RequestId, Vec<BlockId>>` 跟踪即可。返回 handle 会引入 Arc/refcount 开销但没有实际收益。

### RequestBlocks — per-request 视图

```rust
/// Executor 侧看到的 per-request block 引用。
/// 不拥有 block——ownership 在 BlockManager 中。
/// 这是 executor 用来构建 kernel metadata 的数据。
pub struct RequestBlocks<'a> {
    pub request_id: RequestId,
    pub block_ids: &'a [BlockId],
    pub seq_len: usize,
    pub block_size: usize,
}

impl RequestBlocks<'_> {
    pub fn num_blocks(&self) -> usize { self.block_ids.len() }

    /// 最后一个 block 中有效的 token 数。
    pub fn last_block_len(&self) -> usize {
        if self.seq_len == 0 { return 0; }
        let rem = self.seq_len % self.block_size;
        if rem == 0 { self.block_size } else { rem }
    }

    /// 是否需要新 block（当前 block 已满）。
    pub fn needs_new_block(&self) -> bool {
        self.seq_len > 0 && self.seq_len % self.block_size == 0
    }
}
```

### PhysicalBackend trait — 物理层

```rust
/// KV cache 的物理存储后端。
/// 不同注意力类型实现不同的 backend。
/// BlockManager 不知道这个 trait 的存在。
pub trait PhysicalBackend {
    /// 每个 block 占多少 GPU 内存 (bytes)。
    /// BlockManager 用 total_gpu_memory / block_bytes() 算 total_blocks。
    fn block_bytes(&self) -> usize;

    /// 返回 kernel 需要的物理指针和 stride 信息。
    /// block_ids 来自 BlockManager 的 blocks_for() 结果。
    fn kernel_metadata(&self, block_ids: &[BlockId]) -> KernelKvMetadata;
}
```

具体实现：

#### FullAttentionBackend

```rust
/// Full attention (Qwen3, Qwen3.5 的 8 个 attn 层)。
/// 单 buffer，page-first layout：
///   block_i: [L0_K | L0_V | L1_K | L1_V | ...]
pub struct FullAttentionBackend {
    num_layers: usize,
    num_kv_heads: usize,   // per TP rank
    head_dim: usize,
    block_size: usize,
    buffer: CudaSlice<bf16>,
}
```

#### MlaBackend

```rust
/// MLA (K2 attention)。
/// 双 buffer + 共享 block index。
///   ckv_buffer: block_i → [L0(block_size × compressed_dim) | L1 | ...]
///   kpe_buffer: block_i → [L0(block_size × rope_dim) | L1 | ...]
///
/// pegaflow 启发：K/V 分离 segment 优化拷贝带宽。
/// MLA 的 ckv 和 kpe 就是天然的两个 segment。
pub struct MlaBackend {
    num_layers: usize,
    compressed_dim: usize,  // K2: 512
    rope_dim: usize,        // K2: 64
    block_size: usize,
    ckv_buffer: CudaSlice<bf16>,
    kpe_buffer: CudaSlice<bf16>,
}
```

**为什么双 buffer 而非交错？** FlashInfer MLA decode kernel 需要 ckv 和 kpe 的独立 base pointer。交错布局需要每次算 offset，而且 ckv/kpe 维度不同导致 stride 不规则。双 buffer 各自 stride 一致，kernel 简单直接。

#### KernelKvMetadata

```rust
/// PhysicalBackend 产出的 kernel 侧元数据。
/// 不同 backend 填充不同的 variant。
pub enum KernelKvMetadata {
    FullAttention {
        buffer_ptr: u64,     // CudaSlice device pointer
        page_indices: Vec<i32>,
        // strides: per-layer K/V block len, layer stride, page stride
        kv_block_len: usize,
        layer_stride: usize,
        page_stride: usize,
    },
    Mla {
        ckv_ptr: u64,
        kpe_ptr: u64,
        page_indices: Vec<i32>,
        ckv_page_stride: usize,
        kpe_page_stride: usize,
    },
}
```

### 组装：谁持有什么

```
pegainfer-core:
  BlockId, BlockState, BlockManager, BlockManagerStats
  RequestBlocks
  PhysicalBackend trait, KernelKvMetadata

pegainfer-core (或 pegainfer-kernels):
  FullAttentionBackend
  MlaBackend

per-model crate (scheduler):
  持有 BlockManager
  admission 时查 block_manager.free_blocks()
  forward 时构建 RequestBlocks 传给 executor

per-model crate (executor):
  持有 PhysicalBackend
  收到 RequestBlocks + token data → 调 backend.kernel_metadata() → 调 kernel
```

## Scheduler Admission

Scheduler 只看 `BlockManager`，不碰 `PhysicalBackend`。

```rust
fn admit_requests(
    candidates: &mut VecDeque<PendingRequest>,
    active: &[ActiveRequest],
    bm: &BlockManager,
) -> AdmissionOutcome {
    let block_size = bm.block_size();

    // active 请求未来还需要的 block 数（worst case）
    let future_active: usize = active.iter().map(|r| {
        let max_tokens = r.prompt_len + r.max_tokens.saturating_sub(1);
        let max_blocks = max_tokens.div_ceil(block_size);
        max_blocks.saturating_sub(r.current_tokens().div_ceil(block_size))
    }).sum();

    let mut budget = bm.free_blocks().saturating_sub(future_active);
    let mut admitted = Vec::new();
    let mut deferred = Vec::new();
    let mut rejected = Vec::new();

    while let Some(req) = candidates.pop_front() {
        let max_tokens = req.prompt_len() + req.max_tokens().saturating_sub(1);
        let needed = max_tokens.div_ceil(block_size);

        if needed > bm.total_blocks() {
            // 即使整个 pool 都给这个 request 也不够
            rejected.push(req);
        } else if needed <= budget {
            budget -= needed;
            admitted.push(req);
        } else {
            deferred.push(req);
        }
    }

    AdmissionOutcome { admitted, deferred, rejected }
}
```

这是 preemption-free 模型：admit 就保证跑完。将来如果要支持 preemption（vLLM V1 的做法），加一个 `preempt()` 方法把最低优先级 request 的 block 回收即可——BlockManager 已经追踪了谁持有什么。

## TP 支持

TP 下每个 rank 有自己的 `PhysicalBackend`（因为 GPU buffer 是 per-GPU 的），但共享一个 `BlockManager`。

```
                    ┌──────────────────┐
                    │   BlockManager   │  ← scheduler (rank 0) 操作
                    │   (逻辑，一份)    │
                    └────────┬─────────┘
                             │ BlockId
            ┌────────────────┼────────────────┐
            │                │                │
   ┌────────▼──────┐ ┌──────▼────────┐ ┌─────▼───────┐
   │ MlaBackend    │ │ MlaBackend    │ │ MlaBackend  │
   │ rank=0, GPU=0 │ │ rank=1, GPU=1 │ │ rank=7 GPU7 │
   └───────────────┘ └───────────────┘ └─────────────┘
```

- 所有 rank 的 `PhysicalBackend` 有相同数量的 block，相同的 block_size
- `BlockManager` 只跑在 scheduler 线程（rank 0），block 分配决策通过现有 NCCL 广播或简单的 metadata 传播同步到其他 rank
- 各 rank 的 `kernel_metadata()` 返回各自 GPU buffer 上的指针，但 page_indices 相同

这和 Dynamo 的 Leader/Worker 模型对齐：BlockManager = Leader，各 rank 的 PhysicalBackend = Worker。

## DP 支持

每个 DP rank 是完全独立的 engine instance：

```
DP=0: [BlockManager₀] + [PhysicalBackend₀] + [Scheduler₀]
DP=1: [BlockManager₁] + [PhysicalBackend₁] + [Scheduler₁]
...
DP=7: [BlockManager₇] + [PhysicalBackend₇] + [Scheduler₇]
```

Load balancer 查询各 `BlockManager.free_blocks()` 做路由。EP all-to-all 天然在 DP rank 间同步。

## CUDA Graph 兼容

1. `PhysicalBackend` 的 GPU buffer 在创建时一次性分配，生命周期和 engine 相同。指针不变。
2. Decode 路径用 pre-allocated `CudaSlice<i32>` 做 FlashInfer metadata（page_indices, page_indptr, last_page_len, etc.）。每步只 H2D memcpy 更新内容，不分配新内存。
3. Bucket CUDA Graph：不同 batch size 不同 graph。Padding slot 的 page_indices 指向一个固定的 padding block（`BlockManager` 初始化时保留 1 个 block 永不分配）。

## 容量计算示例

K2 on H20 × 8 (TP=8, MLA):

```
H20 HBM per rank: 96 GB
Model weights (INT4 + scales): ~3.2 GB per rank
Runtime buffers: ~2 GB
Available for KV per rank: ~90 GB

MLA: compressed_dim=512, rope_dim=64, 61 layers, block_size=16
Per block per rank:
  ckv: 16 × 512 × 61 × 2 bytes = 999,424 bytes
  kpe: 16 × 64 × 61 × 2 bytes  = 124,928 bytes
  total: 1,124,352 bytes ≈ 1.07 MB

90 GB / 1.07 MB ≈ 85,000 blocks
85,000 × 16 tokens = 1,360,000 tokens capacity

bs=64, avg 4K tokens/req → 256 blocks/req → 16,384 blocks → 19% utilization
bs=64, avg 16K tokens/req → 1,000 blocks/req → 64,000 blocks → 75% utilization
```

## 已落地代码

### `pegainfer-block-manager` crate

独立 crate，零 GPU 依赖，纯逻辑。从 Dynamo `kvbm-logical`（Apache-2.0）的 `BlockStore` 精简而来。

**文件**: `pegainfer-block-manager/src/lib.rs`

**核心类型**:
- `BlockId(u32)` — block identity
- `RequestId(u64)` — request identity
- `SlotState { Free, Reserved(RequestId), Active(RequestId) }` — Dynamo 的 slot 状态机精简版
- `BlockManager` — unified slot tracking + free list + per-request block 列表
- `BlockManagerStats` — telemetry snapshot
- `admit_by_block_budget()` — scheduler admission helper

**API**:
```rust
impl BlockManager {
    fn new(total_blocks, block_size) -> Self;
    fn free_blocks(&self) -> usize;
    fn allocate(&mut self, request_id, count) -> Option<Vec<BlockId>>;  // all-or-nothing
    fn grow(&mut self, request_id) -> Option<BlockId>;
    fn activate(&mut self, request_id);         // Reserved → Active
    fn release(&mut self, request_id);          // → Free
    fn blocks_for(&self, request_id) -> Option<&[BlockId]>;
    fn padding_block(&self) -> BlockId;         // CUDA Graph padding
}
```

**从 Dynamo 搬运并精简的**:
- slot vector + free list (VecDeque FIFO) 统一管理，对应 Dynamo 的 `BlockStore::inner`
- all-or-nothing `allocate()`，对应 Dynamo 的 `allocate_atomic()`
- Block 0 永久 reserved 做 padding，对应 Dynamo 的 null block

**没搬的**（当前不需要）:
- `T: BlockMetadata` 泛型（我们不按存储层参数化）
- `SequenceHash` / `BlockRegistry` / `InactiveIndex`（没有 prefix cache）
- `ImmutableBlock` / `WeakBlock` / `Primary` / `Duplicate`（没有共享引用）
- `Mutex`（scheduler 单线程，`&mut self`）
- metrics pipeline

**测试**: 16 个 unit tests 全部通过，覆盖 allocate/grow/activate/release 生命周期、all-or-nothing 原子性、admission 三分法。

## 实现路径

### Step 1: ✓ BlockManager logical layer

已完成。`pegainfer-block-manager` crate 独立，零 GPU 依赖。

### Step 2: MlaBackend (PhysicalBackend)

在 `pegainfer-core` 中实现 `MlaBackend`：双 buffer (ckv + kpe)，block_size=16，共享 BlockId 映射。对齐 K2 现有 FlashInfer MLA kernel 的 metadata 接口。

### Step 3: K2 迁移

K2 的 `KimiWorkerDecodeArena` 迁移到 `BlockManager` + `MlaBackend`。固定 arena 变成动态分配。K2 scheduler 加 admission。

验证：K2 E2E 正确性 + TPOT 不退化。

### Step 4: DP 路由

K2 DP mode 每个 rank 独立 `BlockManager`。Load balancer 基于 `free_blocks()` 做 least-loaded routing。

## 待讨论

1. **block_size**: 16（FlashInfer 推荐，K2 和 Qwen3 已用）。
2. **Inactive 池的 eviction policy**: 将来 prefix cache 需要时选择。候选：LRU（vLLM）、TinyLFU（pegaflow）、LFU（Dynamo）。不在本轮 scope。
