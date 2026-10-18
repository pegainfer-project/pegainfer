# pegaflow KV 卸载接入 Spec

> **TL;DR**: 把 `pegaflow-core` 当**进程内 Rust 库**做 KV 卸载的物理后端（HBM→DRAM/SSD/RDMA），补上 kvbm 留着没写的卸载层。connector 大脑（决定 load/save 哪些 block）用 kvbm logical/physical 分层思想自建，pegaflow 退为语义无关的 raw block transfer 后端。**路线已调整为 Qwen3-4B full-attn 首发**（原计划 Kimi 首发）：page-first 单 buffer 经 pegaflow `block_stride_bytes`（PR #331）适配。**端到端已在真实 GPU 上跑通并验证**：async SAVE + async LOAD 接进 `Qwen3Executor` + scheduler，`tests/kv_offload_cpu_hit.rs` 覆盖纯 CPU-hit 与 GPU+CPU 组合 hit，恢复后 logits 与冷算一致；连接层 `OffloadEngine` + `tests/cpu_roundtrip.rs` 字节级一致。默认关（builder flag opt-in）；**server CLI 已接**（#316：`--kv-offload` / `--kv-offload-host-gib` / `--no-prefix-cache`，plain 与 `--enable-lora` 两条启动路径都透传）。纯-L2 基准实测 Qwen3-4B mean TTFT 195→40ms（−79%，evict-before-probe → `gpu_hit=0`，全前缀从 host tier 恢复）。**Qwen3.5 linear/SSM state 明确排除**；**DeepSeek sparse 暂缓**。
>
> Last touched: 2026-06

## 0. 实现状态（2026-06）

已落地并验证：

- **pegaflow `block_stride_bytes`**（PR #331 → novitalabs/pegaflow，`feat/inproc-load` 基于其上）：解耦"块间步长"与"每块拷贝大小"，让 page-first fused buffer 能注册。**已合入 master**。
- **pegaflow 进程内 load API**（PR #333，**已合入**，squash 进 #331 的 `07cac7e`）：`LoadCompletion::{Shm,Channel}` + `batch_load_kv_blocks_multi_layer_inproc` → `oneshot::Receiver`，去掉 in-process 调用方对 shm `LoadState` 的依赖（Rust 进程内不需要），非阻塞 poll。
- **`pegainfer-kv-offload::OffloadEngine`**：拥有 `PegaEngine` + 内嵌 tokio runtime；`Registration::from_buffer` 把 fused page-first buffer 映射成 per-layer 注册（**单段 `[K|V]`**：fused layout 里 K/V 本就连续 = `layer_stride` 一段，`block_stride = page_stride`，`segments=1`——不是 K/V split，那条路需要 `kv_stride > bytes_per_block`，此处不成立）。crate 已按 `config`/`host`/`layout`/`handle`/`engine` 分模块（#799/#802 重构）；所有操作统一为 `OffloadHandle<T>` 可轮询句柄：`submit_save`/`submit_query`/`load` 提交到 pegaflow runtime 后立即返回 handle，`save`（fire-and-forget）/`save_blocking`/`query` 是其薄包装（qwen3 调用方尚未迁移）。`query` 透传 pegaflow 0.23.5 的 `wait_for_full_prefix`（glm52 native handoff 传 `true`：D 无法重算 miss，partial 命中无用）。
- **`KvBuffer::device_ptr`**（kv-cache）：注册用的稳定基址。
- **kvbm↔bytes 桥**（kv-cache `RequestKv`）：`prompt_block_hashes` / `assigned_block_hashes` / `prefix_matched_blocks`，`SequenceHash::as_u128()` → 16B content key。
- **`tests/cpu_roundtrip.rs`**：真实 `KvBuffer` 上写已知 pattern → save → query → load 到**另一组** block → 字节级比对 + 零块负向控制。**通过**。
- **live 接线（§9，已落地）**：`Qwen3Executor` 持 `Option<OffloadEngine>`（`Qwen3OffloadOptions` opt-in，默认关）；SAVE hook（`save_sealed_blocks`，async fire-and-forget）+ 非阻塞 prefetch admission（`begin_kv_prefetch`/`drain_ready_prefetch`/`wait_ready_prefetch`，scheduler `loading` 态）。`tests/kv_offload_cpu_hit.rs` 单测序跑两幕——纯 CPU restore（`gpu_hit==0`）与 GPU+CPU 组合 hit（G=3+C=3 拼成一段连续前缀）——恢复后 first-token logits 与冷算一致（mean Δ≈0.03 nat，bf16 floor）。
- **三处正确性加固**（toxic-review 后）：① query lease 在 `reserve_loaded_blocks` 失败 / `load` 提交失败时显式 `release_query_lease`，不再泄漏到 600s TTL；② admission 拒绝（context/KV budget/未知 LoRA）时 `drop_request` 释放已 settle 的 prefetch 状态，不再泄漏已 commit 的 block；③ async SAVE 把被保存 block 的 `ImmutableBlock` 强引用（`KvBlockGuard`）随 spawn 持到 D2H 落地才 drop——封死"请求结束→slot 重分配→D2H 抓到错 KV 写进旧 hash"的静默腐蚀窗口。

**server CLI 已接（#316）**：`--kv-offload`（bool）/ `--kv-offload-host-gib`（f64，默认 8.0，pegaflow 启动即整块 `cudaHostAlloc`，RSS 立即反映）/ `--no-prefix-cache`（vLLM 风格；不带 offload = 关前缀匹配，带 offload = 纯-L2 模式，evict-before-probe 使每个前缀从 host tier 恢复）。plain 与 `--enable-lora` 两条路径都透传 `offload_options` + `no_prefix_cache`；LoRA 下安全，因前缀 block hash 以 adapter 名加 salt（`compute_salt_hash`），恢复的 KV（HBM 或 host tier）永不跨 adapter。三处 #316 review 加固：echo 请求不 offer prefetch（其 prefill 跳过 `match_and_add_prefix`，prefetch 块用不上）、admission 按 `prefetched_blocks` 抵扣已 settle 前缀块、`drop_request` 等在途 H2D 落地再放 reservation。**依赖已从 fork 摘除**：PR #331+#333 均合入上游 master（squash 进 `07cac7e`），`third_party/pegaflow` 已删，`pegaflow-core` 改为 pin 到该 rev 的 **git 依赖**（见 §5.2），GPU 测试在 git-dep 下行为不变（delta 一致）。

相关：[kv-cache-design.md](kv-cache-design.md)（logical/physical 分层，已把 pegaflow 列为设计调研）· [qwen3-kvbm-integration-spec.md](qwen3-kvbm-integration-spec.md)（kvbm-logical 已接入）· `models/kimi-k2/kv-cache-design.md`（Kimi 已用 `BlockPool`）· `models/qwen3/prefix-cache.md`（HBM 内前缀复用已落地）。

---

## 1. 定位：pegaflow 是 raw 后端，connector 大脑要自建

pegaflow（`third_party/pegaflow`，novita，Apache-2.0）原本是 **vLLM 的 KV connector 服务端**：KV 的编排逻辑（何时 save、query 几个 block、prefix 匹配、与 scheduler 的 admission/preemption 交互）全在 vLLM 的 Python connector 那一侧，`pegaflow-core` 只是底下干 D2H/H2D + 分层存储的**肌肉**。

pegainfer 不是 vLLM，那套 Python connector 一行用不上。接入要做的是**用 Rust 自建那颗 connector 大脑**——而 kvbm 的 logical/physical 分层正是它的骨架：

```
per-model scheduler   ← 策略：哪些 block 该 resident（full 前缀 / MLA 全前缀 / 未来稀疏选择）
  ↓ 产出 load/save 意图（一组 block）
connector（kvbm logical/physical 思想）← 机制：block identity、状态机、GPU slot 编排、transfer 调度
  ↓ 语义无关的 raw transfer
pegaflow-core         ← 机制底座：D2H/H2D、DRAM/SSD/RDMA 分层
```

## 2. 战略决策：pegaflow 取代 kvbm 死代码做物理 tier

pegainfer 仓里 vendored 的 `kvbm-physical` / `kvbm-engine` 设计目标就是分层卸载，但**至今零接线、是死代码**（无任何非 kvbm crate 依赖）。同时养两套分层卸载违反项目复杂度红线。本 spec 采纳：**`kvbm-logical`（逻辑层 + 前缀匹配）保留，pegaflow-core 顶替它下面缺失的物理卸载层，砍掉 `kvbm-physical`/`kvbm-engine`**。理由：pegaflow 同组维护、已上 PyPI、有 H800 benchmark、库化干净；kvbm 那两层是纯负债。已执行（2026-06）：vendored `kvbm/` 目录只留 `kvbm-logical` fork，`dynamo-tokens`/`dynamo-kv-hashing` 改为 ai-dynamo/dynamo 上游 git 依赖（pin rev），其余 8 个 vendored crate 删除。

## 3. 三模型三 KV 形态 → connector 边界（实据）

| 模型 | KV 形态 | active set | 跨请求复用 | 卸载结论 |
| --- | --- | --- | --- | --- |
| **Qwen3 / Qwen3.5 full-attn** | paged，page-first 单 buffer，`PagePool` | 无（dense 全前缀） | 有（前缀缓存已落地） | **已首发（#316）**：page-first 与 pegaflow `stride==copy-size` ABI 冲突已由 `block_stride`（§5.R1）解掉，端到端跑通 |
| **Kimi-K2 MLA** | paged，per-layer ckv/kpe arena，后端是 `BlockPool`；latent 68.6 KiB/token，无 per-head | 无（dense 全前缀） | 有（HBM 内 prefix cache 已落地） | **下一候选**：layout 直接适配 pegaflow registration（接入面最干净），复用 Qwen3-4B 这套 connector 模式即可 |
| **Qwen3.5 linear（24 层）** | per-request `RecurrentState` [32,128,128] f32 2 MiB/层，非 paged、独立分配 | 无（每步读写整个 matrix） | **零**（this-request 有损摘要，非 content-addressable） | **排除**：offload 无 prefix/dedup 收益；省显存是 per-request swap-out，另一套机制 |

**边界结论**：connector 只收 **block-structured、content-addressable** 的 KV（MLA latent / full-attn paged）。recurrent/SSM state 不进 connector。稀疏的 active-set gather 是独立的、未来的课题。

证据：Kimi `pegainfer-kimi-k2/src/runner/{worker.rs:612-619, cache.rs:63-80, mla.rs:38-48}`、`scheduler.rs:16,27,146,180`、`pool.rs:123`；Qwen3.5 linear `pegainfer-qwen35/src/...recurrent.rs`、`batch_decode_graph.rs:82-86`。

## 4. 路线

1. **Qwen full-attn 已首发（#316）** —— 给 pegaflow 加了 `block_stride_bytes`（R1）解掉 page-first ABI 冲突，async SAVE + 非阻塞 prefetch admission 接进 `Qwen3Executor` + scheduler，server CLI 已接。
2. **Kimi MLA 下一候选** —— pegaflow 做 `BlockPool` 下的 host/SSD tier；block evict 时 demote 到 host，前缀 query 命中时从 host restore。带宽便宜（latent 小），layout 零阻抗，直接复用 Qwen3-4B 的 connector 模式。
3. **linear 排除、sparse 暂缓**。

## 5. 可行性（对抗验证结论，附证据）

四条承重假设由 10-agent workflow 对抗验证：

1. **✅ 进程内注册裸指针，无 IPC、无第二进程**：`register_context_layer_batch(data_ptrs: &[u64])`（`pegaflow-core/src/lib.rs:242-259`）收裸设备地址，拷贝路径直接喂给 driver API `cuMemcpyDtoHAsync_v2`（`transfer/memcpy.rs:82-89`）；IPC 只在 server/Python 层，core 零 IPC 调用点。cudarc 附设备 **primary context**（与 pegainfer 同一），自建 worker stream。
2. **✅ 依赖无致命冲突**：cudarc 单 major（0.19.3↔0.19.7 统一），cuda-12080/12090 共存（build.rs 取高版本），tokio/tonic/prost 兼容。**依赖行**（git rev pin 到上游 master `07cac7e`，含 #331+#333；`default-features=false` 砍掉 pegaflow 自带的 `cuda-12`/`rdma`，靠 workspace cudarc 提供的 `cuda-12090`+`nvrtc` 满足——pegaflow-core 无 `cfg(cuda-12)` gate）：
   ```toml
   pegaflow-core = { git = "https://github.com/novitalabs/pegaflow.git", rev = "07cac7e50e8ae7be15ad1b9311401039c9ee439b", default-features = false }
   ```
   下次再改 pegaflow：临时换回 path dep 共同开发 → 提 PR → 合入后 re-pin rev。
   **为何 `cuda-12` 而非 `cuda-13`**（本机明明是 CUDA 13.3 toolkit / 13.0 driver）：pegainfer 有意锁 `cudarc/cuda-12090`（`Cargo.toml:92-93`，issue #263——配 cudarc 0.19.5+ 的 per-symbol lazy loading，压低 binding level 以**不抬高 runtime driver floor**、保宽部署兼容；故意不用 `cuda-version-from-build-system` 自动，否则 driver floor 会跟着构建机 toolkit 走）。cudarc 在 workspace 是**单实例、feature 取并集后选最高版本**：pegaflow 用 `cuda-12` 并集后仍是 12090、不抬 floor；用 `cuda-13`（→ `cudarc/cuda-13000`）会把**整个 workspace 含 pegainfer 自己**顶到 13000、driver floor 抬到 CUDA 13，撞翻 #263。整体迁 cu13 是独立决策（须同时改 pegainfer 的 cudarc + revisit #263），本期不做。
3. **⚠️ Layout**：block-hash 键直接适配（`u64→Vec<u8>`）；page-first layout **不适配**（见 §5.R1）；Kimi per-layer 布局**天然适配**。
4. **✅ 流同步**：host-side 粗同步可解——save 前 pegainfer 必须 `synchronize()` compute stream（pegaflow 私有 stream 只自同步，`gpu_worker.rs:520-528`），restore 前自旋 poll `LoadState`。代价：损 compute/offload 重叠（见 §6.R3）。

## 6. connector 接口（dense-first，稀疏留门不展开）

两层分离，稀疏复杂性全关在 policy 侧：

```rust
// mechanism —— pegaflow backend，永不懂稀疏/前缀
trait KvOffloadBackend {
    fn load(&self, items: &[(BlockHash, GpuSlot)]) -> LoadHandle; // 任意集合，phase 无关
    fn offload(&self, items: &[(GpuSlot, BlockHash, OffloadHint)]);
    fn poll(&self, h: LoadHandle) -> LoadState;
}
enum OffloadHint { ReusableAcrossRequests, TransientDiscard }

// policy —— per-model scheduler，懂自己的拓扑
trait KvResidencyPolicy {
    fn required_blocks(&self, req: &RequestCtx, phase: Phase) -> SmallVec<BlockId>;
    fn save_hint(&self, block: BlockId) -> OffloadHint;
}
```

**现在做对、未来免费受益的三个决策**（即便 dense-first 也按这个写，成本为零）：
- 接口说 **block 集合**不说 prefix-count（full attention 产出的集合恰好连续 = 退化特例）；
- admission 按 **active working set ≤ HBM** 写（dense 下 active=total，退化）；
- `load` **phase-agnostic**（不绑 prefill，未来 decode gather 是"启用"不是"重设计"）。

第一版：`required_blocks` 对 Kimi/Qwen 就是"全前缀"，`OffloadHint` 全 `ReusableAcrossRequests`，只走 prefill-前 + evict 路径。

## 7. 风险

| # | 风险 | 等级 | 处置 |
| --- | --- | --- | --- |
| R1 | Qwen page-first vs pegaflow `stride==copy-size` ABI 不兼容 | major | 给 `KVCacheRegistration` 加 `block_stride_bytes`（改 pegaflow ~几十行，`instance.rs` + `transfer/mod.rs`）；**Kimi 首发绕开此风险** |
| R2 | save 前漏 `synchronize()` → 静默 D2H 半写 KV，pegaflow 不校验 | major | bridge 层把 synchronize 设成不可绕过 + debug 断言 |
| R3 | host-side 粗同步损 compute/offload 重叠 | minor | 第一版接受；后续给 pegaflow 加 device-side event-injection |
| R4 | 依赖误配（裸 default-features=false / 漏 cuda-12） | minor | §5.2 依赖行已定，CI 编译验证 |
| R5 | 稀疏 active-set offload 的 token-vs-block 粒度落差 | 已知开放 | 见下，不在本期 |

**稀疏（已知开放问题，不在本期）**：连 dynamo KVBM 都没解 sparse attention offloading——它的复用是 radix 前缀、offload 是 frequency/LRU、tier 是整请求异步流动，对 SWA 只在 router 透传 `kv_cache_spec_sliding_window` 做 window-aware 前缀，对 topk 零处理。没有现成抽象可继承。pegainfer 侧 DeepSeek 的 indexer 已产出显式可拦截的 active-set 信号，但 token/row 粒度 ≠ block 粒度，且 compressor 已控 footprint 当前不需 offload。机制层（内容寻址 + 可插拔 policy + 语义无关 transfer）本就不堵稀疏，真正缺的 decode-loop gather 大脑到时候结合具体模型新写更准。

## 8. 下一步：Kimi MLA 最小 spike

**目标**：进程内跑通一个 page 的 register→save→evict→load，证伪"无先例"风险 + 量带宽。

1. 新 bridge crate，path-dep pegaflow-core（§5.2 依赖行），`cargo build` 验依赖。
2. Kimi：`new_with_config` → `register_context_layer_batch`（per-layer ckv/kpe，segments=2，per-layer 布局天然适配）。
3. 一个 page：`synchronize` → `save` →（手动 evict）→ `query` 命中 → `load` 回 GPU → 比对 bytes 一致。
4. 量 host↔HBM 带宽 + save 前 synchronize 的 host stall（确认 R3 可接受）。
5. 通过后再决定给 pegaflow 加 `block_stride` 上 Qwen page-first（R1）。

**阻塞**：等 §2 战略决策最终拍板（pegaflow 取代 kvbm 卸载层 = 是）。

## 9. live 接线设计（Qwen3-4B，**已落地**）

> 状态：已实现并在真实 GPU 上验证（§0）。下文是设计与实现一致的记录；落地时相对原设计的偏差与加固见末尾「实现注记」。

连接层已就绪（§0），把它接进 `Qwen3Executor` + `scheduler.rs` 的真实推理路径。`Qwen3Executor` 持 `kv_mgr`（`BlockPool`+`KvBuffer`）与 `request_kvs`；在构造（`from_runtime`/`single`，model 移入 RankWorker 之前，此时 `KvBuffer` + `device_ctx().stream` 都在手）建一个 `Option<OffloadEngine>`，opt-in（builder flag，**不加 env**），默认关，保现有路径不动。

**SAVE（async，best-effort）**：`apply_prefill`/`apply_decode` 封块后（此刻 compute stream 已随 `run_step` 同步 → 满足 §0 的跨 stream ordering 约束），取 `rkv.assigned_block_hashes()`，按 per-request `saved_cursor`（初值 = `prefix_matched_blocks()`，GPU-hit 前缀已 resident，跳过）保存新封的 `(page_id, hash)`，`offload.save(...)` fire-and-forget，推进 cursor。

**LOAD（async，GPU+CPU hit，非阻塞 admission）**：admission 把 `match_and_add_prefix` 拆成"建 RequestKv → 算 GPU hit G → query CPU [G..F] → 异步 load → LoadingKv 轮询"：

1. `rkv = pool.new_request(...)`；`hashes = rkv.prompt_block_hashes()`（F 块）。
2. `manager.match_blocks(&seq_hashes)` 数出 GPU 命中前缀 G（**持其 `ImmutableBlock` 不 drop**，防 load 期间被 evict）。
3. `offload.query(req_id, &hashes[G..F])` → CPU 命中 C（连续）+ lease。
4. `manager.allocate_blocks(C)` 拿 C 个 `MutableBlock`（DMA 落点），取 `block_id()` 列表；`offload.load(lease, page_ids)` → `LoadHandle`。请求进 `LoadingKv{rkv, handle, muts, hashes[G..G+C], gpu_imms}` holding 态，**不 prefill**。
5. 每 tick `handle.poll()`：`Ready` → 对每个 `mut` `.stage(hash, bs)` + `manager.register_block(..)` 注册进 registry（用的就是 `BlockPool::new` 给 padding 块用的同一套公开 API，**无需改 kvbm**）；随后 `rkv.match_and_add_prefix()` 自然命中 G+C 连续前缀，`kv_position=(G+C)*bs`；drop holding 的 imms（sequence 自持）。请求转入正常 prefill（suffix = 剩余 token）。
6. `C==0` → 直接 prefill（纯 GPU hit，与今日行为一致）。

**为何 register→rematch 而非直接注入 sequence**：复用现成的 `match_and_add_prefix`（GPU+CPU 在它眼里就是一段连续前缀），零 kvbm 改动；register 与 rematch 同 tick、且 holding 了 G 的 imms，eviction 窗口为零。最坏（真被 evict）只是少命中、退化为多 prefill，不损正确性。

**scheduler 状态机**：`scheduler_loop` 新增 `loading: Vec<PendingRequest>`，每 tick `reclaim_ready_prefetch`（settle 完的回 `deferred` 队首）+ `offer_prefetch`（未 offer 的 deferred 试 prefetch，起 load 的移入 `loading`）；空闲且有 `loading` 时 `block_on_loading` 阻塞等一个 DMA。`OffloadEngine` 的 `block_on`（query/flush）只在 scheduler 这个**纯 OS 线程**调用，`debug_assert` 护住误用。

风险：preemption/release 时须 drop holding 的 mutable/immutable（RAII 已覆盖）；admission KV 预算要把 loading 占用的 C 块计入 in-flight。

**实现注记（相对原设计的偏差 + toxic-review 加固）**：

- **prefetch 状态落在 executor 而非 scheduler**：`PrefixProbe`（持 G 的 imms + commit 后的 C 块）、`LoadReservation`（C 个 MutableBlock DMA 落点）、`LoadHandle` 都封进 `Qwen3Executor.prefetch: HashMap<RequestId, PrefetchState>`，scheduler 只跟 `RequestId`。commit 在 `seq_hashes[gpu_hit + i]`（GPU+CPU 偏移对齐，组合 hit 测试守这条）。
- **lease 泄漏修复**：`query` 创建的 pegaflow lease 在 `reserve_loaded_blocks` 失败 / `load` 提交失败时 `OffloadEngine::release_query_lease` 显式释放（`QueryLeaseId` 是 `Copy` 裸 token、无 Drop，丢掉只会挂到 600s TTL）。
- **拒绝清理**：admission（context/KV budget）与未知 LoRA 拒绝路径补 `drop_request`——否则一个已 settle prefetch 的请求被拒后，commit 的 block + map entry 永久泄漏。
- **SAVE 防 slot 复用腐蚀**：async `save()` 把被存 block 的 `ImmutableBlock` 强引用（`KvBlockGuard`，与 `block_ids` 1:1）随 spawn 持到 pegaflow D2H 落地才 drop。否则短请求结束 → slot 回收重分配 → 新请求覆写 → 在途 D2H 抓到新 KV 写进旧 hash = 静默腐蚀。guard 在 offload 线程并发 drop 是安全的（kvbm `BlockStore` 单 Mutex、有并发 drop race 处理）；`flush_saves` await 各 save 任务后 guard 才落，故 evict 前先 flush 仍能把 block 排空。
- **测试**：`tests/kv_offload_cpu_hit.rs` 合一个顺序 `#[test]`（避免两 executor 撞同一 device + pegaflow instance_id），先纯 CPU 后组合 hit。
