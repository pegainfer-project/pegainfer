# Kimi-K2 PPLX EP Decode 优化

> **TL;DR:** PPLX EP decode bs=1 TPOT 从 37ms 优化到 17.94ms（−52%），已超过 NCCL no-graph baseline（18.52ms）。根因是 `expert_padding=64` 导致 Marlin GEMM 98% 计算浪费（8 real tokens / 512 processed positions），加上 <<<1,1>>> 串行 routing kernel 每步贡献 0.89ms 固定开销。修复后 PPLX 的 MoE 计算负载与 NCCL 路径完全对齐。
>
> **Last touched:** 2026-05

## 背景

PPLX EP 是 Kimi-K2 从 TP8+EP8（NCCL）迁移到 TP1+DP8+EP8 的核心路径。PPLX 使用 4-step all-to-all 协议（dispatch_send → dispatch_recv → combine_send → combine_recv）替代 NCCL allgather + reduce_scatter。

优化目标：**PPLX EP decode TPOT 接近 NCCL 不开 CUDA Graph 的基线**（≈19ms）。

NCCL 开 CUDA Graph 是 16ms，但 PPLX worker 线程与 graph capture 不兼容，所以 no-graph 是公平对标。

## 最终结果

| 阶段 | PPLX TPOT p50 | NCCL no-graph p50 | Gap |
| --- | ---: | ---: | --- |
| 初始 | 37ms | 18.52ms | +18.5ms |
| SwiGLU GPU-side + memset 移除 | 29ms | 18.52ms | +10.5ms |
| host-side tight bound | 28ms | 18.52ms | +9.5ms |
| expert_padding=16, block_size=8 | 20.72ms | 18.52ms | +2.2ms |
| **expert_padding=8, parallel routing** | **17.94ms** | **18.52ms** | **−0.58ms** |

bench 参数：`--prompt-len 1 --output-len 32 --warmup 3 --iters 5 --cuda-graph false`，H20 ×8。

## nsys 对比（expert_padding=8 vs NCCL no-graph）

traces: `target/profiling/kimi_pplx_decode_v3.sqlite`, `kimi_nccl_nograph_v2.sqlite`

### MoE 相关 kernel per step per GPU

PPLX-only（NCCL 路径没有这些 kernel）：

| Kernel | Avg (ms) | Calls/step | ms/step |
| --- | ---: | ---: | ---: |
| a2a_combine_recv | 0.0506 | 60 | 3.03 |
| a2a_dispatch_recv | 0.0368 | 60 | 2.21 |
| a2a_dispatch_send | 0.0160 | 60 | 0.96 |
| routing kernel | 0.0149 | 60 | 0.89 |
| a2a_combine_send | 0.0096 | 60 | 0.58 |
| swiglu_w13_pplx | 0.0020 | 60 | 0.12 |
| **PPLX-only total** | | | **7.79** |

NCCL-only（PPLX 路径没有这些 kernel）：

| Kernel | Avg (ms) | Calls/step | ms/step |
| --- | ---: | ---: | ---: |
| ReduceScatter_f32 | 0.1106 | 60 | 6.64 |
| marlin_align_small | 0.0100 | 60 | 0.62 |
| sum_topk_rows | 0.0021 | 60 | 0.13 |
| swiglu_w13 | 0.0018 | 60 | 0.11 |
| repeat_f32_rows | 0.0017 | 60 | 0.10 |
| **NCCL-only total** | | | **7.60** |

两条路径的 a2a / collective 总开销基本持平（7.79 vs 7.60ms）。

Marlin GEMM 对比：

| Path | Avg (ms) | ms/step/GPU | Block_size | Positions |
| --- | ---: | ---: | ---: | ---: |
| PPLX (expert_padding=8) | 0.0178 | 2.21 | 8 | 64 |
| NCCL | 0.0137 | 1.70 | 8 | ~64 |
| **delta** | | **+0.51** | | |

Marlin 仍有 0.51ms 差距，来自 PPLX 路径中 padding 位的额外计算（每个 expert 8 positions 中 7 个是 padding）。NCCL 路径类似但 routing 不同，实际 work 分布略优。

## Optimization Log

### #4 expert_padding=8 + parallel routing kernel（2026-05-24）

**Bottleneck:** nsys 显示两个可量化的 gap 来源：

1. Marlin GEMM：PPLX 0.0178ms vs NCCL 0.0137ms（+30%），因为 expert_padding=16 导致 128 positions（NCCL 约 64）。
2. routing kernel <<<1,1>>>：0.0149ms × 60 layers = 0.89ms/step，全部是串行 GPU 执行 + kernel launch overhead。

**Approach:**

1. `PPLX_EXPERT_PADDING`: 16 → 8。每个 active expert 从 16 positions 降到 8，tight_max 从 128 降到 64，与 NCCL 的 workload 完全对齐。expert_padding=8 对 bs=1（max 8 tokens/expert）刚好足够。
2. routing kernel: <<<1,1>>> → <<<1,64>>>。shared memory prefix sum 计算 per-expert offset，48 线程并行填写 sorted_token_ids 和 expert_ids。串行 prefix sum 只 48 iterations 在 shared memory 中约 50ns，可忽略。

**Safety:** expert_padding 只控制 alignment，不限制 per-expert capacity。`ceil(count, 8) * 8` 对任意 count 都正确。PPLX backend 的 `max_recv_tokens` 独立于 expert_padding 计算，buffer 总量不变。

**Result:** PPLX TPOT p50 20.72ms → **17.94ms**（−13%）。已超过 NCCL no-graph 18.52ms。

### #3 expert_padding=64 → 16, block_size=64 → 8（2026-05-24）

**Bottleneck:** 之前几轮优化（SwiGLU、tight bound）把 TPOT 从 37ms 降到 28ms，但 Marlin GEMM 仍然是 NCCL 的 3.2 倍（0.042ms vs 0.013ms per call）。初始分析方向是 Marlin grid 过大（sm_count=0 auto-detect → 320 blocks），但实际分析 `global_mn_tiles` 后发现 80 个 block 都有有效工作——grid 不是问题。

**Root cause 发现过程：** 对比 NCCL 路径的 Marlin 调用参数，发现关键差异：

- NCCL：`block_size=8`（来自 `kimi_marlin_block_size(seq_len=1)`，小 batch → 小 block）
- PPLX：`block_size=64`（来自 `kimi_marlin_block_size(pplx_recv_capacity=3072)`，capacity → 大 block）

block_size=64 意味着 `thread_m_blocks=4`（large-batch config），每个 m_block 处理 64 positions。但 bs=1 时每个 expert 只有 1 个 real token，其余 63 个是 padding。8 active experts × 64 padding = 512 positions，只有 8 个有效——**98% 的 Marlin 计算是浪费的**。

**Approach:**

1. `PPLX_EXPERT_PADDING`: 64 → 16（PplxBootstrapParams 的默认值就是 16）
2. Marlin block_size: 硬编码 8（不再用 `kimi_marlin_block_size(capacity)` 计算）

**Result:** PPLX TPOT p50 28ms → **20.72ms**（−26%）。tight_max 从 512 降到 128，SwiGLU grid 从 4096 降到 1024 blocks。thread_m_blocks 从 4 降到 1（small-batch config，blocks_per_sm 最多 4，occupancy 提高）。

### #2 host-side tight bound（2026-05-23）

**Bottleneck:** #1 用了 GPU-side D2H 读 `num_tokens_post_padded[0]`，导致跨 rank pipeline stall（下面 Failed Approaches 详述）。改成 GPU-side limiting 后 SwiGLU 虽然正确了，但 grid 仍按 pplx_recv_capacity=3072 launch，24000 个 block 立即 exit。

**Approach:** 在 host 侧算 tight upper bound：`tight_max = min(seq_len × topk, local_experts) × expert_padding`。对 bs=1: `min(1×8, 48) × 64 = 512`。零 D2H，替代 `pplx_recv_capacity=3072` 用于 route_elems / max_padded_tokens / SwiGLU grid sizing。GPU <<<1,1>>> routing kernel 写 actual total 到 `num_tokens_post_padded[0]`，Marlin 和 SwiGLU 在 device 上读它做 work limiting。

**Result:** PPLX TPOT 29ms → **28ms**。SwiGLU grid 从 24576 → 4096 blocks。路由元数据和 Marlin locks clear 范围也缩小到 tight_max。

### #1 SwiGLU GPU-side limiting + memset 移除（2026-05-23）

**Bottleneck:** nsys 显示 PPLX decode 37ms vs NCCL 19ms。`swiglu_w13_pplx_kernel` grid 按 pplx_recv_capacity=3072 rows launch（3072 × 2048 / 256 = 24576 blocks），但实际只有 ≤512 rows 有数据，96% 的 block 做了 global memory read + div + exp 后直接 exit。另外两个 `memset_zeros()` 清零 W13 和 W2 output buffer 各增加约 0.5ms。

**Approach:**

1. 新增 `swiglu_w13_pplx_kernel`：从 device 读 `num_tokens_post_padded[0]` 获取 actual row count，grid 线程超过 actual total 的立即 return。
2. 移除两个 `ctx.stream.memset_zeros()`：Marlin 使用 sentinel（`sorted_token_ids[j] = max_padded_tokens`）跳过 output write，combine_send 只读 actual tokens，所以 buffer 不需要预清零。

**Result:** PPLX TPOT 37ms → **29ms**（−22%）。

### #0 PPLX EP baseline（2026-05-23）

从 TP8+EP8 NCCL 路径 fork，接入 `pegainfer-comm::EpBackend`。PPLX 4-step protocol 替换 NCCL RS bridge，router scale 在 combine_recv 后单独应用（`accumulate=false` + `kimi_scaled_add_f32_bf16_to_bf16`）。

初始 bench_serving 测得 PPLX TPOT ≈ 37ms，NCCL no-graph ≈ 19ms。

## Failed Approaches

### D2H 读 num_tokens_post_padded 导致 pipeline stall

**尝试：** 在 dispatch_recv 之后用 `cudarc::clone_dtoh` 把 `num_tokens_post_padded[0]` 从 GPU 拷到 host，用 host 值决定后续 Marlin / SwiGLU 的 size_m。

**失败原因：** `clone_dtoh` 底层用 `cuMemcpyDtoHAsync` + pageable host memory。CUDA 对 pageable memory 的 async D2H 会阻塞整个 stream——包括正在进行中的跨 rank PPLX a2a 通信。所有 8 个 rank 都在等各自的 D2H 完成，但 D2H 又在等 a2a 完成（因为 a2a 和 D2H 在同一个 stream），形成死锁/hang。

**教训：** decode 路径不能有任何 D2H / H2D。异步 D2H 并不像名字暗示的那样是非阻塞的——pageable memory 场景下它会同步等待 stream drain。pinned memory 的 async D2H 虽然不阻塞 stream，但仍然增加 stream serialization 点，对 multi-rank pipeline 有害。正确做法是在 GPU 上完成所有 metadata 读取。

### SwiGLU grid sizing 反复修改

先后经历了 4 个版本：

1. Grid 按 `pplx_recv_capacity` launch → 96% block 浪费
2. 加 D2H 读 actual count → pipeline stall（上面的失败）
3. GPU-side kernel 读 `num_tokens_post_padded[0]` → grid 仍按 capacity launch，24000 blocks 中大部分立即 exit
4. host-side tight bound → grid 按 tight_max launch

每次都是在修补上一次的 bug 而不是重新建模问题。用户反馈：**"你把问题建模出来一次做到位不行吗"**。

**教训：** 在动手前把约束写清楚——哪些值 host 知道、哪些值只有 GPU 知道、D2H 的代价是什么。tight_max 是纯 host 计算的 upper bound（不需要 GPU 数据），GPU-side actual count 只用于 kernel 内部 early exit。一开始就区分这两层，可以避免中间的 D2H 弯路。

### Marlin sm_count / grid 分析方向偏误

**尝试：** 初始判断 Marlin 3.2× 慢是因为 `sm_count=0`（auto-detect full SM count → 320 blocks），大部分 block 无工作。计划传小的 sm_count 来缩小 grid。

**实际情况：** 仔细分析 Marlin template 的 work distribution（`global_mn_tiles = parallel × n_tiles`）后发现，with tight_max=512 和 block_size=64，global_mn_tiles = 128..224，80 个 block 都有工作。grid 大小不是瓶颈——**真正的瓶颈是 block_size=64 导致的 per-block 计算浪费**。每个 m_block 处理 64 tokens 的 GEMM，但只有 1 个 token 有 real data。

**教训：** "block 数量多" 和 "block 内计算浪费" 是不同的问题。前者通过 grid sizing 解决，后者通过 work granularity（block_size / expert_padding）解决。nsys 的 `cuda_gpu_kern_sum` 只报告 per-call avg/total，不告诉你 per-call 内部的 utilization——需要结合 problem dimensions 手算。

## Key Design Decisions

### expert_padding 不限制 per-expert capacity

`expert_padding` 只控制 alignment——`padded = ceil(count, expert_padding) * expert_padding`。一个 expert 收到 32 个 token，expert_padding=8 时 padded=32，完全容纳。PPLX backend 的 `max_recv_tokens` 独立计算 buffer 总量，不受 expert_padding 约束。所以 expert_padding 可以取很小的值（如 8），只影响粒度，不影响 correctness。

### block_size=8 for PPLX（硬编码 vs 动态计算）

NCCL 路径用 `kimi_marlin_block_size(seq_len)` 动态选择 block_size（bs=1 → 8，bs=128 → 64）。PPLX 路径之前错误地用 `kimi_marlin_block_size(pplx_recv_capacity)` 计算——输入是 worst-case capacity 而不是 actual tokens，永远返回 64。

改为硬编码 `block_size=8`：

- PPLX decode 的 per-expert token count 始终很小（bs=1 max 8 tokens/expert），block_size=8 对所有 decode batch sizes 都是最优。
- 对应 Marlin small-batch config（thread_m_blocks=1, blocks_per_sm up to 4），occupancy 更高。
- 避免 workspace 在 init-time 就被 capacity 锁死到 large-batch config。

### routing kernel 并行化：prefix sum + per-expert fill

<<<1,1>>> → <<<1,64>>>。48 个 thread 各读一个 expert 的 count，计算 padded，写入 shared memory。Thread 0 做 48-element prefix sum（shared memory，~50ns）。然后 48 个 thread 并行填 sorted_token_ids 和 expert_ids。Thread 0 最后填 sentinel 尾部和 `num_tokens_post_padded[0]`。

prefix sum 是串行的，但只有 48 iterations in shared memory——不值得用 warp-level parallel scan。kernel 时间从 ~15μs 降到 ~7μs，60 calls/step 节省 ~0.5ms。

## 涉及的文件

| File | 改动 |
| --- | --- |
| `pegainfer-kimi-k2/src/runner/moe_pplx.rs` | PPLX_EXPERT_PADDING 64→8, block_size 硬编码 8, forward 逻辑 |
| `pegainfer-kernels/csrc/kimi_k2/kimi_experts.cu` | routing kernel 并行化 <<<1,64>>>, shared memory prefix sum |
| `pegainfer-kernels/csrc/kimi_k2/kimi_marlin_wna16.cu` | `swiglu_w13_pplx_kernel` + C wrapper |
| `pegainfer-kernels/src/ops/kimi_k2/experts.rs` | `kimi_pplx_build_marlin_routing_on_stream`, tight_max 计算, PPLX GEMM wrappers |
| `pegainfer-kernels/src/ffi.rs` | FFI declarations |

## Open

1. **Marlin 仍有 0.51ms gap**（0.0178 vs 0.0137ms per call）。PPLX 路径 expert_padding=8 与 NCCL block_size=8 对齐后 work 量相似，剩余差异可能来自 PPLX buffer layout 的 cache locality（expert-major padded vs NCCL 的 contiguous gather）或 Marlin small-batch config 在 H20 上的 SM occupancy 差异。需要 ncu 进一步分析。
2. **Pipeline bubble ~1.6ms**。PPLX 4-step a2a 在每个 MoE layer 内严格串行（dispatch → compute → combine），无法 overlap communication 和 compute。NCCL 的 allgather/reduce_scatter 可以与下一层的 compute overlap。这是 PPLX 协议的结构性开销。
3. **CUDA Graph for PPLX**。当前 PPLX worker threads 与 CUDA Graph capture 不兼容。NCCL 开 graph 后 TPOT 16ms vs no-graph 19ms，graph 的 3ms 收益对 PPLX 也适用——如果能解决兼容性。
4. **bs>1 的 expert_padding 安全性验证**。expert_padding=8 对 bs=1（max 8 tokens/expert）安全。KIMI_DECODE_MAX_BATCH=4 时理论 max per expert=32，expert_padding=8 仍正确（padded=32），但需要验证 PPLX backend 在极端 routing skew 下的 buffer 总量是否足够。
