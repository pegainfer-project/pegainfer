# Kimi-K2.5 Decode MLA 的 KV 分片并行

> **TL;DR:** decode 阶段的 MLA attention 现在把每个请求的 KV 读切成多片、铺到多个 CTA 上（flashinfer 的 partition-KV + merge 路径）。8×H20 实测 8k 入 / 1k 出：单请求 123.2 s → 20.8 s（8.31 → 49.2 tok/s，5.9×），12 并发 74.7 → 148.0 tok/s（2.0×），decode 每步 118.0 ms → 17.6 ms，TTFT 不变（prefill 未改）。每层 MLA decode kernel 1.60 ms → 53.8 µs，attention 在步长里的占比 83% → 15%。

**Last touched:** 2026-10

## 为什么做

TP8/DP1 路径在 8k 上下文时，decode 每层每请求只有一个 CTA 读完整段 KV：`kv_tile_indices` 全 0、`kv_chunk_size` 是整段长度。一个 CTA 落在 78 个 SM 中的一个上，全机 8 个 rank 合计 8 个 CTA，其余 SM 闲置。实测账（8×H20，8k 入 / 1k 出，单请求）：

- 总时长 123.2 s，其中 decode 120.8 s
- 每步 118.0 ms，attention 占 97.9 ms（83%）
- `BatchDecodeWithPagedKVCacheKernelMLA` 每层 1.60 ms，每步 61 层约 97.6 ms
- 每层每请求读 8192 × 1152 B ≈ 9.4 MB，1.6 ms 对应约 5.9 GB/s，远低于单卡带宽

这条成本随 KV 线性增长，是 8k 以上上下文唯一随长度涨的开销，也是这次只改它的原因。

## 做法

flashinfer 的 MLA decode 自带 partition-KV：`BatchDecodeWithPagedKVCacheDispatchedMLA` 的 `tmp_v` 非空时，按 `padded_batch_size` 起 CTA（一个 CTA 一个"行"），每行只算自己那段 KV，再用 `VariableLengthMergeStates` 按 `o_indptr` 把同一请求的若干行合并回一个输出行。

改动把这条既有路径接起来，不再自己写 kernel：

- **一行 = 一个 (请求, KV 分片)**。`request_indices` 是该行的 slot，`kv_tile_indices` 是该行负责的分片号，`kv_chunk_size` 是统一分片长度（kernel 只读第 0 个元素），`row_valid` 关掉这一步用不到的行，`slot_indptr` 给出每个请求的行区间供合并使用。
- **行容量固定** = `batch_size × tiles`，`tiles = max(64 / batch_size, 1)`。启动几何因此与 KV 长度无关，CUDA graph 可复用；短 KV 只是让尾部行失效，不缩 grid。
- **分片长度** = `max(ceil(最长 KV / tiles), 256)`。256 的下限保证短上下文不会被切得过碎（切碎只增加 merge 的读放大）。
- **每步只重算 plan**：`plan_decode_kv_tiles` 生成 `row_valid` 与 `slot_indptr` 并上传，代价是几百字节的 H2D，数值路径不变。
- **行按请求成群排列**：请求 `r` 占 `[r*tiles, (r+1)*tiles)`。合并 kernel 的输出行数取自 `paged_kv.batch_size`（即 bucket 的 batch），与 `slot_indptr` 的 `batch_size + 1` 条目一一对应，所以行序必须与 decode 行序一致，一个请求的分片必须连续。

缓冲按 arena 分配：`partial_ckv` = 行数 × 本地头数(8) × 512，`partial_lse` = 行数 × 8。64 行时分别是 512 KiB 与 2 KiB，相对每卡 86 GiB 的权重可以忽略。

| decode bucket | tiles | 行数 |
| --- | --- | --- |
| 1 | 64 | 64 |
| 2 | 32 | 64 |
| 4 | 16 | 64 |
| 8 | 8 | 64 |
| 16 | 4 | 64 |
| 32 | 2 | 64 |
| 64 | 1 | 64 |

batch 64 时退化为单分片，与改动前等价；这条设计让常驻行数恒为 64，跨 bucket 的 grid 一致。

## 实测对照

8×H20（96 GB，driver 580.126.20），Marlin INT4 pack-quantized 权重，TP8/DP1，greedy，同一个压测客户端、同一组种子；两列是同一棵上游树（改动前 / 改动后），也就是这次改动要落到的基线。

| 并发 | 原树 时长 / 吞吐 / ITL p50 | 分片后 时长 / 吞吐 / ITL p50 | 吞吐倍数 |
| --- | --- | --- | --- |
| 1 | 123.2 s / 8.31 tok/s / 118.04 ms | 20.8 s / 49.22 tok/s / 17.62 ms | 5.92× |
| 4 | 135.8 s / 30.15 tok/s / 124.38 ms | 34.5 s / 118.80 tok/s / 24.53 ms | 3.94× |
| 8 | 151.6 s / 54.05 tok/s / 129.95 ms | 56.8 s / 144.30 tok/s / 37.24 ms | 2.67× |
| 12 | 164.5 s / 74.69 tok/s / 133.53 ms | 83.0 s / 147.96 tok/s / 54.04 ms | 1.98× |
| 16（每次 8 个） | 302.3 s / 54.19 tok/s / 129.80 ms | 113.5 s / 144.33 tok/s / 37.10 ms | 2.66× |

TTFT p50 五档分别是 2403/7427/11860/16331/11702 ms（原树）与 2639/6958/11846/16368/11733 ms（分片后），差异落在 prefill 抖动内。

并发越高收益越小，原因是剩下的时间不再由 attention 决定：

| 每卡每步的 kernel 时间构成（单请求 8k） | 原树 | 分片后 |
| --- | --- | --- |
| attention（MLA decode + merge） | 83% | 14% |
| NCCL | 5% | 50% |
| GEMM | 3% | 13% |
| MoE | 2% | 11% |
| elementwise / norm / proj | 1% | 4% |
| 步长（wall） | 118.0 ms | 17.6 ms |

上表两列都按占步长的比例算，口径一致所以能直接对比。比例取自 nsys 窗口（原树 8 步、分片后 12 步）：追踪会明显抬高单请求步长，绝对值偏大，这里只取构成。分片后 `BatchDecodeWithPagedKVCacheKernelMLA` 每层 53.8 µs（原 1.60 ms），merge 只花 3.8 µs。步长降到十几毫秒后，每步 123 次 all-reduce 与 60 次 reduce-scatter 的固定开销、以及 61 层 expert 权重流成为主要成本——短步长不再能把 launch 延迟藏在长 attention kernel 后面，这也是 batch 越大收益越小的直接原因。

## 正确性口径

- token 级相等在这条路径上不可用：**原树自己跨进程也不一致**。同一个原树二进制（md5 64731e45）、同一个 prompt、temperature 0：1024 入 / 16 出两次启动给出两段不同文本；8192 入 / 16 出同样两次不同；256 入 / 64 出在两棵树上分别得到 `9a8f5481` 与 `1374b6f5`，而分片版自身在另一次进程里又是 `80f7d803`。这份不确定性来自 TP8 的 all-reduce 求和顺序与 expert split-K 归约，改动前就存在。
- 因此本次的锚点是：① 256 token 上下文只占 1 个 tile，走的是与改动前相同的单行形状、同一个 kernel；② `plan_decode_kv_tiles` 的纯函数单测（分片切分、短 KV 单分片、空闲行掩码、参数校验）；③ 只改 KV 切分，其余 kernel 与缓冲区语义不变，合并走 flashinfer 自己的 `VariableLengthMergeStates`（与其余 MLA 前向共用同一份实现），只改变浮点求和顺序。
- `logprobs` 在 TP8 路径返回 HTTP 500，无法用 logit 差做 gate。精度回归要跑同机 DP8 路径的 `vllm_golden_gate`（K2.6 权重 + 已提交 fixture）：在 96 GB 的 H20 上，原树与分片版都在装载阶段以同一个 `vstack alloc failed: CUDA_ERROR_OUT_OF_MEMORY` 失败（每 rank 权重 89.2 GiB），需要 141 GB 机型（文档记录的绿灯是 8×H200）。

## 代价与边界

- 每层多一次 merge kernel launch（单请求 3.8 µs）。256 token 上下文实测未变慢（0.45 s 对 0.46 s，16 token 输出）。
- 行数上限 64，batch 64 时只有 1 个分片；要把大 batch 也分片，需要把行数上限抬到 128/256，代价是 partial 缓冲与 merge 行数同比增加。
- 分片长度下限 256 token：短于 256 的 KV 不会真正并行（此时它本来也不是瓶颈）。
- 与 CUDA graph：行容量固定、grid 不变，每步只在 graph 外更新 plan 缓冲，合并与 attention 的指针都来自 arena。本次测量全部用 `--cuda-graph false`，因为 `--cuda-graph true` 在这个基线上起不来：原树与分片版都在 warm decode admission bs64 阶段以同一条 `Kimi routed RS f32 hidden failed: status=ncclInternalError` 失败（MoE 的 reduce-scatter 在 capture 里报错），与本次改动无关。分片路径的 graph 行为要等那个问题修好后才能验证。

## 下一步

- 8k 上下文下的剩余成本在 collective（每卡每步 4.96 ms）与 expert 权重流（1.09 ms）上，把 attention 再拆没有空间。
- batch 大于 8 时把行容量抬到 128 行，让每个请求仍能拿到多分片。
- 把 merge 与 decode 合成一次 launch，省掉 3.8 µs × 61 的 launch 开销（当前约 0.23 ms/步）。

## 复现

两棵树（`origin/main` 与本次改动）各自构建，同一份权重、同一台机器、同一个客户端：

```bash
cargo build --release --no-default-features --features kimi-k2 -p pegainfer-server --bin pegainfer

./target/release/pegainfer --model-path /data/models/Kimi-K2.5 \
  --served-model-name kimi-k2.5 --port 8124 \
  --tp-size 8 --dp-size 1 --cuda-graph false --ep-backend nccl
```

客户端固定发同一段 8192-token 的 token-id 提示，`ignore_eos`、`temperature=0`、`stream=true`，逐 chunk 记到达时间算 TTFT 与 ITL；两棵树用同一份客户端、同一组种子，并发取 1/4/8/12。

逐步与 kernel 归因：

```bash
nsys launch --session-new=k2skv --trace=cuda ./target/release/pegainfer <serve args>
nsys start --session=k2skv --sample=none --cpuctxsw=none -o b1decode
# 跑一个 8192 入 / 32 出的请求
nsys stop --session=k2skv && nsys shutdown --session=k2skv --kill=sigterm
nsys export --type sqlite -o b1decode.sqlite b1decode.nsys-rep
```

统计按 `CUPTI_ACTIVITY_KIND_KERNEL` 的 `start/end`（纳秒）切窗口，`JOIN StringIds ON id = demangledName` 取名，逐 `deviceId` 汇总即可。窗口里的步数按 attention launch 数推（该卡 attention 次数 ÷ 61 层），本文的窗口因此是 12 步；追踪会明显抬高单请求步长，所以上表只用占比、不取绝对值。
