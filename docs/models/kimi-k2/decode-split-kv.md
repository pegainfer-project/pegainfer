# Kimi-K2.5 Decode MLA 的 KV 分片并行

> **TL;DR:** decode 阶段的 MLA attention 现在把每个请求的 KV 读切成多片、铺到多个 CTA 上（flashinfer 的 partition-KV + merge 路径）。8×H20 实测 7168 入 / 1024 出：单请求 110.0 s → 20.3 s（9.31 → 50.4 tok/s，6.0×），4 并发 33.6 → 114.6 tok/s（4.7×），8 并发 63.2 → 175.5 tok/s（3.4×），decode 每步 105.4 ms → 17.5 ms，TTFT 不变（prefill 未改）。每层 MLA decode kernel 1.60 ms → 53.8 µs，attention 在步长里的占比 83% → 14%。decode arena 同时统一到 batch 行索引：query、page table、KV 追加与合并输出落在同一行，slot 与行混用读错 query / 输出行的缺陷随之消失。

**Last touched:** 2026-11

## 为什么做

TP8/DP1 路径在 8k 上下文时，decode 每层每请求只有一个 CTA 读完整段 KV：`kv_tile_indices` 全 0、`kv_chunk_size` 是整段长度。一个 CTA 落在 78 个 SM 中的一个上，全机 8 个 rank 合计 8 个 CTA，其余 SM 闲置。实测账（8×H20，7168 入 / 1024 出，单请求）：

- 总时长 110.0 s，其中 decode 107.9 s（1024 步）
- 每步 105.4 ms；作者在同一形状的 nsys 窗口里量到 attention 占 83%
- `BatchDecodeWithPagedKVCacheKernelMLA` 每层 1.60 ms，每步 61 层约 97.6 ms
- 每层每请求读 7168 × 1152 B ≈ 8.3 MB，1.6 ms 对应约 5.2 GB/s，远低于单卡带宽

这条成本随 KV 线性增长，是 8k 以上上下文唯一随长度涨的开销，也是这次只改它的原因。

## 做法

flashinfer 的 MLA decode 自带 partition-KV：`BatchDecodeWithPagedKVCacheDispatchedMLA` 的 `tmp_v` 非空时，按 `padded_batch_size` 起 CTA（一个 CTA 一个"行"），每行只算自己那段 KV，再用 `VariableLengthMergeStates` 按 `o_indptr` 把同一请求的若干行合并回一个输出行。

改动把这条既有路径接起来，不再自己写 kernel：

- **一行 = 一个 (batch 行, KV 分片)**。arena 的 `row_batch` 是该行服务的 batch 行（FlashInfer 的 `request_indices`），`row_tiles` 是该行负责的分片号（`kv_tile_indices`），`tile_tokens` 是统一分片长度（`kv_chunk_size`，kernel 只读第 0 个元素），`row_valid` 关掉这一步用不到的行，`row_indptr` 给出每个 batch 行的行区间供合并使用（`o_indptr`）。
- **decode arena 只有一个索引空间：batch 行**。第 r 行的 query、page-table 行、KV 追加位置、attention 输出与合并窗口都落在第 r 行。这条不变量顺带修掉一个更早的缺陷：改动前 `request_indices` 填的是请求的 slot 而输出写在行上，一旦批次里出现空位（请求结束后的下一轮，或同一批里混进单 token 请求），query 与 page table 取自 slot、hidden 取自行，读到的是别的请求的 query；分片路径按 slot 建 `o_indptr` 时，缺陷还会扩大到"整行 attention 输出写到 slot 行、下游按行读回旧值"。行索引统一之后，这类不一致没有存在的空间。
- **prefill 仍用 slot 作 page-table 行**：DP8 的 synchronized prefill 依赖每个 rank 各自的 slot，但 CSR 行顺序由调用者给出的闭包决定，所以 prefill 传 `|_| slot`、decode 传 `|row| row`。
- **行容量固定** = `batch_size × tiles`，`tiles = max(64 / batch_size, 1)`。启动几何因此与 KV 长度无关，CUDA graph 可复用；短 KV 只是让尾部行失效，不缩 grid。
- **分片长度** = `max(ceil(最长 KV / tiles), 256)`。256 的下限保证短上下文不会被切得过碎（切碎只增加 merge 的读放大）。
- **每步重算 plan**：`plan_decode_kv_tiles` 生成 `row_batch`、`row_tiles`、`row_valid`、`row_indptr` 与 `tile_tokens` 并上传，代价是几百字节的 H2D，数值路径不变。
- **分片行按 batch 行紧排**：第 r 行的 `min(ceil(kv/tile_tokens), tiles)` 个分片铺在 `row_indptr[r]` 到 `row_indptr[r + 1]` 之间，行区间连续且不含别的行。`VariableLengthMergeStates` 会把窗口内的每一行都当成该行的分片参与合并，所以窗口里不能夹空闲行——否则会在合并时读进被掩码的 partial、丢掉真正的分片。KV 长度不齐的并发请求因此按行紧排，尾部剩余的行全部掩码；空闲行的窗口为空，合并直接写出零行。
- **合并的输出行数取自 `paged_kv.batch_size`**（即 bucket 的 batch），与 `row_indptr` 的 `batch_size + 1` 条目一一对应，合并结果写回同一个 batch 行。

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

8×H20（96 GB，driver 580.126.20），Kimi-K2.6 INT4 pack-quantized 权重，TP8/DP1，greedy，7168 入 / 1024 出，每个请求一条独立提示；两列是同一次会话里先起原树、再起分片后的两个服务进程（prefix cache 都为空），同一份客户端、同一组种子。

| 并发 | 原树 时长 / 吞吐 / ITL p50 | 分片后 时长 / 吞吐 / ITL p50 | 步长倍数 |
| --- | --- | --- | --- |
| 1 | 110.0 s / 9.31 tok/s / 105.44 ms | 20.3 s / 50.43 tok/s / 17.53 ms | 6.0× |
| 4 | 121.6 s / 33.64 tok/s / 110.96 ms | 35.7 s / 114.62 tok/s / 23.71 ms | 4.7× |
| 8 | 129.2 s / 63.22 tok/s / 116.10 ms | 45.5 s / 175.47 tok/s / 34.32 ms | 3.4× |

TTFT p50 分别是 2131/5058/3451 ms（原树）与 2112/5174/4397 ms（分片后）：prefill 没改，差异来自并发下 prefill 与 decode 的排队顺序。

口径：`KIMI_MAX_REQUEST_TOKENS = 8192` 把提示与生成长度一起算，admission 因此在 8192 入 / 1024 出 时直接拒绝（`needs 8199 KV tokens, exceeding the per-request capacity of 8192 tokens`），合法上限就是 7168 入 / 1024 出；表按这个上限测，8k 形状要在本地临时抬高该常量才能跑。

并发越高收益越小，原因是剩下的时间不再由 attention 决定：

| 每卡每步的 kernel 时间构成（单请求 8k 的 nsys 窗口） | 原树 | 分片后 |
| --- | --- | --- |
| attention（MLA decode + merge） | 83% | 14% |
| NCCL | 5% | 50% |
| GEMM | 3% | 13% |
| MoE | 2% | 11% |
| elementwise / norm / proj | 1% | 4% |
| 步长（wall，7168 入的压测 ITL p50） | 105.4 ms | 17.5 ms |

上表两列都按占步长的比例算，口径一致所以能直接对比。比例取自作者在 8×H20 上抓的 nsys 窗口（原树 8 步、分片后 12 步），本机没有重抓，所以绝对步长一栏换成压测的 ITL p50；追踪会抬高单请求步长，占比仍是可比的。分片后 `BatchDecodeWithPagedKVCacheKernelMLA` 每层 53.8 µs（原 1.60 ms），merge 只花 3.8 µs。步长降到十几毫秒后，每步 123 次 all-reduce 与 60 次 reduce-scatter 的固定开销、以及 61 层 expert 权重流成为主要成本——短步长不再能把 launch 延迟藏在长 attention kernel 后面，这也是 batch 越大收益越小的直接原因。

## 正确性口径

- token 级相等在这条路径上不可用：**同一个提示的 greedy 输出随批次组成变化**。本轮实测，同一段「梯度下降」提示单独跑与放进三行批次里跑，输出到第 68 个字符才出现措辞差异（语义一致），差异来自 TP8 的 all-reduce 求和顺序与 expert split-K 归约，改动前就存在。所以闸门只能卡在「批次形状固定的行为复现 + 纯函数单测」上。
- **混合批次的行为复现**：同一波里放「普通提示 + 4 token 提示 + 普通提示」，第三个请求在原树上从第 9 个字符开始崩成 ` sof- editor` / ` sof8h` 一类碎片（同一提示单独跑是通顺的：`梯度下降是一种通过不断沿着当前最陡下降方向调整参数…`），行索引修正后同批次输出保持通顺，只与单独运行差一处分句措辞。这条复现就是「空位出现后行与 slot 不再对应」的实际后果，也是这次把 decode 统一到行索引的直接理由。
- 因此本次的锚点是：① 256 token 上下文只占 1 个 tile，走的是与改动前相同的单行形状、同一个 kernel；② `plan_decode_kv_tiles` 与 `build_page_table` 的纯函数单测（分片切分、短 KV 单分片、长短混合时按行紧排、空闲行掩码与空窗口、page table 行闭包与重复行拒绝、参数校验）；③ 只改 KV 切分与行索引，其余 kernel 与缓冲区语义不变，合并走 flashinfer 自己的 `VariableLengthMergeStates`（与其余 MLA 前向共用同一份实现），只改变浮点求和顺序。
- `logprobs` 在 TP8 路径返回 HTTP 500，无法用 logit 差做 gate。精度回归要跑同机 DP8 路径的 `vllm_golden_gate`（K2.6 权重 + 已提交 fixture）：在 96 GB 的 H20 上，原树与分片版都在装载阶段以同一个 `vstack alloc failed: CUDA_ERROR_OUT_OF_MEMORY` 失败（每 rank 权重 89.2 GiB），需要 141 GB 机型（文档记录的绿灯是 8×H200）。

## 代价与边界

- 每层多一次 merge kernel launch（单请求 3.8 µs）。256 token 上下文实测未变慢（0.45 s 对 0.46 s，16 token 输出）。
- 行索引修正本身不改变步长：同一台机器上分片版（含修正）与原分片版的 ITL p50 在噪声内一致（17.53 / 23.71 / 34.32 ms 对 18.48 / 23.34 / 34.97 ms，并发 1/4/8）。
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

./target/release/pegainfer --model-path /mnt/qjh000/models/Kimi-K2.6 \
  --served-model-name kimi-k2.6 --port 8124 \
  --tp-size 8 --dp-size 1 --cuda-graph false --ep-backend nccl
```

`--model-path` 指向本机的 INT4 checkpoint；上表在一台 8×H20（driver 580.126.20，`/mnt/qjh000/models/Kimi-K2.6`）上测得。另一台 8×H20（driver 535.161.08）装不下 CUDA 13 运行时，本轮没有参与测量。

客户端每个请求发一条独立的 7168-token 提示（`--seed` 派生的 LCG，避免 prefix cache 命中），`ignore_eos`、`temperature=0`、`stream=true`，逐 chunk 记到达时间算 TTFT 与 ITL；两棵树各自重启服务进程后测，同一份客户端、并发取 1/4/8。

逐步与 kernel 归因：

```bash
nsys launch --session-new=k2skv --trace=cuda ./target/release/pegainfer <serve args>
nsys start --session=k2skv --sample=none --cpuctxsw=none -o b1decode
# 跑一个 7168 入 / 32 出的请求
nsys stop --session=k2skv && nsys shutdown --session=k2skv --kill=sigterm
nsys export --type sqlite -o b1decode.sqlite b1decode.nsys-rep
```

统计按 `CUPTI_ACTIVITY_KIND_KERNEL` 的 `start/end`（纳秒）切窗口，`JOIN StringIds ON id = demangledName` 取名，逐 `deviceId` 汇总即可。窗口里的步数按 attention launch 数推（该卡 attention 次数 ÷ 61 层），本文的窗口因此是 12 步；追踪会明显抬高单请求步长，所以上表只用占比、不取绝对值。
