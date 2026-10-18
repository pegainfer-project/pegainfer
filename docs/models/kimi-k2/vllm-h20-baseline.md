# Kimi-K2 vLLM H20 Baseline (decode-heavy)

> **TL;DR:** vLLM `0.19.0` + Kimi-K2.5 + 8× H20，TP1+DP8+EP8（NCCL allgather/reducescatter all2all）跑 `bench serve` decode-heavy profile（input=1, output=128, ignore-eos）。bs=1..256 扫描。这是 vLLM 侧的 baseline 数据快照，作为 pegainfer TP1+DP8+EP8 active line 的硬上限（pegainfer 当前数据见 [tp1-dp8-ep8-performance.md](tp1-dp8-ep8-performance.md)）。下面的 pegainfer 列是 **历史 TP8+EP8 bs=4 bring-up 对照**（TPOT med `19.13ms` vs vLLM `24.97ms`，HTTP 比 in-process `14.39ms` 高 33%），保留作为 frontend/streaming overhead 的早期记录。
>
> **Last touched:** 2026-06

## 2026-05-25 bs64 warmup-after rerun

The original bs64 row was rechecked because TP1+DP8+EP8 was expected to be
closer to 30ms TPOT on some prior measurements. The rerun used the same H20
node and vLLM TP1+DP8+EP8 server shape, but explicitly ran a full bs64/o128
warmup before the measured bs64/o128 pass.

Artifacts:

- Output dir: `$RESULT_ROOT/kimi-vllm-dp8-warmup-20260525`
- Server log: `$RESULT_ROOT/kimi-vllm-dp8-warmup-20260525/server.log`
- Warmup result: `$RESULT_ROOT/kimi-vllm-dp8-warmup-20260525/warmup_bs64_o128.json`
- Measured result: `$RESULT_ROOT/kimi-vllm-dp8-warmup-20260525/measure_bs64_o128_after_warmup.json`

Server log evidence:

- `Worker_DP0_EP0` through `Worker_DP7_EP7` started on 8 GPUs.
- `Using AgRsAll2AllManager all2all manager`.
- CUDA graph capture ran for `PIECEWISE=51` and `FULL=35`.

| run | reqs | duration (s) | out tok/s | TTFT p50/p99 (ms) | TPOT p50/p95/p99 (ms) | ITL p50/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bs64 warmup | 64 | 13.40 | 611.36 | 341.13 / 345.97 | 102.77 / 103.02 / 103.03 | 104.62 / 107.97 |
| bs64 measured after warmup | 256 | 55.11 | 594.57 | 161.30 / 303.20 | 107.20 / 109.00 / 109.20 | 108.92 / 116.35 |

Result: the explicit warmup improves the old bs64 row slightly
(`109.00/109.76ms` → `107.20/109.20ms` TPOT p50/p99), but the H20 result is
still 100ms-class, not 30ms-class. The 30ms expectation remains an open
cross-hardware/backend check.

## Setup

| 项 | 值 |
| --- | --- |
| GPU | 8× NVIDIA H20（143 GB） |
| Model | Kimi-K2.5（local `$MODEL_DIR`，INT4 + BF16 scale Marlin WNA16） |
| vLLM | `0.19.0`（venv `$VLLM_DIR/.venv`） |
| Sharding | **vLLM**: TP=1, DP=8, EP=8，all2all backend `allgather_reducescatter`（NCCL，默认） |
| Sharding | **pegainfer**: TP=8, EP=8，NCCL F32 hidden bridge + RS routed bridge |
| Profile | input_len=1, output_len=128, `--ignore-eos`, `--random-range-ratio 0` |
| Bench | `vllm bench serve --backend openai --endpoint /v1/completions`（同一 client，两边对齐） |
| 数据 | `$VLLM_DIR/kimi_dp8_baseline/result_*.json` |

## vLLM bs sweep

`vllm bench serve` 同 client 打 vLLM 0.19.0 TP1+DP8+EP8 server。`num_prompts = max(bs*4, 32)`，`request_rate=inf`（asap）。

| bs | reqs | duration (s) | out tok/s | TTFT med (ms) | TTFT p99 (ms) | TPOT med (ms) | TPOT p99 (ms) | ITL med (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1   |   32 |  74.93 |   54.7 |   70.8 |  104.0 |  17.94 |  18.12 |  17.93 |
| 2   |   32 |  43.24 |   94.7 |   60.3 |  121.0 |  20.78 |  22.77 |  19.40 |
| 4   |   32 |  25.94 |  157.9 |   69.6 |  135.4 |  24.97 |  29.46 |  23.02 |
| 8   |   32 |  13.27 |  308.6 |   71.8 |  121.8 |  26.44 |  26.57 |  26.40 |
| 16  |   64 |  17.12 |  478.4 |   79.5 |  109.9 |  33.01 |  33.20 |  33.19 |
| 32  |  128 |  28.96 |  565.7 |  128.7 |  189.3 |  55.96 |  56.25 |  56.55 |
| 64  |  256 |  56.12 |  583.9 |  175.2 |  276.2 | 109.00 | 109.76 | 110.52 |
| 128 |  512 |  85.10 |  770.1 |  287.4 |  446.0 | 165.11 | 165.57 | 166.91 |
| 256 | 1024 | 115.86 | 1131.3 |  451.2 |  669.8 | 223.96 | 224.38 | 225.60 |

### 形态

- **TPOT 在 bs≤8 几乎线性**（17.9 → 26.4 ms，单 token 慢 47%；吞吐 5.7×）。bs=8 是 vLLM TP1+DP8+EP8 "免费午餐" 区——8 路独立请求各自落到一个 DP rank，跨 rank 不需要 reduce。
- **bs=8→16→32**：TPOT 26→33→56 ms。一个 DP rank 上要排 2 / 4 个 decode，rank-local MLA / MoE compute 直接 ×2、×4。
- **bs=64..256**：TPOT 109 / 165 / 224 ms 线性增长；fed-batch 已经塞满，aggregate tok/s 还有但单请求体验恶化。
- **TTFT**：bs≤16 维持在 ~70-130 ms（input 只有 1 token，prefill = 一发 graph-safe forward），bs=256 升到 451 ms，被排队拖到。

### 关键 inflection

| 指标 | bs=8 | bs=256 |
| --- | --- | --- |
| Aggregate output tok/s | `308.6` | `1131.3`（峰值，`3.7×` of bs=8） |
| Per-request TPOT med | `26.4ms`（≈38 tok/s/req） | `224.0ms`（≈4.5 tok/s/req） |
| TTFT med | `71.8ms` | `451.2ms` |

**bs=8 ≈ 拐点**：从这一点开始多塞 batch 单请求体验快速恶化，aggregate throughput 增益逐渐被 8 倍 batch / 8 倍 latency 抵消。Decode 性能口径下 vLLM TP1+DP8+EP8 的 "sweet spot" 是 bs=8（8 路 DP 各自 bs=1）。

## pegainfer bs=4 对照点

下表是历史 TP8+EP8 bring-up 对照（当时 `KIMI_RUNNER_MAX_BATCH=4`，没扫 bs；该 const 现在是 `64`，bucketed）。同 client / 同 profile（`vllm bench serve`，input=1, output=128, ignore-eos, max_concurrency=4）打 pegainfer OpenAI-compatible server：

| 指标 | pegainfer TP8+EP8（HTTP, vllm bench） | vLLM TP1+DP8+EP8 bs=4 | pegainfer in-process bench, bs4 |
| --- | ---: | ---: | ---: |
| TPOT median | `19.13ms` | `24.97ms` | `14.39ms` |
| TPOT p99    | `23.63ms` | `29.46ms` | `14.83ms` |
| ITL median  | `17.42ms` | `23.02ms` | — |
| TTFT median | `313.10ms` | `69.60ms` | — |
| TTFT p99    | `4239.97ms` | `135.40ms` | — |
| Output tok/s | `159.99` | `157.94` | `≈278` |

数据来源：
- pegainfer HTTP：`result_pegainfer_bs4.json`，server 是 `target/release/pegainfer --model-path $MODEL_DIR --port 8124 --cuda-graph true`，client 同 vLLM 那条 bench。
- vLLM bs=4：上面 sweep 表的 bs=4 行。
- pegainfer in-process：`bench_serving request --cuda-graph true --concurrency 4`，见 optimization.md。

### 结论

1. **同硬件、同 client、同 profile，pegainfer TPOT 比 vLLM 低 23%**（`19.13 vs 24.97`）。预期内：pegainfer 走 TP=8 把单 token MLA / dense / shared expert 的 GEMM 切到 8 rank，每发 token 跨 rank reduce 一次；vLLM TP=1 时单 rank 自己跑完整 GEMM，靠 DP=8 拿 throughput 但单请求慢。Decode latency 主线上 TP8 仍然赢。

2. **TTFT 这边 vLLM 完胜**：median `69.60ms` vs pegainfer `313.10ms`。pegainfer p99 飙到 `4239.97ms`——基本是 first-request 冷启动（first NCCL collective stream drain + scheduler warmup）。decode 优先的方案在 prefill 路径上欠的债集中爆在 p99。

3. **HTTP overhead 异常高**：pegainfer 同 bs=4，HTTP 口径 TPOT med `19.13ms`，in-process bench 是 `14.39ms`——4.74ms / token，~33% overhead。streaming JSON + frontend bridge 不该这么多。**这条单独提出来作为后续要查的 finding**，优先级介于 decode kernel 和 prefill 之间。

4. **Aggregate throughput 不公平比较（历史）**：当时 pegainfer 卡在 `KIMI_RUNNER_MAX_BATCH=4`（现已是 `64`，bucketed）不能扫 bs，vLLM TP1+DP8 在 bs=256 拉到 `1131 tok/s`。这条数据当时给 TP1+DP8+EP8 milestone 提供了上限：H20 ×8、相同 client 口径，**vLLM TP1+DP8+EP8 baseline 单请求 TPOT `17.94ms`（bs=1）/ aggregate `1131 tok/s`（bs=256）**。pegainfer 的 TP1+DP8+EP8 已落地，bs64 service output `1336 tok/s` / TPOT p50 `47.3ms`（见 [tp1-dp8-ep8-performance.md](tp1-dp8-ep8-performance.md)）。

## 复现命令

vLLM server（H20 node）：

```bash
source $VLLM_DIR/.venv/bin/activate
vllm serve $MODEL_DIR \
  --trust-remote-code \
  --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
  --served-model-name kimi-k2.5 \
  --port 8123 --max-num-seqs 256 --max-model-len 4096
```

pegainfer server（8×H20 node）。Build 用 `cargo build --release -p pegainfer-server --features kimi-k2 --bin pegainfer`，parallel shape 由 CLI flag 选（当前 active line 是 TP1+DP8+EP8 PPLX，下面的 flag 即对齐 vLLM 形态做 apples 对照）：

```bash
LD_LIBRARY_PATH=$RESULT_ROOT/pegainfer-nccl-lib:$LD_LIBRARY_PATH \
  $PEGAINFER_DIR/target/release/pegainfer \
  --model-path $MODEL_DIR --port 8124 --cuda-graph true \
  --tp-size 1 --dp-size 8 --ep-backend pplx
```

> 注：表里的 pegainfer 数据是历史 TP8+EP8 bs=4 口径（当时用旧的 `kimi-k2-pplx-ep` feature / `PEGAINFER_KIMI_PARALLEL` env，二者均已移除）。上面是当前 CLI 复现 active TP1+DP8+EP8 的命令，不是产生表里 TP8 数据的命令；8×H20 才能跑。

bench（client 端，对哪个 server 改 `--base-url` 即可）：

```bash
vllm bench serve \
  --backend openai \
  --model $MODEL_DIR --tokenizer $MODEL_DIR --trust-remote-code \
  --base-url http://127.0.0.1:8124 --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 1 --random-output-len 128 --random-range-ratio 0 \
  --num-prompts 32 --max-concurrency 4 --ignore-eos \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99
```

完整 sweep 脚本：`$VLLM_DIR/kimi_dp8_baseline/kimi_dp8_sweep.sh`。
