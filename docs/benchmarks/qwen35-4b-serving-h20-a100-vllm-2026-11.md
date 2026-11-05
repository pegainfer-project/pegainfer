# Qwen3.5-4B 单卡端到端：H20 与 A100 上的 PegaInfer vs vLLM（2026-11）

**TL;DR**: 两台机器各自单卡、各自用同机的 `vllm bench serve` 客户端、同一组用例，各跑两轮、零失败。H20 上 vLLM 0.29.0 的 decode 步时全面领先：c16 每步 6.33 ms 对 PegaInfer 默认姿态 8.83 ms（慢 39%），qps16 15.34 对 20.53 ms（慢 34%），输出吞吐 c16 1749.9 对 1428.7 tok/s、qps16 1401.1 对 1114.8 tok/s；只有 bs1 咬住（3.84 对 4.03 ms）。`--decode-overlap stream --max-batch 32 --qwen35-scheduler-policy auto` 姿态把 c8/c16 的 TTFT 压到 vLLM 的 0.31/0.37 倍、qps16 的 ITL p99 压到 44.0 ms（vLLM 两轮 175.0 与 356.1 ms），代价是 c16 步时 9.68 ms、c8 ITL p99 33.8 ms。同一棵代码树在 A100-40GB 上对 vLLM 0.31.0 的差距小得多（c16 步时慢 9%、qps16 慢 17%）：H20 的 4.0 TB/s 显存带宽只被 PegaInfer 用掉一部分——c16 步时从 A100 到 H20 只缩 1.40 倍，vLLM 缩 1.79 倍；qps16 几乎不动（21.53 → 20.53 ms），vLLM 缩 1.20 倍。

## 环境与口径

| 项目 | A100 机器 | H20 机器 |
| --- | --- | --- |
| GPU | 8× A100-SXM4-40GB（sm_80），单卡，其余卡闲置 | 8× H20 96 GB（sm_90），单卡，其余卡闲置 |
| 显存带宽 | 1.555 TB/s | 4.0 TB/s |
| PegaInfer | main 上的 `9ef13bc5`，release `--features qwen35`，nvcc 目标 sm_80 | 同一棵代码树，nvcc 目标 sm_90 |
| vLLM 服务端 | 0.31.0 | 0.29.0 |
| 客户端 | `vllm bench serve` 0.31.0 | `vllm bench serve` 0.29.0 |
| 模型 | Qwen3.5-4B BF16 TP1 纯文本 | 同名权重另一份拷贝 |

PegaInfer 启动参数：`--model-path <model> --served-model-name q35 --port 18099`；overlap 姿态追加 `--decode-overlap stream --max-batch 32 --qwen35-scheduler-policy auto`（`--max-batch` 上限由共享 SM 的 decode 路径决定，stream 姿态只接受 32）。H20 上 KV 池 134551 blocks × 16 = 2,152,816 tokens，`max_model_len` 262144。

vLLM 启动参数（两边相同）：`--dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 64 --no-enable-prefix-caching --disable-log-stats --uvicorn-log-level warning --language-model-only`。`--max-num-seqs 64` 与 PegaInfer 默认 `max_batch 64` 相同。

客户端参数（两边相同）：`--backend openai --dataset-name random --random-range-ratio 0 --temperature 0 --ignore-eos --seed 42 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99`，`--tokenizer` 指向本地权重目录（不传会去解析模型名 `q35`，离线环境里会挂住）。

用例：bs1 = 1024 输入 / 256 输出 / 20 条 / 并发 1；c8 与 c16 = 同长度 / 64 条 / 并发 8、16；qps16 = 1024 输入 / 128 输出 / 64 条 / 最大并发 32 / 到达速率 16；8k1k = 8192 输入 / 1024 输出 / 8 条 / 并发 1。每个用例两轮，表中列 r2。两轮差异：TPOT ≤ 2.1%，输出吞吐 ≤ 0.9%，TTFT ≤ 12%，ITL p99 在 vLLM 的 qps16 上跨度最大（H20 两轮 356.1 与 175.0 ms）。

## H20 结果

单位：TTFT 与 ITL p99 为毫秒，TPOT 为毫秒/步，吞吐为输出 tok/s。

| 用例 | 指标 | vLLM 0.29.0 | PegaInfer 默认 | PegaInfer overlap |
| --- | --- | --- | --- | --- |
| bs1 @1024/256 | TTFT mean | **77.3** | 81.2 | 83.9 |
| | TPOT mean | **3.84** | 4.03 | 4.02 |
| | ITL p99 | 5.2 | **4.5** | 4.5 |
| | 输出 tok/s | **242.0** | 230.9 | 230.6 |
| c8 @1024/256 | TTFT mean | 432.2 | 422.7 | **135.2** |
| | TPOT mean | **4.91** | 5.85 | 6.80 |
| | ITL p99 | **6.4** | 6.3 | 33.8 |
| | 输出 tok/s | **1215.4** | 1068.9 | 1091.1 |
| c16 @1024/256 | TTFT mean | 723.9 | 611.2 | **266.9** |
| | TPOT mean | **6.33** | 8.83 | 9.68 |
| | ITL p99 | **7.7** | 78.6 | 37.9 |
| | 输出 tok/s | **1749.9** | 1428.7 | 1476.4 |
| qps16 @1024/128 | TTFT mean | 426.1 | 540.0 | **373.1** |
| | TPOT mean | **15.34** | 20.53 | 20.36 |
| | ITL p99 | 175.0～356.1 | 325.3 | **44.0** |
| | 输出 tok/s | **1401.1** | 1114.8 | 1131.9 |
| 8k1k @8192/1024 | TTFT mean | **547.2** | 633.2 | 623.7 |
| | TPOT mean | **4.11** | 4.72 | 4.71 |
| | ITL p99 | 5.4 | **5.2** | 5.4 |
| | 输出 tok/s | **215.7** | 187.5 | 188.3 |

对 vLLM 的比值（默认 / overlap）：c8 TTFT 0.98 / 0.31，c16 TTFT 0.84 / 0.37，c16 TPOT 1.40 / 1.53，qps16 TPOT 1.34 / 1.33、TTFT 1.27 / 0.88、ITL p99 1.86 / 0.25，8k1k TPOT 1.15 / 1.15 与吞吐 0.87 / 0.87。

## A100 结果

同一棵树、同一组用例，服务端 vLLM 0.31.0。

| 用例 | 指标 | vLLM 0.31.0 | PegaInfer 默认 | PegaInfer overlap |
| --- | --- | --- | --- | --- |
| bs1 @1024/256 | TTFT mean | 109.8 | **63.5** | 66.2 |
| | TPOT mean | 8.16 | **8.13** | 8.14 |
| | ITL p99 | 9.3 | **8.5** | 8.7 |
| | 输出 tok/s | 116.8 | **119.8** | 119.5 |
| c8 @1024/256 | TTFT mean | 297.5 | 328.0 | **132.2** |
| | TPOT mean | 9.93 | **9.89** | 10.41 |
| | ITL p99 | 12.1 | **10.9** | 29.6 |
| | 输出 tok/s | 723.5 | 718.3 | **730.9** |
| c16 @1024/256 | TTFT mean | **402.2** | 474.4 | 243.2 |
| | TPOT mean | **11.33** | 12.36 | 12.73 |
| | ITL p99 | 85.5 | 66.9 | **31.9** |
| | 输出 tok/s | **1241.2** | 1127.5 | 1154.7 |
| qps16 @1024/128 | TTFT mean | **246.2** | 338.9 | 274.1 |
| | TPOT mean | 18.45 | 21.53 | 20.36 |
| | ITL p99 | 89.0 | 244.1 | **36.6** |
| | 输出 tok/s | **1287.1** | 1128.9 | 1134.7 |
| 8k1k @8192/1024 | TTFT mean | 502.5 | **440.4** | 445.0 |
| | TPOT mean | **8.37** | 8.63 | 8.65 |
| | ITL p99 | 9.5 | **8.8** | 8.9 |
| | 输出 tok/s | **112.9** | 110.5 | 110.2 |

## 步时随硬件的缩放

同一段代码在两台机器上的每步 decode 时间，以及 vLLM 的同项。带宽比值 4.0 / 1.555 ≈ 2.57 倍。

| 用例 | PegaInfer 默认 A100 → H20 | vLLM A100 → H20 |
| --- | --- | --- |
| bs1 | 8.13 → 4.03（2.02 倍） | 8.16 → 3.84（2.13 倍） |
| c8 | 9.89 → 5.85（1.69 倍） | 9.93 → 4.91（2.02 倍） |
| c16 | 12.36 → 8.83（1.40 倍） | 11.33 → 6.33（1.79 倍） |
| qps16 | 21.53 → 20.53（1.05 倍） | 18.45 → 15.34（1.20 倍） |
| 8k1k | 8.63 → 4.72（1.83 倍） | 8.37 → 4.11（2.04 倍） |

## 数字说明了什么

- **差距随并发放大，且随硬件换代放大。** A100 上 c16 每步只慢 9%、qps16 慢 17%；H20 上变成 39% 与 34%。bs1 两个平台都接近平手（A100 上 PegaInfer 反超，H20 上慢 5%），宽 batch 的每步开销是差距的主要来源。
- **每步时间没有跟着带宽缩。** c16 每步要读一遍 9.3 GB 权重，按带宽算下限是 A100 6.0 ms、H20 2.3 ms；实测 PegaInfer 是下限的 2.1 倍与 3.8 倍，vLLM 是 1.9 倍与 2.7 倍。倍数上升说明每步里有一段不随带宽缩短的固定开销，H20 上它占的比例更大。这一段与 `models/qwen35/decode-kernel-attribution.md` 记录的 GEMM 逐 kernel 打平一致：差距落在每步的 kernel 数量、层数与启动开销上。
- **qps16 几乎不吃硬件。** 21.53 → 20.53 ms，等于把 A100 换成带宽 2.57 倍的卡只换来 5%。同一格 vLLM 缩 1.20 倍。混合到达下每步要插入 prefill，步时由调度的插入方式决定，显存带宽用不上——这与仓库里既有的记录（qps16 归到调度一侧）吻合。
- **默认姿态的尾延迟在 c16 与 qps16 上是数量级差。** c16 ITL p99 78.6 ms 对 vLLM 7.7 ms，qps16 325.3 ms 对 175.0～356.1 ms。A100 上同一格是 66.9 对 85.5 ms（PegaInfer 更好），说明这条尾随机器与 vLLM 版本变化。
- **overlap 姿态换 TTFT 与尾延迟，代价在步时。** c8 TTFT 135.2 ms（默认 422.7，vLLM 432.2），c16 266.9 ms（默认 611.2，vLLM 723.9），qps16 ITL p99 44.0 ms（默认 325.3）。代价是 c16 步时 9.68 ms（默认 8.83，vLLM 6.33）与 c8 ITL p99 33.8 ms（默认 6.3）。A100 上同一姿态的形态相同：c8 TTFT 132.2 ms、c16 ITL p99 31.9 ms、c8 ITL p99 29.6 ms。
- **bs1 的 TTFT 差别来自 vLLM 版本。** A100 上 PegaInfer 63.5 ms 对 vLLM 109.8 ms（1.7 倍），H20 上 81.2 ms 对 77.3 ms（落后 5%）。两个 vLLM 版本在单条请求上的固定开销不同（0.31.0 在 A100 上是 110 ms，0.29.0 在 H20 上是 77 ms），跨机器的 vLLM 基线不能直接对比。
- **两家的 prefill/decode 取舍反过来。** vLLM 0.31.0 在 A100 上 c16 TTFT 402.2 ms、ITL p99 85.5 ms；0.29.0 在 H20 上 c16 TTFT 723.9 ms、ITL p99 7.7 ms。同一格一个用长首字换稳定步时，另一个反过来。比较两机的尾延迟时要把这层差异带上。

## 复现

服务端（每台机器一次，跑完再换下一个姿态）：

```bash
# PegaInfer
./target/release/pegainfer --model-path /path/to/Qwen3.5-4B --served-model-name q35 --port 18099
# PegaInfer overlap
./target/release/pegainfer --model-path /path/to/Qwen3.5-4B --served-model-name q35 --port 18099 \
  --decode-overlap stream --max-batch 32 --qwen35-scheduler-policy auto
# vLLM
python3 -m vllm.entrypoints.cli.main serve /path/to/Qwen3.5-4B --served-model-name q35 --port 18099 \
  --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 64 \
  --no-enable-prefix-caching --disable-log-stats --uvicorn-log-level warning --language-model-only
```

单格压测：

```bash
vllm bench serve --backend openai --base-url http://127.0.0.1:18099 --endpoint /v1/completions \
  --model q35 --tokenizer /path/to/Qwen3.5-4B --dataset-name random --random-range-ratio 0 \
  --input-len 1024 --output-len 256 --num-prompts 64 --max-concurrency 16 \
  --temperature 0 --ignore-eos --seed 42 --disable-tqdm \
  --save-result --result-dir ./json --result-filename c16-r1.json
```

整套矩阵按三个姿态依次各跑一遍，逐格写 JSON 后汇总。原始结果：`qwen35-4b-serving-h20-vllm029-2026-11.json` 与 `qwen35-4b-serving-a100-vllm031-2026-11.json`，每个文件 31 条记录（30 个用例 + 冒烟）。
