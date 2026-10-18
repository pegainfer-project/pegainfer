# Accuracy Parity Playbook

> **TL;DR:** 精度对齐先定义真值，再找 first diff，再只拆那一段。对 bf16 模型，很多“只差一点”的 bug 不是算子全错，而是少了一次 materialization round、走错了 truth path、或者被 argmax tie 放大。
>
> **Status:** Active reference. Updated 2026-03-27.

## What Counts As "Correct"

- 真值是外部实现，不是 pegainfer 自己生成的 JSON。
- 对生成步，真值必须是 HF 的真实 incremental `past_key_values` 路径。
- 不要把“重建完整前缀后的 full-prefill”当生成步真值。它在后段层上可能连 HF 自己都不等于 HF incremental。

## Accuracy Ladder

按这个顺序收敛，别一开始就盯整段文本：

1. 先比 greedy token trace，找到 first diff step。
2. 固定 exact token ids，只比较这个共同前缀。
3. 先比 final logits。
4. 再比目标层的 coarse checkpoints。
5. 只有 coarse checkpoint 真开始漂了，才继续拆内部算子。

这样做的目的很简单：不要在已经分叉的文本上做层级比较，那样定位几乎总是错的。

## Verified Commands

这些命令已经在 Qwen3.5-4B 精度排查里跑过，可以直接复用。

先找 first diff：

```bash
./target/release/qwen35_trace_greedy \
  --model-path models/Qwen3.5-4B \
  --prompt 'Tell me a story' \
  --steps 25 \
  --topk 5 \
  --out target/accuracy/tell_story_trace_after_convfix.json

./.venv/bin/python tools/accuracy/hf_dump_qwen35_greedy_tokens.py \
  --model-path models/Qwen3.5-4B \
  --prompt 'Tell me a story' \
  --steps 25 \
  --out target/accuracy/hf_tell_story_trace_after_convfix.json
```

固定 exact token ids，比某一层：

```bash
./target/release/qwen35_dump_layer_ids \
  --model-path models/Qwen3.5-4B \
  --token-ids-json target/accuracy/python_prime_prefix_step1_ids.json \
  --layer 0 \
  --out target/accuracy/peg_python_prime_layer0_prefill_after_convfix.json

./.venv/bin/python tools/accuracy/hf_dump_qwen35_layer_ids.py \
  --model-path models/Qwen3.5-4B \
  --token-ids-json target/accuracy/python_prime_prefix_step1_ids.json \
  --layer 0 \
  --out target/accuracy/hf_python_prime_layer0_prefill.json

./.venv/bin/python tools/accuracy/compare_qwen35_dump.py \
  target/accuracy/hf_python_prime_layer0_prefill.json \
  target/accuracy/peg_python_prime_layer0_prefill_after_convfix.json
```

固定 exact token ids，比最终 incremental logits：

```bash
./target/release/qwen35_dump_incremental_final_ids \
  --model-path models/Qwen3.5-4B \
  --prompt-token-ids-json target/accuracy/chinese_prompt_ids_after_convfix.json \
  --decode-token-ids-json target/accuracy/chinese_decode_common_ids_after_convfix.json \
  --out target/accuracy/peg_chinese_incremental_final_after_convfix.json

./.venv/bin/python tools/accuracy/hf_dump_qwen35_incremental_final_ids.py \
  --model-path models/Qwen3.5-4B \
  --prompt-token-ids-json target/accuracy/chinese_prompt_ids_after_convfix.json \
  --decode-token-ids-json target/accuracy/chinese_decode_common_ids_after_convfix.json \
  --out target/accuracy/hf_chinese_incremental_final_after_convfix.json
```

检查线性注意力 `seq_len=1` 的 production path 和 recurrent path 是否等价：

```bash
./target/release/qwen35_compare_linear_seq1_ids \
  --model-path models/Qwen3.5-4B \
  --prompt-token-ids-json target/accuracy/python_prime_prompt_ids.json \
  --decode-token-ids-json target/accuracy/python_prime_decode_ids_step1.json \
  --layer 0
```

历史 case-level 覆盖率排查命令如下。`test_data/Qwen3.5-4B.json`
已经退役，新排查应优先走 `pegainfer-qwen35/tests/hf_golden_gate.rs`
或重建同等的 HF `past_key_values` token-id/logits dump：

```bash
./target/release/qwen35_generate_cases \
  --model-path models/Qwen3.5-4B \
  --cases test_data/Qwen3.5-4B.json \
  --out target/accuracy/qwen35_cases_after_silu_fix.json

./.venv/bin/python tools/accuracy/hf_generate_qwen35_cases.py \
  --model-path models/Qwen3.5-4B \
  --cases test_data/Qwen3.5-4B.json \
  --out target/accuracy/hf_cases_after_convfix.json
```

如果怀疑是某个小 kernel 的 bf16 语义错了，先跑聚焦的算子测试：

```bash
cargo test --release test_conv1d_prefill_handoff_matches_single_prefill -- --nocapture
```

## High-Signal Tricks

### 1. Use Exact Token IDs, Not Round-Tripped Text

- 用 token ids 固定前缀，避免换行、空格、中文分词这些低价值噪音。
- first diff 一旦出来，后面所有 layer/logit dump 都应该基于这个 exact prefix。

### 2. Match The Real Runtime Path

- debug tooling 必须走真实生产路径。
- 如果 production decode 实际上是 `prefill_forward(&[token])`，那 dump 也必须这么跑。
- 手工调用已经退役的 decode kernel，会把你带到错误结论。

### 3. Always Ask Whether HF Materialized A BF16 Tensor Here

很多小 drift 不是算法错，是少了一次 bf16 round。

这次已经验证过的两个坑：

- `conv1d`：
  HF/PyTorch 是 conv 输出先 materialize 成 bf16，再做 `SiLU`。
- `silu_mul`：
  HF/PyTorch bf16 语义等价于 `silu(gate)` 先 round 到 bf16，再和 `up` 相乘。

排查方法：

- 不要先改 kernel。
- 先在 Python 里用真实输入和真实权重离线复算。
- 如果“补上一层 bf16 round”能把某个 checkpoint 从 `0.03125` 直接打到 `0.0`，那就基本坐实了。

### 4. Separate State Bugs From Numeric Drift

- 先做 peg internal consistency：
  - incremental decode vs fresh full-prefill
  - `seq_len=1` prefill vs decode reference
  - handoff state vs single-shot prefill state
- 如果 peg 自己都不一致，先修 state bug。
- 如果 peg 自己一致，但 peg vs HF 还差，才值得继续查 HF 语义差。

### 5. Treat Argmax Ties As A Distinct Class

剩余 case 如果 top-1 和 top-2 只差 `0.125`，它和“大算子错了”不是一类问题。

判断标准：

- 看 first diff step 的 top logits，而不是只看输出文本。
- 如果前几名 token 都在 `0.125` 档位内，先归类为 tie-sensitive。
- tie-sensitive case 适合继续压小数值差，不适合凭文本做过度归因。

### 6. Reject Bad Leads Fast

- 如果一个改动让 layer-level HF diff 变差，就立刻回退。
- “看起来像 HF 语义”不够，必须让真实 checkpoint 变好才算数。

## Common Failure Modes

- 把 HF full-prefill 当生成步真值。
- 在文本已经分叉以后继续做 layer 对比。
- 用不走 production path 的 debug kernel 下结论。
- 看到 top-1 不同，就误判成结构性 bug。
- 只看 `max_abs`，不看 first-diff step 和 top logits。

## How To Record Findings

- 只写验证过的结论。
- 明确写出：
  - 用的 prompt 或 token ids
  - 比的 checkpoint
  - `max_abs` / `mean_abs`
  - 这说明了什么，不说明什么
- 被证伪的线索也要记下来，但要明确标记为 rejected lead。

## Current Next Use

- 这个 playbook 已经把 Qwen3.5-4B 的 HF exact parity 从 `2/13` 推到 `11/13`。
- 下一次再做精度问题，先照这个顺序走，不要从整段输出直接猜 kernel。
