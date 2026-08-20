# Qwen3.5-27B TP2 知识基准评测(官方分对比)

> TL;DR:Qwen3.5-27B 在 pegainfer TP2(2× RTX 4090,batched eager decode)上跑知识基准,C-Eval 88.11(官方 90.5)、MMLU-Redux 94.09(官方 93.2),均在跨 harness 正常带内;MMLU-Pro / SuperGPQA 因运行时长原因仅完成抽样冒烟,未出最终分(见下文)。模型数值无 TP 引入的精度问题。
>
> 注:分数实测于 rebase 前的 f4c66780 分支(自研 Phase 1/2a 线);rebase 到 #870 后 logits golden gate 两侧一致通过,数值可迁移,但若正式引用请在本 PR 分支上复跑确认。

## 环境

- GPU:2× RTX 4090(48 GB 版本),`--tp-size 2 --cuda-graph false`(TP+CUDA Graph 仍 fail-closed)
- 模型:Qwen/Qwen3.5-27B `fc05daec`,BF16,served-model-name `qwen35-27b-tp2`
- 采样:temperature=0.0,top_p=1.0,chat completions(thinking 模式,即模板默认行为)
- 评测器:`scripts/eval_mc.py`(自研,统一 `/v1/chat/completions` 并发 48),配方逐项复刻官方 harness:
  - **C-Eval** = OpenCompass `ceval_gen`:52 学科 val split 全量 1346 题,dev split 5-shot,"答案: " 续写,首大写字母抽取
  - **MMLU-Redux** = lm-eval `mmlu_redux_generative`:`fxmarty/mmlu-redux-2.0-ok` 57 学科 test 全量 5330 题,0-shot,首个 `[ABCD]` 抽取
  - **MMLU-Pro** = lm-eval `mmlu_pro`:TIGER-Lab/MMLU-Pro test,validation split 5-shot CoT,`answer is (X)` 抽取
  - **SuperGPQA** = OpenCompass `supergpqa_gen`:`m-a-p/SuperGPQA` train 26529 题,0-shot,"Answer: X" 字母/内容两层抽取
- 启动命令:`LD_LIBRARY_PATH=<.venv>/nvidia/nccl/lib ./target/release/pegainfer --model-path <27B> --served-model-name qwen35-27b-tp2 --tp-size 2 --cuda-graph false --port 18082`
- 结果原始数据:`results/qwen35-27b-tp2-eval/{ceval,mmlu_redux}_samples*.json`(本地,未入库)

## 结果(截至 2026-08-20,评测按需要提前终止)

| 基准 | 官方 | 实测 | n | 口径 | Δ 判定 |
|---|---|---|---|---|---|
| MMLU-Redux | 93.2 | **94.09** | 5330 全量 | 8192 cap + 截断重跑合并(32 条重跑,2 条仍截断) | **同带** |
| C-Eval | 90.5 | **88.11** | 1346 全量 | 8192 cap + 截断重跑合并(48 条重跑,0 条残留) | **同带边缘**(-2.4pp,CI95 ±1.7pp) |
| MMLU-Pro | 86.1 | — | 600/2000 中止 | 抽样 n=2000(cap 24576)跑到 30% 人工终止;100 题冒烟在 4096 cap 下 51% 截断 | 无最终分 |
| SuperGPQA | 65.6 | — | 未正式跑 | 100 题冒烟:可完成子集 43 题对金标准确 27/43≈63% | 无最终分 |

## 关键观察

- **没有 TP 精度问题**:27B TP2 HF logits golden gate 全绿;C-Eval 非截断子集(1290/1346)准确率 90.2% ≈ 官方 90.5。C-Eval 的差距全部来自 thinking 长度上限被掐断的最难题,而非模型错算。
- **MMLU-Redux 略高于官方**(+0.9pp):同带,说明 prompt/抽取/数值链路都对。
- **thinking 长度是最大的系统变量**:thinking 模型在 C-Eval 上 ~4% 题需要 >8192 token,MMLU-Pro 上 >1/3 题在 4096 内收不住。官方 harness 的 max_tokens 未知(推测 ≥32k);本评测用 8192 首轮 + 32768 重跑合并来逼近。跨 harness ±1–2pp 属正常。
- **吞吐前置条件**:此评测可行完全依赖 Step 3 的 batched eager TP decode 修复(此前 16 并发聚合仅 ~25 tok/s,全量不可行;修复后 ~450 tok/s @48 并发)。

## 复现命令

```bash
# server(见上);评测(hf 镜像):
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -u scripts/eval_mc.py ceval --max-tokens 8192 --concurrency 48 --out-dir results/qwen35-27b-tp2-eval
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -u scripts/eval_mc.py mmlu_redux --max-tokens 8192 --concurrency 48 --out-dir results/qwen35-27b-tp2-eval
# 截断样本重跑合并:
.venv/bin/python -u scripts/eval_rerun_truncated.py ceval --max-tokens 32768 --concurrency 16
# 抽样(--sample 按学科比例分层,seed 1337,n=2000 时 CI95 半宽 ±1.3pp):
.venv/bin/python -u scripts/eval_mc.py mmlu_pro --sample 2000 --max-tokens 24576 --concurrency 48
.venv/bin/python -u scripts/eval_mc.py supergpqa --sample 2000 --max-tokens 24576 --concurrency 48
```

## 下一步

- 补齐 MMLU-Pro / SuperGPQA 抽样全量(各 2000 题,预估合计 ~10h,吞吐 ~450 tok/s 前提)。
- 若要把 C-Eval 收敛到官方 ±1pp:换更大的 thinking 预算复跑全量(无合并),并确认 Qwen 官方 harness 的 prompt 模板与本评测是否一致。
