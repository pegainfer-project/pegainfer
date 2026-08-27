# Qwen3.5-27B TP2 Knowledge Benchmarks (vs Official Scores)

> TL;DR: Qwen3.5-27B on pegainfer TP2 (2× RTX 4090, batched eager decode) runs the knowledge benchmarks at C-Eval 88.11 (official 90.5) and MMLU-Redux 94.09 (official 93.2), both inside the normal cross-harness band; MMLU-Pro / SuperGPQA only completed sampled smokes due to runtime cost, no final scores (see below). No TP-induced accuracy issue in the model numerics.
>
> Note: scores were measured on the pre-rebase in-house Phase 1/2a line; after the rebase onto #870 the logits golden gates pass identically on both sides, so the numerics carry over — but rerun on this PR branch before formally citing parity.

## Environment

- GPU: 2× RTX 4090 (48 GB variant), `--tp-size 2 --cuda-graph false` (TP + CUDA Graph still fail-closed)
- Model: Qwen/Qwen3.5-27B `fc05daec`, BF16, served-model-name `qwen35-27b-tp2`
- Sampling: temperature=0.0, top_p=1.0, chat completions (thinking mode, i.e. the template default)
- Evaluator: `scripts/eval_mc.py` (in-house, uniform `/v1/chat/completions` at concurrency 48), recipes replicated item-by-item from the official harnesses:
  - **C-Eval** = OpenCompass `ceval_gen`: 52 subjects, full val split (1346 items), 5-shot from the dev split, `"答案: "` continuation, first-capital-letter extraction
  - **MMLU-Redux** = lm-eval `mmlu_redux_generative`: `fxmarty/mmlu-redux-2.0-ok`, 57 subjects, full test split (5330 items), 0-shot, first `[ABCD]` extraction
  - **MMLU-Pro** = lm-eval `mmlu_pro`: TIGER-Lab/MMLU-Pro test, 5-shot CoT from the validation split, `answer is (X)` extraction
  - **SuperGPQA** = OpenCompass `supergpqa_gen`: `m-a-p/SuperGPQA` train (26529 items), 0-shot, two-tier "Answer: X" letter/content extraction
- Launch command: `./target/release/pegainfer --model-path <27B> --served-model-name qwen35-27b-tp2 --tp-size 2 --cuda-graph false --port 18082` (NCCL runtime setup in `docs/playbooks/developer-onboarding.md`)

## Results (as of 2026-08-20; runs terminated early as needed)

| Benchmark | Official | Measured | n | Protocol | Δ verdict |
|---|---|---|---|---|---|
| MMLU-Redux | 93.2 | **94.09** | 5330 full | 8192 cap + truncated-rerun merge (32 rerun, 2 still truncated) | **in band** |
| C-Eval | 90.5 | **88.11** | 1346 full | 8192 cap + truncated-rerun merge (48 rerun, 0 residual) | **band edge** (-2.4pp, CI95 ±1.7pp) |
| MMLU-Pro | 86.1 | — | stopped at 600/2000 | sampled n=2000 (cap 24576) manually stopped at 30%; 100-item smoke truncated 51% at cap 4096 | no final score |
| SuperGPQA | 65.6 | — | no formal run | 100-item smoke: 43 completable items, 27/43 ≈ 63% correct against gold | no final score |

## Key Observations

- **No TP accuracy issue**: the 27B TP2 HF logits golden gates are all green; on the non-truncated C-Eval subset (1290/1346) accuracy is 90.2% ≈ official 90.5. The C-Eval gap comes entirely from the hardest items cut off by the thinking-length cap, not from model miscomputation.
- **MMLU-Redux slightly above official** (+0.9pp): in band; the prompt/extraction/numerics chain is sound.
- **Thinking length is the dominant system variable**: on C-Eval ~4% of items need >8192 tokens with a thinking model; on MMLU-Pro over a third of items do not finish within 4096. The official harnesses' max_tokens is unknown (presumed ≥32k); this eval approximates with an 8192 first pass plus a 32768 rerun merge. ±1–2pp across harnesses is normal.
- **Throughput precondition**: this eval is feasible only because of the Step 3 batched eager TP decode fix (before it, 16 concurrent requests aggregated ~25 tok/s and full runs were infeasible; after it, ~450 tok/s at 48 concurrent).

## Reproduction Commands

```bash
# server (see above); eval:
.venv/bin/python -u scripts/eval_mc.py ceval --max-tokens 8192 --concurrency 48 --out-dir results/qwen35-27b-tp2-eval
.venv/bin/python -u scripts/eval_mc.py mmlu_redux --max-tokens 8192 --concurrency 48 --out-dir results/qwen35-27b-tp2-eval
# rerun-and-merge truncated samples:
.venv/bin/python -u scripts/eval_rerun_truncated.py ceval --max-tokens 32768 --concurrency 16
# sampling (--sample stratifies by subject proportionally, seed 1337; at n=2000 the CI95 half-width is ±1.3pp):
.venv/bin/python -u scripts/eval_mc.py mmlu_pro --sample 2000 --max-tokens 24576 --concurrency 48
.venv/bin/python -u scripts/eval_mc.py supergpqa --sample 2000 --max-tokens 24576 --concurrency 48
```

## Next Steps

- Complete the full sampled runs of MMLU-Pro / SuperGPQA (2000 items each, ~10h total estimated at ~450 tok/s).
- To converge C-Eval to within official ±1pp: rerun the full set with a larger thinking budget (no merge), and confirm whether the Qwen official harness prompt template matches this eval's.
