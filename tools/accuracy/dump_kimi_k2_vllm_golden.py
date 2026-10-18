#!/usr/bin/env python3
"""Generate the vLLM golden fixture for the Kimi-K2 accuracy gate.

The gate (`pegainfer-kimi-k2/tests/vllm_golden_gate.rs`) compares pegainfer's
greedy decisions against vLLM *without* running vLLM at test time and *without*
binding to one engine's exact bit pattern. So we precompute, once, on the
serving hardware:

  * a fixed prompt set (raw completion, no chat template, token ids pinned in
    the fixture so the gate never re-tokenizes),
  * vLLM's greedy continuation (`tail`) of D tokens per prompt,
  * vLLM's top-K logprobs at every generated position.

Why vLLM and not HuggingFace: Kimi-K2.6 is INT4 (compressed-tensors). vLLM
executes the same quantized model through marlin kernels — the closest
equal-precision reference available. The HF route decompresses to bf16, a
different numerical regime, and needs a fragile trust_remote_code + stubbed
vision tower load.

The Rust gate replays the same sequences through pegainfer two ways:
  * teacher-forced argmax sweep — prefill `prompt + tail[..i]`, max_tokens=1,
    per position i: pegainfer's pick must sit within a logprob tie tolerance
    of vLLM's own argmax (in vLLM's logprobs — the "regret" check);
  * free-greedy decode parity — generate D tokens and compare against the
    tail, classifying any first divergence as benign tie vs real bug using
    the stored margins.

Output is safetensors, not JSON: machine-only numeric data, nobody reads it,
and the binary layout is ~3.5x smaller (same convention as the Qwen goldens).

Run on a host with 8 GPUs and the vLLM venv (the gate's pegainfer run needs
the same GPUs, so generation and gating are sequential on one box):

    .venv/bin/python tools/accuracy/dump_kimi_k2_vllm_golden.py \
        --model-path /data/models/Kimi-K2.6 \
        --out test_data/kimi-k2.6-vllm-golden.safetensors
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

DECODE_TOKENS = 32
TOP_K = 32
# Per-slot KV arena is 2048 tokens (worker.rs KIMI_DECODE_ROPE_CACHE_TOKENS);
# prompt + tail must fit with headroom. Issue #239: over-long prompts are not
# rejected by the engine, so the fixture must never produce one.
MAX_PROMPT_TOKENS = 1900 - DECODE_TOKENS

_LONG_EN_SEED = (
    "The history of computing is a history of trade-offs between generality and "
    "efficiency. Early machines were programmed by rewiring; stored-program "
    "computers traded raw speed for flexibility, and every layer added since — "
    "assemblers, compilers, operating systems, virtual machines — repeats the "
    "same bargain at a new altitude. "
)

_LONG_ZH_SEED = (
    "长江发源于青藏高原的唐古拉山脉，自西向东流经十一个省级行政区，最终在上海汇入东海。"
    "沿途的地貌从冰川、峡谷到平原、三角洲，几乎涵盖了所有主要的地形类型。"
)

# Raw completion prompts. Mix of language, domain, and length; the two long
# ones are built deterministically from fixed seed paragraphs to push RoPE
# positions toward the slot capacity without external data.
PROMPTS: list[tuple[str, str]] = [
    ("en-factual", "The three primary colors of light are red, green, and"),
    ("zh-factual", "中国四大发明是造纸术、印刷术、指南针和"),
    (
        "code-python",
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    ),
    (
        "code-rust",
        "/// Compute the dot product of two slices.\n"
        "fn dot(a: &[f32], b: &[f32]) -> f32 {\n",
    ),
    ("math", "To solve the equation 3x + 7 = 22, first subtract 7 from both sides:"),
    (
        "en-continuation",
        "The expedition reached the base camp at dawn. The mountain rose above "
        "them, its summit hidden in cloud. After three days of waiting for the "
        "weather to clear, the team leader made a decision:",
    ),
    (
        "zh-continuation",
        "深夜的实验室里只剩下一盏台灯还亮着。她盯着屏幕上的曲线看了很久，终于明白了问题出在哪里：",
    ),
    ("list", "Top five most spoken languages in the world:\n1."),
    (
        "json",
        '{"name": "pegainfer", "language": "Rust", "purpose":',
    ),
    (
        "translation",
        "Translate to French: 'The library opens at nine in the morning.'\nFrench:",
    ),
    ("long-en", _LONG_EN_SEED * 24 + "The lesson for modern system designers is"),
    ("long-zh", _LONG_ZH_SEED * 18 + "这条河流对中国经济最重要的意义在于"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tp", type=int, default=8)
    args = parser.parse_args()

    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=4096,
        max_logprobs=TOP_K,
    )
    tokenizer = llm.get_tokenizer()

    prompt_token_ids = []
    for name, text in PROMPTS:
        ids = tokenizer.encode(text)
        assert len(ids) <= MAX_PROMPT_TOKENS, (
            f"{name}: {len(ids)} prompt tokens > {MAX_PROMPT_TOKENS} "
            f"(per-slot KV arena is 2048 incl. the decode tail)"
        )
        prompt_token_ids.append(ids)

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=DECODE_TOKENS,
        logprobs=TOP_K,
        ignore_eos=True,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompt_token_ids],
        sampling_params=sp,
    )

    tails = []  # [S, D]
    ids_all, lp_all = [], []  # [S, D, K]
    margins = []
    for (name, _text), out in zip(PROMPTS, outputs):
        gen = out.outputs[0]
        assert len(gen.token_ids) == DECODE_TOKENS, (
            f"{name}: vLLM returned {len(gen.token_ids)} tokens, "
            f"expected {DECODE_TOKENS}"
        )
        topk_ids, topk_lps = [], []
        for pos, (sampled, lp_dict) in enumerate(zip(gen.token_ids, gen.logprobs)):
            # vLLM returns top-K plus (if outside it) the sampled token; greedy
            # sampling means the sampled token IS the argmax, so sorting by
            # logprob and truncating to K keeps it at rank 0.
            ranked = sorted(lp_dict.items(), key=lambda kv: kv[1].logprob, reverse=True)
            ranked = ranked[:TOP_K]
            assert ranked[0][0] == sampled, (
                f"{name} pos {pos}: greedy sample {sampled} is not the "
                f"top-logprob token {ranked[0][0]}"
            )
            topk_ids.append([tok for tok, _ in ranked])
            topk_lps.append([lp.logprob for _, lp in ranked])
            margins.append(topk_lps[-1][0] - topk_lps[-1][1])
        tails.append(list(gen.token_ids))
        ids_all.append(topk_ids)
        lp_all.append(topk_lps)

    margins.sort()
    pct = lambda q: margins[min(int(len(margins) * q), len(margins) - 1)]
    print(
        f"top1-top2 margin over {len(margins)} positions: "
        f"p1 {pct(0.01):.4f} p10 {pct(0.10):.4f} p50 {pct(0.50):.4f} "
        f"min {margins[0]:.4f}"
    )

    prompt_flat = [t for ids in prompt_token_ids for t in ids]
    tensors = {
        "prompt_tokens": np.asarray(prompt_flat, dtype=np.int32),
        "prompt_lens": np.asarray([len(ids) for ids in prompt_token_ids], dtype=np.int32),
        "tail_tokens": np.asarray(tails, dtype=np.int32),  # [S, D]
        "topk_ids": np.asarray(ids_all, dtype=np.int32),  # [S, D, K]
        "topk_logprobs": np.asarray(lp_all, dtype=np.float32),  # [S, D, K]
    }
    meta = {
        "reference": "vllm",
        "vllm_version": vllm.__version__,
        "model": Path(args.model_path).name,
        "tensor_parallel_size": str(args.tp),
        "decode_tokens": str(DECODE_TOKENS),
        "top_k": str(TOP_K),
        "seq_names": ",".join(name for name, _ in PROMPTS),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator": "tools/accuracy/dump_kimi_k2_vllm_golden.py",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=meta)
    n_pos = len(PROMPTS) * DECODE_TOKENS
    print(f"wrote {out_path} ({len(PROMPTS)} seqs, {n_pos} positions, top-{TOP_K})")


if __name__ == "__main__":
    main()
