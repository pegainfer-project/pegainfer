"""
GSM8K evaluation for thinking models (Qwen3.5) via pegainfer /v1/completions API.

lm-eval's local-completions backend applies stop sequences during generation,
which causes "Question:" inside <think> blocks to prematurely truncate output.
This script works around that by:
  1. Not using "Question:" as a stop sequence
  2. Using only <|im_end|> as stop
  3. Stripping <think>...</think> blocks client-side before answer extraction
  4. Applying gsm8k's standard extraction regexes

Usage:
  python scripts/eval_gsm8k_thinking.py \
    --base-url http://localhost:8000/v1/completions \
    --model Qwen3.5-4B \
    --tokenizer models/Qwen3.5-4B \
    --num-fewshot 8 \
    --limit 0 \
    --seed 1234
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def build_fewshot_prefix(train_split, num_fewshot: int, seed: int) -> str:
    """Build few-shot prefix from training examples, matching lm-eval's format."""
    import random

    rng = random.Random(seed)
    indices = rng.sample(range(len(train_split)), num_fewshot)
    parts = []
    for idx in indices:
        ex = train_split[idx]
        parts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
    return "\n\n".join(parts)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks. Handle unclosed blocks too."""
    # Closed think blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Unclosed think block (model ran out of tokens mid-think)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def extract_answer_strict(text: str) -> str | None:
    """gsm8k strict-match: extract number after ####"""
    m = re.search(r"####\s*(-?[0-9.,]+)", text)
    return m.group(1).replace(",", "") if m else None


def extract_answer_flexible(text: str) -> str | None:
    """gsm8k flexible-extract: take the last number in the text."""
    nums = re.findall(r"(-?\$?[0-9.,]{2,})|(-?[0-9]+)", text)
    flat = [n for g in nums for n in g if n]
    if not flat:
        return None
    raw = flat[-1].replace(",", "").replace("$", "").rstrip(".")
    return raw


def extract_gold(answer_text: str) -> str:
    """Extract gold answer from gsm8k answer field (after ####)."""
    m = re.search(r"####\s*(-?[0-9.,]+)", answer_text)
    if m:
        return m.group(1).replace(",", "")
    raise ValueError(f"Cannot extract gold answer from: {answer_text}")


def normalize(s: str) -> str:
    """Normalize for comparison: strip whitespace, leading zeros, trailing dots."""
    s = s.strip().lstrip("0") or "0"
    s = s.rstrip(".")
    return s


def main():
    parser = argparse.ArgumentParser(description="GSM8K eval for thinking models")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer path (for reference only)")
    parser.add_argument("--num-fewshot", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0, help="0 = all samples")
    parser.add_argument("--seed", type=int, default=1234, help="per-request seed; -1 omits it (engine RNG)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature (0 = greedy; >0 for sampled-score parity runs)",
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"]
    test = ds["test"]

    prefix = build_fewshot_prefix(train, args.num_fewshot, args.seed)

    samples = list(test)
    if args.limit > 0:
        samples = samples[: args.limit]

    strict_correct = 0
    flex_correct = 0
    total = len(samples)
    results = []

    for ex in tqdm(samples, desc="Evaluating"):
        prompt = prefix + "\n\n" + f"Question: {ex['question']}\nAnswer:"
        payload = {
            "prompt": prompt,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stop": ["<|im_end|>"],
        }
        # seed < 0 omits the per-request seed: sampled-parity runs want the
        # engine RNG (three independent runs per arm), not a fixed replay.
        if args.seed >= 0:
            payload["seed"] = args.seed
        resp = requests.post(args.base_url, json=payload)
        resp.raise_for_status()
        result = resp.json()
        raw_text = result["choices"][0]["text"]
        finish = result["choices"][0]["finish_reason"]

        cleaned = strip_think(raw_text)
        gold = extract_gold(ex["answer"])

        strict = extract_answer_strict(cleaned)
        flex = extract_answer_flexible(cleaned)

        strict_match = strict is not None and normalize(strict) == normalize(gold)
        flex_match = flex is not None and normalize(flex) == normalize(gold)

        if strict_match:
            strict_correct += 1
        if flex_match:
            flex_correct += 1

        results.append({
            "question": ex["question"],
            "gold": gold,
            "raw_len": len(raw_text),
            "finish_reason": finish,
            "cleaned": cleaned[:200],
            "strict_pred": strict,
            "flex_pred": flex,
            "strict_match": strict_match,
            "flex_match": flex_match,
        })

    strict_acc = strict_correct / total
    flex_acc = flex_correct / total

    summary = {
        "model": args.model,
        "task": "gsm8k",
        "num_fewshot": args.num_fewshot,
        "total": total,
        "strict_match": strict_acc,
        "flexible_extract": flex_acc,
        "strict_correct": strict_correct,
        "flex_correct": flex_correct,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }

    print(f"\n{'='*50}")
    print(f"Model: {args.model}")
    print(f"GSM8K ({args.num_fewshot}-shot), n={total}")
    print(f"  strict-match:     {strict_acc:.4f} ({strict_correct}/{total})")
    print(f"  flexible-extract: {flex_acc:.4f} ({flex_correct}/{total})")
    print(f"{'='*50}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "samples": results}, f, indent=2, ensure_ascii=False)
        print(f"Results written to {out_path}")

    return 0 if flex_acc > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
