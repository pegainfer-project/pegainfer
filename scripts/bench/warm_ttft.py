#!/usr/bin/env python3
"""Warm (prefix-cache-hit) TTFT sweep over input lengths.

For each input length L: build one fixed random-token prompt (seeded), send it
once cold to populate the GPU prefix cache, then re-send the identical prompt N
times and measure TTFT for each warm sample. Output is 1 token, so warm TTFT
isolates cache lookup + suffix prefill + first-token sampling.

Every sample fully drains the SSE stream to [DONE] before the next request —
an early disconnect can leave the previous request decoding and inflate the
next sample's TTFT.

Usage:
  python scripts/bench/warm_ttft.py \
      --base-url http://localhost:8000 --model /data/Qwen3-4B \
      --tokenizer /data/Qwen3-4B --lengths 256,512,1024,2048,4096,8192,16384 \
      --samples 20 --seed 42 --output results.json
"""

import argparse
import json
import time

import numpy as np
import requests
from transformers import AutoTokenizer


def build_prompt(tokenizer, length: int, rng: np.random.Generator) -> tuple[str, int]:
    """Build a prompt that re-encodes to exactly `length` tokens (best effort)."""
    vocab_size = tokenizer.vocab_size
    ids = rng.integers(0, vocab_size, size=length * 2).tolist()
    text = tokenizer.decode(ids, skip_special_tokens=True)
    for _ in range(8):
        reencoded = tokenizer.encode(text, add_special_tokens=False)
        if len(reencoded) == length:
            break
        text = tokenizer.decode(reencoded[:length], skip_special_tokens=True)
    actual = len(tokenizer.encode(text, add_special_tokens=False))
    return text, actual


def measure_ttft(session: requests.Session, base_url: str, model: str, prompt: str) -> float:
    """Send one streaming completion (max_tokens=1), return TTFT in seconds."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": True,
    }
    start = time.perf_counter()
    ttft = None
    with session.post(
        f"{base_url}/v1/completions", json=payload, stream=True, timeout=600
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            if ttft is None and line.startswith(b"data:") and b"[DONE]" not in line:
                ttft = time.perf_counter() - start
            # keep draining to [DONE]
    if ttft is None:
        raise RuntimeError("stream ended without a data chunk")
    return ttft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--lengths", default="256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    lengths = [int(x) for x in args.lengths.split(",")]
    session = requests.Session()

    results = []
    for length in lengths:
        # Fresh RNG per length so the prompt set is independent of sweep order.
        rng = np.random.default_rng(args.seed + length)
        prompt, actual = build_prompt(tokenizer, length, rng)
        cold = measure_ttft(session, args.base_url, args.model, prompt)
        warm = [
            measure_ttft(session, args.base_url, args.model, prompt)
            for _ in range(args.samples)
        ]
        warm_ms = sorted(w * 1000 for w in warm)
        entry = {
            "target_len": length,
            "actual_len": actual,
            "cold_ttft_ms": round(cold * 1000, 2),
            "warm_ttft_ms": {
                "p50": round(float(np.percentile(warm_ms, 50)), 2),
                "p90": round(float(np.percentile(warm_ms, 90)), 2),
                "p99": round(float(np.percentile(warm_ms, 99)), 2),
                "samples": [round(w, 2) for w in warm_ms],
            },
        }
        results.append(entry)
        print(
            f"len={length} (actual {actual}): cold {entry['cold_ttft_ms']}ms, "
            f"warm p50 {entry['warm_ttft_ms']['p50']}ms p99 {entry['warm_ttft_ms']['p99']}ms",
            flush=True,
        )

    with open(args.output, "w") as f:
        json.dump(
            {
                "base_url": args.base_url,
                "model": args.model,
                "seed": args.seed,
                "samples_per_length": args.samples,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
