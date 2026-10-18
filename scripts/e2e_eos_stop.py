#!/usr/bin/env python3
"""E2E check that a serving pegainfer engine stops at EOS (issue #238).

Sends the same prompt twice through `/v1/completions` on an already-running
server:

1. default: the answer must end early with `finish_reason: "stop"` and fewer
   than `max_tokens` completion tokens;
2. `ignore_eos: true`: generation must run to `finish_reason: "length"` with
   exactly `max_tokens` completion tokens (benchmarks rely on this).

The default prompt is in Kimi-K2 chat format; pass --prompt for other models.

Usage:
    python3 scripts/e2e_eos_stop.py --base-url http://127.0.0.1:8124 --model kimi-k2.5
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

KIMI_PROMPT = (
    "<|im_system|>system<|im_middle|>You are a helpful assistant.<|im_end|>"
    "<|im_user|>user<|im_middle|>What is 2+2? Reply with just the number.<|im_end|>"
    "<|im_assistant|>assistant<|im_middle|>"
)


def request_completion(
    base_url: str, model: str, prompt: str, max_tokens: int, ignore_eos: bool, timeout: float
) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": ignore_eos,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    choice = payload["choices"][0]
    return {
        "finish_reason": choice["finish_reason"],
        "completion_tokens": payload["usage"]["completion_tokens"],
        "text": choice["text"],
    }


def check(label: str, condition: bool, detail: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return condition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8124")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=KIMI_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    stop_run = request_completion(
        args.base_url, args.model, args.prompt, args.max_tokens, ignore_eos=False, timeout=args.timeout
    )
    print(f"stop run: {stop_run['finish_reason']}, {stop_run['completion_tokens']} tokens")
    print(f"  text: {stop_run['text']!r}")

    length_run = request_completion(
        args.base_url, args.model, args.prompt, args.max_tokens, ignore_eos=True, timeout=args.timeout
    )
    print(f"ignore_eos run: {length_run['finish_reason']}, {length_run['completion_tokens']} tokens")

    ok = True
    ok &= check(
        "EOS stops generation",
        stop_run["finish_reason"] == "stop",
        f"finish_reason={stop_run['finish_reason']}",
    )
    ok &= check(
        "stop run ends early",
        stop_run["completion_tokens"] < args.max_tokens,
        f"{stop_run['completion_tokens']} < {args.max_tokens}",
    )
    ok &= check(
        "ignore_eos runs to length",
        length_run["finish_reason"] == "length",
        f"finish_reason={length_run['finish_reason']}",
    )
    ok &= check(
        "ignore_eos generates max_tokens",
        length_run["completion_tokens"] == args.max_tokens,
        f"{length_run['completion_tokens']} == {args.max_tokens}",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
