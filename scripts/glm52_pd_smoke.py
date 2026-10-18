#!/usr/bin/env python3
"""Small streaming smoke for the GLM5.2 native-MTP P/D router path."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:10001")
    parser.add_argument("--model", default="glm-5.2-fp8")
    parser.add_argument(
        "--prompt",
        default=(
            "Paris is the capital of France. Explain in one short paragraph "
            "why it became politically important."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    payload = json.dumps(
        {
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
    finish_reason: str | None = None
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(json.dumps(event["error"], ensure_ascii=False))
            choices = event.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            text = choice.get("text", "")
            if text and first_token_at is None:
                first_token_at = time.perf_counter()
            pieces.append(text)
            finish_reason = choice.get("finish_reason") or finish_reason

    ended = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("router completed without emitting text")
    print(
        json.dumps(
            {
                "ok": True,
                "ttft_ms": round((first_token_at - started) * 1000, 2),
                "total_ms": round((ended - started) * 1000, 2),
                "finish_reason": finish_reason,
                "text": "".join(pieces),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
