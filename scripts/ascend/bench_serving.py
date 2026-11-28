#!/usr/bin/env python3
"""Manual Ascend serving benchmark: TTFT, TPOT and tokens/s.

Starts the same release binary as `check_http.py`, sends streaming greedy
completions, and reports per-request and aggregate latency. This is a
correctness-adjacent baseline only; run it on the same pod session as the
change you compare against.
"""
import argparse
import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=18321)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--binary", default=None)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    binary = Path(args.binary) if args.binary else (
        Path(os.environ.get("CARGO_TARGET_DIR", "target")) / "release/pegainfer")
    command = [str(binary), "--model-path", args.model_path,
               "--served-model-name", "ascend-qwen3", "--port", str(args.port),
               "--cuda-graph=false", "--no-prefix-cache", "--ascend-slots", "4",
               "--ascend-max-context", str(args.context), "--ascend-chunk-size", "128"]
    base = f"http://127.0.0.1:{args.port}"
    payload = {
        "model": "ascend-qwen3",
        "prompt": "The capital of France is",
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    def health():
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def one(index):
        conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=600)
        try:
            start = time.monotonic()
            conn.request("POST", "/v1/completions", json.dumps(payload),
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            assert response.status == 200, response.read()
            first = None
            last = None
            usage = None
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                text = raw[6:].strip()
                if text == b"[DONE]":
                    break
                event = json.loads(text)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("text"):
                        now = time.monotonic()
                        first = first or now
                        last = now
            assert first is not None and last is not None and usage, "no streamed token/usage"
            tokens = usage["completion_tokens"]
            ttft = first - start
            tpot = (last - first) / (tokens - 1) if tokens > 1 else 0.0
            return {
                "index": index,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": tokens,
                "ttft_ms": ttft * 1000,
                "tpot_ms": tpot * 1000,
                "tok_per_s": tokens / (last - start) if last > start else 0.0,
            }
        finally:
            conn.close()

    with (output / "server.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                assert proc.poll() is None, f"server exited: {proc.returncode}"
                if health():
                    break
                time.sleep(0.2)
            else:
                raise TimeoutError("server not ready")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                results = list(pool.map(one, range(args.requests)))
            report = {
                "command": command,
                "requests": args.requests,
                "concurrency": args.concurrency,
                "max_tokens": args.max_tokens,
                "results": results,
                "ttft_ms_mean": sum(r["ttft_ms"] for r in results) / len(results),
                "tpot_ms_mean": sum(r["tpot_ms"] for r in results) / len(results),
                "tok_per_s_mean": sum(r["tok_per_s"] for r in results) / len(results),
            }
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            report["server_exit_code"] = proc.returncode
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
