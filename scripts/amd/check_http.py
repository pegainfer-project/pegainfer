#!/usr/bin/env python3
"""Real ROCm server lifecycle gate; launch an isolated server and retain evidence."""
import argparse
import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=18301)
    parser.add_argument("--context", type=int, default=512)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    binary = Path(os.environ["CARGO_TARGET_DIR"]) / "release/pegainfer"
    command = [str(binary), "--model-path", args.model_path,
               "--served-model-name", "rocm-qwen3", "--port", str(args.port),
               "--cuda-graph=false", "--no-prefix-cache", "--rocm-slots", "4",
               "--rocm-max-context", str(args.context), "--rocm-chunk-size", "128"]
    base = f"http://127.0.0.1:{args.port}"
    report = {"command": command, "checks": [], "passed": False}
    handles = []

    def record(name, **fields):
        report["checks"].append({"name": name, "passed": True, **fields})
        print(name, json.dumps(fields, ensure_ascii=False), flush=True)

    def request(path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(base + path, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def payload(**extra):
        return {"model": "rocm-qwen3", "prompt": "The capital of France is",
                "temperature": 0, "max_tokens": 24, "ignore_eos": True, **extra}

    def complete(**extra):
        start = time.monotonic()
        status, raw = request("/v1/completions", payload(**extra))
        assert status == 200, (status, raw)
        result = json.loads(raw)
        assert result["choices"][0]["text"], result
        return result, time.monotonic() - start

    def metrics():
        status, body = request("/metrics")
        assert status == 200, body
        values = {}
        for name in ["num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"]:
            found = re.findall(r"^vllm:" + name + r"(?:\{[^\n]*\})? ([\d.eE+\-]+)$", body, re.M)
            assert found, (name, body[:3000])
            values[name] = sum(map(float, found))
        return values

    def wait_for(predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = metrics()
            if predicate(state):
                return state
            time.sleep(0.05)
        raise AssertionError(f"scheduler state did not converge: {state}")

    def stream(index):
        data = payload(prompt=f"Explain the number {index + 10} in a short sentence. " + "A useful example. " * (index % 4 * 25),
                       stream=True, stream_options={"include_usage": True})
        conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=120)
        try:
            start = time.monotonic()
            conn.request("POST", "/v1/completions", json.dumps(data),
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            assert response.status == 200, response.read()
            chunks, terminal, usage, first, done = [], None, None, None, False
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                text = raw[6:].strip()
                if text == b"[DONE]":
                    done = True
                    break
                event = json.loads(text)
                assert "error" not in event, event
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("text"):
                        first = first or time.monotonic()
                        chunks.append(choice["text"])
                    terminal = choice.get("finish_reason") or terminal
            assert done and terminal == "length" and chunks, (done, terminal, chunks)
            assert usage and usage["completion_tokens"] == 24, usage
            return {"index": index, "text": "".join(chunks), "usage": usage,
                    "ttft_ms": (first - start) * 1000,
                    "elapsed_ms": (time.monotonic() - start) * 1000}
        finally:
            conn.close()

    with (output / "server.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                assert proc.poll() is None, f"server exited: {proc.returncode}"
                try:
                    if request("/health")[0] == 200:
                        break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("server not ready")
            status, models = request("/v1/models")
            assert status == 200 and json.loads(models)["data"][0]["id"] == "rocm-qwen3"
            record("health_and_models")
            result, elapsed = complete()
            assert result["usage"]["completion_tokens"] == 24
            assert result["choices"][0]["finish_reason"] == "length"
            record("completion", response=result, elapsed_s=elapsed)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(stream, range(16)))
            record("streaming_concurrency8_slots4", responses=results)
            stop = result["choices"][0]["text"][:8]
            status, raw = request("/v1/completions", payload(stop=[stop]))
            stopped = json.loads(raw)
            assert status == 200 and stopped["choices"][0]["finish_reason"] == "stop", stopped
            assert stop not in stopped["choices"][0]["text"]
            record("stop_string", response=stopped)
            status, raw = request("/v1/chat/completions", {
                "model": "rocm-qwen3", "messages": [{"role": "user", "content": "What is 2 + 2?"}],
                "temperature": 0, "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False}})
            chat = json.loads(raw)
            assert status == 200 and chat["choices"][0]["message"]["content"], chat
            record("chat", response=chat)
            for label, changes in [("sampling", {"temperature": 0.7}),
                                   ("logprobs", {"logprobs": 1}),
                                   ("prompt_logprobs", {"prompt_logprobs": 0}),
                                   ("echo", {"echo": True}),
                                   ("kv_transfer", {"vllm_xargs": {"kv_transfer_params": {"do_remote_prefill": True}}}),
                                   ("context_overflow", {"prompt": [100] * args.context, "max_tokens": 1})]:
                status, body = request("/v1/completions", payload(**changes))
                assert status == 400, (label, status, body)
                assert json.loads(body)["error"]["message"], body
                record("reject_" + label, status=status, response=json.loads(body))
            for label, changes in [("top_k_one", {"temperature": 1, "top_k": 1}),
                                   ("temperature_below_epsilon", {"temperature": 0.000009}),]:
                response, _ = complete(**changes)
                assert response["usage"]["completion_tokens"] == 24
                record("accept_" + label, response=response)
            status, body = request("/v1/completions", payload(temperature=0.00001))
            assert status == 400, (status, body)
            record("reject_temperature_at_epsilon", status=status, response=json.loads(body))
            edge, _ = complete(prompt=[100] * (args.context - 2), max_tokens=2)
            assert edge["usage"]["completion_tokens"] == 2
            record("context_boundary", usage=edge["usage"])
            wait_for(lambda s: s["num_requests_running"] == s["num_requests_waiting"] == 0)
            # Four live requests occupy all slots. A fifth waits, then disconnects.
            # Keep these sockets open until the metrics prove admission and queuing.
            for _ in range(5):
                if len(handles) == 4:
                    wait_for(lambda s: s["num_requests_running"] == 4 and s["num_requests_waiting"] == 0)
                body = json.dumps(payload(prompt="Count upwards from one:", max_tokens=args.context - 16,
                                          stream=True)).encode()
                sock = socket.create_connection(("127.0.0.1", args.port), timeout=10)
                handles.append(sock)
                header = (f"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                          f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode()
                sock.sendall(header + body)
            admitted = wait_for(lambda s: s["num_requests_running"] == 4 and s["num_requests_waiting"] == 1)
            handles[-1].shutdown(socket.SHUT_RDWR)
            handles.pop().close()
            queued_cancelled = wait_for(lambda s: s["num_requests_waiting"] == 0 and s["num_requests_running"] == 4)
            for sock in handles:
                sock.shutdown(socket.SHUT_RDWR)
                sock.close()
            handles.clear()
            reclaimed = wait_for(lambda s: all(value == 0 for value in s.values()))
            record("queued_and_active_cancellation", admitted=admitted,
                   queued_cancelled=queued_cancelled, reclaimed=reclaimed)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                reused = list(pool.map(stream, range(4)))
            record("readmission_after_cancellation", responses=reused)
            record("idle_resources", metrics=wait_for(lambda s: all(v == 0 for v in s.values())))
            report["passed"] = True
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            for sock in handles:
                sock.close()
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            report["server_exit_code"] = proc.returncode
            if report["passed"] and proc.returncode != 0:
                report["passed"] = False
                report["error"] = "server did not shut down gracefully"
            (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    assert report["passed"], report.get("error")


if __name__ == "__main__":
    main()
