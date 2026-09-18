#!/usr/bin/env python3
"""Measure vLLM long-prefill threshold effects under live decode load.

The harness starts one vLLM server per threshold configuration. Each run:

1. warms the model;
2. starts two long-lived decode requests (A and B);
3. submits a long prompt C followed immediately by a short prompt D;
4. records streamed token IDs, TTFT, completion latency, and server metadata.

Use an ABBA threshold order to reduce startup/thermal ordering bias. The script
does not interpret wall-clock timings as scheduler traces; it reports them as
HTTP observations and keeps output-token equality as a separate result.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.metadata
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bench_http_common import (
    current_commit,
    detect_hardware_toolchain,
    model_fingerprint,
    sha256_file,
    shell_command,
    write_json,
)


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent


@dataclass
class RequestMeasurement:
    label: str
    request_id: str
    prompt_tokens: int
    max_tokens: int
    ok: bool
    status: int | None
    error: str | None
    start_wall_s: float
    first_token_wall_s: float | None
    end_wall_s: float
    ttft_ms: float | None
    latency_ms: float
    output_token_ids: list[int]
    output_text_sha256: str
    finish_reason: str | None
    usage: dict[str, Any] | None
    metrics: dict[str, Any] | None


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated non-negative integers")
    return values


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def tail_text(path: Path, max_bytes: int = 12_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return f"failed to read log: {exc}"


def wait_for_server(process: subprocess.Popen[bytes], port: int, log_path: Path, timeout_s: float) -> float:
    started = time.perf_counter()
    deadline = started + timeout_s
    while time.perf_counter() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"vLLM exited with code {exit_code} before readiness\n{tail_text(log_path)}"
            )
        conn = http.client.HTTPConnection("127.0.0.1", port=port, timeout=2.0)
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            response.read()
            if response.status == 200:
                return (time.perf_counter() - started) * 1000.0
        except OSError:
            pass
        finally:
            conn.close()
        time.sleep(0.5)
    raise TimeoutError(f"vLLM was not ready within {timeout_s}s\n{tail_text(log_path)}")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def stream_completion(
    *,
    port: int,
    served_model_name: str,
    label: str,
    request_id: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    timeout_s: float,
    sent_event: threading.Event | None = None,
    first_token_event: threading.Event | None = None,
) -> RequestMeasurement:
    started = time.perf_counter()
    started_wall = time.time()
    deadline = started + timeout_s
    status: int | None = None
    first_token_wall: float | None = None
    first_token_perf: float | None = None
    output_token_ids: list[int] = []
    output_text: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    conn = http.client.HTTPConnection("127.0.0.1", port=port, timeout=timeout_s)

    body = {
        "model": served_model_name,
        "prompt": prompt_token_ids,
        "add_special_tokens": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 7,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "return_token_ids": True,
        "request_id": request_id,
        "cache_salt": request_id,
    }

    try:
        conn.request(
            "POST",
            "/v1/completions",
            body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        if sent_event is not None:
            sent_event.set()
        response = conn.getresponse()
        status = response.status
        if response.status != 200:
            error_body = response.read(8192).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {response.status}: {error_body}")

        while True:
            if time.perf_counter() > deadline:
                raise TimeoutError(f"request exceeded {timeout_s}s")
            raw = response.readline()
            if not raw:
                raise RuntimeError("stream ended before [DONE]")
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            payload = json.loads(data)
            if payload.get("error"):
                raise RuntimeError(json.dumps(payload["error"], sort_keys=True))
            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
            if isinstance(payload.get("metrics"), dict):
                metrics = payload["metrics"]
            choices = payload.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            token_ids = choice.get("token_ids") or []
            text = choice.get("text") or ""
            if token_ids or text:
                if first_token_perf is None:
                    first_token_perf = time.perf_counter()
                    first_token_wall = time.time()
                    if first_token_event is not None:
                        first_token_event.set()
                output_token_ids.extend(int(token) for token in token_ids)
                output_text.append(text)
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])

        ended = time.perf_counter()
        return RequestMeasurement(
            label=label,
            request_id=request_id,
            prompt_tokens=len(prompt_token_ids),
            max_tokens=max_tokens,
            ok=True,
            status=status,
            error=None,
            start_wall_s=started_wall,
            first_token_wall_s=first_token_wall,
            end_wall_s=time.time(),
            ttft_ms=(
                None if first_token_perf is None else (first_token_perf - started) * 1000.0
            ),
            latency_ms=(ended - started) * 1000.0,
            output_token_ids=output_token_ids,
            output_text_sha256=hashlib.sha256(
                "".join(output_text).encode("utf-8")
            ).hexdigest(),
            finish_reason=finish_reason,
            usage=usage,
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001 - retain request-level failures in the artifact.
        ended = time.perf_counter()
        return RequestMeasurement(
            label=label,
            request_id=request_id,
            prompt_tokens=len(prompt_token_ids),
            max_tokens=max_tokens,
            ok=False,
            status=status,
            error=f"{type(exc).__name__}: {exc}",
            start_wall_s=started_wall,
            first_token_wall_s=first_token_wall,
            end_wall_s=time.time(),
            ttft_ms=(
                None if first_token_perf is None else (first_token_perf - started) * 1000.0
            ),
            latency_ms=(ended - started) * 1000.0,
            output_token_ids=output_token_ids,
            output_text_sha256=hashlib.sha256(
                "".join(output_text).encode("utf-8")
            ).hexdigest(),
            finish_reason=finish_reason,
            usage=usage,
            metrics=metrics,
        )
    finally:
        if sent_event is not None:
            sent_event.set()
        conn.close()


def run_in_thread(kwargs: dict[str, Any]) -> tuple[threading.Thread, list[RequestMeasurement], threading.Event]:
    holder: list[RequestMeasurement] = []
    done = threading.Event()

    def target() -> None:
        try:
            holder.append(stream_completion(**kwargs))
        finally:
            done.set()

    thread = threading.Thread(target=target, name=str(kwargs["label"]), daemon=True)
    thread.start()
    return thread, holder, done


def server_command(args: argparse.Namespace, threshold: int, port: int) -> list[str]:
    vllm = Path(sys.executable).with_name("vllm")
    if not vllm.exists():
        raise FileNotFoundError(f"vLLM executable is missing next to Python: {vllm}")
    return [
        str(vllm),
        "serve",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        args.served_model_name,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--long-prefill-token-threshold",
        str(threshold),
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enforce-eager",
        "--generation-config",
        "vllm",
        "--seed",
        "0",
        "--uvicorn-log-level",
        "warning",
    ]


def run_server_case(args: argparse.Namespace, run_index: int, threshold: int) -> dict[str, Any]:
    port = reserve_port()
    command = server_command(args, threshold, port)
    log_path = args.output.with_name(
        f"{args.output.stem}.run-{run_index:02d}.threshold-{threshold}.server.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    server_env = os.environ.copy()
    server_env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PATH": str(Path(sys.executable).parent)
            + os.pathsep
            + server_env.get("PATH", ""),
            "VLLM_USE_V1": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        server_env.pop(name, None)
    if args.python_include_dir is not None:
        existing_cpath = server_env.get("CPATH")
        include_paths = [str(args.python_include_dir), str(args.python_include_dir.parent)]
        server_env["CPATH"] = os.pathsep.join(include_paths)
        if existing_cpath:
            server_env["CPATH"] += os.pathsep + existing_cpath

    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=server_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        background_threads: list[threading.Thread] = []
        try:
            startup_ms = wait_for_server(process, port, log_path, args.server_timeout)
            warmup = stream_completion(
                port=port,
                served_model_name=args.served_model_name,
                label="warmup",
                request_id=f"warmup-r{run_index}",
                prompt_token_ids=[101] * 64,
                max_tokens=8,
                timeout_s=args.request_timeout,
            )
            if not warmup.ok:
                raise RuntimeError(f"warmup failed: {warmup.error}")

            background: list[dict[str, Any]] = []
            background_state: list[tuple[list[RequestMeasurement], threading.Event]] = []
            for offset, label in enumerate(("A", "B")):
                first_token = threading.Event()
                thread, holder, done = run_in_thread(
                    {
                        "port": port,
                        "served_model_name": args.served_model_name,
                        "label": label,
                        "request_id": f"{label.lower()}-r{run_index}",
                        "prompt_token_ids": [301 + offset] * 16,
                        "max_tokens": args.background_tokens,
                        "timeout_s": args.request_timeout,
                        "first_token_event": first_token,
                    }
                )
                background_threads.append(thread)
                background_state.append((holder, done))
                background.append(
                    {
                        "label": label,
                        "request_id": f"{label.lower()}-r{run_index}",
                        "first_token_event": first_token,
                    }
                )

            for item in background:
                if not item["first_token_event"].wait(args.request_timeout):
                    raise TimeoutError(f"background request {item['label']} did not emit a token")

            trials: list[dict[str, Any]] = []
            for trial_index in range(args.repeats):
                if any(done.is_set() for _, done in background_state):
                    raise RuntimeError("A/B stopped decoding before all C/D probes were submitted")

                c_sent = threading.Event()
                c_thread, c_holder, c_done = run_in_thread(
                    {
                        "port": port,
                        "served_model_name": args.served_model_name,
                        "label": "C",
                        "request_id": f"c-r{run_index}-t{trial_index}",
                        "prompt_token_ids": [401 + trial_index] * args.long_prompt_tokens,
                        "max_tokens": args.probe_output_tokens,
                        "timeout_s": args.request_timeout,
                        "sent_event": c_sent,
                    }
                )
                if not c_sent.wait(10):
                    raise TimeoutError("C request was not sent")
                time.sleep(args.inter_arrival_ms / 1000.0)
                d_thread, d_holder, d_done = run_in_thread(
                    {
                        "port": port,
                        "served_model_name": args.served_model_name,
                        "label": "D",
                        "request_id": f"d-r{run_index}-t{trial_index}",
                        "prompt_token_ids": [501 + trial_index] * args.short_prompt_tokens,
                        "max_tokens": args.probe_output_tokens,
                        "timeout_s": args.request_timeout,
                    }
                )
                c_thread.join(args.request_timeout)
                d_thread.join(args.request_timeout)
                if not c_done.is_set() or not d_done.is_set() or not c_holder or not d_holder:
                    raise TimeoutError(f"C/D trial {trial_index} did not complete")
                c_result = c_holder[0]
                d_result = d_holder[0]
                if not c_result.ok or not d_result.ok:
                    raise RuntimeError(
                        f"C/D trial {trial_index} failed: C={c_result.error}, D={d_result.error}"
                    )
                trials.append(
                    {
                        "trial_index": trial_index,
                        "c": asdict(c_result),
                        "d": asdict(d_result),
                        "d_started_after_c_ms": (
                            d_result.start_wall_s - c_result.start_wall_s
                        )
                        * 1000.0,
                        "d_first_token_after_c_ms": (
                            None
                            if d_result.first_token_wall_s is None
                            else (d_result.first_token_wall_s - c_result.start_wall_s)
                            * 1000.0
                        ),
                    }
                )

            background_snapshot = []
            for item, (holder, done) in zip(background, background_state, strict=True):
                background_snapshot.append(
                    {
                        "label": item["label"],
                        "request_id": item["request_id"],
                        "still_running_after_probes": not done.is_set(),
                        "completed_result": asdict(holder[0]) if holder else None,
                    }
                )

            return {
                "run_index": run_index,
                "threshold": threshold,
                "port": port,
                "server_command": shell_command(command),
                "server_log": display_path(log_path),
                "startup_ms": startup_ms,
                "warmup": asdict(warmup),
                "background": background_snapshot,
                "trials": trials,
            }
        finally:
            stop_server(process)
            for thread in background_threads:
                thread.join(timeout=5)


def median(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.median(clean) if clean else None


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    thresholds = sorted({int(run["threshold"]) for run in runs})
    by_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        selected = [run for run in runs if int(run["threshold"]) == threshold]
        trials = [trial for run in selected for trial in run["trials"]]
        by_threshold[str(threshold)] = {
            "server_runs": len(selected),
            "trial_samples": len(trials),
            "c_ttft_ms_median": median([trial["c"]["ttft_ms"] for trial in trials]),
            "d_ttft_ms_median": median([trial["d"]["ttft_ms"] for trial in trials]),
            "c_latency_ms_median": median([trial["c"]["latency_ms"] for trial in trials]),
            "d_latency_ms_median": median([trial["d"]["latency_ms"] for trial in trials]),
            "d_first_token_after_c_ms_median": median(
                [trial["d_first_token_after_c_ms"] for trial in trials]
            ),
            "all_background_requests_active": all(
                item["still_running_after_probes"]
                for run in selected
                for item in run["background"]
            ),
            "request_failures": sum(
                int(not trial[label]["ok"])
                for trial in trials
                for label in ("c", "d")
            ),
        }

    output_checks: dict[str, Any] = {}
    for label in ("c", "d"):
        per_trial: dict[str, Any] = {}
        for trial_index in sorted(
            {int(trial["trial_index"]) for run in runs for trial in run["trials"]}
        ):
            outputs = [
                tuple(trial[label]["output_token_ids"])
                for run in runs
                for trial in run["trials"]
                if int(trial["trial_index"]) == trial_index
            ]
            per_trial[str(trial_index)] = {
                "samples": len(outputs),
                "unique_outputs": len(set(outputs)),
                "all_equal": len(set(outputs)) == 1,
                "token_ids": list(outputs[0]) if outputs else [],
            }
        output_checks[label] = per_trial

    comparison: dict[str, Any] = {}
    if 0 in thresholds:
        baseline = by_threshold["0"]
        for threshold in thresholds:
            if threshold == 0:
                continue
            candidate = by_threshold[str(threshold)]
            for request in ("c", "d"):
                base = baseline[f"{request}_ttft_ms_median"]
                value = candidate[f"{request}_ttft_ms_median"]
                comparison[f"threshold_{threshold}_{request}_ttft_delta_ms"] = (
                    None if base is None or value is None else value - base
                )
                comparison[f"threshold_{threshold}_{request}_ttft_ratio"] = (
                    None if base in (None, 0.0) or value is None else value / base
                )

    return {
        "by_threshold": by_threshold,
        "output_token_equality": output_checks,
        "comparison_to_threshold_0": comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threshold-order", type=parse_int_list, default=[0, 128, 128, 0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--long-prompt-tokens", type=int, default=1024)
    parser.add_argument("--short-prompt-tokens", type=int, default=16)
    parser.add_argument("--probe-output-tokens", type=int, default=8)
    parser.add_argument("--background-tokens", type=int, default=1024)
    parser.add_argument("--inter-arrival-ms", type=float, default=2.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--python-include-dir",
        type=Path,
        help="optional Python development-header directory for Triton runtime builds",
    )
    parser.add_argument("--server-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    if args.python_include_dir is not None:
        args.python_include_dir = args.python_include_dir.resolve()
    if not args.model.joinpath("config.json").is_file():
        raise SystemExit(f"model config is missing: {args.model / 'config.json'}")
    if args.python_include_dir is not None and not args.python_include_dir.joinpath(
        "Python.h"
    ).is_file():
        raise SystemExit(
            f"Python.h is missing from --python-include-dir: {args.python_include_dir}"
        )
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if any(
        value <= 0
        for value in (
            args.max_model_len,
            args.max_num_batched_tokens,
            args.long_prompt_tokens,
            args.short_prompt_tokens,
            args.probe_output_tokens,
            args.background_tokens,
        )
    ):
        raise SystemExit("token and model limits must be positive")
    if args.max_num_batched_tokens < 4:
        raise SystemExit("--max-num-batched-tokens must leave room for four sequences")
    return args


def main() -> int:
    args = parse_args()
    scheduler_source = (
        Path(importlib.util.find_spec("vllm").origin).parent
        / "v1/core/sched/scheduler.py"
    )
    document: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "vllm_long_prefill_token_threshold_ab",
        "started_at_unix_s": time.time(),
        "git_commit": current_commit(),
        "script": display_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "vllm_version": importlib.metadata.version("vllm"),
            "vllm_scheduler_source": str(scheduler_source),
            "vllm_scheduler_source_sha256": sha256_file(scheduler_source),
            "hardware": detect_hardware_toolchain(),
            "selected_gpu": args.gpu,
            "python_include_dir": (
                None
                if args.python_include_dir is None
                else str(args.python_include_dir)
            ),
        },
        "model": {
            "path": str(args.model),
            "fingerprint": model_fingerprint(str(args.model)),
        },
        "workload": {
            "threshold_order": args.threshold_order,
            "repeats_per_server": args.repeats,
            "max_model_len": args.max_model_len,
            "max_num_seqs": 4,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "long_prompt_tokens": args.long_prompt_tokens,
            "short_prompt_tokens": args.short_prompt_tokens,
            "probe_output_tokens": args.probe_output_tokens,
            "background_decode_requests": 2,
            "background_tokens": args.background_tokens,
            "inter_arrival_ms": args.inter_arrival_ms,
            "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 7},
            "prefix_caching": False,
            "enforce_eager": True,
        },
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        for run_index, threshold in enumerate(args.threshold_order):
            print(f"running server {run_index + 1}/{len(args.threshold_order)} threshold={threshold}", flush=True)
            run = run_server_case(args, run_index, threshold)
            document["runs"].append(run)
            document["partial"] = True
            write_json(args.output, document)
        document["summary"] = summarize_runs(document["runs"])
        document["partial"] = False
        document["finished_at_unix_s"] = time.time()
        write_json(args.output, document)
    except Exception as exc:  # noqa: BLE001 - preserve partial evidence before failing.
        document["partial"] = True
        document["error"] = f"{type(exc).__name__}: {exc}"
        document["finished_at_unix_s"] = time.time()
        write_json(args.output, document)
        raise

    print(json.dumps(document["summary"], indent=2, sort_keys=True))
    print(f"result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
