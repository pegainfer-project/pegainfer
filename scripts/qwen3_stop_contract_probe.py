#!/usr/bin/env python3
"""Validate the Qwen3 stop contract against a live, un-migrated Qwen3.5 server.

Both servers must already be running. The default explicit stop set covers the
vocabulary, so its first returned token must stop generation. --stop-token-id
selects a known trigger instead. Token IDs, token-ID-formatted logprobs, usage,
and the first terminal position are checked together. SSE is consumed through
[DONE] to HTTP EOF, including content sharing a frame with finish metadata.

Provide each model's primary EOS ID as resolved by the serving tokenizer;
an unverified EOS must never count as a passing contract check. A healthy legacy
server may fail the new stop semantics, but malformed responses or unavailable
servers are not evidence of a compatibility gap.

Example (Qwen3-4B and Qwen3.5-0.8B):
    python3 scripts/qwen3_stop_contract_probe.py \
      --qwen3-eos-token-id 151645 --qwen35-eos-token-id 248046 \
      --require-legacy-gap --out stop-contract-ab.json

--self-check tests this probe's assertions using isolated malformed HTTP
responses. It needs no GPU and is not evidence about inference correctness.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class InvalidResponse(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise InvalidResponse(code, detail)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def finite_logprob(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and -sys.float_info.max <= value <= 0
        and math.isfinite(value)
    )


def token_ids(value: Any, vocab_size: int) -> list[int]:
    require(isinstance(value, list), "token_ids_missing", "expected a token_ids array")
    require(
        all(is_int(item) and 0 <= item < vocab_size for item in value),
        "token_id_range",
        "token IDs must be integers within the model vocabulary",
    )
    return value


def choice_data(
    choice: dict[str, Any], vocab_size: int, want_logprobs: bool, *, stream: bool
) -> tuple[list[int], list[float]]:
    require(isinstance(choice.get("text"), str), "text_type", "choice.text must be a string")
    finish = choice.get("finish_reason")
    require(finish in (None, "stop", "length"), "finish_reason", "unexpected finish_reason")
    require(
        stream or finish is not None, "finish_reason", "a completion requires terminal metadata"
    )
    reason = choice.get("stop_reason")
    require(
        reason is None or is_int(reason),
        "stop_reason_type",
        "stop_reason must be an integer or null",
    )
    require(
        reason is None or 0 <= reason < vocab_size,
        "stop_reason_range",
        "stop_reason is outside the vocabulary",
    )
    require(
        reason is None or finish == "stop",
        "stop_reason_without_stop",
        "stop_reason requires a stop finish",
    )
    ids = choice.get("token_ids")
    lp = choice.get("logprobs")
    if stream and ids is None:
        # vLLM emits prompt metadata and a pure finish frame without generated IDs.
        require(
            not choice["text"]
            and lp is None
            and (finish is not None or choice.get("prompt_token_ids") is not None),
            "token_ids_missing",
            "content requires returned token IDs",
        )
        return [], []
    ids = token_ids(ids, vocab_size)
    require(
        not (stream and finish is not None and choice["text"] and not ids),
        "terminal_text_without_tokens",
        "text-only decoder flush must precede finish metadata",
    )
    if lp is None:
        require(
            not (want_logprobs and ids),
            "logprobs_missing",
            "generated tokens need their own logprobs",
        )
        return ids, []
    require(isinstance(lp, dict), "logprobs_shape", "expected completions-style logprobs")
    fields = ("tokens", "token_logprobs", "top_logprobs", "text_offset")
    require(
        all(isinstance(lp.get(key), list) and len(lp[key]) == len(ids) for key in fields),
        "logprob_count_mismatch",
        "all logprob arrays must cover exactly the returned IDs",
    )
    require(
        lp["tokens"] == [f"token_id:{item}" for item in ids],
        "logprob_token_mismatch",
        "logprob token identities differ from returned IDs",
    )
    require(
        all(finite_logprob(value) for value in lp["token_logprobs"]),
        "logprob_value",
        "invalid token logprob",
    )
    require(
        all(is_int(value) and value >= 0 for value in lp["text_offset"]),
        "logprob_offset",
        "text offsets must be nonnegative integers",
    )
    for top in lp["top_logprobs"]:
        require(isinstance(top, dict), "top_logprobs_shape", "expected a top-logprob mapping")
        for key, value in top.items():
            require(
                isinstance(key, str)
                and key.startswith("token_id:")
                and key[9:].isascii()
                and key[9:].isdigit()
                and 0 <= int(key[9:]) < vocab_size
                and finite_logprob(value),
                "top_logprobs_value",
                "invalid top-logprob token or value",
            )
    return ids, lp["token_logprobs"]


def response_choice(body: Any, model: str) -> dict[str, Any]:
    require(isinstance(body, dict), "response_shape", "expected a JSON object")
    require("error" not in body, "server_error", "server returned an error response")
    require(
        body.get("model") == model, "model_mismatch", "response model differs from requested model"
    )
    choices = body.get("choices")
    require(
        isinstance(choices, list) and len(choices) == 1,
        "choice_count",
        "n=1 requires exactly one choice",
    )
    choice = choices[0]
    require(isinstance(choice, dict), "choice_shape", "expected a choice object")
    require(
        is_int(choice.get("index")) and choice["index"] == 0,
        "choice_index",
        "expected choice index 0",
    )
    require(
        body.get("stop_reason") is None, "stop_reason_location", "stop_reason belongs to the choice"
    )
    return choice


def completion_count(usage: Any, ids: list[int]) -> int:
    require(isinstance(usage, dict), "usage_missing", "expected completion usage")
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    require(
        all(is_int(usage.get(key)) and usage[key] >= 0 for key in fields),
        "usage_type",
        "usage counts must be nonnegative integers",
    )
    require(
        usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
        "usage_total",
        "inconsistent total_tokens",
    )
    require(
        usage["completion_tokens"] == len(ids),
        "completion_count_mismatch",
        "usage differs from returned token count",
    )
    return usage["completion_tokens"]


def read_stream(response: Any, model: str, vocab_size: int, timeout: float) -> dict[str, Any]:
    result: dict[str, Any] = {"token_ids": [], "token_logprobs": [], "text": "", "done_seen": False}
    finished = False
    prompt_seen = False
    usage = None
    data_lines: list[str] = []
    deadline = time.monotonic() + timeout

    def event(data: str) -> None:
        nonlocal finished, prompt_seen, usage
        require(not result["done_seen"], "stream_after_done", "received an event after [DONE]")
        if data == "[DONE]":
            require(finished, "stream_missing_finish", "[DONE] arrived before finish metadata")
            require(
                usage is not None,
                "stream_missing_usage",
                "include_usage requires a final usage frame",
            )
            result["done_seen"] = True
            return
        body = json.loads(data)
        require(isinstance(body, dict), "response_shape", "expected an SSE JSON object")
        require("error" not in body, "server_error", "server returned an SSE error")
        require(
            body.get("model") == model,
            "model_mismatch",
            "stream model differs from requested model",
        )
        if body.get("choices") == []:
            require(
                finished and usage is None,
                "stream_usage_order",
                "usage must occur once after finish",
            )
            require(
                isinstance(body.get("usage"), dict), "usage_missing", "empty choices require usage"
            )
            require(
                all(
                    body.get(key) is None
                    for key in ("text", "token_ids", "logprobs", "finish_reason", "stop_reason")
                ),
                "stream_usage_content",
                "usage-only frame contains completion content",
            )
            usage = body["usage"]
            return
        require(not finished, "stream_after_finish", "received a choice after finish metadata")
        choice = response_choice(body, model)
        require(body.get("usage") is None, "stream_usage_order", "unexpected per-delta usage")
        prompt = choice.get("prompt_token_ids")
        if prompt is not None:
            require(
                not prompt_seen and not result["token_ids"],
                "stream_prompt_order",
                "prompt metadata must precede generated tokens",
            )
            token_ids(prompt, vocab_size)
            prompt_seen = True
        ids, values = choice_data(choice, vocab_size, True, stream=True)
        result["token_ids"].extend(ids)
        result["token_logprobs"].extend(values)
        result["text"] += choice["text"]
        if choice.get("finish_reason") is not None:
            result["finish_reason"] = choice["finish_reason"]
            result["stop_reason"] = choice.get("stop_reason")
            finished = True

    while True:
        require(
            time.monotonic() <= deadline,
            "stream_timeout",
            "stream did not reach HTTP EOF within timeout",
        )
        raw = response.readline(1_048_577)
        if not raw:
            break
        require(len(raw) <= 1_048_576, "sse_line_size", "SSE line exceeds 1 MiB")
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                event("\n".join(data_lines))
                data_lines.clear()
        elif line.startswith(":"):
            continue
        else:
            field, _, value = line.partition(":")
            if field == "data":
                require(not result["done_seen"], "stream_after_done", "received data after [DONE]")
                data_lines.append(value.removeprefix(" "))
    require(not data_lines, "sse_incomplete_event", "EOF before the SSE event delimiter")
    require(result["done_seen"], "stream_missing_done", "EOF before [DONE]")
    result["completion_tokens"] = completion_count(usage, result["token_ids"])
    return result


def request_completion(
    url: str,
    model: str,
    vocab_size: int,
    args: argparse.Namespace,
    *,
    ignore_eos: bool,
    stop_ids: list[int] | None = None,
    logprobs: bool = False,
    stream: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": args.prompt,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "n": 1,
        "ignore_eos": ignore_eos,
        "stream": stream,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
    }
    if stop_ids is not None:
        payload["stop_token_ids"] = stop_ids
    if logprobs or stream:
        payload["logprobs"] = 1
    if stream:
        payload["stream_options"] = {"include_usage": True}
    request = Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        },
    )
    started = time.monotonic()
    result: dict[str, Any] = {"http_status": None, "valid_response": False, "error_codes": []}
    try:
        with urlopen(request, timeout=args.timeout) as response:
            result["http_status"] = response.status
            require(response.status == 200, "http_status", "expected HTTP 200")
            expected_type = "text/event-stream" if stream else "application/json"
            require(
                response.headers.get_content_type() == expected_type,
                "content_type",
                "unexpected response Content-Type",
            )
            if stream:
                result.update(read_stream(response, model, vocab_size, args.timeout))
            else:
                body = json.loads(response.read())
                choice = response_choice(body, model)
                ids, values = choice_data(choice, vocab_size, logprobs, stream=False)
                result.update(
                    {
                        "token_ids": ids,
                        "token_logprobs": values,
                        "text": choice["text"],
                        "finish_reason": choice.get("finish_reason"),
                        "stop_reason": choice.get("stop_reason"),
                        "completion_tokens": completion_count(body.get("usage"), ids),
                    }
                )
        result["valid_response"] = True
        values = result["token_logprobs"]
        result["trigger_logprob"] = values[-1] if values else None
    except InvalidResponse as error:
        result.update(error=str(error), error_codes=[error.code])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        result.update(error=f"invalid_json: {error}", error_codes=["invalid_json"])
    except HTTPError as error:
        result.update(
            http_status=error.code, error=f"http_error: {error}", error_codes=["http_error"]
        )
    except (URLError, OSError, HTTPException) as error:
        result.update(error=f"transport_error: {error}", error_codes=["transport_error"])
    result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
    return result


def validate_sequence(
    result: dict[str, Any],
    stop_ids: set[int],
    max_tokens: int,
    *,
    ignore_eos: bool,
    eos_id: int,
    require_stop: bool,
) -> list[str]:
    if not result["valid_response"]:
        return result["error_codes"]
    try:
        ids = result["token_ids"]
        require(
            0 < len(ids) <= max_tokens,
            "token_budget_exceeded",
            "output must contain 1..max_tokens tokens",
        )
        first_stop = next(
            (
                index
                for index, token in enumerate(ids)
                if token in stop_ids or (not ignore_eos and token == eos_id)
            ),
            None,
        )
        if first_stop is None:
            require(
                result["finish_reason"] == "length"
                and result["stop_reason"] is None
                and len(ids) == max_tokens,
                "length_outcome",
                "a sequence without a trigger must exhaust the token budget",
            )
            require(
                not require_stop, "stop_not_exercised", "the selected stop token was not generated"
            )
        else:
            require(
                first_stop == len(ids) - 1,
                "tokens_after_stop",
                f"first stop at {first_stop}, but output has {len(ids)} tokens",
            )
            require(
                result["finish_reason"] == "stop",
                "stop_finish_reason",
                "the trigger must finish with stop",
            )
            expected_reason = None if not ignore_eos and ids[-1] == eos_id else ids[-1]
            require(
                result["stop_reason"] == expected_reason,
                "stop_reason_mismatch",
                f"expected stop_reason={expected_reason!r}",
            )
    except InvalidResponse as error:
        return [error.code]
    return []


def probe_models(url: str, model: str, timeout: float) -> dict[str, Any]:
    try:
        with urlopen(url.rstrip("/") + "/v1/models", timeout=timeout) as response:
            require(response.status == 200, "http_status", "expected HTTP 200")
            body = json.loads(response.read())
        require(
            isinstance(body, dict) and isinstance(body.get("data"), list),
            "models_shape",
            "expected a model list",
        )
        require(
            any(isinstance(item, dict) and item.get("id") == model for item in body["data"]),
            "model_missing",
            "requested model is not served",
        )
        return {"expected_model_present": True, "error_codes": []}
    except InvalidResponse as error:
        return {"expected_model_present": False, "error_codes": [error.code], "error": str(error)}
    except (UnicodeDecodeError, json.JSONDecodeError, URLError, OSError, HTTPException) as error:
        return {
            "expected_model_present": False,
            "error_codes": ["models_unavailable"],
            "error": str(error),
        }


def run_target(
    name: str,
    url: str,
    model: str,
    vocab_size: int,
    args: argparse.Namespace,
    eos_token_id: int,
) -> dict[str, Any]:
    require(vocab_size > 0, "vocab_size", "vocabulary size must be positive")
    require(
        is_int(eos_token_id) and 0 <= eos_token_id < vocab_size,
        "eos_id",
        "supply the model's primary EOS ID within its vocabulary",
    )
    selected = args.stop_token_id
    require(
        selected is None or 0 <= selected < vocab_size,
        "stop_id",
        "selected stop ID is outside the vocabulary",
    )
    stops = list(range(vocab_size)) if selected is None else [selected]
    stop_set = set(stops)
    models = probe_models(url, model, args.timeout)
    cases: dict[str, Any] = {}

    def call(
        label: str,
        *,
        explicit: bool,
        ignore_eos: bool = True,
        lp: bool = False,
        stream: bool = False,
        descending: bool = False,
    ) -> dict[str, Any]:
        ids = list(reversed(stops)) if descending else stops
        result = request_completion(
            url,
            model,
            vocab_size,
            args,
            ignore_eos=ignore_eos,
            stop_ids=ids if explicit else None,
            logprobs=lp,
            stream=stream,
        )
        result["validation_errors"] = validate_sequence(
            result,
            stop_set if explicit else set(),
            args.max_tokens,
            ignore_eos=ignore_eos,
            eos_id=eos_token_id,
            require_stop=explicit,
        )
        return {"name": label, **result}

    for label, options in (
        ("control", {"explicit": False}),
        ("explicit_stop_ignore_eos", {"explicit": True}),
        ("explicit_stop_eos_enabled", {"explicit": True, "ignore_eos": False}),
        ("stop_descending", {"explicit": True, "descending": True}),
        ("trigger_logprob", {"explicit": True, "lp": True}),
        ("streaming", {"explicit": True, "stream": True}),
    ):
        cases[label] = call(label, **options)
    with ThreadPoolExecutor(max_workers=6) as pool:
        cases["mixed"] = list(
            pool.map(
                lambda explicit: call("explicit" if explicit else "control", explicit=explicit),
                [False, True] * 3,
            )
        )
    ascending = cases["explicit_stop_ignore_eos"]
    descending = cases["stop_descending"]
    order_errors = ascending["validation_errors"] + descending["validation_errors"]
    if not order_errors and any(
        ascending[key] != descending[key] for key in ("token_ids", "finish_reason", "stop_reason")
    ):
        order_errors = ["stop_set_order_changed_output"]
    failures = {
        "model_present": models["error_codes"],
        "baseline_control": cases["control"]["validation_errors"],
        "explicit_stop_ignore_eos": ascending["validation_errors"],
        "explicit_stop_eos_enabled": cases["explicit_stop_eos_enabled"]["validation_errors"],
        "stop_set_order_invariant": order_errors,
        "trigger_logprob_preserved": cases["trigger_logprob"]["validation_errors"],
        "stream_reports_typed_stop": cases["streaming"]["validation_errors"],
        "mixed_controls_pass_3_of_3": [
            code
            for item in cases["mixed"]
            if item["name"] == "control"
            for code in item["validation_errors"]
        ],
        "mixed_explicit_stops_pass_3_of_3": [
            code
            for item in cases["mixed"]
            if item["name"] == "explicit"
            for code in item["validation_errors"]
        ],
    }
    checks = {key: not errors for key, errors in failures.items()}
    responses = [value for key, value in cases.items() if key != "mixed"] + cases["mixed"]
    healthy = (
        checks["model_present"]
        and checks["baseline_control"]
        and checks["mixed_controls_pass_3_of_3"]
        and all(item["valid_response"] for item in responses)
    )
    explicit = [item for item in responses if item["name"] != "control"]
    ignored_stops = [
        item["valid_response"]
        and item.get("finish_reason") == "length"
        and item.get("stop_reason") is None
        and item.get("completion_tokens") == args.max_tokens
        and any(token in stop_set for token in item.get("token_ids", []))
        for item in explicit
    ]
    explicit_stop_gap = (
        healthy
        and any(ignored_stops)
        and all(
            not item["validation_errors"] or ignored
            for item, ignored in zip(explicit, ignored_stops)
        )
        and "stop_set_order_changed_output" not in order_errors
    )
    return {
        "name": name,
        "url": url,
        "model": model,
        "vocab_size": vocab_size,
        "eos_token_id": eos_token_id,
        "stop_set_size": len(stops),
        "model_probe": models,
        "cases": cases,
        "checks": checks,
        "failures": failures,
        "healthy": healthy,
        "explicit_stop_gap": explicit_stop_gap,
        "new_contract_passed": all(checks.values()),
    }


def print_target(target: dict[str, Any]) -> None:
    print(f"\n{target['name']}: {target['url']} ({target['model']})")
    for name, passed in target["checks"].items():
        detail = "" if passed else ": " + ", ".join(sorted(set(target["failures"][name])))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}{detail}")
    print(f"healthy={target['healthy']}, new_contract_passed={target['new_contract_passed']}")


# These fixtures exercise the probe, not the inference engine. Each negative
# case changes one response property and must fail for its named diagnostic.
SELF_MODEL = "probe-self-check"
SELF_STOP = 17
SELF_EOS = 31
SELF_VOCAB = 32


def fixture_choice(
    ids: list[int], finish: str | None, reason: int | None, lp: bool
) -> dict[str, Any]:
    choice: dict[str, Any] = {
        "index": 0,
        "text": "",
        "token_ids": ids,
        "finish_reason": finish,
        "stop_reason": reason,
        "logprobs": None,
    }
    if lp:
        choice["logprobs"] = {
            "tokens": [f"token_id:{item}" for item in ids],
            "token_logprobs": [-0.5] * len(ids),
            "top_logprobs": [{f"token_id:{item}": -0.5} for item in ids],
            "text_offset": list(range(len(ids))),
        }
    return choice


class FixtureHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def send_json(self, body: dict[str, Any]) -> None:
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        self.send_json({"data": [{"id": SELF_MODEL}]})

    def do_POST(self) -> None:
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not (
            payload.get("return_token_ids")
            and payload.get("return_tokens_as_token_ids")
            and payload.get("n") == 1
        ):
            self.send_error(400, "probe must request exact token identities")
            return
        mode = self.server.mode
        explicit = "stop_token_ids" in payload
        lp = bool(payload.get("logprobs"))
        ids = [SELF_STOP] if explicit else [7] * payload["max_tokens"]
        finish, reason = ("stop", SELF_STOP) if explicit else ("length", None)
        if mode in ("valid_prefix", "missing_final_logprob", "cross_frame_missing_logprob"):
            ids = [7, SELF_STOP]
        if mode == "earlier_stop":
            ids = [SELF_STOP, 7, SELF_STOP]
        if mode == "over_budget":
            ids = [7] * payload["max_tokens"] + [SELF_STOP]
        if mode == "legacy" and explicit:
            ids, finish, reason = [7] * payload["max_tokens"], "length", None
        if mode == "eos":
            ids, reason = [SELF_EOS], None
        if mode == "wrong_eos_priority":
            ids, reason = [SELF_EOS], SELF_EOS
        if mode == "wrong_trigger" and explicit:
            reason = 19
        if mode == "fake_eos":
            reason = None
        if mode == "string_reason":
            reason = str(SELF_STOP)
        if mode == "boolean_reason":
            reason = True
        if mode == "invalid_id":
            ids = [-1]
        choice = fixture_choice(ids, finish, reason, lp)
        if mode == "wrong_logprob_token":
            choice["logprobs"]["tokens"][-1] = "token_id:19"
        if mode == "missing_final_logprob":
            choice["logprobs"]["token_logprobs"].pop()
        if mode == "null_logprob":
            choice["logprobs"]["token_logprobs"][-1] = None
        if mode == "missing_ids":
            choice.pop("token_ids")
        if mode == "choice_index":
            choice["index"] = 1
        if mode == "missing_finish":
            choice["finish_reason"] = None
            choice["stop_reason"] = None
        if mode == "server_error":
            self.send_json({"model": SELF_MODEL, "error": {"message": "injected failure"}})
            return
        model = "wrong-model" if mode == "wrong_model" else SELF_MODEL
        usage = {"prompt_tokens": 1, "completion_tokens": len(ids), "total_tokens": 1 + len(ids)}
        if mode == "wrong_count":
            usage["completion_tokens"] += 1
            usage["total_tokens"] += 1
        body = {"model": model, "choices": [choice], "usage": usage}
        if mode == "extra_choice":
            body["choices"].append(fixture_choice([19], None, None, lp))
        if not payload["stream"]:
            self.send_json(body)
            return
        if payload.get("stream_options") != {"include_usage": True}:
            self.send_error(400, "probe must request stream usage")
            return
        prompt = {"model": model, "choices": [{"index": 0, "text": "", "prompt_token_ids": [1]}]}
        terminal = {
            "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": finish, "stop_reason": reason}],
        }
        delta = copy.deepcopy(body)
        delta.pop("usage")
        delta["choices"][0]["finish_reason"] = None
        delta["choices"][0]["stop_reason"] = None
        flush = {"model": model, "choices": [fixture_choice([], None, None, True)]}
        flush["choices"][0]["text"] = "decoded text"
        frames = [prompt, delta, flush, terminal]
        if mode == "terminal_token":
            frames = [prompt, {"model": model, "choices": [choice]}]
        if mode == "terminal_extra_token":
            terminal["choices"] = [fixture_choice([SELF_STOP], finish, reason, True)]
            usage["completion_tokens"] += 1
            usage["total_tokens"] += 1
        if mode == "hidden_terminal_logprob":
            hidden = fixture_choice([19], finish, reason, True)
            hidden["token_ids"] = []
            terminal["choices"] = [hidden]
        if mode == "cross_frame_missing_logprob":
            delta["choices"] = [fixture_choice([7], None, None, True)]
            terminal["choices"] = [fixture_choice([SELF_STOP], finish, reason, False)]
        if mode == "terminal_text_only":
            terminal["choices"] = [fixture_choice([], finish, reason, True)]
            terminal["choices"][0]["text"] = "extra text"
        if mode == "after_finish":
            frames.append({"model": model, "choices": [fixture_choice([19], None, None, True)]})
        if mode != "missing_usage":
            frames.append({"model": model, "choices": [], "usage": usage})
        if mode == "usage_content":
            frames[-1]["token_ids"] = [19]
        if mode == "sse_error":
            frames.insert(-1, {"model": model, "error": {"message": "injected failure"}})
        wire = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
        if mode != "missing_done":
            wire += "data: [DONE]\n\n"
        if mode == "after_done":
            wire += f"data: {json.dumps(delta)}\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            self.wfile.write(wire.encode())
        except (BrokenPipeError, ConnectionResetError):
            pass


def run_self_check() -> int:
    args = argparse.Namespace(prompt="probe", max_tokens=8, timeout=2.0, stop_token_id=None)
    cases = [
        ("valid", False, False, True, None),
        ("valid", True, False, True, None),
        ("terminal_token", True, False, True, None),
        ("valid_prefix", False, True, True, None),
        ("eos", False, False, False, None),
        ("earlier_stop", False, True, True, "tokens_after_stop"),
        ("earlier_stop", True, True, True, "tokens_after_stop"),
        ("over_budget", False, True, True, "token_budget_exceeded"),
        ("over_budget", True, True, True, "token_budget_exceeded"),
        ("wrong_trigger", False, False, True, "stop_reason_mismatch"),
        ("wrong_trigger", True, False, True, "stop_reason_mismatch"),
        ("fake_eos", False, False, False, "stop_reason_mismatch"),
        ("wrong_eos_priority", False, False, False, "stop_reason_mismatch"),
        ("string_reason", False, False, True, "stop_reason_type"),
        ("boolean_reason", False, False, True, "stop_reason_type"),
        ("missing_final_logprob", False, True, True, "logprob_count_mismatch"),
        ("cross_frame_missing_logprob", True, True, True, "logprobs_missing"),
        ("null_logprob", False, False, True, "logprob_value"),
        ("wrong_logprob_token", False, False, True, "logprob_token_mismatch"),
        ("wrong_logprob_token", True, False, True, "logprob_token_mismatch"),
        ("terminal_extra_token", True, False, True, "tokens_after_stop"),
        ("hidden_terminal_logprob", True, False, True, "logprob_count_mismatch"),
        ("terminal_text_only", True, False, True, "terminal_text_without_tokens"),
        ("after_finish", True, False, True, "stream_after_finish"),
        ("usage_content", True, False, True, "stream_usage_content"),
        ("server_error", False, False, True, "server_error"),
        ("sse_error", True, False, True, "server_error"),
        ("missing_finish", False, False, True, "finish_reason"),
        ("after_done", True, False, True, "stream_after_done"),
        ("missing_usage", True, False, True, "stream_missing_usage"),
        ("missing_done", True, False, True, "stream_missing_done"),
        ("wrong_count", False, False, True, "completion_count_mismatch"),
        ("wrong_count", True, False, True, "completion_count_mismatch"),
        ("extra_choice", False, False, True, "choice_count"),
        ("extra_choice", True, False, True, "choice_count"),
        ("choice_index", True, False, True, "choice_index"),
        ("missing_ids", False, False, True, "token_ids_missing"),
        ("missing_ids", True, False, True, "token_ids_missing"),
        ("invalid_id", False, False, True, "token_id_range"),
        ("wrong_model", False, False, True, "model_mismatch"),
        ("wrong_model", True, False, True, "model_mismatch"),
    ]
    failures = 0
    with ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            for mode, stream, single, ignore_eos, expected in cases:
                server.mode = mode
                stops = [SELF_STOP] if single else list(range(SELF_VOCAB))
                result = request_completion(
                    url,
                    SELF_MODEL,
                    SELF_VOCAB,
                    args,
                    ignore_eos=ignore_eos,
                    stop_ids=stops,
                    logprobs=True,
                    stream=stream,
                )
                actual = validate_sequence(
                    result,
                    set(stops),
                    args.max_tokens,
                    ignore_eos=ignore_eos,
                    eos_id=SELF_EOS,
                    require_stop=True,
                )
                ok = actual == ([] if expected is None else [expected])
                failures += not ok
                print(
                    f"[{'PASS' if ok else 'FAIL'}] {mode} ({'SSE' if stream else 'JSON'}): {actual or 'valid'}, expected={expected or 'valid'}"
                )
            for mode, expected_pass, expected_health, expected_gap in (
                ("valid", True, True, False),
                ("legacy", False, True, True),
                ("wrong_trigger", False, True, False),
                ("wrong_model", False, False, False),
            ):
                server.mode = mode
                target = run_target("self-check", url, SELF_MODEL, SELF_VOCAB, args, SELF_EOS)
                ok = (
                    target["new_contract_passed"] == expected_pass
                    and target["healthy"] == expected_health
                    and target["explicit_stop_gap"] == expected_gap
                )
                failures += not ok
                print(
                    f"[{'PASS' if ok else 'FAIL'}] full probe {mode}: pass={target['new_contract_passed']}, healthy={target['healthy']}"
                )
        finally:
            server.shutdown()
            thread.join()
    return int(failures > 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix, port, model, vocab in (
        ("qwen3", 18081, "qwen3-adapted", 151936),
        ("qwen35", 18082, "qwen35-legacy", 248320),
    ):
        parser.add_argument(f"--{prefix}-url", default=f"http://127.0.0.1:{port}")
        parser.add_argument(f"--{prefix}-model", default=model)
        parser.add_argument(f"--{prefix}-vocab-size", type=int, default=vocab)
        parser.add_argument(
            f"--{prefix}-eos-token-id",
            type=int,
            help="Primary EOS ID resolved by the serving tokenizer (required for live validation)",
        )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP read timeout and SSE deadline in seconds, including the tail after [DONE]",
    )
    parser.add_argument(
        "--stop-token-id", type=int, help="Use one known trigger instead of all vocabulary IDs"
    )
    parser.add_argument(
        "--out", type=Path, help="Write results and exact failure diagnostics as JSON"
    )
    parser.add_argument(
        "--strict-both",
        action="store_true",
        help="Require both healthy targets to satisfy the new contract",
    )
    parser.add_argument(
        "--require-legacy-gap",
        action="store_true",
        help="Require a healthy legacy target with a semantic stop-contract failure",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Check isolated malformed HTTP responses and exact rejection reasons; no GPU required",
    )
    args = parser.parse_args()
    if not args.self_check:
        if args.qwen3_eos_token_id is None or args.qwen35_eos_token_id is None:
            parser.error(
                "live validation requires both --qwen3-eos-token-id and --qwen35-eos-token-id"
            )
        if args.max_tokens <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error("--max-tokens and --timeout must be positive and finite")
        if args.strict_both and args.require_legacy_gap:
            parser.error("--strict-both conflicts with --require-legacy-gap")
    return args


def main() -> int:
    args = parse_args()
    if args.self_check:
        return run_self_check()
    try:
        targets = {
            prefix: run_target(
                prefix,
                getattr(args, f"{prefix}_url"),
                getattr(args, f"{prefix}_model"),
                getattr(args, f"{prefix}_vocab_size"),
                args,
                getattr(args, f"{prefix}_eos_token_id"),
            )
            for prefix in ("qwen3", "qwen35")
        }
    except InvalidResponse as error:
        print(str(error), file=sys.stderr)
        return 2
    adapted, legacy = targets["qwen3"], targets["qwen35"]
    gap = adapted["new_contract_passed"] and legacy["explicit_stop_gap"]
    report = {
        "schema_version": 3,
        "config": {key: value for key, value in vars(args).items() if key != "out"},
        "targets": targets,
        "legacy_gap_observed": gap,
    }
    for target in targets.values():
        print_target(target)
    print(f"\nlegacy_gap_observed={gap}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return int(
        not adapted["new_contract_passed"]
        or not legacy["healthy"]
        or (args.strict_both and not legacy["new_contract_passed"])
        or (args.require_legacy_gap and not gap)
    )


if __name__ == "__main__":
    raise SystemExit(main())
