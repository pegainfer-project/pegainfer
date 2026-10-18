#!/usr/bin/env python3
"""Dump vLLM one-token Kimi-K2 serving/top-logprobs fixtures.

vLLM's public generation API exposes generated tokens and top logprobs rather
than stable raw logits. Use this fixture to cross-check temperature-0 greedy
serving behavior against PegaInfer full-vocab candidate logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
import transformers
import vllm
from safetensors.torch import save_file
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DEFAULT_MESSAGES = [{"role": "user", "content": "Hello"}]


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def transformers_safe_model_path(model_path: Path, out_dir: Path) -> Path:
    """Return a symlink path whose basename is safe for HF/vLLM dynamic modules."""
    resolved = model_path.resolve(strict=True)
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in resolved.name)
    safe_path = out_dir / f"_transformers_model_{safe_name}"
    if safe_path.exists() or safe_path.is_symlink():
        if safe_path.is_symlink() and safe_path.resolve(strict=True) == resolved:
            return safe_path
        raise FileExistsError(f"{safe_path} exists and is not the expected symlink")
    safe_path.symlink_to(resolved, target_is_directory=True)
    return safe_path


def load_prompt_payload(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"messages": DEFAULT_MESSAGES}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return {"messages": payload}
    if not isinstance(payload, dict):
        raise ValueError(f"prompt json must be a list or object, got {type(payload)!r}")
    return payload


def load_prompt_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [{"id": "default", "payload": {"messages": DEFAULT_MESSAGES}}]
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    raw_cases = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("prompt set must contain a non-empty cases list")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise ValueError(f"prompt case {index} must be an object")
        case_id = str(item.get("id") or item.get("case_id") or f"case_{index:03d}")
        if "/" in case_id or "\\" in case_id or case_id in {"", ".", ".."}:
            raise ValueError(f"invalid case id {case_id!r}")
        if case_id in seen:
            raise ValueError(f"duplicate case id {case_id!r}")
        seen.add(case_id)
        payload = item.get("payload")
        if payload is None:
            payload = {key: value for key, value in item.items() if key not in {"id", "case_id"}}
        if isinstance(payload, list):
            payload = {"messages": payload}
        if not isinstance(payload, dict):
            raise ValueError(f"prompt case {case_id!r} payload must be an object or list")
        cases.append({"id": case_id, "payload": payload})
    return cases


def render_prompt(
    tokenizer: Any,
    payload: dict[str, Any],
    *,
    thinking: bool,
    preserve_thinking: bool,
    add_generation_prompt: bool,
) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("prompt payload must contain a non-empty messages list")
    chat_kwargs = dict(payload.get("chat_template_kwargs") or {})
    chat_kwargs.setdefault("thinking", thinking)
    chat_kwargs.setdefault("preserve_thinking", preserve_thinking)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **chat_kwargs,
    )


def logprob_item_to_pair(token_id: int, item: Any) -> tuple[int, float]:
    value = getattr(item, "logprob", item)
    return int(token_id), float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt-json", type=Path)
    parser.add_argument("--prompt-set-json", type=Path)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--thinking", type=parse_bool, default=True)
    parser.add_argument("--preserve-thinking", type=parse_bool, default=False)
    parser.add_argument("--add-generation-prompt", type=parse_bool, default=True)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.tp_size <= 0:
        raise ValueError("--tp-size must be positive")
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    if args.prompt_json is not None and args.prompt_set_json is not None:
        raise ValueError("--prompt-json and --prompt-set-json are mutually exclusive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_load_path = transformers_safe_model_path(args.model_path, args.out_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_load_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if args.prompt_set_json is None:
        cases = [{"id": None, "payload": load_prompt_payload(args.prompt_json)}]
    else:
        cases = load_prompt_cases(args.prompt_set_json)
    rendered_prompts: list[str] = []
    input_ids_by_case: list[torch.Tensor] = []
    for case in cases:
        rendered_prompt = render_prompt(
            tokenizer,
            case["payload"],
            thinking=args.thinking,
            preserve_thinking=args.preserve_thinking,
            add_generation_prompt=args.add_generation_prompt,
        )
        rendered_prompts.append(rendered_prompt)
        input_ids_by_case.append(
            tokenizer(rendered_prompt, return_tensors="pt")["input_ids"].to(torch.int64)
        )

    llm_kwargs: dict[str, Any] = {
        "model": str(model_load_path),
        "tensor_parallel_size": args.tp_size,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=args.top_k,
    )
    outputs = llm.generate(rendered_prompts, sampling)
    case_summaries: list[dict[str, Any]] = []
    for case, rendered_prompt, input_ids, request_output in zip(
        cases, rendered_prompts, input_ids_by_case, outputs, strict=True
    ):
        completion = request_output.outputs[0]
        generated_ids = [int(token_id) for token_id in completion.token_ids]
        if not generated_ids:
            raise RuntimeError(f"vLLM returned no generated tokens for case {case['id']!r}")
        first_token = generated_ids[0]

        top_logprobs = completion.logprobs[0] if completion.logprobs else {}
        pairs = [logprob_item_to_pair(token_id, item) for token_id, item in top_logprobs.items()]
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        if not pairs:
            pairs = [(first_token, float("nan"))]

        top_ids = torch.tensor([pair[0] for pair in pairs], dtype=torch.int64)
        top_logprobs_tensor = torch.tensor([pair[1] for pair in pairs], dtype=torch.float32)
        generated = torch.tensor(generated_ids, dtype=torch.int64)
        case_out_dir = args.out_dir if case["id"] is None else args.out_dir / case["id"]
        case_out_dir.mkdir(parents=True, exist_ok=True)

        save_file(
            {
                "input_ids": input_ids.reshape(-1).cpu().contiguous(),
                "generated_token_ids": generated.contiguous(),
                "top_logprob_ids": top_ids.contiguous(),
                "top_logprobs_f32": top_logprobs_tensor.contiguous(),
            },
            str(case_out_dir / "reference.safetensors"),
        )
        metadata = {
            "engine": "vllm",
            "case_id": case["id"],
            "model_path": str(args.model_path),
            "transformers_model_path": str(model_load_path),
            "messages": case["payload"]["messages"],
            "rendered_prompt": rendered_prompt,
            "thinking": args.thinking,
            "preserve_thinking": args.preserve_thinking,
            "add_generation_prompt": args.add_generation_prompt,
            "seq_len": int(input_ids.numel()),
            "input_ids": [int(v) for v in input_ids.reshape(-1).cpu().tolist()],
            "generated_token_id": first_token,
            "generated_token_ids": generated_ids,
            "top_k_requested": args.top_k,
            "top_k_returned": int(top_ids.numel()),
            "tp_size": args.tp_size,
            "config_sha256": sha256_file(args.model_path / "config.json"),
            "tokenizer_config_sha256": sha256_file(args.model_path / "tokenizer_config.json"),
            "chat_template_sha256": sha256_file(args.model_path / "chat_template.jinja"),
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "vllm": vllm.__version__,
            },
        }
        with (case_out_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            f.write("\n")
        with (case_out_dir / "prompt.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "messages": case["payload"]["messages"],
                    "rendered_prompt": rendered_prompt,
                    "input_ids": metadata["input_ids"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")
        case_summaries.append(
            {
                "id": case["id"],
                "reference_dir": str(case_out_dir),
                "seq_len": metadata["seq_len"],
                "generated_token_id": first_token,
                "top_k_returned": metadata["top_k_returned"],
            }
        )

    if args.prompt_set_json is not None:
        with (args.out_dir / "cases.json").open("w", encoding="utf-8") as f:
            json.dump({"cases": case_summaries}, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(json.dumps({"out_dir": str(args.out_dir), "cases": case_summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
