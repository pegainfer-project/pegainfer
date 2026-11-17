#!/usr/bin/env python3
"""Dump Hugging Face greedy generation for DeepSeek-V2-Lite EP=2 comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import transformers.dynamic_module_utils as dynamic_module_utils
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_u32_le(values: list[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(4, byteorder="little", signed=False))
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def allow_missing_optional_flash_attn() -> None:
    # This is a standalone oracle-dump script; the process exits after loading the
    # model, so keeping the compatibility patch installed is intentional.
    original_check_imports = dynamic_module_utils.check_imports

    def check_imports(filename: str, *args, **kwargs):
        try:
            return original_check_imports(filename, *args, **kwargs)
        except ImportError as exc:
            message = str(exc)
            if getattr(exc, "name", None) == "flash_attn" or (
                "requires the following packages" in message and "flash_attn" in message
            ):
                return dynamic_module_utils.get_relative_imports(filename)
            raise

    dynamic_module_utils.check_imports = check_imports


def load_model(model_path: str, device_map: str, device: str):
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if device_map == "none":
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            **kwargs,
        )
    model.eval()
    return model


def first_parameter_device(model, fallback: str) -> str:
    if hasattr(model, "device"):
        return str(model.device)
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return fallback


def generate_with_transformers(
    model,
    tokenizer,
    prompt: str,
    output_len: int,
    device: str,
    ignore_eos: bool,
):
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_token_ids:
        raise RuntimeError("tokenizer returned empty prompt")

    input_device = first_parameter_device(model, device) if device == "cuda" else device
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(input_device)
    generation_kwargs = {
        "max_new_tokens": output_len,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if ignore_eos:
        generation_kwargs["eos_token_id"] = None

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated_token_ids = output_ids[0, input_len:].tolist()
    if not generated_token_ids:
        raise RuntimeError(f"HF generated no tokens for prompt {prompt!r}")
    finish_reason = "length" if len(generated_token_ids) >= output_len else "eos"
    generated_text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "generated_text": generated_text,
        "token_sha256": sha256_u32_le(generated_token_ids),
        "text_sha256": sha256_text(generated_text),
        "finish_reason": finish_reason,
    }


def load_case_set(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    raw_cases = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("case set must contain a non-empty cases list")

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise ValueError(f"case {index} must be an object")
        case_id = str(item.get("id") or item.get("case_id") or f"case_{index:03d}")
        if case_id in seen or case_id in {"", ".", ".."} or "/" in case_id or "\\" in case_id:
            raise ValueError(f"invalid or duplicate case id {case_id!r}")
        seen.add(case_id)

        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"case {case_id!r} must provide a non-empty prompt")
        output_len = int(item.get("output_len", item.get("max_new_tokens", 16)))
        if output_len <= 0:
            raise ValueError(f"case {case_id!r} output_len must be positive")
        batch_size = int(item.get("batch_size", 1))
        ignore_eos = bool(item.get("ignore_eos", False))
        if not 1 <= batch_size <= 8:
            raise ValueError(f"case {case_id!r} batch_size must be in 1..=8")
        if batch_size > 1 and not ignore_eos:
            raise ValueError(
                f"case {case_id!r} batch_size={batch_size} requires ignore_eos=true"
            )

        cases.append(
            {
                "id": case_id,
                "prompt": prompt,
                "output_len": output_len,
                "batch_size": batch_size,
                "ignore_eos": ignore_eos,
            }
        )
    return cases


def base_payload(args, model) -> dict[str, Any]:
    return {
        "model_path": args.model_path,
        "model_type": getattr(getattr(model, "config", None), "model_type", None),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "compat_patches": [
            "dynamic_module_utils.check_imports ignores missing optional flash_attn import"
        ],
        "device_map": args.device_map,
        "device": args.device,
        "dtype": "torch.bfloat16",
        "generation_mode": "transformers_generate_use_cache",
    }


def case_payload(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case["id"],
        "prompt": case["prompt"],
        "output_len": case["output_len"],
        "batch_size": case["batch_size"],
        "ignore_eos": case["ignore_eos"],
        "prompt_token_ids": result["prompt_token_ids"],
        "generated_token_ids": result["generated_token_ids"],
        "generated_text": result["generated_text"],
        "token_sha256": result["token_sha256"],
        "text_sha256": result["text_sha256"],
        "finish_reason": result["finish_reason"],
        "generation": {
            "do_sample": False,
            "max_new_tokens": case["output_len"],
            "use_cache": True,
            "ignore_eos": case["ignore_eos"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", required=True, help="HF model path or id")
    parser.add_argument("--prompt", default="Hello", help="Prompt text")
    parser.add_argument("--output-len", type=int, default=16, help="Greedy output length")
    parser.add_argument(
        "--case-set-json",
        type=Path,
        help="Optional JSON case set; emits schema=2 with one HF row per case",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="HF device_map value; use 'none' for a single-device local load",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device used when device_map=none",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Keep generating even if the model emits eos",
    )
    parser.add_argument("--out", default="-", help="Write JSON to file; '-' prints to stdout")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if model_path.exists() and not model_path.is_dir():
        print(f"error: model path {model_path} is not a directory", file=sys.stderr)
        return 1

    allow_missing_optional_flash_attn()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = load_model(args.model_path, args.device_map, args.device)

    if args.case_set_json is None:
        result = generate_with_transformers(
            model,
            tokenizer,
            args.prompt,
            args.output_len,
            args.device,
            args.ignore_eos,
        )
        payload = base_payload(args, model)
        payload.update(
            {
                "prompt": args.prompt,
                "output_len": args.output_len,
                "prompt_token_ids": result["prompt_token_ids"],
                "generated_token_ids": result["generated_token_ids"],
                "generated_text": result["generated_text"],
                "token_sha256": result["token_sha256"],
                "text_sha256": result["text_sha256"],
                "finish_reason": result["finish_reason"],
                "generation": {
                    "do_sample": False,
                    "max_new_tokens": args.output_len,
                    "use_cache": True,
                    "ignore_eos": args.ignore_eos,
                },
            }
        )
    else:
        cases = load_case_set(args.case_set_json)
        case_results = []
        for case in cases:
            result = generate_with_transformers(
                model,
                tokenizer,
                case["prompt"],
                case["output_len"],
                args.device,
                case["ignore_eos"],
            )
            case_results.append(case_payload(case, result))
        payload = base_payload(args, model)
        payload.update(
            {
                "schema": 2,
                "report_type": "deepseek-v2-lite-ep2-hf-greedy-case-set",
                "case_set_json": str(args.case_set_json),
                "case_count": len(case_results),
                "cases": case_results,
            }
        )
    text = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.out == "-":
        print(text)
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
