#!/usr/bin/env python3
"""Dump a full-model Kimi-K2 one-token logits fixture from HF remote code.

This is the parity source for PegaInfer full-model logits gates. It intentionally
loads the model from an existing local model directory and writes the rendered
prompt, input ids, raw last-token logits, top-k logits, and top-k logprobs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


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
    """Return a symlink path whose basename is safe for HF dynamic modules."""
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


def force_attention_implementation(config: Any, attn_implementation: str | None) -> Any:
    if not attn_implementation:
        return config
    for obj in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "vision_config", None),
    ):
        if obj is None:
            continue
        setattr(obj, "_attn_implementation", attn_implementation)
        setattr(obj, "_attn_implementation_internal", attn_implementation)
    return config


def force_text_only_outer_config(config: Any) -> bool:
    """Disable the Kimi multimodal shell while keeping the language model."""
    vision = getattr(config, "vision_config", None)
    if vision is None:
        return False
    for attr in ("vt_num_hidden_layers", "num_hidden_layers"):
        if hasattr(vision, attr):
            setattr(vision, attr, 0)
    if hasattr(vision, "mm_projector_type"):
        setattr(vision, "mm_projector_type", "identity")
    return True


def patch_kimi_remote_code_init_weights(model_load_path: Path) -> str:
    """Make Kimi remote-code init tolerate compressed-tensors Linear modules.

    Transformers 4.56.2 always calls `_initialize_missing_keys` after building
    the model. Kimi's remote `_init_weights` treats every `nn.Linear` as having
    a dense `.weight`, while compressed-tensors routed experts replace that with
    `weight_packed`/`weight_scale`/`weight_shape`. The checkpoint still owns
    those tensors, so the safe behavior for these modules is to skip dense
    fallback initialization instead of crashing during fixture generation.
    """
    module_name = f"transformers_modules.{model_load_path.name}.modeling_deepseek"
    module = importlib.import_module(module_name)
    base_cls = module.DeepseekV3PreTrainedModel
    if getattr(base_cls, "_pegainfer_compressed_init_patch", False):
        return module_name

    original_init_weights = base_cls._init_weights

    def patched_init_weights(self: Any, layer: torch.nn.Module) -> None:
        if isinstance(layer, torch.nn.Linear) and getattr(layer, "weight", None) is None:
            bias = getattr(layer, "bias", None)
            if bias is not None:
                bias.data.zero_()
            return
        original_init_weights(self, layer)

    base_cls._init_weights = patched_init_weights
    base_cls._pegainfer_compressed_init_patch = True
    return module_name


def patch_accelerate_dispatch_empty_cache() -> bool:
    """Avoid per-tensor `torch.cuda.empty_cache()` during huge Kimi dispatch."""
    try:
        import accelerate.utils.memory as accelerate_memory
        import accelerate.utils.modeling as accelerate_modeling
    except ImportError:
        return False

    def no_clear_device_cache(*_args: Any, **_kwargs: Any) -> None:
        return None

    accelerate_memory.clear_device_cache = no_clear_device_cache
    accelerate_modeling.clear_device_cache = no_clear_device_cache
    return True


def assert_no_meta_tensors(named_tensors: Iterable[tuple[str, torch.Tensor]], kind: str) -> None:
    meta_names = [name for name, tensor in named_tensors if tensor.device.type == "meta"]
    if meta_names:
        preview = ", ".join(meta_names[:8])
        suffix = "" if len(meta_names) <= 8 else f", ... (+{len(meta_names) - 8})"
        raise RuntimeError(f"model still has meta {kind}: {preview}{suffix}")


def tensor_to_list_i64(tensor: torch.Tensor) -> list[int]:
    return [int(v) for v in tensor.to(torch.int64).cpu().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt-json", type=Path)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--thinking", type=parse_bool, default=True)
    parser.add_argument("--preserve-thinking", type=parse_bool, default=False)
    parser.add_argument("--add-generation-prompt", type=parse_bool, default=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument(
        "--load-outer-vision",
        action="store_true",
        help="keep Kimi-K2.5 vision/projector modules when loading the HF model",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_load_path = transformers_safe_model_path(args.model_path, args.out_dir)

    local_files_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(
        model_load_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    payload = load_prompt_payload(args.prompt_json)
    rendered_prompt = render_prompt(
        tokenizer,
        payload,
        thinking=args.thinking,
        preserve_thinking=args.preserve_thinking,
        add_generation_prompt=args.add_generation_prompt,
    )
    encoded = tokenizer(rendered_prompt, return_tensors="pt")
    input_ids_cpu = encoded["input_ids"].to(torch.int64).cpu()

    config = AutoConfig.from_pretrained(
        model_load_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    text_only_outer = False
    if not args.load_outer_vision:
        text_only_outer = force_text_only_outer_config(config)
    config = force_attention_implementation(config, args.attn_implementation)
    patched_remote_module = patch_kimi_remote_code_init_weights(model_load_path)
    accelerate_empty_cache_patched = patch_accelerate_dispatch_empty_cache()

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": args.device_map,
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
        "config": config,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_load_path, **model_kwargs)
    assert_no_meta_tensors(model.named_parameters(), "parameters")
    assert_no_meta_tensors(model.named_buffers(), "buffers")
    model.eval()

    input_ids = input_ids_cpu.to(next(model.parameters()).device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1].detach().float().cpu().contiguous()

    vocab_size = int(logits.numel())
    top_k = min(args.top_k, vocab_size)
    topk_logits, topk_ids = torch.topk(logits, k=top_k)
    logprobs = torch.log_softmax(logits, dim=-1)
    topk_logprobs = logprobs.index_select(0, topk_ids).contiguous()
    argmax_id = torch.argmax(logits).reshape(1).to(torch.int64)

    save_file(
        {
            "input_ids": input_ids_cpu.reshape(-1).contiguous(),
            "logits_f32": logits,
            "topk_ids": topk_ids.to(torch.int64).contiguous(),
            "topk_logits_f32": topk_logits.float().contiguous(),
            "topk_logprobs_f32": topk_logprobs.float().contiguous(),
            "argmax_id": argmax_id.cpu().contiguous(),
        },
        str(args.out_dir / "reference.safetensors"),
    )

    metadata = {
        "engine": "hf_remote_code",
        "model_path": str(args.model_path),
        "transformers_model_path": str(model_load_path),
        "model_class": model.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "messages": payload["messages"],
        "rendered_prompt": rendered_prompt,
        "thinking": args.thinking,
        "preserve_thinking": args.preserve_thinking,
        "add_generation_prompt": args.add_generation_prompt,
        "seq_len": int(input_ids_cpu.numel()),
        "input_ids": tensor_to_list_i64(input_ids_cpu.reshape(-1)),
        "top_k": top_k,
        "vocab_size": vocab_size,
        "dtype": "bf16_forward/fp32_dump",
        "device_map": args.device_map,
        "text_only_outer_vision_disabled": text_only_outer,
        "patched_remote_module": patched_remote_module,
        "accelerate_dispatch_empty_cache_patched": accelerate_empty_cache_patched,
        "local_files_only": local_files_only,
        "config_sha256": sha256_file(args.model_path / "config.json"),
        "tokenizer_config_sha256": sha256_file(args.model_path / "tokenizer_config.json"),
        "chat_template_sha256": sha256_file(args.model_path / "chat_template.jinja"),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    with (args.out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with (args.out_dir / "prompt.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "messages": payload["messages"],
                "rendered_prompt": rendered_prompt,
                "input_ids": metadata["input_ids"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "seq_len": metadata["seq_len"],
                "vocab_size": vocab_size,
                "argmax_id": int(argmax_id.item()),
                "top1_logit": float(topk_logits[0].item()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
