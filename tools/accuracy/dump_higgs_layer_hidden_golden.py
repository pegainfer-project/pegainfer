#!/usr/bin/env python3
"""Dump HF Higgs/Qwen3 per-layer hidden snapshots for the one-step prompt."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from safetensors.torch import save_file

from dump_higgs_one_step_golden import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPTS,
    DEFAULT_REVISION,
    load_backbone,
    load_tokenizer,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    snapshot_dir = Path(args.snapshot_dir)
    config_path = snapshot_dir / "config.json"
    tokenizer_path = snapshot_dir / "tokenizer.json"
    model_file = snapshot_dir / "model.safetensors"
    index_path = snapshot_dir / "model.safetensors.index.json"
    for path in (config_path, tokenizer_path, model_file, index_path):
        if not path.exists():
            raise FileNotFoundError(path)

    config = json.load(open(config_path))
    text_cfg = dict(config["text_config"])
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    adapter = load_tokenizer(snapshot_dir)
    prompt_ids = [adapter.build_prompt(prompt) for prompt in prompts]
    max_len = max(len(ids) for ids in prompt_ids)
    pad_id = int(text_cfg.get("eos_token_id") or 151643)
    input_ids = torch.full(
        (len(prompt_ids), max_len), pad_id, dtype=torch.long, device=args.device
    )
    attention_mask = torch.zeros(
        (len(prompt_ids), max_len), dtype=torch.long, device=args.device
    )
    for row, ids in enumerate(prompt_ids):
        input_ids[row, : len(ids)] = torch.tensor(
            ids, dtype=torch.long, device=args.device
        )
        attention_mask[row, : len(ids)] = 1
    prompt_lens = torch.tensor([len(ids) for ids in prompt_ids], dtype=torch.int64)
    prompt_lens_device = prompt_lens.to(args.device)
    row_idx = torch.arange(len(prompt_ids), device=args.device)

    backbone = load_backbone(model_file, text_cfg, args.device)
    embedding_hidden: torch.Tensor | None = None
    layer_snapshots: list[torch.Tensor | None] = [None] * len(backbone.layers)

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            layer_snapshots[layer_idx] = (
                hidden[row_idx, prompt_lens_device - 1, :].detach().contiguous().clone()
            )

        return hook

    def embedding_hook(_module, _inputs, output):
        nonlocal embedding_hidden
        embedding_hidden = output[row_idx, prompt_lens_device - 1, :].detach().contiguous().clone()

    hooks = [backbone.embed_tokens.register_forward_hook(embedding_hook)]
    hooks.extend(layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(backbone.layers))
    try:
        with torch.inference_mode():
            out = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for hook in hooks:
            hook.remove()

    final_hidden = out.last_hidden_state[
        row_idx, prompt_lens_device - 1, :
    ].detach().contiguous()
    if embedding_hidden is None:
        raise RuntimeError("missing embedding snapshot")
    tensors = {
        "prompt.input_ids_padded": input_ids.cpu().to(torch.int64),
        "prompt.attention_mask": attention_mask.cpu().to(torch.int64),
        "prompt.lengths": prompt_lens.cpu(),
        "embedding.last_hidden.bf16": embedding_hidden.cpu().to(torch.bfloat16),
        "final_hidden.bf16": final_hidden.cpu().to(torch.bfloat16),
    }
    for layer_idx, hidden in enumerate(layer_snapshots):
        if hidden is None:
            raise RuntimeError(f"missing layer snapshot {layer_idx}")
        tensors[f"layer.{layer_idx:02}.last_hidden.bf16"] = hidden.cpu().to(torch.bfloat16)

    metadata = {
        "fixture_kind": "higgs-prefill-layer-hidden-golden",
        "schema_version": "1",
        "model_id": args.model_id,
        "model_revision": args.revision,
        "reference": "Transformers Qwen3Model forward hooks after each decoder layer plus final model norm",
        "prompt_count": str(len(prompts)),
        "prompts_json": json.dumps(prompts, ensure_ascii=False),
        "hidden_size": str(text_cfg["hidden_size"]),
        "num_hidden_layers": str(text_cfg["num_hidden_layers"]),
        "config_sha256": sha256_file(config_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_path),
        "model_index_sha256": sha256_file(index_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name(0)
        if args.device.startswith("cuda")
        else args.device,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=metadata)
    print(f"wrote {out_path} size={out_path.stat().st_size}")
    print(f"layers {len(layer_snapshots)}")


if __name__ == "__main__":
    main()
