#!/usr/bin/env python3
"""Dump HF Higgs/Qwen3 prefill stage snapshots for the one-step prompt.

The filename is kept for compatibility with earlier layer-0 workflows, but the
tool supports any decoder layer via --layer-idx.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from safetensors.torch import save_file

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from dump_higgs_one_step_golden import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPTS,
    DEFAULT_REVISION,
    HiggsTokenizerAdapter,
    load_backbone,
    load_sglang_omni_reference,
    load_tokenizer,
    sha256_file,
)


def last_token(hidden: torch.Tensor, row_idx: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
    return hidden[row_idx, prompt_lens - 1, :].detach().contiguous().clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sglang-omni-src", default="")
    parser.add_argument("--require-sglang-omni-source", action="store_true")
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
    reference = load_sglang_omni_reference(
        args.sglang_omni_src,
        required=args.require_sglang_omni_source,
    )
    adapter_cls = reference.get("tokenizer_adapter_cls", HiggsTokenizerAdapter)
    adapter = load_tokenizer(snapshot_dir, adapter_cls)
    prompt_ids = [adapter.build_prompt(prompt) for prompt in prompts]
    max_len = max(len(ids) for ids in prompt_ids)
    pad_id = int(text_cfg.get("eos_token_id") or 151643)
    input_ids = torch.full((len(prompt_ids), max_len), pad_id, dtype=torch.long, device=args.device)
    attention_mask = torch.zeros((len(prompt_ids), max_len), dtype=torch.long, device=args.device)
    for row, ids in enumerate(prompt_ids):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=args.device)
        attention_mask[row, : len(ids)] = 1
    prompt_lens = torch.tensor([len(ids) for ids in prompt_ids], dtype=torch.int64)
    prompt_lens_device = prompt_lens.to(args.device)
    row_idx = torch.arange(len(prompt_ids), device=args.device)

    backbone = load_backbone(model_file, text_cfg, args.device)
    if args.layer_idx < 0 or args.layer_idx >= len(backbone.layers):
        raise ValueError(f"--layer-idx {args.layer_idx} out of range for {len(backbone.layers)} layers")
    layer = backbone.layers[args.layer_idx]
    stage_prefix = f"layer{args.layer_idx}"
    stages: dict[str, torch.Tensor] = {}
    q_norm_all: torch.Tensor | None = None
    k_norm_all: torch.Tensor | None = None

    def capture(name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            stages[name] = last_token(tensor, row_idx, prompt_lens_device)

        return hook

    def capture_pre(name: str):
        def hook(_module, inputs):
            stages[name] = last_token(inputs[0], row_idx, prompt_lens_device)

        return hook

    def capture_q_norm(_module, _inputs, output):
        nonlocal q_norm_all
        q_norm_all = output.detach().contiguous().clone()

    def capture_k_norm(_module, _inputs, output):
        nonlocal k_norm_all
        k_norm_all = output.detach().contiguous().clone()

    hooks = [
        layer.register_forward_pre_hook(capture_pre(f"{stage_prefix}.input_hidden.bf16")),
        layer.input_layernorm.register_forward_hook(capture(f"{stage_prefix}.input_norm.bf16")),
        layer.self_attn.q_proj.register_forward_hook(capture(f"{stage_prefix}.q_proj.bf16")),
        layer.self_attn.k_proj.register_forward_hook(capture(f"{stage_prefix}.k_proj.bf16")),
        layer.self_attn.v_proj.register_forward_hook(capture(f"{stage_prefix}.v_proj.bf16")),
        layer.self_attn.q_norm.register_forward_hook(capture_q_norm),
        layer.self_attn.k_norm.register_forward_hook(capture_k_norm),
        layer.self_attn.o_proj.register_forward_pre_hook(capture_pre(f"{stage_prefix}.attn_output.bf16")),
        layer.self_attn.o_proj.register_forward_hook(capture(f"{stage_prefix}.o_proj.bf16")),
        layer.post_attention_layernorm.register_forward_hook(capture(f"{stage_prefix}.post_attn_norm.bf16")),
        layer.mlp.gate_proj.register_forward_hook(capture(f"{stage_prefix}.gate_proj.bf16")),
        layer.mlp.up_proj.register_forward_hook(capture(f"{stage_prefix}.up_proj.bf16")),
        layer.mlp.down_proj.register_forward_pre_hook(capture_pre(f"{stage_prefix}.silu_mul.bf16")),
        layer.mlp.down_proj.register_forward_hook(capture(f"{stage_prefix}.down_proj.bf16")),
        layer.register_forward_hook(capture(f"{stage_prefix}.output_hidden.bf16")),
    ]
    try:
        with torch.inference_mode():
            backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for hook in hooks:
            hook.remove()

    if q_norm_all is None or k_norm_all is None:
        raise RuntimeError("missing q/k norm snapshots")
    position_ids = torch.arange(max_len, device=args.device).unsqueeze(0)

    def norm_to_bhsd(name: str, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise RuntimeError(f"{name} expected rank-4 q/k norm output, got {tuple(tensor.shape)}")
        if tensor.shape[1] == max_len:
            return tensor.transpose(1, 2).contiguous()
        if tensor.shape[2] == max_len:
            return tensor.contiguous()
        raise RuntimeError(f"{name} cannot infer seq axis from shape {tuple(tensor.shape)}")

    q_norm_bhsd = norm_to_bhsd("q_norm", q_norm_all)
    k_norm_bhsd = norm_to_bhsd("k_norm", k_norm_all)
    cos, sin = backbone.rotary_emb(q_norm_bhsd, position_ids)
    stages[f"{stage_prefix}.q_norm.bf16"] = last_token(
        q_norm_bhsd.transpose(1, 2).reshape(len(prompt_ids), max_len, -1),
        row_idx,
        prompt_lens_device,
    )
    stages[f"{stage_prefix}.k_norm.bf16"] = last_token(
        k_norm_bhsd.transpose(1, 2).reshape(len(prompt_ids), max_len, -1),
        row_idx,
        prompt_lens_device,
    )
    q_rope, k_rope = apply_rotary_pos_emb(q_norm_bhsd, k_norm_bhsd, cos, sin)
    stages[f"{stage_prefix}.q_norm_rope.bf16"] = last_token(
        q_rope.transpose(1, 2).reshape(len(prompt_ids), max_len, -1),
        row_idx,
        prompt_lens_device,
    )
    stages[f"{stage_prefix}.k_norm_rope.bf16"] = last_token(
        k_rope.transpose(1, 2).reshape(len(prompt_ids), max_len, -1),
        row_idx,
        prompt_lens_device,
    )

    expected = [
        f"{stage_prefix}.input_hidden.bf16",
        f"{stage_prefix}.input_norm.bf16",
        f"{stage_prefix}.q_proj.bf16",
        f"{stage_prefix}.k_proj.bf16",
        f"{stage_prefix}.v_proj.bf16",
        f"{stage_prefix}.q_norm.bf16",
        f"{stage_prefix}.k_norm.bf16",
        f"{stage_prefix}.q_norm_rope.bf16",
        f"{stage_prefix}.k_norm_rope.bf16",
        f"{stage_prefix}.attn_output.bf16",
        f"{stage_prefix}.o_proj.bf16",
        f"{stage_prefix}.post_attn_norm.bf16",
        f"{stage_prefix}.gate_proj.bf16",
        f"{stage_prefix}.up_proj.bf16",
        f"{stage_prefix}.silu_mul.bf16",
        f"{stage_prefix}.down_proj.bf16",
        f"{stage_prefix}.output_hidden.bf16",
    ]
    missing = [name for name in expected if name not in stages]
    if missing:
        raise RuntimeError(f"missing stage snapshots: {missing}")

    tensors = {
        "prompt.input_ids_padded": input_ids.cpu().to(torch.int64),
        "prompt.attention_mask": attention_mask.cpu().to(torch.int64),
        "prompt.lengths": prompt_lens.cpu(),
    }
    for name in expected:
        tensors[name] = stages[name].cpu().to(torch.bfloat16)

    metadata = {
        "fixture_kind": "higgs-layer-stage-golden",
        "schema_version": "1",
        "layer_idx": str(args.layer_idx),
        "model_id": args.model_id,
        "model_revision": args.revision,
        "reference": "Transformers Qwen3Model per-layer module hooks",
        "sglang_omni_source_dir": reference.get("source_dir", ""),
        "sglang_omni_source_commit": reference.get("source_commit", ""),
        "prompt_count": str(len(prompts)),
        "prompts_json": json.dumps(prompts, ensure_ascii=False),
        "config_sha256": sha256_file(config_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_path),
        "model_index_sha256": sha256_file(index_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=metadata)
    print(f"wrote {out_path} size={out_path.stat().st_size}")
    print(f"stages {len(expected)}")


if __name__ == "__main__":
    main()
