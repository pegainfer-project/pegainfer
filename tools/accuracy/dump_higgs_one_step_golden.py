#!/usr/bin/env python3
"""Dump a Higgs Audio one-step audio-logits golden.

This generator intentionally uses the SGLang-Omni Higgs prompt/head contract
without importing the full SGLang server stack. The prompt builder mirrors
`sglang_omni.models.higgs_tts.text_tokenizer.HiggsTokenizerAdapter`; the fused
audio head mirrors `HiggsFusedMultiTextHead.generate`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RotaryEmbedding

AUDIO_PLACEHOLDER_ID = -100
REQUIRED_SPECIALS = ("<|tts|>", "<|ref_audio|>", "<|text|>", "<|audio|>")
DEFAULT_MODEL_ID = "bosonai/higgs-tts-3-4b"
DEFAULT_REVISION = "7556c17e05201fccd9c8cc120bc216dcc7b5d561"
DEFAULT_PROMPTS = ("Hello from PegaInfer.",)
TRACE_MODULE_SUFFIXES = (
    "input_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "self_attn.o_proj",
    "post_attention_layernorm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def git_short_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def load_sglang_omni_reference(
    source_dir: str,
    *,
    required: bool,
) -> dict[str, Any]:
    if not source_dir:
        if required:
            raise ValueError("--require-sglang-omni-source requires --sglang-omni-src")
        return {}

    src = Path(source_dir).resolve()
    if not (src / "sglang_omni/models/higgs_tts/text_tokenizer.py").exists():
        raise FileNotFoundError(f"SGLang-Omni source missing Higgs tokenizer: {src}")
    sys.path.insert(0, str(src))
    try:
        tokenizer_mod = importlib.import_module(
            "sglang_omni.models.higgs_tts.text_tokenizer"
        )
        modeling_mod = importlib.import_module("sglang_omni.models.higgs_tts.modeling")
    except Exception:
        if required:
            raise
        return {}

    return {
        "source_dir": str(src),
        "source_commit": git_short_commit(src),
        "tokenizer_adapter_cls": tokenizer_mod.HiggsTokenizerAdapter,
        "fused_head_cls": modeling_mod.HiggsFusedMultiTextHead,
    }


class HiggsTokenizerAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        vocab = dict(tokenizer.get_added_vocab())
        missing = [t for t in REQUIRED_SPECIALS if t not in vocab]
        if missing:
            raise ValueError(f"Tokenizer is missing Higgs TTS specials: {missing}")
        self.tts_id = int(vocab["<|tts|>"])
        self.ref_audio_id = int(vocab["<|ref_audio|>"])
        self.text_id = int(vocab["<|text|>"])
        self.audio_id = int(vocab["<|audio|>"])
        self.ref_text_id = vocab.get("<|ref_text|>")

    def build_prompt(
        self,
        prompt_text: str,
        *,
        num_ref_tokens: int = 0,
        reference_text: str | None = None,
    ) -> list[int]:
        if num_ref_tokens < 0:
            raise ValueError(f"num_ref_tokens must be >= 0, got {num_ref_tokens}")
        ids: list[int] = [self.tts_id]
        if reference_text and num_ref_tokens > 0 and self.ref_text_id is not None:
            ids.append(int(self.ref_text_id))
            ids.extend(self._tok.encode(reference_text, add_special_tokens=False))
        if num_ref_tokens > 0:
            ids.append(self.ref_audio_id)
            ids.extend([AUDIO_PLACEHOLDER_ID] * num_ref_tokens)
        ids.append(self.text_id)
        ids.extend(self._tok.encode(prompt_text, add_special_tokens=False))
        ids.append(self.audio_id)
        return [int(x) for x in ids]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def remap_backbone_key(src: str) -> str | None:
    if src == "tied.embedding.text_embedding.weight":
        return "embed_tokens.weight"
    if src.startswith("body.layers."):
        return src.removeprefix("body.")
    if src.startswith("body.norm."):
        return src.removeprefix("body.")
    return None


def load_backbone(model_file: Path, text_cfg: dict[str, Any], device: str) -> Qwen3Model:
    cfg = Qwen3Config(**text_cfg)
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3Model(cfg)
    model.to_empty(device=device)
    # to_empty() does not populate non-persistent RoPE buffers created on the
    # meta device, so rebuild rotary_emb on the real device before loading params.
    model.rotary_emb = Qwen3RotaryEmbedding(cfg, device=device)
    model.to(dtype=torch.bfloat16)
    model.eval()
    params = dict(model.named_parameters())
    loaded: set[str] = set()
    with safe_open(str(model_file), framework="pt", device="cpu") as f:
        for src in f.keys():
            dst = remap_backbone_key(src)
            if dst is None:
                continue
            if dst not in params:
                raise KeyError(f"remapped key {src} -> {dst}, but Qwen3Model has no such parameter")
            p = params[dst]
            t = f.get_tensor(src)
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(f"shape mismatch {src}->{dst}: ckpt {tuple(t.shape)} vs model {tuple(p.shape)}")
            p.data.copy_(t.to(device=device, dtype=p.dtype, non_blocking=False))
            loaded.add(dst)
    missing = sorted(set(params) - loaded)
    if missing:
        raise RuntimeError(f"missing {len(missing)} backbone parameters, first: {missing[:8]}")
    return model


def load_modality_head_weight(model_file: Path, device: str) -> torch.Tensor:
    key = "tied.embedding.modality_embeddings.0.embedding.weight"
    with safe_open(str(model_file), framework="pt", device="cpu") as f:
        if key not in f.keys():
            raise KeyError(f"missing fused modality embedding/head weight {key}")
        weight = f.get_tensor(key)
    if tuple(weight.shape) != (8208, 2560):
        raise ValueError(f"unexpected modality head weight shape {tuple(weight.shape)}")
    return weight.to(device=device, dtype=torch.bfloat16)


def load_tokenizer(snapshot_dir: Path, adapter_cls: type[Any]) -> Any:
    raw = Tokenizer.from_file(str(snapshot_dir / "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw)
    return adapter_cls(tokenizer)


def trace_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(tensor):
        return tensor.detach().cpu().to(torch.bfloat16).contiguous()
    return tensor.detach().cpu().contiguous()


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def module_trace_name(module_name: str) -> str | None:
    if not module_name.startswith("layers."):
        return None
    parts = module_name.split(".", 2)
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    layer_idx = int(parts[1])
    suffix = parts[2]
    if suffix not in TRACE_MODULE_SUFFIXES:
        return None
    return f"layer.{layer_idx:02}.{suffix}.output.bf16"


def register_trace_hooks(model: Qwen3Model, tensors: dict[str, torch.Tensor]) -> list[Any]:
    handles = []

    def make_hook(trace_name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = first_tensor(output)
            if tensor is not None:
                tensors[trace_name] = trace_tensor(tensor)

        return hook

    for module_name, module in model.named_modules():
        trace_name = module_trace_name(module_name)
        if trace_name is not None:
            handles.append(module.register_forward_hook(make_hook(trace_name)))
    return handles


def compute_modality_logits(
    last_hidden: torch.Tensor,
    modality_weight: torch.Tensor,
    audio_cfg: dict[str, Any],
    reference: dict[str, Any],
) -> torch.Tensor:
    head_cls = reference.get("fused_head_cls")
    if head_cls is None:
        logits = F.linear(last_hidden, modality_weight)
        return logits.reshape(
            last_hidden.shape[0],
            int(audio_cfg["num_codebooks"]),
            int(audio_cfg["vocab_size"]),
        )

    head = head_cls(
        num_codebooks=int(audio_cfg["num_codebooks"]),
        vocab_size=int(audio_cfg["vocab_size"]),
        hidden_size=last_hidden.shape[-1],
    ).to(device=last_hidden.device, dtype=torch.bfloat16)
    head.eval()
    with torch.no_grad():
        head.weight.copy_(modality_weight)
    return head.generate(last_hidden)


def add_hidden_state_trace(
    tensors: dict[str, torch.Tensor],
    hidden_states: tuple[torch.Tensor, ...] | None,
    prompt_lens: torch.Tensor,
) -> None:
    if hidden_states is None:
        return
    # Transformers Qwen3 returns output_hidden_states as:
    #   embedding, layer0_output, ..., layer34_output, final_norm_output
    # The raw layer35 decoder output is not present in this tuple because the
    # final item is appended after model.norm. Store that tensor as final_hidden
    # only; labeling it layer.35.* creates a false last-layer divergence.
    last_idx = len(hidden_states) - 1
    cpu_lens = prompt_lens.cpu()
    for idx, state in enumerate(hidden_states):
        if idx == last_idx:
            continue
        if idx == 0:
            sequence_name = "embedding.sequence_hidden.bf16"
            last_name = "embedding.last_hidden.bf16"
        else:
            sequence_name = f"layer.{idx - 1:02}.sequence_hidden.bf16"
            last_name = f"layer.{idx - 1:02}.last_hidden.bf16"
        tensors[sequence_name] = trace_tensor(state)
        rows = []
        for batch_idx, prompt_len in enumerate(cpu_lens.tolist()):
            rows.append(state[batch_idx, int(prompt_len) - 1, :])
        tensors[last_name] = trace_tensor(torch.stack(rows, dim=0))


def write_trace_file(
    out_path: Path,
    *,
    base_tensors: dict[str, torch.Tensor],
    trace_tensors: dict[str, torch.Tensor],
    hidden_states: tuple[torch.Tensor, ...] | None,
    prompt_lens: torch.Tensor,
    last_hidden: torch.Tensor,
    logits: torch.Tensor,
    logprobs: torch.Tensor,
    metadata: dict[str, str],
) -> int:
    tensors = dict(base_tensors)
    add_hidden_state_trace(tensors, hidden_states, prompt_lens)
    tensors.update(trace_tensors)
    tensors.update(
        {
            "audio_head.input_hidden.bf16": last_hidden.cpu().to(torch.bfloat16),
            "audio_head.flat_logits.f32": logits.reshape(logits.shape[0], -1)
            .cpu()
            .to(torch.float32),
            "audio_logprobs.f32": logprobs.cpu().to(torch.float32),
        }
    )
    trace_metadata = dict(metadata)
    trace_metadata.update(
        {
            "fixture_kind": "higgs-one-step-trace-golden",
            "schema_version": "3",
            "trace_contract": "prompt;embedding;per-layer hidden except final decoder raw hidden;per-layer module outputs;qkv/mlp stage hooks;audio-head logits/logprobs/topk/argmax",
            "trace_tensor_count": str(len(tensors)),
            "trace_module_suffixes": ";".join(TRACE_MODULE_SUFFIXES),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=trace_metadata)
    return len(tensors)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--snapshot-dir", default="")
    ap.add_argument("--out", default="test_data/higgs-one-step-audio-logits.safetensors")
    ap.add_argument(
        "--trace-out",
        default="",
        help="Optional rich trace safetensors path with prompt, per-layer, per-module, and audio-head intermediate tensors.",
    )
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--download", action="store_true")
    ap.add_argument(
        "--sglang-omni-src",
        default="",
        help="Optional SGLang-Omni source tree; imports Higgs tokenizer/head modules directly.",
    )
    ap.add_argument(
        "--require-sglang-omni-source",
        action="store_true",
        help="Fail if --sglang-omni-src cannot provide the Higgs reference modules.",
    )
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    if args.snapshot_dir:
        snapshot_dir = Path(args.snapshot_dir)
    else:
        local_dir = f"models/higgs-tts-3-4b-{args.revision}"
        if args.download or not Path(local_dir, "model.safetensors").exists():
            snapshot_download(
                repo_id=args.model_id,
                revision=args.revision,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        else:
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
                hf_hub_download(args.model_id, name, revision=args.revision)
        snapshot_dir = Path(local_dir)

    config_path = snapshot_dir / "config.json"
    tokenizer_path = snapshot_dir / "tokenizer.json"
    model_file = snapshot_dir / "model.safetensors"
    index_path = snapshot_dir / "model.safetensors.index.json"
    for p in (config_path, tokenizer_path, model_file, index_path):
        if not p.exists():
            raise FileNotFoundError(p)

    config = json.load(open(config_path))
    text_cfg = dict(config["text_config"])
    audio_cfg = dict(config["audio_encoder_config"])
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    reference = load_sglang_omni_reference(
        args.sglang_omni_src,
        required=args.require_sglang_omni_source,
    )
    adapter_cls = reference.get("tokenizer_adapter_cls", HiggsTokenizerAdapter)
    adapter = load_tokenizer(snapshot_dir, adapter_cls)
    prompt_ids = [adapter.build_prompt(p) for p in prompts]
    max_len = max(len(x) for x in prompt_ids)
    pad_id = int(text_cfg.get("eos_token_id") or 151643)
    input_ids = torch.full((len(prompt_ids), max_len), pad_id, dtype=torch.long, device=args.device)
    attention_mask = torch.zeros((len(prompt_ids), max_len), dtype=torch.long, device=args.device)
    for i, ids in enumerate(prompt_ids):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=args.device)
        attention_mask[i, : len(ids)] = 1
    prompt_lens = torch.tensor([len(x) for x in prompt_ids], dtype=torch.int64)

    backbone = load_backbone(model_file, text_cfg, args.device)
    modality_weight = load_modality_head_weight(model_file, args.device)
    trace_tensors: dict[str, torch.Tensor] = {}
    trace_hooks = register_trace_hooks(backbone, trace_tensors) if args.trace_out else []
    with torch.inference_mode():
        out = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=bool(args.trace_out),
        )
        hidden = out.last_hidden_state
        row_idx = torch.arange(len(prompts), device=args.device)
        last_hidden = hidden[row_idx, prompt_lens.to(args.device) - 1, :].contiguous()
        logits = compute_modality_logits(last_hidden, modality_weight, audio_cfg, reference)
        logprobs = torch.log_softmax(logits.to(torch.float32), dim=-1)
        top_vals, top_ids = torch.topk(logprobs, k=64, dim=-1)
        argmax_ids = torch.argmax(logits, dim=-1).to(torch.int64)
    for handle in trace_hooks:
        handle.remove()

    tensors = {
        "prompt.input_ids_padded": input_ids.cpu().to(torch.int64),
        "prompt.attention_mask": attention_mask.cpu().to(torch.int64),
        "prompt.lengths": prompt_lens.cpu(),
        "final_hidden.bf16": last_hidden.cpu().to(torch.bfloat16),
        "audio_logits.f32": logits.cpu().to(torch.float32),
        "audio_top64.ids": top_ids.cpu().to(torch.int64),
        "audio_top64.logprobs.f32": top_vals.cpu().to(torch.float32),
        "audio_argmax.ids": argmax_ids.cpu().to(torch.int64),
    }
    metadata = {
        "fixture_kind": "higgs-one-step-audio-logits-golden",
        "schema_version": "1",
        "model_id": args.model_id,
        "model_revision": args.revision,
        "reference": "SGLang-Omni Higgs prompt builder plus Transformers Qwen3 backbone plus SGLang fused modality head semantics",
        "sglang_omni_reference_files": "sglang_omni/models/higgs_tts/text_tokenizer.py;sglang_omni/models/higgs_tts/modeling.py;sglang_omni/models/higgs_tts/model.py",
        "sglang_omni_source_dir": reference.get("source_dir", ""),
        "sglang_omni_source_commit": reference.get("source_commit", ""),
        "sglang_omni_direct_imports": "text_tokenizer.py;modeling.py" if reference else "",
        "sglang_omni_full_model_imported": "false",
        "prompt_count": str(len(prompts)),
        "prompts_json": json.dumps(prompts, ensure_ascii=False),
        "num_codebooks": str(audio_cfg["num_codebooks"]),
        "codebook_vocab_size": str(audio_cfg["vocab_size"]),
        "hidden_size": str(text_cfg["hidden_size"]),
        "config_sha256": sha256_file(config_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_path),
        "model_index_sha256": sha256_file(index_path),
        "model_safetensors_size": str(model_file.stat().st_size),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        "cuda_peak_allocated": str(torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0),
        "cuda_peak_reserved": str(torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else 0),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=metadata)
    print(f"wrote {out_path} size={out_path.stat().st_size}")
    if args.trace_out:
        trace_path = Path(args.trace_out)
        trace_tensor_count = write_trace_file(
            trace_path,
            base_tensors=tensors,
            trace_tensors=trace_tensors,
            hidden_states=out.hidden_states,
            prompt_lens=prompt_lens,
            last_hidden=last_hidden,
            logits=logits,
            logprobs=logprobs,
            metadata=metadata,
        )
        print(f"wrote trace {trace_path} size={trace_path.stat().st_size} tensors={trace_tensor_count}")
    print("argmax", argmax_ids.cpu().tolist())


if __name__ == "__main__":
    main()
