#!/usr/bin/env python3
"""Generate the HuggingFace bf16 logprob golden for the Qwen3.5 gate.

The Rust gate replays these fixed token sequences through pegainfer with
teacher-forced decode and compares top-K logprobs against this stored HF oracle.
For Qwen3.5 the HF oracle follows the same incremental shape: prefill the prompt
with `use_cache=True`, then feed one fixed decode token at a time through
`past_key_values`.

    uv run --no-project python tools/accuracy/dump_qwen35_hf_golden.py \
        --model-path /data/models/Qwen3.5-4B \
        --out test_data/qwen35-4b-hf-golden.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}

SEED = 0x_5EED_3535
NUM_SEQS = 12
MIN_PROMPT_LEN = 1
MAX_PROMPT_LEN = 128
DECODE_TOKENS = 8
VOCAB_CEILING = 100_000
TOP_K = 64


# Keep in sync with `fixture_size_name` in
# pegainfer-qwen35/src/executor/hf_golden_gate.rs (the size-key geometry table).
SIZE_NAMES = {
    (1024, 24): "Qwen3.5-0.8B",
    (2048, 24): "Qwen3.5-2B",
    (2560, 32): "Qwen3.5-4B",
    (4096, 32): "Qwen3.5-9B",
    (5120, 64): "Qwen3.5-27B",
}


def model_name_from_config(model_path: Path) -> str:
    config = json.loads((model_path / "config.json").read_text())
    text = config.get("text_config", config)
    key = (text["hidden_size"], text["num_hidden_layers"])
    if key not in SIZE_NAMES:
        raise SystemExit(
            f"no size-name mapping for hidden/layers {key}; extend SIZE_NAMES"
        )
    return SIZE_NAMES[key]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_revision(path: Path) -> str:
    metadata = path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if metadata.exists():
        first = metadata.read_text().splitlines()[0].strip()
        if first:
            return first

    try:
        if (path / ".git").exists():
            out = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if out:
                return out
    except Exception:
        pass

    parts = path.parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def load_model(model_path: str, dtype: str, device_map: str):
    kwargs = {"trust_remote_code": True, "torch_dtype": DTYPES[dtype]}
    if device_map == "none":
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device_map, **kwargs
        )
    model.eval()
    return model


def input_device(model) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cuda"


def parse_prompt_lens(value: str) -> list[int]:
    lens = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        length = int(part)
        if length <= 0:
            raise argparse.ArgumentTypeError("prompt lengths must be positive")
        lens.append(length)
    if not lens:
        raise argparse.ArgumentTypeError("at least one prompt length is required")
    return lens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--out",
        default=None,
        help="output path; defaults to test_data/qwen35-{size}-hf-golden.safetensors "
        "derived from the model config (the only names the gate looks up)",
    )
    parser.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="'none' for single-GPU, 'auto' to shard larger models",
    )
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-seqs", type=int, default=NUM_SEQS)
    parser.add_argument("--min-prompt-len", type=int, default=MIN_PROMPT_LEN)
    parser.add_argument("--max-prompt-len", type=int, default=MAX_PROMPT_LEN)
    parser.add_argument(
        "--prompt-lens",
        type=parse_prompt_lens,
        default=None,
        help="comma-separated fixed prompt lengths; overrides --num-seqs/min/max",
    )
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--vocab-ceiling", type=int, default=VOCAB_CEILING)
    args = parser.parse_args()

    if args.num_seqs <= 0:
        parser.error("--num-seqs must be positive")
    if args.min_prompt_len <= 0:
        parser.error("--min-prompt-len must be positive")
    if args.max_prompt_len < args.min_prompt_len:
        parser.error("--max-prompt-len must be >= --min-prompt-len")
    if args.decode_tokens <= 0:
        parser.error("--decode-tokens must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.vocab_ceiling <= 1:
        parser.error("--vocab-ceiling must be greater than 1")

    size_key = (
        model_name_from_config(Path(args.model_path)).removeprefix("Qwen3.5-").lower()
    )
    gate_names = {
        f"qwen35-{size_key}-hf-golden.safetensors",
        f"qwen35-{size_key}-hf-long-golden.safetensors",
    }
    if args.out is None:
        kind = "-hf-long-golden" if args.prompt_lens else "-hf-golden"
        args.out = f"test_data/qwen35-{size_key}{kind}.safetensors"
    elif Path(args.out).name not in gate_names:
        raise SystemExit(
            f"--out basename {Path(args.out).name!r} will not be found by the gate; "
            f"expected one of {sorted(gate_names)}"
        )

    gen = torch.Generator().manual_seed(args.seed)
    prompts, decodes = [], []
    prompt_lens = args.prompt_lens
    if prompt_lens is None:
        prompt_lens = [
            int(
                torch.randint(
                    args.min_prompt_len,
                    args.max_prompt_len + 1,
                    (1,),
                    generator=gen,
                ).item()
            )
            for _ in range(args.num_seqs)
        ]
    for plen in prompt_lens:
        prompts.append(
            torch.randint(1, args.vocab_ceiling, (plen,), generator=gen).tolist()
        )
        decodes.append(
            torch.randint(
                1, args.vocab_ceiling, (args.decode_tokens,), generator=gen
            ).tolist()
        )

    model = load_model(args.model_path, args.dtype, args.device_map)
    if args.vocab_ceiling > model.config.vocab_size:
        parser.error(
            f"--vocab-ceiling ({args.vocab_ceiling}) cannot exceed "
            f"model vocab_size ({model.config.vocab_size})"
        )
    dev = input_device(model)

    prompt_flat: list[int] = []
    prompt_lens: list[int] = []
    ids_all, lp_all = [], []
    for prompt, decode in zip(prompts, decodes):
        prompt_lens.append(len(prompt))
        prompt_flat.extend(prompt)

        input_ids = torch.tensor([prompt], dtype=torch.long, device=dev)
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=True)
        past_key_values = out.past_key_values
        logits_steps = [out.logits[0, -1].float()]

        for token in decode:
            input_ids = torch.tensor([[token]], dtype=torch.long, device=dev)
            with torch.inference_mode():
                out = model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = out.past_key_values
            logits_steps.append(out.logits[0, -1].float())

        logprobs = F.log_softmax(torch.stack(logits_steps), dim=-1)
        ids_seq, lp_seq = [], []
        for pos in range(args.decode_tokens + 1):
            vals, idx = torch.topk(logprobs[pos], args.top_k)
            ids_seq.append(idx.tolist())
            lp_seq.append(vals.tolist())
        ids_all.append(ids_seq)
        lp_all.append(lp_seq)

    tensors = {
        "prompt_tokens": torch.tensor(prompt_flat, dtype=torch.int32),
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.int32),
        "decode_tokens": torch.tensor(decodes, dtype=torch.int32),
        "topk_ids": torch.tensor(ids_all, dtype=torch.int32),
        "topk_logprobs": torch.tensor(lp_all, dtype=torch.float32),
    }
    model_path = Path(args.model_path)
    model_revision = args.model_revision or infer_revision(model_path)
    tokenizer_revision = args.tokenizer_revision or model_revision
    meta = {
        "fixture_kind": "hf-logits-golden",
        "model": model_name_from_config(model_path),
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "config_sha256": sha256_file(model_path / "config.json"),
        "dtype": args.dtype,
        "backend": "HuggingFace AutoModelForCausalLM eval, use_cache=True",
        "forward_path": "prompt prefill, then one-token teacher-forced decode through past_key_values",
        "seed": str(args.seed),
        "top_k": str(args.top_k),
        "decode_tokens": str(args.decode_tokens),
        "num_seqs": str(len(prompts)),
        "prompt_len_min": str(min(prompt_lens)),
        "prompt_len_max": str(max(prompt_lens)),
        "prompt_lens": ",".join(str(length) for length in prompt_lens),
        "vocab_ceiling": str(args.vocab_ceiling),
        "margin_tol": "0.20",
        "mean_tol": "0.06",
        "p99_tol": "0.20",
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    print(
        f"wrote {out}: {len(prompts)} sequences, "
        f"{len(prompts) * (args.decode_tokens + 1)} positions, top{args.top_k}, {args.dtype}, "
        f"prompt_lens={prompt_lens}, "
        f"model_revision={model_revision}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
