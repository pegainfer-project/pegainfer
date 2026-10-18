#!/usr/bin/env python3
"""Generate the HuggingFace bf16 logprob golden for the Qwen3 logits gate.

The size is read from config.json, so one script serves every size.

The gate (`pegainfer-qwen3/tests/hf_golden_gate.rs`) compares pegainfer's
logprobs against HF *without* running HF at test time and *without* binding to
one GPU's exact bit pattern. So we precompute, once, on the GPU:

  * a seed-pinned set of fixed token sequences (`prompt + teacher-forced tail`),
  * HF's top-K next-token logprobs at every evaluated position.

The Rust gate replays the *same fixed sequences* through pegainfer (prefill +
teacher-forced decode) and asserts its logprobs land within a bf16 tolerance of
this golden — argmax must match HF wherever HF has a clear (> a few ULP) winner,
logprobs within the bf16 noise floor.

bf16 (not fp32) on purpose: it is the same precision regime as pegainfer, so the
comparison is apples-to-apples, and it runs on the GPU — `device_map=auto` scales
the same script to the large models. fp32 only mattered for the one-time tie
*adjudication* (compare_qwen3_4b_hf_logprobs.py --dtype float32); the gate's
margin threshold makes it unnecessary here.

Output is safetensors, not JSON: it is machine-only numeric data, nobody reads it.

    uv run --no-project python tools/accuracy/dump_qwen3_hf_golden.py \
        --model-path /data/models/Qwen3-14B \
        --out test_data/qwen3-14b-hf-golden.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}

SEED = 0x_5EED_604D
# Breadth of the per-position comparison: 48 seqs × (DECODE_TOKENS + 1) = 816
# evaluated positions. The bs=1 gate pass scores every one (one request's KV at a
# time, so it scales cheaply); larger only lengthens the ~25s run for diminishing
# coverage, since the mean/p99 the gate asserts are already stable by here.
NUM_SEQS = 48
MIN_PROMPT_LEN = 1
# Up to 16 KV blocks (block_size 16): spans sub-page to multi-block prompts, so
# the gate exercises multi-block KV indexing and cross-block attention, not just
# single-block prompts. Not long-context: 256 does not cross the 4096 RoPE cache.
MAX_PROMPT_LEN = 256
# Teacher-forced tail; gate evaluates DECODE_TOKENS + 1 positions. Long enough to
# exercise the decode path past the first step (KV append, decode-step indexing),
# not just prefill + one token.
DECODE_TOKENS = 16
VOCAB_CEILING = 100_000  # clear of high-id special tokens, matches the gate
TOP_K = 64

# (hidden_size, num_hidden_layers) -> fixture size token. Keep in sync with
# `fixture_size_name` in pegainfer-qwen3/tests/hf_golden_gate.rs.
SIZE_NAMES = {
    (1024, 28): "0.6b",
    (2048, 28): "1.7b",
    (2560, 36): "4b",
    (4096, 36): "8b",
    (5120, 40): "14b",
    (5120, 64): "32b",
}


def size_from_config(model_path: Path) -> str:
    config = json.loads((model_path / "config.json").read_text())
    text = config.get("text_config", config)
    key = (text["hidden_size"], text["num_hidden_layers"])
    if key not in SIZE_NAMES:
        raise SystemExit(f"no size mapping for hidden/layers {key}; extend SIZE_NAMES")
    return SIZE_NAMES[key]


def load_model(model_path: str, dtype: str, device_map: str):
    kwargs = {"trust_remote_code": True, "torch_dtype": DTYPES[dtype]}
    if device_map == "none":
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, **kwargs)
    model.eval()
    return model


def input_device(model) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cuda"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    parser.add_argument("--device-map", default="auto", help="'none' for single-GPU, 'auto' to shard big models")
    args = parser.parse_args()

    size = size_from_config(Path(args.model_path))
    expected = f"qwen3-{size}-hf-golden.safetensors"
    if Path(args.out).name != expected:
        raise SystemExit(
            f"--out basename {Path(args.out).name!r} will not be found by the gate; "
            f"expected {expected!r} for this Qwen3-{size} model"
        )

    gen = torch.Generator().manual_seed(SEED)
    prompts, decodes = [], []
    for _ in range(NUM_SEQS):
        plen = int(torch.randint(MIN_PROMPT_LEN, MAX_PROMPT_LEN + 1, (1,), generator=gen).item())
        prompts.append(torch.randint(1, VOCAB_CEILING, (plen,), generator=gen).tolist())
        decodes.append(torch.randint(1, VOCAB_CEILING, (DECODE_TOKENS,), generator=gen).tolist())

    model = load_model(args.model_path, args.dtype, args.device_map)
    dev = input_device(model)

    prompt_flat: list[int] = []
    prompt_lens: list[int] = []
    ids_all, lp_all = [], []  # [S, D+1, K]
    for prompt, decode in zip(prompts, decodes):
        prompt_lens.append(len(prompt))
        prompt_flat.extend(prompt)
        full = prompt + decode
        input_ids = torch.tensor([full], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits = model(input_ids).logits[0].float()
        logprobs = F.log_softmax(logits, dim=-1)
        ids_seq, lp_seq = [], []
        for pos in range(len(prompt) - 1, len(prompt) + DECODE_TOKENS):  # P-1 .. P+D-1
            vals, idx = torch.topk(logprobs[pos], TOP_K)
            ids_seq.append(idx.tolist())
            lp_seq.append(vals.tolist())
        ids_all.append(ids_seq)
        lp_all.append(lp_seq)

    tensors = {
        "prompt_tokens": torch.tensor(prompt_flat, dtype=torch.int32),
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.int32),
        "decode_tokens": torch.tensor(decodes, dtype=torch.int32),  # [S, D]
        "topk_ids": torch.tensor(ids_all, dtype=torch.int32),  # [S, D+1, K]
        "topk_logprobs": torch.tensor(lp_all, dtype=torch.float32),  # [S, D+1, K]
    }
    meta = {
        "dtype": args.dtype,
        "seed": str(SEED),
        "top_k": str(TOP_K),
        "decode_tokens": str(DECODE_TOKENS),
        "num_seqs": str(NUM_SEQS),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out}: {NUM_SEQS} sequences, {NUM_SEQS * (DECODE_TOKENS + 1)} positions, top{TOP_K}, {args.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
