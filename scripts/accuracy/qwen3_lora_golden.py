#!/usr/bin/env python3
"""Generate the PEFT LoRA golden for `tests/lora_golden_gate.rs`.

A trained adapter is unnecessary: a seed-pinned random rank-1 q/v adapter perturbs
every logit, so any application bug (transposed / missing / mis-scaled delta) shifts
logprobs past the gate tolerances. One fixture carries the fixed sequences, base +
LoRA top-K reference grids, and the adapter tensors themselves (`adapter/<name>`) —
the gate reconstructs the exact PEFT directory, nothing reproduces RNG or PEFT
versions at test time. The B matrices are calibrated into [MIN_EFFECT, MAX_EFFECT];
outside that band the script refuses to write (a null fixture would let a silently
dropped adapter pass).

    uv run --no-project python scripts/accuracy/qwen3_lora_golden.py \
        --model-path /data/models/Qwen3-4B \
        --out test_data/qwen3-4b-lora-golden.safetensors
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

SEED = 0x_10AA_604D
NUM_SEQS = 16
MIN_PROMPT_LEN = 1
MAX_PROMPT_LEN = 256
DECODE_TOKENS = 16
VOCAB_CEILING = 100_000
TOP_K = 64

RANK = 1
ALPHA = 1
TARGET_MODULES = ["q_proj", "v_proj"]
# Mean |LoRA − base| logprob over the base top-K, averaged over all evaluated positions.
# Below MIN the delta drowns in the gate tolerances; above MAX the model is perturbed
# past anything resembling a fine-tune.
TARGET_EFFECT = 0.5
MIN_EFFECT = 0.1
MAX_EFFECT = 2.0


def tensor_name(layer: int, target: str, side: str) -> str:
    return f"base_model.model.model.layers.{layer}.self_attn.{target}.{side}.weight"


def fill_lora_weights(
    model, gen: torch.Generator, b_scale: float
) -> dict[str, torch.Tensor]:
    """Seed-pinned normal init for every lora_A/lora_B; returns canonical-name → tensor."""
    out = {}
    pat = re.compile(
        r"layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_(A|B)\.default\.weight$"
    )
    for name, param in sorted(model.named_parameters()):
        m = pat.search(name)
        if m is None:
            continue
        layer, target, side = int(m.group(1)), m.group(2), m.group(3)
        w = torch.randn(param.shape, generator=gen, dtype=torch.float32)
        if side == "A":
            w /= param.shape[1] ** 0.5
        else:
            w *= b_scale
        with torch.no_grad():
            param.copy_(w.to(param.dtype))
        out[tensor_name(layer, target, f"lora_{side}")] = w.to(torch.bfloat16)
    return out


def topk_grid(model, dev, prompts, decodes):
    """[S, D+1, K] top-K ids/logprobs, plus the full logprob rows for calibration."""
    ids_all, lp_all, rows = [], [], []
    for prompt, decode in zip(prompts, decodes):
        full = prompt + decode
        input_ids = torch.tensor([full], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits = model(input_ids).logits[0].float()
        logprobs = F.log_softmax(logits, dim=-1)
        ids_seq, lp_seq = [], []
        for pos in range(len(prompt) - 1, len(prompt) + DECODE_TOKENS):
            vals, idx = torch.topk(logprobs[pos], TOP_K)
            ids_seq.append(idx.tolist())
            lp_seq.append(vals.tolist())
            rows.append(logprobs[pos].cpu())
        ids_all.append(ids_seq)
        lp_all.append(lp_seq)
    return ids_all, lp_all, rows


def mean_effect(base_ids, base_rows, lora_rows) -> float:
    deltas = []
    for ids, brow, lrow in zip(
        [i for seq in base_ids for i in seq], base_rows, lora_rows
    ):
        idx = torch.tensor(ids, dtype=torch.long)
        deltas.append((lrow[idx] - brow[idx]).abs().mean().item())
    return sum(deltas) / len(deltas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gen = torch.Generator().manual_seed(SEED)
    prompts, decodes = [], []
    for _ in range(NUM_SEQS):
        plen = int(
            torch.randint(
                MIN_PROMPT_LEN, MAX_PROMPT_LEN + 1, (1,), generator=gen
            ).item()
        )
        prompts.append(torch.randint(1, VOCAB_CEILING, (plen,), generator=gen).tolist())
        decodes.append(
            torch.randint(1, VOCAB_CEILING, (DECODE_TOKENS,), generator=gen).tolist()
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    model = get_peft_model(
        model,
        LoraConfig(
            r=RANK,
            lora_alpha=ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
        ),
    )
    dev = str(next(model.parameters()).device)

    with model.disable_adapter():
        base_ids, base_lp, base_rows = topk_grid(model, dev, prompts, decodes)

    # One proportional correction: the delta is linear in B per layer, so a
    # single rescale lands in the band unless the nonlinearity is extreme.
    b_scale = 0.05
    weight_gen = torch.Generator().manual_seed(SEED + 1)
    adapter = fill_lora_weights(model, weight_gen, b_scale)
    _, _, lora_rows = topk_grid(model, dev, prompts, decodes)
    effect = mean_effect(base_ids, base_rows, lora_rows)
    print(f"b_scale {b_scale}: mean effect {effect:.4f} nat")
    if not (MIN_EFFECT <= effect <= MAX_EFFECT):
        b_scale *= TARGET_EFFECT / effect
        weight_gen = torch.Generator().manual_seed(SEED + 1)
        adapter = fill_lora_weights(model, weight_gen, b_scale)
        _, _, lora_rows = topk_grid(model, dev, prompts, decodes)
        effect = mean_effect(base_ids, base_rows, lora_rows)
        print(f"rescaled b_scale {b_scale:.6f}: mean effect {effect:.4f} nat")
    assert MIN_EFFECT <= effect <= MAX_EFFECT, (
        f"adapter effect {effect:.4f} nat outside [{MIN_EFFECT}, {MAX_EFFECT}] — "
        "the fixture would either drown in gate tolerances or derail the model"
    )

    lora_ids, lora_lp, _ = topk_grid(model, dev, prompts, decodes)

    prompt_flat = [t for p in prompts for t in p]
    tensors = {
        "prompt_tokens": torch.tensor(prompt_flat, dtype=torch.int32),
        "prompt_lens": torch.tensor([len(p) for p in prompts], dtype=torch.int32),
        "decode_tokens": torch.tensor(decodes, dtype=torch.int32),
        "base_topk_ids": torch.tensor(base_ids, dtype=torch.int32),
        "base_topk_logprobs": torch.tensor(base_lp, dtype=torch.float32),
        "lora_topk_ids": torch.tensor(lora_ids, dtype=torch.int32),
        "lora_topk_logprobs": torch.tensor(lora_lp, dtype=torch.float32),
    }
    for name, w in adapter.items():
        tensors[f"adapter/{name}"] = w
    meta = {
        "rank": str(RANK),
        "lora_alpha": str(ALPHA),
        "target_modules": json.dumps(TARGET_MODULES),
        "b_scale": f"{b_scale:.6f}",
        "mean_effect_nat": f"{effect:.4f}",
        "seed": str(SEED),
        "top_k": str(TOP_K),
        "model": Path(args.model_path).name,
        "torch_version": torch.__version__,
        "peft_version": __import__("peft").__version__,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    print(
        f"wrote {out}: {NUM_SEQS} seqs, {NUM_SEQS * (DECODE_TOKENS + 1)} positions, "
        f"top{TOP_K}, adapter rank {RANK} q/v, effect {effect:.4f} nat"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
