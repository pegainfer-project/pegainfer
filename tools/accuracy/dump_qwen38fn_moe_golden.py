#!/usr/bin/env python3
"""Generate the MoE block golden: the sparse mixture-of-experts sublayer output.

Dumps the Qwen3.8-Flash-Next MoE reference (`Qwen4ExpTextSparseMoeBlock`,
transformers 5.16.1 `models/qwen4_exp`) at the frozen checkpoint's real geometry
into a fixture a MoE kernel gate replays: the reference's own seeded weights plus,
per case, its synthetic token rows and the router logits, renormalized top-k
routing weights, selected experts and expert/shared/block outputs it computes.

Not stored, and why: the fused 512-expert stack is 2.52e9 parameters
(`experts.gate_up_proj` [512, 1280, 2560] + `experts.down_proj` [512, 2560, 640]),
~5 GB in bf16. The fixture stores instead the activation `experts` returns (the
routed-expert output entering the block sum) and the weights a gate checks
directly: the router `gate_weight` [512, 2560] and the shared expert's three
projections plus its scalar gate weight. The stack is reproducible from the
recorded `seed` at this draw order — bfloat16 parameters (the checkpoint's dtype),
`normal_(0.0, 0.05)` for every parameter with `dim() >= 2` and `normal_(0.0, 0.5)`
otherwise, over `named_parameters()` order.

Consumer notes: `hidden` is the flat `[R, 2560]` row batch (`R` = 8, 12, 1); the
block is row-wise, so a case is just its token count and a short batch hits only a
small slice of the 512 experts (the dump prints the distinct-expert counts).
`router_logits` are pre-softmax; `routing_weights` are the renormalized top-k probs
(`norm_topk_prob` is an upstream default the checkpoint's `config.json` does not
carry, so a port reading only `config.json` silently drops it and the top-k weights
stop summing to 1). `expert_out` is the reference's `experts(hidden, selected,
weights)` return and `shared_out` its shared branch before the scalar sigmoid gate,
so `out = expert_out + sigmoid(F.linear(hidden, shared_expert_gate_weight)) *
shared_out`.

Metadata is a single JSON key (`qwen4_moe_golden`, format_version 2).

    python tools/accuracy/dump_qwen38fn_moe_golden.py --out test_data/qwen38fn-moe-golden.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextSparseMoeBlock

CONFIG = "pegainfer-qwen4/tests/frozen_config.json"
METADATA_KEY = "qwen4_moe_golden"
SEED = 0x_03_38_F9_5A_11_82

# Bumped when the fixture layout changes (2: real geometry, metadata-only
# provenance, the expert stack recorded as an activation instead of stored).
FORMAT_VERSION = 2

# Token counts: two prefill batches and one decode row. The block is row-wise, so
# these fix the token shapes rather than a sequence; 12 rows already exercise the
# renorm (a token's top-10 out of 512 experts) and the shared gate.
CASES = [8, 12, 1]


def load_text_config(path: Path):
    raw = path.read_bytes()
    config = Qwen4ExpConfig(**json.loads(raw)).text_config
    return config, hashlib.sha256(raw).hexdigest()


def build_block(config) -> Qwen4ExpTextSparseMoeBlock:
    """The reference block at real geometry, seeded.

    The 2.5e9 parameters are drawn in bfloat16 — the checkpoint's dtype — which is
    also what keeps the construction at ~5 GB (building in fp32 and casting would
    peak at ~15 GB for a stack that is never stored)."""
    torch.manual_seed(SEED)
    block = Qwen4ExpTextSparseMoeBlock(config)
    with torch.no_grad():
        for _, parameter in block.named_parameters():
            if parameter.dim() >= 2:
                parameter.normal_(0.0, 0.05)
            else:
                parameter.normal_(0.0, 0.5)
    return block


def capture_hook(store: dict, name: str):
    """Record what a submodule returns, so the golden is the reference's own
    output and not this script's re-implementation of it."""

    def hook(_module, _inputs, output):
        store[name] = output

    return hook


def run_reference(block: Qwen4ExpTextSparseMoeBlock, hidden: torch.Tensor):
    """One reference pass over `hidden` [1, R, H]: the router triple, the routed
    expert activation and the shared expert output, all captured from the
    submodules, plus the block output the reference returns ([R, H] flat)."""
    captured: dict[str, object] = {}
    handles = [
        block.gate.register_forward_hook(capture_hook(captured, "router")),
        block.experts.register_forward_hook(capture_hook(captured, "experts")),
        block.shared_expert.register_forward_hook(capture_hook(captured, "shared")),
    ]
    try:
        with torch.no_grad():
            out = block(hidden)
    finally:
        for handle in handles:
            handle.remove()
    logits, weights, selected = captured["router"]
    return logits, weights, selected, captured["experts"], captured["shared"], out.view(-1, out.shape[-1])


def case_tensors(block: Qwen4ExpTextSparseMoeBlock, index: int) -> dict[str, torch.Tensor]:
    """One case: the token rows and everything the reference computes for them."""
    seq_len = CASES[index]
    hidden_dim = block.gate.hidden_dim
    gen = torch.Generator().manual_seed(SEED + index)
    hidden = torch.randn(1, seq_len, hidden_dim, generator=gen, dtype=torch.bfloat16)
    flat = hidden.view(-1, hidden_dim)
    logits, weights, selected, expert_out, shared_out, out = run_reference(block, hidden)

    experts_hit = int(torch.unique(selected).numel())
    print(
        f"case{index}: T={seq_len} experts_hit={experts_hit}/{block.gate.num_experts} "
        f"max_weight={float(weights.max()):.4f}"
    )
    prefix = f"case{index}_"
    return {
        prefix + "hidden": flat.contiguous(),
        prefix + "router_logits": logits.contiguous(),
        prefix + "routing_weights": weights.contiguous(),
        prefix + "selected_experts": selected.to(torch.int32).contiguous(),
        prefix + "expert_out": expert_out.contiguous(),
        prefix + "shared_out": shared_out.contiguous(),
        prefix + "out": out.contiguous(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config, config_sha256 = load_text_config(root / CONFIG)
    # The trap this golden carries: `norm_topk_prob` is an upstream default the
    # checkpoint's config.json does not carry, so a port that reads only
    # config.json silently drops it and the top-k weights stop summing to 1.
    torch.set_default_dtype(torch.bfloat16)
    block = build_block(config)
    tensors = {
        "gate_weight": block.gate.weight.detach().contiguous(),
        "shared_gate_proj": block.shared_expert.gate_proj.weight.detach().contiguous(),
        "shared_up_proj": block.shared_expert.up_proj.weight.detach().contiguous(),
        "shared_down_proj": block.shared_expert.down_proj.weight.detach().contiguous(),
        "shared_expert_gate_weight": block.shared_expert_gate.weight.detach().contiguous(),
    }
    for index in range(len(CASES)):
        tensors.update(case_tensors(block, index))

    manifest = {
        "format_version": FORMAT_VERSION,
        "config": CONFIG,
        "config_sha256": config_sha256,
        "seed": SEED,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "platform": platform.machine(),
    }
    # One key holding sorted JSON, since safetensors writes its metadata map in a
    # random order and several keys would make two identical runs differ.
    metadata = {METADATA_KEY: json.dumps(manifest, sort_keys=True)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata=metadata)
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.2f} MiB, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
