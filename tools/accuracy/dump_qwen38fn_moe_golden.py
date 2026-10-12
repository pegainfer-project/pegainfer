#!/usr/bin/env python3
"""Generate the MoE block golden: the sparse mixture-of-experts sublayer output.

Dumps the Qwen3.8-Flash-Next MoE reference (`Qwen4ExpTextSparseMoeBlock`,
transformers 5.16.1 `models/qwen4_exp`) into a fixture a MoE kernel gate replays:
same seeded weights and synthetic hidden states, comparing the router logits, the
routing weights and the block output.

Every layer of this line is MoE, so this is on the critical path. The traps it
pins: the router renormalizes its top-k weights (`norm_topk_prob`, a field absent
from the checkpoint's `config.json` and therefore an upstream default), the experts
are stored as fused 3D tensors, and the shared expert rides a learned scalar gate
that is summed with the routed output.

The router and block math is re-derived inside this script and asserted to match
the module bit-for-bit (a mismatch aborts the dump).

Fixture: provenance `format_version` / `config_sha256` / `ref_transformers`
(assert on load); per case, the input `hidden` and the outputs. A reduced expert
count and intermediate width keep the file small; the routed/shared split, top-k
renormalization and fused expert layout are all preserved. Contract details a
replaying consumer needs:

- `router_logits` are pre-softmax; `routing_weights` are the renormalized top-k
  probs (`norm_topk_prob=True`), so they sum to 1 per token up to bf16 rounding.
- `experts_gate_up_proj` is `[E, 2*I, H]` with gate = rows `[0, I)` and up =
  rows `[I, 2I)` (the reference's `chunk(2, dim=-1)` halves); `experts_down_proj`
  is `[E, H, I]` in `F.linear` layout (output dim first).
- `shared_out` already carries the sigmoid of the shared gate (one scalar per
  token, not per channel).
- `expert_out` / `shared_out` are flat `[T, H]` (where the reference computes
  them); `hidden` / `out` keep the batch dim `[1, T, H]`.
- One case aligns every token with expert 0's router row, so a fixed set of 4
  experts takes all tokens (one of them nearly all the routing weight) and the
  other 12 are empty — the skewed/empty distribution the MoE spike requires;
  an S=1 case pins the decode row count.

    python tools/accuracy/dump_qwen38fn_moe_golden.py --out <fixture>
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import save_file

from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextSparseMoeBlock

# Reduced text geometry (real values in comments).
HIDDEN = 512            # real 2560
NUM_EXPERTS = 16        # real 512
TOP_K = 4               # real 10
MOE_INTERMEDIATE = 64   # real 640
SHARED_INTERMEDIATE = 64  # real 640
NORM_TOPK_PROB = True
HIDDEN_ACT = "silu"
SEED = 0x_03_38_F9_5A_11_82

# Bumped when the fixture layout changes; the gate asserts it on load.
FORMAT_VERSION = 1

# (seq_len, input mode). "randn" = synthetic hidden states, near-uniform
# routing. "skew" = every token aligned with expert 0's router row at
# SKEW_SCALE: softmax collapses onto one dominant expert and all tokens share
# the same top-k, so TOP_K experts take every token (measured renorm weights
# [0.984, 0.006, 0.006, 0.005]) and the other NUM_EXPERTS - TOP_K stay empty —
# the skewed/empty distribution the MoE spike requires. The S=1 case pins the
# decode row count.
CASES = [
    (8, "randn"),   # prefill, near-uniform
    (12, "randn"),  # prefill, near-uniform
    (1, "randn"),   # decode shape
    (16, "skew"),   # skewed routing + empty experts
]
SKEW_SCALE = 4.0


class Config:
    """Minimal config carrying exactly the attributes the MoE block reads."""

    hidden_size = HIDDEN
    num_experts = NUM_EXPERTS
    num_experts_per_tok = TOP_K
    moe_intermediate_size = MOE_INTERMEDIATE
    shared_expert_intermediate_size = SHARED_INTERMEDIATE
    norm_topk_prob = NORM_TOPK_PROB
    hidden_act = HIDDEN_ACT
    initializer_range = 0.02
    _experts_implementation = "eager"


def config_digest() -> bytes:
    """sha256 over the frozen geometry so a consumer can assert this oracle's config."""
    canon = (
        f"hidden={HIDDEN};experts={NUM_EXPERTS};topk={TOP_K};moeint={MOE_INTERMEDIATE};"
        f"sharedint={SHARED_INTERMEDIATE};norm_topk={NORM_TOPK_PROB};act={HIDDEN_ACT}"
    )
    return hashlib.sha256(canon.encode()).digest()


def bytes_tensor(data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


def rederive_router(module: Qwen4ExpTextSparseMoeBlock, flat: torch.Tensor):
    """Re-derive the router: fp32 softmax -> top-k -> optional renormalization."""
    logits = F.linear(flat, module.gate.weight)
    probs = torch.softmax(logits, dtype=torch.float, dim=-1)
    top_value, indices = torch.topk(probs, module.gate.top_k, dim=-1)
    if module.gate.norm_topk_prob:
        top_value = top_value / top_value.sum(dim=-1, keepdim=True)
    return logits, top_value.to(logits.dtype), indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cfg = Config()
    torch.manual_seed(SEED)
    module = Qwen4ExpTextSparseMoeBlock(cfg)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.05)
            else:
                p.normal_(0.0, 0.5)
    module = module.to(torch.bfloat16)

    tensors = {
        "format_version": torch.tensor([FORMAT_VERSION], dtype=torch.int32),
        "config_sha256": bytes_tensor(config_digest()),
        "ref_transformers": bytes_tensor(transformers.__version__.encode()),
        "gate": module.gate.weight.detach().contiguous(),
        "experts_gate_up_proj": module.experts.gate_up_proj.detach().contiguous(),
        "experts_down_proj": module.experts.down_proj.detach().contiguous(),
        "shared_gate_proj": module.shared_expert.gate_proj.weight.detach().contiguous(),
        "shared_up_proj": module.shared_expert.up_proj.weight.detach().contiguous(),
        "shared_down_proj": module.shared_expert.down_proj.weight.detach().contiguous(),
        "shared_expert_gate": module.shared_expert_gate.weight.detach().contiguous(),
    }

    for i, (seq_len, mode) in enumerate(CASES):
        if mode == "skew":
            row = (SKEW_SCALE * module.gate.weight[0].float()).to(torch.bfloat16)
            hidden = row.view(1, 1, HIDDEN).expand(1, seq_len, HIDDEN).contiguous()
        else:
            gen = torch.Generator().manual_seed(SEED + i)
            hidden = torch.randn(1, seq_len, HIDDEN, generator=gen).to(torch.bfloat16)
        flat = hidden.view(-1, HIDDEN)

        with torch.no_grad():
            final = module(hidden)
            logits, scores, selected = module.gate(flat)
            r_logits, r_scores, r_selected = rederive_router(module, flat)
            shared = module.shared_expert(flat)
            shared_gate = F.sigmoid(module.shared_expert_gate(flat))
            shared_out = shared_gate * shared
            experts_out = module.experts(flat, selected, scores)

        assert torch.equal(scores, r_scores), f"case {i} (S={seq_len}) router scores re-derivation != module"
        assert torch.equal(selected, r_selected), f"case {i} (S={seq_len}) router top-k re-derivation != module"
        assert torch.equal(logits, r_logits), f"case {i} (S={seq_len}) router logits re-derivation != module"
        assert torch.equal(
            final.view(-1, HIDDEN), experts_out + shared_out
        ), f"case {i} (S={seq_len}) block != experts + shared"
        # Renormalization must hold: the top-k weights sum to 1 (bf16 rounding only).
        assert torch.allclose(scores.float().sum(-1), torch.ones(flat.shape[0]), atol=1e-2), (
            f"case {i} (S={seq_len}) top-k weights do not sum to 1"
        )
        assert bool(torch.isfinite(final.float()).all()), f"case {i} (S={seq_len}) non-finite output"
        if mode == "skew":
            # The case must actually construct the condition it claims to cover:
            # all tokens share one top-k set, so NUM_EXPERTS - TOP_K experts are
            # empty and one expert carries a dominant renormalized weight.
            n_hit = len(torch.unique(selected))
            assert n_hit == TOP_K, f"case {i} (S={seq_len}) skew case hit {n_hit} experts, expected {TOP_K}"
            assert float(scores.max()) > 0.5, f"case {i} (S={seq_len}) skew case has no dominant expert"

        p = f"case{i}_"
        tensors[p + "seq_len"] = torch.tensor([seq_len], dtype=torch.int32)
        tensors[p + "hidden"] = hidden.contiguous()
        tensors[p + "router_logits"] = logits.contiguous()
        tensors[p + "routing_weights"] = scores.contiguous()
        tensors[p + "selected_experts"] = selected.to(torch.int32).contiguous()
        tensors[p + "expert_out"] = experts_out.contiguous()
        tensors[p + "shared_out"] = shared_out.contiguous()
        tensors[p + "out"] = final.contiguous()
        print(
            f"case{i}: S={seq_len} mode={mode} experts_hit={len(torch.unique(selected))} "
            f"max_w={float(scores.max()):.4f} rederivation_ok=True"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.2f} MiB, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
