#!/usr/bin/env python3
"""Generate the MoE block golden: the sparse mixture-of-experts sublayer output.

Dumps the Qwen3.8-Flash-Next MoE reference (`Qwen4ExpTextSparseMoeBlock`,
transformers 5.16.1 `models/qwen4_exp`) at the frozen checkpoint's real geometry
into a fixture a MoE kernel gate replays: the reference's own seeded weights and
synthetic hidden states, compared against the router logits, the routing weights,
the selected experts and the expert/shared/block outputs it computes.

Every layer of this line is MoE, so this is on the critical path. The traps it
pins: the router renormalizes its top-k weights (`norm_topk_prob`, a field absent
from the checkpoint's `config.json` and therefore an upstream default — a port
that reads only `config.json` silently drops it, and the top-k weights stop
summing to 1), the experts are stored as fused 3D tensors, and the shared expert
rides a learned scalar (one per token, not per channel) sigmoid gate that is
summed with the routed output.

Boundary: at real geometry the fused 512-expert stack is 2.52e9 parameters
(`experts.gate_up_proj` [512, 1280, 2560] + `experts.down_proj` [512, 2560, 640]),
~5 GB in bf16, so it is NOT stored. What is stored instead is the activation the
reference's `experts` forward returns (the routed-expert output entering the block
sum) per case, plus the weights a gate can check directly: the router `gate_weight`
[512, 2560] (~2.6 MB) and the shared expert's three projections plus its scalar
gate weight (~10 MB). The unstored stack stays reproducible from the `seed`
recorded in the metadata and the draw order this script uses (see below).

Contract details a replaying consumer needs:

- `hidden` is the flat `[R, 2560]` batch of token rows (`R` = 8 prefill, 12
  prefill, 1 decode). The block is row-wise — no token-to-token interaction — so a
  case is just its token count and every row is evaluated; no case needs row
  slicing. A short batch hits only a small slice of the 512 experts (78, 108 and 10
  distinct experts for the three cases, printed by the dump), so unrouted experts
  are covered by construction.
- `router_logits` are pre-softmax; `routing_weights` are the renormalized top-k
  probs (`norm_topk_prob=True`), so they sum to 1 per token up to bf16 rounding.
- `expert_out` is the recorded return of `experts(hidden, selected, weights)`: per
  hit expert `gate, up = F.linear(rows, gate_up_proj[e]).chunk(2, dim=-1)` (gate =
  columns `[0, I)`, up = `[I, 2I)`), then `act_fn(gate) * up`, then
  `F.linear(·, down_proj[e])` (`down_proj` is `[E, H, I]`, output dim first),
  scaled by `weights[row, slot]` and accumulated with `index_add_`.
- `shared_out` is the shared expert's output BEFORE its gate;
  `sigmoid(F.linear(hidden, shared_expert_gate_weight))` multiplies it and
  `out = expert_out + that product`.
- To rebuild the unstored expert stack, construct the block with the recorded
  `seed` and this script's draw order: bfloat16 parameters (the checkpoint's
  dtype), `normal_(0.0, 0.05)` for every parameter with `dim() >= 2` and
  `normal_(0.0, 0.5)` otherwise, over `named_parameters()` order.

After writing, the fixture is reloaded and each case replayed against a block
rebuilt from the recorded seed: the stored weights must be that draw, and the
router logits, the renormalized top-k weights, the selected experts, the shared
output, the block sum and the recorded `expert_out` (the one tensor the boundary
keeps out of the file) must all reproduce bit-for-bit.

    python tools/accuracy/dump_qwen38fn_moe_golden.py --out <fixture>
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file, save_file

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextSparseMoeBlock

CONFIG = "pegainfer-qwen4/tests/frozen_config.json"
METADATA_KEY = "qwen4_moe_golden"
SEED = 0x_03_38_F9_5A_11_82

# Bumped when the fixture layout changes (2: real geometry, metadata-only
# provenance, the expert stack recorded as an activation instead of stored).
FORMAT_VERSION = 2

# Token counts: two prefill batches and one decode row. The block is row-wise, so
# these pin the token shapes rather than a sequence; 12 rows already exercise the
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


def rederive_router(block: Qwen4ExpTextSparseMoeBlock, flat: torch.Tensor):
    """Re-derive the router: fp32 softmax -> top-k -> optional renormalization."""
    logits = F.linear(flat, block.gate.weight)
    probs = torch.softmax(logits, dtype=torch.float, dim=-1)
    top_value, indices = torch.topk(probs, block.gate.top_k, dim=-1)
    if block.gate.norm_topk_prob:
        top_value = top_value / top_value.sum(dim=-1, keepdim=True)
    return logits, top_value.to(logits.dtype), indices


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
    """One case: the token rows and everything the reference computes for them.

    The captured values are the source of truth; the re-derivations below are the
    cheap consistency checks that abort the dump.
    """
    seq_len = CASES[index]
    hidden_dim = block.gate.hidden_dim
    gen = torch.Generator().manual_seed(SEED + index)
    hidden = torch.randn(1, seq_len, hidden_dim, generator=gen, dtype=torch.bfloat16)
    flat = hidden.view(-1, hidden_dim)
    logits, weights, selected, expert_out, shared_out, out = run_reference(block, hidden)

    r_logits, r_weights, r_selected = rederive_router(block, flat)
    assert torch.equal(logits, r_logits), f"case {index} (T={seq_len}): router logits re-derivation != module"
    assert torch.equal(weights, r_weights), f"case {index} (T={seq_len}): routing weights re-derivation != module"
    assert torch.equal(selected, r_selected), f"case {index} (T={seq_len}): top-k selection re-derivation != module"

    # The block is the routed output plus the shared output under its learned
    # scalar sigmoid gate.
    gate_scalar = F.sigmoid(block.shared_expert_gate(flat))
    assert torch.equal(out, expert_out + gate_scalar * shared_out), (
        f"case {index} (T={seq_len}): block != routed output + sigmoid(shared gate) * shared output"
    )
    # Renormalization must hold: the top-k weights sum to 1. Without it the top-10
    # probs of a 512-way softmax sum to ~0.03, well outside bf16 rounding.
    assert torch.allclose(weights.float().sum(-1), torch.ones(seq_len, dtype=torch.float32), atol=1e-2), (
        f"case {index} (T={seq_len}): routing weights do not sum to 1 (renorm dropped?)"
    )
    assert bool(torch.isfinite(out.float()).all()), f"case {index} (T={seq_len}): non-finite block output"

    experts_hit = int(torch.unique(selected).numel())
    print(
        f"case{index}: T={seq_len} experts_hit={experts_hit}/{block.gate.num_experts} "
        f"max_weight={float(weights.max()):.4f} renorm_sum_ok=True rederivation_ok=True"
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


def reproduce_with_reference(check: dict[str, torch.Tensor], config) -> None:
    """Replay every case against a block rebuilt from the recorded seed: the
    stored weights must be that draw, and the recorded routed-expert activation —
    the one tensor the boundary keeps out of the file — must reproduce too."""
    replay = build_block(config)
    stored_weights = {
        "gate_weight": replay.gate.weight,
        "shared_gate_proj": replay.shared_expert.gate_proj.weight,
        "shared_up_proj": replay.shared_expert.up_proj.weight,
        "shared_down_proj": replay.shared_expert.down_proj.weight,
        "shared_expert_gate_weight": replay.shared_expert_gate.weight,
    }
    for name, parameter in stored_weights.items():
        assert torch.equal(parameter.detach(), check[name]), f"stored {name} != the seeded reference draw"
    print(f"reload: {len(stored_weights)} stored weights match the seeded reference draw")

    for index, seq_len in enumerate(CASES):
        prefix = f"case{index}_"
        gen = torch.Generator().manual_seed(SEED + index)
        hidden = torch.randn(1, seq_len, config.hidden_size, generator=gen, dtype=torch.bfloat16)
        assert torch.equal(hidden.view(-1, config.hidden_size), check[prefix + "hidden"]), (
            f"case {index}: seeded hidden rows != stored rows"
        )
        logits, weights, selected, expert_out, shared_out, out = run_reference(replay, hidden)
        assert torch.equal(logits, check[prefix + "router_logits"]), f"case {index}: reload router logits mismatch"
        assert torch.equal(weights, check[prefix + "routing_weights"]), f"case {index}: reload routing weights mismatch"
        assert torch.equal(selected.to(torch.int32), check[prefix + "selected_experts"]), (
            f"case {index}: reload selected experts mismatch"
        )
        assert torch.equal(expert_out, check[prefix + "expert_out"]), f"case {index}: reload expert_out mismatch"
        assert torch.equal(shared_out, check[prefix + "shared_out"]), f"case {index}: reload shared_out mismatch"
        assert torch.equal(out, check[prefix + "out"]), f"case {index}: reload out mismatch"
        print(f"case{index}: T={seq_len} reproduce OK (recorded expert_out via the seeded reference)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config, config_sha256 = load_text_config(root / CONFIG)
    # The trap: `norm_topk_prob` is an upstream default the checkpoint's config.json
    # does not carry, so the frozen config only supplies it through the reference's
    # class default. If that ever stops being true, the port's config plumbing (not
    # this golden) needs another look.
    assert "norm_topk_prob" not in json.loads((root / CONFIG).read_bytes())["text_config"], (
        "the frozen config now carries norm_topk_prob; re-check why the trap this golden pins changed"
    )
    assert config.norm_topk_prob is True, "the reference no longer renormalizes its top-k weights"

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

    del block, tensors
    gc.collect()
    check = load_file(str(args.out))
    assert check["gate_weight"].shape == (config.num_experts, config.hidden_size), "stored gate_weight shape"
    reproduce_with_reference(check, config)
    print(f"reload: {len(CASES)} cases reproduce bit-for-bit")


if __name__ == "__main__":
    main()
