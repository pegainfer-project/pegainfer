#!/usr/bin/env python3
"""Generate the Gated DeltaNet golden: the gated-norm sublayer output.

Dumps the Qwen3.8-Flash-Next GDN reference (`Qwen4ExpTextGatedDeltaNet`,
transformers 5.16.1 `models/qwen4_exp`) at the real frozen-config geometry into
a fixture a GDN kernel gate replays: for each case the recorded activations
feeding the delta rule, and the reference's gated-norm output.

Boundary rule -- the real geometry makes the module ~110 MiB, over GitHub's
100 MiB push limit, so the plain `in_proj_*` / `out_proj` GEMMs are not stored;
their *outputs* are recorded with forward hooks instead: the pre-conv
`mixed_qkv`, its post-conv `query`/`key`/`value` split (pre-repeat), `z`, and
the derived gate pair `g = -A_log.exp()*softplus(a+dt_bias)` / `beta = sigmoid(b)`.
The only weights stored are `conv1d.weight` [10240,1,4] and `norm.weight` [128],
the two component-local weights a GDN kernel applies (depthwise conv, gated
norm); `out_proj` [6144,2560] is dropped -- the golden's expected output is the
gated-norm output *before* it.

`A_log` [48] / `dt_bias` [48] are not stored either -- the recorded `g` already
holds their post-derivation form, and nothing can consume `hidden` with the
`in_proj_*` weights absent. The boundary sits after the gate derivation; a gate
starting from the raw `a`/`b` projections needs its own dump (which must also
store those two vectors).

`conv1d.weight` pairs with `mixed_qkv` to cover the depthwise conv stage end to
end. Traps preserved: the value/key head ratio 3 (`repeat_interleave`,
query/key stored pre-repeat) and the sigmoid gated norm -- the reference's
`output_gate_type` is `"sigmoid"` while the in-tree kernel hardcodes silu.

The reference `forward` (cache_params=None) always takes the chunk (prefill)
path; cases cover S=16 (partial chunk) and S=64 (one full chunk), no row
selection needed (widest stored tensor: S=64 `mixed_qkv` [1,64,10240], 1.25 MiB).
Metadata is a single JSON key (`qwen4_gdn_golden`, format_version 2).

    python tools/accuracy/dump_qwen38fn_gdn_golden.py --out <fixture>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import save_file

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedDeltaNet

# Real frozen checkpoint config; every dimension below comes from it.
CONFIG = "pegainfer-qwen4/tests/frozen_config.json"

SEED = 0x_03_38_F9_5A_11_80

# Bumped when the fixture layout changes.
FORMAT_VERSION = 2
# Single metadata key; the gate asserts the digest on load.
META_KEY = "qwen4_gdn_golden"

# Prefill chunk path: S=16 is a partial chunk, S=64 is exactly one full chunk.
CASES = [16, 64]


def load_config(root: Path) -> tuple[Qwen4ExpConfig, str]:
    """The real text config + the sha256 of its frozen bytes."""
    raw = (root / CONFIG).read_bytes()
    config = Qwen4ExpConfig(**json.loads(raw)).text_config
    return config, hashlib.sha256(raw).hexdigest()


def split_qkv(mixed: torch.Tensor, config: Qwen4ExpConfig):
    """[B, S, conv_dim] post-conv -> pre-repeat query/key [B,S,nk,hk], value [B,S,nv,hv]."""
    batch, seq_len, _ = mixed.shape
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    query, key, value = torch.split(mixed, [key_dim, key_dim, value_dim], dim=-1)
    query = query.reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
    key = key.reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
    value = value.reshape(batch, seq_len, config.linear_num_value_heads, config.linear_value_head_dim)
    return query, key, value


def capture(
    module: Qwen4ExpTextGatedDeltaNet, hidden: torch.Tensor, config: Qwen4ExpConfig
) -> dict[str, torch.Tensor]:
    """Run the reference once and record what it computes.

    The big plain-GEMM weights are not stored, so their outputs are captured with
    forward hooks at the exact submodules. `conv1d` is called through the module
    level `causal_conv1d_fn` (not `module.conv1d`), so that op is monkeypatched
    for the duration of the call to record the post-conv `mixed_qkv`.
    """
    cap: dict[str, torch.Tensor] = {}

    def record(name: str):
        return lambda _m, _i, out: cap.__setitem__(name, out.detach().clone())

    handles = [
        module.in_proj_qkv.register_forward_hook(record("mixed_qkv")),
        module.in_proj_z.register_forward_hook(record("z")),
        module.in_proj_a.register_forward_hook(record("a")),
        module.in_proj_b.register_forward_hook(record("b")),
        module.norm.register_forward_hook(record("out")),
    ]
    original_conv = M.causal_conv1d_fn

    def patched_conv(*args, **kwargs):
        out = original_conv(*args, **kwargs)
        cap["conv"] = out.detach().clone()
        return out

    M.causal_conv1d_fn = patched_conv
    try:
        with torch.no_grad():
            module(hidden, None, None)
    finally:
        M.causal_conv1d_fn = original_conv
        for handle in handles:
            handle.remove()

    batch, seq_len, _ = hidden.shape
    query, key, value = split_qkv(cap["conv"].transpose(1, 2), config)
    z = cap["z"].reshape(batch, seq_len, config.linear_num_value_heads, config.linear_value_head_dim)
    # The reference derives g/beta from the a/b projections (elementwise, same op).
    g = -module.A_log.float().exp() * F.softplus(cap["a"].float() + module.dt_bias)
    beta = cap["b"].sigmoid()
    out = cap["out"].reshape(z.shape)
    return {
        "mixed_qkv": cap["mixed_qkv"],
        "query": query,
        "key": key,
        "value": value,
        "z": z,
        "g": g,
        "beta": beta,
        "out": out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="write the fixture here")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repo root holding the frozen config (defaults to this script's repo)",
    )
    args = parser.parse_args()

    config, config_sha256 = load_config(args.root)

    torch.manual_seed(SEED)
    module = Qwen4ExpTextGatedDeltaNet(config, 0)
    gen = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, param in module.named_parameters():
            scale = 0.05 if param.dim() >= 2 else 0.5
            param.copy_(torch.randn(param.shape, generator=gen) * scale)
        # Keep the recurrence bounded: the reference inits A_log = log(uniform(.01,16))
        # and dt_bias = 1; make the gated-norm gain non-trivial too.
        module.A_log.data = torch.log(torch.empty(config.linear_num_value_heads).uniform_(0.01, 16, generator=gen))
        module.dt_bias.data = torch.ones(config.linear_num_value_heads)
        module.norm.weight.data = torch.randn(config.linear_value_head_dim, generator=gen) * 0.2 + 1.0
    module = module.to(torch.bfloat16)

    # Only the weights a GDN kernel applies to the recorded activations.
    # `A_log`/`dt_bias` are absent: the recorded `g` is already their
    # post-derivation form and the `a`/`b` projections are not stored.
    tensors = {
        "conv1d_weight": module.conv1d.weight.detach().contiguous(),
        "norm_weight": module.norm.weight.detach().contiguous(),
    }

    for i, seq_len in enumerate(CASES):
        case_gen = torch.Generator().manual_seed(SEED + i)
        hidden = torch.randn(1, seq_len, config.hidden_size, generator=case_gen).to(torch.bfloat16)
        cap = capture(module, hidden, config)

        p = f"case{i}_"
        # Recorded activations, not the ~110 MiB GEMM weights: pre-conv
        # mixed_qkv + conv1d_weight cover the depthwise conv; the pre-repeat
        # query/key cover the V/K ratio 3; norm_weight covers the sigmoid gate.
        tensors[p + "seq_len"] = torch.tensor([seq_len], dtype=torch.int32)
        tensors[p + "mixed_qkv"] = cap["mixed_qkv"].contiguous()
        tensors[p + "query"] = cap["query"].contiguous()
        tensors[p + "key"] = cap["key"].contiguous()
        tensors[p + "value"] = cap["value"].contiguous()
        tensors[p + "z"] = cap["z"].contiguous()
        tensors[p + "g"] = cap["g"].contiguous()
        tensors[p + "beta"] = cap["beta"].contiguous()
        tensors[p + "out"] = cap["out"].contiguous()
        print(f"case{i}: S={seq_len} out_absmax={cap['out'].float().abs().max().item():.4f}")

    manifest = {
        "format_version": FORMAT_VERSION,
        "config": CONFIG,
        "config_sha256": config_sha256,
        "seed": SEED,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "platform": platform.machine(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata={META_KEY: json.dumps(manifest, sort_keys=True)})
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.2f} MiB, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
