#!/usr/bin/env python3
"""Generate the Gated DeltaNet golden: the linear-attention sublayer output.

Dumps the Qwen3.8-Flash-Next GDN reference (`Qwen4ExpTextGatedDeltaNet`,
transformers 5.16.1 `models/qwen4_exp`) into a fixture a GDN kernel gate replays:
same seeded weights and synthetic hidden states, comparing the sublayer output.

GDN geometry, tensor names and dtypes match the in-tree Qwen3.5 stack, but this
checkpoint's `output_gate_type: "sigmoid"` selects the gated-norm activation, and
the shared kernel hardcodes silu. This golden pins the reference including that
sigmoid gate.

The reference `forward` re-derives the sublayer inside this script (conv1d ->
split -> gated delta rule -> sigmoid gated norm -> out_proj) and asserts the
re-derivation matches the module bit-for-bit (a mismatch aborts the dump). The
chunk (prefill) and recurrent (decode) rules are cross-checked at dump time only:
neither the recurrent output nor a recurrent state is stored, so this fixture
oracles the prefill path -- a decode kernel gate needs its own dump.

Fixture: provenance `format_version` / `config_sha256` / `ref_transformers`
(assert on load); per case, the prefill input `hidden` and its output `out`. A
reduced geometry keeps the file small; the traps preserved are the key/value head
ratio > 1 (repeat_interleave), the depthwise conv, and the sigmoid gated norm.
The gate is this golden's reason to exist (the shared kernel hardcodes silu), so
the dump asserts `norm.activation == "sigmoid"` and `config_sha256` records it: a
silu regen fails at the source instead of writing a wrong fixture.

    python tools/accuracy/dump_qwen38fn_gdn_golden.py --out <fixture>
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file

from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextGatedDeltaNet,
    causal_conv1d_fn,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

# Reduced text geometry (real values in comments).
HIDDEN = 512                    # real 2560
NUM_V_HEADS = 8                 # real 48
NUM_K_HEADS = 4                 # real 16
HEAD_K_DIM = 64                 # real 128
HEAD_V_DIM = 64                 # real 128
CONV_KERNEL = 4
RMS_EPS = 1e-6
OUTPUT_GATE_TYPE = "sigmoid"

SEED = 0x_03_38_F9_5A_11_80

FORMAT_VERSION = 1

CASES = [8, 16]  # prefill sequence lengths


class Config:
    """Minimal config carrying exactly the attributes the GDN reads."""

    hidden_size = HIDDEN
    linear_num_value_heads = NUM_V_HEADS
    linear_num_key_heads = NUM_K_HEADS
    linear_key_head_dim = HEAD_K_DIM
    linear_value_head_dim = HEAD_V_DIM
    linear_conv_kernel_dim = CONV_KERNEL
    hidden_act = "silu"
    rms_norm_eps = RMS_EPS
    output_gate_type = OUTPUT_GATE_TYPE
    layer_types = ["linear_attention"]


def config_digest() -> bytes:
    canon = (
        f"hidden={HIDDEN};nv={NUM_V_HEADS};nk={NUM_K_HEADS};kdim={HEAD_K_DIM};"
        f"vdim={HEAD_V_DIM};conv={CONV_KERNEL};gate={OUTPUT_GATE_TYPE}"
    )
    return hashlib.sha256(canon.encode()).digest()


def bytes_tensor(data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


def project(module: Qwen4ExpTextGatedDeltaNet, hidden: torch.Tensor):
    """The pre-attention half of the module's forward: conv1d -> split -> g/beta."""
    batch, seq_len, _ = hidden.shape
    mixed = module.in_proj_qkv(hidden).transpose(1, 2)
    mixed = causal_conv1d_fn(
        mixed, module.conv1d.weight.squeeze(1), module.conv1d.bias, activation=module.activation
    ).transpose(1, 2)
    query, key, value = torch.split(mixed, [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    query = query.reshape(batch, seq_len, -1, module.head_k_dim)
    key = key.reshape(batch, seq_len, -1, module.head_k_dim)
    value = value.reshape(batch, seq_len, -1, module.head_v_dim)
    z = module.in_proj_z(hidden).reshape(batch, seq_len, -1, module.head_v_dim)
    beta = module.in_proj_b(hidden).sigmoid()
    g = -module.A_log.float().exp() * torch.nn.functional.softplus(
        module.in_proj_a(hidden).float() + module.dt_bias
    )
    rep = module.num_v_heads // module.num_k_heads
    if rep > 1:
        query = query.repeat_interleave(rep, dim=2)
        key = key.repeat_interleave(rep, dim=2)
    return query, key, value, g, beta, z


def finish(module: Qwen4ExpTextGatedDeltaNet, core: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """The gated-norm + out_proj half of the module's forward."""
    batch, seq_len = core.shape[0], core.shape[1]
    core = core.reshape(-1, module.head_v_dim)
    core = module.norm(core, z.reshape(-1, module.head_v_dim))
    return module.out_proj(core.reshape(batch, seq_len, -1))


def forward_ref(module: Qwen4ExpTextGatedDeltaNet, hidden: torch.Tensor) -> torch.Tensor:
    """Re-derive the module's forward from its own submodules (cache_params=None,
    so always the chunk path). Mirrors `Qwen4ExpTextGatedDeltaNet.forward`."""
    query, key, value, g, beta, z = project(module, hidden)
    core, _ = torch_chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=None,
        output_final_state=False, use_qk_l2norm_in_kernel=True,
    )
    return finish(module, core, z)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cfg = Config()
    torch.manual_seed(SEED)
    module = Qwen4ExpTextGatedDeltaNet(cfg, 0)
    # This golden exists to pin the `output_gate_type: "sigmoid"` gated norm (the
    # in-tree kernel hardcodes silu). The re-derivation below calls `module.norm`
    # on both arms, so it is same-source for the gate and cannot catch a
    # sigmoid->silu flip. Assert it here so a regen that loses the gate fails at
    # the source rather than silently writing a silu fixture.
    assert module.norm.activation == "sigmoid", module.norm.activation
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.05)
            else:
                p.normal_(0.0, 0.5)
        # Keep the recurrence bounded: the reference inits A_log = log(uniform(.01,16)) and dt_bias = 1.
        module.A_log.data = torch.log(torch.empty(NUM_V_HEADS).uniform_(0.01, 16))
        module.dt_bias.data = torch.ones(NUM_V_HEADS)
        # GDN's gated norm is plain `w` (default ones); make it non-trivial.
        module.norm.weight.data.normal_(1.0, 0.2)
    module = module.to(torch.bfloat16)

    tensors = {
        "format_version": torch.tensor([FORMAT_VERSION], dtype=torch.int32),
        "config_sha256": bytes_tensor(config_digest()),
        "ref_transformers": bytes_tensor(transformers.__version__.encode()),
        "in_proj_qkv": module.in_proj_qkv.weight.detach().contiguous(),
        "in_proj_z": module.in_proj_z.weight.detach().contiguous(),
        "in_proj_a": module.in_proj_a.weight.detach().contiguous(),
        "in_proj_b": module.in_proj_b.weight.detach().contiguous(),
        "conv1d_weight": module.conv1d.weight.detach().contiguous(),
        "A_log": module.A_log.detach().contiguous(),
        "dt_bias": module.dt_bias.detach().contiguous(),
        "norm_weight": module.norm.weight.detach().contiguous(),
        "out_proj": module.out_proj.weight.detach().contiguous(),
    }

    for i, seq_len in enumerate(CASES):
        gen = torch.Generator().manual_seed(SEED + i)
        hidden = torch.randn(1, seq_len, HIDDEN, generator=gen).to(torch.bfloat16)

        with torch.no_grad():
            out = module(hidden, None, None)
            rederived = forward_ref(module, hidden)
            query, key, value, g, beta, z = project(module, hidden)
            recurrent, _ = torch_recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=True,
            )
            recurrent = finish(module, recurrent, z)

        assert torch.equal(out, rederived), f"case {i} (S={seq_len}) re-derivation != module output"
        # The recurrent (decode) and chunk (prefill) rules are two algorithms for the
        # same recurrence; the golden must sit where they agree. Both emit bf16 O(1)
        # and agree to bf16 rounding (observed bit-identical on CPU); 2e-2 (~5 bf16
        # ULP at |out|~1) leaves room for cross-backend reduction order without
        # admitting a real divergence. The bit-for-bit guard above is primary; this
        # is a secondary net.
        assert torch.allclose(out.float(), recurrent.float(), atol=2e-2, rtol=2e-2), (
            f"case {i} (S={seq_len}) chunk and recurrent rules disagree"
        )
        assert bool(torch.isfinite(out.float()).all()), f"case {i} (S={seq_len}) non-finite output"

        p = f"case{i}_"
        tensors[p + "seq_len"] = torch.tensor([seq_len], dtype=torch.int32)
        tensors[p + "hidden"] = hidden.contiguous()
        tensors[p + "out"] = out.contiguous()
        print(f"case{i}: S={seq_len} out_mean={out.float().mean().item():.4f} rederivation_ok=True")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.2f} MiB, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
