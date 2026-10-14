#!/usr/bin/env python3
"""Generate the Gated DeltaNet golden: the gated-norm sublayer output.

Dumps the Qwen3.8-Flash-Next GDN reference (`Qwen4ExpTextGatedDeltaNet`,
transformers 5.16.1 `models/qwen4_exp`) at the **real frozen-config geometry**
into a fixture a GDN kernel gate replays: for each case the recorded activations
feeding the delta rule, and the expected gated-norm output.

Boundary rule -- the real geometry makes the module ~110 MiB, over GitHub's
100 MiB push limit, so the per-layer weights are not all stored:

  stored   `conv1d.weight` [10240,1,4] and `norm.weight` [128] -- the two
           component-local weights a GDN kernel applies to the recorded
           activations (depthwise conv, gated norm).
  recorded `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b` are plain
           GEMMs (~110 MiB combined), so their *outputs* are captured with
           forward hooks instead: the pre-conv `mixed_qkv`, its post-conv
           `query`/`key`/`value` split, `z`, and the derived gate pair
           `g = -A_log.exp()*softplus(a+dt_bias)` / `beta = sigmoid(b)`.
  dropped  `out_proj` [6144,2560] is a plain GEMM too: the golden's expected
           output is the gated-norm output *before* it (what the GDN kernel
           produces).

`A_log` [48] / `dt_bias` [48] are not stored: the recorded `g` is already their
post-derivation form, so nothing in the fixture can apply them -- re-deriving
`g` would need the `in_proj_a` / `in_proj_b` outputs, which are not stored. The
boundary sits after the gate derivation; a kernel gate that starts from the raw
`a`/`b` projections needs its own dump (which must also store those two
vectors). The layer input `hidden` is not stored for the same reason: nothing in
the fixture can consume it, since the `in_proj_*` weights are absent.

`conv1d.weight` is stored with its pre-conv input `mixed_qkv` so the depthwise
conv stage is covered end to end (post-conv `query`/`key`/`value` is reproduced
from them). The remaining traps are preserved: the value/key head ratio 3
(`repeat_interleave`, query/key stored pre-repeat) and the sigmoid gated norm.
`output_gate_type: "sigmoid"` selects that gate while the in-tree kernel
hardcodes silu, so the dump asserts `norm.activation == "sigmoid"` and pins
`config_sha256`: a silu regen fails at the source instead of writing a fixture
that contradicts the same-source re-derivation.

The reference `forward` (cache_params=None) always takes the chunk (prefill)
path; the recurrent (decode) rule is the same recurrence under a second
algorithm, so both are run and cross-checked in fp32. Neither a recurrent output
nor a recurrent state is stored -- a decode kernel gate needs its own dump.

No case is long enough for row selection: the widest stored tensor is the
S=64 pre-conv `mixed_qkv` [1,64,10240] at 1.25 MiB, comfortably under 10 MiB, so
every row is stored. Cases cover the prefill chunk path at S=16 (partial chunk)
and S=64 (one full chunk).

    python tools/accuracy/dump_qwen38fn_gdn_golden.py --out <fixture>
    python tools/accuracy/dump_qwen38fn_gdn_golden.py --verify <fixture>

The dump self-checks (captured activations reproduce the captured gated-norm
output and the module's returned output bit-for-bit; chunk/recurrent agree in
fp32; outputs finite) and, after writing, reloads the fixture and reproduces
the stored conv split (`query`/`key`/`value`) and gated-norm output (`out`)
from the stored inputs plus `conv1d_weight` / `norm_weight`, printing one line
per case.
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
from safetensors import safe_open
from safetensors.torch import save_file

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextGatedDeltaNet,
    Qwen4ExpTextRMSNormGated,
    causal_conv1d_fn,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

# Real frozen checkpoint config; every dimension below comes from it.
CONFIG = "pegainfer-qwen4/tests/frozen_config.json"

SEED = 0x_03_38_F9_5A_11_80

# Bumped when the fixture layout changes.
FORMAT_VERSION = 2
# Single metadata key; the gate asserts the digest on load.
META_KEY = "qwen4_gdn_golden"

# Prefill chunk path: S=16 is a partial chunk, S=64 is exactly one full chunk.
CASES = [16, 64]

# The chunk and recurrent rules are two algorithms for one recurrence; in fp32
# they must agree far below the bf16 output rounding this fixture stores.
FP32_REC_TOL = 1e-4


def load_config(root: Path) -> tuple[Qwen4ExpConfig, str]:
    """The real text config + the sha256 of its frozen bytes."""
    raw = (root / CONFIG).read_bytes()
    config = Qwen4ExpConfig(**json.loads(raw)).text_config
    return config, hashlib.sha256(raw).hexdigest()


def build_norm(config: Qwen4ExpConfig) -> Qwen4ExpTextRMSNormGated:
    return Qwen4ExpTextRMSNormGated(
        config.linear_value_head_dim,
        eps=config.rms_norm_eps,
        activation=config.output_gate_type or config.hidden_act,
    )


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
            output = module(hidden, None, None)
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
        "output": output,
        "mixed_qkv": cap["mixed_qkv"],
        "query": query,
        "key": key,
        "value": value,
        "z": z,
        "g": g,
        "beta": beta,
        "out": out,
    }


def rederive_out(
    config: Qwen4ExpConfig,
    norm: Qwen4ExpTextRMSNormGated,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    z: torch.Tensor,
    recurrent: bool = False,
) -> torch.Tensor:
    """The gated-norm output from the recorded delta-rule inputs (pre-repeat)."""
    ratio = config.linear_num_value_heads // config.linear_num_key_heads
    query = query.repeat_interleave(ratio, dim=2) if ratio > 1 else query
    key = key.repeat_interleave(ratio, dim=2) if ratio > 1 else key
    rule = torch_recurrent_gated_delta_rule if recurrent else torch_chunk_gated_delta_rule
    core, _ = rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    shape = z.shape
    return norm(core.reshape(-1, shape[-1]), z.reshape(-1, shape[-1])).reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write the fixture here (then self-verify)")
    parser.add_argument("--verify", type=Path, help="verify an existing fixture and exit")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repo root holding the frozen config (defaults to this script's repo)",
    )
    args = parser.parse_args()
    if not args.out and not args.verify:
        parser.error("one of --out / --verify is required")

    config, config_sha256 = load_config(args.root)

    if args.out:
        torch.manual_seed(SEED)
        module = Qwen4ExpTextGatedDeltaNet(config, 0)
        # This golden exists to pin the `output_gate_type: "sigmoid"` gated norm
        # (the in-tree kernel hardcodes silu). A silu regen would otherwise write
        # a fixture whose gate contradicts `config_sha256`, so fail at the source.
        assert module.norm.activation == "sigmoid", module.norm.activation
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

            # Self-check 1: the recorded activations reproduce the captured
            # gated-norm output bit-for-bit. The reference runs the chunk rule, so
            # pass the recorded bf16 inputs through it unchanged.
            redone = rederive_out(config, module.norm, cap["query"], cap["key"], cap["value"], cap["g"], cap["beta"], cap["z"])
            assert torch.equal(redone, cap["out"]), f"case {i} (S={seq_len}) re-derivation != captured gated-norm output"
            # Self-check 2: the module's returned output is out_proj(recorded output).
            assert torch.equal(module.out_proj(cap["out"].reshape(1, seq_len, -1)), cap["output"]), (
                f"case {i} (S={seq_len}) out_proj(recorded) != module output"
            )
            # Self-check 3: the two reference rules are one recurrence in fp32.
            ratio = config.linear_num_value_heads // config.linear_num_key_heads
            q32 = cap["query"].repeat_interleave(ratio, dim=2).float()
            k32 = cap["key"].repeat_interleave(ratio, dim=2).float()
            v32 = cap["value"].float()
            chunk, _ = torch_chunk_gated_delta_rule(
                q32, k32, v32, g=cap["g"], beta=cap["beta"].float(), initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=True,
            )
            rec, _ = torch_recurrent_gated_delta_rule(
                q32, k32, v32, g=cap["g"], beta=cap["beta"].float(), initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=True,
            )
            gap = (chunk - rec).abs().max().item()
            assert gap < FP32_REC_TOL, f"case {i} (S={seq_len}) chunk and recurrent rules disagree (fp32 gap {gap:.3e})"
            assert bool(torch.isfinite(cap["out"].float()).all()), f"case {i} (S={seq_len}) non-finite output"

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
            print(
                f"case{i}: S={seq_len} out_absmax={cap['out'].float().abs().max().item():.4f} "
                f"rederivation_ok=True rec_fp32_gap={gap:.3e}"
            )

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

    target = args.verify or args.out
    verify(target, args.root)


def verify(path: Path, root: Path) -> None:
    """Reload the fixture and reproduce every stored output from the stored
    inputs + stored weights, bit-for-bit. Prints one line per case."""
    with safe_open(str(path), framework="pt") as f:
        manifest = json.loads(f.metadata()[META_KEY])
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    assert manifest["format_version"] == FORMAT_VERSION, manifest["format_version"]

    config, config_sha256 = load_config(root)
    assert manifest["config"] == CONFIG, manifest["config"]
    assert manifest["config_sha256"] == config_sha256, "config digest mismatch"

    norm = build_norm(config)
    with torch.no_grad():
        norm.weight.copy_(tensors["norm_weight"])
    norm = norm.to(torch.bfloat16)

    for i, seq_len in enumerate(CASES):
        p = f"case{i}_"
        assert int(tensors[p + "seq_len"]) == seq_len
        # Conv stage: reproduce post-conv query/key/value from the stored pre-conv
        # mixed_qkv + the stored depthwise conv weight, bit-for-bit.
        pre = tensors[p + "mixed_qkv"]
        post = causal_conv1d_fn(
            pre.transpose(1, 2), tensors["conv1d_weight"].squeeze(1), None, activation=config.hidden_act
        ).transpose(1, 2)
        query, key, value = split_qkv(post, config)
        assert torch.equal(query, tensors[p + "query"]), f"case {i} conv stage: query mismatch"
        assert torch.equal(key, tensors[p + "key"]), f"case {i} conv stage: key mismatch"
        assert torch.equal(value, tensors[p + "value"]), f"case {i} conv stage: value mismatch"
        # Gated-norm stage: reproduce the stored output from the stored rule inputs.
        out = rederive_out(
            config, norm, tensors[p + "query"], tensors[p + "key"], tensors[p + "value"],
            tensors[p + "g"], tensors[p + "beta"], tensors[p + "z"],
        )
        assert torch.equal(out, tensors[p + "out"]), f"case {i} gated-norm output mismatch"
        # Secondary net: the recurrent (decode) rule is the same recurrence.
        rec = rederive_out(
            config, norm, tensors[p + "query"], tensors[p + "key"], tensors[p + "value"],
            tensors[p + "g"], tensors[p + "beta"], tensors[p + "z"], recurrent=True,
        )
        gap = (out.float() - rec.float()).abs().max().item()
        assert gap < 2e-2, f"case {i} chunk and recurrent rules disagree ({gap:.3e})"
        assert bool(torch.isfinite(out.float()).all()), f"case {i} non-finite output"
        print(f"case{i}: S={seq_len} conv=ok gated_norm=ok recurrent_gap={gap:.3e} finite=True")


if __name__ == "__main__":
    main()
