#!/usr/bin/env python3
"""Generate the hyper-connection gated-residual golden at real geometry.

Dumps the Qwen3.8-Flash-Next gated-residual reference (`Qwen4ExpTextGatedResidual`,
transformers 5.16.1 `models/qwen4_exp`) into a fixture a hyper-connection kernel gate
replays. Every dimension is taken from the real frozen checkpoint config
(`hc_count` 4 -> a 10240-wide stream, `hidden_size` 2560, `hc_lowrank` 320); nothing
is shrunk and nothing is dropped, because the mix GEMMs *are* the component under test.

This is the residual block of the line: the stream is 4 branches wide; a grouped
`(1+w)` RMSNorm (group size 2560, i.e. one group per branch) feeds a rank-`hc_lowrank`
mix gate (`silu` -> `sigmoid`) that combines the branches by **mean**, and a per-branch
injection weight re-scatters the sublayer output. `/hc_count` is applied on both the mix
and the inject paths, and the injection weights live in `(0, 2)`. The top-level readout
uses the same module with `use_combine=False` (there is no final-norm tensor), so both
variants are captured: `use_combine=True` returns the mixed stream, the un-normed hyper
echo and the injection weights, `use_combine=False` returns only the mixer readout.

The mix/injection formulas are re-derived from the module's own submodules and asserted
bit-for-bit equal to the module output (a mismatch aborts the dump); the grouped `(1+w)`
norm is additionally reconstructed independently of the module so a grouping/eps
regression cannot hide. This re-derivation is cheap and allowed here because it is
elementwise.

After writing, the fixture is reloaded and every stored output is reproduced bit-for-bit
from the stored inputs and stored weights.

    python tools/accuracy/dump_qwen38fn_hc_golden.py --out test_data/qwen38fn-hc-golden.safetensors

Pass --verify-only to skip the dump and re-check an existing fixture in place.
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
from safetensors.torch import load_file, save_file

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

CONFIG = "pegainfer-qwen4/tests/frozen_config.json"
METADATA_KEY = "qwen4_hc_golden"

SEED = 0x_03_38_F9_5A_11_81

# Bumped when the fixture layout changes; the gate asserts it on load. v2 records the
# manifest as JSON metadata (no loose provenance tensors) and uses real geometry.
FORMAT_VERSION = 2

CASES = [6, 10]  # sequence lengths; the input stream is [1, S, hc_count * hidden_size]


def load_text_config(path: Path):
    """Load the real frozen config exactly as the indexer golden does."""
    raw = path.read_bytes()
    config = Qwen4ExpConfig(**json.loads(raw)).text_config
    return config, hashlib.sha256(raw).hexdigest()


def build_modules(config) -> tuple[Qwen4ExpTextGatedResidual, Qwen4ExpTextGatedResidual]:
    """The combine variant (residual block) and the top-level mixer (readout)."""
    torch.manual_seed(SEED)
    combine = Qwen4ExpTextGatedResidual(config, use_combine=True)
    mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
    with torch.no_grad():
        # The norm weights start at zero; give the (1 + w) scale something to do.
        for module in (combine, mixer):
            for p in module.parameters():
                p.normal_(0.0, 0.05 if p.dim() >= 2 else 0.5)
    return combine.to(torch.bfloat16), mixer.to(torch.bfloat16)


def grouped_norm(weight: torch.Tensor, x: torch.Tensor, hc_count: int, hidden: int, eps: float) -> torch.Tensor:
    """Independent grouped `(1+w)` RMSNorm: `hc_count` groups of `hidden`, per-group rms
    in fp32, scaled by `(1+w)`, cast back to x's dtype. Mirrors
    `Qwen4ExpTextRMSNorm(hc_count*hidden, group_size=hidden)` WITHOUT calling it, so a
    grouping/eps regression in the reference turns the dump-time assertion red."""
    xf = x.float().unflatten(-1, (hc_count, hidden))
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out.flatten(-2) * (1.0 + weight.float())).to(x.dtype)


def branch_mix(module: Qwen4ExpTextGatedResidual, normed: torch.Tensor) -> torch.Tensor:
    """Re-derive the per-branch mix products from the module's OWN submodules: a
    `silu -> sigmoid` gate at rank `hc_lowrank`, `/hc_count` on the down projection.
    Shape `[..., hc_count, hidden_size]`."""
    hc = module.hc_count
    hidden = module.hidden_size
    mix = F.silu(module.input_mix_weight_down(normed) / hc)
    mix = torch.sigmoid(module.input_mix_weight_up(mix)).unflatten(-1, (hc, hidden))
    return mix * normed.unflatten(-1, (hc, hidden))


def inject_weights(module: Qwen4ExpTextGatedResidual, normed: torch.Tensor):
    """Re-derive the injection weights `2*sigmoid(W @ x / hc_count)` from the module's OWN
    submodule; None for the `use_combine=False` mixer (no injection head)."""
    if module.block_inject_weight is None:
        return None
    return 2 * torch.sigmoid(module.block_inject_weight(normed) / module.hc_count)


def case_tensors(combine, mixer, config, index: int) -> dict[str, torch.Tensor]:
    seq_len = CASES[index]
    gen = torch.Generator().manual_seed(SEED + index)
    hyper = torch.randn(1, seq_len, config.hc_count * config.hidden_size, generator=gen).to(torch.bfloat16)

    with torch.no_grad():
        mixed, hyper_out, inject = combine(hyper)
        mixed_only = mixer(hyper)
        mod_normed = combine.hc_norm(hyper)
        normed = grouped_norm(combine.hc_norm.weight, hyper, config.hc_count, config.hidden_size, config.rms_norm_eps)
        branches = branch_mix(combine, mod_normed)
        r_mixed = branches.mean(dim=-2)
        r_inject = inject_weights(combine, mod_normed)
        r_mixed_only = branch_mix(mixer, mixer.hc_norm(hyper)).mean(dim=-2)

    assert torch.equal(mod_normed, normed), f"case {index} (S={seq_len}) grouped (1+w) norm re-derivation != module"
    assert torch.equal(mixed, r_mixed), f"case {index} (S={seq_len}) mix re-derivation != module"
    assert torch.equal(inject, r_inject), f"case {index} (S={seq_len}) injection re-derivation != module"
    assert torch.equal(mixed_only, r_mixed_only), (
        f"case {index} (S={seq_len}) mixer (use_combine=False) re-derivation != module"
    )
    assert torch.equal(hyper_out, hyper), f"case {index} (S={seq_len}) module echoed a different hyper stream"
    assert bool(torch.isfinite(mixed.float()).all()), f"case {index} (S={seq_len}) non-finite mix"
    assert bool(torch.isfinite(mixed_only.float()).all()), f"case {index} (S={seq_len}) non-finite mixer readout"
    # Injection is 2 * sigmoid, so it lives strictly inside (0, 2).
    assert bool(((inject > 0) & (inject < 2)).all()), f"case {index} (S={seq_len}) injection out of range"
    # The branch combine is a mean, not a sum: the per-branch products average to the readout.
    assert torch.equal(mixed, branches.mean(dim=-2)), f"case {index} (S={seq_len}) combine is not a mean"
    assert not torch.equal(mixed, branches.sum(dim=-2)), f"case {index} (S={seq_len}) combine looks like a sum"

    p = f"case{index}_"
    return {
        p + "seq_len": torch.tensor([seq_len], dtype=torch.int32),
        p + "hyper_input": hyper.contiguous(),
        p + "normed": normed.contiguous(),
        p + "mixed": mixed.contiguous(),
        p + "inject": inject.contiguous(),
        p + "mixed_only": mixed_only.contiguous(),
    }


def verify(out: Path, config) -> list[str]:
    """Reload the fixture and reproduce EVERY stored output bit-for-bit from the stored
    inputs and stored weights. Returns one line per case."""
    t = load_file(str(out))
    hc, hidden, eps = config.hc_count, config.hidden_size, config.rms_norm_eps

    def norm_from(weight, x):
        return grouped_norm(weight, x, hc, hidden, eps)

    def mix_from(normed, down_w, up_w):
        m = F.silu(F.linear(normed, down_w) / hc)
        m = torch.sigmoid(F.linear(m, up_w)).unflatten(-1, (hc, hidden))
        return (m * normed.unflatten(-1, (hc, hidden))).mean(dim=-2)

    lines = []
    for index, seq_len in enumerate(CASES):
        p = f"case{index}_"
        hyper = t[p + "hyper_input"]
        normed = norm_from(t["hc_norm"], hyper)
        assert torch.equal(normed, t[p + "normed"]), f"reload case {index}: normed"
        mixed = mix_from(normed, t["input_mix_weight_down"], t["input_mix_weight_up"])
        assert torch.equal(mixed, t[p + "mixed"]), f"reload case {index}: mixed"
        inject = 2 * torch.sigmoid(F.linear(normed, t["block_inject_weight"]) / hc)
        assert torch.equal(inject, t[p + "inject"]), f"reload case {index}: inject"
        mixer_normed = norm_from(t["mixer_norm"], hyper)
        mixed_only = mix_from(mixer_normed, t["mixer_down"], t["mixer_up"])
        assert torch.equal(mixed_only, t[p + "mixed_only"]), f"reload case {index}: mixed_only"
        assert int(t[p + "seq_len"][0]) == seq_len, f"reload case {index}: seq_len"
        lines.append(
            f"case{index}: S={seq_len} inject_mean={inject.float().mean().item():.4f} reload_reproduction_ok=True"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true", help="reload and re-check an existing fixture")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config, config_sha256 = load_text_config(root / CONFIG)
    if args.verify_only:
        for line in verify(args.out, config):
            print("verify " + line)
        return

    combine, mixer = build_modules(config)

    # Provenance faithfulness: config_sha256 digests rms_norm_eps, so abort if either
    # module norm was built with a different eps than the config claims.
    assert combine.hc_norm.eps == config.rms_norm_eps, "combine norm eps drifted from the config"
    assert mixer.hc_norm.eps == config.rms_norm_eps, "mixer norm eps drifted from the config"

    # The mix GEMMs are the component under test, so ALL weights are stored.
    tensors = {
        "hc_norm": combine.hc_norm.weight.detach().contiguous(),
        "input_mix_weight_down": combine.input_mix_weight_down.weight.detach().contiguous(),
        "input_mix_weight_up": combine.input_mix_weight_up.weight.detach().contiguous(),
        "block_inject_weight": combine.block_inject_weight.weight.detach().contiguous(),
        "mixer_norm": mixer.hc_norm.weight.detach().contiguous(),
        "mixer_down": mixer.input_mix_weight_down.weight.detach().contiguous(),
        "mixer_up": mixer.input_mix_weight_up.weight.detach().contiguous(),
    }
    for index in range(len(CASES)):
        tensors.update(case_tensors(combine, mixer, config, index))

    manifest = {
        "format_version": FORMAT_VERSION,
        "config": CONFIG,
        "config_sha256": config_sha256,
        "seed": SEED,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "platform": platform.machine(),
    }
    # One key holding sorted JSON, since safetensors writes its metadata map in a random
    # order and several keys would make two identical runs differ.
    metadata = {METADATA_KEY: json.dumps(manifest, sort_keys=True)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata=metadata)
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.2f} MiB, {len(tensors)} tensors)")

    for line in verify(args.out, config):
        print("verify " + line)


if __name__ == "__main__":
    main()
