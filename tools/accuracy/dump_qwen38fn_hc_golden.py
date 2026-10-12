#!/usr/bin/env python3
"""Generate the hyper-connection gated-residual golden.

Dumps the Qwen3.8-Flash-Next gated-residual reference (`Qwen4ExpTextGatedResidual`,
transformers 5.16.1 `models/qwen4_exp`) into a fixture a hyper-connection kernel
gate replays: same seeded weights and synthetic 4-branch stream input, comparing
the mixed stream, the injection weights and the grouped `(1+w)` norm output.

This is the residual block of the line: the stream is 4 branches wide; a grouped
`(1+w)` RMSNorm feeds a rank-`hc_lowrank` mix gate (silu -> sigmoid) that combines
the branches by **mean**, and a per-branch injection weight re-scatters the sublayer
output. The top-level readout uses the same module with `use_combine=False`, so
both variants are captured (`mixed_only`).

The reference formulas are re-derived inside this script and asserted to match the
module bit-for-bit (a mismatch aborts the dump).

Fixture: provenance `format_version` / `config_sha256` / `ref_transformers`
(assert on load); per case, the input `hyper_input` (4 x hidden) and the outputs.
A reduced geometry keeps the file small.

    python tools/accuracy/dump_qwen38fn_hc_golden.py --out <fixture>
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import save_file

from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

# Reduced text geometry (real values in comments).
HIDDEN = 512        # real 2560
HC_COUNT = 4        # real 4 (the 10240-wide stream)
HC_LOWRANK = 32     # real 320
RMS_EPS = 1e-6

SEED = 0x_03_38_F9_5A_11_81

FORMAT_VERSION = 1

CASES = [6, 10]  # sequence lengths


class Config:
    """Minimal config carrying exactly the attributes the gated residual reads."""

    hidden_size = HIDDEN
    hc_count = HC_COUNT
    hc_lowrank = HC_LOWRANK
    rms_norm_eps = RMS_EPS


def config_digest() -> bytes:
    canon = f"hidden={HIDDEN};hc={HC_COUNT};lowrank={HC_LOWRANK}"
    return hashlib.sha256(canon.encode()).digest()


def bytes_tensor(data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


def rederive(module: Qwen4ExpTextGatedResidual, hyper: torch.Tensor):
    """Re-derive the module's forward: grouped norm -> silu/sigmoid mix -> mean,
    plus the block injection weights."""
    normed = module.hc_norm(hyper)
    mix = F.silu(module.input_mix_weight_down(normed) / module.hc_count)
    mix = torch.sigmoid(module.input_mix_weight_up(mix)).unflatten(-1, (module.hc_count, module.hidden_size))
    mixed = (mix * normed.unflatten(-1, (module.hc_count, module.hidden_size))).mean(dim=-2)
    inject = 2 * torch.sigmoid(module.block_inject_weight(normed) / module.hc_count)
    return normed, mixed, inject


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cfg = Config()
    torch.manual_seed(SEED)
    combine = Qwen4ExpTextGatedResidual(cfg, use_combine=True)
    mixer = Qwen4ExpTextGatedResidual(cfg, use_combine=False)
    with torch.no_grad():
        for module in (combine, mixer):
            for p in module.parameters():
                p.normal_(0.0, 0.05 if p.dim() >= 2 else 0.5)
    combine = combine.to(torch.bfloat16)
    mixer = mixer.to(torch.bfloat16)

    tensors = {
        "format_version": torch.tensor([FORMAT_VERSION], dtype=torch.int32),
        "config_sha256": bytes_tensor(config_digest()),
        "ref_transformers": bytes_tensor(transformers.__version__.encode()),
        "hc_norm": combine.hc_norm.weight.detach().contiguous(),
        "input_mix_weight_down": combine.input_mix_weight_down.weight.detach().contiguous(),
        "input_mix_weight_up": combine.input_mix_weight_up.weight.detach().contiguous(),
        "block_inject_weight": combine.block_inject_weight.weight.detach().contiguous(),
        "mixer_norm": mixer.hc_norm.weight.detach().contiguous(),
        "mixer_down": mixer.input_mix_weight_down.weight.detach().contiguous(),
        "mixer_up": mixer.input_mix_weight_up.weight.detach().contiguous(),
    }

    for i, seq_len in enumerate(CASES):
        gen = torch.Generator().manual_seed(SEED + i)
        hyper = torch.randn(1, seq_len, HC_COUNT * HIDDEN, generator=gen).to(torch.bfloat16)

        with torch.no_grad():
            mixed, hyper_out, inject = combine(hyper)
            mixed_only = mixer(hyper)
            normed, r_mixed, r_inject = rederive(combine, hyper)

        assert torch.equal(mixed, r_mixed), f"case {i} (S={seq_len}) mix re-derivation != module"
        assert torch.equal(inject, r_inject), f"case {i} (S={seq_len}) injection re-derivation != module"
        assert torch.equal(hyper_out, hyper), f"case {i} (S={seq_len}) module echoed a different hyper stream"
        assert bool(torch.isfinite(mixed.float()).all()), f"case {i} (S={seq_len}) non-finite mix"
        # Injection is 2 * sigmoid, so it lives strictly inside (0, 2).
        assert bool(((inject > 0) & (inject < 2)).all()), f"case {i} (S={seq_len}) injection out of range"

        p = f"case{i}_"
        tensors[p + "seq_len"] = torch.tensor([seq_len], dtype=torch.int32)
        tensors[p + "hyper_input"] = hyper.contiguous()
        tensors[p + "normed"] = normed.contiguous()
        tensors[p + "mixed"] = mixed.contiguous()
        tensors[p + "inject"] = inject.contiguous()
        tensors[p + "mixed_only"] = mixed_only.contiguous()
        print(
            f"case{i}: S={seq_len} inject_mean={inject.float().mean().item():.4f} "
            f"rederivation_ok=True"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.2f} MiB, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
