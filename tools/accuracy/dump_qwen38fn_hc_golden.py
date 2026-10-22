#!/usr/bin/env python3
"""Generate the hyper-connection gated-residual golden at real geometry.

Dumps the Qwen3.8-Flash-Next gated-residual reference (`Qwen4ExpTextGatedResidual`,
transformers 5.16.1 `models/qwen4_exp`) into a fixture a hyper-connection kernel gate
replays, at the real frozen-checkpoint geometry (`hc_count` 4, `hidden_size` 2560,
`hc_lowrank` 320); nothing is shrunk, because the mix GEMMs *are* the component under test.

The residual block is a 4-branch stream: a grouped `(1+w)` RMSNorm (one group per branch)
feeds a rank-`hc_lowrank` mix gate (`silu` -> `sigmoid`) that combines the branches by
**mean**, and a per-branch injection weight re-scatters the sublayer output (`/hc_count`
on both paths); the top-level readout reuses the module with `use_combine=False`. Cases
S=6/10; the manifest is one JSON metadata key (`qwen4_hc_golden`, format_version 2).

    python tools/accuracy/dump_qwen38fn_hc_golden.py --out test_data/qwen38fn-hc-golden.safetensors
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


def case_tensors(combine, mixer, config, index: int) -> dict[str, torch.Tensor]:
    seq_len = CASES[index]
    gen = torch.Generator().manual_seed(SEED + index)
    hyper = torch.randn(1, seq_len, config.hc_count * config.hidden_size, generator=gen).to(torch.bfloat16)

    with torch.no_grad():
        mixed, _, inject = combine(hyper)
        mixed_only = mixer(hyper)
        normed = combine.hc_norm(hyper)

    p = f"case{index}_"
    return {
        p + "seq_len": torch.tensor([seq_len], dtype=torch.int32),
        p + "hyper_input": hyper.contiguous(),
        p + "normed": normed.contiguous(),
        p + "mixed": mixed.contiguous(),
        p + "inject": inject.contiguous(),
        p + "mixed_only": mixed_only.contiguous(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config, config_sha256 = load_text_config(root / CONFIG)

    combine, mixer = build_modules(config)

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


if __name__ == "__main__":
    main()
