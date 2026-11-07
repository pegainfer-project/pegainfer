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
S=6/10.

v3 adds the scatter half (`hyper_input + inject * sublayer`, once after attention and once
after the MLP). It is recorded from `Qwen4ExpTextDecoderLayer.forward` itself: the layer is
built with `__new__` (no weight allocation), the two real gated-residual modules sit on it,
and the attention/MLP sublayers are stubs that return fixed bf16 tensors — so the recorded
values are the reference's own two scatter lines given those inputs, not a re-derivation.
The manifest is one JSON metadata key (`qwen4_hc_golden`).

    python tools/accuracy/dump_qwen38fn_hc_golden.py --out test_data/qwen38fn-hc-golden.safetensors
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedResidual,
)

CONFIG = "pegainfer-qwen4/tests/frozen_config.json"
METADATA_KEY = "qwen4_hc_golden"

SEED = 0x_03_38_F9_5A_11_81

# Bumped when the fixture layout changes; the gate asserts it on load. v2 records the
# manifest as JSON metadata (no loose provenance tensors) and uses real geometry; v3 adds
# the scatter half recorded through the reference decoder layer's own forward.
FORMAT_VERSION = 3

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


class _Stub(torch.nn.Module):
    """A sublayer that returns a fixed tensor: the scatter's input, not a component
    under test. The layer's own forward still runs the scatter lines."""

    def __init__(self, out: torch.Tensor):
        super().__init__()
        self.out = out

    def forward(self, *args, **kwargs):
        return self.out


def scatter_tensors(combine, config, index: int) -> dict[str, torch.Tensor]:
    """Record both scatter steps of a decoder layer on a fixed sublayer output.

    The layer is instantiated without ``__init__`` so no weights are allocated; the
    two gated-residual modules are real and carry the recorded weights, and the
    attention/MLP sublayers are stubs. The captured values are what
    ``Qwen4ExpTextDecoderLayer.forward`` computes for the scatter lines.
    """
    seq_len = CASES[index]
    hidden = config.hidden_size
    stream = config.hc_count * hidden
    # The scatter runs on the case's own recorded `hyper_input`, drawn exactly as
    # `case_tensors` draws it, so the fixture stores one stream per case.
    hyper_gen = torch.Generator().manual_seed(SEED + index)
    hyper = torch.randn(1, seq_len, stream, generator=hyper_gen).to(torch.bfloat16)
    gen = torch.Generator().manual_seed(SEED + 0x100 + index)
    attn_out = torch.randn(1, seq_len, hidden, generator=gen).to(torch.bfloat16)
    mlp_out = torch.randn(1, seq_len, hidden, generator=gen).to(torch.bfloat16)

    hc_attn = combine
    hc_mlp = copy.deepcopy(combine)
    layer = Qwen4ExpTextDecoderLayer.__new__(Qwen4ExpTextDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.ple = None
    layer.layer_type = "linear_attention"
    layer.attn_hyper_connection = hc_attn
    layer.mlp_hyper_connection = hc_mlp
    layer.linear_attn = _Stub(attn_out)
    layer.self_attn = _Stub(attn_out)
    layer.mlp = _Stub(mlp_out)

    captured: dict[str, torch.Tensor] = {}

    def capture_inject(name):
        def hook(_module, _inputs, output):
            captured[name] = output[2].detach().clone()

        return hook

    def capture_input(name):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach().clone()

        return hook

    hc_attn.register_forward_hook(capture_inject("attn_inject"))
    hc_mlp.register_forward_hook(capture_inject("mlp_inject"))
    hc_mlp.register_forward_pre_hook(capture_input("attn_scattered"))

    with torch.no_grad():
        mlp_scattered = layer.forward(
            hyper, position_embeddings=(torch.empty(0), torch.empty(0))
        )

    p = f"case{index}_"
    return {
        p + "scatter_attn_sublayer": attn_out.contiguous(),
        p + "scatter_attn_inject": captured["attn_inject"].contiguous(),
        p + "scatter_attn_out": captured["attn_scattered"].contiguous(),
        p + "scatter_mlp_sublayer": mlp_out.contiguous(),
        p + "scatter_mlp_inject": captured["mlp_inject"].contiguous(),
        p + "scatter_mlp_out": mlp_scattered.detach().contiguous(),
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
    for index in range(len(CASES)):
        tensors.update(scatter_tensors(combine, config, index))

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
