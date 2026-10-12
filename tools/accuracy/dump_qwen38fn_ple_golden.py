#!/usr/bin/env python3
"""Generate the PLE (n-gram) block golden.

Dumps the Qwen3.8-Flash-Next per-layer-embedding reference
(`Qwen4ExpTextPLELayer` + `Qwen4ExpTextNGramEmbedding`, transformers 5.16.1
`models/qwen4_exp`) into a fixture a PLE kernel gate replays: same seeded weights
and synthetic hidden states / token ids, comparing the hashed n-gram embedding and
the injected block output.

The traps it pins: the mul-xor n-gram hash with per-head prime vocabularies and
offsets (collisions are by design), the EOS-resetting context shift, the grouped
`(1+w)` norms, the signed-sqrt gate, and a **dilated** depthwise conv (dilation =
`ngram_size`), which the non-dilated in-tree conv1d cannot serve.

The block math is re-derived inside this script and asserted to match the module
bit-for-bit, and the EOS segment reset is asserted directly (tokens before an EOS
must not move the embedding after it). A mismatch aborts the dump.

Fixture: provenance `format_version` / `config_sha256` / `ref_transformers`
(assert on load); per case, the inputs (`hidden`, `input_ids`) and the outputs. A
reduced n-gram vocabulary keeps the table small.

    python tools/accuracy/dump_qwen38fn_ple_golden.py --out <fixture>
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import save_file

from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

# Reduced text geometry (real values in comments).
HIDDEN = 512          # real 2560
HC_COUNT = 4          # real 4 (the 10240-wide stream)
PLE_EMBED_DIM = 512   # real 2560
NGRAM_SIZE = 3
HEADS_PER_NGRAM = 8   # 16 heads = 2 orders x 8
NGRAM_VOCAB_BASE = 200        # real 20_000_000
NGRAM_DIVISOR = 128           # real 128
PLE_CONV_KERNEL = 4
VOCAB_SIZE = 1024             # real 248320
EOS_TOKEN_ID = 0
RMS_EPS = 1e-6

SEED = 0x_03_38_F9_5A_11_83

FORMAT_VERSION = 1

# (seq_len, eos_position). EOS is placed mid-sequence to exercise the reset.
CASES = [(8, -1), (12, 5)]


class Config:
    """Minimal config carrying exactly the attributes the PLE block reads."""

    hidden_size = HIDDEN
    hc_count = HC_COUNT
    ple_embed_dim = PLE_EMBED_DIM
    ngram_size = NGRAM_SIZE
    heads_per_ngram = HEADS_PER_NGRAM
    ngram_vocab_size_base = NGRAM_VOCAB_BASE
    make_ngram_vocab_size_divisible_by = NGRAM_DIVISOR
    ple_conv_kernel_size = PLE_CONV_KERNEL
    vocab_size = VOCAB_SIZE
    eos_token_id = EOS_TOKEN_ID
    seed = 1234
    rms_norm_eps = RMS_EPS
    layer_types = ["linear_attention"] * 4


def config_digest() -> bytes:
    canon = (
        f"hidden={HIDDEN};hc={HC_COUNT};pledim={PLE_EMBED_DIM};ngram={NGRAM_SIZE};"
        f"heads={HEADS_PER_NGRAM};base={NGRAM_VOCAB_BASE};div={NGRAM_DIVISOR};"
        f"conv={PLE_CONV_KERNEL};vocab={VOCAB_SIZE}"
    )
    return hashlib.sha256(canon.encode()).digest()


def bytes_tensor(data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


def short_conv(module: Qwen4ExpTextPLELayer, x: torch.Tensor) -> torch.Tensor:
    """`_short_conv` with no cache: pad the state length (dilation-aware) then conv."""
    seq_len = x.shape[1]
    xs = x.transpose(1, 2)
    xs = F.pad(xs, (module.short_conv_state_len, 0))
    xs = xs[..., -(module.short_conv_state_len + seq_len) :]
    return F.silu(module.conv1d(xs)).transpose(1, 2)


def rederive(module: Qwen4ExpTextPLELayer, hidden: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    """Re-derive the block forward: grouped norms -> signed-sqrt gate -> dilated conv."""
    key = module.norm_key(module.key_proj(embeddings)).unflatten(-1, (module.hc_count, module.hidden_size))
    value = module.value_proj(embeddings)
    query = module.norm_query(hidden).unflatten(-1, (module.hc_count, module.hidden_size))
    gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(module.hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated = gated.flatten(-2)
    return gated + short_conv(module, module.norm_conv(gated))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cfg = Config()
    torch.manual_seed(SEED)
    module = Qwen4ExpTextPLELayer(cfg, 1, 0)
    with torch.no_grad():
        for name, p in module.named_parameters():
            p.normal_(0.0, 0.05 if p.dim() >= 2 else 0.5)
    module = module.to(torch.bfloat16)

    ng = module.ple_embedding
    tensors = {
        "format_version": torch.tensor([FORMAT_VERSION], dtype=torch.int32),
        "config_sha256": bytes_tensor(config_digest()),
        "ref_transformers": bytes_tensor(transformers.__version__.encode()),
        "key_proj": module.key_proj.weight.detach().contiguous(),
        "value_proj": module.value_proj.weight.detach().contiguous(),
        "norm_key": module.norm_key.weight.detach().contiguous(),
        "norm_query": module.norm_query.weight.detach().contiguous(),
        "norm_conv": module.norm_conv.weight.detach().contiguous(),
        "conv1d_weight": module.conv1d.weight.detach().contiguous(),
        "ngram_embedding": ng.ngram_embedding.weight.detach().contiguous(),
        "layer_multipliers": ng.layer_multipliers.detach().contiguous(),
        "ngram_heads_offsets": ng.ngram_heads_offsets.detach().contiguous(),
        "ngram_heads_vocab_sizes": ng.ngram_heads_vocab_sizes.detach().contiguous(),
    }

    for i, (seq_len, eos_pos) in enumerate(CASES):
        gen = torch.Generator().manual_seed(SEED + i)
        hidden = torch.randn(1, seq_len, HC_COUNT * HIDDEN, generator=gen).to(torch.bfloat16)
        ids_gen = torch.Generator().manual_seed(SEED + i)
        input_ids = torch.randint(1, VOCAB_SIZE, (1, seq_len), generator=ids_gen)
        if eos_pos >= 0:
            input_ids[0, eos_pos] = EOS_TOKEN_ID

        with torch.no_grad():
            embeddings = ng(input_ids, None)
            out = module(hidden, input_ids, None)
            rederived = rederive(module, hidden, embeddings)

        assert torch.equal(out, rederived), f"case {i} (S={seq_len}) re-derivation != module output"
        assert bool(torch.isfinite(out.float()).all()), f"case {i} (S={seq_len}) non-finite output"

        if eos_pos >= 0:
            # EOS resets the n-gram window: perturbing a pre-EOS token must not move
            # any embedding at or after the reset boundary segment.
            other = input_ids.clone()
            other[0, 0] = (other[0, 0] + 1) % VOCAB_SIZE
            if other[0, 0] == EOS_TOKEN_ID:
                other[0, 0] = (other[0, 0] + 1) % VOCAB_SIZE
            with torch.no_grad():
                embeddings_other = ng(other, None)
            after = slice(eos_pos + 1, seq_len)
            assert torch.equal(embeddings[:, after], embeddings_other[:, after]), (
                f"case {i} (S={seq_len}) embedding after EOS depends on pre-EOS tokens"
            )

        p = f"case{i}_"
        tensors[p + "seq_len"] = torch.tensor([seq_len], dtype=torch.int32)
        tensors[p + "hidden"] = hidden.contiguous()
        tensors[p + "input_ids"] = input_ids.to(torch.int32).contiguous()
        tensors[p + "ngram_emb"] = embeddings.contiguous()
        tensors[p + "out"] = out.contiguous()
        print(
            f"case{i}: S={seq_len} eos_pos={eos_pos} table={tuple(ng.ngram_embedding.weight.shape)} "
            f"rederivation_ok=True"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.2f} MiB, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
