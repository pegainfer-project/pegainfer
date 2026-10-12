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

The block math **and the n-gram hash** are re-derived inside this script (the
hash in plain Python, independent of the reference) and asserted to match the
module bit-for-bit; the EOS segment reset is additionally asserted as a
property: perturbing the token right before an EOS must not move any embedding
at or after the boundary, and must move the embeddings inside its own segment
(so the reset assert cannot pass vacuously). A mismatch aborts the dump.

Fixture: provenance `format_version` / `config_sha256` / `ref_transformers`
(assert on load); the `eos_token_id` scalar; per case, the inputs (`hidden`,
`input_ids`) and the outputs. A reduced n-gram vocabulary keeps the table
small. Scope boundary: the full-sequence (cache=None) path only -- the decode
continuation states (the 2-token id window and the dilated conv state) are not
captured and need their own dump.

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

from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextNGramEmbedding,
    Qwen4ExpTextPLELayer,
)

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
NGRAM_SEED = 1234             # config.seed: feeds the splitmix hash mixers (layer_multipliers)

SEED = 0x_03_38_F9_5A_11_83

# Bumped when the fixture layout changes; the gate asserts it on load.
FORMAT_VERSION = 1

# (seq_len, eos_positions). One EOS-free case, one mid-sequence EOS, and a
# double EOS that exercises the second reset boundary (cummax across segments).
CASES = [(8, ()), (12, (5,)), (14, (4, 9))]


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
    seed = NGRAM_SEED
    rms_norm_eps = RMS_EPS


def config_digest() -> bytes:
    canon = (
        f"hidden={HIDDEN};hc={HC_COUNT};pledim={PLE_EMBED_DIM};ngram={NGRAM_SIZE};"
        f"heads={HEADS_PER_NGRAM};base={NGRAM_VOCAB_BASE};div={NGRAM_DIVISOR};"
        f"conv={PLE_CONV_KERNEL};vocab={VOCAB_SIZE};eos={EOS_TOKEN_ID};"
        f"eps={RMS_EPS};seed={NGRAM_SEED}"
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


def rederive_embeddings(ng: Qwen4ExpTextNGramEmbedding, input_ids: torch.Tensor) -> torch.Tensor:
    """Re-derive the hashed n-gram embedding in plain Python (no reference code):
    EOS-reset sliding window -> per-order mul-xor mix -> per-head remainder+offset.

    Order o (2..ngram_size) owns heads [(o-2)*heads_per_ngram, (o-1)*heads_per_ngram)
    and mixes window shifts 0..o-1; position p sees token p-k only if it lies
    strictly after the last EOS before p, otherwise the EOS pad. The reference's
    multiplier bound keeps every product below 2**63, so Python ints and torch
    int64 agree bit-for-bit.
    """
    ids = input_ids[0].tolist()
    multipliers = ng.layer_multipliers.tolist()
    sizes = ng.ngram_heads_vocab_sizes.tolist()
    offsets = ng.ngram_heads_offsets.tolist()
    rows = []
    last_eos = -1  # last EOS position strictly before p
    for p in range(len(ids)):
        if p > 0 and ids[p - 1] == EOS_TOKEN_ID:
            last_eos = p - 1
        row = []
        for order in range(2, ng.ngram_size + 1):
            mixed = 0
            for k in range(order):
                q = p - k
                token = ids[q] if 0 <= q and q > last_eos else EOS_TOKEN_ID
                mixed ^= token * multipliers[k]
            base = (order - 2) * ng.heads_per_ngram
            for h in range(ng.heads_per_ngram):
                row.append(mixed % sizes[base + h] + offsets[base + h])
        rows.append(row)
    ngram_ids = torch.tensor(rows, dtype=torch.long).unsqueeze(0)
    return ng.ngram_embedding(ngram_ids).flatten(-2)


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

    # Pin the headline trap: the conv must stay dilated (dilation = ngram_size).
    # rederive() shares module.conv1d and module.short_conv_state_len, so a
    # reference that switched to a consistent non-dilated conv would otherwise
    # sail through every assert below.
    assert module.conv1d.dilation == (NGRAM_SIZE,), "reference conv dilation != ngram_size"
    assert module.short_conv_state_len == (PLE_CONV_KERNEL - 1) * NGRAM_SIZE, (
        "reference short-conv state length != (kernel - 1) * dilation"
    )

    ng = module.ple_embedding
    tensors = {
        "format_version": torch.tensor([FORMAT_VERSION], dtype=torch.int32),
        "config_sha256": bytes_tensor(config_digest()),
        "ref_transformers": bytes_tensor(transformers.__version__.encode()),
        "eos_token_id": torch.tensor([EOS_TOKEN_ID], dtype=torch.int32),
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

    for i, (seq_len, eos_positions) in enumerate(CASES):
        gen = torch.Generator().manual_seed(SEED + i)
        hidden = torch.randn(1, seq_len, HC_COUNT * HIDDEN, generator=gen).to(torch.bfloat16)
        ids_gen = torch.Generator().manual_seed(SEED + i)
        input_ids = torch.randint(1, VOCAB_SIZE, (1, seq_len), generator=ids_gen)
        for eos_pos in eos_positions:
            input_ids[0, eos_pos] = EOS_TOKEN_ID

        with torch.no_grad():
            embeddings = ng(input_ids, None)
            out = module(hidden, input_ids, None)
            rederived = rederive(module, hidden, embeddings)
            rederived_emb = rederive_embeddings(ng, input_ids)

        assert torch.equal(embeddings, rederived_emb), (
            f"case {i} (S={seq_len}) hash re-derivation != module embedding"
        )
        assert torch.equal(out, rederived), f"case {i} (S={seq_len}) block re-derivation != module output"
        assert bool(torch.isfinite(out.float()).all()), f"case {i} (S={seq_len}) non-finite output"

        for eos_pos in eos_positions:
            # EOS resets the n-gram window: perturb the token right before the EOS
            # -- it lies inside the window radius (ngram_size) of position
            # eos_pos+1, so without the segment reset that embedding would move.
            # (Perturbing a farther pre-EOS token proves nothing: its window never
            # reaches the asserted region, reset or not.)
            if eos_pos < 1:
                continue  # a position-0 EOS is indistinguishable from the EOS-padded initial context
            other = input_ids.clone()
            other[0, eos_pos - 1] = (other[0, eos_pos - 1] + 1) % VOCAB_SIZE
            if other[0, eos_pos - 1] == EOS_TOKEN_ID:
                other[0, eos_pos - 1] = (other[0, eos_pos - 1] + 1) % VOCAB_SIZE
            with torch.no_grad():
                embeddings_other = ng(other, None)
            after = slice(eos_pos + 1, seq_len)
            assert torch.equal(embeddings[:, after], embeddings_other[:, after]), (
                f"case {i} (S={seq_len}) embedding at/after EOS {eos_pos} depends on pre-EOS tokens"
            )
            # Positive control: the perturbation must move the embedding inside its
            # own segment, so the reset assert above cannot pass vacuously.
            assert not torch.equal(embeddings[:, : eos_pos + 1], embeddings_other[:, : eos_pos + 1]), (
                f"case {i} (S={seq_len}) perturbed token {eos_pos - 1} changed nothing pre-EOS"
            )

        p = f"case{i}_"
        tensors[p + "seq_len"] = torch.tensor([seq_len], dtype=torch.int32)
        tensors[p + "hidden"] = hidden.contiguous()
        tensors[p + "input_ids"] = input_ids.to(torch.int32).contiguous()
        tensors[p + "ngram_emb"] = embeddings.contiguous()
        tensors[p + "out"] = out.contiguous()
        print(
            f"case{i}: S={seq_len} eos={list(eos_positions)} table={tuple(ng.ngram_embedding.weight.shape)} "
            f"hash_ok=True rederivation_ok=True"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.2f} MiB, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
