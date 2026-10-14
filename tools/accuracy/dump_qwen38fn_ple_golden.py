#!/usr/bin/env python3
"""Generate the PLE (n-gram) block golden at real frozen-config geometry.

Dumps the Qwen3.8-Flash-Next per-layer-embedding reference
(`Qwen4ExpTextPLELayer` + `Qwen4ExpTextNGramEmbedding`, transformers 5.16.1
`models/qwen4_exp`) into a fixture a PLE kernel gate replays. Every dimension
comes from the real frozen checkpoint config (`CONFIG` below): hidden 2560,
`hc_count` 4 (stream width 10240), `ple_embed_dim` 2560, `ngram_size` 3,
`heads_per_ngram` 8 (16 heads, head_dim_per_ngram 160), `ngram_vocab_size_base`
20,000,000, divisor 128.

The traps it keeps covering: the mul-xor n-gram hash with the real per-head prime
vocabularies and offsets (collisions are by design), the EOS-resetting context
shift, the grouped `(1+w)` norms (group_size 2560), the signed-sqrt gate, and the
**dilated** depthwise conv (dilation = `ngram_size` = 3, state length 9), which
the non-dilated in-tree conv1d cannot serve.

Three fixture boundaries are moved because the real tensors cannot be stored;
all three are named in the PR body:

(a) The n-gram table is `320001536 x 160` bf16 (~95 GiB) -- impossible to
    instantiate or commit. `nn.Embedding` is capped to a tiny stub table while the
    module builds (`_tiny_embedding`) and the real ids are folded into it with a
    recorded `remainder`. The **hash -> id** math never touches the table, so the
    captured ids are the REAL ones; the embedding *activation* is a stub-table
    artifact recorded as the reference's output.
(b) `key_proj` [10240, 2560] and `value_proj` [2560, 2560] are ~52 MB + ~13 MB;
    their weights are NOT stored. Their outputs are captured with forward hooks
    and stored instead as `key`/`value` (the merged QSA indexer golden does the
    same for `index_qk_proj`), named so they are not mistaken for those weights.
(c) Otherwise all shapes are real: the stored norms, conv weight, the I64 hash
    buffers (multipliers / per-head offsets / vocab sizes) and every per-case
    activation carry the real geometry. There is no final GEMM -- the block output
    is the fixture's terminal tensor.

The recorded values are the source of truth (captured via forward hooks, never
hand-recomputed). Two cheap assertions abort the dump if the capture drifts: the
captured ids must equal an independent plain-Python re-derivation of the mul-xor
hash (so a stubbed table can never pass off fake ids), and the captured key/value
projections must reproduce the module's returned block output bit-for-bit. The
EOS segment reset is asserted as a property: perturbing the token right before an
EOS must not move any embedding at or after the reset boundary, and must move the
embeddings inside its own segment (positive control, so it cannot pass
vacuously). A mismatch aborts the dump.

Fixture: one JSON `qwen4_ple_golden` metadata blob (format_version 2, config path
+ sha256, seed, versions) instead of loose provenance tensors; the `eos_token_id`
scalar; the storeable weights; and per case the inputs, the real `rows` slice, the
hashed ids, the recorded activations and the expected block output.

Scope boundary: the full-sequence (cache=None) path only -- the decode
continuation states (the 2-token id window and the dilated conv state) are not
captured and need their own dump.

    python tools/accuracy/dump_qwen38fn_ple_golden.py --out <fixture>
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import safe_open, save_file

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

# Frozen checkpoint config. Every geometry below is read from it, not written here.
CONFIG = "pegainfer-qwen4/tests/frozen_config.json"

SEED = 0x_03_38_F9_5A_11_83

# Bumped when the fixture layout changes; the gate asserts it on load.
FORMAT_VERSION = 2

# Real n-gram table would be `padded_vocab_size x 160` (~95 GiB bf16). Cap the stub
# table here; real ids are folded in with `remainder(STUB_VOCAB)` (boundary (a)).
STUB_VOCAB = 4096

# One PLE layer: config.ple_layer_ids == [2] -> layer_idx 1 -> ple_layer_index 0.
LAYER_IDX = 1
PLE_LAYER_INDEX = 0

# (seq_len, eos_positions). S=8 has no EOS (pure local window); S=12 places one
# mid-sequence EOS so the segment reset has a boundary on both sides of it.
CASES = [(8, ()), (12, (5,))]


@contextlib.contextmanager
def _tiny_embedding(cap: int):
    """Cap `nn.Embedding` at `cap` rows so the 95 GiB n-gram table builds.

    Scoped to the module constructor; the reference hash math is untouched, so the
    ids it computes stay real.
    """
    original = torch.nn.Embedding.__init__

    def patched(self, num_embeddings, embedding_dim, *args, **kwargs):
        return original(self, min(num_embeddings, cap), embedding_dim, *args, **kwargs)

    torch.nn.Embedding.__init__ = patched
    try:
        yield
    finally:
        torch.nn.Embedding.__init__ = original


def short_conv(
    hidden_states: torch.Tensor, weight: torch.Tensor, state_len: int, dilation: int
) -> torch.Tensor:
    """`_short_conv` with no cache: pad the dilation-aware state length, then conv."""
    seq_len = hidden_states.shape[1]
    xs = hidden_states.transpose(1, 2)
    xs = F.pad(xs, (state_len, 0))
    xs = xs[..., -(state_len + seq_len) :]
    out = F.conv1d(xs, weight, dilation=dilation, groups=weight.shape[0])
    return F.silu(out).transpose(1, 2)


def grouped_norm(x: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float) -> torch.Tensor:
    """`Qwen4ExpTextRMSNorm`: fp32 grouped norm times the `(1 + w)` scale, back to input dtype."""
    xf = x.reshape(*x.shape[:-1], -1, group_size).float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out.flatten(-2) * (1.0 + weight.float())).type_as(x)


def block_forward(
    hidden_states: torch.Tensor,
    key_raw: torch.Tensor,
    value: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    conv_w: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    state_len: int,
    dilation: int,
    eps: float,
) -> torch.Tensor:
    """The PLE block math from the projection outputs on: norms -> gate -> dilated conv."""
    key = grouped_norm(key_raw, norm_key_w, hidden_size, eps).unflatten(-1, (hc_count, hidden_size))
    query = grouped_norm(hidden_states, norm_query_w, hidden_size, eps).unflatten(-1, (hc_count, hidden_size))
    gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
    convolved = short_conv(grouped_norm(gated, norm_conv_w, hidden_size, eps), conv_w, state_len, dilation)
    return gated + convolved


def rederive_ngram_ids(ng, input_ids: torch.Tensor) -> torch.Tensor:
    """Independent plain-Python re-derivation of the mul-xor hash (`[1, S, heads]`).

    EOS-reset sliding window -> per-order mul-xor mix -> per-head remainder + offset.
    Order o (2..ngram_size) owns heads [(o-2)*hpn, (o-1)*hpn) and mixes window shifts
    0..o-1; position p sees token p-k only if it lies strictly after the last EOS
    before p, otherwise the EOS pad. The reference's multiplier bound keeps every
    product below 2**63, so Python ints and torch int64 agree bit-for-bit.
    """
    ids = input_ids[0].tolist()
    multipliers = ng.layer_multipliers.tolist()
    sizes = ng.ngram_heads_vocab_sizes.tolist()
    offsets = ng.ngram_heads_offsets.tolist()
    eos = ng.eos_token_id
    rows = []
    last_eos = -1  # last EOS position strictly before p
    for p in range(len(ids)):
        if p > 0 and ids[p - 1] == eos:
            last_eos = p - 1
        row = []
        for order in range(2, ng.ngram_size + 1):
            mixed = 0
            for k in range(order):
                q = p - k
                token = ids[q] if 0 <= q and q > last_eos else eos
                mixed ^= token * multipliers[k]
            base = (order - 2) * ng.heads_per_ngram
            for h in range(ng.heads_per_ngram):
                row.append(mixed % sizes[base + h] + offsets[base + h])
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long).unsqueeze(0)


def _perturb(input_ids: torch.Tensor, pos: int, eos: int, vocab_size: int) -> torch.Tensor:
    """Flip token `pos` to a different, non-EOS id."""
    other = input_ids.clone()
    new = (int(other[0, pos]) + 1) % vocab_size
    if new == eos:
        new = (new + 1) % vocab_size
    other[0, pos] = new
    assert int(other[0, pos]) != int(input_ids[0, pos])
    return other


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    raw = (root / CONFIG).read_bytes()
    config_sha256 = hashlib.sha256(raw).hexdigest()
    cfg = Qwen4ExpConfig(**json.loads(raw)).text_config

    hidden_size = cfg.hidden_size
    hc_count = cfg.hc_count
    vocab_size = cfg.vocab_size
    eos_token_id = cfg.eos_token_id
    eps = cfg.rms_norm_eps

    torch.manual_seed(SEED)
    with _tiny_embedding(STUB_VOCAB):
        module = Qwen4ExpTextPLELayer(cfg, LAYER_IDX, PLE_LAYER_INDEX)
    with torch.no_grad():
        # RMSNorm weights default to zero; give every weight a non-trivial value so a
        # kernel that ignored one of the `(1 + w)` scales would not sail through.
        for p in module.parameters():
            p.normal_(0.0, 0.05 if p.dim() >= 2 else 0.5)
    module = module.to(torch.bfloat16)
    ng = module.ple_embedding

    # Pin the headline trap: the conv must stay dilated (dilation = ngram_size).
    assert ng.ngram_embedding.num_embeddings == STUB_VOCAB, "stub embedding table not capped"
    assert module.conv1d.dilation == (cfg.ngram_size,), "reference conv dilation != ngram_size"
    assert module.conv1d.groups == hc_count * hidden_size, "reference conv is not depthwise"
    assert module.short_conv_state_len == (cfg.ple_conv_kernel_size - 1) * cfg.ngram_size, (
        "reference short-conv state length != (kernel - 1) * dilation"
    )

    tensors = {
        "eos_token_id": torch.tensor([eos_token_id], dtype=torch.int64),
        "layer_multipliers": ng.layer_multipliers.detach().contiguous(),
        "ngram_heads_offsets": ng.ngram_heads_offsets.detach().contiguous(),
        "ngram_heads_vocab_sizes": ng.ngram_heads_vocab_sizes.detach().contiguous(),
        "norm_key": module.norm_key.weight.detach().contiguous(),
        "norm_query": module.norm_query.weight.detach().contiguous(),
        "norm_conv": module.norm_conv.weight.detach().contiguous(),
        "conv1d_weight": module.conv1d.weight.detach().contiguous(),
    }

    # Forward hooks capture exactly what the reference computes (boundaries a + b).
    captured: dict[str, torch.Tensor] = {}

    def capture_ids(cap: int):
        def hook(_mod, inputs):
            ids = inputs[0]
            captured["ids"] = ids.detach().clone()
            return (ids.remainder(cap),)  # fold real ids into the stub table

        return hook

    def capture(name: str):
        def hook(_mod, _inputs, output):
            captured[name] = output.detach().clone()

        return hook

    handles = [
        ng.ngram_embedding.register_forward_pre_hook(capture_ids(STUB_VOCAB)),
        ng.register_forward_hook(capture("emb")),
        module.key_proj.register_forward_hook(capture("key")),
        module.value_proj.register_forward_hook(capture("value")),
    ]

    for i, (seq_len, eos_positions) in enumerate(CASES):
        gen = torch.Generator().manual_seed(SEED + i)
        hidden = torch.randn(1, seq_len, hc_count * hidden_size, generator=gen).to(torch.bfloat16)
        ids_gen = torch.Generator().manual_seed(SEED + i)
        input_ids = torch.randint(1, vocab_size, (1, seq_len), generator=ids_gen)
        # Keep random draws from colliding with the EOS the reset assert keys on.
        input_ids[input_ids == eos_token_id] = eos_token_id + 1
        for eos_pos in eos_positions:
            input_ids[0, eos_pos] = eos_token_id

        with torch.no_grad():
            out = module(hidden, input_ids, None)
        ngram_ids = captured["ids"]
        emb = captured["emb"]
        key_raw = captured["key"]
        value = captured["value"]

        # Source of truth is the hook capture; this only aborts the dump if the
        # captured ids stopped being the real mul-xor hash.
        rederived_ids = rederive_ngram_ids(ng, input_ids)
        assert torch.equal(ngram_ids, rederived_ids), (
            f"case {i} (S={seq_len}) captured n-gram ids != independent hash re-derivation"
        )
        # Captured projections must reproduce the module's returned block output.
        rederived = block_forward(
            hidden,
            key_raw,
            value,
            module.norm_key.weight,
            module.norm_query.weight,
            module.norm_conv.weight,
            module.conv1d.weight,
            hc_count,
            hidden_size,
            module.short_conv_state_len,
            cfg.ngram_size,
            eps,
        )
        assert torch.equal(out, rederived), f"case {i} (S={seq_len}) captured key/value != module output"
        assert bool(torch.isfinite(out.float()).all()), f"case {i} (S={seq_len}) non-finite output"

        for eos_pos in eos_positions:
            # EOS resets the n-gram window: perturb the token right before the EOS
            # -- it sits inside the window radius (ngram_size) of position
            # eos_pos + 1, so without the segment reset that embedding would move.
            other = _perturb(input_ids, eos_pos - 1, eos_token_id, vocab_size)
            with torch.no_grad():
                emb_other = ng(other, None)
            after = slice(eos_pos + 1, seq_len)
            assert torch.equal(emb[:, after], emb_other[:, after]), (
                f"case {i} (S={seq_len}) embedding at/after EOS {eos_pos} depends on pre-EOS tokens"
            )
            # Positive control: the perturbation must move the embedding inside its
            # own segment, so the reset assert above cannot pass vacuously.
            assert not torch.equal(emb[:, : eos_pos + 1], emb_other[:, : eos_pos + 1]), (
                f"case {i} (S={seq_len}) perturbed token {eos_pos - 1} changed nothing pre-EOS"
            )

        # Row selection: our cases are short, so every row is evaluated, but the
        # convention is kept so a longer case can drop rows without a layout change.
        rows = torch.arange(seq_len, dtype=torch.int32)
        p = f"case{i}_"
        tensors[p + "seq_len"] = torch.tensor([seq_len], dtype=torch.int32)
        tensors[p + "rows"] = rows.contiguous()
        tensors[p + "input_ids"] = input_ids[0].to(torch.int64).contiguous()
        tensors[p + "hidden"] = hidden[0].index_select(0, rows.long()).contiguous()
        tensors[p + "ngram_ids"] = ngram_ids[0].index_select(0, rows.long()).contiguous()
        tensors[p + "ngram_emb"] = emb[0].index_select(0, rows.long()).contiguous()
        tensors[p + "key"] = key_raw[0].index_select(0, rows.long()).contiguous()
        tensors[p + "value"] = value[0].index_select(0, rows.long()).contiguous()
        tensors[p + "out"] = out[0].index_select(0, rows.long()).contiguous()
        print(
            f"case{i}: S={seq_len} eos={list(eos_positions)} rows={seq_len} "
            f"heads={ngram_ids.shape[-1]} hash_ok=True block_ok=True reset_ok=True"
        )

    for h in handles:
        h.remove()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": FORMAT_VERSION,
        "config": CONFIG,
        "config_sha256": config_sha256,
        "seed": SEED,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "platform": platform.machine(),
    }
    save_file(tensors, str(args.out), metadata={"qwen4_ple_golden": json.dumps(manifest, sort_keys=True)})
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.2f} MiB, {len(tensors)} tensors)")

    verify(args.out, eps)


def verify(path: Path, eps: float) -> None:
    """Reload the fixture and reproduce every reproducible output bit-for-bit."""
    with safe_open(str(path), framework="pt") as handle:
        manifest = json.loads(handle.metadata()["qwen4_ple_golden"])
        flat = {name: handle.get_tensor(name) for name in handle.keys()}
    assert manifest["format_version"] == FORMAT_VERSION, "fixture format_version mismatch"

    ngram_size = flat["layer_multipliers"].shape[0]
    heads_total = flat["ngram_heads_vocab_sizes"].shape[0]
    heads_per_ngram = heads_total // (ngram_size - 1)
    conv_w = flat["conv1d_weight"]
    dilation = ngram_size
    state_len = (conv_w.shape[-1] - 1) * dilation
    eos = int(flat["eos_token_id"][0])

    def hash_ids(input_ids: torch.Tensor) -> list[list[int]]:
        ids = input_ids.tolist()
        mult = flat["layer_multipliers"].tolist()
        sizes = flat["ngram_heads_vocab_sizes"].tolist()
        offsets = flat["ngram_heads_offsets"].tolist()
        rows, last_eos = [], -1
        for pos in range(len(ids)):
            if pos > 0 and ids[pos - 1] == eos:
                last_eos = pos - 1
            row = []
            for order in range(2, ngram_size + 1):
                mixed = 0
                for k in range(order):
                    q = pos - k
                    token = ids[q] if 0 <= q and q > last_eos else eos
                    mixed ^= token * mult[k]
                base = (order - 2) * heads_per_ngram
                for h in range(heads_per_ngram):
                    row.append(mixed % sizes[base + h] + offsets[base + h])
            rows.append(row)
        return rows

    for i, (seq_len, eos_positions) in enumerate(CASES):
        p = f"case{i}_"
        rows = flat[p + "rows"].long()
        hidden = flat[p + "hidden"][None]
        key_raw = flat[p + "key"][None]
        value = flat[p + "value"][None]
        hidden_size = value.shape[-1]
        hc_count = key_raw.shape[-1] // hidden_size

        hash_ok = torch.equal(
            torch.tensor(hash_ids(flat[p + "input_ids"]), dtype=torch.long).index_select(0, rows),
            flat[p + "ngram_ids"],
        )
        reproduced = block_forward(
            hidden,
            key_raw,
            value,
            flat["norm_key"],
            flat["norm_query"],
            flat["norm_conv"],
            conv_w,
            hc_count,
            hidden_size,
            state_len,
            dilation,
            eps,
        )[0].index_select(0, rows)
        block_ok = torch.equal(reproduced, flat[p + "out"])
        assert hash_ok and block_ok, f"case {i} verify failed: hash_ok={hash_ok} block_ok={block_ok}"
        print(f"case{i}: S={seq_len} eos={list(eos_positions)} hash_ok={hash_ok} block_bitwise_ok={block_ok}")


if __name__ == "__main__":
    main()
