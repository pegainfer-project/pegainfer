#!/usr/bin/env python3
"""Generate the PLE (n-gram) block golden at real frozen-config geometry.

Dumps the Qwen3.8-Flash-Next per-layer-embedding reference (`Qwen4ExpTextPLELayer`
+ `Qwen4ExpTextNGramEmbedding`, transformers 5.16.1 `models/qwen4_exp`) into a
fixture a PLE kernel gate replays. All geometry comes from the real frozen config
(`CONFIG`): hidden 2560, `hc_count` 4 (stream 10240), `ple_embed_dim` 2560,
`ngram_size` 3, `heads_per_ngram` 8 (16 heads x 160), and the real per-head prime
vocabularies/offsets/multipliers. The values cover the mul-xor n-gram hash
(collisions by design), the EOS-resetting context shift, the grouped `(1+w)` norms
(group_size 2560), the signed-sqrt gate, and the **dilated** depthwise conv
(dilation = `ngram_size` = 3, state length 9) the non-dilated in-tree conv1d
cannot serve.

Three boundaries are moved because the real tensors cannot be stored:

(a) The n-gram table (~95 GiB bf16) cannot be built, so `nn.Embedding` is capped to
    a tiny stub (`_tiny_embedding`) with the real ids folded in via a recorded
    `remainder`. The captured ids are the REAL ones and are stored (`ngram_ids`);
    the embedding *activation* (a lookup into that stub, weights not stored) is not
    in the fixture -- the `key`/`value` projections it feeds are.
(b) `key_proj` [10240, 2560] and `value_proj` [2560, 2560] weights are NOT stored;
    forward hooks capture their outputs as `key`/`value`.
(c) Otherwise all shapes are real (norms, conv weight, I64 hash buffers, every
    per-case activation); there is no final GEMM -- the block output is terminal.

Cases: S=8 (no EOS) and S=12 (one mid-sequence EOS, the segment reset). Fixture:
one JSON `qwen4_ple_golden` blob (format_version 2, config path + sha256, seed,
versions), the `eos_token_id` scalar, the storeable weights, and per case the
inputs, `rows` slice, hashed ids, recorded projections and expected block output.
Scope: full-sequence (cache=None) only; the decode states (2-token id window and
dilated conv state) need their own dump.

    python tools/accuracy/dump_qwen38fn_ple_golden.py --out <fixture>
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file

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
        module.key_proj.register_forward_hook(capture("key")),
        module.value_proj.register_forward_hook(capture("value")),
    ]

    for i, (seq_len, eos_positions) in enumerate(CASES):
        gen = torch.Generator().manual_seed(SEED + i)
        hidden = torch.randn(1, seq_len, hc_count * hidden_size, generator=gen).to(torch.bfloat16)
        ids_gen = torch.Generator().manual_seed(SEED + i)
        input_ids = torch.randint(1, vocab_size, (1, seq_len), generator=ids_gen)
        # Keep random draws off the EOS id, reserved for the explicit EOS positions.
        input_ids[input_ids == eos_token_id] = eos_token_id + 1
        for eos_pos in eos_positions:
            input_ids[0, eos_pos] = eos_token_id

        with torch.no_grad():
            out = module(hidden, input_ids, None)
        ngram_ids = captured["ids"]
        key_raw = captured["key"]
        value = captured["value"]

        # Row selection: our cases are short, so every row is evaluated, but the
        # convention is kept so a longer case can drop rows without a layout change.
        rows = torch.arange(seq_len, dtype=torch.int32)
        p = f"case{i}_"
        tensors[p + "seq_len"] = torch.tensor([seq_len], dtype=torch.int32)
        tensors[p + "rows"] = rows.contiguous()
        tensors[p + "input_ids"] = input_ids[0].to(torch.int64).contiguous()
        tensors[p + "hidden"] = hidden[0].index_select(0, rows.long()).contiguous()
        tensors[p + "ngram_ids"] = ngram_ids[0].index_select(0, rows.long()).contiguous()
        tensors[p + "key"] = key_raw[0].index_select(0, rows.long()).contiguous()
        tensors[p + "value"] = value[0].index_select(0, rows.long()).contiguous()
        tensors[p + "out"] = out[0].index_select(0, rows.long()).contiguous()
        print(
            f"case{i}: S={seq_len} eos={list(eos_positions)} rows={seq_len} "
            f"heads={ngram_ids.shape[-1]}"
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

if __name__ == "__main__":
    main()
