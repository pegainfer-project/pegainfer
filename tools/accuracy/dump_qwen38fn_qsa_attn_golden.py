#!/usr/bin/env python3
"""Generate the Qwen3.8-Flash-Next QSA sparse full-attention golden.

Dumps one full-attention layer (`Qwen4ExpTextAttention`, transformers 5.16.1
`models/qwen4_exp`) at the real frozen-config geometry: for a handful of query
rows, the captured post-projection activations, the effective sparse+causal mask,
and the reference's attention weights + gated output. Every stored value is a
capture of the reference itself (forward hooks plus its own eager return); nothing
here re-implements it, and the dump only checks that the capture is faithful.

Not stored, and why: the plain `q_proj`/`k_proj`/`v_proj`/`o_proj` GEMMs (98 MB)
-- their outputs are stored instead, and the input to `o_proj` is the golden's
expected output; the indexer's `index_qk_proj`/`q_layernorm`/`k_layernorm` -- the
sparse mask they produce is stored as data and their input (the layer's hidden
state) is not part of this fixture (the indexer has its own oracle,
`qwen38fn-qsa-indexer-golden.safetensors`). The only weights stored are
`q_norm`/`k_norm`, the per-head `(1+w)` RMS norm the kernel applies to the
captured `queries`/`keys`.

Cases: S=8/10/64 evaluate every row; S=2055 evaluates rows 2049..2055, around the
real boundary where the 2048-token budget starts dropping a 4-token block. Only
the evaluated query rows are stored (a full case would carry a ~50 MB activation
and a ~200 MB attention-weight matrix); key/value-side activations stay full
length.

Metadata is a single JSON key (`qwen4_qsa_attn_golden`, format_version 2). The
reference's mask builder passes an int32 index to `Tensor.scatter` (torch wants
int64 on some versions), so `_scatter_int64` casts it for that call only, scoped
-- the dump runs on torch 2.14.1 and older.

    python tools/accuracy/dump_qwen38fn_qsa_attn_golden.py --out test_data/qwen38fn-qsa-attn-golden.safetensors
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
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextAttention,
    Qwen4ExpTextRotaryEmbedding,
)

# Repo root (tools/accuracy/<this file> -> repo root).
ROOT = Path(__file__).resolve().parents[2]
# Frozen checkpoint config: the single source of geometry (no hand-written dims).
CONFIG = "pegainfer-qwen4/tests/frozen_config.json"
SEED = 0x_03_38_51_11_7E
# Bumped when the fixture layout changes; consumers assert it.
FORMAT_VERSION = 2
META_KEY = "qwen4_qsa_attn_golden"
# Any full-attention layer index; the geometry is layer-independent.
LAYER_IDX = 3

# (seq_len, evaluated query rows). `None` -> evaluate every row.
#   S=8/10 : full rows, no truncation.
#   S=64   : full rows, 16 complete blocks (< 512) -> keeps all.
#   S=2055 : the real boundary; evaluate the rows around the event.
CASES = [
    (8, None),
    (10, None),
    (64, None),
    (2055, list(range(2049, 2055))),
]


@contextlib.contextmanager
def _scatter_int64():
    """Let the reference's mask builder run: its scatter passes an int32 index
    (`selected_token_indices` is int32) which torch requires to be int64 on some
    versions. Scoped to the reference call; restores the original afterwards."""
    original = torch.Tensor.scatter

    def patched(self, dim, index, src, *args, **kwargs):
        if torch.is_tensor(index) and index.dtype != torch.int64:
            index = index.long()
        return original(self, dim, index, src, *args, **kwargs)

    torch.Tensor.scatter = patched
    try:
        yield
    finally:
        torch.Tensor.scatter = original


def load_config() -> tuple[Qwen4ExpConfig, str]:
    """Load the frozen text config (real geometry) and hash the raw bytes."""
    raw = (ROOT / CONFIG).read_bytes()
    config = Qwen4ExpConfig(**json.loads(raw)).text_config
    # Force the eager path: it returns attention weights and an additive float mask,
    # matching what the serving kernel consumes.
    config._attn_implementation = "eager"
    return config, hashlib.sha256(raw).hexdigest()


def build_module(config: Qwen4ExpConfig) -> Qwen4ExpTextAttention:
    """Seeded reference attention. Norm weights get a non-trivial (1+w) scale."""
    torch.manual_seed(SEED)
    module = Qwen4ExpTextAttention(config, LAYER_IDX)
    with torch.no_grad():
        # The indexer's norms are perturbed along with the attention's: they are
        # not stored, but they shape the mask this fixture records as data.
        for norm in (
            module.q_norm,
            module.k_norm,
            module.indexer.q_layernorm,
            module.indexer.k_layernorm,
        ):
            norm.weight.normal_(0.0, 0.5)
    return module.to(torch.bfloat16).eval()


def causal_mask(seq_len: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Additive causal mask [1,1,S,S] (0 visible / min_dtype masked) and its bool form."""
    idx = torch.arange(seq_len)
    visible = (idx[None, :] <= idx[:, None])[None, None]  # True where key <= query
    min_dtype = torch.finfo(dtype).min
    mask = torch.where(
        visible, torch.zeros((), dtype=dtype), torch.tensor(min_dtype, dtype=dtype)
    )
    return mask, visible


def dump(args) -> None:
    config, config_sha256 = load_config()
    module = build_module(config)
    rotary = Qwen4ExpTextRotaryEmbedding(config)

    captured: dict[str, torch.Tensor] = {}
    module.q_proj.register_forward_hook(lambda m, i, o: captured.__setitem__("q_act", o))
    module.k_proj.register_forward_hook(lambda m, i, o: captured.__setitem__("k_act", o))
    module.v_proj.register_forward_hook(lambda m, i, o: captured.__setitem__("v_act", o))
    module.o_proj.register_forward_pre_hook(lambda m, i: captured.__setitem__("o_in", i[0]))
    module.indexer.register_forward_hook(lambda m, i, o: captured.__setitem__("indexer_mask", o))

    tensors: dict[str, torch.Tensor] = {
        "q_norm": module.q_norm.weight.detach().contiguous(),
        "k_norm": module.k_norm.weight.detach().contiguous(),
    }

    for i, (seq_len, rows) in enumerate(CASES):
        if rows is None:
            rows = list(range(seq_len))
        row_idx = torch.tensor(rows, dtype=torch.long)

        gen = torch.Generator().manual_seed(SEED + i)
        hidden = torch.randn(1, seq_len, config.hidden_size, generator=gen).to(torch.bfloat16)
        positions = torch.arange(seq_len, dtype=torch.int32)
        pos_emb = rotary(hidden, positions.to(torch.long)[None])
        mask, causal_bool = causal_mask(seq_len, torch.bfloat16)

        captured.clear()
        with torch.no_grad(), _scatter_int64():
            out, attn_weights = module(hidden, pos_emb, mask, None)

        # The captured o_proj input must reproduce the module's returned output.
        assert torch.equal(module.o_proj(captured["o_in"]), out), f"case {i}: o_proj replay mismatch"

        # Effective sparse+causal mask = the reference's own indexer mask AND causal.
        selected_bool = captured["indexer_mask"] == 0
        eff = causal_bool & selected_bool  # [1,1,S,S]

        q_rows = captured["q_act"][0].index_select(0, row_idx).contiguous()
        o_rows = captured["o_in"][0].index_select(0, row_idx).contiguous()
        mask_rows = eff[0, 0].index_select(0, row_idx).contiguous()
        attn_rows = attn_weights[:, :, row_idx, :].contiguous()

        p = f"case{i}_"
        tensors[p + "seq_len"] = torch.tensor([seq_len], dtype=torch.int32)
        tensors[p + "rows"] = row_idx.to(torch.int32).contiguous()
        tensors[p + "positions"] = positions.contiguous()
        tensors[p + "queries"] = q_rows
        tensors[p + "keys"] = captured["k_act"][0].contiguous()
        tensors[p + "values"] = captured["v_act"][0].contiguous()
        tensors[p + "mask"] = mask_rows
        tensors[p + "attn_out"] = o_rows
        tensors[p + "attn_weights"] = attn_rows

        print(f"dump case{i}: S={seq_len} R={len(rows)} attn_out={tuple(o_rows.shape)}")

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    dump(args)


if __name__ == "__main__":
    main()
