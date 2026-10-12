#!/usr/bin/env python3
"""Generate the QSA sparse-attention golden: the full attention sublayer output.

Dumps the Qwen3.8-Flash-Next QSA attention reference (`Qwen4ExpTextAttention`,
transformers 5.16.1 `models/qwen4_exp`) into a fixture a sparse-attention kernel
gate replays: same seeded weights and synthetic hidden states, comparing the
sublayer output, the softmax weights and the effective sparse mask.

This is the sibling of the indexer golden: that one pins *which* tokens are
selected, this one pins the *attend* over them (Q+gate fused projection, per-head
`(1+w)` q/k norm, partial RoPE, GQA repeat, sigmoid output gate).

The reference `forward` builds the QSA mask by calling the indexer, and its mask
builder passes an int32 index to `scatter` (torch wants int64), so `_scatter_int64`
casts it for the reference call only.

Fixture: provenance `format_version` / `config_sha256` / `ref_transformers`
(assert on load); per case, the inputs (`hidden`, `positions`), the outputs
(`attn_out`, `attn_weights`) and the effective `mask` (True = blocked). A reduced
geometry keeps the file small; every structural trap is preserved (partial RoPE,
GQA group > 1, fused Q+gate, sparse mask). Boundary cases shrink `indexer_budget`
to 16 (4 blocks) so block selection truncates without a 2048-token input.

    python tools/accuracy/dump_qwen38fn_qsa_attn_golden.py --out <fixture>
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file

from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextAttention,
    Qwen4ExpTextRotaryEmbedding,
)

# Reduced text geometry (real values in comments); keeps the fixture small while
# preserving every structural trap.
HIDDEN = 512            # real 2560
NUM_Q_HEADS = 4         # real 24
NUM_KV_HEADS = 2        # real 2 (GQA group 2 here, 12 in the real model)
HEAD_DIM = 128          # real 256
PARTIAL_ROTARY = 0.5    # real 0.25; rotary_dim = HEAD_DIM * PARTIAL_ROTARY = 64
RMS_EPS = 1e-6
MAX_POS = 262144
INDEXER_N_HEADS = 4
INDEXER_KV_HEADS = 1
INDEXER_HEAD_DIM = 128
INDEXER_RATIO = 4
# Full budget (never truncates on the small cases).
REAL_BUDGET = 2048
# Reduced budget for the boundary case: 16 // 4 = block_topk 4.
BOUNDARY_BUDGET = 16

SEED = 0x_03_38_F9_5A_11_7F

FORMAT_VERSION = 1

# (seq_len, indexer_budget, expect_sparse). Tail = seq_len % 4 tokens are always
# selected; with a full budget a short sequence keeps every visible block, so only
# the reduced-budget case actually drops keys.
CASES = [
    (8, REAL_BUDGET, False),        # tail 0, no truncation
    (10, REAL_BUDGET, False),       # tail 2, no truncation
    (20, BOUNDARY_BUDGET, True),    # 5 complete blocks > block_topk 4 -> truncate
]


class Config:
    """Minimal config carrying exactly the attributes the attention + rotary read."""

    hidden_size = HIDDEN
    num_attention_heads = NUM_Q_HEADS
    num_key_value_heads = NUM_KV_HEADS
    head_dim = HEAD_DIM
    attention_bias = False
    attention_dropout = 0.0
    rms_norm_eps = RMS_EPS
    max_position_embeddings = MAX_POS
    _attn_implementation = "eager"
    rope_parameters = {
        "rope_type": "default",
        "rope_theta": 1e7,
        "partial_rotary_factor": PARTIAL_ROTARY,
        "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
    }
    indexer_n_heads = INDEXER_N_HEADS
    indexer_kv_heads = INDEXER_KV_HEADS
    indexer_head_dim = INDEXER_HEAD_DIM
    indexer_budget = REAL_BUDGET
    indexer_compress_ratio = INDEXER_RATIO


@contextlib.contextmanager
def _scatter_int64():
    """Let the reference's mask builder run: its scatter passes an int32 index
    (the module builds `selected_token_indices` as int32) which torch requires to
    be int64. Scoped to the reference call; restores the original afterwards."""
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


def causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    """[1, 1, S, S] additive causal mask: 0 where key <= query, finfo.min elsewhere."""
    tri = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    return torch.zeros(1, 1, seq_len, seq_len, dtype=dtype).masked_fill(tri, torch.finfo(dtype).min)


def config_digest() -> bytes:
    canon = (
        f"hidden={HIDDEN};qh={NUM_Q_HEADS};kvh={NUM_KV_HEADS};hd={HEAD_DIM};"
        f"prf={PARTIAL_ROTARY};inh={INDEXER_N_HEADS};inkv={INDEXER_KV_HEADS};"
        f"ind={INDEXER_HEAD_DIM};ratio={INDEXER_RATIO};rope_theta=1e7"
    )
    return hashlib.sha256(canon.encode()).digest()


def bytes_tensor(data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cfg = Config()
    torch.manual_seed(SEED)
    module = Qwen4ExpTextAttention(cfg, 0)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.05)
            elif "layernorm" in name or name.endswith("norm.weight"):
                p.normal_(0.0, 0.5)  # (1+w) norms: non-trivial scale
            else:
                p.normal_(0.0, 0.5)
    module = module.to(torch.bfloat16)
    rotary = Qwen4ExpTextRotaryEmbedding(cfg)

    tensors = {
        "format_version": torch.tensor([FORMAT_VERSION], dtype=torch.int32),
        "config_sha256": bytes_tensor(config_digest()),
        "ref_transformers": bytes_tensor(transformers.__version__.encode()),
        "q_proj": module.q_proj.weight.detach().contiguous(),
        "k_proj": module.k_proj.weight.detach().contiguous(),
        "v_proj": module.v_proj.weight.detach().contiguous(),
        "o_proj": module.o_proj.weight.detach().contiguous(),
        "q_norm": module.q_norm.weight.detach().contiguous(),
        "k_norm": module.k_norm.weight.detach().contiguous(),
        "index_qk_proj": module.indexer.index_qk_proj.weight.detach().contiguous(),
        "q_layernorm": module.indexer.q_layernorm.weight.detach().contiguous(),
        "k_layernorm": module.indexer.k_layernorm.weight.detach().contiguous(),
    }

    for i, (seq_len, budget, expect_sparse) in enumerate(CASES):
        module.indexer.token_budget = budget
        module.indexer.block_topk = budget // module.indexer.compress_ratio

        gen = torch.Generator().manual_seed(SEED + i)
        hidden = torch.randn(1, seq_len, HIDDEN, generator=gen).to(torch.bfloat16)
        positions = torch.arange(seq_len, dtype=torch.long)[None]
        pos_emb = rotary(hidden, positions)
        base_mask = causal_mask(seq_len, torch.bfloat16)

        with torch.no_grad():
            with _scatter_int64():
                attn_out, attn_weights = module(hidden, pos_emb, base_mask, None)
                selected = module.indexer(hidden, pos_emb, base_mask, None)

        # The mask the module actually attended under: causal (+) sparse selection.
        effective = base_mask + selected
        blocked = effective < 0
        causal_blocked = base_mask < 0
        # Everything the mask blocks must carry zero softmax mass, and every row
        # must still sum to 1 over the positions it kept.
        assert torch.equal(attn_weights.masked_fill(~blocked, 0), torch.zeros_like(attn_weights)), (
            f"case {i} (S={seq_len}) attention mass on blocked positions"
        )
        assert torch.allclose(attn_weights.sum(-1).float(), torch.ones(1, NUM_Q_HEADS, seq_len), atol=3e-2), (
            f"case {i} (S={seq_len}) attention rows do not sum to 1"
        )
        # With a reduced budget the selector must drop at least one causally valid
        # key; with a full budget it keeps every visible block (dense).
        dropped = (~causal_blocked) & blocked
        assert bool(dropped.any()) == expect_sparse, (
            f"case {i} (S={seq_len}) sparse={bool(dropped.any())} expected={expect_sparse}"
        )

        p = f"case{i}_"
        tensors[p + "seq_len"] = torch.tensor([seq_len], dtype=torch.int32)
        tensors[p + "budget"] = torch.tensor([budget], dtype=torch.int32)
        tensors[p + "hidden"] = hidden.contiguous()
        tensors[p + "positions"] = positions.to(torch.int32).contiguous()
        tensors[p + "attn_out"] = attn_out.contiguous()
        tensors[p + "attn_weights"] = attn_weights.contiguous()
        tensors[p + "mask"] = blocked.contiguous()
        print(
            f"case{i}: S={seq_len} budget={budget} block_topk={module.indexer.block_topk} "
            f"blocked_frac={blocked.float().mean().item():.3f} mass_ok=True"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.2f} MiB, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
