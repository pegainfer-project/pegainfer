#!/usr/bin/env python3
"""Generate the QSA indexer golden: per-block scores + selected token indices.

Dumps the Qwen3.8-Flash-Next QSA token indexer reference
(`Qwen4ExpTextQSAIndexer`, transformers 5.16.1 `models/qwen4_exp`) into a fixture
the scoring/selection kernel gate replays: same seeded weights and synthetic
hidden states, comparing per-block scores and selected token indices.

The reference `forward` returns only the attention mask, so the score/selection
internals are re-derived here and validated by asserting the re-derivation
reproduces the module's mask on every case (a mismatch aborts). Its mask builder
also passes an int32 index to `scatter` (torch wants int64), so `_scatter_int64`
casts it for the reference call only.

Fixture: provenance `format_version` / `config_sha256` / `ref_transformers`
(assert on load); per case, `scores` is NaN-padded with an explicit
`scores_valid` mask (select by the mask, not the padding) and `selected` is
-1-padded with `n_selected`. Boundary cases shrink `indexer_budget` to 16 (4
blocks) to hit the truncation branch without a 2048-token input.

    python tools/accuracy/dump_qwen38fn_qsa_indexer_golden.py --out <fixture>
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
    Qwen4ExpTextQSAIndexer,
    Qwen4ExpTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

# Qwen3.8-Flash-Next text geometry.
HIDDEN = 2560
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
INDEXER_N_HEADS = 4
INDEXER_KV_HEADS = 1
INDEXER_HEAD_DIM = 128
INDEXER_RATIO = 4
RMS_EPS = 1e-6
MAX_POS = 262144
# Full budget (real) -- never truncates on the small cases below.
REAL_BUDGET = 2048
# Reduced budget for the boundary cases: 16 // 4 = block_topk 4, so
# n_complete crosses 4 at seq_len 16 without a 2048-token input.
BOUNDARY_BUDGET = 16

SEED = 0x_03_38_F9_5A_11_7E

# Bumped when the fixture layout changes; the gate asserts it on load.
FORMAT_VERSION = 1

# (seq_len, indexer_budget). Tail = seq_len % 4 tokens are always kept; the
# boundary budget exercises topk(min(budget//ratio, n_complete)) under/at/over.
CASES = [
    (5, REAL_BUDGET),      # tail 1, no truncation
    (8, REAL_BUDGET),      # tail 0, no truncation
    (64, REAL_BUDGET),     # tail 0, 16 complete blocks (< 512, keep all)
    (14, BOUNDARY_BUDGET),  # 3 complete blocks (< 4) + tail 2
    (16, BOUNDARY_BUDGET),  # at block_topk (4), tail 0
    (20, BOUNDARY_BUDGET),  # 5 complete blocks (> 4) -> truncate to 4
]


class Config:
    """Minimal config carrying exactly the attributes the indexer + rotary read."""

    hidden_size = HIDDEN
    num_attention_heads = NUM_Q_HEADS
    num_key_value_heads = NUM_KV_HEADS
    head_dim = HEAD_DIM
    rms_norm_eps = RMS_EPS
    max_position_embeddings = MAX_POS
    rope_parameters = {
        "rope_type": "default",
        "rope_theta": 1e7,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
    }
    indexer_n_heads = INDEXER_N_HEADS
    indexer_kv_heads = INDEXER_KV_HEADS
    indexer_head_dim = INDEXER_HEAD_DIM
    indexer_budget = REAL_BUDGET
    indexer_compress_ratio = INDEXER_RATIO


def causal_mask(seq_len: int) -> torch.Tensor:
    """[1, 1, S, S] bool causal mask; True where key <= query (the reference's visible set)."""
    idx = torch.arange(seq_len)
    return (idx[None, :] <= idx[:, None])[None, None]


@contextlib.contextmanager
def _scatter_int64():
    """Let the reference's mask builder run: its scatter passes an int32 index
    (`modeling` builds `selected_token_indices` as int32) which torch requires to
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


def rederive(
    module: Qwen4ExpTextQSAIndexer,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Re-derive indexer scores + selected tokens.

    Returns (scores [S, ceil(S/4)] fp32 NaN-padded, selected [S, budget+ratio-1]
    int32 -1-padded, n_blocks [S]). Mirrors the reference forward, calling the
    module's own submodules for the numerics.
    """
    batch_size, seq_len, _ = hidden.shape
    hidden_shape = (batch_size, seq_len, -1, module.index_head_dim)
    full_cos, full_sin = position_embeddings
    current_cos, current_sin = full_cos[:, -seq_len:, :], full_sin[:, -seq_len:, :]

    qk = module.index_qk_proj(hidden)
    q, token_k = torch.split(
        qk,
        [module.index_n_heads * module.index_head_dim, module.index_kv_heads * module.index_head_dim],
        dim=-1,
    )
    q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
    q = module.q_layernorm(q)
    q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)

    visible_token_indices = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    max_blocks = (seq_len + module.compress_ratio - 1) // module.compress_ratio
    width = module.token_budget + module.compress_ratio - 1
    selected = torch.full((batch_size, seq_len, width), -1, dtype=torch.int32)
    scores_out = torch.full((batch_size, seq_len, max_blocks), float("nan"), dtype=torch.float32)
    n_blocks = torch.zeros((batch_size, seq_len), dtype=torch.int32)

    for b in range(batch_size):
        for qi in range(seq_len):
            local_visible = torch.nonzero(visible_token_indices[b, 0, qi], as_tuple=False).flatten()
            n_complete = int(local_visible.shape[-1]) // module.compress_ratio
            n_blocks[b, qi] = n_complete
            if n_complete > 0:
                block_tokens = local_visible[: n_complete * module.compress_ratio].view(n_complete, module.compress_ratio)
                key_groups = raw_keys[b].index_select(0, block_tokens.flatten())
                key_groups = key_groups.view(*block_tokens.shape, module.index_head_dim)
                pooled = key_groups.float().mean(dim=1).to(raw_keys.dtype)
                pooled = module.k_layernorm(pooled)
                group_starts = block_tokens[:, 0]
                block_keys = apply_rotary_pos_emb(
                    pooled.unsqueeze(1),
                    cos=full_cos[b].index_select(0, group_starts),
                    sin=full_sin[b].index_select(0, group_starts),
                ).squeeze(1)
                scores = torch.matmul(
                    q[b, qi].float(), block_keys.float().transpose(-1, -2)
                ).transpose(-1, -2)
                scores = torch.relu(scores).sum(dim=-1) / (module.index_head_dim**0.5)
                scores_out[b, qi, :n_complete] = scores.float()
                sel = scores.topk(min(module.block_topk, n_complete), dim=0).indices
                selected_tokens = block_tokens.index_select(0, sel).flatten()
            else:
                selected_tokens = torch.tensor([], dtype=torch.long)
            tail = local_visible[n_complete * module.compress_ratio :]
            selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
            selected[b, qi, : selected_tokens.numel()] = selected_tokens

    return scores_out, selected, n_blocks


def selected_mask(selected: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Rebuild the reference's returned mask [b, 1, S, S] bool from selected indices."""
    b, s, _ = selected.shape
    mask = torch.zeros((b, s, seq_len + 1), dtype=torch.bool)
    scatter = torch.where(selected >= 0, selected, torch.full_like(selected, seq_len)).long()
    return mask.scatter(-1, scatter, True)[..., :seq_len].unsqueeze(1)


def config_digest() -> bytes:
    """sha256 over the frozen geometry + norm eps so a consumer can assert this oracle's config."""
    canon = (
        f"hidden={HIDDEN};qh={NUM_Q_HEADS};kvh={NUM_KV_HEADS};hd={HEAD_DIM};"
        f"inh={INDEXER_N_HEADS};inkv={INDEXER_KV_HEADS};ind={INDEXER_HEAD_DIM};"
        f"ratio={INDEXER_RATIO};budget={REAL_BUDGET};rope_theta=1e7;prf=0.25;"
        f"eps={RMS_EPS}"
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
    module = Qwen4ExpTextQSAIndexer(cfg, 0)
    with torch.no_grad():
        # RMSNorm weights default to zero; give them a non-trivial (1+w) scale.
        module.q_layernorm.weight.normal_(0.0, 0.5)
        module.k_layernorm.weight.normal_(0.0, 0.5)
    module = module.to(torch.bfloat16)
    rotary = Qwen4ExpTextRotaryEmbedding(cfg)

    tensors = {
        "format_version": torch.tensor([FORMAT_VERSION], dtype=torch.int32),
        "config_sha256": bytes_tensor(config_digest()),
        "ref_transformers": bytes_tensor(transformers.__version__.encode()),
        "index_qk_proj": module.index_qk_proj.weight.detach().contiguous(),
        "q_layernorm": module.q_layernorm.weight.detach().contiguous(),
        "k_layernorm": module.k_layernorm.weight.detach().contiguous(),
    }

    for i, (seq_len, budget) in enumerate(CASES):
        module.token_budget = budget
        module.block_topk = budget // module.compress_ratio

        gen = torch.Generator().manual_seed(SEED + i)
        hidden = torch.randn(1, seq_len, HIDDEN, generator=gen).to(torch.bfloat16)
        positions = torch.arange(seq_len, dtype=torch.int32)
        pos_emb = rotary(hidden, positions.to(torch.long)[None])
        attn = causal_mask(seq_len)

        with _scatter_int64():
            ref_mask = module(hidden, pos_emb, attn, None)
        scores, selected, n_blocks = rederive(module, hidden, pos_emb, attn)
        my_mask = selected_mask(selected, seq_len)
        assert torch.equal(ref_mask, my_mask), f"case {i} (S={seq_len}) re-derivation != reference mask"
        n_sel = (selected >= 0).sum(dim=-1).to(torch.int32)

        p = f"case{i}_"
        tensors[p + "seq_len"] = torch.tensor([seq_len], dtype=torch.int32)
        tensors[p + "budget"] = torch.tensor([budget], dtype=torch.int32)
        tensors[p + "ratio"] = torch.tensor([module.compress_ratio], dtype=torch.int32)
        tensors[p + "hidden"] = hidden.contiguous()
        tensors[p + "positions"] = positions.contiguous()
        tensors[p + "scores"] = scores.contiguous()
        max_blocks = scores.shape[-1]
        valid = torch.arange(max_blocks)[None, None, :] < n_blocks[:, :, None]
        tensors[p + "scores_valid"] = valid.contiguous()
        tensors[p + "selected"] = selected.contiguous()
        tensors[p + "n_blocks"] = n_blocks.contiguous()
        tensors[p + "n_selected"] = n_sel.contiguous()
        print(
            f"case{i}: S={seq_len} budget={budget} block_topk={module.block_topk} "
            f"n_blocks[last]={int(n_blocks[0, -1])} n_sel[last]={int(n_sel[0, -1])} "
            f"mask_ok=True"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.2f} MiB, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
