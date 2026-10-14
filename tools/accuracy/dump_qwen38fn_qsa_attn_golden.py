#!/usr/bin/env python3
"""Generate the Qwen3.8-Flash-Next QSA sparse full-attention golden.

This dumps the *real-geometry* sparse attention of one full-attention layer
(`Qwen4ExpTextAttention`, transformers 5.16.1 `models/qwen4_exp`) into a fixture
the sparse-attention kernel gate replays: for a handful of query rows it stores
the reference's own post-projection activations, the effective sparse+causal
mask, and the reference's attention weights + gated output. Nothing here
re-implements the reference math -- the stored values are captured from the
reference module (forward hooks), and the reload step only replays them.

Geometry is the frozen checkpoint config (`hidden_size` 2560, 24 Q / 2 KV heads,
`head_dim` 256, `partial_rotary_factor` 0.25 -> 64 rotary dims, GQA group 12,
sigmoid output gate, indexer 4 q / 1 kv @ 128 with the real `indexer_budget`
2048 -> 512 block budget).

Boundary rule (why the plain GEMM weights are absent)
-----------------------------------------------------
`q_proj`/`k_proj`/`v_proj`/`o_proj` are plain GEMMs (98 MB of weights) and are
NOT stored. Instead their *outputs* are captured with forward hooks and stored:
the fused Q+gate `q_proj` output, the `k_proj`/`v_proj` outputs, and the input to
`o_proj` (the gated attention activation) -- that last one is the golden's
expected output, since `o_proj` is a plain cuBLAS GEMM. Only the small weights
that carry real math are stored: `q_norm`/`k_norm` (the per-head `(1+w)` RMS
norm) and the indexer's `index_qk_proj` (3.3 MB) + `q_layernorm`/`k_layernorm`.
Each captured activation is named for the projection that produced it
(`queries`/`keys`/`values`), so no stored tensor shares a name with the weight
it came from.

Real boundary (S=2055, budget 2048 -> 512 blocks)
-------------------------------------------------
With the real budget a query keeps `topk(min(512, n_complete))` complete 4-token
blocks plus the tail. Over the evaluated rows (`range(2049, 2055)`, i.e. visible
counts 2050..2055):
  * visible 2050 / 2051 -> n_complete 512, keeps all 512 blocks (no drop);
  * visible 2052        -> n_complete 513, drops one block (4 causally-valid keys);
  * visible 2053..2055  -> drops one block and keeps a 1..3 token tail.
The dump asserts those drops (and that the short cases drop nothing).

Cases record only the evaluated query rows (a full 2055-row case would store a
~50 MB `[2055, 12288]` activation and a ~200 MB weight matrix); key/value-side
activations stay full length (they are small).

Fixture metadata is a single JSON key (`qwen4_qsa_attn_golden`, format_version 2);
there are no loose provenance tensors.

The reference's mask builder passes an int32 index to `Tensor.scatter` (torch
wants int64 on some versions), so `_scatter_int64` casts it for the reference
call only, scoped -- the dump runs on torch 2.14.1 and older.

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
from safetensors.torch import load_file, save_file

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextAttention,
    Qwen4ExpTextRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
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

# Small weights stored once and shared by every case. The big plain GEMMs
# (`q_proj`/`k_proj`/`v_proj`/`o_proj`) are intentionally absent -- their outputs
# are captured per case instead.
STORED_WEIGHTS = ("q_norm", "k_norm", "index_qk_proj", "q_layernorm", "k_layernorm")

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


def rotary_embeddings(rotary: Qwen4ExpTextRotaryEmbedding, dtype: torch.dtype, positions: torch.Tensor):
    """cos/sin [1,S,rotary_dim] from stored positions (the rotary module is deterministic)."""
    dummy = torch.zeros(1, positions.numel(), 1, dtype=dtype)
    return rotary(dummy, positions.to(torch.long)[None])


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen4ExpTextRMSNorm: (x * rsqrt(mean(x^2)+eps) * (1+w)), computed in fp32."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.float())).type_as(x)


def replay(config: Qwen4ExpConfig, rotary, weights: dict, case: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the evaluated rows' (gated attn_out, attn_weights) from stored inputs.

    Mirrors the reference forward for the evaluated rows only; used to self-check
    that the stored outputs are fully determined by the stored inputs + weights.
    """
    seq_len = int(case["seq_len"].item())
    head_dim = config.head_dim
    eps = config.rms_norm_eps
    groups = config.num_attention_heads // config.num_key_value_heads
    scaling = head_dim**-0.5

    q = case["queries"]                    # [R, 12288] fused Q+gate (captured)
    k = case["keys"]                       # [S, 512]   (captured)
    v = case["values"]                     # [S, 512]   (captured)
    mask = case["mask"]                    # [R, S] bool effective sparse+causal
    n_rows = int(q.shape[0])

    # Fused Q+gate split: each head's 512 block is [256 q | 256 gate].
    query_states, gate = torch.chunk(q.view(1, n_rows, -1, head_dim * 2), 2, dim=-1)
    gate = gate.reshape(1, n_rows, -1)                  # [1, R, 6144]
    query_states = rms_norm(query_states, weights["q_norm"], eps).transpose(1, 2)  # [1,QH,R,D]
    key_states = rms_norm(k.view(1, seq_len, -1, head_dim), weights["k_norm"], eps).transpose(1, 2)
    value_states = v.view(1, seq_len, -1, head_dim).transpose(1, 2)

    cos, sin = rotary_embeddings(rotary, q.dtype, case["positions"])
    # Queries sit at the evaluated rows; keys cover every position. The reference
    # rotates both in one call with the full cos/sin, which is row-wise identical.
    row_pos = case["rows"].to(torch.long)
    query_states = apply_rotary_pos_emb(
        query_states, cos=cos.index_select(1, row_pos), sin=sin.index_select(1, row_pos)
    )
    key_states = apply_rotary_pos_emb(key_states, cos=cos, sin=sin)
    key_states = repeat_kv(key_states, groups)
    value_states = repeat_kv(value_states, groups)

    attn = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    min_dtype = torch.finfo(q.dtype).min
    fmask = torch.where(
        mask[None, None], torch.zeros((), dtype=q.dtype), torch.tensor(min_dtype, dtype=q.dtype)
    )
    attn = attn + fmask
    attn_weights = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous().reshape(1, n_rows, -1)
    out = (out * torch.sigmoid(gate)).squeeze(0)        # [R, 6144]
    return out, attn_weights


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
        "index_qk_proj": module.indexer.index_qk_proj.weight.detach().contiguous(),
        "q_layernorm": module.indexer.q_layernorm.weight.detach().contiguous(),
        "k_layernorm": module.indexer.k_layernorm.weight.detach().contiguous(),
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

        # Self-check: blocked (sparse-dropped or future) positions carry zero softmax mass.
        blocked_mass = (attn_weights.float() * (~eff).float()).abs().max().item()
        assert blocked_mass == 0.0, f"case {i}: blocked positions leaked softmax mass ({blocked_mass})"

        # Drop accounting: causally-valid keys the sparse budget discarded, per row.
        causal_count = causal_bool[0, 0][row_idx].sum(-1)
        kept_count = eff[0, 0][row_idx].sum(-1)
        drops = (causal_count - kept_count).tolist()
        assert all(d == 0 for d in drops[:2]) if seq_len == 2055 else all(d == 0 for d in drops), (
            f"case {i}: unexpected sparse drop on a short case: {drops}"
        )
        if seq_len == 2055:
            assert all(d == 0 for d in drops[:2]) and all(d == 4 for d in drops[2:]), (
                f"case {i}: real boundary drops unexpected: {drops}"
            )

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

        print(
            f"dump case{i}: S={seq_len} R={len(rows)} drops={drops} "
            f"blocked_mass={blocked_mass:.1e} attn_out={tuple(o_rows.shape)}"
        )

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


def verify(path: Path) -> None:
    """Reload the fixture and reproduce every stored output bit-for-bit."""
    config, _ = load_config()
    rotary = Qwen4ExpTextRotaryEmbedding(config)
    data = load_file(str(path))
    weights = {name: data[name] for name in STORED_WEIGHTS}

    for i, (seq_len, _) in enumerate(CASES):
        p = f"case{i}_"
        case = {k[len(p) :]: v for k, v in data.items() if k.startswith(p)}
        out, attn = replay(config, rotary, weights, case)
        rows = case["rows"].numel()
        ok_out = torch.equal(out, case["attn_out"])
        ok_attn = torch.equal(attn, case["attn_weights"])
        print(
            f"verify case{i}: S={seq_len} R={rows} "
            f"attn_out={'OK' if ok_out else 'FAIL'} attn_weights={'OK' if ok_attn else 'FAIL'}"
        )
        assert ok_out and ok_attn, f"case {i}: stored outputs not reproducible bit-for-bit"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="skip the dump and only re-verify an existing fixture",
    )
    args = parser.parse_args()

    if not args.verify_only:
        dump(args)
    verify(args.out)


if __name__ == "__main__":
    main()
