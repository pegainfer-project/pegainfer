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
    Qwen4ExpTextQSAIndexer,
    Qwen4ExpTextRotaryEmbedding,
)

CONFIG = "pegainfer-qwen4/tests/frozen_config.json"
METADATA_KEY = "qwen4_qsa_indexer_golden"
SEED = 0x_03_38_F9_5A_11_7E
FORMAT_VERSION = 2
# 16 tokens keep 4 blocks, so short cases reach the truncation branch.
SHRUNK_BUDGET = 16

# (seq_len, budget, evaluated rows). A budget of None is the config's, and rows
# of None evaluate every query row.
CASES = [
    (5, None, None),  # tail 1, no truncation
    (8, None, None),  # tail 0, no truncation
    (64, None, None),  # 16 complete blocks, all kept
    (14, SHRUNK_BUDGET, None),  # 3 complete blocks + tail 2
    (16, SHRUNK_BUDGET, None),  # 4 complete blocks, all kept
    (20, SHRUNK_BUDGET, None),  # 5 complete blocks, one dropped
    # Visible 2050 and 2051 keep all 512 blocks, 2052 starts dropping one block,
    # and 2053 to 2055 drop one while keeping a tail of 1 to 3.
    (2055, None, range(2049, 2055)),
]


def load_text_config(path: Path):
    raw = path.read_bytes()
    config = Qwen4ExpConfig(**json.loads(raw)).text_config
    return config, hashlib.sha256(raw).hexdigest()


def build_indexer(config) -> Qwen4ExpTextQSAIndexer:
    torch.manual_seed(SEED)
    layer = config.layer_types.index("qwen_sparse_attention")
    indexer = Qwen4ExpTextQSAIndexer(config, layer)
    with torch.no_grad():
        # The norm weights start at zero; give the (1 + w) scale something to do.
        indexer.q_layernorm.weight.normal_(0.0, 0.5)
        indexer.k_layernorm.weight.normal_(0.0, 0.5)
    return indexer.to(torch.bfloat16)


@contextlib.contextmanager
def recorded_topk(record: list[torch.Tensor]):
    """Record the input of every `Tensor.topk` call. Inside the indexer each call
    takes one query row's block scores."""
    original = torch.Tensor.topk

    def topk(self, *args, **kwargs):
        record.append(self.detach().clone())
        return original(self, *args, **kwargs)

    torch.Tensor.topk = topk
    try:
        yield
    finally:
        torch.Tensor.topk = original


def run_reference(indexer, rotary, hidden):
    """The reference's `index_qk_proj` output, per-row block scores and mask."""
    seq_len = hidden.shape[1]
    projected, scores = [], []
    handle = indexer.index_qk_proj.register_forward_hook(
        lambda _module, _inputs, output: projected.append(output.detach().clone())
    )
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()[None, None]
    try:
        with torch.no_grad(), recorded_topk(scores):
            position = rotary(hidden, torch.arange(seq_len)[None])
            mask = indexer(hidden, position, causal, None)
    finally:
        handle.remove()
    return projected[0][0], scores, mask[0, 0]


def case_tensors(indexer, rotary, index) -> dict[str, torch.Tensor]:
    """One case: the indexer's inputs after `index_qk_proj`, and what the reference
    computes for each evaluated query row.

    `keys` [S, head_dim] holds every token's raw key and `queries` [R, heads,
    head_dim] the raw queries at the query positions `rows` [R]. Row j of
    `scores` [R, ceil(S / ratio)] holds (rows[j] + 1) // ratio block scores
    followed by NaN, and row j of `mask` [R, S] is the set of tokens the
    reference keeps for that query. torch.topk leaves the order of tied scores
    open, so a block tied with the last kept one at the cut is an equally valid
    pick.
    """
    seq_len, _, rows = CASES[index]
    gen = torch.Generator().manual_seed(SEED + index)
    hidden = torch.randn(1, seq_len, indexer.index_qk_proj.in_features, generator=gen)
    projected, row_scores, mask = run_reference(indexer, rotary, hidden.bfloat16())
    ratio = indexer.compress_ratio
    # Every row from ratio - 1 on sees a complete block and calls topk once.
    calls = len(row_scores)
    assert calls == seq_len - ratio + 1, f"case {index}: {calls} topk calls"
    if not all(torch.isfinite(s).all() for s in row_scores):
        raise SystemExit(f"case {index}: non-finite block score from the reference")
    rows = torch.arange(seq_len) if rows is None else torch.tensor(list(rows))
    scores = torch.full((len(rows), -(-seq_len // ratio)), float("nan"))
    for slot, row in enumerate(rows.tolist()):
        if row >= ratio - 1:
            scores[slot, : (row + 1) // ratio] = row_scores[row - ratio + 1]
    q_width = indexer.index_n_heads * indexer.index_head_dim
    queries = projected[rows, :q_width].reshape(len(rows), indexer.index_n_heads, -1)
    return {
        f"case{index}_keys": projected[:, q_width:].contiguous(),
        f"case{index}_queries": queries.contiguous(),
        f"case{index}_rows": rows.to(torch.int32),
        f"case{index}_scores": scores,
        f"case{index}_mask": mask[rows].contiguous(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump the QSA indexer golden from the transformers reference."
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config, config_sha256 = load_text_config(root / CONFIG)
    indexer = build_indexer(config)
    rotary = Qwen4ExpTextRotaryEmbedding(config)
    tensors = {
        "q_layernorm": indexer.q_layernorm.weight.detach().contiguous(),
        "k_layernorm": indexer.k_layernorm.weight.detach().contiguous(),
    }
    for index, (seq_len, budget, _) in enumerate(CASES):
        budget = budget or config.indexer_budget
        indexer.token_budget = budget
        indexer.block_topk = budget // indexer.compress_ratio
        tensors[f"case{index}_budget"] = torch.tensor([budget], dtype=torch.int32)
        tensors.update(case_tensors(indexer, rotary, index))
        print(f"case{index}: S={seq_len} budget={budget}")

    manifest = {
        "format_version": FORMAT_VERSION,
        "config": CONFIG,
        "config_sha256": config_sha256,
        "seed": SEED,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "platform": platform.machine(),
    }
    # One key holding sorted JSON, since safetensors writes its metadata map in a
    # random order and several keys would make two identical runs differ.
    metadata = {METADATA_KEY: json.dumps(manifest, sort_keys=True)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata=metadata)
    print(f"wrote {args.out} with {len(tensors)} tensors")


if __name__ == "__main__":
    main()
