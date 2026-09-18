#!/usr/bin/env python3
"""Build a DFlash2 component golden from captured native-model tensors, offline.

This is a test-data tool, not a runtime dependency. It never downloads weights or
substitutes random tensors for a missing capture. The input producer must record
the target/draft revisions and final normalized hidden-state provenance.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def checkpoint_file(checkpoint, name):
    relative = Path(name)
    if relative.anchor or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe checkpoint shard path: {name}")
    root = checkpoint.resolve()
    path = checkpoint / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"checkpoint shard escapes its directory: {name}")
    return path


def load_selector(checkpoint):
    """Load only the three selector tensors, preserving checkpoint BF16 values."""
    names = [
        "candidate_selector.hidden_projection.weight",
        "candidate_selector.predecessor_codebook",
        "candidate_selector.successor_codebook",
    ]
    index = checkpoint / "model.safetensors.index.json"
    single = checkpoint_file(checkpoint, "model.safetensors")
    if single.exists() and index.exists():
        raise ValueError("ambiguous checkpoint: both single-file weights and shard index exist")
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = {name: checkpoint_file(checkpoint, weight_map[name]) for name in names}
    else:
        files = {name: single for name in names}
    tensors = {}
    for name, path in files.items():
        with safe_open(path, framework="pt", device="cpu") as source:
            tensor = source.get_tensor(name)
            if tensor.dtype != torch.bfloat16 or not torch.isfinite(tensor).all():
                raise ValueError(f"{name}: expected finite BF16 weights")
            tensors[name] = tensor
    return [tensors[name] for name in names], files


def strict_candidates(logits, k):
    # Input indices are token IDs in ascending order. Stable descending sort
    # therefore implements (-logit, token_id), including ties at the kth value.
    candidates = torch.argsort(logits, dim=-1, descending=True, stable=True)[..., :k]
    return logits.gather(-1, candidates), candidates


def walk(candidates, scores):
    batch, slots, k = candidates.shape
    predecessor_index = torch.zeros(batch, dtype=torch.long)
    batch_ids = torch.arange(batch)
    paths, indices, margins = [], [], []
    for slot in range(slots):
        row = scores[batch_ids, slot, predecessor_index]
        maximum = row.amax(-1, keepdim=True)
        # Lattice ties use token ID, not the arbitrary candidate array index.
        token_ids = candidates[:, slot]
        chosen_id = token_ids.masked_fill(row != maximum, torch.iinfo(torch.int64).max).amin(-1)
        predecessor_index = (token_ids == chosen_id[:, None]).to(torch.int64).argmax(-1)
        best_two = row.topk(2, dim=-1).values
        margins.append(best_two[:, 0] - best_two[:, 1])
        paths.append(chosen_id)
        indices.append(predecessor_index)
    return torch.stack(paths, 1), torch.stack(indices, 1), torch.stack(margins, 1)


def reference(hidden, logits, anchors, weights, k):
    projection, predecessor_codebook, successor_codebook = weights
    projected = torch.nn.functional.linear(hidden.float(), projection.float()).to(torch.bfloat16)
    unary, candidates = strict_candidates(logits[:, 1:].float(), k)
    batch, slots, _ = candidates.shape
    previous_ids = torch.cat((anchors[:, None, None].expand(batch, 1, k), candidates[:, :-1]), dim=1)
    a = predecessor_codebook[previous_ids].float()
    b = successor_codebook[candidates].float()
    h = projected[:, 1:].float()
    # Preserve the runtime plan's FP32 gate tensor before the shared GEMM.
    gated_predecessor = a * h[:, :, None, :]
    scores = unary[:, :, None, :] + torch.matmul(gated_predecessor, b.transpose(-2, -1))
    if not torch.isfinite(projected).all() or not torch.isfinite(scores).all():
        raise ValueError("nonfinite reference projection or lattice score")
    paths, indices, margins = walk(candidates, scores)

    # A second expression of vLLM's pinned _score_edges equation checks the
    # FP32 indexing contract; PyTorch may use the same backend for both. Keep
    # its original BF16 arithmetic as a separate rounding diagnostic.
    pinned_scores = unary[:, :, None, :] + torch.einsum("btpr,btcr->btpc", a * h[:, :, None], b)
    score_delta = (scores - pinned_scores).abs().max().item()
    pinned_paths, _, _ = walk(candidates, pinned_scores)
    if not torch.equal(paths, pinned_paths):
        raise ValueError("oracle expressions disagree on path; inspect near-tie margins")
    upstream_scores = unary[:, :, None, :] + torch.einsum(
        "btpr,btcr->btpc",
        a.to(torch.bfloat16) * projected[:, 1:, None],
        b.to(torch.bfloat16),
    ).float()
    upstream_paths, _, upstream_margins = walk(candidates, upstream_scores)
    return {
        "candidate_ids": candidates.tolist(),
        "candidate_values": unary.tolist(),
        "projected_hidden": projected.float().tolist(),
        "edge_scores": scores.tolist(),
        "path_ids": paths.tolist(),
        "path_indices": indices.tolist(),
        "selected_edge_margins": margins.tolist(),
        "minimum_selected_edge_margin": margins.min().item(),
        "reference_expression_max_abs_delta": score_delta,
        "upstream_bf16_max_abs_delta": (scores - upstream_scores).abs().max().item(),
        "upstream_bf16_path_ids": upstream_paths.tolist(),
        "upstream_bf16_path_difference_count": (paths != upstream_paths).sum().item(),
        "upstream_bf16_minimum_selected_edge_margin": upstream_margins.min().item(),
        "projection_shape": list(projected.shape),
        "candidate_shape": list(candidates.shape),
        "edge_shape": list(scores.shape),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    config_path = args.checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    profile = config["dflash_config"]
    if "DFlash2DraftModel" not in config.get("architectures", []):
        raise ValueError("expected native DFlash2DraftModel checkpoint")
    if profile["selector_top_k"] != 16:
        raise ValueError("Phase 1 fixture contract requires candidate_k=16")
    for source in [config, profile]:
        if (
            float(source.get("output_multiplier", 1.0)) != 1.0
            or source.get("final_logit_softcapping")
        ):
            raise ValueError("Phase 1 supports untransformed logits only")
    provenance = json.loads(args.provenance.read_text())
    if provenance.get("input_origin") != "pinned_target_and_native_draft_forward":
        raise ValueError("real target/native-draft forward provenance is required")
    for key in [
        "target_model",
        "target_revision",
        "draft_model",
        "draft_revision",
        "reference_revision",
        "reference_source_sha256",
        "capture_source_sha256",
        "draft_checkpoint_sha256",
    ]:
        if not provenance.get(key):
            raise ValueError(f"missing provenance: {key}")
    metadata = provenance["target_metadata"]
    if (
        metadata["model_id"] != provenance["target_model"]
        or metadata["revision"] != provenance["target_revision"]
    ):
        raise ValueError("target metadata and capture provenance disagree")
    inputs = load_file(args.inputs)
    hidden, logits, anchors = (inputs[name] for name in ["hidden", "logits", "anchors"])
    if (
        hidden.dtype != torch.bfloat16
        or logits.dtype != torch.bfloat16
        or anchors.dtype != torch.int64
    ):
        raise ValueError("capture contract: hidden/logits BF16, anchors I64")
    if hidden.ndim != 3 or logits.ndim != 3 or anchors.ndim != 1:
        raise ValueError("capture ranks must be [batch,block,hidden/vocab] and [batch]")
    batch, block, hidden_size = hidden.shape
    vocab = config["vocab_size"]
    rank, k = profile["selector_rank"], profile["selector_top_k"]
    if (
        hidden_size != config["hidden_size"]
        or block != profile["block_size"]
        or tuple(logits.shape) != (batch, block, vocab)
        or anchors.shape[0] != batch
    ):
        raise ValueError("capture shape does not match checkpoint profile")
    if (
        not torch.isfinite(hidden).all()
        or not torch.isfinite(logits).all()
        or ((anchors < 0) | (anchors >= vocab)).any()
    ):
        raise ValueError("nonfinite capture or out-of-vocabulary anchor")
    weights, files = load_selector(args.checkpoint)
    expected_shapes = [(rank, hidden_size), (vocab, rank), (vocab, rank)]
    if [tuple(t.shape) for t in weights] != expected_shapes:
        raise ValueError("selector tensor shapes do not match config")
    index_path = args.checkpoint / "model.safetensors.index.json"
    checkpoint_files = set(files.values())
    if index_path.exists():
        checkpoint_files.update(
            checkpoint_file(args.checkpoint, name)
            for name in json.loads(index_path.read_text())["weight_map"].values()
        )
    checkpoint_hashes = {
        path.relative_to(args.checkpoint).as_posix(): sha256(path)
        for path in checkpoint_files
    }
    captured_checkpoint_hash = provenance.get("draft_checkpoint_sha256")
    if captured_checkpoint_hash and captured_checkpoint_hash not in checkpoint_hashes.values():
        raise ValueError("capture and selector checkpoint hashes differ")
    with torch.inference_mode():
        golden = reference(hidden, logits, anchors, weights, k)
    golden.update({
        "format_version": 1,
        "candidate_k": k,
        "hidden_dtype": "BF16",
        "logits_dtype": "BF16",
        "projection_output_dtype": "BF16",
        "edge_accumulation_dtype": "F32",
        "candidate_ties": "logit_desc_token_id_asc",
        "path_ties": "score_desc_token_id_asc",
        "projected_hidden_includes_anchor": True,
    })
    args.output.mkdir(parents=True, exist_ok=True)
    target_inputs = args.output / "inputs.safetensors"
    if args.inputs.resolve() != target_inputs.resolve():
        shutil.copyfile(args.inputs, target_inputs)
    reference_path = args.output / "reference.json"
    reference_path.write_text(json.dumps(golden, indent=2) + "\n")
    manifest = {
        "format_version": 1,
        "provenance": provenance,
        "config_sha256": sha256(config_path),
        "checkpoint_files": checkpoint_hashes,
        "checkpoint_index_sha256": sha256(index_path) if index_path.exists() else None,
        "inputs_sha256": sha256(target_inputs),
        "reference_sha256": sha256(reference_path),
        "generator_sha256": sha256(Path(__file__)),
        "oracle": "CPU PyTorch FP32 projection/accumulation, BF16 projection output",
        "reference_math_source": "https://github.com/vllm-project/vllm/blob/3406ec1dae9916f920b90f0dbf90dcf54923d042/vllm/model_executor/models/qwen3_dflash2.py",
        "torch_version": torch.__version__,
        "shape": {
            "batch": batch, "block": block, "hidden": hidden_size,
            "vocab": vocab, "rank": rank, "candidate_k": k,
        },
        "minimum_selected_edge_margin": golden["minimum_selected_edge_margin"],
        "reference_expression_max_abs_delta": golden["reference_expression_max_abs_delta"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
