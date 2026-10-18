"""Dump the Gemma 4 HF reference: layer-boundary activations and top-K logprobs.

    python tools/accuracy/dump_gemma4_hf_golden.py <model-dir> <out.safetensors> \
        --source-repo google/gemma-4-12B-it \
        --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

Three cases -- one BOS token, a nine-token prefill, and one exactly the
sliding window -- with layer-boundary activations on the first two. What each
case is for, why the cuts sit where they do, and the two constants this pins
are in `docs/models/gemma4/hf-golden.md`; the reasoning is kept there rather
than duplicated here, because it is the same reasoning and it drifts when it
lives twice.

The dump aborts rather than writing if a run is not bitwise reproducible, if a
probe layer is not the layer type it was selected for, or if the logits escape
the declared softcap. Two runs against the same checkpoint produce the same
file, so regeneration is checked with sha256 and nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SEED = 0x_4E11_A404
TOP_K = 64
SHORT_LEN = 9
HASHED_FILES = ("config.json", "generation_config.json")


def input_device(model, device: str):
    """Where the prompt ids go: the requested device, or the sharded model's first
    device when `--device auto` spread the weights over several GPUs."""
    return model.device if device == "auto" else device


# Where the final RMSNorm sits, as a cut target alongside integer layer indices.
FINAL_NORM = "final_norm"
METADATA_KEY = "gemma4_golden"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("out")
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--fingerprint-dir",
        type=Path,
        help="the checkpoint the gates serve, when the tower here is a copy derived from it",
    )
    parser.add_argument(
        "--tower",
        help="how the tower was derived; given together with --fingerprint-dir",
    )
    return parser.parse_args()


def safetensors_header_sha256s(model_dir: Path) -> dict[str, str]:
    """Header digests of every weight file: the single `model.safetensors` a 12B
    checkpoint ships, or each shard the index of a 26B/31B checkpoint names, with
    the index itself hashed in full so the shard layout is pinned too."""
    single = model_dir / "model.safetensors"
    index = model_dir / "model.safetensors.index.json"
    if single.exists():
        files = [single]
    elif index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = [model_dir / name for name in sorted(set(weight_map.values()))]
    else:
        raise SystemExit(
            "expected model.safetensors or model.safetensors.index.json in the checkpoint"
        )
    digests = {}
    for path in files:
        with path.open("rb") as handle:
            (header_len,) = struct.unpack("<Q", handle.read(8))
            header = handle.read(header_len)
        if len(header) != header_len:
            raise SystemExit(
                f"short read on the safetensors header of {path.name}: {len(header)} of {header_len}"
            )
        digests[f"{path.name}#header"] = hashlib.sha256(header).hexdigest()
    if not single.exists():
        digests[index.name] = hashlib.sha256(index.read_bytes()).hexdigest()
    return digests


def file_hashes(model_dir: Path) -> dict[str, str]:
    hashes = {}
    for name in HASHED_FILES:
        path = model_dir / name
        if not path.exists():
            raise SystemExit(f"required file missing from the checkpoint: {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes.update(safetensors_header_sha256s(model_dir))
    return hashes


def probe_layers(layer_types: list[str]) -> dict[str, int]:
    """Pick probe layers from the layer map rather than hardcoding indices.

    Both ends of both layer types: type dispatch is exercised at each end, and
    the first sliding layer is the only one whose input is the scaled
    embedding, which isolates the embedding scale from everything downstream.
    """
    sliding = [i for i, t in enumerate(layer_types) if t == "sliding_attention"]
    full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    if not sliding or not full:
        raise SystemExit(f"expected both layer types; saw {sorted(set(layer_types))}")
    return {
        "sliding_first": sliding[0],
        "global_first": full[0],
        "sliding_last": sliding[-1],
        "global_last": full[-1],
    }


def build_cuts(
    selected: dict[str, int], n_layers: int
) -> list[tuple[str, str, object]]:
    """Layer-boundary activations, deduplicated and ordered by depth.

    Each cut is (label, kind, target): kind is "in" or "out", target is a layer
    index or FINAL_NORM. Input of layer i and output of layer i-1 are the same
    activation, so adjacent probe layers share one tensor rather than two.
    """
    cuts: list[tuple[int, str, str, object]] = []
    seen: set[int] = set()

    def add(label: str, kind: str, target: object, depth: int) -> None:
        if depth in seen:
            return
        seen.add(depth)
        cuts.append((depth, label, kind, target))

    for name, index in sorted(selected.items(), key=lambda kv: kv[1]):
        add(f"{name}_in", "in", index, index)
        add(f"{name}_out", "out", index, index + 1)
    add(f"{FINAL_NORM}_out", "out", FINAL_NORM, n_layers + 1)

    cuts.sort()
    ordered = [(label, kind, target) for _, label, kind, target in cuts]
    if ordered[0][1:] != ("in", 0):
        raise SystemExit(
            "the first cut must be the scaled embedding (input of layer 0)"
        )
    return ordered


def excluded_ids(tokenizer) -> set[int]:
    """Special and added ids, which the golden must not sample.

    Not cosmetic: the modality ids are exactly the inputs the engine is
    required to reject, so a golden containing them would compare against a
    request the engine will never serve.
    """
    return set(tokenizer.all_special_ids) | {
        int(i) for i in tokenizer.added_tokens_decoder
    }


def sample_tokens(length: int, bos: int, excluded: set[int], vocab: int) -> list[int]:
    generator = torch.Generator().manual_seed(SEED + length)
    tokens = [bos]
    while len(tokens) < length:
        for token in torch.randint(1, vocab, (length,), generator=generator).tolist():
            if token in excluded:
                continue
            tokens.append(token)
            if len(tokens) == length:
                break
    return tokens


def run_case(model, text_model, cuts, tokens, device):
    captured: dict[tuple[str, object], torch.Tensor] = {}
    handles = []

    def record(kind, target):
        # On the host: `--device auto` can leave the probes on different
        # GPUs, which the stack below refuses.
        def pre(_mod, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            captured[(kind, target)] = hidden.detach().to("cpu", copy=True)

        def post(_mod, _args, _kwargs, out):
            hidden = out[0] if isinstance(out, tuple) else out
            captured[(kind, target)] = hidden.detach().to("cpu", copy=True)

        module = text_model.norm if target == FINAL_NORM else text_model.layers[target]
        if kind == "in":
            return module.register_forward_pre_hook(pre, with_kwargs=True)
        return module.register_forward_hook(post, with_kwargs=True)

    for _, kind, target in cuts:
        handles.append(record(kind, target))

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0]
    finally:
        for handle in handles:
            handle.remove()

    if not cuts:
        return None, logits
    return torch.stack(
        [captured[(kind, target)][0] for _, kind, target in cuts]
    ), logits


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if bool(args.fingerprint_dir) != bool(args.tower):
        raise SystemExit("--fingerprint-dir and --tower go together")

    config = AutoConfig.from_pretrained(str(model_dir))
    text_config = config.get_text_config()
    layer_types = list(text_config.layer_types)
    selected = probe_layers(layer_types)
    cuts = build_cuts(selected, len(layer_types))
    window = text_config.sliding_window
    softcap = text_config.final_logit_softcapping

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    excluded = excluded_ids(tokenizer)
    bos = text_config.bos_token_id
    # Only the two short cases carry probes: at window width the activation
    # tensors would dwarf the rest of the fixture, and nothing consumes them.
    cases = [
        ("single", [bos], True),
        (
            "short",
            sample_tokens(SHORT_LEN, bos, excluded, text_config.vocab_size),
            True,
        ),
        (
            "edge",
            sample_tokens(window, bos, excluded, text_config.vocab_size),
            False,
        ),
    ]

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    text_model = model.model.language_model
    for name, index in selected.items():
        expected = name.startswith("sliding")
        if text_model.layers[index].self_attn.is_sliding != expected:
            kind = "sliding" if expected else "global"
            raise SystemExit(
                f"probe layer {name}={index} is not the {kind} layer it was selected as"
            )

    tensors: dict[str, torch.Tensor] = {}
    for name, tokens, probed in cases:
        case_cuts = cuts if probed else []
        hidden, logits = run_case(
            model, text_model, case_cuts, tokens, input_device(model, args.device)
        )
        replay_hidden, replay_logits = run_case(
            model, text_model, case_cuts, tokens, input_device(model, args.device)
        )
        reproducible = torch.equal(logits, replay_logits) and (
            hidden is None or torch.equal(hidden, replay_hidden)
        )
        if not reproducible:
            raise SystemExit(
                f"case {name!r} is not bitwise reproducible within one process"
            )
        if logits.float().abs().max().item() > softcap:
            raise SystemExit(
                f"case {name!r} produced logits outside the declared softcap {softcap}"
            )

        logprobs = F.log_softmax(logits.float(), dim=-1)
        values, indices = torch.topk(logprobs, TOP_K, dim=-1)
        tensors[f"{name}_tokens"] = torch.tensor(tokens, dtype=torch.int32)
        tensors[f"{name}_topk_ids"] = indices.to(torch.int32).cpu()
        tensors[f"{name}_topk_logprobs"] = values.cpu()
        if hidden is not None:
            tensors[f"{name}_hidden"] = hidden.cpu()
        print(
            f"{name}: {len(tokens)} tokens, probes={'yes' if probed else 'no'}",
            flush=True,
        )

    manifest = {
        "source_repo": args.source_repo,
        "revision": args.revision,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model_class": type(model).__name__,
        "seed": SEED,
        "top_k": TOP_K,
        "dtypes": "hidden=bfloat16 (the compute dtype), logprobs=float32 over float32 logits",
        "cut_labels": [label for label, _, _ in cuts],
        "probe_layers": selected,
        "probed_cases": [name for name, _, probed in cases if probed],
        "sliding_window": window,
        "final_logit_softcapping": softcap,
        # Not sqrt(hidden_size): the buffer is cast to the weight dtype before
        # the multiply, so bf16 rounding is part of the reference.
        "embed_scale_bf16": float(
            text_model.embed_tokens.embed_scale.to(torch.bfloat16)
        ),
        "file_sha256": file_hashes(args.fingerprint_dir or model_dir),
    }
    if args.fingerprint_dir:
        manifest["reference_tower"] = {
            "derivation": args.tower,
            "file_sha256": file_hashes(model_dir),
        }
    # One key, sorted: safetensors serializes its metadata map in a randomized
    # order, so a multi-key block makes two runs of this script differ byte for
    # byte while carrying identical content. Collapsing it is what lets the
    # regeneration check be a plain sha256 rather than a field-by-field diff.
    metadata = {METADATA_KEY: json.dumps(manifest, sort_keys=True)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=metadata)
    print(
        f"wrote {out}: {len(cases)} cases, {len(cuts)} cuts "
        f"({', '.join(label for label, _, _ in cuts)}), top{TOP_K}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
