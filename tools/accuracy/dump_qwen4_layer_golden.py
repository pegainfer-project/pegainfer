import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, DynamicCache
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRotaryEmbedding,
)

# A chain of Qwen3.8-Flash-Next decoder layers with the bookends around them, on
# the checkpoint's own weights: the token embedding repeated into the
# hyper-connection stream, the given layers in order, the top-level mixer and the
# untied head. The reference runs over a fixed prompt once on the accelerator,
# chained, and every stage the engine can be held against is recorded, each
# layer's under a `layer{N}_` prefix. It runs again on the CPU with every layer
# fed the accelerator's input for it and the head fed the accelerator's last
# stream, so the spread between the two at a stage is that layer's own and does
# not compound down the chain. With `--decode D` the last D prompt tokens are
# fed one at a time on the layers' caches after a prefill of the rest, so the
# recorded steps hold the engine's decode path on a carried state. The passes
# go layer by layer rather than pass by pass: one layer is built, every pass
# goes through it on the previous layer's output for that pass, and it is freed
# before the next layer is read, so all 48 layers fit one card and one host.
# The n-gram layer's table stays in its shards and is read row by row.
#
#     python tools/accuracy/dump_qwen4_layer_golden.py --model /data/models/Qwen3.8-Flash-Next \
#         --layers 0,3 --decode 8 --out golden/qwen4-layers03-decode8.safetensors
#     python tools/accuracy/dump_qwen4_layer_golden.py --model /data/models/Qwen3.8-Flash-Next \
#         --layers all --decode 8 --out golden/qwen4-all-decode8.safetensors

METADATA_KEY = "qwen4_layer_golden"
FORMAT_VERSION = 5
# Recorded from the CPU pass as well: the accelerator versus CPU spread of the
# reference, each layer on the same input, is the floor the engine is held to at
# these stages.
CPU_STAGES = (
    "ple_out",
    "attn_gated",
    "attn_out",
    "stream_after_attn",
    "router_logits",
    "moe_out",
    "stream_out",
    "gdn_state",
    "logits",
)
PROMPT = (
    "The hyper-connection stream carries four copies of the residual through every "
    "layer, and the mixer reads them back before the head. Routing picks ten of the "
    "five hundred and twelve experts per token."
)
TEXT_PREFIX = "model.language_model."
LAYER_PREFIX = TEXT_PREFIX + "layers.{layer}."
MIXER_PREFIX = TEXT_PREFIX + "hyper_connection_mixer."
EMBED_NAME = TEXT_PREFIX + "embed_tokens.weight"
HEAD_NAME = "lm_head.weight"
NGRAM_SHARD = ".ple.ple_embedding.ngram_embedding.shard_"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_of(name: str) -> str:
    """The stage behind the optional `decode{i}_` and `layer{N}_` prefixes."""
    for _ in range(2):
        head, _, tail = name.partition("_")
        if (head.startswith("decode") and head[len("decode") :].isdigit()) or (
            head.startswith("layer") and head[len("layer") :].isdigit()
        ):
            name = tail
    return name


def parse_layers(spec: str, count: int) -> list[int]:
    """`all`, or comma separated indices and `a-b` ranges, increasing."""
    if spec == "all":
        return list(range(count))
    layers: list[int] = []
    for item in spec.split(","):
        first, _, last = item.partition("-")
        layers.extend(range(int(first), int(last or first) + 1))
    if not layers or layers != sorted(set(layers)) or layers[-1] >= count:
        raise SystemExit(f"--layers {spec} must be increasing layer indices below {count}")
    return layers


class Checkpoint:
    """The shard index, read once; a group's tensors are read when the group is built."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]

    def tensors(self, wanted: dict[str, str]) -> dict[str, torch.Tensor]:
        """`wanted` maps checkpoint names to the keys the module uses."""
        by_shard: dict[str, list[str]] = {}
        for name in wanted:
            by_shard.setdefault(self.weight_map[name], []).append(name)
        out = {}
        for shard, names in sorted(by_shard.items()):
            with safe_open(self.model_dir / shard, framework="pt") as f:
                for name in names:
                    out[wanted[name]] = f.get_tensor(name)
        return out

    def bookends(self):
        """The embedding weight, the top-level mixer's state and the head's."""
        mixer = {name: name[len(MIXER_PREFIX) :] for name in self.weight_map if name.startswith(MIXER_PREFIX)}
        return self.tensors({EMBED_NAME: "weight"})["weight"], self.tensors(mixer), self.tensors({HEAD_NAME: "weight"})

    def layer(self, layer: int):
        """The layer's state dict and, for the n-gram layer, its table pieces in order as (file, tensor name)."""
        prefix = LAYER_PREFIX.format(layer=layer)
        wanted: dict[str, str] = {}
        pieces: dict[int, tuple[Path, str]] = {}
        for name in self.weight_map:
            if not name.startswith(prefix):
                continue
            if NGRAM_SHARD in name:
                piece = int(name[name.index(NGRAM_SHARD) + len(NGRAM_SHARD) :].removesuffix(".weight"))
                pieces[piece] = (self.model_dir / self.weight_map[name], name)
            else:
                wanted[name] = name[len(prefix) :]
        missing = set(range(len(pieces))) - pieces.keys()
        if missing:
            raise SystemExit(f"layer {layer}: table pieces {sorted(missing)} are not in the index")
        return self.tensors(wanted), [pieces[i] for i in range(len(pieces))]


def build(module_factory, state: dict[str, torch.Tensor], device):
    """A reference module in bf16 with the checkpoint's tensors, nothing initialised."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            module = module_factory()
    finally:
        torch.set_default_dtype(previous)
    module = module.to_empty(device=device)
    module.load_state_dict(state, strict=True)
    return module.eval()


class ShardedEmbedding(torch.nn.Module):
    """The n-gram table read row by row from its pieces in the checkpoint, in place of the 102 GB `nn.Embedding`.

    `weight` is an empty buffer that only carries the device the rows are returned on, which is all the reference
    reads from it. Nothing else is held, so the module deep copies with the layer.
    """

    def __init__(self, pieces: list[tuple[Path, str]]):
        super().__init__()
        self.pieces = pieces
        with safe_open(pieces[0][0], framework="pt") as f:
            self.rows_per_piece = f.get_slice(pieces[0][1]).get_shape()[0]
        self.register_buffer("weight", torch.empty(0), persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).tolist()
        by_piece: dict[int, list[tuple[int, int]]] = {}
        for i, row in enumerate(flat):
            by_piece.setdefault(row // self.rows_per_piece, []).append((i, row % self.rows_per_piece))
        rows = [None] * len(flat)
        for piece, wanted in by_piece.items():
            file, name = self.pieces[piece]
            with safe_open(file, framework="pt") as f:
                view = f.get_slice(name)
                for i, row in wanted:
                    rows[i] = view[row : row + 1]
        return torch.cat(rows).reshape(*ids.shape, -1).to(self.weight.device)


class Recorder:
    """Forward hooks on the submodules whose boundaries the engine reproduces."""

    def __init__(self, layer: Qwen4ExpTextDecoderLayer):
        self.captured: dict[str, torch.Tensor] = {}
        self.handles = []
        if layer.ple is not None:
            self.output(layer.ple.ple_embedding, lambda out: {"ple_embeds": out})
            self.output(layer.ple.key_proj, lambda out: {"ple_key": out})
            self.output(layer.ple.value_proj, lambda out: {"ple_value": out})
            self.output(layer.ple, lambda out: {"ple_out": out})
            self.inputs(layer.attn_hyper_connection, lambda args: {"stream_ple": args[0]})
        self.output(layer.attn_hyper_connection, lambda out: {"attn_mixed": out[0], "attn_inject": out[2]})
        sublayer = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
        if layer.layer_type == "linear_attention":
            self.output(sublayer.in_proj_qkv, lambda out: {"gdn_qkv": out})
            self.output(sublayer.in_proj_z, lambda out: {"gdn_z": out})
            self.output(sublayer.in_proj_b, lambda out: {"gdn_b": out})
            self.output(sublayer.in_proj_a, lambda out: {"gdn_a": out})
            self.inputs(sublayer.norm, lambda args: {"gdn_core": args[0]})
            self.output(sublayer.norm, lambda out: {"gdn_normed": out})
            self.output(sublayer, lambda out: {"attn_out": out})
        else:
            self.output(sublayer.q_proj, lambda out: {"attn_q_proj": out})
            self.output(sublayer.k_proj, lambda out: {"attn_k_proj": out})
            self.output(sublayer.v_proj, lambda out: {"attn_v_proj": out})
            self.output(sublayer.q_norm, lambda out: {"attn_q_normed": out})
            self.output(sublayer.k_norm, lambda out: {"attn_k_normed": out})
            self.output(sublayer.indexer.index_qk_proj, lambda out: {"attn_indexer_qk": out})
            self.output(sublayer.indexer, lambda out: {"attn_indexer_mask": out})
            self.inputs(sublayer.o_proj, lambda args: {"attn_gated": args[0]})
            self.output(sublayer, lambda out: {"attn_out": out[0]})
        self.inputs(layer.mlp_hyper_connection, lambda args: {"stream_after_attn": args[0]})
        self.output(layer.mlp_hyper_connection, lambda out: {"mlp_mixed": out[0], "mlp_inject": out[2]})
        self.output(
            layer.mlp.gate,
            lambda out: {"router_logits": out[0], "routing_weights": out[1], "selected_experts": out[2]},
        )
        self.output(layer.mlp.experts, lambda out: {"expert_out": out})
        self.output(layer.mlp.shared_expert, lambda out: {"shared_out": out})
        self.output(layer.mlp.shared_expert_gate, lambda out: {"shared_gate_logit": out})
        self.output(layer.mlp, lambda out: {"moe_out": out})

    def output(self, module, pick):
        def hook(_module, _inputs, out):
            self.captured.update({k: v.detach() for k, v in pick(out).items()})

        self.handles.append(module.register_forward_hook(hook))

    def inputs(self, module, pick):
        def hook(_module, args):
            self.captured.update({k: v.detach() for k, v in pick(args).items()})

        self.handles.append(module.register_forward_pre_hook(hook))

    def take(self) -> dict[str, torch.Tensor]:
        captured, self.captured = self.captured, {}
        return captured

    def close(self):
        for handle in self.handles:
            handle.remove()


class Chain:
    """The reference on one device: the bookends resident, the layers built one at a time."""

    def __init__(self, text_config, checkpoint: Checkpoint, device):
        self.text_config = text_config
        self.checkpoint = checkpoint
        self.device = device
        embed, mixer, head = checkpoint.bookends()
        self.embed = embed.to(device)
        self.mixer = build(lambda: Qwen4ExpTextGatedResidual(text_config, use_combine=False), mixer, device)
        self.lm_head = build(
            lambda: torch.nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False), head, device
        )
        self.rotary = Qwen4ExpTextRotaryEmbedding(config=text_config).to(device)

    def layer(self, layer: int) -> Qwen4ExpTextDecoderLayer:
        state, pieces = self.checkpoint.layer(layer)

        def factory():
            module = Qwen4ExpTextDecoderLayer(self.text_config, layer)
            if module.ple is not None:
                module.ple.ple_embedding.ngram_embedding = ShardedEmbedding(pieces)
            return module

        return build(factory, state, self.device)

    def inputs(self, token_ids: torch.Tensor, start: int):
        """What one pass feeds every layer: the embedding stream, the position tables and the causal mask."""
        embeds = F.embedding(token_ids, self.embed)
        tokens = token_ids.shape[1]
        kv_len = start + tokens
        # Every position so far: attention slices the current rows out of the
        # tables, the indexer reads the keys' positions from them.
        positions = torch.arange(kv_len, device=self.device).view(1, 1, -1).expand(3, 1, -1)
        # The additive mask the eager attention path reads, causal from `start`;
        # the indexer derives its visible set from it as well.
        query_positions = torch.arange(start, kv_len, device=self.device)[:, None]
        visible = torch.arange(kv_len, device=self.device)[None, :] <= query_positions
        attention_mask = torch.zeros(tokens, kv_len, dtype=embeds.dtype, device=self.device)
        attention_mask = attention_mask.masked_fill(~visible, torch.finfo(embeds.dtype).min)[None, None]
        return {
            "stream": embeds.repeat(1, 1, self.text_config.hc_count),
            "position_embeddings": self.rotary(embeds, positions),
            "attention_mask": attention_mask,
            "ple_input_ids": token_ids,
        }

    def run(self, ids: torch.Tensor, layers: list[int], decode: int, feed: dict[str, torch.Tensor] | None = None):
        """The prefill of the first `tokens - decode` prompt tokens, then one step per remaining token.

        Returns every recorded stage of every pass, 2-D and contiguous: `stream_in`
        (the embedding stream), each layer's stages under `layer{N}_`, `mixed` and
        `logits`; the decode passes carry a `decode{i}_` prefix on top. Given `feed`,
        another run's recording, every layer and the head read that run's input for
        them instead of this run's own chain.
        """
        ids = ids.to(self.device)
        prefill = ids.shape[1] - decode
        passes = [("", ids[:, :prefill], 0)]
        passes += [(f"decode{step}_", ids[:, prefill + step : prefill + step + 1], prefill + step) for step in range(decode)]
        flat = {}

        def collect(prefix: str, name: str, value: torch.Tensor):
            value = value.detach()
            if value.dim() >= 3:
                value = value.reshape(-1, value.shape[-1])
            flat[prefix + name] = value.contiguous().cpu()

        inputs = [self.inputs(token_ids, start) for _, token_ids, start in passes]
        streams = [pass_inputs.pop("stream") for pass_inputs in inputs]
        for (prefix, _, _), stream in zip(passes, streams):
            collect(prefix, "stream_in", stream)
        cache = DynamicCache(config=self.text_config)
        with torch.no_grad():
            for index, layer in enumerate(layers):
                module = self.layer(layer)
                recorder = Recorder(module)
                source = f"layer{layers[index - 1]}_stream_out" if index else "stream_in"
                for p, (prefix, _, _) in enumerate(passes):
                    if feed is not None:
                        streams[p] = feed[prefix + source].to(self.device).view_as(streams[p])
                    streams[p] = module(streams[p], past_key_values=cache, conv_mask=None, **inputs[p])
                    captured = recorder.take()
                    captured["stream_out"] = streams[p]
                    if module.layer_type == "linear_attention":
                        # The cache updates this tensor in place, so each pass keeps its own copy.
                        captured["gdn_state"] = cache.layers[layer].recurrent_states[0].clone()
                    for name, value in captured.items():
                        collect(prefix, f"layer{layer}_{name}", value)
                recorder.close()
                del module
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            for (prefix, _, _), stream in zip(passes, streams):
                if feed is not None:
                    stream = feed[f"{prefix}layer{layers[-1]}_stream_out"].to(self.device).view_as(stream)
                mixed = self.mixer(stream)
                collect(prefix, "mixed", mixed)
                collect(prefix, "logits", self.lm_head(mixed))
        return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Qwen3.8-Flash-Next decoder layers with their bookends on real weights")
    parser.add_argument("--model", required=True, type=Path, help="checkpoint directory (sharded safetensors)")
    parser.add_argument("--layers", default="0", help="comma separated layer indices or a-b ranges, increasing, or all")
    parser.add_argument("--tokens", type=int, default=32, help="prompt tokens kept")
    parser.add_argument("--ids", help="comma separated token ids to record in place of the fixed prompt, all of them kept")
    parser.add_argument("--decode", type=int, default=0, help="prompt tokens fed one at a time after the prefill")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    config = Qwen4ExpConfig.from_pretrained(args.model)
    text_config = config.text_config
    # The arithmetic reference for attention, with the additive mask built above.
    text_config._attn_implementation = "eager"
    layers = parse_layers(args.layers, text_config.num_hidden_layers)
    if args.ids:
        ids = torch.tensor([[int(token) for token in args.ids.split(",")]], dtype=torch.int64)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        ids = tokenizer(PROMPT, return_tensors="pt").input_ids[:, : args.tokens]
        if ids.shape[1] != args.tokens:
            raise SystemExit(f"the prompt tokenizes to {ids.shape[1]} tokens, fewer than the {args.tokens} asked for")
    tokens = ids.shape[1]
    if not 0 <= args.decode < tokens:
        raise SystemExit(f"--decode {args.decode} must leave at least one prompt token of the {tokens}")

    checkpoint = Checkpoint(args.model)
    kinds = Counter(text_config.layer_types[layer] for layer in layers)
    print(f"{len(layers)} layers: " + ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items())))
    device = torch.device(args.device)
    accel = Chain(text_config, checkpoint, device).run(ids, layers, args.decode)
    for layer in layers:
        hits = int(torch.unique(accel[f"layer{layer}_selected_experts"]).numel())
        print(f"{device} layer {layer}: experts hit {hits}/{text_config.num_experts}")

    host = Chain(text_config, checkpoint, torch.device("cpu")).run(ids, layers, args.decode, feed=accel)
    floors = [name for name in host if stage_of(name) in CPU_STAGES]
    worst: dict[str, tuple[float, str]] = {}
    for name in floors:
        gap = float((accel[name].float() - host[name].float()).abs().max())
        if gap >= worst.get(stage_of(name), (-1.0, ""))[0]:
            worst[stage_of(name)] = (gap, name)
    for stage, (gap, name) in worst.items():
        print(f"{stage}: accelerator vs cpu floor up to {gap:.3e} at {name}")

    tensors = {"input_ids": ids.reshape(-1).to(torch.int64)}
    tensors.update({name: value for name, value in accel.items()})
    tensors.update({f"cpu_{name}": host[name] for name in floors})
    manifest = {
        "format_version": FORMAT_VERSION,
        "layers": layers,
        "layer_types": [text_config.layer_types[layer] for layer in layers],
        "tokens": tokens,
        "prefill_tokens": tokens - args.decode,
        "decode_steps": args.decode,
        "prompt": None if args.ids else PROMPT,
        "config_sha256": sha256_of(args.model / "config.json"),
        "index_sha256": sha256_of(args.model / "model.safetensors.index.json"),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata={METADATA_KEY: json.dumps(manifest, sort_keys=True)})
    print(f"wrote {args.out} ({args.out.stat().st_size / 2**20:.1f} MiB, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
