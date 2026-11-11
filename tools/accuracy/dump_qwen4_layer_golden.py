import argparse
import copy
import hashlib
import json
import platform
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
# untied head. The reference runs once on the accelerator and once on the CPU
# over a fixed prompt, and every stage the engine can be held against is
# recorded, each layer's under a `layer{N}_` prefix. With `--decode D` the last D
# prompt tokens are fed one at a time on the layers' caches after a prefill of
# the rest, so the recorded steps hold the engine's decode path on a carried
# state. Only the layers' own tensors and the bookends are read, so a single GPU
# is enough for a few layers without a PLE block.
#
#     python tools/accuracy/dump_qwen4_layer_golden.py --model /data/models/Qwen3.8-Flash-Next \
#         --layers 0,3 --decode 8 --out golden/qwen4-layers03-decode8.safetensors

METADATA_KEY = "qwen4_layer_golden"
FORMAT_VERSION = 4
# Recorded from the CPU pass as well: the accelerator versus CPU spread of the
# reference is the floor the engine is held to at these stages.
CPU_STAGES = ("attn_out", "stream_after_attn", "router_logits", "moe_out", "stream_out", "gdn_state", "logits")
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


def load_state(model_dir: Path, layers: list[int]):
    """Each layer's tensors keyed as the module names them, and the bookends."""
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    wanted: dict[str, tuple[str, str]] = {EMBED_NAME: ("embed", "weight"), HEAD_NAME: ("lm_head", "weight")}
    for name in weight_map:
        if name.startswith(MIXER_PREFIX):
            wanted[name] = ("mixer", name[len(MIXER_PREFIX) :])
        for layer in layers:
            prefix = LAYER_PREFIX.format(layer=layer)
            if name.startswith(prefix):
                if ".ple." in name:
                    raise SystemExit(f"layer {layer} carries the PLE block, whose n-gram table this dump does not load")
                wanted[name] = (f"layer{layer}", name[len(prefix) :])
    by_shard: dict[str, list[str]] = {}
    for name in wanted:
        by_shard.setdefault(weight_map[name], []).append(name)
    state: dict[str, dict[str, torch.Tensor]] = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(model_dir / shard, framework="pt") as f:
            for name in names:
                group, key = wanted[name]
                state.setdefault(group, {})[key] = f.get_tensor(name)
    return state


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


class Recorder:
    """Forward hooks on the submodules whose boundaries the engine reproduces."""

    def __init__(self, layer: Qwen4ExpTextDecoderLayer):
        self.captured: dict[str, torch.Tensor] = {}
        self.handles = []
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
    """The reference modules for one device: the layers in order and the bookends."""

    def __init__(self, text_config, layers: list[int], state, device):
        self.text_config = text_config
        self.device = device
        self.embed = state["embed"]["weight"].to(device)
        self.layers = [
            (layer, build(lambda layer=layer: Qwen4ExpTextDecoderLayer(text_config, layer), state[f"layer{layer}"], device))
            for layer in layers
        ]
        self.mixer = build(lambda: Qwen4ExpTextGatedResidual(text_config, use_combine=False), state["mixer"], device)
        self.lm_head = build(
            lambda: torch.nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False),
            state["lm_head"],
            device,
        )
        self.rotary = Qwen4ExpTextRotaryEmbedding(config=text_config).to(device)

    def to(self, device):
        other = copy.copy(self)
        other.device = device
        other.embed = self.embed.to(device)
        other.layers = [(layer, copy.deepcopy(module).to(device)) for layer, module in self.layers]
        other.mixer = copy.deepcopy(self.mixer).to(device)
        other.lm_head = copy.deepcopy(self.lm_head).to(device)
        other.rotary = copy.deepcopy(self.rotary).to(device)
        return other

    def run(self, ids: torch.Tensor, decode: int):
        """The prefill of the first `tokens - decode` prompt tokens, then one step per remaining token.

        Returns every recorded stage of every pass, 2-D and contiguous: `stream_in`
        (the embedding stream), each layer's stages under `layer{N}_`, `mixed` and
        `logits`; the decode passes carry a `decode{i}_` prefix on top.
        """
        ids = ids.to(self.device)
        cache = DynamicCache(config=self.text_config)
        recorders = [(layer, module, Recorder(module)) for layer, module in self.layers]

        def forward(token_ids: torch.Tensor, start: int):
            embeds = F.embedding(token_ids, self.embed)
            stream = embeds.repeat(1, 1, self.text_config.hc_count)
            tokens = token_ids.shape[1]
            kv_len = start + tokens
            # Every position so far: attention slices the current rows out of the
            # tables, the indexer reads the keys' positions from them.
            positions = torch.arange(kv_len, device=self.device).view(1, 1, -1).expand(3, 1, -1)
            position_embeddings = self.rotary(embeds, positions)
            # The additive mask the eager attention path reads, causal from `start`;
            # the indexer derives its visible set from it as well.
            query_positions = torch.arange(start, kv_len, device=self.device)[:, None]
            visible = torch.arange(kv_len, device=self.device)[None, :] <= query_positions
            attention_mask = torch.zeros(tokens, kv_len, dtype=embeds.dtype, device=self.device)
            attention_mask = attention_mask.masked_fill(~visible, torch.finfo(embeds.dtype).min)[None, None]
            stages = {"stream_in": stream}
            with torch.no_grad():
                for layer, module, recorder in recorders:
                    stream = module(
                        stream,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        conv_mask=None,
                        past_key_values=cache,
                        ple_input_ids=token_ids,
                    )
                    captured = recorder.take()
                    captured["stream_out"] = stream
                    if module.layer_type == "linear_attention":
                        # The cache updates this tensor in place, so each pass keeps its own copy.
                        captured["gdn_state"] = cache.layers[layer].recurrent_states[0].clone()
                    stages.update({f"layer{layer}_{name}": value for name, value in captured.items()})
                mixed = self.mixer(stream)
                stages["mixed"] = mixed
                stages["logits"] = self.lm_head(mixed)
            return stages

        flat = {}

        def collect(prefix: str, stages: dict[str, torch.Tensor]):
            for name, value in stages.items():
                value = value.detach()
                if value.dim() >= 3:
                    value = value.reshape(-1, value.shape[-1])
                flat[prefix + name] = value.contiguous().cpu()

        prefill = ids.shape[1] - decode
        collect("", forward(ids[:, :prefill], 0))
        for step in range(decode):
            collect(f"decode{step}_", forward(ids[:, prefill + step : prefill + step + 1], prefill + step))
        for _, _, recorder in recorders:
            recorder.close()
        return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Qwen3.8-Flash-Next decoder layers with their bookends on real weights")
    parser.add_argument("--model", required=True, type=Path, help="checkpoint directory (sharded safetensors)")
    parser.add_argument("--layers", default="0", help="comma separated layer indices, in order")
    parser.add_argument("--tokens", type=int, default=32, help="prompt tokens kept")
    parser.add_argument("--decode", type=int, default=0, help="prompt tokens fed one at a time after the prefill")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    layers = [int(index) for index in args.layers.split(",")]
    if layers != sorted(set(layers)):
        raise SystemExit(f"--layers {args.layers} must be increasing")
    config = Qwen4ExpConfig.from_pretrained(args.model)
    text_config = config.text_config
    # The arithmetic reference for attention, with the additive mask built above.
    text_config._attn_implementation = "eager"
    if not 0 <= args.decode < args.tokens:
        raise SystemExit(f"--decode {args.decode} must leave at least one prompt token of the {args.tokens}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids[:, : args.tokens]
    if ids.shape[1] != args.tokens:
        raise SystemExit(f"the prompt tokenizes to {ids.shape[1]} tokens, fewer than the {args.tokens} asked for")

    state = load_state(args.model, layers)
    for layer in layers:
        print(f"layer {layer}: {len(state[f'layer{layer}'])} tensors, layer_type={text_config.layer_types[layer]}")
    device = torch.device(args.device)
    chain = Chain(text_config, layers, state, device)
    accel = chain.run(ids, args.decode)
    for layer in layers:
        hits = int(torch.unique(accel[f"layer{layer}_selected_experts"]).numel())
        print(f"{device} layer {layer}: experts hit {hits}/{text_config.num_experts}")

    host = chain.to(torch.device("cpu")).run(ids, args.decode)
    floors = [name for name in host if stage_of(name) in CPU_STAGES]
    for name in floors:
        diff = (accel[name].float() - host[name].float()).abs()
        print(f"{name}: accelerator vs cpu max {diff.max():.3e} mean {diff.mean():.3e}")

    tensors = {"input_ids": ids.reshape(-1).to(torch.int64)}
    tensors.update({name: value for name, value in accel.items()})
    tensors.update({f"cpu_{name}": host[name] for name in floors})
    manifest = {
        "format_version": FORMAT_VERSION,
        "layers": layers,
        "layer_types": [text_config.layer_types[layer] for layer in layers],
        "tokens": args.tokens,
        "prefill_tokens": args.tokens - args.decode,
        "decode_steps": args.decode,
        "prompt": PROMPT,
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
