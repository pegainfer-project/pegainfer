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
    Qwen4ExpTextRotaryEmbedding,
)

# One decoder layer of Qwen3.8-Flash-Next on the checkpoint's own weights: the
# reference `Qwen4ExpTextDecoderLayer` runs once on the accelerator and once on
# the CPU over a fixed prompt, and every stage the engine can be held against is
# recorded. With `--decode D` the last D prompt tokens are fed one at a time on
# the layer's cache after a prefill of the rest, so the recorded steps hold the
# engine's decode path on a carried state. Only the layer's own tensors and the
# token embedding are read, so a single GPU is enough for any layer without a
# PLE block.
#
#     python tools/accuracy/dump_qwen4_layer_golden.py --model /data/models/Qwen3.8-Flash-Next \
#         --layer 0 --out golden/qwen4-layer0.safetensors

METADATA_KEY = "qwen4_layer_golden"
FORMAT_VERSION = 3
# Recorded from the CPU pass as well: the accelerator versus CPU spread of the
# reference is the floor the engine is held to at these stages.
CPU_STAGES = ("attn_out", "stream_after_attn", "router_logits", "moe_out", "stream_out", "gdn_state")
PROMPT = (
    "The hyper-connection stream carries four copies of the residual through every "
    "layer, and the mixer reads them back before the head. Routing picks ten of the "
    "five hundred and twelve experts per token."
)
LAYER_PREFIX = "model.language_model.layers.{layer}."
EMBED_NAME = "model.language_model.embed_tokens.weight"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_of(name: str) -> str:
    """The stage behind an optional `decode{i}_` prefix."""
    head, _, tail = name.partition("_")
    return tail if head.startswith("decode") and head[len("decode") :].isdigit() else name


def load_layer_state(model_dir: Path, layer: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """The layer's tensors keyed as the module names them, and the token embedding."""
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = LAYER_PREFIX.format(layer=layer)
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name.startswith(prefix):
            by_shard.setdefault(shard, []).append(name)
    if any(".ple." in name for names in by_shard.values() for name in names):
        raise SystemExit(f"layer {layer} carries the PLE block, whose n-gram table this dump does not load")
    state = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(model_dir / shard, framework="pt") as f:
            for name in names:
                state[name[len(prefix) :]] = f.get_tensor(name)
    with safe_open(model_dir / weight_map[EMBED_NAME], framework="pt") as f:
        embed = f.get_tensor(EMBED_NAME)
    return state, embed


def build_layer(text_config, layer: int, state: dict[str, torch.Tensor], device) -> Qwen4ExpTextDecoderLayer:
    """The reference layer in bf16 with the checkpoint's tensors, nothing initialised."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            module = Qwen4ExpTextDecoderLayer(text_config, layer)
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


def run(layer: Qwen4ExpTextDecoderLayer, text_config, ids: torch.Tensor, embed: torch.Tensor, device, decode: int):
    """The prefill of the first `tokens - decode` prompt tokens, then one step per remaining token.

    Returns every recorded stage of every pass, 2-D and contiguous; the decode
    passes carry a `decode{i}_` prefix and a linear attention layer also records
    its recurrent state after each pass.
    """
    ids = ids.to(device)
    embed = embed.to(device)
    rotary = Qwen4ExpTextRotaryEmbedding(config=text_config).to(device)
    # A linear attention layer always runs on a cache, so its recurrent state is
    # recorded whether or not steps follow; the prefill arithmetic is the same.
    cache = DynamicCache(config=text_config) if layer.layer_type == "linear_attention" else None
    recorder = Recorder(layer)

    def forward(token_ids: torch.Tensor, start: int):
        embeds = F.embedding(token_ids, embed)
        stream = embeds.repeat(1, 1, text_config.hc_count)
        tokens = token_ids.shape[1]
        positions = torch.arange(start, start + tokens, device=device).view(1, 1, -1).expand(3, 1, -1)
        position_embeddings = rotary(embeds, positions)
        # The additive causal mask the eager attention path reads; the indexer
        # derives its visible set from it as well.
        visible = torch.ones(tokens, tokens, dtype=torch.bool, device=device).tril()
        attention_mask = torch.zeros(tokens, tokens, dtype=embeds.dtype, device=device)
        attention_mask = attention_mask.masked_fill(~visible, torch.finfo(embeds.dtype).min)[None, None]
        with torch.no_grad():
            out = layer(
                stream,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                conv_mask=None,
                past_key_values=cache,
                ple_input_ids=token_ids,
            )
        stages = {"stream_in": stream, **recorder.take(), "stream_out": out}
        if cache is not None:
            # The cache updates this tensor in place, so each pass keeps its own copy.
            stages["gdn_state"] = cache.layers[layer.linear_attn.layer_idx].recurrent_states[0].clone()
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
    recorder.close()
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one Qwen3.8-Flash-Next decoder layer on real weights")
    parser.add_argument("--model", required=True, type=Path, help="checkpoint directory (sharded safetensors)")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32, help="prompt tokens kept")
    parser.add_argument("--decode", type=int, default=0, help="prompt tokens fed one at a time after the prefill")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    config = Qwen4ExpConfig.from_pretrained(args.model)
    text_config = config.text_config
    # The arithmetic reference for attention, with the additive mask built above.
    text_config._attn_implementation = "eager"
    if args.decode and text_config.layer_types[args.layer] != "linear_attention":
        raise SystemExit("decode steps are recorded for linear attention layers only")
    if not 0 <= args.decode < args.tokens:
        raise SystemExit(f"--decode {args.decode} must leave at least one prompt token of the {args.tokens}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids[:, : args.tokens]
    if ids.shape[1] != args.tokens:
        raise SystemExit(f"the prompt tokenizes to {ids.shape[1]} tokens, fewer than the {args.tokens} asked for")

    state, embed = load_layer_state(args.model, args.layer)
    print(f"layer {args.layer}: {len(state)} tensors, layer_type={text_config.layer_types[args.layer]}")
    device = torch.device(args.device)
    layer = build_layer(text_config, args.layer, state, device)
    accel = run(layer, text_config, ids, embed, device, args.decode)
    hits = int(torch.unique(accel["selected_experts"]).numel())
    print(f"{device}: experts hit {hits}/{text_config.num_experts}")

    cpu_layer = copy.deepcopy(layer).cpu()
    host = run(cpu_layer, text_config, ids, embed, torch.device("cpu"), args.decode)
    floors = [name for name in host if stage_of(name) in CPU_STAGES]
    for name in floors:
        diff = (accel[name].float() - host[name].float()).abs()
        print(f"{name}: accelerator vs cpu max {diff.max():.3e} mean {diff.mean():.3e}")

    tensors = {"input_ids": ids.reshape(-1).to(torch.int64)}
    tensors.update({name: value for name, value in accel.items()})
    tensors.update({f"cpu_{name}": host[name] for name in floors})
    manifest = {
        "format_version": FORMAT_VERSION,
        "layer": args.layer,
        "layer_type": text_config.layer_types[args.layer],
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
