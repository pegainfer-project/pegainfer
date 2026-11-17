"""Write a bf16 copy of a Gemma 4 NVFP4 checkpoint, for HF to load as a reference tower.

    python tools/accuracy/dequantize_gemma4_nvfp4.py <nvfp4-dir> <out-dir> --device cuda:0

ModelOpt's `NVFP4QTensor` unpacks the quantized weights; the per-expert weights
are then restacked into the fused `gate_up_proj` and `down_proj` parameters
transformers declares.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

GROUP = 16
DTYPE = torch.bfloat16
SHARD_BYTES = 4 << 30
SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")
EXPERT = re.compile(
    r"^(.*\.layers\.\d+)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
SKIP_FILES = {"hf_quant_config.json", "model.safetensors.index.json"}


class Checkpoint:
    def __init__(self, root: Path, weight_map: dict[str, str], device: str):
        self.root = root
        self.map = weight_map
        self.device = device
        self.handles: dict[str, object] = {}

    def get(self, key: str) -> torch.Tensor:
        shard = self.map[key]
        handle = self.handles.get(shard)
        if handle is None:
            handle = self.handles[shard] = safe_open(self.root / shard, framework="pt")
        return handle.get_tensor(key).to(self.device)


def dequantize(src: Checkpoint, base: str) -> torch.Tensor:
    packed = src.get(f"{base}.weight")
    shape = list(packed.shape)
    shape[-1] *= 2
    tensor = NVFP4QTensor(torch.Size(shape), DTYPE, packed)
    return tensor.dequantize(
        dtype=DTYPE,
        scale=src.get(f"{base}.weight_scale"),
        double_scale=src.get(f"{base}.weight_scale_2"),
        block_sizes={-1: GROUP},
    )


def weight(src: Checkpoint, base: str) -> torch.Tensor:
    if f"{base}.weight_scale" in src.map:
        return dequantize(src, base)
    return src.get(f"{base}.weight").to(DTYPE)


def expected_keys(model_dir: Path) -> set[str]:
    config = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    keys = set(model.state_dict().keys())
    if config.get_text_config().tie_word_embeddings:
        keys.discard("lm_head.weight")
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    index_path = args.src / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    src = Checkpoint(args.src, weight_map, args.device)
    args.dst.mkdir(parents=True, exist_ok=True)

    experts: dict[str, set[int]] = {}
    plain: list[str] = []
    for key in weight_map:
        if key.endswith(SCALE_SUFFIXES):
            continue
        match = EXPERT.match(key)
        if match:
            experts.setdefault(match.group(1), set()).add(int(match.group(2)))
        else:
            plain.append(key)

    shard: dict[str, torch.Tensor] = {}
    held = 0
    index: dict[str, str] = {}
    total = 0

    def emit(name: str, tensor: torch.Tensor) -> None:
        nonlocal held
        shard[name] = tensor.to("cpu").contiguous()
        held += shard[name].numel() * shard[name].element_size()
        if held >= SHARD_BYTES:
            flush()

    def flush() -> None:
        nonlocal held, total
        if not shard:
            return
        name = f"model-{len(set(index.values())) + 1:05d}.safetensors"
        save_file(shard, args.dst / name, metadata={"format": "pt"})
        index.update(dict.fromkeys(shard, name))
        total += held
        print(f"wrote {name}: {len(shard)} tensors, {held / 2**30:.1f} GiB", flush=True)
        shard.clear()
        held = 0

    for key in sorted(plain):
        if key.endswith(".weight"):
            emit(key, weight(src, key[: -len(".weight")]))
        else:
            tensor = src.get(key)
            emit(key, tensor.to(DTYPE) if tensor.is_floating_point() else tensor)

    for prefix in sorted(experts):
        count = len(experts[prefix])
        gate_up = torch.stack(
            [
                torch.cat(
                    [
                        weight(src, f"{prefix}.experts.{i}.gate_proj"),
                        weight(src, f"{prefix}.experts.{i}.up_proj"),
                    ],
                    dim=0,
                )
                for i in range(count)
            ]
        )
        down = torch.stack(
            [weight(src, f"{prefix}.experts.{i}.down_proj") for i in range(count)]
        )
        emit(f"{prefix}.experts.gate_up_proj", gate_up)
        emit(f"{prefix}.experts.down_proj", down)
        print(
            f"{prefix}: {count} experts -> {tuple(gate_up.shape)}, {tuple(down.shape)}",
            flush=True,
        )
    flush()

    want = expected_keys(args.src)
    produced = set(index)
    if produced != want:
        raise SystemExit(
            f"produced {len(produced)} keys, transformers declares {len(want)}: "
            f"missing {sorted(want - produced)[:4]}, extra {sorted(produced - want)[:4]}"
        )

    (args.dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": index}, indent=1)
    )
    for path in args.src.iterdir():
        if (
            path.is_file()
            and path.name not in SKIP_FILES
            and path.suffix != ".safetensors"
        ):
            shutil.copy2(path, args.dst / path.name)
    config = json.loads((args.dst / "config.json").read_text())
    config.pop("quantization_config", None)
    (args.dst / "config.json").write_text(json.dumps(config, indent=1))
    shards = len(set(index.values()))
    print(f"bf16 checkpoint at {args.dst}: {total / 2**30:.1f} GiB in {shards} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
