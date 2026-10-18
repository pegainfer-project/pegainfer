#!/usr/bin/env python3
import argparse
import json
import tempfile
from pathlib import Path

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_moe_permute_scales,
    marlin_make_workspace_new,
)
from vllm.scalar_type import scalar_types
from safetensors.torch import safe_open


E = 48
HIDDEN = 7168
INTERMEDIATE = 2048
TOPK = 8
GROUP_SIZE = 32


def fill_qweight(numel: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(numel, device=device, dtype=torch.int64)
    values = (idx * 1103515245 + 12345 + (idx // 97) * 17) & 0x7FFFFFFF
    return values.to(torch.int32)


def fill_bf16(numel: int, modulo: int, scale: float, offset: float, device: torch.device):
    idx = torch.arange(numel, device=device, dtype=torch.int64)
    values = torch.remainder(idx, modulo).to(torch.float32)
    return ((values + offset) * scale).to(torch.bfloat16)


def build_inputs(tokens: int, block_size: int, device: torch.device):
    hidden = fill_bf16(tokens * HIDDEN, 23, 1.0 / 32.0, -11.0, device).view(
        tokens, HIDDEN
    )
    topk_ids = torch.empty((tokens, TOPK), device=device, dtype=torch.int32)
    for token in range(tokens):
        for route in range(TOPK):
            topk_ids[token, route] = (token * 13 + route * 5) % E
    weights = torch.arange(1, TOPK + 1, device=device, dtype=torch.float32)
    weights = weights / weights.sum()
    topk_weights = weights.repeat(tokens, 1).contiguous()
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size, E, None
    )
    return hidden, topk_ids, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded


def read_tensor(model_path: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = weight_map[name]
    with safe_open(str(model_path / shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def build_real_kimi_layer_weights(
    model_path: Path,
    layer_idx: int,
    rank: int,
    device: torch.device,
):
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    expert_start = rank * E
    expert_end = expert_start + E
    w13 = torch.empty(
        (E, HIDDEN // (32 // 4), 2 * INTERMEDIATE),
        device=device,
        dtype=torch.int32,
    )
    w2 = torch.empty(
        (E, INTERMEDIATE // (32 // 4), HIDDEN),
        device=device,
        dtype=torch.int32,
    )
    w13_scale = torch.empty(
        (E, HIDDEN // GROUP_SIZE, 2 * INTERMEDIATE),
        device=device,
        dtype=torch.bfloat16,
    )
    w2_scale = torch.empty(
        (E, INTERMEDIATE // GROUP_SIZE, HIDDEN),
        device=device,
        dtype=torch.bfloat16,
    )

    for local_expert, global_expert in enumerate(range(expert_start, expert_end)):
        prefix = f"language_model.model.layers.{layer_idx}.mlp.experts.{global_expert}"
        gate_weight = read_tensor(
            model_path, weight_map, f"{prefix}.gate_proj.weight_packed"
        ).t().contiguous()
        up_weight = read_tensor(
            model_path, weight_map, f"{prefix}.up_proj.weight_packed"
        ).t().contiguous()
        down_weight = read_tensor(
            model_path, weight_map, f"{prefix}.down_proj.weight_packed"
        ).t().contiguous()
        gate_scale = read_tensor(
            model_path, weight_map, f"{prefix}.gate_proj.weight_scale"
        ).t().contiguous()
        up_scale = read_tensor(
            model_path, weight_map, f"{prefix}.up_proj.weight_scale"
        ).t().contiguous()
        down_scale = read_tensor(
            model_path, weight_map, f"{prefix}.down_proj.weight_scale"
        ).t().contiguous()

        w13[local_expert, :, :INTERMEDIATE].copy_(gate_weight.to(device))
        w13[local_expert, :, INTERMEDIATE:].copy_(up_weight.to(device))
        w2[local_expert].copy_(down_weight.to(device))
        w13_scale[local_expert, :, :INTERMEDIATE].copy_(gate_scale.to(device))
        w13_scale[local_expert, :, INTERMEDIATE:].copy_(up_scale.to(device))
        w2_scale[local_expert].copy_(down_scale.to(device))

    empty_perm = torch.empty((E, 0), device=device, dtype=torch.int32)
    w13 = ops.gptq_marlin_moe_repack(
        w13,
        empty_perm,
        HIDDEN,
        2 * INTERMEDIATE,
        4,
        is_a_8bit=False,
    )
    w2 = ops.gptq_marlin_moe_repack(
        w2,
        empty_perm,
        INTERMEDIATE,
        HIDDEN,
        4,
        is_a_8bit=False,
    )
    w13_scale = marlin_moe_permute_scales(
        w13_scale, HIDDEN, 2 * INTERMEDIATE, GROUP_SIZE
    )
    w2_scale = marlin_moe_permute_scales(
        w2_scale, INTERMEDIATE, HIDDEN, GROUP_SIZE
    )
    return w13, w2, w13_scale, w2_scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--layer-idx", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or Path(tempfile.gettempdir()) / "kimi_marlin_wna16_reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    hidden, topk_ids, topk_weights, sorted_token_ids, expert_ids, num_tokens = build_inputs(
        args.tokens, args.block_size, device
    )
    route_elems = args.tokens * TOPK

    if args.model_path is None:
        w13 = fill_qweight(E * (HIDDEN // 16) * (2 * INTERMEDIATE * 2), device).view(
            E, HIDDEN // 16, 2 * INTERMEDIATE * 2
        )
        w2 = fill_qweight(E * (INTERMEDIATE // 16) * (HIDDEN * 2), device).view(
            E, INTERMEDIATE // 16, HIDDEN * 2
        )
        w13_scale_raw = fill_bf16(
            E * (HIDDEN // GROUP_SIZE) * (2 * INTERMEDIATE),
            17,
            1.0 / 64.0,
            1.0,
            device,
        ).view(E, HIDDEN // GROUP_SIZE, 2 * INTERMEDIATE)
        w2_scale_raw = fill_bf16(
            E * (INTERMEDIATE // GROUP_SIZE) * HIDDEN,
            19,
            1.0 / 64.0,
            1.0,
            device,
        ).view(E, INTERMEDIATE // GROUP_SIZE, HIDDEN)
        w13_scale = marlin_moe_permute_scales(
            w13_scale_raw, HIDDEN, 2 * INTERMEDIATE, GROUP_SIZE
        )
        w2_scale = marlin_moe_permute_scales(
            w2_scale_raw, INTERMEDIATE, HIDDEN, GROUP_SIZE
        )
        weight_source = "deterministic_synthetic"
    else:
        w13, w2, w13_scale, w2_scale = build_real_kimi_layer_weights(
            args.model_path, args.layer_idx, args.rank, device
        )
        weight_source = "kimi_checkpoint_marlin_repack"

    workspace = marlin_make_workspace_new(device, 4)
    quant_type = scalar_types.uint4b8

    w13_out = torch.empty((route_elems, 2 * INTERMEDIATE), device=device, dtype=torch.bfloat16)
    w13_out = ops.moe_wna16_marlin_gemm(
        hidden,
        w13_out,
        w13,
        None,
        w13_scale,
        None,
        None,
        None,
        None,
        None,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens,
        topk_weights,
        moe_block_size=args.block_size,
        top_k=TOPK,
        mul_topk_weights=False,
        b_q_type=quant_type,
        size_m=args.tokens,
        size_n=2 * INTERMEDIATE,
        size_k=HIDDEN,
        is_k_full=True,
        use_atomic_add=True,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    activated = torch.empty((route_elems, INTERMEDIATE), device=device, dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul(activated, w13_out)

    route_output = torch.empty((route_elems, HIDDEN), device=device, dtype=torch.bfloat16)
    route_output = ops.moe_wna16_marlin_gemm(
        activated,
        route_output,
        w2,
        None,
        w2_scale,
        None,
        None,
        None,
        None,
        None,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens,
        topk_weights,
        moe_block_size=args.block_size,
        top_k=1,
        mul_topk_weights=True,
        b_q_type=quant_type,
        size_m=route_elems,
        size_n=HIDDEN,
        size_k=INTERMEDIATE,
        is_k_full=True,
        use_atomic_add=True,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    final = torch.empty_like(hidden)
    torch.sum(route_output.view(args.tokens, TOPK, HIDDEN), dim=1, out=final)

    w13_out.cpu().contiguous().view(torch.uint16).numpy().tofile(
        out_dir / "w13_out_bf16.bin"
    )
    route_output.cpu().contiguous().view(torch.uint16).numpy().tofile(
        out_dir / "route_output_bf16.bin"
    )
    final.cpu().contiguous().view(torch.uint16).numpy().tofile(out_dir / "final_bf16.bin")
    metadata = {
        "engine": "vllm.moe_wna16_marlin_gemm",
        "weight_source": weight_source,
        "model_path": str(args.model_path) if args.model_path is not None else None,
        "layer_idx": args.layer_idx,
        "rank": args.rank,
        "tokens": args.tokens,
        "topk": TOPK,
        "local_experts": E,
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "group_size": GROUP_SIZE,
        "block_size": args.block_size,
        "w13_out_shape": [route_elems, 2 * INTERMEDIATE],
        "route_output_shape": [route_elems, HIDDEN],
        "final_shape": [args.tokens, HIDDEN],
        "quant_type": "uint4b8",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
