"""Adapter for fp8_gemm.* — the wide-route fp8 groupwise GEMM (rows > 8).

Production symbol: `glm52_fp8_groupwise_gemm_sm100_cuda`
(csrc/glm52/glm52_fp8_gemm.cu — one extern "C" shim over FlashInfer's
CutlassGroupwiseScaledGEMMSM100<1, 128, 128, ScaleMajorK, MmaSM=2,
e4m3 -> bf16>; FFI mirror pegainfer-kernels/src/ffi/glm52.rs; production call
pegainfer-glm52/src/fp8.rs fp8_linear_large_m_into via the
Glm52Fp8GemmScratch wide route — rows past FP8_GEMV_MAX_ROWS = 8, #812).

Four manifests share this adapter, one per whitelisted projection shape that
the A/B against the GEMV routes needs: q_b [16384, 2048], o_proj
[6144, 16384], shared gate|up [4096, 6144], shared down [6144, 2048].

Activation is generated bf16 and quantized with the production-mirroring
per-token-group helper (refs/quant.fp8_per_token_group_quant_ref, ue8m0
off — the GEMM's SFA is plain f32), so the kernel sees exactly the
production wide route's input pair (e4m3 bytes + f32 [rows, k/128] scales).
Weight uses the data.normal_quantized_fp8 block recipe. The FFI
requires m % 4 == 0 and k % 128 == 0; every axis row and every manifest
shape satisfies both by construction (48 is the #813 full-occupancy verify
bucket).

The SM100 CUTLASS template instantiates only for sm_100a-family targets:
on any other arch the symbol returns CUDA_ERROR_NOT_SUPPORTED (stub build) —
surfaced here as KernelLaunchError, never silent. Rows are deterministic per
shape but NOT bit-identical to the GEMV/mma routes — the torch-tolerance
gate covers the reorder.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import fp8_gemm as ref_mod
from kernel_lab.refs import quant as quant_ref

SYMBOL = "glm52_fp8_groupwise_gemm_sm100_cuda"

# Mirrors pegainfer-glm52/src/fp8.rs FP8_GEMM_WORKSPACE_BYTES — the CUTLASS
# grouped-scheduler/tensormap workspace the FlashInfer entry carves
# internally; allocation is bench warmup material, never timed.
WORKSPACE_BYTES = 32 << 20


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "n", "k"} at production capacity (all rows live)."""
    torch = require_torch()
    rows, n, k = shape["rows"], shape["n"], shape["k"]
    act_bf16 = data.normal_bf16((rows, k), seed=data.derive_seed(seed, "act"))
    act_q, act_s = quant_ref.fp8_per_token_group_quant_ref(act_bf16)
    weight, w_scales = data.normal_quantized_fp8(
        n, k, seed=data.derive_seed(seed, "weight")
    )
    dev = act_q.device
    return {
        "act_q": act_q.contiguous(),            # e4m3 uint8 [rows, k]
        "act_s": act_s.contiguous(),            # f32 [rows, k/128]
        "weight": weight,                       # e4m3 uint8 [n, k]
        "w_scales": w_scales.contiguous(),      # f32 [n/128, k/128]
        "out": torch.empty((rows, n), dtype=torch.bfloat16, device=dev),
        "workspace": torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=dev),
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["act_q"]),
        as_dev_ptr(tensors["act_s"]),
        as_dev_ptr(tensors["weight"]),
        as_dev_ptr(tensors["w_scales"]),
        as_dev_ptr(tensors["out"]),
        as_dev_ptr(tensors["workspace"]),
        WORKSPACE_BYTES,
        shape["rows"], shape["n"], shape["k"],
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    return ref_mod.groupwise_gemm_ref(
        tensors["act_q"], tensors["act_s"], tensors["weight"],
        tensors["w_scales"], shape["rows"], shape["n"], shape["k"],
    )
