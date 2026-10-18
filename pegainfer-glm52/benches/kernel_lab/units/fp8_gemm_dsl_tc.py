"""Adapter for fp8_gemm_dsl_tc.* — CuTe DSL tcgen05 fp8 groupwise GEMM.

Same unit family as fp8_gemm.* (identical inputs/outputs/reference — this
adapter delegates `make_inputs`/`reference` to `fp8_gemm_groupwise`), but the
GEMM is the tcgen05 CuTe DSL kernel
(`units/fp8_gemm_dsl_tc_kernel.py`, `Fp8BlockwiseGemmTcgen05`): TMA loads +
tcgen05.mma (1-CTA, M=64/N=128/K=128 tile) into a double-buffered TMEM block
accumulator, per 128-K block folded into the f32 register total acc by
SFA x SFB (blockwise algorithm, same as the CUTLASS sm_103 route).
JIT-compiled per (rows, n, k) with `cute.compile`; compile latency is
absorbed by bench warmup.

`SYMBOL` is a placeholder — there is no .so symbol; manifests carry
`capability.python_native = true` and `test_registry` skips the .so symbol
resolution check for such units. `capability.sm_tcgen05_only = true`
fail-closes the check/bench/compare paths unless the device major is 10 —
tcgen05 exists only on sm_100a-family parts, and elsewhere the run would
die deep inside the DSL runtime instead of at this boundary.

sm_103 target (GB300). Tile config: BLOCK_M=64, BLOCK_K=128 (one scale block
per MMA tile), BLOCK_N in {64, 128} and optional split-K per shape (TILE_CFG
below), 4 acc warps + 1 TMA/MMA warp, TMEM acc stages / AB smem stages
auto-sized, f32 partials + `_SplitKReduce` when split_k > 1.
"""
from __future__ import annotations

from kernel_lab.units import fp8_gemm_groupwise

SYMBOL = "cutedsl_fp8_blockwise_gemm_tcgen05"

# Per-(n, k) (block_n, split_k) tile config. block_n is the MMA N tile
# (128 = one CTA per 128-N scale block; 64 splits it in two); split_k > 1
# adds grid.z CTAs folding disjoint 128-K-block ranges into f32 partials
# reduced by a second DSL kernel. Both knobs raise CTA occupancy for
# CTA-starved shapes (148 SMs on GB300; the per-CTA TMA issue rate is the
# limiter there). q_b keeps (128, 1) — 128 CTAs, already 83% of achievable
# BW; N=64 (256 CTAs = 1.73 waves) and split-K both regressed it.
TILE_CFG = {
    (16384, 2048): (128, 1),   # q_b: 128 CTAs
    (6144, 16384): (128, 2),   # o_proj: 96 CTAs
    (4096, 6144): (128, 4),    # shared gate|up: 128 CTAs
    (6144, 2048): (64, 1),     # shared down: 96 CTAs (short-k: split-K's
                               # fixed prologue+reduce overhead exceeds the
                               # CTA-parallelism gain; tile N=64 alone wins)
}
# rows > 64 spans multiple 64-row M tiles: the CTA count doubles on its own
# and the starvation the table above compensates for is gone — split-K and
# tile-N shrinking flip to net losses (sweep 2026-08-04, rows=96). gate|up
# keeps split-K x2: its 32-CTA-per-tile grid is still under half a wave.
TILE_CFG_MULTI_M = {
    (16384, 2048): (128, 1),
    (6144, 16384): (128, 1),
    (4096, 6144): (128, 2),
    (6144, 2048): (128, 1),
}
DEFAULT_CFG = (128, 1)
MMA_TILE_M = 64


def tile_cfg(rows: int, n: int, k: int) -> tuple:
    table = TILE_CFG if rows <= MMA_TILE_M else TILE_CFG_MULTI_M
    return table.get((n, k), DEFAULT_CFG)


def make_inputs(shape: dict, seed: int) -> dict:
    return fp8_gemm_groupwise.make_inputs(shape, seed)


def reference(tensors: dict, shape: dict):
    return fp8_gemm_groupwise.reference(tensors, shape)


def run(lib, tensors: dict, shape: dict, stream) -> None:
    # `lib` is unused: the kernel is a DSL JIT executor, not a .so symbol.
    from kernel_lab.units import fp8_gemm_dsl_tc_kernel

    block_n, split_k = tile_cfg(shape["rows"], shape["n"], shape["k"])
    fp8_gemm_dsl_tc_kernel.run_fp8_blockwise_gemm_tc(
        shape["rows"],
        shape["n"],
        shape["k"],
        tensors["act_q"],
        tensors["act_s"],
        tensors["weight"],
        tensors["w_scales"],
        tensors["out"],
        stream.value if stream.value else 0,
        block_n=block_n,
        split_k=split_k,
    )
