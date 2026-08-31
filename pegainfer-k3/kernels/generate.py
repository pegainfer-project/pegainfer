#!/usr/bin/env python3
"""Generate the K3 TileLang batched decode kernels (AOT).

Emits one `.cu` per kernel family under `--out-dir`, covering the batched
decode kernel set:

    k3_rms_norm_rbs_batched.cu      k3_conv_silu_batched.cu
    k3_land_batched.cu              k3_kda_core_batched.cu
    k3_land_rms_norm_rbs_batched.cu k3_attnres_scores_batched.cu
    k3_add2_batched.cu              k3_attnres_mix_batched.cu
    k3_mul_sigmoid_batched.cu       k3_situ_batched.cu

The batch size is a static compile-time dimension, so a single-stream step is
served by the `B = 1` instantiation of the same family — its per-row spelling
is word for word the certified single-row kernel, which is what the upstream
bitwise gate proves. There is therefore no separate bs=1 kernel set. The dense
projections are served by cuBLASLt, the routed experts by the DeepGEMM
masked grouped-GEMM chain, and MLA decode by the hand-written absorbed
paged-attention kernel (`csrc/k3/k3_mla_paged_attn.cu`), so neither a GEMV
family nor an attention family is generated here.

Each file is [shared TileLang preamble] + [one renamed `main_kernel` per
instantiation] + [one hand-written `extern "C"` dispatch launcher]. TileLang
always names the entry point `main_kernel`, so every instantiation is renamed
to a shape-tagged symbol before concatenation; the launcher is the only
non-generated code in the artifact and returns `cudaErrorInvalidValue` for a
configuration that was not instantiated. One family per translation unit is
also what lets nvcc compile them in parallel.

The kernel definitions themselves are the vendored, certified spellings in
`tilelang_defs.py` — this file only walks the configuration lists, checks the
codegen contract, and wraps. It never rewrites kernel bodies.

Output is a Cargo `OUT_DIR` build artifact and is never checked in. Run
standalone with `--vendor-includes` to produce a self-contained pre-generated
directory for build hosts that cannot run TileLang (see README.md).
"""

import argparse
import multiprocessing
import os
import re
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import tilelang
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

import tilelang_defs as defs

# TileLang releases change codegen in ways this generator depends on: the
# entry-point symbol name, the (alphabetically sorted) kernel parameter list
# asserted per family below, and the host-side encoding the launch geometry
# and the TMA descriptor constants are recovered from. Bumping requires
# re-checking all three AND re-running the upstream bitwise parity gate — do
# not just widen this.
KNOWN_GOOD_TILELANG = "0.1.12"
if tilelang.__version__ != KNOWN_GOOD_TILELANG:
    raise RuntimeError(
        f"tilelang {tilelang.__version__} is not the validated "
        f"{KNOWN_GOOD_TILELANG}; point PEGAINFER_K3_TILELANG_PYTHON at an env "
        "with the pinned version, or revalidate the codegen contract and bump "
        "KNOWN_GOOD_TILELANG"
    )

tilelang.set_log_level("WARNING")

# --------------------------------------------------------------------------- #
# K3 model configuration.
#
# Every constant below is the value `K3Engine.__init__` reads out of the K3
# text config, under the engine's own name; the shape arguments further down
# reproduce the engine's derivations from them. Change these together with the
# model config, never on their own.
# --------------------------------------------------------------------------- #

HIDDEN = 7168               # hidden_size                     -> engine H
RMS_EPS = 1e-5              # rms_norm_eps                    -> engine EPS
KDA_HEADS = 96              # linear_attn_config.num_heads    -> engine KH
KDA_HEAD_DIM = 128          # linear_attn_config.head_dim     -> engine KD
KDA_DIM = KDA_HEADS * KDA_HEAD_DIM                        # -> engine KP
CONV_WIDTH = 4              # short_conv_kernel_size          -> engine CW
GATE_LOWER_BOUND = -5.0     # gate_lower_bound                -> engine LB
MLA_HEADS = 96              # num_attention_heads             -> engine NH
QK_NOPE_DIM = 128           # qk_nope_head_dim
ROPE_DIM = 64               # qk_rope_head_dim                -> engine ROPE
QK_DIM = QK_NOPE_DIM + ROPE_DIM                           # -> engine QK
V_DIM = 128                 # v_head_dim                      -> engine VD
Q_LORA = 1536               # q_lora_rank                     -> engine QL
KV_LORA = 512               # kv_lora_rank                    -> engine KVL
LATENT = 3584               # routed_expert_hidden_size       -> engine LAT
MOE_INTER = 3072            # moe_intermediate_size           -> engine MI
SHARED_EXPERTS = 2          # num_shared_experts
SHARED_INTER = MOE_INTER * SHARED_EXPERTS                 # -> engine SI
DENSE_INTER = 33792         # intermediate_size               -> engine INT
TOPK = 16                   # num_experts_per_token           -> engine TOPK
VOCAB = 163840              # vocab_size                      -> engine V
NUM_LAYERS = 93             # num_hidden_layers               -> engine NL
ATTNRES_BLOCK_SIZE = 12     # attn_res_block_size             -> engine BS

# num_experts. The router and its top-k are instantiated for both the full
# expert table and the per-rank shard an expert-parallel deployment holds, so
# the same artifact serves a single-GPU and a 4-way-EP deployment.
EXPERTS = [896, 896 // 4]   # engine E, and engine Es under 4-way EP

# Segment counts of the partials `land`/`conv_silu` merge. The engine's
# producers are framework GEMMs (cuBLASLt, DeepGEMM), which emit a single
# segment, so only SK=1 is instantiated; the reference engine's SK=8 GEMV
# shapes have no launch site here and are not generated.
SPLIT_K = [1]

# `engine._wsm_n`: the beta / low-rank-gate projection is padded up to a
# multiple of 64 so its tile divides evenly.
WSM_N = (KDA_HEADS + KDA_HEAD_DIM + 63) // 64 * 64
# `engine`'s fused MLA projection width: q_a | kv_a | k_rope | output gate.
MLA_FUSED = Q_LORA + KV_LORA + ROPE_DIM + KDA_DIM

# Attention-residual candidate counts. The snapshot history grows one entry per
# `attn_res_block_size` layers, so a step sees every count from 1 up to the
# number of blocks in the model.
ATTNRES_BLOCKS = (NUM_LAYERS + ATTNRES_BLOCK_SIZE - 1) // ATTNRES_BLOCK_SIZE
ATTNRES_NB = list(range(1, ATTNRES_BLOCKS + 1))

# Default thread count of the vendored kernels; `kda_core` overrides it
# (head_dim).
THREADS = 256

# Batch buckets. A step with `rows` live rows runs the next bucket up and
# discards the tail rows, which keeps the number of compiled shapes — and the
# number of distinct CUDA Graphs — bounded.
B_BUCKETS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]

# Prefill chunk buckets: the chunked-prefill step runs the same batched
# families at chunk width, up to the MegaMoE protocol maximum (16896 rows).
# Every family gets the extended ladder except `kda_core` — chunks cross the
# KDA recurrence through FlashKDA, so the fused core only ever sees decode
# buckets. Mirrors `K3_PREFILL_BUCKETS` in `ops/k3_tilelang.rs`.
B_PREFILL_BUCKETS = [256, 512, 1024, 2048, 4224, 8448, 16896]
B_CHUNK_BUCKETS = B_BUCKETS + B_PREFILL_BUCKETS


# --------------------------------------------------------------------------- #
# Per-family shape lists, in `engine._warm_kernels` order.
# --------------------------------------------------------------------------- #

# rms_norm_rbs(H, eps): the layer norms (H), the MLA kv latent norm (KVL) and
# the routed-latent norm (LAT).
RMS_NORM_N = [HIDDEN, KV_LORA, LATENT]

# land(NT, N, OFF, SK): merge one column span of a (SK, NT) partial and land
# bf16 once. The engine's `lands` list, verbatim, plus the chunked-prefill
# conv-input landing (the sequential engine never lands that projection alone —
# its conv kernel casts in place; the chunk builds windows from the landed rows
# before the conv runs, so it needs the standalone cast).
LAND_CONFIGS = [
    #  NT                N                   OFF               engine call site
    (4 * KDA_DIM, KDA_DIM, 3 * KDA_DIM),              # KDA output gate
    (KDA_DIM, KDA_DIM, 0),                            # chunked-prefill conv inputs
    (WSM_N, KDA_HEADS, 0),                            # KDA beta
    (WSM_N, KDA_HEAD_DIM, KDA_HEADS),                 # KDA low-rank gate input
    (MLA_FUSED, KV_LORA + ROPE_DIM, Q_LORA),          # MLA kv_a|k_rope
    (MLA_FUSED, KDA_DIM, Q_LORA + KV_LORA + ROPE_DIM),  # MLA output gate
    (MLA_HEADS * QK_DIM, MLA_HEADS * QK_DIM, 0),      # MLA q_b
    (MLA_HEADS * 256, MLA_HEADS * 256, 0),            # MLA kv_b
    (HIDDEN, HIDDEN, 0),                              # o_proj / routed / shared
    (LATENT, LATENT, 0),                              # routed latent
    (2 * SHARED_INTER, SHARED_INTER, 0),              # shared gate
    (2 * SHARED_INTER, SHARED_INTER, SHARED_INTER),   # shared up
    (2 * DENSE_INTER, DENSE_INTER, 0),                # dense gate
    (2 * DENSE_INTER, DENSE_INTER, DENSE_INTER),      # dense up
    (VOCAB, VOCAB, 0),                                # logits
]

# land_rms_norm_rbs(NT, N, OFF, SK, eps): MLA's q_a, the one place a merge and
# a round-before-scale norm are fused.
LAND_RMS_NORM_CONFIGS = [(MLA_FUSED, Q_LORA, 0)]

# add2 / mul_sigmoid / situ / conv_silu / kda_core take a single width each;
# situ has two (shared, dense) — the routed-expert situ is fused into the
# masked-GEMM chain and the mega kernel.
ADD2_N = [HIDDEN]
MUL_SIGMOID_N = [KDA_DIM]
SITU_N = [SHARED_INTER, DENSE_INTER]


# --------------------------------------------------------------------------- #
# Codegen contract.
#
# TileLang emits the kernel parameters sorted by name, not in prim_func order;
# the launchers below bind arguments to these exact signatures.
# --------------------------------------------------------------------------- #

BF16 = "bfloat16_t"

RMS_NORM_PARAMS = (
    f"(const {BF16}* __restrict__ G, {BF16}* __restrict__ O, "
    f"const {BF16}* __restrict__ X)"
)
LAND_PARAMS = f"({BF16}* __restrict__ O, const float* __restrict__ P)"
LAND_RMS_NORM_PARAMS = (
    f"(const {BF16}* __restrict__ G, {BF16}* __restrict__ O, "
    "const float* __restrict__ P)"
)
BINARY_ABO_PARAMS = (
    f"(const {BF16}* __restrict__ A, const {BF16}* __restrict__ Bt, "
    f"{BF16}* __restrict__ O)"
)
SITU_PARAMS = (
    f"(const {BF16}* __restrict__ G, {BF16}* __restrict__ O, "
    f"const {BF16}* __restrict__ U)"
)
CONV_SILU_PARAMS = (
    f"(const {BF16}* __restrict__ Cs, const float* __restrict__ Cw, "
    f"const float* __restrict__ P, {BF16}* __restrict__ Sn, "
    f"{BF16}* __restrict__ X, {BF16}* __restrict__ Y)"
)
KDA_CORE_PARAMS = (
    f"(const float* __restrict__ Alog, const {BF16}* __restrict__ Bt, "
    f"const float* __restrict__ Dt, const {BF16}* __restrict__ G2, "
    "const float* __restrict__ GP, const float* __restrict__ Go, "
    f"const {BF16}* __restrict__ K, {BF16}* __restrict__ Out, "
    f"const {BF16}* __restrict__ Q, const float* __restrict__ State, "
    f"float* __restrict__ StateN, const {BF16}* __restrict__ V)"
)
O_NORM_GATE_PARAMS = (
    f"(const {BF16}* __restrict__ G2, const float* __restrict__ Go, "
    f"{BF16}* __restrict__ Out, const {BF16}* __restrict__ X)"
)
SCORES_PARAMS = (
    f"(const {BF16}* __restrict__ Bl, const {BF16}* __restrict__ Ps, "
    "float* __restrict__ Sc, const float* __restrict__ Sw)"
)
MIX_PARAMS = (
    f"(const {BF16}* __restrict__ Bl, {BF16}* __restrict__ O, "
    f"const {BF16}* __restrict__ Ps, const float* __restrict__ Sc)"
)

ENTRY_SYMBOL = "main_kernel"
KERNEL_MARKER = 'extern "C" __global__ void'

# Beyond 48 KiB of dynamic shared memory a kernel has to opt in per function
# with cudaFuncSetAttribute. Recovered per instantiation, never assumed.
MAX_STATIC_SMEM = 48 * 1024
# Blackwell's per-block shared memory ceiling. A recovered size above this
# would launch-fail at run time, so fail the build instead.
MAX_DYNAMIC_SMEM = 227 * 1024

# TileLang lowers a TMA copy into a warp-specialized kernel that takes
# `CUtensorMap` descriptors instead of pointers and adds a producer warpgroup
# to the block. None of the batched bodies use a bulk copy at all, so the
# launchers bind plain pointers and the requested thread count. That is an
# assumption about TileLang, not about the kernels, so it is asserted rather
# than trusted.
TENSORMAP_BUILDER = "__tvm_tensormap_create_tiled"


# --------------------------------------------------------------------------- #
# Host-stub recovery.
#
# TileLang bakes the launch geometry into the host stub as a packed-call
# argument stack. There is no public accessor for it, hence the parse — and
# hence the strict equality check every caller does against its analytically
# known grid, so a codegen change fails loudly instead of launching a wrong
# grid.
# --------------------------------------------------------------------------- #

_SLOT_INT = re.compile(
    r"\(\(\(TVMFFIAny\*\)stack_ffi_any\)\[(\d+)\]\.v_int64\) = \(\(int64_t\)(-?\d+)\);"
)
_SLOT_PTR = re.compile(
    r"\(\(\(TVMFFIAny\*\)stack_ffi_any\)\[(\d+)\]\.v_ptr\) = (\w+);"
)
_PACKED_CALL = re.compile(
    r"TVMFFIFunctionCall\((\w+?)_packed, \(TVMFFIAny\*\) stack_ffi_any, (\d+),"
)


class HostStub:
    """The launch geometry recovered from one lowered kernel."""

    def __init__(self, kernel, num_params: int, label: str):
        self.label = label
        slots: dict[int, object] = {}
        calls: list[tuple[str, list]] = []
        for line in kernel.get_host_source().splitlines():
            match = _SLOT_INT.search(line)
            if match:
                slots[int(match.group(1))] = int(match.group(2))
                continue
            match = _SLOT_PTR.search(line)
            if match:
                slots[int(match.group(1))] = match.group(2)
                continue
            match = _PACKED_CALL.search(line)
            if match:
                count = int(match.group(2))
                calls.append((match.group(1), [slots.get(i) for i in range(count)]))

        launch = None
        for callee, args in calls:
            if callee == TENSORMAP_BUILDER:
                raise RuntimeError(
                    f"{label}: TileLang lowered this body to TMA. The launcher "
                    "binds plain pointers and the requested thread count, "
                    "neither of which is valid for a warp-specialized kernel; "
                    "the descriptors have to be rebuilt on the host first"
                )
            if callee == ENTRY_SYMBOL:
                launch = args[num_params:]
        if launch is None or len(launch) < 4:
            raise RuntimeError(f"{label}: could not recover the launch geometry")
        if any(not isinstance(value, int) for value in launch):
            raise RuntimeError(f"{label}: launch geometry has non-constant entries")
        self.launch: list[int] = launch

    def check_launch(self, grid: list[int], block_x: int) -> int:
        """Assert the recovered geometry, return the dynamic smem byte count.

        The stub carries exactly as many grid entries as the kernel's
        `T.Kernel` arity (unit dimensions included), then the three block
        dimensions, then the dynamic shared memory size — which is omitted
        entirely when it is zero, so it is read off the tail.
        """
        expected = [*grid, block_x, 1, 1]
        if self.launch[:len(expected)] != expected:
            raise RuntimeError(
                f"{self.label}: launch geometry {self.launch} does not start "
                f"with the expected {expected}"
            )
        tail = self.launch[len(expected):]
        if len(tail) > 1:
            raise RuntimeError(
                f"{self.label}: unexpected trailing launch arguments {tail}"
            )
        smem = tail[0] if tail else 0
        if smem > MAX_DYNAMIC_SMEM:
            raise RuntimeError(
                f"{self.label}: dynamic smem {smem} exceeds the device ceiling"
            )
        return smem


def split_source(source: str) -> tuple[str, str]:
    """Split `get_kernel_source()` into (include preamble, kernel bodies)."""
    marker = source.index(KERNEL_MARKER)
    return source[:marker], source[marker:]


def merge_preambles(preambles: list[str], cu_stem: str) -> str:
    """Reduce a family's preambles to the one that subsumes the others.

    A preamble is include directives only, and TileLang adds to it as a body
    uses more features. Every preamble in a family must therefore be a
    line-wise subsequence of the longest, which is the one emitted.
    """
    longest = max(preambles, key=len)
    reference = longest.splitlines()
    for preamble in preambles:
        remaining = iter(reference)
        if not all(line in remaining for line in preamble.splitlines()):
            raise RuntimeError(
                f"{cu_stem}: TileLang preambles are not nested; "
                "the family cannot share one translation unit"
            )
    return longest


# --------------------------------------------------------------------------- #
# Emission
# --------------------------------------------------------------------------- #

def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


# --------------------------------------------------------------------------- #
# Instantiation plan.
#
# Building a plan is pure arithmetic over the shape lists above; compiling one
# instantiation is a TileLang lowering that takes on the order of a second.
# The two are kept apart so the lowerings can fan out over a process pool at
# instantiation granularity — a family-granular pool is bound by its largest
# family, and the families are very unevenly sized.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Inst:
    """One compiled shape of one family, and how the launcher reaches it."""

    family: str
    order: int
    label: str          # human-readable shape, for error messages
    factory: str        # attribute of `tilelang_defs`
    args: tuple         # its argument tuple
    num_params: int     # prim_func parameter count, to slice the launch args
    params: str         # expected generated C parameter list
    symbol: str         # shape-tagged name the entry point is renamed to
    grid: tuple         # analytic grid, asserted against the host stub
    threads: int        # requested consumer threads
    guard: str          # launcher branch condition
    call_args: tuple    # launcher arguments, in generated parameter order


@dataclass(frozen=True)
class Plan:
    """One generated `.cu`: its launcher and every instantiation in it."""

    stem: str
    signature: str
    doc: str
    insts: tuple


def compile_inst(inst: Inst) -> dict:
    """Lower one instantiation and render its launcher branch."""
    kernel = getattr(defs, inst.factory)(*inst.args)
    stub = HostStub(kernel, inst.num_params, inst.label)
    smem = stub.check_launch(list(inst.grid), inst.threads)

    source = kernel.get_kernel_source()
    if inst.params not in source:
        raise RuntimeError(
            f"kernel signature drifted for {inst.symbol}; update the launcher"
        )
    preamble, body = split_source(source)
    if body.count(ENTRY_SYMBOL) != 2:
        raise RuntimeError(
            f"expected exactly two {ENTRY_SYMBOL} occurrences in {inst.symbol}"
        )

    prologue = []
    if smem > MAX_STATIC_SMEM:
        prologue.append(
            f"    // Past 48 KiB the kernel has to opt in; once per symbol.\n"
            f"    static const cudaError_t opt_in = cudaFuncSetAttribute(\n"
            f"        reinterpret_cast<const void*>({inst.symbol}),\n"
            f"        cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});\n"
            f"    if (opt_in != cudaSuccess) {{\n"
            f"      return static_cast<int>(opt_in);\n"
            f"    }}"
        )

    grid = list(inst.grid) + [1] * (3 - len(inst.grid))
    prologue_text = "\n".join(prologue)
    if prologue_text:
        prologue_text += "\n"
    bound = ",\n        ".join(inst.call_args)
    branch = (
        f"  if ({inst.guard}) {{\n"
        f"{prologue_text}"
        f"    {inst.symbol}<<<dim3({grid[0]}, {grid[1]}, {grid[2]}), "
        f"{inst.threads}, {smem}, stream>>>(\n"
        f"        {bound});\n"
        f"    return static_cast<int>(cudaGetLastError());\n"
        f"  }}"
    )
    return {
        "family": inst.family,
        "order": inst.order,
        "preamble": preamble,
        "body": body.replace(ENTRY_SYMBOL, inst.symbol),
        "branch": branch,
    }


# TileLang's `debug.h` *defines* `debug_print_msg` and the `uint16_t`
# specialization of `debug_print_buffer_value` with external linkage, so every
# translation unit that includes it exports the same two symbols. One
# translation unit per family is fine on its own, but a binary that links
# several of them gets duplicate definitions. The kernels never call either
# helper, so each unit is given privately named copies.
DEBUG_HEADER = "#include <tl_templates/cuda/debug.h>"
_DEBUG_HELPERS = ("debug_print_msg", "debug_print_buffer_value")


def isolate_debug_helpers(preamble: str, cu_stem: str) -> str:
    """Rename `debug.h`'s externally linked helpers per translation unit."""
    if DEBUG_HEADER not in preamble:
        return preamble
    renames = "".join(
        f"#define {name} {cu_stem}_{name}\n" for name in _DEBUG_HELPERS
    )
    restores = "".join(f"#undef {name}\n" for name in _DEBUG_HELPERS)
    return preamble.replace(
        DEBUG_HEADER, f"{renames}{DEBUG_HEADER}\n{restores}".rstrip("\n")
    )


def write_family(plan: Plan, out_dir: Path, results: list[dict]) -> Path:
    """Concatenate one family's lowered instantiations into its `.cu`."""
    results = sorted(results, key=lambda row: row["order"])
    branches = "\n".join(row["branch"] for row in results)
    launcher = (
        f"\n{plan.doc}\n"
        f'extern "C" int {plan.signature} {{\n'
        f"{branches}\n"
        "  return static_cast<int>(cudaErrorInvalidValue);\n"
        "}\n"
    )
    path = out_dir / f"{plan.stem}.cu"
    path.write_text(
        "// Generated by pegainfer-k3/kernels/generate.py -- do not edit.\n"
        "#include <cuda_runtime.h>\n"
        "\n"
        + isolate_debug_helpers(
            merge_preambles([row["preamble"] for row in results], plan.stem),
            plan.stem,
        )
        + "\n".join(row["body"] for row in results)
        + "\n"
        + launcher
    )
    return path


# --------------------------------------------------------------------------- #
# Families
#
# Every launcher takes the batch size and the family's static dimensions and
# dispatches to the matching instantiation, so a configuration that was never
# generated is a `cudaErrorInvalidValue` rather than a wrong-shape launch.
# --------------------------------------------------------------------------- #

_STEM = "k3_{}_batched"


def _bf16(name: str, const: bool = True) -> str:
    return f"reinterpret_cast<{'const ' if const else ''}{BF16}*>({name})"


def plan_rms_norm_rbs() -> Plan:
    insts = []
    for width in RMS_NORM_N:
        for batch in B_CHUNK_BUCKETS:
            insts.append(Inst(
                family="rms_norm_rbs",
                order=len(insts),
                label=f"rms_norm_rbs_batched H={width} B={batch}",
                factory="rms_norm_rbs_batched",
                args=(width, batch, RMS_EPS, THREADS),
                num_params=3,
                params=RMS_NORM_PARAMS,
                symbol=f"k3_rms_norm_rbs_b{batch}_h{width}_kernel",
                grid=(batch,),
                threads=THREADS,
                guard=f"b == {batch} && h == {width}",
                call_args=(_bf16("G"), _bf16("O", False), _bf16("X")),
            ))
    return Plan(
        stem=_STEM.format("rms_norm_rbs"),
        signature=(
            "k3_rms_norm_rbs_batched(\n"
            "    const void* X,\n"
            "    const void* G,\n"
            "    void* O,\n"
            "    int b,\n"
            "    int h,\n"
            "    cudaStream_t stream)"
        ),
        doc=(
            "// KimiRMSNorm, round-before-scale: the normalized value lands in bf16\n"
            "// first and only then multiplies gamma. One row per block; gamma is a\n"
            "// weight shared by every row and eps is compiled in."
        ),
        insts=tuple(insts),
    )


def plan_land() -> Plan:
    insts = []
    for nt, n, off in LAND_CONFIGS:
        for split_k in SPLIT_K:
            for batch in B_CHUNK_BUCKETS:
                npad = ceildiv(n, THREADS) * THREADS
                insts.append(Inst(
                    family="land",
                    order=len(insts),
                    label=f"land_batched NT={nt} N={n} OFF={off} SK={split_k} B={batch}",
                    factory="land_batched",
                    args=(nt, n, off, split_k, batch, THREADS),
                    num_params=2,
                    params=LAND_PARAMS,
                    symbol=f"k3_land_b{batch}_nt{nt}_n{n}_off{off}_sk{split_k}_kernel",
                    grid=(batch, npad // THREADS),
                    threads=THREADS,
                    guard=(
                        f"b == {batch} && nt == {nt} && n == {n} && "
                        f"off == {off} && split_k == {split_k}"
                    ),
                    call_args=(_bf16("O", False), "P"),
                ))
    return Plan(
        stem=_STEM.format("land"),
        signature=(
            "k3_land_batched(\n"
            "    const float* P,\n"
            "    void* O,\n"
            "    int b,\n"
            "    int nt,\n"
            "    int n,\n"
            "    int off,\n"
            "    int split_k,\n"
            "    cudaStream_t stream)"
        ),
        doc=(
            "// Merge the column span [off, off+n) of each row's (split_k, nt) f32\n"
            "// partial and land bf16 once -- the landing of every matmul. split_k = 1\n"
            "// is the single-partial case a framework GEMM produces, where the merge\n"
            "// degenerates to the slice and the cast."
        ),
        insts=tuple(insts),
    )


def plan_land_rms_norm_rbs() -> Plan:
    insts = []
    for nt, n, off in LAND_RMS_NORM_CONFIGS:
        for split_k in SPLIT_K:
            for batch in B_CHUNK_BUCKETS:
                insts.append(Inst(
                    family="land_rms_norm_rbs",
                    order=len(insts),
                    label=(
                        f"land_rms_norm_rbs_batched NT={nt} N={n} OFF={off} "
                        f"SK={split_k} B={batch}"
                    ),
                    factory="land_rms_norm_rbs_batched",
                    args=(nt, n, off, split_k, batch, RMS_EPS, THREADS),
                    num_params=3,
                    params=LAND_RMS_NORM_PARAMS,
                    symbol=(
                        f"k3_land_rms_norm_rbs_b{batch}_nt{nt}_n{n}"
                        f"_off{off}_sk{split_k}_kernel"
                    ),
                    grid=(batch,),
                    threads=THREADS,
                    guard=(
                        f"b == {batch} && nt == {nt} && n == {n} && "
                        f"off == {off} && split_k == {split_k}"
                    ),
                    call_args=(_bf16("G"), _bf16("O", False), "P"),
                ))
    return Plan(
        stem=_STEM.format("land_rms_norm_rbs"),
        signature=(
            "k3_land_rms_norm_rbs_batched(\n"
            "    const float* P,\n"
            "    const void* G,\n"
            "    void* O,\n"
            "    int b,\n"
            "    int nt,\n"
            "    int n,\n"
            "    int off,\n"
            "    int split_k,\n"
            "    cudaStream_t stream)"
        ),
        doc=(
            "// k3_land_batched fused with the round-before-scale norm: MLA's q_a,\n"
            "// the one place the engine fuses a merge and a norm."
        ),
        insts=tuple(insts),
    )


def _plan_binary(family: str, widths: list[int], doc: str) -> Plan:
    insts = []
    for width in widths:
        for batch in B_CHUNK_BUCKETS:
            insts.append(Inst(
                family=family,
                order=len(insts),
                label=f"{family}_batched N={width} B={batch}",
                factory=f"{family}_batched",
                args=(width, batch, THREADS),
                num_params=3,
                params=BINARY_ABO_PARAMS,
                symbol=f"k3_{family}_b{batch}_n{width}_kernel",
                grid=(batch, width // THREADS),
                threads=THREADS,
                guard=f"b == {batch} && n == {width}",
                call_args=(_bf16("A"), _bf16("Bt"), _bf16("O", False)),
            ))
    return Plan(
        stem=_STEM.format(family),
        signature=(
            f"k3_{family}_batched(\n"
            "    const void* A,\n"
            "    const void* Bt,\n"
            "    void* O,\n"
            "    int b,\n"
            "    int n,\n"
            "    cudaStream_t stream)"
        ),
        doc=doc,
        insts=tuple(insts),
    )


def plan_add2() -> Plan:
    return _plan_binary(
        "add2",
        ADD2_N,
        "// O = A + B in bf16 addition (the residual adds, and routed + shared).\n"
        "// One block per (row, column segment).",
    )


def plan_mul_sigmoid() -> Plan:
    return _plan_binary(
        "mul_sigmoid",
        MUL_SIGMOID_N,
        "// O = A * bf16(sigmoid(B)), the MLA sigmoid output gate. The sigmoid is\n"
        "// taken in f32 and lands in bf16 before the product.",
    )


def plan_situ() -> Plan:
    insts = []
    for width in SITU_N:
        for batch in B_CHUNK_BUCKETS:
            insts.append(Inst(
                family="situ",
                order=len(insts),
                label=f"situ_batched N={width} B={batch}",
                factory="situ_batched",
                args=(width, batch, THREADS),
                num_params=3,
                params=SITU_PARAMS,
                symbol=f"k3_situ_b{batch}_n{width}_kernel",
                grid=(batch, width // THREADS),
                threads=THREADS,
                guard=f"b == {batch} && n == {width}",
                call_args=(_bf16("G"), _bf16("O", False), _bf16("U")),
            ))
    return Plan(
        stem=_STEM.format("situ"),
        signature=(
            "k3_situ_batched(\n"
            "    const void* G,\n"
            "    const void* U,\n"
            "    void* O,\n"
            "    int b,\n"
            "    int n,\n"
            "    cudaStream_t stream)"
        ),
        doc=(
            "// 4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25), computed in f32 and landed in\n"
            "// bf16 once; the two betas are compiled in."
        ),
        insts=tuple(insts),
    )


def plan_conv_silu() -> Plan:
    insts = []
    for split_k in SPLIT_K:
        for batch in B_CHUNK_BUCKETS:
            insts.append(Inst(
                family="conv_silu",
                order=len(insts),
                label=f"conv_silu_batched KP={KDA_DIM} W={CONV_WIDTH} "
                      f"SK={split_k} B={batch}",
                factory="conv_silu_batched",
                args=(KDA_DIM, CONV_WIDTH, split_k, batch, THREADS),
                num_params=6,
                params=CONV_SILU_PARAMS,
                symbol=(
                    f"k3_conv_silu_b{batch}_kp{KDA_DIM}"
                    f"_w{CONV_WIDTH}_sk{split_k}_kernel"
                ),
                grid=(batch, KDA_DIM // THREADS),
                threads=THREADS,
                guard=(
                    f"b == {batch} && kp == {KDA_DIM} && "
                    f"width == {CONV_WIDTH} && split_k == {split_k}"
                ),
                call_args=(
                    _bf16("Cs"), "Cw", "P",
                    _bf16("Sn", False), _bf16("X", False), _bf16("Y", False),
                ),
            ))
    return Plan(
        stem=_STEM.format("conv_silu"),
        signature=(
            "k3_conv_silu_batched(\n"
            "    const float* P,\n"
            "    const float* Cw,\n"
            "    const void* Cs,\n"
            "    void* X,\n"
            "    void* Y,\n"
            "    void* Sn,\n"
            "    int b,\n"
            "    int kp,\n"
            "    int width,\n"
            "    int split_k,\n"
            "    cudaStream_t stream)"
        ),
        doc=(
            "// Causal depthwise convolution over the `width`-slot window plus silu,\n"
            "// one token per row. The projection partial's bf16 landing is written to\n"
            "// X and is also the newest window slot; Sn is the shifted window the\n"
            "// caller carries. Conv weights Cw are f32 and have no batch axis; the\n"
            "// window state is [b, width-1, kp], one independent window per row."
        ),
        insts=tuple(insts),
    )


def plan_kda_core() -> Plan:
    insts = []
    # The gate partial is produced by a single-segment projection, so the
    # engine compiles kda_core with split-K 1 on the gate input.
    split_k_gate = 1
    for batch in B_BUCKETS:
        insts.append(Inst(
            family="kda_core",
            order=len(insts),
            label=f"kda_core_batched KH={KDA_HEADS} KD={KDA_HEAD_DIM} B={batch}",
            factory="kda_core_batched",
            args=(KDA_HEADS, KDA_HEAD_DIM, split_k_gate, batch,
                  GATE_LOWER_BOUND, RMS_EPS),
            num_params=12,
            params=KDA_CORE_PARAMS,
            symbol=f"k3_kda_core_b{batch}_kh{KDA_HEADS}_kd{KDA_HEAD_DIM}_kernel",
            grid=(batch, KDA_HEADS),
            threads=KDA_HEAD_DIM,
            guard=(
                f"b == {batch} && num_heads == {KDA_HEADS} && "
                f"head_dim == {KDA_HEAD_DIM} && split_k_gate == {split_k_gate}"
            ),
            call_args=(
                "Alog", _bf16("Bt"), "Dt", _bf16("G2"), "GP", "Go",
                _bf16("K"), _bf16("Out", False), _bf16("Q"),
                "State", "StateN", _bf16("V"),
            ),
        ))
    return Plan(
        stem=_STEM.format("kda_core"),
        signature=(
            "k3_kda_core_batched(\n"
            "    const void* Q,\n"
            "    const void* K,\n"
            "    const void* V,\n"
            "    const float* GP,\n"
            "    const float* Dt,\n"
            "    const float* Alog,\n"
            "    const void* Bt,\n"
            "    const void* G2,\n"
            "    const float* Go,\n"
            "    const float* State,\n"
            "    float* StateN,\n"
            "    void* Out,\n"
            "    int b,\n"
            "    int num_heads,\n"
            "    int head_dim,\n"
            "    int split_k_gate,\n"
            "    cudaStream_t stream)"
        ),
        doc=(
            "// One delta-rule step per row, one (row, head) per block, with\n"
            "// threads = head_dim. The recurrent state is\n"
            "// [b, head, v_dim, k_dim] f32 with decay along k, so a row's state is\n"
            "// the contiguous single-row block; State and StateN must not alias.\n"
            "// Dt, Alog and Go are weights and carry no batch axis; the gate lower\n"
            "// bound and eps are compiled in."
        ),
        insts=tuple(insts),
    )


def plan_o_norm_gate() -> Plan:
    insts = []
    for batch in B_CHUNK_BUCKETS:
        insts.append(Inst(
            family="o_norm_gate",
            order=len(insts),
            label=f"o_norm_gate_batched KH={KDA_HEADS} KD={KDA_HEAD_DIM} B={batch}",
            factory="o_norm_gate_batched",
            args=(KDA_HEADS, KDA_HEAD_DIM, batch, RMS_EPS),
            num_params=4,
            params=O_NORM_GATE_PARAMS,
            symbol=f"k3_o_norm_gate_b{batch}_kh{KDA_HEADS}_kd{KDA_HEAD_DIM}_kernel",
            grid=(batch, KDA_HEADS),
            threads=KDA_HEAD_DIM,
            guard=(
                f"b == {batch} && num_heads == {KDA_HEADS} && "
                f"head_dim == {KDA_HEAD_DIM}"
            ),
            call_args=(
                _bf16("G2"), "Go", _bf16("Out", False), _bf16("X"),
            ),
        ))
    return Plan(
        stem=_STEM.format("o_norm_gate"),
        signature=(
            "k3_o_norm_gate_batched(\n"
            "    const void* X,\n"
            "    const void* G2,\n"
            "    const float* Go,\n"
            "    void* Out,\n"
            "    int b,\n"
            "    int num_heads,\n"
            "    int head_dim,\n"
            "    cudaStream_t stream)"
        ),
        doc=(
            "// kda_core's tail on its own: per (row, head) the f32 rms_norm of the\n"
            "// bf16 attention landing times the o_norm gamma, landed once, times the\n"
            "// bf16 sigmoid of the output gate. Chunked prefill computes attention\n"
            "// through FlashKDA and finishes rows here; eps is compiled in."
        ),
        insts=tuple(insts),
    )


def plan_attnres_scores() -> Plan:
    insts = []
    for blocks in ATTNRES_NB:
        for batch in B_CHUNK_BUCKETS:
            insts.append(Inst(
                family="attnres_scores",
                order=len(insts),
                label=f"attnres_scores_batched NB={blocks} B={batch}",
                factory="attnres_scores_batched",
                args=(blocks, ATTNRES_BLOCKS, HIDDEN, batch, RMS_EPS, THREADS),
                num_params=4,
                params=SCORES_PARAMS,
                symbol=f"k3_attnres_scores_b{batch}_nb{blocks}_h{HIDDEN}_kernel",
                grid=(batch, blocks + 1),
                threads=THREADS,
                guard=f"b == {batch} && blocks == {blocks} && h == {HIDDEN}",
                call_args=(_bf16("Bl"), _bf16("Ps"), "Sc", "Sw"),
            ))
    return Plan(
        stem=_STEM.format("attnres_scores"),
        signature=(
            "k3_attnres_scores_batched(\n"
            "    const void* Ps,\n"
            "    const void* Bl,\n"
            "    const float* Sw,\n"
            "    float* Sc,\n"
            "    int b,\n"
            "    int blocks,\n"
            "    int h,\n"
            "    cudaStream_t stream)"
        ),
        doc=(
            "// Attention-residual candidate scoring, one (row, candidate) per block:\n"
            "// weightless RMS normalization then a dot with the pre-fused f32 scoring\n"
            "// vector. Candidate `blocks` is the running prefix sum Ps itself, the\n"
            "// ones below it are that row's snapshot history Bl."
        ),
        insts=tuple(insts),
    )


def plan_attnres_mix() -> Plan:
    insts = []
    for blocks in ATTNRES_NB:
        for batch in B_CHUNK_BUCKETS:
            insts.append(Inst(
                family="attnres_mix",
                order=len(insts),
                label=f"attnres_mix_batched NB={blocks} B={batch}",
                factory="attnres_mix_batched",
                args=(blocks, ATTNRES_BLOCKS, HIDDEN, batch, THREADS),
                num_params=4,
                params=MIX_PARAMS,
                symbol=f"k3_attnres_mix_b{batch}_nb{blocks}_h{HIDDEN}_kernel",
                grid=(batch, HIDDEN // THREADS),
                threads=THREADS,
                guard=f"b == {batch} && blocks == {blocks} && h == {HIDDEN}",
                call_args=(_bf16("Bl"), _bf16("O", False), _bf16("Ps"), "Sc"),
            ))
    return Plan(
        stem=_STEM.format("attnres_mix"),
        signature=(
            "k3_attnres_mix_batched(\n"
            "    const void* Ps,\n"
            "    const void* Bl,\n"
            "    const float* Sc,\n"
            "    void* O,\n"
            "    int b,\n"
            "    int blocks,\n"
            "    int h,\n"
            "    cudaStream_t stream)"
        ),
        doc=(
            "// Softmax over each row's blocks+1 scores, then a probability-weighted\n"
            "// mix of the un-normalized candidates landing bf16 once. One block per\n"
            "// (row, column segment); each block redoes its row's softmax."
        ),
        insts=tuple(insts),
    )


PLANNERS = [
    plan_rms_norm_rbs,
    plan_land,
    plan_land_rms_norm_rbs,
    plan_add2,
    plan_mul_sigmoid,
    plan_situ,
    plan_conv_silu,
    plan_kda_core,
    plan_o_norm_gate,
    plan_attnres_scores,
    plan_attnres_mix,
]

# A planner that exists but never makes the list would silently drop its
# launcher and only surface as a link error in the k3 feature build.
_ORPHANS = sorted(
    name
    for name, value in list(globals().items())
    if name.startswith("plan_") and callable(value) and value not in PLANNERS
)
if _ORPHANS:
    raise RuntimeError(f"family planners missing from PLANNERS: {_ORPHANS}")


def vendor_includes(out_dir: Path) -> tuple[Path, Path]:
    """Copy the TileLang/CUTLASS header trees next to the generated CUDA.

    Only used for the pre-generated tier: the build host that compiles the
    artifact may have no TileLang install to take the include paths from.
    """
    include_dir = out_dir / "include"
    template_dst = include_dir / "tilelang"
    cutlass_dst = include_dir / "cutlass"
    for src, dst in (
        (TILELANG_TEMPLATE_PATH, template_dst),
        (CUTLASS_INCLUDE_DIR, cutlass_dst),
    ):
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    return template_dst, cutlass_dst


def _pin_arch(arch: str) -> None:
    """Pin the TileLang target without touching a character of certified code.

    The vendored `_compile` targets the local GPU, which a build host may not
    have. The kernel bodies resolve `_compile` through the module namespace at
    call time, so rebinding it here is enough.
    """
    target = {"kind": "cuda", "arch": arch}
    defs._compile = lru_cache(maxsize=None)(
        lambda prim: tilelang.compile(prim, target=target)
    )


def _init_worker(arch) -> None:
    """Pin the target once per worker, not once per instantiation.

    `fork` already carries the parent's binding, but re-pinning here keeps the
    pool correct if the start method ever changes -- and doing it in the
    initializer rather than in the task keeps `_compile`'s cache alive across
    the many instantiations one worker handles.
    """
    if arch:
        _pin_arch(arch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--arch",
        default=None,
        help=(
            "TileLang CUDA arch (e.g. sm_103a). Default: let TileLang pick the "
            "local GPU, which is what the upstream certification ran. Required "
            "on build hosts with no visible GPU."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("PEGAINFER_K3_TILELANG_JOBS", "0")),
        help="instantiations to lower concurrently (default: PEGAINFER_K3_TILELANG_JOBS, "
        "else one per CPU capped at 32; 1 disables the pool). Each worker holds a "
        "TileLang lowering, so lower it on memory-tight hosts",
    )
    parser.add_argument(
        "--vendor-includes",
        action="store_true",
        help="copy the TileLang/CUTLASS headers into <out-dir>/include and "
        "point the manifest at the copies (self-contained pre-generated dir)",
    )
    args = parser.parse_args()

    if args.arch:
        _pin_arch(args.arch)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plans = [planner() for planner in PLANNERS]
    work = [inst for plan in plans for inst in plan.insts]
    # Longest-first: the tail of a pool run is whatever job started last, so
    # the big shapes have to go in first or they set the wall clock.
    work.sort(key=lambda inst: -sum(inst.grid))

    jobs = args.jobs or min(os.cpu_count() or 1, 32)
    started = time.monotonic()
    if jobs > 1:
        # Instantiations are independent lowerings and TileLang lowering is
        # CPU-bound, so they fan out. `fork` keeps the pinned `_compile`
        # binding without re-importing TileLang per worker.
        context = multiprocessing.get_context("fork")
        with context.Pool(jobs, initializer=_init_worker,
                          initargs=(args.arch,)) as pool:
            results = pool.map(compile_inst, work, chunksize=1)
    else:
        results = [compile_inst(inst) for inst in work]
    lowered = time.monotonic() - started

    by_family: dict[str, list[dict]] = {}
    for row in results:
        by_family.setdefault(row["family"], []).append(row)

    cu_paths = []
    for plan in plans:
        family = plan.insts[0].family
        rows = by_family[family]
        if len(rows) != len(plan.insts):
            raise RuntimeError(f"{plan.stem}: lost instantiations in the pool")
        cu_paths.append(write_family(plan, out_dir, rows))
    elapsed = time.monotonic() - started

    for plan in plans:
        print(f"# {plan.stem}: {len(plan.insts)} instantiations")
    print(
        f"# total: {len(work)} instantiations, {lowered:.1f}s lowering "
        f"+ {elapsed - lowered:.1f}s writing over {jobs} job(s)"
    )

    if args.vendor_includes:
        template_include, cutlass_include = vendor_includes(out_dir)
    else:
        template_include = Path(TILELANG_TEMPLATE_PATH)
        cutlass_include = Path(CUTLASS_INCLUDE_DIR)

    lines = [f"CU_PATH={path}" for path in cu_paths]
    lines.append(f"TILELANG_TEMPLATE_PATH={template_include}")
    lines.append(f"CUTLASS_INCLUDE_DIR={cutlass_include}")
    if args.arch:
        # The bodies are lowered for exactly this arch and may use
        # arch-conditional instructions, so the consumer has to assemble them
        # for it and not for the generic SM list.
        lines.append(f"ARCH={args.arch}")
    # The manifest lets a build host consume a pre-generated directory without
    # re-running (or even having) TileLang; build.rs parses the same key=value
    # lines from either stdout or this file.
    (out_dir / "manifest.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
