#!/usr/bin/env python3
"""Generate the K3 batched TileLang decode kernels (AOT).

Emits one `.cu` per kernel family under `--out-dir`:

    k3_router_topk.cu        router_topk_batched, E x B instantiations
    k3_attnres_scores.cu     attnres_scores_batched, NB x B instantiations
    k3_attnres_mix.cu        attnres_mix_batched, NB x B instantiations

Each file is [shared TileLang preamble] + [one renamed `main_kernel` per
instantiation] + [one hand-written `extern "C"` dispatch launcher]. TileLang
always names the entry point `main_kernel`, so every instantiation is renamed
to a shape-tagged symbol before concatenation; the launcher is the only
non-generated code in the artifact and returns `cudaErrorInvalidValue` for a
shape that was not instantiated.

The kernel definitions themselves are the vendored, certified spellings in
`tilelang_defs.py` — this file only walks the shape buckets, checks the
codegen contract, and wraps. It never rewrites kernel bodies.

Output is a Cargo `OUT_DIR` build artifact and is never checked in. Run
standalone with `--vendor-includes` to produce a self-contained pre-generated
directory for build hosts that cannot run TileLang (see README.md).
"""

import argparse
import re
import shutil
from functools import lru_cache
from pathlib import Path

import tilelang
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

import tilelang_defs as defs

# TileLang releases change codegen in ways this generator depends on: the
# entry-point symbol name, the (alphabetically sorted) kernel parameter list
# asserted per family below, and the host-side launch-parameter encoding the
# dynamic-smem size is recovered from. Bumping requires re-checking all three
# AND re-running the upstream bitwise parity gate — do not just widen this.
KNOWN_GOOD_TILELANG = "0.1.12"
if tilelang.__version__ != KNOWN_GOOD_TILELANG:
    raise RuntimeError(
        f"tilelang {tilelang.__version__} is not the validated "
        f"{KNOWN_GOOD_TILELANG}; point PEGAINFER_K3_TILELANG_PYTHON at an env "
        "with the pinned version, or revalidate the codegen contract and bump "
        "KNOWN_GOOD_TILELANG"
    )

tilelang.set_log_level("WARNING")

THREADS = 256

# Serving batch buckets. Every kernel is row-independent, so a step with B
# real rows runs the next bucket up and ignores the tail rows; the ops layer
# owns the rounding and the caller owns bucket-sized buffers.
B_BUCKETS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]

# Routed-expert counts in the K3 layer stack, and the fixed route width.
ROUTER_EXPERTS = [224, 896]
ROUTER_TOPK = 16

# Attention-residual candidate block count. The block history grows 1..8
# across the 93 layers, so all eight widths are live.
ATTNRES_NB = [1, 2, 3, 4, 5, 6, 7, 8]
ATTNRES_H = 7168
ATTNRES_EPS = 1e-5

# TileLang emits the kernel parameters sorted by name, not in prim_func order;
# the launchers below bind arguments to these exact signatures.
ROUTER_PARAMS = (
    "(const float* __restrict__ Bias, int* __restrict__ Idx, "
    "const bfloat16_t* __restrict__ Rs, const float* __restrict__ S, "
    "float* __restrict__ Wts)"
)
SCORES_PARAMS = (
    "(const bfloat16_t* __restrict__ Bl, const bfloat16_t* __restrict__ Ps, "
    "float* __restrict__ Sc, const float* __restrict__ Sw)"
)
MIX_PARAMS = (
    "(const bfloat16_t* __restrict__ Bl, bfloat16_t* __restrict__ O, "
    "const bfloat16_t* __restrict__ Ps, const float* __restrict__ Sc)"
)

ENTRY_SYMBOL = "main_kernel"
KERNEL_MARKER = 'extern "C" __global__ void'

# Every instantiation stays far under the 48 KiB static limit, so no launcher
# needs cudaFuncSetAttribute. Asserted per instantiation rather than assumed.
MAX_STATIC_SMEM = 48 * 1024

_STACK_INT = re.compile(
    r"\(\(\(TVMFFIAny\*\)stack_ffi_any\)\[(\d+)\]\.v_int64\) = \(\(int64_t\)(\d+)\);"
)


def launch_config(kernel, num_params: int) -> list[int]:
    """Recover `[grid..., block.x, block.y, block.z, dyn_smem_bytes]`.

    TileLang bakes the launch geometry into the host stub as the trailing
    integer arguments of the packed device-function call: stack slots
    `[0, num_params)` carry the tensor pointers, everything after them is the
    launch parameter list, ending with the dynamic shared memory size. There
    is no public accessor for it, hence the parse — and hence the strict
    equality check every caller does against its analytically known geometry,
    so a codegen change fails loudly instead of launching a wrong grid.
    """
    source = kernel.get_host_source()
    slots = {int(idx): int(val) for idx, val in _STACK_INT.findall(source)}
    launch = [slots[i] for i in sorted(slots) if i >= num_params]
    if len(launch) < 4:
        raise RuntimeError(f"could not recover launch config; parsed {launch}")
    return launch


def split_source(source: str) -> tuple[str, str]:
    """Split `get_kernel_source()` into (include preamble, kernel bodies)."""
    marker = source.index(KERNEL_MARKER)
    return source[:marker], source[marker:]


class Family:
    """One generated `.cu`: shared preamble, renamed bodies, one launcher."""

    def __init__(self, cu_stem: str):
        self.cu_stem = cu_stem
        self.preamble: str | None = None
        self.bodies: list[str] = []
        self.dispatch: list[str] = []

    def add(self, kernel, symbol: str, expected_params: str) -> None:
        source = kernel.get_kernel_source()
        if expected_params not in source:
            raise RuntimeError(
                f"kernel signature drifted for {symbol}; update the launcher"
            )
        preamble, body = split_source(source)
        if self.preamble is None:
            self.preamble = preamble
        elif self.preamble != preamble:
            raise RuntimeError(f"TileLang preamble differs across {self.cu_stem}")
        if body.count(ENTRY_SYMBOL) != 2:
            raise RuntimeError(
                f"expected exactly two {ENTRY_SYMBOL} occurrences in {symbol}"
            )
        self.bodies.append(body.replace(ENTRY_SYMBOL, symbol))

    def write(self, out_dir: Path, launcher: str) -> Path:
        path = out_dir / f"{self.cu_stem}.cu"
        path.write_text(
            "// Generated by pegainfer-k3/kernels/generate.py -- do not edit.\n"
            "#include <cuda_runtime.h>\n"
            "\n"
            f"{self.preamble}"
            + "\n".join(self.bodies)
            + "\n"
            + launcher
        )
        return path


def check_launch(launch: list[int], grid: list[int], label: str) -> int:
    """Assert the recovered geometry, return the dynamic smem byte count."""
    expected = [*grid, THREADS, 1, 1]
    if launch[:-1] != expected:
        raise RuntimeError(
            f"{label}: launch geometry {launch[:-1]} != expected {expected}"
        )
    smem = launch[-1]
    if not 0 < smem <= MAX_STATIC_SMEM:
        raise RuntimeError(f"{label}: dynamic smem {smem} outside the launcher's bound")
    return smem


def build_router(out_dir: Path) -> Path:
    family = Family("k3_router_topk")
    for experts in ROUTER_EXPERTS:
        smem_seen = None
        for batch in B_BUCKETS:
            kernel = defs.router_topk_batched(experts, ROUTER_TOPK, batch, THREADS)
            symbol = f"k3_router_topk_e{experts}_k{ROUTER_TOPK}_b{batch}_kernel"
            label = f"router e={experts} b={batch}"
            smem = check_launch(launch_config(kernel, 5), [batch], label)
            # smem holds two f32 score rows and never depends on the batch.
            if smem_seen is None:
                smem_seen = smem
            elif smem_seen != smem:
                raise RuntimeError(f"{label}: smem {smem} != {smem_seen} at other B")
            family.add(kernel, symbol, ROUTER_PARAMS)
            family.dispatch.append(
                f"""  if (num_experts == {experts} && batch == {batch}) {{
    {symbol}<<<dim3({batch}, 1, 1), {THREADS}, {smem}, stream>>>(
        bias,
        idx,
        reinterpret_cast<const bfloat16_t*>(routed_scale),
        scores,
        weights);
    return static_cast<int>(cudaGetLastError());
  }}"""
            )

    launcher = f"""
// Sigmoid router + biased top-k over one bucket of rows. `batch` is the
// bucket, not the live row count: the caller rounds up and owns
// bucket-sized buffers, and the tail rows are computed and discarded.
extern "C" int k3_router_topk(
    const float* scores,
    const float* bias,
    const void* routed_scale,
    int* idx,
    float* weights,
    int num_experts,
    int topk,
    int batch,
    cudaStream_t stream) {{
  if (topk != {ROUTER_TOPK}) {{
    return static_cast<int>(cudaErrorInvalidValue);
  }}
{chr(10).join(family.dispatch)}
  return static_cast<int>(cudaErrorInvalidValue);
}}
"""
    return family.write(out_dir, launcher)


def build_attnres_scores(out_dir: Path) -> Path:
    family = Family("k3_attnres_scores")
    for blocks in ATTNRES_NB:
        for batch in B_BUCKETS:
            kernel = defs.attnres_scores_batched(
                blocks, ATTNRES_H, batch, ATTNRES_EPS, THREADS
            )
            symbol = f"k3_attnres_scores_nb{blocks}_b{batch}_kernel"
            label = f"attnres_scores nb={blocks} b={batch}"
            smem = check_launch(
                launch_config(kernel, 4), [batch, blocks + 1], label
            )
            family.add(kernel, symbol, SCORES_PARAMS)
            family.dispatch.append(
                f"""  if (num_blocks == {blocks} && batch == {batch}) {{
    {symbol}<<<dim3({batch}, {blocks + 1}, 1), {THREADS}, {smem}, stream>>>(
        reinterpret_cast<const bfloat16_t*>(blocks_in),
        reinterpret_cast<const bfloat16_t*>(prefix_sum),
        scores_out,
        score_weight);
    return static_cast<int>(cudaGetLastError());
  }}"""
            )

    launcher = f"""
// Attention-residual candidate scoring: one block per (row, candidate),
// weightless RMS normalization then a dot with the fused f32 scoring vector.
// Candidate `num_blocks` is the running prefix sum, candidates below it are
// the per-row block snapshots. `batch` is the padded bucket.
extern "C" int k3_attnres_scores(
    const void* prefix_sum,
    const void* blocks_in,
    const float* score_weight,
    float* scores_out,
    int num_blocks,
    int hidden,
    int batch,
    cudaStream_t stream) {{
  if (hidden != {ATTNRES_H}) {{
    return static_cast<int>(cudaErrorInvalidValue);
  }}
{chr(10).join(family.dispatch)}
  return static_cast<int>(cudaErrorInvalidValue);
}}
"""
    return family.write(out_dir, launcher)


def build_attnres_mix(out_dir: Path) -> Path:
    family = Family("k3_attnres_mix")
    for blocks in ATTNRES_NB:
        for batch in B_BUCKETS:
            kernel = defs.attnres_mix_batched(blocks, ATTNRES_H, batch, THREADS)
            symbol = f"k3_attnres_mix_nb{blocks}_b{batch}_kernel"
            label = f"attnres_mix nb={blocks} b={batch}"
            smem = check_launch(
                launch_config(kernel, 4), [batch, ATTNRES_H // THREADS], label
            )
            family.add(kernel, symbol, MIX_PARAMS)
            family.dispatch.append(
                f"""  if (num_blocks == {blocks} && batch == {batch}) {{
    {symbol}<<<dim3({batch}, {ATTNRES_H // THREADS}, 1), {THREADS}, {smem}, stream>>>(
        reinterpret_cast<const bfloat16_t*>(blocks_in),
        reinterpret_cast<bfloat16_t*>(out),
        reinterpret_cast<const bfloat16_t*>(prefix_sum),
        scores);
    return static_cast<int>(cudaGetLastError());
  }}"""
            )

    launcher = f"""
// Attention-residual mixing: softmax over each row's `num_blocks + 1` scores,
// then a probability-weighted mix of the UN-normalized candidates landing in
// bf16 once. `batch` is the padded bucket.
extern "C" int k3_attnres_mix(
    const void* prefix_sum,
    const void* blocks_in,
    const float* scores,
    void* out,
    int num_blocks,
    int hidden,
    int batch,
    cudaStream_t stream) {{
  if (hidden != {ATTNRES_H}) {{
    return static_cast<int>(cudaErrorInvalidValue);
  }}
{chr(10).join(family.dispatch)}
  return static_cast<int>(cudaErrorInvalidValue);
}}
"""
    return family.write(out_dir, launcher)


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
        "--vendor-includes",
        action="store_true",
        help="copy the TileLang/CUTLASS headers into <out-dir>/include and "
        "point the manifest at the copies (self-contained pre-generated dir)",
    )
    args = parser.parse_args()

    if args.arch:
        # The vendored `_compile` targets the local GPU, which a build host may
        # not have. Rebinding it here pins the arch without touching a single
        # character of the certified kernel bodies: they resolve `_compile`
        # through the module namespace at call time.
        target = {"kind": "cuda", "arch": args.arch}
        defs._compile = lru_cache(maxsize=None)(
            lambda prim: tilelang.compile(prim, target=target)
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cu_paths = [
        build_router(out_dir),
        build_attnres_scores(out_dir),
        build_attnres_mix(out_dir),
    ]

    if args.vendor_includes:
        template_include, cutlass_include = vendor_includes(out_dir)
    else:
        template_include = Path(TILELANG_TEMPLATE_PATH)
        cutlass_include = Path(CUTLASS_INCLUDE_DIR)

    lines = [f"CU_PATH={path}" for path in cu_paths]
    lines.append(f"TILELANG_TEMPLATE_PATH={template_include}")
    lines.append(f"CUTLASS_INCLUDE_DIR={cutlass_include}")
    # The manifest lets a build host consume a pre-generated directory without
    # re-running (or even having) TileLang; build.rs parses the same key=value
    # lines from either stdout or this file.
    (out_dir / "manifest.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
