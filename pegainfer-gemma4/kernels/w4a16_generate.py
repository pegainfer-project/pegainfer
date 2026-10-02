"""AOT-compile the Gemma 4 W4A16 decode GEMMs into one CUDA file.

`pegainfer-kernels/build.rs` runs this under the `gemma4` feature and prints
the family's `KEY=VALUE` manifest, as `generate.py` does for the attention
kernels, whose lowering, launcher rendering and manifest conventions this
reuses. One kernel per (31B linear shape, row bucket) sits behind a single
launcher that dispatches on (n, k, rows) and refuses any other.

The CTA count is part of the kernels (stream-K splits over it, and the split
decides the summation order), so it is fixed here: twice the build device's
SM count, or twice `PEGAINFER_GEMMA4_W4A16_SMS`. `GEOMETRY` carries it to the
crate, which refuses a device with another SM count. The warp count and
pipeline depth are not fixed either: `pick_tiling` widens them until the two
CTAs the fix-up needs stop fitting the arch's shared memory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import tilelang
import w4a16_defs as defs
from generate import (
    DEBUG_HEADER,
    DEBUG_HELPERS,
    Kernel,
    Launcher,
    PASS_CONFIG_NVCC_FLAG,
    lower,
    render_launcher,
    vendor_includes,
)
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

CU_STEM = "gemma4_w4a16"
LAUNCHER = "gemma4_w4a16_gemm"
OCCUPANCY = "gemma4_w4a16_occupancy"
# No fast math: gate|up's epilogue computes the MLP activation, which has to
# round as `gelu_tanh_mul_kernel` (compiled without it) does. The GEMMs'
# bits and time are the same either way.
PASS_CONFIGS = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}

# (n, k) of every text linear in the 31B tower: Q|K|V for the sliding and
# global families, o_proj for each, gate|up and down.
HIDDEN = 5376
SHAPES = [
    (16384, HIDDEN),
    (18432, HIDDEN),
    (HIDDEN, 8192),
    (HIDDEN, 16384),
    (43008, HIDDEN),
    (HIDDEN, 21504),
]
# gate|up, whose GEMMs write gelu(gate) * up.
GELU_MUL = (43008, HIDDEN)

LAUNCHER_PARAMS = [
    ("void*", "x"),
    ("int*", "wq"),
    ("int*", "sq"),
    ("void*", "y"),
    ("float*", "part"),
    ("int*", "flags"),
    ("int*", "fin_off"),
    ("int*", "fin_list"),
    ("int", "n"),
    ("int", "k"),
    ("int", "rows"),
]
OCCUPANCY_PARAMS = [("int", "n"), ("int", "k"), ("int", "rows"), ("int*", "blocks")]


def isolate_debug_helpers(preamble: str) -> str:
    """`debug.h`'s externally linked helpers, renamed for this unit so they do
    not collide with the attention unit's."""
    if DEBUG_HEADER not in preamble:
        return preamble
    renames = "".join(f"#define {name} {CU_STEM}_{name}\n" for name in DEBUG_HELPERS)
    restores = "".join(f"#undef {name}\n" for name in DEBUG_HELPERS)
    return preamble.replace(
        DEBUG_HEADER, f"{renames}{DEBUG_HEADER}\n{restores}".rstrip("\n")
    )


def sm_count() -> int | None:
    """The serving device's SM count, `None` when neither the variable nor a
    visible device says it."""
    stated = os.environ.get("PEGAINFER_GEMMA4_W4A16_SMS")
    if stated:
        return int(stated)
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).multi_processor_count


def sm_of(arch: str) -> int:
    """The compute capability an `--arch` names: `sm_89` and `sm_90a` are 89
    and 90."""
    digits = ""
    for ch in arch.removeprefix("sm_"):
        if not ch.isdigit():
            break
        digits += ch
    return int(digits)


def probe_smem(arch: str, ctas: int, tiling: tuple[int, int]) -> int:
    """Dynamic shared bytes of a rows=16 kernel, the widest bucket, so the one
    the two-resident-CTA budget has to clear. One shape answers for all six:
    the tiles are sized by the bucket and the tiling, not by N or K. `main`
    holds the whole lowering to the same budget anyway."""
    spec, *_ = kernels(ctas, tiling, rows_only=16)
    return lower(spec, arch).smem


def smem_budget(arch: str) -> int:
    """The shared memory per SM of the device `--arch` names. An arch with no
    stated budget takes the smallest one, which keeps the tiling loadable."""
    return defs.SMEM_PER_SM.get(sm_of(arch), min(defs.SMEM_PER_SM.values()))


def pick_tiling(arch: str, ctas: int, budget: int) -> tuple[int, int]:
    """The widest tiling whose two resident CTAs fit `budget` bytes of shared
    memory per SM."""
    for tiling in defs.TILINGS:
        if 2 * (probe_smem(arch, ctas, tiling) + defs.SMEM_PER_BLOCK_RESERVED) <= budget:
            return tiling
    raise RuntimeError(f"no W4A16 tiling fits {budget} B of shared memory per SM")


def symbol(n: int, k: int, rows: int) -> str:
    return f"{CU_STEM}_{n}x{k}_m{rows}_kernel"


CASES = [(n, k, rows) for n, k in SHAPES for rows in defs.BUCKETS]


def kernels(
    ctas: int, tiling: tuple[int, int], rows_only: int | None = None
) -> list[Kernel]:
    out = []
    for n, k, rows in CASES:
        if rows_only is not None and rows != rows_only:
            continue

        def build(arch: str, n=n, k=k, rows=rows):
            return tilelang.compile(
                defs.gemm(n, k, rows, ctas, gelu_mul=(n, k) == GELU_MUL, tiling=tiling),
                target={"kind": "cuda", "arch": arch},
                pass_configs=PASS_CONFIGS,
            )

        out.append(
            Kernel(
                symbol=symbol(n, k, rows),
                build=build,
                descriptor_var={},
                tensor_arg={},
                runtime_rows={},
                bind={
                    "x": "reinterpret_cast<bfloat16_t*>(x)",
                    "wq": "wq",
                    "sq": "sq",
                    "y": "reinterpret_cast<bfloat16_t*>(y)",
                    "part": "part",
                    "flags": "flags",
                    "fin_off": "fin_off",
                    "fin_list": "fin_list",
                },
                grid=f"dim3({ctas})",
                opt_in_var=f"opt_in_{n}_{k}_{rows}",
                when=f"n == {n} && k == {k} && rows == {rows}",
            )
        )
    return out


def known_condition() -> str:
    return " ||\n      ".join(
        f"(n == {n} && k == {k} && rows == {r})" for n, k, r in CASES
    )


def render_occupancy(lowered: list) -> str:
    """Blocks of the (n, k, rows) kernel one SM holds at once. The in-kernel
    fix-up waits on other CTAs, so the crate refuses a device where fewer
    than every CTA can be resident."""
    cases = "".join(
        f"  if ({low.kernel.when}) {{\n"
        f"    cudaFuncSetAttribute(reinterpret_cast<const void*>({low.kernel.symbol}),\n"
        f"        cudaFuncAttributeMaxDynamicSharedMemorySize, {low.smem});\n"
        f"    return static_cast<int>(cudaOccupancyMaxActiveBlocksPerMultiprocessor(\n"
        f"        blocks, {low.kernel.symbol}, {low.block}, {low.smem}));\n"
        f"  }}\n"
        for low in lowered
    )
    signature = ", ".join(f"{kind} {name}" for kind, name in OCCUPANCY_PARAMS)
    return (
        f'extern "C" int {OCCUPANCY}(\n    {signature},\n    cudaStream_t) {{\n'
        f"{cases}"
        f"  return static_cast<int>(cudaErrorNotSupported);\n}}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--vendor-includes", action="store_true")
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sms = sm_count()
    if sms is None:
        # The CTA count is part of the kernels, so without it there is
        # nothing to lower; build.rs links the stub tier and a W4A16
        # checkpoint is refused at load.
        print(
            "UNAVAILABLE=the kernels are compiled for one CTA count: set "
            "PEGAINFER_GEMMA4_W4A16_SMS to the serving device's SM count, or build where "
            "the device is visible"
        )
        return
    ctas = 2 * sms
    budget = smem_budget(args.arch)
    tiling = pick_tiling(args.arch, ctas, budget)
    specs = kernels(ctas, tiling)
    launcher = Launcher(
        name=LAUNCHER,
        params=LAUNCHER_PARAMS,
        bounds=f"  if (!({known_condition()})) {{\n    return static_cast<int>(cudaErrorNotSupported);\n  }}\n",
        prelude="",
        kernels=specs,
    )
    lowered = [lower(kernel, args.arch) for kernel in specs]
    first, *rest = lowered
    for low in rest:
        if low.preamble != first.preamble:
            raise RuntimeError(
                f"{low.kernel.symbol}: its preamble differs from {first.kernel.symbol}'s"
            )

    cu_path = out_dir / f"{CU_STEM}.cu"
    cu_path.write_text(
        "// Generated by pegainfer-gemma4/kernels/w4a16_generate.py. Do not edit.\n"
        + isolate_debug_helpers(first.preamble)
        + "".join(low.body for low in lowered)
        + render_launcher(launcher, lowered)
        + render_occupancy(lowered)
    )

    if args.vendor_includes:
        template_include, cutlass_include = vendor_includes(out_dir)
    else:
        template_include = Path(TILELANG_TEMPLATE_PATH)
        cutlass_include = Path(CUTLASS_INCLUDE_DIR)

    def named(path: Path) -> str:
        try:
            return str(Path(path).relative_to(out_dir))
        except ValueError:
            return str(path)

    lines = [f"CU_PATH={named(cu_path)}"]
    lines.append(f"TILELANG_TEMPLATE_PATH={named(template_include)}")
    lines.append(f"CUTLASS_INCLUDE_DIR={named(cutlass_include)}")
    lines.append(f"GEOMETRY={ctas},{defs.BLOCK_N},{defs.BLOCK_K},{tiling[0]}")
    smem = max(low.smem for low in lowered)
    # The probe answered with one rows=16 kernel; the whole lowering has to
    # clear the same budget, or the crate's occupancy check refuses it at load.
    if 2 * (smem + defs.SMEM_PER_BLOCK_RESERVED) > budget:
        raise RuntimeError(
            f"the {args.arch} lowering asks for {smem} B per CTA, past the two "
            f"resident CTAs a {budget} B budget holds"
        )
    lines.append(f"SMEM={smem}")
    lines.extend(
        f"NVCC_FLAG={PASS_CONFIG_NVCC_FLAG[key]}"
        for key, on in PASS_CONFIGS.items()
        if on and PASS_CONFIG_NVCC_FLAG[key]
    )
    for name, params in ((LAUNCHER, LAUNCHER_PARAMS), (OCCUPANCY, OCCUPANCY_PARAMS)):
        lines.append(f"LAUNCHER={name}|{', '.join(kind for kind, _ in params)}")
    lines.append(f"ARCH={args.arch}")
    (out_dir / "manifest.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    for low in lowered:
        print(
            f"# {low.kernel.symbol}: block {low.block} threads, {low.smem} B dynamic shared"
        )


if __name__ == "__main__":
    main()
