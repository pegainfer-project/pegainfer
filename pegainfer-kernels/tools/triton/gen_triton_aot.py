import argparse
import re
import runpy
import sys
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.driver import ty_to_cpp

try:
    from triton.tools.compile import CompileArgs, compile_kernel
except ImportError:
    CompileArgs = None
    compile_kernel = None


class OfflineCudaDriver:
    def __init__(self, target: GPUTarget):
        self._target = target

    def get_current_target(self) -> GPUTarget:
        return self._target

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)


def activate_target_driver(target: str | None) -> None:
    if target is None:
        return

    backend, arch, warp_size = target.split(":")
    if backend != "cuda":
        raise ValueError(f"unsupported Triton AOT target backend: {backend}")

    triton.runtime.driver.set_active(
        OfflineCudaDriver(GPUTarget(backend=backend, arch=int(arch), warp_size=int(warp_size)))
    )


def compile_kernel_compat(args: argparse.Namespace):
    if compile_kernel is not None:
        func_name, output_files = compile_kernel(
            CompileArgs(
                path=args.kernel_path,
                kernel_name=args.kernel_name,
                signature=args.signature,
                grid=args.grid,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
                out_name=args.out_name,
                out_path=Path(args.out_dir) / args.out_name,
            )
        )
        c_path = next(path for path in output_files if path.suffix == ".c")
        return func_name, c_path

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "triton.tools.compile",
            args.kernel_path,
            "--kernel-name",
            args.kernel_name,
            "--num-warps",
            str(args.num_warps),
            "--num-stages",
            str(args.num_stages),
            "--out-name",
            args.out_name,
            "--out-path",
            str(out_dir / args.out_name),
            "--signature",
            args.signature,
            "--grid",
            args.grid,
        ]
        runpy.run_module("triton.tools.compile", run_name="__main__")
    finally:
        sys.argv = old_argv

    candidates = sorted(out_dir.glob(f"{args.out_name}.*.c"))
    if not candidates:
        raise RuntimeError(f"failed to find generated Triton C stub in {out_dir}")
    c_path = max(candidates, key=lambda path: path.stat().st_mtime)
    match = re.search(r"CUresult\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", c_path.read_text())
    if match is None:
        raise RuntimeError(f"failed to parse Triton function name from {c_path}")
    return match.group(1), c_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-path", required=True)
    parser.add_argument("--kernel-name", required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    activate_target_driver(args.target)

    func_name, c_path = compile_kernel_compat(args)

    print(f"FUNC_NAME={func_name}")
    print(f"C_PATH={c_path}")


if __name__ == "__main__":
    main()
