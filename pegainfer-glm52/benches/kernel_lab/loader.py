"""ctypes loader for libglm52_kernel_lab.so.

The .so is produced by `pegainfer-kernels/build.rs` when PEGAINFER_KERNEL_LAB=1
— the same objects and nvcc flags as the production static archive. This
module never imports torch; stream/device-pointer helpers take raw ints.

ABI notes (mirror: pegainfer-kernels/src/ffi/glm52.rs):
- `Half*` / bf16 device pointers -> `c_void_p` holding `tensor.data_ptr()`;
  bf16 storage travels as raw uint16 bits, the kernel owns the interpretation.
- `cudaStream_t` -> `c_void_p` holding `torch.cuda.current_stream().cuda_stream`
  (0 == legacy default stream).
- `CUresult` -> `c_int`, 0 == CUDA_SUCCESS.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

# DeepEP shim objects inside the .so reference NCCL symbols that are
# deliberately left unresolved at link time; RTLD_NOW would fail the dlopen
# (verified on x86_64, 2026-07). Phase-1 units never call into DeepEP, so
# lazy binding is safe.
_DLOPEN_MODE = os.RTLD_LAZY | os.RTLD_LOCAL

CUDA_ERROR_NAMES = {
    0: "CUDA_SUCCESS",
    1: "CUDA_ERROR_INVALID_VALUE",
    2: "CUDA_ERROR_OUT_OF_MEMORY",
    3: "CUDA_ERROR_NOT_INITIALIZED",
    4: "CUDA_ERROR_DEINITIALIZED",
    100: "CUDA_ERROR_NO_DEVICE",
    700: "CUDA_ERROR_ILLEGAL_ADDRESS",
    701: "CUDA_ERROR_LAUNCH_FAILED",
    719: "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
    801: "CUDA_ERROR_NOT_SUPPORTED",
}


class KernelLaunchError(RuntimeError):
    """A production symbol returned a non-zero CUresult."""

    def __init__(self, symbol: str, code: int):
        self.code = code
        super().__init__(f"{symbol} failed: {CUDA_ERROR_NAMES.get(code, f'CUresult({code})')}")


def repo_root() -> Path:
    # kernel_lab/ -> benches/ -> pegainfer-glm52/ -> repo root
    return Path(__file__).resolve().parents[3]


def default_so_path() -> Path:
    return repo_root() / "target" / "release" / "libglm52_kernel_lab.so"


def require_torch():
    """Import torch or die with a clear message (lazy-dependency contract)."""
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "kernel_lab: this command needs PyTorch with CUDA. torch is an "
            "optional, lazily imported dependency — install glm52-kernel-lab "
            "into the repo .venv that already carries the oracle-pinned torch "
            "(see pegainfer-glm52/benches/README.md)."
        ) from exc
    return torch


def _preload_cuda_libs() -> None:
    """Best-effort preload of cudart/cublas/nvrtc from the CUDA toolkit when
    the .so's DT_NEEDED entries are not on the default loader path."""
    for root in filter(None, (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda")):
        lib64 = Path(root) / "lib64"
        if not lib64.is_dir():
            continue
        for name in ("libcudart.so", "libcublasLt.so", "libcublas.so", "libnvrtc.so"):
            for candidate in sorted(lib64.glob(name + "*")):
                try:
                    ctypes.CDLL(str(candidate), mode=os.RTLD_LAZY | os.RTLD_GLOBAL)
                    break
                except OSError:
                    continue
        return


def load_library(path: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    so = Path(path) if path else default_so_path()
    if not so.is_file():
        raise SystemExit(
            f"kernel_lab: {so} not found — build it with `kernel_lab build` "
            "(PEGAINFER_KERNEL_LAB=1 cargo build --release -p pegainfer-kernels "
            "--features glm52)"
        )
    try:
        return ctypes.CDLL(str(so), mode=_DLOPEN_MODE)
    except OSError:
        _preload_cuda_libs()
        return ctypes.CDLL(str(so), mode=_DLOPEN_MODE)


def resolve(lib: ctypes.CDLL, name: str, argtypes: list):
    """Fetch a symbol and pin its ctypes signature (CUresult return)."""
    try:
        fn = getattr(lib, name)
    except AttributeError as exc:
        raise SystemExit(f"kernel_lab: symbol {name} not found in {lib._name}") from exc
    fn.restype = ctypes.c_int
    fn.argtypes = argtypes
    return fn


def as_dev_ptr(tensor) -> ctypes.c_void_p:
    """Raw device pointer of a torch tensor as void* (bf16 passed as bits)."""
    return ctypes.c_void_p(tensor.data_ptr())


def current_stream_ptr() -> ctypes.c_void_p:
    """cudaStream_t for torch's current stream (0 == legacy default stream)."""
    torch = require_torch()
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
