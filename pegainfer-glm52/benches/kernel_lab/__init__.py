"""glm52-kernel-lab — GLM5.2 decode per-kernel check/bench/compare harness.

Import-safe on a torch-less CPU box: torch is imported lazily and only on the
check/bench/compare paths (see kernel_lab.loader.require_torch).
"""

__version__ = "0.1.0"
