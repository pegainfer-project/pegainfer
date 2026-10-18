"""CPU-only checks for the fp8_gemm unit group (the rows>8 wide route).

MUST NOT import torch and must not depend on sibling unit groups — manifests
and the adapter are loaded by exact name, never via registry.discover over
the whole manifests dir.
"""
import importlib
import re
import sys
from pathlib import Path

BENCHES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHES))

from kernel_lab import loader, manifest  # noqa: E402

MANIFESTS = BENCHES / "manifests"
REPO = BENCHES.parents[1]
CU_GEMV = REPO / "pegainfer-kernels/csrc/glm52/glm52_moe_gemv.cu"
CU_GEMM = REPO / "pegainfer-kernels/csrc/glm52/glm52_fp8_gemm.cu"
FP8_RS = REPO / "pegainfer-glm52/src/fp8.rs"

SYMBOL = "glm52_fp8_groupwise_gemm_sm100_cuda"
EXPECTED_ROWS = [4, 8, 16, 32, 48, 64, 96]
# unit -> (n, k) — the four whitelisted production projections that anchor
# the GEMV-register-tile / multi-subtile-mma / fp8-GEMM three-way A/B.
UNITS = {
    "fp8_gemm.q_b": (16384, 2048),
    "fp8_gemm.o_proj": (6144, 16384),
    "fp8_gemm.shared_gate_up": (4096, 6144),
    "fp8_gemm.shared_down": (6144, 2048),
}


def _load(name):
    return manifest.load_manifest(MANIFESTS / f"{name}.toml")


def test_manifest_metadata():
    for name, (n, k) in UNITS.items():
        m = _load(name)
        assert m.symbol == SYMBOL
        assert m.adapter == "fp8_gemm_groupwise"
        # Per-rank projection shapes — EP-independent by construction.
        assert m.phase == 1
        assert m.capability.get("blackwell_only") is True
        # Scratch is manifest-declared: these units carve the fixed CUTLASS
        # workspace inside the FFI (fp8.rs FP8_GEMM_WORKSPACE_BYTES).
        assert "FP8_GEMM_WORKSPACE_BYTES" in m.scratch
        assert list(m.rows) == EXPECTED_ROWS
        assert m.shape == {"n": n, "k": k}
        assert m.tolerance["rel_l2"] == 0.02
        assert m.reference["mode"] == "torch_tolerance"


def test_rows_axis_satisfies_ffi_rules():
    # The FFI requires m % 4 == 0 and k % 128 == 0; every axis row and every
    # manifest k must satisfy both by construction or the unit would die at
    # the boundary instead of benching.
    for name, (n, k) in UNITS.items():
        m = _load(name)
        assert all(r % 4 == 0 for r in m.rows), name
        assert k % 128 == 0 and n % 16 == 0, name


def test_adapter_matches_manifest_and_stays_torch_free():
    mod = importlib.import_module("kernel_lab.units.fp8_gemm_groupwise")
    assert "torch" not in sys.modules, "fp8_gemm adapter must stay torch-free at import"
    assert mod.SYMBOL == SYMBOL
    for fn in ("make_inputs", "run", "reference"):
        assert callable(getattr(mod, fn)), fn


def test_shapes_exist_in_production_whitelist():
    """Each unit's (n, k) must be a real production projection — parse the
    batched GEMV whitelist (single fact source for linear shapes)."""
    src = CU_GEMV.read_text()
    wl = src.split("bool whitelisted_linear_shape(int n, int k)", 1)[1]
    for name, (n, k) in UNITS.items():
        pat = rf"n == {n}\s+&& k == {k}"
        assert re.search(pat, wl), f"{name}: ({n}, {k}) missing from whitelisted_linear_shape"


def test_symbol_is_extern_c_and_workspace_constant_matches():
    cu = CU_GEMM.read_text()
    assert f'extern "C" CUresult {SYMBOL}(' in cu
    rs = FP8_RS.read_text()
    m = re.search(r"FP8_GEMM_WORKSPACE_BYTES: usize = (\d+) << (\d+)", rs)
    assert m, "fp8.rs workspace constant moved"
    from kernel_lab.units import fp8_gemm_groupwise as adapter
    assert adapter.WORKSPACE_BYTES == int(m.group(1)) << int(m.group(2))
