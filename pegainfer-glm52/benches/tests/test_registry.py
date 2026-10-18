"""CPU-only registry checks — MUST NOT import torch (dev boxes may lack it).

The torch-lazy-import design is what keeps this suite green without a GPU:
manifest/registry/adapter imports are all stdlib-only. The optional .so
symbol check runs only when target/release/libglm52_kernel_lab.so exists.
"""
import sys
from pathlib import Path

import pytest

BENCHES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHES))

from kernel_lab import loader, manifest, registry  # noqa: E402

MANIFESTS = BENCHES / "manifests"
EXPECTED_PHASE1_UNITS = {
    # fp8_gemm group — the rows>8 wide route (the sm_103 tcgen05 DSL twins are
    # phase 3 and intentionally not phase-gated here).
    "fp8_gemm.q_b",
    "fp8_gemm.o_proj",
    "fp8_gemm.shared_gate_up",
    "fp8_gemm.shared_down",
}
EXPECTED_ROWS = {4, 8, 16, 32, 48, 64, 96}


def test_no_torch_leak():
    assert "torch" not in sys.modules, "registry imports must stay torch-free"


def test_manifests_parse():
    files = sorted(MANIFESTS.glob("*.toml"))
    assert files, "no manifests found"
    for f in files:
        m = manifest.load_manifest(f)
        assert m.unit and m.symbol and m.adapter and m.phase >= 1


def test_unit_names_unique_and_match_filename():
    manifests = [manifest.load_manifest(f) for f in sorted(MANIFESTS.glob("*.toml"))]
    names = [m.unit for m in manifests]
    assert len(names) == len(set(names)), f"duplicate unit names: {names}"
    for m in manifests:
        assert m.unit == m.path.stem


def test_adapter_modules_expose_contract():
    units = registry.discover(MANIFESTS)
    for unit in units.values():
        for fn in ("make_inputs", "run", "reference"):
            assert callable(getattr(unit.adapter, fn, None)), f"{unit.name} adapter lacks {fn}()"
        # Adapter-declared symbol must match the manifest metadata.
        assert getattr(unit.adapter, "SYMBOL", None) == unit.manifest.symbol


def test_shape_derivation():
    m = manifest.load_manifest(MANIFESTS / "fp8_gemm.q_b.toml")
    shapes = {v.rows: v for v in m.derive_shapes()}
    assert set(shapes) == EXPECTED_ROWS
    for rows, v in shapes.items():
        assert v.weight_bytes == 16384 * 2048
        # scale_len = ceil(n/128) * ceil(k/128) * 4 (moe_gemv.rs convention)
        assert v.scale_len_bytes == (16384 // 128) * (2048 // 128) * 4 == 8192
        assert v.act_elems == rows * 2048
        assert v.out_elems == rows * 16384


def test_mtp_verify_rows_derivation():
    # rows 16/32/64 = MTP span-mapped verify rows (span-8 x bucket-8 = 64):
    # buffers scale linearly with rows, scale/weight are rows-independent.
    # scratch is declared per-unit by the manifest `scratch` key (fp8_gemm =
    # the 32 MiB workspace carved inside the FFI, fp8.rs
    # FP8_GEMM_WORKSPACE_BYTES); derive_shapes just passes it through.
    m = manifest.load_manifest(MANIFESTS / "fp8_gemm.q_b.toml")
    shapes = {v.rows: v for v in m.derive_shapes()}
    for rows in (16, 32, 64):
        v = shapes[rows]
        assert v.act_elems == rows * 2048
        assert v.out_elems == rows * 16384
        assert v.weight_bytes == 16384 * 2048
        assert v.scale_len_bytes == 8192
    # span-8 x bucket-8 capacity: 64 rows over the full q_b projection.
    assert shapes[64].out_elems == 64 * 16384 == 1048576


def test_rows_axis_within_kernel_whitelist():
    # Guards the EP4 bucket-label-drift class of accident: the manifest axis
    # must stay within the rows the .cu dispatch admits (decode buckets plus
    # the MTP multi-subtile mma rows).
    for m in (manifest.load_manifest(f) for f in MANIFESTS.glob("*.toml")):
        assert set(m.rows) <= set(manifest.DECODE_ROWS)


def test_phase1_unit_names_stable():
    units = registry.discover(MANIFESTS)
    phase1 = {name for name, u in units.items() if u.manifest.phase == 1}
    assert phase1 == EXPECTED_PHASE1_UNITS
    for name in phase1:
        assert manifest.UNIT_NAME_RE.match(name)


def test_tcgen05_units_are_capability_gated():
    # blackwell_only (major >= 10) waves through Blackwell parts that do have
    # the tensor-core ISA but not the tcgen05 units; the DSL tc kernels must
    # be impossible to misrun, so they carry the stricter sm_tcgen05_only
    # gate (fail-closed unless device major == 10) on top of python_native.
    files = sorted(MANIFESTS.glob("fp8_gemm_dsl_tc.*.toml"))
    assert len(files) == 4
    for f in files:
        m = manifest.load_manifest(f)
        assert m.capability.get("python_native") is True
        assert m.capability.get("sm_tcgen05_only") is True


def test_demo_symbol_resolvable_when_so_present():
    so = loader.default_so_path()
    if not so.is_file():
        pytest.skip("libglm52_kernel_lab.so not built (run `kernel_lab build`)")
    lib = loader.load_library(so)
    for unit in registry.discover(MANIFESTS).values():
        # Python-native units (e.g. CuTe DSL JIT, fp8_gemm_dsl_tc.*) carry a
        # placeholder symbol by contract — there is no .so export to resolve.
        if unit.manifest.capability.get("python_native"):
            continue
        assert getattr(lib, unit.manifest.symbol, None) is not None, (
            f"{unit.manifest.symbol} not exported by {so}"
        )



def test_dsl_tc_scratch_matches_tile_cfg():
    """Manifest `scratch` must agree with the TILE_CFG split_k source of truth."""
    from kernel_lab.units.fp8_gemm_dsl_tc import TILE_CFG

    by_unit = {m.unit: m for m in manifest.load_dir(MANIFESTS) if m.unit.startswith("fp8_gemm_dsl_tc.")}
    assert len(by_unit) == 4
    shape_of = {
        "fp8_gemm_dsl_tc.q_b": (16384, 2048),
        "fp8_gemm_dsl_tc.o_proj": (6144, 16384),
        "fp8_gemm_dsl_tc.shared_gate_up": (4096, 6144),
        "fp8_gemm_dsl_tc.shared_down": (6144, 2048),
    }
    for unit, (n, k) in shape_of.items():
        _, split_k = TILE_CFG[(n, k)]
        text = by_unit[unit].scratch
        if split_k == 1:
            assert text.startswith("none"), (unit, split_k, text)
        else:
            assert f"({split_k}, rows, n)" in text, (unit, split_k, text)


def test_scratch_default_when_key_absent(tmp_path):
    """Manifests without a `scratch` key load with the explicit default."""
    src = (MANIFESTS / "fp8_gemm.q_b.toml").read_text(encoding="utf-8")
    scrubbed = "\n".join(line for line in src.splitlines() if not line.startswith("scratch ="))
    assert scrubbed != src  # the on-disk manifest really has the key
    legacy = tmp_path / "fp8_gemm.q_b.toml"
    legacy.write_text(scrubbed, encoding="utf-8")
    m = manifest.load_manifest(legacy)
    assert m.scratch == "unit-managed (see notes)"
    assert all(v.scratch_rule == "unit-managed (see notes)" for v in m.derive_shapes())
