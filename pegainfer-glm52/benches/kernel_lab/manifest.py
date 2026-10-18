"""Manifest loading / validation + shape derivation (pure stdlib, torch-free).

One TOML per unit at benches/manifests/<unit>.toml; the file stem must equal
the `unit` field. The TOML carries metadata only — the calling convention
lives in kernel_lab/units/<adapter>.py.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

UNIT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")
ADAPTER_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
# Rows axis of the GEMV ABI: 1-8 mirror GLM52_DECODE_BUCKETS / the .cu batch
# whitelist; 16-64 are the MTP span-mapped verify rows — 16/32/48 are the
# #812 verify-span buckets (bucket-6 x span-8 = 48 at full occupancy) and 64
# is the span-8 x bucket-8 upper probe served by the multi-subtile mma.
# Anything else must crash at the boundary.
DECODE_ROWS = (1, 2, 4, 8, 16, 32, 48, 64, 96)
FP8_BLOCK = 128


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ShapeVariant:
    """One axis point with every derived buffer size."""

    rows: int
    n: int
    k: int
    act_elems: int        # bf16 [rows, k]
    weight_bytes: int     # e4m3 [n, k]
    scale_len_bytes: int  # f32 [ceil(n/128), ceil(k/128)] as raw bytes
    out_elems: int        # bf16 [rows, n]
    scratch_rule: str     # human-readable scratch sizing rule (manifest `scratch` key)


@dataclass(frozen=True)
class Manifest:
    unit: str
    phase: int
    symbol: str
    adapter: str
    capability: dict = field(default_factory=dict)
    axes: dict = field(default_factory=dict)
    shape: dict = field(default_factory=dict)
    contract: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)
    scratch: str = "unit-managed (see notes)"
    path: Path | None = field(default=None, compare=False)

    @property
    def rows(self) -> tuple[int, ...]:
        return tuple(self.axes.get("rows", ()))

    @property
    def tolerance(self) -> dict:
        return self.reference.get("tolerance", {})

    def derive_shapes(self) -> list[ShapeVariant]:
        """Buffer sizes per rows value. scale_len mirrors the Rust side:
        ceil(n/128) * ceil(k/128) * 4 (moe_gemv.rs gemv_batched_launch)."""
        n = int(self.shape["n"])
        k = int(self.shape["k"])
        scale_len = _ceil_div(n, FP8_BLOCK) * _ceil_div(k, FP8_BLOCK) * 4
        return [
            ShapeVariant(
                rows=r,
                n=n,
                k=k,
                act_elems=r * k,
                weight_bytes=n * k,
                scale_len_bytes=scale_len,
                out_elems=r * n,
                scratch_rule=self.scratch,
            )
            for r in self.rows
        ]


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _fail(path: Path, msg: str) -> ManifestError:
    return ManifestError(f"{path}: {msg}")


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise _fail(path, str(exc)) from exc
    if not isinstance(raw, dict):
        raise _fail(path, "manifest is not a TOML table")

    unit = raw.get("unit")
    if not isinstance(unit, str) or not UNIT_NAME_RE.match(unit):
        raise _fail(path, f"unit {unit!r} must match {UNIT_NAME_RE.pattern}")
    if unit != path.stem:
        raise _fail(path, f"unit {unit!r} != file stem {path.stem!r}")

    phase = raw.get("phase")
    if not isinstance(phase, int) or phase < 1:
        raise _fail(path, f"phase {phase!r} must be an int >= 1")

    symbol = raw.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        raise _fail(path, "symbol must be a non-empty string")

    adapter = raw.get("adapter")
    if not isinstance(adapter, str) or not ADAPTER_NAME_RE.match(adapter):
        raise _fail(path, f"adapter {adapter!r} must be a snake_case module name")

    capability = raw.get("capability", {})
    if not isinstance(capability, dict):
        raise _fail(path, "capability must be a table")
    if "blackwell_only" in capability and not isinstance(capability["blackwell_only"], bool):
        raise _fail(path, "capability.blackwell_only must be a bool")
    if "sm_tcgen05_only" in capability and not isinstance(capability["sm_tcgen05_only"], bool):
        raise _fail(path, "capability.sm_tcgen05_only must be a bool")

    axes = raw.get("axes")
    if not isinstance(axes, dict):
        raise _fail(path, "[axes] table required")
    rows = axes.get("rows")
    if (
        not isinstance(rows, list)
        or not rows
        or any(r not in DECODE_ROWS for r in rows)
        or len(set(rows)) != len(rows)
    ):
        raise _fail(path, f"axes.rows must be unique values from {DECODE_ROWS}")

    shape = raw.get("shape")
    if not isinstance(shape, dict):
        raise _fail(path, "[shape] table required")
    n, k = shape.get("n"), shape.get("k")
    if not isinstance(n, int) or n <= 0 or not isinstance(k, int) or k <= 0:
        raise _fail(path, "shape.n / shape.k must be positive ints")
    if k % FP8_BLOCK != 0:
        raise _fail(path, f"shape.k={k} must be a multiple of {FP8_BLOCK} (block-scale stride)")

    contract = raw.get("contract")
    if not isinstance(contract, dict) or not contract.get("inputs") or not contract.get("outputs"):
        raise _fail(path, "[contract] needs inputs/outputs")

    reference = raw.get("reference")
    if not isinstance(reference, dict) or not reference.get("mode"):
        raise _fail(path, "[reference] needs a mode")
    if reference["mode"] == "torch_tolerance":
        tol = reference.get("tolerance")
        if not isinstance(tol, dict) or not isinstance(tol.get("rel_l2"), (int, float)) or tol["rel_l2"] <= 0:
            raise _fail(path, "[reference.tolerance] needs a positive rel_l2")

    scratch = raw.get("scratch", "unit-managed (see notes)")
    if not isinstance(scratch, str) or not scratch:
        raise _fail(path, "scratch must be a non-empty string when present")

    return Manifest(
        unit=unit,
        phase=phase,
        symbol=symbol,
        adapter=adapter,
        capability=capability,
        axes=axes,
        shape=shape,
        contract=contract,
        reference=reference,
        scratch=scratch,
        path=path,
    )


def load_dir(manifests_dir: str | Path) -> list[Manifest]:
    return [load_manifest(p) for p in sorted(Path(manifests_dir).glob("*.toml"))]
