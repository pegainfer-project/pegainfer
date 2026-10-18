"""Scan benches/manifests/ and pair each manifest with its adapter module.

Importing adapter modules is torch-free by contract (module level is stdlib
only; torch is imported inside functions), so discovery works on CPU boxes.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from kernel_lab import manifest as manifest_mod

MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"
ADAPTER_FUNCS = ("make_inputs", "run", "reference")


@dataclass
class RegisteredUnit:
    manifest: manifest_mod.Manifest
    adapter: object  # kernel_lab.units.<adapter> module

    @property
    def name(self) -> str:
        return self.manifest.unit


def discover(manifests_dir: str | Path | None = None) -> dict[str, RegisteredUnit]:
    manifests = manifest_mod.load_dir(manifests_dir or MANIFESTS_DIR)
    units: dict[str, RegisteredUnit] = {}
    for m in manifests:
        if m.unit in units:
            raise manifest_mod.ManifestError(f"duplicate unit name {m.unit!r}")
        module = importlib.import_module(f"kernel_lab.units.{m.adapter}")
        for fn in ADAPTER_FUNCS:
            if not callable(getattr(module, fn, None)):
                raise manifest_mod.ManifestError(
                    f"{m.unit}: adapter kernel_lab.units.{m.adapter} lacks {fn}()"
                )
        symbol = getattr(module, "SYMBOL", None)
        if symbol is not None and symbol != m.symbol:
            raise manifest_mod.ManifestError(
                f"{m.unit}: manifest symbol {m.symbol!r} != adapter SYMBOL {symbol!r}"
            )
        units[m.unit] = RegisteredUnit(manifest=m, adapter=module)
    return units
