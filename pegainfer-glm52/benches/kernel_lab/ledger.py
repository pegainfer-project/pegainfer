"""Baseline ledger: benches/baselines/<unit>.json.

File shape: {"unit": <name>, "entries": [entry, ...]} where each entry is
{unit, shape, arch, gpu, clocks_sm_mhz, median_us, p50_us, p99_us, git_rev,
timestamp}. Entries key on (shape, arch) — H200 numbers never migrate to
Blackwell and vice versa (arch-bucketed by contract).
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASELINES_DIR = Path(__file__).resolve().parents[1] / "baselines"


def ledger_path(unit: str) -> Path:
    return BASELINES_DIR / f"{unit}.json"


def _shape_key(shape: dict) -> str:
    return json.dumps(shape, sort_keys=True)


def load(unit: str) -> dict:
    path = ledger_path(unit)
    if not path.is_file():
        return {"unit": unit, "entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def find_entry(unit: str, shape: dict, arch: str) -> dict | None:
    for entry in load(unit)["entries"]:
        if _shape_key(entry["shape"]) == _shape_key(shape) and entry["arch"] == arch:
            return entry
    return None


def record(unit: str, shape: dict, arch: str, gpu: str, clocks_sm_mhz: int | None, stats) -> dict:
    doc = load(unit)
    entry = {
        "unit": unit,
        "shape": shape,
        "arch": arch,
        "gpu": gpu,
        "clocks_sm_mhz": clocks_sm_mhz,
        "median_us": round(stats.median_us, 3),
        "p50_us": round(stats.p50_us, 3),
        "p99_us": round(stats.p99_us, 3),
        "git_rev": git_rev(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    doc["entries"] = [
        e
        for e in doc["entries"]
        if not (_shape_key(e["shape"]) == _shape_key(shape) and e["arch"] == arch)
    ]
    doc["entries"].append(entry)
    doc["entries"].sort(key=lambda e: (_shape_key(e["shape"]), e["arch"]))
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path(unit).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return entry


def compare(unit: str, shape: dict, arch: str, stats, threshold_pct: float):
    """Delta table of the current bench vs the ledger entry.

    Returns (rows, baseline): rows = (metric, baseline_us, current_us,
    delta_pct, over_threshold); baseline is None when no entry matches."""
    baseline = find_entry(unit, shape, arch)
    if baseline is None:
        return [], None
    rows = []
    for metric in ("median_us", "p50_us", "p99_us"):
        base_us = float(baseline[metric])
        cur_us = getattr(stats, metric)
        delta = (cur_us - base_us) / base_us * 100.0 if base_us else float("nan")
        rows.append((metric, base_us, cur_us, delta, delta > threshold_pct))
    return rows, baseline


def git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[3],
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
