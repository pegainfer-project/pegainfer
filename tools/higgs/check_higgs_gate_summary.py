#!/usr/bin/env python3
"""Validate a Higgs Audio one-step CUDA gate summary."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_KEYS = {
    "status",
    "repo",
    "commit",
    "label",
    "model_dir",
    "golden",
    "sm",
    "nvcc_jobs",
    "actual",
    "session_actual",
    "compare_log",
    "session_smoke_log",
    "session_compare_log",
    "auto_view",
    "semantic_comparison",
    "session_semantic_comparison",
    "duplicate_request_id_guard",
    "artifacts_nonempty",
}

OK_KEYS = {
    "status",
    "semantic_comparison",
    "session_semantic_comparison",
    "duplicate_request_id_guard",
    "artifacts_nonempty",
}

ARTIFACT_KEYS = (
    "golden",
    "actual",
    "session_actual",
    "compare_log",
    "session_smoke_log",
    "session_compare_log",
)

AUTO_VIEW_FILES = (
    "config.json",
    "generation_config.json",
    "higgs-qwen3-tensor-aliases.json",
)


def parse_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_no}: expected key=value, got {raw_line!r}")
        key, value = line.split("=", 1)
        if not key:
            raise ValueError(f"{path}:{line_no}: empty key")
        if key in values:
            raise ValueError(f"{path}:{line_no}: duplicate key {key!r}")
        values[key] = value
    return values


def require_equal(values: dict[str, str], key: str, expected: str | None) -> None:
    if expected is not None and values.get(key) != expected:
        raise ValueError(f"{key}={values.get(key)!r}, expected {expected!r}")


def require_nonempty_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"required artifact missing or empty: {path}")


def validate(args: argparse.Namespace) -> dict[str, str]:
    summary = args.summary
    if not summary.is_file() or summary.stat().st_size == 0:
        raise ValueError(f"summary missing or empty: {summary}")

    values = parse_summary(summary)
    missing = sorted(REQUIRED_KEYS - values.keys())
    if missing:
        raise ValueError(f"summary missing required keys: {', '.join(missing)}")

    extra = sorted(values.keys() - REQUIRED_KEYS)
    if extra and not args.allow_extra:
        raise ValueError(f"summary has unknown keys: {', '.join(extra)}")

    for key in OK_KEYS:
        if values[key] != "ok":
            raise ValueError(f"{key}={values[key]!r}, expected 'ok'")

    if not values["commit"]:
        raise ValueError("commit is empty")

    if not values["sm"].isdigit():
        raise ValueError(f"sm must be numeric, got {values['sm']!r}")

    if not values["nvcc_jobs"].isdigit():
        raise ValueError(f"nvcc_jobs must be numeric, got {values['nvcc_jobs']!r}")

    require_equal(values, "label", args.expected_label)
    require_equal(values, "sm", args.expected_sm)
    require_equal(values, "nvcc_jobs", args.expected_nvcc_jobs)
    require_equal(values, "model_dir", args.expected_model_dir)
    require_equal(values, "golden", args.expected_golden)

    if args.check_files:
        for key in ARTIFACT_KEYS:
            require_nonempty_file(Path(values[key]))
        auto_view = Path(values["auto_view"])
        if not auto_view.is_dir():
            raise ValueError(f"auto_view is not a directory: {auto_view}")
        for name in AUTO_VIEW_FILES:
            require_nonempty_file(auto_view / name)

    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--expected-label")
    parser.add_argument("--expected-sm")
    parser.add_argument("--expected-nvcc-jobs")
    parser.add_argument("--expected-model-dir")
    parser.add_argument("--expected-golden")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--allow-extra", action="store_true")
    args = parser.parse_args()

    values = validate(args)
    print(
        "higgs gate summary: ok "
        f"commit={values['commit']} "
        f"label={values['label']} "
        f"sm={values['sm']}"
    )


if __name__ == "__main__":
    main()
