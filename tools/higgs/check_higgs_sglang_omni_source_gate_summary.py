#!/usr/bin/env python3
"""Validate a Higgs Audio SGLang-Omni source-reference gate summary."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_KEYS = {
    "status",
    "repo",
    "commit",
    "label",
    "model_dir",
    "sglang_omni_src",
    "sglang_omni_commit",
    "golden",
    "reference",
    "compare_log",
    "readiness_log",
    "sglang_omni_direct_imports",
    "sglang_omni_full_model_import",
    "source_reference_strict_comparison",
    "artifacts_nonempty",
}

OK_KEYS = {
    "status",
    "sglang_omni_direct_imports",
    "source_reference_strict_comparison",
    "artifacts_nonempty",
}

ARTIFACT_KEYS = (
    "golden",
    "reference",
    "compare_log",
    "readiness_log",
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


def require_log_line(path: Path, expected: str) -> None:
    lines = {line.strip() for line in path.read_text().splitlines()}
    if expected not in lines:
        raise ValueError(f"{path} missing expected line: {expected}")


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

    if not values["sglang_omni_commit"]:
        raise ValueError("sglang_omni_commit is empty")

    if not values["sglang_omni_full_model_import"]:
        raise ValueError("sglang_omni_full_model_import is empty")

    require_equal(values, "label", args.expected_label)
    require_equal(values, "model_dir", args.expected_model_dir)
    require_equal(values, "sglang_omni_src", args.expected_sglang_omni_src)
    require_equal(values, "sglang_omni_commit", args.expected_sglang_omni_commit)
    require_equal(values, "golden", args.expected_golden)

    if args.check_files:
        for key in ARTIFACT_KEYS:
            require_nonempty_file(Path(values[key]))
        require_log_line(Path(values["compare_log"]), "higgs one-step strict comparison: ok")
        require_log_line(Path(values["readiness_log"]), "direct_higgs_imports=ok")
        require_log_line(
            Path(values["readiness_log"]),
            f"full_higgs_model_import={values['sglang_omni_full_model_import']}",
        )

    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--expected-label")
    parser.add_argument("--expected-model-dir")
    parser.add_argument("--expected-sglang-omni-src")
    parser.add_argument("--expected-sglang-omni-commit")
    parser.add_argument("--expected-golden")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--allow-extra", action="store_true")
    args = parser.parse_args()

    values = validate(args)
    print(
        "higgs sglang-omni source gate summary: ok "
        f"commit={values['commit']} "
        f"label={values['label']} "
        f"sglang_omni_commit={values['sglang_omni_commit']} "
        f"full_model_import={values['sglang_omni_full_model_import']}"
    )


if __name__ == "__main__":
    main()
