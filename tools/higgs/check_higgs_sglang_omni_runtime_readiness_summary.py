#!/usr/bin/env python3
"""Validate a Higgs Audio SGLang-Omni full-runtime readiness summary."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_KEYS = {
    "status",
    "repo",
    "commit",
    "label",
    "python",
    "python_version",
    "sglang_omni_src",
    "sglang_omni_commit",
    "readiness_log",
    "pyproject_torch",
    "pyproject_sglang",
    "torch_version",
    "torch_cuda",
    "torch_has_cuda_pool_api",
    "sglang_version",
    "transformers_version",
    "sglang_omni_direct_imports",
    "sglang_omni_full_model_import",
    "runtime_ready",
    "artifacts_nonempty",
}

OK_KEYS = {
    "status",
    "sglang_omni_direct_imports",
    "sglang_omni_full_model_import",
    "runtime_ready",
    "artifacts_nonempty",
}


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

    require_equal(values, "status", args.expected_status)
    require_equal(values, "label", args.expected_label)
    require_equal(values, "python", args.expected_python)
    require_equal(values, "sglang_omni_src", args.expected_sglang_omni_src)
    require_equal(values, "sglang_omni_commit", args.expected_sglang_omni_commit)

    if values["status"] not in {"ok", "fail"}:
        raise ValueError(f"status must be 'ok' or 'fail', got {values['status']!r}")

    if values["runtime_ready"] not in {"ok", "fail"}:
        raise ValueError(
            f"runtime_ready must be 'ok' or 'fail', got {values['runtime_ready']!r}"
        )

    if values["status"] == "ok":
        for key in OK_KEYS:
            if values[key] != "ok":
                raise ValueError(f"{key}={values[key]!r}, expected 'ok'")

    if not values["commit"]:
        raise ValueError("commit is empty")

    if not values["sglang_omni_commit"]:
        raise ValueError("sglang_omni_commit is empty")

    nonempty_keys = (
        "python_version",
        "pyproject_torch",
        "pyproject_sglang",
        "torch_version",
        "torch_cuda",
        "torch_has_cuda_pool_api",
        "sglang_version",
        "transformers_version",
    )
    for key in nonempty_keys:
        if not values[key]:
            raise ValueError(f"{key} is empty")

    if values["status"] == "ok" and values["sglang_omni_direct_imports"] != "ok":
        raise ValueError("status=ok requires sglang_omni_direct_imports=ok")

    if values["runtime_ready"] == "ok" and values["sglang_omni_direct_imports"] != "ok":
        raise ValueError("runtime_ready=ok requires sglang_omni_direct_imports=ok")

    if values["runtime_ready"] == "ok" and values["sglang_omni_full_model_import"] != "ok":
        raise ValueError("runtime_ready=ok requires sglang_omni_full_model_import=ok")

    if values["status"] == "ok" and values["runtime_ready"] != "ok":
        raise ValueError("status=ok requires runtime_ready=ok")

    if values["status"] == "fail" and values["runtime_ready"] == "ok":
        raise ValueError("status=fail cannot report runtime_ready=ok")

    if args.check_files:
        readiness_log = Path(values["readiness_log"])
        require_nonempty_file(readiness_log)
        log_mirrors = {
            "python.version": values["python_version"],
            "pyproject.dependency.torch": values["pyproject_torch"],
            "pyproject.dependency.sglang": values["pyproject_sglang"],
            "package.torch.version": values["torch_version"],
            "package.torch.cuda": values["torch_cuda"],
            "package.torch.has_cuda_begin_allocate_current_thread_to_pool": values[
                "torch_has_cuda_pool_api"
            ],
            "package.sglang.version": values["sglang_version"],
            "package.transformers.version": values["transformers_version"],
        }
        for key, value in log_mirrors.items():
            require_log_line(readiness_log, f"{key}={value}")
        require_log_line(
            readiness_log,
            f"direct_higgs_imports={values['sglang_omni_direct_imports']}",
        )
        require_log_line(
            readiness_log,
            f"full_higgs_model_import={values['sglang_omni_full_model_import']}",
        )

    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--expected-status")
    parser.add_argument("--expected-label")
    parser.add_argument("--expected-python")
    parser.add_argument("--expected-sglang-omni-src")
    parser.add_argument("--expected-sglang-omni-commit")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--allow-extra", action="store_true")
    args = parser.parse_args()

    values = validate(args)
    print(
        "higgs sglang-omni runtime readiness summary: ok "
        f"status={values['status']} "
        f"label={values['label']} "
        f"sglang_omni_commit={values['sglang_omni_commit']} "
        f"full_model_import={values['sglang_omni_full_model_import']}"
    )


if __name__ == "__main__":
    main()
