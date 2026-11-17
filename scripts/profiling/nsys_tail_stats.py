#!/usr/bin/env python3
"""Summarize nsys sqlite traces with tail-aware latency statistics."""

from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def format_value(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value.replace("|", "\\|")
    number = float(value)
    if abs(number) >= 1000.0:
        return f"{number:.1f}"
    if abs(number) >= 10.0:
        return f"{number:.2f}"
    return f"{number:.3f}"


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"pragma table_info({table})")]


def table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute("select name from sqlite_master where type='table'")
    }


def string_value_column(conn: sqlite3.Connection) -> str | None:
    if "StringIds" not in table_names(conn):
        return None
    columns = table_columns(conn, "StringIds")
    if "value" in columns:
        return "value"
    if "string" in columns:
        return "string"
    return None


def collect_kernel_durations(conn: sqlite3.Connection) -> dict[str, list[int]]:
    tables = table_names(conn)
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        return {}
    columns = table_columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
    name_column = next(
        (
            column
            for column in ("demangledName", "shortName", "mangledName", "name")
            if column in columns
        ),
        None,
    )
    if name_column is None:
        raise RuntimeError(f"kernel table has no recognized name column: {columns}")

    string_column = string_value_column(conn)
    if string_column and name_column != "name":
        query = f"""
            select coalesce(s.{string_column}, cast(k.{name_column} as text)),
                   k.end - k.start
            from CUPTI_ACTIVITY_KIND_KERNEL k
            left join StringIds s on k.{name_column} = s.id
            where k.end > k.start
        """
    else:
        query = f"""
            select cast(k.{name_column} as text), k.end - k.start
            from CUPTI_ACTIVITY_KIND_KERNEL k
            where k.end > k.start
        """
    return collect_rows(conn, query)


def collect_api_durations(conn: sqlite3.Connection) -> dict[str, list[int]]:
    tables = table_names(conn)
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        table = "CUPTI_ACTIVITY_KIND_RUNTIME"
    elif "CUDA_API_TRACE" in tables:
        table = "CUDA_API_TRACE"
    else:
        return {}

    columns = table_columns(conn, table)
    name_column = next(
        (column for column in ("nameId", "name", "cbid") if column in columns),
        None,
    )
    if name_column is None:
        raise RuntimeError(f"CUDA API table has no recognized name column: {columns}")

    string_column = string_value_column(conn)
    if name_column == "nameId" and string_column:
        query = f"""
            select coalesce(s.{string_column}, cast(a.{name_column} as text)),
                   a.end - a.start
            from {table} a
            left join StringIds s on a.{name_column} = s.id
            where a.end > a.start
        """
    else:
        query = f"""
            select cast(a.{name_column} as text), a.end - a.start
            from {table} a
            where a.end > a.start
        """
    return collect_rows(conn, query)


def collect_rows(conn: sqlite3.Connection, query: str) -> dict[str, list[int]]:
    rows: dict[str, list[int]] = {}
    for name, duration_ns in conn.execute(query):
        rows.setdefault(str(name), []).append(int(duration_ns))
    return rows


def summarize(durations: dict[str, list[int]]) -> list[dict[str, float | int | str]]:
    summaries: list[dict[str, float | int | str]] = []
    for name, values in durations.items():
        sorted_values = sorted(values)
        count = len(sorted_values)
        total = float(sum(sorted_values))
        avg = total / count
        variance = sum((value - avg) ** 2 for value in sorted_values) / count
        p50 = percentile(sorted_values, 0.50)
        p95 = percentile(sorted_values, 0.95)
        p99 = percentile(sorted_values, 0.99)
        max_value = float(sorted_values[-1])
        summaries.append(
            {
                "name": name,
                "count": count,
                "total_ms": total / 1_000_000.0,
                "avg_us": avg / 1000.0,
                "std_us": math.sqrt(variance) / 1000.0,
                "p50_us": p50 / 1000.0,
                "p95_us": p95 / 1000.0,
                "p99_us": p99 / 1000.0,
                "max_us": max_value / 1000.0,
                "p99_p50": p99 / p50 if p50 else 0.0,
                "max_p50": max_value / p50 if p50 else 0.0,
            }
        )
    return summaries


def tail_score(row: dict[str, float | int | str]) -> float:
    max_us = float(row["max_us"])
    count = int(row["count"])
    max_p50 = float(row["max_p50"])
    return max_us * math.log2(count + 1) * max(1.0, max_p50)


def markdown_table(rows: list[dict[str, float | int | str]], limit: int) -> str:
    headers = [
        "name",
        "count",
        "total_ms",
        "avg_us",
        "std_us",
        "p50_us",
        "p95_us",
        "p99_us",
        "max_us",
        "p99/p50",
        "max/p50",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows[:limit]:
        values = [
            row["name"],
            row["count"],
            row["total_ms"],
            row["avg_us"],
            row["std_us"],
            row["p50_us"],
            row["p95_us"],
            row["p99_us"],
            row["max_us"],
            row["p99_p50"],
            row["max_p50"],
        ]
        lines.append("| " + " | ".join(format_value(value) for value in values) + " |")
    return "\n".join(lines)


def render_report(sqlite_path: Path, limit: int) -> str:
    with sqlite3.connect(sqlite_path) as conn:
        kernel_rows = summarize(collect_kernel_durations(conn))
        api_rows = summarize(collect_api_durations(conn))

    kernel_by_total = sorted(kernel_rows, key=lambda row: float(row["total_ms"]), reverse=True)
    kernel_by_tail = sorted(kernel_rows, key=tail_score, reverse=True)
    api_by_total = sorted(api_rows, key=lambda row: float(row["total_ms"]), reverse=True)
    api_by_tail = sorted(api_rows, key=tail_score, reverse=True)

    sections = [
        "# nsys tail summary",
        "",
        f"source: `{sqlite_path}`",
        "",
        "## Kernel by total time",
        markdown_table(kernel_by_total, limit),
        "",
        "## Kernel tail candidates",
        markdown_table(kernel_by_tail, limit),
        "",
        "## CUDA API by total time",
        markdown_table(api_by_total, limit),
        "",
        "## CUDA API tail candidates",
        markdown_table(api_by_tail, limit),
        "",
    ]
    return "\n".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = render_report(args.sqlite, args.limit)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
