"""kernel_lab CLI: build / list / check / bench / compare.

Iteration loop (line doc docs/models/glm52/fp8-blockwise-gemm-lab.md): edit
.cu -> `kernel_lab build` -> `check` -> `bench` -> `compare`; the top gate is
a same-session end-to-end serving A/B (vllm-bench over HTTP).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from kernel_lab import ledger, loader, registry, timing
from kernel_lab.refs import compute_metrics

DEFAULT_SEED = 0x5EED


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="kernel_lab", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="PEGAINFER_KERNEL_LAB=1 cargo build the .so")
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("list", help="list registered units (CPU-only, no torch)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("check", help="run vs torch reference, gate on manifest tolerance")
    _unit_args(p)
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("bench", help="time the production launch at capacity shapes")
    _unit_args(p)
    _bench_args(p)
    p.add_argument("--save", action="store_true", help="write results into the baseline ledger")
    p.set_defaults(func=_cmd_bench)

    p = sub.add_parser("compare", help="delta vs the ledger, or interleaved A/B vs a saved .so")
    _unit_args(p)
    _bench_args(p)
    p.add_argument("--threshold", type=float, default=5.0, help="regression threshold in %% (default 5)")
    p.add_argument("--baseline-so", default=None, help="saved pre-change .so for same-session interleaved A/B")
    p.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


def _unit_args(p) -> None:
    p.add_argument("unit")
    p.add_argument("--rows", type=int, action="append", default=None,
                   help="rows axis value (repeatable); default: all manifest rows")
    p.add_argument("--so", default=None, help="path to libglm52_kernel_lab.so")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)


def _bench_args(p) -> None:
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--inner", type=int, default=10)
    p.add_argument("--cold-l2", action="store_true",
                   help="rotate tensor clones so each call misses L2 (quack-style "
                        "cold-cache protocol); refused together with --save to keep "
                        "ledgers warm-protocol comparable")


# --- build -------------------------------------------------------------------

def _cmd_build(args) -> int:
    root = loader.repo_root()
    env = dict(os.environ, PEGAINFER_KERNEL_LAB="1")
    cmd = ["cargo", "build", "--release", "-p", "pegainfer-kernels", "--features", "glm52"]
    print("kernel_lab build:", " ".join(cmd), f"(cwd={root})", flush=True)
    rc = subprocess.run(cmd, cwd=root, env=env).returncode
    if rc != 0:
        print(f"kernel_lab build: cargo failed with {rc}", file=sys.stderr)
        return rc
    so = loader.default_so_path()
    ok = so.is_file()
    print(f"kernel_lab .so: {so}" + ("" if ok else "  (MISSING — build.rs did not emit it)"))
    return 0 if ok else 1


# --- list --------------------------------------------------------------------

def _unit_summary(u) -> dict:
    m = u.manifest
    return {
        "phase": m.phase,
        "symbol": m.symbol,
        "adapter": m.adapter,
        "capability": m.capability,
        "rows": list(m.rows),
        "shape": m.shape,
        "tolerance": m.tolerance,
        "derived": [vars(v) for v in m.derive_shapes()],
    }


def _cmd_list(args) -> int:
    units = registry.discover()
    if args.json:
        print(json.dumps({name: _unit_summary(u) for name, u in units.items()}, indent=2))
        return 0
    for name, u in units.items():
        m = u.manifest
        biggest = max(m.derive_shapes(), key=lambda v: v.rows)
        print(name)
        print(f"  phase={m.phase}  symbol={m.symbol}  adapter=kernel_lab.units.{m.adapter}")
        print(f"  capability={m.capability or '{}'}  rows={list(m.rows)}")
        print(f"  shape: n={m.shape['n']} k={m.shape['k']}  accumulation={m.contract.get('accumulation', '?')}")
        print(f"  derived @rows={biggest.rows}: act={biggest.act_elems} bf16 elems, "
              f"weight={biggest.weight_bytes} B, scale={biggest.scale_len_bytes} B, "
              f"out={biggest.out_elems} bf16 elems")
        print(f"  scratch: {biggest.scratch_rule}")
        print(f"  reference: mode={m.reference.get('mode')} rel_l2<={m.tolerance.get('rel_l2')}")
        if m.tolerance.get("note"):
            print(f"  tolerance note: {m.tolerance['note']}")
        if m.contract.get("notes"):
            print(f"  notes: {m.contract['notes']}")
    return 0


# --- shared GPU plumbing -----------------------------------------------------

def _get_unit(units, name: str):
    if name not in units:
        raise SystemExit(f"kernel_lab: unknown unit {name!r}; available: {', '.join(units)}")
    return units[name]


def _gpu_context():
    torch = loader.require_torch()
    if not torch.cuda.is_available():
        raise SystemExit("kernel_lab: no CUDA device visible")
    major, minor = torch.cuda.get_device_capability()
    return torch, f"sm_{major}{minor}", torch.cuda.get_device_name(), major


def _select_rows(manifest, requested) -> list[int]:
    rows = requested if requested else list(manifest.rows)
    bad = [r for r in rows if r not in manifest.rows]
    if bad:
        raise SystemExit(f"{manifest.unit}: rows {bad} not in manifest axes {list(manifest.rows)}")
    return rows


def _setup(args, need_capability: bool):
    units = registry.discover()
    u = _get_unit(units, args.unit)
    torch, arch, gpu, major = _gpu_context()
    if need_capability and u.manifest.capability.get("blackwell_only") and major < 10:
        raise SystemExit(f"{u.name}: Blackwell-only unit (fail-closed); device capability major={major}")
    # tcgen05 units exist only on sm_100a-family datacenter chips (device
    # major 10); blackwell_only permits other Blackwell parts, where a DSL
    # UMMA kernel would die deep inside the JIT instead of failing cleanly.
    if need_capability and u.manifest.capability.get("sm_tcgen05_only") and major != 10:
        raise SystemExit(f"{u.name}: tcgen05 (SM100+) only (fail-closed); device capability major={major}")
    lib = loader.load_library(args.so)
    stream = loader.current_stream_ptr()
    return u, torch, arch, gpu, lib, stream


def _shapes(u, args) -> list[dict]:
    return [
        {"rows": r, "n": u.manifest.shape["n"], "k": u.manifest.shape["k"]}
        for r in _select_rows(u.manifest, args.rows)
    ]


# --- check -------------------------------------------------------------------

def _cmd_check(args) -> int:
    u, torch, arch, gpu, lib, stream = _setup(args, need_capability=True)
    ok = True
    for shape in _shapes(u, args):
        tensors = u.adapter.make_inputs(shape, args.seed)
        u.adapter.run(lib, tensors, shape, stream)
        torch.cuda.synchronize()
        want = u.adapter.reference(tensors, shape)
        metrics = compute_metrics(tensors["out"], want)
        limit = u.manifest.tolerance.get("rel_l2")
        passed = limit is None or metrics["rel_l2"] <= limit
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {u.name} rows={shape['rows']} "
              f"n={shape['n']} k={shape['k']} ({arch}, {gpu})")
        print(f"       rel_l2={metrics['rel_l2']:.4e} (tol {limit})  cosine={metrics['cosine']:.6f}  "
              f"max_abs={metrics['max_abs']:.4e}  mean_abs={metrics['mean_abs']:.4e}")
    if u.manifest.tolerance.get("note"):
        print(f"tolerance note: {u.manifest.tolerance['note']}")
    return 0 if ok else 1


# --- bench / compare ---------------------------------------------------------

def _print_stats(tag: str, shape: dict, stats) -> None:
    clocks = f"{stats.clocks_sm_mhz} MHz" if stats.clocks_sm_mhz else "unknown (nvidia-smi miss)"
    print(f"{tag} rows={shape['rows']} n={shape['n']} k={shape['k']}: "
          f"median={stats.median_us:.2f} us  p50={stats.p50_us:.2f}  p99={stats.p99_us:.2f}  "
          f"mean={stats.mean_us:.2f}  rounds={len(stats.samples_us)}  clocks.sm={clocks}")


def _cmd_bench(args) -> int:
    u, torch, arch, gpu, lib, stream = _setup(args, need_capability=True)
    if args.cold_l2 and args.save:
        raise SystemExit("bench: --cold-l2 numbers are a different protocol; refusing --save "
                         "to keep ledgers comparable")
    for shape in _shapes(u, args):
        tensors = u.adapter.make_inputs(shape, args.seed)
        if args.cold_l2:
            copies = timing.l2_rotate_copies(tensors)
            stats = timing.bench_rotated(
                lambda ts: u.adapter.run(lib, ts, shape, stream),
                copies, args.warmup, args.rounds, args.inner,
            )
            _print_stats(f"bench[cold-l2 x{len(copies)}]", shape, stats)
        else:
            stats = timing.bench(
                lambda: u.adapter.run(lib, tensors, shape, stream),
                args.warmup, args.rounds, args.inner,
            )
            _print_stats("bench", shape, stats)
        if args.save:
            ledger.record(u.name, shape, arch, gpu, stats.clocks_sm_mhz, stats)
            print(f"       ledger -> {ledger.ledger_path(u.name)}")
    return 0


def _cmd_compare(args) -> int:
    u, torch, arch, gpu, lib, stream = _setup(args, need_capability=True)
    any_over = False
    if args.baseline_so:
        base_lib = loader.load_library(args.baseline_so)
        for shape in _shapes(u, args):
            base_tensors = u.adapter.make_inputs(shape, args.seed)
            cur_tensors = u.adapter.make_inputs(shape, args.seed)
            stats_base, stats_cur = timing.interleaved(
                lambda: u.adapter.run(base_lib, base_tensors, shape, stream),
                lambda: u.adapter.run(lib, cur_tensors, shape, stream),
                args.warmup, args.rounds, args.inner,
            )
            delta = (stats_cur.median_us - stats_base.median_us) / stats_base.median_us * 100.0
            over = delta > args.threshold
            any_over |= over
            verdict = "REGRESSION" if over else "OK"
            print(f"[{verdict}] {u.name} rows={shape['rows']} interleaved A/B "
                  f"(baseline={args.baseline_so}): median {stats_base.median_us:.2f} -> "
                  f"{stats_cur.median_us:.2f} us  delta={delta:+.2f}% (threshold {args.threshold}%)")
        return 1 if any_over else 0

    for shape in _shapes(u, args):
        tensors = u.adapter.make_inputs(shape, args.seed)
        stats = timing.bench(
            lambda: u.adapter.run(lib, tensors, shape, stream),
            args.warmup, args.rounds, args.inner,
        )
        _print_stats("current", shape, stats)
        rows_out, baseline = ledger.compare(u.name, shape, arch, stats, args.threshold)
        if baseline is None:
            print(f"       no ledger entry for this shape/arch ({arch}) — "
              f"run `kernel_lab bench {u.name} --rows {shape['rows']} --save` first")
            continue
        print(f"       baseline: git={baseline['git_rev']} @ {baseline['timestamp']} "
              f"({baseline['gpu']}, clocks.sm={baseline['clocks_sm_mhz']} MHz)")
        for metric, base_us, cur_us, delta, over in rows_out:
            any_over |= over
            print(f"       {metric:10s} {base_us:9.2f} -> {cur_us:9.2f} us  "
                  f"delta={delta:+6.2f}%  {'REGRESSION' if over else 'ok'}")
    return 1 if any_over else 0


if __name__ == "__main__":
    sys.exit(main())
