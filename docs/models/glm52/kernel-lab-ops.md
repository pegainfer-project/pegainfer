# kernel_lab ops & maintenance manual

> **TL;DR:** `pegainfer-glm52/benches/kernel_lab` is a manifest-driven check/bench/compare harness that measures the production CUDA objects (or python-native CuTe DSL kernels) per unit: TOML manifests are the behavior domain, adapters implement a three-function calling convention around `torch`-lazy imports, `--cold-l2` and `--save` are mutually exclusive protocols, the ledger buckets by `(shape, arch)`, and the CPU-only pytest guards the contract. This doc is for the person who maintains or extends the harness; the fp8 line's numbers live in `fp8-blockwise-gemm-lab.md`.

## Architecture at a glance

```
manifests/<unit>.toml            kernel_lab/registry.py        kernel_lab/units/<adapter>.py
 (unit contract: shape / rows /     discover()                   make_inputs(shape, seed)   -- seeded, reproducible
  tolerance / capability /          | imports adapters           run(lib, tensors, shape, stream)
  contract text)                    | (stdlib-only, torch-free)  reference(tensors, shape) -> f32 torch tensor
        |                                |                              |
        v                                v                              v
 kernel_lab/manifest.py            --python_native?--> kernel_lab/loader.py
  load_dir: TOML -> dataclass,                      default so: target/release/libglm52_kernel_lab.so
  name/shape/schema validation                      (build.rs links it from the SAME objects+flags as
                                                    the production archive when PEGAINFER_KERNEL_LAB=1;
                                                    ctypes, RTLD_LAZY — DeepEP shim NCCL symbols left
                                                    unresolved and never called)
                                                           |
                                                    kernel_lab/timing.py
                                                    warm protocol: bench()   — quack-style cold protocol:
                                                    bench_rotated() (--cold-l2, rotates >=3x L2 of clones);
                                                    interleaved A/B: interleaved()
                                                           |
                                                    kernel_lab/ledger.py
                                                    baselines/<unit>.json, entries keyed on (shape, arch)
                                                           |
                                                    benches/tests/ (CPU-only pytest; never imports torch)
```

Command surface (`python3 -m kernel_lab ...`): `build` (env-sets `PEGAINFER_KERNEL_LAB=1`
and runs the cargo build), `list` (CPU-only, torch-free), `check`
(torch reference vs kernel, gate on the manifest's `rel_l2`), `bench`
(`--rows/--warmup/--rounds/--inner/--cold-l2/--save`), `compare`
(ledger delta, or `--baseline-so` same-session interleaved A/B; default
threshold 5%). `--so PATH` overrides which .so gets dlopened everywhere.

Facts worth memorizing before touching anything:

- **Manifests are the registry.** `registry.discover()` glob-loads
  `manifests/*.toml`, imports `kernel_lab.units.<adapter>`, and asserts the
  three adapter functions plus an optional `SYMBOL` match. Adding or deleting
  a TOML adds or deletes the unit; there is no unit list in code other than
  the phase-1 set in `tests/test_registry.py`.
- **Import-time torch-free contract.** Every module top level is stdlib-only;
  `loader.require_torch()` imports torch inside functions. That is what keeps
  `list` and the pytest suite green on CPU/dev boxes — `tests/test_registry.py::test_no_torch_leak`
  fails the suite the moment any module-level `import torch` sneaks in.
- **The .so IS the production archive.** build.rs reuses the same nvcc task
  objects (`--compiler-options -fPIC` everywhere); tuning output numbers are
  production SASS numbers. DL is lazy (`RTLD_LAZY | RTLD_LOCAL`).
- **Seeds are derived, not re-rolled.** `data.derive_seed(seed, purpose)` gives
  each tensor family its stream, so re-generating one input never reshuffles
  another; `DEFAULT_SEED = 0x5EED` in `__main__` unless `--seed`.

## Adding a new CUTLASS (.so-symbol) unit

1. **Kernel side first**: the production symbol must exist in
   `pegainfer-kernels/csrc/glm52/*.cu` and export through the release build;
   note its ABI (arg order/types) — the adapter drives it raw via ctypes.
2. **Manifest**: copy an existing TOML (e.g. `manifests/fp8_gemm.q_b.toml`),
   rename to `<unit>.toml`; the file stem must equal the `unit` field and match
   `^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$`. Fill: `phase`, `symbol` (the extern "C"
   name), `adapter` (snake_case module), `capability`, `[axes].rows` (subset
   of `manifest.DECODE_ROWS`), `[shape].n/k` (k % 128 == 0), `[contract]`,
   `[reference]` tolerance (`rel_l2 > 0`). Optional top-level `scratch`: the
   human-readable scratch sizing rule surfaced in stats/ledger rows; omit it
   and the default is "unit-managed (see notes)".
3. **Adapter** `kernel_lab/units/<adapter>.py`: implement exactly
   `make_inputs(shape, seed)` / `run(lib, tensors, shape, stream)` /
   `reference(tensors, shape)` and set module-level `SYMBOL` equal to the
   manifest's. Drive the symbol via `loader.resolve` with an explicit ctypes
   signature; raise `loader.KernelLaunchError(SYMBOL, rc)` on non-zero
   `CUresult`. Reuse `kernel_lab.data` input factories and `kernel_lab.refs`
   torch references (lazy torch inside functions).
4. **Registry latch**: extend `EXPECTED_PHASE1_UNITS` (or the matching set for
   your phase) in `tests/test_registry.py`, and add unit-metadata tests to
   `tests/` against production anchors rather than magic numbers (see
   `tests/test_fp8_gemm.py` — it parses the .cu whitelist and the Rust
   workspace constant instead of duplicating them).
5. **Run the loop**: `kernel_lab build` (repo root) → `check <unit>` over the
   full rows axis → `bench <unit> --save` under the ledger discipline below.
6. CPU guard: `pytest tests/ -q` green before declaring done.

## Adding a python-native (CuTe DSL) unit

Same lifecycle, three deltas:

- **Manifest** carries a *placeholder* `symbol` (no .so export exists) plus
  `capability = { python_native = true }` — that key tells `test_registry`'s
  symbol-resolution check to skip you. If the kernel uses hardware that only
  exists on sm_100a-family parts (tcgen05), also set
  `capability = { sm_tcgen05_only = true }` so the run path fail-closes
  instead of dying inside the DSL runtime.
- **Adapter `run(lib, ...)` ignores `lib`** and lazy-imports the DSL kernel
  module (module level imports `cutlass`/`cuda.bindings` — keep that import
  inside `run()`, never at adapter top level, or CPU discovery breaks).
- **The shortcut**: `__main__._setup` still dlopens *a* .so even for
  python-native units (it is just unused by the adapter). When iterating
  DSL-kernel-only on a remote box, pass `--so <any previously built
  libglm52_kernel_lab.so>` — no cargo rebuild needed per `.py` edit.

**JIT/cache notes (learned on the fp8 line)**: `cute.compile` is cached per
shape/config in-process; wrapper views via `from_dlpack` cost ~70us CPU per
call, so the adapter caches wrapped views keyed on `(shape, data_ptrs,
stream)` — a changed pointer re-wraps but does not recompile; the call cache
self-caps at 512 entries. DSL wheel and vendored CUTLASS examples can drift
(4.6.0 wheel vs 4.5 examples did); when something breaks, diff against the
vendored example first.

**SASS/PTX audit recipe** (verified locally, 2026-08 — any GPU box; arch
cross-compile needs no matching card):

```bash
# Pre-create the dump dir; a JIT-cache-hit compile dumps nothing, so point
# the persistent cache somewhere fresh when you want a re-dump.
mkdir -p /tmp/dsl_dump /tmp/dsl_cache_fresh
PYTHONPATH=pegainfer-glm52/benches \
CUTE_DSL_ARCH=sm_103a CUTE_DSL_KEEP=ptx,cubin CUTE_DSL_DUMP_DIR=/tmp/dsl_dump \
CUTE_DSL_CACHE_DIR=/tmp/dsl_cache_fresh \
.venv/bin/python -c '<invoke the adapter run() once>'
# Artifacts land as <mangled-fn>.sm_103a.ptx / .sm_103a.cubin under the dump dir.
# SASS tooling: host cuobjdump may not know sm_103a without an external
# nvdisasm — use the pip wheel one: .venv's nvidia/cu13/bin/nvdisasm.
grep -c 'tcgen05\.mma\|tcgen05\.ld\|cp\.async\.bulk\.tensor\|mma\.sync' /tmp/dsl_dump/*.ptx
```

Rule of the house: **"it compiles for another arch" proves nothing** — ptxas
may silently emulate instructions with no hardware path; a retarget needs the
SASS audit (genuine tcgen05.mma present, zero fallback hits) before the
numbers may be believed.

## Capability keys (manifest `[capability]` table)

| Key | Type | Semantics | Enforced at |
|---|---|---|---|
| `blackwell_only` | bool | Device major < 10 refuses to run (`_setup`, `SystemExit`) | check/bench/compare |
| `sm_tcgen05_only` | bool | Device major ≠ 10 refuses to run with "tcgen05 (SM100+) only" | check/bench/compare |
| `python_native` | bool | No .so symbol exists; `test_registry` skips symbol resolution | tests |

These gates fire lazily — `list` never touches a device. New keys need
a type check in `manifest.load_manifest` plus a gate in `__main__._setup`
plus a latch test in `tests/test_registry.py` (mirror
`test_tcgen05_units_are_capability_gated`).

## AOT productization (`export_to_c`) — the DSL-to-production recipe

Verified 2026-08-04 on GB300. The DSL ships its own AOT: a `cute.compile`d
function (`CudaDialectJitCompiledFunction`) has `export_to_c(dir, name)`,
emitting `<name>.h` + `<name>.o` — the `.o` is PIC host-launch code with the
cubin embedded, the `.h` exposes `<name>_Kernel_Module_Load` (loads the cubin
onto every device via cudaLibrary) and `cute_dsl_<name>_wrapper(module, &mA...,
stream)`. Facts that shape any integration:

- **Shapes are baked in.** The header's tensor structs carry only `void
  *data`; layout/strides/shape are compile-time. One export per (rows, n, k,
  config) — which matches glm52's per-bucket decode scratch exactly.
- **grid.x tiles the rows axis** (64-row M tiles) since the rows-96 step;
  any m exports. Historical trap worth keeping: before that, rows>64
  compiled without complaint and silently computed only the first 64 rows
  (measured rel_l2 0.58 = sqrt(1/3) at rows=96) — a grid formula is the
  kernel author's contract, the DSL never validates coverage.
- **`libcute_dsl_runtime.so` is a hard link+run dependency.** The exported
  object calls `_cuda*`-prefixed wrapper symbols resolved from the DSL
  wheel's `lib/` (self-contained, plain libc). Link with `-lcute_dsl_runtime`
  and have the dir on `LD_LIBRARY_PATH` at run time.
- **Graph-capture split.** `Module_Load` is illegal mid-capture (library
  load); the launch wrapper itself is capture-safe. Load once before the
  first bucket capture.

The wired-up instance: `pegainfer-kernels/tools/cutedsl/export_glm52_fp8_dsl.py`
(build-time export + generated dispatch table, content-hash cached) +
`csrc/glm52/glm52_fp8_dsl_gemm.c` (uniform `(m, n, k)`-keyed entry) + the
`PEGAINFER_CUTEDSL_PYTHON` gate in build.rs. Numerical gate:
`pegainfer-kernels/tests/glm52_fp8_dsl_gate.rs`.

## Timing protocols — which one answers your question

- **Warm (default)**: same buffer set hammered repeatedly; L2-resident
  bandwidth can exceed the nominal HBM peak. Use for same-conditions A/B
  (`compare --baseline-so` interleaved) and for the regression ledger
  (`--save`). Numbers are *comparable*, not *absolute*.
- **`--cold-l2`**: rotates N buffer clones so one sweep moves ≥ 3x L2
  (quack `_bench_cuda_graph_l2_rotate` pattern) — the realistic HBM-bound
  reading. **Mutually exclusive with `--save`**: ledgers stay a warm-protocol
  historical series; cold numbers would poison regressions. Use cold numbers
  for roofline claims; never mix the two series in one statement.
- **Known floor**: torch.cuda.Event timing carries a Python/launch floor —
  below ~11us per call the number is conservative. During the fp8 line's
  tuning, pure-GPU truth came from CUDA-graph replay (capture N launches,
  replay, amortize); folding that into the harness is a planned follow-up —
  until then, treat sub-~11us absolute numbers with suspicion and trust only
  same-protocol deltas.
- Clocks: the harness *records* `nvidia-smi clocks.sm` into stats and the
  ledger; it never locks clocks. Locking is the bench owner's discipline —
  note the lock state in the doc/ledger entry when you `--save`.

## Ledger discipline

- `bench --save` writes `baselines/<unit>.json`
  (`{"unit", "entries":[{unit, shape, arch, gpu, clocks_sm_mhz, median_us,
  p50_us, p99_us, git_rev, timestamp}]}`, keyed on `(shape, arch)`). Only
  save on the **same protocol (warm), same card type, recorded clocks** —
  the arch bucketing is the only cross-card isolation; an H-series number in
  a Blackwell bucket poisons comparisons for everyone.
- `compare` reads the ledger entry at `(shape, arch=current device)` and
  prints median/p50/p99 deltas against the `--threshold` (default 5%); a
  missing bucket prints "no ledger entry" and does not fail.
- Check a ledger in before publishing: `python3 -c
  "import json; d=json.load(open('baselines/<unit>.json')); ..."` — the
  `unit` key, unique `(shape, arch)` pairs, and nonempty `entries` are the
  structure `ledger.record/compare` consume.

## Remote usage pattern (GB300 tray sessions)

Recorded pattern from the 2026-08-03 GB300 tray17 session:

1. Docker container on the tray with idle GB300s; torch comes from the
   image, not pip: create the lab venv with
   `uv venv --system-site-packages` so it inherits the image's pinned
   torch (that session: 2.11.0+cu130 aarch64), then
   `uv pip install nvidia-cutlass-dsl==4.6.0` (aarch64 wheels exist).
2. `rsync` the `pegainfer-glm52/benches/` snapshot over; DSL `.py` edits are
   then edit-rsync-run — no cargo rebuild (python-native units), with the
   `--so` placeholder pointing at a previously built
   `libglm52_kernel_lab.so`.
3. Building the .so on the tray is plain
   `PEGAINFER_KERNEL_LAB=1 cargo build --release -p pegainfer-kernels
   --features glm52` (equals `kernel_lab build`); the run-only loop does not
   need it.
4. Keep one venv per session and document box state (other jobs share
   clocks/SM availability — compare within a session, never across them).

## Next

- Fold the CUDA-graph replay protocol into `timing.py` (sub-11us truth).
- Ledger schema versioning if/when the entry shape grows.
- If more unit families ship to main, split this manual's per-family sections.
