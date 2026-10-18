# glm52 kernel_lab — fp8 blockwise GEMM lab

Manifest-driven per-kernel check/bench/compare bench harness for GLM5.2. This
checkout ships only the fp8 blockwise GEMM wide route (rows > 8, #812). The
single source of truth for kernels stays `pegainfer-kernels/csrc/glm52/*.cu` —
the harness ctypes-dlopens `libglm52_kernel_lab.so`, which
`pegainfer-kernels/build.rs` links from the exact same objects and nvcc flags
as the production static archive when `PEGAINFER_KERNEL_LAB=1` (a default
build runs zero extra commands), so the thing under test is the production
SASS.

## Units on the books (8)

- `fp8_gemm.{q_b,o_proj,shared_gate_up,shared_down}` — ctypes units over the
  production CUTLASS route (symbol `glm52_fp8_groupwise_gemm_sm100_cuda`, the
  four per-rank projection shapes). sm_100a-family only; any other target
  compiles a `CUDA_ERROR_NOT_SUPPORTED` stub and `KernelLaunchError`
  fail-closes at runtime (`capability.blackwell_only`).
- `fp8_gemm_dsl_tc.{same four shapes}` — hand-written CuTe DSL tcgen05 lab
  line (TMA + tcgen05.mma + TMEM block accumulator + software blockscale;
  per-shape tile-N/split-K tuning). Python JIT units
  (`capability.python_native`, placeholder `SYMBOL`), sm_100a-family parts
  only — other chips lack tcgen05, so `capability.sm_tcgen05_only`
  fail-closes at the check/bench/compare entry. Runtime dependency: the repo
  `.venv`'s `nvidia-cutlass-dsl 4.6.0`.

Unit behavior domain, acceptance numbers, and tuning records live in
`docs/models/glm52/fp8-blockwise-gemm-lab.md`; operating/maintenance notes in
`docs/models/glm52/kernel-lab-ops.md`.

## Quickstart

```bash
# build (run at the repo root; produces target/release/libglm52_kernel_lab.so)
PYTHONPATH=pegainfer-glm52/benches python3 -m kernel_lab build

# list (CPU-only, works without torch; doubles as the behavior-domain listing)
PYTHONPATH=pegainfer-glm52/benches python3 -m kernel_lab list

# check (torch-reference comparison, PASS/FAIL on the manifest tolerance)
PYTHONPATH=pegainfer-glm52/benches python3 -m kernel_lab check fp8_gemm.q_b --rows 64

# bench (time capacity shapes; --save writes into the baselines/<unit>.json
# ledger, bucketed by (shape, arch))
PYTHONPATH=pegainfer-glm52/benches python3 -m kernel_lab bench fp8_gemm.q_b --rows 16 --rows 64 --save

# bench --cold-l2 (quack-style cold-cache protocol: rotate N buffer clones so
# one sweep moves >= 3x L2; cures the warm-cache artifact of repeatedly
# hammering one buffer set while it sits in L2 and bandwidth numbers exceed
# the nominal peak. Mutually exclusive with --save — the ledgers are warm-
# protocol history and incomparable cold numbers would pollute the baseline.)
PYTHONPATH=pegainfer-glm52/benches python3 -m kernel_lab bench fp8_gemm_dsl_tc.q_b --rows 64 --cold-l2

# compare (delta vs the ledger by default; --baseline-so does same-session
# interleaved A/B against a saved .so)
PYTHONPATH=pegainfer-glm52/benches python3 -m kernel_lab compare fp8_gemm.q_b --rows 64
```

`--so PATH` — override for check/bench/compare; defaults to
`target/release/libglm52_kernel_lab.so`. Pair with `compare --baseline-so`
for a same-session A/B: stash the old .so before editing a kernel, then run
`--so new.so --baseline-so old.so` for interleaved per-round timing that
cancels clock/thermal drift.

Once installed into the repo `.venv`, the `kernel_lab` console script works
directly: `uv pip install -e pegainfer-glm52/benches`.

## Environment

- Tuning happens on Blackwell datacenter cards (GB300, sm_103). The ledger is
  arch-bucketed; cross-card numbers are an order-of-magnitude reference only.
- torch: reuse the repo `.venv`'s oracle-pinned version (2.11.0+cu130). The
  harness itself is stdlib + lazy torch — `list`, pytest, and the manifest
  chain work on a torch-less CPU box; `check`/`bench`/`compare` fail with a
  clear error when torch is missing.
- Iteration loop: edit the `.cu` (or the DSL kernel `.py`) -> `kernel_lab
  build` -> `check` -> `bench` -> `compare`; the top gate is a same-session
  `glm52_step_bench` A/B. Locked clocks are the bench owner's discipline
  (the harness only records `nvidia-smi clocks.sm`, never enforces).

## Tests

```bash
.venv/bin/python -m pytest pegainfer-glm52/benches/tests/ -q   # CPU-only, never imports torch
```

## Scope

This branch intentionally ships only the fp8 blockwise GEMM line above. Other
decode-op families (GEMV, indexer, norm, router, bookends, shared-expert,
quant, MLA) are out of scope here.
