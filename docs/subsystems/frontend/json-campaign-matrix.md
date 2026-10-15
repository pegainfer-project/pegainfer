# JSON campaign matrix expansion

> **TL;DR:** Calibration matrix axes are supplied by an explicit campaign JSON and expanded into immutable manifest cells; no reviewer or hardware-specific matrix is implicit in Python defaults. Implemented in commits `5ba09a58`, `efddc6b7`, and `73d14b59`.
>
> **Last touched:** 2026-09

## Preparation

- **Read**:
  - `docs/index.md` — routes simulator and calibration work under `subsystems/frontend/`.
  - `docs/subsystems/frontend/simulated-inference-engine.md` — confirms the simulator consumes concrete profile shapes rather than campaign syntax.
  - `docs/subsystems/frontend/sim-step-contract.md` — confirms the step-contract boundary is downstream of calibration artifacts.
  - `docs/playbooks/bench-vs-vllm.md` — confirms vllm-bench is the shared client and warns against conflating client sweep settings with per-cell evidence.
- **Relevant history**:
  - Existing calibration code has hardcoded matrix defaults in `scripts/sim-calibrate/vllm_bench_matrix.py` and a one-off builder outside version control.
  - `planner.py` and `manifest.py` currently accept expanded JSON cells/dimensions, but not a first-class matrix recipe.
- **Plan**:
  1. Remove implicit matrix defaults and require explicit axis/scalar configuration.
  2. Add JSON matrix expansion and provenance to planner/manifest while preserving concrete cell shape.
  3. Switch the formal campaign entrypoint to JSON-only configuration and keep one vllm-bench invocation per cell.
  4. Remove legacy hardcoded campaign references and verify old manifests remain readable.
- **Risks / open questions**:
  - Existing artifact roots must remain readable even though new campaign generation no longer supplies implicit reviewer axes.
  - Surface families need explicit completeness/divisibility validation before any benchmark process starts.

## Execution Log

### Step 1: Remove implicit matrix defaults

- Replaced the hardcoded axes and default scalar values in `vllm_bench_matrix.py` with explicit JSON axis/scalar inputs.
- Added inclusive `start/stop/step` and explicit `values` parsing, canonical sorting, required-value checks, and surface divisibility validation.
- Updated the existing matrix tests to pass explicit non-reviewer axes.
- Commit: `5ba09a58`.
- Result: targeted matrix and command tests passed.

### Step 2: Expand JSON matrices in the manifest

- Added `workload.matrices` expansion for cold prefill, steady decode, cached-prefill, and explicit mixed cells.
- Reused the axis parser for cached-prefix axes and recorded the normalized matrix recipe in the manifest workload identity.
- Kept concrete `workload.cells` compatibility for old manifests and test fixtures; the planner entrypoint now rejects specs without `workload.matrices`.
- Added one existing-suite smoke test for cold/decode/mixed JSON expansion.
- Commit: `efddc6b7`.
- Result: matrix, manifest, and runner tests passed.

### Step 3: Make the formal entrypoint JSON-only

- Added the tracked `campaign.py` loader and made `planner.py` use it.
- Removed the obsolete `/tmp/build_calibration_specs.py` hardcoded builder after confirming it only encoded the old implicit matrix.
- Kept `vllm_bench.py` cell-driven; no upstream `.venv` binary sweep mode was introduced.
- Commit: `73d14b59` plus the documentation/cleanup commit below.
- Result: a temporary custom JSON dry-run expanded six cells successfully.

### Verification

- Targeted Python tests: passed.
- Full `scripts/sim-calibrate/tests` discovery: 39 passed; 3 pre-existing localhost HTTP tests could not open sockets in the sandbox (`PermissionError: Operation not permitted`).
- Manual negative checks: missing explicit cold required values and non-divisible surface axes both fail closed.

## Debrief

- **Outcome**: New calibration generation is JSON-matrix-only through `planner.py`; downstream runner/collector/fit/report still consume the unchanged concrete cell contract.
- **Important boundary**: mixed campaign JSON declares concrete shape and a logical workload file/generator. The actual custom JSONL remains a separate input passed to `vllm_bench_runner.py --workload-path`; it is not silently synthesized by the matrix expander.
- **Compatibility**: Existing manifests and direct `build_manifest(... workload.cells ...)` fixtures remain readable. No reviewer-specific preset JSON was added.
- **Follow-up**: A future change can add a checked-in mixed JSONL generator or runner-side logical asset resolution, but that is intentionally outside this range-parameterization change.
