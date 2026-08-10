# Audit pegainfer-bench consumers with Hawk

> **TL;DR:** The all-features Hawk sweep retained `pegainfer-bench` (Qwen3 and Kimi-K2 reports are real consumers), removed 215 findings from the workspace result (576 → 361), and deleted the dead state exposed by the visibility reductions; 332 remaining findings belong to the deliberately excluded `kvbm-logical` fork.
>
> **Last touched:** 2026-08

## Preparation

- **Read**:
  - `docs/index.md` — routes visibility work through the Hawk playbook and report-tooling history through the kernels subsystem.
  - `docs/playbooks/hawk-visibility-audit.md` — requires the all-features profile and warns that undeclared plain binaries are invisible to Hawk.
  - `docs/subsystems/kernels/kernel-op-reports.md` — records that Qwen3 and Kimi-K2 model reports intentionally share `pegainfer-bench`.
  - `docs/conventions/coding-style.md` — requires tests to protect meaningful behavior rather than implementation ceremony.
- **Relevant history**:
  - `docs/subsystems/kernels/kernel-op-reports.md` — `pegainfer-bench` was extracted only after a second model-report consumer existed; model-specific regression machinery deliberately stayed outside it.
- **Plan**:
  1. Run the documented all-features Hawk audit and isolate findings involving `pegainfer-bench` and its Qwen3/Kimi-K2 consumers.
  2. Compare Hawk reachability with Cargo target declarations and direct Rust references; fix only confirmed configuration or dead-code drift.
  3. Run focused compile checks plus the repository warning gates relevant to changed targets.
  4. Record the result here, commit with a Commitizen message, push the branch, and open an English draft PR.
- **Risks / open questions**:
  - `hawk.toml` currently declares only the serving binary, while `pegainfer-bench` is consumed by feature-gated report binaries; Hawk may therefore report real tooling APIs as dead unless those shipped binaries are declared.

## Execution Log

### 1. Establish the all-features baseline

- Ran the documented Hawk profile with all model features enabled.
- The first attempt proved that a relative `PEGAINFER_TRITON_PYTHON=.venv/bin/python` is resolved from the kernel crate and fails before Hawk analysis. Updated the playbook to pass `$PWD/.venv/bin/python` and to take the NCCL root from a caller-supplied variable.
- Baseline on the same `main` commit used by the final branch: 576 findings — 27 `dead_public`, 462 `unnecessary_public`, and 87 `unnecessary_restricted_visibility`.
- `cargo tree --workspace --all-features -i pegainfer-bench` and direct references confirmed two consumers: Qwen3 model reports and Kimi-K2 kernel/model reports. Hawk reported no finding in `pegainfer-bench` itself, so the crate stays.

### 2. Reduce visibility and delete exposed dead code

- Ran Hawk `--fix` with `kvbm-logical` excluded, matching the documented upstream-fork boundary.
- Applied the proven `pub` → crate/module-private reductions across model, frontend, KV, and kernel crates.
- Deleted state and wrappers that became genuinely unreachable after the reduction, including:
  - the unused frontend `ModelInfo` and DeepSeek-V2-Lite stats model-path copies;
  - the old KV-cache lifetime-only state and tests;
  - the unconnected KV-store router-event cursor and `RegisteredBlock` duplicate;
  - unused GLM5.2 scratch storage, router wrapper, MTP test accessors, and an unconsumed oracle probe table;
  - unused KV-offload re-exports and Qwen3.5 generic TP ping-command machinery.
- Test-only probes that still protect behavior are compiled only under `cfg(test)` rather than exported as production API.

### 3. Verify the result

- `cargo fmt --all --check` — passed after formatting the visibility reductions.
- `cargo check --release --workspace --all-features --all-targets` — passed with no Rust `dead_code` or `unused_*` warnings.
- CI CPU Clippy package set with `--all-targets -- -D warnings` — passed.
- CI Qwen3 CUDA Clippy package set, including `pegainfer-bench`, with `--all-targets -- -D warnings` — passed.
- `cargo test --release --workspace --lib` — passed after providing the local NCCL runtime library to the dynamic loader.
- Final all-features Hawk check after rebasing onto the latest `main`, using the corrected playbook command — passed. Result: 361 findings (26 dead / 321 public / 14 restricted), down 215. Of those, 332 are in the excluded `kvbm-logical` fork; `pegainfer-bench` remains at zero.
- A full all-features workspace Clippy was also attempted. It reaches pre-existing pedantic failures in unchanged GLM5.2 kernel/test code and other never-CI-gated model code; the repository playbook explicitly keeps that baseline outside a visibility PR. The two actual CI Clippy gates above are green.
- Opened draft PR [#858](https://github.com/pegainfer-project/pegainfer/pull/858) from `chore/hawk-pegainfer-bench-audit`.

## Debrief

- **Outcome**: Reduced the closed-world public surface, removed the dead state exposed by that reduction, retained `pegainfer-bench` on evidence of two model-report consumers, and repaired the Hawk command so future reruns reach analysis reliably.
- **Pitfalls encountered**:
  - Build-script working directories make a workspace-relative Python path unreliable even when the interpreter exists.
  - Visibility reduction activates rustc and Clippy lints that public items suppress; those findings must be separated from unrelated all-features pedantic baseline noise.
  - Workspace tests require the NCCL shared library to be visible to the runtime loader after link succeeds.
- **Lessons learned**:
  - Treat Hawk as a two-stage proof: reduce visibility first, then let ordinary `dead_code` identify state that can actually be deleted.
  - A Hawk finding is not evidence that a tooling crate is unused; Cargo inverse dependencies and direct target references remain the authority for report binaries.
