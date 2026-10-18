# Build Script Submodule Initialization

> **TL;DR:** `pegainfer-kernels/build.rs` now initializes missing git submodules automatically for first-time builds before checking vendored third-party kernel headers.
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` - routed the task to the kernels subsystem because `pegainfer-kernels/build.rs` owns CUDA and third-party kernel build setup.
  - `docs/subsystems/kernels/pegainfer-kernels-boundary.md` - confirmed the kernels crate owns the MoE/MLA third-party substrate boundary and already documents DeepEP/DeepGEMM/FlashMLA submodule checks.
  - `docs/playbooks/developer-onboarding.md` - confirmed the default build is pure Rust + CUDA and feature-gated builds pull in additional kernel dependencies at build time.
  - `pegainfer-kernels/build.rs` - found the existing `require_moe_submodules` and `require_glm52_submodules` checks that currently panic with a manual `git submodule update --init --recursive ...` instruction.
  - `.gitmodules` - confirmed the top-level submodules are `flashinfer`, `DeepEP`, `FlashMLA`, and `DeepGEMM`.
- **Relevant history**:
  - `docs/subsystems/kernels/pegainfer-kernels-boundary.md` records that `moe` and `glm52` features depend on vendored third-party substrate, so build-time initialization should stay in the kernels crate rather than model crates.
- **Plan**:
  1. Create an isolated worktree under `/home/ziyang` on branch `fix/build-rs-submodule-init`.
  2. Use `gh issue create` to open an issue describing the missing automatic submodule initialization.
  3. Add a small `pegainfer-kernels/build.rs` helper that runs `git submodule update --init --recursive` from the workspace root before submodule-dependent checks and include clear diagnostics if git is unavailable or fails.
  4. Verify with formatting and a focused Rust build-script compile/test path that does not require a full CUDA build when possible.
  5. Commit using Commitizen format, push the branch, and use `gh pr create` to open a PR linked to the issue.
- **Risks / open questions**:
  - Cargo build scripts should avoid unnecessary network work on every build; the helper should only run when the repository is a git checkout and submodule marker files are missing.
  - Full `cargo build --release` may be too expensive or blocked by local CUDA/model prerequisites, so verification may need to focus on `cargo fmt`, targeted Rust checks, and a direct missing-submodule behavior probe.

## Execution Log

### Step 1: Create isolated worktree and task doc
- Created `/home/ziyang/pegainfer-buildrs-submodule-init` on branch `fix/build-rs-submodule-init`, tracking `origin/main`.
- Observed the new worktree starts with uninitialized top-level submodules (`git submodule status --recursive` shows leading `-` entries).
- Added this task document and index entry to keep the PR context with the kernels subsystem.
- Result: success.

### Step 2: Open the tracking issue
- Ran `gh issue create` for the first-time-build failure mode, but shell command substitution in a double-quoted markdown body accidentally executed the backtick-wrapped command examples first. That initialized the worktree submodules but did not modify tracked source.
- Retried enough to get the authoritative GitHub error: `GraphQL: Resource not accessible by personal access token (createIssue)`.
- Confirmed with `gh repo view openinfer-project/openinfer --json viewerPermission,hasIssuesEnabled,isArchived,nameWithOwner` that issues are enabled but the current account has `viewerPermission: READ`.
- Retried with a single-quoted body that explicitly described the new-user first-build failure. It failed with the same `createIssue` permission error.
- Added `code` remote as `https://github.com/Ma1oneZhang/openinfer` and confirmed it is a fork of `openinfer-project/openinfer` with `viewerPermission: ADMIN`.
- Tried `gh issue create --repo Ma1oneZhang/openinfer`; it failed because the fork has issues disabled.
- Enabled issues on the fork with `gh repo edit Ma1oneZhang/openinfer --enable-issues`.
- Created fork tracking issue `Ma1oneZhang/openinfer#1`: <https://github.com/Ma1oneZhang/openinfer/issues/1>.
- Result: upstream issue creation is blocked by GitHub token/repository permission, but the fork issue exists.

### Step 3: Implement automatic initialization
- Modified `pegainfer-kernels/build.rs` with `ensure_git_submodules_initialized`.
- The helper runs only in a git checkout with `.gitmodules`, inspects `git submodule status --recursive`, and runs `git submodule update --init --recursive` only when at least one status line starts with `-`.
- The build script calls the helper at the start of `main`, before CUDA feature generators or vendored header checks can fail.
- Kept explicit diagnostics for git inspection/update failures so users still get a clear manual command when git is missing or the update fails.
- Result: implemented.

### Step 4: Verify
- Ran `cargo fmt --check` successfully.
- Ran `PEGAINFER_CUDA_SM=80 PEGAINFER_NVCC_JOBS=1 cargo check -p pegainfer-kernels --no-default-features` successfully. This executed the build script with initialized submodules and completed the default CUDA kernel build path in `1m 47s`.
- Result: success.

### Step 5: Push branch and open PR
- Committed `fix(kernels): initialize submodules during build`.
- Pushed branch `fix/build-rs-submodule-init` to `code` remote (`https://github.com/Ma1oneZhang/openinfer`).
- Attempted upstream PR creation with `gh pr create --repo openinfer-project/openinfer --head Ma1oneZhang:fix/build-rs-submodule-init --base main`; GitHub returned `GraphQL: Resource not accessible by personal access token (createPullRequest)`.
- Created fork PR `Ma1oneZhang/openinfer#2`: <https://github.com/Ma1oneZhang/openinfer/pull/2>.
- Result: fork issue and fork PR exist; upstream issue/PR creation remains blocked by the current token's upstream repository permissions.

## Debrief

- **Outcome**: `pegainfer-kernels/build.rs` now auto-initializes missing git submodules for new-user/fresh-worktree first builds, but skips the update when all submodules are already initialized. The change is committed on branch `fix/build-rs-submodule-init`, pushed to `code`, and covered by fork issue/PR links.
- **Pitfalls encountered**:
  - Backtick-wrapped markdown in a double-quoted `gh issue create --body` was interpreted by the shell. Use single quotes or body files for GitHub CLI markdown bodies.
  - The current GitHub token can read `openinfer-project/openinfer` but cannot create upstream issues or PRs (`createIssue` / `createPullRequest` GraphQL errors). The fork has admin permission, so issue/PR creation succeeded there after enabling fork issues.
- **Lessons learned**:
  - For first-build dependency setup, the build script should check `git submodule status --recursive` first and only run the networked update when a line starts with `-`.
- **Follow-ups**:
  - To open the intended upstream PR with `gh`, grant the current token Pull requests write permission on `openinfer-project/openinfer` or re-authenticate `gh` with a token that has that permission.
