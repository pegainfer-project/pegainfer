# Stop-Token Policy Contract

> **TL;DR:** Keep EOS policy, explicit request stop IDs, generated tokens, and
> the concrete stop cause separate from wire parsing through every migrated
> scheduler; preserve the trigger token and drop only speculative suffixes.
>
> **Last touched:** 2026-08

## Preparation

- **Read:**
  - `docs/index.md` — frontend has two contract generations; Qwen3 and the
    simulator use the step contract while the remaining model lines use the
    legacy event path.
  - `docs/subsystems/frontend/frontend-architecture.md` — migration must keep
    the legacy bridge compatible until each model has its own lifecycle tests.
  - `pegainfer-frontend/src/engine/stop.rs` — shared `StopPolicy` and
    `StopCause` implementation.
- **Plan:**
  1. Audit every scheduler's prefill, decode, speculative, and P/D terminal
     paths for policy propagation and trigger-token ordering.
  2. Add CPU contract coverage where a multi-token or handoff path can lose the
     trigger or miscount completion tokens.
  3. Run formatting, frontend/model checks, and GPU tests where the server
     toolchain supports them; record environmental blockers separately.

## Contract

`StopPolicy` has two independent inputs:

- `eos`: `ModelDefault`, an explicit primary EOS ID, or `Ignore`;
- `token_ids`: explicit request stop IDs, active regardless of `eos`.

`StopPolicy::classify` gives EOS precedence when the same ID appears in both
sets. A scheduler emits the sampled token first, then emits `Finished` with:

- `StopCause::Eos(id)` for EOS stops (wire `stop_reason` remains absent);
- `StopCause::Token(id)` for explicit request stops (wire `stop_reason` is the
  actual token ID);
- `None` for length stops.

The completion count is incremented exactly once for every emitted token,
including the trigger. For speculative spans, only the prefix through the
first terminal token is committed; later candidates are discarded.

## Migration Matrix

| Model/path | Policy carried | Prefill | Decode/speculative | Terminal evidence |
| --- | --- | --- | --- | --- |
| Qwen3 step contract | yes | migrated | migrated, including speculative verify suffix truncation | `StopCause` |
| Qwen3.5 legacy scheduler | yes | migrated | migrated | `StopCause` |
| DeepSeek-V2-Lite | yes | migrated | migrated | `StopCause` |
| Kimi-K2 TP/DP | yes | migrated | migrated | `StopCause` |
| GLM5.2 | yes | migrated | decode, DSpark/MTP, and native P/D handoff migrated | `StopCause` / native handoff cause |
| K3 | yes | migrated | scheduler span truncation migrated; CUDA build gate pending | `StopCause` |
| Legacy bridge | compatibility | N/A | consumes typed cause when present; sentinel only for old `None` producers | typed cause or fallback |

The legacy bridge's `StopCause::None` sentinel branch is intentionally retained
until every old producer is migrated. It must never run when a real
`StopCause` is present.

`min_tokens` is a separate, still-unimplemented sampling contract: vLLM
requires it to mask EOS and explicit stop IDs until the requested completion
count, but the shared request types do not carry that threshold. Both bridge
generations therefore reject a non-zero value at the common wire-validation
boundary instead of letting legacy models silently ignore it. Implementing the
masking semantics is a follow-up that must add the field to the scheduler
contract and every sampler path.

## Execution Log

### Shared and model migration

- Added the shared `StopPolicy` / `StopCause` types and threaded them through
  the request, ledger, step, and event contracts.
- Updated wire conversion so `ignore_eos` does not disable explicit
  `stop_token_ids`.
- Updated Qwen3, Qwen3.5, DeepSeek-V2-Lite, Kimi-K2, GLM5.2, K3, and Qwen3
  speculative paths to emit the trigger token before terminal metadata.
- Added native GLM5.2 P/D serialization of the typed stop cause and preserved
  the anchor/cause distinction.
- Unified `min_tokens` handling at the shared vLLM wire boundary; legacy and
  stepped bridges now fail closed with the same error until sampler masking is
  implemented.

### K3 multi-token contract tests

- Extended `pegainfer-k3/src/scheduler/tests.rs` with a scripted
  `decode_many` fixture.
- Added tests for an explicit stop in the middle of a span, a length cap in the
  middle of a span, and EOS precedence over an overlapping explicit stop ID.
- The tests assert emitted IDs, suffix removal, `StopCause`, completion count,
  and slot release.

### Verification

- `cargo fmt --all` — pass (run from the Linux login shell).
- `git diff --check` — pass.
- `cargo test --release -p pegainfer-frontend --lib` — **77 passed, 0 failed**.
  This includes both legacy and stepped `min_tokens` rejection tests, typed
  explicit-stop mapping, EOS handling, and the shared wire-policy tests.
- `cargo test --release -p pegainfer-qwen3 --lib` — **93 passed, 0 failed**.
  This covers prefill/decode, speculative-span truncation, EOS/explicit-stop
  precedence, and request cleanup.
- `cargo test --release -p pegainfer-deepseek-v2-lite --lib` —
  **5 passed, 0 failed**.
- `cargo test --release -p pegainfer-sim --tests -- --test-threads=1` —
  **22 passed, 0 failed** (7 unit, 12 frontend HTTP, 3 tool-call round-trip).
  The serial test flag is required because several simulator tests bind a
  fixed localhost port.
- `cargo build --release -p pegainfer-server --bin pegainfer` — pass when the
  user-local `protoc-31.1` and CUDA 12.8 library paths are selected (see the
  command below). The default login environment links the obsolete system
  CUDA 10.1 libraries and is not a valid build environment for this tree.
- `cargo test --release -p pegainfer-k3 --lib scheduler::tests` — blocked before
  Rust test compilation by the server CUDA toolchain: the installed headers do
  not define `CUmemFabricHandle`, `CU_MEM_HANDLE_TYPE_FABRIC`, or
  `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED`; TileLang is also absent.
  This is a mainline K3 environment prerequisite, not a stop-policy compiler
  error.

### Qwen3 HTTP smoke

- Server: Qwen3-0.6B at `/home/ricardo.zheng/models/Qwen3/Qwen3-0.6B`, served
  as `qwen3-0.6b` on `127.0.0.1:18080`; `/v1/models` returned HTTP 200 and the
  expected model metadata.
- `ignore_eos=true`, `max_tokens=8`: HTTP 200, `finish_reason=length`,
  `stop_reason=null`, `completion_tokens=8`. This proves ignored EOS does not
  terminate the request early.
- `stop_token_ids=[0..151935]`, `ignore_eos=true`: HTTP 200,
  `finish_reason=stop`, `stop_reason=12095`, `completion_tokens=1`. The first
  generated token was preserved and reported as the actual explicit stop ID.
- `min_tokens=1`: HTTP 500 with the standard OpenAI error envelope. The server
  log contains the detailed rejection, and the request is rejected before
  scheduler submission by the shared wire validator.

## Remaining Risks

- K3's real DSpark executor performs speculative KV work before the scheduler
  applies request stop policy. Terminal slot release resets that state, so no
  suffix reaches the next request, but this is extra work rather than early
  executor-side truncation.
- The legacy sentinel fallback must be removed only after each remaining
  producer has a resolver and real HTTP lifecycle gate.
- A full GPU K3 test should be rerun after the server upgrades CUDA headers and
  installs TileLang.

## Next Step

HTTP smoke is complete. Keep the legacy sentinel fallback as a separate
migration after maintainer feedback; it is not required for the typed-cause
paths covered here.

## Debrief

- **Outcome:** The shared stop-policy migration is implemented and verified at
  unit, model-crate, simulator HTTP, and real Qwen3 HTTP levels. EOS, explicit
  stop IDs, trigger-token preservation, completion counts, speculative suffix
  truncation, and fail-closed `min_tokens` behavior are covered.
- **Environment caveats:** The default login shell selects obsolete CUDA 10.1
  libraries and `protoc 3.6.1`; the successful build used the user-local
  `protoc-31.1` and CUDA 12.8 paths recorded below. K3 GPU validation remains
  blocked by missing CUDA fabric headers and TileLang.
- **Follow-up:** Before opening a PR, perform the final diff review, decide
  which local-only `.codex` artifacts stay untracked, then stage only the
  intended source and documentation files. Do not remove the legacy sentinel
  fallback in this change.

Linux build environment used for the successful checks:

```bash
export PROTOC=/database/ricardo.zheng/.local/opt/protoc-31.1/bin/protoc
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu
export RUSTFLAGS=-Lnative=/usr/local/cuda-12.8/lib64
```
