# DFlash2 Phase 1 Selector

> **TL;DR:** Issue #930 Phase 1 adds a bounded, deterministic top-16 candidate selector on top of the existing Qwen3 DFlash backbone; dynamic convolution, sliding-window execution, and sampled rejection are deliberately out of scope.
>
> **Last touched:** 2026-09

## Preparation

- **Read**:
  - `docs/index.md` - routes Qwen3 model and kernel design records.
  - `docs/models/qwen3/dflash-speculative-decoding.md` - defines the existing proposer/verify/KV transaction contract and the batch layout.
  - `docs/models/qwen3/dspark-integration.md` - documents the legacy anchor-first/Markov path that must remain unchanged.
  - `docs/models/qwen3/kernels-crate.md` - assigns CUDA primitives and FFI ownership to `pegainfer-kernels`.
  - `docs/models/qwen3/model-crate.md` - documents Qwen3 model-crate boundaries and single-GPU speculative decoding.
  - `docs/conventions/coding-style.md` - requires focused tests and project logging conventions.
  - `CLAUDE.md` - defines build, branch, and AI-assisted contribution requirements.
- **Relevant history**:
  - `docs/models/qwen3/dflash-speculative-decoding.md` - the existing DFlash lane owns proposal while the shared verify and KV transaction contracts stay method-agnostic.
  - No prior DFlash2 Phase 1 task record exists in this checkout.
- **Plan**:
  1. Audit the current configuration and loader scaffold; keep legacy DFlash and DSpark behavior unchanged, load an independent native output head when the checkpoint declares untied embeddings, and reject hybrid capabilities that Phase 1 cannot execute.
  2. Load and validate the selector projection/codebooks, add a fixed-size GPU selector primitive and Rust wrapper, and account for its persistent and scratch allocations.
  3. Dispatch `TopKSelector` from the DFlash draft lane without changing draft span, verify, KV transaction, or CUDA-Graph shapes.
  4. Run formatting, compile, focused selector/reference checks, GPU-vs-reference checks, and legacy DFlash/DSpark regression checks; record actual results and limitations.
- **Risks / open questions**:
  - The only discovered DFlash2 checkpoint also declares Phase 2 convolution and sliding-window capabilities; it must fail closed until those execution paths exist.
  - Selector tie-breaking, anchor mapping, request-major row offsets, and scratch reservation must be deterministic and shape-safe.

## Execution Log

### Step 1: Normalize the DFlash2 capability contract

- Added a `DFlashProposal::TopKSelector` capability and an explicit
  `DFlashLayout` in `pegainfer-qwen3/src/config.rs`.
- Legacy DFlash and DSpark schemas remain on their existing proposal paths.
- Native DFlash2 configurations parse their root or nested
  `tie_word_embeddings` field into a head-source contract. Legacy and tied
  checkpoints reuse the verifier embedding/output projection; untied native
  checkpoints load only their separate `lm_head.weight`.
- Phase 2 convolution, sliding-window attention, and anchor-first selector
  layouts still fail closed before GPU weight allocation.

### Step 2: Load selector weights and wire the proposer

- Added SafeTensors manifest checks for the hidden projection and predecessor /
  successor codebooks.
- Added persistent selector scratch and a two-launch CUDA implementation:
  deterministic top-16 candidate extraction followed by a request-local path
  walk using the predecessor/successor codebooks.
- Kept the existing full-block draft result contract, verify span, KV updates,
  and CUDA-Graph shapes unchanged.
- The native verifier embedding is intentionally reused instead of loading a
  duplicate `embed_tokens.weight`; the downloaded Qwen3-4B DFlash2 checkpoint's
  embedding bytes match the verifier exactly, while its `lm_head.weight` is
  loaded when the schema is untied.

### Step 3: Fix anchor-drop row mapping

- The DFlash backbone emits an anchor-inclusive block. For the current
  anchor-drop layout, row 0 is discarded by the executor and rows 1..N-1 are
  the real proposal positions.
- The selector now uses compact candidate/output rows for those real positions,
  while reading the corresponding rows from the original anchor-inclusive
  logits/hidden buffers. Every request-local walk starts from the verified
  anchor token, so no draft depends on a candidate from the discarded row 0.
- The host wrapper reconstructs `[anchor, selected_1, ..., selected_N-1]` for
  the unchanged executor contract and rejects an invalid GPU token id before
  it can reach token lookup.

### Step 4: Lightweight verification

Commands were run in the Linux feature checkout
`/database/ricardo.zheng/projects/open-access/pegainfer`
with `/usr/bin` present in `PATH` (the build script invokes `git`):

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | Passed |
| `git diff --check` | Passed |
| `cargo check --release -p pegainfer-qwen3 --tests` | Passed; CUDA `sm_89` build |
| `cargo test --release -p pegainfer-qwen3 --lib` | 88 passed, 0 failed |
| `cargo test --release -p pegainfer-build --lib` | 8 passed, 0 failed |

The native selector gate was also run with the real Qwen3-4B DFlash2 tensors
and a selector-only config overlay. The overlay removes only the checkpoint's
Phase 2 convolution/sliding-window declarations; it does not replace selector,
backbone, embedding, or output-head weights:

```bash
export PATH=/home/ricardo.zheng/.cargo/bin:/usr/local/cuda/bin:/usr/bin:/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export PEGAINFER_CUDA_SM=89
PEGAINFER_TEST_MODEL_PATH=/database/ricardo.zheng/models/Qwen3/Qwen3-4B \
PEGAINFER_DFLASH2_TEST_MODEL_PATH=/tmp/dflash2-phase1-native-overlay \
cargo test --release -p pegainfer-qwen3 \
  --test dflash_speculative_gate \
  dflash2_native_selector_untied_head_greedy_gate \
  -- --ignored --nocapture --test-threads=1
```

Result: `1 passed, 0 failed`; the selector-only native launch dispatched the
CUDA top-k/path-walk path and matched the plain Qwen3 greedy continuation.

### Step 5: Remove redundant scaffolding

- Kept the selector tensor preflight because the shared loader does not check
  SafeTensors dtype or malformed rank; removed its unused `SelectorManifest`
  wrapper and duplicate positive-value checks.
- Removed the unused selector scratch accessor and ABI-only bf16 assertion.
- Shortened comments to the anchor mapping, two-launch dependency, and
  unsupported-capability boundaries.
- Re-ran formatting, Qwen3/build tests, and the server release build.

| Cleanup verification | Result |
| --- | --- |
| `cargo fmt --all -- --check` | Passed |
| `git diff --check` | Passed |
| `cargo check --release -p pegainfer-qwen3 --tests` | Passed |
| `cargo test --release -p pegainfer-qwen3 --lib` | 88 passed, 0 failed |
| `cargo test --release -p pegainfer-build --lib` | 8 passed, 0 failed |
| `cargo build --release -p pegainfer-server --bin pegainfer` | Passed |

### Step 6: Address review feedback

- Removed the native-schema-wide head rejection. The loader now distinguishes
  verifier-owned tied heads from an untied native `lm_head.weight`, so a
  selector-only DFlash2 checkpoint can reach the selector path.
- Added an ignored GPU gate that imports a native untied checkpoint, runs the
  selector CUDA launches with real weights, and checks greedy losslessness.
  The public Qwen3-4B DFlash2 artifact is currently Phase 2-capable, so the
  gate uses a config-only overlay while preserving every model tensor.

## Debrief

- **Outcome:** Phase 1 selector wiring, anchor-drop mapping, native untied-head
  loading, and a focused cleanup of redundant scaffolding are complete in the
  feature branch. The checkout is based on upstream main; no changes are
  staged or committed.
- **Pitfalls encountered:** The first verification command omitted system
  directories from `PATH`, so `pegainfer-kernels/build.rs` could not spawn
  `git`. Re-running with `/usr/bin:/bin` succeeded. The row mapping bug was a
  real semantic issue that compilation alone could not detect.
- **Lessons learned:** Selector buffers must distinguish the input block shape
  from the compact set of positions actually proposed. The anchor is a
  request-level predecessor, not a selector candidate when the executor drops
  row 0.
- **Follow-ups:** Run the full native checkpoint import only after Phase 2
  convolution and sliding-window execution land, because the public checkpoint
  intentionally advertises those capabilities and Phase 1 rejects them. Phase 3
  remains responsible for sampled losslessness/rejection sampling.
