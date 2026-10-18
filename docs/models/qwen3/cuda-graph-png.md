# Qwen3 decode CUDA Graph PNG export

> **TL;DR:** `--dump-graph-png PATH` now exports the live Qwen3 rank-0,
> batch-1 SplitKv decode graph during startup: an unfolded detailed `.dot` for
> LLM/script inspection and a 192-DPI Cairo PNG that folds repeated physical
> layers for human browsing. Qwen3-4B produced 507 kernel nodes and 506 edges;
> kernel naming sets the repository driver floor at CUDA Driver API 12.3.
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` — located the Qwen3 model, decode-attention, and kernel-report records.
  - `docs/models/qwen3/model-crate.md` — established that Qwen3 owns its decode graph and kernel plan rather than exposing model details through the server.
  - `docs/subsystems/kernels/kernel-op-reports.md` — established the existing distinction between semantic `KernelCall` reports and physical kernel execution.
  - `pegainfer-core/src/cuda_graph.rs` — confirmed `CudaGraphState` retains the source `CUgraph` after capture and is the correct graph-inspection boundary.
  - `pegainfer-qwen3/src/batch_decode.rs` and `batch_decode_buffers.rs` — confirmed graph identity is `(bucket, attention_path, stream cache)`; batch 1 deterministically selects SplitKv.
  - `pegainfer-qwen3/src/unified_forward.rs`, `weights.rs`, and `executor.rs` — confirmed single-GPU memory profiling captures a temporary graph, while TP startup pre-captures live graphs; the export must explicitly target the live batch-1 graph instead of relying on capture order.
  - `pegainfer-server/src/config.rs` and `main.rs` — located CLI validation and the server-to-`Qwen3LaunchOptions` handoff.
  - CUDA 13.3 headers and the main workspace's locked cudarc 0.19.7 bindings — confirmed availability of `cuGraphGetNodes`, `cuGraphGetEdges`, `cuGraphNodeGetType`, `cuGraphKernelNodeGetParams`, and `cuFuncGetName`.
- **Relevant history**:
  - `docs/models/qwen3/decode-attention.md` records that batch 1 uses the SplitKv path; choosing batch 1 is therefore a semantic choice, not merely a smaller launch shape.
  - The local `cuda_graph_demo*` experiment proved that CUDA kernel parameters plus Graphviz produce a readable physical DAG, while verbose handles and default attributes are noise.
- **Plan**:
  1. Add `--dump-graph-png PATH` to the server CLI, accept it only for Qwen3 with CUDA Graph enabled, and validate Graphviz availability before model loading.
  2. Thread the optional output path through `Qwen3LaunchOptions` to the Qwen3 executor startup boundary without changing normal serving behavior.
  3. Reuse Qwen3's live decode buffers to capture the rank-0, batch-1, SplitKv, full-SM graph during startup; TP reuses its existing sweep, while TP1 performs only this requested pre-capture.
  4. Inspect `CUgraph` directly through the CUDA Driver API and build one node/edge model. Emit a detailed `.dot` sidecar for LLM consumption (full demangled name, raw symbol, launch shape, node type, all edges), then render the requested PNG from a compact human label view of the same graph. Rust generates DOT; Graphviz owns layout/rendering.
  5. Build in release mode, launch `/data/models/Qwen3-4B` on the local RTX 5070 Ti with the new flag, inspect the real PNG, and verify the server reaches readiness without the flag changing inference behavior.
  6. Record actual node/edge counts and rendering pitfalls, then complete this document's execution log and debrief.
- **Risks / open questions**:
  - A 36-layer physical DAG may be too tall for convenient PNG browsing. This first slice preserves every node; repeated-layer folding is intentionally deferred until the real output demonstrates the need.
  - `dot` and C++ demangling are diagnostic-tool dependencies only. Missing tools must fail before the expensive model load rather than silently emit a degraded graph.
  - Existing untracked `cuda_graph_demo*` files belong to the ongoing visual experiment and will not be overwritten or removed by this task.

## Execution Log

### Step 1: Review gate and output contract

- User approved the batch-1/rank-0 scope.
- Added the explicit two-audience contract: detailed `.dot` for LLMs and compact `.png` for humans, both sharing the same graph-node IDs.
- Result: approved; implementation started.

### Step 2: CLI, live-graph plumbing, and CUDA graph inspection

- Added `--dump-graph-png PATH` to `pegainfer-server`; model applicability and CUDA-Graph/LoRA exclusions fail before engine loading.
- Added `Qwen3LaunchOptions::dump_graph_png` and routed it through the Qwen3 scheduler to a rank-0 worker command.
- Single GPU pre-captures only the requested live batch-1 graph; TP reuses the completed startup sweep and reads rank 0's existing batch-1 graph.
- Added direct `CUgraph` inspection in `pegainfer-core` via `cuGraphGetNodes`, `cuGraphGetEdges`, `cuGraphNodeGetType`, `cuGraphKernelNodeGetParams_v2`, and `cuFuncGetName`.
- Detailed `.dot` retains full demangled/raw names and launch shapes; the PNG uses compact labels from the same node IDs and is rendered by external Graphviz.
- Commands: `cargo fmt --all`; `git diff --check`; `cargo check --release -p pegainfer-core -p pegainfer-qwen3 -p pegainfer-server`.
- Result: release check passed on sm_120; no new dependency and no Python path.

### Step 3: Real Qwen3-4B export and repeated-layer folding

- Built `pegainfer-server` in release mode and launched:
  `LD_LIBRARY_PATH=/data/opt/nccl-2.30.4/lib target/release/pegainfer --model-path /data/models/Qwen3-4B --dump-graph-png qwen3_4b_decode.png --port 18080`.
- The live batch-1 SplitKv graph contained 507 kernel nodes and 506 edges. The server reached scheduler and HTTP readiness after the export.
- The first unfolded PNG was `470x32767`: Graphviz hit its practical raster-height ceiling, making the image unusable even though the DOT was correct.
- The physical graph is a linear chain with a 14-kernel block repeated 36 times between a two-node prologue and the final LM-head GEMV. Added topology/signature-based repeated-run detection to fold only the PNG view; the detailed DOT remains all 507 nodes and 506 edges.
- Human labels identify cuBLAS GEMV directly instead of exposing the unhelpful demangled leaf `kernel<…>`.
- Result: the compact 96-DPI image became `693x1525` while preserving a representative 14-kernel layer and an explicit `×36` boundary.

### Step 4: Raster quality and final validation

- User review found the default Graphviz raster text blurry. Rendering now requires the Cairo PNG backend and uses 192 DPI.
- Final artifacts from the real model are `qwen3_4b_decode.dot` (319 KiB, 507 full raw/demangled symbols) and `qwen3_4b_decode.png` (336 KiB, `1386x3050`). Graphviz successfully reparsed the detailed DOT.
- Validation passed:
  - `cargo test --release -p pegainfer-core cuda_graph::dump::tests` — 4 passed.
  - `cargo test --release -p pegainfer-server config::tests::qwen3_` — 4 passed.
  - `cargo test --release -p pegainfer-qwen3 --lib` — 67 passed.
  - `cargo clippy --release -p pegainfer-core -p pegainfer-qwen3 -p pegainfer-server -- -D warnings`.
  - `cargo fmt --all --check`; `git diff --check`; `dot -Tdot qwen3_4b_decode.dot -o /dev/null`.
- Error/resolution: the first strict Clippy run rejected implicit Rust borrows passed to CUDA FFI and a local constant placed after statements. Switched the CUDA output parameters to explicit raw pointers and moved the constant before statements; the rerun passed.

### Step 5: PR preparation and artifact cleanup

- Backed up the reviewed high-DPI image to `/tmp/qwen3_4b_decode.png`; after final validation its SHA-256 is `462cd42e58cd4f4b2cbe688f30d8c696b5f79bdda63b06c657d94a4b5651143c`.
- Removed all repository-root experiment sources, scripts, binaries, DOT files, and PNG files, including the Qwen3-4B output pair. Only implementation, tests, and documentation remain in the worktree.
- The starting branch name was unrelated (`perf/glm52-fuse-decode-feed-rope`) but contained no commits ahead of `origin/main`; the worktree was migrated to the dedicated `feat/qwen3-cuda-graph-dump` branch after refreshing the mainline.

### Step 6: Submission review fixes

- The required `toxic-reviewer` pass traced CLI → Qwen launch → scheduler → rank worker → captured graph → CUDA Driver API and requested changes before submission.
- `cuFuncGetName` requires CUDA Driver API 12.3. At the user's direction, raised the documented repository floor from R535/CUDA 12.2 to R545/CUDA 12.3 and added a pre-load `cuDriverGetVersion` diagnostic so older drivers return a clear error instead of panicking in cudarc's lazy symbol loader.
- Renamed `shared_mem_bytes`/`smem` to `dynamic_shared_mem_bytes`/`dynamic_smem`; `CUDA_KERNEL_NODE_PARAMS.sharedMemBytes` does not include statically allocated shared memory.
- Removed Qwen3 presentation metadata from `pegainfer-core`: the Qwen3 executor now supplies the graph title to the reusable renderer.
- Extended the existing TP2 CUDA Graph integration test to request an export when Graphviz Cairo and `c++filt` are available, then assert that the PNG and detailed DOT contain the expected data. External renderer absence is the only reason to disable export coverage; every validation/export error otherwise fails the test. The test compiled and skipped its GPU body on the single-GPU development host.
- Added the missing CLI test for rejecting graph export with LoRA.
- Rebuilt and reran the real single-GPU Qwen3-4B export into `/tmp`; the server reached HTTP readiness and the detailed DOT retained 507 nodes/506 edges with the corrected dynamic-shared-memory field.
- `cargo test --release --workspace --lib` first exposed the documented missing `PEGAINFER_NCCL_ROOT`; after rerunning with `/data/opt/nccl-2.30.4`, the all-model build completed and tests advanced through 493 `kvbm-logical` cases before the unrelated `pegainfer-comm-cuda-lib::test_gdr::gdr_copy_flag_GPU` failed because this host cannot create a GDRCopy handle. All Qwen3/core/server tests in this change's scope passed.

## Debrief

- **Outcome:** one CLI argument produces both audiences' views from the same captured graph. The `.dot` is the source of truth; the PNG is a derived, folded projection.
- **Design boundary:** CUDA exposes kernel symbols and launch configuration (`grid`, `block`, dynamic shared memory) but not tensor shapes or semantic model roles. Those belong in a later kernel-plan enrichment pass rather than being guessed during graph capture.
- **Operational behavior:** CUDA Driver API older than 12.3, missing Graphviz Cairo or `c++filt`, a non-PNG path, disabled CUDA Graph, or LoRA mode fails before model loading. Normal serving does not capture or render this diagnostic graph unless the flag is present.
- **Next action:** use the detailed DOT as the physical-execution input when a semantic Qwen3/GLM kernel-plan annotator is built; keep model tensor metadata separate and join it by stable execution order or explicit runtime annotations.
