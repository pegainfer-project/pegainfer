# GLM5.2 decode CUDA Graph PNG export

> **TL;DR:** `--dump-graph-png PATH` now exports GLM5.2 rank 0's live,
> pre-captured whole-step graph: EP8 and TP4 use bucket 1; TP8 uses its fixed
> bucket 8. On 8×H200, DSpark-off exports contain 3,144/2,986 nodes and
> 399/21 PDL edges respectively, with a complete DOT and readable folded PNG.
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` — located the existing Qwen3 graph exporter and the GLM5.2 whole-step graph and TP8 topology records.
  - `docs/models/qwen3/cuda-graph-png.md` — established the existing CLI contract: early dependency validation, one complete detailed DOT, and one folded 192-DPI Cairo PNG from the same live graph.
  - `docs/models/glm52/whole-step-decode-graph.md` — established that every GLM5.2 per-rank whole-step graph covers embed through device argmax, is pre-captured before serving, and currently contains about 2,000 nodes.
  - `pegainfer-core/src/cuda_graph.rs` and `pegainfer-core/src/cuda_graph/dump.rs` — confirmed `CudaGraphState` retains the source `CUgraph`, already exposes the model-agnostic inspector/renderer, and fails if asked to dump an uncaptured graph.
  - `pegainfer-glm52/src/lib.rs` — confirmed GLM5.2 launches eight physical workers in either the default DP8/EP8 topology or the mirrored TP8 MoE topology, and identified the launch boundary where dump dependencies can fail before weight loading.
  - `pegainfer-glm52/src/model/mod.rs` — confirmed each decode bucket owns its own persistent `CudaGraphState`; EP8 has buckets 1/2/4/8, while TP8 serves only the fixed bucket-8 shape.
  - `pegainfer-glm52/src/scheduler/mod.rs` — confirmed all required bucket graphs are pre-captured across all ranks before the coordinator accepts requests, providing a safe point to export rank 0 without running an extra model step.
  - `pegainfer-glm52/src/runner.rs` — located the rank-local command boundary needed to inspect the graph on the CUDA-context-owning worker thread.
  - `pegainfer-server/src/config.rs` and `pegainfer-server/src/main.rs` — located model-specific CLI applicability, validation tests, and the server-to-`Glm52LaunchOptions` handoff.
- **Relevant history**:
  - `docs/models/qwen3/cuda-graph-png.md` records that an unfolded 507-node graph exceeded Graphviz's practical PNG height; GLM5.2's roughly 2,000-node branched graph must be judged from a real render rather than assumed readable.
  - `docs/models/glm52/whole-step-decode-graph.md` records fork/join event nodes and collective kernels inside the graph, so Qwen3's linear repeated-run folding may not recognize the 78-layer GLM5.2 body.
- **Plan**:
  1. Generalize the server help/applicability tests and thread the optional PNG path through `Glm52LaunchOptions`, validating the `.png` path, CUDA driver, Graphviz Cairo renderer, and demangler before loading the checkpoint.
  2. Add one rank-local graph-dump command and a narrow `Glm52RankModel` accessor. After the existing all-rank pre-capture completes, export rank 0 bucket 1 for EP8 or rank 0 bucket 8 for TP8; normal serving and graph capture stay unchanged when the flag is absent.
  3. Reuse the shared complete-DOT exporter. Render real EP8 and TP8 graphs on 8×H200; if the PNG is not readable, extend the shared renderer to fold repeated branched layer subgraphs while preserving every physical node and edge in the DOT sidecar.
  4. Add CPU-side CLI/model-contract coverage, then run release formatting, checks, tests, and Clippy for `pegainfer-core`, `pegainfer-glm52`, and `pegainfer-server` with the GLM5.2 feature.
  5. On 8×H200, launch both EP8 and TP8 with the flag, verify startup reaches readiness, validate the detailed DOT with Graphviz, inspect PNG dimensions/content, and run one request on each topology. Record sanitized node/edge/kernel counts and artifacts without internal hostnames or private paths.
  6. Measure dump-disabled versus dump-enabled startup/serving behavior before the required `toxic-reviewer` pass. Address every correctness or performance objection until the change is ready for review, then complete the execution log and debrief.
- **Risks / open questions**:
  - The shared renderer currently folds only a repeated linear run. GLM5.2 has layer-internal branches, so readable PNG output may require a topology-aware repeated-DAG folding algorithm.
  - Export must run on the rank worker that owns the CUDA context; moving the raw graph handle to the coordinator would violate the existing ownership boundary.
  - TP8 has no bucket-1 graph. Calling its bucket-8 shape "batch 1" would be misleading even though it represents one mirrored logical rank, so the artifact title and log must state both topology and physical bucket.

## Execution Log

### Step 1: Review gate and execution target

- User approved execution and authorized validation on an 8×H200 development host.
- Internal host aliases and private paths remain outside repository documentation and future GitHub text.
- Result: approved; implementation started.

### Step 2: First live EP8 export and graph-inspection fixes

- The first 8×H200 EP8 export reached serving readiness and produced a valid
  detailed graph with 3,149 nodes, 2,933 kernels, and 3,339 edges.
- The graph contains 399 programmatic-dependent-launch edges. The legacy edge
  query rejected this graph with `CUDA_ERROR_LOSSY_QUERY`, so inspection now
  uses the CUDA 12.3 edge-v2 API and records ports plus dependency type in the
  detailed DOT. The human view distinguishes non-default dependencies.
- Feeding roughly 3,000 symbols to `c++filt` exposed a pipe deadlock: the
  parent filled stdin while the child filled stdout. Input now runs on a
  writer thread while the parent drains output, with a stream larger than one
  pipe covered by a regression test.
- Exact repeated-DAG folding reduced the physical graph to six repeated blocks,
  but the first PNG still reached Graphviz's 32,767-pixel raster-height limit.
  Inspection showed five DSpark hidden-state copies in a launch that did not
  request a drafter. The user clarified the runtime contract: without
  `--dspark-path`, the live graph itself must omit DSpark work rather than the
  PNG hiding it.

### Step 3: Make DSpark capture launch-configured

- The launch-time DSpark decision now reaches rank-model construction.
  Per-bucket capture storage is optional: ordinary decode allocates no DSpark
  capture row and submits no capture copies; DSpark launches retain both.
- Attempting to read captured context from a model built without DSpark fails
  explicitly instead of returning an unrelated buffer.
- Local release validation: GLM5.2/server feature check passed; GLM5.2 library
  tests passed 57/57 with 14 hardware tests intentionally ignored.
- Live 8×H200 EP8 validation passed without a drafter: 3,144 nodes, 2,928
  kernels, 3,334 edges, zero `copy_hidden_rows_kernel` nodes, and all 399
  programmatic dependencies retained. The detailed DOT reparsed successfully,
  an HTTP completion returned 200, and the folded PNG shrank from the raster
  ceiling to 2,611×11,538. Removing the five unused capture copies restored a
  162-node four-layer macroblock repeated 17 times.

### Step 4: TP8 live export

- User authorized the previously deferred TP8 run. Validation uses rank 0's
  fixed physical bucket-8 graph and keeps DSpark disabled.
- Live 8×H200 validation passed: 2,986 nodes, 2,770 kernels, 3,026
  edges, zero `copy_hidden_rows_kernel` nodes, and 21 programmatic
  dependencies. The detailed DOT reparsed successfully, an HTTP completion
  returned 200, and the PNG is 3,045×11,911.
- The title identifies `MoE TP8`, `mirrored`, and physical bucket 8; it does
  not call this a batch-1 graph. Export took about 1.69 seconds after
  pre-capture; total TP8 startup was 167.5 seconds, dominated by the known
  all-layer expert-slice load.
- Result: passed; the service was stopped and all GPU processes released after
  copying the artifact.

### Step 5: Renderer correctness review

- The required post-measurement review accepted CUDA-context ownership,
  pre-capture/export teardown, PDL metadata, DSpark gating, and topology/bucket
  selection, then found three renderer defects before handoff.
- Repeated blocks now require a real, identical dependency boundary between
  adjacent copies. Identical but disconnected parallel subgraphs are not
  presented as a sequential `×N` block.
- Candidate selection now preserves the primitive period instead of rewarding
  a larger opaque bundle. A 14-kernel body repeated 36 times remains `14×36`,
  preserving the established Qwen3 output contract.
- Graphviz and `c++filt` share one subprocess communication path that writes
  stdin while draining stdout/stderr. A test moves more than one pipe of data
  in both output streams to prevent the startup deadlock from returning.
- Human rendering moved into its own module, GLM5.2 graph pre-capture/export
  moved out of the scheduler body, and the captured layer-step body moved out
  of `model/mod.rs`. All touched files are below the project's 1,000-line
  ceiling.
- Post-fix 8×H200 exports preserved the measured graph structure and image
  dimensions: EP8 remained 3,144 nodes/3,334 edges/399 PDL at 2,611×11,538;
  TP8 remained 2,986 nodes/3,026 edges/21 PDL at 3,045×11,911. Both detailed
  DOT files reparsed with Graphviz, both one-token requests returned HTTP 200,
  and both graphs still contained zero DSpark copy kernels.
- Release validation passed for `pegainfer-core` (27/27, including all 12 graph
  tests), GLM5.2 library tests (57/57, 14 hardware tests ignored), server
  graph-CLI tests (4/4), the GLM5.2/server feature check, formatting, diff
  whitespace, and strict
  `pegainfer-core` Clippy. Strict GLM5.2 Clippy remains blocked by seven
  unrelated existing lints outside this change.

### Step 5b: TP4 live export after the TP4-serving rebase

- Rebasing onto the merged GLM5.2 TP4/GB300 serving path (`#637`) re-based the
  extracted step body and scheduler graph module on the generalized
  tensor-replicated runtime: the export bucket now keys on the full-bucket
  contract rather than mirroring (EP8 and TP4 have a true bucket-1 graph; TP8
  keeps its fixed bucket 8), and the exported step body carries the TP4
  vocabulary-parallel greedy tail.
- 4×GB300 TP4 live export: bucket-1 graph at 2,547 nodes / 2,334 kernels /
  2,587 edges, complete DOT reparsed with Graphviz, 2,171×17,098 PNG, and the
  greedy HTTP smoke unchanged. The DSpark-off graph contains the 75 TP4
  MoE eight-row bridge copies (one per MoE layer — a legitimate serving use of
  the same copy kernel) and zero DSpark aux-capture copies.

### Step 6: Review handoff

- Three post-measurement review passes reached Ready for Review after fixing
  renderer semantics, subprocess communication, Qwen period compatibility,
  and every file-size violation introduced or exposed by this work.
- The publication scope contains implementation, tests, and living docs. The
  two rendered PNGs remain untracked local artifacts because they are for
  visual inspection rather than repository assets.

## Performance Work

No optimization claim is made. A same-build, same-cap EP8 A/B on 8×H200 makes
the opt-in diagnostic cost explicit:

| metric | no dump | `--dump-graph-png` | delta |
|---|---:|---:|---:|
| `Engine loaded` | 73.207 s | 75.841 s | +2.634 s |
| identical one-token request, server wall | 20.423 ms | 20.442 ms | +0.019 ms |

The enabled run spent 1.139 seconds inspecting/folding/rendering after graph
pre-capture. The remaining startup delta is primarily the enabled path waiting
for pre-capture before publishing the engine; without a dump, pre-capture can
finish behind frontend startup. The request delta is noise-sized, consistent
with export having no serving-time branch.

## Debrief

The opt-in export now follows the same caller contract for Qwen3 and GLM5.2,
while model-specific code selects the only graph that the chosen topology can
actually serve. CUDA inspection remains on the rank worker that owns the
context; the coordinator only requests export after the all-rank pre-capture
barrier and reports success or tears every worker down on failure.

The live graph, not merely its rendering, follows launch configuration:
without `--dspark-path`, no capture buffer is allocated and no hidden-state
copy is submitted. The detailed DOT is the lossless artifact, including PDL
ports and dependency types. The PNG is a verified projection: it folds only
topologically identical, sequential repeated DAGs and preserves primitive
periods such as Qwen3's 14-kernel layer.

The measured cost is diagnostic startup work, not a decode optimization:
same-build EP8 startup increased by 2.634 seconds, while the observed
one-token serving delta was 0.019 ms. The two user-facing PNGs remain local
artifacts and must not be included in a code commit.

Next action: review the published pull request; there is no known
implementation or validation blocker.
