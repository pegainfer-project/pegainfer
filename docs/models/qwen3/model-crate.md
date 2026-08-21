# Qwen3-4B Model Crate

**Created**: 2026-05-03
**Last touched**: 2026-07
**TL;DR**: `crates/pegainfer-qwen3` now owns Qwen3 config, weights, execution, scheduler, tests, benches, and kernel plan. Root `pegainfer` loads Qwen3 through a generic `EngineHandle` and no longer contains `Qwen3Model`, `Qwen3Executor`, `ModelRuntimeConfig`, root Qwen3 tests, or `src/model/qwen3/*`. The old `ModelForward` path has been removed; decode length-limit now emits the final token before `Finished`. Long-context `bs=1` TPOT was traced to non-partition FlashInfer paged decode under-filling the GPU; Qwen3 runtime gates FlashInfer split-K decode on `padded_bs<=32` (the `seq_len>=1024` gate was dropped in #437) with a 64-token chunk floor (`Tuned` capped at 64 chunks; opt-in `--batch-invariant` pins a fixed 160-token split), cutting 4k/64 serving steady TPOT from about `11.7ms` to `6.46ms` on RTX 5090. Qwen3 now keeps a single model-crate bench entry: `qwen3_kernel_snapshot`, a JSON snapshot runner with warm/cold-L2 latency, default-on CUPTI counters, and compare. Correctness/truth is intentionally out of this snapshot for now.

## Determinism scope

`--batch-invariant` is scoped to one process, one toolchain, and one GPU. The cuBLASLt algo is re-tuned on every boot and cached per executor thread (`thread_local`), so it does not promise stability across cuBLAS or driver versions; startup logging records the pinned `splitk` and `reduction_scheme` per `(M,K)`. The builder rejects decode-overlap, DFlash speculative decoding, LoRA, and tensor parallelism.

It also **requires `--no-prefix-cache` and rejects `--kv-offload`**. A prefix hit advances a prompt's `chunk_start` off the request-local chunk grid, re-cutting its prefill chunks according to prior traffic. `--no-prefix-cache` closes that — except under offload, where it only disables HBM retention and deliberately leaves prefix *matching* on (host-tier reuse is the point of that mode), so that pair is rejected outright.

## Preparation

- **Read**:
  - `docs/index.md` - identified the kernels/core crate split and per-model boundary docs.
  - `docs/models/qwen3/kernels-crate.md` - Qwen3 kernel source/build ownership and human kernel index already live in `pegainfer-kernels`; model-owned DAG metadata should live with the model crate.
  - `docs/subsystems/kernels/pegainfer-kernels-boundary.md` - records the per-model engine direction and says root should be reusable frontend/control-plane infrastructure, not a universal model abstraction.
  - `src/main.rs`, `src/lib.rs`, `src/server_engine.rs`, `src/scheduler.rs`, `src/model_executor.rs`, `src/model/qwen3/*`, `src/bin/bench_serving.rs`, and Qwen3 tests - mapped what root currently knows about Qwen3.
- **Relevant history**:
  - The earlier shared-runtime work (now consolidated into `docs/subsystems/runtime/runtime.md`) was a useful simplification, but the next boundary should not make `ModelForward` the long-term universal engine API.
- **Plan**:
  1. Define the model crate/root interface before moving code.
  2. Move the generic text-generation handle/request/event types into `pegainfer-core` so root and model crates can communicate without model crates depending on root.
  3. Create `crates/pegainfer-qwen3` and move Qwen3 config, weights, forward paths, decode buffers, `Qwen3Executor`, Qwen3 scheduler internals, Qwen3 correctness tests, and Qwen3-specific benches into it.
  4. Keep root `pegainfer` as frontend plus model registry. The registry can know crate names, but `main`, `vllm_frontend`, and generic benchmark code should only see `EngineHandle`, `ModelInfo`, and tokenizer path.
  5. Add a model-owned `kernel_plan.rs` in the Qwen3 crate as the LLM/human index from model DAG phases to reusable kernels. Do not add a hand-maintained public TOML in `pegainfer-kernels`.
  6. Verify locally with format/metadata, then on the CUDA validation host with release build, clippy, Qwen3 crate e2e, and root `bench_serving snapshot`. Keep microbench timing in Criterion benches instead of duplicating it as a test.
- **Risks / open questions**:
  - If the scheduler stays in root, root still knows Qwen3's execution shape. To meet the stated goal, the Qwen3 scheduler should move into the Qwen3 crate and expose only a generic handle.
  - `bench_serving` previously had a direct `ModelForward` path for Qwen3 and a scheduler path for Qwen3.5. It needed to become generic over `EngineHandle`, while Qwen3 crate-local benches should use the model executor phase API.
  - Qwen3.5 remains in root for this phase. The registry may temporarily wrap root-local Qwen3.5, but new Qwen3 code should not depend on that temporary shape.

## Interface Proposal

The root-visible interface should be request/response oriented, not prefill/decode oriented.

```rust
// pegainfer-core
pub struct EngineLoadOptions {
    pub enable_cuda_graph: bool,
    pub device_ordinals: Vec<usize>,
    pub seed: u64,
}

pub struct ModelInfo {
    pub id: &'static str,
    pub display_name: String,
    pub max_model_len: Option<u32>,
}

pub struct GenerateRequest {
    pub prompt_tokens: Vec<u32>,
    pub params: SamplingParams,
    pub max_tokens: usize,
    pub token_tx: tokio::sync::mpsc::UnboundedSender<TokenEvent>,
    pub logprobs: usize,
    pub echo: bool,
}

pub enum TokenEvent {
    Token { id: u32, logprob: Option<TokenLogprob> },
    PromptTokens { ids: Vec<u32>, logprobs: Vec<Option<TokenLogprob>> },
    Finished { finish_reason: FinishReason, prompt_tokens: usize, completion_tokens: usize },
}

#[derive(Clone)]
pub struct EngineHandle {
    submit_tx: tokio::sync::mpsc::UnboundedSender<GenerateRequest>,
}
```

```rust
// pegainfer-qwen3
pub fn start_engine(
    model_path: &std::path::Path,
    options: EngineLoadOptions,
) -> anyhow::Result<EngineHandle>;
```

`Qwen3Model`, `BatchDecodeBuffers`, and `KvState` should not be root-facing APIs. The deliberate low-level escape hatch is `pegainfer_qwen3::runtime`, which exposes `Qwen3Executor` plus prefill/decode/unified plan types. That is the production phase boundary used by the scheduler and by model-local benches; root should still use `start_engine`.

## Execution Log

### Step 1: Add generic engine API to core
- Added `pegainfer_core::engine` with:
  - `EngineLoadOptions`
  - `ModelInfo`
  - `TokenLogprob`
  - `FinishReason`
  - `GenerateRequest`
  - `TokenEvent`
  - `EngineHandle`
- Root `server_engine` now re-exports `FinishReason` and `TokenLogprob` for compatibility.
- Root `scheduler.rs` is reduced to compatibility re-exports for `SchedulerHandle`, `SchedulerRequest`, and `TokenEvent`.

### Step 2: Extract Qwen3 crate
- Added `crates/pegainfer-qwen3`.
- Moved Qwen3-owned code into the crate:
  - config/weights/forward/prefill/decode/unified forward
  - batch decode buffers
  - `Qwen3Executor`
  - Qwen3 scheduler internals
  - Qwen3 e2e and paged-attention correctness tests
  - Qwen3 regression data generator
  - Qwen3 prefill Criterion bench
- Added `kernel_plan.rs` as the model-owned kernel routing index. It is typed Rust metadata, not a hand-maintained public TOML.

### Step 3: Remove root Qwen3 execution knowledge
- Root no longer has:
  - `src/model/qwen3.rs`
  - `src/model/qwen3/*`
  - `src/model_executor.rs`
  - Qwen3 root tests: `tests/e2e.rs`, `tests/paged_attention.rs`, `tests/bench_prefill.rs`
- Root `main.rs` starts Qwen3 through `pegainfer_qwen3::start_engine(...)`.
- Root `vllm_frontend.rs` accepts a generic `EngineHandle`.
- Root `bench_serving` uses the same generic scheduler bench path for Qwen3 instead of constructing `Qwen3Model` directly.
- Checked root with `rg` and confirmed no hits for `Qwen3Model`, `Qwen3Executor`, `ModelRuntimeConfig`, `model_executor`, `src/model/qwen3`, or stale "Qwen3 continuous" comments under root source/tests/benches/README.

### Step 4: Link and validation fixes
- Added explicit `stdc++` link output in `pegainfer-kernels` build script. Once Qwen3 became an independent crate with its own tests, the FlashInfer C++ CUDA objects needed the C++ runtime linked for test binaries as well as root binaries.
- Fixed the Qwen3 crate prefill test to respect `PEGAINFER_TEST_MODEL_PATH`.
- The validation build directory still has no `.git`, so `bench_serving snapshot` writes `commit: unknown`; after pulling it back with `rsync -e 'ssh -S none'`, the local snapshot commit field was set to current local `HEAD` short hash `0f54a1d`.

### Step 5: Verification
- Local:
  - `cargo fmt --all --check` passes.
  - `cargo metadata --no-deps --format-version 1` passes.
- CUDA validation host (RTX 5090):
  - `PEGAINFER_CUDA_SM=120 cargo clippy --release --all-targets -- -D warnings` passes.
  - `PEGAINFER_CUDA_SM=120 cargo build --release` passes.
  - `PEGAINFER_CUDA_SM=120 cargo test --release --workspace --no-run` passes.
  - `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=<model-path> cargo test --release -p pegainfer-qwen3 --test e2e -- --nocapture` passes.
  - `RUST_LOG=warn PEGAINFER_CUDA_SM=120 cargo run --release --bin bench_serving -- --model-path <model-path> snapshot` passes:
    - `prefill_heavy (10000,1)`: TTFT p50 `500.90ms`, p99 `503.30ms`
    - `decode_heavy (1024,256)`: TPOT p50 `7.57ms`, p99 `7.74ms`
    - This run exposed a scheduler length-limit bug: `max_tokens=256` emitted only `255` token events because the limit path finished without emitting the final decoded token. It was fixed in Step 7.
- Snapshot pulled back to `bench_snapshots/rtx-5090/qwen3-4b.json`.

### Step 6: Bench Boundary Cleanup
- Removed the duplicate Qwen3 `tests/bench_prefill.rs`; performance timing belongs under Criterion benches, while tests keep correctness/e2e coverage.
- Rejected a bench-only support API and also rejected using `ModelForward` as the benchmark entry.
- Added an explicit `runtime` module that re-exports the scheduler's real `Qwen3Executor` phase API: `PrefillPlan`, `DecodePlan`, `UnifiedPlan`, request items, and result types.
- Removed top-level public `Qwen3Model`, `ModelRuntimeConfig`, and `Qwen3State` re-exports. External low-level tools must opt into `runtime`; root continues to use `start_engine`.
- Replaced `crates/pegainfer-qwen3/benches/qwen3_prefill.rs` with `benches/qwen3_runtime.rs`. It measures executor prefill TTFT over `128`, `512`, `1024`, `2048`, `4096`, and `10000` token prompts, plus executor decode TPOT for batch sizes `1`, `2`, `4`, `8`, `16`, and `32` at a `1024` token context.
- Updated `tests/paged_attention.rs` to use the same executor phase API: prefill once to create KV state, then decode through `execute_decode`.
- Verification after the cleanup:
  - Local `cargo fmt --all --check` and `cargo metadata --no-deps --format-version 1` pass.
  - Local `cargo check --release -p pegainfer-qwen3 --benches --tests` cannot run on the Mac without CUDA/nvcc; with `PEGAINFER_CUDA_SM=120` it still fails at local `nvcc`.
  - CUDA host `PEGAINFER_CUDA_SM=120 cargo check --release -p pegainfer-qwen3 --benches --tests` passes.
  - CUDA host `PEGAINFER_CUDA_SM=120 cargo clippy --release --all-targets -- -D warnings` passes.
  - CUDA host `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=<model-path> cargo test --release -p pegainfer-qwen3 --test paged_attention -- --nocapture` passes.
  - CUDA host full Criterion bench passes with `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=<model-path> cargo bench -p pegainfer-qwen3 --bench qwen3_runtime`:
    - Prefill TTFT: `128 -> 11.804ms`, `512 -> 23.200ms`, `1024 -> 44.114ms`, `2048 -> 87.327ms`, `4096 -> 179.60ms`, `10000 -> 505.55ms`.
    - Decode one-step batch time at 1024-token context: `bs1 -> 9.3095ms`, `bs2 -> 9.3207ms`, `bs4 -> 9.4059ms`, `bs8 -> 10.960ms`, `bs16 -> 11.718ms`, `bs32 -> 13.196ms`.

### Step 7: Retire ModelForward and Fix Length Limit
- Deleted `pegainfer_core::model::{ModelForward, GenerationState}` and removed the root `src/model.rs` re-export.
- Deleted the Qwen3 `forward.rs` compatibility path. Qwen3 tests that used it now build their baselines from `batch_prefill(bs=1)` plus `batch_decode(bs=1)`, so they exercise the same phase APIs as production.
- Fixed Qwen3 decode length-limit handling by adding `DecodeEffect::EmitAndFinish`. At that point EOS still finished without emitting the stop token; the current typed stop contract emits the trigger token and its `StopCause` together. Length-limited output likewise retains the sampled final token before `Finished { finish_reason: Length }`.
- Regenerated `test_data/Qwen3-4B.json` because every length-limited golden output now includes the final requested token.
- Re-ran `bench_serving snapshot` on the CUDA validation host and pulled back `bench_snapshots/rtx-5090/qwen3-4b.json`; `decode_heavy (1024,256)` now records `generated_tokens min=max=avg=256`.
- Performance stayed within noise on RTX 5090:
  - `prefill_heavy (10000,1)`: TTFT p50 `501.69ms`, p99 `503.16ms`.
  - `decode_heavy (1024,256)`: TPOT p50 `7.56ms`, p99 `7.73ms`.
- Final verification after this step:
  - Local `cargo fmt --all --check`, `cargo metadata --no-deps --format-version 1`, and `git diff --check` pass.
  - CUDA host `PEGAINFER_CUDA_SM=120 cargo clippy --release --all-targets -- -D warnings` passes.
  - CUDA host `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=<model-path> cargo test --release -p pegainfer-qwen3 --test e2e -- --nocapture` passes.

### Step 8: Decode Context-Length Sweep and Compile Audit
- Added `crates/pegainfer-qwen3/src/bin/qwen3_decode_context.rs` as a production-path fixed-context decode probe. It prefills a fresh request to a selected context length, then measures or profiles real `Qwen3Executor::execute_decode`; the optional `cudaProfilerStart/Stop` range only exists for profiler capture and does not run in normal serving.
- GPU fixed-context command:
  - `PEGAINFER_CUDA_SM=120 target/release/qwen3_decode_context --model-path <model-path> --iters 10 --contexts 128,512,1024,2048,4096,8192,10000`
- Result on RTX 5090:

| Context | Decode p50 |
| --- | ---: |
| 128 | `6.1107ms` |
| 512 | `6.7094ms` |
| 1024 | `7.4256ms` |
| 2048 | `8.8918ms` |
| 4096 | `11.7912ms` |
| 8192 | `17.5457ms` |
| 10000 | `20.0653ms` |

- Linear fit across the sweep: `TPOT ~= 5.9789ms + 1.411us/token * context`, `R^2=0.99997`.
- `nsys` with `--cuda-graph-trace=node` shows the growth is almost entirely FlashInfer paged decode attention:

| Context | Total kernel time / step | Attention / step | Non-attention / step |
| --- | ---: | ---: | ---: |
| 1024 | `7.3287ms` | `1.5390ms` | `5.7897ms` |
| 10000 | `19.6907ms` | `13.8868ms` | `5.8039ms` |

- H2D traffic in the profiled decode range was only about `20-23us/step`, so metadata dirty caching is good runtime hygiene but cannot explain a multi-ms TPOT gap.
- Compile audit on the same validation worktree:
  - GPU reports compute capability `12.0`; default toolkit is CUDA `12.9` (`nvcc V12.9.86`), driver `575.57.08`.
  - `crates/pegainfer-kernels/build.rs` emits `-O3 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 --compiler-options -fPIC`; FlashInfer translation units add `--std=c++17` and the FlashInfer include path.
  - `cuobjdump -lelf` confirms both `libkernels_cuda.a` and `target/release/pegainfer` contain `sm_120.cubin`. `compute_120` PTX fallback is also embedded, but the matching SASS is present, so this is not PTX-JIT-only execution.
  - CUDA `13.1` is installed and can build the same code into `sm_120` cubins, but the current driver/runtime combination cannot run it (`cudaError=35` after linking `libcudart.so.13`). Until the driver is upgraded, CUDA `12.9` is the latest runnable toolkit on this box.
- Interpretation: the compile target is correct. The `bs=1` long-context slope is the known non-partition FlashInfer paged decode issue: grid shape is effectively `(batch_size, num_kv_heads) = (1, 8)`, so only 8 CTAs scan the whole KV context. At `ctx=4096`, Qwen3-4B attention reads about `604MB` (`576MiB`) of K/V per token; the measured attention time is about `5.7ms`, or roughly `105GB/s` effective aggregate bandwidth, far below the RTX 5090 memory system because the kernel under-fills the GPU. The next real fix is partition-KV/split-K decode for `bs=1` or low-batch, not build-flag tuning.

### Step 9: Pure Paged Decode Attention Bench
- Added `crates/pegainfer-qwen3/benches/qwen3_attention.rs`.
- The bench does not load Qwen3 weights. It constructs synthetic non-zero Q and paged KV buffers using Qwen3-4B attention shape: `num_qo_heads=32`, `num_kv_heads=8`, `head_dim=128`, `page_size=16`, one layer.
- The bench calls the FlashInfer paged decode FFI directly and uses CUDA events around the kernel launches. It measures decode attention only; it excludes QKV projection, KV append, O projection, MLP, scheduler, tokenizer, and host-side serving overhead.
- Added `paged_attention_decode_split_kv_cuda` as a reusable kernel entry for FlashInfer partition-KV/split-K decode. Runtime dispatch still uses the existing non-partition path; this step only exposes and benchmarks the candidate operator.
- The split-K bench uses `chunk_size=512` and `max_chunks_per_request=64`. Active chunks are packed for `o_indptr`, and remaining graph-stability slots are masked with `block_valid_mask=0`.
- Bench setup runs a non-timed D2H sanity check comparing split-K output with the non-partition output for every synthetic case. The GPU run below passed that check.
- Registered it as a model-crate Criterion bench instead of a kernels-crate bench because the shape, context sweep, and interpretation are Qwen3-specific; the implementation still directly indexes the reusable kernel entry point.
- Local verification:
  - `cargo fmt --all --check` passes.
  - `cargo metadata --no-deps --format-version 1` passes.
  - `git diff --check` passes.
- GPU compile:
  - `PEGAINFER_CUDA_SM=120 cargo check --release -p pegainfer-qwen3 --bench qwen3_attention` passes.
  - `PEGAINFER_CUDA_SM=120 cargo clippy --release -p pegainfer-qwen3 --all-targets -- -D warnings` passes.
- GPU run:
  - `PEGAINFER_CUDA_SM=120 cargo bench -p pegainfer-qwen3 --bench qwen3_attention -- --noplot` passes.

Single-layer `bs=1` context sweep on RTX 5090:

| KV length | Non-partition | Split-K 512/padded64 | Speedup |
| --- | ---: | ---: | ---: |
| 128 | `6.56us` | `8.91us` | `0.74x` |
| 512 | `18.70us` | `20.96us` | `0.89x` |
| 1024 | `34.94us` | `21.17us` | `1.65x` |
| 2048 | `65.78us` | `20.85us` | `3.16x` |
| 4096 | `129.23us` | `21.59us` | `5.99x` |
| 8192 | `254.98us` | `22.81us` | `11.18x` |
| 10000 | `311.51us` | `23.31us` | `13.36x` |

Batch sweep at `kv_len=1024`:

| Batch size | Non-partition | Split-K 512/padded64 | Speedup |
| --- | ---: | ---: | ---: |
| 1 | `34.97us` | `20.89us` | `1.67x` |
| 2 | `34.96us` | `27.07us` | `1.29x` |
| 4 | `34.90us` | `38.44us` | `0.91x` |
| 8 | `35.00us` | `45.95us` | `0.76x` |
| 16 | `35.05us` | `51.11us` | `0.69x` |
| 32 | `86.12us` | `92.51us` | `0.93x` |

Interpretation: the pure operator data reproduces the same shape as the full decode profile. At `bs=1`, non-partition time grows almost linearly with KV length, while graph-stable split-K stays near `21-23us/layer` once context reaches 1k. Multiplying the synthetic ctx10000 split-K result by 36 layers gives about `0.84ms` attention-only time instead of about `11.2ms` from non-partition. That is the right order of magnitude for fixing the long-context TPOT slope. The batch sweep also shows the guard must be conservative: at `kv_len=1024`, split-K only wins for `bs<=2`, and non-partition is better once batch already provides enough request/head CTAs.

### Step 10: Runtime Split-K Decode Gate
- Integrated `paged_attention_decode_split_kv_cuda` into the Qwen3 decode runtime.
- `BatchDecodeBuffers` now owns split-K metadata and workspace:
  - `split_request_indices_d`
  - `split_kv_tile_indices_d`
  - `split_kv_chunk_size_d`
  - `split_o_indptr_d`
  - `split_block_valid_mask_d`
  - `split_tmp_v`
  - `split_tmp_s`
- CUDA graph cache is now keyed by `(batch_bucket, attention_path)` instead of only `batch_bucket`. This matters because a request can first capture the `bs=1` non-partition graph at short context and later cross the split-K threshold; the split-K path needs its own graph capture.
- Runtime gate:
  - split-K when `padded_bs <= 2 && max_seq_len >= 1024`
  - otherwise keep non-partition decode
- Split-K metadata uses `chunk_size=max(512, ceil(max_seq_len / 64))` and `64` reserved chunk slots per request. Real chunk slots are packed for `o_indptr`; unused graph-stability slots are masked with `block_valid_mask=0`.
- Padding batch slots get zero active split chunks. Their output is discarded, and the batch columns remain independent through GEMMs.

GPU validation:

| Check | Result |
| --- | --- |
| `PEGAINFER_CUDA_SM=120 cargo check --release -p pegainfer-qwen3 --all-targets` | pass |
| `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=<model-path> cargo test --release -p pegainfer-qwen3 --test e2e -- --nocapture` | pass |
| `PEGAINFER_CUDA_SM=120 cargo clippy --release --all-targets -- -D warnings` | pass |

Fixed-context decode probe after runtime integration:

| Prompt context | Decode KV len | p50 TPOT before | p50 TPOT after |
| --- | ---: | ---: | ---: |
| 1024 | 1025 | `7.43ms` | `6.74ms` |
| 4096 | 4097 | `11.79ms` | `6.82ms` |
| 10000 | 10001 | `20.07ms` | `7.06ms` |

Command:

```bash
PEGAINFER_CUDA_SM=120 target/release/qwen3_decode_context \
  --model-path <model-path> \
  --iters 10 \
  --contexts 1024,4096,10000
```

Cross-threshold smoke:

```bash
PEGAINFER_CUDA_SM=120 target/release/qwen3_decode_context \
  --model-path <model-path> \
  --iters 600 \
  --contexts 512
```

Result: pass, `p50=6.7156ms`. This exercises a single request growing from non-partition territory into the split-K threshold with separate graph captures.

Serving request check after rebuilding `bench_serving`:

```bash
RUST_LOG=warn PEGAINFER_CUDA_SM=120 target/release/bench_serving \
  --model-path <model-path> \
  request --prompt-len 4096 --output-len 64
```

Result:

| Metric | Before | After |
| --- | ---: | ---: |
| `first_decode_step_ms p50` | `11.74ms` | `6.82ms` |
| `steady_tpot_ms p50` | `11.72ms` | `6.77ms` |
| `e2e_ms p50` | `916.34ms` | `604.24ms` |

Interpretation: split-K removes the long-context attention slope for the low-batch case. The remaining `~6.8-7.1ms` TPOT is now dominated by the non-attention decode body: GEMMs/GEMVs, MLP, norms, logits, sampling, and graph replay overhead. Next optimization work should not keep pushing paged attention first; it should re-profile the post-split decode step and pick the new largest kernel family.

### Step 11: Attention Theoretical Bandwidth Estimate
- Updated `crates/pegainfer-qwen3/benches/qwen3_attention.rs` to print a one-time theoretical bandwidth report before Criterion runs.
- The report queries CUDA Driver attributes:
  - `CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE`
  - `CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH`
- On RTX 5090, CUDA reports `14001MHz` memory clock and a `512-bit` memory bus. Using `2` transfers per memory clock gives `1792.128GB/s`, matching the public RTX 5090 bandwidth figure.
- The report uses Qwen3 KV read bytes only:
  - `bs * kv_len * num_kv_heads * head_dim * 2(K,V) * sizeof(bf16)`
  - This is a counter-free lower-bound estimate, not measured DRAM bytes.
- Verification command:

```bash
PEGAINFER_CUDA_SM=120 cargo bench -p pegainfer-qwen3 \
  --bench qwen3_attention -- --noplot
```

Key RTX 5090 report rows:

| Case | KV read | Time | Effective GB/s | Peak % |
| --- | ---: | ---: | ---: | ---: |
| `ctx bs1 non_partition 4096` | `16.777MB` | `129.285us` | `129.769` | `7.24%` |
| `ctx bs1 non_partition 10000` | `40.960MB` | `309.744us` | `132.238` | `7.38%` |
| `ctx bs1 split_k512_padded64 4096` | `16.777MB` | `21.494us` | `780.536` | `43.55%` |
| `ctx bs1 split_k512_padded64 10000` | `40.960MB` | `23.294us` | `1758.386` | `98.12%` |

Batch sweep sanity rows at `kv_len=1024`:

| Case | KV read | Time | Effective GB/s | Peak % |
| --- | ---: | ---: | ---: | ---: |
| `batch non_partition bs8` | `33.554MB` | `34.935us` | `960.482` | `53.59%` |
| `batch non_partition bs16` | `67.109MB` | `35.004us` | `1917.174` | `106.98%` |
| `batch split_k512_padded64 bs32` | `134.218MB` | `92.533us` | `1450.489` | `80.94%` |

Interpretation: the estimate is good enough to prove the original `bs=1` non-partition path was badly under-filling memory bandwidth. It is not good enough to make final hardware-utilization claims because single-layer KV working sets fit in the RTX 5090's `96MiB` L2; the `bs16` non-partition row exceeding `100%` of DRAM peak is the warning sign. The next measurement step should use CUPTI Profiler or NCU counters for `dram__bytes_*`, `lts__t_bytes.*`, and `*_pct_of_peak_sustained_elapsed`.

### Step 12: CUPTI Counters and Split-K Retune
- Added `crates/pegainfer-cupti`, a small CUPTI Range Profiler wrapper used by the attention bench. It profiles only the attention launch range and lets the bench clear L2 before `cuptiRangeProfilerStart`, so cache-clear traffic is excluded from the measured range.
- Extended `crates/pegainfer-qwen3/benches/qwen3_attention.rs`:
  - `PEGAINFER_QWEN3_ATTENTION_CUPTI=1` prints cold-L2 CUPTI rows for `gpu__time_duration.sum`, `dram__bytes.sum`, `dram__bytes_op_read.sum`, `dram__bytes_op_write.sum`, and `lts__t_bytes.sum`.
  - `PEGAINFER_QWEN3_ATTENTION_SPLITK_SWEEP=1` sweeps split-K chunk sizes and max chunk slots.
  - `PEGAINFER_QWEN3_ATTENTION_REPORT_ONLY=1` prints reports without running Criterion samples.
- GPU CUPTI command:

```bash
PEGAINFER_CUDA_SM=120 \
PEGAINFER_QWEN3_ATTENTION_REPORT_ONLY=1 \
PEGAINFER_QWEN3_ATTENTION_CUPTI=1 \
cargo bench -p pegainfer-qwen3 --bench qwen3_attention -- --noplot
```

Key cold-L2 CUPTI rows at `bs=1,ctx=10000`:

| Path | GPU time | DRAM read | DRAM total | DRAM GB/s | Peak % | KV read / DRAM read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| non-partition | `425.920us` | `41.028MB` | `55.123MB` | `129.421` | `7.22%` | `99.83%` |
| split-K 512/64 | `76.896us` | `41.019MB` | `54.431MB` | `707.849` | `39.50%` | `99.86%` |
| split-K 256/64 | `66.976us` | `41.020MB` | `48.712MB` | `727.301` | `40.58%` | `99.85%` |

Interpretation: FlashInfer is not rereading KV many times from DRAM. The non-partition path reads roughly the theoretical KV bytes but uses only about `7%` of peak DRAM bandwidth because `bs=1` launches too little work. Split-K increases parallelism and moves the same required KV read to roughly `40%` of peak DRAM bandwidth in the cold-L2 CUPTI range.

Split-K sweep command:

```bash
PEGAINFER_CUDA_SM=120 \
PEGAINFER_QWEN3_ATTENTION_REPORT_ONLY=1 \
PEGAINFER_QWEN3_ATTENTION_SPLITK_SWEEP=1 \
cargo bench -p pegainfer-qwen3 --bench qwen3_attention -- --noplot
```

Representative cold-L2 sweep rows:

| Case | 256/64 | 512/64 | Result |
| --- | ---: | ---: | --- |
| `bs1 ctx1024` | `22.197us` | `30.823us` | 256/64 wins |
| `bs1 ctx4096` | `26.703us` | `35.159us` | 256/64 wins |
| `bs1 ctx10000` | `38.912us` | `46.824us` | 256/64 wins |
| `bs2 ctx1024` | `23.713us` | `34.705us` | 256/64 wins |
| `bs2 ctx8192` | `54.637us` | `55.200us` | tied, 256/64 slightly ahead |
| `bs2 ctx10000` | `62.417us` | `63.620us` | tied, 256/64 slightly ahead |

Current runtime config: `SPLIT_KV_CHUNK_TOKENS=64` (chunk-size floor), `SPLIT_KV_TUNED_MAX_CHUNKS=64` (default `Tuned` chunk-count cap and grid), `SPLIT_KV_MAX_CHUNKS_PER_REQUEST=256` (workspace bound + the `Pin`/`PerToken` fixed-split basis → 160 tokens). See decode-attention.md.

Historical note (this section's retune): `BatchDecodeBuffers` used `SPLIT_KV_CHUNK_TOKENS=256` with `SPLIT_KV_MAX_CHUNKS_PER_REQUEST=64`, keeping the original `512/64` graph-stable padded budget while doubling active chunks for low-batch long-context decode. Later superseded by the 64-token chunk floor and the split-KV size/grid decouple.

Production decode probe after retune:

```bash
PEGAINFER_CUDA_SM=120 cargo build --release \
  -p pegainfer-qwen3 --bin qwen3_decode_context
PEGAINFER_CUDA_SM=120 target/release/qwen3_decode_context \
  --model-path <model-path> \
  --iters 10 \
  --contexts 1024,4096,10000
```

| Prompt context | Decode KV len | p50 TPOT |
| --- | ---: | ---: |
| 1024 | 1025 | `6.4002ms` |
| 4096 | 4097 | `6.5327ms` |
| 10000 | 10001 | `7.0436ms` |

Serving check after syncing the root `src/` worktree on the CUDA validation host:

```bash
RUST_LOG=warn PEGAINFER_CUDA_SM=120 cargo run --release \
  --bin bench_serving -- \
  --model-path <model-path> \
  request --prompt-len 4096 --output-len 64 --warmup 5 --iters 20
```

| Metric | p50 | p95 | Samples |
| --- | ---: | ---: | ---: |
| `ttft_ms` | `177.21ms` | `177.86ms` | 20 |
| `first_decode_step_ms` | `6.51ms` | `6.52ms` | 20 |
| `steady_tpot_ms` | `6.46ms` | `6.48ms` | 1240 |
| `e2e_ms` | `584.79ms` | `585.25ms` | 20 |

Verification:

| Check | Result |
| --- | --- |
| `PEGAINFER_CUDA_SM=120 cargo clippy --release -p pegainfer-cupti -p pegainfer-qwen3 --bench qwen3_attention -- -D warnings` | pass |
| `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=<model-path> cargo test --release -p pegainfer-qwen3 --test e2e -- --nocapture` | pass |
| `PEGAINFER_CUDA_SM=120 PEGAINFER_QWEN3_ATTENTION_REPORT_ONLY=1 cargo bench -p pegainfer-qwen3 --bench qwen3_attention -- --noplot` | pass |
| `cargo fmt --all --check` | pass |
| `cargo metadata --no-deps --format-version 1` | pass |
| `git diff --check` | pass |

Note: an initial remote e2e run failed because the remote `test_data/Qwen3-4B.json` was stale and expected the pre length-limit baseline. Syncing the tracked baseline fixed it; this was not a split-K numerical drift.

### Step 13: Kernel Snapshot MVP
- Extracted the Qwen3 paged decode attention case construction into `crates/pegainfer-qwen3/src/kernel_bench.rs`.
- Added `crates/pegainfer-qwen3/benches/qwen3_kernel_snapshot.rs` as a deterministic `harness=false` runner.
- Removed the temporary correctness envelope from the snapshot runner. We do not have a settled truth source for this layer yet, so correctness belongs in a separate design rather than a misleading "non-partition equals truth" field.
- CUPTI is default-on in the snapshot runner. `--no-cupti` is available only for latency-only smoke runs.

Snapshot command:

```bash
PEGAINFER_CUDA_SM=120 cargo bench -p pegainfer-qwen3 \
  --bench qwen3_kernel_snapshot -- \
  run \
  --contexts 1024 \
  --batch-sizes 1 \
  --variants non_partition,split_kv_256x64 \
  --iters 4 \
  --out $RESULT_ROOT/qwen3_kernel_snapshot_smoke.json
```

Compare command:

```bash
cargo bench -p pegainfer-qwen3 \
  --bench qwen3_kernel_snapshot -- \
  compare \
  --base $RESULT_ROOT/qwen3_kernel_snapshot_smoke.json \
  --new $RESULT_ROOT/qwen3_kernel_snapshot_smoke.json
```

The JSON snapshot records:
- model/op identity: `qwen3`, `paged_decode_attention`
- hardware: GPU name, compute capability, memory clock, memory bus width, theoretical peak bandwidth, L2 size, cache-clear size
- measurement recipe: warm iters, cold-L2 iters, `INNER_LAUNCHES`
- CUPTI recipe: enabled flag and metric list
- per-case shape: batch, KV length, head shape, page size, dtype
- per-case variant/params: non-partition or split-K `chunk_tokens/max_chunks`
- warm and cold-L2 CUDA event latency
- CUPTI counters: GPU time, DRAM read/write/total bytes, L2 bytes, SM throughput percentage, active-warp percentage, DRAM bandwidth, peak percentage, and theoretical KV-read over DRAM-read percentage
- theoretical KV read bytes

GPU smoke result for `bs=1,ctx=1024,iters=4`:

| Variant | Warm | Cold-L2 | CUPTI GPU | CUPTI DRAM read | SM throughput | Active warps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `non_partition` | about `35us` | about `46us` | `75.776us` | `4.236MB` | `0.75%` | `8.27%` |
| `split_kv_256x64` | about `13us` | about `21us` | `48.736us` | `4.249MB` | `1.31%` | `11.77%` |

Snapshot compare result:

```text
kernel snapshot compare complete: warnings=0 failures=0
```

CUPTI note: the standalone snapshot runner originally crashed inside `libnvperf_host.so` at `NVPW_CUDA_Profiler_DecodeCounters`. The root cause was the verbose user range name, not the attention case or Rust callback trampoline. The fix is to use compact range names such as `qk/non_partition/b1/k1024` and keep full metadata in JSON fields. The first profiled launch also needs an unprofiled warmup launch; otherwise CUDA lazy initialization pollutes the first CUPTI GPU time.

```bash
PEGAINFER_CUDA_SM=120 cargo bench -p pegainfer-qwen3 \
  --bench qwen3_kernel_snapshot -- \
  run \
  --contexts 1024 \
  --batch-sizes 1 \
  --variants non_partition,split_kv_256x64 \
  --iters 4 \
  --out $RESULT_ROOT/qwen3_kernel_snapshot_cupti_smoke.json
```

Verification:

| Check | Result |
| --- | --- |
| `PEGAINFER_CUDA_SM=120 cargo clippy --release -p pegainfer-cupti -p pegainfer-qwen3 --bench qwen3_kernel_snapshot -- -D warnings` | pass |
| `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=<model-path> cargo test --release -p pegainfer-qwen3 --test e2e -- --nocapture` | pass |
| `PEGAINFER_CUDA_SM=120 cargo bench -p pegainfer-qwen3 --bench qwen3_kernel_snapshot -- run --contexts 1024 --batch-sizes 1 --variants non_partition,split_kv_256x64 --iters 4 --out $RESULT_ROOT/qwen3_kernel_snapshot_cupti_smoke.json` | pass |

The SM counters are intentionally minimal. `sm__throughput.avg.pct_of_peak_sustained_elapsed` shows whether SMs are busy over elapsed time; `smsp__warps_active.avg.pct_of_peak_sustained_active` shows active-warp residency while SM partitions are active. At `bs=1,ctx=10000`, non-partition measured `1.19%` SM throughput and `6.59%` DRAM peak, while split-K measured `8.74%` SM throughput and `41.06%` DRAM peak for nearly identical DRAM read bytes. That is the kernel snapshot evidence for low-batch underfill.

### Step 14: Consolidate Bench Entry Points
- Deleted the retired Criterion benches:
  - `crates/pegainfer-qwen3/benches/qwen3_runtime.rs`
  - `crates/pegainfer-qwen3/benches/qwen3_attention.rs`
- Removed their `[[bench]]` entries and the Qwen3 crate-local `criterion` dev dependency.
- Qwen3 now has exactly one model-crate bench entry: `qwen3_kernel_snapshot`.
- Rationale: the human CSV report, split-K tuning sweep, and machine-readable JSON runner were duplicating case construction, metric selection, and interpretation. Kernel maintenance should have one durable artifact first; optional human views should be generated from snapshot data rather than maintained as separate benches.

Verification after consolidation:

| Check | Result |
| --- | --- |
| `cargo fmt --all --check` | pass |
| `cargo metadata --no-deps --format-version 1` | pass |
| `git diff --check` | pass |
| `PEGAINFER_CUDA_SM=120 cargo check --release -p pegainfer-qwen3 --bench qwen3_kernel_snapshot` on the CUDA validation host | pass |
| `PEGAINFER_CUDA_SM=120 cargo clippy --release -p pegainfer-cupti -p pegainfer-qwen3 --bench qwen3_kernel_snapshot -- -D warnings` on the CUDA validation host | pass |
| `PEGAINFER_CUDA_SM=120 cargo bench -p pegainfer-qwen3 --bench qwen3_kernel_snapshot -- run --contexts 1024 --batch-sizes 1 --variants non_partition,split_kv_256x64 --iters 4 --out $RESULT_ROOT/qwen3_kernel_snapshot_single_bench_smoke.json` on the CUDA validation host | pass |
| `PEGAINFER_CUDA_SM=120 cargo bench -p pegainfer-qwen3 --bench qwen3_kernel_snapshot -- compare --base $RESULT_ROOT/qwen3_kernel_snapshot_single_bench_smoke.json --new $RESULT_ROOT/qwen3_kernel_snapshot_single_bench_smoke.json` on the CUDA validation host | pass |

## Debrief

The Qwen3 split now enforces the intended dependency direction: model execution code depends on `pegainfer-core` and `pegainfer-kernels`; root depends on the model crate only at registry/startup glue points. Root still has a `ModelType::Qwen3` enum and default Qwen3 model path because the product needs a loader choice, but it no longer sees Qwen3 layers, KV state, TP rank workers, or prefill/decode/unified plans.

Next cleanup should be a generic model registry module so `main.rs` and `bench_serving.rs` stop matching model crate names directly. Performance-wise, the next target is the post-split decode body: GEMM/GEMV, MLP, norms, logits, sampling, and graph replay overhead now dominate the remaining `~6.5-7.0ms` TPOT. Kernel DevOps-wise, the next target is defining a real correctness/truth source for kernel snapshots instead of treating one implementation path as the oracle.
