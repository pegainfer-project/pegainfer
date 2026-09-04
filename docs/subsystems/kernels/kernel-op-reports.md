# Kernel Op Reports

**Created**: 2026-05-04
**Last touched**: 2026-08
**Status**: active; prefill/report commit ready, decode tuning deferred
**TL;DR**: The harness source moved out of the library in #944 — it now lives in `pegainfer-qwen3/src/bin/report_support/` and is compiled only by the two `kernel-report` binaries, so paths in the Execution Log below are historical. `qwen3_kernel_snapshot` is no longer a Cargo bench. Qwen3 has feature-gated `qwen3_kernel_report` per-op kernel tooling and `qwen3_model_report` model-level decode operator reporting. Decode now routes through an eager `BatchDecodeDag`, so the executable forward sequence is also the `KernelCall` contract source for runtime tracing; `qwen3_model_report` disables CUDA Graph, traces that DAG, then joins traced TensorSpecs against measured microbench results to emit by-op, by-call-site, coverage, schedule-preview, latency-stat, and Graphviz DOT reports. Prefill remains covered by `qwen3_kernel_report` stage reports; measured FA2 `CTA_TILE_Q=64` is the Qwen3 production prefill default on the RTX 5090 grid. The model-agnostic timing loop, latency stats, and by-op/by-call-site rollup now live in the `pegainfer-bench` crate, shared by the Kimi-K2 model report; the qwen3 attention-specific regression framework (manifest/snapshot/provenance/CUPTI/cold-L2) deliberately stays in `qwen3_kernel_report`. `qwen3_model_report` traces + measures projection GEMMs under the production `numeric_policy()`: Pin and PerToken are faithful (Pin's serving counter-observed via `pin_served`, PerToken's via `per_token_served`), while under the default Tuned policy the GEMM rows are classified `unfaithful_gemmex` and excluded from the totals without being measured (the untuned context would run GemmEx, not production's per-process-tuned algo); the JSON config records the numeric policy and a partial-total flag.

## Preparation

- **Read**:
  - `docs/index.md` - located the active benchmarking, CUPTI, kernel-boundary, and Qwen3 model-crate docs.
  - `docs/models/qwen3/model-crate.md` - confirmed `qwen3_kernel_snapshot` was the current Qwen3 kernel snapshot runner and already captured warm/cold-L2 latency plus default CUPTI counters.
  - `docs/conventions/bench-regression.md` - clarified that the existing serving benchmark remains the model-level regression artifact; this task should not mix per-op reports with E2E snapshots.
  - `docs/subsystems/kernels/pegainfer-kernels-boundary.md` - confirmed kernels should become first-class measurable assets and model DAG manifests should live with model crates.
  - `docs/models/qwen3/kernels-crate.md` - confirmed kernel source/build ownership now lives in `pegainfer-kernels`, while model-owned DAG metadata belongs in the Qwen3 crate.
  - `docs/playbooks/profiling-guide.md` - confirmed the diagnostic split between kernel composition/proportions and benchmark-grade latency.
- **Relevant history**:
  - `docs/models/qwen3/model-crate.md` showed the current single-op snapshot already found the low-batch long-context decode-attention bottleneck.
- **Plan**:
  1. Add direct Qwen3 crate dev-dependencies for generic infrastructure (`clap` derive for CLI and `toml` for manifest parsing) instead of extending the hand-written parser.
  2. Add a model-local TOML manifest for Qwen3-4B kernel reports, initially covering only op names, phases, shape sweeps, and variants.
  3. Replace `crates/pegainfer-qwen3/benches/qwen3_kernel_snapshot.rs` with a manifest-driven `qwen3_kernel_report` bin; do not keep a bench wrapper.
  4. Add a composition command that reads per-op case results and emits a decode phase report by joining the manifest's op repeat rules with measured per-op reports.
  5. Run formatting and the strongest local compile checks available; GPU execution may still require the CUDA validation host because this machine lacks local CUDA tooling.
- **Risks / open questions**:
  - The first composition report can only explain measured ops. It should report uncovered/residual structure explicitly instead of pretending to estimate full TPOT from one op.
  - Adding more providers later should not require changing the per-op report schema, so the first schema needs stable IDs and generic `serde_json::Value` fields for shape and selector keys.

## Execution Log

### Step 1: Move from bench target to bin
- Removed the `qwen3_kernel_snapshot` bench target from `crates/pegainfer-qwen3/Cargo.toml`.
- Moved the report runner to `crates/pegainfer-qwen3/src/bin/qwen3_kernel_report.rs`.
- Added a `kernel-report` feature for generic tool dependencies (`clap`, `toml`, `sha2`, `hex`) and `pegainfer-cupti`; the bin requires that feature so normal Qwen3 library/server builds do not pull CUPTI into the default dependency graph.
- Removed the temporary `cargo bench` compatibility argument handling after the tool became a normal binary.

### Step 2: Add model-local manifest
- Added `crates/pegainfer-qwen3/kernel_manifests/qwen3.toml`.
- The first manifest now stays deliberately thin: `model`, `[[ops]]`, `phase`, per-op shape sweep fields, and variant labels. Provider-owned facts such as dtype, head counts, head dimension, page size, thresholds, and composition policy stay in Rust.

### Step 3: Refactor report schema and commands
- Added manifest-driven `run`, `compare`, and `compose` commands using `clap` derive instead of the old hand-written parser.
- Per-op reports now include `report_type`, `parallel_strategy`, `phase`, `manifest_hash`, `case_id`, `selector_key`, `shape_source`, thresholds, raw CUPTI metric maps, and `selections` for best-variant lookup.
- `compose` emits the built-in Qwen3 decode-attention-only contribution report. It no longer reads a hand-written composition DAG from the TOML manifest.

### Step 4: Local validation
- `cargo fmt --all --check` passed.
- `cargo metadata --no-deps --format-version 1` passed.
- Local `cargo check --release -p pegainfer-qwen3 --bench qwen3_kernel_snapshot` previously failed before Rust type checking because this local host lacks a usable `nvcc`; GPU validation moved to the CUDA validation host.

### Step 5: GPU minimal validation
- Rebuilt the disposable validation worktree at `<validation-worktree>` from local `HEAD` commit `612850f`, then rsynced the current working tree changes over it.
- Copied initialized FlashInfer headers from `<validation-checkout>/third_party/flashinfer` into the clean worktree's `crates/pegainfer-kernels/third_party/flashinfer` directory.
- `PEGAINFER_CUDA_SM=120 cargo build --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report` passed.
- `PEGAINFER_CUDA_SM=120 cargo run --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report -- run --no-cupti --iters 1 --contexts 1024 --batch-sizes 1 --variants non_partition --out $RESULT_ROOT/qwen3_kernel_op_report_min.json` passed.
- `PEGAINFER_CUDA_SM=120 cargo run --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report -- compare --base $RESULT_ROOT/qwen3_kernel_op_report_min.json --new $RESULT_ROOT/qwen3_kernel_op_report_min.json` passed with `warnings=0 failures=0`.
- `PEGAINFER_CUDA_SM=120 cargo run --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report -- compose --input $RESULT_ROOT/qwen3_kernel_op_report_min.json --batch-size 1 --context 1024 --out $RESULT_ROOT/qwen3_kernel_composition_min.json` passed.
- CUPTI minimal validation passed with `non_partition,split_kv_256x64` at `bs=1,ctx=1024`; the report contained 2 cases, 1 selection, CUPTI metrics for both cases, and selected `split_kv_256x64`.
- Default package build without the report feature also passed: `PEGAINFER_CUDA_SM=120 cargo build --release -p pegainfer-qwen3`.

### Step 6: Full GPU manifest run
- Ran the full manifest command on the validation worktree:
  - `PEGAINFER_CUDA_SM=120 time cargo run --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report -- run --out $RESULT_ROOT/qwen3_kernel_report_full.json`
- Result:
  - `126` cases: `6` batch sizes x `7` context lengths x `3` variants.
  - `42` selections.
  - `0` case errors.
  - `126` cases with CUPTI measurements.
  - Runtime: `2:42.83 elapsed`.
  - Manifest hash: `62aada084b61795862c5d4dd23fa89d1`.
- Self-compare passed:
  - `PEGAINFER_CUDA_SM=120 cargo run --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report -- compare --base $RESULT_ROOT/qwen3_kernel_report_full.json --new $RESULT_ROOT/qwen3_kernel_report_full.json`
  - Output: `kernel report compare complete: warnings=0 failures=0`.
- Representative selections:
  - `bs=1,ctx=1024`: `split_kv_256x64`.
  - `bs=1,ctx=4096`: `split_kv_256x64`.
  - `bs=1,ctx=10000`: `split_kv_256x64`.
  - `bs=32,ctx=1024`: `non_partition`.
  - `bs=32,ctx=10000`: `non_partition`.
- Selection counts:
  - `split_kv_256x64`: `20`.
  - `non_partition`: `15`.
  - `split_kv_512x64`: `7`.
- Composed the full report for `bs=1,ctx=4096`:
  - `PEGAINFER_CUDA_SM=120 cargo run --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report -- compose --input $RESULT_ROOT/qwen3_kernel_report_full.json --batch-size 1 --context 4096 --out $RESULT_ROOT/qwen3_kernel_composition_full_bs1_ctx4096.json`
  - Output total: cold-L2 `958.473us`, `split_kv_256x64` repeated across 36 layers.
  - Coverage note still applies: only `paged_decode_attention` is included; linear, MLP, norm, embedding, and sampling are not covered yet.
- Preserved the generated JSONs under:
  - `<kernel-report-dir>`
  - `<kernel-report-dir>`

### Step 7: Remove warm-cache latency
- CUDA official APIs expose `cudaCtxResetPersistingL2Cache` / `cuCtxResetPersistingL2Cache` for resetting persisting L2 lines to normal status, not for evicting ordinary L2 cache contents. The report runner therefore keeps the explicit L2 sweep buffer for cache-cleared timing.
- Removed warm-cache latency from the report schema:
  - `measurement.warm_iters` removed.
  - `warm_latency_us` removed from cases and selections.
  - composition `total_warm_latency_us` and `single_warm_latency_us` removed.
  - selection, compare, and composition now use the single `latency_us`, measured after L2 sweep.
- Updated the manifest to use `[measurement] iters = 128` and threshold fields `latency_warn_pct` / `latency_fail_pct`.
- Added a short code comment to `L2CacheClear::clear` explaining why the benchmark uses a sweep kernel instead of CUDA's persisting-L2 reset API.

### Step 8: Slim manifest
- Simplified `kernel_manifests/qwen3.toml` to a flat multi-op sweep format:
  - `paged_decode_attention`: `batch_size`, `kv_len`, `variants`.
  - `paged_prefill_attention`: `batch_size`, `seq_len`, `variants`.
- Removed manifest-owned `parallel_strategy`, `hardware_class`, measurement defaults, provider shape constants, thresholds, and compositions.
- The `paged_prefill_attention` entry is deliberately single-request for now. Multi-request prefill should use `PrefillPagedPlan::new_batch`, but that is a separate packing/reporting decision.

### Step 9: Add paged prefill attention provider
- Added a `paged_prefill_attention` report path backed by the existing production `PrefillPagedPlan` and `prefill_attention_paged_into` kernel wrapper.
- Generalized the run path from decode-only specs to `KernelSpec::{Decode, Prefill}`.
- Added `--seq-lens` for prefill reports; `--contexts` remains accepted as a prefill length alias for quick CLI use.
- Changed per-case shape fields so `kv_len` and `seq_len` are optional op-specific fields instead of forcing decode shape into prefill.
- Removed report-owned CUPTI interpretation. The report now stores raw CUPTI metric names and values under `case.cupti`; diagnosis labels, derived bandwidth percentages, and read-amplification gates belong in offline analysis/reporting code.
- Added one unmeasured pre-launch before cold-L2 timing and recorded it as `measurement.pre_measure_launches = 1`. This avoids CUDA/FlashInfer lazy-init pollution without bringing back a warm-cache latency metric.
- Removed dead bench-era report code:
  - fixed hardcoded `CONTEXT_LENGTHS` / `BATCH_SIZES` sweep constants from `kernel_bench.rs`;
  - removed old split-K tuning sweep constants;
  - removed `default_attention_kernel_specs`;
  - removed the old multi-launch `measure_decode_only` helper and `INNER_LAUNCHES`.
- GPU validation in `<validation-worktree>`:
  - `PEGAINFER_CUDA_SM=120 cargo build --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report` passed.
  - `PEGAINFER_CUDA_SM=120 cargo build --release -p pegainfer-qwen3` passed.
  - Decode report, `bs=1,ctx=1024`, no CUPTI, `iters=3`: `2` cases, `0` errors, selected `split_kv_256x64`; measured `non_partition=45.739us`, `split_kv_256x64=20.480us`.
  - Prefill report, `bs=1,seq=128`, no CUPTI, `iters=3`: `1` case, `0` errors, `24.917us`.
  - Prefill report, `bs=1,seq=1024`, no CUPTI, `iters=3`: `1` case, `0` errors, `142.325us`.
  - Prefill report, `bs=1,seq=128`, CUPTI on, `iters=3`: `1` case, `0` errors, `latency_us=24.576`; `case.cupti` contains the raw configured metric names.
  - Prefill CUPTI self-compare passed with `warnings=0 failures=0`.

### Step 10: Keep raw CUPTI metrics
- Removed the `diagnosis` field from cases, selections, and composition output.
- Removed derived CUPTI fields such as `gpu_time_us`, `dram_gb_s`, `dram_peak_pct`, and `kv_read_over_dram_read_pct`.
- Removed the compare-time DRAM read-amplification gate. `compare` now gates only `latency_us`; metric interpretation is intentionally outside the runner.
- Bumped op report schema to `4` and composition report schema to `3`.
- GPU validation:
  - `PEGAINFER_CUDA_SM=120 cargo build --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report` passed.
  - `PEGAINFER_CUDA_SM=120 cargo build --release -p pegainfer-qwen3` passed.
  - Decode raw-CUPTI validation: `2` cases, `0` errors, schema `4`; `case.cupti` keys were exactly the configured CUPTI metric names.
  - Prefill raw-CUPTI validation: `1` case, `0` errors, schema `4`; `case.cupti` keys were exactly the configured CUPTI metric names.
  - Decode and prefill self-compare passed with `warnings=0 failures=0`.
  - Composition consumed the schema-4 decode report and wrote a schema-3 composition report.

### Step 11: Full raw-CUPTI cold-L2 manifest run
- Preserved full-run JSONs under `<kernel-report-dir>`.
- Decode full command:
  - `PEGAINFER_CUDA_SM=120 time cargo run --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report -- run --out <kernel-report-dir>`
  - Result: schema `4`, `126` cases, `42` selections, `0` errors, `126` CUPTI cases, `128` measured iterations, elapsed `2:11.77`.
  - Selection counts: `split_kv_256x64=22`, `non_partition=13`, `split_kv_512x64=7`.
  - `case.cupti` contains exactly the configured CUPTI metric names. Cases and selections do not contain `diagnosis`.
- Prefill full command:
  - `PEGAINFER_CUDA_SM=120 time cargo run --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report -- run --op paged_prefill_attention --out <kernel-report-dir>`
  - Result: schema `4`, `7` cases, `7` selections, `0` errors, `7` CUPTI cases, `128` measured iterations, elapsed `0:07.43`.
  - Latency by `seq_len`: `128=24.687us`, `512=53.467us`, `1024=143.462us`, `2048=318.688us`, `4096=911.097us`, `8192=3015.025us`, `10000=4316.861us`.
  - `case.cupti` contains exactly the configured CUPTI metric names. Cases and selections do not contain `diagnosis`.
- Both decode and prefill full JSONs passed self-compare with `warnings=0 failures=0`.
- Decode composition command:
  - `PEGAINFER_CUDA_SM=120 cargo run --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report -- compose --input <kernel-report-dir> --batch-size 1 --context 4096 --out <kernel-report-dir>`
  - Result: schema `3`, no `diagnosis`, total decode-attention-only contribution `958.527us`.

### Step 12: Split prefill stages
- Treated `prefill_attention_paged_into` as a wrapper over three separately measurable kernel stages:
  - `prefill_qk_norm_rope`: `qk_norm_rope_batched_decode_cuda` (per-token positions; the scalar `prefill_qk_norm_rope_only_cuda` wrapper was removed with the prefix-cache RoPE fix)
  - `prefill_kv_scatter`: `paged_kv_scatter_cuda`
  - `prefill_attention_core`: `batch_prefill_paged_cuda`
- Added `PrefillStage` launch paths in `kernel_bench.rs`. For stage reports, prerequisites run outside the timed/profiled launch and then L2 is swept before the measured stage. This keeps each stage report tied to one target kernel while still preparing valid inputs.
- Added the three stage ops to `kernel_manifests/qwen3.toml`; each currently covers `batch_size=[1]` and the same `seq_len` grid as the full prefill report.
- Preserved stage JSONs under `<kernel-report-dir>`.
- GPU validation:
  - `PEGAINFER_CUDA_SM=120 cargo build --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report` passed.
  - `PEGAINFER_CUDA_SM=120 cargo build --release -p pegainfer-qwen3` passed.
  - Full stage reports passed self-compare with `warnings=0 failures=0`.
- Stage latency by `seq_len`:
  - `128`: full `24.632us`, QK+RoPE `8.181us`, KV scatter `5.418us`, attention core `13.148us`.
  - `512`: full `53.280us`, QK+RoPE `18.412us`, KV scatter `6.136us`, attention core `34.822us`.
  - `1024`: full `142.055us`, QK+RoPE `30.703us`, KV scatter `7.619us`, attention core `110.177us`.
  - `2048`: full `318.263us`, QK+RoPE `57.084us`, KV scatter `10.493us`, attention core `258.477us`.
  - `4096`: full `908.824us`, QK+RoPE `108.435us`, KV scatter `16.344us`, attention core `797.264us`.
  - `8192`: full `3008.127us`, QK+RoPE `210.866us`, KV scatter `28.840us`, attention core `2781.202us`.
  - `10000`: full `4310.998us`, QK+RoPE `257.516us`, KV scatter `40.663us`, attention core `4031.471us`.
- At `seq_len=10000`, stage raw CUPTI summaries:
  - QK+RoPE: `257.516us`, `gpu__time_duration.sum=285.664us`, DRAM `203.183MB`, L2 `301.279MB`, SM throughput `51.880%`, active warps `88.793%`.
  - KV scatter: `40.663us`, `gpu__time_duration.sum=75.232us`, DRAM `65.342MB`, L2 `83.175MB`, SM throughput `1.900%`, active warps `78.045%`.
  - Attention core: `4031.471us`, `gpu__time_duration.sum=4043.648us`, DRAM `205.315MB`, L2 `6650.592MB`, SM throughput `41.410%`, active warps `16.020%`.
  - Full wrapper: `4310.998us`, `gpu__time_duration.sum=4320.128us`, DRAM `458.553MB`, L2 `7032.714MB`, SM throughput `38.654%`, active warps `21.252%`.
- Initial conclusion from the stage split: long-sequence prefill time is dominated by `batch_prefill_paged_cuda`, not QK+RoPE or KV scatter. The low full-wrapper SM throughput is largely the attention core's behavior; the core has very high L2 traffic and low active-warps percentage.

### Step 13: Compare paged and single prefill attention core at 10k
- Added report support for `single_prefill_attention_core`, backed by `single_prefill_cuda` with contiguous HND K/V buffers. This is a comparison point for the same single-request causal prefill attention core without paged KV indirection.
- Generalized `AttentionPrefillCase` to support `batch_size > 1` using `PrefillPagedPlan::new_batch`, so the current paged attention core can be profiled at `bs=2`.
- Preserved sequential JSONs under `<kernel-report-dir>`:
  - `paged_attention_core_bs1_seq10000.json`
  - `single_attention_core_bs1_seq10000.json`
  - `paged_attention_core_bs2_seq10000.json`
- The earlier concurrent run in `<kernel-report-dir>` is not used for conclusions because the paged `bs=1` wall latency was inflated by simultaneous GPU work.
- All three sequential JSONs passed self-compare with `warnings=0 failures=0`.
- Sequential results:
  - Paged `bs=1,seq=10000`: `3983.682us`, `gpu__time_duration.sum=4019.648us`, SM throughput `41.527%`, active warps `16.030%`, DRAM `205.001MB`, L2 `6650.173MB`.
  - Single contiguous `bs=1,seq=10000`: `3953.402us`, `gpu__time_duration.sum=4021.152us`, SM throughput `41.502%`, active warps `16.006%`, DRAM `204.169MB`, L2 `6646.446MB`.
  - Paged `bs=2,seq=10000`: `7483.966us` total, `3741.983us` per request, `gpu__time_duration.sum=7495.968us`, SM throughput `44.530%`, active warps `16.331%`, DRAM `409.032MB`, L2 `13298.354MB`.
- Conclusion: contiguous single-request FlashInfer is only `1.008x` faster than the paged path at 10k, so paged KV indirection is not the dominant cost. Moving paged attention core from `bs=1` to `bs=2` makes total latency `1.879x` of `bs=1` and improves per-request latency to `0.939x`; batching helps slightly, but active warps remain around `16%`, so the 10k bottleneck remains inside the attention-core kernel behavior.
- Final validation:
  - Local `cargo fmt --all --check` passed.
  - Local `cargo metadata --no-deps --format-version 1` passed.
  - Local `git diff --check` passed.
  - CUDA host `PEGAINFER_CUDA_SM=120 cargo build --release -p pegainfer-qwen3` passed.

### Step 14: Add direct tensor-path CUPTI metrics
- Investigated the RTX 5090 metric catalog with `/usr/local/cuda/bin/ncu --query-metrics-mode all --query-metrics --devices 0`. Non-interactive shells did not have `ncu` in `PATH`, but the binary exists under `/usr/local/cuda/bin/ncu`.
- Added direct tensor-path metrics to the default CUPTI list instead of adding runner-owned MFU fields:
  - `sm__cycles_elapsed.avg.per_second`
  - `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`
  - `sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed`
  - `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed`
  - `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.per_second`
- Preserved the tensor-metric validation JSON under `<kernel-report-dir>`.
- GPU validation for `prefill_attention_core bs=1 seq=10000` passed with `0` case errors. Key raw CUPTI values:
  - `latency_us=3978.180`
  - `gpu__time_duration.sum=4015.616us`
  - `sm__cycles_elapsed.avg.per_second=2.926GHz`
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed=41.561%`
  - `sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed=41.561%`
  - `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed=41.561%`
  - `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.per_second=211.727e12`
- Conclusion: for the kernel-level 10k prefill attention question, use the CUPTI/NVPerf BF16-HMMA peak-percentage metric directly. The earlier offline MFU estimate using an external RTX 5090 spec table is a useful sanity check, but it is not the report's primary answer.

### Step 15: NCU explanation for 10k prefill attention core
- Collected a targeted Nsight Compute profile for the second `BatchPrefillWithPagedKVCacheKernel` launch in `prefill_attention_core bs=1 seq=10000`:
  - Report: `<kernel-report-dir>`
  - Command used `/usr/local/cuda/bin/ncu` with `--kernel-name regex:BatchPrefillWithPagedKVCache`, `--launch-skip-before-match 1`, `--launch-count 1`, and sections `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`, `Occupancy`.
- Kernel identity:
  - `BatchPrefillWithPagedKVCacheKernel<KernelTraits<1,128,2,2,8,8,4,1,...>>`
  - Grid/block: `(313,1,8)x(32,4,1)`.
- Key NCU findings:
  - Compute throughput `42.34%`; memory throughput `20.84%`; DRAM throughput `1.71%`.
  - L2 hit rate `96.93%`; DRAM is not the limiter.
  - Achieved occupancy `16.02%`; theoretical occupancy `16.67%`.
  - Registers per thread `245`; shared memory per block `50.18 KiB`; occupancy is limited by registers and shared memory.
  - Active warps per scheduler `1.92` out of max `12`; eligible warps per scheduler only `0.20`.
  - Schedulers had no eligible warp `84.43%` of cycles and issued only `0.16` warp per scheduler per cycle.
  - Dominant stall: `Stall Math Pipe Throttle = 7.64` cycles per issued instruction, about `61.8%` of the average issue gap.
- Conclusion: the 10k attention-core performance is explained by a resource-heavy tensor-math tile. The kernel is not DRAM-bound and not primarily paged-KV-bound; it is limited by low occupancy/eligible-warps caused by high register and shared-memory use, with tensor pipe pressure concentrated in too few resident warps.

### Step 16: Inspect local FlashInfer prefill implementation
- Initialized the local FlashInfer submodule:
  - `git submodule update --init crates/pegainfer-kernels/third_party/flashinfer`
  - Checked out `779c24d1c9e6fcc51aa2359884696fbf4ac69b3b`.
- Confirmed the current PegaInfer wrapper calls FlashInfer's FA2 paged prefill path:
  - `crates/pegainfer-kernels/csrc/paged_attention.cu` computes `cta_tile_q = FA2DetermineCtaTileQ(packed_qo_len, head_dim)` and dispatches `BatchPrefillWithPagedKVCacheDispatched<CTA_TILE_Q, 128, 128, PosEncodingMode::kNone, false, MaskMode::kCausal, ...>`.
  - For Qwen3 `seq_len=10000`, `packed_qo_len = seq_len * (num_qo_heads / num_kv_heads) = 40000`, and FlashInfer's `FA2DetermineCtaTileQ` selects `CTA_TILE_Q=128`.
  - Rust prefill planning also calls `batch_prefill_cta_tile_q`, so any `CTA_TILE_Q` override must be plumbed into both the plan metadata and the kernel launch.
- Matched the NCU kernel traits to FlashInfer source:
  - `KernelTraits<1,128,2,2,8,8,4,1,...>` means causal mask, `CTA_TILE_Q=128`, `NUM_MMA_Q=2`, `NUM_MMA_KV=2`, `NUM_WARPS_Q=4`, `NUM_WARPS_KV=1`, and 128 threads/block.
  - `NUM_MMA_KV` is not a free runtime flag in the current wrapper. FlashInfer derives it from register and shared-memory limits with `DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), ...)`.
- Checked backend alternatives in the same local source:
  - FlashInfer's Python `backend="cutlass"` path explicitly rejects SM12x/RTX 5090 and says to use `backend='fa2'`.
  - `trtllm_fmha_v2_prefill` has SM120 support and supports `Q_PAGED_KV_HND`/`Q_PAGED_KV_NHD`, BF16, GQA, causal masks, and head dim 128, but it is exposed through FlashInfer's JIT/TVM FFI path (`fmha_v2_jit_binding.cu` + generated sources), not through the header-only FA2 path currently compiled by `pegainfer-kernels`.
- Tuning conclusion:
  - First experiment: add report-only `prefill_attention_core` variants for `fa2_cta128`, `fa2_cta64`, and `fa2_cta16`, keeping plan metadata and launch dispatch consistent, then run the 10k CUPTI/NCU comparison.
  - Second experiment, only after the tile-Q sweep: consider a more invasive FMHAv2/SM120 integration or a FlashInfer Python-side benchmark to determine whether that backend beats FA2 for Qwen3's `Q_PAGED_KV` BF16 shape before wiring it into Rust/CUDA.

### Step 17: Clarify `trtllm_fmha_v2_prefill`
- `trtllm_fmha_v2_prefill` is FlashInfer's Python API wrapper around TRT-LLM FMHA v2 kernels. It supports layouts including `Q_PAGED_KV_HND` and `Q_PAGED_KV_NHD`, BF16, GQA, causal masking, and SM120 code generation.
- The API requires:
  - `workspace_buffer`, initialized to zero on first use. FlashInfer tests use a reusable `128 MiB` `uint8` buffer.
  - `seq_lens`, `cum_seq_lens_q`, `cum_seq_lens_kv`.
  - `block_tables` shaped `[batch_size, max_num_pages_per_seq]` for paged KV.
  - `Q` shaped `[total_tokens, num_qo_heads, head_dim]`.
  - Paged KV shaped `[pages, 2, num_kv_heads, page_size, head_dim]` for HND or `[pages, 2, page_size, num_kv_heads, head_dim]` for NHD.
- The wrapper currently transposes `Q_PAGED_KV_NHD` to HND with `.transpose(-3, -2).contiguous()`. PegaInfer's KV pool is page-first NHD with separate K/V offsets across layers, so the zero-copy route would need either an HND view/storage path or a lower-level wrapper that bypasses the Python NHD transpose.
- Under the hood, FlashInfer generates FMHA v2 sources through `gen_fmha_v2_module(...)`, compiles generated kernels plus `fmha_v2_run.cu`, and exports `run` through TVM FFI (`fmha_v2_jit_binding.cu`). Direct Rust/CUDA integration is therefore not a drop-in header include like the current FA2 path.
- Practical use options:
  - Lowest risk: benchmark it through FlashInfer's Python API on RTX 5090 with Qwen3-equivalent tensors to decide whether it beats FA2 at `seq_len=10000`.
  - Medium effort: build a standalone C++ prototype that reuses generated FMHA v2 sources and constructs the required `TensorView` objects, accepting the TVM FFI dependency.
  - Higher effort: vendor a small generated-kernel subset and write a dedicated C ABI around FMHA v2 dispatch, avoiding Python at runtime but still adding nontrivial build integration.

### Step 18: Measure FlashInfer Python paths on RTX 5090
- Read FlashInfer's README and confirmed the intended package split:
  - `flashinfer-python` is the core package.
  - `flashinfer-cubin` provides precompiled cubins.
  - `flashinfer-jit-cache` provides prebuilt JIT cache packages for specific CUDA versions.
- The validation environment already has a uv-created venv at `<python-venv>`:
  - Python `3.13.11`, `uv 0.9.21`.
  - `torch 2.10.0+cu128`.
  - `flashinfer-python 0.6.6` and `flashinfer-cubin 0.6.6`.
  - The official package does not expose `trtllm_fmha_v2_prefill`; it exposes `get_trtllm_gen_prefill_module` and the public `BatchPrefillWithPagedKVCacheWrapper` API.
- Benchmarked local-source `trtllm_fmha_v2_prefill` through the uv venv and local FlashInfer checkout because the package API did not export it:
  - Report: `<kernel-report-dir>`
  - Shape: `seq_len=10000`, `page_size=16`, `num_qo_heads=32`, `num_kv_heads=8`, `head_dim=128`, BF16, causal, `128 MiB` workspace, CUDA event timing, `100` measured launches after `10` untimed launches.
  - Used a dedicated `FLASHINFER_WORKSPACE_BASE` under the report directory and explicitly set `FLASHINFER_JIT_DEBUG=0`, `FLASHINFER_JIT_VERBOSE=0`, `FLASHINFER_JIT_LINEINFO=0`.
  - Verified generated `build.ninja` files had `-O3` and did not contain `-O0`, `-G`, or `CUTLASS_DEBUG_TRACE_LEVEL`.
- `trtllm_fmha_v2_prefill` release-JIT results:
  - `Q_PAGED_KV_HND`, `bs=1`: median `6322.928us`, p95 `6591.840us`.
  - `Q_PAGED_KV_HND`, `bs=2`: median `11526.048us` total, `5763.024us` per request.
  - `Q_PAGED_KV_NHD`, `bs=1`: median `6343.856us`.
- Comparison with current production FA2 attention-core reports:
  - Current paged FA2 `bs=1,seq=10000`: `3983.682us`.
  - Current contiguous single FA2 `bs=1,seq=10000`: `3953.402us`.
  - Current paged FA2 `bs=2,seq=10000`: `7483.966us` total, `3741.983us` per request.
  - Local-source FMHA v2 HND is `1.587x` slower than current paged FA2 at `bs=1`; its `bs=2` total is `1.540x` slower than current paged FA2 `bs=2`.
  - NHD and HND are effectively the same latency in this direct function test (`1.003x` ratio), but the Python NHD path includes a contiguous transpose before the kernel and is therefore not a zero-copy integration path for PegaInfer.
- Also tested the official package public wrapper:
  - Report: `<kernel-report-dir>`
  - `backend="fa2"`, `kv_layout="NHD"`, `bs=1`: median `4095.216us`, close to our current FA2 path.
  - `backend="trtllm-gen"` initially hit a Python wrapper bug when passing `max_token_per_sequence` / `max_sequence_kv`: `UnboundLocalError: qo_indptr_host`.
  - Retrying without those max arguments reached the underlying TRT-LLM runner and failed for all tested `trtllm-gen` cases with `Unsupported architecture` on RTX 5090:
    `<kernel-report-dir>`.
- Conclusion: there is no measured reason to wire `trtllm_fmha_v2_prefill` into PegaInfer for the Qwen3 10k BF16 prefill attention core on RTX 5090. The current FA2 path is materially faster, and the official package's public `trtllm-gen` wrapper is not usable on this GPU through the tested release package.

### Step 19: Tune FA2 prefill CTA tile Q
- Added report-only FA2 prefill variants for `CTA_TILE_Q`:
  - `default`: FlashInfer heuristic, currently `128` for Qwen3 10k.
  - `cta_q128`, `cta_q64`, `cta_q16`: explicit tile override.
- Implementation details:
  - Added `batch_prefill_paged_cuda_with_cta_tile_q` and matching plan helpers so launch dispatch and `request_indices` / `qo_tile_indices` metadata use the same tile size.
  - Kept the original C ABI functions as auto-heuristic wrappers.
  - Exposed `PrefillPagedPlan::new_with_cta_tile_q` / `new_batch_with_cta_tile_q` through `pegainfer-core`.
  - Switched Qwen3 production prefill planning to model-local `PREFILL_ATTENTION_CTA_TILE_Q = 64`; the global FlashInfer heuristic is unchanged.
- Preserved tile-sweep JSONs under:
  - `<kernel-report-dir>`
  - `<kernel-report-dir>`
  - `<kernel-report-dir>`
  - `<kernel-report-dir>`
  - `<kernel-report-dir>`
  - `<kernel-report-dir>`
  - `<kernel-report-dir>`
- `prefill_attention_core`, `bs=1,seq=10000`, no CUPTI, `64` measured iterations:
  - `default`: `4012.176us`.
  - `cta_q128`: `4015.460us`.
  - `cta_q64`: `3783.602us`.
  - `cta_q16`: `7541.217us`.
  - Best: `cta_q64`, about `5.7%` faster than default.
- `prefill_attention_core`, all manifest seq lengths, no CUPTI:
  - `cta_q64` won all 7 sequence lengths.
  - Speedup vs default: `128=37.3%`, `512=0.05%`, `1024=55.1%`, `2048=18.2%`, `4096=10.9%`, `8192=7.0%`, `10000=6.0%`.
  - `cta_q16` is consistently bad for long seq and should not be selected.
- `paged_prefill_attention` full op, all manifest seq lengths, no CUPTI:
  - `cta_q64` won all 7 sequence lengths.
  - Speedup vs default: `128=17.1%`, `512=3.4%`, `1024=37.1%`, `2048=14.3%`, `4096=9.5%`, `8192=6.6%`, `10000=5.9%`.
  - At `seq_len=10000`, full op moved from `4305.936us` to `4067.570us`.
- CUPTI for `prefill_attention_core bs=1 seq=10000`:
  - `default`: `4007.283us`, `gpu__time_duration.sum=4056.064us`, BF16 HMMA peak metric `41.149%`, BF16 HMMA per-second `209.615e12`, active warps `16.030%`, DRAM `203.816MB`, L2 `6649.381MB`.
  - `cta_q64`: `3775.040us`, `gpu__time_duration.sum=3780.768us`, BF16 HMMA peak metric `44.397%`, BF16 HMMA per-second `224.878e12`, active warps `16.220%`, DRAM `204.687MB`, L2 `13212.309MB`.
  - Interpretation: q64 improves tensor-pipe utilization and wall time, even though L2 traffic increases. This is still compute/tensor-pipe limited, not DRAM limited.
- CUPTI for full `paged_prefill_attention bs=1 seq=10000`:
  - `default`: `4301.220us`, `gpu__time_duration.sum=4348.224us`, BF16 HMMA peak metric `38.453%`, active warps `21.233%`.
  - `cta_q64`: `4061.360us`, `gpu__time_duration.sum=4051.776us`, BF16 HMMA peak metric `41.259%`, active warps `21.371%`.
- `bs=2,seq=10000` check:
  - Attention core: default `7542.625us` total vs `cta_q64=7403.535us`, about `1.9%` faster.
  - Full paged prefill op: default `8241.567us` total vs `cta_q64=8113.489us`, about `1.6%` faster.
- GPU validation:
  - `PEGAINFER_CUDA_SM=120 cargo build --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report` passed.
  - `PEGAINFER_CUDA_SM=120 cargo build --release -p pegainfer-qwen3` passed.
  - `PEGAINFER_CUDA_SM=120 cargo test --release -p pegainfer-qwen3` was not used as the acceptance gate for this kernel-level change; Qwen3 correctness now relies on the tolerance-based HF golden gate rather than the old exact batch-vs-sequential token check.

### Step 20: Commit validation
- Fixed clippy cleanup found during commit prep:
  - `launch_qk_norm_rope` no longer returns `Result<()>` because it cannot report failure.
  - `qwen3_kernel_report` no longer uses `Option::map(...).unwrap_or_else(...)` or a redundant selector-key clone.
- Local checks passed:
  - `cargo fmt --all --check`
  - `cargo metadata --no-deps --format-version 1 >$RESULT_ROOT/pegainfer_metadata.json`
  - `git diff --check`
- GPU release clippy passed on the synced validation worktree:
  - `PEGAINFER_CUDA_SM=120 PEGAINFER_TRITON_PYTHON=<python-venv>/bin/python cargo clippy --release -p pegainfer-kernels --all-targets -- -D warnings`
  - `PEGAINFER_CUDA_SM=120 PEGAINFER_TRITON_PYTHON=<python-venv>/bin/python cargo clippy --release -p pegainfer-core --all-targets -- -D warnings`
  - `PEGAINFER_CUDA_SM=120 PEGAINFER_TRITON_PYTHON=<python-venv>/bin/python cargo clippy --release -p pegainfer-qwen3 --features kernel-report --all-targets -- -D warnings`
  - `PEGAINFER_CUDA_SM=120 PEGAINFER_TRITON_PYTHON=<python-venv>/bin/python cargo clippy --release -p pegainfer --bin pegainfer -- -D warnings`

### Step 21: Add runtime-traced Qwen3 model operator report
- Added non-enum tensor metadata vocabulary under `pegainfer-kernels/src/tensor.rs`:
  - marker traits for dtype, layout, and axis tags;
  - erased `TensorSpec`, `TensorArg`, `AttrSpec`, and `KernelCall` for schedules, reports, and future instrumentation.
- Added `pegainfer-core::ops::call_spec` builders so op TensorSpec construction lives next to op wrappers, not in the model-report CLI.
- Added `pegainfer-core::ops::call_trace` and traced op wrappers behind the `kernel-call-trace` feature. Normal Qwen3 builds re-export the direct kernel ops and compile out trace labels/recording. The `pegainfer-qwen3` `kernel-report` feature enables `kernel-call-trace`.
- Wired Qwen3 batch decode labels into the actual `batch_decode` path through feature-gated macros, so label formatting and trace collection disappear when the feature is off. The trace path forces CUDA Graph off before recording.
- Added `qwen3_model_report`, a feature-gated per-model CLI:
  - `cargo run --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_model_report -- decode --batch-size 16 --kv-len 2048 --format text`
  - The CLI loads `models/Qwen3-4B` by default, creates KV state for the requested decode context, runs one real `batch_decode` trace, microbenches the traced KernelCalls, and emits JSON under `target/model_reports/qwen3-4b/`.
  - Missing bench providers fail loudly with the missing `op`, `label`, and TensorSpec; no estimated or nearest-neighbor rows are used.
- Validation:
  - `cargo fmt --all --check` passed.
  - `git diff --check` passed.
  - `cargo check --release -p pegainfer-qwen3 --lib` passed, confirming trace wrappers are not required for the normal library path.
  - `cargo check --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_model_report` passed.
  - `cargo run --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_model_report -- decode --batch-size 16 --kv-len 2048 --format text` passed and wrote `target/model_reports/qwen3-4b/decode-bs16-kv2048.json`.
  - JSON audit confirmed `model/phase/config/schedule/by_op/by_call_site/coverage` are present, `schedule_source` is `runtime trace: Qwen3Model::batch_decode with CUDA Graph disabled`, and paged KV is represented as `bf16[page, layer, kv, pos_in_page, kv_head, head_dim] layout=paged_kv_page_first`.
  - `cargo test --release -p pegainfer-qwen3 --lib` passed: `7 passed`.

### Step 22: Add latency stats, readable tables, and DOT output
- `qwen3_model_report` now records latency samples per unique traced `KernelCall` instead of a single mean. JSON schema `2` includes `mean_us`, sample `stddev_us`, `min_us`, `p50_us`, `p95_us`, `p99_us`, and `max_us`.
- Default report iterations increased from `8` to `32` so p99 has enough samples to be useful for quick operator diagnostics.
- Text output now uses `comfy-table` for by-op, by-call-site, coverage, and compact schedule-preview sections. The detailed TensorSpec schedule stays in JSON.
- Each run writes a sibling Graphviz DOT file next to the JSON, for example:
  - `target/model_reports/qwen3-4b/decode-bs16-kv2048.json`
  - `target/model_reports/qwen3-4b/decode-bs16-kv2048.dot`
- Validation:
  - `cargo check --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_model_report` passed.
  - `cargo run --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_model_report -- decode --batch-size 16 --kv-len 2048 --format text` passed and wrote JSON plus DOT.
  - `dot -Tsvg target/model_reports/qwen3-4b/decode-bs16-kv2048.dot -o $RESULT_ROOT/qwen3-model-report.svg` passed.
  - JSON audit confirmed schema `2`, default `iters=32`, and `stddev_us` / `p99_us` fields in by-op rows.
  - `cargo fmt --all --check`, `git diff --check`, and `cargo check --release -p pegainfer-qwen3 --lib` passed.

### Step 23: Make decode tracing come from an eager DAG builder
- Added `pegainfer-qwen3/src/batch_decode_dag.rs`.
- `BatchDecodeDag` is eager: each method records the op's `KernelCall` contract when `kernel-call-trace` is active and immediately launches the production kernel. The batch decode forward path now calls DAG methods instead of wrapping free-form op calls with ad hoc trace labels.
- Normal builds still compile out label construction through `dag_label!`; the `kernel-report` feature enables labels and `KernelCall` recording.
- `all_reduce_hidden` now has an untraced internal path so the DAG builder owns decode all-reduce trace records without double-counting.
- Validation:
  - `cargo check --release -p pegainfer-qwen3 --lib` passed.
  - `cargo check --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_model_report` passed.
  - `cargo run --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_model_report -- decode --batch-size 16 --kv-len 2048 --format text` passed; by-op counts remained `gemm=109`, `gemm_rows=108`, `paged_decode_attention=36`, `all_reduce_hidden=72`.

### Step 24: Check eager-DAG decode runtime impact
- Ran current worktree decode-heavy request benchmark:
  - `cargo run --release --bin bench_serving -- --model-path models/Qwen3-4B request --prompt-len 1024 --output-len 256 --warmup 5 --iters 20`
  - Result: `steady_tpot_ms p50=11.33`, `p95=11.35`, `p99=11.45`, samples `5080`.
- Ran current worktree fixed-context decode probe:
  - `target/release/qwen3_decode_context --model-path models/Qwen3-4B --contexts 1024 --iters 20`
  - Result: `p50=11.3781ms`, `avg=11.6660ms`.
- Built a detached baseline worktree at `HEAD=3ffe745` and ran the same fixed-context decode probe on the same GPU/model:
  - `PEGAINFER_TRITON_PYTHON=$LOCAL_PEGAINFER_DIR/.venv/bin/python cargo run --release -p pegainfer-qwen3 --bin qwen3_decode_context --manifest-path $RESULT_ROOT/pegainfer-bench-baseline/Cargo.toml -- --model-path $LOCAL_PEGAINFER_DIR/models/Qwen3-4B --contexts 1024 --iters 20`
  - Result: `p50=11.3610ms`, `avg=11.6449ms`.
- Interpretation: eager DAG is not measurable as a decode overhead in this probe. Current worktree vs same-machine `HEAD` baseline is `+0.0171ms` p50 (`+0.15%`), well under the benchmark-regression `2%` TPOT threshold.
- The standard `bench_serving snapshot --warmup 5 --iters 20` could not complete because `prefill-heavy (10000,1)` hit CUDA OOM on this run. The existing tracked `bench_snapshots/rtx-5090/qwen3-4b.json` was restored from backup after the failed snapshot attempt.
- Existing snapshot data at `bench_snapshots/rtx-5090/qwen3-4b.json` still has `steady_tpot_ms p50=7.559606ms`, which is much faster than both current worktree and detached `HEAD` measurements on this machine. Since detached `HEAD` reproduces the same `~11.36ms` fixed-context number as the eager-DAG worktree, that older snapshot delta is not attributable to the eager DAG change.

### Step 25: PR readiness cleanup
- Cleaned release clippy findings surfaced by the model-report work:
  - `pegainfer-kernels::tensor::KernelCall` builder methods are now `#[must_use]`, and attrs are explicitly string-valued at the erased report boundary.
  - `pegainfer-kernels/build.rs` and `pegainfer-core/src/cpu_topology.rs` no longer trip release clippy on `map(...).unwrap_or_else(...)`, raw pointer borrows, or redundant closures.
  - Qwen3 e2e/regen tests and `qwen3_model_report` text/DOT rendering now pass the all-targets clippy style gates.
- Local checks passed:
  - `cargo fmt --all --check`
  - `git diff --check`
  - `cargo clippy --release -p pegainfer-qwen3 --features kernel-report --all-targets -- -D warnings`
  - `cargo test --release -p pegainfer-qwen3 --lib`
  - `cargo run --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_model_report -- decode --batch-size 16 --kv-len 2048 --format text`
- Qwen3-4B e2e was run with an absolute model path:
  - `PEGAINFER_TEST_MODEL_PATH=$LOCAL_PEGAINFER_DIR/models/Qwen3-4B cargo test --release -p pegainfer-qwen3 --test e2e -- --nocapture`
  - Current worktree produced greedy-output mismatches on repeated runs.
  - A detached baseline worktree at `HEAD=3ffe745` reproduced the same Kanye prompt mismatch, so this e2e failure is not introduced by the eager-DAG/model-report changes.

### Step 26: Full-forward manifest coverage + roofline rank pass (#456)
- Extended the manifest beyond the attention family to the full Qwen3-4B forward: `decode_projection_gemm` / `prefill_projection_gemm` (variants are the production `(out_dim, in_dim)` set that `decode_projection_pin_shapes` warms; gate/up share a shape; lm_head is decode-only because prefill gathers last-token columns before the logits GEMM), `{decode,prefill}_rms_norm`, `{decode,prefill}_fused_add_rms_norm`, `decode_qk_norm_rope`, `{decode,prefill}_silu_mul`, `{decode,prefill}_embedding`, and `decode_sampling` (`argmax` / `sampling` through `pegainfer_sample::select_batch`; its measured span includes the D2H token readback + sync — that is the production step tail, not harness overhead). One weight-free `DenseCase` harness drives all of them through the same `pegainfer_kernels::ops` entry points the decode DAG launches. `decode_*` ops sweep `batch_size`, `prefill_*` sweep `seq_len` at batch 1.
- Added the `rank` subcommand — the "separate report layer" the Step 10 lesson called for, now in-repo instead of a notebook. It consumes op-report snapshot JSONs and ranks selection winners by analytic speed-of-light: the op's minimum DRAM traffic and FLOPs (from its semantics and shape) against `hardware.peak_gb_s` and an explicit `--peak-bf16-tflops` (CLI-provided on purpose: no queryable CUDA attribute exists and a baked-in per-arch table would rot). `sol%` is a true headroom bound regardless of how the kernel is written; raw CUPTI metrics, when present in the snapshot, add achieved-DRAM and tensor-pipe columns for diagnosis. Dense projection/sampling variants are distinct kernels, not competing implementations, so their variant is folded into `selector_key` and `build_selections` never collapses them.
- Reproducible commands (H100 80GB / sm_90 validation; CUPTI unavailable there — `RmProfilingAdminOnly: 1` on the host, so `--no-cupti` and the analytic bound carried the run):
  - `cargo run --release -p pegainfer-qwen3 --features kernel-report --bin qwen3_kernel_report -- run --op decode_projection_gemm --no-cupti --out reports/decode_projection_gemm.json` (per op)
  - `... -- rank --input reports/<op>.json ... --peak-bf16-tflops 989 --out reports/rank_decode.json`
- H100 findings (cold L2, full manifest, 128 iters): sanity anchors land where they should — decode lm_head 84–87% of DRAM SoL, split-KV decode attention 79–84% at large KV, big prefill GEMMs 78–85% of tensor peak, `fused_add_rms_norm` 86% at 10k tokens. Ranked offenders at production shapes: FlashInfer `sampling` has a ~95µs flat cost rising to 722µs at bs=256 — ~8-12% of a decode step (measured on peaked synthetic logits, top-1 mass ~0.5; a flat vocabulary overstated it 35-55% because the rejection sampler's round count tracks distribution flatness) — and `argmax`, insensitive to the distribution, reaches 103µs / 22% SoL at bs=256; `prefill_embedding` sits flat at ~27% of DRAM SoL (113µs vs 31µs bound at 10k); decode q_proj GEMMs pin at 17.9–21.5µs / 32.0–35.6% SoL and o_proj at 20.2–25.1µs / 28.9–32.3% across every batch size (gate_up 43.7–51.7%, down 39.3–46.8%) — the small-N cuBLAS territory the tracking issue flagged. Codex review caught that an untuned context falls back to GemmEx at N ≤ `GEMM_LT_MAX_N`, so `DenseCase` now runs `gemm_lt_tune` for the projection shape (L2-cold weight rotation, budget-capped) before measuring; measured tuned-vs-GemmEx on H100: q_proj −7.8…−10.6% and lm_head −2.7…−3.4% at N ≤ 16, N > 32 unchanged (GemmEx either way). An early rotation capped at 8 copies left small weights L2-resident during tuning and picked a +5% slower kv_proj algo at bs 4–8; sizing the rotation off the L2 sweep (`cache_clear_bytes`) removed that anomaly — the tuner must be DRAM-cold like the executor's 36-layer rotation; prefill attention core at 10k is 33% of tensor peak on H100 (the Step 15 occupancy analysis, worse than the 42% seen on RTX 5090). Sub-20µs elementwise cases rank at the bottom on launch floor, not kernel quality — read `sol%` together with `latency_us`.
- CI gate decision (issue ask 4): keep roofline ranking out of CI. `sol%` is hardware-, driver-, and clock-dependent, the GPU runners differ per contributor, and the bench-regression convention already covers E2E gates. The intended workflow is manual: regenerate snapshots on the target GPU, `rank`, and `compare` against a stored snapshot for latency regressions.

## Debrief

- **Outcome**: Replaced the Qwen3 kernel snapshot bench with a normal, feature-gated `qwen3_kernel_report` binary and added `qwen3_model_report` for runtime-traced decode operator reports. The runner now covers paged decode attention, paged prefill attention, split prefill stages, and contiguous single-request prefill attention core with cold-L2 latency only. GPU manifest runs passed for decode/prefill, targeted 10k stage comparisons passed for paged `bs=1`, single contiguous `bs=1`, and paged `bs=2`, FlashInfer Python/package alternatives were measured for the same Qwen3 10k BF16 prefill attention-core shape, Qwen3 production prefill now uses measured FA2 `CTA_TILE_Q=64`, model-level decode reports now emit latency stats, readable tables, and Graphviz DOT, and decode tracing now comes from the same eager DAG builder that launches the kernels.
- **Pitfalls encountered**:
  - `cargo bench` appends a trailing `--bench` argument to `harness=false` binaries. That made the new `clap` parser fail and confirmed this should be a real bin, not a bench target.
  - A normal `src/bin` target cannot use dev-dependencies. Making report-only dependencies ordinary non-optional dependencies would pull CUPTI into default Qwen3/server builds, so the bin now uses `required-features = ["kernel-report"]`.
  - The first attempted full-run command removed `$RESULT_ROOT/qwen3_kernel_report_full.json` before running. Future profiling commands should not delete prior JSONs; save new reports under a timestamped or explicit report directory.
  - The validation checkout is dirty and lacked the local commit object, so validation used a bundle-derived clean worktree plus rsync of current changes.
  - FlashInfer's JIT environment treats `FLASHINFER_JIT_VERBOSE=1` as a debug build switch for backward compatibility. The first FMHA v2 cache contained `-O0 -G`; those results were discarded, and the release measurement used a separate cache directory with explicit debug flags disabled.
  - The official `flashinfer-python 0.6.6` package did not expose `trtllm_fmha_v2_prefill`. Its public `trtllm-gen` wrapper also has a plan-argument bug with explicit max lengths and then reports unsupported architecture on RTX 5090 once that is worked around.
  - The temporary validation worktree does not have model weights under the default model path, so model-loading tests fail with `No such file or directory`. For kernel-report work, keep using report binaries and release builds unless a model path is explicitly staged.
  - The local Qwen3-4B e2e greedy baseline can fail on `HEAD=3ffe745` with the same Kanye prompt mismatch seen in this worktree. Treat that as a separate baseline/test-data issue, not as evidence against the eager-DAG/model-report PR.
- **Lessons learned**:
  - Kernel reporting should be treated as tooling, not benchmarking harness plumbing. Cargo bench's hidden behavior is a poor fit for a manifest-driven CLI.
  - Report tooling that needs profiler libraries should be feature-gated from the production model crate dependency graph.
  - Full manifest profiling is cheap enough for Qwen3 paged decode attention on the CUDA validation host with the previous grid: about three minutes for 126 CUPTI-covered cases.
  - Warm-cache latency is not a useful selector metric for paged decode attention because the production path is dominated by KV reads that should be treated as cold or streaming.
  - Metric interpretation should stay out of the runner. Store raw CUPTI values in the JSON, then compute bandwidth, utilization labels, or IO-model checks in a separate report/notebook layer.
  - The prefill wrapper should stay reportable as a whole, but stage reports are necessary for actionable optimization. Whole-op CUPTI hid that `batch_prefill_paged_cuda` dominates long prompts.
  - At `seq_len=10000`, switching from paged to contiguous single-request FlashInfer barely changes latency or CUPTI counters. The next useful work is attention-core kernel investigation or alternative kernels, not removing page metadata.
  - Prefer CUPTI/NVPerf's tensor-path peak metrics over runner-side MFU calculations for kernel reports. The runner should collect the raw metric name/value; interpretation stays in the human report.
  - A 42% tensor peak can still be the expected result for this FlashInfer tile: NCU shows 245 registers/thread and 50 KiB shared memory/block limit occupancy to 16%, leaving too few eligible warps to keep tensor pipes full.
  - FlashInfer's current FA2 heuristic picks `CTA_TILE_Q=128` for Qwen3 10k prefill. Testing smaller Q tiles is the first low-friction kernel-level tuning knob because it changes register/shared-memory pressure without replacing the backend.
  - FlashInfer FMHA v2 is not a shortcut for RTX 5090 Qwen3 10k prefill attention. The release-JIT local-source path is slower than FA2, and the official package path either falls back to comparable FA2 performance or cannot run `trtllm-gen` on this GPU.
  - `CTA_TILE_Q=64` is the right Qwen3 FA2 prefill default on RTX 5090 for the measured single-request grid: it wins every tested seq length and materially improves the 10k kernel by about 6%. The batched `bs=2` case still benefits, but less, so future selector tables should keep batch size in the key.
- **Follow-ups**:
  - ~~Add the next provider for decode linear/MLP floor so composition reports explain more than decode attention.~~ Done in Step 26 (#456): the manifest now covers GEMMs, norms, embedding, and sampling for both phases.
  - Extend prefill tile-Q measurements to more batch sizes and model families before making a global kernel default.
  - Extend `compose` beyond `decode_attention_only` now that per-op reports exist for the full forward.
  - **Done (2026-05):** the model-agnostic layer was extracted into the `pegainfer-bench` crate — first the timing loop / latency stats / `KernelCall` accessors, then the by-op/by-call-site rollup (`RollupRow`, `CallSiteRow`, the accumulator, and row projection). Both qwen3 and Kimi-K2 model reports reuse it; row field order is preserved so the report JSON is unchanged. The *regression* framework (TOML manifest, `KernelSnapshot`, git/hardware/build provenance, `RegressionThresholds`, compare/compose, CUPTI, cold-L2) was evaluated and deliberately **kept** in `qwen3_kernel_report`: its schema is attention-domain-specific (`CaseShape` carries head/page dims, `CaseParams` carries chunk/`cta_tile_q`) with no second consumer, so sharing it would be premature abstraction. When a second model needs kernel-level regression gating, lift the generic primitives then — not before.
