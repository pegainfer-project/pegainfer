# Qwen Batched Sampling

> **TL;DR:** Issue #284 migrates Qwen3/Qwen3.5 mixed greedy/non-greedy token selection to one batched FlashInfer sampling call per step, while keeping greedy/effectively-greedy rows on batched argmax. The #284 evidence remains correctness/nsys/direct-bench evidence; #412 adds separate HTTP serving evidence in [qwen-mixed-sampling-http.md](../../benchmarks/qwen-mixed-sampling-http.md).
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - routed this task to Qwen model docs, scheduler docs, kernels docs, and profiling playbook.
  - `docs/models/qwen3/roadmap.md` - records #284 as phase 2 after batched greedy sampling.
  - `docs/models/qwen35/roadmap.md` - treats sampling as shared with Qwen3 and gated by HF logits tests.
  - `docs/subsystems/scheduler/scheduler.md` - identifies the old O(batch) per-row sampler as the scheduler hot-path issue.
  - `docs/playbooks/profiling-guide.md` - nsys trace must use release builds and `--cuda-graph-trace=node` for composition proof.
- **Relevant history**:
  - `docs/models/kimi-k2/sampling.md` - Kimi already uses the desired split: greedy rows stay on argmax, non-greedy rows use one batched FlashInfer pass.
- **Plan**:
  1. Rework shared Qwen token selection to compact non-greedy rows and call `gpu_sample_batch_into` once per step with a u64 seed.
  2. Remove legacy single-row sampling wrappers and the custom single-block softmax kernel.
  3. Update Qwen3/Qwen3.5 scratch, seed flow, tests, kernel plans, and roadmap docs.
  4. Validate with local Rust/CUDA checks, then run HF gates and an nsys composition proof on the validation machine.
- **Risks / open questions**:
  - Qwen3.5 still has single-logit prefill/unified helper paths; they need one-row batched sampling without changing request lifecycle behavior.
  - Kimi still uses FlashInfer top1 helpers, so top1 scratch sizing must remain available even after removing the single-row sampler.

## Execution Log

### Step 1: Shared sampling API and kernel cleanup

- Removed the legacy single-row FlashInfer wrapper surface from Rust exports.
- Deleted the custom single-block `logits_to_probs_kernel` and `gpu_sample_flashinfer_cuda` entry point from `pegainfer-kernels/csrc/shared/flashinfer_sampling.cu`.
- Kept the batched `gpu_sample_batch_flashinfer_cuda` path and moved top1 row-state scratch sizing to `flashinfer_top1_row_states_bytes_cuda()`.

### Step 2: Qwen decode token selection

- Qwen3 plan/executor paths now carry one `sample_seed: u64` per step instead of per-row `f32` seed material.
- Qwen3 and Qwen3.5 batch decode use `select_batch_tokens_into`: greedy rows go through indexed batched argmax, non-greedy rows compact into one batched FlashInfer sampling call.
- Qwen3.5 single-logit prefill/unified helpers now use `argmax` for greedy params and a one-row `gpu_sample_batch_into` call for non-greedy params.
- `top_p <= 1 / vocab` is treated as effectively greedy in the shared selector. That preserves the mathematical single-token nucleus semantics and avoids FlashInfer top-p boundary behavior on high-entropy rows.

### Step 3: Public wrappers, tests, and docs

- Server ops/tests/benchmarks no longer re-export or call the deleted single-row sampler.
- Kimi top1 scratch-size calls were renamed to `flashinfer_top1_row_states_bytes` while leaving Kimi sampling behavior unchanged.
- Kernel plans, roadmap docs, and the scheduler index now describe #284 as the batched Qwen sampling path.
- `gpu_sample_batch_flashinfer_cuda` chooses the matching FlashInfer primitive for a compacted batch: plain sampling when no top-k/top-p filter is active, TopP when only top-p is active, and TopKTopP when any row uses top-k.
### Step 4: Validation

- Local static gates passed: `cargo fmt --all --check`, `git diff --check`, and the old-symbol grep for `gpu_sample_flashinfer_cuda`, `logits_to_probs_kernel`, `u64::from(...to_bits)`, and `FLASHINFER_TOPK_ROW_STATES_BYTES`.
- Validation machine gates passed:
  - `cargo check --release --offline -p pegainfer-core -p pegainfer-kernels`
  - `cargo check --release --offline -p pegainfer-qwen3`
  - `cargo check --release --offline -p pegainfer-qwen35 --features qwen35`
  - `cargo test --release --offline -p pegainfer-kernels batch_sampling_top_p_only_small_nucleus_collapses_to_argmax -- --nocapture`
  - `cargo test --release --offline -p pegainfer-qwen3 --tests --no-run`
  - `cargo test --release --offline -p pegainfer-qwen3 --test hf_golden_gate -- --nocapture`
  - `cargo test --release --offline -p pegainfer-qwen3 --test sampling_behavior -- --nocapture`
  - `cargo test --release --offline -p pegainfer-qwen35 --features qwen35 --test hf_golden_gate -- --nocapture`
  - `cargo test --release --offline -p pegainfer-qwen35 --features qwen35 --test sampling_behavior -- --nocapture`
  - `cargo test --release --offline -p pegainfer-qwen35 --features qwen35 unified_step_decode_matches_graph_decode -- --nocapture`
  - `cargo test --release --offline -p pegainfer-qwen35 --features qwen35 --test e2e_scheduler -- --nocapture`
- Qwen3.5 scheduler e2e now alternates greedy and sampled requests in the concurrent phase, so the real decode batch covers mixed token-selection rows.
- Qwen3.5 `sampling_behavior` now proves `temperature` / `top_k` / `top_p` still steer the scheduler path, `top_k=1` and tiny `top_p` collapse to greedy, and hot sampling still varies across runs.
- `nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --export=sqlite` on the Qwen3.5 mixed scheduler test showed:
  - `logits_to_probs_kernel`: 0
  - `gpu_sample_flashinfer_cuda`: 0
  - `gather_cast_logits_f32_kernel`: 100
  - `OnlineSoftmaxMapKernel`: 100
  - `OnlineSoftmaxReduceKernel`: 100
  - `TopKTopPSamplingFromProbKernel`: 100
- Qwen3 greedy TPOT A/B on the same validation host, same model, same `request --prompt-len 4096 --output-len 64 --warmup 5 --iters 20 --seed 42` workload:
  - upstream/main steady TPOT p50: `6.537992ms`; p99: `6.561389ms`; output hash: `83f4c3f2614d57b5`.
  - patched steady TPOT p50: `6.508904ms`; p99: `6.540855ms`; output hash: `83f4c3f2614d57b5`.
  - p50 delta: `-0.44%`, within the 2% regression gate.
- `vllm bench serve` was not available in the validation environment, so no HTTP production-serving benchmark is claimed here.

## Debrief

- **Outcome**: The legacy single-row FlashInfer sampler and custom one-block softmax are removed from the runtime surface. Qwen3/Qwen3.5 batch decode now share the compacted non-greedy row path through `gpu_sample_batch_into`, and greedy rows remain on indexed batched argmax.
- **Pitfalls encountered**:
  - The validation copy initially had an uninitialized FlashInfer submodule; the release check failed on missing `flashinfer/sampling.cuh`. Initializing the pinned submodule fixed the environment and the same checks then passed.
  - The validation machine had Qwen3.5 weights but no Qwen3-4B weights. The Qwen3-4B snapshot was fetched before running the Qwen3 gates.
  - Qwen3.5 on the validation GPU required a Triton 3.4 Python for feature builds; older Triton rejected the SM target before the patch was exercised.
  - A Qwen3 test helper initially missed the `RngExt` import needed for `rng.random()`. The full Qwen3 integration compile caught it.
  - Qwen3 `sampling_behavior` caught a real regression: using TopKTopP for a top-p-only tiny nucleus did not match greedy on high-entropy rows. The final patch routes top-p-only batches through the matching FlashInfer primitive and treats `top_p <= 1 / vocab` as argmax.
  - Qwen3.5 unified decode needed a second pass after review because it still extracted decode logits row-by-row before sampling. The final version samples active decode rows from the graph logits buffer in one batched selector call.
- **Lessons learned**:
  - For scheduler hot-path changes, make the e2e test exercise mixed params directly; otherwise a passing greedy e2e does not prove the #284 acceptance path.
  - Keep nsys evidence as kernel composition proof, not a throughput claim. The current evidence proves the old per-row softmax wrapper is gone and the batched FlashInfer sequence is used for mixed Qwen3.5 decode.
