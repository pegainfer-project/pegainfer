# Initial Ascend (CANN) support for Qwen3

**TL;DR**: Qwen3-0.6B/4B BF16 runs on an Ascend 910B3 through `pegainfer-ascend` (native Rust + AscendCL/ACLNN), using single-device eager execution, greedy decoding, fixed KV slots, chunked prefill, and mixed batches. Both models pass the shared HF golden and HTTP lifecycle gates; the device gates are manual and nothing Ascend runs in the repository's CI. The first release is TP1. The M1 runtime pass adds cached workspaces, one synchronization per forward, grouped prefill through PromptFlashAttentionV3, repeatable executors for norm/embedding/argmax, AddRmsNorm and cross-layer fusion, a fused gate/up GEMM with ClippedSwiglu, and FRACTAL_NZ weights: 0.6B golden drops from 143s to 54s and the serving benchmark reaches about 141 tokens/s with 5.5ms TPOT on the dev pod. Host RSS stays flat over a 1000-step soak.

Last touched: 2026-11

## Supported profile

| Area | Current support |
| --- | --- |
| Platform | Linux aarch64, CANN 9.2.1, Ascend 910B3 (`Ascend910_9362`); one die with `ASCEND_RT_VISIBLE_DEVICES=0` |
| Models | Qwen3-0.6B and Qwen3-4B, original BF16 safetensors weights |
| Execution | TP1, eager ACLNN operators, BF16 in/out with FP32 accumulation |
| Scheduling | Fixed KV slots, queued excess requests, chunked prefill and mixed prefill/decode batches |
| Selection | Greedy; explicitly set HTTP `temperature=0` or `top_k=1` |
| HTTP | Completions, chat, SSE, usage, stop strings/EOS, cancellation, and metrics |
| Context | Default 4096, configurable 1–8192 within checkpoint limits; serving gates run at 512 |
| Capacity | Slots default 4 (1–16); chunks default 128 (1–512) |
| Unsupported | CANN graph, TP/HCCL, random sampling, logprobs/prompt scoring, prefix cache, paged KV, offload/P-D, LoRA, speculative decoding, quantization, and other model families |

This backend always runs eager. If the shared `cuda_graph=true` default is present, startup logs that it is ignored; explicitly requesting `--cuda-graph=true` rejects startup. Prefix caching is always disabled, and supplying `--no-prefix-cache` logs that the flag is redundant. Unsupported HTTP sampling, scoring, and transfer parameters receive an explanatory 400 before an SSE stream opens.

The v1 attention path runs decode-style `aclnnIncreFlashAttentionV4` once per query row against the request's KV slot. It is a correctness baseline, not an optimized prefill implementation; grouping prefill rows per slot with `aclnnPromptFlashAttentionV3` is the next performance step.

## Build and run

Prerequisites are the Ascend driver/firmware, CANN 9.2.1 with `atc`/`bisheng`/ACLNN op packages, and the pinned Rust toolchain. Source the CANN environment before building or running:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0

cargo build --locked --release \
  -p pegainfer-server --no-default-features --features ascend
```

The build links `libascendcl`, `libopapi`, and `libnnopbase` from `$ASCEND_HOME_PATH/aarch64-linux/lib64`. The crate declares the FFI by hand, so `cargo check -p pegainfer-ascend --features ascend` works on a machine without CANN; only linking needs the SDK.

Checkpoints used by the gates:

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir /opt/atomgit/models/Qwen3-0.6B
modelscope download --model Qwen/Qwen3-4B   --local_dir /opt/atomgit/models/Qwen3-4B
```

The upstream CUDA/ROCm gates pin HF revisions `c1899de289a04d12100db370d81485cdf75e47ca` (0.6B) and `1cfa9a7208912126459214e8b04321603b3df60c` (4B). The local gates used the ModelScope `master` snapshots; record the resolved commit when re-qualifying.

Serve:

```bash
cargo run --locked --release \
  -p pegainfer-server --no-default-features --features ascend -- \
  --model-path /opt/atomgit/models/Qwen3-4B --served-model-name qwen3-ascend \
  --port 18302 --ascend-slots 4 --ascend-max-context 512 --ascend-chunk-size 128
```

## Runtime contracts

The `device` module of `pegainfer-ascend` owns the AscendCL context, stream, and buffers. A mutex serializes submissions and rebinds the context on the calling thread. Host copies synchronize before returning; buffer destruction synchronizes before free. A partially failed model forward poisons the model, preventing reuse of partially updated KV.

Executor ownership follows CANN. The non-repeatable executor returned by an ACLNN `*GetWorkspaceSize` call is released by the framework after the op runs, so the backend never destroys it (destroying it double-frees on CANN 9.2.1). GEMM/Add/Silu/Mul cache a repeatable executor instead; `Device::drop` releases those executors first, then their tensor descriptors, cached IFA descriptors/lists, int arrays, scratch, the stream, and the context. `aclrtResetDevice`/`aclFinalize` stay out of the drop path: tests create several short-lived devices in one process and CANN 9.2.1 can hang in finalize at exit. A regression test runs every operator repeatedly, drops the device, and opens a fresh one.

The `ops` module wraps ACLNN. Linear weights are packed on the host into FRACTAL_NZ `[1, ceil(out/16), ceil(in/16)*16, 16]` storage and run through `aclnnMatmulWeightNz`; the embedding gather weight stays ND. The KV cache is laid out as `[slot, kv_head, context, dim]` so a slot view is directly usable as the PFA/IFA BNSD cache; the language head packs only the first `decodable_vocab` rows when the tied embedding is transposed, because `decodable_vocab` (151669) is smaller than `vocab_size` (151936). RoPE is `aclnnApplyRotaryPosEmbV2` in BSND `half` mode, with cos/sin rows gathered from a `[context, dim]` table.

## M1 runtime optimizations

- **Workspace reuse.** `Device` caches ACLNN workspaces by byte size and frees them at device teardown; ops no longer call `aclrtMalloc`/`aclrtFree` per submission.
- **One sync per forward.** Operators no longer synchronize individually. Descriptors are deferred and released after a single `Device::flush_deferred()` at the end of each `Model::forward`; `Buffer::read` also synchronizes at the sampling/test boundary.
- **Grouped prefill attention.** Consecutive rows of one slot run through one `aclnnPromptFlashAttentionV3` call with `actualSeqLengths`/`actualSeqLengthsKv` and a 2048-or-larger int8 causal mask built at load. Q/out stay row-major and are presented as BNSD through explicit strides; single-row decode groups still use IFA.
- **Executor cache (partial).** norm, embedding and argmax cache one repeatable executor per `(op, shape, addresses)`; GEMM-family ops already did. RoPE, IndexCopy and IFA still create one-shot executors per call.
- **Fused normalization and MLP.** residual add + RMSNorm uses `aclnnAddRmsNorm` (also across layer boundaries and before the final norm); gate/up weights are concatenated into `[h, 2i]` and activated with `aclnnClippedSwiglu`.
- **Benchmark.** `scripts/ascend/bench_serving.py` measures TTFT, TPOT and tokens/s against the release server.

The Qwen3 scheduler reuses the production `RequestLedger` and step contract. Each request owns a contiguous KV slot; reclaimed slots restart at position 0. Fixed-slot KV storage is `slots × layers × kv_heads × context × head_dim × 2(K,V) × 2(BF16)` bytes, with the kv-head-major layout above. Cancellation reclaims slots while retaining the allocation for reuse.

CUDA remains the default backend. The Ascend line lives in its own crate and joins the server through `ModelLine` behind the server's `ascend` feature; the CUDA crates carry no backend features, so an Ascend build with default features off never compiles them.

## Verification

All gates are manual. They were last run on Ascend 910B3 / CANN 9.2.1 / aarch64.

```bash
# Operator contracts (GEMM, RMSNorm, SiLU/Mul/Add, embedding, RoPE, attention)
# plus the executor-lifecycle/teardown regression
cargo test --release -p pegainfer-ascend --features ascend \
  --test contract -- --ignored --test-threads=1

# Committed HF golden, 0.6B and 4B
PEGAINFER_TEST_MODEL_PATH=/opt/atomgit/models/Qwen3-0.6B \
  cargo test --release -p pegainfer-ascend --features ascend \
  --test golden -- --ignored --test-threads=1

# Decode soak: host RSS must stay flat (CANN owns the one-shot executors)
PEGAINFER_TEST_MODEL_PATH=/opt/atomgit/models/Qwen3-0.6B \
  PEGAINFER_SOAK_STEPS=1000 \
  cargo test --release -p pegainfer-ascend --features ascend \
  --test soak -- --ignored --test-threads=1

# Serving benchmark: TTFT / TPOT / tokens/s
python3 scripts/ascend/bench_serving.py \
  --model-path /opt/atomgit/models/Qwen3-0.6B \
  --output target/ascend-bench-06b --requests 4 --max-tokens 32

# Release HTTP lifecycle (0.6B and 4B)
python3 scripts/ascend/check_http.py \
  --model-path /opt/atomgit/models/Qwen3-0.6B \
  --output target/ascend-http-06b --context 512
```

HF golden results, using the CUDA gate's bounds (`MEAN_TOL=0.06`, `P99_TOL=0.20`):

| model | case | mean | p99 |
|---|---|---|---|
| 0.6B | sequential | 0.0403 | 0.150 |
| 0.6B | ragged batch4 | 0.0398 | 0.150 |
| 0.6B | mixed prefill/decode | 0.0401 | 0.145 |
| 4B | sequential | 0.0308 | 0.107 |
| 4B | ragged batch4 | 0.0313 | 0.108 |
| 4B | mixed prefill/decode | 0.0322 | 0.113 |

The HTTP gate covers health/models, completions, streaming concurrency, stop strings, chat, unsupported-parameter 400s, context boundaries, queued and active cancellation, slot re-admission, idle metrics, and shutdown. On SIGINT the server drains the scheduler, logs `scheduler drained after frontend shutdown, exiting`, exits 0, and leaves no NPU process; the script requires that path. CANN installs its own SIGINT handler after the process has been up for a few minutes, so the frontend re-installs its handler every 100 ms; otherwise CTRL+C would become a signal exit after a few minutes of uptime. A request that outlives the shared frontend shutdown deadline is aborted with a 500, which is frontend policy rather than an Ascend difference.

## CANN 9.2.1 notes

These were reproduced in minimal C++ programs and the Rust teardown test, independent of the serving path:

- **Non-repeatable executors belong to CANN.** The framework releases the executor returned by `*GetWorkspaceSize` after the op runs; calling `aclDestroyAclOpExecutor` on it double-frees (reproduced for MatMul with a non-zero workspace, AddRmsNorm and IFA). `aclnn-smoke/smoke_exec_lifecycle.cpp` runs 1000 iterations without destroying and host RSS stays flat, so not destroying leaks nothing.
- **Repeatable executors are not reclaimed.** GEMM/Add/Silu/Mul set `aclSetAclOpExecutorRepeatable` and cache the executor; CANN keeps it until the caller destroys it, which the backend does when the device drops.
- Destroying the IFA `aclTensor` descriptors between calls crashes the next call, so they are cached by address. A tensor list and a tensor it contains share a free path: destroying both double-frees, so only the standalone query/output tensors are listed for teardown while the K/V lists own their members.
- `aclrtDestroyContext` is safe after the stream and retained resources are released. `aclrtResetDevice` / `aclFinalize` are not called from `Device::drop`: unit tests create several short-lived devices in one process, and CANN 9.2.1 can hang in finalize at exit.
- `aclnnMul` with `out == self` traps; the fused gate/up path uses `aclnnClippedSwiglu` with a high clip limit, which matches SiLU*up.
- `aclnnAddRmsNorm` and `aclnnClippedSwiglu` are used as non-repeatable one-shot executors and are not destroyed (destroying the AddRmsNorm executor aborts on CANN 9.2.1).
- PFA V3 executor destruction also double-frees; the backend leaves PFA executors to CANN.
- FRACTAL_NZ weights are packed on the host at load; `aclnnMatmulWeightNz` requires the NZ storage layout, not an ND tensor.

## Outstanding

- QKV projection fusion and `qkv_rms_norm_rope_cache`: the fused QKV output needs strided downstream norm/RoPE/cache support; the copy-based experiment was slower and was reverted.
- Decode attention batching across slots, LM-head + ArgMax fusion, and the remaining executor caches (RoPE/IndexCopy/IFA).
- Batched KV append needs a transpose or a contiguous cache layout because IndexCopy rejects the strided multi-row form.
- Paged KV, prefix cache, sampling/logprobs, CANN graph, TP/HCCL, and other model lines.
- No repository CI; keep the device gates manual.
