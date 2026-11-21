# Initial Ascend (CANN) support for Qwen3

**TL;DR**: Qwen3-0.6B/4B BF16 runs on an Ascend 910B3 through `pegainfer-ascend` (native Rust + AscendCL/ACLNN), using single-device eager execution, greedy decoding, fixed KV slots, chunked prefill, and mixed batches. Both models pass the shared HF golden and HTTP lifecycle gates; the device gates are manual and nothing Ascend runs in the repository's CI. The first release is TP1 and keeps one executor per call for several operators because CANN 9.2.1 aborts on some executor-destroy paths; a fully cached executor path is the main follow-up.

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

The `ops` module wraps ACLNN. GEMM uses `aclnnMatmul`; the KV cache is laid out as `[slot, kv_head, context, dim]` so a slot view is directly usable as the BNSD IFA cache; the language head packs only the first `decodable_vocab` rows when the tied embedding is transposed, because `decodable_vocab` (151669) is smaller than `vocab_size` (151936). RoPE is `aclnnApplyRotaryPosEmbV2` in BSND `half` mode, with cos/sin rows gathered from a `[context, dim]` table.

The Qwen3 scheduler reuses the production `RequestLedger` and step contract. Each request owns a contiguous KV slot; reclaimed slots restart at position 0. Fixed-slot KV storage is `slots × layers × kv_heads × context × head_dim × 2(K,V) × 2(BF16)` bytes, with the kv-head-major layout above. Cancellation reclaims slots while retaining the allocation for reuse.

CUDA remains the default backend. The Ascend line lives in its own crate and joins the server through `ModelLine` behind the server's `ascend` feature; the CUDA crates carry no backend features, so an Ascend build with default features off never compiles them.

## Verification

All gates are manual. They were last run on Ascend 910B3 / CANN 9.2.1 / aarch64.

```bash
# Operator contracts: GEMM, RMSNorm, SiLU/Mul/Add, embedding, RoPE, attention
cargo test --release -p pegainfer-ascend --features ascend \
  --test contract -- --ignored --test-threads=1

# Committed HF golden, 0.6B and 4B
PEGAINFER_TEST_MODEL_PATH=/opt/atomgit/models/Qwen3-0.6B \
  cargo test --release -p pegainfer-ascend --features ascend \
  --test golden -- --ignored --test-threads=1

# Release HTTP lifecycle (0.6B and 4B)
python3 scripts/ascend/check_http.py \
  --model-path /opt/atomgit/models/Qwen3-0.6B \
  --output target/ascend-http-06b --context 512
```

HF golden results, using the CUDA gate's bounds (`MEAN_TOL=0.06`, `P99_TOL=0.20`):

| model | case | mean | p99 |
|---|---|---|---|
| 0.6B | sequential | 0.0398 | 0.153 |
| 0.6B | ragged batch4 | 0.0400 | 0.152 |
| 0.6B | mixed prefill/decode | 0.0396 | 0.140 |
| 4B | sequential | 0.0307 | 0.106 |
| 4B | ragged batch4 | 0.0317 | 0.108 |
| 4B | mixed prefill/decode | 0.0311 | 0.110 |

The HTTP gate covers health/models, completions, streaming concurrency, stop strings, chat, unsupported-parameter 400s, context boundaries, queued and active cancellation, slot re-admission, and idle metrics. CANN 9.2.1 replaces the process SIGINT disposition, so the server exits on the signal instead of running the frontend's graceful path; the script accepts exit code 0 or SIGINT and rejects SIGKILL.

## CANN 9.2.1 notes

These were reproduced in minimal C++ programs, independent of this crate:

- `aclDestroyAclOpExecutor` double-frees or hangs for several operators (AddRmsNorm, PromptFlashAttentionV3, MatMul with a non-zero workspace, Mul, Embedding, and IFA after descriptor reuse). Operators with stable descriptors use cached repeatable executors; the remaining operators retain their executor for the process.
- Destroying the `aclTensor` descriptors used by IFA between calls crashes the next call, so descriptors and tensor lists are cached by address.
- `aclrtDestroyContext` / `aclFinalize` can hang process exit; the backend leaves teardown to the OS.
- `aclnnMul` with `out == self` traps; SwiGLU writes the product to a distinct output.
- `aclnnAddRmsNorm` is not used in v1: it agrees numerically but aborts on executor destruction. The model uses `aclnnAdd` + `aclnnRmsNorm`.

## Outstanding

- Full executor caching (the v1 path retains several executors per call; bounded for a test run, growing for a long-lived server).
- Prefill performance (`PromptFlashAttentionV3` per slot) and a serving benchmark baseline.
- Paged KV, prefix cache, sampling/logprobs, CANN graph, TP/HCCL, and other model lines.
- No repository CI; keep the device gates manual.
