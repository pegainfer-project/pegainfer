# Initial ROCm 10.1 support for Qwen3

**TL;DR**: The native Rust + HIP/rocBLAS backend runs Qwen3-0.6B/4B BF16 on Radeon AI PRO R9700 (`gfx1201`), with TP1, greedy decoding, eager execution, chunked prefill, mixed batches, and the shared HTTP frontend. Both models passed HF golden gates and real HTTP lifecycle checks. `gfx1100` has only been cross-compiled; RDNA 3 hardware validation is outstanding. This is an RDNA 4 preview with a limited feature set, not a stable release qualified on both architectures.

Last touched: 2026-11-12

## Supported profile

| Area | Current support |
| --- | --- |
| Platform | Linux x86_64; locked ROCm 10.1.0 foundation image; R9700 / gfx1201 |
| Models | Qwen3-0.6B and Qwen3-4B with original BF16 safetensors weights; other sizes are not yet qualified |
| Execution | Single GPU, rocBLAS BF16 GEMM with FP32 accumulation, native HIP eager operators |
| Scheduling | Up to `--rocm-slots` active requests, with excess requests queued; prefill chunks and decode rows can share a batch |
| Token selection | Greedy; HTTP clients must explicitly set `temperature=0` or `top_k=1` |
| HTTP | `/v1/completions`, `/v1/chat/completions`, SSE, usage, stop strings, EOS, cancellation, and metrics |
| Context | Default 4096; configurable from 1 to 8192, bounded by the checkpoint limit; the 4B HTTP boundary was tested at 4096; 8192 has operator-level qualification only |
| Slots and chunks | Slots default to 4, range 1–16; chunk size defaults to 128, range 1–512; serving qualification used slots=4/chunk=128 |
| Unsupported features | HIP Graph, TP/RCCL, random sampling, logprobs/prompt scoring, prefix cache, paged KV, offload/P-D, LoRA, speculative decoding, quantization, and other model families |
| Other architectures | gfx1100 kernels cross-compile; gfx1101/1102/1200 are accepted compiler targets, but none has runtime qualification in this project |

The backend uses fixed KV slots. Reclaimed slots restart writing at position 0. The CUDA backend's shared prefix, page, and offload state has not been ported. Startup logs identify the execution profile; explicitly requesting `--cuda-graph=true` or a `--tp-size` greater than 1 rejects startup. Unsupported HTTP sampling, scoring, and KV transfer requests receive a 400 response with a reason before an SSE stream opens. The scheduler also validates internal requests.

## Build and run

Run these commands from the worktree root. The qualification workspace provides `/home/zihaomu/zihaomu_workspace/rocm10-launch.sh` and the locked foundation image. The project launcher validates the foundation, then checks the project HIP/rocBLAS file hashes and compiler identity. It defaults to HIP ordinal 2, which maps to physical GPU0, PCI `0000:03:00.0`. HIP enumeration differs from amd-smi physical numbering on this host: HIP ordinal 0 maps to physical GPU3. Before the shared verifier performs GPU computation, the project launcher checks PCI metadata and rejects an unexpected mapping or the excluded physical GPU3.

```bash
HIP_VISIBLE_DEVICES=2 scripts/amd/run_rocm10.sh bash scripts/amd/bootstrap.sh
HIP_VISIBLE_DEVICES=2 scripts/amd/run_rocm10.sh cargo build --locked --release \
  -p pegainfer-server --no-default-features --features rocm,qwen3
```

`bootstrap.sh` installs the Rust toolchain and OpenSSL development files under the workspace's `artifacts/amd/toolchain`, without changing the foundation image. The existing `rust-toolchain.toml` pins Rust to `nightly-2026-07-10`. The first build needs network access to fetch dependencies pinned in `Cargo.lock`. The OpenSSL packages and rustup installer are checked against pinned SHA256 hashes. If upstream removes a pinned version, qualify its replacement instead of silently substituting it.

Model downloads also use pinned revisions. Both commands below have been executed in the qualification workspace, where the weights are already present:

```bash
HIP_VISIBLE_DEVICES=2 scripts/amd/run_rocm10.sh python scripts/amd/download_models.py \
  --model Qwen/Qwen3-0.6B --output ../models/Qwen3-0.6B
HIP_VISIBLE_DEVICES=2 scripts/amd/run_rocm10.sh python scripts/amd/download_models.py \
  --model Qwen/Qwen3-4B --output ../models/Qwen3-4B
```

The exact serving arguments are recorded in `http.4B.command` in the qualification report. Run that binary through the project launcher to use the same runtime libraries as the build. A host-accessible launch example appears at the end of this document. Each request must explicitly select greedy decoding.

## Implementation and ownership

| Entry point | Responsibility |
| --- | --- |
| `pegainfer-build/src/rocm.rs` | Independent HIP compilation and archiving; project-local linker aliases for the wheel SDK; compiled gfx targets recorded in the artifact |
| `pegainfer-device` | HIP device/stream, rocBLAS handle, typed buffers, synchronous host copies, and device rebinding across threads |
| `pegainfer-kernels/csrc/hip/qwen3.cpp` | Embedding, RMSNorm, residual addition, SiLU, RoPE, KV append, causal GQA attention, argmax, and rocBLAS GEMM |
| `pegainfer-kernels/src/rocm.rs` | ABI wrappers that check shapes, capacity, and buffer ownership by device/stream |
| `pegainfer-qwen3/src/rocm/model.rs` | Shared Qwen3 config/EOS/vocabulary handling; BF16 weight loading, memory budgeting, contiguous KV appends, and ragged forward execution |
| `pegainfer-qwen3/src/rocm/scheduler.rs` | Shared `RequestLedger` and step contract; queues, slots, chunks, stopping, and cancellation |
| `pegainfer-frontend` / `pegainfer-server` | Shared vLLM Rust HTTP frontend; an optional request-validation router in `ServePlan`, with the existing default CUDA routing preserved |
| `scripts/amd` | Project environment lock, downloads, build environment, real operator/golden/HTTP/CLI qualification, and evidence capture |

The Qwen3 CUDA implementation is kept in `cuda.rs`, conditionally included at the original crate root to preserve symbol paths. CUDA remains the default feature. ROCm builds branch before CUDA toolkit discovery or submodule initialization and do not compile `cudarc`, CUPTI, or CUDA sampling/KV/offload dependencies. Other models that depend on CUDA explicitly retain the CUDA feature, preserving their backend selection after workspace dependencies disable default features. CUDA and ROCm features are mutually exclusive; use separate CI jobs for each backend rather than `--all-features`.

Device resources are kept alive by `Arc`, and a mutex serializes submissions to the stream. Each submission rebinds the HIP device on its calling thread. H2D/D2H copies synchronize the stream before returning, so temporary host memory remains valid until the copy completes. Buffer destruction waits for the stream before freeing memory; the runtime is destroyed after its last buffer owner is released. A model becomes unusable when a GPU step begins and returns to a usable state only after the entire step succeeds. A failed step cannot reuse partially updated KV.

Fixed-slot KV allocation is `slots × context × layers × kv_heads × head_dim × 2(K,V) × 2(BF16)` bytes. Qwen3-4B with slots=4/context=4096 reserves approximately **2.25 GiB** for KV, in addition to weights, scratch buffers, library workspace, and headroom. Startup checks the budget against available VRAM and retains error context if allocation fails. The KV usage metric reports **the fraction of reserved capacity assigned to active slots**, rather than the fraction containing written tokens. Cancellation reclaims slots while keeping the underlying allocation available for reuse.

The initial attention kernel establishes a correctness baseline: each query/head computes FP32 scores, softmax, and reductions over its own contiguous KV, then writes BF16 output. Wave width is determined by the target; the implementation does not inherit CUDA warp/PTX assumptions. This is not an optimized paged/flash attention implementation, and long prefills are expensive. Future optimizations must continue to pass the same numerical and lifecycle gates.

## Validation results

Environment identities are pinned in [environment.lock.json](../../../scripts/amd/environment.lock.json). The tested device was physical GPU0, PCI `0000:03:00.0`, UUID `29ff7551-0000-1000-80dc-b03ca15afb63`, exposed as logical device 0 inside the container. The stack uses ROCm **10.1.0**, HIP component **7.16.26385**, and AMD clang revision `d6f6cb691863e9258034ab48c1dfce21689d5606`. A HIP version beginning with 7.16 is the expected component identity in this ROCm 10.1.0 image.

The serving runtime does not depend on Python or Torch. CPU Torch supplies the independent native-operator reference; the foundation's Torch/Triton pairing remains unchanged.

### Numerical gates

The tests reuse the repository's HF BF16 fixtures: 48 prompts of length 8–247, each followed by 16 teacher-forced decode tokens. Each execution path covers **816 scored positions / 6528 top-8 logprob comparisons** and checks GPU argmax against the CPU tie-breaking rule. Existing thresholds remain unchanged: mean ≤ 0.06, p99 ≤ 0.20, and HF argmax regret ≤ 0.20.

| Model | Path | Mean | p99 |
| --- | --- | ---: | ---: |
| Qwen3-0.6B | sequential | 0.040036 | 0.148445 |
| Qwen3-0.6B | ragged batch4 | 0.039923 | 0.148720 |
| Qwen3-0.6B | mixed prefill/decode batch4 | 0.039452 | 0.145736 |
| Qwen3-4B | sequential | 0.031401 | 0.111609 |
| Qwen3-4B | ragged batch4 | 0.031099 | 0.108667 |
| Qwen3-4B | mixed prefill/decode batch4 | 0.031145 | 0.111609 |

Each model also passes a repeated-logits check at a fixed shape. The mixed pass executes 52 steps containing both prefill and decode rows. Across both models and all three paths, the gates cover **4896 scored positions**.

The independent HIP operator checks cover nine cases: RMSNorm, GEMM, residual addition, SiLU, RoPE, high-position RoPE through position 8191, ragged causal attention, attention at context=8192, and argmax ties. Maximum absolute error was 0.0078125 for high-position RoPE and zero for the other measured numerical cases. These are results for the tested inputs, not an exact-equivalence claim for every shape.

**Reference provenance limitation:** Current weight revisions are pinned, but the committed upstream golden metadata lacks checkpoint revisions and config hashes. These gates establish compatibility with the existing fixtures; they do not prove that the fixture's original checkpoint is byte-identical to the current snapshot. The qualification record preserves the original fixture metadata/hashes and current model file hashes. A follow-up should generate references with complete provenance while retaining the existing gate.

### Serving and resource lifecycle

- Each model serves 16 streaming requests at client concurrency=8 with 4 engine slots. Inputs vary in length and cross chunk boundaries; checks cover SSE `[DONE]`, finish reasons, and usage.
- Non-streaming completions, chat, and stop strings are exercised. The 4B boundary case uses a 4094-token prompt and 2 output tokens to reach 4096; over-limit requests return 400.
- With 4 active requests and 1 queued request, the test disconnects the queued client and then the active clients. Running, waiting, and KV usage metrics must return to zero, after which 4 new requests complete successfully.
- Real HTTP requests verify 400 responses for sampling, logprobs, prompt scoring, echo, and KV transfer. Production CLI checks reject TP2, graph execution, invalid capacities, and quantized checkpoints.
- SIGINT shuts down the server cleanly and the harness waits for process exit. Device tests verify buffer reads after the creating `Device` is dropped, including a thread handoff, plus copy bounds, zero initialization, and allocation-size overflow rejection.
- Shared host unit tests pass: core 9, frontend 97, Qwen3 config 7, and server 4. Release build, formatting, and Clippy checks pass. Clippy still reports warnings, including shared config items unused under ROCm; this is not a warning-free result.

HTTP reports retain latency and text for diagnosis. No paired performance benchmark or two-hour stability run was performed, so these results do not establish a performance gain over CUDA or vLLM.

### Reproduction and evidence

```bash
HIP_VISIBLE_DEVICES=2 scripts/amd/run_rocm10.sh bash scripts/amd/qualify.sh \
  ../artifacts/amd/qualification/rocm10-v1
```

This command requires a fresh output directory to prevent old logs from entering a new qualification record. Choose a new directory when rerunning it. The script runs the release build, device tests, native numerical checks, host/server unit tests, CLI checks, golden and HTTP gates for both models, formatting, and Clippy. It writes `qualification.json` only after every stage passes.

The accepted evidence is retained in the workspace at `artifacts/amd/qualification/rocm10-v1/`: foundation report, individual logs, HTTP requests/responses, tracked patch, and SHA256 hashes for changed source files, the executable, `Cargo.lock`, model files, and fixtures. Qualification ran before commit, so the report identifies the tested candidate by its baseline commit and dirty-source hashes. Implementation files were checked against those hashes again before commit. Raw logs and model files remain in the workspace and are not published in Git.

The RDNA 3 compilation check uses the same ROCm 10.1 toolchain with a separate target directory:

```bash
HIP_VISIBLE_DEVICES=2 PEGAINFER_ROCM_ARCH=gfx1100 scripts/amd/run_rocm10.sh \
  cargo build --release -p pegainfer-kernels --no-default-features --features rocm
```

Its log is `artifacts/amd/qualification/rocm10-v1/cross-gfx1100.log`. The command runs the foundation's gfx1201 environment check, then **only compiles** gfx1100 kernel/device artifacts. It does not execute a gfx1100 code object or validate RDNA 3 rocBLAS solutions. Model loading requires the actual device architecture to match one of the binary's compiled targets.

**Device-selection incident:** The initial tests incorrectly treated HIP ordinal 0 as amd-smi physical GPU0. Recording the actual PCI address revealed that it was the excluded physical GPU3. That qualification was invalidated and moved to `rocm10-v1-invalid-device-selection`. The table above reflects the complete suite rerun on physical GPU0 after adding the PCI guard. `artifacts/amd/INVALID_DEVICE_SELECTION.md` marks the earlier bootstrap/native/HTTP records as excluded from final qualification. An incorrect `HIP_VISIBLE_DEVICES=0` selection is now verified to fail before the verifier performs GPU computation.

## Relationship to the full adaptation plan

This release establishes a working native Qwen3 path through model execution and serving. It reuses config, `RequestLedger`, and the HTTP contract, but has not migrated the CUDA executor, paged KV, sampling, or device abstractions to common types. Fixed slots and rocBLAS are deliberate intermediate choices to establish correctness early. They do not complete milestones M1–M3 of the full adaptation plan.

Next, run the same gates on gfx1100 hardware, add CUDA GPU regression coverage and separate backend CI jobs, then replace fixed slots with paged KV while preserving ownership and lifecycle guarantees. Add random sampling/logprobs before proceeding to HIP Graph and attention/GEMM optimization. Long-context HF references, GPU failure injection, full OOM recovery, and qualification at higher concurrency also remain outstanding.

## Host-accessible launch check

The following command started the 4B server on physical GPU0 and served requests from the host. The recorded check used an outer timeout to send SIGINT afterward. For interactive use, run the command directly and stop it with Ctrl-C:

```bash
HIP_VISIBLE_DEVICES=2 ROCM10_NETWORK=host scripts/amd/run_rocm10.sh \
  ../artifacts/amd/target/rocm10.1-gfx1201/release/pegainfer \
  --model-path ../models/Qwen3-4B --served-model-name qwen3-rocm --port 18302 \
  --cuda-graph=false --no-prefix-cache --rocm-slots 4 \
  --rocm-max-context 4096 --rocm-chunk-size 128
```

The service uses the shared frontend's listener configuration. Host networking exposes the port on the host network. The recorded client request below intentionally tests Chinese input; its payload is preserved verbatim:

```bash
curl --fail --silent --show-error http://127.0.0.1:18302/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-rocm","messages":[{"role":"user","content":"请用中文回答：2 加 2 等于几？"}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

The recorded response was `2 加 2 等于 4。` ("2 plus 2 equals 4."). `host-server.log`, `host-response.json`, and `host-curl-response.json` are retained in the final qualification directory. The check left no background inference service running.
