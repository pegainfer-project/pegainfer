# Initial ROCm 10.1 support for Qwen3

**TL;DR**: Qwen3-0.6B/4B BF16 runs on R9700 (`gfx1201`) through native Rust + HIP/rocBLAS, using single-GPU eager execution, greedy decoding, chunked prefill, and mixed batches. Both models pass independent HF golden and HTTP lifecycle gates. `scripts/amd/check_ci.sh` cross-compiles gfx1100/gfx1201 and runs CPU contract tests against the public ROCm 10.1 SDK; nothing ROCm runs in the repository's CI, so run it by hand, and the GPU gates are manual. RDNA 3 execution and a stable release with the CUDA backend's full feature set remain outstanding.

Last touched: 2026-11-17

## Supported profile

| Area | Current support |
| --- | --- |
| Platform | Linux x86_64, ROCm 10.1.0; R9700 / gfx1201 qualified for execution |
| Models | Qwen3-0.6B and Qwen3-4B, original BF16 safetensors weights |
| Execution | TP1, eager HIP operators, rocBLAS BF16 GEMM with FP32 accumulation |
| Scheduling | Fixed KV slots, queued excess requests, chunked prefill and mixed prefill/decode batches |
| Selection | Greedy; explicitly set HTTP `temperature=0` or `top_k=1` |
| HTTP | Completions, chat, SSE, usage, stop strings/EOS, cancellation, and metrics |
| Context | Default 4096, configurable 1–8192 within checkpoint limits; 4B serving boundary tested at 4096; 8192 is not covered by the serving gates |
| Capacity | Slots default 4 (1–16); chunks default 128 (1–512); serving qualification uses slots=4/chunk=128 |
| Other targets | gfx1100 cross-compiles but lacks hardware qualification; gfx1101/1102/1200 are accepted compiler targets without project qualification |
| Unsupported | HIP Graph, TP/RCCL, random sampling, logprobs/prompt scoring, prefix cache, paged KV, offload/P-D, LoRA, speculative decoding, quantization, and other model families |

This backend always runs eager. If the shared `cuda_graph=true` default is present, startup explicitly logs that it is ignored; explicitly requesting `--cuda-graph=true` rejects startup. Prefix caching is always disabled, and supplying `--no-prefix-cache` logs that the flag is redundant. Unsupported HTTP sampling, scoring, and transfer parameters receive an explanatory 400 before an SSE stream opens. The scheduler separately validates requests submitted outside HTTP.

## Public SDK and build entry point

Prerequisites are Rust/rustup (the repository pins its toolchain), Python with venv/pip support, a C++ compiler and `ar`, OpenSSL development files, pkg-config, and protoc (on Ubuntu: `protobuf-compiler libssl-dev pkg-config`).

Use an existing ROCm 10.1 development installation, or install the public SDK into a dedicated venv:

```bash
bash scripts/amd/install_sdk.sh .venv-rocm gfx1201
export ROCM_PATH="$(.venv-rocm/bin/rocm-sdk path --root)"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export PEGAINFER_ROCM_ARCH=gfx1201
cargo build --locked --release \
  -p pegainfer-server --no-default-features --features rocm,qwen3
```

The installer uses AMD's public [ROCm Python package distribution](https://rocm.docs.amd.com/en/docs-10.1.0/install/rocm.html), pins ROCm 10.1.0, and runs `rocm-sdk init` to expand the development tree. The optional second argument installs device-specific runtime libraries (`device-gfx1201` here), needed to run on that target; compilation and the CPU checks need only `libraries,devel`. Installation itself does not require a GPU or install a kernel driver. Existing system installations can set `ROCM_PATH` directly.

The build helper accepts standard SDK `lib`/`lib64` layouts and versioned wheel libraries. Split SDKs can override `PEGAINFER_ROCM_CORE`, `PEGAINFER_ROCM_LIBRARIES`, `PEGAINFER_ROCM_DEV_INCLUDE`, and `PEGAINFER_HIPCC`. Set the gfx target explicitly; compilation never discovers or initializes a GPU. ROCm and CUDA builds use different features, so a separate `CARGO_TARGET_DIR` per backend avoids rebuilding one after the other. GPU selection is the caller's responsibility; HIP ordinals need not match management-tool numbering.

## Models and serving

The gates were qualified on these checkpoint revisions; existing snapshots can also be supplied directly. `hf` comes with `huggingface_hub`, and `PEGAINFER_MODEL_ROOT` defaults to `models` for the manual gates:

```bash
export PEGAINFER_MODEL_ROOT="${PEGAINFER_MODEL_ROOT:-models}"
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir "$PEGAINFER_MODEL_ROOT/Qwen3-0.6B"
hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --local-dir "$PEGAINFER_MODEL_ROOT/Qwen3-4B"
cargo run --locked --release \
  -p pegainfer-server --no-default-features --features rocm,qwen3 -- \
  --model-path "$PEGAINFER_MODEL_ROOT/Qwen3-4B" --served-model-name qwen3-rocm \
  --port 18302 --rocm-slots 4 --rocm-max-context 4096 --rocm-chunk-size 128
```

The service uses the shared frontend's listener configuration. Select the intended GPU before launching; its container-visible ordinal defaults to 0. The serving runtime has no Python/Torch dependency.

## Runtime contracts

`pegainfer-device` owns HIP devices, streams, buffers, and rocBLAS handles. `Arc` keeps the runtime alive until its buffers are released; a mutex serializes submissions. Submissions rebind the device on the calling thread. Host copies synchronize before returning, and buffer destruction synchronizes before freeing memory. A partially failed model forward poisons the model, preventing reuse of partially updated KV.

`pegainfer-kernels/csrc/hip/qwen3.cpp` implements embedding, RMSNorm, residual addition, SiLU, RoPE, KV append, causal GQA attention, and argmax; GEMM calls rocBLAS. The Rust ABI wrappers validate shapes, capacities, scalars, and device/stream ownership. Device-side token IDs, slots, and positions must be in range. Violations trap and surface at launch or stream synchronization; callers must discard the failed runtime. Trap tests use separate processes because the device context may become unusable.

The Qwen3 scheduler reuses the production `RequestLedger` and step contract. Each request owns a contiguous KV slot; reclaimed slots restart at position 0. Fixed-slot KV storage is `slots × context × layers × kv_heads × head_dim × 2(K,V) × 2(BF16)` bytes. Qwen3-4B with slots=4/context=4096 reserves approximately **2.25 GiB** for KV, plus weights, scratch, library workspace, and headroom. The KV metric reports the fraction of reserved capacity assigned to active slots. Cancellation reclaims slots while retaining the underlying allocation for reuse.

The attention kernel establishes a correctness baseline with FP32 scores, softmax, and reductions over request-local KV, followed by BF16 output. It is not an optimized paged/flash attention implementation; long prefills are expensive. Wave width follows the compilation target.

CUDA remains the default backend. Its implementation is included at the original Qwen3 crate root to preserve symbol paths. ROCm builds branch before CUDA discovery/submodule initialization and exclude CUDA-only sampling, KV/offload, and CUPTI dependencies. Backend features are mutually exclusive, so each is checked in its own build.

## Verification

The CPU checks need the SDK but no GPU, and should be run for any change to the ROCm backend or to the shared crates it builds:

```bash
PEGAINFER_ROCM_ARCH=gfx1201 bash scripts/amd/check_ci.sh
```

This builds the server and all relevant test targets, runs CPU tests, and explicitly lints the ROCm packages and test targets. CPU coverage includes the HTTP contract's parameter types, greedy epsilon/top-k boundaries, body forwarding and fallback routes; scheduler admission, chunk budgets, slot reuse, cancellation before/during forward, and fatal errors; and ABI shape overflow checks. Scheduler tests drive the production scheduler and ledger through a test executor to inject deterministic failures. They do not establish GPU numerical correctness.

The GPU gates are manual, using an already selected GPU, the server built above, and the model snapshots:

```bash
cargo test --locked --release -p pegainfer-device --features rocm --lib -- --ignored --test-threads=1
cargo test --locked --release -p pegainfer-kernels --no-default-features --features rocm \
  --test rocm_contract -- --ignored --test-threads=1
PEGAINFER_TEST_MODEL_PATH="$PEGAINFER_MODEL_ROOT/Qwen3-4B" cargo test --locked --release \
  -p pegainfer-qwen3 --no-default-features --features rocm --test rocm_golden -- --ignored --nocapture --test-threads=1
python3 scripts/amd/check_cli.py
python3 scripts/amd/check_http.py --model-path "$PEGAINFER_MODEL_ROOT/Qwen3-4B" --context 4096 \
  --output target/rocm-http-4b
```

Repeat the golden and HTTP lines for Qwen3-0.6B (`--context 512`). Together they check device ownership/copies, Rust wrapper failure paths, isolated device traps, both models' HF golden gates, and real HTTP/CLI behavior. GPU tests are marked ignored so the CPU checks compile them without executing them. Clippy currently reports warnings, including CUDA-oriented shared config unused in ROCm builds.

### Numerical evidence

The R9700 eager profile passes the committed HF BF16 fixtures: 48 prompts of length 8–247 and 16 teacher-forced decode tokens. Each path scores **816 positions / 6528 top-8 logprob values**, including GPU argmax/tie-breaking checks. Bounds remain mean ≤ 0.06, p99 ≤ 0.20, and HF argmax regret ≤ 0.20.

| Model | Path | Mean | p99 |
| --- | --- | ---: | ---: |
| Qwen3-0.6B | sequential | 0.040036 | 0.148445 |
| Qwen3-0.6B | ragged batch4 | 0.039923 | 0.148720 |
| Qwen3-0.6B | mixed prefill/decode batch4 | 0.039452 | 0.145736 |
| Qwen3-4B | sequential | 0.031401 | 0.111609 |
| Qwen3-4B | ragged batch4 | 0.031099 | 0.108667 |
| Qwen3-4B | mixed prefill/decode batch4 | 0.031145 | 0.111609 |

Across both models and all three paths this covers **4896 positions**. The mixed pass includes 52 mixed prefill/decode steps per model; fixed-shape repeated logits are also checked.

Each model's HTTP gate exercises 16 variable-length streaming requests at concurrency=8 with 4 slots, completion/chat, stop/usage, exact 400 rejection responses, both greedy paths and the epsilon boundary, queued/active cancellation, slot readmission, idle metrics, and clean SIGINT shutdown. The 4B boundary request uses 4094 prompt tokens plus 2 output tokens.

**Provenance limit:** Checkpoint revisions are pinned, but upstream golden metadata lacks checkpoint revisions/config hashes. These gates establish compatibility with the committed fixtures, not byte identity with the fixture's original checkpoint. No CUDA/vLLM performance parity or long-duration stability claim is made.

## Remaining work

Next, run the same gates on RDNA 3 hardware and retain CUDA GPU regression coverage. Then migrate fixed slots to paged KV while preserving lifecycle guarantees, add sampling/logprobs, and qualify HIP Graph and optimized attention/GEMM. Long-context HF references, full OOM recovery, additional failure injection, and higher-concurrency qualification remain outstanding. These follow-ups do not expand the current qualified profile.
