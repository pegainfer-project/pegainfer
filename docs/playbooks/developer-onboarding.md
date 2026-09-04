# Developer Onboarding: Setting Up the pegainfer Dev Environment from Scratch

**Status**: Complete
**TL;DR**: Full new-developer onboarding — toolchain check, unified venv, build, tests, benchmark smoke test.

---

## Prerequisites

- GPU machine with CUDA toolkit installed (`/usr/local/cuda`)
- Model files under `models/` (at least one of the supported models below)

## 1. Verify Toolchain

```bash
rustc --version   # need 1.91+ (Rust 2024 edition)
uv --version      # Python package manager
/usr/local/cuda/bin/nvcc --version  # CUDA compiler
```

## 2. Create Unified Python venv (optional for the default build)

The default build (Qwen3 only) is pure Rust + CUDA and needs no Python. Set up
`.venv` when you build feature-gated model lines (`qwen35` needs triton) or
run the HF reference scripts (torch/transformers).

```bash
cd pegainfer/
uv venv
uv pip install triton torch transformers accelerate pytest
```

Verify:

```bash
.venv/bin/python -c "import triton; print(triton.__version__)"
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> build.rs auto-detects `.venv/bin/python` for Triton AOT compilation. Override with `PEGAINFER_TRITON_PYTHON` if needed.
>
> Multi-GPU (TP/NCCL) runtime note: cudarc dlopens `libnccl.so` (its search list has no plain `libnccl.so.2`), while the `nvidia-nccl-cu13` wheel ships only `libnccl.so.2`. Add `libnccl.so -> libnccl.so.2` inside `.venv/lib/python3.*/site-packages/nvidia/nccl/lib/` and put that dir on `LD_LIBRARY_PATH` when running TP.

## 3. Build

```bash
cargo build --release                          # Qwen3 only, no Python
cargo build --release --features qwen35     # adds Qwen3.5 (Triton AOT at build time)
```

First build takes ~30s. Compiles CUDA kernels (`pegainfer-kernels/csrc/*.cu`); with `--features qwen35` it also generates the Triton AOT kernels (`pegainfer-kernels/tools/triton/*.py`).

## 4. Run Tests

```bash
cargo test -r --workspace --lib   # unit tests (~9s)
cargo test -r -p pegainfer-qwen3 --test hf_golden_gate   # Qwen3-4B logits vs HF golden (~7s, needs GPU + model)
```

> **Always use `--release`**. Debug builds are extremely slow for GPU code and will timeout.

Tests requiring Qwen3-8B are marked `#[ignore]` and won't affect the default flow.

## 5. Start the HTTP Server

```bash
RUST_LOG=info cargo run --release -- --port 8000
```

Test the API:

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_tokens":16}' | python3 -m json.tool
```

## 6. Benchmark Smoke Test

Benching is HTTP-level (the in-process `bench_serving` bin is retired — it
bypassed the real serving path). With the server from step 5 still running:

```bash
python3 scripts/bench_http_serving.py --base-url http://localhost:8000 \
  --model models/Qwen3-4B --num-requests 8 --concurrency 2 --max-tokens 32
```

It reports streaming TTFT/ITL/TPOT, request latency, QPS, and error rate. If
requests complete with zero errors and sane latencies, the environment is
working. For serious load benching use the external `vllm-bench` client.

## Supported Models

All commands default to `models/Qwen3-4B`. Use `--model-path` to switch.

| Model | Path | Notes |
| --- | --- | --- |
| Qwen3-4B | `models/Qwen3-4B` | Default. Tied embeddings (no separate lm_head). |
| Qwen3.5-4B | `models/Qwen3.5-4B` | Hybrid attention (mixed full + sliding window layers). |

### Running a Different Model

Server:

```bash
RUST_LOG=info cargo run -r -- --model-path models/Qwen3.5-4B --port 8000
```

Benchmark (server running with that model):

```bash
python3 scripts/bench_http_serving.py --base-url http://localhost:8000 \
  --model models/Qwen3.5-4B --num-requests 8 --concurrency 2 --max-tokens 32
```

Accuracy tests live in each model crate:

```bash
cargo test -r -p pegainfer-qwen3  --test hf_golden_gate   # Qwen3-4B logits vs stored HF golden (bf16 tolerance)
cargo test -r -p pegainfer-qwen35 --test hf_golden_gate   # Qwen3.5-4B logits vs stored HF golden (bf16 tolerance)
cargo test -r -p pegainfer-qwen35 --test e2e_scheduler    # Qwen3.5-4B scheduler request-flow integration
```

Qwen3-4B no longer pins exact greedy text: a bit-wise baseline false-positives across GPUs (per-card bf16 GEMM drifts the low bits). `hf_golden_gate` instead teacher-forces a fixed set of sequences and asserts pegainfer's logprobs land within the bf16 noise floor of a stored HuggingFace reference — across bs=1, batched, and the CUDA-graph path. The reasoning and tolerances are in `docs/models/qwen3/accuracy-gate.md`.

Qwen3.5-4B follows the same HF-golden rule with one model-specific caveat: its fixture is produced through HF `use_cache=True` / `past_key_values`, and its replay surface is sequential graph plus bucket-straddling batched graph plus slot compaction because Qwen3.5 does not currently have an eager batched decode path. A broader PegaInfer-owned rand/hash corpus is deferred until the project decides how to handle cross-architecture exact-token drift.

### Regenerating Reference Data

After a change that alters numerical output, regenerate the reference. The Qwen3-4B golden is recomputed on GPU through HuggingFace:

```bash
uv run --no-project python tools/accuracy/dump_qwen3_hf_golden.py \
    --model-path models/Qwen3-4B --out test_data/qwen3-4b-hf-golden.safetensors
```

Qwen3.5 uses size-keyed HF logits goldens (4B and 9B committed, short + long each). The dumper auto-names the fixture from the model config and `--prompt-lens`; 9B pins a revision:

```bash
# 4B short + long
python3 tools/accuracy/dump_qwen35_hf_golden.py --model-path models/Qwen3.5-4B
python3 tools/accuracy/dump_qwen35_hf_golden.py --model-path models/Qwen3.5-4B --prompt-lens 4097,8192 --decode-tokens 8
# 9B (add --prompt-lens 4097,8192 --decode-tokens 8 for the long variant)
python3 tools/accuracy/dump_qwen35_hf_golden.py --model-path models/Qwen3.5-9B \
    --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

The older Qwen3.5 exact greedy JSON baseline and regeneration test are retired. Re-run the corresponding HF logits gate to confirm the new reference passes.

## Next Steps

- `docs/playbooks/profiling-guide.md` — profiling toolchain (nsys, ncu, fastrace, Perfetto)
- `docs/playbooks/bench-vs-vllm.md` — comparative benchmarking against vLLM
- `CLAUDE.md` (workspace + project level) — architecture and dev conventions
