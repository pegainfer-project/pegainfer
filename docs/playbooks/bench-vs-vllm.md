# pegainfer vs vLLM Comparative Benchmarking

> **TL;DR:** Run both engines on the same GPU sequentially, benchmark with `vllm bench serve` as a unified client, compare TTFT/TPOT/throughput side by side.

## Method

Both servers are tested on the **same GPU, same model, same port** (sequentially — one GPU means no parallel serving). `vllm bench serve` drives both, ensuring identical client behavior and measurement methodology.

Key flags applied to both: `--ignore-eos` (forces exact output length), `--dataset-name random` (synthetic prompts).

## Setup

```
MODEL_PATH=models/Qwen3-4B        # or models/Qwen3.5-4B
VLLM_PYTHON=.venv/bin/python      # pegainfer/.venv with vllm installed
VLLM_CMD=.venv/bin/vllm
PORT=8000
RESULTS_DIR=bench_results/<timestamp>
```

Prerequisites:
- `cargo build --release` — pegainfer up-to-date
- vLLM installed in `.venv` (`uv pip install vllm`)
- Kill any existing process on the port before starting

## Workflow

### 1. Benchmark vLLM

Start server, wait for ready, run benchmarks, kill server:

```bash
# Start
$VLLM_CMD serve $MODEL_PATH --port $PORT --max-model-len 8192 --gpu-memory-utilization 0.9 \
  --served-model-name <short-name> &

# Poll until ready (up to 180s — vLLM torch.compile warmup)
until curl -sf http://localhost:$PORT/v1/models >/dev/null; do sleep 5; done

# Benchmark (repeat for each config)
$VLLM_CMD bench serve \
  --backend openai --model <short-name> --port $PORT \
  --dataset-name random --input-len <in> --output-len <out> \
  --num-prompts 20 --request-rate inf --max-concurrency 1 \
  --ignore-eos --temperature 0 --tokenizer $MODEL_PATH \
  --save-result --result-dir $RESULTS_DIR \
  --result-filename vllm-in<in>-out<out>.json

# Cleanup
pkill -f "vllm serve"
```

### 2. Benchmark pegainfer

Same flow, different server:

```bash
# Start (Qwen3.5 requires PEGAINFER_TRITON_PYTHON for AOT Triton kernels)
RUST_LOG=warn PEGAINFER_TRITON_PYTHON=./.venv/bin/python \
  cargo run --release -- --model-path $MODEL_PATH --port $PORT &

# Poll until ready — pegainfer has no /v1/models; probe with a minimal completions request
until curl -sf http://localhost:$PORT/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<short-name>","prompt":"hi","max_tokens":1}' >/dev/null; do sleep 5; done

# Benchmark (same vllm bench serve client)
$VLLM_CMD bench serve \
  --backend openai --model <short-name> --port $PORT \
  --dataset-name random --input-len <in> --output-len <out> \
  --num-prompts 20 --request-rate inf --max-concurrency 1 \
  --ignore-eos --temperature 0 --tokenizer $MODEL_PATH \
  --save-result --result-dir $RESULTS_DIR \
  --result-filename pega-in<in>-out<out>.json

# Cleanup
pkill -f "target/release/pegainfer"
```

### 3. Compare

Read both JSON results. Key metrics:

| Metric | What it measures |
|--------|-----------------|
| TTFT (mean/median/p99) | Prefill speed. vLLM's mean >> median indicates torch.compile cold start |
| TPOT (mean/median/p99) | Per-token decode latency |
| ITL (mean/median/p99) | Inter-token latency (includes scheduling jitter) |
| Output tok/s | End-to-end decode throughput |
| Request throughput | Requests completed per second |

## Typical Configs

| Config | What it tests |
|--------|---------------|
| `in=1, out=1, n=200` | Minimal overhead — launch latency, framework tax |
| `in=1024, out=256, n=20` | Realistic workload — prefill + sustained decode |
| `in=1, out=512, n=20` | Pure decode — long generation from short prompt |
| `in=2048, out=32, n=20` | Prefill-heavy — TTFT dominates |

## Gotchas

- **vLLM cold start:** torch.compile triggers on the first 1–3 requests. With `n=10`, mean TTFT is inflated 5–50× and p99 is always a cold-start spike — neither is meaningful. **Read median only.** For stable p99, use `n>=30` (cold-start requests become a small tail of the distribution). Example: Qwen3.5-4B at (2048,1), n=10: mean=1279ms, median=222ms, p99=9846ms.
- **vLLM prefix cache:** Disable with `vllm serve --no-enable-prefix-caching` for random synthetic prefill probes unless prefix-cache behavior is intentionally part of the experiment. A 2026-05-04 `input_len=4096,output_len=64` run showed vLLM prefix cache hits even on the random dataset.
- **Greedy must be explicit:** vLLM 0.19.1 `vllm bench serve` no longer forces `temperature=0`; pass `--temperature 0` in both vLLM and pegainfer runs when comparing kernel/runtime speed.
- **Context limit includes output:** `--max-model-len` is prompt plus generated tokens. For `input-len=4096, output-len=64`, use at least `4160`; `8192` is the simple safe value on RTX 5090.
- **vLLM serve log flag drift:** vLLM 0.19.1 rejects the old `--disable-log-requests` flag. Omit it unless `vllm serve --help=all` on that machine shows a supported equivalent.
- **Text-only Qwen3.5 on vLLM:** Some Qwen3.5 checkpoints expose multimodal metadata. For text-only benchmarking, start `vllm serve` with `--language-model-only` (equivalent to `--limit-mm-per-prompt '{"image":0,"video":0}'`).
- **pegainfer empty prompts:** Random dataset may produce empty prompts which pegainfer rejects. Check failed request count.
- **pegainfer streaming usage:** The vLLM frontend can overreport streaming `usage.completion_tokens` in `vllm bench serve` results. For fixed-output probes, trust TTFT/TPOT/ITL and recompute output throughput from `num_prompts * output_len / duration` until usage accounting is fixed.
- **Zombie processes:** Always `pkill` after benchmarking. Leftover servers block the port and hold GPU memory.
- **CUDA Graph:** pegainfer enables CUDA Graph by default. For apples-to-apples decode comparison, note this in the report. vLLM also uses CUDA Graph by default.
- **Concurrency:** `--max-concurrency N` controls client-side in-flight request pressure, not internal batch size. Confirm actual batching from server trace fields such as active-set and decode-batch maxima before interpreting scaling.
- **Qwen3.5-4B CUDA Graph OOM on 16 GB:** torch.compile + CUDA Graph needs ~1 GiB extra for graph profiling on top of the 12 GiB model+activation footprint. Workaround: `--max-num-seqs 1` reduces the graph capture to batch_size=1 only and fits in 16 GB. Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for marginal help.
