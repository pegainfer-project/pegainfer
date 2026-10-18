# Frontend CPU Profiling Baseline (pegainfer-sim)

**Created**: 2026-06-05
**Last touched**: 2026-06
**TL;DR**: CPU-side profiling of the vLLM/OpenAI frontend path using `pegainfer-sim` with fixed TTFT=5ms / TPOT=12ms. At 200 req / concurrency=16 / prompt=128 words / output=64 tokens the frontend adds ~150ms TTFT overhead above the 5ms simulated floor and shows no throughput bottleneck (QPS=18.2, 0 failures). Top hotspots: heap allocation (malloc/realloc ~10%), stream polling (~7.5%), clock_gettime (~2%), JSON serialization (~1%). No single frontend bottleneck dominates — the overhead is distributed across tokio runtime, IPC bridge, and HTTP framing.

## Reproducible Benchmark

### Prerequisites

```bash
# Build sim binary (requires protoc)
cargo build --release -p pegainfer-sim
```

### Create a tiny local model dir (avoids HF download)

```bash
mkdir -p /tmp/pegainfer-sim-model

cat > /tmp/pegainfer-sim-model/tokenizer.json << 'EOF'
{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    { "id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true }
  ],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": { "type": "WordLevel", "vocab": { "<unk>": 0, "alpha": 1, "beta": 2 }, "unk_token": "<unk>" }
}
EOF

cat > /tmp/pegainfer-sim-model/tokenizer_config.json << 'EOF'
{ "unk_token": "<unk>", "tokenizer_class": "PreTrainedTokenizerFast" }
EOF

cat > /tmp/pegainfer-sim-model/config.json << 'EOF'
{ "model_type": "pegainfer_sim", "max_position_embeddings": 8192 }
EOF
```

### Start server

```bash
cargo run --release -p pegainfer-sim -- \
  --model-id /tmp/pegainfer-sim-model \
  --port 8732 \
  --base-ttft-ms 5 \
  --tpot-ms 12 \
  --prefill-tokens-per-ms 100 \
  --max-model-len 8192
```

### Run benchmark

```bash
python3 scripts/bench_http_serving.py \
  --base-url http://127.0.0.1:8732 \
  --model /tmp/pegainfer-sim-model \
  --num-requests 200 \
  --concurrency 16 \
  --prompt-words 128 \
  --max-tokens 64 \
  --warmup 4 \
  --out /tmp/sim-bench-result.json
```

### Run with perf profiling

In a separate terminal after starting the server and confirming it responds:

```bash
# Summary stats (IPC, cache misses, branch mispredictions)
SIM_PID=$(pgrep -f "target/release/pegainfer-sim")
perf stat -p $SIM_PID \
  -e cycles,instructions,cache-references,cache-misses,branch-misses,task-clock,context-switches,cpu-migrations \
  -- timeout 15 python3 scripts/bench_http_serving.py \
    --base-url http://127.0.0.1:8732 \
    --model /tmp/pegainfer-sim-model \
    --num-requests 200 \
    --concurrency 16 \
    --prompt-words 128 \
    --max-tokens 64 \
    --warmup 4

# Function-level hotspot capture
perf record -g -p $SIM_PID -o /tmp/sim-perf.data -- \
  timeout 10 python3 scripts/bench_http_serving.py \
    --base-url http://127.0.0.1:8732 \
    --model /tmp/pegainfer-sim-model \
    --num-requests 100 \
    --concurrency 16 \
    --prompt-words 128 \
    --max-tokens 32 \
    --warmup 2

perf report -i /tmp/sim-perf.data --stdio --no-children --percent-limit 1
```

## Results

### Baseline: 100 req, concurrency=8, prompt=64 words, output=32 tokens

| Metric | Value |
|---|---|
| Requests | 100 completed, 0 failed |
| QPS | 18.4 |
| Wall time | 5.4s |
| TTFT avg / p50 / p95 / p99 | 151ms / 153ms / 284ms / 297ms |
| TPOT avg / p50 / p95 / p99 | 51.8ms / 13.5ms / 157ms / 158ms |
| ITL avg / p50 / p95 / p99 | 76.7ms / 13.3ms / 303ms / 305ms |
| Input tok/s | 1180 |
| Output tok/s | 590 |

Simulated TTFT floor = 5ms + 2 tokens / 100 tok/ms ≈ 5ms. Observed TTFT ~150ms, so **frontend overhead is ~145ms** at concurrency=8.

The p50 TPOT is 13ms (matching the 12ms simulated TPOT + ~1ms jitter), but the avg/max are inflated by ~300ms ITL spikes. These spikes appear when a request's first token arrives during a batch wave — the request waits for the next token-emission cycle in the stream. This is an artifact of the IPC bridge batching, not a CPU cost.

### High-concurrency: 200 req, concurrency=16, prompt=128 words, output=64 tokens

| Metric | Value |
|---|---|
| Requests | 200 completed, 0 failed |
| QPS | 18.2 |
| Wall time | 11.0s |
| TTFT avg / p50 / p95 / p99 | 153ms / 155ms / 290ms / 303ms |
| TPOT avg / p50 / p95 / p99 | 126ms / 129ms / 158ms / 159ms |
| ITL avg / p50 / p95 / p99 | 128ms / 14ms / 306ms / 308ms |
| Input tok/s | 2326 |
| Output tok/s | 1163 |
| perf task-clock | 2599ms over 12s wall |
| IPC | 0.25 (737M instructions / 2939M cycles) |
| Cache miss rate | 58% (94M misses / 161M refs) |
| Branch mispredictions | 15.3M |

Frontend overhead at concurrency=16 is similar (~150ms TTFT), indicating the overhead is per-request, not queueing-bound.

## CPU Hotspot Breakdown (perf, self %)

From `perf record -g` during the 200-req run:

| Category | Self % | Function(s) |
|---|---|---|
| **Heap allocation** | ~10% | `malloc` (3.2%), `cfree` (1.3%), `realloc` chains (3.9%) |
| **Stream polling** | ~7.5% | `futures_util::stream::StreamExt::poll_next_unpin` (4.7%), `Instrumented::poll_next` (2.8%) |
| **Clock / timing** | ~2% | `__vdso_clock_gettime` (1.3%), `Timespec::now` (1.8%) |
| **Tokio runtime** | ~3% | `Context::run` (1.0%), `process_at_time` (1.2%), `Steal::steal_into` (0.7%) |
| **HTTP framing** | ~2% | `hyper::Dispatcher::poll_catch` (1.3%), `http_body_util::MapErr::poll_frame` (0.9%), `hyper::Buffered::poll_flush` (0.5%), `ChunkSize::new` (1.0%) |
| **Serialization** | ~1.5% | `serde_json::format_escaped_str_contents` (1.0%), `rmp_serde::Decoder::any_inner` (0.5%) |
| **Vec growth** | ~2.5% | `RawVecInner::finish_grow` (1.5%), `bytes::shared_to_vec` (1.2%) |
| **IPC bridge** | ~1% | `PushSocket::send` (0.7%), `mpsc::Tx::push` (0.7%) |
| **Tokenizer** | ~1% | `ModelWrapper::id_to_token` (0.6%), `AddedVocabulary::simple_id_to_token` (0.5%) |
| **Simulated engine** | ~1% | `run_simulated_request` (0.9%) |

### Observations

1. **No dominant hotspot.** The top single function (`malloc`) is only 3.2%. The cost is spread across many small contributors typical of async Rust / tokio workloads.

2. **Heap allocation is the largest measured category** (~10% combined self). `malloc` 3.2% + `cfree` 1.3% + `realloc` chains 3.9% + `RawVecInner::finish_grow` 1.5% + `bytes::shared_to_vec` 1.2%. These originate from per-token `EngineCoreOutputs` construction, msgpack encode/decode buffers, and SSE framing in hyper. This is a measured frontend cost — the simulated engine itself is only 0.9%.

3. **Stream polling is the second-largest category** (~7.5% combined self). `poll_next_unpin` 4.7% + `Instrumented::poll_next` 2.8%. Each token traverses a 5-layer poll chain: `mpsc::UnboundedReceiver` → `EngineCoreOutputStream` → `GenerateOutputStream` → `decoded_text_event_stream` → `completion_chunk_stream`. The `Instrumented` wrapper appears at every layer, doubling the poll overhead per hop.

4. **The ~145ms TTFT overhead is measured but not decomposed.** The simulated floor is ~5ms; the observed p50 is ~155ms. Perf measures *where CPU is spent* but cannot directly attribute wall-clock latency to individual functions in an async workload. The 145ms gap is real and frontend-attributed (sim engine does `tokio::time::sleep(5ms)` then immediately sends the first token), but its internal breakdown requires instrumentation timestamps, not inference from perf samples.

5. **IPC bridge CPU is low (~1%), but latency contribution is unknown.** `PushSocket::send` 0.7% + `mpsc::Tx::push` 0.7% show the ZMQ path is not CPU-bound. Whether it contributes 5ms or 50ms to the 145ms TTFT overhead cannot be determined from perf alone — that depends on syscall + scheduling latency per hop, which perf stat does not capture.

6. **Low instructions-per-cycle (0.25) and 58% cache miss rate** are consistent with the pointer-heavy, allocation-scattered profile. This is a measured system property, not a model-side artifact.

## Proposed Optimization Directions

Each direction below is tied to the specific measured frontend overhead from the perf data above.

### 1. Reduce IPC bridge hops for single-engine deployments

**Measured basis**: The data path crosses 5 mpsc channels + 1 ZMQ Unix socket between `run_simulated_request` and `completion_sse_stream` (confirmed by source trace in `pegainfer-vllm-frontend/src/lib.rs` lines 303–847). The `output_loop` serializes all requests through a single `PushSocket`.

**Direction**: For single-engine (non-distributed) deployments, bypass ZMQ and connect `LocalEngineBridge` directly through an in-process mpsc channel. This would remove the `encode_msgpack` → ZMQ send → ZMQ recv → `decode_msgpack` round-trip and its associated allocation (`rmp_serde::Decoder::any_inner` at 0.5%).

**Risk**: Changes the `LocalEngineBridge` abstraction; must not break multi-engine transport mode.

### 2. Reduce per-token allocation in EngineCoreOutputs construction

**Measured basis**: Heap allocation is 10% of measured CPU (`malloc` 3.2%, `realloc` 3.9%, `RawVecInner::finish_grow` 1.5%). `send_token_output()` creates a new `EngineCoreOutputs` with nested `Vec`s on every token batch.

**Direction**: Pre-allocate reusable output buffers on the bridge task and clear/reuse them across token emissions. Batch more tokens per output (the existing `collect_ready_token_batch` already does this for decode tokens) to amortize allocation.

**Risk**: Requires careful buffer lifecycle management across the async output path.

### 3. Flatten the stream poll chain

**Measured basis**: Stream polling overhead is 7.5% of measured CPU (`poll_next_unpin` 4.7% + `Instrumented::poll_next` 2.8%). Each token traverses 5 `#[try_stream]` generator layers.

**Direction**: Merge adjacent stream layers where possible. The `completion_chunk_stream` → `completion_sse_stream` → `Sse::new` chain could be collapsed into a single stream that yields `Event` directly, removing 2 poll hops and their associated `Instrumented` overhead. This is constrained by the vllm-server crate boundary (external dependency).

**Risk**: Tight coupling to vllm-server internal APIs; upstream changes may require rework.

### Investigation needed (not yet a proposal)

- **TTFT overhead decomposition**: Add `Instant::now()` timestamps at `LocalEngineBridge::start_request` entry, after `handle.submit()`, after first `token_rx.recv()`, and at `send_token_output` to decompose the 145ms gap into measured phases. Without this data, any per-phase latency estimate would be speculation.
- **ITL spike characterization**: The ~300ms ITL spikes at p95/p99 appear in both load levels. Determine whether these originate from `output_loop` serialization, tokio scheduling starvation, or ZMQ back-pressure.
