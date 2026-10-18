# Qwen3.5 Phase 2A TP2 Multi-Turn Serving Gate

> **TL;DR:** A real 2x RTX 3090 Qwen3.5 TP2 server completed 12/12 dependent conversations and 44/44 measured turns at client concurrency 4 over server capacity 2, then admitted and completed another 4/4 conversations and 8/8 turns without restart; all requests produced the configured token counts, history grew on every turn, shutdown was clean, and both GPUs returned to idle.

This is a correctness and lifecycle gate for Phase 2A. It is not a comparative
performance claim. The run was captured on 2026-08-19 UTC.

## Pass Criteria

The run passes only when all of the following hold:

1. The dry run builds 12 conversations with 2-5 turns and no server traffic.
2. The primary run completes 12/12 conversations and 44/44 measured turns with zero failures.
3. Every measured turn returns exactly 24 output tokens, for 1,056 output tokens total.
4. Per-turn sample counts prove mixed conversation lengths, and prompt accounting grows with carried chat history.
5. Client concurrency 4 completes against `max_batch=2` without deadlock or rank divergence.
6. Without restarting the server, a second probe completes 4/4 newly admitted conversations and 8/8 turns with zero failures.
7. Graceful shutdown exits the scheduler and releases both GPUs.

All seven criteria passed.

## Reproduction Pins

| Component | Value |
| --- | --- |
| PegaInfer server commit | `8829189992d9290f1aa128fb00950dd854f43211` |
| PegaInfer binary SHA-256 | `97ab83bbf666add4e4b0854c1bea77688279ca27dc87bc15641fb7d7e29cd9af` |
| vLLM client commit | `2b7fcbf52782f8729fd6ce6c9ab803617d72897b` |
| `vllm-bench` version | `0.1.0` |
| `vllm-bench` binary SHA-256 | `8131ed513d22da21186eb7ccba06dfb6d0c8657624bb72a733948841a5dd1ffe` |
| GPUs | 2x NVIDIA RTX 3090, SM86, 24 GiB each |
| Model fixture | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a), revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `config.json` SHA-256 | `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670` |
| `tokenizer.json` SHA-256 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `model.safetensors.index.json` SHA-256 | `cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d` |
| `model.safetensors-00001-of-00002.safetensors` SHA-256 | `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61` |
| `model.safetensors-00002-of-00002.safetensors` SHA-256 | `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188` |
| TP/runtime | TP=2, eager, CUDA Graph disabled, `max_batch=2`, `max_prefill_tokens=64`, scheduler policy `off` |

The repository's older pinned vLLM revision predates the Rust `vllm-bench`
workspace. The client was therefore built from the exact maintained upstream
commit above. Source inspection confirmed that `--multi-turn` runs turns
sequentially within each conversation, appends each assistant response, and
sends the accumulated message history on the next turn. A separate sequential
chat driver was not needed.

Set these paths before running the commands. `MODEL_DIR` must point to the
pinned public revision above and match the complete config, tokenizer, index,
and weight-shard hash manifest; no private absolute model path is part of the
reproduction contract.

```bash
MODEL_DIR=/path/to/public/Qwen3.5-4B
VLLM_SRC=/tmp/vllm-phase2a-bench-src
VLLM_BENCH="$VLLM_SRC/rust/target/release/vllm-bench"
RESULT_DIR=/tmp/qwen35-tp2-phase2a-results
```

Build the pinned client:

```bash
git clone --filter=blob:none --no-checkout https://github.com/vllm-project/vllm.git "$VLLM_SRC"
git -C "$VLLM_SRC" fetch --depth 1 origin 2b7fcbf52782f8729fd6ce6c9ab803617d72897b
git -C "$VLLM_SRC" checkout --detach 2b7fcbf52782f8729fd6ce6c9ab803617d72897b
cargo build --release -p vllm-bench --manifest-path "$VLLM_SRC/rust/Cargo.toml"
"$VLLM_BENCH" --version
```

Build and start PegaInfer from the server commit above:

```bash
PEGAINFER_CUDA_SM=86 \
PEGAINFER_TRITON_PYTHON="$PWD/.venv/bin/python" \
PROTOC="$PWD/.venv/protoc-root/usr/bin/protoc" \
PROTOC_INCLUDE="$PWD/.venv/protoc-root/usr/include" \
LD_LIBRARY_PATH="$PWD/.venv/protoc-root/usr/lib/x86_64-linux-gnu" \
cargo build --offline --release --locked \
  -p pegainfer-server --no-default-features --features qwen35
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PEGAINFER_CUDA_SM=86 \
PEGAINFER_TRITON_PYTHON="$PWD/.venv/bin/python" \
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/nccl/lib" \
RUST_LOG=info \
target/release/pegainfer \
  --model-path "$MODEL_DIR" \
  --served-model-name qwen35-tp2-phase2a \
  --port 18080 \
  --tp-size 2 \
  --cuda-graph=false \
  --max-batch 2 \
  --max-prefill-tokens 64 \
  --qwen35-scheduler-policy off
```

## Workload Commands

The dry run fixes the generated dataset before server traffic:

```bash
"$VLLM_BENCH" \
  --backend openai-chat \
  --model qwen35-tp2-phase2a \
  --tokenizer "$MODEL_DIR" \
  --dataset-name random \
  --multi-turn \
  --multi-turn-min-turns 2 \
  --multi-turn-max-turns 5 \
  --random-input-len 128 \
  --per-turn-input-len 64 \
  --random-output-len 24 \
  --num-prompts 12 \
  --multi-turn-concurrency 4 \
  --max-model-len 2048 \
  --seed 446 \
  --dry-run
```

It produced 12 conversations, 44 turns, and 3,584 user-message tokens.
The measured primary run used the same seed and shape:

```bash
mkdir -p "$RESULT_DIR"
"$VLLM_BENCH" \
  --backend openai-chat \
  --base-url http://127.0.0.1:18080 \
  --model qwen35-tp2-phase2a \
  --tokenizer "$MODEL_DIR" \
  --dataset-name random \
  --multi-turn \
  --multi-turn-min-turns 2 \
  --multi-turn-max-turns 5 \
  --random-input-len 128 \
  --per-turn-input-len 64 \
  --random-output-len 24 \
  --num-prompts 12 \
  --multi-turn-concurrency 4 \
  --max-model-len 2048 \
  --seed 446 \
  --ignore-eos \
  --temperature 0 \
  --ready-check-timeout-sec 30 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --save-result \
  --save-detailed \
  --result-dir "$RESULT_DIR" \
  --result-filename qwen35-tp2-phase2a-multiturn.json \
  --metadata \
    server_commit=8829189992d9290f1aa128fb00950dd854f43211 \
    client_commit=2b7fcbf52782f8729fd6ce6c9ab803617d72897b \
    gpu=2x_RTX_3090_sm86 \
    tp_size=2 \
    server_max_batch=2 \
    server_max_prefill_tokens=64
```

The post-workload probe was run immediately afterward, without restarting or
reloading the server:

```bash
"$VLLM_BENCH" \
  --backend openai-chat \
  --base-url http://127.0.0.1:18080 \
  --model qwen35-tp2-phase2a \
  --tokenizer "$MODEL_DIR" \
  --dataset-name random \
  --multi-turn \
  --multi-turn-num-turns 2 \
  --random-input-len 32 \
  --per-turn-input-len 16 \
  --random-output-len 8 \
  --num-prompts 4 \
  --multi-turn-concurrency 4 \
  --max-model-len 512 \
  --seed 447 \
  --ignore-eos \
  --temperature 0 \
  --ready-check-timeout-sec 30 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --save-result \
  --result-dir "$RESULT_DIR" \
  --result-filename qwen35-tp2-phase2a-post-readmission.json \
  --metadata \
    purpose=post_workload_readmission \
    server_commit=8829189992d9290f1aa128fb00950dd854f43211 \
    client_commit=2b7fcbf52782f8729fd6ce6c9ab803617d72897b \
    tp_size=2 \
    server_max_batch=2
```

## Results

| Metric | Primary run | Post-workload probe |
| --- | ---: | ---: |
| Conversations completed/failed | 12 / 0 | 4 / 0 |
| Measured turns completed/failed | 44 / 0 | 8 / 0 |
| Readiness requests, unmeasured | 1 | 1 |
| Client concurrency | 4 | 4 |
| Duration | 13.865 s | 0.786 s |
| Input tokens | 12,496 | 416 |
| Output tokens | 1,056 | 64 |
| Request throughput | 3.173 req/s | 10.185 req/s |
| Output throughput | 76.164 tok/s | 81.475 tok/s |

Primary latency percentiles:

| Metric | Mean | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: |
| TTFT | 727.54 ms | 727.69 ms | 847.64 ms | 899.25 ms |
| TPOT | 21.70 ms | 21.55 ms | 22.56 ms | 22.93 ms |
| ITL | 19.97 ms | 20.60 ms | 26.92 ms | 30.08 ms |
| E2EL | 1226.73 ms | 1222.62 ms | 1365.46 ms | 1406.76 ms |

Turn-level evidence from the primary run:

| Turn | Conversations reaching turn | Client-accounted input per request | Server prompt-token range | Output per request |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 152 | 138-140 | 24 |
| 2 | 12 | 240 | 212-214 | 24 |
| 3 | 9 | 328 | 286-288 | 24 |
| 4 | 8 | 416 | 360-362 | 24 |
| 5 | 3 | 504 | 435 | 24 |

The `[12, 12, 9, 8, 3]` sample vector proves that conversations had mixed 2-5
turn lengths. Both the client accounting and the server's rendered prompt
tokens increase per turn, which proves that later requests carried earlier
user and assistant messages instead of flattening the dataset to its first
turn. Client and server token counts differ because the server applies the chat
template and tokenizer to the accumulated messages.

This client revision still writes the default value `turns_per_conversation=3`
in the top-level JSON even when min/max turn sampling is enabled. The measured
per-turn sample vector and `avg_turns_completed=3.6667` are the authoritative
values for this variable-length run.

The client emitted a warning that `--ignore-eos` can interact with multi-turn
output limits. In this run, every one of the 44 server responses logged
`output_tokens=24` with `finish_reason=length`; the saved result totals exactly
`44 * 24 = 1,056` output tokens. The post-workload probe similarly totals
`8 * 8 = 64`.

The primary result is stored in
[qwen35-tp2-phase2a-multiturn.json](qwen35-tp2-phase2a-multiturn.json), SHA-256
`9247a35bc3f5475ec59397ddd2dd3461b9c033e1d27ef25f64c7517da5398676`.
The no-restart readmission result is stored in
[qwen35-tp2-phase2a-post-readmission.json](qwen35-tp2-phase2a-post-readmission.json),
SHA-256 `a60890446d3bb08f83da37bf7c14c2a9063eccbdc480ae3cbb028ccd52ce2573`.

After the second run, Ctrl-C produced `scheduler: all handles dropped, exiting`
and process exit code 0. A subsequent `nvidia-smi` showed both TP devices at 1
MiB used, 24,126 MiB free, and 0% utilization, confirming process-level resource
release.
