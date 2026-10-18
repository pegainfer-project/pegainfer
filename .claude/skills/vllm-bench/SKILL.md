---
name: vllm-bench
description: "Use vllm-bench to benchmark OpenAI-compatible or vLLM serving endpoints, especially multi-turn chat load tests. Use when the user asks for vllm bench, vllm-bench, multi-turn benchmark, chat serving benchmark, TTFT/TPOT/throughput measurement, concurrency sweep, load test, 压测, 多轮压测, or wants help installing, running --help, choosing flags, validating a local PegaInfer/vLLM server, saving JSON results, or interpreting vllm-bench metrics."
---

# vllm-bench

`vllm-bench` is a Rust benchmark client for online LLM serving. It does not run
the model. It drives an already running OpenAI-compatible or vLLM endpoint and
reports TTFT, TPOT, ITL, E2EL, throughput, and multi-turn per-turn breakdowns.

Use this workflow for multi-turn work:

1. Install or locate `vllm-bench`.
2. Run `vllm-bench --help` and use it as the authority for current flags.
3. Confirm the server base URL and model id from `/v1/models`.
4. Dry-run the dataset locally.
5. Run a tiny smoke benchmark.
6. Scale conversations, turns, and concurrency only after the smoke passes.
7. Save JSON for any result that may be compared later.

## Install

Install from the upstream repository:

```bash
cargo install --git https://github.com/vllm-project/vllm-bench vllm-bench
```

The trailing `vllm-bench` package name matters: the repo also contains another
binary, so omitting it can fail with "multiple packages with binaries found".
Cargo installs the binary to `~/.cargo/bin/vllm-bench`.

Alternatives from the README:

```bash
curl -fsSL https://github.com/vllm-project/vllm-bench/releases/latest/download/vllm-bench-$(uname -m)-linux-musl \
  -o vllm-bench && chmod +x vllm-bench

git clone https://github.com/vllm-project/vllm-bench.git
cd vllm-bench
./install.sh
```

After install, run:

```bash
vllm-bench --version
vllm-bench --help
```

## Choose Backend

For multi-turn chat, always use:

```bash
--backend openai-chat
```

Common backends:

| Backend | Endpoint | Use |
| --- | --- | --- |
| `vllm` / `openai` | `/v1/completions` | text completions |
| `openai-chat` | `/v1/chat/completions` | chat, multi-turn, multimodal |
| `openai-embeddings` | `/v1/embeddings` | text embeddings |
| `vllm-rerank` | `/v1/rerank` | rerank |

Prefer `http://127.0.0.1:<port>` over `localhost`; some servers bind IPv4 while
`localhost` resolves to IPv6.

Confirm the model id before benchmarking:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

Use the returned id as `--model`, or omit `--model` and let `vllm-bench`
auto-fetch it.

## Multi-Turn Flags

Core flags:

```bash
--multi-turn
--multi-turn-num-turns <N>
--num-prompts <CONVERSATIONS>
--multi-turn-concurrency <CONCURRENT_CONVERSATIONS>
```

In multi-turn mode, `--num-prompts` means conversation count, not request count.
A run with `--num-prompts 50 --multi-turn-num-turns 5` can send up to 250 chat
requests.

Synthetic random conversations:

```bash
--dataset-name random
--random-input-len <TOKENS_FOR_TURN_1>
--per-turn-input-len <TOKENS_FOR_TURNS_2_PLUS>
--random-output-len <OUTPUT_TOKENS_PER_TURN>
```

If `--per-turn-input-len` is omitted or `0`, later turns reuse
`--random-input-len`. Use `--multi-turn-min-turns` and `--multi-turn-max-turns`
when conversation lengths should vary.

Think time and prefix sharing:

```bash
--multi-turn-delay-ms 500
--multi-turn-prefix-global-ratio 0.2
--multi-turn-prefix-conversation-ratio 0.5
```

Prefix sharing works only with `--dataset-name random`; the two ratios must sum
to less than `1.0`.

When either prefix-sharing flag is non-zero, `vllm-bench` switches to fixed-
length per-turn prompts with no accumulated chat history. Use this only for
prefix-cache experiments; leave both ratios at `0` for realistic multi-turn
conversation growth.

## Dry Run

Dry-run before sending traffic. It loads the tokenizer and builds conversations
without hitting the server:

```bash
vllm-bench \
  --backend openai-chat \
  --model Qwen/Qwen3-4B \
  --tokenizer <model-path> \
  --dataset-name random \
  --multi-turn --multi-turn-num-turns 2 \
  --random-input-len 16 --random-output-len 4 \
  --num-prompts 2 \
  --dry-run
```

`<model-path>` is a local tokenizer/model directory; for a public example,
replace it with the served model id or any local model directory.

## Example: Generic vLLM Multi-Turn

Use this after a vLLM OpenAI server is already running on port 8000:

```bash
vllm-bench \
  --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --model <model-id-from-/v1/models> \
  --dataset-name random \
  --multi-turn \
  --multi-turn-num-turns 5 \
  --random-input-len 512 \
  --per-turn-input-len 256 \
  --random-output-len 128 \
  --num-prompts 50 \
  --multi-turn-concurrency 10 \
  --ready-check-timeout-sec 60 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --save-result
```

Read the output as:

- `Conversations completed/total`: end-to-end conversation success.
- `Successful requests`: successful turns.
- `Total input tokens`: includes growing chat history in normal multi-turn mode.
- `per_turn_metrics` in JSON: TTFT/TPOT/ITL/E2EL by turn index.

Expect later turns to have larger input token counts because each request sends
the prior user/assistant history plus the next user message.

## Example: Local PegaInfer Qwen3-4B Smoke

This was validated locally with PegaInfer Qwen3-4B and `vllm-bench` built from
this repo. Start PegaInfer:

```bash
cd <pegainfer-repo>
cargo run --release -p pegainfer-server -- \
  --model-path <model-path> \
  --served-model-name Qwen/Qwen3-4B \
  --port 18080 \
  --cuda-graph=false
```

Confirm the model:

```bash
curl -s http://127.0.0.1:18080/v1/models
```

Run a tiny multi-turn benchmark:

```bash
cd <vllm-bench-repo>
target/release/vllm-bench \
  --backend openai-chat \
  --base-url http://127.0.0.1:18080 \
  --model Qwen/Qwen3-4B \
  --tokenizer <model-path> \
  --dataset-name random \
  --multi-turn \
  --multi-turn-min-turns 2 \
  --multi-turn-max-turns 3 \
  --random-input-len 32 \
  --per-turn-input-len 12 \
  --random-output-len 8 \
  --num-prompts 3 \
  --multi-turn-concurrency 2 \
  --ready-check-timeout-sec 30 \
  --extra-body '{"min_tokens":null}' \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90 \
  --save-result \
  --result-dir /tmp \
  --result-filename vllm-bench-multi-turn-smoke.json
```

The `--extra-body '{"min_tokens":null}'` is for PegaInfer compatibility. The
random multi-turn path auto-adds `min_tokens` unless overridden; this pins
output length on vLLM but PegaInfer currently rejects that field. `--ignore-eos`
also skips `min_tokens` and worked in local smoke testing, but it asks the
server to ignore EOS and can grow contexts more aggressively.

The local smoke run completed 3/3 conversations, 8/8 turns, and wrote
`/tmp/vllm-bench-multi-turn-smoke.json` with `per_turn_metrics`.

## Scale Safely

After the smoke passes:

1. Increase `--num-prompts` first to improve measurement stability.
2. Increase `--multi-turn-concurrency` to find the service knee.
3. Increase `--random-input-len`, `--per-turn-input-len`, and
   `--random-output-len` to match the target workload.
4. Save every serious run with `--save-result --metadata KEY=VALUE`.

For deterministic latency A/B runs, pass `--temperature 0` explicitly. Omit it
only when the benchmark is intentionally measuring server default generation
behavior.

Use a sweep when searching for the concurrency knee:

```bash
vllm-bench \
  --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --model <model-id> \
  --dataset-name random \
  --multi-turn --multi-turn-num-turns 5 \
  --random-input-len 512 --per-turn-input-len 256 --random-output-len 128 \
  --num-prompts 200 \
  --sweep-max-concurrency 1,2,4,8,16,32 \
  --sweep-summary-percentiles 90,99 \
  --save-result
```

For A/B comparisons, keep the workload flags identical and compare result JSON:

```bash
vllm-bench --compare baseline.json candidate.json
```

## Troubleshooting

- `min_tokens is not supported by this engine`: pass
  `--extra-body '{"min_tokens":null}'` or, if appropriate, `--ignore-eos`.
- All-zero metrics with failed requests: inspect the first HTTP error printed
  above the result block; it usually names the unsupported request field.
- Connection fails on `localhost`: switch to `127.0.0.1`.
- Model id mismatch: use `curl /v1/models` and set `--model` to that exact id.
- Tokenizer load fails: set `--tokenizer` to a local model directory with
  `tokenizer.json`, or rely on server-side tokenize/detokenize fallback if the
  server implements it.
- Dataset construction is suspicious: run the same command with `--dry-run`.
- Long serious runs need reproducibility: set `--seed`, `--result-filename`,
  and `--metadata` entries for server commit, model, GPU, and key flags.

## Metrics

Use database-style thinking: requests are transactions, prompt tokens are read
set size, output tokens are write volume, and concurrency is the number of
in-flight conversations.

- TTFT: queueing plus prefill latency until the first output token.
- TPOT: decode time per output token after the first token.
- ITL: per-token latency distribution.
- E2EL: total turn latency.
- Request throughput: turns per second.
- Output throughput: generated tokens per second.
- Multi-turn JSON: inspect `per_turn_metrics` to see how latency changes as
  history grows.
