# Simulated Inference Engine

> **TL;DR:** `pegainfer-sim` is a CPU-only `Scheduler` that serves through the vLLM/OpenAI frontend with configurable TTFT/TPOT. It launches as `LaunchedEngine::Stepped`. It is a frontend/bench harness, not a real-model performance path.
>
> **Last touched:** 2026-08

## Scope

A server path that can run `vllm bench serve` and the frontend HTTP e2e suite without GPU or model weights, while still exercising the same `pegainfer-frontend` stack used by real model lines.

Out of scope:

- No CUDA, kernel, KV-cache, or real model execution.
- No claim about real model serving throughput.
- No jitter, tail-latency distribution, or batching realism beyond fixed TTFT/TPOT.

## Behavior

CLI knobs: model id, port, max model length, base TTFT, prefill throughput, TPOT, fallback token id.

Timing: TTFT is `base_ttft_ms + prompt_len / prefill_tokens_per_ms`; TPOT is a fixed delay between generated tokens. `SimScheduler::step` emits at most one token per request per step and parks up to 1ms while waiting, so a CPU-only sim does not spin a core the way a GPU scheduler can.

Output token ids cycle through the prompt tokens, or replay a scripted sequence (tool-call tests). Empty prompts use the fallback id.

The frontend still needs tokenizer/model metadata; the simulator never loads weights.

## Frontend Metadata Contract

Serving through the vLLM/OpenAI frontend still constructs the normal text/chat backend. That path needs enough local metadata to tokenize and detokenize.

For CPU-only tests that do not intend to exercise tokenizer encoding, use token-id prompts. Generated token ids still pass through detokenization, so the fixture must provide at least `tokenizer.json`. `tokenizer_config.json` and `config.json` are useful for EOS and context-window metadata; no weight files are required.

Chat-completions tests also need a `chat_template` in `tokenizer_config.json`. Keep the template deterministic and ensure it renders at least one token the simulated engine can stream as observable content; otherwise response-shape tests can pass without exercising `delta.content`.

## Implementation

`SimScheduler` implements the step-contract `Scheduler` (same shape as `pegainfer-frontend/examples/echo-server.rs`). `start_engine` / `start_engine_with_partitions` return an `Engine`; the CLI and tests pass `engine.into()` into `vllm::serve` as `LaunchedEngine::Stepped`.

`pegainfer-sim` stays a separate crate so simulation changes do not live inside real model crates. It already depends on `pegainfer-frontend` (`pegainfer-sim/Cargo.toml`).

The cutover record is `sim-step-contract.md`.
