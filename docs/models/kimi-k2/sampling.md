# Kimi-K2 sampling: param surface and design

**TL;DR**: temperature/top_k/top_p are honored on TP1/DP8 via one batched FlashInfer pass (greedy rows keep the in-graph argmax, zero perf cost); TP8 rejects non-greedy explicitly; everything else on the OpenAI surface is documented below — nothing is silently ignored anymore (#237).

Last touched: 2026-06

## Param surface (`/v1/completions`)

What a client can send vs. what actually happens. "Frontend" = the vllm-server
OpenAI layer + `pegainfer-vllm-frontend` conversion
(`pegainfer-vllm-frontend/src/lib.rs` `convert_sampling`); "engine" = the kimi
scheduler/worker.

| Param | TP1/DP8 | TP8 | Where decided |
|---|---|---|---|
| `temperature` > 0 | **honored** (per-row, batched FlashInfer) | **rejected** at admission | engine |
| `temperature` = 0 | **honored** (greedy argmax) | honored (greedy) | frontend normalizes to `{0.0, top_k=-1, top_p=1.0}` |
| `temperature` < 0 | treated as 0 → greedy | same | frontend (`convert_sampling` lowers `<= 0` to greedy; vLLM would 400 — divergence, documented) |
| `top_p` ∈ (0,1] | **honored** | rejected if non-greedy | engine |
| `top_p` = 0 / out of range | **rejected** (HTTP 500, see below) | rejected | engine (`lifecycle.rs validate_sampling_params`) |
| `top_k` ≥ 1 | **honored** (`top_k=1` routes greedy) | rejected if non-greedy | engine |
| `top_k` = 0 | all tokens (disabled) | — | frontend maps 0 → -1; protocol type is `u32`, negatives don't parse |
| `seed` | **accepted, ignored** — engine seed is fixed at 42, per-request seed is dropped at `convert_sampling` | same | frontend |
| `logprobs` | honored; for sampled rows the logprob follows the **sampled** token (reported rank is a placeholder, see PR #96) | honored (greedy only) | engine |
| `max_tokens`, `echo`, `stop` (EOS) | honored | honored | engine / frontend |
| `min_p`, `frequency_penalty`, `presence_penalty`, `repetition_penalty`, `logit_bias`, `min_tokens`, `prompt_logprobs`, custom `stop_token_ids` | **accepted, ignored** — dropped at `convert_sampling`, never reach the engine | same | frontend (all models, not kimi-specific) |

Rejection UX pitfall: an engine-side rejection surfaces as a generic HTTP 500
(`"Internal server error"`). The real message ("top_p must be in (0, 1]…",
"Kimi TP8 path does not support sampling yet…") goes through
`TokenEvent::Rejected` → engine-core `Error` finish, and the upstream
vllm-server OpenAI layer swallows the text. Check the server log
(`vllm_engine_core_client::client::stream "request failed"`) when a client
reports a 500. Fixing the mapping is a vllm-rust-workspace change, not a
pegainfer one.

## Design (TP1/DP8)

- Classification: `SamplingParams::is_greedy()` = `temperature < 1e-5 || top_k == 1`
  (vLLM's epsilon; tiny temperatures overflow `1/temperature`). Greedy rows must
  never enter the FlashInfer softmax — temperature 0 there means *uniform*, not
  argmax.
- Greedy rows: unchanged in-graph batched argmax. All-greedy batches launch
  zero sampling kernels and allocate nothing.
- Non-greedy rows: one batched pass per decode step —
  `gather_cast_logits_f32` (compact bf16→f32) → vendored `OnlineSoftmax<float>`
  (per-row temperature, in-place, vocab-splitting path for 160K vocab) →
  `TopKTopPSamplingFromProb` (per-row top_k/top_p arrays, deterministic CDF) →
  one D2H+sync. 3 kernels for the whole batch, not per row.
- Scratch (`SamplingScratch::batch_sampling`) is allocated lazily on the first
  sampling request (~42 MB at batch 64 × 163840 vocab f32) — greedy-only
  serving pays nothing.
- Seeds: `DpCoordinator` owns a `StdRng` seeded from the engine seed (42) and
  draws a fresh u64 per rank per step. Ranks must not share a step seed: the
  philox subsequence inside FlashInfer is the *row index*, so same-index rows
  on two ranks with the same seed would draw identical uniforms.
- TP8: each rank holds a vocab shard; per-shard softmax cannot express the
  global distribution. Non-greedy requests are rejected per-request at
  admission (one bad request must not fail the decode batch), and
  `ensure_greedy_tp8` in the executor backstops the invariant (#226).

## Verified on 8×H200 (K2.6, TP1/DP8 PPLX, 2026-06-06)

- Greedy determinism: identical solo requests → byte-identical output.
- Sampling: `temperature=0.8, top_p=0.9` × 3 → three distinct continuations.
- `top_k=1` + `temperature=0.8` → deterministic (greedy route).
- Mixed concurrent batches (greedy + sampled): greedy rows behave exactly like
  an all-greedy concurrent control. Note the control itself diverges across
  concurrent identical requests — batch-composition bf16 numerics on near-tie
  argmax, pre-existing and unrelated to sampling.
- Greedy bs64 TPOT: p50 29.63ms / p99 31.75ms vs main baseline 30.07/31.70 —
  no regression.

## Next

- Per-request `seed` support (currently engine-global): needs a seed field
  through `GenerateRequest` and per-row philox offsets instead of the
  per-step scalar.
- Migrate qwen3/qwen35 off the per-row single-block sampler onto
  `gpu_sample_batch_into` (see the qwen sampling audit issue).
