# pegainfer-sample — the shared sampling layer

**TL;DR:** `pegainfer-sample` is the crate qwen3, qwen35, and Kimi-K2 route through to turn a logits arena into next-token ids (`select_batch`) and into host logprobs (`token_logprob_from_row`). It replaces `pegainfer-core::ops::select_batch_tokens_into` (deleted) and the three copies of `compute_logprobs_from_cpu`/`host_token_logprob`. Kimi keeps its sharded-vocab greedy argmax (a DP concern the crate can't express) but routes non-greedy sampling and logprobs through here. The DeepSeek lines do **not** route through — both are greedy-only and pick with a bare argmax (v2-lite on the GPU via `core::ops::argmax`, v4 on the host via `argmax_f32`), so they need no batched sampling policy. See *Next step* for the V4 host-argmax caveat.

Last touched: 2026-06

## Why a crate

Token selection and host logprob math are model-agnostic — the only model-shaped input is vocab width. Before this, the selection logic lived in `pegainfer-core::ops` (re-exporting kernels primitives) and the logprob math was copy-pasted three times (qwen3 `executor.rs`, qwen35 `logprobs.rs`, kimi `runtime.rs`), each drifting slightly (f32 vs f64 accumulation, opposite top-k tie-breaking, unchecked vs checked indexing).

A dedicated crate makes the door from logits to tokens explicit and lets `pegainfer-core::ops` drop the sampling *policy* it used to forward (`select_batch_tokens_into`, `sampling_params_effectively_greedy`). The bare argmax primitives (`argmax`, `argmax_batch_bf16_into`) stay in the `core::ops` facade — they are kernel ops, not sampling policy, and DeepSeek-V2-Lite's single-rank greedy still reaches them there. The crate sits **beside** `pegainfer-core`, atop the two lower layers it actually needs:

- `pegainfer-engine` — the CUDA-free contract crate: `SamplingParams`, `TokenLogprob`.
- `pegainfer-kernels` — the CUDA build owner: `gpu_sample_batch_into`, the batched argmax, scratch sizing.

It deliberately does **not** depend on `pegainfer-core` (which would be a cycle in spirit — core is the per-model runtime contract, sampling is a leaf utility).

## What it owns

- **`select_batch(ctx, logits, params: &[&SamplingParams], seed, &mut SampleScratch) -> Vec<u32>`** — one next-token id per arena row. Greedy rows (and effectively-greedy ones, see below) resolve together through a batched indexed argmax; the remaining rows compact into one FlashInfer temperature/top-k/top-p pass. There is no per-row escape hatch, so a caller cannot regress to `for i { sample(i) }`. This is the #284 batched-selection logic (see `subsystems/scheduler/qwen-batched-sampling.md`) lifted verbatim into the crate.
- **`SampleScratch`** — the allocate-once device buffers for `select_batch`, sized `max_rows × vocab`. Consolidates the six loose scratch buffers callers used to thread by hand (row indices, argmax partials ×2, top-1 values, argmax out, the FlashInfer sampling scratch) into one struct. Decode needs pointer-stable buffers, so it is reused across steps, never reallocated per step. `max_rows()` lets a caller grow it when a batch exceeds the bucket it was sized for. The vocab width is baked into every buffer, so `select_batch` rejects a logits arena of a different width — a mismatch would otherwise be a silent kernel OOB rather than a clean error.
- **`token_logprob_from_row<T: Copy + Into<f32>>(row, picked, top_k) -> Option<TokenLogprob>`** — host log-softmax of the picked token plus the top-k. Generic over the row element so qwen feeds `f32` (already host-side) and Kimi feeds `bf16` straight from the device arena, with **no widening copy**. The crate also re-exports `gpu_sample_batch_into` / `BatchSamplingRow` / `BatchSamplingScratch` for the one caller (Kimi) that drives its own greedy pass.

## Two decisions worth keeping

**The effectively-greedy predicate.** A row takes the deterministic argmax path when it is explicitly greedy **or** when `top_p <= 1/vocab`. The softmax maximum is always `>= 1/vocab`, so that bound makes the nucleus exactly one token — the argmax. Routing it to argmax keeps an effectively-greedy request reproducible: the rejection sampler would otherwise pick an arbitrary member of a bf16-tied top, and (since the seed is per-step, not per-request) that would surface as nondeterminism. The `top_p > 0.0 && top_p.is_finite()` guards matter — a degenerate `top_p = 0` must fall through to the sampler, not argmax.

**Unified logprob math.** The three prior copies were reconciled to one: f64 exponential accumulation (a 160k-wide bf16 vocab loses precision summing exps in f32) and ascending-token-id tie-breaking in the top-k (the order among exactly-tied logits is otherwise unspecified; ascending is the deterministic choice and the one Kimi's test pins). qwen's top-k tie order flips as a result — invisible to its gates, which compare logits and token ids, never top-logprob ordering.

## Where Kimi-K2 stops

Kimi's **non-greedy** sampling is exactly `gpu_sample_batch_into`, so it shares the crate's primitive. Its **logprobs** go through `token_logprob_from_row` (the bf16 path). But its **greedy/top-1 path stays model-side** and must: Kimi runs a *vocab-sharded* local argmax (`vocab_start` may be `!= 0`) whose top-1 **logit value** (`local_top_logit_f32`) feeds a cross-rank DP reduction to recover the global argmax (#236/#237). `select_batch` assumes a whole, unsharded vocab arena and returns only token ids — it cannot express the shard-local-then-reduce protocol. Folding that into a "model-agnostic" crate would drag DP/sharding into it; that is the wrong boundary. So Kimi depends on the crate for the two pieces that *are* model-agnostic and keeps the one that is not.

## Seed model

One engine seed at startup (`StdRng::seed_from_u64`), advanced to a fresh `u64` per decode step and passed into `select_batch`. Rows decorrelate through the philox subsequence; there is deliberately no per-request RNG. Same seed → same tokens (the crate's `sampling_is_seed_deterministic_and_actually_samples` test pins this).

## Tests

- `pegainfer-sample` lib tests (CPU, no GPU): `token_logprob_from_row` over bf16 + f32 inputs, exact log-softmax, tie order, k>vocab, empty/out-of-range guards.
- `pegainfer-sample/tests/select_batch.rs` (GPU, model-free): greedy/sampling/mixed routing, per-row placement, scratch reuse, the tiny-top_p-under-bf16-ties regression, and the capacity-rejection invariant.
- Model gates downstream: qwen3 `sampling_behavior` + `hf_golden_gate`, qwen35 `sampling_behavior` + `e2e_scheduler`.

## Next step

Kimi's prefill single-row greedy still uses a separate FFI entry (`flashinfer_top1_cuda`) from its batched decode argmax (`argmax_batch_bf16_split_cuda`); unifying those two is a Kimi-local cleanup, not a crate concern. If a fourth model needs sharded selection, revisit whether a `select_batch_sharded` (returning ids + top-1 values) earns its place — today it would have exactly one caller.
