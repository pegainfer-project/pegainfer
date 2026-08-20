# Qwen3.5 Speculative Verifier

> **TL;DR:** PR #667 extracts the target-only verifier from #626; for the C16, prompt-1024, span-5 test, reusing existing Q/K prep and paged K/V scatter cut aggregate prep-kernel GPU time by 21.1%, while drafter and serving wiring remain follow-ups.
>
> **Last touched:** 2026-07

## Contract

- Target verification only; no draft model, scheduler/server wiring, sampling, or serving-performance claim.
- Verifier reuses the existing batched Q/K prep and paged K/V scatter; normal single-request prefill keeps its fused path.
- Each request supplies a non-empty `[current token, draft tokens...]` span. A one-token span is valid when one output token remains.
- Greedy acceptance commits the matching draft prefix plus one target token.
- Full acceptance keeps verified KV and recurrent state. Partial acceptance truncates KV, restores recurrent/convolution state, and replays only the accepted span.
- Backup, verify, commit, and rollback use the context stream. Stream overrides are rejected before mutation.
- Any error after mutation restores every canonical state component; rollback failure is executor-fatal.
- Verifier logits and sampling use `selection_vocab`, so checkpoint padding rows cannot produce token ids the frontend cannot decode.

## Verified

- Passed: Qwen3.5 release check and Clippy; RTX 5090 verifier tests 11/11. Earlier gates also passed HF golden 2/2, scheduler E2E 1/1, page-pool 4/4, and KV-pool 6/6.

| C16, prompt 1024, span 5 | Aggregate prep-kernel GPU time, 3 runs | Median | Whole-test median, 5 runs |
| --- | --- | --- | --- |
| Fused verifier kernels (`698ccbd`) | 52.256 / 52.448 / 52.351 us | 52.351 us | 7.1696 s |
| Shared Q/K prep + paged K/V scatter | 41.312 / 41.312 / 41.600 us | 41.312 us (-21.1%) | 7.1202 s (-0.7%) |

The common attention kernel stayed within 0.2%, and both profiles launched the same number of kernels. Keep the shared path: the prep-kernel reduction was consistent, while five whole-test runs did not establish a meaningful improvement. This result is specific to this RTX 5090 verifier test and does not establish serving performance.

- Claim boundary: sampling, calibrated logprob parity, and serving integration remain unverified.

## Next

- Add the DFlash drafter and its independent forward oracle.
- Before serving integration, move verifier scratch and recurrent backups to an executor-owned persistent workspace; then wire opt-in fallback/admission rules and collect same-host benchmark evidence.
