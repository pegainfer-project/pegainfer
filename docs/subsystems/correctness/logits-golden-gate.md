# Numerical correctness: the logits golden gate

**TL;DR**: How to guard that a model's logits stay correct across prompts, hardware, and batch size — *without* binding to one GPU's exact bits. The pattern: store a reference (HuggingFace bf16) of top-K logprobs for fixed teacher-forced sequences, replay them through pegainfer, and assert (a) a structural *regret* check on the argmax and (b) the **mean** and **p99** of the per-token logprob delta stay at the bf16 noise floor. NOT exact text, NOT a hash, NOT bit-identical-across-batch, NOT the absolute max. Qwen3-4B is the reference implementation (`pegainfer-qwen3/tests/hf_golden_gate.rs`, see `models/qwen3/accuracy-gate.md`); Qwen3.5-4B applies the same method with an HF `past_key_values` oracle and graph-only replay (`pegainfer-qwen35/tests/hf_golden_gate.rs`, see `models/qwen35/accuracy.md`).

Last touched: 2026-05

## The invariant we actually protect

> Whatever the prompt, the hardware, or the batch size, the engine must produce stable, dependable logits — they must not drift.

Note what this does *not* say: it does not say "bit-identical". Two facts make bit-identity a false invariant, and conflating either with a real bug is the trap every naive correctness test falls into:

1. **Cross-hardware.** pegainfer's logits come out of bf16 GEMM. Different GPUs (and different kernel/cuBLAS builds) use different tile shapes and accumulation orders, so the low mantissa bits differ by 1–2 ULP. bf16 has a 7-bit mantissa, so 1 ULP at logit magnitude ~16 is ≈0.125 nat — enough to flip an argmax on a near-tie.
2. **Cross-batch.** The batched decode path is *not* batch-invariant: batch composition changes the order in which partial results are reduced, which drifts logits ~1 ULP the same way. So "batched == sequential, bit-for-bit" is false by construction — an `executor_equivalence`-style test that asserts it is asserting a falsehood and will flake on benign noise. (We have *measured* this batch-dependent drift; we have not isolated which kernel produces it, so the doc attributes it to reduction order, not to a named library.)

The correct invariant is therefore *bounded* drift, not *zero* drift. Everything below is about bounding it strictly enough to catch real bugs while absorbing the irreducible bf16 tail.

## Why not exact text or a logprob hash

Both are **hardware-bound**: green on the machine that produced them, red everywhere else, with no signal distinguishing a real bug from one ULP. Exact-greedy text is worse — one argmax tie-flip cascades through free-running decode and surfaces as a catastrophic-looking text mismatch from a one-ULP cause. A reference of *logprobs with a tolerance* is portable and diagnostic: the delta distribution itself tells you whether a red gate is systematic drift (real bug) or a lone tail outlier (bf16 noise).

## The four design choices

**1. A reference of equal precision, stored once.** Use HuggingFace as the numerical golden truth, dumped in **bf16** — the same precision regime as pegainfer, so the comparison is apples-to-apples — on GPU with `device_map=auto` so the one script scales to large models. Store top-K logprobs per evaluated position as safetensors (machine-only numeric data, nobody reads it). fp32 is reserved for one-time tie *adjudication*, not for the gate.

**2. Teacher-forcing, not free greedy.** Feed both engines the *identical* fixed token sequence (the reference's own prompt + decode tail) and score per position. Free-running greedy lets one argmax flip cascade, making every later position incomparable. Teacher-forcing isolates each position so a disagreement is a real per-position disagreement.

**3. A structural regret check on the argmax — magnitude-independent.** pegainfer's chosen token must be one the reference also ranks near its own best. *Regret* = how far below the reference's argmax (in the reference's own logprobs) pegainfer's pick sits; it must stay ≤ a tie tolerance (Qwen3-4B: 0.20 nat). Where the reference has a clear winner, the only token within tolerance is its argmax, so this enforces exact agreement there; at a genuine bf16 tie the runner-up is within a ULP or two, so a tie-flip is not a failure. A pick the reference scores clearly worse — or one absent from its top-K entirely (confidently wrong on a token the reference does not even rank) — is a real wrong-token bug. This regret form is deliberate: a plain margin-gated equality check leaves a hole in the sub-tolerance tie band where a garbage argmax escapes every check.

**4. Mean + p99 of the logprob delta — NOT the absolute max.** On the head tokens, bound `|pegainfer − reference|` two ways:
- **mean** — trips on *systematic* drift. A uniform logit shift of `d` nat moves every delta by ~`d`, so the mean catches a small global regression that a single-token check would miss. Averaged over thousands of deltas it is hardware-stable yet sensitive.
- **p99** — bounds the tail without chasing the single worst token.

Do **not** assert the absolute max. The measured finding (see below) is that mean and p99 are *dead stable* across coverage and batch composition, but the absolute max **grows with sample count** — more positions = a fatter draw from the same irreducible bf16 tail. Gating on max is a treadmill: it red-lines on more coverage or a new GPU with no underlying bug. Print it, never assert it.

## Picking tolerances: from the measured floor, strictly

Measure the noise-floor distribution on real hardware, then set each bound a *small, recorded* multiple above its measured value — not a comfortable round number. A loose gate with comfortable headroom silently misses any real drift smaller than the headroom, which defeats the point. Qwen3-4B: measured mean ≈0.032 → `MEAN_TOL` 0.06 (≈2×); measured p99 ≈0.12 → `P99_TOL` 0.20 (≈1.6×). Record the measurement and the multiple right next to the constant so the next person knows it was calibrated, not guessed.

## Replay across the failure surfaces

Replay the same reference through every code path that can independently corrupt logits, all under the same tolerances:

- **bs=1 sequential (eager)** — the tightest comparison. Rerun it once and assert the two runs are **bit-identical**: a cheap, exact **determinism** check (a nondeterministic kernel or uninitialised scratch fails here — and this assertion *can* be exact, because nothing about the inputs changed between runs).
- **batched eager** — eager runs at the *exact* batch width (no padding), so requests of differing lengths share each kernel launch: this is the **cross-request isolation** surface, where KV mixing or a per-request indexing bug corrupts a neighbour's logits.
- **batched CUDA graph** — the captured decode path **pads** the batch up to its bucket, so this is where **padding-slot leaks** and graph pointer/buffer bugs surface. Run it straddling a bucket boundary (just past the boundary = the most pad slots) to maximise padding stress, at ≥2 ratios. Derive the straddle sizes from the real CUDA-graph buckets, not from magic numbers.

Caveat worth re-checking per model: **eager does not pad.** Confirm in your model's `batch_decode.rs` that `padded_bs = bucket_for(bs)` is gated on `enable_cuda_graph`. Padding-slot isolation is exercised *only* by the graph passes; do not claim the eager batched pass covers it.

## What "good" looks like (Qwen3-4B reference, RTX 5070 Ti, sm_120)

mean and p99 are flat across every pass; only the absolute max moves:

| Pass | positions | mean | p99 | max |
|------|-----------|------|-----|-----|
| bs=1 eager | 816 | 0.032 | 0.12 | 0.37 |
| batched eager (9, no pad) | 153 | 0.034 | 0.13 | 0.44 |
| graph (9 padded) | 153 | 0.034 | 0.13 | 0.44 |
| graph (5 padded) | 85 | 0.032 | 0.11 | 0.14 |

The single worst token is the **same** one across bs=1 / eager-9 / graph-9 — a deep-tail token at logprob ≈−10, far below the argmax: the reference is fixed at −10.2508 while pegainfer reads −9.876 at bs=1 and −9.813 in the 9-seq batch. The delta swings 0.37→0.44 purely from the batch-dependent reduction order, with **zero** effect on the argmax. eager-9 and graph-9 are bit-identical, which proves the CUDA-graph path matches eager at the same composition; the only mover is batch composition. This is exactly the benign reduction-order noise the tolerance is built to absorb, and exactly why the max is printed but not asserted.

## Applying it to a new model line

1. **Dumper** (`tools/accuracy/dump_<model>_hf_golden.py`) — seed-pinned fixed sequences spanning short→long prompts (cover multi-block KV / high RoPE positions) plus a teacher-forced decode tail; HF bf16; top-K logprobs at positions `P-1 .. P+D-1`; safetensors out.
2. **Gate** (`tests/hf_golden_gate.rs`) — load the golden, teacher-force the same sequences, apply the regret + mean + p99 guards, replay the paths that model line actually owns: bs=1, batched eager when it exists, graph-padded bucket straddles, and any model-local state handoff surface such as Qwen3.5 slot compaction.
3. **Calibrate** — measure the floor, set tolerances as a small recorded multiple, write them down.

Qwen3-4B is the reference implementation. Qwen3.5 currently has no eager batched decode path, so its instance covers sequential graph replay, bucket-straddling batched graph replay, and slot-compaction replay.
