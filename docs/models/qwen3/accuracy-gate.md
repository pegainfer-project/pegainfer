# Qwen3 accuracy gate

**TL;DR**: Qwen3's logits are guarded (per committed size — all six mapped geometries, 0.6B through 32B, keyed from config) by `tests/hf_golden_gate.rs` — a tolerance check against stored HuggingFace bf16 goldens, *not* an exact-text or hash baseline. It teacher-forces 48 fixed sequences and asserts pegainfer's logprobs stay at the bf16 noise floor of HF across bs=1 / batched eager, and through CUDA-graph when the model's GQA group is compiled (0.6B/1.7B are group-2, compiled — their graph passes run for real; 14B is group-5, uncompiled — it reroutes decode eager and skips the graph passes). Strict guards: a structural **regret** check on the argmax + **mean** delta ≤ 0.06 nat + **p99** delta ≤ 0.20 nat; the absolute max is printed but not asserted (it is coverage-unstable). This is the reference implementation of the pattern in `subsystems/correctness/logits-golden-gate.md` — read that for the *why*; this doc is the Qwen3 *specifics*.

Last touched: 2026-07

## The gate (size-keyed)

The methodology (why HF, why a tolerance not a hash, why teacher-forcing, why regret + mean + p99 and not absolute max) lives in `subsystems/correctness/logits-golden-gate.md`. Concretely (all six sizes committed, keyed from config geometry):

| Knob | Value | Where |
|------|-------|-------|
| Goldens (committed) | `qwen3-{0.6b,1.7b,4b,8b,14b,32b}-hf-golden.safetensors`; size-keyed from config `(hidden_size, num_hidden_layers)` via `COMMITTED_FIXTURE_SIZES` | dumped by `tools/accuracy/dump_qwen3_hf_golden.py` |
| Sequences | 48 seed-fixed (`SEED=0x5EED604D`), prompt 1–256 tokens, 16 decode tokens | dumper constants |
| Positions scored | 48 × (16+1) = **816** | `P-1 .. P+D-1` per sequence |
| Reference top-K | HF bf16 top-64 logprobs per position | dumper |
| Regret tolerance | `MARGIN_TOL` = 0.20 nat | gate |
| Mean / p99 bounds | `MEAN_TOL` = 0.06, `P99_TOL` = 0.20 | gate |
| Head tokens compared | top `HEAD_K` = 8 of pegainfer's own picks | gate |
| Graph-bucket straddles | `BUCKET_STRADDLES = [9, 5]` (9→bucket 16 = 7 pad; 5→bucket 8 = 3 pad) | gate, from `batch_decode.rs` buckets |

Prompt lengths reach 256 tokens (up to 16 KV blocks at block_size 16), exercising multi-block KV indexing and cross-block attention. It does **not** reach the 4096-position RoPE-cache boundary, so long-context (>4096) logits fidelity is uncovered here — qwen35's separate long golden (4097/8192) caught a real drift at that boundary (#250), so porting a long golden to the qwen3 sizes is a crate-level follow-up.

## The four replay passes

The same golden is replayed four ways, all under the same tolerances:

| Pass | What it catches |
|------|-----------------|
| bs=1 sequential (eager) | tightest comparison; **also rerun once and asserted bit-identical** → determinism (a nondeterministic kernel or uninitialised decode scratch fails here) |
| batched eager (9, no pad) | **cross-request isolation** — eager runs at the exact batch width with no padding, so differing-length requests share each launch; KV mixing or a per-request indexing bug corrupts a neighbour |
| batched CUDA graph (9→16, 5→8) | the captured decode path **pads** to its bucket, so **padding-slot leaks** + graph pointer/buffer bugs surface here; run at two real/pad ratios straddling bucket boundaries |

Eager does **not** pad — `batch_decode.rs` sets `padded_bs = bucket_for(bs)` only when CUDA graph is enabled. So padding-slot isolation is exercised solely by the graph passes; the eager batched pass guards cross-request contamination instead.

This **replaces** the old `executor_equivalence` test, which asserted batched output was *bit-identical* to sequential — a false invariant (the batched decode path is not batch-invariant; batch composition changes the reduction order and drifts logits ~1 ULP). The mean/p99 here are indistinguishable across passes, proving there is no contamination, only reduction-order noise the tolerance absorbs.

## Measured noise floor (RTX 5070 Ti, sm_120)

Verified run, all four passes green in 26s:

| Pass | positions | mean | p50 | p99 | max |
|------|-----------|------|-----|-----|-----|
| bs=1 eager | 816 | 0.0317 | 0.0242 | 0.1196 | 0.3749 |
| batched eager (9, no pad) | 153 | 0.0337 | 0.0260 | 0.1297 | 0.4374 |
| graph (9 padded) | 153 | 0.0337 | 0.0260 | 0.1297 | 0.4374 |
| graph (5 padded) | 85 | 0.0316 | 0.0253 | 0.1080 | 0.1410 |

**mean (~0.032) and p99 (~0.12) are dead stable; only the absolute max moves** — which is why max is printed, not asserted. The single worst token (seq 7 / pos 5 / token 68172) is the *same* across bs=1 / eager-9 / graph-9: a deep-tail token at logprob ≈−10, far below the argmax. HF is fixed at −10.2508; pegainfer reads −9.8759 at bs=1 and −9.8134 in the 9-seq batch — the delta swings 0.3749→0.4374 purely from batch-dependent reduction order, with zero effect on the argmax. eager-9 and graph-9 are bit-identical, so the CUDA-graph path matches eager exactly at the same composition; only batch composition moves the number. As coverage grew (108→816 positions over the redesign) the max climbed 0.26→0.44 while mean/p99 held — the absolute max is a coverage treadmill, not a drift signal.

Tolerances were calibrated from this floor, strictly: `MEAN_TOL` 0.06 ≈ 2× the measured mean; `P99_TOL` 0.20 ≈ 1.6× the measured p99. Not comfortable round numbers — a loose gate would silently miss real drift smaller than its headroom.

## Regenerating the golden

After a change that legitimately alters numerical output, recompute the golden on GPU through HuggingFace (bf16, `device_map=auto` so it scales to larger models), then re-run the gate:

```bash
uv run --no-project python tools/accuracy/dump_qwen3_hf_golden.py \
    --model-path /data/models/Qwen3-4B --out test_data/qwen3-4b-hf-golden.safetensors

PEGAINFER_TEST_MODEL_PATH=/data/models/Qwen3-4B \
    cargo test --release -p pegainfer-qwen3 --test hf_golden_gate -- --nocapture
```

## Diagnosing a red gate

The gate prints the full delta distribution and the worst position (`seq`, `pos`, `token`, both logprobs) before it fails. Read that first:

- **`mean` over `MEAN_TOL` (or `p99` over `P99_TOL`), max near the floor** → a *systematic* drift: something shifted every logit a little (a kernel change, a dtype/rounding change, a norm/RoPE regression). Real bug — bisect the change.
- **`mean`/`p99` at the floor, one lone `max` outlier** → a localised token error, or just a new bf16 tail outlier on different hardware. Adjudicate with fp32: regenerate the golden with `--dtype float32` and compare. If pegainfer tracks fp32 truth as well as HF-bf16 does, it is bf16 noise — the gate does not assert max precisely so this should not have failed; if you must widen `MEAN_TOL`/`P99_TOL`, record the measurement and multiple here.
- **regret / argmax violation** → HF had a clear winner (regret > 0.20 nat) and pegainfer disagreed, or pegainfer's pick is absent from HF's top-64 entirely. Almost always a real wrong-token bug; 0.20 nat is far above a tie.

## Coverage matrix

| Size | Geometry | GQA group | Committed golden | Validation |
|------|----------|-----------|------------------|------------|
| 0.6B | (1024, 28) | 2 (compiled) | yes | GH200 sm_90: six passes green incl. both cuda-graph passes, mean 0.0387 / p99 0.136 |
| 1.7B | (2048, 28) | 2 (compiled) | yes | GH200 sm_90: six passes green, mean 0.0332 / p99 0.123 |
| 4B | (2560, 36) | 4 (compiled) | yes | RTX 5070 Ti sm_120 floor above; default `PEGAINFER_TEST_MODEL_PATH` size |
| 8B | (4096, 36) | 4 (compiled) | yes | GH200 sm_90: six passes green, mean 0.0268–0.0300 / p99 ≤ 0.125 |
| 14B | (5120, 40) | 5 (uncompiled, reroutes eager) | yes | GH200 sm_90 (#562): mean 0.029 / p99 0.102; graph passes skipped by reroute |
| 32B | (5120, 64) | 8 (compiled) | yes | GH200 sm_90: six passes green, mean 0.0347–0.0428 / p99 ≤ 0.191 — closest of the six to `P99_TOL` 0.20 |

## Next step

Open follow-up: port qwen35's **long** golden (4097/8192) to the qwen3 sizes — qwen3 has no long-context logits coverage at any size yet, and qwen35's long fixture is what caught the #250 RoPE-boundary drift. See `subsystems/correctness/logits-golden-gate.md` for the portable pattern; `models/qwen35/accuracy.md` for the long-replay shape.
