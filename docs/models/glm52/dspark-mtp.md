# GLM5.2 DSpark speculative decoding

**TL;DR:** Replace the token-by-token decode cadence with DSpark speculative rounds, using the community `RedHatAI/GLM-5.2-speculator.dspark` checkpoint (NOT the native MTP layer — its tensors stay dropped at load). The draft is a **qwen3-architecture** 5-layer dense backbone at GLM's hidden 6144 (64-head MHA, head_dim 64, q/k-norm) + rank-256 Markov head — the same DSpark shape we already ship for Qwen3-4B (`docs/models/qwen3/dspark-integration.md`), so `dspark.rs`/the Markov kernel/the verify-accept seam port over. The verify step maps 1:1 onto the D2.5 multi-row decode bucket: **span 8 = anchor + 7 drafts = one bucket-8 step where all 8 rows are consecutive positions of ONE slot** — no new attention kernels, no graph shape changes, just relaxed row→slot mapping. Checkpoint facts verified on jz-38: draft `embed_tokens`/`lm_head` are **byte-identical to the target's** (sha256-compared) → skip loading both (~3.8 GB loaded per rank instead of 7.6). Milestones: **M1 span-steps DONE** (jz-38 gates green: 1621-token prompt byte-identical, **TTFT 35.8 s → 9.25 s = 3.9×**, bucket-8 solo step 45.6 ms), **M2 draft lane DONE in shadow mode** (measured shadow accept incl. bonus 3.44 code / 2.4–2.9 prose vs the checkpoint's 3.967 val; kernel fix: draft is head_dim 64, both qwen3 dflash attention kernels were hard-wired to 128), **M3 verify/commit rounds DONE — speculative decoding is live**: c1 measured **1.37–2.25× tok/s over plain** (solo 22.5 → 15.7 ms/token = 1.43×) with **span 4 as the default** — the ~32 ms bucket-4 verify step beats span 8's ~46 ms on every prompt class (span 8 even loses to plain on low-accept prose). Parity: mostly byte-identical vs plain, near-tie divergence across bucket shapes is the known D2 FP property; spec itself is deterministic. **M4 sampled verify DONE — the greedy-only gate is lifted**: non-greedy requests speculate via prefix-match over per-row SAMPLED tokens (lossless for a deterministic draft; every committed token is a true sample), seeded rows mix the request-local step so a seeded spec stream replays its plain stream. c1 (1.0, 0.95) measured **plain 20.7 → spec 8.7 ms/tok on code (2.38×)**; accept incl. bonus matches the shadow-probe predictions per class (code 2.55/2.67, narrative 1.72/1.72). Full rejection sampling (sampled draft + residual resampling) was probe-measured at ≤ +1.5% over this rule on code and REJECTED — see "Sampled verify". Confidence head parsed but unused (Phase 2).

Last touched: 2026-07

## Checkpoint (verified on jz-38, 2026-07-04)

`/data/models/GLM-5.2-speculator.dspark` (7.6 GB `model.safetensors`, BF16, MIT). Trained with
[vllm-project/speculators](https://github.com/vllm-project/speculators) (online, hidden states
streamed from a live vLLM GLM-5.2-FP8 server) — **not** DeepSpec; anchor/block layout must be
pinned against the speculators source, not the DeepSpec repo (see Open questions).

| fact | value |
| --- | --- |
| draft backbone | `model_type: qwen3`, 5 layers, hidden **6144** (= target hidden), 64 q-heads / 64 kv-heads (MHA), head_dim 64, q/k-norm `[64]`, inter 12288, rope_theta 8e6, rms_eps 1e-5 |
| block / drafts | `block_size=8`, deployment `num_speculative_tokens: 7` → verify span 8 (= our decode bucket 8) |
| aux target layers | `[8, 23, 39, 55, 70]` → `fc.weight [6144, 30720]` context projection + `hidden_norm` |
| Markov head | `markov_w1/w2 [154880, 256]` (~158 MB), `markov_head_type: vanilla` |
| confidence head | `proj [1, 6400]` (6144+256) + bias — Phase 2, parsed but not loaded |
| embed / lm_head | `[154880, 6144]` each, **sha256 == target's `model.embed_tokens.weight` / `lm_head.weight`** → skip, reuse target head (saves 2×1.9 GB/rank) |
| mask token | 154856 |
| quality (their val) | mean accepted length **3.967** (incl. bonus), per-position accept 0.83→0.46 across 7 positions |

Loaded per rank: 5 backbone layers ≈ 3.3 GB + fc 0.38 GB + markov 0.16 GB ≈ **3.8 GB bf16**,
replicated on all 8 ranks (draft is dense + rank-local; DP over slots, no collectives).

Draft KV: 5 layers × 2 × 4096 dims × bf16 = **80 KiB/token** → 320 MiB per 4096-token slot,
2.6 GiB/rank at 8 slots. Fits H200 alongside the FP8 expert bank.

## Why the D2.5 bucket infra is exactly the verify step

The engine is decode-only ("prompt tokens ride the decode path one position at a time" —
`scheduler.rs`), and a step already carries **per-row** `(token, position)` inputs and returns
**per-row** argmax. A DSpark verify of one request is a step whose 8 rows are
`[(anchor, p), (d1, p+1), …, (d7, p+7)]` — all on the same slot:

- **KV/causality**: per layer, the KV-write kernel covers all rows before attention launches;
  row k's `seq_len = p+k+1` includes rows `< k` and excludes rows `> k`. Same for the indexer
  k-cache. No new kernels.
- **Block table**: the partial-bucket dtod-gather path already maps row→slot arbitrarily; verify
  gathers slot s's block-table row 8 times. Only the full-bucket identity assert (a decode-mode
  invariant) is bypassed in span mode.
- **Rejection rollback is free**: rejected rows' KV entries sit above the committed length —
  `seq_lens` caps what attention reads, and the next step overwrites those positions. Paged-KV
  overwrite semantics replace qwen3's explicit KV transaction.
- **DeepEP protocol**: a verify step is a normal bucket-8 step (global rows = 8×8); ranks without
  a verifying slot carry padding rows, exactly like today.
- **Bucket ladder = span ladder**: buckets {1,2,4,8} let the scheduler verify spans {1,3,7}+anchor,
  or split a rank's 8 rows across 2 slots (span 4 each) when multiple requests share a rank. The
  D2.5 planner generalizes from "one row per slot" to "rows = Σ per-slot spans ≤ bucket".

New step surface (contained): allow repeated slots with strictly increasing per-slot positions,
per-row block-table gather for bucket 8, and an **aux-hidden capture buffer** — 5 dtod copies of
the residual stream after layers {8,23,39,55,70} into a pre-allocated `[8, 30720]` buffer inside
the whole-step graph (~480 KB/step; pointer-stable, graph-safe). The buffer and copy nodes exist
only when `--dspark-path` is present; ordinary decode graphs contain neither.

## Round cadence (M3)

Per spec round, per rank (all ranks in lock-step on the global step; draft is between steps):

1. **Verify** (global step, bucket 8): feed span, read 8 argmax rows + capture buffer.
2. **Accept** (host, coordinator): longest matching prefix + bonus/correction — port
   `pegainfer-qwen3/src/speculative.rs` verbatim (`accept_greedy`, `build_verify_results`).
3. **Draft** (rank-local command, no collectives): append accepted rows' captured hidden to the
   request's pending context → fc projection → 5-layer backbone over `[anchor, mask×7]` →
   Markov sample loop (port `dspark.rs::sample_block` + the `markov_step_argmax` kernel at
   V=154880) → reply proposed span to the coordinator.

Coordinator sees two command round-trips per round (Step, then Draft); channel latency is µs
against a ~30–70 ms step. Draft forward ≈ 1–3 ms (5 dense layers × 8 rows + 8 sequential
rank-256 GEMV+argmax micro-steps). Draft/verify overlap is a later optimization, not Phase 1.

**Projected win, updated with M2's measured accept:** spec solo ≈ (45.6 ms bucket-8 verify +
~3 ms draft) per committed tokens/round → at code-prompt accept 3.44 ≈ **14.1 ms/token
(1.6×)**; at prose accept 2.4–2.9 ≈ 17–20 ms/token (**1.1–1.3×**) — span 8 is marginal on
prose because the bucket-8 step costs 2× the bucket-1 step. **Span 4 (bucket 4) may win:**
re-scoring M2's accept histograms with drafts capped at 3 gives ≈2.8 committed/round on the
code prompt, and if a bucket-4 step lands near ~30 ms that is ≈12 ms/token (1.9×). The M3 A/B
(span 8 vs span 4, and measuring the bucket-4 step time) decides; don't assume.

## Milestones

- **M1 — span steps for prompt ingestion** (no draft model) — **DONE, all jz-38 gates green
  (2026-07-04, `62defc6` + fairness fix)**. Measured (1621-token prompt, ×3 stable):
  **TTFT 35.2–36.6 s → 9.25 s (3.9×)**, 1621 → 203 steps; solo decode 22.5–22.6 ms/step
  (main 22.4–22.5, no regression); **bucket-8 one-real-slot step = 45.6 ms** — the number
  M3's round math needed. Correctness: 1621-token prompt output **byte-identical** span-path
  vs main's token-by-token walk (+ determinism ×2); short-prompt outputs identical to main;
  9-way mutual + ==solo; queue-80 / drain / mixed long-prefill-during-8-way-decode (full
  lengths; text diverges from solo = the known cross-shape FP association property) / SIGTERM
  all PASS. Toxic review: approve, no blockers; fairness finding fixed (leftover rows now
  round-robin across co-resident prefills — ascending-slot greed starved the later prefill
  to 1 row/step). Two notes left open: re-check the now-always-on full-bucket block-table
  gather (8×256 B dtod/step) at the next c64 bench; the repetitive gate prompt makes GLM
  emit EOS as its first token, and the frontend then reports `completion_tokens=0` with the
  prompt echoed in `text` — pre-existing (main identical), frontend accounting, not engine.
  TTFT is 3.9× rather than 8× because a bucket-8 step costs 45.6 ms vs 22.4 — which also
  says a span-4 verify (bucket 4) is worth measuring against span-8 in M3.
- **M2 — draft lane** (rank-local), implemented as **shadow mode** — **DONE, jz-38 gate green
  (2026-07-04, `f03cb29`)**. `--dflash-draft-model-path` loads the drafter on every rank; the
  decode path is UNCHANGED, and per spec round the coordinator (a) appends every live slot's
  step rows from the in-graph capture buffer to that slot's draft pending context, (b) proposes
  7 drafts at the current anchor once the previous round is fully scored, and (c) scores
  proposals against the tokens plain decode actually emits (`GLM5.2 dspark shadow:` log lines).
  **Measured shadow accept (incl. bonus = committed tokens/round): code-gen prompt 3.44,
  technical prose 2.89, narrative prose 2.40, highly-predictable counting 4.74** (43–82 rounds
  each; checkpoint's own val: 3.967 on their eval set) — the right ballpark everywhere, and the
  zero-accept fraction on the code prompt (10/59 = 17%) matches their published first-position
  accept 0.83. A capture-layout bug would have collapsed this to ≈1.x. Output parity: all 4
  prompts + solo byte-identical with the drafter on vs off (shadow perturbs nothing; the
  in-graph capture copies change no results). Pieces: `dspark.rs` (loader with config
  crash-early pin, batched anchor-drop forward reusing target embed/lm_head, Markov loop at
  positions 1..=7), 5 in-graph capture copies at `GLM52_DSPARK_AUX_LAYERS = {7,22,38,54,69}`
  (see pin #5), rank-local `Draft` command (resets → appends → proposals, FIFO-ordered before
  the next step), draft KV sized `max_model_len + 8`. The anchor-position invariant
  (`anchor_pos == committed + pending`) is asserted in the draft — scheduler/draft drift
  crashes instead of proposing from the wrong rope phase. Cost: solo decode 22.50–22.56
  ms/step with the in-graph capture copies (M1 baseline 22.4–22.6 — free), 23.32–23.44 with
  the shadow drafter proposing every round ≈ **2.2 ms per draft round** — inside the 1–3 ms
  the round-cadence math assumed. Bring-up bug worth remembering: both
  draft attention kernels were hard-wired to qwen3's head_dim 128 — the dspark draft is 64
  MHA heads × head_dim 64, needing the dflash qk-norm-rope warp-count fix + a FlashInfer
  HEAD_DIM=64 prefill instantiation (the crash-early launcher checks caught it on the first
  draft round).
- **M3 — round loop — DONE, jz-38 gate green (2026-07-04, `6f83e30` + span-4 default)**. A
  decoding slot's span IS the verify span `[anchor, d1..dk]`; `advance_span` runs
  `accept_greedy` over the span's per-row argmax and commits the accepted prefix + the
  correction/bonus; the coordinator emits multi-token commits, appends only committed rows'
  hidden, re-proposes from the new anchor. Rollback is free (seq_lens caps reads; the next
  span overwrites). The plain path is the zero-draft special case — drafter off = M1 behavior
  unchanged. **Measured (c1, greedy, 200-token completions + 512-token solo ×3):**

  | | plain | spec span-8 | spec span-4 |
  | --- | --- | --- | --- |
  | solo ms/token | 22.5 | 19.4 (1.16×) | **15.7 (1.43×)** |
  | code tok/s | 41.3 | 62.5 | **70.1 (1.70×)** |
  | tech prose tok/s | 43.2 | 36.5 — SLOWER than plain | **68.6 (1.59×)** |
  | narrative tok/s | 43.6 | 33.6 — SLOWER | **59.7 (1.37×)** |
  | counting tok/s | 43.4 | 95.1 | **97.7 (2.25×)** |
  | **sharegpt ×30 aggregate** | 40.2 | — | **61.1 (1.52×)** |

  The closing A/B per the qwen3 methodology (30 sampled ShareGPT first-turn prompts at
  `/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json` on jz-38, c1 sequential,
  temperature 0 + ignore_eos, 200 tokens each): **40.2 → 61.1 tok/s aggregate (1.52×), p50
  per-request 42.7 → 66.3 (1.55×)**; accept incl. bonus 2.489 over 2395 rounds (hist
  [695, 595, 343, 762] — 32% of rounds accept all 3 fed drafts); 18/30 outputs byte-identical
  to plain, 12 near-tie divergent.

  Per-round wall matches the step-time model exactly: span-8 ≈ 46.6 ms/round (M1's 45.6 ms
  bucket-8 step + draft), span-4 ≈ 33.9 ms/round → **bucket-4 step ≈ 32 ms**. Span 4 wins on
  every prompt class (even high-accept counting: 46.6/4.7 > 33.9/3.6 ms/token), so
  `GLM52_DSPARK_SPAN_DRAFTS = 3` is the default; span 8 is strictly dominated and loses to
  plain decode outright on low-accept prose. Parity vs plain: 3/5 outputs byte-identical; the
  other two diverge at a near-tie hundreds of tokens in (bucket-8/4 vs bucket-1 FP association
  order — the known D2 property); spec is deterministic (same prompt twice → identical).
  Truncating the proposal (not the block) costs nothing: the drafter still proposes 7, the
  span feeds 3.

## Sampled verify — non-greedy speculation (M4, 2026-07-06)

**Mechanism.** The verify span's committed rows are SAMPLED instead of argmaxed (the #586
post-graph FlashInfer pass, now over every span row), and `accept_greedy` prefix-matches the
sampled tokens against the drafts. For a deterministic (greedy) draft this rule is lossless
speculative sampling: row `k`'s token is a true sample from the target distribution given the
accepted prefix, and it commits whether or not it matches the draft — the match only decides
how many tokens ride one step. **Seed contract:** row `k` samples at request-local step
`completion + k` with `mix_seed(request_seed, step)` — the exact step a plain decode would
sample that token at, so a seeded request's speculative stream replays its plain stream
token-for-token (up to the D2 cross-bucket FP property, see Parity below).

**Why not full rejection sampling (min(1, p/q) + residual resampling)?** Probe-measured
before building (branch `probe/glm52-ab-accept`, raw-logits dump + offline eval, 318
rounds/class at the customer-dominant (1.0, 0.95)):

| span-4 accept incl. bonus | A: sampled verify | B: rejection sampling |
| --- | --- | --- |
| code | 2.67 | 2.71 |
| tech prose | 2.36 | 2.50 |
| narrative | 1.72 | 1.91 |
| counting | 3.81 | 3.76 |

B's theoretical headroom does not materialize: the greedy draft is a *confidence
concentrator* — the Markov head's full distribution q is noisier than its argmax, so
`Σ min(p,q)` is dragged down by q's spread while `p(argmax q)` is not. B buys ≤ +1.5%
throughput on code at the cost of draft sampling, q materialization, a residual-resampling
kernel, and statistical-only gates (no byte parity). Temperature sensitivity is mild:
(0.1, 1.0) accepts only 4–15% above (1.0, 0.95) — the draft is right exactly where the
target is peaked.

**Measured (jz-38 2026-07-06, c1, temp 1.0 / top_p 0.95, ignore_eos, span-4):**

| | plain sampled | spec sampled |
| --- | --- | --- |
| solo code ms/token (512 tok) | 20.72 | **8.71 (2.38×)** |
| code / tech prose / narrative / counting ms/token (320 tok) | 20.7 | 10.8 / 13.2 / 15.6 / 7.6 |
| accept incl. bonus (measured vs probe prediction) | — | 2.55/2.67 · 2.06/2.36 · **1.72/1.72** · 3.67/3.81 |

Plain sampled solo is 20.7 (not greedy's 19.5): a non-greedy request forfeits launch-ahead —
which the drafter disables anyway, so speculation loses nothing.

**Under real concurrency (vllm-bench Rust client, sharegpt first turns, output 200,
temp 1.0 / top_p 0.95, same seed both arms, jz-38 2026-07-06):**

| c | tok/s off → on | TPOT p50 off → on (ms) | TPOT p99 off → on | TTFT p50 off → on |
| --- | --- | --- | --- | --- |
| 1 | 40.6 → 57.7 (1.42×) | 20.4 → 12.5 | 20.6 → 18.7 | 321 → 320 |
| 2 | 81.7 → 98.7 (1.21×) | 21.9 → 15.2 | 25.4 → 20.0 | 121 → 385 |
| 4 | 130.6 → 163.9 (1.25×) | 25.7 → 18.2 | 28.0 → 25.2 | 258 → 435 |
| 8 | 225.3 → 281.3 (1.25×) | 29.6 → 19.6 | 34.3 → 30.3 | 230 → 511 |

The win holds through c8 (≤ 1 request/rank — verify spans don't yet contend for slots):
throughput +21–25%, TPOT p50 1.4–1.6×. Two honest costs visible in the tails: **TPOT p99
improves less than p50** (round-granular commits jitter the ITL — a rejected round pays a
bucket-4 step for one token), and **the spec arm's TTFT p50 is ~150–280 ms worse under
load** (the drafter forces prefix caching off, and verify spans lift the global bucket
while co-resident prompts prefill). c > 8 (multi-slot contention) remains unmeasured.
Note the Python `vllm bench serve` c8 client hang (#548) did NOT reproduce with the Rust
`vllm-bench` client — 8/8 runs completed 100%.

**Parity (gates 2026-07-06).** Seeded determinism: same seeded request twice under spec →
byte-identical. Seeded spec-vs-plain: all 4 prompts eventually diverge at a near-tie — but
P2 tracked plain for **35 consecutive sampled tokens** before diverging, which is ~1e-5
probability under a broken seed stream: the divergences are the D2 cross-bucket FP property,
AMPLIFIED by sampling (an inverse-CDF draw flips whenever the uniform lands within the
bucket-4-vs-bucket-1 logits wiggle of a boundary — greedy only flips at exact argmax ties).
Treat sampled spec-vs-plain parity as a statistical gate; byte gates are determinism (above)
and greedy neutrality (drafter-on greedy on this diff byte-matches main).

**Frontend gap:** `wire.rs` still strips `seed` (the shared frontend serves models whose
schedulers don't feed request-local steps — qwen3's legacy path, #284); the parity gates used
a gate-only passthrough patch (`seed: params.seed.map(|s| s as u64)`), reverted after. Seeded
requests over HTTP land with the sampling-parity issue, not this milestone. Note the seeded
verify span samples one row per FlashInfer call (the replayable-stream contract pins the
philox subsequence to 0, so rows can't batch) — 4 sequential passes per round, cost
unmeasured; the 2.38× above is the unseeded batched path.

## Layout pinned against speculators source (2026-07-04)

Read from `vllm-project/speculators` `src/speculators/models/{dflash,dspark}` (main). The four
bring-up-critical semantics, so M2 doesn't rediscover qwen3's Bug 2:

1. **Anchor-drop, 7 drafts, span 8.** `_build_base_config_kwargs` sets
   `GreedyTokenProposalConfig(speculative_tokens=block_size - 1)` with the comment "First block
   position is the anchor, not emitted during gen", and training zeroes the loss at block
   position 0 (`aligned_loss_mask[:, ::block_size] = 0`). Block input = `[anchor, mask×7]`;
   position k (1..7) predicts the token at `anchor_pos + k`. Our `drafts_start = 1` /
   `verify_span = block_size = 8` — the qwen3 *anchor-drop* path, NOT the DeepSpec anchor-first
   one. Crash-early if a config ever arrives with DeepSpec's `num_anchors` marker.
2. **Intra-block attention is bidirectional; context is strictly `< anchor`.**
   `create_anchor_block_mask_mod`: full-attention layers use `non_causal=True` within the block
   (matches qwen3's `single_prefill_nhd_noncausal_into`), and base-context visibility is
   `kv_base_pos < q_anchor` — the anchor's own captured hidden row is NOT attended. Consistent
   with the inference bookkeeping: the next round's anchor is the bonus token, whose hidden is
   only captured when it is fed in the next verify, so pending context always ends one row
   before the anchor.
3. **Block position ids = `anchor_pos + k`** (`get_base_indices_for_anchored_blocks`), rope
   from the flat `rope_parameters.rope_theta = 8e6`. Same committed+context position walk as
   qwen3; the draft never sees positions past the launch-time `max_model_len` cap.
4. **Markov loop: prev(k) = block token k−1, applied at positions 1..7.** Training builds
   `prev_token_ids = [b0, b0..b6]` and biases the whole block, but position 0 is dropped, so
   inference is 7 sequential steps: `prev = anchor → bias → argmax(position 1) = d1 → prev = d1
   → …`. Vanilla head: `bias = markov_w2(markov_w1[prev])`, exactly our qwen3
   `sample_block`/`markov_step_argmax` math — port with the loop starting at position 1
   (anchor-drop) instead of 0 (anchor-first).

5. **Aux capture ids are vLLM "before layer idx" semantics — off by one vs qwen3's
   convention.** Pinned from vLLM `deepseek_v2.py` (the model family GLM5.2 serves as) +
   `launch_vllm.py`: the extractor captures `hidden_states + residual` at the TOP of the layer
   loop, before running layer `idx`, and `--include-last-layer` legally appends
   `id = num_hidden_layers` (handled after the loop) — so id k = the residual stream after k
   layers, i.e. after layer k−1. The checkpoint's `[8, 23, 39, 55, 70]` therefore map to OUR
   post-layer capture points `{7, 22, 38, 54, 69}` (`GLM52_DSPARK_AUX_LAYERS`). The qwen3
   DFlash checkpoint (DeepSpec-trained) uses the opposite "after layer id" convention —
   copying qwen3's capture wiring verbatim would have been the exact Bug-2 repeat this
   section exists to prevent.

## Decisions

- **dspark over native MTP** (user call): the native MTP-layer tensors stay dropped at
  `build_model`. DSpark gives a trained, measured (3.97 accepted) drafter without bringing up
  MTP-layer training parity.
- **Prefix-match accept for BOTH greedy and sampled verify** (M4 supersedes the Phase-1
  greedy-only call): `accept_greedy` is the single seam — argmax tokens for greedy requests,
  per-row sampled tokens for non-greedy. Full rejection sampling measured out (≤ +1.5% on
  code, see "Sampled verify"); Markov head is required (it *is* the dspark draft quality);
  confidence head deferred.
- **Reuse target embed + lm_head**: proven byte-identical; draft logits = target lm_head GEMM
  over draft final hidden (bf16 dense GEMM `[8, 6144] × [6144, 154880]`, rank-local).
- **Verify span 8 default**: matches checkpoint training width AND the bucket ladder's top.
  Span is a scheduler knob, not a constant — smaller spans are the load/latency fallback.

## Next action

D3 measurement is closed (sharegpt c1 = 1.52× greedy; sampled c1 code = 2.38× post-D5).
Later levers, measured not assumed: adaptive span (feed more drafts only when recent accept
is high — 32% of sharegpt rounds max out the 3-draft span), draft/verify overlap (#582),
multi-slot spec behavior at c>8 (verify spans contend for bucket rows — the planner already
splits, but the win under contention is unmeasured), seed passthrough in the shared frontend
(the sampling-parity issue / #284 — unblocks seeded requests over HTTP), and the M2 review's
S1 (templating the hd64/hd128 prefill copy-paste).
