# K3 speculative decoding: the DSpark drafter

**TL;DR**: Speculative decoding for K3 is live end-to-end and accepts at
reference rates — the RadixArk community DSpark drafter
(`--dflash-draft-model-path /mnt/shared/weights/kimi-k3-dspark`) proposes
6-token blocks, a packed verify step commits them, and after the Markov
row off-by-one fix full-depth EP4 serve commits 3.3 tokens/round on the
cycle probe and 3.13 on a 4-prompt prose probe vs the same-checkpoint
sglang reference's 3.0 / ~2.8; the full 896-expert EP16 target commits
4.0 on code, 2.8–2.9 on English prose, 1.3 on Chinese. Verify is *not* bitwise plain decode
(chunkwise FlashKDA vs the fused core — near-tie argmaxes flip); the
`spec_verify` gates certify what is exact instead.

Last touched: 2026-08

## The checkpoint

[RadixArk/Kimi-K3-DSpark](https://huggingface.co/RadixArk/Kimi-K3-DSpark)
(community, trained with SpecForge against a live SGLang K3 target; **not**
a Moonshot artifact). Local copy `/mnt/shared/weights/kimi-k3-dspark`,
revision `3c5bac30` (2026-08-16), sha-verified. An earlier Aug-6 pull was a
*different, incompatible* revision (MLA-style draft, different taps) — if
tensors don't match this doc, check `REVISION` against HF before debugging.

Geometry (pinned by `dspark::validate_config` at load): 5 Qwen3-style GQA
layers — hidden 7168, 64 Q / 16 KV heads, head_dim 64, per-head q/k
RMSNorm, SwiGLU 14336, YaRN rope (factor 16, theta 10000, original 65536,
attention_factor baked into the tables); block 7 = anchor + 6×MASK
(163824); anchor-drop Markov head rank 256; `fc` [7168, 35840]. No
embed/lm_head of its own — it reuses the target's.

**Aux tap convention**: the context feature concatenates the target's
residual *after* 0-based layers `[7, 23, 51, 67, 83]` — the ids as spelled
in the checkpoint's `dflash.py` (`hidden_states[layer_id + 1]` over the HF
list), **no** off-by-one. This differs from GLM5.2's vLLM-trained
checkpoint, whose ids index *inputs* (there `[8, 23, ...]` means after
`[7, 22, ...]`). All five taps are MLA layers.

## What landed (`pegainfer-k3`)

- **Verify step mode** (`K3StepMode::Verify(&[K3KdaGroup])`): rows are
  packed per-slot segments — deferred-commit replay rows (the previous
  round's accepted span, re-run to advance KDA state) then the speculative
  span (anchor + drafts). KDA runs chunkwise FlashKDA per segment with
  parity as a per-segment double buffer; MLA runs absorbed paged attention
  over a packed verify block table with store-semantics latent appends
  (rejected drafts leave stale latents that the next round overwrites).
  Once a slot uses verify, all its decode goes through verify — a plain
  decode step advances every row at the global parity and would clobber
  the per-slot committed slabs.
- **Aux capture** (`K3AuxSink`): after each tap layer, the step's leading
  rows of the post-layer residual are copied into a `[rows, 35840]` slab.
  Prefill chunks capture their live tokens; a verify step captures its
  packed rows and the accepted span rows become the drafter's next pending
  context. A padding step (rows = 0) captures nothing.
- **Draft lane** (`dspark.rs` / `dspark_slot.rs`): eager 5-layer forward
  with varlen dual-source attention (`single_prefill_nhd_noncausal` over
  [cached context ; block] per slot), `dflash_qk_norm_rope` at GQA group
  4, Markov-biased greedy draft sampling. Per-slot state (draft KV,
  pending capture rows, projected context) is preallocated to
  `max_ctx + 7`; propose never allocates.
- **Scheduling**: `StepExecutor::decode_many` returns a token *list* per
  slot per round (default impl wraps plain decode); the scheduler walks
  each list through stop/length in order. With the lane armed, K3
  overrides it with `decode_spec` = propose per slot + packed verify.
  Per-request acceptance telemetry logs at release
  (`K3 slot N spec: R rounds, A drafts accepted, T tokens/round`).
- **Row budget**: one slot's verify round packs up to `2 ×
  K3_DSPARK_BLOCK` = 14 rows, so `decode_spec` splits the verify batch
  into budget-sized steps (free-running EP peers cover the extras with
  padding steps), and arming requires `max_batch ≥ 14`.
- **CLI**: `--dflash-draft-model-path <dir>` (shared frontend flag), armed
  per rank at launch. v1 is greedy-acceptance, eager-only; the confidence
  head is not loaded.

## Gates (`pegainfer-k3/tests/spec_verify.rs`)

Bitwise "verify == plain decode" is impossible by construction: verify KDA
is chunkwise FlashKDA against decode's fused core, and its projections run
at the packed bucket — the same cross-bucket noise class as chunked
prefill (~0.3 logit noise floor; at 4-layer truncation 6/64 argmax flips,
all on margins ≤ 0.3125). The gates hold what *is* exact — identical
launch geometry ⇒ identical bits:

1. determinism + history/page-scramble independence of a verify walk;
2. rejected-draft *content* cannot leak (always-rejected drafts with
   differing garbage ⇒ bit-identical trajectories);
3. packed-slot isolation (changing slot B's tokens moves nothing in A);
4. corruption-value invariance (`^1` vs `^2` at the same position ⇒
   identical committed streams);
5. margin-bounded oracle tracking (flips only against margin < 4.0, ≤ 25%);
6. DSpark round trip (real drafter weights, clamped taps at 4 layers):
   prefill capture → propose → verify → accepted-row capture round-trips
   with position invariants holding, bit-stable across release + re-prefill.

Run: `PEGAINFER_K3_TEST_224=<ckpt> PEGAINFER_K3_TEST_DSPARK=<dspark>
cargo test --release -p pegainfer-k3 --test spec_verify -- --ignored`.

## Full-depth serve validation (2026-08-19, tray04)

93 layers, EP4, `PEGAINFER_K3_MAX_BATCH=16`, real taps: sequential and
8-way concurrent (2 slots/rank, exercising the verify split) greedy smokes
complete with zero errors. Spec-vs-plain A/B on four prose prompts: 2/4
byte-identical, 2 fork on a single-token near-tie ("should be
at/about...", "rotation curves/speeds") — the certified noise class.
Acceptance 1.0–2.08 tokens/round at first serve — see the acceptance
investigation below for why, and for the fix that brought it to reference
rates.

Bugs the first serve found (all fixed, none reachable single-thread /
single-rank): the executor's cuBLAS bind was one-shot per executor but the
handle is `thread_local` (load on the launch thread starved the step
thread — first plain-cublas GEMM hit a null handle); the padding step
built a rows=0 aux sink; the row budget overflowed at 9 rows against
`MAX_BATCH=8`.

Memory: the draft arena is ≈ 500 MB/slot at max_ctx 4096 (the [cache_len,
35840] pending slab dominates) — spec serving wants an explicit
`PEGAINFER_K3_MAX_BATCH` well below the EP default 64.

## The acceptance collapse and its root cause (2026-08-19)

First serve accepted ~1.0/round where the same checkpoint under sglang
accepted 3.0 (cycle probe) — and the investigation that found the one-line
bug first *exonerated everything else*, which is worth keeping:

- **Target numerics are equivalent to sglang's.** Per-layer fp32-truth
  isolation (`~/k3-spec-smoke/l0_truth.py`, `l1_truth.py`, `ln_truth.py`,
  run in the tray09 `sgl-k3-truth` container against raw per-layer dumps
  from both engines on the same 200-token prompt): at every KDA depth
  probed (L1–L85) our single-layer error vs the fp32 reference matches
  sglang's within 0.2–0.8pp (both 1–3.8%). MegaMoE's FP8 activation path
  costs only ~0.2pp over sglang's bf16-activation MXFP4 path — not the
  story it looked like.
- **Decode is self-consistent with prefill.** Teacher-forcing the decode
  path's own greedy output back through chunked prefill (via a temporary
  per-layer dump hook, removed before merge) shows the same drift shape as
  any two correct engines: a ~0.5% bf16 seed amplified up to ~70% by L58
  and pulled back by every snapshot layer (L%12==0). Engine-vs-engine
  prefill diverges identically — the architecture amplifies noise between
  attn-res resets; it is inherent, not a bug.
- **Greedy repetition loops are the checkpoint, not the engine.** The
  224-expert pruned target degenerates into repetition on English
  prose/code prompts under sglang too, near verbatim the same loops. Judge
  acceptance against sglang on the *same prompts*, not against RadixArk's
  published numbers (3.9–5.5, full model, coherent text).

The actual bug: `markov_propose` sampled draft `k` from block row `k+1`'s
logits. Row 0 — the anchor row — is the one that predicts the token right
after the anchor (the reference `run_markov_block` in sglang
`srt/models/dspark.py` starts at row 0). Every draft was proposed one
position ahead of where verify compares it, so acceptance died at index 0
regardless of draft quality. A temporary per-round trace of
anchor/drafts/sampled (removed before merge) made it obvious in one
probe: `anchor=2000 drafts=[4000, 303, ...]` vs
`sampled=[3000, 4000, 303, ...]` — the correct continuation shifted left
by one. Fix: read rows `0..block-1` and extract from row 0.

Post-fix (tray08, EP4 full depth): cycle probe 3.3 committed/round
(sglang: 3.0), 4-prompt prose probe 3.13 (sglang on the same four
prompts: 5.47/2.62/1.07/2.02 ≈ 2.8). Output text byte-identical to
plain decode on the cycle probe.

Full 896-expert target (EP16, trays 04–07, same probe suite): every
prose/code continuation is coherent (no pruned-checkpoint repetition
loops), per-request committed/round: cycle 4.00, English prose
2.77 / 2.89, Chinese prose 1.33, Python code 4.00 (block cap is 7).
The drafter was trained against full K3, and it shows — code and
self-similar text saturate near RadixArk's published band (3.9–5.5,
measured on chat-template benches); Chinese is the drafter's weak
spot, not the verify path's.

**Next** (perf follow-up PR — semantics-neutral, kept out of #931 so
the bring-up PR stays reviewable): measure end-to-end spec-on vs
spec-off tokens/s on EP16 to size the win, then CUDA-graph the verify
step (never existed — spec v1 is eager-only; launch geometry varies
with per-slot pending lengths, needs bucketing by lag-profile) and
batch the propose pass (one drafter call per slot today). These two go
together by design: propose is 7 query rows through 5 layers —
launch-bound, so batching only pays at concurrency — and graph capture
constrains the same packed geometry, so batching it separately first
would just get redone when the graph lands.

**Not planned** (decided 2026-08-19): adaptive block length via the
confidence head. The checkpoint ships a confidence head we don't load;
using it to truncate low-confidence draft blocks would cut wasted
verify rows where acceptance is weak (e.g. Chinese at 1.33/round).
Recorded as an idea only — revisit if low-acceptance traffic shows up
in real serving profiles.
