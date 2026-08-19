# K3 speculative decoding: the DSpark drafter

**TL;DR**: Speculative decoding for K3 is live end-to-end — the RadixArk
community DSpark drafter (`--dflash-draft-model-path
/mnt/shared/weights/kimi-k3-dspark`) proposes 6-token blocks, a packed
verify step commits them, and full-depth EP4 serve on 4×GB300 passes
sequential + concurrent greedy smokes with acceptance up to 2.08
tokens/round against the 224-expert pruned target. Verify is *not* bitwise
plain decode (chunkwise FlashKDA vs the fused core — near-tie argmaxes
flip); the `spec_verify` gates certify what is exact instead.

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
Acceptance 1.0–2.08 tokens/round; the drafter was trained against the
*full* K3, so the pruned 224-expert target depresses it — judge draft
quality on the full model, not this checkpoint.

Bugs the first serve found (all fixed, none reachable single-thread /
single-rank): the executor's cuBLAS bind was one-shot per executor but the
handle is `thread_local` (load on the launch thread starved the step
thread — first plain-cublas GEMM hit a null handle); the padding step
built a rows=0 aux sink; the row budget overflowed at 9 rows against
`MAX_BATCH=8`.

Memory: the draft arena is ≈ 500 MB/slot at max_ctx 4096 (the [cache_len,
35840] pending slab dominates) — spec serving wants an explicit
`PEGAINFER_K3_MAX_BATCH` well below the EP default 64.

**Next**: acceptance measurement on the full 896-expert target (EP16);
batched propose (one call per slot today); CUDA-graph the verify step
(launch geometry varies with pending lengths — needs bucketing by
lag-profile); adaptive block length via the confidence head.
