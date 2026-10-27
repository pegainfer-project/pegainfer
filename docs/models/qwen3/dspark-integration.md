# DSpark Integration (Qwen3-4B)

**TL;DR:** DeepSeek's **DSpark** (paper + DeepSpec repo, Jun 2026) is **our DFlash drafter plus two bolt-ons**: (1) a *semi-autoregressive* **Markov head** — a rank-256 logit bias added in a short sequential loop over the block so each draft token conditions on the previously sampled one; this is the whole draft-quality win (+16–18% accepted length over DFlash in their offline table, paper Table 1), and (2) a **confidence head** (tiny linear → per-position survival probability) feeding a **hardware-aware prefix scheduler** that trims verify length under load — the production throughput/Pareto win. The released checkpoint `dspark_qwen3_4b_block7` has an **identical backbone to our `Qwen3-4B-DFlash-b16`** (same hidden 2560 / 5 layers / `target_layer_ids=[1,9,17,25,33]` / KV-injection) — it differs only by `block_size=7`, +4 tensors (`markov_w1 [V,256]`, `markov_w2 [V,256]`, `confidence_head.proj.weight [1,2816]`, `.bias [1]`), and an "anchor-is-the-first-prediction" block-layout tweak. **Our verify/accept/KV-transaction stack is reused unchanged** — DSpark only changes the *propose* step (the later env-gated hedge, `PEGAINFER_SPEC_HEDGE*`, additionally expands verify with extra chain spans over scratch KV pages; off by default). On the 5090 greedy sweep, DSpark beats the matched block7 DFlash baseline by **+3.6% geomean output tok/s** overall, with the expected wins on text/code (+3.1% to +15.8%; random is the synthetic exception) and a better accepted-draft distribution (**2.52 vs 2.30** accepted draft tokens/round, full-7 accept **17.4% vs 13.9%**). Decisions settled: **block_size = 7**, **reuse the target head** (DSpark's `embed_tokens`/`lm_head` are byte-identical to target Qwen3-4B's — skip loading them). **Phase 1 delta over DFlash is small: config + 2 tensor loads + one sequential Markov loop in `execute_dflash_draft`; no new verify/KV/graph** *(true of Phase 1 as landed — the later opt-in hedge adds expanded verify spans, scratch KV pages, and per-shape Markov graphs, all behind `PEGAINFER_SPEC_HEDGE*`)*. **Phase 2 (confidence-head static-threshold truncation) is closed without a win, not merged** — the sweep resolves the harm but not whether mild truncation pays, and the cost model agrees with the measurement; see [Phase 2](#phase-2--confidence-head--static-threshold-draft-truncation--no-win-not-merged). What transfers is the *measure-per-(model, GPU, drafter)* rule, not the verdict; the untested trim target remains **DFlash-b16** (bimodal accept ⇒ trimming can cut a mode, not a tail).

Last touched: 2026-10

## Implementation status — Phase 1 built & lossless

**Phase 1 (Markov head) is implemented and passes the losslessness gate.** Greedy DSpark output is byte-for-byte identical to plain greedy on the 5070 Ti dev box:

```
PEGAINFER_TEST_MODEL_PATH=/data/models/Qwen3-4B \
PEGAINFER_DFLASH_TEST_MODEL_PATH=/data/models/dspark_qwen3_4b_block7 \
PEGAINFER_REQUIRE_HEDGE_GATE=1 \
cargo test --release -p pegainfer-qwen3 --test dflash_speculative_gate
# → 5 passed. The fifth, hedged_ladder_passes_the_lossless_gates, re-runs the
#   lossless suites with PEGAINFER_SPEC_HEDGE on and asserts that hedged rounds
#   and chain spans actually executed, with at least one chain win and one
#   loss. It needs a DSpark checkpoint; the REQUIRE flag above turns a missing
#   one into a failure instead of a green skip, so always set it when the run
#   is meant to be evidence. See the test's own doc for what it does and does
#   not prove.
#   The "kernel-gap flip … Not a spec bug" diagnostics are the pre-existing
#   DFlash prefill/decode numeric gap, unrelated.
```

What was actually built (the new DSpark code lives in its own module `pegainfer-qwen3/src/dspark.rs`):

- **`dspark.rs`** — `MarkovHead { w1, w2 }` + `MarkovScratch` + `sample_block()` (the sequential semi-autoregressive loop, batched across requests) + `reservation_bytes()`. This is the home of all DSpark-specific logic; `dflash.rs` only holds an `Option<MarkovHead>` and an `Option<MarkovScratch>` and exposes `uses_markov_head()` / `markov_draft_tokens()` / `natural_drafts()`.
- **Custom CUDA kernel** `markov_step_argmax` (`pegainfer-kernels/csrc/shared/argmax.cu` + FFI + `ops::markov_step_argmax_into`). **We chose to add one kernel** — the earlier "no new kernel needed" prediction below was revised: the base block logits are request-major `[N·block, V]`, so step `k` needs a *strided* argmax over row `i·block+k` plus a *per-request* bias row `i`. Composing that from existing ops would mean slicing/re-batching one column per step (messier and slower than a single strided argmax-with-bias kernel). The kernel reads `base[(i·block+k)·V+v] + bias[i·V+v]` and writes the chosen token id as `u32` so it feeds straight back as the next step's prev-token lookup. Two-stage (partial+finalize), reusing the existing batched-argmax tiling.
- **Per-step math** = `embedding_batch(markov_w1, prev) → gemm(markov_w2) → markov_step_argmax`, looped `block_size` times; matches `VanillaMarkov.sample_block_tokens` exactly.
- **Kernel polish after bring-up** — the two-stage Markov argmax uses programmatic dependent launch on SM90+ so the finalize kernel can be scheduled as soon as partial tiles publish their results; SM80 falls back to the normal stream-ordered launch. PDL is not a free launch, just a launch-gap reducer for this tiny finalize. The finalize kernel also writes the full request-major sampled block on device, so DSpark does one D2H per draft block instead of one D2H per step.
- **Dual-schema config** — `DFlashConfig::from_file` now resolves both our nested `b16` schema (`dflash_config:{…}`, flat `rope_theta`) and DeepSpec's flat schema (`mask_token_id`/`target_layer_ids` flat, `rope_theta` under `rope_parameters`). `markov_rank == 0` ⇒ plain DFlash. The struct was flattened (no more `dflash_config` nesting).
- **Anchor-first span is a *checkpoint property*, not a markov one** — DeepSpec `Qwen3DSparkModel` checkpoints (both `dspark` *and* the `markov_rank==0` `dflash`) are anchor-first: position 0 is already the first real prediction, so all `block_size` positions draft (span `block_size+1`). Our native `DFlashDraftModel` (`b16`) is anchor-drop: position 0 is a throwaway slot, only `block[1..]` draft (span `block_size`). `DFlashConfig::anchor_first` is derived from `num_anchors` presence (set ⇒ DeepSpec ⇒ anchor-first); `natural_drafts()` and `drafts_start` key on it. *(This was originally — incorrectly — keyed on the markov head; see [5090 bring-up](#5090-bring-up--two-bugs-found--fixed).)*

**Next:** see [Results — 5090](#results--5090-greedy) for the measured A/B and [5090 bring-up](#5090-bring-up--two-bugs-found--fixed) for the two bugs the bring-up surfaced. The 5070 Ti is dev/compile/correctness only; perf runs on the 5090.

### Released checkpoint tensor layout (`dspark_qwen3_4b_block7`, single `model.safetensors`)

5 backbone layers (`layers.0..4`), all BF16. Top-level (non-layer) tensors:

| tensor | shape | use |
| --- | --- | --- |
| `markov_head.markov_w1.weight` | `[151936, 256]` | prev-token embedding lookup (`r=256`) — **loaded** |
| `markov_head.markov_w2.weight` | `[151936, 256]` | `Linear(256→V)` bias projection — **loaded** |
| `fc.weight` | `[2560, 12800]` | KV-injection context projection (`12800 = 2560×5` target layers) |
| `hidden_norm.weight` / `norm.weight` | `[2560]` | context RMSNorm / final RMSNorm |
| `confidence_head.proj.weight` / `.bias` | `[1, 2816]` / `[1]` | Phase 2 only — **parsed but never loaded** (logged and discarded at load; see [Phase 2](#phase-2--confidence-head--static-threshold-draft-truncation--no-win-not-merged)) |
| `embed_tokens.weight` / `lm_head.weight` | `[151936, 2560]` | byte-identical to target — **skipped**, reuse target head |

Per layer: `self_attn.{q,k,v,o}_proj` (`q 4096`, `kv 1024`, head_dim 128, 32q/8kv), `self_attn.{q,k}_norm [128]`, `mlp.{gate,up,down}_proj` (inter 9728), `input_layernorm` / `post_attention_layernorm [2560]`. Config: `block_size=7`, `markov_rank=256`, `markov_head_type="vanilla"`, `enable_confidence_head=true`, `target_layer_ids=[1,9,17,25,33]`, `mask_token_id=151669`, `rope_theta=1e6` (under `rope_parameters`).

## Sources (cloned / downloaded, local)

- Paper PDF: `../DeepSpec/DSpark_paper.pdf` — *"DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"* (Cheng et al., DeepSeek-AI / PKU, 2026). Converted to text with `uv run --with pymupdf` and read in full.
- Training/eval repo: `../DeepSpec` (`github.com/deepseek-ai/DeepSpec`). Reference modeling for the draft side:
  - `deepspec/modeling/dspark/qwen3/modeling.py` — backbone + heads (the canonical inference math).
  - `deepspec/modeling/dspark/markov_head.py` — `VanillaMarkov` / `GatedMarkovHead` / `RNNHead`.
  - `deepspec/eval/dspark/draft_ops.py` — the inference draft loop (`forward_dspark_draft_block` → `build_dspark_proposal`), the file to mirror.
  - `config/dspark/dspark_qwen3_4b.py` — the released checkpoint's training config.
- Released draft weights: `/data/models/dspark_qwen3_4b_block7` (downloaded). Target stays `/data/models/Qwen3-4B`.

DeepSpec also re-trains **DFlash** and **Eagle3** in the same framework; their DFlash config matches our checkpoint's geometry, which is why the backbones line up.

## What DSpark actually contributes (paper)

Per-token spec-decode latency is `L = (T_draft + T_verify) / τ` (τ = accepted tokens/round). DSpark attacks two of the three levers:

### 1. Semi-autoregressive generation → raises τ (draft quality)

Parallel drafters (DFlash) emit all block logits in one pass, so every position **marginalizes over all possible predecessors** instead of conditioning on the one actually sampled → "multi-modal collision" (e.g. "of problem" / "no course") → acceptance decays along the block. Their position-wise analysis (Fig 2): DFlash's *conditional* acceptance falls from ~0.87→0.78 (code) and 0.72→0.63 (chat) across the block; an autoregressive drafter stays flat/rises but starts lower (shallow net). DSpark wants both: the deep **parallel backbone** for the high-leverage first token, plus a **lightweight sequential head** for suffix coherence.

The sequential head adds a prefix-dependent **bias** to the parallel base logits and factorizes the block autoregressively (paper Eq. 4):

```
p_k(v | x0, x_<k) ∝ exp( U_k(v) + B_k(x0, x_<k, v) )
```

`U_k` = parallel backbone base logit at position k; `B_k` = the sequential bias. Two instantiations; **DSpark default = Markov head** (RNN head gives only marginal extra gain at longer blocks and is harder to deploy — paper §4.3.2):

- **Markov head** (first-order, low-rank): `B(x_{k-1}, ·) = W1[x_{k-1}] · W2`, `W1 ∈ R^{V×r}` (embedding lookup), `W2 ∈ R^{r×V}` (logit projection), `r = 256`. Cheap: per step it's a gather + one rank-256 GEMV over the vocab. Once position 1 samples "of", the head boosts "course" / suppresses "problem" at position 2.

The backbone change vs DFlash is **minor** (paper §3.1): instead of "anchor + γ masks, predict the γ mask positions", DSpark treats the **anchor itself as the first prediction position** → γ inputs (anchor + γ−1 masks) yield γ draft logits. Slightly less compute, same quality.

**Result (paper Table 1, offline, scheduler OFF, τ = accepted length incl. bonus):** on Qwen3-4B, DSpark beats DFlash by **+16.3% macro-avg** (e.g. GSM8K 5.40→6.11, MBPP 4.40→5.13, MT-Bench 3.07→3.64) and beats Eagle3 by +30.9%. A **2-layer** DSpark already beats a **5-layer** DFlash (Fig 3 — the sequential head is very parameter-efficient). Latency overhead of the sequential loop at batch 128 is **+0.2–1.3%** per round because target verify dominates (Fig 4).

### 2. Confidence-scheduled verification → cuts effective T_verify (systems)

Verifying the *whole* block wastes target batch capacity on tokens that will be rejected — the cost depends on domain (code accepts more than chat) **and** on live engine load. Two pieces:

- **Confidence head:** `c_k = σ( w·[h_k ; W1[x_{k-1}]] )` ∈ (0,1) = P(draft k survives | all earlier accepted). Trained against the analytic per-step acceptance `c*_k = 1 − ½‖p_draft − p_target‖₁` (TV distance). Post-hoc **Sequential Temperature Scaling (STS)** calibrates the cumulative product (raw head is overconfident, ECE 3–8% → ~1%).
- **Hardware-aware prefix scheduler** (paper Alg. 1): per request the prefix-survival prob is `a_{r,j} = Π_{i≤j} c_{r,i}`; globally sort all `(r,j)` by `a_{r,j}`, greedily admit tokens, and pick the verify lengths that maximize `Θ = τ · SPS(B)` where `SPS(B)` is a **once-profiled** "steps/sec vs batch-size" cost table. Early-stop on Θ drop keeps it lossless (non-anticipating). Production adaptations (§5.2): async/ZOS-compatible (decide truncation length from confidence two steps prior → "dynamic top-K"), variable-length verify kernels.

**Result (paper §5.4, live DeepSeek-V4 traffic vs MTP-1):** +60–85% tok/s/user (Flash) / +57–78% (Pro) at matched throughput, and it preserves serving capacity under strict-SLA regimes where the baseline collapses. Verify budget auto-expands to ~4–6 tokens under light load and shrinks under heavy load (Fig 8).

**The split that matters for us:** the **draft-quality win (1) is local and self-contained**; the **systems win (2) is a scheduler/engine change**. They are independently shippable, and (1) is most of the per-request speedup at the low/medium concurrency our single-card serving lives in.

**Our own attempt at (2) did not reproduce the paper's win, and the reason is instructive** — see [Phase 2](#phase-2--confidence-head--static-threshold-draft-truncation--no-win-not-merged). The paper measures against MTP-1 at DeepSeek-V4 scale (a much larger target whose verify forward dominates the step); on Qwen3-4B the fixed cost of the target forward is a far larger share of the step, so trimming rows reclaims much less than the paper's setting implies.

## How DSpark maps onto our DFlash

Our DFlash drafter (`pegainfer-qwen3/src/dflash.rs`, lane in `executor/dflash_lane.rs`) already **is** the DSpark parallel backbone. Side-by-side of the released `dspark_qwen3_4b_block7` config vs our `Qwen3-4B-DFlash-b16`:

| | DFlash-b16 (ours) | DSpark-block7 (released) |
| --- | --- | --- |
| hidden / layers / heads | 2560 / 5 / 32q-8kv | **identical** |
| `target_layer_ids` | `[1,9,17,25,33]` | **identical** |
| KV-injection context (`fc`) | `[2560, 12800]` (5×2560) | **identical** |
| `mask_token_id` | 151669 | 151669 |
| `block_size` | 16 | **7** |
| extra tensors | — | `markov_w1 [151936,256]`, `markov_w2 [151936,256]`, `confidence_head.proj.weight [1,2816]`, `.bias [1]` |
| block→draft mapping | predict block, **drop position 0**, use `block[1..]` (K=block−1) | **anchor is position 0's prediction**, use all (K=block) |

So the backbone weight loader, the `fc`+`hidden_norm` context projection, the per-layer KV-injection attention, the verify forward, `accept_greedy`, and the KV transaction are **all reusable as-is**. The draft↔verify boundary is already a pure token span (see `dflash-speculative-decoding.md` — "propose differs, verify is shared"), which is exactly the seam DSpark slots into.

### The one function that changes (Phase 1)

`execute_dflash_draft` (`executor/dflash_lane.rs:212`) today does: `draft_logits_batched` → `select_step_tokens` (independent greedy argmax per row) → split into per-request blocks. DSpark replaces the **argmax step** with the sequential Markov loop. Mirroring `markov_head.sample_block_tokens` + `build_dspark_proposal`:

```
# base_logits[N, block, V] from draft_logits_batched (unchanged)
for each request block (anchor = current_token):
    prev = anchor
    for k in 0..block_size:
        bias  = markov_w2 @ markov_w1[prev]     # gather [256] then rank-256 GEMV → [V]
        tok_k = argmax(base_logits[k] + bias)   # greedy; our engine is greedy spec
        prev  = tok_k
    drafts = [tok_0 .. tok_{block-1}]           # anchor-first: keep ALL positions
```

`base_logits` come from the **untouched** batched backbone forward. The loop is `block_size` sequential steps, each a `[N,256]·[256,V]` GEMV + per-row argmax — small compute, but it serializes drafting (see Risks). Output feeds the existing verify span builder `[current_token, draft_1, …]` and everything downstream is unchanged.

**Losslessness is free in greedy:** the Markov head only changes *which* tokens we propose; verify still checks each draft against the target's own argmax, so `accept_greedy` keeps losslessness regardless of how drafts were produced. A better proposal can only raise accept length, never corrupt output. Our existing `dflash_speculative_gate.rs` covers it unchanged.

## Integration plan (phased)

### Phase 1 — Markov head (the draft-quality win) — recommended first

1. **Config:** extend `DFlashConfig` (`config.rs`) with `markov_rank: usize`, `markov_head_type: String`, `enable_confidence_head: bool`, and `anchor_first: bool` (derived from `num_anchors`), all defaulting off/false so existing DFlash configs still parse. A `markov_rank == 0` config = plain DFlash, so this is a superset, not a fork.
2. **Weights:** load `markov_head.markov_w1.weight` ([V,256] embedding) and `markov_head.markov_w2.weight` ([V,256], the `r→V` projection — `bias = w1[prev] @ w2ᵀ`) in `dflash/loading.rs`.
3. **Propose:** add the sequential loop above. Cleanest kernel shape: a `markov_bias` op (gather rows of `markov_w1` for the current `prev` tokens of all N requests → `[N,256]`, GEMV against `markov_w2` → `[N,V]`, add into the k-th logit slice), then reuse our batched argmax for that one column. Repeat `block_size` times. The base-logit GEMM stays a single batched pass; only the bias+argmax is sequential.
4. **Block layout:** switch the draft span to anchor-first (keep all `block_size` outputs) to match how the checkpoint was trained — keyed on `anchor_first` (derived from the checkpoint's `num_anchors`, i.e. the DeepSpec format), **not** the markov head, so the native `b16` (anchor-drop) and a `markov_rank==0` DeepSpec baseline (anchor-first) both draft correctly. *(Originally keyed on markov presence — that was Bug 2; see [5090 bring-up](#5090-bring-up--two-bugs-found--fixed).)*
5. **Gate:** run `dflash_speculative_gate.rs` (losslessness) + measure accept length and single-stream/concurrent A/B vs the DFlash-b16 baseline on this box. The bar is the paper's +16% accept on Qwen3-4B (offline), translated to our greedy/serving setting — **measure before claiming the win** (CLAUDE.md Performance Work rule).

**Decided: `block_size = 7`** — the checkpoint's trained width (matches the paper). Smaller blocks than DFlash-b16 but higher per-position accept; A/B compares block7-DSpark vs block16-DFlash on accept length **and** tok/s, not accept alone.

### Phase 2 — confidence head + static-threshold draft truncation — **no win, not merged**

Truncation is *lossless* but it is not *free*, and the first attempt did not have the resolution to
answer the question at c16. In the prototype (never merged): load `confidence_head.proj.{weight,bias}`
(Linear `2816→1`, `2816 = 2560 + 256`), compute `c_k = σ(proj([h_k ; markov_w1[prev_k]]))` per
position, truncate the proposed block at the first `c_k < threshold` (mirrors
`_confident_prefix_length` in `draft_ops.py`). The draft block shrinks, the verify span shrinks
with it, and the verify still corrects — so output stays lossless. What the threshold does *not*
do is pay for itself.

**The row that settles how much the thresholds are worth is the control, not the threshold
rows.** The instrumentation-only arm (threshold 0: instrument the confidence head, truncate nothing) does
strictly *more* work than the static baseline — one extra `[2816→1]` GEMM per draft block — and
therefore cannot legitimately be faster. It measured faster anyway:

| mode | c1 | c8 | c16 |
| --- | ---: | ---: | ---: |
| static block7 | 358.0 | 1847.7 | 2709.0 |
| **threshold 0 (instrument only, no truncation)** | **358.1 (+0.03%)** | **1869.3 (+1.2%)** | **2829.1 (+4.4%)** |
| τ=0.40 | 324.9 (−9.2%) | 1856.8 (+0.5%) | 2749.7 (+1.5%) |
| τ=0.50 | 306.2 (−14.5%) | 1733.8 (−6.2%) | 2671.4 (−1.4%) |
| τ=0.60 | 293.8 (−17.9%) | 1682.6 (−8.9%) | 2494.9 (−7.9%) |
| τ=0.90 | 243.0 (−32.1%) | 1393.6 (−24.6%) | 2215.3 (−18.2%) |

(30 prompts, text+code, greedy, 128 out, RTX 5090; #717.)

**The control's movement is cell-specific: +0.03% at c1, +1.2% at c8, +4.4% at c16.** These are
single runs per cell, not a paired floor (the L20 gate study's paired floor is a different
statistic), and a control that does more work yet wins could also be a systematic effect such as
run order. What it does establish is that at c16, differences below ~4% cannot be pinned on the
threshold. The honest reading of the sweep is therefore:

- **c1: truncation is clearly harmful at every threshold.** −9.2% → −32.1%, monotone in τ, far
  outside the control's 0.03% movement. This is a real result.
- **c8/c16: truncation beyond the mildest setting is resolvably harmful.** At c8, τ=0.50/0.60/0.90
  are −6.2%/−8.9%/−24.6% against a 1.2% control; at c16, τ=0.60/0.90 are −7.9%/−18.2% against
  4.4%. The cells inside the control movement are τ=0.40 (+0.5%/+1.5%) and τ=0.50 at c16 (−1.4%).
  So what this sweep cannot answer is whether *mild* truncation pays at high concurrency — not
  whether truncation beyond it hurts.

**At τ=0.50 the c1 regression is mostly lost tokens; a graph miss is a possible extra cost.** The
91.6% token retention at that threshold alone predicts ~−8.4% at c1, where the fixed term dominates and
trimming saves almost no rows; the measured −14.5% leaves room for an additional cost. A **graph
miss** is the plausible candidate: truncating each request's span independently makes the batch's
row count differ from `batch_size × span`, so the `span_variant` lookup misses and the step runs
eager (a uniform static trim now has its own captured variant; a per-request variable one still
falls back). That extra is **inferred, not measured on this config** — the only graph-vs-eager
A/B is DFlash-b16 at ~251 tok/s (250.9 → 274.3, +9.3%), a different checkpoint and throughput,
and not every truncated round misses. There is no retention number for τ=0.40, so its split is
unknown. What *is* measured is the regime split: c1 is launch-bound, where a graph miss is
expensive and trimming buys nothing; at higher concurrency the rows start to cost something, but
at c16 the row-marginal share is still too small to pay for the token loss (§"What truncation can
and cannot buy" below).

**The cost model and the measurement agree, so this is not a harness failure.** At τ=0.5,
truncation removes 47.8% of the verified rows but keeps 91.6% of the tokens. With block7's ~9%
row-marginal share of the step, the row saving is ~4.3% of step time, while the 8.4% token loss
adds ~9.2% more rounds — a net of roughly −4%, and the sweep measured −1.4% at c16 (inside the
control's 4.4% movement). The c8/c16 cells are consistent with the model predicting no win on this
config, not with the harness hiding one. What remains open is whether a config with a larger row
share (DFlash-b16, or another GPU) can pay; that needs its own sweep, with the floor measured the
way the gate will be applied (an interleaved A/B).

#### What truncation can and cannot buy

The economics of trimming follow from where the time actually goes. From the DFlash per-batch
timings (`dflash-speculative-decoding.md`): verify at batch 1/8/16 costs 8.1 / 9.6 / 13.6 ms
for 16 / 128 / 256 rows, i.e. a **fixed term ≈ 7.9 ms plus ≈13–31 µs per row**. The fixed term is
the 36 layers' weight reads, which do not depend on the row count at all. Trimming rows only
reclaims the marginal term, so the return on trimming is bounded by the marginal term's share of
the step:

| config | rows/round | row-marginal share of step | proposal utilization |
| --- | ---: | ---: | ---: |
| DSpark block7 @ c16 | 16×8 = 128 | ~9% | 2.18/7 = 31% |
| DFlash b16 @ c16 | 16×16 = 256 | ~29% | 1.29/16 = 8% |

Halving DSpark block7's span reclaims about **half** of a ~9% share — roughly 5% of step time —
while costing the 10–15% of committed tokens that live in positions 5–7 (the confidence median
has already decayed to 0.290 there), i.e. up to +18% more rounds. Net negative with margin. **Confidence
truncation on block7 has no money in it on this 5090 roofline.** DFlash-b16 is a different shape: 92% of its verify
rows are given away, and its accept distribution is **bimodal** (structured spans accept up to
15, free text 0), so trimming can cut an entire mode rather than shave a tail. That is where a
trim line has to be tested — and it needs no confidence head at all, only a static K.

**The row marginal is a property of the GPU, not of the checkpoint.** On an L20, a *static*
K=4 trim of the block-7 **DFlash** drafter beats the untrimmed drafter by **+9.1% at c32**
(bringing it from 6.5% below plain decode to parity — paired median +1.8%, under the study's
2% bar) and **+13.8% at c64** (where it is still ~26% below plain; the gate is that
concurrency's lever), because that box is far more row-bound at high concurrency —
see [benchmarks/qwen3-4b-spec-decode-l20.md](../../benchmarks/qwen3-4b-spec-decode-l20.md). So
"block7 has no money in it" is a 5090 statement; the portable rule is the one the L20 sweep
derives: measure the crossover per (model, GPU, drafter), and arm a trim (or the load gate)
**at or above** it, never below.

**Status: closed without a win, not merged.** Nothing in `main` carries a confidence-driven
truncation path. (The separate, head-free static trim above is measured in the L20 benchmark doc.)

### Phase 3 — hardware-aware prefix scheduler (the systems/throughput win) — larger, design-first

The full Alg. 1 scheduler needs: an `SPS(B)` profiling table at engine init, batch-global survival-prob sorting each step, and **variable verify length per request**. Our verify path is **greedy, single-path, and CUDA-graph bucketed** (`verify_graph.rs`, captured per batch×span bucket — see `dflash-speculative-decoding.md`), so per-request variable span **fights the captured-shape invariant** (`total_tokens == batch_size × span`). This is a real engine change (likely: dynamic top-K truncation to a few discrete span buckets + an eager fallback), justified only at the high concurrency where verify waste bites. **Deferred** — and the deferral is now evidence-backed rather than a guess: Phase 2's cost model showed that the verify path's *fixed* cost (36 layers' weight reads) dominates the step, so per-request trimming has to beat a fixed term it cannot touch. Before building this, the prerequisites are a measured per-`(batch bucket, span)` cost table and a per-drafter spec-off baseline, so the scheduler optimizes a real cost curve rather than the paper's `SPS(B)` shape (whose key is wrong for us: the graph is captured per bucket, so the cost is a function of rows, not batch size). Note this is also where the EAGLE **proposer trait** discussion reconnects: DSpark is a second concrete proposer, so it's the forcing function for the trait that DFlash alone didn't justify.

## Risks / open questions

- **Sequential-loop latency at small batch.** The paper's +0.2–1.3% overhead is at batch 128 where target verify dominates. At our single-stream regime (where DFlash gives 1.56–1.82×) the `block_size` sequential bias+argmax launches are more exposed. The bias is a full-vocab (V=151936) materialization per step × block_size. Mitigations to evaluate: fuse gather+GEMV+add+argmax into one kernel per step; or only materialize top-k of the base logit before adding bias. **Measure draft-step time before/after**, like the DFlash batching A/B did.
- **Greedy vs sampling.** Our engine is greedy-lossless spec; the paper evaluates at temperature 1.0 with rejection sampling. Markov bias works identically under greedy (argmax of base+bias). The confidence head's TV-distance training target assumes sampling, so as a *learned acceptance ranker* it only ranks accept-likelihood — it does not make the Phase 2 truncation pay for itself (see [Phase 2](#phase-2--confidence-head--static-threshold-draft-truncation--no-win-not-merged)); STS calibration only matters for Phase 3's throughput math.
- **`tie_word_embeddings` — RESOLVED, reuse target.** Byte-compared (sha256): DSpark's `embed_tokens.weight` *and* `lm_head.weight` are **both identical** to target Qwen3-4B's `model.embed_tokens.weight` (`eabe5625…`; target is tied, no separate `lm_head`). They're just the frozen target head serialized into the checkpoint. Like plain DFlash, DSpark reuses the target's head through `compute_logits_into` and skips loading its own `embed_tokens`/`lm_head` (saves ~1.5 GiB).
- **RNN head — skip.** Marginal gain, harder to deploy (paper §4.3.2). Markov only.
- **CUDA-graph draft.** The draft-side piecewise graph (tracked in the DFlash doc) now has to also cover the sequential Markov steps; the eager-attention boundary story is unchanged but the bias loop is new capture surface.

## 5090 bring-up — two bugs found & fixed

Bringing the A/B up on the 5090 surfaced two bugs that both presented as "DSpark ≈ DFlash, suspiciously identical throughput, zero accept logs":

**Bug 1 — `vllm bench serve` defaults to non-greedy, which silently disabled spec decoding at the time.** The scheduler's all-or-nothing gate (then greedy-only; sampled verify has since lifted that requirement, #512) meant vllm-bench's default non-zero temperature fell the whole batch back to plain decode — for *both* drafters — and the draft/verify/accept path never ran (zero `cumulative_accept_rate` lines in the server log). **Fix: bench with `--temperature 0`.** Proof (local curl, identical prompt): greedy = 80 tok in 0.50 s with accept logs firing; non-greedy = 0.86 s with none.

**Bug 2 — anchor layout was keyed on the markov head instead of the checkpoint format.** The `markov_rank==0` baseline `dflash_qwen3_4b_block7` is the *same* `Qwen3DSparkModel` architecture as `dspark` (anchor-first, `num_anchors=512`), just without the markov tensors. The integration routed "no markov head ⇒ anchor-drop", dropping block position 0 — a *real* prediction in an anchor-first checkpoint — so every draft shifted by one and the baseline's accept rate collapsed to **~0.003** (spec ran as pure overhead, *slower* than plain decode). That made the DFlash baseline invalid. **Fix: derive `DFlashConfig::anchor_first` from `num_anchors` presence** (DeepSpec ⇒ anchor-first) and key `natural_drafts()`/`drafts_start` on it, decoupled from `markov_rank`. The native anchor-drop `DFlashDraftModel` (`b16`) stays healthy under the same path (accept 0.142 on random), confirming the bug was specific to feeding an anchor-first checkpoint to the anchor-drop path. *(Crash-early lesson: a layout mismatch should assert, not silently degrade to accept≈0.)*

## Results — 5090 (greedy)

`/root/.cargo/bin/vllm-bench --temperature 0 --ignore-eos`, target `/data/Qwen3-4B`, RTX 5090 GPU7, CUDA 13.1. Cells are **output tok/s (cumulative `accept_rate`)**. `accept_rate` is accepted draft tokens / verified draft tokens; block size `K=7`.

**DSpark `dspark_qwen3_4b_block7`:**

| dataset | c1 | c4 | c8 |
| --- | --- | --- | --- |
| chat (sharegpt) | 305.51 (0.290) | 1043.32 (0.344) | 1610.78 (0.353) |
| poem (sonnet) | 410.71 (0.370) | 1119.93 (0.370) | 1655.55 (0.370) |
| rand (random) | 251.29 (0.331) | 738.20 (0.351) | 1081.67 (0.357) |
| code (speed-bench coding) | 464.15 (0.364) | 1474.70 (0.373) | 2389.03 (0.376) |

**DFlash `dflash_qwen3_4b_block7`** (matched baseline: same DeepSpec anchor-first backbone, `markov_rank=0`, no Markov head):

| dataset | c1 | c4 | c8 |
| --- | --- | --- | --- |
| chat (sharegpt) | 274.20 (0.227) | 901.02 (0.310) | 1484.67 (0.316) |
| poem (sonnet) | 391.71 (0.308) | 1086.64 (0.334) | 1526.09 (0.328) |
| rand (random) | 326.11 (0.332) | 790.06 (0.332) | 1090.65 (0.323) |
| code (speed-bench coding) | 419.78 (0.349) | 1365.61 (0.347) | 2188.12 (0.338) |

**DSpark throughput delta vs DFlash:**

| dataset | c1 | c4 | c8 |
| --- | --- | --- | --- |
| chat | +11.4% | +15.8% | +8.5% |
| poem | +4.9% | +3.1% | +8.5% |
| rand | -22.9% | -6.6% | -0.8% |
| code | +10.6% | +8.0% | +9.2% |

Geomean: c1 **-0.1%** (random drags the low-concurrency aggregate), c4 **+4.8%**, c8 **+6.3%**, all valid cases **+3.6%**. The result is directionally what DSpark promises: it improves real text/code drafts by making later draft positions conditional on previous accepted tokens. The random synthetic set is useful as a stress case but is not a draft-quality benchmark; it has no semantic continuation for the Markov head to exploit, so DFlash's simpler proposer can win on c1/c4.

Accepted length distribution recovered from the server `accepted_draft` logs, grouped by benchmark time windows from `progress.log`. `accepted_draft` excludes the guaranteed bonus/correction target token, so `committed_tokens = accepted_draft + 1`.

| config | rounds | mean accepted draft | zero-accept | full-7 accept | hist accepted_draft 0..7 |
| --- | ---: | ---: | ---: | ---: | --- |
| DSpark | 19,294 | 2.52 | 29.2% | 17.4% | `[5636, 3942, 2394, 1549, 1042, 742, 639, 3350]` |
| DFlash | 21,214 | 2.30 | 32.2% | 13.9% | `[6838, 4340, 2685, 1648, 1071, 962, 731, 2939]` |

Per-case mean accepted draft tokens:

| dataset | c1 DSpark/DFlash | c4 DSpark/DFlash | c8 DSpark/DFlash |
| --- | --- | --- | --- |
| chat | 1.94 / 1.55 | 2.15 / 1.68 | 2.05 / 1.63 |
| poem | 3.23 / 2.92 | 3.26 / 2.61 | 3.07 / 2.59 |
| rand | 1.86 / 2.59 | 2.06 / 2.40 | 2.13 / 2.10 |
| code | 3.18 / 2.97 | 3.54 / 2.75 | 3.44 / 3.04 |

- **Method:** `run_sweep.sh` (session scratch, on the 5090 at `/data/dspark-bench/`) launches each server and runs `/root/.cargo/bin/vllm-bench` across chat / poem / rand / code × c1/c4/c8, parsing tok/s + `cumulative_accept_rate` from the server log. Raw artifacts were copied to `/tmp/pegainfer-bench/dspark-sweep-20260629/` locally; `accept_distribution.csv` there has the per-case histograms.
- **MTP metric note:** the Python `vllm bench serve --help=all` on this 5090 exposes a `spec_bench` dataset, but not direct MTP/spec accept metrics. The Rust `vllm-bench` result JSON also only contains standard serving metrics. For accepted length, use PegaInfer's `accepted_draft` / `committed_tokens` log lines until we add a structured metric.
- **Invalid rows:** `low_entropy`/`high_entropy` are absent from the selected SPEED-Bench split in this harness, so their JSON files are missing and those rows are dropped.
- **The accept trace** (`cumulative_accept_rate`) is `debug`-level only. It is useful for one-off DSpark/DFlash acceptance analysis, but production's default `info` logging must not emit one line per request per verify round; use a debug run or a structured metric for future acceptance studies.

## Integration delta — what's new on top of DFlash (Phase 1)

We already support DFlash, so almost everything is reused. The *only* new surface:

| Area | File | Change | Effort |
| --- | --- | --- | --- |
| **Config schema** | `config.rs` | DSpark `config.json` is **flat** (`mask_token_id` / `target_layer_ids` / `block_size` / `markov_rank` / `markov_head_type` / `enable_confidence_head` at top level), unlike our DFlash config's nested `dflash_config: {…}`. Add the flat fields (or a small adapter); `markov_rank` default 0 = legacy DFlash, so it's a superset not a fork. | trivial |
| **Weight load** | `dflash/loading.rs` | Load 2 tensors: `markov_head.markov_w1.weight [V,256]`, `markov_head.markov_w2.weight [V,256]` (~156 MiB total). **Do not** load `embed_tokens`/`lm_head` (reuse target — proven identical). | trivial |
| **Propose loop** | `dspark.rs` (`MarkovHead::sample_block`), called from `executor/dflash_lane.rs` (`execute_dflash_draft`) | `block_size`-step Markov loop, batched across requests: per step `embedding_batch(markov_w1, prev)→[N,256]`, `gemm(markov_w2)→[N,V]` bias, then `markov_step_argmax` over the step-`k` base-logit rows + bias → next prev. | the only real work |
| **Custom kernel** | `pegainfer-kernels/csrc/shared/argmax.cu` | `markov_step_argmax` — strided argmax over `base[(i·block+k)·V+v] + bias[i·V+v]`, writes `u32` token. + FFI decl + `ops::markov_step_argmax_into`. | small |
| **Block layout** | `config.rs::anchor_first` + `executor/dflash_lane.rs` + `dflash.rs::natural_drafts()` | Anchor-first (all `block_size` drafts, span `block_size+1`) vs anchor-drop (`block[1..]`, span `block_size`), keyed on `anchor_first` (derived from `num_anchors` = DeepSpec format) — **not** on the markov head, so the `markov_rank==0` DeepSpec baseline drafts correctly and the native `b16` stays anchor-drop. | small |
| **Memory reservation** | `dflash/reservation.rs` | Add `MarkovHead::reservation_bytes` (2 markov tensors ~156 MiB + sample scratch) to the fixed term. | trivial |

Reused **unchanged**: backbone forward (`draft_logits_batched`), `fc`/`hidden_norm` context projection, KV-injection attention, verify forward + `verify_graph.rs`, `accept_greedy` / `build_verify_results`, KV transaction, prefill capture, the losslessness gate and perf harness.

**One new CUDA kernel was added** (revising the original prediction — see Implementation status at top). The Markov bias is `gather + GEMM(M=N,K=256,N=V) + add + argmax`; the gather and GEMM reuse existing ops (`embedding_batch` on `markov_w1`, `gemm_into`), but the final **strided argmax-with-bias** over request-major `[N·block,V]` base logits warranted one purpose-built kernel rather than per-step column slicing.

**Difficulty: low, as predicted.** Small PR — one sequential loop + one focused kernel + config/loader plumbing, no verify/KV/graph changes. The thing still to *watch* (Risks §) is sequential-loop latency at small batch: the per-step `[N,V]` bias materialization × `block_size` is the exposed cost; measure draft-step time on the 5090 and fuse gather+gemm+argmax only if it shows.
