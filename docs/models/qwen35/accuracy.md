# Qwen3.5 Accuracy

> **TL;DR:** Qwen3.5 accuracy now has short and long HF-backed logits goldens (`src/executor/hf_golden_gate.rs`, `test_data/qwen35-4b-hf-golden.safetensors`, and `test_data/qwen35-4b-hf-long-golden.safetensors`). The HF fixtures use `AutoModelForCausalLM` with `use_cache=True` / `past_key_values`, so they match pegainfer's prefill + decode shape. The long fixture crosses the old 4096-position RoPE cache boundary with 4097- and 8192-token prompts, and the #250 fix recovers full GSM8K 8-shot at `batch_size=1` to `strict-match` 79.38% / `flexible-extract` 79.30% vs the HF 79.45% baseline. The gate is size-portable: it derives a `0.8b`/`2b`/`4b`/`9b`/`27b` fixture key from the pointed model's config content; all five sizes ship committed short + long fixtures and run the same logits gate — 9B confirms the #516 untied-lm_head fix, 27B covers the #564 group-6 decode reroute, and 0.8B/2B are the first checkpoints with GDN expansion factor 1 (`linear_num_value_heads == linear_num_key_heads`), validated on GH200 sm_90. The older exact-text `test_data/Qwen3.5-4B.json` and its regeneration test are retired; `scheduler::e2e_tests` stays a scheduler liveness/integration check that now also gates model-wide collapse (free-running output must not degenerate into token loops). A broader PegaInfer-owned rand/hash corpus is deferred until the project decides how to handle cross-architecture exact-token drift.
>
> **Last touched:** 2026-09. The HF logits gate passes on RTX 5090 `sm_120` (4B/9B/27B) and on GH200 `sm_90` (0.8B/2B) and covers the qwen35-owned replay surfaces: sequential graph decode, bucket-straddling batched graph decode, slot-compaction replay after a mid-batch request drop, and a long-prompt sequential replay at 4097/8192 tokens. A full GSM8K 8-shot `lm_eval` run against `/v1/completions` also passes at HF-baseline accuracy. Current accuracy command is crate-local and needs an absolute `PEGAINFER_TEST_MODEL_PATH`: `cargo test --release -p pegainfer-qwen35 --features qwen35 --lib executor::hf_golden_gate -- --nocapture --test-threads=1`. Run `scheduler::e2e_tests` only when scheduler request-flow behavior changes.

## Goal

- External truth source: Hugging Face Transformers Qwen3.5-4B in `eval()` mode with `use_cache=True` / `past_key_values` for the small logits gate.
- Short-term success: the short and long HF logits gates stay green under the calibrated regret / mean / p99 tolerances, and GSM8K 8-shot stays within the HF baseline band for task-score regressions.
- Debugging success: any future prompt-level drift is either eliminated or explained by a recorded numeric tolerance.

## Current State

- Reusable debugging method now lives in [../../playbooks/accuracy-parity-playbook.md](../../playbooks/accuracy-parity-playbook.md).
- `pegainfer-qwen35/src/executor/hf_golden_gate.rs` checks pegainfer logits against pinned HF bf16 `past_key_values` oracles, a short + long pair per committed size (`0.8b`, `2b`, `4b`, `9b`, `27b`):
  - `test_data/qwen35-0.8b-hf-golden.safetensors` / `qwen35-0.8b-hf-long-golden.safetensors` — 0.8B short + long; first GDN expansion-factor-1 checkpoint (16/16 linear heads vs the 2× of 4B/9B/27B); GH200 sm_90 floors mean 0.027–0.030 / p99 ≤ 0.115 across all five graph passes.
  - `test_data/qwen35-2b-hf-golden.safetensors` / `qwen35-2b-hf-long-golden.safetensors` — 2B short + long, same expansion-factor-1 path; GH200 sm_90 floors mean 0.023–0.029 / p99 ≤ 0.110.
  - `test_data/qwen35-4b-hf-golden.safetensors` / `qwen35-4b-hf-long-golden.safetensors` — 4B short mixed-shape + long 4097/8192-token replay surfaces.
  - `test_data/qwen35-9b-hf-golden.safetensors` / `qwen35-9b-hf-long-golden.safetensors` — 9B (untied lm_head, #516) short + long, within the 4B tolerances.
  - `test_data/qwen35-27b-hf-golden.safetensors` / `qwen35-27b-hf-long-golden.safetensors` — 27B (group-6 full attention, decodes via the #564 reroute) short + long, within the 4B tolerances.
- `pegainfer-qwen35/tests/e2e.rs`, `pegainfer-qwen35/tests/regen_test_data.rs`, and `test_data/Qwen3.5-4B.json` are retired. They were exact-text PegaInfer self-baselines, not HF accuracy gates.
- `pegainfer-qwen35/src/scheduler/e2e_tests.rs` still loads the model and exercises sequential, repeated, concurrent, and consumer-drop scheduler paths, but it no longer reads an exact-text JSON fixture.
- A broader PegaInfer-owned rand/hash corpus was considered for issue #186, but checked-in exact token/hash data may drift across GPU architectures (`sm_80`, `sm_90`, `sm_120`). Keep that as follow-up design work until the cross-architecture stability policy is explicit.
- `docs/models/qwen35/optimization.md` records historical exact-text baseline churn. New accuracy work should use the HF logits gate before interpreting prompt-level text drift.
- The #250 GSM8K 8-shot recovery run now closes the task-score side of the old long-prompt divergence: pegainfer scored `strict-match` 79.38% and `flexible-extract` 79.30% vs the HF 79.45% baseline.
- Most of the historical op-level CPU-reference tests were retired along with
  the kernels they covered; the surviving low-level guard is
  `pegainfer-qwen35/src/recurrent.rs`: `conv1d_prefill_handoff_matches_single_prefill`.
  - `src/ops/tests.rs`: `test_argmax_tie_prefers_smallest_index_across_thread_strides`
- Historical accuracy tooling was recorded for layer `0` prefill, but these paths are not present in the current tree after the model-crate split:
  - `src/bin/qwen35_dump_layer0.rs` dumps pegainfer layer-0 checkpoints to JSON
  - `tools/accuracy/hf_dump_qwen35_layer0.py` dumps matching HF checkpoints on GPU
  - `tools/accuracy/compare_qwen35_dump.py` reports `max_abs` / `mean_abs` per checkpoint
  - `src/bin/qwen35_dump_decode_layer_ids.rs` dumps the real production-path incremental step for an explicit token-id prefix
  - `tools/accuracy/hf_dump_qwen35_decode_layer_ids.py` dumps the matching HF incremental step through `past_key_values`

## Current Accuracy Gates

### Size-portable fixture selection

- `hf_golden_gate.rs` derives the fixture key from config CONTENT, never the directory name: `text_config.hidden_size` / `num_hidden_layers` of `(1024, 24)` → `0.8b`, `(2048, 24)` → `2b`, `(2560, 32)` → `4b`, `(4096, 32)` → `9b`, `(5120, 64)` → `27b` (geometries from each size's config.json on the HF Hub). The mapping lives in `fixture_size_name` and must stay in sync with `SIZE_NAMES` in `tools/accuracy/dump_qwen35_hf_golden.py`.
- Default fixture paths are `test_data/qwen35-{size}-hf-golden.safetensors` and `test_data/qwen35-{size}-hf-long-golden.safetensors`; `PEGAINFER_QWEN35_HF_GOLDEN` / `PEGAINFER_QWEN35_HF_LONG_GOLDEN` override them.
- Failure semantics: an unreadable/malformed config or a missing committed fixture (`0.8b`/`2b`/`4b`/`9b`/`27b`) panics; a recognized size with no committed fixture (none currently — every mapped geometry now has one) skips and prints the expected path; an env override pointing at a missing file panics.
- Tolerances are shared across sizes from the 4B calibration until a new size has a green baseline to calibrate against (the `MARGIN_TOL`/`MEAN_TOL`/`P99_TOL` consts); the 0.8B, 2B, 9B, and 27B floors all sit well inside them.
- The model-wide collapse net folds into `src/scheduler/e2e_tests.rs` (Phase 2): its free-running completions fail when at least half collapse into token loops (distinct-token ratio, same-token run, or exact repeated tail period), reusing the scheduler test's model load — the size-independent net under the fixture gate.

### HF logits golden

- Fixture: `test_data/qwen35-4b-hf-golden.safetensors` (~59 KiB).
- Dumper: `tools/accuracy/dump_qwen35_hf_golden.py`.
- Oracle path: HF `AutoModelForCausalLM.eval()` in bf16, prompt prefill with `use_cache=True`, then one-token teacher-forced decode through `past_key_values`.
- Model snapshot: `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Config hash: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.
- Shape: 12 seed-fixed prompt-token sequences, prompt length 1-128, 8 teacher-forced decode tokens, 108 scored positions, top-64 HF logprobs per position.
- Tolerances: regret `0.20`, mean head-delta `0.06`, p99 head-delta `0.20`; max is printed only.
- Replay surface: sequential bs=1 through the Qwen3.5 graph decode path, batched graph passes at 5->8 and 3->4 bucket straddles, and slot-compaction replay after a mid-batch request drop. Qwen3.5 currently has no eager batched decode path.

Verified on RTX 5090 `sm_120` with Triton 3.4.0 for build-time AOT:

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
export TRITON_PYTHON=/path/to/triton34-venv/bin/python

PEGAINFER_CUDA_SM=120 \
PEGAINFER_TRITON_PYTHON=$TRITON_PYTHON \
PEGAINFER_TEST_MODEL_PATH=$MODEL_PATH \
PEGAINFER_TEST_MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
cargo test --release -p pegainfer-qwen35 --features qwen35 --lib \
  executor::hf_golden_gate::pega_logprobs_match_hf_golden_within_qwen35_tolerance \
  -- --exact --test-threads=1 --nocapture
```

Observed floor from that run:

| Pass | positions | mean | p50 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| sequential bs=1 graph | 108 | 0.0248 | 0.0175 | 0.0862 | 0.1193 |
| batched graph (5 padded) | 45 | 0.0256 | 0.0199 | 0.0757 | 0.1068 |
| batched graph (3 padded) | 27 | 0.0260 | 0.0179 | 0.1007 | 0.1206 |
| slot-compaction graph | 38 | 0.0285 | 0.0219 | 0.1031 | 0.1168 |

This table is the current PR validation snapshot. It supersedes the older A800
`sm_80` observation for this doc section; per-architecture bf16/CUDA reduction
differences should not be read as an accuracy improvement or regression by
themselves.

### Long-prompt HF logits golden

- Fixture: `test_data/qwen35-4b-hf-long-golden.safetensors` (~58 KiB).
- Dumper: `tools/accuracy/dump_qwen35_hf_golden.py` with `--prompt-lens 4097,8192 --decode-tokens 8`.
- Oracle path: same HF `AutoModelForCausalLM.eval()` bf16 `past_key_values` path as the short golden.
- Model snapshot: `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Config hash: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.
- Shape: 2 seed-fixed prompt-token sequences, prompt lengths 4097 and 8192, 8 teacher-forced decode tokens, 18 scored positions, top-64 HF logprobs per position.
- Purpose: protects the RoPE cache boundary and long prefill-to-decode logits path. This logits-level gate is paired with the full GSM8K 8-shot run below for task-score evidence.

Verified on RTX 5090 `sm_120` with CUDA 12.8 and Triton 3.4.0 for build-time AOT:

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
export TRITON_PYTHON=/path/to/triton34-venv/bin/python

PEGAINFER_CUDA_SM=120 \
PEGAINFER_TRITON_PYTHON=$TRITON_PYTHON \
PEGAINFER_TEST_MODEL_PATH=$MODEL_PATH \
PEGAINFER_TEST_MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
cargo test --release -p pegainfer-qwen35 --features qwen35 --lib \
  executor::hf_golden_gate::pega_logprobs_match_hf_long_golden_within_qwen35_tolerance \
  -- --exact --test-threads=1 --nocapture
```

Observed long-prompt floor from that run:

| Pass | positions | mean | p50 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| long sequential bs=1 graph | 18 | 0.0216 | 0.0238 | 0.0700 | 0.0747 |

### GSM8K 8-shot task-score recovery

The #250 fix was also checked through the serving path with `lm-eval==0.4.11`
and `local-completions` pointed at the full `/v1/completions` endpoint.
The run used all 1,319 GSM8K examples, no `--limit`, `batch_size=1`, and the
cached `openai/gsm8k` dataset snapshot.

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
export TRITON_PYTHON=/path/to/triton34-venv/bin/python
export LM_EVAL_BIN=/path/to/lm_eval
export RESULT_ROOT=results/qwen35-gsm8k-8shot-pegainfer-issue250

# Terminal 1: start the server.
PEGAINFER_CUDA_SM=120 \
PEGAINFER_TRITON_PYTHON=$TRITON_PYTHON \
cargo +nightly run --release -p pegainfer-server --bin pegainfer -- \
  --model-path "$MODEL_PATH" \
  --served-model-name qwen35-eval \
  --port 18082

# Terminal 2: run lm-eval after `/v1/completions` is ready.
$LM_EVAL_BIN run \
  --model local-completions \
  --model_args "model=qwen35-eval,base_url=http://127.0.0.1:18082/v1/completions,tokenizer_backend=huggingface,tokenizer=$MODEL_PATH,tokenized_requests=False" \
  --tasks gsm8k \
  --num_fewshot 8 \
  --batch_size 1 \
  --output_path "$RESULT_ROOT"
```

Result file:
`results/qwen35-gsm8k-8shot-pegainfer-issue250/qwen35-eval/results_*.json`

| Filter | exact_match | stderr | Delta vs HF 79.45% |
| --- | ---: | ---: | ---: |
| strict-match | 79.38% | 1.11% | -0.07 pp |
| flexible-extract | 79.30% | 1.12% | -0.15 pp |

This proves the issue #250 slice recovers GSM8K 8-shot for the measured
serving path. It does not make claims about MMLU, HellaSwag, ARC, long-context
admission, non-greedy sampling, or `batch_size > 1` task-score evals.

### Deferred rand/hash corpus

Issue #186 also discussed a larger PegaInfer-owned rand/hash regression corpus after the HF gate is trusted. That idea is still useful, but checked-in exact token/hash data may depend on GPU architecture and CUDA stack. Do not land it as a normal regression gate until the corpus policy says whether it is per-arch, tolerance-adjudicated through HF, or generated only as a local diagnostic.

## Progress Log

### 2026-03-27 — layer `0` short-sequence alignment

Prompt:

```text
The capital of France is
```

Commands used:

```bash
./target/release/qwen35_dump_layer0 \
  --model-path models/Qwen3.5-4B \
  --prompt 'The capital of France is' \
  --out target/accuracy/peg_layer0.json

./.venv/bin/python tools/accuracy/hf_dump_qwen35_layer0.py \
  --model-path models/Qwen3.5-4B \
  --prompt 'The capital of France is' \
  --out target/accuracy/hf_layer0.json

./.venv/bin/python tools/accuracy/compare_qwen35_dump.py \
  target/accuracy/hf_layer0.json \
  target/accuracy/peg_layer0.json
```

Observed diffs:

- exact match:
  - `embedding`
  - `input_layernorm`
  - `linear_qkv`
  - `linear_z`
  - `linear_b`
  - `linear_a`
- bf16-scale drift after the first nonlinear/kernelized step:
  - `conv1d_out`: `max_abs=0.03125`, `mean_abs≈6.13e-05`
  - `gdr_out`: `max_abs=9.77e-04`, `mean_abs≈1.14e-06`
  - `recurrent_state`: `max_abs≈1.41e-02`, `mean_abs≈8.74e-06`
  - `normed_out`: `max_abs=0.015625`, `mean_abs≈3.57e-05`
  - `attn_out`: `max_abs=0.0078125`, `mean_abs≈5.14e-05`
  - `post_attention_layernorm`: `max_abs=0.0234375`, `mean_abs≈1.19e-03`
  - `layer_out`: `max_abs=0.015625`, `mean_abs≈1.04e-04`

Interpretation:

- Weight loading, tokenizer agreement, offset RMSNorm semantics, and all four linear projections for layer `0` are already exactly aligned with HF.
- The first visible drift is `conv1d_out`, but the error remains small and does not explode through GDR or the MLP path on this short prompt.
- This is good enough to treat short-sequence layer `0` as effectively aligned for now. The next meaningful risk is the chunk boundary, not the first few tokens.

### 2026-03-27 — short-sequence coarse sweep through layer `3`

Added generic coarse dump tooling:

- `src/bin/qwen35_dump_layer.rs`
- `tools/accuracy/hf_dump_qwen35_layer.py`

Same prompt:

```text
The capital of France is
```

Observed coarse results:

- layer `1` (`linear_attention`):
  - `input_layernorm`: `max_abs=0.125`, `mean_abs≈2.62e-03`
  - `layer_out`: `max_abs=0.015625`, `mean_abs≈2.23e-04`
- layer `2` (`linear_attention`):
  - `input_layernorm`: `max_abs=0.0625`, `mean_abs≈4.08e-03`
  - `layer_out`: `max_abs=0.03125`, `mean_abs≈3.98e-04`
- layer `3` (`full_attention`):
  - `input_layernorm`: `max_abs=0.5`, `mean_abs≈6.24e-03`
  - `attn_out`: `max_abs=0.015625`, `mean_abs≈2.24e-04`
  - `layer_out`: `max_abs=0.03125`, `mean_abs≈5.44e-04`

Interpretation:

- Short-sequence drift still does not explode as activations move from the three initial linear-attention layers into the first full-attention layer.
- The largest visible gap in this sweep is at normalized activations, not at residual outputs, which is consistent with bf16 accumulation and reduction-order differences rather than an obvious semantic mismatch.
- This strengthens the case that the next real risk is long-sequence / chunk-boundary behavior, not the first handful of tokens.

### 2026-03-27 — long-sequence prefill stays aligned, first decode step diverges

Chunk-boundary prompt:

```text
hello hello hello ... hello   # 65 copies, tokenizes to exactly 65 tokens
```

Observed:

- layer `0` at `seq_len=65` still looks healthy:
  - `gdr_out`: `max_abs≈2.93e-03`, `mean_abs≈3.41e-06`
  - `layer_out`: `max_abs=0.015625`, `mean_abs≈1.80e-04`
- layer `3` at `seq_len=65` also stays bounded:
  - `input_layernorm`: `max_abs=1.25`, `mean_abs≈1.04e-02`
  - `layer_out`: `max_abs=0.046875`, `mean_abs≈7.68e-04`
- full prefill final output remains close enough that greedy selection still agrees:
  - `final_norm`: `max_abs=0.5`, `mean_abs≈2.63e-02`
  - `logits`: `max_abs=0.15625`, `mean_abs≈4.10e-02`
  - greedy argmax agrees with HF
  - top-10 logits set fully overlaps with HF

The real break happens on the first decode step after prefill:

- `prefill_next_token` matches HF: `23066`
- `decode_next_token` does **not** match HF:
  - HF: `23066`
  - pegainfer: `213603`
- `decode_logits` vs HF:
  - `max_abs=23.75`
  - `mean_abs≈4.10`

Most importantly, this is not just an HF mismatch. pegainfer decode is also inconsistent with pegainfer prefill:

- compare `decode_logits` after `65`-token prefill + one decode step
- against longer-prefill logits for the equivalent `66`-token prompt
- result:
  - argmax differs
  - `max_abs=23.6875`
  - `mean_abs≈4.13`

Interpretation:

- Prefill-side accuracy is now good enough to stop treating it as the primary suspect.
- The current high-priority bug is in the decode path or in prefill-to-decode state handoff.
- The most likely problem areas are linear-attention decode state (`conv_state`, recurrent state, or their update path), not tokenizer or static weights.

### 2026-03-27 — fixed `conv1d_prefill` state handoff, decode consistency restored

After replacing the HD256 decode attention kernel with the validated prefill path, first-decode HF mismatch improved but did not disappear. The next step was to compare incremental decode against fresh full-prefill inside pegainfer itself.

On prompt:

```text
The capital of France is
```

incremental decode and fresh full-prefill still diverged at generated token index `13`. That reproduced without HF and narrowed the issue to multi-step state accumulation.

New operator-level handoff coverage:

- `src/ops/tests.rs`: `test_conv1d_prefill_handoff_matches_single_prefill`

The test failed before the fix with:

```text
state diff 1.125
```

Root cause:

- `csrc/conv1d.cu` updated `conv_state` incorrectly when `seq_len < kernel_size - 1`.
- Repeated `seq_len=1` prefill calls only wrote the newest token into the tail slot and left earlier slots stale instead of shifting the historical state window forward.

Fix:

- `csrc/conv1d.cu` now snapshots the old per-channel state before writing and correctly rebuilds the final `(kernel_size - 1)` token window from `[old_state, x_seq]`.

Verification after the fix:

- `test_conv1d_prefill_handoff_matches_single_prefill`: passes
- `qwen35_check_incremental --prompt 'The capital of France is' --steps 35`:
  - incremental decode matches fresh full-prefill for all `35` checked steps
- HF token-level comparison for the same prompt improved:
  - first mismatch moved from generated token index `13`
  - to generated token index `23`

Interpretation:

- The remaining Qwen3.5 mismatch is no longer a decode-state handoff bug.
- The current residual gap is a smaller pure-prefill numeric drift that accumulates over layers.

### 2026-03-27 — exact-token-id HF comparison shows remaining drift is cumulative, not catastrophic

New exact-token-id tooling was added to avoid newline/whitespace round-trip ambiguity:

- `src/bin/qwen35_dump_prefill_final_ids.rs`
- `src/bin/qwen35_dump_layer_ids.rs`
- `tools/accuracy/hf_dump_qwen35_prefill_final_ids.py`
- `tools/accuracy/hf_dump_qwen35_layer_ids.py`
- `src/bin/qwen35_dump_greedy_tokens.rs`
- `tools/accuracy/hf_dump_qwen35_greedy_tokens.py`
- `src/bin/qwen35_check_incremental.rs`
- `src/bin/qwen35_generate_cases.rs`
- `tools/accuracy/hf_generate_qwen35_cases.py`

Key observations:

- Full 13-case greedy output comparison vs HF improved from `2/13` exact matches to `5/13`.
- For `python_prime`, the first HF mismatch still happens very early (generated token index `1`), but exact-token-id layer sweeps show only modest drift in early layers and gradual accumulation later:
  - layer `0` `layer_out`: `max_abs=0.0078125`, `mean_abs≈1.03e-04`
  - layer `7` `layer_out`: `max_abs=0.25`, `mean_abs≈1.14e-03`
  - layer `23` `layer_out`: `max_abs=0.25`, `mean_abs≈4.51e-03`
  - layer `31` `layer_out`: `max_abs=0.1875`, `mean_abs≈1.30e-02`
- At the exact `python_prime` divergence prefix, final logits differ by:
  - `max_abs=0.1484375`
  - `mean_abs≈2.13e-02`

Interpretation:

- The big, discrete decode-state bugs are fixed.
- What remains is smaller but still generation-relevant prefill drift, likely distributed across many layers rather than exploding at one obvious single checkpoint.

### 2026-03-27 — retired e2e baseline was stale; early remaining divergences may be tie-sensitive

Historical repo e2e status:

```bash
cargo test --release --test e2e_qwen35 -- --nocapture
```

Current result:

- fails immediately on `Hello`
- because `test_data/Qwen3.5-4B.json` was the old self-generated baseline
- a fresh candidate baseline was generated to:
  - [target/accuracy/Qwen3.5-4B.current.json]($LOCAL_PEGAINFER_DIR/target/accuracy/Qwen3.5-4B.current.json)

Important new finding while checking remaining HF mismatches:

- for `python_prime`, after the common HF prefix of one generated token, the residual difference is already small enough that top logits are tied
- HF exact-token-id prefill on that prefix has max-logit tokens:
  - `[32, 1206]`
- pegainfer exact-token-id prefill on that prefix has max-logit tokens:
  - `[727, 1206]`
- all of these tied tokens are at logit `20.0` in bf16/f32 dump

Interpretation:

- Some of the remaining prompt-level text drift is now in the “tie-sensitive” regime, where tiny residual numeric differences or argmax tie-breaking can flip the chosen token even when logits are nearly identical.
- That means the next debugging step should separate:
  - true semantic mismatches with clear logit margin
  - tie-break-only mismatches where parity may require matching HF's exact argmax behavior on equal logits

### 2026-03-27 — `argmax` tie-break fixed; HF incremental path must now be treated as the real truth

New findings and fixes:

- `csrc/sampling.cu` argmax tie-break was wrong across thread strides.
- Before the fix, equal-valued logits could select a larger token id just because its owning CUDA thread had a smaller `tid`.
- This was fixed to prefer the smallest token id on exact ties, matching standard host-side `argmax` semantics.
- New tests:
  - `test_argmax_tie_prefers_smallest_index`
  - `test_argmax_tie_prefers_smallest_index_across_thread_strides`

Impact:

- HF exact-match cases improved again, from `5/13` to `9/13`.
- Remaining mismatches after the argmax fix:
  - `hello`
  - `tell_story`
  - `python_prime`
  - `chinese_capital`

Most important new conclusion:

- HF full-prefill logits are **not** a reliable truth source for later generated tokens.
- On `Hello`, HF behaves differently depending on whether the prefix is:
  - provided directly as an input sequence
  - or reached incrementally through `past_key_values`

Concrete example:

- Prefix reached by real HF incremental generation:
  - prompt token: `Hello`
  - generated tokens so far: `,`, ` I`
- At the next step:
  - HF incremental trace picks token `2688`
  - HF full-prefill on the same explicit token-id sequence instead picks token `1044`

That means:

- for token `t > 0`, the correct comparison target is HF's real incremental decode trace, not a fresh full-prefill of the reconstructed prefix
- some earlier “HF parity” conclusions based only on full-prefill after a generated prefix were too optimistic or simply the wrong reference

Additional validation:

- `test_conv1d_prefill_seq1_matches_decode` passes, so using `seq_len=1` prefill for linear conv1d is numerically equivalent to the dedicated decode kernel for that operator
- this reduces suspicion on linear `conv1d` replacement itself; the remaining gap is more likely in broader incremental-path semantics, especially around later attention layers

### 2026-03-27 — production-path incremental dumps corrected; `Hello` step `2` no longer implicates the first full-attention layer

New tooling:

- `src/bin/qwen35_dump_decode_layer_ids.rs`
- `tools/accuracy/hf_dump_qwen35_decode_layer_ids.py`

Important correction:

- pegainfer's production decode path currently does **not** run the old per-layer decode kernels
- `Qwen35Model::decode_one_token()` now reuses `prefill_forward(&[token])`
- an earlier manual incremental dump implementation that walked `decode_full_attention_layer()` / `decode_linear_attention_layer()` directly produced large false mismatches and was corrected to mirror the real runtime path

Validated comparison on `Hello` at generated step `2`:

- prompt token ids: `[9419]`
- common generated prefix before the divergent token: `[11, 353]`

HF incremental vs HF full-prefill on the exact same prefix remains close at layer `3`:

- `layer_input`: `max_abs=0.0078125`, `mean_abs≈4.50e-04`
- `attn_out`: `max_abs=0.015625`, `mean_abs≈2.43e-04`
- `layer_out`: `max_abs=0.00390625`, `mean_abs≈5.95e-04`

peg production-path incremental vs HF incremental is also still close at layer `3`:

- `layer_input`: `max_abs=0.015625`, `mean_abs≈5.50e-04`
- `attn_out`: `max_abs=0.015625`, `mean_abs≈3.64e-04`
- `layer_out`: `max_abs=0.0078125`, `mean_abs≈7.65e-04`

But later in the stack, the same step shows the familiar cumulative drift again:

- layer `31` `layer_input`: `max_abs=0.25`, `mean_abs≈9.28e-03`
- layer `31` `attn_out`: `max_abs=0.046875`, `mean_abs≈2.93e-03`
- layer `31` `layer_out`: `max_abs=0.203125`, `mean_abs≈1.29e-02`

Crucially, this later-layer gap is not unique to pegainfer. On the same exact prefix, HF's own incremental decode also separates from HF full-prefill by a similar amount at layer `31`:

- HF incremental vs HF full-prefill, layer `31` `layer_input`: `max_abs=0.125`, `mean_abs≈8.28e-03`
- HF incremental vs HF full-prefill, layer `31` `attn_out`: `max_abs=0.015625`, `mean_abs≈2.70e-03`
- HF incremental vs HF full-prefill, layer `31` `layer_out`: `max_abs=0.125`, `mean_abs≈1.07e-02`

pegainfer shows the same qualitative pattern on the same step:

- peg incremental vs peg full-prefill, layer `31` `layer_input`: `max_abs=0.125`, `mean_abs≈8.52e-03`
- peg incremental vs peg full-prefill, layer `31` `attn_out`: `max_abs=0.015625`, `mean_abs≈2.82e-03`
- peg incremental vs peg full-prefill, layer `31` `layer_out`: `max_abs=0.125`, `mean_abs≈1.12e-02`

The remaining peg-vs-HF difference on this step is therefore smaller than the raw incremental-vs-prefill split itself. Final logits on the common prefix differ by:

- `max_abs=0.125`
- `mean_abs≈1.77e-02`

and the top candidates show why text still diverges:

- peg top logits: `1044=19.625`, `2688=19.625`, `599=19.375`
- HF top logits: `2688=19.625`, `1044=19.5`, `599=19.25`
- several competing tokens are separated by only `0.125`, so a modest numeric shift is enough to flip greedy top-1

Interpretation:

- the remaining `Hello` mismatch is **not** caused by a fresh catastrophic failure at the first full-attention layer
- HF's own incremental path also drifts away from reconstructed full-prefill by the last layer, so “HF full-prefill on the generated prefix” is doubly unsafe as a truth source
- the corrected incremental traces support the earlier “cumulative drift” thesis
- any future decode-layer dump must mirror the production path exactly; otherwise it will point at already-retired kernels and waste debugging time

### 2026-03-27 — `conv1d` pre-`SiLU` rounding bug fixed, HF exact match rises to `11/13`

New validated tooling and findings:

- `src/bin/qwen35_compare_linear_seq1_ids.rs` now reports stage-by-stage differences for a linear-attention layer between:
  - the production-like `seq_len=1` chunk-wise prefill path
  - the recurrent decode-style path
- On the exact `python_prime` divergence prefix, this comparison showed:
  - `conv1d_out` and `conv_state` were already exact
  - `gdr_out`, `attn_out`, and recurrent state were only bf16-scale apart
  - so the earlier suspicion that the `seq_len=1` linear-attention recurrence path was broken was false

The next useful split was peg vs HF on the same exact token-id prefix:

- for `python_prime` layer `0`, `layer_input`, `input_layernorm`, `linear_qkv`, `linear_z`, `linear_b`, and `linear_a` were exact
- the first non-zero checkpoint was `conv1d_out`:
  - before the fix: `max_abs=0.03125`, `mean_abs≈5.85e-05`
- `gdr_out` and recurrent state stayed much closer to HF than `conv1d_out`, which localized the first systematic peg-vs-HF drift to the conv1d activation boundary itself

Root cause:

- HF fallback executes `Conv1d` on bf16 tensors, materializes a bf16 conv result, and only then applies `SiLU`
- `csrc/conv1d.cu` previously accumulated the conv sum in fp32 and applied `SiLU` directly on that fp32 sum before the final bf16 cast
- an offline replay on the exact layer-0 inputs and weights confirmed this precisely:
  - old behavior reproduced the observed HF gap
  - rounding the conv sum to bf16 before `SiLU` reduced `conv1d_out` diff from `max_abs=0.03125` to `0.0`

Fix:

- `csrc/conv1d.cu` now rounds the conv sum to bf16 before applying `SiLU` in both:
  - `conv1d_decode_kernel`
  - `conv1d_prefill_kernel`

Verification:

- `cargo test --release test_conv1d_prefill_seq1_matches_decode -- --nocapture`: passes
- `cargo test --release test_conv1d_prefill_handoff_matches_single_prefill -- --nocapture`: passes
- `python_prime` layer `0` exact-token-id HF comparison after the fix:
  - `conv1d_out`: `max_abs=0.0`, `mean_abs=0.0`
  - `gdr_out`: `max_abs=9.77e-04`, `mean_abs≈2.21e-07`
  - `recurrent_state`: `max_abs≈6.55e-03`, `mean_abs≈2.29e-06`
  - `layer_out`: `max_abs=0.0078125`, `mean_abs≈6.90e-05`
- full 13-case greedy comparison vs HF improved again:
  - before this fix: `9/13`
  - after this fix: `11/13`

Current remaining mismatches:

- `tell_story`
  - first diff step moved to generated token index `15`
  - peg top token at the first diff: `271=21.0`
  - HF token at the same step: `198`
  - peg top-2 gap at the first diff is only `0.125`, so this case is now clearly tie-sensitive
- `chinese_capital`
  - first diff step is generated token index `18`
  - peg top logits at the first diff: `134082=19.5`, `97274=19.125`
  - HF top logits at the same prefix: `97274=19.25`, `134082=19.25`
  - the remaining delta is still small (`0.125` to `0.25`) but not yet an exact tie

Rejected lead:

- A later experiment to force additional bf16 rounding inside `rms_norm_gated_kernel` looked plausible from the HF Python source, but it made the real layer-0 HF comparison worse and was reverted immediately

## Why Start Layer-By-Layer

- End-to-end text drift appears too late. Once tokens diverge, attribution is weak.
- Qwen3.5-4B is hybrid: layers `0,1,2` are linear attention and layer `3` is the first full-attention layer.
- The recent correctness history is concentrated in linear-attention prefill, especially chunk-wise GDR, so layer `0` is the highest-signal first slice.

## Proposed Alignment Ladder

1. Freeze one deterministic prompt and one deterministic mode.
   Use prefill only first, no sampling, and prefer a short ASCII prompt whose tokenization is easy to inspect.
2. Start with `seq_len <= 64` for readability.
   Once layer `0` matches, repeat at `seq_len = 65` so the chunk boundary is exercised (`chunk_size = 64` in the current Qwen3.5 prefill scratch layout).
3. Compare coarse checkpoints first.
   - embedding output
   - post-input RMSNorm
   - layer `0` post-attention/residual
   - layer `0` post-MLP/residual
4. If layer `0` diverges, split only that layer further.
   - linear `QKV` projection
   - `Z`, `B`, `A` projections
   - `conv1d` output
   - GDR output and recurrent state
   - `rms_norm_gated` output
   - `out_proj` output
5. Once layer `0` matches, advance in this order.
   - layer `1`
   - layer `2`
   - layer `3` (first full-attention layer)
   - full prefill last hidden state
   - first decode token logits
6. Return to prompt-level text comparison only after hidden states and logits are aligned.

## HF Parity Rules

- Use the same tokenizer assets as the runtime model directory.
- Keep HF in `eval()` mode and turn off sampling.
- Compare in host-side `f32` even if runtime kernels use bf16 internally.
- Record `max_abs`, `mean_abs`, and a short slice preview at each checkpoint.
- Stop at the first divergent checkpoint instead of collecting a huge unreadable dump.

## Current Blocker / Next Action

- Current blocker: the large decode-state bugs are fixed and exact HF parity is up to `11/13`, but `tell_story` and `chinese_capital` still diverge because late incremental logits are only `0.125` to `0.25` apart near the greedy decision boundary.
- Next action: stay on exact token-id prefixes at the first HF divergence point for those two prompts, then continue checking for any remaining kernel-level HF-semantic mismatches of the same class as the fixed `conv1d` bug.
- Recommendation: keep HF incremental decode as the only truth source for generated-token steps, and avoid any debug path that does not execute the same kernels and state updates as the real runtime.
