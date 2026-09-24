# Serve Qwen3.8-27B on the Qwen3.5 line

> **TL;DR:** Qwen3.8-27B is not a new model line: its text tower is
> shape-identical to Qwen3.5-27B's and its `config.json` still reports
> `model_type: qwen3_5`, so the probe, the weights, the kernels, the scheduler
> and the server wiring all pass unchanged. Two things actually blocked it —
> Qwen3.8 stores the gated-DeltaNet scalars as bf16 where the loader demanded
> f32 (the f32 loader now widens 1D bf16; the conversion is exact), and the
> logits-golden fixture key could not tell two `(5120, 64)` checkpoints apart
> (the gate now picks the fixture whose recorded `config_sha256` matches the
> checkpoint). `output_gate_type: "swish"` in its config is **not** a numerics
> change and must not be implemented: the attention output gate is sigmoid,
> measured. Qwen3.8-27B clears this line's TP2 logits gate (mean 0.0238 / p99
> 0.0862 short, 0.0226 / 0.0860 long) and the chat-template parity gate.
> Tracked in #1067.
>
> **Last touched:** 2026-09

## What the checkpoint actually is

Measured against `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, diffed field-by-field and
tensor-by-tensor against `Qwen/Qwen3.5-27B`:

| Fact | Value | Consequence |
| --- | --- | --- |
| identity | `model_type: qwen3_5`, `architectures: [Qwen3_5ForConditionalGeneration]`, `text_config.model_type: qwen3_5_text` | probe (`config/mod.rs` `probe_config_json`) accepts it as-is |
| text geometry | 64 layers (16 full + 48 linear), hidden 5120, head_dim 256, 24 Q / 4 KV heads, 48 V / 16 QK linear heads @128, conv kernel 4, intermediate 17408, vocab 248,320, ctx 262,144, `rope_parameters` identical | no config, kernel or buffer change |
| tensors | 1,199 tensors, 55.56 GB, key set identical; the 851 names the loader asks for are all present | no converter, no name map |
| dtype | 100% BF16 (3.5: 1,103 BF16 + 96 F32) | **the only load blocker** |
| shapes | 0 differing tensors in shape; 96 differ in dtype only | see below |
| chat template | adds `reasoning_effort`, `preserve_thinking` | the one real behaviour delta |
| unused weights | 333 `model.visual.*` (never loaded, same in 3.5) + 15 `mtp.*` | MTP is a separate opportunity |

## The gate is sigmoid. Do not implement "swish".

Qwen3.8's `text_config` adds `output_gate_type: "swish"`, and PegaInfer's
full-attention output gate is hardcoded sigmoid
(`pegainfer-kernels/csrc/qwen35/prefill_attention_hd256.cu:190`). That looked
like a numerics break. It is not, on three independent grounds:

1. **No reader.** `output_gate_type` appears zero times in transformers'
   `qwen3_5` config *and* modelling code — checked at 5.2.0, **5.8.0** (the
   release matching the checkpoint's declared `5.8.0.dev0`) and 5.17.0. The
   only consumer in the whole wheel is `qwen4_exp`, a different architecture.
2. **Reading it would change nothing.** swish ≡ silu, and the gated RMSNorm
   silu is exactly what PegaInfer already applies
   (`pegainfer-kernels/csrc/shared/norm.cu:58`), matching HF's
   `Qwen3_5RMSNormGated`.
3. **The weights say so.** Swap `torch.sigmoid` for `x·sigmoid(x)` at the
   module's single call site and teacher-force both hypotheses: Qwen3.8-27B
   NLL **1.82** (sigmoid, coherent prose) vs 14.27 (silu, token soup); a
   Qwen3.5-0.8B control reads 4.28 vs 14.18, proving the instrument. The
   numbers are the ones this support must not regress.

`TextConfig` has no `#[serde(deny_unknown_fields)]`
(`pegainfer-qwen35/src/config/model.rs`), so an unlisted field is dropped
silently — a checkpoint can load, boot and serve garbage. That policy decision
is #1068, not this PR.

## BF16-stored GDN scalars: widen, don't relax

The 16,896-byte `total_size` delta between the generations is exactly
`48 layers × (A_log 48 + norm.weight 128) × 2 bytes`: those two vectors per
linear layer are f32 in Qwen3.5 and bf16 in Qwen3.8 (a transformers 5.x
save-time cast policy change, not an architecture change). PegaInfer demanded
f32 at both loads, so Qwen3.8 failed loudly with `expected dtype F32, got
BF16` — the behaviour we want from a fail-closed loader.

The fix: `tensor_f32_cow` (`pegainfer-core/src/weight_loader.rs`) accepts 1D
bf16 and widens it — bf16 → f32 is exact, and HF itself upcasts at the point
of use (`g = -self.A_log.float().exp() * …`), so both storages reach the
kernels as the same values. Other dtypes and ranks stay rejected, pinned by
`tensor_f32_cow_rejects_wrong_dtype_and_rank`.

The same audit closed a second silent-trust hole: the unsharded load path
took each tensor's own shape as truth, so a checkpoint whose tensors disagreed
with its `config.json` reached a config-sized GEMM. `load_tensor_2d` now takes
the config-derived `(rows, cols)` and checks the header before upload; the
27B-class extents are pinned by
`unsharded_ranges_cover_the_full_checkpoint_tensor_shapes`.

## Fixtures are matched by `config_sha256`

Both 27B checkpoints are `(5120, 64)`, so geometry cannot name a fixture.
`hf_golden_gate.rs` scans the committed fixtures and keeps the one whose
safetensors metadata records the local `config.json`'s sha256;
`check_fixture_metadata` re-asserts that hash plus `model_revision` before a
single logit is compared, so a mispairing fails an assert instead of silently
comparing against the wrong oracle. The `SIZE_NAMES` table in
`tools/accuracy/dump_qwen35_hf_golden.py` only names the dumper's default
output; the gate never reads it.

## Chat template parity

Qwen3.8's template is where the generations genuinely differ for serving: it
reads `reasoning_effort` (default `xhigh`, restricted to `xhigh|medium|low`,
`raise_exception` otherwise), gates the reasoning instructions on
`enable_thinking`, and honours `preserve_thinking`. The renderer can express
all of it — `ChatOptions.reasoning_effort` (the vendored enum already has
`XHigh`) and `ChatOptions.template_kwargs` — but note that when
`reasoning_effort` is set the renderer *also* injects `enable_thinking`,
which HF leaves undefined; the template treats both the same, which is
precisely the kind of equivalence this gate exists to confirm.

`pegainfer-frontend/tests/qwen38_chat_template_parity.rs` covers nine cases
against `qwen38-chat-golden.json`, bound to the checkpoint by file digests,
with the shared render/compare machinery in
`pegainfer-frontend/tests/common/mod.rs` and the reference dumped by the
generic `tools/accuracy/dump_chat_template_golden.py qwen38`. A
`preserve_thinking` case is deliberately absent: against `MULTI_TURN` (whose
assistant turn carries no reasoning) it renders byte-identical to
`multi_turn`, so it tests nothing.

## Serving budget (unchanged from Qwen3.5-27B)

- Text tower ≈ **53.5 GB** bf16 (ViT ≈ 2 GB is never loaded), so one 46 GB
  card cannot hold it: serve 27B at TP2 (~26.9 GB/rank).
- GQA group is `24/4 = 6`, outside `SUPPORTED_GQA_GROUP_SIZES`, so 27B decode
  stays on the batched eager path — independent of the generation.
- Full-attention KV is 16 layers × 2 × 4 heads × 256 × 2 B = **64 KiB/token**;
  a single 262,144-token sequence reserves 16 GiB before any GDN recurrent
  state.
- The TP executor builds each rank at the caller's `--max-batch`, not the
  64-slot default: the recurrent-state pool is reserved at load time, so the
  default sizing could refuse to start on a card that could serve the
  requested capacity.

## Verification

Each row names the head it ran at. TP2 rows are against `Qwen/Qwen3.8-27B` @
`1d4bf0f2` (the text tower does not fit one card); single-GPU rows are the
Qwen3.5 sizes the shape-checked `load_tensor_2d` path actually meets on a real
checkpoint — those loads now assert every tensor's shape against the config,
and they are what "Qwen3.5 behaviour is unchanged" points to.

| Gate | Head / device | Result |
| --- | --- | --- |
| `hf_golden_gate` short, Qwen3.8-27B TP2 | pre-review head, 2×L20 (sm_89) | 2 passed / 0 failed. Sequential eager: 108 positions, mean 0.0238 / p99 0.0862 / max 0.1593. Batched eager: 72 positions, mean 0.0231 / p99 0.0823. No argmax violation. |
| `hf_golden_gate` long, Qwen3.8-27B TP2 | pre-review head, 2×L20 (sm_89) | 1 passed / 0 failed. 4097 + 8192-token prompts, 18 positions, mean 0.0226 / p99 0.0860 / max 0.0882. |
| `hf_golden_gate` Qwen3.5-0.8B, single GPU (tied head) | merged tree, 1×A40 (sm_80 build) | 2 passed / 0 failed. Short sequential: 108 positions, mean 0.0298 / p99 0.1137. Long: 18 positions, mean 0.0286 / p99 0.0926. |
| `hf_golden_gate` Qwen3.5-2B, single GPU (tied head) | merged tree, 1×A40 (sm_80 build) | 2 passed / 0 failed. Short sequential: 108 positions, mean 0.0301 / p99 0.1172. Long: 18 positions, mean 0.0238 / p99 0.0778. |
| `hf_golden_gate` Qwen3.5-4B, single GPU (untied head) | merged tree, 1×A40 (sm_80 build) | 2 passed / 0 failed. Short sequential: 108 positions, mean 0.0238 / p99 0.0813. Long: 18 positions, mean 0.0223 / p99 0.0705. |
| `qwen38_chat_template_parity` | merged tree | 1 passed / 0 failed, 9 cases byte-identical to the HF render. |
| `pegainfer-qwen35 --lib` (feature build, Triton AOT) | review-fix head, 1×L20 | 102 passed / 0 failed, the GPU recurrent tests included. |
| `pegainfer-core --lib` | review-fix head | 38 passed / 0 failed (f32-cow: 1D bf16 accepted, other dtypes/ranks rejected). |
| clippy `-D warnings` (core/qwen35/qwen3/frontend) + `cargo fmt --all --check` | merged tree | clean. |

Tolerances are the line's existing 4B calibration (`MEAN_TOL 0.06`,
`P99_TOL 0.20`) — no new constants, and every single-GPU size sits inside
them. The TP2 rows predate the review rework and the merge; the numerics
paths they cover are unchanged by either (the loader asserts the same values
earlier, widening is the same exact upcast, and the gate compares against the
same fixtures, now selected by `config_sha256`).

## How to run

The reference side needs a Python with torch 2.8 / transformers ≥ 5.2 and the
gated-DeltaNet fast path; the gates need two 46 GB-class devices for the 27B
text tower.

```bash
# 0. the GDN fast path — without it the dumper produces an all-NaN oracle
python3 -m pip install flash-linear-attention einops

# 1. reference fixtures (short and long), pinned to the revision the gate asserts
python3 tools/accuracy/dump_qwen35_hf_golden.py --model-path $D \
  --model-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --tokenizer-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
python3 tools/accuracy/dump_qwen35_hf_golden.py --model-path $D \
  --model-revision … --tokenizer-revision … \
  --prompt-lens 4097,8192 --decode-tokens 8
python3 tools/accuracy/dump_chat_template_golden.py qwen38 \
  $D test_data/qwen38-chat-golden.json \
  --source-repo Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

# 2. the gates
PEGAINFER_TEST_MODEL_PATH=$D PEGAINFER_TEST_MODEL_REVISION=1d4bf0f2… \
  PEGAINFER_TEST_TP_DEVICES=2,6 \
  cargo test -r --locked -p pegainfer-qwen35 --features qwen35 --test hf_golden_gate \
  -- --ignored --test-threads 1 _tp2
PEGAINFER_TEST_MODEL_PATH=$D \
  cargo test -r -p pegainfer-frontend --test qwen38_chat_template_parity -- --ignored

# 3. the single-GPU Qwen3.5 rows (any one size; the fixture is picked by
#    config_sha256, so point the path at whichever size's checkpoint)
PEGAINFER_TEST_MODEL_PATH=models/Qwen3.5-4B \
  PEGAINFER_TEST_MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  cargo test -r --locked -p pegainfer-qwen35 --features qwen35 --test hf_golden_gate \
  -- --nocapture
```

Two oracle traps, both now guarded:

- **`PEGAINFER_TEST_MODEL_REVISION` is not optional on a hand-downloaded
  checkpoint.** Without HF's `.cache/huggingface/download/*.metadata` the gate
  **skips** with `local model revision is unknown`, which reads like a pass in
  a test summary and proves nothing.
- **Dump the oracle through the fla fast path, and check it for NaN before
  trusting it.** Transformers' eager GDN fallback returns all-NaN logprobs on
  this geometry without `flash-linear-attention` (only at the 48 v / 16 k head
  expansion ratio), which previously read as a passing gate. The dumper
  refuses to write a golden with any non-finite reference logprob.

## Follow-ups

1. **Loader trusts what it does not verify** — #1068: unknown `config.json`
   fields are dropped silently, and the 1D loads still take whatever length
   the tensor carries.
2. **Native MTP drafting** — both generations ship the head
   (`mtp_num_hidden_layers: 1`, 15 `mtp.*` tensors) and nobody loads it.
3. **Group-6 batch-decode kernels** so 27B can capture CUDA Graphs (tracked in
   `docs/models/qwen35/tp-design.md`).
