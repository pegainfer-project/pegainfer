# Serve Qwen3.8-27B on the Qwen3.5 line

> **TL;DR:** Qwen3.8-27B is not a new model line: its text tower is
> shape-identical to Qwen3.5-27B's and its `config.json` still reports
> `model_type: qwen3_5`, so the probe, the weights, the kernels, the scheduler
> and the server wiring all pass unchanged. Two things actually blocked it —
> Qwen3.8 stores the gated-DeltaNet scalars as bf16 where the loader demanded
> f32 (the f32 loader now widens 1D bf16; the conversion is exact), and the
> logits-golden fixture key could not tell two `(5120, 64)` checkpoints apart
> (the gate now picks the fixture whose recorded `config_sha256` matches the
> checkpoint, and refuses to run when the local revision cannot be resolved).
> `output_gate_type: "swish"` in its config is **not** a numerics change and
> must not be implemented: the attention output gate is sigmoid, measured. Its
> two real serving deltas are both in the chat template, not the tower:
> `reasoning_effort` accepts only `xhigh|medium|low`, so the frontend maps the
> OpenAI-only `high`/`max` onto `xhigh` and `minimal` onto `low` before the
> template sees them, and its tool format is the Qwen Coder one that `Auto`
> parser selection cannot reach from a `Qwen3.8-27B` directory — name the
> parser with `--tool-call-parser`. Both serving notes are measured against the
> checkpoint's own template, and the TP2 short/long logits rows plus the seven
> chat cases have been re-run at the current head. Tracked in #1067.
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
`tensor_f32_cow_accepts_bf16_and_rejects_invalid_dtype_or_rank`.

The same audit closed a second silent-trust hole on the single-GPU path: there
the unsharded load took each tensor's own shape as truth, so a checkpoint whose
tensors disagreed with its `config.json` reached a config-sized GEMM.
`load_tensor_2d` now takes the config-derived `(rows, cols)` and checks the
header before upload. That check covers the unsharded projections and the
replicated embedding / untied `lm_head`; the **TP shard loaders still
bounds-check only** (full config-shape verification there is not part of this
change), and the 1D loads still take the length the tensor carries (#1068).

## Fixtures are matched by `config_sha256`

Both 27B checkpoints are `(5120, 64)`, so geometry cannot name a fixture.
`hf_golden_gate.rs` scans the committed fixtures and keeps the one whose
safetensors metadata records the local `config.json`'s sha256;
`check_fixture_metadata` re-asserts that hash plus `model_revision` before a
single logit is compared, so a mispairing fails an assert instead of silently
comparing against the wrong oracle. The revision is the only field that pins the
*weights* — the config hash pins geometry, which two 27B checkpoints share — so
an unresolvable local revision is now a **panic naming
`PEGAINFER_TEST_MODEL_REVISION`**, not a skip: a skip reported `ok` without
comparing anything. The `SIZE_NAMES` table in
`tools/accuracy/dump_qwen35_hf_golden.py` only names the dumper's default
output; the gate never reads it.

## Chat template parity

Qwen3.8's template is where the generations genuinely differ for serving: it
reads `reasoning_effort` (default `xhigh`, restricted to `xhigh|medium|low`,
`raise_exception` otherwise) and gates the reasoning instructions on
`enable_thinking`. The renderer can express both — `ChatOptions.reasoning_effort`
(the vendored enum already has `XHigh`) and `ChatOptions.template_kwargs` — but
note that when `reasoning_effort` is set the renderer *also* injects
`enable_thinking`, which HF leaves undefined; the template treats both the same,
which is precisely the kind of equivalence this gate exists to confirm.

**The template's `reasoning_effort` check is inside the thinking branch**, so
only the three OpenAI values that have no template equivalent ever failed it:
`enable_thinking=false` — which the renderer derives from
`reasoning_effort=none` — skips the check entirely. The check is a
`raise_exception` call, which the pinned frontend does not register, so a
rejected render arrives as an unknown-function error and is mapped to
`server_error` (500), not a 400.

`pegainfer-frontend/src/vllm/reasoning_effort.rs` closes that: a route layer over
`/v1/chat/completions` rewrites the *top-level* field before the upstream
renderer converts it, `high`/`max` → `xhigh` and `minimal` → `low`.

| `reasoning_effort` | before | after |
| --- | --- | --- |
| `high`, `max`, `minimal` | 500 `server_error` (`unknown function: raise_exception is unknown (in chat:49)`) | 200 |
| `none`, `low`, `medium`, `xhigh` | 200 | 200 |

Measured against the checkpoint's own template (`pegainfer-sim` pointed at the
27B directory, CPU only). `none`, `low`, `medium` and `xhigh` pass through
untouched, and a request with no top-level field is not modified at all. An
explicit `chat_template_kwargs.reasoning_effort` is deliberately **not**
rewritten: that is the caller addressing the template directly, and a value
outside its vocabulary still fails the render there — the layer rewrites the API
field, not the template contract.

## Tool-call parsing

`tool_call_parser` defaults to `Auto`, which resolves the parser by
case-insensitive substring match against the **model path** — `qwen3.5` maps to
`qwen3_coder` and bare `qwen3` to `qwen3_xml`. A directory named
`Qwen3.8-27B` contains `qwen3` but not `qwen3.5`, so `Auto` selects the JSON
`qwen3_xml` parser; `--served-model-name` does not enter the match. If the
checkpoint emits the Qwen Coder format, name it explicitly:

```bash
--tool-call-parser qwen3_coder
```

Measured (`pegainfer-sim` on the checkpoint's directory, CPU only): a
tools-bearing request against a server whose model path is
`/…/models/Qwen3.8-27B` logs `using tool parser parser_name="qwen3_xml"` — the
mismatch the flag exists to fix. The flag appears in `--help` with
`[default: auto]`, and an unregistered name is refused before an engine load:

```text
Error: invalid --tool-call-parser: tool parser `this-parser-does-not-exist` is not registered (choose from: … qwen3_coder, qwen3_xml, …)
```

`pegainfer-sim/tests/tool_call_roundtrip.rs` covers the HTTP path: an `Auto`
request on a family-less model directory fails, the same request with an
explicit parser parses into `tool_calls`, and the streaming / non-streaming
cases still pass. What this does **not** establish is which grammar *this*
checkpoint emits — that needs a GPU run with the real weights; `qwen3_coder` is
the name to try first if its calls are the `<function=…>` form.

Parser selection is per-deployment, not per-request, and `--served-model-name`
does not influence it.

`pegainfer-frontend/tests/qwen38_chat_template_parity.rs` covers seven cases
against `qwen38-chat-golden.json`, bound to the checkpoint by file digests,
with the shared render/compare machinery in
`pegainfer-frontend/tests/common/mod.rs` and the reference dumped by the
generic `tools/accuracy/dump_chat_template_golden.py qwen38`. The template also
honours `preserve_thinking`, but no case exercises it: against `MULTI_TURN`
(whose assistant turn carries no reasoning) it renders byte-identical to
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

Every row names the head it ran at. `current head` is the review-fix head
(`80301ad2` + this change) built with `--features qwen35` at
`PEGAINFER_CUDA_SM=89`; the TP2 rows are `Qwen/Qwen3.8-27B` @ `1d4bf0f2` on
2×L20 (the text tower does not fit one card), and the single-GPU rows are the
Qwen3.5 sizes the unsharded `load_tensor_2d` path actually meets on a real
checkpoint — those loads assert every 2D tensor's shape against the config, and
they are what "Qwen3.5 behaviour is unchanged" points to.

The TP2 and chat rows were re-run at the current head after the fixture-lookup
and fail-closed-revision changes (the revision check decides whether the gate
compares at all), and they reproduce the earlier numbers exactly. The
`reasoning_effort` layer and the formatting pass landed after those runs: the
logits gate and the chat-parity test both drive the engine and the renderer
directly and never issue an HTTP request, so neither can see that layer, and the
rows that do exercise it were re-run on the final bytes.

| Gate | Head / device | Result |
| --- | --- | --- |
| `hf_golden_gate` short, Qwen3.8-27B TP2 | current head, 2×L20 (sm_89) | 1 passed / 0 failed. Sequential eager: 108 positions, mean 0.0238 / p50 0.0199 / p99 0.0862 / max 0.1593. Batched eager: 72 positions, mean 0.0231 / p99 0.0823 / max 0.1274. No argmax violation. |
| `hf_golden_gate` long, Qwen3.8-27B TP2 | current head, 2×L20 (sm_89) | 1 passed / 0 failed. 4097 + 8192-token prompts, 18 positions, mean 0.0226 / p50 0.0206 / p99 0.0860 / max 0.0882. |
| `qwen38_chat_template_parity` (7 cases) | current head | 1 passed / 0 failed, every case byte-identical to the HF render. |
| `hf_golden_gate` fail-closed unit tests | current head | 4 passed / 0 failed: unresolved revision panics naming `PEGAINFER_TEST_MODEL_REVISION`, a mismatched revision panics, a matching one is accepted, and zero fixture matches panic. |
| `hf_golden_gate` Qwen3.5-0.8B, single GPU (tied head) | merged tree, 1×A40 (sm_80 build) | 2 passed / 0 failed. Short sequential: 108 positions, mean 0.0298 / p99 0.1137. Long: 18 positions, mean 0.0286 / p99 0.0926. |
| `hf_golden_gate` Qwen3.5-2B, single GPU (tied head) | merged tree, 1×A40 (sm_80 build) | 2 passed / 0 failed. Short sequential: 108 positions, mean 0.0301 / p99 0.1172. Long: 18 positions, mean 0.0238 / p99 0.0778. |
| `hf_golden_gate` Qwen3.5-4B, single GPU (untied head) | merged tree, 1×A40 (sm_80 build) | 2 passed / 0 failed. Short sequential: 108 positions, mean 0.0238 / p99 0.0813. Long: 18 positions, mean 0.0223 / p99 0.0705. |
| serving probe (sim + real 27B template, CPU) | current head | Auto selects `qwen3_xml` on the 27B path; `reasoning_effort` maps `high`/`max`→`xhigh` and `minimal`→`low` (measured 200 where they were 500), pass-through values unchanged, kwargs-only effort still rendered raw; `--tool-call-parser` exposed and its invalid-name rejection fires before engine load. |
| `reasoning_effort` normalization (unit) | current head | 4 passed / 0 failed: the aliases map, `none`/`low`/`medium`/`xhigh` pass through, an absent/non-string/non-JSON body is untouched, and a kwargs-only effort is never rewritten. |
| `pegainfer-qwen35 --lib` (feature build, Triton AOT) | review-fix head, 1×L20 | 102 passed / 0 failed, the GPU recurrent tests included. |
| `pegainfer-core --lib` | review-fix head | 38 passed / 0 failed (f32-cow: 1D bf16 accepted, other dtypes/ranks rejected). |
| `pegainfer-frontend --lib` | current head | 84 passed / 0 failed, the CLI consume-or-reject schema tests included. |
| `frontend_e2e` + `tool_call_roundtrip` (CPU, `pegainfer-sim`) | current head | 23 passed / 0 failed and 5 passed / 0 failed; `frontend_e2e` carries the guarded-template `reasoning_effort` case (and still fails a kwargs-only effort, which is the deliberate boundary), `tool_call_roundtrip` carries the `Auto`-vs-explicit parser contrast. |
| clippy `-D warnings` | current head | `pegainfer-qwen35 --features qwen35 --all-targets`, `pegainfer-frontend` and `pegainfer-sim` (all targets) clean; `cargo fmt --all --check` clean. core/qwen3 as recorded earlier. |

Tolerances are the line's existing 4B calibration (`MEAN_TOL 0.06`,
`P99_TOL 0.20`) — no new constants, and every size sits inside them. Run each
TP2 test by exact name (see "How to run"): a `_tp2` substring filter also
selects the graph test, whose group-6 skip is not an eager pass.

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

# 2. the gates. Run each TP2 test by exact name: a `_tp2` substring filter also
#    selects the graph test, whose group-6 skip is not an eager pass.
PEGAINFER_TEST_MODEL_PATH=$D PEGAINFER_TEST_MODEL_REVISION=1d4bf0f2… \
  PEGAINFER_TEST_TP_DEVICES=2,6 \
  cargo test -r --locked -p pegainfer-qwen35 --features qwen35 --test hf_golden_gate \
  -- --ignored --exact --nocapture --test-threads 1 \
  pega_logprobs_match_hf_golden_within_qwen35_tolerance_tp2
PEGAINFER_TEST_MODEL_PATH=$D PEGAINFER_TEST_MODEL_REVISION=1d4bf0f2… \
  PEGAINFER_TEST_TP_DEVICES=2,6 \
  cargo test -r --locked -p pegainfer-qwen35 --features qwen35 --test hf_golden_gate \
  -- --ignored --exact --nocapture --test-threads 1 \
  pega_logprobs_match_hf_long_golden_within_qwen35_tolerance_tp2
PEGAINFER_TEST_MODEL_PATH=$D \
  cargo test -r --locked -p pegainfer-frontend --test qwen38_chat_template_parity \
  -- --ignored --exact --nocapture chat_renders_match_hf_reference

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
  cannot resolve the local revision and **panics** naming the variable, because
  `config_sha256` pins geometry but not the weights. (Fixtures also refuse a
  recorded revision of `unknown`, and a mismatch fails.)
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
4. **`reasoning_effort` through `chat_template_kwargs` is still raw** — the
   normalization layer rewrites the API field only; a caller that addresses the
   template directly gets the template's own rejection (and, at the pinned
   `vllm-chat` rev, a 500 rather than a 400, because `raise_exception` is not
   registered there). Registering it and carrying an intentional template
   rejection as a typed request error is a `vllm-chat` change — a fork or an
   upstream PR plus a rev bump — not something this repo can do from its side.
