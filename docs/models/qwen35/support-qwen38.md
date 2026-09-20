# Serve Qwen3.8-27B on the Qwen3.5 line

> **TL;DR:** Qwen3.8-27B is not a new model line: its text tower is
> shape-identical to Qwen3.5-27B's and its `config.json` still reports
> `model_type: qwen3_5`, so the probe, the weights, the kernels, the scheduler and
> the server wiring all pass unchanged. Two things actually blocked it — Qwen3.8
> stores the gated-DeltaNet scalars as bf16 where the loader demanded f32, and the
> logits-golden fixture key had no generation dimension to tell two
> `(5120, 64)` checkpoints apart. Both are fixed here. `output_gate_type: "swish"`
> in its config is **not** a numerics change and must not be implemented: the
> attention output gate is sigmoid, measured. Qwen3.8-27B now clears this line's
> TP2 logits gate (mean 0.0238 / p99 0.0862 short, 0.0226 / 0.0860 long) and the
> chat-template parity gate. Tracked in #1067.
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
| shard layout | 18 shards `model-000NN-of-00018.safetensors` (3.5: 11 shards, differently named) | irrelevant — the loader follows `weight_map` from the index |
| dtype | 100% BF16 (3.5: 1,103 BF16 + 96 F32) | **the only load blocker** |
| shapes | 0 differing tensors in shape; 96 differ in dtype only | see below |
| tokenizer | same 33 added tokens, same max id 248,076, same vocab width | no decodable-vocab change |
| chat template | 8,952 chars vs 7,756; adds `reasoning_effort`, `preserve_thinking` | the one real behaviour delta |
| unused weights | 333 `model.visual.*` (never loaded, same in 3.5) + 15 `mtp.*` | MTP is a separate opportunity |

## The gate is sigmoid. Do not implement "swish".

Qwen3.8's `text_config` adds `output_gate_type: "swish"`, and PegaInfer's
full-attention output gate is hardcoded sigmoid
(`pegainfer-kernels/csrc/qwen35/prefill_attention_hd256.cu:190`). That looked like
a numerics break. It is not, on three independent grounds:

1. **No reader.** `output_gate_type` appears zero times in transformers'
   `qwen3_5` config *and* modelling code — checked at 5.2.0, **5.8.0** (the release
   matching the checkpoint's declared `5.8.0.dev0`) and 5.17.0. The only consumer
   in the whole wheel is `qwen4_exp`, a different architecture, whose validator
   accepts `{"sigmoid","silu"}` — `"swish"` is not even a legal value there.
2. **Reading it would change nothing.** swish ≡ silu, and the gated RMSNorm that
   `qwen4_exp` points at is exactly where PegaInfer already applies silu
   (`pegainfer-kernels/csrc/shared/norm.cu:58`), matching HF's
   `Qwen3_5RMSNormGated`.
3. **The weights say so.** Probe method (and the result for both generations) is
   in the next section.

Why the probe was needed at all: `TextConfig` has no
`#[serde(deny_unknown_fields)]` (`pegainfer-qwen35/src/config/model.rs`), so an
unlisted field is dropped **silently** — a checkpoint can load, boot and serve
garbage. That is the class of failure this document keeps tripping over; see the
loader-trust follow-up issue.

### The two-hypothesis probe

`torch.sigmoid` is swapped for `x·sigmoid(x)` at the module's single
`torch.sigmoid(` call site, and the same inputs run under both. Teacher-forced
NLL on coherent technical prose, plus greedy continuations:

| Checkpoint | sigmoid | silu | top-1 agreement |
| --- | --- | --- | --- |
| Qwen3.5-0.8B (control) | NLL 4.276 — ` Paris.\nThe capital of France is…` | NLL 14.176 — `ermenurgieve奥特diaitinicks…` | 0/3 |
| Qwen3.8-27B | **NLL 1.822** — ` Paris.\nThe capital of Germany is Berlin.` / `    pivot = arr[len(arr) // 2]` / ` Celsius at sea level. At higher altitudes` | NLL 14.268 — `pozy舞良陆tmp9胤-plugins三aben_soup经济开发区` | 0/3 |

A ~12 nat/token gap and multilingual token soup: the gate the weights were
trained under is sigmoid, and the control proves the instrument measures it.
Reproduce with `probe_gate.py` (method above); the numbers are the ones the
follow-up PR must not regress.

## BF16-stored GDN scalars: widen, don't relax

The 16,896-byte difference between the two `total_size` values is exactly
`48 layers × (A_log 48 + norm.weight 128) × 2 bytes`: those two vectors per
linear layer are f32 in Qwen3.5 and bf16 in Qwen3.8. Qwen3.5-0.8B stores them f32
too, so this is a **save-time cast policy change in transformers 5.x** (norm
parameters get cast to `torch_dtype`), not an architecture change.

PegaInfer demanded f32 at both loads (`weights/layers.rs` A_log /
`linear_attn.norm.weight`), so Qwen3.8 failed to load — loudly, with
`expected dtype F32, got BF16`, which is the behaviour we want from a fail-closed
loader.

The fix is a widened loader used **only** for these two vectors
(`load_tensor_1d_f32_widened`, `load_tensor_1d_f32_shard_widened` in
`pegainfer-core/src/weight_loader.rs`), not a relaxation of `tensor_f32_cow`:
that function's refusal of bf16 is pinned by
`tensor_f32_cow_rejects_wrong_dtype_and_rank`, and other lines rely on it.
bf16 → f32 is exact, and HF itself upcasts at the point of use
(`g = -self.A_log.float().exp() * …`), so both storages reach the kernels as the
same values.

The same audit found a second silent-trust hole and closed it: on the unsharded
(single-GPU) path every projection falls back to a `tensor_2d` that takes the
tensor's own shape as truth, so only `lm_head` was checked against the config. The
`config.json`-derived extents are now asserted for `embed_tokens`, `q_proj`
(including its per-head `[q, gate]` doubling), `k/v/o_proj`, the MLP pair and the
fused linear qkv, with `unsharded_ranges_cover_the_full_checkpoint_tensor_shapes`
pinning those extents to the shapes read off the real 27B checkpoint.

## The fixture key gained a generation

`fixture_size_name` keyed fixtures on `(hidden_size, num_hidden_layers)`, and both
27B checkpoints are `(5120, 64)`. The key is now `(generation, size)`, with the
generation read from the same incidental field (`output_gate_type` present →
`qwen38`). Because the discriminator is incidental, safety comes from
composition: each fixture records `config_sha256` + `model_revision`, and
`check_fixture_metadata` asserts them before a single logit is compared — so a
mispairing fails an assert instead of silently comparing against the wrong oracle.
`SIZE_NAMES` in `tools/accuracy/dump_qwen35_hf_golden.py` carries the same table
and must move with it.

## Chat template parity

Qwen3.8's template is where the two generations genuinely differ for serving: it
reads `reasoning_effort` (default `xhigh`, restricted to `xhigh|medium|low`,
`raise_exception` otherwise), gates the reasoning instructions on
`enable_thinking`, and honours `preserve_thinking`. The renderer can express all
of it — `ChatOptions.reasoning_effort` (the vendored enum already has `XHigh`) and
`ChatOptions.template_kwargs` — but note that when `reasoning_effort` is set the
renderer *also* injects `enable_thinking`, which HF leaves undefined; the template
treats both the same (`enable_thinking is undefined or enable_thinking is true`),
which is precisely the kind of equivalence this gate exists to confirm.

`pegainfer-frontend/tests/qwen38_chat_template_parity.rs` covers ten cases
(plain, multi-turn, unicode, no-generation-prompt, each effort level, thinking
disabled alone and with an effort, `preserve_thinking`) against the
`qwen38-chat-golden.json` reference, bound to the checkpoint by file digests — the
same shape as `gemma4_tokenizer_parity.rs`, with the shared render/compare
machinery now in `pegainfer-frontend/tests/common/mod.rs`.

## Serving budget (unchanged from Qwen3.5-27B)

- Text tower ≈ **53.5 GB** bf16 (ViT ≈ 2 GB is never loaded), so one 46 GB card
  cannot hold it: serve 27B at TP2 (`docs/models/qwen35/tp-design.md`'s per-rank
  figure predates this checkpoint; it is ~26.9 GB/rank, not 17.5).
- GQA group is `24/4 = 6`, outside `SUPPORTED_GQA_GROUP_SIZES`, so 27B decode
  stays on the batched eager path — independent of the generation.
- Full-attention KV is 16 layers × 2 × 4 heads × 256 × 2 B = **64 KiB/token**; a
  single 262,144-token sequence reserves 16 GiB before any GDN recurrent state.
- The 15 `mtp.*` tensors (both generations ship `mtp_num_hidden_layers: 1`) are
  loaded by nobody. Native-MTP speculation is the obvious free lunch on this line;
  precedent exists in `glm52` (`GLM52_MTP_DRAFTS`) and qwen3 (DFlash/DSpark).

## Verification

All on two 46 GB `sm_89` devices, `--features qwen35`, against
`Qwen/Qwen3.8-27B` @ `1d4bf0f2`, TP2 (the text tower does not fit one card):

| Gate | Result |
| --- | --- |
| `hf_golden_gate` short, TP2 | 2 passed / 0 failed. Sequential eager: 108 positions, mean 0.0238 / p50 0.0199 / p99 0.0862 / max 0.1593. Batched eager: 72 positions, mean 0.0231 / p99 0.0823. No argmax violation. Graph variant self-skips (group 6 has no compiled batch-decode kernel). |
| `hf_golden_gate` long, TP2 | 1 passed / 0 failed. 4097 + 8192-token prompts, 18 positions, mean 0.0226 / p99 0.0860 / max 0.0882. |
| `qwen38_chat_template_parity` | 1 passed / 0 failed, 10 cases byte-identical to the HF render. |
| `pegainfer-core --lib` | `tensor_f32_widen_cow_*` 2 passed (widening is exact; f16/i64/wrong-rank still rejected). |
| `pegainfer-qwen35 --lib` | 103 passed / 0 failed, incl. `unsharded_ranges_cover_the_full_checkpoint_tensor_shapes`. |
| `clippy --all-targets -D warnings` (core, qwen35, frontend) + `cargo fmt --all --check` | clean. |

Tolerances are the line's existing 4B calibration (`MEAN_TOL 0.06`, `P99_TOL 0.20`)
and 3.8-27B sits well inside them — the same band the 0.8B/2B floors report
(mean 0.023–0.030, p99 ≤ 0.115), so no new size- or generation-specific tolerance
was needed.

## How to run

The reference side needs a Python with torch 2.8 / transformers ≥ 5.2 and the
gated-DeltaNet fast path; the gates need two 46 GB-class devices for the 27B text
tower.

```bash
# 0. the GDN fast path — without it the dumper produces an all-NaN oracle
python3 -m pip install flash-linear-attention einops

# 1. checkpoint, straight from HF
curl -s "https://huggingface.co/api/models/Qwen/Qwen3.8-27B/tree/main?recursive=true" \
  | python3 -c 'import json,sys; [print(f["path"]) for f in json.load(sys.stdin)]' \
  | while read f; do case "$f" in *.safetensors|*.json|*.txt) curl -sL --fail \
      -o "$D/$f" "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/$f";; esac; done

# 2. reference fixtures (short and long), pinned to the revision the gate asserts
python3 tools/accuracy/dump_qwen35_hf_golden.py --model-path $D \
  --model-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --tokenizer-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --max-memory-gib 21                       # shared tray: accelerate plans against TOTAL memory
python3 tools/accuracy/dump_qwen35_hf_golden.py --model-path $D \
  --model-revision … --tokenizer-revision … --max-memory-gib 21 \
  --prompt-lens 4097,8192 --decode-tokens 8
python3 tools/accuracy/dump_qwen38_chat_golden.py $D test_data/qwen38-chat-golden.json \
  --source-repo Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

# 3. the gates
PEGAINFER_TEST_MODEL_PATH=$D PEGAINFER_TEST_MODEL_REVISION=1d4bf0f2… \
  PEGAINFER_TEST_TP_DEVICES=2,6 \
  cargo test -r --locked -p pegainfer-qwen35 --features qwen35 --test hf_golden_gate \
  -- --ignored --test-threads 1 _tp2
PEGAINFER_TEST_MODEL_PATH=$D \
  cargo test -r -p pegainfer-frontend --test qwen38_chat_template_parity -- --ignored
```

`PEGAINFER_TEST_MODEL_REVISION` is not optional on a hand-downloaded checkpoint:
without HF's `.cache/huggingface/download/*.metadata` the gate **skips** and prints
`local model revision is unknown`, which reads like a pass in a test summary and
proves nothing.

**Dump the oracle through the fla fast path, and check it for NaN before trusting
it.** The first 27B dump here came out with all 6,912 reference logprobs NaN, and
the gate reported it as `mean NaN … worst head delta NaN @ seq 0 pos 0 (pega
-6.8022, HF NaN)` — pegainfer's side was fine. The cause was transformers'
eager torch fallback for gated DeltaNet (the pod lacked
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention), and
the load printed `The fast path is not available … Falling back to torch
implementation`). The fallback is not universally broken — the same code produced
finite logits for Qwen3.5-0.8B — so the failure only shows up at this
architecture's value/key head expansion ratio (48 v / 16 k heads). With
`PYTHONPATH` pointing at an `fla` install the dump is finite end to end.

So: install `flash-linear-attention` (+ `einops`) before dumping, keep it out of
the shadowing set (`transformers`/`numpy`/`tokenizers`/`safetensors` must stay the
image's). The dumper now refuses to write a golden with any non-finite reference
logprob and points at that install, so the fallback cannot produce a silently
useless oracle again.

## TP capacity now reaches the loader

Getting 27B onto a partially-occupied card exposed a plumbing bug unrelated to the
generation: `Qwen35TpExecutor::from_runtime_with_capacity` built each rank with
`Qwen35Model::from_safetensors_with_runtime`, which defaults to
`batch_decode_graph::MAX_BATCH` (64) slots. The recurrent-state pool is sized at
load time, so the requested capacity — and the later `min_rank_max_batch`
negotiation — could never give that memory back: a caller asking for 8 decode
slots still reserved 64. On a 46 GB card shared with another tenant that is the
difference between fitting and failing with
`insufficient device memory … recurrent state needs 9396 MB (2 x 64 decode slots)`.
The TP builder now calls `from_safetensors_with_runtime_and_capacity` with the
caller's number, matching the single-GPU path. Default `--max-batch 64` is
byte-identical; a lowered `--max-batch` now also shrinks the graph-bucket
pre-capture sweep, which is the documented intent of the flag.

## Follow-ups

1. **Loader trusts what it does not verify** — #1068. Two instances remain:
   unknown `config.json` fields are dropped silently because `TextConfig` has no
   `deny_unknown_fields`, and the 1D loads still take whatever length the tensor
   carries. The 2D assertion hole is closed here, because a shape that disagrees
   with the config was the next failure in line. Deciding the field policy is a
   cross-line call (fail on any new field, or declare-and-whitelist), which is why
   it is its own issue rather than a bolt-on to this one.
2. **Native MTP drafting** for this line — both generations ship the head
   (`mtp_num_hidden_layers: 1`, 15 `mtp.*` tensors) and nobody loads it. Precedent
   exists in `glm52` (`GLM52_MTP_DRAFTS`) and qwen3 (DFlash/DSpark).
3. **Group-6 batch-decode kernels** so 27B can capture CUDA Graphs (tracked in
   `docs/models/qwen35/tp-design.md`).
