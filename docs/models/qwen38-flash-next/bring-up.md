# Qwen3.8-Flash-Next bring-up

> **TL;DR:** `Qwen/Qwen3.8-Flash-Next` is a **new line** (`qwen4_exp`), not a Qwen3.5
> variant — its `model_type` differs, so the Qwen3.5 probe refuses it by construction.
> Frozen at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Landed so far: a
> fail-closed probe that pins *structure rather than widths*, and a config layer that
> validates the geometry and derives what `config.json` leaves implicit. Nothing runs —
> detection succeeds and `launch` refuses, because the text graph is slice B. Four facts
> this line must not get wrong, all measured rather than assumed: the embedding is
> **untied**; there is **no final-norm tensor** (the readout is the hyper-connection
> mixer); `ple_layer_ids: [2]` is **1-indexed**, so the n-gram block lives on `layers.1`;
> and `output_gate_type: "sigmoid"` is a **live** field selecting the GDN gated-norm
> activation — the inverse of the Qwen3.8-27B lesson, where the same field name is inert.
>
> **Last touched:** 2026-10

Tracked in #1103 (roadmap) with slices A #1104 / B #1127 / C #1128 and component
bricks #1105 (QSA), #1106 (n-gram), #1107 (MoE), #1108 (hyper-connections).

## The freeze

| | |
| --- | --- |
| Repo / revision | `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Access | public, non-gated |
| Weight licence | **`license: other`** — not Apache-2.0, unlike the code this line ports from |
| Size | 131 shards, 360.0 GB, 1658 tensor keys |
| Identity | `model_type: qwen4_exp`, `architectures: [Qwen4ExpForConditionalGeneration]`, `text_config.model_type: qwen4_exp_text` |
| Reference impl | `transformers/models/qwen4_exp/modeling_qwen4_exp.py`; the reference first ships in **transformers 5.16.0** — the `5.8.0.dev0` in `config.json` is the checkpoint's own declared version, not a release that carries it (absent through 5.15.0) |

File hashes at that revision (HF blob sha1; `tokenizer.json` is LFS, so its sha256 is
given too):

| File | Bytes | sha1 | sha256 (LFS) |
| --- | --- | --- | --- |
| `config.json` | 4,745 | `491017e9980e44ef01afa2c4782f5c7e169b9b26` | — |
| `tokenizer.json` | 12,809,320 | `9328ce9c41e80f6dd7bc2c66d8ae1fc93bf87440` | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `tokenizer_config.json` | 17,928 | `5de744b3fca2129d7186979ae47c06be33903243` | — |
| `chat_template.jinja` | 8,952 | `c0c686f9c38d70d179fb7b5f5aa7530bc913dda3` | — |
| `generation_config.json` | 202 | `023756cfadf88e5bf69eefeee3e172f38c448d64` | — |

The committed `config.json` (`pegainfer-qwen38-flash-next/tests/frozen_config.json`)
is the real file, not a transcription, and a test re-hashes it against
`FROZEN_CONFIG_SHA256` so the fixture cannot drift from the revision it claims to be.
Every probe and config test runs against it.

## What landed

`pegainfer-qwen38-flash-next`, feature `qwen38-flash-next`, **with no CUDA dependency
at all** — so its 51 tests need no device, no weights and no Triton-equipped build.
Preserving that is the point of the crate's shape: when slice B adds the device side,
`config` and `probe` must move behind `#[cfg(any(feature, test))]` the way
`pegainfer-gemma4/src/lib.rs` does.

- **Probe** (`src/probe.rs`) — fail-closed identity plus structure, following the
  Gemma 4 pattern of pinning *structure rather than widths*. Pinned because an
  alternative would need a kernel or serving path this line does not have: the GDN
  head dims (they are the Triton AOT specialization keys), `head_dim` 256, the conv
  kernel ceiling, `rope_type: default` (no YaRN path), `output_gate_type: sigmoid`,
  `mamba_ssm_dtype: float32`, `dtype: bfloat16` (so the **FP8 sibling fails closed**
  rather than loading as BF16), and untied embeddings. Deliberately *not* pinned:
  expert count, hidden size, layer count, head counts, `hc_count` — so a smaller
  `qwen4_exp` checkpoint reuses this probe, which is what the L20 small-model
  deployment case needs. `smaller_widths_still_pass` is what keeps that promise honest.
  The same list carries `mrope_interleaved` and `mrope_section`, which are accepted
  rather than refused: a text-only sequence is their degenerate case in the reference
  itself, which expands one position per token across all three axes before
  interleaving. The decision is tracked in #1105. Beyond the pins, the probe mirrors
  the rules the reference's own `validate_architecture` enforces — `hc_count > 1`,
  QSA's `indexer_kv_heads == 1` with the budget a whole multiple of the compress
  ratio, the rotary width fitting the index head, `rope_theta` present, and PLE
  confined to a **single** `linear_attention` layer — since a config the reference
  refuses cannot be served by any port. Multi-layer PLE is refused for now: the
  reference offsets each layer's prime run by its layer index (`global_head_idx =
  L * ngram_heads + i`), and this crate derives one table.
- **Config** (`src/config.rs`) — validated geometry plus the derivations the rest of the
  line needs. Two fields the reference reads are **absent from `config.json`** and
  come from upstream defaults: `norm_topk_prob` (`true`, and it decides whether the
  router renormalizes its top-k) and `seed`. Both are modelled explicitly with their
  defaults, because a port that reads only `config.json` silently drops the router
  renormalization — this is #1068's mirror image.
- **Registration** — the server binary hand-mirrors every line's probe identities in a
  hint descriptor its own comment says cannot be derived away, so the mirror is tested
  and not just the probe: `feature_gate_hint` names `--features qwen38-flash-next` for
  both `qwen4_exp` and `qwen4_exp_text` when the feature is off, and stays silent when
  it is on.

## The n-gram table is derived, not read

`config.json` states none of the table's dimensions; they follow from five fields:

```
heads            = (ngram_size - 1) * heads_per_ngram = 2 * 8 = 16
head_dim         = ple_embed_dim / heads              = 2560 / 16 = 160
head i vocab     = the (i+1)-th prime after (ngram_vocab_size_base - 1)
total            = sum of the 16 primes                = 320,001,446
padded           = ceil(total / 128) * 128             = 320,001,536
rows per shard   = padded / split_ngram_parts          = 2,500,012   (128 shards)
bytes at BF16    = 102,400,491,520                     = 102.4 GB
```

**102.4 GB, not the ~51 GB vLLM and SGLang quote** — that figure is the fp8 footprint,
and this checkpoint ships BF16. Every host-RAM budget in the issue set was 2× under
until it said which dtype it meant. The FP8 sibling is a different revision and would
need freezing separately.

The derivation lands here rather than with its first consumer because #1106 (the n-gram
brick) needs the geometry independently of any tensor contract, and because it is pure
arithmetic over `config.json` — `nth_prime_after_matches_the_reference` pins the prime
search against the reference's `_find_nth_prime_after`.

## Traps

- **`output_gate_type` has three unrelated consumers.** On Qwen3.8-27B it is inert and
  must *not* be implemented (`docs/models/qwen35/support-qwen38.md`). Here it is live and
  selects the GDN gated-norm activation; the shared `csrc/shared/norm.cu` gated norm
  **became selectable in #1129** (a sigmoid entry point beside Qwen3.5's SiLU, merged
  before this line's detection PR), so what remains is this line's call site, which
  lands with the text graph. It is *also* the marker the frontend's `reasoning_effort`
  layer arms on. Read the field's consumer, never its name.
- **Two RMSNorm conventions in one layer.** `(1+w)` for `q_norm`/`k_norm`, the indexer
  layernorms, `hc_norm` and the PLE norms; plain `w` only for the GDN gated norm. The
  four 10240-wide ones are additionally **grouped by 2560** — likely a reshape of the
  existing batched `(1+w)` kernel rather than a new kernel, but unverified.
- **The PLE conv is dilated** (kernel 4, dilation `ngram_size` = 3, state length 9) even
  though its weight tensor `[10240, 1, 4]` is shaped exactly like the GDN conv's. The
  in-tree `conv1d.cu` is non-dilated and caps its state at four taps, so it cannot serve
  this.
- **Do not use `load_shard_info_fixed`.** It is a Qwen3.5-specific shard-*filename*
  workaround living in shared `pegainfer-core`; this checkpoint uses standard
  `model-0000N-of-00131` naming and wants the plain loader.

## Still open

| Criterion | Blocked on |
| --- | --- |
| Tensor manifest enforced before the first H2D | Nothing — CPU only. The Gemma 4 classifier is the porting precedent; the deltas this checkpoint forces are an untied LM head, rank-3 shapes (the depthwise conv and the fused expert stacks, which carry **no `.weight` suffix**), the `model.visual.` / `mtp.` skip set, and 1-D lengths, which the shared loader does **not** check (#1068). |
| GDN reuse proof, executed | 1 GPU. Answered as far as tensors go — geometry equals the Triton AOT constants, conv width equals the in-tree cap, names match down to the `model.language_model.layers.N.` prefix — but `conv1d_decode_batch_cuda` / `gated_delta_rule_decode_batch_cuda` are wrapped only inside `pegainfer-qwen35/src/recurrent.rs`, so slice B has to decide whether to lift them into shared `ops` or re-wrap them. |
| Bookend gate (embedding → mixer → lm_head → argmax) | 1 GPU, three shards (1 + 130 + 131, 5.29 GB, ~2.6 GB of weights on device) and a CPU f64 reference for the mixer. No layer weights needed. |
| HF golden fixtures (logits + intermediate probes) | the fleet — 360 GB, plus ≥102.4 GB host RAM if the n-gram table is materialized |
| Tokenizer / chat-template ID fixtures | transformers **5.16.0+** — the checkpoint's own tokenizer is `Qwen2Tokenizer`, which is older, but fixtures are compared against the reference implementation; nothing in the tree commits token-ID fixtures today |

Note that `pegainfer-server` cannot be compiled on a CPU-only box, so the registration
above is verifiable only where a CUDA build works.

A local box here is **sm_120**, which is neither support-matrix row (H200 sm_90, L20
sm_89) — anything proven there establishes numerics and must never be recorded as a
platform result.
