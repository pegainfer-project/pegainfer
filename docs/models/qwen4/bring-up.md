# Qwen4 bring-up

> **TL;DR:** `Qwen/Qwen3.8-Flash-Next`, the first checkpoint of the Qwen4 architecture preview, is a **new line** (`qwen4_exp`), not a Qwen3.5 variant — its `model_type` differs, so the Qwen3.5 probe refuses it by construction. Frozen at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Landed so far: an identity probe and one typed config validation that pins *structure rather than widths* and derives what `config.json` leaves implicit, plus a tensor manifest that derives **exactly** the checkpoint's 1294 text tensors, tested against a snapshot of every shard header and not yet called at runtime. All of the model crate is CPU-only; its first device code lives beside it in `pegainfer-kernels` behind that crate's own `qwen4` feature — the QSA token indexer's block-scoring kernels, gated against a committed reference golden. Nothing runs — detection succeeds and `launch` refuses, because the text graph is slice B. Four facts this line must not get wrong, all measured rather than assumed: the embedding is **untied**; there is **no final-norm tensor** (the readout is the hyper-connection mixer); `ple_layer_ids: [2]` is **1-indexed**, so the n-gram block lives on `layers.1`; and `output_gate_type: "sigmoid"` is a **live** field selecting the GDN gated-norm activation — the inverse of the Qwen3.8-27B lesson, where the same field name is inert.
>
> **Last touched:** 2026-10

Tracked in #1103 (roadmap) with slices A #1104 / B #1127 / C #1128 and component bricks #1105 (QSA), #1106 (n-gram), #1107 (MoE), #1108 (hyper-connections).

## The freeze

| | |
| --- | --- |
| Repo / revision | `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Access | public, non-gated |
| Weight licence | **`license: other`** — not Apache-2.0, unlike the code this line ports from |
| Size | 131 shards, 360.0 GB, 1658 tensor keys |
| Identity | `model_type: qwen4_exp`, `architectures: [Qwen4ExpForConditionalGeneration]`, `text_config.model_type: qwen4_exp_text` |
| Reference impl | `transformers/models/qwen4_exp/modeling_qwen4_exp.py`; the reference first ships in **transformers 5.16.0** — the `5.8.0.dev0` in `config.json` is the checkpoint's own declared version, not a release that carries it (absent through 5.15.0) |

File hashes at that revision (HF blob sha1; `tokenizer.json` is LFS, so its sha256 is given too):

| File | Bytes | sha1 | sha256 (LFS) |
| --- | --- | --- | --- |
| `config.json` | 4,745 | `491017e9980e44ef01afa2c4782f5c7e169b9b26` | — |
| `tokenizer.json` | 12,809,320 | `9328ce9c41e80f6dd7bc2c66d8ae1fc93bf87440` | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `tokenizer_config.json` | 17,928 | `5de744b3fca2129d7186979ae47c06be33903243` | — |
| `chat_template.jinja` | 8,952 | `c0c686f9c38d70d179fb7b5f5aa7530bc913dda3` | — |
| `generation_config.json` | 202 | `023756cfadf88e5bf69eefeee3e172f38c448d64` | — |

The committed `config.json` (`pegainfer-qwen4/tests/frozen_config.json`) is the real file, not a transcription, and a test re-hashes it against `FROZEN_CONFIG_SHA256` so the fixture cannot drift from the revision it claims to be. The geometry and contract tests all parse it; only the identity tests build their own input.

## What landed

`pegainfer-qwen4`, enabled in the server by `--features qwen4`, **with no CUDA dependency at all** — so its tests need no device, no weights and no Triton-equipped build. Preserving that is the point of the crate's shape: when slice B adds the device side behind a crate feature of its own, `config`, `probe` and `manifest` move behind `#[cfg(any(<that feature>, test))]` the way `pegainfer-gemma4/src/lib.rs` does.

- **Probe** (`src/probe.rs`) — identity only: the outer `model_type`, `architectures` and the inner `text_config.model_type`, the same split as Qwen3.5's probe.
- **Config** (`src/config.rs`) — every other rule, behind one typed parse (`RawConfig` → `Config`), so detection and loading cannot disagree. The line's `ModelLine::probe` builds the whole `Config`, so `detect` refuses an unservable `qwen4_exp` config before the frontend starts. It follows the Gemma 4 pattern of pinning *structure rather than widths*. Pinned because an alternative would need a kernel or serving path this line does not have: the GDN head dims (they are the Triton AOT specialization keys), `head_dim` 256, the conv kernel ceiling, `rope_type: default` (no YaRN path), an even rotary width (the reference rotates in pairs, so an odd one would rotate one more dim than computed), `output_gate_type: sigmoid`, `mamba_ssm_dtype: float32`, `dtype: bfloat16` with no `quantization_config`, and untied embeddings. The official FP8 sibling and nvidia's NVFP4 keep `dtype: bfloat16` and differ from this checkpoint only by `quantization_config`, so that block is what refuses them; the dtype alone would not. The GDN and attention limits come from the kernels' shared wrappers (`pegainfer_kernels::ops`, behind the kernels' `qwen35` feature): this line's 36 `linear_attention` layers will reuse the Qwen3.5 decode kernels through them once slice B gives the line a device path, and `src/config.rs` mirrors the three GDN constants because the config layer must still test with no device. Deliberately *not* pinned: expert count, hidden size, layer count, head counts, `hc_count` — so a smaller `qwen4_exp` checkpoint passes the same rules, which is what the L20 small-model deployment case needs (`smaller_widths_still_pass`). `mrope_interleaved` and `mrope_section` are accepted rather than refused: a text-only sequence is their degenerate case in the reference itself, which expands one position per token across all three axes before interleaving. The decision is tracked in #1105. Beyond the pins, the config mirrors the rules the reference's own `validate_architecture` enforces — `hc_count > 1`, QSA's `indexer_kv_heads == 1` with the budget a whole multiple of the compress ratio, the rotary width fitting the index head, `rope_theta` present, and PLE confined to a **single** `linear_attention` layer — plus the two head-group integrality rules Qwen3.5's config enforces (`AttentionHeadDivisibility`, `LinearHeadDivisibility`). Multi-layer PLE is refused for now: the reference offsets each layer's prime run by its position in `ple_layer_ids` (`global_head_idx = L * ngram_heads + i`), and this crate derives one table. `norm_topk_prob` is **absent from `config.json`** and comes from the upstream default (`true`, and it decides whether the router renormalizes its top-k), so it is modelled with that default: a port that reads only `config.json` silently drops the router renormalization — this is #1068's mirror image. The other absent field, `seed`, only matters if the n-gram hash multipliers are recomputed instead of read from the checkpoint, so it is not modelled.
- **Manifest** (`src/manifest/`, test only until slice B's loader calls it) — the tensor contract, ported from `pegainfer-gemma4/src/manifest/`: `schema.rs` derives it from `Config` and `validate.rs` classifies headers against it. It derives **exactly** the frozen text set, 1658 keys − 333 `model.visual.*` − 31 `mtp.*` = **1294 tensors**, with `lm_head.weight` the only one outside `model.language_model.`. Beyond the untied head, it differs from Gemma 4's in rank-3 shapes (the depthwise conv and two fused expert stacks per layer, which carry no `.weight` suffix) and in a mandatory skip set, because all 333 vision tensors share shard 1 with 16 text tensors. `tests/frozen_tensors.json` (148 KB, written by `tools/accuracy/dump_qwen4_frozen_tensors.py`) holds every shard header of the frozen revision, and `the_contract_is_the_frozen_checkpoint` checks all 1658 tensors against the contract.
- **Registration** — the server binary hand-mirrors every line's probe identities in a hint table that cannot be derived away. With the feature off, `feature_gate_hint` names `--features qwen4` for both `qwen4_exp` and `qwen4_exp_text`; with it on, the hint stays silent. A test under the feature binds that row to this crate's `MODEL_TYPE` / `TEXT_MODEL_TYPE`, so the mirror cannot drift from the probe. The hint is **identity-only** — it cannot run a compiled-out probe — so it also fires for the quantized siblings and the tied/YaRN variants of `qwen4_exp`, which the config refuses once the feature is on; the hint names the feature to rebuild with, not a promise of service. `pegainfer-server/tests/detect_gate.rs` then drives the shipped binary against a model directory holding this crate's frozen `config.json`, both ways: the stock build must name the feature, the enabled build must detect the line and refuse to serve. That is the only coverage of `main`'s CLI → registry → hint → `launch` wiring, and neither outcome reaches CUDA — `launch` bails before the engine is built.

**1-D lengths are part of the contract** (259 vectors), because the shared loader takes them on trust (#1068), and here many carry meaning: `A_log[48]`, `dt_bias[48]`, `norm.weight[128]`, `q_norm[256]`, the indexer layernorms at `[128]`, `layer_multipliers[3]` and the two `[16]` n-gram partition vectors. In the Qwen3.5 GDN loader this line would reuse, `A_log` goes through `tensor_f32_cow`, which accepts BF16 or F32 but never checks the length, and on a single rank `dt_bias` goes through `load_tensor_1d`, which only checks that the byte count is even before reading the bytes as BF16.

## The QSA indexer's scoring half

`#1105` splits QSA into three gateable units. The middle one — the token indexer's block scores — is in `pegainfer-kernels/csrc/qwen4/` behind the kernels crate's own `qwen4` feature: the model crate above still has **no CUDA dependency at all**, and this is the line's first device code. The directory is named after the model line like `qwen35` and `glm52`, not after the checkpoint.

Two kernels against the frozen transformers-5.16.1 semantics. One pools every complete micro-block of `indexer_compress_ratio` tokens into a single key — a **float32** mean of the block's raw keys, cast back to bf16 exactly as the reference casts it, then `(1 + w)` RMSNorm and partial RoPE at the block's first token — and the other norms each of a query row's four heads at the row's own position and reduces `sum_h relu(q_h . k_b) / sqrt(indexer_head_dim)`. The pooled half is a separate pass because it does not depend on the query row at all: the reference rebuilds it per row from the same keys through the same bf16 rounding points, so one pass over the keys is the same numbers for every row that can see them. `ratio = 4` divides the 16-token KV page, so a micro-block never crosses one and the selection stage will need no block↔page mapping — but the pool pass does not use that, and the **block grid starts at the first visible token**, which this kernel requires to be token 0 (causal text over an un-evicted key cache; a visible list anchored elsewhere would move the grid, and nothing in the kernel can detect that, so such a caller has to hand it a shifted key array). The target is the portable subset: no TMA, no cluster, no wgmma. Both kernels are SIMT shuffle reductions rather than `mma.sync` bodies — one block's score is a 128-wide dot over 4 query heads, a shape `m16n8k16` cannot tile without padding 4 rows to 16.

```
cargo test --release -p pegainfer-kernels --features qwen4 \
  --test qwen4_qsa_indexer_score_parity -- --nocapture
```

The gate replays `test_data/qwen38fn-qsa-indexer-golden.safetensors` (PR #1150) and holds the fixture's recorded `format_version`, `transformers_version` and `config_sha256` against the frozen config — and derives its geometry from that config, cross-checked against the fixture's own tensor shapes — so a golden from another checkpoint or another semantics fails as a score mismatch instead of being skipped. It reads the fixture's **validity prefix** per row rather than comparing through the NaN padding the reference leaves, and proves the kernel writes exactly that prefix by checking the untouched tail still holds a sentinel. Six of seven cases land within 9.6e-7 of the reference and are held to a 2e-6 bound; the 2055-token case, the only one with positions past 63, keeps a 5e-3 bound until its 2.4e-3 residual is traced (its shape suggests one flipped bf16 rounding decision). In-range `rows`, and `visible_len` up to the row's own token, are contract: anything else traps on the device, and a poisoned key keeps its NaN score instead of reading as a valid zero. The kernel takes no budget: the reference scores every complete block and truncates only at selection. **Next: that selection** — top-k over the block scores plus the unconditional tail, which is a unit of its own and depends on the same golden, not on this kernel.

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

**102.4 GB, not the ~51 GB vLLM and SGLang quote** — that figure is the fp8 footprint, and this checkpoint ships BF16. Every host-RAM budget in the issue set was 2× under until it said which dtype it meant. The FP8 sibling converts this table to fp8 (`modules_to_convert`); it is a separate checkpoint, refused here by its `quantization_config`, and serving it would need freezing separately.

The derivation lands here rather than with its first consumer because #1106 (the n-gram brick) needs the geometry independently of any tensor contract, and because it is pure arithmetic over `config.json`. The prime search is a direct port of the reference's `_is_prime` / `_find_nth_prime_after` (trial division by odd numbers), and `nth_prime_after_matches_the_reference` pins its values.

That arithmetic is confirmed **by value**: the checkpoint stores the partition as two `I64[16]` vectors, `ngram_heads_vocab_sizes` and `ngram_heads_offsets`, both equal to the derived primes and offsets (20,000,003 through 20,000,171). The header snapshot carries them, and `the_ngram_partition_equals_the_stored_integer_vectors` compares them with the derivation.

## GDN decode reuse

The 36 `linear_attention` layers will run Qwen3.5's Gated DeltaNet decode kernels once slice B wires the text graph, because the geometry is the same: 16 key and 48 value heads at head dim 128, a 4-tap depthwise conv, and the same tensor names under `model.language_model.layers.N.`. The two host wrappers, `conv1d_decode_batch_into` and `gated_delta_rule_decode_batch_into`, live in `pegainfer-kernels/src/ops/qwen35.rs` behind the kernels' `qwen35` feature, which compiles `csrc/qwen35/{conv1d,gated_delta_rule}.cu`, and Qwen3.5 calls them through `pegainfer_kernels::ops`. Head counts are runtime arguments, so 48 value heads need no rebuild.

The three kernel limits (`GDN_AOT_KEY_HEAD_DIM`, `GDN_AOT_VALUE_HEAD_DIM`, `LINEAR_CONV_MAX_KERNEL_DIM`) are defined next to the wrappers. This line's `src/config.rs` keeps a CPU-only mirror of them, so its config tests need no device, until slice B adds the device dependency.

Still to verify is the reuse on this line's activations. The tensor contract derives every GDN shape from the frozen config and validates all 1658 headers, but nothing has run this line's GDN block against a reference yet. That is slice B's M0 exit (#1127), against the `transformers` recurrence, and it has to cover several decode steps from a nonzero state (so `(v - k·s)` is live), finite checks before any reduction, and the gated norm plus `out_proj` that complete the block (the norm's sigmoid entry point is #1129's).

## Traps

- **`output_gate_type` has three unrelated consumers.** On Qwen3.8-27B it is inert and must *not* be implemented (`docs/models/qwen35/support-qwen38.md`). Here it is live and selects the GDN gated-norm activation; the shared `csrc/shared/norm.cu` gated norm **became selectable in #1129** (a sigmoid entry point beside Qwen3.5's SiLU, merged before this line's detection PR), so what remains is this line's call site, which lands with the text graph. It is *also* the marker the frontend's `reasoning_effort` layer arms on. Read the field's consumer, never its name.
- **Two RMSNorm conventions in one layer.** `(1+w)` for `q_norm`/`k_norm`, the indexer layernorms, `hc_norm` and the PLE norms; plain `w` only for the GDN gated norm. The four 10240-wide ones are additionally **grouped by 2560** — likely a reshape of the existing batched `(1+w)` kernel rather than a new kernel, but unverified.
- **The PLE conv is dilated** (kernel 4, dilation `ngram_size` = 3, state length 9) even though its weight tensor `[10240, 1, 4]` is shaped exactly like the GDN conv's. The in-tree `conv1d.cu` is non-dilated and caps its state at four taps, so it cannot serve this.
- **Attention writes back at `hidden_size`.** `o_proj [2560, 6144]` returns to the 2560-wide hidden size, and `attn_hyper_connection.block_inject_weight [4, 10240]` carries it into the 10240-wide hyper-connection stream.
- **GQA group 12 is not a compiled decode group.** `SUPPORTED_GQA_GROUP_SIZES` is `[1, 2, 3, 4, 6, 8]`, so the FlashInfer batch decode path cannot serve this attention as it stands. QSA is sparse and needs kernels of its own, so whether its decode is captured in a graph is slice B's decision, made with those kernels.
- **The QSA gate's bound is measured per case, and its floor was measured too.** Cases 0–5 sit at ≤9.6e-7 and carry a 2e-6 bound; the long case's 2.4e-3 residual is not traced yet; its shape suggests one flipped bf16 rounding decision, and the reference moves by that size across environments (re-deriving its case 2 on x86 from the fixture's own inputs moves one of its 496 scores by 2.2e-3), so it keeps 5e-3 until traced. What the bound must catch was measured on this fixture: keeping the pooled mean in f32 moves scores by 1.55e-2, 100x `rms_eps` by 1.25e-2, dropping `rms_eps` or pooling by sum by 3.3e-3, the pooled key's RoPE by 2.7, and a post-sum relu by 0.83 — every fault sits two to six orders beyond the tight bound. Architecture variance does not loosen it: the gate prints identical numbers on sm_89 and sm_120, and the kernel reads the shared attention cos/sin table (built by `pegainfer_core::rope` with `frequency_dim` equal to the 64 rotated dims, as the reference spaces its frequencies) rather than computing trig, so there is nothing device-specific left to vary. `#1105`'s selection stage must still treat a block tied with the last kept one at the cut as an equally valid pick, because the long case's noise can flip that order.
- **Do not use `load_shard_info_fixed`.** It is a Qwen3.5-specific shard-*filename* workaround living in shared `pegainfer-core`; this checkpoint uses standard `model-0000N-of-00131` naming and wants the plain loader.

## Still open

| Criterion | Blocked on |
| --- | --- |
| Tensor manifest enforced before the first H2D | slice B's loader, which must classify the headers it maps before any upload |
| GDN decode reuse, executed | slice B: the wrappers are shared in `pegainfer-kernels::ops` and the contract pins their geometry and tensor names, but nothing has run this line's GDN block against the reference. The model-graph fixture that does so is slice B's M0 exit (#1127); the GDN prefill path and the gated norm composition land with it. |
| Bookend gate (embedding → mixer → lm_head → argmax) | 1 GPU, three shards (1 + 130 + 131, 5.29 GB, ~2.6 GB of weights on device) and a CPU f64 reference for the mixer. No layer weights needed. |
| HF golden fixtures (logits + intermediate probes) | the fleet — 360 GB, plus ≥102.4 GB host RAM if the n-gram table is materialized |
| Tokenizer / chat-template ID fixtures | transformers **5.16.0+** — the checkpoint's own tokenizer is `Qwen2Tokenizer`, which is older, but fixtures are compared against the reference implementation; nothing in the tree commits token-ID fixtures today |

Note that `pegainfer-server` cannot be compiled on a CPU-only box, so the registration above is verifiable only where a CUDA build works.

A local box here is **sm_120**, which is neither support-matrix row (H200 sm_90, L20 sm_89) — anything proven there establishes numerics and must never be recorded as a platform result.
