# Qwen4 bring-up

> **TL;DR:** `Qwen/Qwen3.8-Flash-Next`, the first checkpoint of the Qwen4 architecture preview, is a **new line** (`qwen4_exp`), not a Qwen3.5 variant — its `model_type` differs, so the Qwen3.5 probe refuses it by construction. Frozen at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Landed so far: an identity probe and one typed config validation that pins *structure rather than widths* and derives what `config.json` leaves implicit, plus a tensor manifest that derives **exactly** the checkpoint's 1294 text tensors, tested against a snapshot of every shard header and not yet called at runtime. All of the model crate is CPU-only; its first device code lives beside it in `pegainfer-kernels` behind that crate's own `qwen4` feature — the QSA token indexer's block-scoring and block-selection kernels, gated against a committed reference golden, plus the component kernels extracted from the five brick PRs #1151 to #1155 (hyper-connection mix, MoE routing and shared-expert combine, the PLE block and its n-gram hasher). Behind the model crate's own `qwen4` feature, layer 0 (`linear_attention`), layer 1 (the n-gram layer, its 102.4 GB table read row by row from the mapped shards on the host) and layer 3 (the first attention layer, dense up to the QSA budget) now run in prefill on the checkpoint's weights and decode token by token on their carried state (conv and recurrent state for GDN, the paged K/V for attention, the last tokens and block inputs for the n-gram block); all of it lands within the reference's own bf16 noise floor, and `model::Model` chains any loaded subset of layers between the embedding stream and the readout (mixer and untied head). The model gate below holds every layer on the reference's own input through all 48 layers on one GH200, one layer resident at a time, and the engine's own 48 layer chain picks the reference's argmax except at near ties. Decode runs several sequences as one batch, one slot each, and the gate holds every row against its own reference. `launch` still refuses: the step and serving across four cards are next. Four facts this line must not get wrong, all measured rather than assumed: the embedding is **untied**; there is **no final-norm tensor** (the readout is the hyper-connection mixer); `ple_layer_ids: [2]` is **1-indexed**, so the n-gram block lives on `layers.1`; and `output_gate_type: "sigmoid"` is a **live** field selecting the GDN gated-norm activation — the inverse of the Qwen3.8-27B lesson, where the same field name is inert.
>
> **Last touched:** 2026-11

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

`pegainfer-qwen4`, enabled in the server by `--features qwen4`. Without that feature the crate has no CUDA dependency, so its probe, config and manifest tests need no device, no weights and no Triton build. The feature adds the device half (`weights`, `layer`, `moe`) and pulls in the kernels crate with its `qwen4` and `qwen35` features, so a build with it needs Python and Triton like the Qwen3.5 line.

- **Probe** (`src/probe.rs`) — identity only: the outer `model_type`, `architectures` and the inner `text_config.model_type`, the same split as Qwen3.5's probe.
- **Config** (`src/config.rs`) — every other rule, behind one typed parse (`RawConfig` → `Config`), so detection and loading cannot disagree. The line's `ModelLine::probe` builds the whole `Config`, so `detect` refuses an unservable `qwen4_exp` config before the frontend starts. It follows the Gemma 4 pattern of pinning *structure rather than widths*. Pinned because an alternative would need a kernel or serving path this line does not have: the GDN head dims (they are the Triton AOT specialization keys), `head_dim` 256, the conv kernel ceiling, `rope_type: default` (no YaRN path), an even rotary width (the reference rotates in pairs, so an odd one would rotate one more dim than computed), `output_gate_type: sigmoid`, `mamba_ssm_dtype: float32`, `dtype: bfloat16` with no `quantization_config`, and untied embeddings. The official FP8 sibling and nvidia's NVFP4 keep `dtype: bfloat16` and differ from this checkpoint only by `quantization_config`, so that block is what refuses them; the dtype alone would not. The GDN and attention limits come from the kernels' shared wrappers (`pegainfer_kernels::ops`, behind the kernels' `qwen35` feature): this line's 36 `linear_attention` layers will reuse the Qwen3.5 decode kernels through them once slice B gives the line a device path, and `src/config.rs` mirrors the three GDN constants because the config layer must still test with no device. Deliberately *not* pinned: expert count, hidden size, layer count, head counts, `hc_count` — so a smaller `qwen4_exp` checkpoint passes the same rules, which is what the L20 small-model deployment case needs (`smaller_widths_still_pass`). `mrope_interleaved` and `mrope_section` are accepted rather than refused: a text-only sequence is their degenerate case in the reference itself, which expands one position per token across all three axes before interleaving. The decision is tracked in #1105. Beyond the pins, the config mirrors the rules the reference's own `validate_architecture` enforces — `hc_count > 1`, QSA's `indexer_kv_heads == 1` with the budget a whole multiple of the compress ratio, the rotary width fitting the index head, `rope_theta` present, and PLE confined to a **single** `linear_attention` layer — plus the two head-group integrality rules Qwen3.5's config enforces (`AttentionHeadDivisibility`, `LinearHeadDivisibility`). Multi-layer PLE is refused for now: the reference offsets each layer's prime run by its position in `ple_layer_ids` (`global_head_idx = L * ngram_heads + i`), and this crate derives one table. `norm_topk_prob` is **absent from `config.json`** and comes from the upstream default (`true`, and it decides whether the router renormalizes its top-k), so it is modelled with that default: a port that reads only `config.json` silently drops the router renormalization — this is #1068's mirror image. The other absent field, `seed`, only matters if the n-gram hash multipliers are recomputed instead of read from the checkpoint, so it is not modelled.
- **Manifest** (`src/manifest/`, test only until slice B's loader calls it) — the tensor contract, ported from `pegainfer-gemma4/src/manifest/`: `schema.rs` derives it from `Config` and `validate.rs` classifies headers against it. It derives **exactly** the frozen text set, 1658 keys − 333 `model.visual.*` − 31 `mtp.*` = **1294 tensors**, with `lm_head.weight` the only one outside `model.language_model.`. Beyond the untied head, it differs from Gemma 4's in rank-3 shapes (the depthwise conv and two fused expert stacks per layer, which carry no `.weight` suffix) and in a mandatory skip set, because all 333 vision tensors share shard 1 with 16 text tensors. `tests/frozen_tensors.json` (148 KB, written by `tools/accuracy/dump_qwen4_frozen_tensors.py`) holds every shard header of the frozen revision, and `the_contract_is_the_frozen_checkpoint` checks all 1658 tensors against the contract.
- **Registration** — the server binary hand-mirrors every line's probe identities in a hint table that cannot be derived away. With the feature off, `feature_gate_hint` names `--features qwen4` for both `qwen4_exp` and `qwen4_exp_text`; with it on, the hint stays silent. A test under the feature binds that row to this crate's `MODEL_TYPE` / `TEXT_MODEL_TYPE`, so the mirror cannot drift from the probe. The hint is **identity-only** — it cannot run a compiled-out probe — so it also fires for the quantized siblings and the tied/YaRN variants of `qwen4_exp`, which the config refuses once the feature is on; the hint names the feature to rebuild with, not a promise of service. `pegainfer-server/tests/detect_gate.rs` then drives the shipped binary against a model directory holding this crate's frozen `config.json`, both ways: the stock build must name the feature, the enabled build must detect the line and refuse to serve. That is the only coverage of `main`'s CLI → registry → hint → `launch` wiring, and neither outcome reaches CUDA — `launch` bails before the engine is built.

- **Component kernels from the five bricks** (#1151 to #1155, committed under their author's name) — `csrc/qwen4/qwen4_hc_block.cu` behind `hc_mix_into` and `hc_scatter_add_into` (the gated residual's mix pass and the scatter back into the 10240-wide stream), `qwen4_moe_route.cu` behind `qwen4_moe_route_into` (fp32 softmax, canonical top-10, `norm_topk_prob`), `qwen4_moe_shared_branch_into` and `qwen4_moe_combine_into`, `qwen4_ple_block.cu` behind `ple_block_into` (the per-branch gate with its double rounding and signed sqrt, then the dilated causal conv) with the host-side hasher in `src/ngram.rs`, and the `qwen4_qsa_norm_rope.cuh` helpers the indexer now shares. Every kernel header spells out the reference's rounding points, which is the part of those PRs worth keeping. Left behind: the goldens (66 MiB, all generated from `torch.manual_seed` and `randn` weights, so they pin the port against itself rather than against the checkpoint), their replay readers, the test-only knobs that had reached production signatures (`combine_scale`, `probs_bf16`, the PLE gate variants) and the prefill-only QSA attention kernel. The five `tools/accuracy/dump_qwen38fn_*_golden.py` scripts stay as the component oracles; the layer golden below is what the engine is held against. The GDN brick (#1152) contributed a finding rather than code: Qwen3.5's conv and chunkwise delta-rule kernels match the reference at this line's geometry, so this line calls the qwen35 production entries rather than a split copy.

**1-D lengths are part of the contract** (259 vectors), because the shared loader takes them on trust (#1068), and here many carry meaning: `A_log[48]`, `dt_bias[48]`, `norm.weight[128]`, `q_norm[256]`, the indexer layernorms at `[128]`, `layer_multipliers[3]` and the two `[16]` n-gram partition vectors. In the Qwen3.5 GDN loader this line would reuse, `A_log` goes through `tensor_f32_cow`, which accepts BF16 or F32 but never checks the length, and on a single rank `dt_bias` goes through `load_tensor_1d`, which only checks that the byte count is even before reading the bytes as BF16.

## The QSA indexer's scoring half

`#1105` splits QSA into three gateable units. The middle one — the token indexer's block scores — is in `pegainfer-kernels/csrc/qwen4/` behind the kernels crate's own `qwen4` feature: the model crate above still has **no CUDA dependency at all**, and this is the line's first device code. The directory is named after the model line like `qwen35` and `glm52`, not after the checkpoint.

Two kernels against the frozen transformers-5.16.1 semantics. One pools every complete micro-block of `indexer_compress_ratio` tokens into a single key — a **float32** mean of the block's raw keys, cast back to bf16 exactly as the reference casts it, then `(1 + w)` RMSNorm and partial RoPE at the block's first token — and the other norms each of a query row's four heads at the row's own position and reduces `sum_h relu(q_h . k_b) / sqrt(indexer_head_dim)`. The pooled half is a separate pass because it does not depend on the query row at all: the reference rebuilds it per row from the same keys through the same bf16 rounding points, so one pass over the keys is the same numbers for every row that can see them. `ratio = 4` divides the 16-token KV page, so a micro-block never crosses one and the selection stage needs no block↔page mapping — but the pool pass does not use that, and the **block grid starts at the first visible token**, which this kernel requires to be token 0 (causal text over an un-evicted key cache; a visible list anchored elsewhere would move the grid, and nothing in the kernel can detect that, so such a caller has to hand it a shifted key array). The target is the portable subset: no TMA, no cluster, no wgmma. Both kernels are SIMT shuffle reductions rather than `mma.sync` bodies — one block's score is a 128-wide dot over 4 query heads, a shape `m16n8k16` cannot tile without padding 4 rows to 16.

```
cargo test --release -p pegainfer-kernels --features qwen4 \
  --test qwen4_qsa_indexer_score_parity -- --nocapture
```

A decode row's blocks are independent, so the score pass splits each row across `grid.x` CTAs to target four resident CTAs per SM, taking the device's SM count from the caller; on an L20 (sm_89), the 2055-token six-row shape moved from 42.7 to 8.3 µs p50, the 512-token, 300-row shape from 12.6 to 11.0 µs, and the 512-row prefill held within 2%.

The gate replays `test_data/qwen38fn-qsa-indexer-golden.safetensors` (PR #1150) and holds the fixture's recorded `format_version`, `transformers_version` and `config_sha256` against the frozen config — and derives its geometry from that config, cross-checked against the fixture's own tensor shapes — so a golden from another checkpoint or another semantics fails as a score mismatch instead of being skipped. It reads the fixture's **validity prefix** per row rather than comparing through the NaN padding the reference leaves, and proves the kernel writes exactly that prefix by checking the untouched tail still holds a sentinel. Six of seven cases land within 9.6e-7 of the reference and are held to a 2e-6 bound; the 2055-token case, the only one with positions past 63, keeps a 5e-3 bound until its 2.4e-3 residual is traced (its shape suggests one flipped bf16 rounding decision). In-range `rows`, and `visible_len` up to the row's own token, are contract: anything else traps on the device, and a poisoned key keeps its NaN score instead of reading as a valid zero. The kernel takes no budget: the reference scores every complete block and truncates only at selection.

## The QSA indexer's selection half

`csrc/qwen4/qwen4_qsa_indexer_select.cu` (`qsa_indexer_select_into`) turns the block scores into the token-level mask the sparse attention reads; the contract (tie rule, unconditional tail, trap, grid) is on the kernel header, and the wrapper doc carries the Rust-side buffer rules.

The select is an exact 8-bit radix select over the row's f32 keys: one 256k-token row at budget 2048 takes 298 µs on an L20, and the all-kept fast path takes 1.4 µs.

Two boundaries the code cannot state. A row's visible list starts at token 0 (the scoring pass's grid), so a prefix-evicted visible list is out of contract and needs a shifted key array on both stages. And an accuracy gate must not require mask equality with the reference at a tied cut: `torch.topk` leaves that pick open, so the reference's choice can differ from the kernel's canonical one and still be valid.

## Grouped `(1+w)` RMSNorm

On the 26-SM sm_120 card, the production 2560-wide group moved from 56.0 to 41.6 µs p50 at 512 tokens (128 tokens: 17.2 to 14.9 µs).

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

## Layer 0 on the checkpoint's weights

The reference is `tools/accuracy/dump_qwen4_layer_golden.py`: it reads the named layers' tensors and the bookends (the token embedding, the top-level mixer, the head) from the shards, runs the `Qwen4ExpTextDecoderLayer`s in order between the embedding stream and the readout over a fixed 32 token prompt once on the accelerator and once on the CPU, and records every layer's stage boundaries under a `layer{N}_` prefix (the two hyper-connection mixes, the four GDN projections, the delta rule output, the gated norm, the attention output, the stream after the scatter, the router logits and picks, the shared and routed expert outputs, the block output, the stream out). The accelerator versus CPU spread of the reference is the floor nothing in bf16 can be held below.

The engine side is `weights::load_layer` (maps only the shards that hold the layer, checks that layer's slice of the tensor contract, uploads through the shared staged loader with the new `expert_stack` record for the fused `[512, 1280, 2560]` and `[512, 2560, 640]` stacks), `layer::layer_prefill` (hyper-connection mix, the GDN block on the qwen35 prefill wrappers that now live in `pegainfer_kernels::ops`, the scatter, the second mix, the MoE block, the second scatter) and `moe::moe_prefill` (router, a route metadata kernel that orders the picks by expert, one cuBLAS GEMM pair per expert over its slot range through `gemm_rows_span_at_into_checked`, the fused SiLU, a weighted combine kernel, the shared branch and the block sum). The routed experts are a correctness first design: about a thousand launches and one host sync per layer at 2k tokens. Decode wants the pointer batched form and prefill a grouped GEMM; both are follow-ups.

The gate is `model::golden_gate`, run with `PEGAINFER_QWEN4_MODEL_PATH` and `PEGAINFER_QWEN4_LAYER_GOLDEN` set. It loads the layers the golden names with the bookends around them, requires the embedding stream to match exactly, prints the max and mean difference per stage of every layer, holds five stages per layer (the attention output, the stream after it, the router logits, the block output, the stream out) and the logits to four times the reference's own floor (the largest it shows on that stage over the sequence, as [Several sequences in one decode step](#several-sequences-in-one-decode-step) measures), and requires every token's picks to be a top 10 of the engine's own logits. Picks that differ from the reference's are printed with the reference's own gap in bf16 steps; with the logits inside the floor, such a swap can only sit where that gap is inside the floor too.

```
python tools/accuracy/dump_qwen4_layer_golden.py --model /data/models/Qwen3.8-Flash-Next --layers 0 --out golden/qwen4-layers0.safetensors
PEGAINFER_QWEN4_MODEL_PATH=/data/models/Qwen3.8-Flash-Next PEGAINFER_QWEN4_LAYER_GOLDEN=golden/qwen4-layers0.safetensors \
  cargo test --release -p pegainfer-qwen4 --features qwen4 --lib model::golden_gate -- --nocapture
```

On one sm_89 card against a reference recorded on the same card:

| Stage | max difference | reference's own floor |
| --- | --- | --- |
| `attn_out` | 7.8e-3 | 3.9e-3 |
| `stream_after_attn` | 2.0e-3 | 9.8e-4 |
| `router_logits` | 6.2e-2 | 6.2e-2 |
| `moe_out` | 7.8e-3 | 7.8e-3 |
| `stream_out` | 3.9e-3 | 5.9e-3 |
| `gdn_state` (the recurrent state the prompt leaves) | 1.0e-2 | 7.5e-3 |

One of the 32 tokens swaps one expert; the reference's logits for the two are one bf16 step apart. The same script on a GH200 gives floors of 1.2e-2 on `attn_out` and 9.3e-3 on `stream_out`, and the reference itself routes one token differently across the two cards (167 against 166 experts hit), so the tie is in the data.

## Layer 3 on the checkpoint's weights

Layer 3 is the first `full_attention` layer. Its QSA keeps every causal key while the visible prefix fits the indexer budget (2051 tokens on this checkpoint), so up to that length the layer is dense gated attention in the Qwen3.5 head shape: 24 query and 2 key/value heads at head dim 256, `[q | gate]` fused per head in `q_proj`, `(1 + w)` q/k norm, partial RoPE over 64 dims with the frequencies spaced over those 64, a sigmoid output gate before `o_proj`. The same reference script records the attention branch (the three projections, the normed q and k, the indexer's mask, the gated attention output) through the eager path with an explicit causal mask; transformers 5.16.1 returns that mask additive, zero where a key is kept, and the gate checks it is zero over the whole causal triangle before it treats dense attention as the reference. The commands are the layer 0 ones with `--layers 3`.

The engine side runs the layer through `pegainfer_kernels::ops::attention_prefill_hd256_into`: the Qwen3.5 hd256 prefill (the norm and RoPE with the paged K/V write, the FlashInfer causal prefill, the output gate), lifted out of `pegainfer-qwen35` into one checked entry that both lines call. K/V go into a one layer paged scratch for the prompt. FlashInfer's prefill takes the GQA group of 12 at run time; only the decode dispatch has a compiled group list. `weights::load_layer` reads either layer kind into a `DecoderLayer`, with the n-gram block on the one layer that carries it.

On the same sm_89 card, 32 tokens:

| Stage | max difference | reference's own floor |
| --- | --- | --- |
| `attn_q_proj`, `attn_k_proj`, `attn_v_proj` | 0 | |
| `attn_gated` (`o_proj`'s input) | 1.6e-2, one bf16 step at its magnitude | |
| `attn_out` | 7.8e-3 | 1.2e-2 |
| `stream_after_attn` | 1.6e-2 | 1.6e-2 |
| `router_logits` | 3.1e-2 | 3.9e-2 |
| `moe_out` | 3.0e-2 | 3.0e-2 |
| `stream_out` | 2.6e-2 | 2.9e-2 |

Five of the 32 tokens pick a different expert set; the reference's own logits for each swapped pair are 0 to 3 bf16 steps apart, inside its own accelerator versus CPU spread on the logits. That is why the gate holds the logits to the floor and the picks to the engine's own logits rather than to a fixed tie width: layer 0's one swap sat one step apart, and this layer's noisier input (its `stream_after_attn` floor is sixteen times layer 0's) moves the gap to three.

Decode takes FlashInfer's prefill kernel too. Its batch decode dispatches a compiled GQA group list that lacks this line's 12, and its prefill kernel takes the group at run time, so a decode step is `attention_decode_via_prefill_hd256_into`: the q/k norm and partial RoPE at each request's position, the token's K/V written into its page, the causal prefill with one query row per request over that request's pages, then the sigmoid gate. That is the route Qwen3.5 takes for the groups its decode kernels do not compile. The reference with `--decode 8` runs the layer on a `DynamicCache` (the sparse attention layer maps to its indexed cache layer) with the mask and the position tables covering every key so far. Max over the 8 steps after a 24 token prefill on the same sm_89 card:

| Stage | max difference | reference's own floor |
| --- | --- | --- |
| `attn_q_proj`, `attn_k_proj`, `attn_v_proj`, normed q and k off the rotated dims | 0 (one step 3.9e-3) | |
| `attn_gated` | 7.8e-3 | |
| `attn_out` | 3.9e-3 | 7.8e-3 |
| `stream_after_attn` | 7.8e-3 | 1.2e-2 |
| `router_logits` | 2.3e-2 | 2.3e-2 |
| `moe_out` | 1.1e-2 | 1.5e-2 |
| `stream_out` | 1.4e-2 | 1.4e-2 |

One of the 8 steps picks a different expert set, with the reference's logits for the pair inside its own spread.

## The model around the layers

`model::Model` is what surrounds the layers: `embed` repeats each token's embedding `hc_count` times into the stream (one `embedding_batch` gather with every id repeated, since `[rows, hc_count × hidden]` and `[hc_count × rows, hidden]` are the same bytes), any subset of the decoder layers in checkpoint order, and `readout`, the top-level mixer (`hc_mix_into` with no injection head, there is no final norm) followed by the untied head. `Model::prefill` walks one prompt through the loaded layers in its slot of a `ModelState` (one `LayerState` per layer, holding the mixer's state, plus the n-gram block's on layer 1, for every slot), `Model::decode` one token in each of a batch of slots, and both return every layer's trace with the mixed row and the logits. The subset is what one card can hold against the reference; serving loads all 48.

The gate (`model::golden_gate`, the one gate for this line) holds the embedding stream exactly and every loaded layer stage by stage on the reference's own input for it: each pass feeds a layer the stream the reference fed it (the embedding stream for the first, the previous loaded layer's output after that), and the dump script feeds its CPU pass the same way, so every floor is that layer's own. The head is held on the reference's last stream. The engine's own chain through the loaded layers runs beside them (`Model::prefill` and `Model::decode` in the resident driver), and its logits and argmax agreement are printed with the reference's own gap where they disagree; they are not held, for the reason the next section measures. The chain the box can run is layers 0 and 3 with the bookends, a 24 token prefill then 8 decode steps:

```
python tools/accuracy/dump_qwen4_layer_golden.py --model /data/models/Qwen3.8-Flash-Next --layers 0,3 --decode 8 --out golden/qwen4-layers03-decode8.safetensors
```

On the same sm_89 card:

| Pass | `layer3_stream_out` on the reference's input | head on the reference's stream | the engine's own chain: `logits`, argmax |
| --- | --- | --- | --- |
| prefill, 24 tokens | 1.5e-3 on the 22 tokens whose expert sets agree (floor 8.1e-3) | 0 (floor 3.1e-2) | 2.1e-1, 23 of 24 (the odd one a one step tie) |
| 8 decode steps, max | 9.8e-4 (floors 2.4e-4 to 7.5e-3; step 7 picks another expert set) | 1.6e-2 (floors 9.8e-4 to 1.6e-2) | 2.8e-1, 7 of 8 (the odd one an exact tie) |

The single layer goldens carry the bookends too. Layer 0 alone: the head on the reference's stream matches it exactly and the engine's own readout lands within 4.7e-2 (argmax 30 of 32, two ties). Layer 3 alone: 3.1e-2 and 2.2e-1 (31 of 32).

## All 48 layers on one card

`PEGAINFER_QWEN4_GATE_STREAMED` turns the gate layer major: the bookends stay resident, one layer is loaded at a time, every pass goes through it twice (on the reference's input, held, and on the engine's own chain), and the layer is dropped before the next loads. A layer's pass reads only its own state and the previous layer's output for that pass, so the numbers are the resident forward's; the box runs every golden through both drivers and they print the same line for every stage. The dump script records the same way, one layer built and freed at a time with the n-gram table read from its shards, so the reference's 48 layers fit one card too:

```
python tools/accuracy/dump_qwen4_layer_golden.py --model /data/models/Qwen3.8-Flash-Next --layers all --decode 8 --out golden/qwen4-all-decode8.safetensors
```

On one GH200 that golden (2.2 GiB, 11647 tensors) records in under five minutes and the streamed gate takes four.

Each layer reads the reference's input because a chain compounds. Held on its own chain, a layer's distance from the reference grows with depth, and so does the reference's own accelerator versus CPU spread it is held against, so every stage of every pass is one draw of compounded rounding. Chained, the engine sat at a median 0.83 of the floor over 2403 held stages through layer 46 (p99 2.0), and one stage still landed at 5.3 times it (layer 46's router logits in the first decode step). At that depth the chained floors also come within an order of magnitude of the signal (layer 44's `moe_out` floor 3.0e-2 against a largest value of 0.26 in the first decode step), so a multiple of them holds almost nothing. Fed the reference's input, a layer's distance stays its own and its floor stays small.

A token that picks another expert set than the reference runs other experts, so its routed output has nothing to match: `moe_out` and `stream_out` are held on the tokens whose sets agree. The flip itself is bounded by the router logits hold, since two experts that trade places differ in the reference's logits by at most twice the engine's distance from it. Before this rule the chain of layers 0, 1 and 3 stopped on such a flip at layer 3's decode step 5, with the router logits at the reference's own floor.

All 48 layers, a 24 token prefill then 8 decode steps, each held stage of each pass over the floor it is held to (the largest the reference shows on that stage over the sequence, which the next section explains):

| Stage | held | median | p99 | max |
| --- | --- | --- | --- | --- |
| `ple_out` | 9 | 0 | 0 | 0 |
| `attn_out` | 432 | 0.32 | 1.0 | 1.3 |
| `stream_after_attn` | 432 | 0.06 | 0.50 | 1.0 |
| `router_logits` | 432 | 0.50 | 1.0 | 1.0 |
| `moe_out` | 432 | 0.05 | 0.58 | 1.0 |
| `stream_out` | 432 | 0.07 | 0.50 | 1.0 |
| `gdn_state` | 324 | 0.45 | 1.1 | 1.1 |

48 of the 1536 token routings pick a different expert set. The head on the reference's last stream lands within 6.3e-2 in prefill (floor 1.25e-1) and matches bit for bit in all 8 decode steps, argmax 24 of 24 and 8 of 8. The engine's own chain through all 48 layers lands within 1.7 of the reference's logits in prefill and 1.8 in decode, at logit magnitudes up to 22, and picks the reference's argmax on 21 of 24 prompt tokens and 7 of 8 steps; each disagreement sits at a tie or one to two bf16 steps in the reference's own logits. The reference's own CPU chain sits up to 2.2 from its accelerator chain on the same logits.

## Several sequences in one decode step

A `ModelState` holds several sequences, one per slot. Each attention layer keeps every slot's K/V in one paged pool (`layer::KvPool`, slot `s` owning the pages from `s × ⌈capacity / 16⌉`), so one plan covers a batch; the GDN conv and recurrent states and the n-gram block's carried tokens and rows are kept per slot. `Model::prefill` runs one prompt in its slot. `Model::decode` takes one token for each of a batch of slots and runs them as the rows of one forward: the hyper-connection mixes, every projection, the MoE block and the readout work row by row, the GDN decode kernels take one state pointer per row, the n-gram block gathers and projects the batch's rows together and runs its block per slot on that slot's carried rows, and attention goes through `attention_decode_via_prefill_hd256_into` with one query row per slot. One sequence per step takes the same path: on every golden above, every stage prints the number it printed before this path existed.

The gate takes several goldens joined like `PATH`. Each is one sequence in its own slot: the prompts are prefilled one at a time, and each decode step stacks the reference's input rows of every sequence that still has a step into one batch and holds each row against its own golden. Two shorter sequences come from the same prompt, 15 tokens then 6 steps (the decode crosses the first page boundary) and 1 token then 8 steps:

```
python tools/accuracy/dump_qwen4_layer_golden.py --model /data/models/Qwen3.8-Flash-Next --layers all --tokens 21 --decode 6 --out golden/qwen4-all-t21-decode6.safetensors
python tools/accuracy/dump_qwen4_layer_golden.py --model /data/models/Qwen3.8-Flash-Next --layers all --tokens 9 --decode 8 --out golden/qwen4-all-t9-decode8.safetensors
PEGAINFER_QWEN4_GATE_STREAMED=1 PEGAINFER_QWEN4_MODEL_PATH=/data/models/Qwen3.8-Flash-Next \
  PEGAINFER_QWEN4_LAYER_GOLDEN=golden/qwen4-all-decode8.safetensors:golden/qwen4-all-t21-decode6.safetensors:golden/qwen4-all-t9-decode8.safetensors \
  cargo test --release --locked -p pegainfer-qwen4 --features qwen4 --lib model::golden_gate -- --nocapture --test-threads=1
```

Each held stage is held to four times the largest floor the reference shows on it over the sequence, prefill and every decode step, and a bf16 stage's floor in a pass is at least one bf16 step of its largest value. Held pass by pass against that pass's floor alone, two of these three sequences fail with one sequence per step: the 15 token one's layer 3 attention output in decode step 2 sits at 1.4e-2 against a line of 8.3e-3, where the same layer's prefill floor is 1.2e-2, and the 1 token one's layer 31 attention output in decode step 6 at 9.3e-3 against 7.8e-3. A pass's spread is one draw, and a draw can come out near nothing: on the 1 token prompt the reference's two devices agree bit for bit on layer 3's router logits, and the recurrent state's spread shrinks with the decay (layer 0's sat 2.4e-2 apart at decode step 3 and 3.4e-3 apart at step 5) while the distance it is held against carries every earlier step. Holding every pass to the sequence's largest floor is the same as holding our largest distance over the sequence to four times the reference's largest.

A row decoded in a batch sits a bf16 step or two from where it sits alone. The reference decodes one row at a time, so with one sequence per step the engine's GEMMs run the reference's shapes and many stages match it bit for bit; a batch of three runs other shapes, for which cuBLAS picks other kernels. A row that read another slot's state would sit far outside that. Each of three mutations stops the batch of three through layers 0 and 3 at its first decode step on the sm_89 card:

| Mutation | stops at | ours | line |
| --- | --- | --- | --- |
| the first two slots trade K/V pages | `decode0_layer3_attn_out` | 1.1e-1 | 1.6e-2 |
| the GDN state pointers in reverse order | `decode0_layer0_attn_out` | 5.5e-1 | 3.1e-2 |
| every new token one position early | `decode0_layer3_attn_out` | 4.2e-2 | 1.6e-2 |

The batch of three through layers 0 and 3 on that card and through layers 0, 1 and 3 on a GH200 passes in both drivers, which print the same line for every stage. All 48 layers, the three sequences as one batch on one GH200 (226 s streamed; 97 of the 2976 token routings pick a different expert set):

| Stage | held | median | p99 | max |
| --- | --- | --- | --- | --- |
| `ple_out` | 25 | 0 | 0 | 0 |
| `attn_out` | 1200 | 0.40 | 1.7 | 2.4 |
| `stream_after_attn` | 1200 | 0.06 | 0.50 | 1.0 |
| `router_logits` | 1200 | 0.50 | 1.0 | 1.0 |
| `moe_out` | 1200 | 0.06 | 0.80 | 2.0 |
| `stream_out` | 1200 | 0.08 | 0.50 | 1.0 |
| `gdn_state` | 900 | 0.48 | 1.3 | 1.9 |

| Sequence | head on the reference's stream, prefill and decode | the engine's own chain, prefill and decode | its argmax, prefill and decode |
| --- | --- | --- | --- |
| 24 tokens, 8 steps | 6.3e-2, 0 | 1.7, 1.8 | 21 of 24, 7 of 8 |
| 15 tokens, 6 steps | 0, 0 | 1.5, 1.6 | 13 of 15, 5 of 6 |
| 1 token, 8 steps | 0, 0 | 7.8e-2, 1.1 | 1 of 1, 6 of 8 |

The head matches the reference's argmax everywhere. Every disagreement of the engine's own chain sits at a tie or one to two bf16 steps in the reference's own logits; the 15 token sequence alone picks 6 of 6 steps, and in the batch its step 4 lands one step the other way.

## Layer 1 and the n-gram table

Layer 1 is a `linear_attention` layer with the n-gram block in front of its attention mix: the stream gets the block's output added, and the mix, the scatter and everything after read that sum. The block's input is the token ids. Each token hashes to 16 table rows (bigram and trigram, 8 heads each) that are concatenated to 2560 wide, projected by `key_proj` to the stream's 10240 and by `value_proj` to 2560, and handed with the stream to `ple_block_into` (the per branch gate, then the dilated conv).

The table never reaches the device. `weights::NgramRows` maps the 128 pieces the checkpoint ships (shards 5 to 37) and reads a token's 16 rows of 320 bytes from the mapped files, so a prompt costs one gather on the host and one copy up. The hash multipliers are read from the checkpoint, because the seed that made them is not in `config.json`. The stored per head primes and offsets must equal the ones the config derives, or the load refuses.

Decode carries two things. The hash looks back two tokens, and the conv nine rows (three taps behind the current one at dilation 3). The block kernel takes a whole sequence with a zero state in front, so a decode step reruns the block over the last nine rows' inputs (stream, key and value) plus the new row and keeps the last output row. Everything before the conv is a per row map, so the rows it recomputes are the rows the conv saw. That is ten rows of work for one, and no new kernel; a conv that carries its own state is the step's business.

The reference does not need the 102 GB table in host memory: the dump script replaces the reference's `nn.Embedding` with a module that reads the hashed rows from the same shards. The gate holds the gathered rows bit for bit (the table is the same, so any difference is in the hash) and the block output to four times the reference's own floor, beside the stages every layer is held at. Only the cluster holds the table in full, so layer 1 is recorded and gated on one GH200, with the bookends, a 24 token prefill then 8 decode steps:

```
python tools/accuracy/dump_qwen4_layer_golden.py --model /data/models/Qwen3.8-Flash-Next --layers 1 --decode 8 --out golden/qwen4-layers1-decode8.safetensors
```

| Stage | prefill, 24 tokens | 8 decode steps, max |
| --- | --- | --- |
| `ple_embeds`, `ple_key`, `ple_value`, `ple_out` | 0 | 0 |
| `attn_out` | 5.9e-3 (floor 4.7e-2) | 3.9e-3 (floors 4.4e-3 to 9.8e-3) |
| `gdn_state` | 5.9e-4 (floor 1.1e-3) | 5.7e-4 (floors 1.0e-3 to 1.5e-3) |
| `stream_out` | 5.1e-3 (floor 1.4e-2) | 4.5e-3 (floors 1.9e-3 to 7.1e-3) |
| `logits`, the engine's own readout | 1.05 | 8.9e-1 |
| argmax | 24 of 24 | 8 of 8 |

The n-gram block matches the reference bit for bit: the gathered rows, both projections and the block output. Three prompt tokens and one decode step pick a different expert set, each where the reference's own logits for the pair are tied or one bf16 step apart.

## GDN decode reuse

The 36 `linear_attention` layers run Qwen3.5's Gated DeltaNet kernels, because the geometry is the same: 16 key and 48 value heads at head dim 128, a 4-tap depthwise conv, and the same tensor names under `model.language_model.layers.N.`. The prefill wrappers (`conv1d_prefill_batch_into`, `gated_delta_rule_prefill_chunkwise_into`, `GdrChunkwiseScratch35`) and the decode wrappers (`conv1d_decode_batch_into`, `gated_delta_rule_decode_batch_into`) live in `pegainfer-kernels/src/ops/qwen35.rs` behind the kernels' `qwen35` feature, which compiles `csrc/qwen35/{conv1d,gated_delta_rule}.cu` and the Triton chunkwise kernels, and both lines call them through `pegainfer_kernels::ops`. Head counts are runtime arguments, so 48 value heads need no rebuild.

The three kernel limits (`GDN_AOT_KEY_HEAD_DIM`, `GDN_AOT_VALUE_HEAD_DIM`, `LINEAR_CONV_MAX_KERNEL_DIM`) are defined next to the wrappers. This line's `src/config.rs` keeps a CPU-only mirror of them, so its config tests need no device, until slice B adds the device dependency.

Both halves are verified on this line's activations. Prefill: layer 0's GDN block, chunkwise path plus the sigmoid gated norm and `out_proj`, lands within the reference's own floor (the gate above), and so does the recurrent state it leaves (`gdn_state` in the table). Decode: `layer::layer_decode` runs one token through the layer on a `MixerState::Linear` (the conv taps and the `[48, 128, 128]` f32 state, the layout the kernels and the reference share), through the decode wrappers with one pointer per request, which the batched step fills with one per slot. The reference for it is the same script with `--decode 8`: a 24 token prefill on the layer's cache, then the last 8 prompt tokens one at a time through `causal_conv1d_update` and the fp32 recurrence, every stage and the state recorded per step on both devices. The gate replays the steps on the state its own prefill left and holds each step like a prefill. Max over the 8 steps on the same sm_89 card:

| Stage | max difference | reference's own floor |
| --- | --- | --- |
| `gdn_state` | 1.5e-2 | 1.4e-2 |
| `attn_out` | 3.9e-3 | 3.9e-3 |
| `stream_after_attn` | 9.8e-4 | 4.9e-4 |
| `router_logits` | 6.2e-2 | 6.2e-2 |
| `moe_out` | 3.9e-3 | 3.9e-3 |
| `stream_out` | 3.9e-3 | 3.9e-3 |

One of the 8 steps picks a different expert set, with the reference's logits for the pair inside its own spread. The recorded state is cloned per pass on purpose: the cache rewrites it in place, and on the CPU `.cpu()` does not copy, so without the clone every recorded step aliases the final state.

## Traps

- **`output_gate_type` has three unrelated consumers.** On Qwen3.8-27B it is inert and must *not* be implemented (`docs/models/qwen35/support-qwen38.md`). Here it is live and selects the GDN gated-norm activation; the shared `csrc/shared/norm.cu` gated norm **became selectable in #1129** (a sigmoid entry point beside Qwen3.5's SiLU, merged before this line's detection PR), so what remains is this line's call site, which lands with the text graph. It is *also* the marker the frontend's `reasoning_effort` layer arms on. Read the field's consumer, never its name.
- **Two RMSNorm conventions in one layer.** `(1+w)` for `q_norm`/`k_norm`, the indexer layernorms, `hc_norm` and the PLE norms; plain `w` only for the GDN gated norm. The four 10240-wide ones are additionally **grouped by 2560** — likely a reshape of the existing batched `(1+w)` kernel rather than a new kernel, but unverified.
- **The PLE conv is dilated** (kernel 4, dilation `ngram_size` = 3, state length 9) even though its weight tensor `[10240, 1, 4]` is shaped exactly like the GDN conv's. The in-tree `conv1d.cu` is non-dilated and caps its state at four taps, so it cannot serve this.
- **Attention writes back at `hidden_size`.** `o_proj [2560, 6144]` returns to the 2560-wide hidden size, and `attn_hyper_connection.block_inject_weight [4, 10240]` carries it into the 10240-wide hyper-connection stream.
- **GQA group 12 is not a compiled decode group.** `SUPPORTED_GQA_GROUP_SIZES` is `[1, 2, 3, 4, 6, 8]`, so the FlashInfer batch decode path cannot serve this attention as it stands. Prefill is not affected: the FlashInfer prefill kernel takes the group at run time, which is how the layer 3 gate runs, and the layer 3 decode gate runs the same kernel with a one token plan. QSA is sparse and needs kernels of its own, so whether its decode is captured in a graph is slice B's decision, made with those kernels.
- **The QSA gate's bound is measured per case, and its floor was measured too.** Cases 0–5 sit at ≤9.6e-7 and carry a 2e-6 bound; the long case's 2.4e-3 residual is not traced yet; its shape suggests one flipped bf16 rounding decision, and the reference moves by that size across environments (re-deriving its case 2 on x86 from the fixture's own inputs moves one of its 496 scores by 2.2e-3), so it keeps 5e-3 until traced. What the bound must catch was measured on this fixture: keeping the pooled mean in f32 moves scores by 1.55e-2, 100x `rms_eps` by 1.25e-2, dropping `rms_eps` or pooling by sum by 3.3e-3, the pooled key's RoPE by 2.7, and a post-sum relu by 0.83 — every fault sits two to six orders beyond the tight bound. Architecture variance does not loosen it: the gate prints identical numbers on sm_89 and sm_120, and the kernel reads the shared attention cos/sin table (built by `pegainfer_core::rope` with `frequency_dim` equal to the 64 rotated dims, as the reference spaces its frequencies) rather than computing trig, so there is nothing device-specific left to vary. `#1105`'s selection stage must still treat a block tied with the last kept one at the cut as an equally valid pick, because the long case's noise can flip that order.
- **Do not use `load_shard_info_fixed`.** It is a Qwen3.5-specific shard-*filename* workaround living in shared `pegainfer-core`; this checkpoint uses standard `model-0000N-of-00131` naming and wants the plain loader.

## Still open

| Criterion | Blocked on |
| --- | --- |
| Tensor manifest enforced before the first H2D | slice B's loader, which must classify the headers it maps before any upload |
| A captured decode step | the batched step allocates its buffers and builds its plan on every call; capture wants fixed buffers and a plan refilled in place, attention runs FlashInfer's prefill kernel because the compiled decode list has no GQA group 12, and QSA past the dense range needs the indexer's selection in the attention kernel |
| Routed experts beyond correctness | one GEMM pair per expert and one host sync per layer; the pointer batched form for decode and a grouped GEMM for prefill |
| Serving all 48 layers | 234 GiB of bf16 text weights do not fit one card: EP4 across four GH200s, with the step on top |
| Tokenizer / chat-template ID fixtures | transformers **5.16.0+** — the checkpoint's own tokenizer is `Qwen2Tokenizer`, which is older, but fixtures are compared against the reference implementation; nothing in the tree commits token-ID fixtures today |

Note that `pegainfer-server` cannot be compiled on a CPU-only box, so the registration above is verifiable only where a CUDA build works.

A local box here is **sm_120**, which is neither support-matrix row (H200 sm_90, L20 sm_89) — anything proven there establishes numerics and must never be recorded as a platform result.
