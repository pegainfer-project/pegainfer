# DSpine Speculative Decoding (Qwen3-8B)

**TL;DR:** Tencent's `tencent/Qwen3-8B-DSpine-b16` runs on the existing DFlash
speculative transaction behind `--dflash-draft-model-path` — no new CLI flag or
crate. DSpine adds two things over DFlash: a per-layer **adjacent-causal code
bridge** and a **serial last-write refinement** over top-K candidates. V1 is
greedy-only; non-greedy requests keep the existing sampled-verify path
(lossless, but below SGLang's sampled serial chain). Verified on one L20 (46 GB):
the generic greedy losslessness gate (five sequential prompts × 64 tokens) is
lossless (structured prompts accept whole 15-draft blocks), and single-stream
decode measures **47.0 → 120.1 tok/s (2.56×)**. The bridge footprint is reserved
before KV pool sizing; probabilistic sampling parity is the tracked follow-up.

Last touched: 2026-10

## Checkpoint and config

`Qwen3-8B-DSpine-b16` is a DFlash draft layout (`block_size=16`, anchor-drop,
`target_layer_ids=[1,9,17,25,33]`, five Qwen3 layers) plus
`selected_prefix_bridge.*`. Config detection is
`dflash_config.selected_prefix_code_bridge`; the resolved fields
(`code_size=1024`, `rank=512`, `eps=1e-6`, `head_topk=16`, active layers
`[1..5]`) live in `DFlashConfig::dspine`.

The loader accepts only the checkpoint shapes it can serve:

- `codebook [V,C]`, `read_weight [C,H]`, `early_read_weights [L-1,C,H]`,
  `feature_write_weights [L,H,R]`, `gate/predecessor_gate/prev [R,C]`,
  `gate.bias [R]`, `out.weight [C,R]`.
- `out.weight` must be all-zero (terminal disabled).
- `independent_write_version` and `terminal_disabled_version` (plus
  `draft_block_causal_version` / `last_write_refine_version` when present) must
  be `[1]` I64 scalars.
- The 3D tensors are flattened per layer by
  `weight_loader::load_tensor_3d_layer`; the repo has no 3D device type.

Read/early-read and write weights are stacked into `[L*C,H]` and `[L*H,R]` so a
layer's slice is one `gemm_rows_into_checked`. Gate/predecessor/prev are stacked
into `[3R,C]` (`[q|s|v]`).

## Bridge forward

After each active draft layer's residual add, with
`raw = layer_output = residual_before_mlp + mlp_out`:

```
rms        = sqrt(max(mean(raw²), FLT_MIN))
normalized = raw / (rms + bridge_eps)
projected  = normalized @ read_weight(layer)ᵀ
codes      = sqrt(C) * projected / (||projected|| + eps)
             # block row 0 is sqrt(C) * codebook[anchor_token]
packed     = codes @ [gate; predecessor_gate; prev]ᵀ     # [q|s|v]
message[r] = 0                                            if r % 16 == 0
             sigmoid(q[r] + s[r-1] + bias) * v[r-1]       otherwise
delta      = message @ feature_write_weights(layer)ᵀ
out        = raw                                          on anchors
             raw + rms * delta                            otherwise
```

The write fuses the **next layer's input RMSNorm**; the layer loop skips that
norm on the next iteration (`input_normed`). The last layer skips the next norm
and `compute_logits_into` applies the draft's own final norm before the shared
target LM head. Anchor rows therefore pass through bit-exactly, and the stash
`{raw, rms, packed}` from the **last** layer feeds the refinement.

Draft attention is causal inside the block (`single_prefill_nhd_causal_into`)
over the same target-context prefix; plain DFlash keeps the existing noncausal
kernel. The kernel implementations are in
`pegainfer-kernels/csrc/shared/dspine.cu` and
`pegainfer-kernels/src/ops/dspine.rs`.

## Serial last-write refinement (greedy)

At load, two vocabulary tables are built from the **draft's** final norm and the
shared target LM head:

```
TG [V,R]   = bf16(bf16(lm_head * norm_w) @ W_last)          # W_last: [H,R]
TSV [V,2R] = bf16(bf16(sqrt(C) * codebook) @ [predecessor; prev])
```

Per draft round, after the pass-1 logits:

1. Top-K (K=`head_topk`) each proposal row (block rows `1..15`), ascending ids.
2. For proposal row `n` (block row `b*16+1+i`) and sender lane `k`:
   `sid` is the anchor for `i=0`, else `cand[n-1,k]`;
   `base[j] = <raw[row]·norm_w, lm_head[cand[n,j]]>`;
   `m = sigmoid(q + TSV[sid,:R] + bias) * TSV[sid,R:]`;
   `score[j] = base[j] + rms[row] * <m, TG[cand[n,j]]>`;
   `idx[n,k] = argmax_j score[j]` (lowest lane on ties).
3. Walk `idx` from the anchor (`lane = idx[n0,0]`) and emit
   `[anchor, p1..p15]`; the anchor slot is dropped by the existing
   `drafts_start = 1` rule.

Only the serial last write is refined; earlier writes stay parallel. Dead
lanes are masked with `-inf`/`INT_MAX`; no kernel multiplies a mask value, so
there is no `0 * -inf` NaN path.

## Memory

`DFlashMemoryReservation` adds one DSpine term before KV pool sizing:
weights/tables, the load-time table-build transients (scaled LM head and
codebook, per-layer pieces), and the lane scratch (projected/codes/packed/
message/rms, candidates/idx/tokens). At `max_batch=256`, block 16, the pinned
term is **2,497,840,128 B**. The token-scaling draft KV term is unchanged;
plain DFlash/DSpark reserve zero.

## Evidence

- Host-reference kernel tests (`dspine::tests`, ignored without a GPU):
  read/finish, codes/messages, top-K (ascending-id contract) + serial table +
  chain, scale kernels.
- `dflash_speculative_gate::dspine_checkpoint_is_lossless_and_accepts`
  (ignored; set `PEGAINFER_TEST_MODEL_PATH` and
  `PEGAINFER_DSPINE_TEST_MODEL_PATH`) replays three prompts × 32 greedy tokens
  against a plain engine and asserts losslessness, `accepted > 0`, and a live
  speculative path.
- Measured on one L20 (46 GB), Qwen3-8B + DSpine b16 (manual harness runs):
  - generic greedy gate (`dflash_speculative_greedy_matches_plain_greedy`,
    five sequential prompts × 64 tokens): lossless; structured prompts
    accepted whole 15-draft blocks.
  - single-stream A/B (`dflash_speculative_perf`): 47.0 → 120.1 tok/s (2.56×).
  - Qwen3-4B DFlash gate still passes 100% lossless (plain DFlash path
    byte-identical: the DSpine branches are all gated on `uses_dspine()`).

## V1 scope and follow-ups

- **Greedy only.** Non-greedy requests use the existing sampled-verify path.
  Full parity needs SGLang's sampled serial chain, draft→verify `q` relay, and
  chain rejection sampling (tracked in #1169).
- No draft-side CUDA Graph, TP>1, prefix cache, or hedge. Draft attention is
  full causal, matching the checkpoint.
- The score table currently runs one block per proposal row with block
  reductions; fusing it with the top-K or capturing the draft tail is a
  performance follow-up, not a correctness one.

## Running it

```bash
cargo run --release -- \
  --model-path /path/Qwen3-8B \
  --dflash-draft-model-path /path/Qwen3-8B-DSpine-b16
```

The startup log names the bridge (`DSpine bridge: code_size=..., rank=...,
top_k=..., layers=...`). Gates:

```bash
PEGAINFER_TEST_MODEL_PATH=/path/Qwen3-8B \
PEGAINFER_DSPINE_TEST_MODEL_PATH=/path/Qwen3-8B-DSpine-b16 \
cargo test --release -p pegainfer-qwen3 --test dflash_speculative_gate \
  dspine_checkpoint_is_lossless_and_accepts -- --ignored --nocapture

PEGAINFER_TEST_MODEL_PATH=/path/Qwen3-8B \
PEGAINFER_DFLASH_TEST_MODEL_PATH=/path/Qwen3-8B-DSpine-b16 \
cargo test --release -p pegainfer-qwen3 --test dflash_speculative_perf \
  -- --nocapture
```

## Implementation map

| Area | Location |
| --- | --- |
| Config / validation | `pegainfer-qwen3/src/config.rs` (`DSpineConfig`) |
| Bridge weights, TG/TSV, scratch, reservation | `pegainfer-qwen3/src/dflash/dspine.rs` |
| Loading | `pegainfer-qwen3/src/dflash/loading.rs` |
| Forward hook + proposal | `pegainfer-qwen3/src/dflash.rs` (`draft_logits_batched`, `dspine_draft_tokens`) |
| Lane dispatch | `pegainfer-qwen3/src/executor/dflash_lane.rs` |
| CUDA kernels | `pegainfer-kernels/csrc/shared/dspine.cu` |
| Rust wrappers | `pegainfer-kernels/src/ops/dspine.rs` |
| 3D loader | `pegainfer-core/src/weight_loader.rs` (`load_tensor_3d_layer`) |
