# GLM5.2 DSA Indexer Forward (PR2 model-crate)

> **TL;DR:** Wire `pegainfer-glm52/src/indexer.rs` — `Glm52IndexerLayerWeights` + `glm52_indexer_forward` — composing the 6 kernel ops already on main (#489) into a DSA decode indexer that produces `topk_indices[2048]`. Aligned to vllm's `DeepseekV32Indexer` (the production reference). Three ops are missing from the repo and must be added: LayerNorm (k_norm, eps=1e-6, with bias), interleaved indexer RoPE (64-dim, q+k), and weights-fold (`weights * q_scale * softmax_scale * n_heads^-0.5`). Oracle gate extends the existing harness (#499) with a `topk_indices` set-overlap assertion. `from_host` only; `from_device` deferred to PR4.
>
> **Last touched:** 2026-07

## Why this correction still matters

The original bring-up data-flow omitted three steps that vLLM's `DeepseekV32Indexer.forward` performs: **k_norm is a LayerNorm (not RMSNorm) with bias**, **per-head `weights_proj` + head-weighted score reduction** (fused inside DeepGEMM's `fp8_paged_mqa_logits`), and **ReLU** (also fused inside DeepGEMM). It also listed Hadamard; vLLM's DSv32 path does NOT apply Hadamard (only TokenSpeed does). Aligning to vLLM — which already landed the `weights` parameter in our DeepGEMM wrapper (#489) — means Hadamard is dropped for now. This doc is the corrected contract.

## vllm cross-reference (the alignment target)

`vllm/vllm/models/deepseek_v32/nvidia/attention.py:39-157` — `DeepseekV32Indexer`:

```
forward(hidden, qr, positions, rotary_emb):
  q   = wq_b(qr)                    # qr = q_a_layernorm(q_a_proj(hidden))  [from MLA layer]
  q   = q.view(-1, n_heads=32, head_dim=128)
  q_pe, q_nope = split(q, [rope_dim=64, 64])    # first 64 = rope, last 64 = pass-through

  kw  = wk_weights_proj(hidden)     # fused GEMM: [hidden] -> [head_dim + n_heads]
  k      = kw[:, :128]
  weights= kw[:, 128:]             # [n_heads=32]

  k   = k_norm(k)                   # LayerNorm(128, eps=1e-6, bias=True)
  k_pe, k_nope = split(k, [64, 64])

  q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))   # interleaved RoPE
  q = cat([q_pe, q_nope], dim=-1)  # [N, 32, 128]
  k = cat([k_pe.squeeze(1), k_nope], dim=-1)  # [N, 128]

  q_fp8, q_scale = per_token_group_quant_fp8(q, group=128)  # q is [N*32, 128] flattened
  weights = weights.unsqueeze(-1) * q_scale * softmax_scale * n_heads**-0.5
  weights = weights.squeeze(-1)     # [N, 32]  -- fold q_scale into weights for DeepGEMM

  sparse_attn_indexer(hidden, q_fp8, k, weights, ...)   # -> topk_indices[N, 2048]
```

Key facts confirmed from vllm source:
- **LayerNorm** (`nn.LayerNorm`, eps=1e-6, **with bias**) — `attention.py:76`. NOT RMSNorm. repo has no LayerNorm kernel.
- **Interleaved RoPE** — `attention.py:297-298,339`: `is_neox_style=not indexer_rope_interleave`. GLM5.2 config `indexer_rope_interleave=true` → `is_neox_style=false` → interleaved pairs. Only first 64 dims get RoPE; last 64 are pass-through.
- **weights fold** — `attention.py:152-155`: `weights * q_scale * softmax_scale * n_heads^-0.5`. The `q_scale` from per-token-group quant is folded INTO weights, so DeepGEMM receives pre-scaled weights and q_fp8 (scale=1 effectively).
- **DeepGEMM fuses ReLU + per-head weighting** — `sparse_attn_indexer.py:562-571`: `fp8_fp4_paged_mqa_logits((q_fp8, None), kv_cache, weights, ...)` — the `weights` tensor is passed as a kernel argument, confirming the DeepGEMM kernel internally applies ReLU and per-head weighted reduction. Our wrapper (#489) already accepts `weights: &CudaSlice<u8>`.
- **Hadamard** — vllm DSv32 path does NOT apply Hadamard. TokenSpeed does (`glm5.py:398`). transformers 5.13.0.dev0 (local, PR #46842) skips it (comment: "orthogonal, dot products preserved"). **Aligning to vllm: no Hadamard in PR2.** The already-landed `glm52_indexer_hadamard_bf16` kernel stays in-tree as dead code (may be needed if oracle divergence shows it matters).

## transformers cross-reference (the oracle source)

Local transformers at `/data/code/workspace-rustllm/transformers` (5.13.0.dev0, commit `8698b5a525`):

`src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py:172-263` — `GlmMoeDsaIndexer`:
- Same structure as vllm: `wq_b(q_resid)`, `wk(hidden)`, `k_norm` (LayerNorm eps=1e-6 bias), `weights_proj(hidden)`.
- `apply_rotary_pos_emb_interleave` (line 240) — **fixed** in 5.13.0.dev0 (was non-interleave in 5.12.1, contradiction with config; PR #46842 fixed it).
- `scores = relu(q @ k^T * scale)` then `weights @ scores` head-weighted reduction.
- topk = `torch.topk`.
- Hadamard skipped.

The oracle harness (`tools/accuracy/glm52_oracle.py`) runs transformers 5.12.1 (pinned). **Must bump to 5.13.0.dev0** (or wait for 5.13.0 release) so the indexer RoPE matches vllm. See "Oracle gate" below.

## Scope

### New model-crate file: `pegainfer-glm52/src/indexer.rs`

```
Glm52IndexerLayerWeights {
    wq_b:    ProjWeight,      // [32*128, 2048]  fp8
    wk:      ProjWeight,      // [128, 6144]     fp8
    k_norm_w: DeviceVec,      // [128]  bf16  (LayerNorm gamma)
    k_norm_b: DeviceVec,      // [128]  bf16  (LayerNorm beta — RMSNorm has no beta)
    weights_proj: ProjWeight, // [32, 6144]     fp8  (vllm fuses wk+weights_proj; repo does 2 GEMMs)
}

glm52_indexer_forward(
    ctx, w,
    hidden,        // [6144] bf16
    q_resid,       // [2048] bf16  (from q_a_layernorm(q_a_proj(hidden)) — produced by MLA layer)
    cos, sin,      // [32] bf16  (indexer RoPE table, interleaved)
    position,
    index_k_cache, // mutable paged fp8 cache
    slot_mapping,  // [1] i64  (current token's cache slot)
    block_table,   // [num_blocks] i32
    seq_lens,      // [1] i32
    num_sms,       // for DeepGEMM scheduling
) -> topk_indices: CudaSlice<i32>  // [2048]
```

Forward steps (each maps to a kernel op):
1. `q = fp8_linear(w.wq_b, q_resid)` → `[4096]` reshape `[32, 128]`
2. `kw = fp8_linear(w.wk, hidden)` → `[128]`; `weights = fp8_linear(w.weights_proj, hidden)` → `[32]`
3. `k = layer_norm(kw, w.k_norm_w, w.k_norm_b, eps=1e-6)` → `[128]` ← **NEW KERNEL**
4. RoPE: `q_rope = interleave_rope(q[:, :64], cos, sin)`; `k_rope = interleave_rope(k[:64], cos, sin)` → q `[32,64]`, k `[64]` ← **NEW KERNEL**
5. q quant + weights fold: `q_fp8, q_scale = per_token_group_quant(q_flat, group=128)`; `weights_out = weights * q_scale * softmax_scale * n_heads^-0.5` ← **NEW KERNEL (or host-side)**
6. k quant + cache: `glm52_indexer_k_quant_and_cache(k, index_k_cache, slot_mapping)` → ✅ landed
7. DeepGEMM logits: `glm52_deepgemm_paged_mqa_logits(q_fp8, index_k_cache, weights_out, ...)` → logits `[max_model_len]` bf16 → ✅ landed (fuses ReLU + per-head weighting)
8. topk: `glm52_flashinfer_topk_2048(logits, seq_lens)` → `local_topk_offsets[2048]` → ✅ landed
9. slots: `glm52_indexer_local_topk_to_slots(local_topk_offsets, seq_lens, block_table)` → `global_slots[2048]` → ✅ landed

### New kernel ops (1) + host-side math (2)

After checking vendored FlashInfer / existing repo code, only **one** new CUDA kernel is needed:

| op | file | what | source |
|---|---|---|---|
| `glm52_layer_norm_bf16` | `ops/glm52/layernorm.rs` + `csrc/shared/flashinfer_norm.cu` (add to existing) | LayerNorm(x, gamma, beta, eps) for `[tokens, dim]` bf16. | **wrap FlashInfer `flashinfer::norm::LayerNorm<T,Tw>`** (`norm.cuh:942`) — same pattern as existing `rms_norm_cuda` wrapper. Template supports bf16 gamma/beta. |
| indexer interleave RoPE | `ops/glm52/indexer_rope.rs` + `csrc/glm52/glm52_indexer_rope.cu` | Interleaved RoPE on `[tokens, rope_dim=64]` (q per-head) and `[tokens, 64]` (k single). Cos/sin `[32]`. | **reuse `rope_block` device function** from `glm52_mla_assembly.cu:40` — identical shape/semantics (interleave, rope_dim=64, cos/sin=[32]). Extract into a small standalone launch (~30 lines). |
| weights fold | host-side (Rust) | `weights[32] * q_scale[32] * softmax_scale * n_heads^-0.5` | **host-side f32 math** — 32 elements, H2D upload 128 bytes. Cheaper than a kernel launch. |

**No hand-written CUDA from scratch.** LayerNorm wraps FlashInfer (like `rms_norm_cuda` already does). RoPE reuses the existing `rope_block` device function. Weights fold is host-side.

**Alternative considered:** fuse all 3 into one kernel (like vllm's `fused_norm_rope` + `fused_q`). Rejected for PR2 — vllm's fusion is across MLA + indexer (different shapes, different cos/sin caches), which adds complexity without a measured perf win at bs=1 decode. Keep them separate and correct first; fuse if ncu flags it.

### Not in PR2

- `from_device` constructor (production loader path from resident EP8 slab) — deferred to PR4 (first consumer of the slab).
- Hadamard rotate — dropped (vllm doesn't apply it; transformers skips it). Kernel stays in-tree as dead code.
- Runner/coordinator changes — still rejecting.
- Prefill indexer — decode-first; prefill rides decode path token-by-token.

## Oracle gate

Extends the existing harness (#499, `tools/accuracy/glm52_oracle.py` + `oracle/mla.rs`):

1. **Bump transformers pin** to 5.13.0.dev0 (or 5.13.0 release if available). The indexer RoPE fix (PR #46842) is required for oracle correctness. `glm52_oracle.py` line 6 already has the pin comment.
2. **Add `--stage indexer` tap set** to the harness: `index_q`, `index_k`, `index_weights`, `mqa_logits`, `topk_indices`. The harness already captures `topk_indices` (line 301) but does not assert it.
3. **Assertion strategy** (from `oracle-harness.md:65`):
   - HF-vs-Rust: **set-overlap** (FlashInfer vs torch.topk tie-break on 1-ULP logit ties differ; exact match is impossible). Assert `overlap >= 2047/2048` (allow 1 tie-break divergence).
   - Rust-vs-Rust (regression pin): **sha256 of slots** (same GPU, same kernel → deterministic).
4. **Short-context regression**: at ctx <= 2048, sparse top-k == full top-k (the PR1 path). Assert the indexer's output matches `[0, 1, ..., position, -1, ...]` exactly.

**Context for the gate**: ctx=4096 (where sparse != full). Requires `PEGAINFER_TEST_MODEL_PATH` pointing to the GLM-5.2-FP8 checkpoint and an H200.

## Build & test

```bash
# Build (SM90a, H200)
export PEGAINFER_DEEPGEMM_ROOT=pegainfer-kernels/third_party/DeepGEMM/deep_gemm
export CUDA_HOME=/usr/local/cuda
export PEGAINFER_NCCL_ROOT=<path>
cargo check --release -p pegainfer-glm52 --features glm52

# Smoke test (no checkpoint needed — synthetic input, verify launch + shape)
cargo test --release -p pegainfer-glm52 --features glm52 --lib indexer_smoke -- --nocapture

# Oracle gate (H200 + checkpoint)
PEGAINFER_TEST_MODEL_PATH=/data/models/GLM-5.2-FP8 \
  cargo test --release -p pegainfer-glm52 --features glm52 --lib indexer_oracle -- --ignored --nocapture
```

## Execution plan

1. Write 3 new kernel ops (LayerNorm, interleave RoPE, weights fold) — CUDA + Rust ops + FFI. Smoke test each.
2. Write `indexer.rs` model-crate forward, composing all 9 steps. `from_host` constructor only.
3. Register `indexer` module in `lib.rs` as `#[allow(dead_code)]`.
4. Bump `glm52_oracle.py` transformers pin to 5.13.0.dev0; add `--stage indexer` tap set.
5. Write `oracle/indexer.rs` — set-overlap assertion + short-context regression.
6. Build + test on jz38 (H200).

## Read

- `docs/models/glm52/oracle-harness.md` — harness design, verification, pitfalls.
- `pegainfer-glm52/src/mla_decode.rs` — PR1 forward pattern to mirror.
- `vllm/vllm/models/deepseek_v32/nvidia/attention.py:39-157` — `DeepseekV32Indexer` (alignment target).
- `vllm/vllm/models/deepseek_v32/nvidia/kernels.py:87-364` — `fused_norm_rope` (k_norm + RoPE + quant + cache fused; reference for the unfused decomposition).
- `vllm/vllm/model_executor/layers/sparse_attn_indexer.py:295-647` — `sparse_attn_indexer` (DeepGEMM logits + topk + slots).
- `transformers/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py:172-263` — `GlmMoeDsaIndexer` (oracle source, 5.13.0.dev0).

## Relevant history

- PR2 kernel ops (#489) landed all 6 ops and smoke tests on H200 before the model-crate forward was wired in #521.
- Oracle harness (#499) landed the self-contained probe pipeline; MLA gate is green. Indexer gate was noted as the next extension.
- transformers 5.12.1 had a RoPE contradiction (config says `indexer_rope_interleave=true` but modeling used non-interleave). Fixed in 5.13.0.dev0 PR #46842 (`8698b5a525`). The harness must bump its pin.
- vllm's `DeepseekV32Indexer` is the production reference for GLM5.2 DSA. It does NOT apply Hadamard (unlike TokenSpeed). The `glm52_indexer_hadamard_bf16` kernel landed in #489 stays as dead code.

## Risks / open questions

- **LayerNorm vs RMSNorm**: repo only has `rms_norm_into`. A LayerNorm kernel is new but trivial (mean + var + affine with bias). ~40 lines CUDA.
- **Interleave RoPE shape**: MLA's interleave RoPE is fused in `query_assemble`/`cache_pack` (operates on `[64, 576]` / `[1, 656]`). Indexer needs `[32, 64]` (q, per-head) and `[1, 64]` (k). Different enough to warrant a separate kernel, not a reuse.
- **weights_proj dtype**: transformers keeps `weights_proj` in fp32 (`_keep_in_fp32_modules`). vllm loads it as bf16 (fused GEMM, `quant_config=None` but still bf16). The checkpoint stores it as fp8 (block-scaled). For the engine: load as fp8, `fp8_linear` produces bf16 output, then the weights_fold kernel upcasts to f32 for the fold math. This matches vllm's `weights.float()` cast.
- **Oracle tie-break**: FlashInfer `TopKDispatch` with `TopKTieBreak::Small` vs `torch.topk(sorted=False)` — tie-break differs on 1-ULP logit ties. Set-overlap with tolerance 1/2048 is the assertion. If this proves flaky, pin Rust-vs-Rust sha256 and drop the HF-vs-Rust set-overlap.
