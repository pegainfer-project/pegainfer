# GLM5.2 Oracle Harness

> **TL;DR:** Self-contained accuracy oracle for the GLM5.2 bricks: `tools/accuracy/glm52_oracle.py` (uv script, pinned `transformers==5.12.1` official `glm_moe_dsa` modeling code) emits paste-ready Rust probe constants; `pegainfer-glm52/src/oracle/mla.rs` replays the identical seeded input through the engine and asserts them. First gate (MLA decode brick) is green on jz38 H200 — 64/64 probes, full-tensor diff RMS 1.8e-5 vs tol 6.9e-5 — and both negative controls (rope-swap, q_b head negate) go red. No MB-scale fixtures in git: probes are hardcoded constants, the full tap dump is an optional local safetensors.
>
> **Last touched:** 2026-07

## Design: why this shape

**Ground truth is the official modeling code, not hand-written math.** The harness instantiates `GlmMoeDsaAttention` from transformers 5.12.1 (GLM5.2 = `GlmMoeDsaForCausalLM` landed upstream 2026-06-08, inherits deepseek_v32). Hand-written Python is limited to *transport*: reading layer-0 tensors from the checkpoint, fp8 block dequant (the checkpoint's documented `weight * weight_scale_inv` per-128Â² contract), and forward hooks. Transport errors produce garbage, not subtle drift — the dangerous failure mode of a hand-written reference is exactly what this avoids.

**No HF model download, no multi-GPU.** Only layer-0 weights are read (safetensors by name, CPU); the checkpoint dir has no modeling `.py` (not a trust_remote_code release), so the pinned transformers version in the script header IS the oracle version. Bump it deliberately, never implicitly.

**Precision emulation (`--precision fp8sim`, default).** The engine runs fp8 GEMMs (TRTLLM CUTLASS) and caches kv_c as fp8 per-128-group; a pure-bf16 HF forward would force a loose tolerance that hides real bugs. `Fp8SimLinear` quant-dequants activations per 128-group before an f32 matmul, and a hook quant-dequants the `kv_a_layernorm` output (cache fidelity) before kv_b decompression. Result: gate tolerance is tight (rel 0.05 × output RMS ≈ 6.9e-5 absolute) and measured headroom is ~4x (diff RMS 1.8e-5). `--precision bf16` runs the untouched official path as a cross-check.

**Fingerprints, typed by what they can promise:**

| kind | assert? | where |
|---|---|---|
| float probes (64 sampled indices + RMS) | tolerance-assert | hardcoded in the Rust gate |
| input bf16 sha256 | exact-assert | hardcoded; fails on PRNG drift *before* any kernel runs |
| float tensor sha256 | **never** — provenance only | comment header |
| full tap dump | optional stats-assert (RMS + p99, max printed) | `PEGAINFER_GLM52_ORACLE_DUMP=/path.safetensors`, not in git |

**Cross-language input without fixtures:** splitmix64 (integer-only) → 53-bit uniform → `(u-0.5)*4` → f32 → bf16. Both sides derive it from the seed; digest-checked.

## Usage

```bash
# 1) generate probes (jz38 or any box with the checkpoint; CPU-only, ~4 min)
uv run tools/accuracy/glm52_oracle.py --model-path /data/models/GLM-5.2-FP8 --emit rust

# 2) paste the emitted block over the GENERATED section in oracle/mlla.rs

# 3) run the gate (H200 + checkpoint)
PEGAINFER_TEST_MODEL_PATH=/data/models/GLM-5.2-FP8 \
  cargo test --release -p pegainfer-glm52 --features glm52 --lib mla_oracle -- --ignored --nocapture

# debugging a divergence: dump all taps, gate then also whole-tensor-diffs `o`
uv run tools/accuracy/glm52_oracle.py --model-path ... --emit safetensors --out /tmp/taps.safetensors
PEGAINFER_GLM52_ORACLE_DUMP=/tmp/taps.safetensors cargo test ... mla_oracle -- --ignored --nocapture
```

Taps captured (stderr table each run): `hidden, cos/sin, q_resid, q_full, ckv, kv_c_cached, attn_v, o, topk_indices`. PR2's indexer gate extends the same registry (`index_q`, `mqa_logits`, `topk_slots`).

## Verification of the verifier (done 2026-07-02, jz38)

- **Green on clean probes:** 64/64 within tol; full-tensor diff RMS 1.82e-5, p99 5.34e-5, max 6.1e-4 (1.2M elements).
- **Negative controls red:** `--inject-fault rope-swap` → 50/64 probes fail; `--inject-fault qb-head-negate` → 25/64 fail. The gate demonstrably has teeth.
- **Fault-strength lesson:** negating a single 128×128 q_b block was *absorbed* (64/64 still green) — softmax smoothing + o_proj's 64-head mixing dilute a one-block weight fault below tolerance. Weight-level negative controls must hit a full head. Corollary for real bugs: the MLA gate is sensitive to systematic errors (RoPE convention, layout, scale relay), not to a few flipped weight blocks — that class needs the upstream taps (q_full probes), which the dump comparison covers.

## Pitfalls (hit during bring-up)

- **numpy double-rounds:** `.astype(np.float32)` then bf16 = f64→f32→bf16. Rust `bf16::from_f64` rounds once and diverges on a handful of values. The gate mirrors `from_f32(x as f32)`; the input digest catches any future drift here by design.
- **Float digests are torch/hw-version-fragile.** Only the *input* digest is asserted (integer-derived, stable); output digests are provenance comments.
- **Absolute max diff grows with element count** (bf16 tail over 1.2M elements) — assert RMS + p99, print max. Same lesson as the qwen3 golden gate.
- **jz38 worktree submodules:** `ecbbe74` (DG_NO_TORCH DeepGEMM) is a local patch not on `deepseek-ai/DeepGEMM` origin — `git submodule update` in a fresh worktree fails to fetch it. Push it from a checkout that has it: `git push ssh://jz-38/<worktree>/pegainfer-kernels/third_party/DeepGEMM <sha>:refs/heads/dg-no-torch`, then check out. FlashMLA needed `--force` to materialize.
- Build env on jz38: `PATH=/root/.cargo/bin:/usr/local/cuda/bin`, `CUDA_HOME=/usr/local/cuda` (12.8), `PEGAINFER_NCCL_ROOT=<repo>/.venv/lib/python3.12/site-packages/nvidia/nccl` (system NCCL 2.29.7 is too old).

## Prototype-era fixtures

`jz38:/data/models/glm52_mla_ref/` still holds the old prototype's dump (`layer0.npz`, `flashmla_probe/`, `moe_probe/`, ...). The generating script was never committed — that irreproducibility is what this harness replaces. Useful only as a third-party cross-check; do not build new tests on it.

## Next

- PR2 indexer gate: same harness, `--stage indexer` tap set (`topk_indices` slot comparison needs set-overlap across implementations, not exact match — FlashInfer vs torch.topk tie-break on 1-ULP logit ties); sha256 of slots IS assertable for Rust-vs-Rust regression pins on the same GPU.
- PR3 composes dense MLP + MoE taps (`GlmMoeDsaMLP` / `GlmMoeDsaMoE` are in the same modeling file).
