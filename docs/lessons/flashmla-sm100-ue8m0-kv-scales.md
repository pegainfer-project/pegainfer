# FlashMLA sm100 truncates fp8 KV-cache scales to powers of two (UE8M0)

**TL;DR**: The FlashMLA V3.2 fp8 sparse decode kernel on Blackwell (sm100/sm103, `sm100::decode::head64`) converts the 4 per-token f32 K/V scales to e8m0 with **round-toward-zero** for its tcgen05 block-scaled MMA — any non-power-of-two scale is silently read up to 2× too small. The cache **writer** must round group scales UP to the next power of two (upstream `tests/quant.py::_cast_scale_inv_to_ue8m0`); then the truncation is lossless. The sm90 kernel reads f32 scales exactly, so H200 masks the bug completely.

## How it bit us (2026-07, GLM5.2 EP4 bring-up on GB300)

- Symptom: attention output ≈ **0.67× the HF oracle at every position** (corr 0.996 — a near-pure scalar shrink), which at the decoder-layer level looks like the layer delta scaled by 0.7 early (attention-dominated) rising to ~0.95 late (MLP-dominated). The EP4 MoE oracle gate failed 23/64 probes; the failure had *nothing to do with the MoE chain*.
- Why nobody saw it earlier: EP8 runs on H200 (sm90 reads f32 scales exactly); TP4 on GB300 uses the FlashInfer backend (heads=16, 576-byte static-e4m3 cache — different format entirely). The heads=64 FlashMLA fp8-DS path had never been accuracy-gated on Blackwell. The dense layer-0 gate *was* failing on GB300 (63/64 with 0 allowed), but the smaller layer-0 delta kept the bias just under the tolerance radar.
- Diagnosis path that worked: layer gate → per-row α-fit against the oracle (engine delta = α·oracle delta) → MLA-only gate (same α at position 0 where softmax ≡ 1 ⇒ V-path, not softmax) → **weights-free kernel gate** (`pegainfer-kernels/tests/glm52_sparse_mla.rs::flashmla_sparse_vs_reference_gate`, FlashMLA vs f64 naive reference on synthetic data): single-key case off by 34%, per-128-group ratios constant (0.55/0.54/0.67/0.95) ⇒ wrong scales ⇒ implied scales all exact powers of two ⇒ read the upstream kernel: `__nv_cvt_float2_to_e8m0x2(..., cudaRoundZero)`.

## The rules

1. **Every writer of the 656-byte fp8_ds_mla cache must produce power-of-two group scales**, rounded UP (`2^ceil(log2(amax/448))`). In GLM5.2 that is the UE8M0 per-token-group quant (`glm52_fp8_per_token_group_quant_bf16_ue8m0_*`) feeding `glm52_mla_cache_pack`. Exact bit trick, no `log2f` rounding hazard: `(bits + 0x007FFFFF) & 0x7F800000`.
2. GEMM **activation** quant (MoE/dense DeepGEMM paths) keeps amax/448 scales — the contract is only about the FlashMLA KV cache.
3. Accuracy harnesses must model the cache with the same pow2 quant (`quant_dequant_groups_ue8m0` in `tools/accuracy/glm52_oracle.py`, exact via `torch.frexp`).
4. **Kimi-K2 uses the same FlashMLA sparse kernels.** It currently runs 8×H200 only; if it ever moves to Blackwell, its cache writer needs the same audit *first*.
5. When bringing up a new arch, run the weights-free kernel gate before any model-level gate — it separates "kernel/glue broken on this arch" from modeling questions in minutes.
