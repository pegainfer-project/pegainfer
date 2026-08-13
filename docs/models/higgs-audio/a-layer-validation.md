# Higgs-Audio A-Layer Validation

This note records the intended review scope for the Higgs-Audio A-layer bring-up.
It is deliberately narrower than full audio generation: the goal is to load the
Higgs checkpoint/config, run the Qwen3 text backbone prefill, expose traceable
hidden-state boundaries, and verify the one-step audio-head contract against
offline golden tensors.

## Scope

Included:

- Higgs-Audio crate wiring and one-step CLI tools.
- Qwen3 config-view loading backed by tensor-name aliases, without copying a
  renamed checkpoint payload.
- Single-GPU Qwen3 diagnostic prefill hooks for final hidden, per-layer hidden,
  and selected layer stage dumps.
- One-step audio projection parity tools and a small offline fixture.

Not included:

- Full delay-pattern decode state machine.
- Multi-codebook autoregressive audio decode.
- Codec/vocoder integration or waveform output.
- Tensor-parallel diagnostic trace support.

## Validation Model

The A-layer validation is golden-trace-driven:

1. Generate or load a reference prompt and hidden/logit tensors from the Python
   or sglang-omni reference stack.
2. Run the PegaInfer Higgs one-step path against the same checkpoint/config.
3. Compare semantic outputs first (`audio_argmax`, cosine similarity, top-k
   overlap), then use layer and stage dumps to localize any drift.

Strict elementwise parity is useful as a diagnostic, but it is not the only
acceptance signal. The current 4090 evidence showed the semantic gate passing
while small BF16/F32 elementwise drift remained:

The RMSNorm rounding ablation showed that the HF/Qwen3-style fused-add-RMSNorm
variant improves strict trace parity slightly, but is not required for the
current one-step semantic gate. To keep this foundation PR focused, the shared
`pegainfer-kernels` rounding change is left out of scope and can be discussed
separately as a Qwen3 numeric-parity change if needed.

Evidence source: old-4090 semantic comparison logs from the Higgs-Audio
trace-driven bring-up run; the auto path and retained-session path reported the
same semantic metrics.

- `audio_argmax.ids`: exact.
- `hidden_cosine`: `0.999994874`.
- `logits_cosine`: `0.999998987`.
- `top64_min_overlap`: `58`.
- `top64_mean_overlap`: `61`.
- `final_hidden.bf16 mean_abs`: about `0.006735`.
- `audio_logits.f32 mean_abs`: about `0.040235`.

## Local Checks

Checks that do not require a Linux CUDA runtime:

```bash
cargo fmt --check -p pegainfer-core -p pegainfer-qwen3 -p pegainfer-higgs-audio
cargo check -p pegainfer-higgs-audio --bins
python3 -m py_compile \
  tools/accuracy/analyze_higgs_one_step_actual.py \
  tools/accuracy/analyze_higgs_projection_drift.py \
  tools/accuracy/analyze_higgs_qk_norm_drift.py \
  tools/accuracy/analyze_higgs_residual_drift.py \
  tools/accuracy/analyze_higgs_rmsnorm_drift.py \
  tools/accuracy/compare_higgs_layer_hidden.py \
  tools/accuracy/compare_higgs_stage_dump.py \
  tools/accuracy/compare_higgs_trace_dump.py \
  tools/accuracy/dump_higgs_layer0_stages_golden.py \
  tools/accuracy/dump_higgs_layer_hidden_golden.py \
  tools/accuracy/dump_higgs_one_step_golden.py \
  tools/higgs/check_higgs_gate_summary.py
```

Linux/4090 checks:

```bash
export PEGAINFER_CUDA_SM=89
cargo check -p pegainfer-higgs-audio --features runtime-qwen3 --bins
tools/higgs/run_higgs_one_step_cuda_gate.sh <higgs-model-dir> <golden.safetensors> <out-dir>
```

On macOS, the runtime-Qwen3 check is expected to stop before useful Rust type
checking because the workspace currently builds CUDA kernels and Linux RDMA
dependencies (`rdma-mummy-sys` expects Linux headers such as `endian.h` and
`linux/types.h`).
