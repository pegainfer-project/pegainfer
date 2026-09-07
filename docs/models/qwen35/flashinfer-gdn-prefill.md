# Qwen3.5 FlashInfer GDN Prefill

> **TL;DR:** Qwen3.5 supports an explicitly selected SM120 FlashInfer GDN
> prefill candidate with lower scratch requirements. Triton remains the default.
>
> **Last touched:** 2026-09

## Integration Boundary

The candidate supports TP1 and Hq/Hk/Hv/D = 16/16/32/128, validated with
Qwen3.5-4B. Linking an AOT bundle does not select it. Invalid bundles fail
during the build; unsupported or unavailable candidates and invalid runtime
artifact identities fail at model load without silently falling back.

`pegainfer-kernels` owns pinned generation, artifact validation and the
stable in-place C ABI. Generated symbols and TMA argument layouts stay
outside the model layer. Generation uses an independent FlashInfer checkout,
preserving the shared headers used by other models.

Backend selection precedes KV budgeting and reserves the selected backend's
peak prefill scratch. Normal prefill retains stream/event ordering without
diagnostic host barriers.

## Validation and Tradeoffs

Validation covers real build/load/HTTP paths, upstream-to-patched state-layout
parity, the native-prepare CPU oracle, and existing HF, continuation and
scheduler/CUDA Graph gates. Candidate entries require the expected artifact
SHA and check the loaded model; HF gates require a resolved model revision.
This acceptance setup stays in private test modules.

The candidate reduces prefill scratch requirements. Same-context HTTP
comparisons against seven-stage Triton, using synthetic token-ID prompts
with Shared-SM enabled, show modest throughput gains with workload-dependent
tail-latency regressions. These results do not establish natural-language
workload performance or justify changing the default backend.

Generation and gate details are documented in the
[kernel tool README](../../../pegainfer-kernels/tools/flashinfer_gdn/README.md).
