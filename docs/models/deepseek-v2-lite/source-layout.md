# DeepSeek-V2-Lite Source Layout

> **TL;DR:** `runtime.rs` was split by responsibility while preserving the narrow DeepSeek-V2-Lite HF / host-staged / NCCL EP2 exactness gate.
>
> **Status:** complete for source layout; NCCL CUDA Graph smoke remains a diagnostic blocker on the 2x RTX 5090 validation host.
>
> **Last touched:** 2026-06

## Why This Exists

This doc is the audit trail for a behavior-preserving refactor. The code change
is intentionally structural: it reduces the 1745-line `runtime.rs` file without
claiming a new NCCL optimization, CUDA Graph fix, or production EP milestone.

The doc is kept because DeepSeek-V2-Lite still has several easy-to-misread
boundaries:

- `host-staged` is still the correctness scaffold and regression oracle.
- NCCL is covered by the same HF / host-staged / NCCL JSON comparison gate.
- attribution and CUDA Graph readiness reports are diagnostic evidence.
- issue #275 device-resident / graph-friendly NCCL work remains a follow-up.

## Layout

`pegainfer-deepseek-v2-lite/src/runtime.rs` is now a facade that keeps the
public generator and result exports stable. Implementation moved into:

| File | Responsibility |
| --- | --- |
| `runtime/backend.rs` | EP backend env parsing, backend runtime enum, EP2 device validation. |
| `runtime/types.rs` | generation result/stat structs and decode graph readiness report data. |
| `runtime/generation.rs` | load, greedy generation, prefill/decode orchestration. |
| `runtime/layers.rs` | embedding, layer forward, MLA attention, sampling, norm helpers. |
| `runtime/moe.rs` | host-staged and NCCL MoE paths plus expert forwarding helpers. |
| `runtime/readiness.rs` | CUDA Graph readiness status, blocker list, graph-smoke wiring. |
| `runtime/helpers.rs` | token hash, same-prompt row check, duration, EOS append helper. |
| `runtime/tests.rs` | existing runtime helper tests. |

## What Stayed

- Public exports from `pegainfer-deepseek-v2-lite/src/lib.rs` still expose
  `DeepSeekV2LiteEp2Generator`, `GenerationResult`,
  `BatchedGenerationResult`, `GenerationStats`, and
  `DecodeGraphReadinessReport`.
- `host-staged`, NCCL, attribution, graph-smoke, HF dump, JSON comparison, and
  `tests/e2e_ep2.rs` stayed in place.
- Helper tests stayed because they cover real invariants: EOS handling,
  same-prompt batch equality, duplicate device ordinals, and backend parsing.
- No issue #275 device-resident NCCL combine changes were mixed into this
  source-layout refactor.

## Verification

Local checks after the split:

```bash
git diff --cached --check
cargo fmt --all --check
```

Both passed.

Remote validation ran on Ubuntu 22.04 with 2x NVIDIA GeForce RTX 5090, driver
580.105.08, CUDA 12.8, `PEGAINFER_CUDA_SM=120`, and a configured
`PEGAINFER_TRITON_PYTHON` environment.

Passed gates:

- `cargo check --offline --release -p pegainfer-deepseek-v2-lite --features deepseek-v2-lite --lib --tests`
- HF oracle dump with `tools/accuracy/hf_dump_dsv2_lite_ep2_greedy.py`
- host-staged `tests/e2e_ep2.rs`
- NCCL `tests/e2e_ep2.rs` using an explicit compatible NCCL library selector
- `tools/accuracy/compare_dsv2_lite_ep2_outputs.py --require-all-exact`
- NCCL attribution without graph smoke

Comparison result:

| Source | Backend | Token SHA256 | Text SHA256 |
| --- | --- | --- | --- |
| HF | - | `4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225` | `0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347` |
| host-staged | `host-staged` | `4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225` | `0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347` |
| NCCL | `nccl` | `4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225` | `0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347` |

The generated text for all three was:

```text
, I am a 19 year old girl from the UK. I am
```

## Remaining Boundary

The optional NCCL graph-smoke diagnostic still fails during capture on the
2x RTX 5090 container, even with `NCCL_GRAPH_REGISTER=0`,
`NCCL_CUMEM_ENABLE=0`, and `NCCL_GRAPH_MIXING_SUPPORT=0`.

Observed failure:

```text
misc/strongstream.cc:357 NCCL WARN Cuda failure 'operation failed due to a previous error during capture'
DeepSeek-V2-Lite --nccl-graph-smoke failed: ... ncclUnhandledCudaError
```

This does not change the correctness result above. It means graph-smoke work
should stay in a separate measured follow-up.
