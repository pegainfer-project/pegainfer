# Model Optimization Pipeline

> **TL;DR:** Standardized workflow for taking a new model from "it runs" to "it runs fast". Two fixed profiles, vLLM as baseline, per-model doc tracks the full journey.
>
> **Status:** Active.

## Standard Profiles

All models use two profiles that isolate the prefill and decode paths:

| Name | input | output | Isolates |
|------|-------|--------|----------|
| prefill-heavy | 2048 | 1 | TTFT, prefill kernel breakdown |
| decode-heavy | 1 | 128 | TPOT, decode kernel breakdown |

No mixed-workload profiles. Mixed performance can be inferred from the two pure paths, and mixed data is hard to attribute.

pegainfer: `scripts/bench_http_serving.py --prompt-words <in> --max-tokens <out>` against a running server (see [profiling-guide](profiling-guide.md)). vLLM: `vllm bench serve` (see [bench-vs-vllm](bench-vs-vllm.md)).

## Per-Model Doc Structure

One doc per model under `projects/`, containing three sections:

### 1. E2E Dashboard

Current pegainfer vs vLLM end-to-end numbers. **Update the pegainfer column after each optimization** to always reflect the latest state. The vLLM column only needs to be measured once at baseline — re-measure when upgrading vLLM.

```markdown
## E2E Dashboard

GPU: ..., Model: ..., vLLM version: ..., single concurrency.

| Profile | Metric | pegainfer | vLLM | delta |
|---------|--------|-----------|------|-------|
| prefill-heavy (2048,1) | TTFT median | ... | ... | ... |
| prefill-heavy (2048,1) | TTFT p99 | ... | ... | ... |
| decode-heavy (1,128) | TPOT median | ... | ... | ... |
| decode-heavy (1,128) | TPOT p99 | ... | ... | ... |
```

### 2. Model Architecture & Operator Coverage

Expand the model's computation graph first, then annotate pegainfer's support status.

**Architecture summary:** layer count, layer type distribution, key shapes.

```markdown
## Architecture

- Layers: 32 (24 linear + 8 full attention)
- hidden_dim: 2560
- Full attention: 16 q_heads, 4 kv_heads, head_dim=256, partial RoPE (rotary_dim=64)
- Linear attention: 16 q_heads, 16 k_heads, 32 v_heads, head_dim=128
- MLP: intermediate_size=9728
```

**Per-layer DAG:** Write out the operator sequence and shape flow for each layer type, separately for prefill and decode.

```markdown
### Full Attention Layer (prefill)

RMSNorm_offset [2560,seq] -> QKV GEMM [2560->8192+1024+1024,seq]
  -> QK Norm + Partial RoPE + KV Write + Attention [seq,seq] -> O GEMM [4096->2560,seq]
  -> Residual + RMSNorm_offset -> Gate GEMM [2560->9728,seq] -> Up GEMM [2560->9728,seq]
  -> SiLU*Mul [9728,seq] -> Down GEMM [9728->2560,seq] -> Residual
```

**Operator performance table:** For each DAG node, record nsys-measured time and hardware utilization. Important operators should include the theoretical limit and bottleneck type (compute-bound / memory-bound).

```markdown
| Operator | Time | % | Bound | Utilization | Notes |
|----------|------|---|-------|-------------|-------|
| cuBLAS GEMM (7x) | 163ms | 77% | compute | 21% MFU (90/419 TFLOPS) | at ceiling |
| FA2 | 36.5ms | 17% | memory | 47% BW (288MB/600GB/s) | GQA 4x redundant reads |
| ... | ... | ... | | | |
```

Model-specific operators go in the per-model doc. Cross-model operators (e.g. FlashAttention) can have their own doc under `resources/`.

### 3. Optimization Log

Append-only. Each entry records one optimization attempt (successful or failed).

```markdown
## Optimization Log

### #0 Baseline (date)

nsys kernel breakdown (prefill-heavy):
| Kernel | Time | % | Notes |
| ... | ... | ... | ... |

nsys kernel breakdown (decode-heavy):
| Kernel | Time | % | Notes |
| ... | ... | ... | ... |

### #1 Optimization name (date)

**Bottleneck:** which kernel, what % of total
**Approach:** what was done
**Changes:** which files were modified
**Result:** before -> after (with delta)
**E2E impact:** both profiles before -> after
```

The baseline entry (#0) includes the full kernel breakdown. Subsequent entries only record the changed kernels before/after — no need to repeat the full table.

## Guidelines

**One bottleneck at a time.** Changing two things simultaneously makes attribution impossible. Finish updating e2e data for the previous optimization before starting the next.

**Record failed attempts.** Failed attempts prevent future repetition and capture knowledge about hardware limits.
