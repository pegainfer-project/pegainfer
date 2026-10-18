# Long-P Throughput Boundary And ZeRO-Prefill Lessons

> **TL;DR:** ZeRO-Prefill gives us a boundary for a future long-prefill cluster, not a router design or an implementation commitment.
>
> - **Want:** a long-P engine path that maximizes batch throughput once an external router has already selected long-prefill work.
> - **Avoid:** putting long/delta classification, batch admission policy, or router state inside pegainfer.
> - **Why:** long prefill can provide enough compute to hide expert-weight movement, while decode and delta-prefill have different latency and state constraints.
>
> **Status:** Discussion record. No implementation, measurement threshold, or connector API is committed here.

## Scope

This note records what we learned from "ZeRO-Prefill: Zero Redundancy Overheads in MoE Prefill Serving" ([arXiv:2605.02960](https://arxiv.org/abs/2605.02960)) and how it should shape future PegaFlow/PegaInfer planning for large MoE serving.

The assumed product shape is P/D separation with an external router. The router is responsible for deciding whether work belongs to long prefill, delta prefill, or decode. This document only describes what pegainfer should care about after the router has already handed it a long-prefill batch.

The goal is a reusable boundary record, not an implementation plan. Exact backend design, telemetry fields, measurement thresholds, and connector protocols are outside this document.

## Paper-Fact: Problem

ZeRO-Prefill targets throughput-oriented long-prefill workloads with very large batches and abundant prefix sharing. These workloads are different from latency-oriented chat decode: the objective is total batch throughput rather than per-request streaming latency.

The paper identifies three main waste sources in conventional MoE expert parallelism:

- Sub-saturated per-device compute and expert-routing stragglers.
- HBM pressure from expert weights, activations, and KV cache.
- Synchronous activation AllToAll communication on every MoE layer.

## Paper-Fact: AsyncEP Boundary

The paper's backend idea is AsyncEP. Instead of routing activations across GPUs for every MoE layer, it gathers the next layer's expert weights in the background while the current layer computes. When the next layer starts, top-k expert dispatch is local.

The overlap condition is expressed as a saturation threshold:

- `T = t_EP * F_GPU * gamma`
- `t_EP`: the slowest expert-weight transfer time for the layer.
- `F_GPU`: GPU compute throughput.
- `gamma`: a jitter buffer; the paper uses `1.2`.

If the per-GPU prefill compute for a batch is large enough to cover `T`, expert-weight movement can be hidden behind computation. If it is not large enough, weight streaming becomes visible latency rather than hidden work.

The paper also treats CPU-DRAM expert offload as a first-class option. In D2D-only mode, GPUs still need resident expert shards plus streaming buffers. In hybrid offload mode, expert backing can live in CPU pinned memory while GPUs keep only the current and near-future working set.

## Paper-Fact: KV Boundary

ZeRO-Prefill includes a KV-cache-free mode for prefill-only workloads that directly produce logits, embeddings, classification results, or verification scores. That mode is only valid when the request does not need to continue into decode.

## Derivation: Compute-Bound Batch

The paper's waste sources matter most when a long-prefill batch has enough work to make compute the dominant resource. In our P/D-separated roadmap, short delta-prefill and decode should not be assumed to satisfy the same condition.

For pegainfer, the first long-P goal is to keep selected long-prefill work compute-bound. Once the router has already selected a long-prefill batch, the engine should avoid fragmenting it into chunks that lose MFU or make expert transfer visible again.

**Want:** execution that preserves enough per-GPU prefill work to make long-P throughput the main objective.

**Avoid:** treating long-prefill work like latency-first delta-prefill or decode work after it reaches the engine.

**Defer:** exact batch shape, chunking thresholds, and admission policy. Those belong in future measurement work and router policy, not in this boundary record.

## Derivation: Expert Weight Movement

The second long-P goal is to move expert-weight transfer out of the critical path when the batch has enough compute to hide it. ZeRO-Prefill's AsyncEP is one candidate design for this, but the boundary is more important than the specific mechanism.

**Want:** engine-side measurements and execution paths that can tell whether expert transfer is hidden by current-layer compute.

**Avoid:** committing to AsyncEP or a separate long-P backend before measuring current long-prefill MFU and per-layer breakdown.

**Defer:** whether the eventual backend uses AsyncEP-style D2D gather, hybrid CPU offload, or a different MoE execution strategy.

## Derivation: HBM And KV

For P/D-separated serving, a long-P cluster that hands work to a decode cluster must preserve the KV state or an equivalent handoff artifact. KV-cache-free execution is therefore a separate service shape, not a drop-in optimization for P-to-D handoff.

The third long-P goal is to control HBM pressure without breaking P-to-D handoff. Long prefill can stress HBM through expert weights, activations, and KV state at the same time.

**Want:** separate treatment for two service shapes:

- Long-P to D handoff, where KV or an equivalent artifact must be retained.
- Prefill-only services, where KV-cache-free execution may be valid.

**Avoid:** enabling KV-cache-free execution on any path that must continue into decode.

**Defer:** exact HBM residency policy, offload policy, and P-to-D KV handoff protocol.

## Derivation: Observability

The fourth long-P goal is to make throughput bottlenecks visible before choosing a backend direction. The first practical question is not "should we implement ZeRO-Prefill?", but "where does current long-prefill throughput go?"

The future measurement spec should explain at least:

- Overall MFU for long-prefill batches.
- Per-layer compute time.
- Attention time versus MoE time.
- Existing activation communication cost.
- Grouped expert GEMM saturation.
- Expert-token distribution.
- HBM peak and residency pressure.
- Whether communication is hidden behind compute or exposed on the critical path.

**Want:** measurements that distinguish compute saturation, communication exposure, and HBM pressure.

**Avoid:** using one aggregate throughput number as the only decision input.

**Defer:** numeric action thresholds. The measurement spec should own those thresholds; this document only records why they are needed.

## Derivation: Future Reuse

When PegaFlow/PegaInfer planning revisits long-prefill MoE serving, use this note as the entry point:

1. Assume the router has already selected a long-prefill batch.
2. Evaluate whether the engine keeps that batch compute-bound.
3. Measure whether expert-weight movement is hidden or exposed.
4. Keep KV-retaining P-to-D handoff separate from KV-cache-free prefill-only services.
5. Use data from a long-prefill measurement run before committing to an AsyncEP-style backend split.

This keeps the long-P document aligned with the decode-side DPLB record: engines should provide accurate execution facts, while external routing layers own classification and policy decisions.
