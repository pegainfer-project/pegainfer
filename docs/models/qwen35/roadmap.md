# Qwen3.5-4B Roadmap

> **TL;DR:** Qwen3.5-4B has the main correctness, admission, chunked-prefill,
> sampling, and step-tail gates needed for the dense single-GPU roadmap. The
> retained RTX 5090 #469 serving sweep is now the current HTTP performance
> boundary: zero failed benchmark requests, but vLLM is faster across the
> retained envelope. Next work should attribute the HTTP/frontend/scheduler gap
> before changing kernels, then keep mixed-load and HTTP lifecycle evidence
> separate.
>
> **Last touched:** 2026-07

Tracking issue: `[Model] Qwen3.5 dense roadmap v2: Stable single-GPU serving`
(#654). It supersedes #249. Sibling maturity bar:
`docs/models/qwen3/roadmap.md` and `docs/models/qwen3/serving-perf-5090.md`.

## Maturity Target

The product boundary here is dense BF16 Qwen3.5 4B/9B/27B on one GPU. 4B is
the serving-performance anchor. Larger sizes inherit correctness gates, not 4B
performance or production-readiness claims.

Qwen3.5 needs a Qwen3-like serving story with hybrid-state constraints called
out:

| Dimension | Target | Qwen3.5-specific constraint |
| --- | --- | --- |
| Correctness | Short and long HF logits gates stay green for enabled paths | State is full-attention KV plus recurrent f32 plus conv bf16 |
| Serving stability | Requests are served, deferred, rejected, or recovered explicitly | Batch-level errors must not poison unrelated rows |
| Performance | Same-host HTTP comparison names GPU, versions, workload, concurrency/QPS, and output sanity | Historical single-concurrency parity is not the current claim |
| Mixed load | Long prefill overlap with active decode is measured, including saturated/failure cells | Chunking bounds stalls but does not reduce total prefill work |
| Prefix reuse | One valid boundary restores KV, recurrent, and conv state together | KV-only prefix cache is incorrect |
| Parallelism | TP starts from a design and correctness gate | Recurrent/conv state sharding is part of the design |

## Current State

| Area | State | Evidence |
| --- | --- | --- |
| Accuracy | Done: 4B/9B/27B short and long HF bf16 logits gates | `docs/models/qwen35/accuracy.md`, #186, #250, #654 |
| Admission / long context | Done: full-lifetime KV accounting and explicit context/KV rejection | `docs/models/qwen35/kv-admission.md`, #254, #290, #654 |
| Prefill | Done: direct paged writes, bounded scheduler chunking, and resumed `base_pos > 0` coverage | #252, #305, #313, #314, #333, #375, #654 |
| Sampling / step tail | Done: mixed batched sampling and batched final norm/lm_head/token selection | #284, #353, #654 |
| Serving-vs-vLLM evidence | Retained #469 RTX 5090 sweep exists. PegaInfer completed every cell with zero failures, but vLLM 0.25.1 is faster across the retained HTTP envelope. | `docs/benchmarks/qwen35-4b-serving-vllm-rtx5090-2026-07.md`, #469 |
| Serving overhead | Open: direct c16 TPOT is close to vLLM HTTP c16, while PegaInfer HTTP c16 is slower. Start with HTTP/frontend/scheduler/event attribution. | #469, #654 |
| Mixed-load evidence | Open: prove injected prefill overlaps active decode and keep starvation setups as negative controls | #470 |
| HTTP lifecycle | Open: retain cancel, disconnect, overload, rejection, recovery, health, and memory-return evidence | #471 |
| Fault isolation | Open risk: batch-level execution errors can still fail multiple active requests | #654 |
| Prefix reuse | Open: bounded joint KV/recurrent/conv snapshot design and implementation | #257 |
| DFlash | In flight and opt-in: correctness-first work must stay default-off until gates pass | #434, PR #626, #654 |
| Tensor parallel | Done through Phase 2b: eager dense TP2, mixed-step unified execution, and sharded linear/GDR state, verified on 2× RTX 4090 for 9B and 27B; TP CUDA Graph + perf gates remain | `docs/models/qwen35/tp-implementation.md`, #446 |

## Active Contract

Correctness:

- HF logits gates are the primary oracle.
- Exact text output is not a portable correctness gate.
- Qwen3.5 state moves, caches, rolls back, or transfers as one transaction:
  paged full-attention KV, recurrent f32 state, conv bf16 state, and logical
  position.
- Any chunking, sampling, prefix, speculative, or TP change must preserve the
  short and long gates.

Serving:

- Direct diagnostics, profiler attribution, HTTP pressure, and serving claims
  stay labeled separately.
- One request's KV, prefill, context, sampling, or execution failure must not
  silently lose or wedge unrelated requests.
- Lifecycle evidence needs cancel, disconnect, overload, rejection, follow-up
  completion, health, and memory-return checks.
- A performance win does not establish production readiness.

Performance:

- The retained #469 sweep is the current HTTP boundary for 4B on 1x RTX 5090:
  PegaInfer passed the matrix with zero failures, but did not match vLLM.
- New claims must include GPU, driver, CUDA/toolchain, PegaInfer commit, vLLM
  version, model revision, serve flags, bench flags, workload, concurrency/QPS,
  completed/failed counts, average output tokens, and output sanity/hash.
- The 1024/256 c16 direct diagnostic is attribution evidence only. It suggests
  looking at HTTP/frontend/scheduler/event overhead before a kernel-first
  rewrite.
- Product-mode prefix cache results are not part of the #469 main comparison.

## Roadmap V2

### Now

1. **Finish #469 as a retained benchmark record.**
   - Keep the #469 doc as a benchmark snapshot, not a parity claim.
   - Preserve failed, timeout, unsupported, OOM, and overload cells if any appear
     in follow-up runs.
   - Use the retained sweep to update README wording only with the full setup
     and non-parity boundary attached.

2. **Make #470 a valid mixed-load ITL gate.**
   - Prove injected prefill overlaps active decode.
   - Keep starvation or capped-concurrency setups as negative controls.
   - Record saturated, rejected, failed, timeout, or OOM cells instead of
     dropping them.

3. **Add #471 as the HTTP lifecycle gate.**
   - Retain cancel, disconnect during prefill/decode, explicit rejection,
     admissible overload, post-pressure completion, health, and GPU memory
     return.
   - Keep the recovery request and output hash in the artifact.

4. **Finish DFlash correctness without changing default serving claims.**
   - Accept-all, accepted-prefix, reject-first, fallback, long-context, memory
     reservation, and joint KV/recurrent/conv restore all need real GPU gates.
   - DFlash remains opt-in and default-off until correctness and tail behavior
     are retained.

### Next

1. **Attribute the #469 HTTP gap.**
   - Start with 1024/256 c1/c16 and QPS 8/12/16.
   - Trace queue wait, scheduler planning, scheduled-to-first-token,
     prefill/decode step time, decode batch size, send/event overhead, terminal
     latency, and idle gaps.
   - Move to kernel profiling only if GPU step time explains the serving gap.

2. **Close only measured 4B serving gaps.**
   - Promote changes from repeated same-host HTTP A/B runs.
   - Do not trade away correctness gates, lifecycle recovery, or mixed-load ITL.

3. **Implement prefix reuse after #257 settles the boundary.**
   - Snapshot format, eviction/salt semantics, hit observability, and warm-TTFT
     benchmark are separate acceptance surfaces.
   - If the joint-state snapshot is unbounded or too expensive, record the
     no-go condition.

### Later

- TP for a concrete 9B or 27B TP=2 target, then 4B TP only if same-host TP=1 vs
  TP=2 shows value.
- LoRA with real-adapter logprob parity.
- Exporter-specific FP8/NVFP4/MXFP4 loading and accuracy lanes.
- MTP/DSpark under a separate speculative path.
- Hybrid-state offload and P/D under a separate transfer/restore contract.
- MoE and VLM under separate model roadmaps.

## Done Criteria

Qwen3.5 reaches the dense single-GPU stable bar when:

- Cancel, disconnect, overload, rejection, and a clean follow-up request stay
  green without wedges, silent loss, or leaked request state.
- Mixed-load cells prove the intended prefill/decode overlap and retain failure
  accounting.
- 4B c1/c4/c8/c16 and QPS throughput/TPOT match or beat the pinned vLLM
  baseline inside the documented RTX 5090 envelope.
- Short and long HF gates remain green for 4B/9B/27B across every enabled mode.
- Prefix reuse restores one valid joint KV/recurrent/conv boundary, survives
  eviction and memory pressure, and preserves cold-serving behavior when the
  snapshot pool cannot insert.
- DFlash, when enabled, passes real GPU correctness and same-commit A/B gates.
- Unsupported size, context, sampling, topology, cache, and speculative
  combinations fail explicitly.
- No open blocker remains for the documented single-GPU dense stable claim.
