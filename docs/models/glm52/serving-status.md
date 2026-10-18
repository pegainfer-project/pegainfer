# GLM5.2 serving status

> **TL;DR:** GLM5.2 is **Blackwell-only** (compute capability ≥ 10). Decode serving is EP4 / EP8 / one-domain EP-N with SM100 DeepGEMM masked grouped expert GEMMs; TP4 is **prefill-only** (NCCL). Hopper (SM9x), decode TP8/TP4 LL, and the old SM90 DeepGEMM path are removed. Continuous batching, whole-step CUDA Graphs, sampling, DSpark, paged KV, prefix caching, host offload, and target-only vLLM→PegaInfer P/D remain on the EP decode path. The line stays Bring-up until long-context indexer correctness and lifecycle reliability are closed.
>
> **Last touched:** 2026-07

## Current shape

GLM5.2 is a model-owned distributed serving engine. Launch fails closed on Hopper and older GPUs. The project tier remains **Bring-up** until the correctness and reliability contracts below are continuously enforceable.

### Topologies

| `--moe-topo` | Intended use | Evidence boundary |
| --- | --- | --- |
| `ep8` | Default high-throughput decode (8 GPUs / multi-tray EP8) | Strongest feature coverage on the free-running EP path: bucketed continuous batching, DSpark, prefix cache, offload, P/D. The SM100 DeepGEMM chain has cross-process EP8 execution coverage. |
| `ep4` | Throughput decode on 4×GB300 | SM100 DeepGEMM numerical oracle, automatic VRAM-budget, and four-bucket graph-capture gates pass. Same-session replacement A/B improved decode p50 by 3.64% and aggregate throughput by 3.9%. |
| `ep16` / `ep32` / `ep64` | Scale within one NVLink/IMEX domain | Per-width DeepEP shims + SM100 DeepGEMM template instantiations; multi-process via `--glm52-ranks` + `--glm52-rendezvous`. The current replacement was execution-tested at EP4 and EP8, not these wider widths. |
| `tp4` | **Prefill-only** on 4×GB300 | Requires `--glm52-prefill-only` (and `--tp-size=4`). Layer-outer NCCL bf16 all-reduce path; no decode CUDA graph / no LL packet MoE. See `tp4-prefill-only.md`. |

**Removed (no longer parse / fail at launch):**

- Hopper SM9x targets (including 8×H200 as a supported floor)
- `--moe-topo=tp8` and decode-time TP LL (phase MoE + attention AR)
- TP4 as a decode topology (decode used FlashMLA/LL; only prefill-only remains)

Historical measurement records for removed paths stay under `docs/models/glm52/` (e.g. `moe-tp8-low-latency.md`, older Hopper EP8 notes) but are not launch contracts.

See `tp4-prefill-only.md`, `ep4-gb300.md`, `free-running-dp.md`, and `cross-node-scaling.md` for active topology records.

## Serving capabilities

| Area | Current contract |
| --- | --- |
| Hardware | Blackwell (SM ≥ 10.0); multi-process EP probes only **local** GPU ordinals |
| Scheduling | Up to 8 slots per logical EP rank; `{1,2,4,8}` whole-step graph buckets; least-loaded admission |
| Attention | DSA indexer plus sparse MLA decode (FlashMLA SM100 / FlashInfer on TP4 prefill heads=16) |
| Sampling | `temperature`, `top_p`, `top_k`, `min_p`, and engine-level `seed`, honor-or-reject |
| Speculation | DSpark greedy and sampled verify on EP decode; span 4 default; verify spans reuse decode buckets |
| KV | 64-token paged pool, full-lifetime admission, prefix cache on by default |
| Offload | PegaFlow host-tier save/restore behind `--kv-offload` (EP; not TP4 prefill-only without native MTP) |
| P/D | Native-MTP handoff: PegaInfer TP4 prefill-only → EP decode (`first_step=verify`); the vLLM-prefill compat path is removed |
| Observability | Per-logical-partition running/waiting/KV gauges and decode graph export (EP) |
| Cross-node EP | One process per node hosting its own ranks (`--glm52-ranks` + `--glm52-rendezvous`); free-running per-rank engines; DeepEP is the only runtime coupling |

The supported P/D path is the native-MTP handoff (`pd-native-mtp-handoff.md`). The retired vLLM-prefill target-only path (#657) was removed along with its doc; DSpark draft state was never part of any transfer protocol.

## Sampling and API limits

The model engine supports `temperature`, `top_p`, `top_k`, `min_p`, and `seed` on both plain and speculative paths. Engine-level seeded replay is deterministic for the same occupancy timeline.

The following surfaces are not part of the GLM5.2 contract:

- `logprobs`, prompt logprobs, and `n > 1`;
- presence, frequency, and repetition penalties;
- GLM-specific guarantees for stop strings, stop token IDs, or `min_tokens` beyond the shared frontend behavior.

HTTP `seed` is still lost in the shared frontend before reaching the engine. Bucket changes can also alter floating-point association, so a greedy request may diverge at a near-tied token when its occupancy timeline changes. Runs with the same request and bucket timeline remain deterministic.

## Promotion blockers

### 1. Reproducible long-context correctness

Issue #541 is the main tier blocker. The indexer oracle once passed against a moving Transformers development reference, but that reference changed and the result is not reproducible. The current engine has passed end-to-end 4k/8k/16k NIAH, yet that probe cannot replace a pinned sparse-index selection gate.

The padded-vocabulary contract is also under repair in #680/#698. The checkpoint contains token IDs the frontend tokenizer cannot decode; every EP, sampling, and DSpark token-producing path must be structurally bounded to the decodable prefix.

### 2. Request lifecycle reliability

Issue #551 records one request that entered the frontend but never reached a terminal engine event. More than 3,500 later requests and extended soaks did not reproduce it. It remains a background reliability boundary until a trace identifies the cause or a sufficiently strong retained soak demotes it.

### 3. Feature composition

DSpark is mutually exclusive with prefix caching, host offload, and P/D. A prefix hit skips the target forwards that normally produce DSpark's historical auxiliary state. Issue #590 must first measure a position-correct boundary cold start before the project considers transferring the additional draft K/V payload.

In the multi-process cross-node shape, KV offload registers each node's local arenas on its own pegaflow host under the shared deterministic namespace.

## Performance work

Measured open work is topology-specific:

- The SM100 DeepGEMM replacement measured a same-session EP4 decode p50 improvement of 3.64% and aggregate throughput improvement of 3.9%. Re-profile this chain before carrying forward any expert-kernel optimization from the retired weight-only path.
- The cross-process EP8 execution smoke measured approximately 22.3 ms p50 and 358–359 tok/s aggregate; it was not an EP8 A/B.
- #582: graph the DSpark draft round only after its fixed launch cost matters; it is currently a small fraction of the verify round.
- Older Hopper EP8/TP8 investigations (#542/#559/#569/#608/#625) are historical evidence only — do not implement without a matched A/B on the **current Blackwell** topology.

No optimization should be carried forward from historical records without a matched A/B on the current topology.

## Background work

- #587: expose active slots, current bucket, and queue depth in addition to the scheduler gauges already shipped.
- PegaFlow metaserver recovery: republish the existing block catalog after reconnect; new saves recover today, old remote prefixes do not.
- General scale-out beyond a single NVLink/IMEX domain: preserve the per-node-process contract (free-running ranks + bootstrap rendezvous), but use a data plane designed and measured for IB/RoCE rather than treating the one-rack result as universal.
