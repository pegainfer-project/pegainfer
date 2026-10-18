# Kimi-K2 MoE EP: PPLX → DeepEP migration

> **TL;DR:** Implemented and 8×H200-verified — Kimi-K2's MoE EP backend is now DeepEP (elastic API, AOT-instantiated, no torch/NVRTC/NVSHMEM); PPLX is fully deleted from the kimi path (`moe_pplx.rs` gone, kimi crate no longer depends on `pegainfer-comm`). Decode = `do_expand=true` + `do_cpu_sync=false`: fixed worst-case buffers, zero host syncs/allocs per step → **CUDA graph capture enabled (#227): bs64 steady TPOT p50 26.03 ms vs 29.61 eager (−12%), replay only at full bucket occupancy**. Prefill = `do_cpu_sync=true` with host spin on pinned counters. Marlin consumes the DeepEP recv buffer **in place** (expert_alignment 8 == Marlin block size; identity routing + sentinels). Same-node A/B vs PPLX on hth200-29: **eager bs64 TPOT p50 29.61 vs 29.79 ms (parity), comm itself 7µs/layer faster**; golden gate equivalent to main (free-greedy near-tie reds on both backends, teacher-forced 0 violations both). The initial port was +14% TPOT slower until two capacity-proportional adapter kernels were fixed (see the lesson below).
>
> **Last touched:** 2026-06

## Architecture as built

```
pegainfer-kernels/
  third_party/DeepEP            # submodule d4f41e4 (2026-05-26)
  csrc/deepep/deepep_shim.cu    # AOT template instantiation (Kimi config baked:
                                #   384 experts / 48 local, topk 8, hidden 7168, 8 ranks)
                                # + torch-free host launch over cudaLaunchKernelExC
  src/ffi/deepep.rs             # repr(C) DeepEpInfo + extern decls
  src/ops/deepep.rs             # DeepEp wrapper: decode_dispatch/decode_combine (no sync),
                                #   prefill_dispatch_send/wait_counts/recv + prefill_combine
pegainfer-kimi-k2/
  src/runner/moe_deepep.rs      # the MoE layer:
                                #   forward_moe_layer_decode_deepep_normed  (host-quiet)
                                #   forward_moe_layer_prefill_deepep       (cpu-sync)
```

Build needs `PEGAINFER_NCCL_ROOT` pointing at NCCL ≥ 2.30 (device API: `ncclDevComm`,
windows, GIN). The binary links `libnccl.so.2` via `LD_LIBRARY_PATH` at runtime.
Local dev: `PEGAINFER_NCCL_ROOT=/data/opt/nccl-2.30.4`.

Backend selection: TP1/DP8 **requires** `--ep-backend=deepep` (default), TP8/DP1
requires `nccl` — both enforced with `ensure!` in `runner/bringup.rs`. There is no
PPLX fallback by design ("我们并不是很喜欢 pplx ep"). `pegainfer-comm`/PPLX survive
only for the deepseek crates, which use their own `pegainfer_comm::EpBackend` type.

## The contracts the integration stands on (verified in upstream source)

**psum_expert post-epilogue semantics** (`dispatch.cuh:209,251-254`,
`dispatch_copy_epilogue.cuh:117`): dispatch writes the *exclusive* prefix sum of
*aligned* per-local-expert counts (49 entries); the copy epilogue then `atomicAdd`s
+1 per real slot, so afterwards `psum[i] = aligned_start_i + real_count_i` and
`psum[48]` = total aligned expanded rows, untouched. The Marlin routing kernel
(`kimi_deepep_build_marlin_routing_kernel`, `<<<1,64>>>`) reconstructs
`start_i = round_up(psum[i-1], 8)`, `count_i = psum[i] − start_i`, and reads
`num_tokens_post_padded` straight from `psum[48]` — no counts-conversion kernel.

**Marlin in-place consumption**: DeepEP `expert_alignment=8` equals Marlin's block
size, so the expanded recv buffer (expert-major, per-expert 8-aligned segments) *is*
Marlin's "sorted+padded" format. Routing = identity `sorted_token_ids` with sentinel
(=capacity) on pad rows + per-block `expert_ids` + device-resident
`num_tokens_post_padded`. Pad rows compute garbage that combine never reads
(row-independent GEMMs, sentinel-guarded epilogue). Marlin core untouched.

**Weights/scale flow**: dispatch carries per-slot f32 router weights
(`recv_topk_weight`, same row space as `recv_x`); W2 applies them
(`mul_topk_weights=true, top_k=1`); combine sums **unweighted**
(`topk_weights=nullptr`, asserted upstream); `KIMI_K2_ROUTER_SCALE` is applied at
the residual via `kimi_residual_add_scaled_bf16` — same convention as the NCCL
backend. Net numerics delta vs PPLX: combine output is bf16 where PPLX returned
f32, i.e. the routed sum rounds to bf16 one step earlier. Must be re-gated on
hardware (vllm_golden_gate + det contract), expected within the bf16 ULP floor.

**Decode is host-quiet**: all buffers persistent and worst-case-sized at
`enable_deepep` (8528 expanded rows ≈ 350 MB/rank); per step there are zero
allocations, zero D2H syncs, zero `seq_len` mutations. `decode_combine` passes the
worst-case 1024 as `num_reduced_tokens` — that is the upstream *sentinel*
(`combine.cuh:43-44`): the kernel reloads the actual count from `psum_rank` on
device.

**CUDA graph capture (#227)**: enabled for the DeepEP decode arm, replay
**only at full bucket occupancy** (`active_len == decode_batch_size`, always 8
under the DP scheduler) — the captured graph bakes the row count, so partial
buckets run eager. No cross-rank barrier (unlike the TP8/NCCL arm): DeepEP
decode has no host-side collectives, and DP ranks reach full occupancy on
*different* steps — a shared barrier would deadlock. While one rank captures,
peers' device-side dispatch spins simply wait until its first graph launch;
the safety margin for that pause is the DeepEP device timeout
(`kTimeoutCycles` ≈ 100 s, traps on the *peer* rank) vs a capture window of
tens of ms — keep captures far below that ceiling. Replay numerics are
device-driven by construction (positions/page tables/tokens are uploaded into
persistent buffers *before* the graph region): 56/64 bench traces were
byte-identical to the eager runs' dominant trace families. Cooperative,
PDL, and cluster launches (the shim's full attribute set) all capture and
replay cleanly on CUDA 12.9 / driver 590.

**Prefill lock-step**: all 8 DP ranks call the layer fn simultaneously
(synchronized prefill pads idle ranks with 1 dummy token). Order per layer:
enqueue everything GPU-side (shared expert, router on aux joined via event,
`prefill_dispatch_send`) **before** the host spins on pinned counters
(`prefill_wait_counts`, 120 s throwing timeout). Capacity:
`recv_capacity = ep_max_seq_len + 7` (only the owner sends real tokens; 7 dummies),
`expanded_capacity = (recv_capacity·8 + 48·7).next_multiple_of(8)` — tight bound
(≤ 8 distinct local experts per token, ≤ 7 pad per expert); counts are `ensure!`d
against capacity before recv (crash early).

## Lesson: capacity-proportional adapter kernels ate the comm win

The first hardware run was **+4.1 ms TPOT (+14%) slower** than PPLX at bs64 —
yet the nsys A/B showed DeepEP's comm chain (dispatch + combine, all kernels)
was 7.3 µs/layer *faster* than PPLX's send/recv pairs. The entire regression
was two adapter kernels whose cost scaled with the worst-case capacity
(8528 expanded rows) instead of the actual rows (~512 at bs64); PPLX's
capacity was ~1100 rows so the same patterns never showed there:

| per layer per step | PPLX | DeepEP (first port) | DeepEP (fixed) |
| --- | ---: | ---: | ---: |
| comm total | 150.5 µs | 143.2 µs | 143.2 µs |
| Marlin routing builder | 5.1 µs | **34.0 µs** | ~5 µs |
| swiglu (expanded) | 5.5 µs | **42.4 µs** | ~5 µs |

- **Routing builder**: one thread per expert wrote its segment serially, and
  tid 0 wrote the `[total, capacity)` tail sentinels alone — ~8000 serial
  writes when actual ≪ capacity. Fix: `<<<1,1024>>>`, shared segment table,
  every thread grid-strides the full range with a binary search per slot.
- **Swiglu**: the kernel always early-exited past the device-side row count,
  but the *grid* was capacity-sized — 68k blocks draining for ~512 live rows
  cost more than the math. Fix: fixed occupancy-sized grid (`sm_count*8`)
  that grid-strides up to the device count.

Moral (same family as the #204 small-N lesson): with `cpu_sync=false` the
host never knows the real size, so **every** launch dimension derived from
capacity must be re-derived from the device count or made size-independent —
audit each adapter kernel for capacity-proportional cost, don't wait for the
bench.

## Known costs / next levers

- Prefill allocates ~5 device buffers per MoE layer (carried verbatim from the
  PPLX prefill); TTFT is 62.2 vs PPLX 60.2 ms p50 after the kernel fixes —
  this alloc churn plus the per-layer host spin is the remaining 2 ms.
- combine_impl med ≈ 91 µs vs ≈ 37 µs NVLink theoretical for the bs64 payload —
  same magnitude as PPLX's combine_recv (85 µs), so not a migration cost;
  it is the natural #228 overlap/tuning target.

## 8×H200 verification (hth200-29, 2026-06-07)

Same-node A/B, main `d0a0276` (PPLX) vs `feat/kimi-deepep` + kernel fixes,
in-process `bench_serving`, prompt 1 / output 128 / concurrency 64:

| bs64 | PPLX eager | DeepEP eager | DeepEP + graph |
| --- | ---: | ---: | ---: |
| steady TPOT p50 / p99 | 29.79 / 31.14 ms | 29.61 / 31.74 ms | **26.03 / 30.18 ms** |
| TTFT p50 | 60.2 ms | 62.2 ms | 59.0 ms |
| decode tok/s | 33.6 | 33.7 | **38.25** |

Graph run: all 64 requests generated the full 128 tokens; trace hash
distribution matches the eager runs' dominant families (30+26 of 64
byte-identical) — per-step positions and page tables are device-driven, not
baked into the capture.

`vllm_golden_gate`: teacher-forced 384 positions **0 violations both backends**
(DeepEP exact 97.9–98.2%, |Δlogprob| mean 0.030, p99 0.21); free-greedy is red
on **both** backends on this node at 2 near-tie positions each (DeepEP: json
pos5 + translation pos31; PPLX: long-zh pos20 + long-en pos31, 4.25 nat) — the
known #286 marginal class, not a migration regression. Det contract green.

Node env facts (also apply to other hth200 nodes until proven otherwise):

- `NCCL_NVLS_ENABLE=0` required: NVLS multicast bind fails with CUDA error 401
  (fabric manager runs, NVSwitch multicast state broken; driver 590/CUDA 13.1
  vs NCCL cuda13.2 build). Without it `ncclCommInitRank` → "unhandled cuda
  error".
- `EP_DISABLE_GIN=1` required (and correct for single-node): the GDAKI GIN
  plugin loads but deeper init fails without DOCA GPUNetIO; intranode traffic
  is NVLink windows, GIN is inter-node-only.
- System NCCL is exactly 2.30.4 (`/usr/include` + `/usr/lib/x86_64-linux-gnu`);
  `PEGAINFER_NCCL_ROOT` wants the `include/`+`lib/` layout — a symlink tree at
  `/data/opt/nccl-2.30.4` bridges it.
- The bastion swallows ssh exit codes — poll remote jobs with output markers,
  never `$?`. `pkill -f <pattern>` self-matches the ssh wrapper command line —
  kill by PID or match on `/proc/<pid>/cwd`.

## Next

1. Prefill alloc/spin cleanup (the +2 ms TTFT).
2. #228 overlap: combine_impl tuning (91 µs vs 37 µs theory).
3. #300 — gate coverage gap: `vllm_golden_gate` runs with
   `enable_cuda_graph: false` and its concurrent pass peaks at ~2 active/rank —
   the graph replay path has no dedicated numerics gate beyond the trace-hash
   evidence above. A full-bucket graph-vs-eager comparison is the missing test.
