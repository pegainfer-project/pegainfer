# GLM5.2 fp8 blockwise GEMM lab (kernel_lab sm_103 line)

> **TL;DR:** the kernel_lab fp8 blockwise GEMM experiment line ships 8 units: the production CUTLASS wide route `fp8_gemm.{q_b,o_proj,shared_gate_up,shared_down}` (`glm52_fp8_groupwise_gemm_sm100_cuda`, sm_100a-family only; GB300 sm_103 baselines at rows=64 are 19.77 / 55.77 / 24.03 / 12.92 us, check rel_l2 <= 9.1e-5) plus the CuTe DSL tcgen05 lab line `fp8_gemm_dsl_tc.*` (TMA + tcgen05.mma + TMEM block accumulator + software blockscale, per-shape tile-N/split-K). The DSL variant passes 28/28 on GB300 (rows axis {4..96}, rel_l2 <= 9.1e-5, SASS-audited genuine tcgen05) and beats the CUTLASS baseline on all four shapes: **1.37–1.97x** at rows=64 (best 9.45us / shared_down; tuning round 2 — tile-N/split-K against CTA starvation — adds o_proj −32%, gate_up −40%, down −21%) and **1.34–1.76x** at rows=96 (two M tiles, own `TILE_CFG_MULTI_M` regime). The win survives cold-L2 / CUDA-graph rechecks. Production integration shipped (#835 + #845): build-time `export_to_c` AOT dispatched inside `glm52_fp8_groupwise_gemm_sm100_offset_launch` for the wide-route decode buckets 16–96, measured −8.7~−9.9% whole-step p50 in the wide regime on the ep4 step bench.
>
> **Last touched:** 2026-08

## What this line ships

- **Harness core** (`pegainfer-glm52/benches/kernel_lab/`): manifest-driven registry (the manifest IS the behavior domain — delete a TOML and the unit is gone, no list code to touch), a ctypes loader (dlopens `target/release/libglm52_kernel_lab.so` with `RTLD_LAZY`; DeepEP shim NCCL symbols stay unresolved in the .so and these units never call into them), timing (torch.cuda.Event warm protocol + quack-style `--cold-l2` buffer rotation; cold is mutually exclusive with `--save` to protect ledger comparability), and a ledger (`baselines/<unit>.json` bucketed by `(shape, arch)` — numbers never migrate across arch buckets by contract).
- **Build chain**: with `PEGAINFER_KERNEL_LAB=1`, `pegainfer-kernels/build.rs` `nvcc -shared`-links the *same objects and same nvcc flags* as the production static archive into `libglm52_kernel_lab.so` and copies it to `target/<profile>/`; without the env var a default build runs zero extra commands and zero extra link lines.
- **Units**: `fp8_gemm.*` x4 (ctypes straight to the production CUTLASS symbol) + `fp8_gemm_dsl_tc.*` x4 (`capability.python_native` Python JIT units; needs the `.venv`'s `nvidia-cutlass-dsl 4.6.0`). Input pairs are fully same-source across both families (`make_inputs` delegates to fp8_gemm_groupwise: act gets the production per-token-group quantization, weight the 128x128 block quantization recipe, both e4m3 + plain f32 scales, ue8m0 off), one torch reference, one rel_l2 <= 0.02 gate.
- **Capability gating**: `fp8_gemm.*` carries `blackwell_only` (fail-closed outside major>=10; the .cu additionally compiles an `801` stub outside the sm_100a family, so KernelLaunchError is a second belt). `fp8_gemm_dsl_tc.*` carries the stricter `sm_tcgen05_only` (`__main__._setup` fail-closes with "tcgen05 (SM100+) only" whenever the device major != 10) — `blackwell_only` alone would permit Blackwell parts that have the tensor-core ISA but no tcgen05 hardware units, where the run would die deep inside the DSL runtime.
- **Tests**: `pegainfer-glm52/benches/tests/` CPU-only pytest (never imports torch): registry/manifest/schema invariants, fp8 unit metadata vs production anchors (`whitelisted_linear_shape` / `FP8_GEMM_WORKSPACE_BYTES` / the `extern "C"` signature), and a latch test for the tcgen05 gating keys.

Usage and environment: `pegainfer-glm52/benches/README.md`; operating and extension discipline: `kernel-lab-ops.md`.

## CUTLASS sm_103 baselines (the production route's current ledger)

All four `fp8_gemm.*` units pass `check` over the full rows axis {4,8,16,32,48,64} (GB300 sm_103, tray03, 2026-08-01; measured rel_l2 <= 9.1e-5 against the 4e-3 bf16 store floor, ~220x headroom). rows=64 median us (warm event protocol, arch=sm_103 ledger bucket):

| Unit (n x k) | us | Note |
|---|---|---|
| fp8_gemm.q_b (16384x2048) | 19.77 | 128 128-N CTAs, SMs nearly fed |
| fp8_gemm.o_proj (6144x16384) | 55.77 | |
| fp8_gemm.shared_gate_up (4096x6144) | 24.03 | only 32 CTAs, CTA-starved |
| fp8_gemm.shared_down (6144x2048) | 12.92 | 48 CTAs |

This table is both the DSL line's comparison anchor and the subtrahend in the production-profit section below.

## The DSL tcgen05 line (`fp8_gemm_dsl_tc.*`, sm_103 native)

**Implementation choices** (`kernel_lab/units/fp8_gemm_dsl_tc_kernel.py`, `Fp8BlockwiseGemmTcgen05`): the semantic skeleton follows the vendored CUTLASS 4.5 `examples/python/CuTeDSL/blackwell/blockwise_gemm/blockwise_gemm.py` — **TMEM block accumulator MMA per 128-K block (first MMA issued with `Field.ACCUMULATE=False` to self-clear) -> t2r back into registers -> `final += block * (SFA*SFB)`**, bit-isomorphic to CUTLASS `Sm100BlockwiseScaleConfig<1,128,128,K,K>`. Everything else was stripped: no persistent tile scheduler (the grid is only 32–128 CTAs), no 2-SM MMA / cluster / multicast, no cp.async scale warp ring, no TMA-store epilogue. What stays: `PipelineTmaUmma` + `PipelineUmmaAsync` (double-buffered TMEM acc) + the warp specialization. Configuration: tile 64x128x128, 160 threads (4 acc/epilogue warps + 1 TMA/MMA warp), TMEM acc stages and AB smem stages auto-sized, registers -> bf16 -> row-predicated SIMT STG straight to gmem; rows<64 rides TMA zero-fill + epilogue row predicates. JIT via `cute.compile` cached per `(rows,n,k)`; compilation is fully local/offline (the wheel embeds ptxas; `CUTE_DSL_ARCH=sm_103a` cross-compiles without a sm_103 card).

**Pitfalls (in scuff order, all acceptance-backed)**

- **The v1 -> v2 5.3x**: v1 (per-element guarded LDG for scales + one shared set of t2r registers) ran o_proj at 153.7us — 2.8x *slower* than CUTLASS. Decisive experiment: acc stages 2->4->8 changed nothing, so the limiter was not MMA<->fold overlap but a ~1us per-k-tile **serial latency chain** inside the acc warps. Three fixes: (1) full-vector one-shot preload of SFA/SFB into smem (zero scale LDG in the mainloop); (2) back-to-back emission of the two (64,64) sub-tiles' `tcgen05.ld` with independent registers and a single wait_ld; (3) acc stages up to 4. The mainloop dropped to MMA-hidden latency.
- TMA wants rank-3 tensors ((M,K,L) modeling): the runner `unsqueeze(-1)`s to L=1.
- TmemAllocator (DSL 4.6) keeps the 4.5 usage: all threads `allocate(cols)` -> `wait_for_alloc()` -> `retrieve_ptr(f32)`, closing barrier then `free()`.
- sm_103 SASS tooling: cuobjdump needs an external nvdisasm for sm_103a — use the pip wheel's `nvidia/cu13/bin/nvdisasm`.

**Acceptance (GB300 tray17, all green)**: `check` passes 24/24 over the full rows axis (rel_l2 <= 9.1e-5, cosine=1.0 — same magnitude as the CUTLASS baselines, pure accumulation-order noise). rows=64 median us (warm event, clocks.sm 2070 MHz, same protocol as the baseline table):

| Unit (n x k) | DSL tcgen05 | CUTLASS sm_103 | Speedup |
|---|---|---|---|
| q_b (16384x2048) | **10.03** | 19.77 | 1.97x |
| o_proj (6144x16384) | **28.80** | 55.77 | 1.94x |
| shared_gate_up (4096x6144) | **12.77** | 24.03 | 1.88x |
| shared_down (6144x2048) | **9.45** | 12.92 | 1.37x |

(v1 first-round numbers for the record: 15.52 / 153.66 / 36.11 / 15.39 — o_proj's 153.66->28.80 is exactly the scale-preload + back-to-back-t2r pair.)

**SASS + PTX evidence** (answering "is the DSL win an artifact"): the o_proj cubin (sm_103a, 1326-line SASS) shows per k-tile 4 tcgen05-MMA instructions (sm_103 SASS form; the first with accumulate disabled, self-clearing the block accumulator) + 2x `UTMALDG.2D` + the mbarrier SYNCS group; tmem reclaim is `LDTM.16dp256bit.x8 x2`; the epilogue is 32x `F2FP.BF16.F32.PACK_AB` + 32x `STG.E`. Fallback-evidence zero hits: **HMMA = 0, mma.sync = 0, F2FP.E4M3 = 0**. The PTX cross-check (local offline `CUTE_DSL_ARCH=sm_103a` recompile) shows `tcgen05.mma.cta_group::1.kind::f8f6f4` x4/k-tile + `tcgen05.{alloc,ld,commit,dealloc}` + TMA `cp.async.bulk.tensor`; `mma.sync` zero hits. Resources `REG:128, LOCAL:0, STACK:0` — no spills. Lesson (verified again while writing this line's docs): **retargeting a DSL kernel across arches is not "hardware-native" just because it compiles — always SASS-audit the retarget**; ptxas is allowed to silently emulate instructions that have no hardware path.

**Cold-L2 recheck** (`bench --cold-l2` = a quack-style buffer rotation of >= 3x L2; the ledger stays a warm-protocol series and cold refuses `--save`): the GB300 tcgen05 numbers move only slightly warm->cold at rows=64: q_b 10.03->10.73, o_proj 28.80->29.88, gate_up 12.77->13.94, down 9.45->10.10 — **the win over CUTLASS still stands under the cold-cache protocol** and the working sets are far past L2, so this is not a residency artifact.

**Bandwidth calibration**: the measured achievable ceiling = 1 GiB D2D `copy_` at 6.75 TB/s (84% of nominal 8 TB/s). Event timing has a Python launch floor below ~11us shapes, so CUDA-graph replay (capture 10 launches, replay x20, amortized) gives the pure-GPU truth: q_b 6.38us -> **5.61 TB/s (83%)**, o_proj 58%, gate_up 33%, down 32% — the ranking maps exactly onto CTA counts (128/48/32/48 vs 148 SMs), which empirically motivated tuning round 2.

**Tuning round 2: CTA-starvation treatment (tile-N=64 + split-K).** Split-K: grid.z CTAs each fold one slice of the 128-K block range, writing f32 partials of `(rows,n,S)`; a second `_SplitKReduce` DSL kernel reduces them on the same stream. Sweep matrix (graph-replay protocol; **this session shared the four GPUs with other jobs occupying memory, so absolute values are systematically ~15–27% slow — compare within the same session**):

| shape (n x k) | (bN128, S1) | (64,1) | (128,2) | (64,2) | (128,4) | Chosen |
|---|---|---|---|---|---|---|
| q_b 16384x2048 | 8.10 | — | 12.59 | — | — | **(128,1)**, refactor A/B zero-cost |
| o_proj 6144x16384 | 30.10 | 24.82 | **20.31** | 27.69 | 25.54 | **(128,2)** −32% |
| gate_up 4096x6144 | 13.75 | 11.07 | 9.82 | 9.05 | **8.18** | **(128,4)** −40% |
| down 6144x2048 | 8.83 | **6.93** | 7.62 | 9.45 | 10.27 | **(64,1)** −21% |

Notes: q_b's apparent +27% regression was disproven by same-session interleaved A/B x3 against a bit-equivalent pre-refactor kernel copy (8.01 vs 8.02us) — box drift, not code; same-session A/B is the protocol whenever the box is shared. Down wants no split-K: at kb_tot=16 the ~2us per-CTA fixed cost (scale preload + pipeline fill + tmem alloc) dominates, S=2 doubles that fixed cost and adds +22% partial round-trip traffic — pure loss; o_proj/gate_up amortize it over long k and net out positive. Chosen configs live in `units/fp8_gemm_dsl_tc.py::TILE_CFG`; `check` still passes 24/24. Resulting water levels (same 6.75 TB/s basis): o_proj 5.05 (75%), gate_up 3.19 (47%), down ~2.0 (30%), q_b 4.4 (66%; the previous clean-box round saw 83%).

## Production-integration profit estimate

The wide route (rows>8, #812) fires once per projection per layer across 78 layers. GB300 same-protocol rows=64 per-instance deltas: q_b −9.7us, o_proj −27.0, gate_up −11.3, down −3.5 — **−51.4us/layer -> x78 ≈ −4.0 ms/step** as the upper bound. Two realistic discounts: (1) gate_up/down are the shared expert and ride an aux stream overlapped with MoE collectives under the EP layout, so full-hiding folds that part off — **−2.9 ms/step**; (2) the win is **capacity-regime-only** — at c=32/instance (8 rows/rank) the GEMV chain runs and there is zero profit; c>=64/instance (16+ rows/rank) enters the payback zone. Against the ep4 anchor (c=32/n=128 step≈39ms, see `ep4-gb300.md`): in-regime TPOT improvement ≈ **7–10%**; round-2's extra gains can raise that by a few points once re-measured on a quiet window. **Prerequisites unchanged**: cubin export (the DSL's `cute.compile` supports it) + a Rust runtime loader (quack's `cache/jit.py` export_to_c + tvm-ffi load_module is a working reference) + the #812 oracle gate.

## rows=96: M-tiled grid + its own tile configs

The kernel was single-M-tile by grid formula only — `mma_tile_coord_mnl`
already carried `bidx` — so the 96-row (flagship 32-slot x MTP) bucket cost
one grid line plus tile-local -> global row offsets in the SFA staging and the
epilogue predicate/RestM slice. Two findings (GB300, same-session A/B,
rows=96, median us):

| shape | CUTLASS | DSL, <=64 cfg | DSL, tuned | best cfg | speedup |
|---|---|---|---|---|---|
| q_b | 35.63 | 28.14 | 26.54 | (128, sk1) | 1.34x |
| o_proj | 71.58 | 51.22 | 45.90 | (128, **sk1**) | 1.56x |
| shared_gate_up | 40.08 | 27.12 | 22.77 | (128, **sk2**) | 1.76x |
| shared_down | 27.86 | 25.28 | 20.27 | (**128**, sk1) | 1.37x |

- **The <=64 tile configs are wrong at 96**: two M tiles double the CTA count
  by themselves, the starvation the (tile-N, split-K) knobs compensate for is
  gone, and every knob flips to a net loss except gate_up's split-K x2 (its
  per-tile grid is still under half a wave). `TILE_CFG_MULTI_M` in the
  adapter carries the second regime; `tile_cfg(rows, n, k)` picks.
- **Why 96 trails the <=64 speedups (1.3-1.8x vs 1.4-2.8x)**: 96 = 64 + 32 —
  the second tile computes 64 rows for 32 real ones (25% wasted MMA overall),
  while CUTLASS amortizes its scheduling overhead better at larger m. A
  48-row-capable M tile (M=48 tcgen05 atom does not exist; would need a
  64+32 heterogeneous grid) is the only obvious way to close it; not planned.

e2e note: the step bench cannot reach 96 rows (32-slot ceiling, no MTP
drafter) — 96-bucket e2e waits for an MTP serving A/B; correctness is covered
by the lab check (8/8 with the 96 axis) and the DSL-vs-CUTLASS gate at m=96.

## Scope

This PR ships the fp8 blockwise GEMM line only. The harness's other decode-op families (GEMV norms, indexer, router, bookends, shared-expert, quant, MLA) and any non-datacenter-arch kernel variants are deliberately out of scope here.

## Next

- sm_103 residuals: the short-k prologue floor (~2us) — overlap scale preload with the main pipeline; vectorize the epilogue STG (currently 32x 32-bit scalar stores per thread); re-measure round-2 absolutes on a quiet GB300 window.
- ~~Productization form decision~~ **landed 2026-08-04**: `export_to_c` AOT objects linked into pegainfer-kernels behind `PEGAINFER_CUTEDSL_PYTHON`, exact-(m,n,k) table dispatch inside `glm52_fp8_groupwise_gemm_sm100_offset_launch`, buckets 16-96, `GLM52_FP8_DSL=0` kill switch. Recipe in `kernel-lab-ops.md`; re-measured clean-box kernel deltas 1.4-2.8x (PR #835 thread).
- **e2e measured (glm52_step_bench, EP4 tray03, 2026-08-04, `GLM52_KV_POOL_BLOCKS=18000`)**: whole-step p50 at rank-0 rows 16: 38.32 -> 34.52 ms (**-9.9%**); rows 32: 45.50 -> 41.55 ms (**-8.7%**, 703 -> 770 tok/s); rows 8 (GEMV control): -0.1% = noise. Matches the -7~10% in-regime estimate above. Caveats hit on the way: the step bench needed a pinned pool (the #823 budget-fill's 32-slot arena floor leaves <1 GiB for graph capture in this container) and single-bucket invocations (a pre-existing launch-ahead desync fires at phase drain; measurement rows print before it — also, EP4 placement puts all bench requests on rank 0, so the bucket label maps to rank-0 rows = 4x label; both filed for follow-up).
