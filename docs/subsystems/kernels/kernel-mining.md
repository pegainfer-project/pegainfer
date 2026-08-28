# Kernel Mining & Capsule Substrate

> **TL;DR:** Lift a launched kernel's cubin + full call ABI out of any external CUDA engine (vLLM/sglang) and out of our own decode graph, so a per-model manifest (`qwen34b.toml`, planned) can bind a stolen or self-authored cubin to a logical op and replay it. Two capture sides are built and GB300-verified: our CUDA-graph JSON dumper (`--dump-graph-png` now writes a machine `.json`) and a provider-agnostic CUPTI injection lib (`tools/kernel-capture/`). JSON-level A/B of PegaInfer vs vLLM Qwen3-4B bs=1 decode already located the levers — attention kernel and GEMM tile dispatch — and cleared norm/rope/silu/kv-append as structurally equivalent. **Next: nsys-quantify the attention two-pass vs one-pass gap before porting `fmhaSm100fKernel`.**
>
> **Last touched:** 2026-08

## Why

`docs/roadmap/direction.md` already names the "kernel ledger": for each kernel, its supported shapes/SM/dtype, its measured cost, and where it sits in a model's DAG, maintained machine-readably. This work is the capture layer under that ledger, driven by a concrete question: **when a bench shows vLLM winning bs=1 decode, how do we mine the responsible kernel out fast?**

Design intent (settled in discussion, not yet built as a runtime):
- The framework does **not** maintain serving-path kernels. It maintains a per-op *contract* (named ports, dtype/layout/axes, attrs — today's `pegainfer-core::ops::call_spec` shapes are exactly this) plus a *reference* twin for the accuracy gate. Kernels are external artifacts.
- A per-`(model, sm)` manifest (`qwen34b.toml`) lists the handful of ops the model touches; each row resolves to `builtin:` (only cuBLASLt + NCCL), `cubin:<sha256>#entry`, or `native:` during migration. Startup expands the op list, resolves every row, refuses to serve on a miss.
- Loader does three static checks, fail-closed at launch: TOML↔op-schema (named ports typed), TOML↔cubin (`cuFuncGetParamInfo` param count/size/offset vs the declared args), and guard consistency (no bare `const:` that duplicates a guard dim).
- The semantic **oracle stays engine-owned** — never pushed to the kernel provider. Shape can't define `eps` placement, accumulation dtype, or rounding; only an executable reference can. Providers may ship measurements as evidence, but admission is decided by our per-op numeric gate + the model-level `hf_golden_gate`.
- Performance A/B is incumbent-vs-candidate within a `(op, bucket)` catalog slot; the first candidate races the simulator/roofline expectation.

## Two capture sides (both built, both GB300 sm_103 verified)

### Our side — CUDA-graph JSON dump

`pegainfer-core/src/cuda_graph/dump.rs`. `--dump-graph-png PATH` now also writes `PATH.json` (`schema: pegainfer-cuda-graph-dump/v1`) beside the PNG/DOT. Per kernel node: symbol, demangled name, grid/block, dynamic smem, six function attributes (regs, static/const/local bytes, ptx/binary version), and every staged parameter via `cuFuncGetParamInfo` — 8-byte values resolved against the allocation map (`cuPointerGetAttribute`) into device/host ranges. Driver floor moved 12.3 → 12.4.

Verified: Qwen3-4B bs=1 decode graph = 543 kernel nodes, 2785 params, 1086 device pointers classified. Only cuBLASLt `nvjet` kernels launch with packed `extra` buffers and report `params: null`.

This is the source for the `qwen34b.toml` first draft (dump our own graph, tag everything `native:`, migrate rows to `cubin:` one at a time — each migration deletes a kernel we maintain).

### Mining side — CUPTI injection lib

`tools/kernel-capture/capture.c` (+ `build.sh`). Loaded into any CUDA process via `CUDA_INJECTION64_PATH`; provider-agnostic, no filesystem archaeology:

```bash
CUDA_INJECTION64_PATH=.../libkernelcapture.so KERNEL_CAPTURE_DIR=out \
  python -m vllm.entrypoints.openai.api_server --model ... --enforce-eager
```

- `CUPTI_CBID_RESOURCE_MODULE_LOADED` → dumps each cubin from memory (`module_<id>.cubin`). Triton, cuBLAS/cuBLASLt, CUTLASS, FlashInfer, hand-written CUDA all become module-load events with real ELF bytes.
- `cuLaunchKernel`/`cuLaunchKernelEx` EXIT callback → one `launches.jsonl` record per launch: symbol, grid/block/smem, attributes, staged params + pointer classification. Schema deliberately matches the graph dumper's, so the two sides join by symbol with no format conversion.

**Load-bearing gotcha — runtime-launched kernels.** Kernels launched through PyTorch's `<<<>>>` (all `vllm::*` and `at::native::*` ops) hand `cuLaunchKernel` a CUfunction that `cuFuncGetParamInfo`/`cuFuncGetAttribute` reject → first cut got launch config but empty params + zero attrs for ~66% of launches (only driver-API kernels like FMHA answered). ENTER-vs-EXIT is not the cause (lazy loading is a red herring here); `CUDA_MODULE_LOADING=EAGER` is worse — it hangs vLLM startup under CUPTI and asks the target to cooperate. **Fix (`cc61e20a`):** at module-load, self-load a private copy of the cubin and `cuModuleEnumerateFunctions` to cache each kernel's layout+attrs from the driver's own parse (reentrancy-guarded against the recursive MODULE_LOADED); `record_launch` resolves from the live handle else the cache. Result on vLLM 0.26.0 Qwen3-4B bs=1: 11418/11418 launches carry attributes, 8374 full param records, 3044 cuBLASLt-nvjet `null` (correct — extra-buffer), 7467 pointers classified. `rms_norm` decodes cleanly: out/input/weight pointers tagged device, `eps=1e-6` recovered as a scalar.

## JSON-level A/B: PegaInfer vs vLLM Qwen3-4B bs=1 decode (same GB300, bf16)

Structurally equivalent — each engine writes its own, no order-of-magnitude gap, **not worth stealing**:

| op | PegaInfer | vLLM |
|---|---|---|
| fused_add_rms / rms | `pegainfer::norm::FusedAddRMSNormRoundKernel` | `vllm::fused_add_rms_norm_kernel` |
| rope | `prefill_qk_norm_rope_warp_kernel` | `vllm::rotary_embedding_kernel` |
| silu | `silu_mul_kernel` | `vllm::act_and_mul_kernel` |
| kv_append | `flashinfer::AppendPagedKVCacheKernel` | `vllm::reshape_and_cache_flash_kernel` |

Two real levers:

1. **Attention (primary bs=1 suspect).** PegaInfer runs the FlashInfer **two-pass** split-KV path — `BatchDecodeWithPagedKVCacheKernel` (grid[64,8,1] block[16,4,2] **smem 9KB** regs56) **+ a separate `PersistentVariableLengthMergeStatesKernel`**. vLLM runs a **single fused** `fmhaSm100fKernel_...Q8Kv128PersistentSwapsAbForGen` (grid[1,8,1] block[512,1,1] **smem 143KB** regs128) — a persistent mega-kernel tuned for the generation phase (short Q, long KV, swap-AB), spending Blackwell's large smem to finish in one pass. This is the first port target: it's a C symbol, self-contained, params complete from the first launch. **Same kernel family glm52 already vendors legally** (`pegainfer-kernels/cubin/glm52/fmhaSm100fKernel_*SwapsAbForGen.cubin`).
2. **GEMM tile dispatch (steal the decision, not the cubin).** Both use cuBLASLt `nvjet`; PegaInfer selects 4 tiles (narrow `64x8`/`64x16`), vLLM selects 10, sharing 1 — vLLM's heuristic picks finer tiles per `(M=1,N,K)` (incl. large-N `128x256`/`128x192`), echoing the known cuBLAS narrow-K cliff. cuBLASLt is `builtin:`/proprietary → never a stolen cubin; the value is the shape→tile heuristic for our own selector.

## Storage & license discipline (answered by existing precedent)

The repo already commits FMHA cubins: `pegainfer-kernels/cubin/glm52/` holds 7 SM100 cubins, ~110–140 KB each (882 KB total), sha256-pinned in `trtllm_gen/flashInferMetaInfo.h`, loaded by the embedded loader, listed explicitly in `build.rs`, sourced+licensed in a README (FlashInfer 0.6.12, Apache-2.0). That is the pattern for any accepted capsule cubin — small, sha256-pinned, README with provider+version+license, explicit in `build.rs`. No git-lfs (repo has none; per-cubin ~100KB doesn't need it).

Three-way license split for mined cubins:
- `vllm::*` — Apache-2.0, committable (but these are the equivalent ops, low value).
- `fmhaSm100fKernel_*` — FlashInfer/TRT-LLM Apache-2.0; **same legal path glm52 already uses** — the clean one to steal.
- `nvjet_*` (cuBLASLt) — NVIDIA proprietary, **never redistribute**; this coincides with the design decision to keep GEMM as a `builtin:` special-case rather than a stolen cubin.

Capture artifacts (the 235 MB of 66 cubins + `launches.jsonl` per run) are mining scratch — never committed. They live on local disk under `/data/susun/`; `tools/kernel-capture/.gitignore` excludes the built `.so` and any output dir.

## State & next

- **Built + verified:** graph JSON dumper (`b66b534f`), CUPTI capture lib (`ff647924`), ABI-cache + pointer-classification fix (`cc61e20a`). Branch `feat/graph-dump-json`.
- **Next action:** nsys A/B the attention span — PegaInfer's `BatchDecode` + `MergeStates` two-pass vs vLLM's single `fmhaSm100fKernel`, on bs=1 mid-context — to size the win before porting. If it pays, archive the Qwen `fmhaSm100fKernel` variant under the glm52 cubin pattern (README + sha256, not yet wired to build), load it in a standalone test via `cuModuleLoadData` + the captured ABI, and numerically A/B against our decode attention.
- **Not yet built:** the `qwen34b.toml` manifest, the loader with the three static checks, the per-op reference/gate tier, the catalog selector. Capture is done; these are the consumers.
