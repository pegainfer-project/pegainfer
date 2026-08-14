# K3 TileLang Kernels

**TL;DR**: `generate.py` AOT-compiles the vendored TileLang kernel definitions
in `tilelang_defs.py` into CUDA that `pegainfer-kernels/build.rs` hands to
nvcc under the `k3` feature. Thirteen batched families cover every non-GEMM
step of a K3 decode iteration — three tiers (generate, pre-generated, stub),
and the generated CUDA is a Cargo `OUT_DIR` artifact that is never checked in.

## What lives here

| File | Role |
| --- | --- |
| `tilelang_defs.py` | Vendored, **certified** TileLang kernel definitions. Byte-identical to upstream; do not respell. |
| `generate.py` | Walks the configuration lists, dumps `get_kernel_source()` per instantiation, emits one `.cu` per kernel family with a hand-written dispatch launcher. |

## Instantiation matrix

Every shape is a **static compile dimension** — there is one kernel per
configuration tuple and the launcher dispatches on it at run time. The shape
arguments come from the K3 text config, reproduced in `generate.py` from named
constants under the engine's own names.

The batch size is a compile dimension too, so there is no separate single-row
kernel set: `B = 1` is a bucket whose per-row spelling is the certified
single-row kernel. B buckets are `1, 2, 4, 8, 16, 32, 48, 64, 96, 128`, so
every row below is its shape count × 10.

| Kernel | Shapes per bucket | Instantiations | Launcher |
| --- | --- | --- | --- |
| `rms_norm_rbs_batched` | H ∈ {7168, 512, 3584} | 30 | `k3_rms_norm_rbs_batched` |
| `land_batched` | 14 (NT, N, OFF) spans, SK = 1 | 140 | `k3_land_batched` |
| `land_rms_norm_rbs_batched` | MLA q_a, SK = 1 | 10 | `k3_land_rms_norm_rbs_batched` |
| `add2_batched` | N = 7168 | 10 | `k3_add2_batched` |
| `mul_sigmoid_batched` | N = 12288 | 10 | `k3_mul_sigmoid_batched` |
| `situ_batched` | N ∈ {6144, 33792} | 20 | `k3_situ_batched` |
| `conv_silu_batched` | KP = 12288, W = 4, SK = 1 | 10 | `k3_conv_silu_batched` |
| `kda_core_batched` | 96 heads × 128 head_dim | 10 | `k3_kda_core_batched` |
| `mla_attn_batched` | 96 heads, qk 192, v 128, CAP ∈ {128} | 10 | `k3_mla_attn_batched` |
| `router_topk_batched` | E ∈ {896, 224}, TOPK = 16 | 20 | `k3_router_topk_batched` |
| `attnres_scores_batched` | NB ∈ 1..8, H = 7168 | 80 | `k3_attnres_scores_batched` |
| `attnres_mix_batched` | NB ∈ 1..8, H = 7168 | 80 | `k3_attnres_mix_batched` |

**430 instantiations**, about 20 seconds of generation and 7 seconds of nvcc.
The pool fans out at *instantiation* granularity, not family granularity — the
families differ by more than an order of magnitude in size, so a family-granular
pool would be bound by `land_batched` alone. It defaults to one worker per CPU
capped at 32; each worker holds a TileLang lowering, so lower it with
`PEGAINFER_K3_TILELANG_JOBS` on memory-tight hosts.

Two lists in `generate.py` are deliberately narrow and are one-line changes:

* `MAX_CTX` — the MLA context-capacity list. Both the instantiation count and
  the per-block shared memory scale with the capacity, so serving longer
  contexts is a deliberate widening, not a free one.
* `SPLIT_K` — the segment counts the partial consumers (`land`,
  `land_rms_norm_rbs`, `conv_silu`) accept. Only `1` — the single partial a
  framework GEMM produces — has a launch site; the reference engine's
  split-K-8 GEMV shapes are not generated.

Neither needs a launcher edit.

### What is deliberately *not* here

No GEMV family. Dense projections run on cuBLASLt and the routed experts on
the DeepGEMM masked grouped-GEMM chain (`csrc/k3/k3_moe_chain.cu` and the
`k3.deepgemm.*` ops), so the upstream `gemv`, `expert_gemv` and
`packed_expert_gemv` kernels would be dead weight. Their consumers are still
here — `land_batched` at `SK = 1` is exactly the bf16 landing of a cuBLASLt
f32 output.

TileLang always names the entry point `main_kernel`, so every instantiation is
renamed to a shape-tagged symbol before the sources are concatenated. Each
family gets exactly one hand-written `extern "C" int k3_<name>(..., cudaStream_t)`
launcher that dispatches on the runtime configuration and returns
`cudaErrorInvalidValue` for anything not instantiated. One family per
translation unit is also what lets nvcc compile them in parallel.

## Why the definitions are vendored, not imported

The upstream Python repo certified these kernels row-by-row against the
reference implementation, and the batched variants are gated *bitwise* against
the certified bs=1 kernels. A semantically equivalent respelling can still move
a rounding boundary, so the vendored copy is byte-identical and the port
carries no local edits. Changing a kernel means changing it upstream, re-running
the parity gate there, and re-vendoring here.

That is also why whole factories are vendored rather than re-spelled: some of
them branch or loop in plain Python around the IR (tile counts, window taps,
unrolled constant tables), so the body that gets built differs per shape and
there is no single body to transcribe. The eager builder also rewrites Python
loops into IR loops, after which the loop variable can no longer index a
constant table — a transcription that looks equivalent is not.

`generate.py` therefore never touches a kernel body. The one thing it does
rebind is `_compile`, and only when `--arch` is passed: the vendored helper
compiles for the local GPU, which build containers usually do not have, so the
generator pins the arch explicitly. The kernel bodies resolve `_compile`
through the module namespace at call time, so this pins the target without
editing a character of certified code.

## What the generator recovers from the host stub

TileLang exposes the device source but not the launch contract, so
`generate.py` parses the TVM host stub for the **launch geometry** — grid,
block and dynamic shared memory — and checks it for exact equality against the
analytically known grid. A codegen change fails the build instead of launching
a wrong grid. Past 48 KiB of dynamic shared memory the launcher emits the
per-symbol `cudaFuncSetAttribute` opt-in.

The batched kernels put B as the **first** grid dimension, so every analytic
grid carries the batch factor; the stub carries exactly as many grid entries as
the kernel's `T.Kernel` arity, unit dimensions included, and the check does not
trim them.

The same parse also asserts that no body lowers to TMA. A warp-specialized
TileLang kernel takes `CUtensorMap` descriptors instead of pointers and adds a
producer warpgroup to the block, so the launchers — which bind plain pointers
and the requested thread count — would be silently wrong. None of the batched
bodies do this today (their only bulk copy is `mla_attn`'s one-dimensional
query load), but that is a property of TileLang, not of the kernels, so it is
asserted rather than assumed.

## TileLang codegen is not byte-reproducible

Re-running the same instantiation can swap the names of two *aliases of the
same* dynamic-shared-memory offset — `mla_attn` emits `void* workspace` and
`void* workspace_1`, both `buf_dyn_shmem + 0`, and which one each `AllReduce`
scratch argument gets flips run to run (measured 8/10 vs 2/10 on an otherwise
identical invocation). Nothing else moves: same offsets, same instructions.

The practical consequence is only for tooling: a byte-diff of a generated
body against a fresh upstream `get_kernel_source()` dump has to canonicalize
those aliases before comparing, or it will report a spurious mismatch. Do not
"fix" it by pinning a compile order — the variation is inside TileLang, not in
the order families are generated.

## Build tiers

`build.rs` (section `k3 tilelang`) takes the first tier that works:

1. **Generate** (preferred). Finds a Python that can `import tilelang` —
   `PEGAINFER_K3_TILELANG_PYTHON`, then `PEGAINFER_TILELANG_PYTHON`, then the
   workspace `.venv`, then `python3`/`python` — and runs
   `generate.py --out-dir <OUT_DIR>/tilelang/k3 --arch sm_<max>a`. The arch
   comes from the same SM list nvcc targets, so generation never needs a
   visible GPU.
2. **Pre-generated**. `PEGAINFER_K3_TILELANG_PREGEN=<dir>` points at a
   directory built earlier by

   ```bash
   python3 generate.py --out-dir pregen --vendor-includes --arch sm_103a
   ```

   `--vendor-includes` copies the TileLang and CUTLASS header trees into
   `<dir>/include`, making the directory self-contained on a host that has no
   TileLang install to take include paths from. `build.rs` reads
   `<dir>/manifest.txt` for the CUDA files and the two include roots.
3. **Stub**. Neither of the above: `build.rs` writes launchers that return
   `cudaErrorNotSupported`, so a featureless or Python-less CI build still
   links and a decode attempt fails loudly instead of computing garbage.

The version pin is asserted inside `generate.py` (`KNOWN_GOOD_TILELANG`).
TileLang releases move the entry-point name, the parameter ordering, and the
host-side encoding the launch geometry and TMA constants are recovered from —
all of it is checked per instantiation, so a version bump fails at build time
rather than at the first launch.

## Batch buckets are a caller contract

`b` in every launcher is the **bucket**, not the live row count. The ops layer
(`pegainfer-kernels/src/ops/k3_tilelang.rs`) rounds up with `k3_batch_bucket`;
the caller must pass buffers sized for the bucket, and the kernel computes and
discards the tail rows — so those rows still have to point at valid memory.
Allocating at `K3_MAX_BATCH` once and reusing keeps the pointers stable for
CUDA Graph capture.

Per-slot state is `[b, ...]` contiguous with each row holding exactly the
single-row layout: the `conv_silu` windows, the `kda_core` recurrent state and
the MLA `Kc`/`Vc` caches. That contract is what the upstream bitwise gate
proves, and it is why the caches are slot-indexed rather than paged.
