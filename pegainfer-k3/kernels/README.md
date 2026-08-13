# K3 TileLang Kernels

**TL;DR**: `generate.py` AOT-compiles the vendored TileLang kernel definitions
in `tilelang_defs.py` into CUDA that `pegainfer-kernels/build.rs` hands to
nvcc under the `k3` feature. Three tiers — generate, pre-generated, stub —
and the generated CUDA is a Cargo `OUT_DIR` artifact that is never checked in.

## What lives here

| File | Role |
| --- | --- |
| `tilelang_defs.py` | Vendored, **certified** TileLang kernel definitions. Byte-identical to upstream; do not respell. |
| `generate.py` | Walks the shape buckets, dumps `get_kernel_source()` per instantiation, emits one `.cu` per kernel family with a hand-written dispatch launcher. |

Three kernels are ported, all row-independent, all with a **static** batch
dimension:

| Kernel | Instantiations | Launcher |
| --- | --- | --- |
| `router_topk_batched` | E ∈ {224, 896} × B buckets, TOPK = 16 | `k3_router_topk` |
| `attnres_scores_batched` | NB ∈ 1..8 × B buckets, H = 7168 | `k3_attnres_scores` |
| `attnres_mix_batched` | NB ∈ 1..8 × B buckets, H = 7168 | `k3_attnres_mix` |

B buckets: `1, 2, 4, 8, 16, 32, 48, 64, 96, 128` — 180 instantiations, about
three minutes of generation.

TileLang always names the entry point `main_kernel`, so every instantiation is
renamed to a shape-tagged symbol before the sources are concatenated. Each
family gets exactly one hand-written `extern "C" int k3_<name>(..., cudaStream_t)`
launcher that dispatches on the runtime shape and returns
`cudaErrorInvalidValue` for anything not instantiated.

## Why the definitions are vendored, not imported

The upstream Python repo certified these kernels row-by-row against the
reference implementation, and the batched variants are gated *bitwise* against
the certified bs=1 kernels. A semantically equivalent respelling can still move
a rounding boundary, so the vendored copy is byte-identical and the port
carries no local edits. Changing a kernel means changing it upstream, re-running
the parity gate there, and re-vendoring here.

`generate.py` therefore never touches a kernel body. The one thing it does
rebind is `_compile`, and only when `--arch` is passed: the vendored helper
compiles for the local GPU, which build containers usually do not have, so the
generator pins the arch explicitly. The kernel bodies resolve `_compile`
through the module namespace at call time, so this pins the target without
editing a character of certified code.

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
host-side launch-parameter encoding that the dynamic shared-memory size is
recovered from — all three are checked per instantiation, so a version bump
fails at build time rather than at the first launch.

## Batch buckets are a caller contract

`batch` in every launcher is the **bucket**, not the live row count. The ops
layer (`pegainfer-kernels/src/ops/k3_tilelang.rs`) rounds up; the caller must
pass buffers sized for the bucket, and the kernel computes and discards the
tail rows. Allocating at `K3_MAX_BATCH` once and reusing keeps the pointers
stable for CUDA Graph capture.
