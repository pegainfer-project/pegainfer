# FlashInfer GDN SM120 candidate

**TL;DR:** This directory generates the explicitly selected Qwen3.5 Hv32
candidate and the isolated layout oracle; Rust validates all final link inputs.

Qwen3.5 defaults to Triton whether or not a candidate is linked. Building with
`PEGAINFER_QWEN35_GDN_AOT_BUNDLE` only makes the candidate available; the
Qwen3.5 production selector must explicitly request `flashinfer-candidate`.
That selection fails before KV allocation if its artifact, device, geometry,
TP configuration, or module load is invalid. Local candidate hashes establish
internal consistency and compatibility, not trusted distribution provenance.

`source-lock.json` owns the frozen geometry, dtypes, dynamic-token bounds,
in-place ABI, tensor views, workspace formula, toolchain and FlashInfer pin.
Both the generation tools and the independent typed Rust validator consume
these project-owned values. `pegainfer-build::qwen35_gdn` rejects missing or
unknown fields, mismatches, symlinks, path traversal, non-regular files and
actual size/hash errors before the kernel build compiles or links the candidate.
The build snapshots verified bytes into `OUT_DIR` and validates those final
link inputs too. These Rust dependencies are enabled only by Qwen3.5.

`generate.py` prepares the pinned source, applies the HKV specialization and
packages `manifest.json`, `kernel.h`, `kernel.o`, and the native static runtime.
Pass `--flashinfer-dir` explicitly to an independent clean checkout at the
`source-lock.json` FlashInfer pin (`a0efa0adfe49bb836ab1a147d6572980b870f3d4`).
Keep that generation checkout outside the serving repository. The shared
`pegainfer-kernels/third_party/flashinfer` submodule stays at the repository's
existing pin: its headers also implement other models' production attention.
The GDN source pin must not replace that shared dependency or its include path.
Its Python interpreter must match `requirements-cu13.lock` and the exact
Python/toolchain versions in the source lock. Qwen3.5's normal Triton build
uses a separate environment. Python/CuTe and manifests are generation/build
inputs; serving uses the statically linked object through the stable C ABI.
The lock uses the official Linux x86_64 CPython 3.12 Torch CUDA 12.8 wheel
with its published hash: the PyPI CUDA 12.6 build cannot run the SM120 oracle.

`layout_reference.py` is called by the canonical production gate runner. It
validates the supplied candidate, then exports an
isolated upstream HVK oracle with the same compiler. The reference records
the actual candidate object hash for comparison with the linked Rust backend. The upstream source retains both original layout expressions;
only the existing export type-annotation hunks are applied. The small
`upstream_adapter.c` launches that generated object in-place and is never
included by the production build.
The runner requires `PEGAINFER_GDN_FLASHINFER_DIR` pointing to the same independent
checkout and passes it through `--flashinfer-dir`. It validates the frozen source
before any builds and records both the shared submodule and generation checkout
SHAs in its provenance log. This variable is consumed only by the gate runner;
Cargo and the server do not use it.

The validator writes raw little-endian fixtures for T=1/63/64/65/128 and a
64+64 resumed sequence under its temporary output directory. Inputs and states
are nonzero and asymmetric; the state transposes occur only at the upstream
input/output comparison boundary. The Rust Gate 1 consumes these same files
through the production safe wrapper and compares the complete output/state,
including the first continuation state. Fixed absolute and relative tolerances
are zero because the patch changes storage layout alone. Final Hv32 GPU
validation must establish whether this contract passes; no historical result
substitutes for it.

The canonical runner additionally executes the single Rust host mutation test
against the real generated bundle. It shares one valid fixture across all
metadata and file/path rejection cases; it does not manufacture a fake object
or claim host validation proves GPU execution. One corrupted real object also
exercises the actual build-script rejection before linking.
Gate 3 reuses both existing short and long HF golden tests. The long test checks
4097/8192-token outputs against an external oracle; the production HTTP and
Shared-SM checks cover request flow and overlap, so they do not replace it.

Generated objects, oracle source copies, references, logs, model weights and
profiling results remain outside the source tree's tracked files. The runner
owns their temporary/evidence directory. Neither the upstream oracle nor a
separate-state ABI exists in the serving binary.
