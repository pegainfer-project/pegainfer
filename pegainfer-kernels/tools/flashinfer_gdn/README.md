# FlashInfer GDN SM120 candidate

**TL;DR:** This directory generates the explicitly selected Qwen3.5 Hv32
candidate and the isolated layout oracle; Rust validates all final link inputs.

Qwen3.5 defaults to Triton whether or not a candidate is linked. Building with
`PEGAINFER_QWEN35_GDN_AOT_BUNDLE` only makes the candidate available; the
Qwen3.5 production selector must explicitly request `flashinfer-candidate`.
That selection fails before KV allocation if its artifact, device, geometry,
TP configuration, or module load is invalid. The build generates a private
Rust identity from the validated final link inputs. The loaded GDN object retains
it; serving does not depend on test environment variables. Local candidate hashes establish
internal consistency and compatibility, not trusted distribution provenance.

## Quick start

Use Linux x86_64 and an SM120 GPU with Qwen3.5-4B at TP1. Download the complete
`Qwen/Qwen3.5-4B` checkpoint, including weights, configuration and tokenizer.

1. Follow the repository's [source build setup](../../../README.md#build-from-source);
   the Qwen3-only prebuilt release does not contain this backend. Prepare a separate
   CuTe generation environment using [requirements-cu13.lock](requirements-cu13.lock)
   and the Python/toolchain versions in [source-lock.json](source-lock.json).
   Keep the normal Qwen3.5 Triton build environment separate.
2. Run `compile_sm120.py` with the generation environment's Python. Point
   `--flashinfer-dir` to an independent clean checkout at the source-lock pin,
   and `--output` to a new bundle directory outside the repository. Do not
   replace the repository's shared FlashInfer submodule.
3. Set `PEGAINFER_QWEN35_GDN_AOT_BUNDLE` to the generated bundle's absolute path
   and `PEGAINFER_TRITON_PYTHON` to the Triton environment's Python. Build the
   server in release mode with `--features qwen35`; the bundle is validated
   and linked at build time.
4. Start the resulting `pegainfer` binary with `--model-path` pointing to the
   downloaded Qwen3.5-4B checkpoint and
   `--qwen35-gdn-backend flashinfer-candidate`. Check the startup log for
   `resolved=flashinfer-candidate` and `object_sha256`. Omitting the backend
   option selects Triton even in this build.

Once ready, use the [HTTP API](../../../README.md#api) to send a nonempty
completion request, using the model ID returned by `/v1/models`.

## Generation and validation

`source-lock.json` owns the frozen geometry, dtypes, dynamic-token bounds,
in-place ABI, tensor views, workspace formula, toolchain and FlashInfer pin.
Both the generation tools and the independent typed Rust validator consume
these project-owned values. `pegainfer-build::qwen35_gdn` rejects missing or
unknown fields, mismatches, symlinks, path traversal, non-regular files and
actual size/hash errors before the kernel build compiles or links the candidate.
The build snapshots verified bytes into `OUT_DIR` and validates those final
link inputs too. These Rust dependencies are enabled only by Qwen3.5.

`compile_sm120.py` runs in the pinned interpreter, checks actual toolchain/PTX
provenance, prepares the pinned HKV source and directly emits `manifest.json`,
`kernel.h`, `kernel.o`, and the native static runtime. Use `--output` for a new
destination and pass `--flashinfer-dir` explicitly to an independent clean
checkout at the
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
against the real generated bundle, explicitly selecting it with `--ignored`;
ordinary Cargo test runs leave it ignored. It shares one valid fixture across all
metadata and file/path rejection cases. One corrupted real object also
exercises the actual build-script rejection before linking.
Every Qwen3.5 build overwrites the generated Rust identity, including when the
bundle is removed. The runner also rebuilds the same target
without a bundle and checks archive/link inputs and explicit startup rejection.
Candidate executable snapshots are retained before that rebuild.

Candidate HF, scheduler, continuation and Shared-SM entries explicitly select
the candidate and check the actual loaded model against
`PEGAINFER_TEST_QWEN35_GDN_OBJECT_SHA256`, which the runner computes from
`kernel.o`. Missing inputs or a mismatched identity fail even when an entry is
run directly. These ignored GPU entries reuse the original scenario bodies in
crate-private test modules; ordinary entries continue to select Triton.

Before positive HF replay, the candidate short test directly asserts errors from
the same fixture/revision/identity prerequisite functions: missing/malformed
expected SHA, missing model/fixture, unresolved/wrong revision and wrong actual
model identity. The last check uses the executor's already-loaded model.
All HF entries reject an unresolved revision. Gate 3 reuses both existing short
and long HF golden scenarios. The long test checks
4097/8192-token outputs against an external oracle; HTTP and Shared-SM lifecycle
checks do not replace it. Actual GPU overlap requires Nsight timeline analysis;
a passing lifecycle test alone does not establish it.

Generated objects, oracle source copies, references, logs, model weights and
profiling results remain outside the source tree's tracked files. The runner
owns their temporary/evidence directory.
