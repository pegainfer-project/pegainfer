# FlashInfer GDN SM120 AOT bundle

This directory owns the generation-only FlashInfer/CuTe environment for the
Qwen3.5 GDN prefill specialization. Serving does not import Python, CuTe,
FlashInfer, or Triton for this kernel and does not load PTX. The release build
validates the manifest, then statically links the exported native object and
`libcuda_dialect_runtime_static.a` behind the stable PegaInfer C ABI.

The source lock pins FlashInfer and the small HKV state-layout specialization.
The generator emits only the production Hv32 candidate. SM120 +
Hq/Hk/Hv/D=`16/16/32/128`, BF16 inputs, FP32 HKV state, single GPU is eligible
for production selection. Other capabilities retain the Triton path. An
eligible configuration requires the validated AOT object and does not silently
fall back when that object was not linked.

## Contract-validated local generation

Run these commands from the repository root. The artifact contract currently
requires Python 3.12.3 exactly; use an interpreter with that version rather
than the serving or Triton environment.

Initialize the pinned FlashInfer source and create an isolated generation
environment:

```bash
git submodule update --init \
  pegainfer-kernels/third_party/flashinfer

python3 -c \
  'import sys; assert sys.version.split()[0] == "3.12.3", sys.version'
python3 -m venv target/flashinfer-gdn-cu13-venv

export PEGAINFER_GDN_AOT_PYTHON="$PWD/target/flashinfer-gdn-cu13-venv/bin/python"

"$PEGAINFER_GDN_AOT_PYTHON" -m pip install \
  -r pegainfer-kernels/tools/flashinfer_gdn/requirements-cu13.lock
```

The version assertion stops immediately unless `python3` is exactly Python
3.12.3. The generation-only CUDA 13 packages are pinned in
`requirements-cu13.lock`; they are not serving dependencies. Retired CUDA
12.8/PTX generation workflows are not supported.

Generate a fresh production-only bundle. The generator refuses to overwrite
an existing output directory, so remove or rename an old local output before
reusing the same path.

```bash
"$PEGAINFER_GDN_AOT_PYTHON" \
  pegainfer-kernels/tools/flashinfer_gdn/generate.py \
  --python "$PEGAINFER_GDN_AOT_PYTHON" \
  --flashinfer-dir pegainfer-kernels/third_party/flashinfer \
  --output target/flashinfer-gdn-sm120
```

The only generated variant is
`target/flashinfer-gdn-sm120/qwen35_4b_candidate/`.

The contract pins and validates the source, patch, generator, package versions,
compiler metadata, ABI, geometry, and hashes recorded by each bundle. Repeated
generation has been byte-identical on the same host, but cross-host object
identity is not currently guaranteed. Release distribution must therefore
preserve and validate the complete bundle and its manifest rather than assume a
globally fixed object hash.

Validate a generated or downloaded complete bundle against its pinned source:

```bash
python3 pegainfer-kernels/tools/flashinfer_gdn/artifact_contract.py \
  validate-bundle target/flashinfer-gdn-sm120 \
  --flashinfer-dir pegainfer-kernels/third_party/flashinfer
```

For a Qwen3.5 release build, point the kernel build at the validated variant.
Qwen3.5 still needs its normal build-time Triton AOT environment; see
[`../triton/README.md`](../triton/README.md).

```bash
export PEGAINFER_QWEN35_GDN_AOT_BUNDLE="$PWD/target/flashinfer-gdn-sm120/qwen35_4b_candidate"
export PEGAINFER_CUDA_SM=120
export PEGAINFER_TRITON_PYTHON="$PWD/.venv/bin/python"

cargo build --release \
  -p pegainfer-server \
  --no-default-features \
  --features qwen35 \
  --bin pegainfer
```

The production confidence gate is not the Python packager validating itself.
On an SM120 runner with the pinned model snapshot, invoke the canonical runner;
it validates the real bundle, builds through production `build.rs`, and runs
the five exact GPU gates with fail-on-skip/test-count checks:

```bash
env \
  PEGAINFER_CUDA_SM=120 \
  PEGAINFER_TRITON_PYTHON="$PWD/target/flashinfer-gdn-cu13-venv/bin/python" \
  PEGAINFER_QWEN35_GDN_AOT_BUNDLE="$PWD/target/flashinfer-gdn-sm120/qwen35_4b_candidate" \
  PEGAINFER_TEST_MODEL_PATH="$PWD/models/Qwen3.5-4B" \
  PEGAINFER_TEST_MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  PEGAINFER_GDN_EXPECT_BRANCH=test/qwen35-gdn-production-gates \
  CARGO_TARGET_DIR="$PWD/target/gdn-production-gates" \
  pegainfer-qwen35/tools/run_gdn_production_gates.sh
```

When `PEGAINFER_QWEN35_GDN_AOT_BUNDLE` is set, `pegainfer-kernels/build.rs`
rechecks schema, SM, geometry, ABI, object/header/runtime hashes and sizes. A
missing, incomplete, or incompatible selected path fails the build instead of
silently linking a different kernel.

When the variable is not set, the build contains no FlashInfer GDN object.
Unsupported SM, geometry, or tensor-parallel configurations still use the
explicit Triton capability fallback. The supported SM120/Hv32/single-GPU
configuration instead fails model startup with a missing-AOT error; it does not
silently change backend. The model crate never receives the bundle path and
sees only a semantic GDN operation.

Generated headers, objects, static archives, bundles, model weights, `target/`,
logs, and benchmark JSON are release/build artifacts and must not be committed.
