# FlashInfer GDN SM120 AOT bundle

This directory owns the generation-only FlashInfer/CuTe environment for the
Qwen3.5 GDN prefill specialization. Serving does not import Python, CuTe,
FlashInfer, or Triton for this kernel and does not load PTX. The release build
validates the manifest, then statically links the exported native object and
`libcuda_dialect_runtime_static.a` behind the stable PegaInfer C ABI.

The source lock pins FlashInfer and the small HKV state-layout specialization.
The generated bundle contains an Hv32 production candidate and an Hv48
diagnostic variant. Only SM120 + Hq/Hk/Hv/D=`16/16/32/128`, BF16 inputs, FP32
HKV state, single GPU is eligible for production selection. Other capabilities
retain the Triton path; a selected but invalid bundle fails at build time.

The canonical generator CLI is:

```bash
python3 pegainfer-kernels/tools/flashinfer_gdn/generate.py --help
```

Its CUDA 13 environment is pinned in `requirements-cu13.lock`. The Stage 13 GPU
gate records the exact environment creation, generation, validation, and
release-link commands after they have been run on the target toolchain; do not
copy the retired CUDA 12.8/PTX commands from older benchmark logs.

Host-side source, state-layout, and package-contract checks:

```bash
python3 -m unittest discover \
  -s pegainfer-kernels/tools/flashinfer_gdn/tests \
  -v
```

Validate a generated or downloaded complete bundle against its pinned source:

```bash
python3 pegainfer-kernels/tools/flashinfer_gdn/artifact_contract.py \
  validate-bundle target/flashinfer-gdn-sm120 \
  --flashinfer-dir pegainfer-kernels/third_party/flashinfer
```

At build time, `PEGAINFER_QWEN35_GDN_AOT_BUNDLE` points to the validated
`qwen35_4b_candidate/` directory. `pegainfer-kernels/build.rs` rechecks schema,
SM, geometry, ABI, object/header/runtime hashes and sizes before linking. The
model crate never receives this path and sees only a semantic GDN operation.

Generated headers, objects, static archives, bundles, model weights, `target/`,
logs, and benchmark JSON are release/build artifacts and must not be committed.
