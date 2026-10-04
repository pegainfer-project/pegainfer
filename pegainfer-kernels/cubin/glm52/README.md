# GLM5.2 FlashInfer FMHA cubins

These seven SM100-family cubins are the minimal selector closure for GLM5.2
TP4 sparse decode (`batch={1,2,4,8}`, `topk={256,2048}`, 16 heads, E4M3
Q/K/V, BF16 output). They come from FlashInfer 0.7.0's
`flashinfer-cubin` bundle `2d6a5a029eefcc388ec0ceb87efb55d8bcce5c3c` and match the
vendored FlashInfer runner at `v0.7.0` (`4d75a33f`). The runner's `KernelParams`
layout is baked into these cubins: bump the submodule and the cubins together.

The SHA-256 values are part of `trtllm_gen/flashInferMetaInfo.h` and checked by
the embedded cubin loader. Keep the seed kernels: FlashInfer loads them before
right-sizing V per CTA, even though they do not appear as the final Nsight
kernel symbols.

License: Apache-2.0; see the vendored [FlashInfer license](../../third_party/flashinfer/LICENSE).
