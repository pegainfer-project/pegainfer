# K3 vendored decode kernels (sm_103)

External serving kernels for the K3 decode path, bound at runtime by the
capsule loader in `csrc/k3/k3_capsule.cu`. Provenance and license per file;
sha256 pins in `SHA256SUMS` are verified by the embedded loader. All sources
are Apache-2.0 (`SPDX-License-Identifier: Apache-2.0`, "Copyright contributors
to the vLLM project"); cuBLAS `nvjet_*` kernels are deliberately absent
(NVIDIA proprietary — never redistributed).

Both files are **offline single-instantiation builds**: the v0.28.0 wheel
packs these kernels inside multi-MB fatbins (2.3 MB flashkda, 16.7 MB `_C`),
so instead of vendoring those, the exact launched template instantiation was
compiled standalone from the v0.28.0 tagged sources
(`docker.io/vllm/vllm-openai:v0.28.0`,
`sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`,
linux/arm64, symbols confirmed by CUPTI module-load capture of its Kimi-K3
DP4×EP4 decode on GB300) with CUDA 13.2 `nvcc -cubin -arch=sm_103a -O3`
(template arguments recovered from the captured launch symbols; TUs are the
upstream `.cu` with the torch host-launcher sections excised — no functional
edits to device code):

| file | instantiation | upstream source (tag v0.28.0) |
|---|---|---|
| `kda_decode_fusion_h96_sm103.cubin` | `kda_decode_fusion_many_heads_kernel<true,true,96,96,true,0,0,0,0,0,1,1,1,1,1,1>` | `csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu` |
| `single_group_topk_e512t22_sm103.cubin` | `single_group_topk_warp_kernel<float,float,int,SCORING_SIGMOID,512,22>` | `csrc/libtorch_stable/moe/grouped_topk_kernels.cu` (+ `moeTopKFuncs.cuh`) |

Rebuild note: the offline TUs are byte-derivable from the upstream tag; the
recipe (cut point + explicit instantiation) is recorded in
`docs/models/k3/vllm-kernel-ab.md`'s port log. A re-vendor against a newer
vLLM must re-capture the launch symbols first — template flag sets are not
stable across versions.

Mined-but-unwired cubins from the same capture (attn-res online kernel,
CuTe-DSL skinny GEMM M=1 pair, `LLBf16Dotprod` router GEMM) are **not**
vendored here — only kernels a launch site actually binds earn a row. They
live in the capture archive (`/data/susun/kernel-capture/`, tray03) with
their ABI manifest, see the A/B doc's "not ported" rationale.
