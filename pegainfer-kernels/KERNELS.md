# PegaInfer Kernels Index

**Scope**: this crate owns CUDA/Triton build output, FFI declarations, kernel ABI tensor helpers, paged-KV layout metadata, and Rust operator wrappers. Runtime policy objects such as `KvPool`, `PagePool`, and `SamplingParams` stay outside this crate.

Use this file as the LLM entrypoint before editing kernels. Start from `op_id`, then jump to the Rust wrapper, FFI symbol, and source file.

## Shared Dense And Sampling Helpers

| op_id | Runtime owner | Rust wrapper | FFI symbol | Source | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- | --- |
| `shared.linear.gemm_per_token` | model-specific decode accuracy gates | `ops::gemm_per_token` / `ops::gemm_per_token_into_checked` | `gemm_per_token_cuda` | `csrc/shared/linear.cu` | cuBLAS | computes each row through the N=1 decode GEMM boundary; used when row-wise parity is required before performance optimization |
| `shared.sampling.argmax_batch_bf16` | batched greedy gates | `ops::argmax_batch_bf16_into` | `argmax_batch_bf16_cuda` | `csrc/shared/argmax.cu` | CUDA | one greedy top-1 result per row over contiguous `HiddenStates` logits |
| `shared.elementwise.accumulate_bf16_token_scaled_to_f32` | DeepSeek-V2-Lite NCCL device combine | `ops::accumulate_bf16_token_scaled_to_f32_into` | `accumulate_bf16_token_scaled_to_f32_cuda` | `csrc/shared/elementwise.cu` | CUDA | accumulates one bf16 expert-output token into a selected row of reusable f32 device scratch before the NCCL combine all-reduce |
| `shared.sampling.argmax_batch_bf16_indexed` | selected batched greedy gates | `ops::argmax_batch_bf16_indexed_into` | `argmax_batch_bf16_indexed_cuda` | `csrc/shared/argmax.cu` | CUDA | compact greedy top-1 results for selected source rows over `HiddenStates` logits |
| `shared.sampling.batch_topk_topp` | mixed greedy/non-greedy decode epilogues | `ops::gpu_sample_batch_into` / `pegainfer-core::ops::select_batch_tokens_into` | `gpu_sample_batch_flashinfer_cuda` | `csrc/shared/flashinfer_sampling.cu` | FlashInfer CUDA | compact non-greedy rows, gather/cast bf16 logits to f32, FlashInfer OnlineSoftmax + Sampling/TopP/TopKTopP by active per-row filters |

## Qwen3-4B Dense Full-Attention Path

Qwen3-4B uses bf16 dense full attention with `hidden_size=2560`, `num_attention_heads=32`, `num_key_value_heads=8`, `head_dim=128`, and GQA group size 4. TP shards these head/intermediate dimensions per rank; the kernel IDs remain the same.

| op_id | Phase | Rust wrapper | FFI symbol | Source | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3_4b.embedding.batch` | prefill/unified | `ops::embedding_batch` | `embedding_batched_cuda` | `csrc/shared/elementwise.cu` | CUDA | token ids u32, output `HiddenStates` column-major `[hidden, tokens]` |
| `qwen3_4b.norm.rms_batch` | prefill/decode/unified | `ops::rms_norm_batch_into` | `rms_norm_batched_cuda` | `csrc/shared/flashinfer_norm.cu` | FlashInfer CUDA | bf16 hidden states, one row per token |
| `qwen3_4b.norm.rms_vec` | logits | `ops::rms_norm` / `ops::rms_norm_into` | `rms_norm_cuda` | `csrc/shared/flashinfer_norm.cu` | FlashInfer CUDA | bf16 vector |
| `qwen3_4b.linear.gemm_rows` | qkv projection | `ops::gemm_rows_into` | `gemm_cuda` | `csrc/shared/linear.cu` | cuBLAS | row slices from fused QKV matrix |
| `qwen3_4b.linear.gemm` | o/mlp/lm_head | `ops::gemm_into` / `ops::gemm` | `gemm_cuda` | `csrc/shared/linear.cu` | cuBLAS | weight row-major, hidden column-major |
| `qwen3_4b.attn.qk_norm_rope` | attention prep | `ops::qk_norm_rope_batch_decode_into` or direct FFI in unified path | `qk_norm_rope_batched_decode_cuda` | `csrc/shared/prefill_attention.cu` | CUDA | full RoPE, `head_dim=128`, per-token positions |
| `qwen3_4b.kv.scatter` | prefill/decode/unified | direct FFI from model paths | `paged_kv_scatter_cuda` | `csrc/shared/paged_attention.cu` | FlashInfer-layout CUDA wrapper | page-first `KvLayout`, NHD K/V blocks |
| `qwen3_4b.attn.prefill_paged` | prefill/unified | `ops::prefill_attention_paged_into` or direct FFI in unified path | `batch_prefill_paged_cuda` | `csrc/shared/paged_attention.cu` | FlashInfer CUDA | `HEAD_DIM=128`, causal, paged KV |
| `qwen3_4b.attn.decode_paged` | decode/unified | `ops::paged_attention_batch_decode_into` or direct FFI in unified path | `paged_attention_decode_cuda` | `csrc/shared/paged_attention.cu` | FlashInfer CUDA | `HEAD_DIM=128`, no partition-KV |
| `qwen3_4b.norm.fused_add_rms` | residual | `ops::fused_add_rms_norm_batch_into` | `fused_add_rms_norm_batched_cuda` | `csrc/shared/flashinfer_norm.cu` | FlashInfer CUDA | residual add plus RMSNorm over batch |
| `qwen3_4b.mlp.silu_mul_fused` | MLP | `ops::silu_mul_fused_batch_into` | `silu_mul_fused_cuda` | `csrc/shared/fused_proj.cu` | CUDA | input `[2 * intermediate, batch]`, output `[intermediate, batch]` |
| `qwen3_4b.elementwise.add` | residual/unified | `ops::add_batch_into` | `add_cuda` | `csrc/shared/elementwise.cu` | CUDA | same-shape `HiddenStates` |
| `qwen3_4b.sampling.greedy` | decode output | `pegainfer-core::ops::select_batch_tokens_into` | `argmax_batch_bf16_split_indexed_cuda` | `csrc/shared/argmax.cu` | CUDA | compact greedy rows, one indexed batched argmax read-back per step |
| `qwen3_4b.sampling.random` | decode output | `pegainfer-core::ops::select_batch_tokens_into` | `gpu_sample_batch_flashinfer_cuda` | `csrc/shared/flashinfer_sampling.cu` | FlashInfer CUDA | compact non-greedy rows, one batched FlashInfer sampling call per step |

## Kimi-K2 Text TP8/EP8 Path

Kimi-K2 uses the `pegainfer-kimi-k2` model crate. The kernel-crate surface
is text-only and targets TP8/EP8 with bs > 1 from the start. Shared BF16 ops
reuse existing PegaInfer wrappers. Kimi-specific MoE router and routed INT4
expert entry points live under model-specific ops modules. Kimi router uses the
existing graph-safe GEMM path plus a device-side top8 selector. Routed experts
run on the vLLM Marlin WNA16 backend; the earlier CUTLASS example69
expert-major INT4 path (probe launcher, expert-major route/expand/reduce
kernels, and the `weight_shape` metadata tensor) was retired and removed
(#234). Scale metadata separates checkpoint `[expert,out,group]` and vLLM
Marlin group-major+perm64 `[expert,group,out]`; packed-weight metadata likewise
separates checkpoint offset-binary and Marlin uint4b8 no-actorder. The Marlin
runtime package is fused W13 (`gate_then_up`) plus W2: W13 uses
`[expert,K/16,4096*2]` packed weight and `[expert,K/32,4096]` scale, both in
vLLM layout. Kimi EP dispatch/combine uses the DeepEP shim rather than NCCL
AG/RS.

| op_id | Runtime owner | Rust wrapper | FFI symbols | Source | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- | --- |
| `kimi_k2.norm.rms_batch` | `pegainfer-kimi-k2` | `ops::rms_norm_batch_into` | `rms_norm_batched_cuda` | `csrc/shared/flashinfer_norm.cu` | FlashInfer CUDA | BF16 hidden states, one row per token; Kimi hidden `7168`, q LoRA `1536`, and kv LoRA `512` all use the parameterized wrapper. This is not a fallback path. |
| `kimi_k2.norm.rms_vec` | `pegainfer-kimi-k2` | `ops::rms_norm_into` | `rms_norm_cuda` | `csrc/shared/flashinfer_norm.cu` | FlashInfer CUDA | BF16 single vector path; exposed in `pegainfer-kimi-k2` headers as `RmsNormBackend::FlashInferVec`. |
| `kimi_k2.norm.fused_add_rms` | `pegainfer-kimi-k2` | `ops::fused_add_rms_norm_batch_into` | `fused_add_rms_norm_batched_cuda` | `csrc/shared/flashinfer_norm.cu` | FlashInfer CUDA | Residual add plus RMSNorm over bs > 1 token batches. |
| `kimi_k2.linear.dense_bf16` | `pegainfer-kimi-k2` | `ops::gemm_into` / `ops::gemm_rows_into` | `gemm_cuda` | `csrc/shared/linear.cu` | cuBLAS | BF16 attention, dense MLP, shared expert, router gate, and lm_head shard projections. |
| `kimi_k2.attn.mla_fused_qkv_a` | `pegainfer-kimi-k2` | `ops::gemm_graphsafe_into_checked` | `gemm_graphsafe_cuda` | `csrc/shared/linear.cu` | graph-safe cuBLAS GEMM | Load-time `DeviceMatrix::vstack(q_a_proj, kv_a_proj_with_mqa)` creates weight `[2112,7168]`; decode writes `qkv_a [B,2112]` without D2H or step-time allocation. |
| `kimi_k2.attn.mla_split_qkv_a` | `pegainfer-kimi-k2` | `ops::kimi_mla_split_qkv_a` | `kimi_mla_split_qkv_a_cuda` | `csrc/kimi_k2/kimi_mla.cu` | CUDA | Splits fused `qkv_a [B,2112]` into `q_a [B,1536]`, compressed KV `[B,512]`, and raw `k_rope [B,64]`. This replaces the old separate `kv_a` split path. |
| `kimi_k2.attn.mla_rope_split_decode` | `pegainfer-kimi-k2` | `ops::kimi_mla_rope_split_decode_rt` | `kimi_mla_rope_split_decode_cuda` | `csrc/kimi_k2/kimi_mla.cu` | CUDA | Decode-step split+RoPE prep: `q_proj [B,8,192]` and current `k_rope [B,64]` plus device positions produce `q_nope [B,8,128]`, `q_pe [B,8,64]`, and `append_kpe [B,64]` in Kimi split-half RoPE layout. |
| `kimi_k2.attn.mla_absorb_q` | `pegainfer-kimi-k2` | `ops::kimi_mla_absorb_q_nope_rt` | `kimi_mla_absorb_q_nope_cuda` | `csrc/kimi_k2/kimi_mla.cu` | graph-safe cuBLAS strided-batched GEMM | Uses the `W_UK` slice inside `kv_b_proj [8,256,512]` directly: `q_nope [B,8,128] -> q_abs_nope [B,8,512]`, one cuBLAS batch per local head, no weight repack. |
| `kimi_k2.attn.mla_paged_append` | `pegainfer-kimi-k2` | `ops::kimi_mla_paged_kv_append` | `kimi_mla_paged_kv_append_cuda` | `csrc/kimi_k2/kimi_mla.cu` | FlashInfer MLA page helper | Appends compressed MLA KV step tensors into paged cache: `append_ckv [nnz,512]`, `append_kpe [nnz,64]`, device `batch_indices/positions`, page table CSR, and explicit ckv/kpe strides. Runtime may use separate ckv/kpe buffers or strided views into concat storage. |
| `kimi_k2.attn.mla_decode_paged` | `pegainfer-kimi-k2` | `ops::kimi_flashinfer_batch_decode_mla_rt` | `kimi_flashinfer_batch_decode_mla_cuda` | `csrc/kimi_k2/kimi_mla.cu` | FlashInfer BatchDecode MLA | Consumes absorbed `q_abs_nope [B,8,512]`, `q_pe [B,8,64]`, paged compressed KV, and decode plan arrays; writes latent attention output `[B,8,512]`. `W_UK_T [H,128,512]` absorption and `W_UV [H,512,128]` v-up stay model-side. |
| `kimi_k2.attn.mla_v_up` | `pegainfer-kimi-k2` | `ops::kimi_mla_v_up_rt` | `kimi_mla_v_up_cuda` | `csrc/kimi_k2/kimi_mla.cu` | graph-safe cuBLAS strided-batched GEMM | Uses the `W_UV` slice inside `kv_b_proj [8,256,512]` directly: FlashInfer latent `[B,8,512] -> attn_out [B,8,128]`, one cuBLAS batch per local head, no D2H. |
| `kimi_k2.moe.router_noaux_tc` | `pegainfer-kimi-k2` | `ops::kimi_router_noaux_tc_launch` | `kimi_k2_router_noaux_tc_cuda` | `csrc/kimi_k2/kimi_router.cu` | graph-safe GEMM + CUDA selector | BF16 hidden `[padded_tokens,7168]`, gate `[384,7168]`, correction bias `[384]`, output top8 route weights/indices for active tokens; logits projection uses library GEMM, selection stays device-resident. H20 rank0 gate covers real K2.5 layer1 gate/bias. |
| `kimi_k2.moe.marlin_align_block_size` | `pegainfer-kimi-k2` | `ops::kimi_moe_marlin_align_block_size` | `kimi_moe_marlin_align_block_size_cuda` | `csrc/kimi_k2/kimi_experts.cu` | CUDA routing metadata | Device-resident vLLM Marlin/WNA16 alignment: `sorted_token_ids`, `expert_ids`, and `num_tokens_post_padded` for local EP experts. It ignores non-local experts like vLLM `ignore_invalid_experts=True`, pads each local expert to block size `8/16/32/48/64`, uses sentinel `active_tokens * topk`, and performs no D2H or allocation in the decode step. |
| `kimi_k2.moe.int4_marlin_package` | `pegainfer-kimi-k2` | `ops::kimi_marlin_int4_reorder_weight`, `ops::kimi_marlin_int4_reorder_scale`, `ops::kimi_marlin_int4_fuse_w13` | `kimi_marlin_int4_reorder_weight_cuda`, `kimi_marlin_int4_reorder_scale_cuda`, `kimi_marlin_int4_fuse_w13_cuda` | `csrc/kimi_k2/kimi_marlin_int4.cu` | CUDA load-time package helpers | Weight package preserves vLLM `uint4b8` bias=8 nibbles. Single projections repack checkpoint `[expert,out,K/8] int32` into Marlin no-actorder `[expert,K/16,N*2] int32`; scale package converts checkpoint `[expert,out,K/32]` into vLLM Marlin group-major+perm64 `[expert,K/32,out]`. Final runtime package fuses gate/up into W13 `[expert,K/16,4096*2]` and W13 scale `[expert,K/32,4096]`; W2 remains `[expert,2048/16,7168*2]` and `[expert,2048/32,7168]`. These are load/package helpers, not decode hot-path kernels. |

## GLM5.2 MoE Bring-Up Surface

GLM5.2 uses the `pegainfer-kernels/glm52` feature, which depends on the shared
`moe` substrate for DeepEP, DeepGEMM, and FlashMLA. The current surface is only
the GLM5.2 decode substrate that has a stable caller shape; router/indexer,
pipeline-parallel P2P, TRTLLM fallbacks, and local route/scatter/combine stay out
until the model crate proves their contracts.

| op_id | Runtime owner | Rust wrapper | FFI symbols | Source | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- | --- |
| `glm52.deepgemm.scale_layout` | future GLM5.2 model crate | `ops::glm52_deepgemm_mn_major_tma_aligned_f32_launch` | `glm52_deepgemm_mn_major_tma_aligned_f32_cuda` | `csrc/glm52/glm52_deepgemm_layout.cu` | CUDA layout helper for DeepGEMM | Converts row-major f32 scales `[rows, scale_cols]` into MN-major/TMA-aligned storage with `aligned_rows` padded to 16-byte f32 row alignment. |
| `glm52.flashmla.sparse_decode` | future GLM5.2 model crate | `ops::glm52_flashmla_sparse_decode_num_sm_parts`, `ops::glm52_flashmla_sparse_decode_metadata_launch`, `ops::glm52_flashmla_sparse_decode_launch` | `glm52_flashmla_sparse_decode_num_sm_parts_cuda`, `glm52_flashmla_sparse_decode_metadata_cuda`, `glm52_flashmla_sparse_decode_launch_cuda` | `csrc/glm52/glm52_flashmla_sparse.cu` | FlashMLA SM90 sparse decode | SM90-only sparse decode for V32 layout: `batch<=128`, `heads=64`, `qk_dim=576`, `v_dim=512`, `page_size=64`, packed KV token stride `656`, fixed `topk=2048`, and `num_sm_parts<=160`. Dynamic `topk_length` is intentionally not exposed because this kernel asserts it must be null. |

## Kimi-K3 TileLang Decode Surface

K3 uses the `pegainfer-kernels/k3` feature. The surface is TileLang-generated
CUDA, AOT-compiled at build time from the vendored kernel definitions in
`pegainfer-k3/kernels/tilelang_defs.py` (byte-identical to the certified
upstream spellings — see `pegainfer-k3/kernels/README.md` for the generate /
pre-generated / stub build tiers). It covers every non-GEMM step of a K3
decode iteration; dense projections go to cuBLASLt and the routed experts to
the DeepGEMM masked grouped-GEMM chain below.

Batch buckets: the decode ladder is `1..128` (15 buckets); every family
except the decode-only `kda_core` additionally carries the five prefill
chunk buckets `256/512/1024/2048/4224` (15 total) — the chunked-prefill
step runs the same batched stages at chunk width, whose ceiling is the
MegaMoE protocol maximum.

Every kernel here is **batched and compiled per static shape tuple** — expert
count, attention-residual block count, MLA context capacity and the batch
bucket are all baked in. There is no separate single-row kernel set: `b = 1`
is a bucket whose per-row spelling is the certified single-row kernel, which
is what the upstream bitwise gate proves. Buckets are
`{1,2,4,8,16,32,48,64,96,128}` and `ops::k3_batch_bucket` rounds the live row
count up, so callers must size buffers for the bucket, not for the live rows,
and the discarded tail rows must still point at valid memory.

Each family is one translation unit with one hand-written `extern "C"`
dispatch launcher keyed on the configuration tuple; launchers return a raw
`cudaError_t`, with `cudaErrorInvalidValue` for a configuration that was never
instantiated and `cudaErrorNotSupported` from the stub tier.

Per-slot state tensors (`conv_silu` windows, `kda_core` recurrent state, MLA
`Kc`/`Vc`) are `[b, ...]` contiguous with each row holding exactly the
single-row layout — the caches are slot-indexed, not paged, and there is no
block table.

K3 dimensions the instantiations are derived from: hidden `7168`, KDA
`96x128`, MLA `96` heads with `qk=192`/`v=128`, q LoRA `1536`, kv LoRA `512`,
routed latent `3584`, MoE intermediate `3072` (shared `6144`), dense
intermediate `33792`, top-k `16`, vocab `163840`, split-K `8`.

The f32 side inputs are f32 on purpose: the checkpoint stores the router
correction bias, `conv1d.weight`, `dt_bias`, `A_log` and `o_norm.weight` as
f32, and narrowing any of them to bf16 measurably flips routing decisions.

| op_id | Runtime owner | Rust wrapper | FFI symbol | Source (OUT_DIR `.cu`) | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- | --- |
| `k3.norm.rms_rbs` | `pegainfer-k3` | `ops::k3_rms_norm_rbs_batched_launch` | `k3_rms_norm_rbs_batched` | `k3_rms_norm_rbs_batched.cu` | TileLang AOT CUDA | KimiRMSNorm, round-before-scale: the normalized value lands bf16 *before* multiplying gamma. One row per block; `X`/`O` are `[b, h]` bf16 and gamma `[h]` is shared. h ∈ {7168, 512, 3584} (layer norms, MLA kv latent, routed latent) × 15 buckets; eps compiled in. |
| `k3.linear.land` | `pegainfer-k3` | `ops::k3_land_batched_launch` | `k3_land_batched` | `k3_land_batched.cu` | TileLang AOT CUDA | Merge the column span `[off, off+n)` of each row's `[split_k, nt]` f32 partial and land bf16 once — the landing of every matmul. 15 (nt, n, off) spans × 15 buckets (14 from the sequential engine verbatim, plus `(12288, 12288, 0)` for chunked prefill's conv-input landing — the sequential engine casts conv inputs inside the conv kernel, the chunk lands them before its window build), split_k = 1 only — the single partial a framework GEMM (cuBLASLt, DeepGEMM) produces, where the merge degenerates to the slice and the cast. Masked tail store, so `n` need not divide 256. |
| `k3.linear.land_rms_rbs` | `pegainfer-k3` | `ops::k3_land_rms_norm_rbs_batched_launch` | `k3_land_rms_norm_rbs_batched` | `k3_land_rms_norm_rbs_batched.cu` | TileLang AOT CUDA | `k3_land_batched` fused with the round-before-scale norm. One span — MLA q_a, `[0, 1536)` of the `14400`-wide fused projection — × 15 buckets, split_k = 1. |
| `k3.elementwise.add2` | `pegainfer-k3` | `ops::k3_add2_batched_launch` | `k3_add2_batched` | `k3_add2_batched.cu` | TileLang AOT CUDA | `O = A + Bt` in bf16 addition (residual adds, routed + shared). n = 7168 × 15 buckets. |
| `k3.elementwise.mul_sigmoid` | `pegainfer-k3` | `ops::k3_mul_sigmoid_batched_launch` | `k3_mul_sigmoid_batched` | `k3_mul_sigmoid_batched.cu` | TileLang AOT CUDA | `O = A * bf16(sigmoid(Bt))`, the MLA sigmoid output gate; the sigmoid is taken in f32 and lands bf16 before the product. n = 12288 × 15 buckets. |
| `k3.act.situ` | `pegainfer-k3` | `ops::k3_situ_batched_launch` | `k3_situ_batched` | `k3_situ_batched.cu` | TileLang AOT CUDA | `4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25)` in f32, landing bf16 once; the betas are compiled in. n ∈ {6144 (shared), 33792 (dense)} × 15 buckets; the routed-expert situ is fused into the masked chain and the mega kernel, so no wide routed instantiation exists here. |
| `k3.kda.conv_silu` | `pegainfer-k3` | `ops::k3_conv_silu_batched_launch` | `k3_conv_silu_batched` | `k3_conv_silu_batched.cu` | TileLang AOT CUDA | Causal depthwise convolution over the 4-slot window plus silu, one token per row. Consumes the projection's `[b, split_k, 12288]` f32 partial: its bf16 landing is `X`, the newest window slot; `Sn` is the shifted state the caller carries. Conv weights `[4, 12288]` are **f32** and have no batch axis; the window state is `[b, 3, 12288]`, one independent window per row. split_k = 1 × 15 buckets. |
| `k3.kda.core` | `pegainfer-k3` | `ops::k3_kda_core_batched_launch` | `k3_kda_core_batched` | `k3_kda_core_batched.cu` | TileLang AOT CUDA | One delta-rule step per row, one (row, head) per block, `threads = head_dim`. State `[b, 96, 128, 128]` f32 laid out `[head, v_dim, k_dim]` per row with decay along k, read from `State` and written to `StateN` (must not alias). `Dt`/`Alog`/`Go` f32 weights with no batch axis, `Bt`/`G2` bf16; gate lower bound and eps compiled in. Gate partial uses split-K 1. 10 buckets — the decode ladder only; prefill chunks cross the recurrence through FlashKDA. |
| `k3.kda.o_norm_gate` | `pegainfer-k3` | `ops::k3_o_norm_gate_batched_launch` | `k3_o_norm_gate_batched` | `k3_o_norm_gate_batched.cu` | TileLang AOT CUDA | `kda_core`'s tail on its own: per (row, head) the f32 rms_norm of the bf16 attention landing `X` times the o_norm gamma `Go [128]` f32, landed once, times the bf16 sigmoid of the output gate `G2` — word-for-word the fused core's last loop. Chunked prefill computes attention through FlashKDA and finishes rows here; eps compiled in. 15 buckets. |
| `k3.moe.router_topk` | `pegainfer-k3` | `ops::k3_router_topk_batched_launch` | `k3_router_topk_batched` | `k3_router_topk_batched.cu` | TileLang AOT CUDA | Sigmoid router + biased top-k over already-merged `[b, E]` f32 score rows, one row per block. Serial O(topk*E) scan by thread 0 with lowest-index tie-break; weights gathered from the **un-biased** scores, denominator `+1e-20`, scaled by the bf16 `Rs[0]`. E ∈ {896 (full table), 224 (4-way EP shard)}, TOPK = 16, × 15 buckets. |
| `k3.attnres.scores` | `pegainfer-k3` | `ops::k3_attnres_scores_batched_launch` | `k3_attnres_scores_batched` | `k3_attnres_scores_batched.cu` | TileLang AOT CUDA | Attention-residual candidate scoring, one block per (row, candidate): weightless RMS normalization then a dot with the pre-fused f32 scoring vector `[7168]`. Candidate `NB` is that row's prefix sum, below it its own snapshot history `[b, NB, 7168]`. NB ∈ 1..8 (the history grows one entry per 12 layers over 93 layers) × 15 buckets. |
| `k3.attnres.mix` | `pegainfer-k3` | `ops::k3_attnres_mix_batched_launch` | `k3_attnres_mix_batched` | `k3_attnres_mix_batched.cu` | TileLang AOT CUDA | Softmax over each row's `NB+1` scores, then a probability-weighted mix of the **un-normalized** candidates landing bf16 once. Grid `(b, 7168/256)`; each block redoes its row's softmax. NB ∈ 1..8 × 15 buckets. |

## Kimi-K3 MLA Paged Attention

| op_id | Runtime owner | Rust wrapper | FFI symbol | Source | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- | --- |
| `k3.attn.mla_paged_absorbed` | `pegainfer-k3` | `ops::k3_mla_paged_attn_launch` | `k3_mla_paged_attn_cuda` | `csrc/k3/k3_mla_paged_attn.cu` | CUDA | Absorbed-MLA NoPE decode over the paged **latent** cache, one (row, head) per block. The cache row is `[kv_lora 512 | rope 64]` bf16 in 64-token pages (`[page][layer][token][576]`, per-layer byte offset passed in); the block walks the per-row device block table by logical position, so page permutation is bit-identical and a `-1` page reads as zero latent. Query is absorbed against `w_kv_b`'s W_UK rows and the attended latent expanded with W_UV; the softmax is a 3-sweep recompute (max / sum / probs+attend), so there is no O(ctx) storage and **no compile-time context cap**. `N` is a per-row device i32 length — no host sync. Score landings replay the certified chain (f32 dot over 576 → bf16 → × bf16 scale in bf16 → f32 softmax → bf16 probabilities → f32 latent accumulation → bf16 → f32 W_UV expansion → bf16), documented step-by-step in the source header. Batch is a plain launch dimension (no per-bucket instantiation). |

## Kimi-K3 FlashKDA (chunked prefill)

| op_id | Runtime owner | Rust wrapper | FFI symbol | Source | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- | --- |
| `k3.kda.flash_kda_fwd` | `pegainfer-k3` | `ops::k3_flash_kda_fwd_launch` | `k3_flash_kda_fwd` | `csrc/k3/k3_flash_kda.cu` over `third_party/flash-kda` | CUTLASS/CuTe CUDA (vendored FlashKDA, MIT, MoonshotAI) | Chunkwise KDA forward: one sequence of `T` tokens per call in 16-token intra-chunk tiles (kernel 1: q/k f32 l2norm, gate activation `exp2(lower_bound·log2e·sigmoid(exp(A_log)·(g+dt_bias)))` cumsum, beta sigmoid, UT-transform workspace; kernel 2: inter-chunk state recurrence + output). q/k/v/g/out `[T, 96, 128]` bf16 (`g` pre-activation), beta `[H, T]` bf16 logits (the shim's `k3_flash_kda_beta_transpose` produces the layout), `A_log [96]` / `dt_bias [96,128]` f32, recurrent state `[96, 128, 128]` f32 `[head, v, k]` carried in and out (the engine's slab as-is; in/out must not alias — parity double-buffers per chunk). Workspace from `k3_flash_kda_workspace_bytes`. Compiled for the accelerated SM90+ targets (sm_103a here); NOT_SUPPORTED stub elsewhere. Upstream pins CUTLASS `5c149f5`; built against the flashinfer-vendored CUTLASS. The o_norm × output gate lives outside (`k3.kda.o_norm_gate`). |

## Kimi-K3 FlashMLA (chunked prefill)

| op_id | Runtime owner | Rust wrapper | FFI symbol | Source | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- | --- |
| `k3.mla.prefill_gather` | `pegainfer-k3` | `ops::k3_mla_prefill_gather_launch` | `k3_mla_prefill_gather` | `csrc/k3/k3_flash_mla_prefill.cu` | CUDA | Chunked prefill context gather: walk one block-table row and split `t` cached 576-wide latent rows (`[kv_lora 512 \| rope 64]`) into dense `[t, 512]` latent (the kv_b GEMM input) and `[t, 64]` rope buffers. uint4-vectorized; page/layer offsets in elements. |
| `k3.mla.prefill_expand_k` | `pegainfer-k3` | `ops::k3_mla_prefill_expand_k_launch` | `k3_mla_prefill_expand_k` | `csrc/k3/k3_flash_mla_prefill.cu` | CUDA | Assemble per-head K rows `[t, 96, 192]` from the kv_b expansion `[t, 96, 256]` (`nope \| value` per head — nope half copied) and the shared per-token rope broadcast across heads. V needs no copy: the FMHA reads it as a strided view into the expansion (`offset 128, head stride 256`). |
| `k3.mla.flash_mla_prefill_fwd` | `pegainfer-k3` | `ops::k3_flash_mla_prefill_fwd_launch` | `k3_flash_mla_prefill_fwd` | `csrc/k3/k3_flash_mla_prefill.cu` over `third_party/FlashMLA/csrc/sm100/prefill/dense` | CUTLASS SM100 FMHA (FlashMLA-vendored NVIDIA kernel) | Dense MLA forward, one sequence per call: q `[t_q, 96, 192]` bf16 (per-head `nope \| rope`, never rotated — NoPE), k `[t_kv, 96, 192]`, v the strided `[t_kv, 96, 128]` view, out `[t_q, 96, 128]` bf16; strides in elements. `CausalMask<false>` aligns Q to the *end* of the KV axis, so with `t_kv = context + t_q` chunk token `i` sees exactly `context + i + 1` keys — one call serves `[context \| chunk]`, no LSE merge while the workspace spans the context. Non-persistent causal tile scheduler (`heads % 8 == 0` required). Compiled sm_100f only (`K3_FLASH_MLA_SM100F`), against FlashMLA's own CUTLASS; NOT_SUPPORTED stub elsewhere. |

## Kimi-K3 MoE Bring-Up Surface

Kimi-K3 uses the `pegainfer-kernels/k3` feature. Unlike `glm52`, `k3` does not
depend on the `moe` substrate — there is no DeepEP/NCCL requirement yet — but it
does require the DeepGEMM submodule, whose device headers the FP8xFP4 masked
grouped GEMM is AOT-instantiated from (no JIT, no torch).

Per-expert shapes: W13 is the fused gate|up projection with `n=6144`, `k=3584`;
W2 is the down projection with `n=3584`, `k=3072`. Local expert-group counts are
instantiated for 56 (EP4 dev / EP16 full), 112 (EP8 full), and 224 (single-GPU
bring-up).

The A side is the FP8 e4m3 / per-1x128 UE8M0 activation recipe GLM5.2 uses. The
B side is MXFP4: e2m1 weights packed K-major two-per-byte, with group-32 UE8M0
scale factors. Both scale operands land in the same Blackwell packed-UE8M0 i32
layout `[groups, ceil(k / gran_k / 4), mn]` (MN-major, 4 exponent bytes per i32,
LSB first), but the differing granularity gives them different packed K extents:
`k/512` for the activation, `k/128` for the weights.

| op_id | Runtime owner | Rust wrapper | FFI symbol | Source | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- | --- |
| `k3.deepgemm.masked_grouped_fp8_fp4` | `pegainfer-k3` | `ops::k3_deepgemm_sm100_masked_grouped_fp8_fp4_launch` | `k3_deepgemm_sm100_masked_grouped_fp8_fp4_launch_cuda` | `csrc/k3/k3_deepgemm_fp8_fp4_grouped_sm100.cu` | DeepGEMM SM100 MGroupedMasked (tcgen05) | Activation `[groups, masked_cap, k]` fp8 e4m3, act scale i32 `[groups, k/512, masked_cap]`, weight `[groups, n, k]` fp4 e2m1 (2 per byte), weight scale i32 `[groups, k/128, n]`, out `[groups, masked_cap, n]` bf16. `masked_cap % 128 == 0`; `groups` dispatches over {56, 112, 224}; `num_sms` selects the B200 (148) / GB300 (152) instantiation. Requires sm_100f — NOT_SUPPORTED stub otherwise. |
| `k3.deepgemm.fp4_sf_prepare` | `pegainfer-k3` | `ops::k3_fp4_sf_prepare_launch` | `k3_fp4_sf_prepare_cuda` | `csrc/k3/k3_deepgemm_fp8_fp4_grouped_sm100.cu` | CUDA layout helper for DeepGEMM | Loader-time repack of checkpoint MXFP4 weight scales `[groups, n, k/32]` u8 UE8M0 exponent bytes (K-major) into the MN-major packed SFB tensor `[groups, k/128, n]` i32. Transpose plus 4:1 pack; not a step-time kernel. |

The batched-decode chain around that GEMM lives in `csrc/k3/k3_moe_chain.cu`.
An *entry* is one expanded `(token, topk-slot)` pair at index
`token * topk + slot`; entry order is the chain's deterministic order (it fixes
the masked row each entry lands in and the combine's accumulation order), and an
entry whose `topk_idx` falls outside `[0, groups)` is inactive — that is how
padded batch rows, and later an EP shard's non-local experts, are excluded.
Every consumer re-reads `topk_idx` rather than trusting the slot map alone. The
local gather is fused into the W13 quant, so no bf16 expert-major staging buffer
exists. Nothing allocates, reads back, or varies its launch geometry with device
state, so the chain is CUDA-graph capturable; `tests/k3_moe_chain_gate.rs` is
the end-to-end numerical gate (and asserts two runs are bit-identical).

| op_id | Runtime owner | Rust wrapper | FFI symbol | Source | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- | --- |
| `k3.moe.local_route_metadata` | `pegainfer-k3` | `ops::k3_moe_local_route_metadata_launch` | `k3_moe_local_route_metadata_cuda` | `csrc/k3/k3_moe_chain.cu` | CUDA | `topk_idx [tokens, topk]` i32 -> `masked_m [groups]` i32 and `slot_map [tokens * topk]` i32 (`expert * masked_cap + rank`, `-1` when inactive). One block per expert compacts that expert's entries in entry order with a ballot scan, so the row assignment is scheduling-independent; the `-1` fill is partitioned by `entry % groups`, disjoint from the claimed writes. An expert claiming more than `masked_cap` entries traps. |
| `k3.moe.gather_fp8_quant_masked` | `pegainfer-k3` | `ops::k3_moe_gather_fp8_quant_masked_launch` | `k3_moe_gather_fp8_quant_masked_cuda` | `csrc/k3/k3_moe_chain.cu` | CUDA | Local gather fused with the W13 A-operand quant: bf16 `[tokens, hidden]` read through the routing map -> fp8 e4m3 `[groups * masked_cap, hidden]` plus MN-major UE8M0 f32 scales `[groups, hidden/128, masked_cap]`. Grid `(min(entries, 256), hidden/128)` x 128 threads, grid-strided over entries. `hidden % 128 == 0`. |
| `k3.moe.situ_and_mul_fp8_quant_masked` | `pegainfer-k3` | `ops::k3_situ_and_mul_fp8_quant_masked_launch` | `k3_situ_and_mul_fp8_quant_masked_cuda` | `csrc/k3/k3_moe_chain.cu` | CUDA | K3 situ activation over the masked W13 output `[groups * masked_cap, 2*inter]` bf16 (gate = first `inter` columns) followed by the W2 A-operand quant. In f32 over the bf16 GEMM output: `4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25)`. Router weights are **not** applied here. Out fp8 `[groups * masked_cap, inter]` + f32 scales `[groups, inter/128, masked_cap]`. |
| `k3.moe.fp8_scale_pack_ue8m0` | `pegainfer-k3` | `ops::k3_fp8_scale_pack_ue8m0_launch` | `k3_fp8_scale_pack_ue8m0_cuda` | `csrc/k3/k3_moe_chain.cu` | CUDA | f32 MN-major group scales `[groups, scale_cols, cap]` -> packed UE8M0 i32 SFA `[groups, scale_cols/4, cap]`, LSB-first exponent bytes. Dense full-cover pass (no stale byte can reach the GEMM); inputs must already be powers of two, which the two quant kernels guarantee. `scale_cols = k/128`, multiple of 4. |
| `k3.moe.weighted_combine` | `pegainfer-k3` | `ops::k3_moe_weighted_combine_launch` | `k3_moe_weighted_combine_cuda` | `csrc/k3/k3_moe_chain.cu` | CUDA | Masked W2 output `[groups * masked_cap, hidden]` bf16 x `topk_weight [tokens, topk]` f32 -> `[tokens, hidden]` bf16. Grid `(tokens, ceil(hidden/256))`; each thread owns one column and accumulates its token's topk slots **in slot order** in f32 with no atomics, rounding to bf16 once. Tokens with no active entry land an exact zero, so the output is fully covered. |

## Kimi-K3 MegaMoE Surface

The production MoE transport: one fused DeepGEMM "mega" kernel per MoE layer
covers dispatch, the two grouped GEMMs, situ, and combine across all EP ranks
via NVLink peer access, synchronizing with device-side barriers (no host
collectives). `ranks ∈ {1, 4}` × 3 block configs × 2 activations are
AOT-instantiated; the serving path pins the block config to the protocol
maximum on every rank and step so peer traffic shapes are invariant.
`tests/k3_mega_parity_gate.rs` holds the kernel at bit parity with the
reference implementation.

| op_id | Runtime owner | Rust wrapper | FFI symbol | Source | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- | --- |
| `k3.moe.mega` | `pegainfer-k3` | `ops::k3_mega_moe_launch` | `k3_mega_moe_launch_cuda` | `csrc/k3/k3_mega_moe_sm100.cu` | DeepGEMM SM100 fused MoE (tcgen05) | End-to-end routed-expert layer over the rank's symmetric buffer: fp8 e4m3 x (gran-8 interleaved) fp4 W13, situ, fp8 x fp4 W2, routing weights applied **before** W2, cross-rank dispatch/combine in-kernel. Requires every rank's buffer pointer table and identical block config fleet-wide; 152-SM instantiation. |
| `k3.moe.mega_quant_x` | `pegainfer-k3` | `ops::k3_mega_write_inputs_launch` | `k3_mega_quant_x_cuda`, `k3_mega_write_routing_cuda` | `csrc/k3/k3_mega_moe_sm100.cu` | CUDA | Step-time input staging into the symm buffer: activation quant to fp8 e4m3 at gran-32 with packed ue8m0 scales, plus the routing table write (`topk_idx`/weights) the mega kernel reads. |
| `k3.moe.mega_prepare` | `pegainfer-k3` | `ops::k3_mega_prepare_l1_weights_launch`, `ops::k3_mega_prepare_sf_launch` | `k3_mega_prepare_sf_cuda` | `csrc/k3/k3_mega_moe_sm100.cu` | CUDA layout helpers | Load-time weight transforms: gran-8 gate/up interleave of W13 and the UTCCP 4×32 scale-factor transpose. Not step-time kernels. |
| `k3.moe.mega_symm` | `pegainfer-k3` | `ops::k3_mega_symm_buffer_layout`, `ops::k3_mega_open_peer_access`, `ops::k3_mega_token_alignment`, `ops::k3_mega_max_tokens_per_rank` | `k3_mega_symm_buffer_layout_cuda`, `k3_mega_open_peer_access`, `k3_mega_token_alignment`, `k3_mega_max_tokens_per_rank` | `csrc/k3/k3_mega_moe_sm100.cu` | Host helpers | Symmetric-buffer sizing/offsets, peer-access grants (both `cudaDeviceEnablePeerAccess` **and** `cudaMemPoolSetAccess` — pool grant must precede slab allocation for `cudaMallocAsync` memory), the 384-token alignment constant, and the 4224-row protocol maximum the slab and launch are built for. |

## Non-Qwen3 Compatibility

The crate still builds CUDA/Triton symbols needed by the current root binary:

- Qwen3.5 HD256 full-attention kernels: `csrc/qwen35/prefill_attention_hd256.cu`, `csrc/shared/paged_attention.cu`.
- Qwen3.5 linear-attention decode kernels: `csrc/qwen35/conv1d.cu`, `csrc/qwen35/gated_delta_rule.cu`.
- Qwen3.5 chunk-wise GDR prefill Triton AOT kernels: `tools/triton/gated_delta_rule_chunkwise_kernels.py`.

These are preserved for build compatibility. They are not part of the Qwen3-4B Phase 1 API surface.

## Editing Rule

When adding or replacing a kernel used by Qwen3-4B, Kimi-K2, GLM5.2, or K3,
update this routing table.

Do not add model-specific machine-readable manifests here. The kernels crate
owns reusable operator implementations; model crates should own model DAG
metadata. If a Qwen3-4B manifest becomes useful for tracing or simulation, put
it beside the Qwen3-4B model crate and generate or validate it from code.
