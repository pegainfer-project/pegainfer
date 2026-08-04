# Qwen3.5 Tensor Parallelism Design

> **TL;DR:** Qwen3.5 tensor parallelism should reuse Qwen3's controller/worker TP runtime and stay degree-parametric. Phase 1 is correctness-first eager dense TP: validate `TP=2` first, fail closed on indivisible degrees and `TP > 1` CUDA Graph, shard dense full-attention/MLP, and keep linear-attention/GDR state replicated per rank before tackling sharded GDR state.
>
> **Last touched:** 2026-06

## Goal

Add tensor-parallel support for `Qwen3.5-4B` by reusing the Qwen3 TP runtime instead of designing a second parallel execution stack.

The implementation should be degree-parametric where the model dimensions divide cleanly. `TP=2` is the first validation target, not an architectural limit. Unsupported or indivisible degrees must fail closed before model load.

## Qwen3 Runtime Reuse

Reuse the Qwen3 TP shape:

- controller/worker broadcast execution model
- `RequestId` request identity
- coarse-grained prefill/decode/unified/drop step protocol
- rank-local worker-owned model state
- rank-local CUDA context, cuBLAS, graph, and NCCL resources
- hidden all-reduce after row-parallel projections
- replicated embedding/lm_head as the first-pass simplification

Qwen3.5-specific design work should stay focused on model geometry and state ownership: hybrid layer layout, gated q projection, linear-attention conv state, and GDR recurrent state.

## Boundaries

This design does not cover multi-node TP, data parallelism, pipeline parallelism, vocab-parallel embedding/lm_head, or Qwen3.5 prefix-cache/recurrent-state snapshots.

Phase 1 does not shard linear attention or change GDR kernel shapes. Phase 2 does not change GDR math, does not all-reduce recurrent state, and does not move recurrent state ownership back into the scheduler.

## Settled Phase 1 Contract

These decisions are settled before implementation starts.

- `TP=1` must preserve the current single-GPU behavior.
- `TP=2` is the first correctness target. The implementation may stay degree-parametric, but unsupported or indivisible degrees must fail before model load.
- `TP > 1` is eager-only in Phase 1. CUDA Graph under TP must fail closed instead of silently falling back or partially capturing.
- Reuse the Qwen3 controller/worker broadcast execution model and avoid a second long-lived Qwen3.5-specific TP runtime shape.
- Shard dense full-attention and MLP operators.
- Replicate embedding and tied `lm_head`.
- Replicate linear-attention/GDR weights in Phase 1.
- Each rank worker owns and mutates its own full linear-attention conv state and GDR recurrent state copy.
- The scheduler owns logical request lifecycle and logical KV/page lifecycle only.
- Full-attention KV is physically rank-local and sharded by local KV heads, but one logical request/page assignment is mirrored across all ranks.
- `DropRequest`, finish cleanup, cancellation cleanup, and slot reuse must release or reset the corresponding rank-local KV/recurrent/conv state on every rank.
- Qwen3.5 gated `q_proj` slicing is an explicit acceptance gate: every rank must receive both q rows and gate rows for its local query heads.
- MLP gate/up row sharding and down column sharding require explicit reconstruction or layout tests.

## Still Open / Future Discussion

These topics should not block Phase 1 eager dense TP, but they remain design work before any later implementation.

- TP CUDA Graph support: graph state ownership per rank, synchronized capture/replay order, NCCL capture behavior, graph padding slots, and recurrent/conv D2D slot compaction under capture.
- Sharded linear-attention/GDR execution: local GDR AOT kernel shapes, local recurrent-state layout, local conv state layout, and Phase 2 weight slicing.
- TP-aware prefix cache or recurrent-state snapshots.
- Vocab-parallel embedding or `lm_head`.
- Multi-node TP, data parallelism, and pipeline parallelism.
- Performance optimization claims. Phase 1 is a correctness/runtime milestone, not a throughput milestone.

## Why Dense First, GDR Second

Qwen3.5 has two separable TP problems.

The dense part is already proven by Qwen3: full-attention head sharding, local KV heads, MLP intermediate sharding, all-reduce after row-parallel projections, and worker-thread CUDA/NCCL execution.

The linear-attention part is Qwen3.5-specific: conv state and GDR recurrent state are long-lived request state, current GDR AOT kernels are built for the global value-head shape, and slot compaction / graph padding / `DropRequest` must all preserve rank-local recurrent state. If dense TP and GDR TP land together, failures are hard to attribute. Phase 1 narrows correctness debugging to runtime + dense sharding; Phase 2 then isolates the GDR/recurrent contract.

## Architecture Summary

Qwen3.5-4B:

- 32 layers: 24 linear attention + 8 full attention
- full-attention layers: `3, 7, 11, 15, 19, 23, 27, 31`
- `hidden_size = 2560`
- `intermediate_size = 9216`
- tied embedding/lm_head
- `vocab_size = 248320`

Full attention:

- `num_attention_heads = 16`
- `num_key_value_heads = 4`
- `head_dim = 256`
- `q_dim = num_attention_heads * head_dim = 4096`
- `kv_dim = num_key_value_heads * head_dim = 1024`
- q projection includes an output gate, so gated q projection output dim is `2 * q_dim = 8192`

Linear attention:

- `linear_num_key_heads = 16`
- `linear_key_head_dim = 128`
- `linear_num_value_heads = 32`
- `linear_value_head_dim = 128`
- `linear_q_dim = linear_num_key_heads * linear_key_head_dim = 2048`
- `linear_k_dim = linear_q_dim`
- `linear_v_dim = linear_num_value_heads * linear_value_head_dim = 4096`
- `linear_qkv_dim = linear_q_dim + linear_k_dim + linear_v_dim = 8192`
- `linear_z_dim = linear_v_dim = 4096`
- recurrent state per linear layer: `[linear_num_value_heads, linear_key_head_dim, linear_value_head_dim] f32`
- conv state per linear layer: `linear_qkv_dim * (conv_kernel_dim - 1)` bf16

## Partition Contract

For any candidate `tp`, require:

- `num_attention_heads % tp == 0`
- `num_key_value_heads % tp == 0`
- `intermediate_size % tp == 0`
- Phase 2 additionally requires `linear_num_key_heads % tp == 0` and `linear_num_value_heads % tp == 0`

Full attention local dimensions:

- `local_q_heads = num_attention_heads / tp`
- `local_kv_heads = num_key_value_heads / tp`
- `local_q_dim = local_q_heads * head_dim`
- `local_kv_dim = local_kv_heads * head_dim`
- `local_gated_q_dim = 2 * local_q_dim`

Qwen3.5 full-attention `q_proj` must be sharded by head-local q/gate pairs. Each rank owns a contiguous query-head range, and for each owned head it must receive both that head's q rows and that head's gate rows. Do not reuse a naive contiguous row shard if the physical layout can split q rows from their gate rows.

MLP local dimensions:

- `local_intermediate = intermediate_size / tp`
- local fused `gate_up_proj` rows: `2 * local_intermediate`
- local `down_proj` input cols: `local_intermediate`

Linear-attention local dimensions for Phase 2:

- `local_linear_key_heads = linear_num_key_heads / tp`
- `local_linear_value_heads = linear_num_value_heads / tp`
- `local_linear_q_dim = local_linear_key_heads * linear_key_head_dim`
- `local_linear_k_dim = local_linear_q_dim`
- `local_linear_v_dim = local_linear_value_heads * linear_value_head_dim`
- `local_linear_qkv_dim = local_linear_q_dim + local_linear_k_dim + local_linear_v_dim`
- `local_linear_z_dim = local_linear_v_dim`
- local recurrent state: `[local_linear_value_heads, linear_key_head_dim, linear_value_head_dim] f32`
- local conv state: `local_linear_qkv_dim * (conv_kernel_dim - 1)` bf16

## Phase 1: Dense TP, Replicated Linear Attention

Shard:

- full-attention `q_proj`, `k_proj`, `v_proj`, `o_proj`
- full-attention KV cache over local KV heads
- MLP `gate_proj`, `up_proj`, `down_proj`

Replicate:

- embedding and tied lm_head
- all linear-attention weights
- all linear-attention conv state
- all GDR recurrent state
- existing GDR kernels and scratch shapes

Execution:

- full-attention: local q/k/v + local attention + local `o_proj`, then all-reduce hidden
- MLP: local gate/up + local activation + local `down_proj`, then all-reduce hidden
- linear attention: every rank runs the full layer and updates a full local recurrent-state copy; do not all-reduce replicated linear-attention output

State ownership:

- scheduler owns request admission, request identity, logical page allocation, streaming handles, sampling params, generation counters, and finish bookkeeping
- rank workers own rank-local model shards, rank-local physical KV buffers, rank-local decode buffers, and rank-local recurrent/conv state
- rank 0 is not special for state mutation; it follows the same worker command protocol as other ranks
- non-primary workers may return acknowledgement or step failure only, while the primary worker returns artifacts for scheduler-side result resolution
- all workers must observe the same ordered `RunPrefillStep`, `RunDecodeStep`, `RunUnifiedStep`, `DropRequest`, and `Shutdown` commands

CUDA Graph:

- Phase 1 TP execution is eager-only
- `tp_size > 1` with CUDA Graph enabled must return an explicit startup/configuration error before serving requests
- TP graph capture is a follow-up because Qwen3.5 graph state includes recurrent slots, slot compaction, padding slots, and NCCL ordering questions

Validation scope:

- first validated degree: `TP=2`
- Qwen3.5 HF logits gate
- Qwen3.5 scheduler e2e
- long prompt / chunked prefill path
- slot-compaction replay
- finish/drop followed by slot reuse without stale recurrent or conv state
- gated `q_proj` head-local q/gate slicing test
- MLP gate/up shard and down shard reconstruction/layout test
- basic TP2 serving smoke
- startup fails closed for unsupported or indivisible degrees
- startup fails closed for `tp_size > 1` with CUDA Graph enabled

## Phase 2: Sharded Linear Attention / GDR

Phase 2 converts linear attention from replicated execution to true TP execution.

Shard:

- `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`
- `dt_bias`, `A_log`
- conv state
- GDR recurrent state
- linear-attention `out_proj`

Execution:

- each rank computes local q/k/v/z/b/a
- each rank updates only local conv state and local GDR recurrent state
- each rank runs local gated RMSNorm/output-gate work
- each rank runs local `out_proj`
- all-reduce happens after `out_proj`

Never all-reduce GDR recurrent state or conv state. Their ownership is rank-local and request-local.

### vLLM Reference

Use vLLM's `Qwen3NextForCausalLM` / `QwenGatedDeltaNetAttention` as the reference contract, not as code to copy mechanically:

- GDN state shape depends on `tp_size`
- q/k/v/z projections are tensor-parallel column projections
- `out_proj` is row-parallel and reduces back to full hidden
- `dt_bias` and `A_log` are sharded over local value heads
- b/a projections are local-value-head aware; some quantized paths may replicate small projections and slice locally
- GDR prefill/decode kernels consume local head/state shapes

PegaInfer-specific work remains: worker-owned rank-local recurrent state, `RequestId` lifecycle, local-state slot compaction, `DropRequest` cleanup, and fail-closed kernel-shape validation.

Validation scope:

- Phase 1 gates still pass
- long HF logits replay under the validated degree
- slot compaction replay
- recurrent-state cleanup on finish/drop
- no stale local recurrent state after slot reuse

## References

- `docs/models/qwen3/tp-design.md`
- `pegainfer-qwen3/src/config.rs`
- `pegainfer-qwen3/src/executor.rs`
- `pegainfer-qwen35/src/config.rs`
- `pegainfer-qwen35/src/weights.rs`
- `pegainfer-qwen35/src/recurrent_state.rs`
- `pegainfer-qwen35/src/batch_decode.rs`
- vLLM `Qwen3NextForCausalLM`
- vLLM `QwenGatedDeltaNetAttention`
